"""Fast GPU session: a rented RunPod card running ComfyUI with Qwen-Image 2.1 and Wan 2.2. While it is ready,
Qwen-Image pictures and edits and Wan 2.2 videos run there instead of on this PC's 8 GB card.

The pod is reached through RunPod's HTTPS gateway (https://<pod>-8189.proxy.runpod.net), not SSH, so machines
without a public address (most cheap community ones) work too. On the pod, cloud_agent.py answers only to the key
made for this session; it runs the setup, takes the trained characters and passes requests on to ComfyUI. On this
PC a small relay on 127.0.0.1:<local_port> adds the key and forwards, so the rest of the app talks to it like any
ComfyUI.

The person picks the card from options() (live RunPod stock and prices); nothing is chosen automatically. Jobs run
one at a time, as on the PC. The pod is set up from scratch each time (no paid storage between sessions) by
cloud_setup.sh, and it is deleted when idle for idle_minutes, after max_hours, or from the Stop button. A pod left
from before an app restart is picked up again, or deleted if it can't be reached, so a forgotten pod never keeps
billing."""
import base64, json, os, re, secrets, subprocess, threading, time, urllib.error, urllib.parse, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

APP_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(APP_DIR, "cloud_gpu.json")  # the running pod and its key, so a restart can find it
SETUP = os.path.join(APP_DIR, "cloud_setup.sh")
AGENT = os.path.join(APP_DIR, "cloud_agent.py")
DOWNLOADER = os.path.join(APP_DIR, "cloud_dl.py")  # sent inside the setup script
RUNPODCTL = os.path.expanduser(r"~\.local\bin\runpodctl.exe")
GATE_PORT = 8189
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
DIRECT = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # the VPN's system proxy breaks localhost
# The pod's start command: unpack the agent from RM_AGENT and run it (the image's own start script, sshd and
# Jupyter, isn't needed).
START_CMD = "bash -c 'echo \"$RM_AGENT\" | base64 -d > /root/agent.py && python3 /root/agent.py > /root/agent.log 2>&1'"
# full-quality Qwen-Image 2.1, used on the rented card once its background download is done (state["extras"])
HQ_QWEN = {"unet": "qwen_image_2.1_bf16.safetensors", "text_encoder": "qwen3vl_8b_bf16.safetensors"}
# The video engine a card gets: MiniMax H3 (the best open one, 80 GB+) or LTX-2.5. H3's licence excludes the US, the
# EU, the UK and South Korea, so it is only rented in other data centres (else the card gets LTX-2.5).
H3_EXCLUDED = ("US", "EU", "UK", "GB", "KR")  # data centre id prefixes / country codes
H3_BF16 = "minimax_h3_fl2va_bf16.safetensors"                 # 62 GB full precision: 141 GB+ cards
H3_FULL = "minimax_h3_fl2va_int8_convrot.safetensors"           # 34 GB: 80 GB+ cards
H3_PRUNED = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"  # 21 GB slimmed H3: fits a 32 GB RTX 5090
# H3's reference-to-video model (Video only with a character picture): FastH3 Omni (FastVideo + Nuva Lab, 20 Oct
# 2026), H3 distilled to 8 sampling steps with the same encoder and VAEs, so a clip takes well under half the time
# of the 20-step base model. One pruned export: int8 (22 GB, 32 GB+ cards) or bf16 (44 GB).
OMNI_INT8 = "fasth3_omniref_8step_pruned_int8_convrot.safetensors"
OMNI_BF16 = "fasth3_omniref_8step_pruned_bf16.safetensors"
H3_REF = {H3_BF16: OMNI_BF16, H3_FULL: OMNI_INT8, H3_PRUNED: OMNI_INT8}
FLUX2_MIN_MEM, FLUX2_DISK = 48, 60
# downloads that arrive after the card is ready, each switched on by /root/done_<name> on the pod
EXTRAS = ("video", "qwen_hq", "ideogram", "flux2", "krea2", "krea2ref", "krea2_edit", "moss")
# file sizes in GB (Comfy-Org / comfyicu repos, 23 Oct 2026), for the container disk
GB = {"krea2_turbo_bf16.safetensors": 24.48, "krea2_turbo_fp8_scaled.safetensors": 12.24,
      "qwen3vl_4b_bf16.safetensors": 8.27, "moss_image_sft_bf16.safetensors": 9.47, "qwen3.5_9b_bf16.safetensors": 17.98,
      "ltx-2.5-22b-distilled-transformer-bf16.safetensors": 39.13,
      "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors": 20.03,
      "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors": 24.46,
      "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors": 14.32,
      H3_BF16: 61.73, H3_FULL: 31.7, H3_PRUNED: 19.53, OMNI_INT8: 20.69, OMNI_BF16: 41.2,
      "qwen3vl_32b_minimax_h3_bf16.safetensors": 47.97,
      "qwen3vl_32b_minimax_h3_int8_convrot.safetensors": 25.28, "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors": 14.61}
# always on the pod: Qwen-Image 2.1 int8 + bf16 + VAE, upscaler, Ideogram 4 (both models, encoder, VAE), Krea and
# MOSS VAEs, the Krea identity / edit LoRAs, plus ComfyUI, packages and room to unpack
BASE_GB = 16.1 + 29.58 + 0.07 + 28.05 + 0.47 + 1.8 + 40


def pod_files(card):
    """The highest-quality file of each model this card's memory holds (user 23 Oct 2026: "the max version for all
    models on the rented pod"): bf16 when the file takes at most ~80 % of the card (pictures) or ~70 % (video, whose
    frames need room), else the int8 / fp8 file used before. Qwen-Image 2.1 is bf16 on every card already
    (HQ_QWEN) and Ideogram 4 has no bf16 release. Values are file names in the Comfy-Org repos."""
    m = card["mem"]
    return {
        "KREA_UNET": "krea2_turbo_bf16.safetensors" if 24.48 <= 0.8 * m else "krea2_turbo_fp8_scaled.safetensors",
        "KREA_TE": "qwen3vl_4b_bf16.safetensors",
        "MOSS_UNET": "moss_image_sft_bf16.safetensors",
        "MOSS_TE": "qwen3.5_9b_bf16.safetensors",
        "LTX_UNET": "ltx-2.5-22b-distilled-transformer-bf16.safetensors" if 39.13 <= 0.7 * m else
                    "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors",
        "LTX_TE": "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors" if 24.46 <= 0.8 * m else
                  "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",
    }


# what cloud_setup.sh downloads when it gets no file plan (its ${VAR:-default}s): a pod started by an older app
POD_DEFAULT_FILES = {"KREA_UNET": "krea2_turbo_fp8_scaled.safetensors", "KREA_TE": "qwen3vl_4b_fp8_scaled.safetensors",
                     "MOSS_UNET": "moss_image_sft_bf16.safetensors", "MOSS_TE": "qwen3.5_9b_bf16.safetensors",
                     "LTX_UNET": "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors",
                     "LTX_TE": "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors"}


def files_on_pod():
    """The rented card's model file names: its saved plan, else the setup script's defaults."""
    return state.get("files") or POD_DEFAULT_FILES


def disk_gb(card, video, h3_unet=""):
    """Container disk for this card's files (10 % spare), FLUX.2 [dev] on 48 GB+ cards."""
    f = pod_files(card)
    total = BASE_GB + sum(GB[f[k]] for k in ("KREA_UNET", "KREA_TE", "MOSS_UNET", "MOSS_TE"))
    if video == "h3":
        total += GB[h3_unet] + GB[H3_REF[h3_unet]] + GB[comfy_te(card)] + 5.41  # main + reference model, encoder, VAEs
    else:
        total += GB[f["LTX_UNET"]] + GB[f["LTX_TE"]] + 1.71
    if card["mem"] >= FLUX2_MIN_MEM:
        total += FLUX2_DISK
    return int(total * 1.1) + 1


def no_extras():
    return {k: False for k in EXTRAS}


def h3_plan(card):
    """(H3 model file, short edge in px) for this card, or None when it gets LTX-2.5. A 5090 was shown running H3
    (864x480, 15 s in 8 min); the slimmed model keeps it on the card, at 480 px there and H3's native 768 on 48 GB.
    Full precision only where it leaves at least the room for frames the int8 model has on an 80 GB card."""
    if card["mem"] - GB[H3_BF16] >= 80 - GB[H3_FULL]:
        return H3_BF16, 768
    if card["mem"] >= 80:
        return H3_FULL, 768
    if card["mem"] >= 48:
        return H3_PRUNED, 768
    if card["mem"] >= 32:
        return H3_PRUNED, 480
    return None


def h3_max_seconds(mem):
    """The longest H3 clip a card can hold, per resolution. Measured on an RTX 5090 (slimmed H3): 768x1344 for 5 s
    used 31 of 31.4 GB, so its 720p stays at 5 s; the rest scales with the memory left beside the model."""
    if mem >= 80:
        return {"480p": 15, "720p": 15}
    if mem >= 48:
        return {"480p": 15, "720p": 10}
    return {"480p": 10, "720p": 5}


def video_label(video, unet=""):
    return "LTX-2.5" if video != "h3" else "MiniMax H3" + (" lite" if unet == H3_PRUNED else "")
CHUNK = 8 << 20  # trained characters go up in 8 MB pieces (one request through the gateway can't be huge)

# The cards offered (the RTX 4090 and stronger; the list is shown cheapest first). speed = rough speed for
# Qwen-Image / Wan against an RTX 4090 (estimates from the cards' compute and memory speed, not measured here).
CATALOG = [
    {"id": "NVIDIA GeForce RTX 4090", "label": "RTX 4090", "mem": 24, "speed": 1.0},
    {"id": "NVIDIA GeForce RTX 5090", "label": "RTX 5090", "mem": 32, "speed": 1.3, "blackwell": True},
    {"id": "NVIDIA L40S", "label": "L40S", "mem": 48, "speed": 1.0},
    {"id": "NVIDIA RTX 6000 Ada Generation", "label": "RTX 6000 Ada", "mem": 48, "speed": 1.0},
    {"id": "NVIDIA A100 80GB PCIe", "label": "A100 80 GB", "mem": 80, "speed": 1.0},
    {"id": "NVIDIA RTX PRO 6000 Blackwell Server Edition", "label": "RTX PRO 6000", "mem": 96, "speed": 1.4,
     "blackwell": True},
    {"id": "NVIDIA H100 80GB HBM3", "label": "H100", "mem": 80, "speed": 2.0},
    {"id": "NVIDIA H200", "label": "H200", "mem": 141, "speed": 2.2},
    {"id": "NVIDIA B200", "label": "B200", "mem": 180, "speed": 3.0, "blackwell": True},
]

lock = threading.RLock()
_lifecycle = threading.RLock()  # startup and deletion must finish before another lifecycle worker runs
_generation = [0]  # queued workers from an earlier rental must never alter a new one
state = {"status": "off", "message": "", "pod": None, "key": "", "price": 0.0, "gpu": "",
         "started": 0.0, "last_used": 0.0, "loras": [], "video": None, "h3_te": "", "h3_unet": "", "h3_short": 768,
         "where": "", "files": {},
         "extras": no_extras()}
active = [0]       # cloud jobs running right now (never idle-stopped meanwhile)
relay = [None]     # the local relay server
_opts = [0.0, []]  # options() cache: when, list
_broken = {}       # card id -> when a community machine for it turned out faulty (RunPod hands the same one out again)


def cfg():
    with open(os.path.join(APP_DIR, "config.json"), encoding="utf-8") as f:
        c = json.load(f).get("cloud_gpu", {})
    return {"image": "runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404", "local_port": 8190,
            "idle_minutes": 15, "max_hours": 4, **c}


def _set(**kw):
    with lock:
        state.update(kw)
        keep = {k: state[k] for k in ("pod", "key", "price", "gpu", "started", "video", "h3_te", "h3_unet", "h3_short",
                                      "where", "files")}
        if state["pod"]:
            tmp = STATE_FILE + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(keep, f)
            for attempt in range(40):
                try:
                    os.replace(tmp, STATE_FILE)
                    break
                except PermissionError:
                    if attempt == 39:
                        raise
                    time.sleep(0.05)
        elif os.path.exists(STATE_FILE):
            os.remove(STATE_FILE)


def info():
    """For the page: status, what it costs so far, when it stops by itself (never the key)."""
    c = cfg()
    with lock:
        s = {k: v for k, v in state.items() if k != "key"}
    up = time.time() - s["started"] if s["started"] else 0
    idle_left = c["idle_minutes"] * 60 - (time.time() - s["last_used"]) if s["status"] == "ready" and not active[0] else None
    return {**s, "label": s["gpu"] or "GPU", "minutes": round(up / 60), "cost": round(up / 3600 * s["price"], 2),
            "video_label": video_label(s["video"], s["h3_unet"]) if s["video"] else "",
            **_card_facts(s),
            "idle_stop_in": max(0, round(idle_left / 60)) if idle_left is not None else None,
            "max_hours": c["max_hours"], "idle_minutes": c["idle_minutes"], "busy": active[0], "balance": balance()}


def card_video(card):
    """What the card list says this card makes videos with, right now: H3 only if one is free where it may run."""
    plan = h3_plan(card)
    if not plan:
        return video_label("ltx25")
    if h3_stock().get(card["id"]):
        return video_label("h3", plan[0])
    return "LTX-2.5 (none free right now where MiniMax H3 is allowed)"


def _card_facts(s):
    """Memory, speed and the H3 length limits of the rented card, for the page's slider and time estimate."""
    card = next((c for c in CATALOG if c["label"] == s.get("gpu")), None)
    if not card:
        return {}
    return {"mem": card["mem"], "speed": card["speed"], "flux2": card["mem"] >= FLUX2_MIN_MEM,
            "h3_max": h3_max_seconds(card["mem"]) if s.get("video") == "h3" else None}


def options():
    """The cards in CATALOG with RunPod's live stock and both prices (community = cheaper hosts, secure = RunPod's
    own data centres). Asked at most once a minute."""
    if time.time() - _opts[0] < 60 and _opts[1]:
        return _opts[1]
    code, out, err = _run([RUNPODCTL, "gpu", "list"], timeout=60)
    live = {g["gpuId"]: g for g in json.loads(out[out.index("["):])} if code == 0 and "[" in out else {}
    res = []
    for c in CATALOG:
        g = live.get(c["id"], {})
        stock = g.get("stockStatus") or "none"
        res.append({**c, "stock": stock, "available": bool(g.get("available")) and stock.lower() != "none",
                    "community": g.get("communityPricePerHr") if g.get("communityCloud") else None,
                    "secure": g.get("securePricePerHr") if g.get("secureCloud") else None,
                    "community_broken": time.time() - _broken.get(c["id"], 0) < 3600,
                    "video": card_video(c)})
    res.sort(key=lambda o: min([p for p in (o["community"], o["secure"]) if p is not None] or [1e9]))  # cheapest first
    _opts[:] = [time.time(), res]
    return res


def url():
    """ComfyUI on the rented card (through the local relay), or None while there's no ready session."""
    return f"http://127.0.0.1:{cfg()['local_port']}" if state["status"] == "ready" else None


def has_lora(file):
    return file in state["loras"]


class using:
    """with cloudgpu.using(): ... marks a cloud job so the idle timer doesn't stop the pod under it."""
    def __enter__(self):
        active[0] += 1
        _set(last_used=time.time())

    def __exit__(self, *a):
        active[0] -= 1
        _set(last_used=time.time())


def _run(args, timeout=120, data=None):
    r = subprocess.run(args, input=data, capture_output=True, timeout=timeout, creationflags=NO_WINDOW)
    return r.returncode, (r.stdout or b"").decode("utf-8", "replace"), (r.stderr or b"").decode("utf-8", "replace")


_balance = [0.0, None, False]  # when last read, dollars (None = not known yet), a read is running


def balance():
    """RunPod's account balance in dollars, for the Fast GPU button (user 23 Oct 2026). Read with `runpodctl user`
    in the background at most once a minute, so the page's status poll never waits on RunPod; only the balance is
    kept (the reply also has the account's email). None until the first read."""
    with lock:
        if time.time() - _balance[0] > 60 and not _balance[2]:
            _balance[2] = True
            threading.Thread(target=_read_balance, daemon=True).start()
        return _balance[1]


def _read_balance():
    try:
        code, out, _ = _run([RUNPODCTL, "user", "-o", "json"], timeout=30)
        if code == 0:
            _balance[1] = round(float(json.loads(out)["clientBalance"]), 2)
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError):
        pass  # keep the last known value
    finally:
        with lock:
            _balance[0], _balance[2] = time.time(), False


def _call(method, path, data=None, timeout=60, headers=None):
    """A request to the pod's agent through the gateway: (status, body). Gateway errors (502 while the pod boots,
    404 before the port is routed) come back as their status; no connection at all raises."""
    req = urllib.request.Request(f"https://{state['pod']}-{GATE_PORT}.proxy.runpod.net{path}", data=data, method=method,
                                 headers={"X-RM-Key": state["key"], "User-Agent": "Reelmint", **(headers or {})})
    try:
        with DIRECT.open(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def h3_allowed(place):
    """True for a data centre id (CA-MTL-1, EUR-IS-3...) or country code outside H3's excluded territories."""
    head = str(place or "").upper().split("-")[0]
    return bool(head) and head not in H3_EXCLUDED


_h3_stock = [0.0, {}]  # when, {gpu id: [data centres where H3 may run that have the card in stock]}


def h3_stock():
    """Per card, RunPod's data centres outside H3's excluded territories (Canada, India, Japan, Singapore, Australia,
    Iceland, Norway...) that have it in stock now. Renting must name only these: asking for all 17 allowed ones
    failed ("no longer any instances") while two of them had free RTX 5090s (6 Oct). Asked at most once a minute."""
    if time.time() - _h3_stock[0] < 60:
        return _h3_stock[1]
    code, out, err = _run([RUNPODCTL, "datacenter", "list"], timeout=60)
    stock = {}
    try:
        for d in json.loads(out[out.index("["):]):
            if h3_allowed(d.get("id")):
                for g in d.get("gpuAvailability") or []:
                    if g.get("gpuId") and str(g.get("stockStatus") or "").lower() not in ("", "none"):
                        stock.setdefault(g["gpuId"], []).append(d["id"])
    except (ValueError, KeyError):
        return _h3_stock[1]
    _h3_stock[:] = [time.time(), stock]
    return stock


def _place(pod_id):
    """Where the rented pod is: its data centre id, or the machine's country code ("" if RunPod doesn't say)."""
    code, out, err = _run([RUNPODCTL, "pod", "get", pod_id, "--include-machine"], timeout=60)
    try:
        p = json.loads(out[out.index("{"):])
    except ValueError:
        return ""
    m = p.get("machine") or {}
    return str(p.get("dataCenterId") or m.get("dataCenterId") or m.get("location") or "")


def _create(c, gpu, cloud, key, disk_gb, datacenters=None):
    with open(AGENT, "rb") as f:
        agent = base64.b64encode(f.read().replace(b"\r\n", b"\n")).decode()
    args = [RUNPODCTL, "pod", "create", "--name", "reelmint-fast-gpu", "--image", c["image"], "--gpu-id", gpu,
            "--cloud-type", cloud, "--container-disk-in-gb", str(disk_gb), "--ports", f"{GATE_PORT}/http",
            "--docker-args", START_CMD, "--env", json.dumps({"RM_KEY": key, "RM_AGENT": agent})]
    if datacenters:
        args += ["--data-center-ids", ",".join(datacenters)]
    code, out, err = _run(args, timeout=180)
    text = out + err
    if code != 0 or "{" not in out:
        return None, text.strip().splitlines()[-1] if text.strip() else "no answer from RunPod"
    return json.loads(out[out.index("{"):]), ""


def start(gpu_id, cloud):
    """Rent the card the person picked (gpu_id from CATALOG, cloud "COMMUNITY" or "SECURE")."""
    card = next((c for c in CATALOG if c["id"] == gpu_id), None)
    if not card or cloud not in ("COMMUNITY", "SECURE"):
        raise ValueError("pick a card and community or secure first")
    with lock:
        if state["status"] not in ("off", "error") or state["pod"]:
            return False
        _generation[0] += 1
        generation = _generation[0]
        state.update(status="starting", gpu=card["label"],
                     message=f"Renting an {card['label']} ({cloud.lower()} cloud)…")
    threading.Thread(target=_start, args=(card, cloud, generation), daemon=True).start()
    return True


def _pod_phase():
    """What RunPod says the pod is doing, in a few words ("" if it can't say)."""
    try:
        code, out, err = _run([RUNPODCTL, "pod", "get", state["pod"]], timeout=30)
        p = json.loads(out[out.index("{"):])
    except Exception:
        return ""
    if p.get("runtimeStatus") == "running":
        return "booting"
    # while the image is pulled RunPod says runtimeStatus "initializing", runtimeStatusReason "awaiting_container"
    if p.get("runtimeStatusReason") == "awaiting_container" or re.search(r"pull|download|image",
                                                                         str(p.get("lastStatusChange", "")), re.I):
        return "RunPod is still preparing it"
    return ""


def _wait_agent(minutes):
    """Until the pod has booted and its agent answers through the gateway (False if Stop was pressed). Meanwhile
    the panel shows what RunPod says the machine is doing: on some machines the ~10 GB image isn't there yet and
    downloading it took 12+ minutes (billed), so the person can see it and stop."""
    start = time.time()
    phase = ""
    while time.time() - start < minutes * 60:
        if state["status"] != "starting":
            return False
        try:
            code, body = _call("GET", "/rm/ping", timeout=20)  # the gateway itself can answer 200 while the pod boots
            if code == 200 and body == b"reelmint-agent":
                return True
            if code == 403:
                raise RuntimeError("the pod didn't accept this session's key")
        except (urllib.error.URLError, OSError):
            pass
        if int(time.time() - start) % 20 < 6:  # ask RunPod about every 20 s
            phase = _pod_phase() or phase
        mins = int((time.time() - start) // 60)
        _set(message=f"Starting the machine{f' ({phase})' if phase else ''}… {mins} min" if mins else
             f"Starting the machine{f' ({phase})' if phase else ''}…")
        time.sleep(5)
    raise RuntimeError(f"the machine didn't start within {minutes} minutes"
                       f"{f' (RunPod: {phase})' if phase else ''}; another machine is usually ready in under a minute,"
                       " so try again")


def _start(card, cloud, generation=None):
    with _lifecycle:
        if (generation is not None and generation != _generation[0]) or state["status"] != "starting":
            return
        return _start_locked(card, cloud)


def _start_locked(card, cloud):
    c = cfg()
    key = secrets.token_urlsafe(24)
    plan = h3_plan(card)
    video = "h3" if plan else "ltx25"
    h3_unet, h3_short = plan or ("", 768)
    note, why = "", ""
    try:
        pod = None
        if video == "h3":
            for attempt in range(2):
                dcs = h3_stock().get(card["id"], [])
                if dcs:
                    pod, why = _create(c, card["id"], cloud, key, disk_gb(card, "h3", h3_unet), dcs)
                if pod or not dcs or state["status"] != "starting":
                    break
                _h3_stock[0] = 0  # RunPod's stock list was stale: read it again and try once more
                time.sleep(3)
            if not pod:  # none free where H3 may run: rent it anywhere, videos use LTX-2.5
                video, note = "ltx25", " (no free one where MiniMax H3 is allowed, so videos use LTX-2.5)"
        if not pod and state["status"] == "starting":
            pod, why = _create(c, card["id"], cloud, key, disk_gb(card, video, h3_unet))
        if not pod:
            if state["status"] != "starting":
                return
            return _set(status="error", message=f"Couldn't rent an {card['label']} (RunPod: {why}). "
                                                "Pick another card or try again.")
        price = pod.get("costPerHr") or next((o[cloud.lower()] for o in _opts[1] if o["id"] == card["id"]), 0)
        # Record the rental before another blocking call, including one returned after Stop was pressed.
        _set(pod=pod["id"], key=key, price=float(price or 0), started=time.time(), last_used=time.time(), loras=[])
        if state["status"] != "starting":
            _delete("Stopped during startup")
            return
        where = _place(pod["id"])
        if state["status"] != "starting":
            _delete("Stopped during startup")
            return
        if video == "h3" and not h3_allowed(where):  # RunPod placed it elsewhere after all: no H3 there
            video, note = "ltx25", f" (it's in {where or 'an unknown place'}, where MiniMax H3 isn't allowed: videos use LTX-2.5)"
        h3_te = comfy_te(card) if video == "h3" else ""
        h3_unet = h3_unet if video == "h3" else ""
        files = pod_files(card)
        _set(video=video, h3_te=h3_te, h3_unet=h3_unet, h3_short=h3_short, where=where, files=files,
             extras=no_extras(), message="Starting the machine" + note + "…")
        if not _wait_agent(8):  # good machines answer in under a minute
            return
        with open(SETUP, "rb") as f:
            script = f.read().replace(b"\r\n", b"\n")
        # what this card installs: its video engine (and H3's text encoder), read by cloud_setup.sh
        with open(DOWNLOADER, "rb") as f:
            dl = base64.b64encode(f.read().replace(b"\r\n", b"\n")).decode()
        # + the card's file plan (pod_files: the best precision its memory holds)
        plan_lines = "".join(f"{k}={v}\n" for k, v in files.items())
        script = script.replace(b"#!/bin/bash\n", f"#!/bin/bash\nVIDEO={video}\nH3_TE={h3_te}\nH3_UNET={h3_unet}\nH3_REF={H3_REF.get(h3_unet, '')}\nFLUX2={int(card['mem'] >= FLUX2_MIN_MEM)}\n{plan_lines}DL_B64={dl}\n".encode(), 1)
        for _ in range(24):  # just after boot the gateway can still answer 404 for a while
            if state["status"] != "starting":
                return
            try:
                code, body = _call("POST", "/rm/setup", script)
            except (urllib.error.URLError, OSError):
                code = 0
            if code == 200:
                break
            time.sleep(5)
        else:
            raise RuntimeError(f"couldn't start the setup ({code})")
        _set(message=f"Installing ComfyUI, Qwen-Image 2.1 and {video_label(video, h3_unet)}{note}…")
        threading.Thread(target=_send_loras, daemon=True).start()
        deadline = time.time() + 40 * 60
        while time.time() < deadline:
            if state["status"] != "starting":
                return  # stopped meanwhile
            try:
                code, body = _call("GET", "/rm/status", timeout=20)
            except (urllib.error.URLError, OSError):
                code, body = 0, b""
            st = body.decode("utf-8", "replace").strip() if code == 200 else ""
            if st == "READY":
                break
            if st.startswith("ERROR"):
                raise RuntimeError(st[6:].strip())
            if st:
                _set(message=st[0].upper() + st[1:] + "…")
            time.sleep(8)
        else:
            raise RuntimeError("setup took over 40 minutes")
        _open_relay()
        with lock:
            if state["status"] != "starting":
                return
            _set(status="ready", message=f"{state['gpu']} ready for pictures · {video_label(video, h3_unet)} videos in a few "
                                         f"minutes{note}",
                 last_used=time.time())
        _free_local()
    except Exception as e:
        try:  # keep the end of the pod's setup log for finding out what went wrong (the pod is deleted next)
            code, body = _call("GET", "/rm/log?f=setup", timeout=20)
            if code == 200:
                with open(os.path.join(APP_DIR, "cloud_last_setup.log"), "wb") as f:
                    f.write(body)
        except Exception:
            pass
        if "FAULTY_GPU" in str(e) and cloud == "COMMUNITY":  # twice in a row RunPod gave out the same broken 5090
            _broken[card["id"]] = time.time()
            _opts[0] = 0
            fix = (f"its GPU didn't work); it was given back. RunPod keeps offering the same machine, so Community "
                   f"{card['label']} is off for an hour: pick Secure or another card")
        else:
            fix = "its GPU didn't work); pick it again for another machine, or another card"
        _delete(f"Stopped: {e}".replace("FAULTY_GPU", f"this {card['label']} was faulty ({fix}"), error=True)


def _free_local():
    """The PC's own ComfyUI keeps its last model on the 8 GB card (~2.5 GB), which made YouTube stutter while the
    rented GPU did the work: unload it (the next local job loads it again)."""
    try:
        with open(os.path.join(APP_DIR, "config.json"), encoding="utf-8") as f:
            local = json.load(f)["comfy_url"]
        req = urllib.request.Request(local + "/free", data=b'{"unload_models": true, "free_memory": true}',
                                     headers={"Content-Type": "application/json"})
        DIRECT.open(req, timeout=10).read()
    except Exception:
        pass  # ComfyUI not running: nothing to free


def _send_lora(src, name):
    """One trained character, in pieces; a piece that fails is sent again, a gap starts the file over."""
    size = os.path.getsize(src)
    q = urllib.parse.quote(name)
    with open(src, "rb") as f:
        off = 0
        while off < size:
            f.seek(off)
            piece = f.read(CHUNK)
            for _ in range(5):
                try:
                    code, body = _call("PUT", f"/rm/lora?name={q}&offset={off}", piece, timeout=300,
                                       headers={"Content-Type": "application/octet-stream"})
                except (urllib.error.URLError, OSError):
                    code = 0
                if code in (200, 409) or not state["pod"]:
                    break
                time.sleep(10)
            if code == 409:
                off = 0
                continue
            if code != 200:
                return False
            off += len(piece)
    code, body = _call("POST", f"/rm/lora?name={q}&size={size}", b"")
    return code == 200


def _send_loras():
    """Copy the Qwen-Image 2.1, Krea 2 and MOSS LoRAs to the pod (uploads are slow, so it runs beside the setup)."""
    c = json.load(open(os.path.join(APP_DIR, "config.json"), encoding="utf-8"))
    models = os.path.join(os.path.dirname(APP_DIR), "models", "loras")
    for l in c.get("loras", []):
        src = os.path.join(models, l["file"])
        if l.get("model") or l.get("engine") not in ("qwen-image-21", "krea2-turbo", "moss-image") \
                or not os.path.exists(src) or not state["pod"]:
            continue
        try:
            ok = _send_lora(src, l["file"])
        except Exception:
            ok = False
        if ok:
            with lock:
                state["loras"] = state["loras"] + [l["file"]]


class _Relay(BaseHTTPRequestHandler):
    """127.0.0.1:<local_port> -> the pod's agent, with the session key added."""
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def forward(self):
        data = self.rfile.read(int(self.headers.get("Content-Length") or 0)) or None
        hdrs = {k: v for k, v in self.headers.items() if k.lower() in ("content-type", "accept")}
        try:
            if not state["pod"]:
                raise OSError("no rented GPU")
            code, body = _call(self.command, self.path, data, timeout=300, headers=hdrs)
            ctype = "application/octet-stream"
        except (urllib.error.URLError, OSError) as e:
            code, body, ctype = 502, str(e).encode(), "text/plain"
        if body[:1] in (b"{", b"["):
            ctype = "application/json"
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    do_GET = do_POST = do_PUT = do_DELETE = forward


def _open_relay():
    if relay[0] is None:
        relay[0] = ThreadingHTTPServer(("127.0.0.1", cfg()["local_port"]), _Relay)
        threading.Thread(target=relay[0].serve_forever, daemon=True).start()
    for _ in range(10):
        try:
            DIRECT.open(f"http://127.0.0.1:{cfg()['local_port']}/system_stats", timeout=30).read()
            return
        except Exception:
            time.sleep(3)
    raise RuntimeError("couldn't reach ComfyUI on the pod")


def stop(reason="Stopped"):
    with lock:
        if state["status"] == "stopping" or (state["status"] == "off" and not state["pod"]):
            return False
        generation = _generation[0]
        state.update(status="stopping", message="Deleting the pod…")
    threading.Thread(target=_delete, args=(reason, False, generation), daemon=True).start()
    return True


def _delete(reason, error=False, generation=None):
    with _lifecycle:
        if generation is not None and generation != _generation[0]:
            return
        return _delete_locked(reason, error)


def _delete_locked(reason, error=False):
    pod = state["pod"]
    for _ in range(3):
        if not pod:
            break
        try:
            code, out, err = _run([RUNPODCTL, "pod", "delete", pod], timeout=60)
        except (OSError, subprocess.SubprocessError):
            code, out, err = 1, "", "delete command failed"
        if code == 0 or "not found" in (out + err).lower():
            pod = None
            break
        time.sleep(5)
    if pod:  # keep it in the state file so the watchdog tries again
        return _set(status="error", message=f"Couldn't delete pod {pod}: delete it on runpod.io (it is still billing)")
    cost = info()["cost"]
    _set(status="error" if error else "off", pod=None, key="", started=0.0, loras=[], video=None, h3_te="", h3_unet="",
         where="", files={}, extras=no_extras(),
         message=f"{reason} (this session: about ${cost:.2f})")


def _watch():
    """Every 30 s: stop when idle or past max_hours, retry a failed delete."""
    while True:
        time.sleep(30)
        try:
            c = cfg()
            s = state["status"]
            if s == "ready":
                if time.time() - state["started"] > c["max_hours"] * 3600:
                    stop(f"Stopped after {c['max_hours']} hours")
                elif not active[0] and time.time() - state["last_used"] > c["idle_minutes"] * 60:
                    stop(f"Stopped after {c['idle_minutes']} idle minutes")
                elif not all(state["extras"].values()):
                    _check_extras()
            elif s in ("error", "off") and state["pod"]:
                _delete("Stopped")
        except Exception:
            pass


def comfy_te(card):
    """H3's text encoder for this card: full precision (48 GB) on 80 GB+ cards; below that the 16 GB 4-bit one on
    Blackwell (tested on an RTX PRO 6000), int8 elsewhere."""
    if card["mem"] >= 80:
        return "qwen3vl_32b_minimax_h3_bf16.safetensors"
    return "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors" if card.get("blackwell") else \
        "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"


def _check_extras():
    """The video engine, full-quality Qwen and Ideogram 4 download after the card is ready (pictures work at once):
    switch each on when it's in."""
    try:
        code, body = _call("GET", "/rm/extras", timeout=20)
        got = json.loads(body) if code == 200 else {}
    except (urllib.error.URLError, OSError, ValueError):
        return
    extras = {k: bool(got.get(k)) for k in EXTRAS}
    if extras != state["extras"]:
        _set(extras=extras)


def resume():
    """App start: pick up a pod from before the restart, or delete it if it can't be reached."""
    try:
        with open(STATE_FILE, encoding="utf-8") as f:
            saved = json.load(f)
    except Exception:
        saved = None
    if saved and saved.get("pod"):
        _set(**{"key": "", **saved}, status="starting", message="Reconnecting to the running GPU…", last_used=time.time())
        try:
            code, body = _call("GET", "/rm/status", timeout=30)
            if code != 200 or body.strip() != b"READY":
                raise RuntimeError("pod not ready")
            code, body = _call("GET", "/rm/loras", timeout=30)
            with lock:
                state["loras"] = json.loads(body) if code == 200 else []
            _open_relay()
            _set(status="ready", message=f"{saved.get('gpu', 'GPU')} ready")
            _check_extras()
        except Exception:
            _delete("Stopped the pod left from before the restart")
    threading.Thread(target=_watch, daemon=True).start()
