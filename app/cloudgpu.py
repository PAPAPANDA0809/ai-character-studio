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
import base64, json, os, secrets, subprocess, threading, time, urllib.error, urllib.parse, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

APP_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(APP_DIR, "cloud_gpu.json")  # the running pod and its key, so a restart can find it
SETUP = os.path.join(APP_DIR, "cloud_setup.sh")
AGENT = os.path.join(APP_DIR, "cloud_agent.py")
RUNPODCTL = os.path.expanduser(r"~\.local\bin\runpodctl.exe")
GATE_PORT = 8189
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
DIRECT = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # the VPN's system proxy breaks localhost
# The pod's start command: unpack the agent from RM_AGENT and run it (the image's own start script, sshd and
# Jupyter, isn't needed).
START_CMD = "bash -c 'echo \"$RM_AGENT\" | base64 -d > /root/agent.py && python3 /root/agent.py > /root/agent.log 2>&1'"
CHUNK = 8 << 20  # trained characters go up in 8 MB pieces (one request through the gateway can't be huge)

# The cards offered (the RTX 4090 and stronger; the list is shown cheapest first). speed = rough speed for
# Qwen-Image / Wan against an RTX 4090 (estimates from the cards' compute and memory speed, not measured here).
CATALOG = [
    {"id": "NVIDIA GeForce RTX 4090", "label": "RTX 4090", "mem": 24, "speed": 1.0},
    {"id": "NVIDIA GeForce RTX 5090", "label": "RTX 5090", "mem": 32, "speed": 1.3},
    {"id": "NVIDIA L40S", "label": "L40S", "mem": 48, "speed": 1.0},
    {"id": "NVIDIA RTX 6000 Ada Generation", "label": "RTX 6000 Ada", "mem": 48, "speed": 1.0},
    {"id": "NVIDIA A100 80GB PCIe", "label": "A100 80 GB", "mem": 80, "speed": 1.0},
    {"id": "NVIDIA RTX PRO 6000 Blackwell Server Edition", "label": "RTX PRO 6000", "mem": 96, "speed": 1.4},
    {"id": "NVIDIA H100 80GB HBM3", "label": "H100", "mem": 80, "speed": 2.0},
    {"id": "NVIDIA H200", "label": "H200", "mem": 141, "speed": 2.2},
    {"id": "NVIDIA B200", "label": "B200", "mem": 180, "speed": 3.0},
]

lock = threading.Lock()
state = {"status": "off", "message": "", "pod": None, "key": "", "price": 0.0, "gpu": "",
         "started": 0.0, "last_used": 0.0, "loras": []}
active = [0]       # cloud jobs running right now (never idle-stopped meanwhile)
relay = [None]     # the local relay server
_opts = [0.0, []]  # options() cache: when, list
_broken = {}       # card id -> when a community machine for it turned out faulty (RunPod hands the same one out again)


def cfg():
    with open(os.path.join(APP_DIR, "config.json"), encoding="utf-8") as f:
        c = json.load(f).get("cloud_gpu", {})
    return {"image": "runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404", "disk_gb": 90, "local_port": 8190,
            "idle_minutes": 15, "max_hours": 4, **c}


def _set(**kw):
    with lock:
        state.update(kw)
        keep = {k: state[k] for k in ("pod", "key", "price", "gpu", "started")}
    if state["pod"]:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(keep, f)
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
            "idle_stop_in": max(0, round(idle_left / 60)) if idle_left is not None else None,
            "max_hours": c["max_hours"], "idle_minutes": c["idle_minutes"], "busy": active[0]}


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
                    "community_broken": time.time() - _broken.get(c["id"], 0) < 3600})
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


def _create(c, gpu, cloud, key):
    with open(AGENT, "rb") as f:
        agent = base64.b64encode(f.read().replace(b"\r\n", b"\n")).decode()
    args = [RUNPODCTL, "pod", "create", "--name", "reelmint-fast-gpu", "--image", c["image"], "--gpu-id", gpu,
            "--cloud-type", cloud, "--container-disk-in-gb", str(c["disk_gb"]), "--ports", f"{GATE_PORT}/http",
            "--docker-args", START_CMD, "--env", json.dumps({"RM_KEY": key, "RM_AGENT": agent})]
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
        if state["status"] not in ("off", "error"):
            return False
        state.update(status="starting", gpu=card["label"],
                     message=f"Renting an {card['label']} ({cloud.lower()} cloud)…")
    threading.Thread(target=_start, args=(card, cloud), daemon=True).start()
    return True


def _wait_agent(minutes):
    """Until the pod has booted and its agent answers through the gateway (False if Stop was pressed)."""
    deadline = time.time() + minutes * 60
    while time.time() < deadline:
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
        time.sleep(5)
    raise RuntimeError(f"the pod didn't start within {minutes} minutes")


def _start(card, cloud):
    c = cfg()
    key = secrets.token_urlsafe(24)
    try:
        pod, why = _create(c, card["id"], cloud, key)
        if pod and state["status"] != "starting":  # Stop was pressed while it was being rented
            _run([RUNPODCTL, "pod", "delete", pod["id"]], timeout=60)
            return
        if not pod:
            return _set(status="error", message=f"Couldn't rent an {card['label']} (RunPod: {why}). "
                                                "Pick another card or try again.")
        price = pod.get("costPerHr") or next((o[cloud.lower()] for o in _opts[1] if o["id"] == card["id"]), 0)
        _set(pod=pod["id"], key=key, price=float(price or 0), started=time.time(), last_used=time.time(), loras=[],
             message="Starting the machine…")
        if not _wait_agent(12):
            return
        with open(SETUP, "rb") as f:
            script = f.read().replace(b"\r\n", b"\n")
        for _ in range(24):  # just after boot the gateway can still answer 404 for a while
            try:
                code, body = _call("POST", "/rm/setup", script)
            except (urllib.error.URLError, OSError):
                code = 0
            if code == 200:
                break
            time.sleep(5)
        else:
            raise RuntimeError(f"couldn't start the setup ({code})")
        _set(message="Installing ComfyUI, Qwen-Image 2.1 and Wan 2.2…")
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
        _set(status="ready", message=f"{state['gpu']} ready", last_used=time.time())
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
    """Copy the Qwen-Image 2.1 trained characters to the pod (uploads are slow, so it runs beside the setup)."""
    c = json.load(open(os.path.join(APP_DIR, "config.json"), encoding="utf-8"))
    models = os.path.join(os.path.dirname(APP_DIR), "models", "loras")
    for l in c.get("loras", []):
        src = os.path.join(models, l["file"])
        if l.get("engine") != "qwen-image-21" or not os.path.exists(src) or not state["pod"]:
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
        if state["status"] in ("off", "stopping"):
            return False
        state.update(status="stopping", message="Deleting the pod…")
    threading.Thread(target=_delete, args=(reason,), daemon=True).start()
    return True


def _delete(reason, error=False):
    pod = state["pod"]
    for _ in range(3):
        if not pod:
            break
        code, out, err = _run([RUNPODCTL, "pod", "delete", pod], timeout=60)
        if code == 0 or "not found" in (out + err).lower():
            pod = None
            break
        time.sleep(5)
    if pod:  # keep it in the state file so the watchdog tries again
        return _set(status="error", message=f"Couldn't delete pod {pod}: delete it on runpod.io (it is still billing)")
    cost = info()["cost"]
    _set(status="error" if error else "off", pod=None, key="", started=0.0, loras=[],
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
            elif s == "error" and state["pod"]:
                _delete("Stopped")
        except Exception:
            pass


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
        except Exception:
            _delete("Stopped the pod left from before the restart")
    threading.Thread(target=_watch, daemon=True).start()
