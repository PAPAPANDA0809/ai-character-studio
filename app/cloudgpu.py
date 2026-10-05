"""Fast GPU session: a rented RunPod card running ComfyUI with Qwen-Image 2.1 and Wan 2.2, reached through an SSH
tunnel. While it is ready, Qwen-Image pictures and edits and Wan 2.2 videos run there instead of on this PC's 8 GB card.

The person picks the card from options() (live RunPod stock and prices); nothing is chosen automatically. Jobs run
one at a time, as on the PC. The pod is set up from scratch each time (no paid storage between sessions) by cloud_setup.sh, and it is deleted
when idle for idle_minutes, after max_hours, or from the Stop button. A pod left from before an app restart is
picked up again, or deleted if it can't be reached, so a forgotten pod never keeps billing."""
import json, os, re, subprocess, threading, time, urllib.request

APP_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(APP_DIR, "cloud_gpu.json")  # the running pod, so a restart can find it
SETUP = os.path.join(APP_DIR, "cloud_setup.sh")
RUNPODCTL = os.path.expanduser(r"~\.local\bin\runpodctl.exe")
# Git's ssh, not Windows' own: Windows OpenSSH refuses the RunPod key because the key file is also readable by a
# sandbox group ("UNPROTECTED PRIVATE KEY FILE"). Git's ssh takes /c/... style paths.
SSH = r"C:\Program Files\Git\usr\bin\ssh.exe"
SCP = r"C:\Program Files\Git\usr\bin\scp.exe"


def msys(path):
    path = os.path.abspath(path).replace(os.sep, "/")
    return "/" + path[0].lower() + path[2:]


KEY = msys(os.path.expanduser(r"~\.runpod\ssh\runpodctl-ssh-key"))
KNOWN_HOSTS = msys(os.path.join(APP_DIR, "cloud_known_hosts"))
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
DIRECT = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # the VPN's system proxy breaks localhost

# The cards offered (the RTX 4090 and stronger; the list is shown cheapest first). speed = rough speed for Qwen-Image / Wan against an RTX 4090 (estimates from the
# cards' compute and memory speed, not measured here).
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
state = {"status": "off", "message": "", "pod": None, "ip": "", "port": 0, "price": 0.0, "gpu": "",
         "started": 0.0, "last_used": 0.0, "loras": []}
active = [0]       # cloud jobs running right now (never idle-stopped meanwhile)
tunnel = [None]    # the ssh -L process
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
        keep = {k: state[k] for k in ("pod", "ip", "port", "price", "gpu", "started")}
    if state["pod"]:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(keep, f)
    elif os.path.exists(STATE_FILE):
        os.remove(STATE_FILE)


def info():
    """For the page: status, what it costs so far, when it stops by itself."""
    c = cfg()
    with lock:
        s = dict(state)
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
    """ComfyUI on the rented card, or None while there's no ready session."""
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


def _ssh(cmd, timeout=40, data=None):
    return _run([SSH, "-i", KEY, "-p", str(state["port"]), "-o", "StrictHostKeyChecking=no", "-o", f"UserKnownHostsFile={KNOWN_HOSTS}",
                 "-o", "ConnectTimeout=20", "-o", "BatchMode=yes", f"root@{state['ip']}", cmd], timeout, data)


def _create(c, gpu, cloud):
    args = [RUNPODCTL, "pod", "create", "--name", "reelmint-fast-gpu", "--image", c["image"], "--gpu-id", gpu,
            "--cloud-type", cloud, "--container-disk-in-gb", str(c["disk_gb"]), "--ports", "22/tcp", "--wait", "--wait-timeout", "8m"]
    if cloud == "COMMUNITY":
        args.append("--public-ip")  # ssh straight to the pod
    code, out, err = _run(args, timeout=600)
    text = out + err
    if code != 0 or "{" not in out:
        kept = re.search(r"\b([a-z0-9]{14})\b", text)  # timed out after renting: the error carries the pod id
        if kept and "kept" in text:
            _run([RUNPODCTL, "pod", "delete", kept.group(1)], timeout=60)
        return None, text.strip().splitlines()[-1] if text.strip() else "no answer from RunPod"
    pod = json.loads(out[out.index("{"):])
    return pod, ""


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


def _start(card, cloud):
    c = cfg()
    try:
        pod, why = _create(c, card["id"], cloud)
        if pod and state["status"] != "starting":  # Stop was pressed while it was being rented
            _run([RUNPODCTL, "pod", "delete", pod["id"]], timeout=60)
            return
        if not pod:
            if cloud == "COMMUNITY":  # the usual cause: stock counts every machine, but only ones with a public address work
                return _set(status="error", message=f"No community {card['label']} with a public address is free right now "
                            f"(RunPod: {why}). Pick Secure or another card.")
            return _set(status="error", message=f"Couldn't rent an {card['label']} ({why}). Pick another card or try again.")
        ssh = pod.get("ssh", {})
        _set(pod=pod["id"], ip=ssh.get("ip", ""), port=int(ssh.get("port", 0)), price=float(pod.get("costPerHr", 0)),
             started=time.time(), last_used=time.time(), loras=[], message="Installing ComfyUI, Qwen-Image 2.1 and Wan 2.2…")
        with open(SETUP, "rb") as f:
            script = f.read().replace(b"\r\n", b"\n")
        for _ in range(12):  # ssh can answer before the pod has installed the key: retry for ~2 minutes
            # ";" not "&&": with "&&" the whole chain went to the background and the session ended before cat had
            # read the script (it arrived empty)
            code, out, err = _ssh(f"cat > /root/setup.sh; mkdir -p /root/loras; "
                                  "setsid nohup bash /root/setup.sh > /root/setup.log 2>&1 < /dev/null & echo started",
                                  timeout=60, data=script)
            if "started" in out or state["status"] != "starting":
                break
            time.sleep(10)
        if "started" not in out:
            raise RuntimeError("couldn't reach the pod: " + (err.strip().splitlines() or ["no answer"])[-1])
        threading.Thread(target=_send_loras, daemon=True).start()
        deadline = time.time() + 40 * 60
        while time.time() < deadline:
            if state["status"] != "starting":
                return  # stopped meanwhile
            code, out, err = _ssh("cat /root/status 2>/dev/null")
            st = out.strip()
            if st == "READY":
                break
            if st.startswith("ERROR"):
                raise RuntimeError(st[6:].strip())
            if st:
                _set(message=st[0].upper() + st[1:] + "…")
            time.sleep(8)
        else:
            raise RuntimeError("setup took over 40 minutes")
        _open_tunnel()
        _set(status="ready", message=f"{state['gpu']} ready", last_used=time.time())
        _free_local()
    except Exception as e:
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


def _send_loras():
    """Copy the Qwen-Image 2.1 trained characters to the pod (uploads are slow, so it runs beside the setup)."""
    c = json.load(open(os.path.join(APP_DIR, "config.json"), encoding="utf-8"))
    models = os.path.join(os.path.dirname(APP_DIR), "models", "loras")
    for l in c.get("loras", []):
        src = os.path.join(models, l["file"])
        if l.get("engine") != "qwen-image-21" or not os.path.exists(src) or not state["pod"]:
            continue
        for _ in range(3):
            code, out, err = _run([SCP, "-q", "-i", KEY, "-P", str(state["port"]), "-o", "StrictHostKeyChecking=no",
                                   "-o", f"UserKnownHostsFile={KNOWN_HOSTS}", "-o", "BatchMode=yes", msys(src),
                                   f"root@{state['ip']}:/root/loras/{l['file']}"], timeout=3600)
            if code == 0 or not state["pod"]:
                break
            time.sleep(15)
        if code == 0:
            with lock:
                state["loras"] = state["loras"] + [l["file"]]


def _open_tunnel():
    if tunnel[0] and tunnel[0].poll() is None:
        tunnel[0].kill()
    port = cfg()["local_port"]
    tunnel[0] = subprocess.Popen([SSH, "-i", KEY, "-p", str(state["port"]), "-o", "StrictHostKeyChecking=no",
                                  "-o", f"UserKnownHostsFile={KNOWN_HOSTS}", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=20",
                                  "-o", "ExitOnForwardFailure=yes", "-N", "-L", f"{port}:127.0.0.1:8188", f"root@{state['ip']}"],
                                 stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=NO_WINDOW)
    for _ in range(30):
        try:
            DIRECT.open(f"http://127.0.0.1:{port}/system_stats", timeout=5).read()
            return
        except Exception:
            time.sleep(2)
    raise RuntimeError("couldn't connect to ComfyUI on the pod")


def stop(reason="Stopped"):
    with lock:
        if state["status"] in ("off", "stopping"):
            return False
        state.update(status="stopping", message="Deleting the pod…")
    threading.Thread(target=_delete, args=(reason,), daemon=True).start()
    return True


def _delete(reason, error=False):
    if tunnel[0] and tunnel[0].poll() is None:
        tunnel[0].kill()
    tunnel[0] = None
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
    _set(status="error" if error else "off", pod=None, ip="", port=0, started=0.0, loras=[],
         message=f"{reason} (this session: about ${cost:.2f})")


def _watch():
    """Every 30 s: reconnect a dropped tunnel, stop when idle or past max_hours, retry a failed delete."""
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
                elif tunnel[0] is None or tunnel[0].poll() is not None:
                    _open_tunnel()
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
        _set(**saved, status="starting", message="Reconnecting to the running GPU…", last_used=time.time())
        try:
            code, out, err = _ssh("cat /root/status 2>/dev/null; ls /root/loras 2>/dev/null")
            lines = out.split()
            if not lines or lines[0] != "READY":
                raise RuntimeError("pod not ready")
            with lock:
                state["loras"] = lines[1:]
            _open_tunnel()
            _set(status="ready", message=f"{saved.get('gpu', 'GPU')} ready")
        except Exception:
            _delete("Stopped the pod left from before the restart")
    threading.Thread(target=_watch, daemon=True).start()
