"""Fast GPU session: a rented RunPod card (an RTX 4090 by default) running ComfyUI with Qwen-Image 2.1, reached
through an SSH tunnel. While it is ready, Qwen-Image pictures and edits run there instead of on this PC's 8 GB card.

The pod is set up from scratch each time (no paid storage between sessions) by cloud_setup.sh, and it is deleted
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

lock = threading.Lock()
state = {"status": "off", "message": "", "pod": None, "ip": "", "port": 0, "price": 0.0, "gpu": "",
         "started": 0.0, "last_used": 0.0, "loras": []}
active = [0]       # cloud jobs running right now (never idle-stopped meanwhile)
tunnel = [None]    # the ssh -L process


def cfg():
    with open(os.path.join(APP_DIR, "config.json"), encoding="utf-8") as f:
        c = json.load(f).get("cloud_gpu", {})
    # tries: [RunPod GPU id, label, cloud type] in order, cheapest first (prices per hour, Oct 2026)
    return {"tries": [["NVIDIA GeForce RTX 4090", "RTX 4090", "COMMUNITY"],   # $0.34
                      ["NVIDIA GeForce RTX 5090", "RTX 5090", "COMMUNITY"],   # $0.69
                      ["NVIDIA GeForce RTX 4090", "RTX 4090", "SECURE"],      # $0.74
                      ["NVIDIA GeForce RTX 5090", "RTX 5090", "SECURE"]],     # $0.99
            # CUDA 13 build: ComfyUI's fast int8 kernels need it ("You need pytorch with cu130")
            "image": "runpod/pytorch:1.4.0-cu1300-torch291-ubuntu2404", "disk_gb": 60, "local_port": 8190,
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
    return {**s, "label": s["gpu"] or c["tries"][0][1], "minutes": round(up / 60), "cost": round(up / 3600 * s["price"], 2),
            "idle_stop_in": max(0, round(idle_left / 60)) if idle_left is not None else None,
            "max_hours": c["max_hours"], "idle_minutes": c["idle_minutes"], "busy": bool(active[0])}


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


def start():
    with lock:
        if state["status"] not in ("off", "error"):
            return False
        state.update(status="starting", message="Renting the GPU…")
    threading.Thread(target=_start, daemon=True).start()
    return True


def _start(faulty=0, first=0):
    """faulty = pods given up on because their card didn't work (community hosts): up to 2 more are tried, starting
    at the next entry of tries (RunPod handed the same broken host out again for the same card type).
    first = where in tries to start."""
    c = cfg()
    tried = first
    try:
        pod, label, why = None, "", ""
        for tried in range(first, len(c["tries"])):  # cheapest first; the next one when none is free
            gpu, label, cloud = c["tries"][tried]
            if state["status"] != "starting":
                return  # stopped meanwhile
            _set(gpu=label, message=f"Renting an {label} ({'community' if cloud == 'COMMUNITY' else 'secure'} cloud)…")
            pod, why = _create(c, gpu, cloud)
            if pod and state["status"] != "starting":  # Stop was pressed while this one was being rented
                _run([RUNPODCTL, "pod", "delete", pod["id"]], timeout=60)
                return
            if pod:
                break
        if not pod:
            return _set(status="error", gpu="", message=f"No GPU free right now ({why}). Try again in a few minutes.")
        ssh = pod.get("ssh", {})
        _set(pod=pod["id"], ip=ssh.get("ip", ""), port=int(ssh.get("port", 0)), price=float(pod.get("costPerHr", 0)),
             gpu=label, started=time.time(), last_used=time.time(), loras=[], message="Installing ComfyUI and Qwen-Image 2.1…")
        with open(SETUP, "rb") as f:
            script = f.read().replace(b"\r\n", b"\n")
        for _ in range(12):  # ssh can answer before the pod has installed the key: retry for ~2 minutes
            # ";" not "&&": with "&&" the whole chain went to the background and the session ended before cat had
            # read the script (it arrived empty)
            code, out, err = _ssh("cat > /root/setup.sh; mkdir -p /root/loras; "
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
    except Exception as e:
        if "FAULTY_GPU" in str(e) and faulty < 2 and tried + 1 < len(c["tries"]):
            _delete("The rented card didn't work", error=True)
            _set(status="starting", message="That card was faulty; renting a different one…")
            return _start(faulty + 1, tried + 1)
        _delete(f"Stopped: {e}".replace("FAULTY_GPU", "the rented cards didn't work; try again later"), error=True)


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
