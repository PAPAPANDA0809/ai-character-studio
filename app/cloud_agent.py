"""Runs on the rented pod (the pod's start command unpacks it from the RM_AGENT variable). It is the only thing the
app talks to, through RunPod's HTTPS gateway (https://<pod>-8189.proxy.runpod.net), so pods without a public
address work too. Every request needs the session's key in X-RM-Key (RM_KEY, made by the app for each pod).

  GET  /rm/ping                "reelmint-agent"
  GET  /rm/status              setup progress (/root/status): "... ", READY or ERROR: ...
  POST /rm/setup               body = cloud_setup.sh; started once
  PUT  /rm/lora?name=&offset=  a piece of a trained character (offset 0 starts the file)
  POST /rm/lora?name=&size=    the last piece arrived: check the size and make it visible to ComfyUI
  GET  /rm/loras               the trained characters on the pod
  GET  /rm/extras              {"video": bool, "qwen_hq": bool, "ideogram": bool, ...}: the downloads after READY
  GET  /rm/log?f=setup|comfy|pip|longcat  the end of a log, for finding out what went wrong
  PUT  /rm/talk/in?id=&name=image.png|audio.wav   a talking-video job's picture and voice track
  POST /rm/talk?id=            body = job.json: start LongCat-Video-Avatar on it (one job at a time)
  GET  /rm/talk?id=            {"status": ..., "size": bytes of out.mp4, "log": the end of its log on an error}
  GET  /rm/talk/out?id=&offset=&size=  a piece of the finished video
  POST /rm/talk/stop?id=       stop the job
  anything else                passed on to ComfyUI on 127.0.0.1:8188
"""
import hmac, http.client, json, os, re, signal, subprocess, urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

KEY = os.environ.get("RM_KEY", "")
LORAS = "/root/loras"
TALK = "/root/talk"
LOGS = {"setup": "/root/setup.log", "comfy": "/root/comfy.log", "pip": "/root/pip.log", "longcat": "/root/longcat_setup.log"}
started = [False]
talk_proc = [None]  # the running LongCat job


def read(path, default=""):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return default


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def send(self, code, body=b"", ctype="text/plain; charset=utf-8"):
        if isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def body(self):
        return self.rfile.read(int(self.headers.get("Content-Length") or 0))

    def handle_any(self):
        data = self.body()
        if not KEY or not hmac.compare_digest(self.headers.get("X-RM-Key", ""), KEY):
            return self.send(403, "no")
        u = urllib.parse.urlsplit(self.path)
        q = dict(urllib.parse.parse_qsl(u.query))
        if u.path == "/rm/ping":  # the gateway can answer 200 itself while the pod boots: this proves it's the agent
            return self.send(200, "reelmint-agent")
        if u.path == "/rm/status":
            return self.send(200, read("/root/status", "starting").strip() or "starting")
        if u.path == "/rm/setup" and self.command == "POST":
            if not started[0]:
                started[0] = True
                os.makedirs(LORAS, exist_ok=True)
                with open("/root/setup.sh", "wb") as f:
                    f.write(data)
                subprocess.Popen(["bash", "/root/setup.sh"], stdin=subprocess.DEVNULL, stdout=open(LOGS["setup"], "ab"),
                                 stderr=subprocess.STDOUT, start_new_session=True, cwd="/root")
            return self.send(200, "started")
        if u.path == "/rm/lora":
            name = os.path.basename(q.get("name", ""))
            if not re.fullmatch(r"[\w.-]+\.safetensors", name):
                return self.send(400, "bad name")
            part = os.path.join(LORAS, ".part-" + name)
            if self.command == "PUT":
                offset = int(q.get("offset", 0))
                if offset == 0:
                    open(part, "wb").close()
                if os.path.getsize(part) != offset:  # a piece went missing: the app starts this file again
                    return self.send(409, str(os.path.getsize(part)))
                with open(part, "ab") as f:
                    f.write(data)
                return self.send(200, "ok")
            if self.command == "POST":
                if not os.path.exists(part) or os.path.getsize(part) != int(q.get("size", -1)):
                    return self.send(409, "incomplete")
                os.replace(part, os.path.join(LORAS, name))
                return self.send(200, "ok")
        if u.path == "/rm/extras":  # downloaded after READY: full-quality Qwen, Ideogram 4
            return self.send(200, json.dumps({k: os.path.exists(f"/root/done_{k}") for k in ("video", "qwen_hq", "qwen_turbo", "ideogram", "flux2", "krea2", "krea2ref", "krea2_edit", "moss", "longcat")}),
                             "application/json")
        if u.path.startswith("/rm/talk"):
            return self.talk(u.path, q, data)
        if u.path == "/rm/loras":
            files = sorted(f for f in os.listdir(LORAS) if not f.startswith(".")) if os.path.isdir(LORAS) else []
            return self.send(200, json.dumps(files), "application/json")
        if u.path == "/rm/log":
            return self.send(200, "\n".join(read(LOGS.get(q.get("f"), "/dev/null")).splitlines()[-80:]))
        # ComfyUI
        try:
            c = http.client.HTTPConnection("127.0.0.1", 8188, timeout=600)
            hdrs = {k: v for k, v in self.headers.items() if k.lower() in ("content-type", "accept")}
            c.request(self.command, self.path, body=data or None, headers=hdrs)
            r = c.getresponse()
            out = r.read()
            self.send(r.status, out, r.getheader("Content-Type") or "application/octet-stream")
        except OSError as e:
            self.send(503, f"ComfyUI isn't running yet ({e})")

    def talk(self, path, q, data):
        """Talking videos (LongCat-Video-Avatar 1.6): it runs as its own process beside ComfyUI, not as a ComfyUI
        node (none supports 1.6 yet); the app frees ComfyUI's memory before it starts a job."""
        jid = q.get("id", "")
        if not re.fullmatch(r"[0-9a-f]{8,32}", jid):
            return self.send(400, "bad id")
        d = os.path.join(TALK, jid)
        running = talk_proc[0] is not None and talk_proc[0].poll() is None
        if path == "/rm/talk/in" and self.command == "PUT":
            name = q.get("name", "")
            if name not in ("image.png", "audio.wav"):
                return self.send(400, "bad name")
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, name), "wb") as f:
                f.write(data)
            return self.send(200, "ok")
        if path == "/rm/talk" and self.command == "POST":
            if running:
                return self.send(409, "another talking video is being made")
            if not os.path.exists("/root/done_longcat"):
                return self.send(409, "LongCat isn't installed on this card yet")
            if not all(os.path.exists(os.path.join(d, n)) for n in ("image.png", "audio.wav")):
                return self.send(400, "picture or voice missing")
            with open(os.path.join(d, "job.json"), "wb") as f:
                f.write(data)
            with open(os.path.join(d, "status"), "w") as f:
                f.write("starting")
            env = {**os.environ, "PYTHONPATH": "/root/lcpkgs", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
            with open(os.path.join(d, "log"), "ab") as log:
                talk_proc[0] = subprocess.Popen(["python3", "/root/longcat_run.py", d], stdin=subprocess.DEVNULL,
                                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True, cwd=d,
                                                env=env)
            return self.send(200, "started")
        if path == "/rm/talk" and self.command == "GET":
            st = read(os.path.join(d, "status"), "").strip()
            if not st:
                return self.send(404, "no such job")
            if st not in ("DONE",) and not st.startswith("ERROR") and not running:
                st = "ERROR: the LongCat process stopped unexpectedly"
            out = os.path.join(d, "out.mp4")
            log = "\n".join(read(os.path.join(d, "log")).splitlines()[-30:]) if st.startswith("ERROR") else ""
            return self.send(200, json.dumps({"status": st, "size": os.path.getsize(out) if os.path.exists(out) else 0,
                                              "log": log}), "application/json")
        if path == "/rm/talk/out":
            with open(os.path.join(d, "out.mp4"), "rb") as f:
                f.seek(int(q.get("offset", 0)))
                return self.send(200, f.read(int(q.get("size", 8 << 20))), "application/octet-stream")
        if path == "/rm/talk/stop" and self.command == "POST":
            if running:
                os.killpg(talk_proc[0].pid, signal.SIGKILL)
            return self.send(200, "stopped")
        return self.send(404, "unknown")

    do_GET = do_POST = do_PUT = do_DELETE = handle_any


ThreadingHTTPServer(("0.0.0.0", 8189), H).serve_forever()
