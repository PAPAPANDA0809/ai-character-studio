"""AI Character Studio - local web app. Run: python server.py  (then open http://127.0.0.1:7860)"""
import json
import mimetypes
import os
import re
import sys
import urllib.parse
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # embedded Python doesn't add the script dir
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import comfy
import pipeline

STATIC = os.path.join(pipeline.APP_DIR, "static")


def ollama_alive(cfg):
    if cfg["llm"]["provider"] != "ollama":
        return None
    try:
        urllib.request.urlopen(cfg["llm"]["ollama_url"] + "/api/tags", timeout=3)
        return True
    except Exception:
        return False


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def send_json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path):
        if not os.path.isfile(path):
            return self.send_json({"error": "not found"}, 404)
        ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
        size = os.path.getsize(path)
        rng = self.headers.get("Range")
        start, end = 0, size - 1
        if rng and (m := re.match(r"bytes=(\d*)-(\d*)", rng)):
            if m.group(1):
                start = int(m.group(1))
                end = int(m.group(2)) if m.group(2) else size - 1
            elif m.group(2):
                start = size - int(m.group(2))
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        with open(path, "rb") as f:
            f.seek(start)
            left = end - start + 1
            while left > 0:
                chunk = f.read(min(1 << 20, left))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    return
                left -= len(chunk)

    def body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self):
        path = self.path.split("?")[0]
        try:
            if path == "/":
                return self.send_file(os.path.join(STATIC, "index.html"))
            if path == "/api/status":
                cfg = pipeline.load_config()
                return self.send_json({
                    "comfy": comfy.Comfy(cfg["comfy_url"]).alive(),
                    "ollama": ollama_alive(cfg),
                    "llm": cfg["llm"]["provider"] + ":" + (cfg["llm"]["ollama_model"] if cfg["llm"]["provider"] == "ollama"
                                                           else cfg["llm"].get("openai_model", "")),
                    "busy": pipeline.current, "queued": pipeline.jobs.qsize(),
                })
            if path == "/api/projects":
                return self.send_json(pipeline.list_projects())
            if path == "/api/characters":
                return self.send_json(pipeline.list_characters())
            if path == "/api/engines":
                return self.send_json(pipeline.engines())
            if m := re.fullmatch(r"/characters/([^/]+)", path):
                name = urllib.parse.unquote(m.group(1))
                if name not in pipeline.list_characters():
                    return self.send_json({"error": "not found"}, 404)
                return self.send_file(os.path.join(pipeline.CHARACTERS, name))
            if m := re.fullmatch(r"/api/projects/([\w\-]+)", path):
                return self.send_json(pipeline.load(m.group(1)))
            if m := re.fullmatch(r"/files/([\w\-]+)/(.+)", path):
                full = os.path.normpath(pipeline.pdir(m.group(1), *m.group(2).split("/")))
                if not full.startswith(os.path.normpath(pipeline.PROJECTS)):
                    return self.send_json({"error": "bad path"}, 400)
                return self.send_file(full)
            self.send_json({"error": "not found"}, 404)
        except Exception as e:
            self.send_json({"error": str(e)}, 500)

    def do_POST(self):
        path = self.path.split("?")[0]
        try:
            if path == "/api/characters":  # raw image bytes, ?name=file.png
                q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
                data = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                if not data:
                    return self.send_json({"error": "empty upload"}, 400)
                return self.send_json({"name": pipeline.save_character(q.get("name", ["character.png"])[0], data)})
            b = self.body()
            if path == "/api/plan":
                p = pipeline.create_project(b["idea"], b.get("keyframes", 4), b.get("style", ""),
                                            b.get("character", ""), b.get("aspect", "portrait"), b.get("seed"),
                                            b.get("reference", ""), b.get("image_engine"), b.get("video_engine"),
                                            b.get("mode", "video"))
                return self.send_json(p)
            if m := re.fullmatch(r"/api/projects/([\w\-]+)/engines", path):
                return self.send_json(pipeline.set_engines(m.group(1), b.get("image_engine"), b.get("video_engine"), b.get("mode")))
            if m := re.fullmatch(r"/api/projects/([\w\-]+)/reference", path):
                return self.send_json(pipeline.set_reference(m.group(1), b.get("reference", "")))
            if m := re.fullmatch(r"/api/projects/([\w\-]+)/plan", path):
                return self.send_json(pipeline.update_plan(m.group(1), b["plan"]))
            if m := re.fullmatch(r"/api/projects/([\w\-]+)/keyframes", path):
                pipeline.enqueue(m.group(1), "keyframes", indices=b.get("indices"), reseed=bool(b.get("reseed")))
                return self.send_json({"ok": True})
            if m := re.fullmatch(r"/api/projects/([\w\-]+)/animate", path):
                pipeline.enqueue(m.group(1), "animate", only=b.get("only"))
                return self.send_json({"ok": True})
            if m := re.fullmatch(r"/api/projects/([\w\-]+)/stitch", path):
                pipeline.enqueue(m.group(1), "stitch")
                return self.send_json({"ok": True})
            self.send_json({"error": "not found"}, 404)
        except Exception as e:
            self.send_json({"error": str(e)}, 500)


if __name__ == "__main__":
    cfg = pipeline.load_config()
    port = cfg.get("app_port", 7860)
    print(f"AI Character Studio running at http://127.0.0.1:{port}")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
