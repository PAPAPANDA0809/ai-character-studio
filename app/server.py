"""Reelmint - local web app. Run: python server.py  (then open http://127.0.0.1:7860)"""
import gzip
import io
import http.cookies
import json
import mimetypes
import os
import re
import sys
import threading
import urllib.parse
import urllib.request
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # embedded Python doesn't add the script dir
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import auth
import cloudgpu
import comfy
import llm
import pipeline
import imageundo
import push

STATIC = os.path.join(pipeline.APP_DIR, "static")
PREVIEW_W = (480, 1440)  # ?w= sizes: tiles and thumbnails, and the big picture on phones


def preview(full, w):
    """A JPEG copy of picture `full` at most w pixels wide, kept in .previews beside it (made on first request).
    The name carries the picture's time and size, so a replaced picture gets a new copy; older copies are removed.
    A full PNG is ~3.5 MB, the 480 copy ~40 KB and the 1440 one ~300 KB: phones on mobile data load in moments."""
    from PIL import Image
    st = os.stat(full)
    d = os.path.join(os.path.dirname(full), ".previews")
    stem = f"{os.path.basename(full)}.{w}."
    out = os.path.join(d, f"{stem}{int(st.st_mtime)}-{st.st_size}.jpg")
    if not os.path.exists(out):
        os.makedirs(d, exist_ok=True)
        with Image.open(full) as im:
            im = im.convert("RGB")
            im.thumbnail((w, w * 4), Image.LANCZOS)
            tmp = f"{out}.{threading.get_ident()}.tmp"
            im.save(tmp, "JPEG", quality=82 if w < 1000 else 90, optimize=True)
        os.replace(tmp, out)
        for old in os.listdir(d):
            if old.startswith(stem) and old != os.path.basename(out) and old.endswith(".jpg"):
                try:
                    os.remove(os.path.join(d, old))
                except OSError:
                    pass
    return out


def ollama_alive(cfg):
    if cfg["llm"]["provider"] != "ollama":
        return None
    try:
        llm.open_url(cfg["llm"]["ollama_url"] + "/api/tags", timeout=3)
        return True
    except Exception:
        return False


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def send_json(self, obj, code=200):
        if isinstance(obj, dict) and "plan" in obj and "id" in obj:  # a project: say which pictures exist on disk
            obj = pipeline.with_present(obj)
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_body(body)

    def send_body(self, body):
        """Length, gzip for text the browser accepts it for (phones on mobile data), end of headers, body."""
        if len(body) > 1024 and "gzip" in self.headers.get("Accept-Encoding", ""):
            body = gzip.compress(body, 6)
            self.send_header("Content-Encoding", "gzip")
        self.send_header("Vary", "Accept-Encoding")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_zip(self, user, items):
        """items = [{project, path, name}] of the person's own files, stored as they are (PNG/MP4 don't shrink)."""
        buf, used = io.BytesIO(), set()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
            for it in items[:500]:
                pid, rel = str(it.get("project", "")), str(it.get("path", ""))
                if not re.fullmatch(r"[\w\-]+", pid) or pipeline.user_of(pid) != user:
                    continue
                root = os.path.normpath(pipeline.pdir(pid))
                full = os.path.normpath(pipeline.pdir(pid, *rel.split("/")))
                if not full.startswith(root + os.sep) or not os.path.isfile(full):
                    continue
                name = os.path.basename(str(it.get("name") or "")) or os.path.basename(full)
                stem, ext = os.path.splitext(name)
                n = 1
                while name in used:  # the same title twice: number it
                    n += 1
                    name = f"{stem}_{n}{ext}"
                used.add(name)
                z.write(full, name)
        body = buf.getvalue()
        self.send_response(200)
        self.send_header("Content-Type", "application/zip")
        self.send_header("Content-Disposition", f'attachment; filename="reelmint_{len(used)}_files.zip"')
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path):
        if not os.path.isfile(path):
            return self.send_json({"error": "not found"}, 404)
        ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
        st = os.stat(path)
        size = st.st_size
        # the browser keeps a copy and asks "changed?" each time; unchanged files cost a few bytes, not megabytes
        etag = f'"{int(st.st_mtime)}-{size}"'
        if self.headers.get("If-None-Match") == etag:
            self.send_response(304)
            self.send_header("ETag", etag)
            self.send_header("Cache-Control", "private, no-cache")
            self.end_headers()
            return
        if ctype.startswith("text/") or ctype in ("application/javascript", "application/json", "image/svg+xml"):
            self.send_response(200)  # pages and scripts: small enough to read whole and compress
            self.send_header("Content-Type", ctype + ("; charset=utf-8" if ctype.startswith("text/") else ""))
            self.send_header("ETag", etag)
            self.send_header("Cache-Control", "private, no-cache")
            with open(path, "rb") as f:
                return self.send_body(f.read())
        rng = self.headers.get("Range")
        start, end = 0, size - 1
        if rng and (m := re.match(r"bytes=(\d*)-(\d*)", rng)):
            if m.group(1):
                start = int(m.group(1))
                end = int(m.group(2)) if m.group(2) else size - 1
            elif m.group(2):
                start = max(0, size - int(m.group(2)))
            end = min(end, size - 1)
            if start > end:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("ETag", etag)
        self.send_header("Cache-Control", "private, no-cache")
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

    # ---- auth helpers
    def token(self):
        c = http.cookies.SimpleCookie(self.headers.get("Cookie") or "")
        return c["rm_session"].value if "rm_session" in c else None

    def user(self):
        return auth.user_for(self.token())

    def set_cookie_json(self, obj, token, max_age):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Set-Cookie", f"rm_session={token}; Path=/; Max-Age={max_age}; HttpOnly; SameSite=Lax")
        self.end_headers()
        self.wfile.write(body)

    def redirect(self, where):
        self.send_response(302)
        self.send_header("Location", where)
        self.end_headers()

    def owns(self, user, pid):
        return pipeline.user_of(pid) == user

    def do_GET(self):
        path = self.path.split("?")[0]
        try:
            # public
            if m := re.fullmatch(r"/static/([\w\-.]+\.(?:png|json|svg|ico|css))", path):
                return self.send_file(os.path.join(STATIC, m.group(1)))
            if path == "/login":
                return self.send_file(os.path.join(STATIC, "login.html"))
            if path == "/sw.js":  # the service worker lives at the root so it covers the whole app (notifications only)
                return self.send_file(os.path.join(STATIC, "sw.js"))
            if path == "/api/status":
                cfg = pipeline.load_config()
                return self.send_json({
                    "comfy": comfy.Comfy(cfg["comfy_url"]).alive(),
                    "ollama": ollama_alive(cfg),
                    "llm": cfg["llm"]["provider"] + ":" + (cfg["llm"]["ollama_model"] if cfg["llm"]["provider"] == "ollama"
                                                           else cfg["llm"].get("openai_model", "")),
                    "busy": bool(pipeline.current["pid"]), "queued": pipeline.jobs.qsize(),
                })
            if path == "/api/me":
                u = self.user()
                return self.send_json({"user": u, "admin": bool(u and auth.is_admin(u)), "setup": not auth.has_users()})
            u = self.user()
            if u and path == "/api/push/key":
                return self.send_json({"key": push.public_key(), "on": push.has(u)})
            if not u:
                return self.redirect("/login") if path == "/" else self.send_json({"error": "sign in required"}, 401)
            # signed in
            if path == "/":
                return self.send_file(os.path.join(STATIC, "index.html"))
            if path == "/api/projects":
                return self.send_json(pipeline.list_projects(u))
            if path == "/api/characters":
                return self.send_json(pipeline.list_characters(u))
            if m := re.fullmatch(r"/api/gallery/(images|videos)", path):
                return self.send_json(pipeline.gallery(m.group(1), u))
            if path == "/api/cloud":
                return self.send_json(cloudgpu.info())
            if path == "/api/cloud/options":  # the cards to pick from, with RunPod's live stock and prices
                return self.send_json(cloudgpu.options())
            if path == "/api/engines":
                return self.send_json(pipeline.engines())
            if path == "/api/users":
                if not auth.is_admin(u):
                    return self.send_json({"error": "admins only"}, 403)
                return self.send_json([{"user": k, "admin": v.get("admin", False)} for k, v in auth.users().items()])
            if m := re.fullmatch(r"/characters/([^/]+)", path):
                name = urllib.parse.unquote(m.group(1))
                if name not in pipeline.list_characters(u):
                    return self.send_json({"error": "not found"}, 404)
                return self.send_file(os.path.join(pipeline.char_dir(u), name))
            if m := re.fullmatch(r"/api/projects/([\w\-]+)", path):
                if not self.owns(u, m.group(1)):
                    return self.send_json({"error": "not found"}, 404)
                return self.send_json(pipeline.load(m.group(1)))
            if m := re.fullmatch(r"/files/([\w\-]+)/(.+)", path):
                if not self.owns(u, m.group(1)):
                    return self.send_json({"error": "not found"}, 404)
                root = os.path.normpath(pipeline.pdir(m.group(1)))
                full = os.path.normpath(pipeline.pdir(m.group(1), *m.group(2).split("/")))
                if not full.startswith(root + os.sep):  # the separator stops "al/../alice/..." passing as "al"
                    return self.send_json({"error": "bad path"}, 400)
                w = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query).get("w", [""])[0]
                if (w.isdigit() and int(w) in PREVIEW_W and os.path.isfile(full)
                        and full.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))):
                    full = preview(full, int(w))
                return self.send_file(full)
            self.send_json({"error": "not found"}, 404)
        except Exception as e:
            self.send_json({"error": str(e)}, 500)

    def do_POST(self):
        path = self.path.split("?")[0]
        try:
            # public: sign in, first-run setup, sign out
            if path in ("/api/login", "/api/setup"):
                b = self.body()
                if path == "/api/setup":
                    name = auth.create_admin(b.get("user"), b.get("password"))
                    pipeline.migrate_legacy(name)
                elif not auth.check(b.get("user"), b.get("password")):
                    return self.send_json({"error": "wrong username or password"}, 401)
                name = b["user"].strip().lower()
                return self.set_cookie_json({"ok": True, "user": name}, auth.new_session(name), auth.SESSION_DAYS * 86400)
            if path == "/api/logout":
                auth.end_session(self.token())
                return self.set_cookie_json({"ok": True}, "", 0)
            u = self.user()
            if not u:
                return self.send_json({"error": "sign in required"}, 401)
            if path == "/api/characters":  # raw image bytes, ?name=file.png
                q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
                data = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                if not data:
                    return self.send_json({"error": "empty upload"}, 400)
                return self.send_json({"name": pipeline.save_character(u, q.get("name", ["character.png"])[0], data)})
            b = self.body()
            if path == "/api/push/subscribe":
                push.subscribe(u, b.get("subscription"))
                return self.send_json({"ok": True})
            if path == "/api/push/unsubscribe":
                push.unsubscribe(u, b.get("endpoint", ""))
                return self.send_json({"ok": True})
            if path == "/api/push/test":
                push.send(u, "🔔 Reelmint", "Notifications are on: you'll hear when pictures and videos are ready.")
                return self.send_json({"ok": True})
            if path == "/api/password":
                if not auth.check(u, b.get("old")):
                    return self.send_json({"error": "current password is wrong"}, 400)
                auth.set_user(u, b.get("new"))
                return self.send_json({"ok": True})
            if path in ("/api/users", "/api/users/delete"):
                if not auth.is_admin(u):
                    return self.send_json({"error": "admins only"}, 403)
                if path == "/api/users":
                    return self.send_json({"user": auth.set_user(b.get("user"), b.get("password"), bool(b.get("admin")))})
                if b.get("user") == u:
                    return self.send_json({"error": "you can't delete yourself"}, 400)
                auth.delete_user(b.get("user"))
                return self.send_json({"ok": True})
            if path == "/api/enhance":
                text, engine = llm.enhance(pipeline.load_config()["llm"], b.get("idea", ""), b.get("mode", "video"),
                                           b.get("style", ""), b.get("character", ""),
                                           len(b.get("references") or []) or bool(b.get("reference")),
                                           b.get("keyframes"))
                return self.send_json({"text": text, "engine": engine})
            if path == "/api/characters/delete":
                if not pipeline.delete_character(u, b.get("name", "")):
                    return self.send_json({"error": "not found"}, 404)
                return self.send_json({"ok": True})
            if path in ("/api/cloud/start", "/api/cloud/stop"):  # rented GPU: costs money, so admins only
                if not auth.is_admin(u):
                    return self.send_json({"error": "admins only"}, 403)
                if path.endswith("start"):  # the card the person picked, community or secure cloud
                    cloudgpu.start(b.get("gpu", ""), b.get("cloud", ""))
                else:
                    cloudgpu.stop()
                return self.send_json(cloudgpu.info())
            if path == "/api/restore_image":
                return self.send_json(imageundo.restore(u, b.get("token")))
            if path == "/api/delete" and b.get("undo"):
                files = b.get("files", [])
                if b.get("projects") or len(files) != 1:
                    raise ValueError("Undo supports one image at a time")
                return self.send_json(imageundo.delete(u, files[0].get("project", ""), files[0].get("path", "")))
            if path == "/api/delete":
                return self.send_json({"deleted": pipeline.delete_items(u, b.get("projects", []), b.get("files", []))})
            if path == "/api/zip":  # several picked pictures/videos as one download (one save dialog, not one each)
                return self.send_zip(u, b.get("items", []))
            if path == "/api/plan":
                p = pipeline.create_project(u, b["idea"], b.get("keyframes", 4), b.get("style", ""),
                                            b.get("character", ""), b.get("aspect", "portrait"), b.get("seed"),
                                            b.get("references") or b.get("reference", ""), b.get("image_engine"), b.get("video_engine"),
                                            b.get("mode", "video"), b.get("quality"), b.get("video_res"), b.get("lora"))
                return self.send_json(p)
            m = re.fullmatch(r"/api/projects/([\w\-]+)/(\w+)", path)
            if not m or not self.owns(u, m.group(1)):
                return self.send_json({"error": "not found"}, 404)
            pid, action = m.groups()
            if action == "engines":
                return self.send_json(pipeline.set_engines(pid, b.get("image_engine"), b.get("video_engine"),
                                                           b.get("mode"), b.get("t2v_engine"), b.get("quality"),
                                                           b.get("video_res"), b.get("lora"), b.get("style"),
                                                           b.get("video_seconds")))
            if action == "reference":
                return self.send_json(pipeline.set_reference(pid, b.get("references") or b.get("reference", "")))
            if action == "pin":
                return self.send_json(pipeline.pin(pid, b.get("pinned")))
            if action == "rename":
                return self.send_json(pipeline.rename(pid, b.get("title", "")))
            if action == "plan":
                return self.send_json(pipeline.update_plan(pid, b["plan"]))
            if action == "keyframes":
                pipeline.enqueue(pid, "keyframes", indices=b.get("indices"), reseed=bool(b.get("reseed")),
                                 engine="gpt-image" if b.get("engine") == "gpt-image" else None,
                                 edit=str(b.get("edit") or "").strip() or None)
                return self.send_json({"ok": True})
            if action in ("animate", "t2v"):
                pipeline.enqueue(pid, action, only=b.get("only"))
                return self.send_json({"ok": True})
            if action == "stop":
                return self.send_json({"ok": pipeline.cancel(pid)})
            if action == "animate_image":
                pipeline.enqueue(pid, "animate_image", index=int(b.get("index", 0)), motion=b.get("motion", ""),
                                 seconds=b.get("seconds"), video_res=b.get("video_res"))
                return self.send_json({"ok": True})
            if action == "use_version":
                return self.send_json(pipeline.use_version(pid, b.get("index", 0), b.get("name", "")))
            if action == "stitch":
                pipeline.enqueue(pid, "stitch")
                return self.send_json({"ok": True})
            self.send_json({"error": "not found"}, 404)
        except ValueError as e:
            self.send_json({"error": str(e)}, 400)
        except Exception as e:
            self.send_json({"error": str(e)}, 500)


class Server(ThreadingHTTPServer):
    allow_reuse_address = False  # on Windows reuse lets a second copy share the port; it must fail instead


if __name__ == "__main__":
    cfg = pipeline.load_config()
    port = cfg.get("app_port", 7860)
    srv = Server(("127.0.0.1", port), Handler)  # raises "address in use" if another copy is running
    pipeline.start()
    print(f"Reelmint running at http://127.0.0.1:{port}")
    srv.serve_forever()
