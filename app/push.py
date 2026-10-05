"""Phone notifications (Web Push): "Picture ready" when a job finishes, even with the browser closed.

The page (on the https address; browsers only allow push there) subscribes through static/sw.js and hands the
subscription to /api/push/subscribe. Each person's subscriptions are kept in push_subs.json; the VAPID key that signs
the messages is made on first use and kept in push_keys.pem (both are local secrets, not in git). Messages go through
the browser maker's push service (Google's for Chrome on Android) and only carry the title and the project link."""
import base64, json, os, threading

APP_DIR = os.path.dirname(os.path.abspath(__file__))
KEY_FILE = os.path.join(APP_DIR, "push_keys.pem")
SUBS_FILE = os.path.join(APP_DIR, "push_subs.json")
CLAIM = "https://desktop-h7p36q5.tail46ae30.ts.net"  # who sends (required by push services; no e-mail address)
lock = threading.Lock()


def _vapid():
    from py_vapid import Vapid02
    with lock:
        if not os.path.exists(KEY_FILE):
            v = Vapid02()
            v.generate_keys()
            v.save_key(KEY_FILE)
    return Vapid02.from_file(KEY_FILE)


def public_key():
    """The key the page gives pushManager.subscribe (base64url, uncompressed P-256 point)."""
    from cryptography.hazmat.primitives import serialization
    raw = _vapid().public_key.public_bytes(serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint)
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def _load():
    try:
        with open(SUBS_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _save(subs):
    tmp = SUBS_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(subs, f)
    os.replace(tmp, SUBS_FILE)


def subscribe(user, sub):
    if not isinstance(sub, dict) or not str(sub.get("endpoint", "")).startswith("https://") or "keys" not in sub:
        raise ValueError("not a push subscription")
    with lock:
        subs = _load()
        mine = [s for s in subs.get(user, []) if s["endpoint"] != sub["endpoint"]]
        subs[user] = mine + [{"endpoint": sub["endpoint"], "keys": sub["keys"]}]
        _save(subs)


def unsubscribe(user, endpoint):
    with lock:
        subs = _load()
        subs[user] = [s for s in subs.get(user, []) if s["endpoint"] != endpoint]
        _save(subs)


def has(user):
    return bool(_load().get(user))


def send(user, title, body, url="/", tag=None):
    """To every device the person turned notifications on for (in the background; failures are ignored, and a
    device that has unsubscribed or expired is forgotten)."""
    subs = _load().get(user, [])
    if subs:
        threading.Thread(target=_send, args=(user, subs, title, body, url, tag), daemon=True).start()


def _send(user, subs, title, body, url, tag):
    from pywebpush import webpush, WebPushException
    data = json.dumps({"title": title, "body": body, "url": url, "tag": tag or url})
    for s in subs:
        try:
            webpush(s, data, vapid_private_key=KEY_FILE, vapid_claims={"sub": CLAIM}, ttl=3600, timeout=20)
        except WebPushException as e:
            if e.response is not None and e.response.status_code in (404, 410):
                unsubscribe(user, s["endpoint"])
        except Exception:
            pass


def job_done(user, pid, title, task, status, detail=""):
    """The worker's report: what finished in which project (stopped jobs aren't announced)."""
    what = {"keyframes": "Pictures ready", "animate": "Video clips ready", "animate_image": "Video ready",
            "t2v": "Video ready", "stitch": "Final video ready"}.get(task, "Done")
    if status == "error":
        send(user, f"⚠ {title}", f"Something went wrong: {detail}"[:180], f"/#project/{pid}")
    elif status == "done":
        send(user, f"✅ {what}", title, f"/#project/{pid}")
