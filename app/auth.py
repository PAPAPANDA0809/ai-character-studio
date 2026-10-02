"""Tiny local user accounts: PBKDF2 password hashes in users.json, session tokens in sessions.json."""
import hashlib
import hmac
import json
import os
import re
import secrets
import threading
import time

APP_DIR = os.path.dirname(os.path.abspath(__file__))
USERS = os.path.join(APP_DIR, "users.json")
SESSIONS = os.path.join(APP_DIR, "sessions.json")
SESSION_DAYS = 90
_lock = threading.Lock()


def _read(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _write(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def _hash(password, salt):
    return hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), 200_000).hex()


def valid_name(name):
    return bool(re.fullmatch(r"[a-z0-9_]{2,20}", name or ""))


def users():
    return _read(USERS)


def has_users():
    return bool(users())


def set_user(name, password, admin=False):
    name = (name or "").strip().lower()
    if not valid_name(name):
        raise ValueError("username must be 2-20 lowercase letters, digits or _")
    if len(password or "") < 6:
        raise ValueError("password must be at least 6 characters")
    with _lock:
        u = users()
        salt = secrets.token_hex(16)
        u[name] = {"salt": salt, "hash": _hash(password, salt), "admin": admin or u.get(name, {}).get("admin", False),
                   "created": u.get(name, {}).get("created", time.time())}
        _write(USERS, u)
    return name


def delete_user(name):
    with _lock:
        u = users()
        u.pop(name, None)
        _write(USERS, u)
        s = _read(SESSIONS)
        _write(SESSIONS, {t: v for t, v in s.items() if v["user"] != name})


def check(name, password):
    rec = users().get((name or "").strip().lower())
    return bool(rec) and hmac.compare_digest(rec["hash"], _hash(password or "", rec["salt"]))


def new_session(name):
    token = secrets.token_urlsafe(32)
    with _lock:
        s = _read(SESSIONS)
        now = time.time()
        s = {t: v for t, v in s.items() if v["expires"] > now}
        s[token] = {"user": name, "expires": now + SESSION_DAYS * 86400}
        _write(SESSIONS, s)
    return token


def end_session(token):
    with _lock:
        s = _read(SESSIONS)
        s.pop(token, None)
        _write(SESSIONS, s)


def user_for(token):
    if not token:
        return None
    v = _read(SESSIONS).get(token)
    if not v or v["expires"] < time.time() or v["user"] not in users():
        return None
    return v["user"]


def is_admin(name):
    return users().get(name, {}).get("admin", False)
