"""Download the models listed in downloads.txt (dest|url), resuming partial files, then verify SHA-256
against Hugging Face's LFS hashes. Re-downloads anything that fails the check.

Usage: python download_models.py [list-file]   (only one copy may run at a time)
"""
import concurrent.futures as cf
import hashlib
import json
import os
import re
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
LOCK = os.path.join(HERE, ".download.lock")


def hf_meta(url):
    """(size, sha256) for a huggingface.co resolve URL, else (None, None)."""
    m = re.match(r"https://huggingface\.co/([^/]+/[^/]+)/resolve/([^/]+)/(.+)", url)
    if not m:
        return None, None
    repo, rev, path = m.groups()
    folder = os.path.dirname(path)
    api = f"https://huggingface.co/api/models/{repo}/tree/{rev}/{folder}".rstrip("/")
    with urllib.request.urlopen(api, timeout=60) as r:
        for f in json.load(r):
            if f.get("path") == path:
                lfs = f.get("lfs") or {}
                return f.get("size"), lfs.get("oid")
    return None, None


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(dest, url, size):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    for attempt in range(200):
        have = os.path.getsize(dest) if os.path.exists(dest) else 0
        if size and have == size:
            return
        if size and have > size:
            os.remove(dest)
            have = 0
        req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
        try:
            with urllib.request.urlopen(req, timeout=60) as r, open(dest, "ab" if have else "wb") as f:
                if have and r.status != 206:  # server ignored Range: start over
                    f.truncate(0)
                while chunk := r.read(1 << 20):
                    f.write(chunk)
        except Exception as e:
            print(f"  retry {os.path.basename(dest)}: {e}", flush=True)
            time.sleep(min(30, 2 + attempt))
        if not size:
            return


def one(line):
    dest, url = line.split("|", 1)
    size, oid = hf_meta(url)
    for _ in range(3):
        fetch(dest, url, size)
        if not oid or sha256(dest) == oid:
            print(f"OK {dest}", flush=True)
            return True
        print(f"  checksum mismatch, re-downloading {dest}", flush=True)
        os.remove(dest)
    print(f"FAIL {dest}", flush=True)
    return False


def main():
    if os.path.exists(LOCK) and time.time() - os.path.getmtime(LOCK) < 6 * 3600:
        sys.exit("another download is already running (delete setup/.download.lock if not)")
    open(LOCK, "w").write(str(os.getpid()))
    try:
        lst = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "downloads.txt")
        lines = [l.strip() for l in open(lst, encoding="utf-8") if "|" in l]
        with cf.ThreadPoolExecutor(3) as ex:
            results = list(ex.map(one, lines))
        print("ALLDONE", "ok" if all(results) else "with failures", flush=True)
    finally:
        os.remove(LOCK)


if __name__ == "__main__":
    main()
