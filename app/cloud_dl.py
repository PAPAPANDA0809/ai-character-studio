"""Model downloader for the rented pod (cloud_setup.sh runs it): python3 cloud_dl.py <url> <dest>.
One connection per file was slow on some hosts (~45 MB/s) and a stalled connection hung the setup for good (curl
waited forever at 99 %). Here a file comes in 8 pieces at once over separate connections; a piece that stalls for
30 s or fails is fetched again (up to 12 times, re-resolving Hugging Face's signed download link if it expired).
The file is written as <dest>.part and renamed only when every byte is in."""
import os, queue, sys, threading, time, urllib.error, urllib.request

PIECE = 64 << 20
WORKERS = 8
TIMEOUT = 30


def resolve(url):
    """(final url, size): Hugging Face answers with a redirect to a signed CDN link."""
    req = urllib.request.Request(url, headers={"Range": "bytes=0-0", "User-Agent": "reelmint"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        total = r.headers.get("Content-Range", "").rsplit("/", 1)[-1]
        return r.url, int(total) if total.isdigit() else int(r.headers["Content-Length"])


def main(url, dest):
    final, size = resolve(url)
    part = dest + ".part"
    with open(part, "wb") as f:
        f.truncate(size)
    pieces = queue.Queue()
    for start in range(0, size, PIECE):
        pieces.put((start, min(start + PIECE, size) - 1, 0))
    link = [final]
    failed = []

    def work():
        with open(part, "r+b") as f:
            while not failed:
                try:
                    start, end, tries = pieces.get_nowait()
                except queue.Empty:
                    return
                try:
                    req = urllib.request.Request(link[0], headers={"Range": f"bytes={start}-{end}", "User-Agent": "reelmint"})
                    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                        f.seek(start)
                        got = 0
                        while True:
                            chunk = r.read(1 << 20)
                            if not chunk:
                                break
                            f.write(chunk)
                            got += len(chunk)
                    if got != end - start + 1:
                        raise IOError(f"short piece {got} of {end - start + 1}")
                except Exception as e:
                    if tries >= 12:
                        failed.append(str(e))
                        return
                    if isinstance(e, urllib.error.HTTPError) and e.code in (401, 403):
                        try:
                            link[0] = resolve(url)[0]  # the signed link expired
                        except Exception:
                            pass
                    time.sleep(min(30, 2 ** tries))
                    pieces.put((start, end, tries + 1))

    threads = [threading.Thread(target=work) for _ in range(WORKERS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if failed or not pieces.empty():
        raise SystemExit(f"download failed: {failed[:1]}")
    os.replace(part, dest)


if __name__ == "__main__":
    for attempt in range(3):
        try:
            main(sys.argv[1], sys.argv[2])
            break
        except SystemExit:
            raise
        except Exception as e:  # resolving failed: try again a little later
            if attempt == 2:
                raise SystemExit(f"download failed: {e}")
            time.sleep(10)
