"""Keep the web server alive under Windows Task Scheduler, independently of chat apps."""
import ctypes
from ctypes import wintypes
import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import socket
import subprocess
import sys
import time


def main():
    root = Path(__file__).resolve().parent.parent
    app = root / "app"
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateMutexW.argtypes = (ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR)
    kernel.CreateMutexW.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    mutex = kernel.CreateMutexW(None, False, r"Local\ReelmintServerSupervisor")
    if not mutex:
        raise ctypes.WinError(ctypes.get_last_error())
    if ctypes.get_last_error() == 183:
        kernel.CloseHandle(mutex)
        return
    logger = logging.getLogger("reelmint-supervisor")
    logger.setLevel(logging.INFO)
    handler = RotatingFileHandler(app / "reelmint-supervisor.log", maxBytes=1_000_000,
                                  backupCount=0, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logger.addHandler(handler)
    python = root / "ComfyUI_windows_portable" / "python_embeded" / "python.exe"
    backoff = 5
    waiting = False
    logger.info("Windows background launcher started (PID %s)", __import__("os").getpid())
    try:
        while True:
            try:
                config = json.loads((app / "config.json").read_text(encoding="utf-8"))
                port = int(config.get("app_port", 7860))
                if not 1 <= port <= 65535:
                    raise ValueError("Invalid app port")
                try:
                    connection = socket.create_connection(("127.0.0.1", port), timeout=2)
                except OSError:
                    connection = None
                if connection is not None:
                    connection.close()
                    if not waiting:
                        logger.info("Port %s already serves the app; waiting without starting a duplicate", port)
                    waiting = True
                    time.sleep(5)
                    continue
                waiting = False
                output = app / "reelmint-server.log"
                if output.exists() and output.stat().st_size > 5_000_000:
                    output.write_bytes(b"")
                with output.open("ab", buffering=0) as log:
                    child = subprocess.Popen([str(python), "-u", "server.py"], cwd=app,
                                             stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                             creationflags=subprocess.CREATE_NO_WINDOW)
                    logger.info("Started Reelmint server (PID %s)", child.pid)
                    started = time.monotonic()
                    code = child.wait()
                elapsed = time.monotonic() - started
                backoff = 5 if elapsed > 60 else min(60, backoff * 2)
                logger.warning("Server exited with code %s after %.0f seconds; retrying in %s seconds",
                               code, elapsed, backoff)
            except Exception:
                logger.exception("Could not start the web server; retrying in 60 seconds")
                backoff = 60
            time.sleep(backoff)
    finally:
        kernel.CloseHandle(mutex)


if __name__ == "__main__":
    main()
