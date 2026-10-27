"""`python -m chatd`: the chat service (see chatd/__init__.py).

    python -m chatd [--host 127.0.0.1] [--port 8016] [--data ~/.tandemllm/chat]
                    [--engine http://127.0.0.1:8000] [--max-runs 2] [--no-titles]
                    [--front 0.0.0.0:4180 --dashboard dashboard/dist]

`--front` also serves chatd/front.py from this process: the dashboard and the Chat tab in front of
an engine whose server has no forward (one process to start and stop when trying the chat).

Environment: QSE_CHAT_FORWARD_KEY (else `<data>/forward.key`, made on the first start, mode 600),
QSE_CHAT_ENGINE_KEY (a bearer token for the engine's API, when it wants one), QSE_CHAT_MAX_RUNS,
QSE_CHAT_RETRY_S. SIGTERM ends every run `interrupted` (Continue resumes it after the restart).
"""

from __future__ import annotations

import argparse
import os
import secrets
import signal
import sys
import threading
import time

from chatd import __version__
from chatd.http import make_server, serve_in_thread
from chatd.service import Service


def load_key(data_dir: str) -> str:
    k = (os.environ.get("QSE_CHAT_FORWARD_KEY") or "").strip()
    if k:
        return k
    path = os.path.join(data_dir, "forward.key")
    try:
        with open(path) as fh:
            k = fh.read().strip()
    except OSError:
        k = ""
    if len(k) < 32:
        k = secrets.token_hex(32)
        os.makedirs(data_dir, exist_ok=True)
        fd = os.open(path + ".tmp", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as fh:
            fh.write(k + "\n")
        os.replace(path + ".tmp", path)
    return k


def log(line: str) -> None:
    print(time.strftime("%Y-%m-%d %H:%M:%S ") + line, file=sys.stderr, flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m chatd", description=__doc__.split("\n")[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8016)
    ap.add_argument("--data", default=os.environ.get("QSE_CHAT_DATA") or "~/.tandemllm/chat")
    ap.add_argument("--engine", default=os.environ.get("QSE_CHAT_ENGINE") or
                    "http://127.0.0.1:8000")
    ap.add_argument("--max-runs", type=int, default=int(os.environ.get("QSE_CHAT_MAX_RUNS") or 2))
    ap.add_argument("--retry-s", type=float, default=float(os.environ.get("QSE_CHAT_RETRY_S")
                                                           or 600))
    ap.add_argument("--no-titles", action="store_true", help="keep the first line as the title "
                                                             "(no title call to the engine)")
    ap.add_argument("--front", default=None, help="HOST:PORT for chatd/front.py in this process")
    ap.add_argument("--dashboard", default=None, help="the built dashboard, for --front")
    a = ap.parse_args(argv)
    data = os.path.expanduser(a.data)
    os.makedirs(data, mode=0o700, exist_ok=True)
    key = load_key(data)
    svc = Service(data, a.engine, engine_key=os.environ.get("QSE_CHAT_ENGINE_KEY") or None,
                  max_runs=a.max_runs, retry_window_s=a.retry_s, titles=not a.no_titles, log=log)
    srv = make_server(svc, a.host, a.port, key)
    svc.start()
    serve_in_thread(srv)
    log(f"[chatd] {__version__} on {a.host}:{a.port}, data {data}, engine {a.engine}, "
        f"max runs {a.max_runs}")
    front = None
    if a.front:
        from http.server import ThreadingHTTPServer

        from chatd import front as front_mod
        fhost, _, fport = a.front.rpartition(":")
        handler = type("Front", (front_mod.FrontHandler,), {
            "dist": os.path.abspath(os.path.expanduser(a.dashboard or "dashboard/dist")),
            "engine": front_mod._addr(a.engine), "chatd": ("127.0.0.1", a.port), "key": key})
        front = ThreadingHTTPServer((fhost or "127.0.0.1", int(fport)), handler)
        front.daemon_threads = True
        threading.Thread(target=front.serve_forever, name="chatd-front", daemon=True).start()
        log(f"[front] http://{a.front}/dashboard/#/chat (the rest to {a.engine})")
    done = threading.Event()

    def stop(*_):
        done.set()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    done.wait()
    log("[chatd] stopping: open runs end interrupted")
    if front is not None:
        front.shutdown()
    srv.shutdown()
    svc.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
