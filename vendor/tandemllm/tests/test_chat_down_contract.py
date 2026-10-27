"""The 502 the Chat tab reads when chatd is down (VIS-179, QA VIS-180): every way the engine server's
forward (server/chat_forward.py) fails to reach chatd answers 502 with a JSON body whose
`error.type` is `chat_unavailable` and whose `message` is a string. The dashboard picks its state
from the status and that type (dashboard/src/chat/backend.ts classifyProbe), never from the text.

  1  no forward key anywhere                          test_no_forward_key
  2  nothing listens on QSE_CHAT_URL                  test_nothing_listens
  3  chatd accepts and sends nothing                  test_chatd_sends_nothing
  4  chatd resets the connection while the body goes  test_chatd_closes_the_connection

CPU only, standard library only. Run: python tests/test_chat_down_contract.py
"""

from __future__ import annotations

import http.client
import json
import os
import socket
import struct
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

from chatd_util import run_all  # noqa: E402

from server import chat_forward  # noqa: E402


def _forward(method: str, path: str, env: dict, body: bytes | None = None) -> tuple[int, dict]:
    keep = {k: os.environ.get(k) for k in ("QSE_CHAT_FORWARD_KEY", "QSE_CHAT_URL", "QSE_CHAT_DATA")}
    for k in keep:
        os.environ.pop(k, None)
    os.environ.update(env)

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def do_GET(self):
            chat_forward.route(self, "GET", None)

        def do_POST(self):
            chat_forward.route(self, "POST", None)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        if body is None:
            c = http.client.HTTPConnection("127.0.0.1", srv.server_address[1], timeout=20)
            c.request(method, path)
            r = c.getresponse()
            return r.status, json.loads(r.read())
        # a big body from a raw socket: the forward may answer before it has read all of it
        sock = socket.create_connection(srv.server_address, timeout=20)
        head = (f"{method} {path} HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
                f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n").encode()

        def send():
            try:
                sock.sendall(head + body)
            except OSError:
                pass
        threading.Thread(target=send, daemon=True).start()
        raw = b""
        while True:
            try:
                chunk = sock.recv(65536)
            except OSError:
                break
            if not chunk:
                break
            raw += chunk
            if b"\r\n\r\n" in raw:
                hdr, _, rest = raw.partition(b"\r\n\r\n")
                n = int([ln.split(b":")[1] for ln in hdr.split(b"\r\n") if ln.lower().startswith(b"content-length")][0])
                if len(rest) >= n:
                    break
        sock.close()
        hdr, _, rest = raw.partition(b"\r\n\r\n")
        return int(hdr.split(b" ")[1]), json.loads(rest)
    finally:
        srv.shutdown()
        srv.server_close()
        for k, v in keep.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v


def _assert_unavailable(st: int, obj: dict) -> str:
    assert st == 502, (st, obj)
    assert obj["error"]["type"] == "chat_unavailable", obj
    assert isinstance(obj["error"]["message"], str) and obj["error"]["message"], obj
    return obj["error"]["message"]


def _listener(handle) -> tuple[socket.socket, int]:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    s.listen(4)

    def run():
        try:
            c, _ = s.accept()
        except OSError:
            return
        handle(c)
    threading.Thread(target=run, daemon=True).start()
    return s, s.getsockname()[1]


def test_no_forward_key():
    with tempfile.TemporaryDirectory() as d:
        msg = _assert_unavailable(*_forward("GET", "/v1/chat-app/capabilities", {"QSE_CHAT_DATA": d}))
    assert "no forward key" in msg


def test_nothing_listens():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    dead = s.getsockname()[1]
    s.close()
    msg = _assert_unavailable(*_forward("GET", "/v1/chat-app/capabilities",
                                        {"QSE_CHAT_FORWARD_KEY": "k" * 64,
                                         "QSE_CHAT_URL": f"http://127.0.0.1:{dead}"}))
    assert msg == "the chat service is not running"


def test_chatd_sends_nothing():
    def handle(c):
        c.recv(65536)
        c.close()
    s, port = _listener(handle)
    try:
        msg = _assert_unavailable(*_forward("GET", "/v1/chat-app/capabilities",
                                            {"QSE_CHAT_FORWARD_KEY": "k" * 64,
                                             "QSE_CHAT_URL": f"http://127.0.0.1:{port}"}))
        assert msg == "the chat service sent nothing"
    finally:
        s.close()


def test_chatd_closes_the_connection():
    def handle(c):                       # a reset at once: the forward's send of the body fails
        c.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
        c.close()
    s, port = _listener(handle)
    try:
        st, obj = _forward("POST", "/v1/chat-app/conversations",
                           {"QSE_CHAT_FORWARD_KEY": "k" * 64, "QSE_CHAT_URL": f"http://127.0.0.1:{port}"},
                           body=b"x" * (8 << 20))
        msg = _assert_unavailable(st, obj)
        assert msg in ("the chat service closed the connection", "the chat service sent nothing"), msg
    finally:
        s.close()


if __name__ == "__main__":
    run_all(globals())
