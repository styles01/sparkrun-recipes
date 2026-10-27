"""The engine server's forward of `/v1/chat-app/*` to chatd (SRV-343, server/chat_forward.py).

The forward runs here inside a small stdlib handler with the engine's own `Auth` (server/auth.py)
and a `_session` that stands in for the engine's login, in front of a real chatd on a free port.
The matrix: no password; password with the session cookie; with the bearer token; with neither
(401, and nothing reaches chatd); trusted-local; the probe that always passes; a forged forward
header; a cross-site write; chatd down (502). Then the SSE latency through the forward, a
WebSocket splice of 10 MB each way, an idle WebSocket past the handler's timeout, a body that
stalls past it (408), and the cap on the sign-in body.

CPU only. Run: python -m pytest -q tests/test_chat_forward.py
"""

from __future__ import annotations

import http.client
import json
import os
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import fake_backend
from chatd_util import KEY, Chat, run_all

from server import auth as auth_mod
from server import chat_forward

ADMIN = "a" * 40


def _engine(auth, chatd_port: int | None, idle_timeout: float | None = None):
    """A stand-in for server/app.py's Handler: the chat route first, as do_GET/do_POST do.
    `idle_timeout`: the handler's socket timeout, as the proxy's Handler.timeout."""
    os.environ["QSE_CHAT_FORWARD_KEY"] = KEY
    os.environ["QSE_CHAT_URL"] = f"http://127.0.0.1:{chatd_port or 1}"
    seen = {"session": []}

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        timeout = idle_timeout
        _status = None

        def log_message(self, *a):
            pass

        def _session(self, method, body=None):
            seen["session"].append((method, body))
            self.send_response(204)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def _go(self, m):
            if not chat_forward.route(self, m, auth):
                self.send_response(404)
                self.send_header("Content-Length", "0")
                self.end_headers()

        def do_GET(self):
            self._go("GET")

        def do_POST(self):
            self._go("POST")

        def do_PATCH(self):
            self._go("PATCH")

        def do_DELETE(self):
            self._go("DELETE")

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, seen


def _req(port, method, path, body=None, headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=20)
    h = {"Content-Type": "application/json", **(headers or {})}
    conn.request(method, path, body=json.dumps(body).encode() if body is not None else None,
                 headers=h)
    r = conn.getresponse()
    raw = r.read()
    conn.close()
    try:
        return r.status, json.loads(raw) if raw else None
    except ValueError:
        return r.status, raw


PROXIED = {"X-Forwarded-For": "192.168.1.20", "X-Real-IP": "192.168.1.20"}


def test_the_auth_matrix():
    c = Chat()
    try:
        # no password set (AU1): every chat route answers
        srv, _ = _engine(auth_mod.Auth(None), c.port)
        p = srv.server_address[1]
        st, obj = _req(p, "GET", "/v1/chat-app/capabilities", headers=PROXIED)
        assert st == 200 and obj["session"] == {"password_set": False, "signed_in": True,
                                                "sandbox": "not_installed"}
        st, obj = _req(p, "GET", "/v1/chat-app/conversations", headers=PROXIED)
        assert st == 200 and obj["conversations"] == []
        st, _ = _req(p, "GET", "/v1/chat/completions", headers=PROXIED)
        assert st == 404                          # the OpenAI namespace is never forwarded
        srv.shutdown()

        a = auth_mod.Auth(ADMIN, login=True)
        srv, seen = _engine(a, c.port)
        p = srv.server_address[1]
        before = c.svc.store.one("SELECT COUNT(*) FROM conversations")[0]
        st, obj = _req(p, "POST", "/v1/chat-app/conversations", {}, headers=PROXIED)
        assert st == 401 and obj["error"]["type"] == "unauthorized"
        assert c.svc.store.one("SELECT COUNT(*) FROM conversations")[0] == before
        # the probe passes and says who is not signed in
        st, obj = _req(p, "GET", "/v1/chat-app/capabilities", headers=PROXIED)
        assert st == 200 and obj["session"]["password_set"] is True
        assert obj["session"]["signed_in"] is False
        # the bearer token
        st, obj = _req(p, "GET", "/v1/chat-app/capabilities",
                       headers={**PROXIED, "Authorization": f"Bearer {ADMIN}"})
        assert obj["session"]["signed_in"] is True
        st, obj = _req(p, "POST", "/v1/chat-app/conversations", {},
                       headers={**PROXIED, "Authorization": f"Bearer {ADMIN}"})
        assert st == 200 and obj["id"]
        # the dashboard's session cookie
        value, _ = a.make_cookie()
        st, obj = _req(p, "GET", "/v1/chat-app/conversations",
                       headers={**PROXIED, "Cookie": f"{auth_mod.COOKIE}={value}"})
        assert st == 200 and len(obj["conversations"]) == 1
        # trusted-local: loopback with no forwarding headers (the box's own scripts)
        st, _ = _req(p, "GET", "/v1/chat-app/conversations")
        assert st == 200
        # a forged forward header is dropped; the engine adds its own
        st, _ = _req(p, "GET", "/v1/chat-app/conversations",
                     headers={**PROXIED, "X-Tandem-Forward": KEY, "X-Tandem-Signed-In": "1"})
        assert st == 401
        # POST /session is the dashboard's own login, with the chat's field name
        st, _ = _req(p, "POST", "/v1/chat-app/session", {"password": "pw"}, headers=PROXIED)
        assert st == 204 and seen["session"][-1] == ("POST", {"token": "pw"})
        st, _ = _req(p, "DELETE", "/v1/chat-app/session", headers=PROXIED)
        assert seen["session"][-1] == ("DELETE", None)
        # a cross-site write is refused even when signed in
        st, obj = _req(p, "POST", "/v1/chat-app/conversations", {},
                       headers={**PROXIED, "Authorization": f"Bearer {ADMIN}",
                                "Sec-Fetch-Site": "cross-site"})
        assert st == 403
        srv.shutdown()
    finally:
        c.close()


def test_chatd_down_is_a_json_502():
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    dead = sock.getsockname()[1]
    sock.close()
    srv, _ = _engine(auth_mod.Auth(None), dead)
    try:
        st, obj = _req(srv.server_address[1], "GET", "/v1/chat-app/conversations")
        assert st == 502 and obj["error"]["type"] == "chat_unavailable"
    finally:
        srv.shutdown()


def test_sse_through_the_forward_is_unbuffered():
    """Each event reaches the client within 100 ms of chatd emitting it."""
    c = Chat()
    srv, _ = _engine(auth_mod.Auth(None), c.port)
    try:
        cid = c.new()
        conv = c.svc.conv(cid)
        conn = http.client.HTTPConnection("127.0.0.1", srv.server_address[1], timeout=10)
        conn.request("GET", f"/v1/chat-app/conversations/{cid}/events",
                     headers={"Accept": "text/event-stream"})
        r = conn.getresponse()
        assert r.status == 200 and r.getheader("Content-Type").startswith("text/event-stream")
        assert r.readline().startswith(b"event: current")
        r.readline(), r.readline()

        def next_id():
            # the next stored event; an ephemeral context.changed may come between
            while True:
                block = [r.readline()]
                while block[-1] not in (b"\n", b""):
                    block.append(r.readline())
                if block[0].startswith(b"id: "):
                    return block[0]
        lat = []
        for i in range(20):
            t0 = time.monotonic()
            seq = c.svc.emit(conv, {"type": "queue.changed", "queue": []})
            line = next_id()
            lat.append(time.monotonic() - t0)
            assert line == f"id: {seq}\n".encode(), line
            time.sleep(0.02)
        conn.close()
        assert max(lat) < 0.1, lat
    finally:
        srv.shutdown()
        c.close()


def _echo_upstream() -> tuple[socket.socket, dict]:
    """A chatd stand-in that answers 101 and echoes every byte; its listening socket and the
    request head it got."""
    up = socket.socket()
    up.bind(("127.0.0.1", 0))
    up.listen(1)
    got_head = {}

    def upstream():
        s, _ = up.accept()
        head = b""
        while b"\r\n\r\n" not in head:
            head += s.recv(4096)
        got_head["raw"] = head
        s.sendall(b"HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\n"
                  b"Connection: Upgrade\r\n\r\n")
        while True:
            d = s.recv(65536)
            if not d:
                break
            s.sendall(d)
        s.close()
    threading.Thread(target=upstream, daemon=True).start()
    return up, got_head


def _upgrade(port: int) -> tuple[socket.socket, bytes]:
    cl = socket.create_connection(("127.0.0.1", port))
    cl.sendall(b"GET /v1/chat-app/conversations/c1/terminals/shell/ws HTTP/1.1\r\n"
               b"Host: x\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
               b"Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\nSec-WebSocket-Version: 13"
               b"\r\n\r\n")
    head = b""
    while b"\r\n\r\n" not in head:
        head += cl.recv(4096)
    assert head.startswith(b"HTTP/1.1 101"), head
    return cl, head


def test_a_websocket_is_spliced_both_ways_byte_for_byte():
    """chatd's 101 goes back, then 10 MB each way arrive unchanged (an echo upstream)."""
    up, got_head = _echo_upstream()
    srv, _ = _engine(auth_mod.Auth(None), up.getsockname()[1])
    try:
        cl, head = _upgrade(srv.server_address[1])
        assert b"X-Tandem-Forward: " + KEY.encode() in got_head["raw"]
        assert b"Upgrade: websocket" in got_head["raw"]
        payload = os.urandom(10 * 1024 * 1024)
        back = bytearray(head.split(b"\r\n\r\n", 1)[1])

        def reader():
            while len(back) < len(payload):
                d = cl.recv(1 << 20)
                if not d:
                    break
                back.extend(d)
        t = threading.Thread(target=reader)
        t.start()
        cl.sendall(payload)
        t.join(30)
        assert bytes(back) == payload
        cl.close()
    finally:
        srv.shutdown()
        up.close()


def test_an_idle_websocket_outlives_the_handler_timeout():
    """An open terminal sends nothing while idle and chatd never pings: the splice lifts the
    front server's idle timeout, so the socket is still spliced long after it."""
    up, _ = _echo_upstream()
    srv, _ = _engine(auth_mod.Auth(None), up.getsockname()[1], idle_timeout=0.3)
    try:
        cl, _ = _upgrade(srv.server_address[1])
        time.sleep(1.2)
        cl.settimeout(5)
        cl.sendall(b"still here")
        got = b""
        while len(got) < 10:
            d = cl.recv(64)
            assert d, "the idle WebSocket was closed"
            got += d
        assert got == b"still here"
        cl.close()
    finally:
        srv.shutdown()
        up.close()


def test_a_client_that_stalls_inside_its_body_gets_a_408_and_the_connection_closes():
    """The front server's request timeout fires inside the forward's body loop: that is the
    client's doing, not chatd's (no 502), and the unread rest of the body must not be parsed as
    the next request."""
    up = socket.socket()
    up.bind(("127.0.0.1", 0))
    up.listen(1)

    def drain():
        s, _ = up.accept()
        while s.recv(65536):
            pass
        s.close()
    threading.Thread(target=drain, daemon=True).start()
    srv, _ = _engine(auth_mod.Auth(None), up.getsockname()[1], idle_timeout=0.3)
    errors = []
    srv.handle_error = lambda request, addr: errors.append(addr)
    try:
        t0 = time.monotonic()
        out = _raw(srv.server_address[1], b"POST /v1/chat-app/conversations HTTP/1.1\r\n"
                                         b"Host: x\r\nContent-Length: 100000\r\n\r\n"
                                         + b"a" * 1000)
        assert out.startswith(b"HTTP/1.1 408") and b"request_timeout" in out, out[:200]
        assert time.monotonic() - t0 < 5
        assert errors == [], "the next readline must not run on the unread body"
    finally:
        srv.shutdown()
        up.close()


def _raw(port: int, head: bytes) -> bytes:
    cl = socket.create_connection(("127.0.0.1", port), timeout=10)
    cl.sendall(head)
    out = b""
    while True:
        d = cl.recv(65536)
        if not d:
            break
        out += d
        if b"\r\n\r\n" in out:
            h, _, body = out.partition(b"\r\n\r\n")
            n = [int(ln.split(b":")[1]) for ln in h.lower().split(b"\r\n")
                 if ln.startswith(b"content-length:")]
            if n and len(body) >= n[0]:
                break
    cl.close()
    return out


def test_the_sign_in_body_is_bounded():
    """`POST /session` reads its body before any sign-in: a length past the cap is 413 and is
    never read, a length that is not a number is 400; the login is not called."""
    srv, seen = _engine(auth_mod.Auth(ADMIN, login=True), 1)
    p = srv.server_address[1]
    try:
        # the bodies are never sent: a refusal answers at once, without waiting for them
        # (and an unread body in the server's buffer would reset the connection)
        for n in (chat_forward.SESSION_BODY_MAX + 1, 1 << 30):
            t0 = time.monotonic()
            out = _raw(p, b"POST /v1/chat-app/session HTTP/1.1\r\nHost: x\r\n"
                          b"Content-Length: %d\r\n\r\n" % n)
            assert out.startswith(b"HTTP/1.1 413") and b"too_large" in out, out[:200]
            assert time.monotonic() - t0 < 5
        for bad in (b"-5", b"abc"):
            out = _raw(p, b"POST /v1/chat-app/session HTTP/1.1\r\nHost: x\r\n"
                          b"Content-Length: " + bad + b"\r\n\r\n")
            assert out.startswith(b"HTTP/1.1 400"), (bad, out[:200])
        assert seen["session"] == []
        st, _ = _req(p, "POST", "/v1/chat-app/session", {"password": "pw"}, headers=PROXIED)
        assert st == 204 and seen["session"] == [("POST", {"token": "pw"})]
    finally:
        srv.shutdown()


def test_the_front_serves_the_dashboard_and_forwards_both_ways():
    """chatd/front.py: /dashboard/ from a folder, /v1/chat-app to chatd, the rest to the engine."""
    import tempfile

    from chatd import front
    fk = fake_backend.start(models=["fake-a"])
    c = Chat(fake=fk)
    dist = tempfile.mkdtemp()
    with open(os.path.join(dist, "index.html"), "w") as fh:
        fh.write("<!doctype html><title>dash</title>")
    handler = type("F", (front.FrontHandler,), {"dist": dist, "engine": ("127.0.0.1", fk[1]),
                                                "chatd": ("127.0.0.1", c.port), "key": KEY})
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    p = srv.server_address[1]
    try:
        st, body = _req(p, "GET", "/dashboard/")
        assert st == 200 and b"<title>dash</title>" in body
        st, obj = _req(p, "GET", "/v1/chat-app/capabilities")
        assert st == 200 and obj["models"][0]["id"] == "fake-a"
        st, obj = _req(p, "GET", "/v1/models")
        assert st == 200 and obj["data"][0]["id"] == "fake-a"
    finally:
        srv.shutdown()
        c.close()
        fk[0].shutdown()


def test_a_browser_that_leaves_while_chatd_holds_the_answer_closes_the_forward():
    """#86: a long-poll or a quiet event stream through the port preview. The forward waited in
    recv for chatd and noticed a browser that left only on its next write, so chatd (and the
    relay in the sandbox) kept serving nobody. The browser's end now closes the socket to chatd."""
    up = socket.socket()
    up.bind(("127.0.0.1", 0))
    up.listen(4)
    seen = {}

    def upstream(stream: bool):
        s, _ = up.accept()
        head = b""
        while b"\r\n\r\n" not in head:
            head += s.recv(4096)
        if stream:                       # a stream that has said one thing and is quiet now
            s.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\ndata: 0\n\n")
        s.settimeout(10)
        try:
            seen[stream] = s.recv(1) == b"" and time.monotonic()
        except OSError:
            seen[stream] = None
        s.close()
    srv, _ = _engine(auth_mod.Auth(None), up.getsockname()[1])
    try:
        for stream in (False, True):
            threading.Thread(target=upstream, args=(stream,), daemon=True).start()
            cl = socket.create_connection(("127.0.0.1", srv.server_address[1]))
            cl.sendall(b"GET /v1/chat-app/conversations/c1/sandbox/ports/3000/poll HTTP/1.1\r\n"
                       b"Host: x\r\n\r\n")
            if stream:
                cl.settimeout(5)
                got = b""
                while b"data: 0" not in got:
                    got += cl.recv(4096)
            time.sleep(0.3)
            t0 = time.monotonic()
            cl.close()
            for _ in range(100):
                if stream in seen:
                    break
                time.sleep(0.02)
            assert seen.get(stream) and seen[stream] - t0 < 1.0, (stream, seen)
    finally:
        srv.shutdown()
        up.close()


if __name__ == "__main__":
    run_all(globals())
