"""`/v1/chat-app/*`: the Chat tab's routes, checked here and forwarded to chatd (SRV-343).

The browser talks to one origin; chatd (python -m chatd, 127.0.0.1:8016) holds the chats. This
module is the engine server's side of that: the auth check, then a byte forward.

    auth        the dashboard's one switch (server/auth.py, AU1): no admin password set, every chat
                route answers; a password set, the `qse_dash` session or the admin bearer token, or
                trusted-local. `GET /capabilities` and `GET /session` always pass (the dashboard's
                probe must get JSON, and the capabilities say `signed_in: false` so the tab shows
                its sign-in); `POST /session {password}` signs in with the dashboard's own login
                (`DELETE` signs out). A write with `Sec-Fetch-Site: cross-site` is refused.
    forward     the request line, the headers (hop-by-hop ones and any `X-Tandem-*` a client sent
                dropped) and the body, to chatd with `X-Tandem-Forward: <key>`,
                `X-Tandem-Principal: owner`, `X-Tandem-Password-Set` and `X-Tandem-Signed-In`;
                the answer goes back as chatd wrote it, unbuffered (an SSE frame reaches the
                browser as soon as chatd writes it); while it waits for chatd, a browser that
                closes its connection closes the one to chatd (a held long-poll or a quiet
                stream ends in chatd and in the sandbox at once, #86)
    websocket   `Upgrade: websocket` gets chatd's 101 and the two sockets are spliced, one thread
                each way, until either side closes
    chatd down  502 `chat_unavailable` with a JSON body; the engine's other routes are untouched

The key: QSE_CHAT_FORWARD_KEY (secrets.env, ops/make-secrets.sh), else the file chatd writes on its
first start, `<QSE_CHAT_DATA or ~/.tandemllm/chat>/forward.key`. chatd: QSE_CHAT_URL, default
http://127.0.0.1:8016. This module imports nothing of the engine; the standalone front
(`python -m chatd.front`) uses it the same way.
"""

from __future__ import annotations

import json
import os
import select
import socket
import threading
import urllib.parse

PREFIX = "/v1/chat-app"
OPEN_GETS = (PREFIX + "/capabilities", PREFIX + "/session")
CONNECT_TIMEOUT_S = 3.0
CHUNK = 65536
SESSION_BODY_MAX = 4096          # `POST /session {password}`, read here before any sign-in
# never passed on: hop-by-hop headers, and what only this side may say
_DROP = {"connection", "keep-alive", "proxy-connection", "proxy-authorization", "te", "trailer",
         "transfer-encoding", "upgrade", "host", "x-tandem-forward", "x-tandem-principal",
         "x-tandem-password-set", "x-tandem-signed-in"}


def wanted(path: str) -> bool:
    p = path.split("?")[0]
    return p == PREFIX or p.startswith(PREFIX + "/")


def data_dir(env=None) -> str:
    env = os.environ if env is None else env
    return os.path.expanduser(env.get("QSE_CHAT_DATA") or "~/.tandemllm/chat")


def forward_key(env=None) -> str | None:
    env = os.environ if env is None else env
    k = (env.get("QSE_CHAT_FORWARD_KEY") or "").strip()
    if k:
        return k
    try:
        with open(os.path.join(data_dir(env), "forward.key")) as fh:
            return fh.read().strip() or None
    except OSError:
        return None


def target(env=None) -> tuple[str, int]:
    env = os.environ if env is None else env
    u = urllib.parse.urlsplit(env.get("QSE_CHAT_URL") or "http://127.0.0.1:8016")
    return u.hostname or "127.0.0.1", u.port or 8016


def _json(handler, code: int, kind: str, message: str) -> None:
    raw = json.dumps({"error": {"type": kind, "message": message}}).encode()
    try:
        handler.send_response(code)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Cache-Control", "no-store")
        handler.send_header("Content-Length", str(len(raw)))
        handler.end_headers()
        handler.wfile.write(raw)
    except (BrokenPipeError, ConnectionResetError):
        handler.close_connection = True


def route(handler, method: str, auth, *, peer: str | None = None) -> bool:
    """Answer `handler`'s request when its path is a chat route; False for any other path.
    `auth`: server/auth.py's Auth (None: no auth at all, e.g. the front in a test)."""
    path = handler.path.split("?")[0]
    if not wanted(path):
        return False
    path = path.rstrip("/")
    password_set = bool(auth is not None and auth.password_set)
    peer = peer if peer is not None else (handler.client_address or ("",))[0]
    signed_in = (not password_set or auth.admin_any(handler.headers)
                 or auth.trusted_local(peer, handler.headers))
    if path == PREFIX + "/session" and method in ("POST", "DELETE"):
        # the dashboard's own login (`POST /v1/dashboard/session {token}`), under the chat's name
        if not hasattr(handler, "_session"):
            _json(handler, 404, "not_found", "no admin password is set")
            return True
        if method == "DELETE":
            handler._session("DELETE")
            return True
        raw = (handler.headers.get("Content-Length") or "0").strip()
        if not (raw.isascii() and raw.isdigit()):
            handler.close_connection = True
            _json(handler, 400, "bad_request", "Content-Length is not a length")
            return True
        n = int(raw)
        if n > SESSION_BODY_MAX:
            # the body is not read: the connection cannot be reused
            handler.close_connection = True
            _json(handler, 413, "too_large", f"a sign-in body is at most {SESSION_BODY_MAX} "
                                             "bytes")
            return True
        try:
            body = json.loads(handler.rfile.read(n) or b"{}") if n else {}
        except ValueError:
            body = {}
        pw = body.get("password") if isinstance(body, dict) else None
        handler._session("POST", {"token": pw})
        return True
    if not signed_in and not (method == "GET" and path in OPEN_GETS):
        _json(handler, 401, "unauthorized", "sign in to the dashboard first")
        return True
    if method not in ("GET", "HEAD") and \
            (handler.headers.get("Sec-Fetch-Site") or "").lower() == "cross-site":
        _json(handler, 403, "forbidden", "a cross-site write is refused")
        return True
    key = forward_key()
    if not key:
        _json(handler, 502, "chat_unavailable", "chatd has not started on this box yet (no "
                                                "forward key)")
        return True
    forward(handler, method, [("X-Tandem-Forward", key), ("X-Tandem-Principal", "owner"),
                              ("X-Tandem-Password-Set", "1" if password_set else "0"),
                              ("X-Tandem-Signed-In", "1" if signed_in else "0")])
    return True


def forward(handler, method: str, extra: list[tuple[str, str]],
            addr: tuple[str, int] | None = None) -> None:
    """The request to `addr` (chatd by default) with the `extra` headers, the answer back as it
    comes. Also the front's way to the engine (chatd/front.py), with no extra headers."""
    host, port = addr or target()
    try:
        sock = socket.create_connection((host, port), timeout=CONNECT_TIMEOUT_S)
    except OSError:
        return _json(handler, 502, "chat_unavailable", "the chat service is not running")
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.settimeout(None)
    h = handler.headers
    upgrade = (h.get("Upgrade") or "").lower() == "websocket"
    lines = [f"{method} {handler.path} HTTP/1.1", f"Host: {host}:{port}"]
    for k, v in h.items():
        if k.lower() not in _DROP:
            lines.append(f"{k}: {v}")
    if upgrade:
        lines += ["Upgrade: websocket", "Connection: Upgrade"]
    else:
        lines.append("Connection: close")
    lines += [f"{k}: {v}" for k, v in extra]
    if (h.get("Transfer-Encoding") or "").lower() == "chunked":
        sock.close()
        return _json(handler, 411, "length_required", "send a Content-Length")
    try:
        sock.sendall(("\r\n".join(lines) + "\r\n\r\n").encode("latin-1"))
        left = int(h.get("Content-Length") or 0)
        while left > 0:
            chunk = handler.rfile.read(min(CHUNK, left))
            if not chunk:
                break
            sock.sendall(chunk)
            left -= len(chunk)
    except TimeoutError:
        # the client stalled inside its body (a front server's request timeout): not chatd's
        # doing, and the rest of the body would be read as the next request
        sock.close()
        handler.close_connection = True
        return _json(handler, 408, "request_timeout", "the request body stopped arriving")
    except OSError:
        sock.close()
        handler.close_connection = True                  # the body is left unread
        return _json(handler, 502, "chat_unavailable", "the chat service closed the connection")
    try:
        _relay(handler, sock, upgrade)
    finally:
        try:
            sock.close()
        except OSError:
            pass


class _BrowserGone(ConnectionAbortedError):
    """The browser closed its connection while the forward waited for chatd."""


def _recv(sock: socket.socket, conn) -> bytes:
    """chatd's next bytes. While none come, the browser's socket (its request already read) is
    watched: its end raises _BrowserGone. A byte from it (a next request) ends the watch."""
    watch = conn is not None and not hasattr(conn, "pending")      # an SSLSocket cannot peek
    if not watch:
        return sock.recv(CHUNK)
    p = select.poll()
    p.register(sock, select.POLLIN)
    p.register(conn, select.POLLIN)
    while True:
        for fd, _ in p.poll():
            if fd == sock.fileno():
                return sock.recv(CHUNK)
            try:
                peek = conn.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT)
            except (BlockingIOError, InterruptedError):
                continue
            except OSError:
                peek = b""
            if not peek:
                raise _BrowserGone("the browser closed its connection")
            p.unregister(conn)
            return sock.recv(CHUNK)


def _relay(handler, sock: socket.socket, upgrade: bool) -> None:
    """chatd's answer to the browser as it comes; the status and headers are read on the way to
    know how the browser's connection goes on (kept alive after a sized answer, closed after a
    stream, spliced after a 101)."""
    conn = getattr(handler, "connection", None)
    head = b""
    while b"\r\n\r\n" not in head:
        try:
            chunk = _recv(sock, conn) if not upgrade else sock.recv(CHUNK)
        except _BrowserGone:
            handler.close_connection = True
            return
        except OSError:
            chunk = b""
        if not chunk:
            if not head:
                return _json(handler, 502, "chat_unavailable", "the chat service sent nothing")
            break
        head += chunk
    status_line, _, rest = head.partition(b"\r\n")
    parts = status_line.split(b" ", 2)
    status = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 502
    hdrs = rest.split(b"\r\n\r\n", 1)[0].lower()
    handler._status = status
    sized = b"\ncontent-length:" in b"\n" + hdrs
    out = handler.wfile
    try:
        # the answer starts: a front server's request timeout (proxy/app.py) never cuts a stream
        handler.connection.settimeout(None)
        out.write(head)
        out.flush()
        if status == 101 and upgrade:
            handler.close_connection = True
            return _splice(handler, sock)
        while True:
            chunk = _recv(sock, conn)
            if not chunk:
                break
            out.write(chunk)
            out.flush()
    except (BrokenPipeError, ConnectionResetError, OSError):
        handler.close_connection = True
        return
    if not sized:
        handler.close_connection = True


def _splice(handler, sock: socket.socket) -> None:
    """A WebSocket: the browser's bytes to chatd on a thread, chatd's to the browser here.
    The front server's idle timeout (Handler.timeout) is lifted first: an open terminal sends
    nothing while idle and chatd never pings, so the timeout would close it."""
    try:
        handler.connection.settimeout(None)
    except OSError:
        return

    def up():
        try:
            while True:
                data = handler.rfile.read1(CHUNK)
                if not data:
                    break
                sock.sendall(data)
        except (OSError, ValueError):
            pass
        try:
            sock.shutdown(socket.SHUT_WR)
        except OSError:
            pass
    t = threading.Thread(target=up, name="chat-ws-up", daemon=True)
    t.start()
    try:
        while True:
            data = sock.recv(CHUNK)
            if not data:
                break
            handler.wfile.write(data)
    except (OSError, ValueError):
        pass
    try:
        handler.connection.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass
    t.join(5)
