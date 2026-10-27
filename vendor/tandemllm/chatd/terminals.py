"""Terminals and the port preview (SRV-357, Chat 5). Mixed into `chatd.service.Service`.

TERMINALS. `shell` is the user's own PTY in the chat's sandbox (`bash -l` through an exec with a
TTY). chatd holds it, not the browser: the last 1 MiB of output stays in a ring buffer, a device
that attaches gets that first and then the live bytes, several devices may attach and type at once,
and a terminal nobody has attached for 30 minutes is closed. Any other session name is an agent
call id: a read-only view of that call's output (its scrollback, then its live `tool.output`), so
the user's shell and the agent never type into the same PTY.

    WS /conversations/{id}/terminals/{session}/ws   binary frames = PTY bytes both ways;
                                                    text frames = {"type":"resize","cols","rows"}

THE PORT PREVIEW. The container has no network, so a dev server inside it is reached through an
exec of `relay.py connect <port>` (localhost: ::1, then 127.0.0.1): chatd writes the HTTP request to its stdin and reads the answer
from its stdout. Each answer carries the preview CSP (CONTRACT section 5) and, for the preview's
opaque origin, `Access-Control-Allow-Origin: null`; an HTML answer's absolute `/` URLs, a script's
absolute imports and a style sheet's `url(/...)` are rewritten to the port's prefix (a dev server
started without `--base`), and a page gets the preview's shim (chatd/preview.py). The server sees
a local client: no Origin, Referer or Sec-Fetch-Site/Mode. A redirect to an absolute path or to
the app's own address (`Location: /login`, `http://127.0.0.1:<port>/login`) is rewritten to the
prefix the client used (SRV-631). A WebSocket (hot reload) goes through the same relay.

The answer goes to the client as it arrives (#86): chatd reads the app's status line and headers
(64 KiB at most, within HEAD_WAIT_S, 300 s: a long-poll's hold, a dev server's first compile; past
it 504), then a page, script or style sheet up to REWRITE_LIMIT (8 MiB) is read whole, rewritten
and sent sized. A bigger one passes as the app sent it, with `X-Tandem-Rewrite: skipped`: a
dependency bundle that size imports by relative paths and keeps working. Everything else streams:
an event stream, a chunked or until-close answer goes on unsized with no read limit; an answer with
a Content-Length keeps it, with READ_IDLE_S (60 s) as the longest gap. While it waits, chatd
watches the client's socket; a client that closed, or that takes no bytes for WRITE_S (60 s),
ends the relay's exec (`end_exec`) and the request's hold on `box.busy`.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import socket
import struct
import threading
import time

from chatd.events import Subscriber
from chatd.sandbox import ClientGone, ExecReader, Unavailable, demux, relay_cmd

RING = 1024 * 1024
IDLE_CLOSE_S = 30 * 60
GUID = b"258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
_PORT_PATH = re.compile(r"/sandbox/ports/\d+")
# the port preview's bounds (#86): the app's head, a page/script/style sheet read whole for its
# rewrite, the longest gap inside a body that is not a stream, a write the client does not take
HEAD_WAIT_S = 300.0
HEAD_LIMIT = 64 * 1024
REWRITE_LIMIT = 8 * 1024 * 1024
READ_IDLE_S = 60.0
WRITE_S = 60.0
BODY_CHUNK = 64 * 1024


def port_target(raw: str) -> str | None:
    """The request target for the app on a sandbox port: what follows /sandbox/ports/<n> as the
    client sent it, still encoded, its trailing slash and query kept (a decoded %0d%0a would add a
    header); None when it names no port or holds a space or a line break."""
    m = _PORT_PATH.search(raw)
    if m is None:
        return None
    rest = raw[m.end():]
    path = rest if rest.startswith("/") else "/" + rest
    return None if any(ch in path for ch in " \r\n") else path


def port_prefix(raw: str) -> str | None:
    """The port's prefix as the client sent it: the path up to and including /sandbox/ports/<n>,
    with /v1/chat-app when the client used it; None when it names no port."""
    m = _PORT_PATH.search(raw)
    return raw[:m.end()] if m else None


# ---------------------------------------------------------------------- RFC 6455, the server side
def ws_accept(handler) -> bool:
    key = handler.headers.get("Sec-WebSocket-Key")
    if (handler.headers.get("Upgrade") or "").lower() != "websocket" or not key:
        return False
    accept = base64.b64encode(hashlib.sha1(key.encode() + GUID).digest()).decode()
    handler.send_response(101, "Switching Protocols")
    handler.send_header("Upgrade", "websocket")
    handler.send_header("Connection", "Upgrade")
    handler.send_header("Sec-WebSocket-Accept", accept)
    handler.end_headers()
    handler.wfile.flush()
    handler.close_connection = True
    return True


class WsConn:
    def __init__(self, handler):
        self.r = handler.rfile
        self.w = handler.wfile
        self.lock = threading.Lock()
        self.closed = False

    def send(self, opcode: int, data: bytes) -> None:
        n = len(data)
        head = bytes([0x80 | opcode])
        if n < 126:
            head += bytes([n])
        elif n < 65536:
            head += bytes([126]) + struct.pack(">H", n)
        else:
            head += bytes([127]) + struct.pack(">Q", n)
        with self.lock:
            if self.closed:
                return
            try:
                self.w.write(head + data)
                self.w.flush()
            except (OSError, ValueError):
                self.closed = True

    def binary(self, data: bytes) -> None:
        self.send(0x2, data)

    def text(self, s: str) -> None:
        self.send(0x1, s.encode())

    def recv(self) -> tuple[int, bytes] | None:
        """The next whole message (fragments joined); pings answered here. None: closed."""
        buf, first_op = b"", None
        while True:
            h = self.r.read(2)
            if len(h) < 2:
                return None
            fin, op = h[0] & 0x80, h[0] & 0x0F
            masked, n = h[1] & 0x80, h[1] & 0x7F
            if n == 126:
                n = struct.unpack(">H", self.r.read(2))[0]
            elif n == 127:
                n = struct.unpack(">Q", self.r.read(8))[0]
            mask = self.r.read(4) if masked else b""
            data = self.r.read(n) if n else b""
            if masked:
                data = bytes(b ^ mask[i % 4] for i, b in enumerate(data))
            if op == 0x8:
                self.send(0x8, data[:2])
                self.closed = True
                return None
            if op == 0x9:
                self.send(0xA, data)
                continue
            if op == 0xA:
                continue
            if first_op is None:
                first_op = op
            buf += data
            if fin:
                return first_op, buf


class Terminal:
    def __init__(self, svc, cid: str, session: str, exec_id: str, sock: socket.socket, first: bytes):
        self.svc, self.cid, self.session = svc, cid, session
        self.exec_id, self.sock = exec_id, sock
        self.ring = bytearray(first[-RING:])
        self.clients: set[WsConn] = set()
        self.lock = threading.Lock()
        self.alive = True
        self.created = time.time()
        self.detached_at = time.time()
        threading.Thread(target=self._pump, name="chatd-term", daemon=True).start()

    def _pump(self) -> None:
        while True:
            try:
                d = self.sock.recv(65536)
            except OSError:
                d = b""
            if not d:
                break
            with self.lock:
                self.ring += d
                if len(self.ring) > RING:
                    del self.ring[:len(self.ring) - RING]
                clients = list(self.clients)
            for c in clients:
                c.binary(d)
        self.alive = False
        for c in list(self.clients):
            c.binary(b"\r\n\x1b[2m[the shell ended]\x1b[0m\r\n")

    def write(self, data: bytes) -> None:
        try:
            self.sock.sendall(data)
        except OSError:
            self.alive = False

    def resize(self, cols: int, rows: int) -> None:
        try:
            self.svc.sandboxes.engine.call("POST", f"/exec/{self.exec_id}/resize?h={int(rows)}"
                                                   f"&w={int(cols)}", timeout=5)
        except OSError:
            pass

    def close(self) -> None:
        self.alive = False
        try:
            self.sock.close()
        except OSError:
            pass


class TerminalsMixin:
    def _terms(self) -> dict:
        if not hasattr(self, "_terminals"):
            self._terminals = {}
        return self._terminals

    def _calls(self, cid: str) -> list[dict]:
        conv = self.conv(cid)
        with conv.lock:
            row = self._row(cid)
            branch, _ = self._branch(conv, row["head_id"])
        return [p for m in branch if m["role"] == "assistant" for p in m["parts"]
                if p.get("type") == "tool_call"]

    def terminals_list(self, cid: str) -> list[dict]:
        out = []
        t = self._terms().get((cid, "shell"))
        if t is not None:
            out.append({"session": "shell", "title": "Terminal", "owner": "user",
                        "alive": t.alive})
        for p in self._calls(cid)[-20:]:
            if p.get("name") in ("bash", "render_page"):
                try:
                    cmd = json.loads(p.get("input") or "{}").get("command") or p["name"]
                except ValueError:
                    cmd = p["name"]
                out.append({"session": p["id"], "title": str(cmd)[:60], "owner": "agent",
                            "alive": p.get("status") == "running"})
        return out

    def terminal_open(self, cid: str, session: str = "") -> dict:
        t = self._shell(cid)
        return {"session": t.session, "title": "Terminal", "owner": "user", "alive": t.alive}

    def _shell(self, cid: str) -> Terminal:
        terms = self._terms()
        t = terms.get((cid, "shell"))
        if t is not None and t.alive:
            return t
        self.sandboxes.ensure(cid)
        exec_id, sock, first = self.sandboxes.open_exec(
            cid, ["bash", "-l"], tty=True, stdin=True,
            env=["TERM=xterm-256color", "PS1=\\[\\e[32m\\]sandbox\\[\\e[0m\\]:\\[\\e[34m\\]\\w"
                 "\\[\\e[0m\\]$ "])
        t = terms[(cid, "shell")] = Terminal(self, cid, "shell", exec_id, sock, first)
        return t

    def terminal_ws(self, handler, cid: str, session: str) -> None:
        self.conv(cid)
        if not ws_accept(handler):
            raise self._api_error(400, "bad_request", "a WebSocket upgrade is expected")
        ws = WsConn(handler)
        if session != "shell":
            return self._call_view(ws, cid, session)
        try:
            t = self._shell(cid)
        except Unavailable as exc:
            ws.binary(f"\x1b[31mThe sandbox is not available: {exc}\x1b[0m\r\n".encode())
            ws.send(0x8, b"")
            return
        box = self.sandboxes.box(cid)
        box.busy += 1
        with t.lock:
            ws.binary(bytes(t.ring))
            t.clients.add(ws)
        try:
            while t.alive:
                m = ws.recv()
                if m is None:
                    break
                op, data = m
                if op == 0x2:
                    t.write(data)
                elif op == 0x1:
                    try:
                        c = json.loads(data)
                    except ValueError:
                        continue
                    if c.get("type") == "resize":
                        t.resize(c.get("cols") or 80, c.get("rows") or 24)
                    elif c.get("type") == "input":
                        t.write(str(c.get("data") or "").encode())
                self.sandboxes.touch(cid)
        finally:
            with t.lock:
                t.clients.discard(ws)
                if not t.clients:
                    t.detached_at = time.time()
            box.busy -= 1

    def _call_view(self, ws: WsConn, cid: str, call_id: str) -> None:
        """An agent call's output, read-only: what it wrote so far, then live."""
        conv = self.conv(cid)
        sub = Subscriber()
        with conv.lock:
            part = next((p for p in self._calls(cid) if p.get("id") == call_id), None)
            live = conv.live
            if part is None and live is not None:
                part = next((p for p in live["parts"] if p.get("id") == call_id), None)
            if part is None:
                ws.binary(f"no output for {call_id}\r\n".encode())
                ws.send(0x8, b"")
                return
            conv.fan.add(sub)
            try:
                cmd = json.loads(part.get("input") or "{}").get("command") or part.get("name")
            except ValueError:
                cmd = part.get("name")
            head = f"\x1b[2m$ {cmd}\x1b[0m\r\n" + ((part.get("output") or {}).get("content")
                                                    or "")
            ws.binary(head.replace("\r\n", "\n").replace("\n", "\r\n").encode())
            running = part.get("status") == "running"
        try:
            while running and not ws.closed:
                items = sub.take(15.0)
                if items is None:
                    break
                for _, kind, data in items:
                    if kind not in ("tool.output", "tool.done"):
                        continue
                    ev = json.loads(data)
                    if ev.get("call_id") != call_id:
                        continue
                    if kind == "tool.done":
                        running = False
                        break
                    ws.binary(ev["delta"].replace("\r\n", "\n").replace("\n", "\r\n").encode())
        finally:
            with conv.lock:
                conv.fan.remove(sub)
        ws.binary(b"\r\n\x1b[2m[the command ended]\x1b[0m\r\n")

    def close_idle_terminals(self) -> None:
        now = time.time()
        for key, t in list(self._terms().items()):
            if not t.alive or (not t.clients and now - t.detached_at > IDLE_CLOSE_S):
                t.close()
                self._terms().pop(key, None)

    # ------------------------------------------------------------------ the port preview
    def proxy_port(self, handler, cid: str, port: int) -> None:
        """A plain HTTP request to the app on `port` (see the module's doc): the request goes to
        the relay, the answer comes back as it arrives (#86)."""
        self.conv(cid)
        box = self.sandboxes.box(cid)
        if box.state != "running":
            raise self._api_error(503, "sandbox_stopped", "the sandbox is not running; start it "
                                                          "or ask the agent to start the server")
        path = port_target(handler.path)
        if path is None:
            raise self._api_error(400, "bad_request", "the path holds a space or a line break, "
                                                     "or names no port")
        if (handler.headers.get("Upgrade") or "").lower() == "websocket":
            return self._proxy_ws(handler, cid, port, path)
        try:
            n = int(handler.headers.get("Content-Length") or 0)
        except ValueError:
            n = -1
        if n < 0:
            raise self._api_error(400, "bad_request", "Content-Length is not a length")
        lines = [f"{handler.command} {path} HTTP/1.1", f"Host: 127.0.0.1:{port}",
                 "Connection: close"]
        for k, v in handler.headers.items():
            # no accept-encoding: the rewrites read the answer's bytes; no expect: no 100 to wait on
            if k.lower() in ("host", "connection", "cookie", "authorization", "upgrade",
                             "keep-alive", "x-tandem-forward", "accept-encoding", "origin",
                             "referer", "sec-fetch-site", "sec-fetch-mode", "expect", "te",
                             "proxy-connection") or k.lower().startswith("x-tandem"):
                continue
            lines.append(f"{k}: {v}")
        exec_id, sock, first = self.sandboxes.open_exec(
            cid, relay_cmd("connect", str(port)), stdin=True)
        with box.lock:
            box.busy += 1
        reader = ExecReader(sock, first, watch=handler.connection)
        timeout = handler.connection.gettimeout()
        try:
            handler.connection.settimeout(WRITE_S)     # a client that stops reading ends it
            # a relay that already answered and left (nothing listens: its own 502) closed its
            # stdin; what it wrote is still read below. A client that stalls in its body ends it.
            try:
                sock.sendall(("\r\n".join(lines) + "\r\n\r\n").encode("latin-1"))
                while n > 0:
                    chunk = handler.rfile.read(min(BODY_CHUNK, n))
                    if not chunk:
                        raise ClientGone()
                    n -= len(chunk)
                    sock.sendall(chunk)
            except (BrokenPipeError, ConnectionResetError):
                handler.close_connection = True
            if self._port_answer(handler, reader, cid, port, path):
                _drain(reader, 2.0)     # the app closes after its answer: let the relay end
        except (ClientGone, OSError, ValueError):
            handler.close_connection = True
        finally:
            if not reader.ended:
                self.sandboxes.end_exec(exec_id)
            with box.lock:
                box.busy -= 1
            try:
                sock.close()
            except OSError:
                pass
            try:
                handler.connection.settimeout(timeout)
            except OSError:
                pass

    def _port_answer(self, handler, reader: ExecReader, cid: str, port: int, path: str) -> bool:
        """The app's answer from `reader` to the client: a bounded page, script or style sheet
        rewritten and sized; anything else streamed as it arrives (the module's doc). True when
        the whole answer went out."""
        from chatd.panes import PREVIEW_CSP
        from chatd.preview import cors, rewrite_css, rewrite_location, rewrite_page, rewrite_script
        extra = cors(handler.headers.get("Origin")) + (
            ("Content-Security-Policy", PREVIEW_CSP), ("X-Content-Type-Options", "nosniff"))
        words = f"the server on port {port}"
        deadline = time.monotonic() + HEAD_WAIT_S
        buf = b""
        while True:                     # the head, past any 1xx (103 Early Hints)
            while b"\r\n\r\n" not in buf:
                if len(buf) > HEAD_LIMIT:
                    return _port_error(handler, 502, f"{words} sent a head over 64 KiB", extra)
                try:
                    d = reader.read(max(0.0, deadline - time.monotonic()))
                except TimeoutError:
                    return _port_error(handler, 504, f"{words} did not answer within "
                                                     f"{HEAD_WAIT_S:g} s", extra)
                if not d:
                    return _port_error(handler, 502, f"{words} closed the connection before it "
                                                     "answered", extra)
                buf += d
            head, _, buf = buf.partition(b"\r\n\r\n")
            status_line, _, hdrs = head.partition(b"\r\n")
            parts = status_line.split(b" ", 2)
            if not status_line.startswith(b"HTTP/") or len(parts) < 2 or not parts[1].isdigit():
                return _port_error(handler, 502, f"{words} answered something that is not HTTP",
                                   extra)
            status = int(parts[1])
            if not 100 <= status < 200:
                break
        keep, ctype, clen, chunked = [], None, None, False
        for h in hdrs.split(b"\r\n"):
            k, _, v = h.decode("latin-1").partition(":")
            kl, v = k.strip().lower(), v.strip()
            if kl == "transfer-encoding":
                chunked = chunked or "chunked" in v.lower()
                continue
            if kl == "content-length":
                clen = int(v) if v.isdigit() else None
                continue
            if not kl or kl in ("connection", "content-security-policy", "set-cookie",
                                "keep-alive", "vary", "x-frame-options") or \
                    kl.startswith("access-control-"):
                continue
            if kl == "content-type":
                ctype = v
                continue
            if kl in ("location", "content-location"):
                v = rewrite_location(v, port_prefix(handler.path), port)
            keep.append((k.strip(), v))
        if chunked:
            clen = None
        headers = tuple(keep) + extra
        if handler.command == "HEAD" or status in (204, 304):
            _port_head(handler, status, ctype, headers, clen if clen is not None else 0)
            return True
        low = (ctype or "").lower()
        prefix = f"/v1/chat-app/conversations/{cid}/sandbox/ports/{port}"
        rewrite = None
        if "event-stream" not in low and (clen is None or clen <= REWRITE_LIMIT):
            if low.startswith("text/html"):
                rewrite = lambda b: rewrite_page(b, prefix, port)               # noqa: E731
            elif "javascript" in low:
                rewrite = lambda b: rewrite_script(b, prefix, path)             # noqa: E731
            elif low.startswith("text/css"):
                rewrite = lambda b: rewrite_css(b, prefix)                      # noqa: E731
        stream = rewrite is None and (clen is None or "event-stream" in low)
        pieces = _body(reader, buf, chunked, clen, None if stream else READ_IDLE_S)
        sent = 0
        if rewrite is not None:
            body = bytearray()
            try:
                for piece in pieces:
                    body += piece
                    if len(body) > REWRITE_LIMIT:
                        break
            except TimeoutError:
                return _port_error(handler, 504, f"{words} stopped sending its answer for "
                                                 f"{READ_IDLE_S:g} s", extra)
            if len(body) <= REWRITE_LIMIT:
                out = rewrite(bytes(body))
                _port_head(handler, status, ctype, headers, len(out))
                handler.wfile.write(out)
                return True
            # over the cap: what came so far and the rest as it comes, as the app sent it
            _port_head(handler, status, ctype, headers + (
                ("X-Tandem-Rewrite", f"skipped, over {REWRITE_LIMIT >> 20} MiB"),), clen)
            handler.wfile.write(body)
            sent = len(body)
            del body
        else:
            _port_head(handler, status, ctype, headers, None if stream else clen)
        try:
            for piece in pieces:
                handler.wfile.write(piece)
                sent += len(piece)
        except TimeoutError:            # a sized body that stalled: the client sees it short
            handler.close_connection = True
            return False
        if clen is not None and sent != clen:
            handler.close_connection = True
        return True

    def _proxy_ws(self, handler, cid: str, port: int, path: str) -> None:
        """A WebSocket from the preview to a server in the sandbox (a dev server's live reload, an
        app's socket): the upgrade goes to it through the relay, its 101 comes back as it is, then
        bytes both ways until either side closes. When the browser closes its side the relay's
        stdin closes (it then closes its side to the server); a relay still there 5 s later is
        ended."""
        lines = [f"GET {path} HTTP/1.1", f"Host: 127.0.0.1:{port}", "Upgrade: websocket",
                 "Connection: Upgrade"]
        for k, v in handler.headers.items():
            kl = k.lower()
            if kl.startswith("sec-websocket-") or kl in ("user-agent", "accept-language"):
                lines.append(f"{k}: {v}")
        box = self.sandboxes.box(cid)
        exec_id, sock, first = self.sandboxes.open_exec(
            cid, relay_cmd("connect", str(port)), stdin=True)
        box.busy += 1
        handler.close_connection = True
        out = handler.wfile

        def down(stream: int, data: bytes) -> None:
            if stream == 1:
                out.write(data)
                out.flush()

        def up() -> None:
            try:
                while True:
                    data = handler.rfile.read1(65536)
                    if not data:
                        break
                    sock.sendall(data)
                    self.sandboxes.touch(cid)
            except OSError:
                pass
            try:
                sock.shutdown(socket.SHUT_WR)
            except OSError:
                pass
            time.sleep(5)
            self.sandboxes.end_exec(exec_id)
        try:
            sock.sendall(("\r\n".join(lines) + "\r\n\r\n").encode("latin-1"))
            threading.Thread(target=up, daemon=True).start()
            demux(sock, first, down)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            box.busy -= 1
            try:
                sock.close()
            except OSError:
                pass


def _port_head(handler, status: int, ctype: str | None, headers: tuple, length: int | None
               ) -> None:
    """The answer's status and headers; `length` None: unsized, the connection closes after it."""
    handler.send_response(status)
    if ctype:
        handler.send_header("Content-Type", ctype)
    handler.send_header("Cache-Control", "no-store")
    for k, v in headers:
        handler.send_header(k, v)
    if length is None:
        handler.send_header("Connection", "close")
        handler.close_connection = True
    else:
        handler.send_header("Content-Length", str(length))
    handler.end_headers()


def _drain(reader: ExecReader, wait: float) -> None:
    """Up to `wait` s for the relay to end on its own, what it still writes dropped."""
    deadline = time.monotonic() + wait
    try:
        while reader.read(max(0.0, deadline - time.monotonic())):
            pass
    except (TimeoutError, ClientGone, OSError):
        pass


def _port_error(handler, status: int, words: str, headers: tuple) -> None:
    body = (words + "\n").encode()
    handler._send(status, body, "text/plain; charset=utf-8", headers)


def _body(reader: ExecReader, rest: bytes, chunked: bool, clen: int | None,
          idle: float | None):
    """The answer's body in pieces as the relay delivers them, after the head's `rest`: chunked
    decoded, a Content-Length counted, else until the app closes. `idle`: the longest wait for
    the next piece (None for a stream); past it TimeoutError."""
    if chunked:
        dec = Dechunk()
        out = dec.feed(rest)
        if out:
            yield out
        while not dec.done:
            d = reader.read(idle)
            if not d:
                return
            out = dec.feed(d)
            if out:
                yield out
        return
    if clen is not None:
        left = clen
        piece = rest[:left]
        while True:
            if piece:
                yield piece
                left -= len(piece)
            if left <= 0:
                return
            d = reader.read(idle)
            if not d:
                return
            piece = d[:left]
    if rest:
        yield rest
    while True:
        d = reader.read(idle)
        if not d:
            return
        yield d


_HEX = re.compile(rb"[0-9A-Fa-f]{1,16}")


class Dechunk:
    """A chunked body (RFC 9112 7.1) decoded as it arrives: `feed(bytes)` gives the data it
    completes; `done` after the last chunk and its trailers. Extensions and trailers are dropped;
    a size line or trailer over 4 KiB, or a size that is not hex, raises ValueError."""

    def __init__(self):
        self.buf = bytearray()
        self.left = 0
        self.state = "size"
        self.done = False

    def _line(self) -> bytes | None:
        i = self.buf.find(b"\r\n")
        if i < 0:
            if len(self.buf) > 4096:
                raise ValueError("a chunk's size line or trailer is over 4 KiB")
            return None
        line = bytes(self.buf[:i])
        del self.buf[:i + 2]
        return line

    def feed(self, data: bytes) -> bytes:
        self.buf += data
        out = bytearray()
        while not self.done:
            if self.state == "size":
                line = self._line()
                if line is None:
                    break
                size = line.split(b";")[0].strip()
                if not _HEX.fullmatch(size):
                    raise ValueError("a chunk size that is not hex")
                self.left = int(size, 16)
                self.state = "data" if self.left else "trailer"
            elif self.state == "data":
                take = min(self.left, len(self.buf))
                if not take:
                    break
                out += self.buf[:take]
                del self.buf[:take]
                self.left -= take
                if not self.left:
                    self.state = "crlf"
            elif self.state == "crlf":
                if len(self.buf) < 2:
                    break
                del self.buf[:2]
                self.state = "size"
            else:
                line = self._line()
                if line is None:
                    break
                if not line:
                    self.done = True
        return bytes(out)
