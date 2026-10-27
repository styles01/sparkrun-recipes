"""A small HTTP/1.1 client for the relay: keep-alive connections, and a body read in pieces as they
arrive with a check between waits.

`http.client` buffers the socket behind a file object, so a `select` on the socket cannot tell
whether data is already waiting, and a timeout in the middle of a chunked read leaves its parser in
an unknown state. The relay needs both: it reads an upstream stream while it checks every quarter
second whether its own client has left (then it closes the upstream connection, so the backend
stops generating) and whether the idle gap has passed. So this module owns its buffer:

    conn = pool.get()                       a pooled connection, or a new one (TLS for https)
    conn.send(method, path, headers, body)
    head = conn.read_head(deadline, poll)   status, reason, headers; waits up to `deadline`
    for piece in conn.body(idle, poll):     bytes as they arrive (chunked decoded)
    pool.put(conn)                          back for the next request, when the body was whole

`poll()` is called at least every 0.25 s while a read waits; a true answer raises `Aborted`. A
connection taken from the pool that turns out to be closed by the server before any byte of the
answer is reported as `Stale`, and the relay sends again on a fresh one without counting a retry.
A cloud TLS handshake costs about 100 ms; the pool pays it once per connection.
"""

from __future__ import annotations

import socket
import ssl
import threading
import time
from urllib.parse import urlsplit

TICK = 0.25
MAX_HEAD = 64 * 1024


class UpstreamError(Exception):
    """A connection that failed before a response head arrived (the relay may retry it)."""


class Stale(UpstreamError):
    """A pooled connection the server had closed: send again on a fresh one."""


class Timeout(Exception):
    def __init__(self, which: str):
        super().__init__(which)
        self.which = which                       # connect | first_byte | idle


class Aborted(Exception):
    """`poll()` asked to stop (the client left)."""


class Closed(Exception):
    """The upstream closed the connection before the body was whole."""


class Conn:
    def __init__(self, sock, pool: "Pool", reused: bool = False):
        self.sock = sock
        self.pool = pool
        self.buf = bytearray()
        self.reused = reused
        self.keep = False
        self.done = False
        self.t_sent: float | None = None
        self.t_first_byte: float | None = None
        self.connect_ms = 0.0                 # DNS, TCP and TLS of a new connection; 0 when reused

    # ----------------------------------------------------------------- writing
    def send(self, method: str, path: str, headers: list[tuple[str, str]], body: bytes) -> None:
        lines = [f"{method} {path} HTTP/1.1", f"Host: {self.pool.host_header}",
                 f"Content-Length: {len(body)}", "Connection: keep-alive"]
        for k, v in headers:
            if "\r" in k or "\n" in k or "\r" in str(v) or "\n" in str(v):
                continue
            lines.append(f"{k}: {v}")
        data = ("\r\n".join(lines) + "\r\n\r\n").encode("latin-1", "replace") + body
        try:
            self.sock.settimeout(self.pool.connect_timeout)
            self.sock.sendall(data)
        except (OSError, ssl.SSLError) as exc:
            self.close()
            if self.reused:
                raise Stale(str(exc))
            raise UpstreamError(f"send failed: {exc}")
        self.t_sent = time.perf_counter()

    # ----------------------------------------------------------------- reading
    def _fill(self, deadline: float, poll, which: str) -> bool:
        """Read what arrives into the buffer. False at end of file."""
        while True:
            if poll is not None and poll():
                raise Aborted()
            left = deadline - time.monotonic()
            if left <= 0:
                raise Timeout(which)
            self.sock.settimeout(min(TICK, left))
            try:
                got = self.sock.recv(65536)
            except (socket.timeout, ssl.SSLWantReadError):
                continue
            except (OSError, ssl.SSLError):
                got = b""
            if not got:
                return False
            if self.t_first_byte is None:
                self.t_first_byte = time.perf_counter()
            self.buf += got
            return True

    def read_head(self, timeout: float, poll=None) -> tuple[int, str, list[tuple[str, str]]]:
        deadline = time.monotonic() + timeout
        while True:
            end = self.buf.find(b"\r\n\r\n")
            if end >= 0:
                break
            if len(self.buf) > MAX_HEAD:
                self.close()
                raise UpstreamError("response head too large")
            try:
                ok = self._fill(deadline, poll, "first_byte")
            except Timeout:
                self.close()
                raise
            if not ok:
                self.close()
                if self.reused and not self.buf:
                    raise Stale("closed before the response")
                raise UpstreamError("connection closed before the response head")
        head = bytes(self.buf[:end]).decode("latin-1")
        del self.buf[:end + 4]
        lines = head.split("\r\n")
        parts = lines[0].split(" ", 2)
        try:
            status = int(parts[1])
        except (IndexError, ValueError):
            self.close()
            raise UpstreamError(f"bad status line {lines[0][:80]!r}")
        reason = parts[2] if len(parts) > 2 else ""
        hdrs = []
        for line in lines[1:]:
            k, sep, v = line.partition(":")
            if sep:
                hdrs.append((k.strip(), v.strip()))
        self.headers = hdrs
        get = {k.lower(): v for k, v in hdrs}
        self.chunked = "chunked" in get.get("transfer-encoding", "").lower()
        self.length = int(get["content-length"]) if get.get("content-length", "").isdigit() \
            else None
        conn_h = get.get("connection", "").lower()
        self.keep = (conn_h != "close") and (self.chunked or self.length is not None)
        if status in (204, 304) or 100 <= status < 200:
            self.length = 0
        return status, reason, hdrs

    def body(self, idle: float, poll=None):
        """The body, piece by piece as it arrives. `idle` bounds every wait for the next byte."""
        if self.length is not None and not self.chunked:
            left = self.length
            while left > 0:
                if not self.buf and not self._fill(time.monotonic() + idle, poll, "idle"):
                    self.close()
                    raise Closed()
                piece = bytes(self.buf[:left])
                del self.buf[:len(piece)]
                left -= len(piece)
                yield piece
            self.done = True
            return
        if not self.chunked:
            while True:
                if self.buf:
                    piece = bytes(self.buf)
                    self.buf.clear()
                    yield piece
                if not self._fill(time.monotonic() + idle, poll, "idle"):
                    self.done = True
                    self.keep = False
                    return
        while True:                                  # chunked
            while b"\r\n" not in self.buf:
                if not self._fill(time.monotonic() + idle, poll, "idle"):
                    self.close()
                    raise Closed()
            line, _, rest = bytes(self.buf).partition(b"\r\n")
            try:
                size = int(line.split(b";")[0].strip() or b"0", 16)
            except ValueError:
                self.close()
                raise Closed()
            self.buf = bytearray(rest)
            if size == 0:
                # trailers until an empty line
                while b"\r\n" not in self.buf:
                    if not self._fill(time.monotonic() + idle, poll, "idle"):
                        break
                idx = self.buf.find(b"\r\n")
                while idx > 0:
                    del self.buf[:idx + 2]
                    while b"\r\n" not in self.buf:
                        if not self._fill(time.monotonic() + idle, poll, "idle"):
                            break
                    idx = self.buf.find(b"\r\n")
                if idx == 0:
                    del self.buf[:2]
                self.done = True
                return
            left = size
            while left > 0:
                if not self.buf and not self._fill(time.monotonic() + idle, poll, "idle"):
                    self.close()
                    raise Closed()
                piece = bytes(self.buf[:left])
                del self.buf[:len(piece)]
                left -= len(piece)
                yield piece
            while len(self.buf) < 2:
                if not self._fill(time.monotonic() + idle, poll, "idle"):
                    self.close()
                    raise Closed()
            del self.buf[:2]

    def read_all(self, idle: float, poll=None, limit: int = 256 * 1024 * 1024) -> bytes:
        out = bytearray()
        for piece in self.body(idle, poll):
            out += piece
            if len(out) > limit:
                self.close()
                raise UpstreamError("response body over the limit")
        return bytes(out)

    def close(self) -> None:
        self.keep = False
        try:
            self.sock.close()
        except OSError:
            pass


class Pool:
    """The keep-alive connections of one backend."""

    def __init__(self, base_url: str, connect_timeout: float = 5.0, max_idle: int = 16,
                 ssl_context: ssl.SSLContext | None = None):
        u = urlsplit(base_url)
        self.tls = u.scheme == "https"
        self.host = u.hostname or "localhost"
        self.port = u.port or (443 if self.tls else 80)
        default_port = 443 if self.tls else 80
        self.host_header = self.host if self.port == default_port else f"{self.host}:{self.port}"
        if ":" in self.host and not self.host.startswith("["):
            self.host_header = f"[{self.host}]" + ("" if self.port == default_port
                                                   else f":{self.port}")
        self.prefix = u.path.rstrip("/")
        self.connect_timeout = connect_timeout
        self.max_idle = max_idle
        self._ctx = ssl_context
        self._idle: list[tuple[float, Conn]] = []
        self._lock = threading.Lock()
        self.opened = 0
        # the host's addresses, resolved once a minute: a `.local` name costs ~0.5 s of mDNS on
        # macOS per lookup, and a server that closes each stream's connection (Connection: close)
        # would pay it on every request
        self._addrs: list | None = None
        self._addrs_at = 0.0
        self.last_connect_ms = 0.0

    def _resolve(self, fresh: bool = False) -> list:
        now = time.monotonic()
        if fresh or self._addrs is None or now - self._addrs_at > 60.0:
            infos = socket.getaddrinfo(self.host, self.port, type=socket.SOCK_STREAM)
            self._addrs = [i[4] for i in infos]
            self._addrs_at = now
        return self._addrs

    def path(self, route: str) -> str:
        """The upstream path of a client route: `/v1/chat/completions` -> base path + the part
        after `/v1` (`https://api.deepseek.com/v1` + `/chat/completions`)."""
        tail = route[3:] if route.startswith("/v1/") else route
        return (self.prefix or "") + tail

    def _context(self) -> ssl.SSLContext:
        if self._ctx is None:
            self._ctx = ssl.create_default_context()
        return self._ctx

    def get(self, fresh: bool = False) -> Conn:
        if not fresh:
            with self._lock:
                while self._idle:
                    t, c = self._idle.pop()
                    if time.monotonic() - t < 50.0 and _alive(c.sock):
                        c.reused = True
                        c.connect_ms = 0.0
                        c.buf.clear()
                        c.t_sent = c.t_first_byte = None
                        c.done = False
                        return c
                    c.close()
        t0 = time.perf_counter()
        try:
            sock = None
            last = None
            try:
                addrs = self._resolve()
            except OSError as exc:
                raise UpstreamError(f"cannot resolve {self.host}: {exc}")
            for i, addr in enumerate(addrs):
                try:
                    sock = socket.create_connection(addr[:2], timeout=self.connect_timeout)
                    if i:
                        # the address that answered goes first next time (a host that lists
                        # IPv6 addresses but listens on IPv4 only refuses each in 60-190 ms)
                        self._addrs = [addr] + [x for x in addrs if x != addr]
                    break
                except socket.timeout:
                    raise
                except OSError as exc:
                    last = exc
                    if i == len(addrs) - 1:
                        self._addrs = None              # resolve again next time
                        raise last
            if sock is None:
                # an empty addr list never entered the loop: the AttributeError used to
                # escape get() unconverted instead of naming the host
                raise UpstreamError(f"no addresses for {self.host}")
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            if self.tls:
                # 2026-10-17: a handshake that raises leaves nothing to close unless we do
                # it here. The rebinding below only happens when wrap_socket RETURNS, so
                # the excepts at the bottom of this try raise with the connected socket
                # owned by no one -- a backend pointed at a plain port as https leaked one
                # fd per request, and at relay request rate the proxy walks to EMFILE,
                # after which the healthy backends fail too.
                try:
                    sock = self._context().wrap_socket(sock, server_hostname=self.host)
                except BaseException:
                    # any raise at all leaves the connected socket owned by no one —
                    # a non-OSError here (bad server_hostname) used to leak the fd too
                    try:
                        sock.close()
                    except Exception:                                     # noqa: BLE001
                        pass
                    raise
        except socket.timeout:
            raise Timeout("connect")
        except (UpstreamError, Timeout):
            raise                             # already converted on the way up
        except (OSError, ssl.SSLError) as exc:
            raise UpstreamError(f"connect to {self.host_header} failed: {exc}")
        except Exception as exc:                                   # noqa: BLE001
            # a non-OSError from wrap_socket (a bad server_hostname is a ValueError) used to
            # escape raw; callers of get() handle UpstreamError and Timeout and nothing else
            raise UpstreamError(
                f"connect to {self.host_header} failed: {type(exc).__name__}: {exc}") from exc
        self.opened += 1
        self.last_connect_ms = (time.perf_counter() - t0) * 1e3
        c = Conn(sock, self)
        c.connect_ms = self.last_connect_ms
        return c

    def put(self, conn: Conn) -> None:
        if not (conn.keep and conn.done) or conn.buf:
            conn.close()
            return
        with self._lock:
            if len(self._idle) >= self.max_idle:
                conn.close()
                return
            self._idle.append((time.monotonic(), conn))

    def close(self) -> None:
        with self._lock:
            for _, c in self._idle:
                c.close()
            self._idle.clear()


def _alive(sock) -> bool:
    """An idle pooled socket the server has not closed: nothing to read, not end of file."""
    try:
        if isinstance(sock, ssl.SSLSocket) and sock.pending():
            return False
        import select
        r, _, _ = select.select([sock], [], [], 0)
        if not r:
            return True
        sock.settimeout(0)
        try:
            return bool(sock.recv(1, socket.MSG_PEEK))
        except (BlockingIOError, ssl.SSLWantReadError):
            return True
        except (OSError, ssl.SSLError, ValueError):
            return False
    except (OSError, ValueError):
        return False
