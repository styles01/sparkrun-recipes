"""The model API, as a client: chatd never imports the engine (D1, D15).

    Engine.stream(body, headers)   POST /v1/chat/completions with `stream: true`; a `Call` reads its
                                   SSE lines and can be closed from another thread (Stop: the engine
                                   sees the socket close and books the request `abandoned`)
    Engine.models()                GET /v1/models: the list and each model's `tandem` card, when
                                   the server sends one (SRV-347); a server without cards still lists
    Engine.facts()                 GET /v1/dashboard/system (else /health): weights, max_len, drafter,
                                   memory -- what our engine says about itself; {} for any other
    Engine.post_json(path, body)   the cache peek and prewarm (SRV-349), the title call
    Engine.release_memory(n)       POST /v1/cache/release: the engine gives back what it holds
                                   idle, for a chat's sandbox (chatd/sandbox.py); 0 from a server
                                   without the route
    LiveTail                       ONE tail of GET /v1/dashboard/live for the whole service while a
                                   run is on (polls `?follow=0` once a second when the stream is
                                   refused); rows by conversation id and by request id

Everything here is model-agnostic: what a model can do comes from the server's answers, and a
server that answers less (vLLM, a cloud API through the proxy) gives a chat with fewer figures,
never a broken one.
"""

from __future__ import annotations

import http.client
import json
import socket
import threading
import time
import urllib.parse

# never `tandem-...`: the ledger counts those as internal calls. `qse-dashboard` in it: an engine
# older than the `chat-app` client kind counts chatd as the dashboard (its chat profile)
USER_AGENT = "chatd/0.1 (qse-dashboard)"
CONNECT_TIMEOUT_S = 10.0
READ_TIMEOUT_S = 900.0       # a 158k prefill is about 3.5 minutes of silence


class EngineError(Exception):
    def __init__(self, status: int, message: str, retryable: bool,
                 retry_after: float | None = None):
        super().__init__(message)
        self.status = status
        self.retryable = retryable
        self.retry_after = retry_after        # the refusal's Retry-After, in seconds


class Call:
    """One streamed model call. Iterate `events()`: ("data", obj), ("comment", text), ("end", None)."""

    def __init__(self, conn: http.client.HTTPConnection, resp: http.client.HTTPResponse):
        self.conn = conn
        self.resp = resp
        self.closed = False

    def events(self):
        data: list[str] = []
        try:
            while True:
                raw = self.resp.readline()
                if not raw:
                    break
                line = raw.decode("utf-8", "replace").rstrip("\r\n")
                if line == "":
                    if data:
                        text = "\n".join(data)
                        data = []
                        if text.strip() == "[DONE]":
                            yield ("done", None)
                            continue
                        try:
                            yield ("data", json.loads(text))
                        except ValueError:
                            continue
                    continue
                if line.startswith(":"):
                    yield ("comment", line[1:].strip())
                elif line.startswith("data:"):
                    data.append(line[5:].lstrip(" ") if line[5:6] == " " else line[5:])
        except Exception:                                           # noqa: BLE001
            # a drop (OSError, IncompleteRead), or the socket closed under the read by `close()`
            # from another thread, which http.client reports in more ways than one
            if not self.closed:
                yield ("broken", None)
                return
        yield ("end", None)

    def close(self) -> None:
        """From any thread: shut the socket, so a blocked read returns at once."""
        self.closed = True
        sock = self.conn.sock
        try:
            if sock is not None:
                sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            self.conn.close()
        except OSError:
            pass


class Engine:
    def __init__(self, base: str, key: str | None = None):
        u = urllib.parse.urlsplit(base.rstrip("/"))
        self.base = base.rstrip("/")
        self.host = u.hostname or "127.0.0.1"
        self.port = u.port or (443 if u.scheme == "https" else 80)
        self.https = u.scheme == "https"
        self.prefix = u.path or ""
        self.key = key

    def _conn(self, timeout: float) -> http.client.HTTPConnection:
        cls = http.client.HTTPSConnection if self.https else http.client.HTTPConnection
        return cls(self.host, self.port, timeout=timeout)

    def _headers(self, extra: dict | None = None) -> dict:
        h = {"User-Agent": USER_AGENT, "Accept": "application/json"}
        if self.key:
            h["Authorization"] = f"Bearer {self.key}"
        h.update(extra or {})
        return h

    def request(self, method: str, path: str, body: bytes | None = None,
                headers: dict | None = None, timeout: float = 10.0) -> tuple[int, object]:
        conn = self._conn(timeout)
        try:
            h = self._headers(headers)
            if body is not None:
                h["Content-Type"] = "application/json"
            conn.request(method, self.prefix + path, body=body, headers=h)
            r = conn.getresponse()
            raw = r.read()
            try:
                return r.status, json.loads(raw) if raw else None
            except ValueError:
                return r.status, None
        finally:
            conn.close()

    def get_json(self, path: str, timeout: float = 3.0) -> tuple[int, object]:
        try:
            return self.request("GET", path, timeout=timeout)
        except (OSError, http.client.HTTPException):
            return 0, None

    def post_json(self, path: str, obj, headers: dict | None = None,
                  timeout: float = 30.0) -> tuple[int, object]:
        try:
            return self.request("POST", path, json.dumps(obj).encode(), headers, timeout)
        except (OSError, http.client.HTTPException):
            return 0, None

    def release_memory(self, nbytes: int, why: str = "sandbox") -> int:
        """Bytes the engine gave back of the `nbytes` asked for; 0 when it gave none, is busy,
        or has no such route (an engine from before it, another server)."""
        st, obj = self.post_json("/v1/cache/release", {"bytes": int(nbytes), "why": why},
                                 timeout=30.0)
        if st != 200 or not isinstance(obj, dict):
            return 0
        return int(obj.get("released_bytes") or 0)

    def stream(self, body: bytes, headers: dict) -> Call:
        """Open a streamed completion. EngineError for a refusal (retryable for 429 and 5xx and a
        connection that fails, not for a 4xx)."""
        conn = self._conn(CONNECT_TIMEOUT_S)
        try:
            h = self._headers(headers)
            h["Content-Type"] = "application/json"
            h["Accept"] = "text/event-stream"
            conn.request("POST", self.prefix + "/v1/chat/completions", body=body, headers=h)
            if conn.sock is not None:
                conn.sock.settimeout(READ_TIMEOUT_S)
            resp = conn.getresponse()
        except (OSError, http.client.HTTPException) as exc:
            conn.close()
            raise EngineError(0, f"the engine did not answer ({type(exc).__name__})", True) from None
        if resp.status != 200:
            raw = resp.read()
            conn.close()
            msg = f"HTTP {resp.status}"
            try:
                err = json.loads(raw).get("error")
                if isinstance(err, dict) and err.get("message"):
                    msg = str(err["message"])[:300]
            except (ValueError, AttributeError):
                pass
            try:
                after = float(resp.getheader("Retry-After") or "")
            except ValueError:
                after = None
            raise EngineError(resp.status, msg, resp.status == 429 or resp.status >= 500, after)
        return Call(conn, resp)

    def models(self) -> list[dict]:
        st, obj = self.get_json("/v1/models")
        if st != 200 or not isinstance(obj, dict):
            return []
        return [m for m in obj.get("data") or [] if isinstance(m, dict) and m.get("id")]

    def facts(self) -> dict:
        """Our engine's own facts (weights, max_len, drafter, memory); {} from another server."""
        st, sysd = self.get_json("/v1/dashboard/system")
        st2, health = self.get_json("/health")
        health = health if st2 == 200 and isinstance(health, dict) and "max_len" in health else None
        if st == 200 and isinstance(sysd, dict):
            eng = sysd.setdefault("engine", {})
            if health is not None and eng.get("default_max_tokens") is None:
                eng["default_max_tokens"] = health.get("default_max_tokens")   # the answer's room
            return sysd
        if health is not None:
            return {"engine": health, "memory": health.get("memory") or {}}
        return {}


def _newer(a: dict, b: dict) -> bool:
    """Is row `a` the one to show over `b`: in flight over finished, then the later end."""
    if (a.get("phase") == "done") != (b.get("phase") == "done"):
        return a.get("phase") != "done"
    return float(a.get("ended_ms_ago") or 0) < float(b.get("ended_ms_ago") or 0)


class LiveTail:
    """One reader of `/v1/dashboard/live` for every run in chatd (SRV-349: a phone watching a chat
    costs the engine nothing extra). It runs while `wanted()` says a run is on and stops 15 s after.
    The engine allows 4 live streams; on a 429 (or a server without the route) it polls the
    snapshot once a second, and a server that has neither gives no rows."""

    IDLE_STOP_S = 15.0

    def __init__(self, engine: Engine, wanted, on_rows=None):
        self.engine = engine
        self.wanted = wanted
        self.on_rows = on_rows
        self.lock = threading.Lock()
        self.by_conv: dict[str, dict] = {}
        self.by_request: dict[str, dict] = {}
        self.engine_state: dict = {}
        self.at = 0.0
        self.thread: threading.Thread | None = None
        self.missing_until = 0.0

    def ensure(self) -> None:
        with self.lock:
            if self.thread is not None and self.thread.is_alive():
                return
            self.thread = threading.Thread(target=self._loop, name="chatd-live", daemon=True)
            self.thread.start()

    def row(self, conversation_id: str | None, request_id: str | None) -> dict | None:
        """The newest row of a run, by its conversation id (SRV-347) or its request id; None when
        the last frame is older than 3 s."""
        with self.lock:
            if time.monotonic() - self.at > 3.0:
                return None
            if conversation_id and conversation_id in self.by_conv:
                return self.by_conv[conversation_id]
            if request_id and request_id in self.by_request:
                return self.by_request[request_id]
            return None

    def _take(self, snap: dict) -> None:
        rows = snap.get("requests") or []
        conv, req = {}, {}
        for r in rows if isinstance(rows, list) else []:
            if not isinstance(r, dict):
                continue
            if r.get("conversation_id"):
                # a conversation's request in flight, else its newest finished one: the snapshot
                # keeps finished rows for 30 s after the live ones, so an agent's next call came
                # after its previous calls' rows and the last of those was taken (2026-10-16)
                k = str(r["conversation_id"])
                if k not in conv or _newer(r, conv[k]):
                    conv[k] = r
            if r.get("request_id"):
                req[str(r["request_id"])] = r
        with self.lock:
            self.by_conv, self.by_request = conv, req
            self.engine_state = snap.get("engine") or {}
            self.at = time.monotonic()
        if self.on_rows is not None:
            self.on_rows()

    def _loop(self) -> None:
        idle_since = None
        while True:
            if self.wanted():
                idle_since = None
            else:
                idle_since = idle_since or time.monotonic()
                if time.monotonic() - idle_since > self.IDLE_STOP_S:
                    with self.lock:
                        self.thread = None
                    return
            if time.monotonic() < self.missing_until:
                time.sleep(1.0)
                continue
            if not self._stream():
                self._poll_once()
                time.sleep(1.0)

    def _stream(self) -> bool:
        """Follow the SSE stream until it ends or nobody needs it. False: use polling."""
        conn = self.engine._conn(30.0)
        try:
            conn.request("GET", self.engine.prefix + "/v1/dashboard/live",
                         headers=self.engine._headers({"Accept": "text/event-stream"}))
            resp = conn.getresponse()
            if resp.status != 200:
                resp.read()
                if resp.status == 404:
                    self.missing_until = time.monotonic() + 60.0
                return False
            data: list[str] = []
            idle_since = None
            while True:
                raw = resp.readline()
                if not raw:
                    return True
                line = raw.decode("utf-8", "replace").rstrip("\r\n")
                if line == "":
                    if data:
                        try:
                            obj = json.loads("\n".join(data))
                            if isinstance(obj, dict):
                                self._take(obj)
                        except ValueError:
                            pass
                        data = []
                    if self.wanted():
                        idle_since = None
                    else:
                        idle_since = idle_since or time.monotonic()
                        if time.monotonic() - idle_since > self.IDLE_STOP_S:
                            return True
                    continue
                if line.startswith("data:"):
                    data.append(line[5:].lstrip(" "))
        except (OSError, http.client.HTTPException):
            return False
        finally:
            conn.close()

    def _poll_once(self) -> None:
        st, obj = self.engine.get_json("/v1/dashboard/live?follow=0", timeout=3.0)
        if st == 200 and isinstance(obj, dict):
            self._take(obj)
        elif st == 404:
            self.missing_until = time.monotonic() + 60.0
