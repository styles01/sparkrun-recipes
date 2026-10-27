"""One request through to a backend and back, streamed or not, measured on the way.

THE BYTES. The response's bytes go to the client unchanged. Beside them the adapter reads every
SSE event for the ledger. When the client did not ask for `stream_options.include_usage`, the
proxy asks for it (unless the backend's `inject_usage` is off) and drops that one usage-only frame
before the client, so the client gets what the backend sent, less the frame it did not ask for.
(OpenAI then also writes `"usage": null` into each chunk; that is the backend's byte, passed on.)
The request body goes upstream as the client sent it unless the model id or the usage request
changes it; then it is the client's JSON with those fields set.

TIMEOUTS per backend (local: connect 5 s, first byte 600 s, idle gap 120 s; cloud: 10, 120, 60),
each its own `finish_reason`: `timeout_connect`, `timeout_first_byte`, `timeout_idle`.

RETRIES only before any byte went to the client: a connect error, a 429, 502 or 503. At most 2
across the route's fallbacks. A fallback is tried at once (another backend); the same backend is
tried again after 1 s, then 2 s, with jitter, or after its `Retry-After` up to 30 s. After a byte
went out a failure is an `error` frame to the client, then `[DONE]`. A 429 that survives the tries
goes to the client with its `Retry-After`.

THE COST, FOR A CLIENT THAT ASKS. A streamed request with `X-Tandem-Cost: 1` (chatd sends it) gets
one SSE comment just before `data: [DONE]`: `: tandem-cost {"cost_micros": N, "backend": id}`, the
price of the request (every request the harness sent for it) as its ledger row will have it, N
null without a price. A comment is ignored by every SSE reader; without the header the bytes are
the backend's.

THE CLIENT LEAVES: the upstream connection is closed within a quarter second (the backend stops
generating), and the row says `abandoned`.

CONCURRENCY: a thread per request (the server's), no lock held across a request, and per backend
at most `max_inflight` requests at once when the user set one (none by default: the proxy adds no
limit, a backend takes what it accepts): past it the request goes to a fallback when one has room,
else a 503 with `Retry-After: 1`. The engine's queue, lock and `max_queue` are never
touched.
"""

from __future__ import annotations

import json
import random
import re
import select
import socket
import threading
import time

from proxy import http1
from proxy.adapters import ENDPOINTS, Stamps
from proxy.measure import RelayRecord

RETRYABLE = (429, 502, 503)
MAX_RETRIES = 2
RETRY_AFTER_CAP = 30.0
# response headers passed to the client (besides Content-Type)
PASS_HEADERS = ("retry-after", "x-request-id", "openai-processing-ms")
PASS_PREFIXES = ("x-ratelimit-",)


_DONE = re.compile(rb"(?:^|\n)data: ?\[DONE\]")


class CostTap:
    """The client's stream writer, with the request's cost (`cost()`, bytes) written once just
    before the first `data: [DONE]` (see THE COST above)."""

    def __init__(self, w, cost):
        self.w, self.cost, self.sent = w, cost, False

    def write(self, data: bytes):
        if not self.sent:
            m = _DONE.search(data)
            if m is not None:
                self.sent = True
                i = m.start() + (1 if data[m.start():m.start() + 1] == b"\n" else 0)
                data = data[:i] + self.cost() + data[i:]
        return self.w.write(data)

    def flush(self) -> None:
        self.w.flush()


class Inflight:
    """Per backend: the requests relayed now, and the cap."""

    def __init__(self):
        self._n: dict[str, int] = {}
        self._lock = threading.Lock()

    def take(self, bid: str, cap: int | None) -> bool:
        """Count one more request on `bid`; False when a cap is set and reached. None: no cap."""
        with self._lock:
            n = self._n.get(bid, 0)
            if cap is not None and n >= cap:
                return False
            self._n[bid] = n + 1
            return True

    def give(self, bid: str) -> None:
        with self._lock:
            self._n[bid] = max(0, self._n.get(bid, 0) - 1)

    def count(self, bid: str) -> int:
        return self._n.get(bid, 0)

    def total(self) -> int:
        return sum(self._n.values())


def client_gone(handler) -> bool:
    """The client's socket reads as end of file (it left). No socket: it stays."""
    sock = getattr(handler, "connection", None)
    if sock is None:
        return False
    try:
        r, _, _ = select.select([sock], [], [], 0)
        if not r:
            return False
        return not sock.recv(1, socket.MSG_PEEK)
    except (OSError, ValueError):
        return True


def _error_body(message: str, kind: str, code=None) -> dict:
    err = {"message": message, "type": kind}
    if code is not None:
        err["code"] = code
    return {"error": err}


class Relay:
    """The relay of one proxy: its backends' pools, in-flight counts and the record sink."""

    def __init__(self, service):
        self.svc = service                 # proxy.app.Service: config, table, adapters, account
        self.inflight = Inflight()
        self._pools: dict[str, tuple] = {}
        self._lock = threading.Lock()
        self.sleep = time.sleep

    def pool(self, backend: dict) -> http1.Pool:
        key = (backend["url"], backend["timeouts"]["connect"])
        with self._lock:
            got = self._pools.get(backend["id"])
            if got is None or got[0] != key:
                if got is not None:
                    got[1].close()
                got = (key, http1.Pool(backend["url"], backend["timeouts"]["connect"]))
                self._pools[backend["id"]] = got
            return got[1]

    def drop_pool(self, bid: str) -> None:
        with self._lock:
            got = self._pools.pop(bid, None)
        if got is not None:
            got[1].close()

    # ----------------------------------------------------------------- the request
    def serve(self, handler, path: str, raw: bytes, body: dict, route, rec: RelayRecord) -> None:
        """Relay one request. Writes the answer on `handler` and fills `rec`; the caller writes
        the ledger row."""
        stream = bool(body.get("stream")) and path != "/v1/embeddings"
        rec.public_model = route.public
        cands = route.candidates()
        tries = 0
        last = None                         # (status, reason, headers, body bytes) to relay
        first_bid = cands[0][0]
        i = 0
        while i < len(cands):
            bid, upstream_model = cands[i]
            backend = self.svc.table.backend_for(bid)
            if backend is None:
                i += 1
                continue
            adapter = self.svc.adapter(backend)
            if path not in adapter.routes():
                last = (404, "Not Found", [], json.dumps(_error_body(
                    f"backend '{bid}' ({backend['kind']}) does not serve {path}",
                    "invalid_request_error", "unsupported_route")).encode())
                i += 1
                continue
            if not self.inflight.take(bid, backend.get("max_inflight")):
                rec.upstream_status = None
                last = (503, "Service Unavailable", [("Retry-After", "1")], json.dumps(
                    _error_body(f"backend '{bid}' has {backend['max_inflight']} requests in "
                                f"flight (max_inflight)", "server_busy", "max_inflight")).encode())
                i += 1
                continue
            try:
                outcome = self._attempt(handler, path, raw, body, backend, adapter,
                                        upstream_model, stream, rec)
            finally:
                self.inflight.give(bid)
            rec.backend, rec.backend_kind, rec.backend_model = bid, backend["kind"], upstream_model
            if bid != first_bid:
                rec.fallback_from = first_bid
            if outcome[0] == "done":
                rec.retries = tries
                return
            # a failure before any byte reached the client
            last = outcome[1]
            retry_after = outcome[2]
            if tries >= MAX_RETRIES:
                break
            tries += 1
            self.svc.count_retry(bid, outcome[3])
            if i + 1 < len(cands):
                i += 1                       # a fallback, at once
                continue
            wait = retry_after if retry_after is not None else (1.0 if tries == 1 else 2.0)
            wait = min(RETRY_AFTER_CAP, wait) * (1.0 + random.uniform(0, 0.25))
            if client_gone(handler):
                rec.finish_reason = "abandoned"
                rec.status = 499
                rec.retries = tries
                return
            self.sleep(wait)
        rec.retries = tries
        status, reason, headers, data = last or (502, "Bad Gateway", [], json.dumps(
            _error_body("no backend could take the request", "upstream_error")).encode())
        rec.status = status
        if rec.finish_reason is None:
            rec.finish_reason = "refused" if status in (429, 503) else "error"
        if rec.backend is None:
            rec.backend = first_bid
            b = self.svc.table.backend_for(first_bid)
            rec.backend_kind = b["kind"] if b else None
            rec.backend_model = cands[0][1]
        self._write_plain(handler, status, headers, data, rec)

    def _attempt(self, handler, path, raw, body, backend, adapter, upstream_model, stream, rec):
        """One try on one backend. ("done",) when the client got an answer (good or not);
        ("retry", (status, reason, headers, bytes), retry_after, reason) when nothing went out
        and the failure may be tried again."""
        try:
            key = self.svc.cfg.key_for(backend["id"])
        except Exception as exc:                                   # noqa: BLE001 (SecretError)
            rec.error_type = "key_unreadable"
            return ("retry", (502, "Bad Gateway", [], json.dumps(_error_body(
                f"backend '{backend['id']}': {exc}", "upstream_error", "key_unreadable"))
                .encode()), None, "key")
        hx = getattr(rec, "hx", None)
        if hx is not None:
            # the harness (P4): the notes, the thinking switch, the side cap, the compaction
            # section for this backend's kind; the client's bytes when nothing changes
            body2, changed = hx.upstream(body, adapter, upstream_model)
            if changed:
                body, raw = body2, json.dumps(body2, ensure_ascii=False).encode()
        up_body, injected = adapter.prepare(path, raw, body, upstream_model)
        headers = adapter.headers(key) + adapter.forward(handler.headers)
        if hx is not None:
            headers = hx.headers(adapter, headers)
        if backend.get("kind") == "tandemllm":
            # the parallel view: our engine files the request under this id, so the proxy's lane
            # reads the engine's own live count, prefill and state (server/parallel.py peers)
            headers = headers + [("X-Tandem-Relay-Id", rec.request_id)]
            if getattr(rec, "user", None):
                # 2026-10-05: whose memory the engine reads and writes (server/app.py
                # `_memory_user` takes it from a trusted proxy address only)
                headers = headers + [("X-Tandem-User", str(rec.user))]
        collect = getattr(self.svc, "collect", None)
        if collect is not None:                     # P6: an id the engine's plugin reports back
            up_body, headers = collect.tag(backend, path, body, up_body, headers, rec)
        pool = self.pool(backend)
        upath = pool.path(path)
        to = backend["timeouts"]
        poll = (lambda: client_gone(handler))
        conn = None
        for fresh in (False, True):
            try:
                conn = pool.get(fresh=fresh)
                rec.connect_ms += conn.connect_ms
                conn.send("POST", upath, headers, up_body)
                if rec.t_sent is None:
                    rec.stamp_sent()
                status, reason, uh = conn.read_head(float(to["first_byte"]), poll)
                break
            except http1.Stale:
                continue
            except http1.Aborted:
                if conn is not None:
                    conn.close()
                rec.finish_reason, rec.status = "abandoned", 499
                rec.client_gone_at = time.perf_counter()
                return ("done",)
            except http1.Timeout as t:
                rec.error_type = f"timeout_{t.which}"
                kind = "timeout_" + t.which
                if t.which == "connect":
                    return ("retry", (504, "Gateway Timeout", [], json.dumps(_error_body(
                        f"backend '{backend['id']}' did not accept a connection in "
                        f"{to['connect']} s", "upstream_timeout", kind)).encode()), None, "connect")
                rec.finish_reason = "timeout"
                self._write_plain(handler, 504, [], json.dumps(_error_body(
                    f"backend '{backend['id']}' sent nothing in {to['first_byte']} s",
                    "upstream_timeout", kind)).encode(), rec)
                return ("done",)
            except http1.UpstreamError as exc:
                rec.error_type = "connect_error"
                return ("retry", (502, "Bad Gateway", [], json.dumps(_error_body(
                    f"backend '{backend['id']}': {exc}", "upstream_error", "connect_error"))
                    .encode()), None, "connect")
        else:
            return ("retry", (502, "Bad Gateway", [], json.dumps(_error_body(
                f"backend '{backend['id']}' closed the connection", "upstream_error"))
                .encode()), None, "connect")
        rec.upstream_status = status
        hmap = {k.lower(): v for k, v in uh}
        if status in RETRYABLE:
            try:
                data = conn.read_all(float(to["idle"]), poll)
                pool.put(conn)
            except (http1.Timeout, http1.Closed, http1.UpstreamError, http1.Aborted):
                data = b""
                conn.close()
            ra = None
            try:
                ra = float(hmap.get("retry-after", ""))
            except ValueError:
                ra = None
            keep = [(k, v) for k, v in uh if _passed(k)]
            ctype = hmap.get("content-type")
            if ctype:
                keep.append(("Content-Type", ctype))
            return ("retry", (status, reason, keep, data), ra, str(status))
        sse = "text/event-stream" in hmap.get("content-type", "")
        if stream and sse and status == 200:
            self._stream(handler, path, conn, pool, uh, adapter, injected, backend, rec, poll,
                         headers)
        else:
            self._whole(handler, path, conn, pool, status, uh, adapter, backend, rec, poll)
        return ("done",)

    # ----------------------------------------------------------------- answers
    def _start(self, handler, status: int, uh, backend: dict, stream: bool, length=None) -> None:
        handler._px_written = True
        handler.send_response(status)
        ctype = next((v for k, v in uh if k.lower() == "content-type"), "application/json")
        handler.send_header("Content-Type", ctype)
        for k, v in uh:
            if _passed(k):
                handler.send_header(k, v)
        handler.send_header("X-Tandem-Backend", backend["id"])
        if stream:
            handler.send_header("Cache-Control", "no-cache")
            handler.send_header("X-Accel-Buffering", "no")
            handler.send_header("Connection", "close")
            handler.close_connection = True
        else:
            handler.send_header("Content-Length", str(length or 0))
        handler.end_headers()

    def _stream(self, handler, path, conn, pool, uh, adapter, injected, backend, rec, poll,
                headers=None):
        st = Stamps()
        to = backend["timeouts"]
        hx = getattr(rec, "hx", None)
        self._start(handler, 200, uh, backend, True)
        w = handler.wfile
        if (handler.headers.get("X-Tandem-Cost") or "").strip() == "1":
            w = CostTap(w, lambda: self._cost_line(st, adapter, path, backend, rec))
        try:
            self._pump(conn, path, adapter, st, injected, rec, w, to, poll, hx)
            if hx is not None:
                self._settle(handler, path, backend, adapter, st, rec, w, to, poll, hx, pool,
                             conn, headers)
            else:
                pool.put(conn)
            rec.finish_reason = _finish(st)
            if hx is not None and hx.stop == "argloop":
                rec.finish_reason = "stop"
            if st.error:
                rec.error_type = "upstream_stream_error"
        except http1.Aborted:
            conn.close()
            rec.finish_reason = "abandoned"
            rec.client_gone_at = time.perf_counter()
        except (BrokenPipeError, ConnectionResetError, OSError):
            conn.close()                       # the client's socket failed: it left
            rec.finish_reason = "abandoned"
            rec.client_gone_at = time.perf_counter()
        except http1.Timeout as t:
            conn.close()
            rec.finish_reason = "timeout"
            rec.error_type = f"timeout_{t.which}"
            self._stream_error(w, f"backend '{backend['id']}' sent nothing for {to['idle']} s",
                               rec.error_type)
        except (http1.Closed, http1.UpstreamError):
            conn.close()
            rec.finish_reason = "error"
            rec.error_type = "upstream_closed"
            self._stream_error(w, f"backend '{backend['id']}' closed the stream early",
                               "upstream_closed")
        self._finish(st, adapter, path, backend, rec)
        rec.thinking = st.reasoning
        rec.tool_calls = len(st.tools)

    def _pump(self, conn, path, adapter, st, injected, rec, w, to, poll, hx) -> bool:
        """Copy one upstream stream to the client, event by event, the adapter reading each and
        the harness (when it runs) deciding what goes out now. True when it ended on its own;
        False when the harness stopped it (the connection is then closed)."""
        buf = bytearray()
        for piece in conn.body(float(to["idle"]), poll):
            t_arr = time.perf_counter()
            buf += piece
            out = bytearray()
            while True:
                end, sep = _event_end(buf)
                if end < 0:
                    break
                event = bytes(buf[:end + sep])
                del buf[:end + sep]
                token, drop = adapter.read_frame(path, event[:end], st, injected)
                if token:
                    rec.stamp_token(t_arr, st.kind, st.tool_name, st.tool_chars)
                if hx is None:
                    if not drop:
                        out += event
                else:
                    out += hx.feed(event, event[:end], drop)
                    if hx.stop is not None:
                        break
            if out:
                w.write(out)
                w.flush()
                now = time.perf_counter()
                if rec.t_first_out is None and rec.t_first_frame is not None:
                    rec.t_first_out = now
            if hx is not None and hx.stop is not None:
                conn.close()                   # the backend stops generating
                return False
        if buf and (hx is None or hx.stop is None):
            if hx is None:
                w.write(bytes(buf))            # a last event without its blank line
            else:
                w.write(hx.feed(bytes(buf), bytes(buf), False))
            w.flush()
        return True

    def _settle(self, handler, path, backend, adapter, st, rec, w, to, poll, hx, pool, conn,
                headers) -> None:
        """The harness's end of a stream: the first request's usage counted, then each second
        request it asks for (E1, a regenerate, the close of a reasoning loop) relayed as the
        continuation of the first answer, then what it still owes the client."""
        if hx.stop is None:
            pool.put(conn)
        hx.usages.append(adapter.read_usage(st.usage, path) if st.usage is not None else None)
        n = 0
        while n < 4:
            body = hx.follow_up()
            pending = hx._lines()
            if pending:
                w.write(pending)
                w.flush()
            if body is None:
                break
            n += 1
            st2 = Stamps()
            ok = self._follow(path, backend, adapter, body, st2, rec, w, to, poll, hx, headers)
            if not ok:
                hx.follow_failed("upstream")
                break
            u = adapter.read_usage(st2.usage, path) if st2.usage is not None else None
            hx.usages.append(u)
            cost = adapter.price(u, self.svc.price_of(backend["id"], rec.backend_model or
                                                      rec.public_model), rec.ts) if u else None
            hx.extra_done(u, cost)
            if st2.tools:
                st.tools |= {("f", n, i) for i in st2.tools}
            if st2.reasoning:
                st.reasoning = True
            st.finish_reason = st2.finish_reason or st.finish_reason
            st.error = st2.error or st.error
        w.write(hx.tail())
        w.flush()

    def _follow(self, path, backend, adapter, body, st, rec, w, to, poll, hx, headers) -> bool:
        """One second request, streamed through the harness as the continuation of the first
        answer. False when it could not be sent or did not answer 200 with a stream."""
        try:
            key = self.svc.cfg.key_for(backend["id"])
        except Exception:                                          # noqa: BLE001
            return False
        body = dict(body, stream=True)
        raw = json.dumps(body, ensure_ascii=False).encode()
        up_body, injected = adapter.prepare(path, raw, body, rec.backend_model or
                                            body.get("model"))
        hdrs = adapter.headers(key) + (headers[len(adapter.headers(key)):] if headers else [])
        pool = self.pool(backend)
        try:
            conn = pool.get()
            conn.send("POST", pool.path(path), hdrs, up_body)
            status, _, uh = conn.read_head(float(to["first_byte"]), poll)
        except (http1.UpstreamError, http1.Timeout, http1.Closed, http1.Aborted, OSError):
            return False
        hmap = {k.lower(): v for k, v in uh}
        if status != 200 or "text/event-stream" not in hmap.get("content-type", ""):
            try:
                conn.read_all(float(to["idle"]), poll)
                pool.put(conn)
            except Exception:                                      # noqa: BLE001
                conn.close()
            return False
        ended = self._pump(conn, path, adapter, st, injected, rec, w, to, poll, hx)
        if ended:
            pool.put(conn)
        return True

    def _stream_error(self, w, message: str, code: str) -> None:
        try:
            w.write(("data: " + json.dumps(_error_body(message, "upstream_error", code))
                     + "\n\ndata: [DONE]\n\n").encode())
            w.flush()
        except OSError:
            pass

    def _whole(self, handler, path, conn, pool, status, uh, adapter, backend, rec, poll):
        st = Stamps()
        to = backend["timeouts"]
        try:
            data = conn.read_all(float(to["idle"]), poll)
            pool.put(conn)
        except http1.Aborted:
            conn.close()
            rec.finish_reason, rec.status = "abandoned", 499
            rec.client_gone_at = time.perf_counter()
            return
        except http1.Timeout as t:
            conn.close()
            rec.finish_reason, rec.error_type = "timeout", f"timeout_{t.which}"
            self._write_plain(handler, 504, [], json.dumps(_error_body(
                f"backend '{backend['id']}' stopped sending for {to['idle']} s",
                "upstream_timeout", rec.error_type)).encode(), rec)
            return
        except (http1.Closed, http1.UpstreamError):
            conn.close()
            rec.finish_reason, rec.error_type = "error", "upstream_closed"
            self._write_plain(handler, 502, [], json.dumps(_error_body(
                f"backend '{backend['id']}' closed the connection early", "upstream_error",
                "upstream_closed")).encode(), rec)
            return
        ctype = next((v for k, v in uh if k.lower() == "content-type"), "")
        if "json" in ctype:
            try:
                adapter.read_body(path, json.loads(data), st)
            except ValueError:
                pass
        elif "event-stream" in ctype:
            # a stream the client did not ask for: read its frames for the counts
            for ev in data.replace(b"\r\n", b"\n").split(b"\n\n"):
                adapter.read_frame(path, ev, st, False)
        try:
            self._start(handler, status, uh, backend, False, len(data))
            handler.wfile.write(data)
            handler.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            rec.finish_reason = "abandoned"
        if status >= 400 and rec.finish_reason is None:
            rec.finish_reason = "error"
            rec.error_type = rec.error_type or f"upstream_{status}"
        if rec.finish_reason is None:
            rec.finish_reason = _finish(st)
        rec.status = status
        self._finish(st, adapter, path, backend, rec)
        rec.thinking = st.reasoning
        rec.tool_calls = len(st.tools)

    def _finish(self, st: Stamps, adapter, path, backend, rec) -> None:
        hx = getattr(rec, "hx", None)
        if hx is not None and hx.usages:
            # the harness's turn: every request it sent, summed (null when one did not report)
            u = adapter.reconcile(hx.total_usage() or adapter.read_usage(None, path), st)
        else:
            u = adapter.reconcile(adapter.read_usage(st.usage, path), st)
        rec.set_usage(u)
        rec.cost_micros = adapter.price(u, self.svc.price_of(backend["id"], rec.backend_model
                                                              or rec.public_model), rec.ts)
        rec.system_fingerprint = st.fingerprint

    def _cost_line(self, st: Stamps, adapter, path, backend, rec) -> bytes:
        """The cost comment: the usage so far (the harness's requests summed, else the stream's
        own) at the model's price, as `_finish` prices the ledger row."""
        hx = getattr(rec, "hx", None)
        if hx is not None and hx.usages:
            u = hx.total_usage() or adapter.read_usage(None, path)
        else:
            u = adapter.read_usage(st.usage, path)
        cost = adapter.price(u, self.svc.price_of(backend["id"], rec.backend_model
                                                  or rec.public_model), rec.ts)
        return (": tandem-cost " + json.dumps({"cost_micros": cost, "backend": backend["id"]})
                + "\n\n").encode()

    def _write_plain(self, handler, status: int, headers, data: bytes, rec) -> None:
        rec.status = status
        handler._px_written = True
        try:
            handler.send_response(status)
            ctype = next((v for k, v in headers if k.lower() == "content-type"),
                         "application/json")
            handler.send_header("Content-Type", ctype)
            for k, v in headers:
                if k.lower() != "content-type" and (_passed(k) or k.lower() == "retry-after"):
                    handler.send_header(k, v)
            handler.send_header("Content-Length", str(len(data)))
            handler.end_headers()
            handler.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError, OSError):
            handler.close_connection = True


FINISH = {"stop": "stop", "length": "length", "tool_calls": "tool_calls",
          "function_call": "tool_calls", "content_filter": "content_filter", "error": "error",
          "eos": "stop", "abort": "error"}


def _finish(st) -> str:
    """The ledger's finish reason from the backend's: OpenAI's set, `function_call` read as
    `tool_calls`; an unknown word is `stop` when the stream ended cleanly."""
    if st.error:
        return "error"
    return FINISH.get(st.finish_reason or "stop", "stop")


def _passed(name: str) -> bool:
    n = name.lower()
    return n in PASS_HEADERS or n.startswith(PASS_PREFIXES)


def _event_end(buf: bytearray) -> tuple[int, int]:
    """(index of the blank line that ends the first SSE event, its length), or (-1, 0)."""
    a = buf.find(b"\n\n")
    b = buf.find(b"\r\n\r\n")
    if a < 0 and b < 0:
        return -1, 0
    if b >= 0 and (a < 0 or b < a):
        return b, 4
    return a, 2


def endpoint_of(path: str) -> str:
    return ENDPOINTS.get(path, path.rsplit("/", 1)[-1])
