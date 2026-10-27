"""The live log stream and the no-content rule, on a CPU; the SSE parts over real sockets.

the Gherkin is the matrix: the log file unchanged by the tee, live follow, the level filter,
resume by Last-Event-ID, a slow reader bounded and told about its gap, four streams at most, no
request text in the log through every error path we know, and the `[body]` keys line. Also: a
traceback is one entry, the heartbeat, `follow=0`, and every SSE payload against the contract.

Run: python tests/test_logbuf.py
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import socket
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_app_loop import Req, serve  # noqa: E402  (first: the CPU environment)
from test_usage import UTok, _ids, _script  # noqa: E402

from http.server import ThreadingHTTPServer  # noqa: E402

from server import app, auth, logbuf  # noqa: E402
from tools import contract_check  # noqa: E402

SENTINEL = "qqz-SENTINEL-4d2e88-content"
ADMIN = "adm-" + "5" * 40


class Captured:
    """Sys.stdout and sys.stderr teed into a fresh LogBuffer over two StringIOs, for a block."""

    def __init__(self):
        self.buf = logbuf.LogBuffer()
        self.out, self.err = io.StringIO(), io.StringIO()

    def __enter__(self):
        app.STATE["auth"] = auth.Auth(ADMIN)
        self._old = sys.stdout, sys.stderr
        sys.stdout = logbuf.Tee(self.out, self.buf, "stdout")
        sys.stderr = logbuf.Tee(self.err, self.buf, "stderr")
        app.STATE["log_buffer"] = self.buf
        return self

    def __exit__(self, *exc):
        sys.stdout, sys.stderr = self._old
        return False

    def text(self) -> str:
        return self.out.getvalue() + self.err.getvalue() + json.dumps(list(self.buf.ring))


def _server():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), app.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, httpd.server_address[1]


def _open_stream(port, query="", headers=None, rcvbuf=None):
    s = socket.create_connection(("127.0.0.1", port), timeout=10)
    if rcvbuf:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, rcvbuf)
    headers = dict({"Authorization": f"Bearer {ADMIN}"}, **(headers or {}))
    extra = "".join(f"{k}: {v}\r\n" for k, v in headers.items())
    s.sendall(f"GET /v1/dashboard/logs{query} HTTP/1.1\r\nHost: x\r\n{extra}\r\n".encode())
    return s


def _read_events(s, until, timeout=5.0):
    """Parse SSE from a socket until `until(events)` is true or the timeout passes."""
    s.settimeout(0.2)
    raw, events, end = b"", [], time.time() + timeout
    while time.time() < end and not until(events):
        try:
            chunk = s.recv(65536)
        except socket.timeout:
            continue
        if not chunk:
            break
        raw += chunk
        body = (raw.partition(b"\r\n\r\n")[2] if raw.startswith(b"HTTP/") else raw
                ).decode("utf-8", "replace")
        events = []
        for block in body.split("\n\n")[:-1]:        # the last piece may still be arriving
            if block.startswith(":"):
                events.append({"event": "ping"})
                continue
            ev = dict(line.split(": ", 1) for line in block.split("\n") if ": " in line)
            if "data" in ev:
                events.append({"event": ev.get("event"), "id": ev.get("id"),
                               "data": json.loads(ev["data"])})
    return events, raw


def _logs(events):
    return [e["data"] for e in events if e.get("event") == "log"]


# ------------------------------------------------------------------ the tee

def test_the_log_file_is_unchanged():
    buf = logbuf.LogBuffer()
    plain, teed = io.StringIO(), io.StringIO()
    t = logbuf.Tee(teed, buf, "stdout")
    pieces = ["[req] chatcmpl-0123456789abcdef01234567 stream prompt=5 completion=3 ",
              "finish=stop 12 ms 99.00 tok/s\n", "[server] one\n[cache] put declined: x\n",
              "partial", " line\n", "no newline yet"]
    for p in pieces:
        plain.write(p)
        t.write(p)
    # whole lines only until the flush (the trailing partial waits for its line), then every byte
    assert teed.getvalue() == plain.getvalue()[:plain.getvalue().rfind("\n") + 1]
    t.flush()
    assert teed.getvalue() == plain.getvalue(), "the tee changed the bytes"
    got = list(buf.ring)
    assert [e["source"] for e in got] == ["req", "server", "cache", "stdout"], got
    assert got[0]["level"] == "info" and got[0]["request_id"] == "chatcmpl-0123456789abcdef01234567"
    assert got[2]["level"] == "warning" and got[3]["msg"] == "partial line"
    assert all(not contract_check.check(e, "logs-line") for e in got)


def test_a_line_is_never_split_by_the_other_stream():
    # Under `python -u` a print() is two writes, the text and then "\n". stdout and stderr share
    # one log file (start.sh's `>"$LOG" 2>&1`), so an access line printed by a poller's thread
    # between the two ended up INSIDE the [req] line (box, 2026-09-24 23:26: `...4x1127.0.0.1 -
    # "GET /v1/dashboard/summary HTTP/1.1" 200 -`), and rowlog could not parse it. The tees
    # write whole lines only.
    buf = logbuf.LogBuffer()
    shared = io.StringIO()
    out = logbuf.Tee(shared, buf, "stdout")
    err = logbuf.Tee(shared, buf, "stderr")
    req = "[req] chatcmpl-0123456789abcdef01234567 stream finish=stop accept=7:0x5,4x1"
    out.write(req)
    err.write('127.0.0.1 - "GET /v1/dashboard/summary HTTP/1.1" 200 -\n')
    out.write("\n")
    lines = shared.getvalue().split("\n")
    assert req in lines, shared.getvalue()
    assert '127.0.0.1 - "GET /v1/dashboard/summary HTTP/1.1" 200 -' in lines, shared.getvalue()
    assert [e["msg"] for e in buf.ring if e["source"] == "req"] == [req]
    # a partial line with an explicit flush (print(..., end="", flush=True)) still goes out now
    out.write("loading ")
    out.flush()
    assert shared.getvalue().endswith("loading ")


def test_a_traceback_is_one_entry_and_levels_come_from_the_lines():
    buf = logbuf.LogBuffer()
    t = logbuf.Tee(io.StringIO(), buf, "stderr")
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        import traceback
        traceback.print_exc(file=t)
    t.write("[req] cmpl-0123456789abcdef01234567 json prompt=1 completion=0 finish=error 3 ms "
            "0.00 tok/s  !! RuntimeError: boom\n")
    t.write("[req] c stream prompt=1 completion=9 finish=abandoned 3 ms 1.00 tok/s\n")
    t.write('127.0.0.1 - "GET /health HTTP/1.1" 200 -\n')
    t.write("[ledger] write failed, 1 rows lost: OperationalError\n")
    got = list(buf.ring)
    assert got[0]["source"] == "traceback" and got[0]["level"] == "error"
    assert got[0]["msg"].count("\n") >= 2 and got[0]["msg"].endswith("RuntimeError: boom")
    assert [(e["level"], e["source"]) for e in got[1:]] == [
        ("error", "req"), ("warning", "req"), ("debug", "http"), ("error", "ledger")]


# ------------------------------------------------------------------ the stream

def test_live_follow_level_filter_and_resume():
    serve()
    app.STATE["tok"] = UTok()
    httpd, port = _server()
    real = app.generate_stream
    app.generate_stream = _script(_ids("hello there"))
    try:
        with Captured() as cap:
            print("[server] before the subscriber", flush=True)
            s = _open_stream(port, "?level=info")
            events, raw = _read_events(s, lambda ev: len(_logs(ev)) >= 1)
            assert raw.startswith(b"HTTP/1.1 200") and b"text/event-stream" in raw
            assert _logs(events)[0]["msg"] == "[server] before the subscriber", events
            t0 = time.time()
            head, body = Req("/v1/chat/completions",
                             {"messages": [{"role": "user", "content": "q"}]}).response()
            events, _ = _read_events(s, lambda ev: any(e["source"] == "req" for e in _logs(ev)))
            got = [e for e in _logs(events) if e["source"] == "req"]
            assert got and time.time() - t0 < 1.0, events
            assert got[0]["request_id"] and json.loads(body)["id"] == got[0]["request_id"]
            for e in _logs(events):
                assert not contract_check.check(e, "logs-line"), e
            s.close()

            # the level filter
            print("[server] plain info", flush=True)
            print("[cache] put declined: kv.length=9 ctx=8", flush=True)
            print("[req] x json prompt=1 completion=0 finish=error 1 ms 0.00 tok/s", flush=True)
            s = _open_stream(port, "?level=warning&backlog=50")
            events, _ = _read_events(s, lambda ev: len(_logs(ev)) >= 2)
            assert {e["level"] for e in _logs(events)} <= {"warning", "error"}, _logs(events)
            s.close()

            # resume: everything after seq N, once
            for i in range(30):
                print(f"[server] line {i}", flush=True)
            n = cap.buf.seq - 10
            s = _open_stream(port, "?backlog=5000", {"Last-Event-ID": str(n)})
            events, _ = _read_events(s, lambda ev: len(_logs(ev)) >= 10)
            seqs = [e["seq"] for e in _logs(events)]
            assert seqs == list(range(n + 1, cap.buf.seq + 1)), seqs
            ids = [int(e["id"]) for e in events if e.get("event") == "log"]
            assert ids == seqs
            s.close()
    finally:
        app.generate_stream = real
        httpd.shutdown()


def test_the_heartbeat_and_follow_zero():
    serve()
    app.STATE["log_ping_s"] = 0.2
    httpd, port = _server()
    try:
        with Captured():
            print("[server] something", flush=True)
            s = _open_stream(port, "?level=debug")
            events, _ = _read_events(s, lambda ev: sum(e["event"] == "ping" for e in ev) >= 2)
            assert sum(e["event"] == "ping" for e in events) >= 2, events
            s.close()
            head, body = _get("/v1/dashboard/logs?follow=0&backlog=10")
            assert head.startswith("HTTP/1.1 200"), head
            assert not contract_check.check(body, "logs-json"), contract_check.check(body, "logs-json")
            assert body["lines"][-1]["msg"] == "[server] something" and body["last_seq"] >= 1
            for q in ("?level=loud", "?backlog=9999", "?grep=" + "x" * 129, "?since=yesterday"):
                head, body = _get("/v1/dashboard/logs" + q)
                assert head.startswith("HTTP/1.1 400") and body["error"]["type"] == "bad_request", q
    finally:
        httpd.shutdown()


def _get(path):
    req = Req(path, {})
    req.headers["Authorization"] = f"Bearer {ADMIN}"
    req.command = "GET"
    req.do_GET()
    head, _, body = req.wfile.getvalue().partition(b"\r\n\r\n")
    return head.decode(), json.loads(body)


def test_a_slow_subscriber_is_bounded_and_told_about_the_gap():
    serve()
    httpd, port = _server()
    try:
        with Captured() as cap:
            s = _open_stream(port, "?backlog=0", rcvbuf=4096)
            time.sleep(0.3)                            # subscribed; now it stops reading
            line = "[server] " + "x" * 200
            for _ in range(20_000):
                print(line)
            sub = cap.buf.subs[0]
            assert len(sub.q) <= logbuf.SUB_QUEUE and len(cap.buf.ring) <= logbuf.RING
            events, _ = _read_events(s, lambda ev: any(e.get("event") == "gap" for e in ev),
                                     timeout=20.0)
            gaps = [e["data"]["dropped"] for e in events if e.get("event") == "gap"]
            assert gaps and gaps[0] > 0, events[-3:]
            assert not contract_check.check({"dropped": gaps[0]}, "gap")
            s.close()
    finally:
        httpd.shutdown()


def test_at_most_four_streams():
    serve()
    httpd, port = _server()
    try:
        with Captured() as cap:
            open_ = [_open_stream(port) for _ in range(4)]
            end = time.time() + 5
            while cap.buf.subscribers < 4 and time.time() < end:
                time.sleep(0.05)
            fifth = _open_stream(port)
            fifth.settimeout(5)
            raw = b""
            while b"\r\n\r\n" not in raw or len(raw.partition(b"\r\n\r\n")[2]) < int(
                    raw.split(b"Content-Length: ")[1].split(b"\r\n")[0]):
                raw += fifth.recv(4096)
            assert raw.startswith(b"HTTP/1.1 429"), raw[:200]
            assert json.loads(raw.partition(b"\r\n\r\n")[2])["error"]["type"] == "too_many"
            for s in open_:
                s.close()
            print("[server] wake the streams so they notice", flush=True)
            end = time.time() + 5
            while cap.buf.subscribers and time.time() < end:
                print("[server] tick", flush=True)
                time.sleep(0.1)
            assert cap.buf.subscribers == 0, "closed readers are cleaned up"
            assert "Traceback" not in cap.err.getvalue(), "no BrokenPipe traceback"
    finally:
        httpd.shutdown()


def test_a_closed_stream_frees_its_place_at_once():
    """The follow-up. A closed tab's stream kept its place under the four-stream cap
    until the handler next WROTE -- the heartbeat, 15 s later, and a write to a closed socket does
    not fail before the second one -- so reopening the Dev tab a few times in a row got 429 'at most
    4 log streams' (nine in the engine log, 2026-09-25 02:0x). A quiet log is the case: nothing is
    printed here, and the heartbeat is 30 s. Four streams close and four open at once, three times
    over: every one is served, not refused; then the last four close and their places are free
    within two seconds with nobody asking for them."""
    serve()
    app.STATE["log_ping_s"] = 30.0
    httpd, port = _server()

    def status(s):
        s.settimeout(5)
        head = b""
        while b"\r\n" not in head:
            head += s.recv(4096)
        return head.split(b"\r\n")[0].decode()

    try:
        with Captured() as cap:
            streams = []
            for _ in range(4):
                for s in streams:
                    s.close()
                streams = [_open_stream(port, "?backlog=0") for _ in range(4)]
                codes = [status(s) for s in streams]
                assert codes == ["HTTP/1.1 200 OK"] * 4, codes
            for s in streams:
                s.close()
            t0 = time.time()
            while cap.buf.subscribers and time.time() - t0 < 5:
                time.sleep(0.02)
            freed = time.time() - t0
            assert cap.buf.subscribers == 0 and freed < 2.0, (cap.buf.subscribers, freed)
            assert "Traceback" not in cap.err.getvalue(), "no BrokenPipe traceback"
    finally:
        app.STATE.pop("log_ping_s", None)
        httpd.shutdown()


# ------------------------------------------------------------------ no content

class TemplateBoom(UTok):
    def apply_chat_template(self, messages, **kw):
        raise ValueError(f"cannot render message {messages[-1]['content']!r}")


def test_no_content_in_the_log_by_default():
    body = {"messages": [{"role": "user", "content": f"please {SENTINEL} now"}], "stream": False,
            "tools": [{"type": "function", "function": {"name": "f", "description": SENTINEL}}]}

    def engine_boom(*a, **k):
        yield 65
        raise RuntimeError(f"the engine choked on {SENTINEL}")

    def parser_boom(text):
        raise ValueError(f"bad call in {SENTINEL}")

    for log_content in (False, True):
        logbuf.CONFIG["log_content"] = log_content
        try:
            with Captured() as cap:
                # the template
                serve()
                app.STATE["tok"] = TemplateBoom()
                head, _ = Req("/v1/chat/completions", body).response()
                assert head.startswith("HTTP/1.1 500"), head
                # the engine, both paths
                for stream in (False, True):
                    serve()
                    app.STATE["tok"] = UTok()
                    real = app.generate_stream
                    app.generate_stream = engine_boom
                    try:
                        Req("/v1/chat/completions", dict(body, stream=stream)).response()
                    finally:
                        app.generate_stream = real
                # the tool-call parser
                serve()
                app.STATE["tok"] = UTok()
                real_gen, real_parse = app.generate_stream, app.parse_tool_calls
                app.generate_stream = _script(_ids("x"))
                app.parse_tool_calls = parser_boom
                try:
                    Req("/v1/chat/completions", body).response()
                finally:
                    app.generate_stream, app.parse_tool_calls = real_gen, real_parse
                # a body that is not JSON
                req = Req("/v1/chat/completions", {})
                req.rfile = io.BytesIO(f'{{"messages": "{SENTINEL}'.encode())
                req.headers["Content-Length"] = str(len(req.rfile.getvalue()))
                req.response()
            text = cap.text()
            assert "Traceback" in text and "RuntimeError" in text, "the errors were logged"
            if log_content:
                assert SENTINEL in text, "the test can see content when it is let through"
            else:
                assert SENTINEL not in text, [l for l in text.splitlines() if SENTINEL in l][:3]
                assert "message withheld" in text
        finally:
            logbuf.CONFIG["log_content"] = False


def test_the_request_keys_line():
    serve()
    app.STATE.update(tok=UTok(), log_request_keys=True)
    real = app.generate_stream
    app.generate_stream = _script(_ids("ok"))
    try:
        with Captured() as cap:
            Req("/v1/chat/completions", {
                "model": "qwen38-spark-engine", "stream": True, "max_tokens": 64,
                "temperature": 0.7, "stream_options": {"include_usage": True},
                "messages": [{"role": "user", "content": SENTINEL}], "user": SENTINEL,
                "stop": [SENTINEL], "tools": [{"type": "function", "function": {"name": SENTINEL}}],
                "tool_choice": {"type": "function", "function": {"name": SENTINEL}},
                "chat_template_kwargs": {"enable_thinking": False}}).response()
    finally:
        app.generate_stream = real
        app.STATE["log_request_keys"] = False
    lines = [e for e in cap.buf.ring if e["source"] == "body"]
    assert len(lines) == 1, list(cap.buf.ring)
    msg = lines[0]["msg"]
    assert SENTINEL not in msg, msg
    for part in ('"max_tokens": 64', '"temperature": 0.7', '"include_usage": true',
                 '"tool_choice": "function"', '"enable_thinking": false', "keys=chat_template_kwargs",
                 '"messages": 1', '"tools": 1', "user"):
        assert part in msg, (part, msg)
    assert lines[0]["request_id"] is not None


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:64s} ok")
            passed += 1
    print(f"{passed} passed")
