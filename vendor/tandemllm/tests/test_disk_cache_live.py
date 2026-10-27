"""tools/disk_cache_live.py, the post-deploy check of the disk tier, against a stub of the engine's
three routes (it runs on the box after the deploy and cannot run here against the real one):

  * the pass: the cold run writes the session head (the writer's event), the RAM caches are emptied,
    the next session's first request starts from the disk at that entry with the same text;
  * the text differing is a FAIL, a disk hit shorter than the written head is a FAIL;
  * TandemLLM's step lines (`› reading ...`, `› waiting: ...`), which differ between two identical
    answers, and the ids the server draws for tool calls are not the text: the 22:08 run of
    2026-10-02 failed `same_text` on them alone, and a call's arguments that differ
    still fail;
  * a server whose tier does not run, and a box whose lock is held, are refused before a request.

Run: python tests/test_disk_cache_live.py
"""

from __future__ import annotations

import fcntl
import json
import os
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools import disk_cache_live as L  # noqa: E402

HEAD = 23_552


class Stub:
    """The engine as the check sees it. `differ`: run 3 writes another text; `short`: run 3
    reuses less than the head; `running`: the tier runs."""

    def __init__(self, differ=False, short=False, running=True, failed=False, lines=False,
                 calls=False, args_differ=False):
        self.differ, self.short, self.running, self.failed = differ, short, running, failed
        self.lines, self.calls_on, self.args_differ = lines, calls, args_differ
        self.n_calls = 0
        self.events, self.cleared, self.calls = [], 0, []

    def disk(self):
        return {"running": self.running, "on": self.running, "settings": {"mode": "on"},
                "open_jobs": 0, "events": list(self.events), "top": []}

    def complete(self, body, session):
        n = len(json.dumps(body)) // 4
        user = body["messages"][-1]["content"]
        cold = session.endswith("-a")
        if cold:
            if self.failed:
                # 2026-10-02 21:09: the writer's copy failed inside a graph capture
                self.events.append({"kind": "error", "ts": time.time(), "error":
                                    "AcceleratorError: CUDA error: operation not permitted "
                                    "when stream is capturing"})
            self.events.append({"kind": "write", "ts": time.time(), "tokens": HEAD,
                                "kinds": ["opencode-head"], "anchors": 4,
                                "chunks": 1 if self.failed else 23, "dropped": None})
        reused = 0 if cold else (HEAD - 1024 if self.short else HEAD)
        if not cold:
            self.events.append({"kind": "hit", "ts": time.time(), "tokens": reused,
                                "read": reused, "bytes": reused * 106_496, "ms": 520.0})
        text = f"answer to {user}" + (" (other)" if self.differ and session.endswith("-b") else "")
        return text, n, reused, ("disk" if reused else "none")


def serve(stub):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, obj):
            data = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            stub.calls.append(("GET", self.path))
            if self.path == "/v1/conductor/memory/disk":
                return self._json(stub.disk())
            self.send_error(404)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            stub.calls.append(("POST", self.path))
            if self.path == "/v1/cache/clear":
                stub.cleared += 1
                return self._json({})
            if self.path == "/v1/chat/completions":
                text, n, reused, src = stub.complete(body, self.headers.get("x-session-id"))
                assert self.headers.get("User-Agent", "").startswith("opencode/")
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                sess = self.headers.get("x-session-id")
                chunks = []
                if stub.lines:
                    # the step lines as 0.3.14 streamed them on 2026-10-02 at 22:08: a reading
                    # line on the cold run, a waiting line on the others
                    line = ("reading 16,427 tokens at ~800 tok/s, about 18 s left"
                            if sess.endswith("-a") else
                            "waiting: another request is running, position 1")
                    chunks.append({"choices": [{"delta": {"content": f"\n› {line}\n\n"}}]})
                    chunks.append({"choices": [{"delta": {
                        "content": "\n› deciding: code_edit, xhigh effort\n\n"}}]})
                chunks += [{"choices": [{"delta": {"reasoning_content": text[:5]}}]},
                           {"choices": [{"delta": {"content": text[5:]}}]}]
                if stub.calls_on:
                    stub.n_calls += 1
                    arg = "src" if stub.args_differ and sess.endswith("-b") else "."
                    chunks += [{"choices": [{"delta": {"tool_calls": [
                                   {"index": 0, "id": f"call_{stub.n_calls:04d}{sess[-1]}",
                                    "type": "function",
                                    "function": {"name": "tool_00", "arguments": ""}}]}}]},
                               {"choices": [{"delta": {"tool_calls": [
                                   {"index": 0, "function": {"arguments": '{"path": "'}}]}}]},
                               {"choices": [{"delta": {"tool_calls": [
                                   {"index": 0, "function": {"arguments": arg + '"}'}}]}}]}]
                chunks += [{"choices": [{"delta": {},
                                         "finish_reason": "error" if stub.failed else "stop"}]},
                          {"choices": [], "usage": {"prompt_tokens": n, "completion_tokens": 9},
                           "timings": {"cache_n": reused, "cache_source": src, "ttft_ms": 1.0,
                                       "prompt_per_second": 700.0}}]
                for c in chunks:
                    self.wfile.write(f"data: {json.dumps(c)}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")
                return
            self.send_error(404)

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def check(stub, *extra):
    httpd = serve(stub)
    d = tempfile.mkdtemp(prefix="disk-live-")
    L.LOCK = os.path.join(d, "box.flock")
    out = os.path.join(d, "out.json")
    try:
        rc = L.main(["--server", f"http://127.0.0.1:{httpd.server_address[1]}",
                     "--head-tokens", "2000", "--token-file", os.path.join(d, "none"),
                     "--wait-s", "5", "--out", out, *extra])
        got = json.load(open(out)) if os.path.exists(out) else None
        return rc, got
    finally:
        httpd.shutdown()


def test_the_pass_reads_the_head_from_the_disk_with_the_same_text():
    stub = Stub()
    rc, got = check(stub)
    assert rc == 0, got
    assert got["gates"] == {"head_written": True, "answered": True, "disk_hit": True,
                            "same_text": True}
    assert got["runs"]["disk"]["reused"] == HEAD and got["other_session_from_disk"]
    assert stub.cleared == 1 and [c for c in stub.calls if c[0] == "POST"].count(
        ("POST", "/v1/chat/completions")) == 3
    # the clear comes after the cold run and before the disk run
    posts = [c[1] for c in stub.calls if c[0] == "POST"]
    assert posts == ["/v1/chat/completions", "/v1/cache/clear", "/v1/chat/completions",
                     "/v1/chat/completions"]


def test_another_text_or_a_short_hit_fails():
    rc, got = check(Stub(differ=True))
    assert rc == 1 and got["gates"]["same_text"] is False and got["gates"]["disk_hit"]
    rc, got = check(Stub(short=True))
    assert rc == 1 and got["gates"]["disk_hit"] is False and got["gates"]["same_text"]


def test_step_lines_and_call_ids_are_not_the_text():
    rc, got = check(Stub(lines=True, calls=True))
    assert rc == 0, got["gates"]
    cold, disk = got["runs"]["cold"], got["runs"]["disk"]
    assert cold["text"] != disk["text"] and cold["calls"] != disk["calls"], "the raw streams differ"
    assert cold["model_text"] == disk["model_text"] and "›" not in disk["model_text"]
    assert disk["model_text"].startswith("answer to ")
    assert disk["call_list"] == [[0, "tool_00", '{"path": "."}']]
    # what the model wrote still decides: other arguments, or another text, fail
    rc, got = check(Stub(lines=True, calls=True, args_differ=True))
    assert rc == 1 and got["gates"]["same_text"] is False
    rc, got = check(Stub(lines=True, differ=True))
    assert rc == 1 and got["gates"]["same_text"] is False


def test_a_writer_error_or_a_failed_request_fails():
    """The 21:09 run passed `head_written` on a head with 1 of its chunks and `same_text` on two
    requests that both ended in an error: both are FAILs now."""
    rc, got = check(Stub(failed=True))
    assert rc == 1 and got["gates"]["head_written"] is False and got["gates"]["answered"] is False
    assert got["entry"]["writer_errors"], got["entry"]


def test_no_tier_or_a_held_box_is_refused_before_a_request():
    stub = Stub(running=False)
    rc, got = check(stub)
    assert rc == 2 and got is None
    assert not any(c == ("POST", "/v1/chat/completions") for c in stub.calls)
    stub = Stub()
    httpd = serve(stub)
    d = tempfile.mkdtemp(prefix="disk-live-")
    L.LOCK = os.path.join(d, "box.flock")
    fd = os.open(L.LOCK, os.O_CREAT | os.O_RDWR)
    fcntl.flock(fd, fcntl.LOCK_EX)
    try:
        assert L.main(["--server", f"http://127.0.0.1:{httpd.server_address[1]}"]) == 2
        assert stub.calls == []
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
        httpd.shutdown()


def test_the_head_is_about_as_long_as_asked():
    system, tools = L.head(24_000, "n")
    chars = len(system) + len(json.dumps(tools))
    assert 24_000 * L.CHARS_PER_TOKEN <= chars <= 24_000 * L.CHARS_PER_TOKEN + 400
    assert system.startswith("You are a coding agent (live check n).") and len(tools) == 16


def _main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")


if __name__ == "__main__":
    _main()
