"""What the server does while a long prefill runs, and when the client has left.

2026-09-26, opencode at 64k-209k tokens: a prefill took 73-354 s and sent nothing but headers, so
the client looked hung and the operator cancelled it (13:14, 19:28, 23:32 in opencode's log). The engine
noticed only when it wrote its first token: it finished every dead prefill -- 344 s of it after the
23:32 cancel -- and the re-sent request queued behind it (13 s at 13:14). Now:

  * after every prefill chunk the handler checks the socket; a client that left ends the prefill
    there, the lock is released, and the request is `abandoned`, streamed or not;
  * a client that leaves while queued is dropped before it takes the lock;
  * a streamed prefill that runs past `--prefill-heartbeat-s` sends SSE comments, which no client
    counts as a token; a short prefill sends none, so /the first-chunk rules hold.

Run: python tests/test_prefill_watch.py
"""

from __future__ import annotations

import json
import os
import sys
import threading

for _k in ("NORM", "GDN", "HEAD", "ATTN", "GDNBLOCK", "GDNTREE", "GDNPRE"):
    os.environ.setdefault(f"QWEN38_FUSED_{_k}", "0")
os.environ.setdefault("QWEN38_TREE_CHAIN_DELEGATE", "0")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine import cache  # noqa: E402
from server import app  # noqa: E402
from test_app_loop import Req, serve  # noqa: E402

PROMPT = "abcdefghij" * 10          # 100 tokens under the test tokenizer: 6 chunks of 16


def prefix_on(chunk=16, resident=True):
    eng = serve(max_len=512)
    app.STATE.update(prefix_chunk=chunk, prefix_cache=True,
                     state_store=cache.StateStore(1 << 30, chunk=chunk),
                     resident=cache.ResidentPrefix(1 << 30, chunk, stash_bytes=1 << 30)
                     if resident else None, prefill_heartbeat=0.0)
    return eng


class Leaves(Req):
    """A client that closes its socket after the handler has looked `after` times."""

    def __init__(self, path, body, after):
        super().__init__(path, body)
        self.after, self.looks = after, 0

    def _client_gone(self):
        self.looks += 1
        return self.looks > self.after


def _counts():
    return dict(app.INFLIGHT)


def test_a_client_that_leaves_mid_prefill_stops_it_there():
    for stream in (True, False):
        eng = prefix_on()
        before = _counts()
        calls = []
        real = cache.prefill

        def spy(*a, **k):
            inner = k.get("on_chunk")
            k["on_chunk"] = (lambda d, t: (calls.append(d), inner(d, t))) if inner else None
            return real(*a, **k)

        cache.prefill = spy
        try:
            req = Leaves("/v1/completions", {"prompt": PROMPT, "max_tokens": 8,
                                             "stream": stream}, after=2)
            req.response()
        finally:
            cache.prefill = real
        assert calls == [16, 32, 48], f"stream={stream}: the prefill ran on: {calls}"
        assert eng.kv.length == 48, eng.kv.length
        assert not app.LOCK.locked(), "the lock is released at once"
        after = _counts()
        assert after["abandoned"] == before["abandoned"] + 1, (stream, before, after)
        assert after["errors"] == before["errors"], "a client leaving is not an engine error"
        if not stream:
            assert b"internal_error" not in req.wfile.getvalue()
        res = app.STATE["resident"]
        assert res.valid == 48 and max(res.anchors) == 48, res.report()


def test_the_retry_resumes_where_the_abandoned_prefill_stopped():
    prefix_on()
    Leaves("/v1/completions", {"prompt": PROMPT, "max_tokens": 4, "stream": True},
           after=3).response()
    head, body = Req("/v1/completions", {"prompt": PROMPT + "xyz", "max_tokens": 4,
                                         "stream": True}).response()
    assert head.startswith("HTTP/1.1 200"), head
    lp = app.STATE["last_prefill"]
    assert (lp["reused"], lp["kind"]) == (64, "resident"), lp


def test_a_client_that_leaves_while_queued_never_runs():
    prefix_on()
    before = _counts()
    app.LOCK.acquire()                       # somebody else's long prefill
    try:
        req = Leaves("/v1/completions", {"prompt": PROMPT, "max_tokens": 4}, after=1)
        t = threading.Thread(target=req.response)
        t.start()
        t.join(10)
        assert not t.is_alive(), "the queued request must give up on its own"
    finally:
        app.LOCK.release()
    after = _counts()
    assert after["abandoned"] == before["abandoned"] + 1
    assert after["waiting"] == 0 and after["running"] == before["running"]
    assert req.wfile.getvalue() == b"", "nothing is written to a client that left"


def test_a_long_streamed_prefill_sends_comments_and_nothing_else():
    prefix_on()
    app.STATE["prefill_heartbeat"] = 1e-9
    real = app.HEARTBEAT_EVERY_S
    app.HEARTBEAT_EVERY_S = 0.0
    try:
        head, body = Req("/v1/completions", {"prompt": PROMPT, "max_tokens": 6,
                                             "stream": True}).response()
    finally:
        app.HEARTBEAT_EVERY_S = real
    comments = [ln for ln in body.splitlines() if ln.startswith(":")]
    assert comments == [f": prefill {d}/100" for d in (16, 32, 48, 64, 80, 96)], comments
    first_data = next(i for i, ln in enumerate(body.splitlines()) if ln.startswith("data: "))
    last_comment = max(i for i, ln in enumerate(body.splitlines()) if ln.startswith(":"))
    assert last_comment < first_data, "every comment precedes the first event"
    events = [json.loads(ln[6:]) for ln in body.splitlines()
              if ln.startswith("data: {")]
    assert events and events[-1]["choices"][0]["finish_reason"] in ("length", "stop")


def test_a_short_prefill_sends_no_comment():
    prefix_on()
    app.STATE["prefill_heartbeat"] = 5.0            # the served default
    _, body = Req("/v1/completions", {"prompt": PROMPT, "max_tokens": 6,
                                      "stream": True}).response()
    assert not [ln for ln in body.splitlines() if ln.startswith(":")], body[:200]


def test_a_json_request_never_gets_a_comment():
    prefix_on()
    app.STATE["prefill_heartbeat"] = 1e-9
    real = app.HEARTBEAT_EVERY_S
    app.HEARTBEAT_EVERY_S = 0.0
    try:
        head, body = Req("/v1/completions", {"prompt": PROMPT, "max_tokens": 6}).response()
    finally:
        app.HEARTBEAT_EVERY_S = real
    assert json.loads(body)["choices"][0]["text"] is not None


def _main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")


if __name__ == "__main__":
    _main()
