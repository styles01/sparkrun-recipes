"""`/v1/dashboard/live` through the real handler, on a CPU.

tests/test_live.py tests the registry on its own. This file tests what only the server can show:
the route and its auth, the SSE first event, the row of a request that went through `_complete`
(registered before the queue, finished in its `finally`), and that the final live row carries the
same numbers as the response's `timings`.

Run: python tests/test_live_app.py   (needs CPU torch: the box, or any machine with it)
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_app_loop import Req, serve  # noqa: E402  (first: the CPU environment)
from test_usage import UTok, _ids, _script  # noqa: E402

from server import app, auth, live  # noqa: E402
from tools import contract_check  # noqa: E402

ADMIN = "adm-" + "7" * 40


def _get(path, headers=None, peer="127.0.0.1", gone=(True,)):
    req = Req(path, {})
    req.command, req.client_address = "GET", (peer, 0)
    req.headers.update(headers or {})
    it = iter(gone)
    req._reader_gone = lambda: next(it, True)     # the harness has no socket: say when to stop
    with contextlib.redirect_stdout(io.StringIO()):
        req.do_GET()
    head, _, body = req.wfile.getvalue().partition(b"\r\n\r\n")
    return head.decode(), body.decode()


def _events(body: str) -> list[dict]:
    out = []
    for block in body.split("\n\n"):
        for line in block.splitlines():
            if line.startswith("data: "):
                out.append(json.loads(line[6:]))
    return out


def _setup():
    serve()
    app.STATE.update(tok=UTok(), auth=auth.Auth(ADMIN, None, trust_loopback=False))
    app.STATE.pop("live", None)


def test_the_route_is_authenticated_and_valid():
    _setup()
    head, body = _get("/v1/dashboard/live?follow=0")
    assert head.startswith("HTTP/1.1 401"), head
    head, body = _get("/v1/dashboard/live?follow=0", {"Authorization": f"Bearer {ADMIN}"})
    assert head.startswith("HTTP/1.1 200") and "no-store" in head, head
    snap = json.loads(body)
    errs = contract_check.check(snap, "live")
    assert not errs, errs[:5]
    assert snap["requests"] == [] and snap["counts"]["in_flight"] == 0


def test_a_request_through_the_handler_leaves_a_row_equal_to_its_timings():
    _setup()
    real = app.generate_stream
    try:
        app.generate_stream = _script(_ids("Danube, Rhine, Elbe.") + [3])
        app.STATE["cfg_eos"] = 3
        with contextlib.redirect_stdout(io.StringIO()):
            head, body = Req("/v1/chat/completions",
                             {"messages": [{"role": "user", "content": "rivers"}],
                              "temperature": 0.7}).response()
        assert head.startswith("HTTP/1.1 200"), head
        timings = json.loads(body)["timings"]
    finally:
        app.generate_stream = real
    head, body = _get("/v1/dashboard/live?follow=0", {"Authorization": f"Bearer {ADMIN}"})
    snap = json.loads(body)
    assert not contract_check.check(snap, "live")
    (row,) = snap["requests"]
    assert row["phase"] == "done" and row["finish_reason"] == "stop" and row["status"] == 200
    assert row["tokens"] == timings["predicted_n"] == 21
    assert row["ttft_ms"] == timings["ttft_ms"] and row["queue_ms"] == timings["queue_ms"]
    assert row["decode_tps"] == timings["predicted_per_second"]
    assert row["decode_ms"] == timings["predicted_ms"] and row["elapsed_ms"] == timings["total_ms"]
    assert row["temperature"] == 0.7 and row["model"] == "t", row
    assert row["client"]["kind"] in ("curl", "other")      # the harness sends no User-Agent
    assert row["ended_ms_ago"] is not None and row["ended_ms_ago"] < 5000
    c = snap["counts"]
    assert c["served"] == 1 and c["completed_1m"] == 1 and c["in_flight"] == 0
    # a refusal is a done row too, with no lock time
    app.INFLIGHT["waiting"] = 8
    with contextlib.redirect_stdout(io.StringIO()):
        head, _ = Req("/v1/chat/completions",
                      {"messages": [{"role": "user", "content": "q"}]}).response()
    app.INFLIGHT["waiting"] = 0
    assert head.startswith("HTTP/1.1 503"), head
    snap = json.loads(_get("/v1/dashboard/live?follow=0",
                           {"Authorization": f"Bearer {ADMIN}"})[1])
    assert snap["requests"][0]["finish_reason"] == "refused"
    assert snap["requests"][0]["queue_ms"] is None and snap["counts"]["refused"] == 1


def test_the_stream_sends_history_first_then_samples():
    _setup()
    reg = app.live_registry()
    reg.tick()
    reg.tick()
    # a fast sampler, so each event follows a tick; the reader stays for two events, then leaves
    reg.interval = 0.05
    reg.start()
    try:
        head, body = _get("/v1/dashboard/live", {"Authorization": f"Bearer {ADMIN}"},
                          gone=(False, False, True))
    finally:
        reg.stop()
    assert head.startswith("HTTP/1.1 200") and "text/event-stream" in head, head
    events = _events(body)
    assert len(events) >= 2, body[:300]
    assert "history" in events[0] and len(events[0]["history"]) == 2
    assert "history" not in events[1] and events[1]["sample"] is not None
    for e in events:
        assert not contract_check.check(e, "live")
    assert reg._streams == 0                    # unsubscribed on the way out


def test_the_fifth_stream_is_refused():
    _setup()
    reg = app.live_registry()
    for _ in range(live.MAX_STREAMS):
        assert reg.subscribe()
    head, body = _get("/v1/dashboard/live", {"Authorization": f"Bearer {ADMIN}"})
    assert head.startswith("HTTP/1.1 429"), head
    assert json.loads(body)["error"]["type"] == "too_many"
    for _ in range(live.MAX_STREAMS):
        reg.unsubscribe()


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:64s} ok")
            passed += 1
    print(f"{passed} passed")
