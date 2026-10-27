"""Prometheus contract 0.2.0 wired into the real server, on a CPU.

tests/test_metrics.py tests the page and the counters on their own; this file tests what only the
server can show: the state store telling a session hit from a prefix hit on the random 4-layer
model, and a scrape through the handler after real requests -- the per-request histograms fed by
the RequestRecord, the route and status counters, the build info.

Run: python tests/test_metrics_app.py
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_app_loop import FakeTok, Req, serve  # noqa: E402  (first: the CPU environment)
from test_usage import UTok, _ids, _script  # noqa: E402
from test_metrics import parse  # noqa: E402

import torch  # noqa: E402

from engine import cache  # noqa: E402
from server import app, auth, metrics  # noqa: E402


def test_a_session_hit_and_a_prefix_hit_are_told_apart():
    serve(None, max_len=256)
    store = cache.StateStore(1 << 30, chunk=16)
    app.STATE.update(tok=FakeTok(), state_store=store, prefix_cache=True, session_cache=True,
                     prefix_chunk=16)
    first = list(range(1, 41))                               # checkpoints at 16 and 32
    out = list(app.generate_stream(torch.tensor(first), 6, set()))
    app._remember(first, out, "conv-1")                      # the turn's end: a session entry
    assert app.STATE["last_prefill"]["kind"] is None
    # the next turn of the same conversation resumes from the session entry
    turn2 = list(app.STATE["last_ctx"]) + [7, 8, 9]
    list(app.generate_stream(torch.tensor(turn2), 3, set()))
    assert app.STATE["last_prefill"]["kind"] == "session", app.STATE["last_prefill"]
    # a new conversation that shares the first 32 tokens resumes from a chunk boundary
    other = first[:32] + [50, 51, 52, 53]
    list(app.generate_stream(torch.tensor(other), 3, set()))
    lp = app.STATE["last_prefill"]
    assert lp["kind"] == "prefix" and lp["reused"] == 32, lp
    assert store.stats["hits_session"] == 1 and store.stats["hits_prefix"] == 1, store.stats
    metrics.bind(cache_stats=app.cache_stats)
    fam = parse(metrics.render())
    rows = {lb.get("kind"): v for _, lb, v in fam["qse_cache_hits_total"]["samples"]
            if lb["cache"] == "state"}
    assert rows == {"session": 1.0, "prefix": 1.0}, rows


def _get(path, headers=None, peer="127.0.0.1"):
    req = Req(path, {})
    req.command, req.client_address = "GET", (peer, 0)
    req.headers.update(headers or {})
    with contextlib.redirect_stdout(io.StringIO()):
        req.do_GET()
    head, _, body = req.wfile.getvalue().partition(b"\r\n\r\n")
    return head.decode(), body.decode()


def test_a_scrape_after_real_requests():
    serve()
    # an admin password and the dashboard login on, so the proxied dashboard read below is refused
    # and counted as such (with no password every route is open, AU1)
    app.STATE.update(tok=UTok(), version="0.1.0-test", git_sha="abc1234",
                     code_sha256="f" * 64, args={"max_len": 256},
                     auth=auth.Auth("adm-" + "m" * 40, login=True))
    metrics.REGISTRY.reset()
    metrics.install(app)
    real = app.generate_stream
    try:
        app.generate_stream = metrics.track_stream(_script(_ids("Danube, Rhine.") + [3]))
        app.STATE["cfg_eos"] = 3
        with contextlib.redirect_stdout(io.StringIO()):
            head, body = Req("/v1/chat/completions",
                             {"messages": [{"role": "user", "content": "rivers"}]}).response()
        assert head.startswith("HTTP/1.1 200"), head
        app.INFLIGHT["waiting"] = 8                           # the queue is full: a refusal
        with contextlib.redirect_stdout(io.StringIO()):
            head, _ = Req("/v1/chat/completions",
                          {"messages": [{"role": "user", "content": "q"}]}).response()
        app.INFLIGHT["waiting"] = 0
        assert head.startswith("HTTP/1.1 503"), head
        head, _ = _get("/v1/dashboard/summary", {"X-Forwarded-For": "203.0.113.20"})
        assert head.startswith("HTTP/1.1 4"), head
    finally:
        app.generate_stream = real
    head, text = _get("/metrics")
    fam = parse(text)
    codes = {(lb["route"], lb["code"]): v for _, lb, v in fam["qse_http_requests_total"]["samples"]}
    assert codes[("chat", "200")] == 1 and codes[("chat", "503")] == 1, codes
    assert any(r == "dashboard" and c.startswith("4") for r, c in codes), codes
    assert all("/" not in r for r, _ in codes)
    fin = {lb["finish_reason"]: v for _, lb, v in fam["qse_requests_total"]["samples"]}
    assert fin == {"stop": 1.0, "refused": 1.0}, fin
    count = {n: v for n, lb, v in fam["qse_request_decode_tokens_per_second"]["samples"]
             if n.endswith("_count")}
    assert count == {"qse_request_decode_tokens_per_second_count": 1.0}, count
    (_, info, one), = fam["qse_build_info"]["samples"]
    assert one == 1.0 and info["code_sha256"] == "f" * 64 and info["git_sha"] == "abc1234"
    assert metrics.VERSION == "0.2.0"
    assert 'version="0.2.0"' in text                          # qse_engine_info's contract label


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:64s} ok")
            passed += 1
    print(f"{passed} passed")
