"""The parallel view (server/parallel.py, `GET /v1/dashboard/parallel`): one lane per request in
flight on every backend, the sums per backend and in total, the per-second series.

    python tests/test_parallel.py

The unit half drives a `LiveRegistry` and the board with records made here and a clock of its
own; the proxy half runs the standalone proxy in front of the fake backend with 8 streams at once
and reads the view while they decode. CPU only, no model.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from proxy.measure import RelayRecord  # noqa: E402
from proxy_util import Proxy, fake, run  # noqa: E402
from server import live as live_mod  # noqa: E402
from server import parallel as par  # noqa: E402
from server.usage import RequestRecord  # noqa: E402
from tools import contract_check  # noqa: E402

EXAMPLE = os.path.join(ROOT, "docs", "contract", "dashboard-v1", "parallel.example.json")


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def _board(cost=None):
    perf = Clock()
    reg = live_mod.LiveRegistry(perf=perf, clock=lambda: 1.7e9 + perf.t, activity=False)
    board = par.ParallelBoard(reg, cost=cost, perf=perf, clock=lambda: 1.7e9 + perf.t)
    return reg, board, perf


def _relayed(rid, perf, backend="ds", model="m-flash", kind="deepseek"):
    rec = RelayRecord(rid, "chat", True, t_arrival=perf.t)
    rec.backend, rec.backend_kind, rec.backend_model, rec.model = backend, kind, model, model
    rec.client_kind, rec.conv = "other", f"agent-{rid}"
    return rec


def _sample(reg, perf, dt=1.0):
    perf.t += dt
    with reg._lock:
        live = list(reg._live.values())
    reg._sample(live, perf.t, 1.7e9 + perf.t)


def _finish(reg, rec, perf, tokens, cached=None, prompt=100, cost=None):
    rec.set_usage({"prompt_tokens": prompt, "completion_tokens": tokens, "cached_tokens": cached,
                   "reasoning_tokens": None, "cache_write_tokens": None})
    rec.finish_reason = "stop"
    rec.cost_micros = cost
    rec.t_end = perf.t
    reg.finish(rec)


def test_frame_check_tolerance_and_verification():
    fc = par.FrameCheck()
    assert not fc.verified
    assert fc.add(100, 101) and not fc.verified        # one pass is not enough
    assert fc.add(500, 503)                             # 2 tokens + 1 % of 503
    assert fc.verified
    assert not fc.tools_verified
    assert not fc.add(40, 180, tools=True)              # a call's arguments, several a frame
    assert fc.verified and not fc.tools_verified, "the text kind keeps its own record"
    assert fc.add(60, 61, tools=True) and fc.add(70, 70, tools=True)
    assert not fc.tools_verified, "one failing tool answer among the last 8"
    assert not fc.add(100, 180)                         # several tokens a frame
    assert not fc.verified, "one failing answer among the last 8 takes it back"
    assert fc.view()["last"] == {"frames": 100, "tokens": 180}


def test_unverified_frames_are_null_not_estimated():
    reg, board, perf = _board()
    rec = _relayed("a", perf)
    reg.register(rec)
    rec.stamp_sent()
    perf.t += 0.3
    for _ in range(40):
        perf.t += 0.01
        rec.stamp_token(perf.t, "content")
    _sample(reg, perf)
    doc = board.snapshot()
    lane = doc["lanes"][0]
    assert lane["state"] == "writing" and lane["frames"] == 40
    assert lane["tokens"] is None and lane["tokens_source"] is None
    assert lane["decode_now"] is None and lane["decode_avg"] is None
    assert doc["totals"]["decode_unreported"] == 1 and doc["totals"]["decode_tps"] is None
    assert lane["prompt_tokens"] is None and lane["cache_hit"] is None, "usage comes at the end"
    assert lane["ttft_ms"] == 310.0


def test_verified_lanes_rates_series_cost_and_window():
    reg, board, perf = _board(cost=lambda rec, n: n * 2)     # 2 micro-dollars a token
    # two answers whose frames matched their usage: frames count as tokens from now on
    for i in range(2):
        r = _relayed(f"w{i}", perf)
        reg.register(r)
        r.stamp_sent()
        for _ in range(30):
            perf.t += 0.01
            r.stamp_token(perf.t, "content")
        _finish(reg, r, perf, 31, cached=40, prompt=100, cost=500)
    assert board._frames[("ds", "m-flash")].verified
    # three streams at once: 50, 50 and 20 frames a second; one thinking, one in a tool call
    recs = [_relayed(x, perf) for x in ("a", "b", "c")]
    for r in recs:
        reg.register(r)
        r.stamp_sent()
    perf.t += 0.5
    rates = {"a": (50, "reasoning", None), "b": (50, "tool", "write"), "c": (20, "content", None)}
    for _sec in range(3):
        for k in range(50):
            for r in recs:
                n, kind, tool = rates[r.request_id]
                if k < n:
                    r.stamp_token(perf.t + k * 0.02, kind, tool)
        _sample(reg, perf)
    doc = board.snapshot()
    lanes = {l["id"]: l for l in doc["lanes"]}
    assert lanes["a"]["state"] == "thinking" and lanes["a"]["thinking_tokens"] == 150
    assert lanes["b"]["state"] == "tool_call" and lanes["b"]["tool"] == "write"
    # tool-call frames were never checked against a usage: no live count for that lane
    assert lanes["b"]["tokens"] is None and lanes["b"]["tool_frames"] == 150
    assert lanes["b"]["decode_now"] is None
    assert lanes["c"]["state"] == "writing" and lanes["c"]["tokens"] == 60
    assert abs(lanes["a"]["decode_now"] - 50) < 1 and abs(lanes["c"]["decode_now"] - 20) < 1
    assert lanes["c"]["cost_micros"] == 120 and lanes["c"]["cost_partial"] is True
    assert lanes["a"]["client"]["session"] == "agent-a"
    t = doc["totals"]
    assert t["in_flight"] == 3 and t["decoding"] == 3 and abs(t["decode_tps"] - 70) < 2
    assert t["decode_unreported"] == 1
    assert t["cache_hit"] == 0.4 and t["cache_reported"] == 2 and t["cost_window"] == 1000
    assert t["cost_in_flight"] == 150 * 2 + 120
    ds = doc["backends"][0]
    assert ds["id"] == "ds" and ds["in_flight"] == 3 and ds["frames"]["verified"]
    # the series: each second's tokens per backend (the last full second: 50 + 20)
    assert doc["series"][-1]["by"]["ds"] == 70.0
    assert contract_check.check(doc, "parallel") == []
    if os.environ.get("WRITE_EXAMPLE") == "1":          # docs/contract/dashboard-v1's example
        with open(EXAMPLE, "w") as f:
            json.dump(doc, f, indent=2)
            f.write("\n")
    since = doc["series"][-2]["t"]
    assert len(board.snapshot(since)["series"]) == 1
    # one ends: its lane stays a few seconds as done, with the usage's numbers
    _finish(reg, recs[2], perf, 61, cached=0, prompt=200, cost=999)
    doc = board.snapshot()
    done = next(l for l in doc["recent"] if l["id"] == "c")
    assert done["state"] == "done" and done["tokens"] == 61 and done["tokens_source"] == "usage"
    assert done["cost_micros"] == 999 and done["cost_partial"] is False
    assert doc["totals"]["in_flight"] == 2 and doc["totals"]["finished"] == 3
    perf.t += par.DONE_KEEP_S + 1
    board._built = None
    assert board.snapshot()["recent"] == []


def test_contract_example_validates():
    with open(EXAMPLE) as f:
        assert contract_check.check(json.load(f), "parallel") == []
    bad = {"contract_version": "1.0", "at": 1, "windows": {}, "lanes": [{"id": "x"}],
           "recent": [], "backends": [], "totals": {}, "series": []}
    assert contract_check.check(bad, "parallel"), "the schema is strict"


def test_engine_lane_reads_the_record():
    reg, board, perf = _board()
    rec = RequestRecord("e1", "chat", True, t_arrival=perf.t)
    rec.model, rec.prompt_tokens = "TandemLLM", 1000
    reg.register(rec)
    doc = board.snapshot()
    assert doc["lanes"][0]["state"] == "queued" and doc["lanes"][0]["backend"] == "engine"
    rec.t_lock = perf.t + 0.1
    rec.t_first = perf.t + 0.5
    perf.t += 0.5
    for _ in range(3):
        rec.n_live += 60
        _sample(reg, perf)
    board._built = None
    lane = board.snapshot()["lanes"][0]
    assert lane["state"] == "writing" and lane["tokens"] == 180 and lane["tokens_source"] == "engine"
    assert abs(lane["decode_now"] - 60) < 1 and lane["kind"] == "tandemllm"


def test_engine_behind_the_proxy_reads_its_own_lanes():
    """A TandemLLM backend: the lane takes the engine's own count, state, prefill and cache."""
    calls = []

    def fetch(url):
        calls.append(url)
        return {"lanes": [{"relay_id": "px1", "tokens": 120, "state": "thinking",
                           "label": "Thinking", "tool": None, "thinking_tokens": 100,
                           "prefill": {"done": 1000, "total": 1000, "pct": 100.0, "cached": 800,
                                       "tps": None, "tps_avg": 2100.0, "eta_ms": None},
                           "decode_now": 55.0, "decode_avg": 54.0, "prompt_tokens": 1000,
                           "cached_tokens": 800}]}
    perf = Clock()
    reg = live_mod.LiveRegistry(perf=perf, clock=lambda: 1.7e9 + perf.t, activity=False)
    board = par.ParallelBoard(reg, peers=lambda: [("eng", "http://127.0.0.1:8000/v1")],
                              fetch=fetch, perf=time.perf_counter)
    rec = _relayed("px1", perf, backend="eng", model="TandemLLM", kind="tandemllm")
    reg.register(rec)
    rec.stamp_sent()
    rec.stamp_token(perf.t + 0.4, "reasoning")      # one frame: a block of several tokens
    _sample(reg, perf)
    deadline = time.time() + 3
    while time.time() < deadline and not board._peer:
        time.sleep(0.05)
    assert calls and calls[0] == "http://127.0.0.1:8000/v1/dashboard/parallel?since=9999999999"
    board._built = None
    lane = board.snapshot()["lanes"][0]
    assert lane["via"] == "engine" and lane["tokens"] == 120 and lane["tokens_source"] == "engine"
    assert lane["state"] == "thinking" and lane["thinking_tokens"] == 100
    assert lane["decode_now"] == 55.0 and lane["cache_hit"] == 0.8 and lane["prefill"]["tps_avg"] == 2100.0
    assert lane["ttft_ms"] == 400.0, "TTFT stays the one the proxy saw"
    assert contract_check.check(board.snapshot(), "parallel") == []
    # a backend whose board does not answer is left alone for a minute
    board2 = par.ParallelBoard(reg, peers=lambda: [("eng", "http://h:1")], fetch=lambda u: None)
    _sample(reg, perf)
    deadline = time.time() + 3
    while time.time() < deadline and "eng" not in board2._peer_off:
        time.sleep(0.05)
    assert "eng" in board2._peer_off and board2._peer_lane(rec) is None


def _engine_series(phase):
    """Six requests back to back on one TandemLLM backend at 60 tok/s (each queued 4 s behind the
    one before, as the engine shows it), the engine's board read once a second at `phase` after
    each of the registry's samples; returns the series' values for that backend."""
    perf = Clock()
    perf.t = 0.0
    reg = live_mod.LiveRegistry(perf=perf, clock=lambda: 1.7e9 + perf.t, activity=False)
    board = par.ParallelBoard(reg, peers=lambda: [], fetch=lambda u: None, perf=perf,
                              clock=lambda: 1.7e9 + perf.t)
    rate, n, reqs, t = 60.0, 600, [], 0.0
    for i in range(6):
        reqs.append((f"px{i}", t, t + n / rate))
        t += n / rate + 0.3
    recs, now = {}, 0.0
    while now < t + 2:
        now = round(now + 0.05, 2)
        perf.t = now
        for rid, s, e in reqs:
            if rid not in recs and now >= s - 4.0:
                rec = recs[rid] = _relayed(rid, perf, backend="feng", model="m", kind="tandemllm")
                reg.register(rec)
                rec.stamp_sent()
            rec = recs.get(rid)
            if rec is not None and rec.t_first is None and now >= s:
                rec.stamp_token(now, "content")
            if rec is not None and rec.t_end is None and now >= e:
                _finish(reg, rec, perf, n)
        if round((now - phase) % 1.0, 2) in (0.0, 1.0):
            lanes = {}
            for rid, s, e in reqs:
                if s <= now < e:
                    lanes[rid] = {"relay_id": rid, "tokens": int(rate * (now - s)),
                                  "decode_now": rate, "state": "writing"}
                elif s - 4.0 <= now < s:
                    lanes[rid] = {"relay_id": rid, "tokens": 0, "decode_now": None,
                                  "state": "queued"}
            board._peer["feng"] = (now, lanes)
        if round(now % 1.0, 2) in (0.0, 1.0):
            with reg._lock:
                live = list(reg._live.values())
            board.on_sample(live, perf.t, 1.7e9 + perf.t)
    return [x["by"].get("feng", 0) for x in board._series], n * len(reqs)


def test_an_engine_backend_s_series_sums_to_its_tokens_and_stays_level():
    """T1 on the box (N = 16, the fake engine as a second TandemLLM backend at 60 tok/s): the
    series read 145 next to 0 at turn boundaries. The end added the tokens since the engine's last
    read, which its rate had counted already, and the board was read every 2 s, so a lane that
    had started still read "queued, 0 tokens" for a whole sample. Now each lane's sum is its count
    and the read comes PEER_LEAD_S before each sample."""
    for phase in (0.5, 1.0 - par.PEER_LEAD_S):
        vals, true = _engine_series(phase)
        assert round(sum(vals)) == true, (phase, sum(vals), true)
        assert max(vals) <= 1.6 * 60, (phase, vals)
        assert all(v > 0 for v in vals[1:-2]), (phase, vals)
    vals, _ = _engine_series(1.0 - par.PEER_LEAD_S)
    assert max(vals) <= 1.2 * 60 and min(vals[1:-2]) >= 0.5 * 60, vals
    # the read's timing: PEER_LEAD_S before the sample after the last one
    reg, board, perf = _board()
    assert board._peer_delay() == par.PEER_S                  # no sample yet
    board._series.append({"t": 0.0, "_p": perf.t, "by": {}, "in_flight": 0})
    perf.t += 0.1
    assert abs(board._peer_delay() - (0.9 - par.PEER_LEAD_S)) < 1e-9
    perf.t += 0.8                                             # past the moment: read now
    assert board._peer_delay() == 0.0


def _stream(port, text, model="fake-a", session=None, out=None):
    import http.client
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
    h = {"Content-Type": "application/json"}
    if session:
        h["x-session-id"] = session
    c.request("POST", "/v1/chat/completions", json.dumps(
        {"model": model, "messages": [{"role": "user", "content": text}], "stream": True}), h)
    r = c.getresponse()
    times, data = [], b""
    while True:
        line = r.readline()
        if not line:
            break
        data += line
        if line.startswith(b"data: {") and b'"content": "' in line and b'"content": ""' not in line:
            times.append(time.perf_counter())
    c.close()
    if out is not None:
        out.append((session, times))
    return data


def test_proxy_eight_streams_at_once():
    srv, port = fake(models=["fake-a"], cache=False)
    env = {"TANDEM_BACKENDS": "f1", "TANDEM_BACKEND_F1_KIND": "generic",
           "TANDEM_BACKEND_F1_URL": f"http://127.0.0.1:{port}",
           "TANDEM_PRICE_1": "f1/fake-a=1,0.1,2"}
    with Proxy(env) as px:
        px.sync()
        for i in range(2):                          # the frame check: two answers
            _stream(px.port, "warm FAKE_LONG 20")
        out: list = []
        ths = [threading.Thread(target=_stream, args=(px.port, "go FAKE_LONG 160 FAKE_TPS 40"),
                                kwargs={"session": f"agent-{i}", "out": out}) for i in range(8)]
        for t in ths:
            t.start()
        time.sleep(2.6)
        st, hdrs, body = px.get("/v1/dashboard/parallel")
        assert st == 200 and hdrs.get("Cache-Control") == "no-store"
        doc = json.loads(body)
        lanes = doc["lanes"]
        assert len(lanes) == 8, [l["state"] for l in lanes]
        assert {l["client"]["session"] for l in lanes} == {f"agent-{i}" for i in range(8)}
        for l in lanes:
            assert l["state"] == "writing" and l["tokens_source"] == "frames", l
            assert 30 <= l["decode_now"] <= 48, l["decode_now"]
            assert l["cost_partial"] and l["cost_micros"] == l["tokens"] * 2
        t = doc["totals"]
        assert t["in_flight"] == 8 and t["decoding"] == 8 and 240 <= t["decode_tps"] <= 384, t
        assert doc["backends"][0]["frames"]["verified"]
        assert contract_check.check(doc, "parallel") == []
        assert doc["series"] and max(s["by"].get("f1", 0) for s in doc["series"]) > 200
        for th in ths:
            th.join()
        time.sleep(0.5)
        doc = json.loads(px.get("/v1/dashboard/parallel")[2])
        assert doc["totals"]["in_flight"] == 0 and doc["totals"]["finished"] == 10
        mine = [l for l in doc["recent"] if l["client"]["session"]]
        assert len(mine) == 8 and all(l["finish"] == "stop" for l in mine)
        assert all(l["tokens"] == 160 and l["tokens_source"] == "usage" for l in mine)
        # each lane's rate against the one its client measured off the same frames (within 3 %)
        seen = {sess: (len(t) - 1) / (t[-1] - t[0]) for sess, t in out}
        for l in mine:
            want = seen[l["client"]["session"]]
            assert abs(l["decode_avg"] - want) <= 0.03 * want, (l["decode_avg"], want)
        assert doc["totals"]["cost_window"] is not None and doc["build_us"]["max"] < 50_000
    srv.shutdown()


def test_proxy_without_usage_stays_unreported():
    srv, port = fake(models=["fake-a"], cache=False)
    env = {"TANDEM_BACKENDS": "f1", "TANDEM_BACKEND_F1_KIND": "generic",
           "TANDEM_BACKEND_F1_URL": f"http://127.0.0.1:{port}"}
    with Proxy(env) as px:
        px.sync()
        for _ in range(2):
            _stream(px.port, "warm FAKE_NO_USAGE FAKE_LONG 10")
        th = threading.Thread(target=_stream, args=(px.port, "x FAKE_NO_USAGE FAKE_LONG 80 FAKE_TPS 40"))
        th.start()
        time.sleep(1.6)
        doc = json.loads(px.get("/v1/dashboard/parallel")[2])
        lane = doc["lanes"][0]
        assert lane["frames"] > 0 and lane["tokens"] is None and lane["decode_now"] is None
        assert doc["totals"]["decode_unreported"] == 1
        th.join()
        assert doc["backends"][0]["frames"] is None, "never checked: no usage to check against"
    srv.shutdown()


if __name__ == "__main__":
    run(globals())
