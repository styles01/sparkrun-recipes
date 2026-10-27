"""The live rates without the warm-up (VIS-40): the prefill tile, the banner, the decode tile,
the charts and the in-flight rows show the speed NOW, and the end-to-end numbers do not move.

The matrix is VIS-41's (the Gherkin of VIS-40); each test names its scenario. Fake records with
scripted chunk marks and a hand-moved clock, no torch, so it runs on any machine.

Run: python tests/test_live_rates.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server import activity, live, metrics, usage  # noqa: E402
from test_live import Clock, _rec, _reg, _row, _valid  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def teardown_function(_fn):
    Clock.restore()                              # the Clock patches time.perf_counter


def _prefilling(c, reg, rid, prompt, start=0, chunk=1024):
    """A request in its prefill: locked, the loop started with `start` tokens restored."""
    rec = _rec(c, rid)
    reg.register(rec)
    rec.lock_acquired()
    rec.prompt_tokens = prompt
    rec.prefill_info = {"start": start, "chunk": chunk, "t0": c.perf(),
                        "kind": "prefix" if start else None}
    return rec


def _chunk(c, rec, secs, size=1024):
    """One chunk of `size` new tokens that took `secs`: what the handler's hook appends."""
    c.tick(secs)
    base = rec.pf_marks[-1][0] if rec.pf_marks else rec.prefill_info["start"]
    rec.pf_marks.append((min(base + size, rec.prompt_tokens), rec.prompt_tokens, c.perf()))


def _first_token(c, rec, secs, lp=None):
    c.tick(secs)
    src = rec.track(iter(range(100_000)))
    next(src)
    if lp is not None:
        rec.absorb_prefill(lp)
    return src


def _finish(c, rec, src, tokens, secs=0.0):
    c.tick(secs)
    for _ in range(tokens - rec.n_live):
        next(src)
    rec.completion_tokens, rec.finish_reason = tokens, "stop"
    rec.end()


def _decode(c, reg, src, secs, tokens, block=4, hz=4):
    """`secs` of decode at `tokens`/s in blocks of `block`, the sampler ticking `hz` times a
    second (tokens come out between ticks, as on the box)."""
    step = 1.0 / hz
    per_tick = tokens * step
    carry = 0.0
    for _ in range(int(round(secs * hz))):
        c.tick(step)
        carry += per_tick
        while carry >= block:
            for _ in range(block):
                next(src)
            carry -= block
        reg.tick()


def _event(reg):
    return _valid(json.loads(reg.event()[1].decode().split("data: ", 1)[1]))


# ------------------------------------------------------------------ scenario 1 + 2 + 9
def test_the_running_prefill_drives_the_headline_not_the_last_finished_request():
    """Scenario 1: A finished at 433 tok/s end to end; B prefills at 743 tok/s now."""
    c = Clock()
    reg = _reg(c, hz=4)
    reg.subscribe()
    a = _prefilling(c, reg, "A", 195_083, start=192_512)
    _chunk(c, a, 4.0)                            # the restore and the first chunk
    _chunk(c, a, 1.2)
    lp = {"reused": 192_512, "forwarded": 2_571, "ms": 5935.0, "kind": "prefix"}
    src = _first_token(c, a, 5.935 - 5.2, lp)
    reg.last_prefill_fn = lambda: lp
    _finish(c, a, src, 40, 1.0)
    reg.finish(a)
    assert round(a.prefill_tps, 1) == 433.2                         # the screenshot's number
    b = _prefilling(c, reg, "B", 107_604)
    _chunk(c, b, 3.0)                            # the warm-up chunk
    for _ in range(13):
        _chunk(c, b, 1024 / 743)
    reg.tick()
    ev = _event(reg)
    rb = _row(ev, "B")
    assert rb["activity"]["prefill"]["done"] == 14_336
    assert ev["now"]["prefill_live_tps"] == 743.0, ev["now"]
    assert rb["activity"]["prefill"]["tps_live"] == 743.0          # the banner's source
    assert ev["now"]["prefill_live_count"] == 1
    # the 1.0 field keeps its meaning: the last finished request's end-to-end rate
    assert ev["now"]["prefill_tps"] == round(a.prefill_tps, 2) == 433.19
    # the finished row keeps its end-to-end number, with the windowed one beside it
    ra = _row(ev, "A")
    assert ra["prefill_tps"] == round(a.prefill_tps, 2) == a.timings()["prompt_per_second"]
    assert ra["mostly_cached"] is True
    assert ra["activity"]["prefill"]["tps_live_avg"] == round((195_083 - 193_536) / (5.935 - 4.0), 2)


def test_tile_and_banner_agree_on_every_event():
    """Scenario 2: 20 consecutive events of one chunked prefill past its warm-up chunk. The
    tile's number (`now.prefill_live_tps`) and the banner's (the row's `tps_live`) come out of
    the same computation in the same event."""
    c = Clock()
    reg = _reg(c, hz=4)
    reg.subscribe()
    b = _prefilling(c, reg, "B", 107_604)
    _chunk(c, b, 3.0)
    seen = []
    for i in range(20):
        # chunks of uneven length land between the ticks
        c.tick(0.05)
        if i % 3 == 0:
            _chunk(c, b, 0.9 + 0.1 * (i % 4))
        reg.tick()
        ev = _event(reg)
        tile = ev["now"]["prefill_live_tps"]
        banner = _row(ev, "B")["activity"]["prefill"]["tps_live"]
        assert tile == banner, (i, tile, banner)
        seen.append(tile)
    assert all(v is not None for v in seen[3:]), seen


def test_nothing_prefilling_shows_no_live_rate():
    """Scenario 9: idle. No live rate, and no last-finished value in the live field."""
    c = Clock()
    reg = _reg(c)
    lp = {"reused": 0, "forwarded": 500, "ms": 250.0, "kind": None}
    rec = _prefilling(c, reg, "done", 500)
    src = _first_token(c, rec, 0.25, lp)
    _finish(c, rec, src, 3, 0.1)
    reg.finish(rec)
    reg.tick()
    s = _valid(reg.snapshot())
    assert s["now"]["prefill_tps"] == 2000.0                  # 1.0: the last finished request
    assert s["now"]["prefill_live_tps"] is None and s["now"]["prefill_live_count"] == 0
    assert s["now"]["prefilling"] is False


# ------------------------------------------------------------------ scenario 3 + 4 + 5 + 15
def test_the_headline_is_the_now_rate_not_an_average():
    """Scenario 3: chunks ran at 900 tok/s and now run at 600."""
    c = Clock()
    reg = _reg(c)
    b = _prefilling(c, reg, "B", 100_000)
    _chunk(c, b, 3.0)
    for _ in range(10):
        _chunk(c, b, 1024 / 900)
    for _ in range(5):
        _chunk(c, b, 1024 / 600)
    p = _row(reg.snapshot(), "B")["activity"]["prefill"]
    assert p["tps_live"] == 600.0, p
    assert p["tps_avg"] < 800 and 600 < p["tps_live_avg"] < 900, p   # the averages lag


def test_the_prefill_warm_up_chunk_is_left_out():
    """Scenario 4: the first chunk took 3.0 s and later ones take 1.4 s each."""
    c = Clock()
    reg = _reg(c, hz=4)
    reg.subscribe()
    b = _prefilling(c, reg, "B", 107_604)
    _chunk(c, b, 3.0)
    reg.tick()
    ev = _event(reg)
    p = _row(ev, "B")["activity"]["prefill"]
    assert p["tps_live"] is None and ev["now"]["prefill_live_tps"] is None
    assert p["warmup"] == {"chunks": 1, "reached": True}
    # SRV-591: one mark's `tps_now` is its rows since the loop started, as `tps_avg`
    assert p["tps_now"] == p["tps_avg"] == round(1024 / 3.0, 2)
    _chunk(c, b, 1.4)
    _chunk(c, b, 1.4)
    reg.tick()
    ev = _event(reg)
    p = _row(ev, "B")["activity"]["prefill"]
    assert p["tps_live"] == round(1024 / 1.4, 2) == 731.43, p        # chunks 2 and 3 only
    assert ev["now"]["prefill_live_tps"] == 731.43
    # the ETA runs on the rate after the warm-up, not on the loop's average
    assert abs(p["eta_ms"] - (107_604 - 3 * 1024) / (2048 / 2.8) * 1e3) < 1.0, p


def test_a_prefill_of_one_chunk_or_less_has_no_live_rate():
    """Scenario 5: 900 new tokens: the only chunk is the last, which the hook never reports."""
    c = Clock()
    reg = _reg(c)
    rec = _prefilling(c, reg, "C", 900)
    reg.tick()
    p = _row(reg.snapshot(), "C")["activity"]["prefill"]
    assert p["tps_live"] is None and p["warmup"] == {"chunks": 0, "reached": False}
    lp = {"reused": 0, "forwarded": 900, "ms": 600.0, "kind": None}
    src = _first_token(c, rec, 0.6, lp)
    _finish(c, rec, src, 10, 0.2)
    reg.finish(rec)
    r = _row(reg.snapshot(), "C")
    assert r["prefill_tps"] == rec.timings()["prompt_per_second"] == 1500.0
    assert r["activity"]["prefill"]["tps_live"] is None
    assert r["activity"]["prefill"]["tps_live_avg"] is None


def test_a_single_call_prefill_has_no_live_rate():
    """Scenario 15: `chunk = 0`, one call, no marks: tokens and elapsed only."""
    c = Clock()
    reg = _reg(c)
    _prefilling(c, reg, "S", 4000, chunk=0)
    c.tick(12.0)
    reg.tick()
    s = _valid(reg.snapshot())
    p = _row(s, "S")["activity"]["prefill"]
    assert p["progress"] == "single_call" and p["tps_live"] is None and p["pct"] is None
    assert s["now"]["prefill_live_tps"] is None and s["counts"]["prefilling"] == 1
    assert s["sample"]["prefill_live_tps"] is None


# ------------------------------------------------------------------ scenario 6 + 7 + 10
def test_decode_leaves_out_the_first_fifty_tokens():
    """Scenario 6: 23 tokens = warming up; at 120 the rate runs from the anchor on."""
    c = Clock()
    reg = _reg(c, hz=4)
    reg.subscribe()
    rec = _prefilling(c, reg, "D", 300)
    src = _first_token(c, rec, 0.3)
    for _ in range(23):
        next(src)
    reg.tick()
    ev = _event(reg)
    r = _row(ev, "D")
    assert r["decode_warmup"] == {"tokens": 23, "reached": False}, r["decode_warmup"]
    assert r["decode_live_now_tps"] is None and r["decode_live_avg_tps"] is None
    assert ev["now"]["decode_live_tps"] is None and ev["now"]["decode_warming"] == 1
    # the slow start: 27 more tokens over 2 s, then 40 tok/s in blocks of four
    for _ in range(27):
        next(src)
    c.tick(2.0)
    reg.tick()
    anchor = rec.decode_anchor
    assert anchor is not None and anchor[1] - 1 >= 50, anchor
    _decode(c, reg, src, 3.0, 40)
    ev = _event(reg)
    r = _row(ev, "D")
    n = r["tokens"]
    assert n >= 120 - 4 and r["decode_warmup"]["reached"] is True
    now_p = c.perf()
    assert r["decode_live_avg_tps"] == round((n - anchor[1]) / (now_p - anchor[0]), 2)
    assert 36 <= r["decode_live_now_tps"] <= 44, r
    assert ev["now"]["decode_live_tps"] == r["decode_live_now_tps"]
    assert ev["now"]["decode_warming"] == 0
    # the 1.0 average still counts from the first token, the slow start included
    assert r["decode_tps"] < r["decode_live_avg_tps"]


def test_the_anchor_is_the_first_tick_at_or_past_fifty_in_blocks_of_four():
    c = Clock()
    reg = _reg(c, hz=4)
    rec = _prefilling(c, reg, "D", 10)
    src = _first_token(c, rec, 0.1)
    seen = []
    for _ in range(20):
        for _ in range(4):
            next(src)
        c.tick(0.25)
        reg.tick()
        seen.append(rec.n_live)
        if rec.decode_anchor is not None:
            break
    assert rec.decode_anchor[1] == 53 == next(x for x in seen if x - 1 >= 50), seen


def test_a_short_reply_never_shows_a_live_decode_rate():
    """Scenario 7: 30 decoded tokens; the finished row keeps timings.predicted_per_second."""
    c = Clock()
    reg = _reg(c, hz=4)
    rec = _prefilling(c, reg, "E", 200)
    src = _first_token(c, rec, 0.2)
    _decode(c, reg, src, 0.75, 40)
    r = _row(reg.snapshot(), "E")
    assert r["decode_live_now_tps"] is None and r["decode_warmup"]["reached"] is False
    _finish(c, rec, src, 31, 0.1)
    reg.finish(rec)
    reg.tick()
    s = reg.snapshot()
    r = _row(s, "E")
    assert r["decode_live_avg_tps"] is None and r["decode_warmup"] == {"tokens": 30,
                                                                         "reached": False}
    assert r["decode_tps"] == rec.timings()["predicted_per_second"]
    assert all(x.get("decode_live_tps") is None for x in s["history"]), s["history"]


def test_nothing_decoding_leaves_the_decode_live_fields_empty():
    """Scenario 10."""
    c = Clock()
    reg = _reg(c)
    reg.tick()
    s = _valid(reg.snapshot())
    assert s["now"]["decode_live_tps"] is None and s["now"]["decode_warming"] == 0
    assert s["sample"]["decode_live_tps"] is None


def test_the_decode_chart_leaves_out_the_warm_up_tokens():
    c = Clock()
    reg = _reg(c, hz=1)
    rec = _prefilling(c, reg, "D", 10)
    src = _first_token(c, rec, 0.1)
    reg.tick()
    _decode(c, reg, src, 1.0, 40, hz=1)          # 40 tokens: warming up
    _decode(c, reg, src, 1.0, 40, hz=1)          # 80: past 50 at this tick
    h = reg.snapshot()["history"]
    assert h[-1]["decode_tps"] == 40.0 and h[-1]["decode_live_tps"] is None, h
    _decode(c, reg, src, 1.0, 40, hz=1)
    h = reg.snapshot()["history"]
    assert h[-1]["decode_live_tps"] == 40.0 == h[-1]["decode_tps"], h


# ------------------------------------------------------------------ scenario 8
GOLDEN_TIMINGS = {"cache_n": 192512, "prompt_n": 2571, "prompt_ms": 5935.0,
                  "prompt_per_token_ms": 2.31, "prompt_per_second": 433.19,
                  "predicted_n": 181, "predicted_ms": 4000.0, "predicted_per_token_ms": 22.22,
                  "predicted_per_second": 45.0}


def test_end_to_end_numbers_are_unchanged():
    """Scenario 8: for the same fixed record, `timings`, `usage`, the ledger row and the
    Prometheus observations are today's formulas (`usage.py`), byte for byte, whatever the live
    view did with the marks and the anchor."""
    c = Clock()

    def run(with_live: bool):
        reg = _reg(c, hz=4)
        rec = _rec(c, "golden")
        reg.register(rec)
        c.tick(0.4)
        rec.lock_acquired()
        rec.prompt_tokens = 195_083
        rec.prefill_info = {"start": 192_512, "chunk": 1024, "t0": c.perf(), "kind": "prefix"}
        if with_live:
            reg.subscribe()
            _chunk(c, rec, 3.0)
            _chunk(c, rec, 1.4)
            reg.tick()
            c.tick(5.935 - 4.4)
        else:
            c.tick(5.935)
        src = rec.track(iter(range(1000)))
        next(src)
        rec.absorb_prefill({"reused": 192_512, "forwarded": 2_571, "kind": "prefix"})
        for _ in range(20):
            c.tick(4.0 / 20)
            for _ in range(9):
                next(src)
            if with_live:
                reg.tick()
        rec.completion_tokens, rec.finish_reason, rec.blocks = 181, "stop", 40
        rec.end()
        reg.finish(rec)
        if with_live:
            assert rec.decode_anchor is not None and rec.pf_marks
        return rec, reg

    old, _ = run(False)
    new, reg = run(True)
    t_old, t_new = old.timings(), new.timings()
    for k, v in GOLDEN_TIMINGS.items():
        assert t_old[k] == v == t_new[k], (k, t_old[k], v, t_new[k])
    assert json.dumps(t_old, sort_keys=True) == json.dumps(t_new, sort_keys=True)
    assert json.dumps(old.usage(), sort_keys=True) == json.dumps(new.usage(), sort_keys=True)
    row_old, row_new = old.row("v", "sha"), new.row("v", "sha")
    for x in (row_old, row_new):
        x.pop("ts_ms")
    assert row_old == row_new, (row_old, row_new)
    assert row_new["prefill_tps"] == 433.19 and row_new["decode_tps"] == 45.0
    # Prometheus: the histograms observe the end-to-end rates, nothing windowed
    h = metrics.request_prefill_tps
    before = dict(h.sums)
    metrics.on_record(new)
    key = next(iter(h.sums))
    assert abs(h.sums[key] - before.get(key, 0.0) - new.prefill_tps) < 1e-9
    # and the finished live row shows those numbers, the windowed ones beside them
    r = _row(reg.snapshot(), "golden")
    assert r["prefill_tps"] == t_new["prompt_per_second"]
    assert r["decode_tps"] == t_new["predicted_per_second"]
    assert r["decode_live_avg_tps"] is not None
    assert r["activity"]["prefill"]["tps_live_avg"] is not None


# ------------------------------------------------------------------ scenario 11 + 12
def test_the_prefill_chart_gets_a_sample_every_second_during_a_prefill():
    """Scenario 11: 20 s of prefill after the warm-up chunk give ~20 one-second samples."""
    c = Clock()
    reg = _reg(c, hz=1)
    rec = _prefilling(c, reg, "P", 200_000)
    _chunk(c, rec, 3.0)
    reg.tick()
    got = []
    for _ in range(20):
        _chunk(c, rec, 1.0 / 2, size=512)        # two 512-token chunks a second: 1,024 tok/s
        _chunk(c, rec, 1.0 / 2, size=512)
        reg.tick()
        got.append(reg.snapshot()["sample"]["prefill_live_tps"])
    assert got[0] is not None and all(v == 1024.0 for v in got), got
    live_samples = [s["prefill_live_tps"] for s in reg.snapshot()["history"]
                    if s["prefill_live_tps"] is not None]
    assert len(live_samples) == 20 and max(live_samples) == 1024.0
    assert all(s["prefill_tps"] is None for s in reg.snapshot()["history"])   # nothing finished


def test_two_prefills_at_once_are_summed():
    """Scenario 12 (a server with slots): the headline is the sum, each row has its own."""
    c = Clock()
    reg = _reg(c, hz=4)
    reg.subscribe()
    f = _prefilling(c, reg, "F", 100_000)
    g = _prefilling(c, reg, "G", 100_000)
    t0 = c.perf()
    for rec, first, each in ((f, 3.0, 1024 / 743), (g, 2.0, 1024 / 510)):
        t = t0 + first
        for k in range(1, 6):                    # the warm-up chunk, then four at the rate
            rec.pf_marks.append((k * 1024, 100_000, t))
            t += each
    c.tick(3.0 + 4 * 1024 / 510)
    reg.tick()
    ev = _event(reg)
    rf, rg = (_row(ev, x)["activity"]["prefill"]["tps_live"] for x in ("F", "G"))
    assert (rf, rg) == (743.0, 510.0), (rf, rg)
    assert ev["now"]["prefill_live_tps"] == 1253.0 and ev["now"]["prefill_live_count"] == 2
    assert ev["sample"]["prefill_live_tps"] == 1253.0


# ------------------------------------------------------------------ scenario 13 + 14
def test_a_fully_cached_prompt_records_no_rate_or_sample():
    """Scenario 13: a response-cache replay, no token forwarded."""
    c = Clock()
    reg = _reg(c)
    rec = _rec(c, "H")
    reg.register(rec)
    rec.lock_acquired()
    rec.prompt_tokens = 195_083
    rec.absorb_response_cache()
    src = rec.track(iter(range(500)))
    for _ in range(200):
        next(src)
    reg.tick()
    c.tick(1.0)
    reg.tick()
    s = _valid(reg.snapshot())
    r = _row(s, "H")
    assert r["forwarded_tokens"] == 0 and r["mostly_cached"] is False
    assert r["activity"]["prefill"] is None and r["decode_warmup"] is None
    assert s["now"]["prefill_live_tps"] is None and s["now"]["decode_live_tps"] is None
    assert s["sample"]["prefill_live_tps"] is None and s["sample"]["decode_live_tps"] is None


def test_a_mostly_cached_prompt_is_labelled_and_left_out_of_the_peak():
    """Scenario 14: 195,083 tokens of which 192,512 cached (2,571 new, 1.3 %)."""
    c = Clock()
    reg = _reg(c, hz=1)
    i = _prefilling(c, reg, "I", 195_083, start=192_512)
    _chunk(c, i, 4.0)
    _chunk(c, i, 1.2)
    reg.tick()
    s = _valid(reg.snapshot())
    r = _row(s, "I")
    assert r["mostly_cached"] is True
    assert r["activity"]["prefill"]["tps_live"] == round(1024 / 1.2, 2)   # shown, it is real
    assert s["now"]["prefill_live_tps"] == round(1024 / 1.2, 2)
    assert s["sample"]["prefill_live_tps"] is None                         # not in the peak
    # the rule's edges: a cold prompt is not cached at all; 4,096+ new and 10 %+ is not "mostly"
    cold = _prefilling(c, reg, "cold", 900)
    big = _prefilling(c, reg, "big", 100_000, start=80_000)
    few = _prefilling(c, reg, "few", 20_000, start=17_000)
    thin = _prefilling(c, reg, "thin", 200_000, start=190_000)
    s = reg.snapshot()
    assert [_row(s, x)["mostly_cached"] for x in ("cold", "big", "few", "thin")] == \
        [False, False, True, True]
    assert _row(s, "cold")["mostly_cached"] is False and cold.prefill_info["start"] == 0
    assert big and few and thin


# ------------------------------------------------------------------ the settings
def test_the_windows_are_settings_and_the_event_reports_them():
    env = {}
    assert live.windows_from_env(env) == {"decode_warmup_tokens": 50, "prefill_warmup_chunks": 1,
                                          "now_window_s": 2.0, "mostly_cached_pct": 10.0,
                                          "mostly_cached_tokens": 4096}
    w = live.windows_from_env({"QSE_LIVE_DECODE_WARMUP_TOKENS": "8",
                               "QSE_LIVE_PREFILL_WARMUP_CHUNKS": "2",
                               "QSE_LIVE_NOW_WINDOW_S": "3.5",
                               "QSE_LIVE_MOSTLY_CACHED_PCT": "5",
                               "QSE_LIVE_MOSTLY_CACHED_TOKENS": ""})
    assert w == {"decode_warmup_tokens": 8, "prefill_warmup_chunks": 2, "now_window_s": 3.5,
                 "mostly_cached_pct": 5.0, "mostly_cached_tokens": 4096}
    for bad in ({"QSE_LIVE_DECODE_WARMUP_TOKENS": "fifty"},
                {"QSE_LIVE_PREFILL_WARMUP_CHUNKS": "-1"}, {"QSE_LIVE_NOW_WINDOW_S": "0"},
                {"QSE_LIVE_MOSTLY_CACHED_PCT": "101"}):
        try:
            live.windows_from_env(bad)
        except SystemExit:
            continue
        raise AssertionError(bad)
    c = Clock()
    reg = _reg(c, windows=w)
    s = _valid(reg.snapshot())
    assert s["windows"] == w
    # a 2-chunk warm-up needs a third mark before the rate appears; 8 decoded tokens anchor
    b = _prefilling(c, reg, "B", 50_000)
    _chunk(c, b, 3.0)
    _chunk(c, b, 1.0)
    assert _row(reg.snapshot(), "B")["activity"]["prefill"]["tps_live"] is None
    _chunk(c, b, 1.0)
    assert _row(reg.snapshot(), "B")["activity"]["prefill"]["tps_live"] == 1024.0
    d = _prefilling(c, reg, "D", 10)
    src = _first_token(c, d, 0.1)
    for _ in range(8):
        next(src)
    reg.tick()
    assert d.decode_anchor is not None and d.decode_anchor[1] == 9


def test_the_prefill_live_rate_with_no_warm_up_starts_at_the_loop():
    c = Clock()
    rec = _rec(c, "Z")
    rec.lock_acquired()
    rec.prompt_tokens = 10_000
    rec.prefill_info = {"start": 2048, "chunk": 1024, "t0": c.perf(), "kind": "prefix"}
    assert activity.prefill_live(rec, 0)["reached"] is True
    _chunk(c, rec, 2.0)
    pl = activity.prefill_live(rec, 0)
    assert pl["tps_live"] == 512.0 and pl["tps_live_avg"] == 512.0, pl


# ------------------------------------------------------------------ the hot path
def test_the_chunk_hook_appends_one_tuple_and_nothing_runs_per_token():
    """The chunk hook gained no work: one tuple a chunk, as it built before (a chunk of a
    prefill window replaces its window's mark instead, `activity.pf_mark`, 2026-10-09). The
    token path (`RequestRecord.track`) is unchanged (tests/test_live.py holds it)."""
    src = open(os.path.join(ROOT, "server", "app.py")).read()
    watch = src.split("        def watch(done: int, total: int) -> None:")[1].split(
        "        watch.info = {}")[0]
    code = "\n".join(ln.split("#")[0] for ln in watch.splitlines())
    mark = 'activity_mod.pf_mark(rec.pf_marks, done, total, now, watch.info.get("window"))'
    assert code.count(mark) == 1, code
    assert code.count("(") - code.count("()") <= 13 and ".pf =" not in code, code
    track = open(os.path.join(ROOT, "server", "usage.py")).read().split(
        "def track(self, source):")[1].split("def lock_acquired")[0]
    assert "anchor" not in track and "pf_marks" not in track


def test_an_older_record_without_marks_still_renders():
    """A record from before the hook's list (no `pf_marks`): the prefill block still renders."""
    c = Clock()
    rec = _rec(c, "old")
    rec.lock_acquired()
    rec.prompt_tokens = 5000
    del rec.pf_marks
    rec.prefill_info = {"start": 0, "chunk": 1024, "t0": c.perf(), "kind": None}
    p = activity.prefill_of(rec, c.perf())
    assert p is not None and p["tps_live"] is None and p["warmup"]["chunks"] == 0, p
    assert isinstance(usage.RequestRecord("x", "chat", True).pf_marks, list)


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            Clock.restore()
            print(f"  {name:64s} ok")
            passed += 1
    print(f"{passed} passed")
