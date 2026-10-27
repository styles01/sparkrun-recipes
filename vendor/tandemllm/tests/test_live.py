"""The live registry: phases, rates, the 30 s tail, the sampler gate, the schema.

No torch here, so it runs on any machine: the registry reads `RequestRecord`s, which import
nothing heavy. The route through the handler is in tests/test_live_app.py (needs the CPU engine).

Run: python tests/test_live.py
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server import live, usage  # noqa: E402
from tools import contract_check  # noqa: E402


_PERF = time.perf_counter


class Clock:
    """A clock and a perf counter that move together, by hand.

    The record stamps itself with `time.perf_counter`, so a Clock installs its own counter there
    and the real-time tests below call `Clock.restore()` first."""

    def __init__(self, t0: float = 1_790_000_000.0):
        self.t = t0
        usage.time.perf_counter = self.perf

    @staticmethod
    def restore() -> None:
        usage.time.perf_counter = _PERF

    def now(self) -> float:
        return self.t

    def perf(self) -> float:
        return self.t - 1_000_000.0            # perf_counter's zero is not the epoch

    def tick(self, s: float) -> None:
        self.t += s


class Blocks:
    def __init__(self, n=0):
        self.blocks = n


def _reg(clock, blocks=None, last_prefill=None, **kw):
    return live.LiveRegistry(blocks=lambda: blocks, last_prefill=lambda: last_prefill,
                             clock=clock.now, perf=clock.perf, **kw)


def _rec(clock, rid="chatcmpl-1", stream=True):
    r = usage.RequestRecord(rid, "chat", stream, t_arrival=clock.perf(), ts=clock.now())
    r.model, r.client_id, r.client_kind = "qwen38-spark-engine", "k:abc", "curl"
    return r


def _valid(snap):
    errs = contract_check.check(json.loads(json.dumps(snap)), "live")
    assert not errs, errs[:8]
    return snap


def _row(snap, rid):
    return next(r for r in snap["requests"] if r["request_id"] == rid)


# ------------------------------------------------------------------ phases

def test_a_request_moves_through_its_phases():
    c = Clock()
    reg = _reg(c)
    rec = _rec(c)
    reg.register(rec)
    assert _row(_valid(reg.snapshot()), "chatcmpl-1")["phase"] == "queued"
    c.tick(0.4)
    rec.lock_acquired()
    r = _row(_valid(reg.snapshot()), "chatcmpl-1")
    assert r["phase"] == "prefill" and r["queue_ms"] == 400.0 and r["ttft_ms"] is None
    rec.prompt_tokens = 300
    c.tick(0.6)
    src = rec.track(iter([1, 2, 3]))
    next(src)                                   # the first token: t_first
    r = _row(_valid(reg.snapshot()), "chatcmpl-1")
    assert r["phase"] == "decode" and r["tokens"] == 1 and r["ttft_ms"] == 1000.0
    list(src)
    rec.completion_tokens, rec.finish_reason = 3, "stop"
    rec.end()
    reg.finish(rec)
    r = _row(_valid(reg.snapshot()), "chatcmpl-1")
    assert r["phase"] == "done" and r["finish_reason"] == "stop" and r["tokens"] == 3
    assert r["ended_ms_ago"] == 0.0
    counts = reg.snapshot()["counts"]
    assert counts["in_flight"] == 0 and counts["served"] == 1 and counts["completed_1m"] == 1


def test_a_refusal_and_a_400_are_done_rows_without_a_lock_time():
    c = Clock()
    reg = _reg(c)
    ref = _rec(c, "chatcmpl-ref")
    reg.register(ref)
    ref.status, ref.finish_reason = 503, "refused"
    ref.end()
    reg.finish(ref)
    bad = _rec(c, "chatcmpl-bad")
    reg.register(bad)
    bad.lock_acquired()
    bad.status = 400
    bad.end()
    reg.finish(bad)
    snap = _valid(reg.snapshot())
    a, b = _row(snap, "chatcmpl-ref"), _row(snap, "chatcmpl-bad")
    assert a["phase"] == "done" and a["finish_reason"] == "refused" and a["status"] == 503
    assert a["queue_ms"] is None and a["tokens"] == 0
    assert b["phase"] == "done" and b["status"] == 400 and b["finish_reason"] is None
    counts = snap["counts"]
    assert counts["refused"] == 1 and counts["served"] == 1 and counts["errors"] == 0


# ------------------------------------------------------------------ rates

def _decoding(c, reg, rid="chatcmpl-1", prompt=300, reused=100, forwarded=200,
              prompt_s=0.5):
    """A request 1 s into decode with 41 tokens: the first token at t_first, 40 more since."""
    rec = _rec(c, rid)
    reg.register(rec)
    rec.lock_acquired()
    rec.prompt_tokens = prompt
    c.tick(prompt_s)
    src = rec.track(iter(range(41)))
    next(src)
    lp = {"reused": reused, "forwarded": forwarded, "ms": prompt_s * 1e3, "kind": "prefix"}
    c.tick(1.0)
    for _ in range(40):
        next(src)
    return rec, src, lp


def test_tokens_and_rates_while_decoding():
    c = Clock()
    blocks = Blocks(10)
    reg = _reg(c, blocks=blocks)
    rec, _, lp = _decoding(c, reg)
    reg.last_prefill_fn = lambda: lp
    r = _row(_valid(reg.snapshot()), "chatcmpl-1")
    assert r["tokens"] == 41 and r["decode_tps"] == 40.0, r
    assert r["prefill_tps"] == 400.0 and r["forwarded_tokens"] == 200 and r["cached_tokens"] == 100
    assert r["cache_source"] == "prefix" and r["prompt_ms"] == 500.0
    assert r["blocks"] == 10 and r["tokens_per_block"] == 4.0
    assert r["decode_ms"] == 1000.0 and r["elapsed_ms"] == 1500.0
    assert r["decode_tps_now"] is None            # no two samples yet


def test_a_side_slots_request_reads_its_own_slots_blocks():
    """`--parallel N`: a side slot's generation publishes its BlockStats in the slot's values,
    which `STATE` read from the live thread does not see; live (2026-10-16, Nemotron 3.5 Super VL,
    QSE_PARALLEL=3) a request on slot 1 had no rounds until it ended."""
    c = Clock()
    own = Blocks(8)
    own.t_first, own.t_last, own.accept = None, None, {3: {2: 8}}
    side = {1: own}
    reg = _reg(c, blocks=Blocks(20), slot_blocks=side.get, hz=4)
    rec, _, lp = _decoding(c, reg)
    rec.slot = 1
    reg.last_prefill_fn = lambda: lp
    r = _row(_valid(reg.snapshot()), "chatcmpl-1")
    assert r["blocks"] == 8 and r["tokens_per_block"] == 5.0, r
    assert r["activity"]["decode"]["rounds"] == 8 and \
        r["activity"]["decode"]["accept_mean"] == 2.0, r["activity"]["decode"]
    rec.slot = 0                                   # slot 0 is STATE's own
    assert _row(reg.snapshot(), "chatcmpl-1")["blocks"] == 20
    side.clear()                                   # a side slot without counts: none, not slot 0's
    rec.slot = 1
    assert _row(reg.snapshot(), "chatcmpl-1")["blocks"] is None


def test_the_final_row_equals_the_response_timings():
    c = Clock()
    reg = _reg(c, blocks=Blocks(10))
    rec, src, lp = _decoding(c, reg)
    c.tick(0.5)
    list(src)                                   # nothing left: 41 tokens in all
    rec.completion_tokens, rec.finish_reason = 41, "stop"
    rec.absorb_prefill(lp)
    rec.blocks = 10
    rec.end()
    reg.finish(rec)
    t = rec.timings()
    r = _row(_valid(reg.snapshot()), "chatcmpl-1")
    assert r["ttft_ms"] == t["ttft_ms"] and r["prefill_tps"] == t["prompt_per_second"]
    assert r["decode_tps"] == t["predicted_per_second"] and r["tokens"] == t["predicted_n"]
    assert r["tokens_per_block"] == t["tokens_per_block"] and r["decode_ms"] == t["predicted_ms"]
    assert r["elapsed_ms"] == t["total_ms"] and r["queue_ms"] == t["queue_ms"]


def test_the_last_two_samples_give_the_two_second_rate():
    c = Clock()
    reg = _reg(c)
    a = _rec(c, "a")
    b = _rec(c, "b")
    for rec in (a, b):
        reg.register(rec)
        rec.lock_acquired()
        rec.prompt_tokens = 10
    sa, sb = a.track(iter(range(1000))), b.track(iter(range(1000)))
    for _ in range(100):
        next(sa)
    for _ in range(50):
        next(sb)
    reg.tick()                                  # t=0: a=100, b=50
    c.tick(1.0)
    for _ in range(45):
        next(sa)
    for _ in range(20):
        next(sb)
    reg.tick()                                  # t=1: a=145, b=70
    c.tick(1.0)
    for _ in range(45):
        next(sa)
    for _ in range(30):
        next(sb)
    reg.tick()                                  # t=2: a=190, b=100
    snap = _valid(reg.snapshot())
    assert _row(snap, "a")["decode_tps_now"] == 45.0
    assert _row(snap, "b")["decode_tps_now"] == 25.0
    assert snap["now"]["decode_tps"] == 70.0    # the sum over both, tokens over the last 2 s
    assert snap["sample"]["decode_tps"] == 75.0  # the last second: 45 + 30
    assert [s["tokens"] for s in snap["history"]] == [150, 215, 290]
    assert snap["counts"] == {"in_flight": 2, "queued": 0, "prefilling": 0, "decoding": 2,
                              "completed_1m": 0, "served": 0, "errors": 0, "refused": 0}


def test_a_prefill_is_credited_to_the_second_it_finished_in():
    c = Clock()
    lp = {"reused": 0, "forwarded": 500, "ms": 250.0, "kind": None}
    reg = _reg(c, last_prefill=None)
    rec = _rec(c)
    reg.register(rec)
    rec.lock_acquired()
    rec.prompt_tokens = 500
    reg.tick()
    s = reg.snapshot()
    assert s["sample"]["prefill_tps"] is None and s["now"]["prefilling"] is True
    assert s["now"]["prefill_tps"] is None
    c.tick(0.25)
    src = rec.track(iter([1, 2]))
    next(src)
    reg.last_prefill_fn = lambda: lp
    c.tick(0.75)
    reg.tick()
    s = _valid(reg.snapshot())
    assert s["sample"]["prefill_tps"] == 2000.0 and s["now"]["prefilling"] is False
    assert s["now"]["prefill_tps"] == 2000.0 and s["now"]["last_prefill_ms_ago"] == 750.0
    c.tick(1.0)
    reg.tick()
    s = reg.snapshot()
    assert s["sample"]["prefill_tps"] is None      # credited once
    assert s["now"]["prefill_tps"] == 2000.0 and s["now"]["last_prefill_ms_ago"] == 1750.0


def test_a_fast_request_that_finished_between_ticks_still_credits_its_prefill():
    c = Clock()
    lp = {"reused": 0, "forwarded": 100, "ms": 100.0, "kind": None}
    reg = _reg(c, last_prefill=lambda: lp)
    rec = _rec(c)
    reg.register(rec)
    rec.lock_acquired()
    rec.prompt_tokens = 100
    c.tick(0.1)
    list(rec.track(iter([1, 2, 3])))
    rec.absorb_prefill(lp)
    rec.completion_tokens, rec.finish_reason = 3, "stop"
    rec.end()
    reg.finish(rec)
    c.tick(0.5)
    reg.tick()
    s = reg.snapshot()
    assert s["sample"]["prefill_tps"] == 1000.0
    assert s["sample"]["tokens"] == 3


# ------------------------------------------------------------------ the tail, the history

def test_a_finished_request_stays_thirty_seconds():
    c = Clock()
    reg = _reg(c)
    for rid in ("old", "new"):
        rec = _rec(c, rid)
        reg.register(rec)
        rec.lock_acquired()
        rec.prompt_tokens = 5
        list(rec.track(iter([1])))
        rec.completion_tokens, rec.finish_reason = 1, "stop"
        rec.end()
        reg.finish(rec)
        c.tick(2.0)                             # old ended 31 s before the check, new 29 s
        if rid == "old":
            c.tick(0.0)
    c.tick(27.0)
    snap = reg.snapshot()
    ids = [r["request_id"] for r in snap["requests"]]
    assert ids == ["new"], ids
    assert _row(snap, "new")["ended_ms_ago"] == 29000.0
    assert snap["counts"]["completed_1m"] == 2
    c.tick(40.0)
    assert reg.snapshot()["counts"]["completed_1m"] == 0


def test_the_history_holds_the_newest_three_hundred_samples():
    c = Clock()
    reg = _reg(c)
    rec = _rec(c)
    reg.register(rec)
    for _ in range(400):
        reg.tick()
        c.tick(1.0)
    h = reg.snapshot()["history"]
    assert len(h) == 300
    assert h[-1]["t"] - h[0]["t"] == 299.0
    assert reg.snapshot(history=False).get("history") is None


def test_rows_are_ordered_decoding_prefilling_queued_then_finished():
    c = Clock()
    reg = _reg(c)
    done = _rec(c, "done")
    reg.register(done)
    done.lock_acquired()
    done.prompt_tokens = 1
    list(done.track(iter([1])))
    done.completion_tokens, done.finish_reason = 1, "stop"
    done.end()
    reg.finish(done)
    dec = _rec(c, "dec")
    reg.register(dec)
    dec.lock_acquired()
    dec.prompt_tokens = 1
    next(dec.track(iter([1, 2])))
    pre = _rec(c, "pre")
    reg.register(pre)
    pre.lock_acquired()
    que = _rec(c, "que")
    reg.register(que)
    snap = _valid(reg.snapshot())
    assert [r["request_id"] for r in snap["requests"]] == ["dec", "pre", "que", "done"]
    assert snap["counts"]["in_flight"] == 3 and snap["counts"]["queued"] == 1
    assert snap["counts"]["prefilling"] == 1 and snap["counts"]["decoding"] == 1


# ------------------------------------------------------------------ the sampler thread

def test_the_sampler_sleeps_when_idle_and_wakes_for_a_request():
    Clock.restore()
    reg = live.LiveRegistry(blocks=lambda: None, last_prefill=lambda: None, interval=0.02,
                            keep_s=0.05)
    reg.start()
    try:
        time.sleep(0.15)
        assert reg.samples_taken == 0, reg.samples_taken
        rec = usage.RequestRecord("x", "chat", True)
        reg.register(rec)
        rec.lock_acquired()
        time.sleep(0.15)
        n = reg.samples_taken
        assert n >= 3, n
        rec.end()
        reg.finish(rec)
        time.sleep(0.2)                         # the tail closes the series, then silence
        m = reg.samples_taken
        time.sleep(0.15)
        assert reg.samples_taken == m, (m, reg.samples_taken)
        # a live stream keeps it awake without a request
        assert reg.subscribe() is True
        time.sleep(0.1)
        assert reg.samples_taken > m
        reg.unsubscribe()
    finally:
        reg.stop()


def test_at_most_four_streams():
    Clock.restore()
    reg = live.LiveRegistry(blocks=lambda: None, last_prefill=lambda: None)
    assert [reg.subscribe() for _ in range(5)] == [True, True, True, True, False]
    reg.unsubscribe()
    assert reg.subscribe() is True


def test_wait_tick_returns_after_the_next_sample():
    Clock.restore()
    reg = live.LiveRegistry(blocks=lambda: None, last_prefill=lambda: None, interval=0.02)
    reg.start()
    try:
        reg.subscribe()
        t0 = time.monotonic()
        assert reg.wait_tick(1.0) is True
        assert reg.wait_tick(1.0) is True
        assert time.monotonic() - t0 < 0.5
    finally:
        reg.stop()
    assert reg.wait_tick(0.05) is False        # stopped: nothing ticks



def test_a_stream_never_skips_an_event_that_came_between_two_waits():
    """A stream writer waits for an event newer than the one it sent, not for "the next tick":
    a tick that lands between the writer's read of `event()` and its next wait used to be
    skipped (activity_check on the box, 2026-09-27: 3 of 47 events), because `wait_tick`
    counts ticks from the moment it is called."""
    Clock.restore()
    reg = live.LiveRegistry(blocks=lambda: None, last_prefill=lambda: None)
    reg.subscribe()
    reg.tick()
    sent = reg.event()[0]
    reg.tick()                                   # lands while the writer is between two waits
    t0 = time.monotonic()
    assert reg.wait_event(sent, 1.0) is True     # the newer event is there: no wait
    assert reg.event()[0] == sent + 1            # and it is the very next one, nothing skipped
    assert time.monotonic() - t0 < 0.1
    assert reg.wait_event(reg.event()[0], 0.05) is False   # nothing newer: waits out the timeout
    assert reg.wait_tick(0.05) is False          # the old wait would have slept past that event

# ------------------------------------------------------------------ the hot path

def test_track_counts_every_token_and_nothing_else():
    Clock.restore()
    rec = usage.RequestRecord("x", "chat", True)
    out = list(rec.track(iter(range(1000))))
    assert len(out) == 1000 and rec.n_live == 1000
    rec.completion_tokens = len(out)
    assert rec.n_live == rec.completion_tokens
    # the registry never touches the record from the token path: only register/finish take the lock
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "server", "usage.py")).read()
    body = src.split("def track(self, source):")[1].split("def lock_acquired")[0]
    code = body.split('"""')[-1]                # after the docstring
    assert "acquire" not in code and "with " not in code and ".append" not in code, code
    assert "dict(" not in code and "[]" not in code and "{}" not in code, code


def test_the_example_validates_and_the_snapshot_is_json():
    c = Clock()
    reg = _reg(c)
    snap = reg.snapshot()
    _valid(snap)
    assert snap["contract_version"] == "1.2" and snap["sample"] is None and snap["history"] == []
    assert snap["now"] == {"decode_tps": None, "prefill_tps": None, "prefilling": False,
                           "tokens_per_block": None, "last_prefill_ms_ago": None,
                           "prefill_live_tps": None, "prefill_live_count": 0,
                           "decode_live_tps": None, "decode_warming": 0}
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs",
                        "contract", "dashboard-v1")
    with open(os.path.join(root, "live.example.json")) as f:
        errs = contract_check.check(json.load(f), "live")
    assert not errs, errs[:8]


def test_register_and_finish_are_thread_safe():
    c = Clock()
    reg = _reg(c)

    def churn(i):
        for j in range(50):
            rec = _rec(c, f"r{i}-{j}")
            reg.register(rec)
            rec.lock_acquired()
            rec.prompt_tokens = 1
            list(rec.track(iter([1])))
            rec.completion_tokens, rec.finish_reason = 1, "stop"
            rec.end()
            reg.finish(rec)
            reg.tick()
    ts = [threading.Thread(target=churn, args=(i,)) for i in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    snap = reg.snapshot()
    assert snap["counts"]["served"] == 200 and snap["counts"]["in_flight"] == 0


# ------------------------------------------------------------------ contract 1.1

def _busy(c, reg, rid="busy"):
    rec = _rec(c, rid)
    reg.register(rec)
    rec.lock_acquired()
    rec.prompt_tokens = 10
    src = rec.track(iter(range(10_000)))
    next(src)
    return rec, src


def test_four_hertz_while_busy_with_a_stream_one_hertz_otherwise():
    c = Clock()
    reg = _reg(c, hz=4)
    assert reg.cadence() == 1.0                     # nothing in flight, nobody watching
    rec, src = _busy(c, reg)
    assert reg.cadence() == 1.0                     # in flight, nobody watching
    reg.subscribe()
    assert reg.cadence() == 0.25
    seqs = []
    for _ in range(12):                             # 3 s at 4 Hz
        reg.tick()
        seqs.append(reg.event()[0])
        for _ in range(10):
            next(src)
        c.tick(0.25)
    assert seqs == list(range(seqs[0], seqs[0] + 12)), seqs
    hist = reg.snapshot()["history"]
    assert len(hist) == 3, [h["t"] for h in hist]   # history stays one sample a second
    ev = reg.event()[1].decode()
    assert ev.startswith(f"event: live\nid: {seqs[-1]}\ndata: {{"), ev[:60]
    snap = json.loads(ev.split("data: ", 1)[1])
    _valid(snap)
    assert snap["seq"] == seqs[-1] and snap["interval_s"] == 0.25
    row = snap["requests"][0]
    assert row["activity"]["decode"]["tps_now"] == 40.0, row["activity"]["decode"]
    reg.unsubscribe()


def test_one_encode_per_tick_for_every_reader():
    """The tick encodes; the streams only write. However many read, the encoding work is the
    same, and all of them get the very same bytes."""
    c = Clock()
    reg = _reg(c, hz=4)
    _busy(c, reg)
    real = live.json.dumps

    def work(readers):
        calls = []

        def counting(*a, **k):
            calls.append(1)
            return real(*a, **k)
        for _ in range(readers):
            reg.subscribe()
        live.json.dumps = counting
        try:
            before = reg.encodes
            reg.tick()
            got = [reg.event() for _ in range(readers)]      # what each stream writes
        finally:
            live.json.dumps = real
            for _ in range(readers):
                reg.unsubscribe()
        assert reg.encodes == before + 1
        assert len({id(g[1]) for g in got}) == 1
        return len(calls)
    work(1)                                         # warm: `recent` is encoded once, then reused
    assert work(1) == work(4)
    n = reg.encodes
    reg.tick()
    assert reg.encodes == n                         # nobody watching: nothing encoded


def test_the_tick_event_is_the_snapshot():
    """The tick's event splices finished rows and `recent` encoded once; it must say exactly
    what a fresh snapshot says."""
    c = Clock()
    reg = _reg(c, hz=4)
    for i in range(3):
        rec = _rec(c, f"old{i}")
        reg.register(rec)
        rec.lock_acquired()
        rec.prompt_tokens = 5
        list(rec.track(iter([1, 2, 3])))
        rec.completion_tokens, rec.finish_reason = 3, "length" if i else "stop"
        rec.end()
        reg.finish(rec)
        c.tick(1.0)
    _busy(c, reg)
    reg.subscribe()
    for _ in range(3):
        reg.tick()
        c.tick(0.25)
    reg.tick()
    ev = json.loads(reg.event()[1].decode().split("data: ", 1)[1])
    snap = reg.snapshot(history=False)
    _valid(ev)
    for d in (ev, snap):
        d.pop("seq"), d.pop("sampler")
    assert ev == snap, json.dumps([ev, snap])[:600]
    reg.unsubscribe()


def test_recent_keeps_twenty_for_fifteen_minutes():
    c = Clock()
    reg = _reg(c)
    for i in range(25):
        rec = _rec(c, f"r{i}")
        reg.register(rec)
        rec.lock_acquired()
        rec.prompt_tokens = 5
        if i == 7:
            rec.finish_reason = "abandoned"          # left during its prefill
        else:
            list(rec.track(iter([1, 2])))
            rec.completion_tokens, rec.finish_reason = 2, "stop"
        rec.end()
        reg.finish(rec)
        c.tick(20.0)
    snap = _valid(reg.snapshot())
    ids = [r["request_id"] for r in snap["recent"]]
    assert ids == [f"r{i}" for i in range(24, 4, -1)], ids
    ab = next(r for r in snap["recent"] if r["request_id"] == "r7")
    assert ab["stop"]["reason"] == "abandoned" and ab["stop"]["state"] == "prefilling"
    assert "silent prefill" in ab["stop"]["sentence"]
    c.tick(15 * 60 - 20 * 10)                       # r14 and older are past 15 minutes
    ids = [r["request_id"] for r in reg.snapshot()["recent"]]
    assert ids[-1] == "r15" and len(ids) == 10, ids


def test_a_recent_row_says_where_its_prompt_came_from():
    """The Last 20 table's cache cell -- the prompt, what came out of a cache, what
    the prefill forwarded and the source, as the ledger row has them; none for a request whose
    prefill never finished."""
    c = Clock()
    reg = _reg(c)
    for rid, src, cached, fwd in (("e", "end", 73300, 80), ("d", "disk", 28672, None)):
        rec = _rec(c, rid)
        reg.register(rec)
        rec.lock_acquired()
        rec.prompt_tokens = 73374 if rid == "e" else 33956
        rec.cached_tokens, rec.cache_source, rec.prompt_n = cached, src, fwd
        list(rec.track(iter([1, 2])))
        rec.completion_tokens, rec.finish_reason = 2, "stop"
        rec.end()
        reg.finish(rec)
        c.tick(1.0)
    gone = _rec(c, "g")
    reg.register(gone)
    gone.lock_acquired()
    gone.prompt_tokens, gone.finish_reason = 9000, "abandoned"
    gone.end()
    reg.finish(gone)
    snap = _valid(reg.snapshot())
    by = {r["request_id"]: r for r in snap["recent"]}
    assert {k: by["e"][k] for k in ("prompt_tokens", "cached_tokens", "forwarded_tokens",
                                    "cache_source")} == {
        "prompt_tokens": 73374, "cached_tokens": 73300, "forwarded_tokens": 80,
        "cache_source": "end"}, by["e"]
    # no forwarded count from the prefill: the prompt less what was cached
    assert (by["d"]["forwarded_tokens"], by["d"]["cache_source"]) == (5284, "disk")
    assert by["g"]["cache_source"] is None and by["g"]["prompt_tokens"] is None


def test_a_relayed_recent_row_names_its_backend():
    """0.11.3: a cloud request's row in the Last 20 says which backend answered, and no engine
    cache tier: its cached tokens are the provider's (DeepSeek's prompt_cache_hit_tokens). It
    read "cold, 105K reused" beside the engine's own rows (2026-10-17)."""
    from proxy.measure import RelayRecord
    c = Clock()
    reg = _reg(c)
    rec = RelayRecord("chatcmpl-px-1", "chat", True, t_arrival=c.perf())
    rec.ts = c.now()
    rec.model, rec.client_id, rec.client_kind = "deepseek-flash", "anon", "curl"
    rec.backend, rec.backend_kind = "deepseek", "deepseek"
    reg.register(rec)
    rec.stamp_sent()
    c.tick(1.0)
    rec.stamp_token(c.perf(), "reasoning")
    c.tick(2.0)
    rec.stamp_token(c.perf(), "content")
    rec.set_usage({"prompt_tokens": 46420, "completion_tokens": 9950, "cached_tokens": 46208,
                   "reasoning_tokens": 3407})
    rec.finish_reason = "tool_calls"
    rec.end()
    reg.finish(rec)
    snap = _valid(reg.snapshot())
    r = next(x for x in snap["recent"] if x["request_id"] == "chatcmpl-px-1")
    assert r["backend"] == "deepseek" and r["cache_source"] is None
    assert (r["prompt_tokens"], r["cached_tokens"], r["tokens"]) == (46420, 46208, 9950)
    own = _rec(c, "e")
    reg.register(own)
    own.lock_acquired()
    own.prompt_tokens, own.cache_source = 10, "end"
    list(own.track(iter([1, 2])))
    own.completion_tokens, own.finish_reason = 2, "stop"
    own.end()
    reg.finish(own)
    assert next(x for x in reg.snapshot()["recent"] if x["request_id"] == "e")["backend"] is None


def test_the_kill_switch_keeps_the_new_fields_null_at_one_hertz():
    c = Clock()
    reg = _reg(c, hz=4, activity=False)
    rec, _ = _busy(c, reg)
    reg.subscribe()
    assert reg.cadence() == 1.0
    reg.tick()
    snap = _valid(json.loads(reg.event()[1].decode().split("data: ", 1)[1]))
    assert snap["contract_version"] == "1.2" and snap["recent"] is None
    assert snap["engine"]["waiting_for_client"] is None and snap["interval_s"] == 1.0
    assert all(r["activity"] is None and r["timeline"] is None for r in snap["requests"])
    assert reg.snapshot()["requests"][0]["phase"] == "decode"      # 1.0 unchanged
    assert live.LiveRegistry(hz=0).activity is False               # QSE_LIVE_HZ=0
    reg.unsubscribe()


def test_the_hz_setting():
    assert live.hz_from_env(None) == 4 and live.hz_from_env("") == 4
    assert [live.hz_from_env(v) for v in ("0", "1", "2", "4")] == [0, 1, 2, 4]
    for bad in ("3", "8", "fast"):
        try:
            live.hz_from_env(bad)
        except SystemExit:
            continue
        raise AssertionError(bad)


def test_every_one_point_zero_field_is_still_there():
    """A 1.0 client reads a 1.1 snapshot: every 1.0 field present, same meaning."""
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs",
                        "contract", "dashboard-v1")
    schema = json.load(open(os.path.join(root, "live.schema.json")))
    c = Clock()
    reg = _reg(c)
    _busy(c, reg)
    snap = reg.snapshot()
    one_zero = ("contract_version", "generated_at", "interval_s", "counts", "now", "requests",
                "sample", "history")
    assert all(k in snap for k in one_zero)
    req10 = ("request_id", "phase", "finish_reason", "status", "model", "client", "endpoint",
             "stream", "thinking", "temperature", "prompt_tokens", "cached_tokens",
             "forwarded_tokens", "tokens", "blocks", "tokens_per_block", "elapsed_ms", "queue_ms",
             "prompt_ms", "ttft_ms", "decode_ms", "prefill_tps", "decode_tps", "decode_tps_now",
             "cache_source", "max_tokens", "ended_ms_ago")
    assert all(k in snap["requests"][0] for k in req10)
    assert set(req10) <= set(schema["$defs"]["request"]["required"])


def test_a_row_names_its_conversation_and_its_thinking_budget():
    """SRV-347: chatd follows its own requests by the conversation id it sends (`x-session-id:
    chat-<id>`), and the panel shows the reasoning cap in force; both null when absent."""
    c = Clock()
    reg = _reg(c)
    rec = _rec(c, "chatcmpl-chat")
    rec.conv = "chat-01JTEST"

    class Think:
        budget = 8192
    rec.live_refs = (Think(), None)
    reg.register(rec)
    plain = _rec(c, "chatcmpl-plain")
    reg.register(plain)
    snap = _valid(reg.snapshot())
    r = _row(snap, "chatcmpl-chat")
    assert r["conversation_id"] == "chat-01JTEST" and r["thinking_budget"] == 8192, r
    p = _row(snap, "chatcmpl-plain")
    assert p["conversation_id"] is None and p["thinking_budget"] is None, p
    Think.budget = 0                       # uncapped
    assert _row(reg.snapshot(), "chatcmpl-chat")["thinking_budget"] is None


def test_the_recorded_box_messages_validate():
    """Tests/fixtures/live-1.1-box.json: real 1.1 events from the box, the UI's reference."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fx = json.load(open(os.path.join(root, "tests", "fixtures", "live-1.1-box.json")))
    assert len(fx["messages"]) >= 8
    states = set()
    for name, msg in fx["messages"].items():
        _valid(msg)
        states |= {r["activity"]["state"] for r in msg["requests"] if r["activity"]}
    assert {"prefilling", "thinking", "tool_call", "done"} <= states, states


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:64s} ok")
            passed += 1
    print(f"{passed} passed")
