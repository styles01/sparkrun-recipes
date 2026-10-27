"""Parallel requests, phase A: the scheduler's parts on their own (server/slots.py), CPU only.

  * the GPU turn: exclusive, handed over only to a waiter that is a quantum behind in GPU time,
    never handed over by a holder that is alone, and `held_by` true only for the holder;
  * the joint round's barrier: two decoding slots meet in one round run by the last submitter,
    a slot that left decode does not keep the other waiting, and a slot alone never waits;
  * the row shares keep a joint pass at or under 32 rows, and a block that cannot share runs
    in a pass of its own;
  * the admission counts bytes: alone always admitted, otherwise MemAvailable less the caches'
    growth, the other requests' reservations and this one's worst case against the floor; a
    reservation released frees its bytes; the check is atomic;
  * the side slots: the smallest free one whose window holds the request, none when every one is
    busy or too short;
  * the per-request keys a side slot keeps cover everything the handler publishes.

Run: python tests/test_slots.py
"""

from __future__ import annotations

import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from server import slots  # noqa: E402

GiB = slots.GiB


# ------------------------------------------------------------------ the turn
def test_turn_is_exclusive_and_alone_never_hands_over():
    t = slots.Turn(quantum_s=0.0)
    t.acquire(0)
    assert t.held_by(0) and not t.held_by(1)
    assert t.pause(0) is False, "alone, a holder keeps the turn"
    got = []

    def other():
        t.acquire(1)
        got.append(time.monotonic())
        t.release(1)

    th = threading.Thread(target=other)
    th.start()
    time.sleep(0.05)
    assert not got, "the turn is exclusive"
    t.release(0)
    th.join(2)
    assert got, "released, the waiter gets it"


def test_pause_hands_over_to_a_waiter_behind_and_comes_back():
    clock = [0.0]
    t = slots.Turn(quantum_s=0.05, clock=lambda: clock[0])
    t.acquire(0)
    order = []

    def other():
        t.acquire(1)
        order.append("1 runs")
        t.release(1)

    th = threading.Thread(target=other)
    th.start()
    while not t.waiting():
        time.sleep(0.001)
    clock[0] = 0.01                         # slot 0 is only 10 ms ahead: under the quantum
    assert t.pause(0) is False
    clock[0] = 0.2                          # 200 ms ahead: the waiter goes first
    assert t.pause(0) is True
    order.append("0 back")
    th.join(2)
    assert order == ["1 runs", "0 back"], order
    assert t.held_by(0)
    t.release(0)
    assert t.stats["handoffs"] == 1


def test_pause_by_a_thread_that_does_not_hold_it_is_a_no_op():
    t = slots.Turn()
    assert t.pause(3) is False
    t.acquire(0)
    assert t.pause(1) is False
    assert t.held_by(0)
    t.release(0)


# ------------------------------------------------------------------ the barrier
class _Req:
    pass


def _pool(n=2, timeout=0.005):
    ss = [slots.Slot(i, 1024, values=None if i == 0 else {}) for i in range(n)]
    turn = slots.Turn(quantum_s=0.0)
    meet = slots.Meeting(32, timeout)
    return slots.Pool(ss, turn=turn, meeting=meet), turn, meet


def test_two_decoding_slots_meet_in_one_round_run_by_the_last():
    pool, turn, meet = _pool(2)
    for s in pool.slots:
        s.phase = "decode"
    runs = []

    def run(m):
        with m.cv:
            pend = list(m.pending.values())
            m.pending.clear()
            runs.append(sorted(r.slot for r in pend))
            for r in pend:
                r.out, r.ms = f"logits-{r.slot}", False
                m.transit.add(r.slot)
            m.cv.notify_all()

    res = {}

    def slot(i):
        turn.acquire(i)
        req = slots.VerifyReq(i, None, None, [-1, 0, 1], 0)
        slots.barrier(meet, pool, turn, i, req, run)
        res[i] = (req.out, turn.held_by(i))
        turn.release(i)

    ths = [threading.Thread(target=slot, args=(i,)) for i in (0, 1)]
    for th in ths:
        th.start()
    for th in ths:
        th.join(5)
    assert runs == [[0, 1]], f"one joint round for both: {runs}"
    assert res[0][0] == "logits-0" and res[1][0] == "logits-1"
    assert res[0][1] and res[1][1], "each slot holds the turn again when the barrier returns"
    assert meet.stats["timeouts"] == 0


def test_a_prefill_holding_the_turn_does_not_starve_a_split_round():
    """SRV-311: slots 1 and 2 decode, slot 1 waits in this round, slot 2 is in another meeting
    (the packed draft) and never submits here, and slot 3 holds the turn for a long prefill
    (not decoding). The waiter times out and runs its round once it gets the turn, instead of
    waiting for the prefill to end."""
    pool, turn, meet = _pool(4)
    pool.slots[1].phase = pool.slots[2].phase = "decode"
    runs = []

    def run(m):
        with m.cv:
            pend = list(m.pending.values())
            m.pending.clear()
            runs.append(sorted(r.slot for r in pend))
            for r in pend:
                r.out, r.ms = f"logits-{r.slot}", False
            m.cv.notify_all()

    stop = threading.Event()

    def prefill():
        turn.acquire(3)
        while not stop.is_set():
            time.sleep(0.02)              # a prefill chunk
            turn.pause(3)                 # offered between chunks
        turn.release(3)

    pf = threading.Thread(target=prefill)
    pf.start()
    time.sleep(0.05)
    res = {}

    def slot1():
        turn.acquire(1)
        req = slots.VerifyReq(1, None, None, [-1, 0, 1], 0)
        t0 = time.monotonic()
        slots.barrier(meet, pool, turn, 1, req, run)
        res["s"] = time.monotonic() - t0
        res["out"] = req.out
        turn.release(1)

    th = threading.Thread(target=slot1)
    th.start()
    th.join(3)
    stop.set()
    pf.join(3)
    assert res.get("out") == "logits-1", res
    assert res["s"] < 1.0, f"waited {res['s']:.2f} s behind the prefill"
    assert runs == [[1]] and meet.stats["timeouts"] == 1


def test_a_slot_that_left_decode_does_not_keep_the_other_waiting():
    pool, turn, meet = _pool(2, timeout=0.002)
    pool.slots[0].phase = "decode"
    pool.slots[1].phase = "decode"
    done = []

    def run(m):
        with m.cv:
            pend = list(m.pending.values())
            m.pending.clear()
            for r in pend:
                r.out = "x"
            done.append(len(pend))
            m.cv.notify_all()

    turn.acquire(0)
    req = slots.VerifyReq(0, None, None, [-1, 0], 0)

    def leave():
        time.sleep(0.05)
        pool.slots[1].phase = ""               # slot 1's request ended without a block

    th = threading.Thread(target=leave)
    th.start()
    t0 = time.monotonic()
    slots.barrier(meet, pool, turn, 0, req, run)
    th.join()
    assert req.out == "x" and done == [1]
    assert time.monotonic() - t0 < 2.0
    turn.release(0)


def test_the_turn_idle_runs_the_round_after_the_timeout():
    pool, turn, meet = _pool(2, timeout=0.002)
    pool.slots[0].phase = pool.slots[1].phase = "decode"   # slot 1 decodes but never submits

    def run(m):
        with m.cv:
            for r in m.pending.values():
                r.out = "y"
            m.pending.clear()
            m.cv.notify_all()

    turn.acquire(0)
    req = slots.VerifyReq(0, None, None, [-1, 0], 0)
    slots.barrier(meet, pool, turn, 0, req, run)
    assert req.out == "y" and meet.stats["timeouts"] == 1
    assert turn.held_by(0)
    turn.release(0)


def test_shares_keep_a_joint_pass_under_32_rows():
    assert slots.share(1) is None, "alone, a slot keeps the router's own budget"
    for n in range(2, 9):
        s = slots.share(n)
        assert s is not None and s >= 2 and s * n <= 32, (n, s)
    assert slots.share(2) == 16 and slots.share(4) == 8


def test_batches_cut_at_max_rows_and_keep_unbatchable_blocks_alone():
    meet = slots.Meeting(32, 0.002)
    meet.pending = {0: slots.VerifyReq(0, None, None, list(range(16)), 0),
                    1: slots.VerifyReq(1, None, None, list(range(16)), 0),
                    2: slots.VerifyReq(2, None, None, list(range(8)), 0, batchable=False),
                    3: slots.VerifyReq(3, None, None, list(range(8)), 0)}
    groups = sorted([r.slot for r in g] for g in meet.batches())
    assert groups == [[0, 1], [2], [3]], groups
    assert all(sum(len(meet.pending[i].parents) for i in g) <= 32 for g in groups)


# ------------------------------------------------------------------ admission
def test_admission_alone_is_always_admitted():
    adm = slots.Admission(15 * GiB, meminfo=lambda: {"MemAvailable": 1 * GiB})
    ok, d = adm.reserve("a", 10 * GiB)
    assert ok and d.get("alone"), "a lone request is the single-sequence server's request"
    adm.release("a")
    assert adm.reserved() == 0


def test_admission_counts_the_requests_in_flight_and_the_growth():
    avail = [40 * GiB]
    adm = slots.Admission(15 * GiB, meminfo=lambda: {"MemAvailable": avail[0]},
                          growth=lambda: 5 * GiB)
    assert adm.reserve("a", 8 * GiB)[0]
    ok, d = adm.reserve("b", 8 * GiB)       # 40 - 5 - 8 - 8 = 19 >= 15
    assert ok, d
    ok, d = adm.reserve("c", 8 * GiB)       # 40 - 5 - 16 - 8 = 11 < 15
    assert not ok and d["after"] == 11 * GiB, d
    assert "c" not in adm.held
    adm.release("a")
    ok, d = adm.reserve("c", 8 * GiB)       # 40 - 5 - 8 - 8 = 19
    assert ok, d
    assert slots.Admission.describe(d).endswith("(>= floor 15.0)")


def test_admission_without_meminfo_admits_and_says_so():
    adm = slots.Admission(15 * GiB, meminfo=lambda: None)
    adm.reserve("a", 1)
    ok, d = adm.reserve("b", 100 * GiB)
    assert ok and d["avail"] is None
    assert "not checked" in slots.Admission.describe(d)


def test_admission_check_and_reserve_are_atomic():
    """Two requests racing for the last bytes: exactly one gets them."""
    adm = slots.Admission(10 * GiB, meminfo=lambda: {"MemAvailable": 30 * GiB})
    adm.reserve("first", 1 * GiB)            # not alone any more: the floor applies
    wins = []
    barrier = threading.Barrier(8)

    def go(k):
        barrier.wait()
        if adm.reserve(k, 12 * GiB)[0]:
            wins.append(k)

    ths = [threading.Thread(target=go, args=(f"r{i}",)) for i in range(8)]
    for th in ths:
        th.start()
    for th in ths:
        th.join()
    # 30 - 1 - 12 = 17 >= 10 for the first; 30 - 13 - 12 = 5 < 10 for any second
    assert len(wins) == 1, wins


def test_slot_bytes_follow_the_byte_math():
    assert slots.slot_bytes(65536, constant=0) == 65536 * 106_752
    assert slots.slot_bytes(0) == 7 * GiB + GiB // 2


# ------------------------------------------------------------------ side slots
def test_take_side_gives_the_smallest_free_slot_that_fits():
    ss = [slots.Slot(0, 262144), slots.Slot(1, 65536, values={}), slots.Slot(2, 16384, values={})]
    pool = slots.Pool(ss)
    s = pool.take_side(10000, owner="a")
    assert s is ss[2] and s.owner == "a" and s.busy
    s2 = pool.take_side(10000, owner="b")
    assert s2 is ss[1], "the next free one that fits"
    assert pool.take_side(100, owner="c") is None, "every side slot busy"
    pool.release_side(s)
    assert pool.take_side(20000) is None, "16k is too short for 20k, and 64k is busy"
    assert pool.take_side(16000) is ss[2]
    assert pool.stats["side_grants"] == 3 and pool.stats["peak_running"] >= 2


def test_the_overlay_covers_what_the_generation_publishes():
    for k in ("engine", "drafter", "max_len", "resident", "mem", "last_ctx", "last_prompt_n",
              "hidden_spans", "blocks", "last_prefill", "lane", "disk_cache", "spill"):
        assert k in slots.OVERLAY, k


if __name__ == "__main__":
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            n += 1
            print(f"ok  {name}")
    print(f"{n} passed")
