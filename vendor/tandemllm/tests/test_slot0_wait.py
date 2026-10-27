"""Parallel requests: a request whose conversation slot 0 holds waits for slot 0 only briefly, then
takes a free side slot (0.8.2). CPU only.

The live case (2026-10-13, Flash-Next 3.05, `--parallel 3`, two 65,536-token side slots): slot 0 ran
an Open WebUI chat's 19,291-token `write_file` call for 496 s. A new Open WebUI chat (9,093 tokens,
its first user message at 9,087) matched another chat's NVMe entry at 8,192 tokens -- Open WebUI's
system prompt and tool list -- was sent to slot 0 as "continues its conversation (spill)", and waited
there 436,810 ms while slots 1 and 2 were free; the Lanes view said only "Queued". /health showed
`max_running: 2`, the high-water mark, which read as a cap.

  * `_par_side_need`: a spill match at the client's head is no conversation of its own (the request
    takes a side slot at once); a real continuation (view, resident, spill past the first user
    message) gets its side window with `slot0_first`, and one too long for every side slot still
    waits for slot 0;
  * the handler, in process on the fake engine with slot 0 held: the continuing request is served
    on a side slot after `SIDE_SLOT0_WAIT_S`, while slot 0 is still held; with the old unbounded
    wait it is still queued, and its queued line says why;
  * `Pool.limit`: every slot built, fewer only with a reason (the floor at load, a memory wait).

Run: python tests/test_slot0_wait.py
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_par_continue import HEAD, FakeResident, _need, _restore, _session, _setup, _turns  # noqa: E402,I001

from server import app as A  # noqa: E402
from server import slots as S  # noqa: E402


class FakeSpill:
    """The NVMe spill as `_par_side_need` asks it: one entry, offered at `line` tokens."""

    def __init__(self, line: int):
        self.entries = {"e1": object()}
        self.line = line
        self.stats = {}

    def candidate(self, res, keys, max_len, **kw):
        return (object(), self.line, 0)


@contextlib.contextmanager
def _spill(sp):
    was = A.STATE.get("spill")
    A.STATE["spill"] = sp
    try:
        yield
    finally:
        if was is None:
            A.STATE.pop("spill", None)
        else:
            A.STATE["spill"] = was


def _owui_new_chat():
    """A new chat: the client's head (system prompt and tools), its first user message, the
    generation prompt; and the first user message's start, as `_conv_start` finds it."""
    ids = _session(9, HEAD, 6) + [7, 8, 9]
    return ids, len(HEAD)


# ------------------------------------------------------------------ the decision
def test_a_spill_match_at_the_clients_head_is_a_new_conversation():
    """The live bug's first half: the entry shares only Open WebUI's head (8,192 of 9,093 tokens,
    the first user message at 9,087): no wait for slot 0, a side slot takes it at once."""
    ids, conv = _owui_new_chat()
    other_chat = FakeResident(_session(1, HEAD, 3_000))      # slot 0 holds another chat
    pool, saved = _setup(other_chat)
    real = A._conv_start
    A._conv_start = lambda ids_: conv
    try:
        with _spill(FakeSpill(line=conv - 895)):
            need, why = _need(ids, max_new=32_768, plan_active=False)
        assert need is not None and "wait" not in why and not why.get("slot0_first"), why
        assert why["spill_head"] == conv - 895
        assert pool.stats.get("spill_waits", 0) == 0
    finally:
        A._conv_start = real
        _restore(saved)


def test_a_spilled_conversation_tries_slot_0_first():
    """An entry that holds the conversation past its first user message: slot 0 first (its prefill
    reads the spill), with the side window ready for when slot 0 stays busy."""
    ids, conv = _owui_new_chat()
    ids = ids + [5_000_000 + i for i in range(3_000)]           # a later turn of that chat
    pool, saved = _setup(FakeResident(_session(1, HEAD, 3_000)))
    real = A._conv_start
    A._conv_start = lambda ids_: conv
    try:
        with _spill(FakeSpill(line=conv + 2_048)):
            need, why = _need(ids, max_new=32_768, plan_active=False)
        assert why["wait"] == "spill" and why["slot0_first"] and why["spill"] == conv + 2_048, why
        assert need == len(ids) + 32_768 + A.SIDE_MARGIN, need
        assert pool.stats["spill_waits"] == 1
    finally:
        A._conv_start = real
        _restore(saved)


def test_a_continuing_turn_gets_its_side_window_with_slot_0_first():
    r1, rows, r2 = _turns()
    pool, saved = _setup(FakeResident(rows), view=r1)
    try:
        need, why = _need(r2)
        assert why["wait"] == "view" and why["slot0_first"], why
        assert need == len(r2) + 32_000 + A.SIDE_MARGIN + A.PLAN_MARGIN
        assert pool.stats["resident_waits"] == 1
        assert "home" not in why and "prefer_side" not in why
    finally:
        _restore(saved)


def test_a_continuing_turn_too_long_for_a_side_slot_waits_for_slot_0():
    r1, rows, r2 = _turns()
    pool, saved = _setup(FakeResident(rows), view=r1, side_len=8_192)
    try:
        need, why = _need(r2)
        assert need is None and why["wait"] == "view" and why["too_long"], why
        assert "slot0_first" not in why
        assert pool.stats.get("too_long", 0) == 0, "counted as a continuation, not as too long"
    finally:
        _restore(saved)


# ------------------------------------------------------------------ the queued line
def test_the_queued_line_says_why():
    pool = S.Pool([S.Slot(0, 262_144), S.Slot(1, 65_536), S.Slot(2, 65_536)])
    me = {"rank": A.RANK_TURN, "t": time.monotonic()}
    w = A._queue_why(pool, {"wait": "spill", "slot0_first": True}, 40_000, 0.5, me)
    assert w == (f"slot 0 reads its conversation back from the NVMe; a free side slot takes it "
                 f"after {A.SIDE_SLOT0_WAIT_S:g} s"), w
    w = A._queue_why(pool, {"wait": "view", "slot0_first": True}, 40_000, 0.5, me)
    assert w.startswith("slot 0 holds its conversation"), w
    w = A._queue_why(pool, {"wait": "too_long"}, None, 30.0, me)
    assert w == "too long for a side slot (65,536 tokens); waits for slot 0", w
    w = A._queue_why(pool, {"images": True}, None, 30.0, me)
    assert "images" in w
    for s in pool.slots:
        s.lock.acquire()
    try:
        w = A._queue_why(pool, {"wait": "view", "slot0_first": True}, 40_000, 9.0, me)
        assert w == "all 3 slots are busy", w
    finally:
        for s in pool.slots:
            s.lock.release()
    from server import activity
    assert activity.label({"state": "queued", "queue": {"place": None, "reason": w}}) == \
        "Queued: all 3 slots are busy"
    assert activity.label({"state": "queued", "queue": {"place": 2}}) == "Queued, about 2nd in line"


# ------------------------------------------------------------------ /health
def test_max_running_follows_the_slots_unless_memory_forbids_it():
    adm = S.Admission(15 * S.GiB, meminfo=lambda: {"MemAvailable": 16 * S.GiB})
    pool = S.Pool([S.Slot(0, 262_144), S.Slot(1, 65_536), S.Slot(2, 65_536)], admission=adm)
    r = pool.report()
    assert r["max_running"] == 3 and r["max_running_reason"] is None and r["requested"] == 3, r
    assert "peak_running" in r
    # a request that waits for memory: the ones running are all that may, and why
    adm.reserve("a", 0)                                     # alone: admitted without the check
    ok, _ = adm.reserve("b", 4 * S.GiB)
    assert not ok
    adm.wait("b")
    pool.slots[1].lock.acquire()
    try:
        n, why = pool.limit()
        assert n == 1 and why.startswith("memory: 1 request(s) wait for memory (MemAvailable 16.0"), why
    finally:
        pool.slots[1].lock.release()
    adm.release("b")
    # the floor kept a slot from being built at load
    pool.requested, pool.limit_reason = 4, "slot 3 NOT built (window 65,536, ~1.6 GiB): ..."
    r = pool.report()
    assert r["max_running"] == 3 and r["requested"] == 4 and r["max_running_reason"].startswith(
        "slot 3 NOT built"), r


def test_peak_running_is_the_high_water_mark():
    pool = S.Pool([S.Slot(0, 262_144, lock=threading.Lock()), S.Slot(1, 65_536)])
    pool.slots[0].lock.acquire()
    s = pool.take_side(100)
    pool.took_slot0()
    assert pool.stats["peak_running"] == 2 and pool.stats["slot0_grants"] == 1
    pool.release_side(s)
    pool.slots[0].lock.release()
    assert pool.report()["max_running"] == 2 and pool.stats["peak_running"] == 2


# ------------------------------------------------------------------ through the handler
def _fake_parallel():
    from server import fake_engine
    a = A.parser().parse_args(["--fake-engine", "--fake-tps", "2000", "--max-len", "65536",
                               "--default-max-tokens", "64", "--parallel", "3",
                               "--side-max-len", "8192", "--side-lane", "off"])
    A.STATE.clear()
    A.CLOCK = A.qwait.TurnClock()
    with contextlib.redirect_stdout(io.StringIO()):
        fake_engine.load(A, a)
    return A.STATE["pool"]


def _body():
    words = "lorem ipsum " * 180       # past one prefill chunk (the fake: a token a character)
    return {"model": "TandemLLM", "stream": False, "max_tokens": 32,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": f"Summarize: {words}"}]}


def _send(body, out):
    from test_ui_states import Req
    t0 = time.monotonic()
    head, raw = Req(body, {"User-Agent": "curl/8.5"}).response()
    out.update(head=head, raw=raw, t=time.monotonic() - t0)


def _queued_rec():
    for rec in list(A.live_registry()._live.values()):
        if getattr(rec, "t_lock", None) is None:
            return rec
    return None


def _run_held(wait_s: float):
    """Slot 0 held (a long run between two blocks: the lock held, the GPU turn free), then a request
    that continues slot 0's conversation. Returns (pool, result, still waiting after 1.5 s, the
    queued line seen)."""
    pool = _fake_parallel()
    body = _body()
    ids = A.build_prompt(body, None, count="ids")
    assert len(ids) >= 1_024, len(ids)
    A.STATE["slot0_view"] = list(ids)                      # its conversation is slot 0's
    was = A.SIDE_SLOT0_WAIT_S
    A.SIDE_SLOT0_WAIT_S = wait_s
    A.LOCK.acquire()
    A.LOCK.turn.release(0)
    with A.QUEUE:                                          # a client's request holds it
        A.HOLDER["who"] = {"rank": A.RANK_TURN, "t": time.monotonic(), "t_hold": time.monotonic()}
    out: dict = {}
    th = threading.Thread(target=_send, args=(body, out))
    try:
        th.start()
        line = None
        t_end = time.monotonic() + 1.5
        while time.monotonic() < t_end and th.is_alive():
            rec = _queued_rec()
            if rec is not None and getattr(rec, "queue_why", None):
                line = rec.queue_why
            time.sleep(0.05)
        waiting = th.is_alive()
        if not waiting:
            assert A.LOCK.locked(), "served while slot 0 was still held"
    finally:
        A.SIDE_SLOT0_WAIT_S = was
        with A.QUEUE:
            A.HOLDER["who"] = None
        A.LOCK._lk.release()
        th.join(30)
    return pool, out, waiting, line


def test_the_handler_takes_a_side_slot_after_the_short_wait():
    pool, out, waiting, line = _run_held(0.6)
    assert not waiting, "served on a side slot while slot 0 was held"
    assert '"choices"' in out["raw"], out["raw"][:300]
    assert out["t"] >= 0.6, out["t"]
    assert pool.stats["resident_waits"] == 1 and pool.stats["slot0_gave_up"] == 1, pool.stats
    assert pool.stats["side_grants"] == 1 and pool.stats["slot0_grants"] == 0, pool.stats
    assert line and line.startswith("slot 0 holds its conversation; a free side slot takes it"), line


def test_without_the_bound_it_waits_behind_slot_0():
    """The 0.8.0 behaviour (an unbounded wait): still queued while slot 0 is held; served on slot 0
    once it is free."""
    pool, out, waiting, line = _run_held(1e9)
    assert waiting, "it waited for slot 0"
    assert '"choices"' in out["raw"]
    assert pool.stats.get("slot0_gave_up", 0) == 0 and pool.stats["slot0_grants"] == 1, pool.stats
    assert line and line.startswith("slot 0 holds its conversation"), line


def test_the_queued_line_while_the_router_asks():
    """With the conductor on, a turn is queued while the router's questions run, before the wait
    loop: its line says TandemLLM decides how to handle it (it said only "Queued", live)."""
    from test_ui_states import _go, serve
    cond, _ = serve("on")
    seen = []
    real = cond.plan

    def plan(*a, **kw):
        seen.extend(getattr(r, "queue_why", None) for r in A.live_registry()._live.values()
                    if getattr(r, "t_lock", None) is None)
        return real(*a, **kw)
    cond.plan = plan
    _go({"model": "TandemLLM", "stream": False, "max_tokens": 50,
         "messages": [{"role": "user", "content": "Name two rivers."}]},
        {"User-Agent": "Python/3.12 aiohttp/3.12.15"})
    assert A.OWN_WAIT["question"] in seen, seen


if __name__ == "__main__":
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok    {name}")
            except Exception as exc:                               # noqa: BLE001
                failed += 1
                import traceback
                traceback.print_exc()
                print(f"FAIL  {name}: {exc}")
    sys.exit(1 if failed else 0)
