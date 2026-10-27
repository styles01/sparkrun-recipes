"""FX16 (0.4.8): two chats beside each other -- the side slot keeps its conversation.

THE MEASUREMENT (2026-10-08, 0.4.7, Open WebUI, chats A and B). A, a 64,000-token tool loop, ran on
slot 0; B, a new chat, took the side slot (13,352 tokens, slot 1) and decoded beside it. B's next
turn found slot 0 idle and was carried into it (SRV-213), which moved A to the NVMe (83,160
tokens, 4.1 s). From then on both chats could only run on slot 0: with the served max_tokens of
32,768 and the router's margin, any prompt past about 24,000 tokens was "too long" for the
65,536-token side slot (`need = prompt + max_tokens + margins`), so each turn waited for the
other's (31, 127, 180, 107 and 138 s in the queue) while the side slot stood idle.

THE FIX, three parts (server/app.py):
  1  a side slot takes a request whose window leaves it at least SIDE_MIN_ROOM rows of output,
     with max_tokens cut to the room (`_par_side_need`, `_side_room`)
  2  a new conversation beside one a side slot could never hold takes an idle side slot before
     slot 0 (`prefer_side`), so the long conversation is not evicted for it
  3  a side slot has a resident prefix of its own (`_side_resident`): the next turn of the
     conversation it holds names it as its home (`_side_home`) and resumes at the end anchor
     there, slot 0's conversation untouched; a guest in between (the router's question) leaves
     it in place through the stash; the carry into slot 0 stays the fallback

  CPU only for 1-3's choice (tests/test_par_continue.py's stand-ins); the resume itself on the
  random 4-layer model (tests/test_side_carry.py's pool), held to the end anchor's gate: the same
  greedy tokens as the conversation run on slot 0.

Run: python tests/test_side_home.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import test_forward_tree as T  # noqa: E402,F401  (the CPU environment before the engine imports)

import torch  # noqa: E402

from engine import cache  # noqa: E402
from server import app, endalign  # noqa: E402
from server import slots as S  # noqa: E402
from test_cache import fresh, tokens  # noqa: E402
from test_end_align import tb_of  # noqa: E402
from test_end_anchor import BIG, CH, TOOL, _state, turn  # noqa: E402
from test_par_continue import CHUNK, HEAD, FakeResident, _need, _restore, _session, _setup  # noqa: E402
from test_side_carry import HEAD as SC_HEAD, R_PROMPT, S_PROMPT, side_turn, slot0_turn  # noqa: E402


# ------------------------------------------------------------------ 1. the window and the cut
def test_a_long_chat_takes_the_side_slot_with_its_limit_cut():
    """28,000 tokens + 32,768 + margins is over the 65,536 window; with the output cut to the
    room it fits (0.4.7: 'too_long', the request waited for slot 0)."""
    r1 = _session(1, HEAD, 1_500) + [7, 8, 9]
    pool, saved = _setup(FakeResident(r1), view=r1)
    try:
        chat_b = _session(5, HEAD, 21_000)
        need, why = _need(chat_b)
        margin = app.SIDE_MARGIN + app.PLAN_MARGIN
        assert need is not None and "wait" not in why, why
        assert why["clip"] and need == len(chat_b) + margin + app.SIDE_MIN_ROOM, (need, why)
        room = app._side_room(pool.side()[0], len(chat_b), margin, 32_000)
        assert room == 65_536 - len(chat_b) - margin and room >= app.SIDE_MIN_ROOM, room
        assert pool.stats.get("too_long", 0) == 0
    finally:
        _restore(saved)
    return f"need {need} (cut), max_tokens 32000 -> {room}"


def test_too_long_even_with_the_limit_cut_still_waits():
    r1 = _session(1, HEAD, 1_500) + [7, 8, 9]
    pool, saved = _setup(FakeResident(r1), view=r1)
    try:
        need, why = _need(_session(3, HEAD, 50_000))
        assert need is None and why["wait"] == "too_long" and pool.stats["too_long"] == 1
    finally:
        _restore(saved)


# ------------------------------------------------------------------ 2. a new chat beside a big one
def test_a_new_conversation_beside_a_long_one_prefers_the_side_slot():
    big = _session(1, HEAD, 60_000)                   # slot 0's conversation: never a side slot's
    pool, saved = _setup(FakeResident(big), view=big)
    try:
        need, why = _need(_session(6, HEAD, 2_000))
        assert need is not None and why.get("prefer_side") == len(big), why
        # off: slot 0 first, as 0.4.7
        was = app.SIDE_NEW_FIRST
        app.SIDE_NEW_FIRST = "off"
        try:
            need, why = _need(_session(6, HEAD, 2_000))
            assert "prefer_side" not in why, why
        finally:
            app.SIDE_NEW_FIRST = was
    finally:
        _restore(saved)
    # FX17: a short new chat beside the long one waits for a busy side slot (prefer_side_wait);
    # a long new chat does not
    pool, saved = _setup(FakeResident(big), view=big)
    try:
        need, why = _need(_session(6, HEAD, 2_000))
        assert why.get("prefer_side_wait") is True, why
        need, why = _need(_session(6, HEAD, 12_000))
        assert need is not None and why.get("prefer_side") and "prefer_side_wait" not in why, why
    finally:
        _restore(saved)
    small = _session(1, HEAD, 3_000)                  # a side slot could hold this one
    pool, saved = _setup(FakeResident(small), view=small)
    try:
        need, why = _need(_session(6, HEAD, 2_000))
        assert need is not None and "prefer_side" not in why, why
    finally:
        _restore(saved)


# ------------------------------------------------------------------ 3. the home slot
class SideResident(FakeResident):
    def __init__(self, tokens, chunk=CHUNK):
        super().__init__(tokens, chunk)
        self.end = {"at": self.valid}


def _side_values(rows, view):
    vals = {k: None for k in S.OVERLAY}
    vals.update(max_len=65_536, resident=SideResident(rows), slot_view=list(view))
    return vals


def test_a_turn_of_a_side_conversation_names_its_home():
    r1 = _session(1, HEAD, 1_500) + [7, 8, 9]
    b1 = _session(9, HEAD, 4_000) + [7, 8, 9]                        # B's last prompt, served on slot 1
    gen = [888_000 + i for i in range(900)]
    b_rows = b1 + gen
    pool, saved = _setup(FakeResident(r1), view=r1)
    try:
        pool.slots[1].values = _side_values(b_rows, b1)
        b2 = b1[:-1] + [6] + gen[:300] + [7, 8, 9]                   # B's next turn
        need, why = _need(b2)
        assert need is not None and why.get("home") == 1, why
        # another chat: no home
        need, why = _need(_session(10, HEAD, 900) + [7, 8, 9])
        assert need is not None and "home" not in why, why
        # B's turn grown past the window: no home (it runs on slot 0, carried in)
        huge = b2 + [5_000_000 + i for i in range(60_000)]
        need, why = _need(huge)
        assert need is None and why["wait"] == "too_long" and "home" not in why, why
        # SRV-528: no end anchor (the engine was ahead), the rows in place: still its home
        pool.slots[1].values["resident"].end = None
        need, why = _need(b2)
        assert why.get("home") == 1, why
        # the slot cleared (another conversation ran there: `begin` cut the rows, the view went)
        pool.slots[1].values["resident"].valid = 0
        pool.slots[1].values["slot_view"] = None
        need, why = _need(b2)
        assert "home" not in why, why
    finally:
        pool.slots[1].values = None
        _restore(saved)


def test_take_slot_names_one_slot_and_never_waits():
    pool = S.Pool([S.Slot(0, 512), S.Slot(1, 512), S.Slot(2, 1024)])
    s = pool.take_slot(2, 700, owner="b")
    assert s is pool.slots[2] and s.busy and s.gen == 1
    assert pool.take_slot(2, 700, owner="c") is None, "busy: no waiting"
    assert pool.take_slot(1, 700, owner="c") is None, "too short"
    assert pool.take_slot(9, 10) is None
    pool.release_side(s)
    assert pool.take_slot(2, 700, owner="c") is pool.slots[2]


# ------------------------------------------------------------------ the resume, on the random model
def _pool_home():
    """Slot 0 with its resident prefix, a side slot with one of its own (FX16)."""
    res = _state(park_bytes=BIG)
    vals = {k: None for k in S.OVERLAY}
    side_res = cache.ResidentPrefix(BIG, CH, stash_bytes=BIG, guest_max=200, guest_frac=1.0,
                                    end=True)
    side_al = endalign.EndAlign(tb_of())
    vals.update(engine=fresh(), drafter=None, max_len=512, resident=side_res, end_align=side_al)
    side = S.Slot(1, 512, values=vals)
    pool = S.Pool([S.Slot(0, 512), side])
    app.STATE["pool"] = pool
    return res, pool, side, side_res


def _reference():
    _state()
    out1, _, _ = turn(S_PROMPT, 60, session="S")
    out2, pre2, _ = turn(S_PROMPT + out1 + TOOL, 24, session="S")
    assert pre2["kind"] == "end"
    return out1, out2


def home_turn(pool, side, ids, n, session):
    s = pool.take_slot(side.index, len(ids) + n, owner=session)
    assert s is side
    S.bind(s)
    try:
        return turn(ids, n, session)
    finally:
        S.bind(None)
        pool.release_side(s)


def test_a_side_conversation_resumes_on_its_own_slot():
    ref1, ref2 = _reference()
    old = app.SIDE_CARRY
    try:
        app.SIDE_CARRY = "on"
        res, pool, side, side_res = _pool_home()
        out_r, _, _ = turn(R_PROMPT, 40, session="R")
        r_end = res.end["at"]
        out1, _, _ = side_turn(pool, S_PROMPT, 60, "S")
        assert out1 == ref1
        rec = side.carry
        assert rec is not None and rec["resident"] and rec["at"] == side_res.end["at"], rec
        assert side_res.end["at"] == len(S_PROMPT) + len(out1) - 1
        out2, pre2, _ = home_turn(pool, side, S_PROMPT + out1 + TOOL, 24, "S")
        assert pre2["kind"] == "end" and pre2["reused"] == rec["at"], pre2
        assert pre2["forwarded"] == len(TOOL) + 1, pre2
        assert out2 == ref2, "the gate: the same greedy tokens as the conversation on slot 0"
        assert res.valid == r_end and res.owner == "R", "slot 0's conversation untouched"
        assert res.stats["parks"] == 0 and res.stats["side_carries"] == 0, res.stats
        assert app._carry_live(side, side.carry) and side.carry["at"] == side_res.end["at"]
        # slot 0's conversation goes on where it was
        _, pre_r, _ = turn(R_PROMPT + out_r + TOOL, 8, session="R")
        assert pre_r["kind"] == "end" and pre_r["reused"] == r_end, pre_r
        return f"resumed on slot 1 at {pre2['reused']}, forwarded {pre2['forwarded']}; slot 0 at {r_end}"
    finally:
        app.SIDE_CARRY = old


def test_a_guest_between_the_turns_leaves_the_side_conversation_in_place():
    ref1, ref2 = _reference()
    old = app.SIDE_CARRY
    try:
        app.SIDE_CARRY = "on"
        res, pool, side, side_res = _pool_home()
        turn(R_PROMPT, 40, session="R")
        out1, _, _ = side_turn(pool, S_PROMPT, 60, "S")
        rec = side.carry
        side_turn(pool, tokens(20, seed=95), 5, "q")         # the router's question, a guest
        assert side_res.stats["guests"] == 1 and side_res.stats["stash_restored"] >= 0
        assert side.carry is rec and app._carry_live(side, rec), "the record outlives the guest"
        out2, pre2, _ = home_turn(pool, side, S_PROMPT + out1 + TOOL, 24, "S")
        assert pre2["kind"] == "end" and pre2["reused"] == rec["at"], pre2
        assert out2 == ref2
        return f"guest in between; resumed at {pre2['reused']}"
    finally:
        app.SIDE_CARRY = old


def test_the_carry_into_slot_0_after_a_guest_uses_the_end_anchors_state():
    """The fallback (the side slot busy, the prompt outgrown): the carry copies the side
    resident's end pair, not the live state the guest wrote."""
    ref1, ref2 = _reference()
    old = app.SIDE_CARRY
    try:
        app.SIDE_CARRY = "on"
        res, pool, side, side_res = _pool_home()
        turn(R_PROMPT, 40, session="R")
        out1, _, _ = side_turn(pool, S_PROMPT, 60, "S")
        side_turn(pool, tokens(20, seed=95), 5, "q")
        out2, pre2, _ = slot0_turn(S_PROMPT + out1 + TOOL, 24, "S")
        assert pre2["kind"] == "side" and res.stats["side_carries"] == 1, pre2
        assert out2 == ref2
        assert side_res.valid == 0 and side_res.end is None, "the side slot holds nobody's now"
        assert side.values.get("slot_view") is None
        return f"carried after a guest, resumed at {pre2['reused']}"
    finally:
        app.SIDE_CARRY = old


def test_another_conversation_on_the_side_slot_takes_the_home():
    old = app.SIDE_CARRY
    try:
        app.SIDE_CARRY = "on"
        res, pool, side, side_res = _pool_home()
        turn(R_PROMPT, 40, session="R")
        out1, _, _ = side_turn(pool, S_PROMPT, 60, "S")
        other = SC_HEAD + tokens(240, seed=97)               # too long for a guest: evicts S
        side_turn(pool, other, 20, "O")
        assert side.carry["owner"] == "O" and side_res.owner == "O"
        with torch.no_grad():
            assert app._carry_candidate(pool, res, S_PROMPT + out1 + TOOL, 512) is None
    finally:
        app.SIDE_CARRY = old


def test_a_guest_never_swaps_the_park_in_or_takes_the_end_anchor():
    """0.4.8 run 4: two router questions ran on slot 0 as guests while the side slot was busy;
    the park (another conversation with its end anchor) was swapped in for them and the
    15,8k-token chat lost its rows and end anchor (its next turn re-read it). A copy shorter than
    the parked end is another request: `unpark_end` moves nothing, the resident keeps its end."""
    res = _state(park_bytes=BIG)
    eng = app.STATE["engine"]
    out_p, _, _ = turn(S_PROMPT, 30, session="P")                 # P: later parked with its end
    out_a, _, _ = turn(R_PROMPT, 40, session="A")                 # A replaces P: P parked
    assert res.park_owner == "P" and res.park_end is not None, (res.park_owner, res.park_end)
    a_end, a_valid, a_tokens = res.end["at"], res.valid, list(res.tokens)
    question = S_PROMPT[:40] + tokens(30, seed=96)                 # a short request, shares P's head
    with torch.no_grad():
        moved = res.unpark_end(eng, None, question, 512)
    assert moved == 0, moved
    assert res.owner == "A" and res.end["at"] == a_end and res.valid == a_valid
    assert res.tokens[:a_valid] == a_tokens[:a_valid] and res.park_owner == "P"
    # and A's next turn still resumes at its end
    _, pre, _ = turn(R_PROMPT + out_a + TOOL, 8, session="A")
    assert pre["kind"] == "end" and pre["reused"] == a_end, pre
    return f"park kept for P, A resumed at {a_end}"


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"  {name:72s} ok   {fn() or ''}", flush=True)
            passed += 1
    print(f"{passed} passed")
