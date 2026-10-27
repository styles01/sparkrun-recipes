"""SRV-213 (FX10): a conversation that ran on a side slot is carried into slot 0, on the random model.

THE MEASUREMENT. With `--parallel 2` a new chat or agent session that starts while slot 0 holds a
long conversation is a guest of it (at most 8,192 tokens and under a quarter of it) and runs on
the idle side slot (SRV-207). A side slot keeps no conversation: when the session outgrew the guest
limit it moved to slot 0 and read everything past the client's head on the NVMe again -- UX2 O07
on 15,910 tokens, 7,168 from the disk, 8.7k re-read in 18.3 s; UX1 W11 on a 0.4.0 test build
beside an 80k resident: cached 0.44, TTFT 17.0 s.

THE FIX. At the end of a request on a side slot the server notes what the slot's rows hold
(server/app.py `_side_keep_end`); the conversation's next turn on slot 0 moves slot 0's resident
conversation out as another session's request would (`ResidentPrefix.vacate`: the park or the
NVMe), copies the side slot's rows and its state at the end into slot 0 and resumes at that end
(`_carry_from_side`). Verify blocks wrote that state, so the resume is held to the end anchor's
gate: the same greedy tokens as the same conversation run on slot 0 throughout.

  1  the carried turn resumes at the side slot's end and writes the reference's tokens
  2  off: the turn reads its prompt again, as 0.4.0 did (the measurement's shape)
  3  slot 0's conversation goes to the park and its next turn resumes from it
  4  another request on the side slot in between: nothing is carried (`Slot.gen`)
  5  a guest of slot 0 does not take a side slot whose end is fresh (the router's question
     before the conversation's next user message), unless asked to
  6  the hidden spans of the side request's prompt are replayed at the carried end

Run: python tests/test_side_carry.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import test_forward_tree as T  # noqa: E402,F401  (the CPU environment before the engine imports)

import torch  # noqa: E402

from server import app  # noqa: E402
from server import slots as S  # noqa: E402
from test_cache import fresh, tokens  # noqa: E402
from test_end_anchor import BIG, CH, TOOL, TOOL2, _state, turn  # noqa: E402

HEAD = tokens(64, seed=90)                     # the client's head, shared by every chat
R_PROMPT = HEAD + tokens(150, seed=91)         # slot 0's long conversation
S_PROMPT = HEAD + tokens(70, seed=92)          # the new chat, a guest of R by size


def _pool(**kw):
    kw.setdefault("park_bytes", BIG)
    res = _state(**kw)
    vals = {k: None for k in S.OVERLAY}
    vals.update(engine=fresh(), drafter=None, max_len=512)
    side = S.Slot(1, 512, values=vals)
    pool = S.Pool([S.Slot(0, 512), side])
    app.STATE["pool"] = pool
    return res, pool, side


def side_turn(pool, ids, n, session):
    s = pool.take_side(len(ids) + n, owner=session)
    assert s is not None
    S.bind(s)
    try:
        return turn(ids, n, session)
    finally:
        S.bind(None)
        pool.release_side(s)


def slot0_turn(ids, n, session):
    """A request on slot 0 as the handler serves it: the carry, then the stream and `_remember`."""
    with torch.no_grad():
        app._carry_from_side(list(ids), session)
    return turn(ids, n, session)


def _reference():
    """The new chat's two turns on slot 0 alone: turn 2 resumes at turn 1's end anchor."""
    _state()
    out1, _, _ = turn(S_PROMPT, 60, session="S")
    out2, pre2, _ = turn(S_PROMPT + out1 + TOOL, 24, session="S")
    assert pre2["kind"] == "end"
    return out1, out2


def _run(carry="on"):
    old = app.SIDE_CARRY
    app.SIDE_CARRY = carry
    try:
        res, pool, side = _pool()
        out_r, _, _ = turn(R_PROMPT, 40, session="R")             # slot 0's conversation
        r_end = res.end["at"]
        out1, _, _ = side_turn(pool, S_PROMPT, 60, "S")          # the new chat, on the side slot
        assert res.valid == r_end, "the side slot leaves slot 0 alone"
        rec = side.carry
        out2, pre2, _ = slot0_turn(S_PROMPT + out1 + TOOL, 24, "S")
        return res, pool, side, rec, out_r, out1, out2, pre2
    finally:
        app.SIDE_CARRY = old


def test_the_next_turn_on_slot_0_resumes_at_the_side_slots_end():
    ref1, ref2 = _reference()
    res, pool, side, rec, _, out1, out2, pre2 = _run()
    assert out1 == ref1, "a side slot alone writes what slot 0 writes"
    assert rec is not None and rec["at"] == len(S_PROMPT) + len(out1) - 1, rec
    assert pre2["kind"] == "side" and pre2["reused"] == rec["at"], pre2
    assert pre2["forwarded"] == len(TOOL) + 1, pre2
    assert out2 == ref2, "the gate: the same greedy tokens as an end anchor on slot 0"
    assert res.stats["side_carries"] == 1 and side.carry is None
    return f"resumed at {pre2['reused']}, forwarded {pre2['forwarded']}, same {len(out2)} tokens"


def test_off_the_turn_reads_its_prompt_again():
    _, ref2 = _reference()
    res, _, side, rec, _, _, out2, pre2 = _run(carry="off")
    assert rec is None and side.carry is None
    assert pre2["kind"] != "side" and pre2["reused"] <= len(HEAD), pre2
    assert out2 == ref2
    return f"off: reused {pre2['reused']} (the head), forwarded {pre2['forwarded']}"


def test_slot_0s_conversation_goes_to_the_park_and_comes_back():
    res, pool, side, rec, out_r, _, _, pre2 = _run()
    assert pre2["kind"] == "side"
    assert res.park_owner == "R" and res.park_rows > 0, (res.park_owner, res.park_rows)
    _, pre_r, _ = turn(R_PROMPT + out_r + TOOL2, 8, session="R")
    assert res.stats["park_swaps"] >= 1 and pre_r["reused"] > len(HEAD), (pre_r, res.stats)
    return f"R parked, came back at {pre_r['reused']}"


def test_another_request_on_the_side_slot_in_between_carries_nothing():
    old = app.SIDE_CARRY
    try:
        app.SIDE_CARRY = "on"
        res, pool, side = _pool()
        turn(R_PROMPT, 40, session="R")
        out1, _, _ = side_turn(pool, S_PROMPT, 60, "S")
        rec = side.carry
        s = pool.take_side(100, owner="x")        # another request takes the slot
        pool.release_side(s)
        assert side.carry is rec and rec["gen"] != side.gen
        _, pre2, _ = slot0_turn(S_PROMPT + out1 + TOOL, 8, "S")
        assert pre2["kind"] != "side" and res.stats["side_carries"] == 0, pre2
    finally:
        app.SIDE_CARRY = old


def test_a_guest_does_not_take_a_side_slot_whose_end_is_fresh():
    old = app.SIDE_CARRY
    try:
        app.SIDE_CARRY = "on"
        res, pool, side = _pool()
        turn(R_PROMPT, 40, session="R")
        side_turn(pool, S_PROMPT, 30, "S")
        assert side.carry is not None
        assert pool.take_side(100, owner="q", keep_carry_s=app.CARRY_KEEP_S) is None
        side.carry["t"] -= app.CARRY_KEEP_S + 1   # an old end no longer keeps a guest away
        s = pool.take_side(100, owner="q", keep_carry_s=app.CARRY_KEEP_S)
        assert s is side
        pool.release_side(s)
    finally:
        app.SIDE_CARRY = old


def test_a_short_side_request_notes_no_end_and_keeps_no_guest_away():
    """The router's own questions and a title task (150 to 450 tokens on the box) run on the side
    slot too: under two chunks their end is no conversation to carry, and noting it would send
    every following guest to slot 0 for ten minutes."""
    old = app.SIDE_CARRY
    try:
        app.SIDE_CARRY = "on"
        res, pool, side = _pool()
        turn(R_PROMPT, 40, session="R")
        side_turn(pool, tokens(20, seed=95), 5, "q")       # 25 tokens < 2 x CH
        assert side.carry is None
        s = pool.take_side(100, owner="g", keep_carry_s=app.CARRY_KEEP_S)
        assert s is side
        pool.release_side(s)
    finally:
        app.SIDE_CARRY = old


def test_the_side_requests_hidden_spans_are_replayed_at_the_carried_end():
    """A router note at the side request's tail is in the side slot's rows, not in the client's
    copy: the carried end lists it and the replay puts it back, as at any end anchor."""
    old = app.SIDE_CARRY
    try:
        app.SIDE_CARRY = "on"
        res, pool, side = _pool()
        turn(R_PROMPT, 40, session="R")
        note = tokens(5, seed=93)
        served = S_PROMPT + note
        s = pool.take_side(len(served) + 40, owner="S")
        S.bind(s)
        try:
            # the handler hands the served prompt's spans to `_remember` (`hidden=`)
            out1, _, _ = turn(served, 40, session="S", hidden=[(len(S_PROMPT), list(note))])
        finally:
            S.bind(None)
            pool.release_side(s)
        rec = side.carry
        assert rec["hidden"] == [(len(S_PROMPT), list(note))], rec["hidden"]
        copy = S_PROMPT + out1 + TOOL                 # the client's copy, without the note
        with torch.no_grad():
            info = app._carry_from_side(list(copy), "S")
        assert info and info["end"], info
        ids, spans = res.replay(list(copy))
        assert spans and ids[:rec["at"]] == rec["tokens"], spans
    finally:
        app.SIDE_CARRY = old


def test_w11_a_copy_without_the_reasoning_resumes_at_the_carried_end():
    """UX1 W11 (Open WebUI): the new chat's first answer thought on the side slot, and its
    follow-up came back without the reasoning (SRV-239's shape). The carried end gets the end
    alignment's region, so the copy aligns and resumes at the carried end, as on slot 0."""
    from server import endalign
    from test_end_align import NL, SP, tb_of
    prompt = HEAD + tokens(69, seed=94) + [NL]
    _state()
    out, _, _ = turn(prompt, 60, session="S")
    k = next(i for i in range(12, 30) if out[i] not in out[:i] and out[i] not in prompt
             and out[i] not in (NL, SP))
    old = app.SIDE_CARRY
    try:
        app.SIDE_CARRY = "on"
        res, pool, side = _pool()
        al = app.STATE["end_align"] = endalign.EndAlign(tb_of({out[k]: endalign.THINK_END}))
        turn(R_PROMPT, 40, session="R")
        out1, _, _ = side_turn(pool, prompt, 60, "S")
        assert out1 == out
        copy = prompt[:-1] + [SP] + out[k:] + TOOL        # `<think>\n\n</think>\n\n` + the answer
        with torch.no_grad():
            info = app._carry_from_side(list(copy), "S")
        assert info and info["end"], info
        e = res.end["at"]
        assert al.regions(res) == [(len(prompt), e, len(prompt) + k)], al.regions(res)
        ids, spans, aligned = app._end_replay(res, list(copy))
        assert aligned > 0 and ids[:e] == res.tokens[:e]
        _, pre, _ = turn(ids, 12, session="S")
        assert pre["kind"] == "side" and pre["reused"] == e, pre
        return f"aligned {aligned} tokens, resumed at the carried end {e}"
    finally:
        app.SIDE_CARRY = old


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"  {name:72s} ok   {fn() or ''}", flush=True)
            passed += 1
    print(f"{passed} passed")
