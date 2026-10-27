"""FX16: the end alignment's third pass -- a block the client moved to its last message.

THE MEASUREMENT (2026-10-08 12:14:30, live 0.4.7, Open WebUI). A chat with a web source attached:
Open WebUI puts its retrieval block ("### Task: Respond to the user query using the provided
context ... <context>...</context>") into the LAST user message of every request. For 45 tool-loop
turns the user message was the first one, so the rows held the block there and every turn resumed
at the end anchor. The next user message took the block with it: the first message came bare, the
copy parted from the rows at token 12,000 of 83,160, the hidden-span replay and both alignment
passes refused it, the chat's NVMe entry was not offered, and the turn read 60,814 tokens again
(91 s); the router's notes, hashed on the client's prefix, were dropped with it.

THE RULE (server/endalign.py `_moved_spans`). In a user message of the rows, the copy's message is
the rows' message with ONE contiguous run of bytes taken out, token-aligned, and that run is in the
copy's last user message: the run is served from the rows (as a hidden span is), everything else
of the message must be the rows' token for token. The model sees the message as it saw it when it
answered, and the block again where the client put it now. A message the user edited is not that
(its bytes are not in the last message) and is served as sent, as before.

  1  the block moved: the copy resumes at the end anchor, served with the rows' tokens, and
     writes the same greedy tokens as a cold prefill of the served prompt (the gate); the
     journal's count (`last_moved`) is the run's tokens
  2  the message edited (a token changed): served as sent; the run not in the last message
     (removed, not moved): served as sent; off (`moved=False`): served as sent; the refusal
     says where the copy parted (`last_refusal`)
  3  two moved blocks (the block moved twice over three turns)
  4  with a thinking answer: the reasoning pass and the moved pass together

Run: python tests/test_end_align_moved.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import test_forward_tree as T  # noqa: E402,F401  (the CPU environment before the engine imports)

from server import app, endalign  # noqa: E402
from test_cache import tokens  # noqa: E402
from test_end_align import NL, SP, _cold, _first, _setup, tb_of  # noqa: E402
from test_end_anchor import turn  # noqa: E402

IM_S, USER, IM_E = 90, 91, 92                     # the template's marks in these tests
MARKS = (IM_S, [USER], IM_E)


def _ids(n, seed):
    return [t if t not in (IM_S, USER, IM_E, NL, SP) else 1 for t in tokens(n, seed=seed)]


HEAD = _ids(40, 200)                              # the client's system prompt
BLOCK = _ids(30, 201)                             # the client's retrieval block
Q1, Q2, Q3 = _ids(20, 202), _ids(15, 203), _ids(12, 204)


def user(*parts):
    out = [IM_S, USER, NL]
    for p in parts:
        out += list(p)
    return out + [IM_E]


def _setup_moved(**kw):
    res, al = _setup(**kw)
    al.marks = MARKS
    return res, al


def test_the_block_moved_resumes_at_the_end():
    res, al = _setup_moved()
    prompt = HEAD + user(BLOCK, [NL], Q1) + [NL]           # turn 1 as the client sent it
    out, ctx = _first(prompt)
    e = res.end["at"]
    # turn 2: the first message bare, the block in the new last message
    copy = HEAD + user(Q1) + [NL] + out + user(BLOCK, [NL], Q2) + [NL]
    ids, spans, aligned = app._end_replay(res, copy)
    assert ids[:e] == res.tokens[:e], "served with the rows' tokens"
    assert ids[e:] == copy[len(copy) - (len(ids) - e):], "the client's tail after the end"
    a, b = al.last_moved[0]
    assert res.tokens[a:b] == BLOCK + [NL], (a, b)
    assert aligned > 0 and al.stats["aligned_moved"] == 1 and al.stats["moved_tokens"] == b - a
    got, pre, _ = turn(ids, 12)
    assert pre["kind"] == "end" and pre["reused"] == e, pre
    assert got == _cold(ids, 12), "the end anchor's gate: the same greedy tokens as cold"
    return f"moved {b - a} tokens, resumed at {e}, forwarded {pre['forwarded']}"


def test_an_edited_message_and_a_removed_block_are_served_as_sent():
    res, al = _setup_moved()
    prompt = HEAD + user(BLOCK, [NL], Q1) + [NL]
    out, ctx = _first(prompt)
    e = res.end["at"]
    edited = list(Q1)
    edited[5] = 2 if edited[5] != 2 else 3
    copy = HEAD + user(edited) + [NL] + out + user(BLOCK, [NL], Q2) + [NL]
    ids, spans, aligned = app._end_replay(res, copy)
    assert ids == copy and aligned == 0 and al.last_moved == []
    assert al.last_refusal and al.last_refusal["why"] == "in_prompt", al.last_refusal
    assert al.last_refusal["c"] < e
    # the block removed, not moved: the last message does not hold it
    copy = HEAD + user(Q1) + [NL] + out + user(Q2) + [NL]
    ids, _, aligned = app._end_replay(res, copy)
    assert ids == copy and aligned == 0 and al.last_moved == []
    # off
    al.moved = False
    copy = HEAD + user(Q1) + [NL] + out + user(BLOCK, [NL], Q2) + [NL]
    ids, _, aligned = app._end_replay(res, copy)
    assert ids == copy and aligned == 0
    al.moved = True
    ids, _, aligned = app._end_replay(res, copy)
    assert ids[:e] == res.tokens[:e] and aligned > 0


def test_the_block_moved_twice_over_three_turns():
    res, al = _setup_moved()
    prompt1 = HEAD + user(BLOCK, [NL], Q1) + [NL]
    out1, _ = _first(prompt1)
    copy2 = HEAD + user(Q1) + [NL] + out1 + user(BLOCK, [NL], Q2) + [NL]
    ids2, _, aligned2 = app._end_replay(res, copy2)
    assert aligned2 > 0
    out2, pre2, _ = turn(ids2, 12)
    assert pre2["kind"] == "end"
    e2 = res.end["at"]
    # turn 3: both earlier messages bare, the block in the third
    copy3 = (HEAD + user(Q1) + [NL] + out1 + user(Q2) + [NL] + out2
             + user(BLOCK, [NL], Q3) + [NL])
    ids3, _, aligned3 = app._end_replay(res, copy3)
    assert ids3[:e2] == res.tokens[:e2] and aligned3 > 0, (len(ids3), e2)
    assert len(al.last_moved) == 2, al.last_moved
    got, pre3, _ = turn(ids3, 8)
    assert pre3["kind"] == "end" and pre3["reused"] == e2, pre3
    assert got == _cold(ids3, 8)
    return f"two blocks served from the rows, resumed at {e2}"


def test_with_a_thinking_answer_both_passes_work_together():
    """SRV-239's shape on top: the answer thought, the copy leaves the reasoning out (the
    template's empty block), and the block moved as well."""
    prompt = HEAD + user(BLOCK, [NL], Q1) + [NL]
    _setup_moved()
    out, _ = _first(prompt)
    k = next(i for i in range(8, len(out) - 4) if out[i] not in out[:i] and out[i] not in prompt
             and out[i] not in (NL, SP, IM_S, USER, IM_E))
    res, al = _setup_moved(tb=tb_of({out[k]: endalign.THINK_END}))
    out_b, _ = _first(prompt)
    assert out_b == out
    e = res.end["at"]
    copy = HEAD + user(Q1) + [NL] + [SP] + out[k:] + user(BLOCK, [NL], Q2) + [NL]
    ids, _, aligned = app._end_replay(res, copy)
    assert ids[:e] == res.tokens[:e] and aligned > 0, (len(ids), e)
    assert al.last_moved and al.last_omitted, (al.last_moved, al.last_omitted)
    got, pre, _ = turn(ids, 8)
    assert pre["kind"] == "end" and pre["reused"] == e, pre
    assert got == _cold(ids, 8)


def test_the_slot_choice_sees_the_moved_block_as_the_same_conversation():
    """Live run 2 (0.4.8-test b2): A's next message with the block moved counted as a new
    conversation on the client's tokens and took the other slot, where nothing of A was."""
    from test_par_continue import CHUNK, FakeResident, _need, _restore, _setup
    res, al = _setup_moved()
    prompt = HEAD + user(BLOCK, [NL], Q1) + [NL]
    out, _ = _first(prompt)
    copy = HEAD + user(Q1) + [NL] + out + user(BLOCK, [NL], Q2) + [NL]
    assert endalign._common(copy, prompt) < len(prompt) - 16
    assert al.moved_common(prompt, copy) == len(prompt), al.moved_common(prompt, copy)
    edited = list(Q1)
    edited[3] = 2 if edited[3] != 2 else 3
    bad = HEAD + user(edited) + [NL] + out + user(BLOCK, [NL], Q2) + [NL]
    assert al.moved_common(prompt, bad) == endalign._common(bad, prompt)
    # through the slot choice: a long view of the same shape with a block longer than the
    # 1,024-token chunk, on token lists alone (no engine)
    big, block = [1] * 3000, _ids(1500, 205)
    view = big + HEAD + user(block, [NL], Q1) + [NL]
    copy2 = big + HEAD + user(Q1) + [NL] + out + user(block, [NL], Q2) + [NL]
    assert endalign._common(copy2, view) < len(view) - CHUNK
    assert al.moved_common(view, copy2) == len(view)
    pool, saved = _setup(FakeResident(view), view=view)
    was = app.STATE.get("end_align")
    app.STATE["end_align"] = al
    try:
        need, why = _need(copy2)
        assert (need is None or why.get("slot0_first")) and why["wait"] == "view", why
        assert why.get("view_moved", 0) > 0, why
    finally:
        app.STATE["end_align"] = was
        _restore(saved)


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"  {name:72s} ok   {fn() or ''}", flush=True)
            passed += 1
    print(f"{passed} passed")
