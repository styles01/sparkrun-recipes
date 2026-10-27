"""Parallel requests: a turn that continues slot 0's conversation waits for slot 0 (SRV-197, QA
SRV-198). CPU only: `_par_side_need` and `_par_view_keep` with a stand-in for the resident rows and
the prompt renderer.

The shape is the live box's (`--parallel 2`, a 65,536-token side
slot): an opencode session under the conductor. The rows slot 0 holds are the served prompt -- the
client's messages with the router's note in turn 1 replayed where it was put and this turn's note
at the tail -- plus the generation (the end anchor). The next turn's rendered ids (no notes) part
from those rows at the first note, so the rows' own test (`_continues_resident`) never saw a
continuation; a turn that met slot 0 held for a moment (the shadow reader) took the side slot and
prefilled its whole conversation again: cached share 0.40-0.68 instead of ~0.97.

Run: python tests/test_par_continue.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server import app as A  # noqa: E402
from server import slots as S  # noqa: E402

CHUNK = 1024


class FakeResident:
    """The resident rows as `_par_side_need` and `_par_view_keep` read them."""

    def __init__(self, tokens, chunk=CHUNK):
        self.tokens = list(tokens)
        self.valid = len(self.tokens)
        self.chunk = chunk

    def common(self, ids):
        return A._shared_head(ids, self.tokens[:self.valid])


def _session(seed: int, head: list[int], n_user: int):
    """A conversation's first rendered prompt: the shared head (system prompt and tools), then
    this session's first user message."""
    return head + [seed * 100_000 + i for i in range(n_user)]


def _setup(resident, view=None, side_len=65_536):
    pool = S.Pool([S.Slot(0, 262_144), S.Slot(1, side_len)])
    saved = {k: A.STATE.get(k) for k in ("pool", "resident", "slot0_view", "prefix_chunk")}
    A.STATE["pool"], A.STATE["resident"], A.STATE["prefix_chunk"] = pool, resident, CHUNK
    if view is None:
        A.STATE.pop("slot0_view", None)
    else:
        A.STATE["slot0_view"] = list(view)
    return pool, saved


def _restore(saved):
    for k, v in saved.items():
        if v is None:
            A.STATE.pop(k, None)
        else:
            A.STATE[k] = v


def _need(ids, max_new=32_000, plan_active=True):
    """`_par_side_need` with the renderer answering `ids`."""
    real = A.build_prompt

    class Plan:
        active = plan_active

    def fake(body, client=None, *, count=False, left_out=0):
        assert count == "ids"
        return list(ids)
    A.build_prompt = fake
    try:
        why: dict = {}
        need = A._par_side_need({"messages": []}, None, 0, max_new, Plan(), [], None, why)
        return need, why
    finally:
        A.build_prompt = real


HEAD = list(range(7_000))                         # opencode's system prompt and tools
NOTE = [999_001, 999_002, 999_003, 999_004]       # a router note (hidden from the client)
GEN = [888_000 + i for i in range(1_500)]         # a turn's thinking and tool call


def _turns():
    """Turn 1 and turn 2 of one session as the client renders them, and the rows slot 0 holds
    after turn 1 (note replayed after the first user message, the generation kept at the end)."""
    r1 = _session(1, HEAD, 1_500) + [7, 8, 9]                   # ...<|im_start|>assistant<think>
    rows = r1[:8_500] + NOTE + r1[8_500:] + NOTE + GEN          # served + end anchor's rows
    tool_result = [777_000 + i for i in range(2_000)]
    r2 = r1[:-1] + [6] + GEN[:300] + tool_result + [7, 8, 9]    # the client re-renders the answer
    return r1, rows, r2


def test_continues_view_rule():
    c = A._continues_view
    assert c(20_000, 19_998, CHUNK), "the next turn shares the last prompt but its tail"
    assert c(20_000, 18_976, CHUNK) and not c(20_000, 18_975, CHUNK), "within one chunk"
    assert not c(20_000, 7_000, CHUNK), "another session: only the system prompt is shared"
    assert not c(900, 900, CHUNK), "a view shorter than a chunk: cheap to prefill elsewhere"
    assert not c(0, 0, CHUNK)


def test_the_rows_test_misses_a_conductor_turn_and_the_view_test_sees_it():
    """The bug, on its own: after turn 1 the rows hold a note at 8,500; turn 2 renders without it.
    The rows' test (0.3.44/0.3.45) says "not a continuation" -- also before PL1's dfdeaaa, whose
    added `common >= chunk` only narrows it -- and the client-view test says it is."""
    r1, rows, r2 = _turns()
    res = FakeResident(rows)
    rc = res.common(r2)
    assert rc == 8_500
    assert not A._continues_resident(res.valid, rc, CHUNK)                   # after dfdeaaa
    assert not (rc >= res.valid - CHUNK)                                     # before dfdeaaa
    assert A._continues_view(len(r1), A._shared_head(r2, r1), CHUNK)


def test_a_continuing_turn_waits_for_slot_0():
    r1, rows, r2 = _turns()
    pool, saved = _setup(FakeResident(rows), view=r1)
    try:
        need, why = _need(r2)
        assert (need is None or why.get("slot0_first")) and why["wait"] == "view", why
        assert why["view"] == len(r1) and why["view_common"] == len(r1) - 1
        assert pool.stats["resident_waits"] == 1
    finally:
        _restore(saved)


def test_the_old_rule_alone_sends_it_to_the_side_slot():
    """`QWEN38_SIDE_CONTINUE=resident` (the 0.3.45 rule, kept for the A/B): the same turn fits the
    side slot (11,8xx tokens + 32,000 + margins < 65,536) and is offered it."""
    r1, rows, r2 = _turns()
    pool, saved = _setup(FakeResident(rows), view=r1)
    was = A.SIDE_CONTINUE
    A.SIDE_CONTINUE = "resident"
    try:
        need, why = _need(r2)
        assert need is not None and "wait" not in why, why
        assert need == len(r2) + 32_000 + A.SIDE_MARGIN + A.PLAN_MARGIN
    finally:
        A.SIDE_CONTINUE = was
        _restore(saved)


def test_a_second_session_still_gets_the_side_slot():
    r1, rows, _ = _turns()
    other = _session(2, HEAD, 900) + [7, 8, 9]
    pool, saved = _setup(FakeResident(rows), view=r1)
    try:
        need, why = _need(other)
        assert need is not None and "wait" not in why, why
        assert why["view_common"] == len(HEAD)
        # and its later turns too (its first turn ran on the side slot: the view is still r1)
        later = other + [5_000_000 + i for i in range(6_000)]
        need, why = _need(later)
        assert need is not None and "wait" not in why, why
        assert pool.stats.get("resident_waits", 0) == 0
    finally:
        _restore(saved)


def test_too_long_for_the_side_slot_still_waits():
    r1, rows, _ = _turns()
    pool, saved = _setup(FakeResident(rows), view=r1)
    try:
        long_other = _session(3, HEAD, 40_000)
        need, why = _need(long_other)
        assert need is None and why["wait"] == "too_long"
        assert pool.stats["too_long"] == 1
    finally:
        _restore(saved)


def test_without_a_view_the_rows_test_still_applies():
    """A raw request (no notes): the rows are the rendered prompt, so the old test sees it."""
    p1 = _session(4, HEAD, 3_000)
    rows = p1 + GEN[:200]
    p2 = rows + [1, 2, 3]
    pool, saved = _setup(FakeResident(rows), view=None)
    try:
        need, why = _need(p2, plan_active=False)
        assert (need is None or why.get("slot0_first")) and why["wait"] == "resident", why
    finally:
        _restore(saved)


def test_the_view_is_kept_only_when_the_rows_hold_the_request():
    r1, rows, r2 = _turns()
    served1 = rows[:len(rows) - len(GEN)]
    _, saved = _setup(FakeResident(rows))
    try:
        assert A._par_view_keep(served1, r1) and A.STATE["slot0_view"] == r1
        # a guest (opencode's 599-token title request): served over the tail, the rows put back
        title = [42] * 599
        assert not A._par_view_keep(title, title) and A.STATE["slot0_view"] == r1
        # another session's long turn replaced the rows: its view is kept, the old one goes
        other = _session(5, HEAD, 3_000)
        A.STATE["resident"] = FakeResident(other + GEN)
        assert A._par_view_keep(other, other) and A.STATE["slot0_view"] == other
        # rows shorter than a chunk keep nothing
        A.STATE["resident"] = FakeResident([1] * 500)
        assert not A._par_view_keep([1] * 500, [1] * 500)
        assert not A._par_view_keep(None, r1) and not A._par_view_keep(served1, None)
    finally:
        _restore(saved)


def test_shared_head():
    h = A._shared_head
    assert h([], [1]) == 0 and h([1, 2, 3], [1, 2, 3]) == 3 and h([1, 2, 3], [1, 2]) == 2
    assert h([1, 2, 3, 4], [1, 2, 9, 4]) == 2 and h([5], [6]) == 0
    a = list(range(100_000))
    b = a[:73_123] + [-1] + a[73_124:]
    assert h(a, b) == 73_123


def test_own_work_on_slot_0_does_not_send_a_request_to_the_side_slot():
    """C-dupfinder (2026-10-06, the first fix alone): a new session's turns 2 and 3 met slot 0
    held by the shadow reader and took the side slot (the view was still the previous session's).
    While only TandemLLM's own work holds slot 0 a request waits for it, up to SIDE_AFTER_OWN_S."""
    ok = A._side_ok
    assert ok(0.0, None), "slot 0 busy with a client's request: the side slot is for that"
    for own in (A.RANK_SHADOW, A.RANK_PREP, -1):
        assert not ok(0.0, own) and not ok(A.SIDE_AFTER_OWN_S - 0.1, own), own
        assert ok(A.SIDE_AFTER_OWN_S, own), "own work that does not give way: the side slot"


def test_holder_name():
    saved = A.HOLDER["who"]
    try:
        A.HOLDER["who"] = {"rank": A.RANK_SHADOW}
        assert A._holder_name() == "the shadow reader"
        A.HOLDER["who"] = {"rank": A.RANK_PREP}
        assert A._holder_name() == "prepare/memory job"
        A.HOLDER["who"] = {"rank": 0, "own": True}
        assert A._holder_name() == "the router's question"
        A.HOLDER["who"] = {"rank": 1}
        assert A._holder_name() == "a request"
    finally:
        A.HOLDER["who"] = saved


# ------------------------------------------------------------------ a shared head is not a conversation
IM_S, USER, IM_E, NL, ASSIST = 151_644, 872, 151_645, 198, 77_091


def _owui(head_n: int, user_n: int, seed: int) -> list[int]:
    """An Open WebUI chat's first prompt in the template's marks: the system turn (its prompt and
    tool block, `head_n` tokens in all), the first user message, the generation prompt."""
    head = [IM_S, 9_001, NL] + list(range(10, 10 + head_n - 5)) + [IM_E, NL]
    user = [IM_S, USER, NL] + [seed * 100_000 + i for i in range(user_n)] + [IM_E, NL]
    return head + user + [IM_S, ASSIST, NL]


def _with_marks(fn):
    saved = A._CONV_MARKS
    A._CONV_MARKS = (IM_S, [USER], IM_E)
    try:
        fn()
    finally:
        A._CONV_MARKS = saved


def test_conv_start_is_past_the_first_user_message():
    def t():
        p = _owui(6_675, 20, 1)
        assert A._conv_start(p) == 6_675 + 3 + 20 + 1
        assert A._conv_start([IM_S, 9_001, NL, 5, IM_E]) is None, "no user message"
        assert A._conv_start([IM_S, USER, NL, 5]) is None, "a user message that never ends"
        assert A._conv_start([IM_S, USER, NL, IM_E]) == 4, "a chat with no system turn"
    _with_marks(t)
    saved = A._CONV_MARKS
    A._CONV_MARKS = False
    try:
        assert A._conv_start(_owui(100, 5, 1)) is None, "no tokenizer: the old rule alone"
    finally:
        A._CONV_MARKS = saved


def test_w18_a_new_chat_that_shares_only_the_head_takes_the_side_slot():
    """W18 (the UX2 run): slot 0 held tab A's essay chat, 7,096 rows of which tab B's
    new chat shared 6,675 (Open WebUI's system and tool head); the resident test called it a
    continuation and tab B waited 35.8 s while slot 1 was idle. With the conversation's start
    known, sharing the head is not continuing it."""
    def t():
        a = _owui(6_675, 40, 1)
        rows = a + [999_001] * 100 + [888_000 + i for i in range(7_096 - len(a) - 100)]
        b = _owui(6_675, 15, 2)
        res = FakeResident(rows)
        assert res.valid == 7_096 and res.common(b) == 6_675 + 3
        # the old rule (`QWEN38_SIDE_CONV=off`): a continuation of tab A, it waits for slot 0
        was = A.SIDE_CONV
        A.SIDE_CONV = "off"
        pool, saved = _setup(res, view=None)
        try:
            need, why = _need(b, max_new=32_768, plan_active=False)
            assert (need is None or why.get("slot0_first")) and why["wait"] == "resident", why
        finally:
            A.SIDE_CONV = was
            _restore(saved)
        # the fix: a new conversation, it takes the side slot
        pool, saved = _setup(res, view=None)
        try:
            need, why = _need(b, max_new=32_768, plan_active=False)
            assert need is not None and "wait" not in why, why
            assert why["conv"] == A._conv_start(b) > why["res_common"]
            assert pool.stats.get("resident_waits", 0) == 0
        finally:
            _restore(saved)
    _with_marks(t)


def test_a_short_view_of_another_chat_is_not_continued_either():
    """The view test has the same hole: slot 0's last prompt was a short chat ('hi', head plus a
    few tokens); a new chat shares the head within a chunk of that view."""
    def t():
        hi = _owui(6_675, 3, 1)
        other = _owui(6_675, 30, 2)
        assert A._continues_view(len(hi), A._shared_head(other, hi), CHUNK), "the hole"
        assert not A._continues_view(len(hi), A._shared_head(other, hi), CHUNK,
                                     A._conv_start(other))
        pool, saved = _setup(FakeResident(hi + [888_000] * 50), view=hi)
        try:
            need, why = _need(other, max_new=4_096, plan_active=False)
            assert need is not None and "wait" not in why, why
        finally:
            _restore(saved)
    _with_marks(t)


def test_the_next_turn_of_the_same_chat_still_waits():
    """SRV-197 stays: the next turn shares the first user message and the answer, past `conv`."""
    def t():
        t1 = _owui(6_675, 40, 1)
        rows = t1 + [888_000 + i for i in range(1_500)]
        t2 = (t1 + [888_000 + i for i in range(1_500)] + [IM_E, NL, IM_S, USER, NL, 5, 6, IM_E, NL,
                                                          IM_S, ASSIST, NL])
        pool, saved = _setup(FakeResident(rows), view=t1)
        try:
            need, why = _need(t2, max_new=32_768, plan_active=True)
            assert (need is None or why.get("slot0_first")) and why["wait"] in ("view", "resident"), why
            assert why["conv"] < why["view_common"]
        finally:
            _restore(saved)
        # an agent turn under a plan (notes in the rows): the view decides, as FX4 made it
        r1, rows2, r2 = _turns()
        pool, saved = _setup(FakeResident(rows2), view=r1)
        try:
            need, why = _need(r2)
            assert (need is None or why.get("slot0_first")) and why["wait"] == "view", why
        finally:
            _restore(saved)
    _with_marks(t)


if __name__ == "__main__":
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            n += 1
            print("ok", name)
    print(f"{n} passed")
