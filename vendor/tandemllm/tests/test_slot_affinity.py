"""SRV-528 (QA SRV-529): a conversation stays on the slot that holds its rows. CPU only.

THE INCIDENT (2026-10-17 00:05, Super VL EXL3 4.0, `--parallel 4`: slot 0 window 1,048,576, three
side slots of 262,144; engine-20261016-223733.log). A chat ran seven requests on side slot 2; its
next one (45,683 tokens) went to slot 1, which held nothing of it, and read 45,778 tokens in 80.9 s.
Three things combined, each read in the code and replayed here:

  * `_side_home` named a side slot home only when it kept an end anchor, and every turn of that run
    kept none (`ahead`, SRV-530): each turn was routed as a new conversation;
  * a new conversation takes the lowest free side slot `take_side` allows; slot 1's carry record
    (the chat's previous run ended there at 23:55:15) kept it off for CARRY_KEEP_S (600 s), so the
    run took slot 2, until 00:05:15: the hop came at 00:05:19;
  * `prefer_side` sent every new conversation to a side slot because slot 0 held the partial state
    of a 1,000,000-token test prompt cancelled after 937,984 rows, nobody's conversation.

THE FIX (server/app.py): `_side_home` takes the slot whose rows hold the longest share of the
prompt, end anchor or not (the client's view, or the rows themselves past the first user message);
a slot 0 conversation whose last request stopped inside its prompt, or that no request continued
for SLOT0_KEEP_S, no longer counts for `prefer_side` (`_slot0_stale`); a turn whose slot stays busy
past SIDE_HOME_WAIT_S says it moved and is counted (`hops`, `hop_tokens`).

The replay drives the incident's eight requests (the ledger's prompt lengths and start times)
through `_par_side_need` and the handler's order of `take_slot` / `take_side` / slot 0 on a fake
clock, and charges each turn the rows its slot does not hold (to the end anchor when kept, else to
the last 4,096-token anchor line). On the 0.11.2 tree the request of 00:05:19 lands on slot 1.

Run: python tests/test_slot_affinity.py
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server import app as A  # noqa: E402
from server import slots as S  # noqa: E402
from test_par_continue import FakeResident, _need  # noqa: E402

CH = 4096                                   # NemotronH's anchor grid
HEAD = list(range(9_108))                   # the Chat tab's system prompt and tools
CONV = len(HEAD) + 64                       # its first user message ends here (`_conv_start`)
SIDE = 262_144
GEN = 150                                   # a turn's tool call
#: the incident's requests: (seconds after 23:55:15, the end of the chat's run 1 on slot 1; prompt)
TURNS = [(231, 39_680), (345, 41_932), (431, 42_885), (531, 43_185), (581, 43_376), (596, 43_996),
         (604, 45_683), (699, 46_444)]


class Res(FakeResident):
    """A slot's resident rows, with an end anchor at their end or none."""

    def __init__(self, tokens, end=False, chunk=CH):
        super().__init__(tokens, chunk)
        self.end = {"at": self.valid} if end and self.valid else None


def conv(seed: int, n: int) -> list[int]:
    """A conversation of `n` rendered tokens: the client's head, then its own."""
    return HEAD + [seed * 1_000_000 + i for i in range(n - len(HEAD))]


class Clock:
    def __init__(self, t=10_000.0):
        self.t = t

    def __call__(self):
        return self.t


@contextlib.contextmanager
def world(slot0_rows, slot0_view=None, slot0_seen=None):
    """A pool of the incident's shape and the server state around it, on a fake clock."""
    pool = S.Pool([S.Slot(0, 1_048_576)] + [S.Slot(i, SIDE) for i in (1, 2, 3)])
    keys = ("pool", "resident", "slot0_view", "prefix_chunk", "slot0_seen")
    saved = {k: A.STATE.get(k) for k in keys}
    real_conv, real_mono = A._conv_start, time.monotonic
    clock = Clock()
    A.STATE.update(pool=pool, resident=Res(slot0_rows), prefix_chunk=CH)
    A.STATE.pop("slot0_seen", None)
    if slot0_view is not None:
        A.STATE["slot0_view"] = list(slot0_view)
    else:
        A.STATE.pop("slot0_view", None)
    if slot0_seen is not None:
        A.STATE["slot0_seen"] = slot0_seen
    for s in pool.side():
        s.values = {"resident": Res([]), "slot_view": None}
    A._conv_start = lambda ids: CONV if len(ids) > CONV else None    # the Chat tab's template
    time.monotonic = clock
    try:
        yield pool, clock
    finally:
        time.monotonic = real_mono
        A._conv_start = real_conv
        for k, v in saved.items():
            if v is None:
                A.STATE.pop(k, None)
            else:
                A.STATE[k] = v


def hold(slot, rows, view=None, end=False, carry_t=None):
    """`slot` holds `rows` (its last prompt `view`), with or without an end anchor and its carry
    record (`_side_note_end` writes one only for a kept end)."""
    slot.values = {"resident": Res(rows, end=end), "slot_view": list(view) if view else None}
    slot.carry = ({"gen": slot.gen, "t": carry_t, "at": len(rows), "resident": True}
                  if carry_t is not None else None)


def stale_slot0():
    """Slot 0 after the cancelled 1,000,000-token prompt, as `_par_view_keep` left it."""
    big = [5_000_000 + i for i in range(1_000_000)]
    return big[:937_984], big


# ------------------------------------------------------------------ the scenarios of SRV-528
def test_the_next_turn_goes_to_the_slot_that_holds_the_conversation():
    """The incident: rows on slot 2 without an end anchor, slot 1 holding the chat's earlier run
    with an end anchor and a carry record 604 s old, slot 0 the cancelled prompt's partial state."""
    rows0, big = stale_slot0()
    with world(rows0, big) as (pool, clock):
        A._par_view_keep(big, big)                     # what the abandoned request left
        run1 = conv(1, 61_692)
        hold(pool.slots[1], run1 + [7] * GEN, run1, end=True, carry_t=clock.t - 604)
        mine = conv(2, 46_444)
        hold(pool.slots[2], mine[:43_996 + GEN], mine[:43_996], end=False)
        need, why = _need(mine[:45_683])
        assert need is not None and why.get("home") == 2, why
        assert "prefer_side" not in why, why
    return "home slot 2 (no end anchor), not slot 1"


def test_no_end_anchor_same_slot():
    """The engine was ahead (no end anchor) and no view was kept: the rows alone name the slot."""
    with world([]) as (pool, clock):
        mine = conv(2, 46_444)
        hold(pool.slots[2], mine[:43_996 + GEN], None, end=False)
        need, why = _need(mine[:45_683])
        assert why.get("home") == 2, why
        # the client's head alone (another chat of the same client) is no home
        need, why = _need(conv(3, 20_000))
        assert "home" not in why, why
    return "rows past the first user message: home"


def test_the_carry_guard_does_not_move_a_conversation():
    """Across the 600 s boundary of slot 1's carry record the conversation stays on slot 2; the
    guard only orders new conversations (`take_side`)."""
    with world([]) as (pool, clock):
        other = conv(1, 30_000)
        mine = conv(2, 46_444)
        t0 = clock.t
        hold(pool.slots[1], other, other, end=True, carry_t=t0)
        hold(pool.slots[2], mine[:43_996 + GEN], mine[:43_996], end=True, carry_t=t0)
        for dt in (A.CARRY_KEEP_S - 1, A.CARRY_KEEP_S + 1):
            clock.t = t0 + dt
            need, why = _need(mine[:45_683])
            assert why.get("home") == 2, (dt, why)
            s = pool.take_slot(why["home"], need, owner="c")
            assert s is pool.slots[2]
            pool.release_side(s)
        # a new conversation: slot 1's fresh record keeps it off, then not
        hold(pool.slots[2], [], None)
        clock.t = t0 + A.CARRY_KEEP_S - 1
        s = pool.take_side(10_000, owner="n", keep_carry_s=A.CARRY_KEEP_S)
        assert s.index == 2, s.index
        pool.release_side(s)
        clock.t = t0 + A.CARRY_KEEP_S + 1
        s = pool.take_side(10_000, owner="n", keep_carry_s=A.CARRY_KEEP_S)
        assert s.index == 1, s.index
        pool.release_side(s)
    return "599 s and 601 s: slot 2 both times"


def test_a_side_slot_that_holds_more_than_slot_0_beats_slot_0s_view():
    """Live, 2026-10-17 01:45: a chat whose first prompt was the one slot 0 had just served (the same
    system prompt and first message) counted as continuing slot 0's view at every turn, waited
    SIDE_SLOT0_WAIT_S (3.0 s) for the busy slot 0 and took the lowest free side slot, its own by
    luck. The side slot that holds more of the prompt than slot 0 is its home."""
    with world([]) as (pool, clock):
        first = conv(2, 12_000)                         # the first prompt both chats share
        other = first + [6_000_000 + i for i in range(1_000)]
        A.STATE["resident"] = Res(other, end=True)
        assert A._par_view_keep(other, other)
        mine = first + [7_000_000 + i for i in range(1_500)]
        hold(pool.slots[2], mine[:13_000], mine[:12_900], end=True)
        need, why = _need(mine)
        assert why.get("wait") == "view" and why.get("home") == 2, why
        assert "slot0_first" not in why and why["home_share"] > why["view_common"], why
        # slot 0 holds more of it: slot 0 first, as before
        hold(pool.slots[2], first, first, end=True)
        A.STATE["resident"] = Res(mine[:13_400], end=True)
        assert A._par_view_keep(mine[:13_400], mine[:13_400])
        need, why = _need(mine)
        assert why.get("slot0_first") and "home" not in why, why
    return "home slot 2 (13,000 held) over slot 0's view (12,000 shared)"


def test_the_longest_share_wins_and_a_free_slot_within_a_chunk_goes_first():
    with world([]) as (pool, clock):
        mine = conv(2, 46_444)
        hold(pool.slots[1], mine[:30_000], mine[:30_000])      # an older turn of it, after a hop
        hold(pool.slots[3], mine[:43_996 + GEN], mine[:43_996])
        need, why = _need(mine[:45_683])
        assert why.get("home") == 3, why
        # busy and within a chunk of another slot that holds it too: the free one
        hold(pool.slots[1], mine[:43_996], mine[:43_996 - 200])
        pool.slots[3].lock.acquire()
        try:
            need, why = _need(mine[:45_683])
            assert why.get("home") == 1, why
            hold(pool.slots[1], mine[:30_000], mine[:30_000])
            need, why = _need(mine[:45_683])
            assert why.get("home") == 3, "the busy home: the handler waits SIDE_HOME_WAIT_S for it"
        finally:
            pool.slots[3].lock.release()
    return "longest share; a free slot within a chunk before a busy one"


def test_an_abandoned_million_token_prompt_does_not_push_new_chats_to_the_side_slots():
    rows0, big = stale_slot0()
    with world(rows0, big) as (pool, clock):
        assert A._par_view_keep(big, big)
        assert A.STATE["slot0_seen"][1] is False and A._slot0_stale() == "partial"
        need, why = _need(conv(4, 30_000))
        assert "prefer_side" not in why and why.get("slot0_stale") == "partial", why
        # its own retry still continues it on slot 0
        need, why = _need(big)
        assert why.get("wait") == "view", why
    return "cancelled prefill: not counted for prefer_side; its retry still waits for slot 0"


def test_a_real_long_conversation_on_slot_0_is_still_protected():
    with world([]) as (pool, clock):
        long0 = conv(5, 300_000)
        A.STATE["resident"] = Res(long0 + [9] * GEN, end=True)
        assert A._par_view_keep(long0, long0)
        clock.t += 300                                  # continued 5 minutes ago
        need, why = _need(conv(6, 12_000))
        assert why.get("prefer_side") == 300_000 + GEN and "slot0_stale" not in why, why
        clock.t += A.SLOT0_KEEP_S                       # nobody continued it for SLOT0_KEEP_S
        need, why = _need(conv(6, 12_000))
        assert "prefer_side" not in why and why.get("slot0_stale") == "idle", why
    return "protected 5 min after its last turn, not after SLOT0_KEEP_S"


def test_a_new_conversation_takes_the_lowest_free_slot():
    with world([]) as (pool, clock):
        hold(pool.slots[2], conv(7, 20_000), conv(7, 20_000))
        need, why = _need(conv(8, 20_000))
        assert "home" not in why, why
        s = pool.take_side(need, owner="n")
        assert s.index == 1
        pool.release_side(s)
    return "no home; slot 1"


def test_the_queued_line_names_a_hop():
    pool = S.Pool([S.Slot(0, 1_048_576), S.Slot(1, SIDE), S.Slot(2, SIDE)])
    why = {"home": 2, "home_wait": 10.0, "hop_from": 2}
    line = A._queue_why(pool, why, 50_000, 10.2, {})
    assert line.startswith("moved from slot 2 (busy 10.0 s): re-reading"), line
    return line


# ------------------------------------------------------------------ the replay
def route(pool, ids):
    """The handler's order (server/app.py, `Handler._complete`): the home slot, a side slot for a
    guest or a new conversation beside slot 0's (with the carry guard), slot 0 when free unless a
    short new chat waits for a side slot, then any free side slot."""
    need, why = _need(ids)
    s = None
    if why.get("home") is not None:
        s = pool.take_slot(why["home"], need, owner="c")
    if s is None and (why.get("guest") or why.get("prefer_side")):
        s = pool.take_side(need, owner="c", keep_carry_s=A.CARRY_KEEP_S)
    if s is None and not pool.slots[0].busy and not why.get("prefer_side_wait"):
        return pool.slots[0], why
    if s is None:
        s = pool.take_side(need, owner="c")
    return s, why


def serve(pool, s, ids, end: bool) -> int:
    """The turn on `s`: the rows it reads again (to the end anchor when the prompt holds all of it,
    else to the last anchor line), then the slot holds its rows and its view."""
    rows = ids + [8_000_000 + len(ids) + i for i in range(GEN)]
    res = A.STATE["resident"] if s.index == 0 else s.values["resident"]
    c = res.common(ids)
    e = res.end["at"] if res.end is not None and c >= res.end["at"] else 0
    reused = max(e, (c // CH) * CH)
    if s.index == 0:
        A.STATE["resident"] = Res(rows, end=end)
        A._par_view_keep(ids, ids)
    else:
        hold(s, rows, ids, end=end, carry_t=time.monotonic() if end else None)
        pool.release_side(s)
    return len(ids) - reused, rows


def replay(slot0_busy: bool, end: bool):
    """The eight requests; `end`: whether the turns keep their end anchor (SRV-530 fixed) or not
    (as on 2026-10-16). Returns [(slot, forwarded)]."""
    rows0, big = stale_slot0()
    with world(rows0, big) as (pool, clock):
        A._par_view_keep(big, big)
        t0 = clock.t
        run1 = conv(1, 61_692)
        hold(pool.slots[1], run1 + [7] * GEN, run1, end=True, carry_t=t0)
        if slot0_busy:
            # slot 0 serves a real long conversation, continued a minute ago
            long0 = conv(9, 300_000)
            A.STATE["resident"] = Res(long0, end=True)
            A._par_view_keep(long0, long0)
            pool.slots[0].lock.acquire()
        mine = conv(2, TURNS[-1][1] + GEN)
        out = []
        try:
            for dt, n in TURNS:
                clock.t = t0 + dt
                ids = mine[:n]
                s, why = route(pool, ids)
                fw, rows = serve(pool, s, ids, end)
                mine[n:n + GEN] = rows[n:]         # the next prompt carries this turn's answer
                out.append((s.index, fw))
        finally:
            if slot0_busy:
                pool.slots[0].lock.release()
    return out


def _table(out):
    return ", ".join(f"{n:,}->slot {s} fw {fw:,}" for (s, fw), (_, n) in zip(out, TURNS))


def test_replay_the_incident_on_side_slots():
    """Slot 0 busy with a real conversation: the chat runs on side slots. It stays on the slot it
    started on through 00:05:19, with end anchors and without."""
    lines = []
    for end in (True, False):
        out = replay(slot0_busy=True, end=end)
        first = out[0][0]
        assert all(s == first for s, _ in out), _table(out)
        for k in range(1, len(out)):
            grown = TURNS[k][1] - TURNS[k - 1][1]
            limit = grown if end else grown + CH
            assert out[k][1] <= limit, (k, _table(out))
            if end:
                assert out[k][1] <= 4_096, (k, _table(out))
        lines.append(f"{'end anchors' if end else 'no end anchor'}: slot {first} for all 8, at most "
                     f"{max(fw for _, fw in out[1:]):,} rows forwarded after the first")
    return "; ".join(lines)


def test_replay_the_incident_with_slot_0_free():
    """The evening as it was: slot 0 free with the cancelled prompt's state. The chat now runs on slot
    0 (the 1,048,576-token window) and stays there."""
    out = replay(slot0_busy=False, end=True)
    assert all(s == 0 for s, _ in out), _table(out)
    assert max(fw for _, fw in out[1:]) <= 4_096, _table(out)
    return f"slot 0 for all 8, at most {max(fw for _, fw in out[1:]):,} rows forwarded after the first"


# ------------------------------------------------------------------ through the handler
def test_the_handler_counts_a_hop_when_the_home_slot_stays_busy():
    """The home slot busy past SIDE_HOME_WAIT_S: the turn takes another slot, the log says it moved,
    and /health counts the hop and the tokens it read again."""
    from test_slot0_wait import _body, _fake_parallel
    from test_ui_states import Req
    pool = _fake_parallel()
    body = _body()
    ids = A.build_prompt(body, None, count="ids")
    pool.slots[1].values["resident"] = FakeResident(ids)
    pool.slots[1].values["slot_view"] = list(ids)
    was = A.SIDE_HOME_WAIT_S
    A.SIDE_HOME_WAIT_S = 0.4
    pool.slots[1].lock.acquire()                       # another request has it
    out: dict = {}
    log = io.StringIO()

    def post():
        r = Req(body, {"User-Agent": "curl/8.5"})
        t0 = time.monotonic()
        r.do_POST()
        out.update(raw=r.wfile.getvalue().partition(b"\r\n\r\n")[2].decode(), t=time.monotonic() - t0)
    try:
        with contextlib.redirect_stdout(log):
            th = threading.Thread(target=post)
            th.start()
            th.join(30)
    finally:
        A.SIDE_HOME_WAIT_S = was
        pool.slots[1].lock.release()
    assert '"choices"' in out["raw"], out["raw"][:300]
    assert out["t"] >= 0.4, out["t"]
    assert pool.stats.get("hops") == 1 and pool.stats.get("hop_tokens", 0) > 0, pool.stats
    assert pool.report()["hops"] == 1
    text = log.getvalue()
    assert "moved from slot 1 (busy" in text, text[-2000:]
    return f"hops 1, hop_tokens {pool.stats['hop_tokens']}"


if __name__ == "__main__":
    passed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"  {name:72s} ok   {fn() or ''}", flush=True)
            passed += 1
    print(f"{passed} passed")
