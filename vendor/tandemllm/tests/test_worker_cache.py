"""ENG-371 (QA ENG-372): an agent's worker keeps its own rows between its calls, and its parent's
rows are kept while it waits. CPU, the random 4-layer model and the slot choice on fakes.

THE INCIDENT (2026-10-17 23:38, Qwen3.8-27B EXL3 6.0, `--parallel 2`, `--resident-guest-max 8192`;
chat 01M55WQ92F8D8BKXMKCKWCNXM9, engine-20261017-220207.log). Slot 0 held another chat (24,200
rows, with images), the parent ran on side slot 1. The worker's calls 1 to 3 (2,532, 5,764 and 7,925
tokens) each started from 0:

  * call 1 ran on slot 0 as a GUEST of the other chat (shorter than 8,192 and than a quarter of
    24,200): its rows were stashed over and put back after it, `end anchor not kept (guest)`;
  * call 2 found slot 0 held by call 1's tail, took side slot 1 as a guest of slot 0 and wrote over
    the parent's 15,066 rows there (the stash could not hold its reach: `not_resident`), keeping
    nothing of its own either; the parent's return re-read 11,039 of 16,159 tokens;
  * call 3 (over a quarter of 24,200) was no guest and read all 7,925 again.

And the run before it (chat 01M55TKMFSS, 23:05): the worker took slot 0 from its parent, the park
held the parent, and the memory gate gave the park back at the ceiling unwritten
(`shed level=park why=ceiling freed=1898774528`): the parent's return re-read 11,895 of 17,015.

THE FIX: a request with `X-Tandem-Worker: 1` / `X-Tandem-Parent-Session` is never a guest
(`ResidentPrefix.begin(stay=True)`, `_par_side_need`); a side slot's rows of a waiting parent go to
the NVMe before another session's prefill cuts them (`ResidentPrefix.keeper`, `_keep_parent`); a
parked parent goes to the NVMe before the memory policy takes the park (`_shed_park`). Each test
runs the old path too and shows it re-reading.

Run: python tests/test_worker_cache.py
"""

from __future__ import annotations

import contextlib
import os
import shutil
import sys
import tempfile
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import test_forward_tree as T  # noqa: E402,F401  (the CPU environment before the engine imports)
import torch  # noqa: E402

from engine import cache, capture_lock, kvpool  # noqa: E402
from server import app  # noqa: E402
from server import slots as slots_mod  # noqa: E402
from test_cache import fresh, tokens  # noqa: E402
from test_end_anchor import BIG, CH, TOOL, _state  # noqa: E402
from test_session_spill import _spill  # noqa: E402

WHEAD = tokens(40, seed=60)                  # the worker's system block and tools, every worker's
PHEAD = tokens(40, seed=61)                  # the chat's


#: the largest difference a resume's logits may show against a cold prefill's: the rows below
#: the end anchor were written by the decode's forwards, not the prefill's grid (about 2e-6 on
#: this model); a resume on wrong rows is 1e-2 and more off
TOL = 1e-4


@contextlib.contextmanager
def prefill_logits():
    """The logits of every `cache.prefill` in the block, in order (the request's own prefill as
    `generate_stream` runs it)."""
    got, real = [], cache.prefill

    def spy(*a, **k):
        out = real(*a, **k)
        got.append(out[0].detach().float().clone())
        return out
    cache.prefill = spy
    try:
        yield got
    finally:
        cache.prefill = real


class Pre(dict):
    """`STATE["last_prefill"]` and the prefill's logits, printed without them."""

    def __repr__(self):
        return repr({k: v for k, v in self.items() if k != "logits"})


def wturn(ids, n, session, stay=False):
    """One request as the handler runs it (`turn` of test_end_anchor), `stay` for a worker; the
    prefill's logits in `pre["logits"]`."""
    with prefill_logits() as got:
        out = list(app.generate_stream(torch.tensor(ids), n, set(), None, session, stay=stay))
    pre = Pre(app.STATE["last_prefill"])
    pre["logits"] = got[-1]
    app._remember(list(ids), out, session)
    return out, pre


def cold(ids, n, **kw):
    """The same request with nothing cached: (its tokens, its prefill's logits)."""
    saved = dict(app.STATE)
    _state(**kw)
    try:
        out, pre = wturn(ids, n, "cold")
    finally:
        app.STATE.clear()
        app.STATE.update(saved)
    return out, pre["logits"]


def assert_cold(out, pre, ids, n):
    """The resume is the cold prefill: its logits within TOL, its greedy tokens the same."""
    ref, lg = cold(ids, n)
    d = float((pre["logits"] - lg).abs().max())
    assert d <= TOL, f"logits {d:.3g} off the cold prefill's"
    assert out == ref, "the greedy tokens differ from the cold prefill's"
    return d


def _tail(pre, ids):
    return len(ids) - int(pre["reused"])


# ------------------------------------------------------------------ 1: the incident on slot 0

def _calls_beside_a_long_chat(stay):
    """Another chat on slot 0 (408 rows), then a worker's call 1 (80 tokens: a guest by length),
    the router's question and the parent's context peek between, then call 2."""
    _state(guest_max=8192)
    other = tokens(400, seed=70)
    wturn(other, 8, "chat-other")
    w1 = WHEAD + tokens(40, seed=71)
    out1, pre1 = wturn(w1, 8, "chat-p-task-1", stay)
    q, _ = wturn(tokens(12, seed=72), 2, None)           # the router's question: a guest
    app._peek_find(PHEAD + tokens(300, seed=73))         # the parent's Context panel peek
    w2 = w1 + out1 + TOOL
    out2, pre2 = wturn(w2, 8, "chat-p-task-1", stay)
    return w1, out1, w2, out2, pre2


def test_a_workers_next_call_resumes_at_its_own_end_beside_a_long_chat():
    w1, out1, w2, out2, pre2 = _calls_beside_a_long_chat(stay=True)
    assert pre2["reused"] >= len(w1) + len(out1) - 1, pre2
    assert _tail(pre2, w2) <= len(TOOL) + 1, pre2               # only the new tail is prefilled
    assert_cold(out2, pre2, w2, 8)
    # the old path: a guest, its rows put back under the other chat's after each call
    _, _, w2_, _, pre_old = _calls_beside_a_long_chat(stay=False)
    assert pre_old["reused"] == 0 and _tail(pre_old, w2_) == len(w2_), pre_old


def test_a_short_request_without_the_worker_header_is_still_a_guest():
    res = _state(guest_max=8192)
    other = tokens(400, seed=70)
    out, _ = wturn(other, 8, "chat-other")
    wturn(tokens(60, seed=74), 4, "title")
    assert res.owner == "chat-other" and res.stats["guests"] == 1, res.stats
    _, pre = wturn(other + out + tokens(9, seed=75), 4, "chat-other")
    assert pre["reused"] >= len(other) + len(out) - 1, pre


# ------------------------------------------------------------------ 2: two chats' workers at once

def _two_chats(stay, d):
    """Two parents (348 rows each, the same head) and their workers (80 tokens, the same worker
    head), taking turns on slot 0 with a park that holds one conversation and the NVMe spill.
    Returns the resident prefix, each worker's (tail prefilled, prompt) of calls 2 and 3, and
    each parent's return."""
    per = sum(t.numel() // t.shape[ax] * t.element_size()
              for t, ax in cache._kv_views(fresh(), None))
    res = _state(guest_max=8192, park_bytes=900 * per + 8 * fresh().state.nbytes,
                 park_anchors=4)
    _spill(d, res, min_tokens=64)
    app.PARENTS.clear()
    p = {k: PHEAD + tokens(300, seed=s) for k, s in (("a", 80), ("b", 81))}
    w = {k: WHEAD + tokens(40, seed=s) for k, s in (("a", 82), ("b", 83))}
    pout = {}
    for k in "ab":
        pout[k], _ = wturn(p[k], 8, f"chat-{k}")
    tails, calls = {}, {}
    for step in range(3):
        for k in "ab":
            app._note_parent(f"chat-{k}")
            out, pre = wturn(w[k], 6, f"chat-{k}-task-1", stay)
            calls[k] = (w[k], out, pre)
            if step:
                tails.setdefault(k, []).append((_tail(pre, w[k]), len(w[k])))
            w[k] = w[k] + out + tokens(12, seed=90 + 2 * step + (k == "b"))
    back = {}
    for k in "ab":
        ids = p[k] + pout[k] + tokens(15, seed=95 + (k == "b"))
        out, pre = wturn(ids, 6, f"chat-{k}")
        back[k] = (ids, out, pre)
    return res, tails, back, calls


def test_two_chats_workers_each_keep_their_prefix_and_the_parents_come_back():
    d = tempfile.mkdtemp()
    try:
        res, tails, back, calls = _two_chats(True, d)
        for k in "ab":
            assert all(t <= 12 + 1 for t, _ in tails[k]), (k, tails)   # each call: its tail only
            ids, out, pre = back[k]
            assert _tail(pre, ids) <= 15 + 1, (k, pre)              # the parent: its tail only
        assert res.stats["park_swaps"] >= 4 and res.stats["park_spills"] >= 2, res.stats
        for k in "ab":
            assert_cold(*back[k][1:], back[k][0], 6)
            assert_cold(*calls[k][1:], calls[k][0], 6)        # each worker's last call
    finally:
        shutil.rmtree(d, ignore_errors=True)
    d = tempfile.mkdtemp()
    try:
        _, tails, _, _ = _two_chats(False, d)
        # the old path: call 1, shorter than a quarter of the parent, was a guest, and call 2
        # read again all of its prompt past the worker head the two workers share
        assert all(t >= n - len(WHEAD) for k in "ab" for t, n in tails[k][:1]), tails
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ------------------------------------------------------------------ 3: the parent on a side slot

def _side_parent(keeper, d):
    """The parent ran on a side slot (no park, no spill of its own); its worker's call takes that
    slot; the parent's return runs on slot 0."""
    res0 = _state(guest_max=8192)
    _spill(d, res0, min_tokens=64)
    eng0 = app.STATE["engine"]
    side = cache.ResidentPrefix(BIG, CH, end=True, stash_bytes=BIG, guest_max=8192,
                                guest_frac=1.0, park_bytes=0)
    side.keeper = keeper
    eng1 = fresh()
    app.PARENTS.clear()
    p1 = PHEAD + tokens(300, seed=84)
    app.STATE.update(engine=eng1, resident=side)              # the side slot's thread
    pout, _ = wturn(p1, 8, "chat-p")
    app._note_parent("chat-p")
    wturn(WHEAD + tokens(40, seed=85), 6, "chat-p-task-1", stay=True)
    assert side.owner == "chat-p-task-1"
    app.STATE.update(engine=eng0, resident=res0)              # slot 0
    ids = p1 + pout + tokens(15, seed=86)
    out, pre = wturn(ids, 6, "chat-p")
    return ids, out, pre


def test_a_parent_on_a_side_slot_goes_to_the_nvme_before_its_worker_takes_the_slot():
    d = tempfile.mkdtemp()
    try:
        ids, out, pre = _side_parent(app._keep_parent, d)
        assert _tail(pre, ids) <= 15 + 1, pre
        assert_cold(out, pre, ids, 6)
    finally:
        shutil.rmtree(d, ignore_errors=True)
    d = tempfile.mkdtemp()
    try:
        ids, _, pre = _side_parent(None, d)                     # the old side slot
        assert pre["reused"] == 0, pre
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_a_workers_side_rows_go_to_the_nvme_before_another_session_takes_the_slot():
    """Live 2026-10-18 01:1x: one chat's worker took the side slot the other chat's worker ran
    on (slot 0 busy), and the other worker's next call read its prompt again."""
    d = tempfile.mkdtemp()
    try:
        _state()
        sp, _ = _spill(d, app.STATE["resident"], min_tokens=64)
        side = cache.ResidentPrefix(BIG, CH, end=True, stash_bytes=BIG, guest_frac=1.0)
        side.keeper = app._keep_parent
        app.PARENTS.clear()
        app.WORKERS.clear()
        app.STATE.update(engine=fresh(), resident=side)
        wturn(WHEAD + tokens(200, seed=104), 8, "chat-a-task-1", stay=True)
        app._note_parent("chat-a", worker="chat-a-task-1")
        wturn(WHEAD + tokens(220, seed=105), 6, "chat-b-task-1", stay=True)
        assert len(sp.entries) == 1, sp.entries
    finally:
        app.WORKERS.clear()
        shutil.rmtree(d, ignore_errors=True)


def test_a_side_slots_rows_of_a_session_that_is_no_parent_are_not_written():
    d = tempfile.mkdtemp()
    try:
        _state()
        sp, _ = _spill(d, app.STATE["resident"], min_tokens=64)
        side = cache.ResidentPrefix(BIG, CH, end=True, stash_bytes=BIG, guest_frac=1.0)
        side.keeper = app._keep_parent
        app.PARENTS.clear()
        app.STATE.update(engine=fresh(), resident=side)
        wturn(PHEAD + tokens(400, seed=87), 8, "chat-q")
        wturn(PHEAD + tokens(420, seed=88), 6, "chat-r")
        assert not sp.entries, sp.entries
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _swap_longer(grow):
    """Two workers on slot 0 with one park: A (short) is parked by B (long); A's next call swaps
    them, and B must go to the park whole with its end, although the park's buffers were sized
    for A. Then B's next call."""
    per = sum(t.numel() // t.shape[ax] * t.element_size()
              for t, ax in cache._kv_views(fresh(), None))
    res = _state(guest_max=8192, park_bytes=480 * per + 8 * fresh().state.nbytes)
    if not grow:                                 # the park as before ENG-371: no growth
        res._grow_park = lambda eng, drafter, views, rows, keep: \
            res._park_bufs[0].shape[views[0][1]]
    a = WHEAD + tokens(60, seed=100)
    oa, _ = wturn(a, 6, "chat-a-task-1", stay=True)
    b = WHEAD + tokens(300, seed=101)
    ob, _ = wturn(b, 6, "chat-b-task-1", stay=True)        # A to the park: a small buffer
    assert res.park_owner == "chat-a-task-1"
    a = a + oa + tokens(10, seed=102)
    oa, pa = wturn(a, 6, "chat-a-task-1", stay=True)       # the swap: B to the park
    b = b + ob + tokens(10, seed=103)
    ob2, pb = wturn(b, 6, "chat-b-task-1", stay=True)
    return res, a, pa, b, pb, ob2


def test_a_swap_parks_a_longer_conversation_whole():
    res, a, pa, b, pb, ob2 = _swap_longer(grow=True)
    assert _tail(pa, a) <= 10 + 1, pa                    # A back from the park
    assert _tail(pb, b) <= 10 + 1 and res.stats.get("park_grown"), (pb, res.stats)
    assert_cold(ob2, pb, b, 6)
    _, _, _, b0, pb0, _ = _swap_longer(grow=False)
    # the old swap dropped B's end and every row above A's buffers: B resumed at a grid line
    assert pb0["kind"] != "end" and _tail(pb0, b0) > 150, pb0


# ------------------------------------------------------------------ 4: the park and the memory policy

def _parked_parent(shed, d):
    per = sum(t.numel() // t.shape[ax] * t.element_size()
              for t, ax in cache._kv_views(fresh(), None))
    res = _state(guest_max=8192, park_bytes=900 * per + 8 * fresh().state.nbytes)
    _spill(d, res, min_tokens=64)
    app.PARENTS.clear()
    p1 = PHEAD + tokens(300, seed=89)
    pout, _ = wturn(p1, 8, "chat-p")
    app._note_parent("chat-p")
    wturn(WHEAD + tokens(40, seed=90), 6, "chat-p-task-1", stay=True)
    assert res.park_owner == "chat-p" and res.park_rows, (res.park_owner, res.park_rows)
    shed(res)                                                  # the gate at the ceiling
    assert not res.park_rows
    ids = p1 + pout + tokens(15, seed=91)
    out, pre = wturn(ids, 6, "chat-p")
    return ids, out, pre


def test_a_parked_parent_goes_to_the_nvme_before_the_memory_policy_takes_the_park():
    d = tempfile.mkdtemp()
    try:
        ids, out, pre = _parked_parent(lambda r: app._shed_park(r), d)
        assert _tail(pre, ids) <= 15 + 1, pre
        assert_cold(out, pre, ids, 6)
    finally:
        shutil.rmtree(d, ignore_errors=True)
    d = tempfile.mkdtemp()
    try:
        ids, _, pre = _parked_parent(lambda r: r.shed_park(), d)     # the old shedder
        assert _tail(pre, ids) > 300, pre
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_the_memory_gate_keeps_the_end_the_request_resumes_at():
    """Live 2026-10-18 01:58: the gate gave back the end a worker's call had just been carried
    into slot 0 with, and the call read its 7,953 tokens from 0. With `end_hold` the end stays and
    the levels after it go."""
    res = _state()
    ids = WHEAD + tokens(100, seed=106)
    out, _ = wturn(ids, 6, "chat-a-task-1", stay=True)
    assert res.end is not None
    res.end_hold = True
    assert res.shed_end(False) == 0 and res.end is not None
    res.end_hold = False
    assert res.shed_end(False) > 0 and res.end is None and res.end_lost == "memory"


def _refused_park(spill_on, d):
    """A conversation another session replaces while the memory policy refuses the park: with
    the NVMe spill it is written there (ENG-371), and its next request reads its tail only."""
    per = sum(t.numel() // t.shape[ax] * t.element_size()
              for t, ax in cache._kv_views(fresh(), None))
    res = _state(guest_max=8192, park_bytes=900 * per + 8 * fresh().state.nbytes)
    if spill_on:
        _spill(d, res, min_tokens=64)
    res._park = lambda eng, drafter, keep: res.stats.__setitem__(   # the policy said no
        "refused_park", res.stats["refused_park"] + 1)
    p1 = PHEAD + tokens(300, seed=107)
    out, _ = wturn(p1, 8, "chat-p")
    wturn(WHEAD + tokens(40, seed=108), 6, "chat-p-task-1", stay=True)
    ids = p1 + out + tokens(15, seed=109)
    out2, pre = wturn(ids, 6, "chat-p")
    return ids, out2, pre


def test_a_conversation_the_park_refuses_goes_to_the_nvme_not_away():
    """Live 2026-10-18 02:34: under the memory ceiling the park refused a worker's 27,381 rows,
    they were dropped, and its next call read them all again."""
    d = tempfile.mkdtemp()
    try:
        ids, out, pre = _refused_park(True, d)
        assert _tail(pre, ids) <= 15 + 1, pre
        assert_cold(out, pre, ids, 6)
    finally:
        shutil.rmtree(d, ignore_errors=True)
    d = tempfile.mkdtemp()
    try:
        ids, _, pre = _refused_park(False, d)            # no spill: dropped, as before
        assert pre["reused"] == 0, pre
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_a_carry_that_vacates_slot_0_writes_what_the_park_refuses():
    """The same at a side slot's carry into slot 0 (`vacate`): live 02:34 `slot 0's conversation
    dropped`."""
    d = tempfile.mkdtemp()
    try:
        res = _state(guest_max=8192, park_bytes=1 << 30)
        sp, _ = _spill(d, res, min_tokens=64)
        res._park = lambda eng, drafter, keep: res.stats.__setitem__(
            "refused_park", res.stats["refused_park"] + 1)
        wturn(PHEAD + tokens(300, seed=110), 8, "chat-q-task-1", stay=True)
        assert res.vacate(app.STATE["engine"], None, "chat-r-task-1") == "spilled"
        assert len(sp.entries) == 1 and res.valid == 0
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_a_short_conversation_an_nvme_restore_reads_over_goes_to_the_park():
    """Live 2026-10-18 04:07: worker 1's entry was read back over worker 2's 11,509 rows (shorter
    than an eviction's length, so not written), and worker 2's next call read 11,527 tokens from
    0. The restore parks it first; its next call resumes there."""
    d = tempfile.mkdtemp()
    try:
        res = _state(guest_max=8192, park_bytes=1 << 30)
        sp, _ = _spill(d, res, min_tokens=10_000)          # nothing here is an eviction's length
        a = WHEAD + tokens(300, seed=60)
        oa, _ = wturn(a, 6, "chat-a-task-1", stay=True)
        assert sp.save_resident(res, app.STATE["engine"], None, "evict")
        res.clear()
        b = WHEAD + tokens(200, seed=61)
        ob, _ = wturn(b, 6, "chat-b-task-1", stay=True)
        a2 = a + oa + tokens(10, seed=62)
        with torch.no_grad():
            app._spill_before_replay(a2)                     # A read back over B's rows
        assert res.owner == "chat-a-task-1" and res.park_owner == "chat-b-task-1"
        b2 = b + ob + tokens(10, seed=63)
        out, pre = wturn(b2, 6, "chat-b-task-1", stay=True)
        assert _tail(pre, b2) <= 10 + 1, pre
        assert_cold(out, pre, b2, 6)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_a_park_never_writes_over_the_entry_a_restore_reads_or_a_newer_one():
    """Live 2026-10-18 05:36: a conversation's older park went to the NVMe under its own entry,
    which a restore was about to read (and which held it further); the restore failed its read and
    the turn re-read 10,073 tokens. A save into the entry being read, or of an older, shorter copy
    over the entry, is refused."""
    d = tempfile.mkdtemp()
    try:
        res = _state(guest_max=8192, park_bytes=1 << 30)
        sp, _ = _spill(d, res, min_tokens=10_000)
        x = WHEAD + tokens(200, seed=64)
        ox, _ = wturn(x, 6, "chat-x-task-1", stay=True)
        wturn(WHEAD + tokens(80, seed=65), 6, "chat-y-task-1", stay=True)   # X to the park
        assert res.park_owner == "chat-x-task-1"
        # X further along, written from another slot: the newer entry
        res2 = cache.ResidentPrefix(BIG, CH, end=True, stash_bytes=BIG)
        saved = dict(app.STATE)
        app.STATE.update(engine=fresh(), resident=res2)
        x2 = x + ox + tokens(40, seed=66)
        wturn(x2, 6, "chat-x-task-1", stay=True)
        assert sp.save_resident(res2, app.STATE["engine"], None, "switch")
        app.STATE.clear()
        app.STATE.update(saved)
        (e,) = sp.entries.values()
        n = e.valid
        assert not res._spill_park(app.STATE["engine"], None)          # older: refused
        assert next(iter(sp.entries.values())).valid == n
        sp._keep.add(e.eid)
        try:
            assert sp.save_resident(res2, fresh(), None, "switch") is None   # being read: refused
        finally:
            sp._keep.discard(e.eid)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_the_park_goes_unwritten_wherever_a_write_would_be_unsafe():
    """On a side slot's thread, with `safe`, while the spill's I/O is held, while a guest's rows
    lie over the park's head, and while the park is in use: the park goes (or stays), unwritten."""
    class Sp:
        def __init__(self):
            self._own_io = threading.Lock()

    class Res:
        park_rows, park_base, park_owner, _stash, _park_busy = 300, 40, "chat-p", None, 0

        def __init__(self):
            self.spiller, self.spilled = Sp(), 0

        def _spill_park(self, eng, drafter):
            self.spilled += 1
            return True

        def shed_park(self, safe=False):
            self.park_rows = 0
            return 1

    app.PARENTS.clear()
    app._note_parent("chat-p")
    r = Res()
    slots_mod.bind(slots_mod.Slot(1, 1024))
    try:
        assert app._shed_park(r) == 0 and r.park_rows and r.spilled == 0  # waited for: kept
        app._PARK_LET_GO.on = True                                        # a refused gate's retry
        assert app._shed_park(r) == 1 and r.spilled == 0
    finally:
        app._PARK_LET_GO.on = False
        slots_mod.bind(None)
    r = Res()
    assert app._shed_park(r, True) == 0 and r.park_rows and r.spilled == 0   # waited for: kept
    r = Res()
    with r.spiller._own_io:
        assert app._shed_park(r) == 1 and r.spilled == 0
    r = Res()
    r._stash = {"start": 0}                                          # a guest from row 0
    assert app._shed_park(r) == 1 and r.spilled == 0
    r = Res()
    r._stash = {"start": 64}                                         # above the park's head
    assert app._shed_park(r) == 1 and r.spilled == 1
    r = Res()
    assert app._shed_park(r) == 1 and r.spilled == 1


# ------------------------------------------------------------------ 5: reclaims inside a cache operation
# The KV pool's page gate (server/app.py `_kv_reclaim_caches`) sheds every level with safe=True from
# inside `kv.ensure` and `kvpool.ensure_for`: inside a restore, a swap and a forward. The park's
# shedder must then only give the park back (or nothing, when the park is in use), never write it.

def _budget_state(d):
    per = sum(t.numel() // t.shape[ax] * t.element_size()
              for t, ax in cache._kv_views(fresh(), None))
    mem = cache.Budget(1 << 40, 1 << 41, 0, probe=lambda: (0, 0), empty=lambda: None)
    res = _state(guest_max=8192, park_bytes=900 * per + 8 * fresh().state.nbytes, mem=mem)
    mem.register("park", lambda safe=False: app._shed_park(res, safe))   # as `_load_caches`
    mem.register("stash", res.shed_stash)
    sp, _ = _spill(d, res, min_tokens=64)
    app.PARENTS.clear()
    app.WORKERS.clear()
    return res, mem, sp


def test_a_reclaim_inside_a_restore_does_not_deadlock():
    """`SessionSpill.restore` holds its own I/O lock and commits pages (`ensure_for`); a park
    write from the reclaim there waited on that lock on the same thread, for ever."""
    d = tempfile.mkdtemp()
    real = kvpool.ensure_for
    try:
        res, mem, sp = _budget_state(d)
        w = WHEAD + tokens(300, seed=1)
        wout, _ = wturn(w, 6, "chat-w")
        assert sp.save_resident(res, app.STATE["engine"], None, "evict")
        res.clear()
        p = PHEAD + tokens(300, seed=2)
        wturn(p, 8, "chat-p")
        app._note_parent("chat-p")
        wturn(WHEAD + tokens(40, seed=3), 6, "chat-p-task-1", stay=True)
        assert res.park_owner == "chat-p"
        calls = []

        def ensure_for(t, rows):
            if threading.current_thread().name == "restore" and not calls:
                calls.append(rows)
                app.STATE["mem"] = mem
                app._kv_reclaim_caches(1 << 40, "slot0")
            return real(t, rows)
        kvpool.ensure_for = ensure_for
        done = {}

        def run():
            with torch.no_grad():
                app._spill_before_replay(w + wout + tokens(10, seed=4))
            done["ok"] = True
        th = threading.Thread(target=run, name="restore", daemon=True)
        th.start()
        th.join(30)
        assert not th.is_alive(), "the restore waits on its own I/O lock"
        assert done.get("ok") and calls, (done, calls)
    finally:
        kvpool.ensure_for = real
        shutil.rmtree(d, ignore_errors=True)


def test_a_reclaim_inside_a_guests_forward_leaves_the_rows_under_it():
    """A reclaim during a guest's generation put the resident's rows back under the running guest
    (`settle_guest` from the shedder); the resident later resumed on rows 0.037 off."""
    d = tempfile.mkdtemp()
    try:
        res, mem, _ = _budget_state(d)
        wturn(PHEAD + tokens(200, seed=31), 8, "chat-p")
        app._note_parent("chat-p")
        w = WHEAD + tokens(340, seed=32)
        wo, _ = wturn(w, 8, "chat-p-task-1", stay=True)          # P to the park, W resident
        eng = app.STATE["engine"]
        real, n = eng.kv.ensure, {"k": 0}

        def ensure(rows):
            n["k"] += 1
            if n["k"] == 6:                                      # inside the guest's decode
                assert res._stash is not None
                app.STATE["mem"] = mem
                app._kv_reclaim_caches(1 << 40, "slot0")
            return real(rows)
        eng.kv.ensure = ensure
        try:
            wturn(tokens(30, seed=33), 12, "title")              # a guest of W
        finally:
            eng.kv.ensure = real
        assert res.stats["guests"] == 1 and n["k"] >= 6
        w2 = w + wo + tokens(10, seed=34)
        out, pre = wturn(w2, 8, "chat-p-task-1", stay=True)
        assert pre["reused"] >= len(w) + len(wo) - 1, pre
        assert_cold(out, pre, w2, 8)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_a_reclaim_inside_a_swap_writes_no_mixed_rows():
    """A reclaim in `unpark`'s swap loop wrote a half-swapped mix of two conversations into an
    NVMe entry, its block sums valid; resumed from it, logits 0.17 off."""
    d = tempfile.mkdtemp()
    real = kvpool.ensure_for
    try:
        res, mem, sp = _budget_state(d)
        a = WHEAD + tokens(300, seed=11)
        oa, _ = wturn(a, 6, "chat-a-task-1", stay=True)
        app._note_parent("chat-a", worker="chat-a-task-1")
        wturn(WHEAD + tokens(120, seed=12), 6, "chat-b-task-1", stay=True)   # A to the park
        assert res.park_owner == "chat-a-task-1"
        n = {"k": 0, "arm": False}

        def ensure_for(t, rows):
            if n["arm"]:
                n["k"] += 1
                if n["k"] == 2:
                    app.STATE["mem"] = mem
                    app._kv_reclaim_caches(1 << 40, "slot0")
            return real(t, rows)
        kvpool.ensure_for = ensure_for
        ru = res.unpark

        def unpark(e_, d_):
            n["arm"] = True
            try:
                return ru(e_, d_)
            finally:
                n["arm"] = False
        res.unpark = unpark
        a2 = a + oa + tokens(10, seed=13)
        out, pre = wturn(a2, 6, "chat-a-task-1", stay=True)
        kvpool.ensure_for = real
        assert n["k"] >= 2, n
        assert_cold(out, pre, a2, 6)                             # A back from the park whole
        for e in sp.entries.values():                            # and whatever was written
            res2, eng2 = cache.ResidentPrefix(1 << 40, res.chunk, end=True), fresh()
            assert sp.restore(res2, eng2, None, e)
            tail = list(e.tokens) + tokens(10, seed=21)
            lg = cache.prefill(eng2, None, tail, "cpu", resident=res2, chunk=res.chunk)[0]
            lc = cache.prefill(fresh(), None, tail, "cpu", chunk=res.chunk)[0]
            assert float((lg.float() - lc.float()).abs().max()) <= TOL, e.man.get("owner")
    finally:
        kvpool.ensure_for = real
        shutil.rmtree(d, ignore_errors=True)


def test_the_park_shedder_writes_at_the_memory_gate_only():
    """Through the memory policy: the gate (safe=False, slot 0's thread) writes a parked parent
    to the NVMe before giving the park back; a safe shed keeps it (live 2026-10-18 04:43 a pin's
    admission dropped a worker's parked rows), and gives another's park back unwritten."""
    for safe, parent, freed, written in ((False, True, True, 1), (True, True, False, 0),
                                          (True, False, True, 0)):
        d = tempfile.mkdtemp()
        try:
            res, mem, sp = _budget_state(d)
            wturn(PHEAD + tokens(300, seed=41), 8, "chat-p")
            if parent:
                app._note_parent("chat-p")
            wturn(WHEAD + tokens(40, seed=42), 6, "chat-p-task-1", stay=True)
            assert res.park_owner == "chat-p"
            # a safe shed (a pin's admission, the KV pool's reclaim) keeps a waited-for park
            assert (mem.shed("park", "test", safe=safe) > 0) == freed, (safe, parent)
            assert bool(res.park_rows) != freed and len(sp.entries) == written, (safe, parent)
        finally:
            shutil.rmtree(d, ignore_errors=True)


def test_the_park_is_not_given_back_while_it_is_in_use():
    res = _state(park_bytes=1 << 30)
    wturn(PHEAD + tokens(200, seed=43), 6, "chat-p")
    wturn(WHEAD + tokens(60, seed=44), 6, "chat-p-task-1", stay=True)
    assert res.park_rows
    res._park_busy += 1
    try:
        assert res.shed_park(True) == 0 and res.shed_park(False) == 0 and res.park_rows
    finally:
        res._park_busy -= 1
    assert res.shed_park(True) > 0 and not res.park_rows


def test_the_gate_holds_the_end_the_request_resumes_at_until_its_prefill():
    """`_gate_holding_end`: held through the gate and until the prefill's `begin` (a side slot's
    gate cannot drop it between); a refusal with the end held runs once more without it."""
    res = _state()
    ids = WHEAD + tokens(100, seed=45)
    out, _ = wturn(ids, 6, "chat-a-task-1", stay=True)
    nxt = ids + out + tokens(8, seed=46)
    assert app._end_held(res, nxt, len(nxt), False)
    assert not app._end_held(res, tokens(50, seed=47), 50, False)

    class Mem:
        def __init__(self, refuse):
            self.refuse, self.seen = refuse, []

        def refused(self, exc):
            pass

        def gate(self, n, count=True):
            self.seen.append(res.end_hold)
            if self.refuse and res.end_hold:
                raise cache.MemoryRefused(2, 1, n)
    m = Mem(False)
    app._gate_holding_end(m, res, nxt, len(nxt), False)
    assert m.seen == [True] and res.end_hold, "held until the prefill"
    assert res.shed_end(False) == 0 and res.end is not None
    wturn(nxt, 4, "chat-a-task-1", stay=True)
    assert not res.end_hold, "the prefill's begin lets it go"
    m = Mem(True)
    nxt2 = nxt + tokens(4, seed=48)
    res.end_hold = False
    o, _ = wturn(nxt, 4, "chat-a-task-1", stay=True)
    nxt2 = nxt + o + tokens(4, seed=48)
    app._gate_holding_end(m, res, nxt2, len(nxt2), False)
    assert m.seen == [True, False] and not res.end_hold, m.seen


def test_a_gate_refused_with_a_waited_for_park_kept_runs_once_more_and_may_take_it():
    """A side slot's gate keeps slot 0's parked parent (it cannot write it there); when that gate
    would refuse, it runs once more and the park may go, so no request gets a 503 for it."""
    res = _state(park_bytes=1 << 30)
    wturn(PHEAD + tokens(200, seed=55), 6, "chat-p")
    app.PARENTS.clear()
    app._note_parent("chat-p")
    wturn(WHEAD + tokens(60, seed=56), 6, "chat-p-task-1", stay=True)
    assert res.park_owner == "chat-p"

    class Mem:
        def __init__(self):
            self.passes = 0

        def refused(self, exc):
            pass

        def gate(self, n, count=True):
            self.passes += 1
            if not app._shed_park(res, False):
                raise cache.MemoryRefused(2, 1, n)
    slots_mod.bind(slots_mod.Slot(1, 1024))
    try:
        m = Mem()
        app._gate_holding_end(m, None, [], 10, False)
        assert m.passes == 2 and not res.park_rows, (m.passes, res.park_rows)
    finally:
        slots_mod.bind(None)
        app.PARENTS.clear()


def test_a_layer_mark_waits_for_no_capture():
    """SRV-591's layer marks are CUDA calls on the prefilling thread, a side slot's too: while a
    graph capture holds `capture_lock.LOCK` (another thread) the mark is skipped."""
    class Ev:
        recorded = 0

        def record(self):
            Ev.recorded += 1

        def query(self):
            return True
    live = {"layers": [], "n": 4, "t0": 0.0, "rows": (0, 64)}
    real_ev, real_ok = torch.cuda.Event, torch.cuda.is_available
    torch.cuda.Event, torch.cuda.is_available = Ev, (lambda: True)
    try:
        mark = cache._layer_marks(live, "cuda")
        held, go = threading.Event(), threading.Event()

        def capture():
            with capture_lock.LOCK:
                held.set()
                go.wait(10)
        th = threading.Thread(target=capture)
        th.start()
        held.wait(10)
        mark()
        assert Ev.recorded == 0 and not live["layers"], "a CUDA call inside the capture"
        go.set()
        th.join(10)
        mark()
        assert Ev.recorded == 1 and live["k"] == 1
    finally:
        torch.cuda.Event, torch.cuda.is_available = real_ev, real_ok


def test_a_worker_on_a_side_slot_writes_another_clients_rows_before_it_cuts_them():
    """Not a guest there (`stay`), a worker's call replaces the side slot's conversation: that
    one goes to the NVMe first, whoever's it is."""
    d = tempfile.mkdtemp()
    try:
        _state()
        sp, _ = _spill(d, app.STATE["resident"], min_tokens=64)
        side = cache.ResidentPrefix(BIG, CH, end=True, stash_bytes=BIG, guest_max=8192,
                                    guest_frac=1.0)
        side.keeper = app._keep_parent
        app.PARENTS.clear()
        app.WORKERS.clear()
        app.STATE.update(engine=fresh(), resident=side)
        wturn(PHEAD + tokens(300, seed=49), 8, "owui-chat")
        wturn(WHEAD + tokens(40, seed=50), 6, "chat-a-task-1", stay=True)
        assert [e.man.get("owner") for e in sp.entries.values()] == ["owui-chat"], sp.entries
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_the_park_takes_nothing_and_nothing_is_written():
    """A conversation that shares all but a few rows with the next one of another session: the
    park has nothing to take (fewer rows than a chunk) and that is no refusal to spill for."""
    d = tempfile.mkdtemp()
    try:
        res = _state(guest_max=8192, park_bytes=1 << 30)
        sp, _ = _spill(d, res, min_tokens=64)
        p = PHEAD + tokens(300, seed=51)
        wturn(p, 1, "chat-a")
        wturn(p[:-3] + tokens(3, seed=52), 4, "chat-b")
        assert not sp.entries and res.stats["refused_park"] == 0, sp.entries
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_a_grown_park_asks_the_memory_policy_for_both_buffers():
    """`_grow_park` holds the old buffers and the new at once while it copies: it asks for the
    new buffers' whole size, not the growth."""
    res = _state(park_bytes=1 << 30)
    wturn(PHEAD + tokens(60, seed=53), 6, "chat-a")
    wturn(WHEAD + tokens(60, seed=54), 6, "chat-b")
    assert res.park_rows

    class Rec:
        asked = []

        def admit(self, level, n):
            Rec.asked.append((level, n))
            return True
    res.mem = Rec()
    eng = app.STATE["engine"]
    views = cache._kv_views(eng, None)
    per = sum(t.numel() // t.shape[ax] * t.element_size() for t, ax in views)
    size = res._grow_park(eng, None, views, 400, res.park_rows)
    assert size >= 400 and Rec.asked == [("park", size * per)], (size, Rec.asked)


# ------------------------------------------------------------------ 5b: the second review

def test_the_sandbox_release_writes_and_frees_a_parked_parent():
    """The sandbox has priority over what the engine holds idle: `POST /v1/cache/release`
    (`_memory_release`) frees the park a safe shed keeps for a parent, writing it to the NVMe
    first. Before, it freed nothing and the sandbox was short."""
    for waited in (True, False):
        d = tempfile.mkdtemp()
        try:
            res, mem, sp = _budget_state(d)
            app.STATE["mem"] = mem
            wturn(PHEAD + tokens(300, seed=71), 8, "chat-p")
            if waited:
                app._note_parent("chat-p")
            wturn(WHEAD + tokens(40, seed=72), 6, "chat-p-task-1", stay=True)
            assert res.park_rows
            out = app._memory_release(1 << 40, "sandbox")
            assert not res.park_rows and out["released_bytes"] > 0, out
            assert len(sp.entries) == (1 if waited else 0), (waited, sp.entries)
        finally:
            app.PARENTS.clear()
            shutil.rmtree(d, ignore_errors=True)


def test_the_kv_pool_frees_a_kept_park_before_it_refuses():
    """The page gate inside a forward or a restore (its reclaimers in order) frees a waited-for
    park unwritten, last, rather than raise PoolFull mid-generation; a park in use stays."""
    res = _state(park_bytes=1 << 30)
    wturn(PHEAD + tokens(200, seed=73), 6, "chat-p")
    app.PARENTS.clear()
    app._note_parent("chat-p")
    wturn(WHEAD + tokens(60, seed=74), 6, "chat-p-task-1", stay=True)
    assert res.park_rows
    pool = kvpool.KVPool(hard_floor=1 << 30,
                         meminfo=lambda: (2 << 30) if not res.park_rows else (1 << 29))
    app._add_kv_reclaimers(pool)
    assert [n for n, _ in pool.reclaimers][-1] == "kept_park"
    res._park_busy += 1
    try:
        assert app._kv_reclaim_kept_park(1 << 30, "slot0") == 0 and res.park_rows
    finally:
        res._park_busy -= 1
    pool.gate("slot0", 1 << 20)                          # no PoolFull: the kept park went
    assert not res.park_rows
    app.PARENTS.clear()


def test_a_pool_full_inside_a_swap_moves_no_row():
    """A PoolFull from the second view's page commit inside `unpark`'s swap left the rows in
    place half swapped, and a later resume on them was 0.19 off a cold prefill: every view's
    pages are committed before a row moves."""
    d = tempfile.mkdtemp()
    real = kvpool.ensure_for
    try:
        per = sum(t.numel() // t.shape[ax] * t.element_size()
                  for t, ax in cache._kv_views(fresh(), None))
        res = _state(guest_max=8192, park_bytes=900 * per + 8 * fresh().state.nbytes)
        _spill(d, res, min_tokens=64)
        app.PARENTS.clear()
        a = WHEAD + tokens(300, seed=11)
        oa, _ = wturn(a, 6, "chat-a")
        b = WHEAD + tokens(120, seed=12)
        ob, _ = wturn(b, 6, "chat-b")
        n = {"k": 0, "arm": False}

        def ensure_for(t, rows):
            if n["arm"]:
                n["k"] += 1
                if n["k"] == 2:
                    raise kvpool.PoolFull("KV pool: under the hard floor after every reclaim")
            return real(t, rows)
        kvpool.ensure_for = ensure_for
        ru = res.unpark

        def unpark(e_, d_):
            n["arm"] = True
            try:
                return ru(e_, d_)
            finally:
                n["arm"] = False
        res.unpark = unpark
        try:
            wturn(a + oa + tokens(10, seed=13), 6, "chat-a")
        except kvpool.PoolFull:
            pass
        kvpool.ensure_for = real
        res.unpark = ru
        assert n["k"] >= 2 and res.owner == "chat-b" and res.park_owner == "chat-a"
        b2 = b + ob + tokens(10, seed=14)
        out, pre = wturn(b2, 6, "chat-b")
        assert_cold(out, pre, b2, 6)
    finally:
        kvpool.ensure_for = real
        shutil.rmtree(d, ignore_errors=True)


def test_a_reclaim_inside_a_swap_of_a_park_nobody_waits_for():
    """The same swap with the park nobody waits for: the reclaim leaves it alone while it is in
    use (`_park_busy`), and the swap ends whole."""
    d = tempfile.mkdtemp()
    real = kvpool.ensure_for
    try:
        res, mem, sp = _budget_state(d)
        a = WHEAD + tokens(300, seed=11)
        oa, _ = wturn(a, 6, "chat-a-task-1")
        wturn(WHEAD + tokens(120, seed=12), 6, "chat-b-task-1")
        n = {"k": 0, "arm": False}

        def ensure_for(t, rows):
            if n["arm"]:
                n["k"] += 1
                if n["k"] == 2:
                    app.STATE["mem"] = mem
                    app._kv_reclaim_caches(1 << 40, "slot0")
            return real(t, rows)
        kvpool.ensure_for = ensure_for
        ru = res.unpark

        def unpark(e_, d_):
            n["arm"] = True
            try:
                return ru(e_, d_)
            finally:
                n["arm"] = False
        res.unpark = unpark
        a2 = a + oa + tokens(10, seed=13)
        out, pre = wturn(a2, 6, "chat-a-task-1")
        kvpool.ensure_for = real
        assert n["k"] >= 2 and res.park_owner == "chat-b-task-1" and res.park_rows, n
        assert_cold(out, pre, a2, 6)
    finally:
        kvpool.ensure_for = real
        shutil.rmtree(d, ignore_errors=True)


def test_a_hold_left_by_a_request_that_prefilled_nothing_blocks_no_other():
    """A request that passes the gate and prefills nothing (A1's give-up, a prepared summary, a
    cache hit, a client gone) left `end_hold` set, and every other request then got a 503:
    every gate sets the hold to its own request's, and the handler's `finally` clears it."""
    res = _state()
    ids = WHEAD + tokens(100, seed=81)
    out, _ = wturn(ids, 6, "chat-a-task-1", stay=True)
    nxt = ids + out + tokens(8, seed=82)

    class Pass:
        def gate(self, n, count=True):
            pass

        def refused(self, exc):
            pass

    class NeedsEnd(Pass):                  # over the peak by less than the end anchor's bytes
        def gate(self, n, count=True):
            if not res.shed_end(False):
                raise cache.MemoryRefused(2, 1, n)
    app._gate_holding_end(Pass(), res, nxt, len(nxt), False)      # no prefill follows
    assert res.end_hold
    other = tokens(50, seed=83)
    app._gate_holding_end(NeedsEnd(), res, other, len(other), False)   # admitted
    assert not res.end_hold and res.end is None
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "server", "app.py")).read()
    assert "res_h.end_hold = False" in src.split("kv_trim_current()\n                if (plan")[0]


def test_a_refusal_counts_once_at_the_outcome():
    """A gate run once more and admitted is no refusal (Budget stats); a final refusal is one."""
    mem = cache.Budget(1 << 20, 1 << 20, 0, probe=lambda: (0, 0), empty=lambda: None)
    try:
        mem.gate(1 << 30, count=False)
    except cache.MemoryRefused as exc:
        assert mem.stats["refused"] == 0
        mem.refused(exc)
    assert mem.stats["refused"] == 1

    class Mem:
        def __init__(self, refuse_twice):
            self.twice, self.passes, self.counted = refuse_twice, 0, 0

        def gate(self, n, count=True):
            self.passes += 1
            if self.passes == 1 or self.twice:
                if count:
                    self.counted += 1
                raise cache.MemoryRefused(2, 1, n)

        def refused(self, exc):
            self.counted += 1
    res = _state()
    ids = WHEAD + tokens(100, seed=84)
    out, _ = wturn(ids, 6, "chat-a-task-1", stay=True)
    nxt = ids + out + tokens(8, seed=85)
    m = Mem(False)
    app._gate_holding_end(m, res, nxt, len(nxt), False)
    assert m.passes == 2 and m.counted == 0, (m.passes, m.counted)
    m = Mem(True)
    try:
        app._gate_holding_end(m, res, nxt, len(nxt), False)
    except cache.MemoryRefused:
        pass
    assert m.passes == 2 and m.counted == 1, (m.passes, m.counted)
    m = Mem(True)
    try:
        app._gate_holding_end(m, res, tokens(40, seed=86), 40, False)   # nothing held
    except cache.MemoryRefused:
        pass
    assert m.passes == 1 and m.counted == 1, (m.passes, m.counted)


def test_a_side_slots_write_waits_for_slot_0s_spill_io():
    """`_keep_parent` skipped the write while slot 0's save or restore held the spill's own
    I/O, and the conversation it cut went unwritten: it waits for the lock now."""
    d = tempfile.mkdtemp()
    try:
        _state()
        sp, _ = _spill(d, app.STATE["resident"], min_tokens=64)
        side = cache.ResidentPrefix(BIG, CH, end=True, stash_bytes=BIG, guest_frac=1.0)
        side.keeper = app._keep_parent
        app.PARENTS.clear()
        app.WORKERS.clear()
        app.STATE.update(engine=fresh(), resident=side)
        wturn(PHEAD + tokens(300, seed=87), 8, "owui-chat")
        sp._own_io.acquire()
        threading.Timer(0.5, sp._own_io.release).start()      # slot 0's I/O ends meanwhile
        wturn(WHEAD + tokens(40, seed=88), 6, "chat-a-task-1", stay=True)
        assert [e.man.get("owner") for e in sp.entries.values()] == ["owui-chat"], sp.entries
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ------------------------------------------------------------------ 6: the slot choice

class FakeRes:
    """Slot 0's resident as `_par_side_need` reads it."""

    def __init__(self, toks, owner=None, chunk=CH, park=0):
        self.tokens, self.valid, self.chunk, self.owner = list(toks), len(toks), chunk, owner
        self.guest_max, self.end, self.park_rows, self._park = 8192, None, park, park

    def common(self, ids):
        return app._shared_head(ids, self.tokens[:self.valid])

    def park_offer(self, ids, c, max_len, align=True):
        return self._park


def _need(res, ids, side=None, **kw):
    pool = slots_mod.Pool([slots_mod.Slot(0, 262_144),
                           slots_mod.Slot(1, 262_144, values={"resident": side, "slot_view": None}
                                          if side is not None else None)])
    saved = {k: app.STATE.get(k) for k in ("pool", "resident", "slot0_view", "prefix_chunk",
                                           "spill")}
    app.STATE.update(pool=pool, resident=res, prefix_chunk=CH, spill=None)
    app.STATE.pop("slot0_view", None)
    real = app.build_prompt
    app.build_prompt = lambda body, client=None, *, count=False, left_out=0: list(ids)

    class Plan:
        active = True
    try:
        why: dict = {}
        need = app._par_side_need({"messages": []}, None, 0, 4096, Plan(), [], None, why, **kw)
        return need, why
    finally:
        app.build_prompt = real
        for k, v in saved.items():
            if v is None:
                app.STATE.pop(k, None)
            else:
                app.STATE[k] = v


def test_the_worker_headers():
    assert app._worker_of({"X-Tandem-Worker": "1", "X-Tandem-Parent-Session": "chat-a"}) == \
        (True, "chat-a")
    assert app._worker_of({"X-Tandem-Parent-Session": "chat-a"}) == (True, "chat-a")
    assert app._worker_of({"X-Tandem-Worker": "1"}) == (True, None)
    assert app._worker_of({"x-session-id": "chat-a"}) == (False, None)


def test_a_worker_is_no_guest_of_slot_0_and_never_waits_for_a_side_slot():
    other = FakeRes(tokens(24_200, seed=1), owner="chat-other")
    ids = tokens(2_532, seed=2)
    _, why = _need(other, ids)
    assert why.get("guest"), why                       # the old rule: a guest of the other chat
    _, why = _need(other, ids, stay=True, parent="chat-p")
    assert not why.get("guest") and not why.get("prefer_side_wait"), why


def test_a_worker_beside_its_parent_on_slot_0_takes_a_side_slot_first():
    parent = FakeRes(tokens(9_000, seed=3), owner="chat-p")
    _, why = _need(parent, tokens(2_532, seed=4), stay=True, parent="chat-p")
    assert why.get("prefer_side") == 9_000 and not why.get("prefer_side_wait"), why
    _, why = _need(parent, tokens(2_532, seed=4), stay=True, parent="chat-q")
    assert not why.get("prefer_side"), why


def test_a_conversation_slot_0_has_parked_tries_slot_0_first():
    _, why = _need(FakeRes(tokens(12_000, seed=5), park=11_000), tokens(11_500, seed=6))
    assert why.get("wait") == "park" and why.get("slot0_first") and why["park"] == 11_000, why
    _, why = _need(FakeRes(tokens(12_000, seed=5)), tokens(11_500, seed=6))
    assert why.get("wait") is None, why


def test_a_worker_whose_copy_parts_inside_its_own_rows_stays_on_slot_0():
    """Live 2026-10-18 01:1x: a worker's call 2 (its write_file content a stub, so the copy parts
    inside call 1's answer: 8,627 of 10,844 rows) counted as no continuation of slot 0 and took
    the side slot another worker's rows were on."""
    own = FakeRes(tokens(10_844, seed=10), owner="chat-b-task-1")
    ids = own.tokens[:8_627] + tokens(100, seed=11)
    _, why = _need(own, ids, session="chat-b-task-1")
    assert why.get("wait") is None, why                # not a worker: as before
    _, why = _need(own, ids, session="chat-b-task-1", stay=True, parent="chat-b")
    assert why.get("wait") == "own" and why.get("slot0_first"), why
    # it waits for slot 0 up to the time a re-read of its prompt would take (live 02:02: 3.1 s,
    # then the side slot read 15,170 tokens from 0)
    assert app.SIDE_SLOT0_WAIT_S <= why["slot0_wait_s"] <= app.PARENT_WAIT_MAX_S, why
    _, why = _need(own, ids, session="chat-a-task-1", stay=True, parent="chat-a")
    assert why.get("wait") is None, why                # another worker's rows


def test_a_side_slot_holding_less_than_slot_0s_park_is_no_home():
    ids = tokens(11_500, seed=12)
    side = FakeRes(ids[:5_000])
    _, why = _need(FakeRes(tokens(12_000, seed=5), park=11_000), ids, side=side)
    assert why.get("wait") == "park" and why.get("home") is None and why["slot0_first"], why
    _, why = _need(FakeRes(tokens(12_000, seed=5)), ids, side=side)
    assert why.get("home") == 1, why                   # no park: the side slot is its home


def test_a_park_that_cannot_be_read_skips_no_parent_check():
    class Broken(FakeRes):
        def park_offer(self, ids, c, max_len, align=True):
            raise RuntimeError("the park changed while it was read")
    app.PARENTS.clear()
    app._note_parent("chat-a")
    try:
        _, why = _need(Broken(tokens(9_000, seed=13), park=1), tokens(9_500, seed=14),
                       session="chat-a")
        assert why.get("wait") == "parent", {k: v for k, v in why.items() if k != "ids"}
    finally:
        app.PARENTS.clear()


def test_a_returning_parent_tries_slot_0_for_as_long_as_a_re_read_would_take():
    """Live 2026-10-18 01:00: two parents' returns, their rows on the NVMe, took the side slot
    (`prefer_side`) and read 11,119 and 11,010 tokens again: the client's copy has none of the
    router's notes, so no entry matched it before the lock. A parent whose worker ran waits for
    slot 0, which reads its rows back, up to the time reading its prompt would take."""
    app.PARENTS.clear()
    worker = FakeRes(tokens(17_970, seed=7), owner="chat-a-task-1")
    ids = tokens(11_594, seed=8)
    _, why = _need(worker, ids, session="chat-a")
    assert why.get("wait") is None, why                # no worker ran: as before
    app._note_parent("chat-a")
    _, why = _need(worker, ids, session="chat-a")
    assert why.get("wait") == "parent" and why.get("slot0_first"), why
    assert app.SIDE_SLOT0_WAIT_S <= why["slot0_wait_s"] < app.PARENT_WAIT_MAX_S, why
    _, long_ = _need(worker, tokens(200_000, seed=9), session="chat-a")
    assert long_["slot0_wait_s"] == app.PARENT_WAIT_MAX_S, long_     # never minutes
    _, why = _need(worker, ids, session="chat-a-task-1", stay=True, parent="chat-a")
    assert why.get("wait") != "parent" and "slot0_wait_s" not in why, why
    app.PARENTS.clear()


def test_a_parent_waits_for_its_worker_for_parent_keep_s():
    app.PARENTS.clear()
    app._note_parent("chat-p", now=100.0)
    assert app._waiting_parent("chat-p", now=100.0 + app.PARENT_KEEP_S - 1)
    assert not app._waiting_parent("chat-p", now=100.0 + app.PARENT_KEEP_S)
    assert not app._waiting_parent(None) and not app._waiting_parent("chat-q", now=100.0)
    app._note_parent("chat-q", now=100.0 + app.PARENT_KEEP_S + 1)
    assert "chat-p" not in app.PARENTS                 # a stale entry goes


if __name__ == "__main__":
    passed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            with torch.no_grad():
                fn()
            print(f"  {name:90s} ok")
            passed += 1
    print(f"{passed} passed")
