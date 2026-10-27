"""Live sessions on the NVMe (engine/spill.py), on the random 4-layer model.

THE CLAIM. A conversation written to the disk and read back is the conversation that was in the
engine, byte for byte: every row of every position-indexed buffer, the recurrent state and the
convolution tails of the end anchor and of the saved grid anchors, the tokens, the `taint` line,
the hidden spans and the session. The next turn after a restore therefore resumes where it would
have resumed had the rows never left (the end anchor), and writes the same greedy tokens.

  1  a planned stop: the drain writes the resident conversation, a new process (a new engine on the
     same weights, a reopened spill) reads it back before its first request; rows, anchors and the
     end anchor equal the old engine's bit for bit, and the next turn resumes at the end with the
     same tokens as without the restart
  2  an eviction: another session's request that is not a guest writes the conversation to the disk
     first (it is not parked); the conversation's next turn reads it back (`spill`) and writes the
     same tokens as with no other request in between; the other session is written in turn when
     it is long enough
  3  incremental by content: written again a turn later, only the blocks that changed are written;
     read back over rows that hold the file's bytes, only the others are read
  4  the hidden spans: the handler's hook reads the conversation back before the replay, so a copy
     without the hidden phrase resumes at the end anchor
  5  a file that does not hold what was written fails the restore (its sums), the entry goes, and
     the prefill is a cold one; an entry of another fingerprint is swept at open
  6  the budget: least recently used first; a conversation that does not fit is not written
  7  the parked conversation is written at the drain with its head from the rows in place
  8 a conversation evicted by a request served from the store keeps its end anchor
     on the disk (the request used to drop it before the eviction wrote it); and written again
     with unchanged rows after its end anchor's buffers went, the end anchor already on the disk
     stays
  9  SRV-269: a chat whose client leaves the earlier answers' reasoning out (Open WebUI) comes
     back from the NVMe: the restore's offer aligns the copy as the resume after it does; a copy
     changed in more than the reasoning reads nothing back
  10 such a chat, read back once, is not read back again on its next turns: what RAM offers counts
     the resident end as the alignment serves it; a parked end is asked the same, and comes back
     before the replay through the server's hook
  11 a long prompt no entry serves although one shares a long head with it: one line says why
     (the end not reached and the alignment refused, no end, the read cost, a line too short)

Run: python tests/test_session_spill.py
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import test_forward_tree as T  # noqa: E402,F401  (the CPU environment before the engine imports)

import torch  # noqa: E402

from engine import cache, disk_cache, spill  # noqa: E402
from server import app  # noqa: E402
from test_cache import DEV, fresh, tokens  # noqa: E402
from test_end_anchor import CH, BIG, TOOL, TOOL2, _state, turn, _hidden_turn, PHRASE  # noqa: E402

HUGE = 10 ** 15


def _spill(d, res, *, fp="t", budget_gb=1.0, min_tokens=64, free=None):
    t = disk_cache.DiskTier(os.path.join(d, "tier"), chunk=CH, fingerprint="tier", min_tokens=CH,
                            free_bytes=lambda p: HUGE, settings={"mode": "off"}).open()
    sp = spill.SessionSpill(os.path.join(d, "tier", "sessions"), chunk=CH, fingerprint=fp,
                            tier=t, settings={"budget_gb": budget_gb, "min_tokens": min_tokens,
                                              "drain_min_tokens": CH},
                            free_bytes=free or (lambda p: HUGE), floor_bytes=0).open()
    res.spiller = sp
    app.STATE["spill"] = sp
    return sp, t


def _rows(eng, n):
    return [t.narrow(ax, 0, n).clone() for t, ax in cache._kv_views(eng, None)]


def _same_rows(a, b):
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert torch.equal(x, y)


def _restart(d, fp="t", **kw):
    """A new process: a new engine on the same weights, a new resident prefix, the spill reopened
    from its directory; the drained conversation read back as `spill_startup` does."""
    res = _state(**kw)
    sp, t = _spill(d, res, fp=fp)
    app.spill_startup()
    return res, sp, t


# ------------------------------------------------------------------ 1. a planned stop

def test_a_drained_conversation_comes_back_after_a_restart_bit_for_bit():
    d = tempfile.mkdtemp(prefix="qse-spill-")
    try:
        prompt = tokens(150, seed=70)
        # the reference: no restart between the two turns
        _state()
        out1, _, _ = turn(prompt, 60)
        out2_ref, pre_ref, _ = turn(prompt + out1 + TOOL, 24)
        assert pre_ref["kind"] == "end"
        # the run: turn 1, the drain, a restart, turn 2
        res = _state()
        sp, t = _spill(d, res)
        out1b, _, ctx1 = turn(prompt, 60)
        assert out1b == out1
        e, taint = res.end["at"], res.taint
        eng0 = app.STATE["engine"]
        rows0 = _rows(eng0, e)
        end0 = tuple(x.clone() for x in res._end_bufs)
        anchors0 = {b: (S.clone(), c.clone()) for b, (S, c) in res.anchors.items()}
        app.spill_drain()
        assert sp.stats["saves"] == 1 and sp.drained() is not None
        t.close()
        res2, sp2, t2 = _restart(d)
        eng1 = app.STATE["engine"]
        assert eng1 is not eng0
        assert res2.valid == e and res2.end["at"] == e and res2.taint == taint
        assert res2.tokens == res.tokens[:e]
        _same_rows(_rows(eng1, e), rows0)
        assert torch.equal(res2._end_bufs[0], end0[0]) and torch.equal(res2._end_bufs[1], end0[1])
        kept = sorted(anchors0)[-4:]
        assert sorted(res2.anchors) == kept
        for b in kept:
            assert torch.equal(res2.anchors[b][0], anchors0[b][0])
            assert torch.equal(res2.anchors[b][1], anchors0[b][1])
        out2, pre2, _ = turn(prompt + out1 + TOOL, 24)
        assert pre2["kind"] == "end" and pre2["reused"] == e, pre2
        assert out2 == out2_ref
        t2.close()
        return f"{e} tokens drained and read back bit for bit; turn 2 resumed at the end, same 24 tokens"
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ------------------------------------------------------------------ 2. an eviction

def _evict_run(with_spill: bool, other_len=200, d=None):
    res = _state(park_bytes=0, guest_max=40)
    sp = t = None
    if with_spill:
        sp, t = _spill(d, res)
    prompt = tokens(220, seed=71)
    out1, _, _ = turn(prompt, 40, session="A")
    e = res.end["at"]
    other = tokens(other_len, seed=72)
    turn(other, 8, session="B")                     # not a guest: longer than guest_max
    out2, pre2, _ = turn(prompt + out1 + TOOL, 16, session="A")
    return res, sp, t, e, out2, pre2


def test_another_sessions_request_writes_the_conversation_first_and_it_comes_back():
    d = tempfile.mkdtemp(prefix="qse-spill-")
    try:
        _, _, _, e0, ref, pre_ref = _evict_run(False)
        assert pre_ref["reused"] < e0, "without the spill the conversation is re-read"
        res, sp, t, e, out2, pre2 = _evict_run(True, d=d)
        assert e == e0
        assert res.stats["spills"] >= 1 and sp.stats["saves"] >= 2, (res.stats, sp.stats)
        assert pre2["kind"] == "spill" and pre2["reused"] == e, pre2
        assert out2 == ref
        assert res.stats["parks"] == 0
        # B (208 tokens, over min_tokens) was written in turn when A came back
        owners = sorted(x.man["owner"] for x in sp.entries.values())
        assert owners == ["A", "B"], owners
        t.close()
        return f"evicted at {e}, read back by its next turn ({pre2['reused']} reused, was {pre_ref['reused']})"
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_a_session_snapshot_does_not_drop_a_long_conversation():
    d = tempfile.mkdtemp(prefix="qse-spill-")
    try:
        store = cache.StateStore(BIG, chunk=CH)
        res = _state(store=store)
        sp, t = _spill(d, res)
        prompt = tokens(220, seed=73)
        out1, _, _ = turn(prompt, 30, session="A")
        e = res.end["at"]
        res._spill(app.STATE["engine"], None, "evict")
        assert sp.stats["saves"] == 1
        res.clear()                                   # what a session snapshot's restore does
        out2, pre2, _ = turn(prompt + out1 + TOOL, 8, session="A")
        assert pre2["kind"] == "spill" and pre2["reused"] == e
        t.close()
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ------------------------------------------------------------------ 3. incremental

def test_written_again_only_the_changed_blocks_are_written_and_read():
    d = tempfile.mkdtemp(prefix="qse-spill-")
    try:
        res = _state()
        sp, t = _spill(d, res)
        prompt = tokens(200, seed=74)
        out1, _, _ = turn(prompt, 30, session="A")
        eng = app.STATE["engine"]
        first = sp.save_resident(res, eng, None, "evict")
        n1 = first["blocks_written"]
        out2, _, _ = turn(prompt + out1 + TOOL, 20, session="A")
        second = sp.save_resident(res, eng, None, "evict")
        assert second["blocks_kept"] >= (len(prompt) + len(out1)) // CH - 2, second
        assert second["blocks_written"] < n1, (first, second)
        # read back over rows that hold the same bytes: nothing of them is read
        info = sp.restore(res, eng, None, sp.entries[second["eid"]])
        assert info["blocks_read"] == 0 and info["blocks_in_place"] == len(
            sp.entries[second["eid"]].man["blocks"]), info
        t.close()
        return f"{n1} blocks, then {second['blocks_written']} written and {second['blocks_kept']} kept"
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ------------------------------------------------------------------ 4. the hidden spans

def test_the_handlers_hook_reads_it_back_before_the_replay():
    d = tempfile.mkdtemp(prefix="qse-spill-")
    try:
        res = _state()
        sp, t = _spill(d, res)
        prompt = tokens(100, seed=62)
        out, ctx = _hidden_turn(prompt, session="A")
        e = res.end["at"]
        copy = prompt + out + TOOL                    # the client's copy: no phrase
        sp.save_resident(res, app.STATE["engine"], None, "evict")
        res.clear()
        for tt, ax in cache._kv_views(app.STATE["engine"], None):
            tt.fill_(0)
        app._spill_before_replay(copy)
        assert res.end is not None and res.end["at"] == e and res.end["hidden"]
        ids, spans = res.replay(copy)
        assert spans and ids == ctx + TOOL
        _, pre, _ = turn(ids, 8, session="A", hidden=spans)
        assert pre["kind"] == "spill" and pre["reused"] == e, pre
        t.close()
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ------------------------------------------------------------------ 5. a bad file

def test_a_file_that_fails_its_sums_fails_the_restore_and_the_prefill_is_cold():
    d = tempfile.mkdtemp(prefix="qse-spill-")
    try:
        res = _state()
        sp, t = _spill(d, res)
        prompt = tokens(200, seed=75)
        out1, _, _ = turn(prompt, 20, session="A")
        info = sp.save_resident(res, app.STATE["engine"], None, "evict")
        e = sp.entries[info["eid"]]
        blk = os.path.join(e.path, "r0000032.blk")
        with open(blk, "r+b") as f:
            f.seek(spill.ALIGN + 100)
            b = f.read(1)
            f.seek(spill.ALIGN + 100)
            f.write(bytes([b[0] ^ 0xFF]))
        res.clear()
        _, pre, _ = turn(prompt + out1 + TOOL, 4, session="A")
        assert pre["kind"] is None and pre["reused"] == 0, pre
        # an uncompressed block fails its sums; a compressed one its sums, or its frame (0.4.3)
        failed = sp.stats["failed"]
        assert failed.get("sums", 0) + failed.get("read", 0) == 1 and not sp.entries, (
            sp.stats, sp.events)
        t.close()
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_another_fingerprint_is_swept_and_an_incomplete_entry_too():
    d = tempfile.mkdtemp(prefix="qse-spill-")
    try:
        res = _state()
        sp, t = _spill(d, res)
        out1, _, _ = turn(tokens(200, seed=76), 20, session="A")
        sp.save_resident(res, app.STATE["engine"], None, "drain")
        os.makedirs(os.path.join(sp.root, "deadbeef"), mode=0o700)   # a crash before its manifest
        t.close()
        res2 = _state()
        sp2, t2 = _spill(d, res2, fp="other")
        assert not sp2.entries and sp2.stats["swept"] == 2
        assert os.listdir(sp2.root) == []
        t2.close()
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ------------------------------------------------------------------ 6. the budget

def test_least_recently_used_goes_first_and_what_does_not_fit_is_not_written():
    d = tempfile.mkdtemp(prefix="qse-spill-")
    try:
        res = _state()
        eng = app.STATE["engine"]
        sp, t = _spill(d, res)
        out_a, _, _ = turn(tokens(200, seed=77), 20, session="A")
        first = sp.save_resident(res, eng, None, "evict")
        sp.cfg["budget_gb"] = 1.5 * first["bytes"] / 1e9
        out_b, _, _ = turn(tokens(200, seed=78), 20, session="B")
        sp.save_resident(res, eng, None, "evict")
        owners = sorted(x.man["owner"] for x in sp.entries.values())
        assert owners == ["B"], owners
        assert sp.stats["evicted"].get("budget") == 1
        sp.cfg["budget_gb"] = 1e-6
        _, _, _ = turn(tokens(200, seed=79), 20, session="C")
        before = sp.stats["refused"].get("room", 0)        # B's own eviction was refused too
        assert sp.save_resident(res, eng, None, "evict") is None
        assert sp.stats["refused"].get("room") == before + 1
        assert all(x.man["owner"] != "C" for x in sp.entries.values())
        t.close()
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ------------------------------------------------------------------ 7. the park

def test_the_parked_conversation_is_written_with_its_head_from_the_rows_in_place():
    d = tempfile.mkdtemp(prefix="qse-spill-")
    try:
        res = _state(park_bytes=BIG, guest_max=10)
        sp, t = _spill(d, res, min_tokens=10 ** 6)     # evictions park in RAM here
        head = tokens(64, seed=80)
        a = head + tokens(150, seed=81)
        out_a, _, _ = turn(a, 10, session="A")
        b = head + tokens(140, seed=82)
        turn(b, 10, session="B")
        assert res.park_rows and res.park_owner == "A"
        eng = app.STATE["engine"]
        top = res.park_base + res.park_rows
        want = []
        for (tt, ax), buf in zip(cache._kv_views(eng, None), res._park_bufs):
            want.append(torch.cat([tt.narrow(ax, 0, res.park_base),
                                   buf.narrow(ax, 0, res.park_rows)], dim=ax).clone())
        info = sp.save_park(res, eng, None, "drain")
        assert info and info["tokens"] == top
        res.clear()
        res.shed_park()
        for tt, ax in cache._kv_views(eng, None):
            tt.fill_(0)
        got = sp.restore(res, eng, None, sp.entries[info["eid"]])
        assert got and res.valid == top and res.owner == "A"
        for (tt, ax), w in zip(cache._kv_views(eng, None), want):
            assert torch.equal(tt.narrow(ax, 0, top), w)
        _, pre, _ = turn(a + out_a + TOOL2, 4, session="A")
        assert pre["kind"] in ("resident", "end"), pre
        t.close()
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _three_sessions(d=None, park_spill="on"):
    """A, B, C in turn, all under the spill's eviction length (they park in RAM), then A again.
    The park holds one conversation, so C's prefill gives A up."""
    old = getattr(cache, "SPILL_PARK", None)
    cache.SPILL_PARK = park_spill
    try:
        res = _state(park_bytes=BIG, guest_max=10)
        sp = t = None
        if d is not None:
            sp, t = _spill(d, res, min_tokens=10 ** 6)
        head = tokens(64, seed=80)
        a = head + tokens(150, seed=81)
        out_a, _, _ = turn(a, 10, session="A")
        turn(head + tokens(140, seed=82), 10, session="B")
        assert res.park_owner == "A"
        turn(head + tokens(130, seed=83), 10, session="C")
        assert res.park_owner == "B"
        out2, pre, _ = turn(a + out_a + TOOL2, 6, session="A")
        return res, sp, t, pre, out2
    finally:
        cache.SPILL_PARK = old


def test_fx10_a_parked_conversation_a_new_park_gives_up_goes_to_the_nvme_first():
    """FX10 (the per-entry cap, item 3): a conversation over the state store's cap and under the
    spill's eviction length lived only in the RAM park, which holds one; a third session dropped
    it and its next turn read everything past the shared head again."""
    _, _, _, ref, out_ref = _three_sessions(None)
    assert ref["reused"] <= 64, ref                    # the head only: A was dropped
    d = tempfile.mkdtemp(prefix="qse-spill-")
    try:
        res, sp, t, pre, out2 = _three_sessions(d)
        assert pre["kind"] == "spill" and pre["reused"] > 64 + CH, pre
        # ENG-371: A's restore parks the conversation in place first, so the park's B goes to the
        # NVMe in turn: A's entry, and B's beside it
        assert res.stats.get("park_spills") >= 1, res.stats
        assert "A" in [x.man["owner"] for x in sp.entries.values()]
        assert out2 == out_ref, "the same tokens as the re-read"
        t.close()
        return f"A came back from the NVMe at {pre['reused']} (was {ref['reused']}), same tokens"
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_fx10_park_spill_off_gives_it_up_as_before():
    d = tempfile.mkdtemp(prefix="qse-spill-")
    try:
        res, sp, t, pre, _ = _three_sessions(d, park_spill="off")
        assert res.stats.get("park_spills", 0) == 0 and not sp.entries, res.stats
        assert pre["reused"] <= 64, pre
        t.close()
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _shared_head_run(d, sessions):
    """A, B, C in turn with one long shared head (6 chunks, over the 4 a save matches on), as Open
    WebUI's chats share its system prompt and tools; none is a guest, each written in turn; then A."""
    res = _state(park_bytes=0, guest_max=40)
    sp, t = _spill(d, res)
    head = tokens(6 * CH, seed=90)
    a = head + tokens(140, seed=91)
    out_a, _, _ = turn(a, 40, session=sessions[0])
    e = res.end["at"]
    turn(head + tokens(130, seed=92), 8, session=sessions[1])      # A written to the NVMe
    turn(head + tokens(120, seed=93), 8, session=sessions[2])      # B written in turn
    out2, pre2, _ = turn(a + out_a + TOOL, 16, session=sessions[0])
    return res, sp, t, e, pre2, out2


def test_fx10b_chats_without_an_owner_sharing_a_long_head_keep_their_own_entries():
    """FX10b (SRV-257 live, 0.4.1-test): Open WebUI chats carry no owner, and every one starts with
    the same ~6k-token head. A save reused any entry that shared 4 chunks with it, so B's save wrote
    over A's entry and A came back cold. With owners (opencode sessions) it never showed."""
    d = tempfile.mkdtemp(prefix="qse-spill-")
    try:
        _, _, t, e0, ref, out_ref = _shared_head_run(d, ("A", "B", "C"))
        assert ref["kind"] == "spill" and ref["reused"] == e0, ref
        t.close()
    finally:
        shutil.rmtree(d, ignore_errors=True)
    d = tempfile.mkdtemp(prefix="qse-spill-")
    try:
        res, sp, t, e, pre, out2 = _shared_head_run(d, (None, None, None))
        assert e == e0
        assert pre["kind"] == "spill" and pre["reused"] == e, pre
        assert out2 == out_ref, "the same tokens as with owners"
        assert len(sp.entries) >= 2 and all(x.man["owner"] is None for x in sp.entries.values())
        t.close()
        return f"A came back from its own entry at {pre['reused']} beside {len(sp.entries) - 1} other(s)"
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_fx10b_the_same_conversation_written_again_keeps_its_entry():
    """The guard above keeps incremental saves: a conversation that grew matches its own entry."""
    sp = spill.SessionSpill.__new__(spill.SessionSpill)
    import threading
    sp.lock, sp.chunk, sp.clock = threading.Lock(), CH, (lambda: 0)
    head = tokens(6 * CH, seed=90)
    a = head + tokens(140, seed=91)
    sp.entries = {"aaaa": spill.Entry("/x", {"eid": "aaaa", "owner": None}, a)}
    assert sp._eid_for(None, a + tokens(50, seed=94)) == "aaaa"                 # grown
    assert sp._eid_for(None, a[:-2 * CH] + tokens(70, seed=95)) == "aaaa"       # a changed tail
    assert sp._eid_for(None, head + tokens(130, seed=92)) != "aaaa"             # another chat


def test_srv285_open_webui_chats_own_their_entries():
    """SRV-285 (UX3, 2026-10-08): the caches' owner key read the body's ids and opencode's
    x-session-id, never Open WebUI's X-OpenWebUI-Chat-Id, so every Open WebUI entry had no owner.
    Two chats sharing a pasted 12k document: B's save took A's entry (A's tail past the shared head
    was under 4 chunks) and A came back from a park that wrote nothing. The chat id is the owner; an
    owned save never takes another owner's entry, and an owner-less one never takes an owned one."""
    class H(dict):
        def get(self, k, d=None):
            return {kk.lower(): v for kk, v in self.items()}.get(k.lower(), d)
    assert app._session_header(H({"X-OpenWebUI-Chat-Id": "4f1c-a"})) == "c:4f1c-a"
    assert app._session_header(H({"x-session-id": "ses_1", "X-OpenWebUI-Chat-Id": "x"})) == "ses_1"
    assert app._session_header(H({})) is None
    body = {"metadata": {"chat_id": "ignored"}, "conversation_id": "body-id"}
    assert (app.conversation_id(body, H({"X-OpenWebUI-Chat-Id": "x"}))
            or app._session_header(H({"X-OpenWebUI-Chat-Id": "x"}))) == "body-id"
    sp = spill.SessionSpill.__new__(spill.SessionSpill)
    import threading
    sp.lock, sp.chunk, sp.clock = threading.Lock(), CH, (lambda: 0)
    doc = tokens(12 * CH, seed=96)                               # the pasted document, both chats
    a = doc + tokens(2 * CH, seed=97)
    b = doc + tokens(2 * CH + 5, seed=98)
    # before: no owner, and B's tail is within 4 chunks of A's entry -> B took A's entry
    sp.entries = {"aaaa": spill.Entry("/x", {"eid": "aaaa", "owner": None}, a)}
    assert sp._eid_for(None, b) == "aaaa", "the hazard an owner-less chat has"
    # with the chat ids: each its own
    sp.entries = {"aaaa": spill.Entry("/x", {"eid": "aaaa", "owner": "c:A"}, a)}
    assert sp._eid_for("c:B", b) != "aaaa"
    assert sp._eid_for("c:A", a + tokens(40, seed=99)) == "aaaa"
    assert sp._eid_for(None, b) != "aaaa", "an owner-less save never takes an owned entry"


# ------------------------------------------------------------------ its own I/O path

def test_its_own_lock_and_buffers_the_tier_waits_and_unchanged_anchors_stay():
    """A save takes neither the tier's lock nor its chunk slots (the tier's writer
    holding its lock does not stop it), the tier's writer waits while it runs, a block's sums over
    every buffer are `block_sums`' integers, and a second save of an unchanged conversation writes
    no anchor again."""
    import threading
    d = tempfile.mkdtemp(prefix="qse-spill-")
    try:
        res = _state()
        sp, t = _spill(d, res)
        prompt = tokens(200, seed=76)
        turn(prompt, 30, session="A")
        eng = app.STATE["engine"]
        views = cache._kv_views(eng, None)
        assert spill.block_sums_views(views, 16, 16) == [spill.block_sums(x, ax, 16, 16)
                                                         for x, ax in views]
        seen = []
        real = spill.SessionSpill._tier_hold

        def hold(self, dd):
            real(self, dd)
            seen.append(t.spill_active)
        spill.SessionSpill._tier_hold = hold
        held = threading.Event()
        try:
            with t._io:                      # the tier's writer in the middle of a file
                done = []
                th = threading.Thread(target=lambda: done.append(
                    sp.save_resident(res, eng, None, "evict")))
                th.start()
                th.join(30)
                held.set()
            assert done and done[0] is not None, "the save waited for the tier's lock"
        finally:
            spill.SessionSpill._tier_hold = real
        assert seen == [1, 0] and t.spill_active == 0, seen
        assert not t.buf_chunk, "the tier's chunk slots were not used"
        first = done[0]
        again = sp.save_resident(res, eng, None, "drain")
        assert again["blocks_written"] == 0 and again["bytes_written"] == 0, again
        assert again["bytes"] == first["bytes"], (first, again)
        info = sp.restore(res, eng, None, sp.entries[again["eid"]])
        assert info is not None and info["end"], info
        t.close()
        return f"{first['bytes_written']} bytes, then {again['bytes_written']}"
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ------------------------------------------------------------------ 8. the end anchor stays

def _store_evict_run(with_other: bool, d):
    """A, then B (not a guest: A goes to the disk), A's turn 2 (read back), then, with
    `with_other`, C, which shares B's head and starts from the store's grid entry (kind
    prefix) and evicts A again; then A's turn 3. The live case (13:34): the drained
    conversation read back at the start, then a 104k request from the disk tier evicted it."""
    store = cache.StateStore(BIG, chunk=CH)
    res = _state(store=store, park_bytes=0, guest_max=40)
    sp, t = _spill(d, res)
    prompt = tokens(220, seed=81)
    out1, _, _ = turn(prompt, 40, session="A")
    other = tokens(50, seed=82)                     # below min_tokens: B is never on the disk
    turn(other, 8, session="B")
    ctx2 = prompt + out1 + TOOL
    out2, pre2, _ = turn(ctx2, 16, session="A")
    assert pre2["kind"] == "spill", pre2
    e2 = res.end["at"]
    pre3 = None
    if with_other:
        _, pre3, _ = turn(other[:40] + tokens(20, seed=83), 8, session="C")
    out4, pre4, _ = turn(ctx2 + out2 + TOOL2, 12, session="A")
    return res, sp, t, e2, pre3, out4, pre4


def test_an_eviction_by_a_request_from_the_store_keeps_the_end_anchor():
    d = tempfile.mkdtemp(prefix="qse-spill-")
    d2 = tempfile.mkdtemp(prefix="qse-spill-")
    try:
        _, _, t0, e_ref, _, ref, pre_ref = _store_evict_run(False, d2)
        t0.close()
        assert pre_ref["kind"] == "end" and pre_ref["reused"] == e_ref, pre_ref
        res, sp, t, e2, pre3, out4, pre4 = _store_evict_run(True, d)
        assert e2 == e_ref
        assert pre3["kind"] == "prefix", pre3
        a = [x for x in sp.entries.values() if x.man.get("owner") == "A"]
        assert len(a) == 1 and a[0].man["end"] is not None, a[0].man.get("end") if a else sp.entries
        assert pre4["kind"] == "spill" and pre4["reused"] == e2, pre4
        assert out4 == ref
        t.close()
        return f"C started from the store at {pre3['reused']}; A came back at its end, {e2}"
    finally:
        shutil.rmtree(d, ignore_errors=True)
        shutil.rmtree(d2, ignore_errors=True)


def test_written_again_without_its_end_buffers_the_end_on_the_disk_stays():
    d = tempfile.mkdtemp(prefix="qse-spill-")
    try:
        res = _state()
        sp, t = _spill(d, res)
        prompt = tokens(200, seed=84)
        out1, _, _ = turn(prompt, 30, session="A")
        eng = app.STATE["engine"]
        e = res.end["at"]
        first = sp.save_resident(res, eng, None, "drain")
        assert first["end"], first
        end0 = tuple(x.clone() for x in res._end_bufs)
        res.shed_end()                       # a cache level gave the end's buffers back
        assert res.end is None
        again = sp.save_resident(res, eng, None, "evict")
        assert again["end"] and again.get("end_carried") and again["bytes_written"] == 0, again
        man = sp.entries[again["eid"]].man
        assert man["end"]["at"] == e and any(f["file"] == "end.anc" for f in man["anchor_files"])
        info = sp.restore(res, eng, None, sp.entries[again["eid"]])
        assert info is not None and info["end"] and res.end["at"] == e, info
        assert torch.equal(res._end_bufs[0], end0[0]) and torch.equal(res._end_bufs[1], end0[1])
        # rows that moved: the old end is not carried
        out2, _, _ = turn(prompt + out1 + TOOL, 10, session="A")
        res.shed_end()
        moved = sp.save_resident(res, eng, None, "evict")
        assert moved is not None and not moved["end"] and not moved.get("end_carried"), moved
        assert sp.entries[moved["eid"]].man["end"] is None
        t.close()
        return f"end at {e} kept on the disk through a save without its buffers"
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ------------------------------------------------------------------ 9. SRV-269

def test_srv269_a_chat_without_its_reasoning_comes_back_from_the_nvme():
    """SRV-269 (2026-10-07, live): an Open WebUI chat (130,381 client tokens, 182,053 rows with
    the reasoning Open WebUI does not send back) went to the NVMe while two other chats ran, and
    back in it the whole chat was read again cold (about 4 minutes): the restore's offer replayed
    the hidden spans only, the copy parted at the first answer's reasoning, and no entry was
    offered. The offer now aligns the copy as the resume after the restore does (SRV-239)."""
    import test_end_align as EA
    from server import diskcache
    d = tempfile.mkdtemp(prefix="qse-spill-")
    try:
        prompt = tokens(99, seed=93) + [EA.NL]
        res, al, out, k = EA._thinking(prompt)
        sp, t = _spill(d, res)
        diskcache.wire_end_align(sp, al)
        e = res.end["at"]
        rows_before = _rows(app.STATE["engine"], e)
        # the client's copy: the answer without its reasoning (`<think>\n\n</think>\n\n`), and a
        # new message shorter than the reasoning, so the copy is shorter than the rows up to the
        # end anchor (live: 130,381 tokens against an end at 182,053)
        copy = prompt[:-1] + [EA.SP] + out[k:] + TOOL[:4]
        assert len(copy) <= res.end["at"], (len(copy), res.end["at"])
        assert res.replay(copy)[1] == [], "the hidden spans alone do not replay this copy"
        sp.save_resident(res, app.STATE["engine"], None, "evict")
        assert sp.entries and next(iter(sp.entries.values())).man["extra"]["end_regions"]
        res.clear()
        for tt, ax in cache._kv_views(app.STATE["engine"], None):
            tt.fill_(0)
        # a copy that differs in more than the reasoning is not offered the end anchor (a grid
        # anchor of the head it shares at most)
        j = next(i for i in range(k + 2, len(out) - 2) if out[i] not in (EA.NL, EA.SP))
        other = next(x for x in range(1, 95) if x not in (out[j], out[k]))
        changed = prompt[:-1] + [EA.SP] + out[k:j] + [other] + out[j + 1:] + TOOL[:4]
        got = sp.candidate(res, changed, len(changed) - 1, replay=True)
        assert got is None or got[1] < len(prompt), got and got[1]
        got = sp.candidate(res, copy, len(copy) - 1, replay=True)
        assert got is not None and got[1] == e, got and got[1]
        # the chat's own copy: read back from the NVMe before the replay
        app._spill_before_replay(copy)
        assert sp.stats["restores"] == 1, sp.stats
        assert res.end is not None and res.end["at"] == e
        _same_rows(_rows(app.STATE["engine"], e), rows_before)
        ids, spans, aligned = app._end_replay(res, copy)
        assert aligned > 0 and ids == res.tokens[:e] + out[e - len(prompt):] + TOOL[:4]
        got, pre, _ = turn(ids, 8)
        assert pre["reused"] == e and pre["kind"] in ("spill", "end"), pre
        t.close()
        assert got == EA._cold(ids, 8), "the end anchor's gate: the same greedy tokens as cold"
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_srv269_the_switch_save_never_evicts_the_entry_it_reads_next():
    """SRV-269 (found in FX11's live check, 2026-10-07): with the budget full, the restore's
    switch save of the resident conversation swept the least recently used entry -- the one the
    restore was about to read -- and the read then failed into a cold prefill. The entry asked for
    is kept; the resident conversation is not written when it does not fit."""
    d = tempfile.mkdtemp(prefix="qse-spill-")
    try:
        res = _state()
        sp, t = _spill(d, res)
        a = tokens(150, seed=95)
        out_a, _, _ = turn(a, 24, session="A")
        sp.save_resident(res, app.STATE["engine"], None, "evict")
        (ea,) = list(sp.entries.values())
        res.clear()
        b = tokens(140, seed=96)
        turn(b, 24, session="B")
        assert res.valid >= sp.min_tokens
        # room for A alone: B's switch save can only fit by evicting A
        sp.cfg["budget_gb"] = (ea.bytes + ea.bytes // 2) / spill.GB
        assert sp.budget < ea.bytes * 2
        copy = a + out_a + TOOL
        got = sp.maybe_restore(res, app.STATE["engine"], None, copy, len(copy) - 1)
        assert got is not None and got["eid"] == ea.eid, (got, sp.stats["evicted"])
        assert ea.eid in sp.entries and sp.stats["refused"].get("room", 0) >= 1, sp.stats
        assert res.tokens[:res.valid] == ea.tokens[:res.valid] and res.valid == ea.valid
        t.close()
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ------------------------------------------------------------------ 10. not read back again

def test_fx16_a_restored_chat_continued_without_its_reasoning_is_not_restored_again():
    """Live 0.4.7 (2026-10-08): after an Open WebUI chat came back from the NVMe, every later turn
    restored the same entry again in place (`[spill] restore blocks_read=0 blocks_in_place=87
    ms=398`, ten times in ten minutes): `candidate` asked the alignment of the entries, never of
    the resident's own end, so the copy (the reasoning left out) looked like RAM held nothing of
    it. RAM's offer now counts the aligned resident end; the entry is read only when it beats it."""
    import test_end_align as EA
    from server import diskcache
    d = tempfile.mkdtemp(prefix="qse-spill-")
    try:
        prompt = tokens(99, seed=93) + [EA.NL]
        res, al, out, k = EA._thinking(prompt)
        sp, t = _spill(d, res)
        diskcache.wire_end_align(sp, al)
        e = res.end["at"]
        copy = prompt[:-1] + [EA.SP] + out[k:] + TOOL
        sp.save_resident(res, app.STATE["engine"], None, "evict")
        res.clear()
        for tt, ax in cache._kv_views(app.STATE["engine"], None):
            tt.fill_(0)
        app._spill_before_replay(copy)
        assert sp.stats["restores"] == 1, sp.stats
        ids, _, aligned = app._end_replay(res, copy)
        assert aligned
        out2, pre, _ = turn(ids, 10)
        assert pre["reused"] == e and pre["kind"] in ("spill", "end"), pre
        assert out[k] not in out2, "this turn's answer has no reasoning of its own"
        e2 = res.end["at"]
        # the turn after, as Open WebUI sends it: still without the first answer's reasoning
        copy3 = copy + out2 + TOOL2
        events = len(sp.events)
        app._spill_before_replay(copy3)
        assert sp.stats["restores"] == 1, ("restored again", sp.stats["last_restore"])
        assert not any(ev["kind"] == "restore" for ev in sp.events[events:])
        assert res.end["at"] == e2
        ids3, _, a3 = app._end_replay(res, copy3)
        assert a3 and ids3[:e2] == res.tokens[:e2]
        out3, pre3, _ = turn(ids3, 8)
        assert pre3["kind"] == "end" and pre3["reused"] == e2, pre3
        t.close()
        assert out3 == EA._cold(ids3, 8), "the end anchor's gate: the same greedy tokens as cold"
        return f"read back once at {e}; the next turn resumed at its end {e2} with no restore"
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_fx16_a_parked_end_comes_back_through_the_servers_hook_not_an_entry():
    """The server's hook before its replay (`_spill_before_replay` -> `maybe_restore`): when no
    entry beats the parked end, the park comes back before the replay (engine/cache.py
    `unpark_end`), so the copy without its reasoning resumes at the end; the parked end is
    written to the NVMe with its regions when a third chat gives the park up."""
    import test_end_align as EA
    from server import diskcache, endalign
    d = tempfile.mkdtemp(prefix="qse-spill-")
    try:
        head = tokens(4 * CH, seed=110)
        a = head + tokens(99, seed=111) + [EA.NL]
        _state()
        out0, _, _ = turn(a, 60, session="A")
        k = next(i for i in range(12, 30) if out0[i] not in out0[:i] and out0[i] not in a
                 and out0[i] not in (EA.NL, EA.SP))
        res = _state(park_bytes=BIG, guest_max=10)
        al = app.STATE["end_align"] = endalign.EndAlign(EA.tb_of({out0[k]: endalign.THINK_END}))
        sp, t = _spill(d, res, min_tokens=10 ** 6)          # evictions park in RAM here
        diskcache.wire_end_align(sp, al, res)
        other = tokens(150, seed=102)                        # an entry of another chat on the disk
        turn(other, 8, session="X")
        sp.save_resident(res, app.STATE["engine"], None, "evict")
        out, _, _ = turn(a, 60, session="A")
        e = res.end["at"]
        turn(head + tokens(140, seed=103), 8, session="B")
        assert res.park_owner == "A" and res.park_end["at"] == e
        copy = a[:-1] + [EA.SP] + out[k:] + TOOL
        app._spill_before_replay(copy)
        assert sp.stats["restores"] == 0 and res.owner == "A" and res.end["at"] == e, res.report()
        ids, _, aligned = app._end_replay(res, copy)
        assert aligned
        out2, pre, _ = turn(ids, 8, session="A")
        assert pre["kind"] == "end" and pre["reused"] == e, pre
        assert res.stats["park_tokens_reused"] == e
        # B is parked now, with its end; a third chat gives it up: to the NVMe, its end with it
        e_b = res.park_end["at"]
        turn(head + tokens(130, seed=104), 8, session="C")
        b_entry = [x for x in sp.entries.values() if x.man["owner"] == "B"]
        assert len(b_entry) == 1 and b_entry[0].man["end"]["at"] == e_b, sp.report()
        assert b_entry[0].man["extra"]["end_regions"], b_entry[0].man["extra"]
        t.close()
        assert out2 == EA._cold(ids, 8), "the end anchor's gate: the same greedy tokens as cold"
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ------------------------------------------------------------------ 11. why no entry served

def _unserved_run(d, copy_of, *, end=True, read_bps=None):
    """A, written to the NVMe; the rows in place cleared; then `copy_of(prompt, out)` through the
    server's hook. Returns (spill, the `no entry served` lines, the tier)."""
    import contextlib
    import io
    res = _state(park_bytes=0, guest_max=40)
    sp, t = _spill(d, res)
    prompt = tokens(220, seed=105)
    out, _, _ = turn(prompt, 40, session="A")
    if not end:
        res.shed_end()
    sp.save_resident(res, app.STATE["engine"], None, "evict")
    res.clear()
    for tt, ax in cache._kv_views(app.STATE["engine"], None):
        tt.fill_(0)
    if read_bps is not None:
        sp.read_bps = read_bps
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        app._spill_before_replay(copy_of(prompt, out))
    lines = [x for x in buf.getvalue().splitlines() if x.startswith("[spill] no entry served:")]
    return sp, lines, t


def _changed_at_100(prompt, out):
    """Ten tokens of the user message changed (not whitespace, not reasoning)."""
    ch = [(x % 90) + 1 for x in tokens(10, seed=106)]
    ch[0] = ch[0] % 90 + 1 if ch[0] == prompt[100] else ch[0]
    assert ch[0] != prompt[100]
    return prompt[:100] + ch + prompt[110:] + out + TOOL


def test_fx16_no_entry_served_says_why_in_one_line():
    d = tempfile.mkdtemp(prefix="qse-spill-")
    try:
        sp, lines, t = _unserved_run(d, _changed_at_100)
        assert len(lines) == 1, lines
        eid = next(iter(sp.entries))
        assert lines[0] == (f"[spill] no entry served: eid={eid} owner=A tokens={sp.entries[eid].valid} "
                            "shares=100 offer=0 ram=0 why=end_no_align"), lines[0]
        assert sp.stats["restores"] == 0 and not any(ev["kind"] == "restore" for ev in sp.events)
        assert sp.stats["refused"].get("end_no_align") == 1, sp.stats["refused"]
        t.close()
    finally:
        shutil.rmtree(d, ignore_errors=True)
    d = tempfile.mkdtemp(prefix="qse-spill-")
    try:
        sp, lines, t = _unserved_run(d, _changed_at_100, end=False)    # an entry without an end
        assert len(lines) == 1 and lines[0].endswith("why=anchors_below"), lines
        t.close()
    finally:
        shutil.rmtree(d, ignore_errors=True)
    d = tempfile.mkdtemp(prefix="qse-spill-")
    try:
        # the whole conversation, but a read that costs more than the re-read
        sp, lines, t = _unserved_run(d, lambda p, o: p + o + TOOL, read_bps=1.0)
        assert len(lines) == 1 and lines[0].endswith("why=slow"), lines
        assert sp.stats["refused"].get("slow") == 1 and sp.stats["restores"] == 0
        t.close()
    finally:
        shutil.rmtree(d, ignore_errors=True)
    d = tempfile.mkdtemp(prefix="qse-spill-")
    try:
        # served: no line
        sp, lines, t = _unserved_run(d, lambda p, o: p + o + TOOL)
        assert lines == [] and sp.stats["restores"] == 1, (lines, sp.stats)
        t.close()
        return "end_no_align, anchors_below and slow each one line; a served copy none"
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _main():
    import inspect
    fails = 0
    for name, fn in sorted(inspect.getmembers(sys.modules[__name__], inspect.isfunction),
                           key=lambda x: x[1].__code__.co_firstlineno):
        if not name.startswith("test_"):
            continue
        try:
            msg = fn()
            print(f"PASS {name}" + (f": {msg}" if msg else ""))
        except Exception:                                              # noqa: BLE001
            import traceback
            fails += 1
            print(f"FAIL {name}")
            traceback.print_exc()
    print(f"{'ALL PASS' if not fails else f'{fails} FAILED'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(_main())
