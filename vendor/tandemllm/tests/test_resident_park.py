"""The memory limits on the random 4-layer model (OPS-47, QA OPS-48): the parked session, the caps,
the shed, and the claim that none of it changes a token.

Every resume here lands on the chunk grid, so a resumed request is held to bit equality with a cold
prefill of its whole prompt, as in tests/test_resident.py and tests/test_resident_pin.py.

  * A, B, A: two sessions that share their head (opencode's tool block) and nothing after it. B
    replaces A's rows; A's next turn swaps them back from the park; B's next turn swaps again.
  * the caps: a long run under small budgets never holds more than each cap.
  * shed then resume: every level given back, and the next turn is the same bytes as the run that
    gave nothing back, and as a cold prefill.
  * a refused allocation (the memory policy says no) leaves the cold path, never a wrong state.
  * the trace replay: one opencode-shaped trace with the caches off, small and large; the token
    streams are identical, only where each prefill started differs (the plan's gate a).
  * the parked end: an Open WebUI chat (its copy leaves the first answer's reasoning out) parked
    by another chat comes back at its end anchor, with the same greedy tokens and the same engine
    state as the turn that never left; the swap parks the other chat's end the same way; a
    conversation without an end, or an end the park cannot hold, parks as before.

Run: python tests/test_resident_park.py
"""

from __future__ import annotations

import os
import sys

for _k in ("NORM", "GDN", "HEAD", "ATTN", "GDNBLOCK", "GDNTREE", "GDNPRE"):
    os.environ.setdefault(f"QWEN38_FUSED_{_k}", "0")

import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine import cache  # noqa: E402
from test_cache import DEV, fresh, tokens  # noqa: E402
from test_resident import cold, decode, same_state  # noqa: E402

CH = 16
BIG = 1 << 30
HEAD = tokens(130, seed=1)                   # the tool block and system text both sessions share


def per_row(eng) -> int:
    return sum(t.numel() // t.shape[ax] * t.element_size() for t, ax in cache._kv_views(eng, None))


def resident(eng, park_rows: int = 400, park_anchors: int = 4, **kw):
    park = park_rows * per_row(eng) + park_anchors * eng.state.nbytes if park_rows else 0
    kw.setdefault("stash_bytes", BIG)
    return cache.ResidentPrefix(kw.pop("budget", BIG), CH, park_bytes=park,
                                park_anchors=park_anchors, **kw)


def turn(eng, res, ids, session, **kw):
    info = {}
    lg, reused, fwd = cache.prefill(eng, None, ids, DEV, chunk=CH, resident=res, conv_id=session,
                                    info=info, **kw)
    return lg, reused, info


def grow(ids, seed):
    """The next turn: the prompt, an answer, a tool result (the rows the answer's decode wrote
    are a `decode` call of the test's own, where it ran)."""
    return ids + tokens(7, seed=seed) + tokens(11, seed=seed + 1)


# ------------------------------------------------------------------ 6: A, B, A

def test_a_session_switch_back_swaps_the_parked_rows_in_bit_for_bit():
    eng = fresh()
    res = resident(eng)
    a = HEAD + tokens(150, seed=2)
    turn(eng, res, a, "A")
    a = grow(a, 10)
    turn(eng, res, a, "A")
    decode(eng, len(a), k=5, seed=12)
    b = HEAD + tokens(140, seed=3)                   # another session, the same head
    _, reused, info = turn(eng, res, b, "B")
    assert reused == 128 and info["kind"] == "resident", (reused, info)
    assert res.stats["parks"] == 1 and res.park_owner == "A" and res.park_base == 128
    decode(eng, len(b), k=5, seed=13)
    a2 = grow(a, 14)                            # A again: from the park, not re-read
    warm, reused, info = turn(eng, res, a2, "A")
    top = max(x for x in range(CH, len(a), CH))      # A's newest anchor below its prompt
    assert reused == top and info["kind"] == "resident", (reused, top)
    assert res.stats["park_hits"] == 1 and res.stats["park_swaps"] == 1
    assert res.park_owner == "B", "B was parked in A's place"
    ref, lg = cold(a2)
    assert torch.equal(warm, lg), (warm - lg).abs().max().item()
    same_state(eng, ref, len(a2))
    decode(eng, len(a2), k=5, seed=15)
    b2 = grow(b, 16)                            # and B again, swapped back too
    warm, reused, _ = turn(eng, res, b2, "B")
    assert reused == max(x for x in range(CH, len(b), CH)) and res.stats["park_hits"] == 2
    ref, lg = cold(b2)
    assert torch.equal(warm, lg)
    same_state(eng, ref, len(b2))
    sw = res.switches[-1]
    assert sw["from"] == "A" and sw["to"] == "B" and sw["via"] == "park" and sw["reread"] > 0


def test_a_park_smaller_than_the_session_holds_what_fits_and_still_resumes_exactly():
    eng = fresh()
    res = resident(eng, park_rows=70, park_anchors=2)
    a = HEAD + tokens(200, seed=4)
    turn(eng, res, a, "A")
    decode(eng, len(a), k=5, seed=20)
    turn(eng, res, HEAD + tokens(100, seed=5), "B")
    assert res.park_base == 128 and res.park_rows == 64, res.report()     # 192 is the top that fits
    assert res.park_bytes <= res.park_budget
    warm, reused, _ = turn(eng, res, grow(a, 21), "A")
    assert reused == 192
    ref, lg = cold(grow(a, 21))
    assert torch.equal(warm, lg)


def test_a_parked_prompt_is_not_pinned_too():
    """The park keeps the rows above the shared head and the head stays in place: pinning it as
    well held the same session twice (3.8 GiB beside 4 GiB in the 2026-10-01 gate)."""
    eng = fresh()
    res = resident(eng, pin_bytes=BIG)
    turn(eng, res, HEAD + tokens(150, seed=34), "A")
    turn(eng, res, HEAD + tokens(140, seed=35), "B")
    assert res.stats["parks"] == 1 and res.stats["pins"] == 0 and res.pin_rows == 0
    turn(eng, res, [0] + tokens(260, seed=36), "B", compacting=True)      # B's compaction: pinned
    assert res.stats["pins"] == 1 and res.stats["parks"] == 1


def test_the_same_session_parting_from_itself_is_not_parked():
    eng = fresh()
    res = resident(eng)
    a = HEAD + tokens(150, seed=6)
    turn(eng, res, a, "A")
    turn(eng, res, HEAD + tokens(150, seed=7), "A")  # an edited earlier message: A again
    assert res.stats["parks"] == 0


def test_a_compaction_does_not_park_and_drops_its_parked_rows():
    eng = fresh()
    res = resident(eng)
    a = HEAD + tokens(150, seed=8)
    turn(eng, res, a, "A")
    turn(eng, res, [0] + tokens(260, seed=9), "A", compacting=True)
    assert res.stats["parks"] == 0
    turn(eng, res, a, "A")
    turn(eng, res, HEAD + tokens(120, seed=30), "B")            # A parked
    assert res.park_owner == "A"
    assert res.drop_session("A") > 0 and res.park_rows == 0 and res.stats["park_dropped"] == 1
    assert res.drop_session("B") == 0


def test_without_session_ids_a_prompt_that_parts_from_the_rows_is_another_session():
    eng = fresh()
    res = resident(eng)
    a = HEAD + tokens(150, seed=31)
    turn(eng, res, a, None)
    turn(eng, res, HEAD + tokens(120, seed=32), None)
    assert res.stats["parks"] == 1 and res.stats["switches"] == 1
    warm, reused, _ = turn(eng, res, grow(a, 33), None)
    assert res.stats["park_hits"] == 1
    ref, lg = cold(grow(a, 33))
    assert torch.equal(warm, lg)


# ------------------------------------------------------------------ 1: the caps hold

def test_every_cache_stays_under_its_cap_over_a_long_interleaved_run():
    eng = fresh()
    ab = eng.state.nbytes
    pr = per_row(eng)
    res = cache.ResidentPrefix(5 * ab, CH, stash_bytes=40 * pr, guest_max=40,
                               pin_bytes=100 * pr + 3 * ab, pin_anchors=3,
                               park_bytes=90 * pr + 2 * ab, park_anchors=2)
    store = cache.StateStore(3 * ab + 200 * pr, chunk=CH, max_entry_bytes=ab + 150 * pr,
                             per_prefill=1)
    sess = {"A": HEAD + tokens(60, seed=40), "B": HEAD + tokens(70, seed=41)}
    for t in range(6):
        for s in ("A", "B"):
            sess[s] = sess[s] + tokens(25, seed=50 + t * 2 + (s == "B"))
            cache.prefill(eng, None, sess[s], DEV, store=store, chunk=CH, checkpoint=True,
                          resident=res, conv_id=s)
            decode(eng, len(sess[s]), k=4, seed=70 + t)
            cache.prefill(eng, None, tokens(20, seed=90 + t), DEV, store=store, chunk=CH,
                          checkpoint=True, resident=res, conv_id="title")   # a guest
            assert res.anchor_bytes <= 5 * ab, res.report()
            assert res.stash_bytes <= 40 * pr and res.pin_bytes <= 100 * pr + 3 * ab
            assert res.park_bytes <= 90 * pr + 2 * ab and store.bytes <= store.budget


# ------------------------------------------------------------------ shed then resume

def _filled(eng):
    """Caches holding something at every level, from the same requests: A's compaction pins its
    head, B parks A's next prompt, a title call stashes B's rows, the store has prefix entries."""
    res = resident(eng, pin_bytes=200 * per_row(eng) + 8 * eng.state.nbytes)
    store = cache.StateStore(BIG, chunk=CH, per_prefill=1)

    def run(ids, s, **kw):
        cache.prefill(eng, None, ids, DEV, store=store, chunk=CH, checkpoint=True, resident=res,
                      conv_id=s, **kw)
    a = HEAD + tokens(150, seed=60)
    run(a, "A")
    a = grow(a, 61)
    run(a, "A")
    decode(eng, len(a), k=4, seed=62)
    run([0] + tokens(250, seed=63), "A", compacting=True)               # pins A's head
    decode(eng, 251, k=4, seed=64)
    a = HEAD + tokens(60, seed=65)                                       # the turn after it
    run(a, "A")
    run(HEAD + tokens(140, seed=66), "B")                                # parks A
    decode(eng, 270, k=4, seed=67)
    run(tokens(20, seed=68), "title")                                    # a guest: a stash
    return res, store, a


def test_shed_every_level_then_the_next_turn_is_the_same_bytes():
    eng1, eng2 = fresh(), fresh()
    res1, store1, a1 = _filled(eng1)
    res2, store2, a2 = _filled(eng2)
    assert a1 == a2 and res1.pin_rows and res1.park_rows and res1._stash is not None
    assert res1.pin_bytes <= res1.pin_budget and res1.park_bytes <= res1.park_budget
    assert store1.bytes > 0
    freed = store2.shed() + res2.shed_anchors() + res2.shed_stash() + res2.shed_park() \
        + res2.shed_pin()
    # what is left is the anchors a level never takes: the newest few and the head lines
    assert freed > 0 and store2.bytes == 0
    assert res2.stash_bytes == res2.pin_bytes == res2.park_bytes == 0
    assert res2.bytes == len(res2.anchors) * eng2.state.nbytes and not res2._free
    assert len(res2.anchors) <= res2.tail + res2.pin_anchor_max
    nxt = grow(a1, 66)
    w1, r1, _ = cache.prefill(eng1, None, nxt, DEV, store=store1, chunk=CH, checkpoint=True,
                              resident=res1, conv_id="A")
    w2, r2, _ = cache.prefill(eng2, None, nxt, DEV, store=store2, chunk=CH, checkpoint=True,
                              resident=res2, conv_id="A")
    assert r1 > 0 and r2 == 0, (r1, r2)              # warm from the park against cold
    assert torch.equal(w1, w2), (w1 - w2).abs().max().item()
    same_state(eng1, eng2, len(nxt))
    ref, lg = cold(nxt)
    assert torch.equal(w1, lg)


def test_a_safe_shed_never_touches_rows_in_use():
    eng = fresh()
    res, store, a = _filled(eng)
    pending = res._stash is not None
    assert pending and res.shed_stash(safe=True) == 0 and res._stash is not None
    anchors = dict(res.anchors)
    res.shed_anchors(safe=True)
    assert res.anchors == anchors, "a safe shed frees only the free list"


# ------------------------------------------------------------------ the memory policy says no

class CacheProbe:
    """An allocator that is the caches and nothing else, with a ceiling the test sets."""

    def __init__(self, res, store, ceiling):
        self.res, self.store, self.ceiling = res, store, ceiling

    def probe(self):
        n = self.res.bytes + (self.store.bytes if self.store is not None else 0)
        return n, n


def _policy(res, store, ceiling):
    p = CacheProbe(res, store, ceiling)
    mem = cache.Budget(ceiling, ceiling * 4, 0, probe=p.probe, empty=lambda: None,
                       peak_probe=lambda reset=False: 0)
    res.mem = mem
    if store is not None:
        mem.register("store", store.shed)
    for lv in ("anchors", "stash", "park", "pin"):
        mem.register(lv, getattr(res, f"shed_{lv}"))
    return mem


def test_refused_allocations_take_the_cold_path_and_stay_exact():
    eng = fresh()
    ab = eng.state.nbytes
    res = resident(eng)
    store = cache.StateStore(BIG, chunk=CH, per_prefill=1)
    mem = _policy(res, store, 4 * ab)                # room for four anchors and nothing else
    a = HEAD + tokens(150, seed=70)
    cache.prefill(eng, None, a, DEV, store=store, chunk=CH, checkpoint=True, resident=res,
                  conv_id="A", mem=mem)
    assert res.stats["refused_anchor"] > 0 and res._allocated <= 4
    assert mem.stats["admit_refused"] > 0
    decode(eng, len(a), k=4, seed=71)
    b = HEAD + tokens(140, seed=72)
    cache.prefill(eng, None, b, DEV, store=store, chunk=CH, checkpoint=True, resident=res,
                  conv_id="B", mem=mem)
    assert res.park_rows == 0 and res.stats["refused_park"] >= 1
    decode(eng, len(b), k=4, seed=73)
    g = tokens(20, seed=74)                          # a guest whose stash is refused: rows cut
    cache.prefill(eng, None, g, DEV, store=store, chunk=CH, checkpoint=True, resident=res,
                  conv_id="t", mem=mem)
    assert res.stats["refused_stash"] >= 1
    for ids, s in ((grow(a, 75), "A"), (b + tokens(9, seed=76), "B")):
        warm, _, _ = cache.prefill(eng, None, ids, DEV, store=store, chunk=CH, checkpoint=True,
                                   resident=res, conv_id=s, mem=mem)
        ref, lg = cold(ids)
        assert torch.equal(warm, lg)
        same_state(eng, ref, len(ids))


# ------------------------------------------------------------------ 9: the trace replay

def _greedy(eng, logits, n_prompt, k):
    out, tok, pos = [], int(logits[0, -1].argmax()), n_prompt
    for _ in range(k):
        out.append(tok)
        lg = eng.forward(torch.tensor([tok], device=DEV), start=pos, last_only=True)
        pos += 1
        tok = int(lg[0, -1].argmax())
    return out


def _trace():
    """The night of 2026-09-30, scaled down: one session growing turn by turn with a short title
    call between, its compaction and the turn after it, and a second session in between."""
    a = HEAD + tokens(90, seed=80)
    b = HEAD + tokens(70, seed=81)
    return [("A", a, 6, False), ("title", tokens(18, seed=82), 3, False),
            ("A", None, 6, False), ("B", b, 6, False), ("A", None, 6, False),
            ("B", None, 5, False), ("A", None, 5, False),
            ("A", [0] + tokens(200, seed=83), 4, True),          # the compaction
            ("A", HEAD + tokens(25, seed=84), 6, False), ("B", None, 4, False)]


def _replay(make):
    eng = fresh()
    res, store = make(eng)
    sess, streams, starts = {}, [], []
    for i, (s, ids, k, comp) in enumerate(_trace()):
        if ids is None:
            ids = sess[s] + streams[sess[s + ":last"]] + tokens(9, seed=100 + i)
        if comp and store is not None:
            store.drop_session(s)
        if comp and res is not None:
            res.drop_session(s)
        lg, reused, _ = cache.prefill(eng, None, ids, DEV, store=store, chunk=CH,
                                      checkpoint=store is not None, resident=res, conv_id=s,
                                      compacting=comp)
        streams.append(_greedy(eng, lg, len(ids), k))
        starts.append(reused)
        if s != "title":
            sess[s], sess[s + ":last"] = ids, len(streams) - 1
    return streams, starts


def test_the_trace_gives_the_same_tokens_with_the_caches_off_small_and_large():
    def off(eng):
        return None, None

    def small(eng):                  # the "2 GiB" run: everything capped, the park too short
        ab, pr = eng.state.nbytes, per_row(eng)
        return (cache.ResidentPrefix(5 * ab, CH, stash_bytes=30 * pr, guest_max=40,
                                     pin_bytes=80 * pr + 2 * ab, pin_anchors=2,
                                     park_bytes=40 * pr + 2 * ab, park_anchors=2),
                cache.StateStore(2 * ab, chunk=CH, max_entry_bytes=ab + 60 * pr, per_prefill=1))

    def large(eng):                  # the "8 GiB" run: nothing is ever short of room
        return (cache.ResidentPrefix(BIG, CH, stash_bytes=BIG, pin_bytes=BIG, park_bytes=BIG),
                cache.StateStore(BIG, chunk=CH, per_prefill=1))

    s0, r0 = _replay(off)
    s2, r2 = _replay(small)
    s8, r8 = _replay(large)
    assert s0 == s2 == s8, "a cache limit changed a token"
    assert sum(r0) == 0 and sum(r8) > sum(r2) > 0, (r0, r2, r8)


# ------------------------------------------------------------------ the parked end

def _owui(park: bool):
    """Chat A (an Open WebUI chat whose first answer thinks), then, with `park`, chat B (not a
    guest: A is parked with its end), then A's next turn as Open WebUI sends it: the first answer
    without its reasoning (`<think>\n\n</think>\n\n` + the answer), and a new message. The server's
    order: `unpark_end` (engine/spill.py `maybe_restore` calls it before the replay), the end
    alignment's replay, the turn. Returns what the turn served, wrote and left in the engine."""
    import test_end_align as EA
    from server import app, diskcache, endalign
    from test_end_anchor import TOOL, _state, turn
    head = tokens(4 * CH, seed=110)
    a = head + tokens(99, seed=111) + [EA.NL]          # the prompt ends `<think>\n`
    _state()
    out0, _, _ = turn(a, 60, session="A")
    k = next(i for i in range(12, 30) if out0[i] not in out0[:i] and out0[i] not in a
             and out0[i] not in (EA.NL, EA.SP))
    res = _state(park_bytes=BIG, guest_max=10)
    al = app.STATE["end_align"] = endalign.EndAlign(EA.tb_of({out0[k]: endalign.THINK_END}))
    diskcache.wire_end_align(None, al, res)
    eng = app.STATE["engine"]
    out, _, _ = turn(a, 60, session="A")
    assert out == out0
    e = res.end["at"]
    regions = [list(r) for r in al.regions(res)]
    assert regions == [[len(a), e, len(a) + k]], regions
    copy = a[:-1] + [EA.SP] + out[k:] + TOOL
    info = {"e": e, "res": res, "eng": eng, "al": al}
    if park:
        b = head + tokens(140, seed=112)
        out_b, _, _ = turn(b, 10, session="B")
        e_b = res.end["at"]
        assert res.park_owner == "A" and res.park_base == len(head), res.report()
        assert res.park_end["at"] == e and res.park_base + res.park_rows == e
        assert res.park_regions == regions and res.stats["park_ends"] == 1
        assert res.owner == "B" and res.end["at"] == e_b
        c = res.common(copy)
        # the copy reaches the parked end through the alignment only, as the server's copy
        assert res.park_offer(copy, c, len(copy) - 1) == e
        assert res.park_offer(copy, c, len(copy) - 1, align=False) < len(a)
        assert res.unpark_end(eng, None, copy, len(copy) - 1) == e
        assert res.owner == "A" and res.end["at"] == e and al.regions(res) == [tuple(regions[0])]
        # the swap parked B with its end, its regions too
        assert res.park_owner == "B" and res.park_end["at"] == e_b
        assert res.park_regions and res.park_regions[0][:2] == [len(b), e_b]
        info.update(b=b, out_b=out_b, e_b=e_b)
    ids, spans, aligned = app._end_replay(res, copy)
    assert aligned > 0 and ids[:e] == res.tokens[:e], (aligned, len(ids), e)
    out2, pre, _ = turn(ids, 12, session="A")
    assert pre["kind"] == "end" and pre["reused"] == e, pre
    info.update(ids=ids, out2=out2, n=int(eng.kv.length))
    return info


def test_a_parked_chat_without_its_reasoning_comes_back_at_its_end():
    """0.4.7 live (2026-10-08, Open WebUI chats A and B): `parks 4, park_hits 0`; the park held
    rows and grid anchors only, and a copy without the first answer's reasoning parts from the
    rows below every one the disk head did not already give: chat B came back with 16,370 tokens
    re-read (23.7 s). The park keeps the end anchor now, and the copy resumes there: the same
    served prompt, the same 12 greedy tokens, the same engine state as the turn that never left
    (the end anchor's gate, and here bit for bit: the rows and the state are the bytes that were
    in place)."""
    from test_end_anchor import TOOL2, turn
    ref = _owui(park=False)
    got = _owui(park=True)
    res = got["res"]
    assert got["ids"] == ref["ids"] and got["out2"] == ref["out2"], "the same served prompt and tokens"
    assert got["n"] == ref["n"]
    same_state(got["eng"], ref["eng"], got["n"])
    assert res.stats["park_hits"] == 1 and res.stats["park_tokens_reused"] == got["e"], res.stats
    assert res.stats["park_end_hits"] == 1
    # and B's turn after it: its end came back with the swap (the prefill's own park path: the
    # copy holds every token up to it), the end anchor's gate against a cold prefill
    b_ids = got["b"] + got["out_b"] + TOOL2
    out_b2, pre_b, _ = turn(b_ids, 6, session="B")
    assert pre_b["kind"] == "end" and pre_b["reused"] == got["e_b"], pre_b
    assert res.stats["park_hits"] == 2 and res.stats["park_tokens_reused"] == got["e"] + got["e_b"]
    import test_end_align as EA
    assert out_b2 == EA._cold(b_ids, 6), "the end anchor's gate: the same greedy tokens as cold"
    return f"A back at its end {got['e']} from the park, B at {got['e_b']}; tokens and state equal"


def _plain_abab(end: bool):
    """A, B, A, B on the cache alone (no generation keeps an end): what each prefill reused, its
    logits, the park after each."""
    eng = fresh()
    res = resident(eng, end=end)
    a = HEAD + tokens(150, seed=120)
    b = HEAD + tokens(140, seed=121)
    seen = []
    for ids, s in ((a, "A"), (b, "B"), (grow(a, 122), "A"), (grow(b, 123), "B")):
        lg, reused, info = turn(eng, res, ids, s)
        decode(eng, len(ids), k=5, seed=124)
        seen.append((reused, info["kind"], lg, res.park_owner, res.park_base, res.park_rows,
                     sorted(res.park_anchors)))
        assert res.park_end is None
    return seen, res


def test_a_parked_conversation_without_an_end_parks_as_before():
    off, r0 = _plain_abab(False)
    on, r1 = _plain_abab(True)
    for x, y in zip(off, on):
        assert x[:2] == y[:2] and x[3:] == y[3:], (x[:2], y[:2], x[3:], y[3:])
        assert torch.equal(x[2], y[2])
    for k in ("parks", "park_hits", "park_swaps", "park_tokens_reused"):
        assert r0.stats[k] == r1.stats[k], k
    assert r1.stats["park_ends"] == 0 and r1.stats["park_hits"] == 2


def test_an_end_the_park_cannot_hold_parks_the_grid_as_before():
    """Rows up to the end that do not fit the park: the rows up to the newest anchor that fits
    are parked, as before, and the end anchor is dropped with the rows in place."""
    from server import app
    from test_end_anchor import TOOL, _state, turn as served
    eng0 = fresh()
    rows = 70
    res = _state(park_bytes=rows * per_row(eng0) + 2 * eng0.state.nbytes, park_anchors=2,
                 guest_max=10)
    a = HEAD + tokens(150, seed=125)
    out, _, _ = served(a, 20, session="A")
    e = res.end["at"]
    assert e - 128 > rows
    served(HEAD + tokens(140, seed=126), 6, session="B")
    assert res.park_owner == "A" and res.park_end is None and res.stats["park_ends"] == 0
    assert res.park_base == 128 and res.park_rows == 64, res.report()     # 192, the grid line
    assert res.park_bytes <= res.park_budget
    _, pre, _ = served(a + out + TOOL, 4, session="A")
    assert pre["kind"] == "resident" and pre["reused"] == 192, pre
    assert app.STATE["resident"].stats["park_hits"] == 1


def _main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")


if __name__ == "__main__":
    _main()
