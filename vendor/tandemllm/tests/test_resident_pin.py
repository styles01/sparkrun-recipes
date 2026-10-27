"""A long unrelated request between two turns, and the state store's boundaries, on the random
4-layer model.

The case is opencode's compaction (2026-09-30, seven times in one night): a growing conversation
whose prompt starts with a long tool block, then a request with no tools and no system text that
shares nothing with it and is far too long to be a guest, then a normal request that starts with
the same tool block again. Before the pin that last request re-read everything; the rows it
needed had been overwritten and the state store had evicted its own copy, because one cold
prefill stored a snapshot at every chunk boundary of its own prompt.

Everything here resumes on the chunk grid, so the resumed request is held to bit equality with a
cold prefill of its whole prompt, as in tests/test_resident.py.

Run: python tests/test_resident_pin.py
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


def per_row(eng) -> int:
    return sum(t.numel() // t.shape[ax] * t.element_size() for t, ax in cache._kv_views(eng, None))


def resident(eng, pin_rows: int = 0, pin_anchors: int = 8, **kw):
    """A resident prefix whose pin holds `pin_rows` rows and `pin_anchors` anchors."""
    pin = pin_rows * per_row(eng) + pin_anchors * eng.state.nbytes if pin_rows else 0
    kw.setdefault("stash_bytes", BIG)
    return cache.ResidentPrefix(BIG, CH, pin_bytes=pin, pin_anchors=pin_anchors, **kw)


TOOLS = tokens(130, seed=1)                  # the tool block and system text every turn starts with


def conversation(eng, res, turns: int = 3) -> list[int]:
    ids = TOOLS + tokens(120, seed=2)
    cache.prefill(eng, None, ids, DEV, chunk=CH, resident=res)
    for t in range(turns):
        ids = ids + decode(eng, len(ids), k=6, seed=30 + t) + tokens(15, seed=40 + t)
        cache.prefill(eng, None, ids, DEV, chunk=CH, resident=res)
    decode(eng, len(ids), k=6, seed=50)
    return ids


def compaction(eng, res, n: int = 260) -> list[int]:
    """No tools, no system text: shares nothing with the conversation, and is not a guest."""
    ids = [0] + tokens(n - 1, seed=60)
    _, reused, _ = cache.prefill(eng, None, ids, DEV, chunk=CH, resident=res)
    assert reused == 0
    decode(eng, n, k=8, seed=61)
    return ids


# ------------------------------------------------------------------ 1. the compaction

def test_the_turn_after_a_compaction_resumes_from_the_pinned_tool_block():
    eng = fresh()
    res = resident(eng, pin_rows=160)
    conversation(eng, res)
    compaction(eng, res)
    assert res.stats["pins"] == 1 and res.pin_rows == 160, res.report()
    nxt = TOOLS + tokens(30, seed=70)              # the summary as the first user message
    info = {}
    warm, reused, fwd = cache.prefill(eng, None, nxt, DEV, chunk=CH, resident=res, info=info)
    assert (reused, info["kind"]) == (128, "resident"), (reused, info)
    assert res.stats["pin_hits"] == 1 and res.stats["pin_tokens_reused"] == 128
    ref, lg = cold(nxt)
    assert torch.equal(warm, lg), (warm - lg).abs().max().item()
    same_state(eng, ref, len(nxt))


def test_without_the_pin_the_turn_after_a_compaction_starts_from_zero():
    """The 2026-09-30 behaviour, kept as the control."""
    eng = fresh()
    res = resident(eng, pin_rows=0)
    conversation(eng, res)
    compaction(eng, res)
    _, reused, _ = cache.prefill(eng, None, TOOLS + tokens(30, seed=70), DEV, chunk=CH,
                                 resident=res)
    assert reused == 0 and res.stats["pins"] == 0


def test_the_conversation_goes_on_after_the_pinned_resume():
    eng = fresh()
    res = resident(eng, pin_rows=160)
    conversation(eng, res)
    compaction(eng, res)
    ids = TOOLS + tokens(30, seed=70)
    cache.prefill(eng, None, ids, DEV, chunk=CH, resident=res)
    ids = ids + decode(eng, len(ids), k=6, seed=71) + tokens(12, seed=72)
    warm, reused, _ = cache.prefill(eng, None, ids, DEV, chunk=CH, resident=res)
    assert reused == 144, reused
    ref, lg = cold(ids)
    assert torch.equal(warm, lg)
    same_state(eng, ref, len(ids))


def test_a_head_longer_than_the_pin_is_pinned_up_to_what_fits():
    eng = fresh()
    res = resident(eng, pin_rows=70, pin_anchors=3)
    conversation(eng, res)
    compaction(eng, res)
    assert res.pin_rows == 64 and sorted(res.pin_anchors) == [16, 32, 64], res.report()
    budget = 70 * per_row(eng) + 3 * eng.state.nbytes
    pinned = sum(b.numel() * b.element_size() for b in res._pin_bufs)
    assert pinned + len(res.pin_anchors) * eng.state.nbytes <= budget
    warm, reused, _ = cache.prefill(eng, None, TOOLS + tokens(30, seed=70), DEV, chunk=CH,
                                    resident=res)
    assert reused == 64
    ref, lg = cold(TOOLS + tokens(30, seed=70))
    assert torch.equal(warm, lg)


def test_a_request_sharing_less_than_the_live_rows_does_not_unpin():
    eng = fresh()
    res = resident(eng, pin_rows=160)
    conversation(eng, res)
    comp = compaction(eng, res)
    # the compaction prompt's own head again: the live rows serve it better than the pin
    again = comp[:200] + tokens(10, seed=80)
    _, reused, _ = cache.prefill(eng, None, again, DEV, chunk=CH, resident=res)
    assert reused == 192 and res.stats["pin_hits"] == 0


def test_a_guest_between_the_compaction_and_the_next_turn():
    eng = fresh()
    res = resident(eng, pin_rows=160)
    conversation(eng, res)
    compaction(eng, res)
    cache.prefill(eng, None, [3] + tokens(20, seed=81), DEV, chunk=CH, resident=res)
    decode(eng, 21, k=8, seed=82)
    nxt = TOOLS + tokens(30, seed=70)
    warm, reused, _ = cache.prefill(eng, None, nxt, DEV, chunk=CH, resident=res)
    assert reused == 128
    ref, lg = cold(nxt)
    assert torch.equal(warm, lg)
    same_state(eng, ref, len(nxt))


def test_a_date_line_change_resumes_at_the_change_and_pins_the_rest_of_the_head():
    """opencode writes the date into its system text after the tool block: at midnight the prompt
    changes at the tool block's end. The turn resumes there, as before; the head above the change
    is pinned (it is no use to the new day, and costs one copy)."""
    eng = fresh()
    res = resident(eng, pin_rows=160)
    ids = conversation(eng, res)
    moved = ids[:100] + [ids[100] % 90 + 1] + ids[101:]
    warm, reused, _ = cache.prefill(eng, None, moved, DEV, chunk=CH, resident=res)
    assert reused == 96, reused
    ref, lg = cold(moved)
    assert torch.equal(warm, lg)
    same_state(eng, ref, len(moved))
    ids2 = moved + decode(eng, len(moved), k=6, seed=83) + tokens(10, seed=84)
    _, reused2, _ = cache.prefill(eng, None, ids2, DEV, chunk=CH, resident=res)
    assert reused2 >= len(moved) - CH and res.stats["pin_hits"] == 0


def test_a_growing_conversation_never_pins():
    eng = fresh()
    res = resident(eng, pin_rows=160)
    conversation(eng, res, turns=4)
    assert res.stats["pins"] == 0


def test_a_drafter_that_reallocated_its_cache_refuses_the_pin_and_stays_exact():
    eng = fresh()
    res = resident(eng, pin_rows=160)
    conversation(eng, res)
    compaction(eng, res)
    eng.kv.k = eng.kv.k.clone()                   # the buffer the pin was copied from is gone
    nxt = TOOLS + tokens(30, seed=70)
    warm, reused, _ = cache.prefill(eng, None, nxt, DEV, chunk=CH, resident=res)
    assert reused == 0 and res.stats["pin_refused"] == 1
    ref, lg = cold(nxt)
    assert torch.equal(warm, lg)


def test_the_head_survives_eviction_so_a_short_shared_head_still_resumes():
    """A long conversation under a small anchor budget: the smallest-gap rule alone thins the
    head out, and a request sharing only a 70-token tool block then finds no pinned anchor at or
    below it. The head lines (every 4 chunks) are kept."""
    eng = fresh()
    per = eng.state.nbytes
    pin = 160 * per_row(eng) + 8 * per
    res = cache.ResidentPrefix(6 * per, CH, tail=2, stash_bytes=BIG, pin_bytes=pin)
    head = TOOLS[:70]
    ids = head + tokens(330, seed=100)
    cache.prefill(eng, None, ids, DEV, chunk=CH, resident=res)
    assert 64 in res.anchors and len(res.anchors) <= 6, sorted(res.anchors)
    decode(eng, len(ids), k=6, seed=101)
    compaction(eng, res, n=300)
    nxt = head + tokens(40, seed=102)
    warm, reused, _ = cache.prefill(eng, None, nxt, DEV, chunk=CH, resident=res)
    assert reused == 64 and res.stats["pin_hits"] == 1, (reused, res.report())
    ref, lg = cold(nxt)
    assert torch.equal(warm, lg)
    same_state(eng, ref, len(nxt))


# ------------------------------------------------------------------ 2. the store's boundaries

def test_one_cold_prefill_stores_one_boundary():
    eng = fresh()
    one = cache.StateStore(BIG, chunk=CH, per_prefill=1)
    cache.prefill(eng, None, tokens(200, seed=90), DEV, store=one, chunk=CH, checkpoint=True)
    assert one.report()["boundaries"] == [192] and one.stats["puts"] == 1
    every = cache.StateStore(BIG, chunk=CH)
    cache.prefill(fresh(), None, tokens(200, seed=90), DEV, store=every, chunk=CH,
                  checkpoint=True)
    assert every.stats["puts"] == 12


def test_a_cold_prefill_no_longer_evicts_what_else_is_stored():
    eng = fresh()
    snap = cache.capture(eng)                    # sizes: a 0-token snapshot is the state alone
    kv_row = per_row(eng)
    budget = 2 * (snap.nbytes + 200 * kv_row)
    store = cache.StateStore(budget, chunk=CH, per_prefill=1)
    a = tokens(150, seed=91)
    cache.prefill(eng, None, a, DEV, store=store, chunk=CH, checkpoint=True)
    cache.prefill(fresh(), None, tokens(180, seed=92), DEV, store=store, chunk=CH,
                  checkpoint=True)
    assert store.stats["evictions"] == 0 and len(store.report()["boundaries"]) == 2
    warm, reused, _ = cache.prefill(fresh(), None, a + [5, 6], DEV, store=store, chunk=CH,
                                    checkpoint=True)
    assert reused == 144
    ref, lg = cold(a + [5, 6])
    assert torch.equal(warm, lg)


def test_the_longest_boundary_under_the_cap_is_the_one_chosen():
    """By `capture`'s own size estimate, so a chosen boundary is never refused for size. (The
    choice is checked, not the put: this model's KV is fp32 and the estimate is for bf16.)"""
    eng = fresh()
    cap = cache._snapshot_estimate(eng, 100, None)
    store = cache.StateStore(BIG, chunk=CH, max_entry_bytes=cap, per_prefill=1)
    assert store.boundaries(eng, None, 0, 300, CH) == {96}
    assert store.stats["skipped_big"] == 1
    two = cache.StateStore(BIG, chunk=CH, max_entry_bytes=cap, per_prefill=2)
    assert two.boundaries(eng, None, 40, 300, CH) == {80, 96}
    assert cache.StateStore(BIG, chunk=CH, per_prefill=1).boundaries(eng, None, 0, 300, CH) \
        == {288}
    assert cache.StateStore(BIG, chunk=CH).boundaries(eng, None, 0, 300, CH) is None


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:75s} ok", flush=True)
            passed += 1
    print(f"{passed} passed")
