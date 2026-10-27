"""The resident prefix: the live KV buffer as the prefix cache, on the random 4-layer model.

THE CLAIM is the one tests/test_cache.py makes for the store, held to the same standard: a request
that resumes from the rows already in the buffer and an anchor's recurrent state computes the same
logits, S, conv and KV rows -- to the last bit -- as a cold prefill of its whole prompt on the same
chunk grid. Not the same argmax: the same tensors.

The cases are the ones an agent client produces (2026-09-26, opencode, 60k -> 211k tokens):

  * the next turn's prompt is the previous prompt plus the answer plus a tool result;
  * the previous turn's decode wrote rows past its prompt, and they must not be reused;
  * a short unrelated request (a title call, a chat) runs between two turns;
  * a client gives up part way through a long prefill and sends again;
  * a prompt that shares only a head with the resident one.

Run: python tests/test_resident.py
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

CH = 16
BIG = 1 << 30


def resident(**kw):
    kw.setdefault("stash_bytes", BIG)
    return cache.ResidentPrefix(kw.pop("budget", BIG), CH, **kw)


def cold(ids, drafter=None):
    eng = fresh()
    lg, _, _ = cache.prefill(eng, drafter, ids, DEV, store=None, chunk=CH)
    return eng, lg


def same_state(a, b, n):
    assert torch.equal(a.state.S, b.state.S), (a.state.S - b.state.S).abs().max().item()
    assert torch.equal(a.state.conv, b.state.conv)
    assert torch.equal(a.kv.k[:, :, :, :n], b.kv.k[:, :, :, :n])
    assert torch.equal(a.kv.v[:, :, :, :n], b.kv.v[:, :, :, :n])
    assert a.kv.length == b.kv.length == n


def decode(eng, n_prompt, k=9, seed=99):
    """What a generation does to the buffer: rows past the prompt, and a recurrent state that has
    absorbed them. Written by a forward that is NOT on the prefill's grid, as a verify block's is."""
    gen = tokens(k, seed=seed)
    eng.forward(torch.tensor(gen), start=n_prompt, last_only=True)
    return gen


# ------------------------------------------------------------------ 1. the next turn

def test_the_next_turn_resumes_in_place_bit_for_bit():
    turn1 = tokens(100, seed=1)
    res, eng = resident(), fresh()
    cache.prefill(eng, None, turn1, DEV, chunk=CH, resident=res)
    assert sorted(res.anchors) == list(range(16, 100, 16)), res.anchors
    answer = decode(eng, len(turn1))
    turn2 = turn1 + answer + tokens(23, seed=2)            # prompt + answer + tool result
    info = {}
    warm, reused, fwd = cache.prefill(eng, None, turn2, DEV, chunk=CH, resident=res, info=info)
    # the last anchor at or below the old prompt: the answer is re-read, as a cold prefill does
    assert (reused, fwd, info["kind"]) == (96, len(turn2) - 96, "resident")
    ref, lg = cold(turn2)
    assert torch.equal(warm, lg), (warm - lg).abs().max().item()
    same_state(eng, ref, len(turn2))


def test_reuse_is_not_capped_by_the_store_entry_size():
    """The 2026-09-26 bug: every turn past 13,312 tokens was a cold prefill because a store
    snapshot of it was over the per-entry cap. The resident prefix has no per-length cost."""
    turn1 = tokens(300, seed=3)
    store = cache.StateStore(BIG, chunk=CH, max_entry_bytes=1)     # nothing fits the store
    res, eng = resident(), fresh()
    cache.prefill(eng, None, turn1, DEV, store=store, chunk=CH, checkpoint=True, resident=res)
    assert store.report()["entries"] == 0 and store.stats["skipped_big"] > 0
    turn2 = turn1 + decode(eng, 300) + tokens(30, seed=4)
    warm, reused, _ = cache.prefill(eng, None, turn2, DEV, store=store, chunk=CH,
                                    checkpoint=True, resident=res)
    assert reused == 288, reused
    ref, lg = cold(turn2)
    assert torch.equal(warm, lg)
    same_state(eng, ref, len(turn2))


def test_a_longer_store_entry_still_wins_and_a_shorter_one_is_not_a_miss():
    turn1 = tokens(64, seed=5)
    store = cache.StateStore(BIG, chunk=CH)
    res, eng = resident(), fresh()
    cache.prefill(eng, None, turn1, DEV, store=store, chunk=CH, checkpoint=True, resident=res)
    misses = store.stats["misses"]
    warm, reused, _ = cache.prefill(eng, None, turn1 + [7, 8, 9], DEV, store=store, chunk=CH,
                                    checkpoint=True, resident=res)
    assert reused == 48                                    # both hold 48; the resident is free
    assert store.stats["misses"] == misses, "a resident hit is not a store miss"
    assert res.stats["hits"] == 1


# ------------------------------------------------------------------ 2. divergence and abandon

def test_a_divergence_drops_every_anchor_above_it():
    head = tokens(50, seed=6)
    a = head + tokens(150, seed=7)
    b = head + tokens(120, seed=8)               # long: a new conversation, not a guest
    res, eng = resident(), fresh()
    cache.prefill(eng, None, a, DEV, chunk=CH, resident=res)
    warm, reused, _ = cache.prefill(eng, None, b, DEV, chunk=CH, resident=res)
    assert reused == 48
    assert res.tokens == b and all(x <= len(b) for x in res.anchors)
    ref, lg = cold(b)
    assert torch.equal(warm, lg)
    same_state(eng, ref, len(b))


class Gone(ConnectionResetError):
    pass


def test_an_abandoned_prefill_keeps_what_it_did():
    """A client that gives up at 60 s used to cost the retry the whole dead prefill; now the
    prefill stops at the next chunk and the retry resumes where it stopped."""
    ids = tokens(200, seed=9)
    res, eng = resident(), fresh()

    def leave(done, total):
        if done >= 80:
            raise Gone("client left")

    try:
        cache.prefill(eng, None, ids, DEV, chunk=CH, resident=res, on_chunk=leave)
        raise AssertionError("the prefill should have stopped")
    except Gone:
        pass
    assert res.valid == 80 and max(res.anchors) == 80
    retry = ids + tokens(5, seed=10)
    warm, reused, _ = cache.prefill(eng, None, retry, DEV, chunk=CH, resident=res)
    assert reused == 80
    ref, lg = cold(retry)
    assert torch.equal(warm, lg)
    same_state(eng, ref, len(retry))


def test_on_chunk_is_called_between_chunks_only():
    ids = tokens(40, seed=11)
    seen = []
    cache.prefill(fresh(), None, ids, DEV, chunk=CH, on_chunk=lambda d, t: seen.append((d, t)))
    assert seen == [(16, 40), (32, 40)]


# ------------------------------------------------------------------ 3. guests

def test_a_guest_between_two_turns_does_not_cost_the_conversation():
    turn1 = tokens(240, seed=12)
    res, eng = resident(), fresh()
    cache.prefill(eng, None, turn1, DEV, chunk=CH, resident=res)
    answer = decode(eng, 240)
    title = tokens(20, seed=13)                          # opencode's title call, say
    cache.prefill(eng, None, title, DEV, chunk=CH, resident=res)
    decode(eng, 20, k=12, seed=14)
    assert res.stats["guests"] == 1 and res.tokens[:240] == turn1
    turn2 = turn1 + answer + tokens(11, seed=15)
    warm, reused, _ = cache.prefill(eng, None, turn2, DEV, chunk=CH, resident=res)
    assert reused == 224, reused
    ref, lg = cold(turn2)
    assert torch.equal(warm, lg)
    same_state(eng, ref, len(turn2))


def test_a_guest_that_outruns_the_stash_cuts_the_prefix_and_stays_exact():
    turn1 = tokens(400, seed=16)
    eng = fresh()
    per_row = sum(t.numel() // t.shape[ax] * t.element_size()
                  for t, ax in cache._kv_views(eng, None))
    res = resident(stash_bytes=40 * per_row)             # 40 rows
    cache.prefill(eng, None, turn1, DEV, chunk=CH, resident=res)
    guest = tokens(30, seed=17)
    cache.prefill(eng, None, guest, DEV, chunk=CH, resident=res)
    decode(eng, 30, k=40, seed=18)                       # writes rows up to 70 (+ slack) > 40
    turn2 = turn1 + tokens(9, seed=19)
    warm, reused, _ = cache.prefill(eng, None, turn2, DEV, chunk=CH, resident=res)
    assert res.stats["stash_short"] == 1
    assert reused == 32, reused                          # the last anchor inside the saved rows
    ref, lg = cold(turn2)
    assert torch.equal(warm, lg)
    same_state(eng, ref, len(turn2))


def test_a_guest_that_shares_the_system_prompt_starts_from_an_anchor():
    system = tokens(64, seed=20)
    turn1 = system + tokens(300, seed=21)
    res, eng = resident(), fresh()
    cache.prefill(eng, None, turn1, DEV, chunk=CH, resident=res)
    sub = system + tokens(12, seed=22)                   # a sub-agent: same head, short
    warm, reused, _ = cache.prefill(eng, None, sub, DEV, chunk=CH, resident=res)
    assert reused == 64 and res.stats["guests"] == 1
    ref, lg = cold(sub)
    assert torch.equal(warm, lg)
    turn2 = turn1 + tokens(5, seed=23)
    warm2, reused2, _ = cache.prefill(eng, None, turn2, DEV, chunk=CH, resident=res)
    assert reused2 == 352
    ref2, lg2 = cold(turn2)
    assert torch.equal(warm2, lg2)
    same_state(eng, ref2, len(turn2))


# ------------------------------------------------------------------ 4. memory and safety

def test_the_anchor_budget_bounds_memory_and_keeps_the_newest():
    eng = fresh()
    per = eng.state.nbytes
    res = resident(budget=6 * per, tail=3)
    ids = tokens(480, seed=24)
    cache.prefill(eng, None, ids, DEV, chunk=CH, resident=res)
    got = sorted(res.anchors)
    assert len(got) == 6 and res.report()["anchor_buffers"] <= 6, got
    assert got[-3:] == [432, 448, 464], got              # the tail is always kept
    assert got[0] < 240, f"the rest spreads over the prompt, not only its end: {got}"
    # and every anchor kept is still exact
    turn2 = ids[:got[1] + 5] + tokens(40, seed=25)
    warm, reused, _ = cache.prefill(eng, None, turn2, DEV, chunk=CH, resident=res)
    assert reused == got[1]
    _, lg = cold(turn2)
    assert torch.equal(warm, lg)


def test_a_session_start_leaves_no_anchors():
    """A session snapshot was written by verify blocks: nothing after it is a cold state."""
    turn1 = tokens(48, seed=26)
    store = cache.StateStore(BIG, chunk=CH)
    res, eng = resident(), fresh()
    cache.prefill(eng, None, turn1, DEV, store=store, chunk=CH, resident=res)
    store.put(turn1, cache.capture(eng), kind="session")
    turn2 = turn1 + tokens(60, seed=27)
    cache.prefill(eng, None, turn2, DEV, store=store, chunk=CH, resident=res)
    assert res.valid == 0 and not res.anchors


class Pos:
    """A drafter with a position-indexed cache that can resume in place, as DFlash2 can."""

    def __init__(self):
        self.buf = torch.zeros(512, 4)
        self.ctx_len = 0

    def reset(self):
        self.ctx_len = 0

    def sync(self, toks, hidden, first_pos, rows=None):
        n = len(toks)
        self.buf[first_pos:first_pos + n] = hidden[:n, :4].float()
        self.ctx_len = first_pos + n

    def state_resume(self, n):
        self.ctx_len = n

    def kv_views(self):
        return [(self.buf, 0)]


def test_the_drafter_picks_up_its_own_rows():
    turn1 = tokens(120, seed=28)
    res, eng, d = resident(), fresh(), Pos()
    cache.prefill(eng, d, turn1, DEV, chunk=CH, resident=res)
    d.buf[120:140] = 7.0                                  # the decode wrote past the prompt
    guest = tokens(10, seed=29)
    cache.prefill(eng, d, guest, DEV, chunk=CH, resident=res)
    turn2 = turn1 + tokens(20, seed=30)
    d.reset()
    cache.prefill(eng, d, turn2, DEV, chunk=CH, resident=res)
    ref, refd = fresh(), Pos()
    cache.prefill(ref, refd, turn2, DEV, store=None, chunk=CH)
    assert d.ctx_len == len(turn2)
    assert torch.equal(d.buf[:len(turn2)], refd.buf[:len(turn2)])


def test_a_drafter_that_cannot_resume_turns_it_off():
    class NeedsHidden:
        def sync(self, *a, **k):
            pass

    class Wrapped:
        def sync(self, *a, **k):
            pass

        def can_resume(self):
            return False

    assert cache.resident_capable(None) and cache.resident_capable(Pos())
    assert not cache.resident_capable(NeedsHidden()) and not cache.resident_capable(Wrapped())
    res, eng = resident(), fresh()
    ids = tokens(64, seed=31)
    cache.prefill(eng, NeedsHidden(), ids, DEV, chunk=CH, resident=res)
    _, reused, _ = cache.prefill(eng, NeedsHidden(), ids + [1], DEV, chunk=CH, resident=res)
    assert reused == 0 and not res.anchors


def test_clear_forgets_everything():
    res, eng = resident(), fresh()
    cache.prefill(eng, None, tokens(64, seed=32), DEV, chunk=CH, resident=res)
    res.clear()
    assert res.valid == 0 and not res.anchors and res.report()["anchors"] == []


def _main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")


if __name__ == "__main__":
    _main()
