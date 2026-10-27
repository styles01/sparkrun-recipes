"""The serving-time caches, on a random 4-layer model a laptop can run in fp32.

What is worth testing here is one claim and three properties.

THE CLAIM: a restore is exact. A request that resumes at a checkpoint must compute the same logits,
to the last bit, as one that forwarded the whole prefix under the same chunking. Not "the same
argmax", not "close" -- the same tensor, because the snapshot is the same bytes and the arithmetic
after it sees the same inputs. If that holds on this model it holds on the 27 B one, because the
thing being tested is the state layout and not the size: the same 48-layer recurrence, the same
convolution window, the same KV, the same partial rotary.

The properties: the store must never answer with a state that a DIFFERENT prefix produced (a hash
is a hint, the tokens are the answer); the byte budget must actually bound the memory; and the
suffix store must survive a restart and find what it was told.

Run: python tests/test_cache.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import time

# Triton kernels are a different arithmetic order for the same quantities and this file runs on a
# CPU. Same convention as tests/test_forward_tree.py.
for _k in ("NORM", "GDN", "HEAD", "ATTN", "GDNBLOCK", "GDNTREE", "GDNPRE"):
    os.environ.setdefault(f"QWEN38_FUSED_{_k}", "0")

import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine import cache  # noqa: E402
from engine.models.qwen3_5.adapter import Qwen38Engine  # noqa: E402  (the served class: its cold verbs)
from test_forward_tree import FakeWeights, tiny_config  # noqa: E402

DEV = "cpu"
DT = torch.float32


def fresh(seed: int = 3):
    cfg = tiny_config()
    eng = Qwen38Engine(cfg, FakeWeights(cfg, seed), max_len=512, device=DEV)
    eng.kv.k = eng.kv.k.to(DT)
    eng.kv.v = eng.kv.v.to(DT)
    eng.state.conv = eng.state.conv.to(DT)
    return eng


def tokens(n: int, seed: int = 0) -> list[int]:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(1, 96, (n,), generator=g).tolist()


# ------------------------------------------------------------------ 1. the hash is a hash

def test_prefix_hashes_are_prefixes():
    ids = tokens(64)
    h = cache.prefix_hashes(ids)
    assert len(h) == len(ids) + 1
    for i in (0, 1, 17, 63, 64):
        assert h[i] == cache.prefix_hashes(ids[:i])[i]
    # a different token at the end is a different hash for that length and the same for shorter
    other = list(ids)
    other[-1] ^= 1
    h2 = cache.prefix_hashes(other)
    assert h2[-1] != h[-1] and h2[-2] == h[-2]


def test_the_hash_memo_is_a_fresh_pass_for_every_prompt():
    memo = cache._PrefixHashMemo()
    a = tokens(300, 5)
    seqs = [a, a + tokens(20, 6), a[:150] + tokens(40, 7), [], tokens(9, 8), a]
    for ids in seqs:
        assert memo(ids) == cache.prefix_hashes(ids)


def test_extend_hashes_matches_a_fresh_pass():
    a, b = tokens(20, 1), tokens(9, 2)
    assert cache.extend_hashes(cache.prefix_hashes(a), b) == cache.prefix_hashes(a + b)


# ------------------------------------------------------------------ 2. a restore is exact

def test_restore_reproduces_the_state_bit_for_bit():
    eng = fresh()
    ids = tokens(40)
    with torch.no_grad():
        eng.forward(torch.tensor(ids), start=0, last_only=True)
    snap = cache.capture(eng)
    S, conv = eng.state.S.clone(), eng.state.conv.clone()
    k, v, L = eng.kv.k.clone(), eng.kv.v.clone(), eng.kv.length
    # scribble over everything, the way another request would
    with torch.no_grad():
        eng.reset()
        eng.forward(torch.tensor(tokens(33, seed=9)), start=0, last_only=True)
    cache.restore(eng, snap)
    assert torch.equal(eng.state.S, S)
    assert torch.equal(eng.state.conv, conv)
    assert torch.equal(eng.kv.k[:, :, :, :L, :], k[:, :, :, :L, :])
    assert torch.equal(eng.kv.v[:, :, :, :L, :], v[:, :, :, :L, :])
    assert eng.kv.length == L and eng.state.primed


def test_resumed_prefill_equals_a_cold_one_bit_for_bit():
    """THE claim. Same chunking on both sides, so the two are the same arithmetic."""
    ids = tokens(100, seed=4)
    chunk = 16
    eng = fresh()
    store = cache.StateStore(1 << 30, chunk=chunk)
    cold, reused, fwd = cache.prefill(eng, None, ids, DEV, store=store, chunk=chunk,
                                      checkpoint=True)
    assert (reused, fwd) == (0, len(ids))
    assert store.report()["entries"] >= 5

    # a second request with the same prompt: everything up to the last checkpoint is restored
    eng2 = fresh()
    warm, reused2, fwd2 = cache.prefill(eng2, None, ids, DEV, store=store, chunk=chunk,
                                        checkpoint=True)
    assert reused2 == 96, reused2          # 100 tokens, boundaries at 16..96
    assert fwd2 == 4
    assert torch.equal(cold, warm), (cold - warm).abs().max().item()
    assert torch.equal(eng.state.S, eng2.state.S)
    assert torch.equal(eng.kv.k[:, :, :, :100, :], eng2.kv.k[:, :, :, :100, :])


def test_a_longer_prompt_sharing_a_prefix_resumes_at_the_last_shared_boundary():
    """The shared-system-prompt case: same 64 tokens, different tail."""
    chunk = 16
    shared = tokens(64, seed=5)
    a = shared + tokens(20, seed=6)
    b = shared + tokens(25, seed=7)
    store = cache.StateStore(1 << 30, chunk=chunk)
    eng = fresh()
    cache.prefill(eng, None, a, DEV, store=store, chunk=chunk, checkpoint=True)

    eng2 = fresh()
    lg_warm, reused, fwd = cache.prefill(eng2, None, b, DEV, store=store, chunk=chunk,
                                         checkpoint=True)
    assert reused == 64 and fwd == 25
    eng3 = fresh()
    lg_cold, _, _ = cache.prefill(eng3, None, b, DEV, store=None, chunk=chunk)
    assert torch.equal(lg_cold, lg_warm)


def test_a_session_resumes_from_the_end_of_the_previous_turn():
    """Turn two forwards the new tokens and nothing else.

    And it is NOT bit-identical to a cold prefill of turn two, which is the one place in this file
    where that sentence is true. A session boundary is wherever the previous turn happened to stop
    -- 50 here -- and a cold prefill's chunk grid is anchored at 0, so the cold run computes
    positions 48..56 in one forward and the warm run computes 50..56 in one forward. The chunked
    delta rule blocks a 9-row call differently from a 7-row one and the two arithmetics part in the
    last bits. On the real engine there is a second, larger reason: the state a turn ENDS in was
    written by a speculative verify block, not by a prefill chunk, so it is the state that really
    produced the previous turn rather than the state a re-read of it would produce.

    So the gate for the session cache is the gate the whole engine is held to -- the same argmax --
    and not bitwise equality. `tools/cache_gate.py` measures it on the 27 B model.
    """
    chunk = 16
    turn1 = tokens(50, seed=11)
    store = cache.StateStore(1 << 30, chunk=chunk)
    eng = fresh()
    cache.prefill(eng, None, turn1, DEV, store=store, chunk=chunk, checkpoint=True)
    # what the server does at the end of a turn
    store.put(turn1, cache.capture(eng), conv_id="c1")

    turn2 = turn1 + tokens(7, seed=12)
    eng2 = fresh()
    warm, reused, fwd = cache.prefill(eng2, None, turn2, DEV, store=store, chunk=chunk)
    assert reused == 50 and fwd == 7
    eng3 = fresh()
    cold, _, _ = cache.prefill(eng3, None, turn2, DEV, store=None, chunk=chunk)
    assert int(cold.argmax(-1)) == int(warm.argmax(-1))
    rel = (cold - warm).abs().max().item() / cold.abs().max().item()
    assert rel < 1e-5, rel


def test_a_session_resume_at_a_grid_boundary_is_bit_exact():
    """The same case with the boundary on the chunk grid: then the tail splits identically."""
    chunk = 16
    turn1 = tokens(48, seed=11)
    store = cache.StateStore(1 << 30, chunk=chunk)
    eng = fresh()
    cache.prefill(eng, None, turn1, DEV, store=store, chunk=chunk, checkpoint=True)
    store.put(turn1, cache.capture(eng), conv_id="c1")
    turn2 = turn1 + tokens(7, seed=12)
    eng2 = fresh()
    warm, reused, fwd = cache.prefill(eng2, None, turn2, DEV, store=store, chunk=chunk)
    assert reused == 48 and fwd == 7
    eng3 = fresh()
    cold, _, _ = cache.prefill(eng3, None, turn2, DEV, store=None, chunk=chunk)
    assert torch.equal(cold, warm)


def test_a_full_prompt_hit_still_forwards_its_last_token():
    """A snapshot holds no logits, so the answer always costs at least one forward."""
    chunk = 8
    ids = tokens(32, seed=13)
    store = cache.StateStore(1 << 30, chunk=chunk)
    eng = fresh()
    cache.prefill(eng, None, ids, DEV, store=store, chunk=chunk, checkpoint=True)
    store.put(ids, cache.capture(eng))
    eng2 = fresh()
    _, reused, fwd = cache.prefill(eng2, None, ids, DEV, store=store, chunk=chunk)
    # the entry at 32 is a complete answer to the prompt and is still not usable: the logits the
    # request needs are the ones the 32nd token's own forward produces, and no snapshot holds
    # them. So the resume is the last boundary BELOW the prompt, which is 24.
    assert (reused, fwd) == (24, 8)


def test_the_lower_right_bias_computes_the_same_thing_as_the_boolean_mask():
    """The fused form of a chunk's mask has to BE the mask, not merely be faster.

    `engine/model.py` hands a prefill chunk at `start > 0` a `causal_lower_right` bias instead of a
    materialised boolean, because the boolean costs 1.7-1.9x on the board. The two are the same
    relation -- `ones(T, ctx).tril(ctx - T)` -- and this asserts it end to end rather than on the
    mask alone, at a threshold low enough for a 16-token chunk to reach it.
    """
    import importlib
    import engine.model as M
    ids = tokens(80, seed=81)
    prev = M.CHUNK_LOWER_RIGHT_FROM
    try:
        M.CHUNK_LOWER_RIGHT_FROM = 0            # the boolean everywhere
        a, _, _ = cache.prefill(fresh(), None, ids, DEV, chunk=16)
        M.CHUNK_LOWER_RIGHT_FROM = 4            # the fused bias from 4 rows up
        b, _, _ = cache.prefill(fresh(), None, ids, DEV, chunk=16)
    finally:
        M.CHUNK_LOWER_RIGHT_FROM = prev
        importlib.invalidate_caches()
    assert int(a.argmax(-1)) == int(b.argmax(-1))
    rel = (a - b).abs().max().item() / a.abs().max().item()
    assert rel < 1e-5, rel


# ------------------------------------------------------------------ 3. the store's own rules

def test_the_chunk_grid_is_anchored_at_zero_not_at_the_resume():
    """A session resume is off the grid; everything it checkpoints must be back on it.

    Otherwise the misalignment propagates: the next request resumes from an off-grid checkpoint,
    chunks from there, stores more off-grid boundaries, and the bit-exactness the prefix cache is
    supposed to have is gone for good after one conversation.
    """
    chunk = 16
    store = cache.StateStore(1 << 30, chunk=chunk)
    eng = fresh()
    turn1 = tokens(50, seed=71)
    cache.prefill(eng, None, turn1, DEV, store=store, chunk=chunk, checkpoint=True)
    store.put(turn1, cache.capture(eng), conv_id="c")          # the off-grid session boundary
    turn2 = turn1 + tokens(60, seed=72)
    eng2 = fresh()
    _, reused, _ = cache.prefill(eng2, None, turn2, DEV, store=store, chunk=chunk, checkpoint=True)
    assert reused == 50
    grid = [L for L in store.report()["boundaries"] if L % chunk]
    assert grid == [50], grid     # the session put is the only off-grid entry in the store


def test_a_collision_is_rejected_rather_than_answered():
    store = cache.StateStore(1 << 30, chunk=8)
    eng = fresh()
    a = tokens(24, seed=21)
    with torch.no_grad():
        eng.forward(torch.tensor(a), start=0, last_only=True)
    store.put(a, cache.capture(eng))
    # forge a lookup for a DIFFERENT prefix whose hash is claimed to be a's
    b = tokens(24, seed=22)
    forged = cache.prefix_hashes(a)
    assert store.find(b, forged, max_len=24) is None
    assert store.stats["rejected_collisions"] >= 1


def test_the_budget_bounds_the_memory():
    eng = fresh()
    with torch.no_grad():
        eng.forward(torch.tensor(tokens(40, seed=31)), start=0, last_only=True)
    one = cache.capture(eng).nbytes
    store = cache.StateStore(int(one * 2.5), chunk=8)
    for i in range(10):
        ids = tokens(40, seed=100 + i)
        store.put(ids, cache.capture(eng))
    r = store.report()
    assert r["bytes"] <= store.budget
    assert r["entries"] <= 3 and r["evictions"] >= 7


def test_capture_refuses_before_cloning():
    eng = fresh()
    a = tokens(40, seed=61)
    with torch.no_grad():
        eng.forward(torch.tensor(a), start=0, last_only=True)
    one = cache.capture(eng).nbytes
    assert cache.capture(eng, max_bytes=one // 4) is None, "the cap must refuse before cloning"
    assert cache.capture(eng, max_bytes=one * 3) is not None


def test_a_rolled_back_block_snapshots_as_the_prefix_it_kept():
    """`rollback_to` restored the recurrent state to the kept prefix and left
    `kv.length` at the end of the REJECTED block, and `capture` snapshots `kv.length` rows. A
    generation that ended on a chain block rejected at its last slot -- `kv.length == len(ctx)`,
    so `put` accepts it -- stored a snapshot keyed by `len(ctx)` tokens whose recurrent state had
    seen one fewer and whose last KV row was the rejected draft's. The next turn resuming from it
    was conditioned on a context nobody wrote. The snapshot after a rollback must be the kept
    prefix, exactly: resuming it must equal forwarding that prefix cold."""
    eng = fresh()
    a, blk = tokens(20, seed=71), tokens(6, seed=72)
    with torch.no_grad():
        eng.forward(torch.tensor(a), start=0, last_only=True)
        eng.forward_block(torch.tensor(blk), start=20)
        eng.rollback_to(3)
    assert eng.kv.length == 23, f"kv.length {eng.kv.length}: the kept prefix is 20 + 3"
    snap = cache.capture(eng)
    assert snap.length == 23
    warm, cold = fresh(), fresh()
    cache.restore(warm, snap)
    with torch.no_grad():
        x = warm.forward(torch.tensor([7]), start=23, last_only=True)
        cold.forward(torch.tensor(a + blk[:3]), start=0, last_only=True)
        y = cold.forward(torch.tensor([7]), start=23, last_only=True)
    assert torch.allclose(x, y, atol=1e-4, rtol=1e-4), (x - y).abs().max()

class _Arm:
    """A snapshottable drafter arm that says what its snapshot costs per token."""

    def __init__(self, per_token, block=8):
        self.per_token = per_token
        self.eng = type("E", (), {"tap": None})()
        self.cfg = type("C", (), {"block_size": block})()

    def state_snapshot(self):
        return ("arm",)

    def snapshot_bytes_per_token(self):
        return self.per_token


def test_the_snapshot_estimate_counts_every_arm_the_snapshot_carries():
    """The estimate added ONE arm's 20 kB a token, and the served length router snapshots
    two until the latch releases one. The drafter is asked, so 1-arm and 2-arm stacks both count."""
    from engine.drafters.dflash2 import DFlash2Drafter
    from engine.lenrouter import LengthRouter
    eng = fresh()
    kv = 2 * len(eng.cfg.attention_layers) * eng.cfg.num_key_value_heads * eng.cfg.head_dim * 2
    L = 1000

    def est(d):
        return cache._snapshot_estimate(eng, L, d)

    base = est(None)
    assert est(_Arm(21_000)) - base == int(1.25 * 21_000 * L), "one arm"
    router = LengthRouter(_Arm(21_000), _Arm(21_000, 16), latch=True, drop_idle=True,
                          learn_cost=False)
    assert est(router) - base == int(1.25 * 42_000 * L), "two arms until the latch releases one"
    router.idle = "s"
    assert est(router) - base == int(1.25 * 21_000 * L), "a released arm is not in the snapshot"
    # the real arm's figure is its own draft K and V per position
    arm = DFlash2Drafter.__new__(DFlash2Drafter)
    arm._ck = torch.zeros(5, 4, 64, 128, dtype=torch.bfloat16)
    assert arm.snapshot_bytes_per_token() == 2 * 5 * 4 * 128 * 2
    arm._ck = None
    assert arm.snapshot_bytes_per_token() == 0
    assert kv > 0

def test_a_single_snapshot_over_the_entry_cap_is_declined():
    """The 2026-09-19 wedge: a multi-GB boundary snapshot cloned before evictions run pushed the
    board to the memory edge. `put` must refuse an entry bigger than the cap the server sets."""
    eng = fresh()
    a = tokens(40, seed=51)
    with torch.no_grad():
        eng.forward(torch.tensor(a), start=0, last_only=True)
    one = cache.capture(eng).nbytes
    store = cache.StateStore(int(one * 2.5), chunk=8, max_entry_bytes=one - 1)
    store.put(a, cache.capture(eng))
    r = store.report()
    assert r["entries"] == 0 and r["declined_big"] == 1
    ok = cache.StateStore(int(one * 2.5), chunk=8, max_entry_bytes=one)
    ok.put(a, cache.capture(eng))
    assert ok.report()["entries"] == 1, "an entry within the cap is accepted"


def test_the_recurrent_state_is_the_bulk_of_a_short_snapshot():
    """The sizing argument this cache is budgeted by, asserted rather than assumed."""
    eng = fresh()
    with torch.no_grad():
        eng.forward(torch.tensor(tokens(16, seed=41)), start=0, last_only=True)
    parts = cache.capture(eng).parts()
    assert parts["recurrent"] > parts["kv"]


# ------------------------------------------------------------------ 4. drafters

class _FakeDrafter:
    """A drafter with a position-indexed cache, which is the thing that must not get a hole."""

    def __init__(self):
        self.buf = torch.zeros(256, 4)
        self.ctx_len = 0

    def sync(self, tokens, hidden, first_pos, rows=None):
        n = len(tokens)
        self.buf[first_pos:first_pos + n] = torch.tensor(
            [[float(t)] * 4 for t in tokens])
        self.ctx_len = first_pos + n

    def state_snapshot(self):
        return ("fake", self.ctx_len, self.buf[:self.ctx_len].clone())

    def state_restore(self, snap):
        _, n, buf = snap
        self.buf[:n] = buf
        self.ctx_len = n


def test_the_drafter_cache_crosses_a_restore_without_a_hole():
    chunk = 8
    ids = tokens(40, seed=51)
    store = cache.StateStore(1 << 30, chunk=chunk)
    eng, d = fresh(), _FakeDrafter()
    cache.prefill(eng, d, ids, DEV, store=store, chunk=chunk, checkpoint=True)
    want = d.buf[:40].clone()

    eng2, d2 = fresh(), _FakeDrafter()
    _, reused, _ = cache.prefill(eng2, d2, ids, DEV, store=store, chunk=chunk, checkpoint=True)
    assert reused > 0
    assert d2.ctx_len == 40
    assert torch.equal(d2.buf[:40], want)


def test_a_drafter_that_cannot_snapshot_turns_the_cache_off():
    class NeedsHidden:
        def sync(self, *a, **k):
            pass

    class Lookup:
        def prime(self, toks):
            pass

    assert cache.drafter_is_cacheable(None)
    assert cache.drafter_is_cacheable(Lookup())
    assert cache.drafter_is_cacheable(_FakeDrafter())
    assert not cache.drafter_is_cacheable(NeedsHidden())


# ------------------------------------------------------------------ 5. response cache

def test_the_response_cache_keys_on_prompt_and_params():
    rc = cache.ResponseCache(1 << 20, ttl_s=60)
    ids = tokens(12, seed=61)
    k1 = cache.ResponseCache.key(ids, max_new=64, stops=())
    k2 = cache.ResponseCache.key(ids, max_new=128, stops=())
    k3 = cache.ResponseCache.key(ids[:-1] + [ids[-1] ^ 1], max_new=64, stops=())
    assert rc.get(k1) is None
    rc.put(k1, [1, 2, 3], ids)
    assert rc.get(k1) == [1, 2, 3]
    assert rc.get(k2) is None and rc.get(k3) is None
    assert rc.report()["hits"] == 1


def test_the_response_cache_expires_and_is_bounded():
    rc = cache.ResponseCache(1 << 20, ttl_s=1e-6)
    k = cache.ResponseCache.key([1, 2], max_new=8)
    rc.put(k, [9], [1, 2])
    # past the TTL before the read: on a fast CPU the read came within the microsecond and hit
    time.sleep(0.002)
    assert rc.get(k) is None and rc.report()["expired"] == 1

    rc = cache.ResponseCache(200, ttl_s=0)
    for i in range(20):
        rc.put(cache.ResponseCache.key([i], max_new=8), [i] * 4, [i])
    assert rc.bytes <= 200 and rc.report()["evictions"] > 0


# ------------------------------------------------------------------ 6. the persistent store

def test_the_suffix_store_finds_what_it_was_told_and_survives_a_restart():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "suffix")
        st = cache.PersistentSuffixStore(path, rebuild_every=1 << 30).open()
        seq = list(range(200, 260))
        st.append(seq * 3)
        st.rebuild(background=False)
        n, pos = st.lookup([200, 201, 202, 203, 204, 205, 206, 207], min_order=4)
        assert n == 8 and pos
        assert st.continuation(pos[0], 3) == [208, 209, 210]

        # a restart reads the log back and reindexes
        st2 = cache.PersistentSuffixStore(path, rebuild_every=1 << 30).open()
        n2, pos2 = st2.lookup([200, 201, 202, 203, 204, 205, 206, 207], min_order=4)
        assert n2 == 8 and pos2
        assert st2.report()["tokens"] == st.report()["tokens"]


def test_a_prefill_without_the_prefix_cache_is_still_chunked():
    """With the prefix cache off a long prompt used to be ONE forward of all its rows."""
    assert cache.prefill_chunk(False, 1024, 8192) == 8192
    assert cache.prefill_chunk(True, 1024, 8192) == 1024          # the cache's grid, unchanged
    assert cache.prefill_chunk(False, 1024, 0) == 0               # the old single call, on request


def test_a_readonly_store_reads_what_is_there_and_writes_nothing():
    """A benchmark measured against real traffic's store must not write itself into it."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "suffix")
        st = cache.PersistentSuffixStore(path, rebuild_every=1 << 30).open()
        st.append(list(range(300, 320)))
        st.rebuild(background=False)
        size = os.path.getsize(os.path.join(path, "tokens.bin"))
        ro = cache.PersistentSuffixStore(path, rebuild_every=1, readonly=True).open()
        n, pos = ro.lookup(list(range(300, 308)), min_order=4)
        assert n == 8 and pos
        ro.append(list(range(900, 950)))
        ro.rebuild(background=False)
        assert os.path.getsize(os.path.join(path, "tokens.bin")) == size
        assert ro.lookup(list(range(900, 908)), min_order=4)[0] < 4


def test_a_continuation_stops_at_the_document_boundary():
    with tempfile.TemporaryDirectory() as d:
        st = cache.PersistentSuffixStore(os.path.join(d, "s"), rebuild_every=1 << 30).open()
        st.append([11, 12, 13, 14, 15, 16, 17, 18])
        st.append([50, 51, 52])
        st.rebuild(background=False)
        n, pos = st.lookup([11, 12, 13, 14, 15, 16, 17, 18], min_order=4)
        assert n == 8
        # the match ends the first document; nothing from the second may follow it
        assert st.continuation(pos[0], 4) == []


def test_the_store_set_asks_everyone_and_keeps_the_positions_apart():
    with tempfile.TemporaryDirectory() as d:
        a = cache.PersistentSuffixStore(os.path.join(d, "a"), rebuild_every=1 << 30).open()
        b = cache.PersistentSuffixStore(os.path.join(d, "b"), rebuild_every=1 << 30).open()
        a.append([1, 2, 3, 4, 5, 6, 7, 8, 9])
        b.append([1, 2, 3, 4, 5, 6, 7, 8, 90, 91])
        a.rebuild(background=False)
        b.rebuild(background=False)
        s = cache.SuffixStoreSet([a, b])
        n, pos = s.lookup([1, 2, 3, 4, 5, 6, 7, 8], min_order=4)
        assert n == 8 and len(pos) == 2
        conts = sorted(s.continuation(p, 2) for p in pos)
        assert conts == [[9], [90, 91]], conts


def test_the_lookup_drafter_reads_the_persistent_store():
    from engine.drafters.ngram import NgramDrafter
    with tempfile.TemporaryDirectory() as d:
        st = cache.PersistentSuffixStore(os.path.join(d, "s"), rebuild_every=1 << 30).open()
        body = list(range(300, 340))
        st.append(body * 4)
        st.rebuild(background=False)
        drafter = NgramDrafter(corpus_path="", min_order=4, min_corpus_order=4, min_expected=0.2)
        drafter.add_store(st)
        drafter.prime([1, 2, 3])                       # nothing in the LOCAL index matches
        got = drafter.propose([300, 301, 302, 303, 304, 305, 306, 307], 4)
        assert got == [308, 309, 310, 311], got


def test_the_size_cap_forgets_the_oldest():
    with tempfile.TemporaryDirectory() as d:
        st = cache.PersistentSuffixStore(os.path.join(d, "s"), max_tokens=200,
                                         rebuild_every=1 << 30).open()
        for i in range(20):
            st.append(list(range(1000 + 40 * i, 1040 + 40 * i)))
        st.rebuild(background=False)
        assert st.report()["tokens"] <= 200
        assert st.stats["trims"] >= 1
        # the newest document is still findable
        n, pos = st.lookup(list(range(1760, 1768)), min_order=4)
        assert n == 8 and pos


def _main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")


if __name__ == "__main__":
    _main()
