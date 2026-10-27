"""SRV-307, the shared KV pool (engine/kvpool.py), CPU only: the accounts and the policy on a fake
MemAvailable, with the buffers' bookkeeping fallback (no CUDA driver: a plain tensor, pages
counted, nothing mapped).

  * a buffer commits whole pages, per (layer, head) region, rounded up to where the pages end;
    `need_bytes` says what a commit would add before it is made;
  * `trim` gives back every page past what it keeps; the pool's account follows each owner;
  * the commit gate passes above the hard floor, calls the reclaimers in order until it does,
    stops at the first that is enough, and refuses (`PoolFull`, nothing committed) when nothing
    more comes back;
  * a set grows all its buffers together with the margin; `ensure_for` finds the set of a tensor;
  * the engine's KV cache over the pool has the static buffer's shape, dtype and strides, and its
    `ensure` is a no-op without a pool.

Run: python tests/test_kvpool.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch  # noqa: E402

from engine import kvpool  # noqa: E402

GiB, MiB = kvpool.GiB, kvpool.MiB


class FakeMem:
    def __init__(self, avail: int):
        self.avail = int(avail)

    def __call__(self) -> int:
        return self.avail


def _pool(avail=100 * GiB, floor=10 * GiB, margin=0):
    mem = FakeMem(avail)
    return kvpool.KVPool(hard_floor=floor, margin_rows=margin, meminfo=mem), mem


def _engine_shape(rows=65536):
    return (16, 1, 4, rows, 256)          # 512 B a row: 4,096 rows a 2 MiB page per region


def test_commit_rounds_to_pages_per_region():
    pool, _ = _pool()
    b = kvpool.PagedBuffer(_engine_shape(), 3, torch.bfloat16, "cpu", owner="s", pool=pool)
    assert b.page == 2 * MiB and b.row_bytes == 512 and b.outer == 64
    assert b.need_bytes(1) == 64 * 2 * MiB, "one row needs a page in each of the 64 regions"
    got = b.ensure(3000)
    assert got == 64 * 2 * MiB and b.rows == 4096
    assert b.ensure(4000) == 0, "inside the committed pages: nothing new"
    assert b.need_bytes(4097) == 64 * 2 * MiB
    assert b.ensure(10000) == 2 * 64 * 2 * MiB and b.rows == 12288
    assert pool.owner_bytes("s") == 3 * 64 * 2 * MiB == b.committed_bytes()
    assert pool.stats["pages_mapped"] == 3 * 64


def test_trim_gives_back_past_the_kept_rows():
    pool, _ = _pool()
    b = kvpool.PagedBuffer(_engine_shape(), 3, torch.bfloat16, "cpu", owner="s", pool=pool)
    b.ensure(20000)                        # 5 pages a region
    assert b.trim(5000) == 3 * 64 * 2 * MiB, "keeps the 2 pages a region that hold row 4,999"
    assert b.rows == 8192 and pool.owner_bytes("s") == 2 * 64 * 2 * MiB
    assert b.trim(0) == 2 * 64 * 2 * MiB and b.rows == 0 and pool.committed == 0
    assert b.trim(0) == 0


def test_window_cap_and_odd_layouts():
    pool, _ = _pool()
    # 1,000-row window, 256 B rows: regions of 250 KiB share pages
    b = kvpool.PagedBuffer((5, 8, 1000, 64), 2, torch.bfloat16, "cpu", owner="d", pool=pool)
    total_pages = -(-b.nbytes // b.page)
    b.ensure(10 ** 9)
    assert b.rows == 1000 and b.committed_bytes() == total_pages * b.page
    b.trim(0)
    assert b.committed_bytes() == 0


def test_gate_refuses_under_the_hard_floor_and_commits_nothing():
    pool, mem = _pool(avail=11 * GiB, floor=10 * GiB)
    b = kvpool.PagedBuffer(_engine_shape(), 3, torch.bfloat16, "cpu", owner="s", pool=pool)
    b.ensure(4096)                         # 128 MiB: 11 GiB - 0.125 >= 10
    mem.avail = int(10.1 * GiB)
    try:
        b.ensure(8192)                     # another 128 MiB would leave 9.98 GiB
        raise AssertionError("PoolFull expected")
    except kvpool.PoolFull as e:
        assert "hard floor" in str(e)
    assert b.rows == 4096 and pool.owner_bytes("s") == 64 * 2 * MiB
    assert pool.stats["refused"] == 1


def test_reclaimers_run_in_order_and_stop_when_enough():
    pool, mem = _pool(avail=int(10.05 * GiB), floor=10 * GiB)
    calls = []

    def caches(want, owner):
        calls.append(("caches", owner))
        mem.avail += 64 * MiB              # not enough on its own
        return 64 * MiB

    def side(want, owner):
        calls.append(("side", owner))
        mem.avail += 2 * GiB
        return 2 * GiB

    def slot0(want, owner):
        calls.append(("slot0", owner))
        return 0

    for name, fn in (("caches", caches), ("idle_side", side), ("idle_slot0", slot0)):
        pool.add_reclaimer(name, fn)
    b = kvpool.PagedBuffer(_engine_shape(), 3, torch.bfloat16, "cpu", owner="slot2", pool=pool)
    b.ensure(4096)
    assert calls == [("caches", "slot2"), ("side", "slot2")], calls
    assert b.rows == 4096 and pool.stats["reclaims"] == 2
    assert pool.stats["reclaimed_bytes"] == 64 * MiB + 2 * GiB


def test_a_failing_reclaimer_does_not_stop_the_others():
    pool, mem = _pool(avail=int(10.05 * GiB), floor=10 * GiB)
    events = []
    pool.on_event = lambda k, kw: events.append(k)

    def broken(want, owner):
        raise RuntimeError("boom")

    def ok(want, owner):
        mem.avail += GiB
        return GiB

    pool.add_reclaimer("broken", broken)
    pool.add_reclaimer("ok", ok)
    b = kvpool.PagedBuffer(_engine_shape(), 3, torch.bfloat16, "cpu", owner="s", pool=pool)
    b.ensure(4096)
    assert "reclaim_error" in events and "reclaim" in events and b.rows == 4096


def test_set_grows_together_with_the_margin_and_ensure_for_finds_it():
    pool, _ = _pool(margin=256)
    ps = kvpool.PagedSet("slot1", pool=pool)
    k = ps.add(kvpool.PagedBuffer(_engine_shape(), 3, torch.bfloat16, "cpu", pool=pool))
    d = ps.add(kvpool.PagedBuffer((5, 8, 65536, 64), 2, torch.bfloat16, "cpu", pool=pool))
    assert d.owner == "slot1"
    ps.ensure(4000)                        # 4,256 with the margin: 2 engine pages a region, 1 draft page (16,384 rows of 128 B)
    eng_b, dr_b = ps.bufs
    assert eng_b.rows == 8192 and dr_b.rows == 16384 and ps.rows == 8192
    assert ps.need_bytes(4000) == 0
    kvpool.POOL = pool
    try:
        kvpool.ensure_for(k.tensor, 9000)
        assert ps.rows >= 9000 + 256
        kvpool.ensure_for(torch.zeros(4), 10 ** 6)     # a static tensor: nothing
    finally:
        kvpool.POOL = None
    got = ps.trim(100)
    assert got > 0 and ps.rows == 4096
    assert pool.owner_bytes("slot1") == ps.committed_bytes()
    ps.set_owner("slot3")
    assert pool.owner_bytes("slot1") == 0 and pool.owner_bytes("slot3") == ps.committed_bytes()


def test_report_numbers():
    pool, _ = _pool(avail=50 * GiB)
    b = kvpool.PagedBuffer(_engine_shape(), 3, torch.bfloat16, "cpu", owner="slot0", pool=pool)
    b.ensure(4096)
    r = pool.report()
    assert r["committed_gib"] == 0.125 and r["by_owner_gib"] == {"slot0": 0.125}
    assert r["reserved_va_gib"] == 2.0 and r["buffers"] == 1 and r["mem_available_gib"] == 50.0
    assert r["peak_committed"] == 128 * MiB


def test_kv_cache_over_the_pool_keeps_the_static_layout():
    from engine.model import KVCache

    class Cfg:
        attention_layers = [3, 7]
        num_key_value_heads = 4
        head_dim = 256

    static = KVCache(Cfg, 32768, "cpu", paged=False)
    assert static.pages is None
    static.ensure(10 ** 6)                 # no pool: a no-op
    pool, _ = _pool(margin=256)
    kvpool.POOL = pool
    try:
        paged = KVCache(Cfg, 32768, "cpu", paged=True)
    finally:
        kvpool.POOL = None
    assert paged.pages is not None
    assert paged.k.shape == static.k.shape and paged.k.stride() == static.k.stride()
    assert paged.k.dtype == static.k.dtype
    paged.ensure(100)
    assert paged.pages.rows == 4096 and pool.owner_bytes("slot0") == 2 * 2 * 4 * 2 * MiB
    paged.ensure(200)                      # inside: the fast path
    assert pool.stats["commits"] == 2      # one per buffer (k, v), the first time only


def test_env_switch():
    old = os.environ.get("QSE_KV_POOL")
    try:
        os.environ["QSE_KV_POOL"] = "on"
        assert kvpool.env_on()
        os.environ["QSE_KV_POOL"] = "off"
        assert not kvpool.env_on()
    finally:
        if old is None:
            os.environ.pop("QSE_KV_POOL", None)
        else:
            os.environ["QSE_KV_POOL"] = old


if __name__ == "__main__":
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            n += 1
            print(f"ok  {name}")
    print(f"{n} passed")
