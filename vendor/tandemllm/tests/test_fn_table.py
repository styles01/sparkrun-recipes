"""ENG-239: the n-gram table's policies (engine/models/flashnext/ple.py), no GPU.

  * cold rows (`DirectRows`: O_DIRECT where the filesystem has it, through the hot-row cache) equal
    warm rows (the page cache) bit for bit on 100,000 random ids, repeated ids and both ends of the
    file, through a cache small enough to evict;
  * the cache: a second read of the same ids is all hits; CLOCK evicts within a set, never a row
    referenced since the hand last passed while another way is free;
  * the policy switch and the report (`report`: policy, file bytes, resident bytes, cache stats);
  * `page_cache_bytes` of a file read through the page cache is its size, of one dropped from it
    less (where the platform can say).
"""

from __future__ import annotations

import os
import shutil
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.models.flashnext import pack as P  # noqa: E402
from engine.models.flashnext.ple import DirectRows, page_cache_bytes  # noqa: E402
from test_fn_pack import mini  # noqa: E402


def test_cold_rows_equal_warm_rows():
    d, info = mini()
    try:
        tab = P.PackIndex.open(d).table
        rng = np.random.default_rng(3)
        ids = np.concatenate([rng.integers(0, tab.rows, 100_000), [0, tab.rows - 1, 5, 5, 5]])
        warm = tab.rows_of(ids)
        assert np.array_equal(warm, info["packed"].numpy()[ids])
        tab.set_policy("cold", cache_bytes=64 * 1024)               # a cache far smaller than the ids
        assert tab.policy == "cold"
        cold = tab.rows_of(ids)
        assert np.array_equal(cold, warm)
        st = tab.direct.stats
        assert st["misses"] > 0 and st["blocks_read"] == st["misses"]
        packed = tab.gather_packed(torch.from_numpy(ids[:600]))
        assert torch.equal(packed, info["packed"][torch.from_numpy(ids[:600])])
        rep = tab.report()
        assert rep["policy"] == "cold" and rep["file_bytes"] == tab.rows * tab.row_bytes
        tab.set_policy("warm")
        assert tab.policy == "warm" and tab.direct is None
        return (f"100,005 ids ({tab.rows} rows of {tab.row_bytes} bytes): cold equal to warm, "
                f"{st['hits']} hits / {st['misses']} misses, O_DIRECT {rep['direct']}")
    finally:
        shutil.rmtree(d)


def test_the_hot_row_cache_hits_and_clock_evicts():
    d, info = mini()
    try:
        tab = P.PackIndex.open(d).table
        dr = DirectRows(tab.path, tab.offset, tab.row_bytes, tab.rows, cache_bytes=1 << 20)
        ids = np.arange(0, 2000, 7)
        dr.rows_of(ids)
        h0 = dr.stats["hits"]
        assert np.array_equal(dr.rows_of(ids), info["packed"].numpy()[ids])
        assert dr.stats["hits"] - h0 == ids.size                    # every row cached the first time
        # one set: fill its ways, touch one, add one more -- the untouched ways go first
        one = DirectRows(tab.path, tab.offset, tab.row_bytes, tab.rows, cache_bytes=tab.row_bytes * DirectRows.WAYS)
        assert one.sets == 1
        first = np.arange(DirectRows.WAYS)
        one.rows_of(first)                                          # 4 misses, every bit set
        one.rows_of(np.array([100]))                                # the hand clears the set, takes way 0
        assert 100 in one.keys[0] and int(first[0]) not in one.keys[0]
        one.rows_of(np.array([1]))                                  # a hit: its bit set again
        one.rows_of(np.array([200]))                                # the first clear way: not row 1's
        assert 1 in one.keys[0] and 200 in one.keys[0]
        assert np.array_equal(one.rows_of(np.array([100, 200, 1])), info["packed"].numpy()[[100, 200, 1]])
        dr.close()
        one.close()
        return "a second read of 286 rows all hits; CLOCK keeps a referenced row"
    finally:
        shutil.rmtree(d)


def test_page_cache_residency():
    d, _ = mini()
    try:
        p = os.path.join(d, P.TABLE_FILE)
        size = os.path.getsize(p)
        with open(p, "rb") as f:
            f.read()                                                # through the page cache
        got = page_cache_bytes(p, 0, size)
        if got is None:
            return "skipped: the platform cannot report page-cache residency"
        assert got >= size - 4096, (got, size)
        fd = os.open(p, os.O_RDONLY)
        os.fsync(fd)                                                # dirty pages are not dropped
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)
        after = page_cache_bytes(p, 0, size)
        assert after < got, (after, got)
        return f"{got} of {size} bytes resident after a read, {after} after DONTNEED"
    finally:
        shutil.rmtree(d)


if __name__ == "__main__":
    fails = 0
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        try:
            print(f"  {name:<46} ok   {fn() or ''}")
        except AssertionError as e:
            fails += 1
            print(f"  {name:<46} FAIL {e}")
    print(f"{len(tests) - fails} passed" + (f", {fails} FAILED" if fails else ""))
    sys.exit(1 if fails else 0)
