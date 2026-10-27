"""ENG-225: the host side of Flash-Next's prefill kernels, without a GPU.

  * `group_plan` (the pairs of a chunk grouped by expert): every pair exactly once, in ascending
    expert id, stable inside an expert; per expert its count, its rows in the padded layout (starting
    at a multiple of 16, back to back in sorted order) and the items of up to `rows` pairs (an expert
    with no pair has none), the bounds never below the rows and items;
  * `FlashNextEngine._pre_ctl` (what every layer of a chunk shares): the GDN and PLE conv taps along
    the chain are `window_src` of a chain (distances 3, 2, 1 over a state of 3; 9, 6, 3 over 9); each
    row's attention control is the decode step's at its position (dense up to 512 complete blocks,
    then 2,048 selected positions and the tail); every complete block sits in the shared pool, the
    chunk pools the blocks it completes; the context class is the last row's.
"""

from __future__ import annotations

import os
import random
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.fn_seq_kernels import window_src  # noqa: E402


def test_group_plan():
    from tools.fn_prefill_kernels import group_plan
    rng = random.Random(5)
    for M, S, E, rows in ((1, 11, 513, 64), (37, 11, 25, 32), (300, 5, 9, 64), (16, 3, 40, 32)):
        ids = torch.tensor([[rng.randrange(E - 1) for _ in range(S - 1)] + [E - 1] for _ in range(M)], dtype=torch.int32)
        g = group_plan(ids.reshape(-1), E, rows)
        order, eid, cnt, pstart, prow, iend = (g[k] for k in ("order", "eid", "cnt", "pstart", "prow", "iend"))
        flat = ids.reshape(-1)
        assert sorted(order.tolist()) == list(range(M * S))
        assert torch.equal(eid, flat[order].int()) and bool((eid[1:] >= eid[:-1]).all())
        start = 0
        for e in range(E):
            idx = order[flat[order] == e]
            assert bool((idx[1:] > idx[:-1]).all())
            n = int((flat == e).sum())
            assert int(cnt[e]) == n and int(pstart[e]) == start and start % 16 == 0
            assert prow[eid == e].tolist() == list(range(start, start + n))
            start += -(-n // 16) * 16
            assert int(iend[e]) - (int(iend[e - 1]) if e else 0) == -(-n // rows)
        assert int(cnt[E - 1]) == M and int(iend[-1]) <= g["items"] and start <= g["R"] and g["R"] % 16 == 0
    return "every pair once, expert order, stable; counts, padded rows, items, bounds"


class _Cfg:
    has_indexer = True
    indexer_compress_ratio = 4
    indexer_head_dim = 128
    linear_conv_kernel_dim = 4
    ple_conv_kernel_size = 4
    ngram_size = 3


def _eng():
    from engine.models.flashnext.engine import FlashNextEngine
    return type("E", (), {"cfg": _Cfg(), "device": "cpu", "dtype": torch.bfloat16, "rope_cos": torch.zeros(1),
                          "rope_sin": torch.zeros(1), "qsa_class": FlashNextEngine.qsa_class,
                          "_rot": FlashNextEngine._rot})()


def test_pre_ctl():
    from engine.models.flashnext.engine import FlashNextEngine
    from engine.models.flashnext.ref import FNState
    eng = _eng()
    for L0, T in ((0, 1), (0, 7), (5, 300), (1500, 2048), (2045, 9), (3001, 128), (8190, 2048)):
        P = FlashNextEngine._pre_ctl(eng, L0, T, FNState())
        chain = [-1] + list(range(T - 1))
        assert P["csrc"].tolist() == window_src(chain, (3, 2, 1), 3), (L0, T)
        assert P["psrc"].tolist() == window_src(chain, (9, 6, 3), 9), (L0, T)
        for m in range(T):
            p = L0 + m
            nblk = (p + 1) // 4
            n, Lb, dense = P["ctl"][m, :3].tolist()
            assert dense == int(nblk <= 512) and Lb > p
            assert n == (p + 1 if nblk <= 512 else 2048 + p + 1 - 4 * nblk), (L0, T, m)
            assert int(P["nblk"][m]) == nblk
            assert P["tail"][m].tolist() == [4 * nblk + k for k in range(3)]
        b0, b1 = P["pool"]
        assert (b0, b1) == (L0 // 4, (L0 + T) // 4)
        assert P["pool_slots"].tolist() == [[4 * b + k for k in range(4)] for b in range(b0, b1)]
        assert P["pool_pos"].tolist() == [4 * b for b in range(b0, b1)]
        assert P["rpos"].tolist() == list(range(L0, L0 + T)) and P["cos"] is eng.rope_cos   # a text state
        assert P["cls"] == FlashNextEngine.qsa_class(eng, L0 + T - 1)
    return "chain conv taps = window_src; per-row dense/sparse counts, tails, pool blocks, class"


if __name__ == "__main__":
    fails = 0
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        try:
            print(f"  {name:<44} ok   {fn() or ''}")
        except AssertionError as e:
            fails += 1
            print(f"  {name:<44} FAIL {e}")
    print(f"{len(tests) - fails} passed" + (f", {fails} FAILED" if fails else ""))
    sys.exit(1 if fails else 0)
