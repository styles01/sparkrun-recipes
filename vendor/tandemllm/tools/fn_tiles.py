"""ENG-223/151: the EXL3 skinny GEMM's split for Flash-Next's dense shapes, measured cold, one row.

    python tools/fn_tiles.py --json ops/fn-exl3-tiles.json

For every dense projection shape of the packs (GDN q|k|v|z as one group, out_proj, attention
q|k|v, o_proj, the indexer, the head) a ring of distinct random blocks larger than L2 is called
once each inside a CUDA graph at M = 1, for every (cfg, S, wk) the kernel builds; the fastest is
kept if it beats the default split (`exl3_linear.default_tile`) by more than `--min-gain`. S and wk
set the summation order, so a kept entry is a different (row-invariant) arithmetic: the table is
Flash-Next's own and is applied by its loader (`engine/models/flashnext/load.py`), never to the 27B.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools import exl3_linear as X  # noqa: E402

# (N, Kin, K) of the 3.05 and 4.05 packs' dense projections (K5 / K6 dense, K3 / K4 indexer)
SHAPES = [(16384, 2560, 5), (2560, 6144, 5), (13312, 2560, 5), (2560, 6144, 6), (16384, 2560, 6),
          (13312, 2560, 6), (640, 2560, 3), (640, 2560, 4), (248320, 2560, 5), (248320, 2560, 6)]


def ring_us(blocks, cfg) -> float:
    xs = torch.randn(1, blocks[0].K, device="cuda").to(torch.bfloat16)
    outs = [torch.empty(1, b.N, dtype=torch.float32 if b.out_fp32 else torch.bfloat16, device="cuda") for b in blocks]
    for b, o in zip(blocks, outs):
        X.exl3_matmul(xs, b, out=o, cfg=cfg["cfg"], S=cfg["S"], wk=cfg["wk"])
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for b, o in zip(blocks, outs):
            X.exl3_matmul(xs, b, out=o, cfg=cfg["cfg"], S=cfg["S"], wk=cfg["wk"])
    g.replay()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(10):
        g.replay()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) * 1e3 / 10 / len(blocks)


def ring_us_dense(blocks, split) -> float:
    """`ring_us` for tools/fn_kernels.dense at a (S, warps) split (the input rotation inside)."""
    from tools import fn_kernels as K
    b0 = blocks[0]
    K.DENSE_SPLIT[(b0.tile_n, b0.K, b0.kb)] = split
    xs = torch.randn(1, b0.K, device="cuda").to(torch.bfloat16)
    outs = [torch.empty(1, b.N, dtype=torch.float32 if b.out_fp32 else torch.bfloat16, device="cuda") for b in blocks]
    for b, o in zip(blocks, outs):
        K.dense(xs, b, out=o)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for b, o in zip(blocks, outs):
            K.dense(xs, b, out=o)
    g.replay()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(10):
        g.replay()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) * 1e3 / 10 / len(blocks)


def sweep_dense(a) -> None:
    """--dense: the split of tools/fn_kernels.dense per shape, against the library path (the
    rotation launch included), cold ring, M = 1 -> {"NxKinxK": {"S", "warps", "us", "us_exl3"}}."""
    from tools import fn_kernels as K
    shapes = SHAPES if a.shapes == "all" else [tuple(int(v) for v in s.split("x")) for s in a.shapes.split(",")]
    out = {"_about": "Flash-Next dense EXL3 splits for tools/fn_kernels.dense (tools/fn_tiles.py --dense, M = 1, "
                     f"cold ring, CUDA graph), {time.strftime('%Y-%m-%d %H:%M')}; us_exl3 = exl3_matmul with its "
                     "rot_in at its own best split; applied by engine/models/flashnext/load.py"}
    for N, Kin, kb in shapes:
        one = X.random_block(N, Kin, kb, seed=1).nbytes
        n = max(2, int(a.ring_mb * 1e6 // one)) if N < 100000 else 2
        blocks = [X.random_block(N, Kin, kb, seed=s) for s in range(n)]
        if N > 100000:
            for b in blocks:
                b.out_fp32 = True
        lib = ring_us(blocks, X.pick(N, Kin, kb))
        best, best_us = None, 1e30
        for S in (1, 2, 4, 8):
            if (Kin // 16) % (8 * S) or (N // 128) * S > 4096:
                continue
            for w in (4, 8):
                us = ring_us_dense(blocks, (S, w))
                if us < best_us:
                    best, best_us = (S, w), us
        K.DENSE_SPLIT.pop((N, Kin, kb), None)
        print(f"{N}x{Kin}x{kb}: dense {best} {best_us:.1f} us ({one / best_us / 1e3:.1f} GB/s) vs exl3_matmul "
              f"{lib:.1f} us", flush=True)
        out[f"{N}x{Kin}x{kb}"] = {"S": best[0], "warps": best[1], "us": round(best_us, 2), "us_exl3": round(lib, 2),
                                  "gb_s": round(one / best_us / 1e3, 1)}
        del blocks
        torch.cuda.empty_cache()
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    ap.add_argument("--ring-mb", type=float, default=256)
    ap.add_argument("--min-gain", type=float, default=0.02)
    ap.add_argument("--shapes", default="all")
    ap.add_argument("--dense", action="store_true", help="sweep tools/fn_kernels.dense instead")
    a = ap.parse_args()
    if a.dense:
        return sweep_dense(a)
    shapes = SHAPES if a.shapes == "all" else [tuple(int(v) for v in s.split("x")) for s in a.shapes.split(",")]
    out = {"_about": "Flash-Next EXL3 dense splits (tools/fn_tiles.py, M = 1, cold ring, CUDA graph), "
                     f"{time.strftime('%Y-%m-%d %H:%M')}; applied by engine/models/flashnext/load.py"}
    for N, Kin, kb in shapes:
        one = X.random_block(N, Kin, kb, seed=1).nbytes
        n = max(2, int(a.ring_mb * 1e6 // one)) if N < 100000 else 2
        blocks = [X.random_block(N, Kin, kb, seed=s) for s in range(n)]
        if N > 100000:
            for b in blocks:
                b.out_fp32 = True
        d = X.default_tile(N, Kin, kb)
        base = ring_us(blocks, d)
        best, best_us = d, base
        for cfg, S, wk in itertools.product((0, 1, 4), (1, 2, 4, 8), (4, 8)):
            if N // 128 * S > 1024 or (Kin // 16) // (S * wk) < 2:
                continue
            c = {"cfg": cfg, "S": S, "wk": wk, "cfg2": 2}
            try:
                us = ring_us(blocks, c)
            except Exception as e:  # noqa: BLE001  (a split the kernel refuses for this shape)
                print(f"  {N}x{Kin}x{kb} {c}: {e}", flush=True)
                continue
            if us < best_us:
                best, best_us = c, us
        gain = (base - best_us) / base
        row = {"S": best["S"], "wk": best["wk"], "cfg": best["cfg"], "cfg2": 2, "us_m1_default": round(base, 2),
               "us_m1": round(best_us, 2), "gb_s": round(one / best_us / 1e3, 1), "gb_s_default": round(one / base / 1e3, 1)}
        print(f"{N}x{Kin}x{kb}: default {d} {base:.1f} us -> {best} {best_us:.1f} us ({100 * gain:.1f} %), "
              f"{row['gb_s']} GB/s", flush=True)
        if gain > a.min_gain:
            out[f"{N}x{Kin}x{kb}"] = row
        else:
            out[f"_{N}x{Kin}x{kb}"] = dict(row, kept="default")
        del blocks
        torch.cuda.empty_cache()
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
