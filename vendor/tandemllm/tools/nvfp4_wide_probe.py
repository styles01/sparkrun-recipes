"""What does the W4A16 kernel cost as M grows, and can a tile shape flatten it?

The verify staircase -- 16 nodes 143.9 ms, 24 nodes 215.3 -- is not bytes. The weights are read
once whatever M is, and at M = 64 the step's arithmetic is a few TFLOP, which the tensor cores do in
single-digit milliseconds. So the step at 16 rows is a property of the KERNEL, and this file is the
measurement that says which property: the tile shape `pick_config` chooses, or the shape of the dot
inside the tile.

`_nvfp4_linear_kernel` already reuses its weight tile across all `BLOCK_M` rows -- the decode cost
is amortised over M and only the `tl.dot` scales. But it issues that dot **eight times at K = 16**
per 128-wide K step, and a K = 16 MMA is the least efficient shape the tensor cores have. If the
staircase is the dot, a wider K is the fix; if it is `pick_config` handing out BLOCK_M = 16 when M
is 24, the fix is a table entry.

Read-only with respect to `tools/nvfp4_linear.py`, which is track A's: this passes tile shapes in
rather than editing the chooser.
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.nvfp4_linear import NVFP4Block, nvfp4_matmul, pick_config  # noqa: E402


def quantise(N: int, K: int, device: str) -> NVFP4Block:
    """A random projection in the NVFP4 layout. Timing only -- the values never matter here, only
    that the packed layout and its scales are the ones the kernel reads."""
    from tools.quant_nvfp4 import quantize_clipped
    w = torch.randn(N, K, device=device, dtype=torch.float32) * 0.02
    return quantize_clipped(w, None)


def time_call(fn, iters: int = 30) -> float:
    for _ in range(4):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--rows", default="1,8,16,24,32,48,64,96,128")
    ap.add_argument("--shapes", default="all", help="all | mlp | gdn | attn")
    ap.add_argument("--block-m", default="16,32,64,128")
    ap.add_argument("--block-n", default="32,64,128")
    ap.add_argument("--split-k", default="1,2,4,8")
    ap.add_argument("--warps", default="1,2,4,8")
    ap.add_argument("--iters", type=int, default=30)
    a = ap.parse_args()

    # Every projection the verify pass actually reads, with how many of each there are per step.
    # The block cycle is 419 of these calls and the mean call reaches 139 GB/s, so the question is
    # which of them loses the bandwidth and at what M -- the MLP is 70 % of the bytes but the
    # small projections are two thirds of the calls.
    ALL = [(17408, 5120, "mlp gate/up", 128), (5120, 17408, "mlp down", 64),
           (10240, 5120, "gdn qkv", 48), (6144, 5120, "gdn z", 48),
           (5120, 6144, "gdn out", 48), (12288, 5120, "attn q", 16),
           (1024, 5120, "attn kv", 32), (5120, 6144, "attn o", 16)]
    shapes = [(n, k, nm) for n, k, nm, _ in ALL
              if a.shapes == "all" or nm.split()[0] == a.shapes]
    counts = {nm: c for _, _, nm, c in ALL}
    rows = [int(x) for x in a.rows.split(",")]
    bms = [int(x) for x in a.block_m.split(",")]
    bns = [int(x) for x in a.block_n.split(",")]
    sks = [int(x) for x in a.split_k.split(",")]
    ws = [int(x) for x in a.warps.split(",")]

    for N, K, name in shapes:
        w = quantise(N, K, a.device)
        gb = w.nbytes / 1e9
        print(f"\n### {name}  N={N} K={K}  {gb * 1e3:.1f} MB of packed weight, "
                  f"{counts[name]} per verify step")
        print(f"{'M':>5} {'default ms':>11} {'best ms':>9} {'best tile':>26} "
              f"{'best GB/s':>9} {'vs M=16':>8} {'dflt GB/s':>9}")
        base = None
        for M in rows:
            x = torch.randn(M, K, device=a.device, dtype=torch.bfloat16)
            cfg = pick_config(N, K, M)
            dflt = time_call(lambda: nvfp4_matmul(x, w), a.iters)
            best, best_cfg = dflt, dict(cfg)
            for bm, bn, sk, nw in itertools.product(bms, bns, sks, ws):
                if bm < M and bm < 128 and M <= 128 and bm * 2 < M:
                    continue                      # more than two row tiles is never the answer
                try:
                    t = time_call(lambda: nvfp4_matmul(x, w, block_m=bm, block_n=bn, split_k=sk,
                                                       num_warps=nw), max(8, a.iters // 3))
                except Exception:
                    continue
                if t < best:
                    best, best_cfg = t, {"block_m": bm, "block_n": bn, "split_k": sk,
                                         "num_warps": nw}
            if M == 16:
                base = best
            tile = (f"m{best_cfg.get('block_m')} n{best_cfg.get('block_n')} "
                    f"k{best_cfg.get('split_k')} w{best_cfg.get('num_warps')}")
            ratio = f"{best / base:7.2f}x" if base else "       -"
            print(f"{M:5d} {dflt:11.3f} {best:9.3f} {tile:>26} {gb / (best / 1e3):7.1f} "
                  f"{ratio:>8} {gb / (dflt / 1e3):7.1f}", flush=True)


if __name__ == "__main__":
    main()
