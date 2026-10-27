"""Two tiles, one shape, alternating, many repeats: is the sweep's winner a winner?

`tools/nvfp4_wide_probe.py` searches ~100 tiles per (shape, M) and reports the minimum. The
minimum of a hundred noisy timings is biased low by the noise, and the baseline it is compared
against is a single timing of one tile. So the sweep can only ever propose; this decides, by timing
the two candidates ALTERNATELY within one repeat -- which cancels drift -- and repeating.
"""
import os, sys, time, argparse
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.nvfp4_linear import nvfp4_matmul, pick_config  # noqa: E402
from tools.quant_nvfp4 import quantize_clipped  # noqa: E402

SHAPES = [(17408, 5120, "mlp gate/up", 128), (5120, 17408, "mlp down", 64),
          (10240, 5120, "gdn qkv", 48), (6144, 5120, "gdn z", 48),
          (5120, 6144, "gdn out + attn o", 64), (12288, 5120, "attn q", 16),
          (1024, 5120, "attn kv", 32)]
DECODE = {"block_m": 16, "block_n": 16, "split_k": 8, "num_warps": 1, "num_stages": 3}


def t(fn, iters):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


ap = argparse.ArgumentParser()
ap.add_argument("--M", type=int, default=14)
ap.add_argument("--iters", type=int, default=120)
ap.add_argument("--reps", type=int, default=5)
a = ap.parse_args()
M = a.M
print(f"M = {M}, {a.iters} iters x {a.reps} alternating repeats\n")
print(f"{'shape':>18} {'n':>4} {'decode ms':>10} {'block ms':>10} {'delta':>8} "
      f"{'per step ms':>12}")
tot = 0.0
for N, K, name, cnt in SHAPES:
    w = quantize_clipped(torch.randn(N, K, device="cuda", dtype=torch.float32) * 0.02, None)
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    blk = pick_config(N, K, M)
    d, b = [], []
    for _ in range(a.reps):
        d.append(t(lambda: nvfp4_matmul(x, w, **DECODE), a.iters))
        b.append(t(lambda: nvfp4_matmul(x, w, **blk), a.iters))
    dm, bm = min(d), min(b)
    tot += cnt * (dm - bm)
    print(f"{name:>18} {cnt:4d} {dm:10.4f} {bm:10.4f} {100 * (bm - dm) / dm:7.1f}% "
          f"{cnt * (dm - bm):12.2f}")
    del w, x
    torch.cuda.empty_cache()
print(f"\ntotal saved per verify step at M={M}: {tot:.2f} ms")
