"""Does v2 compute the same product as v1, on every projection shape and every row count?

Both kernels sum K in fp32 over fp16 products, so they cannot be bit-identical: v1 accumulates in
eight K = 16 pieces per 128-wide step, v2 in one or two wide ones. The bar is the one the engine
cares about -- that the bf16 result the next layer reads is the same number -- so this reports the
max absolute difference against v1 and against a full bf16 reference, and the fraction of elements
where the two bf16 outputs differ at all.
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.nvfp4_linear import nvfp4_matmul, quantize_to_nvfp4  # noqa: E402
from tools.nvfp4_linear_v2 import nvfp4_matmul_v2  # noqa: E402

SHAPES = [(17408, 5120, "mlp gate/up"), (5120, 17408, "mlp down"), (10240, 5120, "gdn qkv"),
          (6144, 5120, "gdn z"), (5120, 6144, "gdn out + attn o"), (12288, 5120, "attn q"),
          (1024, 5120, "attn kv"), (5120, 5120, "square (fallback shape)")]

ap = argparse.ArgumentParser()
ap.add_argument("--rows", default="1,2,4,8,14,16,24,32")
ap.add_argument("--dots", type=int, default=0, help="0 = both")
a = ap.parse_args()
rows = [int(r) for r in a.rows.split(",")]
dots = [1, 2] if a.dots == 0 else [a.dots]

torch.manual_seed(0)
print(f"{'shape':>26} {'M':>4} {'dots':>5} {'max|v2-v1|':>12} {'max|v2-ref|':>12} "
      f"{'max|v1-ref|':>12} {'bf16 differ':>12}")
worst = 0.0
bad = 0
for N, K, name in SHAPES:
    ref = torch.randn(N, K, device="cuda", dtype=torch.float32) * 0.02
    w = quantize_to_nvfp4(ref)
    wb = w.dequant().float()
    for M in rows:
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        exact = (x.float() @ wb.T)
        y1 = nvfp4_matmul(x, w).float()
        for d in dots:
            y2 = nvfp4_matmul_v2(x, w, dots=d).float()
            dv = (y2 - y1).abs().max().item()
            d2 = (y2 - exact).abs().max().item()
            d1 = (y1 - exact).abs().max().item()
            frac = (y2 != y1).float().mean().item()
            ok = dv <= max(2.0 * d1, 1e-6)
            bad += 0 if ok else 1
            worst = max(worst, dv / max(d1, 1e-12))
            print(f"{name:>26} {M:4d} {d:5d} {dv:12.3e} {d2:12.3e} {d1:12.3e} "
                  f"{100 * frac:11.2f}% {'' if ok else '  <-- FAIL'}")
        del x, exact, y1
    del ref, w, wb
    torch.cuda.empty_cache()
print(f"\nworst ratio |v2-v1| / |v1-exact| = {worst:.3f}   failures = {bad}")
print("GATE", "PASS" if bad == 0 else "FAIL")
