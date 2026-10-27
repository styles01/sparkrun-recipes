"""The verify attention at long context (SPD-95's second half): `decode_attention` launch knobs at
16 rows over 8k..128k of context, time against the KV bytes, and every output compared bit for bit
with the served launch (bm 32, bn 32, 4 warps, 2 stages, at most 64 splits).

Usage: attn_sweep.py --ctx 8192,32768,65536,131072 --rows 16
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools import attn_kernels as A  # noqa: E402


def timed(fn, reps=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(reps):
        fn()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / reps * 1e3            # us


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", default="8192,32768,65536,131072")
    ap.add_argument("--rows", type=int, default=16)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    torch.manual_seed(0)
    Hq, Hkv, D, T = 24, 4, 256, a.rows
    out = []
    grid = list(itertools.product((32, 64, 128), (32, 64), (4, 8), (2, 3, 4), (64, 128, 256)))
    for L in (int(x) for x in a.ctx.split(",")):
        k = (torch.randn(1, Hkv, L, D, device="cuda") * 0.5).to(torch.bfloat16)
        v = (torch.randn(1, Hkv, L, D, device="cuda") * 0.5).to(torch.bfloat16)
        q = (torch.randn(1, Hq, T, D, device="cuda") * 0.5).to(torch.bfloat16)
        bm = torch.ones(T, T, dtype=torch.bool, device="cuda").tril()
        start = L - T
        kv_bytes = 2 * Hkv * L * D * 2
        ref = A.decode_attention(q, k, v, start, bm)
        base_us = timed(lambda: A.decode_attention(q, k, v, start, bm))
        print(f"[attn] ctx {L}: served {base_us:.1f} us, {kv_bytes / base_us / 1e3:.0f} GB/s of KV",
              flush=True)
        for bmr, bn, nw, ns, msp in grid:
            try:
                f = lambda: A.decode_attention(q, k, v, start, bm, bm=bmr, bn=bn, num_warps=nw,
                                               num_stages=ns, max_splits=msp)
                got = f()
                us = timed(f)
            except Exception as e:                                   # noqa: BLE001
                continue
            row = {"ctx": L, "bm": bmr, "bn": bn, "warps": nw, "stages": ns, "max_splits": msp,
                   "us": round(us, 1), "GBps": round(kv_bytes / us / 1e3), "served_us": round(base_us, 1),
                   "identical": bool(torch.equal(ref, got))}
            out.append(row)
        best_same = min((r for r in out if r["ctx"] == L and r["identical"]), key=lambda r: r["us"])
        best_any = min((r for r in out if r["ctx"] == L), key=lambda r: r["us"])
        print(f"[attn] ctx {L}: best identical {json.dumps(best_same)}", flush=True)
        print(f"[attn] ctx {L}: best any       {json.dumps(best_any)}", flush=True)
        del k, v
        torch.cuda.empty_cache()
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
