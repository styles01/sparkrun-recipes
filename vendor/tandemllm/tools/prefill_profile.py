"""Where a long prefill's GPU time goes: one `cache.prefill` under the torch profiler, kernel time
summed by kind (EXL3 slab decode, the library GEMM, output rotation, attention, GDN, norms, the rest),
plus the library GEMM's achieved TFLOP/s on the prefill's own shapes.

Usage: prefill_profile.py --n 8192 --chunk 1024 [--cached] [--window 8192]
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time
import types
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import cache  # noqa: E402
from engine.config import load_config  # noqa: E402
from engine.loader import Weights  # noqa: E402
from engine.model import Qwen38Engine  # noqa: E402
from tools.prefill_window_check import prompt_ids  # noqa: E402
from tools.verify_spec import tree_router  # noqa: E402

KINDS = [
    ("exl3 recon", ("recon_kernel",)),
    ("exl3 rot_in", ("rot_in_kernel",)),
    ("exl3 rot_out", ("rot_out_kernel",)),
    ("exl3 skinny", ("gemm_kernel",)),
    ("attention", ("flash", "fmha", "attention", "sdpa", "Attention")),
    ("gdn", ("chunk", "gdn", "conv", "wy", "delta", "recur", "solve", "trsm")),
    ("gemm (library)", ("gemm", "Gemm", "cutlass", "sm90", "sm100", "sm120", "nvjet", "xmma", "ampere")),
    ("norm", ("norm", "rms")),
]


def kind(name: str) -> str:
    for k, keys in KINDS:
        if any(x in name for x in keys):
            return k
    return "other"


def mm_rate(M: int, K: int, N: int, reps: int = 10) -> float:
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    for _ in range(2):
        torch.mm(a, b, out_dtype=torch.float32)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        torch.mm(a, b, out_dtype=torch.float32)
    torch.cuda.synchronize()
    return 2 * M * K * N * reps / (time.perf_counter() - t0) / 1e12


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8192)
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--cached", action="store_true")
    ap.add_argument("--window", type=int, default=None)
    ap.add_argument("--max-len", type=int, default=40960)
    ap.add_argument("--mm", action="store_true", help="also the library GEMM's rate on the shapes")
    ap.add_argument("--shapes", action="store_true", help="aten ops grouped by input shape")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    if a.window is not None:
        cache.PREFILL_WINDOW = cache.PREFILL_WINDOW_FREE = a.window
    repo = Path(__file__).resolve().parent.parent
    cfg = load_config(None)
    w = Weights(cfg.path, device="cuda", skip_mtp=True)
    eng = Qwen38Engine(cfg, w, max_len=a.max_len, device="cuda")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.path)
    ns = types.SimpleNamespace(dflash2_ckpt="train/ft-b8-v2", dflash2_path="greedy",
                               max_len=a.max_len, lenrouter="train/ft-b16", corpus="",
                               drop_idle=True, deep=0, deep_after=2, len_fixed=0)
    lr = tree_router(eng, ns, narrow=False)
    ids = prompt_ids(tok, repo, a.n)

    def once():
        eng.reset()
        lr.reset()
        res = cache.ResidentPrefix(8 << 30, a.chunk, tail=4) if a.cached else None
        st = cache.StateStore(4 << 30, chunk=a.chunk, max_entry_bytes=2 << 30) if a.cached else None
        cache.prefill(eng, lr, ids, "cuda", store=st, chunk=a.chunk, checkpoint=a.cached,
                      resident=res)
        torch.cuda.synchronize()

    once()                                   # warm: kernels compiled, allocator grown
    t0 = time.perf_counter()
    once()
    wall = time.perf_counter() - t0
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=a.shapes) as prof:
        once()
    by = collections.Counter()
    names = collections.Counter()
    ops = collections.Counter()
    for e in prof.key_averages(group_by_input_shape=a.shapes):
        if e.key.startswith("aten::") or e.key.startswith("triton") or "::" in e.key:
            sd = getattr(e, "self_device_time_total", None) or getattr(e, "self_cuda_time_total", 0)
            if sd > 0 and e.device_type.name == "CPU":
                shp = str(getattr(e, "input_shapes", ""))[:120] if a.shapes else ""
                ops[(e.key + " " + shp)[:200]] += sd / 1e6
    for e in prof.key_averages():
        if e.device_type.name == "CPU":
            continue
        dt = getattr(e, "device_time_total", None) or getattr(e, "cuda_time_total", 0)
        if dt <= 0:
            continue
        by[kind(e.key)] += dt / 1e6
        names[e.key[:90]] += dt / 1e6
    tot = sum(by.values())
    out = {"n": a.n, "chunk": a.chunk, "cached": a.cached, "window": a.window,
           "wall_s": round(wall, 3), "gpu_s": round(tot, 3),
           "by_kind_s": {k: round(v, 3) for k, v in by.most_common()},
           "top": [(k, round(v, 3)) for k, v in names.most_common(25)],
           "ops_self_device_s": [(k, round(v, 3)) for k, v in ops.most_common(40)]}
    if a.mm:
        out["mm_tflops"] = {f"{M}x{K}x{N}": round(mm_rate(M, K, N), 1)
                            for M in (a.chunk, 8192) for K, N in
                            ((5120, 16384), (5120, 17408), (17408, 5120), (6144, 5120))}
    print(json.dumps(out, indent=1), flush=True)
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
