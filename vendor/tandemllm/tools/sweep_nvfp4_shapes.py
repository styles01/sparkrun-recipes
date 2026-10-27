"""Tile choice for the NVFP4 projections that are not the MLP, measured in the engine.

`tools/nvfp4_linear.py::_CONFIG` carries a measured tile for the two MLP shapes and a fallback for
everything else. Phase 4 put the GDN and attention projections into the same kernel, and the
fallback is not their tile: at 12:29 the GDN mixer read 64.9 MB of NVFP4 in 0.479 ms, which is
135 GB/s, against the MLP kernel's 211 on its own shapes.

The timing is the engine's own mixer, not a standalone kernel, for the reason recorded at 13:40 of
phase 2: the same configuration measured 0.254 ms and 0.627 ms in two processes that differed only
in what they had allocated before. One process here, one set of weights, the configuration flipped
between runs.

    python tools/sweep_nvfp4_shapes.py --rows 1,8
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.config import load_config  # noqa: E402
from engine.loader import Weights  # noqa: E402
from engine.model import Qwen38Engine  # noqa: E402
from tools.nvfp4_linear import NVFP4Block, set_config  # noqa: E402


def timeit(fn, n: int = 40, warm: int = 8) -> float:
    with torch.no_grad():
        for _ in range(warm):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--rows", default="1,8", help="row counts to tune for")
    ap.add_argument("--block-m", default="", help="prefill bucket only; decode tiles fix it")
    ap.add_argument("--block-n", default="16,32,64")
    ap.add_argument("--split-k", default="1,2,4,8,16")
    ap.add_argument("--warps", default="1,2,4")
    ap.add_argument("--stages", default="3")
    ap.add_argument("--bucket", default="decode", choices=("decode", "mid", "prefill"),
                    help="which tile bucket to write; pick_config uses decode to 32 rows, "
                         "mid to 128, prefill above")
    ap.add_argument("--mlp", action="store_true", help="tune the MLP shapes as well")
    a = ap.parse_args()

    cfg = load_config(a.model)
    w = Weights(cfg.path, device=a.device, skip_mtp=True)
    print(f"[load] {w.report()}")
    rows_m = [int(x) for x in a.rows.split(",")]
    eng = Qwen38Engine(cfg, w, max_len=max(rows_m) + 64, device=a.device)

    gdn_l, att_l = cfg.linear_layers[0], cfg.attention_layers[0]
    shapes = {}
    for group, layer in (("gdn", gdn_l), ("attn", att_l)):
        shapes[group] = sorted({(b.N, b.K) for name, b in w.q.items()
                                if name.startswith(f"layers.{layer}.")
                                and ".mlp." not in name and isinstance(b, NVFP4Block)})
        print(f"[shapes] {group}: {shapes[group]}")
    if a.mlp:
        shapes["mlp"] = sorted({(b.N, b.K) for name, b in w.q.items()
                                if name.startswith(f"layers.{gdn_l}.mlp.")
                                and isinstance(b, NVFP4Block)})
        print(f"[shapes] mlp: {shapes['mlp']}")
    if not shapes["gdn"] and not shapes["attn"]:
        raise SystemExit("no NVFP4 projections outside the MLP; nothing to tune")

    block_m = [int(x) for x in a.block_m.split(",")] if a.block_m else [0]
    grid = list(itertools.product([int(x) for x in a.block_n.split(",")],
                                  [int(x) for x in a.split_k.split(",")],
                                  [int(x) for x in a.warps.split(",")],
                                  block_m,
                                  [int(x) for x in a.stages.split(",")]))
    for M in rows_m:
        h = torch.randn(1, M, cfg.hidden_size, device=a.device, dtype=torch.bfloat16) * 0.02
        pos = torch.arange(0, M, device=a.device)
        prefill = M > 128
        eng.state.primed = not prefill
        cases = [("gdn", lambda: eng.linear_attention(h, f"layers.{gdn_l}", gdn_l, not prefill)),
                 ("attn", lambda: eng.attention(h, f"layers.{att_l}", att_l, 0, pos))]
        if a.mlp:
            cases.append(("mlp", lambda: eng.mlp(h, f"layers.{gdn_l}")))
        for group, fn in cases:
            if not shapes.get(group):
                continue
            if group == "mlp":
                nbytes = sum(w.q[n].nbytes for n in w.q if n.startswith(f"layers.{gdn_l}.mlp."))
            else:
                nbytes = sum(w.q[n].nbytes for n in w.q
                             if n.startswith(f"layers.{gdn_l if group == 'gdn' else att_l}.")
                             and ".mlp." not in n)
            best = None
            rows = []
            for bn, sk, nw, bm, st in grid:
                cfg_d = {"block_n": bn, "split_k": sk, "num_warps": nw, "num_stages": st}
                if bm:
                    cfg_d["block_m"] = bm
                for N, K in shapes[group]:
                    set_config(N, K, a.bucket, cfg_d)
                tag = f"bn={bn:3d} sk={sk:2d} w={nw} bm={bm:4d} st={st}"
                try:
                    ms = timeit(fn, n=8 if prefill else 40, warm=3) * 1e3
                except Exception as exc:                         # pragma: no cover
                    rows.append((float("inf"), f"{tag}  failed: {type(exc).__name__}"))
                    continue
                rows.append((ms, f"{tag}  {ms:8.3f} ms  {nbytes / ms / 1e6:7.1f} GB/s"))
                if best is None or ms < best[0]:
                    best = (ms, bn, sk, nw, bm, st)
            rows.sort()
            print(f"\n[{group}] M={M} bucket={a.bucket}  "
                  f"{nbytes / 1e6:.1f} MB of quantised weight")
            for _, line in rows[:8]:
                print("   ", line)
            print(f"    best bn={best[1]} sk={best[2]} w={best[3]} bm={best[4]} st={best[5]} "
                  f"at {best[0]:.3f} ms")


if __name__ == "__main__":
    main()
