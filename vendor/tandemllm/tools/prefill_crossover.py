"""Where the v2 prefill tile and the unpack path cross, at the row counts a prefill chunk has.

`tools/nvfp4_linear.py` sends a projection of 33..PREFILL_V2_UNTIL-1 rows to the v2 kernel's prefill
tile and one of DEQUANT_FROM rows or more (512) to the unpack path (unpack to bf16, library GEMM,
drop). PREFILL_V2_UNTIL is 1,024, and a prefill with the prefix cache on -- :8000's -- is forwarded
in chunks of exactly 1,024 rows, so every full chunk of a served prefill takes the unpack path.
measured v2 1.4-1.65x ahead at 512 rows and level at 2,048; 1,024 was never measured. This
times both paths on the MLP shapes (17 of the model's 27 B parameters) at the chunk sizes, with
distinct weights per call so nothing is warm in L2, and names the crossover per shape.

    # inside ops/hold.sh: CUDA
    python tools/prefill_crossover.py --rows 512,768,1024,1536,2048,4096,8192
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SHAPES = {"gate/up": (17408, 5120), "down": (5120, 17408)}


def crossover(table: dict[int, dict[str, float]]) -> int | None:
    """The smallest row count from which the unpack path is faster at every larger row count
    measured (None when v2 wins at the largest). `table[M] = {"v2": ms, "unpack": ms}`."""
    rows = sorted(table)
    for i, m in enumerate(rows):
        if all(table[x]["unpack"] < table[x]["v2"] for x in rows[i:]):
            return m
    return None


def main() -> None:
    import torch
    import torch.nn.functional as F
    from tools.nvfp4_linear import PREFILL_V2_TILE, quantize_to_nvfp4
    from tools.nvfp4_linear_v2 import nvfp4_matmul_v2

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", default="512,768,1024,1536,2048,4096,8192")
    ap.add_argument("--weights", type=int, default=4, help="distinct weights a shape, cycled")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", default="results/phase4/prefill-crossover.json")
    a = ap.parse_args()
    torch.manual_seed(0)
    out = {"tile": PREFILL_V2_TILE, "shapes": {}, "args": vars(a)}
    for name, (N, K) in SHAPES.items():
        ws = [quantize_to_nvfp4(torch.randn(N, K, device="cuda") * 0.02) for _ in range(a.weights)]
        table = {}
        for M in (int(x) for x in a.rows.split(",")):
            x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
            paths = {"v2": lambda w: nvfp4_matmul_v2(x, w, **PREFILL_V2_TILE),
                     "unpack": lambda w: F.linear(x, w.dequant_fast())}
            row = {}
            for p, fn in paths.items():
                for w in ws:
                    fn(w)
                torch.cuda.synchronize()
                best = float("inf")
                for _ in range(a.reps):
                    t0 = time.perf_counter()
                    for w in ws:
                        fn(w)
                    torch.cuda.synchronize()
                    best = min(best, (time.perf_counter() - t0) / len(ws))
                row[p] = best * 1e3
            row["v2_tflops"] = 2 * M * N * K / row["v2"] / 1e9
            row["unpack_tflops"] = 2 * M * N * K / row["unpack"] / 1e9
            table[M] = row
            print(f"[crossover] {name:8s} M={M:>5}  v2 {row['v2']:7.3f} ms ({row['v2_tflops']:5.1f} TF/s)  "
                  f"unpack {row['unpack']:7.3f} ms ({row['unpack_tflops']:5.1f} TF/s)  "
                  f"v2/unpack {row['v2'] / row['unpack']:.3f}", flush=True)
            del x
        out["shapes"][name] = {"table": table, "crossover": crossover(table)}
        print(f"[crossover] {name}: unpack wins from {out['shapes'][name]['crossover']} rows", flush=True)
        del ws
        torch.cuda.empty_cache()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"[crossover] {a.out}")


if __name__ == "__main__":
    main()
