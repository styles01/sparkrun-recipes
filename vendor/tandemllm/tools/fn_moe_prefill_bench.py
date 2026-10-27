"""ENG-225: where the grouped MoE of a prompt chunk spends its time, on a real layer and real routing.

    PYTHONPATH=<repo>:~/fn-ref/site python tools/fn_moe_prefill_bench.py --pack ~/fn2/packs/3.05 \
        --layers 1 --rows 2048,4096 --items 32,64

Loads the first `--layers` layers only (a debug model: about 3 GB, fine beside :8000), reads a prose
prompt through them on the prefill kernels and keeps the last layer's MoE input and router ids. Then
per chunk length and item size: ms of each part of `moe_grouped` (the plan, the pairs' rotation, the
gate|up GEMM, the activation's rotation, the down GEMM), CUDA events over `--reps` calls, and the
GEMMs' TFLOP/s on the useful rows.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.models.flashnext.engine import FlashNextEngine  # noqa: E402
from engine.models.flashnext.load import load  # noqa: E402
from tools.fn_generate import class_prompt  # noqa: E402


def timed(fn, reps: int) -> float:
    fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(reps):
        fn()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / reps


@torch.no_grad()
def main() -> None:
    from tools import fn_prefill_kernels as PK
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--rows", default="2048,4096")
    ap.add_argument("--items", default="32,64")
    ap.add_argument("--reps", type=int, default=10)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    idx, W, banks, tables, _ = load(a.pack, "cuda", first=a.layers, log=None)
    rows = [int(r) for r in a.rows.split(",")]
    eng = FlashNextEngine(idx.cfg, W, tables, banks, device="cuda", max_len=max(rows) + 64)
    tok = AutoTokenizer.from_pretrained(a.pack)
    got = {}
    real = eng._moe_pre

    def keep(x, li):
        out = real(x, li)
        got["x"], got["ids"] = x.clone(), out[1].clone()
        return out
    eng._moe_pre = keep
    st = eng.new_state(eng.max_len)
    eng.prefill(torch.tensor(class_prompt(tok, "prose", max(rows))), st, chunk=max(rows), path="kernels")
    li = a.layers - 1
    bank = banks[li]
    mod = PK._module()
    for M in rows:
        x, ids = got["x"][:M].contiguous(), got["ids"][:M].contiguous()
        S = ids.shape[1]
        P = M * S
        cnt = torch.bincount(ids.reshape(-1).long(), minlength=bank.n)
        for r in (int(v) for v in a.items.split(",")):
            gp = PK.group_plan(ids.reshape(-1), bank.n, r)
            R, prow, cn, ps, ie, items = gp["R"], gp["prow"], gp["cnt"], gp["pstart"], gp["iend"], gp["items"]
            row = torch.div(gp["order"], S, rounding_mode="floor").to(torch.int32)
            xh = torch.empty(2, R, bank.H, dtype=torch.float16, device="cuda")
            ygu = torch.empty(R, 2 * bank.I, dtype=torch.bfloat16, device="cuda")
            xd = torch.empty(1, R, bank.I, dtype=torch.float16, device="cuda")
            yd = torch.empty(P, bank.H, dtype=torch.float32, device="cuda")
            dst = torch.zeros(R, dtype=torch.int32, device="cuda")
            dst.index_copy_(0, prow.long(), gp["order"].to(torch.int32))
            kb = bank.ks[-1]
            res = {"rows": M, "item_rows": r, "pairs": P, "items": int(ie[-1]), "experts_used": int((cnt > 0).sum()),
                   "max_pairs_an_expert": int(cnt[:-1].max()),
                   "plan_ms": timed(lambda: PK.group_plan(ids.reshape(-1), bank.n, r), a.reps),
                   "rot_ms": timed(lambda: mod.pre_pair_rot(x, row, gp["eid"], prow, bank.tr, bank.kb, bank.su, bank.sv, 0,
                                                            xh), a.reps),
                   "gu_ms": timed(lambda: mod.pre_group_gemm(xh, cn, ps, ie, bank.tr, bank.kb, bank.su, bank.sv, 0, bank.I,
                                                             ygu, None, items, r // 16, kb), a.reps),
                   "silu_rot_ms": timed(lambda: mod.pre_pair_silu_rot(ygu, gp["eid"], prow, bank.tr, bank.kb, bank.su,
                                                                      bank.sv, xd), a.reps),
                   "down_ms": timed(lambda: mod.pre_group_gemm(xd, cn, ps, ie, bank.tr, bank.kb, bank.su, bank.sv, 2, bank.H,
                                                               yd, dst, items, r // 16, kb), a.reps),
                   "all_ms": timed(lambda: PK.moe_grouped(x, ids, bank, rows=r), a.reps)}
            f = 2.0 * P * bank.H * bank.I
            res["gu_tflops"] = round(2 * f / res["gu_ms"] / 1e9, 1)
            res["down_tflops"] = round(f / res["down_ms"] / 1e9, 1)
            print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items()}), flush=True)


if __name__ == "__main__":
    main()
