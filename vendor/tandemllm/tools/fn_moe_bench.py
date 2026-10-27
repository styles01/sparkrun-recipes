"""ENG-223: GB/s of the Flash-Next MoE kernels on the real shapes, weights cold in L2.

NB banks of E routed experts (640 x 2560 at K `--k`) plus a shared expert (K `--shared-k`) are
allocated; one CUDA graph runs the MoE of every bank in turn (random top-10 ids a row, M rows), so
each call reads weights the previous call did not touch. Bytes are the distinct experts' trellis,
suh and svh a call reads (the shared expert counted once). Prints one JSON line per M.

    python tools/fn_moe_bench.py --k 3 --rows 1 4 16 32
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools import exl3_linear as X  # noqa: E402
from tools import fn_kernels as K  # noqa: E402

H, I = 2560, 640


def bank(E: int, kb: int, skb: int, seed: int):
    trip = []
    sizes = []
    for e in range(E + 1):
        k = skb if e == E else kb
        bl = [X.random_block(I, H, k, seed=seed * 7919 + 3 * e + j) if j < 2 else
              X.random_block(H, I, k, seed=seed * 7919 + 3 * e + j) for j in range(3)]
        trip.append(tuple((b.t, b.suh[0].contiguous(), b.svh) for b in bl))
        sizes.append(sum(b.nbytes for b in bl))
    return K.MoEBank(trip, "cuda"), sizes


def parts_ms(banks, xs, idss, wss, orders, bufs, outs, reps) -> dict:
    """ms a call of gate|up, down and combine, each alone in its own graph over the ring."""
    mod = K._module()
    out = {}
    for part in ("gu", "down", "combine"):
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for i, (b, _) in enumerate(banks):
                flat = idss[i].reshape(-1)
                if part == "gu":
                    mod.fn_moe_gu(xs[i], flat, orders[i], b.tr, b.kb, b.su, b.sv, bufs[i]["ygu"], b.ks[-1])
                elif part == "down":
                    mod.fn_moe_down(bufs[i]["ygu"], flat, orders[i], b.tr, b.kb, b.su, b.sv, bufs[i]["yd"], b.ks[-1])
                else:
                    mod.fn_moe_combine(bufs[i]["yd"], flat, wss[i].reshape(-1), outs[i])
        g.replay()
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(reps):
            g.replay()
        e1.record()
        torch.cuda.synchronize()
        out[part] = round(e0.elapsed_time(e1) / reps / len(banks), 4)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--shared-k", type=int, default=5)
    ap.add_argument("--experts", type=int, default=128)
    ap.add_argument("--banks", type=int, default=12)
    ap.add_argument("--rows", type=int, nargs="+", default=[1, 4, 16, 32])
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--st", type=int, nargs="+", default=[2, 3, 4], help="MoE cp.async stages to compare")
    ap.add_argument("--warps", type=int, nargs="+", default=[8], help="MoE warps a CTA to compare")
    a = ap.parse_args()
    banks = [bank(a.experts, a.k, a.shared_k, s) for s in range(a.banks)]
    g = torch.Generator().manual_seed(0)
    for w in a.warps:
        K._module().fn_set_moe_warps(w)
        for st in a.st:
            K._module().fn_set_moe_st(st)
            run(a, banks, g, st, w)


def run(a, banks, g, st, warps) -> None:
    for M in a.rows:
        xs, idss, wss, outs, bufs, byts = [], [], [], [], [], 0
        for b, sizes in banks:
            x = (torch.randn(M, H, generator=g) * 0.5).to(torch.bfloat16).cuda()
            ids = torch.stack([torch.cat([torch.randperm(a.experts, generator=g)[:10], torch.tensor([a.experts])])
                               for _ in range(M)]).to(torch.int32).cuda()
            w = torch.rand(M, 11, generator=g).cuda()
            uniq = set(ids.flatten().tolist())
            byts += sum(sizes[e] for e in uniq)
            xs.append(x); idss.append(ids); wss.append(w)
            outs.append(torch.empty(M, H, dtype=torch.bfloat16, device="cuda"))
            bufs.append({})
        orders = [K.expert_order(i) for i in idss]
        for i, (b, _) in enumerate(banks):
            K.moe(xs[i], idss[i], wss[i], b, order=orders[i], bufs=bufs[i], out=outs[i])
        torch.cuda.synchronize()
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            for i, (b, _) in enumerate(banks):
                K.moe(xs[i], idss[i], wss[i], b, order=orders[i], bufs=bufs[i], out=outs[i])
        gr.replay()
        torch.cuda.synchronize()
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        ev[0].record()
        for _ in range(a.reps):
            gr.replay()
        ev[1].record()
        torch.cuda.synchronize()
        ms = ev[0].elapsed_time(ev[1]) / a.reps / len(banks)
        gbs = byts / len(banks) / (ms / 1e3) / 1e9
        parts = parts_ms(banks, xs, idss, wss, orders, bufs, outs, a.reps)
        print(json.dumps({"warps": warps, "st": st, "M": M, "k": a.k, "shared_k": a.shared_k, "experts": a.experts, "banks": a.banks,
                          "ms_per_call": round(ms, 4), "mb_per_call": round(byts / len(banks) / 1e6, 2),
                          "gb_s": round(gbs, 1), "parts_ms": parts}), flush=True)


if __name__ == "__main__":
    main()
