"""Stage 0: how many vocabulary rows would a two-stage exact head have to re-check?

The e4m3 head (1.27 GB) is read twice a block. A two-stage head reads an NVFP4 copy of it instead
(0.64 GB), bounds each row's error, and recomputes from the e4m3 head only the rows whose bound
still reaches the best one. That is exact by construction -- every row it drops is provably below
the maximum -- and it pays only if the re-checked set is small. This tool measures that set on real
hidden states before any kernel is written.

THE BOUND. For a hidden row h and vocabulary row i, with W_i the e4m3 row the engine reads and Ŵ_i
its NVFP4 copy, s_i = <h, W_i> and ŝ_i = <h, Ŵ_i>:

    row      |s_i - ŝ_i| <= ||h||_2 * ||W_i - Ŵ_i||_2                       (Cauchy-Schwarz)
    group    |s_i - ŝ_i| <= sum_g ||h_g||_2 * ||W_ig - Ŵ_ig||_2              (per 16-wide group)

plus each side's fp32 accumulation error, bounded by K * u * ||h|| * ||row|| with u = 2^-24 (the
products of a bf16 activation and an e4m3/e2m1 weight are exact in fp32; only the sums round).
The group bound is never looser than the row bound (Cauchy-Schwarz over the groups). Candidates at
rank k: every row whose upper bound reaches the k-th largest lower bound. The exact top-k is always
inside that set -- `sound()` checks it on every row measured, so a bug in the bound shows as a miss
rather than as a small number.

    python tools/head_twostage.py --traces results/lat-b16 --rows 10000 --out results/spd43
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

U32 = 2.0 ** -24
GROUP = 16


def dequant_rows(head, r0: int, r1: int) -> torch.Tensor:
    """Fp32 rows [r0, r1) of an e4m3 head (FP8Head) or a plain tensor."""
    if hasattr(head, "s") and hasattr(head, "w") and head.w.dtype == torch.float8_e4m3fn:
        return head.w[r0:r1].float() * head.s[r0:r1, None]
    return head[r0:r1].float()


def nvfp4_rows(nv, r0: int, r1: int) -> torch.Tensor:
    """Fp32 rows [r0, r1) of an NVFP4 block, EXACT: e2m1 x e4m3 x the tensor scale is what the
    skinny kernel multiplies out (its operand is the e2m1 value, the scales are applied in fp32);
    `NVFP4Block.dequant` rounds the product to bf16, which would put its own error in the bound."""
    from tools.nvfp4_linear import FP4_GRID
    grid = FP4_GRID.to(nv.w.device)
    b = nv.w[r0:r1]
    lo, hi = (b & 0x0F).long(), (b >> 4).long()
    vals = torch.empty(r1 - r0, nv.K, dtype=torch.float32, device=nv.w.device)
    vals[:, 0::2] = grid[lo & 7] * torch.where(lo >= 8, -1.0, 1.0)
    vals[:, 1::2] = grid[hi & 7] * torch.where(hi >= 8, -1.0, 1.0)
    return vals * (nv.s[r0:r1].float().repeat_interleave(GROUP, 1) * nv.s2)


def error_norms(head, nv, *, rows: int = 8192, group: int = GROUP) -> dict:
    """Per vocabulary row: ||W_i||, ||Ŵ_i||, ||W_i - Ŵ_i||, and the per-group error norms
    [N, K / group] (fp32). `nv` is an NVFP4Block or a plain fp32 tensor (the tests)."""
    N, K = head.shape
    dev = nv.w.device if hasattr(nv, "s2") else nv.device
    w_n = torch.empty(N, device=dev)
    q_n = torch.empty(N, device=dev)
    e_n = torch.empty(N, device=dev)
    e_g = torch.empty(N, K // group, device=dev)
    for r0 in range(0, N, rows):
        r1 = min(r0 + rows, N)
        w = dequant_rows(head, r0, r1).to(dev)
        q = nvfp4_rows(nv, r0, r1) if hasattr(nv, "s2") else nv[r0:r1].float()
        d = w - q
        w_n[r0:r1] = w.norm(dim=1)
        q_n[r0:r1] = q.norm(dim=1)
        e_n[r0:r1] = d.norm(dim=1)
        e_g[r0:r1] = d.reshape(r1 - r0, K // group, group).norm(dim=2)
    return {"w": w_n, "q": q_n, "e": e_n, "eg": e_g, "K": K, "group": group}


def radius(h: torch.Tensor, en: dict, mode: str = "row") -> torch.Tensor:
    """[M, N] bound on |s - ŝ| for hidden rows h [M, K], fp32 accumulation included."""
    h = h.float()
    hn = h.norm(dim=1, keepdim=True)                                     # [M, 1]
    acc = en["K"] * U32 * hn * (en["w"] + en["q"])[None, :]
    if mode == "row":
        return hn * en["e"][None, :] + acc
    hg = h.reshape(h.shape[0], -1, en["group"]).norm(dim=2)             # [M, K / group]
    return hg @ en["eg"].T + acc


def candidates(s_hat: torch.Tensor, rad: torch.Tensor, k: int = 1) -> torch.Tensor:
    """Mask [M, N]: the rows a two-stage head must re-check to return the exact top-k."""
    lower = s_hat - rad
    thr = lower.topk(k, dim=1).values[:, -1:]
    return (s_hat + rad) >= thr


def sound(s: torch.Tensor, mask: torch.Tensor, k: int = 1) -> int:
    """Rows whose exact top-k is NOT inside the candidate set (must be 0)."""
    top = s.topk(k, dim=1).indices
    return int((~mask.gather(1, top)).any(dim=1).sum())


def hidden_rows(eng, traces: str, limit: int) -> tuple[torch.Tensor, list[str]]:
    """The final-norm hidden rows that predict each continuation token of the recorded traces
    (prefill of prompt + continuation), up to `limit` rows, with each row's trace class."""
    hs, klass = [], []
    n = 0
    for path in sorted(glob.glob(os.path.join(traces, "*.json"))):
        tr = json.load(open(path))
        p, o = tr["prompt_ids"], tr["output_ids"]
        ids = torch.tensor(p + o[:-1], device="cuda")
        eng.reset()
        with torch.no_grad():
            eng.forward(ids, 0, last_only=True)
        h = eng.hidden_post_norm[0, len(p) - 1:].detach().clone()
        hs.append(h)
        klass += [tr.get("klass", "?")] * h.shape[0]
        n += h.shape[0]
        if n >= limit:
            break
    return torch.cat(hs)[:limit], klass[:limit]


def pct(x: torch.Tensor, q: float) -> float:
    return float(torch.quantile(x.float(), q))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traces", default="results/lat-b16")
    ap.add_argument("--rows", type=int, default=10000)
    ap.add_argument("--nvfp4", default=os.environ.get("QWEN38_NVFP4"))
    ap.add_argument("--fp8-head", default=os.environ.get("QWEN38_FP8_HEAD"))
    ap.add_argument("--ks", default="1,16")
    ap.add_argument("--chunk", type=int, default=16, help="verify rows a head call, as the engine")
    ap.add_argument("--out", default="results/spd43")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    from engine.config import load_config
    from engine.loader import Weights
    from engine.model import Qwen38Engine
    from tools.head_gemv import head_matmul_fp8, head_to_nvfp4

    torch.backends.cuda.matmul.allow_tf32 = False      # the reference ŝ is an fp32 product
    cfg = load_config(None)
    w = Weights(cfg.path, skip_mtp=True, nvfp4=a.nvfp4, fp8_head=a.fp8_head)
    eng = Qwen38Engine(cfg, w, max_len=8192)
    head = w.norm("lm_head.weight")
    t = time.perf_counter()
    h, klass = hidden_rows(eng, a.traces, a.rows)
    print(f"[spd43] {h.shape[0]} hidden rows from {a.traces} in {time.perf_counter() - t:.0f} s",
          flush=True)
    t = time.perf_counter()
    nv = head_to_nvfp4(head)
    en = error_norms(head, nv)
    rel = en["e"] / en["w"].clamp_min(1e-30)
    # ŝ as an ideal NVFP4 kernel computes it (the exact product, fp32 sums): which kernel the
    # second stage would use is stage 1's question; the set size is a property of the bound
    wq = torch.cat([nvfp4_rows(nv, r0, min(r0 + 8192, nv.N)) for r0 in range(0, nv.N, 8192)])
    print(f"[spd43] NVFP4 head {nv.nbytes / 1e9:.2f} GB in {time.perf_counter() - t:.0f} s; "
          f"||W - Ŵ|| / ||W|| per row: p50 {pct(rel, 0.5):.4f} p99 {pct(rel, 0.99):.4f}", flush=True)

    ks = [int(x) for x in a.ks.split(",")]
    V = head.N
    sizes = {(m, k): [] for m in ("row", "group") for k in ks}
    misses = {(m, k): 0 for m in ("row", "group") for k in ks}
    ratio = []                       # the realised error over the row bound, per (row, token)
    for r0 in range(0, h.shape[0], a.chunk):
        x = h[r0:r0 + a.chunk].to(torch.bfloat16).contiguous()
        s = head_matmul_fp8(x, head)
        s_hat = x.float() @ wq.T
        for mode in ("row", "group"):
            rad = radius(x, en, mode)
            if mode == "row":
                ratio.append(((s - s_hat).abs() / rad).amax(dim=1))
            for k in ks:
                m = candidates(s_hat, rad, k)
                sizes[(mode, k)].append(m.sum(dim=1).float())
                misses[(mode, k)] += sound(s, m, k)
    res = {"rows": int(h.shape[0]), "vocab": V, "traces": a.traces,
           "rel_err_p50": pct(rel, 0.5), "rel_err_p99": pct(rel, 0.99),
           "max_realised_over_bound": float(torch.cat(ratio).max()), "sets": {}}
    print(f"[spd43] realised |s - ŝ| / row bound: max {res['max_realised_over_bound']:.4f} "
          "(the bound's slack; must be <= 1)")
    print(f"{'bound':<6} {'k':>3} {'p50':>8} {'p90':>8} {'p99':>8} {'max':>8} {'p99 %V':>7} "
          f"{'misses':>6}")
    for (mode, k), v in sizes.items():
        v = torch.cat(v)
        row = {"p50": pct(v, 0.5), "p90": pct(v, 0.9), "p99": pct(v, 0.99), "max": float(v.max()),
               "misses": misses[(mode, k)]}
        per = {}
        for c in sorted(set(klass)):
            sel = v[torch.tensor([x == c for x in klass], device=v.device)]
            per[c] = {"n": int(sel.numel()), "p50": pct(sel, 0.5), "p99": pct(sel, 0.99)}
        row["per_class"] = per
        res["sets"][f"{mode}-k{k}"] = row
        print(f"{mode:<6} {k:>3} {row['p50']:8.0f} {row['p90']:8.0f} {row['p99']:8.0f} "
              f"{row['max']:8.0f} {100 * row['p99'] / V:6.2f}% {row['misses']:>6}")
    go = res["sets"]["group-k1"]["p99"] < 0.05 * V or res["sets"]["row-k1"]["p99"] < 0.05 * V
    res["go"] = bool(go)
    print(f"[spd43] stage 0 {'GO' if go else 'NO-GO'}: p99 of the argmax set against 5 % of the "
          f"vocabulary ({0.05 * V:.0f} rows)")
    json.dump(res, open(os.path.join(a.out, "stage0.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
