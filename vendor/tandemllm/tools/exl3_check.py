"""The EXL3 kernels' correctness gate on REAL tensors from an EXL3 checkpoint (ENG-199/200).

    python tools/exl3_check.py ~/exl3-quants/4.0 [--per-kind 2] [--rows 1,2,4,8,16,24,32]
                               [--exllamav3] [--json OUT]

For a sample of the checkpoint's projections (every kind, every rate K the checkpoint uses, the
head), at every row count:

  * decode: the trellis through our reconstruct kernel equals the pure-torch decode bit for bit
    (and exllamav3's own `reconstruct` with --exllamav3);
  * product: our skinny GEMM against the fp32 reference (`reference_matmul`: the same two
    roundings the kernels make on purpose, everything else fp32) -- max |diff| relative to max |ref|;
  * row independence: row 0 of every call equals row 0 of the M = 1 call, bit for bit;
  * with --exllamav3: exllamav3's own LinearEXL3.forward on the same weight (its fp16 input path),
    the distance reported against the same reference.

The activations are N(0, 1) rows scaled to the RMS of a real normed hidden state (about 1), so
the products have realistic magnitudes; a rotated input that would overflow fp16 is reported.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict

import torch
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools import exl3_linear as X  # noqa: E402

KINDS = ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj", "linear_attn.in_proj_qkv",
         "linear_attn.in_proj_z", "linear_attn.out_proj", "self_attn.q_proj", "self_attn.k_proj",
         "self_attn.v_proj", "self_attn.o_proj", "lm_head")


def index(path: str) -> dict:
    idx = os.path.join(path, "model.safetensors.index.json")
    wm = json.load(open(idx))["weight_map"]
    return {k[: -len(".trellis")]: os.path.join(path, f) for k, f in wm.items()
            if k.endswith(".trellis") and not k.startswith("mtp.")}


def kind_of(base: str) -> str | None:
    for k in KINDS:
        if base.endswith(k):
            return k
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--per-kind", type=int, default=2, help="tensors per (kind, K)")
    ap.add_argument("--rows", default="1,2,4,8,16,24,32")
    ap.add_argument("--exllamav3", action="store_true")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    rows = [int(r) for r in a.rows.split(",")]
    bases = index(a.path)
    # the rate of each tensor from its trellis shape, without loading it
    picks: dict = defaultdict(list)
    for base, fpath in sorted(bases.items(), key=lambda kv: [int(t) if t.isdigit() else t
                                                             for t in re.split(r"(\d+)", kv[0])]):
        k = kind_of(base)
        if k is None:
            continue
        with safe_open(fpath, framework="pt", device="cpu") as f:
            kb = f.get_slice(f"{base}.trellis").get_shape()[2] // 16
        if len(picks[(k, kb)]) < a.per_kind:
            picks[(k, kb)].append((base, fpath))
    ext = None
    if a.exllamav3:
        from exllamav3.ext import exllamav3_ext as ext  # noqa: F401
        from exllamav3.modules.quant.exl3 import LinearEXL3
    out = []
    g = torch.Generator(device="cuda").manual_seed(0)
    for (k, kb), items in sorted(picks.items()):
        for base, fpath in items:
            with safe_open(fpath, framework="pt", device="cuda") as f:
                w = X.read_block(f, base, out_fp32=(k == "lm_head"))
            rec = {"tensor": base, "kind": k, "K": kb, "shape": list(w.shape)}
            a_ = X.decode_trellis(w.t, 0, min(w.N // 16, 64))
            b_ = w.inner(0, min(w.N, 1024))
            rec["decode_bitexact"] = bool(torch.equal(a_.view(torch.int16), b_.view(torch.int16)))
            if ext is not None:
                c_ = torch.empty(w.K, w.N, dtype=torch.half, device="cuda")
                ext.reconstruct(c_, w.t, kb, False, True)
                rec["decode_eq_exllamav3"] = bool(torch.equal(c_[:, :1024].view(torch.int16),
                                                              b_.view(torch.int16)))
                del c_
            x = torch.randn(max(rows), w.K, device="cuda", generator=g).to(torch.bfloat16)
            xh = X.rotate_in(x, w)
            rec["xh_finite"] = bool(torch.isfinite(xh).all())
            ref = X.reference_matmul(x, w).float()
            worst, single, indep = 0.0, None, True
            per_m = {}
            for M in rows:
                y = X.exl3_matmul(x[:M], w).float()
                rel = ((y - ref[:M]).abs().max() / ref[:M].abs().max().clamp_min(1e-30)).item()
                per_m[M] = rel
                worst = max(worst, rel)
                if single is None:
                    single = y[:1].clone()
                indep = indep and bool(torch.equal(y[:1], single))
            rec["rel_max"] = worst
            rec["rel_by_M"] = per_m
            rec["row_independent"] = indep
            if ext is not None:
                mul1 = torch.tensor([X.MUL1 - (1 << 32)], dtype=torch.int32, device="cuda")
                lin = LinearEXL3(None, w.K, w.N, suh=w.suh[0], svh=w.svh, trellis=w.t, mul1=mul1,
                                 out_dtype=torch.float)
                ex = lin.forward(x.half().contiguous(), {}).float()
                rec["exllamav3_rel"] = ((ex - ref).abs().max() / ref.abs().max()).item()
            out.append(rec)
            print(f"{base:55s} K{kb} {w.N}x{w.K}  decode {'=' if rec['decode_bitexact'] else 'DIFF'}"
                  + (f" (exl3 {'=' if rec.get('decode_eq_exllamav3') else 'DIFF'})" if ext else "")
                  + f"  ours rel {worst:.2e}"
                  + (f"  exllamav3 rel {rec['exllamav3_rel']:.2e}" if ext else "")
                  + f"  row-indep {indep}  xh finite {rec['xh_finite']}", flush=True)
            del w, x, ref
            torch.cuda.empty_cache()
    ok = all(r["decode_bitexact"] and r["row_independent"] and r["xh_finite"] and r["rel_max"] < 1e-2
             for r in out)
    print(f"{len(out)} tensors, K {sorted({r['K'] for r in out})}: {'PASS' if ok else 'FAIL'}")
    if a.json:
        json.dump({"path": a.path, "rows": rows, "results": out, "pass": ok}, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
