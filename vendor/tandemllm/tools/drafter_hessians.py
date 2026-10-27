"""Hessians of a DFlash2 drafter's linear inputs, for an EXL3 conversion of the drafter (ENG-215).

exllamav3's `quantize_exl3(weight, H_data, quant_args)` quantises one tensor against H = sum x^T x
of the activations it multiplies; its own converter does not know the DFlash2 architecture. This
captures those sums by running the drafter as it serves -- the target (the loaded weight set, e.g.
EXL3 4.5) teacher-forced over calibration text, the drafter synced at every position and drafting a
block at every position (record_lattice.py's loop) -- with the drafter in bf16 and every linear's
input added to its Hessian:

    qkv     layers.i.self_attn.{q,k,v}_proj share one H (the block rows), and k/v also see the
            context rows the sync projects (`context_kv`): added to the same H
    o       layers.i.self_attn.o_proj
    gate_up layers.i.mlp.{gate,up}_proj share one H
    down    layers.i.mlp.down_proj
    fc      fc (the five tapped target layers, [N, 25600]), its own H
    head    the draft head's input (the drafter's normed block rows `_tokens_from` multiplies by
            lm_head), for a drafter-calibrated EXL3 head (`drafter_exl3_quant.py --head-only`)

Writes one `<name>.pt` per Hessian ({"H": fp32 [in, in] on the CPU, "count": rows}) and an index.

Usage: drafter_hessians.py --ckpt train/ft-b16 --text bench/calib.txt --seqs 8 --len 1024 --out DIR
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["QWEN38_DRAFT_NVFP4"] = "0"          # the bf16 drafter: the weights being quantised
os.environ["QWEN38_DRAFT_FC_NVFP4"] = "0"
os.environ["QWEN38_DRAFT_GRAPH"] = "0"
from engine.config import load_config  # noqa: E402
from engine.drafters import dflash2 as D2  # noqa: E402
from engine.loader import Weights  # noqa: E402
from engine.model import Qwen38Engine  # noqa: E402

GROUP = {"self_attn.q_proj": "qkv", "self_attn.k_proj": "qkv", "self_attn.v_proj": "qkv",
         "self_attn.o_proj": "o", "mlp.gate_proj": "gate_up", "mlp.up_proj": "gate_up",
         "mlp.down_proj": "down"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--text", default="bench/calib.txt")
    ap.add_argument("--seqs", type=int, default=8)
    ap.add_argument("--len", type=int, default=1024)
    ap.add_argument("--prompt", type=int, default=64)
    ap.add_argument("--block", type=int, default=16)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cfg = load_config(None)
    eng = Qwen38Engine(cfg, Weights(cfg.path, device="cuda", skip_mtp=True),
                       max_len=a.prompt + a.len + 64, device="cuda")
    dr = D2.DFlash2Drafter(eng, a.ckpt, blocks=1, max_len=a.prompt + a.len + 64)
    m = dr._build()
    # which group each projection weight belongs to, by tensor identity
    owner: dict[int, str] = {}
    for i in range(m.cfg.num_hidden_layers):
        for proj, g in GROUP.items():
            owner[id(m.w[f"layers.{i}.{proj}.weight"])] = f"layers.{i}.{g}"
    H: dict[str, torch.Tensor] = {}
    count: dict[str, int] = {}

    def add(name: str, x: torch.Tensor) -> None:
        x = x.reshape(-1, x.shape[-1]).float()
        x = x[torch.isfinite(x).all(dim=1)]
        if name not in H:
            H[name] = torch.zeros(x.shape[1], x.shape[1], dtype=torch.float32, device=x.device)
            count[name] = 0
        H[name].addmm_(x.T, x)
        count[name] += x.shape[0]

    lin0 = D2._lin

    def lin(x, w):
        name = owner.get(id(w))
        if name is not None:
            add(name, x)
        return lin0(x, w)
    D2._lin = lin
    pc0 = D2.DFlash2Module.project_context

    def project_context(self, target_hidden):
        add("fc", target_hidden)
        return pc0(self, target_hidden)
    D2.DFlash2Module.project_context = project_context
    tf0 = D2.DFlash2Drafter._tokens_from

    def tokens_from(self, m, pred, anchor, first=0):
        add("head", pred)
        return tf0(self, m, pred, anchor, first)
    D2.DFlash2Drafter._tokens_from = tokens_from

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.path)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ids = tok(open(os.path.join(root, a.text)).read()).input_ids
    need = a.seqs * (a.prompt + a.len)
    while len(ids) < need:
        ids = ids + ids
    t0 = time.perf_counter()
    with torch.no_grad():
        for s in range(a.seqs):
            seq = ids[s * (a.prompt + a.len):(s + 1) * (a.prompt + a.len)]
            prompt, out = seq[:a.prompt], seq[a.prompt:]
            eng.reset()
            dr.reset()
            eng.forward(torch.tensor(prompt, device="cuda"), start=0, last_only=True)
            dr.sync(prompt, eng.hidden_post_norm[0], 0)
            pos, ctx, i = len(prompt), list(prompt), 0
            while i < len(out):
                blk = out[i:i + a.block]
                eng.forward_block(torch.tensor(blk, device="cuda"), start=pos)
                hid = eng.hidden_post_norm[0]
                for j, t in enumerate(blk):
                    dr.propose_tree(ctx + out[i:i + j + 1], budget=0)
                    dr.sync([t], hid[j:j + 1], pos + j, rows=[j])
                pos += len(blk)
                ctx = ctx + list(blk)
                i += len(blk)
            print(f"[hess] seq {s + 1}/{a.seqs}  {time.perf_counter() - t0:.0f} s  "
                  + " ".join(f"{k}:{v}" for k, v in sorted(count.items())[:3]), flush=True)
    index = {}
    for name, h in H.items():
        torch.save({"H": h.cpu(), "count": count[name]}, os.path.join(a.out, f"{name}.pt"))
        index[name] = {"dim": h.shape[0], "count": count[name]}
    json.dump({"ckpt": a.ckpt, "text": a.text, "seqs": a.seqs, "len": a.len, "hessians": index},
              open(os.path.join(a.out, "index.json"), "w"), indent=1)
    print(f"[hess] {len(H)} Hessians -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
