"""ENG-225 diagnosis: each part of a prompt chunk on the prefill kernels against the reference's, fed the
SAME inputs, layer by layer.

    PYTHONPATH=<repo>:~/fn-ref/site python tools/fn_prefill_diag.py --pack ~/fn2/packs/3.05 --ctx 1024

The reference forward runs the prompt once with a trace (the streams after every layer). Then for each
layer, from the reference's input streams of that layer: the attention-site mixer, the GDN or attention
block (both paths fed the reference mixer's x, each on its own fresh state), the MLP-site mixer, the MoE
(both fed the reference's x) and the whole layer; max |diff| / max |ref| of each, and the router's
top-10 sets that differ.
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


def rel(a, b) -> float:
    return round(((a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-30)).item(), 5)


@torch.no_grad()
def main() -> None:
    from tools import fn_kernels as K
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--layers", default="")
    ap.add_argument("--json")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    idx, W, banks, tables, _ = load(a.pack, "cuda")
    eng = FlashNextEngine(idx.cfg, W, tables, banks, device="cuda", max_len=a.ctx + 64)
    c = eng.cfg
    tok = AutoTokenizer.from_pretrained(a.pack)
    ids = torch.tensor(class_prompt(tok, "prose", a.ctx), device="cuda")
    T = ids.numel()
    A = eng.new_state(eng.max_len)
    trace: list = []
    eng.forward(ids, A, trace=trace)
    h_in = [eng.embed(ids).repeat(1, c.hc_count)] + trace[:-1]
    B = eng.new_state(eng.max_len)
    R = eng.new_state(eng.max_len)
    eng._pre = eng._pre_ctl(0, T, B)
    rows = []
    want = [int(x) for x in a.layers.split(",")] if a.layers else range(c.num_hidden_layers)
    for li in want:
        p = eng.pre(li)
        h = h_in[li]
        r = {"layer": li, "kind": "gdn" if c.is_linear(li) else "attn"}
        if li in c.ple_layers:
            d_ref = eng.ple(h, ids, li, R)
            d_pre = eng._ple_pre(h, ids, li, B)
            r["ple"] = rel(d_pre, d_ref)
            h = h + d_ref
        x_ref, h0, inj = eng.mixer(h, p + "attn_hyper_connection")
        n = K.hc_norm(h, eng._site_norm(li), c.hidden_size, c.rms_norm_eps)
        x_pre, di = eng._mix_pre(n, p + "attn_hyper_connection")
        x_dec, _ = eng._mix(n, p + "attn_hyper_connection")
        r["mix_attn"] = rel(x_pre, x_ref)
        r["mix_attn_decode_kernels"] = rel(x_dec, x_ref)
        if c.is_linear(li):
            o_ref = eng.gdn(x_ref, li, R)
            o_pre = eng._gdn_pre(x_ref, li, B)
        else:
            o_ref = eng.attention(x_ref, li, R, torch.arange(T, device="cuda"))
            o_pre = eng._attn_pre(x_ref, li, B)
        r["block"] = rel(o_pre, o_ref)
        h2 = eng.inject(h0, o_ref, inj)
        x2_ref, h20, inj2 = eng.mixer(h2, p + "mlp_hyper_connection")
        n2 = K.hc_norm(h2, W[p + "mlp_hyper_connection.hc_norm.weight"], c.hidden_size, c.rms_norm_eps)
        x2_pre, _ = eng._mix_pre(n2, p + "mlp_hyper_connection")
        r["mix_mlp"] = rel(x2_pre, x2_ref)
        m_ref = eng.moe(x2_ref, li)
        trace_ids, eng.route_trace = eng.route_trace, []
        r["moe"] = rel(eng._moe_pre(x2_ref, li), m_ref)
        ids_p = eng.route_trace[-1][1]
        eng.route_trace = trace_ids
        ids_d, _ = K.router(K.gemv(x2_ref, W[p + "mlp.router"]), c.num_experts, c.num_experts_per_tok)
        ids_d = ids_d[:, :-1].long()
        r["router_rows_differ"] = int((torch.sort(ids_p, 1).values != torch.sort(ids_d, 1).values).any(1).sum())
        r["layer_out_ref_vs_trace"] = rel(eng.inject(h20, m_ref, inj2), trace[li])
        rows.append(r)
        print(f"[diag] {json.dumps(r)}", flush=True)
    if a.json:
        json.dump(rows, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
