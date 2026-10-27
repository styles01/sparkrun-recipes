"""Quality of the hyper-connection mixers' weight forms against the pack's own fp16 mixers (ENG-221 byte
cuts, P2.4): int8 rows (what decode serves today), int8 or int4 with a scale a group of G
weights (`int8g128`, `int4g32`), as fake-quantised fp16 weights on the same kernels, so only the weight
values differ; `form@scope` quantises only some mixers (`@down`, `@up`, `@L0-11`) to find where a
loss comes from.

    PYTHONPATH=<repo>:~/fn-ref/site python tools/fn_mix_quality.py --pack ~/fn2/packs/3.05 \
        --ctx 8192 --classes prose,code --variants int8,int4g32,int4g128 --json out.json

Per class: the prompt through the prefill kernels with each variant's mixers, every row's next-token
logits (the same head on each run's streams) against the fp16 run's: top-1 agreement, mean and p99 of
KL(fp16 || variant), max logit diff; then greedy decode of `--new` tokens from each state against the
fp16 run's text (first difference and the fp16 run's margin there). The noise yardstick is `ref`: the
fp16 mixers through the reference forward instead of the prefill kernels (two arithmetics, the same
weights).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.models.flashnext.engine import DecodeGraph, FlashNextEngine  # noqa: E402
from engine.models.flashnext.load import load  # noqa: E402
from engine.models.flashnext.pack import Int8Rows  # noqa: E402
from tools.fn_generate import class_prompt, mem_available_gib  # noqa: E402
from tools.fn_prefill_check import head_logits  # noqa: E402


def fake(w: torch.Tensor, form: str) -> torch.Tensor:
    """The fp16 weight w [N, K] after a round trip through `form`: int8 (one scale a row, as served),
    int8gG (int8 with a scale a group of G weights) or int4gG."""
    if form == "int8":
        return Int8Rows.of(w).dequant().half()
    N, K = w.shape
    bits, G = form.split("g")
    G = math.gcd(int(G), K)                             # the up projection's K is 320: g128 -> 64
    lim = 127.0 if bits == "int8" else 7.0
    g = w.float().view(N, K // G, G)
    s = g.abs().amax(-1, keepdim=True).clamp_min(1e-30) / lim
    return (torch.round(g / s).clamp(-lim - (bits != "int8"), lim) * s).view(N, K).half()


def in_scope(key: str, scope: str | None) -> bool:
    """A variant's mixers: all (None), `down` (the down|inject weights), `up`, or `L<a>-<b>` (layers a..b)."""
    if scope is None:
        return True
    if scope in ("down", "up"):
        return key.endswith(".down_inject") == (scope == "down")
    a, b = (int(x) for x in scope[1:].split("-"))
    return key.startswith("layers.") and a <= int(key.split(".")[1]) <= b


@torch.no_grad()
def run(eng: FlashNextEngine, ids: list[int], new: int, eos: set, ref: bool = False):
    st = eng.new_state(eng.max_len)
    stacks = []
    chunk = 4096
    for i in range(0, len(ids), chunk):
        t = torch.tensor(ids[i:i + chunk], device=eng.device)
        (eng.forward if ref else eng.forward_pre)(t, st, head="none")
        stacks.append(eng.stack)
        if st.length > 4096:
            torch.cuda.empty_cache()
    if ref:
        eng.build_pool(st)
    stack = torch.cat(stacks)
    g = DecodeGraph(eng, st)
    toks, margins = [int(head_logits(eng, stack[-1:])[0].argmax())], []
    while len(toks) < new and toks[-1] not in eos:
        v, i = torch.topk(g(toks[-1])[0], 2)
        toks.append(int(i[0]))
        margins.append(float(v[0] - v[1]))
    del g, st
    torch.cuda.empty_cache()
    return stack, toks, margins


@torch.no_grad()
def compare(eng, base, other, block: int = 256) -> dict:
    sb, tb, mb = base
    so, to, _ = other
    agree, kls, dmax = [], [], []
    for r0 in range(0, sb.shape[0], block):
        lb, lo = head_logits(eng, sb[r0:r0 + block]), head_logits(eng, so[r0:r0 + block])
        agree.append(lb.argmax(-1) == lo.argmax(-1))
        pb = torch.log_softmax(lb, -1)
        kls.append((pb.exp() * (pb - torch.log_softmax(lo, -1))).sum(-1))
        dmax.append((lb - lo).abs().amax(-1))
        del lb, lo, pb
    agree, kls, dmax = torch.cat(agree).float(), torch.cat(kls).sort().values, torch.cat(dmax).sort().values
    d = next((j for j in range(min(len(tb), len(to))) if tb[j] != to[j]), None)
    return {"agree": round(float(agree.mean()), 5), "kl_mean": round(float(kls.mean()), 5),
            "kl_p99": round(float(kls[int(len(kls) * 0.99)]), 4), "max_logit_diff_p50": round(float(dmax[len(dmax) // 2]), 3),
            "max_logit_diff_p99": round(float(dmax[int(len(dmax) * 0.99)]), 3),
            "continuation": {"tokens": len(tb), "equal": tb == to, "first_diff": d,
                             "base_margin_at_diff": None if d is None or d == 0 else round(mb[d - 1], 4)}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--classes", default="prose,code")
    ap.add_argument("--variants", default="int8,int4g32,int4g128")
    ap.add_argument("--new", type=int, default=128)
    ap.add_argument("--json")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    idx, W, banks, tables, timing = load(a.pack, "cuda", mix="fp16")
    eng = FlashNextEngine(idx.cfg, W, tables, banks, device="cuda", max_len=a.ctx + a.new + 64)
    tok = AutoTokenizer.from_pretrained(a.pack)
    eos = set(idx.cfg.eos_token_ids)
    keys = [k for k in W if k.endswith(".down_inject") or k.endswith("input_mix_weight_up.weight")]
    orig = {k: W[k].clone() for k in keys}
    res = {"pack": a.pack, "mixer_tensors": len(keys), "mem_after_load_gib": mem_available_gib(),
           "knobs": {"ctx": a.ctx, "new": a.new, "greedy": True, "prefill": "kernels, 4,096-row chunks",
                     "mix_storage": "fp16 (fake-quantised values)"}, "rows": []}
    for c in a.classes.split(","):
        ids = class_prompt(tok, c, a.ctx)
        base = run(eng, ids, a.new, eos)
        row = {"class": c, "prompt_tokens": len(ids), "ref": compare(eng, base, run(eng, ids, a.new, eos, ref=True))}
        for v in a.variants.split(","):
            form, scope = (v.split("@") + [None])[:2]
            for k in keys:
                if in_scope(k, scope):
                    W[k].copy_(fake(orig[k], form))
            row[v] = compare(eng, base, run(eng, ids, a.new, eos))
            for k in keys:
                W[k].copy_(orig[k])
        row["mem_gib"] = mem_available_gib()
        res["rows"].append(row)
        print(f"[mix-quality] {json.dumps(row)}", flush=True)
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
