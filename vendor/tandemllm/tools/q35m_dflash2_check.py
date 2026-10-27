"""ENG-391: the DFlash2 head on a Qwen3.5 MoE pack (the Spark): the taps a decode step writes against the
prefill's, then the load-time gate (the mean run of drafted tokens the trunk's greedy keeps) on a few prompts.

    python tools/q35m_dflash2_check.py --pack DIR --dflash2 DIR
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROMPTS = ["Explain how a refrigerator keeps food cold, for a curious ten-year-old.",
           "Write a Python function that merges two sorted lists into one sorted list, with a docstring.",
           "Give me a packing list for a three-day hiking trip in autumn."]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--dflash2", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--shift", type=int, default=0, help="taps at the target layer id + shift")
    a = ap.parse_args()
    from transformers import AutoTokenizer

    from engine.drafters.lilicorr import gate
    from engine.models.qwen3_5_moe import dflash2 as D2
    from engine.models.qwen3_5_moe.engine import Qwen35MoeEngine
    from engine.models.qwen3_5_moe.load import load
    from engine.models.qwen3_5_moe.serving import Qwen35MoeServing
    idx, W, banks, _ = load(a.pack, "cuda", mtp=False)
    eng = Qwen35MoeEngine(idx.cfg, W, banks, device="cuda", max_len=8192)
    D2.TAP_SHIFT = a.shift
    dc = D2.config_of(a.dflash2)
    eng.set_taps(dc["target_layer_ids"], dc["window"])
    srv = Qwen35MoeServing(eng, 8192)
    head = D2.build_head(srv, a.dflash2)
    tok = AutoTokenizer.from_pretrained(a.pack)
    out = {"gates": []}
    # the taps of decode steps equal the prefill's rows of the same tokens
    text = tok.apply_chat_template([{"role": "user", "content": PROMPTS[0]}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)
    ids = tok(text, add_special_tokens=False)["input_ids"]
    st = eng.new_state(4096)
    lg = eng.prefill(torch.tensor(ids), st)
    seq = list(ids) + [int(lg[0].argmax())]
    for _ in range(16):
        seq.append(int(eng.decode_eager(seq[-1], st)[0].argmax()))
    st2 = eng.new_state(4096)
    eng.prefill(torch.tensor(seq[:-1]), st2)
    R = eng.ring_rows
    pos = torch.arange(len(ids), len(seq) - 1) % R
    d = (st.ring.index_select(0, pos.cuda()).float() - st2.ring.index_select(0, pos.cuda()).float()).abs()
    scale = st2.ring.index_select(0, pos.cuda()).float().abs().amax()
    out["taps_decode_vs_prefill"] = {"rows": int(pos.numel()), "max_abs": round(float(d.max()), 4),
                                     "scale": round(float(scale), 2)}
    print("[dflash2-check] taps:", out["taps_decode_vs_prefill"], flush=True)
    for p in PROMPTS:
        text = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
        pid = tok(text, add_special_tokens=False)["input_ids"]
        g = gate(eng, head, eng.new_state(len(pid) + 256), pid, steps=96, every=4)
        out["gates"].append({"prompt": p[:40], "mean_run": g["mean_run"], "positions": g["positions"]})
        print("[dflash2-check]", out["gates"][-1], flush=True)
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
