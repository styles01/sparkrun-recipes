"""Greedy generation from the reference build -- the quickest way to tell a loaded model from a
mis-loaded one. A teacher-forced loss can be argued with; forty tokens of output cannot."""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.config import load_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--chat", action="store_true")
    ap.add_argument("--nll", default=None, help="a text file to score teacher-forced")
    ap.add_argument("--nll-n", type=int, default=512)
    a = ap.parse_args()

    from transformers import AutoConfig, AutoTokenizer
    from transformers.models.qwen3_5.modeling_qwen3_5 import (Qwen3_5ForCausalLM,
                                                              Qwen3_5TextRotaryEmbedding)
    from engine.loader import Weights

    cfg = load_config(a.model)
    tok = AutoTokenizer.from_pretrained(cfg.path)
    hf = AutoConfig.from_pretrained(cfg.path).text_config
    hf._attn_implementation = "sdpa"
    with torch.device("meta"):
        model = Qwen3_5ForCausalLM(hf)
    w = Weights(cfg.path, device=a.device, skip_mtp=True)
    sd: dict[str, torch.Tensor] = {}
    for name, t in list(w.t.items()):
        if name == "lm_head.weight":
            sd[name] = t
        elif name.startswith("layers.") or name in ("embed_tokens.weight", "norm.weight"):
            sd["model." + name] = t
    for base in sorted(w.q):
        if base.startswith("layers."):
            blk = w.q.pop(base)
            sd["model." + base + ".weight"] = blk.dequant()
            blk.w = None
    torch.cuda.empty_cache()
    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    print(f"[gen] missing {len(missing)} unexpected {len(unexpected)}")
    print(f"[gen] missing sample {missing[:6]}")
    print(f"[gen] unexpected sample {unexpected[:6]}")
    model.model.rotary_emb = Qwen3_5TextRotaryEmbedding(hf, device=a.device)
    model.eval()
    del sd, w

    if a.nll:
        import torch.nn.functional as F
        natural = open(a.nll).read()
        nids = tok(natural, return_tensors="pt").input_ids[:, :a.nll_n].to(a.device)
        with torch.no_grad():
            lg = model(input_ids=nids).logits[0].float()
        loss = F.cross_entropy(lg[:-1], nids[0, 1:], reduction="none")
        q = loss.shape[0] // 4
        print("[nll] %d tokens, mean %.3f, by quarter %s"
              % (nids.shape[1], loss.mean().item(),
                 [round(loss[i * q:(i + 1) * q].mean().item(), 3) for i in range(4)]))

    text = a.prompt
    if a.chat:
        text = tok.apply_chat_template([{"role": "user", "content": a.prompt}],
                                       tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").input_ids.to(a.device)
    print(f"[gen] {ids.shape[1]} prompt tokens")
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=a.n, do_sample=False,
                             pad_token_id=tok.pad_token_id or 0)
    print("=" * 72)
    print(tok.decode(out[0, ids.shape[1]:], skip_special_tokens=False))
    print("=" * 72)


if __name__ == "__main__":
    main()
