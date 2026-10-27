"""Whole-model checks of a weight configuration in THIS engine: held-out perplexity and greedy text.

    python tools/exl3_eval.py --exl3 ~/exl3-quants/4.0 --rows heldout_rows.safetensors \
        --gen 256 --json out.json
    python tools/exl3_eval.py --nvfp4 A,B,C --fp8-head H --rows ... --gen 256 --json nv.json

* `--rows`: exllamav3's pre-tokenized test rows (`ids` [rows, len], optional `ranges` [rows, 2]
  or `prefix_len`): logits at [a, b - 1) scored against the tokens at [a + 1, b), log-softmax over
  the tokenizer's real vocabulary (1 + its largest id), exactly as exllamav3's eval/ppl.py scores
  them -- so the number reads next to the BF16 reference that script wrote for the same rows.
  One prefill a row (the engine's prefill path), all logits kept.
* `--gen N`: plain greedy decoding (one token a step, the M = 1 path) of N tokens on fixed chat
  prompts, thinking off; the token ids are saved so another engine's text can be compared token
  by token (tools/exl3_eval.py --compare A.json B.json).

One configuration a process: a weight set is loaded once, used, and the process exits.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROMPTS = [
    ("canal", "Explain, in one continuous passage of prose, how a lock and a weir differ and why a "
              "canal needs both."),
    ("letter", "Write a short formal letter to a landlord reporting a broken heating system and "
               "asking for a repair date."),
    ("code", "Write a Python function that merges two sorted lists into one sorted list without "
             "using sort(), and explain its running time."),
    ("german", "Erkläre in einfachen Worten, warum der Himmel blau ist."),
]


def load_rows(path: str):
    from safetensors import safe_open
    with safe_open(path, framework="pt", device="cpu") as f:
        keys = set(f.keys())
        ids = f.get_tensor("ids").to(torch.long)
        if "ranges" in keys:
            ranges = [tuple(int(v) for v in r) for r in f.get_tensor("ranges").tolist()]
        else:
            p = int(f.get_tensor("prefix_len").flatten()[0]) if "prefix_len" in keys else 0
            ranges = [(p, ids.shape[1])] * ids.shape[0]
    return ids, ranges


@torch.no_grad()
def perplexity(eng, ids: torch.Tensor, ranges, vocab: int, save_ref: str | None = None,
               ref: str | None = None) -> dict:
    """NLL over the scored positions; with `save_ref`, every scored position's full log-softmax
    is written (fp16, one file a row) as the reference; with `ref`, the KL divergence
    KL(ref || this) and the top-1 agreement against that reference, on the same positions."""
    lp_sum, n = 0.0, 0
    per_row = []
    kl_sum, agree, kl_rows = 0.0, 0, []
    for r, (a, b) in enumerate(ranges):
        eng.reset()
        row = ids[r, :b].to(eng.device)
        logits = eng.forward(row, start=0)[0]
        lg = logits[a:b - 1, :vocab].float()
        tgt = row[a + 1:b]
        lsm = F.log_softmax(lg, dim=-1)
        lp = lsm.gather(1, tgt[:, None]).squeeze(1)
        lp_sum += float(lp.sum())
        n += tgt.numel()
        per_row.append(float(-lp.mean()))
        if save_ref:
            os.makedirs(save_ref, exist_ok=True)
            torch.save(lsm.to(torch.float16).cpu(), os.path.join(save_ref, f"row{r}.pt"))
        if ref:
            rl = torch.load(os.path.join(ref, f"row{r}.pt")).to(lsm.device).float()
            kl = (rl.exp() * (rl - lsm)).sum(-1)               # per position
            kl_sum += float(kl.sum())
            kl_rows.append(float(kl.mean()))
            agree += int((rl.argmax(-1) == lsm.argmax(-1)).sum())
            del rl, kl
        del logits, lg, lp, lsm
    nll = -lp_sum / n
    out = {"nll": nll, "ppl": math.exp(nll), "tokens": n, "rows": len(ranges), "nll_by_row": per_row}
    if ref:
        out.update(kl=kl_sum / n, top1=agree / n, kl_by_row=kl_rows, ref=ref)
    return out


@torch.no_grad()
def greedy(eng, tok, prompt: str, new: int) -> list[int]:
    msg = [{"role": "user", "content": prompt}]
    ids = tok.apply_chat_template(msg, add_generation_prompt=True, return_tensors="pt",
                                  return_dict=True, enable_thinking=False)["input_ids"][0]
    eng.reset()
    out, gaps = [], []
    logits = eng.forward(ids.to(eng.device), start=0, last_only=True)[0, -1]
    pos = ids.numel()
    eos = set(getattr(eng.cfg, "eos_token_ids", None) or [tok.eos_token_id])
    for _ in range(new):
        top = logits.float().topk(2)
        nxt = int(top.indices[0])
        gaps.append(float(top.values[0] - top.values[1]))   # the margin a numeric change must beat
        out.append(nxt)
        if nxt in eos:
            break
        logits = eng.forward(torch.tensor([nxt], device=eng.device), start=pos, last_only=True)[0, -1]
        pos += 1
    greedy.gaps = gaps
    return [int(v) for v in ids.tolist()], out


def compare(a_path: str, b_path: str) -> None:
    A, B = json.load(open(a_path)), json.load(open(b_path))
    for name in A.get("gen", {}):
        if name not in B.get("gen", {}):
            continue
        a, b = A["gen"][name]["ids"], B["gen"][name]["ids"]
        same = 0
        for x, y in zip(a, b):
            if x != y:
                break
            same += 1
        n = min(len(a), len(b))
        agree = sum(1 for x, y in zip(a, b) if x == y) / max(n, 1)
        gap = A["gen"][name].get("gap") or []
        g = f", our top-2 logit gap there {gap[same]:.3f} (median gap {sorted(gap)[len(gap) // 2]:.2f})" \
            if same < len(gap) else ""
        print(f"{name:8s} first divergence at token {same} of {n}  (positional agreement {agree:.3f}){g}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="base snapshot (default: the engine's)")
    ap.add_argument("--exl3", default=None)
    ap.add_argument("--nvfp4", default=None)
    ap.add_argument("--fp8-head", default=None)
    ap.add_argument("--rows", default=None)
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--gen", type=int, default=0)
    ap.add_argument("--json", default=None)
    ap.add_argument("--save-ref", default=None, help="write this configuration's log-softmax as the reference")
    ap.add_argument("--ref", default=None, help="KL and top-1 agreement against a saved reference")
    ap.add_argument("--compare", nargs=2, default=None)
    a = ap.parse_args()
    if a.compare:
        compare(*a.compare)
        return
    from engine.config import load_config
    from engine.loader import Weights
    from engine.model import Qwen38Engine
    from transformers import AutoTokenizer
    t0 = time.time()
    cfg = load_config(a.model)
    w = Weights(cfg.path, skip_mtp=True, nvfp4=a.nvfp4 or "", fp8_head=a.fp8_head or "",
                exl3=a.exl3 or "")
    eng = Qwen38Engine(cfg, w, max_len=8192)
    tok = AutoTokenizer.from_pretrained(cfg.path)
    vocab = 1 + max(tok.get_vocab().values())
    res = {"model": cfg.path, "exl3": a.exl3, "nvfp4": a.nvfp4, "fp8_head": a.fp8_head,
           "report": w.report(), "decode_step_GB": w.decode_step_bytes(cfg.num_hidden_layers),
           "load_s": round(time.time() - t0, 1), "vocab": vocab}
    print(f"[eval] loaded in {res['load_s']} s: {res['report']}; step {res['decode_step_GB']}", flush=True)
    if a.rows:
        ids, ranges = load_rows(os.path.expanduser(a.rows))
        if a.max_rows:
            ids, ranges = ids[:a.max_rows], ranges[:a.max_rows]
        t1 = time.time()
        res["ppl_rows"] = perplexity(eng, ids, ranges, vocab,
                                     save_ref=os.path.expanduser(a.save_ref) if a.save_ref else None,
                                     ref=os.path.expanduser(a.ref) if a.ref else None)
        res["ppl_rows"]["s"] = round(time.time() - t1, 1)
        kl = (f", KL vs ref {res['ppl_rows']['kl']:.5f}, top-1 {res['ppl_rows']['top1']:.4f}"
              if "kl" in res["ppl_rows"] else "")
        print(f"[eval] rows: ppl {res['ppl_rows']['ppl']:.6f} nll {res['ppl_rows']['nll']:.5f} over "
              f"{res['ppl_rows']['tokens']} tokens ({res['ppl_rows']['s']} s){kl}", flush=True)
    if a.gen:
        res["gen"] = {}
        for name, prompt in PROMPTS:
            t1 = time.time()
            p_ids, out = greedy(eng, tok, prompt, a.gen)
            dt = time.time() - t1
            text = tok.decode(out, skip_special_tokens=True)
            res["gen"][name] = {"prompt_ids": p_ids, "ids": out, "text": text, "s": round(dt, 1),
                                "gap": [round(v, 4) for v in greedy.gaps]}
            print(f"[eval] gen/{name}: {len(out)} tokens in {dt:.1f} s: {text[:200]!r}", flush=True)
    if a.json:
        json.dump(res, open(os.path.expanduser(a.json), "w"), indent=1)


if __name__ == "__main__":
    main()
