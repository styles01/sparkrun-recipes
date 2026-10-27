"""ENG-391: a Qwen3.5 MoE EXL3 pack's quality on this engine (the Spark).

  bundle   the converter job's reference bundle (8 wikitext rows x 1,024: input ids, the pack's logits from
           the converter's own forward, the BF16 checkpoint's logits): every row through the prefill path,
           teacher-forced; KL and top-1 against BF16 (the pack's quality) and against the converter's
           forward of the same pack (the engine's correctness), with top-1 agreement by the reference's
           top-1 - top-2 margin
  greedy   N prompts x 64 greedy tokens through the decode rows (the served path); the ids, tokens and
           logits saved for a teacher-forced run of another runtime on the same pack (`--save`)

    python tools/q35m_quality.py --pack DIR --bundle DIR --out q.json [--save runs.pt]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROMPTS = [
    "Explain how a refrigerator keeps food cold, for a curious ten-year-old.",
    "Schreibe eine kurze Nachricht an einen Kollegen, dass das Meeting auf Donnerstag verschoben wird.",
    "Write a Python function that merges two sorted lists into one sorted list, with a docstring.",
    "What are three differences between TCP and UDP? Use a short list.",
    "Summarise the plot of Romeo and Juliet in four sentences.",
    "Write a SQL query that returns the ten customers with the highest total order value.",
    "A train leaves at 14:35 and arrives at 18:10. How long is the journey? Show the steps.",
    "Give me a packing list for a three-day hiking trip in autumn.",
]
EDGES = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, float("inf")]


def buckets_new() -> dict:
    return {f"{a}-{b}": [0, 0] for a, b in zip(EDGES, EDGES[1:])}


def bucket_add(bk: dict, ref: torch.Tensor, eng: torch.Tensor) -> None:
    top2 = ref.topk(2, -1).values
    margin = (top2[:, 0] - top2[:, 1]).tolist()
    agree = (eng.argmax(-1) == ref.argmax(-1)).tolist()
    for m, a in zip(margin, agree):
        for lo, hi in zip(EDGES, EDGES[1:]):
            if lo <= m < hi:
                bk[f"{lo}-{hi}"][0] += 1
                bk[f"{lo}-{hi}"][1] += int(a)
                break


def kl_rows(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
    """KL(p || q) per row (fp32)."""
    lp = torch.log_softmax(p_logits.float(), -1)
    lq = torch.log_softmax(q_logits.float(), -1)
    return (lp.exp() * (lp - lq)).sum(-1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--bundle", default="")
    ap.add_argument("--tag", default="4.0")
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--out", required=True)
    ap.add_argument("--save", default="")
    ap.add_argument("--no-greedy", action="store_true")
    a = ap.parse_args()
    from safetensors import safe_open
    from transformers import AutoTokenizer

    from engine.models.qwen3_5_moe.engine import Qwen35MoeEngine
    from engine.models.qwen3_5_moe.load import load
    idx, W, banks, timing = load(a.pack, "cuda", mtp=False)
    eng = Qwen35MoeEngine(idx.cfg, W, banks, device="cuda", max_len=8192)
    V = idx.cfg.vocab_size
    out: dict = {"pack": a.pack, "load": timing}
    if a.bundle:
        fi = safe_open(os.path.join(a.bundle, f"input_ids_{a.tag}.safetensors"), "pt")
        fb = safe_open(os.path.join(a.bundle, "logits_bf16.safetensors"), "pt")
        fx = safe_open(os.path.join(a.bundle, f"logits_exl3_{a.tag}.safetensors"), "pt")
        # one tensor a row ("tensor.<r>": ids [1, T], logits [1, T, vocab] fp16), as eval/model_diff.py saves them
        keys = sorted(fi.keys(), key=lambda k: int(k.rsplit(".", 1)[-1]))
        ids = [fi.get_tensor(k) for k in keys]
        acc = {"bf16": {"kl": 0.0, "kl_rev": 0.0, "top1": 0, "nll": 0.0, "bk": buckets_new()},
               "exl3": {"kl": 0.0, "kl_rev": 0.0, "top1": 0, "nll": 0.0, "bk": buckets_new()}}
        n_pos, nll_eng = 0, 0.0
        for r, key in enumerate(keys):
            row = ids[r].view(-1).long()
            st = eng.new_state(2048)
            lg = eng.forward_pre(row.cuda(), st, head="all").float()          # [T, V]
            T = row.numel()
            lab = row[1:].cuda()
            nll_eng += float(torch.nn.functional.cross_entropy(lg[:-1], lab, reduction="sum"))
            for name, f in (("bf16", fb), ("exl3", fx)):
                ref = f.get_tensor(key)[0, :, :V].float().cuda()
                k = acc[name]
                k["kl"] += float(kl_rows(ref, lg).sum())
                k["kl_rev"] += float(kl_rows(lg, ref).sum())
                k["top1"] += int((ref.argmax(-1) == lg.argmax(-1)).sum())
                k["nll"] += float(torch.nn.functional.cross_entropy(ref[:-1], lab, reduction="sum"))
                bucket_add(k["bk"], ref, lg)
                del ref
            n_pos += T
            del st, lg
            torch.cuda.empty_cache()
            print(f"[q35m-quality] bundle row {r}: done", flush=True)
        res = {"rows": len(ids), "positions": n_pos,
               "engine_ppl": round(math.exp(nll_eng / (n_pos - len(ids))), 4)}
        for name, k in acc.items():
            res[f"vs_{name}"] = {"kl": round(k["kl"] / n_pos, 5), "kl_reverse": round(k["kl_rev"] / n_pos, 5),
                                 "top1_pct": round(100 * k["top1"] / n_pos, 2),
                                 "ref_ppl": round(math.exp(k["nll"] / (n_pos - len(ids))), 4),
                                 "top1_by_margin": {b: {"positions": v[0], "agree": v[1]} for b, v in k["bk"].items()}}
        out["bundle"] = res
        print("[q35m-quality] bundle:", json.dumps(res), flush=True)
    if a.no_greedy:
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1)
        return 0
    tok = AutoTokenizer.from_pretrained(a.pack)
    runs = []
    t0 = time.time()
    for p in PROMPTS:
        text = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
        pid = tok(text, add_special_tokens=False)["input_ids"]
        st = eng.new_state(2048)
        lg = eng.prefill(torch.tensor(pid), st)
        rows = [lg[0].float().cpu()]
        gen = [int(lg[0].argmax())]
        for _ in range(a.tokens - 1):
            lg = eng.decode_eager(gen[-1], st)
            rows.append(lg[0].float().cpu())
            gen.append(int(lg[0].argmax()))
        runs.append({"prompt": p, "ids": pid, "gen": gen, "logits": torch.stack(rows).to(torch.bfloat16)})
        del st
    out["greedy"] = {"prompts": len(runs), "tokens": a.tokens, "s": round(time.time() - t0, 1),
                     "texts": [tok.decode(r["gen"]) for r in runs]}
    if a.save:
        torch.save({"runs": runs}, a.save)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print("[q35m-quality] done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
