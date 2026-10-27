"""SPD-107: one quality comparison for every LFM2.5 profile, through this engine, against one reference.

    python tools/lfm_quality.py sets --bf16 <dir> --out sets.safetensors [--rows 100]
    python tools/lfm_quality.py selfgen --bf16 <dir> --sets sets.safetensors [--prompts 48 --tokens 384]
    python tools/lfm_quality.py eval --bf16 <dir> --pack <dir> --sets sets.safetensors --json out.json
    python tools/lfm_quality.py compare a.json b.json

`sets` (needs the `datasets` package): rows of 2,048 tokens of wikitext-2 (test, the raw text joined) and of
OpenAssistant oasst1 validation pairs in LFM2.5's chat template (an assistant reply after its user turn, no
reasoning block), the shapes exllamav3's model_diff reads. `selfgen` adds the model's own text: the BF16
engine's greedy answers, reasoning on, to oasst1 validation user turns the chat rows do not hold, each a row
of its own (prompt and answer); what the server serves looks like this.

`eval` runs the BF16 checkpoint and a pack in one process, each row through both engines' prefill (every
row's logits, fp32), and compares them position by position: the KL divergence KL(BF16 || pack) of the
softmax over the whole vocabulary and whether the top tokens agree. Every pack is held to the same rows and
the same reference. Per set: the mean KL, the top-1 agreement and each row's mean KL (for `compare`).

`compare`: per set, the mean of the per-row differences in KL between two packs and a bootstrap 95 %
interval over the rows (the same rows in both, so the difference is paired).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch  # noqa: E402

ROW = 2048
CHAT = "<|startoftext|><|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n{a}<|im_end|>\n"


def cmd_sets(a) -> None:
    from datasets import load_dataset
    from safetensors.torch import save_file
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(a.bf16, "tokenizer.json"))
    wiki = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])
    w = tok.encode(wiki, add_special_tokens=False).ids
    d = load_dataset("OpenAssistant/oasst1", split="validation")
    byid = {r["message_id"]: r for r in d}
    ids, used = [], set()
    for r in d:
        if r["role"] != "assistant" or r["parent_id"] not in byid or len(ids) >= a.rows * ROW:
            continue
        ids += tok.encode(CHAT.format(q=byid[r["parent_id"]]["text"], a=r["text"]), add_special_tokens=False).ids
        used.add(r["parent_id"])
    # the user turns no chat row holds, for selfgen
    rest = [r["text"] for r in d if r["role"] == "prompter" and r["parent_id"] is None
            and r["message_id"] not in used and 20 <= len(r["text"]) <= 600]
    n_w, n_c = min(a.rows, len(w) // ROW), min(a.rows, len(ids) // ROW)
    save_file({"wikitext": torch.tensor(w[:n_w * ROW]).view(n_w, ROW),
               "oasst1": torch.tensor(ids[:n_c * ROW]).view(n_c, ROW)},
              a.out, metadata={"prompts": json.dumps(rest)})
    print(json.dumps({"wikitext_rows": n_w, "oasst1_rows": n_c, "free_prompts": len(rest)}))


def _engine(path: str, max_len: int):
    from engine.models.lfm2_moe.engine import Lfm2MoeEngine
    from engine.models.lfm2_moe.load import apply_splits, load
    idx, W, experts, _ = load(path, "cuda", log=None)
    if idx.is_exl3:
        apply_splits()
    return Lfm2MoeEngine(idx.cfg, W, experts, device="cuda", max_len=max_len)


def _sets(path: str) -> tuple[dict, list]:
    from safetensors import safe_open
    with safe_open(path, "pt") as f:
        out = {k: f.get_tensor(k) for k in f.keys()}
        meta = f.metadata() or {}
    self_rows = [out.pop(k) for k in sorted(k for k in list(out) if k.startswith("self."))]
    return {"wikitext": list(out["wikitext"]), "oasst1": list(out["oasst1"]), "self": self_rows}, \
        json.loads(meta.get("prompts", "[]"))


def cmd_selfgen(a) -> None:
    from safetensors import safe_open
    from safetensors.torch import save_file
    from transformers import AutoTokenizer

    from engine.models.lfm2_moe.engine import DecodeGraph
    tok = AutoTokenizer.from_pretrained(a.bf16)
    with safe_open(a.sets, "pt") as f:
        base = {k: f.get_tensor(k) for k in f.keys() if not k.startswith("self.")}
        meta = f.metadata() or {}
    prompts = json.loads(meta.get("prompts", "[]"))[:a.prompts]
    eng = _engine(a.bf16, 4096)
    t0 = time.time()
    for j, p in enumerate(prompts):
        ids = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True,
                                      tokenize=True, return_dict=False)
        st = eng.new_state(4096)
        t = int(eng.prefill(torch.tensor(ids), st)[0].argmax())
        dg = DecodeGraph(eng, st)
        gen = [t]
        for _ in range(a.tokens - 1):
            t = dg.greedy(t)
            gen.append(t)
            if t == tok.eos_token_id:
                break
        base[f"self.{j:03d}"] = torch.tensor(list(ids) + gen)
        print(f"[selfgen] {j + 1}/{len(prompts)} {len(ids)}+{len(gen)} tokens, {time.time() - t0:.0f} s", flush=True)
        del dg
    save_file(base, a.sets, metadata=meta)


@torch.no_grad()
def _logprobs(eng, ids: torch.Tensor) -> torch.Tensor:
    st = eng.new_state(ids.numel() + 64)
    lg = eng.prefill(ids, st, head="all")
    return torch.log_softmax(lg.float(), -1)


@torch.no_grad()
def cmd_eval(a) -> None:
    sets, _ = _sets(a.sets)
    ref, pack = _engine(a.bf16, 4096), _engine(a.pack, 4096)
    out = {"pack": a.pack, "ref": a.bf16, "sets": {}}
    t0 = time.time()
    for name, rows in sets.items():
        rows = rows[:a.rows] if a.rows else rows
        kls, agree, n = [], 0, 0
        for ids in rows:
            lr = _logprobs(ref, ids)
            lp = _logprobs(pack, ids)
            kl = (lr.exp() * (lr - lp)).sum(-1)              # per position, over the whole vocabulary
            kls.append(float(kl.mean()))
            agree += int((lr.argmax(-1) == lp.argmax(-1)).sum())
            n += kl.numel()
            del lr, lp, kl
        out["sets"][name] = {"rows": len(rows), "positions": n, "kl": sum(kls) / max(1, len(kls)),
                             "top1_pct": round(100 * agree / max(1, n), 2), "row_kl": kls}
        print(json.dumps({"set": name, "rows": len(rows), "positions": n, "kl": round(out["sets"][name]["kl"], 5),
                          "top1_pct": out["sets"][name]["top1_pct"], "s": round(time.time() - t0)}), flush=True)
    with open(a.json, "w") as fh:
        json.dump(out, fh, indent=1)


def cmd_compare(a) -> None:
    x, y = (json.load(open(p)) for p in (a.a, a.b))
    g = torch.Generator().manual_seed(0)
    for name in x["sets"]:
        d = torch.tensor(y["sets"][name]["row_kl"]) - torch.tensor(x["sets"][name]["row_kl"])
        boot = torch.stack([d[torch.randint(0, len(d), (len(d),), generator=g)].mean() for _ in range(2000)])
        lo, hi = torch.quantile(boot, torch.tensor([0.025, 0.975])).tolist()
        print(json.dumps({"set": name, "kl_a": round(x["sets"][name]["kl"], 5), "kl_b": round(y["sets"][name]["kl"], 5),
                          "b_minus_a": round(float(d.mean()), 5), "ci95": [round(lo, 5), round(hi, 5)],
                          "b_lower_rows": int((d < 0).sum()), "rows": len(d)}))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sets")
    s.add_argument("--bf16", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--rows", type=int, default=100)
    g = sub.add_parser("selfgen")
    g.add_argument("--bf16", required=True)
    g.add_argument("--sets", required=True)
    g.add_argument("--prompts", type=int, default=48)
    g.add_argument("--tokens", type=int, default=384)
    e = sub.add_parser("eval")
    e.add_argument("--bf16", required=True)
    e.add_argument("--pack", required=True)
    e.add_argument("--sets", required=True)
    e.add_argument("--json", required=True)
    e.add_argument("--rows", type=int, default=0, help="at most this many rows a set (0: all)")
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    a = ap.parse_args()
    {"sets": cmd_sets, "selfgen": cmd_selfgen, "eval": cmd_eval, "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    main()
