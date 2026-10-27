"""Single-stream decode throughput across a workload mix.

Five prompts, chosen so that the two regimes a lookup drafter lives between are both represented:
fresh generation, where a suffix memory knows nothing, and reproduction, where most of the output
is already somewhere in the context. Reporting a single tok/s for this model without saying which
regime it came from is how a recipe ends up quoting a number nobody can reproduce.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.config import load_config  # noqa: E402
from engine.drafters.dflash2 import DFlash2Drafter  # noqa: E402
from engine.drafters.engram import EngramDrafter  # noqa: E402
from engine.drafters.mtp import MTPDrafter  # noqa: E402
from engine.router import RouterDrafter  # noqa: E402
from engine.loader import Weights  # noqa: E402
from engine.model import Qwen38Engine  # noqa: E402
from engine.spec import Relax, generate_greedy, generate_spec  # noqa: E402

SNIPPET = '''
def load_shard(path, device):
    """Read one safetensors shard and return its tensors, quantised weights kept as codes."""
    out = {}
    with safe_open(path, framework="pt", device=device) as f:
        for key in f.keys():
            if ".visual." in key:
                continue
            out[key] = f.get_tensor(key)
    return out


def step_bytes(weights, n_layers):
    """Bytes one autoregressive step has to read, by group."""
    per_layer = {}
    for name, tensor in weights.items():
        if not name.startswith("layers."):
            continue
        idx = int(name.split(".")[1])
        per_layer[idx] = per_layer.get(idx, 0) + tensor.numel() * tensor.element_size()
    layers = sum(v for k, v in per_layer.items() if k < n_layers)
    head = weights["lm_head.weight"]
    return layers + head.numel() * head.element_size()
'''

PROMPTS = {
    "prose": ("Write three paragraphs about why the memory system, and not the arithmetic units, "
              "sets the speed of a language model that generates one token at a time."),
    "chat": ("I have a machine with 121 GB of unified memory and about 273 GB/s of bandwidth. "
             "Explain in plain language what that means for running a 27-billion-parameter model."),
    "code": ("Write a Python function that reads a safetensors file header and prints every tensor "
             "name, dtype and shape, sorted by the number of bytes it occupies."),
    "edit": ("Here is a Python module:\n\n```python" + SNIPPET + "```\n\n"
             "Rewrite it with type hints on every function and parameter. Change nothing else: "
             "keep the same function names, the same docstrings, the same logic and the same "
             "order. Output the complete module."),
    "quote": ("Read this passage and then repeat it back to me word for word, exactly as written, "
              "with no commentary:\n\n" + open(os.path.join(
                  os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                  "bench", "refprompt.txt")).read().split("\n\n")[0]),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--new", type=int, default=128)
    ap.add_argument("--ks", default="8,16")
    ap.add_argument("--baseline", action="store_true", help="also time the non-speculative path")
    ap.add_argument("--only", default=None)
    ap.add_argument("--mtp", default=None,
                    help="comma separated draft depths for the checkpoint's mtp head")
    ap.add_argument("--mtp-hidden", default="post", choices=["post", "pre"])
    ap.add_argument("--no-engram", action="store_true")
    ap.add_argument("--router", default=None,
                    help="comma separated mtp depths for the engram+mtp router")
    ap.add_argument("--dflash2", default=None,
                    help="comma separated chain lengths in blocks for the block drafter; "
                         "1 = one 8-wide block (7 proposals), 2 = two chained (14)")
    ap.add_argument("--dflash2-path", default="greedy", choices=["greedy", "viterbi", "both"],
                    help="how the block drafter walks its own lattice")
    ap.add_argument("--dflash2-head", default=None,
                    help="reduced-vocabulary head for the block drafter's own head read")
    ap.add_argument("--dflash2-head-both", action="store_true",
                    help="run every block-drafter configuration twice, with and without that head")
    ap.add_argument("--dflash2-ckpt", default=None)
    ap.add_argument("--think", default="off", choices=("on", "off", "both"),
                    help="whether the chat template opens a reasoning block. This tool has always "
                         "passed no `enable_thinking`, and the template's default is ON, so every "
                         "acceptance number in this file before 15:50 on 2026-09-17 was measured "
                         "on reasoning text. The bench row this program is measured against sends "
                         "`enable_thinking: false`, so `off` is the regime that decides anything "
                         "and it is the default here now")
    ap.add_argument("--dflash2-block", type=int, default=0,
                    help="override the drafter's block length (0 = the checkpoint's own)")
    ap.add_argument("--relax-tau", type=float, default=1.0, help="LOSSY, see engine/spec.py::Relax")
    ap.add_argument("--relax-rank", type=int, default=1, help="LOSSY, see engine/spec.py::Relax")
    ap.add_argument("--dflash2-ckpts", default=None,
                    help="several drafter checkpoints, comma separated, compared in ONE process. "
                         "`base` means the released one. A fine-tuned drafter has to be measured "
                         "against the drafter it started from on the same weights, the same "
                         "prompts and the same allocation history, or the comparison is with a "
                         "different afternoon")
    ap.add_argument("--dflash2-tap", default="entry", choices=["entry", "output", "both"],
                    help="which hidden state target_layer_ids names")
    a = ap.parse_args()

    cfg = load_config(a.model)
    w = Weights(cfg.path, device=a.device, skip_mtp=False)
    eng = Qwen38Engine(cfg, w, max_len=a.max_len, device=a.device)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.path)
    eos = cfg.eos_token_ids

    walks = ["greedy", "viterbi"] if a.dflash2_path == "both" else [a.dflash2_path]
    heads = [None, a.dflash2_head] if a.dflash2_head_both else [a.dflash2_head]
    taps = ["entry", "output"] if a.dflash2_tap == "both" else [a.dflash2_tap]
    ckpts = ([None if c in ("base", "") else os.path.expanduser(c)
              for c in a.dflash2_ckpts.split(",")] if a.dflash2_ckpts else [a.dflash2_ckpt])
    dflash2_configs = [(int(b), wk, hd, tp, ck)
                       for b in (a.dflash2.split(",") if a.dflash2 else [])
                       for wk in walks for hd in heads for tp in taps for ck in ckpts]
    dflash2_cache: dict[tuple, DFlash2Drafter] = {}

    rows = []
    thinks = [True, False] if a.think == "both" else [a.think == "on"]
    jobs = [(f"{n}{'' if len(thinks) == 1 else ('/th' if th else '/no')}", t, th)
            for n, t in PROMPTS.items() for th in thinks
            if not (a.only and n != a.only)]
    for name, text, think in jobs:
        ids = tok(tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                          add_generation_prompt=True, enable_thinking=think),
                  return_tensors="pt").input_ids[0].to(a.device)
        print(f"\n### {name}  ({ids.numel()} prompt tokens, thinking "
              f"{'on' if think else 'off'})")
        if a.baseline:
            _, st = generate_greedy(eng, ids, a.new, eos)
            print("   ", st.line("no drafter"))
            rows.append((name, "none", 0, st))
        for d in ([int(x) for x in a.mtp.split(",")] if a.mtp else []):
            md = MTPDrafter(eng, max_len=a.max_len, hidden=a.mtp_hidden, depth=d)
            _, st = generate_spec(eng, ids, a.new, md, d, eos,
                                  relax=Relax(a.relax_tau, a.relax_rank))
            print("   ", st.line(f"mtp-{a.mtp_hidden} d={d}"))
            rows.append((name, f"mtp-{a.mtp_hidden}", d, st))
        for d in ([int(x) for x in a.router.split(",")] if a.router else []):
            rd = RouterDrafter(EngramDrafter(), MTPDrafter(eng, max_len=a.max_len,
                                                          hidden=a.mtp_hidden), mtp_depth=d)
            _, st = generate_spec(eng, ids, a.new, rd, max(d, 16), eos,
                                  relax=Relax(a.relax_tau, a.relax_rank))
            print("   ", st.line(f"router d={d}"))
            print(f"     router chose engram {rd.stats['engram']}x "
                  f"({rd.stats['engram_tokens']} tok), mtp {rd.stats['mtp']}x "
                  f"({rd.stats['mtp_tokens']} tok)")
            rows.append((name, "router", d, st))
        for nb, walk, hd, tp, ck in dflash2_configs:
            dd = dflash2_cache.get((nb, walk, hd, tp, ck))
            if dd is None:
                dd = DFlash2Drafter(eng, ck, blocks=nb, path=walk, tap=tp,
                                    draft_head=hd or "", max_len=a.max_len,
                                    block=a.dflash2_block or None)
                dd._build()
                dflash2_cache[(nb, walk, hd, tp, ck)] = dd
            dd.attach()
            width = (dd.cfg.block_size - 1) * nb
            _, st = generate_spec(eng, ids, a.new, dd, width, eos,
                                  relax=Relax(a.relax_tau, a.relax_rank))
            dd.detach()
            tag = ("+head" if hd else "") + ("" if tp == "entry" else "/out") \
                + ("" if ck is None else "/" + os.path.basename(ck.rstrip("/")))
            print("   ", st.line(f"dflash2-{walk[:3]}{tag} b={nb}"))
            rows.append((name, f"df2-{walk[:3]}{tag}", width, st))

        for k in ([] if a.no_engram else [int(x) for x in a.ks.split(",")]):
            eg = EngramDrafter()
            _, st = generate_spec(eng, ids, a.new, eg, k, eos,
                                  relax=Relax(a.relax_tau, a.relax_rank))
            print("   ", st.line(f"engram k={k}"))
            print(f"     engram hit {eg.stats['hits']}/{eg.stats['calls']} calls, "
                  f"orders {dict(sorted(eg.stats['order_hist'].items(), reverse=True))}")
            rows.append((name, "engram", k, st))

    print("\n" + "=" * 92)
    print(f"{'workload':10s} {'drafter':12s} {'k':>3} {'tok/s':>8} {'acc/block':>10} "
          f"{'draft acc':>10} {'rollbacks':>10} {'tok':>5}")
    for name, d, k, st in rows:
        print(f"{name:10s} {d:12s} {k:3d} {st.tok_s:8.2f} {st.accept_len:10.2f} "
              f"{st.accept_rate * 100:9.1f}% {st.rollbacks:10d} {st.tokens:5d}")


if __name__ == "__main__":
    main()
