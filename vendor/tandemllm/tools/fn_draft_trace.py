"""ENG-229 groundwork: the MTP draft at every position of a greedy text, for CPU simulations of drafting
policies (tools/fn_lookup_sim.py).

    PYTHONPATH=<repo>:~/fn-ref/site python tools/fn_draft_trace.py --pack ~/fn2/packs/3.05 \
        --prompts code,code-intervals,code-lru,code-backup,prose,chat --ctx 1024 --new 256 --out trace.json

Per prompt (tools/fn_generate.py's classes, or the extra code requests below with the held-out code as
filler): plain greedy decode through the decode graph, and before each step the served drafter -- the
MTP chain of `--depth` from graphs, the hot 32k draft head -- asked from that position as if every
round committed one token. Recorded: the prompt ids, the greedy text, and per position the draft's
top-2 tokens and probabilities at each depth. Speculation is lossless, so the greedy text is what
every policy commits; a simulation only has to price the rounds.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.models.flashnext.engine import DecodeGraph, FlashNextEngine  # noqa: E402
from engine.models.flashnext.load import load  # noqa: E402
from tools.fn_specloop import MTPGraphDrafter  # noqa: E402
from tools import fn_generate as G  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# code requests beside tools/exl3_fastrun.py's (the nginx log parser), so one text does not decide
EXTRA = {
    "code-intervals": ("code", "Write a Python function merge_intervals(intervals) that merges overlapping "
                               "intervals, with type hints, a docstring and five pytest tests."),
    "code-lru": ("code", "Implement an LRUCache class in Python with O(1) get and put using a dict and a doubly "
                         "linked list, then show a short usage example."),
    "code-backup": ("code", "Write a bash script that backs up a directory into a dated tar.gz, keeps the last "
                            "seven archives, and logs what it did. Explain each part briefly."),
    # the lookup drafter's own ground (ENG-229): text that repeats the prompt
    "edit": ("code", "Take the last complete Python function in the code above, add type hints and a docstring, "
                     "rename nothing, and print the whole edited function."),
    "quote": ("prose", "Copy the last two paragraphs of the text above exactly as written, word for word."),
}


def prompt_ids(tok, name: str, ctx: int) -> list[int]:
    if name not in EXTRA:
        return G.class_prompt(tok, name, ctx)
    cls, ask = EXTRA[name]
    saved = G.ASK[cls]
    G.ASK[cls] = ask
    try:
        return G.class_prompt(tok, cls, ctx)
    finally:
        G.ASK[cls] = saved


@torch.no_grad()
def trace(eng: FlashNextEngine, ids: list[int], new: int, depth: int, eos: set) -> dict:
    st = eng.new_state(eng.max_len)
    first = int(eng.prefill(torch.tensor(ids), st, keep_stack=True)[0].argmax())
    dr = MTPGraphDrafter(eng, depth, eng.max_len)
    dr.prime(eng.prompt_stack, ids, first)
    dr.capture_all()
    dg = DecodeGraph(eng, st)
    out, drafts = [first], []
    b = eng._mtp_bufs(depth)
    t0 = time.perf_counter()
    while len(out) < new and out[-1] not in eos:
        dr.propose(out[-1])
        drafts.append({"tok": b["tok_h"][:depth].tolist(), "prob": [[round(p, 5) for p in r] for r in b["prob_h"][:depth].tolist()]})
        nxt = dg.greedy(out[-1])
        row = dg.stacks[eng.qsa_class(st.length - 1)].clone()
        dr.accepted(row, [nxt])
        out.append(nxt)
    return {"prompt": ids, "out": out, "drafts": drafts, "s": round(time.perf_counter() - t0, 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--prompts", default="code,code-intervals,code-lru,code-backup,prose,chat")
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--new", type=int, default=256)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--hot", type=int, default=32768)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    idx, W, banks, tables, _ = load(a.pack, "cuda", mtp=True)
    eng = FlashNextEngine(idx.cfg, W, tables, banks, device="cuda", max_len=a.ctx + a.new + 64)
    if a.hot:
        eng.set_draft_head(np.load(os.path.join(ROOT, "ops", "fn-hot-vocab.npy"))[:a.hot])
    tok = AutoTokenizer.from_pretrained(a.pack)
    eos = set(idx.cfg.eos_token_ids)
    res = {"knobs": {"ctx": a.ctx, "new": a.new, "depth": a.depth, "hot": a.hot, "drafter": "MTP chain, graphs",
                     "greedy": True}, "traces": {}}
    for name in a.prompts.split(","):
        res["traces"][name] = t = trace(eng, prompt_ids(tok, name, a.ctx), a.new, a.depth, eos)
        print(f"[trace] {name}: {len(t['out'])} tokens, {t['s']} s; text: {tok.decode(t['out'][:40])!r}", flush=True)
    json.dump(res, open(a.out, "w"))


if __name__ == "__main__":
    main()
