"""ENG-153 groundwork: how far the MTP layer's greedy draft chain agrees with the target, per depth.

    PYTHONPATH=<repo>:~/fn-ref/site python tools/fn_mtp_accept.py --pack ~/fn2/packs/3.05 \
        --classes prose,code,chat --ctx 1024 --new 192 --depth 6 --json out.json

The target decodes greedy through its decode graph (plain, no speculation). The MTP layer is primed
over the prompt (its own KV cache, as the trunk's streams and the next tokens); then at every
committed token t it is fed the real pair (the trunk's streams at t, the token t + 1) and drafts a
greedy chain of `--depth` tokens for t + 2 .. t + 1 + depth from its own streams (the deeper feeds
rolled back). Drafts are scored against the text the target actually produced: per depth k, the
share of rounds whose first k drafts are all right, and the expected tokens a round of a chain of
depth k (1 + the accepted drafts). Nothing here measures speed; the draft runs eager. The full head
is used for drafting (no hot-vocabulary slice).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.models.flashnext.engine import DecodeGraph, FlashNextEngine  # noqa: E402
from engine.models.flashnext.load import load  # noqa: E402
from engine.models.flashnext.ref import FNState  # noqa: E402
from tools.fn_generate import class_prompt  # noqa: E402


@torch.no_grad()
def run_class(eng: FlashNextEngine, ids: list[int], new: int, depth: int, eos: set) -> dict:
    st = eng.new_state(eng.max_len)
    logits = eng.prefill(torch.tensor(ids), st, keep_stack=True)
    S = eng.prompt_stack                               # the prompt's streams [L, hc H]
    L = len(ids)
    st_m = FNState(cap=eng.max_len)
    x = int(logits[0].argmax())                        # x_L
    if L > 1:
        eng.mtp_feed(S[: L - 1], torch.tensor(ids[1:L]), st_m)          # prime over the prompt
    g = DecodeGraph(eng, st)
    g.capture()
    stack_t = S[L - 1: L]
    toks = [x]
    drafts = []
    while len(toks) < new and toks[-1] not in eos:
        nxt = toks[-1]
        drafts.append(eng.mtp_chain(stack_t, nxt, st_m, depth))   # for tokens after `nxt`
        toks.append(g.greedy(nxt))
        stack_t = eng.stack.clone()
    # drafts[i] predicts toks[i + 1], toks[i + 2], ...
    hits = [0] * depth
    rounds = 0
    tokens_round = [0.0] * depth
    for i, d in enumerate(drafts):
        truth = toks[i + 1: i + 1 + depth]
        if len(truth) < depth:
            break
        rounds += 1
        acc = 0
        for k in range(depth):
            if d[k] == truth[k]:
                acc += 1
            else:
                break
        for k in range(depth):
            if acc > k:
                hits[k] += 1
            tokens_round[k] += 1 + min(acc, k + 1)
    return {"prompt_tokens": L, "new_tokens": len(toks), "rounds": rounds,
            "accept_at_depth": [round(h / max(rounds, 1), 4) for h in hits],
            "tokens_a_round_chain": [round(t / max(rounds, 1), 3) for t in tokens_round]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--classes", default="prose,code,chat")
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--new", type=int, default=192)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--json")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    idx, W, banks, tables, timing = load(a.pack, "cuda", mtp=True)
    eng = FlashNextEngine(idx.cfg, W, tables, banks, device="cuda", max_len=4096)
    tok = AutoTokenizer.from_pretrained(a.pack)
    eos = set(idx.cfg.eos_token_ids)
    res = {"pack": a.pack, "load": timing, "knobs": {"ctx": a.ctx, "new": a.new, "depth": a.depth, "greedy": True,
                                                     "thinking": False, "draft_head": "full (248,320)"},
           "classes": {}}
    for c in a.classes.split(","):
        r = run_class(eng, class_prompt(tok, c, a.ctx), a.new, a.depth, eos)
        res["classes"][c] = r
        print(f"[mtp] {c}: {json.dumps(r)}", flush=True)
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
