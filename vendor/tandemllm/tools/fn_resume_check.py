"""A resumed prefill against a cold one at long contexts, in one process (the SPD-99 follow-up).

    PYTHONPATH=<repo>:~/fn-ref/site python tools/fn_resume_check.py --pack ~/flashnext-packs/3.05 \
        --tokens 70000,140000 --json out.json

For each length n (prose held-out text through the chat template, tools/fn_generate.py's prompts):
  cold     the whole prompt prefilled from an empty state;
  cont     the prompt to the last 4,096 grid point g, then the rest, in the same state;
  resumed  the prompt to g, the state's sections and the first g rows of its views copied out, the
           state reset, both written back, `resume_at(g)` (the caches' restore), then the rest;
  dirty    the buffers first filled by another prompt (the code prompt, a little longer), then only the
           state's own reset (`reset_inplace`, the sections and the length; K, V, the indexer keys and
           the pool keep the other prompt's rows, as a live engine's do after any request), then the
           prompt cold: a kernel that reads a row past the length shows here;
then `--new` greedy tokens from each through the decode step. Recorded: the last logits' largest
difference to the cold ones and whether the argmax agrees, and the greedy tokens' first difference.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import cache  # noqa: E402
from engine.models.flashnext.serving import build  # noqa: E402
from tools.fn_generate import class_prompt  # noqa: E402


@torch.no_grad()
def greedy(eng, logits, start: int, new: int) -> list[int]:
    out = [int(logits.view(-1).argmax())]
    pos = start
    while len(out) < new:
        lg = eng.forward(torch.tensor([out[-1]], device="cuda"), pos)
        pos += 1
        out.append(int(lg.view(-1).argmax()))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--tokens", default="70000,140000")
    ap.add_argument("--new", type=int, default=32)
    ap.add_argument("--json")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.pack)
    lens = [int(x) for x in a.tokens.split(",")]
    eng, _, _ = build(a.pack, max(lens) + a.new + 64, tok, spec=False, log=lambda s: print(s, flush=True))
    res = []
    for n in lens:
        ids = torch.tensor(class_prompt(tok, "prose", n), device="cuda")
        n = ids.numel()
        g = (n - 1) // 4096 * 4096
        row = {"n": n, "grid": g}
        t0 = time.time()
        eng.reset()
        cold = eng.forward(ids, 0).clone()
        row["cold_s"] = round(time.time() - t0, 1)
        row["cold_tokens"] = greedy(eng, cold, n, a.new)
        eng.reset()
        eng.forward(ids[:g], 0)
        cont = eng.forward(ids[g:], g).clone()
        row["cont_tokens"] = greedy(eng, cont, n, a.new)
        eng.reset()
        eng.forward(ids[:g], 0)
        eng.settle()
        secs = cache.state_clone(eng)
        views = [t.narrow(ax, 0, g).clone() for _, t, ax in eng.cache_state().views()]
        eng.reset()
        cache.state_load(eng, secs)
        for (_, t, ax), x in zip(eng.cache_state().views(), views):
            t.narrow(ax, 0, g).copy_(x)
        eng.resume_at(g)
        res_lg = eng.forward(ids[g:], g).clone()
        row["resumed_tokens"] = greedy(eng, res_lg, n, a.new)
        other = torch.tensor(class_prompt(tok, "code", n + 512), device="cuda")
        eng.reset()
        eng.forward(other, 0)
        eng.settle()
        eng._chain, eng.picks = None, None
        eng.st.reset_inplace()
        dirty = eng.forward(ids, 0).clone()
        row["dirty_tokens"] = greedy(eng, dirty, n, a.new)
        for name, lg in (("cont", cont), ("resumed", res_lg), ("dirty", dirty)):
            row[f"{name}_max_diff"] = float((lg - cold).abs().max())
            row[f"{name}_argmax_same"] = int(lg.argmax()) == int(cold.argmax())
            toks = row[f"{name}_tokens"]
            row[f"{name}_first_token_diff"] = next((i for i, (x, y) in enumerate(zip(toks, row["cold_tokens"])) if x != y), None)
        print(f"[resume] n {n} grid {g}: cont diff {row['cont_max_diff']:.4g} first token diff {row['cont_first_token_diff']}; "
              f"resumed diff {row['resumed_max_diff']:.4g} first token diff {row['resumed_first_token_diff']}; "
              f"dirty diff {row['dirty_max_diff']:.4g} first token diff {row['dirty_first_token_diff']}; "
              f"cold prefill {row['cold_s']} s", flush=True)
        res.append(row)
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
