"""Flash-Next's state on the KV pool against its static buffers, in one process (ENG-241 step 1).

    PYTHONPATH=<repo>:~/fn-ref/site python tools/fn_pool_check.py --pack ~/flashnext-packs/3.05 \
        --tokens 6000,40000 --json out.json

One engine (the weights once), two states of the same window: `static` built without the pool,
`paged` with it (K, V and the indexer keys reserved as addresses, committed as rows are written). For
each length n (prose held-out text through the chat template): the prompt prefilled in each, then
`--new` greedy tokens through the decode step, then the same continuation again as verify blocks of 5
rows committed whole (the verify and commit graphs). Recorded: whether the last logits, the greedy tokens
and the verify blocks' logits are equal bit for bit, and the paged state's committed bytes against
the rows it holds (the pool commits what the rows need, not the window).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import kvpool  # noqa: E402
from engine.models.flashnext.serving import FlashNextServing, build  # noqa: E402
from tools.fn_generate import class_prompt  # noqa: E402

GiB = 1 << 30


@torch.no_grad()
def run(s: FlashNextServing, ids: torch.Tensor, new: int, block: int = 5) -> dict:
    n = ids.numel()
    s.reset()
    lg = s.forward(ids, 0)
    first = lg.clone()
    out = [int(lg.view(-1).argmax())]
    for pos in range(n, n + new - 1):
        lg = s.forward(torch.tensor([out[-1]], device="cuda"), pos)
        out.append(int(lg.view(-1).argmax()))
    # the same tokens again as chain blocks (row 0 the token at the length), each committed whole
    s.reset()
    s.forward(ids, 0)
    blocks, at = [], n
    for i in range(0, new - block + 1, block):
        blocks.append(s.forward_tree(torch.tensor(out[i:i + block], device="cuda"), list(range(-1, block - 1)),
                                     at).clone())
        s.commit_tree(list(range(block)))
        at += block
    return {"first": first, "tokens": out, "blocks": blocks}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--tokens", default="6000,40000")
    ap.add_argument("--new", type=int, default=64)
    ap.add_argument("--json")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.pack)
    lens = [int(x) for x in a.tokens.split(",")]
    L = max(lens) + a.new + 1024
    eng, _, _ = build(a.pack, L, tok, spec=False, log=lambda s: print(s, flush=True))
    kvpool.POOL = kvpool.KVPool(hard_floor=10 * GiB, margin_rows=256)
    paged = FlashNextServing(eng.fn, L)
    assert paged.kv.pages is not None and paged.kv.pages.on_device(), "the paged state is not on the pool"
    print(f"[pool] paged state: {len(paged.kv.pages.bufs)} buffers, {paged.kv.pages.committed_bytes() / GiB:.3f} GiB "
          f"committed at load", flush=True)
    res = []
    for n in lens:
        ids = torch.tensor(class_prompt(tok, "prose", n), device="cuda")
        a_ = run(eng, ids, a.new)
        b_ = run(paged, ids, a.new)
        rows = paged.st.length
        r = {"n": int(ids.numel()), "first_equal": bool(torch.equal(a_["first"], b_["first"])),
             "tokens_equal": a_["tokens"] == b_["tokens"],
             "blocks_equal": len(a_["blocks"]) == len(b_["blocks"])
             and all(torch.equal(x, y) for x, y in zip(a_["blocks"], b_["blocks"])),
             "rows": rows, "committed_gib": round(paged.kv.pages.committed_bytes() / GiB, 3),
             "committed_rows": paged.kv.pages.rows,
             "static_window_gib": round(sum(b.nbytes for b in paged.kv.pages.bufs) / GiB, 3)}
        print(f"[pool] {r}", flush=True)
        res.append(r)
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
