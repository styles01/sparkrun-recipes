"""The cost, on the device the engine uses: what a constrained block pays, and what a new state pays.

Two numbers, measured apart because they are paid apart:

  * **per block, warm** -- a verify block's rows masked from cached state masks: the DFA walks of
    the draft tokens, the stack of the rows' masks and one masked fill, on `[rows, vocab]` bf16
    logits, synchronised, median of `--iters`. A chain of 16 rows and a 16-node tree. An
    unconstrained request calls none of this (0 ms, nothing on its path).
  * **per new state, cold** -- the first time a state is reached its mask is computed over the
    whole vocabulary on the CPU and copied to the device; every later visit is the cached tensor.

The states are the ones a real answer visits: the reference answers of the eval-json-v1 items,
tokenized and walked through their schema's DFA; and json_object over the same text.

    python tools/grammar_cost.py --items ~/inference-atlas/datasets/eval-json-v1/items.jsonl \\
        --json results/api/grammar-cost.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=None)
    ap.add_argument("--items", required=True)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--rows", type=int, default=16)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    from engine.config import load_config
    from engine import grammar as G
    from engine.tree import DraftTree
    from tools.bench_structured import pick, schema_of
    cfg = load_config(a.model)
    tok = AutoTokenizer.from_pretrained(cfg.path)
    t0 = time.perf_counter()
    vocab = G.Vocab.from_tokenizer(tok, cfg.vocab_size)
    vocab_s = time.perf_counter() - t0
    eos = {tok.eos_token_id}
    items = pick([json.loads(x) for x in open(Path(a.items).expanduser()) if x.strip()], a.n)
    cold, states_seen, blocks = [], 0, []
    dev = torch.device(a.device)
    lg = torch.randn(a.rows, cfg.vocab_size, dtype=torch.bfloat16, device=dev)
    sync = torch.cuda.synchronize if dev.type == "cuda" else (lambda: None)
    for it in items:
        schema, _ = schema_of(it)
        for label, pattern in (("schema", G.WS + G.schema_regex(schema) + G.WS),
                               ("json_object", G.json_object_regex(3))):
            gram = G.grammar_for(pattern, vocab)
            ids = tok(json.dumps(it["answer"], ensure_ascii=False), add_special_tokens=False).input_ids
            cons = G.Constraint(gram, eos, dev)
            cons.seed([])
            state = cons.state
            for t in ids + [None]:
                if not gram.cached(state, dev):
                    t1 = time.perf_counter()
                    gram.mask(state, frozenset(eos), dev)
                    sync()
                    cold.append((time.perf_counter() - t1) * 1e3)
                    states_seen += 1
                if t is None:
                    break
                state = gram.dfa.walk(state, vocab.bytes[t])
            # warm blocks: the answer's first rows as a chain and as a tree
            cons.seed([])
            draft = ids[:a.rows - 1]
            for _ in range(3):                                  # warm-up and cache fill
                x = lg[:len(draft) + 1].clone()
                cons.apply_chain(x, draft)
            ts = []
            for _ in range(a.iters):
                x = lg[:len(draft) + 1].clone()
                sync()
                t1 = time.perf_counter()
                cons.apply_chain(x, draft)
                sync()
                ts.append((time.perf_counter() - t1) * 1e3)
            tree = DraftTree.from_sequences(0, [(draft[:8], 0.9), (draft[:3] + draft[4:11], 0.5)])
            for _ in range(3):
                cons.apply_tree(lg[:len(tree)].clone(), tree)
            tt = []
            for _ in range(a.iters):
                x = lg[:len(tree)].clone()
                sync()
                t1 = time.perf_counter()
                cons.apply_tree(x, tree)
                sync()
                tt.append((time.perf_counter() - t1) * 1e3)
            blocks.append({"id": it["id"], "grammar": label, "states": gram.dfa.states, "chain_rows": len(draft) + 1,
                           "chain_ms": statistics.median(ts), "tree_nodes": len(tree),
                           "tree_ms": statistics.median(tt)})
    chain = [b["chain_ms"] for b in blocks]
    tree_ms = [b["tree_ms"] for b in blocks]
    out = {"device": str(dev), "vocab_tokens": len(vocab.ids), "vocab_build_s": round(vocab_s, 2),
           "cold_state_ms": {"n": len(cold), "median": statistics.median(cold),
                             "p90": sorted(cold)[int(0.9 * len(cold))], "max": max(cold)},
           "states_visited": states_seen,
           "warm_chain_ms": {"median": statistics.median(chain), "max": max(chain),
                             "rows": a.rows},
           "warm_tree_ms": {"median": statistics.median(tree_ms), "max": max(tree_ms)},
           "blocks": blocks}
    print(json.dumps({k: v for k, v in out.items() if k != "blocks"}, indent=1))
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
