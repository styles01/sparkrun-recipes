#!/usr/bin/env python3
"""The joint verify on the real model: does `forward_tree_multi` give each slot what its own
`forward_tree` gives it? (parallel requests, server/slots.py)

Four engines over ONE copy of the weights: two run each round alone (`forward_tree`, with its
verify graphs, as a lone request does), two run the same rounds together (`forward_tree_multi`,
eager, as a joint round does). Each pair starts from the same prompts (random ids, fixed seed),
gets the same trees each round (16 + 16 rows, then 8 + 8 + ... shapes the shares make) and commits
the same path: the target's own greedy walk of the lone run. Per round it prints whether the
logits are bit-identical, the largest difference, whether every row's argmax agrees, and at the
end the recurrent state and KV rows of each slot.

    python tools/parallel_verify_check.py --rounds 24

Run with the served environment (ops/serve.env sourced): the kernels, the fold, the graphs.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _tree(g, anchor: int, n: int, branch_at: int):
    """A tree of `n` nodes: a spine, and a second branch from node `branch_at`."""
    from engine.tree import DraftTree
    toks = torch.randint(1000, 100000, (n - 1,), generator=g).tolist()
    tokens, parents = [anchor], [-1]
    spine = (n - 1) * 2 // 3
    for i in range(spine):
        tokens.append(toks[i])
        parents.append(len(tokens) - 2)
    for j in range(n - 1 - spine):
        tokens.append(toks[spine + j])
        parents.append(min(branch_at, spine) if j == 0 else len(tokens) - 2)
    t = DraftTree(tokens, parents)
    t.check()
    return t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=24)
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--no-graphs", action="store_true",
                    help="the lone verifies eager too (QWEN38_VERIFY_GRAPH off in this process): tells "
                         "a difference the batch makes from one the verify graphs make")
    ap.add_argument("--nvfp4", default=os.environ.get("QWEN38_NVFP4"))
    ap.add_argument("--fp8-head", default=os.environ.get("QWEN38_FP8_HEAD"))
    a = ap.parse_args()
    from engine.config import load_config
    from engine.loader import Weights
    import engine.model as em
    from engine.model import Qwen38Engine, forward_tree_multi
    if a.no_graphs:
        em.VERIFY_GRAPH = False
    cfg = load_config(None)
    w = Weights(cfg.path, skip_mtp=True, nvfp4=a.nvfp4, fp8_head=a.fp8_head)
    alone = [Qwen38Engine(cfg, w, max_len=a.max_len) for _ in range(2)]
    joint = [Qwen38Engine(cfg, w, max_len=a.max_len) for _ in range(2)]
    g = torch.Generator().manual_seed(11)
    prompts = [torch.randint(1000, 100000, (n,), generator=g).cuda() for n in (300, 517)]
    pos = []
    with torch.no_grad():
        for i, p in enumerate(prompts):
            la = alone[i].forward(p, start=0, last_only=True)
            lj = joint[i].forward(p, start=0, last_only=True)
            assert torch.equal(la, lj), "the two prefills differ"
            pos.append(p.numel())
        # each round's anchor is the token the last round decided (a random id the first time)
        anchors = [int(prompts[i][-1]) for i in range(2)]
        # the shapes a joint round makes: 16 + 16 at two slots, 8 + 8 at four (shares), mixed
        shapes = [(16, 16), (16, 9), (8, 8), (12, 16), (5, 8)]
        bit, worst, argmax_ok, rounds = 0, 0.0, 0, 0
        rounds_alone = [0]
        for r in range(a.rounds):
            n0, n1 = shapes[r % len(shapes)]
            trees = [_tree(g, anchors[0], n0, 1 + r % 3), _tree(g, anchors[1], n1, 2)]
            blocks = [torch.tensor(t.tokens).cuda() for t in trees]
            lone = [alone[i].forward_tree(blocks[i], trees[i].parents, start=pos[i])
                    for i in range(2)]
            # as the server's round (server/app.py `_run_round`): a block that is not batchable (a
            # chain-shaped one, which keeps its chain kernel) verifies alone, the rest together
            ok = [joint[i].tree_batchable(trees[i].parents) for i in range(2)]
            if all(ok):
                multi = forward_tree_multi([(joint[i], blocks[i], trees[i].parents, pos[i])
                                            for i in range(2)])
            else:
                multi = [joint[i].forward_tree(blocks[i], trees[i].parents, start=pos[i])
                         for i in range(2)]
                rounds_alone[0] += 1
            same = all(torch.equal(x, y) for x, y in zip(lone, multi))
            which = [i for i in range(2) if not torch.equal(lone[i], multi[i])]
            d = max(float((x.float() - y.float()).abs().max()) for x, y in zip(lone, multi))
            am = all(torch.equal(x.argmax(-1), y.argmax(-1)) for x, y in zip(lone, multi))
            bit += same
            argmax_ok += am
            worst = max(worst, d)
            rounds += 1
            # commit the lone run's greedy walk on both
            for i in range(2):
                picks = lone[i].argmax(-1).tolist()
                path, new = alone[i].accept_tree(trees[i], picks)
                alone[i].commit_tree(path)
                joint[i].commit_tree(path)
                pos[i] += len(path)
                anchors[i] = new[-1]
            print(f"round {r:2d} rows {n0:2d}+{n1:2d}{'' if all(ok) else ' (a chain: each alone)'}: logits {'IDENTICAL' if same else 'differ (slot ' + str(which) + ')'} "
                  f"(max |d| {d:.3g}), argmax {'same' if am else 'DIFFERS'}", flush=True)
        st_same = []
        for i in range(2):
            Sa, Sj = alone[i].committed_state(), joint[i].committed_state()
            n = alone[i].kv.length
            st_same.append(torch.equal(Sa, Sj) and torch.equal(
                alone[i].kv.k[..., :n, :], joint[i].kv.k[..., :n, :]) and n == joint[i].kv.length)
        print(f"PARALLEL_VERIFY graphs={'off' if a.no_graphs else 'on'} rounds={rounds} joint={rounds - rounds_alone[0]} bit_identical={bit}/{rounds} "
              f"argmax_same={argmax_ok}/{rounds} max_abs_diff={worst:.3g} "
              f"state_and_kv_identical={st_same}", flush=True)
        sys.exit(0 if argmax_ok == rounds else 1)


if __name__ == "__main__":
    main()
