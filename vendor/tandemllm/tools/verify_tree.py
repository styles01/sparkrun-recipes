"""The tree verify, on the real model: does it compute what a chain verify computes?

Three gates, in the order of how much they can fail by.

1. **A tree that is a chain must be the chain.** Same tokens, same start, `forward_tree` against
   `forward_block`: logits, recurrent state, convolution state, KV.
2. **A branch must be worth what that branch alone is worth.** Verify a branching tree, take the
   path the target accepted, and compare it against verifying only that path's tokens as a chain
   from the same entry state. This is the losslessness claim stated as an identity rather than as a
   generation: whatever the other branches did, they may not have reached the answer.
3. **Greedy output must not depend on the drafter, including when the drafter proposes a tree.**
   The M2 gate with `AdversarialTreeDrafter` attached, which puts the accepted path somewhere other
   than the first L rows of the DFS order on nearly every block.

And one measurement, because the pricing needs it: what a verify costs as a function of tree size
and tree SHAPE. The whole thesis of this track is that it depends on the first and not the second.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.config import load_config  # noqa: E402
from engine.drafters.fixed import AdversarialDrafter, AdversarialTreeDrafter  # noqa: E402
from engine.loader import Weights  # noqa: E402
from engine.model import Qwen38Engine  # noqa: E402
from engine.spec import generate_greedy, generate_spec, generate_spec_tree  # noqa: E402
from engine.tree import DraftTree, TreeBuilder  # noqa: E402
from tools.verify_spec import PROMPTS, bf16_ulp, compare  # noqa: E402


def snapshot(eng):
    return (eng.state.S.clone(), eng.state.conv.clone(),
            eng.kv.k[..., :eng.kv.length + 64, :].clone(),
            eng.kv.v[..., :eng.kv.length + 64, :].clone(), eng.kv.length)


def restore(eng, s):
    eng.state.S.copy_(s[0]); eng.state.conv.copy_(s[1])
    eng.kv.k[..., :s[2].shape[-2], :] = s[2]
    eng.kv.v[..., :s[3].shape[-2], :] = s[3]
    eng.kv.length = s[4]


# One bf16 ulp at a logit of magnitude 16-32, which is where this model's top logits sit. The
# engine's logits ARE bf16, so two answers that differ by this are as close as it can represent
# them being.
BF16_ULP_AT_20 = 0.125


def d(a, b):
    return (a.float() - b.float()).abs().max().item()


def kl(a, b) -> float:
    pa = torch.log_softmax(a.float(), -1)
    pb = torch.log_softmax(b.float(), -1)
    return float((pa.exp() * (pa - pb)).sum(-1).mean())


def random_tree(anchor: int, rng: random.Random, n: int, vocab: int, branch: int = 3) -> DraftTree:
    b = TreeBuilder(anchor)
    count = 0
    guard = 0
    while count < n and guard < 500:
        guard += 1
        parent = rng.randrange(0, count + 1)
        if len(b.kids[parent]) >= branch:
            continue
        b.add(parent, rng.randrange(1000, min(vocab, 200000)), 1.0 / (count + 2), "rnd")
        count += 1
    return b.build()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--nodes", type=int, default=12)
    ap.add_argument("--new", type=int, default=64)
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--curve", action="store_true", help="also measure the verify cost by shape")
    ap.add_argument("--no-lossless", action="store_true")
    a = ap.parse_args()

    cfg = load_config(a.model)
    w = Weights(cfg.path, device=a.device, skip_mtp=True)
    eng = Qwen38Engine(cfg, w, max_len=a.max_len, device=a.device)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.path)
    rng = random.Random(7)
    ok = True

    print("=" * 96)
    print("gate 1 and 2: a tree verify against the chain verify it has to agree with")
    print("=" * 96)
    for name, text in PROMPTS.items():
        ids = tok(tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                          add_generation_prompt=True, enable_thinking=False),
                  return_tensors="pt").input_ids[0].to(a.device)
        eng.reset()
        with torch.no_grad():
            lg = eng.forward(ids, start=0, last_only=True)
        pos = ids.numel()
        anchor = int(lg[0, -1].argmax())
        snap = snapshot(eng)

        # --- 1. a chain tree, against the chain verify and against the engine's own noise floor
        chain = [anchor] + [rng.randrange(1000, 200000) for _ in range(7)]
        from engine import gdn
        runs = []
        for flag in (True, False, True):
            gdn.UT_INVERSE = flag
            restore(eng, snap)
            with torch.no_grad():
                lg = eng.forward_block(torch.tensor(chain, device=a.device), start=pos)
                eng.rollback_to(len(chain))
            runs.append((lg.clone(), eng.state.S.clone(), eng.state.conv.clone()))
        gdn.UT_INVERSE = True
        lg_b, S_b, conv_b = runs[0]
        det = d(runs[0][0], runs[2][0])
        f_l, f_kl, f_s = d(lg_b, runs[1][0]), kl(lg_b, runs[1][0]), d(S_b, runs[1][1])

        restore(eng, snap)
        t = DraftTree.chain(anchor, chain[1:])
        with torch.no_grad():
            lg_t = eng.forward_tree(torch.tensor(t.tokens, device=a.device), t.parents, start=pos)
            eng.commit_tree(list(range(len(chain))))
        after_t = snapshot(eng)
        dl, dk, ds = d(lg_b, lg_t), kl(lg_b, lg_t), d(S_b, after_t[0])
        dc = d(conv_b, after_t[1])
        same_argmax = int((lg_b.argmax(-1) == lg_t.argmax(-1)).sum())
        # The band is 4x the floor, not 1x: the floor is one sample of a random quantity -- the
        # same reordering measured on three prompts gives KL 7.9e-05, 1.3e-04 and 4.8e-05, a factor
        # of nearly three between prompts. What the gate is asking is whether the tree is IN FAMILY
        # with a reordering the engine already ships, not whether it beat one sample of it.
        v1 = (same_argmax == len(chain) and dl <= max(f_l, BF16_ULP_AT_20)
              and dk <= 4 * f_kl and ds <= 4 * f_s and det == 0.0)
        ok &= v1
        print(f"{name:6s} chain-as-tree B=8   argmax {same_argmax}/{len(chain)}   "
              f"max|dlogit| {dl:.4f}  KL {dk:.2e}  state {ds:.2e}  conv {dc:.2e}")
        print(f"{'':6s}   the engine's own floor:  rerun {det:.4f}   "
              f"serial UT inverse: max|dlogit| {f_l:.4f}  KL {f_kl:.2e}  state {f_s:.2e}   "
              f"{'PASS' if v1 else 'FAIL'}")

        # --- 2. a branching tree, every root-to-leaf path
        restore(eng, snap)
        tree = random_tree(anchor, rng, a.nodes, cfg.vocab_size)
        tree.check()
        with torch.no_grad():
            lg_tree = eng.forward_tree(torch.tensor(tree.tokens, device=a.device), tree.parents,
                                       start=pos)
        picks_tree = lg_tree.argmax(-1).tolist()
        worst_l = worst_s = worst_kl = 0.0
        bad = ties = 0
        for leaf in tree.leaves():
            path = tree.path(leaf)
            restore(eng, snap)
            with torch.no_grad():
                lg_tree2 = eng.forward_tree(torch.tensor(tree.tokens, device=a.device),
                                            tree.parents, start=pos)
                eng.commit_tree(path)
            st_tree = snapshot(eng)
            restore(eng, snap)
            toks = [tree.tokens[i] for i in path]
            with torch.no_grad():
                lg_chain = eng.forward_block(torch.tensor(toks, device=a.device), start=pos)
                eng.rollback_to(len(toks))
            st_chain = snapshot(eng)
            worst_l = max(worst_l, d(lg_chain, lg_tree2[path]))
            worst_s = max(worst_s, d(st_chain[0], st_tree[0]))
            worst_kl = max(worst_kl, kl(lg_chain, lg_tree2[path]))
            # An argmax that moves is only a failure where there IS an argmax to move. Where the
            # top two logits are within one bf16 ulp of each other the model cannot tell them
            # apart, and which one a reduction returns is the rounding of the matmul that produced
            # them -- the 10:02 entry, in its tree form.
            ac, at_ = lg_chain.float(), lg_tree2[path].float()
            for r in range(len(path)):
                if int(ac[r].argmax()) == int(at_[r].argmax()):
                    continue
                two = ac[r].topk(2).values
                gap = float(two[0] - two[1])
                if gap <= BF16_ULP_AT_20:
                    ties += 1
                else:
                    bad += 1
        v2 = worst_l <= max(f_l, BF16_ULP_AT_20) * 2 and worst_s <= f_s * 8 and bad == 0
        ok &= v2
        print(f"{name:6s} tree n={tree.n_draft:<2d} {len(tree.leaves()):2d} leaves  "
              f"every path against verifying that path alone:  max|dlogit| {worst_l:.4f}  "
              f"KL {worst_kl:.2e}  state {worst_s:.2e}  argmax moved {bad} times "
              f"({ties} of them inside one bf16 ulp)  {'PASS' if v2 else 'FAIL'}")
        restore(eng, snap)

    if a.curve:
        print()
        print("=" * 96)
        print("the verify cost against tree size and tree SHAPE")
        print("=" * 96)
        ids = tok("hello " * 128, return_tensors="pt").input_ids[0].to(a.device)
        eng.reset()
        with torch.no_grad():
            eng.forward(ids, start=0, last_only=True)
        pos = ids.numel()
        print(f"{'nodes':>6} {'shape':<14} {'verify ms':>10} {'commit ms':>10} {'ms/node':>9}")
        for n, shape in [(1, "chain"), (3, "chain"), (7, "chain"), (11, "chain"), (15, "chain"),
                         (23, "chain"), (31, "chain"),
                         (7, "branch b=2"), (15, "branch b=2"), (15, "branch b=3"),
                         (23, "branch b=3"), (31, "branch b=3")]:
            if shape == "chain":
                t = DraftTree.chain(1000, [rng.randrange(1000, 200000) for _ in range(n)])
            else:
                t = random_tree(1000, rng, n, cfg.vocab_size, branch=int(shape[-1]))
            toks = torch.tensor(t.tokens, device=a.device)
            path = t.path(t.leaves()[0])
            with torch.no_grad():
                for _ in range(2):
                    eng.forward_tree(toks, t.parents, start=pos)
                    eng.commit_tree(path)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(a.reps):
                    eng.forward_tree(toks, t.parents, start=pos)
                torch.cuda.synchronize()
                dt = (time.perf_counter() - t0) / a.reps
                eng.forward_tree(toks, t.parents, start=pos)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(a.reps):
                    eng.commit_tree(path)
                    eng.forward_tree(toks, t.parents, start=pos)
                torch.cuda.synchronize()
                cm = (time.perf_counter() - t0) / a.reps - dt
            print(f"{t.n_draft + 1:6d} {shape:<14} {dt * 1e3:10.2f} {cm * 1e3:10.2f} "
                  f"{dt * 1e3 / (t.n_draft + 1):9.2f}")

    if not a.no_lossless:
        print()
        print("=" * 96)
        print("gate 3: greedy output must not depend on the drafter, including a TREE drafter")
        print("=" * 96)
        for name, text in PROMPTS.items():
            ids = tok(tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                              add_generation_prompt=True, enable_thinking=False),
                      return_tensors="pt").input_ids[0].to(a.device)
            base, st = generate_greedy(eng, ids, a.new, cfg.eos_token_ids, record_gaps=True)
            gaps, tops = st.gaps, st.tops
            mn = min(gaps) if gaps else 0.0
            print(f"{name:6s} gap median {sorted(gaps)[len(gaps) // 2]:7.3f}  minimum {mn:7.4f}  "
                  f"{sum(1 for g, t in zip(gaps, tops) if g <= bf16_ulp(t))} of {len(gaps)} "
                  f"within one bf16 ulp")
            for label, mk in (
                ("adversarial tree", lambda: AdversarialTreeDrafter(
                    cfg.vocab_size, seed=1, truth=base[1:], accept_prefix=4,
                    budget=a.nodes, branch=3)),
                ("all-noise tree", lambda: AdversarialTreeDrafter(
                    cfg.vocab_size, seed=2, truth=[], accept_prefix=0, budget=a.nodes, branch=3)),
                ("adversarial chain", lambda: AdversarialDrafter(
                    cfg.vocab_size, seed=1, truth=base[1:], accept_prefix=4)),
            ):
                dr = mk()
                if hasattr(dr, "propose_tree"):
                    got, _ = generate_spec_tree(eng, ids, a.new, dr, a.nodes, cfg.eos_token_ids)
                else:
                    got, _ = generate_spec(eng, ids, a.new, dr, 8, cfg.eos_token_ids)
                good, why = compare(base, got, gaps, tok, tops)
                ok &= good
                print(f"    {label:20s} {why}")

    print()
    print("VERDICT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
