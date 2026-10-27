"""Does E2's question in place leave the real engine exactly where it was? (The claim check, 2026-10-02.)

At SP2 the claim check forwards Qwen's questions after the KV's rows and puts the engine back
(`server/app.py` `_ask_in_place`: settle, copy the recurrent state, forward, copy it back, reset the
KV's length). tests/test_claim_check.py proves it on the CPU's reference path; this runs it on the
served kernels -- the fused GDN commit, a commit pending into the next verify, the verify graphs,
the FP8 KV when the environment sets them -- where state can sit outside `state.S`.

For each moment SP2 can meet (a chain verify with a partial accept and its rollback, a full accept
whose walked state is pending, a tree verify with its path commit, a single-token decode step),
the scenario runs twice from the same prefill: once straight on (an eight-row chain verify, its
rollback and a decode step after the moment), once with two questions (120 and 320 random ids)
asked in place at that moment first. Every logit after the moment and the state after each step
must be bit-identical, and the KV rows up to the length.

    python tools/claim_identity.py                 # flags as the environment sets them (serve.env)
    python tools/claim_identity.py --len 8192      # the moment at a longer context

It needs the GPU and the weights: run it inside a hold, never beside the service. Exit 0 = PASS.
The token ids are random, fixed by a seed: it measures the arithmetic, not the text.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TREE_PARENTS = [-1, 0, 1, 2, 1, 4, 0, 6, 6, 8, 9, 10]


def _S(eng) -> torch.Tensor:
    f = getattr(eng, "committed_state", None)
    return (f() if f is not None else eng.state.S).float().cpu()


def _snap(eng) -> dict:
    n = eng.kv.length
    return {"S": _S(eng), "conv": eng.state.conv.float().cpu(),
            "k": eng.kv.k[..., :n, :].float().cpu(), "v": eng.kv.v[..., :n, :].float().cpu()}


def run(eng, moment: str, ask, length: int, seed: int = 11) -> dict:
    """The scenario up to `moment`, `ask(eng, pos)` there (or nothing), then the steps after."""
    from engine.tree import DraftTree
    g = torch.Generator().manual_seed(seed)
    hi = min(100000, eng.cfg.vocab_size)
    r = lambda n: torch.randint(min(1000, hi // 2), hi, (n,),            # noqa: E731
                                generator=g).to(eng.device)
    eng.reset()
    out = {}
    with torch.no_grad():
        eng.forward(r(length), start=0, last_only=True)
        pos = length
        if moment == "chain_partial":
            eng.forward_block(r(16), start=pos)
            eng.rollback_to(6)
            pos += 6
        elif moment == "chain_full":
            eng.forward_block(r(8), start=pos)
            pos += 8                               # a full accept: the walk stays pending
        elif moment == "tree":
            tree = DraftTree(tokens=r(len(TREE_PARENTS)).tolist(), parents=TREE_PARENTS)
            eng.forward_tree(torch.tensor(tree.tokens).to(eng.device), tree.parents, start=pos)
            path = tree.path(len(TREE_PARENTS) - 1)
            eng.commit_tree(path)
            pos += len(path)
        else:                                      # single: one decode step
            eng.forward(r(1), start=pos, last_only=True)
            pos += 1
        q1, q2 = r(120), r(320)
        if ask is not None:
            out["asked"] = ask(eng, pos, q1, q2)
        out["at_moment"] = _snap(eng)
        out["chain8"] = eng.forward_block(r(8), start=pos).float().cpu()
        eng.rollback_to(5)
        pos += 5
        out["after_chain8"] = _snap(eng)
        out["decode"] = eng.forward(r(1), start=pos, last_only=True).float().cpu()
        out["after_decode"] = _snap(eng)
    return out


MOMENTS = ("chain_partial", "chain_full", "tree", "single")


def asker(letters: list[int]):
    """The two questions as the claim check asks them, through the served `_ask_in_place`."""
    from server.app import _ask_in_place

    def ask(eng, pos, q1, q2):
        t0 = time.perf_counter()
        x = _ask_in_place(eng, pos, eng.device, q1.tolist(), letters[:2])
        y = _ask_in_place(eng, pos, eng.device, q2.tolist(), letters)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return {"ms": round((time.perf_counter() - t0) * 1e3, 1), "q1": x, "q2": y}
    return ask


def check(eng, length: int, moment: str, ask) -> tuple[list[str], list[str], dict]:
    """(what differs with the questions, what differs between two plain runs, the asked run)."""
    run(eng, moment, None, length)                                # warm every kernel once
    ref = run(eng, moment, None, length)
    got = run(eng, moment, ask, length)
    again = run(eng, moment, None, length)
    return same(ref, got), same(ref, again), got


def same(a: dict, b: dict, prefix: str = "") -> list[str]:
    bad = []
    for k, x in b.items():
        if k == "asked":
            continue
        y = a[k]
        if isinstance(x, dict):
            bad += same(y, x, prefix + k + ".")
        elif not torch.equal(x, y):
            bad.append(f"{prefix}{k}: max |d| {(x - y).abs().max().item():.3g}")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--len", type=int, default=1024, help="the context at the moment")
    ap.add_argument("--nvfp4", default=os.environ.get("QWEN38_NVFP4"))
    ap.add_argument("--fp8-head", default=os.environ.get("QWEN38_FP8_HEAD"))
    a = ap.parse_args()
    from engine.config import load_config
    from engine.loader import Weights
    from engine.model import Qwen38Engine
    cfg = load_config(None)
    w = Weights(cfg.path, skip_mtp=True, nvfp4=a.nvfp4, fp8_head=a.fp8_head)
    eng = Qwen38Engine(cfg, w, max_len=a.len + 1024)
    ask = asker([32, 33, 34, 35, 36])
    fails = 0
    for moment in MOMENTS:
        bad, det, got = check(eng, a.len, moment, ask)
        fails += bool(bad) or bool(det)
        print(f"{moment:14s} {'PASS' if not bad and not det else 'FAIL'}  questions "
              f"{got['asked']['ms']} ms at {a.len} tokens"
              + (f"; differs: {', '.join(bad[:4])}" if bad else "")
              + (f"; NOT DETERMINISTIC without the question: {', '.join(det[:4])}" if det else ""))
    print(f"CLAIM-CHECK IDENTITY {'PASS' if fails == 0 else 'FAIL'} ({fails} failures)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
