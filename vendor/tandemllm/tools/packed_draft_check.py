"""Parallel requests on the board: the packed draft against each slot's own draft graph.

Two or four engines over ONE set of served weights, each with its own block drafter over the same
checkpoint (the wide arm by default), each prefilled with its own prompt. For each slot the lone
draft call (`graph_run`, the served graph) and the packed call (`packed_draft`) are compared: the
candidates and the lattice bit for bit, and the greedy walk's tokens. Then the times: N lone graph
calls against one packed call.

    # inside ops/hold.sh, ops/serve.env exported
    python tools/packed_draft_check.py --ckpt train/ft-b16 --block 16 --n 2,4 --out results/parallel/packed-draft.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> None:
    import torch
    from engine import cache
    from engine.config import load_config
    from engine.drafters.dflash2 import DFlash2Drafter
    from engine.drafters.draft_graph import packed_draft
    from engine.loader import Weights
    from engine.model import Qwen38Engine

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="train/ft-b16")
    ap.add_argument("--block", type=int, default=16)
    ap.add_argument("--n", default="2,4")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--out", default="results/parallel/packed-draft.json")
    a = ap.parse_args()
    cfg = load_config(None)
    w = Weights(cfg.path, skip_mtp=True, nvfp4=os.environ.get("QWEN38_NVFP4"),
                fp8_head=os.environ.get("QWEN38_FP8_HEAD"))
    nmax = max(int(x) for x in a.n.split(","))
    engs, ds, anchors, pos = [], [], [], []
    g = torch.Generator().manual_seed(11)
    with torch.no_grad():
        for j in range(nmax):
            e = Qwen38Engine(cfg, w, max_len=16384)
            d = DFlash2Drafter(e, a.ckpt, blocks=1, path="greedy", max_len=16384, block=a.block)
            d._build()
            L = 900 + 413 * j
            ids = torch.randint(0, cfg.vocab_size, (L,), generator=g).tolist()
            logits, _, _ = cache.prefill(e, d, ids, "cuda", chunk=1024)
            engs.append(e)
            ds.append(d)
            anchors.append(int(logits[0, -1].argmax()))
            pos.append(L)
        out = {"ckpt": a.ckpt, "block": a.block, "runs": {}}
        for n in (int(x) for x in a.n.split(",")):
            lone = [tuple(t.clone() for t in ds[j].graph_run(anchors[j], pos[j])) for j in range(n)]
            reqs = [(ds[j], anchors[j], pos[j]) for j in range(n)]
            packed = [tuple(t.clone() for t in o) for o in packed_draft(reqs, graph=True)]
            eager = packed_draft(reqs, graph=False)
            res = {"slots": [], "packed_graph_eq_eager": all(
                torch.equal(a_[0], b_[0]) and torch.equal(a_[1], b_[1]) for a_, b_ in zip(packed, eager))}
            for j in range(n):
                cand_eq = torch.equal(lone[j][0], packed[j][0])
                score_eq = torch.equal(lone[j][1], packed[j][1])
                wl = ds[j].module.walk_host(lone[j][0], lone[j][1])[0]
                wp = ds[j].module.walk_host(packed[j][0], packed[j][1])[0]
                res["slots"].append({"cand_bit_identical": cand_eq, "scores_bit_identical": score_eq,
                                     "walk_identical": wl == wp,
                                     "scores_max_abs_diff": float((lone[j][1].float()
                                                                   - packed[j][1].float()).abs().max())})
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(a.reps):
                for j in range(n):
                    ds[j].graph_run(anchors[j], pos[j])
            torch.cuda.synchronize()
            res["lone_graphs_ms"] = round((time.perf_counter() - t0) * 1e3 / a.reps, 2)
            for mode in (True, False):
                t0 = time.perf_counter()
                for _ in range(a.reps):
                    packed_draft(reqs, graph=mode)
                torch.cuda.synchronize()
                res["packed_graph_ms" if mode else "packed_eager_ms"] = round(
                    (time.perf_counter() - t0) * 1e3 / a.reps, 2)
            out["runs"][n] = res
            print(f"n={n}: {json.dumps(res)}", flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
