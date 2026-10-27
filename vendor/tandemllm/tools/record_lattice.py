"""Record the block drafter's whole lattice at every position of a trace.

`tools/record_traces.py` logs what the prediction head proposed at the block boundaries the
recording run happened to visit, and the simulator has to say so: a policy that accepts a different
number of tokens walks off that grid and the row it prints is an estimate.

This one has no grid to walk off. The block drafter conditions on the anchor token and on the draft
KV built from the target's hidden states at the committed positions -- and in a lossless engine the
committed prefix is the greedy prefix whatever the drafting policy was. So the lattice at position p
is the same lattice for every policy, and recording it at **every** p makes the simulator exact for
this drafter, not approximate.

Getting it cheaply is the trick. Advancing the target one token at a time would cost 108 ms a
position. The continuation is already known, so instead the true tokens are pushed through
`forward_block` `--block` at a time -- one 175 ms pass for sixteen positions -- and the drafter is
then walked across those sixteen positions one row of the tap at a time, which is what the `rows`
argument to `sync` is for. What is left is the drafter's own cost, about 35 ms a position, and that
is irreducible: it is the lattice.

Output is one `.npz` beside each `.json` trace: the candidate ids `[N, slots, k]` and the selector's
scores `[N, slots, k, k]` in fp16, about 2 MB per 512-token trace. Token ids and scores only, no
text; `results/` is gitignored and never syncs back.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.config import load_config  # noqa: E402
from engine.drafters.dflash2 import DFlash2Drafter  # noqa: E402
from engine.loader import Weights  # noqa: E402
from engine.model import Qwen38Engine  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--traces", default=None)
    ap.add_argument("--block", type=int, default=16, help="true tokens pushed per target pass")
    ap.add_argument("--limit", type=int, default=0, help="stop each trace after N positions")
    ap.add_argument("--only", default=None)
    ap.add_argument("--ckpt", default=None, help="the drafter checkpoint (default: the released "
                    "weights); re-records with the served ft-b8-v2 and ft-b16")
    ap.add_argument("--draft-block", type=int, default=0,
                    help="the drafter's block length (0: the checkpoint's own)")
    a = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tdir = a.traces or os.path.join(root, "results", "traces")
    files = sorted(glob.glob(os.path.join(tdir, "*.json")))
    if a.only:
        want = set(a.only.split(","))
        files = [f for f in files if os.path.basename(f)[:-5] in want]
    if not files:
        raise SystemExit(f"no traces in {tdir}")

    cfg = load_config(a.model)
    w = Weights(cfg.path, device=a.device, skip_mtp=True)
    eng = Qwen38Engine(cfg, w, max_len=a.max_len, device=a.device)
    dr = DFlash2Drafter(eng, a.ckpt, blocks=1, max_len=a.max_len, block=a.draft_block or None)

    for path in files:
        with open(path) as f:
            tr = json.load(f)
        prompt = tr["prompt_ids"]
        out = tr["output_ids"]
        if a.limit:
            out = out[:a.limit]
        eng.reset()
        dr.reset()
        t0 = time.perf_counter()
        with torch.no_grad():
            eng.forward(torch.tensor(prompt, device=a.device), start=0, last_only=True)
            dr.sync(prompt, eng.hidden_post_norm[0], 0)
        pos = len(prompt)
        ctx = list(prompt)
        cands: list[np.ndarray] = []
        scores: list[np.ndarray] = []
        at: list[int] = []
        i = 0
        with torch.no_grad():
            while i < len(out):
                blk = out[i:i + a.block]
                # one target pass for the whole block: the continuation is known, so every token in
                # it is accepted by construction and the pass is the 175 ms a sixteen-row verify
                # costs rather than sixteen 108 ms steps
                eng.forward_block(torch.tensor(blk, device=a.device), start=pos)
                hid = eng.hidden_post_norm[0]
                for j, t in enumerate(blk):
                    # the drafter is current through position pos+j-1, so `t` at pos+j is its anchor
                    d = dr.propose_tree(ctx + out[i:i + j + 1], budget=0)
                    if d is not None and dr._lattice is not None:
                        cand_t, sc_t = dr._lattice
                        cands.append(cand_t.to(torch.int32).cpu().numpy())
                        scores.append(sc_t.to(torch.float16).cpu().numpy())
                        at.append(i + j)
                    dr.sync([t], hid[j:j + 1], pos + j, rows=[j])
                pos += len(blk)
                ctx = ctx + list(blk)
                i += len(blk)
        dt = time.perf_counter() - t0
        if not cands:
            print(f"{tr['name']:8s} the drafter proposed nothing; no lattice written")
            continue
        npz = path[:-5] + ".lattice.npz"
        np.savez_compressed(npz, cand=np.stack(cands), scores=np.stack(scores),
                            at=np.array(at, dtype=np.int32),
                            prompt_len=np.array([len(prompt)], dtype=np.int32))
        mb = os.path.getsize(npz) / 2**20
        print(f"{tr['name']:8s} {len(at):4d} positions of {len(out):4d}  "
              f"{dt:5.1f} s  {dt / max(len(at), 1) * 1e3:5.1f} ms/position  {mb:5.1f} MB  -> {npz}")


if __name__ == "__main__":
    main()
