"""Second Look headroom (RND-33): does the block drafter use its own earlier slots when they are
real tokens instead of MASK?

The served DFlash2 block is `[anchor, MASK x (bs-1)]`: slot j is predicted with every earlier slot
masked, so the drafter never sees its own guesses except through the selector's bigram scores. A
two-pass draft would run the block again as `[anchor, t1..tu, MASK x (bs-1-u)]` with pass 1's first u
tokens unmasked. This tool measures what that second pass can give, on recorded continuations, with
no engine change and nothing trained.

At every position of each trace (the target's greedy text, e.g. `tools/regen_traces.py`'s) the true
tokens are pushed through the target 16 at a time exactly as `tools/record_lattice.py` does, and
before the drafter is synced past a position the block is run:
  pass 1   [anchor, MASK ...]                          (as served)
  pass 2u  [anchor, t1 .. tu, MASK ...] for u in --unmask, t = the TRUE next tokens, embedded with
           the target's `embed_tokens`, positions and the draft context KV unchanged.
Per slot it reports the head's top-1 agreement with the true token, the selector walk's agreement
and whether the true token is among the slot's 16 candidates, for pass 1 and every pass 2u.

The chain number is exact, not a bound: when pass 1's walk got slots 1..u right, pass 1's guesses ARE
the true tokens, so the teacher-forced pass 2u is the real second pass; when it got one of them wrong
the chain has already ended at or before u and pass 2 does not matter. So "pass 1 walk on slots 1..u,
pass 2u walk on slots u+1.." is the committed length a two-pass chain draft would get.

Usage: second_look_probe.py --traces DIR --ckpt train/ft-b8-v2 [--classes prose,chat] [--limit 1000]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.config import load_config  # noqa: E402
from engine.drafters.dflash2 import DFlash2Drafter  # noqa: E402
from engine.loader import Weights  # noqa: E402
from engine.model import Qwen38Engine  # noqa: E402


def block_pass(dr: DFlash2Drafter, ids: list[int], pos0: int):
    """One drafter block over `ids` (len = block size) at absolute positions pos0.., against the
    drafter's current context KV, as `_one_block_steps` builds it. Returns (head top-1 [bs-1],
    walk tokens [bs-1], candidates [bs-1, k]) as numpy arrays."""
    m = dr._build()
    cfg = dr.cfg
    bs = cfg.block_size
    dev = dr.eng.device
    t = torch.tensor(ids, dtype=torch.long, device=dev)
    noise = F.embedding(t, dr.eng.w.norm("embed_tokens.weight")).to(m.dtype)
    positions = torch.arange(pos0, pos0 + bs, device=dev)
    lo = 0
    if cfg.sliding_window is not None:
        lo = max(0, pos0 - cfg.sliding_window + 1)
    ctx_pos = torch.arange(lo, dr.ctx_len, device=dev)
    ctx_kv = [(dr._ck[i, :, lo:dr.ctx_len], dr._cv[i, :, lo:dr.ctx_len])
              for i in range(cfg.num_hidden_layers)]
    hidden = m.forward_block(noise, positions, ctx_kv, ctx_pos)
    pred = hidden[1:]
    walk = dr._tokens_from(m, pred, int(ids[0]), first=pos0 + 1)
    cand, scores = dr._lattice
    cand = cand.cpu().numpy()
    return cand[:, 0], np.array(walk[:bs - 1]), cand


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--traces", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--classes", default="prose,chat")
    ap.add_argument("--limit", type=int, default=1000, help="positions per class")
    ap.add_argument("--unmask", default="1,3,5")
    ap.add_argument("--block", type=int, default=16, help="true tokens pushed per target pass")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    us = [int(x) for x in a.unmask.split(",")]
    classes = a.classes.split(",")

    files = sorted(glob.glob(os.path.join(a.traces, "*.json")))
    by: dict[str, list[dict]] = defaultdict(list)
    for f in files:
        tr = json.load(open(f))
        k = tr.get("klass", tr["name"].split("-")[0])
        if k in classes:
            by[k].append(tr)

    cfg = load_config(a.model)
    w = Weights(cfg.path, device=a.device, skip_mtp=True)
    eng = Qwen38Engine(cfg, w, max_len=a.max_len, device=a.device)
    dr = DFlash2Drafter(eng, a.ckpt, blocks=1, max_len=a.max_len)
    bs = dr.cfg.block_size
    S = bs - 1
    mask = dr.cfg.mask_token_id
    arms = ["p1"] + [f"p2u{u}" for u in us]
    result: dict = {"ckpt": a.ckpt, "block_size": bs, "unmask": us, "classes": {}}

    for klass in classes:
        n = 0
        # per arm: [S] sums of top-1 hits, walk hits, in-candidates, and the slot's sample count
        top1 = {r: np.zeros(S) for r in arms}
        wlk = {r: np.zeros(S) for r in arms}
        inc = {r: np.zeros(S) for r in arms}
        cnt = np.zeros(S)
        chain = {r: 0 for r in arms}               # committed tokens (accepted + 1) per position
        blocks = {r: [0, 0] for r in arms}          # block replay: [blocks, committed]
        t0 = time.perf_counter()
        for tr in by.get(klass, []):
            if n >= a.limit:
                break
            prompt, out = tr["prompt_ids"], tr["output_ids"]
            accs: dict[str, dict[int, int]] = {r: {} for r in arms}
            eng.reset()
            dr.reset()
            with torch.no_grad():
                eng.forward(torch.tensor(prompt, device=a.device), start=0, last_only=True)
                dr.sync(prompt, eng.hidden_post_norm[0], 0)
                pos = len(prompt)
                i = 0
                while i < len(out) and n < a.limit:
                    blk = out[i:i + a.block]
                    eng.forward_block(torch.tensor(blk, device=a.device), start=pos)
                    hid = eng.hidden_post_norm[0]
                    for j, t in enumerate(blk):
                        a_idx = i + j                                  # anchor = out[a_idx]
                        truth = out[a_idx + 1:a_idx + 1 + S]
                        if len(truth) == S and n < a.limit and dr.ctx_len == pos + j:
                            p1 = block_pass(dr, [t] + [mask] * S, pos + j)
                            runs = {"p1": p1}
                            for u in us:
                                ids = [t] + list(truth[:u]) + [mask] * (S - u)
                                runs[f"p2u{u}"] = block_pass(dr, ids, pos + j)
                            tt = np.array(truth)
                            for r, (h1, wk, cd) in runs.items():
                                top1[r] += (h1 == tt)
                                wlk[r] += (wk == tt)
                                inc[r] += (cd == tt[:, None]).any(-1)
                            cnt += 1
                            # chain: pass 1 walk on slots 1..u, pass 2u walk after
                            w1 = p1[1]
                            for r in arms:
                                if r == "p1":
                                    seq = w1
                                else:
                                    u = int(r[3:])
                                    seq = np.concatenate([w1[:u], runs[r][1][u:]])
                                acc = 0
                                while acc < S and seq[acc] == tt[acc]:
                                    acc += 1
                                chain[r] += acc + 1
                                accs[r][a_idx] = acc
                            n += 1
                        dr.sync([t], hid[j:j + 1], pos + j, rows=[j])
                    pos += len(blk)
                    i += len(blk)
            # the engine's walk over the trace: a block at the anchor, then jump past what it committed
            for r in arms:
                p = 0
                while p in accs[r]:
                    blocks[r][0] += 1
                    blocks[r][1] += accs[r][p] + 1
                    p += accs[r][p] + 1
        dt = time.perf_counter() - t0
        c = np.maximum(cnt, 1)
        res = {"positions": n, "seconds": round(dt, 1),
               "top1": {r: (top1[r] / c).round(4).tolist() for r in arms},
               "walk": {r: (wlk[r] / c).round(4).tolist() for r in arms},
               "in_cand": {r: (inc[r] / c).round(4).tolist() for r in arms},
               "chain_committed": {r: round(chain[r] / max(n, 1), 4) for r in arms},
               "chain_block_replay": {r: round(blocks[r][1] / max(blocks[r][0], 1), 4) for r in arms}}
        result["classes"][klass] = res
        print(f"\n== {klass}: {n} positions, block {bs}, {dt:.0f} s")
        print("slot      " + " ".join(f"{s:6d}" for s in range(1, S + 1)))
        for what in ("top1", "walk", "in_cand"):
            for r in arms:
                print(f"{what:7s} {r:5s}" + " ".join(f"{v:6.3f}" for v in res[what][r]))
        lo, hi = 3, min(7, S)                      # slots 4..7 (0-based 3..6)
        for r in arms[1:]:
            d = (np.array(res["top1"][r][lo:hi]) - np.array(res["top1"]["p1"][lo:hi])).mean() * 100
            print(f"slots 4..{hi} top-1 gain {r} vs p1: {d:+.1f} points")
        print("chain block replay, committed tokens/block: " +
              "  ".join(f"{r} {v:.3f}" for r, v in res["chain_block_replay"].items()))
        print("chain committed tokens/position: " +
              "  ".join(f"{r} {v:.3f}" for r, v in res["chain_committed"].items()))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
