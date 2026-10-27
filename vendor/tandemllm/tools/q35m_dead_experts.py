"""ENG-391: the dead routed experts of a Qwen3.5 MoE EXL3 pack (all-zero in the checkpoint, stored with zero output
scales) and whether the router picks them: every expert whose three projections have svh all zero, then the
routing of the bundle's rows and of a few chat prompts through the prefill kernels (route trace), the picks that
land on a dead expert counted, and every logit checked finite.

    python tools/q35m_dead_experts.py --pack DIR [--bundle DIR] [--out f.json]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--bundle", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    from engine.models.qwen3_5_moe.engine import Qwen35MoeEngine
    from engine.models.qwen3_5_moe.load import load
    idx, W, banks, _ = load(a.pack, "cuda", mtp=False, log=None)
    eng = Qwen35MoeEngine(idx.cfg, W, banks, device="cuda", max_len=4096)
    c = idx.cfg
    dead = {}
    for li in range(c.num_hidden_layers):
        b = banks[li]
        ds = [e for e in range(c.num_experts)
              if all(float(b.block(e, j).svh.abs().max()) == 0.0 for j in range(3))]
        if ds:
            dead[li] = ds
    print("[dead] dead experts by layer:", dead, flush=True)
    rows = []
    if a.bundle:
        from safetensors import safe_open
        f = safe_open(os.path.join(a.bundle, "input_ids_4.0.safetensors"), "pt")
        rows += [f.get_tensor(k).view(-1) for k in sorted(f.keys())]
    picks, hits, total, finite = Counter(), Counter(), 0, True
    for r in rows:
        eng.route_trace = []
        st = eng.new_state(2048)
        lg = eng.forward_pre(r.cuda(), st, head="all")
        finite = finite and bool(torch.isfinite(lg).all())
        for li, ids in eng.route_trace:
            ids = ids.reshape(-1).tolist()
            total += len(ids)
            for e in ids:
                if e in dead.get(li, ()):
                    hits[(li, e)] += 1
        eng.route_trace = None
        del st
    out = {"dead": {str(k): v for k, v in dead.items()}, "n_dead": sum(len(v) for v in dead.values()),
           "routed_picks": total, "picks_on_dead": sum(hits.values()),
           "by_expert": {f"{li}.{e}": n for (li, e), n in sorted(hits.items())}, "logits_finite": finite}
    print("[dead]", json.dumps(out), flush=True)
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
