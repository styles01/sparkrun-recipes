"""RND-35 replay (CPU): slot-0 lattice scores blended with the MTP layer's log-probabilities
(`tools/mtp_slot0.py`), scores[:, 0] += beta * logp_mtp, then lattice_oracle's replay per class at
the served budgets. beta = 0 is the recorded lattice (must reproduce lattice_oracle's table).

Usage: mtp_blend.py DIR [DIR ...] --betas 0,0.25,0.5,1,2
"""
import argparse, collections, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.tree_sweep import load  # noqa: E402
from tools.lattice_oracle import replay  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--betas", default="0,0.25,0.5,1,2")
    ap.add_argument("--budgets", default="16,32")
    a = ap.parse_args()
    budgets = [int(b) for b in a.budgets.split(",")]
    for d in a.dirs:
        traces = load(d)
        base = {t["name"]: t["scores"].copy() for t in traces}
        mt = {}
        for t in traces:
            p = os.path.join(d, t["name"] + ".mtp.npz")
            mt[t["name"]] = np.load(p)["logp"] if os.path.exists(p) else None
        print(f"\n== {d}")
        for beta in (float(b) for b in a.betas.split(",")):
            for t in traces:
                s = base[t["name"]].copy()
                if mt[t["name"]] is not None and beta:
                    s[:, 0, :, :] += beta * mt[t["name"]][:, None, :]
                t["scores"] = s
            by = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
            for t in traces:
                for k, b in [("chain", 0)] + [("tree", b) for b in budgets]:
                    B, C, _, _ = replay(t, k, b)
                    by[t["klass"]][(k, b)][0] += B
                    by[t["klass"]][(k, b)][1] += C
                    by["all"][(k, b)][0] += B
                    by["all"][(k, b)][1] += C
            cols = [("chain", 0)] + [("tree", b) for b in budgets]
            line = " | ".join(f"{kl} " + "/".join(f"{v[1] / max(v[0], 1):.2f}" for v in (by[kl][c] for c in cols))
                              for kl in ("prose", "chat", "code", "row", "all") if kl in by)
            print(f"beta {beta:5.2f}: (chain/T{'/T'.join(map(str, budgets))}) {line}", flush=True)


if __name__ == "__main__":
    main()
