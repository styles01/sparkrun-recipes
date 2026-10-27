"""Lattice oracle (CPU only): what the DFlash2 lattice can give a greedy verify at best.

Per class: the chain (the greedy walk), `engine.tree.lattice_tree` at node budgets (nodes mode,
temperature 1.0), and the full-lattice coverage ceiling (consecutive slots whose true token is among
that slot's k candidates = a tree holding every lattice node). Committed tokens a block include the
target's own token. Also the chain's first-miss slot distribution and the share of misses whose true
token is in that slot's candidates. Replays `tools/tree_sweep.py`'s loop over `record_lattice.py`
recordings. Usage: lattice_oracle.py <dir with *.json + *.lattice.npz> [...]
"""
import sys, os, glob, json, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.tree_sweep import load, greedy_walk, log_softmax, accepted
from engine.tree import lattice_tree, DraftTree

def coverage_run(cand, target, a):
    L = cand.shape[0]; n = 0
    while n < L and a + 1 + n < len(target) and int(target[a + 1 + n]) in set(int(x) for x in cand[n]):
        n += 1
    return n

def chain_run(cand, scores, target, a):
    path = greedy_walk(scores); n = 0
    for l, c in enumerate(path):
        if a + 1 + n >= len(target) or int(cand[l][c]) != int(target[a + 1 + n]): break
        n += 1
    return n

def replay(tr, kind, budget=0, temp=1.0):
    target = tr["target"]; a = 0; blocks = committed = 0
    first_miss = collections.Counter(); rank_at_miss = collections.Counter()
    while a < len(target) - 1:
        i = tr["index"].get(a)
        if i is None: a += 1; continue
        cand, scores = tr["cand"][i], tr["scores"][i]
        if kind == "cover": acc = coverage_run(cand, target, a)
        elif kind == "chain": acc = chain_run(cand, scores, target, a)
        else:
            greedy = greedy_walk(scores); logp = log_softmax(scores, temp).tolist()
            t = lattice_tree(int(target[a]), cand.tolist(), logp, greedy, budget - 1)
            acc = accepted(t, target, a)
        if kind == "chain":
            L = cand.shape[0]
            if acc < L and a + 1 + acc < len(target):
                first_miss[acc] += 1
                row = cand[acc].tolist(); tt = int(target[a + 1 + acc])
                rank_at_miss[row.index(tt) if tt in row else -1] += 1
        blocks += 1; committed += min(acc + 1, len(target) - 1 - a); a += acc + 1
    return blocks, committed, first_miss, rank_at_miss

def main(tdir, budgets=(16, 32, 64, 128, 256)):
    traces = load(tdir)
    by = collections.defaultdict(list)
    for tr in traces: by[tr["klass"]].append(tr)
    print(f"\n== {tdir}: {len(traces)} traces, slots={traces[0]['cand'].shape[1]}, k={traces[0]['cand'].shape[2]}")
    kinds = [("chain", 0)] + [("tree", b) for b in budgets] + [("cover", 0)]
    hdr = "class      pos  " + " ".join(f"{(k if b == 0 else 'T' + str(b)):>7s}" for k, b in kinds)
    print(hdr)
    tot = {}
    for klass, trs in sorted(by.items()):
        row = []; npos = sum(len(t["target"]) for t in trs)
        for k, b in kinds:
            B = C = 0; fm = collections.Counter(); rk = collections.Counter()
            for tr in trs:
                bb, cc, f, r = replay(tr, k, b); B += bb; C += cc; fm += f; rk += r
            row.append(C / max(B, 1))
            tot.setdefault((k, b), [0, 0]); tot[(k, b)][0] += B; tot[(k, b)][1] += C
            if k == "chain":
                n = sum(rk.values()) or 1
                inlat = {r: c / n for r, c in sorted(rk.items())}
                print(f"  {klass:9s} chain first-miss slot dist {dict(sorted(fm.items()))}; target in lattice at miss: "
                      f"{1 - rk.get(-1, 0) / n:.2f} (rank 1-3: {sum(v for r, v in inlat.items() if 0 <= r <= 2):.2f})")
        print(f"{klass:9s} {npos:5d}  " + " ".join(f"{v:7.2f}" for v in row))
    print("all             " + " ".join(f"{tot[(k, b)][1] / max(tot[(k, b)][0], 1):7.2f}" for k, b in kinds))

if __name__ == "__main__":
    for d in sys.argv[1:]: main(os.path.expanduser(d))
