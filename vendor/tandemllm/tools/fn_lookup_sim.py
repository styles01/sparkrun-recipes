"""ENG-229 on the CPU: the lookup drafter merged with the MTP draft in one StairCut cut, simulated over
traces of real greedy texts (tools/fn_draft_trace.py), against the MTP draft alone.

    PYTHONPATH=<repo> python tools/fn_lookup_sim.py --trace trace.json --prices bench.json \
        --corpus ~/tandem-live/corpus --json out.json

Speculation is lossless, so every policy commits the trace's greedy text; a policy only decides each
round's block, and the simulation prices it. A round at position i (the last committed token):

  mtp      the MTP chain drafted from i (the trace's top-1 tokens and probabilities), its spine cut where
           (1 + sum of q) / (verify price of the rows + draft + commit) is largest -- StairCut's rule,
           q the product of calibrated probabilities down the spine (calibration: the trace's own
           acceptance by probability tenth, as StairCut learns online);
  merged   the same spine plus the lookup's best branch (engine/drafters/ngram.py NgramDrafter: the
           prompt and the text so far, and with `--corpus` the static corpus; vote shares smoothed),
           joined where they share a prefix; both cut together by the same rule.

Accepted tokens = the longest path of the block that matches the text; the round commits them and one
more. Prices: StairCut's measured verify ms by row count (`--prices`, a fn_spec_bench JSON) extended by
the last step per row; draft and commit ms from the same JSON.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.drafters.ngram import NgramDrafter  # noqa: E402


def prices(path: str, cls: str) -> dict:
    r = json.load(open(path))
    c = r["classes"][cls]
    sp = next(iter(c["spec"].values()))
    V = {int(k): v for k, v in sp["verify_ms_by_rows"].items()}
    V.setdefault(1, c["plain"]["ms_p50"])          # one row: the decode step
    return {"V": V, "draft": sp["draft_ms"], "commit": sp["commit_ms"]}


def vprice(P: dict, rows: int) -> float:
    """The measured price of `rows`, linear between measured counts and past the last one."""
    V = P["V"]
    ks = sorted(V)
    if rows in V:
        return V[rows]
    if rows > ks[-1]:
        step = (V[ks[-1]] - V[ks[0]]) / max(ks[-1] - ks[0], 1)
        return V[ks[-1]] + step * (rows - ks[-1])
    lo = max(k for k in ks if k < rows)
    hi = min(k for k in ks if k > rows)
    return V[lo] + (V[hi] - V[lo]) * (rows - lo) / (hi - lo)


def calibration(tr: dict) -> list:
    """Acceptance of the draft's top-1 token at each depth, given the depths above it were right, by
    probability tenth (the trace's own; StairCut learns the same online)."""
    hit, n = [0.0] * 10, [0.0] * 10
    out = tr["out"]
    for i, d in enumerate(tr["drafts"]):
        for j, (tk, pr) in enumerate(zip(d["tok"], d["prob"])):
            if i + 1 + j >= len(out):
                break
            b = min(int(pr[0] * 10), 9)
            ok = tk[0] == out[i + 1 + j]
            hit[b] += ok
            n[b] += 1
            if not ok:
                break
    return [(h + 1) / (c + 2) for h, c in zip(hit, n)]


def lookup_path(look: NgramDrafter, context: list, anchor: int) -> tuple[list, list]:
    """The lookup tree's best branch from the anchor: (tokens, path scores)."""
    _, cands = look.candidates(context, 16)
    if not cands:
        return [], []
    tree = look.build_tree(anchor, cands, 16)
    node, toks, sc = 0, [], []
    while True:
        kids = tree.children_of(node)
        if not kids:
            return toks, sc
        node = max(kids, key=lambda c: tree.scores[c])
        toks.append(tree.tokens[node])
        sc.append(tree.scores[node])


def lookup_calibration(tr: dict, look: NgramDrafter) -> list:
    """The chance a lookup path is right down to a node, by the node's path score tenth, over every
    position of the trace (StairCut learns the same online)."""
    out, prompt = tr["out"], tr["prompt"]
    hit, n = [0.0] * 10, [0.0] * 10
    look.reset()
    look.prime(prompt + out[:1])
    for i in range(len(tr["drafts"])):
        toks, sc = lookup_path(look, prompt + out[:i + 1], out[i])
        m = match(toks, out, i)
        for j, v in enumerate(sc):
            b = min(int(v * 10), 9)
            hit[b] += j < m
            n[b] += 1
        look.observe(out[i + 1:i + 2])
    return [(h + 1) / (c + 2) for h, c in zip(hit, n)]


def match(path: list, out: list, i: int) -> int:
    a = 0
    while a < len(path) and i + 1 + a < len(out) and path[a] == out[i + 1 + a]:
        a += 1
    return a


def simulate(tr: dict, P: dict, cal: list, look: NgramDrafter | None, lookup_ms: float = 0.2,
             lcal: list | None = None, depth: int = 4, per_depth_ms: float = 0.95) -> dict:
    out, prompt = tr["out"], tr["prompt"]
    i, rounds, ms, fired, from_lookup = 0, 0, 0.0, 0, 0
    if look is not None:
        look.reset()
        look.prime(prompt + out[:1])
    while i < len(tr["drafts"]):
        d = tr["drafts"][i]
        spine = [t[0] for t in d["tok"]][:depth]
        q, acc = [], 1.0
        for pr in d["prob"][:depth]:
            acc *= cal[min(int(pr[0] * 10), 9)]
            q.append(acc)
        best = (-1.0, 0, [])
        lpath, lq = [], []
        if look is not None:
            lpath, sc = lookup_path(look, prompt + out[:i + 1], out[i])
            lq = [lcal[min(int(v * 10), 9)] if lcal else v for v in sc]
        cost0 = P["draft"] + per_depth_ms * (depth - 4) + P["commit"] + (lookup_ms if look is not None else 0.0)
        for k in range(len(spine) + 1):
            for L in range(len(lpath) + 1):
                shared = 0
                while shared < min(k, L) and spine[shared] == lpath[shared]:
                    shared += 1
                rows = 1 + k + L - shared
                e = 1.0 + sum(q[:k]) + sum(lq[shared:L]) + sum(max(0.0, lq[j] - q[j]) for j in range(shared))
                v = e / (vprice(P, rows) + cost0)
                if v > best[0]:
                    best = (v, rows, (k, L, shared))
        _, rows, (k, L, shared) = best
        a_s = match(spine[:k], out, i)
        a_l = match(lpath[:L], out, i)
        a = max(a_s, a_l)
        fired += L > shared
        from_lookup += a_l > a_s
        ms += vprice(P, rows) + cost0
        rounds += 1
        if look is not None:
            look.observe(out[i + 1:i + 2 + a])
        i += a + 1
    tok = min(i, len(tr["drafts"]))
    return {"rounds": rounds, "tokens_a_round": round(tok / rounds, 3), "est_tok_s": round(1e3 * tok / ms, 2),
            "lookup_branch_rounds": fired, "rounds_won_by_lookup": from_lookup}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--prices", required=True, help="a fn_spec_bench JSON with verify_ms_by_rows")
    ap.add_argument("--prices-class", default="code")
    ap.add_argument("--corpus", default="")
    ap.add_argument("--json")
    a = ap.parse_args()
    T = json.load(open(a.trace))["traces"]
    P = prices(a.prices, a.prices_class)
    res = {"prices": P, "rows": {}}
    for name, tr in T.items():
        cal = calibration(tr)
        D = len(tr["drafts"][0]["tok"])
        row = {f"mtp_d{d}": simulate(tr, P, cal, None, depth=d) for d in range(4, D + 1)}
        for tag, path, mo in (("local", "", 3), ("corpus", a.corpus, 3), ("local_order6", "", 6)):
            if tag == "corpus" and not a.corpus:
                continue
            look = NgramDrafter(corpus_path=path, min_order=mo)
            lcal = lookup_calibration(tr, look)
            row[f"merged_{tag}"] = simulate(tr, P, cal, look, lcal=lcal)
            row[f"merged_{tag}_raw_scores"] = simulate(tr, P, cal, look)
        res["rows"][name] = row
        print(f"[sim] {name}: " + "; ".join(f"{k} {v['est_tok_s']} ({v['tokens_a_round']}/r, lookup won {v['rounds_won_by_lookup']})"
                                            for k, v in row.items()), flush=True)
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
