"""The first measurement, offline: would a gathered draft head leave the drafter's lattice as it is?

The drafter's head reads all 248,320 vocabulary rows for its slots every round. A head over a
gathered row set -- the V most frequent tokens of a corpus plus every id of the request (prompt and
output so far) -- gives exactly the full head's candidates and logits for a slot row whose
full-vocabulary top-16 lies inside the set: the same ids, the same fp32 values, the same lattice,
the same tree. Only rows where the top-16 leaves the set can change the draft, and only the draft.

On recorded lattices (tools/record_lattice.py: `cand` is each slot row's full top-16), per workload
and set size, this counts:

    rows     slot rows whose whole top-16 is inside the set           (the ticket's go line: 99 %)
    blocks   positions whose every slot row is                         (lattice identical)
    true     positions where the target's next token, if the full top-16 held it, is still in the set

    python tools/draft_vocab_cover.py results/lat/lat-b16 results/lat/lat-b8 \\
        --rank results/p5/corpus-rank.npy --sizes 16384,32768,65536
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.tree_sweep import load  # noqa: E402

VOCAB = 248320


def cover(traces: list[dict], hot: np.ndarray) -> dict[str, list[int]]:
    """Per class: [rows, rows covered, positions, positions covered, true held, true kept]."""
    out: dict[str, list[int]] = {}
    for tr in traces:
        inset = np.zeros(VOCAB, dtype=bool)
        inset[hot] = True
        inset[np.asarray(tr["prompt"], dtype=np.int64)] = True
        target = tr["target"]
        acc = out.setdefault(tr["klass"], [0] * 6)
        # positions in order, so the output so far can join the set as it is committed
        known = 0
        for a in sorted(tr["index"]):
            while known <= a and known < len(target):
                inset[target[known]] = True
                known += 1
            cand = tr["cand"][tr["index"][a]]                       # [L, k]
            ok = inset[cand].all(axis=1)
            acc[0] += len(ok)
            acc[1] += int(ok.sum())
            acc[2] += 1
            acc[3] += int(ok.all())
            if a + 1 < len(target):
                nxt = target[a + 1]
                if (cand[0] == nxt).any():
                    acc[4] += 1
                    acc[5] += int(inset[nxt])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--rank", required=True, help="token ids by corpus frequency, most frequent first")
    ap.add_argument("--sizes", default="16384,32768,65536")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    rank = np.load(a.rank).astype(np.int64)
    report = {}
    for tdir in a.traces:
        traces = load(tdir)
        for tr in traces:
            tr["prompt"] = json.load(open(os.path.join(tdir, tr["name"] + ".json")))["prompt_ids"]
        L = traces[0]["cand"].shape[1]
        print(f"\n{tdir}: {len(traces)} traces, {L} slots, "
              f"{sum(len(t['index']) for t in traces)} positions")
        for v in (int(x) for x in a.sizes.split(",")):
            per = cover(traces, rank[:v])
            tot = [sum(p[i] for p in per.values()) for i in range(6)]
            per["ALL"] = tot
            print(f"  hot {v:>6} + request ids:  " + "  ".join(
                f"{k} rows {100 * p[1] / p[0]:.2f} % blocks {100 * p[3] / p[2]:.1f} % "
                f"true {100 * p[5] / max(p[4], 1):.2f} %" for k, p in sorted(per.items())))
            report.setdefault(tdir, {})[v] = {k: {"rows": p[1] / p[0], "blocks": p[3] / p[2],
                                                  "true": p[5] / max(p[4], 1)}
                                              for k, p in per.items()}
    if a.json:
        json.dump(report, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
