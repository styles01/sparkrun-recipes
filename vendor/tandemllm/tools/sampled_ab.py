"""The sampled five-workload bench, read back -- configurations against a base, round by round.

`tools/accept_hist.py --serve --factors --temperature 0.7` writes one JSON per server (per workload: tokens a round,
ms a round, tok/s, each request's own tokens a round). A hold runs every configuration once a round and alternates
them, so round r of one configuration and round r of another are a same-hour pair. This prints, per workload and
statistic, each configuration against the base in every round, and calls a difference resolved only when it has the
same sign in EVERY round and the pooled difference is larger than twice its standard error over the requests (the
text is sampled: two requests of one workload are different texts, so the requests are the samples). ALL's error is
the stratified one -- each workload's difference weighted by its share of the requests -- because pooling the requests
of five workloads would count the spread BETWEEN workloads (2 tokens a round on prose, 12 on edit) as noise.

    python tools/sampled_ab.py results/p5b/h2 --base chain --configs det,mixed --rounds 3
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

WORKLOADS = ("prose", "chat", "code", "edit", "quote", "ALL")
STATS = ("tok_blk", "ms_blk", "tok_s")


def _mean_se(xs: list[float]) -> tuple[float, float]:
    n = len(xs)
    m = sum(xs) / n
    if n < 2:
        return m, float("inf")
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, math.sqrt(var / n)


def compare(base: list[dict], other: list[dict], workload: str) -> dict:
    """`base` / `other`: one factors dict per round (accept_hist --factors). Per statistic: the rounds' values,
    the per-round deltas (%), and for tokens a round the per-request mean difference against its standard error."""
    out = {}
    for stat in STATS:
        b = [r[workload][stat] for r in base if workload in r]
        o = [r[workload][stat] for r in other if workload in r]
        deltas = [100.0 * (y - x) / x for x, y in zip(b, o)]
        same_sign = bool(deltas) and (all(d > 0 for d in deltas) or all(d < 0 for d in deltas))
        out[stat] = {"base": b, "other": o, "delta_pct": deltas, "same_sign": same_sign}
    pb = [x for r in base if workload in r for x in r[workload]["per_request_tok_blk"]]
    po = [x for r in other if workload in r for x in r[workload]["per_request_tok_blk"]]
    mb, sb = _mean_se(pb)
    mo, so = _mean_se(po)
    se = math.sqrt(sb * sb + so * so)
    out["tok_blk"]["requests"] = (len(pb), len(po))
    out["tok_blk"]["req_mean"] = (mb, mo)
    out["tok_blk"]["req_se"] = (sb, so)
    out["tok_blk"]["z"] = (mo - mb) / se if se > 0 else 0.0
    out["tok_blk"]["resolved"] = out["tok_blk"]["same_sign"] and abs(out["tok_blk"]["z"]) > 2.0
    return out


def stratify(total: dict, parts: list[dict]) -> dict:
    """ALL's tokens a round judged on the per-workload comparisons: the difference of the means weighted by each
    workload's share of the base's requests, against the error of that weighted sum. `total` is ALL's `compare`,
    `parts` the workloads'; the rounds' same-sign rule stays ALL's own."""
    n = [c["tok_blk"]["requests"][0] for c in parts]
    tot = sum(n)
    d = sum(w / tot * (c["tok_blk"]["req_mean"][1] - c["tok_blk"]["req_mean"][0]) for w, c in zip(n, parts))
    se = math.sqrt(sum((w / tot) ** 2 * (c["tok_blk"]["req_se"][0] ** 2 + c["tok_blk"]["req_se"][1] ** 2)
                       for w, c in zip(n, parts)))
    t = dict(total["tok_blk"])
    t["z"] = d / se if se > 0 else 0.0
    t["stratified"] = True
    t["resolved"] = t["same_sign"] and abs(t["z"]) > 2.0
    return {**total, "tok_blk": t}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir")
    ap.add_argument("--base", default="chain")
    ap.add_argument("--configs", default="det,mixed")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    d = Path(a.dir)
    load = lambda c: [json.loads((d / f"{c}-r{r}.json").read_text())  # noqa: E731
                      for r in range(1, a.rounds + 1) if (d / f"{c}-r{r}.json").exists()]
    base = load(a.base)
    res = {}
    for cfg in a.configs.split(","):
        other = load(cfg)
        res[cfg] = {}
        print(f"== {cfg} against {a.base} ({len(other)} / {len(base)} rounds)")
        for w in WORKLOADS:
            c = compare(base, other, w)
            if w == "ALL":
                c = stratify(c, [res[cfg][x] for x in WORKLOADS if x != "ALL"])
            res[cfg][w] = c
            t = c["tok_blk"]
            verdict = ("RESOLVED " + ("better" if t["z"] > 0 else "worse")) if t["resolved"] else "n.r."
            print(f"  {w:6s} tok/blk " + " ".join(f"{x:5.2f}->{y:5.2f}" for x, y in zip(t["base"], t["other"]))
                  + f"  ({' '.join(f'{p:+.1f}' for p in t['delta_pct'])} %, z {t['z']:+.1f}"
                  + (" stratified" if t.get("stratified") else "") + f") {verdict}"
                  + "   ms/blk " + " ".join(f"{p:+.1f}" for p in c["ms_blk"]["delta_pct"])
                  + "   tok/s " + " ".join(f"{p:+.1f}" for p in c["tok_s"]["delta_pct"]) + " %")
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
