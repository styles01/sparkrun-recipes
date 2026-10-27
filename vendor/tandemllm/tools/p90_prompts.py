"""Which prompts sit at the row's top ranks, per row3 report: where a p90 difference comes from.

A row's p90 is the 45th of its 50 per-request decode rates, so a p90 that "resolves worse" can be one
prompt whose answer changed -- an arithmetic change flips a near-tie and the text decodes slower --
rather than a slower engine. This prints, for each report, every run's rates at
ranks 41-50, and per prompt that ever reaches the top twelve, its median rate over the report's runs.
The rate is row3's: (completion_tokens - 1) / (e2e - ttft).

    python tools/p90_prompts.py results/row3/p1final-nostore.json results/row3/spd31r-nostore.json
"""

from __future__ import annotations

import json
import statistics
import sys


def per_prompt(report: str) -> list[dict[str, float]]:
    """One dict a run: prompt id -> decode rate, warm-ups and empty answers left out."""
    runs = []
    for r in json.load(open(report))["runs"]:
        reqs = json.load(open(r["record"]))["raw"]["payload"]["requests"]
        runs.append({q["prompt_id"]: (q["completion_tokens"] - 1)
                     / ((q["e2e_ms"] - q["ttft_ms"]) / 1000.0)
                     for q in reqs if not q.get("warmup") and q.get("completion_tokens")})
    return runs


def table(data: dict[str, list[dict[str, float]]], top: int = 12) -> list[str]:
    lines = []
    for lab, runs in data.items():
        for i, rs in enumerate(runs):
            v = sorted(rs.values())
            lines.append(f"{lab:22s} run {i + 1}: ranks 41-50: "
                         + " ".join(f"{x:5.1f}" for x in v[40:50]))
    seen: set[str] = set()
    for runs in data.values():
        for rs in runs:
            seen |= {k for k, _ in sorted(rs.items(), key=lambda kv: -kv[1])[:top]}
    first = list(data.values())[0]
    lines.append("prompt            " + " ".join(f"{lab[:14]:>14s}" for lab in data))
    for p in sorted(seen, key=lambda p: -statistics.median(r.get(p, 0.0) for r in first)):
        lines.append(f"{p:16s}  " + " ".join(
            f"{statistics.median(r.get(p, 0.0) for r in runs):14.1f}" for runs in data.values()))
    return lines


if __name__ == "__main__":
    reports = sys.argv[1:]
    if not reports:
        raise SystemExit(__doc__)
    print("\n".join(table({r.split("/")[-1].removesuffix(".json"): per_prompt(r) for r in reports})))
