"""The rows of a tool-eval-bench batch (tools/teb_batch.sh) as one table, and each miss with the
grader's reason.

    python3 tools/teb_summary.py OUT [label ...] [--md] [--misses]

One line a row: score, deployability, responsiveness, minutes, the response cache's hits during
the row (a row with hits replayed answers: its timings are not the engine's), the scenarios run
and the ones not passed (`TC-23:p` partial, `TC-68:f` fail). `--md` prints the table as Markdown,
`--misses` adds the grader's one-line summary of each miss. The standard library only.
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os


def _time(path: str) -> dt.datetime | None:
    try:
        with open(path) as f:
            return dt.datetime.fromisoformat(f.read().strip().replace("Z", "+00:00"))
    except (OSError, ValueError):
        return None


def row(out: str, label: str) -> dict | None:
    """One row's figures from OUT/<label>.{json,log,start,end}; None without the result file."""
    try:
        with open(os.path.join(out, f"{label}.json")) as f:
            d = json.load(f)
    except (OSError, ValueError):
        return None
    hits = None
    try:
        with open(os.path.join(out, f"{label}.log")) as f:
            for line in f:
                if line.startswith("response_cache_hits "):
                    hits = line.split()[1]
    except OSError:
        pass
    a, b = _time(os.path.join(out, f"{label}.start")), _time(os.path.join(out, f"{label}.end"))
    results = (d.get("scores") or {}).get("scenario_results") or []
    misses = [(r["scenario_id"], r["status"], r.get("summary") or "") for r in results
              if r.get("status") != "pass"]
    flags = ""
    try:
        with open(os.path.join(out, f"{label}.flags")) as f:
            flags = f.read().strip()
    except OSError:
        pass
    return {"label": label, "score": d.get("final_score"), "deployability": d.get("deployability"),
            "responsiveness": d.get("responsiveness"), "n": d.get("total_scenarios"),
            "minutes": round((b - a).total_seconds() / 60, 1) if a and b else None,
            "hits": hits, "misses": misses, "flags": flags,
            "engine": (d.get("metadata") or {}).get("engine_version"),
            "bench": (d.get("metadata") or {}).get("tool_version")}


def labels(out: str) -> list[str]:
    """The batch's rows in the order they ran."""
    got = []
    for p in glob.glob(os.path.join(out, "*.start")):
        name = os.path.basename(p)[:-len(".start")]
        got.append((_time(p) or dt.datetime.min.replace(tzinfo=dt.timezone.utc), name))
    return [n for _, n in sorted(got)]


def _short(misses: list) -> str:
    return " ".join(f"{s}:{st[0]}" for s, st, _ in misses) or "none"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out")
    ap.add_argument("labels", nargs="*")
    ap.add_argument("--md", action="store_true", help="a Markdown table")
    ap.add_argument("--misses", action="store_true", help="the grader's reason for each miss")
    a = ap.parse_args(argv)
    rows = [r for r in (row(a.out, n) for n in (a.labels or labels(a.out))) if r is not None]
    if not rows:
        print(f"no rows in {a.out}")
        return 1
    if a.md:
        print("| row | score | deployability | responsiveness | time | cache hits | not passed |")
        print("|-|-|-|-|-|-|-|")
        for r in rows:
            part = ", ".join(s for s, st, _ in r["misses"] if st == "partial")
            fail = ", ".join(s for s, st, _ in r["misses"] if st != "partial")
            miss = "; ".join(x for x in (part and f"{part} partial", fail and f"{fail} failed")
                             if x) or "none"
            t = f"{r['minutes']} min" if r["minutes"] is not None else "-"
            print(f"| {r['label']} | {r['score']} | {r['deployability']} | {r['responsiveness']} "
                  f"| {t} | {r['hits']} | {miss} |")
    else:
        for r in rows:
            mark = "" if r["hits"] in ("0", None) else "  (replayed answers: timings invalid)"
            print(f"{r['label']:14} {r['score']!s:>4} dep {r['deployability']} "
                  f"resp {r['responsiveness']} {r['minutes']!s:>5} min hits {r['hits']} "
                  f"n={r['n']} {_short(r['misses'])}{mark}")
    if a.misses:
        print()
        for r in rows:
            for sid, st, why in r["misses"]:
                print(f"{r['label']:14} {sid} {st:8} {why}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
