"""The speed guard's result for a build, as a report the Conductor view reads.

    python tools/speed_guard.py BASE.json CAND.json --server http://127.0.0.1:8000 \
        [--out ~/.qwen38-spark-engine/conductor/evals] [--note TEXT]

BASE and CAND are two `tools/row3.py` reports from the same hour: the release (the raw engine's
never-worse baseline) and the build. The report goes to the conductor's `evals/` directory as
`speed-guard-<version>-<code_sha>.json`, and `GET /v1/conductor/evals` serves it beside the
suite reports. The build is the one the server at `--server` serves (`/v1/conductor/evals`'s
`engine`), or `--version` and `--code-sha` when the rows were taken before the deploy.

The verdict follows `tools/row3.py --compare`'s rule (`verdicts`): a statistic is RESOLVED when
its difference is bigger than the run-to-run noise and the run ranges do not overlap.

    pass               every side ran 2 runs or more and nothing resolved worse
    worse              something resolved worse (the statistics are listed)
    no_noise_estimate  a side ran 1 run: the noise is unknown, so the row cannot say whether
                       any difference is real; the deltas are reported as measured

The project's short-gate rule (2026-10-01 12:58) runs ONE pair of 1 run each, so a short gate ends in
`no_noise_estimate` by design; the report says so instead of reading as a pass or a failure.

A release carries its result (VIS-68, 2026-10-07):

    python tools/speed_guard.py ship REPORT.json --tree DIR [--version 0.4.0]

copies a candidate's report into the release tree as `release/speed-guard.json`, the build named
as the tree's own (`VERSION`, and the code hash `tools/row3.py` takes over engine/, server/, tools/
and ops/, which the release's cleaned comments change), the measured build kept as `measured_on`.
The server serves it beside the box's own reports, so a release's build line reads its result out
of the box.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.row3 import HIGHER_IS_BETTER, verdicts  # noqa: E402

DEFAULT_OUT = os.path.expanduser("~/.qwen38-spark-engine/conductor/evals")
STATS = ("mean", "p50", "p90", "ttft_p50_ms", "wall_s", "tok_blk", "ms_blk")


def _side(rep: dict) -> dict:
    return {"label": rep.get("label"), "runs": len(rep.get("runs") or ()),
            "code_sha": (rep.get("code_sha256") or "")[:16] or None,
            "suffix_store": rep.get("suffix_store")}


def report(base: dict, cand: dict, build: dict, note: str | None = None,
           at: str | None = None) -> dict:
    """The speed_guard report for `build` from two row3 reports."""
    runs = min(len(base.get("runs") or ()), len(cand.get("runs") or ()))
    noise_known = runs >= 2
    stats = []
    for v in verdicts(base, cand):
        if v["stat"] not in STATS or v["verdict"] == "n/a":
            continue
        stats.append({"stat": v["stat"], "base": round(v["base"], 3), "cand": round(v["other"], 3),
                      "delta_pct": round(v["delta"], 2),
                      "higher_is_better": HIGHER_IS_BETTER[v["stat"]],
                      "noise_pct": round(v["noise"], 2) if noise_known else None,
                      "verdict": v["verdict"] if noise_known else "no noise estimate"})
    worse = [s["stat"] for s in stats if noise_known and s["verdict"] == "RESOLVED WORSE"]
    if not noise_known:
        verdict = "no_noise_estimate"
        why = (f"{runs} run a side: the run-to-run noise cannot be estimated, so this row cannot "
               "say whether any difference is real; the deltas are as measured")
    elif worse:
        verdict, why = "worse", "resolved worse: " + ", ".join(worse)
    else:
        verdict, why = "pass", f"nothing resolved worse ({runs} runs a side)"
    return {"kind": "speed_guard", "contract_version": "1.0",
            "created_at": at or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "build": {"version": build.get("version"), "code_sha": build.get("code_sha")},
            "base": _side(base), "cand": _side(cand), "spec": cand.get("spec"),
            "runs_each": runs, "noise_estimated": noise_known, "stats": stats,
            "verdict": verdict, "why": why, "note": note}


def served_build(server: str) -> dict:
    with urllib.request.urlopen(server.rstrip("/") + "/v1/conductor/evals", timeout=10) as r:
        eng = json.load(r).get("engine") or {}
    return {"version": eng.get("version"), "code_sha": eng.get("code_sha")}


def file_name(build: dict) -> str:
    safe = lambda s: "".join(c if c.isalnum() or c in ".-_" else "_" for c in str(s or "unknown"))
    return f"speed-guard-{safe(build.get('version'))}-{safe(build.get('code_sha'))}.json"


def ship(report_path: str, tree: str, version: str | None = None) -> str:
    """Write `report_path`'s report into `tree` as release/speed-guard.json for `tree`'s own build."""
    from pathlib import Path
    from tools.row3 import code_hash
    with open(report_path) as f:
        rep = json.load(f)
    if rep.get("kind") != "speed_guard":
        raise SystemExit(f"speed_guard ship: {report_path} is no speed_guard report")
    ver = version or open(os.path.join(tree, "VERSION")).read().strip()
    out = dict(rep, measured_on=rep.get("measured_on") or rep.get("build"),
               build={"version": ver, "code_sha": code_hash(Path(tree))[:16]})
    out.pop("file", None)
    d = os.path.join(tree, "release")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "speed-guard.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    return path


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "ship":
        sp = argparse.ArgumentParser(description="copy a speed_guard report into a release tree")
        sp.add_argument("report")
        sp.add_argument("--tree", required=True)
        sp.add_argument("--version")
        a = sp.parse_args(sys.argv[2:])
        path = ship(a.report, a.tree, a.version)
        print(f"[speed_guard] shipped {a.report} as {path}")
        return
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base")
    ap.add_argument("cand")
    ap.add_argument("--server", help="read the build from this server's /v1/conductor/evals")
    ap.add_argument("--version")
    ap.add_argument("--code-sha")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--note")
    ap.add_argument("--print", action="store_true", help="print the report, write nothing")
    a = ap.parse_args()
    build = served_build(a.server) if a.server else {}
    build = {"version": a.version or build.get("version"),
             "code_sha": a.code_sha or build.get("code_sha")}
    if not build["version"]:
        raise SystemExit("speed_guard: no build (pass --server, or --version and --code-sha)")
    base, cand = (json.load(open(p)) for p in (a.base, a.cand))
    rep = report(base, cand, build, a.note)
    if a.print:
        print(json.dumps(rep, indent=1))
        return
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, file_name(build))
    tmp = os.path.join(a.out, "." + os.path.basename(path))
    with open(tmp, "w") as f:
        json.dump(rep, f, indent=1)
    os.replace(tmp, path)
    print(f"[speed_guard] {build['version']} {build['code_sha']}: {rep['verdict']} -- {rep['why']}")
    for s in rep["stats"]:
        print(f"  {s['stat']:<12} {s['base']:9.2f} -> {s['cand']:9.2f} ({s['delta_pct']:+6.2f} %)"
              f"  {s['verdict']}")
    print(f"[speed_guard] wrote {path}")


if __name__ == "__main__":
    main()
