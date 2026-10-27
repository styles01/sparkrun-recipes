"""The judge's report (J1 step 7): per question, how often Laya answered, how sure it was, how
often it agreed with the Tier 0 twin and, on the rows labelled by hand, how often it was right.

A question may move from shadow to act only at confident precision >= 0.95 and coverage >= 0.3
(the plan, 2026-10-02, step 7): `confident` is an answer at `--sure` (0.8) or above,
`coverage` the confident share of the questions asked, `precision` the share of confident answers
labelled `right` among those labelled at all (the dashboard's right / wrong buttons on the
journal's `kind: outcome` rows, or `--labels FILE`: a JSON object row id -> right | wrong). Without
labels the twin's agreement is shown and the gate says `needs labels`.

    python tools/judge_report.py [--ledger ~/.qwen38-spark-engine/usage/ledger.sqlite3]
        [--since 2026-10-02] [--sure 0.8] [--labels labels.json] [--json]

Read-only on the ledger; standard library only; nothing on the GPU.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sqlite3
import sys

PRECISION, COVERAGE = 0.95, 0.3


def rows(ledger: str, since_ms: int = 0) -> list[dict]:
    conn = sqlite3.connect(f"file:{ledger}?mode=ro", uri=True)
    try:
        cur = conn.execute("SELECT id, ts_ms, decision, detail, label FROM conductor_events "
                           "WHERE kind = 'outcome' AND task = 'judge' AND ts_ms >= ? "
                           "ORDER BY ts_ms", (since_ms,))
        out = []
        for rid, ts, decision, detail, label in cur:
            d = json.loads(detail) if detail else {}
            out.append({"id": rid, "ts_ms": ts, "answer": decision, "label": label, **d})
        return out
    finally:
        conn.close()


def report(rs: list[dict], sure: float = 0.8, labels: dict | None = None) -> dict:
    labels = labels or {}
    out = {}
    for q in sorted({r.get("question") for r in rs if r.get("question")}):
        qs = [r for r in rs if r.get("question") == q]
        answered = [r for r in qs if r["answer"] not in (None, "unknown")]
        conf = [r for r in answered if float(r.get("confidence") or 0) >= sure]
        twin = [r for r in conf if r.get("twin") is not None]
        lab = [(r, labels.get(r["id"], r.get("label"))) for r in conf]
        lab = [(r, v) for r, v in lab if v in ("right", "wrong")]
        right = sum(1 for _, v in lab if v == "right")
        cov = round(len(conf) / len(qs), 4) if qs else None
        prec = round(right / len(lab), 4) if lab else None
        ms = sorted(float(r["ms"]) for r in answered if r.get("ms") is not None)
        gate = ("needs labels" if prec is None else
                "may act" if prec >= PRECISION and (cov or 0) >= COVERAGE else "stays shadow")
        out[q] = {"asked": len(qs), "answered": len(answered), "confident": len(conf),
                  "coverage": cov, "twin_agree": round(sum(1 for r in twin if r.get("agree"))
                                                       / len(twin), 4) if twin else None,
                  "labelled": len(lab), "precision": prec,
                  "ms_p50": ms[len(ms) // 2] if ms else None,
                  "errors": {e: sum(1 for r in qs if r.get("error") == e)
                             for e in sorted({r.get("error") for r in qs if r.get("error")})},
                  "answers": {a: sum(1 for r in answered if r["answer"] == a)
                              for a in sorted({r["answer"] for r in answered})},
                  "gate": gate}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ledger", default=os.path.expanduser(
        "~/.qwen38-spark-engine/usage/ledger.sqlite3"))
    ap.add_argument("--since", help="YYYY-MM-DD, local time")
    ap.add_argument("--sure", type=float, default=0.8)
    ap.add_argument("--labels", help="JSON: row id -> right | wrong")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    since = int(dt.datetime.strptime(a.since, "%Y-%m-%d").timestamp() * 1000) if a.since else 0
    labels = json.load(open(a.labels)) if a.labels else None
    rep = report(rows(a.ledger, since), a.sure, labels)
    if a.json:
        print(json.dumps(rep, indent=1))
        return 0
    if not rep:
        print("no judge rows (the profile's outcomes.judge is off, or nothing was asked yet)")
        return 0
    print(f"{'question':10s} {'asked':>6s} {'answered':>8s} {'confident':>9s} {'coverage':>8s} "
          f"{'twin':>6s} {'labelled':>8s} {'precision':>9s} {'p50 ms':>7s}  gate")
    for q, r in rep.items():
        def f(v):
            return "-" if v is None else f"{v:.2f}" if isinstance(v, float) else str(v)
        print(f"{q:10s} {r['asked']:6d} {r['answered']:8d} {r['confident']:9d} "
              f"{f(r['coverage']):>8s} {f(r['twin_agree']):>6s} {r['labelled']:8d} "
              f"{f(r['precision']):>9s} {f(r['ms_p50']):>7s}  {r['gate']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
