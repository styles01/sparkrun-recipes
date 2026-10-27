"""tools/teb_batch.sh and tools/teb_summary.py (OPS-209): the rows a batch runs and the table it
prints, on a fake results folder and a dry run. CPU only, no bench, no server.

Run: python -m pytest tests/test_teb_tools.py
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tempfile
from contextlib import redirect_stdout

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import teb_summary  # noqa: E402


def _row(out, label, score, results, hits="0", start="2026-10-20T10:00:00Z",
         end="2026-10-20T10:12:30Z"):
    with open(os.path.join(out, f"{label}.json"), "w") as f:
        json.dump({"final_score": score, "deployability": 84, "responsiveness": 55,
                   "total_scenarios": len(results),
                   "scores": {"scenario_results": [
                       {"scenario_id": s, "status": st, "summary": why} for s, st, why in results]},
                   "metadata": {"engine_version": "0.12.2+exl3-6.0", "tool_version": "2.8.1"}},
                  f)
    with open(os.path.join(out, f"{label}.log"), "w") as f:
        f.write(f"exit 0\nresponse_cache_hits {hits}\n")
    for name, t in (("start", start), ("end", end)):
        with open(os.path.join(out, f"{label}.{name}"), "w") as f:
            f.write(t + "\n")


def _run(*argv) -> str:
    buf = io.StringIO()
    with redirect_stdout(buf):
        assert teb_summary.main(list(argv)) == 0
    return buf.getvalue()


def test_the_table_names_each_miss_and_the_replayed_rows():
    out = tempfile.mkdtemp(prefix="teb-")
    _row(out, "raw-xhigh", 97, [("TC-01", "pass", ""), ("TC-23", "partial", "unclear"),
                                ("TC-68", "fail", "not JSON")])
    _row(out, "cond-tools", 96, [("TC-01", "pass", ""), ("TC-49", "partial", "intent")],
         hits="12", start="2026-10-20T10:13:00Z", end="2026-10-20T10:20:00Z")
    text = _run(out)
    lines = text.strip().splitlines()
    assert lines[0].split()[:2] == ["raw-xhigh", "97"], lines
    assert "12.5 min" in lines[0] and "TC-23:p TC-68:f" in lines[0]
    assert lines[1].startswith("cond-tools") and "replayed answers" in lines[1]
    md = _run(out, "--md", "--misses")
    assert "| raw-xhigh | 97 | 84 | 55 | 12.5 min | 0 | TC-23 partial; TC-68 failed |" in md
    assert "cond-tools     TC-49 partial  intent" in md


def test_a_dry_run_lists_the_standard_rows():
    got = subprocess.run(["bash", os.path.join(ROOT, "tools", "teb_batch.sh"), "--dry-run",
                          "--base-url", "http://box:8000/v1", "standard"],
                         capture_output=True, text=True, check=True).stdout.splitlines()
    assert [ln.split(":")[0] for ln in got] == ["raw-xhigh", "cond", "hard-raw", "hard-cond"]
    for ln in got:
        assert "--seed 42 --temperature 0 --timeout 600 --no-live --parallel 1" in ln
        assert "--base-url http://box:8000/v1" in ln
    assert "--header X-Tandem=raw --backend-kwargs '{\"reasoning_effort\":\"xhigh\"}'" in got[0]
    # the conductor rows pin no profile: the server resolves each request by what it offers
    assert "X-Tandem" not in got[1] and "X-Tandem" not in got[3]
    assert got[3].endswith("--hardmode-only")



def test_a_row_with_no_flags_runs():
    """The `cond` row has no flags; bash 3.2 under set -u called its empty array unbound and the batch
    stopped before the row (2026-10-21). A fake bench prints its arguments, a fake /health answers."""
    with tempfile.TemporaryDirectory() as d:
        bench = os.path.join(d, "bench.sh")
        with open(bench, "w") as fh:
            fh.write('#!/bin/bash\necho "args: $*"\n')
        os.chmod(bench, 0o755)
        health = "echo '{\"version\": \"x\", \"cache\": {\"response_cache\": {\"hits\": 0}}}'"
        got = subprocess.run(["bash", os.path.join(ROOT, "tools", "teb_batch.sh"), "--out", d,
                              "--health", health, "--bench", bench, "cond:",
                              "raw:--header X-Tandem=raw --backend-kwargs '{\"reasoning_effort\": \"low\"}'"],
                             capture_output=True, text=True)
        assert got.returncode == 0, got.stdout + got.stderr
        cond = open(os.path.join(d, "cond.log")).read()
        assert cond.startswith("args: run ") and "--label cond --json-file" in cond and "exit 0" in cond, cond
        raw = open(os.path.join(d, "raw.log")).read()
        assert "--backend-kwargs {\"reasoning_effort\": \"low\"}" in raw, raw
        assert "response_cache_hits 0" in raw

if __name__ == "__main__":
    from scriptmode import run
    run(globals())
