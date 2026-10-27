"""tools/speed_guard.py: the speed guard's report for a build, on a CPU.

The Conductor view's guard line read "speed guard not run" for every build, because nothing wrote
a speed_guard report. This writes one from two row3 reports and says what one run a side can and
cannot say (a test build's short gate: mean 49.98 -> 50.54, TTFT 424.9 -> 425.6 ms, 1 run each).

Run: python tests/test_speed_guard.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from tools import speed_guard  # noqa: E402

STATS = ("mean", "p50", "p90", "max", "ttft_p50_ms", "wall_s", "tok_blk", "ms_blk")


def row(label: str, runs: list[dict], sha: str = "f" * 64) -> dict:
    summ = {}
    for k in STATS:
        vals = [r[k] for r in runs]
        med = sorted(vals)[len(vals) // 2]
        summ[k] = {"median": med, "min": min(vals), "max": max(vals),
                   "spread_pct": 100.0 * (max(vals) - min(vals)) / med}
    return {"label": label, "runs": runs, "summary": summ, "code_sha256": sha,
            "suffix_store": "off", "spec": "packets/ownengine.serve-single-i256-o256-v1.json"}


def run(mean, ttft, wall=236.0, msb=92.8):
    return {"mean": mean, "p50": mean - 3.5, "p90": mean + 18, "max": mean + 35,
            "ttft_p50_ms": ttft, "wall_s": wall, "tok_blk": 4.58, "ms_blk": msb}


BUILD = {"version": "0.3.4-test", "code_sha": "3c64a688569d2967"}


def test_one_run_a_side_says_the_noise_is_unknown():
    base = row("c34off-base", [run(49.98, 424.9)], "a" * 64)
    cand = row("c34off-cand", [run(50.54, 425.6)], "b" * 64)
    rep = speed_guard.report(base, cand, BUILD, at="2026-10-01T11:55:00Z")
    assert rep["kind"] == "speed_guard" and rep["build"] == BUILD
    assert rep["verdict"] == "no_noise_estimate" and rep["noise_estimated"] is False
    assert rep["runs_each"] == 1 and "cannot be estimated" in rep["why"]
    by = {s["stat"]: s for s in rep["stats"]}
    assert by["mean"]["base"] == 49.98 and by["mean"]["cand"] == 50.54
    assert by["mean"]["delta_pct"] == 1.12 and by["mean"]["noise_pct"] is None
    assert by["ttft_p50_ms"]["delta_pct"] == 0.16 and by["ttft_p50_ms"]["higher_is_better"] is False
    assert all(s["verdict"] == "no noise estimate" for s in rep["stats"]), \
        "row3's rule calls every 1-run difference RESOLVED; the report must not"
    assert rep["base"] == {"label": "c34off-base", "runs": 1, "code_sha": "a" * 16,
                           "suffix_store": "off"}


def test_three_runs_a_side_pass_and_worse():
    base = row("b", [run(50.0, 425), run(50.5, 426), run(49.6, 427)])
    same = row("c", [run(50.2, 426), run(49.8, 425), run(50.4, 428)])
    rep = speed_guard.report(base, same, BUILD)
    assert rep["verdict"] == "pass" and rep["noise_estimated"] is True
    assert all(s["noise_pct"] is not None for s in rep["stats"])
    slow = row("c", [run(45.0, 425, wall=260), run(45.3, 426, wall=262), run(44.8, 427, wall=261)])
    rep = speed_guard.report(base, slow, BUILD)
    assert rep["verdict"] == "worse" and "mean" in rep["why"] and "wall_s" in rep["why"]


def test_the_cli_writes_one_file_per_build_into_evals():
    base = row("c34off-base", [run(49.98, 424.9)])
    cand = row("c34off-cand", [run(50.54, 425.6)])
    with tempfile.TemporaryDirectory() as d:
        pb, pc = os.path.join(d, "b.json"), os.path.join(d, "c.json")
        json.dump(base, open(pb, "w"))
        json.dump(cand, open(pc, "w"))
        out = os.path.join(d, "evals")
        cmd = [sys.executable, os.path.join(os.path.dirname(HERE), "tools", "speed_guard.py"),
               pb, pc, "--version", BUILD["version"], "--code-sha", BUILD["code_sha"],
               "--out", out, "--note", "short gate"]
        got = subprocess.run(cmd, capture_output=True, text=True, check=True)
        assert "no_noise_estimate" in got.stdout
        files = os.listdir(out)
        assert files == ["speed-guard-0.3.4-test-3c64a688569d2967.json"], files
        rep = json.load(open(os.path.join(out, files[0])))
        assert rep["note"] == "short gate" and rep["verdict"] == "no_noise_estimate"
        # the same build again replaces its file
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        assert len(os.listdir(out)) == 1
        bad = subprocess.run(cmd[:4] + ["--out", out], capture_output=True, text=True)
        assert bad.returncode != 0 and "no build" in bad.stderr


def test_ship_puts_the_report_into_a_release_tree_for_its_own_build():
    """VIS-68: a release reads its speed check out of the box: the fold ships the candidate's report
    as release/speed-guard.json, named for the tree's own version and code hash; the server serves
    it beside the box's reports (server/conductor/routes.py)."""
    from pathlib import Path
    from tools.row3 import code_hash
    rep = speed_guard.report(row("b", [run(49.98, 424.9)]), row("c", [run(50.54, 425.6)]), BUILD)
    with tempfile.TemporaryDirectory() as d:
        tree = os.path.join(d, "rel")
        for sub in ("engine", "server", "tools", "ops"):
            os.makedirs(os.path.join(tree, sub))
            open(os.path.join(tree, sub, "x.py"), "w").write(f"# {sub}\n")
        open(os.path.join(tree, "VERSION"), "w").write("0.4.0\n")
        src = os.path.join(d, "guard.json")
        json.dump(rep, open(src, "w"))
        before = code_hash(Path(tree))
        cmd = [sys.executable, os.path.join(os.path.dirname(HERE), "tools", "speed_guard.py"), "ship", src,
               "--tree", tree]
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        got = json.load(open(os.path.join(tree, "release", "speed-guard.json")))
        assert got["build"] == {"version": "0.4.0", "code_sha": before[:16]}, got["build"]
        assert got["measured_on"] == BUILD and got["verdict"] == "no_noise_estimate"
        assert code_hash(Path(tree)) == before, "the shipped report is outside the hashed code"


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:60s} ok")
            passed += 1
    print(f"{passed} passed")
