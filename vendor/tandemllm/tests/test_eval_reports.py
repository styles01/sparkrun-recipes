"""The suite reports the Compare screen reads (VIS-42 follow-ups, OPS-63), on a CPU.

  1  a night with a suffix is kept: eval night 2 of 2026-10-01 is `2026-10-01-r2`, and the live
     server answered its --publish with a 422 ("report.night: YYYY-MM-DD")
  2  a report another one supersedes is listed with `superseded_by`, also when its own file was
     never marked (kept files are not rewritten); a marked one keeps its own words
  3  VIS-68: a release tree's shipped speed-guard report (release/speed-guard.json) is listed
     beside the box's own, marked `shipped`; a broken or foreign file is left out

Run: python tests/test_eval_reports.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.conductor import routes  # noqa: E402

ROW = {"suite": "minicalc", "arm": "raw"}


def _rep(night, **kw):
    return {"kind": "suite", "night": night, "rows": [ROW], **kw}


def test_a_night_may_carry_a_suffix():
    for night in ("2026-10-01", "2026-10-01-r2", "2026-10-02-ab1"):
        assert routes.eval_errors(_rep(night)) == [], night
    for night in ("2026-10-01-R2", "2026-10-01-r2/..", "night 2", "2026-10-01-", "",
                  "2026-10-01-" + "x" * 17):
        errs = routes.eval_errors(_rep(night))
        assert errs and "report.night" in errs[0], night


def _get(root):
    cond = types.SimpleNamespace(store=None, root=root, state={"version": "0.3.8-test",
                                                               "code_sha": "abc"})
    st, body, _ = routes._route(None, cond, "GET", ["evals"], {}, {}, None)
    assert st == 200
    return {r["file"]: r for r in body["reports"]}


def test_a_superseded_report_says_so():
    root = tempfile.mkdtemp(prefix="evals-")
    d = os.path.join(root, "evals")
    os.makedirs(d)
    files = {
        "suite-2026-10-01-155808.json": _rep("2026-10-01"),
        "suite-2026-10-01-180512.json": _rep("2026-10-01", supersedes={
            "file": "suite-2026-10-01-155808.json", "why": "the reader bug"}),
        "suite-2026-10-01-r2-061500.json": _rep("2026-10-01-r2"),
        # marked by hand on the box: its own words stay
        "suite-2026-09-30-100000.json": _rep("2026-09-30", superseded_by={
            "file": "suite-2026-09-30-110000.json", "why": "kept as published"}),
        "suite-2026-09-30-110000.json": _rep("2026-09-30", supersedes={
            "file": "suite-2026-09-30-100000.json", "why": "rebuilt"}),
        # a report that names a file the box does not have changes nothing
        "suite-2026-09-29-100000.json": _rep("2026-09-29", supersedes={"file": "gone.json"}),
    }
    for name, rep in files.items():
        with open(os.path.join(d, name), "w") as f:
            json.dump(rep, f)
    got = _get(root)
    assert got["suite-2026-10-01-155808.json"]["superseded_by"] == {
        "file": "suite-2026-10-01-180512.json", "why": "the reader bug"}
    assert "superseded_by" not in got["suite-2026-10-01-180512.json"]
    assert "superseded_by" not in got["suite-2026-10-01-r2-061500.json"]
    assert got["suite-2026-09-30-100000.json"]["superseded_by"]["why"] == "kept as published"
    # the files on the box are not rewritten
    with open(os.path.join(d, "suite-2026-10-01-155808.json")) as f:
        assert "superseded_by" not in json.load(f)


def test_a_shipped_speed_guard_report_is_listed():
    root = tempfile.mkdtemp(prefix="evals-")
    os.makedirs(os.path.join(root, "evals"))
    real = routes.SHIPPED_GUARD
    try:
        routes.SHIPPED_GUARD = os.path.join(root, "speed-guard.json")
        assert "release/speed-guard.json" not in _get(root)            # none shipped
        with open(routes.SHIPPED_GUARD, "w") as f:
            json.dump({"kind": "speed_guard", "build": {"version": "0.4.0", "code_sha": "abc"},
                       "verdict": "no_noise_estimate"}, f)
        got = _get(root)["release/speed-guard.json"]
        assert got["shipped"] is True and got["build"]["version"] == "0.4.0"
        for bad in ("{not json", json.dumps({"kind": "suite"})):
            with open(routes.SHIPPED_GUARD, "w") as f:
                f.write(bad)
            assert "release/speed-guard.json" not in _get(root)
    finally:
        routes.SHIPPED_GUARD = real


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:64s} ok")
            passed += 1
    print(f"{passed} passed")
