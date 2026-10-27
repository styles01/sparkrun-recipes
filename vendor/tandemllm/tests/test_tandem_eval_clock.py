"""A suite whose prompt names a wall-clock deadline gets each run's own (OPS-63, QA OPS-64).

photo3d's prompt says "until 07:15" and its PROGRESS.yaml `stop_at: "07:15"`. In eval night 2
(2026-10-01-r2) photo3d raw r2 started at 06:44 and stopped itself at 07:15 after 30.9 minutes
("stopped at time limit 07:15"), and tandem r2 started after 07:15. Now:

  1  every run's prompt, continuation and rewrite_files carry its own deadline (start plus the
     suite's timeout_min), and the record keeps both      test_a_run_gets_its_own_clock
  2  an older night's run whose window held the fixed clock is flagged in the report
                                                         test_an_older_run_that_hit_the_clock_*

Against the fake opencode and the fake engine API of tests/test_tandem_eval.py.

Run: python tests/test_tandem_eval_clock.py
"""

from __future__ import annotations

import datetime as dt
import json
import os
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from test_tandem_eval import FakeEngine, _setup  # noqa: E402
from tools import tandem_eval as te  # noqa: E402


def _iso(local: dt.datetime) -> str:
    return te.now_iso(time.mktime(local.timetuple()))


def test_the_run_clock_is_its_start_plus_the_budget():
    t0 = time.mktime(dt.datetime(2026, 10, 2, 6, 44, 10).timetuple())
    assert te.run_clock({"clock": "07:15", "timeout_min": 120}, t0) == "08:44"
    assert te.run_clock({"timeout_min": 120}, t0) is None
    s = {"clock": "07:15", "rewrite": ["~/demos/x"]}
    assert te.rewrite("in ~/demos/x until 07:15, or at 07:15", s, "/r", "08:44") == \
        "in /r until 08:44, or at 08:44"
    assert te.rewrite("until 07:15", s, "/r") == "until 07:15", "no run clock: unchanged"


def test_a_run_gets_its_own_clock():
    eng = FakeEngine()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            path, oc, store = _setup(tmp)
            doc = json.load(open(path))
            suite = doc["suites"][0]
            tpl = suite["template"]
            with open(os.path.join(tpl, "PROGRESS.yaml"), "w") as f:
                f.write('stop_at: "07:15"\nstage: S1\n')
            suite.update(clock="07:15", rewrite_files=["PROGRESS.yaml"],
                         prompt="Read SPEC.md and do it, alone, until 07:15. Run python3 check.py.",
                         timeout_min=2)
            json.dump(doc, open(path, "w"))
            old = dict(os.environ)
            os.environ.update({"FAKE_OC_STORE": store, "TANDEM_EVAL_KEY": "k",
                               "QSE_ADMIN_TOKEN": "adm"})
            try:
                rc = te.main(["--suites", path, "--n", "1", "--base", eng.url + "/v1",
                              "--work", os.path.join(tmp, "work"), "--out",
                              os.path.join(tmp, "out"), "--night", "2026-10-02",
                              "--opencode", oc])
            finally:
                os.environ.clear()
                os.environ.update(old)
            assert rc == 0
            rep = json.load(open(os.path.join(tmp, "out", "2026-10-02", "report.json")))
            assert not te.validate(rep), te.validate(rep)[:5]
            calls = [json.loads(x) for x in open(os.path.join(store, "calls.jsonl"))]
            for r, c in zip(rep["runs"], calls):
                own = r["clock"]["run"]
                start = dt.datetime.fromtimestamp(te._iso_ms(r["started_at"]) / 1000)
                want = (start + dt.timedelta(minutes=2)).strftime("%H:%M")
                assert r["clock"]["suite"] == "07:15" and own == want, (r["clock"], want)
                assert f"until {own}." in c["argv"][-1]
                assert own == "07:15" or "07:15" not in c["argv"][-1]
                prog = open(os.path.join(r["dir"], "PROGRESS.yaml")).read()
                assert f'stop_at: "{own}"' in prog, prog
                assert "clock_hit" not in r
    finally:
        eng.close()


def _rec(start: dt.datetime, minutes: float, **kw) -> dict:
    return {"suite": "photo3d", "arm": "raw", "rep": 2, "tag": "te-x-photo3d-raw-r2-aa",
            "started_at": _iso(start), "ended_at": _iso(start + dt.timedelta(minutes=minutes)),
            "error": None, **kw}


def test_an_older_run_that_hit_the_clock_is_flagged():
    s = {"id": "photo3d", "clock": "07:15", "timeout_min": 120}
    # night 2's photo3d raw r2: started 06:44:07, stopped at the prompt's 07:15
    hit = te.clock_hit(_rec(dt.datetime(2026, 10, 2, 6, 44, 7), 30.89), s)
    assert hit == {"at": "07:15", "minutes_from_start": 30.9}, hit
    # a run that ended before the clock, a run that started after it (the clock is tomorrow's),
    # a run with its own clock, a suite without one: not flagged
    assert te.clock_hit(_rec(dt.datetime(2026, 10, 2, 5, 0), 60), s) is None
    assert te.clock_hit(_rec(dt.datetime(2026, 10, 2, 7, 20), 60), s) is None
    assert te.clock_hit(_rec(dt.datetime(2026, 10, 2, 6, 44), 31,
                             clock={"suite": "07:15", "run": "08:44"}), s) is None
    assert te.clock_hit(_rec(dt.datetime(2026, 10, 2, 6, 44), 31), {"id": "photo3d"}) is None
    # a run across midnight that reaches the next morning's clock
    assert te.clock_hit(_rec(dt.datetime(2026, 10, 1, 23, 30), 8 * 60), s)["at"] == "07:15"


def test_an_older_run_that_hit_the_clock_is_flagged_in_the_report():
    suites = [{"id": "photo3d", "title": "photo to 3D", "clock": "07:15", "timeout_min": 120}]
    base = {"store_clear": None, "dir": None, "session_id": "s", "exit_code": 0,
            "timed_out": False, "wall_minutes": 30.89, "passed": False,
            "gates": {"total": 5, "passed": 2, "by_gate": []}, "minutes_to_pass": None,
            "final_check": None, "tokens": None, "cache_hit": None, "gen_tps": None,
            "turns": 40, "tool_calls": 39, "tool_errors": 0, "empty_stops": 0,
            "last_finish": "stop", "continuations": {"needed": 0, "sent": 0, "driver": "watch"},
            "engine": None, "conductor": None,
            "pristine": {"ok": True, "changed": [], "restored": []}}
    runs = [dict(_rec(dt.datetime(2026, 10, 2, 6, 44, 7), 30.89), **base)]
    meta = {"night": "2026-10-01-r2", "build": {"version": "0.3.7-test", "code_sha": "add2cc4"},
            "base": "b", "profile": "overnight", "driver": {"mode": "watch", "max_continues": 0}}
    rep = te.build_report(meta, runs, suites, {"status": "missing", "source": None, "row": None})
    assert not te.validate(rep), te.validate(rep)[:5]
    assert rep["runs"][0]["clock_hit"] == {"at": "07:15", "minutes_from_start": 30.9}
    assert "ran into its prompt's fixed clock 07:15 30.9 minutes after its start" in te.to_md(rep)
    assert "clock_hit" not in runs[0], "the record itself is not changed"


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:64s} ok")
            passed += 1
    print(f"{passed} passed")
