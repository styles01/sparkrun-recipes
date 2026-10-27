"""OPS-63's open item (QA OPS-64): `tools/tandem_eval.py --resume` continues a night across
interruptions.

A night of 24 runs takes about nine hours, and the Mac, the ssh link or the session can end it in
the middle. Every run's record is written when the run ends; `--resume` with the same night runs
only what the night has no record for:

  1  a night goes on where it stopped, the kept records untouched     test_resume_goes_on_*
  2  a cut run (no record) and an errored run are run again          test_resume_runs_a_cut_*
  3  the store is cleared before the first resumed run               (in 2)
  4  a resume on another build says so; nothing to resume refuses    test_resume_on_another_*,
                                                                       test_resume_with_nothing_*

Against the fake opencode and the fake engine API of tests/test_tandem_eval.py.

Run: python tests/test_tandem_eval_resume.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from test_tandem_eval import BUILD, FakeEngine, _setup  # noqa: E402
from tools import tandem_eval as te  # noqa: E402

NIGHT = "2026-10-01-r2"


class Harness:
    """One suites file, fake opencode and output folder, for several invocations of a night."""

    def __init__(self, tmp, eng, n_suites=1):
        self.tmp, self.eng = tmp, eng
        self.path, self.oc, self.store = _setup(tmp, n_suites)
        self.dir = os.path.join(tmp, "out", NIGHT)

    def main(self, *extra):
        old = dict(os.environ)
        os.environ.update({"FAKE_OC_STORE": self.store, "TANDEM_EVAL_KEY": "k",
                           "QSE_ADMIN_TOKEN": "adm"})
        try:
            return te.main(["--suites", self.path, "--base", self.eng.url + "/v1",
                            "--work", os.path.join(self.tmp, "work"),
                            "--out", os.path.join(self.tmp, "out"), "--night", NIGHT,
                            "--opencode", self.oc, *extra])
        finally:
            os.environ.clear()
            os.environ.update(old)

    def report(self) -> dict:
        with open(os.path.join(self.dir, "report.json")) as f:
            return json.load(f)

    def records(self) -> dict:
        folder = os.path.join(self.dir, "runs")
        return {n[:-5]: json.load(open(os.path.join(folder, n))) for n in os.listdir(folder)}

    def night(self) -> dict:
        with open(os.path.join(self.dir, "night.json")) as f:
            return json.load(f)


def test_resume_goes_on_where_the_night_stopped():
    eng = FakeEngine()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            h = Harness(tmp, eng)
            assert h.main("--n", "1") == 0
            first = h.records()
            assert len(first) == 2 and h.night()["build"] == BUILD
            assert h.main("--n", "2", "--resume") == 0
            rep, recs = h.report(), h.records()
            assert not te.validate(rep), te.validate(rep)[:5]
            assert len(rep["runs"]) == len(recs) == 4
            # the kept records are the night's own, untouched; rep 1 ran on both arms
            for tag, r in first.items():
                assert recs[tag] == r
            assert sorted((r["arm"], r["rep"]) for r in rep["runs"]) == \
                [("raw", 0), ("raw", 1), ("tandem", 0), ("tandem", 1)]
            assert [r["n"] for r in rep["rows"]] == [2, 2]
            res = rep["resumed"]
            assert len(res) == 1 and (res[0]["kept"], res[0]["left"]) == (2, 2)
            assert res[0]["run_again"] == [] and res[0]["build_changed"] is False
            assert rep["build"] == BUILD and "changed_during_night" not in rep["build"]
            assert "Resumed at" in open(os.path.join(h.dir, "report.md")).read()
            # a night with nothing left runs nothing and writes the same report
            calls = len(open(os.path.join(h.store, "calls.jsonl")).readlines())
            assert h.main("--n", "2", "--resume") == 0
            assert len(open(os.path.join(h.store, "calls.jsonl")).readlines()) == calls
            assert len(h.report()["runs"]) == 4 and h.report()["resumed"][-1]["left"] == 0
    finally:
        eng.close()


def test_resume_runs_a_cut_and_an_errored_run_again():
    eng = FakeEngine()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            h = Harness(tmp, eng, n_suites=2)
            assert h.main("--n", "1") == 0
            recs = h.records()
            assert len(recs) == 4
            by = {(r["suite"], r["arm"]): t for t, r in recs.items()}
            cut, bad = by[("answer1", "raw")], by[("answer1", "tandem")]
            # the run that was cut never wrote its record; another ended in an error
            os.remove(os.path.join(h.dir, "runs", cut + ".json"))
            r = recs[bad]
            r["error"] = "RuntimeError: opencode gave no session"
            with open(os.path.join(h.dir, "runs", bad + ".json"), "w") as f:
                json.dump(r, f)
            n_clears = sum(c == ("POST", "/v1/cache/clear") for c in eng.calls)
            assert h.main("--n", "1", "--resume") == 0
            rep, now = h.report(), h.records()
            assert len(rep["runs"]) == len(now) == 4 and not any(r["error"] for r in rep["runs"])
            new = set(now) - set(recs)
            assert len(new) == 2 and cut not in now and bad not in now, "new tags"
            assert {(now[t]["suite"], now[t]["arm"]) for t in new} == \
                {("answer1", "raw"), ("answer1", "tandem")}
            kept = [d for d in os.listdir(h.dir) if d.startswith("runs-superseded-")]
            assert len(kept) == 1 and os.listdir(os.path.join(h.dir, kept[0])) == [bad + ".json"]
            assert rep["resumed"][0]["run_again"] == [bad]
            # the prefix store is cleared before the first resumed run, and between arms
            assert sum(c == ("POST", "/v1/cache/clear") for c in eng.calls) == n_clears + 2
            assert all(now[t]["store_clear"]["status"] == 200 for t in new)
            assert rep["isolation"]["unique_tags"]
    finally:
        eng.close()


def test_resume_on_another_build_says_so():
    eng = FakeEngine()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            h = Harness(tmp, eng)
            assert h.main("--n", "1") == 0
            doc = h.night()
            doc["build"] = {"version": "0.3.7-test", "code_sha": "add2cc4"}
            with open(os.path.join(h.dir, "night.json"), "w") as f:
                json.dump(doc, f)
            assert h.main("--n", "2", "--resume") == 0
            rep = h.report()
            assert rep["build"] == {"version": "0.3.7-test", "code_sha": "add2cc4",
                                    "changed_during_night": True}
            assert rep["resumed"][0]["build_changed"] is True
            assert "on another build" in open(os.path.join(h.dir, "report.md")).read()
    finally:
        eng.close()


def test_resume_with_nothing_to_resume_refuses():
    eng = FakeEngine()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            h = Harness(tmp, eng)
            try:
                h.main("--n", "1", "--resume")
                raise AssertionError("an empty night is not resumed")
            except SystemExit as exc:
                assert "nothing to resume" in str(exc)
    finally:
        eng.close()


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:64s} ok")
            passed += 1
    print(f"{passed} passed")
