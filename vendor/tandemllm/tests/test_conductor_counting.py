"""One counting for the classifier's accuracy (night plan A2), on a CPU, no model.

On 2026-10-01 the same Laya answers on the 216 labelled prompts were reported as 63.9 % in the
morning and 50.0 % at night. Both were `accuracy` = right AND at or above the stage's threshold,
over all prompts; the morning counted at 0.60, the night at 0.80. `tools/conductor_eval.py` now
reports four numbers with the threshold they were counted at, and re-counts an old report at any
threshold without asking a model:

  1  the four numbers from per-prompt answers                 test_the_four_numbers_*
  2  every run's report carries them, at the stage's threshold test_a_report_*
  3  an old report re-counts at other thresholds              test_rescore_*

Run: python tests/test_conductor_counting.py
"""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.conductor import schema  # noqa: E402
from tools import conductor_eval  # noqa: E402


def _ans(label, conf, error=None):
    return {"label": label, "confidence": conf, "probs": {label: conf}, "ms": 1.0,
            "error": error, "accepted": conf >= 0.8}


# eight prompts: best guess right on six; at 0.60 five sure (four right), at 0.80 three (three)
ROWS = [
    {"label": "chat", "decider": _ans("chat", 0.95)},
    {"label": "chat", "decider": _ans("chat", 0.85)},
    {"label": "plan", "decider": _ans("plan", 0.81)},
    {"label": "plan", "decider": _ans("plan", 0.70)},
    {"label": "debug", "decider": _ans("code_edit", 0.65)},
    {"label": "debug", "decider": _ans("debug", 0.40)},
    {"label": "research", "decider": _ans("chat", 0.30)},
    {"label": "research", "decider": _ans("research", 0.9, error="timeout")},
]


def test_the_four_numbers_at_two_thresholds():
    at6 = conductor_eval.counting(ROWS, "decider", 0.6)
    assert at6 == {"threshold": 0.6, "n": 8, "errors": 1, "confident": 5, "confident_right_n": 4,
                   "all_right": 0.75, "confident_right": 0.5, "coverage": 0.625,
                   "confident_precision": 0.8}, at6
    at8 = conductor_eval.counting(ROWS, "decider", 0.8)
    assert (at8["all_right"], at8["confident_right"], at8["coverage"],
            at8["confident_precision"]) == (0.75, 0.375, 0.375, 1.0), at8
    # all_right ignores the threshold; an errored answer is never confident
    assert conductor_eval.counting(ROWS, "decider", 0.99)["confident"] == 0
    assert conductor_eval.counting(ROWS, "decider", 0.99)["confident_precision"] is None
    assert conductor_eval.counting([], "decider", 0.8)["all_right"] == 0.0


class Fixed:
    """A decider that answers from a table: text -> (label, confidence)."""

    def __init__(self, table):
        self.table = table

    def choice(self, state, question, options, max_chars=None):
        from server.conductor.classify import Answer
        label, conf = self.table[state.split("\n")[0]]
        return Answer("decider", label, conf, {label: conf}, 1.0, model="fixed")


def test_a_report_carries_the_counting_at_the_stage_threshold():
    items = [{"text": f"prompt {i}", "label": r["label"]} for i, r in enumerate(ROWS[:7])]
    dec = Fixed({f"prompt {i}": (r["decider"]["label"], r["decider"]["confidence"])
                 for i, r in enumerate(ROWS[:7])})
    router = schema.seed("router")
    got = conductor_eval.report({"s": items}, ["sized"], router, dec, None)
    st = got["variants"]["sized"]["s"]["stages"]
    c = st["decider"]["counting"]
    assert c["threshold"] == 0.8 and c["confident"] == 3 and c["confident_right_n"] == 3
    # the same figures as the stage's own columns, under their names
    assert c["confident_right"] == st["decider"]["accuracy"]
    assert c["all_right"] == st["decider"]["top1"]
    assert c["confident_precision"] == st["decider"]["accuracy_answered"]
    assert c["coverage"] == round(st["decider"]["answered"] / st["decider"]["n"], 4)
    assert "counting" not in st["cascade"] and "counting" not in st["router"]


def _legacy(path, rows, th):
    """A report as the morning of 2026-10-01 wrote it: one set, the stage named arbiter."""
    with open(path, "w") as f:
        json.dump({"kind": "classifier", "set": {"name": "tests/fixtures/x.jsonl", "n": len(rows)},
                   "config": {"thresholds": {"arbiter": th, "qwen": 0.55}},
                   "rows": [{"text": "t", "label": r["label"], "arbiter": r["decider"],
                             "qwen": None} for r in rows]}, f)


def test_rescore_reads_the_old_shape_at_its_own_and_other_thresholds():
    tmp = tempfile.mkdtemp(prefix="rescore-")
    old = os.path.join(tmp, "morning.json")
    _legacy(old, ROWS, 0.6)
    own = conductor_eval.rescore(old)
    assert [(c["stage"], c["threshold"], c["confident_right"]) for c in own] == \
        [("decider", 0.6, 0.5)]
    both = conductor_eval.rescore(old, [0.6, 0.8])
    assert [c["confident_right"] for c in both] == [0.5, 0.375]
    assert {c["all_right"] for c in both} == {0.75}, "the best guess does not move"
    with redirect_stdout(io.StringIO()) as buf:
        assert conductor_eval.main(["--rescore", old, "--at", "0.6,0.8"]) == 0
    text = buf.getvalue()
    assert "confident right" in text and "coverage" in text and " 0.60 " in text, text
    assert "NOT A COUNT: 1 of 8 answers" in text, "an errored answer is said, not hidden"


def test_rescore_reads_a_report_with_variants():
    tmp = tempfile.mkdtemp(prefix="rescore-")
    path = os.path.join(tmp, "night.json")
    with open(path, "w") as f:
        json.dump({"kind": "classifier", "config": {"thresholds": {"decider": 0.8}},
                   "set": {"name": "a", "n": 8},
                   "rows": {"today:a": ROWS, "sized:a": ROWS}}, f)
    got = conductor_eval.rescore(path)
    assert [(c["variant"], c["confident_right"], c["coverage"]) for c in got] == \
        [("today", 0.375, 0.375), ("sized", 0.375, 0.375)]


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:64s} ok")
            passed += 1
    print(f"{passed} passed")
