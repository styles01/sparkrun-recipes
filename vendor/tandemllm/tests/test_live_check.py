"""tools/live_check.py sends text no run has sent before (OPS-63 follow-up, 2026-10-01).

On a 0.4.0 test build, "a live decode rate from token 50 on" failed: the persistent suffix store had
learned the four fixed prompts' answers from the day's earlier runs, so every request decoded at
about 186 tok/s and ended in about 1.3 s, too short for a "now" rate. The prompts are now drawn
per run from a seeded generator with the run's nonce in the text and in the asked-for answer.

Run: python tests/test_live_check.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from tools import live_check as lc  # noqa: E402

# the four prompts every run sent until 2026-10-01
OLD = [
    "Explain how speculative decoding verifies a block of drafted tokens, in about 200 words.",
    "Write a short story of about 250 words about a lighthouse keeper who counts ships.",
    "List twelve rivers of Europe with their length in kilometres and one city on each.",
    "Describe the difference between prefill and decode in an inference engine, 150 words.",
]


def test_fresh_prompts_differ_within_a_run_and_between_runs():
    a = lc.fresh_prompts(123456789)
    assert len(a) == 4 and len(set(a)) == 4
    assert not set(a) & set(OLD)
    assert lc.fresh_prompts(123456789) == a                     # the seed repeats a run
    # the nonce is in every prompt, and the answer is asked to repeat it
    assert all("56789-" in p and "first sentence and again in the last" in p for p in a)
    seen = set()
    for seed in range(1000, 1300):
        ps = lc.fresh_prompts(seed * 7919, 6)
        assert len(set(ps)) == 6
        assert not seen & set(ps), seed                         # no prompt twice across 300 runs
        seen |= set(ps)
    # the story itself varies, not only the number: 300 runs draw many different openings
    openings = {p.split(". The ")[0] for p in seen}
    assert len(openings) > 1000, len(openings)


def _fake_server(sent):
    def chat(base, model, prompt, max_tokens, think):
        sent.append(prompt)
        return f"chatcmpl-{len(sent):024d}", {"predicted_ms": 100}, {}, 0.1

    def snapshot(base, token):
        return {"requests": [], "counts": {"in_flight": 0, "served": 0},
                "now": {"decode_tps": None}, "windows": {}}
    return chat, snapshot


def test_every_run_sends_new_text_and_records_it():
    real = lc.chat, lc.snapshot, lc._composite
    runs = []
    try:
        for argv in ([], [], ["--seed", "42"], ["--seed", "42"]):
            sent: list[str] = []
            lc.chat, lc.snapshot = _fake_server(sent)
            lc._composite = lambda base: False
            with tempfile.TemporaryDirectory() as tmp:
                out = os.path.join(tmp, "live.json")
                lc.main(["--token", "t", "--pairs", "2", "--period", "5", "--out", out] + argv)
                rep = json.load(open(out))
            assert rep["prompts"] == sent and len(sent) == 4 and len(set(sent)) == 4
            assert not set(sent) & set(OLD)
            runs.append((rep["seed"], sent))
    finally:
        lc.chat, lc.snapshot, lc._composite = real
    (s1, p1), (s2, p2), (s3, p3), (s4, p4) = runs
    assert s1 != s2 and not set(p1) & set(p2)                  # two plain runs share no prompt
    assert s3 == s4 == 42 and p3 == p4 == lc.fresh_prompts(42)  # --seed repeats a run


if __name__ == "__main__":
    passed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:64s} ok")
            passed += 1
    print(f"{passed} passed")
