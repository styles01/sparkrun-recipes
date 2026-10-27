"""tools/lane_check.py's verdict under the near-tie rule (docs/exactness.md "Two runs
of the same request"): two runs agree when their tokens are the same up to the first fork and the
fork is a near tie (each run took the other's runner-up, the top-2 gap within --tie-nats in both
rows). No server, no torch.

Run: python tests/test_lane_check.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.lane_check import fork  # noqa: E402


def e(tok, lp, *alts):
    """One streamed logprobs entry: the token taken, its logprob, the top alternatives."""
    tops = [(tok, lp), *alts]
    return {"token": tok, "logprob": lp, "bytes": list(tok.encode()),
            "top_logprobs": [{"token": t, "logprob": v, "bytes": list(t.encode())}
                             for t, v in sorted(tops, key=lambda x: -x[1])]}


A = [e("The", -0.01, ("A", -5.0)), e(" cache", -0.7, (" store", -0.72)), e(" works", -0.1)]


def test_same_tokens():
    f = fork(A, list(A), 0.125)
    assert f["agrees"] and f["at"] is None, f


def test_fork_at_a_near_tie():
    b = [A[0], e(" store", -0.69, (" cache", -0.71)), e(" holds", -0.2)]
    f = fork(A, b, 0.125)
    assert f["agrees"] and f["near_tie"] and f["at"] == 1 and f["gap_nats"] == [0.02, 0.02], f


def test_fork_with_a_wide_gap_fails():
    a = [A[0], e(" cache", -0.05, (" store", -3.0))]
    b = [A[0], e(" store", -0.05, (" cache", -3.0))]
    f = fork(a, b, 0.125)
    assert not f["agrees"] and f["near_tie"] is False and f["gap_nats"] == [2.95, 2.95], f


def test_fork_to_a_third_token_fails():
    b = [A[0], e(" table", -0.69, (" cache", -0.71))]
    f = fork(A, b, 0.125)
    assert not f["agrees"] and not f["crossed"], f


def test_shorter_run_with_the_same_tokens():
    f = fork(A, A[:2], 0.125)
    assert f["agrees"] and f["at"] == 2, f


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:60s} ok")
            passed += 1
    print(f"{passed} passed")
