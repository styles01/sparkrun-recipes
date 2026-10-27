"""CPU tests for tools/sampled_ab.py (sampled bench read back, round by round).

Run: python tests/test_sampled_ab.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.sampled_ab import compare, stratify  # noqa: E402


def _round(tok, ms, reqs):
    return {"prose": {"tok_blk": tok, "ms_blk": ms, "tok_s": 1e3 * tok / ms, "per_request_tok_blk": reqs}}


def test_resolved_only_when_every_round_agrees_and_the_requests_say_so():
    base = [_round(2.0, 90.0, [1.9, 2.0, 2.1]), _round(2.1, 91.0, [2.0, 2.1, 2.2])]
    better = [_round(2.5, 92.0, [2.4, 2.5, 2.6]), _round(2.6, 93.0, [2.5, 2.6, 2.7])]
    c = compare(base, better, "prose")
    assert c["tok_blk"]["resolved"] and c["tok_blk"]["z"] > 2 and c["tok_blk"]["same_sign"]
    assert all(d > 0 for d in c["ms_blk"]["delta_pct"])
    mixed = [_round(2.5, 90.0, [2.4, 2.5, 2.6]), _round(2.0, 90.0, [1.9, 2.0, 2.1])]
    assert not compare(base, mixed, "prose")["tok_blk"]["resolved"]          # one round the other way
    noisy = [_round(2.05, 90.0, [1.0, 2.0, 3.2]), _round(2.15, 90.0, [1.1, 2.2, 3.1])]
    assert not compare(base, noisy, "prose")["tok_blk"]["resolved"]          # same sign, inside the spread
    return "every round the same sign AND |z| > 2 over the requests; one or the other alone is not enough"


def _two(tok_a, reqs_a, tok_b, reqs_b):
    return {"prose": {"tok_blk": tok_a, "ms_blk": 90.0, "tok_s": 1e3 * tok_a / 90.0, "per_request_tok_blk": reqs_a},
            "edit": {"tok_blk": tok_b, "ms_blk": 90.0, "tok_s": 1e3 * tok_b / 90.0, "per_request_tok_blk": reqs_b},
            "ALL": {"tok_blk": (tok_a + tok_b) / 2, "ms_blk": 90.0, "tok_s": 1e3 * (tok_a + tok_b) / 180.0,
                    "per_request_tok_blk": reqs_a + reqs_b}}


def test_all_is_judged_within_the_workloads_not_across_them():
    """Two workloads 10 tokens a round apart, each +0.5 with a tight spread: pooled, the 10-token gap between the
    workloads swamps the gain (|z| < 2); stratified, it is resolved. A gain inside each workload's spread stays
    unresolved either way."""
    base = [_two(2.0, [1.9, 2.0, 2.1], 12.0, [11.9, 12.0, 12.1])] * 2
    other = [_two(2.5, [2.4, 2.5, 2.6], 12.5, [12.4, 12.5, 12.6])] * 2
    parts = [compare(base, other, w) for w in ("prose", "edit")]
    pooled = compare(base, other, "ALL")
    assert not pooled["tok_blk"]["resolved"] and abs(pooled["tok_blk"]["z"]) < 2
    s = stratify(pooled, parts)
    assert s["tok_blk"]["resolved"] and s["tok_blk"]["z"] > 2 and s["tok_blk"]["stratified"]
    noisy = [_two(2.1, [1.0, 2.0, 3.3], 12.1, [11.0, 12.0, 13.3])] * 2
    parts = [compare(base, noisy, w) for w in ("prose", "edit")]
    assert not stratify(compare(base, noisy, "ALL"), parts)["tok_blk"]["resolved"]
    return f"pooled z {pooled['tok_blk']['z']:+.1f} (n.r.), stratified z {s['tok_blk']['z']:+.1f} (resolved)"


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"  {name:62s} ok   {fn() or ''}")
            passed += 1
    print(f"{passed} passed")
