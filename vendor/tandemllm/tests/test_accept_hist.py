"""CPU tests for tools/accept_hist.py and the histogram the length router writes for it.

The histogram travels as one token of a log line and is read back by a regex, so the one thing that
must hold is the round trip: what `LengthRouter.report()` writes is what `parse_log` reads, and the
cap rates are computed over the blocks that were actually counted.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.lenrouter import _hist_str  # noqa: E402
from tools.accept_hist import parse_hist, parse_log, per_request, summarize  # noqa: E402


def test_the_histogram_round_trips_through_the_log_token():
    h = {16: {16: 5, 4: 5}, 8: {2: 3, 8: 1}}
    tok = _hist_str(h)
    assert " " not in tok
    assert parse_hist(tok) == h
    assert parse_hist(_hist_str({})) == {}


def test_parse_log_reads_only_lines_that_carry_it():
    text = ("[req] something\n"
            "[drafter] lenrouter 8x0 (0.00 tok/block) widths {} commits - cap arm 0 depth 0\n"
            "[drafter] lenrouter 8x4 ... widths {8: 4} commits 8:2x3,8x1 cap arm 1 depth 2\n"
            "[drafter] an old line without the histogram\n")
    recs = parse_log(text)
    assert len(recs) == 2
    assert recs[1] == {"commits": {8: {2: 3, 8: 1}}, "cap_arm": 1, "cap_depth": 2}


def test_summarize_sums_requests_and_rates_over_counted_blocks():
    recs = [{"commits": {16: {16: 2, 4: 2}}, "cap_arm": 2, "cap_depth": 2},
            {"commits": {16: {16: 1}, 8: {3: 3}}, "cap_arm": 1, "cap_depth": 2}]
    s = summarize(recs)
    assert s["blocks"] == 8
    assert abs(s["mean"] - (16 * 3 + 4 * 2 + 3 * 3) / 8) < 1e-12
    assert s["arms"][16]["full"] == 3 and abs(s["arms"][16]["full_pct"] - 60.0) < 1e-9
    assert s["arms"][8]["full"] == 0
    assert abs(s["cap_arm_pct"] - 37.5) < 1e-9 and abs(s["cap_depth_pct"] - 50.0) < 1e-9


def test_summarize_of_nothing_is_zero_not_a_division_error():
    s = summarize([])
    assert s["blocks"] == 0 and s["mean"] == 0.0 and s["cap_arm_pct"] == 0.0


def test_the_server_s_own_warm_up_lines_do_not_shift_the_workloads():
    """The server runs requests of its own at startup, and each prints a [drafter] line. The
    first --serve run on 2026-09-23 counted from the front and was off by one; counting from the
    end maps the last n lines onto the n requests before the flush."""
    recs = [{"id": i} for i in range(8)]            # 2 startup lines, warm, 5 workloads (flush's line last)
    got = per_request(recs, 6)
    assert [r["id"] for r in got] == [2, 3, 4, 5, 6, 7]
    assert per_request(recs[:3], 6) is None


# ---------------------------------------------------------------- the per-slot curve

def test_the_curve_is_censored_by_the_depth_a_block_offered():
    """A 15-deep block that missed at slot 1, one that accepted 3 then missed, one that took all
    15; a 7-deep block that took all 7. Slot 8 rests only on the 15-deep blocks that got there."""
    from tools.rowlog import curve, first_miss
    acc = {15: {0: 1, 3: 1, 15: 1}, 7: {7: 1}}
    cv = {c["slot"]: c for c in curve(acc)}
    assert cv[1]["n"] == 4 and abs(cv[1]["rate"] - 3 / 4) < 1e-12
    assert cv[2]["n"] == 3 and cv[2]["rate"] == 1.0
    assert cv[4]["n"] == 3 and abs(cv[4]["rate"] - 2 / 3) < 1e-12      # the 3-then-miss block
    assert cv[8]["n"] == 1 and cv[8]["rate"] == 1.0                    # the 7-deep block is gone
    assert cv[15]["n"] == 1 and cv[15]["rate"] == 1.0
    assert first_miss(acc) == {1: 1, 4: 1, 8: 1, 16: 1}


def test_widths_8_and_16_and_tree_paths_go_in_one_curve():
    from tools.rowlog import curve
    # a width-8 chain (7 draft tokens) accepting 2; a width-16 chain accepting 0; a tree of depth
    # 5 whose accepted path is 4 long
    acc = {7: {2: 1}, 15: {0: 1}, 5: {4: 1}}
    cv = {c["slot"]: c for c in curve(acc)}
    assert cv[1]["n"] == 3 and abs(cv[1]["rate"] - 2 / 3) < 1e-12
    assert cv[3]["n"] == 2 and abs(cv[3]["rate"] - 0.5) < 1e-12
    assert cv[5]["n"] == 1 and cv[5]["rate"] == 0.0
    assert cv[6]["n"] == 0 and cv[6]["rate"] is None


def test_curves_by_workload_skip_the_warm_and_the_flush():
    from tools.accept_hist import curves_by
    reqs = [{"blocks": 3, "accept": {15: {0: 3}}}, {"blocks": 2, "accept": {15: {15: 2}}},
            {"blocks": 1, "accept": {7: {1: 1}}}, {"blocks": 1, "accept": {15: {0: 9}}}]
    by = curves_by(["warm", "prose", "code", "flush"], reqs)
    assert by["prose"] == {15: {15: 2}} and by["code"] == {7: {1: 1}}
    assert by["ALL"] == {15: {15: 2}, 7: {1: 1}}


def _main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    bad = 0
    for fn in fns:
        try:
            fn()
            print(f"  ok  {fn.__name__}")
        except Exception as exc:                                    # noqa: BLE001
            bad += 1
            import traceback
            traceback.print_exc()
            print(f"FAIL  {fn.__name__}: {exc}")
    print(f"\n{len(fns) - bad}/{len(fns)} passed")
    return 1 if bad else 0


def test_factors_by_pools_each_workload_and_skips_warm_and_flush():
    """The sampled bench reads both factors per workload off the [req] lines."""
    from tools.accept_hist import factors_by
    r = lambda c, b, ms: {"blocks": b, "committed": c, "decode_ms": ms,  # noqa: E731
                          "accept": {8: {c // b: b}}}
    names = ["warm", "prose", "prose", "code", "flush"]
    reqs = [r(10, 5, 500.0), r(20, 10, 1000.0), r(30, 10, 1000.0), r(60, 10, 1000.0),
            r(3, 1, 50.0)]
    out = factors_by(names, reqs)
    assert set(out) == {"prose", "code", "ALL"}
    assert out["prose"]["tok_blk"] == 50 / 20 and out["prose"]["ms_blk"] == 100.0
    assert out["prose"]["tok_s"] == 25.0 and out["prose"]["per_request_tok_blk"] == [2.0, 3.0]
    assert out["ALL"]["requests"] == 3 and out["ALL"]["tok_blk"] == 110 / 30
    return "per workload tokens a round, ms a round, tok/s; warm and flush left out"


if __name__ == "__main__":
    sys.exit(_main())
