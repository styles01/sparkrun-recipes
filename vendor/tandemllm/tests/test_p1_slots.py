"""CPU tests for tools/p1_slots.py, the go/no-go arithmetic.

The ceiling is only worth quoting if the loop walk is exact: a drafter right with probability q at
every position must give the renewal numbers a pen-and-paper calculation gives, the walk must agree
with a sampled loop on an irregular text, and the projection must leave the curve alone at share 0
and touch only the slots it was told to.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.p1_slots import (breakeven, combine, curve_from_report, hist_curve,  # noqa: E402
                            loop_ceiling, project, row_hists, tokens_per_round)


def test_a_drafter_that_is_always_right_fills_every_block():
    c = combine([loop_ceiling(np.ones(16 * 400 + 1), 15)])
    assert abs(c["tokens_per_round"] - 16.0) < 1e-9, c["tokens_per_round"]
    assert np.allclose(c["rate"], 1.0)


def test_a_constant_confidence_gives_the_renewal_numbers():
    p, w = 0.8, 15
    c = combine([loop_ceiling(np.full(200_000, p), w)])
    want = 1 + sum(p ** k for k in range(1, w + 1))
    assert abs(c["tokens_per_round"] - want) / want < 1e-3, (c["tokens_per_round"], want)
    assert np.allclose(c["rate"], p, atol=1e-6), "each slot, given the ones before, is p"


def test_the_walk_is_the_sampled_loop_on_an_irregular_text():
    rng = np.random.default_rng(7)
    q = rng.beta(2.0, 0.6, size=600)            # confident mostly, with hard spots
    w = 7
    exact = loop_ceiling(q, w)
    off = np.zeros(w); acc = np.zeros(w); rounds = 0
    reps = 4000
    for _ in range(reps):
        a = 0
        while a < len(q) - 1:
            rounds += 1
            for i in range(1, w + 1):
                if a + i >= len(q):
                    a = len(q)
                    break
                off[i - 1] += 1
                if rng.random() < q[a + i]:
                    acc[i - 1] += 1
                    if i == w:
                        a = a + w + 1
                else:
                    a = a + i
                    break
    assert abs(rounds / reps - exact["rounds"]) / exact["rounds"] < 0.01
    assert np.allclose(off / reps, exact["offered"], rtol=0.03)
    assert np.allclose(acc / reps, exact["accepted"], rtol=0.03)


def test_tokens_a_round_of_a_curve_and_the_projection():
    live = [0.8, 0.5, 0.5]
    assert abs(tokens_per_round(live) - (1 + 0.8 + 0.4 + 0.2)) < 1e-12
    assert tokens_per_round([0.8, float("nan"), 0.5]) == 1.8, "a missing slot ends the curve"
    ceil = [0.9, 0.9, 0.9]
    assert project(live, ceil, 0.0) == live
    assert np.allclose(project(live, ceil, 0.5), [0.85, 0.7, 0.7])
    assert np.allclose(project(live, ceil, 1.0, slots=range(2, 3)), [0.8, 0.9, 0.5])
    assert project([0.95], [0.9], 1.0) == [0.95], "a slot already above the ceiling is left alone"


def test_the_breakeven_share_reaches_the_target_and_none_when_the_gap_cannot():
    live, ceil = [0.7] * 8, [0.8] * 8
    s = breakeven(live, ceil, 0.05)
    t = tokens_per_round(project(live, ceil, s)) / tokens_per_round(live) - 1
    assert 0 < s < 1 and abs(t - 0.05) < 1e-6
    assert breakeven([0.7] * 8, [0.71] * 8, 0.05) is None


def test_the_row_curve_sums_the_reports_by_their_counts():
    d = tempfile.mkdtemp()
    paths = []
    for i, (r1, n1) in enumerate([(0.8, 100), (0.6, 300)]):
        p = os.path.join(d, f"r{i}.json")
        json.dump({"accept_curve": [{"slot": 1, "rate": r1, "n": n1},
                                    {"slot": 2, "rate": 0.5, "n": 10},
                                    {"slot": 3, "rate": None, "n": 0}]}, open(p, "w"))
        paths.append(p)
    live, n = curve_from_report(paths)
    assert np.allclose(live, [0.65, 0.5]) and n == 400


def _req(cid, prompt, accept):
    return (f"[req] chatcmpl-{cid} json prompt={prompt} completion=9 finish=length 100 ms 30.00 tok/s "
            f"blocks=2 committed=9 decode_ms=80.0 accept={accept}\n")


def test_the_row_log_maps_requests_to_prompts_past_the_warmups_and_checks_the_lengths():
    plens = [30, 40]
    run = (_req("w0", 30, "15:0x2") + _req("w1", 40, "15:0x2") + _req("r0", 30, "15:3x2")
           + _req("r1", 40, "15:1x1,0x1"))
    d = tempfile.mkdtemp()
    log = os.path.join(d, "server.log")
    open(log, "w").write(run + run)
    per = row_hists([log], plens, warm=2)
    assert per[0] == {15: {3: 4}} and per[1] == {15: {1: 2, 0: 2}}, per
    live, blocks = hist_curve(per[:1])
    assert blocks == 4 and np.allclose(live[:3], 1.0) and live[3] == 0.0
    open(log, "w").write(run.replace("prompt=40", "prompt=43"))
    assert row_hists([log], plens, warm=2)[1] == {15: {1: 1, 0: 1}}, "filler: a few tokens apart"
    open(log, "w").write(run.replace("prompt=40", "prompt=90"))
    try:
        row_hists([log], plens, warm=2)
        raise AssertionError("a length that does not match must stop the projection")
    except SystemExit as e:
        assert "prompt tokens" in str(e)


def _lattice(chains: list[list[int]], k: int = 3):
    """Lattices whose greedy chain is `chains[i]`: candidate 0 of every slot, scored highest."""
    n, l = len(chains), len(chains[0])
    cand = np.full((n, l, k), 7, dtype=np.int32)
    scores = np.zeros((n, l, k, k), dtype=np.float32)
    for i, ch in enumerate(chains):
        cand[i, :, 0] = ch
        scores[i, :, :, 0] = 1.0
    return cand, scores


def test_chain_walk_is_the_loop():
    from tools.p1_slots import chain_stats, chain_walk
    target = list(range(100, 112))                # the continuation, 12 tokens
    L = 3
    # anchor 0 drafts 101,102,999 (2 accepted) -> anchor 3 drafts 104,105,106 (3) -> anchor 7
    # drafts 108,109,110 (3) -> anchor 11 is the last token: nothing left to draft
    chains = [[0] * L for _ in target]
    chains[0] = [101, 102, 999]
    chains[3] = [104, 105, 106]
    chains[7] = [108, 109, 110]
    cand, scores = _lattice(chains)
    index = {p: p for p in range(len(target))}
    blocks = chain_walk(target, cand, scores, index)
    assert blocks == [(2, 3, 11), (3, 3, 8), (3, 3, 4)]
    st = chain_stats(blocks, L)
    assert st["blocks"] == 3 and abs(st["tokens_per_round"] - (3 + 4 + 4) / 3) < 1e-9
    assert st["rate"] == [1.0, 1.0, 2 / 3]


def test_chain_walk_stops_at_the_text_end():
    from tools.p1_slots import chain_stats, chain_walk
    target = [1, 2, 3, 4]
    chains = [[2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7]]
    cand, scores = _lattice(chains)
    blocks = chain_walk(target, cand, scores, {p: p for p in range(4)})
    assert blocks == [(3, 3, 3)]                   # every slot right, and no bonus past the end
    st = chain_stats(blocks, 3)
    assert st["tokens_per_round"] == 3.0 and st["rate"] == [1.0, 1.0, 1.0]


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


if __name__ == "__main__":
    sys.exit(_main())
