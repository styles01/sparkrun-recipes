"""CPU test for tools/bench_lenrouter.py's Phase 2 labels: a '+'-joined label is the served router
with exactly those knobs, and everything it does not name stays at the served value."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_a_label_is_its_knobs_and_the_rest_is_served():
    for k in ("QWEN38_DF2_TREE_MODE", "QWEN38_DF2_TEMP", "QWEN38_TREE_NODES",
              "QWEN38_TREE_NODES_NARROW", "QWEN38_TREE_WIDE_AFTER"):
        os.environ.pop(k, None)
    from tools.bench_lenrouter import phase2_knobs
    served = {"deep": 0, "wide": 16, "narrow": 8, "after": 0, "mode": "paths", "temp": 1.0}
    assert phase2_knobs("router") == served
    assert phase2_knobs("n16+nodes+t14") == dict(served, narrow=16, mode="nodes", temp=1.4)
    assert phase2_knobs("w32+deep") == dict(served, wide=32, deep=32)
    assert phase2_knobs("w24+n12+paths+t07") == dict(served, wide=24, narrow=12, temp=0.7)
    assert phase2_knobs("w24+a32") == dict(served, wide=24, after=32)
    os.environ["QWEN38_DF2_TREE_MODE"] = "nodes"
    os.environ["QWEN38_TREE_NODES_NARROW"] = "16"
    try:
        assert phase2_knobs("router")["mode"] == "nodes", "the served mode is the environment's"
        assert phase2_knobs("router")["narrow"] == 16, "so is the served budget"
        assert phase2_knobs("deep")["narrow"] == 16 and phase2_knobs("n8")["narrow"] == 8
    finally:
        os.environ.pop("QWEN38_DF2_TREE_MODE")
        os.environ.pop("QWEN38_TREE_NODES_NARROW")


if __name__ == "__main__":
    test_a_label_is_its_knobs_and_the_rest_is_served()
    print("  ok  test_a_label_is_its_knobs_and_the_rest_is_served\n\n1/1 passed")
