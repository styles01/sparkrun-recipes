"""The forced walk of tools/forced_bench.py: a tree is accepted along the reference, not the argmax."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.tree import DraftTree  # noqa: E402


def _walk():
    from tools.forced_bench import _walk as w
    return w


def test_the_walk_follows_the_reference_down_the_tree():
    # anchor 5; children 6 and 7; 6 -> 8 -> 9; 7 -> 10
    t = DraftTree([5, 6, 8, 9, 7, 10], [-1, 0, 1, 2, 0, 4])
    ref = [1, 5, 6, 8, 11, 12]            # position 1 is the anchor
    assert _walk()(t, ref, 1) == [0, 1, 2]
    ref2 = [1, 5, 7, 10, 3]
    assert _walk()(t, ref2, 1) == [0, 4, 5]


def test_the_walk_stops_at_the_end_of_the_reference():
    t = DraftTree([5, 6, 8, 9], [-1, 0, 1, 2])
    assert _walk()(t, [1, 5, 6], 1) == [0, 1]
    assert _walk()(t, [1, 5, 3], 1) == [0]


if __name__ == "__main__":
    import traceback
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  ok  {name}")
            except Exception:
                fails += 1
                print(f"FAIL  {name}")
                traceback.print_exc()
    print(f"\n{fails} failed")
    sys.exit(1 if fails else 0)
