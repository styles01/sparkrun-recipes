"""`tools/prefill_crossover.py`'s verdict: the row count from which unpack wins for good.

Run: python tests/test_prefill_crossover.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.prefill_crossover import crossover  # noqa: E402


def test_the_crossover_is_where_unpack_wins_from_then_on():
    t = {512: {"v2": 1.0, "unpack": 1.5}, 1024: {"v2": 2.0, "unpack": 2.2},
         2048: {"v2": 4.1, "unpack": 4.0}, 8192: {"v2": 20.0, "unpack": 15.0}}
    assert crossover(t) == 2048


def test_a_single_win_below_a_loss_is_not_the_crossover():
    t = {512: {"v2": 1.0, "unpack": 0.9}, 1024: {"v2": 2.0, "unpack": 2.2},
         2048: {"v2": 4.1, "unpack": 4.0}}
    assert crossover(t) == 2048


def test_v2_winning_at_the_top_means_no_crossover():
    t = {512: {"v2": 1.0, "unpack": 1.5}, 8192: {"v2": 10.0, "unpack": 12.0}}
    assert crossover(t) is None


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:64s} ok")
            passed += 1
    print(f"{passed} passed")
