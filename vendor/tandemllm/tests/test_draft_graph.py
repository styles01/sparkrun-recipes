"""The draft graph's context classes, on the CPU.

the graph attended over a window padded to 2,048 and lost: on the row's ~500-token contexts the
padded attention cost more device time than the ~500 launches it saved. The retry gathers the next
power of two of the context, 512 up to the drafter's 2,048 window, one graph per class. What must hold
without a board: the class covers every position the eager call attends to (never fewer), it is the
smallest power of two that does, a context past the window stays in the window's class, and the graph
is only taken where its one body is valid (greedy, one block, the target's head, no sampler).

Run: python tests/test_draft_graph.py
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.drafters.draft_graph import DraftGraph  # noqa: E402


def _span(pos0: int, win: int = 2048) -> int:
    return DraftGraph.span(SimpleNamespace(win=win), pos0)


def test_the_class_is_the_next_power_of_two_of_the_context():
    assert _span(0) == 512 and _span(300) == 512 and _span(511) == 512
    assert _span(512) == 1024 and _span(900) == 1024 and _span(1023) == 1024
    assert _span(1024) == 2048 and _span(1100) == 2048


def test_the_class_always_covers_what_the_eager_call_attends_to():
    """The eager call attends to positions max(0, pos0 - win + 1) .. pos0; the graph gathers `span`
    positions from the same first one."""
    for win in (2048, 1024):
        for pos0 in range(0, 5000, 7):
            lo = max(0, pos0 - win + 1)
            n = pos0 - lo + 1
            s = _span(pos0, win)
            assert s >= n, (win, pos0, s, n)
            assert s <= win
            assert s == 512 or s // 2 < n, (win, pos0, s, n)       # the smallest that covers


def test_a_context_past_the_window_stays_in_the_window_class():
    assert _span(3000) == 2048 and _span(100_000) == 2048


def _drafter(**kw):
    cfg = SimpleNamespace(sliding_window=2048, layer_types=["sliding_attention"] * 5)
    d = SimpleNamespace(blocks=1, head=None, use_selector=True, path="greedy", cfg=cfg,
                        sampler=None)
    for k, v in kw.items():
        setattr(d, k, v)
    return d


def test_only_the_greedy_one_block_call_is_graphed():
    assert DraftGraph.eligible(_drafter())
    assert not DraftGraph.eligible(_drafter(blocks=2))
    assert not DraftGraph.eligible(_drafter(path="sampled"))
    assert not DraftGraph.eligible(_drafter(head=object()))
    assert not DraftGraph.eligible(_drafter(sampler=SimpleNamespace(on=True)))
    assert DraftGraph.eligible(_drafter(sampler=SimpleNamespace(on=False)))
    full = SimpleNamespace(sliding_window=2048, layer_types=["full_attention"] * 5)
    assert not DraftGraph.eligible(_drafter(cfg=full))


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:64s} ok")
            passed += 1
    print(f"{passed} passed")
