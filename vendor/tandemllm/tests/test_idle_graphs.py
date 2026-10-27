"""The server's idle graph captures (`graphs_once`, the capture gap of 2026-10-12): an engine that names
the graphs its context class lacks (`idle_captures`, Flash-Next's) gets them captured in the idle
window, one a step, under the engine lock, and a request that waits stops it between two.

Run: python tests/test_idle_graphs.py
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import test_forward_tree as T  # noqa: E402,F401  (sets the CPU environment before the engine imports)

from server import app  # noqa: E402


class _Eng:
    """Owes `n` graphs; each step takes one."""

    def __init__(self, n):
        self.owed, self.kv, self.locked = n, type("KV", (), {"length": 32_000})(), []

    def idle_captures(self):
        if not self.owed:
            return None
        return self._steps()

    def _steps(self):
        while self.owed:
            self.locked.append(app.LOCK.locked())
            self.owed -= 1
            yield


def _with(eng, idle_for=5.0, drafter=None):
    app.STATE["engine"], app.STATE["drafter"] = eng, drafter
    app.STATE["last_end_mono"] = time.monotonic() - idle_for
    app.STATE.pop("draining", None)


def test_an_idle_engine_captures_what_its_class_lacks_under_the_lock():
    eng = _Eng(3)
    _with(eng)
    assert app.graphs_once() == "captured" and eng.owed == 0 and eng.locked == [True] * 3
    assert not app.LOCK.locked()
    assert app.graphs_once() == "nothing"
    return "3 graphs under the engine lock, the lock released; then nothing"


def test_a_waiting_request_stops_it_and_a_recent_one_holds_it():
    eng = _Eng(5)
    _with(eng, idle_for=0.0)
    assert app.graphs_once() == "recent" and eng.owed == 5
    _with(eng)
    real, calls = app._waiting, {"n": 0}

    def waiting():
        calls["n"] += 1
        return calls["n"] >= 2                     # a request arrives after the second graph
    app._waiting = waiting
    try:
        assert app.graphs_once() == "stopped"
    finally:
        app._waiting = real
    assert eng.owed == 3 and not app.LOCK.locked()
    assert app.graphs_once() == "captured" and eng.owed == 0
    return "recent: none; a waiter stops it after 2 of 5, the next look takes the rest"


def test_a_length_the_drafter_takes_plain_captures_nothing():
    eng = _Eng(4)
    seen = []
    drafter = type("D", (), {"wants_graphs": lambda self, n: seen.append(n) or n < 100_000})()
    _with(eng, drafter=drafter)
    eng.kv.length = 260_800
    assert app.graphs_once() == "plain" and eng.owed == 4 and seen == [260_800]
    eng.kv.length = 32_000
    assert app.graphs_once() == "captured" and eng.owed == 0
    return "260,800 (plain turns there): nothing captured; 32,000: captured"


def test_an_engine_without_idle_captures_is_left_alone():
    _with(object())
    assert app.graphs_once() == "off"
    return "off for an engine that names no graphs (the 27B)"


if __name__ == "__main__":
    fails = 0
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        try:
            print(f"  {name:<70} ok   {fn()}")
        except Exception as exc:                                   # noqa: BLE001
            fails += 1
            import traceback
            traceback.print_exc()
            print(f"  {name:<70} FAIL ({exc})")
    print(f"{len(tests) - fails} passed" + (f", {fails} FAILED" if fails else ""))
    sys.exit(1 if fails else 0)
