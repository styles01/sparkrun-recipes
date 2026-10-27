"""The joint decode graphs' cache (ENG-363, QA ENG-364, engine/models/graph_lru.py): at most `cap` graphs,
a replay makes its graph the newest, past the cap the least recently replayed is reset and dropped and
captured again when a round needs it; the report counts what /health shows. On fake graphs (the CUDA
graphs themselves: tests/test_nemotron_h_engine.py and tests/test_lfm2_moe_engine.py on the GPU).

Run: python tests/test_graph_lru.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.models.graph_lru import JOINT_GRAPHS_MAX, GraphLRU, joint_graphs  # noqa: E402


class FakeCUDAGraph:
    def __init__(self):
        self.resets = 0

    def reset(self):
        self.resets += 1


class FakeJoint:
    made = 0

    def __init__(self, key):
        self.key, self.graph = key, FakeCUDAGraph()
        FakeJoint.made += 1


def test_the_cache_keeps_at_most_cap_and_drops_the_least_recent():
    lru = GraphLRU(cap=3)
    keys = [(("s0", 4), ("s1", 4)), (("s0", 4), ("s2", 4)), (("s1", 4), ("s2", 4)), (("s0", 4), ("s1", 8))]
    got = {k: lru.take(k, lambda k=k: FakeJoint(k)) for k in keys[:3]}
    assert len(lru) == 3 and lru.captures == 3 and lru.evicted == 0
    assert lru.take(keys[0], lambda: FakeJoint("x")) is got[keys[0]]      # a replay: no capture, now newest
    g1 = got[keys[1]].graph
    lru.take(keys[3], lambda: FakeJoint(keys[3]))
    assert keys[1] not in lru and keys[0] in lru and len(lru) == 3        # the least recent went
    assert g1.resets == 1 and got[keys[1]].graph is None and lru.evicted == 1
    again = lru.take(keys[1], lambda: FakeJoint(keys[1]))                 # captured again on demand
    assert again is not got[keys[1]] and lru.captures == 5
    r = lru.report()
    assert r == {"n": 3, "max": 3, "captures": 5, "evicted": 2, "bytes": 0}, r
    return "cap 3: 5 captures, 2 dropped, the replayed one kept"


def test_four_slots_at_one_class_fit_the_default():
    """Every set of 2, 3 or 4 of 4 slots at one class is 11 graphs, under the default cap; 16 slots'
    sets would be 65,519, so the cap holds the cache whatever the slots and windows."""
    from itertools import combinations
    sets = [c for n in (2, 3, 4) for c in combinations(range(4), n)]
    assert len(sets) == 11 < JOINT_GRAPHS_MAX
    lru = GraphLRU()
    for _ in range(3):
        for c in sets:
            lru.take(tuple((f"s{i}", 4) for i in c), lambda: FakeJoint(c))
    assert lru.captures == 11 and lru.evicted == 0
    for c in combinations(range(16), 3):
        lru.take(tuple((f"s{i}", 4) for i in c), lambda: FakeJoint(c))
    assert len(lru) == JOINT_GRAPHS_MAX and lru.captures > 500 and lru.evicted == lru.captures - JOINT_GRAPHS_MAX


def test_one_cache_per_engine_and_a_bad_cap_is_refused():
    class Eng:
        pass
    e = Eng()
    assert joint_graphs(e) is joint_graphs(e) and isinstance(joint_graphs(e), GraphLRU)
    try:
        GraphLRU(cap=0)
        raise AssertionError("cap 0 accepted")
    except ValueError:
        pass


if __name__ == "__main__":
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for k, fn in fns:
        print(f"ok  {k}  {fn() or ''}")
    print(f"\n{len(fns)} passed")
