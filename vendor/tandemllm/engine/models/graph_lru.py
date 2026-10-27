"""The joint decode graphs a model keeps (ENG-363): at most `cap`, the least recently replayed dropped first.

A joint decode step (`decode_multi` of engine/models/nemotron_h and engine/models/lfm2_moe) is captured once
for each set of decoding slots and their attention classes, keyed by the tuple of (slot, class). The sets
grow with the slots (every subset that decodes together, in the order the round lists them) and the
classes with the window, and every graph holds its private pool: kept for good they had no bound. Here
the graphs live in insertion order, a replay moves its graph to the newest end, and past `cap` the oldest
is reset (its pool goes back to the allocator); the next round that needs it captures it again.

    graphs = GraphLRU()
    g = graphs.take(key, make)      # make() returns a captured graph object with a `.graph` CUDAGraph
    graphs.report()                 # {"n", "max", "captures", "evicted", "bytes"} for /health
"""

from __future__ import annotations

import torch

#: graphs kept, a safe bound and not a measurement: 4 slots at one class are 11 sets (any 2, 3 or 4 of them)
JOINT_GRAPHS_MAX = 32


def _reserved() -> int:
    return torch.cuda.memory_reserved() if torch.cuda.is_available() else 0


class GraphLRU:
    def __init__(self, cap: int = JOINT_GRAPHS_MAX):
        if cap < 1:
            raise ValueError(f"a graph cache of {cap}: at least 1")
        self.cap = int(cap)
        self.items: dict = {}
        self.nbytes: dict = {}
        self.captures = 0
        self.evicted = 0

    def __len__(self) -> int:
        return len(self.items)

    def __contains__(self, key) -> bool:
        return key in self.items

    def take(self, key, make):
        """The graph of `key`, now the most recent; captured by `make()` when it is not kept. The
        allocator's reserve growth over the capture is the graph's cost in `report()`."""
        g = self.items.pop(key, None)
        if g is None:
            r0 = _reserved()
            g = make()
            self.nbytes[key] = max(0, _reserved() - r0)
            self.captures += 1
        self.items[key] = g
        while len(self.items) > self.cap:
            old_key = next(iter(self.items))
            old = self.items.pop(old_key)
            self.nbytes.pop(old_key, None)
            gr = getattr(old, "graph", None)
            if gr is not None:
                gr.reset()
                old.graph = None
            self.evicted += 1
        return g

    def report(self) -> dict:
        return {"n": len(self.items), "max": self.cap, "captures": self.captures, "evicted": self.evicted,
                "bytes": sum(self.nbytes.values())}


def joint_graphs(fn) -> GraphLRU:
    """The joint decode graphs of an engine over shared weights (one cache for all its slots)."""
    return fn.__dict__.setdefault("joint_graphs", GraphLRU())
