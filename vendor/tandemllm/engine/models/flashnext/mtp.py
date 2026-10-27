"""Flash-Next's MTP layer as a draft head (ENG-153, ENG-245): what the shared MTP drafter
(engine/methods/mtp_drafter.py) asks of a model's head, over `FlashNextEngine`'s MTP forward.

The head keeps its own cache (the MTP layer's KV and state, `FlashNextEngine.new_mtp_state`). A
round's draft -- the feed of the rows committed since the last one, `depth` - 1 deeper steps, every
head and its top-2 -- is one replay of a captured graph for that feed's row count and context class
(`MTPChainGraph`); the host reads the drafts once.
"""

from __future__ import annotations

import time

import torch

from engine.models.flashnext.engine import FlashNextEngine, MTPChainGraph
from engine.models.flashnext.ref import MTP


class FlashNextMTPHead:
    def __init__(self, fn: FlashNextEngine, depth: int, cap: int, pages=None):
        self.fn, self.depth = fn, depth
        self.st = fn.new_mtp_state(cap, pages=pages)
        self.graphs: dict = {}
        self.captures = 0               # graphs captured so far (a round that captured is not a price)
        self.fresh: set = set()         # graphs captured ahead (`capture_class`) not replayed yet
        # its own chain buffers and staging (FlashNextEngine `_mtp_key`): the engine's first head 0
        self.key = getattr(fn, "_mtp_heads", 0)
        fn._mtp_heads = self.key + 1
        self._n = 0

    def reset(self) -> None:
        self.st.length = 0
        self.st.pool_n = {}

    def views(self) -> list:
        """The head's position-indexed cache as (tensor, token axis): the MTP layer's K, V and indexer
        keys (the caches keep its rows with the trunk's, ENG-157)."""
        st = self.st
        return [(st.k[MTP], 1), (st.v[MTP], 1)] + ([(st.ik[MTP], 0)] if MTP in st.ik else [])

    def resume(self, n: int) -> None:
        """The head's cache is valid for its first `n` rows (a cache wrote them back, or they are in
        place): the length, and the pooled indexer keys built again from those rows."""
        self.st.length = int(n)
        self.st.pool_n = {}
        if n > 0:
            self.fn.build_pool(self.st, layers=[MTP])

    def prime(self, rows: torch.Tensor, nexts: list[int]) -> None:
        """Rows [n, hc H] and the token after each into the head's cache, in one call (a backlog)."""
        self.fn.mtp_prime(rows, nexts, self.st)

    def capture_all(self) -> None:
        """The graphs of every feed a round can bring (1 .. depth + 1 rows) at the current context class."""
        self.fn._mtp_key = self.key
        cls = self.fn.qsa_class(self.st.length + self.depth + 1)
        for n in range(1, self.depth + 2):
            if (n, cls) not in self.graphs:
                self.graphs[(n, cls)] = MTPChainGraph(self.fn, self.st, n, self.depth, cls)
                self.graphs[(n, cls)].capture()
                self.captures += 1

    def capture_class(self, n: int, cls: int) -> None:
        """The graph of an n-row feed at context class `cls`, ahead of the round that needs it (the
        server's idle captures, engine/models/flashnext/serving.py); the cache as it was."""
        self.fn._mtp_key = self.key
        if (n, cls) not in self.graphs:
            g = MTPChainGraph(self.fn, self.st, n, self.depth, cls)
            g.capture()
            self.graphs[(n, cls)] = g
            self.fresh.add((n, cls))

    def launch(self, rows: torch.Tensor, nexts: list[int]) -> None:
        """Feed rows [n, hc H] with their next tokens and draft `depth` tokens from the last: the graph
        is launched, not waited for (`read`)."""
        self.fn._mtp_key = self.key
        fn = self.fn
        n = len(nexts)
        cls = fn.qsa_class(self.st.length + n + self.depth - 1)
        g = self.graphs.get((n, cls))
        if (n, cls) in self.fresh:
            self.fresh.discard((n, cls))
            self.captures += 1          # an idle capture's first replay uploads it: the drafter skips it
        if g is None:
            g = self.graphs[(n, cls)] = MTPChainGraph(fn, self.st, n, self.depth, cls)
            g.capture()
            self.captures += 1          # this round's draft time is the capture's (the drafter skips it)
        b = fn._mtp_bufs(self.depth)
        b["stack"][:n].copy_(rows)
        t0 = time.perf_counter()
        assert fn.stage_mtp(self.st, nexts, self.depth) == cls
        fn.host_ms["stage_mtp"] = fn.host_ms.get("stage_mtp", 0.0) + 1e3 * (time.perf_counter() - t0)
        g.graph.replay()
        self._n = n

    def read(self) -> tuple[list, list]:
        """After `launch`: per depth the (first, second) token and their probabilities; the fed rows
        join the head's cache."""
        self.fn._mtp_key = self.key
        torch.cuda.current_stream().synchronize()
        self.st.length += self._n
        b = self.fn._mtp_bufs(self.depth)
        return b["tok_h"][:self.depth].tolist(), b["prob_h"][:self.depth].tolist()
