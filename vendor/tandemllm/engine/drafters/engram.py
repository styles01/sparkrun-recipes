"""A suffix memory over the tokens this sequence has already seen.

The idea is the oldest one in text compression: if the last n tokens have occurred before, the
tokens that followed them last time are a good guess for what follows now. It reads no weights at
all, which on a bandwidth-bound board makes it the only drafter whose proposal is genuinely free,
and it proposes long blocks exactly where a language model is most predictable -- quoted text, an
edit of something already in the context, repeated identifiers in code, structured output that
repeats a schema.

Where it proposes nothing, it costs a dictionary lookup. That asymmetry is what makes it worth
having even though its acceptance on fresh prose is near zero: it is not competing with a neural
drafter, it is preceding one.

Index: for each order n in `orders`, a map from an n-gram to the position after its most recent
occurrence. Longest order wins; the most recent occurrence wins within an order. Both are one
dictionary write per order per token, so keeping it current costs O(len(orders)) per accepted token.
"""

from __future__ import annotations

from . import Drafter


class EngramDrafter(Drafter):
    name = "engram"

    def __init__(self, orders: tuple[int, ...] = (8, 6, 5, 4, 3, 2), min_order: int = 3):
        self.orders = tuple(sorted(orders, reverse=True))
        self.min_order = min_order
        self.tokens: list[int] = []
        self.index: dict[int, dict[tuple, int]] = {n: {} for n in self.orders}
        self.stats = {"calls": 0, "hits": 0, "proposed": 0, "order_hist": {}}
        self.last_order = 0  # the order of the match the most recent proposal came from

    def reset(self) -> None:
        self.tokens = []
        self.index = {n: {} for n in self.orders}

    def prime(self, tokens: list[int]) -> None:
        """Index a prompt in one go, before the first step."""
        self.tokens = list(tokens)
        for n in self.orders:
            m = self.index[n]
            t = self.tokens
            for i in range(n, len(t)):
                m[tuple(t[i - n:i])] = i

    def observe(self, tokens: list[int]) -> None:
        t = self.tokens
        for tokid in tokens:
            t.append(tokid)
            i = len(t)
            for n in self.orders:
                if i > n:
                    self.index[n][tuple(t[i - 1 - n:i - 1])] = i - 1

    def propose(self, context: list[int], k: int) -> list[int]:
        self.stats["calls"] += 1
        t = self.tokens
        if len(t) < self.min_order or k <= 0:
            return []
        for n in self.orders:
            if n < self.min_order or len(t) < n:
                continue
            hit = self.index[n].get(tuple(t[-n:]))
            if hit is None or hit >= len(t):
                continue
            draft = t[hit:hit + k]
            if not draft:
                continue
            self.stats["hits"] += 1
            self.stats["proposed"] += len(draft)
            self.stats["order_hist"][n] = self.stats["order_hist"].get(n, 0) + 1
            self.last_order = n
            return list(draft)
        self.last_order = 0
        return []
