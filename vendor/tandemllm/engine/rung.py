"""The Rung (ENG-209 step 1, the CPU prototype): one step of StairCut's staircase built from the block
drafter's lattice as fixed-size arrays, the form a single-CTA fill kernel would produce.

A Rung of size T (16, 32 or 64 rows) is T token rows in DFS pre-order, row 0 the anchor, plus the shape
the verify reads from it (ancestor mask, convolution windows, depths). Its rows are the first T-1 nodes
`engine.tree.lattice_tree` inserts at a budget of 63: the greedy path first, then best-first by path
log-probability. That builder is prefix-consistent -- its heap and tie counter evolve the same whatever
the budget, and it stops only on the count -- so the first 15, 31 and 63 inserted nodes ARE its trees at
budgets 15, 31 and 63, and the three Rungs nest as sets. One fill therefore yields every size, and the
cut picks one.

Everything here is arrays and integer indices (no Python objects per node) so that the fill kernel can
be checked against it bit for bit; the reference it is checked against is the host builder itself
(`tests/test_rung.py`, `tools/rung_check.py` over recorded lattices).
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

import numpy as np

SIZES = (16, 32, 64)
NMAX = SIZES[-1] - 1          # drafted nodes in the largest Rung


@dataclass
class Fill:
    """The fill's insertion order: node i (0-based, the anchor excluded) has token `tok[i]`, parent
    `par[i]` in the same numbering (-1 = the anchor), path log-probability `lp[i]`, lattice slot
    `slot[i]` and candidate index `ci[i]`. `n` nodes are valid."""
    anchor: int
    tok: np.ndarray
    par: np.ndarray
    lp: np.ndarray
    slot: np.ndarray
    ci: np.ndarray
    n: int


def fill(anchor: int, cand, logp, greedy, nmax: int = NMAX) -> Fill:
    """`lattice_tree(anchor, cand, logp, greedy, nmax)`'s nodes in the order it inserts them.

    The same arithmetic as the host builder, operation for operation (Python floats, the heap keyed by
    (-logp, tie counter)), so that the order and every tie resolve identically."""
    L = len(cand)
    k = len(cand[0]) if L else 0
    tok = np.zeros(nmax, np.int64)
    par = np.full(nmax, -1, np.int64)
    lp_a = np.zeros(nmax, np.float64)
    slot_a = np.zeros(nmax, np.int64)
    ci_a = np.zeros(nmax, np.int64)
    heap: list[tuple] = []
    tie = 0
    n = 0

    def add(p: int, s: int, c: int, lp: float) -> int:
        nonlocal n
        tok[n], par[n], lp_a[n], slot_a[n], ci_a[n] = int(cand[s][c]), p, lp, s, c
        n += 1
        return n - 1

    row, parent, lp = 0, -1, 0.0
    for s, c in enumerate(list(greedy)[:L]):
        if n >= nmax:
            break
        base = lp
        lp += logp[s][row][c]
        here = parent
        parent = add(here, s, c, lp)
        for c2 in range(k):
            if c2 != c:
                tie += 1
                heapq.heappush(heap, (-(base + logp[s][row][c2]), tie, here, s, c2))
        row = c
    while heap and n < nmax:
        negp, _, p, s, c = heapq.heappop(heap)
        lp = -negp
        nid = add(p, s, c, lp)
        if s + 1 < L:
            for c2 in range(k):
                tie += 1
                heapq.heappush(heap, (-(lp + logp[s + 1][c][c2]), tie, nid, s + 1, c2))
    return Fill(anchor, tok, par, lp_a, slot_a, ci_a, n)


@dataclass
class Rung:
    """One cut of a fill, re-indexed to DFS pre-order. Row 0 is the anchor. `order[r]` is the fill
    index of row r (-1 for the anchor)."""
    tokens: np.ndarray          # [T]
    parents: np.ndarray         # [T], -1 for row 0
    scores: np.ndarray          # [T] exp(path log-probability), 1.0 for the anchor
    depths: np.ndarray          # [T]
    order: np.ndarray           # [T]

    @property
    def T(self) -> int:
        return int(self.tokens.shape[0])

    def ancestor_mask(self) -> np.ndarray:
        """[T, T] bool, m[i, j] = j is i or an ancestor of i."""
        T = self.T
        m = np.zeros((T, T), bool)
        for i in range(T):
            j = i
            while j >= 0:
                m[i, j] = True
                j = int(self.parents[j])
        return m

    def conv_windows(self, width: int) -> np.ndarray:
        """`DraftTree.conv_windows` as an int array [T, width]."""
        out = np.zeros((self.T, width), np.int64)
        for i in range(self.T):
            chain: list[int] = []
            cur = i
            while cur >= 0 and len(chain) < width:
                chain.append(width - 1 + cur)
                cur = int(self.parents[cur])
            for j in range(width - len(chain)):
                chain.append(width - 2 - j)
            out[i] = list(reversed(chain))
        return out


def cut(f: Fill, rows: int) -> Rung:
    """The Rung of `rows` rows: the anchor and the fill's first rows-1 nodes, re-indexed in DFS
    pre-order with children in descending score (exp of the path log-probability), ties in insertion
    order -- `TreeBuilder.build`'s order, so the rows equal the host tree's."""
    m = min(rows - 1, f.n)
    score = [math.exp(float(x)) for x in f.lp[:m]]
    kids: list[list[int]] = [[] for _ in range(m + 1)]        # fill index + 1; 0 = the anchor
    for i in range(m):
        kids[int(f.par[i]) + 1].append(i)
    toks, pars, scs, deps, order = [f.anchor], [-1], [1.0], [0], [-1]

    def walk(node: int, new: int, d: int) -> None:
        for c in sorted(kids[node + 1], key=lambda c: -score[c]):
            r = len(toks)
            toks.append(int(f.tok[c]))
            pars.append(new)
            scs.append(score[c])
            deps.append(d + 1)
            order.append(c)
            walk(c, r, d + 1)

    walk(-1, 0, 0)
    return Rung(np.array(toks, np.int64), np.array(pars, np.int64), np.array(scs, np.float64),
                np.array(deps, np.int64), np.array(order, np.int64))


def expected_accepted(f: Fill, rows: int) -> float:
    """Sum of the path probabilities of the Rung's drafted nodes (`DraftTree.expected_accepted`)."""
    m = min(rows - 1, f.n)
    return float(sum(math.exp(float(x)) for x in f.lp[:m]))


def pick(f: Fill, price_ms, draft_ms: float, host_ms: float, sizes=(16, 32),
         calib=None) -> tuple[int, float]:
    """The step with the most tokens a millisecond: (1 + E[accepted](T)) / (V(T) + D + H).
    `price_ms(rows)` is the verify staircase, `calib(node_index) -> factor` an optional per-node
    calibration of the path probability. Ties go to the smaller step."""
    best, best_v = sizes[0], -1.0
    for T in sizes:
        m = min(T - 1, f.n)
        if calib is None:
            e = sum(math.exp(float(x)) for x in f.lp[:m])
        else:
            e = sum(min(1.0, math.exp(float(f.lp[i])) * calib(i)) for i in range(m))
        v = (1.0 + min(e, float(m))) / ((price_ms(T) + draft_ms + host_ms) / 1000.0)
        if v > best_v:
            best, best_v = T, v
    return best, best_v
