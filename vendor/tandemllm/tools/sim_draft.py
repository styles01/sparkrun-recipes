"""Replay any drafter against what the model really wrote, on a laptop, exactly.

Greedy speculative verification accepts the prefix of a draft that matches what the target would
have produced anyway. So if the target's greedy continuation of a prompt is on disk, a drafter's
acceptance can be computed without the board -- not estimated, computed. That turns the loop that
was costing a 20-minute GPU lock per idea into a two-second CPU run, which is the only way to tune
a firing policy at all: a threshold sweep is fifty runs, not one.

What it reports, per drafter and per workload class:

    fire rate         steps where the drafter proposed anything
    tau_fire          accepted draft tokens per fired block
    tau               accepted draft tokens per block over all steps
    saturation        accepted / drafted, the number published board results quote
    nodes             mean verified nodes per block, which is what the verify curve is priced on
    tok/s             expected throughput under the measured cost model

The cost model is the measured one:

    V(N) = base + per_node * N ms,   base 149.1, per_node 1.896     (FP8 weights)

with the drafter's own cost added: zero for a lookup drafter, `--mtp-ms` per proposed token for the
prediction head, and the measured rollback on blocks that take one.

**What is exact and what is not.** Everything about a lookup drafter is exact -- it is a
deterministic function of the token prefix, and the prefix is on disk. The prediction head is not:
its proposals were recorded only at the block boundaries the recording run happened to visit, so a
policy that accepts a different number of tokens walks off the recorded grid. The simulator says
so: every row prints the fraction of its steps where the head's proposal was known rather than
drawn from the trace's own empirical distribution. Read a row with low coverage as an estimate.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import statistics
import sys
import time
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.drafters.engram import EngramDrafter  # noqa: E402
from engine.drafters.ngram import NgramDrafter  # noqa: E402
from engine.tree import DraftTree  # noqa: E402

VERIFY_BASE_MS = 149.1
VERIFY_PER_NODE_MS = 1.896
MTP_MS_PER_TOKEN = 16.6
ROLLBACK_MS = 22.0

# The NVFP4 weight set has a different and less friendly curve: the FP8 step is
# pure bandwidth and extra rows are nearly free, while the FP4 kernel does sixteen rows of
# tensor-core work whether one is asked for or sixteen. Keyed by block length, which is the drafted
# node count plus the anchor.
# Not a line between 1 and 4: the FP4 kernel tiles sixteen rows whether it is asked for one or
# sixteen, so B = 2 and B = 3 cost what B = 4 costs. See engine/router.py.
NVFP4_VERIFY_MS = {1: 126.12, 2: 145.02, 4: 145.02, 8: 153.52, 16: 175.10}
# Measured inside the decode loop rather than in tools/profile_block.py, which reads 19.8-29.3 ms:
# five workloads of tools/bench_ngram.py at 10:01 give 1179/189, 931/151, 327/52, 25/4 and 49/8
# milliseconds per rollback, which is 6.2 every time. The loop's own instrumentation is the
# authority for a constant the loop pays.
NVFP4_ROLLBACK_MS = 6.2

# Measured through `forward_tree` rather than `forward_block`, keyed by nodes
# including the anchor. A staircase, with the step at sixteen, and independent of the tree's shape.
NVFP4_TREE_MS = {2: 140.16, 4: 143.79, 8: 150.01, 12: 156.60, 16: 164.37, 24: 225.10, 32: 225.10}
# The commit is paid on every tree block, not only on rejections.
TREE_COMMIT_MS = 6.6


def nvfp4_tree_ms(nodes: int) -> float:
    """The measured tree curve plus the unconditional commit; flat past its ends."""
    keys = sorted(NVFP4_TREE_MS)
    b = nodes + 1
    if b <= keys[0]:
        return NVFP4_TREE_MS[keys[0]] + TREE_COMMIT_MS
    if b >= keys[-1]:
        return NVFP4_TREE_MS[keys[-1]] + TREE_COMMIT_MS
    for lo, hi in zip(keys, keys[1:]):
        if lo <= b <= hi:
            f = (b - lo) / (hi - lo)
            return NVFP4_TREE_MS[lo] + f * (NVFP4_TREE_MS[hi] - NVFP4_TREE_MS[lo]) + TREE_COMMIT_MS
    return NVFP4_TREE_MS[keys[-1]] + TREE_COMMIT_MS


def nvfp4_verify_ms(nodes: int) -> float:
    """Interpolate the measured NVFP4 block curve; flat past its ends."""
    keys = sorted(NVFP4_VERIFY_MS)
    b = nodes + 1
    if b <= keys[0]:
        return NVFP4_VERIFY_MS[keys[0]]
    if b >= keys[-1]:
        # past 16 the curve is close to linear at 2.51 ms per token
        return NVFP4_VERIFY_MS[keys[-1]] + 2.507 * (b - keys[-1])
    for lo, hi in zip(keys, keys[1:]):
        if lo <= b <= hi:
            f = (b - lo) / (hi - lo)
            return NVFP4_VERIFY_MS[lo] + f * (NVFP4_VERIFY_MS[hi] - NVFP4_VERIFY_MS[lo])
    return NVFP4_VERIFY_MS[keys[-1]]


@dataclass
class Trace:
    name: str
    klass: str
    prompt_ids: list[int]
    output_ids: list[int]
    mtp_at: dict[int, list[int]] = field(default_factory=dict)
    lattice: object = None          # the block drafter's own lattice, at every position

    @staticmethod
    def load(path: str) -> "Trace":
        with open(path) as f:
            d = json.load(f)
        n_prompt = len(d["prompt_ids"])
        mtp_at = {}
        for pos, draft in d.get("mtp_proposals", []):
            i = pos - n_prompt
            if 0 <= i and draft:
                mtp_at[i] = [int(t) for t in draft]
        tr = Trace(name=d["name"], klass=d.get("klass", d["name"].split("-")[0]),
                   prompt_ids=[int(t) for t in d["prompt_ids"]],
                   output_ids=[int(t) for t in d["output_ids"]], mtp_at=mtp_at)
        npz = path[:-5] + ".lattice.npz"
        if os.path.exists(npz):
            tr.lattice = Lattice.load(npz)
        return tr

    def mtp_accepted_lengths(self) -> list[int]:
        """How many tokens the head got right at each position it was recorded at."""
        out = []
        for i, draft in self.mtp_at.items():
            n = 0
            for j, t in enumerate(draft):
                if i + j >= len(self.output_ids) or self.output_ids[i + j] != t:
                    break
                n += 1
            out.append(n)
        return out


class Lattice:
    """The block drafter's own candidate lattice, recorded at every position of a trace.

    The prediction head had to be replayed off a grid (`_ReplayMTP` above says so on every row);
    this one does not. The block drafter conditions on the anchor token and on draft KV built from
    the target's hidden states at the COMMITTED positions, and in a lossless engine the committed
    prefix is the greedy prefix whatever the drafting policy was. So position p's lattice is the
    same lattice for every policy, `tools/record_lattice.py` writes one for every p, and every
    number the simulator reports about this drafter is exact rather than estimated.
    """

    def __init__(self, cand, scores, at):
        self.cand = cand                    # [N, slots, k] int32
        self.scores = scores                # [N, slots, k, k] float16
        self.index = {int(p): i for i, p in enumerate(at)}

    @staticmethod
    def load(path: str) -> "Lattice":
        import numpy as np
        z = np.load(path)
        return Lattice(z["cand"], z["scores"], z["at"])

    def __contains__(self, pos: int) -> bool:
        return pos in self.index

    def at(self, pos: int):
        i = self.index[pos]
        return self.cand[i].tolist(), self.scores[i].astype("float32")


def _log_softmax(x, temp: float):
    import numpy as np
    y = x / temp
    y = y - y.max(axis=-1, keepdims=True)
    return (y - np.log(np.exp(y).sum(axis=-1, keepdims=True))).tolist()


class _ReplayDFlash2:
    """The block drafter, answering from a recorded lattice instead of from 6.14 GB of weights.

    It has the drafter's interface, so `engine.router.MergedRouter` is tuned here as itself. The
    greedy walk is the released one -- slot 0's argmax, then each slot's argmax in the row the
    previous slot selected -- and `propose_tree` calls the same `engine.tree.lattice_tree` the
    engine calls, so a budget or a temperature chosen here is the one that ships.
    """

    name = "dflash2"
    wants_rows = True

    def __init__(self, temp: float = 1.0, budget: int = 16):
        self.temp = temp
        self.budget = budget
        self.lat: Lattice | None = None
        self.i = 0
        self.known = 0
        self.guessed = 0

    def bind(self, trace: Trace) -> None:
        self.lat = trace.lattice
        self.known = self.guessed = 0

    def at(self, i: int) -> None:
        self.i = i

    def _walk(self, cand, scores) -> list[int]:
        idx = int(scores[0, 0].argmax())
        path = [idx]
        local = scores[1:].argmax(axis=-1)
        for e in range(local.shape[0]):
            idx = int(local[e, idx])
            path.append(idx)
        return [cand[l][c] for l, c in enumerate(path)]

    def propose(self, context: list[int], k: int) -> list[int]:
        if self.lat is None or self.i not in self.lat:
            self.guessed += 1
            return []
        self.known += 1
        cand, scores = self.lat.at(self.i)
        return self._walk(cand, scores)[:k]

    def propose_tree(self, context: list[int], budget: int | None = None, **_):
        from engine.tree import DraftTree, lattice_tree
        if self.lat is None or self.i not in self.lat:
            self.guessed += 1
            return None
        self.known += 1
        cand, scores = self.lat.at(self.i)
        chain = self._walk(cand, scores)
        b = self.budget if budget is None else budget
        if b <= 0:
            return DraftTree.chain(int(context[-1]), chain, source="df2-greedy")
        logp = _log_softmax(scores, self.temp)
        greedy = [cand[l].index(chain[l]) for l in range(len(cand))]
        return lattice_tree(int(context[-1]), cand, logp, greedy, b)

    def observe(self, tokens):
        pass

    def reset(self):
        pass

    def sync(self, *args, **kw):
        pass


@dataclass
class Result:
    steps: int = 0
    fired: int = 0
    tokens: int = 0
    accepted: int = 0
    drafted: int = 0
    nodes: int = 0
    accepted_fired: int = 0
    rollbacks: int = 0
    cost_ms: float = 0.0
    draft_cpu_s: float = 0.0
    known: int = 0
    guessed: int = 0

    @property
    def tok_s(self) -> float:
        return self.tokens / (self.cost_ms / 1000.0) if self.cost_ms else 0.0

    @property
    def tau(self) -> float:
        return self.accepted / self.steps if self.steps else 0.0

    @property
    def tau_fire(self) -> float:
        return self.accepted_fired / self.fired if self.fired else 0.0

    @property
    def fire_rate(self) -> float:
        return self.fired / self.steps if self.steps else 0.0

    @property
    def saturation(self) -> float:
        return self.accepted / self.drafted if self.drafted else 0.0

    @property
    def mean_nodes(self) -> float:
        return self.nodes / self.steps if self.steps else 0.0

    @property
    def coverage(self) -> float:
        tot = self.known + self.guessed
        return self.known / tot if tot else 1.0

    def add(self, other: "Result") -> None:
        for f in ("steps", "fired", "tokens", "accepted", "drafted", "nodes", "accepted_fired",
                  "rollbacks", "known", "guessed"):
            setattr(self, f, getattr(self, f) + getattr(other, f))
        self.cost_ms += other.cost_ms
        self.draft_cpu_s += other.draft_cpu_s


# --- policies -------------------------------------------------------------------------------
#
# A policy is called once per block with the full context and the drafter state, and returns the
# block it wants verified plus what drafting it cost in milliseconds. `None` means "no draft":
# the engine falls back to an ordinary one-token step.


class Policy:
    name = "policy"
    uses_mtp = False

    def reset(self, trace: Trace) -> None:
        pass

    def block(self, ctx: list[int], i: int, trace: Trace) -> tuple[DraftTree | None, float]:
        raise NotImplementedError

    def observe(self, tokens: list[int]) -> None:
        pass


class NoDrafter(Policy):
    name = "none"

    def block(self, ctx, i, trace):
        return None, 0.0


class ChainPolicy(Policy):
    """Any drafter that speaks the old chain interface: engram v1, or v2 through `propose`."""

    def __init__(self, make, name: str, k: int):
        self.make = make
        self.name = f"{name} k={k}"
        self.k = k
        self.d = None

    def reset(self, trace):
        self.d = self.make()
        self.d.prime(trace.prompt_ids)

    def block(self, ctx, i, trace):
        t0 = time.perf_counter()
        draft = self.d.propose(ctx, self.k)
        dt = (time.perf_counter() - t0) * 1000.0
        if not draft:
            return None, dt
        return DraftTree.chain(ctx[-1], draft, source="ngram"), dt

    def observe(self, tokens):
        self.d.observe(tokens)


class TreePolicy(Policy):
    """The lookup drafter proposing a tree, with its own firing policy deciding the steps."""

    def __init__(self, make, name: str, depth: int, budget: int, alternative: float = 0.0):
        self.make = make
        self.name = name
        self.depth = depth
        self.budget = budget
        self.alternative = alternative
        self.d = None

    def reset(self, trace):
        self.d = self.make()
        self.d.prime(trace.prompt_ids)

    def block(self, ctx, i, trace):
        t0 = time.perf_counter()
        tree = self.d.propose_tree(ctx, self.depth, alternative_value=self.alternative)
        dt = (time.perf_counter() - t0) * 1000.0
        return tree, dt

    def observe(self, tokens):
        self.d.observe(tokens)


class MTPPolicy(Policy):
    """Replay the recorded prediction head. Exact where it was recorded, drawn where it was not."""

    name = "mtp"
    uses_mtp = True

    def __init__(self, depth: int, rng: random.Random, mtp_ms: float = MTP_MS_PER_TOKEN):
        self.depth = depth
        self.name = f"mtp d={depth}"
        self.rng = rng
        self.mtp_ms = mtp_ms
        self.pool: list[int] = []

    def reset(self, trace):
        self.pool = trace.mtp_accepted_lengths() or [0]

    def block(self, ctx, i, trace):
        draft = trace.mtp_at.get(i)
        cost = self.mtp_ms * self.depth
        if draft is not None:
            return DraftTree.chain(ctx[-1], draft[:self.depth], source="mtp"), cost
        # off the recorded grid: draw an accepted length and build a chain that realises it
        n = min(self.rng.choice(self.pool), self.depth)
        real = trace.output_ids[i:i + self.depth]
        draft = list(real[:n]) + [-1] * (self.depth - n)
        return DraftTree.chain(ctx[-1], draft[:self.depth], source="mtp"), cost


class _ReplayMTP:
    """Stands in for the prediction head, answering from the trace instead of from weights.

    It has the drafter's interface, so `engine.router.MergedRouter` can be tuned here as itself
    rather than as a reimplementation of itself -- the policy that ships is the policy that was
    measured.
    """

    name = "mtp"

    def __init__(self, depth: int, rng: random.Random):
        self.depth = depth
        self.rng = rng
        self.trace: Trace | None = None
        self.i = 0
        self.pool: list[int] = [0]
        self.known = 0
        self.guessed = 0

    def bind(self, trace: Trace) -> None:
        self.trace = trace
        self.pool = trace.mtp_accepted_lengths() or [0]
        self.known = self.guessed = 0

    def at(self, i: int) -> None:
        self.i = i

    def propose(self, context: list[int], k: int) -> list[int]:
        draft = self.trace.mtp_at.get(self.i)
        if draft is not None:
            self.known += 1
            return list(draft[:k])
        self.guessed += 1
        n = min(self.rng.choice(self.pool), k)
        real = self.trace.output_ids[self.i:self.i + k]
        return list(real[:n]) + [-1] * max(0, min(k, self.depth) - n)

    def observe(self, tokens):
        pass

    def reset(self):
        pass

    def sync(self, *args):
        pass


class RealRouterPolicy(Policy):
    """`engine.router.MergedRouter` itself, with the head replayed from the trace.

    `mode="chain"` is what the verify path can do today: price both drafters and take the better
    chain. `mode="tree"` is what it will do once a block can be verified as a tree, and merges them.
    """

    uses_mtp = True

    def __init__(self, make, depth: int, tree_depth: int, budget: int, rng: random.Random,
                 mode: str = "chain", mtp_ms: float = MTP_MS_PER_TOKEN,
                 name: str | None = None, head: str = "mtp", call_ms: float = 0.0,
                 temp: float = 1.0, verify_table: dict | None = None):
        from engine.router import MergedRouter
        self.make = make
        self.depth = depth
        self.tree_depth = tree_depth
        self.budget = budget
        self.mode = mode
        self.head_kind = head
        # A block drafter's price is FIXED per call: one non-causal pass over its own five layers
        # and one read of the vocabulary head, whatever the block yields. A chained prediction head
        # pays per proposed token. The router has to be told which it is holding, or it shortens
        # something that shortening does not make cheaper.
        self.mtp_ms = 0.0 if head == "dflash2" else mtp_ms
        self.call_ms = call_ms
        self.verify_table = verify_table
        self.head = (_ReplayDFlash2(temp, budget) if head == "dflash2"
                     else _ReplayMTP(depth, rng))
        self._cls = MergedRouter
        self.name = name or f"router/{head}-{mode}"
        self.verifies_tree = mode == "tree"
        self.r = None

    def reset(self, trace):
        self.head.bind(trace)
        kw = dict(mtp_depth=self.depth, node_budget=self.budget, mtp_ms_per_token=self.mtp_ms)
        if self.verifies_tree:
            kw["tree_ms_table"] = NVFP4_TREE_MS
        if self.head_kind == "dflash2":
            kw.update(head_fixed_ms=self.call_ms, adaptive_depth=False, mtp_depth=8)
        if self.verify_table:
            kw["verify_ms_table"] = self.verify_table
        self.r = self._cls(self.make(), self.head, **kw)
        self.r.prime(trace.prompt_ids)

    def block(self, ctx, i, trace):
        self.head.at(i)
        t0 = time.perf_counter()
        if self.mode == "tree":
            tree = self.r.propose_tree(ctx, self.tree_depth)
        else:
            chain = self.r.propose(ctx, self.tree_depth)
            tree = DraftTree.chain(ctx[-1], chain, source="router") if chain else None
        dt = (time.perf_counter() - t0) * 1000.0
        cost = self.mtp_ms * self.r.last_depth + (self.call_ms if self.head_kind == "dflash2"
                                                  and self.r.last != "ngram" else 0.0)
        return tree, cost + dt

    def observe(self, tokens):
        self.r.observe(tokens)


class DFlash2Policy(Policy):
    """The block drafter alone, as a chain or as a tree, from the recorded lattice."""

    uses_mtp = True
    verifies_tree = False

    def __init__(self, temp: float, budget: int, mode: str, call_ms: float):
        self.d = _ReplayDFlash2(temp, budget)
        self.mode = mode
        self.call_ms = call_ms
        self.verifies_tree = mode != "chain"
        self.name = ("dflash2-chain" if mode == "chain"
                     else f"dflash2-tree b={budget} t={temp:g}")

    def reset(self, trace):
        self.d.bind(trace)

    def block(self, ctx, i, trace):
        from engine.tree import DraftTree
        self.d.at(i)
        t0 = time.perf_counter()
        if self.mode == "chain":
            chain = self.d.propose(ctx, 7)
            tree = DraftTree.chain(ctx[-1], chain, source="df2-greedy") if chain else None
        else:
            tree = self.d.propose_tree(ctx)
        dt = (time.perf_counter() - t0) * 1000.0
        return tree, self.call_ms + dt


# --- the replay -----------------------------------------------------------------------------


def simulate(policy: Policy, trace: Trace, base_ms: float, per_node_ms: float,
             rollback_ms: float, verify=None) -> Result:
    """`verify(nodes) -> ms` overrides the linear model when a measured curve is available.

    A policy that says `verifies_tree` is priced on the tree curve instead, and pays the commit on
    every fired block rather than the rollback on rejections -- the two differences between
    `forward_block` and `forward_tree` that the cost model can see.
    """
    if getattr(policy, "verifies_tree", False):
        verify, rollback_ms = nvfp4_tree_ms, 0.0
    r = Result()
    policy.reset(trace)
    out = trace.output_ids
    ctx = list(trace.prompt_ids)
    i = 0
    while i < len(out):
        tree, draft_ms = policy.block(ctx, i, trace)
        r.steps += 1
        if policy.uses_mtp:
            if i in trace.mtp_at:
                r.known += 1
            else:
                r.guessed += 1
        if tree is None or tree.n_draft == 0:
            gained = 1
            r.cost_ms += (verify(0) if verify else base_ms) + draft_ms
        else:
            r.fired += 1
            n = tree.accepted_against(out[i:])
            r.accepted += n
            r.accepted_fired += n
            r.drafted += tree.n_draft
            r.nodes += tree.n_draft
            gained = n + 1
            r.cost_ms += ((verify(tree.n_draft) if verify
                           else base_ms + per_node_ms * tree.n_draft) + draft_ms)
            # any rejected node means the recurrent state has to be replayed to the accepted prefix
            if n < tree.n_draft:
                r.rollbacks += 1
                r.cost_ms += rollback_ms
        r.draft_cpu_s += draft_ms / 1000.0
        new = out[i:i + gained]
        policy.observe(list(new))
        ctx = ctx + list(new)
        i += gained
        r.tokens += len(new)
    return r


def fmt(label: str, r: Result, show_coverage: bool) -> str:
    cov = f"{r.coverage * 100:5.0f}%" if show_coverage else "    -"
    return (f"{label:26s} {r.tok_s:7.2f} {r.tau:7.2f} {r.tau_fire:8.2f} "
            f"{r.fire_rate * 100:7.1f}% {r.saturation * 100:8.1f}% {r.mean_nodes:7.2f} "
            f"{r.draft_cpu_s * 1000 / max(r.steps, 1):8.2f} {cov}")


GRID: dict[str, list] = {
    # ordered so the knobs that moved the dry run most come first
    "alpha": [0.05, 0.1, 0.2, 0.4, 0.6, 1.0],
    "min_corpus_order": [3, 4, 5, 6, 7, 8],
    "max_depth": [4, 8, 12, 16, 24, 32],
    "node_budget": [4, 8, 12, 16, 24, 32],
    "branch_top_k": [1, 2, 3, 4],
    "min_order": [2, 3, 4, 5],
    "corpus_weight": [0.1, 0.25, 0.5, 1.0, 2.0],
    "min_expected": [0.2, 0.5, 1.0, 2.0],
}


def tune(a, traces, make_ngram, rng, verify, rollback, passes: int = 2) -> None:
    """Coordinate descent on mean tok/s over the traces.

    Not a search over a large space: eight knobs, six values each, two passes. What it buys is that
    the firing policy is set by expected throughput on recorded generations rather than by a hit
    rate someone liked the look of. The objective is deliberately tok/s and not acceptance: a
    drafter that fires more often and accepts more tokens can still be slower, because every node
    it adds costs verify time whether the node is right or not.
    """
    best = {"alpha": a.alpha, "min_corpus_order": a.min_corpus_order, "max_depth": a.depth,
            "node_budget": a.budget, "branch_top_k": a.branch_top_k, "min_order": a.min_order,
            "corpus_weight": a.corpus_weight, "min_expected": a.min_expected}

    def score(cfg) -> float:
        if a.tune_policy == "tree":
            p = TreePolicy(make_ngram(**cfg), "tune", cfg["max_depth"], cfg["node_budget"])
        else:
            p = RealRouterPolicy(make_ngram(**cfg), a.mtp_depth, cfg["max_depth"],
                                 cfg["node_budget"], random.Random(a.seed),
                                 mode="tree" if a.tune_policy == "router-tree" else "chain",
                                 mtp_ms=a.mtp_ms)
        total = Result()
        for t in traces:
            total.add(simulate(p, t, a.base_ms, a.per_node_ms, rollback, verify))
        return total.tok_s

    current = score(best)
    print(f"\ntuning {a.tune_policy}: start {current:.2f} tok/s at {best}")
    for p_i in range(passes):
        for knob, values in GRID.items():
            trials = []
            for v in values:
                if v == best[knob]:
                    trials.append((current, v))
                    continue
                cfg = dict(best, **{knob: v})
                trials.append((score(cfg), v))
            # A knob only moves when it is worth moving. Rounding to three decimals was not
            # enough: `min_expected` won by under 0.005 tok/s on traces that contain no `quote`
            # workload, and on the board that choice took the quote row from 38.85 to 8.03 by
            # silencing the drafter. Ties go to the incumbent and the tie band is 0.05 tok/s.
            incumbent = best[knob]
            band = 0.05
            got, v = max(trials, key=lambda t: (round(t[0] / band), t[1] == incumbent))
            got = next(sc for sc, val in trials if val == v)
            mark = "  <-" if v != best[knob] else ""
            print(f"  pass {p_i + 1} {knob:17s} " +
                  " ".join(f"{val}:{sc:.2f}" for sc, val in
                           sorted(trials, key=lambda t: values.index(t[1]))) +
                  f"   best {v}{mark}")
            best[knob], current = v, got
    print(f"\nbest {current:.2f} tok/s at {best}")
    print("  command: " + " ".join(
        f"--{k.replace('_', '-').replace('max-depth', 'depth').replace('node-budget', 'budget')} {v}"
        for k, v in best.items()))


HEADER = (f"{'policy':26s} {'tok/s':>7} {'tau':>7} {'tau|fire':>8} {'fire':>8} "
          f"{'sat':>9} {'nodes':>7} {'cpu ms':>8} {'known':>6}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default=None, help="directory of trace json (default results/traces)")
    ap.add_argument("--corpus", default=os.environ.get("QWEN38_CORPUS", ""),
                    help="global suffix store; empty disables it")
    ap.add_argument("--curve", default="fp8", choices=["fp8", "nvfp4"],
                    help="which measured verify curve to price against")
    ap.add_argument("--base-ms", type=float, default=VERIFY_BASE_MS)
    ap.add_argument("--per-node-ms", type=float, default=VERIFY_PER_NODE_MS)
    ap.add_argument("--rollback-ms", type=float, default=ROLLBACK_MS)
    ap.add_argument("--mtp-depth", type=int, default=3)
    ap.add_argument("--mtp-ms", type=float, default=MTP_MS_PER_TOKEN,
                    help="drafting cost per proposed token; falls with a trimmed draft head")
    ap.add_argument("--depth", type=int, default=16, help="max lookup draft depth")
    ap.add_argument("--budget", type=int, default=16, help="node budget for trees")
    ap.add_argument("--min-expected", type=float, default=0.6)
    ap.add_argument("--branch-top-k", type=int, default=3)
    ap.add_argument("--min-order", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=0.6, help="score smoothing, per tree level")
    ap.add_argument("--corpus-weight", type=float, default=0.5)
    ap.add_argument("--min-corpus-order", type=int, default=5)
    ap.add_argument("--df2-ms", type=float, default=35.0,
                    help="cost of one block-drafter call; fixed per call, not per token")
    ap.add_argument("--df2-temp", type=float, default=1.0,
                    help="temperature on the selector's scores before they are read as "
                         "probabilities; the one knob that says how much to believe the lattice")
    ap.add_argument("--df2-budget", type=int, default=16, help="nodes in the block drafter's tree")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", default=None, help="comma separated policy names to run")
    ap.add_argument("--sweep", default=None,
                    help="tune one knob: alpha | min_expected | budget | depth | branch_top_k "
                         "| min_order | corpus_weight | min_corpus_order")
    ap.add_argument("--by-class", action="store_true", help="break the summary down by class")
    ap.add_argument("--tune", action="store_true",
                    help="coordinate descent over the drafter's knobs, maximising mean tok/s")
    ap.add_argument("--tune-passes", type=int, default=2)
    ap.add_argument("--tune-policy", default="tree", choices=["tree", "router-chain", "router-tree"],
                    help="which policy the tuning optimises")
    a = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tdir = a.traces or os.path.join(root, "results", "traces")
    paths = sorted(glob.glob(os.path.join(tdir, "*.json")))
    if not paths:
        raise SystemExit(f"no traces in {tdir}; run tools/record_traces.py on the board first")
    traces = [Trace.load(p) for p in paths]
    print(f"{len(traces)} traces, {sum(len(t.output_ids) for t in traces):,} target tokens, "
          f"classes {sorted({t.klass for t in traces})}")
    if a.corpus:
        meta_path = os.path.join(a.corpus, "meta.json")
        meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
        print(f"corpus store: {meta.get('n_tokens', '?'):,} tokens from {a.corpus}"
              if isinstance(meta.get("n_tokens"), int) else f"corpus store: {a.corpus}")
        if any(src.get("kind") == "traces" for src in meta.get("sources", [])):
            print("\n  *** the corpus contains the recorded traces. Every number below is the\n"
                  "  drafter looking up the answer it is being scored against. Rebuild the store\n"
                  "  without --traces. ***\n")

    def make_ngram(**kw):
        opts = dict(corpus_path=a.corpus, min_order=a.min_order, max_depth=a.depth,
                    node_budget=a.budget, branch_top_k=a.branch_top_k,
                    min_expected=a.min_expected, verify_base_ms=a.base_ms,
                    verify_per_node_ms=a.per_node_ms, alpha=a.alpha,
                    corpus_weight=a.corpus_weight, min_corpus_order=a.min_corpus_order)
        opts.update(kw)
        return lambda: NgramDrafter(**opts)

    verify = nvfp4_verify_ms if a.curve == "nvfp4" else None
    rollback = NVFP4_ROLLBACK_MS if a.curve == "nvfp4" else a.rollback_ms
    if a.curve == "nvfp4":
        print("pricing against the NVFP4 block curve")

    rng = random.Random(a.seed)
    policies: list[Policy] = [
        NoDrafter(),
        ChainPolicy(lambda: EngramDrafter(), "engram-v1", 8),
        ChainPolicy(make_ngram(), "ngram-chain", a.depth),
        TreePolicy(make_ngram(), f"ngram-tree b={a.budget}", a.depth, a.budget),
        MTPPolicy(a.mtp_depth, rng, a.mtp_ms),
        RealRouterPolicy(make_ngram(), a.mtp_depth, a.depth, a.budget, rng, mode="chain",
                         mtp_ms=a.mtp_ms),
        RealRouterPolicy(make_ngram(), a.mtp_depth, a.depth, a.budget, rng, mode="tree",
                         mtp_ms=a.mtp_ms),
    ]
    table = NVFP4_VERIFY_MS if a.curve == "nvfp4" else None
    if any(t.lattice is not None for t in traces):
        policies += [
            DFlash2Policy(a.df2_temp, 0, "chain", a.df2_ms),
            DFlash2Policy(a.df2_temp, a.df2_budget, "tree", a.df2_ms),
            RealRouterPolicy(make_ngram(), a.mtp_depth, a.depth, a.df2_budget, rng, mode="chain",
                             head="dflash2", call_ms=a.df2_ms, temp=a.df2_temp,
                             verify_table=table),
            RealRouterPolicy(make_ngram(), a.mtp_depth, a.depth, a.df2_budget, rng, mode="tree",
                             head="dflash2", call_ms=a.df2_ms, temp=a.df2_temp,
                             verify_table=table),
        ]
    if a.sweep in ("df2_temp", "df2_budget"):
        policies = [NoDrafter(), DFlash2Policy(a.df2_temp, 0, "chain", a.df2_ms)]
        vals = ([0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0] if a.sweep == "df2_temp"
                else [2, 4, 7, 11, 16, 24, 32, 48])
        for v in vals:
            temp = v if a.sweep == "df2_temp" else a.df2_temp
            bud = v if a.sweep == "df2_budget" else a.df2_budget
            policies.append(DFlash2Policy(temp, bud, "tree", a.df2_ms))
        a.sweep = None
    if a.sweep:
        policies = [NoDrafter()]
        values = {"alpha": [0.1, 0.2, 0.4, 0.6, 1.0, 2.0],
                  "min_expected": [0.2, 0.4, 0.6, 0.8, 1.2, 1.6, 2.4],
                  "budget": [4, 8, 12, 16, 24, 32, 48],
                  "depth": [4, 8, 12, 16, 24, 32],
                  "branch_top_k": [1, 2, 3, 4, 6],
                  "min_order": [2, 3, 4, 5, 6],
                  "corpus_weight": [0.1, 0.25, 0.5, 1.0, 2.0],
                  "min_corpus_order": [3, 4, 5, 6, 7]}[a.sweep]
        arg_for = {"alpha": "alpha", "min_expected": "min_expected", "budget": "node_budget",
                   "depth": "max_depth", "branch_top_k": "branch_top_k",
                   "min_order": "min_order", "corpus_weight": "corpus_weight",
                   "min_corpus_order": "min_corpus_order"}
        for v in values:
            kw = {arg_for[a.sweep]: v}
            depth = v if a.sweep == "depth" else a.depth
            budget = v if a.sweep == "budget" else a.budget
            policies.append(TreePolicy(make_ngram(**kw), f"ngram-tree {a.sweep}={v}",
                                       depth, budget))
    if a.only:
        wanted = set(a.only.split(","))
        policies = [p for p in policies if p.name.split()[0] in wanted]

    if a.tune:
        tune(a, traces, make_ngram, rng, verify, rollback, passes=a.tune_passes)
        return

    print("\n" + HEADER)
    print("-" * len(HEADER))
    summaries: dict[str, dict[str, Result]] = {}
    for p in policies:
        total = Result()
        per_class: dict[str, Result] = {}
        for t in traces:
            r = simulate(p, t, a.base_ms, a.per_node_ms, rollback, verify)
            total.add(r)
            per_class.setdefault(t.klass, Result()).add(r)
        summaries[p.name] = per_class
        print(fmt(p.name, total, p.uses_mtp))

    if a.by_class:
        klasses = sorted({t.klass for t in traces})
        print("\ntok/s by class")
        print(f"{'policy':26s} " + " ".join(f"{k:>8}" for k in klasses) + f" {'mean':>8}")
        for name, per_class in summaries.items():
            vals = [per_class[k].tok_s if k in per_class else 0.0 for k in klasses]
            print(f"{name:26s} " + " ".join(f"{v:8.2f}" for v in vals)
                  + f" {statistics.fmean(vals):8.2f}")


if __name__ == "__main__":
    main()
