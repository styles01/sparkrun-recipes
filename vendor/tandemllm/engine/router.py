"""Choosing a drafter, and a draft length, once per step.

Two drafters with opposite cost structures are available. A suffix memory proposes long blocks for
nothing but only where the context repeats itself, which on this board is 5-10 % of steps. The
checkpoint's prediction head proposes on every step but reads 3.0 GB per token it proposes -- about
17 ms, against a verify step of 151-180 ms -- so a draft it gets wrong is not free.

The choice is an expected-value calculation, not a heuristic, and every term in it is something the
engine has already measured:

    value(drafter, m)  =  E[accepted | drafter, m] / (verify(m + 1) + cost(drafter, m))

`verify` is the measured verify curve; `cost` is zero for the suffix memory and
linear in m for the prediction head; `E[accepted]` is estimated from that drafter's own recent
history, and for the suffix memory it is conditioned on the length of the match it found, which is
the one signal available before committing.

The router deliberately sees nothing about what the text says. It sees match lengths, recent
acceptance and block economics.
"""

from __future__ import annotations

from engine.drafters import Drafter, run_steps, tree_steps
from engine.settings import SETTINGS as _S  # noqa: E402  (every QWEN38_* knob)

# A sampled request's spine tree cut to the node profile the staircase chose for the head's
# deterministic tree (`DraftTree.cut_like`), instead of the whole spine plus its uncut quota.
SAMPLED_CUT = _S.get("SAMPLED_CUT") == "1"

# verify(B) in seconds, measured on this board (verify cost against block length). Linear between the measured points, flat-extrapolated past the ends.
VERIFY_MS = {1: 151.0, 2: 161.4, 4: 164.9, 6: 167.4, 8: 169.7, 12: 173.6, 16: 179.4, 32: 209.8}
MTP_MS_PER_TOKEN = 16.6      # measured: 846 ms of drafting over 17 three-token proposals
ROLLBACK_MS = 22.0           # measured, near-flat in the accepted prefix length

# The same two numbers on the NVFP4 weight set with the trimmed 32k draft head, read off the decode
# loop's own instrumentation in tools/bench_ngram.py at 10:01 rather than off a microbenchmark:
# 3.4 ms per drafted token (2188 ms over 215 three-token drafts), and 6.2 ms per rollback on all
# five workloads. Getting the rollback wrong by 4x is not a rounding error in a router -- it is the
# term that decides how long a chain is worth proposing, and at 23 ms the router shortens chains
# that pay at 6.
MTP_MS_PER_TOKEN_TRIMMED = 3.4
ROLLBACK_MS_NVFP4 = 6.2


# The same curve after the MLPs went to four bits. It is not a line, and
# reading it as one cost 5 % of the prose row at 10:37. The FP4 kernel does sixteen rows of
# tensor-core work whether one row is asked for or sixteen, so B = 1 runs a different path
# (M = 1 GEMV, 126.12 ms) and everything from B = 2 to B = 4 costs what B = 4 costs. Interpolating
# between 126.12 and 145.02 tells a router that a two-token block is 13 ms cheaper than a
# four-token one, and it is not cheaper at all; the router then shortens chains that pay.
#
# The entry for 2 is not a separate measurement. It is the kernel's own tiling asserted as a cost,
# and tools/profile_block.py should be run at B = 2 and B = 3 to confirm it.
VERIFY_MS_NVFP4 = {1: 126.12, 2: 145.02, 4: 145.02, 8: 153.52, 16: 175.10}

# The same board, the same weights, measured through `forward_tree` instead (tools/verify_tree.py
# --curve). Keyed by nodes INCLUDING the anchor, which is what a tree calls
# `n_draft + 1`. Two things in it decide the shape of every tree this router builds.
#
# It is a STAIRCASE with its step at sixteen, not a line: 2 nodes cost 140.2 ms and 16 cost 164.4,
# which is 17 % more step for eight times the width, and then 24 costs 225.1 -- the FP4 kernel's
# second tile of sixteen rows, paid in full. So the node budget is 16, and 17 is expensive.
#
# And the SHAPE is free: the same sixteen nodes cost 164.37 as a chain, 165.39 branching two ways
# and 165.41 branching three. The ancestor mask, the convolution's gather and the ancestor-masked
# UT factorisation are elementwise work on [16, 16] matrices against 19.4 GB of weight reads.
TREE_MS_NVFP4 = {2: 140.16, 4: 143.79, 8: 150.01, 12: 156.60, 16: 164.37, 24: 225.10, 32: 225.10}
# The commit is paid on EVERY tree block rather than only on a rejection, because a tree has no
# single successor state: 6.0 to 7.9 ms, and flat in the tree size.
TREE_COMMIT_MS = 6.6


# The tree verify table the served length router's arms are priced on (server/app.py and every
# tool that rebuilds the served configuration). Its ratios, not its level, decide what a node is
# worth; the length router replaces the level with what the loop pays.
SERVED_TREE_MS = {8: 121.7, 16: 129.2, 32: 163.2}


def served_tree_table(table: dict[int, float] | None = None) -> dict[int, float]:
    """`SERVED_TREE_MS`, or `QWEN38_TREE_MS` ("8:99.1,16:100,32:110") -- prices a wider tree
    on the curve measured, not on the one with the cliff in it. `table` is a weight set's
    tree curve from a price table (engine/prices.py); the environment still wins."""
    import os
    env = _S.get("TREE_MS").strip()
    if not env:
        return dict(table) if table else dict(SERVED_TREE_MS)
    table = {int(k): float(v) for k, v in (kv.split(":") for kv in env.split(","))}
    if not {8, 16} <= set(table):
        raise ValueError(f"QWEN38_TREE_MS={env!r}: the arms are priced at 8 and 16 nodes, both needed")
    return table


def tree_nodes(block_size: int) -> int:
    """How many nodes, anchor included, an arm's tree may have: its block size unless the
    `QWEN38_TREE_NODES` (the 16-wide arm) or `QWEN38_TREE_NODES_NARROW` (the 8-wide one) says more.
    The lattice still has `block_size - 1` slots, so a wider budget buys branches, not depth."""
    key = "TREE_NODES_NARROW" if block_size <= 8 else "TREE_NODES"
    return max(2, int(_S.get(key) or 0) or int(block_size))


def verify_ms(b: int, table: dict[int, float] | None = None) -> float:
    table = table or VERIFY_MS
    keys = sorted(table)
    if b <= keys[0]:
        return table[keys[0]]
    if b >= keys[-1]:
        slope = (table[keys[-1]] - table[keys[-2]]) / (keys[-1] - keys[-2])
        return table[keys[-1]] + slope * (b - keys[-1])
    for lo, hi in zip(keys, keys[1:]):
        if lo <= b <= hi:
            f = (b - lo) / (hi - lo)
            return table[lo] + f * (table[hi] - table[lo])
    return table[keys[-1]]


class _Rate:
    """Exponentially weighted acceptance, kept per drafter."""

    def __init__(self, init: float, alpha: float = 0.15):
        self.value = init
        self.alpha = alpha

    def update(self, accepted: int, proposed: int) -> None:
        if proposed:
            self.value += self.alpha * (accepted / proposed - self.value)


class RouterDrafter(Drafter):
    """Runs the suffix memory first and falls back to the prediction head.

    The suffix memory is tried first because asking it costs a dictionary lookup. It is used when
    the expected value of its proposal beats the head's, which given its zero cost means: when the
    match it found is long enough that its measured acceptance at that match length pays for the
    extra verify width and the rollback it risks.
    """

    name = "router"

    def __init__(self, engram, mtp, mtp_depth: int = 3, min_engram_order: int = 4):
        self.engram = engram
        self.mtp = mtp
        self.mtp_depth = mtp_depth
        self.min_engram_order = min_engram_order
        self.rate_engram = _Rate(0.5)
        self.rate_mtp = _Rate(0.6)
        self.last = None
        self.last_n = 0
        self.stats = {"engram": 0, "mtp": 0, "engram_tokens": 0, "mtp_tokens": 0}

    def reset(self) -> None:
        self.engram.reset()
        self.mtp.reset()
        self.last = None

    def prime(self, tokens: list[int]) -> None:
        self.engram.prime(tokens)

    def state_snapshot(self):
        """The neural sub-drafter's cache. The lookup drafter is rebuilt from tokens by `prime`."""
        return ("router", self.mtp.state_snapshot())

    def snapshot_bytes_per_token(self) -> int:
        from engine.cache import _drafter_bytes_per_token
        return _drafter_bytes_per_token(self.mtp)

    def state_restore(self, snap) -> None:
        kind, sub = snap
        if kind != "router":
            raise ValueError(f"not a router snapshot: {kind!r}")
        self.mtp.state_restore(sub)

    def can_resume(self) -> bool:
        return hasattr(self.mtp, "state_resume")

    def state_resume(self, n: int) -> None:
        fn = getattr(self.mtp, "state_resume", None)
        if fn is None:
            raise RuntimeError("the neural sub-drafter cannot resume in place")
        fn(n)

    def kv_views(self) -> list:
        fn = getattr(self.mtp, "kv_views", None)
        return fn() if fn is not None else []

    def sync(self, tokens, hidden, first_pos) -> None:
        self.mtp.sync(tokens, hidden, first_pos)

    def observe(self, tokens: list[int]) -> None:
        # `tokens` is the accepted block: the drafted prefix plus the model's own token.
        accepted = max(0, len(tokens) - 1)
        if self.last == "engram":
            self.rate_engram.update(accepted, self.last_n)
        elif self.last == "mtp":
            self.rate_mtp.update(accepted, self.last_n)
        self.engram.observe(tokens)

    def _value(self, rate: float, m: int, cost_ms: float) -> float:
        expected = rate * m
        risk = ROLLBACK_MS * (1.0 - rate ** m)
        return expected / (verify_ms(m + 1) + cost_ms + risk)

    def propose(self, context: list[int], k: int) -> list[int]:
        draft = self.engram.propose(context, k)
        if draft and len(draft) >= 1:
            order = self.engram.last_order
            if order >= self.min_engram_order:
                v_engram = self._value(self.rate_engram.value, len(draft), 0.0)
                m = min(self.mtp_depth, k)
                v_mtp = self._value(self.rate_mtp.value, m, MTP_MS_PER_TOKEN * m)
                if v_engram >= v_mtp:
                    self.last, self.last_n = "engram", len(draft)
                    self.stats["engram"] += 1
                    self.stats["engram_tokens"] += len(draft)
                    return draft
        m = min(self.mtp_depth, k)
        if m <= 0:
            return []
        draft = self.mtp.propose(context, m)
        self.last, self.last_n, self.last_depth = "mtp", len(draft), m
        self.stats["mtp"] += 1
        self.stats["mtp_tokens"] += len(draft)
        return draft


# ---------------------------------------------------------------------------------------------
# Second version, 2026-09-17. The first router chose a drafter; this one prices them.
# ---------------------------------------------------------------------------------------------


class MergedRouter(Drafter):
    """Spend one block's node budget across a lookup tree and the block drafter's chain.

    The first router's measured defect was not its shape but its evidence: it estimated the suffix
    memory's acceptance from single-digit sample counts, and on the steps where it guessed wrong it
    spent a whole verify on a lookup that did not pay (-0.8 tok/s on prose, -2.2 on edit). Two
    things fix that, and neither is a heuristic.

    **The estimate comes from the drafter, not from a counter.** `NgramDrafter` returns a tree whose
    scores are path probabilities, so the expected accepted length of its proposal is a number it
    computes per step from the counts it actually found. All the router has to learn is one scalar:
    how optimistic those probabilities have been lately. A single calibration factor needs far less
    evidence than a per-drafter acceptance rate.

    **Nothing is winner-take-all once the verify path can take a tree.** Merging costs the union of
    the two proposals -- shared prefixes are shared nodes -- and at 1.896 ms per node a wrong lookup
    costs a few milliseconds instead of a step. Until `forward_tree` exists the router still has to
    pick a chain, and then it picks the one with the higher expected value under the measured curve;
    `propose_tree` is the same decision with the merge left in, and is what the tree verify will
    call.
    """

    name = "merged"
    wants_rows = True          # it holds a head that reads the engine's tap; see engine/spec.py

    def __init__(self, ngram, mtp, mtp_depth: int = 3, node_budget: int = 16,
                 mtp_ms_per_token: float = MTP_MS_PER_TOKEN, rollback_ms: float = ROLLBACK_MS,
                 alpha: float = 0.15, verify_ms_table: dict[int, float] | None = None,
                 depth_margin: float = 0.05, head_fixed_ms: float = 0.0,
                 adaptive_depth: bool = True, tree_ms_table: dict[int, float] | None = None,
                 commit_ms: float = TREE_COMMIT_MS, head_prior: float = 4.3,
                 always_head: bool = False):
        self.ngram = ngram
        self.mtp = mtp
        self.mtp_depth = mtp_depth
        self.node_budget = node_budget
        self.mtp_ms_per_token = mtp_ms_per_token
        self.rollback_ms = rollback_ms
        self.depth_margin = depth_margin
        # A block drafter costs the same whether its block is read in full or not: one non-causal
        # pass over its own layers and one pass over the vocabulary head. So its price is fixed per
        # call rather than per proposed token, and shortening its block saves nothing. The chained
        # prediction head is the other case: it pays per token, and shortening it is a real saving.
        self.head_fixed_ms = head_fixed_ms
        self.adaptive_depth = adaptive_depth
        # How many nodes the head may put in the tree before the merge. Deliberately the whole
        # budget rather than a share of it: the merge shares prefixes and the prune afterwards is
        # what decides, so starving either source before they meet only hides candidates from the
        # marginal rule.
        self.head_budget = node_budget
        # Tree blocks are priced off their own measured curve, not off the chain one. They are not
        # the same measurement: a chain of eight costs 153.5 ms through `forward_block` and a tree
        # of eight costs 150.0 through `forward_tree`, and that gap is larger than the differences
        # this router decides on.
        self.tree_table = dict(tree_ms_table) if tree_ms_table else dict(TREE_MS_NVFP4)
        tk = sorted(self.tree_table)
        self.tree_base_ms = self.tree_table[tk[0]]
        self.tree_per_node_ms = ((self.tree_table[16] - self.tree_table[tk[0]]) / (16 - tk[0])
                                 if 16 in self.tree_table else 1.73)
        self.commit_ms = commit_ms
        # Merging needs both proposals, and having both means paying for both. A block drafter is
        # 35 ms against a 140-164 ms step, so on a workload where the lookup drafter fires on 94 %
        # of blocks -- `quote` -- paying it every step is a fifth of the budget spent on a proposal
        # that loses. So the head is skipped where the lookup tree alone already beats what the
        # head has been worth lately. `always_head` turns that off, for measuring it.
        self.always_head = always_head
        # The per-block budget (the length router's `calc` mode): when set, every candidate tree is
        # cut to the node count that maximises its own committed tokens per millisecond on
        # `stair_table` (rows -> verify ms, the measured staircase), up to `node_budget` nodes,
        # instead of the fixed budget and the linear per-node price of `DraftTree.prune`.
        self.stair = False
        self.stair_table: dict[int, float] | None = None
        self.stair_opts = ("mtp", "chain", "ngram", "merged")     # the candidates the cut compares
        # the lookup's calibration per (source, match length) instead of one scalar, when set: the
        # one scalar settles at 0.36 on prose and 3.42 on quote because it stands in for the match
        # length it does not see
        self.stair_factor = None          # (rows, chain) -> measured / priced block time
        self.stair_price_fn = None        # (rows, chain) -> verify ms at the current context
        self.stair_snap: tuple | None = None   # node counts the cut may end at (None: any)
        self.stair_skip = False          # the head-skip rule priced on the staircase (off)
        self.stair_buckets = False
        self.calib_b: dict[tuple, _Rate] = {}
        self._lk = None
        # the lookup's per-level continuation rate per (source, match bucket), when set: a lookup
        # node's probability becomes its vote share times rate ** depth, with the rate learned from
        # how far the lookup's own line was followed (a geometric fit), in place of the fixed
        # 1 / (1 + alpha) decay a level and the one calibration scalar
        self.stair_lookrate = False
        self.look_rate: dict[tuple, list] = {}
        self._lk_depth = 0
        # the copy estimator (stair_rho): the lookup's per-level continuation rate as a Beta-binomial
        # per (source, match length 3..8, run bin of committed tokens that followed the lookup's
        # line), counted every round against what was committed, chosen or not; `rho_prior` holds
        # an offline prior per bucket as [successes, failures]
        # the cut's node probabilities recalibrated by (source, rank bucket) when set: the rank is
        # the node's place in the best-first order, and each bucket keeps realised on-path counts
        # against the probabilities it was priced at, every round, over the submitted tree
        self.stair_rankcal = False
        self.rank_stats: dict[tuple, list] = {}
        self._cal_last = None
        self.stair_rho = False
        self.rho_counts: dict[tuple, list] = {}
        self.rho_prior: dict[tuple, list] = {}
        self.copy_run = 0
        self._rho_key = None
        self._rho_top: list[int] | None = None
        # count the rate online from every committed block (off: the prior alone, as measured
        # before the lookup drafter recorded its top line)
        self.rho_online = True
        self.rho_cap = 4000.0           # trials per bucket before the counts are halved
        self.rho_min_bin = 0            # below this run bin the lookup keeps its alpha decay
        # ...and for a match the corpus shares (source both/corpus), below this run bin: a corpus
        # match with no copy run behind it is as often a common phrase of fresh text as a copy
        self.rho_nonlocal_min_bin = 0
        self.skip_min_bin = 0            # copy-run bin from which the lookup may replace the head
        self.rho_min_m = 3              # below this match length it keeps its alpha decay
        # Accepted tokens per call, kept directly rather than derived from a per-token rate. A
        # chained drafter's acceptance is prefix-geometric; a block drafter's is not -- measured,
        # its block of 7 at 46 % acceptance yields 3.2 tokens, which is 0.46 * 7 and not the 0.85
        # a geometric model would predict.
        self.mean_accepted = _Rate(0.5, alpha)
        # the verify curve the pricing runs on; FP8 by default, swapped for the NVFP4 one when the
        # MLPs are four bits, because four bits made width more expensive
        self.verify_table = dict(verify_ms_table) if verify_ms_table else dict(VERIFY_MS)
        keys = sorted(self.verify_table)
        self.prune_base_ms = self.verify_table[keys[0]]
        self.prune_per_node_ms = ((self.verify_table[keys[-1]] - self.verify_table[keys[0]])
                                  / max(keys[-1] - keys[0], 1))
        # how many of the tokens it expected did the lookup drafter actually get, lately
        self.calib = _Rate(1.0, alpha)
        self.calib_head = _Rate(1.0, alpha)
        # What the head's block has actually been accepting lately, in tokens. It starts at the
        # measured figure for the block drafter (4.30 over five workloads) so
        # that the head is asked until evidence says not to. That is the opposite of the 11:03
        # trap rather than a repeat of it: the optimism sits on the option that is NOT the one
        # whose counterfactual is free, and the lookup drafter's tree is built and scored against
        # the target every step whether or not the head runs.
        self.head_accepted = _Rate(head_prior, alpha)
        self.last_head_tree = None
        self.rate_mtp = _Rate(0.6, alpha)
        self.last = None
        self.last_n = 0
        self.last_expected = 0.0
        self.last_depth = 0          # the head chain length this step actually paid for
        self.last_tree = None        # what the lookup drafter offered, chosen or not
        # the head arm's per-token q rows when the request samples (None for the lookup
        # arm, whose proposal is deterministic and takes the p(d) accept).
        self.last_q = None
        self.stats = {"ngram": 0, "mtp": 0, "merged": 0, "ngram_tokens": 0, "mtp_tokens": 0,
                      "declined": 0, "depth_hist": {}}

    # --- state ---------------------------------------------------------------------------------

    def reset(self) -> None:
        self.ngram.reset()
        self.mtp.reset()
        self.last = None
        self.last_tree = None
        self.last_q = None
        self.copy_run = 0
        self._rho_key = None
        self._rho_top = None

    def set_sampling(self, sampler) -> None:
        """The lookup arm never samples; the head arm does when the request samples."""
        if hasattr(self.mtp, "set_sampling"):
            self.mtp.set_sampling(sampler)

    def prime(self, tokens: list[int]) -> None:
        self.ngram.prime(tokens)

    def state_snapshot(self):
        """The neural sub-drafter's cache. The lookup drafter is rebuilt from tokens by `prime`."""
        return ("merged", self.mtp.state_snapshot())

    def snapshot_bytes_per_token(self) -> int:
        from engine.cache import _drafter_bytes_per_token
        return _drafter_bytes_per_token(self.mtp)

    def state_restore(self, snap) -> None:
        kind, sub = snap
        if kind != "merged":
            raise ValueError(f"not a merged snapshot: {kind!r}")
        self.mtp.state_restore(sub)

    def can_resume(self) -> bool:
        return hasattr(self.mtp, "state_resume")

    def state_resume(self, n: int) -> None:
        fn = getattr(self.mtp, "state_resume", None)
        if fn is None:
            raise RuntimeError("the neural sub-drafter cannot resume in place")
        fn(n)

    def kv_views(self) -> list:
        fn = getattr(self.mtp, "kv_views", None)
        return fn() if fn is not None else []

    def sync(self, tokens, hidden, first_pos, rows=None) -> None:
        if getattr(self.mtp, "wants_rows", False):
            self.mtp.sync(tokens, hidden, first_pos, rows=rows)
        else:
            self.mtp.sync(tokens, hidden, first_pos)

    def observe(self, tokens: list[int]) -> None:
        """Learn from the block, including from the drafter that did not get to write it.

        The calibration factor is the only thing standing between the lookup drafter's own estimate
        and reality, and updating it only when the drafter is chosen is a trap: on a `quote`
        workload the drafter alone runs at 43.19 tok/s against the block drafter's 30.50, and the
        router picked it **zero times** out of nineteen blocks, so the factor never moved off 1.00
        and the drafter never got a turn. A policy that only learns about what it chose will keep
        choosing it.

        There is no need for exploration here. The tree was built before the choice was made, the
        tokens the target actually wrote are in this call, and `accepted_against` is a walk down a
        tree of at most sixteen nodes. The counterfactual is free, so it is taken on every step.
        """
        accepted = max(0, len(tokens) - 1)
        if self.last_tree is not None and self.last_expected > 0:
            would_have = self.last_tree.accepted_against(list(tokens))
            self.calib.update(would_have, self.last_expected)
            if self._lk is not None:
                self.calib_b.setdefault(self._lk, _Rate(1.0, self.calib.alpha)).update(
                    would_have, self.last_expected)
            if self.stair_lookrate and self._lk is not None:
                depth = max(self.last_tree.depths()) if self.last_tree.n_draft else 0
                st = self.look_rate.setdefault(self._lk, [2.5, 1.5])     # prior rate 0.625
                st[0] += would_have
                st[1] += 1.0 if would_have < depth else 0.0
        # The head's tree gets the same treatment, for the same reason: its node scores are a
        # softmax of a selector score that was never calibrated against anything, so what the
        # router needs from it is one scalar saying how optimistic it has been lately. It is free
        # here, and it is never conditioned on the head having been chosen.
        if self.last_head_tree is not None:
            got = self.last_head_tree.accepted_against(list(tokens))
            exp_head = self.last_head_tree.expected_accepted()
            if exp_head > 0:
                self.calib_head.update(got, exp_head)
            # And what it was worth in tokens, which is the number the skip decision is made on.
            # Only counted on steps where the head actually ran: on a step where it was skipped
            # there is no tree to score, and pretending otherwise would be the 11:03 trap with the
            # sign flipped.
            self.head_accepted.update(got, 1)
        self.last_head_tree = None
        if self.stair_rho and self.rho_online:
            self._rho_observe(list(tokens))
        if self.stair_rankcal:
            self._rank_observe(list(tokens))
        if self.last == "mtp" and self.last_n:
            self.rate_mtp.update(accepted, self.last_n)
            self.mean_accepted.update(accepted, self.last_n)
        self.last_tree = None
        self.ngram.observe(tokens)

    # --- pricing -------------------------------------------------------------------------------

    def _verify_ms(self, b: int) -> float:
        return verify_ms(b, self.verify_table)

    def _value(self, expected: float, nodes: int, cost_ms: float, p_reject: float) -> float:
        """Tokens per second if every step looked like this one."""
        ms = self._verify_ms(nodes + 1) + cost_ms + self.rollback_ms * p_reject
        return (expected + 1.0) / (ms / 1000.0)

    def _mtp_expected(self, d: int) -> float:
        """Expected accepted tokens from a chain of `d`, under a per-token acceptance rate.

        A chain is accepted prefix-first, so the i-th token only counts if the i-1 before it were
        accepted too: the expectation is the geometric sum, not `rate * d`. Getting this wrong is
        what makes a router propose long chains on text where it should propose short ones.
        """
        rate = self.rate_mtp.value
        return sum(rate ** i for i in range(1, d + 1))

    def _mtp_value(self, d: int) -> float:
        rate = self.rate_mtp.value
        cost = self.mtp_ms_per_token * d + self.head_fixed_ms
        if self.adaptive_depth:
            return self._value(self._mtp_expected(d), d, cost, 1.0 - rate ** d)
        expected = min(self.mean_accepted.value * d, float(d))
        return self._value(expected, d, cost, 1.0 if expected < d else 0.0)

    def _best_mtp_depth(self) -> tuple[int, float]:
        """The chain length worth proposing right now, priced rather than fixed.

        On German the head's acceptance is low enough that a three-token chain is *slower than not
        drafting at all*: it spends 50 ms of drafting and 22 ms of rollback to buy a third of a
        token (measured in the simulator over the recorded traces). A one-token chain on the same
        text still pays. So the depth is chosen per step.

        The router does not have the option of proposing nothing while it holds a head. The head
        conditions its next draft on the target's hidden state at the last committed position, and
        a step that verifies no block never computes that row (engine/spec.py). Depth one is the
        floor, and where even depth one does not pay, the honest reading is that the drafter is
        wrong for the text rather than that the router should switch off.
        """
        values = [(d, self._mtp_value(d)) for d in range(1, self.mtp_depth + 1)]
        best_v = max(v for _, v in values)
        # Ties go to the longer chain. The differences this is deciding between are a few per cent,
        # and the cost model's own terms are not known to a few per cent: the verify curve is flat
        # from B = 2 to B = 4 because the FP4 kernel tiles sixteen rows regardless, so tokens two
        # and three of a chain are close to free and their upside is real. Shortening on a margin
        # this thin is what cost 5 % of the prose row at 10:37.
        if not self.adaptive_depth:
            return self.mtp_depth, self._mtp_value(self.mtp_depth)
        best_d = max(d for d, v in values if v >= best_v * (1.0 - self.depth_margin))
        return best_d, next(v for d, v in values if d == best_d)

    # --- proposing -----------------------------------------------------------------------------

    def _calibrated_expected(self, tree) -> float:
        """Expected accepted length of a tree whose nodes came from two sources.

        Each source's node scores are its own kind of optimism -- counts out of a suffix store on
        one side, a softmax of a selector score on the other -- so each is scaled by the factor
        learned for it. Summing them is the identity used everywhere else here: a node is accepted
        only if its whole path is, so the expectation of a tree is the sum of its nodes' path
        probabilities.
        """
        total = 0.0
        for i in range(1, len(tree.tokens)):
            src = tree.source[i]
            if src.startswith("ngram"):
                total += tree.scores[i] * self.calib.value
            elif src.startswith("df2"):
                total += tree.scores[i] * self.calib_head.value
            else:
                total += tree.scores[i]     # already priced by the caller, e.g. the mtp chain
        return total

    def _source_calib(self, i: int, tree) -> float:
        src = tree.source[i]
        if src.startswith("ngram"):
            if self._lk is not None and self._lk in self.calib_b:
                return self.calib_b[self._lk].value
            return self.calib.value
        if src.startswith("df2"):
            return self.calib_head.value
        return 1.0

    @staticmethod
    def _run_bin(r: int) -> int:
        return 0 if r <= 0 else 1 if r < 8 else 2 if r < 24 else 3

    def _rho(self, src: str, m: int):
        key = (src, max(3, min(int(m), 8)), self._run_bin(self.copy_run))
        self._rho_key = key
        if key[2] < self.rho_min_bin or key[1] < self.rho_min_m:
            return None
        if src != "local" and key[2] < self.rho_nonlocal_min_bin:
            return None
        if key not in self.rho_prior and key not in self.rho_counts:
            return None                    # no evidence for this bucket: the alpha decay
        a, b = self.rho_prior.get(key, (5.0, 3.0))
        s_, f_ = self.rho_counts.get(key, (0.0, 0.0))
        return (a + s_) / (a + b + s_ + f_)

    def _rho_observe(self, tokens: list[int]) -> None:
        """Score the lookup's top line against the committed block, and extend the copy run.

        Counted every round the arm proposed, whichever candidate it submitted: the successes are
        the tokens the line predicted before the first miss, and a miss inside what the line knew
        is one failure. A block that ended before the line did, or a line shorter than the block,
        is censored (no failure). A round this arm did not propose (the deep chain) extends the
        run by what it committed and counts nothing.
        """
        top, self._rho_top = self._rho_top, None
        if top is None:
            self.copy_run = self.copy_run + len(tokens) if len(tokens) > 1 else 0
            return
        n = 0
        while n < len(top) and n < len(tokens) and top[n] == tokens[n]:
            n += 1
        if self._rho_key is not None and top:
            st = self.rho_counts.setdefault(self._rho_key, [0.0, 0.0])
            st[0] += n
            if n < len(tokens) and n < len(top):
                st[1] += 1.0                       # a miss inside what is known: not censored
            if st[0] + st[1] > self.rho_cap:       # forget slowly: a long server's text drifts
                st[0] *= 0.5
                st[1] *= 0.5
        full = bool(top) and n > 0 and n == min(len(top), len(tokens))
        self.copy_run = self.copy_run + len(tokens) if full else 0

    def _node_q(self, i: int, tree, depth: int, rank: int = 0) -> float:
        """A node's calibrated acceptance probability, as the cut adds it up."""
        if (self.stair_lookrate and self._lk is not None and tree.source[i].startswith("ngram")
                and self._lk in self.look_rate):
            a, b = self.look_rate[self._lk]
            rate = a / (a + b)
            return min(1.0, tree.scores[i] * ((1.0 + self.ngram.alpha) * rate) ** depth)
        if self.stair_rho and tree.source[i].startswith("ngram"):
            q = tree.scores[i]
        else:
            q = tree.scores[i] * self._source_calib(i, tree)
        if self.stair_rankcal and rank:
            y, qs = self.rank_stats.get(self._rank_key(tree.source[i], rank), (5.0, 5.0))
            q *= y / qs
        return min(1.0, q)

    @staticmethod
    def _rank_key(src: str, rank: int) -> tuple:
        b = 0 if rank <= 4 else 1 if rank <= 8 else 2 if rank <= 16 else 3 if rank <= 24 else 4
        return ("ngram" if src.startswith("ngram") else "head", b)

    def _rank_observe(self, tokens: list[int]) -> None:
        """The submitted tree's nodes against the committed block: on the path or not."""
        cal, self._cal_last = self._cal_last, None
        if cal is None:
            return
        tree, meta = cal
        kids: dict[int, dict[int, int]] = {}
        for i, p in enumerate(tree.parents[1:], start=1):
            kids.setdefault(p, {})[tree.tokens[i]] = i
        node, on = 0, set()
        for t in tokens:
            nxt = kids.get(node, {}).get(t)
            if nxt is None:
                break
            on.add(nxt)
            node = nxt
        for i, rank, q in meta:
            st = self.rank_stats.setdefault(self._rank_key(tree.source[i], rank), [5.0, 5.0])
            st[0] = 0.995 * st[0] + (1.0 if i in on else 0.0)
            st[1] = 0.995 * st[1] + q

    def _stair_cut(self, tree, cost_ms: float, chain: bool = False):
        """The prefix of `tree`, admitted best-first with its ancestors, whose calibrated
        expected yield per millisecond is highest on the staircase. Returns (tree, value)."""
        if tree is None or tree.n_draft == 0:
            return None, 0.0
        table = self.stair_table or self.tree_table
        curve = self.verify_table if chain else table
        order = sorted(range(1, len(tree.tokens)), key=lambda i: -tree.scores[i])
        rank_of = {n: r for r, n in enumerate(order, start=1)}
        dep = tree.depths()
        keep, gained = {0}, 0.0
        best_keep, best_v = None, 0.0
        for i in order:
            add = [n for n in tree.path(i) if n not in keep]
            if len(keep) - 1 + len(add) > self.node_budget:
                continue
            keep.update(add)
            gained += sum(self._node_q(n, tree, dep[n], rank_of.get(n, 0)) for n in add)
            n = len(keep) - 1
            v_ms = (self.stair_price_fn(n + 1, chain) if self.stair_price_fn is not None
                    else verify_ms(n + 1, curve))
            if chain:
                p_rej = min(1.0, max(0.0, 1.0 - gained / max(n, 1)))
                ms = v_ms + cost_ms + self.rollback_ms * p_rej
            else:
                ms = v_ms + cost_ms + self.commit_ms
            if self.stair_factor is not None:
                ms *= self.stair_factor(n + 1, chain)
            v = (min(gained, float(n)) + 1.0) / (ms / 1000.0)
            snap_ok = (self.stair_snap is None or n in self.stair_snap
                       or n == min(len(tree.tokens) - 1, self.node_budget))
            if v > best_v and snap_ok:
                best_v, best_keep = v, set(keep)
        if best_keep is None:
            return None, 0.0
        return tree.subset(best_keep), best_v

    def _stair_pick(self, head_tree, tree, spine, head_cost: float):
        """The block, with its node count chosen on the staircase: the head's tree, its chain,
        the lookup tree and the merge of the two, each cut to its best prefix, the best of all."""
        opts = []
        if head_tree is not None:
            opts.append((*self._stair_cut(head_tree, head_cost), "mtp"))
            ch = self._chain_of(head_tree)
            if ch is not None:
                opts.append((*self._stair_cut(ch, head_cost, chain=True), "chain"))
        if tree is not None:
            opts.append((*self._stair_cut(tree, 0.0), "ngram"))
            if head_tree is not None:
                opts.append((*self._stair_cut(head_tree.merge(tree), head_cost), "merged"))
        opts = [o for o in opts if o[0] is not None and o[2] in self.stair_opts]
        if not opts:
            self.last, self.last_n = None, 0
            self.stats["declined"] += 1
            return None
        best, _, label = max(opts, key=lambda o: o[1])
        if self.stair_rankcal and best is not None and best.n_draft:
            order = sorted(range(1, len(best.tokens)), key=lambda i: -best.scores[i])
            dep = best.depths()
            self._cal_last = (best, [(n, r, self._node_q(n, best, dep[n], 0))
                                     for r, n in enumerate(order, start=1)])
        if spine is not None and label == "mtp":
            # SAMPLED_CUT: the spine tree to the node profile the cut chose for the head's
            # deterministic tree (else the whole spine and its uncut quota, ~17 rows a block)
            best = spine.cut_like(best) if SAMPLED_CUT else spine
        elif spine is not None and label == "chain":
            sc = spine.spine_chain()
            n = best.n_draft + 1
            if SAMPLED_CUT and n < len(sc.tokens):
                from engine.tree import DraftTree
                sc = DraftTree(sc.tokens[:n], sc.parents[:n], sc.scores[:n], sc.source[:n],
                               q=sc.q[:n])
            best = sc
        self.last, self.last_n = label, best.n_draft
        self.stats[label] = self.stats.get(label, 0) + 1
        self.stats["ngram_tokens" if label == "ngram" else "mtp_tokens"] += best.n_draft
        self.stats["nodes"] = self.stats.get("nodes", 0) + best.n_draft
        return best

    def _tree_value(self, tree, cost_ms: float) -> float:
        """Tokens per second if every step verified this tree.

        The commit is unconditional, unlike the chain rollback: a tree has no single successor
        state, so the accepted path's state is always reconstructed. A tree block is therefore
        about 6 ms dearer than a chain block that accepted everything and about 6 ms cheaper than
        one that did not.
        """
        if tree is None or tree.n_draft == 0:
            return 0.0
        expected = min(self._calibrated_expected(tree), float(tree.n_draft))
        ms = verify_ms(tree.n_draft + 1, self.tree_table) + cost_ms + self.commit_ms
        return (expected + 1.0) / (ms / 1000.0)

    @staticmethod
    def _chain_of(tree):
        """The head's own best line out of its tree: its greedy walk, as a chain."""
        from engine.tree import DraftTree
        if tree is None or tree.n_draft == 0:
            return None
        node, toks, scores = 0, [], []
        while True:
            kids = [c for c in range(node + 1, len(tree.tokens)) if tree.parents[c] == node]
            if not kids:
                break
            node = max(kids, key=lambda c: tree.scores[c])
            toks.append(tree.tokens[node])
            scores.append(tree.scores[node])
        if not toks:
            return None
        return DraftTree.chain(tree.tokens[0], toks, scores=scores, source="df2-greedy")

    def _chain_value(self, tree, cost_ms: float) -> float:
        """The same pricing on the chain curve, with the rollback paid only on a rejection."""
        expected = min(self._calibrated_expected(tree), float(tree.n_draft))
        p_reject = min(1.0, max(0.0, 1.0 - expected / max(tree.n_draft, 1)))
        ms = verify_ms(tree.n_draft + 1, self.verify_table) + cost_ms + self.rollback_ms * p_reject
        return (expected + 1.0) / (ms / 1000.0)

    def _head_tree_steps(self, context: list[int], depth: int):
        """The head's proposal as a tree: its own if it builds one, otherwise its chain."""
        from engine.tree import DraftTree

        if hasattr(self.mtp, "propose_tree"):
            return (yield from tree_steps(self.mtp, context, self.head_budget))
        chain = self.mtp.propose(context, depth) if depth > 0 else []
        if not chain:
            return None
        # NOT scores of 1.0. A chain is accepted prefix-first, so the i-th token counts only if the
        # i-1 above it did; `DraftTree.chain`'s default says every token is certain, and the 11:03
        # entry is what that does to a router -- an option that claims eight accepted tokens before
        # a block has been verified cannot be outbid by anything. The prior here is the head's own
        # measured per-token rate, compounded, which is `_mtp_expected` written per node.
        rate = self.rate_mtp.value
        return DraftTree.chain(context[-1], chain,
                               scores=[rate ** (i + 1) for i in range(len(chain))], source="mtp")

    def propose_tree(self, context: list[int], k: int):
        return run_steps(self.propose_tree_steps(context, k))

    def propose_tree_steps(self, context: list[int], k: int):
        """The block this router wants verified, as a tree.

        Three shapes are priced against the measured tree curve and the widest one usually wins,
        because that curve is nearly flat below sixteen: the verify reads 19.4 GB of weights whether
        it carries two rows or sixteen, so a marginal node costs about 1.7 ms against a 140-164 ms
        step. That is the thesis of this track -- a sixteen-node tree verifies for what a chain of
        sixteen costs, and a chain of sixteen can only accept along one line.

          * the head's own tree -- the block drafter's lattice, its greedy path plus the best
            alternatives around it -- which costs one fixed call whatever its width;
          * the lookup drafter's tree, which costs a dictionary lookup;
          * the two grafted together, which costs the UNION of their nodes, since shared prefixes
            are shared nodes. The 11:18 measurement says that union is worth more than either part
            on `edit`, where the lookup supplies the verbatim stretches and the head the bridges.

        Whichever wins is pruned to the node budget, best-first under the marginal rule.
        """
        self.last_q = None                     # q-aware accept is a chain mechanism (v1)
        depth, _ = self._best_mtp_depth()
        self.stats["depth_hist"][depth] = self.stats["depth_hist"].get(depth, 0) + 1
        self.last_depth = depth
        # The lookup drafter is asked first because asking it is a dictionary lookup, 0.20 ms.
        self.ngram.rho_fn = self._rho if (self.stair and self.stair_rho) else None
        self._rho_key = None
        tree = self.ngram.propose_tree(context, min(k, self.ngram.max_depth))
        self.ngram.rho_fn = None
        self._rho_top = (list(getattr(self.ngram, "last_top", []) or [])
                         if self.stair_rho else None)
        self.last_tree = tree
        self.last_expected = tree.expected_accepted() if tree is not None else 0.0
        if (self.stair_buckets or self.stair_lookrate) and tree is not None:
            m = int(getattr(self.ngram, "last_match_len", 0))
            self._lk = (getattr(self.ngram, "last_source", "?"),
                        0 if m < 5 else 1 if m < 8 else 2 if m < 16 else 3)
        else:
            self._lk = None
        head_cost = self.head_fixed_ms + self.mtp_ms_per_token * depth
        v_ngram = self._tree_value(tree, 0.0)
        prior_ms = (verify_ms(self.node_budget + 1, self.tree_table) + head_cost + self.commit_ms)
        v_head_prior = (self.head_accepted.value + 1.0) / (prior_ms / 1000.0)
        if self.stair and self.stair_skip:
            # the skip priced as the cut prices: the lookup tree at its best staircase cut (its
            # continuation rates included) against the head's recent yield at a 16-row verify
            v_ngram = self._stair_cut(tree, 0.0)[1] if tree is not None else 0.0
            rows_ms = (self.stair_price_fn(16, False) if self.stair_price_fn is not None
                       else verify_ms(16, self.stair_table or self.tree_table))
            v_head_prior = ((self.head_accepted.value + 1.0)
                            / ((rows_ms + head_cost + self.commit_ms) / 1000.0))
            if self.stair_rho and self._run_bin(self.copy_run) < self.skip_min_bin:
                # no block has followed a lookup line yet: an 8-token match is as often a
                # coincidence of fresh text as the start of a copy, so the head drafts beside it
                v_ngram = 0.0
        if not self.always_head and tree is not None and v_ngram > v_head_prior:
            head_tree = None
            self.stats["head_skipped"] = self.stats.get("head_skipped", 0) + 1
        else:
            head_tree = yield from self._head_tree_steps(context, depth)
        # a sampled request's head tree carries its spine's q rows (`spine_tree`), and the
        # walk's rejection step is exact only if nothing chose it by the tokens it drew. So every
        # choice below -- and the calibration `observe` learns for the next ones -- is made on the
        # deterministic tree the same lattice builds, as a greedy request would make it; the spine
        # tree replaces that tree only where the head's own proposal wins.
        spine = head_tree if head_tree is not None and head_tree.q is not None else None
        if spine is not None:
            head_tree = getattr(self.mtp, "last_det_tree", None)
        self.last_head_tree = head_tree

        if self.stair:
            return self._stair_pick(head_tree, tree, spine, head_cost)
        best, v_best, label = head_tree, self._tree_value(head_tree, head_cost), "mtp"
        # The chain is one of the options, priced on the CHAIN curve, because a chain-shaped block
        # is 12.5 ms cheaper than a tree of the same width -- it has a kernel the tree does not
        # `forward_tree` sends a chain-shaped tree to `forward_block`, so
        # proposing a line is all this has to do to take the cheaper price.
        chain_tree = self._chain_of(head_tree)
        if chain_tree is not None:
            v_chain = self._chain_value(chain_tree, head_cost)
            if v_chain > v_best:
                best, v_best, label = chain_tree, v_chain, "chain"
        if v_ngram > v_best:
            best, v_best, label = tree, v_ngram, "ngram"
        if head_tree is not None and tree is not None:
            merged = head_tree.merge(tree)
            if merged.n_draft > self.node_budget:
                merged = merged.prune(self.node_budget, per_node_ms=self.tree_per_node_ms,
                                      base_ms=self.tree_base_ms)
            if self._tree_value(merged, head_cost) > v_best:
                best, label = merged, "merged"
        if spine is not None and label == "mtp":
            best = spine
        elif spine is not None and label == "chain":
            best = spine.spine_chain()
        if best is None:
            self.last, self.last_n = None, 0
            self.stats["declined"] += 1
            return None
        if best.n_draft > self.node_budget:
            best = best.prune(self.node_budget, per_node_ms=self.tree_per_node_ms,
                              base_ms=self.tree_base_ms)
        self.last, self.last_n = label, best.n_draft
        self.stats[label] = self.stats.get(label, 0) + 1
        self.stats["ngram_tokens" if label == "ngram" else "mtp_tokens"] += best.n_draft
        self.stats["nodes"] = self.stats.get("nodes", 0) + best.n_draft
        return best

    def propose(self, context: list[int], k: int) -> list[int]:
        """Chain interface, for the verify path that exists today: price both, take the better."""
        depth, v_mtp = self._best_mtp_depth()
        self.stats["depth_hist"][depth] = self.stats["depth_hist"].get(depth, 0) + 1
        self.last_depth = 0
        tree = self.ngram.propose_tree(context, min(k, self.ngram.max_depth))
        self.last_tree = tree
        self.last_expected = tree.expected_accepted() if tree is not None else 0.0
        if tree is not None and tree.n_draft:
            expected = self.last_expected * self.calib.value
            # a chain-only verify can only take one branch, so price the best one
            best = self._best_branch(tree, k)
            n = len(best)
            p_reject = min(1.0, max(0.0, 1.0 - expected / max(n, 1)))
            if self._value(min(expected, n), n, 0.0, p_reject) >= v_mtp:
                self.last_depth = 0
                self.last, self.last_n = "ngram", n
                self.last_q = None                    # a lookup proposal is deterministic
                self.stats["ngram"] += 1
                self.stats["ngram_tokens"] += n
                return best
        m = min(depth, k)
        if m <= 0:
            self.stats["declined"] += 1
            self.last, self.last_n = None, 0
            self.last_q = None
            return []
        draft = self.mtp.propose(context, m)
        self.last, self.last_n, self.last_depth = "mtp", len(draft), m
        self.last_q = getattr(self.mtp, "last_q", None)   # the head's q rows when sampling
        self.stats["mtp"] += 1
        self.stats["mtp_tokens"] += len(draft)
        return draft

    @staticmethod
    def _best_branch(tree, k: int) -> list[int]:
        node, out = 0, []
        while len(out) < k:
            kids = tree.children_of(node)
            if not kids:
                break
            node = max(kids, key=lambda c: tree.scores[c])
            out.append(tree.tokens[node])
        return out
