"""Choosing the BLOCK LENGTH once per step, and the drafter that goes with it.

Phase 5 measured both lengths on the same five prompts, with the reasoning block closed, and the
table is the whole argument for this file:

| workload | fine-tuned, 8 | fine-tuned, 16 |
|---|---|---|
| prose | 2.75 / 18.53 | 2.36 / 14.24 |
| chat  | 3.94 / 26.56 | 3.86 / 23.15 |
| code  | 5.45 / 36.95 | 6.10 / 36.55 |
| edit  | 7.64 / 52.62 | 13.26 / 80.56 |
| quote | 8.00 / 55.77 | 14.86 / 89.51 |

A sixteen-wide block verifies for 132.38 ms against an eight-wide one's 116.22, and its draft costs
about 32 ms against 26, so it has to commit **15.6 %** more tokens to break even. On `edit` it
commits 74 % more and on `quote` 86 %; on `prose` it commits 14 % fewer. Neither length is right,
and which one is right is a property of the text being written rather than of the model -- so the
engine should be choosing it, per step, and it needs no training to do so.

**The asymmetry that makes this cheap.** A block of sixteen that accepted `n` tokens tells you
exactly what a block of eight would have done with the same draft: `min(n, 7)`. The narrower block
is a prefix of the wider one, the verify pass computes the same rows for it, and the target's argmax
at row `i` does not depend on rows after `i`. So **every wide block scores the narrow option for
free**, and coming back down never costs an experiment.

The other direction is censored. A block of eight that accepted all seven is an observation that
the truth was "at least seven"; what sixteen would have committed is not in the data at all. That is
the 11:03 trap in a new costume -- *a policy that only learns about the option it took keeps taking
it* -- and the only honest fix is to pay for the information. So the router explores upward, and the
rate at which it explores is driven by the one signal that says information is worth buying: how
often the narrow block is hitting its own ceiling.

Three cheap signals, in the order they matter:

1. **the ceiling rate** -- the fraction of recent narrow blocks that accepted every slot. On `quote`
   this is 1.00 and on `prose` it is ~0.00, which is the regime distinction the table is made of;
2. **the free counterfactual** -- `min(n, 7)` on every wide block, which prices the narrow option
   without running it;
3. **the drafter's own lattice** -- once the wide drafter has run, the path probability of slots 8
   to 15 says whether the far half of the block is worth submitting at all, and that decision is
   free because the draft has already been paid for.

Everything is priced on the measured curve -- `verify(w)` from `tools/profile_block.py`, the draft
cost timed around the call the router itself makes, and the verify cost re-learned from the decode
loop that pays it (`on_verify`), because a policy constant belongs to the loop that pays it and a
microbenchmark once read a rollback at 23.4 ms that the loop reads at 6.4.
"""

from __future__ import annotations

import os
from engine.settings import SETTINGS as _S  # noqa: E402  (every QWEN38_* knob)
import time

from engine.drafters import Drafter, run_steps, tree_steps

# What a verify of `width` rows costs, anchor included -- the table the whole policy is priced on,
# and the phase-9 recalibration is that this table changed shape underneath it.
#
# Phase 5 measured 116.22 ms at eight rows and 132.38 at sixteen: the wide block cost 13.9 % more,
# and it had to commit 15.6 % more tokens to be worth taking. Phase 8's v2 kernel deleted both
# terms that were linear in the row count -- the strided activation gather and the split-K partial
# planes -- and the curve went flat. These are the numbers the decode loop itself paid on the five
# workloads (the `verify a/b ms` field of every `lenrouter` report):
#
#     width 8    99.1 ms          width 16    99.6 - 100.9 ms          +0.5 to +1.3 %
#
# A wide block is now about one per cent dearer and carries twice the slots. That is what
# `wide_default` below is: at this curve the question is no longer "is the wide block worth its
# price" but "is there any reason not to take it".
VERIFY_MS_B = {1: 95.56, 8: 99.1, 16: 99.8}
# Timed around `DFlash2Drafter.propose` in this file; these are only the priors the router starts
# from and it replaces them with its own measurements after a handful of blocks. Phase 9's runs
# read 25.2-25.3 ms at eight and 25.9-26.0 at sixteen, against phase 5's 26 and 32: the wide
# drafter's own forward stopped being the wider one when its head went through the same kernel.
DRAFT_MS_B = {8: 25.3, 16: 26.0}
ROLLBACK_MS_B = {8: 6.40, 16: 7.07}
# The same axis through `forward_tree`. 8 and 16 are phase 9's loop-measured numbers, as above. 32
# is NOT measured on the v2 kernel: it is the old table's 16-to-32 difference, 34.0 ms, carried on
# to the new level, and it exists only so the extrapolation above sixteen stays monotone. This
# router never asks for it -- its action set is the two widths the staircase has steps at, and 24
# nodes is a bucket trap.
TREE_MS_B = {8: 99.1, 16: 100.0, 32: 134.0}
# A tree pays its commit on every block rather than only on a rejection, because it has no single
# successor state (engine/router.py::TREE_COMMIT_MS).
TREE_COMMIT_MS_B = 6.6
# The tree verify's staircase at every row count that has a step (tools/verify_curve.py on the
# served kernels, graphs and 32-row verify): what the per-block budget of the `calc` mode prices
# a node count on. 17 rows open the second 16-row tile.
STAIR_MS = {8: 77.90, 16: 78.88, 17: 83.16, 20: 83.68, 24: 87.04, 28: 92.12, 32: 96.98}


def _head_of(child):
    """The block drafter inside whatever the arm is.

    An arm may be the `DFlash2Drafter` itself or a `MergedRouter` holding it as `.mtp` -- the
    combined configuration puts the lookup drafter's tree and the block drafter's lattice into one
    verify, and then the arm the length router picks is that pair rather than a bare drafter. Only
    three things are wanted from underneath: the block size, the tap, and the lattice.
    """
    return getattr(child, "mtp", child)


def verify_ms_b(width: int, table: dict[int, float] | None = None) -> float:
    """The measured verify cost of a block of `width` rows, anchor included.

    Interpolated between measured points and flat-extrapolated past the ends. The interpolation is
    known to be wrong in a specific way -- the FP4 kernel tiles sixteen rows, so the curve is a
    staircase and not a line (which cost 5 % of the prose row) -- and the
    router's action set is deliberately the two widths the staircase has steps at, so it never
    asks this function about a width between them.
    """
    table = table or VERIFY_MS_B
    keys = sorted(table)
    if width <= keys[0]:
        return table[keys[0]]
    if width >= keys[-1]:
        slope = (table[keys[-1]] - table[keys[-2]]) / (keys[-1] - keys[-2])
        return table[keys[-1]] + slope * (width - keys[-1])
    for lo, hi in zip(keys, keys[1:]):
        if lo <= width <= hi:
            f = (width - lo) / (hi - lo)
            return table[lo] + f * (table[hi] - table[lo])
    return table[keys[-1]]


class NoArm:
    """The narrow arm's place when the server runs the wide drafter alone (SPD-75, `--narrow-arm
    off`): no checkpoint, no context cache, nothing synced through a prefill.

    Under `QWEN38_LEN_MODE=wide` the narrow arm drafts nothing once the first block has been
    decided -- `_choose_wide` releases it there -- yet the served stack builds it, keeps its
    5 GiB context cache at a 262,144-token window and syncs it through every prefill. With this
    placeholder the router starts every request with the narrow arm already released, so every
    path that skips a released arm skips it, and the policy code is the one served with both.
    The block size is the narrow checkpoint's, which the router's tables are keyed by.
    """

    name = "none"

    def __init__(self, eng, block: int = 8):
        self.eng = eng
        self.cfg = type("Cfg", (), {"block_size": int(block)})()

    def reset(self) -> None:
        pass

    def prime(self, tokens) -> None:
        pass

    def propose(self, context, k):
        return []

    def observe(self, tokens) -> None:
        pass

    def release(self, free_cache: bool = False) -> None:
        pass

    def state_snapshot(self):
        return None

    def state_restore(self, snap) -> None:
        pass

    def state_resume(self, n: int) -> None:
        pass

    def kv_views(self) -> list:
        return []

    def snapshot_bytes_per_token(self) -> int:
        return 0


class _Est:
    """A sample mean that becomes an exponential one once it has seen enough.

    The prior matters here in a way it does not in a long-running estimator: a generation is thirty
    to ninety blocks, so an arm that is judged on its first two blocks is judged on noise, and an
    arm that never sheds its prior is not judged at all. Sample mean until `warm`, EMA after.
    """

    def __init__(self, init: float, alpha: float = 0.25, warm: int = 4):
        self.value = float(init)
        self.alpha = alpha
        self.warm = warm
        self.n = 0
        self._sum = 0.0

    def update(self, x: float) -> None:
        self.n += 1
        self._sum += float(x)
        if self.n <= self.warm:
            self.value = self._sum / self.n
        else:
            self.value += self.alpha * (float(x) - self.value)


class LengthRouter(Drafter):
    """Two block drafters at two block lengths, one of them chosen per step.

    `small` drafts an eight-wide block (seven proposals) and `large` a sixteen-wide one (fifteen).
    They are separate checkpoints -- the released weights were trained at eight and their top-1
    falls off past slot 7, so a wide block is only interesting with weights fine-tuned at that
    length -- and they are held at the same time. That costs 7.7 GB of the board's 121 and about
    1.5 ms a step of keeping both draft caches current; it buys the choice.

    Only ONE of them drafts on any given step. Running both would cost 26 to 32 ms of a 142 to
    164 ms block, which is a fifth of the budget spent on a proposal that is thrown away.
    """

    name = "lenrouter"
    wants_rows = True

    def __init__(self, small, large, *, alpha: float = 0.25,
                 explore_period: int = 32, ceiling_period: int = 4,
                 ceiling_trigger: float = 0.25,
                 verify_table: dict[int, float] | None = None,
                 draft_table: dict[int, float] | None = None,
                 width_trim: bool = True, fixed: int = 0, learn_cost: bool = True,
                 tree: bool = False, ngram=None, commit_ms: float = TREE_COMMIT_MS_B,
                 wide_default: bool = True, narrow_margin: float = 0.02,
                 narrow_warm: int = 4, narrow_probe_period: int = 4,
                 narrow_probe_after: int = 4, acc_warm: int = 8,
                 latch: bool = False, latch_after: int = 4, drop_idle: bool = False,
                 deep: int = 0, deep_order: int = 8, deep_share: float = 0.5,
                 deep_after: int = 2, tree_wide_after: int | None = None,
                 latch_price: bool | None = None,
                 latch_table: dict[int, float] | None = None,
                 probe_skip_hits: int = 2, switch: bool | None = None,
                 copy_min: int = 8, switch_margin: float = 0.05, switch_after: int = 2,
                 lazy_cap: int = 512, switch_mode: str = "b", dwell: int = 8,
                 max_switches: int = 4, exits: bool = True, max_nodes: int = 31,
                 stair_table: dict[int, float] | None = None, stair_fixed_ms: float = 15.0,
                 calc_start: str = "s", stair_temp: float = 1.4,
                 stair_snap: tuple | None = (7, 15, 23, 31), learn_block: bool = False,
                 class_tables: dict | None = None, rho_prior: dict | None = None,
                 stair_skip: bool | None = None, rho_online: bool = True):
        self.small = small
        self.large = large
        self.head_small = _head_of(small)
        self.head_large = _head_of(large)
        self.eng = self.head_small.eng
        self.w_small = int(self.head_small.cfg.block_size)
        self.w_large = int(self.head_large.cfg.block_size)
        if self.w_large <= self.w_small:
            raise ValueError(f"large block {self.w_large} must exceed small block {self.w_small}")
        self.alpha = alpha
        self.explore_period = int(explore_period)
        self.ceiling_period = int(ceiling_period)
        self.ceiling_trigger = float(ceiling_trigger)
        self.width_trim = bool(width_trim)
        # Whether the router replaces its measured priors with what this process actually pays.
        # On the board it should; in a test whose drafters return instantly it must not, or the
        # draft call prices at zero and the wide drafter looks free.
        self.learn_cost = bool(learn_cost)
        # 0 = route. 8 or 16 pins the router to one configuration, which is how the fixed-length
        # baselines are measured through exactly the same code as the routed one.
        self.fixed = int(fixed)
        # A measurement pin, off by default; see `_choose`.
        self.mix_period = 0

        # PHASE 9, second attempt: ONE decision per request instead of one per block.
        #
        # The first attempt priced the two widths correctly and still lost, and `mix3` says why: a
        # schedule that switches arms as often as the router does, while knowing nothing at all,
        # loses 3 % on chat, 10 % on code and 18 % on quote against never switching. Switching is
        # not free, and a policy that re-decides every block pays for it every block.
        #
        # So the shape changes rather than the pricing. Four wide blocks to measure the wide arm
        # and the ceiling, then up to four narrow ones where the ceiling says the narrow width has
        # room -- the probe the free counterfactual cannot replace, because it prices the WIDE
        # drafter's draft cut short and not the narrow checkpoint -- then one decision, kept for
        # the rest of the request. Two transitions instead of sixty.
        self.latch = bool(latch)
        self.latch_after = int(latch_after)
        self.latched: str | None = None

        # PHASE 10. What the arm that lost the latch goes on costing, and stopping it.
        #
        # After the decision the loser never drafts again, and the router keeps paying for it
        # anyway. Its draft cache is kept current by a `sync` on every committed block -- a
        # `project_context` and a `context_kv` over the whole checkpoint, about 1.5 ms of a 134 ms
        # block -- it holds every tapped row alive between steps, and it is a third of every state
        # snapshot the serving cache stores, which is a third of the entries that fit in the
        # budget. All of that buys an option the latch has already given up.
        #
        # So when the latch closes, the loser is released: no tap, no sync, and `None` in its slot
        # of the snapshot. Its checkpoint stays resident -- the latch is a belief about the TEXT
        # and `reset` throws it away, so the next request needs both arms from its first block and
        # a reload would cost seconds against a 760 ms time to first token. What is freed is the
        # per-request draft KV, which is what a snapshot is made of.
        #
        # One rule has to move with it, and it is a rule that outlived its measurement. A block
        # shorter than the wide width used to be routed to the narrow arm because a wide verify
        # cost 13.9 % more; on phase 8's kernel it costs about one per cent, and the only thing
        # that rule now does is send the last block of every generation to an arm that may have
        # been released -- where it would decline, and the tail would finish one token a step at
        # 95.56 ms each. Under `drop_idle` the latched arm keeps the tail.
        self.drop_idle = bool(drop_idle)
        self.idle: str | None = None
        if self.drop_idle and not self.latch:
            # A component that can never fire looks exactly like one that is switched off, and the
            # release only ever happens from `_choose_latched`. Refuse the combination rather than
            # accept a flag and do nothing with it -- that is the phase-9 trap as a constructor.
            raise ValueError("drop_idle needs latch=True: the release happens when the latch "
                             "closes, and without the latch there is nothing to release")

        # PHASE 9. Which arm the router falls back to when nothing argues against it.
        #
        # Under phase 5's curve a wide block cost 13.9 % more than a narrow one and the honest
        # default was the cheap arm, with the expensive one bought on evidence. Under phase 8's
        # kernel a wide block costs about one per cent more and carries twice the slots, so the
        # default is the wide arm and the narrow one is what has to argue for itself.
        #
        # This is not a change of taste. It removes the failure the phase-8 gate caught: on `chat`
        # the old rule declined the wide block 59 times in 66 and finished 8.04 % behind a fixed
        # sixteen, because a rule written for a 13.9 % price gap keeps refusing to pay a 1 % one.
        self.wide_default = bool(wide_default)
        # How much better the narrow arm has to look before the router comes down, in committed
        # tokens per millisecond. It is not a safety cushion: the narrow number the comparison uses
        # is measured by TRUNCATING wide blocks, which is exact for a chain and OPTIMISTIC for a
        # tree -- the narrow tree holds the accepted path only when that path's nodes ranked inside
        # the smaller budget. The margin is the size of that optimism, and 2 % is twice the cost
        # gap the two widths now differ by, so a tie always resolves wide.
        self.narrow_margin = float(narrow_margin)
        # Blocks of the narrow arm's own before its own history displaces the free counterfactual.
        #
        # The counterfactual is free but it is not the same measurement. It prices the WIDE
        # drafter's draft truncated to eight, and what the router would actually run is the NARROW
        # drafter, which is a different checkpoint fine-tuned at that length and better over its
        # own seven slots: on `prose` the narrow arm commits 2.72 a block where the wide arm's
        # blocks truncate to 2.60, and 4.6 % is the whole of the gap between the two fixed
        # baselines there. So the narrow arm is probed -- a bounded number of times, and only
        # where the ceiling rate says it has a chance -- rather than judged on the wide arm's
        # draft for ever.
        self.narrow_warm = int(narrow_warm)
        self.narrow_probe_period = int(narrow_probe_period)
        self.narrow_probe_after = int(narrow_probe_after)

        # In tree mode the block is verified through `forward_tree`, which is a different curve and
        # a different rollback: the commit is unconditional. Everything else about the policy is
        # the same, because the question is the same -- how many rows to put in one verify.
        self.tree = bool(tree)
        self.commit_ms = float(commit_ms)
        # One lookup drafter for both arms. Sharing it is not an optimisation: its suffix index is
        # updated in `observe`, and two arms each observing every block would index every token
        # twice and make the store disagree with the text it is supposed to be a memory of.
        self.ngram = ngram
        vt = dict(verify_table or (TREE_MS_B if tree else VERIFY_MS_B))
        dt = dict(draft_table or DRAFT_MS_B)
        self.vms = {w: _Est(verify_ms_b(w, vt), alpha, warm=6)
                    for w in (self.w_small, self.w_large)}
        self.dms = {"s": _Est(dt.get(self.w_small, 26.0), alpha, warm=6),
                    "l": _Est(dt.get(self.w_large, 32.0), alpha, warm=6)}
        self.rb = dict(ROLLBACK_MS_B)

        # Committed tokens a block, including the model's own bonus token, per configuration.
        # ("l", small) is the wide drafter's block submitted narrow -- the option the free
        # counterfactual prices, and the one `width_trim` takes.
        # `acc_warm` blocks of sample mean before the exponential one takes over, and it is longer
        # than the default six because of WHICH arm the difference falls on. Acceptance at sixteen
        # slots is heavier-tailed than at seven -- most blocks commit three or four and the rare
        # one commits fifteen, which is where the wide arm's advantage on `code` and `quote`
        # actually lives -- and an exponential mean forgets a tail faster than it forgets a body.
        # Both arms are estimated the same way, so the bias is not symmetric: it costs the wide arm
        # its outliers and the narrow arm nothing.
        self.acc_warm = int(acc_warm)
        self.acc = {("s", self.w_small): _Est(3.2, alpha, warm=self.acc_warm),
                    ("l", self.w_large): _Est(3.6, alpha, warm=self.acc_warm),
                    ("l", self.w_small): _Est(3.2, alpha, warm=self.acc_warm)}
        # How often the narrow block accepted every slot it had: the signal that says the truth is
        # censored and information about the wide block is worth buying.
        self.ceiling = _Est(0.0, alpha)
        # How often a WIDE block ended in a rejection, measured rather than modelled.
        #
        # This exists because of what the flat verify curve did to the rest of the pricing. When a
        # wide verify cost 13.9 % more than a narrow one, everything else in `_cost_ms` was noise
        # around that term; now the two verifies differ by about one per cent and the ROLLBACK term
        # is the largest thing left that distinguishes the arms. It used to be modelled as
        # `1 - expected/(width - 1)`, which on the same acceptance says a narrow block rejects 41 %
        # of the time and a wide one 73 % -- when in truth a block that commits four tokens out of
        # a possible seven and one that commits four out of fifteen have BOTH rejected, every time.
        # The model was reading "how much of the block was wasted" as "how often was anything
        # wasted", and it handed the narrow arm a 3 % discount it had not earned.
        self.rej = _Est(0.9, alpha)
        # How optimistic the wide drafter's own lattice has been, as one scalar.
        self.calib = _Est(1.0, alpha, warm=6)

        self.since = {self.w_small: 0, self.w_large: 0}
        self.blocks = 0
        # the chosen arm's per-token q rows when the request samples.
        self.last_q = None
        self.last_key = None
        self.last_width = 0
        self.last_depth = 0
        self.last_expected = 0.0
        self.last_forced = False
        # SPD, 2026-09-23: the deep chain. When a wide block has just committed its whole width and
        # the lookup drafter holds a long exact continuation of THIS request's text, the next block
        # is that continuation alone, up to `deep` rows -- a width no block drafter here was trained
        # for, and one only reproduction text can fill. 0 is off, and off is the router it was.
        self.deep = int(deep)
        self.deep_order = int(deep_order)
        self.deep_share = float(deep_share)
        # how many full blocks in a row before a deep chain is asked for. The
        # prototype fired after one, and one full wide block happens on new text too -- the row's
        # store-off requests fill a wide block now and then, and a deep chain that breaks there is a
        # 32-row verify for a handful of tokens. Two in a row is a copy; the row does not reach it.
        self.deep_after = max(1, int(deep_after))
        self.full_run = 0
        # the wide arm's tree past its block size only once the request has
        # committed this many tokens (0 = from the first block). A 24-node tree's verify costs ~8 %
        # more a block; on a short answer -- the row's max is a 41-token one committing 13-14 tokens
        # a block down the greedy line -- there is nothing for the branches to buy. Until then the
        # wide arm builds its served chain. `wide_budget` is the budget it grows into.
        if tree_wide_after is None:
            tree_wide_after = int(_S.get("TREE_WIDE_AFTER") or 0)
        self.tree_wide_after = max(0, int(tree_wide_after))
        self.wide_budget = getattr(large, "node_budget", None)
        self.req_tokens = 0
        self.last_deep = False
        self.last_full = False
        # The latch priced on the tree each arm will run after the decision (`QWEN38_LATCH_PRICE`,
        # off by default; off is the router it was).
        #
        # What the latch compares is committed tokens per millisecond, and the milliseconds it had
        # were the ones `on_verify` learns. On the served loop those are the time to LAUNCH a
        # graphed verify, not to run it: the published row's reports read 0.1-0.3 ms for a verify
        # that costs 79-87 ms. So the comparison was tokens per (draft + commit), close to tokens
        # per block, and the 24-node tree the wide arm grows into after `tree_wide_after` tokens
        # (87.0 ms against 78.9 on the served curve) was never on the scale. With the flag the
        # verify is read off the measured curve at the row count each arm will submit for the
        # rest of the request, the draft from this process's own timings with compile and capture
        # outliers left out, and a tie goes to the cheaper arm. The skip of the narrow probes
        # needs `probe_skip_hits` blocks that filled the narrow width, not one: one full block is
        # an opening line, two are a copy.
        if latch_price is None:
            latch_price = _S.get("LATCH_PRICE") == "1"
        self.latch_price = bool(latch_price)
        self.latch_table = dict(latch_table) if latch_table else dict(vt)
        self.probe_skip_hits = max(1, int(probe_skip_hits))
        self.draft_prior = {"s": float(dt.get(self.w_small, 26.0)),
                            "l": float(dt.get(self.w_large, 32.0))}
        self.dms_latch = {"s": _Est(self.draft_prior["s"], alpha, warm=6),
                          "l": _Est(self.draft_prior["l"], alpha, warm=6)}
        self.k_left = 0
        # Per-block arm choice with a lazy catch-up of the arm that is not drafting
        # (`QWEN38_LEN_SWITCH`, off by default; off is the router it was).
        #
        # The latch decides once a request because switching arms per block measured as a loss
        # (`mix3`). That control switched blindly, so it paid the narrow arm's lower ceiling on
        # copy text as well as any cost of the switch itself; the arm that did not draft was kept
        # current on every block either way. What a switch does cost is keeping both draft caches
        # current: the per-block sync of the arm that is not drafting, which `--drop-idle` saves
        # after the latch (+4 % mean on the row). With the flag the arm that is not drafting is not
        # synced per block. Its rows of the target's hidden states are kept, and it is brought up
        # to date in ONE sync when it is chosen again (WhiFlash's lazy catch-up), so a switch
        # costs one batched sync and the choice can follow the text inside a request.
        #
        # The choice itself reads only what the loop measures: the lookup drafter's local match
        # (the text is being copied from the context when a long suffix of it recurs), the narrow
        # width's ceiling hits, the free price of the narrow width from every wide block, and the
        # price of each arm on the measured curve at the rows it verifies now.
        if switch is None:
            switch = _S.get("LEN_SWITCH") == "1"
        self.switch = bool(switch)
        self.copy_min = int(copy_min)
        self.switch_margin = float(switch_margin)
        self.switch_after = max(1, int(switch_after))
        self.lazy_cap = int(lazy_cap)
        # "b": the latch with two exits and a budget ladder (the default of the flag); "narrow":
        # narrow first, re-asked every block (kept for the replay's ablation). `exits=False` is
        # design C, the latch and the ladder alone.
        mode_env = _S.get("LEN_MODE")
        if mode_env:
            switch_mode = mode_env
        if switch_mode not in ("b", "narrow", "calc", "wide"):
            raise ValueError(f"switch_mode {switch_mode!r}: 'b', 'narrow', 'calc' or 'wide'")
        self.switch_mode = switch_mode
        self.dwell = int(dwell)
        self.max_switches = int(max_switches)
        self.exits = bool(exits)
        self._b_reset()
        # "calc": the node count of every block from the staircase (each arm's MergedRouter cuts
        # its candidates to the count that maximises calibrated committed tokens per ms), and the
        # arm from realised tokens per ms, narrow first, inside the dwell and the switch cap.
        # ENG-213: QWEN38_STAIR_MAX_NODES raises the staircase cut's ceiling past a 32-row verify
        # (63 = a 64-row step, which needs QWEN38_VERIFY_ROWS 64 and a staircase measured that far)
        self.max_nodes = int(_S.get("STAIR_MAX_NODES") or max_nodes)
        self.calc_start = calc_start
        self.stair_table = dict(stair_table or STAIR_MS)
        # "wide": the wide checkpoint on every block, its node count from the staircase, and the
        # narrow one released for the request (the replay's best policy: at a per-block budget
        # cut best-first from a 31-node tree the wide lattice beats the narrow one on every class).
        self.calc = self.switch and switch_mode in ("calc", "wide")
        # SPD-75: the wide arm alone. Only the wide policy never needs the narrow arm after its
        # first decision; every other mode would choose an arm that is not there.
        self.narrow = not isinstance(small, NoArm)
        if not self.narrow and not (self.switch and switch_mode == "wide"):
            raise ValueError("the wide arm alone (--narrow-arm off) needs QWEN38_LEN_SWITCH=1 and "
                             f"QWEN38_LEN_MODE=wide; this router is switch={self.switch}, "
                             f"mode={switch_mode!r}")
        if self.calc:
            for arm in (small, large):
                if hasattr(arm, "stair"):
                    arm.stair = True
                    arm.stair_table = dict(self.stair_table)
                    arm.node_budget = arm.head_budget = self.max_nodes
                    # the fixed part of a block the cut prices against: the draft call and the
                    # host, as the loop pays them (the served 27 ms prior is the old head's)
                    arm.head_fixed_ms = float(stair_fixed_ms)
                    # the cut ends at the top of a 16-row tile or at the whole candidate: at most
                    # four tree shapes a context class, and the tile's last rows are nearly free
                    arm.stair_snap = tuple(stair_snap) if stair_snap else None
            # the lattice's softmax temperature for the cut: a flatter path probability ranks the
            # near-root siblings the cut buys (the replay: 1.0 -> 1.4 is +1.3 % over all classes,
            # 2.5 loses); the fixed-budget trees keep QWEN38_DF2_TEMP
            for head in (self.head_small, self.head_large):
                if hasattr(head, "tree_temp"):
                    head.tree_temp = float(stair_temp)
        # What a block really costs, learned in the loop: the time from one proposal to the next is
        # the whole round (draft, verify, commit, sync), and its ratio to the staircase's price is
        # kept per (chain or tree, row tile, context class). The cut prices on the staircase times
        # that ratio, so a tree that the staircase measured at a short context but that costs more
        # at 32k (its attention reads the context once per row) is priced as it is paid.
        self.block_factor: dict[tuple, _Est] = {}
        self._blk = None
        self.learn_block = bool(learn_block)
        # the draft's own cost by context class: the drafter attends over the whole context, so at
        # 32k it costs more than the short-context figure the cut and the head-skip rule price
        self.stair_fixed_ms = float(stair_fixed_ms)
        self.dms_cls: dict[tuple, _Est] = {}
        # The staircase by context class, chain and tree apart: {class: {"chain": {rows: ms},
        # "tree": {rows: ms}}}, measured by tools/verify_curve.py at a few context lengths and
        # interpolated in log2(context) between them. At 32k a tree's verify reads the context once
        # per row and a chain's does not, so one short-context table misprices both.
        if class_tables is None:
            path = _S.get("STAIR_TABLES")
            if path:
                import json
                with open(os.path.expanduser(path)) as f:
                    class_tables = json.load(f)
        self.class_tables = ({int(c): {k: {int(r): float(v) for r, v in t.items()}
                                       for k, t in d.items()}
                              for c, d in class_tables.items()} if class_tables else None)
        # The copy estimator: the lookup's per-level continuation rate by (source, match length,
        # run of committed tokens that followed its line), a Beta prior per bucket from an offline
        # uncensored count, for matches of 8 tokens (the longest the local index keeps).
        if rho_prior is None:
            path = _S.get("STAIR_RHO")
            if path:
                import json
                with open(os.path.expanduser(path)) as f:
                    raw = json.load(f)
                rho_prior = {}
                for k, v in raw.items():
                    if k.startswith("_"):          # provenance, not a bucket
                        continue
                    s_, f_ = v
                    src, m, rb = k.split("|")
                    if s_ + f_ >= 20:              # a thin bucket keeps the alpha decay
                        w = min(50.0, s_ + f_)
                        r_ = s_ / (s_ + f_)
                        rho_prior[(src, int(m), int(rb))] = [w * r_, w * (1.0 - r_)]
        if stair_skip is None:
            stair_skip = _S.get("STAIR_SKIP") == "1"
        if self.calc and stair_skip:
            for arm in (small, large):
                if hasattr(arm, "stair_skip"):
                    arm.stair_skip = True
        if self.calc:
            for arm in (small, large):
                if hasattr(arm, "stair") and rho_prior:
                    arm.stair_rho = True
                    arm.rho_prior = dict(rho_prior)
                    arm.rho_min_m = 8
                    arm.rho_online = bool(rho_online)
                    arm.skip_min_bin = 1
            for arm in (small, large):
                if hasattr(arm, "stair"):
                    if self.learn_block:
                        arm.stair_factor = self._stair_factor
                    if self.class_tables:
                        arm.stair_price_fn = self._class_price
        self.cur: str | None = None
        self.want_run = 0
        self.recent_hits: list[int] = []
        self.backlog: dict[str, list] = {"s": [], "l": []}
        # One bound method, held. `self._on_tap is self._on_tap` is False in CPython -- a bound
        # method is built fresh on every attribute access -- so an attach/detach pair that compares
        # them with `is` never detaches anything.
        self._tap_cb = self._on_tap
        self.stats = {"blocks": 0, "small": 0, "large": 0, "trims": 0, "forced": 0,
                      "probes": 0, "tokens_small": 0, "tokens_large": 0, "declined": 0,
                      "ceiling_hits": 0, "latched": "-", "idle": "-", "width_hist": {},
                      # What each verified block committed, by the arm that proposed it, and how
                      # often a block used up everything it could have committed: its arm's whole
                      # width, or the depth of the draft it was actually handed (a tree's prune can
                      # stop short of the arm). The first is the question a deeper draft answers.
                      "commit_hist": {}, "cap_arm": 0, "cap_depth": 0, "deep": 0,
                      "deep_tokens": 0}
        if self.latch_price:
            # the prices the latch decided on, narrow/wide ms (a string, so `reset` clears it)
            self.stats["price"] = "-"
        self._no_narrow()
        self.attach()

    # --- the tap, shared -------------------------------------------------------------------

    def attach(self) -> None:
        """One tap, two consumers.

        `DFlash2Drafter.attach` overwrites `eng.tap` with its own callback, so two of them cannot
        both be attached; the engine emits one row per layer per forward and whoever holds the tap
        gets it. The router holds it and hands each row to both, which is what lets the drafter
        that did NOT propose this step stay current -- and it has to stay current, because its own
        cache is indexed by absolute position and a gap in it makes it decline for ever after.
        """
        self.eng.tap = self._tap_cb

    def detach(self) -> None:
        if self.eng.tap is self._tap_cb:
            self.eng.tap = None

    def _on_tap(self, h) -> None:
        if self.idle != "s":
            self.head_small._on_tap(h)
        if self.idle != "l":
            self.head_large._on_tap(h)

    def tap_indices(self):
        """The union of the tap invocations the arms that receive the tap keep; None when an arm
        cannot say (engine/model.py `prefill_window` then runs chunk-major)."""
        out = set()
        for key, head in (("s", self.head_small), ("l", self.head_large)):
            if self.idle == key:
                continue
            fn = getattr(head, "tap_indices", None)
            if fn is None:
                return None
            got = fn()
            if got is None:
                return None
            out |= set(got)
        return out

    def tapped(self):
        """The tapped rows of an arm that still receives the tap (`DFlash2Drafter.tapped`)."""
        for key, head in (("s", self.head_small), ("l", self.head_large)):
            fn = getattr(head, "tapped", None)
            got = fn() if self.idle != key and fn is not None else None
            if got is not None:
                return got
        return None

    # --- the idle arm ----------------------------------------------------------------------

    def _release_idle(self, key: str) -> None:
        """Stop paying for the arm the latch just gave up on.

        Called exactly once a request, from the step that closes the latch. It is deliberately not
        an unload: `release` frees the draft KV and leaves the checkpoint where it is, because the
        next request starts with no latch and needs both arms again.
        """
        if not self.drop_idle or self.idle is not None:
            return
        self.idle = key
        head = self.head_small if key == "s" else self.head_large
        rel = getattr(head, "release", None)
        if rel is not None:
            rel()
        self.stats["idle"] = key

    def _alive(self, key: str) -> str:
        """`key` if that arm can still draft, and the other one if it cannot."""
        if self.idle is None or key != self.idle:
            return key
        return "l" if key == "s" else "s"

    # --- drafter interface -----------------------------------------------------------------

    def reset(self) -> None:
        """A new request: the drafters' caches go, and so does the evidence about this text.

        What the router has learned splits cleanly in two. The COSTS -- what a verify of eight or
        sixteen rows takes, what a draft call takes -- are properties of the board and they are
        kept, so a long-lived server's constants get better rather than being thrown away every
        request. The ACCEPTANCE is a property of the text being written, and carrying one request's
        beliefs into the next would start a fresh-prose request with a quotation's policy. So the
        arms, the ceiling rate and the calibration are cleared.
        """
        self.small.reset()
        self.large.reset()
        for key, width in list(self.acc):
            init = 3.6 if width == self.w_large else 3.2
            self.acc[(key, width)] = _Est(init, self.alpha, warm=self.acc_warm)
        self.ceiling = _Est(0.0, self.alpha)
        self.rej = _Est(0.9, self.alpha)
        self.calib = _Est(1.0, self.alpha, warm=6)
        self.since = {self.w_small: 0, self.w_large: 0}
        self.blocks = 0
        self.last_key = None
        self.last_width = 0
        self.last_full = False
        self.full_run = 0
        self.req_tokens = 0
        self.last_deep = False
        # The latch is a belief about the text, so it goes with the arms rather than with the
        # costs: a new request starts by measuring again.
        self.latched = None
        # ... and so both arms come back. `DFlash2Drafter._build` reallocates the draft KV of one
        # that was released, lazily, on its first sync.
        self.idle = None
        self.cur = None
        self._blk = None
        self.want_run = 0
        self.recent_hits = []
        self.backlog = {"s": [], "l": []}
        self._b_reset()
        # per request, so `report()` after a generation describes that generation
        self.stats = {k: ({} if isinstance(v, dict) else ("-" if isinstance(v, str) else 0))
                      for k, v in self.stats.items()}
        self.last_q = None
        self._no_narrow()

    def _no_narrow(self) -> None:
        """Without the narrow arm every request starts with it released (SPD-75)."""
        if not self.narrow:
            self.idle = "s"
            self.stats["idle"] = "s"
            self.stats["latched"] = "l"

    def set_sampling(self, sampler) -> None:
        """Both arms can sample their proposals; the length policy is unchanged."""
        for arm in (self.small, self.large):
            if hasattr(arm, "set_sampling"):
                arm.set_sampling(sampler)

    def prime(self, tokens: list[int]) -> None:
        if self.ngram is not None:
            # Primed once, through the object both arms share, for the reason `self.ngram` is
            # shared at all: priming it twice would put every prompt position in the index twice.
            self.ngram.prime(tokens)
            return
        for d in (self.small, self.large):
            if hasattr(d, "prime"):
                d.prime(tokens)

    def state_snapshot(self):
        """Both drafters' caches. The POLICY is not in here, on purpose.

        The arms are beliefs about the text being written and `reset` throws them away at every
        request for the reason its own docstring gives; a state cache that carried them back in
        would undo that through the back door. What the cache restores is the two draft KVs, which
        are facts about positions rather than beliefs about a workload.

        A RELEASED arm is `None` here rather than stale. Its cache stopped being written the moment
        the latch closed, so it covers a prefix of the positions this snapshot is for -- and `sync`
        sets `ctx_len` from the chunk it was last given rather than from the first hole, so an arm
        restored as if it were current would draft against zeros for the positions nobody wrote and
        nothing would report it, because verification is exact either way. The honest snapshot is
        the one that says which arm is in it.
        """
        self._flush()
        return ("lenrouter",
                None if self.idle == "s" else self.small.state_snapshot(),
                None if self.idle == "l" else self.large.state_snapshot(),
                self.idle)

    def snapshot_bytes_per_token(self) -> int:
        """The arms `state_snapshot` carries, per position: both until the latch releases one."""
        from engine.cache import _drafter_bytes_per_token
        return sum(_drafter_bytes_per_token(d) for key, d in (("s", self.small), ("l", self.large))
                   if key != self.idle)

    def state_restore(self, snap) -> None:
        kind, small, large = snap[0], snap[1], snap[2]
        idle = snap[3] if len(snap) > 3 else None
        if kind != "lenrouter":
            raise ValueError(f"not a lenrouter snapshot: {kind!r}")
        self.backlog = {"s": [], "l": []}
        if small is not None:
            self.small.state_restore(small)
        if large is not None:
            self.large.state_restore(large)
        # An arm the snapshot does not carry cannot be made current without forwarding the prefix
        # that this restore exists to skip. So the request inherits the decision the snapshot was
        # taken under: the arm that IS in it, latched from the first block. This is a continuation
        # of the text that argued for that arm, which is the same evidence the latch would have
        # spent its first eight blocks gathering again.
        if idle is not None:
            self.idle = idle
            self.latched = "l" if idle == "s" else "s"
            self.stats["latched"] = self.latched
            self.stats["idle"] = idle

    def can_resume(self) -> bool:
        return all(hasattr(d, "state_resume") or not hasattr(d, "sync")
                   for d in (self.small, self.large))

    def state_resume(self, n: int) -> None:
        """Both arms pick up at `n` from the draft KV already in place (`ResidentPrefix`). Called
        after `reset`, so no arm is idle: both were synced by the prefill that wrote those rows."""
        self.backlog = {"s": [], "l": []}
        for key, d in (("s", self.small), ("l", self.large)):
            if key != self.idle and hasattr(d, "state_resume"):
                d.state_resume(n)

    def end_mark(self) -> str | None:
        """At the end of a generation, for the resident prefix's end anchor: every arm brought up
        to the committed rows (the lazy backlog synced), and the arm the latch released, whose
        draft KV stopped being written at the latch."""
        self._flush()
        return self.idle

    def resume_end(self, n: int, idle: str | None) -> None:
        """Pick up at the end anchor `n`. The released arm's draft KV has no rows for what the
        generation wrote after the latch, so the request inherits the latch, as `state_restore`
        does for a session snapshot: the arm that was drafting, latched from the first block."""
        if idle is not None:
            self.idle = idle
            self.latched = "l" if idle == "s" else "s"
            self.stats["latched"] = self.latched
            self.stats["idle"] = idle
        self.state_resume(n)                 # the released arm stays as `reset` left it

    def kv_views(self) -> list:
        self._flush()
        out = []
        for d in (self.small, self.large):
            fn = getattr(d, "kv_views", None)
            if fn is not None:
                out += fn()
        return out

    def sync(self, tokens, hidden, first_pos, rows=None) -> None:
        lag = self._lagging() if self.switch else None
        for key, d in (("s", self.small), ("l", self.large)):
            if key == self.idle:
                continue
            if key == lag and self._defer(key, tokens, first_pos, rows):
                continue
            if getattr(d, "wants_rows", False):
                d.sync(tokens, hidden, first_pos, rows=rows)
            else:
                d.sync(tokens, hidden, first_pos)

    # --- the lazy catch-up (QWEN38_LEN_SWITCH) ---------------------------------------------

    def _lagging(self) -> str | None:
        """The arm that is not drafting now, once the request has a current arm. The prefill's
        sync (no block yet) goes to both."""
        if self.cur is None:
            return None
        return "l" if self.cur == "s" else "s"

    def _defer(self, key: str, tokens, first_pos: int, rows) -> bool:
        """Keep the committed rows of the target's hidden states for an arm that is not
        drafting, instead of syncing it. The tap hands both heads every row of the verify; only
        the rows on the accepted path are copied out, a few hundred kB a block."""
        head = self.head_small if key == "s" else self.head_large
        taps = getattr(head, "_tap_rows", None)
        n = len(tokens)
        # Every way out that syncs this block the ordinary way first brings the arm up to date:
        # its cache is indexed by absolute position and `sync` sets its length from the last
        # chunk, so a block synced ahead of an older backlog would be undone by the catch-up.
        if not taps or any(t is None for t in taps) or n == 0:
            self._catch_up(key)
            return False
        import torch
        if rows is None:
            if taps[0].shape[0] < n:
                self._catch_up(key)
                return False
            kept = [t[:n].clone() for t in taps]
        else:
            if len(rows) != n:
                self._catch_up(key)
                return False
            sel = torch.as_tensor(rows, dtype=torch.long, device=taps[0].device)
            kept = [t[sel] for t in taps]
        bl = self.backlog[key]
        if bl and bl[-1][0] + len(bl[-1][1]) != first_pos:
            # not contiguous with what is kept: bring the arm up to date first, then this block
            self._catch_up(key)
            return False
        bl.append((int(first_pos), [int(x) for x in tokens], kept))
        if sum(len(e[1]) for e in bl) >= self.lazy_cap:
            self._catch_up(key)
        return True

    def _catch_up(self, key: str) -> None:
        """One sync over every position the arm missed, from the rows kept for it."""
        bl = self.backlog[key]
        if not bl:
            return
        import torch
        head = self.head_small if key == "s" else self.head_large
        arm = self.small if key == "s" else self.large
        first = bl[0][0]
        toks = [t for e in bl for t in e[1]]
        taps = [torch.cat([e[2][j] for e in bl], dim=0) for j in range(len(bl[0][2]))]
        self.backlog[key] = []
        saved, head._tap_rows = head._tap_rows, taps
        try:
            if getattr(arm, "wants_rows", False):
                arm.sync(toks, None, first, rows=None)
            else:
                arm.sync(toks, None, first)
        finally:
            head._tap_rows = saved
        self.stats["catchups"] = self.stats.get("catchups", 0) + 1

    def _flush(self) -> None:
        for key in ("s", "l"):
            if self.backlog.get(key):
                self._catch_up(key)

    def on_verify(self, width: int, ms: float) -> None:
        """What the decode loop actually paid for the block it just verified.

        `engine/spec.py` calls this after the block's logits have been read back, so the number
        includes the synchronisation the loop was going to pay anyway and nothing it was not.
        """
        if self.learn_cost and ms > 0 and not self.last_deep:
            self.vms[self._arm(width, self.last_key)].update(ms)

    # --- pricing ---------------------------------------------------------------------------

    def _arm(self, width: int, key: str | None = None) -> int:
        """Which of the two arms a submitted width belongs to. A block from the NARROW drafter is
        the narrow arm's whatever its width: with a node budget past its block size its
        tree can be as wide as the wide arm's. A wide block's width still says which configuration
        it was -- the wide drafter submitted narrow is `("l", w_small)`, the free counterfactual."""
        if key == "s":
            return self.w_small
        return self.w_small if width <= self.w_small else self.w_large

    def _cost_ms(self, key: str, width: int, expected: float) -> float:
        base = self.vms[width].value + self.dms[key].value
        if self.tree:
            return base + self.commit_ms
        return base + self.rb.get(width, 6.5) * self._p_reject(width, expected)

    def _p_reject(self, width: int, expected: float) -> float:
        """How often a block of this width ends in a rollback.

        Measured wherever the loop has measured it, and the two measurements are already being
        kept for other reasons:

          * the NARROW arm rejects exactly when it does not accept every slot it has, which is
            `1 - ceiling` by the definition of the ceiling rate;
          * the WIDE arm's own rate is counted in `observe`.

        The modelled fallback below is only for the first blocks of a request, before either has a
        sample. It is the old formula and it is wrong in the way the `rej` comment describes, which
        is why it is a fallback and not the rule.
        """
        if self._arm(width) == self.w_small:
            if self.ceiling.n:
                return min(1.0, max(0.0, 1.0 - self.ceiling.value))
        elif self.rej.n:
            return min(1.0, max(0.0, self.rej.value))
        return min(1.0, max(0.0, 1.0 - expected / max(width - 1, 1)))

    def _value(self, key: str, width: int) -> float:
        """Committed tokens per millisecond if every step looked like this one."""
        est = self.acc[(key, width)]
        expected = max(est.value - 1.0, 0.0)          # accepted drafts, not committed tokens
        return est.value / self._cost_ms(key, width, expected)

    def _narrow_est(self) -> _Est:
        """The best estimate of what a NARROW block commits, and where it comes from.

        Two sources measure the same quantity and they are not equally good.

          * the narrow arm's own blocks -- exactly right, and only available when the router has
            been running narrow blocks, which under `wide_default` it mostly has not;
          * the free counterfactual -- every wide block truncated to the narrow width. It costs
            nothing, it is available on every block the router takes, and it is measured on the
            SAME text and the SAME step as the wide number it will be compared against, which is
            what makes the comparison paired rather than a comparison between two stretches of
            prose.

        The paired one is used until the narrow arm has enough of its own, because an unpaired
        comparison across a regime change is how a router ends up preferring the arm that happened
        to run during the easy paragraph.
        """
        own = self.acc[("s", self.w_small)]
        if own.n >= self.narrow_warm:
            return own
        cf = self.acc[("l", self.w_small)]
        return cf if cf.n else own

    def _narrow_value(self) -> float:
        """Committed tokens per millisecond if the router went narrow.

        The acceptance may be priced from a wide block, but the COST is the narrow arm's own: if
        the router comes down it runs the small drafter, and the small drafter's forward is what
        it will pay for.
        """
        est = self._narrow_est()
        expected = max(est.value - 1.0, 0.0)
        return est.value / self._cost_ms("s", self.w_small, expected)

    def _wide_value(self) -> float:
        """What running the WIDE drafter is worth, at whichever width it would be submitted at.

        Two configurations share one draft call, and the router does not have to choose between
        them before paying for it: the sixteen-wide submission, and the same draft truncated to
        eight. The second is not a hypothetical -- `_trim` plays it -- and its acceptance is known
        exactly on every wide block that has ever run, because a narrow block is a prefix of a wide
        one. So the value of asking the wide drafter is the better of the two, and the width itself
        is settled afterwards, when the drafter's own scores for slots 8 to 15 are in hand.
        """
        v = self._value("l", self.w_large)
        if self.width_trim and self.acc[("l", self.w_small)].n:
            v = max(v, self._value("l", self.w_small))
        return v

    # --- the choice ------------------------------------------------------------------------

    def _choose(self, k: int) -> str:
        if self.mix_period:
            # Not a policy. A pin that alternates on a fixed schedule with no evidence behind it,
            # so that a run which SWITCHES ARMS as often as the router does can be measured against
            # a run that never switches. The router has lost to a fixed sixteen on the mixed
            # workloads in three consecutive phases and the question that separates the two
            # explanations -- the router chooses badly, or switching itself costs something -- is
            # not answerable from a run where both happen at once.
            return "s" if self.blocks % self.mix_period == 0 else "l"
        if self.fixed == self.w_small:
            return "s"
        if self.fixed == self.w_large:
            return "l"
        if self.switch:
            if k < self.w_large - 1 and self.cur is not None:
                return self.cur                    # the tail stays on the arm that is current
            if k >= self.w_large - 1:
                if self.switch_mode == "b":
                    return self._choose_b()
                if self.switch_mode == "calc":
                    return self._choose_calc()
                if self.switch_mode == "wide":
                    return self._choose_wide()
                return self._choose_switch()
        if k < self.w_large - 1:
            # The tail of a generation, and a rule that outlived the measurement it was written
            # for: a wide block was routed narrow here because it cost 13.9 % more for slots the
            # token budget could not use. On phase 8's kernel it costs about one per cent, so the
            # only thing the rule still does is pick an arm -- and if that arm was released when
            # the latch closed it would decline, and the last fifteen tokens would come out one a
            # step at 95.56 ms each. `_alive` keeps the tail on an arm that can draft it.
            return self._alive("s")
        if self.latch:
            return self._choose_latched()
        if self.wide_default:
            return self._choose_wide_default()
        self.last_forced = False
        if self.acc[("l", self.w_large)].n == 0:
            # The first block goes wide, and the reason is information rather than a guess about
            # the text. A wide block prices BOTH options -- its own, and the narrow one by
            # truncation -- while a narrow block prices only itself. So the first block is strictly
            # more informative at the wide width, and the arm that is wrong gets demoted on block
            # two at a cost of one block.
            #
            # It is also what the measured table asks for. On the five workloads with the tree
            # verify, sixteen beats eight on four of them and by 74 % and 86 % on the two
            # reproduction ones; the cold start that began narrow spent four of `quote`'s nine
            # blocks at eight and finished 15.8 % behind a fixed sixteen. A short generation is all
            # cold start, and the prior is the policy there.
            self.last_forced = True
            return "l"
        period = (self.ceiling_period if self.ceiling.value >= self.ceiling_trigger
                  else self.explore_period)
        if self.since[self.w_large] >= period:
            self.last_forced = True
            self.stats["forced"] += 1
            return "l"
        self.last_forced = False
        return "l" if self._wide_value() > self._value("s", self.w_small) else "s"

    def _choose_latched(self) -> str:
        """Measure for eight blocks, decide once, and then stop deciding."""
        self.last_forced = False
        if self.latched is not None:
            return self.latched
        if self.blocks < self.latch_after:
            # Wide first. It prices its own arm and, by truncation, the ceiling rate -- so the
            # question the probe asks next is already narrowed down by the time it is asked.
            self.last_forced = True
            return "l"
        own = self.acc[("s", self.w_small)]
        if self.latch_price:
            return self._choose_latched_priced(own)
        if (self.ceiling.value < self.ceiling_trigger and own.n < self.narrow_warm
                and self.stats["probes"] < self.narrow_warm + 2):
            # The narrow width has slots to spare on this text, so what it commits with its own
            # checkpoint is worth four blocks to find out. Where the ceiling says it saturates,
            # this is skipped entirely and the decision is taken immediately.
            #
            # The attempt cap is not cosmetic. A probe block does not always become a sample --
            # the last blocks of a generation are short and go to the narrow arm for a reason that
            # has nothing to do with this, a drafter can decline -- so a loop that waits for
            # `narrow_warm` SAMPLES can spend the whole request probing. Bounded attempts, and
            # then decide on whatever came back.
            self.last_forced = True
            self.stats["probes"] += 1
            return "s"
        # Decide on the narrow arm's OWN blocks when there are any. This is the one place the free
        # counterfactual must not stand in for them: it prices the wide drafter's draft cut short,
        # and the entire reason for probing is that the narrow checkpoint drafts its own seven
        # slots better than that. Deciding from the counterfactual here would spend four blocks
        # measuring something and then ignore the measurement -- which is what the first run of
        # this did on `prose`, latching wide on a text where its own probes read 2.89 against the
        # wide arm's 2.49.
        if own.n >= 2:
            expected = max(own.value - 1.0, 0.0)
            v_narrow = own.value / self._cost_ms("s", self.w_small, expected)
            if v_narrow > self._value("l", self.w_large) * (1 + self.narrow_margin):
                self.latched = "s"
                self.stats["latched"] = "s"
                self._release_idle("l")
                return "s"
        self.latched = "l"
        self.stats["latched"] = "l"
        self._release_idle("s")
        return "l"

    # --- the latch priced on what each arm will run (QWEN38_LATCH_PRICE) ----------------------

    def _latch_rows(self, key: str) -> int:
        """Rows, anchor included, of the verify this arm submits for the rest of the request.

        The narrow arm keeps its node budget. The wide arm grows into `wide_budget` once the
        request has committed `tree_wide_after` tokens, so it is priced at the grown
        tree unless what is left of the token budget cannot get it there.
        """
        if not self.tree:
            return self.w_small if key == "s" else self.w_large
        if key == "s":
            nb = getattr(self.small, "node_budget", None)
            return int(nb) + 1 if nb is not None else self.w_small
        wb = self.wide_budget
        if wb is None:
            nb = getattr(self.large, "node_budget", None)
            return int(nb) + 1 if nb is not None else self.w_large
        grows = (not self.tree_wide_after
                 or self.req_tokens + max(self.k_left, 0) >= self.tree_wide_after)
        return int(wb if grows else min(wb, self.w_large - 1)) + 1

    def _latch_draft_ms(self, key: str) -> float:
        """This process's draft time for the arm; the other arm's before this one has a clean
        sample (the two checkpoints share an architecture, 12.4 against 12.7 ms on the row), and
        the prior before either has."""
        own = self.dms_latch[key]
        if own.n:
            return own.value
        other = self.dms_latch["l" if key == "s" else "s"]
        return other.value if other.n else own.value

    def _latch_cost_ms(self, key: str, expected: float) -> float:
        rows = self._latch_rows(key)
        base = verify_ms_b(rows, self.latch_table) + self._latch_draft_ms(key)
        if self.tree:
            return base + self.commit_ms
        return base + self.rb.get(rows, 6.5) * self._p_reject(rows, expected)

    def _note_draft(self, key: str, ms: float) -> None:
        """A draft timing for the priced latch. A sample over three times the prior is a compile
        or a graph capture (the published row's first narrow draft read 782.6 ms), not a price."""
        if self.latch_price and self.learn_cost and 0.0 < ms <= 3.0 * self.draft_prior[key]:
            self.dms_latch[key].update(ms)

    def _latch_to(self, key: str) -> str:
        self.latched = key
        self.stats["latched"] = key
        self._release_idle("l" if key == "s" else "s")
        return key

    def _choose_latched_priced(self, own: _Est) -> str:
        """`_choose_latched` with the flag: the probe skip needs `probe_skip_hits` full narrow
        blocks, and the two arms are compared at the verify each will pay after the latch."""
        if (self.stats["ceiling_hits"] < self.probe_skip_hits and own.n < self.narrow_warm
                and self.stats["probes"] < self.narrow_warm + 2):
            self.last_forced = True
            self.stats["probes"] += 1
            return "s"
        if self.stats["ceiling_hits"] >= self.probe_skip_hits:
            # The narrow width runs out of slots on this text, in the opening blocks or in its own
            # probes: its number is a lower bound, not an estimate, and only the wide arm can go
            # past it. A copy, and wide as it was.
            return self._latch_to("l")
        if own.n >= 2:
            narrow = own
        else:
            # probes attempted but declined: the free counterfactual, priced at the narrow arm
            narrow = self.acc[("l", self.w_small)]
            if not narrow.n:
                return self._latch_to("l")
        wide = self.acc[("l", self.w_large)]
        c_s = self._latch_cost_ms("s", max(narrow.value - 1.0, 0.0))
        c_l = self._latch_cost_ms("l", max(wide.value - 1.0, 0.0))
        v_s, v_l = narrow.value / c_s, wide.value / c_l
        self.stats["price"] = f"{c_s:.1f}/{c_l:.1f}"
        # A tie inside the margin goes to the cheaper arm: the dearer one has to be better by it.
        if c_s <= c_l:
            key = "l" if v_l > v_s * (1.0 + self.narrow_margin) else "s"
        else:
            key = "s" if v_s > v_l * (1.0 + self.narrow_margin) else "l"
        return self._latch_to(key)

    # --- the per-block choice (QWEN38_LEN_SWITCH) --------------------------------------------

    def _local_match(self) -> int:
        """How long a suffix of the context recurs earlier in this request's text: the lookup
        drafter's local index, the same test the deep chain uses. Long means copying."""
        ng, ctx = self.ngram, getattr(self, "_ctx", None)
        if ng is None or ctx is None or not hasattr(ng, "local"):
            return 0
        try:
            n, _ = ng.local.lookup(ctx, ng.min_order)
        except Exception:
            return 0
        return int(n)

    def _now_rows(self, key: str) -> int:
        """Rows the arm verifies on the next block (the wide tree grows after
        `tree_wide_after` committed tokens)."""
        if key == "s" or not self.tree or self.wide_budget is None or not self.tree_wide_after:
            return self._latch_rows(key)
        b = (self.wide_budget if self.req_tokens >= self.tree_wide_after
             else min(self.wide_budget, self.w_large - 1))
        return int(b) + 1

    def _switch_cost(self, key: str, expected: float) -> float:
        rows = self._now_rows(key)
        base = verify_ms_b(rows, self.latch_table) + self._latch_draft_ms(key)
        if self.tree:
            return base + self.commit_ms
        return base + self.rb.get(rows, 6.5) * self._p_reject(rows, expected)

    def _choose_switch(self) -> str:
        """Narrow unless the text says wide, re-asked every block, with hysteresis.

        Up (narrow to wide) on a copy signal (a local match of `copy_min` tokens) or when two of
        the last four narrow blocks used every narrow slot: the narrow number is then a lower
        bound and only a wide block can say by how much. Down (wide to narrow) when the free
        price says the narrow width, at its own verify price, commits more per millisecond than
        the wide block did at its. A change of arm needs `switch_after` blocks in a row that ask
        for it, except a copy signal, which moves up at once.
        """
        self.last_forced = False
        copy = self._local_match() >= self.copy_min
        cur = self.cur
        if cur is None:
            return "l" if copy else "s"
        if cur == "l":
            want = "l"
            cf, own = self.acc[("l", self.w_small)], self.acc[("l", self.w_large)]
            if not copy and cf.n >= 2 and own.n >= 2:
                v_s = cf.value / self._switch_cost("s", max(cf.value - 1.0, 0.0))
                v_l = own.value / self._switch_cost("l", max(own.value - 1.0, 0.0))
                if v_s > v_l * (1.0 + self.switch_margin):
                    want = "s"
        else:
            want = "l" if (copy or sum(self.recent_hits[-4:]) >= 2) else "s"
        if want == cur:
            self.want_run = 0
            return cur
        self.want_run += 1
        if (copy and want == "l") or self.want_run >= self.switch_after:
            self.want_run = 0
            return want
        return cur

    def _b_reset(self) -> None:
        self.hits4: list[int] = []
        self.wide4: list[int] = []
        self.b_blocks = 0
        self.b_phase = "measure"
        self.b_switches = 0
        self.b_last_switch = -10 ** 9
        self.b_run = 0
        self._copy = False
        self.val = {"s": _Est(0.0, 0.3, warm=3), "l": _Est(0.0, 0.3, warm=3),
                    "cf": _Est(0.0, 0.3, warm=3)}
        self.narrow_rows = _Est(16.0, 0.3, warm=3)

    def _choose_b(self) -> str:
        """Design B: today's opening and latch, then two exits, a dwell and a switch cap.

        1. Blocks 1-4 run the wide arm at 16 nodes (the ladder keeps it there).
        2. Two full blocks in the last four, or a copy signal, latch wide. Otherwise the narrow
           checkpoint is probed (up to `narrow_warm` blocks) and the latch compares the two arms
           priced at 16 nodes each; probes that fill the narrow width latch wide.
        3. Up from narrow: a copy signal at once; two full blocks in four after `switch_after`
           blocks that agree.
        4. Down from wide: the wide arm's last four blocks commit less than the narrow probe's
           number minus `switch_margin`, with no copy signal and under two full blocks in four,
           `switch_after` blocks in a row.
        5. No reverse switch within `dwell` blocks, at most `max_switches` a request; after that
           the arm is latched for good and the other one released.
        """
        self.last_forced = False
        self._copy = self._local_match() >= self.copy_min
        hits = sum(self.hits4)
        if self.latched is not None:
            return self.latched
        if self.b_phase == "measure":
            if self.b_blocks < self.latch_after and not self._copy:
                self.last_forced = True
                return "l"
            if self._copy or hits >= self.probe_skip_hits:
                return self._b_decide("l")
            self.b_phase = "probe"
        if self.b_phase == "probe":
            own = self.acc[("s", self.w_small)]
            if (hits < self.probe_skip_hits and own.n < self.narrow_warm
                    and self.stats["probes"] < self.narrow_warm + 2 and not self._copy):
                self.last_forced = True
                self.stats["probes"] += 1
                return "s"
            if self._copy or hits >= self.probe_skip_hits or own.n < 2:
                return self._b_decide("l")
            wide = self.acc[("l", self.w_large)]
            c_s = self._latch_cost_ms("s", max(own.value - 1.0, 0.0))
            rows_l = min(self._latch_rows("l"), self.w_large)          # 16 nodes for both
            c_l = (verify_ms_b(rows_l, self.latch_table) + self._latch_draft_ms("l")
                   + (self.commit_ms if self.tree else 0.0))
            v_s, v_l = own.value / c_s, wide.value / c_l
            self.stats["price"] = f"{c_s:.1f}/{c_l:.1f}"
            if c_s <= c_l:
                key = "l" if v_l > v_s * (1.0 + self.narrow_margin) else "s"
            else:
                key = "s" if v_s > v_l * (1.0 + self.narrow_margin) else "l"
            return self._b_decide(key)
        cur = self.cur or "l"
        if not self.exits:
            return cur
        want = cur
        if cur == "s":
            if self._copy:
                want, now = "l", True
            else:
                want, now = ("l" if hits >= 2 else "s"), False
        else:
            own = self.acc[("s", self.w_small)]
            now = False
            if (not self._copy and hits < 2 and len(self.wide4) == 4 and own.n >= 1
                    and sum(self.wide4) / 4.0 < own.value * (1.0 - self.switch_margin)):
                want = "s"
        if want == cur:
            self.b_run = 0
            return cur
        if self.b_switches and self.b_blocks - self.b_last_switch < self.dwell:
            # a reverse switch inside the dwell: the text has not had time to change
            self.b_run = 0
            return cur
        self.b_run += 1
        if not now and self.b_run < self.switch_after:
            return cur
        self.b_run = 0
        self.b_switches += 1
        self.b_last_switch = self.b_blocks
        self.wide4 = []
        if self.b_switches >= self.max_switches:
            self.latched = want
            self.stats["latched"] = want
            self._release_idle("l" if want == "s" else "s")
        return want

    def _block_ms(self, key: str, rows: int) -> float:
        return (verify_ms_b(rows, self.stair_table) + self._latch_draft_ms(key)
                + (self.commit_ms if self.tree else 0.0))

    def _calc_observe(self, key: str, width: int, committed: int, accepted: int) -> None:
        """Realised committed tokens per ms of the block, and for a wide block the free price of
        the narrow checkpoint: the accepted run cut to its 7 slots, at the narrow arm's own rows."""
        self.val[key].update(committed / self._block_ms(key, width))
        if key == "s":
            self.narrow_rows.update(width)
        else:
            cf = min(accepted, self.w_small - 1) + 1
            rows = int(round(self.narrow_rows.value))
            self.val["cf"].update(cf / self._block_ms("s", rows))

    @staticmethod
    def _ctx_class(n: int) -> int:
        c = 1024
        while c < n and c < 262144:
            c *= 2
        return c

    @staticmethod
    def _tile(rows: int) -> int:
        return 0 if rows <= 16 else 1 if rows <= 24 else 2 if rows <= 32 else 3

    def _stair_factor(self, rows: int, chain: bool) -> float:
        """Measured over priced block time for this shape at the current context class."""
        ctx = getattr(self, "_ctx", None)
        cls = self._ctx_class(len(ctx) if ctx is not None else 0)
        est = self.block_factor.get((bool(chain), self._tile(rows), cls))
        return est.value if est is not None and est.n else 1.0

    def _class_price(self, rows: int, chain: bool) -> float:
        """Verify ms of `rows` at the current context, from the per-class tables (computed once
        a round for every row count: the cut asks for it at every prefix of every candidate)."""
        ctx = getattr(self, "_ctx", None)
        n = max(len(ctx) if ctx is not None else 0, 1)
        cache = getattr(self, "_price_cache", None)
        if cache is None or cache[0] != n >> 8:           # refreshed every 256 tokens of context
            cache = self._price_cache = (n >> 8, {})
        tab = cache[1].get(bool(chain))
        if tab is None:
            top = max(34, self.max_nodes + 3)
            tab = cache[1][bool(chain)] = [self._class_price_raw(r, chain, n) for r in range(top)]
        return tab[min(max(int(rows), 0), len(tab) - 1)]

    def _class_price_raw(self, rows: int, chain: bool, n: int) -> float:
        import math
        kind = "chain" if chain else "tree"
        cls = sorted(c for c, d in self.class_tables.items() if kind in d)
        if not cls:
            return verify_ms_b(rows, self.stair_table)
        lo = max([c for c in cls if c <= n] or [cls[0]])
        hi = min([c for c in cls if c >= n] or [cls[-1]])
        v_lo = verify_ms_b(rows, self.class_tables[lo][kind])
        if hi == lo:
            return v_lo
        v_hi = verify_ms_b(rows, self.class_tables[hi][kind])
        f = (math.log2(n) - math.log2(lo)) / (math.log2(hi) - math.log2(lo))
        return v_lo + f * (v_hi - v_lo)

    def _block_price(self, rows: int, chain: bool) -> float:
        fixed = float(getattr(self.large, "head_fixed_ms", self.draft_prior["l"]))
        return (verify_ms_b(rows, self.stair_table) + fixed
                + (self.commit_ms if self.tree else 0.0))

    def _clock(self) -> None:
        """Close the previous round's timing, now that the next proposal starts."""
        now = time.perf_counter()
        if self._blk is not None:
            t0, rows, chain, cls = self._blk
            ms = (now - t0) * 1e3
            price = self._block_price(rows, chain)
            if 0.5 * price <= ms <= 3.0 * price:
                key = (chain, self._tile(rows), cls)
                est = self.block_factor.get(key)
                if est is None:
                    est = self.block_factor[key] = _Est(1.0, 0.2, warm=3)
                est.update(ms / price)
        self._blk = None
        self._t_start = now

    def _open_block(self, tree, context) -> None:
        if not self.calc or tree is None or tree.n_draft == 0:
            return
        chain = all(p == i - 1 for i, p in enumerate(tree.parents[1:], start=1))
        self._blk = (self._t_start, tree.n_draft + 1, chain, self._ctx_class(len(context)))

    def _choose_wide(self) -> str:
        """The wide checkpoint on every block; the narrow one stops being synced for the
        request, as a released arm (so a snapshot carries only the wide arm)."""
        self.last_forced = False
        self._copy = False
        if self.idle is None and self.stats.get("idle") != "s":
            self.idle = "s"
            rel = getattr(self.head_small, "release", None)
            if rel is not None:
                rel()
            self.stats["idle"] = "s"
            self.stats["latched"] = "l"
        return "l"

    def _choose_calc(self) -> str:
        """Narrow first; up on a copy signal (at once) or two full blocks in four; down when the
        free price of the narrow checkpoint beats what the wide one realises by `switch_margin`.
        The node count of each block is the arm's own calculation (MergedRouter's staircase cut);
        the dwell and the cap are design B's."""
        self.last_forced = False
        self._copy = self._local_match() >= self.copy_min
        if self.latched is not None:
            return self.latched
        cur = self.cur
        if cur is None:
            return "l" if self._copy else self.calc_start
        hits = sum(self.hits4)
        now = False
        if cur == "s":
            if self._copy:
                want, now = "l", True
            else:
                want = "l" if hits >= 2 else "s"
        else:
            want = "l"
            vl, vcf = self.val["l"], self.val["cf"]
            if (not self._copy and hits < 2 and vl.n >= 2 and vcf.n >= 2
                    and vcf.value > vl.value * (1.0 + self.switch_margin)):
                want = "s"
        if want == cur:
            self.b_run = 0
            return cur
        if self.b_switches and self.b_blocks - self.b_last_switch < self.dwell:
            self.b_run = 0
            return cur
        self.b_run += 1
        if not now and self.b_run < self.switch_after:
            return cur
        self.b_run = 0
        self.b_switches += 1
        self.b_last_switch = self.b_blocks
        if want == "l":
            self.val["l"] = _Est(0.0, 0.3, warm=3)
            self.val["cf"] = _Est(0.0, 0.3, warm=3)
        if self.b_switches >= self.max_switches:
            self.latched = want
            self.stats["latched"] = want
            self._release_idle("l" if want == "s" else "s")
        return want

    def _b_decide(self, key: str) -> str:
        """The latch of design B: the decision after the opening, open to the exits."""
        self.b_phase = "run"
        self.stats["latched"] = key
        if not self.exits:
            self.latched = key
            self._release_idle("l" if key == "s" else "s")
        return key

    def _use(self, key: str) -> None:
        """Make `key` the drafting arm: bring it up to date if it was not."""
        if self.cur is not None and key != self.cur:
            self.stats["switches"] = self.stats.get("switches", 0) + 1
            self.recent_hits = []
            # the estimates of the arm being left describe the text it was written on; the
            # wide arm's own and free-price numbers restart when it is chosen again
            if key == "l":
                for w in (self.w_small, self.w_large):
                    self.acc[("l", w)] = _Est(3.6 if w == self.w_large else 3.2, self.alpha,
                                              warm=self.acc_warm)
        self.cur = key
        self._catch_up(key)

    def _choose_wide_default(self) -> str:
        """The phase-9 rule: take the wide block unless there is a reason not to.

        Three conditions have to hold at once before the router comes down, and each of them is
        one of the ways the old rule got `chat` wrong.

        1. **The narrow arm must not be censored.** `ceiling` is the fraction of recent blocks in
           which the narrow width would have accepted every slot it had -- measured on wide blocks
           as well as narrow ones, so the signal exists even when the narrow arm never runs. A
           narrow block that saturates is an observation that says "at least seven" and nothing
           more, and its own measured value is a lower bound being read as an estimate.
        2. **There must be evidence at all.** With neither arm's history and no counterfactual the
           only honest choice is the one that prices both, which is the wide one.
        3. **The narrow arm must be strictly better by `narrow_margin`.** Not merely better: the
           two widths now cost within about one per cent of each other, so an unmargined
           comparison turns measurement noise into a policy, and the narrow number is the
           optimistic one of the two for the reason `narrow_margin` documents.

        Going back up needs no schedule under this rule, because the router is already there. What
        it does need is a way back from a narrow stretch that has gone stale, and that is the same
        forced probe the old policy used, now reached only from the narrow arm.
        """
        self.last_forced = False
        if self.since[self.w_large] >= (self.ceiling_period
                                        if self.ceiling.value >= self.ceiling_trigger
                                        else self.explore_period):
            # Only reachable after a run of narrow blocks: a wide block resets this counter.
            self.last_forced = True
            self.stats["forced"] += 1
            return "l"
        if self.ceiling.value >= self.ceiling_trigger:
            # The narrow width is running out of slots. Its own number would be a lower bound and
            # there is nothing to find out down there.
            return "l"
        own = self.acc[("s", self.w_small)]
        if (own.n < self.narrow_warm and self.blocks >= self.narrow_probe_after
                and self.since[self.w_small] >= self.narrow_probe_period):
            # The bounded downward probe. At most `narrow_warm` blocks a request, never taken
            # while the ceiling says the narrow width saturates, and each one costs about one per
            # cent of a block on the new curve -- against the 5 % a whole request of the wrong arm
            # costs on `prose`.
            self.last_forced = True
            self.stats["probes"] += 1
            return "s"
        if not self._narrow_est().n:
            return "l"
        if self._narrow_value() > self._value("l", self.w_large) * (1.0 + self.narrow_margin):
            return "s"
        return "l"

    # --- the lattice, for the width the wide drafter's own draft deserves --------------------

    def _path_prob(self, child) -> list[float] | None:
        """Per-slot conditional probability along the greedy path, from the drafter's lattice.

        `scores[l, p, c]` is the selector's score for candidate `c` at slot `l` given candidate `p`
        at slot `l-1`; a softmax along `c` of the row the greedy walk actually took is the cheapest
        thing that is monotone in the right direction, which is all this is used for. It is not a
        calibrated probability, and the one scalar `calib` is what stands between it and reality.
        """
        lat = getattr(_head_of(child), "_lattice", None)
        if lat is None:
            return None
        import torch

        cand, scores = lat
        if scores.shape[0] < 2:
            return None
        idx = int(scores[0, 0].argmax())
        path = [idx]
        local = scores[1:].argmax(dim=-1)                       # [L-1, k]
        for e in range(local.shape[0]):
            idx = int(local[e, idx])
            path.append(idx)
        sel = torch.as_tensor(path, dtype=torch.long, device=scores.device)
        k = scores.shape[-1]
        rest = scores[1:].gather(1, sel[:-1].view(-1, 1, 1).expand(-1, 1, k)).squeeze(1)
        rows = torch.cat([scores[0, 0].unsqueeze(0), rest], dim=0).float()
        temp = float(getattr(_head_of(child), "tree_temp", 1.0)) or 1.0
        lp = torch.log_softmax(rows / temp, dim=-1)
        return lp.gather(1, sel[:, None])[:, 0].exp().tolist()

    @staticmethod
    def _expected_prefix(probs: list[float], slots: int) -> float:
        """Expected accepted length of the first `slots` of a chain.

        A chain is accepted prefix-first, so slot `i` counts only if every slot above it was
        accepted: the expectation is the sum of the running products, which is what makes slot 0
        multiply every later term and why a maximiser of the total score is the wrong draft
        (Viterbi at 3.22 against greedy's 4.30).
        """
        total, run = 0.0, 1.0
        for i in range(min(slots, len(probs))):
            run *= probs[i]
            total += run
        return total

    def _trim(self, draft: list[int], probs: list[float] | None) -> tuple[list[int], int]:
        """Submit the wide draft at whichever width prices better, now that it is paid for.

        This is the second half of the decision and the free half: the draft cost is sunk, so the
        only term that changes with the width is the verify, and the drafter's own scores for slots
        8 to 15 are the evidence for whether that half of the block is worth 16 ms.
        """
        if not self.width_trim or len(draft) < self.w_large - 1:
            return draft, min(len(draft) + 1, self.w_large)
        margin = 1.0 + self.narrow_margin if self.wide_default else 1.0
        if probs is None:
            # No lattice to read -- the drafter is running without its selector. The width is then
            # decided on the two arms' own histories, which is the same comparison one step coarser.
            if (self.acc[("l", self.w_small)].n
                    and self._value("l", self.w_small)
                    > self._value("l", self.w_large) * margin):
                self.stats["trims"] += 1
                return draft[:self.w_small - 1], self.w_small
            return draft, self.w_large
        c = max(self.calib.value, 0.05)
        full = min(c * self._expected_prefix(probs, self.w_large - 1), float(self.w_large - 1))
        narrow = min(c * self._expected_prefix(probs, self.w_small - 1), float(self.w_small - 1))
        v_full = (full + 1.0) / self._cost_ms("l", self.w_large, full)
        v_narrow = (narrow + 1.0) / self._cost_ms("l", self.w_small, narrow)
        # The same margin the arm choice uses, and here it matters more: the draft is already paid
        # for, so the only term that changes with the width is a verify that now differs by about
        # one per cent, and the lattice's own prefix expectation saturates well before slot 15.
        # An unmargined comparison trims almost every wide block back to eight and the wide arm
        # then never learns anything -- which is what `chat` was, 59 blocks in 66.
        if v_narrow > v_full * margin:
            self.stats["trims"] += 1
            return draft[:self.w_small - 1], self.w_small
        return draft, self.w_large

    # --- proposing -------------------------------------------------------------------------

    def propose(self, context: list[int], k: int) -> list[int]:
        self.last_q = None
        if k <= 0:
            return []
        self.k_left = k
        self._ctx = context
        key = self._choose(k)
        if self.switch:
            self._use(key)
        child = self.small if key == "s" else self.large
        want = (self.w_small if key == "s" else self.w_large) - 1
        t0 = time.perf_counter()
        draft = child.propose(context, min(k, want))
        self.last_q = getattr(child, "last_q", None)
        if not draft and key == "l" and self.idle != "s":
            # The wide drafter declines where the narrow one would not only at the very end of a
            # sequence, where its block runs past `max_len`. Fall back rather than take the
            # one-token path, which is the expensive one -- unless the narrow arm was released,
            # in which case it would decline too and the fallback is a wasted draft call.
            key, child, want = "s", self.small, self.w_small - 1
            if self.switch:
                self._use("s")
            draft = child.propose(context, min(k, want))
            self.last_q = getattr(child, "last_q", None)
        if self.learn_cost:
            ms = (time.perf_counter() - t0) * 1e3
            self.dms[key].update(ms)
            self._note_draft(key, ms)
        if not draft:
            self.stats["declined"] += 1
            self.last_key, self.last_width, self.last_expected = None, 0, 0.0
            return []
        probs = self._path_prob(child) if key == "l" else None
        if key == "l" and not self.fixed and not self.last_forced:
            # A forced probe is never trimmed. The trim's own fallback, when there is no lattice to
            # read, compares the two arms on histories that both came from wide blocks -- so on a
            # step where the wide block had accepted seven or fewer it says the narrow width is
            # cheaper, trims the probe back to eight, and the wide arm learns nothing. That is the
            # 11:03 trap wearing a third costume: the exploration is silently converted into the
            # option it was meant to explore away from, and the router then never explores again.
            draft, width = self._trim(draft, probs)
            if self.last_q is not None:
                self.last_q = self.last_q[:len(draft)]
        else:
            width = len(draft) + 1
        self.last_key, self.last_width = key, width
        self.last_depth = width - 1
        self.last_expected = (self._expected_prefix(probs, width - 1) if probs is not None
                              else 0.0)
        self.blocks += 1
        self.stats["blocks"] += 1
        self.stats["small" if key == "s" else "large"] += 1
        self.stats["width_hist"][width] = self.stats["width_hist"].get(width, 0) + 1
        for w in self.since:
            self.since[w] = 0 if w == width else self.since[w] + 1
        return draft

    def propose_tree(self, context: list[int], k: int):
        return run_steps(self.propose_tree_steps(context, k))

    @staticmethod
    def _timed_steps(child, context: list[int], k: int):
        """(tree, ms) from `child`'s proposal, the ms without the time it spent stopped: while it
        is stopped the loop streams the last block, and what the router prices is the
        draft call, not the stream."""
        steps = tree_steps(child, context, k)
        ms, t = 0.0, time.perf_counter()
        while True:
            try:
                next(steps)
            except StopIteration as done:
                return done.value, ms + (time.perf_counter() - t) * 1e3
            ms += (time.perf_counter() - t) * 1e3
            yield
            t = time.perf_counter()

    def propose_tree_steps(self, context: list[int], k: int):
        """The same choice, for a verify that takes a tree.

        Under `forward_tree` the arm is no longer a bare drafter: it is a `MergedRouter` that puts
        the lookup drafter's tree and the block drafter's lattice into one node set and prunes to
        its own budget. What the length router adds is which budget -- seven drafted nodes or
        fifteen -- and the budget is what the verify is charged for.

        The width is whatever the arm's merge produced after its prune, which is at most the budget
        and is often less: a lookup tree on fresh prose has one node in it. So the width is read off
        the tree rather than assumed, and `_arm` puts the evidence with the arm that made it.
        """
        self.last_q = None                     # q-aware accept is a chain mechanism (v1)
        self.last_deep = False
        if self.calc:
            self._clock()
        if k <= 0:
            return None
        if self.deep and self.full_run >= self.deep_after and k >= self.w_large:
            tree = self._deep_chain(context, k)
            if tree is not None:
                self._open_block(tree, context)
                return tree
        self.k_left = k
        self._ctx = context
        key = self._choose(min(k, self.w_large - 1))
        if self.switch:
            self._use(key)
        child = self.small if key == "s" else self.large
        if (key == "l" and self.tree_wide_after and self.wide_budget is not None
                and hasattr(child, "head_budget") and not self.calc):
            grow = self.req_tokens >= self.tree_wide_after
            if self.switch and self.switch_mode == "b":
                # the ladder: the 24-node tree only in a copy regime, 16 nodes otherwise
                grow = grow and (self._copy or sum(self.hits4) >= 2)
            b = self.wide_budget if grow else min(self.wide_budget, self.w_large - 1)
            child.node_budget = child.head_budget = b
        want = (self.w_small if key == "s" else self.w_large) - 1
        cls = self._ctx_class(len(context)) if self.calc else 0
        if self.calc and hasattr(child, "head_fixed_ms"):
            est = self.dms_cls.get((key, cls))
            child.head_fixed_ms = est.value if est is not None and est.n else self.stair_fixed_ms
        tree, ms = yield from self._timed_steps(child, context, min(k, want))
        if (self.calc and self.learn_cost and 2.0 <= ms <= 3.0 * max(self.stair_fixed_ms, 30.0)
                and tree is not None):
            est = self.dms_cls.setdefault((key, cls), _Est(self.stair_fixed_ms, 0.3, warm=3))
            if getattr(child, "last_head_tree", True) is not None:     # the head drafted
                est.update(ms)
        if (tree is None or tree.n_draft == 0) and key == "l" and self.idle != "s":
            key, child, want = "s", self.small, self.w_small - 1
            if self.switch:
                self._use("s")
            tree, ms2 = yield from self._timed_steps(child, context, min(k, want))
            ms += ms2
        if self.learn_cost:
            self.dms[key].update(ms)
            self._note_draft(key, ms)
        if tree is None or tree.n_draft == 0:
            self.stats["declined"] += 1
            self.last_key, self.last_width, self.last_expected = None, 0, 0.0
            return None
        width = tree.n_draft + 1
        self.last_key, self.last_width, self.last_expected = key, width, 0.0
        self.last_depth = max(tree.depths())
        self.blocks += 1
        self.stats["blocks"] += 1
        self.stats["small" if key == "s" else "large"] += 1
        self.stats["width_hist"][width] = self.stats["width_hist"].get(width, 0) + 1
        arm = self._arm(width, key)
        for w in self.since:
            self.since[w] = 0 if w == arm else self.since[w] + 1
        self._open_block(tree, context)
        return tree

    def _deep_chain(self, context: list[int], k: int):
        """The lookup drafter's single best continuation as a chain, when it is long and certain.

        Certain means a local match of at least `deep_order` tokens -- the text is being copied
        from earlier in this request, not recalled from the corpus -- and a top continuation that
        holds at least `deep_share` of the vote. Anything less goes to the ordinary arms, so the
        cost of a wrong guess is bounded to texts that have just filled a whole wide block.
        """
        ng = self.ngram
        if ng is None:
            return None
        depth = min(k, self.deep - 1)
        n_local, _ = ng.local.lookup(context, ng.min_order)
        if n_local < self.deep_order:
            return None
        order, cands = ng.candidates(context, depth)
        if not cands or order != n_local:
            return None
        total = sum(w for _, w in cands)
        cont, w = cands[0]
        if len(cont) < self.w_large or w < self.deep_share * total:
            return None
        from engine.tree import DraftTree
        cont = cont[:depth]
        tree = DraftTree(tokens=[context[-1]] + list(cont),
                         parents=[-1] + list(range(len(cont))),
                         scores=[1.0] * (len(cont) + 1),
                         source=["root"] + ["deep"] * len(cont))
        width = tree.n_draft + 1
        self.last_key, self.last_width, self.last_expected = "l", width, 0.0
        self.last_depth = width - 1
        self.last_deep = True
        self.blocks += 1
        self.stats["blocks"] += 1
        self.stats["large"] += 1
        self.stats["deep"] += 1
        self.stats["width_hist"][width] = self.stats["width_hist"].get(width, 0) + 1
        return tree

    def observe(self, tokens: list[int]) -> None:
        """Learn from the block, including about the width that was not submitted.

        `tokens` is what the loop committed: the accepted prefix of the draft plus the model's own
        token, so `len(tokens)` is the block's yield and `len(tokens) - 1` is how much of the draft
        survived. Three things are learned from it and only the first needed the block to be run at
        this configuration.
        """
        self.req_tokens += len(tokens)
        chosen = None if self.last_key is None else (
            self.small if self.last_key == "s" else self.large)
        if chosen is not None and hasattr(chosen, "observe"):
            # Only the arm that proposed. A `MergedRouter` updates its calibration and extends the
            # shared lookup index here, and doing that twice would index every token twice.
            chosen.observe(tokens)
        if self.last_key is None or self.last_width == 0:
            return
        committed = len(tokens)
        accepted = committed - 1
        width = self.last_width
        key = self.last_key
        if self.last_deep:
            # A deep block is evidence about the copy, not about either arm: it goes in the
            # histogram under its own width and nowhere else.
            hist = self.stats["commit_hist"].setdefault(width, {})
            hist[committed] = hist.get(committed, 0) + 1
            self.stats["cap_arm"] += int(committed >= width)
            self.stats["cap_depth"] += int(committed >= width)
            self.stats["deep_tokens"] += committed
            self.last_full = committed >= width
            self.full_run = self.full_run + 1 if self.last_full else 0
            if self.switch and self.switch_mode in ("b", "calc"):
                self.hits4 = (self.hits4 + [int(committed >= self.w_small)])[-4:]
                self.b_blocks += 1
            self.last_deep = False
            self.last_key, self.last_width, self.last_expected = None, 0, 0.0
            return
        # The last block of a generation is whatever is left of the token budget, so the submitted
        # width can be any number between two and the arm's own. Its evidence belongs to the arm it
        # came from, not to a width the router can never choose on purpose.
        self.acc[(key, self._arm(width, key))].update(committed)
        self.stats["tokens_small" if key == "s" else "tokens_large"] += committed
        arm = self._arm(width, key)
        hist = self.stats["commit_hist"].setdefault(arm, {})
        hist[committed] = hist.get(committed, 0) + 1
        self.stats["cap_arm"] += int(committed >= arm)
        self.stats["cap_depth"] += int(committed >= self.last_depth + 1)
        self.last_full = key == "l" and committed >= arm
        self.full_run = self.full_run + 1 if self.last_full else 0

        if self.switch and self.switch_mode in ("b", "calc"):
            full = accepted >= self.w_small - 1
            self.hits4 = (self.hits4 + [int(full)])[-4:]
            if key == "l":
                self.wide4 = (self.wide4 + [committed])[-4:]
            self.b_blocks += 1
            if self.calc:
                self._calc_observe(key, width, committed, accepted)
        if key == "s" or width <= self.w_small:
            # the narrow arm ran out of slots: its lattice is w_small - 1 deep however many nodes
            # its tree had
            hit = 1.0 if accepted >= min(width, self.w_small) - 1 else 0.0
            self.ceiling.update(hit)
            self.stats["ceiling_hits"] += int(hit)
            if self.switch and key == "s":
                self.recent_hits.append(int(hit))
        elif key == "l":
            # PHASE 9. The censoring question asked of a WIDE block: would the narrow width have
            # run out of slots here? It is the same question the narrow arm answers about itself,
            # and it is answerable from a wide block for the same reason the counterfactual is --
            # the narrow block is a prefix of this one. Without this the ceiling rate is only ever
            # measured on an arm that `wide_default` almost never runs, so it sits at its prior
            # and the one signal that says the narrow number is a lower bound never fires.
            hit = 1.0 if accepted >= self.w_small - 1 else 0.0
            self.ceiling.update(hit)
            self.stats["ceiling_hits"] += int(hit)
            self.rej.update(0.0 if accepted >= width - 1 else 1.0)
            # The free counterfactual. For a CHAIN it is exact: the narrow block is a prefix of the
            # wide one, the verify computes the same rows for it, and the target's argmax at row i
            # does not depend on rows after i. For a TREE it is an approximation, and the direction
            # is known: the narrow tree is the wide one's best-first prefix, so it holds the
            # accepted path only when that path's nodes ranked in the top seven. It can therefore
            # overstate the narrow arm on a step where the accepted path came from a low-priority
            # branch, and the exploration schedule is what stops that from being permanent.
            self.acc[("l", self.w_small)].update(min(accepted, self.w_small - 1) + 1)

        if self.last_expected > 0:
            self.calib.update(min(accepted, width - 1) / self.last_expected)
        self.last_key, self.last_width, self.last_expected = None, 0, 0.0

    # --- reporting -------------------------------------------------------------------------

    def report(self) -> str:
        s, l = self.stats["small"], self.stats["large"]
        a_s = self.stats["tokens_small"] / s if s else 0.0
        a_l = self.stats["tokens_large"] / l if l else 0.0
        return (f"lenrouter {self.w_small}x{s} ({a_s:.2f} tok/block) "
                f"{self.w_large}x{l} ({a_l:.2f} tok/block) "
                f"trims {self.stats['trims']} forced {self.stats['forced']} "
                f"probes {self.stats['probes']} latched {self.stats['latched']} "
                f"idle {self.stats['idle']} "
                f"ceiling {self.ceiling.value:.2f} calib {self.calib.value:.2f} "
                f"verify {self.vms[self.w_small].value:.1f}/{self.vms[self.w_large].value:.1f} ms "
                f"draft {self.dms['s'].value:.1f}/{self.dms['l'].value:.1f} ms "
                f"widths {dict(sorted(self.stats['width_hist'].items()))} "
                f"commits {_hist_str(self.stats['commit_hist'])} "
                f"cap arm {self.stats['cap_arm']} depth {self.stats['cap_depth']}"
                + (f" deep {self.stats['deep']} ({self.stats['deep_tokens']} tok)"
                   if self.deep else "")
                + (f" price {self.stats.get('price', '-')} ms" if self.latch_price else "")
                + (f" switches {self.stats.get('switches', 0)} catchups "
                   f"{self.stats.get('catchups', 0)} cur {self.cur or '-'}" if self.switch else "")
                + (" cost " + ",".join(f"{'c' if k[0] else 't'}{k[1]}@{k[2] // 1024}k:{v.value:.2f}"
                                        for k, v in sorted(self.block_factor.items()))
                   if self.calc and self.block_factor else ""))


def _hist_str(h: dict) -> str:
    """`{8: {2: 5, 8: 1}, 16: {...}}` as `8:2x5,8x1|16:...` -- one token, so a log line splits on
    spaces. `tools/accept_hist.py` reads it back."""
    return "|".join(f"{arm}:" + ",".join(f"{c}x{n}" for c, n in sorted(v.items()))
                    for arm, v in sorted(h.items())) or "-"
