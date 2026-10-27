"""Greedy decoding, with and without speculation, and the accounting that compares them.

The speculative loop is the ordinary one: a drafter proposes m tokens, the engine verifies the
block `[last] + draft` in a single pass, the longest prefix of the draft the model agrees with is
accepted, and the model's own token at the first disagreement is accepted as well -- so a block
always yields at least one token and at most m + 1.

What is specific to this model is the rollback. Eleven twelfths of the layers keep a recurrent state
instead of a cache, and the state after a partial accept has to be reconstructed rather than
truncated. `Qwen38Engine.rollback_to` does that from the block trace; the loop here only decides how
many tokens to keep.

The invariant that makes any of this trustworthy: **greedy output must not depend on the drafter.**
`verify_lossless` checks exactly that, with a drafter built to be wrong on purpose.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import torch

from engine.drafters import Drafter
from engine.penalty import PenaltyState


@dataclass
class Relax:
    """The relaxed accept rule. Off by default, and it is the only thing in this file that can
    change what the engine writes.

    Greedy verification accepts a drafted token only when it is the target's argmax, and that is
    what makes the speculative path's output identical to the unspeculated one. Two knobs loosen it:

        tau    accept `t` when `p(t) >= tau * p(argmax)`; `tau = 1` is the greedy rule
        rank   accept `t` when it is among the target's `rank` most likely; `rank = 1` is greedy

    Both are evaluated on logits and neither needs a softmax: `p(t) >= tau * p1` is
    `logit(t) >= logit_max + log(tau)`, and a rank test is a `topk`. A block whose tokens were
    accepted this way is still verified in one pass and still costs one pass; what it loses is the
    guarantee, and docs/exactness.md says so.

    A third, added 2026-09-17 (phase 7, step 3) and off by default like the other two:

        typical  accept `t` when `p(t) >= min(eps, delta * exp(-H))`, where `H` is the entropy of
                 the target's own distribution at that position, in nats.

    The difference from `tau` is what the threshold is relative to. `tau` is relative to the
    argmax: it asks "is this token nearly as likely as the best one", and on a peaked distribution
    -- where the target is certain and a wrong draft token is genuinely wrong -- a fixed `tau` is
    the most permissive it ever is, because `p(argmax)` is close to 1 and `tau * p(argmax)` is a
    high bar only in appearance. The typical rule is relative to how *undecided* the target is:
    where the target is certain the threshold `delta * exp(-H)` is near `delta` and almost nothing
    but the argmax gets in; where it is genuinely uncertain, and many tokens are equally
    reasonable continuations, the threshold falls and the drafter's choice among them stands.

    That is the shape of the error this engine actually makes. Track C measured the tau rule at
    29.4 tok/s on prose against 17 lossless, and its cost is concentrated where the target was
    sure. `eps` is the floor that stops the rule from accepting anything at all on a flat
    distribution; the defaults, 0.09 and 0.3, are the published Medusa ones.

    It costs one `log_softmax` over a 248,320-wide row, and it is paid ONLY on a token the greedy
    rule was going to reject -- which is at most one per block, because the block stops there.
    """

    tau: float = 1.0
    rank: int = 1
    typical: bool = False
    eps: float = 0.09
    delta: float = 0.3

    @property
    def on(self) -> bool:
        return self.tau < 1.0 or self.rank > 1 or self.typical

    def accepts(self, row: torch.Tensor, token: int, argmax: int) -> bool:
        if token == argmax:
            return True
        if not self.on:
            return False
        if self.tau < 1.0:
            if float(row[token]) >= float(row[argmax]) + math.log(self.tau):
                return True
        if self.typical:
            lp = row.float().log_softmax(-1)
            h = float(-(lp.exp() * lp).sum())
            if float(lp[token]) >= math.log(min(self.eps, self.delta * math.exp(-h))):
                return True
        if self.rank > 1:
            top = torch.topk(row, self.rank).indices
            if bool((top == token).any()):
                return True
        return False


@dataclass
class DecodeStats:
    tokens: int = 0
    blocks: int = 0
    drafted: int = 0
    accepted: int = 0
    rollbacks: int = 0
    nodes: int = 0
    relaxed: int = 0
    # For every drafted token the target did not agree with: where that token ranked in the
    # target's own distribution, and its probability as a fraction of the argmax's. This is what
    # says whether a tolerance can ever recover a rejection, and it costs one topk on a block that
    # was going to be rolled back anyway.
    miss_rank: list[int] = field(default_factory=list)
    miss_ratio: list[float] = field(default_factory=list)
    prefill_s: float = 0.0
    decode_s: float = 0.0
    rollback_s: float = 0.0
    draft_s: float = 0.0
    per_block: list[int] = field(default_factory=list)
    # Top-1 minus top-2 logit at each greedy step. A losslessness gate that fails at a position
    # where this is near zero is measuring the prompt, not the engine -- the 07:30 entry in the
    # ledger is that mistake made once already.
    gaps: list[float] = field(default_factory=list)
    tops: list[float] = field(default_factory=list)

    @property
    def tok_s(self) -> float:
        return self.tokens / self.decode_s if self.decode_s else 0.0

    @property
    def accept_len(self) -> float:
        return self.tokens / self.blocks if self.blocks else 0.0

    @property
    def accept_rate(self) -> float:
        return self.accepted / self.drafted if self.drafted else 0.0

    def line(self, label: str) -> str:
        return (f"{label:22s} {self.tokens:5d} tok  {self.tok_s:6.2f} tok/s  "
                f"blocks {self.blocks:4d}  accepted/block {self.accept_len:5.2f}  "
                f"draft acceptance {self.accept_rate * 100:5.1f} %  "
                f"rollbacks {self.rollbacks:4d}  "
                + (f"nodes/block {self.nodes / self.blocks:5.1f}  " if self.nodes else "") +
                f"[prefill {self.prefill_s * 1e3:6.0f} ms  decode {self.decode_s:6.2f} s  "
                f"draft {self.draft_s * 1e3:5.0f} ms  rollback {self.rollback_s * 1e3:6.0f} ms]")


def _stop(tok: int, eos: list[int]) -> bool:
    return tok in eos


def generate_greedy(eng, prompt: torch.Tensor, max_new: int,
                    eos: list[int] | None = None,
                    record_gaps: bool = False,
                    pen: PenaltyState | None = None) -> tuple[list[int], DecodeStats]:
    """One token per forward pass. The baseline every speculative run must reproduce exactly."""
    eos = eos or []
    st = DecodeStats()
    eng.reset()
    if pen is not None:
        pen.seed(prompt.tolist())
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        logits = eng.forward(prompt, start=0, last_only=True)
    torch.cuda.synchronize()
    st.prefill_s = time.perf_counter() - t0
    pos = prompt.numel()

    def gap(lg):
        if not record_gaps:
            return
        two = lg[0, -1].float().topk(2).values
        st.gaps.append(float(two[0] - two[1]))
        st.tops.append(float(two[0]))

    if pen is not None:
        pen.apply_single(logits[0, -1])
    gap(logits)
    tok = int(logits[0, -1].argmax())
    out = [tok]
    if pen is not None:
        pen.commit([tok])
    t0 = time.perf_counter()
    with torch.no_grad():
        while len(out) < max_new and pos < eng.max_len and not _stop(tok, eos):
            logits = eng.forward(torch.tensor([tok], device=prompt.device), start=pos,
                                 last_only=True)
            pos += 1
            if pen is not None:
                pen.apply_single(logits[0, -1])
            gap(logits)
            tok = int(logits[0, -1].argmax())
            out.append(tok)
            if pen is not None:
                pen.commit([tok])
            st.blocks += 1
    torch.cuda.synchronize()
    st.decode_s = time.perf_counter() - t0
    st.tokens = len(out) - 1
    return out, st


def generate_spec(eng, prompt: torch.Tensor, max_new: int, drafter: Drafter, k: int,
                   eos: list[int] | None = None,
                   relax: Relax | None = None,
                   profile_misses: bool = False,
                   pen: PenaltyState | None = None,
                   sampler=None) -> tuple[list[int], DecodeStats]:
    eos = eos or []
    relax = relax or Relax()
    st = DecodeStats()
    eng.reset()
    drafter.reset()
    prompt_list = prompt.tolist()
    if pen is not None:
        pen.seed(prompt_list)
    if hasattr(drafter, "set_sampling"):
        # proposals are drawn under the request's profile, q carried per token.
        drafter.set_sampling(sampler)
    if hasattr(drafter, "prime"):
        drafter.prime(prompt_list)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        logits = eng.forward(prompt, start=0, last_only=True)
        if hasattr(drafter, "sync"):
            drafter.sync(prompt_list, eng.hidden_post_norm[0], 0)
    torch.cuda.synchronize()
    st.prefill_s = time.perf_counter() - t0
    pos = prompt.numel()
    if pen is not None:
        pen.apply_single(logits[0, -1])
    tok = sampler(logits[0, -1], index=len(prompt_list)) if sampler is not None and sampler.on \
        else int(logits[0, -1].argmax())
    out = [tok]
    if pen is not None:
        pen.commit([tok])
    drafter.observe([tok])
    ctx = prompt_list + [tok]

    torch.cuda.synchronize()
    t_dec = time.perf_counter()
    with torch.no_grad():
        # `pos < max_len - 1` because a block is at least two rows (anchor + one
        # draft) even when every output-token clamp is spent; tools may call with a
        # max_new the window cannot hold, and the answer is to stop, not to raise.
        while len(out) < max_new and pos < eng.max_len - 1 and not _stop(tok, eos):
            td = time.perf_counter()
            draft = drafter.propose(ctx, min(k, max_new - len(out)))
            st.draft_s += time.perf_counter() - td
            # the KV write counts ROWS -- the block is `[anchor] + draft` and the anchor's
            # own row is one of them -- while every clamp above counts OUTPUT tokens, and a
            # drafter is free to ignore the count it was handed (the block drafter proposes its
            # whole block). Cap on rows here, where the forward is about to be paid.
            draft = draft[:max(0, eng.max_len - pos - 1)]
            if not draft:
                # A drafter may decline, and the lookup drafter declines on most steps. A drafter
                # that holds a prediction head must not, and the constraint is not a preference:
                # the head conditions its next draft on the target's hidden state at the last
                # committed position, and a single-token step never computes that row. It computes
                # the hidden state of the token it was given, which produced the new token; the new
                # token's own hidden state does not exist until the next forward pass. So a head
                # cannot be brought current here, and `MergedRouter` is built never to reach this
                # line: the head always proposes something, and the router prices it rather than
                # silencing it.
                #
                # A drafter whose own cache is indexed by ABSOLUTE POSITION -- the block drafter's
                # is -- is the opposite case. Its conditioning for position `pos` is the target's
                # hidden state of the token AT `pos`, which this forward does compute, so it can
                # be brought current, and it must be: leave the gap and it falls permanently one
                # position behind the target and declines for ever after.
                prev = tok
                logits = eng.forward(torch.tensor([tok], device=prompt.device), start=pos,
                                     last_only=True)
                if pen is not None:
                    pen.apply_single(logits[0, -1])
                if hasattr(drafter, "sync"):
                    drafter.sync([prev], eng.hidden_post_norm[0], pos)
                pos += 1
                tok = sampler(logits[0, -1], index=len(ctx)) if sampler is not None and sampler.on \
                    else int(logits[0, -1].argmax())
                out.append(tok)
                ctx.append(tok)
                if pen is not None:
                    pen.commit([tok])
                drafter.observe([tok])
                st.blocks += 1
                continue
            block = torch.tensor([tok] + draft, device=prompt.device)
            tv = time.perf_counter()
            lg = eng.forward_block(block, start=pos)
            if pen is not None:
                pen.apply_chain(lg, draft)
            # What the block actually cost, handed to a drafter that prices its own choices. The
            # `.tolist()` above has already brought the device back in step, so this measures the
            # verify and nothing that was not going to be paid anyway; and a policy constant
            # belongs to the loop that pays it -- `tools/profile_block.py` once read a rollback at
            # 23.4 ms that this loop reads at 6.4.
            on_verify = getattr(drafter, "on_verify", None)
            if on_verify is not None:
                on_verify(len(draft) + 1, (time.perf_counter() - tv) * 1e3)
            if sampler is not None and sampler.on:
                # Rejection accept, q-aware where the drafter sampled; deterministic
                # arms take the shortcut. See engine/sample.py.
                qrows = getattr(drafter, "last_q", None)
                n, x = sampler.chain_accept(sampler.probs_rows(lg), draft, qrows, start=len(ctx))
                new = draft[:n] + [x]
            else:
                picks = lg.argmax(-1).tolist()
                n = 0
                for i, d in enumerate(draft):
                    if picks[i] == d:
                        n += 1
                        continue
                    if relax.on and relax.accepts(lg[i], d, picks[i]):
                        # The draft token stands, and every logit after it in this block was
                        # already computed conditioned on it, so the rest needs no recomputation.
                        st.relaxed += 1
                        n += 1
                        continue
                    break
                new = draft[:n] + [picks[n]]
            st.blocks += 1
            st.drafted += len(draft)
            st.accepted += n
            st.per_block.append(len(new))
            if pen is not None:
                pen.commit(new)
            if n < len(draft):
                tr = time.perf_counter()
                eng.rollback_to(n + 1)
                torch.cuda.synchronize()
                st.rollback_s += time.perf_counter() - tr
                st.rollbacks += 1
            if hasattr(drafter, "sync"):
                drafter.sync([int(x) for x in block[:n + 1]], eng.hidden_post_norm[0, :n + 1], pos)
            pos += n + 1
            for t in new:
                out.append(t)
                ctx.append(t)
                if _stop(t, eos):
                    break
            drafter.observe(new)
            tok = out[-1]
    torch.cuda.synchronize()
    st.decode_s = time.perf_counter() - t_dec
    st.tokens = len(out) - 1
    return out, st


def generate_spec_tree(eng, prompt: torch.Tensor, max_new: int, drafter, k: int,
                        eos: list[int] | None = None,
                        pen: PenaltyState | None = None,
                        sampler=None) -> tuple[list[int], DecodeStats]:
    """The same loop, with a draft TREE instead of a chain.

    The only structural difference is what a rejection costs. A chain that is wrong at slot 2 throws
    away slots 3..k with it; a tree keeps whatever branch the target did take. On this board that
    trade is close to free -- the verify step reads the same 19.4 GB of weights whether it carries
    two rows or sixteen -- so the drafter is asked for width and the node budget, not the depth, is
    what the router prices.

    `drafter.propose_tree(context, k)` returns an `engine.tree.DraftTree` or None. None is the same
    decline the chain loop handles: one ordinary step, with the drafter brought current afterwards.
    """
    eos = eos or []
    st = DecodeStats()
    eng.reset()
    drafter.reset()
    prompt_list = prompt.tolist()
    if pen is not None:
        pen.seed(prompt_list)
    if hasattr(drafter, "prime"):
        drafter.prime(prompt_list)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        logits = eng.forward(prompt, start=0, last_only=True)
        if hasattr(drafter, "sync"):
            drafter.sync(prompt_list, eng.hidden_post_norm[0], 0)
    torch.cuda.synchronize()
    st.prefill_s = time.perf_counter() - t0
    pos = prompt.numel()
    if pen is not None:
        pen.apply_single(logits[0, -1])
    tok = sampler(logits[0, -1], index=len(prompt_list)) if sampler is not None and sampler.on \
        else int(logits[0, -1].argmax())
    out = [tok]
    if pen is not None:
        pen.commit([tok])
    drafter.observe([tok])
    ctx = prompt_list + [tok]
    st.nodes = 0

    torch.cuda.synchronize()
    t_dec = time.perf_counter()
    with torch.no_grad():
        while len(out) < max_new and pos < eng.max_len - 1 and not _stop(tok, eos):
            td = time.perf_counter()
            tree = drafter.propose_tree(ctx, min(k, max_new - len(out)))
            st.draft_s += time.perf_counter() - td
            # the tree's KV write counts NODES (the anchor is node 0), and a drafter may
            # return more nodes than the budget it was handed. A DFS pre-order prefix is still a
            # valid tree, so cutting at the row bound only drops candidates.
            if tree is not None:
                tree = tree.truncate(eng.max_len - pos)
            if tree is None or tree.n_draft == 0:
                prev = tok
                logits = eng.forward(torch.tensor([tok], device=prompt.device), start=pos,
                                     last_only=True)
                if pen is not None:
                    pen.apply_single(logits[0, -1])
                if hasattr(drafter, "sync"):
                    drafter.sync([prev], eng.hidden_post_norm[0], pos)
                pos += 1
                tok = sampler(logits[0, -1], index=len(ctx)) if sampler is not None and sampler.on \
                    else int(logits[0, -1].argmax())
                out.append(tok)
                ctx.append(tok)
                if pen is not None:
                    pen.commit([tok])
                drafter.observe([tok])
                st.blocks += 1
                continue
            block = torch.tensor(tree.tokens, device=prompt.device)
            tv = time.perf_counter()
            lg = eng.forward_tree(block, tree.parents, start=pos)
            if pen is not None:
                pen.apply_tree(lg, tree)
            on_verify = getattr(drafter, "on_verify", None)
            if on_verify is not None:
                on_verify(tree.n_draft + 1, (time.perf_counter() - tv) * 1e3)
            if sampler is not None and sampler.on:
                # Rejection accept down the tree: the target's own token is sampled at
                # every node and the walk follows the child carrying it; see engine/sample.py.
                path, new = sampler.tree_walk(sampler.probs_rows(lg), tree.tokens, tree.parents,
                                              start=len(ctx), q=tree.q)
            else:
                picks = lg.argmax(-1).tolist()
                path, new = eng.accept_tree(tree, picks)
            n = len(path) - 1
            st.blocks += 1
            st.nodes += tree.n_draft
            st.drafted += tree.n_draft
            st.accepted += n
            st.per_block.append(len(new))
            if pen is not None:
                pen.commit(new)
            tr = time.perf_counter()
            eng.commit_tree(path)
            torch.cuda.synchronize()
            st.rollback_s += time.perf_counter() - tr
            if n < tree.n_draft:
                st.rollbacks += 1
            if hasattr(drafter, "sync"):
                hid = eng.hidden_post_norm[0, torch.tensor(path, device=prompt.device)]
                _sync(drafter, [int(tree.tokens[i]) for i in path], hid, pos, path)
            pos += n + 1
            for t in new:
                out.append(t)
                ctx.append(t)
                if _stop(t, eos):
                    break
            drafter.observe(new)
            tok = out[-1]
    torch.cuda.synchronize()
    st.decode_s = time.perf_counter() - t_dec
    st.tokens = len(out) - 1
    return out, st


def _sync(drafter, tokens, hidden, first_pos, rows) -> None:
    """Bring a drafter current after a tree block.

    A drafter that reads the hidden states this loop hands it needs no more than the rows of the
    accepted path, which is what `hidden` already is. One does not: the block drafter takes five
    mid-stack residual streams off the engine's tap, and the tap holds one row per NODE, in DFS
    order. Which of those rows the target committed to is `rows`, so it gets them.
    """
    if getattr(drafter, "wants_rows", False):
        drafter.sync(tokens, hidden, first_pos, rows=rows)
    else:
        drafter.sync(tokens, hidden, first_pos)


class ThinkBudget:
    """A cap on how long the model is allowed to reason, enforced by the engine.

    The template opens the reasoning block in the generation prompt itself -- with thinking on it
    ends in `<think>\n` -- so the model is inside the block from the first token it writes, and the
    only thing that ends the block is the model deciding to write `</think>`. On a hard problem it
    can decide that very late, and the tokens in between are paid for at the same rate as the
    answer.

    Budget forcing is the documented way to stop it: when the budget is spent, the block is closed
    FOR the model, by appending a short sentence that says why and then the closing tag, and
    generation continues from there into the answer. The sentence matters -- an abrupt `</think>`
    leaves the model mid-thought and the answers get worse -- and this is the phrasing the vendor's
    own examples use.

    **This changes the output.** It is not a speed trick that preserves what the model would have
    said; it is an instruction to stop thinking and answer, and on a problem that needed the
    thinking the answer will be worse. Every number measured under a budget has to say which budget
    it was measured under.
    """

    PHRASE = ("\n\nConsidering the limited time by the user, I have to give the solution based on "
              "the thinking directly now.\n</think>\n\n")

    # Stall detection. While the block is open, `arms` (the session monitor's reasoning-loop
    # task, ops/supervisor.yaml) watch for the model going in circles and set `reason`; the
    # monitor decides whether that closes the block (server/conductor/monitor/). Without `arms`
    # nothing is computed and only the budget can close the block. Three arms, all deterministic
    # and reading committed tokens only:
    #   * loop: a 1..16-token pattern repeated back to back, counted only when the repeats span
    #     128 tokens or more -- four repeats of a one-token pattern used to fire it, and 51 of its
    #     52 closes in the opencode database were `0x000000`, `0+0+0+0+` and their like;
    #   * repeat (the history arm): the share of the last 256 tokens' 16-grams that already
    #     occurred earlier in the block. A loop whose period is a paragraph of several hundred
    #     tokens shares nothing between neighbouring windows and everything with its own past;
    #   * novelty: the fraction of new 8-grams in the last window against the window before it.
    # The parameters are the policy's; `StallArms` holds their state.

    def __init__(self, tokenizer, budget: int = 0, phrase: str | None = None,
                 arms: dict | None = None):
        self.budget = int(budget or 0)
        self.arms_cfg = arms or None
        self.open_id = _special_id(tokenizer, "<think>")
        self.end_id = _special_id(tokenizer, "</think>")
        # The model does not always emit the SPECIAL token: on prompts that look like raw text
        # (harness traffic, some code-edit contexts) it writes the literal characters instead, and
        # a budget that only watches the special id never arms -- measured 2026-09-19, e05/e06/
        # e08/e10 rambled 256+ identical tokens inside an unclosed literal <think>. Watch both.
        self.open_text = tokenizer("<think>", add_special_tokens=False).input_ids
        self.end_text = tokenizer("</think>", add_special_tokens=False).input_ids
        self._seen: list[int] = []
        self.close_ids = tokenizer(phrase or self.PHRASE,
                                   add_special_tokens=False).input_ids
        self.n = 0
        self.inside = False
        self.done = False
        self.reason: str | None = None
        # when the block closed and when the engine forced it (perf_counter), for the live
        # view; each is written once per request, in a branch that already runs once
        self.t_closed: float | None = None
        self.t_forced: float | None = None
        self._arms: StallArms | None = None

    def start(self, prompt_ids: list[int]) -> "ThinkBudget":
        """Arm for one request. The prompt's own tail says whether the block is already open."""
        self.n, self.inside, self.done = 0, False, False
        self.reason = None
        self.t_closed = self.t_forced = None
        self._arms = StallArms(self.arms_cfg) if self.arms_cfg else None
        self._seen = []
        tail = list(prompt_ids[-6:])
        if self.open_id in tail:
            self.inside = self.end_id not in tail[tail.index(self.open_id):]
        elif self._suffix_is(self.open_text, prompt_ids):
            self.inside = not self._suffix_is(self.end_text, prompt_ids)
        return self

    @staticmethod
    def _suffix_is(seq: list[int], ids) -> bool:
        """True when the tail of `ids` ends with the token sequence `seq`."""
        return bool(seq) and len(ids) >= len(seq) and list(ids)[-len(seq):] == list(seq)

    def observe(self, ids) -> None:
        for t in ids:
            if self.done:
                return
            # Keep a short tail so the literal "<think>" / "</think>" sequences are detectable
            # token-by-token, whichever form the model chose.
            self._seen.append(int(t))
            if len(self._seen) > 8:
                del self._seen[:-8]
            if not self.inside:
                if t == self.open_id or self._suffix_is(self.open_text, self._seen):
                    self.inside = True
                continue
            if t == self.end_id or self._suffix_is(self.end_text, self._seen):
                self.inside, self.done = False, True
                self.t_closed = time.perf_counter()
            else:
                self.n += 1
                if self._arms is None or self.reason is not None:
                    continue
                self.reason = self._arms.feed(int(t), self.n)

    def closes_in(self, ids) -> bool:
        """Would these tokens, observed next, close the open block? Nothing is observed."""
        if not self.inside or self.done:
            return False
        seen = list(self._seen)
        for t in ids:
            seen = (seen + [int(t)])[-8:]
            if t == self.end_id or self._suffix_is(self.end_text, seen):
                return True
        return False

    @property
    def share(self) -> float | None:
        """The history arm's last repeat share, None before its first check."""
        return self._arms.share if self._arms is not None else None

    @property
    def hit(self) -> bool:
        """The budget is spent. A stall only sets `reason`; the monitor decides on a close."""
        if self.done or not self.inside:
            return False
        return bool(self.budget) and self.n >= self.budget


class StallArms:
    """The reasoning-loop detector's three arms over one reasoning block (see `ThinkBudget`).

    `feed(token, n)` takes the block's n-th token and returns the arm's reason once one fires:
    `loop(size=s count=c span=p)`, `repeat16(share=v at=n)` or `novelty(v)`. The parameters
    are the policy's (`reasoning_loop.arms` in ops/supervisor.yaml); an arm left out does not
    run. Per token: a comparison per pattern size, one 16-gram hash, one append; every 64 tokens
    past 2,048 the window's 241 hashes are looked up. Memory: one int per token and the set of
    the hashes that left the window.
    """

    def __init__(self, cfg: dict):
        lp, self.rep, self.nov = cfg.get("loop"), cfg.get("repeat"), cfg.get("novelty")
        self.share: float | None = None
        self._sizes: range = range(0)
        if lp:
            # size s fires when the last max(min_count, ceil(min_span / s)) blocks of s tokens are
            # equal: each token then equals the one s back, `need[s]` times in a row
            self._sizes = range(lp["min_size"], lp["max_size"] + 1)
            self._need = [0] * (lp["max_size"] + 1)
            for s in self._sizes:
                self._need[s] = (max(lp["min_count"], -(-lp["min_span"] // s)) - 1) * s
            self._runs = [0] * (lp["max_size"] + 1)
            self._tail: list[int] = []
            self._keep = lp["max_size"]
        if self.rep:
            self._grams: list[int] = []      # the hash of the gram starting at each token
            self._last: list[int] = []       # the last `gram` tokens
            self._seen: set[int] = set()     # grams that end before the window
            self._streak = 0
        if self.nov:
            self._recent: list[int] = []
            self._next = self.nov["min_tokens"]

    def feed(self, t: int, n: int) -> str | None:
        why = None
        if self._sizes:
            tail, runs, need = self._tail, self._runs, self._need
            k = len(tail)
            for s in self._sizes:
                if s <= k and tail[-s] == t:
                    runs[s] += 1
                    if why is None and runs[s] >= need[s]:
                        why = f"loop(size={s} count={need[s] // s + 1} span={need[s] + s})"
                else:
                    runs[s] = 0
            tail.append(t)
            if k >= self._keep:
                del tail[0]
        if self.rep:
            why = self._repeat(t, n) if why is None else why
        if self.nov:
            w = self.nov["window"]
            rec = self._recent
            rec.append(t)
            if len(rec) > 2 * w:
                del rec[:-2 * w]
            if why is None and n >= self._next and len(rec) >= 2 * w:
                self._next = n + w // 2
                v = self._novelty()
                if v < self.nov["below"]:
                    why = f"novelty({v:.2f})"
        return why

    def _repeat(self, t: int, n: int) -> str | None:
        r, last = self.rep, self._last
        g, w = r["gram"], r["window"]
        last.append(t)
        if len(last) > g:
            del last[0]
        if n >= g:
            self._grams.append(hash(tuple(last)))
        j = n - w - g                        # the gram starting here ends just before the window
        if j >= 0:
            self._seen.add(self._grams[j])
        if n < r["min_tokens"] or n < w or n % r["every"]:
            return None
        window = self._grams[n - w:n - g + 1]
        seen = self._seen
        v = sum(1 for h in window if h in seen) / len(window)
        self.share = v
        self._streak = self._streak + 1 if v >= r["act"] else 0
        if self._streak >= r["in_a_row"]:
            return f"repeat{g}(share={v:.2f} at={n})"
        return None

    def _novelty(self) -> float:
        """Fraction of the last window's grams that do not occur in the window before it."""
        w, g = self.nov["window"], self.nov["gram"]
        last, prev = self._recent[-w:], self._recent[-2 * w:-w]
        n = w - g
        if n <= 0:
            return 1.0
        new = {tuple(last[i:i + g]) for i in range(n)}
        old = {tuple(prev[i:i + g]) for i in range(n)}
        return len(new - old) / n


def _special_id(tokenizer, text: str) -> int:
    i = tokenizer.convert_tokens_to_ids(text)
    if isinstance(i, int) and i >= 0 and i != getattr(tokenizer, "unk_token_id", None):
        return i
    ids = tokenizer(text, add_special_tokens=False).input_ids
    return int(ids[-1])
