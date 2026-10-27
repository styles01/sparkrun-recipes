"""Deterministic anti-repetition penalties, exact under speculative verification.

The engine decodes greedily and refuses to sample, so when the model's argmax settles on one
token there is nothing in the decoder that can break the loop -- measured 2026-09-18, a "Super
Jump Bros" prompt whose pasted level rows are literal `" C C C …"` strings looped ~7,000 `C`
tokens inside the reasoning block.

The fix is the one vLLM and SGLang ship: penalties on the TARGET's logits, before the argmax.
The property that makes that safe here is the one the whole engine already rests on --
acceptance is everywhere "draft token == target argmax" -- so any deterministic function of the
committed history applied to the row before argmax keeps the speculative path's output identical
to the unspeculated one, *under the same rule*. vLLM applies penalties to the target logits before
the rejection argmax (`v1/sample/rejection_sampler.py`); SGLang does the same in "a relaxed
version" because its verifier is not argmax-exact. This one is exact:

  * chain blocks -- vLLM's per-position history: row i of `[anchor] + draft` is conditioned on
    the committed history plus `draft[:i]`, so rows are penalized left to right, one draft token
    entering the working counts after the row that precedes it;
  * tree blocks -- ancestor history: a node is conditioned on its ANCESTOR PATH, not its
    siblings, and the rows arrive in DFS pre-order, so a depth stack adds each node's token on
    the way down and removes it on the way back up;
  * rollback costs nothing: only accepted tokens are ever committed to the base counts, so a
    rejected block leaves no penalty state behind.

Defaults are the identities -- rep 1.0, presence 0, frequency 0 -- and an off spec short-circuits
every entry point, so a run that does not ask for penalties pays nothing and produces the bytes
it produced before this file existed. The response cache key gains the three values, because
greedy-under-penalties is a different function.

`logit_bias` rides on the same machinery: OpenAI's `{token: bias}` added to every target
row at every decision site, before the penalties (vLLM's order). It depends on nothing at all, so
the argument above holds a fortiori -- every row, chain, tree or single, gets the same addition,
and "draft == target argmax" stays the acceptance rule.

The pass costs one gather of the touched columns, a batched elementwise pass over
`[rows, touched]`, and one scatter back. Measured on the board at 16 rows and a 120k-token
history: 0.64 ms a block when the history touches 20k distinct tokens (ordinary text), 1.34 ms in
the worst case where every token is distinct -- against a ~99 ms verify. A full-vocab masked pass
was 4.2 ms and is not what runs.
"""

from __future__ import annotations

import torch


class PenaltySpec:
    """The three standard penalties, the no-repeat rule and `logit_bias`. Defaults are today's
    output."""

    __slots__ = ("rep", "presence", "freq", "no_repeat", "bias")

    def __init__(self, rep: float = 1.0, presence: float = 0.0, freq: float = 0.0,
                 no_repeat: int = 0, bias: dict | None = None):
        if rep <= 0.0:
            raise ValueError(f"repetition_penalty must be > 0, got {rep}")
        if not -2.0 <= presence <= 2.0:
            raise ValueError(f"presence_penalty must be in [-2, 2], got {presence}")
        if not -2.0 <= freq <= 2.0:
            raise ValueError(f"frequency_penalty must be in [-2, 2], got {freq}")
        if no_repeat and no_repeat < 2:
            raise ValueError(f"no_repeat_ngram_size must be 0 or >= 2, got {no_repeat}")
        self.rep = float(rep)
        self.presence = float(presence)
        self.freq = float(freq)
        self.no_repeat = int(no_repeat or 0)
        # `{token id: added logit}`, None when there is none
        self.bias = ({int(k): float(v) for k, v in bias.items() if v} or None) if bias else None

    @property
    def penalizes(self) -> bool:
        """Any rule that reads the history: the count vector and the n-gram index exist."""
        return (self.rep != 1.0 or self.presence != 0.0 or self.freq != 0.0
                or self.no_repeat > 0)

    @property
    def on(self) -> bool:
        return self.penalizes or self.bias is not None

    def key(self) -> tuple:
        """The identity that joins the response-cache key: two runs with the same key answer the
        same question, and two runs with different keys do not."""
        k = (round(self.rep, 6), round(self.presence, 6), round(self.freq, 6), self.no_repeat)
        return k + (tuple(sorted(self.bias.items())),) if self.bias else k

    def __repr__(self) -> str:
        return (f"PenaltySpec(rep={self.rep:g}, presence={self.presence:g}, "
                f"freq={self.freq:g}, no_repeat={self.no_repeat}"
                f"{f', bias={len(self.bias)}' if self.bias else ''})")


class PatternStop:
    """End a generation that is repeating one short pattern forever.

    The penalties above are the industry's first answer and this is its second, for the case the
    first cannot fix: under GREEDY decoding a repeated token is a stable fixed point, and a
    penalty strong enough to break a +20-logit "C" loop is strong enough to damage ordinary text.
    Measured 2026-09-18: the Super Jump Bros prompt still looped on `" C C C …"` at presence 2.0 /
    repetition 1.5 / frequency 0.3, because the loop lives in the reasoning block where the model
    is *writing* those rows deliberately.

    So the backstop does not try to choose differently -- it stops. After every committed block,
    the last `count * size` tokens are checked against every pattern size in `[min_size, max_size]`:
    if the tail is one block repeated `count` times, the generation ends (the server reports
    `finish_reason: "stop"` and annotates its log line). Tokens are never altered; the cost is
    `max_size * count` comparisons a block in pure Python, and the whole thing is off unless asked
    for. This is vLLM's `RepetitionDetectionParams` shape (`max_pattern_size`, `min_pattern_size`,
    `min_count`).
    """

    def __init__(self, max_size: int = 0, min_size: int = 1, count: int = 0):
        self.max_size = int(max_size)
        self.min_size = max(1, int(min_size))
        self.count = int(count)
        self.tail: list[int] = []
        self.hit = False
        self.pattern: tuple[int, tuple[int, ...]] | None = None

    @property
    def on(self) -> bool:
        return self.max_size > 0 and self.count >= 2

    @property
    def label(self) -> str | None:
        """`size=2 count=8` once a pattern has been found, for the server's log line."""
        if not self.hit or self.pattern is None:
            return None
        # The tokens are the evidence the next reader needs to judge whether the pattern was a
        # loop or formatting (a false positive on generated code cut a file mid-line on
        # 2026-09-18; without the tokens there was nothing to check).
        toks = ",".join(str(t) for t in self.pattern[1][:8])
        return f"size={self.pattern[0]} count={self.count} tok=[{toks}]"

    def observe(self, ids) -> bool:
        """Add committed tokens; return True once a repeating pattern is detected."""
        if not self.on or self.hit:
            return self.hit
        self.tail.extend(int(t) for t in ids)
        keep = self.max_size * self.count
        if len(self.tail) > keep:
            del self.tail[:-keep]
        n = len(self.tail)
        for size in range(self.min_size, self.max_size + 1):
            span = size * self.count
            if n < span:
                continue
            base = n - span
            block = self.tail[base:base + size]
            if all(self.tail[base + k * size:base + (k + 1) * size] == block
                   for k in range(1, self.count)):
                self.hit = True
                self.pattern = (size, tuple(block))
                return True
        return False


class PenaltyState:
    """The count vector behind the penalties, one per request, seeded from the prompt.

    `seed` and `commit` take token ids (python ints); `apply_*` take the verify's logits rows and
    penalize them IN PLACE, so the caller computes picks/argmax afterwards exactly as it did
    before. An off spec makes every method a no-op and allocates nothing.
    """

    def __init__(self, spec: PenaltySpec, vocab_size: int, device: str):
        self.spec = spec
        self.vocab = int(vocab_size)
        self.device = device
        self.counts: torch.Tensor | None = None      # committed history, int32 [vocab]
        # no-repeat-n-gram: the committed token sequence and, per (n-1)-gram suffix,
        # the tokens that completed it before. Capped to the last `window` tokens.
        self.history: list[int] = []
        self.followers: dict[tuple, set[int]] = {}
        self.window = 2048
        # The mask is a thinking-phase tool. Measured 2026-09-19: with it active over the whole
        # stream it damaged generated code -- a final file with an EMPTY <script> body, corrupted
        # markup (`<div\nid=...>`, a word split across lines), substituted literals (`height:102%`)
        # and a model that could not rewrite its own draft and ended with "I keep truncating".
        # A hard mask cannot distinguish a runaway loop from a draft, boilerplate, a repeated
        # level row or the literal 100; the answer is left to the conservative pattern guard,
        # whose cut is now visible.
        self.mask = True
        # And it is a PENALTY, not a prohibition: -inf forced an early EOS when a reasoning trace
        # echoed its own context (chat efa916ed, 06:48: 676 tokens, content empty). A token that
        # is confidently preferred can now win; a tight loop still pays the penalty every step,
        # which the stall detector and the budget then close properly.
        self.mask_penalty = 4.0
        self._bias: tuple[torch.Tensor, torch.Tensor] | None = None

    def _add_bias(self, rows: torch.Tensor) -> None:
        """`logit_bias` on a row (1-D) or on every row of a block (2-D), in place."""
        if self.spec.bias is None:
            return
        if self._bias is None or self._bias[0].device != rows.device:
            ids = sorted(self.spec.bias)
            self._bias = (torch.tensor(ids, dtype=torch.long, device=rows.device),
                          torch.tensor([self.spec.bias[i] for i in ids], dtype=torch.float32,
                                       device=rows.device))
        idx, vals = self._bias
        vals = vals.to(rows.dtype)
        if rows.dim() == 1:
            rows.index_add_(0, idx, vals)
        else:
            rows.index_add_(1, idx, vals.unsqueeze(0).expand(rows.shape[0], -1))

    # --- history ------------------------------------------------------------------------------
    def seed(self, ids) -> None:
        """Start a request: RESET, then add. `seed` is what a fresh request begins with, so a
        reused state (the gate tools run plain and speculative back to back) starts clean."""
        if not self.spec.penalizes:
            return
        if self.counts is None:
            self.counts = torch.zeros(self.vocab, dtype=torch.int32, device=self.device)
        else:
            self.counts.zero_()
        # The no-repeat history/index belongs to the request too: a reused state (the gate tools
        # run plain and speculative back to back) must not carry the previous run's n-grams into
        # this one. The first penalized gate FAIL was exactly that, at decision 0.
        self.history.clear()
        self.followers.clear()
        self._add(ids)

    def commit(self, ids) -> None:
        self._add(ids)

    @torch.no_grad()
    def _add(self, ids) -> None:
        if not self.spec.penalizes or not ids:
            return
        if self.counts is None:
            self.counts = torch.zeros(self.vocab, dtype=torch.int32, device=self.device)
        self.counts.index_add_(0, torch.tensor(list(ids), dtype=torch.long, device=self.device),
                               torch.ones(len(ids), dtype=torch.int32, device=self.device))
        n = self.spec.no_repeat
        if n > 0:
            h = self.history
            old = len(h)
            h.extend(int(t) for t in ids)
            if len(h) > self.window:
                # Trim FIRST and index what is kept. Indexing the whole addition and then
                # rebuilding over the kept half built a 262k-entry index for a 262k-token prompt
                # only to throw all but the last thousand away.
                del h[:len(h) - self.window // 2]
                self._index_history()
                return
            for i in range(max(n - 1, old), len(h)):
                self.followers.setdefault(tuple(h[i - (n - 1):i]), set()).add(h[i])

    def _index_history(self) -> None:
        """Rebuild the suffix index over the retained window (amortised over evictions)."""
        n = self.spec.no_repeat
        h = self.history
        self.followers.clear()
        for i in range(n - 1, len(h)):
            self.followers.setdefault(tuple(h[i - (n - 1):i]), set()).add(h[i])

    def _mask(self, row: torch.Tensor, extra: list[int]) -> None:
        """Forbid the token(s) that completed this suffix before -- HF's no_repeat_ngram_size.

        The suffix is the last `n - 1` tokens of the row's own history: `extra` carries the
        in-block (or ancestor-path) tokens above the row, and the committed sequence is
        `self.history`. Tokens the suffix completed before pay a finite penalty (default 4.0 nats),
        never -inf: a confidently preferred token can still win, so the mask can never force an
        early EOS (chat efa916ed); a tight loop pays the penalty every step and the stall detector
        closes it. A suffix whose blocked set covers the vocabulary is skipped.
        """
        n = self.spec.no_repeat
        if n <= 0 or not self.mask:
            return
        L = n - 1
        tail = (self.history + extra)[-L:]
        blocked = set(self.followers.get(tuple(tail), ()))
        if extra:
            # The committed index cannot know n-grams formed INSIDE this block (or on this tree
            # path), and the greedy path would have indexed them by the time it decides the same
            # position -- so scan the block's own tokens too. Occurrences that start in the
            # committed sequence and straddle into `extra` are the reason the scan starts at
            # `len(history) - L + 1` rather than at the block boundary.
            seq = self.history + extra
            for j in range(max(0, len(self.history) - L + 1), len(seq) - L):
                if seq[j:j + L] == tail:
                    blocked.add(seq[j + L])
        if blocked and len(blocked) < self.vocab:
            row[torch.tensor(sorted(blocked), dtype=torch.long, device=row.device)] -= self.mask_penalty

    # --- rows ---------------------------------------------------------------------------------
    @torch.no_grad()
    def apply_single(self, row: torch.Tensor) -> None:
        """A one-row decision: the prefill's first token, a declined block's step, the token
        after a forced reasoning close. History is the committed counts, nothing else."""
        if not self.spec.on:
            return
        self._add_bias(row)
        if not self.spec.penalizes:
            return
        assert self.counts is not None, "apply before seed"
        self._mask(row, [])
        idx = self._block_index(None)
        sub = row.index_select(0, idx).unsqueeze(0)
        cntm = self.counts.index_select(0, idx).unsqueeze(0)
        self._apply_matrix(sub, cntm)
        row.index_copy_(0, idx, sub[0])

    @torch.no_grad()
    def _block_index(self, extra) -> torch.Tensor:
        """The sorted unique tokens the history or the block mentions.

        The penalty only ever touches columns the history contains (or the block is about to), and
        for a 120k-token history that is tens of thousands of columns out of 248,320 -- so the
        block's rows are gathered to these columns, penalized there, and scattered back. A
        full-vocab masked pass costs 4.2 ms a block on the board; this is a fifth of that.
        """
        idx = self.counts.nonzero().flatten()
        if extra:
            ex = torch.tensor(sorted({int(t) for t in extra}), dtype=torch.long,
                              device=self.device)
            idx = torch.unique(torch.cat([idx, ex]))
        return idx

    @torch.no_grad()
    def _apply_matrix(self, sub: torch.Tensor, cntm: torch.Tensor) -> None:
        """The three penalties, batched over `[rows, n]` gathered rows and their per-row counts."""
        s = self.spec
        m = cntm > 0
        if s.rep != 1.0:
            neg = sub < 0
            sub.copy_(torch.where(m, torch.where(neg, sub * s.rep, sub / s.rep), sub))
        if s.presence != 0.0:
            sub -= s.presence * m
        if s.freq != 0.0:
            sub -= s.freq * cntm.to(sub.dtype) * m

    @torch.no_grad()
    def apply_chain(self, lg: torch.Tensor, draft: list[int]) -> None:
        """Rows of `forward_block([anchor] + draft)`, in place, left to right.

        Row i is conditioned on the committed history plus `draft[:i]` (the anchor is committed
        and already in the counts), so row i's counts are row i-1's plus `draft[i-1]` -- built as
        a `[rows, n]` matrix over the touched columns and applied in one batched pass.
        """
        if not self.spec.on:
            return
        self._add_bias(lg)
        if not self.spec.penalizes:
            return
        assert self.counts is not None, "apply before seed"
        rows = lg.shape[0]
        if self.spec.no_repeat > 0:
            for i in range(rows):
                self._mask(lg[i], list(draft[:i]))
        idx = self._block_index(draft)
        n = idx.numel()
        sub = lg.index_select(1, idx)
        cntm = self.counts.index_select(0, idx).unsqueeze(0).expand(rows, n).clone()
        if draft:
            pos = torch.searchsorted(
                idx, torch.tensor([int(t) for t in draft], dtype=torch.long, device=self.device))
            for i in range(1, rows):
                cntm[i].copy_(cntm[i - 1])
                cntm[i, pos[i - 1]] += 1
        self._apply_matrix(sub, cntm)
        lg.index_copy_(1, idx, sub)

    @torch.no_grad()
    def apply_tree(self, lg: torch.Tensor, tree) -> None:
        """Rows of `forward_tree`, in place, each with its ANCESTOR history.

        DFS pre-order means a node's ancestors are the stack's contents after popping back to
        its depth; siblings leave nothing behind. Node 0 is the anchor and is already committed,
        so it is in the base counts and is not pushed again. The per-node count vectors are
        assembled over the touched columns and applied in one batched pass, like the chain.
        """
        if not self.spec.on:
            return
        self._add_bias(lg)
        if not self.spec.penalizes:
            return
        assert self.counts is not None, "apply before seed"
        rows = lg.shape[0]
        depths = tree.depths()
        if self.spec.no_repeat > 0:
            ancestors: list[int] = []
            for j in range(rows):
                d = depths[j]
                while len(ancestors) >= max(d, 1):
                    ancestors.pop()
                self._mask(lg[j], list(ancestors))
                if j > 0:
                    ancestors.append(int(tree.tokens[j]))
        idx = self._block_index(tree.tokens[1:])
        n = idx.numel()
        sub = lg.index_select(1, idx)
        base = self.counts.index_select(0, idx)
        cntm = base.unsqueeze(0).expand(rows, n).clone()
        pos = torch.searchsorted(
            idx, torch.tensor([int(t) for t in tree.tokens], dtype=torch.long, device=self.device))
        work = base.clone()
        # `path` holds the positions (in `idx`) of the current DFS chain at depths 1..d-1 -- the
        # anchor (depth 0) is a committed token and already in the base counts, so it is never
        # pushed. A node at depth d needs exactly d-1 stacked tokens: pop while the stack is
        # deeper than that (`max(d, 1)` keeps the d=0 row from popping an empty stack).
        path: list[int] = []
        for j in range(rows):
            d = depths[j]
            while len(path) >= max(d, 1):
                work[path.pop()] -= 1
            cntm[j].copy_(work)
            if j > 0:
                p = int(pos[j])
                work[p] += 1
                path.append(p)
        self._apply_matrix(sub, cntm)
        lg.index_copy_(1, idx, sub)

    # The three formulas live in `_apply_matrix` (one code path for every call site):
    # rep is multiplicative (the CTRL/HF form: a negative logit is multiplied by it, a positive
    # one divided by it), presence is a flat subtraction on tokens that occur at all, and
    # frequency subtracts per occurrence. All three touch only columns the history contains,
    # which is what makes presence and frequency differ and what makes rep a no-op on fresh
    # vocabulary.
