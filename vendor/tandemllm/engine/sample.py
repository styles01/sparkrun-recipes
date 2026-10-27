"""Sampling: the decoder the greedy fixed point cannot be.

The engine's exact path is greedy, and under greedy a repeated token is a stable point no
penalty can break (measured; see the knowledge card on greedy loops). The model is designed to
run with sampling -- temperature, top-p and the presence penalty are its own loop-breakers -- so
a request that asks for sampling must get real sampling, not a 400.

`temperature`, `min_p`, `top-k` and `top-p` are applied to a row AFTER the penalties, `logit_bias`
and the no-repeat rule (deterministic transforms, still meaningful under sampling), with a
per-request `torch.Generator` so a `seed` reproduces a request exactly. `min_p` keeps the
tokens whose probability is at least `min_p` times the top one's, after the temperature (vLLM's
order). Like the other filters it is part of `_filter`, the one function every target row goes
through -- the single draw, `probs_rows` for the chain and the tree, the keyed draws -- so the
walks below accept against the filtered distribution and stay exact; a sampling drafter draws its
proposal through the same filter, and its q is what it drew from, which is all the q-aware accept
asks of it. Under greedy it is a no-op: the argmax always survives it.

Both halves are here. The single-token path samples one row (`__call__`). The speculative path
verifies a draft chain or a draft tree by rejection sampling (`chain_pick`, `tree_walk`): the
target's OWN token is drawn at each node, and a drafted token survives exactly while the draw
lands on it -- if the draw lands elsewhere, that draw IS the rejection's residual sample. This
is the textbook sampler specialised to a deterministic drafter: with proposal q = delta_d,

    accept with min(1, p(d)/q(d)) = p(d), and the residual (p - q)+ is p restricted to x != d,

which is precisely what "draw x ~ p; if x == d accept, else x is the token" computes -- the same
distribution, with no q to estimate and no second draw. The output therefore follows the
target's own sampled distribution, speculation or not, which is the property `verify_lossless`
checks for greedy and `test_sample.py` checks statistically here.

Greedy is untouched: `temperature = 0` is the argmax it always was, byte for byte.

**A seed reproduces a request, whatever the drafter did.** One generator stream shared
by every draw cannot promise that: the drafter's own proposals and the accept's draws come off it
in an order the drafter decides, and the length router decides the block width from wall-clock
timing, so two runs of the same seeded request consumed the stream differently and diverged. A
seeded request therefore draws with noise KEYED BY POSITION instead: the token at sequence index
`t` is `argmax(log p_t + G_t)`, with `G_t` Gumbel noise from a generator seeded by `(seed, t)` --
the Gumbel-max trick, so `x_t ~ p_t` exactly. The drafter proposes with the SAME `G_t` against its
own `q_t` (`argmax(log q_t + G_t)`), so its proposal and the target's draw agree whenever the two
distributions put the same token on top of that noise, and the accept is "keep the draft while it
equals the target's draw" -- the walk. Every emitted token is the target's own keyed draw,
so the output is the drafter-less, width-less keyed sample, identical across widths, arms, trees
and lookup proposals (to the same one-ulp row arithmetic greedy carries). An unseeded request keeps
the q-aware `min(1, p/q)` accept, which accepts more; a seeded one trades that for reproduction.
"""

from __future__ import annotations

import math
import os

import torch

from engine.settings import SETTINGS as _S

# The top-p filter without the full-row sort (`_top_p_fast`): the nucleus is read off the row's
# top TOPP_K logits whenever their mass already crosses top_p (a row whose nucleus is wider falls
# back to the sort). The kept set is the sort's; the cumulative mass is summed under the full
# row's logsumexp instead of the sorted row's softmax normaliser, which can move a token whose
# cumulative mass sits within float rounding of top_p (counted by tools/sample_filter_check.py).
# 0 = the sort, as before.
TOPP_K = int(_S.get("SAMPLE_TOPP_K") or 0)
# The tree walk's and the drafter's draws batched on the device and brought to the host in ONE
# copy (`tree_walk_batched`): every node's draw is taken up front, which leaves each visited
# node's draw distributed exactly as the one-at-a-time walk's. Unseeded requests only (a seeded
# request keeps its position-keyed draws). 0 = one draw and one copy a node, as before.
BATCH = _S.get("SAMPLE_BATCH") == "1"
# The batched walk's draws taken inside each row's nucleus (`row_draws`): the kept set is
# `_top_p_fast`'s, computed by the same ops on the same tensors (bit-identical), and the draw is a
# multinomial over that row's top TOPP_K slice instead of over the [n, V] probabilities `_filter`
# materialises (four full-vocabulary passes fewer a row). The same distribution: p(i) proportional
# to exp(logit_i / T) over the nucleus either way. A row whose nucleus is wider than TOPP_K takes the
# sort as `_top_p_fast` does. Unseeded det walks only (no q, no min_p / top_k); everything else is
# `probs_rows` + `tree_walk` as before. Default on since 2026-10-09 19:10 (sampled 50-prompt row
# base vs nucleus 2 x 2 runs a side: no class worse beyond the run-to-run spread; board gate
# tools/sampled_dist.py nospec/det/cool PASS, greedy identical). 0 = the probs_rows path.
NUCLEUS = _S.get("SAMPLE_NUCLEUS") == "1"


_MASK64 = (1 << 64) - 1


def _key(seed: int, index: int) -> int:
    """A generator seed for position `index` of a request seeded with `seed` (splitmix64)."""
    x = (int(seed) * 0x9E3779B97F4A7C15 + int(index) + 1) & _MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _MASK64
    return (x ^ (x >> 31)) >> 1


class Sampler:
    """Temperature / top-k / top-p over rows of target logits, one per request."""

    __slots__ = ("temperature", "top_p", "top_k", "seed", "generator", "draft_temperature",
                 "min_p")

    def __init__(self, temperature: float = 0.0, top_p: float = 1.0, top_k: int = 0,
                 seed: int | None = None, draft_temperature: float | None = None,
                 min_p: float = 0.0):
        if temperature < 0.0 or not temperature == temperature:
            raise ValueError(f"temperature must be >= 0, got {temperature}")
        if not 0.0 < top_p <= 1.0:
            raise ValueError(f"top_p must be in (0, 1], got {top_p}")
        if top_k < 0:
            raise ValueError(f"top_k must be >= 0, got {top_k}")
        if not 0.0 <= min_p <= 1.0:
            raise ValueError(f"min_p must be in [0, 1], got {min_p}")
        self.min_p = float(min_p)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.top_k = int(top_k)
        self.seed = None if seed is None else int(seed)
        # how sharply the DRAFTER's proposal distribution is tempered, relative to the
        # request. A cooler draft is sharper, and `min(1, p(d)/q(d))` then approaches `p(d)`.
        # None -> the drafter's own default (QWEN38_DRAFT_TEMP).
        self.draft_temperature = None if draft_temperature is None else float(draft_temperature)
        if self.draft_temperature is not None and not 0.02 <= self.draft_temperature <= 1.0:
            raise ValueError(f"draft_temperature must be in [0.02, 1], got {draft_temperature}")
        self.generator: torch.Generator | None = None

    @property
    def on(self) -> bool:
        # `temperature = 0` is greedy whatever top-p and top-k say -- OpenAI's reading and vLLM's,
        # and the only one that makes sense: both filters shrink a candidate set the argmax is
        # already the maximum of. Reading a lone top-p as "sampling on" made
        # `{"temperature": 0, "top_p": 0.95}` sample the RAW logits, because `_filter` divides by
        # the temperature only when there is one, and it also took the request out of the response
        # cache, which the server consults only for a greedy answer.
        return self.temperature > 0.0

    @property
    def coupled(self) -> bool:
        """Draws keyed by position, so a seed reproduces the request exactly."""
        return self.on and self.seed is not None

    def key(self) -> tuple:
        """Joins the response-cache key: sampled answers are not memoised at all today (the
        server skips the cache when sampling is on), but the identity is kept for completeness."""
        return (round(self.temperature, 6), round(self.top_p, 6), self.top_k, self.seed,
                round(self.min_p, 6))

    def for_choice(self, i: int) -> "Sampler":
        """Choice `i` of an `n > 1` request: the same profile and its own draws. A seeded
        request derives the choice's seed from its own, so every choice reproduces and no two
        share their noise; choice 0 is the request itself."""
        if i == 0:
            return self
        return Sampler(self.temperature, self.top_p, self.top_k,
                       seed=None if self.seed is None else _key(self.seed, -1 - i),
                       draft_temperature=self.draft_temperature, min_p=self.min_p)

    def _rng(self, device) -> torch.Generator:
        if self.generator is None:
            self.generator = torch.Generator(device=device)
            seed = self.seed if self.seed is not None else int.from_bytes(os.urandom(8), "little")
            self.generator.manual_seed(seed)
        return self.generator

    @torch.no_grad()
    def _filter(self, logits: torch.Tensor) -> torch.Tensor:
        """The requested distribution over the last dim of a `[..., V]` float tensor.

        The top-p filter is the HF one: sort descending, keep the shortest prefix whose cumulative
        mass exceeds `top_p`, always keep at least the first token.
        """
        if self.temperature > 0.0:
            logits = logits / self.temperature
        if self.min_p > 0.0:
            # p_i >= min_p * p_max  <=>  logit_i >= logit_max + log(min_p), on the tempered row
            floor = logits.amax(dim=-1, keepdim=True) + math.log(self.min_p)
            logits = logits.masked_fill(logits < floor, float("-inf"))
        if self.top_k > 0:
            k = min(self.top_k, logits.shape[-1])
            cutoff = torch.topk(logits, k, dim=-1).values[..., -1, None]
            logits = torch.where(logits < cutoff, torch.full_like(logits, float("-inf")), logits)
        if self.top_p < 1.0 and TOPP_K > 0:
            logits = self._top_p_fast(logits, TOPP_K)
        elif self.top_p < 1.0:
            sorted_logits, order = torch.sort(logits, descending=True, dim=-1)
            probs = torch.softmax(sorted_logits, dim=-1)
            cumulative = torch.cumsum(probs, dim=-1)
            drop = cumulative > self.top_p
            drop[..., 1:] = drop[..., :-1].clone()   # keep the first token that crosses
            drop[..., 0] = False
            sorted_logits = sorted_logits.masked_fill(drop, float("-inf"))
            logits = torch.empty_like(logits).scatter_(-1, order, sorted_logits)
        return torch.softmax(logits, dim=-1)

    @torch.no_grad()
    def _top_p_fast(self, logits: torch.Tensor, k0: int) -> torch.Tensor:
        """`_filter`'s top-p step from the top `k0` logits of each row (see TOPP_K)."""
        V = logits.shape[-1]
        k0 = min(int(k0), V)
        flat = logits.reshape(-1, V)
        vals, idx = torch.topk(flat, k0, dim=-1, sorted=True)
        lse = torch.logsumexp(flat, dim=-1, keepdim=True)
        cumulative = torch.cumsum(torch.exp(vals - lse), dim=-1)
        drop = cumulative > self.top_p
        drop[..., 1:] = drop[..., :-1].clone()   # keep the first token that crosses
        drop[..., 0] = False
        out = torch.full_like(flat, float("-inf")).scatter_(-1, idx, vals.masked_fill(drop, float("-inf")))
        wide = cumulative[:, -1] <= self.top_p    # the nucleus is wider than k0: the sort
        if k0 < V and bool(wide.any()):
            rows = wide.nonzero().flatten()
            sorted_logits, order = torch.sort(flat[rows], descending=True, dim=-1)
            probs = torch.softmax(sorted_logits, dim=-1)
            cum = torch.cumsum(probs, dim=-1)
            d = cum > self.top_p
            d[..., 1:] = d[..., :-1].clone()
            d[..., 0] = False
            sorted_logits = sorted_logits.masked_fill(d, float("-inf"))
            out[rows] = torch.empty_like(sorted_logits).scatter_(-1, order, sorted_logits)
        return out.view_as(logits)

    def nucleus_ok(self) -> bool:
        """`row_draws` applies: sampling with a top-p and no other filter, the fast filter on."""
        return (self.on and TOPP_K > 0 and self.top_p < 1.0 and self.top_k == 0
                and self.min_p == 0.0)

    @torch.no_grad()
    def row_draws(self, rows: torch.Tensor) -> torch.Tensor:
        """One token a row of `rows` [n, V] (logits, penalties applied), drawn from `_filter`'s
        distribution without building it (`nucleus_ok()` must hold). The kept set is computed by
        `_top_p_fast`'s own ops on the same tensors; the draw is a multinomial over the kept part
        of the row's top-k slice. Rows whose nucleus is wider than k take the sort, as there."""
        V = rows.shape[-1]
        flat = rows.reshape(-1, V).float() / self.temperature
        k0 = min(int(TOPP_K), V)
        vals, idx = torch.topk(flat, k0, dim=-1, sorted=True)
        lse = torch.logsumexp(flat, dim=-1, keepdim=True)
        cumulative = torch.cumsum(torch.exp(vals - lse), dim=-1)
        drop = cumulative > self.top_p
        drop[..., 1:] = drop[..., :-1].clone()   # keep the first token that crosses
        drop[..., 0] = False
        g = self._rng(rows.device)
        pick = torch.multinomial(torch.softmax(vals.masked_fill(drop, float("-inf")), dim=-1),
                                 1, generator=g)
        x = idx.gather(1, pick).squeeze(1)
        wide = cumulative[:, -1] <= self.top_p
        if k0 < V and bool(wide.any()):
            rws = wide.nonzero().flatten()
            sorted_logits, order = torch.sort(flat[rws], descending=True, dim=-1)
            probs = torch.softmax(sorted_logits, dim=-1)
            cum = torch.cumsum(probs, dim=-1)
            d = cum > self.top_p
            d[..., 1:] = d[..., :-1].clone()
            d[..., 0] = False
            p = torch.softmax(sorted_logits.masked_fill(d, float("-inf")), dim=-1)
            x[rws] = order.gather(1, torch.multinomial(p, 1, generator=g)).squeeze(1)
        return x

    def walk_draws(self, xs: list[int], tokens: list[int], parents: list[int]
                   ) -> tuple[list[int], list[int]]:
        """`tree_walk` on draws already taken, one a node (`row_draws`; no q)."""
        n = len(tokens)
        kids: list[list[int]] = [[] for _ in range(n)]
        for c in range(1, n):
            kids[parents[c]].append(c)
        path, node = [0], 0
        while True:
            x = xs[node]
            nxt = next((k for k in kids[node] if tokens[k] == x), None)
            if nxt is None:
                break
            path.append(nxt)
            node = nxt
        return path, [tokens[i] for i in path[1:]] + [x]

    def tree_walk_logits(self, rows: torch.Tensor, tokens: list[int], parents: list[int],
                         start: int | None = None, q: list | None = None
                         ) -> tuple[list[int], list[int]]:
        """`tree_walk(probs_rows(rows), ...)`, by `row_draws` where NUCLEUS and the walk allow."""
        keyed = self.coupled and start is not None
        if (NUCLEUS and BATCH and not keyed and (q is None or all(v is None for v in q))
                and self.nucleus_ok()):
            n = len(tokens)
            return self.walk_draws(self.row_draws(rows[:n]).tolist(), tokens, parents)
        return self.tree_walk(self.probs_rows(rows), tokens, parents, start=start, q=q)

    @torch.no_grad()
    def draws(self, probs: torch.Tensor) -> torch.Tensor:
        """One token from every row of `probs` [n, V], on the device, on the request's RNG."""
        return torch.multinomial(probs, 1, generator=self._rng(probs.device)).squeeze(-1)

    @torch.no_grad()
    def probs_rows(self, rows: torch.Tensor) -> torch.Tensor:
        """`[n, V]` distributions for a verified block's rows (penalties already applied)."""
        return self._filter(rows.float())

    def pick(self, probs_row: torch.Tensor) -> int:
        """One token from one distribution row, in generation order on the request's RNG."""
        return int(torch.multinomial(probs_row, 1, generator=self._rng(probs_row.device)))

    @torch.no_grad()
    def pick_at(self, probs_row: torch.Tensor, index: int) -> int:
        """The token for sequence position `index`: Gumbel-max under the position's own noise.

        `argmax(log p + G)` with `G` iid Gumbel is a draw from `p`, and `G` depends only on
        `(seed, index)` -- so the same position gets the same draw whichever path reached it, and
        a drafter handed a different row gets the same noise (see the module docstring).
        """
        g = self._rng(probs_row.device)
        g.manual_seed(_key(self.seed if self.seed is not None else 0, index))
        u = torch.rand(probs_row.shape[-1], generator=g, device=probs_row.device)
        gumbel = -torch.log(-torch.log(u.clamp_min(1e-20)))
        return int(torch.argmax(torch.log(probs_row.float()) + gumbel))

    @torch.no_grad()
    def chain_accept(self, dists: torch.Tensor, draft: list[int],
                     qrows: list[torch.Tensor | None] | None = None,
                     start: int | None = None) -> tuple[int, int]:
        """Accept a draft chain, q-aware where the drafter sampled its proposal.

        Each position either carries a real proposal distribution `q` -- the drafter sampled the
        token from it, so the textbook rule applies: accept with `min(1, p(d)/q(d))`, and on
        rejection draw from the residual `(p - q)+` renormalised -- or none, which is a
        deterministic drafter whose point mass `q = delta_d` makes the accept `p(d)` and the
        residual the draw itself (the `chain_pick` shortcut). Mixing the two per position is
        exact: every position's acceptance uses the distribution its token was proposed from.
        Returns `(accepted, first new token)`, the same contract as `chain_pick`.

        `start` is the sequence index of the first row's token. A seeded request passes it and
        takes the keyed walk instead, whatever `qrows` says.
        """
        if self.coupled and start is not None:
            return self.chain_pick(dists, draft, start=start)
        for i, d in enumerate(draft):
            if qrows is None or qrows[i] is None:
                x = self.pick(dists[i])
                if x != d:
                    return i, x
                continue
            p_row, q_row = dists[i], qrows[i]
            pd, qd = float(p_row[d]), float(q_row[d])
            u = float(torch.rand(1, generator=self._rng(p_row.device),
                               device=p_row.device))
            if qd > 0.0 and u * qd < pd:
                continue
            r = torch.clamp(p_row - q_row, min=0.0)
            total = float(r.sum())
            x = self.pick(r / total) if total > 0.0 else self.pick(p_row)
            return i, x
        return len(draft), self.pick(dists[len(draft)])

    @torch.no_grad()
    def __call__(self, row: torch.Tensor, index: int | None = None) -> int:
        """One token from one logits row. `temperature = 0` is the argmax, exactly as before.
        `index` is the token's sequence position, which a seeded request draws against."""
        if not self.on:
            return int(row.argmax())
        if self.coupled and index is not None:
            return self.pick_at(self._filter(row.float()), index)
        return self.pick(self._filter(row.float()))

    @torch.no_grad()
    def chain_pick(self, dists: torch.Tensor, draft: list[int],
                   start: int | None = None) -> tuple[int, int]:
        """Accept a draft chain by rejection sampling; returns `(accepted, first new token)`.

        `dists` is `[len(draft) + 1, V]`: every draft position's row plus the bonus row after the
        last draft. Accepted drafts keep the block's later rows valid; the first draw that lands
        off the draft ends the block there, and that draw is the token. With `start` on a seeded
        request each row draws against its own position's noise.
        """
        keyed = self.coupled and start is not None

        def draw(i):
            return self.pick_at(dists[i], start + i) if keyed else self.pick(dists[i])

        for i, d in enumerate(draft):
            x = draw(i)
            if x != d:
                return i, x
        return len(draft), draw(len(draft))

    @torch.no_grad()
    def tree_walk(self, dists: torch.Tensor, tokens: list[int], parents: list[int],
                  start: int | None = None, q: list | None = None) -> tuple[list[int], list[int]]:
        """The same accept, down a draft tree; returns `(node path, new tokens)`.

        At each node the target's own token is drawn and the walk follows the child carrying it.
        Where no child carries it, the draw is the token and the walk stops -- byte-for-byte the
        greedy `accept_tree` walk with the sample in place of the argmax. With `start` on a seeded
        request a node at depth j draws against position `start + j`'s noise.

        `q` (`engine/tree.py::spine_tree`): per node, the distribution its token was sampled
        from, or None. A node whose child carries one is recursive rejection sampling with that child
        first: accept it with `min(1, p(d)/q(d))`; otherwise draw from the residual `(p - q)+` and
        follow any child carrying the draw -- the spine child included, which the residual can still
        land on -- the other children being deterministic candidates, for which "draw, then look" IS
        the rejection step. A seeded request ignores q and takes the keyed walk (as `chain_accept`).
        """
        keyed = self.coupled and start is not None
        if BATCH and not keyed:
            return self._tree_walk_batched(dists, tokens, parents, q)
        path, node = [0], 0
        while True:
            kids = [c for c in range(node + 1, len(tokens)) if parents[c] == node]
            qc = (next((c for c in kids if q[c] is not None), None)
                  if q is not None and not keyed else None)
            p_row = dists[node]
            if qc is not None:
                d, q_row = tokens[qc], q[qc]
                pd, qd = float(p_row[d]), float(q_row[d])
                u = float(torch.rand(1, generator=self._rng(p_row.device), device=p_row.device))
                if qd > 0.0 and u * qd < pd:
                    path.append(qc)
                    node = qc
                    continue
                r = torch.clamp(p_row - q_row, min=0.0)
                total = float(r.sum())
                x = self.pick(r / total) if total > 0.0 else self.pick(p_row)
            else:
                x = (self.pick_at(p_row, start + len(path) - 1) if keyed else self.pick(p_row))
            nxt = next((c for c in kids if tokens[c] == x), None)
            if nxt is None:
                break
            path.append(nxt)
            node = nxt
        new = [tokens[i] for i in path[1:]] + [x]
        return path, new

    @torch.no_grad()
    def _tree_walk_batched(self, dists: torch.Tensor, tokens: list[int], parents: list[int],
                           q: list | None) -> tuple[list[int], list[int]]:
        """`tree_walk` (unseeded) with every node's randomness drawn up front on the device and
        brought over in one copy: per node the target's own draw from p, and where a child
        carries a q, the accept's uniform, p(d), q(d) and a draw from the residual (p - q)+ (from
        p where the residual is empty). Each visited node uses exactly the draws the one-at-a-time
        walk would take there, all independent: the same distribution of paths."""
        n = len(tokens)
        kids: list[list[int]] = [[] for _ in range(n)]
        for c in range(1, n):
            kids[parents[c]].append(c)
        qc = [next((c for c in kids[i] if q[c] is not None), None) if q is not None else None
              for i in range(n)]
        dev = dists.device
        g = self._rng(dev)
        x_all = torch.multinomial(dists[:n], 1, generator=g).squeeze(-1)
        qn = [i for i in range(n) if qc[i] is not None]
        if qn:
            rows = torch.tensor(qn, device=dev)
            p_rows = dists[rows]
            q_rows = torch.stack([q[qc[i]] for i in qn]).to(p_rows.dtype)
            d = torch.tensor([tokens[qc[i]] for i in qn], device=dev).view(-1, 1)
            pd = p_rows.gather(1, d).squeeze(1)
            qd = q_rows.gather(1, d).squeeze(1)
            u = torch.rand(len(qn), generator=g, device=dev)
            r = torch.clamp(p_rows - q_rows, min=0.0)
            total = r.sum(-1, keepdim=True)
            r = torch.where(total > 0.0, r / total.clamp_min(1e-38), p_rows)
            xr = torch.multinomial(r, 1, generator=g).squeeze(-1)
            host = torch.cat([x_all.double(), pd.double(), qd.double(), u.double(),
                              xr.double()]).tolist()
            m = len(qn)
            xs = [int(v) for v in host[:n]]
            pdl, qdl, ul = host[n:n + m], host[n + m:n + 2 * m], host[n + 2 * m:n + 3 * m]
            xrl = [int(v) for v in host[n + 3 * m:]]
            slot = {i: j for j, i in enumerate(qn)}
        else:
            xs = x_all.tolist()
        path, node = [0], 0
        while True:
            c = qc[node]
            if c is not None:
                j = slot[node]
                if qdl[j] > 0.0 and ul[j] * qdl[j] < pdl[j]:
                    path.append(c)
                    node = c
                    continue
                x = xrl[j]
            else:
                x = xs[node]
            nxt = next((k for k in kids[node] if tokens[k] == x), None)
            if nxt is None:
                break
            path.append(nxt)
            node = nxt
        new = [tokens[i] for i in path[1:]] + [x]
        return path, new
