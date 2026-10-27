"""StairCut with its price model and its calibration, written once for every model (ENG-245).

Each round keeps the part of the draft that maximises expected committed tokens per millisecond,

    (1 + sum of q over the kept nodes) / (V(rows) + draft),

where q of a node is the chance the walk reaches and takes it (`ProbDeciles`: the draft head's
probability mapped to the measured rate of acceptance) and V(rows) what a verify block of that many
rows costs with its commit (`PriceModel`: a prior from the model's price file, followed online).
The cut decides only which rows are verified; every kept row is the target's to accept or not, so a
cut moves speed and never text.

Context classes: verify prices, draft costs and the plain step's cost are kept per context class (the
caller's `ctx_class`, Flash-Next's: the context length its sparse attention scores), since a block
that pays at 1k can cost three decode steps at 256k. Whether a class drafts at all is decided on what
each mode really took a token, by wall time (the caller prices a round by its whole span, not by its
verify alone, engine/methods/mtp_drafter.py): `skip` sends a class to the plain step while its drafted
rounds cost more a token than its plain steps (the price file's `spec` and `plain` priors until both
are learned), drafting one round in `PROBE` to measure again -- an interval that doubles while
drafting keeps losing (up to `PROBE_MAX`) and starts over when it wins.

A turn starts with a backlog: the prompt's rows the draft head has not read (`prime`, one attention
and MoE layer over the context per row: 0.2-0.3 ms a row at 256k). `turn` drafts the turn only when
what speculation saves over the expected answer (the plain step's ms a token less a drafted round's,
both learned per class) pays for that backlog; else the turn takes the plain step throughout and the
backlog is never read. On the P0 runs a drafted round at 260,800 tokens cost as much a token as the
plain step, so the backlog (2,748 rows, 0.76 s) made speculation the slower of the two. Every learned
cost starts from its prior (a first sample moves it as much as a later one), and what a class learned of
its drafted rounds ages back to the prior while its turns take the plain step, so one slow request does
not keep a class plain for good.

Strategies: `StairCut.cut` is `chain_siblings` -- a spine (the head's greedy chain) and, with
`siblings`, one leaf a depth (the head's second token there, a child of the spine node above it) --
moved here from Flash-Next's package as it was. With a lookup continuation (`look`, ENG-229) the
round is one tree of both sources -- the spine, and the lookup's chain where it leaves the spine (or
runs on past its end) -- admitted best-first by q; where the two agree, a spine node's q is that of
either being right. The 27B's best-first admission over any tree (engine/router.py) joins as a
further strategy in ENG-257.
"""

from __future__ import annotations

import json
import os


class PriceModel:
    """The verify-and-commit price of a block of `rows` rows, in ms, by kind of block ("chain",
    "tree") and context class: a prior table from a price file, followed online (EMA 0.1 a count, from where the prior puts it; a
    block over 4 times its price is a stall and not learned);
    the counts not measured yet follow the prior shifted by the learned offset (a long context moves
    every count), and a price never falls with more rows (a cut must not buy a row for nothing because
    one count was measured at a busier moment than the next).

    The file (ops/prices/<model>/<profile>.<device>.json, and the 27B's ops/stair-tables-*.json):

        {"<ctx class>": {"<kind>": {"<rows>": ms, ...}, ...}, ..., "step_ms": 3.7, "measured": "..."}

    A context class not in the file takes the nearest lower one (the lowest, below all); rows between
    two measured counts interpolate, rows past the last add `step_ms` a row (else the last two counts'
    slope). Without a file every count starts at 1 ms, the same for all: the cut drafts to the full
    depth until the first rounds price it."""

    OUTLIER = 4.0         # a block this many times its price is not learned (with a prior)

    def __init__(self, table: dict | None = None, step_ms: float | None = None, ema: float = 0.1):
        self.table = {int(c): {str(k): {int(r): float(ms) for r, ms in rows.items()} for k, rows in kinds.items()}
                      for c, kinds in (table or {}).items()}
        self.step, self.ema = step_ms, ema
        self.V: dict = {}              # (kind, class) -> {rows: learned ms}
        self.delta: dict = {}          # (kind, class) -> learned shift of the prior

    @classmethod
    def load(cls, path: str, log=print) -> "PriceModel":
        if not os.path.isfile(path):
            log(f"[prices] no price file {path}: the cut learns its prices online")
            return cls()
        with open(path) as f:
            doc = json.load(f)
        table = {k: v for k, v in doc.items() if k.lstrip("-").isdigit()}
        for c, kinds in table.items():
            for kind, rows in kinds.items():
                for r, ms in rows.items():
                    if int(r) < 1 or not 0.0 < float(ms) < 10_000.0:
                        raise ValueError(f"price file {path}: [{c}][{kind}][{r}] = {ms!r} is not a row count and ms")
        return cls(table, doc.get("step_ms"))

    def _rows(self, kind: str, ctx_class: int) -> dict | None:
        if not self.table:
            return None
        below = [c for c in self.table if c <= ctx_class]
        return self.table[max(below) if below else min(self.table)].get(kind)

    def prior(self, rows: int, kind: str = "chain", ctx_class: int = 0) -> float | None:
        tbl = self._rows(kind, ctx_class)
        if not tbl:
            return None
        if rows in tbl:
            return tbl[rows]
        ks = sorted(tbl)
        if rows > ks[-1]:
            step = self.step if self.step is not None else (
                (tbl[ks[-1]] - tbl[ks[-2]]) / (ks[-1] - ks[-2]) if len(ks) > 1 else 0.0)
            return tbl[ks[-1]] + step * (rows - ks[-1])
        if rows < ks[0]:
            return tbl[ks[0]]
        lo = max(k for k in ks if k < rows)
        hi = min(k for k in ks if k > rows)
        return tbl[lo] + (tbl[hi] - tbl[lo]) * (rows - lo) / (hi - lo)

    def prices(self, n: int, kind: str = "chain", ctx_class: int = 0) -> list[float]:
        """[0, price of 1 row, ..., price of n rows]."""
        V = self.V.get((kind, ctx_class), {})
        d = self.delta.get((kind, ctx_class), 0.0)
        out, p = [0.0], 0.0
        for r in range(1, n + 1):
            v = V.get(r)
            if v is None:
                pr = self.prior(r, kind, ctx_class)
                v = pr + d if pr is not None else 1.0
            p = max(p, v)
            out.append(p)
        return out

    def price(self, rows: int, kind: str = "chain", ctx_class: int = 0) -> float:
        return self.prices(rows, kind, ctx_class)[rows]

    def observe(self, rows: int, ms: float, kind: str = "chain", ctx_class: int = 0) -> None:
        key, a = (kind, ctx_class), self.ema
        pr = self.prior(rows, kind, ctx_class)
        if pr is not None and ms > self.OUTLIER * self.price(rows, kind, ctx_class):
            return                  # a stall (a first request's allocations, a capture elsewhere)
        V, d = self.V.setdefault(key, {}), self.delta.get(key, 0.0)
        # a count's first sample moves it from where the prior put it, as later ones do: the first
        # request's rounds ran 2-3 times their steady cost on 2026-10-12, and one of them was a price
        start = V.get(rows, pr + d if pr is not None else None)
        V[rows] = ms if start is None else (1 - a) * start + a * ms
        if pr is not None:
            self.delta[key] = (1 - a) * d + a * (ms - pr)

    def learned(self, kind: str = "chain", ctx_class: int = 0) -> tuple[dict, float]:
        """(learned ms by row count, learned offset) of one kind and class, for reports."""
        return dict(self.V.get((kind, ctx_class), {})), self.delta.get((kind, ctx_class), 0.0)


class ProbDeciles:
    """The draft head's probability mapped to the measured rate of acceptance, per kind of node
    ("spine", "sib") and tenth of probability: it starts at the probability itself (a prior worth
    `prior_n` rounds a bin), learns from every verified round, and halves old evidence past `cap`
    rounds a bin, so it follows the text."""

    def __init__(self, kinds=("spine", "sib", "look"), prior_n: float = 4.0, cap: float = 400.0):
        self.prior_n, self.cap = prior_n, cap
        self.cal = {k: [[0.0, 0.0] for _ in range(10)] for k in kinds}

    def q(self, kind: str, p: float) -> float:
        hits, n = self.cal[kind][min(int(p * 10), 9)]
        return (hits + self.prior_n * p) / (n + self.prior_n)

    def learn(self, kind: str, p: float, hit: bool) -> None:
        b = self.cal[kind][min(int(p * 10), 9)]
        b[0] += float(hit)
        b[1] += 1.0
        if b[1] > self.cap:
            b[0] *= 0.5
            b[1] *= 0.5


class StairCut:
    """The cut over a draft head's chain (and, with `siblings`, one second-choice leaf a depth). q of a
    spine node: the product of the calibrated first-token probabilities down to it; q of a sibling at
    depth k: the chance the walk reaches depth k, rejects the spine token there and takes the sibling.
    `kind`: the price table's kind for its blocks. `frozen`: observe nothing (a warm-up run)."""

    PROBE = 16            # a skipping class still drafts one round in this many ...
    PROBE_MAX = 256       # ... doubled after each probe that loses, up to this
    OUTLIER = 4.0         # a plain step or draft this many times its price is not learned
    AGE = 0.2             # a plain turn moves what a class learned of its drafted rounds this far to the prior
    FORGET = 8            # plain turns after which a class with no `spec` prior forgets its drafted rounds

    def __init__(self, prices: PriceModel, calibrator: ProbDeciles | None = None, siblings: bool = True,
                 max_rows: int = 32, kind: str = "chain"):
        self.prices, self.cal = prices, calibrator or ProbDeciles()
        self.siblings, self.max_rows, self.kind = siblings, max_rows, kind
        self.draft = {}            # context class -> the draft's cost, ms (EMA 0.1)
        self.plain = {}            # context class -> the plain step's cost, ms (EMA 0.1)
        self.spec = {}             # context class -> a drafted round's ms a token, its whole wall time
        self.rounds = {}           # context class -> [ms, tokens] of a drafted round (EMA `SPEC_EMA` each)
        self.prime = {}            # context class -> the head's ms a backlog row (EMA 0.1)
        self.skipped, self.probe = {}, {}
        self.plain_turns = {}      # context class -> plain turns in a row (`turn`)
        self.age, self.returned = {}, {}   # context class -> its ageing rate; its last turn came back by ageing
        self.cls = 0
        self.last = None
        self.frozen = False

    @property
    def draft_ms(self) -> float | None:
        return self.draft.get(self.cls)

    def plain_ms(self, ctx_class: int) -> float | None:
        """The plain step's cost at a class: learned, else the price file's `plain` prior, else None."""
        p = self.plain.get(ctx_class)
        return p if p is not None else self.prices.prior(1, "plain", ctx_class)

    def _base(self, price: list, draft: float) -> float:
        """Tokens a ms of the anchor alone (a one-row block): a block is cut against it."""
        return 1.0 / (price[1] + draft)

    def spec_ms(self, ctx_class: int) -> float | None:
        """A drafted round's ms a token at a class: learned, else the price file's `spec` prior."""
        s = self.spec.get(ctx_class)
        return s if s is not None else self.prices.prior(1, "spec", ctx_class)

    def skip(self, ctx_class: int = 0) -> bool:
        """True: this round takes the plain step without a draft, because this class's drafted rounds
        cost more a token than its plain steps -- except one round in the class's probe interval, which
        drafts (its cost updates `spec`); every probe doubles the interval, up to `PROBE_MAX`, and a
        class whose drafts win again starts over at `PROBE`."""
        spec, plain = self.spec_ms(ctx_class), self.plain_ms(ctx_class)
        if spec is None or plain is None or spec <= plain:
            self.probe[ctx_class], self.skipped[ctx_class] = self.PROBE, 0
            return False
        self.skipped[ctx_class] = self.skipped.get(ctx_class, 0) + 1
        n = self.probe.get(ctx_class, self.PROBE)
        if self.skipped[ctx_class] < n:
            return True
        self.skipped[ctx_class], self.probe[ctx_class] = 0, min(2 * n, self.PROBE_MAX)
        return False

    def _ema(self, d: dict, kind: str, ms: float, ctx_class: int) -> None:
        """EMA 0.1 from the learned value, else from the price file's prior (a first sample moves it no
        more than a later one: the first request's rounds ran 2-3 times their steady cost), else the
        sample; one over `OUTLIER` times the reference is a stall and not learned."""
        if self.frozen:
            return
        p = d.get(ctx_class)
        ref = p if p is not None else self.prices.prior(1, kind, ctx_class)
        if ref is not None and ms > self.OUTLIER * ref:
            return
        d[ctx_class] = ms if ref is None else 0.9 * ref + 0.1 * ms

    SEED_TOKENS = 2.0     # the tokens a drafted round of the `spec` prior stands for
    SPEC_EMA = 0.05       # a drafted round's weight in `spec`: one request's first rounds (the class's
                          # first replays, the kernels' first calls) moved it past the plain step at 0.1

    def observe_spec(self, ms: float, ctx_class: int = 0, tokens: int = 1) -> None:
        """A drafted round's wall time and the tokens it committed. `spec` is the ratio of the two EMAs
        (from the prior's, `SEED_TOKENS` tokens a round), so a round of one token and one of four weigh
        by what they committed: the EMA of each round's own ms a token sat 30 % above the request's
        ms a token on 2026-10-12 (26.5 against 19.4 at 1k) and sent a drafting class to the plain step."""
        if self.frozen:
            return
        m = self.rounds.get(ctx_class)
        if m is None:
            prior = self.prices.prior(1, "spec", ctx_class)
            m = [prior * self.SEED_TOKENS, self.SEED_TOKENS] if prior is not None else None
        if m is not None and ms / tokens > self.OUTLIER * m[0] / m[1]:
            return
        a = self.SPEC_EMA
        m = [ms, float(tokens)] if m is None else [(1 - a) * m[0] + a * ms, (1 - a) * m[1] + a * tokens]
        self.rounds[ctx_class], self.spec[ctx_class] = m, m[0] / m[1]

    def observe_prime(self, ms_per_row: float, ctx_class: int = 0) -> None:
        self._ema(self.prime, "prime", ms_per_row, ctx_class)

    def turn(self, ctx_class: int, backlog: int, answer: float, extra_ms: float = 0.0) -> bool:
        """Draft this turn? True unless the class has a plain price, a drafted round's price a token
        and a backlog row's price (learned, else the price file's priors), and what the drafts would
        save over `answer` tokens does not pay for reading `backlog` rows into the head and for
        `extra_ms` more (the graphs the class lacks, which the engine captures while idle).

        A class that takes plain turns measures no drafted round, so what it learned of them ages: each
        plain turn moves it `AGE` of the way back to the price file's prior (with none, it is dropped
        after `FORGET` plain turns). A class whose drafts lost to a passing stall drafts again once the
        prior says drafting wins, and measures again (2026-10-12: one cold first request at 1k priced its
        drafted rounds above the plain step, and 1k took the plain step on every turn after); one whose
        prior says drafting loses stays plain and pays no backlog to find out. A turn that came back by
        ageing and lost again halves the class's rate (to `AGE` / 32, about 250 plain turns a return);
        two drafting turns in a row set it back to `AGE`."""
        plain = self.plain_ms(ctx_class)
        spec = self.spec_ms(ctx_class)
        prime = self.prime.get(ctx_class, self.prices.prior(1, "prime", ctx_class))
        if plain is None or spec is None or prime is None:
            return True
        n = self.plain_turns.get(ctx_class, 0)
        if (plain - spec) * answer > backlog * prime + extra_ms:
            if n == 0:
                self.age[ctx_class] = self.AGE              # drafted on from a drafting turn: the full rate
            self.returned[ctx_class], self.plain_turns[ctx_class] = n > 0, 0
            return True
        if not self.frozen and ctx_class in self.spec and (plain - spec) * answer <= backlog * prime:
            rate = self.age.get(ctx_class, self.AGE)
            if n == 0 and self.returned.get(ctx_class):
                rate = self.age[ctx_class] = max(rate / 2, self.AGE / 32)   # the return lost
                self.returned[ctx_class] = False
            n = self.plain_turns[ctx_class] = n + 1
            prior = self.prices.prior(1, "spec", ctx_class)
            m = self.rounds[ctx_class]
            if prior is not None:
                m[0] = (1 - rate) * m[0] + rate * prior * m[1]
                self.spec[ctx_class] = m[0] / m[1]
            elif n * rate >= self.FORGET * self.AGE:
                del self.spec[ctx_class], self.rounds[ctx_class]
        return False

    def observe_plain(self, ms: float, ctx_class: int = 0) -> None:
        """A plain step's measured cost at this class (the loop's own step, its argmax included); a
        stall the caller could not see (a capture elsewhere, a GC pause) is not learned."""
        self._ema(self.plain, "plain", ms, ctx_class)

    def cut(self, anchor: int, tok: list, prob: list, look: tuple | None = None,
            ctx_class: int = 0) -> tuple[list[int], list[int]]:
        """tok / prob: per depth the (first, second) draft token and its probability; `look`: a lookup
        continuation (tokens, confidence) or None; `ctx_class`: the prices' context class. Returns the
        block in DFS pre-order: (tokens, parents), row 0 the anchor (alone: take the plain step)."""
        self.cls = ctx_class
        if look is not None and look[0]:
            return self._merged(anchor, tok, prob, list(look[0]), float(look[1]))
        D = len(tok)
        q = self.cal.q
        reach, spine_q, sib_q = 1.0, [], []
        for k in range(D):
            c1 = q("spine", prob[k][0])
            sib_q.append(reach * (1 - c1) * q("sib", prob[k][1]) if self.siblings else 0.0)
            reach *= c1
            spine_q.append(reach)
        draft = self.draft_ms or 0.0
        price = self.prices.prices(min(self.max_rows, 2 * D + 1), self.kind, self.cls)
        best = (self._base(price, draft), 0, ())
        for n in range(1, D + 1):
            gain = 1.0 + sum(spine_q[:n])
            rows = 1 + n
            if rows > self.max_rows:
                break
            score = gain / (price[rows] + draft)
            if score > best[0]:
                best = (score, n, ())
            # siblings at depths 1..n + 1 (a sibling's parent is the spine node above it), best first
            sibs = []
            for k in sorted(range(min(n + 1, D)), key=lambda k: -sib_q[k]):
                if rows + 1 > self.max_rows or sib_q[k] <= 0.0:
                    break
                g2, r2 = gain + sib_q[k], rows + 1
                s2 = g2 / (price[r2] + draft)
                if s2 <= score:
                    break
                gain, rows, score = g2, r2, s2
                sibs.append(k)
                if score > best[0]:
                    best = (score, n, tuple(sibs))
        _, n, sibs = best
        # DFS pre-order: a depth's sibling leaf before the spine child that continues the chain
        tokens, parents, kind = [anchor], [-1], [None]
        spine_row = 0
        for k in range(n + 1):
            if k in sibs:
                tokens.append(tok[k][1])
                parents.append(spine_row)
                kind.append(("sib", k))
            if k < n:
                tokens.append(tok[k][0])
                parents.append(spine_row)
                kind.append(("spine", k))
                spine_row = len(tokens) - 1
        self.last = (kind, prob, len(tokens), self.cls)
        return tokens, parents

    def _merged(self, anchor: int, tok: list, prob: list, look: list, conf: float) -> tuple[list[int], list[int]]:
        """ENG-229: the spine and the lookup's chain in one block. The lookup shares the spine's nodes
        while its tokens are the spine's (j of them; a shared node's q is 1 - both wrong), then leaves
        it under the spine node at depth j (or runs on past the spine's end): its first own node is
        taken when the spine's is not and the lookup is right, each next one when the lookup is right
        again. Nodes are admitted in order of q -- a parent's q is never below its child's, so a node
        comes after its parent -- for as long as (1 + sum q) / (V(rows) + draft) rises."""
        q = self.cal.q
        pl = q("look", conf)
        D = len(tok)
        j = 0
        while j < min(D, len(look)) and look[j] == tok[j][0]:
            j += 1
        # node: [token, parent node (-1: the anchor), q, (kind, k), depth]
        nodes, reach, up = [], 1.0, -1
        for k in range(D):
            c1 = q("spine", prob[k][0])
            c = 1.0 - (1.0 - c1) * (1.0 - pl) if k < j else c1
            nodes.append([tok[k][0], up, reach * c, ("spine", k) if k >= j else ("both", k), k])
            if k == j and j < len(look):
                lq = reach * (1.0 - c1) * pl           # the lookup's own token where the spine's is wrong
            reach *= c
            up = len(nodes) - 1
        if j == D and j < len(look):
            lq = reach * pl                             # the lookup runs on past the spine's end
        if j < len(look):
            lup = j - 1                                 # the spine node above (-1: the anchor)
            for i, t in enumerate(look[j:]):
                nodes.append([t, lup, lq, ("look", j + i), j + i])
                lup, lq = len(nodes) - 1, lq * pl
        order = sorted(range(len(nodes)), key=lambda i: (-nodes[i][2], nodes[i][4]))
        draft = self.draft_ms or 0.0
        price = self.prices.prices(self.max_rows, self.kind, self.cls)
        best, gain, taken, kept = (self._base(price, draft), 0), 1.0, set(), []
        for i in order:
            if len(kept) + 2 > self.max_rows:
                break
            if nodes[i][1] >= 0 and nodes[i][1] not in taken:
                continue
            taken.add(i)
            kept.append(i)
            gain += nodes[i][2]
            score = gain / (price[len(kept) + 1] + draft)
            if score > best[0]:
                best = (score, len(kept))
        keep = set(kept[:best[1]])
        tokens, parents, kind = [anchor], [-1], [None]
        row: dict = {}

        def emit(par: int, prow: int) -> None:
            # the lookup's branch before the spine's continuation, as `chain_siblings` puts a sibling
            for i in sorted((i for i in keep if nodes[i][1] == par), key=lambda i: nodes[i][3][0] != "look"):
                tokens.append(nodes[i][0])
                parents.append(prow)
                kind.append(nodes[i][3])
                row[i] = len(tokens) - 1
                emit(i, row[i])
        emit(-1, 0)
        self.last = ("merged", kind, prob, len(tokens), conf, parents, self.cls)
        return tokens, parents

    def observe(self, path: list[int], round_ms: float, draft_ms: float) -> None:
        """The last block's accepted path (its rows), what its verify and commit cost and what the draft
        cost: the calibration learns from every node the walk reached, the prices from the round."""
        if self.frozen:
            return
        if self.last[0] == "merged":
            self._observe_merged(path, round_ms, draft_ms)
            return
        kind, prob, rows, cls = self.last
        on = {kind[r] for r in path[1:]}
        for r in range(1, rows):
            t, k = kind[r]
            if not all(("spine", i) in on for i in range(k)):
                continue
            if t == "spine":
                self.cal.learn("spine", prob[k][0], ("spine", k) in on)
            elif ("spine", k) not in on:
                self.cal.learn("sib", prob[k][1], ("sib", k) in on)
        self._learn(rows, round_ms, draft_ms, cls)

    def _learn(self, rows: int, round_ms: float, draft_ms: float, cls: int) -> None:
        self.prices.observe(rows, round_ms, self.kind, cls)
        d = self.draft.get(cls)
        if d is None or draft_ms <= self.OUTLIER * d:
            self.draft[cls] = draft_ms if d is None else 0.9 * d + 0.1 * draft_ms

    def _observe_merged(self, path: list[int], round_ms: float, draft_ms: float) -> None:
        """A merged block: a node the walk reached (its parent on the path) teaches its source -- a
        spine node the spine's bin, a lookup node the lookup's, a shared node both."""
        _, kind, prob, rows, conf, parents, cls = self.last
        on = set(path)
        for r in range(1, rows):
            if parents[r] not in on:
                continue
            t, k = kind[r]
            hit = r in on
            if t in ("spine", "both"):
                self.cal.learn("spine", prob[k][0], hit)
            if t in ("look", "both"):
                self.cal.learn("look", conf, hit)
        self._learn(rows, round_ms, draft_ms, cls)
