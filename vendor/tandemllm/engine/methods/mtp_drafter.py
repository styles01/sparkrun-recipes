"""A model's MTP head as the server loop's tree drafter, written once (ENG-245).

The loop syncs the rows it committed (`sync(tokens, streams, pos)`); a row's MTP input pairs its
streams with the token after it, known when the next row arrives or, for the last row, when the next
proposal names its anchor. The backlog goes into the head's cache in one call (`head.prime`) except
the last `depth` + 1 rows, which ride the round's own feed (`head.launch`); the round's draft is cut by
StairCut (engine/methods/staircut.py), which learns what each round cost: the wall time from this
round's start to the next's (the draft, the verify, the commit, the stream and the loop's own work,
less the draft), and for a plain step the same span -- what the round really took, so the cut weighs
a block against the plain step on one clock. A round that captured a graph teaches nothing.

What it asks of the head (Flash-Next's: engine/models/flashnext/mtp.py): `depth`, `reset()`,
`prime(rows, nexts)`, `capture_all()`, `launch(rows, nexts)` (asynchronous), `read()` -> per depth
the top-2 tokens and their probabilities, `views()` and `resume(n)`. Of the engine: `captured` (the
last block captured a verify or commit graph), `commit_ms` and `last_streams()` (the last committed
row's streams, a section of its cache state).

A lookup source (`ngram`, engine/drafters/ngram.py: the request's own text, the corpus and the server's
suffix store, which the server hands to it through `add_store`) adds its best continuation to the
round's cut (ENG-229): the head and the lookup in one tree, one verify. It is asked while the draft
runs, as deep as its match allows.

Context classes: the engine's `price_class(n)` (absent: one class) names the class of a round whose
context is n tokens; the cut prices it there, the plain steps the loop takes are priced there
(`on_plain`), and where the cut keeps choosing the anchor alone the round takes the plain step with no
draft at all (`StairCut.skip`). A turn's first round weighs the backlog the head has not read against
what drafting would save over an answer of the usual length (`StairCut.turn`); a turn that does not
draft drops its rows (the head reads nothing this turn), and the next one that drafts resumes the
head at its own position past the rows it skipped -- rows it never read change a draft, never the
text.

The caches (ENG-157) keep the head's cache with the engine's rows: `kv_views` (the resident prefix,
the disk tier, the spill), `state_snapshot` / `state_restore` (the state store) and `state_resume(n)`
(a resume in place). After a resume at n the head holds rows below n - 1 and the row at n - 1 waits for
its next token, as it did when the rows were written. A row the head had not fed when a cache took
it is stale; it changes a draft, never the text (the verify decides).
"""

from __future__ import annotations

import contextlib
import time

import torch

from engine.drafters import run_steps
from engine.methods.staircut import StairCut
from engine.tree import DraftTree


class NoHead:
    """The head of a model that has no draft layer (LFM2.5-24B-A2B until Liquid publishes its drafter,
    ENG-E10): it drafts nothing, so the cut admits the lookup's continuation alone (`StairCut._merged`
    with an empty spine) and a round with no match takes the plain step. A real head replaces it with
    no change to the drafter."""

    depth = 0
    captures = 0

    def __init__(self):
        self.st = type("NoHeadState", (), {"length": 0})()

    def reset(self) -> None:
        self.st.length = 0

    def prime(self, rows, nexts) -> None:
        self.st.length += len(nexts)

    def launch(self, rows, nexts) -> None:
        self.st.length += len(nexts)

    def read(self) -> tuple[list, list]:
        return [], []

    def capture_all(self) -> None:
        pass

    def views(self) -> list:
        return []

    def resume(self, n: int) -> None:
        self.st.length = int(n)


class MTPTreeDrafter:
    name = "mtp"
    last_q = None
    wants_rows = False

    def __init__(self, eng, head, cut: StairCut, ngram=None, look_depth: int = 16):
        self.s, self.head, self.cut, self.depth = eng, head, cut, head.depth
        self.ngram, self.look_depth = ngram, look_depth
        self.pending = None                 # (streams, next tokens) of the round's own feed
        self.held = None                    # (tokens, streams) of the last rows synced
        self.queue: list = []               # (streams, next tokens) not fed yet
        self.round = None                   # the draft ms of the block awaiting its verify
        self.verify_ms = None
        self.stats = {"rounds": 0, "draft_ms": 0.0, "fed_rows": 0, "looked": 0, "skipped": 0, "plain": 0}
        self.req = self._req()              # this request's own counts (the verbose report)
        self.cls = 0                        # the context class of the last round
        self._t_prev = None                 # the last round's start (perf_counter)
        self._cls_prev = 0
        self._obs = None                    # what the last round was, to price at the next start
        self._plain = None                  # a plain step was taken: None, or whether it captured
        self._captured = False              # this round's draft captured a graph
        self._drafted = False               # this round launched a draft
        self._turn = None                   # this request drafts (True), takes plain steps (False), or undecided
        self.answer = 256.0                 # the usual answer length, tokens (EMA 0.2 of the requests')
        self._prime_ms = 0.0                # the last feed's backlog prime, ms
        # opt-in (a block drafter whose draft costs a large share of a step): after a round the cut cut
        # to the anchor alone, the next rounds take the plain step without a draft, 1, 2, 4 .. up to
        # `anchor_backoff` of them as such rounds repeat; a round that keeps rows starts over. 0: off
        self.anchor_backoff = 0
        self._anchor_streak = 0
        self._backoff_left = 0

    def report(self) -> str:
        """The verbose server's line at each request: rounds, draft ms, StairCut's prices and offset."""
        c = self.cut
        V, delta = c.prices.learned(c.kind, self.cls)
        q = self.req
        return (f"mtp request: turn={q['turn']} rounds={q['rounds']} plain={q['plain']} wall_ms={q['wall_ms']:.0f} "
                f"draft_ms={q['draft_ms']:.0f} verify_ms={q['verify_ms']:.0f} ({q['verify_rows']} rows) "
                f"prime_ms={q['prime_ms']:.0f} ({q['prime_rows']} rows) "
                f"spans={ {k: [n, round(ms), t] for k, (n, ms, t) in q['spans'].items()} } "
                f"spec_ms={c.spec_ms(self.cls) if c.spec_ms(self.cls) is None else round(c.spec_ms(self.cls), 2)} | "
                f"mtp rounds={self.stats['rounds']} fed={self.stats['fed_rows']} looked={self.stats['looked']} "
                f"skipped={self.stats['skipped']} plain={self.stats['plain']} class={self.cls} "
                f"plain_ms={round(c.plain[self.cls], 2) if self.cls in c.plain else None} "
                f"draft_ms={c.draft_ms if c.draft_ms is None else round(c.draft_ms, 2)} delta={round(delta, 2)} "
                f"V={ {r: round(v, 1) for r, v in sorted(V.items())} } "
                f"spine_cal={[round(c.cal.q('spine', (b + 0.5) / 10), 2) for b in range(10)]}")

    @staticmethod
    def _req() -> dict:
        return {"rounds": 0, "plain": 0, "wall_ms": 0.0, "draft_ms": 0.0, "verify_ms": 0.0, "verify_rows": 0,
                "prime_ms": 0.0, "prime_rows": 0, "tokens": 0, "turn": None,
                "spans": {"block": [0, 0.0, 0], "anchor": [0, 0.0, 0], "plain": [0, 0.0, 0],
                          "unpriced": [0, 0.0, 0]}}

    WARM_TOKENS = 96        # the server's warm-up decodes this many through the rounds (`warming`)

    def capture_all(self) -> None:
        self.head.capture_all()

    @contextlib.contextmanager
    def warming(self):
        """The server's warm-up runs inside: the cut learns nothing (`StairCut.frozen`) and keeps no
        turn or probe state, the usual answer length stays. The first request after a start drafted its
        1k rounds at 22.4 ms a token against 17.8 later (2026-10-12: the class's verify and commit
        graphs captured in it, and the kernels' first calls); priced on that request, 1k took the plain
        step for its next turns. The warm-up's own rounds pay for those instead."""
        cut, answer, frozen = self.cut, self.answer, self.cut.frozen
        cut.frozen = True
        try:
            yield
        finally:
            cut.frozen = frozen
            for d in (cut.skipped, cut.probe, cut.plain_turns, cut.age, cut.returned):
                d.clear()
            self.reset()
            self.answer = answer

    # --- the loop's drafter interface -------------------------------------------------------------
    def reset(self) -> None:
        self.head.reset()
        if self.ngram is not None:
            self.ngram.reset()
        self.pending = None
        self.held, self.queue, self.round, self.verify_ms = None, [], None, None
        self._t_prev, self._obs, self._plain = None, None, None
        if self.req["tokens"]:
            self.answer = 0.8 * self.answer + 0.2 * max(32, self.req["tokens"])
        self.req = self._req()
        self._turn = None
        self._anchor_streak = self._backoff_left = 0

    def prime(self, tokens: list[int]) -> None:
        # the prompt's rows come through `sync`; the lookup indexes its tokens
        if self.ngram is not None:
            self.ngram.prime(tokens)

    def set_sampling(self, sampler) -> None:
        pass                                # the draft is the greedy tree; a sampled walk draws on it

    def sync(self, tokens: list[int], streams: torch.Tensor, pos: int, rows=None) -> None:
        if self._turn is False:
            return                          # a plain turn: the head reads nothing (`_start_turn`)
        toks = [int(t) for t in tokens]
        h = streams.reshape(len(toks), -1)
        if h.shape[0] <= 64:
            h = h.clone()                   # a static buffer the next forward writes again
        if self.held is not None:
            ht, hs = self.held
            self.queue.append((hs, ht[1:] + [toks[0]]))
        self.held = (toks, h)

    def observe(self, tokens: list[int]) -> None:
        if self.ngram is not None:
            self.ngram.observe(tokens)
        self.req["tokens"] += len(tokens)
        # priced at the next round's start, by the wall time between the two; a round that captured a
        # graph took the capture's time and teaches nothing (its price for that row count would start
        # at ~170 ms and the cut would never ask again)
        if self._plain is not None:
            # a plain step after a draft that captured a graph carries the capture's time too; one after
            # a draft the cut cut to the anchor is a drafted round of one token (2026-10-12: priced as
            # plain steps, they put the plain step at 25.7 ms against 21.6 in a plain turn)
            self._obs = (None if self._plain or self._captured else
                         ("anchor", None, None, 1, self._prime_ms) if self._drafted else ("plain",))
        elif self.round is not None and self.verify_ms is not None and not self.s.captured:
            self._obs = ("block", list(range(len(tokens))), self.round, len(tokens), self._prime_ms)
        else:
            self._obs = None
        self._plain, self.round, self.verify_ms = None, None, None

    def on_verify(self, rows: int, ms: float) -> None:
        self.verify_ms = ms
        self.req["verify_ms"] += ms
        self.req["verify_rows"] += rows

    def on_plain(self, ms: float) -> None:
        """The loop took a plain step: after a skipped round its cost prices the plain step of this
        round's class; after a draft the cut chose the anchor alone, it is a drafted round of one token."""
        self.stats["plain"] += 1
        self.req["plain"] += 1
        self._plain = bool(getattr(self.s, "plain_captured", False))   # a capture's time is not the step's

    def propose_tree(self, context: list[int], k: int) -> DraftTree | None:
        return run_steps(self.propose_tree_steps(context, k))

    def propose_tree_steps(self, context: list[int], k: int):
        """The block for the round anchored at context[-1]: the backlog fed, the draft launched (the
        generator stops there so the loop streams the last block meanwhile), the cut."""
        t0 = time.perf_counter()
        if self._t_prev is not None:
            ms = (t0 - self._t_prev) * 1e3
            self.req["wall_ms"] += ms
            # the request's spans by what priced them: drafted rounds, plain steps, unpriced (a capture)
            span = self.req["spans"][self._obs[0] if self._obs is not None else "unpriced"]
            span[0] += 1
            span[1] += ms
            span[2] += self._obs[3] if self._obs is not None and self._obs[0] != "plain" else 0
        if self._obs is not None and self._t_prev is not None:
            if self._obs[0] == "plain":
                self.cut.observe_plain(ms, self._cls_prev)
            else:
                # the round's backlog prime is the turn's (`StairCut.turn` weighs it), not the round's:
                # in the round's price it made a turn's first round at 140k cost 93 ms (2026-10-12)
                ms -= self._obs[4]
                if self._obs[0] == "anchor":
                    self.cut.observe_spec(ms, self._cls_prev, 1)
                else:
                    self.cut.observe(self._obs[1], ms - self._obs[2], self._obs[2])
                    self.cut.observe_spec(ms, self._cls_prev, self._obs[3])
        self._obs, self._t_prev, self._captured, self._drafted = None, t0, False, False
        self._prime_ms = 0.0
        self.cls = self.s.price_class(len(context)) if hasattr(self.s, "price_class") else 0
        self._cls_prev = self.cls
        if self._turn is None:
            self._start_turn(len(context))
        if self._turn is False:
            self.stats["skipped"] += 1
            return None
        # with no draft head (NoHead) each round's own lookup match decides, through the cut's
        # calibrated q: the class-wide skip, learned for a head that drafts every round, priced the
        # lookup by the rounds that matched little (LFM2.5, 2026-10-14: three chat asks, then edit and
        # copy asks the lookup matched throughout took the plain step)
        backoff = False
        if self._backoff_left > 0:
            self._backoff_left -= 1
            backoff = True
        if self.head.depth and (backoff or self.cut.skip(self.cls)):
            # no draft this round: the rows wait in the queue for the next one that drafts
            if self.held is not None:
                ht, hs = self.held
                self.queue.append((hs, ht[1:] + [int(context[-1])]))
                self.held = None
            self.round = None
            self.stats["skipped"] += 1
            return None
        if not self._feed(int(context[-1])) or k < 1:
            return None
        look = None
        if self.head.depth == 0:
            # no draft head (NoHead): a round the lookup has nothing for is the plain step, priced as
            # one, so the drafted rounds' price is learned from rounds that drafted
            look = self._look(context)
            if look is None:
                self.pending = None
                return None
        rows, nexts = self.pending
        self.pending = None
        c0 = getattr(self.head, "captures", 0)
        self.head.launch(rows, nexts)
        captured = self._captured = getattr(self.head, "captures", 0) != c0
        self._drafted = True
        host = time.perf_counter() - t0
        yield                               # the loop streams the last block while the draft runs
        t1 = time.perf_counter()
        if self.head.depth:
            look = self._look(context)
        tok, prob = self.head.read()
        toks, parents = self.cut.cut(int(context[-1]), tok, prob, look, self.cls)
        tree = DraftTree(tokens=toks, parents=parents).truncate(int(k) + 1)
        if self.anchor_backoff:
            if len(toks) <= 1:
                self._anchor_streak += 1
                self._backoff_left = min(1 << (self._anchor_streak - 1), self.anchor_backoff)
            else:
                self._anchor_streak = 0
        # the draft's own cost: its host work on both sides of the stop, not the streaming between
        # (StairCut priced the gap as drafting and stopped drafting, 2026-10-11)
        # less the backlog's prime (rows a skipped round left: owed whether this round drafts or not);
        # a round whose draft captured a graph teaches nothing (its time is the capture's)
        ms = (host + time.perf_counter() - t1) * 1e3 - self._prime_ms
        self.round = None if captured else ms
        self.stats["rounds"] += 1
        self.stats["draft_ms"] += ms
        self.req["rounds"] += 1
        self.req["draft_ms"] += ms
        return tree

    def _start_turn(self, n: int) -> None:
        """The turn's first round (context n tokens): the head is put at the position of the first row
        it has not read (past rows a plain turn skipped), then StairCut decides whether the turn drafts;
        a plain turn drops the backlog."""
        unfed = sum(len(q[1]) for q in self.queue) + (len(self.held[0]) if self.held is not None else 0)
        at = max(0, n - 1 - unfed)
        if self.head.st.length < at:
            self.head.resume(at)
        # the graphs this class lacks: captured by the engine while the server is idle, so a turn that
        # would pay for them waits for that (the first drafted request in a class paid 3.4-5.1 s)
        extra = self.s.capture_cost(n) if hasattr(self.s, "capture_cost") else 0.0
        # no draft head: no backlog to read into it, and each round's match decides (above)
        self._turn = self.req["turn"] = (not self.head.depth
                                         or bool(self.cut.turn(self.cls, unfed, self.answer, extra)))
        if not self._turn:
            self.held, self.queue, self.pending = None, [], None

    @property
    def plain_turn(self) -> bool:
        """This turn takes the plain step throughout (`_start_turn`): a parallel round's barrier does
        not wait for its blocks (server/app.py `_par_block`)."""
        return self._turn is False

    def wants_graphs(self, n: int) -> bool:
        """Whether drafting can pay at context length `n` at all (no price says a drafted round costs
        more a token than the plain step there): the server captures a class's graphs while idle only
        for such a class (260,800 tokens takes the plain step, and its captures would cost memory)."""
        c = self.s.price_class(n) if hasattr(self.s, "price_class") else 0
        plain, spec = self.cut.plain_ms(c), self.cut.spec_ms(c)
        return plain is None or spec is None or spec < plain

    def _look(self, context: list[int]) -> tuple | None:
        """The lookup's best continuation and its share of the vote (smoothed as the lookup's own
        tree is), or None when nothing matches."""
        if self.ngram is None:
            return None
        order, cands = self.ngram.candidates(context, self.look_depth)
        if not cands:
            return None
        self.stats["looked"] += 1
        total = sum(w for _, w in cands)
        share = cands[0][1] / (total + self.ngram.alpha)
        if not self.head.depth and order:
            # no draft head: the match's length goes into the confidence (a copy matches at the longest
            # order, a passing phrase at the shortest), so the cut's calibration learns them apart and a
            # run of short matches that missed does not stop a copy from drafting (LFM2.5 BF16,
            # 2026-10-14: four missed prose rounds held every later round below the cut, a copy ask too)
            share = 1.0 - (1.0 - share) * min(1.0, self.ngram.min_order / order)
        return cands[0][0], share

    # --- the caches (ENG-157) -------------------------------------------------------------------
    def kv_views(self) -> list:
        return self.head.views()

    def snapshot_bytes_per_token(self) -> int:
        return sum(t.numel() // t.shape[ax] * t.element_size() for t, ax in self.head.views())

    def state_resume(self, n: int) -> None:
        n = int(n)
        self.head.resume(max(0, n - 1))
        self.pending, self.queue, self.round, self.verify_ms = None, [], None, None
        self._t_prev, self._obs, self._plain = None, None, None
        self.held = ([0], self.s.last_streams().clone()) if n > 0 else None

    def state_snapshot(self) -> dict:
        """The head's cache after every synced row that knows its next token went in."""
        self._flush()
        L = int(self.head.st.length)
        return {"len": L, "views": tuple(t.narrow(ax, 0, L).clone() for t, ax in self.head.views())}

    def state_restore(self, snap: dict) -> None:
        L = int(snap["len"])
        for (t, ax), x in zip(self.head.views(), snap["views"]):
            t.narrow(ax, 0, L).copy_(x)
        self.head.resume(L)
        self.pending, self.queue, self.round, self.verify_ms = None, [], None, None
        self._t_prev, self._obs, self._plain = None, None, None
        self.held = ([0], self.s.last_streams().clone())

    # --- internals -------------------------------------------------------------------------------
    def _flush(self) -> None:
        """Feed the queue and the round's own feed now (every row there knows its next token)."""
        rows = [q[0] for q in self.queue] + ([self.pending[0]] if self.pending is not None else [])
        nexts = [t for q in self.queue for t in q[1]] + (list(self.pending[1]) if self.pending is not None else [])
        self.queue, self.pending = [], None
        if nexts:
            self.head.prime(torch.cat(rows) if len(rows) > 1 else rows[0], nexts)

    def _feed(self, anchor: int) -> bool:
        """Every synced row paired with its next token (the last with the anchor); all but the last
        depth + 1 into the head's cache now, those into the round's own feed. False: nothing synced."""
        if self.held is not None:
            ht, hs = self.held
            self.queue.append((hs, ht[1:] + [anchor]))
            self.held = None
        if not self.queue:
            return self.pending is not None
        rows = torch.cat([q[0] for q in self.queue]) if len(self.queue) > 1 else self.queue[0][0]
        nexts = [t for q in self.queue for t in q[1]]
        self.queue = []
        n, keep = len(nexts), min(len(nexts), self.depth + 1)
        if n > keep:
            t0 = time.perf_counter()
            self.head.prime(rows[:n - keep], nexts[:n - keep])
            if rows.is_cuda:
                torch.cuda.synchronize()    # the prime's own time, not the verify's after it
            self._prime_ms = (time.perf_counter() - t0) * 1e3
            self.req["prime_ms"] += self._prime_ms
            self.req["prime_rows"] += n - keep
            if n - keep >= 64:
                self.cut.observe_prime(self._prime_ms / (n - keep), self.cls)
        self.pending = (rows[n - keep:], nexts[n - keep:])
        self.stats["fed_rows"] += n
        return True
