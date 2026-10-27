"""The side lane: a short request served in the middle of another one's generation.

THE PROBLEM. The engine holds one sequence, and a request waits for the engine lock until the
request before it ends. An agent turn holds it for minutes (on 2026-10-04 16:43-16:48 one opencode
turn thought for 7 minutes). Everything short waited behind it: opencode's title request of a new
session, and the conductor's own questions to Qwen (the package and library choices of the
router, `server/conductor/classify.py` `Qwen`, non-streamed, `max_tokens` 24 under a JSON choice
grammar, `X-Tandem: raw` on the hidden raw alias), which gave up after their 8 s timeout and left
the router without an answer (the ledger's `abandoned` rows at 9.0 s, client `anon`).

THE LANE. A short request -- at most `PROMPT_MAX` prompt tokens and `NEW_MAX` output tokens, one
choice, no images, thinking off, and either a raw request or a client's side request (S1) -- that
finds the engine busy does not queue. It becomes a `Job` here, and the request that holds the
engine serves it at its next safe point: between two decode blocks (`app.generate_stream`), or
between two prefill chunks. The holder lends the engine (`engine/cache.py` `Borrow`: the rows the
guest can reach, the recurrent state, the convolution tails, a pending commit and the references
the generation reads again are copied aside), the guest is prefilled from position 0 and decoded
a token at a time with no drafter, and the engine is given back byte for byte before the
generation's next block. The generation itself never sees it: the same rows, the same state, the
same drafter (which the guest never touches), so the same tokens.

A guest whose job no holder takes (the holder ended, or holds the lock outside a generation)
takes the engine lock itself, at the side rank, and runs the same job under it: the rows of the
resident conversation come back the same way.

The job runs on the holder's thread, so nothing in it may raise into the generation: every
failure is the guest's (`Job.error`), and the generation goes on.
"""

from __future__ import annotations

import queue
import threading
import time
from collections import deque

#: the longest prompt and output the lane serves (opencode's title request is 587 tokens and 64
#: at most; the router's questions 150-420 and 24)
PROMPT_MAX = 2048
NEW_MAX = 64
#: rows past the guest's own a forward may write (none on the plain path; a margin)
ROW_SLACK = 64
#: how often a guest looks whether its job was taken, and whether the engine is free
POLL_S = 0.05
#: the end of a job's token stream
DONE = object()


class Job:
    """One short request for the lane: its prompt ids and decoding rules, and the queue its tokens
    arrive on. `state`: pending -> running -> done, or pending -> cancelled (the client left
    before a holder took it)."""

    def __init__(self, ids, max_new: int, eos, *, sampler=None, pen=None, lpr=None, pstop=None,
                 cid: str = "", kind: str = "raw"):
        self.ids = [int(t) for t in ids]
        self.max_new = int(max_new)
        self.eos = set(eos)
        self.sampler, self.pen, self.lpr, self.pstop = sampler, pen, lpr, pstop
        self.cid, self.kind = cid, kind
        self.out: queue.Queue = queue.Queue()
        self.done = threading.Event()          # set once the engine is given back
        self.state = "pending"
        self._lock = threading.Lock()
        self.t_submit = time.perf_counter()
        self.t_start: float | None = None
        self.t_end: float | None = None
        self.where: str | None = None          # decode, prefill or self
        self.prefill: dict | None = None       # {reused, forwarded, ms, kind} of its own prefill
        self.blocks = None                     # its own BlockStats
        self.borrowed_bytes = 0
        self.error: BaseException | None = None
        self.n_out = 0

    def claim(self) -> bool:
        with self._lock:
            if self.state != "pending":
                return False
            self.state = "running"
            self.t_start = time.perf_counter()
            return True

    def cancel(self) -> bool:
        with self._lock:
            if self.state != "pending":
                return False
            self.state = "cancelled"
            return True

    @property
    def lent_ms(self) -> float | None:
        """How long the job held the engine (the time the interrupted generation waited)."""
        if self.t_start is None or self.t_end is None:
            return None
        return (self.t_end - self.t_start) * 1e3

    @property
    def wait_ms(self) -> float | None:
        if self.t_start is None:
            return None
        return (self.t_start - self.t_submit) * 1e3


class Lane:
    """The jobs waiting for a safe point, and what served them. `run(job)` executes one job on the
    engine (the server's `_lane_run`, the fake engine's writer in tests); it is called with the
    engine held, by the holder at a safe point or by the guest under the lock."""

    def __init__(self, run, *, prompt_max: int = PROMPT_MAX, new_max: int = NEW_MAX,
                 where: str = "on", room=None):
        self.run = run
        # room(job, where): whether the engine has the memory for the job here; a job without it
        # waits for a later safe point (a prefill's peak passes) or for the lock
        self.room = room or (lambda job, where: True)
        self.prompt_max, self.new_max = int(prompt_max), int(new_max)
        # on: decode blocks and prefill chunks; decode: decode blocks only; off: no lane
        self.where = where
        # what a guest whose client left raises (the server's ClientGone)
        self.gone_exc = ConnectionResetError
        self.jobs: deque = deque()
        self._lock = threading.Lock()
        self.recent: deque = deque(maxlen=50)
        self.stats = {"jobs": 0, "served_decode": 0, "served_prefill": 0, "served_self": 0,
                      "cancelled": 0, "errors": 0, "lent_ms": 0.0}

    @property
    def on(self) -> bool:
        return self.where in ("on", "decode")

    def eligible(self, n_prompt: int, max_new: int) -> bool:
        return self.on and 0 < int(n_prompt) <= self.prompt_max and 0 < int(max_new) <= self.new_max

    def submit(self, job: Job) -> None:
        with self._lock:
            self.jobs.append(job)
            self.stats["jobs"] += 1

    def serve(self, where: str) -> int:
        """At a safe point of the holder's generation: run every job waiting now. Returns how many
        ran. Never raises."""
        if not self.jobs or (where == "prefill" and self.where != "on"):
            return 0
        n = 0
        while True:
            with self._lock:
                if not self.jobs:
                    break
                job = self.jobs[0]
                if job.state == "pending" and not self.room(job, where):
                    self.stats["no_room"] = self.stats.get("no_room", 0) + 1
                    break                      # the next safe point, or the lock
                self.jobs.popleft()
            if not job.claim():
                continue                       # cancelled, or taken by its own thread
            self._execute(job, where)
            n += 1
        return n

    def _execute(self, job: Job, where: str) -> None:
        job.where = where
        try:
            self.run(job)
        except BaseException as exc:                               # noqa: BLE001
            job.error = exc
            self.stats["errors"] += 1
        finally:
            job.t_end = time.perf_counter()
            with job._lock:
                job.state = "done"
            job.done.set()
            self.stats[f"served_{where}"] = self.stats.get(f"served_{where}", 0) + 1
            self.stats["lent_ms"] += job.lent_ms or 0.0
            self.recent.append({"ts": round(time.time(), 3), "cid": job.cid, "kind": job.kind,
                                "where": where, "prompt": len(job.ids), "out": job.n_out,
                                "wait_ms": round(job.wait_ms or 0.0, 1),
                                "lent_ms": round(job.lent_ms or 0.0, 1),
                                "error": type(job.error).__name__ if job.error else None})
            job.out.put(DONE)

    def withdraw(self, job: Job) -> bool:
        """Take a pending job off the line (its client left); False when it already runs."""
        if not job.cancel():
            return False
        with self._lock:
            try:
                self.jobs.remove(job)
            except ValueError:
                pass
        self.stats["cancelled"] += 1
        return True

    def tokens(self, job: Job, *, take, release, gone=lambda: False):
        """The guest's side: submit `job` and yield its tokens as they are decided. `take()` tries
        the engine lock without waiting (True when held), `release()` gives it back; a job no holder
        has taken while the lock is free is run here, under it. `gone()`: the client left -- a
        pending job is withdrawn and `ConnectionResetError` raised. A job's own failure is raised
        after its last token."""
        self.submit(job)
        while True:
            try:
                t = job.out.get(timeout=POLL_S)
            except queue.Empty:
                if job.state != "pending":
                    continue
                if gone():
                    if self.withdraw(job):
                        raise self.gone_exc("client left while its side request waited")
                    continue
                if take():
                    try:
                        if job.claim():
                            with self._lock:
                                try:
                                    self.jobs.remove(job)
                                except ValueError:
                                    pass
                            self._execute(job, "self")
                    finally:
                        release()
                continue
            if t is DONE:
                break
            yield t
        if job.error is not None:
            raise job.error

    def report(self) -> dict:
        with self._lock:
            waiting = len(self.jobs)
        return {"where": self.where, "prompt_max": self.prompt_max, "new_max": self.new_max,
                "waiting": waiting, **{k: (round(v, 1) if isinstance(v, float) else v)
                                       for k, v in self.stats.items()},
                "recent": list(self.recent)[-10:]}


def decode(eng, job: Job, *, device, chunk: int, h2d, block_stats=None) -> None:
    """A job on the engine (the caller holds it): prefill from 0 on the prefill's own grid, then
    one token a forward under the job's rules, its tokens onto `job.out` as they are decided.
    The engine is borrowed (`engine/cache.py` `Borrow`) and given back however this ends."""
    import torch

    from engine import cache
    b = cache.Borrow(eng, len(job.ids) + job.max_new + ROW_SLACK)
    job.borrowed_bytes = b.nbytes
    bs = job.blocks = block_stats() if block_stats is not None else None
    try:
        with torch.no_grad():
            ctx = list(job.ids)
            pen, sampler, lpr, pstop = job.pen, job.sampler, job.lpr, job.pstop
            if pen is not None:
                pen.seed(ctx)
            t0 = time.perf_counter()
            logits, _, fwd = cache.prefill(eng, None, ctx, device, chunk=chunk)
            job.prefill = {"reused": 0, "forwarded": int(fwd), "kind": None,
                           "ms": (time.perf_counter() - t0) * 1e3}
            pos = len(ctx)
            row = logits[0, -1]
            while True:
                if pen is not None:
                    pen.mask = False
                    pen.apply_single(row)
                tok = (sampler(row, index=len(ctx)) if sampler is not None and sampler.on
                       else int(row.argmax()))
                if lpr is not None:
                    lpr.rows(row[None], [tok])
                if pen is not None:
                    pen.commit([tok])
                ctx.append(tok)
                job.n_out += 1
                if bs is not None:
                    if job.n_out == 1:
                        bs.first()
                    else:
                        bs.block()
                stop = pstop is not None and pstop.observe([tok])
                job.out.put(tok)
                if tok in job.eos or job.n_out >= job.max_new or stop:
                    break
                lg = eng.forward(h2d([tok], torch.long, device), start=pos, last_only=True)
                pos += 1
                row = lg[0, -1]
    finally:
        b.give_back()
