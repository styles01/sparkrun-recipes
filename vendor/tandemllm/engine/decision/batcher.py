"""The cross-caller batcher, and the idle window it runs in.

Arbiter's batcher (server/engine.py there): a worker thread per model takes the first job, keeps
draining the queue for `wait_ms` or until `max_batch` rows are in hand, and answers what it
collected in one go. How it answers is `collate`:

  call  (the default) every caller's rows in a forward of their own, one after the other in the
        same idle window: exactly the forward the SDK runs for that call, so an answer never
        depends on who else asked in the same 2 ms. On the GB10 under autocast this is 0.00
        against the SDK on all 22 questions (tools/decider_equivalence.py, 2026-10-01).
  rows  arbiter's way: the rows of every caller collated into one padded forward. Rows are
        independent in the model, but not in the arithmetic: the padded length and the batch
        change the GEMM shapes, and under bf16 autocast that moved a probability by up to 3.4e-3
        (laya-multilingual; no argmax moved) on the same 22 questions. Arbiter's own tool never
        saw it (its batched answers were overwritten by the per-request ones before the
        comparison). Faster under many concurrent callers; not exact.

A single job larger than the cap is chunked either way.

Two things are new here. Every job carries a deadline, and a job whose deadline passes before
its forward starts is dropped with `DeadlineMissed`, never run late. And stage A asks only in the
IDLE window: a batch runs while it holds `gate`, the engine's own lock -- the one every
completion holds from its prefill to its last token -- so a decision forward never runs beside a
Qwen prefill or decode, and a completion that arrives meanwhile waits at most one decision
forward for the lock. In-decode questions (a low-priority stream and a governor beside a running
decode) are stage C of the design and are refused before they get here.

A caller that holds the engine lock itself and waits for a decision would wait for its own lock;
the deadline ends that wait (`deadline_missed`), so it cannot hang.
"""

from __future__ import annotations

import collections
import queue
import statistics
import threading
import time
from concurrent.futures import Future

from engine.decision.base import DeadlineMissed


class _Job:
    __slots__ = ("rows", "deadline", "future", "t_submit")

    def __init__(self, rows, deadline: float | None, t_submit: float):
        self.rows, self.deadline, self.t_submit = rows, deadline, t_submit
        self.future: Future = Future()


class Batcher:
    def __init__(self, model, gate, *, max_batch: int = 64, wait_ms: float = 2.0,
                 collate: str = "call", on_error=None, clock=time.monotonic):
        if collate not in ("call", "rows"):
            raise ValueError(f"collate is call or rows, not {collate!r}")
        self.model, self.gate, self.collate = model, gate, collate
        self.max_batch = max(1, int(max_batch))
        self.wait_s = max(0.0, float(wait_ms)) / 1000.0
        self.on_error, self.clock = on_error, clock
        self.q: queue.Queue = queue.Queue()
        self.depth = 0
        self._depth_lock = threading.Lock()
        self.forwards = 0
        self.rows_total = 0
        self.deadline_missed = 0
        self.errors = 0
        self.last_path: str | None = None
        self.forward_ms: collections.deque = collections.deque(maxlen=200)
        self.batch_rows: collections.deque = collections.deque(maxlen=200)
        self.thread = threading.Thread(target=self._loop, name=f"decider-{model.name}",
                                       daemon=True)
        self.thread.start()

    def submit(self, rows: list, deadline: float | None) -> Future:
        """Queue one caller's rows; the future gets `{answers, input_tokens, batch_rows, path,
        queue_ms, forward_ms}` or the error. `deadline`: monotonic seconds, None for none."""
        job = _Job(rows, deadline, self.clock())
        with self._depth_lock:
            self.depth += len(rows)
        self.q.put(job)
        return job.future

    def stop(self) -> None:
        self.q.put(None)
        self.thread.join(timeout=10)

    def stats(self) -> dict:
        ms = sorted(self.forward_ms)
        return {"forwards": self.forwards, "rows": self.rows_total, "errors": self.errors,
                "deadline_missed": self.deadline_missed, "queue_rows": self.depth,
                "last_path": self.last_path,
                "forward_ms_p50": round(statistics.median(ms), 2) if ms else None,
                "forward_ms_p95": round(ms[min(len(ms) - 1, int(len(ms) * 0.95))], 2)
                if ms else None,
                "rows_per_forward_p50": statistics.median(self.batch_rows)
                if self.batch_rows else None}

    # ------------------------------------------------------------------ the worker
    def _loop(self) -> None:
        while True:
            job = self.q.get()
            if job is None:
                return
            jobs, rows = [job], len(job.rows)
            until = self.clock() + self.wait_s
            stop = False
            while rows < self.max_batch:
                left = until - self.clock()
                if left <= 0:
                    break
                try:
                    nxt = self.q.get(timeout=left)
                except queue.Empty:
                    break
                if nxt is None:
                    stop = True
                    break
                jobs.append(nxt)
                rows += len(nxt.rows)
            try:
                self._run(jobs)
            except Exception as exc:                      # noqa: BLE001 - one bad batch must not kill the worker
                for j in jobs:
                    if not j.future.done():
                        j.future.set_exception(exc)
            finally:
                with self._depth_lock:
                    self.depth -= sum(len(j.rows) for j in jobs)
            if stop:
                return

    def _expire(self, jobs: list) -> list:
        now, live = self.clock(), []
        for j in jobs:
            if j.deadline is not None and now >= j.deadline:
                self.deadline_missed += 1
                j.future.set_exception(DeadlineMissed(
                    "the engine served a request until the question's deadline passed",
                    waited_ms=round((now - j.t_submit) * 1e3, 1)))
            else:
                live.append(j)
        return live

    def _run(self, jobs: list) -> None:
        live = self._expire(jobs)
        got = False
        while live and not got:
            ends = [j.deadline for j in live if j.deadline is not None]
            timeout = (min(ends) - self.clock()) if len(ends) == len(live) else None
            if timeout is not None and timeout <= 0:
                live = self._expire(live)
                continue
            got = self.gate.acquire(timeout=timeout if timeout is not None else -1)
            if not got:
                live = self._expire(live)
        if not got:
            return
        t0 = self.clock()
        try:
            live = self._expire(live)        # a deadline that passed while the lock was taken
            if not live:
                return
            flat, owner = [], []
            for ji, j in enumerate(live):
                for ri, row in enumerate(j.rows):
                    flat.append(row)
                    owner.append((ji, ri))
            results = [[None] * len(j.rows) for j in live]
            tokens = [0] * len(live)
            path = "eager"
            # one forward per caller (`call`), or the callers' rows in one (`rows`)
            groups = ([[i for i, (ji, _) in enumerate(owner) if ji == g] for g in range(len(live))]
                      if self.collate == "call" else [list(range(len(flat)))])
            sizes = []
            try:
                for group in groups:
                    for start in range(0, len(group), self.max_batch):
                        idx = group[start:start + self.max_batch]
                        answers, row_tokens, path = self.model.predict_rows([flat[i] for i in idx])
                        sizes.append(len(idx))
                        for k, i in enumerate(idx):
                            ji, ri = owner[i]
                            results[ji][ri] = answers[k]
                            tokens[ji] += row_tokens[k]
            except Exception as exc:                      # noqa: BLE001
                self.errors += 1
                for j in live:
                    j.future.set_exception(exc)
                if self.on_error is not None:
                    self.on_error(self.model.name, exc)
                return
        finally:
            self.gate.release()
        took = (self.clock() - t0) * 1e3
        self.forwards += len(sizes)
        self.rows_total += len(flat)
        self.last_path = path
        self.forward_ms.append(took / len(sizes))
        self.batch_rows.extend(sizes)
        for ji, j in enumerate(live):
            j.future.set_result({"answers": results[ji], "input_tokens": tokens[ji],
                                 "batch_rows": len(j.rows) if self.collate == "call" else len(flat),
                                 "path": path, "queue_ms": round((t0 - j.t_submit) * 1e3, 2),
                                 "forward_ms": round(took / len(sizes), 2)})
