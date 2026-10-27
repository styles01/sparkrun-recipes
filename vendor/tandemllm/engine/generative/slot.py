"""The small-model slot: one generative model inside the engine, its load and its calls.

    slot = Slot(LlamaModel("minicpm5-2b", d), role="narrator", gate=LOCK, planner=..., device=...)
    slot.load()                                   # through the residency planner, in a thread
    r = slot.run(messages, max_new=24, budget_ms=150)
    r -> {text, finish (stop|length|time), ms, tokens, prefilled, reused} or {why: ...}

Loading is the decision models' (engine/decision/registry.py): every load asks
`planner(name, loading=, service_start=, hold_restart=)` -- the residency planner's check for the
slot's role -- and an allowed load runs in a thread of its own, the weights copied onto the device
in the idle window into the model's own memory pool (engine/decision/device.py). Nothing loads on
its own at start: the slot loads when its first caller asks (a profile that switched it on), and
until it is `loaded` every call answers `why: loading` (or `refused`, `failed`) at once.

A call never waits for the engine. `run(held=True)` is for a caller that already holds the
engine's lock -- the request handler between two of Qwen's rounds -- and it spends at most its
budget there: the device is synchronised first (whatever Qwen queued finishes before the model
starts), the deadline is checked before every token, so the lock is held at most the budget and
one step. `run(held=False)` (the dashboard's and the measurement's try) takes the lock first,
waiting at most `wait_s` for the idle window. A second call while one runs gets `why: busy`. Three
failed calls within a minute unload the model and mark it `failed` until it is loaded again.

A job (N2, a model with `jobs`) is one generation in slices: `begin(ids, max_new=, keep=)` makes it
(or says why not, as `run` does), `advance(job, budget_ms=)` runs one slice of it on the request's
thread, the lock held, and can be called again after Qwen's next round.
"""

from __future__ import annotations

import collections
import statistics
import threading
import time

ERRORS_TO_FAIL = 3
ERROR_WINDOW_S = 60.0
REFUSAL_REUSE_S = 60.0


def _took(gib) -> str:
    """The load's MemAvailable change for its log line: ", MemAvailable -1.47 GiB" when it took
    memory, "+2.87" when MemAvailable rose over the load (a shed or another program let go of
    memory meanwhile; the line read "--2.87" before)."""
    if gib is None:
        return ""
    return f", MemAvailable {'-' if gib >= 0 else '+'}{abs(gib)} GiB"


class Slot:
    def __init__(self, model, *, role: str, gate, planner=None, device=None, log=print,
                 contract=None, journal=None, mem_available=None, clock=time.monotonic):
        """`contract(fields, path) -> [errors]`: the role contract the model must sign (SRV-133);
        a model that breaks it stops the start. `journal(rec)`: the residency journal, which gets
        the load's measured MemAvailable drop (`mem_available() -> GiB` before and after)."""
        if contract is not None:
            errs = contract(model.role(), model.name)
            if errs:
                raise ValueError(f"{model.name} breaks the role contract: {'; '.join(errs)}")
        self.model, self.role = model, role
        self.gate, self.planner, self.device = gate, planner, device
        self.log, self.journal, self.mem_available, self.clock = log, journal, mem_available, clock
        self.state = "not_loaded"
        self.error: str | None = None
        self.failed_at: float | None = None      # the clock when it last turned `failed`
        self.refusal: dict | None = None
        self.loaded_at: float | None = None
        self.load_s: float | None = None
        self.took_gib: float | None = None
        self.calls = collections.Counter()
        self.ms: collections.deque = collections.deque(maxlen=200)
        self._errors: collections.deque = collections.deque()
        self._busy = threading.Lock()
        self._lock = threading.RLock()
        self._where = None
        self.claim_gib = None
        self.loads = 0                          # completed loads since the start
        # the role governor (server/rolemem.py) unloaded this model for memory: nothing loads it
        # again on its own (a caller's `ensure`, the narrator's keeper, the shadow's preload)
        # until the governor lifts the hold after memory came back
        self.memory_hold = False

    # ------------------------------------------------------------------ loading
    def load(self, *, service_start: bool = False, hold_restart: bool = False,
             wait: bool = False) -> dict:
        """Start the load if the planner allows it; the planner's decision with the state."""
        with self._lock:
            if self.state in ("loaded", "loading"):
                return {"allowed": True, "reason": None, "role": self.role, "state": self.state}
            if self.memory_hold:
                return {"allowed": False, "reason": "unloaded_for_memory", "role": self.role,
                        "state": self.state}
            d = (self.planner(self.model.name, loading=False, service_start=service_start,
                              hold_restart=hold_restart)
                 if self.planner is not None else {"allowed": True, "reason": None,
                                                   "role": self.role})
            if not d.get("allowed"):
                self.refusal = dict(d, at=time.time())
                self.log(f"[{self.role}] {self.model.name}: load refused ({d.get('reason')})")
                return dict(d, state=self.state)
            self.refusal, self.error = None, None
            # what the planner admitted the load with: the pool's bytes up to it are this role's,
            # not the engine's (server/memlimits.role_exclude)
            self.claim_gib = d.get("claim_gib")
            self.state = "loading"
        t = threading.Thread(target=self._load, name=f"{self.role}-load", daemon=True)
        t.start()
        if wait:
            t.join()
        return dict(d, state=self.state)

    def _load(self) -> None:
        m = self.model
        t0 = time.time()
        before = self.mem_available() if self.mem_available is not None else None
        try:
            where = None
            if self.device is not None:
                from engine.decision.device import Placement
                where = Placement(self.device, self.gate)
            # the pool is known from the load's first copy on: its bytes are the role's while it
            # loads too (the memory gate of a request served between two batches read them as the
            # engine's and shed the resident caches, 2026-10-04 21:41 and 2026-10-05 06:18), and
            # a failed load drops it (`_drop`)
            with self._lock:
                self._where = where
            m.load(where)
            with self._lock:
                self.state = "loaded"
                self.loads += 1
                self.loaded_at = time.time()
                self.load_s = round(time.time() - t0, 2)
            after = self.mem_available() if self.mem_available is not None else None
            if before is not None and after is not None:
                self.took_gib = round(before - after, 2)
                if self.journal is not None:
                    self.journal({"kind": "loaded", "role": self.role, "before_gib": before,
                                  "after_gib": after, "took_gib": self.took_gib,
                                  "claim_gib": m.memory_claim_gib,
                                  "device_bytes": m.device_bytes(),
                                  "source": f"engine:{m.name}"})
            self.log(f"[{self.role}] {m.name}: loaded in {self.load_s:.1f}s "
                     f"({m.precision() or 'precision unknown'}, {m.device_bytes() / 2**30:.2f} GiB"
                     f"{_took(self.took_gib)})")
        except Exception as exc:                                  # noqa: BLE001
            self.error = f"{type(exc).__name__}: {exc}"[:400]
            self.log(f"[{self.role}] {m.name}: load failed ({self.error})")
            self._drop()
            with self._lock:
                self.state = "failed"
                self.failed_at = self.clock()

    def unload(self, block: bool = True) -> bool:
        """`block=False`: give up (False) when a call is in flight rather than wait for it (the
        memory gate calls this with the engine lock held; a call may be waiting for that lock)."""
        if not block and not self._busy.acquire(blocking=False):
            return False
        try:
            with self._lock:
                if self.state not in ("loaded", "failed"):
                    return False
                self.state = "unloading"
            if block:
                with self._busy:                   # a call in flight finishes first
                    self._drop()
            else:
                self._drop()
        finally:
            if not block:
                self._busy.release()
        with self._lock:
            self.state = "not_loaded"
            self.loaded_at = None
        self.log(f"[{self.role}] {self.model.name}: unloaded")
        return True

    def _drop(self) -> None:
        where, self._where = self._where, None
        try:
            self.model.unload()
        finally:
            if where is not None:
                where.release()

    def ensure(self) -> str | None:
        """None when loaded; else why not, starting a load when none ran (a refusal is reused
        for a minute rather than asking the planner on every line)."""
        st = self.state
        if st == "loaded":
            return None
        if st == "not_loaded" and self.memory_hold:
            return "unloaded_for_memory"         # the governor reloads it, not a caller
        if st == "not_loaded":
            last = self.refusal
            if last is not None and time.time() - last.get("at", 0) < REFUSAL_REUSE_S:
                return "refused"
            d = self.load()
            return "loading" if d.get("allowed") else "refused"
        return st                               # loading, unloading, failed

    def failed_for(self) -> float | None:
        """Seconds since the slot turned `failed`; None when it is not failed."""
        if self.state != "failed" or self.failed_at is None:
            return None
        return self.clock() - self.failed_at

    def reset_failed(self) -> bool:
        """A failed slot back to `not_loaded`, so the next `ensure` or `load` tries again (VIS-153:
        the shadow reader's keeper, once a retry interval passed). False when it was not failed."""
        with self._lock:
            if self.state != "failed":
                return False
            self.state = "not_loaded"
            return True

    # ------------------------------------------------------------------ calls
    def run(self, messages: list[dict], *, max_new: int, budget_ms: float, held: bool = True,
            wait_s: float = 30.0) -> dict:
        """One greedy generation inside `budget_ms` (from the call, prefill included)."""
        why = self.ensure()
        if why is not None:
            self.calls[why] += 1
            return {"why": why}
        if not self._busy.acquire(blocking=False):
            self.calls["busy"] += 1
            return {"why": "busy"}
        try:
            got = held or self.gate is None or self.gate.acquire(timeout=wait_s)
            if not got:
                self.calls["idle_wait"] += 1
                return {"why": "idle_wait"}
            try:
                t0 = self.clock()
                if self.device is not None and self.device.cuda:
                    # whatever Qwen queued finishes first: the model never runs beside it
                    import torch
                    torch.cuda.synchronize(self.device.device)
                ids = self.model.encode(messages)
                r = self.model.generate(ids, max_new, t0 + budget_ms / 1000.0)
                text = self.model.decode(r["ids"]) if r["ids"] else ""
                ms = round((self.clock() - t0) * 1e3, 2)
            finally:
                if not held and self.gate is not None:
                    self.gate.release()
        except Exception as exc:                                  # noqa: BLE001
            self.calls["error"] += 1
            self._failed(exc)
            return {"why": "error", "error": f"{type(exc).__name__}: {exc}"[:200]}
        finally:
            self._busy.release()
        self.calls[r["finish"]] += 1
        self.ms.append(ms)
        return {"text": text, "finish": r["finish"], "ms": ms, "tokens": len(r["ids"]),
                "prompt_tokens": len(ids), "prefilled": r.get("prefilled"),
                "reused": r.get("reused")}

    # ------------------------------------------------------------------ jobs (N2)
    def begin(self, ids: list[int], *, max_new: int, keep: int = 0) -> dict:
        """`{job}`: a generation to advance in slices (a model with `jobs`), or `{why}` at once
        (not loaded yet, which starts the load; refused; failed)."""
        why = self.ensure()
        if why is not None:
            self.calls[why] += 1
            return {"why": why}
        try:
            return {"job": self.model.begin(ids, max_new, keep=keep)}
        except Exception as exc:                                  # noqa: BLE001
            self.calls["error"] += 1
            return {"why": "error", "error": f"{type(exc).__name__}: {exc}"[:200]}

    def advance(self, job, *, budget_ms: float) -> dict:
        """One slice of `job`, on the request's own thread between two of Qwen's rounds (the
        engine's lock is the request's): the device synchronised first, then prefill chunks and
        decode steps until `budget_ms` passed (checked before each) or the job ended.
        `{ms, done}` or `{why}`; the job keeps its own total (`ms`, `slices`)."""
        if self.state != "loaded":
            self.calls[self.state] += 1
            return {"why": self.state}
        if not self._busy.acquire(blocking=False):
            self.calls["busy"] += 1
            return {"why": "busy"}
        try:
            t0 = self.clock()
            if self.device is not None and self.device.cuda:
                import torch
                torch.cuda.synchronize(self.device.device)
            self.model.advance(job, t0 + budget_ms / 1000.0)
            ms = round((self.clock() - t0) * 1e3, 2)
        except Exception as exc:                                  # noqa: BLE001
            self.calls["error"] += 1
            self._failed(exc)
            return {"why": "error", "error": f"{type(exc).__name__}: {exc}"[:200]}
        finally:
            self._busy.release()
        job.ms += ms
        job.slices += 1
        if job.done:
            self.calls[job.finish] += 1
            self.ms.append(round(job.ms, 2))
        return {"ms": ms, "done": job.done}

    def _failed(self, exc: BaseException) -> None:
        now = self.clock()
        q = self._errors
        q.append(now)
        while q and now - q[0] > ERROR_WINDOW_S:
            q.popleft()
        self.log(f"[{self.role}] {self.model.name}: call failed ({type(exc).__name__}: {exc})")
        if len(q) >= ERRORS_TO_FAIL and self.state == "loaded":
            self.error = (f"{len(q)} failed calls within {ERROR_WINDOW_S:.0f} s; last: "
                          f"{type(exc).__name__}: {exc}")[:400]
            q.clear()

            def fail():
                self.unload()
                with self._lock:
                    self.state = "failed"
                    self.failed_at = self.clock()

            threading.Thread(target=fail, name=f"{self.role}-fail", daemon=True).start()

    # ------------------------------------------------------------------ what it reports
    def pool_bytes(self) -> tuple[int, int]:
        """(reserved, allocated) of the model's own pool (Placement.pool_bytes), (0, 0) without."""
        where = self._where
        return where.pool_bytes() if where is not None else (0, 0)

    def device_bytes(self) -> int:
        try:
            return int(self.model.device_bytes()) if self.state == "loaded" else 0
        except Exception:                                         # noqa: BLE001
            return 0

    def status(self) -> dict:
        m = self.model
        ms = sorted(self.ms)
        return {"name": m.name, "role": self.role, "family": m.family, "owned_by": m.owned_by,
                "contract": m.role(), "state": self.state, "error": self.error,
                "refusal": self.refusal, "memory_hold": self.memory_hold,
                "loaded_at": self.loaded_at, "load_s": self.load_s,
                "took_gib": self.took_gib, "precision": m.precision(),
                "device": str(self.device) if self.device is not None else None,
                "device_bytes": self.device_bytes(), "weight_hash": m.weight_hash(),
                "calls": dict(self.calls),
                "ms_p50": round(statistics.median(ms), 2) if ms else None,
                "ms_p90": round(ms[min(len(ms) - 1, int(len(ms) * 0.9))], 2) if ms else None}
