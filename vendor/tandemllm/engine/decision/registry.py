"""The decision models this engine serves: names, the loaded set, routing, loading, asking.

    reg = Registry([LayaModel("laya-english", d), ...], gate=LOCK, planner=..., device=Device())
    reg.load("laya-english")                       # through the residency planner, in a thread
    fut = reg.ask(state, questions, model="auto", window="idle", deadline_ms=100)
    fut.result()  -> {model, answers, usage, routing, path, latency_ms, queue_ms, forward_ms}

Names. A model answers to its name and its aliases. `auto`, `jev-latest`, `jev`, `default` and a
family's own name (`laya`) mean "let the family's router choose" (for Laya: script and language
detection, then the checkpoint); a family with no router answers with its first model. The
router may pick a model that is not loaded: the loaded default of the family answers instead and
the routing says so (arbiter's rule), and an on-demand load of the one it wanted starts.

Loading. Every load goes through `planner(name, loading=, service_start=, hold_restart=)` -- in
the server that is the OPS-41 residency planner (`server/decider_api.py`): MemAvailable minus the
decider's claim at or above floor plus margin, never during a hold, never two loads at once. An
allowed load runs in its own thread: the weights onto the device in the idle window, then one
warm-up question, then `loaded`. A model whose forward fails 3 times within a minute is unloaded
and marked `failed` until someone loads it again; the engine serves on.

The load policy per model (`on_start`, `on_demand`, `off`) is the conductor's `deciders`
document, read through `policy()`; `autoload()` loads the `on_start` ones one after the other
(ops/start.sh asks for it once the engine is healthy and its page cache is dropped).
"""

from __future__ import annotations

import collections
import threading
import time
from concurrent.futures import Future

from engine.decision.base import (DecisionModel, NotReadyError, OverloadedError, WindowError)
from engine.decision.batcher import Batcher

AUTO_NAMES = ("auto", "jev-latest", "jev", "default", "")
ERRORS_TO_FAIL = 3
ERROR_WINDOW_S = 60.0
# a question for a model the planner refused reuses the refusal this long instead of asking the
# planner (and its journal) again on every request; Load in the dashboard always asks again
REFUSAL_REUSE_S = 60.0
WARM_DEADLINE_S = 600.0


# what /health and the loud log line keep of a planner's refusal
REFUSAL_KEYS = ("reason", "at", "mem_available_gib", "engine_growth_gib", "claim_gib", "after_gib",
                "need_gib")


def _loud(name: str, d: dict) -> str:
    """The one loud line when no decision model could load: without one the conductor decides no
    class on a model whose fit leaves the class question to the decider (every turn falls back)."""
    nums = ""
    if d.get("after_gib") is not None:
        nums = (f": {d.get('mem_available_gib')} GiB available - {d.get('engine_growth_gib')} "
                f"the engine may still take - {d.get('claim_gib')} claim = {d['after_gib']} GiB, "
                f"under floor + margin {d.get('need_gib')}")
    return (f"[deciders] WARNING: no decision model is loaded: the residency planner refused "
            f"{name} ({d.get('reason')}{nums}). Turns get the profile's default class until it "
            f"loads; a question asks for the load again once a minute. /health roles.laya.refused "
            f"has the numbers")


class UnknownModel(ValueError):
    """A `model` no served decision model answers to."""


def _allowed(name: str) -> dict:
    return {"allowed": True, "reason": None, "role": name}


class Registry:
    def __init__(self, models: list[DecisionModel], *, gate, planner=None, device=None,
                 max_batch: int = 64, wait_ms: float = 2.0, max_queue: int = 256,
                 collate: str = "call", policy=None, routers: dict | None = None,
                 clock=time.monotonic, log=print, contract=None):
        """`contract(fields, path) -> [errors]`: the role contract every model must sign
        (SRV-133; the server passes the conductor's `schema.role_errors`). A model that breaks it
        stops the start, as an unknown model name does."""
        if not models:
            raise ValueError("a registry needs at least one decision model")
        if contract is not None:
            errs = [e for m in models for e in contract(m.role(), m.name)]
            if errs:
                raise ValueError(f"a decision model breaks the role contract: {'; '.join(errs)}")
        self.models = {m.name: m for m in models}
        self.order = [m.name for m in models]
        self.gate, self.planner, self.device = gate, planner, device
        self.max_batch, self.wait_ms, self.max_queue = int(max_batch), float(wait_ms), int(max_queue)
        self.collate = collate
        self.policy = policy or (lambda: {})
        self.routers = dict(routers or {})
        self.clock, self.log = clock, log
        self.state = {n: "not_loaded" for n in self.order}
        self.error: dict[str, str] = {}
        self.refusal: dict[str, dict] = {}
        self.loaded_at: dict[str, float] = {}
        self.warm: dict[str, bool] = {}
        self.questions = collections.Counter()
        self.picks: dict[str, collections.deque] = collections.defaultdict(
            lambda: collections.deque(maxlen=100))
        self._errors: dict[str, collections.deque] = collections.defaultdict(collections.deque)
        self._batchers: dict[str, Batcher] = {}
        self._placements: dict = {}
        self._loading: str | None = None
        # the loud line for a refusal while no model is loaded goes out once per server
        self._refusal_said = False
        # models the role governor (server/rolemem.py) unloaded for memory: no on-demand load
        # brings them back; the governor lifts the hold and loads them when memory came back
        self.memory_hold: set[str] = set()
        self._lock = threading.RLock()
        self._alias = {}
        for m in models:
            for a in (m.name, *m.aliases):
                self._alias.setdefault(a.lower(), m.name)
        self.families = list(dict.fromkeys(m.family for m in models))

    # ------------------------------------------------------------------ names and routing
    def family_models(self, family: str) -> list[str]:
        return [n for n in self.order if self.models[n].family == family]

    def is_auto(self, model: str | None) -> bool:
        key = (model or "").strip().lower()
        return key in AUTO_NAMES or key in self.families

    def name_of(self, model: str | None) -> str | None:
        """The served model a name or alias means; None for an `auto` name."""
        key = (model or "").strip().lower()
        if self.is_auto(key):
            return None
        if key not in self._alias:
            raise UnknownModel(f"unknown model {model!r}; served: {', '.join(self.order)} "
                               f"(or auto, jev-latest)")
        return self._alias[key]

    def resolve(self, model, state, questions, task=None, lang=None) -> tuple[str, dict]:
        """(the model that answers, the routing record a response carries)."""
        name = self.name_of(model)
        if name is not None:
            return name, {"model": name, "reason": f"explicit model={model!r}",
                          "detection": None, "workflow": None}
        key = (model or "").strip().lower()
        family = key if key in self.families else self.families[0]
        members = self.family_models(family)
        router = self.routers.get(family)
        if router is not None:
            d = dict(router(state, questions, task=task, lang=lang))
        else:
            d = {"model": members[0], "reason": f"the {family} family's first model",
                 "detection": None, "workflow": None}
        want = d["model"]
        if want not in self.models:
            d["requested"] = want
            d["reason"] = f"{d['reason']}; {want} is not served here"
            d["model"] = want = members[0]
        if self.state[want] != "loaded":
            loaded = [n for n in members if self.state[n] == "loaded"]
            if loaded:
                d["requested"] = want
                d["reason"] = f"{d['reason']}; {want} is not loaded here, falling back to {loaded[0]}"
                d["model"] = loaded[0]
                if self.policy().get(want, "on_demand") == "on_demand" and \
                        self.state[want] == "not_loaded":
                    self.load(want)
        return d["model"], d

    def limits(self, model: str | None = "auto") -> dict:
        """A model's limits; for an `auto` name the tightest of the loaded set (of all when none
        is loaded), since the router may pick any of them."""
        name = self.name_of(model)
        if name is not None:
            return self.models[name].limits()
        pool = [n for n in self.order if self.state[n] == "loaded"] or list(self.order)
        out: dict = {}
        for n in pool:
            for k, v in self.models[n].limits().items():
                if v is not None:
                    out[k] = v if k not in out else min(out[k], v)
        return out

    # ------------------------------------------------------------------ asking
    def ask(self, state, questions: dict[str, dict], *, model: str | None = "auto",
            window: str = "idle", deadline_ms: float | None = 100.0, task=None, lang=None,
            source: str | None = None) -> Future:
        """One call's questions, answered in one forward in the next idle window (with
        `collate="rows"`, in one forward with whatever else is waiting). Raises
        `WindowError`, `UnknownModel`, `NotReadyError`, `OptionBudgetError`, `OverloadedError`
        before anything is queued; the future carries `DeadlineMissed` or a model error."""
        if window != "idle":
            raise WindowError(f"window {window!r} is not run in stage A: in-decode questions need "
                              f"the low-priority stream and the governor (stage C)")
        t0 = self.clock()
        name, routing = self.resolve(model, state, questions, task=task, lang=lang)
        st = self.state[name]
        if st != "loaded":
            raise self._not_ready(name, st)
        m = self.models[name]
        rows = m.build_rows(state, questions)
        with self._lock:
            b = self._batchers.get(name)
            if b is None or self.state[name] != "loaded":
                raise self._not_ready(name, self.state[name])
            depth = sum(x.depth for x in self._batchers.values())
            if depth + len(rows) > self.max_queue:
                raise OverloadedError(f"queue is full ({depth} question rows in flight, limit "
                                      f"{self.max_queue})")
            deadline = t0 + deadline_ms / 1000.0 if deadline_ms else None
            inner = b.submit(rows, deadline)
        self.questions[name] += len(questions)
        if source:
            self.picks[source].append(name)
        outer: Future = Future()

        def done(f: Future) -> None:
            try:
                r = f.result()
            except BaseException as exc:                     # noqa: BLE001
                outer.set_exception(exc)
                return
            outer.set_result({"model": name, "answers": dict(zip(questions, r["answers"])),
                              "usage": {"input_tokens": r["input_tokens"], "output_tokens": 0},
                              "routing": routing, "path": r["path"],
                              "latency_ms": round((self.clock() - t0) * 1e3, 2),
                              "queue_ms": r["queue_ms"], "forward_ms": r["forward_ms"],
                              "batch_rows": r["batch_rows"]})

        inner.add_done_callback(done)
        return outer

    def _not_ready(self, name: str, st: str) -> NotReadyError:
        if st in ("loading", "warming"):
            return NotReadyError("warming" if st == "warming" else "loading",
                                 f"{name} is {st}; ask again in a few seconds")
        if st == "failed":
            return NotReadyError("failed", f"{name} failed and was unloaded: "
                                           f"{self.error.get(name, '')}; load it again to retry")
        if name in self.memory_hold:
            err = NotReadyError("unloaded_for_memory",
                                f"{name} was unloaded while memory was short; it loads again "
                                f"once memory has come back")
            err.retry = 30
            return err
        pol = self.policy().get(name, "on_demand")
        if pol == "off":
            err = NotReadyError("not_loaded", f"{name} is not loaded and its load policy is off")
            err.retry = None
            return err
        last = self.refusal.get(name)
        if last is not None and time.time() - last.get("at", 0) < REFUSAL_REUSE_S:
            d = last
        else:
            d = self.load(name)
        if d.get("allowed"):
            return NotReadyError("loading", f"{name} was not loaded; its load has started")
        return NotReadyError("refused", f"{name} is not loaded and the residency planner refused "
                                        f"its load: {d.get('reason')}", decision=d)

    # ------------------------------------------------------------------ loading
    def load(self, name: str, *, service_start: bool = False, hold_restart: bool = False,
             wait: bool = False) -> dict:
        """Start loading `name` if the planner allows it. Returns the planner's decision with the
        model's state; a refusal is kept for the dashboard and nothing loads."""
        with self._lock:
            st = self.state[name]
            if st in ("loaded", "loading", "warming"):
                return dict(_allowed("decider"), state=st)
            if name in self.memory_hold:
                return {"allowed": False, "reason": "unloaded_for_memory", "role": "decider",
                        "state": st}
            d = (self.planner(name, loading=self._loading is not None,
                              service_start=service_start, hold_restart=hold_restart)
                 if self.planner is not None else _allowed("decider"))
            if not d.get("allowed"):
                self.refusal[name] = dict(d, at=time.time())
                self.log(f"[deciders] {name}: load refused ({d.get('reason')})")
                if not self.ready() and not self._refusal_said:
                    self._refusal_said = True
                    self.log(_loud(name, d))
                return dict(d, state=st)
            self.refusal.pop(name, None)
            self.error.pop(name, None)
            self._loading = name
            self.state[name] = "loading"
        t = threading.Thread(target=self._load, args=(name,), name=f"decider-load-{name}",
                             daemon=True)
        t.start()
        if wait:
            t.join()
        return dict(d, state=self.state[name])

    def _load(self, name: str) -> None:
        m = self.models[name]
        t0 = time.time()
        try:
            where = None
            if self.device is not None:
                from engine.decision.device import Placement
                where = Placement(self.device, self.gate)
            m.load(where)
            with self._lock:
                self._placements[name] = where
                self.state[name] = "warming"
                self._batchers[name] = Batcher(m, self.gate, max_batch=self.max_batch,
                                               wait_ms=self.wait_ms, collate=self.collate,
                                               on_error=self._on_error, clock=self.clock)
            st, q = m.warm_question()
            try:
                self._batchers[name].submit(m.build_rows(st, q),
                                            self.clock() + WARM_DEADLINE_S).result()
                self.warm[name] = True
            except Exception as exc:                          # noqa: BLE001
                self.warm[name] = False
                self.log(f"[deciders] {name}: warm-up did not run ({type(exc).__name__}: {exc})")
            with self._lock:
                self.state[name] = "loaded"
                self.loaded_at[name] = time.time()
            self.log(f"[deciders] {name}: loaded in {time.time() - t0:.1f}s "
                     f"({m.precision() or 'precision unknown'})")
        except Exception as exc:                              # noqa: BLE001
            self.error[name] = f"{type(exc).__name__}: {exc}"[:400]
            self.log(f"[deciders] {name}: load failed ({self.error[name]})")
            self._drop(name)
            with self._lock:
                self.state[name] = "failed"
        finally:
            with self._lock:
                self._loading = None

    def unload(self, name: str) -> bool:
        """Drop `name`: the questions already queued are answered first, then its tensors and its
        pool go. False when it was not loaded."""
        with self._lock:
            if self.state[name] not in ("loaded", "warming", "failed"):
                return False
            self.state[name] = "unloading"
        self._drop(name)
        with self._lock:
            self.state[name] = "not_loaded"
            self.loaded_at.pop(name, None)
            self.warm.pop(name, None)
        self.log(f"[deciders] {name}: unloaded")
        return True

    def unload_if_idle(self, name: str) -> bool:
        """Unload `name` only when no question of it is queued or running (its batcher is
        idle, so stopping it never waits): the role governor's call, which may run on a thread
        that holds the engine lock a queued batch would wait for. False otherwise."""
        with self._lock:
            if self.state.get(name) != "loaded":
                return False
            b = self._batchers.get(name)
            if b is not None and b.depth > 0:
                return False
            self.state[name] = "unloading"
        self._drop(name)
        with self._lock:
            self.state[name] = "not_loaded"
            self.loaded_at.pop(name, None)
            self.warm.pop(name, None)
        self.log(f"[deciders] {name}: unloaded (idle)")
        return True

    def _drop(self, name: str) -> None:
        with self._lock:
            b = self._batchers.pop(name, None)
            where = self._placements.pop(name, None)
        if b is not None:
            b.stop()
        try:
            self.models[name].unload()
        finally:
            if where is not None:
                where.release()

    def autoload(self, *, service_start: bool = False, hold_restart: bool = False) -> list[str]:
        """Load the `on_start` models one after the other, in a thread of its own."""
        pol = self.policy()
        names = [n for n in self.order if pol.get(n) == "on_start"]

        def run():
            for n in names:
                d = self.load(n, service_start=service_start, hold_restart=hold_restart,
                              wait=True)
                if not d.get("allowed"):
                    break                     # the next one would be refused for the same reason

        if names:
            threading.Thread(target=run, name="decider-autoload", daemon=True).start()
        return names

    def _on_error(self, name: str, exc: BaseException) -> None:
        now = self.clock()
        q = self._errors[name]
        q.append(now)
        while q and now - q[0] > ERROR_WINDOW_S:
            q.popleft()
        self.log(f"[deciders] {name}: forward failed ({type(exc).__name__}: {exc})")
        if len(q) >= ERRORS_TO_FAIL and self.state.get(name) == "loaded":
            self.error[name] = (f"{len(q)} failed forwards within {ERROR_WINDOW_S:.0f} s; last: "
                                f"{type(exc).__name__}: {exc}")[:400]
            q.clear()

            def fail():
                self.unload(name)
                with self._lock:
                    self.state[name] = "failed"

            # never from the batcher's own thread: unloading joins it
            threading.Thread(target=fail, name=f"decider-fail-{name}", daemon=True).start()

    # ------------------------------------------------------------------ what it reports
    def ready(self) -> bool:
        return any(s == "loaded" for s in self.state.values())

    def loading(self) -> bool:
        return any(s in ("loading", "warming") for s in self.state.values())

    def loaded(self) -> list[str]:
        return [n for n in self.order if self.state[n] == "loaded"]

    def refused(self) -> dict | None:
        """While no model is loaded: the newest refusal of a load (the model, the reason and the
        planner's numbers) for /health; None when one is loaded or none was refused."""
        if self.ready() or not self.refusal:
            return None
        name, d = max(self.refusal.items(), key=lambda kv: kv[1].get("at", 0))
        return {"model": name, **{k: d.get(k) for k in REFUSAL_KEYS if d.get(k) is not None}}

    def status(self) -> list[dict]:
        pol = self.policy()
        out = []
        for n in self.order:
            m = self.models[n]
            b = self._batchers.get(n)
            out.append({"name": n, "family": m.family, "contract": m.contract, "role": m.role(),
                        "aliases": sorted(m.aliases), "owned_by": m.owned_by,
                        "state": self.state[n], "load": pol.get(n, "on_demand"),
                        "limits": m.limits(), "precision": m.precision(),
                        "device": str(self.device.device) if self.device is not None else None,
                        "device_bytes": m.device_bytes(), "weight_hash": m.weight_hash(),
                        "warm": self.warm.get(n), "loaded_at": self.loaded_at.get(n),
                        "questions": self.questions[n], "error": self.error.get(n),
                        "refusal": self.refusal.get(n), "memory_hold": n in self.memory_hold,
                        "stats": b.stats() if b is not None else None})
        return out

    def picks_by_source(self) -> dict[str, dict]:
        return {src: dict(collections.Counter(q)) for src, q in self.picks.items()}
