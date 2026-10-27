"""The small models under memory pressure: unloaded before a conversation is parked or a request
waits, loaded again once memory has come back.

The engine runs up to three small models beside Qwen: the shadow reader (Qwen3.5-0.8B, about
1.9 GiB), the narrator (Qwen3.5-2B, about 4 GiB) and the Laya deciders (about 1.7 GiB). Until
0.6.0 they loaded through the residency planner at the start and were then kept; when the shared
KV pool (engine/kvpool.py) needed pages beyond its floor it gave back the caches, then idle side
conversations, then slot 0's idle conversation (to the NVMe), and only then made a request wait.
The roles never gave anything back.

This module is the `roles` level of that order, between the caches and the conversations:

    caches -> shadow reader -> narrator -> (Laya, only with QSE_ROLE_UNLOAD_LAYA=on)
           -> idle side conversations -> slot 0's idle conversation -> the request waits

WHY THIS ORDER. Nothing the user reads changes while a role is out: the shadow reader only
journals its answers beside the router's (shadow mode) and feeds the narrator's summaries; the
narrator only writes the status lines (they stop while it is out, the answer is the same). An
idle conversation that is parked must be read back or re-read on its next turn, which costs the
user seconds to minutes. So the roles go first, cheapest first.

WHY LAYA STAYS BY DEFAULT. Laya answers the router's questions (the class, the packages, the
effort). Without it the router falls through to its next stage, Qwen itself, whose answer can
differ: the same request would then be given other packages or another effort, so the TEXT can
change, and the fallback spends an engine turn and guest KV rows exactly while memory is short.
Its claim (3 GiB, 1.7 GiB measured) is small next to what one parked conversation gives back,
losslessly. So Laya is unloaded only when `QSE_ROLE_UNLOAD_LAYA=on`, after the narrator, and only
when none of its questions is queued (its batcher can stop at once).

HOW A ROLE STAYS OUT. Unloading sets a memory hold on the role (`Slot.memory_hold`,
`Registry.memory_hold`): a caller that wants it (`Slot.ensure`, a decider question, the narrator's
keeper, the shadow's preload) gets `unloaded_for_memory` at once and nothing loads. Only this
governor lifts the hold.

HOW IT COMES BACK, WITHOUT THRASHING. A thread looks every `tick_s`. The first role out in the
reload order (Laya, narrator, shadow: the reverse of the unload order) loads again when

  * no request waits (in the queue or for memory), and no reclaim or memory wait happened for
    `hold_s` seconds, and
  * the headroom -- MemAvailable less the caches' growth left and what the requests in flight
    reserved (server/slots.py `Admission.check`) -- has stayed at or above
    `floor + the role's claim + margin` for `hold_s` seconds without a break (hysteresis: the
    reload line is a role and a margin above the line the unload was made at).

The load itself goes through the residency planner as at the start (`Slot.load`,
`Registry.load`), one role at a time. A role unloaded again within `thrash_s` of its reload has
its `hold_s` doubled (up to `max_hold_s`), and it halves back after a quiet `thrash_s`.

Every unload and reload is a log line and a row in the residency journal
(`<conductor root>/residency.jsonl`, kinds `role_unloaded`, `role_reload`, `role_reloaded`,
`role_reload_refused`), and `/health` `roles` shows each role's state: `loaded`,
`unloaded_for_memory`, `reloading` (or the slot's own state: `not_loaded`, `loading`, `failed`).
"""

from __future__ import annotations

import collections
import os
import threading
import time

GIB = float(1 << 30)
ORDER = ("shadow", "narrator", "laya")


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, "") or default)
    except ValueError:
        return default


def _env_on(key: str, default: str) -> bool:
    return (os.environ.get(key) or default).strip().lower() in ("1", "on", "true", "yes")


# ------------------------------------------------------------------ the roles it can unload
def slot_detail(slot) -> dict:
    """VIS-153: why a slot is not loaded, for `/health` `roles`: its load error when it failed,
    the planner's reason when its last load was refused."""
    st = str(getattr(slot, "state", ""))
    ref = getattr(slot, "refusal", None) or {}
    return {"error": getattr(slot, "error", None) if st == "failed" else None,
            "refused": ref.get("reason") if st == "not_loaded" else None}


class SlotRole:
    """A generative role over its `engine/generative/slot.py` Slot (the narrator, the shadow
    reader). `release(why) -> bytes` gives the model's pool back its own way (the shadow parks its
    sessions first); without one, `Slot.unload(block=False)` (never waits for a call in flight)."""

    kind = "generative"

    def __init__(self, name: str, slot, release=None, claim_gib=None):
        self.name, self.slot, self._release = name, slot, release
        self._claim = claim_gib

    @property
    def models(self) -> list[str]:
        return [getattr(self.slot.model, "name", self.name)]

    def claim_gib(self) -> float:
        for c in (getattr(self.slot, "claim_gib", None), self._claim() if callable(self._claim)
                  else self._claim, getattr(self.slot.model, "memory_claim_gib", None)):
            try:
                if c:
                    return float(c)
            except (TypeError, ValueError):
                continue
        return 0.0

    def state(self) -> str:
        return str(self.slot.state)

    def detail(self) -> dict:
        return slot_detail(self.slot)

    def loaded(self) -> bool:
        return self.slot.state == "loaded"

    def busy(self) -> bool:
        b = getattr(self.slot, "_busy", None)
        return bool(b is not None and b.locked())

    def reserved_bytes(self) -> int:
        try:
            return int(self.slot.pool_bytes()[0]) or int(self.slot.device_bytes())
        except Exception:                                          # noqa: BLE001
            return 0

    def hold(self, on: bool) -> None:
        self.slot.memory_hold = bool(on)

    def unload(self, why: str) -> bool:
        if self._release is not None:
            try:
                self._release(why)
            except Exception:                                      # noqa: BLE001
                pass
            if self.slot.state == "loaded":
                # the release found nothing to hand back (no pool on the CPU): unload directly
                self.slot.unload(block=False)
        else:
            self.slot.unload(block=False)
        return self.slot.state != "loaded"

    def load(self) -> dict:
        return self.slot.load()


class RegistryRole:
    """The decision models (engine/decision/registry.py): every loaded model of the registry goes
    out together, only when none of their questions is queued, and the same ones come back."""

    kind = "decider"

    def __init__(self, name: str, reg, claim_gib=None):
        self.name, self.reg, self._claim = name, reg, claim_gib
        self.out: list[str] = []

    @property
    def models(self) -> list[str]:
        return list(self.out) or self.reg.loaded()

    def claim_gib(self) -> float:
        c = self._claim() if callable(self._claim) else self._claim
        try:
            return float(c or 0.0) * max(1, len(self.out) or len(self.reg.loaded()) or 1)
        except (TypeError, ValueError):
            return 0.0

    def state(self) -> str:
        names = self.out or self.reg.loaded()
        sts = [self.reg.state.get(n) for n in names]
        if not sts:
            return "not_loaded"
        if all(s == "loaded" for s in sts):
            return "loaded"
        for s in ("loading", "warming", "unloading", "failed"):
            if s in sts:
                return "loading" if s == "warming" else s
        return "not_loaded"

    def detail(self) -> dict:
        """0.11.3: the planner's refusal while no model is loaded (`Registry.refused`)."""
        return {"error": None, "refused": self.reg.refused() if hasattr(self.reg, "refused")
                else None}

    def loaded(self) -> bool:
        return bool(self.reg.loaded()) and all(self.reg.state.get(n) == "loaded"
                                               for n in (self.out or self.reg.loaded()))

    def busy(self) -> bool:
        return any(getattr(b, "depth", 0) > 0 for b in list(self.reg._batchers.values()))

    def reserved_bytes(self) -> int:
        try:
            return int(sum(s["device_bytes"] for s in self.reg.status()
                           if s.get("state") == "loaded"))
        except Exception:                                          # noqa: BLE001
            return 0

    def hold(self, on: bool) -> None:
        names = self.out or self.reg.loaded()
        if on:
            self.reg.memory_hold.update(names)
        else:
            self.reg.memory_hold.difference_update(names)

    def unload(self, why: str) -> bool:
        names = self.reg.loaded()
        self.reg.memory_hold.update(names)
        gone = [n for n in names if self.reg.unload_if_idle(n)]
        self.reg.memory_hold.difference_update(set(names) - set(gone))
        self.out = gone
        return bool(gone)

    def load(self) -> dict:
        d = {"allowed": False, "reason": "nothing to load"}
        for n in self.out:
            d = self.reg.load(n)
            if not d.get("allowed"):
                return d
        return d


# ------------------------------------------------------------------ the governor
class RoleMemory:
    def __init__(self, roles: list, *, floor_gib: float = 15.0, margin_gib: float = 4.0,
                 hold_s: float = 60.0, thrash_s: float = 600.0, max_hold_s: float = 3600.0,
                 tick_s: float = 5.0, meminfo=None, headroom=None, idle=None, empty=None,
                 journal=None, log=print, clock=time.monotonic, wall=time.time,
                 enabled: bool = True):
        """`roles`: adapters in UNLOAD order (shadow, narrator[, laya]). `meminfo() -> bytes`
        MemAvailable; `headroom() -> bytes` what a new request could still take (MemAvailable
        less growth and reservations; MemAvailable when None); `idle() -> bool` no request waits;
        `empty()` hands the allocator's free blocks back to the driver after an unload;
        `journal(rec)` the residency journal."""
        self.roles = list(roles)
        self.by_name = {r.name: r for r in self.roles}
        self.floor = float(floor_gib) * GIB
        self.margin = float(margin_gib) * GIB
        self.hold_s, self.thrash_s, self.max_hold_s = float(hold_s), float(thrash_s), float(max_hold_s)
        self.tick_s = float(tick_s)
        self.meminfo, self.headroom_fn, self.idle = meminfo, headroom, idle or (lambda: True)
        self.empty, self.journal, self.log = empty, journal, log
        self.clock, self.wall = clock, wall
        self.enabled = bool(enabled)
        self.lock = threading.RLock()
        self.st = {r.name: {"out": False, "reloading": False, "since": None, "unloads": 0,
                            "reloads": 0, "hold_s": self.hold_s, "last_reload": None,
                            "last_unload": None, "freed_gib": None, "why": None,
                            "refusal": None} for r in self.roles}
        self.last_pressure: float | None = None
        self.above_since: float | None = None
        self.events: collections.deque = collections.deque(maxlen=50)
        self.stats = {"unloads": 0, "reloads": 0, "reload_refused": 0, "thrash": 0,
                      "freed_bytes": 0, "skipped_busy": 0}
        self._thread = None
        self._stop = threading.Event()

    # -- helpers
    def _avail(self) -> int | None:
        try:
            return None if self.meminfo is None else self.meminfo()
        except Exception:                                          # noqa: BLE001
            return None

    def _head(self) -> int | None:
        if self.headroom_fn is not None:
            try:
                h = self.headroom_fn()
                if h is not None:
                    return int(h)
            except Exception:                                      # noqa: BLE001
                pass
        return self._avail()

    def _event(self, kind: str, **kw) -> dict:
        ev = dict(kind=kind, ts=round(self.wall(), 3), **kw)
        self.events.append(ev)
        if self.journal is not None:
            try:
                self.journal(dict(ev, source="engine:roles"))
            except Exception:                                      # noqa: BLE001
                pass
        return ev

    def reload_line(self, name: str) -> float:
        """The headroom, in bytes, a role's reload waits for: floor + its claim + margin."""
        return self.floor + self.by_name[name].claim_gib() * GIB + self.margin

    def note_pressure(self, why: str = "pressure") -> None:
        """A reclaim ran or a request waits for memory: no reload for the next `hold_s`."""
        with self.lock:
            self.last_pressure = self.clock()
            self.above_since = None

    def out(self, name: str) -> bool:
        return bool(self.st.get(name, {}).get("out"))

    # -- unloading
    def release(self, name: str, why: str, want: int = 0, owner=None) -> int:
        """Unload one role for memory. Bytes it gave back (measured on MemAvailable when it can
        be read, else the pool's reserved bytes); 0 when it was not loaded or was busy."""
        r = self.by_name.get(name)
        if r is None or not self.enabled:
            return 0
        with self.lock:
            s = self.st[name]
            if s["out"] or s["reloading"] or not r.loaded():
                return 0
            if r.busy():
                self.stats["skipped_busy"] += 1
                return 0
            before = self._avail()
            res = r.reserved_bytes()
            models = r.models
            r.hold(True)
            ok = r.unload(why)
            if not ok:
                r.hold(False)
                self.stats["skipped_busy"] += 1
                return 0
            if self.empty is not None:
                try:
                    self.empty()
                except Exception:                                  # noqa: BLE001
                    pass
            after = self._avail()
            freed = (after - before) if before is not None and after is not None else 0
            freed = freed if freed > 0 else res
            now = self.clock()
            if s["last_reload"] is not None and now - s["last_reload"] < self.thrash_s:
                s["hold_s"] = min(self.max_hold_s, s["hold_s"] * 2)
                self.stats["thrash"] += 1
            s.update(out=True, since=now, since_wall=self.wall(), last_unload=now, why=why,
                     freed_gib=round(freed / GIB, 2), refusal=None)
            s["unloads"] += 1
            self.stats["unloads"] += 1
            self.stats["freed_bytes"] += int(freed)
            self.last_pressure = now
            self.above_since = None
        self._event("role_unloaded", role=name, models=models, why=why, owner=str(owner),
                    want_gib=round(want / GIB, 2), freed_gib=round(freed / GIB, 2),
                    mem_available_before_gib=None if before is None else round(before / GIB, 2),
                    mem_available_after_gib=None if after is None else round(after / GIB, 2),
                    hold_s=s["hold_s"], reload_at_gib=round(self.reload_line(name) / GIB, 2))
        self.log(f"[roles] {name} ({', '.join(models)}) unloaded for memory ({why}): "
                 f"{freed / GIB:.2f} GiB back"
                 + (f", MemAvailable {before / GIB:.1f} -> {after / GIB:.1f} GiB"
                    if before is not None and after is not None else "")
                 + f"; reloads when the headroom stays >= {self.reload_line(name) / GIB:.1f} GiB "
                   f"for {s['hold_s']:.0f} s with nothing waiting")
        return int(freed)

    def reclaim(self, want: int, owner=None, why: str = "kv_pool") -> int:
        """The pool's `roles` level: roles in order until `want` bytes came back."""
        if not self.enabled:
            return 0
        self.note_pressure(why)
        got = 0
        for r in self.roles:
            if got >= want:
                break
            got += self.release(r.name, why, want=max(0, want - got), owner=owner)
        return got

    # -- reloading
    def _settle_reloads(self) -> None:
        for r in self.roles:
            s = self.st[r.name]
            if not s["reloading"]:
                continue
            st = r.state()
            if r.loaded():
                now = self.clock()
                waited = now - (s["since"] or now)
                s.update(out=False, reloading=False, since=None, last_reload=now, why=None)
                s["reloads"] += 1
                self.stats["reloads"] += 1
                self._event("role_reloaded", role=r.name, models=r.models,
                            out_s=round(waited, 1),
                            mem_available_gib=None if self._avail() is None
                            else round(self._avail() / GIB, 2))
                self.log(f"[roles] {r.name} ({', '.join(r.models)}) loaded again after "
                         f"{waited:.0f} s out for memory")
            elif st in ("failed", "not_loaded"):
                # the load ended without the model (a failure, or refused on its own thread)
                r.hold(True)
                s.update(reloading=False, refusal=st)
                self.above_since = None
                self.stats["reload_refused"] += 1
                self._event("role_reload_refused", role=r.name, reason=st)

    def tick(self) -> str | None:
        """One look. The role whose reload started, else None."""
        if not self.enabled:
            return None
        with self.lock:
            self._settle_reloads()
            now = self.clock()
            for r in self.roles:                       # a quiet stretch halves a doubled hold
                s = self.st[r.name]
                if (not s["out"] and s["hold_s"] > self.hold_s and s["last_reload"] is not None
                        and now - s["last_reload"] >= self.thrash_s):
                    s["hold_s"] = max(self.hold_s, s["hold_s"] / 2)
            if any(self.st[r.name]["reloading"] for r in self.roles):
                return None
            cand = next((r for r in reversed(self.roles) if self.st[r.name]["out"]), None)
            if cand is None:
                self.above_since = None
                return None
            s = self.st[cand.name]
            hold = s["hold_s"]
            if not self.idle() or (self.last_pressure is not None
                                   and now - self.last_pressure < hold):
                self.above_since = None
                return None
            head = self._head()
            if head is None or head < self.reload_line(cand.name):
                self.above_since = None
                return None
            if self.above_since is None:
                self.above_since = now
            if now - self.above_since < hold:
                return None
            cand.hold(False)
            try:
                d = cand.load()
            except Exception as exc:                               # noqa: BLE001
                d = {"allowed": False, "reason": f"{type(exc).__name__}: {exc}"}
            if not d.get("allowed"):
                cand.hold(True)
                s["refusal"] = d.get("reason")
                self.above_since = None
                self.last_pressure = now      # wait a whole window before asking again
                self.stats["reload_refused"] += 1
                self._event("role_reload_refused", role=cand.name, reason=d.get("reason"),
                            headroom_gib=round(head / GIB, 2))
                self.log(f"[roles] {cand.name}: reload refused by the residency planner "
                         f"({d.get('reason')})")
                return None
            s["reloading"] = True
            self.above_since = None
            self._event("role_reload", role=cand.name, models=cand.models,
                        headroom_gib=round(head / GIB, 2),
                        line_gib=round(self.reload_line(cand.name) / GIB, 2))
            self.log(f"[roles] {cand.name} ({', '.join(cand.models)}): memory came back "
                     f"(headroom {head / GIB:.1f} GiB >= {self.reload_line(cand.name) / GIB:.1f} "
                     f"for {hold:.0f} s, nothing waiting): loading it again")
            return cand.name

    def start(self) -> None:
        if self._thread is not None or not self.enabled:
            return

        def run():
            while not self._stop.wait(self.tick_s):
                try:
                    self.tick()
                except Exception as exc:                           # noqa: BLE001
                    self.log(f"[roles] tick failed: {type(exc).__name__}: {exc}")

        self._thread = threading.Thread(target=run, name="roles-reload", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    # -- what it reports
    def state_of(self, name: str) -> str:
        s = self.st[name]
        if s["reloading"]:
            return "reloading"
        if s["out"]:
            return "unloaded_for_memory"
        return self.by_name[name].state()

    def report(self) -> dict:
        now = self.clock()
        head = self._head()
        avail = self._avail()
        roles = {}
        with self.lock:
            for r in self.roles:
                s = self.st[r.name]
                roles[r.name] = {
                    "models": r.models, "kind": r.kind, "state": self.state_of(r.name),
                    "claim_gib": round(r.claim_gib(), 2),
                    "reload_at_gib": round(self.reload_line(r.name) / GIB, 2),
                    "out_since": (s.get("since_wall") if s["out"] else None),
                    "out_s": round(now - s["since"], 1) if s["out"] and s["since"] else None,
                    "why": s["why"] if s["out"] else None, "freed_gib": s["freed_gib"],
                    "unloads": s["unloads"], "reloads": s["reloads"], "hold_s": s["hold_s"],
                    "refusal": s["refusal"],
                    **(r.detail() if hasattr(r, "detail") else {"error": None, "refused": None})}
            return {"dynamic": self.enabled, "order": [r.name for r in self.roles],
                    "reload_order": [r.name for r in reversed(self.roles)],
                    "policy": {"floor_gib": round(self.floor / GIB, 2),
                               "margin_gib": round(self.margin / GIB, 2),
                               "hold_s": self.hold_s, "thrash_s": self.thrash_s,
                               "max_hold_s": self.max_hold_s},
                    "mem_available_gib": None if avail is None else round(avail / GIB, 2),
                    "headroom_gib": None if head is None else round(head / GIB, 2),
                    "quiet_s": (round(now - self.last_pressure, 1)
                                if self.last_pressure is not None else None),
                    "roles": roles, "stats": dict(self.stats),
                    "events": list(self.events)[-12:][::-1]}


def settings() -> dict:
    """The knobs (serve.env), read once at the start."""
    return {"enabled": _env_on("QSE_ROLES_DYNAMIC", "on"),
            "laya": _env_on("QSE_ROLE_UNLOAD_LAYA", "off"),
            "margin_gib": _env_float("QSE_ROLE_RELOAD_MARGIN_GB", 4.0),
            "hold_s": _env_float("QSE_ROLE_RELOAD_S", 60.0),
            "thrash_s": _env_float("QSE_ROLE_THRASH_S", 600.0),
            "max_hold_s": _env_float("QSE_ROLE_MAX_HOLD_S", 3600.0),
            "tick_s": _env_float("QSE_ROLE_TICK_S", 5.0)}


def build(state: dict, *, floor_gib: float, meminfo=None, headroom=None, idle=None, empty=None,
          journal=None, log=print, cfg: dict | None = None) -> RoleMemory | None:
    """The governor over the roles this server runs (None without any). The shadow and the
    narrator share one slot when they name the same model: then that slot is one role
    (`narrator`)."""
    cfg = cfg or settings()
    roles = []
    narr, svc, reg = state.get("narrator"), state.get("shadow"), state.get("deciders")
    nslot = getattr(narr, "slot", None) if narr is not None else None
    if svc is not None and getattr(svc, "slot", None) is not None and svc.slot is not nslot:
        roles.append(SlotRole("shadow", svc.slot,
                              release=lambda why, _s=svc: _s.release(why, block=False)))
    if nslot is not None:
        rel = None
        if svc is not None and getattr(svc, "slot", None) is nslot:
            rel = lambda why, _s=svc: _s.release(why, block=False)      # noqa: E731
        roles.append(SlotRole("narrator", nslot, release=rel))
    if reg is not None and cfg.get("laya"):
        roles.append(RegistryRole("laya", reg, claim_gib=lambda: _decider_claim(state)))
    if not roles:
        return None
    return RoleMemory(roles, floor_gib=floor_gib, margin_gib=cfg["margin_gib"],
                      hold_s=cfg["hold_s"], thrash_s=cfg["thrash_s"],
                      max_hold_s=cfg["max_hold_s"], tick_s=cfg["tick_s"], meminfo=meminfo,
                      headroom=headroom, idle=idle, empty=empty, journal=journal, log=log,
                      enabled=cfg["enabled"])


def _decider_claim(state: dict) -> float:
    try:
        cond = state.get("conductor")
        from server.conductor import residency, schema
        doc = cond.store.peek("residency")[1] if cond is not None else schema.seed("residency")
        roles = cond.store.peek("roles")[1] if cond is not None else schema.seed("roles")
        c = residency.claims(doc, roles).get("decider")
        return float(c["gib"]) if c else 3.0
    except Exception:                                              # noqa: BLE001
        return 3.0
