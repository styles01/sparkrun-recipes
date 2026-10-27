"""Parallel requests, phase A: sequence slots, a fair GPU turn, lockstep verify rounds, and
admission by bytes.

`--parallel N` (`QSE_PARALLEL`, off by default) builds N - 1 side slots beside the engine. Off,
nothing here is built and the server is the single-sequence server it always was.

  * A `Slot` is one sequence's complete state: an engine (KV, recurrent state, verify graphs) and a
    drafter stack over the ONE copy of the weights every slot shares. Slot 0 is the server's own
    engine behind `LOCK`, with the resident prefix, the disk tier, the spill and the side lane; it
    reads and writes `STATE` as it always did. A side slot has a shorter window (`--side-max-len`),
    no resident prefix, and its own values of the per-request keys (`OVERLAY`): a request bound to
    it sees them through `STATE` (server/app.py `SlotState`), so the 6,000-line handler runs on it
    unchanged.
  * `Turn` decides which slot may run GPU work. A request holds it while it runs and offers it back
    between two blocks and between two prefill chunks; it hands it over only to a waiting slot that
    has used a quantum less GPU time than it has (virtual time, the idea of Linux's CFS). Alone, a
    request never hands over.
  * `Meeting` is the round's barrier: while two or more slots decode, each submits its tree block
    and waits, and one pass verifies them all (engine/model.py `forward_tree_multi`), the weights
    read once. A slot alone never submits and keeps its graphed `forward_tree`.
  * `Admission` counts bytes: before a request on a side slot prefills, its worst case (the
    prefill's workspace plus one state-store entry) is added to what the requests in flight have
    reserved, and the request waits while MemAvailable minus all of that would fall under the
    floor. An out-of-memory on this board freezes the box, so it is never caught, only avoided.

Everything here is plain Python and CPU-testable (tests/test_slots.py).
"""

from __future__ import annotations

import os
import threading
import time

GiB = 1 << 30
MiB = 1 << 20

#: SRV-311: a round's waiters time out while the turn is held by a slot that is not decoding
#: (`barrier`); "off" waits for a free turn as before (kept for the A/B)
IDLE_OWNER = (os.environ.get("QSE_BARRIER_IDLE_OWNER") or "on").strip().lower() != "off"

#: the per-request keys a side slot keeps for itself (server/app.py `SlotState`): what the
#: generation publishes for the handler (`last_ctx`, `blocks`, ...) and the slot-0-only machinery,
#: which a side slot does not have (None)
OVERLAY = ("engine", "drafter", "max_len", "resident", "disk_cache", "spill", "lane",
           "end_align", "mem", "probe", "last_ctx", "last_prompt_n", "hidden_spans", "blocks",
           "last_prefill", "spill_restored", "side_carried")


# ------------------------------------------------------------------ memory
def read_meminfo(path: str = "/proc/meminfo") -> dict | None:
    """`/proc/meminfo` in BYTES, or None where there is none (a Mac, a test)."""
    try:
        out = {}
        with open(path) as f:
            for line in f:
                k, _, v = line.partition(":")
                parts = v.split()
                if parts:
                    out[k] = int(parts[0]) * (1024 if len(parts) > 1 and parts[1] == "kB" else 1)
        return out
    except OSError:
        return None


def slot_bytes(max_len: int, *, per_token: int = 106_752, constant: int = 7 * GiB + GiB // 2) -> int:
    """What one side slot costs before it serves anything: 64 KiB of engine KV, 2 x 20 KiB of
    draft KV (two drafter arms) and 256 B of rotary table a token of window, plus a constant for
    the recurrent state, the verify scratch and fold buffers, the graph pools and the draft
    buffers: 6.9 to 7.5 GiB allocated measured on 2026-10-05 for an 8,192-token slot (MemAvailable
    fell 5.9 to 9.4 GiB a slot), most of it the pools of the 121 verify graphs a slot captures."""
    return int(max_len) * int(per_token) + int(constant)


class Admission:
    """Byte-counting admission for parallel requests.

    `reserve(key, need)` admits a request whose worst case is `need` bytes when

        MemAvailable - cache growth left - what the requests in flight reserved - need >= floor

    and records the reservation until `release(key)`. A reservation counts in full until it is
    released, also the part that has already been allocated (and so already left MemAvailable):
    the check is conservative by up to one request's worst case, never optimistic. `growth()` is
    what the long-lived caches may still take (their budgets minus what they hold). Without
    /proc/meminfo (a Mac, a test without `meminfo`) every request is admitted and says so."""

    def __init__(self, floor_bytes: int, *, meminfo=read_meminfo, growth=None, room=None,
                 committed=None):
        self.floor = int(floor_bytes)
        self.meminfo = meminfo
        self.growth = growth or (lambda: 0)
        # SRV-250: the most the engine may still take under its memory policy (its room to
        # `--mem-peak-gib`), None when unknown. Every prefill, a side slot's too, passes the
        # policy's gate, which keeps the reservation plus that prefill's workspace under the
        # peak, so what the caches, the requests in flight and this one may take together is at
        # most this room
        self.room = room
        # SRV-307: (owner) -> bytes the KV pool has committed for that slot. A request's KV part
        # is the pages it may still commit; what its slot committed since it was admitted has
        # already left MemAvailable and is credited, so it is never counted twice
        self.committed = committed
        self.lock = threading.Lock()
        self._admit = threading.Lock()
        self.held: dict = {}
        self.kv: dict = {}                 # key -> (KV bytes of `held`, owner, committed at admit)
        self.waiting: dict = {}            # SRV-305: key -> arrival time, while it waits for memory
        self.last_refused: dict | None = None   # 0.8.2: the last check that said no (`check`'s d)
        self.stats = {"checks": 0, "admitted": 0, "waits": 0, "refused": 0, "min_after": None,
                      "timeouts": 0}

    def _kv_left(self, key) -> int:
        kv = self.kv.get(key)
        if kv is None:
            return 0
        need, owner, base = kv
        if self.committed is None or owner is None:
            return need
        try:
            done = max(0, int(self.committed(owner)) - int(base))
        except Exception:                                          # noqa: BLE001
            done = 0
        return max(0, need - done)

    def reserved(self) -> int:
        with self.lock:
            return sum(self.held.values())

    def _held(self) -> tuple[int, int]:
        """(the workspace part, the KV part still to commit) of the requests in flight."""
        with self.lock:
            keys = list(self.held)
            ws = sum(self.held[k] - self.kv.get(k, (0,))[0] for k in keys)
            kv = sum(self._kv_left(k) for k in keys)
        return ws, kv

    def check(self, need: int, count: bool = True, kv: int = 0) -> tuple[bool, dict]:
        mi = self.meminfo() if self.meminfo is not None else None
        grow = int(self.growth())
        held_ws, held_kv = self._held()
        need_ws = max(0, int(need) - int(kv))
        d = {"floor": self.floor, "grow_left": grow, "reserved": held_ws + held_kv,
             "need": int(need), "kv": int(kv), "kv_reserved": held_kv}
        if not mi or "MemAvailable" not in mi or self.floor <= 0:
            d.update(avail=None, after=None, ok=True)
            return True, d
        avail = int(mi["MemAvailable"])
        take = grow + held_ws + need_ws
        room = None
        if self.room is not None:
            try:
                room = max(0, int(self.room()))
            except Exception:                                      # noqa: BLE001
                room = None
        if room is not None and room < take:
            take = room
        # the KV pages are outside the engine's memory policy (engine/kvpool.py): never capped by
        # its room, always counted
        take += held_kv + int(kv)
        after = avail - take
        d.update(avail=avail, after=after, ok=after >= self.floor, room=room)
        if not d["ok"]:
            self.last_refused = d
        if count:
            self.stats["checks"] += 1
            m = self.stats["min_after"]
            self.stats["min_after"] = after if m is None else min(m, after)
        return d["ok"], d

    def reserve(self, key, need: int, kv: int = 0, owner=None) -> tuple[bool, dict]:
        """Check and record in one step, so two requests cannot both take the last bytes. A
        request with nothing else in flight is admitted without the check: alone, it is the
        single-sequence server's request, which the engine's own memory policy (engine/cache.py
        `Budget`) and the KV pool's gate (engine/kvpool.py) admit as they always did.

        `kv`: the part of `need` that is KV pages the request's slot (`owner`) may commit."""
        with self._admit:
            with self.lock:
                alone = not self.held
            base = 0
            if self.committed is not None and owner is not None:
                try:
                    base = int(self.committed(owner))
                except Exception:                                  # noqa: BLE001
                    base = 0
            if alone:
                with self.lock:
                    self.held[key] = int(need)
                    self.kv[key] = (int(kv), owner, base)
                    self.waiting.pop(key, None)
                self.stats["admitted"] += 1
                self.stats["alone"] = self.stats.get("alone", 0) + 1
                return True, {"alone": True, "need": int(need), "kv": int(kv), "ok": True,
                              "avail": None}
            ok, d = self.check(need, kv=kv)
            if ok:
                with self.lock:
                    self.held[key] = int(need)
                    self.kv[key] = (int(kv), owner, base)
                    self.waiting.pop(key, None)
                self.stats["admitted"] += 1
        return ok, d

    def wait(self, key) -> int:
        """SRV-305: `key` waits for memory; returns its place in line (1 = next)."""
        with self.lock:
            self.waiting.setdefault(key, time.monotonic())
            t = self.waiting[key]
            return 1 + sum(1 for v in self.waiting.values() if v < t)

    def unwait(self, key) -> None:
        with self.lock:
            self.waiting.pop(key, None)

    def release(self, key) -> None:
        with self.lock:
            self.held.pop(key, None)
            self.kv.pop(key, None)
            self.waiting.pop(key, None)

    @staticmethod
    def describe(d: dict) -> str:
        if d.get("alone"):
            return "admitted alone (nothing else in flight; the engine's memory policy decides)"
        if d.get("avail") is None:
            return "no /proc/meminfo: not checked"
        g = lambda b: f"{b / GiB:.1f}"                                    # noqa: E731
        room = d.get("room")
        capped = room is not None and room < d["grow_left"] + d["reserved"] + d["need"]
        return (f"MemAvailable {g(d['avail'])} GiB - "
                + (f"the engine's room to its peak {g(room)} (cache growth {g(d['grow_left'])}, "
                   f"in flight {g(d['reserved'])}, this request {g(d['need'])})" if capped else
                   f"cache growth {g(d['grow_left'])} - in flight {g(d['reserved'])} - "
                   f"this request {g(d['need'])}")
                + f" = {g(d['after'])} GiB ({'>=' if d['ok'] else '<'} floor {g(d['floor'])})"
                + (f"; KV pages: this request {g(d['kv'])}, in flight {g(d['kv_reserved'])}"
                   if d.get("kv") or d.get("kv_reserved") else ""))

    def report(self) -> dict:
        with self.lock:
            held = dict(self.held)
            waiting = len(self.waiting)
        ws, kv = self._held()
        return {"floor_gib": round(self.floor / GiB, 2), "in_flight": len(held),
                "reserved_gib": round(sum(held.values()) / GiB, 3),
                "kv_to_commit_gib": round(kv / GiB, 3), "waiting_memory": waiting, **self.stats}


# ------------------------------------------------------------------ the turn
class Turn:
    """A fair hand-over of the GPU between slots, by virtual time.

    Every slot has a virtual time: the GPU time it has used. `pause(who)` is called by the holder
    before each unit of work (a decode block, a prefill chunk); it charges the time since the last
    pause, and when a waiting slot is behind the holder by more than one quantum, it hands the turn
    to the waiter furthest behind and queues. A 1.5 s prefill chunk puts its slot 1.5 s ahead, and a
    decoding slot runs about 1.5 s of blocks before the prefill runs again. A slot that starts a
    request (`reset`) starts at the smallest virtual time of the slots running, so a slot that was
    idle has no credit to spend. Alone, a holder never hands over.

    `settle()` runs in the holder's thread before it hands the turn to another slot (the server
    passes `torch.cuda.synchronize`): the next slot starts on an idle device, so its own timings --
    the length router prices widths from measured draft and verify times -- do not include the last
    slot's queued work.
    """

    def __init__(self, quantum_s: float = 0.05, clock=time.monotonic, settle=None):
        self.quantum = float(quantum_s)
        self.clock = clock
        self.settle = settle
        self._cv = threading.Condition()
        self._owner = None
        self._grant = None                 # the waiter the turn was handed to
        self._waiting: list = []           # arrival order
        self._v: dict = {}                 # virtual time, seconds of GPU used
        self._mark = 0.0
        self.stats = {"acquires": 0, "handoffs": 0, "waits": 0}

    @property
    def owner(self):
        return self._owner

    def waiting(self) -> int:
        return len(self._waiting)

    def held_by(self, who) -> bool:
        return self._owner == who

    def _floor(self, but=None) -> float:
        active = [self._v[w] for w in [self._owner, *self._waiting]
                  if w is not None and w != but and w in self._v]
        return min(active) if active else max(self._v.values(), default=0.0)

    def reset(self, who) -> None:
        with self._cv:
            self._v[who] = self._floor(but=who)

    def _charge(self, now: float) -> None:
        if self._owner is not None:
            self._v[self._owner] = self._v.get(self._owner, 0.0) + max(0.0, now - self._mark)
        self._mark = now

    def _behind(self):
        if not self._waiting:
            return None
        return min(self._waiting, key=lambda w: (self._v.get(w, 0.0), self._waiting.index(w)))

    def _wait_for_turn(self, who) -> None:
        self._waiting.append(who)
        try:
            while not (self._owner is None and (self._grant == who or (
                    self._grant is None and self._behind() == who))):
                self._cv.wait()
        finally:
            self._waiting.remove(who)
        self._grant = None
        self._owner = who
        self._mark = self.clock()
        self.stats["acquires"] += 1

    def acquire(self, who) -> None:
        with self._cv:
            if who not in self._v:
                self._v[who] = self._floor(but=who)
            if self._owner is None and not self._waiting and self._grant is None:
                self._owner = who
                self._mark = self.clock()
                self.stats["acquires"] += 1
                return
            self.stats["waits"] += 1
            self._wait_for_turn(who)

    def release(self, who) -> None:
        if self.settle is not None and self._waiting:
            self.settle()
        with self._cv:
            if self._owner != who:
                raise RuntimeError(f"turn released by {who!r}, held by {self._owner!r}")
            self._charge(self.clock())
            self._owner = None
            self._grant = self._behind()
            self._cv.notify_all()

    def pause(self, who) -> bool:
        """Offer the turn back before a unit of work; True if it was handed over (and is back)."""
        with self._cv:
            if self._owner != who:
                return False
            self._charge(self.clock())
            nxt = self._behind()
            if nxt is None or self._v.get(nxt, 0.0) + self.quantum >= self._v[who]:
                return False
        if self.settle is not None:
            self.settle()
        with self._cv:
            self._charge(self.clock())
            nxt = self._behind()
            if nxt is None:
                return False
            self.stats["handoffs"] += 1
            self._owner = None
            self._grant = nxt
            self._cv.notify_all()
            self._wait_for_turn(who)
            return True

    def report(self) -> dict:
        with self._cv:
            return {"quantum_ms": self.quantum * 1e3, "owner": self._owner,
                    "waiting": list(self._waiting),
                    "vtime_s": {str(k): round(v, 3) for k, v in self._v.items()}, **self.stats}


# ------------------------------------------------------------------ slots
class Slot:
    """One sequence. `values` holds a side slot's own values of the `OVERLAY` keys; slot 0 has
    none (it is `STATE`). `phase` is "decode" while its request is in the block loop (the barrier
    waits only for those slots), else ""."""

    def __init__(self, index: int, max_len: int, values: dict | None = None, lock=None):
        self.index = int(index)
        self.max_len = int(max_len)
        self.values = values
        self.lock = lock if lock is not None else threading.Lock()
        self.owner = None
        self.served = 0
        self.phase = ""
        self.t_start = None
        # FX10 (SRV-213): `gen` counts the grants; `carry` is the end of the last request this
        # side slot served (server/app.py `_side_keep_end`), usable while `gen` is unchanged: no
        # other request has written the slot's buffers since
        self.gen = 0
        self.carry: dict | None = None

    @property
    def busy(self) -> bool:
        return self.lock.locked()

    def report(self) -> dict:
        eng = (self.values or {}).get("engine")
        return {"index": self.index, "max_len": self.max_len, "busy": self.busy,
                "owner": self.owner, "served": self.served, "phase": self.phase,
                "kv_length": getattr(getattr(eng, "kv", None), "length", None),
                # FX10: the conversation end a next turn on slot 0 can carry over (tokens), or None
                "carry": (int(self.carry["at"]) if self.carry is not None
                          and self.carry.get("gen") == self.gen else None),
                # a side slot's own verify graphs (`--side-graphs`); None for slot 0 (always on)
                "graphs": (None if eng is None else not getattr(eng, "graphs_off", False))}


class VerifyReq:
    """One slot's tree block, submitted to the round. The slot that runs the round fills `out`
    and `ms`; the submitter reads them when it holds the turn again."""

    __slots__ = ("slot", "eng", "tokens", "parents", "start", "rows", "batchable", "out", "ms",
                 "error", "joint", "tree", "tree_cut")

    def __init__(self, slot: int, eng, tokens, parents, start: int, batchable: bool = True,
                 tree=None):
        self.slot, self.eng, self.tokens, self.parents, self.start = slot, eng, tokens, parents, start
        self.tree = tree                   # the slot's whole tree, for a joint cut (`joint_cut`)
        self.tree_cut = None               # the tree the round verified, when it cut it
        self.rows = len(parents)
        self.batchable = bool(batchable)
        self.out = None
        self.ms = None
        self.error: BaseException | None = None
        self.joint = 1                     # how many blocks shared the pass this one ran in


class DraftReq:
    """One slot's draft call, submitted to the round's packed draft (engine/drafters/draft_graph.py
    `packed_draft`): the drafter, the anchor and the position. The slot that runs the round fills
    `out` (the draft graph's outputs: cand, scores[, logp])."""

    __slots__ = ("slot", "drafter", "anchor", "pos0", "rows", "batchable", "out", "ms", "error",
                 "joint")

    def __init__(self, slot: int, drafter, anchor: int, pos0: int, rows: int):
        self.slot, self.drafter, self.anchor, self.pos0, self.rows = slot, drafter, anchor, pos0, rows
        self.batchable = True
        self.out = None
        self.ms = None
        self.error: BaseException | None = None
        self.joint = 1


#: Tree nodes a slot may verify in a joint round, by how many slots decode: the joint pass stays at
#: or under the 32 rows where every projection kernel is row-invariant (docs/kernels.md). A slot
#: alone keeps the router's own budget.
SHARES = {2: 16, 3: 10, 4: 8}


def joint_cut(trees: list, max_rows: int) -> list[set]:
    """One cut over several slots' trees: each slot keeps its anchor and its best node, then nodes
    are admitted best-first by score across all the trees (each with its ancestors, so a kept set
    stays ancestor-closed) while the total stays at or under `max_rows`. A slot whose text the
    drafters read well (a copy) keeps more of its tree than one whose candidates are weak. Returns
    each tree's kept node indices."""
    keep = [{0} for _ in trees]
    used = len(trees)

    def path(g, i):
        out, j = [], i
        while j not in keep[g]:
            out.append(j)
            j = trees[g].parents[j]
        return out

    for g, t in enumerate(trees):
        if len(t.tokens) > 1 and used < max_rows:
            best = max(range(1, len(t.tokens)), key=lambda i: t.scores[i])
            p = path(g, best)
            if used + len(p) <= max_rows:
                keep[g].update(p)
                used += len(p)
    order = sorted(((-t.scores[i], g, i) for g, t in enumerate(trees) for i in range(1, len(t.tokens))),
                   key=lambda x: (x[0], x[1], x[2]))
    for _, g, i in order:
        if used >= max_rows:
            break
        if i in keep[g]:
            continue
        p = path(g, i)
        if used + len(p) <= max_rows:
            keep[g].update(p)
            used += len(p)
    return keep


def share(n_decoding: int, max_rows: int = 32) -> int | None:
    if n_decoding <= 1:
        return None
    s = SHARES.get(n_decoding, max(2, max_rows // n_decoding))
    return max(2, min(s, max_rows // n_decoding))


class Meeting:
    """The round's barrier. A decoding slot submits its verify block and waits; the round runs when
    every decoding slot has submitted -- run by the last one, which holds the GPU turn -- or, by a
    waiter, when the turn has been idle for `timeout_s` with blocks pending (a slot that stopped
    decoding for a moment: a forced close, a single-token step). After the first joint round the
    slots run in lockstep, because they all get their logits from the same pass. `cv` guards
    `pending` and the results."""

    def __init__(self, max_rows: int = 32, timeout_s: float = 0.002, cut: str = "share"):
        self.max_rows = int(max_rows)
        self.timeout = float(timeout_s)
        # "share": each slot cuts its own tree to its share before it submits; "joint": the slots
        # submit their whole trees and the round cuts them together (`joint_cut`)
        self.cut = cut
        self.cv = threading.Condition()
        self.pending: dict[int, VerifyReq] = {}
        # slots whose round ran and which have not taken the turn back yet: on their way (a woken
        # thread waits for the GIL up to its switch interval), not idle
        self.transit: set = set()
        self.stats = {"rounds": 0, "joint_rounds": 0, "joint_blocks": 0, "submitted": 0,
                      "timeouts": 0, "single_rounds": 0, "alone": 0, "trimmed": 0,
                      "rows_hist": {}, "joint_hist": {},
                      # a slot's round, submit to submit, by how many slots decoded: {n: [ms, rounds]}
                      "round_ms": {},
                      # with `timing` on: a joint pass's GPU time (synchronized), {blocks: [ms, passes]}
                      "pass_ms": {}}
        self.timing = False
        self._last_submit: dict = {}

    def batches(self) -> list:
        """The pending blocks cut into passes: the batchable ones together while they fit in
        `max_rows` rows, every other block in a pass of its own, in arrival order."""
        out, cur, rows = [], [], 0
        if self.cut == "joint":
            # whole trees: one pass for every batchable block, cut together to `max_rows`
            joint = [r for r in self.pending.values() if r.batchable]
            alone = [[r] for r in self.pending.values() if not r.batchable]
            return ([joint] if joint else []) + alone
        for r in self.pending.values():
            if not r.batchable:
                out.append([r])
                continue
            if cur and rows + r.rows > self.max_rows:
                out.append(cur)
                cur, rows = [], 0
            cur.append(r)
            rows += r.rows
        if cur:
            out.append(cur)
        return out

    def report(self) -> dict:
        with self.cv:
            return {"max_rows": self.max_rows, "timeout_ms": self.timeout * 1e3,
                    "pending": sorted(self.pending), **self.stats}


class Pool:
    """The slots: slot 0 (the server's own engine, behind `LOCK`) and the side slots. Requests
    queue for slot 0 the way they always did (server/app.py `_take`); a request that may use a side
    slot also tries `take_side` while it waits."""

    def __init__(self, slots: list[Slot], turn: Turn | None = None,
                 meeting: Meeting | None = None, admission: Admission | None = None,
                 draft_meeting: Meeting | None = None):
        if not slots:
            raise ValueError("a pool needs at least slot 0")
        self.slots = list(slots)
        self.turn = turn
        self.meeting = meeting
        self.draft_meeting = draft_meeting
        self.admission = admission
        self._lk = threading.Lock()
        # `peak_running`: the most slots busy at once so far. /health showed it as `max_running`
        # until 0.8.2, where it read as a limit ("max_running 2" on three slots, 2026-10-13)
        self.stats = {"side_grants": 0, "slot0_grants": 0, "peak_running": 0}
        self.last_joint: dict = {}             # slot -> blocks in its last joint pass
        # 0.8.2: the slots `--parallel` asked for, and why fewer were built (the memory floor at
        # load, `_build_side_slots`); None when all of them were
        self.requested = len(self.slots)
        self.limit_reason: str | None = None

    @property
    def n(self) -> int:
        return len(self.slots)

    def side(self) -> list[Slot]:
        return self.slots[1:]

    def take_side(self, need: int, owner=None, keep_carry_s: float = 0.0) -> Slot | None:
        """A free side slot whose window holds `need` rows, the smallest that does; None.

        `keep_carry_s` (FX10, SRV-213): skip a slot whose last request's end (`carry`) is younger
        than this, so a short unrelated request (a guest of slot 0: the router's question before a
        new user message, a title task) does not write over the rows that conversation's next
        turn carries into slot 0; the caller then serves it on slot 0 with the stash when slot 0
        is free, else takes the slot after all."""
        now = time.monotonic()
        for s in sorted(self.side(), key=lambda s: s.max_len):
            c = s.carry
            # FX16: a record backed by the slot's resident prefix outlives a guest (the stash
            # puts the rows back), so its freshness is its age alone
            if (keep_carry_s > 0 and c is not None
                    and (c.get("gen") == s.gen or c.get("resident"))
                    and now - float(c.get("t") or 0.0) < keep_carry_s):
                continue
            if need <= s.max_len and not s.busy and s.lock.acquire(blocking=False):
                s.gen += 1
                s.owner = owner
                s.t_start = time.monotonic()
                with self._lk:
                    self.stats["side_grants"] += 1
                    self.stats["peak_running"] = max(self.stats["peak_running"], self.running())
                return s
        return None

    def take_slot(self, index: int, need: int, owner=None) -> Slot | None:
        """FX16: one named side slot, without waiting: the slot whose resident rows the request's
        conversation continues (server/app.py `_side_home`). None while another request has it,
        or when its window does not hold `need` rows."""
        for s in self.side():
            if s.index != int(index):
                continue
            if need <= s.max_len and not s.busy and s.lock.acquire(blocking=False):
                s.gen += 1
                s.owner = owner
                s.t_start = time.monotonic()
                with self._lk:
                    self.stats["side_grants"] += 1
                    self.stats["peak_running"] = max(self.stats["peak_running"], self.running())
                return s
            return None
        return None

    def took_slot0(self) -> None:
        """A request of the pool's took slot 0 (server/app.py, once it holds `LOCK`)."""
        with self._lk:
            self.stats["slot0_grants"] += 1
            self.stats["peak_running"] = max(self.stats["peak_running"], self.running())

    def limit(self) -> tuple[int, str | None]:
        """0.8.2: how many requests may run at once now, and why that is fewer than the slots
        `--parallel` asked for (None when it is not): every slot that was built, unless the
        memory floor kept slots from being built at load, or the admission holds requests for
        memory now (then the ones running are all that may)."""
        n, why = self.n, self.limit_reason
        adm = self.admission
        if adm is not None:
            with adm.lock:
                held = len(adm.waiting)
            if held:
                d = adm.last_refused
                n = max(1, self.running())
                why = (f"memory: {held} request(s) wait for memory"
                       + (f" ({Admission.describe(d)})" if d and d.get("avail") is not None
                          else ""))
        return n, why

    def release_side(self, slot: Slot) -> None:
        slot.owner = None
        slot.served += 1
        slot.phase = ""
        slot.t_start = None
        slot.lock.release()

    def running(self) -> int:
        return sum(1 for s in self.slots if s.busy)

    def decoding(self) -> list[int]:
        return [s.index for s in self.slots if s.phase == "decode"]

    def report(self) -> dict:
        limit, why = self.limit()
        return {"n": self.n, "running": self.running(), "decoding": self.decoding(),
                "requested": self.requested, "max_running": limit, "max_running_reason": why,
                "slots": [s.report() for s in self.slots], **self.stats,
                "turn": self.turn.report() if self.turn is not None else None,
                "meeting": self.meeting.report() if self.meeting is not None else None,
                "draft_meeting": (self.draft_meeting.report() if self.draft_meeting is not None
                                  else None),
                "admission": self.admission.report() if self.admission is not None else None}


# ------------------------------------------------------------------ the barrier
def barrier(meet: Meeting, pool: Pool, turn: Turn, slot_index: int, req: VerifyReq, run) -> None:
    """Submit `req` to `meet` and return once its round has run.

    The slot whose submission completes the set of decoding slots runs the round (`run(meet)`,
    holding the GPU turn); the others release the turn and wait. A waiter runs the round itself
    when the set becomes complete because a slot left decode, or when the turn has been idle for
    the barrier's timeout with blocks pending and no missing slot on its way back to the turn."""
    with meet.cv:
        meet.pending[slot_index] = req
        meet.stats["submitted"] += 1
        dec = pool.decoding()
        now = time.monotonic()
        last = meet._last_submit.get(slot_index)
        meet._last_submit[slot_index] = now
        if last is not None and now - last < 5.0:
            r = meet.stats["round_ms"].setdefault(str(len(dec)), [0.0, 0])
            r[0] += (now - last) * 1e3
            r[1] += 1
        complete = all(i in meet.pending for i in dec)
    if not complete:
        turn.release(slot_index)
        idle_since = None
        try:
            with meet.cv:
                while req.out is None and req.error is None:
                    meet.cv.wait(meet.timeout / 2)
                    if req.out is not None or req.error is not None:
                        break
                    if all(i in meet.pending for i in pool.decoding()):
                        break                           # a slot left decode: the set is complete
                    now = time.monotonic()
                    dec = pool.decoding()
                    missing = set(dec) - set(meet.pending)
                    # SRV-311: a slot that holds the turn without decoding (a prefill, slot 0's own
                    # job) never submits to this round, so for the round the turn is as idle as
                    # free. Waiting for it to be free starved the decoders: two meetings (the
                    # packed draft and the joint verify) each held some decoding slots, neither set
                    # completed, and the timeout below never ran while an 87k prompt prefilled
                    # (3 chats at 0-4 tok/s for ~90 s, 2026-10-10).
                    owner = turn.owner
                    idle_turn = owner is None or (IDLE_OWNER and owner not in dec)
                    if idle_turn and not turn.waiting() and not (missing & meet.transit):
                        idle_since = idle_since or now
                        if now - idle_since >= meet.timeout:
                            meet.stats["timeouts"] += 1
                            break
                    else:
                        idle_since = None
        finally:
            turn.acquire(slot_index)
    if req.out is None and req.error is None:
        run(meet)
    with meet.cv:
        meet.transit.discard(slot_index)
    if req.error is not None:
        raise req.error


# ------------------------------------------------------------------ the request's slot
_CUR = threading.local()


def bind(slot: Slot | None) -> None:
    """The slot the current thread's request runs on (None: slot 0 / outside a request)."""
    _CUR.slot = slot


def current() -> Slot | None:
    return getattr(_CUR, "slot", None)


def values() -> dict | None:
    """The current thread's side-slot values, None on slot 0 (the fast path of `SlotState`)."""
    s = getattr(_CUR, "slot", None)
    return s.values if s is not None else None
