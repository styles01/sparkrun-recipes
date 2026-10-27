"""Serving-time caches. Four of them, and not one may change a token.

This engine holds ONE sequence, and every request so far has paid for its whole prompt from
scratch: `eng.reset()`, then a forward over every prompt token. On a multi-turn chat that is the
same two thousand tokens re-read on every turn, and on a fleet of requests sharing a system prompt
it is the same system prompt re-read per request. The board has ~90 GB of unified memory doing
nothing. These four caches spend it.

  1. `StateStore` -- the engine's whole state at a token boundary, kept in RAM and keyed by the
     token prefix that produced it. It serves BOTH the session cache (a conversation's state after
     its last turn) and the prefix cache (checkpoints every `chunk` tokens of a prefill), because
     those two are the same object with two different sets of boundaries.
  2. `ResponseCache` -- the tokens a (prompt, params) request produced. Greedy is deterministic, so
     the second identical request is a dictionary lookup.
  3. `PersistentSuffixStore` -- an append-only token log of what this engine has read and written,
     on the box, with a suffix array over it, handed to the lookup drafter as a corpus.
  4. `SuffixStoreSet` -- several such stores behind one `lookup`, so the private corpus dir and the
     engine's own history are both askable in the same 0.2 ms.
  5. `ResidentPrefix` -- the KV the last prefill left in the engine's own
     buffer, reused in place, with the recurrent state cloned at chunk boundaries. It is what
     serves a growing agent conversation past the store's per-entry cap.
  6. the disk tier (engine/disk_cache.py `DiskTier`) under all of them: prefixes prefilled more
     than once, and the head every opencode session starts with, on the NVMe. `prefill` asks it
     last and writes to it behind the request.

`Budget` (OPS-47) holds them all to one memory policy: a ceiling for the engine between requests,
a declared peak during a prefill, a cap on each cache, and an order in which they are given back.

WHAT MAKES A RESTORE EXACT, which is the whole design and the only interesting part.

A restored state is not *nearly* the state a cold prefill would have left, it is the same bytes: the
KV rows, the 48 recurrent states, the 48 convolution states and the drafter's own KV are cloned out
and copied back. So a request that resumes at token L and forwards tokens L.. computes exactly what
it would have computed had it forwarded 0..L first -- PROVIDED the cold run would have forwarded
0..L as its own step. That proviso is real and it is why `prefill` chunks:

  * `forward(tokens[0:n])` in one call and `forward(tokens[0:L])` then `forward(tokens[L:n])` are
    not the same arithmetic. The attention of the second form reads the first form's rows out of
    the cache rather than recomputing them, SDPA takes a different backend when it is handed a mask
    instead of `is_causal`, and the chunked delta rule blocks the sequence differently.
  * So the cold path and the warm path use the SAME chunking. With the prefix cache on, every
    prefill is a sequence of `chunk`-token forwards, warm or cold. A resume that lands ON that
    chunk grid -- which every PREFIX-cache checkpoint does, because that is where they are taken
    -- then splits the remaining tokens exactly as a cold prefill would, and the two runs are
    bit-identical by construction rather than by luck. `tests/test_cache.py` asserts that.
  * A SESSION boundary is different and the difference is not a detail. It falls wherever the
    previous turn stopped, off the grid, so the tail splits differently; and the state a turn ends
    in was written by speculative verify blocks and a rank-k rollback, not by prefill chunks. It is
    the state that really produced the previous turn, which is arguably the more faithful thing to
    continue from, but it is NOT the state a cold re-read of the conversation would compute. So the
    session cache is held to the gate the whole engine is held to -- the same argmax -- and not to
    bitwise equality. `tools/cache_gate.py` measures both.
  * Against the ONE-SHOT prefill this engine did before, chunking is a different arithmetic order,
    in the same sense that every fused kernel in `engine/model.py` is. Same gate: identical argmax.

A hash is a hint here, never an answer: every hit re-checks the stored token prefix element for
element before the state is restored. A 64-bit collision would otherwise answer a request with
another request's state, which is the one bug in this file that no test downstream would catch.

PRIVACY. Everything here lives in RAM on the box, except `PersistentSuffixStore`, which writes
token ids -- never text -- under a directory outside this repository, and the disk tier, which
writes token ids and engine state under the engine's home. Nothing in this file writes to the
repository, and nothing it holds reaches a log line.
"""

from __future__ import annotations

import gc
import json
import os
import struct
import threading
import time
from collections import OrderedDict, deque

import torch

from engine import capture_lock
from engine import kvpool as _kvp

from engine.settings import SETTINGS as _S

#: FX10: a parked conversation a new park gives up goes to the NVMe spill first (`on`); `off`
#: gives it up as before (`ResidentPrefix._spill_park`)
SPILL_PARK = (_S.get("SPILL_PARK") or "on").strip().lower()
PREFILL_WINDOW = int(_S.get("PREFILL_WINDOW") or 0)
PREFILL_WINDOW_FREE = int(_S.get("PREFILL_WINDOW_FREE") or 0)
#: SRV-309: () -> a cap on a layer-major window's rows right now, or None. The server sets it so a
#: prefill on one slot hands the GPU turn to the slots that decode at least every this many rows
#: (a window is one unit of GPU work: an 8,192-row window at 90k context held the turn ~20 s and
#: three decoding chats wrote ~1 token a second). The windows' chunks are the same chunks in the
#: same order, so the cap changes when the turn is offered, never a bit.
WINDOW_CAP = None

# 64-bit polynomial rolling hash. The multiplier is the FNV-1a prime; what matters is only that
# `h(tokens[:i])` is available for every i after one pass, so probing a candidate boundary costs
# nothing and probing thirty costs nothing thirty times.
_P = 0x100000001B3
_MASK = (1 << 64) - 1


def prefix_hashes(tokens) -> list[int]:
    """`out[i]` is the hash of `tokens[:i]`, so `out` is one longer than `tokens`."""
    h = 0xCBF29CE484222325
    out = [h]
    for t in tokens:
        h = ((h ^ (int(t) + 1)) * _P) & _MASK
        out.append(h)
    return out


def extend_hashes(hashes: list[int], tokens) -> list[int]:
    """Continue a hash list with more tokens, in place. Returns it for chaining."""
    h = hashes[-1]
    for t in tokens:
        h = ((h ^ (int(t) + 1)) * _P) & _MASK
        hashes.append(h)
    return hashes


class _PrefixHashMemo:
    """`prefix_hashes` for the prompt of each prefill, reusing the previous prompt's list over the
    head the two share: a growing conversation hashes only its new tokens (a full pass is ~20 ms
    at 190k tokens). The list returned is the one a full pass computes."""

    def __init__(self):
        self.keys: list = []
        self.hashes: list[int] = prefix_hashes([])

    def __call__(self, keys) -> list[int]:
        keys = list(keys)
        c = _common_prefix(keys, self.keys)
        h = self.hashes[:c + 1]
        extend_hashes(h, keys[c:])
        self.keys, self.hashes = keys, h
        return h


_hash_memo = _PrefixHashMemo()


# --------------------------------------------------------------------------- the model's state
# Every tier reads and writes a sequence through the engine's adapter (ENG-157, engine/models/
# adapter.py): `eng.cache_state()` -- its recurrent sections by name and its position-indexed views
# (engine/models/state.py) --, `eng.resume_at(n)` after the state was written, `eng.settle()` before
# it is read, `eng.cost()` for the bytes. A kept state is the sections' tensors in the state's order
# (for the 27B: S, conv), so the tiers hold any model's.

def state_now(eng) -> tuple:
    """The engine's live sections, in order (read every time: a buffer may have been swapped)."""
    return tuple(eng.cache_state().sections().values())


def state_clone(eng) -> tuple:
    return tuple(t.clone() for t in state_now(eng))


def state_empty(eng) -> tuple:
    return tuple(torch.empty_like(t) for t in state_now(eng))


def state_save(dst: tuple, eng) -> None:
    """The live sections into `dst` (buffers of `state_empty`'s shapes)."""
    for d, t in zip(dst, state_now(eng)):
        d.copy_(t)


def state_load(eng, src: tuple) -> None:
    """A kept state into the live sections (in place: captured graphs keep their tensors)."""
    for t, x in zip(state_now(eng), src):
        t.copy_(x)


def state_copy(dst: tuple, src: tuple) -> None:
    for d, x in zip(dst, src):
        d.copy_(x)


def model_views(eng) -> list:
    """The engine's position-indexed buffers as (tensor, axis)."""
    return [(t, ax) for _, t, ax in eng.cache_state().views()]


def drafter_resume(eng, drafter, n: int) -> None:
    """`drafter.state_resume(n)` with rows [0, n) of the engine's paged set committed first. A drafter
    may read those rows as it resumes (Flash-Next's MTP head pools its indexer keys from them), and
    before a restore has written them they may be pages the KV pool released: on rc-0.8.0 the spill's
    restore at startup read them and the GPU faulted (an illegal memory access). A restore calls the
    drafter's `state_resume` again once its rows are in, so what it built is the restored rows'."""
    if drafter is None or not hasattr(drafter, "state_resume"):
        return
    for t, _ in model_views(eng):
        _kvp.ensure_for(t, n)
    drafter.state_resume(n)


# --------------------------------------------------------------------------- the state snapshot
class StateSnapshot:
    """Everything the engine and its drafter need to be back at token `length`.

    The views are sliced to `length` before they are cloned, because a buffer is `max_len` long
    and cloning all of it would cost the same for a 100-token prefix as for an 8,000-token one. The
    recurrent sections are not sliceable -- the 27B's are a fixed [48, 1, H, Dk, Dv] fp32 tensor and
    its conv tails -- and they are the bulk of a snapshot on any prompt shorter than a few thousand
    tokens. `parts`: the engine's section bytes by kind (`CostModel.parts`).
    """

    __slots__ = ("length", "views", "state", "drafter", "_parts", "_bytes")

    def __init__(self, length: int, views: tuple, state: tuple, drafter=None, parts: dict | None = None):
        self.length = int(length)
        self.views, self.state, self.drafter = tuple(views), tuple(state), drafter
        self._parts = dict(parts or {})
        self._bytes = sum(_nbytes(x) for x in self.views + self.state) + _tree_bytes(drafter)

    @property
    def nbytes(self) -> int:
        return self._bytes

    def parts(self) -> dict:
        return {"kv": sum(_nbytes(x) for x in self.views), **self._parts, "drafter": _tree_bytes(self.drafter)}


def _nbytes(x) -> int:
    return 0 if x is None else x.numel() * x.element_size()


def _tree_bytes(obj) -> int:
    if obj is None:
        return 0
    if torch.is_tensor(obj):
        return _nbytes(obj)
    if isinstance(obj, (tuple, list)):
        return sum(_tree_bytes(x) for x in obj)
    if isinstance(obj, dict):
        return sum(_tree_bytes(x) for x in obj.values())
    return 0


#: One DFlash2 arm's draft KV, measured on the board (engine/drafters/dflash2.py `state_snapshot`).
#: The fallback for a snapshottable drafter that cannot say its own figure.
ARM_BYTES_PER_TOKEN = 20_000


def _drafter_bytes_per_token(drafter) -> int:
    """What the drafter's `state_snapshot` clones per committed position.

    Asked of the drafter rather than assumed: the served length router snapshots BOTH arms until
    the latch releases one, so a constant for one arm undercounted the served stack by
    a whole arm's 20 kB a token.
    """
    fn = getattr(drafter, "snapshot_bytes_per_token", None)
    return int(fn()) if fn is not None else ARM_BYTES_PER_TOKEN


def _snapshot_estimate(eng, length: int, drafter=None) -> int:
    """Bytes `capture` will clone at this length, WITHOUT cloning. The recurrent state is fixed;
    the views are per token (`eng.cost()`); the drafter's cache adds its own per-token figure WHEN
    it is snapshottable."""
    cost = eng.cost()
    per_token = cost.view_bytes_per_token
    if _snapshottable(drafter):
        per_token += _drafter_bytes_per_token(drafter)
    # A conservative bound, not an exact figure: state and KV get their own margins, because the
    # point is "definitely not bigger than the cap", not "exactly this".
    return 2 * cost.section_bytes + int(1.25 * per_token * length)


def mem_words(mem) -> str:
    """The memory policy's figures in words, for a log line that says it refused: allocated
    against the ceiling less the slack, all in GiB."""
    if mem is None:
        return "no policy"
    try:
        alloc, _ = mem.memory()
        return (f"allocated {alloc / (1 << 30):.2f} GiB of a {mem.ceiling / (1 << 30):.2f} GiB "
                f"ceiling less {mem.slack / (1 << 30):.2f} GiB of slack")
    except Exception:                                              # noqa: BLE001
        return "its figures unknown"


def capture(eng, drafter=None, max_bytes: int = 0) -> "StateSnapshot | None":
    """Clone the engine's state at its current length. Cheap in time, dear in bytes.

    `max_bytes > 0` refuses BEFORE cloning when the snapshot would exceed it -- the caller's
    store cap. Cloning first and declining in `put` still allocates and frees the whole snapshot
    every checkpoint, and that churn is what left the allocator holding ~31 GB of reserved
    segments after a 30k prefill on 2026-09-19; the check belongs here.
    """
    if max_bytes and _snapshot_estimate(eng, eng.kv.length, drafter) > max_bytes:
        return None
    eng.settle()                      # a chain accepted in full whose walked state is not in yet
    n = eng.kv.length
    return StateSnapshot(n, tuple(t.narrow(ax, 0, n).clone() for t, ax in model_views(eng)),
                         state_clone(eng), drafter.state_snapshot() if _snapshottable(drafter) else None,
                         eng.cost().parts)


def restore(eng, snap: StateSnapshot, drafter=None) -> None:
    """Put `snap` back. After this the engine is exactly where `capture` was called."""
    n = snap.length
    for (t, ax), x in zip(model_views(eng), snap.views):
        _kvp.ensure_for(t, n)         # SRV-307: the rows must exist before they are written
        t.narrow(ax, 0, n).copy_(x)
    state_load(eng, snap.state)
    eng.resume_at(n)
    if snap.drafter is not None and _snapshottable(drafter):
        drafter.state_restore(snap.drafter)


def _snapshottable(drafter) -> bool:
    return drafter is not None and hasattr(drafter, "state_snapshot")


def drafter_is_cacheable(drafter) -> bool:
    """Can this drafter survive a prefix that was restored rather than forwarded?

    Three answers. A drafter with `state_snapshot` carries its own cache across, exactly. A drafter
    with no `sync` -- the lookup drafters -- is rebuilt from the token ids by `prime`, which the
    server calls anyway. A drafter that wants hidden states and cannot snapshot them would be
    handed a cache with a hole in it, and a hole in a position-indexed draft cache is permanent: it
    would decline for ever and the engine would decode at one token a block. So the cache turns
    itself off rather than trade a warm prefill for a cold decode.
    """
    if drafter is None or _snapshottable(drafter):
        return True
    return not hasattr(drafter, "sync")


# --------------------------------------------------------------------------- the memory budget
GIB = 1 << 30

#: A prefill's transient (the allocator's peak minus what it held before the prefill), as the
#: engine measured it on 2026-10-01: 1.15 GiB on short prompts, 4.87 GiB at 173,855 tokens. It is
#: the attention's scores and the chunk's activations, and it grows with the context the last
#: chunk attends over, so the estimate is the larger of the short-prompt floor and a straight line
#: through zero and the long reading (7.4 GiB at 262,144). `Budget.observe` raises the slope when a
#: prefill takes more than the line said.
WORKSPACE_FLOOR = int(1.15 * GIB)
WORKSPACE_PER_TOKEN = int(4.87 * GIB / 173_855)


class MemoryRefused(RuntimeError):
    """A prefill that would take the engine over its declared peak, with every cache already
    given back. The server answers it with a 503 and a Retry-After instead of running into the
    driver's out-of-memory wedge."""

    def __init__(self, need: int, peak: int, n: int):
        self.need, self.peak, self.n = int(need), int(peak), int(n)
        super().__init__(f"a {n}-token prefill needs {need / GIB:.1f} GiB with every cache "
                         f"given back; the engine's peak is {peak / GIB:.1f} GiB")


def _torch_memory() -> tuple[int, int]:
    import torch as _t
    return int(_t.cuda.memory_allocated()), int(_t.cuda.memory_reserved())


def _torch_empty() -> None:
    import torch as _t
    _t.cuda.empty_cache()


def _torch_peak(reset: bool = False) -> int:
    import torch as _t
    if reset:
        _t.cuda.reset_peak_memory_stats()
        return 0
    return int(_t.cuda.max_memory_allocated())


class Budget:
    """One memory policy for the serving caches (OPS-47, the design decisions of 2026-10-01).

    THE NUMBERS. At load, with empty caches, the engine at a 262,144-token window holds 57 GiB.
    What grew from there to 75 and 83 GiB on 2026-10-01 was three things: the caches (14.5 GiB at
    their caps), a long prefill's workspace (4.9 GiB at 172k tokens) and the allocator's reserve,
    which kept every freed block (0.4, 2.5, 7.0 GiB over the day). So the policy has three limits:

      * `ceiling` -- what the engine holds between requests, allocated plus reserve. A cache grows
        only while `allocated + its bytes` stays under `ceiling - slack` (`admit`), and at the start
        of a request anything over it is given back (`gate`).
      * `peak` -- what the engine may hold during a prefill: the ceiling plus the workspace of the
        longest prompt the window allows. The residency planner counts this, not today's free
        memory, before a role may load beside the engine.
      * `slack` -- the reserve the allocator may keep between requests; over it, `done` hands the
        free blocks back to the driver once the request has finished.

    THE ORDER things are given back in, cheapest to lose first: `LEVELS`. The state store served 0
    hits on 2026-10-01; the guest stash's buffers with no guest's rows in them hold nothing
    (SRV-546); the anchors in the body of a prompt serve only a divergence in its middle
    (none in a night's ledger); the end anchor saves the next turn a re-read of the last answer
    (`ResidentPrefix.keep_end`); the guest stash saves one session from a short
    request, and without it the end anchor's rows are gone too; the parked session saves a
    re-read of a whole other session; the pinned head saved 253 s in a day.

    WHY IT CANNOT CHANGE A TOKEN. Every limit here decides only where a prefill starts: a store
    entry, a resident anchor, the pin, the park, or token 0, each checked against the prompt's
    own tokens before any state is restored. Giving a cache back is the cold path the engine ran
    before the cache existed. A 503 changes no answer either: it is a request that was not served.

    `probe` returns (allocated, reserved) in bytes -- the CUDA caching allocator's, less what the
    decision models hold in the same allocator (`exclude`), since they are roles with claims of
    their own and not the engine. `empty` is `torch.cuda.empty_cache`. Both are arguments so the
    tests can run the policy on a fake allocator.
    """

    # `roles` last: an in-process role whose own pool holds free segments that
    # only dropping the pool returns (the shadow reader after a long conversation)
    LEVELS = ("store", "stash_idle", "anchors", "end", "stash", "park", "pin", "roles")

    def __init__(self, ceiling_bytes: int, peak_bytes: int, slack_bytes: int, *,
                 probe=None, empty=None, exclude=None, peak_probe=None,
                 ws_floor: int = WORKSPACE_FLOOR, ws_per_token: int = WORKSPACE_PER_TOKEN,
                 ws_cap: int | None = None, on_event=None):
        self.ceiling = int(ceiling_bytes)
        self.peak = int(peak_bytes)
        self.slack = int(slack_bytes)
        # bytes the engine holds for good beyond what `ceiling` and `peak` were set for: the side
        # slots of `--parallel N` (server/app.py `_build_side_slots`). Every new value of the two
        # limits (server/memlimits.py `apply`, a residency document) is the configured figure
        # plus this, so a limit set again never takes the slots' bytes out of the caches' room.
        self.fixed = 0
        self._probe = probe or _torch_memory
        self._empty = empty or _torch_empty
        self._peak = peak_probe or _torch_peak
        self.lifetime_peak = 0             # the allocator's peak since start, across the resets
        self._mark: int | None = None
        self.exclude = exclude
        self.ws_floor = int(ws_floor)
        self.ws_per_token = int(ws_per_token)
        self.ws_cap = int(ws_cap) if ws_cap is not None else None
        self.on_event = on_event
        self.shedders: dict = {}
        self.ws_now = 0                    # the running prefill's workspace, until it is done
        self.stats = {"gates": 0, "refused": 0, "admitted": 0, "admit_refused": 0,
                      "empties": 0, "empty_ms": 0.0, "empty_freed": 0, "last_empty_ms": 0.0,
                      "shed": {lv: 0 for lv in self.LEVELS},
                      "shed_bytes": {lv: 0 for lv in self.LEVELS},
                      "observed": None, "last_gate": None, "prefill_peak": 0,
                      "last_prefill_peak": None, "stale_admits": 0}
        self.refused_in_a_row = 0
        # () -> bytes: free memory in the roles' own pools, which counts in `reserved` but which
        # the engine's allocator cannot use (the stale-reserve rule leaves it out)
        self.foreign_free = None
        self.events: deque = deque(maxlen=50)
        self.lock = threading.Lock()

    @property
    def on(self) -> bool:
        return self.ceiling > 0 and self.peak > 0

    def register(self, level: str, fn) -> None:
        """`fn(safe) -> bytes given back`. `safe=True` is a call from inside a cache's own
        operation: only what nothing is using may go (a free buffer, the store), never the rows
        or anchors of the prompt being served."""
        if level not in self.LEVELS:
            raise ValueError(f"unknown level {level!r}: {self.LEVELS}")
        self.shedders[level] = fn

    def _excluded(self) -> tuple[int, int]:
        """(allocated, reserved) bytes that are not the engine's: `exclude()` gives one figure for
        both or the pair (server/memlimits.role_exclude: a role's pool reserves more than it
        holds)."""
        if self.exclude is None:
            return 0, 0
        x = self.exclude()
        if isinstance(x, (tuple, list)):
            return int(x[0]), int(x[1])
        return int(x), int(x)

    def memory(self) -> tuple[int, int]:
        alloc, res = self._probe()
        xa, xr = self._excluded()
        return max(0, alloc - xa), max(0, res - xr)

    def workspace(self, n: int) -> int:
        w = max(self.ws_floor, self.ws_per_token * int(n))
        # a model whose workspace stops growing (its adapter's `workspace_cap`; the 27B's: none)
        return min(w, self.ws_cap) if self.ws_cap is not None else w

    def _event(self, kind: str, **kw) -> dict:
        ev = dict(kind=kind, ts=round(time.time(), 3), **kw)
        self.events.append(ev)
        if self.on_event is not None:
            try:
                self.on_event(ev)
            except Exception:                                      # noqa: BLE001
                pass
        return ev

    def shed(self, level: str, why: str, safe: bool = False) -> int:
        fn = self.shedders.get(level)
        if fn is None:
            return 0
        freed = int(fn(safe) or 0)
        if freed:
            self.stats["shed"][level] += 1
            self.stats["shed_bytes"][level] += freed
            self._event("shed", level=level, why=why, freed=freed, safe=safe)
        return freed

    def empty(self, why: str) -> int:
        """Hand the allocator's free blocks back to the driver. Between requests only: the verify
        and draft graphs keep pools of their own, which this does not touch."""
        _, before = self.memory()
        t0 = time.perf_counter()
        self._empty()
        ms = (time.perf_counter() - t0) * 1e3
        _, after = self.memory()
        freed = max(0, before - after)
        self.stats["empties"] += 1
        self.stats["empty_ms"] += ms
        self.stats["last_empty_ms"] = round(ms, 2)
        self.stats["empty_freed"] += freed
        self._event("empty", why=why, freed=freed, ms=round(ms, 2))
        return freed

    # --- the request's start ---------------------------------------------------------------
    def gate(self, n: int, count: bool = True) -> dict:
        """Before a prefill of an `n`-token prompt: back under the ceiling, room for the
        workspace under the peak, or `MemoryRefused`. Called with the engine lock held and
        before the response has started, so a refusal is still an HTTP status."""
        if not self.on:
            return {}
        with self.lock:
            self.stats["gates"] += 1
            shed: list = []
            alloc, res = self.memory()
            if res > self.ceiling and res > alloc:
                self.empty("ceiling")
                alloc, res = self.memory()
            for lv in self.LEVELS:
                if res <= self.ceiling:
                    break
                if self.shed(lv, "ceiling"):
                    shed.append(lv)
                    self.empty("ceiling")
                    alloc, res = self.memory()
            ws = self.workspace(n)
            if res + ws > self.peak and res > alloc:
                self.empty("peak")
                alloc, res = self.memory()
            for lv in self.LEVELS:
                if res + ws <= self.peak:
                    break
                if self.shed(lv, "peak"):
                    shed.append(lv)
                    self.empty("peak")
                    alloc, res = self.memory()
            if res + ws > self.peak:
                # the last try before a refusal: what only a collection frees (a frame or a
                # traceback that still holds tensors), then the allocator's free blocks back
                gc.collect()
                self.empty("refuse")
                alloc, res = self.memory()
            g = {"n": int(n), "workspace": ws, "allocated": alloc, "reserved": res,
                 "shed": shed, "ts": round(time.time(), 3)}
            self.stats["last_gate"] = g
            foreign = 0
            if self.foreign_free is not None:
                try:
                    foreign = max(0, int(self.foreign_free()))
                except Exception:                                  # noqa: BLE001
                    foreign = 0
            if (res + ws > self.peak and alloc + ws <= self.peak
                    and ws <= res - alloc - foreign):
                # a stale reserve: the workspace fits in what the allocator already holds free,
                # so the prefill takes no new memory from the box. Never refuse it
                # (a 12-token prompt refused for 25 minutes on a reserve nothing could give back)
                self.stats["stale_admits"] += 1
                self._event("stale_admit", n=int(n), allocated=alloc, reserved=res, ws=ws)
                g["stale"] = True
            elif res + ws > self.peak:
                exc = MemoryRefused(res + ws, self.peak, n)
                exc.detail = dict(allocated=alloc, reserved=res)
                if count:
                    self._count_refused(exc)
                raise exc
            self.refused_in_a_row = 0
            self.ws_now = ws
            return g

    def refused(self, exc: "MemoryRefused") -> None:
        """Count a refusal of `gate`: its own, or the caller's when it ran the gate with
        `count=False` and the request is refused after all (ENG-371: a gate run once more and
        admitted is no refusal)."""
        with self.lock:
            self._count_refused(exc)

    def _count_refused(self, exc: "MemoryRefused") -> None:
        self.stats["refused"] += 1
        self.refused_in_a_row += 1
        self._event("refused", n=exc.n, need=exc.need, peak=exc.peak,
                    in_a_row=self.refused_in_a_row, **getattr(exc, "detail", {}))

    def admit(self, level: str, nbytes: int) -> bool:
        """May a cache allocate `nbytes` more? Under `ceiling - slack` allocated and, while a
        prefill is still to run, with its workspace still under the peak. When not, the levels
        below `level` give back what nothing is using, once, and the question is asked again."""
        if not self.on or nbytes <= 0:
            return True
        lower = self.LEVELS[:self.LEVELS.index(level)] if level in self.LEVELS else ()
        for attempt in range(2):
            alloc, res = self.memory()
            if (alloc + nbytes <= self.ceiling - self.slack
                    and max(res, alloc + nbytes) + self.ws_now <= self.peak):
                self.stats["admitted"] += 1
                return True
            if attempt == 0:
                if not any([self.shed(lv, f"admit:{level}", safe=True) for lv in lower]):
                    break
        self.stats["admit_refused"] += 1
        self._event("admit_refused", level=level, nbytes=int(nbytes))
        return False

    def mark(self) -> None:
        """A prefill's forwards start here: the allocator's peak counts from now."""
        try:
            self.lifetime_peak = max(self.lifetime_peak, self._peak())
            self._peak(reset=True)
            self._mark = self._probe()[0]
        except Exception:                                          # noqa: BLE001
            self._mark = None

    def lifetime(self) -> int:
        """`max_memory_allocated` since the process started, whatever `mark` reset."""
        try:
            return max(self.lifetime_peak, self._peak())
        except Exception:                                          # noqa: BLE001
            return self.lifetime_peak

    def prefilled(self, n: int) -> None:
        """The prefill has run: its workspace is free again, and what it really took corrects the
        estimate upward when it was low. The workspace is the allocator's peak over the forwards
        less what is allocated once they are done -- not less what was allocated before them:
        the anchors and the store snapshot a prefill takes on its way are still allocated at its
        end, and counting them as workspace taught the estimate 113 kB a token from a 48k prompt
        (27.6 GiB at 262k) in the 2026-10-01 gate. Only a long prefill (64k tokens and more)
        teaches the slope, where the context's part dominates the fixed one."""
        self.ws_now = 0
        workspace = None
        if self._mark is not None:
            try:
                peak = self._peak()
                self.lifetime_peak = max(self.lifetime_peak, peak)
                workspace = peak - max(self._mark, self._probe()[0])
                x = self._excluded()[0]
                self.stats["last_prefill_peak"] = {"n": int(n), "peak": peak - x}
                self.stats["prefill_peak"] = max(self.stats["prefill_peak"], peak - x)
            except Exception:                                      # noqa: BLE001
                workspace = None
        self._mark = None
        if workspace is None or n < 65_536:
            return
        self.stats["observed"] = {"n": int(n), "workspace": int(workspace),
                                  "estimate": self.workspace(n)}
        if self.ws_cap is not None and workspace > self.ws_cap:
            self._event("estimate_raised", n=int(n), workspace=int(workspace), cap=int(workspace),
                        was_cap=self.ws_cap)
            self.ws_cap = int(workspace)
        per = int(workspace / n)
        if per > self.ws_per_token:
            self._event("estimate_raised", n=int(n), workspace=int(workspace),
                        per_token=per, was=self.ws_per_token)
            self.ws_per_token = per

    def done(self) -> int:
        """After a request: the reserve over its slack goes back to the driver."""
        self.ws_now = 0
        if not self.on:
            return 0
        with self.lock:
            alloc, res = self.memory()
            if res - alloc > self.slack:
                return self.empty("idle")
        return 0

    def report(self) -> dict:
        alloc, res = self.memory()
        return {"ceiling": self.ceiling, "peak": self.peak, "slack": self.slack,
                "fixed": self.fixed, "allocated": alloc, "reserved": res, "reserve": res - alloc,
                "ws_floor": self.ws_floor, "ws_per_token": self.ws_per_token,
                "events": list(self.events)[-20:],
                **{k: (dict(v) if isinstance(v, dict) else v) for k, v in self.stats.items()}}


# --------------------------------------------------------------------------- the state store
class _Entry:
    __slots__ = ("tokens", "snap", "conv_id", "created", "hits", "kind")

    def __init__(self, tokens, snap, conv_id, kind="prefix"):
        self.tokens = tokens
        self.snap = snap
        self.conv_id = conv_id
        self.created = time.time()
        self.hits = 0
        # where the snapshot came from: the end of a turn ("session") or a prefill chunk boundary
        # ("prefix"). One store holds both and a lookup cannot tell them apart otherwise.
        self.kind = kind


class StateStore:
    """Token-prefix -> engine state, under a byte budget, least-recently-used out first.

    Keyed by `(length, hash)` and verified by the tokens themselves. `lengths` is the set of
    boundaries anything has been stored at, which is what makes a lookup cheap: the candidates for
    a new prompt are the stored lengths below it, and each one is one dictionary probe.
    """

    def __init__(self, budget_bytes: int, chunk: int = 256, max_entry_bytes: int = 0,
                 per_prefill: int = 0):
        self.budget = int(budget_bytes)
        # How many chunk boundaries of ONE prefill are stored: its longest ones that fit the
        # per-entry cap. Every boundary (0) nests: a cold prefill used to put 13 snapshots of its
        # own prefix (1,024 to 13,312 tokens, ~11 GB), which is more than the budget, so each cold
        # prefill evicted everything else -- 214 puts and 206 evictions in the 2026-09-30 night.
        self.per_prefill = max(0, int(per_prefill))
        # One snapshot may not exceed this, HOWEVER much budget is free. At a 256k window a
        # boundary-length snapshot is 154 MB + 85.5 kB x tokens -- several GB -- and `put` clones
        # it BEFORE evictions run, so the allocator sees store + clone at once. That transient
        # overshoot on a 121 GiB board (53 GB engine + 24 GB store + 40 GB page cache) is the
        # 2026-09-19 wedge: a long-prompt prefill pushed MemAvailable to zero and the GPU driver
        # locked. Default: a quarter of the budget.
        self.max_entry = int(max_entry_bytes) or 0   # 0 = no per-entry cap (the server sets one)
        self.chunk = int(chunk)
        self._d: OrderedDict = OrderedDict()
        self._lengths: dict[int, int] = {}       # length -> how many entries sit at it
        self.bytes = 0
        self.stats = {"puts": 0, "hits": 0, "misses": 0, "evictions": 0,
                      "tokens_reused": 0, "tokens_forwarded": 0, "rejected_collisions": 0,
                      "hits_session": 0, "hits_prefix": 0}
        # the kind of the entry the latest `find` restored, for the request's `cache_source`
        self.last_kind: str | None = None
        self.lock = threading.Lock()

    # --- writing ---------------------------------------------------------------------------
    def put(self, tokens, snap: StateSnapshot, conv_id: str | None = None,
            hashes: list[int] | None = None, kind: str = "prefix") -> None:
        n = snap.length
        if self.max_entry and snap.nbytes > self.max_entry:
            self.stats["declined_big"] = self.stats.get("declined_big", 0) + 1
            return
        if n == 0 or n > len(tokens):
            # The engine forwarded more tokens than the caller collected, which is every generation
            # that ended inside a block: an abandoned stream, or a token budget that ran out part
            # way through an accepted path. There is no honest snapshot to store for it. The KV
            # could be truncated but the RECURRENT state cannot -- it has already absorbed those
            # tokens and there is no inverse -- so an entry keyed by the shorter prefix would
            # restore a state that has seen text the key does not mention.
            #
            # Counted rather than silent, because a store that declines every put looks exactly
            # like a store that is switched off, and phase 9 spent an hour on the difference.
            self.stats["declined_short"] = self.stats.get("declined_short", 0) + 1
            return
        key = (n, (hashes[n] if hashes is not None else prefix_hashes(tokens[:n])[n]))
        with self.lock:
            old = self._d.pop(key, None)
            if old is not None:
                self.bytes -= old.snap.nbytes
                self._lengths[n] -= 1
                if not self._lengths[n]:
                    del self._lengths[n]
            self._d[key] = _Entry(tuple(tokens[:n]), snap, conv_id, kind)
            self._lengths[n] = self._lengths.get(n, 0) + 1
            self.bytes += snap.nbytes
            self.stats["puts"] += 1
            self._evict_locked()

    def _evict_locked(self, room: int = 0) -> None:
        while self.bytes + room > self.budget and self._d:
            key, entry = self._d.popitem(last=False)
            self.bytes -= entry.snap.nbytes
            n = key[0]
            self._lengths[n] -= 1
            if not self._lengths[n]:
                del self._lengths[n]
            self.stats["evictions"] += 1

    def reserve(self, nbytes: int) -> bool:
        """Make room for a snapshot of `nbytes` BEFORE it is cloned (OPS-47): evict the oldest
        entries until the store plus it fits the budget. `put` evicting after the clone meant
        the allocator held the full store and the new snapshot at once, every checkpoint. False,
        and nothing evicted, when the snapshot is over the entry cap or the whole budget."""
        if (self.max_entry and nbytes > self.max_entry) or nbytes > self.budget:
            return False
        with self.lock:
            self._evict_locked(nbytes)
        return True

    def drop_session(self, conv_id: str | None) -> int:
        """Forget every entry of one conversation -- its compaction has started, and the history
        those entries are keyed by will not be sent again. Returns the bytes given back."""
        if not conv_id:
            return 0
        with self.lock:
            keys = [k for k, e in self._d.items() if e.conv_id == conv_id]
            freed = 0
            for key in keys:
                entry = self._d.pop(key)
                freed += entry.snap.nbytes
                self._lengths[key[0]] -= 1
                if not self._lengths[key[0]]:
                    del self._lengths[key[0]]
            self.bytes -= freed
            if keys:
                self.stats["dropped_session"] = self.stats.get("dropped_session", 0) + len(keys)
            return freed

    def shed(self, safe: bool = False) -> int:
        """`Budget`'s first level: everything, the cheapest cache to lose. Returns the bytes."""
        with self.lock:
            freed = self.bytes
        self.clear()
        return freed

    def set_limits(self, budget_bytes: int | None = None, max_entry_bytes: int | None = None):
        """A new budget or entry cap, from the next request on; over the budget goes now."""
        with self.lock:
            if budget_bytes is not None:
                self.budget = int(budget_bytes)
            if max_entry_bytes is not None:
                self.max_entry = int(max_entry_bytes)
            self._evict_locked()

    def boundaries(self, eng, drafter, start: int, n: int, chunk: int) -> set | None:
        """The chunk boundaries of a prefill over [start, n) to snapshot, or None for every one
        (`per_prefill` 0, where `capture` refuses the ones over the cap as it goes)."""
        if not self.per_prefill:
            return None
        lines = range((start // chunk + 1) * chunk, n, chunk)
        fit = [L for L in lines
               if not self.max_entry or _snapshot_estimate(eng, L, drafter) <= self.max_entry]
        if len(fit) < len(lines):
            self.stats["skipped_big"] = self.stats.get("skipped_big", 0) + 1
        return set(fit[-self.per_prefill:])

    # --- reading ---------------------------------------------------------------------------
    def find(self, tokens, hashes: list[int], max_len: int, min_len: int = 0):
        """The longest stored prefix of `tokens` no longer than `max_len`, or None.

        `max_len` is normally `len(tokens) - 1`: a request needs at least one token to forward,
        because the logits it answers with are the ones that token's own forward produces and no
        snapshot holds them. `min_len`: only an entry LONGER than this counts -- the resident
        prefix already offers that much without a copy, so a shorter entry is not a hit.
        """
        with self.lock:
            cands = sorted((L for L in self._lengths if min_len < L <= max_len), reverse=True)
            for L in cands:
                key = (L, hashes[L])
                entry = self._d.get(key)
                if entry is None:
                    continue
                if entry.tokens != tuple(tokens[:L]):
                    # a 64-bit collision, or two different token streams at one boundary. Either
                    # way this is not the state that prefix would have produced.
                    self.stats["rejected_collisions"] += 1
                    continue
                self._d.move_to_end(key)
                entry.hits += 1
                self.stats["hits"] += 1
                self.stats[f"hits_{entry.kind}"] = self.stats.get(f"hits_{entry.kind}", 0) + 1
                self.last_kind = entry.kind
                return L, entry
            if not min_len:
                # a request the resident prefix serves is not a store miss
                self.stats["misses"] += 1
            self.last_kind = None
            return None

    def report(self) -> dict:
        with self.lock:
            per = {}
            for e in self._d.values():
                for k, v in e.snap.parts().items():
                    per[k] = per.get(k, 0) + v
            return {"entries": len(self._d), "bytes": self.bytes, "budget": self.budget,
                    "boundaries": sorted(self._lengths), "bytes_by_part": per,
                    "chunk": self.chunk, **self.stats}

    def clear(self) -> None:
        with self.lock:
            self._d.clear()
            self._lengths.clear()
            self.bytes = 0


# --------------------------------------------------------------------------- the resident prefix
#: Rows past a generation's final `kv.length` that its rejected verify rows or a drafter's block
#: may have written. The widest block the served stack writes is 33 rows (a 32-node tree plus its
#: anchor); 512 is the margin a guest's stash is checked against.
GUEST_SLACK_ROWS = 512


# --------------------------------------------------------------------------- the borrowed engine
#: Attributes of the engine a guest's forwards may replace, and that the generation it interrupts
#: reads again: kept by reference and put back as they were.
_BORROW_ATTRS = ("mm", "pos_delta", "_rope_rows", "_trace", "hidden_pre_norm", "hidden_post_norm",
                 "tap", "trace", "tree", "_tree", "_tree_start", "_tree_is_chain", "picks",
                 "_pending_walk", "_walk_scratch", "_pend", "_pend_ctx", "_par", "_fold_par")


class Borrow:
    """The engine lent to a short request in the middle of another one's generation
    (the side lane, server/preempt.py), and given back exactly as it was.

    A generation stopped at a block boundary is the engine's KV rows below `kv.length`, the
    recurrent state and convolution tails, the commit a folding verify left pending (`_pend` and
    the factor buffers it names, or a chain's walked state in `_scratch_S`), and a few references.
    The guest is prefilled from position 0 and decodes one token at a time with no drafter, so it
    writes the rows [0, rows) and the state, reads no fold buffer and writes no `_scratch_S`. This
    keeps a copy of exactly what it writes -- the rows of every engine view below `rows`, S, conv,
    `_scratch_S` when a walked state waits in it -- and the attributes by reference, and `give_back`
    copies the bytes back into the same buffers (a captured verify graph holds their addresses).
    The drafter is not touched: its own caches, taps and policy stay where the generation left
    them, which is why the guest runs without one and with `eng.tap` off.

    The pending commit is NOT settled first: a guest's first forward settles it into S (the copy
    has it unsettled) and drops the record, and both come back, so the interrupted generation's
    next verify folds the same record into the same state, op for op.
    """

    def __init__(self, eng, rows: int):
        self.eng = eng
        self.rows = max(0, min(int(rows), int(eng.kv.max_len)))
        self.length = int(eng.kv.length)
        self.primed = bool(eng.state.primed)
        self.attrs = {a: getattr(eng, a, None) for a in _BORROW_ATTRS if hasattr(eng, a)}
        self.views = _kv_views(eng, None)
        self.saved = [t.narrow(ax, 0, self.rows).clone() for t, ax in self.views]
        self.state = state_clone(eng)
        scratch = getattr(eng, "_scratch_S", None)
        self.scratch = (scratch.clone() if scratch is not None and self.attrs.get("_pending_walk")
                        else None)
        self.nbytes = (sum(_nbytes(x) for x in self.saved) + sum(_nbytes(x) for x in self.state)
                       + _nbytes(self.scratch))
        # the guest starts clean: no images, no taps for the drafter, no pending commit or walk
        eng.mm, eng.pos_delta, eng.tap = None, 0, None
        eng._pending_walk = False
        eng.trace, eng.tree = None, None
        eng.reset()

    def give_back(self) -> None:
        eng = self.eng
        for (t, ax), buf in zip(self.views, self.saved):
            _kvp.ensure_for(t, self.rows)     # SRV-307
            t.narrow(ax, 0, self.rows).copy_(buf)
        state_load(eng, self.state)
        if self.scratch is not None:
            eng._scratch_S.copy_(self.scratch)
        for a, v in self.attrs.items():
            setattr(eng, a, v)
        eng.kv.length = self.length
        eng.state.primed = self.primed
        self.saved, self.state, self.scratch = [], None, None


def resident_capable(drafter) -> bool:
    """Can this drafter pick up at a position whose draft KV it already holds in place?

    The same three answers as `drafter_is_cacheable`, with `state_resume` in place of
    `state_snapshot`: a drafter with no `sync` is rebuilt from token ids by `prime`; one with
    `state_resume` is told where its own position-indexed cache is valid up to; any other would
    be left with a hole, so the resident prefix turns itself off for it.
    """
    if drafter is None:
        return True
    fn = getattr(drafter, "can_resume", None)
    if fn is not None:
        return bool(fn())
    if hasattr(drafter, "state_resume"):
        return True
    return not hasattr(drafter, "sync")


def _kv_views(eng, drafter) -> list:
    """Every position-indexed buffer a guest request can overwrite, as (tensor, axis): the engine's
    views and its drafter's."""
    views = model_views(eng)
    fn = getattr(drafter, "kv_views", None) if drafter is not None else None
    if fn is not None:
        views += [(t, ax) for t, ax in fn() if t is not None]
    return views


class ResidentPrefix:
    """The live KV buffer used as the prefix cache, with the recurrent state at a few boundaries.

    THE PROBLEM IT SOLVES. An agent client -- opencode -- sends one conversation
    that only ever grows: every turn's prompt is the previous prompt plus the previous answer
    plus a tool result. `StateStore` keeps whole snapshots, and a snapshot is 147 MiB of
    recurrent state plus 104 kB a token of target and draft KV, so under its per-entry cap
    (a quarter of CACHE_GB) nothing past 13,312 tokens was ever stored: at 190k tokens every
    turn re-read the whole conversation, ~310 s a turn, `skipped_big` 13,039 by the evening.

    THE OBSERVATION. The KV the previous request prefilled is still sitting in the engine's own
    buffer, row for row, and a KV row depends only on the tokens up to it. What cannot be
    recovered from the buffer is the recurrent state: the 48 GDN layers' S and convolution tail
    have absorbed every token and there is no inverse. So this keeps the KV where it is -- no
    copy -- and clones only S and conv (146.8 MiB) at chunk-grid boundaries of the prefill,
    "anchors". A request whose prompt shares the first `c` tokens of the resident prompt resumes
    at the largest anchor `B <= c`: S and conv are copied back, `kv.length = B`, and the drafter
    is told its own position-indexed cache is valid up to `B` (`state_resume`).

    WHY IT IS LOSSLESS, not merely close. An anchor is taken right after the prefill chunk that
    ends at `B`, the rows below `B` were written by that same grid of prefill chunks, and the
    resumed request forwards `B..n` on the same grid a cold prefill would use. Every forward after
    the resume therefore sees the same inputs a cold prefill's does, and `tests/test_resident.py`
    checks the logits, S, conv and KV rows for bit equality. Anchors are taken ONLY from prefill
    chunks, never from rows a verify block wrote: the generated answer is always re-read as
    prompt on the next turn, which is what a cold prefill of that turn does too. A request that
    started from a SESSION snapshot (written by verify blocks) leaves no anchors at all.

    GUESTS. A short unrelated request between two turns (opencode's title call, an Open WebUI
    chat) would overwrite the rows the conversation needs. A request shorter than `guest_frac` of
    the resident prompt and `guest_max` tokens is a guest: before its prefill the rows it can
    reach are copied aside (up to `stash_bytes`), and they are copied back at the start of the
    next request. If the guest ran further than the stash, the resident prefix is cut to what
    the stash covered. A request whose session comes back (`stay`: an agent's worker, ENG-371) is
    never a guest, whatever its length: its next call continues its own rows, and as a guest
    they were put back under the resident's after every call (2026-10-17 23:38: a worker's first
    three calls, 2,532, 5,764 and 7,925 tokens beside a 24,200-token chat, each read from 0).

    MEMORY. `budget_bytes` bounds the anchors (27 at the default 4 GiB) and `stash_bytes` the
    guest stash (2 GiB = ~20k rows at 104 kB a row). Both are allocated once and reused: an
    evicted anchor's buffers take the next anchor, so a 190k prefill that passes 185 boundaries
    allocates no more than the budget.

    A LONG FOREIGN REQUEST (`pin_bytes`). A request too long to be a guest replaces the resident
    prompt. opencode's compaction request is one: no tools, no system text, the history as one
    user message, 110k-132k tokens that share nothing with the conversation. The next build
    request starts with the same reasoning header, tool block and system text as every request
    before the compaction (at least 20,480 tokens), and found nothing: 7 of 7 re-read 52k-58k
    tokens from zero on the 2026-09-30 night. So before such a request cuts the rows, the head of
    the old prompt is PINNED: its KV rows and the anchors inside it are copied aside, as long a
    head as `pin_bytes` holds. A later request that shares more with the pinned head than with
    the rows in place gets them copied back and resumes at the largest pinned anchor it shares.
    The rows and anchors are the bytes a grid prefill wrote, so the resume is as exact as any
    other anchor's.

    ANOTHER SESSION (`park_bytes`, OPS-47). Two opencode sessions used in turn (A, then B, then A)
    share their head and nothing after it, and each switch re-read the other session past the
    head: 81,199 tokens in 125 s on the 2026-09-30 night. So when a request of ANOTHER session
    (`x-session-id`, or no id and a prompt that parts from the rows) replaces the resident prompt,
    the rows it is about to overwrite -- from where the two part up to the newest anchor that fits
    -- are copied aside with the newest `park_anchors` anchors, which move rather than copy. A
    later request that shares more with the parked prompt than with the rows in place gets them
    back: the parked rows and the rows in their place are SWAPPED, so B is parked while A comes
    back, and A, B, A, B is a copy each time instead of a re-read. The head below the parked rows
    has to be in place (it is the head both sessions share); the rows and anchors are a grid
    prefill's bytes, so the resume is as exact as the pin's.

    THE PARKED END (`park_end`). Open WebUI never sends an answer's reasoning back, so a chat's copy
    parts from the rows inside its first answer, and the park held rows and grid anchors only: it
    could offer nothing above the first reasoning, which the disk's head already gave (0.4.7 live,
    2026-10-08: `parks 4, park_hits 0`; chat B, 36,841 tokens parked at 12:14:30, came back at
    12:15:01 with 12,288 cached from the disk and 16,370 re-read, 23.7 s). So when the rows up to
    the end of the last generation fit the park, they are parked whole, and the end anchor goes
    with them: its line, hidden spans and mark, its state buffers (moved, not copied: they
    take one of the `park_anchors` places), and the end alignment's regions. A copy that continues
    the parked conversation to its end -- every token, the hidden spans put back, or aligned by
    the server's end alignment -- is offered the end (`park_offer`), and the server brings the
    park back before its replay (`unpark_end`), as it reads an NVMe entry back (engine/spill.py).
    The resume is the end anchor's, held to its gate; the swap parks the other conversation's end
    the same way.

    THE END OF A GENERATION (`end`). Anchors come from prefill chunks only, so the
    next turn of an agent conversation re-read the whole previous answer: 33,172 tokens, about
    44 s, after a 32,773-token thinking (rrr2, 2026-10-02). `keep_end` keeps ONE more anchor when
    a generation ends: the recurrent state and convolution tails at the last position the engine
    forwarded (147 MiB, the same buffers turn after turn), with the rows in place marked valid up
    to there. A prompt that starts with all of those tokens resumes there and forwards only its
    new messages. Verify blocks wrote that state, so the resume is held to the gate the session
    cache is held to -- the same greedy tokens -- and not to bit equality with a cold prefill;
    so is everything a prefill that resumed there writes. `taint` is the lowest such row: an
    anchor above it is a resume under the same gate, the disk tier writes nothing of a prompt
    that has one and reads over those rows, and the state store takes no prefix snapshot of it.
    A cold prefill, or one resumed at or below `taint`, clears it. The tokens the session
    monitor hid in an answer (a hidden close's phrase, a hidden turn) are in the rows and not in
    the client's copy: `end["hidden"]` lists them with their positions, and `replay` puts them
    back where they sit, as the router replays its notes. The pin and the park carry their rows'
    `taint` with them; the end anchor goes to the park with its rows when they fit there, and is
    dropped otherwise.

    THE MEMORY POLICY (`mem`, a `Budget`). Every buffer this allocates after the first is asked for
    first: an anchor that is refused reuses an evicted one, a stash that is refused leaves the
    guest to overwrite the rows (the prompt is cut there), a pin or a park that is refused is not
    taken, nor is an end anchor. `shed_*` give each part back when the budget says so, in its
    order.
    """

    def __init__(self, budget_bytes: int, chunk: int, *, tail: int = 4, stash_bytes: int = 0,
                 guest_max: int = 16384, guest_frac: float = 0.25,
                 pin_bytes: int = 0, pin_anchors: int = 8, park_bytes: int = 0,
                 park_anchors: int = 4, mem: "Budget | None" = None, end: bool = False):
        self.budget = int(budget_bytes)
        self.chunk = int(chunk)
        self.tail = max(1, int(tail))
        self.stash_budget = int(stash_bytes)
        self.guest_max = int(guest_max)
        self.guest_frac = float(guest_frac)
        self.tokens: list[int] = []        # the prompt the live rows [0, valid) were prefilled from
        self.valid = 0
        self.anchors: dict[int, tuple] = {}
        self._free: list[tuple] = []       # an evicted anchor's buffers, for the next one
        self._allocated = 0                # anchor buffers alive, in use or free
        self._anchor_bytes = 0
        self._stash: dict | None = None    # the rows a guest may overwrite, copied aside
        self._stash_bufs: list | None = None
        self._stash_rows = 0
        self.capturing = False             # this request's prefill may add anchors
        # the pinned head of an earlier prompt: rows [0, pin_rows) of every view, and anchors
        self.pin_budget = int(pin_bytes)
        self.pin_anchor_max = max(1, int(pin_anchors))
        self.pin_tokens: list[int] = []
        self.pin_rows = 0
        self.pin_anchors: dict[int, tuple] = {}
        self._pin_bufs: list | None = None
        self._pin_views: list | None = None
        self._pin_spare: list[tuple] = []  # anchor-shaped buffers the pin owns
        # the parked session: rows [park_base, park_base + park_rows) of `park_tokens`, and anchors
        self.park_budget = int(park_bytes)
        self.park_anchor_max = max(1, int(park_anchors))
        self.park_tokens: list[int] = []
        self.park_base = 0
        self.park_rows = 0
        self.park_owner: str | None = None
        self.park_anchors: dict[int, tuple] = {}
        self._park_bufs: list | None = None
        self._park_views: list | None = None
        # the parked conversation's end anchor (`_park`): the `end` dict with its own state
        # as "pair", and the end alignment's regions as they were at park time
        self.park_end: dict | None = None
        self.park_regions: list = []
        # the server's end alignment (server/diskcache.py `wire_end_align`), so this file imports
        # nothing of the server: (res) -> the regions of `res.end`; (res, regions) -> adopts them
        # for `res.end`; (tokens, end, regions, copy) -> whether `copy` aligns to that end
        self.end_extra_of = None
        self.end_adopt = None
        self.park_align = None
        self.owner: str | None = None      # the session (x-session-id) the rows belong to
        # the end of the last generation (`keep_end`): its line `at`, the hidden spans below it
        # and the drafter's mark; its state in `_end_bufs`, allocated once and reused
        self.end_on = bool(end)
        self.end: dict | None = None
        self.end_lost: str | None = None   # why the last end anchor went (`_drop_end`)
        self._end_bufs: tuple | None = None
        # the tokenizer's whitespace-only tokens (the server's, at load): where the client's copy
        # of an answer may differ from the rows next to a hidden span (`replay`)
        self.glue: frozenset = frozenset()
        # the lowest row a session's state wrote (a generation, or a prefill resumed from its
        # end); None while every row and anchor is a cold grid prefill's
        self.taint: int | None = None
        self.pin_taint: int | None = None
        self.park_taint: int | None = None
        self.mem = mem                     # the memory policy (Budget), None for no limits
        self.switches: deque = deque(maxlen=20)
        self.stats = {"hits": 0, "misses": 0, "tokens_reused": 0, "anchors_taken": 0,
                      "anchors_evicted": 0, "guests": 0, "stash_restored": 0, "stash_short": 0,
                      "invalidated": 0,
                      "pins": 0, "pin_hits": 0, "pin_tokens_reused": 0, "pin_refused": 0,
                      "parks": 0, "park_hits": 0, "park_swaps": 0, "park_tokens_reused": 0,
                      "park_refused": 0, "park_dropped": 0, "switches": 0,
                      "refused_anchor": 0, "refused_stash": 0, "refused_pin": 0,
                      "refused_park": 0,
                      "end_kept": 0, "end_hits": 0, "end_tokens_reused": 0, "end_saved": 0,
                      "end_replayed": 0, "end_ahead": 0, "end_refused": 0, "end_dropped": 0,
                      "end_cut": 0,
                      "spills": 0, "spill_failed": 0, "spill_restores": 0,
                      "park_spills": 0, "park_given_up": 0, "side_carries": 0,
                      "carry_vacated": 0, "park_ends": 0, "park_end_hits": 0}
        # the session spill (engine/spill.py): a long conversation another one
        # replaces is written to the NVMe before its rows are; None without a disk directory
        self.spiller = None
        # ENG-371: (res, eng, drafter, stay) -> None, for a prefix with no park and no spill of its
        # own (a side slot's): another session's rows are about to be cut, by a worker's request
        # when `stay`; the server writes them to the NVMe first (server/app.py `_keep_parent`)
        self.keeper = None
        self._park_busy = 0                # a park, swap or park write runs: `shed_park` waits
        self.end_hold = False              # ENG-371: the request at the memory gate resumes here
        self.lock = threading.Lock()

    # --- lookups ---------------------------------------------------------------------------
    def common(self, ids) -> int:
        """How many leading tokens `ids` shares with the prompt the resident rows hold."""
        m = min(len(ids), self.valid)
        if m == 0:
            return 0
        if list(ids[:m]) == self.tokens[:m]:
            return m
        lo, hi = 0, m            # tokens[:lo] equal, tokens[:hi] not
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if list(ids[:mid]) == self.tokens[:mid]:
                lo = mid
            else:
                hi = mid
        return lo

    def best(self, c: int, max_len: int) -> int:
        """The largest anchor at or below both `c` and `max_len`; 0 for none."""
        lim = min(c, max_len)
        return max((b for b in self.anchors if b <= lim), default=0)

    # --- the request's start ---------------------------------------------------------------
    def settle_guest(self, eng) -> None:
        """Put back what the last guest request may have overwritten. Called before anything
        else reads the resident rows."""
        st = self._stash
        if st is None:
            return
        self._stash = None
        s, r = st["start"], st["rows"]
        # how far the guest wrote, read BEFORE the rows go back: an engine whose length lands a pending
        # commit when read (Flash-Next's chain block) writes the guest's last rows then, and after the
        # copy that write went over the resident's restored rows (2026-10-12: a guest after a guest
        # answered differently from a cold prefill)
        reach = int(eng.kv.length) + GUEST_SLACK_ROWS
        for (t, ax), buf in zip(st["views"], self._stash_bufs):
            _kvp.ensure_for(t, s + r)         # SRV-307
            t.narrow(ax, s, r).copy_(buf.narrow(ax, 0, r))
        self.stats["stash_restored"] += 1
        if reach > s + r and self.valid > s + r:
            # the guest wrote past what was saved: the rows beyond it are not the resident's any
            # more, and neither is any anchor above them
            self.stats["stash_short"] += 1
            self._cut(s + r)

    def is_guest(self, n: int, c: int) -> bool:
        return (self.valid > 0 and c < self.valid and n <= self.guest_max
                and n < self.guest_frac * self.valid and self.stash_budget > 0)

    def resume(self, eng, drafter, b: int) -> None:
        """Bring the engine to `b` from the rows in place and the anchor at `b`."""
        state_load(eng, self.anchors[b])
        eng.resume_at(b)
        if drafter is not None and hasattr(drafter, "state_resume"):
            drafter.state_resume(b)
        self.stats["hits"] += 1
        self.stats["tokens_reused"] += b

    # --- the end of the last generation -------------------------------------------------------
    def keep_end(self, eng, drafter, tokens, n_prompt: int, hidden=(), delta: int = 0,
                 mm: bool = False) -> str:
        """After a generation: the anchor at the last position the engine forwarded, with the
        rows in place valid up to it. `tokens` is the request's context -- its prompt as it was
        prefilled, then what the engine committed -- and `n_prompt` the prompt's length; `hidden`
        the spans `(position, ids)` of tokens in it the client's copy will not carry. Returns what
        happened, for the log: kept, off, guest, not_resident, ahead or memory.

        `mm`: the prompt had images (its tokens are their key ids) and `delta` is the
        rotary offset every row after them takes. Such an end is offered only to a request that
        says so (`end_offer(mm_ok=True)`: a compaction served as this conversation's continuation,
        server/compaction.py); every other request resumes at the anchors in place, as before."""
        if not self.end_on or self.chunk <= 0 or not resident_capable(drafter):
            return "off"
        if self._stash is not None:
            return "guest"                 # the rows in place are the resident prompt's
        n_prompt = int(n_prompt)
        e = int(eng.kv.length)
        if (self.valid != n_prompt or len(self.tokens) != n_prompt or e < n_prompt
                or self.tokens != [int(t) for t in tokens[:n_prompt]]):
            # a prefill from a session snapshot (nothing is resident), or one a client stopped
            return "not_resident"
        if e > len(tokens):
            # the last block forwarded rows past the committed tokens: its pending commit is cut
            # back to them, as `--session-trim` cuts the session snapshot's; a state that has
            # absorbed them has no key
            if not eng.trim_pending(len(tokens)):
                self.stats["end_ahead"] += 1
                return "ahead"
            e = len(tokens)
        if self._end_bufs is None:
            if self.mem is not None and not self.mem.admit("end", eng.cost().section_bytes):
                # SRV-655: the end is worth more than an anchor below it (the next turn resumes
                # there and reads only its new messages): an anchor's buffers take it, the
                # anchor that leaves the smallest gap, before the end is refused
                if not self._free and self.anchors:
                    self._evict_one()
                if not self._free:
                    self.stats["end_refused"] += 1
                    return "memory"
                self._end_bufs = self._free.pop()
                self._allocated -= 1
                self.stats["end_from_anchor"] = self.stats.get("end_from_anchor", 0) + 1
            else:
                self._end_bufs = state_empty(eng)
        eng.settle()                       # a walked state or a pending commit, as `capture`
        state_save(self._end_bufs, eng)
        mark = getattr(drafter, "end_mark", None) if drafter is not None else None
        spans = sorted((int(p), [int(t) for t in h]) for p, h in hidden)
        self.end = {"at": e, "hidden": [(p, h) for p, h in spans if h and p + len(h) <= e],
                    "mark": mark() if mark is not None else None}
        if mm or delta:
            self.end.update(delta=int(delta), mm=bool(mm))
        self.tokens = [int(t) for t in tokens[:e]]
        self.valid = e
        self.capturing = False
        if e > n_prompt:
            self.taint = n_prompt if self.taint is None else min(self.taint, n_prompt)
        self.end_lost = None
        self.stats["end_kept"] += 1
        return "kept"

    def end_offer(self, c: int, max_len: int, mm_ok: bool = False) -> int:
        """The end anchor's line, when the prompt starts with every token up to it (`c`, its
        share of the rows in place) and it beats the anchors in place; else 0. An end after
        images only with `mm_ok` (`keep_end`)."""
        if self.end is not None and self.end.get("mm") and not mm_ok:
            return 0
        e = self.end["at"] if self.end is not None else 0
        return e if (e and c >= e and e <= max_len and e > self.best(c, max_len)) else 0

    def resume_end(self, eng, drafter, saved: int = 0) -> None:
        """Bring the engine to the end anchor's line: the state its generation ended in, the rows
        in place. The drafter picks up there too; one that keeps a mark (the length router's
        released arm) inherits it, as a session snapshot's restore does."""
        e = self.end["at"]
        state_load(eng, self._end_bufs)
        eng.resume_at(e)
        if drafter is not None:
            fn = getattr(drafter, "resume_end", None)
            if fn is not None:
                fn(e, self.end["mark"])
            elif hasattr(drafter, "state_resume"):
                drafter.state_resume(e)
        self.stats["end_hits"] += 1
        self.stats["end_tokens_reused"] += e
        self.stats["end_saved"] += max(0, int(saved))

    def replay(self, ids) -> tuple[list, list]:
        """`ids` with the end anchor's hidden spans put back where the rows hold them, when the
        prompt then starts with every token up to the end anchor; else `ids` as they are.
        Returns (ids, the spans put back).

        The client's copy is re-rendered from text, and next to a hidden span its whitespace is
        not the rows': the template trims the reasoning and writes `\n</think>`, where the hidden
        close's phrase already ended in its newline (0.3.14's live check, 2026-10-02 22:07: the
        copy held one `\n` more than the rows right after the phrase, and the end anchor was
        never reached). So a run of `glue` tokens (whitespace only) next to a span may differ;
        every other token must be the rows' own, and what is served up to the end anchor is
        the rows' tokens -- the context the engine really had."""
        end = self.end
        if end is None or not end["hidden"]:
            return ids, []
        e, rows, glue = end["at"], self.tokens, self.glue
        copy = list(ids)
        i = j = 0                          # i in the rows, j in the client's copy
        for p, h in end["hidden"]:
            q = p
            while q > i and rows[q - 1] in glue:
                q -= 1                     # the rows' whitespace right before the span
            if copy[j:j + q - i] != rows[i:q]:
                return ids, []
            j += q - i
            while j < len(copy) and copy[j] in glue:
                j += 1                     # the copy's whitespace where the span sat
            i = p + len(h)
            while i < e and rows[i] in glue:
                i += 1                     # the rows' whitespace right after it
        if len(copy) - j <= e - i or copy[j:j + e - i] != rows[i:e]:
            return ids, []
        out = rows[:e] + copy[j + e - i:]
        self.stats["end_replayed"] += len(out) - len(copy)
        return out, [(p, list(h)) for p, h in end["hidden"]]

    def _drop_end(self, why: str = "rows") -> None:
        """The end anchor's rows are not in place any more; its buffers stay for the next one.
        `why` stays for the words of the next turn's re-read (`reread_words`)."""
        if self.end is not None:
            self.end = None
            self.end_lost = why
            self.stats["end_dropped"] += 1

    def begin(self, eng, drafter, ids, start: int, c: int, kind: str | None,
              session: str | None = None, compacting: bool = False,
              scratch: bool = False, reach: int | None = None, stay: bool = False) -> None:
        """What the rows will hold once this request's prefill has run.

        `start` is where the prefill resumes (a resident anchor, a store entry, or 0) and `c` how
        much of the resident prompt `ids` shares. A guest keeps the resident prompt and stashes the
        rows it can reach; anything else becomes the resident prompt from here on. `session` is
        the request's session key; `compacting` says it is that session's compaction, whose old
        rows nobody asks for again (they are not parked). `scratch` (a summary
        prepared in the background): the rows this prefill writes are nobody's, no anchor is taken
        of them; the caller puts the resident prompt's bookkeeping back afterwards. `reach`: the
        prompt's length plus the request's token budget, the highest row its generation can
        commit; a guest's stash then holds the rows up to there (and two slack margins), not as
        many as the stash budget holds (FX5). `stay`: the session comes back (ENG-371), so this
        request is never a guest.
        """
        self.capturing = False
        if kind == "session":
            # rows written by verify blocks: nothing anchored after them is a cold prefill's state.
            # A long conversation in place goes to the disk first (it is not a guest's to keep)
            if (self.spiller is not None and (session is None or session != self.owner)
                    and self.valid >= self.spiller.min_tokens):
                self._spill(eng, drafter, "evict")
            self.clear()
            self.owner = session
            return
        n = len(ids)
        if not stay and self.is_guest(n, c):
            self._stash_for(eng, drafter, min(start, (c // self.chunk) * self.chunk), reach)
            return
        if self.valid and c < self.valid:
            self.stats["invalidated"] += 1
        keep = min(start, c)
        parked, refused = self.stats["parks"], self.stats["refused_park"]
        other = session is None or session != self.owner
        # A conversation longer than the RAM park holds above `keep` goes whole to
        # the disk (engine/spill.py) before this prefill writes a row; it is not parked or pinned
        spilled = (self.spiller is not None and self.valid > keep and not compacting and other
                   and self.valid >= self.spiller.min_tokens
                   and self.valid - keep > self._park_cap_rows(eng, drafter)
                   and self._spill(eng, drafter, "evict"))
        if (self.park_budget and self.valid > keep and not compacting and other
                and not spilled):
            self._park(eng, drafter, keep)
            if (self.stats["refused_park"] > refused and self.spiller is not None
                    and self.valid >= self.spiller.min_tokens):
                # ENG-371: the memory policy refused the park: the NVMe, not the floor
                spilled = self._spill(eng, drafter, "evict")
        if (self.keeper is not None and self.valid > keep and not compacting and other
                and not spilled and self.stats["parks"] == parked):
            try:
                self.keeper(self, eng, drafter, stay)
            except Exception:                                      # noqa: BLE001
                pass
        # A parked prompt keeps its rows above `keep`, and its head below `keep` stays in place
        # (the new prompt shares it): pinning that head too held the same session twice -- 3.8 GiB
        # of pin beside 4 GiB of park in the 2026-10-01 gate. The pin stays the compaction's (and
        # any replacement the park did not take).
        if (self.pin_budget and self.valid > keep and self.stats["parks"] == parked
                and not spilled):
            self._pin(eng, drafter, keep)
        self._cut(keep)
        self.tokens = list(ids)
        self.valid = start
        self.owner = session
        self.capturing = self.chunk > 0 and not scratch
        if kind == "end":
            # a prefill from the end of a generation writes a session's rows
            self.taint = start if self.taint is None else min(self.taint, start)
        elif kind != "resident" or self.taint is None or start <= self.taint:
            # a cold start, a store or disk restore, or an anchor of a cold grid's: what this
            # prefill writes is a cold grid prefill's again
            self.taint = None

    # --- the session spill (engine/spill.py) ------------------------------------------------
    def _park_cap_rows(self, eng, drafter) -> int:
        """The rows above the shared head the RAM park can hold (0 without a park)."""
        if not self.park_budget:
            return 0
        views = _kv_views(eng, drafter)
        per_row = sum(t.numel() // t.shape[ax] * t.element_size() for t, ax in views)
        ab = self._anchor_bytes or eng.cost().section_bytes
        return max(0, (self.park_budget - self.park_anchor_max * ab) // max(1, per_row))

    def _spill(self, eng, drafter, reason: str) -> bool:
        """The resident conversation to the disk; whether it is there now."""
        try:
            info = self.spiller.save_resident(self, eng, drafter, reason)
        except Exception:                                          # noqa: BLE001
            info = None
        if info is None:
            self.stats["spill_failed"] += 1
            return False
        self.stats["spills"] += 1
        return True

    def vacate(self, eng, drafter, session) -> str:
        """FX10 (SRV-213): before another slot's rows are copied over the rows in place from row
        0, the resident conversation leaves as a request of another session would make it leave
        (`begin`), with nothing of it kept in place: to the NVMe when it is at least the spill's
        eviction length and longer than the RAM park holds, else to the park (from row 0, so its
        head goes with it). Returns spilled, parked, dropped or empty."""
        if self._stash is not None:
            self.settle_guest(eng)
        if not self.valid:
            return "empty"
        self.stats["carry_vacated"] += 1
        other = session is None or session != self.owner
        if (self.spiller is not None and other and self.valid >= self.spiller.min_tokens
                and self.valid > self._park_cap_rows(eng, drafter)
                and self._spill(eng, drafter, "switch")):
            self.clear()
            return "spilled"
        parked, refused = self.stats["parks"], self.stats["refused_park"]
        if self.park_budget and other:
            self._park(eng, drafter, 0)
        if (self.stats["refused_park"] > refused and other and self.spiller is not None
                and self.valid >= self.spiller.min_tokens and self._spill(eng, drafter, "switch")):
            # ENG-371: the park refused it (live 2026-10-18 02:34: a worker's 27,381 tokens were
            # dropped under the memory ceiling, its next call read them all again)
            self.clear()
            return "spilled"
        how = "parked" if self.stats["parks"] > parked else "dropped"
        self.clear()
        return how

    def adopt(self, tokens, valid: int, taint, owner, count: str = "spill_restores") -> None:
        """A conversation read back from the disk is the resident prompt from here on: its rows
        are in place (engine/spill.py wrote them), its anchors and end anchor follow through
        `adopt_anchor` and `adopt_end`. What the rows held before goes. `count`: the counter it
        adds to (`side_carries` for a conversation carried from a side slot, FX10)."""
        self._stash = None
        self._drop_end()
        for b in list(self.anchors):
            self._free.append(self.anchors.pop(b))
        self.tokens = [int(t) for t in tokens[:valid]]
        self.valid = int(valid)
        self.taint = None if taint is None else int(taint)
        self.owner = owner
        self.capturing = False
        self.stats[count] += 1

    def adopt_anchor(self, eng, b: int):
        """A buffer pair for the restored anchor at `b` (None when the memory policy refuses)."""
        pair = self._buffers(eng)
        if pair is not None:
            self.anchors[int(b)] = pair
        return pair

    def adopt_end(self, eng, end: dict):
        """The end anchor's buffers for a restored end at `end["at"]` (None: off or refused)."""
        if not self.end_on or int(end["at"]) != self.valid:
            return None
        if self._end_bufs is None:
            if self.mem is not None and not self.mem.admit("end", eng.cost().section_bytes):
                self.stats["end_refused"] += 1
                return None
            self._end_bufs = state_empty(eng)
        self.end = {"at": int(end["at"]), "hidden": [(int(p), list(h)) for p, h in end["hidden"]],
                    "mark": end.get("mark")}
        if end.get("served_hidden"):
            # FX16: the rows the client never sent travel with the end (the NVMe, the park)
            self.end["served_hidden"] = [(int(a), int(b)) for a, b in end["served_hidden"]]
        if end.get("mm") or end.get("delta"):
            self.end.update(delta=int(end.get("delta") or 0), mm=bool(end.get("mm")))
        return self._end_bufs

    def advanced(self, i: int) -> None:
        """The prefill has written rows up to `i`."""
        if self.capturing:
            self.valid = i

    # --- anchors ---------------------------------------------------------------------------
    def anchor(self, eng, b: int) -> None:
        """Clone the recurrent state at grid boundary `b`, reusing an evicted anchor's buffers."""
        if not self.capturing or b in self.anchors:
            return
        if not self._anchor_bytes:
            self._anchor_bytes = eng.cost().section_bytes
        cap = max(self.tail + 1, self.budget // max(1, self._anchor_bytes))
        if len(self.anchors) >= cap:
            self._evict_one()
        if not self._free and self.mem is not None and not self.mem.admit(
                "anchors", self._anchor_bytes):
            # over the budget: the next anchor takes an evicted one's buffers, or is not taken
            self.stats["refused_anchor"] += 1
            if not self.anchors:
                return
            self._evict_one()
        if self._free:
            buf = self._free.pop()
            state_save(buf, eng)
        else:
            buf = state_clone(eng)
            self._allocated += 1
        self.anchors[b] = buf
        self.stats["anchors_taken"] += 1

    def _buffers(self, eng) -> tuple | None:
        """A free state buffer for the next anchor, evicting one if the budget is full; None
        when the memory budget refuses a new one and no anchor is left to evict."""
        if not self._anchor_bytes:
            self._anchor_bytes = eng.cost().section_bytes
        cap = max(self.tail + 1, self.budget // max(1, self._anchor_bytes))
        if len(self.anchors) >= cap:
            self._evict_one()
        if not self._free and self.mem is not None and not self.mem.admit(
                "anchors", self._anchor_bytes):
            self.stats["refused_anchor"] += 1
            if not self.anchors:
                return None
            self._evict_one()
        if self._free:
            return self._free.pop()
        self._allocated += 1
        return state_empty(eng)

    def head_line(self, b: int) -> bool:
        """With a pin: one of the `pin_anchor_max` anchors every `4 * chunk` tokens at the head of
        the prompt, which eviction keeps. The head is where an agent conversation parts from
        itself -- its tool block and system text, a compaction, a new day's date line -- and the
        smallest-gap rule alone thins it out as the conversation grows: at 200k tokens the lowest
        surviving anchor sat above a 5k tool block, and the pin had nothing to offer."""
        step = 4 * self.chunk
        return bool(self.pin_budget) and b % step == 0 and b <= step * self.pin_anchor_max

    def _evict_one(self) -> None:
        """Drop the anchor whose removal leaves the smallest gap, never one of the newest
        `tail` (the next turn resumes near the end) nor a head line; an early divergence
        anywhere."""
        bs = sorted(self.anchors)
        body = [(j, b) for j, b in enumerate(bs[:-self.tail] if len(bs) > self.tail else [])
                if not self.head_line(b)]
        if not body:
            victim = next((b for b in bs if not self.head_line(b)), bs[0])
        else:
            best, victim = None, body[0][1]
            for j, b in body:
                lo = bs[j - 1] if j > 0 else 0
                hi = bs[j + 1]
                gap = hi - lo
                if best is None or gap < best:
                    best, victim = gap, b
        self._free.append(self.anchors.pop(victim))
        self.stats["anchors_evicted"] += 1

    def _cut(self, keep: int) -> None:
        """The rows at and above `keep` are not the resident's any more."""
        for b in [b for b in self.anchors if b > keep]:
            self._free.append(self.anchors.pop(b))
        self.valid = min(self.valid, keep)
        self.tokens = self.tokens[:self.valid]
        if self.end is not None and self.end["at"] > self.valid:
            self._drop_end()
        if self.taint is not None and self.taint >= self.valid:
            self.taint = None

    # --- guests ----------------------------------------------------------------------------
    def _stash_for(self, eng, drafter, s: int, reach: int | None = None) -> None:
        views = _kv_views(eng, drafter)
        per_row = sum(t.numel() // t.shape[ax] * t.element_size() for t, ax in views)
        cap = self.stash_budget // max(1, per_row)
        rows = max(0, min(self.valid - s, cap, eng.kv.max_len - s))
        if reach is not None:
            # FX5: only the rows the guest can write (its prompt, its budget, the rows a verify
            # block may write past its last commit, and the margin `settle_guest` checks with):
            # a 410-token classifier question saved 10,000 rows (1 GiB) before, and asked the
            # memory policy for all of them
            rows = min(rows, max(0, int(reach) + 2 * GUEST_SLACK_ROWS - s))
        if rows <= 0:
            self._cut(s)
            return
        if (self._stash_bufs is None or self._stash_rows < rows
                or len(self._stash_bufs) != len(views)):
            # the buffer is kept for the next guest: rounded up to the chunk grid, and as large as
            # the resident prompt when no reach is known (as before FX5)
            grid = max(1, self.chunk or 1)
            size = (min(cap, -(-rows // grid) * grid) if reach is not None
                    else min(cap, max(rows, self.valid)))
            self._stash_bufs, self._stash_rows = None, 0
            if self.mem is not None and not self.mem.admit("stash", size * per_row):
                # no room for the stash: the guest overwrites the rows, which stop being the
                # resident's -- the next turn re-reads them, as it did before the stash existed
                self.stats["refused_stash"] += 1
                self._cut(s)
                return
            self._stash_bufs = [torch.empty(*(t.shape[:ax] + (size,) + t.shape[ax + 1:]),
                                            dtype=t.dtype, device=t.device) for t, ax in views]
            self._stash_rows = size
        for (t, ax), buf in zip(views, self._stash_bufs):
            buf.narrow(ax, 0, rows).copy_(t.narrow(ax, s, rows))
        self._stash = {"start": s, "rows": rows, "views": views}
        self.stats["guests"] += 1
        if rows < self.valid - s:
            # the stash could not hold every row a guest might reach; what it did not save is
            # decided after the guest, from how far it actually wrote
            pass

    # --- the pinned head ---------------------------------------------------------------------
    def _pin(self, eng, drafter, keep: int) -> None:
        """Copy aside the longest anchored head of the resident prompt that `pin_bytes` holds,
        when it is longer than the `keep` rows the next prefill leaves in place."""
        views = _kv_views(eng, drafter)
        per_row = sum(t.numel() // t.shape[ax] * t.element_size() for t, ax in views)
        ab = self._anchor_bytes or eng.cost().section_bytes
        heads = sorted(b for b in self.anchors if b <= self.valid)
        best = 0
        for j, b in enumerate(heads):
            k = min(j + 1, self.pin_anchor_max)
            if b * per_row + k * ab <= self.pin_budget:
                best = b
        if best < keep + self.chunk:
            return
        if self.pin_rows and self.pin_tokens[:best] == self.tokens[:best]:
            return                         # the same head is already pinned
        # anchors spread over the head, the longest always among them: where the next request
        # parts from it is not known yet, and it resumes at the largest pinned anchor below that.
        # The head lines are that spread (eviction keeps them); without them, the anchors
        # nearest to an even spread.
        below = [b for b in heads if b <= best]
        k = self.pin_anchor_max
        lines = [b for b in below if self.head_line(b) and b != best]
        if lines:
            keep_b = sorted(lines[-(k - 1):] + [best]) if k > 1 else [best]
        else:
            keep_b = sorted({max(b for b in below if b <= max(below[0], best * j // k))
                             for j in range(1, k + 1)})
        if (self._pin_bufs is None or len(self._pin_bufs) != len(views)
                or self._pin_bufs[0].shape[views[0][1]] < best):
            size = max(best, self._pin_bufs[0].shape[views[0][1]] if self._pin_bufs else 0)
            size = min(size, max(best, (self.pin_budget - len(keep_b) * ab) // max(1, per_row)))
            spare = max(0, len(keep_b) - len(self._pin_spare) - len(self.pin_anchors))
            if self._pin_bufs is not None:
                # the old head's rows go: the new head is a longer copy of it or another prompt's
                self._drop_pin(rows_only=True)
            if self.mem is not None and not self.mem.admit("pin", size * per_row + spare * ab):
                self.stats["refused_pin"] += 1
                return
            self._pin_bufs = [torch.empty(*(t.shape[:ax] + (size,) + t.shape[ax + 1:]),
                                          dtype=t.dtype, device=t.device) for t, ax in views]
        for (t, ax), buf in zip(views, self._pin_bufs):
            buf.narrow(ax, 0, best).copy_(t.narrow(ax, 0, best))
        self._pin_spare += list(self.pin_anchors.values())
        self.pin_anchors = {}
        for b in keep_b:
            buf = self._pin_spare.pop() if self._pin_spare else state_empty(eng)
            state_copy(buf, self.anchors[b])
            self.pin_anchors[b] = buf
        self._pin_spare = []               # what an earlier, longer head needed is not kept
        self.pin_tokens = list(self.tokens[:best])
        self.pin_rows = best
        self.pin_taint = self.taint if self.taint is not None and self.taint < best else None
        self._pin_views = [(t, ax) for t, ax in views]
        self.stats["pins"] += 1

    def pin_offer(self, ids, c: int, max_len: int) -> int:
        """The pinned anchor a prompt could resume at, when it beats the rows in place; else 0."""
        if not self.pin_rows or not self.pin_anchors:
            return 0
        m = min(len(ids), self.pin_rows)
        if list(ids[:m]) == self.pin_tokens[:m]:
            cp = m
        else:
            lo, hi = 0, m
            while hi - lo > 1:
                mid = (lo + hi) // 2
                if list(ids[:mid]) == self.pin_tokens[:mid]:
                    lo = mid
                else:
                    hi = mid
            cp = lo
        bp = max((b for b in self.pin_anchors if b <= min(cp, max_len)), default=0)
        return bp if bp > self.best(c, max_len) else 0

    def unpin(self, eng, drafter) -> bool:
        """Make the pinned head the resident prompt: its rows copied back, its anchors the live
        ones. False, and nothing changed, if the buffers it was copied from are not the ones in
        service any more (a drafter that reallocated its cache)."""
        views = _kv_views(eng, drafter)
        if (self._pin_views is None or len(views) != len(self._pin_views)
                or any(a is not b or x != y
                       for (a, x), (b, y) in zip(views, self._pin_views))):
            self.stats["pin_refused"] += 1
            return False
        self._stash = None
        self._drop_end()
        for b in list(self.anchors):
            self._free.append(self.anchors.pop(b))
        r = self.pin_rows
        for (t, ax), buf in zip(views, self._pin_bufs):
            _kvp.ensure_for(t, r)             # SRV-307
            t.narrow(ax, 0, r).copy_(buf.narrow(ax, 0, r))
        for b, kept in sorted(self.pin_anchors.items()):
            buf = self._buffers(eng)
            state_copy(buf, kept)
            self.anchors[b] = buf
        self.tokens = list(self.pin_tokens)
        self.valid = r
        self.taint = self.pin_taint
        self.stats["pin_hits"] += 1
        return True

    def _drop_pin(self, rows_only: bool = False) -> int:
        freed = sum(b.numel() * b.element_size() for b in (self._pin_bufs or []))
        self._pin_bufs, self._pin_views = None, None
        self.pin_tokens, self.pin_rows, self.pin_taint = [], 0, None
        if not rows_only:
            freed += (len(self.pin_anchors) + len(self._pin_spare)) * self._anchor_bytes
            self.pin_anchors, self._pin_spare = {}, []
        return freed

    # --- the parked session ------------------------------------------------------------------
    def _park(self, eng, drafter, keep: int) -> None:
        self._park_busy += 1
        try:
            self._park_now(eng, drafter, keep)
        finally:
            self._park_busy -= 1

    def _park_now(self, eng, drafter, keep: int) -> None:
        """Copy aside the rows of the prompt being replaced, from `keep` (where the new prompt
        parts from it) to its newest anchor that `park_bytes` holds; its newest anchors move with
        them. When every row up to the end anchor fits, the rows go up to it and the end anchor
        moves too, in one of the anchors' places (`park_end`). The previous parked session, if
        any, is given up."""
        views = _kv_views(eng, drafter)
        per_row = sum(t.numel() // t.shape[ax] * t.element_size() for t, ax in views)
        ab = self._anchor_bytes or eng.cost().section_bytes
        cap_rows = max(0, (self.park_budget - self.park_anchor_max * ab) // max(1, per_row))
        end = self._end_to_park(keep, cap_rows)
        if end is not None:
            above = sorted(b for b in self.anchors if keep < b <= self.valid)
            top = self.valid
        else:
            above = sorted(b for b in self.anchors if keep < b <= min(self.valid, keep + cap_rows))
            top = above[-1] if above else keep
        rows = top - keep
        if rows < self.chunk:
            return
        # the regions are the alignment's for this end only while it is `self.end`
        regions = self._end_regions() if end is not None else []
        if self.park_rows:
            self._spill_park(eng, drafter)
        self._drop_park_rows()
        have = self._park_bufs[0].shape[views[0][1]] if (
            self._park_bufs is not None and len(self._park_bufs) == len(views)) else 0
        if have < rows:
            # whole 4-line steps, so a session that grows by a turn does not reallocate each time
            size = min(cap_rows, -(-rows // (4 * self.chunk)) * 4 * self.chunk)
            self._park_bufs = None
            if self.mem is not None and not self.mem.admit("park", size * per_row):
                self.stats["refused_park"] += 1
                return
            self._park_bufs = [torch.empty(*(t.shape[:ax] + (size,) + t.shape[ax + 1:]),
                                           dtype=t.dtype, device=t.device) for t, ax in views]
        for (t, ax), buf in zip(views, self._park_bufs):
            buf.narrow(ax, 0, rows).copy_(t.narrow(ax, keep, rows))
        k = self.park_anchor_max - (end is not None)
        for b in (above[-k:] if k > 0 else []):
            self.park_anchors[b] = self.anchors.pop(b)
            self._allocated -= 1
        if end is not None:
            # the end's buffers go with it; the next generation's end asks for its own
            self.park_end = self._parked_end(end, self._end_bufs)
            self.park_regions = regions
            self._end_bufs, self.end = None, None
            self.stats["park_ends"] += 1
        self.park_tokens = list(self.tokens[:top])
        self.park_base, self.park_rows, self.park_owner = keep, rows, self.owner
        self.park_taint = (self.taint if self.taint is not None and self.taint < top
                           else None)
        self._park_views = [(t, ax) for t, ax in views]
        self.stats["parks"] += 1

    def _end_to_park(self, base: int, cap: int) -> dict | None:
        """The end anchor, when it can be parked with rows [base, end) and the park holds them."""
        end = self.end
        if (end is None or self._end_bufs is None or int(end["at"]) != self.valid
                or not self.chunk <= self.valid - base <= cap):
            return None
        return end

    def _end_regions(self) -> list:
        """The end alignment's regions of the end anchor in place (none without the alignment)."""
        if self.end_extra_of is None or self.end is None:
            return []
        try:
            return [list(x) for x in self.end_extra_of(self)]
        except Exception:                                          # noqa: BLE001
            return []

    @staticmethod
    def _parked_end(end: dict, pair: tuple) -> dict:
        """The end anchor as the park keeps it: `end`'s line, spans, mark (and the image offset,
        the served notes), with its state buffers."""
        out = {"at": int(end["at"]), "hidden": [(int(p), list(h)) for p, h in end["hidden"]],
               "mark": end.get("mark"), "pair": pair}
        for k in ("delta", "mm"):
            if k in end:
                out[k] = end[k]
        if end.get("served_hidden") is not None:
            out["served_hidden"] = list(end["served_hidden"])
        return out

    def _spill_park(self, eng, drafter) -> bool:
        """FX10: the parked conversation a new park is about to give up, to the NVMe first.

        The state store keeps a conversation only under its per-entry cap (1 GiB, about 7,400
        tokens on the served stack: `session snapshot not taken` at every longer turn), and the
        spill wrote a replaced conversation only when it was longer than the RAM park holds above
        the shared head. Everything between went to the park, which holds ONE conversation: a
        third session gave the parked one up, and when it came back it read everything past the
        disk tier's head again (about 740 tokens a second cold; 10,000 tokens, ~13.5 s). Written
        here it costs about 0.1 s a GB on a rested drive (the box's evictions on 2026-10-07: 4.6 GB
        in 0.90 s, 7.3 GB in 1.41 s), incremental by content, under the spill's own budget. Its
        head below `park_base` is the rows in place (`save_park` checks that), so this runs before
        anything is cut; a parked end anchor is written with it, its regions too, as the resident
        conversation's is. `QWEN38_SPILL_PARK=off` gives it up as before."""
        sp = self.spiller
        if sp is None or SPILL_PARK == "off" or not self.park_rows:
            return False
        self._park_busy += 1
        try:
            return self._spill_park_now(sp, eng, drafter)
        finally:
            self._park_busy -= 1

    def _spill_park_now(self, sp, eng, drafter) -> bool:
        top = int(self.park_base + self.park_rows)
        if top < int(sp.cfg.get("drain_min_tokens", 4096)):
            self.stats["park_given_up"] += 1
            return False
        try:
            info = sp.save_park(self, eng, drafter, "evict_park")
        except Exception:                                          # noqa: BLE001
            info = None
        if info is None:
            self.stats["park_given_up"] += 1
            return False
        self.stats["park_spills"] += 1
        return True

    def _drop_park_rows(self) -> None:
        """Forget the parked prompt but keep its row buffers: its anchors and its end's buffers
        join the live pool's free list, as far as the pool's cap allows."""
        for b in list(self.park_anchors):
            self._free.append(self.park_anchors.pop(b))
            self._allocated += 1
        if self.park_end is not None:
            self._free.append(self.park_end["pair"])
            self._allocated += 1
        self.park_end, self.park_regions = None, []
        self.park_tokens, self.park_base, self.park_rows, self.park_owner = [], 0, 0, None
        self.park_taint = None
        self._trim_pool()

    def _trim_pool(self) -> None:
        """The live pool back under its cap: anchors evicted, then free buffers let go. Anchors
        that come back from the park can take it over the cap for a moment."""
        if not self._anchor_bytes:
            return
        cap = max(self.tail + 1, self.budget // self._anchor_bytes)
        while len(self.anchors) > cap:
            self._evict_one()
        while self._allocated > cap and self._free:
            self._free.pop()
            self._allocated -= 1

    def park_offer(self, ids, c: int, max_len: int, align: bool = True,
                   mm_ok: bool = False) -> int:
        """The parked anchor a prompt could resume at, when it beats the rows in place; else 0.
        The head below the parked rows must be the rows in place: `ids` shares it with them.

        With a parked end anchor, its line when the prompt holds every token up to it, or (with
        `align`: a client's copy, before the server's replay) when the copy continues the parked
        conversation to it as the replay will serve it -- the hidden spans put back, or aligned
        by the server's end alignment on the parked rows and regions -- the tests an NVMe entry's
        offer makes (engine/spill.py `Entry.offer`). Such a copy is served as the rows up to the
        end and its own tail after it, so `max_len` (the copy's last position) does not bound it.
        An end after images only with `mm_ok`, as `end_offer`: a request with images whose keys
        hold every row up to it (ENG-371: two workers with screenshots took turns on slot 0, and
        each one's next call read its whole prompt again: its parked end was never offered)."""
        pe = self.park_end
        if not self.park_rows or (not self.park_anchors and pe is None) or c < self.park_base:
            return 0
        top = self.park_base + self.park_rows
        m = min(len(ids), top)
        cp = _common_prefix(list(ids[:m]), self.park_tokens[:m])
        if cp <= self.park_base:
            return 0
        bk = max((b for b in self.park_anchors if b <= min(cp, max_len)), default=0)
        if pe is not None and (not pe.get("mm") or mm_ok):
            e = int(pe["at"])
            if cp >= e and e <= max_len:
                bk = e
            elif align and cp < e and not pe.get("mm") and self._continues_park(ids):
                bk = e
        return bk if bk > self.best(c, max_len) else 0

    def _continues_park(self, ids) -> bool:
        """Whether the server's replay serves `ids` from the parked end anchor."""
        pe = self.park_end
        e = int(pe["at"])
        if pe["hidden"]:
            out, spans = ResidentPrefix.replay(_EndRows(pe, self.park_tokens, self.glue), list(ids))
            if spans and len(out) > e:
                return True
        if self.park_align is None:
            return False
        try:
            return bool(self.park_align(self.park_tokens, pe, self.park_regions, list(ids)))
        except Exception:                                          # noqa: BLE001
            return False

    def unpark_end(self, eng, drafter, ids, max_len: int, mm_ok: bool = False) -> int:
        """Before the server's replay: the parked conversation back in place when the client's
        copy `ids` continues it to its end anchor and that beats every other cache in RAM, so the
        replay (hidden spans, end alignment) works on its rows and the prefill resumes at its end.
        The same order as an NVMe entry's restore (engine/spill.py `maybe_restore`, which calls
        this when no entry beats it). Returns the end's line, 0 when nothing moved."""
        pe = self.park_end
        if pe is None or not self.park_rows or self._stash is not None:
            return 0
        e = int(pe["at"])
        if len(ids) <= e:
            return 0           # FX17: a continuation goes past the end; a shorter copy is another request
        c = self.common(ids)
        if self.park_offer(ids, c, max_len, mm_ok=mm_ok) != e:
            return 0
        if e <= max(self.end_offer(c, max_len, mm_ok=mm_ok), self.pin_offer(ids, c, max_len)):
            return 0
        if not self.unpark(eng, drafter) or self.end is None or int(self.end["at"]) != e:
            return 0
        self.stats["park_tokens_reused"] += e
        return e

    def unpark(self, eng, drafter) -> bool:
        self._park_busy += 1
        try:
            return self._unpark_now(eng, drafter)
        finally:
            self._park_busy -= 1

    def _unpark_now(self, eng, drafter) -> bool:
        """Make the parked prompt the resident one, and park the resident one in its place: the
        two sets of rows are swapped a block at a time, and so are the anchors and the end anchors
        (the one in place is parked with its rows when they fit the park's buffers, as `_park`
        parks it). False, and nothing changed, if the buffers the rows came from are not the ones
        in service any more.

        ENG-371: the park's buffers are as large as the conversation that went in, and one in
        place that is longer lost its end and every row above the buffers here: two workers
        taking turns on slot 0, the longer one's next call read its prompt again from 0 (live
        2026-10-18 01:1x, 15,260 tokens). Its end now goes to the park whole: the buffers grow
        to hold it within `park_bytes` (`_grow_park`), and when the memory policy refuses that,
        the conversation goes to the NVMe first (`_spill`), as an eviction writes it."""
        views = _kv_views(eng, drafter)
        if (self._park_views is None or len(views) != len(self._park_views)
                or any(a is not b or x != y
                       for (a, x), (b, y) in zip(views, self._park_views))):
            self.stats["park_refused"] += 1
            return False
        self._stash = None
        base, ra = self.park_base, self.park_rows
        cap = self._park_bufs[0].shape[views[0][1]]
        # the prompt in place is parked in turn unless it is the parked one's own session
        other = self.owner is None or self.owner != self.park_owner
        if (other and self.end is not None and int(self.end["at"]) == self.valid
                and self.valid - base > cap):
            cap = self._grow_park(eng, drafter, views, self.valid - base, ra)
            if (self.valid - base > cap and self.spiller is not None
                    and self.valid >= int(self.spiller.cfg.get("drain_min_tokens", 4096))):
                self._spill(eng, drafter, "switch")
        end_b = self._end_to_park(base, cap) if other else None
        regions_b = self._end_regions() if end_b is not None else []
        if end_b is not None:
            above = sorted(b for b in self.anchors if base < b <= self.valid)
            rb = self.valid - base
        else:
            above = sorted(b for b in self.anchors if base < b <= min(self.valid, base + cap))
            rb = (above[-1] - base) if above and other else 0
            if rb < self.chunk:
                rb = 0
        # SRV-307: every view's pages before a row moves, so a PoolFull leaves both prompts
        # whole (ENG-371: one between two views left the rows in place half swapped, and a
        # later resume on them was 0.19 off a cold prefill)
        for t, ax in views:
            _kvp.ensure_for(t, base + max(ra, rb))
        if end_b is not None:
            self.end = None                # it moves to the park below, its buffers too
        else:
            self._drop_end()
        step = 4 * self.chunk
        for (t, ax), buf in zip(views, self._park_bufs):
            both = min(ra, rb)
            for j in range(0, both, step):
                w = min(step, both - j)
                tmp = t.narrow(ax, base + j, w).clone()
                t.narrow(ax, base + j, w).copy_(buf.narrow(ax, j, w))
                buf.narrow(ax, j, w).copy_(tmp)
            if rb > ra:
                buf.narrow(ax, ra, rb - ra).copy_(t.narrow(ax, base + ra, rb - ra))
            elif ra > rb:
                t.narrow(ax, base + rb, ra - rb).copy_(buf.narrow(ax, rb, ra - rb))
        back, back_tokens, back_owner = self.park_anchors, self.park_tokens, self.park_owner
        back_taint = self.park_taint
        back_end, back_regions = self.park_end, self.park_regions
        self.park_anchors = {}
        self.park_end, self.park_regions = None, []
        if rb:
            k = self.park_anchor_max - (end_b is not None)
            for b in (above[-k:] if k > 0 else []):
                self.park_anchors[b] = self.anchors.pop(b)
                self._allocated -= 1
            if end_b is not None:
                self.park_end = self._parked_end(end_b, self._end_bufs)
                self.park_regions = regions_b
                self._end_bufs = None
                self.stats["park_ends"] += 1
            self.park_tokens = list(self.tokens[:base + rb])
            self.park_rows, self.park_owner = rb, self.owner
            self.park_taint = (self.taint if self.taint is not None and self.taint < base + rb
                               else None)
            self.stats["park_swaps"] += 1
        else:
            self.park_tokens, self.park_rows, self.park_owner = [], 0, None
            self.park_taint = None
        for b in [b for b in self.anchors if b > base]:
            self._free.append(self.anchors.pop(b))
        for b, pair in back.items():
            self.anchors[b] = pair
            self._allocated += 1
        self.tokens = list(back_tokens)
        self.valid = base + ra
        self.owner = back_owner
        self.taint = back_taint
        if back_end is not None:
            self._adopt_parked_end(eng, back_end, back_regions)
        self._trim_pool()
        self.stats["park_hits"] += 1
        return True

    def _grow_park(self, eng, drafter, views, rows: int, keep: int) -> int:
        """ENG-371: the park's buffers grown to hold `rows` (whole 4-line steps, within
        `park_bytes`), the `keep` rows they hold copied over; their size in rows after it, the old
        size when the memory policy refuses the growth or the budget cannot hold `rows`."""
        ax = views[0][1]
        have = self._park_bufs[0].shape[ax]
        per_row = sum(t.numel() // t.shape[a] * t.element_size() for t, a in views)
        cap_rows = self._park_cap_rows(eng, drafter)
        size = min(cap_rows, -(-rows // (4 * self.chunk)) * 4 * self.chunk)
        if size < rows or size <= have:
            return have
        # both buffers are held while the rows are copied over
        if self.mem is not None and not self.mem.admit("park", size * per_row):
            self.stats["refused_park"] += 1
            return have
        bufs = []
        for (t, a), old in zip(views, self._park_bufs):
            b = torch.empty(*(t.shape[:a] + (size,) + t.shape[a + 1:]), dtype=t.dtype,
                            device=t.device)
            b.narrow(a, 0, keep).copy_(old.narrow(a, 0, keep))
            bufs.append(b)
        self._park_bufs = bufs
        self.stats["park_grown"] = self.stats.get("park_grown", 0) + 1
        return size

    def _adopt_parked_end(self, eng, pe: dict, regions: list) -> None:
        """The parked end anchor is the end in place again: its buffers become the end's (or are
        copied into the end's own and join the free list), its served notes and the alignment's
        regions come back with it."""
        pair = pe["pair"]
        if self._end_bufs is None and self.end_on:
            self._end_bufs, pair = pair, None
        got = self.adopt_end(eng, pe)
        if got is not None and pair is not None:
            state_copy(got, pair)
        if pair is not None:
            self._free.append(pair)
            self._allocated += 1
        if got is None:
            return
        if pe.get("served_hidden") is not None:
            self.end["served_hidden"] = list(pe["served_hidden"])
        if self.end_adopt is not None:
            try:
                self.end_adopt(self, regions)
            except Exception:                                      # noqa: BLE001
                pass
        self.stats["park_end_hits"] += 1

    def drop_session(self, session: str | None) -> int:
        """A compaction of `session` started: its parked rows go. Returns the bytes."""
        if session is not None and self.park_owner == session and self.park_rows:
            self.stats["park_dropped"] += 1
            return self.shed_park()
        return 0

    # --- giving memory back (Budget's levels) ------------------------------------------------
    def shed_anchors(self, safe: bool = False) -> int:
        """The free anchor buffers, and unless `safe`, every anchor but the newest `tail` and the
        head lines: a divergence in the middle of a prompt then resumes lower, or from zero."""
        if not safe:
            keep = set(sorted(self.anchors)[-self.tail:]) if self.anchors else set()
            for b in [b for b in self.anchors if b not in keep and not self.head_line(b)]:
                self._free.append(self.anchors.pop(b))
                self.stats["anchors_evicted"] += 1
        n = len(self._free)
        self._free = []
        self._allocated -= n
        return n * self._anchor_bytes

    def shed_end(self, safe: bool = False) -> int:
        """The end anchor and its buffers: the next turn then resumes at the anchors in place.
        `safe` gives back only buffers that hold no end anchor, and so does `end_hold` (ENG-371):
        the request at the memory gate resumes at this end. Live 2026-10-18 01:58 the gate gave
        back the end a worker's call had just been carried into slot 0 with, and the call read
        its 7,953 tokens from 0; the levels after it (the park, the pin, the roles) go first."""
        if self._end_bufs is None or ((safe or self.end_hold) and self.end is not None):
            return 0
        self._drop_end("memory")
        freed = sum(_nbytes(x) for x in self._end_bufs)
        self._end_bufs = None
        return freed

    def shed_stash(self, safe: bool = False) -> int:
        """The guest stash's buffers. A stash still waiting to be put back is the resident
        prompt's rows: unless `safe`, the prompt is cut where the guest began and they go too."""
        freed = 0
        if self._stash is not None:
            if safe:
                return 0
            self._cut(self._stash["start"])
            self._stash = None
            # the anchors the cut freed are not kept as free buffers either
            freed += len(self._free) * self._anchor_bytes
            self._allocated -= len(self._free)
            self._free = []
        freed += sum(b.numel() * b.element_size() for b in (self._stash_bufs or []))
        self._stash_bufs, self._stash_rows = None, 0
        return freed

    def shed_park(self, safe: bool = False) -> int:
        """The parked session's buffers; nothing while a park, a swap or a park's write runs
        (a KV-pool reclaim inside one of them: the buffers are in use)."""
        if self._park_busy:
            return 0
        freed = sum(b.numel() * b.element_size() for b in (self._park_bufs or []))
        freed += len(self.park_anchors) * self._anchor_bytes
        freed += sum(_nbytes(x) for x in (self.park_end or {}).get("pair", ()))
        self.park_anchors = {}
        self.park_end, self.park_regions = None, []
        self._park_bufs, self._park_views = None, None
        self.park_tokens, self.park_base, self.park_rows, self.park_owner = [], 0, 0, None
        self.park_taint = None
        return freed

    def shed_pin(self, safe: bool = False) -> int:
        return self._drop_pin()

    def set_limits(self, *, budget: int | None = None, stash: int | None = None,
                   guest_max: int | None = None, pin: int | None = None,
                   park: int | None = None) -> None:
        """New caps from the next request on; what is over a lower cap goes now."""
        if budget is not None:
            self.budget = int(budget)
            if self._anchor_bytes:
                cap = max(self.tail + 1, self.budget // self._anchor_bytes)
                while len(self.anchors) > cap:
                    self._evict_one()
                extra = max(0, self._allocated - cap)
                take = min(extra, len(self._free))
                del self._free[len(self._free) - take:]
                self._allocated -= take
        if guest_max is not None:
            self.guest_max = int(guest_max)
        if stash is not None:
            self.stash_budget = int(stash)
            if self._stash is None and self._stash_bufs is not None and sum(
                    b.numel() * b.element_size() for b in self._stash_bufs) > self.stash_budget:
                self.shed_stash()
        if pin is not None:
            self.pin_budget = int(pin)
            if self.pin_bytes > self.pin_budget:
                self._drop_pin()
        if park is not None:
            self.park_budget = int(park)
            if self.park_bytes > self.park_budget:
                self.shed_park()

    # --- bookkeeping -----------------------------------------------------------------------
    def clear(self) -> None:
        for b in list(self.anchors):
            self._free.append(self.anchors.pop(b))
        self.tokens, self.valid = [], 0
        self._stash = None
        self.capturing = False
        self.end, self.taint = None, None

    @property
    def pin_bytes(self) -> int:
        pin = sum(b.numel() * b.element_size() for b in (self._pin_bufs or []))
        return pin + (len(self.pin_anchors) + len(self._pin_spare)) * self._anchor_bytes

    @property
    def park_bytes(self) -> int:
        park = sum(b.numel() * b.element_size() for b in (self._park_bufs or []))
        park += sum(_nbytes(x) for x in (self.park_end or {}).get("pair", ()))
        return park + len(self.park_anchors) * self._anchor_bytes

    @property
    def stash_bytes(self) -> int:
        return sum(b.numel() * b.element_size() for b in (self._stash_bufs or []))

    @property
    def anchor_bytes(self) -> int:
        """The live pool: anchors in use and free ones, not the pin's or the park's."""
        return self._allocated * self._anchor_bytes

    @property
    def end_bytes(self) -> int:
        return sum(_nbytes(x) for x in (self._end_bufs or ()))

    @property
    def bytes(self) -> int:
        return (self.anchor_bytes + self.stash_bytes + self.pin_bytes + self.park_bytes
                + self.end_bytes)

    def report(self) -> dict:
        return {"valid": self.valid, "anchors": sorted(self.anchors),
                "anchor_bytes": self._anchor_bytes, "anchor_buffers": self._allocated,
                "budget": self.budget, "stash_budget": self.stash_budget,
                "stash_rows": self._stash_rows, "stash_pending": self._stash is not None,
                "bytes": self.bytes, "chunk": self.chunk, "tail": self.tail,
                "pin_budget": self.pin_budget, "pin_rows": self.pin_rows,
                "pin_bytes": self.pin_bytes,
                "pin_anchor_list": sorted(self.pin_anchors),
                "anchor_pool_bytes": self.anchor_bytes, "stash_bytes": self.stash_bytes,
                "guest_max": self.guest_max, "owner": _short(self.owner),
                "owner_tag": owner_tag(self.owner), "park_owner_tag": owner_tag(self.park_owner),
                "park_budget": self.park_budget, "park_bytes": self.park_bytes,
                "park_base": self.park_base, "park_rows": self.park_rows,
                "park_owner": _short(self.park_owner),
                "park_anchor_list": sorted(self.park_anchors),
                "park_end_at": self.park_end["at"] if self.park_end else None,
                "end_on": self.end_on, "end_at": self.end["at"] if self.end else None,
                "end_hidden": sum(len(h) for _, h in self.end["hidden"]) if self.end else 0,
                "end_bytes": self.end_bytes, "taint": self.taint,
                "last_switch": self.switches[-1] if self.switches else None,
                "switch_list": list(self.switches), **self.stats}


class _EndRows:
    """What `ResidentPrefix.replay` reads of a conversation that is not in place (the park's):
    its end anchor, its rows' tokens, the server's whitespace tokens."""

    def __init__(self, end, tokens, glue):
        self.end, self.tokens, self.glue = end, tokens, glue
        self.stats = {"end_replayed": 0}


def owner_tag(session: str | None) -> str | None:
    """A session key's tag for a client that knows its own key (chatd's Context panel asks "is
    this chat in the slot?"): the first 16 hex digits of its SHA-256, the key itself not shown."""
    if session is None:
        return None
    import hashlib
    return hashlib.sha256(str(session).encode()).hexdigest()[:16]


def _short(session: str | None) -> str | None:
    """A session key as the dashboard shows it: enough to tell two apart, not the whole id."""
    return None if session is None else (session if len(session) <= 12 else session[:12] + "…")


# --------------------------------------------------------------------------- the prefill itself
def prefill_chunk(prefix_on: bool, prefix_chunk: int, max_rows: int) -> int:
    """The chunk a server's prefill runs at.

    With the prefix cache on it is the cache's grid, as it always was. With it off the engine used
    to forward the whole prompt in ONE call, and on 2026-09-23 a 131,072-token prompt sent that way
    to a test server with a 262,144-token window is the prime suspect for the box running out of
    unified memory and wedging (NVRM NV_ERR_NO_MEMORY at 12:38-13:00): every activation of
    a forward is proportional to its rows. `max_rows` bounds it; 0 restores the single call.
    """
    return prefix_chunk if prefix_on else max(0, int(max_rows))


class _Done:
    """A layer mark on the CPU, where a forward has run when it returns."""

    @staticmethod
    def query() -> bool:
        return True


#: SRV-591: the running window's layer mark on this thread (`_layer_marks`), None between windows
LAYER = threading.local()


def layer_mark() -> None:
    """Between two layers of a layer-major window (engine/model.py calls `WINDOW_YIELD` there; the
    server's calls this): the live view's mark of the layer just issued."""
    mark = getattr(LAYER, "mark", None)
    if mark is not None:
        mark()


def _layer_marks(live: dict, device):
    """SRV-591: a window's layer mark, on the prefill's own thread between two layers: a CUDA
    event recorded behind the layer just issued (a done mark on the CPU, where a forward has run
    when it returns), then the events asked, without waiting, which layers have run: `k` of them
    at `t_k` (the host issues a few layers ahead of the GPU, so this lags by those). The live
    view reads the two numbers only (server/activity.py `window_rate`). The record and the
    queries are CUDA calls, possibly on a side slot's thread: they run only while no graph
    capture holds `capture_lock.LOCK` (engine/capture_lock.py), and the mark is skipped (the
    event too) when one does, so the rate lags by that layer instead."""
    cuda = str(device).startswith("cuda") and torch.cuda.is_available()

    def mark() -> None:
        marks = live["layers"]
        if cuda:
            if not capture_lock.LOCK.acquire(blocking=False):
                return
            try:
                ev = torch.cuda.Event()
                ev.record()
                marks.append(ev)
                k = int(live.get("k") or 0)
                while k < len(marks) and marks[k].query():
                    k += 1
            finally:
                capture_lock.LOCK.release()
        else:
            marks.append(_Done)
            k = len(marks)
        if k > int(live.get("k") or 0):
            live["k"], live["t_k"] = k, time.perf_counter()
    return mark


def _window_plan(eng, i: int, n: int, chunk: int, win: int) -> list | None:
    """The chunks (start, rows) one layer-major window takes from row `i`: the chunk loop's own
    grid, as many chunks as fit in `win` rows; None when fewer than two would (or the engine
    cannot run the window), and the loop forwards one chunk the usual way."""
    if win <= 0 or chunk <= 0 or n - i <= chunk or not hasattr(eng, "prefill_window"):
        return None
    if not eng.window_ok(i):
        return None
    plan, j = [], i
    while j < n:
        t = min(chunk - (j % chunk), n - j)
        if plan and (j + t) - i > max(win, chunk):
            break
        plan.append((j, t))
        j += t
    return plan if len(plan) > 1 else None


def _after_chunk(eng, drafter, res, store, disk, job, ckpt, checkpoint, chunk, i, n, lim, keys,
                 hashes, conv_id, mem) -> None:
    """What the prefill takes at the line `i` a chunk has just reached: the resident anchor, the
    disk's anchor and the store's snapshot (the engine stands at `i`)."""
    if res is not None:
        res.advanced(i)
        if i < n and i % chunk == 0:
            res.anchor(eng, i)
    if job is not None and i < lim and i % chunk == 0:
        _disk(disk, "capture", job, eng, i)
    if (store is not None and checkpoint and i < lim and chunk > 0 and i % chunk == 0
            and (ckpt is None or i in ckpt)):
        # room is made BEFORE the clone (OPS-47): evicting after it held store + snapshot
        est = _snapshot_estimate(eng, i, drafter)
        snap = None
        if not store.reserve(est):
            store.stats["skipped_big"] = store.stats.get("skipped_big", 0) + 1
        elif mem is not None and not mem.admit("store", est):
            store.stats["skipped_mem"] = store.stats.get("skipped_mem", 0) + 1
        else:
            snap = capture(eng, drafter, max_bytes=store.max_entry)
            if snap is not None:
                store.put(keys, snap, conv_id, hashes)
            else:
                store.stats["skipped_big"] = store.stats.get("skipped_big", 0) + 1


def prefill(eng, drafter, ids: list[int], device, *, store: StateStore | None = None,
            chunk: int = 0, conv_id: str | None = None, checkpoint: bool = False,
            resident: "ResidentPrefix | None" = None, on_chunk=None, info: dict | None = None,
            key_ids: list[int] | None = None, mem: "Budget | None" = None,
            compacting: bool = False, checkpoint_upto: int | None = None,
            disk=None, disk_tag: dict | None = None, end_mm: bool = False,
            scratch: bool = False, reach: int | None = None, stay: bool = False):
    """Bring the engine to `len(ids)` and return the logits of the last position.

    Returns `(logits, reused, forwarded)`. `reused` is how many prompt tokens came out of the
    store or the resident prefix and `forwarded` how many were actually run through the 64
    layers -- the two numbers the TTFT of a warm request is made of.

    `chunk = 0` forwards the whole remainder in one call, which is what the engine did before this
    file existed and what a run with no prefix cache still does.

    `resident` (the live buffer as a cache, `ResidentPrefix`) is asked first; a store entry is
    used only when it is LONGER than the resident anchor, because a restore copies the KV and a
    resident resume does not. `on_chunk(done, total)` is called after every chunk -- the server's
    hook for noticing a client that left (it raises, and the prefill stops there with every
    anchor it took so far still good). `info["kind"]` says where the state came from.

    `key_ids`: what the caches compare instead of `ids` -- the prompt with each image's
    placeholder rows replaced by an id derived from the image's content (engine/vision.py). The
    placeholders alone are the same for every image; the rows the engine writes for them are not.
    None for text, where the key is the prompt itself.

    `conv_id` is the session key (the client's conversation id or opencode's `x-session-id`): the
    store files its entries under it and the resident prefix knows whose rows it holds, which is
    what parks a session another one replaces and brings it back (`ResidentPrefix.unpark`).
    `compacting`: this request is that session's compaction, whose old rows are not parked. `mem`
    (`Budget`): every snapshot the store would clone is asked for first, and the forwards'
    allocator peak is measured against the workspace estimate.

    `checkpoint_upto`: the store's snapshots only at lines at or below it. The conductor's
    in-context stage (server/conductor/incontext.py) prefills the turn's prompt up to its last
    message and then a question; the lines up to the parting point are the turn's own, and a
    snapshot above it would hold the question.

    `disk` (engine/disk_cache.py `DiskTier`, the NVMe under the caches in RAM): asked after them,
    and taken only where it starts later than all of them by more than its read costs; its rows
    stream into the buffers in place and its anchor becomes a resident one. Before the forwards
    it decides what this prefill writes (lines prefilled `admit_after` times, a session's first
    long request), an anchor's state is copied out as the prefill passes its line, and the rows
    are written behind it until the next prefill fences them. `disk_tag`: the request's client
    and its injected notes (`spans`), which name an entry on the dashboard. None: every line
    below runs as it did without the tier.

    The resident prefix's end anchor (`ResidentPrefix.keep_end`): a prompt that starts with every
    token of the last generation resumes at its end when that beats the anchors in place, the
    pin and the park; a store entry or the disk is taken only when it starts later still. Such a
    prefill, and one from an anchor above the resident's `taint`, writes a session's rows: no
    prefix snapshot and no disk write is taken of it, and the disk reads over those rows.

    `end_mm`: this prompt may resume at an end anchor kept after images
    (`ResidentPrefix.keep_end`); the caller has set the engine's rotary offset to that end's.
    `scratch`: the rows written are nobody's (`ResidentPrefix.begin`); no prefix snapshot is taken.
    `stay`: the session comes back (an agent's worker, ENG-371): never a guest of the resident.
    """
    n = len(ids)
    if scratch:
        checkpoint = False
    if n == 0:
        raise ValueError("a prefill needs at least one token")
    t_in = time.perf_counter()
    keys = ids if key_ids is None else key_ids
    if len(keys) != n:
        raise ValueError(f"key_ids has {len(keys)} entries for a {n}-token prompt")
    hashes = _hash_memo(keys)
    res = resident if (resident is not None and chunk > 0 and resident_capable(drafter)) else None
    disk = disk if (disk is not None and chunk > 0 and resident_capable(drafter)) else None
    start, kind, c, b = 0, None, 0, 0
    if res is not None:
        if disk is not None and res._stash is not None:
            _disk(disk, "fence", 0)          # a guest's rows go back: a job of its copies none
        res.settle_guest(eng)
        if res.spiller is not None:
            # a conversation on the disk (engine/spill.py) that beats every cache in RAM is read
            # back in place first; the server asks before its replay too (text prompts)
            try:
                got = res.spiller.maybe_restore(res, eng, drafter, keys, n - 1, disk=disk,
                                                mm_ok=end_mm)
            except Exception:                                      # noqa: BLE001
                got = None
            if got and info is not None:
                info["spill"] = got
        c = res.common(keys)
        b = res.best(c, n - 1)
        was_owner, was_valid, was_c = res.owner, res.valid, c
    bp = res.pin_offer(keys, c, n - 1) if res is not None else 0
    # the server has replayed the client's copy already: the park's end only for every token
    bk = res.park_offer(keys, c, n - 1, align=False, mm_ok=end_mm) if res is not None else 0
    be = res.end_offer(c, n - 1, mm_ok=end_mm) if res is not None else 0
    hit, via, dh = None, None, None
    if store is not None:
        hit = store.find(keys, hashes, max_len=n - 1, min_len=max(b, bp, bk, be))
    if disk is not None:
        # after every cache in RAM: the disk only where it starts later than all of them, and
        # over the rows a session's state wrote (from `taint`), which are not a cold grid's
        have = c if res is None or res.taint is None else min(c, res.taint)
        dh = _disk(disk, "find", keys, hashes, n - 1, max(b, bp, bk, be, hit[0] if hit else 0),
                   have)
        if dh is not None:
            hit = None
    if hit is None and dh is None and max(bp, bk) > be:
        # the parked session or the pinned head beats the rows in place and every store entry
        if disk is not None:
            _disk(disk, "fence", 0)
        if bk >= bp and res.unpark(eng, drafter):
            via = "park"
        elif bp and res.unpin(eng, drafter):
            via = "pin"
        if via is not None:
            c = res.common(keys)
            b = res.best(c, n - 1)
            # the parked end came back with its rows (`unpark`); the pin brings none
            be = res.end_offer(c, n - 1, mm_ok=end_mm)
            res.stats[f"{via}_tokens_reused"] += max(b, be)
        elif store is not None:
            hit = store.find(keys, hashes, max_len=n - 1, min_len=b)
    if dh is not None:
        start, kind = dh.L, "disk"
    elif hit is not None:
        L, entry = hit
        start, kind = L, entry.kind
    elif be:
        start, kind = be, "end"
    elif b > 0:
        start, kind = b, "resident"
    if disk is not None:
        # the rows this prefill leaves as they are: a write still copying them may go on
        _disk(disk, "fence", start if kind in ("resident", "end")
              else dh.have if dh is not None else 0)
    if info is not None:
        info["why"] = (reread_words(res, c, start, n, kind, was_valid,
                                    other=conv_id is not None and was_owner is not None
                                    and conv_id != was_owner)
                       if res is not None and n - start >= REREAD_WORDS_TOKENS else None)
    taint = res.taint if res is not None else None
    if res is not None:
        # decided before any row is written: a guest's stash must hold the rows as they are now
        res.begin(eng, drafter, keys, start, c, kind, session=conv_id, compacting=compacting,
                  scratch=scratch, reach=reach, stay=stay)
        res.end_hold = False               # ENG-371: held from the server's memory gate to here
        if kind in ("prefix", "disk"):
            # the restore writes over the rows below it. dropped after `begin`, whose
            # eviction writes the conversation to the NVMe with its end anchor first; dropped
            # before it, the conversation went to the disk without its end and its
            # next turn re-read the tokens above its last grid anchor (live 13:34: 759 tokens)
            res._drop_end()
    if hit is not None:
        restore(eng, hit[1].snap, drafter)
    elif dh is not None:
        if _disk(disk, "restore", eng, drafter, dh, default=False):
            if res is not None:
                res.anchor(eng, start)        # the disk's state is the resident prompt's anchor
        else:
            # nothing was restored that the rows in place did not hold already: the cold path
            start = res.best(c, n - 1) if res is not None else 0
            kind = "resident" if start else None
            if res is not None and res.capturing:
                res.valid = start
            if res is not None and taint is not None and start > taint:
                res.taint = taint             # the anchor it falls back to is a session's
            if start:
                res.resume(eng, drafter, start)
            else:
                eng.reset()
    elif kind == "resident":
        res.resume(eng, drafter, start)
    elif kind == "end":
        res.resume_end(eng, drafter, saved=start - b)
    else:
        eng.reset()
    if res is not None and kind not in ("resident", "end"):
        res.stats["misses"] += 1
    if info is not None:
        info["kind"] = kind
        info["via"] = via                  # the park or the pin the rows came back from, or None
        # what the live view needs to turn the chunk hook's `done` into progress and a
        # rate -- once per prefill, before the loop; the loop itself gains nothing
        info["start"], info["chunk"], info["t0"] = start, int(chunk), time.perf_counter()
    logits = None
    i = start
    lim = n if checkpoint_upto is None else min(n, int(checkpoint_upto) + 1)
    if res is not None and res.taint is not None:
        # a session's rows (an end anchor's prefill, or one above `taint`): no prefix snapshot,
        # and nothing for the disk, whose files are a cold grid prefill's bytes
        checkpoint, kind_disk = False, "session"
    else:
        kind_disk = kind
    ckpt = (store.boundaries(eng, drafter, start, lim, chunk)
            if (store is not None and checkpoint and chunk > 0) else None)
    job = (_disk(disk, "plan", keys, hashes, start, n, lim, kind_disk, conv_id, disk_tag)
           if disk is not None and not scratch else None)
    if mem is not None:
        mem.mark()
    lines = res is not None or job is not None or (store is not None and checkpoint and chunk > 0)
    win = PREFILL_WINDOW if lines else PREFILL_WINDOW_FREE
    try:
        while i < n:
            cap = WINDOW_CAP() if WINDOW_CAP is not None else None
            plan = _window_plan(eng, i, n, chunk, win if cap is None else min(win, int(cap)))
            if plan is not None:
                # LAYER-major over several chunks (`Qwen38Engine.prefill_window`): the same forwards
                # bit for bit, every EXL3 slab decoded once a window instead of once a chunk. The
                # chunk loop below then runs for each chunk in order with the engine put back at
                # that chunk's line, so every anchor, snapshot and drafter sync is the one a
                # chunk-major prefill takes.
                bufs = None
                if lines:
                    bufs = [state_empty(eng) for _ in plan]
                toks = [torch.tensor(ids[a:a + t], device=device, dtype=torch.long)
                        for a, t in plan]
                live = None
                if info is not None:
                    # SRV-591: the window's progress while it runs, one mark between two layers
                    # (`window_live`, server/activity.py), so a prefill of one window has a rate
                    # before its end
                    live = info["window_live"] = {
                        "rows": (plan[0][0], plan[-1][0] + plan[-1][1]),
                        "t0": time.perf_counter(), "layers": [],
                        "n": int(getattr(getattr(eng, "cfg", None), "num_hidden_layers", 0) or 0)}
                    LAYER.mark = _layer_marks(live, device)
                try:
                    logits, hid, taps = eng.prefill_window(toks, [a for a, _ in plan], bufs)
                finally:
                    if live is not None:
                        LAYER.mark = None
                if info is not None:
                    # the window's rows were all computed by now: the chunk hook below reports
                    # them chunk by chunk within milliseconds, so the live view times them here
                    # (rows a, end, the host's clock once the window's forwards were issued)
                    info["window"] = (plan[0][0], plan[-1][0] + plan[-1][1], time.perf_counter())
                    info["window_live"] = None
                if bufs is not None:
                    state_save(bufs[-1], eng)
                end = plan[-1][0] + plan[-1][1]
                for c, (a, t) in enumerate(plan):
                    last = c == len(plan) - 1
                    if bufs is not None:
                        state_load(eng, bufs[c])
                    eng.kv.length = a + t
                    eng.replay_taps(taps[c])
                    if drafter is not None and hasattr(drafter, "sync"):
                        drafter.sync(ids[a:a + t], hid[c][0], a)
                    i = a + t
                    _after_chunk(eng, drafter, res, store, disk, job, ckpt, checkpoint, chunk, i, n,
                                 lim, keys, hashes, conv_id, mem)
                    if on_chunk is not None and i < n:
                        on_chunk(i, n)
                del bufs, toks, hid, taps
                assert i == end
                continue
            if info is not None and info.get("window") is not None:
                info["window"] = None                  # a chunk-major forward times itself
            if chunk > 0:
                # The grid is anchored at absolute position 0, not at the resume point: a resume
                # that lands off the grid -- which is what a session boundary is -- takes a SHORT
                # first step and is back on the grid afterwards. Anchoring it at `start` instead
                # would put every checkpoint of that prefill off the grid too, and the next
                # request resuming from one of those would inherit the misalignment for ever. This
                # way the only forward that differs from a cold run's is the first one after a
                # session resume.
                t = min(chunk - (i % chunk), n - i)
            else:
                t = n - i
            blk = torch.tensor(ids[i:i + t], device=device, dtype=torch.long)
            # a chunk before the last one returns logits nobody reads: one row, not `t` of them
            logits = eng.forward(blk, start=i, last_only=True)
            if drafter is not None and hasattr(drafter, "sync"):
                drafter.sync(ids[i:i + t], eng.hidden_post_norm[0], i)
            i += t
            _after_chunk(eng, drafter, res, store, disk, job, ckpt, checkpoint, chunk, i, n, lim,
                         keys, hashes, conv_id, mem)
            if on_chunk is not None and i < n:
                on_chunk(i, n)
    finally:
        if job is not None:
            # what the prefill wrote, to the writer -- also when a client that left stopped it
            _disk(disk, "seal", job, eng, drafter, i)
    if mem is not None:
        mem.prefilled(n)
    if store is not None:
        store.stats["tokens_reused"] += start
        store.stats["tokens_forwarded"] += n - start
    if res is not None and was_valid and res._stash is None and (
            (conv_id is not None and conv_id != was_owner)
            or (conv_id is None and was_c < was_valid)):
        # another session's rows were in place (or, without session ids, the prompt parted from
        # them): what this switch cost, for the dashboard
        res.stats["switches"] += 1
        res.switches.append({"ts": round(time.time(), 3), "from": _short(was_owner),
                             "to": _short(conv_id), "tokens": n, "reused": start,
                             "reread": n - start, "via": via or kind or "none",
                             "ms": round((time.perf_counter() - t_in) * 1e3, 1)})
    return logits, start, n - start


#: SRV-546: a prefill that reads at least this many tokens again says why, in words the chat's
#: Now line shows (`reread_words`, the live view's `prefill.why`)
REREAD_WORDS_TOKENS = 8192


def reread_words(res, c: int, start: int, n: int, kind, was_valid: int,
                 other: bool = False) -> str:
    """Why a prefill reads `n - start` of its `n` tokens again, in plain words: the rows in place
    (`c` of them shared with the prompt, `was_valid` held, `other`: another session's) and the end
    anchor, before this prefill's `begin` changes them."""
    e = int(res.end["at"]) if res.end is not None else 0
    if kind in ("disk", "prefix") and start:
        return f"the chat came back from the cache tier at {start:,} of {n:,} tokens"
    if not was_valid or c < res.chunk:
        return ("nothing of this chat was cached: a new chat, a restart, or another chat used "
                "the engine's slot since")
    if other:
        return "another chat's conversation was in the engine's slot; this one was not kept there"
    if res.end is not None and c < e:
        return f"the chat's text differs from what the engine read, at token {c:,} of {e:,}"
    if res.end is None and getattr(res, "end_lost", None) == "memory":
        return "the engine gave the last turn's cached state back to free memory"
    return f"the cache held {start:,} of {n:,} tokens"


def _disk(disk, op: str, *a, default=None):
    """One call into the disk tier (engine/disk_cache.py). The tier never fails a prefill: an
    error is its journaled event, and the answer is the one without the tier."""
    try:
        return getattr(disk, op)(*a)
    except Exception as exc:                                       # noqa: BLE001
        try:
            disk._event("error", op=op, error=f"{type(exc).__name__}: {exc}"[:300])
        except Exception:                                          # noqa: BLE001
            pass
        return default


# --------------------------------------------------------------------------- response cache
class ResponseCache:
    """(prompt, params) -> the token ids the engine wrote. Off unless asked for.

    Greedy decoding is a function, so this is memoisation and not a heuristic. It is nevertheless
    opt-in, because a server that answers from memory answers a changed MODEL from memory too, and
    because the one thing a benchmark must never measure is a dictionary.
    """

    def __init__(self, budget_bytes: int = 64 << 20, ttl_s: float = 3600.0):
        self.budget = int(budget_bytes)
        self.ttl = float(ttl_s)
        self._d: OrderedDict = OrderedDict()
        self.bytes = 0
        self.stats = {"hits": 0, "misses": 0, "expired": 0, "evictions": 0, "puts": 0}
        self.lock = threading.Lock()

    @staticmethod
    def key(ids, **params) -> tuple:
        h = prefix_hashes(ids)[-1]
        return (len(ids), h, tuple(sorted((str(k), repr(v)) for k, v in params.items())))

    def get(self, key):
        with self.lock:
            rec = self._d.get(key)
            if rec is None:
                self.stats["misses"] += 1
                return None
            ids, tokens, born = rec
            if self.ttl > 0 and time.time() - born > self.ttl:
                del self._d[key]
                self.bytes -= 8 * len(ids) + 8 * len(tokens)
                self.stats["expired"] += 1
                self.stats["misses"] += 1
                return None
            self._d.move_to_end(key)
            self.stats["hits"] += 1
            return ids

    def put(self, key, ids, prompt_ids) -> None:
        if not ids:
            return
        cost = 8 * len(ids) + 8 * len(prompt_ids)
        with self.lock:
            if key in self._d:
                old = self._d.pop(key)
                self.bytes -= 8 * len(old[0]) + 8 * len(old[1])
            self._d[key] = (list(ids), list(prompt_ids), time.time())
            self.bytes += cost
            self.stats["puts"] += 1
            while self.bytes > self.budget and self._d:
                _, old = self._d.popitem(last=False)
                self.bytes -= 8 * len(old[0]) + 8 * len(old[1])
                self.stats["evictions"] += 1

    def report(self) -> dict:
        with self.lock:
            return {"entries": len(self._d), "bytes": self.bytes, "budget": self.budget,
                    "ttl_s": self.ttl, **self.stats}

    def clear(self) -> None:
        with self.lock:
            self._d.clear()
            self.bytes = 0


# --------------------------------------------------------------------------- suffix stores
DOC_SEP = 1 << 30


def _common_prefix(a: list, b: list, step: int = 4096) -> int:
    """How many leading elements two lists share: slice compares a block at a time, then the
    block that differs element by element, so the cost is one pass over the shared part."""
    m = min(len(a), len(b))
    i = 0
    while i < m:
        j = min(m, i + step)
        if a[i:j] != b[i:j]:
            while a[i] == b[i]:
                i += 1
            return i
        i = j
    return m


class SuffixStoreSet:
    """Several `CorpusSuffixStore`-shaped stores behind one `lookup`.

    A position is an opaque handle to the drafter -- it only ever hands one back to
    `continuation` -- so a store index is packed into its high bits and unpacked on the way out.
    The longest match wins; where two stores match at the same order, both vote, which is what the
    drafter's counting wants.
    """

    STRIDE = 1 << 40

    def __init__(self, stores):
        self.stores = [s for s in stores if s is not None]
        self.max_order = max((getattr(s, "max_order", 8) for s in self.stores), default=8)

    def __bool__(self) -> bool:
        return bool(self.stores)

    def lookup(self, context, min_order: int, max_samples: int = 64):
        best, pos = 0, []
        for i, s in enumerate(self.stores):
            n, p = s.lookup(context, min_order, max_samples=max_samples)
            if not p or n < best:
                continue
            if n > best:
                best, pos = n, []
            pos.extend(i * self.STRIDE + q for q in p)
        return best, pos

    def continuation(self, pos: int, k: int):
        i, p = divmod(pos, self.STRIDE)
        return self.stores[i].continuation(p, k)


class PersistentSuffixStore:
    """An append-only token log of what this engine has read and written, with an index over it.

    The lookup drafter's local index dies with the request and its corpus store is a file somebody
    built by hand. This is the third thing: what the engine itself saw, kept across restarts, so
    that the second time it is asked about a subject it drafts the phrasing it used the first time
    at 0.2 ms a lookup instead of 4 ms of prediction head.

    On disk: `tokens.bin`, int32 little-endian, documents separated by `DOC_SEP`; `meta.json`. No
    text, ever -- the same rule `tools/build_corpus.py` states, for the same reason.

    The suffix array is rebuilt in a background thread when enough has accumulated, and the drafter
    is handed the new store by reference assignment. Until then the new tokens are simply not
    findable, which is the right failure: a drafter that cannot find something declines.
    """

    def __init__(self, path: str, *, max_tokens: int = 48_000_000,
                 rebuild_every: int = 100_000, max_order: int = 8, readonly: bool = False,
                 tokenizer_sha: str | None = None):
        self.path = os.path.expanduser(path)
        self.tokenizer_sha = tokenizer_sha        # recorded on create, checked on open
        # A store that is read but never written: the instrument for measuring a fixed benchmark
        # against real traffic's store without the benchmark writing itself into it.
        self.readonly = bool(readonly)
        self.max_tokens = int(max_tokens)
        self.rebuild_every = int(rebuild_every)
        self.max_order = int(max_order)
        self.store = None                  # a CorpusSuffixStore, swapped in atomically
        self.pending = 0
        self.n_tokens = 0
        self.lock = threading.Lock()
        self._thread: threading.Thread | None = None
        # bumped by clear(): a rebuild that read the log before it does not swap its index in or
        # write its trim back
        self._gen = 0
        self.stats = {"appended": 0, "rebuilds": 0, "trims": 0, "rebuild_ms": 0.0,
                      "rebuild_ms_total": 0.0, "rebuild_last_ts": 0.0, "turns": 0,
                      "turn_tokens": 0, "turn_shared": 0}
        # the prompts of the last few turns, to append only what a new turn adds to one of them
        self._recent: list[list[int]] = []
        self.recent_max = 4
        # bumped by every lookup that matched: the metrics wrapper reads it once a block to count
        # the blocks the store had continuations for, never once a lookup
        self.matched = 0

    # --- disk ------------------------------------------------------------------------------
    @property
    def _bin(self) -> str:
        return os.path.join(self.path, "tokens.bin")

    def open(self) -> "PersistentSuffixStore":
        """Create the directory if needed and build the index over whatever is already there."""
        os.makedirs(self.path, exist_ok=True)
        meta = os.path.join(self.path, "meta.json")
        if not os.path.exists(meta):
            m = {"max_order": self.max_order, "doc_sep": DOC_SEP,
                 "what": "token ids written and read by this engine; no text"}
            if self.tokenizer_sha:
                m["tokenizer_sha256"] = self.tokenizer_sha
            with open(meta, "w") as f:
                json.dump(m, f)
        else:
            from engine.tokfp import KEY, check_store
            with open(meta) as f:
                m = json.load(f)
            check_store(m, self.tokenizer_sha, self.path)
            if self.tokenizer_sha and KEY not in m and not self.readonly:
                # written before the fingerprint existed, by this engine and this model: from now
                # on it is recorded, so another tokenizer's store can be told apart
                m[KEY] = self.tokenizer_sha
                with open(meta + ".tmp", "w") as f:
                    json.dump(m, f)
                os.replace(meta + ".tmp", meta)
        os.chmod(self.path, 0o700)
        self.rebuild(background=False)
        return self

    def append(self, tokens) -> None:
        """Add one document. Returns immediately; the index catches up on its own."""
        if not tokens or self.readonly:
            return
        buf = struct.pack(f"<{len(tokens) + 1}i", *(int(t) for t in tokens), DOC_SEP)
        with self.lock:
            os.makedirs(self.path, exist_ok=True)
            with open(self._bin, "ab") as f:
                f.write(buf)
            self.pending += len(tokens) + 1
            self.stats["appended"] += len(tokens)
            due = self.pending >= self.rebuild_every
        if due:
            self.rebuild(background=True)

    def append_turn(self, prompt, out) -> int:
        """Add one request: its prompt and its output, minus what an earlier turn already added.

        An agent client sends one conversation that only grows, and appending the whole prompt
        after every turn wrote about 200 copies of it: 397.5M tokens appended in the 2026-09-30
        night for about 4.8M new ones, a suffix-array rebuild after every turn (607, 17-28 s
        each), and 16 trims that dropped every other session's content. A prompt that starts with
        the prompt of one of the last `recent_max` turns appends only from where they part, with
        the `max_order` tokens before that point in front, so every n-gram across the join is
        still in the store. Returns the number of tokens appended."""
        prompt = list(prompt)
        shared, which = 0, -1
        for j, doc in enumerate(self._recent):
            c = _common_prefix(prompt, doc)
            if c > shared:
                shared, which = c, j
        start = max(0, shared - self.max_order) if shared else 0
        doc = prompt[start:] + list(out)
        self.append(doc)
        if which >= 0:
            self._recent.pop(which)
        self._recent.append(prompt)
        del self._recent[:-self.recent_max]
        self.stats["turns"] += 1
        self.stats["turn_tokens"] += len(prompt) + len(out)
        self.stats["turn_shared"] += shared
        return len(doc) if not self.readonly else 0

    def clear(self) -> int:
        """Forget everything (reset level 3 beside a running engine, SRV-329): the log is truncated
        under the store's lock, so an append never meets a missing file, and the index goes; the
        folder and meta.json stay. Returns the bytes the log held."""
        with self.lock:
            try:
                n = os.path.getsize(self._bin)
            except OSError:
                n = 0
            os.makedirs(self.path, exist_ok=True)
            with open(self._bin, "wb"):
                pass
            self._gen += 1
            self.store = None
            self.n_tokens = self.pending = 0
            self._recent = []
        return n

    def _read_tokens(self, gen: int | None = None):
        import numpy as np
        if not os.path.exists(self._bin):
            return np.zeros(0, dtype=np.int64)
        a = np.fromfile(self._bin, dtype="<i4").astype(np.int64)
        if a.shape[0] > self.max_tokens:
            # The cap forgets the oldest half rather than the oldest token, so trimming is a rare
            # rewrite and not one per request. The cut is moved forward to the next document
            # boundary, so no continuation is left starting in the middle of a document.
            cut = a.shape[0] - self.max_tokens // 2
            nxt = np.nonzero(a[cut:] >= DOC_SEP)[0]
            cut = cut + int(nxt[0]) + 1 if nxt.size else cut
            a = a[cut:]
            with self.lock:
                if gen is not None and gen != self._gen:
                    return np.zeros(0, dtype=np.int64)     # cleared meanwhile: write nothing back
                a.astype("<i4").tofile(self._bin)
            self.stats["trims"] += 1
        return a

    def rebuild(self, background: bool = True) -> None:
        if background:
            with self.lock:
                if self._thread is not None and self._thread.is_alive():
                    return
                self._thread = threading.Thread(target=self._rebuild_now, daemon=True,
                                                name="suffix-store-rebuild")
                self._thread.start()
            return
        self._rebuild_now()

    def _rebuild_now(self) -> None:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from engine.drafters.ngram import CorpusSuffixStore
        from tools.build_corpus import build_suffix_array
        t0 = time.perf_counter()
        gen = self._gen
        toks = self._read_tokens(gen)
        if toks.shape[0] < self.max_order:
            return
        sa = build_suffix_array(toks, max_order=self.max_order)
        store = CorpusSuffixStore(toks, sa, max_order=self.max_order,
                                  meta={"doc_sep": DOC_SEP, "max_order": self.max_order})
        ms = (time.perf_counter() - t0) * 1e3
        with self.lock:
            if gen != self._gen:
                return                             # cleared while this index was built
            self.store = store
            self.n_tokens = int(toks.shape[0])
            self.pending = 0
            self.stats["rebuilds"] += 1
            self.stats["rebuild_ms"] = ms
            self.stats["rebuild_ms_total"] += ms
            self.stats["rebuild_last_ts"] = time.time()

    # --- the store interface the drafter speaks ---------------------------------------------
    def lookup(self, context, min_order: int, max_samples: int = 64):
        s = self.store
        if s is None:
            return 0, []
        n, pos = s.lookup(context, min_order, max_samples=max_samples)
        if pos:
            self.matched += 1
        return n, pos

    def continuation(self, pos: int, k: int):
        s = self.store
        return s.continuation(pos, k) if s is not None else []

    def report(self) -> dict:
        with self.lock:
            return {"path": self.path, "tokens": self.n_tokens, "pending": self.pending,
                    "max_tokens": self.max_tokens, "indexed": self.store is not None,
                    "matched": self.matched, **self.stats}
