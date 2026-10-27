"""An OpenAI-compatible server over this engine, standard library only.

The engine holds one sequence: one recurrent state per linear-attention layer, one KV buffer, one
drafter cache. So the server is honest about that -- requests are serialised behind a lock, and
`/v1/models` and the usage counts are the only places it pretends to be a fleet. Concurrency
numbers from this server are queueing numbers, and the docs say so.

What it exists for: the bench harness this engine is measured against speaks
`POST /v1/chat/completions` with `stream: true` and `stream_options: {"include_usage": true}`,
reads time-to-first-token from the first content delta and inter-token latency from the gaps
between deltas, and takes the token counts from the usage frame. Any of those missing turns a
measured row into an estimated one, so all three are produced exactly.

Streaming emits one chunk per token rather than per buffer flush, because the gaps between chunks
are the measurement.

    python server/app.py --port 8000 --drafter mtp --depth 3
    QWEN38_NVFP4=~/nvfp4/mlp-clip.safetensors python server/app.py
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import hashlib
import json
import re
import os
import select
import signal
import socket
import subprocess
import sys
import threading
import time
import types
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import cache  # noqa: E402
from engine import kvpool  # noqa: E402  (SRV-307 the shared KV pool)
from engine import capture_lock  # noqa: E402  (no CUDA call inside a graph capture)
from engine.settings import SETTINGS as _S  # noqa: E402  (every QWEN38_* knob)
from engine.drafters import tree_steps  # noqa: E402
from engine.methods import walk  # noqa: E402  (ENG-E8 the walk every model's blocks take)
from engine.spec import Relax, ThinkBudget  # noqa: E402
from engine.penalty import PatternStop, PenaltySpec, PenaltyState  # noqa: E402
from engine.sample import Sampler  # noqa: E402
from server.tokcache import SegmentCache, opens_think_ids  # noqa: E402
from server.stream import (  # noqa: E402
    CLOSE_THINK, OPEN_THINK, AnswerStops, Detokenizer, Reasoning, StopStrings, StrayClose,
    answer_stops, drop_stray_close, opens_think, split_full,
)
from engine.tree import DraftTree  # noqa: E402
from server.toolcall import (JSON_ANY, OPEN as TOOL_OPEN, PythonicCalls, ToolCallBuffer,  # noqa: E402
                             answer_items, parse_tool_calls, rewrite_pythonic, schema_types, strip_blocks)
from server import compat  # noqa: E402
from server import thinking as thinking_mod  # noqa: E402
from engine import grammar as grammar_mod  # noqa: E402
from server import logprobs as lp_mod  # noqa: E402
from server import metrics  # noqa: E402
from server import usage as usage_mod  # noqa: E402
from server import ledger as ledger_mod  # noqa: E402
from server import endalign  # noqa: E402
from server import argloop as argloop_mod  # noqa: E402
from server import regenerate as regenerate_mod  # noqa: E402
from server import skillcanon as skillcanon_mod  # noqa: E402
from server import dashboard_api  # noqa: E402
from server import live as live_mod  # noqa: E402
from server import parallel as parallel_mod  # noqa: E402
from server import activity as activity_mod  # noqa: E402
from server import logbuf  # noqa: E402
from server import auth as auth_mod  # noqa: E402
from server import static  # noqa: E402
from server import images as images_mod  # noqa: E402
from server import memlimits  # noqa: E402
from server import memsplit  # noqa: E402
from server import rolemem  # noqa: E402
from server import refwatch as refwatch_mod  # noqa: E402
from server import decider_api  # noqa: E402
from server import laya_api  # noqa: E402
from server import probe as probe_mod  # noqa: E402
from server import diskcache as diskcache_mod  # noqa: E402
from server import library_api  # noqa: E402
from server import cache_peek, chat_forward, model_card, tokenize_api  # noqa: E402
from server import memory_api  # noqa: E402
from server import memblock  # noqa: E402
from server import compaction as compaction_mod  # noqa: E402
from server import session_tokens as session_tokens_mod  # noqa: E402
from server import preempt  # noqa: E402
from server import queue_wait as qwait  # noqa: E402
from server.conductor import monitor as conductor  # noqa: E402
from server.conductor.monitor.policy import (  # noqa: E402
    CONTINUE_HIDDEN, GIVE_UP, REGENERATE_CALL,
)
from server.conductor import routes as conductor_routes  # noqa: E402
from server.conductor import memory as conductor_memory  # noqa: E402
from server.conductor import narrator as narrator_mod  # noqa: E402
from server.conductor import voice as voice_mod  # noqa: E402
from server.conductor import schema as conductor_schema  # noqa: E402
from server.conductor import shadow as shadow_mod  # noqa: E402
from server.conductor import incontext as incontext_mod  # noqa: E402
from server.conductor import outcomes as conductor_outcomes  # noqa: E402
from server.conductor import states as states_mod  # noqa: E402
from server.conductor import loopescape as loopescape_mod  # noqa: E402
from server.conductor import redact as redact_mod  # noqa: E402
from server.conductor.classify import USER_AGENT as CLASSIFY_UA  # noqa: E402
from server.conductor.store import ConfigError as ConductorConfigError  # noqa: E402
from server.conductor.router import COMPOSITE as COMPOSITE_NAME  # noqa: E402
from server.conductor.schema import SIDE_FIRST_MAX_TOKENS  # noqa: E402

from server import slots as slots_mod  # noqa: E402


class SlotState(dict):
    """The server's state. With `--parallel N` a request that runs on a side slot sees the slot's
    own values of the per-request keys (server/slots.py `OVERLAY`: its engine, its drafter, what
    the generation publishes for the handler, None for the slot-0-only machinery) through the same
    `STATE[...]` the handler always read, so the handler runs on a side slot unchanged. On slot 0,
    and in every server without side slots, it is the plain dict it always was: `slots.values()`
    is None and every lookup falls through."""

    __slots__ = ()

    def __getitem__(self, k):
        v = slots_mod.values()
        if v is not None and k in v:
            return v[k]
        return dict.__getitem__(self, k)

    def get(self, k, d=None):
        v = slots_mod.values()
        if v is not None and k in v:
            return v[k]
        return dict.get(self, k, d)

    def __setitem__(self, k, val):
        v = slots_mod.values()
        if v is not None and k in v:
            v[k] = val
            return
        dict.__setitem__(self, k, val)

    def __contains__(self, k):
        v = slots_mod.values()
        if v is not None and k in v:
            return v[k] is not None
        return dict.__contains__(self, k)

    def pop(self, k, *default):
        v = slots_mod.values()
        if v is not None and k in v:
            out, v[k] = v[k], None
            if out is None and default:
                return default[0]
            return out
        return dict.pop(self, k, *default)

    def setdefault(self, k, d=None):
        v = slots_mod.values()
        if v is not None and k in v:
            if v[k] is None:
                v[k] = d
            return v[k]
        return dict.setdefault(self, k, d)


class EngineLock:
    """`LOCK`: slot 0's engine, held by a request from its prefill to its last token and by the
    engine's own jobs (the summary prepared ahead, the shadow reader, the memory writer, the
    decision models' batches).

    With `--parallel N` the GPU is shared by turns between the slots (server/slots.py `Turn`), and
    whoever holds this lock also takes the turn as slot 0 -- so nothing that runs under the engine
    lock (a decision model's batch, a role's graph capture, a cache clear) ever runs beside a side
    slot's work. The generation itself hands the turn over between blocks and while it waits for a
    joint verify round. Without side slots `turn` is None and this is a plain lock."""

    def __init__(self):
        self._lk = threading.Lock()
        self.turn = None

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        got = self._lk.acquire(blocking, timeout) if blocking else self._lk.acquire(False)
        if got and self.turn is not None:
            try:
                self.turn.reset(0)
                self.turn.acquire(0)
            except BaseException:
                self._lk.release()
                raise
        return got

    def release(self) -> None:
        t = self.turn
        if t is not None and t.held_by(0):
            t.release(0)
        self._lk.release()

    def locked(self) -> bool:
        return self._lk.locked()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc):
        self.release()
        return False


STATE: dict = SlotState()
# the routers' prices for this weight set (engine/prices.py); empty = the code's constants
PRICES: dict = {}
LOCK = EngineLock()

# The tree loop launches the next draft BEFORE it streams the block it just
# accepted: accept, commit, drafter sync, the drafter's bookkeeping, the next proposal up to its
# draft launch -- then the tokens go to the client (detokenizer, SSE writes, socket flushes) while
# the draft runs, and only then does the host wait for the draft. Until now the GPU idled through
# the streaming. The same calls with the same arguments in the same order, except that the yields
# move behind the launch, so the output is the same token for token. Measured and left off:
# with the window detokenizer the whole consumer costs 6.9 us
# a token on the box, so this can move at most ~0.03 ms of a ~96 ms round.
LAUNCH_FIRST = _S.get("LAUNCH_FIRST") == "1"


def _launch(steps):
    """Run a `*_steps` proposal to its first stop (its draft is on the device) or to its end.
    Returns (the generator if it stopped, else None; its result if it ended)."""
    try:
        next(steps)
    except StopIteration as done:
        return None, done.value
    return steps, None


def _collect(steps, done):
    """The proposal `_launch` started: drive it to the end, or hand back what it already returned."""
    if steps is None:
        return done
    while True:
        try:
            next(steps)
        except StopIteration as end:
            return end.value

# A chunk's keyword arguments when the request did not ask for log-probabilities.
NO_LP: dict = {}

# What the client is told when the repetition guard ends a stream. The finish reason is
# "stop" because that is the whole OpenAI vocabulary; this marker is the engine's own voice and
# says plainly that the cut was the engine's, so an incomplete answer is never presented as a
# finished one. Token accounting and the caches never see it.
GUARD_MARKER = "\n\n[engine: repetition guard stopped the output here]"


def _guard_headers(pstop) -> tuple:
    """`X-Engine-Stop` for a NON-streamed response only.

    A stream's headers are on the wire before its first token, and the guard fires at the end, so
    a streamed response can never carry this header. The streamed client gets the same fact in
    band instead: `GUARD_MARKER` is the last content delta before the finish chunk.
    """
    if pstop is not None and pstop.hit:
        return (("X-Engine-Stop", f"pattern-stop({pstop.label})"),)
    return ()

# The engine holds ONE sequence, so requests are serialised behind `LOCK` and everything else here
# is about being honest to a caller who arrives while it is held. An unbounded wait is not honesty:
# a client that queued behind four long generations gets an answer some minutes after it stopped
# caring, having held a socket the whole time. So the depth is bounded and the wait has a limit,
# and both refusals say `Retry-After`.
QUEUE = threading.Lock()


#: seconds between two prefill heartbeat comments, at least
HEARTBEAT_EVERY_S = 1.0


class ClientGone(ConnectionResetError):
    """The client closed its connection while its request waited or prefilled.

    A ConnectionResetError, so every path that already treats a reset reader as an abandoned
    request -- the stream's handler, `_complete_logged` -- treats this one the same way."""
INFLIGHT = {"waiting": 0, "running": 0, "lane": 0, "served": 0, "refused": 0, "errors": 0,
            "timeouts": 0, "abandoned": 0}

# S1: who waits for the engine, in what order, and who holds it. Every waiter has a rank: a short
# side request (a client's title call, capped and without thinking) goes first, because it holds
# the engine a few seconds and the session's turn would otherwise keep it waiting for minutes; a
# turn is next; a side request too long to be short waits until no turn does. A waiter takes the
# lock only while no waiter of a better rank waits, so with no side request every waiter has the
# same rank and the queue is what it always was. `HOLDER` is the request that holds the lock (None
# for anything else that takes it), for the waiting lines.
RANK_SIDE, RANK_TURN, RANK_SIDE_LONG = 0, 1, 2
WAITERS: list[dict] = []
HOLDER: dict = {"who": None}
#: how long a waiter that must let a better-ranked one go first sleeps between two looks
YIELD_S = 0.05


def _first(me: dict) -> bool:
    """No waiter of a better rank than `me` is waiting."""
    with QUEUE:
        return not any(w["rank"] < me["rank"] for w in WAITERS if w is not me)


def _take(me: dict, timeout: float | None) -> bool:
    """The engine lock for `me`, in its rank's turn: without blocking (`timeout` None) or for at
    most `timeout` seconds."""
    if not _first(me):
        if timeout:
            time.sleep(min(timeout, YIELD_S))
        return False
    got = LOCK.acquire(blocking=False) if timeout is None else LOCK.acquire(timeout=timeout)
    if got and not _first(me):
        LOCK.release()                 # a better-ranked waiter came while this one blocked
        return False
    if got:
        with QUEUE:
            HOLDER["who"] = me
        me["t_hold"] = time.monotonic()
    return got


#: the hold times behind the waiting line's estimate (server/queue_wait.py)
CLOCK = qwait.TurnClock()


def _wait_estimate(me: dict) -> tuple[int, float | None]:
    """`me`'s position in the line (1: next) and the seconds until it gets the engine, None
    when there is no basis (server/queue_wait.py)."""
    now = time.monotonic()
    with QUEUE:
        ahead = [w["rank"] for w in WAITERS
                 if w is not me and (w["rank"], w["t"]) < (me["rank"], me["t"])]
        who = HOLDER["who"]
    holder = None
    if who is not None and who.get("t_hold") is not None:
        holder = (who["rank"], now - who["t_hold"])
    elif LOCK.locked():
        return len(ahead) + 1, None      # held by something that is not a request: no basis
    live = _live_of(who.get("rec")) if who is not None and not who.get("own") else None
    return len(ahead) + 1, CLOCK.estimate(holder, ahead, frozenset((RANK_PREP, RANK_SHADOW)),
                                          live=live)


def _live_of(rec) -> "qwait.Live | None":
    """SRV-249: what the holder's record says of its work now (queue_wait.Live): the prefill
    still to read and its rate while it prefills, the tokens written and the decode rate after
    its first token; None before its prefill reported a chunk (no basis)."""
    if rec is None:
        return None
    try:
        thinking, cap = bool(rec.thinking), rec.max_tokens
        if rec.t_first is None:
            marks = rec.pf_marks
            if not marks:
                return None
            done, total, _ = marks[-1]
            tps = activity_mod.prefill_live(rec, 1).get("tps_live_avg")
            return qwait.Live(pf_left=max(0, int(total) - int(done)), pf_rate=tps,
                              thinking=thinking, max_tokens=cap)
        n = int(rec.n_live)
        el = time.perf_counter() - rec.t_first
        return qwait.Live(written=n, rate=(n / el if n >= 8 and el >= 1.0 else None),
                          thinking=thinking, max_tokens=cap)
    except Exception:                                              # noqa: BLE001
        return None


def _clock_wrote(rec) -> None:
    """SRV-249: a finished client request's tokens and decode rate, for the estimate."""
    n = int(getattr(rec, "n_live", 0) or 0)
    t1, t2 = getattr(rec, "t_first", None), getattr(rec, "t_last", None)
    rate = (n - 1) / (t2 - t1) if n >= 8 and t1 is not None and t2 is not None and t2 > t1 \
        else None
    CLOCK.wrote(bool(getattr(rec, "thinking", False)), n, rate)


def _place(me: dict) -> tuple[dict | None, int]:
    """What keeps `me` waiting -- the side request that holds the engine or goes ahead of it, else
    None -- and its position in the line (1: next)."""
    with QUEUE:
        ahead = [w for w in WAITERS if w is not me and (w["rank"], w["t"]) < (me["rank"], me["t"])]
        who = HOLDER["who"]
    side = next((w["side"] for w in [who, *ahead] if w is not None and w.get("side")), None)
    return side, len(ahead) + 1


#: what a waiter waits for when the engine's holder is TandemLLM's own work, not a client's
#: request (the waiting line said "another request is running" for all of them). SRV-177 (UT1,
#: 2026-10-06): in plain words, without the roles' internal names ("the shadow model"); the
#: technical holder stays in the journal and the live view
OWN_WAIT = {"question": "TandemLLM decides how to handle the request first",
            "shadow": "TandemLLM finishes its background work first",
            "prepare": "TandemLLM finishes a summary it prepared first",
            "engine": "TandemLLM finishes its own work first"}


def _own_holder() -> int | None:
    """SRV-177: the rank of TandemLLM's own work that holds the engine now (the shadow or
    prepare job; -1 for the conductor's own question or a lock no request took), else None."""
    with QUEUE:
        who = HOLDER["who"]
    if who is None:
        return -1 if LOCK.locked() else None
    if who.get("own"):
        return -1
    if who.get("rank") in (RANK_SHADOW, RANK_PREP):
        return int(who["rank"])
    return None


def _wait_reason(me: dict) -> tuple[str, int]:
    """The waiting line's reason and `me`'s position (1: next): a client's side request that holds
    the engine or goes ahead of it; TandemLLM's own work that holds it (the conductor's question
    to Qwen over the engine's own port, the shadow reader's burst, the summary written ahead, a
    lock taken by no request: the drain, a cache clear); else another request."""
    side, pos = _place(me)
    if side:
        return f"{side['client']}'s {side['kind']} request runs first", pos
    with QUEUE:
        who = HOLDER["who"]
    if who is None:
        if LOCK.locked():
            return OWN_WAIT["engine"], pos
    elif who.get("own"):
        return OWN_WAIT["question"], pos
    elif who.get("rank") == RANK_SHADOW:
        return OWN_WAIT["shadow"], pos
    elif who.get("rank") == RANK_PREP:
        return OWN_WAIT["prepare"], pos
    return "another request is running", pos


class Deadline:
    """A wall-clock cap on one generation, checked between blocks.

    Between blocks and not inside them: a block is one verify and is not interruptible, so the
    granularity of this is about 130 ms, which is the granularity the engine has. `hit` is what the
    stream reports as `finish_reason`, because a generation that ran out of time and one that ran
    out of tokens are different things to the caller and both used to be "length".
    """

    __slots__ = ("t_end", "hit")

    def __init__(self, seconds: float):
        self.t_end = time.monotonic() + float(seconds) if seconds else 0.0
        self.hit = False

    def expired(self) -> bool:
        if self.t_end and time.monotonic() > self.t_end:
            self.hit = True
        return self.hit


class BlockStats:
    """The two factors of one generation's speed, and where in each block the draft went wrong.

    tok/s is committed tokens a block over block time, and the row used to report only the
    quotient: a change that trades one factor for the other was invisible in it. So every
    forward the decode loop pays -- a verify block, a declined single step, a forced reasoning
    close -- is one block here, and the `[req]` line carries the count and the decode time beside
    the tokens, from which `tools/row3.py` takes tokens/block and ms/block.

    `accept` is the first-miss histogram: per block that had a draft, how many draft
    tokens it could have accepted (a chain's length, a tree's depth) and how many it did, so
    `tools/accept_hist.py --curve` can rebuild P(slot i accepted | the slots before it were),
    censored where a block had no slot i. It rides on `[req]` rather than on the `[drafter]` line,
    which is printed at the start of the NEXT request and so never reaches the row's last one.
    """

    __slots__ = ("blocks", "t_first", "t_last", "accept")

    def __init__(self):
        self.blocks = 0
        self.t_first = self.t_last = None
        self.accept: dict[int, dict[int, int]] = {}

    def first(self) -> None:
        self.t_first = self.t_last = time.perf_counter()

    def block(self, depth: int = 0, accepted: int = 0) -> None:
        self.blocks += 1
        self.t_last = time.perf_counter()
        if depth > 0:
            h = self.accept.setdefault(depth, {})
            h[accepted] = h.get(accepted, 0) + 1

    def fields(self, n_out: int) -> str:
        """` blocks=.. committed=.. decode_ms=.. accept=..`; the prefill's token is not decode's."""
        if self.t_first is None:
            return ""
        ms = (self.t_last - self.t_first) * 1e3
        acc = "|".join(f"{d}:" + ",".join(f"{a}x{n}" for a, n in sorted(h.items()))
                       for d, h in sorted(self.accept.items())) or "-"
        return (f" blocks={self.blocks} committed={max(0, n_out - 1)} decode_ms={ms:.1f}"
                f" accept={acc}")


# ------------------------------------------------------------------ generation
def _lane_chunks(on_chunk):
    """The prefill's chunk hook with the side lane's safe point after it (server/preempt.py): the
    rows below the chunk's line and the state at it are the prefill's, nothing is pending."""
    def hook(done, total):
        if on_chunk is not None:
            on_chunk(done, total)
        lane = STATE.get("lane")
        if lane is not None and lane.jobs:
            lane.serve("prefill")
    hook.info = getattr(on_chunk, "info", None)
    return hook


def _lane_room(job, where: str) -> bool:
    """Whether the engine has the memory for a side-lane job now (OPS-47's peak): the guest's
    borrowed rows and state plus its prefill's workspace on top of what the allocator holds. In a
    long prefill the holder may be at its own peak; the job then waits for a later safe point."""
    mem = STATE.get("mem")
    if mem is None or not mem.on:
        return True
    try:
        _, res = mem.memory()
        need = (cache._snapshot_estimate(STATE["engine"], len(job.ids) + job.max_new)
                + mem.workspace(len(job.ids)))
        return res + need <= mem.peak
    except Exception:                                              # noqa: BLE001
        return where != "prefill"


def _par_health() -> dict:
    """`/health`'s `parallel`: the pool's report and what the side slots read from the disk tier
    (`side_disk`: lookups, hits, tokens restored, failed restores; None without the reader), and the
    joint decode graphs the engine keeps (`joint_graphs`, engine/models/graph_lru.py; None without)."""
    out = dict.get(STATE, "pool").report()
    r = dict.get(STATE, "disk_side")
    out["side_disk"] = r.report() if r is not None else None
    jg = getattr(getattr(dict.get(STATE, "engine"), "fn", None), "__dict__", {}).get("joint_graphs")
    out["joint_graphs"] = jg.report() if hasattr(jg, "report") else None
    if kvpool.POOL is not None:
        out["kv_pool"] = kv_pool_report()
    return out


def _slot_disk():
    """The disk tier a prefill asks: slot 0's own; on a side slot (`--parallel N`) the read-only
    view of it (server/diskcache.py `SideReader`), or None with `QWEN38_SIDE_DISK=off`
    (a side slot had no disk tier, `OVERLAY` makes `disk_cache` None there)."""
    if slots_mod.current() is None:
        return STATE.get("disk_cache")
    return diskcache_mod.side_reader(STATE)


def _lane_run(job) -> None:
    """One side-lane job on the engine (server/preempt.py `decode`), with the engine held: an open
    disk-tier write that has not copied the rows the guest writes stops first."""
    from engine.model import h2d
    disk = STATE.get("disk_cache")
    if disk is not None:
        try:
            disk.guard_rows(len(job.ids) + job.max_new + preempt.ROW_SLACK)
        except Exception:                                          # noqa: BLE001
            logbuf.print_exc()
    preempt.decode(STATE["engine"], job, device=STATE.get("device", "cuda"),
                   chunk=int(STATE.get("prefix_chunk") or 0), h2d=h2d, block_stats=BlockStats)


#: A side slot takes a request only with this many rows to spare past prompt + max_tokens (the
#: router's note and a package go into the prompt after the slot is chosen: `PLAN_MARGIN` more for a
#: request the conductor shapes), so the request gets the max_tokens slot 0 would give it.
SIDE_MARGIN = 64
PLAN_MARGIN = 8192
#: SRV-305: a request waiting for memory gets its line again this often (s)
MEM_LINE_EVERY_S = 15.0

#: The least output a side slot's window must leave a request (its prompt and the margins
#: taken): below it the request waits for slot 0, at or above it the request runs there with its
#: `max_tokens` cut to the room (`_side_room`). 0.4.7 compared the window with prompt + max_tokens
#: whole, and the served default is 32,768: a chat past about 24,000 tokens could never take the
#: 65,536-token side slot, so two Open WebUI chats of 28,000 and 64,000 tokens took turns on slot 0
#: (2026-10-08 12:14-12:22: queue waits of 31, 127, 180, 107 and 138 s, one slot idle). 16,384 is
#: the `high` thinking budget, the most a routed turn thinks before its answer.
SIDE_MIN_ROOM = int(_S.get("SIDE_MIN_ROOM") or 16384)


def _continues_resident(valid: int, common: int, chunk: int, conv: int | None = None) -> bool:
    """Does a request whose prompt shares `common` leading tokens with slot 0's resident rows
    (`valid` of them) continue that conversation, so that it waits for slot 0 and resumes in place?
    Yes when it shares (nearly) all of the rows (within one prefill chunk) AND at least one chunk of
    them: below a chunk, waiting for slot 0 saves less prefill than the wait costs. In an
    earlier test build only the first test ran, and a resident conversation shorter than a chunk
    (valid - chunk <= 0) made every request "continue" it, so no request took a side slot (PL1).

    `conv` (`_conv_start`): where the request's conversation starts to be its own, past the end of
    its first user message. A client's fixed head (Open WebUI's system prompt and tool block, about
    6,700 tokens; opencode's, about 7,400) is the same for every chat of that client, so sharing it
    is not continuing a conversation: the shared part must reach past it. Until then a
    new chat beside a conversation shorter than the head plus one chunk counted as its next turn
    and waited for slot 0 while the side slot was idle (W18: 6,675 of 7,096 rows shared, 35.8 s in
    the queue for '17 x 23'). None (no first user message found): the rule above alone."""
    chunk = max(1, int(chunk))
    return (valid > 0 and common >= valid - chunk and common >= chunk
            and (conv is None or common >= conv))


#: the chat template's marks `_conv_start` looks for: (`<|im_start|>`, the ids of `user`,
#: `<|im_end|>`), from the tokenizer at the first use; False when there is none (the fake engine)
_CONV_MARKS: tuple | None | bool = None


def _marks_of(tok):
    """The template's marks from a tokenizer: (`<|im_start|>`, the ids of `user`, `<|im_end|>`),
    or None when it has none."""
    try:
        im_s = tok.convert_tokens_to_ids("<|im_start|>")
        im_e = tok.convert_tokens_to_ids("<|im_end|>")
        user = list(tok.encode("user", add_special_tokens=False))
        ok = (isinstance(im_s, int) and isinstance(im_e, int) and im_s >= 0 and im_e >= 0
              and user and im_s != getattr(tok, "unk_token_id", None))
        return (im_s, user, im_e) if ok else None
    except Exception:                                              # noqa: BLE001
        return None


def _conv_marks():
    global _CONV_MARKS
    if _CONV_MARKS is None:
        tok = dict.get(STATE, "tok")
        _CONV_MARKS = (_marks_of(tok) if tok is not None else None) or False
    return _CONV_MARKS or None


def _conv_start(ids) -> int | None:
    """Where a rendered prompt's conversation starts to be its own: the position just past the
    `<|im_end|>` of its first user message (`<|im_start|>user ... <|im_end|>`). Everything before
    is the client's head (its system prompt, its tools), which every chat of that client sends the
    same. Structure of the chat template, not content: None when the template's marks are not
    known or the prompt has no user message."""
    marks = _conv_marks()
    if marks is None:
        return None
    im_s, user, im_e = marks
    u = len(user)
    try:
        i = ids.index(im_s)
        while True:
            if list(ids[i + 1:i + 1 + u]) == user:
                return ids.index(im_e, i + 1 + u) + 1
            i = ids.index(im_s, i + 1)
    except Exception:                                              # noqa: BLE001
        return None


def _shared_head(a, b) -> int:
    """How many leading tokens the lists `a` and `b` share (engine/cache.py `Resident.common`'s
    search, for two host lists)."""
    m = min(len(a), len(b))
    if m == 0:
        return 0
    if list(a[:m]) == list(b[:m]):
        return m
    lo, hi = 0, m                          # a[:lo] == b[:lo], a[:hi] != b[:hi]
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if list(a[:mid]) == list(b[:mid]):
            lo = mid
        else:
            hi = mid
    return lo


def _continues_view(view_len: int, common: int, chunk: int, conv: int | None = None) -> bool:
    """Does a request continue the conversation slot 0 holds, judged on the CLIENT's tokens?

    `view_len` is the length of the last slot-0 prompt whose rows stayed resident, as the client's
    messages render (`_par_side_need`'s ids: no router note, no replayed span, no generated row);
    `common` is how many leading tokens this request's rendered ids share with it. The next turn of
    the same conversation starts with all of that prompt except the generation prompt's last few
    tokens, so it continues when it shares the view within one prefill chunk, and the view is at
    least a chunk long (below a chunk, a side slot's prefill costs less than the wait;
    `_continues_resident`'s rule).

    Why not the resident's own rows (`_continues_resident`): under a conductor plan the rows hold
    the router's notes, this turn's at the tail and the earlier turns' replayed where they were
    put, and after a generation also its end anchor's rows. The rendered ids part from those rows
    at the first note, often in the conversation's first turn, so no conductor request ever
    counted as continuing it (`resident_waits` 0 over ~100 opencode
    turns), and a turn that met slot 0 held for a moment (the shadow reader's read, a summary
    being prepared) took the side slot and read its whole conversation again (SRV-197).

    `conv`: as in `_continues_resident`, the shared part must reach past the request's first user
    message; a short last prompt of another chat of the same client shares only the client's head."""
    chunk = max(1, int(chunk))
    return view_len >= chunk and common >= view_len - chunk and (conv is None or common >= conv)


#: which test decides that a request continues slot 0's conversation and waits for it: `view`
#: (the client's rendered tokens against the last resident prompt's, `_continues_view`, then the
#: resident rows' test) or `resident` (the rows' test alone, the old rule; kept for the
#: A/B that measured SRV-197)
SIDE_CONTINUE = (_S.get("SIDE_CONTINUE") or "view").strip().lower()

#: `on`: a request continues slot 0's conversation only when the shared part reaches
#: past its first user message (`_conv_start`); `off`: the client's head alone may count (a 0.4.0 test build,
#: kept for the A/B)
SIDE_CONV = (_S.get("SIDE_CONV") or "on").strip().lower()

#: FX5: whether a guest of slot 0's conversation (`_guest_of_slot0`) takes an idle side slot before
#: slot 0: `on`, or `off` (slot 0 with the stash, the old rule; kept for the A/B)
SIDE_GUESTS = (_S.get("SIDE_GUESTS") or "on").strip().lower()

#: FX10 (SRV-213): `on`: the next turn of a conversation whose last request ran on a side slot is
#: carried into slot 0 from that slot's rows (`_carry_from_side`); `off`: it reads its prompt again
#: on slot 0 from what slot 0's caches hold, as 0.4.0 did
SIDE_CARRY = (_S.get("SIDE_CARRY") or "on").strip().lower()

#: FX10: how long a side slot's last end is kept from a guest of slot 0 (the router's question
#: before the conversation's next user message, a title task) while slot 0 is free
CARRY_KEEP_S = 600.0

#: FX16: `on`: a new conversation beside a resident one a side slot could never hold takes an idle
#: side slot before slot 0 (`_par_side_need`'s `prefer_side`); `off`: slot 0 first, as 0.4.7
SIDE_NEW_FIRST = (_S.get("SIDE_NEW_FIRST") or "on").strip().lower()

#: FX16: how long a turn that continues a side slot's conversation waits for that slot (another
#: request on it: a guest, a new chat) before it takes slot 0, where its rows are carried in
#: (`_carry_from_side`) or read again
SIDE_HOME_WAIT_S = float(_S.get("SIDE_HOME_WAIT_S") or 10.0)

#: FX17: a new chat of at most SIDE_NEW_SHORT rendered tokens (plus the margins) beside a long
#: resident conversation waits SIDE_NEW_WAIT_S for a busy side slot before it takes slot 0, where
#: it would send the long conversation to the NVMe (restorable, about 3 s, but its next turn
#: then waits behind the new chat's). The rule: the eviction costs the long chat's next turn, the
#: wait costs only the new chat's first token, and a new chat's first answer is rarely urgent
#: within 20 s; a long new chat does not wait (it would not fit a busy side slot's leftovers)
SIDE_NEW_WAIT_S = float(_S.get("SIDE_NEW_WAIT_S") or 20.0)
SIDE_NEW_SHORT = 16384

#: 0.8.2: how long a request whose conversation slot 0 holds (or reads back from the NVMe) waits for
#: slot 0 before it takes a free side slot it fits in. Before, it waited for slot 0 as long as a
#: request may run: a new Open WebUI chat, matched to another chat's NVMe entry by its shared system
#: prompt, sat "Queued" 437 s behind a 496 s tool-call generation while two side slots were free
#: (2026-10-13). On the side slot the disk tier serves what it holds; the rest is prefilled
SIDE_SLOT0_WAIT_S = float(_S.get("SIDE_SLOT0_WAIT_S") or 3.0)

#: FX16: a side slot's own resident prefix (engine/cache.py `ResidentPrefix`): the anchors it may
#: hold (GB; 0 builds none, the 0.4.7 slot with no conversation of its own), the stash that keeps
#: its conversation under a guest (GB), with the end anchor on when slot 0's is
SIDE_RESIDENT_GB = float(_S.get("SIDE_RESIDENT_GB") or 0.5)
SIDE_STASH_GB = float(_S.get("SIDE_STASH_GB") or 1.0)

#: SRV-629: `on`: a request with images may take a side slot (each slot is its own engine over the
#: same weights, with its own `mm` and rotary offset; the tower runs in the slot's prefill), routed
#: on its expanded key ids (`_mm_keys`); `off`: slot 0 only, as 0.12.0. Live 2026-10-19: two
#: Chat-tab agent runs with 3 to 8 screenshots a request ran all 31 requests on slot 0 (56.6 min
#: of wall time for 18.9 min of decoding) while slot 1 stayed idle
SIDE_IMAGES = (_S.get("SIDE_IMAGES") or "on").strip().lower()


def _guest_of_slot0(res, n: int, rc: int, chunk: int) -> bool:
    """FX5: a request of `n` rendered tokens that shares `rc` with slot 0's resident rows is a
    guest of the conversation slot 0 holds: short (what `ResidentPrefix.is_guest` calls a guest),
    unrelated to it (less than one chunk shared, so no anchor of slot 0 would serve it), while slot
    0 holds a conversation of at least two chunks. On slot 0 it writes over the rows that
    conversation's next turn resumes from, and only the stash puts them back; on an idle side slot
    it costs the same prefill and touches nothing of slot 0. The router's own questions (the class,
    the package) and a client's title, tags and follow-up tasks are such requests.
    FX15 (2026-10-08): no share of the resident's length here. `guest_frac` bounds what the stash
    copies for a guest that runs ON slot 0; a side slot copies nothing. With it, the package
    question (2,354 tokens, 34 options) was no guest of an 8.1k Open WebUI chat (limit 2,028): it
    ran on slot 0 every turn, displaced the chat's end anchor (the turn resumed at 0.96-0.98 and
    re-read 250-670 tokens) and cost 3.3 s when it missed its own head there (UX3 W02/W05)."""
    try:
        valid = int(getattr(res, "valid", 0) or 0)
        return (valid >= 2 * chunk and rc < chunk
                and n <= int(getattr(res, "guest_max", 0) or 0))
    except Exception:                                              # noqa: BLE001
        return False


def _keyed(ctx: list | None, mm) -> list | None:
    """SRV-629: a request's context as the caches hold it: the image rows as their key ids
    (`_remember`'s `committed`)."""
    if ctx is None or mm is None or len(ctx) < int(mm.n):
        return ctx
    return list(mm.key_ids) + list(ctx[int(mm.n):])


def _par_view_keep(served: list | None, view: list | None) -> bool:
    """At the end of a request on slot 0: remember `view` (its rendered ids) as the conversation
    slot 0 holds when the resident rows hold its served prompt (`served`, notes included): a
    guest (a short prompt served over the resident's tail, which is put back after it) or a
    request whose rows were not kept leaves the view it found. True when kept."""
    res = dict.get(STATE, "resident")
    if res is None or not view or not served:
        return False
    try:
        valid = int(getattr(res, "valid", 0) or 0)
        chunk = max(1, int(getattr(res, "chunk", 1) or 1))
        n = min(len(served), valid)
        if valid < chunk or n < chunk or res.common(served[:n]) < n - chunk:
            return False
    except Exception:                                              # noqa: BLE001
        return False
    STATE["slot0_view"] = list(view)
    # SRV-528: when, and whether its rows hold the whole prompt (a prefill cancelled part way does
    # not), for `_slot0_stale`
    STATE["slot0_seen"] = (time.monotonic(), valid >= len(served))
    return True


#: SRV-528: how long slot 0's conversation keeps new conversations on the side slots
#: (`_par_side_need`'s `prefer_side`) after the last request that continued it
SLOT0_KEEP_S = float(_S.get("SLOT0_KEEP_S") or 1800.0)


def _slot0_stale(now: float | None = None) -> str | None:
    """SRV-528: why slot 0's resident conversation no longer counts for `prefer_side`: `partial`
    when the last request that kept it stopped inside its prompt (a prefill cancelled part way),
    `idle` when no request continued it for SLOT0_KEEP_S; None while it counts. A 1,000,000-token
    test prompt cancelled after 937,984 rows (2026-10-16 22:01) kept slot 0, the 1,048,576-token
    window, idle all evening: every new conversation was sent to a 262,144-token side slot beside
    it. Nothing is dropped: its rows stay until a request needs slot 0, and its own retry still
    resumes there (`_continues_resident`)."""
    seen = dict.get(STATE, "slot0_seen")
    if seen is None:
        return None
    t, whole = seen
    if not whole:
        return "partial"
    if (time.monotonic() if now is None else now) - t >= SLOT0_KEEP_S:
        return "idle"
    return None


#: how long a request waits for slot 0 while only TandemLLM's own work holds it (the shadow reader,
#: a summary being prepared, the memory writer, the router's question) before a side slot may take
#: it. Own work gives way to a waiting request at its next safe point; a side slot has none of slot
#: 0's caches, so a turn sent there by a moment of own work read its whole prompt again (SRV-197).
SIDE_AFTER_OWN_S = 15.0


def _side_ok(waited_s: float, holder) -> bool:
    """May a request that has waited `waited_s` take a side slot, given slot 0's holder
    (`_own_holder()`: None for a client request or a free slot, else TandemLLM's own work)?"""
    return holder is None or waited_s >= SIDE_AFTER_OWN_S


def _holder_name() -> str:
    """Who holds slot 0 now, in words (the `[parallel]` line): free, a request, or TandemLLM's own
    work by its rank."""
    with QUEUE:
        who = HOLDER["who"]
    if who is None:
        return "held" if LOCK.locked() else "free"
    if who.get("own"):
        return "the router's question"
    return {RANK_SHADOW: "the shadow reader", RANK_PREP: "prepare/memory job"}.get(
        who.get("rank"), "a request")


def _mm_keys(ids: list[int], images: list) -> list[int] | None:
    """SRV-629: the rendered ids with each image's one placeholder expanded to its rows, each row
    its image's key id (engine/vision.py `key_id`): the prompt as the caches compare it
    (`MMContext.key_ids`), on the host, before any lock and without the tower. None when the
    expansion fails (the handler answers it as a 400 under the lock) or the tower builds its own
    context (Super VL's RADIO tower: no side slot reads its images yet, they stay on slot 0)."""
    tower = STATE.get("vision")
    if tower is None or getattr(tower, "context", None) is not None:
        return None
    from engine.vision import expand, key_id
    try:
        full, spans = expand(list(ids), images, tower.cfg.image_token_id,
                             tower.cfg.spatial_merge_size)
    except Exception:                                              # noqa: BLE001
        return None
    for sp_ in spans:
        full[sp_.start:sp_.start + sp_.n] = [key_id(images[sp_.image].digest)] * sp_.n
    return full


def _mm_rows(n_text: int, images: list) -> int:
    """SRV-629: the rows a prompt of `n_text` rendered tokens with `images` takes once each
    image's placeholder is expanded (`_vision_context`); `n_text` when that cannot be told."""
    tower = STATE.get("vision")
    if not images or tower is None:
        return int(n_text)
    try:
        ph = int(tower.cfg.image_token_id)
        own = getattr(tower, "context", None)
        if own is not None:
            full, _ = own([ph] * len(images), images)
        else:
            from engine.vision import expand
            full, _ = expand([ph] * len(images), images, ph, tower.cfg.spatial_merge_size)
        return int(n_text) + len(full) - len(images)
    except Exception:                                              # noqa: BLE001
        return int(n_text)


def _par_side_need(body, client, left_out, max_new, plan, images, lane_kind,
                   why: dict | None = None, stay: bool = False,
                   parent: str | None = None, session: str | None = None) -> int | None:
    """The window a side slot needs for this request, or None when it must wait for slot 0: a side
    lane job (served at slot 0's safe points), a request with images while SIDE_IMAGES is off (or
    with a tower that builds its own context, `_mm_keys`), or one no side slot is long enough for.
    SRV-629: a request with images is routed on its expanded key ids and its window counts their
    rows (`why["n_images"]`, `why["n_rows"]`). A request that continues slot 0's resident
    conversation (it resumes there in place; on a side slot it prefills what the disk tier does not
    hold) gets its window too, with `why["slot0_first"]`: the handler tries slot 0 first, for
    SIDE_SLOT0_WAIT_S (0.8.2). Host work only: the ids are rendered on the CPU, before any lock.
    `why`, when given, gets the numbers the choice was made on and the rendered ids (`ids`, for
    `_par_view_keep`).

    ENG-371: `stay`, an agent's worker (`_worker_of`), is never a guest of slot 0's conversation;
    a new one takes a free side slot first when slot 0 holds its `parent`, so the parent stays in
    place, and it never waits SIDE_NEW_WAIT_S for one. A request whose conversation slot 0's park
    holds tries slot 0 first, as one on the NVMe does; so does a parent whose worker ran
    (`session` in PARENTS: its rows went to slot 0's park or the NVMe, which only slot 0 reads
    back, unless a side slot still holds them, `_side_home`), and it waits for slot 0 up to the
    time reading its prompt again would take (`slot0_wait_s`, PARENT_WAIT_MAX_S at most), as a
    worker's call that continues slot 0's rows (or its park or NVMe entry) does. Live
    2026-10-18 01:00: two parents' returns, their rows on the NVMe, took the side slot after
    `prefer_side` and read 11,119 and 11,010 tokens again: the client's copy, without the
    router's notes, matched no entry there."""
    why = why if why is not None else {}
    pool = dict.get(STATE, "pool")
    if pool is None or pool.n < 2:
        return None
    if images and SIDE_IMAGES != "on":
        why["images"] = True              # the Lanes view's reason (`_queue_why`)
        return None
    # a side lane job is served at slot 0's safe points, unless it is a guest of slot 0's
    # conversation (FX5: the router's own questions are short enough for the lane, and when slot 0
    # is free they ran on it as guests)
    lane = lane_kind is not None
    try:
        ids = build_prompt(body, client, count="ids", left_out=left_out)
    except Exception:                                              # noqa: BLE001
        return None                       # the render fails again under the lock, as a 400
    rows = len(ids)
    if images:
        # SRV-629: the window holds the expanded rows, not one placeholder an image (8
        # screenshots: about 7,700 rows), and every test below runs on the key ids, as the caches
        # compare them: the resident rows, the views, the park and the NVMe hold an image's rows
        # as its key, so the rendered ids part from them at the first image
        keys = _mm_keys(ids, images)
        if keys is None:
            why["images"] = True
            return None
        ids = keys
        rows = len(keys)
        why.update(n_images=len(images), n_rows=rows)
    why["ids"] = ids
    # SRV-629: an end kept after images (slot 0's park, an NVMe entry) is offered to a request
    # with images, as its own restore asks (`_spill_before_replay(..., mm_ok=True)`)
    mm_kw = {"mm_ok": True} if images else {}
    res = dict.get(STATE, "resident")
    chunk = int(getattr(res, "chunk", 0) or STATE.get("prefix_chunk") or 1024)
    # Past the client's head, or it is a new conversation (`_continues_resident`)
    conv = _conv_start(ids) if SIDE_CONV == "on" else None
    if conv is not None:
        why["conv"] = conv
    view = dict.get(STATE, "slot0_view")
    if SIDE_CONTINUE != "resident" and view:
        vc = _shared_head(ids, view)
        if not _continues_view(len(view), vc, chunk, conv):
            # FX16: the client's block moved to its last message (server/endalign.py)
            al0 = dict.get(STATE, "end_align")
            if al0 is not None:
                vm = al0.moved_common(view, ids)
                if vm > vc:
                    why["view_moved"] = vm - vc
                    vc = vm
        why.update(view=len(view), view_common=vc)
        if _continues_view(len(view), vc, chunk, conv):
            why["wait"] = "view"
            with pool._lk:
                pool.stats["resident_waits"] = pool.stats.get("resident_waits", 0) + 1
    valid = rc = 0
    if res is not None and not why.get("wait"):
        try:
            valid = int(getattr(res, "valid", 0) or 0)
            rc = res.common(ids) if valid else 0
            why.update(valid=valid, res_common=rc)
            if valid and _continues_resident(valid, rc, chunk, conv):
                why["wait"] = "resident"
                with pool._lk:
                    pool.stats["resident_waits"] = pool.stats.get("resident_waits", 0) + 1
            else:
                if (SIDE_GUESTS == "on" and not stay and not images
                        and _guest_of_slot0(res, len(ids), rc, chunk)):
                    # SRV-629: never a request with images: a guest on slot 0 is served over the
                    # resident rows with the stash, which no image request ever was
                    why["guest"] = valid  # FX5: an idle side slot first (the handler's `_take`)
                # rc-0.8.0: a conversation on the NVMe (a side slot's, written at a planned stop)
                # comes back through slot 0, whose prefill reads the spill; a side slot has none
                sp = STATE.get("spill")
                if not why.get("guest") and sp is not None and sp.entries:
                    with cache_peek.quiet(sp):
                        got = sp.candidate(res, ids, len(ids) - 1, **mm_kw)
                    # 0.8.2: only an entry that holds this conversation past its first user
                    # message (`conv`) is its conversation. A line at the client's head is not:
                    # a new Open WebUI chat matched another chat's entry at 8,192 tokens, its
                    # system prompt and tools (conv 9,087), and waited 437 s for slot 0 behind a
                    # 496 s generation (2026-10-13); the disk tier serves that head on a side
                    # slot too (server/diskcache.py `SideReader`)
                    if got is not None and (conv is None or int(got[1]) >= conv):
                        why.update(wait="spill", spill=int(got[1]))
                        with pool._lk:
                            pool.stats["spill_waits"] = pool.stats.get("spill_waits", 0) + 1
                    elif got is not None:
                        why["spill_head"] = int(got[1])
        except Exception:                                          # noqa: BLE001
            pass
    if res is not None and not why.get("wait") and not why.get("guest"):
        # ENG-371, read outside the engine lock as everything above is: a park the slot's own
        # thread changes meanwhile costs this one choice (the prefill decides on the rows under
        # the lock), and no failure here skips the parent's or the worker's check below
        if getattr(res, "park_rows", 0):
            try:
                bk = int(res.park_offer(ids, rc, len(ids) - 1, **mm_kw) or 0)
            except Exception:                                      # noqa: BLE001
                bk = 0
            if bk and (conv is None or bk >= conv):
                why.update(wait="park", park=bk)     # slot 0's park holds this conversation
        if not why.get("wait") and not stay and _waiting_parent(session):
            why["wait"] = "parent"                   # slot 0 reads its rows back
        if (not why.get("wait") and stay and session is not None and valid
                and getattr(res, "owner", None) == session and rc >= max(chunk, conv or 0)):
            # a worker's own rows on slot 0, its copy parting inside them (a write_file's
            # content folded to a stub): it resumes there, below the part
            why["wait"] = "own"
    if ((stay and why.get("wait") in ("view", "resident", "own", "spill", "park"))
            or (not stay and _waiting_parent(session)
                and why.get("wait") in ("parent", "spill", "park"))):
        # ENG-371: a worker's next call too (live 2026-10-18 02:02: its rows on slot 0, slot 0
        # busy for 3.1 s, the side slot read its 15,170 tokens from 0)
        rate = _peek_rate() or PARENT_REREAD_TOK_S
        why["slot0_wait_s"] = round(min(PARENT_WAIT_MAX_S,
                                        max(SIDE_SLOT0_WAIT_S, len(ids) / rate)), 1)
    if lane and not why.get("guest"):
        return None
    margin = SIDE_MARGIN + (PLAN_MARGIN if plan is not None and getattr(plan, "active", False)
                            else 0)
    why["margin"] = margin
    need = rows + int(max_new) + margin
    why["need"] = need
    if not any(need <= s.max_len for s in pool.side()):
        # the whole output limit does not fit beside the prompt: a side slot with at least
        # SIDE_MIN_ROOM rows of output left takes the request with its limit cut to that room
        # (`_side_room`, applied once the slot is known); else the request waits for slot 0
        least = rows + margin + SIDE_MIN_ROOM
        if SIDE_MIN_ROOM > 0 and any(least <= s.max_len for s in pool.side()):
            why["need"], why["clip"] = least, True
            need = least
        elif why.get("wait"):
            why["too_long"] = True            # its conversation's slot, and no other fits it
            return None
        else:
            why["wait"] = "too_long"
            with pool._lk:
                pool.stats["too_long"] = pool.stats.get("too_long", 0) + 1
            return None
    # FX16: the side slot that holds this conversation's rows (its last turn ran there and
    # nothing wrote over them since): the turn resumes at its end there, beside slot 0
    home = _side_home(pool, ids, need, chunk, conv)
    if why.get("wait"):
        here = max(int(why.get(k) or 0) for k in ("view_common", "res_common", "spill", "park"))
        if home is not None and home[1] > here:
            # SRV-528: a side slot holds more of it than slot 0 does (live 2026-10-17: a chat whose
            # first prompt matched slot 0's view waited 3.0 s for slot 0 at every turn, then took
            # the lowest free side slot)
            why.update(home=home[0], home_share=home[1])
            return need
        # 0.8.2: it continues slot 0's conversation (view, resident) or reads it back from the
        # NVMe there (spill): slot 0 first, for SIDE_SLOT0_WAIT_S, then a free side slot (the
        # handler); never minutes behind another conversation's run on slot 0
        why["slot0_first"] = True
        return need
    if home is not None:
        why["home"] = home[0]
    elif (SIDE_NEW_FIRST == "on" and res is not None
          and not why.get("guest") and not why.get("wait")):
        # FX16: a new conversation beside a resident one a side slot could never hold takes an
        # idle side slot before slot 0 (else it evicts that conversation from slot 0, whose next
        # turn then waits behind it: the side slot is too short for it)
        try:
            valid = int(getattr(res, "valid", 0) or 0)
            head = max(int(why.get("view_common") or 0), int(why.get("res_common") or 0))
            longest = max((s.max_len for s in pool.side()), default=0)
            # another conversation: the shared part ends before its first user message, or,
            # without the template's marks, more than a chunk short of the resident rows
            other = head < conv if conv is not None else head < valid - chunk
            # SRV-546: a conversation with images runs on slot 0 only (side slots read no images):
            # a new chat that takes slot 0 from it sends it to the NVMe, as one too long for a side
            # slot (live 2026-10-17 10:38: a 7,293-token chat took slot 0 beside a free side slot
            # and the 140,191-token pagoda run went to the NVMe). SRV-629: kept with SIDE_IMAGES
            # on: its next turn may take a side slot, but reads its prompt there again
            bound = (valid + margin + SIDE_MIN_ROOM > longest
                     or bool(res.end is not None and res.end.get("mm")))
            # ENG-371: a worker beside its own parent: the parent stays on slot 0
            mine = stay and parent is not None and getattr(res, "owner", None) == parent
            stale = (_slot0_stale() if other and valid >= 2 * chunk and bound and not mine
                     else None)
            if stale is not None:
                why["slot0_stale"] = stale     # SRV-528: nobody's conversation: slot 0 serves it
            elif other and valid >= 2 * chunk and (bound or mine):
                why["prefer_side"] = valid
                if len(ids) <= SIDE_NEW_SHORT and not stay:
                    # FX17: a short new chat waits SIDE_NEW_WAIT_S for a busy side slot before it
                    # takes slot 0 and sends the long conversation to the NVMe
                    why["prefer_side_wait"] = True
        except Exception:                                          # noqa: BLE001
            pass
    return need


def _queue_why(pool, why: dict, side_need, waited: float, me: dict) -> str:
    """0.8.2: why a queued request waits, in words, for the Lanes view's queued line
    (server/parallel.py; it said only "Queued" for 6 min 25 s while two side slots were free):
    the slot it waits for and for how long, or what holds slot 0 (`_wait_reason`)."""
    if pool is None:
        return _wait_reason(me)[0]
    if side_need is None:
        if why.get("images"):
            if SIDE_IMAGES != "on":
                return "it has images, which only slot 0 reads; waits for slot 0"
            # SRV-629: a tower with its own context (Super VL) is read on slot 0 only
            return "this model's images are read on slot 0 only; waits for slot 0"
        if why.get("wait") == "too_long" or why.get("too_long"):
            longest = max((s.max_len for s in pool.side()), default=0)
            return f"too long for a side slot ({longest:,} tokens); waits for slot 0"
        return _wait_reason(me)[0]
    if why.get("home") is not None and why.get("home_wait") is None:
        return f"waits up to {SIDE_HOME_WAIT_S:g} s for slot {why['home']}, which holds its conversation"
    if why.get("hop_from") is not None:
        return (f"moved from slot {why['hop_from']} (busy {why.get('home_wait')} s): re-reading its "
                f"conversation on the next free slot")
    limit = float(why.get("slot0_wait_s") or SIDE_SLOT0_WAIT_S)
    if why.get("slot0_first") and waited < limit:
        where = ("slot 0 reads its conversation back from the NVMe" if why.get("wait") == "spill"
                 else "slot 0 reads its conversation back" if why.get("wait") == "parent"
                 else "slot 0 holds its conversation")
        return f"{where}; a free side slot takes it after {limit:g} s"
    if why.get("prefer_side_wait") and why.get("new_wait") is None:
        return (f"a new chat: waits up to {SIDE_NEW_WAIT_S:g} s for a side slot, so slot 0 keeps "
                f"its long conversation")
    if all(s.busy for s in pool.slots):
        return f"all {pool.n} slots are busy"
    return _wait_reason(me)[0]


def _par_nums(why: dict) -> str:
    """The numbers of the `[parallel]` line: what the slot was chosen on (SRV-197), and for a
    request with images their count and the expanded rows (SRV-629)."""
    nums = " ".join(f"{k}={why[k]}" for k in ("view", "view_common", "valid", "res_common",
                                              "conv", "need", "guest", "park", "slot0_wait_s",
                                              "home", "home_share", "home_wait", "hop_from",
                                              "prefer_side", "slot0_stale", "new_wait", "spill",
                                              "spill_head", "slot0_wait")
                    if k in why)
    if "n_images" in why:
        nums = f"images={why['n_images']} rows={why.get('n_rows')} {nums}".rstrip()
    return nums


def _side_room(slot, n_ids: int, margin: int, max_new: int) -> int:
    """The output limit a request of `n_ids` rendered tokens gets on `slot`: its own, or the
    rows the slot's window leaves it past the prompt and the margins when that is less."""
    room = int(slot.max_len) - int(n_ids) - int(margin)
    return max(1, min(int(max_new), room))


def _side_home(pool, ids: list[int], need: int, chunk: int, conv: int | None) -> tuple | None:
    """FX16: (the index of the side slot that holds this conversation, the tokens of the prompt it
    holds), when its window holds `need` rows; else None. A slot holds it when the prompt continues
    the last prompt served there (judged on the client's tokens, as `_continues_view` judges slot
    0's) or shares at least a chunk of the slot's own rows past the first user message
    (`res.common`). A request of another conversation served there since cleared them
    (`ResidentPrefix.begin`), and so the view.

    SRV-528: with or without an end anchor, and the slot with the longest share wins. FX16 named a
    slot home only when it kept an end anchor and took the first such slot; a turn that kept none
    (the engine ahead, SRV-530) was routed as a new conversation, to the lowest free side slot
    `take_side` allowed, and once the carry guard of the slot below expired (CARRY_KEEP_S) a
    45,778-token conversation moved from slot 2 to slot 1 and was read again in 80.9 s
    (2026-10-17 00:05). Without an end anchor the turn resumes at the slot's anchors. Among the
    slots within a chunk of the longest share a free one goes first."""
    found = []
    for s in pool.side():
        vals = s.values or {}
        res, view = vals.get("resident"), vals.get("slot_view")
        if res is None or need > s.max_len:
            continue
        try:
            if int(getattr(res, "valid", 0) or 0) < chunk:
                continue
            held = 0
            if view:
                vc = _shared_head(ids, view)
                if not _continues_view(len(view), vc, chunk, conv):
                    al_s = vals.get("end_align")
                    if al_s is not None:
                        vc = max(vc, al_s.moved_common(view, ids))     # FX16: a moved block
                if _continues_view(len(view), vc, chunk, conv):
                    held = vc
            if not held:
                # the slot's own rows (no view kept, or the client's copy parts from it): past the
                # first user message, or without the template's marks nearly all of them
                rc = res.common(ids)
                valid = int(getattr(res, "valid", 0) or 0)
                if rc >= chunk and (rc >= conv if conv is not None else rc >= valid - chunk):
                    held = rc
            if held:
                found.append((held, s))
        except Exception:                                          # noqa: BLE001
            continue
    if not found:
        return None
    top = max(h for h, _ in found)
    near = [(h, s) for h, s in found if h >= top - chunk]
    h, s = next(((h, s) for h, s in near if not s.busy), next((h, s) for h, s in found if h == top))
    return s.index, h


def _side_view_keep(slot, served: list | None, view: list | None) -> bool:
    """FX16: at the end of a request on a side slot with a resident prefix: the client's rendered
    ids as the conversation the slot holds, when its rows hold the served prompt (as
    `_par_view_keep` keeps slot 0's). A guest or a request whose rows were not kept leaves the
    view it found; one whose prefill wrote over another conversation's rows replaces it."""
    vals = slot.values or {}
    res = vals.get("resident")
    if res is None or not view or not served:
        return False
    try:
        valid = int(getattr(res, "valid", 0) or 0)
        chunk = max(1, int(getattr(res, "chunk", 1) or 1))
        n = min(len(served), valid)
        if valid < chunk or n < chunk or res.common(served[:n]) < n - chunk:
            return False
    except Exception:                                              # noqa: BLE001
        return False
    vals["slot_view"] = list(view)
    return True


def _par_end(pool, side, cid) -> None:
    """A request's end on its slot: its bytes out of the admission, its slot out of the barrier,
    the turn and the side slot given back."""
    if pool.admission is not None:
        pool.admission.release(cid)
    if side is None:
        pool.slots[0].phase = ""
        return
    side.phase = ""
    try:
        if pool.turn is not None and pool.turn.held_by(side.index):
            pool.turn.release(side.index)
    finally:
        slots_mod.bind(None)
        pool.release_side(side)


# ------------------------------------------------------------------ parallel requests
# `--parallel N` (server/slots.py). Every helper below runs only when `STATE["pool"]` exists; a
# server without side slots never calls one.
def _par_index() -> int:
    """The slot the current thread's request runs on: a side slot's index, else 0."""
    s = slots_mod.current()
    return s.index if s is not None else 0


def _par_slot(pool) -> "slots_mod.Slot":
    return pool.slots[_par_index()]


def _phased(gen):
    """The generation, with its slot out of the round's barrier the moment it ends: the other
    slots stop waiting for a block that will not come while this request finishes up."""
    pool = dict.get(STATE, "pool")
    slot = _par_slot(pool)
    try:
        yield from gen
    finally:
        slot.phase = ""


def _par_chunks(on_chunk):
    """The prefill's chunk hook, then the GPU offered to a waiting slot (a no-op alone)."""
    def hook(done, total):
        if on_chunk is not None:
            on_chunk(done, total)
        pool = dict.get(STATE, "pool")
        if pool is not None and pool.turn is not None:
            pool.turn.pause(_par_index())
    hook.info = getattr(on_chunk, "info", None)
    return hook


def _par_block(pool, tree_mode: bool) -> None:
    """Before a block: this slot decodes -- in the barrier's set when its blocks are trees, which
    go through `_verify_tree`; a chain loop verifies alone and is not waited for -- and the GPU is
    offered to a waiting slot."""
    slot = _par_slot(pool)
    slot.phase = "decode" if tree_mode else "solo"
    if pool.turn is not None and pool.turn.waiting():
        pool.turn.pause(slot.index)


def _share_trim(pool, tree):
    """A joint round's row budget: with N slots decoding, a slot's tree keeps its best
    `slots.share(N)` nodes (best-first by score, with their ancestors, DFS pre-order kept). The
    joint pass then stays at or under 32 rows, where every projection is row-invariant. Dropping
    candidates never changes what is accepted, only how much. A slot alone keeps its tree."""
    meet = pool.meeting
    if meet is None or meet.cut == "joint":
        return tree                    # a joint cut is the round's (`_run_round`)
    budget = slots_mod.share(len(pool.decoding()), meet.max_rows)
    if budget is None or len(tree.parents) <= budget:
        return tree
    order = sorted(range(1, len(tree.tokens)), key=lambda i: -tree.scores[i])
    keep = {0}
    for i in order:
        if len(keep) >= budget:
            break
        path, j = [], i
        while j not in keep:
            path.append(j)
            j = tree.parents[j]
        if len(keep) + len(path) <= budget:
            keep.update(path)
    with meet.cv:
        meet.stats["trimmed"] += 1
    return tree.subset(keep)


def _run_round(meet) -> None:
    """Verify every pending block (the caller holds the GPU turn): the batchable ones in passes of
    at most `max_rows` rows through `forward_tree_multi`, the others (and a pass of one) through
    the lone `forward_tree` with its graphs."""
    from engine.model import forward_tree_multi
    from engine.model import h2d
    with meet.cv:
        groups = meet.batches()
        meet.pending.clear()
    for group in groups:
        if meet.cut == "joint" and len(group) > 1 and all(r.tree is not None for r in group):
            # one cut over every slot's whole tree, to the pass's row budget
            keeps = slots_mod.joint_cut([r.tree for r in group], meet.max_rows)
            for r, keep in zip(group, keeps):
                if len(keep) < len(r.tree.tokens):
                    t = r.tree_cut = r.tree.subset(keep)
                    r.tokens = h2d(t.tokens, torch.long, r.tokens.device)
                    r.parents = t.parents
                    r.rows = len(t.parents)
            with meet.cv:
                meet.stats["trimmed"] += sum(1 for r in group if r.tree_cut is not None)
        if meet.timing:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            if len(group) == 1:
                r = group[0]
                outs = [r.eng.forward_tree(r.tokens, r.parents, start=r.start)]
            else:
                # an engine with its own joint pass (ENG-241), else the 27B's
                multi = getattr(group[0].eng, "forward_tree_multi", None) or forward_tree_multi
                outs = multi([(r.eng, r.tokens, r.parents, r.start) for r in group])
            err = None
        except BaseException as exc:                                    # noqa: BLE001
            outs, err = [None] * len(group), exc
        if meet.timing and err is None:
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1e3
        if meet.timing and err is None:
            with meet.cv:
                pm = meet.stats["pass_ms"].setdefault(str(len(group)), [0.0, 0])
                pm[0] += ms
                pm[1] += 1
        with meet.cv:
            meet.stats["rounds"] += 1
            if len(group) > 1:
                meet.stats["joint_rounds"] += 1
                meet.stats["joint_blocks"] += len(group)
                rows = sum(r.rows for r in group)
                h = meet.stats["rows_hist"]
                h[rows] = h.get(rows, 0) + 1
                jh = meet.stats["joint_hist"]
                jh[len(group)] = jh.get(len(group), 0) + 1
            else:
                meet.stats["single_rounds"] += 1
            for r, lg in zip(group, outs):
                # a joint block's time is the pass's, not what its lone verify costs: the router
                # keeps its learned costs across requests, so it is not told (False)
                r.out, r.ms, r.error = lg, (ms if len(group) == 1 else False), err
                r.joint = len(group)
                meet.transit.add(r.slot)
            meet.cv.notify_all()


def _par_note(pool, key: str, n, value: float) -> None:
    """One sample into the barrier's stats: `stats[key][str(n)] = [sum, count]`. Only while
    timing (`QSE_PARALLEL_TIMING=1`): a served round pays nothing for it."""
    meet = pool.meeting
    if meet is None or not meet.timing:
        return
    with meet.cv:
        r = meet.stats.setdefault(key, {}).setdefault(str(n), [0.0, 0])
        r[0] += float(value)
        r[1] += 1


def _run_draft_round(meet) -> None:
    """The packed draft of every pending draft call (the caller holds the GPU turn): the calls
    grouped by drafter module (the two arms are two modules), cut to the rows the row-invariant
    kernels keep (`max_rows`); a group of one takes its drafter's own graph."""
    from engine.drafters.draft_graph import packed_draft
    with meet.cv:
        reqs = list(meet.pending.values())
        meet.pending.clear()
    by: dict = {}
    for r in reqs:
        by.setdefault(id(r.drafter.module.w), []).append(r)        # one checkpoint, one group
    for group in by.values():
        per = max(1, meet.max_rows // max(1, group[0].rows))
        for k in range(0, len(group), per):
            part = group[k:k + per]
            if meet.timing:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            try:
                if len(part) == 1:
                    outs = [part[0].drafter.graph_run(part[0].anchor, part[0].pos0)]
                else:
                    outs = packed_draft([(r.drafter, r.anchor, r.pos0) for r in part])
                err = None
            except BaseException as exc:                                # noqa: BLE001
                outs, err = [None] * len(part), exc
            if meet.timing and err is None:
                torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) * 1e3
            with meet.cv:
                meet.stats["rounds"] += 1
                if len(part) > 1:
                    meet.stats["joint_rounds"] += 1
                    meet.stats["joint_blocks"] += len(part)
                    jh = meet.stats["joint_hist"]
                    jh[len(part)] = jh.get(len(part), 0) + 1
                else:
                    meet.stats["single_rounds"] += 1
                if meet.timing and err is None:
                    pm = meet.stats["pass_ms"].setdefault(str(len(part)), [0.0, 0])
                    pm[0] += ms
                    pm[1] += 1
                for r, o in zip(part, outs):
                    r.out, r.error, r.joint = o, err, len(part)
                    meet.transit.add(r.slot)
                meet.cv.notify_all()


def _draft_meet(drafter, anchor: int, pos0: int):
    """`dflash2.DRAFT_MEET`: with other slots decoding, this slot's draft call joins the round's
    packed draft (one read of the drafter's layers and one head call for every slot); alone, or
    outside a request, None -- the drafter's own graph, as without side slots."""
    pool = dict.get(STATE, "pool")
    meet = getattr(pool, "draft_meeting", None) if pool is not None else None
    if meet is None or pool.turn is None:
        return None
    i = _par_index()
    if len(pool.decoding()) <= 1 or not pool.turn.held_by(i) or pool.slots[i].phase != "decode":
        return None
    req = slots_mod.DraftReq(i, drafter, anchor, pos0, int(drafter.cfg.block_size))
    slots_mod.barrier(meet, pool, pool.turn, i, req, _run_draft_round)
    return req.out


def _verify_tree(pool, eng, block, parents, pos, tree=None):
    """A tree block's verify: `forward_tree` alone, or a joint round with the other decoding
    slots (`slots.barrier`). Returns (logits, ms, tree): `ms` None when it verified alone (the
    caller times it), the lone verify's time when another slot ran it, False for a joint pass;
    `tree` the tree the round verified (`tree` itself unless a joint cut cut it)."""
    meet, turn = pool.meeting, pool.turn
    i = _par_index()
    if meet is None or turn is None or len(pool.decoding()) <= 1 or not turn.held_by(i):
        if meet is not None:
            with meet.cv:
                meet.stats["alone"] += 1
                # why: the only decoding slot, or the GPU turn not this slot's
                why = meet.stats.setdefault("alone_why", {})
                k = "one" if len(pool.decoding()) <= 1 else "turn"
                why[k] = why.get(k, 0) + 1
        return eng.forward_tree(block, parents, start=pos), None, tree
    req = slots_mod.VerifyReq(i, eng, block, parents, pos, batchable=eng.tree_batchable(parents),
                              tree=tree if meet.cut == "joint" else None)
    slots_mod.barrier(meet, pool, turn, i, req, _run_round)
    pool.last_joint[i] = req.joint
    return req.out, req.ms, (req.tree_cut if req.tree_cut is not None else tree)


def _end_cut(new: list[int], eos, room: int) -> int:
    """SRV-530: how many of a verified block's `new` tokens the generation keeps: through its first
    end token, and at most `room` (the output budget left). The block's rows past that point are
    not committed.

    The walk accepts rows past the end token whenever the draft has them: the lookup copies what
    followed the same tool call earlier in the prompt (`</tool_call>`, `<|im_end|>`, `\\n`,
    `<|im_start|>`), and the target accepts the newline after its end token. Committed, they left
    the engine one or two rows past the committed tokens. Qwen's pending commit is cut back to the
    committed tokens (`trim_pending`); a NemotronH state has walked those rows and cannot be, so the
    end anchor was refused (`ahead`) on 31 of the 54 turns of 2026-10-16, every one a tool call. The
    commit now stops at the last kept token's row, where Qwen's cut back stopped."""
    j = next((i for i, t in enumerate(new) if t in eos), None)
    keep = len(new) if j is None else j + 1
    return max(1, min(keep, int(room)))


def _end_cut_said(rows: int) -> None:
    """SRV-530: a block's rows past the end were left uncommitted: counted (`end_cut` in the
    resident prefix's stats) and said in the log."""
    res = STATE.get("resident")
    if res is not None:
        res.stats["end_cut"] = res.stats.get("end_cut", 0) + 1
    if STATE.get("verbose"):
        print(f"[spec] the accepted path ran {rows} row(s) past the end: not committed", flush=True)


def generate_stream(*args, **kw):
    """`_generate_stream`; with side slots (`--parallel N`), its slot leaves the round's barrier
    the moment the generation ends, however it ends (the handler may abandon it half way)."""
    gen = _generate_stream(*args, **kw)
    if dict.get(STATE, "pool") is None:
        return gen
    return _phased(gen)


#: FX5: the rows a guest's stash keeps beyond its token budget for the monitor's hidden tokens
GUEST_HIDDEN_ROWS = 2048


def _generate_stream(prompt: torch.Tensor, max_new: int, eos: set[int], think=None,
                    conv_id: str | None = None, deadline: "Deadline | None" = None,
                    pen: "PenaltyState | None" = None, pstop: "PatternStop | None" = None,
                    sampler: "Sampler | None" = None, lpr: "lp_mod.Recorder | None" = None,
                    on_prefill=None, mm=None, sup=None, compacting: bool = False, probe=None,
                    disk_tag: dict | None = None, pos_delta: int = 0, end_mm: bool = False,
                    scratch: bool = False, stay: bool = False):
    """Yield token ids as they are decided, speculation included.

    Same loop as `engine.spec.generate_spec`, rewritten as a generator so a token reaches the
    socket at the moment it is accepted rather than at the end of the block. A verified block
    produces several tokens at once and they go out together; that is what the engine does, and
    smoothing it would make the inter-token latency a fiction.

    The prefill is `engine.cache.prefill`, which may resume from a state the store already holds.
    Nothing downstream of it knows or cares: it restores the same bytes a forward would have
    written and returns the same logits.

    `lpr` (`logprobs`) is handed each decided token's row where the token is decided --
    after the penalties and `logit_bias`, before the sampling filter -- in the order the tokens are
    yielded; a request that does not ask passes None and the loop only checks for it.

    `on_prefill(done, total)` is called between prefill chunks: the handler's check for a
    client that left, which raises and ends the prefill there, and its stream heartbeat.

    `mm` (engine/vision.py MMContext): the request's images. The engine reads it while it
    prefills the prompt, and every row after the prompt takes its rotary offset `mm.delta`; the
    caches key on `mm.key_ids`. None for text, and set (to None) on every request, so a request
    never inherits the previous one's images.

    `sup` (server/conductor/monitor/): the request's supervisor, None when it runs raw -- and
    then every line below runs as it did without one. SP1 is after a block's `think.observe`:
    the supervisor may close the reasoning with a hidden phrase (`_force_close` with its
    action). SP2 is where an end token is decided, before anything commits it: a hidden continue
    keeps only the block's rows before the end token (the same `rollback_to` / `commit_tree`
    a partial accept takes), forwards the action's tokens and decodes on. What the client
    receives of a hidden forward is `act.visible`; the rest is in `ctx` and the KV only.

    `probe` (server/probe.py `Capture`, ENG-181): the request's activation snapshots, None unless
    `QWEN38_PROBE_SNAPSHOTS` is on and the request is TandemLLM with tools. It copies rows the
    forwards already wrote -- the prompt's row in the prefill's chunk hook, a block's rows at every
    1,024 reasoning tokens, the row that decided an end token -- and never changes a tensor the
    engine reads. None: every line below runs as it did without it.

    `disk_tag` (server/diskcache.py `request_tag`): what the disk tier is told about the request
    -- its client, the notes spliced into its prompt, where its last user message starts. The
    tier itself is `STATE["disk_cache"]`; None (`--disk-cache off`) is the prefill without it.

    `pos_delta`, `end_mm` (server/compaction.py): a text prompt that continues a
    conversation with images: the rotary offset its rows take and leave to the end anchor kept
    after those images. `scratch`: a summary prepared in the background, whose rows are nobody's.
    """
    from engine.model import h2d           # not at import: the fake engine's server has no triton
    eng = STATE["engine"]
    eng.mm = mm
    eng.pos_delta = mm.delta if mm is not None else int(pos_delta)
    drafter = STATE["drafter"]
    k = STATE["k"]
    ctx = prompt.tolist()
    # SRV-307: the prompt's rows exist before any restore or prefill writes them
    _kv_ens = getattr(getattr(eng, "kv", None), "ensure", None)
    if _kv_ens is not None:
        _kv_ens(len(ctx))
    if pen is not None:
        pen.seed(ctx)
    # The list the engine's own positions index into, published for `_remember`. It is the SAME
    # object, appended to as tokens are committed, so `ctx[:eng.kv.length]` is by construction the
    # prefix the engine really forwarded -- including when a caller abandons this generator half
    # way through a verified block, where `pos` has already advanced past what has been appended
    # and the slice comes out short. Reconstructing it from the tokens the caller collected would
    # be an argument about that invariant instead of a use of it.
    STATE["last_ctx"] = ctx
    # the end anchor's (`_remember`): where the prompt ends in `ctx`, and the spans of tokens a
    # hidden action forwarded that the client never receives (`_force_close`)
    STATE["last_prompt_n"], STATE["hidden_spans"] = len(ctx), []
    bs = STATE["blocks"] = BlockStats()
    if sup is not None:
        # E2's questions are asked in place at SP2 (`_force_close`); a request that records
        # logprobs keeps its end token's row, so it is never asked
        sup.can_ask = lpr is None
    if think is not None:
        think.start(ctx)
    with torch.no_grad():
        if drafter is not None:
            # A drafter that keeps per-request policy state clears it in `reset()`, so this is the
            # last moment the PREVIOUS request's choices can be read. Printed here rather than at
            # the end of the generation because the stream returns from four places inside its
            # loop and none of them is an exit worth wrapping for a log line.
            if STATE.get("verbose") and hasattr(drafter, "report"):
                print(f"[drafter] {drafter.report()}", flush=True)
            drafter.reset()
            if hasattr(drafter, "set_sampling"):
                # a sampler-carrying drafter draws its proposals from its own
                # distribution under the request's profile and carries q for the verify.
                # `--sampled-tree det`: the tree is the greedy request's, built without
                # the sample, and walked by drawing the target's token at each node.
                det = (STATE.get("sampled_tree") == "det" and STATE.get("tree")
                       and sampler is not None and sampler.on)
                drafter.set_sampling(None if det else sampler)
            if hasattr(drafter, "prime"):
                drafter.prime(ctx)
        t_pre = time.perf_counter()
        # the handler's chunk hook carries the dict the prefill describes itself in
        # (start, chunk, t0, kind), so the live view reads this request's own prefill
        where = getattr(on_prefill, "info", None)
        if where is None:
            where = {}
        on_chunk = on_prefill
        if probe is not None:
            probe.bind(eng, drafter)
            on_chunk = probe.chunks(on_prefill, where)
        if STATE.get("lane") is not None:
            on_chunk = _lane_chunks(on_chunk)
        if dict.get(STATE, "pool") is not None:
            on_chunk = _par_chunks(on_chunk)
        logits, reused, forwarded = cache.prefill(
            eng, drafter, ctx, prompt.device, store=STATE.get("state_store"),
            chunk=STATE.get("prefix_chunk", 0), conv_id=conv_id,
            checkpoint=bool(STATE.get("prefix_cache")), resident=STATE.get("resident"),
            on_chunk=on_chunk, info=where,
            key_ids=mm.key_ids if mm is not None else None, mem=STATE.get("mem"),
            compacting=compacting, disk=_slot_disk(), disk_tag=disk_tag,
            end_mm=end_mm, scratch=scratch, stay=stay,
            # FX5: the highest row this generation can write, for a guest's stash; the monitor's
            # hidden tokens are not in `max_new`
            reach=len(ctx) + int(max_new) + (GUEST_HIDDEN_ROWS if sup is not None else 0))
        if probe is not None:
            probe.prefilled(eng, drafter, len(ctx), where.get("start", 0))
        if where.get("why"):
            # SRV-546: a long re-read is never silent (positions only, never text)
            print(f"[cache] re-read {forwarded:,} of {len(ctx):,} tokens (resumed at {reused:,}, "
                  f"{where.get('kind') or 'cold'}): {where['why']}", flush=True)
        if mm is not None:
            mm.release()
        # a conversation read back from the NVMe for this request (engine/spill.py) is `spill`
        spilled = where.get("spill") or STATE.pop("spill_restored", None)
        # FX10: a conversation carried in from a side slot for this request is `side`
        carried = STATE.pop("side_carried", None)
        STATE["last_prefill"] = {"reused": reused, "forwarded": forwarded,
                                 "ms": (time.perf_counter() - t_pre) * 1e3,
                                 "kind": ("spill" if spilled else "side" if carried
                                          else where.get("kind"))
                                 if reused else None}
        if spilled:
            STATE["last_prefill"]["spill"] = spilled
        if carried:
            STATE["last_prefill"]["side"] = carried
        # the park or the pin the rows came back from (usage.tandem.cache): in the prefill, or the
        # parked end brought back before the replay (`_spill_before_replay`)
        via = where.get("via") or ("park" if STATE.pop("park_returned", None) else None)
        if reused and via:
            STATE["last_prefill"]["via"] = via
        pos = prompt.numel()
        if pen is not None:
            pen.mask = bool(think is not None and think.inside)
            pen.apply_single(logits[0, -1])
        tok = sampler(logits[0, -1], index=len(ctx)) if sampler is not None and sampler.on \
            else int(logits[0, -1].argmax())
        start = len(ctx)
        if probe is not None and tok in eos:
            probe.finish(eng, drafter, slice(-1, None), 0)
        fin = (_sp2(sup, think, ctx, start, 0, eng.max_len - pos, end=tok)
               if sup is not None and tok in eos else None)
        if fin is None:
            if lpr is not None:
                lpr.rows(logits[0, -1:], [tok])
            n_out = 1
            ctx.append(tok)
            bs.first()
            if pen is not None:
                pen.commit([tok])
            if drafter is not None:
                drafter.observe([tok])
            yield tok
            if tok in eos:
                return
        else:
            # SP2 on the very first token: nothing has committed it, so it is simply not taken
            bs.first()
            n_out = 0
            for t in _force_close(eng, drafter, think, ctx, pos, prompt.device, pen=pen,
                                  pstop=pstop, sampler=sampler, lpr=lpr, act=fin, sup=sup,
                                  eos=eos, start=start, n_out=n_out):
                n_out += 1
                yield t
                if n_out >= max_new:
                    return
            if (pstop is not None and pstop.hit) or ctx[-1] in eos:
                return
            pos = len(ctx) - 1
            tok = ctx[-1]
        # Sampled requests keep their drafter (rejection sampling under speculation --
        # see engine/sample.py; the output follows the target's sampled distribution either way).
        # v1: a sampled request takes the q-aware CHAIN. `--sampled-tree`: the
        # tree instead -- `det`, the greedy request's tree walked by drawing at each node, or
        # `mixed`, the sampled chain as the tree's spine (its q rows accepted by rejection
        # sampling) with the lattice's siblings beside it (engine/tree.py::spine_tree).
        tree_mode = (STATE.get("tree") and drafter is not None
                     and hasattr(drafter, "propose_tree")
                     and (not (sampler is not None and sampler.on)
                          or STATE.get("sampled_tree", False)))
        ahead = None                  # the next proposal, launched before the last stream
        par = dict.get(STATE, "pool")
        t_post = None                 # parallel requests: when this slot's last verify returned
        while n_out < max_new:
            if deadline is not None and deadline.expired():
                return
            if par is not None:
                # parallel requests: this slot decodes (the round's barrier waits for its blocks
                # when it is a tree loop), and the GPU is offered to a waiting slot between blocks
                # a drafter taking the plain step all turn (`plain_turn`) submits no block: the
                # round's barrier must not wait for it
                _par_block(par, tree_mode and (not getattr(drafter, "plain_turn", False)
                                               or getattr(eng.caps, "joint_plain", False)))
            lane = STATE.get("lane")
            if lane is not None and lane.jobs:
                # A short request waits (server/preempt.py). A block boundary: the
                # proposal in flight lands first, the engine is lent and given back as it was
                if ahead is not None:
                    ahead = (None, _collect(*ahead))
                lane.serve("decode")
            if sup is not None and pos >= eng.max_len - 1:
                return                 # hidden tokens took rows the output clamp did not count
            if tree_mode:
                # The tree path. It is the same loop with three lines changed: the drafter hands
                # back a shape rather than a list, the accept is a walk down that shape instead of
                # a prefix comparison, and the commit takes the path rather than a length. The
                # reason it is worth the branch is in docs/speculative-decoding.md under "tree verify":
                # the step costs the same for two rows as for sixteen.
                t_dr = time.perf_counter() if par is not None else 0.0
                if par is not None and t_post is not None:
                    # the slot's own work between its verify and its next draft: accept, commit,
                    # the drafter's sync, the stream, the turn's offer
                    _par_note(par, "post_ms", len(par.decoding()), (t_dr - t_post) * 1e3)
                    t_post = None
                if ahead is not None:
                    tree, ahead = _collect(*ahead), None
                else:
                    tree = drafter.propose_tree(ctx, min(k, max_new - n_out))
                if par is not None:
                    _par_note(par, "draft_ms", len(par.decoding()), (time.perf_counter() - t_dr) * 1e3)
                # the KV write counts NODES (anchor included) while the budget above
                # counts output tokens, and a drafter may return more nodes than it was handed.
                # A DFS pre-order prefix is a valid tree, so cutting at the row bound only
                # drops candidates.
                if tree is not None:
                    tree = tree.truncate(eng.max_len - pos)
                if par is not None and tree is not None and tree.n_draft > 0:
                    tree = _share_trim(par, tree)
                joint_anchor = False
                if ((tree is None or tree.n_draft == 0) and par is not None
                        and getattr(eng.caps, "joint_plain", False) and len(par.decoding()) > 1):
                    # ENG-241: no draft this round, other slots decoding: the plain step joins the
                    # round's pass as the anchor alone, so the slots read the weights once
                    tree, joint_anchor = DraftTree(tokens=[int(ctx[-1])], parents=[-1]), True
                if tree is None or (tree.n_draft == 0 and not joint_anchor):
                    draft = []
                else:
                    block = h2d(tree.tokens, torch.long, prompt.device)
                    tvt = time.perf_counter()
                    if par is None:
                        lg, vms = eng.forward_tree(block, tree.parents, start=pos), None
                    else:
                        lg, vms, tree = _verify_tree(par, eng, block, tree.parents, pos, tree)
                        t_post = time.perf_counter()
                        _par_note(par, "verify_wait_ms", len(par.decoding()), (t_post - tvt) * 1e3)
                    if pen is not None:
                        pen.mask = bool(think is not None and think.inside)
                        pen.apply_tree(lg, tree)
                    on_verify = getattr(drafter, "on_verify", None)
                    if on_verify is not None and vms is not False:
                        # a joint round's time is the round's, not this block's (`_verify_tree`)
                        on_verify(tree.n_draft + 1,
                                  vms if vms is not None else (time.perf_counter() - tvt) * 1e3)
                    if sampler is not None and sampler.on:
                        # Rejection accept down the tree: the target's own token is
                        # sampled at every node and the walk follows the child carrying it; where
                        # no child carries it, the draw is the token and the walk stops.
                        path, new = sampler.tree_walk_logits(lg, tree.tokens, tree.parents,
                                                             start=len(ctx), q=tree.q)
                    else:
                        # a graphed verify took the argmax itself; a penalty changed the
                        # logits after it, so then the loop takes it
                        picks_d = eng.picks if eng.picks is not None and pen is None else None
                        picks_t = (picks_d if picks_d is not None else lg.argmax(-1)).tolist()
                        path, new = walk.accept_tree(tree, picks_t)
                    accepted = len(path) - 1
                    if par is not None:
                        # tokens a round on this slot: in a joint pass (by blocks in it) or alone
                        _par_note(par, "accept", f"joint{par.last_joint.get(_par_index(), 1)}"
                                  if vms is False else "alone", len(new))
                    if probe is not None:
                        j = next((i for i, t in enumerate(new) if t in eos), None)
                        if j is not None:
                            probe.finish(eng, drafter, slice(path[j], path[j] + 1), n_out + j)
                    fin = (_sp2_block(sup, think, ctx, start, n_out, new, eos, max_new,
                                      eng.max_len - pos) if sup is not None else None)
                    if fin is not None:
                        # the end token is not taken: the path is kept up to the node before it
                        path, new = path[:fin[0] + 1], new[:fin[0]]
                    cut = _end_cut(new, eos, max_new - n_out)
                    if cut < len(new):
                        # SRV-530: rows past the end token (or the budget) are not committed
                        if len(path) > cut + 1:
                            _end_cut_said(len(path) - cut - 1)
                        path, new = path[:cut + 1], new[:cut]
                    if lpr is not None:
                        lpr.rows(lg[h2d(path, torch.long, prompt.device)], new)
                    eng.commit_tree(path)
                    if hasattr(drafter, "sync"):
                        sel = h2d(path, torch.long, prompt.device)
                        toks = [int(tree.tokens[i]) for i in path]
                        if getattr(drafter, "wants_rows", False):
                            drafter.sync(toks, eng.hidden_post_norm[0, sel], pos, rows=path)
                        else:
                            drafter.sync(toks, eng.hidden_post_norm[0, sel], pos)
                    if probe is not None and probe.due(think, len(new)):
                        probe.think(eng, drafter, h2d(path, torch.long, prompt.device))
                    pos += len(path)
                    bs.block(max(tree.depths()), accepted)
                    drafter.observe(new)
                    if pen is not None:
                        pen.commit(new)
                    stop_now = pstop is not None and pstop.observe(new)
                    close = None
                    if (LAUNCH_FIRST and not stop_now and n_out + len(new) < max_new
                            and not any(t in eos for t in new) and fin is None):
                        # the stream goes on after this block, so the next proposal is
                        # started first and the block is streamed while its draft runs. What the
                        # old order did between the yields and the proposal -- the think budget's
                        # look at the block and SP1 -- is done before it, as it only reads the
                        # tokens.
                        n0 = len(ctx)
                        ctx.extend(new)
                        if think is not None:
                            think.observe(new)
                        if sup is not None:
                            close = _sp1(sup, think, n_out + len(new), eng.max_len - pos,
                                         len(ctx))
                        if (think is None or not think.hit) and close is None:
                            ahead = _launch(tree_steps(drafter, ctx,
                                                       min(k, max_new - n_out - len(new))))
                        sent = 0
                        try:
                            for t in new:
                                n_out += 1
                                sent += 1
                                yield t
                        finally:
                            # a reader that stops here leaves `ctx` as the old order did: through
                            # the token it was handed, since `_remember` publishes this list
                            del ctx[n0 + sent:]
                        tok = ctx[-1]
                        if (think is None or not think.hit) and close is None:
                            continue
                    else:
                        for t in new:
                            ctx.append(t)
                            n_out += 1
                            yield t
                            if t in eos or n_out >= max_new:
                                return
                        if stop_now:
                            # The repeating block was yielded first: the client sees what was
                            # written, then the stream ends with finish_reason stop and a [req]
                            # annotation.
                            return
                        tok = ctx[-1]
                        if think is not None:
                            think.observe(new)
                        if sup is not None and fin is None:
                            close = _sp1(sup, think, n_out, eng.max_len - pos, len(ctx))
                    if fin is not None or close is not None:
                        # SP2's hidden continue, or SP1's hidden close
                        if close is not None and close.kind == GIVE_UP:
                            return
                        for t in _force_close(eng, drafter, think, ctx, pos, prompt.device,
                                              pen=pen, pstop=pstop, sampler=sampler, lpr=lpr,
                                              act=fin[1] if fin is not None else close, sup=sup,
                                              eos=eos, start=start, n_out=n_out):
                            n_out += 1
                            yield t
                            if n_out >= max_new:
                                return
                        if (pstop is not None and pstop.hit) or ctx[-1] in eos:
                            return
                        pos = len(ctx) - 1
                        tok = ctx[-1]
                    elif think is not None:
                        if think.hit:
                            for t in _force_close(eng, drafter, think, ctx, pos, prompt.device,
                                                  pen=pen, pstop=pstop, sampler=sampler, lpr=lpr,
                                                  sup=sup, eos=eos, start=start, n_out=n_out):
                                n_out += 1
                                yield t
                                if n_out >= max_new:
                                    return
                            if pstop is not None and pstop.hit:
                                return                 # the guard fired on the phrase
                            if sup is not None and ctx[-1] in eos:
                                return
                            pos += 1 + len(think.close_ids)
                            if sup is not None:
                                pos = len(ctx) - 1     # a hidden continue may have followed
                            tok = ctx[-1]
                    continue
            else:
                draft = (drafter.propose(ctx, min(k, max_new - n_out))
                         if drafter is not None else [])
                # the block's KV write is `1 + len(draft)` rows (the anchor's own row is
                # one of them) while every clamp above counts output tokens, and a drafter may
                # return more than it was handed. Cap on rows here, where the forward is paid.
                draft = draft[:max(0, eng.max_len - pos - 1)]
            if not draft:
                # a drafter that prices its rounds against the plain step learns its cost here
                on_plain = getattr(drafter, "on_plain", None) if drafter is not None else None
                t_plain = time.perf_counter() if on_plain is not None else 0.0
                logits = eng.forward(h2d([tok], torch.long, prompt.device), start=pos,
                                     last_only=True)
                # Bring a position-indexed drafter current, as engine/spec.py's loop does.
                # The block drafter's cache must cover every committed position; skip this one and
                # it is one behind for good -- it declines every later step, and the request
                # finishes one token a forward.
                if drafter is not None and hasattr(drafter, "sync"):
                    drafter.sync([tok], eng.hidden_post_norm[0], pos)
                pos += 1
                bs.block()
                if probe is not None and probe.due(think, 1):
                    probe.think(eng, drafter, slice(0, 1))
                if pen is not None:
                    pen.mask = bool(think is not None and think.inside)
                    pen.apply_single(logits[0, -1])
                tok = sampler(logits[0, -1], index=len(ctx)) if sampler is not None and sampler.on \
                    else int(logits[0, -1].argmax())
                if on_plain is not None:
                    on_plain((time.perf_counter() - t_plain) * 1e3)
                if probe is not None and tok in eos:
                    probe.finish(eng, drafter, slice(0, 1), n_out)
                fin = (_sp2(sup, think, ctx, start, n_out, eng.max_len - pos, end=tok)
                       if sup is not None and tok in eos else None)
                close = None
                if fin is None:
                    if lpr is not None:
                        lpr.rows(logits[0, -1:], [tok])
                    ctx.append(tok)
                    n_out += 1
                    if pen is not None:
                        pen.commit([tok])
                    if pstop is not None and pstop.observe([tok]):
                        yield tok
                        return
                    if drafter is not None:
                        drafter.observe([tok])
                    yield tok
                    if tok in eos:
                        return
                    if think is not None:
                        think.observe([tok])
                        if sup is not None:
                            close = _sp1(sup, think, n_out, eng.max_len - pos, len(ctx))
                if fin is not None or close is not None:
                    # SP2: the end token was never forwarded, so it is simply not taken
                    if close is not None and close.kind == GIVE_UP:
                        return
                    for t in _force_close(eng, drafter, think, ctx, pos, prompt.device, pen=pen,
                                          pstop=pstop, sampler=sampler, lpr=lpr,
                                          act=fin or close, sup=sup, eos=eos, start=start,
                                          n_out=n_out):
                        n_out += 1
                        yield t
                        if n_out >= max_new:
                            return
                    if (pstop is not None and pstop.hit) or ctx[-1] in eos:
                        return
                    pos = len(ctx) - 1
                    tok = ctx[-1]
                elif think is not None:
                    # The same close the verified paths make. A drafter-less server --
                    # and any declined step -- used to observe the token and never look, so the
                    # budget and the stall signal could not close a block on this path at all.
                    if think.hit:
                        for t in _force_close(eng, drafter, think, ctx, pos, prompt.device,
                                              pen=pen, pstop=pstop, sampler=sampler, lpr=lpr,
                                              sup=sup, eos=eos, start=start, n_out=n_out):
                            n_out += 1
                            yield t
                            if n_out >= max_new:
                                return
                        if pstop is not None and pstop.hit:
                            return
                        if sup is not None and ctx[-1] in eos:
                            return
                        pos += 1 + len(think.close_ids)
                        if sup is not None:
                            pos = len(ctx) - 1         # a hidden continue may have followed
                        tok = ctx[-1]
                continue
            block = h2d([tok] + draft, torch.long, prompt.device)
            tv = time.perf_counter()
            lg = eng.forward_block(block, start=pos)
            if pen is not None:
                pen.mask = bool(think is not None and think.inside)
                pen.apply_chain(lg, draft)
            # the same hook the bench loop has: a drafter that prices block widths learns what a
            # width costs from the loop that pays for it (engine/lenrouter.py)
            on_verify = getattr(drafter, "on_verify", None)
            if on_verify is not None:
                on_verify(len(draft) + 1, (time.perf_counter() - tv) * 1e3)
            if sampler is not None and sampler.on:
                # Rejection accept of the chain. A drafter that sampled its proposal carries a q
                # row per token (min(1, p(d)/q(d)), residual (p - q)+); a deterministic
                # arm carries none and gets the shortcut -- draw the target's own token,
                # it matches the draft or it is the token.
                qrows = getattr(drafter, "last_q", None) if drafter is not None else None
                n, x = sampler.chain_accept(sampler.probs_rows(lg), draft, qrows, start=len(ctx))
                new = draft[:n] + [x]
            else:
                picks = lg.argmax(-1).tolist()
                relax = STATE["relax"]
                n = 0
                for i, d in enumerate(draft):
                    if picks[i] == d:
                        n += 1
                        continue
                    if relax.on and relax.accepts(lg[i], d, picks[i]):
                        n += 1
                        continue
                    break
                new = draft[:n] + [picks[n]]
            keep = n + 1                     # the block's rows the commit keeps
            if probe is not None:
                j = next((i for i, t in enumerate(new) if t in eos), None)
                if j is not None:
                    probe.finish(eng, drafter, slice(j, j + 1), n_out + j)
            fin = (_sp2_block(sup, think, ctx, start, n_out, new, eos, max_new, eng.max_len - pos)
                   if sup is not None else None)
            if fin is not None:
                # the end token is not taken: the rows are kept up to the one before it
                keep, new = fin[0] + 1, new[:fin[0]]
            cut = _end_cut(new, eos, max_new - n_out)
            if cut < len(new):
                # SRV-530: rows past the end token (or the budget) are not committed
                if keep > cut + 1:
                    _end_cut_said(keep - cut - 1)
                keep, new = cut + 1, new[:cut]
            if lpr is not None:
                lpr.rows(lg, new)
            if pen is not None:
                pen.commit(new)
            stop_now = pstop is not None and pstop.observe(new)
            if keep < len(draft) + 1:
                eng.rollback_to(keep)
            if drafter is not None and hasattr(drafter, "sync"):
                drafter.sync([int(x) for x in block[:keep]], eng.hidden_post_norm[0, :keep], pos)
            if probe is not None and probe.due(think, len(new)):
                probe.think(eng, drafter, slice(0, keep))
            pos += keep
            bs.block(len(draft), n)
            if drafter is not None:
                drafter.observe(new)
            for t in new:
                ctx.append(t)
                n_out += 1
                yield t
                if t in eos or n_out >= max_new:
                    return
            if stop_now:
                return
            tok = ctx[-1]
            close = None
            if think is not None:
                think.observe(new)
                if sup is not None and fin is None:
                    close = _sp1(sup, think, n_out, eng.max_len - pos, len(ctx))
            if fin is not None or close is not None:
                # SP2's hidden continue, or SP1's hidden close
                if close is not None and close.kind == GIVE_UP:
                    return
                for t in _force_close(eng, drafter, think, ctx, pos, prompt.device, pen=pen,
                                      pstop=pstop, sampler=sampler, lpr=lpr,
                                      act=fin[1] if fin is not None else close, sup=sup, eos=eos,
                                      start=start, n_out=n_out):
                    n_out += 1
                    yield t
                    if n_out >= max_new:
                        return
                if (pstop is not None and pstop.hit) or ctx[-1] in eos:
                    return
                pos = len(ctx) - 1
                tok = ctx[-1]
            elif think is not None:
                if think.hit:
                    for t in _force_close(eng, drafter, think, ctx, pos, prompt.device,
                                          pen=pen, pstop=pstop, sampler=sampler, lpr=lpr,
                                          sup=sup, eos=eos, start=start, n_out=n_out):
                        n_out += 1
                        yield t
                        if n_out >= max_new:
                            return
                    if pstop is not None and pstop.hit:
                        return                         # the guard fired on the phrase
                    if sup is not None and ctx[-1] in eos:
                        return
                    pos += 1 + len(think.close_ids)
                    if sup is not None:
                        pos = len(ctx) - 1             # a hidden continue may have followed
                    tok = ctx[-1]


def context_choice(ids: list[int], letters: list[int], at: int, conv_id: str | None = None,
                   on_chunk=None, images: list | None = None) -> tuple[list[float], dict]:
    """The in-context stage's forward (server/conductor/incontext.py): `ids` prefilled through
    the caches as a turn's prompt is -- the resident prefix's anchors, the store's entries and
    its prefix snapshots (only at lines up to `at`, where the turn's own prompt parts from it),
    the memory gate -- and the logits of its last row at `letters` with how many tokens it
    reused and forwarded. Nothing is decoded and no session snapshot is taken: the turn that
    follows prefills its own prompt over these rows. Under the engine lock.

    `images`: the request's images. Every placeholder is in the conversation before `at`, so the
    stage's prompt expands them exactly as the turn's does (`_mm_prompt`): the same rows, the same
    rotary coordinates, the same `key_ids` the caches compare, and the tower's output from the
    embedding cache when the turn asks for it again."""
    with torch.no_grad():
        logits, reused, forwarded = _context_prefill(ids, at, conv_id, on_chunk, images)
        row = logits[0, -1].float()
        picked = row[torch.tensor(letters, device=row.device)].tolist()
    return [float(x) for x in picked], {"reused": int(reused), "forwarded": int(forwarded),
                                        "model": STATE.get("model")}


def context_rows(ids: list[int], conv_id: str | None = None, on_chunk=None,
                 images: list | None = None) -> tuple:
    """The probe stage's forward (ENG-181): the turn's prompt up to its insertion point
    prefilled as `context_choice` prefills its own, and the rows of its last position -- the
    tapped layers and the post-norm row, fp32 on the host -- with their names and the counts.
    Under the engine lock."""
    eng, drafter = STATE["engine"], STATE["drafter"]
    with torch.no_grad():
        _, reused, forwarded = _context_prefill(ids, len(ids), conv_id, on_chunk, images)
        names, _, rows = probe_mod.rows_of(eng, drafter)
        vec = torch.stack([r[-1] for r in rows]).float().cpu() if rows else None
    return vec, names, {"reused": int(reused), "forwarded": int(forwarded)}


def _context_prefill(ids: list[int], at: int, conv_id, on_chunk, images):
    """`cache.prefill` of a stage's prompt as a turn's, store snapshots only up to `at`."""
    eng, drafter = STATE["engine"], STATE["drafter"]
    eng.mm = None
    eng.pos_delta = 0
    key_ids, mm = None, None
    res = STATE.get("resident")
    if res is not None and not images:
        _spill_before_replay(ids)
    if res is not None and res.end is not None and not images:
        # the end anchor's hidden spans, as the turn's own prompt gets them: all sit before `at`
        full, spans, aligned = _end_replay(res, ids)
        if spans or aligned:
            at += len(full) - len(ids)
            ids = full
    end_mm = False
    if images:
        text = torch.tensor(ids, dtype=torch.long)
        full_t, mm = _mm_prompt(text, images)
        if MM_END == "on" and res is not None:
            # SRV-546: as the turn's own prompt: the NVMe entry read back and the end anchor's
            # replay on the key ids, so the stage resumes at the end and leaves it for the turn
            # (live 2026-10-17 12:2x: a new user turn with a screenshot ran this stage from the
            # grid, its prefill took the rows, and the turn after it re-read 6,448 tokens)
            end_mm = True
            try:
                _spill_before_replay(list(mm.key_ids), mm_ok=True)
                if res.end is not None:
                    full_t, mm, _, _, _ = _mm_end_replay(res, text, full_t, mm, images)
            except Exception:                                      # noqa: BLE001
                logbuf.print_exc()
        full = full_t.tolist()
        at += len(full) - len(ids)
        ids, key_ids = full, mm.key_ids
        eng.mm, eng.pos_delta = mm, mm.delta
    mem = STATE.get("mem")
    if mem is not None:
        memlimits.refresh(STATE)
        mem.gate(len(ids))
    if drafter is not None:
        drafter.reset()           # as a request's start: its caches resume from the rows
    logits, reused, forwarded = _stage_prefill(
        eng, drafter, ids, STATE.get("device", "cuda"), store=STATE.get("state_store"),
        chunk=STATE.get("prefix_chunk", 0), conv_id=conv_id,
        checkpoint=bool(STATE.get("prefix_cache")), resident=STATE.get("resident"),
        on_chunk=on_chunk, mem=mem, checkpoint_upto=at, key_ids=key_ids,
        disk=_slot_disk(), end_mm=end_mm)
    if mm is not None:
        mm.release()
        eng.mm, eng.pos_delta = None, 0
    return logits, reused, forwarded


def _ask_in_place(eng, pos: int, device, ids: list[int], letters: list[int]) -> list[float]:
    """E2's question to Qwen itself, inside the conversation the request holds: `ids` (a hidden
    user turn and the closed reasoning block) forwarded after the KV's `pos` rows, the last row's
    logits at `letters` -- the whole answer of a one-letter choice -- and the engine put back
    where it was: the recurrent state copied back, the KV's length reset (the rows above it are
    written over by the next forward). The drafter, `ctx` and the client never see it. Under the
    engine lock, as the request is."""
    from engine.model import h2d
    if pos + len(ids) >= eng.max_len:
        raise incontext_mod.ContextError("no_room")
    eng.settle()                      # as `cache.capture`: a walked state that is not in yet
    saved, length = cache.state_clone(eng), eng.kv.length
    tap, eng.tap = getattr(eng, "tap", None), None      # the probe's prefill tap reads no question
    try:
        with torch.no_grad():
            lg = eng.forward(h2d(ids, torch.long, device), start=pos, last_only=True)
            row = lg[0, -1].float()
            return [float(x) for x in row[torch.tensor(letters, device=row.device)].tolist()]
    finally:
        eng.tap = tap
        cache.state_load(eng, saved)
        eng.resume_at(length)


def _force_close(eng, drafter, think, ctx, pos, device, pen=None, pstop=None, sampler=None,
                 lpr=None, act=None, sup=None, eos=(), start=0, n_out=0):
    """Close the reasoning block for the model and take the first token of its answer.

    The forced tokens are run through the engine exactly as generated ones are -- one forward at
    `pos` -- so the KV, the recurrent state and the drafter's context all carry them, and the
    answer that follows is conditioned on a block that really does end where it appears to.

    Without `act` the tokens are the budget's phrase, and all of them are streamed: a client's
    own `max_reasoning_tokens` asked for it. With the supervisor's `act` (a hidden close or a
    hidden continue) they are `act.ids`, and only `act.visible` is yielded. With a supervisor,
    an end token drawn after the forced tokens is SP2 like any other, and a hidden continue
    forwards the next action's tokens here in turn. `n_out`: the tokens the caller had yielded,
    so SP2 here knows how many the client was sent (SP3's cut of a regenerated answer).
    """
    # `ctx[-1]` is the last committed token and its own forward has not happened yet -- that is
    # the loop's invariant, `len(ctx) == pos + 1`, and it is why the next verify block starts with
    # it. The forced pass has to carry it, or the closing phrase would be written over its position.
    # After a dropped end token the KV already holds all of `ctx` (`pos == len(ctx)`) and the
    # forced pass is the action's tokens alone: `ctx[pos:]` is what the KV does not hold yet.
    from engine.model import h2d
    _release_held(sup)                     # SP2's line, after the block's tokens before the end
    while True:
        if act is not None and act.detail.get("ask"):
            # SP2's claim check (E2): the KV holds all of `ctx` here (the end token was not taken),
            # so Qwen's questions go in after it and come back out; then the note, or the end
            # token as it was drawn, which ends the answer as it would have without the check
            pend, sup.forwarding = act, act
            act = sup.settle_claim(lambda ids, letters: _ask_in_place(eng, pos, device, ids,
                                                                      letters))
            sup.forwarding = None
            if act is None:
                end = pend.detail["end"]
                ctx.append(end)
                if pen is not None:
                    pen.commit([end])
                if pstop is not None:
                    pstop.observe([end])
                if drafter is not None:
                    drafter.observe([end])
                yield end
                return
            act.detail["t_end"] = pend.detail["t_end"]
        closing = list(think.close_ids) if act is None else list(act.ids)
        shown = closing if act is None else list(act.visible)
        print(f"[think] closed the reasoning block: reason={think.reason or 'budget'} "
              f"at {think.n} tokens" if act is None else
              f"[think] {act.kind.lower()} ({act.reason}) at "
              f"{think.n if think is not None else 0} reasoning tokens, {act.hidden} hidden",
              flush=True)
        if think is not None:
            think.t_forced = time.perf_counter()           # once per forced close
        if act is not None:
            sup.forwarding = act                           # the live view's `supervising`
        forced = [int(t) for t in ctx[pos:]] + closing
        lg = eng.forward(h2d(forced, torch.long, device), start=pos, last_only=True)
        if act is not None:
            sup.forwarding, sup.forwarded_at = None, time.perf_counter()
        if STATE.get("blocks") is not None:
            STATE["blocks"].block()
        if drafter is not None and hasattr(drafter, "sync"):
            drafter.sync(forced, eng.hidden_post_norm[0], pos)
        if drafter is not None:
            drafter.observe(closing)
        if act is not None and len(shown) < len(closing):
            # the hidden part, which the client's copy of the answer will not carry (the end
            # anchor's replay puts it back there)
            STATE.setdefault("hidden_spans", []).append(
                (len(ctx), closing[:len(closing) - len(shown)]))
        ctx.extend(closing)
        if pen is not None:
            # The forced tokens join the history -- they are real context -- but no penalty was
            # consulted in choosing them, which is the point of forcing. The token AFTER them is a
            # genuine decision, so it is penalized against a history that includes them.
            pen.commit(closing)
            pen.apply_single(lg[0, -1])
        if think is not None:
            think.observe(closing)
        if lpr is not None:
            lpr.forced(shown)
        if pstop is not None and pstop.observe(shown):
            # The guard fired on the phrase itself, and a guard hit ENDS the generation: no answer
            # token is drawn, and the caller returns on `pstop.hit`. It used to carry on
            # decoding instead, and its next step forwarded `ctx[-1]` -- the phrase's last token,
            # already written by the forward above -- a second time, one row further on. Ending
            # here leaves the KV holding exactly `ctx`, every token once.
            for t in shown:
                yield t
            return
        for t in shown:
            yield t
        n_out += len(shown)
        nxt = (sampler(lg[0, -1], index=len(ctx)) if sampler is not None and sampler.on
               else int(lg[0, -1].argmax()))
        if act is not None and "t_end" in act.detail:
            # SP2's measure: from the end token to the first token of the continued answer
            sup.resume_ms.append(round((time.perf_counter() - act.detail.pop("t_end")) * 1e3, 1))
        if sup is not None and nxt in eos:
            act = _sp2(sup, think, ctx, start, n_out, eng.max_len - len(ctx), end=nxt)
            if act is not None:
                pos = len(ctx)                  # the KV holds `ctx`; the end token is not taken
                continue
        if lpr is not None:
            lpr.rows(lg[0, -1:], [nxt])
        ctx.append(nxt)
        if pen is not None:
            pen.commit([nxt])
        if drafter is not None:
            drafter.observe([nxt])
        yield nxt
        return


def _sp1(sup, think, n_out: int, room: int, pos: int = 0):
    """SP1 after a block: the supervisor's close (or GIVE_UP) while the reasoning block is open,
    else None. A client's own budget that is spent closes first, with its streamed phrase."""
    if think is None or not think.inside or think.hit or not sup.watches_blocks:
        return None
    return sup.at_block(conductor.BlockEvent(think.n, True, think.reason, think.share, n_out,
                                             pos), room)


def _sp2(sup, think, ctx, start: int, n_out: int, room: int, extra=(), end=None):
    """SP2: the model drew its end token (`end`), and nothing has taken it yet. `extra` are the
    block's tokens before it that `ctx` does not hold yet; `n_out` the tokens the client was sent.
    The supervisor's hidden continue -- or SP3's regeneration of the answer's calls, whose cut is
    recorded at `n_out` -- else None. GIVE_UP included: the end token is then taken as usual and
    the handler adds the sentence. A claim check (E2) is a hidden continue with nothing to
    forward yet: `_force_close` asks its questions and takes `end` after all when it decides
    nothing."""
    inside = think is not None and think.inside and not think.closes_in(extra)
    answer = "" if inside else _answer(ctx, start, extra, think)
    calls = ""
    sup.lift_ids = None
    if (not answer.strip() and getattr(sup.req, "tools_offered", False)
            and getattr(sup.req, "n", 1) == 1):
        # SRV-199: the calls a turn wrote inside its reasoning and then stopped (E1's case C)
        try:
            ids = _trailing_calls([int(t) for t in ctx[start:]] + [int(t) for t in extra],
                                  think.end_id if think is not None else None)
            if ids:
                sup.lift_ids = ids
                calls = STATE["tok"].decode(ids, skip_special_tokens=False)
        except Exception:                                          # noqa: BLE001
            logbuf.print_exc()
    # the block's tokens before the end token (`extra`) reach the client after this returns: the
    # decision's events (its state line) are held until they have, or the line would land in the
    # middle of them (a lifted call's line inside the call's own text)
    real, held = getattr(sup, "on_event", None), []
    if extra and real is not None:
        sup.on_event = held.append
    try:
        act = sup.at_finish(conductor.FinishEvent(
            "eos", inside, answer, think.n if think is not None else 0, n_out,
            len(ctx) + len(extra), calls=calls), room)
    finally:
        if extra and real is not None:
            sup.on_event = real
    if held:
        if act is not None and act.kind in (CONTINUE_HIDDEN, REGENERATE_CALL):
            sup.held_events = held             # `_release_held`, once the block is out
        else:
            for ev in held:
                real(ev)
    if act is None or act.kind not in (CONTINUE_HIDDEN, REGENERATE_CALL):
        return None
    if act.kind == REGENERATE_CALL:
        sup.cuts.append(n_out)
    if act.detail.get("ask"):
        act.detail["end"] = end
    act.detail["t_end"] = time.perf_counter()
    return act


def _release_held(sup) -> None:
    """SP2's held events (`_sp2`) to the supervisor's hook, now that the block's tokens before the
    end token are out."""
    held = getattr(sup, "held_events", None) if sup is not None else None
    if not held:
        return
    sup.held_events = None
    tell = getattr(sup, "_tell", None)
    for ev in held:
        if tell is not None:
            tell(ev)                           # a failing line is logged, never raised
        elif getattr(sup, "on_event", None) is not None:
            sup.on_event(ev)


_CALL_MARKS: dict = {}


def _trailing_calls(gen: list[int], close_id=None) -> list[int] | None:
    """SRV-199: the model's own tokens of the complete `<tool_call>` blocks its output ends with
    -- whitespace between them, and after the last one only whitespace and the reasoning's closer
    (`close_id`) -- or None. One block cut short, text after the last block, or a block that is
    not closed: None. Token ids, so a lifted call is exactly what the model wrote."""
    tok = STATE["tok"]
    if not _CALL_MARKS:
        _CALL_MARKS.update(open=tok.convert_tokens_to_ids("<tool_call>"),
                           close=tok.convert_tokens_to_ids("</tool_call>"))
    o, c = _CALL_MARKS["open"], _CALL_MARKS["close"]
    if o is None or c is None or o == c or not gen:
        return None

    def blank(t: int) -> bool:
        return t == close_id or not tok.decode([t], skip_special_tokens=False).strip()

    j = len(gen)
    while j > 0 and gen[j - 1] != c:
        if not blank(gen[j - 1]):
            return None
        j -= 1
    end, first, i = j, None, j
    while i > 0 and gen[i - 1] == c:
        k = i - 2
        while k >= 0 and gen[k] not in (o, c):
            k -= 1
        if k < 0 or gen[k] != o:
            break                              # a closer without its opener
        first = i = k
        m = i
        while m > 0 and gen[m - 1] not in (o, c) and not tok.decode(
                [gen[m - 1]], skip_special_tokens=False).strip():
            m -= 1
        if m > 0 and gen[m - 1] == c:
            i = m                              # another block before this one
        else:
            break
    return [int(t) for t in gen[first:end]] if first is not None else None


def _sp2_block(sup, think, ctx, start: int, n_out: int, new: list, eos, max_new: int,
               room: int):
    """SP2 inside a verified block: `(index of the end token in new, action)` when the
    supervisor continues past the first end token, else None. The rows up to the one before it
    stay, so the KV holds `ctx + new[:index]` and the continue forwards the action alone."""
    j = next((i for i, t in enumerate(new) if t in eos), None)
    if j is None or n_out + j >= max_new:
        return None
    act = _sp2(sup, think, ctx, start, n_out + j, room - j - 1, extra=new[:j], end=new[j])
    return (j, act) if act is not None else None


def _argloop_fired(v, tbuf, sup, rec, cid: str) -> bool:
    """The loop guard fired on the open call (server/argloop.py). True: the generation stops
    here, the open block is dropped and `v.text` is the sentence the client gets after it; the
    session's next request carries the hidden note (Supervisor.at_request). A held call never
    reaches the client. A call that streamed live (the coding profile's tool gate off: nothing is
    held) has sent its name and part of its arguments: the client keeps a call whose arguments
    never close, which opencode runs as an `invalid` tool and answers with the parse error, so
    the model's next step comes at once, with the note. That is what the token limit gave too,
    63,000 tokens later. False (logged only): `--argloop log`, or no acting monitor."""
    name = None
    m = re.match(r"\s*<function=([^>\s]+)", tbuf.open_text() or "")
    if m:
        name = m.group(1)
    act = STATE.get("argloop", "on") == "on" and sup is not None and sup.mode == "on"
    v.detail["streamed"] = bool(tbuf.open_streamed)
    print(f"[argloop] {cid} {'stopped' if act else 'would stop'} {name or '?'}: "
          + " ".join(f"{k}={x}" for k, x in v.as_dict().items()), flush=True)
    rec.argloop = dict(v.as_dict(), name=name, acted=act)
    if not act:
        return False
    tbuf.drop_open()
    v.text = argloop_mod.stop_text(v, name)
    sup.argloop_stopped(v, name)
    return True


def _release_roles(safe: bool = False) -> int:
    """The memory gate's `roles` level: the shadow reader hands its model's pool back (its
    sessions to the disk, the model unloaded; it loads again on its next use). The pool's
    reserved bytes, which the gate's next `empty` returns; 0 when there was nothing to give."""
    svc = STATE.get("shadow")
    if svc is None or safe:
        return 0
    rm = dict.get(STATE, "role_mem")
    if rm is not None and "shadow" in rm.by_name:
        # the governor's unload: the shadow stays out until memory came back (server/rolemem.py)
        return int(rm.release("shadow", "memory_gate"))
    try:
        return int(svc.release("memory_gate"))
    except Exception as exc:                                       # noqa: BLE001
        print(f"[memory] the shadow's release failed: {type(exc).__name__}: {exc}", flush=True)
        return 0


def _sp3(sup, tbuf, stopped: bool = False) -> str:
    """The end of a streamed answer with held calls: what SP3 decided goes out. SP2 checked the
    answer when it ended on its end token; one cut short (the token limit, a stop string) is
    checked here, where nothing can be written again. Then every held call is released whole
    (repaired where SP3 repaired it) -- or none, when the turn gave up. Returns the content that
    goes out instead of a call: a block that does not parse and was not repaired, as without the
    checks, and a block the token limit cut (unless C4 gave up on it). `stopped`: the client's
    stop string ended the answer, so a block it cut is the text it was (C4 reads `stop`)."""
    held = tbuf.take_held()
    cut = tbuf.cut_block
    tbuf.cut_block = None
    if not sup.checked and (held or cut is not None):
        sup.check_answer(held + ([{"cut": cut, "stop": stopped}] if cut is not None else []),
                         final=True)
    if sup.gave_up:
        for it in held:
            tbuf.discard(it)
        return ""
    text = ""
    for it in held:
        if "bad" not in it:
            tbuf.release(sup.release_call(it))
            continue
        calls = sup.release_block(it["bad"])
        if calls is None:
            text += tbuf.fallback(it)
        for c in calls or ():
            tbuf.release(c)
    if cut is not None:
        text += TOOL_OPEN + cut
    return text


def _sp3_calls(sup, answer: str, eos: bool, names, types, cuts: list,
               stopped: bool = False) -> tuple[str, list]:
    """A non-streamed answer with held calls: `(content, calls)` as SP3 decided, as `_sp3` does
    for a stream. `answer` is the text after the reasoning; `cuts` are the character offsets
    where a regeneration began (what came before each lost its blocks)."""
    if cuts:
        k = max(cuts)                 # every earlier segment is inside the last one's prefix
        answer = strip_blocks(answer[:k]) + answer[k:]
    items = answer_items(answer, names, eos=eos, types=types)
    if stopped:
        items = [dict(it, stop=True) if "cut" in it else it for it in items]
    if not sup.checked and items:
        sup.check_answer(items, final=True)
    content, calls = parse_tool_calls(answer, names=names, eos=eos, types=types)
    if sup.gave_up:
        return strip_blocks(content), []
    out = []
    for it in items:
        if "function" in it:
            call = sup.release_call(it)
            if not out or out[-1]["function"] != call["function"]:
                out.append(call)
        elif "bad" in it:
            got = sup.release_block(it["bad"])
            if got is not None:
                content = content.replace(TOOL_OPEN + it["bad"] + ("</tool_call>" if it["closed"]
                                                                    else ""), "", 1)
                out.extend(got)
    return content, out


def _outcome(sup, calls: list, ids: list, in_think: bool, finish: str) -> None:
    """After the response: its calls into the session state, and, when the monitor decided
    anything, the outcome into the journal (tool_call, text, empty or gave_up)."""
    now = time.time()
    for c in calls:
        sup.session.record_call(c["function"]["name"], c["function"]["arguments"], now)
    if not sup.events:
        sup.finish("")                     # the checks that found nothing are counted
        return
    if sup.gave_up:
        outcome = "gave_up"
    elif calls:
        outcome = "tool_call"
    else:
        text = STATE["tok"].decode(ids, skip_special_tokens=True)
        outcome = "text" if _reasoning_head(text, in_think)[1].strip() else "empty"
    sup.finish(outcome, finish=finish, resume_ms=sup.resume_ms)


def _answer(ctx, start: int, extra, think) -> str:
    """The text written after the reasoning block (all of it when the prompt did not open one),
    hidden tokens included: after a hidden turn the answer is what follows its empty block."""
    gen = [int(t) for t in ctx[start:]] + [int(t) for t in extra]
    end = think.end_id if think is not None else None
    k = next((i for i in range(len(gen) - 1, -1, -1) if gen[i] == end), -1)
    text = STATE["tok"].decode(gen[k + 1:], skip_special_tokens=True)
    i = text.rfind("</think>") if k < 0 else -1
    return text[i + len("</think>"):] if i >= 0 else text


def _remember(prompt_ids: list[int], out_ids: list[int], conv_id: str | None,
              mm=None, hidden=(), delta: int = 0, served_hidden=None) -> None:
    """After a turn: keep the state it ended in, and add its tokens to the suffix store.

    The resident prefix keeps the end of the generation as its end anchor
    (`ResidentPrefix.keep_end`): `hidden` are the spans of the served prompt the client did not
    send (the end anchor's replay, SP4's note), with the generation's own (`_force_close`). For a
    request with images the end is kept with its rotary offset and offered only to a
    compaction served as the conversation's continuation (server/compaction.py); `delta` is that
    offset for a continuation's own generation, which wrote its rows after the images.

    The loop's invariant at the end of a generation is `kv.length == len(ctx) - 1` -- the last
    token has been decided and not forwarded -- or `len(ctx)` when the end token was a drafted
    row the block verified (the commit stops there, `_end_cut`) -- so what is snapshotted is the
    prefix that really was forwarded, and `generate_stream` publishes that very list rather than one rebuilt here. The next turn's prompt begins with all of it plus the chat template's own glue,
    so it resumes here and pays for the glue and the new message rather than for the conversation.
    """
    eng, drafter = STATE["engine"], STATE["drafter"]
    store = STATE.get("state_store")
    committed = STATE.get("last_ctx") or []
    if mm is not None and len(committed) >= mm.n:
        # the state is keyed by the images' content, not by their placeholders
        committed = list(mm.key_ids) + list(committed[mm.n:])
    if store is not None and STATE.get("session_cache") and eng.kv.length:
        # `StateStore.put` declines when the snapshot is longer than the tokens it is given, which
        # is exactly the abandoned-mid-block case above -- and, before phase 9, was also the
        # ordinary case, because the loop advances `pos` by the whole accepted path and appends to
        # `ctx` one token at a time as it yields them. Every generation that ends inside a block
        # ends with `kv.length` ahead of `len(ctx)`, and every generation ends inside a block, so
        # the store was declining EVERY put and the session cache had never stored anything.
        # `puts: 0, hits: 0, misses: 40` after forty requests is what that looks like from
        # outside, and nothing else in the server complains.
        #
        # It still is the ordinary case: an accepted path runs past the end token or the stop,
        # and the loop returns part way through yielding it (2,753 of 3,100 puts on the
        # 2026-09-30 night were 1 to 4 tokens ahead). With the commit folded into the next verify
        # the recurrent state has not absorbed that block yet, and `--session-trim on` cuts it
        # back to the committed tokens (`Qwen38Engine.trim_pending`). Off by default: a session
        # resume is held to the argmax gate, not to bit equality with a cold prefill, and until
        # the puts succeeded none was ever served.
        n_c = len(committed)
        if STATE.get("session_trim") and eng.kv.length > n_c:
            if eng.trim_pending(n_c):
                if drafter is not None and hasattr(drafter, "state_resume"):
                    drafter.state_resume(n_c)
                store.stats["session_trimmed"] = store.stats.get("session_trimmed", 0) + 1
        before = dict(store.stats)
        big = False
        if eng.kv.length > n_c:
            # SRV-253: a snapshot longer than its key is declined by `put` anyway, and the steps
            # before it did harm: `reserve` evicted other entries for it, the memory policy shed
            # caches for it, and `capture` settled the block's pending commit, after which the end
            # anchor below could not cut the engine back to the committed tokens (`trim_pending`):
            # every conversation small enough for the store (under about 7,400 tokens on the served
            # stack) lost its end anchor whenever its generation stopped inside a block, and the
            # next turn re-read the answer from the last grid line (5 of 45 non-streamed first
            # turns on a 0.4.0 test build, `engine-20261007-060639.log`). Nothing is cloned now.
            store.stats["declined_short"] = store.stats.get("declined_short", 0) + 1
            why = "the engine is ahead of the committed tokens"
        else:
            # room is made before the clone, and the memory policy is asked for it (OPS-47)
            est = cache._snapshot_estimate(eng, eng.kv.length, drafter)
            mem = STATE.get("mem")
            big = bool(store.max_entry) and est > store.max_entry
            room = store.reserve(est)
            admitted = room and (mem is None or mem.admit("store", est))
            snap = cache.capture(eng, drafter, max_bytes=store.max_entry) if admitted else None
            if snap is None:
                # SRV-655: the reason itself. Every decline read "too big for the per-entry cap"
                # (Ornith, 2026-10-20: at kv.length 18, a 0.31 GB snapshot under a 1 GiB cap),
                # while the memory policy had refused it: its ceiling left the caches no room
                if big or not room:
                    store.stats["declined_big"] = store.stats.get("declined_big", 0) + 1
                    why = (f"too big for the per-entry cap ({store.max_entry >> 20} MiB)" if big
                           else f"over the store's budget ({store.budget >> 20} MiB)")
                else:
                    store.stats["declined_mem"] = store.stats.get("declined_mem", 0) + 1
                    big = False
                    why = (f"the memory policy refused its {est >> 20} MiB "
                           f"({cache.mem_words(mem)})")
            else:
                store.put(committed, snap, conv_id, kind="session")
                why = ("the engine is ahead of the committed tokens" if
                       store.stats.get("declined_short", 0) > before.get("declined_short", 0)
                       else f"too big for the per-entry cap ({store.max_entry >> 20} MiB)")
        if STATE.get("verbose") and store.stats["puts"] == before["puts"]:
            if big:
                # FX10: over the per-entry cap is the ordinary case of every conversation longer
                # than about 7,400 tokens; the resident prefix keeps it (its end anchor, the park,
                # the NVMe spill), so it is said once per conversation, not at every turn
                key = conv_id or "-"
                if STATE.get("store_big_said") != key:
                    STATE["store_big_said"] = key
                    print(f"[cache] session snapshot not taken: {n_c} tokens is over the store's "
                          f"per-entry cap ({store.max_entry >> 20} MiB); the resident prefix keeps "
                          f"this conversation (end anchor, park, NVMe spill); said once per "
                          f"conversation", flush=True)
            else:
                print(f"[cache] session put declined, {why}: kv.length={eng.kv.length} "
                      f"ctx={n_c}", flush=True)
        # `declined_big` and `declined_short` in /v1/cache/stats count the same events, for when
        # nobody is reading the log.
    res = STATE.get("resident")
    if res is not None and res.end_on and committed:
        if mm is None and not delta:
            why = res.keep_end(eng, drafter, committed, int(STATE.get("last_prompt_n") or 0),
                               list(hidden) + list(STATE.get("hidden_spans") or []))
            if why == "kept" and res.end is not None:
                # SRV-233: where this served prompt held what its client did not send, for the
                # next turn's count of the client's cached tokens
                n_p = int(STATE.get("last_prompt_n") or 0)
                res.end["served_hidden"] = [(int(a), int(b)) for a, b in (served_hidden or ())
                                            if int(b) <= n_p]
            al = STATE.get("end_align")
            if al is not None:
                al.kept(res, int(STATE.get("last_prompt_n") or 0), why)
        elif mm is not None and MM_END == "on":
            # SRV-546: kept as a text end is, with its hidden spans, the client's missing notes
            # and the alignment's regions (rows coordinates: the expanded prompt's), and marked
            # as kept after images: offered to a request with these images (`end_mm`)
            why = res.keep_end(eng, drafter, committed, int(STATE.get("last_prompt_n") or 0),
                               list(hidden) + list(STATE.get("hidden_spans") or []),
                               delta=int(mm.delta), mm=True)
            if why == "kept" and res.end is not None:
                n_p = int(STATE.get("last_prompt_n") or 0)
                res.end["served_hidden"] = [(int(a), int(b)) for a, b in (served_hidden or ())
                                            if int(b) <= n_p]
            al = STATE.get("end_align")
            if al is not None:
                al.kept(res, int(STATE.get("last_prompt_n") or 0), why)
        else:
            why = res.keep_end(eng, drafter, committed, int(STATE.get("last_prompt_n") or 0),
                               delta=int(mm.delta) if mm is not None else int(delta), mm=True)
        if STATE.get("verbose") and why != "kept":
            more = f", {cache.mem_words(STATE.get('mem'))}" if why == "memory" else ""
            print(f"[cache] end anchor not kept ({why}{more}): kv.length={eng.kv.length} "
                  f"ctx={len(committed)}", flush=True)
        if slots_mod.current() is not None:
            # FX16: on a side slot with a resident prefix, what its next turn on slot 0 would
            # carry over is the resident's own end (`_carry_candidate` reads it from there)
            _side_note_end(res, why, conv_id)
    elif res is None and committed and mm is None and not delta:
        why = _side_keep_end(committed, list(hidden) + list(STATE.get("hidden_spans") or []),
                             served_hidden, conv_id)
        if STATE.get("verbose") and why not in ("kept", "off"):
            print(f"[parallel] side end not kept ({why}): kv.length={eng.kv.length} "
                  f"ctx={len(committed)}", flush=True)
    suffix = STATE.get("suffix_store")
    if suffix is not None:
        if mm is not None:
            # the lookup corpus is text: an image's placeholder rows are not written to it
            pad = mm.tower.cfg.image_token_id
            prompt_ids = [t for t in prompt_ids if t != pad]
        elif delta:
            # A compaction's continuation of a conversation with images carries the
            # images' key ids (engine/vision.py `key_id`, outside the vocabulary and past 32 bits:
            # the store's packing raised on them, live, and the stream lost its last chunk)
            vocab = int(eng.cfg.vocab_size)
            prompt_ids = [t for t in prompt_ids if 0 <= int(t) < vocab]
        if STATE.get("suffix_scope") == "all":
            suffix.append_turn(prompt_ids, out_ids)
        else:
            suffix.append(out_ids)


GLUE_TEXTS = ("\n", "\n\n", "\n\n\n", "\n\n\n\n", " ", "  ", " \n", "\t")


#: FX16: a prompt of at least this many tokens that does not resume at the end anchor says why
#: (`_say_reread`): the 2026-10-08 12:14:30 re-read of 60,814 tokens left no line in the log
REREAD_SAY_TOKENS = int(_S.get("REREAD_SAY_TOKENS") or 8192)


def _say_reread(cid: str, res, copy: list[int], plan, why_mm: str | None = None) -> None:
    """One log line and a journal field for a long prompt the end anchor did not serve: where
    the client's copy parts from the rows (token position and message number, never text) and
    what the alignment refused. The prefill then resumes at the last grid line below it."""
    try:
        if res is None or res.end is None:
            return
        e = int(res.end["at"])
        c = _shared_head(copy, res.tokens[:e])
        if c >= e:
            return          # the copy holds every row: the end anchor serves it, nothing to say
        if c < max(2 * int(getattr(res, "chunk", 1024) or 1024), 2048) and (
                why_mm is None or c < 2048):
            # another conversation's rows: nothing of this copy was ever kept (a request with
            # images that the end replay did not serve says so from 2,048 shared tokens on)
            return
        al = STATE.get("end_align")
        ref = dict(al.last_refusal) if al is not None and al.last_refusal else {}
        marks = _conv_marks()
        k = m = None
        if marks is not None:
            im_s = marks[0]
            k = sum(1 for t in res.tokens[:c] if t == im_s)
            m = sum(1 for t in res.tokens[:e] if t == im_s)
        why = ref.get("why") or ("no_end_align" if al is None else "not_asked")
        print(f"[cache] {cid} re-read ahead: the copy ({len(copy)} tokens) parts from the end "
              f"anchor's rows at token {c} of {e}"
              + (f" (message {k} of {m})" if k is not None else "")
              + (f"; with images not served: {why_mm}" if why_mm else "")
              + f"; alignment refused: {why}"
              + (f" at {ref['at']}" if ref.get("at") is not None else "")
              + "; the prefill resumes at the last line below it", flush=True)
        if plan is not None and plan.row is not None:
            plan.row["extra"]["reread"] = {"parts_at": c, "end": e, "message": k,
                                           "messages": m, "why": why}
    except Exception:                                              # noqa: BLE001
        logbuf.print_exc()


def _end_replay(res, ids: list[int]) -> tuple[list[int], list, int]:
    """The end anchor's replay of a client's copy: the hidden spans put back
    (`ResidentPrefix.replay`), or, with `--end-align on`, the copy aligned to the rows where it
    differs from them in whitespace alone (server/endalign.py); when no pass aligns the whole
    copy, the part up to the highest anchor it aligns (SRV-657). `(ids, spans, tokens aligned)`."""
    al = STATE.get("end_align")
    if al is None:
        out, spans = res.replay(ids)
        return out, spans, 0
    got = al.replay(res, ids)
    if al.last_partial is not None and res.end is not None:
        ref = (al.last_refusal or {})
        print(f"[cache] end alignment partial: the rows served up to token {al.last_partial:,} of "
              f"{int(res.end['at']):,} (the copy parts from them at {ref.get('c', 0):,}; the whole "
              f"refused: {ref.get('why')}" + (f" at {ref['at']:,}" if ref.get("at") is not None else "")
              + ")", flush=True)
    return got


def _side_keep_end(committed, hidden, served_hidden, conv_id) -> str:
    """FX10 (SRV-213): at the end of a request on a side slot, what its next turn on slot 0 needs
    to carry the conversation over instead of reading it again: the tokens the slot's rows hold
    (the served prompt, then what was committed), the hidden spans the client's copy lacks, and
    the slot's grant count. The rows and the state stay in the slot's own buffers; nothing is
    copied here. As `ResidentPrefix.keep_end`: a generation that stopped inside a block is cut back
    to its committed tokens when the commit is still pending (else nothing is kept), and the
    pending commit is settled so the state is the state at the end. Returns kept, off, short or
    ahead."""
    slot = slots_mod.current()
    if slot is None or SIDE_CARRY == "off":
        return "off"
    slot.carry = None
    if getattr(slot, "no_carry", False):
        # FX15: the router's own question (its User-Agent, `me["own"]`, at any length: the
        # package question is longer than the side lane's 2,048) or a side-lane job (a client's
        # title task) is no conversation, however long. The package question (2.4k tokens, 34 options) shares its
        # options with the next one: noted as a conversation, the next question was carried into
        # slot 0 over the chat (which was parked) and kept the next guests off the side slot for
        # CARRY_KEEP_S (live 0.4.7-test b1, W01/W02)
        return "lane"
    eng, drafter = STATE["engine"], STATE["drafter"]
    n_prompt = int(STATE.get("last_prompt_n") or 0)
    e, n_c = int(eng.kv.length), len(committed)
    res0 = dict.get(STATE, "resident")
    chunk = int(getattr(res0, "chunk", 0) or STATE.get("prefix_chunk") or 1024)
    if n_prompt <= 0 or e < n_prompt or n_c < n_prompt or min(e, n_c) < 2 * chunk:
        # a short request (the router's own question, a title task) is no conversation to carry,
        # and its end must not keep the next guest off the side slot (`Pool.take_side`)
        return "short"
    if e > n_c:
        if not eng.trim_pending(n_c):
            return "ahead"
        e = n_c
    eng.settle()
    mark = getattr(drafter, "end_mark", None) if drafter is not None else None
    spans = sorted((int(p), [int(t) for t in h]) for p, h in hidden)
    slot.carry = {"gen": slot.gen, "t": time.monotonic(), "at": e,
                  "tokens": [int(t) for t in committed[:e]], "n_prompt": n_prompt,
                  "hidden": [(p, h) for p, h in spans if h and p + len(h) <= e],
                  "served_hidden": [(int(a), int(b)) for a, b in (served_hidden or ())
                                    if int(b) <= n_prompt],
                  "mark": mark() if mark is not None else None, "owner": conv_id}
    return "kept"


def _side_note_end(res, why: str, conv_id) -> None:
    """FX16: after a generation on a side slot that has a resident prefix: the slot's carry record
    (what `_carry_from_side` copies into slot 0 when the conversation's next turn cannot run here)
    is the resident's own end anchor, with the end alignment's regions. The record is valid while
    the resident still holds that end (`_carry_candidate`), a guest's stash included; nothing is
    kept after a request the router sent here itself (`no_carry`), nor one that kept no end."""
    slot = slots_mod.current()
    if slot is None:
        return
    if getattr(slot, "no_carry", False) or why == "guest":
        return                 # a question, a title task, a guest: the slot's conversation stays
    if why != "kept" or getattr(res, "end", None) is None or SIDE_CARRY == "off":
        slot.carry = None
        return
    e = int(res.end["at"])
    al = STATE.get("end_align")
    slot.carry = {"gen": slot.gen, "t": time.monotonic(), "at": e,
                  "tokens": [int(t) for t in res.tokens[:e]],
                  "n_prompt": int(STATE.get("last_prompt_n") or 0),
                  "hidden": [(int(p), [int(x) for x in h]) for p, h in res.end["hidden"]],
                  "served_hidden": [(int(a), int(b)) for a, b in
                                    (res.end.get("served_hidden") or ())],
                  "mark": res.end.get("mark"), "owner": conv_id, "resident": True,
                  "regions": [tuple(x) for x in (al.regions(res) if al is not None else ())]}
    if res.end.get("mm") or res.end.get("delta"):
        # SRV-629: an end kept after images keeps its rotary offset and its mark on slot 0, where
        # it is offered only to a request with these images and a compaction reads its offset
        slot.carry.update(delta=int(res.end.get("delta") or 0), mm=bool(res.end.get("mm")))


def _carry_live(slot, rec) -> bool:
    """Is a side slot's carry record still what its rows hold? Without a resident prefix: no
    request has had the slot since (`Slot.gen`). With one: the resident still holds that end (a
    guest between left it in place through the stash; another conversation's request cleared it)."""
    if rec is None:
        return False
    if not rec.get("resident"):
        return rec.get("gen") == slot.gen
    res = (slot.values or {}).get("resident")
    try:
        end = getattr(res, "end", None)
        return (end is not None and int(end["at"]) == int(rec["at"])
                and int(getattr(res, "valid", 0) or 0) >= int(rec["at"])
                and res.tokens[:int(rec["at"])] == rec["tokens"])
    except Exception:                                              # noqa: BLE001
        return False


def _carry_candidate(pool, res, ids: list[int], max_len: int):
    """(slot, record) of the side slot whose last request this prompt continues, when carrying it
    beats what slot 0 offers by more than a chunk; else None. Continues = shares the record's
    tokens up to where the client's copy differs by design (the first hidden span, or the end of
    the served prompt: the reasoning a client leaves out, the template's re-render), within one
    chunk."""
    chunk = max(1, int(getattr(res, "chunk", 0) or 1024))
    best = None
    for s in pool.side():
        rec = s.carry
        if rec is None or not _carry_live(s, rec) or int(rec["at"]) > max_len:
            continue
        toks = rec["tokens"]
        first = min([int(rec["n_prompt"])] + [int(p) for p, _ in rec["hidden"]])
        need = max(2 * chunk, first - chunk)
        # one list comparison (it runs for every request on slot 0 while an end is noted)
        if len(ids) < need or len(toks) < need or list(ids[:need]) != list(toks[:need]):
            continue
        if best is None or int(rec["at"]) > int(best[1]["at"]):
            best = (s, rec)
    if best is None:
        return None
    c0 = res.common(ids) if res.valid else 0
    # SRV-629: a prompt with images (its keys past the vocabulary) counts slot 0's end and park
    # kept after images, as its prefill does (`end_mm`)
    from engine.vision import KEY_BASE
    kw = {"mm_ok": True} if ids and int(max(ids)) >= KEY_BASE else {}
    here = max(res.best(c0, max_len), res.end_offer(c0, max_len, **kw),
               res.pin_offer(ids, c0, max_len), res.park_offer(ids, c0, max_len, **kw))
    return best if int(best[1]["at"]) > here + chunk else None


def _carry_from_side(ids: list[int], session=None) -> dict | None:
    """FX10 (SRV-213): on slot 0, before the replay: a conversation whose last request ran on a
    side slot (a new chat or agent session that started as a guest beside slot 0's conversation)
    is carried into slot 0 instead of read again. Measured before: UX2 O07 moved slot 1 -> 0 at
    15,910 tokens and re-read 8.7k of them (18.3 s); UX1 W11 beside an 80k resident took slot 0 at
    cached 0.44 (TTFT 17.0 s).

    The resident conversation leaves first (`ResidentPrefix.vacate`: to the NVMe or the park, as a
    request of another session would make it leave), then the side slot's rows [0, end) of every
    position-indexed buffer are copied into slot 0's (a device copy, about 104 KiB a token: 0.9 GB
    for 8,700 tokens) and its state at the end becomes slot 0's end anchor, with the hidden spans
    and the end alignment's region, as `keep_end` would have kept it had the request run here.
    The replay and the prefill then resume there as at any end anchor; what the side slot's
    verify blocks wrote is held to the same gate as an end anchor's rows (the same greedy tokens),
    and the rows count as a session's (`taint` 0: no prefix snapshot, no disk tier write of them).

    The side slot is taken without waiting and only while no other request has had it since
    (`Slot.gen`). Under the engine lock. Never fails a request: anything amiss and the prompt is
    served from what slot 0 holds."""
    pool, res = dict.get(STATE, "pool"), STATE.get("resident")
    if SIDE_CARRY == "off" or pool is None or res is None or pool.n < 2 or res.chunk <= 0:
        return None
    eng, drafter = STATE["engine"], STATE["drafter"]
    max_len = int(STATE.get("max_len") or 0)
    try:
        got = _carry_candidate(pool, res, ids, max_len)
    except Exception:                                              # noqa: BLE001
        return None
    if got is None:
        return None
    slot, rec = got
    if not slot.lock.acquire(blocking=False):
        return None
    t0 = time.perf_counter()
    try:
        if slot.carry is not rec or not _carry_live(slot, rec):
            return None
        from engine import spill as spill_mod
        sv = slot.values or {}
        seng, sdrafter = sv.get("engine"), sv.get("drafter")
        if seng is None:
            return None
        e = int(rec["at"])
        v1 = cache._kv_views(seng, sdrafter)
        st0, st1 = cache.state_now(eng), cache.state_now(seng)
        cuda = any(t.is_cuda for t in st0)
        if (e > min(t.shape[ax] for t, ax in v1) or [t.shape for t in st1] != [t.shape for t in st0]
                or len(v1) != len(cache._kv_views(eng, drafter))):
            return None
        with torch.no_grad():
            if cuda:
                torch.cuda.synchronize()
            disk = STATE.get("disk_cache")
            if disk is not None:
                spill_mod.cache_fence(disk, 0)
            # slot 0's conversation leaves first (its rows and its drafter's), as the spill's
            # restore saves the resident one before it reads (engine/spill.py `maybe_restore`)
            how = res.vacate(eng, drafter, session if session is not None else rec.get("owner"))
            cache.drafter_resume(eng, drafter, e)   # builds a draft cache that does not exist yet
            v0 = cache._kv_views(eng, drafter)
            if (spill_mod._layout(v0) != spill_mod._layout(v1)
                    or e > min(t.shape[ax] for t, ax in v0)):
                res.clear()
                return None
            for (t, ax), (u, _) in zip(v0, v1):
                kvpool.ensure_for(t, e)                         # SRV-307
                t.narrow(ax, 0, e).copy_(u.narrow(ax, 0, e))
            res.adopt(rec["tokens"], e, 0, rec.get("owner"), count="side_carries")
            if drafter is not None and hasattr(drafter, "state_resume"):
                drafter.state_resume(e)    # again, over the rows carried in (`cache.drafter_resume`)
            pair = res.adopt_end(eng, {"at": e, "hidden": rec["hidden"], "mark": rec["mark"],
                                       "delta": rec.get("delta"), "mm": rec.get("mm")})
            if pair is not None:
                # FX16: with a resident prefix on the side slot the state at the end is its end
                # anchor's pair (a guest served there since wrote the live state); else the live
                # state, which nothing wrote since (`Slot.gen`)
                side_res = sv.get("resident")
                bufs = getattr(side_res, "_end_bufs", None) if side_res is not None else None
                if (rec.get("resident") and bufs is not None
                        and getattr(side_res, "end", None) is not None
                        and int(side_res.end["at"]) == e):
                    cache.state_copy(pair, bufs)
                else:
                    cache.state_save(pair, seng)
                res.end["served_hidden"] = list(rec.get("served_hidden") or ())
                al = STATE.get("end_align")
                if al is not None:
                    # FX16: the side slot's own regions when it kept them (every generation of
                    # the conversation served there), else the last one's
                    al.adopt(res, list(rec.get("regions") or ()) or [(int(rec["n_prompt"]), e)])
            if cuda:
                torch.cuda.synchronize()
        slot.carry = None
        side_res = sv.get("resident")
        if side_res is not None:
            # the rows are slot 0's conversation now; the side slot holds nobody's
            try:
                side_res.clear()
            except Exception:                                      # noqa: BLE001
                pass
            sv["slot_view"] = None
        info = {"slot": slot.index, "tokens": e, "end": pair is not None, "resident": how,
                "ms": round((time.perf_counter() - t0) * 1e3, 1)}
        STATE["side_carried"] = info
        print(f"[parallel] carried slot {slot.index}'s conversation into slot 0: {e} tokens, "
              f"end anchor {'kept' if pair is not None else 'refused'}, slot 0's conversation "
              f"{how}, {info['ms']:.0f} ms", flush=True)
        return info
    except Exception as exc:                                       # noqa: BLE001
        print(f"[parallel] carry failed: {type(exc).__name__}: {exc}", flush=True)
        try:
            res.clear()                    # the rows in place are nobody's any more
        except Exception:                                          # noqa: BLE001
            pass
        return None
    finally:
        slot.lock.release()


def _spill_before_replay(ids: list[int], mm_ok: bool = False) -> None:
    """A conversation on the NVMe (engine/spill.py) that this prompt continues further
    than every cache in RAM is read back into the engine's buffers before the end anchor's replay,
    the hidden spans counted as the replay counts them. Under the engine lock. Never fails a
    request: the prefill then starts from what is in place."""
    sp, res = STATE.get("spill"), STATE.get("resident")
    if sp is None or res is None:
        return                 # FX16: with no entry the park's end still comes back here
    parked = int(res.stats.get("park_tokens_reused", 0))
    try:
        with torch.no_grad():
            info = sp.maybe_restore(res, STATE["engine"], STATE["drafter"], ids, len(ids) - 1,
                                    replay=True, disk=STATE.get("disk_cache"), mm_ok=mm_ok)
    except Exception as exc:                                       # noqa: BLE001
        print(f"[spill] restore failed: {type(exc).__name__}: {exc}", flush=True)
        return
    if info:
        STATE["spill_restored"] = info
    elif int(res.stats.get("park_tokens_reused", 0)) > parked:
        STATE["park_returned"] = True      # the parked end came back in place (usage.tandem.cache)


def _peek_ids(body: dict, headers) -> tuple[list[int], str | None]:
    """SRV-349: the prompt a chat body would be prefilled as, without the engine: rendered on the
    host, the session's router notes replayed where its next turn carries them; and the session
    (the body's `session`, else the caches' key a request would use: the body's conversation or
    `x-session-id`)."""
    session = body.get("session") if isinstance(body.get("session"), str) else None
    session = (session or conversation_id(body, headers) or _session_header(headers) or None)
    ids = build_prompt(body, count="ids")
    cond = STATE.get("conductor")
    if cond is not None and session:
        try:
            ids = cond.router.replayed_for("c:" + session[:120], ids, STATE["tok"])
        except Exception:                                          # noqa: BLE001
            logbuf.print_exc()             # the peek reads the copy without the notes
    return ids, session


def _peek_find(ids: list[int]) -> dict:
    """`cache_peek.find` over this process's tiers, as `cache.prefill` would ask them for slot 0.
    Under the engine lock."""
    res, disk = STATE.get("resident"), STATE.get("disk_cache")
    if not cache.resident_capable(STATE.get("drafter")):
        res = disk = None
    side, pool = None, dict.get(STATE, "pool")
    if SIDE_CARRY != "off" and pool is not None and res is not None and pool.n >= 2 \
            and res.chunk > 0:
        max_len = int(STATE.get("max_len") or 0)

        def side(x):
            got = _carry_candidate(pool, res, x, max_len)
            return int(got[1]["at"]) if got is not None else 0
    return cache_peek.find(ids, res=res, spill=STATE.get("spill") if res is not None else None,
                           store=STATE.get("state_store"), disk=disk, side=side)


def _peek_rate() -> float | None:
    """The prefill rate a re-read is estimated at: this process's measured one, else the caches'
    prior (`--disk-cache` / the spill's `prefill_tok_s`)."""
    r = cache_peek.RATES.rate()
    if r:
        return r
    for name in ("spill", "disk_cache"):
        t = STATE.get(name)
        if t is not None and getattr(t, "prefill_tok_s", None):
            return float(t.prefill_tok_s)
    return None


#: with no request waiting, a peek or a prewarm waits this long for the engine lock (s): the end of a
#: turn (its bookkeeping after the answer went out) and the engine's idle work after it
PEEK_IDLE_WAIT_S = 3.0


def _lock_when_idle() -> bool:
    """The engine lock for a peek or a prewarm: without waiting while a request waits for it (the
    route's 409), else for up to PEEK_IDLE_WAIT_S (live: right after a turn its own tail and then
    the idle graph captures, 1,339 ms, held it, and five peeks in a row answered 409)."""
    with QUEUE:
        queued = bool(WAITERS) or INFLIGHT["waiting"] > 0
    return LOCK.acquire(blocking=False) if queued else LOCK.acquire(timeout=PEEK_IDLE_WAIT_S)


def _cache_peek(ids: list[int], session: str | None) -> dict | None:
    """The peek's answer, or None while the engine is busy (a prefill writes the rows it reads)."""
    if not _lock_when_idle():
        return None
    try:
        got = _peek_find(ids)
    finally:
        LOCK.release()
    return cache_peek.answer(ids, got, session, _peek_rate())


def _request_waiting() -> bool:
    """A request waits for the engine (the prewarm's yield, SRV-349): queued, or running on a side
    slot, whose GPU turns the prewarm holds with the engine lock (live: a short request that took
    a side slot waited 1,246 ms behind a prewarm's save and restore)."""
    return bool(WAITERS) or INFLIGHT["waiting"] > 0 or INFLIGHT["running"] > 0


def _cache_prewarm(ids: list[int], session: str | None) -> dict | None:
    """SRV-349: the conversation read back from the NVMe while nothing runs (server/cache_peek.py
    THE PREWARM). None while the engine is busy."""
    with QUEUE:
        idle = not WAITERS and INFLIGHT["running"] == 0 and INFLIGHT["waiting"] == 0
    if not idle or not _lock_when_idle():
        return None
    try:
        got = _peek_find(ids)
        out = cache_peek.answer(ids, got, session, _peek_rate())
        sp, res = STATE.get("spill"), STATE.get("resident")
        why = None
        if got["tier"] != "nvme_spill" or sp is None or res is None:
            why = "not_on_nvme"
        elif (res.valid and res.owner is not None and res.owner != session
              and time.time() - float(STATE.get("last_request_end") or 0) < cache_peek.IDLE_S):
            why = "resident_in_use"
        elif float(got.get("restore_ms") or 0) > cache_peek.MAX_MS:
            why = "too_long"
        if why is not None:
            return {"prewarmed": False, "why": why, **out}
        with torch.no_grad():
            # a request that arrives meanwhile waits at most one block: the save and the restore
            # stop between blocks when one is waiting (engine/spill.py `stop`)
            info = sp.maybe_restore(res, STATE["engine"], STATE["drafter"], ids, len(ids) - 1,
                                    replay=True, disk=STATE.get("disk_cache"), stop=_request_waiting)
    except Exception as exc:                                       # noqa: BLE001
        print(f"[spill] prewarm failed: {type(exc).__name__}: {exc}", flush=True)
        return {"prewarmed": False, "why": "error", "tier": "unknown", "session": session}
    finally:
        LOCK.release()
    if not info:
        return {"prewarmed": False, "why": "yielded" if _request_waiting() else "refused", **out}
    print(f"[spill] prewarm: {session or '-'} read back, {info['tokens']} tokens, "
          f"{info['blocks_read']} blocks read in {info['ms']:.0f} ms", flush=True)
    return {"prewarmed": True, "restore": {k: info.get(k) for k in
                                           ("tokens", "line", "blocks_read", "bytes", "ms")},
            **out}


def _drain_side_slots(sp) -> list:
    """rc-0.8.0: each side slot's conversation to the NVMe too (reason `drain-side`, so the start
    reads back slot 0's), so a chat that last ran on a side slot comes back from the spill after a
    restart instead of cold: its next turn waits for slot 0, whose prefill reads the spill
    (`_par_side_need`). Under the engine lock, after the work in flight."""
    pool, out = dict.get(STATE, "pool"), []
    for s in (pool.side() if pool is not None else ()):
        sv = s.values or {}
        r, seng = sv.get("resident"), sv.get("engine")
        if r is None or seng is None or r.valid < int(sp.cfg["drain_min_tokens"]):
            continue
        if r._stash is not None:
            r.settle_guest(seng)
        out.append((f"slot {s.index}'s", sp.save_resident(r, seng, sv.get("drafter"), "drain-side")))
    return out


def spill_drain() -> None:
    """At a planned stop, after the work in flight: the resident conversation and the parked one
    to the NVMe (engine/spill.py), for the next start."""
    sp, res = STATE.get("spill"), STATE.get("resident")
    if sp is None or res is None:
        return
    eng, drafter = STATE.get("engine"), STATE.get("drafter")
    t0 = time.perf_counter()
    try:
        with LOCK, torch.no_grad():
            disk = STATE.get("disk_cache")
            if res._stash is not None:
                if disk is not None:
                    disk.fence(0)
                res.settle_guest(eng)
            got = []
            if res.valid >= int(sp.cfg["drain_min_tokens"]):
                got.append(("resident", sp.save_resident(res, eng, drafter, "drain")))
            top = res.park_base + res.park_rows
            if res.park_rows and top >= int(sp.cfg["drain_min_tokens"]):
                got.append(("parked", sp.save_park(res, eng, drafter, "drain")))
            got += _drain_side_slots(sp)
    except Exception as exc:                                       # noqa: BLE001
        print(f"[spill] drain failed: {type(exc).__name__}: {exc}", flush=True)
        return
    for what, info in got:
        if info:
            print(f"[spill] drain: the {what} conversation, {info['tokens']} tokens, "
                  f"{info['blocks_written']} blocks written, {info['blocks_kept']} kept, "
                  f"{info['bytes_written'] / 1e9:.2f} GB in {info['ms']:.0f} ms", flush=True)
        else:
            print(f"[spill] drain: the {what} conversation not written", flush=True)
    if not got:
        print(f"[spill] drain: nothing resident to keep ({res.valid} tokens)", flush=True)
    sp.close()
    print(f"[spill] drain done in {(time.perf_counter() - t0) * 1e3:.0f} ms", flush=True)


def spill_startup() -> None:
    """Before the first request: the resident conversation the last planned stop wrote, read back
    into the engine's buffers as the resident prompt (its anchors, its end anchor)."""
    sp, res = STATE.get("spill"), STATE.get("resident")
    if sp is None or res is None:
        return
    e = sp.drained()
    if e is None:
        print(f"[spill] start: no drained conversation to read back ({len(sp.entries)} on "
              f"the disk)", flush=True)
        return
    try:
        with LOCK, torch.no_grad():
            info = sp.restore(res, STATE["engine"], STATE["drafter"], e,
                              disk=STATE.get("disk_cache"), line=e.valid)
    except Exception as exc:                                       # noqa: BLE001
        print(f"[spill] start: restore failed: {type(exc).__name__}: {exc}", flush=True)
        return
    if info:
        print(f"[spill] start: read back the drained conversation, {info['tokens']} tokens, "
              f"{info['bytes'] / 1e9:.2f} GB in {info['ms']:.0f} ms (end anchor {info['end']}, "
              f"anchors {info['anchors']})", flush=True)
    else:
        print("[spill] start: the drained conversation could not be read back", flush=True)


def glue_ids(tok) -> frozenset:
    """The tokenizer's single tokens that are whitespace only (`GLUE_TEXTS`): where a client's
    re-rendered copy of an answer may differ from the rows next to a hidden span
    (`ResidentPrefix.replay`)."""
    out = set()
    for text in GLUE_TEXTS:
        try:
            ids = list(tok(text, add_special_tokens=False).input_ids)
        except Exception:                                          # noqa: BLE001
            continue
        if len(ids) == 1:
            out.add(int(ids[0]))
    return frozenset(out)


def _inserted(before: list[int], after: list[int]) -> tuple | None:
    """`(position, ids)` when `after` is `before` with one run of tokens inserted (SP4's hidden
    note), else None."""
    k = len(after) - len(before)
    if k <= 0:
        return None
    i = 0
    m = len(before)
    while i < m and before[i] == after[i]:
        i += 1
    return (i, after[i:i + k]) if after[i + k:] == before[i:] else None


def _session_header(headers) -> str | None:
    """Opencode's session id (`x-session-id`, sent with every request of a session): the live
    view's `continues` link, and (OPS-47) the session key of the caches when the body names no
    conversation -- the store files entries under it, the resident prefix knows whose rows it
    holds and parks another session's. A wrong id costs a cache hit, never a wrong answer: every
    start is checked against the prompt's own tokens."""
    try:
        v = headers.get("x-session-id") or headers.get("X-Session-Id")
        # SRV-285 (FX15): Open WebUI's chat id, which it sends with every request of a chat when its
        # connection has `X-OpenWebUI-Chat-Id: {{CHAT_ID}}` (the router already keys its session on
        # it). Without it every Open WebUI spill entry had no owner, and a chat that shared a pasted
        # document with another took the other's entry (F57: back in 6.9 s, 1.8 s on 0.4.1)
        if not v:
            w = headers.get("X-OpenWebUI-Chat-Id")
            v = "c:" + w if isinstance(w, str) and w else None
    except Exception:                                              # noqa: BLE001
        v = None
    return v[:128] if isinstance(v, str) and v else None


#: ENG-371: the sessions an agent's worker names as its parent (`X-Tandem-Parent-Session`,
#: chatd/render.py `headers`), each with the time of its worker's last request. The parent comes
#: back when its worker ends, so while it waits its rows go to the NVMe rather than being dropped:
#: from a side slot another session's prefill takes (`_keep_parent`), and from slot 0's park when
#: the memory policy gives the park back (`_shed_park`). An entry lasts PARENT_KEEP_S past its
#: worker's last request.
PARENTS: dict = {}
#: the workers' own sessions, the same way: a worker comes back at its next call, and its rows on a
#: side slot are written before another session's prefill cuts them too
WORKERS: dict = {}
PARENT_KEEP_S = 600.0
#: the longest a returning parent waits for slot 0 before a side slot reads its prompt again, and
#: the prefill rate that wait is estimated at when the engine has measured none (`_par_side_need`)
PARENT_WAIT_MAX_S = 60.0
PARENT_REREAD_TOK_S = 700.0


def _worker_of(headers) -> tuple[bool, str | None]:
    """ENG-371: whether the request is an agent's worker (`X-Tandem-Worker: 1`, or a parent
    session named), and the parent's session key. A worker's session comes back at its next call,
    so the caches never serve it as a guest (engine/cache.py `ResidentPrefix.begin`, `stay`)."""
    try:
        w = (headers.get("X-Tandem-Worker") or "").strip() == "1"
        p = headers.get("X-Tandem-Parent-Session")
    except Exception:                                              # noqa: BLE001
        return False, None
    p = p[:128] if isinstance(p, str) and p else None
    return w or p is not None, p


def _note_parent(parent: str | None, now: float | None = None, worker: str | None = None) -> None:
    """A worker's request: its parent waits for it from now, and the worker's own session comes
    back at its next call (stale entries go)."""
    now = time.monotonic() if now is None else now
    for d, k in ((PARENTS, parent), (WORKERS, worker)):
        if not k:
            continue
        for old in [x for x, t in list(d.items()) if now - t >= PARENT_KEEP_S]:
            d.pop(old, None)
        d[k] = now


def _waiting_parent(owner, now: float | None = None) -> bool:
    """Is `owner` (a cache's session key) a parent whose worker ran within PARENT_KEEP_S?"""
    t = PARENTS.get(owner) if owner else None
    return t is not None and (time.monotonic() if now is None else now) - t < PARENT_KEEP_S


def _comes_back(owner, now: float | None = None) -> bool:
    """A waiting parent, or a worker whose last call was within PARENT_KEEP_S: rows to keep."""
    t = WORKERS.get(owner) if owner else None
    return _waiting_parent(owner, now) or (
        t is not None and (time.monotonic() if now is None else now) - t < PARENT_KEEP_S)


#: how long a side slot's write of the rows it is about to cut waits for slot 0's spill I/O
KEEP_IO_WAIT_S = 10.0


def _keep_parent(res, eng, drafter, stay: bool = False) -> None:
    """ENG-371 (a side slot's `ResidentPrefix.keeper`): before another session's prefill cuts a
    side slot's rows, a parent's or a worker's go to the NVMe (a side slot has no park and no
    spill of its own), as a planned stop writes them (`_drain_side_slots`); so does any client's
    conversation a worker's request (`stay`, never a guest) cuts. They come back on slot 0
    (`_par_side_need`'s `parent` and `spill` waits) and the request prefills its tail. Live
    2026-10-17 23:39: a worker's second call took the side slot its parent's 15,066 rows were on,
    and the parent's return re-read 11,039 of 16,159 tokens; 2026-10-18 01:1x one chat's worker
    took the side slot the other chat's worker ran on, whose next call read 6,115 tokens again."""
    sp = dict.get(STATE, "spill")
    if (sp is None or not (stay or _comes_back(res.owner)) or res._stash is not None
            or res.valid < int(sp.cfg.get("drain_min_tokens", 4096))):
        return
    if sp._own_io.locked():
        # slot 0's save or restore holds the spill's own I/O: the write waits for it (this side
        # thread holds no lock it needs), at most KEEP_IO_WAIT_S, then the rows go unwritten
        if not sp._own_io.acquire(timeout=KEEP_IO_WAIT_S):
            print(f"[spill] {res.owner}: not written, the spill stayed busy "
                  f"{KEEP_IO_WAIT_S:g} s", flush=True)
            return
        sp._own_io.release()
    # a side slot's thread, outside the engine lock: no CUDA call inside slot 0's graph capture
    # (engine/capture_lock.py; live 2026-10-18 01:16 this save broke a verify graph's capture on
    # slot 0, and its request ended in an error)
    with capture_lock.LOCK, torch.no_grad():
        info = sp.save_resident(res, eng, drafter, "switch")
    slot = slots_mod.current()
    where = f"slot {slot.index}" if slot is not None else "slot 0"
    who = ("parent" if _waiting_parent(res.owner) else "worker" if _comes_back(res.owner)
           else "conversation")
    if info:
        print(f"[spill] {who} {res.owner} on {where}: {info['tokens']} tokens to the NVMe before "
              f"another session's prefill, {info['ms']:.0f} ms", flush=True)
    else:
        print(f"[spill] {who} {res.owner} on {where}: not written, its rows go", flush=True)


#: ENG-371: this thread's gate retry lets a waited-for park go (`on`); a shed kept one (`kept`)
_PARK_LET_GO = threading.local()


def _shed_park(res, safe: bool = False) -> int:
    """The memory policy's `park` level. ENG-371: a parked parent or worker (`_comes_back`) is
    written to the NVMe first (`ResidentPrefix._spill_park`) at the memory gate on slot 0's own
    thread, where nothing else writes slot 0's rows below the park. Live 2026-10-17 23:05: the
    gate gave a parked 15,417-token parent back at the ceiling, unwritten, and its return re-read
    11,895 of 17,015 tokens. Nothing is written, and the park goes as before:
      * with `safe`: a cache's own operation, or the KV pool's page gate, which runs inside a
        restore, a swap or a forward (a write there waited on the spill's own I/O lock, or wrote
        half-swapped rows, or would need the guest's rows put back under the running guest);
      * on a side slot's thread, or while the spill's I/O is held;
      * while a guest's rows lie over the park's head (the next request puts them back,
        `settle_guest`; a shedder never does).
    A parked parent or worker is not given back at all where it could not be written, unless a
    refused gate runs once more (`_gate_holding_end`); slot 0's next gate writes it before it
    goes. The write runs inside the gate, under `Budget.lock` (docs/caches.md)."""
    sp, st = res.spiller, res._stash
    # `_park_busy` is read without a lock on a side slot's thread: a stale read costs a park
    # kept or written once more, never rows (`shed_park` checks it again where it frees them)
    if (not safe and res.park_rows and not res._park_busy and _comes_back(res.park_owner)
            and slots_mod.current() is None and sp is not None and not sp._own_io.locked()
            and (st is None or int(st["start"]) >= int(res.park_base))):
        eng, drafter = dict.get(STATE, "engine"), dict.get(STATE, "drafter")
        try:
            with torch.no_grad():
                owner, rows = res.park_owner, res.park_base + res.park_rows
                ok = res._spill_park(eng, drafter)
            print(f"[spill] parked {owner}: {rows} tokens "
                  f"{'to the NVMe' if ok else 'not written'} before the memory policy takes the "
                  f"park", flush=True)
        except Exception:                                          # noqa: BLE001
            logbuf.print_exc()
    if (res.park_rows and _comes_back(res.park_owner) and not getattr(_PARK_LET_GO, "on", False)
            and (safe or slots_mod.current() is not None)):
        # it is waited for and could not be written here: a pin's admission, the KV pool's
        # reclaim or a side slot's gate does not drop it unwritten (live 2026-10-18 04:43: a pin's
        # admission dropped a worker's parked 11,465 rows; 05:02 a side slot's gate a parent's,
        # and each read them again); slot 0's next gate writes it before it goes. A gate that
        # refuses with it kept runs once more and may take it (`_gate_holding_end`)
        _PARK_LET_GO.kept = True
        return 0
    return res.shed_park(safe)


def _end_held(res, keys: list[int], n: int, mm_ok: bool) -> bool:
    """ENG-371: whether a request of `keys` (`n` served rows) resumes at the end anchor in place."""
    return (res is not None and res.end is not None
            and res.end_offer(res.common(keys), n - 1, mm_ok=mm_ok) > 0)


def _gate_holding_end(mem, res, keys: list[int], n: int, mm_ok: bool) -> None:
    """The memory gate (`Budget.gate`) with the end anchor this request resumes at held
    (`ResidentPrefix.end_hold`), so the levels after it go first: live 2026-10-18 01:58 the gate
    gave back the end a worker's call had just been carried into slot 0 with, and the call read
    7,953 tokens from 0. The hold lasts until the prefill's `begin` (engine/cache.py), so a side
    slot's gate cannot drop that end in between. When the gate refuses with the end held (its
    bytes were what the workspace lacked), or with a waited-for park kept (`_shed_park`), it runs
    once more without either: the request then reads its answer again, or the parked one does
    later, rather than a 503 at every retry; only a refusal of the last pass counts. The hold is
    set or cleared at every gate, and the handler clears it in its `finally` (a request that
    passes the gate and prefills nothing: A1's give-up, a prepared summary, a response-cache hit,
    a client that left), so a hold never outlives its request."""
    hold = _end_held(res, keys, n, mm_ok)
    if res is not None:
        res.end_hold = hold                  # never a hold another request left behind
    _PARK_LET_GO.kept = False
    try:
        mem.gate(n, count=False)             # a refusal counts once, at the outcome
    except cache.MemoryRefused as exc:
        if not hold and not _PARK_LET_GO.kept:
            mem.refused(exc)
            raise
        if hold:
            res.end_hold = False
        _PARK_LET_GO.on = True               # and a waited-for park may go unwritten
        try:
            mem.gate(n)
        finally:
            _PARK_LET_GO.on = False


def _compaction_plan(body: dict, conv_id: str | None, kind: str | None, n_client: int,
                     row: dict | None) -> dict | None:
    """A compaction request served as the continuation of the conversation the engine
    holds for its session (server/compaction.py), or None to serve it as the client sent it. Under
    the engine lock: the record is the rows in place or an entry on the NVMe."""
    cfg, prep = STATE.get("compaction"), STATE.get("prepared")
    if cfg is None or prep is None:
        return None
    parts = compaction_mod.split(body)
    why, out = None, None
    if parts is None:
        why = "shape"
    else:
        prep.learn(kind, parts)
        if cfg["mode"] != "continue":
            why = "off"
    if why is None:
        r = compaction_mod.find_record(STATE.get("resident"), STATE.get("spill"), conv_id)
        tok = STATE["tok"]
        if r is None:
            why = "no_record"
        elif not compaction_mod.guard_ok(parts, compaction_mod.record_text(
                tok, r.tokens, int(STATE["engine"].cfg.vocab_size))):
            why = "guard"
        else:
            tail = compaction_mod.tail_ids(tok, parts["system"], parts["instructions"])
            ids = r.tokens + tail
            if len(ids) + int(cfg["max_tokens"]) // 4 >= int(STATE["max_len"]):
                why = "too_long"
            else:
                k = compaction_mod.key(parts["system"], parts["instructions"])
                ready = prep.take(conv_id, r, k)
                out = {"ids": ids, "record": r, "key": k, "prepared": ready, "tail": len(tail),
                       "client_n": int(n_client)}
    st = prep.stats
    if out is None:
        st["client" if why in ("shape", "off") else why] += 1
    else:
        st["continued"] += 1
    info = ({"served": "client", "why": why} if out is None else
            {"served": "prepared" if out["prepared"] is not None else "continuation",
             "record": out["record"].where, "tokens": out["record"].at, "tail": out["tail"],
             "client_prompt": int(n_client), "images": out["record"].mm})
    if row is not None:
        row.setdefault("extra", {})["compaction"] = info
    print(f"[compaction] {conv_id and conv_id[:16]}: " + " ".join(
        f"{k}={v}" for k, v in info.items()), flush=True)
    return out


# The summary prepared in the background takes the engine last, behind every waiter
RANK_PREP = 3
#: how often the preparer looks whether the engine is idle
PREPARE_EVERY_S = 2.0


class _Preempted(Exception):
    """A request waits for the engine: the summary being prepared stops here."""


def _waiting() -> bool:
    with QUEUE:
        return bool(INFLIGHT["waiting"] or WAITERS) or bool(STATE.get("draining"))


def _box_lock():
    """The box lock (`~/.qwen38-box.flock`, or BOX_LOCK) taken without waiting, for the length of
    a summary written ahead: another program's GPU work beside the idle engine (its own `flock -n`)
    waits for it, and a hold's `flock -w` too. None when someone holds it."""
    import fcntl
    path = os.path.expanduser(os.environ.get("BOX_LOCK") or "~/.qwen38-box.flock")
    try:
        fd = os.open(path, os.O_RDONLY | os.O_CREAT | os.O_CLOEXEC, 0o644)
    except OSError:
        return -1                              # no lock file to take: nobody else coordinates
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return fd
    except OSError:
        os.close(fd)
        return None


def _box_waiters(fd) -> int:
    """How many processes wait in `flock` for the box lock this process holds as `fd` (Linux
    /proc/locks, its "->" lines for the file's device and inode: a hold's `flock -w`, another
    program's blocking take). 0 where /proc/locks is not there."""
    if fd is None or fd < 0:
        return 0
    try:
        st = os.fstat(fd)
        me = f"{os.major(st.st_dev):02x}:{os.minor(st.st_dev):02x}:{st.st_ino} "
        with open("/proc/locks") as f:
            return sum(1 for ln in f if "->" in ln and me in ln)
    except OSError:
        return 0


#: how often the preparer looks for the lock's waiters and for other GPU programs (s)
YIELD_LOCK_S, YIELD_GPU_S = 0.5, 5.0


def _gpu_others() -> list[int]:
    """The PIDs of programs on the GPU other than this server and its children (nvidia-smi's
    compute apps); [] when nvidia-smi is not there or does not answer in 3 s."""
    try:
        r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=3)
    except (OSError, subprocess.SubprocessError):
        return []
    me, out = os.getpid(), []
    for ln in r.stdout.split():
        try:
            pid = int(ln.strip().strip(","))
        except ValueError:
            continue
        p, mine = pid, False
        for _ in range(8):                     # its parents, up to this server
            if p == me:
                mine = True
                break
            try:
                with open(f"/proc/{p}/stat") as f:
                    p = int(f.read().rsplit(")", 1)[1].split()[1])
            except (OSError, ValueError, IndexError):
                break
            if p <= 1:
                break
        if not mine:
            out.append(pid)
    return out


class _Yield:
    """The summary written ahead gives the box back. It stops (like a request that
    waits for the engine) when another process waits in `flock` for the box lock it holds, or when
    another program runs on the GPU; both are looked at on a clock, not every token."""

    def __init__(self, box):
        self.box, self.t_lock, self.t_gpu, self.why = box, 0.0, 0.0, None

    def __call__(self) -> bool:
        now = time.monotonic()
        if now - self.t_lock >= YIELD_LOCK_S:
            self.t_lock = now
            if _box_waiters(self.box):
                self.why = "box_waiter"
                return True
        if now - self.t_gpu >= YIELD_GPU_S:
            self.t_gpu = now
            if _gpu_others():
                self.why = "gpu"
                return True
        return False


def _box_unlock(fd) -> None:
    if fd is not None and fd >= 0:
        import fcntl
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def prepare_once() -> str:
    """One look of the preparer (server/compaction.py): when the engine has been idle for
    `idle_s`, no hold or box lock is on, and the resident conversation is an opencode session's of
    at least `min_tokens` whose summary at its end anchor is not written yet, write it now as the
    conversation's continuation, in scratch rows, and stop at the first request that waits.
    Returns what happened, for the log and the tests."""
    cfg, prep, res = STATE.get("compaction"), STATE.get("prepared"), STATE.get("resident")
    if cfg is None or prep is None or res is None or cfg["prepare"] != "on" \
            or STATE.get("draining"):
        return "off"
    with QUEUE:
        if INFLIGHT["waiting"] or INFLIGHT["running"] or WAITERS:
            return "busy"
    if time.monotonic() - float(STATE.get("last_end_mono") or 0.0) < float(cfg["idle_s"]):
        return "recent"
    session = res.owner
    r = compaction_mod.find_record(res, None, session)
    if r is None:
        return "nothing"
    kind = prep.kinds.get(session)
    if kind not in prep.instructions:
        return "client"
    if r.at < int(cfg["min_tokens"]):
        return "short"
    if prep.has(session, r):
        return "done"
    try:
        if decider_api.held():
            return "hold"
    except Exception:                                              # noqa: BLE001
        return "hold"
    system, instructions = prep.instructions[kind]
    tok = STATE["tok"]
    tail = compaction_mod.tail_ids(tok, system, instructions)
    cap = int(cfg["prepare_max_tokens"])
    if r.at + len(tail) + cap >= int(STATE["max_len"]):
        prep.stats["too_long"] += 1
        return "too_long"
    if _gpu_others():
        prep.stats["yielded"] += 1
        return "gpu"
    box = _box_lock()
    if box is None:
        return "hold"
    yielded = _Yield(box)
    me = {"rank": RANK_PREP, "t": time.monotonic(), "side": None}
    if not _take(me, None):
        _box_unlock(box)
        return "busy"
    t0 = time.perf_counter()
    saved = None
    ids: list[int] = []
    why = "failed"
    gen = None
    try:
        # under the lock: the conversation must still be the one looked at
        r2 = compaction_mod.find_record(res, None, session)
        if r2 is None or r2.at != r.at or res.owner != session or _waiting():
            why = "moved"
            return why
        saved = (list(res.tokens), res.valid, res.taint, res.owner, res.capturing)
        prompt = torch.tensor(r.tokens + tail, dtype=torch.long, device=STATE["device"])
        mem = STATE.get("mem")
        if mem is not None:
            memlimits.refresh(STATE)
            mem.gate(int(prompt.numel()))

        def check(done, total):
            if _waiting() or yielded():
                raise _Preempted()
        eos = eos_ids({})
        gen = generate_stream(prompt, cap, eos, None, session, Deadline(600.0),
                              on_prefill=check, compacting=True, pos_delta=r.delta,
                              end_mm=True, scratch=True)
        for t in gen:
            ids.append(int(t))
            if t in eos:
                break
            if _waiting() or yielded():
                raise _Preempted()
        if not ids or ids[-1] not in eos:
            why = "length"
            return why
        k = compaction_mod.key(system, instructions)
        info = {"ms": round((time.perf_counter() - t0) * 1e3), "tail": len(tail),
                "where": r.where, "images": r.mm}
        prep.put(session, r, k, ids, info)
        why = "prepared"
        return why
    except _Preempted:
        if yielded.why is not None:
            prep.stats["yielded"] += 1
            why = f"yielded:{yielded.why}"
        else:
            prep.stats["stopped"] += 1
            why = "stopped"
        return why
    except cache.MemoryRefused:
        why = "memory"
        return why
    except Exception:                                              # noqa: BLE001
        prep.stats["failed"] += 1
        logbuf.print_exc()
        return why
    finally:
        if gen is not None:
            try:
                gen.close()
            except Exception:                                      # noqa: BLE001
                pass
        if saved is not None:
            # the conversation's bookkeeping as it was: its rows below the end anchor were not
            # written, the end anchor's state is in its own buffers, no anchor was taken
            res.tokens, res.valid, res.taint, res.owner, res.capturing = saved
            res._cut(res.valid)
        if STATE.get("mem") is not None:
            try:
                STATE["mem"].done()
            except Exception:                                      # noqa: BLE001
                pass
        kv_trim_current()                  # SRV-307: the job's scratch rows go back to the pool
        with QUEUE:
            HOLDER["who"] = None
        LOCK.release()
        _box_unlock(box)
        if why not in ("moved",):
            print(f"[compaction] prepare {session and session[:16]}: {why}, {r.at} tokens, "
                  f"{len(ids)} out, {(time.perf_counter() - t0) * 1e3:.0f} ms", flush=True)


#: the engine's idle graph captures wait this long after the last request ended (s)
GRAPHS_IDLE_S = 1.0


def graphs_once() -> str:
    """One look for the graphs the engines' context classes lack (`idle_captures`: an engine's verify,
    commit and draft graphs of a class its rounds have not used yet), slot 0's and each side slot's,
    captured in the idle window one at a time under the engine lock (which holds the GPU turn as slot
    0), stopping at the first request that waits. The first drafted request in a class paid 3.4-5.1 s
    of captures (P0, 2026-10-12); the drafter now waits for them (`capture_cost`)."""
    if STATE.get("draining"):
        return "off"
    pool = STATE.get("pool")
    engines = [(STATE.get("engine"), STATE.get("drafter"))] + (
        [(s.values.get("engine"), s.values.get("drafter")) for s in pool.side()] if pool is not None else [])
    engines = [(e, d) for e, d in engines if getattr(e, "idle_captures", None) is not None]
    if not engines:
        return "off"
    with QUEUE:
        if INFLIGHT["waiting"] or INFLIGHT["running"] or WAITERS:
            return "busy"
    if time.monotonic() - float(STATE.get("last_end_mono") or 0.0) < GRAPHS_IDLE_S:
        return "recent"
    # a length where the drafter takes the plain step anyway (260,800 tokens) captures nothing
    engines = [(e, d) for e, d in engines
               if getattr(d, "wants_graphs", None) is None or d.wants_graphs(int(e.kv.length))]
    if not engines:
        return "plain"
    me = {"rank": RANK_PREP, "t": time.monotonic(), "side": None}
    if not _take(me, None):
        return "busy"
    t0, n, why = time.perf_counter(), 0, "nothing"
    try:
        for e, _ in engines:
            steps = e.idle_captures()
            if steps is None:
                continue
            why = "captured"
            for _ in steps:
                n += 1
                if _waiting():
                    why = "stopped"
                    return why
        return why
    except Exception:                                              # noqa: BLE001
        logbuf.print_exc()
        why = "failed"
        return why
    finally:
        with QUEUE:
            HOLDER["who"] = None
        LOCK.release()
        if n:
            print(f"[graphs] idle captures over {len(engines)} engine(s): {n} graphs, {why}, "
                  f"{(time.perf_counter() - t0) * 1e3:.0f} ms", flush=True)


def _prepare_loop() -> None:
    while True:
        time.sleep(PREPARE_EVERY_S)
        try:
            if graphs_once() in ("captured", "stopped"):
                continue                # the graphs had the idle moment
        except Exception:                                          # noqa: BLE001
            logbuf.print_exc()
        try:
            got = prepare_once()
        except Exception:                                          # noqa: BLE001
            logbuf.print_exc()
            got = "failed"
        if got in ("prepared", "stopped") or got.startswith("yielded"):
            continue                    # the summary had the idle moment; the writer waits
        try:
            got = memory_once()
        except Exception:                                          # noqa: BLE001
            logbuf.print_exc()
            got = "failed"
        _writer_look(got)


#: memory_once's outcomes that keep a due session from being written; each is printed when the
#: look turns to it (2026-10-05: inside a hold the box lock made every look "hold", silently)
WRITER_BLOCKED = ("hold", "gpu", "nothing", "moved", "failed")


def _writer_look(got: str) -> None:
    """The writer's last look, kept in its stats (the Notes tab, `/memory/notes/writes`), and
    printed when it turns to an outcome that blocks a due session."""
    mem, _ = _writer_ctx()
    if mem is None:
        return
    st = mem.writer.stats
    last = st.get("last_look")
    st["last_look"] = got
    if got != last:
        st["last_look_since_ms"] = int(time.time() * 1000)
        if got == "no_session":
            print("[memory] writer look: no session (the resident conversation has no memory "
                  "session: no session id, raw, or memory off in its profile)", flush=True)
        elif got in WRITER_BLOCKED:
            res = STATE.get("resident")
            print(f"[memory] writer look: {got} ({_record_why(res, getattr(res, 'owner', None))})",
                  flush=True)


# P-M2 (2026-10-05): the memory writer's hidden turn (server/conductor/memwriter.py), behind the
# summary written ahead
RANK_MEMORY = 3


def _writer_ctx():
    """(the conductor's memory, the writer's settings), or (None, None) without a conductor."""
    cond = STATE.get("conductor")
    if cond is None or getattr(cond, "memory", None) is None:
        return None, None
    from server.conductor import memwriter
    return cond.memory, memwriter.settings_of(cond.memory.cfg())


def memory_once() -> str:
    """One look of the memory writer: when the engine is idle and the resident conversation's
    session is due (a compaction or a GIVE_UP marked it, or it has been idle `idle_s` with
    `min_new_tokens` new tokens since its last pass), Laya's gate says yes, and no hold or box lock
    is on: Qwen writes the session's notes as the conversation's continuation, thinking off, in
    scratch rows, and stops at the first request that waits. Returns what happened."""
    mem, cfg = _writer_ctx()
    res = STATE.get("resident")
    if mem is None or cfg["mode"] == "off" or res is None or STATE.get("draining"):
        return "off"
    with QUEUE:
        if INFLIGHT["waiting"] or INFLIGHT["running"] or WAITERS:
            return "busy"
    idle = time.monotonic() - float(STATE.get("last_end_mono") or 0.0)
    conv = res.owner
    if mem.writer.state(conv) is None:
        # the resident conversation never came with a memory session (no session id, raw, or a
        # profile with memory off): nothing of it is the writer's (MEM2: the smoke's requests
        # printed "nothing" with the record's numbers)
        return "no_session"
    r = _writer_record(res, conv)
    if r is None:
        return "nothing"
    trigger = mem.writer.due(conv, r.at, idle, cfg)
    if trigger is None:
        return "not_due"
    if trigger == "compaction" and cfg["after_compaction"] == "off":
        mem.writer.mark(conv, "idle")
        return "not_due"
    try:
        if decider_api.held():
            return "hold"
    except Exception:                                              # noqa: BLE001
        return "hold"
    if cfg["gate"]:
        t0 = time.perf_counter()
        ans, how = mem.writer.gate(STATE["conductor"].router, conv)
        gate = {"answer": ans, "how": how, "ms": round((time.perf_counter() - t0) * 1e3, 1)}
        if ans == "no":
            mem.writer.skipped(conv, trigger, "gate_no", r.at, {"gate": gate})
            return "gate_no"
    else:
        gate = {"answer": None, "how": "off"}
    if _gpu_others():
        return "gpu"
    box = _box_lock()
    if box is None:
        return "hold"
    yielded = _Yield(box)
    me = {"rank": RANK_MEMORY, "t": time.monotonic(), "side": None}
    if not _take(me, None):
        _box_unlock(box)
        return "busy"
    try:
        r2 = _writer_record(res, conv)
        if r2 is None or r2.at != r.at or res.owner != conv or _waiting():
            return "moved"
        return _writer_turn(mem, cfg, conv, r, trigger, gate,
                            stop=lambda: _waiting() or yielded(), yielded=yielded)
    finally:
        if STATE.get("mem") is not None:
            try:
                STATE["mem"].done()
            except Exception:                                      # noqa: BLE001
                pass
        kv_trim_current()                  # SRV-307: the job's scratch rows go back to the pool
        with QUEUE:
            HOLDER["who"] = None
        LOCK.release()
        _box_unlock(box)


def _writer_record(res, conv):
    """What the writer continues: the session's last generation (its end anchor, as the summary
    written ahead), else the last prompt the rows in place hold (the end anchor was not kept for
    the last turn: on the 2026-10-05 measurement the rows held 4,485 tokens and the end sat at
    3,766), up to its last message's end. The last answer is missing from the second; everything
    before it is there. A prompt with images (key ids past the vocabulary) is not used."""
    r = compaction_mod.find_record(res, None, conv)
    if r is not None or res is None or not conv:
        return r
    if res.owner != conv or res._stash is not None or res.valid <= 0 or \
            len(res.tokens) < res.valid:
        return None
    toks = [int(t) for t in res.tokens[:res.valid]]
    vocab = int(getattr(getattr(STATE.get("engine"), "cfg", None), "vocab_size", 0) or 0)
    if vocab and any(t >= vocab or t < 0 for t in toks):
        return None
    # the prompt ends in its generation prompt (`<|im_start|>assistant\n<think>\n`): the record
    # stops before the last message's `<|im_end|>`, which the writer's tail writes (2026-10-05 on
    # the box: after the open assistant turn the writer answered 48 tokens and no block)
    at = _last_im_end(toks)
    if at is None or at <= 0:
        return None
    return compaction_mod.Record(toks, at, "resident_prompt")


def _last_im_end(toks: list[int]) -> int | None:
    """The position of the last `<|im_end|>` in `toks`, or None."""
    tok = STATE.get("tok")
    try:
        end = tok.convert_tokens_to_ids("<|im_end|>")
    except Exception:                                              # noqa: BLE001
        return None
    if not isinstance(end, int) or end < 0:
        return None
    for i in range(len(toks) - 1, -1, -1):
        if toks[i] == end:
            return i
    return None


def _record_why(res, conv) -> str:
    """Why `_writer_record` has nothing for `conv`, for the log (never a token)."""
    if res is None:
        return "no resident"
    end = getattr(res, "end", None)
    return (f"owner {'=' if res.owner == conv else '!='} conv, valid {res.valid}, "
            f"tokens {len(res.tokens)}, end {None if end is None else end.get('at')}, "
            f"stash {'yes' if res._stash is not None else 'no'}")


def _writer_turn(mem, cfg, conv, r, trigger, gate, stop=lambda: False, yielded=None,
                 deadline_s: float = 600.0) -> str:
    """The writer's hidden turn on the record `r`, under the engine lock (the caller holds it):
    the record plus the writer's user turn, thinking off, at most `max_tokens`, in scratch rows;
    the conversation's bookkeeping is put back after it as the preparer does. `stop()` ends it
    at the next chunk or block."""
    from server.conductor import memwriter
    tok = STATE["tok"]
    said = ""
    text, existing = mem.writer.instructions(conv, cfg)
    tail = compaction_mod.tail_ids(tok, "", text)
    # M3 (2026-10-05): the writer's own NONE. The answer may start with `prefill` (its tokens
    # follow the generation prompt and count as the answer's start), and `<tool_call>` may be
    # banned for the turn (a logit bias of -100: after a compaction Qwen answered a `bash` call)
    prefill = str(cfg.get("prefill") or "")
    pre_ids = list(tok(prefill, add_special_tokens=False).input_ids) if prefill else []
    ban = memwriter.ban_ids(tok) if cfg.get("ban_tool_call") else {}
    tail = tail + pre_ids
    cap = int(cfg["max_tokens"])
    info = {"gate": gate, "tail": len(tail), "record": r.at, "existing": len(existing),
            "ban": sorted(ban) if ban else ("unavailable" if cfg.get("ban_tool_call") else None),
            "prefill": len(pre_ids)}
    if r.at + len(tail) + cap >= int(STATE["max_len"]):
        mem.writer.skipped(conv, trigger, "too_long", r.at, info)
        return "too_long"
    res = STATE["resident"]
    saved = (list(res.tokens), res.valid, res.taint, res.owner, res.capturing)
    ids: list[int] = []
    gen = None
    t0 = time.perf_counter()
    why = "failed"
    try:
        prompt = torch.tensor(r.tokens + tail, dtype=torch.long, device=STATE["device"])
        if STATE.get("mem") is not None:
            memlimits.refresh(STATE)
            STATE["mem"].gate(int(prompt.numel()))

        def check(done, total):
            if stop():
                raise _Preempted()
        eos = eos_ids({})
        pen = None
        if ban:
            pen = PenaltyState(PenaltySpec(bias=ban), int(STATE["engine"].cfg.vocab_size),
                               STATE.get("device", "cuda"))
        gen = generate_stream(prompt, cap, eos, None, conv, Deadline(float(deadline_s)),
                              pen=pen, on_prefill=check, compacting=True, pos_delta=r.delta,
                              end_mm=True, scratch=True)
        t1 = None
        cut = None
        for t in gen:
            if t1 is None:
                t1 = time.perf_counter()
            ids.append(int(t))
            if t in eos:
                break
            # a tool call that is not a memory block ends the turn at its function's name (after
            # a compaction Qwen answered an empty or a `bash` call, 2026-10-05): it writes nothing
            cut = memwriter.foreign_call(prefill + tok.decode(ids, skip_special_tokens=False))
            if cut is not None:
                break
            if stop():
                raise _Preempted()
        out = prefill + tok.decode(ids, skip_special_tokens=True)
        info.update(ms=round((time.perf_counter() - t0) * 1e3),
                    prefill_ms=round(((t1 or time.perf_counter()) - t0) * 1e3),
                    tokens=len(ids), finish=("tool_call" if cut is not None else
                                             "stop" if ids and ids[-1] in eos else "length"))
        if cut is not None:
            info["cut"] = cut
        _writer_text(trigger, out)
        run = mem.writer.apply(conv, out, cfg, trigger, r.at, info)
        said = (f"{len(run['ops'])} op(s)" if run.get("ops") else
                "NONE" if run.get("answer") == "none" else
                f"NONE ({run.get('answer')}{f': {cut}, cut' if cut else ''})" if run.get("none")
                else f"no block ({run.get('answer')})")
        why = "wrote"
        return why
    except _Preempted:
        why = f"yielded:{yielded.why}" if yielded is not None and yielded.why else "stopped"
        info.update(ms=round((time.perf_counter() - t0) * 1e3), tokens=len(ids))
        mem.writer.skipped(conv, trigger, why, r.at, info)
        return why
    except cache.MemoryRefused:
        why = "memory"
        mem.writer.skipped(conv, trigger, why, r.at, info)
        return why
    except Exception:                                              # noqa: BLE001
        logbuf.print_exc()
        mem.writer.skipped(conv, trigger, "failed", r.at, info)
        return why
    finally:
        if gen is not None:
            try:
                gen.close()
            except Exception:                                      # noqa: BLE001
                pass
        res.tokens, res.valid, res.taint, res.owner, res.capturing = saved
        res._cut(res.valid)
        print(f"[memory] writer {conv and conv[:16]}: {trigger} {why}"
              f"{f': {said}' if why == 'wrote' else ''}, {r.at} tokens, "
              f"{len(ids)} out, {(time.perf_counter() - t0) * 1e3:.0f} ms", flush=True)


def _writer_text(trigger: str, out: str) -> None:
    """A measurement's aid: with QWEN38_MEMORY_WRITER_TEXT naming a file (a test engine only,
    never the service: the text is the session's), each writer turn's text is appended to it."""
    path = _S.get("MEMORY_WRITER_TEXT")
    if not path:
        return
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(fd, "a") as f:
            f.write(f"=== {time.strftime('%H:%M:%S')} {trigger}\n{out}\n")
    except OSError:
        pass


def _writer_after_compaction(plan, conv_id) -> None:
    """`after_compaction: turn`: the writer right after a compaction's summary, before the engine
    goes to the next request (the caller still holds the lock), at most `turn_deadline_s`."""
    mem, cfg = _writer_ctx()
    if mem is None or plan is None or not getattr(plan, "compaction", False) or not conv_id:
        return
    mem.writer.mark(conv_id, "compaction")
    if cfg["mode"] == "off" or cfg["after_compaction"] != "turn":
        return
    res = STATE.get("resident")
    r = _writer_record(res, conv_id)
    if r is None:
        # the mark stays: the idle look writes the session once its record is in place
        print(f"[memory] writer {conv_id[:16]}: compaction, no record now "
              f"({_record_why(res, conv_id)}); left to the idle look", flush=True)
        return
    # no gate here: the caller holds the engine lock, and the in-process decider asks only in the
    # idle window (a call from under the lock waits out its deadline: `decider_deadline_missed`
    # on the box, 2026-10-05)
    gate = {"answer": None, "how": "in_turn"}
    try:
        _writer_turn(mem, cfg, conv_id, r, "compaction", gate,
                     deadline_s=float(cfg["turn_deadline_s"]))
    except Exception:                                              # noqa: BLE001
        logbuf.print_exc()


# SR1: the shadow reader reads in the idle window, behind every waiter and the summary written
# ahead; a burst holds the engine at most shadow_mod.BURST_MS and stops before the next grid chunk
# as soon as a request waits (a chunk of the 0.8B: tools/shadow_measure.py)
RANK_SHADOW = 4
SHADOW_EVERY_S = 0.2


def _shadow_begin(plan, prompt_ids: list[int], n_choices: int):
    """(session key, service, settings) when this turn feeds the session's shadow (a TandemLLM
    request of a profile with `shadow.mode: on`, one choice, not a side request), else
    (None, service or None, settings)."""
    svc = STATE.get("shadow")
    cfg = shadow_mod.settings_of(plan.res.doc if plan is not None and plan.active else None)
    if (svc is None or cfg["mode"] != "on" or plan is None or not plan.active
            or plan.session is None or plan.side is not None or n_choices != 1):
        return None, svc, cfg
    key = plan.session.key
    try:
        ask = svc.router_ask(plan, prompt_ids, STATE["tok"], cfg)
        info = svc.begin_turn(key, prompt_ids, ask=ask)
        if plan.row is not None:
            plan.row.setdefault("extra", {})["shadow"] = {"behind": info["behind"],
                                                          "router_ask": ask is not None}
    except Exception:                                              # noqa: BLE001
        logbuf.print_exc()
        return None, svc, cfg
    return key, svc, cfg


def shadow_once() -> str:
    """One look of the shadow loop: when nothing waits or runs and no hold or box lock is on,
    read what the sessions' shadows are missing, for at most one burst."""
    svc = STATE.get("shadow")
    if svc is None or STATE.get("draining"):
        return "off"
    if not svc.has_work():
        return "nothing"
    with QUEUE:
        if INFLIGHT["waiting"] or INFLIGHT["running"] or WAITERS:
            return "busy"
    try:
        if decider_api.held():
            return "hold"
    except Exception:                                              # noqa: BLE001
        return "hold"
    if svc.ready() is not None:
        return "loading"
    box = _box_lock()
    if box is None:
        return "hold"
    me = {"rank": RANK_SHADOW, "t": time.monotonic(), "side": None}
    if not _take(me, None):
        _box_unlock(box)
        return "busy"
    try:
        out = svc.work(deadline=svc.clock() + shadow_mod.BURST_MS / 1000.0, stop=_waiting)
    finally:
        with QUEUE:
            HOLDER["who"] = None
        LOCK.release()
        _box_unlock(box)
    if out.get("rows") or out.get("asks"):
        print(f"[shadow] read {out['rows']} rows, {out['asks']} questions, {out['ms']:.0f} ms"
              f"{', stopped (a request waits or the burst ended)' if out.get('stopped') else ''}", flush=True)
    return "stopped" if out.get("stopped") else "read"


def _shadow_loop() -> None:
    last = 0.0
    while True:
        time.sleep(SHADOW_EVERY_S)
        try:
            shadow_once()
        except Exception:                                          # noqa: BLE001
            logbuf.print_exc()
        if time.monotonic() - last >= SHADOW_PRELOAD_S:
            last = time.monotonic()
            try:
                shadow_preload_once()
            except Exception:                                      # noqa: BLE001
                logbuf.print_exc()


# The shadow reader's first load, ahead of its first turn. It loaded with the first
# turn of a profile with shadow.mode on: after a restart that turn's class question waited out its
# 8 s behind the load's copies and gave up, and the load landed on full caches (NP1's open issue 3,
# 2026-10-04 21:41 and 2026-10-05 06:18). Now, as the narrator's keeper does, the first load runs in
# an idle window soon after the start, when no other model is loading and some profile has the
# shadow on; once it has loaded, the first turn's start and the release's reload are as before.
SHADOW_PRELOAD_S = 20.0
# VIS-153: a failed shadow load (its weights not on the box yet, a load error) is tried again in an
# idle window once this long has passed since it failed; before, `failed` stayed until a restart
SHADOW_RETRY_S = 300.0


def _shadow_profiles_on() -> bool:
    """Does some profile have `shadow.mode: on`?"""
    cond = STATE.get("conductor")
    if cond is None:
        return False
    for name in cond.store.names():
        try:
            _, doc = cond.store.peek("profiles", name)
        except Exception:                                          # noqa: BLE001
            continue
        if shadow_mod.settings_of(doc)["mode"] == "on":
            return True
    return False


def shadow_preload_once() -> str | None:
    """One look: what the shadow's first load did ("loading", "refused"), else None."""
    svc = STATE.get("shadow")
    if svc is None or STATE.get("draining"):
        return None
    slot = svc.slot
    narr = STATE.get("narrator")
    # SRV-534: with the narrator role off (the default) its keeper never loads a shared slot
    if narr is not None and slot is getattr(narr, "slot", None) and \
            narr.settings()["mode"] != "off":
        return None                             # the narrator's slot: its keeper loads it
    since = slot.failed_for() if hasattr(slot, "failed_for") else None
    retry = since is not None and since >= SHADOW_RETRY_S
    if not retry and (getattr(slot, "loads", 0) or slot.state != "not_loaded"):
        return None                             # loaded once already, or a load in its course
    if getattr(slot, "memory_hold", False):
        return None                             # out for memory: the role governor reloads it
    with QUEUE:
        busy = bool(INFLIGHT["waiting"] or INFLIGHT["running"] or WAITERS) or LOCK.locked()
    if busy or decider_api._other_loading(STATE, "shadow") or \
            (narr is not None and narr.slot.state == "loading"):
        return None
    if not _shadow_profiles_on():
        return None
    if retry and slot.reset_failed():
        print(f"[shadow] {slot.model.name}: the load failed {since:.0f} s ago ({slot.error}); "
              f"trying again", flush=True)
        STATE["shadow_preload"] = None
    why = slot.ensure()
    got = "loading" if why == "loading" else "refused" if why == "refused" else why
    if got != STATE.get("shadow_preload"):
        STATE["shadow_preload"] = got
        reason = (slot.refusal or {}).get("reason") if got == "refused" else None
        print(f"[shadow] {slot.model.name}: first load ahead of the first turn: {got}"
              f"{f' ({reason})' if reason else ''}", flush=True)
    return got


# v3: the narrator's model stays loaded. ops/start.sh asks for its load at the start, after the
# page cache is dropped (`POST /v1/conductor/narrator/load`, the residency planner's check with
# service_start); from then on this keeper loads it again whenever it is not loaded -- the shadow
# reader released a shared slot's pool, three failed calls unloaded it, a refusal passed -- in an
# idle window (no request running or waiting), so the copy never waits behind a long turn (the
# 12-minute load of 2026-10-04 started during one and copied its batches between its rounds).
NARRATOR_KEEP_S = 20.0


def _narrator_keeper() -> None:
    while True:
        time.sleep(NARRATOR_KEEP_S)
        try:
            narrator_once()
        except Exception:                                          # noqa: BLE001
            logbuf.print_exc()


def narrator_once() -> str | None:
    """One look: the load it started, or None."""
    n = STATE.get("narrator")
    if n is None or not STATE.get("narrator_keep") or STATE.get("draining"):
        return None
    slot = n.slot
    if slot.state == "loaded" or slot.state in ("loading", "unloading"):
        return None
    if getattr(slot, "memory_hold", False):
        return None                             # out for memory: the role governor reloads it
    if n.settings()["mode"] == "off":
        return None
    with QUEUE:
        busy = INFLIGHT["waiting"] > 0 or LOCK.locked()
    if busy or decider_api._other_loading(STATE, "narrator"):
        return None
    if slot.state == "failed":
        slot.state = "not_loaded"              # a failed model is tried again, once a window
    d = slot.load()
    why = "loading" if d.get("allowed") else f"refused ({d.get('reason')})"
    print(f"[narrator] keeper: {slot.model.name} was not loaded; {why}", flush=True)
    return why


def shadow_drain() -> None:
    """At a planned stop, after the work in flight: the shadows to the NVMe."""
    svc = STATE.get("shadow")
    if svc is None:
        return
    try:
        with LOCK:
            info = svc.drain()
        print(f"[shadow] drain: {info}", flush=True)
    except Exception:                                              # noqa: BLE001
        logbuf.print_exc()


def _compacting(plan, body: dict, client_kind: str | None = None) -> bool:
    """Is this the session's compaction? opencode's request has no tools, no system text and the
    history as one user message; the conductor's router classes it `summarize` on that shape
    (structurally), and without a conductor the shape alone says it."""
    row = plan.row if plan is not None else None
    if row is not None and row.get("source") == "structural" and row.get("class_id") == "summarize":
        return True
    try:
        from server.conductor import signals
        return bool(signals.shape(body, client_kind=client_kind).compaction)
    except Exception:                                              # noqa: BLE001
        return False


def _memory_event(ev: dict) -> None:
    """A `Budget` event: the log for all, the residency journal for what an operator acts on."""
    kind = ev.get("kind")
    if kind in ("shed", "refused", "estimate_raised"):
        print(f"[memory] {kind} " + " ".join(f"{k}={v}" for k, v in ev.items()
                                             if k not in ("kind", "ts")), flush=True)
        cond = STATE.get("conductor")
        if cond is not None:
            try:
                from server.conductor import residency
                residency.Journal(cond.root).write(dict(ev, kind=f"memory_{kind}", role="engine",
                                                        source="engine"))
            except Exception:                                      # noqa: BLE001
                pass
    elif kind == "empty" and STATE.get("verbose"):
        print(f"[memory] reserve back to the driver: {ev.get('freed', 0) / 2**30:.2f} GiB in "
              f"{ev.get('ms')} ms ({ev.get('why')})", flush=True)


# ------------------------------------------------------------------ SRV-307: the shared KV pool
def _kv_pool_event(kind: str, kw: dict) -> None:
    """A pool event (engine/kvpool.py): the log, and the residency journal for a refusal."""
    if kind in ("reclaim", "refused", "reclaim_error"):
        print(f"[kv-pool] {kind} " + " ".join(f"{k}={v}" for k, v in kw.items()), flush=True)


def _kv_keep_rows(eng, res, slot=None) -> int:
    """The rows a slot still needs after a generation: its KV length, the rows its resident
    prefix holds valid (a guest's stash puts rows back below it, the end anchor and the anchors
    sit inside it) and, on a side slot, the end its next turn may carry into slot 0."""
    keep = int(getattr(eng.kv, "length", 0) or 0)
    if res is not None:
        keep = max(keep, int(getattr(res, "valid", 0) or 0))
    if slot is not None and slot.carry is not None and slot.carry.get("gen") == slot.gen:
        keep = max(keep, int(slot.carry.get("at") or 0))
    return keep


def kv_trim_current() -> int:
    """At a generation's end, in its own thread: the pages past what the slot keeps go back."""
    if kvpool.POOL is None:
        return 0
    eng = STATE.get("engine")
    pages = getattr(getattr(eng, "kv", None), "pages", None)
    if pages is None:
        return 0
    try:
        return pages.trim(_kv_keep_rows(eng, STATE.get("resident"), slots_mod.current()))
    except Exception:                                              # noqa: BLE001
        logbuf.print_exc()
        return 0


def _kv_reclaim_caches(want: int, owner) -> int:
    """Pool reclaim, level 1: the memory policy's levels, cheapest first (engine/cache.py `Budget`
    `LEVELS`), only what nothing is using, and the allocator's free blocks back to the driver."""
    mem = dict.get(STATE, "mem")
    if mem is None:
        return 0
    got = 0
    for lv in mem.LEVELS:
        got += int(mem.shed(lv, "kv_pool", safe=True) or 0)
        got += int(mem.empty("kv_pool") or 0)
        if got >= want:
            break
    return got


def _kv_reclaim_kept_park(want: int, owner) -> int:
    """ENG-371, the pool's last reclaim, after every other: slot 0's park that `_shed_park` keeps
    for a parent or a worker (it is waited for and could not be written where the earlier
    levels asked), given back unwritten, so the page gate inside a forward or a restore never
    refuses for it, and the sandbox's release (`_memory_release`) always gets it. Nothing while
    the park is in use (`_park_busy`, read without a lock: a stale read costs a re-read, never
    wrong rows, as the busy park is not touched)."""
    res = dict.get(STATE, "resident")
    if res is None or not res.park_rows or res._park_busy:
        return 0
    _PARK_LET_GO.on = True
    try:
        n = int(res.shed_park(True) or 0)
    finally:
        _PARK_LET_GO.on = False
    if n:
        print(f"[memory] the park went unwritten for the KV pool ({owner}): {_gib(n)} GiB",
              flush=True)
    return n


def _add_kv_reclaimers(pool) -> None:
    """The KV pool's reclaim order, cheapest first; ENG-371's kept park last."""
    pool.add_reclaimer("caches", _kv_reclaim_caches)
    pool.add_reclaimer("roles", _kv_reclaim_roles)
    pool.add_reclaimer("idle_side", _kv_reclaim_side)
    pool.add_reclaimer("idle_slot0", _kv_reclaim_slot0)
    pool.add_reclaimer("kept_park", _kv_reclaim_kept_park)


def _kv_reclaim_roles(want: int, owner) -> int:
    """Pool reclaim, level 2 (server/rolemem.py): the small models, shadow reader first, then the
    narrator (then Laya, only with QSE_ROLE_UNLOAD_LAYA=on), before any conversation is parked or
    a request waits. They load again once memory has stayed back for a while. The setup guide's
    assistant goes before all of them (server/setup/wren.py), and nothing loads it again."""
    got = 0
    wren = dict.get(STATE, "wren")
    if wren is not None:
        got = int(wren.reclaim(want) or 0)
    rm = dict.get(STATE, "role_mem")
    if rm is None or got >= want:
        return got
    return got + int(rm.reclaim(want - got, owner, "kv_pool") or 0)


def _kv_reclaim_side(want: int, owner) -> int:
    """Pool reclaim, level 3: the conversations idle side slots keep, oldest first. The slot's
    next turn re-reads its prompt (or comes back from the disk tier); nothing is lost but time."""
    pool = dict.get(STATE, "pool")
    if pool is None:
        return 0
    got = 0
    idle = [s for s in pool.side() if f"slot{s.index}" != owner and not s.busy]
    idle.sort(key=lambda s: float((s.carry or {}).get("t") or 0.0))
    for s in idle:
        if got >= want:
            break
        eng = (s.values or {}).get("engine")
        pages = getattr(getattr(eng, "kv", None), "pages", None)
        if pages is None or pages.committed_bytes() == 0 or not s.lock.acquire(blocking=False):
            continue
        try:
            res = (s.values or {}).get("resident")
            if res is not None:
                res.clear()
            s.carry = None
            eng.kv.length = 0
            got += pages.trim(0)
        finally:
            s.lock.release()
    return got


def _kv_reclaim_slot0(want: int, owner) -> int:
    """Pool reclaim, level 4: slot 0's idle conversation, to the NVMe first when the spill is on
    (engine/spill.py, the tier a replaced conversation goes to), then its pages."""
    if owner == "slot0":
        return 0
    eng = dict.get(STATE, "engine")
    pages = getattr(getattr(eng, "kv", None), "pages", None)
    if pages is None or pages.committed_bytes() == 0 or not LOCK._lk.acquire(False):
        return 0
    try:
        res = dict.get(STATE, "resident")
        if res is not None:
            if res._stash is not None:
                res.settle_guest(eng)
            if res.spiller is not None and res.valid >= getattr(res.spiller, "min_tokens", 1 << 30):
                res._spill(eng, dict.get(STATE, "drafter"), "kv_pool")
            res.clear()
        eng.kv.length = 0
        return pages.trim(0)
    finally:
        LOCK._lk.release()


def _kv_reclaim_idle(d: dict, owner) -> int:
    """SRV-307, before a request waits for memory: the pages idle conversations keep (idle side
    slots, then slot 0's idle one, to the NVMe first), as much as the admission is short. Not the
    caches: the admission already counts their growth, and shedding them leaves it unchanged."""
    pool = kvpool.POOL
    if pool is None or d.get("after") is None:
        return 0
    want = int(d.get("floor", 0)) - int(d["after"])
    rm = dict.get(STATE, "role_mem")
    if rm is not None and want > 0:
        rm.note_pressure("admission")
    got = 0
    for name, fn in list(pool.reclaimers):
        if name == "caches" or got >= want:
            continue
        try:
            n = int(fn(want - got, owner) or 0)
        except Exception:                                          # noqa: BLE001
            logbuf.print_exc()
            n = 0
        if n:
            got += n
            pool.stats["reclaims"] += 1
            pool.stats["reclaimed_bytes"] += n
            print(f"[kv-pool] reclaim for admission level={name} bytes={n} owner={owner}",
                  flush=True)
    return got


def _memory_release(want: int, why: str = "sandbox") -> dict:
    """SRV-483, `POST /v1/cache/release`: room on the unified memory for something beside the
    engine, a chat's sandbox (chatd/sandbox.py), which has priority over what the engine holds
    idle (ruling of 2026-10-15). The KV pool's own order, cheapest first: the caches and the
    allocator's free blocks (under the engine lock, only while it is idle), then the small models,
    the idle side slots' conversations, slot 0's idle conversation (to the NVMe first when the
    spill is on). Nothing a running request uses goes, and nothing loads back for `hold_s`
    (server/rolemem.py `note_pressure`). `want` is a target: the answer says what was given back
    and what MemAvailable is now."""
    before = memsplit.mem_available()
    got, levels = 0, {}
    if want > 0 and _lock_when_idle():
        try:
            n = _kv_reclaim_caches(want, "release")
            if n:
                got, levels["caches"] = got + n, n
            mem = dict.get(STATE, "mem")
            if mem is not None and got < want:
                # ENG-371: the park a safe shed keeps for a parent or a worker; idle and under the
                # lock its write is safe: to the NVMe, then given back (the sandbox wins)
                p = int(mem.shed("park", "release", safe=False) or 0)
                if p:
                    got, levels["park"] = got + p, p
        finally:
            LOCK.release()
    pool = kvpool.POOL
    rm = dict.get(STATE, "role_mem")
    if rm is not None and got < want:
        rm.note_pressure(why)
    for name, fn in list(pool.reclaimers) if pool is not None else []:
        if name == "caches" or got >= want:
            continue
        try:
            n = int(fn(want - got, "release") or 0)
        except Exception:                                          # noqa: BLE001
            logbuf.print_exc()
            n = 0
        if n:
            got, levels[name] = got + n, n
            pool.stats["reclaims"] += 1
            pool.stats["reclaimed_bytes"] += n
    after = memsplit.mem_available()
    print(f"[memory] release for {why}: asked {_gib(want)} GiB, gave back {_gib(got)} GiB "
          f"({', '.join(levels) or 'nothing idle'}), MemAvailable {_gib(before)} -> "
          f"{_gib(after)} GiB", flush=True)
    return {"asked_bytes": int(want), "released_bytes": int(got), "levels": levels,
            "mem_available_bytes": after}


#: SRV-309: while another slot decodes, a prefill's layer-major window takes at most this many
#: rows, so the GPU turn is offered to the decoders at least that often (0: no cap, the default).
#: Bit-identical (tests/gpu/test_kvpool_identity_gpu.py --window-cap), but it did not help: on
#: 2026-10-10 three decoding chats still wrote 0.6-1 tok/s while an 87k prompt prefilled with the
#: cap at 2,048 (0.9-1.3 without it). The stall was the rounds' barrier (SRV-311, slots.py
#: `barrier`); with that fixed the cap still added nothing measurable (11.2-12.4 vs 10.3-11.4 tok/s)
PAR_WINDOW_ROWS = int(os.environ.get("QSE_PARALLEL_WINDOW_ROWS", "0") or 0)


#: SRV-321: between two layers of a layer-major window the GPU turn is offered to a waiting slot,
#: as between two chunks ("off" offers it after the window only, kept for the A/B)
PAR_LAYER_YIELD = (os.environ.get("QSE_PARALLEL_LAYER_YIELD") or "on").strip().lower() != "off"


def _par_window_yield() -> None:
    """SRV-321: a prefill's window between two layers: the GPU offered to a waiting slot (the
    decoders run their rounds; a no-op when nobody waits). Without it the turn was offered only
    after the window, and an 8,192-row EXL3 window held it ~10 s: three decoding chats wrote 0
    tokens for 42 of the 147 seconds a 37k-token prompt prefilled beside them (2026-10-10)."""
    pool = dict.get(STATE, "pool")
    if pool is not None and pool.turn is not None and pool.turn.waiting():
        pool.turn.pause(_par_index())


#: SRV-321: whether `_window_layer` offers the GPU turn between two layers (side slots with a turn,
#: `QSE_PARALLEL_LAYER_YIELD` not off)
WINDOW_TURN = [False]


def _window_layer() -> None:
    """`engine.model.WINDOW_YIELD`, between two layers of a layer-major window: the live view's
    layer mark (SRV-591, engine/cache.py `layer_mark`), then, with side slots, the GPU offered to
    a waiting slot (`_par_window_yield`). Neither touches the window's rows."""
    cache.layer_mark()
    if WINDOW_TURN[0]:
        _par_window_yield()


def _par_window_cap() -> int | None:
    pool = dict.get(STATE, "pool")
    if pool is None:
        return None
    me = _par_index()
    return PAR_WINDOW_ROWS if any(i != me for i in pool.decoding()) else None


def _roles_idle() -> bool:
    """No request waits: none in the queue, none waiting for memory."""
    if INFLIGHT.get("waiting"):
        return False
    pool = dict.get(STATE, "pool")
    adm = getattr(pool, "admission", None) if pool is not None else None
    return not (adm is not None and adm.waiting)


def _roles_headroom() -> int | None:
    """What a new request could still take: MemAvailable less the caches' growth left and what
    the requests in flight reserved (the admission's own sum); MemAvailable without it."""
    pool = dict.get(STATE, "pool")
    adm = getattr(pool, "admission", None) if pool is not None else None
    if adm is not None:
        try:
            _, d = adm.check(0, count=False)
            if d.get("after") is not None:
                return int(d["after"])
        except Exception:                                          # noqa: BLE001
            pass
    return memsplit.counted_available()


def _roles_start(a) -> None:
    def journal(rec):
        cond = STATE.get("conductor")
        if cond is not None:
            from server.conductor import residency
            residency.Journal(cond.root).write(rec)

    def empty():
        mem = dict.get(STATE, "mem")
        if mem is not None:
            mem.empty("role_unload")
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()

    floor = float(getattr(a, "parallel_floor_gb", 15.0) or 15.0)
    rm = rolemem.build(STATE, floor_gib=floor, meminfo=memsplit.counted_available,
                       headroom=_roles_headroom, idle=_roles_idle, empty=empty, journal=journal,
                       log=lambda m: print(m, flush=True))
    STATE["role_mem"] = rm
    if rm is None:
        return
    print(f"[roles] dynamic {'on' if rm.enabled else 'off'}: unload order "
          f"{' -> '.join(r.name for r in rm.roles)} (after the caches, before idle conversations);"
          f" reload at floor {floor:g} + claim + margin {rm.margin / (1 << 30):g} GiB held "
          f"{rm.hold_s:g} s with nothing waiting", flush=True)
    if not a.fake_engine or os.environ.get("QSE_ROLE_TICK_FAKE") == "1":
        rm.start()


def _safe(fn):
    """`fn()`, or `{"error": ...}` when it raises: a report never breaks `/health`."""
    try:
        return fn()
    except Exception as exc:                                       # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"[:200]}


def roles_report() -> dict:
    """`/health` `roles`: each small model's state (loaded, unloaded_for_memory, reloading, or its
    slot's own), the governor's policy and its last events."""
    rm = dict.get(STATE, "role_mem")
    if rm is not None:
        return rm.report()
    out = {"dynamic": False, "roles": {}}
    nar, svc, reg = (dict.get(STATE, k) for k in ("narrator", "shadow", "deciders"))
    if svc is not None and getattr(svc, "slot", None) is not getattr(nar, "slot", None):
        out["roles"]["shadow"] = {"models": [svc.slot.model.name], "state": svc.slot.state,
                                  **rolemem.slot_detail(svc.slot)}
    if nar is not None:
        out["roles"]["narrator"] = {"models": [nar.slot.model.name], "state": nar.slot.state,
                                    **rolemem.slot_detail(nar.slot)}
    if reg is not None:
        out["roles"]["laya"] = {"models": reg.loaded(), "state": "loaded" if reg.ready()
                                else "not_loaded"}
        refused = reg.refused() if hasattr(reg, "refused") else None
        if refused is not None:
            # 0.11.3: why Laya is not loaded (the planner's numbers), until a model loads
            out["roles"]["laya"]["refused"] = refused
    return out


def _kv_after_load(a) -> None:
    """After slot 0's engine and drafters exist: the memory policy no longer holds slot 0's
    window (its pages are the pool's now, outside the allocator), and the reclaimers in order."""
    pool = kvpool.POOL
    eng = dict.get(STATE, "engine")
    pages = getattr(getattr(eng, "kv", None), "pages", None)
    if pool is None or pages is None:
        _window_above_limits(a, eng)
        return
    static = sum(b.nbytes for b in pages.bufs)
    mem = dict.get(STATE, "mem")
    if mem is not None and mem.on:
        mem.fixed -= static
        mem.ceiling -= static
        mem.peak -= static
        print(f"[kv-pool] memory policy: ceiling and peak -{_gib(static)} GiB (slot 0's window is "
              f"the pool's now): {_gib(mem.ceiling)} / {_gib(mem.peak)} GiB", flush=True)
    _add_kv_reclaimers(pool)
    print(f"[kv-pool] on: slot 0 window {eng.kv.max_len:,} rows reserved as "
          f"{_gib(static)} GiB of addresses, page {pages.bufs[0].page >> 20} MiB, hard floor "
          f"{_gib(pool.hard_floor)} GiB, margin {pool.margin_rows} rows", flush=True)


def _window_above_limits(a, eng) -> None:
    """ENG-363: slot 0's window above the one the limits are set at (memlimits.LIMITS_WINDOW) holds
    its KV from the load on when the pool is off: the policy's ceiling and peak go up by the KV of
    the rows above it, as they do for the side slots, so the caches are not shed and a long prefill
    not refused for slot 0's own window (Super VL at 1,048,576 tokens: +6 GiB)."""
    mem = dict.get(STATE, "mem")
    if mem is None or not mem.on or eng is None:
        return
    extra = int(eng.cost().view_bytes_per_token) * max(0, int(a.max_len) - memlimits.LIMITS_WINDOW)
    if extra <= 0:
        return
    mem.fixed += extra             # kept when a residency document sets the limits again
    mem.ceiling += extra
    mem.peak += extra
    print(f"[memory] ceiling and peak +{_gib(extra)} GiB for slot 0's window above "
          f"{memlimits.LIMITS_WINDOW:,} tokens: {_gib(mem.ceiling)} / {_gib(mem.peak)} GiB", flush=True)


def kv_pool_report() -> dict | None:
    pool = kvpool.POOL
    if pool is None:
        return None
    out = pool.report()
    slots = []
    par = dict.get(STATE, "pool")
    engs = [(0, dict.get(STATE, "engine"))]
    if par is not None:
        engs += [(s.index, (s.values or {}).get("engine")) for s in par.side()]
    for i, e in engs:
        ps = getattr(getattr(e, "kv", None), "pages", None)
        if ps is not None:
            slots.append({"slot": i, "window": int(e.kv.max_len), "kv_length": int(e.kv.length),
                          "committed_rows": ps.rows,
                          "committed_gib": round(ps.committed_bytes() / (1 << 30), 3)})
    out["slots"] = slots
    return out


def _probe_done(rec) -> None:
    """A request's probe snapshots to the ring's writer, and the probe stage's shadow answer on
    its own `prompt` row into the journal row (`extra.probe`). Never raises into the request."""
    cap, plan = rec.probe, rec.plan
    try:
        row = plan.row if plan is not None and plan.row is not None else {}
        stage = STATE.get("probe_stage")
        if stage is not None and plan is not None and plan.router_cfg is not None:
            from server.conductor import schema as cond_schema
            ps = cond_schema.probe_stage(plan.router_cfg)
            if ps["mode"] != "off" and row.get("source") != "probe":
                row.setdefault("extra", {})["probe"] = stage.stage(
                    ps, cap.vector("prompt"), cap.names, reason="shadow")
        STATE["probe"].submit(cap, {
            "ts": rec.ts, "request_id": rec.request_id, "row_id": row.get("id"),
            "session": plan.session.key if plan is not None and plan.session is not None
            else None, "class_id": row.get("class_id"), "effort": row.get("effort"),
            "source": row.get("source"), "profile": plan.res.profile if plan is not None
            else None, "client": rec.client_kind, "finish": rec.finish_reason,
            "tool_calls": int(rec.tool_calls or 0), "reasoning_tokens": rec.reasoning_tokens,
            "prompt_tokens": rec.prompt_tokens, "pos": cap.pos}, int(rec.tool_calls or 0))
    except Exception as exc:                                       # noqa: BLE001
        print(f"[probe] {rec.request_id}: snapshots dropped ({type(exc).__name__}: {exc})",
              flush=True)
        cap.release()


def _finishing(rec, step: str) -> None:
    """One of the three steps after the token loop (flush, saving_state, final_chunk).
    The first also records the state the loop ended in. Three assignments a request."""
    if rec.step is None:
        rec.end_state = activity_mod.state_of(rec, ignore_step=True)
        rec.t_finishing = time.perf_counter()
    rec.step = step


def _stop_detail(finish: str, ids: list, eos, cut: bool, pstop) -> str | None:
    """What refines a `stop`: the end token, a stop string or the repetition guard."""
    if finish != "stop":
        return None
    if pstop is not None and pstop.hit:
        return "pattern_guard"
    if cut:
        return "stop_string"
    return "eos" if ids and ids[-1] in eos else None


def conversation_id(body: dict, headers) -> str | None:
    """A label for the conversation, if the client offers one. It is never load-bearing.

    The state store matches on the token prefix and checks it element for element, so a wrong or
    missing conversation id costs a cache hit and can never produce a wrong one. What the id buys
    is a readable `/v1/cache/stats` and, when two conversations share a prefix, a way to tell which
    entry belongs to which.
    """
    for key in ("conversation_id", "session_id", "user"):
        v = body.get(key)
        if isinstance(v, str) and v:
            return v[:128]
    meta = body.get("metadata")
    if isinstance(meta, dict):
        v = meta.get("conversation_id") or meta.get("session_id")
        if isinstance(v, str) and v:
            return v[:128]
    try:
        v = headers.get("X-Conversation-Id")
    except Exception:
        v = None
    return v[:128] if isinstance(v, str) and v else None


def _normalize_tool_arguments(messages: list) -> list:
    """Assistant tool_calls arrive with `arguments` as a JSON STRING (OpenAI shape); the chat
    template iterates it as a mapping, so it must be a dict before rendering. Unparseable strings
    are wrapped rather than raising a 500 mid-prompt."""
    out = []
    for m in messages:
        calls = m.get("tool_calls") if isinstance(m, dict) else None
        if calls:
            m = dict(m)
            fixed = []
            for c in calls:
                c = dict(c)
                fn = dict(c.get("function") or {})
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        fn["arguments"] = json.loads(args) if args.strip() else {}
                    except ValueError:
                        fn["arguments"] = {"value": args}
                c["function"] = fn
                fixed.append(c)
            m["tool_calls"] = fixed
        out.append(m)
    return out


def _with_directive(messages: list, note: str) -> list:
    """`messages` with `note` as the last paragraph of the system message (one is added if the
    conversation has none; the template renders it after the tools either way)."""
    if messages and isinstance(messages[0], dict) and messages[0].get("role") == "system":
        first = dict(messages[0])
        c = first.get("content")
        if isinstance(c, list):
            first["content"] = list(c) + [{"type": "text", "text": "\n\n" + note}]
        else:
            first["content"] = (c.rstrip() + "\n\n" + note) if c else note
        return [first] + list(messages[1:])
    return [{"role": "system", "content": note}] + list(messages)


def continues_final(body: dict, messages: list | None = None) -> bool:
    """W17 (FX15, 2026-10-08): whether the prompt continues the request's last message, an
    assistant's partial answer, instead of closing it and starting a new answer. Open WebUI's
    "Continue Response" after Stop sends the conversation with the partial answer last; rendered
    with a generation prompt, the partial was closed as a finished turn and the model wrote the
    story again from its title (chatcmpl-9ea66357: 1,613 new tokens, the saved answer twice). A
    last message from the assistant with text and no tool calls is continued: the template's
    `continue_final_message` leaves it open, after its closed (empty) think block, so no new turn
    and no new reasoning. The body's `continue_final_message` (vLLM's name) or
    `add_generation_prompt` says otherwise when it is given."""
    explicit = body.get("continue_final_message")
    if isinstance(explicit, bool) and not explicit:
        return False
    if body.get("add_generation_prompt") is True and explicit is not True:
        return False
    msgs = messages if messages is not None else body.get("messages")
    if not isinstance(msgs, list) or not msgs or not isinstance(msgs[-1], dict):
        return False
    last = msgs[-1]
    if last.get("role") != "assistant" or last.get("tool_calls"):
        return False
    c = last.get("content")
    if isinstance(c, list):
        c = "".join(x.get("text", "") for x in c if isinstance(x, dict) and x.get("type") == "text")
    return isinstance(c, str) and bool(c.strip())


def build_prompt(body: dict, client: "thinking_mod.Client | None" = None, *, count: bool = False,
                 left_out: int = 0) -> tuple[torch.Tensor, str, bool]:
    """The prompt ids, what kind of request it was, and whether it ends inside `<think>`.

    The third value is the whole of bug 1. `add_generation_prompt=True` with thinking enabled ends
    the rendered prompt with `<|im_start|>assistant\n<think>\n`, so the OPENING tag is part of the
    prompt and the model never generates it: the only tag in the output is `</think>`, and a client
    that folds a reasoning block on a matched pair sees an unmatched closer and folds nothing.
    Whoever renders the prompt is the only code that can know this, so it is answered here and the
    server puts the tag back.

    `client`: the request's thinking (server/thinking.py), read from the body when not given. Its
    off forms render the template's own thinking off, and its effort word the template's sentence.

    `count`: the prompt's length only, as an int, with no tensor on the device: the router asks
    for it before the engine lock (the in-context stage's threshold), where nothing may touch the
    GPU's streams. `count="think"`: (the length, whether it ends inside `<think>`), the side lane's
    question (server/preempt.py).

    `count="ids"`: the prompt's ids as a host list (parallel requests: which slot may take it is
    decided from them before any lock, `_par_side_need`).

    `left_out`: the request's oldest images that `images_mod.collect` left out; each is rendered as
    the `images_mod.LEFT_OUT` text, so the template writes no placeholder for it. The router and
    the monitor read the client's messages as sent; only the prompt changes.
    """
    tok = STATE["tok"]
    if "messages" in body:
        if client is None:
            client = thinking_mod.read(body, strict=False)
        kwargs = dict(body.get("chat_template_kwargs") or {})
        if client.off:
            kwargs.setdefault("enable_thinking", False)
        kwargs.setdefault("enable_thinking", True)
        # Tool calling, request side. The Qwen template has native tool support and
        # renders the official <tool_call><function=...> protocol when `tools` is passed; without
        # this the model never sees the client's tool schemas (chat 53d7ca38: it announced a web
        # search and stopped; chat efa916ed: it invented write_file from its priors).
        tools = body.get("tools")
        mode, name = compat.tool_choice(body)
        if tools and mode == "none":
            tools = None
        if tools:
            kwargs["tools"] = tools
        if body.get("tool_choice") is not None:
            kwargs["tool_choice"] = body["tool_choice"]
        messages = body["messages"]
        if left_out:
            messages = images_mod.leave_out(
                messages, left_out, (STATE.get("image_limits") or images_mod.Limits()).max_images)
        messages = _normalize_tool_arguments(messages)
        # the template has no notion of `tool_choice`, so `required` and a named function
        # are asked for in words -- one sentence at the end of the system turn, after the tool
        # definitions and the client's own system prompt. `auto` and `none` add nothing.
        note = compat.directive(mode, name) if tools else None
        if note:
            messages = _with_directive(messages, note)
        # The template resolves `reasoning_effort` to xhigh unless told otherwise, and xhigh is a
        # paragraph of instructions telling the model to check its assumptions and consider
        # alternatives. A server default is the cheapest way to make it think less, because it
        # changes what the model is asked for rather than cutting it off part-way.
        # The client's effort word, in whichever shape it came, as the word the template knows
        # (it raises on any other, so `high`, `none` and the like used to be a 500).
        if client.template_effort:
            kwargs["reasoning_effort"] = client.template_effort
        elif kwargs.get("reasoning_effort") not in thinking_mod.TEMPLATE_WORDS:
            kwargs.pop("reasoning_effort", None)
        effort = STATE.get("reasoning_effort")
        if effort:
            kwargs.setdefault("reasoning_effort", effort)
        # W17: a trailing partial answer is continued, not closed and answered again
        if continues_final(body, messages):
            kwargs.update(add_generation_prompt=False, continue_final_message=True)
        else:
            kwargs["add_generation_prompt"] = True
        # a client that asked for no thinking, to a model whose template opens the reasoning block
        # anyway (LFM2.5 always does): the block is closed right after the generation prompt, as the
        # Qwen template's own thinking off renders it, so the answer starts at once
        close = (list(tok(CLOSE_THINK, add_special_tokens=False).input_ids)
                 if client.off and kwargs.get("add_generation_prompt") else [])
        seg = STATE.get("segments")
        if seg is None:
            enc = tok.apply_chat_template(messages, return_tensors="pt", return_dict=True,
                                          **kwargs)
            cids = _as_ids(enc, device="cpu")
            if close and opens_think(tok.decode(cids[-8:].tolist(), skip_special_tokens=False)):
                cids = torch.cat([cids, torch.tensor(close, dtype=cids.dtype)])
            if count == "ids":
                return cids.tolist()
            if count == "think":
                return int(cids.numel()), opens_think(tok.decode(cids.tolist(),
                                                                 skip_special_tokens=False))
            if count:
                return int(cids.numel())
            ids = cids.to(STATE["device"])
            # Decoded from the ids rather than rendered a second time, so what is inspected is
            # exactly the token sequence the engine is about to be given.
            return ids, "chat", opens_think(tok.decode(ids.tolist(), skip_special_tokens=False))
        # the same ids a message at a time, the messages earlier turns already had looked up
        # (server/tokcache.py); the reasoning tag read from the prompt's decoded tail
        text = tok.apply_chat_template(messages, tokenize=False, **kwargs)
        ids = seg.encode(text)
        if close and opens_think(tok.decode(ids[-8:], skip_special_tokens=False)):
            ids = list(ids) + close
        if count == "ids":
            return list(ids)
        if count == "think":
            return len(ids), opens_think_ids(tok, ids, opens_think)
        if count:
            return len(ids)
        return (torch.tensor(ids, dtype=torch.long, device=STATE["device"]), "chat",
                opens_think_ids(tok, ids, opens_think))
    text = body.get("prompt")
    if isinstance(text, list):
        text = text[0]
    ids = _as_ids(tok(text or "", return_tensors="pt"))
    # SRV-443: a text prompt answers every `count` as a chat prompt does (`_par_side_need` asks
    # for the host ids before it picks a slot; a tensor there was a 500 on every
    # /v1/completions request with parallel slots)
    if count == "ids":
        return [int(t) for t in ids.tolist()]
    if count == "think":
        return int(ids.numel()), False
    if count:
        return int(ids.numel())
    return ids, "text", False


def _vision_context(ids: list[int], images: list, eng, on_encode=None) -> tuple:
    """`(ids, mm)`: the prompt with each image's placeholder expanded to its rows and the request's
    image context, as the served tower writes them: its own `context` (Super VL's RADIO tower:
    `<img>` rows `</img>`, no rotary, engine/models/nemotron_h/vision.py), else engine/vision.py's
    (the Qwen towers: the placeholder's rows and their three-axis rotary)."""
    tower = STATE["vision"]
    own = getattr(tower, "context", None)
    if own is not None:
        return own(ids, images, cache=STATE.get("image_cache"), on_encode=on_encode)
    from engine.vision import MMContext, expand
    full, spans = expand(ids, images, tower.cfg.image_token_id, tower.cfg.spatial_merge_size)
    return full, MMContext(full, spans, images, tower, eng.cfg, cache=STATE.get("image_cache"),
                           on_encode=on_encode)


def _mm_prompt(prompt: torch.Tensor, images: list, rec=None):
    """The prompt with each image's one placeholder expanded to its rows, and the
    request's `MMContext`. `rec.encoding` is the live view's "encoding image i of n"."""
    def on_encode(i, n):
        if rec is not None:
            rec.encoding = (i, n) if i is not None else None

    ids, mm = _vision_context(prompt.tolist(), images, STATE["engine"], on_encode)
    # SRV-546: what `_mm_collapse` and `_mm_text_pos` read of the expansion
    mm.n_text = int(prompt.numel())
    mm.placeholder = int(STATE["vision"].cfg.image_token_id)
    if STATE.get("verbose"):
        grids = ",".join("x".join(str(g) for g in im.grid) for im in images)
        print(f"[vision] {len(images)} image(s) grids={grids} rows={mm.image_tokens} "
              f"delta={mm.delta} sources={','.join(im.source for im in images)}", flush=True)
    return torch.tensor(ids, dtype=torch.long, device=prompt.device), mm


#: SRV-546: a request with images resumes at the end anchor as a text request does: the replay
#: and the whitespace alignment run on its expanded key ids (`_mm_end_replay`), an end anchor or
#: an NVMe entry kept after images is offered to it (`end_mm`). Off: the anchors in place only
MM_END = (_S.get("MM_END") or "on").strip().lower()


def _stage_prefill(eng, drafter, ids, device, **kw):
    """`cache.prefill` for the router's in-context stage, which says a long re-read as the turn
    does (SRV-546: positions only)."""
    seen: dict = {}
    logits, reused, forwarded = cache.prefill(eng, drafter, ids, device, info=seen, **kw)
    if seen.get("why"):
        print(f"[cache] the router's stage re-read {forwarded:,} of {len(ids):,} tokens (resumed "
              f"at {reused:,}, {seen.get('kind') or 'cold'}): {seen['why']}", flush=True)
    return logits, reused, forwarded


def _mm_collapse(keys: list[int], ref) -> list[int] | None:
    """The text prompt of `keys` (an expanded prompt whose image rows are their key ids): each
    image's rows, and the marks the tower's expansion puts around them, back to the one
    placeholder the template writes. `ref` is the request's own context (`MMContext`), whose
    expansion says how many marks an image takes. None when the rows do not come out whole."""
    from engine.vision import KEY_BASE
    pre = _mm_pre(ref)
    ph = getattr(ref, "placeholder", None)
    if pre is None or ph is None:
        return None
    ph = int(ph)
    out: list[int] = []
    i, n = 0, len(keys)
    while i < n:
        t = int(keys[i])
        if t < KEY_BASE:
            out.append(t)
            i += 1
            continue
        r = i
        while r < n and int(keys[r]) == t:
            r += 1
        if pre:
            if len(out) < pre or r + pre > n:
                return None
            del out[-pre:]
        out.append(ph)
        i = r + pre
    return out


def _mm_runs(keys: list[int]) -> list[int]:
    """The image key of each run of image rows in `keys`, in order."""
    from engine.vision import KEY_BASE
    runs, prev = [], None
    for t in keys:
        t = int(t)
        if t >= KEY_BASE and t != prev:
            runs.append(t)
        prev = t if t >= KEY_BASE else None
    return runs


def _mm_end_replay(res, text: torch.Tensor, prompt: torch.Tensor, mm, images: list, rec=None):
    """SRV-546: the end anchor's replay for a request with images. A text request's copy is
    replayed (the hidden spans put back) and aligned (the whitespace the parser and the template
    took out of the model's own tokens) before the end anchor serves it; a request with images was
    served as rendered, so its copy parted from the rows at the first such spot of the whole
    conversation and it resumed at a grid anchor below it (live 2026-10-17 10:10:28: the first
    screenshot of a 114,336-token run parted at 88,640, an edit_file parameter's indentation,
    and resumed at 29,696: 84,640 tokens re-read in 149 s). Here the replay runs on the
    expanded key ids, which are the rows' own coordinates, and the served prompt is rebuilt
    from what it returns. `(prompt, mm, spans put back, tokens aligned, added)`; the inputs as
    they were when nothing applies."""
    keys = list(mm.key_ids)
    out, spans, aligned = _end_replay(res, keys)
    # SRV-657: why the replay did not serve, for `_say_reread`'s line (live 2026-10-21 00:05:59:
    # 43,856 tokens re-read after a guest, and the log said nothing of where or why)
    STATE["mm_end_why"] = "nothing to replay or align"
    if not spans and not aligned:
        return prompt, mm, [], 0, 0
    if out[:len(keys)] == keys and len(out) == len(keys):
        return prompt, mm, [], 0, 0
    STATE["mm_end_why"] = "the images differ from the rows' (ones or order)"
    if _mm_runs(out) != _mm_runs(keys):
        return prompt, mm, [], 0, 0         # the images are not the same ones, in the same order
    STATE["mm_end_why"] = "the replayed ids do not collapse to text"
    new_text = _mm_collapse(out, mm)
    if new_text is None:
        return prompt, mm, [], 0, 0
    t = torch.tensor(new_text, dtype=text.dtype, device=text.device)
    prompt2, mm2 = _mm_prompt(t, images, rec)
    STATE["mm_end_why"] = "the rebuilt prompt's rows are not the replay's"
    if list(mm2.key_ids) != out:
        return prompt, mm, [], 0, 0         # never serve rows the replay did not ask for
    STATE["mm_end_why"] = None
    mm.release()
    return prompt2, mm2, list(spans), int(aligned), len(out) - len(keys)


def _mm_pre(mm) -> int | None:
    """The marks the tower's expansion puts before (and after) each image's rows: 0 for the
    Qwen towers, 1 for Super VL's `<img>` and `</img>`; None for a context it cannot tell."""
    spans = list(getattr(mm, "spans", None) or ())
    n_text = int(getattr(mm, "n_text", 0) or 0)
    if not spans or n_text <= 0:
        return None
    extra = int(mm.n) - n_text - sum(int(s.n) - 1 for s in spans)
    if extra < 0 or extra % (2 * len(spans)):
        return None
    return extra // (2 * len(spans))


def _mm_text_pos(mm, p: int) -> int:
    """The expanded position of text position `p` of a request with images: `p` plus the rows
    each image whose placeholder sits before it added (its rows and marks, less the placeholder)."""
    pre = _mm_pre(mm)
    if pre is None:
        return int(p)
    add = 0
    for s in mm.spans:
        if int(s.start) - pre - add >= p:      # the image's placeholder in text positions
            break
        add += int(s.n) - 1 + 2 * pre
    return int(p) + add


def _as_ids(enc, device: str | None = None) -> torch.Tensor:
    """One 1-D int tensor of token ids, whatever shape the tokeniser handed back.

    The chat-template call returns a mapping, a plain tensor or a fast-tokeniser Encoding
    depending on the version and the arguments, and getting this wrong is a 500 on every request.
    """
    obj = enc
    for key in ("input_ids",):
        if hasattr(obj, key):
            obj = getattr(obj, key)
            break
        if isinstance(obj, dict) or hasattr(obj, "keys"):
            try:
                obj = obj[key]
                break
            except Exception:
                pass
    if not torch.is_tensor(obj):
        if hasattr(obj, "ids"):
            obj = torch.tensor(obj.ids, dtype=torch.long)
        else:
            obj = torch.tensor(obj, dtype=torch.long)
    while obj.dim() > 1:
        obj = obj[0]
    return obj.to(device or STATE["device"])


def eos_ids(body: dict) -> set[int]:
    tok = STATE["tok"]
    out = set()
    for t in (tok.eos_token_id, STATE["cfg_eos"]):
        if isinstance(t, int):
            out.add(t)
        elif isinstance(t, (list, tuple)):
            out.update(int(x) for x in t)
    return out


# ------------------------------------------------------------------ HTTP
def _chunk(cid: str, model: str, created: int, delta: dict, finish=None, usage=None,
           error=None, extra: dict | None = None, logprobs: dict | None = None) -> str:
    body = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model,
            "choices": [] if delta is None and finish is None else
            [{"index": 0, "delta": delta or {}, "finish_reason": finish, "logprobs": logprobs}]}
    if usage is not None:
        body["usage"] = usage
    if extra:
        # `usage`, `timings` and `metrics` of the whole request, on the ONE chunk of the
        # stream that carries them -- see server/usage.py's `placement`.
        body.update(extra)
    if error is not None:
        # Not in the OpenAI schema, and deliberately alongside a real `finish_reason` rather than
        # instead of one: a client that only knows the schema still sees the stream end, and a
        # client that looks can find out why.
        body["error"] = error
    return "data: " + json.dumps(body, ensure_ascii=False) + "\n\n"


def _log_request(cid: str, n_prompt: int, n_out: int, finish: str, t0: float, *,
                 stream: bool, exc: BaseException | None = None,
                 pen: PenaltySpec | None = None, pattern: str | None = None,
                 rec: "usage_mod.RequestRecord | None" = None, temp: float = 0.0,
                 tools: str | None = None, sup=None) -> None:
    """One line per generation, always, whatever happened to it.

    The server used to log the HTTP status and nothing else, so an answer that stopped at the
    default token limit and an answer that stopped because the engine raised looked the same from
    the outside -- a 200 and a short reply. Everything needed to tell those apart is here.

    `rec` is the request's record: it takes the loop's block counts from here, where they
    are popped, and its end time is this line's, so `timings.total_ms` is the `ms` printed.

    `tools` is `parsed:N` -- the calls read out of the answer -- on a request that carried
    tools or produced a call; a tool-free line is what it was.

    `sup`: a request the conductor's monitor watched says `conductor=MODE acts=N hidden=N`
    (and `gave_up` when it did); a raw request's line is what it was.
    """
    now = time.perf_counter()
    ms = (now - t0) * 1e3
    rate = (n_out - 1) / (ms / 1e3) if n_out > 1 and ms > 0 else 0.0
    with QUEUE:
        if finish == "error":
            INFLIGHT["errors"] += 1
        elif finish == "timeout":
            INFLIGHT["timeouts"] += 1
        elif finish == "abandoned":
            INFLIGHT["abandoned"] += 1
    # an exception's message, capped and withheld if it quotes the request
    tail = f"  !! {type(exc).__name__}: {logbuf.safe_message(exc)}" if exc is not None else ""
    pen_s = (f" pen=({pen.rep:g},{pen.presence:g},{pen.freq:g},n={pen.no_repeat})"
             if pen is not None and pen.penalizes else "")
    if pen is not None and pen.bias:
        pen_s += f" bias={len(pen.bias)}"
    tools_s = f" tools={tools}" if tools else ""
    if sup is not None:
        tools_s += (f" conductor={sup.mode} acts={sup.acts} hidden={sup.hidden_tokens}"
                    + (" gave_up" if sup.gave_up else ""))
    pat_s = f" pattern-stop({pattern})" if pattern else ""
    # which requests sample. Only a sampled request says so, as `pen=` only says it when
    # on, so a greedy line is what it was.
    temp_s = f" temp={temp:g}" if temp > 0 else ""
    job = rec.lane_job if rec is not None else None
    # a side-lane job keeps its own counts; the generation it interrupted keeps STATE's
    bs = job.blocks if job is not None else STATE.pop("blocks", None)
    blk_s = bs.fields(n_out) if bs is not None else ""
    if job is not None:
        blk_s += (f" lane={job.where} wait_ms={job.wait_ms or 0:.0f}"
                  f" lent_ms={job.lent_ms or 0:.0f}")
    if rec is not None:
        rec.absorb_blocks(bs)
        rec.t_end = now
    # SRV-197: the sequence slot it ran on, with `--parallel N`
    slot_s = (f" slot={rec.slot}" if rec is not None and getattr(rec, "slot", None) is not None
              else "")
    print(f"[req] {cid} {'stream' if stream else 'json'} prompt={n_prompt} "
          f"completion={n_out} finish={finish} {ms:.0f} ms {rate:.2f} tok/s{temp_s}{pen_s}{pat_s}"
          f"{slot_s}{tools_s}{blk_s}{tail}", flush=True)


def _tools_note(body: dict, parsed: int, dropped: int = 0) -> str | None:
    """The `[req]` line's `tools=` field: None for a request without tools that called nothing."""
    if not body.get("tools") and not parsed:
        return None
    return f"parsed:{parsed}" + (f",dropped:{dropped}" if dropped else "")


def _conductor_fields(rec: "usage_mod.RequestRecord") -> None:
    """The ledger's conductor columns (SRV-115): the monitor's executed acts, the tokens the model
    saw that the client did not send (the monitor's hidden tokens, the router's notes, new and
    replayed), and whether the conductor changed what the model saw or wrote (a monitor act; a
    router note, budget or thinking off). Null for a request the conductor never saw."""
    sup, plan = rec.sup, rec.plan
    row = plan.row if plan is not None and plan.active else None
    if sup is None and row is None:
        return
    notes = sum(int(n.get("tokens") or 0) for n in (row.get("injected") or [])) if row else 0
    notes += int((row.get("replayed") or {}).get("tokens") or 0) if row else 0
    rec.conductor_acts = sup.acts if sup is not None else 0
    rec.conductor_hidden_tokens = (sup.hidden_tokens if sup is not None else 0) + notes
    rec.conductor_acted = int(bool(sup is not None and sup.acted)
                              or bool(row and row.get("acted")))
    if row is not None:
        try:
            metrics.on_route(row)
        except Exception:                                          # noqa: BLE001
            pass                             # a metric must never fail a request


#: SRV-349: a head's (system, tools) tokens by its bytes; every turn of a chat sends the same head
_HEADS: "collections.OrderedDict[str, dict]" = collections.OrderedDict()
#: a package text's tokens by its hash (the harness's package sizes)
_TEXT_TOKENS: "collections.OrderedDict[str, int]" = collections.OrderedDict()


def _head_parts(body: dict, client) -> dict | None:
    """The tokens of a chat body's system messages and tool schemas as the template renders them
    (server/tokenize_api.py's differences of renders, over the head alone: the system messages
    and an empty user turn, with the tools and without; an empty turn alone), remembered by the
    head's bytes. None for a body that does not render."""
    msgs = body.get("messages")
    if not isinstance(msgs, list):
        return None
    system = [m for m in msgs if isinstance(m, dict) and m.get("role") in ("system", "developer")]
    head = {k: v for k, v in body.items() if k not in ("messages", "stream", "stream_options")}
    try:
        key = hashlib.sha256(json.dumps([system, head], sort_keys=True, default=str)
                             .encode()).hexdigest()
    except (TypeError, ValueError):
        return None
    got = _HEADS.get(key)
    if got is not None:
        return got
    empty = [{"role": "user", "content": ""}]
    base = {k: v for k, v in head.items() if k not in ("tools", "tool_choice")}
    try:
        n_bare = int(build_prompt({**base, "messages": system + empty}, client, count=True))
        n_tools = (int(build_prompt({**head, "messages": system + empty}, client, count=True))
                   if body.get("tools") else n_bare)
        n_empty = (int(build_prompt({**base, "messages": empty}, client, count=True))
                   if system else n_bare)
    except Exception:                                              # noqa: BLE001
        return None
    got = {"system": max(0, n_bare - n_empty), "tools": max(0, n_tools - n_bare)}
    _HEADS[key] = got
    while len(_HEADS) > 64:
        _HEADS.popitem(last=False)
    return got


def _text_tokens(text: str | None) -> int | None:
    if not text:
        return None
    key = hashlib.sha256(text.encode()).hexdigest()
    n = _TEXT_TOKENS.get(key)
    if n is None:
        n = _TEXT_TOKENS[key] = len(STATE["tok"](text, add_special_tokens=False).input_ids)
        while len(_TEXT_TOKENS) > 256:
            _TEXT_TOKENS.popitem(last=False)
    return n


def _harness_of(plan) -> dict | None:
    """SRV-349, `usage.tandem.harness`: the router's class and effort level for this turn and the
    packages in its prompt (this turn's notes and the replayed ones, `plan.spans`), each with its
    text's tokens. None for a request no active plan served.

    A note keeps its parts as {kind, name} with one text hash each (`Plan._fresh_block`), not the
    texts, so every package counted null (the Context panel's "— tokens", 2026-10-14). The count
    comes from this turn's block (`plan.block["items"]` hold the texts), else from an earlier turn
    that counted the same name and hash, else from the package library when its text has that
    hash."""
    if plan is None or not getattr(plan, "active", False) or getattr(plan, "row", None) is None:
        return None
    row = plan.row
    texts: dict = {}
    shas: dict = {}
    notes = list(getattr(getattr(plan, "session", None), "notes", None) or ())
    for n in notes:
        for p, h in zip(getattr(n, "parts", None) or (), getattr(n, "shas", None) or ()):
            if p.get("kind") == "package" and h:
                shas.setdefault(p.get("name"), h)
    block = getattr(plan, "block", None) or {}
    for it in block.get("items") or ():
        part = it.get("part") or {}
        if part.get("kind") == "package" and it.get("text"):
            texts.setdefault(part.get("name"), it["text"])
            if it.get("sha"):
                shas[part.get("name")] = it["sha"]
    for parts in [getattr(n, "parts", None) for n in notes] + \
            [i.get("parts") for i in getattr(plan, "injected", None) or ()]:
        for p in parts or ():
            if p.get("kind") == "package" and p.get("text"):
                texts.setdefault(p.get("name"), p["text"])
    names = []
    for span in getattr(plan, "spans", None) or ():
        for name in span[3]:
            if name not in names:
                names.append(name)
    out = {"class": row.get("class_id"), "effort": row.get("effort"),
           "packages": [{"name": n, "tokens": _package_tokens(n, texts.get(n), shas.get(n))}
                        for n in names]}
    out["memory"] = _memory_part((row.get("extra") or {}).get("memory"))
    return out


def _memory_part(info) -> dict:
    """SRV-471, `harness.memory`: the memory notes' part of the turn for the Context panel, from
    the decision row's `extra.memory`: the mode and why it is off, the index's notes and tokens,
    the bodies sent with the message and their tokens, the card's tokens. A row without it (the
    profile has no memory switch: the release's request) is off, why `unset`; an engine before
    SRV-471 sends no `memory` key at all."""
    if not isinstance(info, dict) or not info.get("mode"):
        info = {"mode": "off", "why": "unset"}
    idx = info.get("index") or {}
    bodies = info.get("bodies") or []
    return {"mode": info["mode"], "why": info.get("why"),
            "index_notes": int(idx.get("notes") or 0), "index_tokens": int(idx.get("tokens") or 0),
            "bodies": len(bodies), "body_tokens": int(info.get("body_tokens") or 0),
            "card_tokens": int(info.get("card_tokens") or 0)}


#: (package name, text hash) -> tokens, for the packages a replayed note holds
_PKG_TOKENS: "collections.OrderedDict[tuple, int]" = collections.OrderedDict()


def _package_tokens(name: str, text: str | None, sha: str | None) -> int | None:
    n = _text_tokens(text)
    if n is None and sha:
        n = _PKG_TOKENS.get((name, sha))
    if n is None:
        lib = _library_text(name)
        if lib and (not sha or hashlib.sha256(lib.encode()).hexdigest()[:12] == sha):
            n = _text_tokens(lib)
    if n is not None and sha:
        _PKG_TOKENS[(name, sha)] = n
        while len(_PKG_TOKENS) > 512:
            _PKG_TOKENS.popitem(last=False)
    return n


def _library_text(name: str) -> str | None:
    cond = STATE.get("conductor")
    if cond is None:
        return None
    try:
        text, _ = cond.library.text(cond.store.peek("packages")[1], name)
    except Exception:                                              # noqa: BLE001
        return None
    return text


def _account(rec: "usage_mod.RequestRecord") -> None:
    """Every request that reached a completion route, whatever became of it: one ledger row.

    Called once, from `Handler._complete`'s `finally` -- served, refused at the queue, rejected
    with a 400, failed or abandoned. The ledger's `submit` never blocks.
    """
    STATE["last_request_ts"] = rec.ts
    STATE["last_request_end"] = time.time()          # the prewarm's "in use" (SRV-349)
    if not rec.replay:
        cache_peek.RATES.note(int(rec.forwarded), rec.prompt_ms)    # the peek's estimate
    try:
        metrics.on_record(rec)
    except Exception:                                              # noqa: BLE001
        pass                             # a metric must never fail a request
    led = STATE.get("ledger")
    if led is not None:
        led.submit(rec.row(STATE.get("version", ""), STATE.get("code_sha", "")))


def _grammar_vocab(calls: bool = False) -> "grammar_mod.Vocab":
    """Every token as bytes, for the constraint masks: built on the first constrained
    request (half a second over 248k tokens) and kept. `calls`: SRV-619, the
    vocabulary of a calls-or-document constraint, the call markers' own ids allowed (built once
    more, on the first such request)."""
    key = "grammar_vocab_calls" if calls else "grammar_vocab"
    v = STATE.get(key)
    if v is None:
        v = STATE[key] = grammar_mod.Vocab.from_tokenizer(
            STATE["tok"], STATE["engine"].cfg.vocab_size,
            markers=grammar_mod.CALL_MARKERS if calls else ())
    return v


def _auth() -> "auth_mod.Auth":
    """The access policy: from the environment at startup; tests set their own."""
    a = STATE.get("auth")
    if a is None:
        a = STATE["auth"] = auth_mod.Auth.from_env()
    return a


def _live() -> dict:
    """The dashboard's status pill: never cached, never takes the engine lock."""
    last = STATE.get("last_request_ts")
    return {"status": "draining" if STATE.get("draining") else ("busy" if LOCK.locked() else "ok"),
            "running": INFLIGHT["running"], "waiting": INFLIGHT["waiting"],
            "lane": INFLIGHT["lane"],
            "uptime_s": round(time.time() - STATE.get("started", time.time()), 1),
            "last_request_at": dashboard_api.iso_utc(last * 1000) if last else None,
            **_monitor_mode(),
            # 1.3: the proxy's backends (--backends); absent without it
            **({"backends": _proxy_backends()} if STATE.get("proxy") is not None else {})}


def _system() -> dict:
    """`/v1/dashboard/system`: what is running, how it is configured, what it holds."""
    mem = {"gpu_allocated_bytes": None, "gpu_reserved_bytes": None,
           "gpu_max_allocated_bytes": None}
    try:
        if torch.cuda.is_available():
            budget = STATE.get("mem")
            mem = {"gpu_allocated_bytes": int(torch.cuda.memory_allocated()),
                   "gpu_reserved_bytes": int(torch.cuda.memory_reserved()),
                   "gpu_max_allocated_bytes": int(budget.lifetime() if budget is not None
                                                  else torch.cuda.max_memory_allocated())}
    except Exception:                                              # noqa: BLE001
        pass
    mem.update(dashboard_api.meminfo())
    try:
        mem["split"] = memsplit.split(STATE)     # OPS-41: where the resident set goes
    except Exception:                                              # noqa: BLE001
        mem["split"] = None
    try:
        mem["limits"] = memlimits.view(STATE)    # OPS-47: every limit and what it holds now
    except Exception:                                              # noqa: BLE001
        mem["limits"] = None
    mem["roles"] = _safe(roles_report)            # the small models under memory pressure
    led = STATE.get("ledger")
    li = led.info() if led is not None else None
    hosts = _hosts()
    gpu = None
    if hosts:
        from proxy import hostcollect
        gpu = hostcollect.gpu_block(STATE)      # NVML or one nvidia-smi call, not two
    if gpu is None:
        gpu = STATE.setdefault("gpu_sampler", dashboard_api.GpuSampler()).sample()
    try:
        caches = cache_stats()
    except Exception:                                              # noqa: BLE001
        caches = None
    return {
        "engine": {"version": STATE.get("version", ""), "git_sha": STATE.get("git_sha", ""),
                   "code_sha256": STATE.get("code_sha256", ""),
                   "started_at": dashboard_api.iso_utc_s(STATE.get("started", time.time())),
                   "uptime_s": round(time.time() - STATE.get("started", time.time()), 1),
                   "pid": os.getpid(), "model": served_name(), "base_model": STATE.get("model"),
                   "weights": weights_profile(), "max_len": STATE.get("max_len"),
                   "drafter": type(STATE.get("drafter")).__name__
                   if STATE.get("drafter") is not None else None,
                   "tree": bool(STATE.get("tree")),
                   "reasoning_format": STATE.get("reasoning_format"),
                   "reasoning_effort": STATE.get("reasoning_effort")
                   or ("off" if STATE.get("thinking_default") == "off" else None),
                   "status": _live()["status"]},
        "flags": {"args": dashboard_api.redact_args(STATE.get("args") or {}),
                  "env": dashboard_api.redact_env(dict(os.environ))},
        "memory": mem,
        "gpu": gpu,
        "caches": caches,
        "queue": {"running": INFLIGHT["running"], "waiting": INFLIGHT["waiting"],
                  "lane": INFLIGHT["lane"],
                  "side_lane": STATE["lane"].report() if STATE.get("lane") is not None else None,
                  # `--parallel N` only: a server without side slots sends what it always did
                  **({"parallel": _par_health()}
                     if dict.get(STATE, "pool") is not None else {}),
                  "max_queue": STATE.get("max_queue"),
                  "queue_timeout_s": STATE.get("queue_timeout"),
                  "request_timeout_s": STATE.get("request_timeout")},
        "inflight": {k: INFLIGHT[k] for k in ("served", "refused", "errors", "timeouts",
                                             "abandoned")},
        "ledger": {"enabled": li is not None,
                   "rows": (li or {}).get("rows") or 0, "bytes": (li or {}).get("bytes") or 0,
                   "oldest": dashboard_api.iso_utc((li or {}).get("oldest_ms")),
                   "queue": (li or {}).get("queue") or 0, "dropped": (li or {}).get("dropped") or 0},
        "disk": {"state_dir_free_bytes": dashboard_api.disk_free()},
        # VIS-54: `off` without --backends, so the Backends view need not ask /v1/config (a 404)
        "proxy": ({"mode": "embedded", "backends": _proxy_backends() or []}
                  if STATE.get("proxy") is not None else {"mode": "off", "backends": []}),
        **({"hosts": hosts} if hosts is not None else {}),
    }


def _hosts() -> list | None:
    """P1.5: this machine (the engine's process and every backend on it) and the backends'
    remote hosts (proxy/hostcollect.py); sampled when the System view asks, at most every 5 s,
    the process scan every 15 s. `--host-collector off` leaves it out."""
    if (STATE.get("args") or {}).get("host_collector", "on") == "off":
        return None
    from proxy import hostcollect
    px = STATE.get("proxy")
    try:
        backends = px.cfg.backends() if px is not None else []
    except Exception:                                              # noqa: BLE001
        backends = []
    return hostcollect.hosts_for(
        STATE, backends,
        hostcollect.engine_entry((STATE.get("args") or {}).get("port"), served_name()),
        paths=[os.path.expanduser("~/.qwen38-spark-engine")])


def dashboard() -> "dashboard_api.DashboardAPI":
    api = STATE.get("dashboard_api")
    if api is None or api.ledger is not STATE.get("ledger"):
        px = STATE.get("proxy")
        api = STATE["dashboard_api"] = dashboard_api.DashboardAPI(
            STATE.get("ledger"), live=_live, system=_system,
            backends=(lambda: px.cfg.backends()) if px is not None else None)
    return api


_MEM_CACHE: dict = {"t": 0.0, "v": None}


def _mem_gib() -> dict:
    """MemAvailable and this process's RSS in GiB, read at most every 5 s; nulls where
    /proc does not exist."""
    now = time.monotonic()
    if _MEM_CACHE["v"] is not None and now - _MEM_CACHE["t"] < 5.0:
        return _MEM_CACHE["v"]
    out = {"mem_available_gib": None, "rss_gib": None}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    out["mem_available_gib"] = round(int(line.split()[1]) / (1 << 20), 1)
                    break
        with open("/proc/self/statm") as f:
            pages = int(f.read().split()[1])
        out["rss_gib"] = round(pages * os.sysconf("SC_PAGE_SIZE") / (1 << 30), 1)
    except (OSError, ValueError, IndexError):
        pass
    _MEM_CACHE.update(t=now, v=out)
    return out


def served_name() -> str:
    """The model name the server answers with: `TandemLLM` when the conductor is configured (one
    model, the owner 2026-10-01), else `--served-model`. `STATE["model"]` stays the base engine's name
    (`base_model` where a surface shows both): the cache label, the metrics and the classifier's own
    call keep using it."""
    return COMPOSITE_NAME if STATE.get("conductor") is not None else str(STATE.get("model") or "")


def _card_facts() -> dict:
    """What the loader serves, for the model card (server/model_card.py): the pack, the window,
    the default output limit, the vision tower, the effort levels' caps."""
    w = weights_profile()
    return {"pack": w.get("name") or w.get("format"), "max_len": STATE.get("max_len"),
            "max_output": STATE.get("default_max_tokens"), "vision": STATE.get("vision") is not None,
            "effort_caps": _effort_caps(), "label": served_name(), "loaded": True}


def _effort_caps() -> dict:
    """The effort levels' reasoning caps a client's effort word maps to: the router document's
    levels when the conductor is loaded (the Routing screen edits them), else their seed."""
    store = getattr(STATE.get("conductor"), "store", None)
    try:
        levels = store.peek("router")[1]["effort"]["levels"] if store is not None else None
    except Exception:                                              # noqa: BLE001
        levels = None
    return thinking_mod.caps(levels)


def _live_engine() -> dict:
    """The engine block of the live stream: plain attribute reads off STATE, no lock."""
    eng = STATE.get("engine")
    store = STATE.get("state_store")
    kv = getattr(eng, "kv", None)
    try:
        st = ({"entries": len(store._d), "bytes_gib": round(store.bytes / (1 << 30), 2)}
              if store is not None else None)
    except (AttributeError, RuntimeError, TypeError):
        st = None
    return {"model": served_name(), "version": STATE.get("version", ""),
            "draining": bool(STATE.get("draining")),
            "kv": {"length": int(getattr(kv, "length", 0) or 0),
                   "max_len": int(STATE.get("max_len") or 0)},
            "memory": _mem_gib(), "store": st, **_monitor_mode()}


def _monitor_mode() -> dict:
    """`{"conductor": {"mode": off|shadow|on}}` for the live stream's engine block and the summary's
    status (VIS-44: the header's indicator); empty on a server with no monitor policy, so an
    older dashboard and a server without the monitor show nothing."""
    mon = STATE.get("monitor")
    if mon is None:
        return {}
    return {"conductor": {"mode": "off" if mon.mode == "raw" else mon.mode}}


def _slot_blocks(i: int):
    """Side slot `i`'s running `BlockStats` (its own values; None without side slots)."""
    pool = dict.get(STATE, "pool")
    slots = getattr(pool, "slots", None) or []
    v = slots[i].values if 0 < i < len(slots) else None
    return v.get("blocks") if v else None


def live_registry() -> "live_mod.LiveRegistry":
    """The in-flight view behind `/v1/dashboard/live`. Reads the running
    `BlockStats`, `last_prefill` and the engine block off STATE; never the engine lock."""
    reg = STATE.get("live")
    if reg is None:
        hz = int(STATE.get("live_hz", live_mod.DEFAULT_HZ))
        reg = STATE["live"] = live_mod.LiveRegistry(
            blocks=lambda: STATE.get("blocks"), last_prefill=lambda: STATE.get("last_prefill"),
            slot_blocks=_slot_blocks, engine=_live_engine, hz=hz,
            activity=bool(STATE.get("live_activity", True)) and hz > 0,
            queue_timeout=lambda: float(STATE.get("queue_timeout", 120.0)),
            windows=STATE.get("live_windows"))
        # the parallel view (/v1/dashboard/parallel): the engine's own requests and, with
        # --backends, every relayed one, on the same registry
        STATE["parallel"] = parallel_mod.ParallelBoard(
            reg, engine_name=served_name,
            cost=lambda rec, tokens: (STATE["proxy"].live_cost(rec, tokens)
                                      if STATE.get("proxy") is not None else None),
            peers=lambda: (STATE["proxy"].peer_backends()
                           if STATE.get("proxy") is not None else []))
        if dict.get(STATE, "pool") is not None:
            STATE["parallel"].slots = _par_brief
    return reg


def _par_brief() -> dict:
    """The engine's slots for the Parallel view: each slot's window, whether it is busy and what it
    does, and the joint rounds so far (server/slots.py)."""
    pool = dict.get(STATE, "pool")
    meet = pool.meeting.report() if pool.meeting is not None else {}
    return {"n": pool.n, "running": pool.running(), "decoding": len(pool.decoding()),
            "slots": [{"index": s.index, "window": s.max_len, "busy": s.busy, "phase": s.phase,
                       "request": s.owner if s.index else None} for s in pool.slots],
            "rounds": meet.get("rounds"), "joint_rounds": meet.get("joint_rounds"),
            "joint_hist": meet.get("joint_hist"), "timeouts": meet.get("timeouts"),
            "admission": pool.admission.report() if pool.admission is not None else None}


def start_proxy(a) -> None:
    """`--backends PATH`: the proxy in this process (proxy/service.py). Imported only then, so a
    server without it loads and serves exactly as before."""
    path = getattr(a, "backends", "off") or "off"
    if path == "off":
        STATE["proxy"] = None
        return
    from proxy import config as proxy_config
    from proxy.service import Service
    try:
        cfg = proxy_config.ConfigDB(path).open()
        cfg.apply_env()
    except proxy_config.ConfigError as exc:
        raise SystemExit(f"[proxy] configuration error: {exc.message}")
    svc = Service(cfg, ledger=STATE.get("ledger"), live=live_registry(), auth=_auth(),
                  reserved=tuple(x for x in (served_name(), STATE.get("model")) if x),
                  registry=metrics.REGISTRY, engine_version=STATE.get("version", ""),
                  code_sha=STATE.get("code_sha", ""), log=lambda m: print(m, flush=True),
                  # P4: the harness in front of the backends runs this engine's own conductor
                  conductor=STATE.get("conductor"),
                  monitor_base=(STATE["monitor"].policy if STATE.get("monitor") is not None
                                else None))
    svc.engine_info = lambda: {"model": served_name(), "base_model": STATE.get("model"),
                               "embedded": True, **_engine_own_settings()}
    STATE["proxy"] = svc.start()
    print(f"[proxy] on: {cfg.path}, {len(cfg.backends())} backends "
          f"({', '.join(b['id'] for b in cfg.backends()) or 'none yet: the Backends screen'}); "
          f"{served_name()} stays the engine's", flush=True)


def _engine_own_settings() -> dict:
    """What only this engine has, for the Backends screen (PL1): its parallel requests (the
    sequences it decodes at once: slot 0 and its side slots; a relayed backend takes what it
    accepts and gets no such setting from us) and the small models it runs beside Qwen."""
    pool = dict.get(STATE, "pool")
    a = dict.get(STATE, "args") or {}
    par = {"n": 1, "requested": int(a.get("parallel") or 1), "side_max_len": None,
           "side_graphs": None, "floor_gib": a.get("parallel_floor_gb")}
    if pool is not None and getattr(pool, "n", 1) > 1:
        side = pool.side()
        par.update(n=pool.n, side_max_len=side[0].max_len if side else None,
                   side_graphs=a.get("side_graphs"))
    roles: dict = {}
    try:
        dec = dict.get(STATE, "deciders")
        if dec is not None:
            st = dec.status() if hasattr(dec, "status") else []
            roles["deciders"] = [m.get("name") for m in (st or []) if m.get("state") == "loaded"]
    except Exception:                                              # noqa: BLE001
        pass
    try:
        nar = dict.get(STATE, "narrator")
        slot = getattr(nar, "slot", None) if nar is not None else None
        if slot is not None:
            rm = dict.get(STATE, "role_mem")
            state = (rm.state_of("narrator") if rm is not None and "narrator" in rm.by_name
                     else slot.state)
            roles["narrator"] = {"model": getattr(slot.model, "name", None), "state": state}
    except Exception:                                              # noqa: BLE001
        pass
    return {"parallel": par, "roles": roles}


def _proxy_backends() -> list | None:
    px = STATE.get("proxy")
    if px is None:
        return None
    try:
        return px.backends_view()
    except Exception:                                              # noqa: BLE001
        return None


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "TandemLLM"
    # 2026-10-05: no Nagle, as uvicorn's servers (vLLM, SGLang). The headers and the first frame
    # go out as two writes; on a keep-alive connection (the proxy's pool, an SDK's) Linux holds
    # the second for the reader's delayed ACK, up to 40 ms (the night pass measured that stall
    # on tests/fake_backend.py)
    disable_nagle_algorithm = True
    _status: int | None = None
    _tandem: str | None = None          # the conductor mode of the completion in hand
    _images_left_out = 0                # the completion's earlier images left out (server/images.py)

    def send_response(self, code, message=None):
        # The status of this request as it went out -- what the ledger row records.
        self._status = code
        super().send_response(code, message)
        if self._tandem is not None:
            # every completion response says which mode it ran in, so a bench client can
            # assert it was served raw (docs/server.md, "The conductor's session monitor")
            self.send_header("X-Tandem", self._tandem)
        if self._images_left_out:
            self.send_header("X-Tandem-Images-Left-Out", str(self._images_left_out))

    def log_message(self, fmt, *args):
        if STATE.get("verbose"):
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    # -------------------------------------------------------------- helpers
    def _send_bytes(self, code: int, raw: bytes, extra_headers: tuple = ()) -> None:
        """Write one response, tolerating a client that hung up first.

        A health checker that disconnects mid-write used to raise BrokenPipeError out of here
        and into socketserver's handle_error, which prints a full traceback that reads as a
        server crash. The request is gone either way; swallow it and mark the connection closed.
        """
        if self._early is not None:
            return self._early_end(code, raw)
        try:
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            for k, v in extra_headers:
                self.send_header(k, v)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True

    def _json(self, code: int, payload: dict, extra_headers: tuple = ()) -> None:
        self._send_bytes(code, json.dumps(payload, ensure_ascii=False).encode(), extra_headers)

    # -------------------------------------------------------------- a stream opened early (S1)
    # A streamed request whose profile shows its state lines opens its stream while it still
    # waits for the engine, so the client sees why ("waiting: ..."). Whatever would have been a
    # JSON refusal after that (the queue's timeout, a prompt too long, the memory gate) ends the
    # stream instead: an `error` beside `finish_reason: "error"`, then `[DONE]`, and the ledger
    # keeps the refusal's status.
    _early: dict | None = None
    _early_body = False                # the generation's own stream has taken it over
    _bar = None                        # SRV-237: the request's Open WebUI status line, if any

    def _open_early(self, cid: str, model: str, created: int, plan) -> None:
        if self._early is not None:
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        for k, v in (plan.headers() if plan is not None else ()):
            self.send_header(k, v)
        if getattr(self, "_mem_wait", False):
            self.send_header("X-Tandem-Wait", "memory")      # SRV-305
        self.end_headers()
        self._early = {"cid": cid, "model": model, "created": created, "role": False,
                       "done": False}
        self._streamed = True

    def _early_write(self, data: str) -> None:
        """One chunk of a stream opened early, the role chunk in front of the first."""
        e = self._early
        role = "" if e["role"] else _chunk(e["cid"], e["model"], e["created"],
                                           {"role": "assistant", "content": ""})
        e["role"] = True
        self.wfile.write((role + data).encode())
        self.wfile.flush()

    def _early_end(self, code: int, raw: bytes) -> None:
        e = self._early
        if e["done"]:
            return
        e["done"] = True
        try:
            err = json.loads(raw).get("error") or {}
        except ValueError:
            err = {}
        try:
            ev = self._bar.close() if self._bar is not None else None
            if ev is not None:                   # SRV-249: no status line left shimmering
                self.wfile.write(_chunk(e["cid"], e["model"], e["created"], None,
                                        extra={"event": ev}).encode())
            self._early_write(_chunk(e["cid"], e["model"], e["created"], {}, finish="error",
                                     error={"message": str(err.get("message") or code),
                                            "type": err.get("type") or "error", "code": code}))
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
        self._status = code

    def _busy(self, code: int, message: str, retry: int = 5) -> None:
        """A refusal the caller can act on: a status, a reason, and when to come back."""
        self._send_bytes(code, json.dumps({"error": {"message": message, "type": "server_busy",
                                                     "code": code}}).encode(),
                         extra_headers=(("Retry-After", str(retry)),))

    def _read(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        self._raw = self.rfile.read(n)
        return json.loads(self._raw or b"{}")

    # -------------------------------------------------------------- access
    def _peer(self) -> str:
        return (self.client_address or ("",))[0]

    def _allowed(self, route: str) -> bool:
        """The route's policy (server/auth.py); on a refusal the answer is already written."""
        verdict = _auth().decide(route, self._peer(), self.headers)
        if verdict == "ok":
            return True
        path = self.path.split("?")[0]
        if verdict == "absent":
            self._json(404, {"error": {"type": "not_found", "message": f"no route {path}"}})
        else:
            self._json(401, {"error": {"type": "unauthorized",
                                       "message": "a bearer token or the dashboard session is "
                                                  "needed"}},
                       extra_headers=(("WWW-Authenticate", 'Bearer realm="TandemLLM"'),))
        return False

    def _session(self, method: str, body: dict | None = None) -> None:
        """`/v1/dashboard/session`: POST logs in, GET says whether, DELETE logs out.

        With the dashboard login off, or no admin password set (AU1), there is nothing to log in
        to: GET says so (`login: false`, `password_set`), and POST and DELETE do not exist. The
        page has no sign-in screen either way; the cookie flow is kept for the later sign-in."""
        a = _auth()
        # AU1: no admin password set, or the login off -- nothing to sign in to. GET says so
        # (`login: false`, `password_set`), POST and DELETE do not exist.
        if not a.login or not a.password_set:
            if method == "GET":
                return self._json(200, {"contract_version": dashboard_api.SESSION_CONTRACT,
                                        "authenticated": True, "login": False,
                                        "password_set": a.password_set, "expires_at": None},
                                  extra_headers=(("Cache-Control", "no-store"),))
            return self._json(404, {"error": {"type": "not_found",
                                              "message": "the dashboard login is off" if
                                              a.password_set else "no admin password is set"}})
        secure = "; Secure" if (self.headers.get("X-Forwarded-Proto") or "") == "https" else ""
        if method == "GET":
            exp = a.session(self.headers)
            if exp is None and not a.admin_bearer(self.headers):
                return self._allowed("dashboard")               # the 401
            exp = exp or int(time.time()) + auth_mod.SESSION_S
            return self._json(200, {"contract_version": dashboard_api.SESSION_CONTRACT,
                                    "authenticated": True, "login": True, "password_set": True,
                                    "expires_at": dashboard_api.iso_utc_s(exp)})
        if method == "DELETE":
            a.revoke(self.headers)
            return self._send_empty(204, (("Set-Cookie", f"{auth_mod.COOKIE}=; HttpOnly; "
                                                         f"SameSite=Strict; Path=/; Max-Age=0"
                                                         f"{secure}"),))
        who = self.headers.get("X-Real-IP") or self._peer()
        if a.login_blocked(who):
            return self._json(429, {"error": {"type": "too_many",
                                              "message": "too many failed logins; wait a minute"}},
                              extra_headers=(("Retry-After", "60"),))
        token = (body or {}).get("token") if isinstance(body, dict) else None
        if not isinstance(token, str) or not auth_mod._eq(token, a.admin):
            a.login_failed(who)
            return self._json(401, {"error": {"type": "unauthorized", "message": "wrong token"}},
                              extra_headers=(("WWW-Authenticate", 'Bearer realm="TandemLLM"'),))
        value, _ = a.make_cookie()
        return self._send_empty(204, (("Set-Cookie", f"{auth_mod.COOKIE}={value}; HttpOnly; "
                                                     f"SameSite=Strict; Path=/; "
                                                     f"Max-Age={auth_mod.SESSION_S}{secure}"),))

    def _cache_peek(self, body, prewarm: bool = False) -> None:
        """`POST /v1/cache/peek` and `/v1/cache/prewarm` (SRV-349, server/cache_peek.py)."""
        if not isinstance(body, dict) or not isinstance(body.get("messages"), list) \
                or not body["messages"]:
            return self._json(400, {"error": {"message": "messages must be a non-empty list",
                                              "type": "invalid_request_error",
                                              "param": "messages"}})
        px = STATE.get("proxy")
        if px is not None and body.get("model"):
            from proxy import route as proxy_route
            try:
                routed = px.route("/v1/chat/completions", body) is not None
            except proxy_route.NotFound:
                routed = True
            if routed:
                return self._json(404, {"error": {"message": "this model is not the engine's: no "
                                                             "cache to read", "type": "not_found"}})
        try:
            ids, session = _peek_ids(body, self.headers)
        except Exception as exc:                                   # noqa: BLE001
            # a body the template cannot render is the client's
            return self._json(400, {"error": {"message": f"{type(exc).__name__}: "
                                                         f"{str(exc)[:300]}",
                                              "type": "invalid_request_error"}})
        out = _cache_prewarm(ids, session) if prewarm else _cache_peek(ids, session)
        if out is None:
            return self._json(409, {"error": {"message": "the engine is serving a request; ask "
                                                         "again when it is idle", "type": "busy"}})
        return self._json(200, out)

    def _chat(self, method: str) -> bool:
        """`/v1/chat-app/*`: the Chat tab's service, chatd, behind the dashboard's auth
        (server/chat_forward.py, SRV-343); False for every other path."""
        return chat_forward.route(self, method, _auth())

    def _send_empty(self, code: int, headers: tuple = ()) -> None:
        try:
            self.send_response(code)
            for k, v in headers:
                self.send_header(k, v)
            self.send_header("Content-Length", "0")
            self.end_headers()
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True

    def _dashboard_get(self, path: str) -> None:
        """`GET /v1/dashboard/{summary,usage,requests,system,logs,live,metrics}`."""
        from urllib.parse import parse_qs, urlsplit
        if path == "/v1/dashboard/session":
            return self._session("GET")
        if not self._allowed("dashboard"):
            return
        if path == "/v1/dashboard/metrics":
            # The /metrics page under the dashboard's rule: the Performance view reads its
            # 5-minute figures from it, and with the login off a browser has no session that
            # /metrics would take. /metrics itself keeps its own rule for the scrapers.
            return metrics.serve(self)
        q = {k: v[-1] for k, v in parse_qs(urlsplit(self.path).query).items()}
        if path in ("/v1/dashboard/logs", "/v1/dashboard/live"):
            try:
                return (self._dashboard_logs if path.endswith("logs") else self._dashboard_live)(q)
            except dashboard_api.ApiError as exc:
                return self._json(exc.status, exc.body())
        if path == "/v1/dashboard/parallel":
            # every request in flight, the engine's and the relayed ones (server/parallel.py)
            live_registry()
            try:
                since = float(q["since"]) if q.get("since") else None
            except ValueError:
                return self._json(400, {"error": {"type": "bad_request",
                                                  "message": "since: a number (unix seconds)"}})
            return self._json(200, STATE["parallel"].snapshot(since),
                              extra_headers=(("Cache-Control", "no-store"),))
        api = dashboard()
        fn = {"/v1/dashboard/summary": api.summary, "/v1/dashboard/usage": api.usage,
              "/v1/dashboard/requests": api.requests,
              "/v1/dashboard/system": lambda _q: api.system()}.get(path)
        if fn is None:
            return self._json(404, {"error": {"type": "not_found", "message": f"no route {path}"}})
        try:
            return self._json(200, fn(q), extra_headers=(("Cache-Control", "no-store"),))
        except dashboard_api.ApiError as exc:
            return self._json(exc.status, exc.body())

    def _dashboard_logs(self, q: dict) -> None:
        """`GET /v1/dashboard/logs`: the backlog, then live lines, as server-sent events.

        Never takes the engine lock. A reader that disconnects is cleaned up without a traceback
        (rule); one that does not read loses its oldest lines and gets `event: gap`.

        A reader that CLOSED is noticed within a second, not at the next write: on a
        quiet log that was the heartbeat, up to `log_ping_s` (15 s) later -- and a write to a
        closed socket only fails the time after -- so a closed tab kept its place under the
        four-stream cap and reopening the Dev tab a few times in a row got 429. And a new stream
        that finds the cap full takes the place of a closed one at once (`LogBuffer.subscribe`).
        """
        buf = STATE.get("log_buffer") or logbuf.BUFFER
        level = q.get("level") or "info"
        if level not in logbuf.LEVELS:
            raise dashboard_api.ApiError(400, "bad_request",
                                         f"level must be one of {', '.join(logbuf.LEVELS)}")
        grep = q.get("grep") or None
        if grep is not None and len(grep) > 128:
            raise dashboard_api.ApiError(400, "bad_request", "grep is at most 128 characters")
        try:
            backlog = int(q.get("backlog") or 500)
            lei = self.headers.get("Last-Event-ID")
            since = q.get("since")
            after = int(lei) if lei else (int(since) if since and since.isdigit() else None)
        except ValueError:
            raise dashboard_api.ApiError(400, "bad_request", "backlog and Last-Event-ID are integers")
        if not 0 <= backlog <= 5000:
            raise dashboard_api.ApiError(400, "bad_request", "backlog must be 0..5000")
        if after is None and since:
            try:
                t = dashboard_api._dt.datetime.fromisoformat(since.replace("Z", "+00:00"))
            except ValueError:
                raise dashboard_api.ApiError(400, "bad_request",
                                             "since is a sequence number or RFC 3339")
            after = buf.after_time(dashboard_api.iso_utc(t.timestamp() * 1000))
        if q.get("follow", "1") in ("0", "false", "no"):
            return self._json(200, {"contract_version": dashboard_api.LOGS_CONTRACT,
                                    "lines": buf.lines(level=level, after=after, grep=grep,
                                                       limit=backlog),
                                    "last_seq": buf.seq}, extra_headers=(("Cache-Control",
                                                                          "no-store"),))
        sub = buf.subscribe(level, grep, self._reader_gone)  # before the backlog: nothing falls between
        if sub is None:
            raise dashboard_api.ApiError(429, "too_many",
                                         f"at most {logbuf.MAX_SUBSCRIBERS} log streams at once")
        ping = float(STATE.get("log_ping_s", 15.0))
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Accel-Buffering", "no")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            w = self.wfile
            last = after or 0

            def send(entries):
                nonlocal last
                for e in entries:
                    if e["seq"] > last:
                        w.write(f"event: log\nid: {e['seq']}\ndata: "
                                f"{json.dumps(e, ensure_ascii=False)}\n\n".encode())
                        last = e["seq"]

            send(buf.lines(level=level, after=after, grep=grep, limit=backlog))
            w.flush()
            quiet = time.monotonic()
            while not STATE.get("draining"):
                items, dropped = sub.take(min(ping, 1.0))
                if sub.closed or self._reader_gone():
                    break
                if dropped:
                    w.write(f"event: gap\ndata: {json.dumps({'dropped': dropped})}\n\n".encode())
                if items:
                    send(items)
                elif not dropped:
                    if time.monotonic() - quiet < ping:
                        continue
                    w.write(b": ping\n\n")
                w.flush()
                quiet = time.monotonic()
        except (BrokenPipeError, ConnectionResetError, OSError):
            self.close_connection = True
        finally:
            buf.unsubscribe(sub)

    def _dashboard_live(self, q: dict) -> None:
        """`GET /v1/dashboard/live` (contract 1.1): the requests in flight.

        `follow=0` is one JSON snapshot. Otherwise server-sent events: first the full snapshot
        with the 5-minute `history` and `recent` (a reconnect gets the same: the snapshot is the
        state, ids are not replayed), then the event the sampler encoded for each tick -- the same
        bytes for every open stream -- whenever its `seq` moved on: `QSE_LIVE_HZ` a second while
        a request is in flight, once a second otherwise. `: ping` every 15 s. Never takes the
        engine lock; a closed reader is noticed within 250 ms; at most `live.MAX_STREAMS` at once.
        """
        reg = live_registry()
        if q.get("follow", "1") in ("0", "false", "no"):
            return self._json(200, reg.snapshot(), extra_headers=(("Cache-Control", "no-store"),))
        if not reg.subscribe():
            raise dashboard_api.ApiError(429, "too_many",
                                         f"at most {live_mod.MAX_STREAMS} live streams at once")
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Accel-Buffering", "no")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            w = self.wfile
            sent, data = reg.first_event()
            w.write(data)
            w.flush()
            pinged = time.monotonic()
            while not STATE.get("draining"):
                reg.wait_event(sent, 0.25)
                ev = reg.event()
                if ev is not None and ev[0] > sent:
                    sent = ev[0]
                    w.write(ev[1])
                    w.flush()
                if time.monotonic() - pinged >= live_mod.PING_S:
                    pinged = time.monotonic()
                    w.write(b": ping\n\n")
                    w.flush()
                if self._reader_gone():
                    break
        except (BrokenPipeError, ConnectionResetError, OSError):
            self.close_connection = True
        finally:
            reg.unsubscribe()

    def _reader_gone(self) -> bool:
        """The client closed its end: the socket reads as end-of-file. A stream's reader sends
        nothing after its request, so a readable socket here is the close (or a reset)."""
        try:
            ready, _, _ = select.select([self.connection], [], [], 0)
            return bool(ready) and not self.connection.recv(1, socket.MSG_PEEK)
        except (OSError, ValueError):
            return True

    def _client_gone(self) -> bool:
        """`_reader_gone` for a completion: the request body is read, so the client sends nothing
        more, and a socket that reads as end-of-file is a client that left. No socket (a test's
        handler) is a client that stays."""
        if getattr(self, "connection", None) is None:
            return False
        return self._reader_gone()

    def _par_admit(self, pool, cid: str, prompt, rec=None, say=None, max_new: int = 0,
                   stream: bool = False, rows: int | None = None) -> bool:
        """Parallel requests: admission by bytes (server/slots.py `Admission`). The request's own
        worst case -- its prefill's workspace (engine/cache.py `Budget.workspace`) and, with the
        shared KV pool (SRV-307), the KV pages its prompt and `max_new` may still commit on its
        slot -- is reserved while it runs; with other requests in flight it waits, in one-second
        slices with the GPU turn given up, while MemAvailable less the engine's growth
        (`_par_growth`), the other requests' reservations and its own would fall under the floor.
        False: refused, the response is sent.

        SRV-250 (FX7, 2026-10-07): every request was counted as 2.1 GiB (the workspace's 1.15 GiB
        floor plus a whole state-store entry, 1 GiB, whatever its length) against 6.4 to 8.4 GiB of
        cache growth: with MemAvailable at the box's idle 27 GiB, a 576-token title request waited
        63 and 83 s beside a 9,000-token one, and two requests rarely ran at once. The store's
        snapshot asks the engine's memory policy first (`Budget.admit`), inside its ceiling, so it
        is in the growth the policy allows, not in this request's need. The wait is the queue's
        (`rec.t_lock` moves to the admission, `wait_mem_ms` books it) and `say(short_gib, place)`
        gives the client its waiting line.

        SRV-305 (2026-10-10): a wait for memory is never a silent refusal. The request waits in
        state `waiting_memory` (the dashboard counts it, `rec.wait_reason` is "memory"), its line
        says so every `MEM_LINE_EVERY_S`, a streamed request waits as long as a request may run
        (`--request-timeout`), and a non-streamed one gets, after `--queue-timeout`, a 503 with
        `Retry-After` and the code `memory_wait_timeout`.

        SRV-629: `rows`, the prompt's rows once its images are expanded (`_mm_rows`), where the
        text prompt has one placeholder an image: the workspace and the KV pages are reserved for
        what the prefill writes."""
        self._mem_wait = False
        adm = pool.admission
        if adm is None:
            return True
        n = max(int(prompt.numel()), int(rows or 0))
        mem = dict.get(STATE, "mem")
        ws = mem.workspace(n) if mem is not None else max(
            cache.WORKSPACE_FLOOR, n * cache.WORKSPACE_PER_TOKEN)
        kv, owner = 0, None
        if kvpool.POOL is not None:
            eng = STATE.get("engine")
            pages = getattr(getattr(eng, "kv", None), "pages", None)
            if pages is not None:
                # the served prompt carries the router's notes and the anchors' replay beyond the
                # client's tokens: PLAN_MARGIN rows of room for them
                rows = min(int(eng.kv.max_len), n + int(max_new) + PLAN_MARGIN + SIDE_MARGIN)
                kv = pages.need_bytes(rows)
                owner = pages.owner
        need = ws + kv
        ok, d = adm.reserve(cid, need, kv=kv, owner=owner)
        if not ok and _kv_reclaim_idle(d, owner):
            # SRV-307: the pages idle conversations keep come back before anyone waits
            ok, d = adm.reserve(cid, need, kv=kv, owner=owner)
        if ok:
            return True
        print(f"[parallel] {cid} waits for memory: {slots_mod.Admission.describe(d)}", flush=True)
        adm.stats["waits"] += 1
        self._mem_wait = True
        if rec is not None:
            rec.wait_reason = "memory"
            rec.waiting_memory = True
        t0 = time.perf_counter()
        i = _par_index()
        limit = (float(STATE.get("request_timeout") or 3600.0) if stream
                 else float(STATE.get("queue_timeout", 120.0)))
        until = time.monotonic() + limit
        said = -1e9
        try:
            while time.monotonic() < until:
                if self._client_gone():
                    raise ClientGone("client left while waiting for memory")
                place = adm.wait(cid)
                if say is not None and time.monotonic() - said >= MEM_LINE_EVERY_S:
                    said = time.monotonic()
                    short = max(0.0, (d.get("floor", 0) - (d.get("after") or 0)) / slots_mod.GiB)
                    try:
                        try:
                            say(short, place)
                        except TypeError:
                            say(short)                     # a one-argument line (the tests')
                    except Exception:                              # noqa: BLE001
                        logbuf.print_exc()
                held = pool.turn is not None and pool.turn.held_by(i)
                if held:
                    pool.turn.release(i)
                try:
                    time.sleep(1.0)
                finally:
                    if held:
                        pool.turn.acquire(i)
                # the first in line only: arrival order among the requests that wait for memory
                if place > 1:
                    continue
                ok, d = adm.reserve(cid, need, kv=kv, owner=owner)
                if not ok and _kv_reclaim_idle(d, owner):
                    ok, d = adm.reserve(cid, need, kv=kv, owner=owner)
                if ok:
                    waited = time.perf_counter() - t0
                    if rec is not None:
                        rec.wait_mem_ms = round(waited * 1e3, 1)
                        if rec.t_lock is not None:
                            rec.t_lock += waited            # the wait is the queue's, not prefill
                    print(f"[parallel] {cid} admitted after {waited:.1f} s: "
                          f"{slots_mod.Admission.describe(d)}", flush=True)
                    return True
        finally:
            adm.unwait(cid)
            if rec is not None:
                rec.waiting_memory = False
        adm.stats["refused"] += 1
        adm.stats["timeouts"] += 1
        running = pool.running()
        print(f"[parallel] {cid} REFUSED after {limit:.0f} s waiting for memory: "
              f"{slots_mod.Admission.describe(d)}", flush=True)
        if rec is not None:
            rec.stop_detail = "memory_wait_timeout"
        self._send_bytes(503, json.dumps({"error": {
            "message": (f"memory is almost full; {running} request(s) are running; "
                        f"try again in about 30 s"),
            "type": "server_busy", "code": "memory_wait_timeout"}}).encode(),
            extra_headers=(("Retry-After", "30"),))
        return False

    def _prefill_watch(self, stream: bool, rec=None):
        """`on_prefill` for this request. After every prefill chunk: raise `ClientGone` if
        the client left -- the lock is released at once and the rows prefilled so far stay good
        for its retry -- and, once the prefill has run `--prefill-heartbeat-s`, send a
        streamed client an SSE comment at most once a second. A comment is not an event: no
        client reads it as a token, and no first-token clock starts on it."""
        hb = float(STATE.get("prefill_heartbeat") or 0.0)
        t0 = time.perf_counter()
        last = [t0]

        def watch(done: int, total: int) -> None:
            if self._client_gone():
                if rec is not None and rec.client_gone_at is None:
                    rec.client_gone_at = time.perf_counter()     # seen by the handler first
                raise ClientGone(f"client left during the prefill at {done}/{total}")
            now = time.perf_counter()
            if rec is not None:
                # the prefill's progress for the live view, one tuple a chunk (a chunk is
                # a forward of hundreds to thousands of rows); no sync, so `done` is what the host
                # has issued. The marks stay for the whole prefill, at most prompt/chunk of them:
                # the live rate leaves out the first chunk and reads the last ~2 s after it
                activity_mod.pf_mark(rec.pf_marks, done, total, now, watch.info.get("window"))
            if stream and hb > 0 and now - t0 >= hb and now - last[0] >= HEARTBEAT_EVERY_S:
                last[0] = now
                self.wfile.write(f": prefill {done}/{total}\n\n".encode())
                self.wfile.flush()
            if watch.say is not None:
                watch.say(done, total, now)     # S1: the prefill's progress line, when it is due
        watch.info = {}                  # the prefill fills it: start, chunk, t0, kind
        watch.say = None
        return watch

    # -------------------------------------------------------------- routes
    def _strip_session(self) -> bool:
        """`/s/<token>/v1/library...` and `/s/<token>/v1/laya...`, the addresses the
        runtime notes give the model (server/session_tokens.py): the prefix off, the session the
        token names on the handler (`_sess`, None for an unknown token). True when the request is
        answered here (a session address of any other route is a 404)."""
        self._sess = self._stok = None
        path, tok = session_tokens_mod.strip(self.path)
        if tok is None:
            return False
        p = path.split("?")[0].rstrip("/")
        if not (p in ("/v1/library", "/v1/laya", "/v1/memory") or p.startswith("/v1/library/")
                or p.startswith("/v1/laya/") or p.startswith("/v1/memory/")):
            self._json(404, {"error": {"message": f"no route {p} under a session address",
                                       "type": "not_found"}})
            return True
        self.path, self._stok = path, tok
        reg = STATE.get("session_tokens")
        self._sess = reg.key_for(tok) if reg is not None else None
        return False

    def do_GET(self):
        self._tandem = None
        if self._strip_session():
            return
        if self._chat("GET"):
            return
        raw = self.path.split("?")[0]
        if raw == "/dashboard" or raw.startswith("/dashboard/"):
            # the dashboard's static shell: public, no data in it
            return static.serve(self, self.path, STATE.get("dashboard_dir"))
        path = raw.rstrip("/") or "/"
        if path.startswith("/v1/dashboard/"):
            return self._dashboard_get(path)
        if path.startswith("/v1/conductor/"):
            return self._conductor("GET")
        if path.startswith("/v1/setup/"):
            # the setup guide (server/setup_api.py, docs/setup.md): imported on its first call
            from server import setup_api
            return setup_api.handle(self, sys.modules[__name__], "GET", path, None)
        if path == "/v1/config" or path.startswith("/v1/config/"):
            return self._config("GET", path, None)
        if (path == "/v1/library" or path.startswith("/v1/library/")) and \
                STATE.get("conductor") is not None:
            # R1: the package library for the model's own scripts, open like the chat API
            # (server/library_api.py)
            return library_api.handle_get(self, STATE["conductor"])
        if (path == "/v1/memory" or path.startswith("/v1/memory/")) and \
                STATE.get("conductor") is not None:
            # 2026-10-05: the model's own memory, under its session's address only
            # (server/memory_api.py)
            return memory_api.handle(self, STATE["conductor"], "GET", None)
        if path in ("/health", "/healthz", "/v1/health"):
            # What a watchdog needs to decide whether to restart, and what a person needs to
            # decide whether it is wedged or merely busy. `draining` is the difference between
            # "not taking work" and "broken", and a restarter that cannot tell them apart will
            # kill a server in the middle of a graceful shutdown.
            code = 503 if STATE.get("draining") else 200
            if _auth().decide("health_full", self._peer(), self.headers) != "ok":
                # through the proxy, the status and nothing about the configuration (a page there
                # reaches the status responder through the same proxy, OPS-175)
                return self._json(code, {"status": "draining" if STATE.get("draining") else "ok"})
            return self._json(code, {
                "status": "draining" if STATE.get("draining") else "ok",
                "status_port": _status_port(),
                "model": served_name(), "base_model": STATE.get("model"),
                "version": STATE.get("version", ""), "weights": weights_profile(),
                "uptime_s": round(time.time() - STATE.get("started", time.time()), 1),
                "engine_busy": LOCK.locked(),
                "inflight": dict(INFLIGHT),
                "max_queue": STATE.get("max_queue"),
                "request_timeout_s": STATE.get("request_timeout"),
                "max_len": STATE.get("max_len"),
                "sampling": {"temperature": STATE.get("temperature", 0.0),
                             "top_p": STATE.get("top_p", 1.0),
                             "top_k": STATE.get("top_k", 0),
                             # SRV-241: what a regenerate samples with, and how many there were
                             "regenerate": (STATE["regenerate"].report()
                                            if STATE.get("regenerate") is not None
                                            else {"mode": "off"}),
                             # SRV-475: the checkpoint's sampling and the loop escape's counts
                             "card": STATE.get("card_sampling"),
                             "loop_escape": (STATE["escapes"].report()
                                             if STATE.get("escapes") is not None else None),
                             # SRV-477: `--reasoning-effort none|off` turns thinking off
                             "thinking_default": STATE.get("thinking_default", "on")},
                "penalty": {"rep": STATE["pen_spec"].rep,
                            "presence": STATE["pen_spec"].presence,
                            "freq": STATE["pen_spec"].freq,
                            "no_repeat": STATE["pen_spec"].no_repeat},
                "default_max_tokens": STATE.get("default_max_tokens"),
                "cache": cache_stats(),
                "memory": _memory(),
                # SRV-307: the shared KV pool's committed pages by slot (absent without the pool)
                **({"kv_pool": kv_pool_report()} if kvpool.POOL is not None else {}),
                # the small models and the memory governor (server/rolemem.py)
                "roles": _safe(roles_report),
                **({"price_table": STATE["price_table"]} if STATE.get("price_table") else {}),
                # `--parallel N`: the slots, the GPU turn, the joint rounds and the admission
                **({"parallel": _par_health()}
                   if dict.get(STATE, "pool") is not None else {}),
            })
        if path == "/metrics/up":
            # a public page with one number and nothing else, so the scrape can tell an
            # engine that is down from a metrics token that is wrong (the real page's 401).
            raw = (b"# HELP qse_up 1 while the server takes work, 0 while it drains\n"
                   b"# TYPE qse_up gauge\nqse_up " + (b"0" if STATE.get("draining") else b"1")
                   + b"\n")
            try:
                self.send_response(200)
                self.send_header("Content-Type", metrics.CONTENT_TYPE)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
            except (BrokenPipeError, ConnectionResetError):
                self.close_connection = True
            return
        if path == "/metrics":
            # Prometheus, in its own text format. Everything it reports is collected in
            # server/metrics.py, which wraps this module rather than editing it; see its header.
            if not self._allowed("metrics"):
                return
            return metrics.serve(self)
        if path == "/v1/cache/stats":
            if not self._allowed("cache_read"):
                return
            return self._json(200, cache_stats())
        if path == "/v1/models":
            # with the conductor on, `TandemLLM` alone: the raw engine's name is a hidden alias
            cond = STATE.get("conductor")
            out = {"object": "list", "data": (
                cond.models(STATE["model"], STATE["started"]) if cond is not None else
                [{"id": STATE["model"], "object": "model", "created": STATE["started"],
                  "owned_by": "local"}])}
            if STATE.get("proxy") is not None:
                # the backends' real models after the engine's own (proxy/route.py)
                if not STATE["proxy"].authorize(self)[0]:
                    return
                out["data"] = out["data"] + STATE["proxy"].models(STATE["started"])
            # SRV-347: the engine model's capability card (server/model_card.py)
            out["data"] = model_card.annotate(out["data"], _card_facts())
            if STATE.get("deciders") is not None:
                # beside `data`, so an OpenAI client still sees one chat model (SRV-131)
                out["decision_models"] = decider_api.decision_models(STATE["deciders"],
                                                                     STATE["started"])
            return self._json(200, out)
        if path == "/v1/laya" and STATE.get("deciders") is not None:
            # L1: Laya for the model's scripts, what the route takes (server/laya_api.py)
            return laya_api.handle_get(self, STATE["deciders"], STATE)
        if path.startswith("/v1/laya/kit/") and STATE.get("deciders") is not None:
            # X1: the playtest kit the model copies into a project
            return laya_api.handle_kit(self, STATE, path[len("/v1/laya/kit/"):])
        if path == "/readyz" and STATE.get("deciders") is not None:
            # for Jev clients: ready once a decision model is loaded and warm (SRV-131)
            code, out = decider_api.readyz(STATE["deciders"], STATE)
            return self._json(code, out)
        if path.startswith("/v1/models/"):
            # the key the list asks for, before any lookup: a 404-vs-401 difference would
            # name the models to a stranger (proxy/app.py does the same)
            if STATE.get("proxy") is not None and not STATE["proxy"].authorize(self)[0]:
                return
            from urllib.parse import unquote
            name = unquote(path[len("/v1/models/"):])
            cond = STATE.get("conductor")
            if cond is not None:
                card = cond.model_card(name, STATE["model"], STATE["started"])
            else:
                card = ({"id": STATE["model"], "object": "model", "created": STATE["started"],
                         "owned_by": "local"} if name == STATE["model"] else None)
            if card is not None:
                card = model_card.annotate([card], _card_facts())[0]     # SRV-347
            if card is None and STATE.get("deciders") is not None:
                card = decider_api.card(STATE["deciders"], name, STATE["started"])
            if card is None and STATE.get("proxy") is not None:
                card = STATE["proxy"].table.card(name, STATE["started"])
            if card is None:
                return self._json(404, {"error": {"message": f"no model {name!r}",
                                                  "type": "invalid_request_error",
                                                  "param": "model", "code": "model_not_found"}})
            return self._json(200, card)
        return self._json(404, {"error": {"message": f"no route {path}", "type": "not_found"}})

    def do_OPTIONS(self):
        # L1: the preflight of a page's fetch to the Laya route; every other path as before
        # (BaseHTTPRequestHandler's 501)
        if self._strip_session():
            return
        path = self.path.split("?")[0].rstrip("/")
        if (path in ("/v1/laya", "/v1/laya/ask", "/v1/laya/drive/end")
                or path.startswith("/v1/laya/kit/")) and STATE.get("deciders") is not None:
            return laya_api.handle_options(self)
        self.send_error(501, f"Unsupported method ('{self.command}')")

    def do_DELETE(self):
        if self._chat("DELETE"):
            return
        path = self.path.split("?")[0].rstrip("/")
        if path == "/v1/dashboard/session":
            return self._session("DELETE")
        if path.startswith("/v1/conductor/memory/notes"):
            # 2026-10-05: a note, or a user's whole memory (server/conductor/routes.py)
            return self._conductor_write("DELETE")
        if path == "/v1/config" or path.startswith("/v1/config/"):
            return self._config("DELETE", path, None)
        return self._json(404, {"error": {"message": f"no route {path}", "type": "not_found"}})

    def do_PUT(self):
        if self._chat("PUT"):
            return
        return self._conductor_write("PUT")

    def do_PATCH(self):
        if self._chat("PATCH"):
            return
        return self._conductor_write("PATCH")       # VIS-48: a session's keep

    def _config(self, method: str, path: str, body) -> None:
        """`/v1/config/*`: the Backends and Setup screens (proxy/config_api.py); 404 without
        `--backends`."""
        px = STATE.get("proxy")
        if px is None:
            return self._json(404, {"error": {"type": "not_found",
                                              "message": "the proxy is off on this server "
                                                         "(--backends off)"}})
        from proxy import config_api
        return config_api.handle(self, px, method, path, body)

    def _conductor_write(self, method: str):
        path = self.path.split("?")[0].rstrip("/")
        if path == "/v1/config" or path.startswith("/v1/config/"):
            try:
                body = self._read()
            except Exception as exc:                              # noqa: BLE001
                return self._json(400, {"error": {"message": f"bad json: {exc}",
                                                  "type": "invalid_request_error"}})
            return self._config(method, path, body)
        if not path.startswith("/v1/conductor/"):
            return self._json(404, {"error": {"message": f"no route {path}",
                                              "type": "not_found"}})
        try:
            body = self._read()
        except Exception as exc:                                  # noqa: BLE001
            return self._json(400, {"error": {"message": f"bad json: {exc}",
                                              "type": "invalid_request_error"}})
        return self._conductor(method, body)

    def _conductor(self, method: str, body: dict | None = None) -> None:
        """`/v1/conductor/*` (server/conductor/routes.py); 404 when the server runs without it."""
        cond = STATE.get("conductor")
        if cond is None:
            return self._json(404, {"error": {"type": "not_found",
                                              "message": "the conductor is off on this server "
                                                         "(--conductor-config off)"}})
        return conductor_routes.handle(self, cond, method, body, _auth())

    def do_POST(self):
        self._tandem = None
        if self._strip_session():
            return
        if self._chat("POST"):                      # before the JSON read: uploads are multipart
            return
        path = self.path.split("?")[0].rstrip("/")
        try:
            body = self._read()
        except Exception as exc:
            if path in ("/v1/chat/completions", "/v1/completions"):
                rec = usage_mod.RequestRecord("-", "chat" if path.endswith("chat/completions")
                                              else "completions", False)
                rec.status, rec.model = 400, str(STATE.get("model", ""))
                rec.client_id, rec.client_kind = ledger_mod.client_of(self.headers)
                _account(rec)
            return self._json(400, {"error": {"message": f"bad json: {exc}",
                                              "type": "invalid_request_error"}})
        if path == "/v1/dashboard/session":
            return self._session("POST", body)
        if path == "/v1/tokenize":
            # SRV-347: exact counts with the model's own template (server/tokenize_api.py)
            return tokenize_api.handle(self, body, lambda b: build_prompt(b, count=True),
                                       lambda t: len(STATE["tok"](t)["input_ids"]))
        if path in ("/v1/cache/peek", "/v1/cache/prewarm"):
            # SRV-349: where a conversation's rows are, and the NVMe read back while idle
            if not self._allowed("cache_read"):
                return
            return self._cache_peek(body, prewarm=path.endswith("prewarm"))
        if path.startswith("/v1/conductor/"):
            return self._conductor("POST", body)
        if path.startswith("/v1/setup/"):
            from server import setup_api
            return setup_api.handle(self, sys.modules[__name__], "POST", path, body)
        if path == "/v1/config" or path.startswith("/v1/config/"):
            return self._config("POST", path, body)
        if path.startswith("/v1/collect/"):          # P6: a vLLM or SGLang plugin's batch
            px = STATE.get("proxy")
            if px is None:
                return self._json(404, {"error": {"type": "not_found",
                                                  "message": "the proxy is off on this server "
                                                             "(--backends off)"}})
            from proxy import collect_api
            return collect_api.handle(self, px, path, body)
        if path == "/v1/cache/release":
            # SRV-483: a chat's sandbox needs room; the engine gives back what it holds idle
            if not self._allowed("cache_clear"):
                return
            b = body if isinstance(body, dict) else {}
            try:
                want = int(b.get("bytes") or 0)
            except (TypeError, ValueError):
                want = -1
            if want < 0 or want > 1 << 40:
                return self._json(400, {"error": {"message": "bytes must be an integer from 0",
                                                  "type": "invalid_request_error",
                                                  "param": "bytes"}})
            return self._json(200, _memory_release(want, str(b.get("why") or "sandbox")[:40]))
        if path == "/v1/cache/clear":
            if not self._allowed("cache_clear"):
                return
            # Measuring a warm number against a cold one needs a way back to cold that is not a
            # server restart, because a restart also throws away the Triton autotuning and the
            # first row would pay for the compiler -- the trap the phase-6 table was thrown away
            # for. This clears the caches and nothing else.
            # `?tier=disk` empties the disk tier instead, `?tier=all` both (C1).
            from urllib.parse import parse_qs, urlsplit
            tier = (parse_qs(urlsplit(self.path).query).get("tier") or ["ram"])[-1]
            if tier not in ("ram", "disk", "all"):
                return self._json(400, {"error": {"message": "tier is ram, disk or all",
                                                  "type": "invalid_request_error"}})
            if tier in ("ram", "all"):
                with LOCK:
                    for name in ("state_store", "response_cache", "resident"):
                        obj = STATE.get(name)
                        if obj is not None:
                            obj.clear() if hasattr(obj, "clear") else None
                    if STATE.get("resident") is not None:
                        STATE["resident"].shed_park()  # another session's rows are a warm start
            if tier in ("disk", "all"):
                # the writer deletes the files between two of its writes; no request waits on it
                diskcache_mod.clear(STATE)
                if STATE.get("spill") is not None:
                    with LOCK:
                        STATE["spill"].clear()
            return self._json(200, cache_stats())
        if (path == "/v1/memory" or path.startswith("/v1/memory/")) and \
                STATE.get("conductor") is not None:
            # 2026-10-05: a note from the model's own script (server/memory_api.py)
            return memory_api.handle(self, STATE["conductor"], "POST", body)
        if path == "/v1/laya/ask" and STATE.get("deciders") is not None:
            # L1: Laya for the model's scripts, open like the chat API (server/laya_api.py)
            return laya_api.handle_post(self, STATE["deciders"], STATE, body)
        if path == "/v1/laya/drive/end" and STATE.get("deciders") is not None:
            # X1: a driver session's end: its journal row (server/laya_api.py DRIVER SESSIONS)
            return laya_api.handle_drive_end(self, STATE, body)
        if path in ("/v1/systemone", "/v1/predict") and STATE.get("deciders") is not None:
            # typed decisions (SRV-131): the Jev contract, answered in the idle window
            return decider_api.handle(self, STATE["deciders"], body)
        self._proxy_user = None
        px = STATE.get("proxy")
        if px is not None and path in ("/v1/chat/completions", "/v1/completions",
                                       "/v1/embeddings", "/v1/responses"):
            if self._proxied(px, path, body):
                return
        if path not in ("/v1/chat/completions", "/v1/completions"):
            return self._json(404, {"error": {"message": f"no route {path}",
                                              "type": "not_found"}})
        if not isinstance(body, dict):                      # SRV-167: `[]`, `"hi"`, `null`
            return self._json(400, compat.Refusal("body", "the request body must be a JSON "
                                                          "object").body())
        with logbuf.serving(body):
            return self._complete_logged(body, path)

    def _loop_escape(self, body: dict, plan, rec, sampler, cid: str):
        """SRV-475: this request's sampling under the profile's `sampling` section. While the
        session repeats one call (loopescape.signal) and the escape is on, the model card's
        sampling with a fresh seed and C3's escape note (`rec.loop_escape`, for SP4); with
        `agentic_default: card`, the card's sampling on every such turn. A client that names any
        sampling parameter keeps its own. Journaled in the decision row (`loop_escape`, from which
        turn, back at which turn; `sampling`) and in the usage frame (`tandem.sampling`).
        Returns the sampler the request decodes with."""
        prof = loopescape_mod.profile_sampling(plan.res.doc)
        msgs = body.get("messages") or []
        st = None
        if prof["loop_escape"] == "on":
            mon = STATE.get("monitor")
            pol = STATE["conductor"].monitor_policy(plan, mon.policy if mon else None) \
                if mon is not None else None
            c3 = pol.tasks.get("C3") if pol is not None else None
            repeats = int(c3["repeats"]) if c3 is not None else loopescape_mod.REPEATS
            wait_s = float(c3.get("wait_s", 0.0) or 0.0) if c3 is not None \
                else loopescape_mod.WAIT_S
            key = (plan.session.key if plan.session is not None
                   else regenerate_mod.session_of(body, self.headers))
            sess = mon.sessions.get(key) if mon is not None else None
            sig = loopescape_mod.signal(msgs, loopescape_mod.waits_now(sess, msgs, time.time()),
                                        repeats=repeats, wait_s=wait_s)
            st = STATE["escapes"].step(key, sig, loopescape_mod.turn_of(msgs))
        mode = ("loop_escape" if st is not None and st["state"] == "on"
                else "card" if prof["agentic_default"] == "card" else None)
        if st is None and mode is None:
            return sampler
        info = dict(st or {})
        if mode is not None:
            info["mode"] = mode
            if regenerate_mod.client_sampling(body):
                info["client_params"] = True           # the client's own sampling, as sent
            else:
                regen = STATE.get("regenerate")
                fallback = ({"temperature": regen.temperature, "top_p": regen.top_p,
                             "top_k": regen.top_k} if regen is not None else None)
                sp = loopescape_mod.sampling_of(STATE.get("card_sampling"), fallback)
                if sp is None:
                    info["no_sampling"] = True         # no card and no regenerate settings
                else:
                    sampler = Sampler(temperature=sp["temperature"], top_p=sp["top_p"],
                                      top_k=sp["top_k"], seed=sp["seed"],
                                      draft_temperature=sampler.draft_temperature)
                    rec.temperature = sampler.temperature
                    rec.sampling = mode
                    info.update(sp)
        if st is not None and st["state"] == "on":
            rec.loop_escape = info                     # SP4: C3's escape note
        rec.sampling_detail = {k: info.get(k) for k in (
            "mode", "state", "why", "name", "since_turn", "turn", "back_at_turn", "temperature",
            "top_p", "top_k", "seed", "source", "client_params") if info.get(k) is not None}
        if plan.row is not None:
            plan.row["extra"]["loop_escape" if st is not None else "sampling"] = info
        print(f"[loop-escape] {cid} " + (
            f"{st['state']} ({st['why']} {st['name']}, since turn {st['since_turn']}"
            + (f", back at turn {st['back_at_turn']})" if st["state"] == "back" else ")")
            if st is not None else "agentic default")
            + (f": temperature {sampler.temperature} top_p {sampler.top_p} top_k {sampler.top_k} "
               f"seed {sampler.seed}" if rec.sampling == mode and mode is not None
               else ": the client's sampling, as sent" if info.get("client_params") else ""),
            flush=True)
        return sampler

    def _memory_user(self, rec) -> str | None:
        """Whose memory a request reads and writes (2026-10-05): the proxy key's label for a
        request the in-process proxy let in; `X-Tandem-User` from a trusted proxy address
        (loopback, or QWEN38_MEMORY_PROXIES); None for a request a proxy relayed without one
        (nothing is read or written); else `owner`, the one person on the LAN."""
        if getattr(rec, "user", None):
            return rec.user
        hdr = (self.headers.get("X-Tandem-User") or "").strip()
        if hdr:
            peers = {"127.0.0.1", "::1"} | {p.strip() for p in _S.get("MEMORY_PROXIES").split(",")
                                             if p.strip()}
            return hdr if self._peer() in peers else None
        if self.headers.get("X-Tandem-Relay-Id"):
            return None
        return "owner"

    def _proxied(self, px, path: str, body) -> bool:
        """`--backends`: the users' keys on every completion route, then a model that routes to a
        backend goes to the relay on this thread (no engine lock, no queue). True when answered
        here; False for the engine's own model, which goes on as before."""
        from proxy import route as proxy_route
        t_arrival = time.perf_counter()
        ok, user = px.authorize(self)
        if not ok:
            px.account_refusal(self, path, body, 401, finish="refused",
                               error_type="invalid_api_key")
            return True
        self._proxy_user = user
        try:
            route = px.route(path, body)
        except proxy_route.NotFound as exc:
            px.account_refusal(self, path, body, 404, user=user, error_type="model_not_found")
            self._json(404, exc.body())
            return True
        if route is None:
            return False
        px.serve(self, path, getattr(self, "_raw", b"") or json.dumps(body).encode(), body,
                 route, user, t_arrival=t_arrival)
        return True

    def _complete_logged(self, body: dict, path: str) -> None:
        try:
            return self._complete(body, chat=path.endswith("chat/completions"))
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as exc:                                  # noqa: BLE001
            logbuf.print_exc()
            if self._early is not None and not self._early["done"] and not self._early_body:
                # a stream opened early that failed before its generation: closed, with the reason
                return self._early_end(500, json.dumps({"error": {
                    "message": str(exc), "type": "internal_error"}}).encode())
            if self._streamed:
                # The response is an event stream whose headers are long gone. A JSON error body
                # written here would be appended to it as garbage, and the client would see the
                # truncation without the reason -- which is precisely bug 2's symptom.
                return
            try:
                if isinstance(exc, grammar_mod.GrammarError):
                    # the constraint left no legal token -- the request's own doing
                    return self._json(400, compat.Refusal(compat.constraint_field(body),
                                                          str(exc)).body())
                return self._json(500, {"error": {"message": str(exc),
                                                  "type": "internal_error"}})
            except Exception:
                return

    _streamed = False

    def _complete(self, body: dict, chat: bool) -> None:
        """One completion request, and its record accounted for on every way out."""
        self._early, self._early_body, self._bar = None, False, None
        t_req = time.perf_counter()
        cid = ("chatcmpl-" if chat else "cmpl-") + uuid.uuid4().hex[:24]
        rec = usage_mod.RequestRecord(cid, "chat" if chat else "completions",
                                      bool(body.get("stream")), t_arrival=t_req)
        rec.model = str(body.get("model") or STATE.get("model", ""))
        if STATE.get("conductor") is not None:
            rec.model_served = served_name()       # from the start: the Live view shows it in flight
        rec.client_id, rec.client_kind = ledger_mod.client_of(self.headers, body)
        if rec.client_kind == "internal":
            # VIS-60: the turn this question served (server/conductor/classify.py PARENT)
            par = self.headers.get(ledger_mod.PARENT_HEADER)
            rec.parent_request = par[:80] if isinstance(par, str) and par else None
        rec.user = getattr(self, "_proxy_user", None)       # the proxy key's label (--backends)
        # a proxy in front of this engine names its own lane for the request (server/parallel.py)
        rid = self.headers.get("X-Tandem-Relay-Id")
        rec.relay_id = rid[:64] if isinstance(rid, str) and rid else None
        # what the live view links and watches -- the conversation (opencode sends its
        # session as `x-session-id`) and the socket the sampler checks for a client that left
        rec.conv = conversation_id(body, self.headers) or _session_header(self.headers)
        rec.sock = getattr(self, "connection", None)
        # the conductor's mode for this request: the server's, lowered by `X-Tandem`, and never
        # more than shadow for `n > 1`
        try:
            n_req = int(body.get("n") or 1)
        except (TypeError, ValueError):
            n_req = 1
        mon = STATE.get("monitor")
        self._tandem = (mon.mode_for(self.headers.get("X-Tandem"), n_req) if mon is not None
                        else "raw")
        self._images_left_out = 0                      # a keep-alive socket's handler is reused
        live_registry().register(rec)                  # visible from now until 30 s after
        try:
            return self._serve(body, chat, rec)
        except (BrokenPipeError, ConnectionResetError):
            raise
        except BaseException:
            if not self._streamed:
                rec.status = 500                   # `do_POST` answers it after this unwinds
            raise
        finally:
            if rec.status != 500:
                rec.status = self._status or rec.status
            if rec.finish_reason is None and rec.status in (429, 503):
                rec.finish_reason = "refused"
            rec.end()
            if rec.regen is not None and STATE.get("regenerate") is not None and \
                    regenerate_mod.answered_ok(rec.status, rec.finish_reason,
                                               rec.completion_tokens):
                STATE["regenerate"].answered(rec.regen, rec.request_id)   # SRV-241
            live_registry().finish(rec)
            if rec.sup is not None and STATE.get("monitor") is not None:
                STATE["monitor"].turn_ended(rec.sup)   # the watch's idle time starts here
            if rec.probe is not None:
                _probe_done(rec)                   # before the row is written: extra.probe
            if STATE.get("conductor") is not None:
                STATE["conductor"].finish(rec.plan, rec)   # its journal row and ledger fields
                cids = getattr(rec.plan, "compaction_ids", None)
                if cids and rec.plan.session is not None:
                    try:
                        STATE["conductor"].router.compacted_into(
                            rec.plan.session.key, STATE["tok"].decode(cids,
                                                                      skip_special_tokens=True))
                    except Exception:                      # noqa: BLE001
                        logbuf.print_exc()
            _conductor_fields(rec)
            _account(rec)

    def _serve(self, body: dict, chat: bool, rec: "usage_mod.RequestRecord") -> None:
        t_req, cid = rec.t_arrival, rec.request_id
        if STATE.get("log_request_keys"):
            print(logbuf.request_keys_line(cid, body), flush=True)
        # SRV-167 (UT1, 2026-10-06): the request's own shape first -- the messages (or the prompt)
        # and the output limit -- so a malformed request is the client's 400 naming the field,
        # never a 500 with the chat template's exception text (which clients retry)
        try:
            compat.request_shape(body, chat)
        except compat.Refusal as exc:
            return self._json(400, exc.body())
        # SRV-477: `--reasoning-effort none|off`, thinking off for a request that says nothing
        # about thinking (one that does keeps what it says)
        if chat and STATE.get("thinking_default") == "off":
            body = thinking_mod.server_default_off(body)
        # TandemLLM's state lines that a client sends back in its earlier reasoning are taken out
        # before anything reads the messages (server/conductor/states.py): Qwen never sees them,
        # and the prompt is the one the request would have without them
        stripped = 0
        if chat and STATE.get("conductor") is not None and isinstance(body.get("messages"), list):
            body["messages"], stripped = states_mod.strip_messages(body["messages"])
        # A call the loop guard (or the token limit) stopped, sent back in the history
        # (Open WebUI's tool loop replays its arguments: 2026-10-06, five byte-identical retries,
        # the prompt 17k -> 46k), is cut down to its head and one marker before anything reads the
        # messages: the model never copies its own loop again, and the router and the monitor read
        # the retry streak from the markers (server/argloop.py `collapse_history`)
        if chat and STATE.get("argloop", "on") == "on" and isinstance(body.get("messages"), list):
            body["messages"], cut = argloop_mod.collapse_history(body["messages"])
            if cut:
                rec.argloop_history = cut
                print(f"[argloop] {cid} history: {len(cut)} stopped call(s) cut down ("
                      + ", ".join(f"{c['name']} period={c['period']} {c['chars']}->"
                                  f"{c['chars'] - c['cut']} chars" for c in cut) + ")",
                      flush=True)
        # a field this engine cannot serve as asked is a 400 that names it, never a silent
        # ignore (server/compat.py holds the disposition of every OpenAI field).
        structured = bool(STATE.get("structured_outputs", True))
        try:
            compat.check(body, chat, structured=structured)
            # the client's thinking budget, whichever shape it came in (server/thinking.py)
            client = thinking_mod.read(body, _effort_caps())
            # the document a request asks for (response_format, structured_outputs); tool_choice
            # required or named is enforced with the same machinery -- the answer as calls to the
            # allowed functions, their values typed by the schemas; SRV-619: tools under
            # auto beside a document -- the calls or the document, one automaton
            ac = (compat.answer_constraint(body, chat,
                                           xml_calls=STATE.get("tool_format") != "pythonic")
                  if structured else None)
            pattern = ac.pattern if ac is not None else None
            rec.constrained = ac.kind if ac is not None else None
            tool_types = ac.types if ac is not None else None
            # compiled before the lock -- a bad constraint is the client's 400, not a
            # queue slot -- and cached, so a client that sends one schema pays for it once
            gram = (grammar_mod.grammar_for(pattern, _grammar_vocab(calls=ac.calls),
                                            parts=ac.parts)
                    if pattern else None)
        except compat.Refusal as exc:
            return self._json(400, exc.body())
        except grammar_mod.GrammarError as exc:
            return self._json(400, compat.Refusal(compat.constraint_field(body), str(exc)).body())
        # a request's images are fetched, decoded and preprocessed here, before it
        # queues -- host work, and a bad image is the client's 400, not a queue slot
        images: list = []
        left_out = 0
        if chat:
            try:
                images = images_mod.collect(
                    body, STATE.get("image_limits") or images_mod.Limits(), STATE.get("image_pre"),
                    act_budget=int(STATE.get("image_act_budget") or 0),
                    act_bytes=(STATE["vision"].activation_bytes if STATE.get("vision") else None))
            except images_mod.ImageRefusal as exc:
                return self._json(400, exc.body())
            rec.images = len(images)
            left_out = getattr(images, "left_out", 0)
            if left_out:
                # a long conversation's oldest images, left out instead of a 400 that every later
                # turn would get again (server/images.py): counted, logged, in the response header
                # and the journal row; never a URL
                lim = STATE.get("image_limits") or images_mod.Limits()
                rec.images_left_out = self._images_left_out = left_out
                metrics.images_left_out_total.inc(float(left_out))
                print(f"[images] {cid}: left out the oldest {left_out} of {images.total} images "
                      f"(the caps: {lim.max_images} images, {lim.max_image_rows} rows a request)",
                      flush=True)
        # Sampling. Greedy is the exact path and stays the default; a request that asks
        # for sampling gets real sampling from engine/sample.py, on the single-token path (no
        # drafter) until rejection sampling lands. Seeded requests reproduce exactly.
        try:
            sampler = Sampler(
                temperature=float(body["temperature"]) if body.get("temperature") is not None
                else float(STATE.get("temperature", 0.0)),
                top_p=float(body["top_p"]) if body.get("top_p") is not None
                else float(STATE.get("top_p", 1.0)),
                top_k=int(body["top_k"]) if body.get("top_k") is not None
                else int(STATE.get("top_k", 0)),
                seed=body.get("seed"),
                draft_temperature=body.get("draft_temperature"),
                min_p=float(body["min_p"]) if body.get("min_p") is not None else 0.0)
        except (TypeError, ValueError) as exc:
            return self._json(400, {"error": {"message": f"bad sampling parameter: {exc}",
                                              "type": "invalid_request_error",
                                              "param": "temperature"}})
        if sampler.temperature > 2.0:
            return self._json(400, {"error": {
                "message": f"temperature must be in [0, 2], got {sampler.temperature}",
                "type": "invalid_request_error", "param": "temperature"}})
        rec.temperature = sampler.temperature          # shown on the live row
        # BUG 2, the first half. This used to default to 256 tokens, and a client that does not
        # send `max_tokens` -- Open WebUI does not -- got an answer that stopped in the middle of
        # a sentence with `finish_reason: "length"` and no other sign that anything had happened.
        # Two hundred and fifty-six tokens is about forty lines of code, which is exactly where
        # the Mario page stopped. A serving default belongs at the context length, not at a
        # number small enough to truncate an ordinary answer.
        max_new = int(body.get("max_tokens") or body.get("max_completion_tokens")
                      or STATE.get("default_max_tokens") or 8192)
        server_budget = int(STATE.get("think_budget") or 0)
        budget = client.budget if client.asked else server_budget

        def routed(b: int) -> int:
            """The budget under the plan: where the router decides the effort, a client's effort
            word is a ceiling (the owner, 2026-10-03 12:30): the router's level, never above the
            word's cap (0 is no cap); under a profile whose `thinking.client_word` is `ignore`
            (opencode, which sends `medium` every turn) the router's level alone. A client's
            number always wins (server/thinking.py)."""
            if client.asked and client.kind != "number" and plan.decides_effort:
                rb = plan.think_budget(server_budget)
                if plan.client_word == "ignore":
                    return rb
                cw = int(client.budget or 0)
                out = rb if not cw else cw if not rb else min(rb, cw)
                if out != rb and plan.row is not None:
                    plan.row["extra"]["client_word"] = {"word": client.effort, "cap": cw,
                                                        "router": rb, "budget": out}
                return out
            return plan.think_budget(b)
        # the budget and the stall close work on BOTH verify paths now -- the forced close
        # runs a chain-shaped forward of the closing phrase, which the tree path writes like any
        # other block. `think` is constructed even without a budget, because the stall
        # detector is the signal that replaces a cut.
        stream = bool(body.get("stream"))
        try:
            fmt = str(body.get("reasoning_format") or STATE.get("reasoning_format") or "tags")
            # SRV-169 (UT1, 2026-10-06): a JSON response format is parsed whole by the client
            # (`json.loads(message.content)`, every SDK's structured-output helper), so without
            # the client's own `reasoning_format` the thinking goes to `reasoning_content` and the
            # content is the JSON alone; a client that asks for `tags` keeps them
            json_answer = rec.constrained == "response_format" and compat.json_answer(body)
            fmt = compat.answer_format(body, fmt, json_answer)
            Reasoning(fmt)                       # validate before anything is generated
        except ValueError as exc:
            return self._json(400, {"error": {"message": str(exc),
                                              "type": "invalid_request_error",
                                              "param": "reasoning_format"}})
        # `body`, `finish` (the default, which is what Open WebUI's base models get),
        # `separate` (the client asked with include_usage) or `none` -- one place, never two.
        where = usage_mod.placement(body, stream, bool(STATE.get("usage_default", True)))
        # Anti-repetition penalties. Deterministic on the target's logits, so greedy and
        # speculative decoding stay identical under the rule -- see engine/penalty.py. The
        # per-request names are the OpenAI ones plus the HF one; defaults come from the server
        # flags. NOTE: clients that already send presence_penalty (Open WebUI's roster rows do)
        # change from ignored to honoured the day this ships -- that is the fix, and it is said
        # out loud in docs/server.md.
        base: PenaltySpec = STATE["pen_spec"]

        def _p(name: str, default: float) -> float:
            v = body.get(name)
            return float(v) if v is not None else default

        try:
            pen_spec = PenaltySpec(rep=_p("repetition_penalty", base.rep),
                                   presence=_p("presence_penalty", base.presence),
                                   freq=_p("frequency_penalty", base.freq),
                                   no_repeat=int(_p("no_repeat_ngram_size", base.no_repeat)),
                                   bias=compat.logit_bias(body, STATE["engine"].cfg.vocab_size))
        except compat.Refusal as exc:
            return self._json(400, exc.body())
        except (TypeError, ValueError) as exc:
            return self._json(400, {"error": {"message": f"bad penalty parameter: {exc}",
                                              "type": "invalid_request_error",
                                              "param": "repetition_penalty"}})
        pen = (PenaltyState(pen_spec, STATE["engine"].cfg.vocab_size, STATE.get("device", "cuda"))
               if pen_spec.on else None)
        pstop = PatternStop(*STATE["pattern_stop"]) if STATE.get("pattern_stop") else None
        stops = body.get("stop") or []
        if isinstance(stops, str):
            stops = [stops]
        # the name the response carries: the one model with the conductor configured, whatever the
        # client asked for (the alias included); without it, the request's name as before
        model = served_name() if STATE.get("conductor") is not None else (body.get("model") or
                                                                           STATE["model"])
        created = int(time.time())
        tok = STATE["tok"]

        # the caches' session key: the body's conversation, else opencode's x-session-id (OPS-47)
        conv_id = conversation_id(body, self.headers) or _session_header(self.headers)
        # ENG-371: an agent's worker (its session comes back at its next call) and its parent
        stay, parent = _worker_of(self.headers)
        if stay:
            _note_parent(parent, worker=conv_id)
        # SRV-349: the prompt's system and tool tokens for `usage.tandem`, rendered here on the
        # host (before the queue), once per head
        head_parts = _head_parts(body, client) if chat else None
        # The conductor (server/conductor/): the request's profile, and for `TandemLLM` the
        # turn's class and what that class gets. Here, before the queue: it may ask arbiter or
        # this server's own raw model, and must never do that holding the engine lock. A raw
        # request gets an inactive plan whose every step hands its input back unchanged.
        plan = None
        if STATE.get("conductor") is not None:
            # 0.8.2: the Lanes view's queued line while the router's questions run (the wait
            # loop sets its own reason after them)
            rec.queue_why = OWN_WAIT["question"]
            try:
                # the prompt's length is rendered only if the in-context stage asks for it
                plan = rec.plan = STATE["conductor"].plan(
                    body, self.headers, client_kind=rec.client_kind, request_id=cid, chat=chat,
                    count_tokens=lambda: build_prompt(body, client, count=True,
                                                       left_out=left_out),
                    user=self._memory_user(rec))
            except ConductorConfigError as exc:
                return self._json(400, {"error": {"message": str(exc),
                                                  "type": "invalid_request_error",
                                                  "param": "X-Tandem-Profile"}})
            except Exception:                                      # noqa: BLE001
                logbuf.print_exc()             # a conductor bug serves the request as raw
                plan = rec.plan = None
        # SRV-241: a regenerate (server/regenerate.py) samples with a fresh seed; every other
        # request keeps the sampler it had (greedy unless the client asked). Under an active plan
        # only: raw requests (benchmarks) and side requests are never looked at.
        regen = STATE.get("regenerate")
        # SRV-621: under `prompt: client` the request keeps its sampler too
        if (regen is not None and chat and plan is not None and plan.active
                and plan.side is None and not getattr(plan, "compaction", False)
                and not getattr(plan, "prompt_client", False)):
            try:
                rd = rec.regen = regen.decide(body, self.headers)
            except Exception:                                      # noqa: BLE001
                logbuf.print_exc()             # a bug here serves the request as sent
                rd = rec.regen = None
            if rd is not None and rd.how is not None:
                info = {"how": rd.how, "of": rd.of}
                if rd.sample:
                    sp = regen.sampling()
                    sampler = Sampler(temperature=sp["temperature"], top_p=sp["top_p"],
                                      top_k=sp["top_k"], seed=sp["seed"],
                                      draft_temperature=sampler.draft_temperature)
                    rec.temperature = sampler.temperature
                    rec.sampling = "regenerate"
                    info.update(sp)
                else:
                    info["client_params"] = True     # the client's own sampling, as sent
                if plan.row is not None:
                    plan.row["extra"]["regenerate"] = info
                print(f"[regenerate] {cid} {rd.how} (of {rd.of}): "
                      + (f"temperature {sampler.temperature} top_p {sampler.top_p} "
                         f"top_k {sampler.top_k} seed {sampler.seed}" if rd.sample
                         else "the client's sampling, as sent"), flush=True)
        # SRV-475: the loop escape and the profile's `sampling.agentic_default`
        # (server/conductor/loopescape.py), on a turn that offers tools under an active plan;
        # raw, bench, side requests, compactions and a regenerate are never looked at
        if (STATE.get("escapes") is not None and chat and plan is not None and plan.active
                and plan.side is None and not getattr(plan, "compaction", False)
                and body.get("tools") and rec.sampling is None):
            try:
                sampler = self._loop_escape(body, plan, rec, sampler, cid)
            except Exception:                                      # noqa: BLE001
                logbuf.print_exc()             # a bug here serves the request as sent
        if rec.sampling is None and sampler.on:
            rec.sampling = "client"
        # SRV-193: a client kind's own format (Open WebUI: reasoning_content) when it names none,
        # under a conductor plan; its reasoning ends with the template's blank lines as content,
        # so each round's block is timed. SRV-617: on the raw path too for a plain API caller
        # (SDK, curl, an unnamed client such as a benchmark), which reads `message.content` as
        # the answer; every other kind on the raw path keeps the release's shape, byte for byte
        fmt_boundary = ""
        # SRV-245: stop strings read the answer alone under a plan; SRV-617: and for a plain
        # client on the raw path whose reasoning moved out of `content`
        answer_side = plan is not None and plan.active
        if not json_answer and (answer_side or rec.client_kind in compat.PLAIN_CLIENTS):
            by_client = compat.answer_format(body, fmt, False, rec.client_kind,
                                             STATE.get("client_reasoning"))
            if by_client != fmt:
                # the blank-line boundary is Open WebUI's (it times a round's reasoning at the
                # first content); an API client's answer starts with its first character (SRV-245)
                fmt, fmt_boundary = by_client, ("\n\n" if rec.client_kind == "open-webui"
                                                else "")
                answer_side = True
                if plan is not None and plan.row is not None:
                    plan.row["extra"]["reasoning_format"] = fmt
        if STATE.get("conductor") is not None:
            if plan is not None:
                budget = routed(budget)
                if stripped and plan.row is not None:
                    plan.row["extra"]["states_stripped"] = stripped
                if left_out and plan.row is not None:
                    plan.row["extra"]["images_left_out"] = left_out
                if (plan.active and plan.row is not None and body.get("no_repeat_ngram_size")
                        and not getattr(plan, "prompt_client", False)
                        and pen_spec.no_repeat != base.no_repeat
                        and (plan.row.get("class_id") in conductor_schema.CODE_CLASSES
                             or bool(body.get("tools")))):
                    # The client's no-repeat n-gram mask is not for code: every
                    # repeated n-gram pays 4 nats, and in code (and in a tool call's arguments)
                    # repeated n-grams are the text itself. Dropped on the code classes' turns and
                    # on turns that offer tools; the server's own default stays. Journaled and
                    # logged; the raw path honours it as before.
                    plan.row["extra"]["penalty_dropped"] = {
                        "no_repeat_ngram_size": int(pen_spec.no_repeat),
                        "why": "code" if plan.row.get("class_id") in conductor_schema.CODE_CLASSES
                        else "tools"}
                    print(f"[penalty] {cid} no_repeat_ngram_size={pen_spec.no_repeat} dropped "
                          f"(class {plan.row.get('class_id')}, tools "
                          f"{'yes' if body.get('tools') else 'no'})", flush=True)
                    pen_spec = PenaltySpec(rep=pen_spec.rep, presence=pen_spec.presence,
                                           freq=pen_spec.freq, no_repeat=base.no_repeat,
                                           bias=pen_spec.bias)
                    pen = (PenaltyState(pen_spec, STATE["engine"].cfg.vocab_size,
                                        STATE.get("device", "cuda")) if pen_spec.on else None)
                if plan.side is not None:
                    # S1: a client's side request: its output capped by its shape
                    max_new = min(max_new, int(plan.side["max_tokens"]))
                if (plan.active and rec.client_kind == "opencode"
                        and STATE.get("skill_canon") is not None):
                    # Opencode's synced skills as their first-seen copies, so a new
                    # session's head is the last one's (server/skillcanon.py)
                    body, n_canon = STATE["skill_canon"].apply(body)
                    if n_canon and plan.row is not None:
                        plan.row["extra"]["skills_canonical"] = n_canon
        if (plan is not None and plan.active and plan.side is None
                and not getattr(plan, "compaction", False)
                and not getattr(plan, "prompt_client", False)
                and STATE.get("prepared") is not None):
            STATE["prepared"].seen(conv_id, rec.client_kind)   # the preparer's client kinds
        # SRV-137: keep a reference to the output for the session transcript (never on raw)
        transcribe = plan is not None and plan.active and plan.transcript == "full"
        # The session monitor (server/conductor/monitor/) runs under the profile: never above its
        # mode (the raw model name runs it raw), with its task modes, caps and thresholds. With a
        # profile resolved, X-Tandem says the profile's mode, the router's as much as the monitor's.
        mon_mode, mon_policy = self._tandem, None
        if plan is not None:
            mon_mode = STATE["conductor"].monitor_mode(plan, self._tandem)
            mon = STATE.get("monitor")
            mon_policy = STATE["conductor"].monitor_policy(plan, mon.policy if mon else None)
            self._tandem = "raw" if plan.res.mode in ("raw", "off") else plan.res.mode
        # choices, log-probabilities. /14: which answers are read for calls -- every
        # chat answer unless `tool_choice` is none -- the request's tool names for the JSON gate,
        # and `parallel_tool_calls: false` as a cap of one.
        n_choices = int(body.get("n") or 1)
        lp_top = compat.top_logprobs(body, chat)
        tool_mode = compat.tool_choice(body)[0] if chat else "none"
        parse_calls = chat and tool_mode != "none"
        tool_names = compat.tool_names(body) if parse_calls else []
        # SRV-247 (UX2, 2026-10-07): a call the model writes in an answer to a request that offered
        # no tools is read out of the text and dropped, never handed over: a curl client with
        # memory on received the model's own `tool` call (a GET on the memory search, with the
        # session's share address) and finish `tool_calls`. Under a conductor plan only: the raw
        # path is the release's, byte for byte
        refuse_calls = bool(parse_calls and not body.get("tools") and plan is not None
                            and plan.active)
        max_calls = 1 if body.get("parallel_tool_calls") is False else None
        # a value the model writes as text goes back as the JSON type its schema asks for
        # (`"offset": 150`, not `"150"`) -- clients validate the arguments against the schema and
        # refuse the string. the constrained parameters keep their JSON-literal reading.
        if parse_calls and body.get("tools"):
            typed = schema_types(body.get("tools"))
            for fname, keys in (tool_types or {}).items():
                typed.setdefault(fname, {}).update({k: JSON_ANY for k in keys})
            tool_types = typed or None

        # S1: a client's side request goes ahead of a waiting turn when it is short, and after
        # every waiting turn when it is not (the module's WAITERS)
        rank = RANK_TURN
        if plan is not None and plan.side is not None:
            try:
                short = (int(build_prompt(body, client, count=True, left_out=left_out))
                         <= SIDE_FIRST_MAX_TOKENS)
            except Exception:                                      # noqa: BLE001
                short = True                   # the render fails again under the lock, as a 400
            rank = RANK_SIDE if short else RANK_SIDE_LONG
            plan.side["first"] = short
        # The side lane (server/preempt.py). A short raw request (the conductor's own
        # questions to Qwen) or a client's side request (S1), thinking off, does not wait for a
        # turn to end: a holder serves it at its next safe point. Decided here, before the lock,
        # from the prompt's length and whether it ends inside `<think>` (host work only).
        lane_kind = None
        lane = STATE.get("lane")
        if (lane is not None and lane.on and chat and n_choices == 1 and not images
                and not STATE.get("draining")):
            side_on = (plan is not None and plan.side is not None and plan.res.mode == "on")
            raw = plan is None or not plan.active or plan.res.mode == "raw"
            if side_on or (raw and (plan is None or plan.side is None)):
                try:
                    n_lane, think_lane = build_prompt(body, client, count="think",
                                                      left_out=left_out)
                    if think_lane and side_on and plan.thinking_off:
                        think_lane = False          # the plan closes the open block
                    if not think_lane and lane.eligible(n_lane, max_new):
                        lane_kind = "side" if side_on else "raw"
                except Exception:                                  # noqa: BLE001
                    lane_kind = None             # served the usual way, its 400 under the lock
        # the profile's `ui.states`: TandemLLM's steps as lines the client shows (one choice, a
        # profile that acts, never a side request); the router's go out once the plan is applied,
        # the monitor's as they happen, the progress lines (S1) while the request waits, prefills
        # and thinks
        ui = None
        ui_doc = (plan.res.doc.get("ui") or {}) if plan is not None else {}
        if (plan is not None and plan.active and plan.res.mode == "on" and n_choices == 1
                and ui_doc.get("states") == "on" and plan.side is None
                and not json_answer):             # SRV-169: a JSON answer's content is the JSON
            # `ui.narrator`: the narrator writes each line inside its budget (P1), the
            # template stays the fallback
            narr = STATE.get("narrator")
            lines = (narr.for_request()
                     if narr is not None and ui_doc.get("narrator") == "on" else None)
            # v3 (`ui.narrator: live`, server/conductor/voice.py): the harness's lines in plain
            # words and the user's language, the live lines beside them
            vlang = (voice_mod.lang_of(body.get("messages"))
                     if ui_doc.get("narrator") == "live" else None)
            ui = rec.states = states_mod.States(
                ui_doc.get("kinds"), narrator=lines, voice=vlang,
                labels=narrator_mod.class_labels(STATE) if vlang is not None else None)
        # X1: `inline` (a reasoning part a line) on a `reasoning_content` stream, else `content`
        lines_in = states_mod.channel(ui_doc, fmt)
        # SRV-245 (UX2's P06, 2026-10-07): a plain API client (curl, an SDK) stores what `content`
        # carries as the answer; the progress lines went there inside the `tags` block. Such a
        # client gets its lines in `reasoning_content` when its format has one, else none
        # (a memory line still goes into the answer: SRV-185)
        plain_lines = rec.client_kind in compat.PLAIN_CLIENTS and (
            lines_in == "content" or fmt in ("tags", "both"))
        prog = states_mod.progress(ui_doc)
        live_lines = ui is not None and stream and prog["mode"] == "on"
        last_place: list = [None, 0.0]
        # SRV-237/249: Open WebUI's status line for every line said before the first generated
        # token (states.StatusBar), so its "Thought for N seconds" times the thinking alone
        bar = self._bar = (states_mod.StatusBar()
                           if ui is not None and stream and chat and n_choices == 1
                           and rec.client_kind in states_mod.status_clients() else None)

        def bar_out(kinds_lines, write) -> None:
            for kind, line in kinds_lines:
                ev = bar.line(kind, line)
                if ev is not None:
                    write(_chunk(cid, model, created, None, extra={"event": ev}).encode())

        def queued(me: dict) -> None:
            """The waiting line, when what keeps this request waiting or its place changed, or
            `queue_wait.LINE_EVERY_S` after the last one (with the estimate, when there is a
            basis for one): in the content, or as reasoning where the format has a field for it
            (before the lock nobody knows yet whether a `tags` answer opens a block)."""
            reason, pos = _wait_reason(me)
            now = time.monotonic()
            if now - me["t"] < qwait.QUIET_S:
                return                         # SRV-177: a short wait shows no line
            if ((reason, pos) == last_place[0]
                    and now - last_place[1] < qwait.LINE_EVERY_S):
                return
            last_place[0], last_place[1] = (reason, pos), now
            _, eta = _wait_estimate(me)
            if eta is not None:
                ui.add("queue_eta", reason=reason, n=pos, eta=qwait.about(eta))
            else:
                ui.add("queue", reason=reason, n=pos)
            if not ui.pending:
                return
            if plain_lines and fmt != "reasoning_content":
                ui.take_kinds()                  # SRV-245: never in a plain client's content
                return
            self._open_early(cid, model, created, plan)
            if bar is not None and bar.live:
                bar_out(ui.take_kinds(), self.wfile.write)      # SRV-249: not in the reasoning
                self.wfile.flush()
                return
            key = ("content" if lines_in == "content" and not plain_lines
                   else "reasoning_content")
            for _, line in ui.take_kinds():
                if lines_in == "inline":
                    # X1: the open reasoning part ends, the line opens the next (states.INLINE)
                    self._early_write(_chunk(cid, model, created, {"tool_calls": []}))
                    line = states_mod.inline(line)
                self._early_write(_chunk(cid, model, created, {key: line}))

        def mem_wait(short: float, place: int = 1) -> None:
            """SRV-250/SRV-305: the waiting line while the parallel admission holds this request
            for memory. Shown on every stream with lines (not only with the progress lines on):
            a wait for memory must never look like a hang."""
            if ui is None or not stream:
                return
            ui.add("queue", reason=(f"memory is almost full, your request starts when a running "
                                    f"one frees memory ({short:.1f} GiB short)"), n=place)
            if not ui.pending:
                return
            if plain_lines and fmt != "reasoning_content":
                ui.take_kinds()
                return
            self._open_early(cid, model, created, plan)
            if bar is not None and bar.live:
                bar_out(ui.take_kinds(), self.wfile.write)      # SRV-249: not in the reasoning
                self.wfile.flush()
                return
            key = ("content" if lines_in == "content" and not plain_lines
                   else "reasoning_content")
            for _, line in ui.take_kinds():
                if lines_in == "inline":
                    self._early_write(_chunk(cid, model, created, {"tool_calls": []}))
                    line = states_mod.inline(line)
                self._early_write(_chunk(cid, model, created, {key: line}))

        if STATE.get("draining"):
            rec.stop_detail = "shutting_down"
            return self._busy(503, "the server is shutting down", retry=30)
        # parallel requests (`--parallel N`): the side slot this request may take while slot 0 is
        # busy, from its length and whether it continues slot 0's resident conversation
        par = dict.get(STATE, "pool")
        par_why: dict = {}
        side_need = (_par_side_need(body, client, left_out, max_new, plan, images, lane_kind,
                                    par_why, stay=stay, parent=parent, session=conv_id)
                     if par is not None else None)
        side = None
        me = {"rank": rank, "t": time.monotonic(),
              "side": plan.side if plan is not None and plan.side is not None else None,
              # the conductor's own question to Qwen (server/conductor/classify.py's agent): the
              # waiting line of a turn behind it says so
              "own": (self.headers.get("User-Agent") or "").startswith(CLASSIFY_UA),
              "rec": rec}                 # SRV-249: the holder's live work, for the estimate
        rec.queue_why = _queue_why(par, par_why, side_need, 0.0, me)     # 0.8.2, the Lanes view
        with QUEUE:
            if INFLIGHT["waiting"] >= int(STATE.get("max_queue", 8)):
                INFLIGHT["refused"] += 1
                rec.stop_detail = "queue_full"
                return self._busy(503, f"{INFLIGHT['waiting']} requests are already queued and "
                                       f"this engine serves one at a time", retry=5)
            INFLIGHT["waiting"] += 1
            WAITERS.append(me)
        left = False
        own_wait, own_rank = [0.0], None          # SRV-177: the wait behind TandemLLM's own work
        try:
            # in one-second slices, so a client that gave up while queued -- opencode
            # cancels and re-sends -- does not get a generation nobody reads ahead of its retry
            # A client's side request is never refused for its wait: one too long for
            # the lane (opencode's summary) waits for the engine as long as a request may run,
            # its client's socket watched as before
            # 2026-10-05: a streamed chat request is never refused after minutes either.
            # It waits as long as one request may run, its stream opened after
            # `queue_wait.OPEN_AFTER_S` with an SSE comment every `PING_EVERY_S` (position, the
            # estimate, the time waited); a request without a stream keeps `--queue-timeout`.
            wait_s = float(STATE.get("queue_timeout", 120.0))
            ping = bool(stream and chat and n_choices == 1)
            if (plan is not None and plan.side is not None) or ping:
                wait_s = max(wait_s, float(STATE.get("request_timeout") or 3600.0))
            rec.queue_limit_s = wait_s
            t_wait = time.monotonic()
            wait_until = t_wait + wait_s
            next_ping = t_wait + qwait.OPEN_AFTER_S
            own_rank = _own_holder()
            got = False
            home = par_why.get("home") if side_need is not None else None
            if home is not None:
                # FX16: the side slot that holds this conversation's rows, before anything else;
                # while another request has it, up to SIDE_HOME_WAIT_S in the loop below
                side = par.take_slot(int(home), side_need, owner=cid)
                got = side is not None
                if got:
                    side.no_carry = bool(me["own"]) or lane_kind is not None
                    with par._lk:
                        par.stats["home_side"] = par.stats.get("home_side", 0) + 1
            if not got and side_need is not None and (par_why.get("guest")
                                                      or par_why.get("prefer_side")):
                # FX5: a short request unrelated to slot 0's conversation takes an idle side slot
                # before slot 0, so slot 0's rows stay that conversation's (else: slot 0 as a
                # guest, with the stash, as before)
                # FX10: not over a side slot's fresh end, which its conversation's next turn
                # carries into slot 0; slot 0 with the stash then, when it is free
                # FX16: a new conversation beside one a side slot could never hold, the same way
                side = par.take_side(side_need, owner=cid,
                                     keep_carry_s=CARRY_KEEP_S if SIDE_CARRY != "off" else 0.0)
                got = side is not None
                if got:
                    side.no_carry = bool(me["own"]) or lane_kind is not None  # FX15
                if got:
                    with par._lk:
                        key = "guest_side" if par_why.get("guest") else "new_side"
                        par.stats[key] = par.stats.get(key, 0) + 1
            new_wait = bool(par_why.get("prefer_side_wait")) and not got and side_need is not None
            # 0.8.2: its conversation is slot 0's: slot 0 first, a side slot after SIDE_SLOT0_WAIT_S
            slot0_first = bool(par_why.get("slot0_first")) and side_need is not None
            if not got and home is None and not new_wait:
                got = _take(me, None)
            if (not got and home is None and not new_wait and side_need is not None
                    and not slot0_first and _side_ok(0.0, _own_holder())):
                side = par.take_side(side_need, owner=cid)
                got = side is not None
                if got:
                    side.no_carry = bool(me["own"]) or lane_kind is not None
            micro = not got and lane_kind is not None
            while not got and not micro:
                # 0.8.2: the Lanes view's queued line says why (server/parallel.py)
                rec.queue_why = _queue_why(par, par_why, side_need, time.monotonic() - t_wait, me)
                if new_wait:
                    # FX17: the side slot first, for SIDE_NEW_WAIT_S; then slot 0 as before
                    side = par.take_side(side_need, owner=cid,
                                         keep_carry_s=CARRY_KEEP_S if SIDE_CARRY != "off" else 0.0)
                    got = side is not None
                    if got:
                        side.no_carry = bool(me["own"]) or lane_kind is not None
                        with par._lk:
                            par.stats["new_side"] = par.stats.get("new_side", 0) + 1
                        break
                    if time.monotonic() - t_wait < SIDE_NEW_WAIT_S:
                        time.sleep(0.1)
                        if self._client_gone():
                            left = True
                            rec.client_gone_at = time.perf_counter()
                            break
                        continue
                    new_wait = False
                    par_why["new_wait"] = round(time.monotonic() - t_wait, 1)
                if home is not None:
                    side = par.take_slot(int(home), side_need, owner=cid)
                    got = side is not None
                    if got:
                        side.no_carry = bool(me["own"]) or lane_kind is not None
                        with par._lk:
                            par.stats["home_side"] = par.stats.get("home_side", 0) + 1
                        break
                    if time.monotonic() - t_wait < SIDE_HOME_WAIT_S:
                        time.sleep(0.05)
                        if self._client_gone():
                            left = True
                            rec.client_gone_at = time.perf_counter()
                            break
                        continue
                    # its slot stays busy: slot 0 then (the rows are carried in when they are
                    # still there), or any side slot
                    home = None
                    par_why["home_wait"] = round(time.monotonic() - t_wait, 1)
                    par_why["hop_from"] = int(par_why["home"])      # SRV-528
                if self._client_gone():
                    left = True
                    rec.client_gone_at = time.perf_counter()
                    break
                slice_s = min(1.0, wait_until - time.monotonic())
                if slice_s <= 0:
                    break
                if live_lines and (lines_in == "content" or fmt != "tags"):
                    queued(me)
                if ping and time.monotonic() >= next_ping:
                    next_ping = time.monotonic() + qwait.PING_EVERY_S
                    try:
                        self._open_early(cid, model, created, plan)
                        pos, eta = _wait_estimate(me)
                        self.wfile.write(qwait.comment(pos, eta, time.monotonic() - t_wait))
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        left = True
                        rec.client_gone_at = time.perf_counter()
                        break
                t_slice, holder_own = time.monotonic(), _own_holder()
                got = _take(me, slice_s if side_need is None else min(slice_s, 0.25))
                if holder_own is not None:
                    # SRV-177: the time this request waited behind TandemLLM's own work
                    own_wait[0] += time.monotonic() - t_slice
                    own_rank = holder_own
                waited = time.monotonic() - t_wait
                if (not got and side_need is not None and _side_ok(waited, _own_holder())
                        and (not slot0_first
                             or waited >= float(par_why.get("slot0_wait_s") or SIDE_SLOT0_WAIT_S))):
                    side = par.take_side(side_need, owner=cid)
                    if side is not None:
                        side.no_carry = bool(me["own"]) or lane_kind is not None
                        if slot0_first:
                            # slot 0 stayed busy: the side slot, its prefill from the disk tier
                            # and the rest read again, not a wait behind slot 0's run
                            par_why["slot0_wait"] = round(waited, 1)
                            with par._lk:
                                par.stats["slot0_gave_up"] = par.stats.get("slot0_gave_up", 0) + 1
                    got = side is not None
        finally:
            # Off the waiting list whatever happened, including an exception in `acquire` itself.
            # A counter that leaks on the error path turns the queue bound into a slowly closing
            # door, and the symptom -- 503s on an idle server, hours later -- would be read as a
            # leak somewhere else entirely.
            with QUEUE:
                INFLIGHT["waiting"] -= 1
                WAITERS.remove(me)
        if own_wait[0] > 0:
            rec.wait_own_ms = round(own_wait[0] * 1e3, 1)          # the ledger's `wait_own_ms`
            if got and own_rank in (RANK_SHADOW, RANK_PREP):
                CLOCK.yielded(own_rank, own_wait[0])               # the estimate's basis
        if left:
            with QUEUE:
                INFLIGHT["abandoned"] += 1
            rec.finish_reason = "abandoned"
            print(f"[req] {cid} left while queued ({(time.perf_counter() - t_req) * 1e3:.0f} ms)",
                  flush=True)
            raise ClientGone("client left while queued")
        if not got and not micro:
            with QUEUE:
                INFLIGHT["refused"] += 1
            rec.stop_detail = "queue_timeout"
            return self._busy(429, f"timed out waiting for the engine ({wait_s:.0f} s)", retry=10)
        with QUEUE:
            INFLIGHT["lane" if micro else "running"] += 1
        if par is not None and not micro and side is None:
            par.took_slot0()
        if par is not None and not micro and par_why.get("hop_from") is not None and \
                (side is None or side.index != par_why["hop_from"]):
            # SRV-528: its conversation's slot stayed busy past SIDE_HOME_WAIT_S; the tokens the
            # slot it took re-read are added at its end (`hop_tokens`)
            with par._lk:
                par.stats["hops"] = par.stats.get("hops", 0) + 1
        if par is not None and not micro and (side is not None or par_why.get("wait")
                                              or par_why.get("hop_from") is not None):
            # SRV-197: which slot, and the numbers it was chosen on (the cache share of an agent's
            # turns hangs on it: a side slot has no resident rows to resume)
            nums = _par_nums(par_why)
            if side is not None:
                # A request that shares only the client's head with slot 0's
                # conversation is a new conversation, and says so
                head = max(int(par_why.get("view_common") or 0), int(par_why.get("res_common") or 0))
                conv = par_why.get("conv")
                new_conv = (f" new conversation (shared head {head})"
                            if conv is not None and head >= 1024 and head < conv else "")
                if par_why.get("home") == side.index:
                    new_conv = " continues its conversation here (home)"
                elif par_why.get("hop_from") is not None:
                    new_conv = (f" moved from slot {par_why['hop_from']} (busy "
                                f"{par_why.get('home_wait')} s): re-reads what this slot lacks")
                elif par_why.get("slot0_wait") is not None:
                    new_conv = (f" slot 0 still busy after {par_why['slot0_wait']} s, its "
                                f"conversation ({par_why.get('wait')}) read here")
                clip = ""
                if par_why.get("clip"):
                    # FX16: the slot's window leaves this request less than its output limit
                    room = _side_room(side, int(par_why.get("n_rows")
                                                or len(par_why.get("ids") or ())),
                                      int(par_why.get("margin") or 0), max_new)
                    if room < max_new:
                        clip = f" max_tokens {max_new} -> {room} (the slot's window)"
                        max_new = room
                        par_why["max_new"] = room
                print(f"[parallel] {cid} slot {side.index} ({len(par_why.get('ids') or ())} "
                      f"tokens; slot 0 {_holder_name()}):{new_conv} {nums}{clip}", flush=True)
            elif par_why.get("wait") in ("view", "resident", "spill"):
                print(f"[parallel] {cid} slot 0, continues its conversation "
                      f"({par_why['wait']}): {nums}", flush=True)
            elif par_why.get("hop_from") is not None:
                print(f"[parallel] {cid} slot 0, moved from slot {par_why['hop_from']} (busy "
                      f"{par_why.get('home_wait')} s): carried in or read again: {nums}", flush=True)
        if side is not None:
            # a side slot: this thread sees the slot's engine and drafter through `STATE`, and
            # runs its GPU work in its turns
            slots_mod.bind(side)
            rec.slot = side.index
            par.turn.reset(side.index)
            par.turn.acquire(side.index)
        elif par is not None and not micro:
            rec.slot = 0
        if not micro:
            rec.lock_acquired()
            STATE.pop("spill_restored", None)
            STATE.pop("side_carried", None)
            STATE.pop("park_returned", None)
        # SRV-197: the generation list a request on slot 0 finds; only one that generated
        # (`_generate_stream` publishes a new list) records its conversation as slot 0's
        ctx_found = STATE.get("last_ctx") if par is not None and side is None else None
        deadline = Deadline(float(STATE.get("request_timeout", 0.0)))
        mm_end = None                      # SRV-629: the served images' context, for the view keep
        try:
            prompt, _, in_think = build_prompt(body, client, left_out=left_out)
            if int(prompt.numel()) == 0:
                # SRV-167 (and MP1's 2026-10-06 report): nothing to prefill (a prompt the
                # tokenizer reads as no token) is the client's 400, before any stage reads it,
                # never "a prefill needs at least one token" as a 500
                return self._json(400, compat.Refusal(
                    "messages" if chat else "prompt", "the prompt is empty").body())
            if par is not None and not micro and not self._par_admit(
                    par, cid, prompt, rec=rec, say=mem_wait, max_new=max_new, stream=stream,
                    rows=_mm_rows(int(prompt.numel()), images) if images else None):
                return
            clean_ids = None
            plan_hidden = 0                        # SRV-165: the router's tokens in the prompt
            if plan is not None and plan.context is not None and not micro:
                # Qwen decides the class inside the conversation (server/conductor/incontext.py):
                # one prefill of the prompt up to its last message plus a hidden question, right
                # before this turn's own prefill, which then resumes from the anchors it left
                prompt_n = int(prompt.numel())
                if prompt_n + 1 < int(STATE["max_len"]):
                    gone = self._prefill_watch(False)
                    STATE["conductor"].decide_in_context(
                        plan, prompt.tolist(),
                        lambda ids, letters, at: context_choice(ids, letters, at, conv_id,
                                                                gone, images),
                        lambda ids: context_rows(ids, conv_id, gone, images))
                else:
                    STATE["conductor"].router.settle_context(plan, "prompt_too_long")
                # the turn's class is known now, and with it the budget
                budget = routed(client.budget if client.asked else server_budget)
            if plan is not None and plan.ask is not None and not micro:
                # J1 step 6 (shadow): Qwen asked in place on the turn's narrowed outcome, by the
                # in-context stage's forward; journaled, nothing of the turn changes
                if int(prompt.numel()) + 1 < int(STATE["max_len"]):
                    gone = self._prefill_watch(False)
                    STATE["conductor"].router.ask_in_place(
                        plan, prompt.tolist(), STATE["tok"],
                        lambda ids, letters, at: context_choice(ids, letters, at, conv_id,
                                                                gone, images))
                else:
                    plan.ask = None
            if plan is not None:
                # the notes go into the last turn, before its <|im_end|>; earlier ones replay;
                # thinking off closes the generation prompt's <think> (the head never changes)
                n_rendered = int(prompt.numel())
                prompt, in_think = plan.apply_prompt(prompt, STATE["tok"], in_think)
                if plan.changed:
                    clean_ids = plan.clean_ids
                # SRV-165 (UT1, 2026-10-06: a 'hi' reported 238 prompt tokens, raw 11): the notes
                # the router put in (this turn's and the session's replayed ones, the closed think
                # block of a turn it turned thinking off) are not what the client sent
                plan_hidden = int(prompt.numel()) - n_rendered
                if getattr(plan, "memory", None) is not None:
                    conductor_memory.mark_injected(STATE["conductor"].memory, plan)
                    STATE["conductor"].memory.writer.seen(conv_id, plan.memory,
                                                          plan.memory.get("text"))
            # A compaction as the continuation of the session's conversation
            comp = None
            if (chat and plan is not None and plan.active and getattr(plan, "compaction", False)
                    and not images
                    and n_choices == 1):
                comp = _compaction_plan(body, conv_id, rec.client_kind, int(prompt.numel()),
                                        plan.row)
                if comp is not None:
                    if comp["prepared"] is None:
                        prompt = torch.tensor(comp["ids"], dtype=prompt.dtype,
                                              device=prompt.device)
                    in_think, clean_ids, plan_hidden = False, None, 0
                    max_new = min(max_new, int(STATE["compaction"]["max_tokens"]))
            # ENG-181: the probe snapshots of a TandemLLM request with tools (server/probe.py),
            # at the row of the last message's last token, which no note of this turn moves
            snaps = None
            if (STATE.get("probe") is not None and not micro
                    and probe_mod.wants(plan, body, n_choices)
                    and (self.headers.get("X-Tandem-Probe") or "").lower() != "off"):
                snaps = rec.probe = STATE["probe"].capture(plan.probe_row(prompt, STATE["tok"]),
                                                           int(prompt.numel()))
            # The tokens the monitor hid in earlier answers, put back where the
            # resident rows hold them when the prompt then starts with all of the last
            # generation, so it resumes at its end anchor (engine/cache.py `replay`); a request
            # with images is served as rendered
            # the side lane leaves the resident rows alone: they are lent and given back
            replayed, aligned, res_ = [], 0, None if micro else STATE.get("resident")
            # SRV-233: where the served prompt holds the router's notes (its own positions), and
            # the tokens of the closed think block at the tail when the plan turned thinking off
            hid_at = [(int(a), int(b)) for a, b, *_ in
                      (getattr(plan, "spans", ()) if plan is not None and getattr(
                          plan, "changed", False) else ())]
            hid_tail = max(0, plan_hidden - sum(b - a for a, b in hid_at))
            s1_len = int(prompt.numel())
            # FX17 (0.4.8 run 4): a guest of slot 0's conversation (the router's own question,
            # a side-lane job, a short unrelated request) is served over the resident rows with
            # the stash and leaves every other cache alone: no carry, no NVMe restore, no park
            # swap for it (two router questions on slot 0 while the side slot was busy swapped
            # the park in and the 15,8k-token chat lost its end anchor; its next turn re-read it)
            guest = bool(par_why.get("guest")) or bool(me.get("own")) or lane_kind is not None
            if (res_ is not None and not images and not (comp and comp["prepared"] is not None)
                    and not guest):
                if comp is None:
                    # FX10 (SRV-213): a conversation that ran on a side slot is carried over first
                    with torch.no_grad():
                        _carry_from_side(prompt.tolist(), conv_id)
                # A conversation on the NVMe that this prompt continues is read back
                # first, so the replay below works on its rows and its end anchor
                _spill_before_replay(prompt.tolist(), mm_ok=comp is not None)
            if res_ is not None and res_.end is not None and not images and comp is None:
                before = prompt.tolist()
                ids_, replayed, aligned = _end_replay(res_, before)
                if replayed or aligned:
                    prompt = torch.tensor(ids_, dtype=prompt.dtype, device=prompt.device)
                    if clean_ids is None:
                        clean_ids = before
                    if snaps is not None and snaps.pos is not None:
                        snaps.pos += len(ids_) - len(before)   # the spans sit before it
                    if plan is not None and plan.row is not None:
                        plan.row["extra"]["end_replayed"] = len(ids_) - len(before)
                        if aligned:
                            plan.row["extra"]["end_aligned"] = aligned
                        al_m = STATE.get("end_align")
                        if al_m is not None and al_m.last_moved:
                            # FX16: the client's own block, served where the model saw it
                            plan.row["extra"]["end_moved"] = sum(b - a for a, b in al_m.last_moved)
                elif len(before) >= REREAD_SAY_TOKENS:
                    # FX16: a long prompt that does not resume at the end anchor says where its
                    # copy parted from the rows, so a re-read is never silent (positions only)
                    _say_reread(cid, res_, before, plan)
                    # SRV-233: below the end anchor the rows hold what the client's copy lacks
                    # (the earlier notes the end anchor kept, the hidden spans, a reasoning the
                    # copy left out); above it, this prompt's own notes moved by what came in
                    e_at, d = int(res_.end["at"]), len(ids_) - len(before)
                    al_ = STATE.get("end_align")
                    hid_at = ([(int(a), int(b)) for a, b in (res_.end.get("served_hidden") or ())
                               if int(b) <= e_at]
                              + [(int(p), int(p) + len(h)) for p, h in replayed]
                              + list(al_.last_omitted if al_ is not None and aligned else ())
                              + [(a + d, b + d) for a, b in hid_at if a >= e_at - d])
                    s1_len += d
            # the router's state lines, now that the plan is applied
            if ui is not None:
                ui.route(plan.row, plan.injected,
                         STATE["conductor"].store.peek("packages")[1].get("packages"),
                         plan.compaction)
            # SP4: the conductor's session monitor sees the request before anything is prefilled,
            # and may add a hidden note at the prompt's tail or set the thinking budget
            sup, client_ids = None, None
            final_ids = None                       # A1's give-up answer (server/argloop.py)
            # what the replay added to the client's copy, net of the whitespace it aligned
            hidden_prompt = plan_hidden + (int(prompt.numel()) - len(before)
                                           if replayed or aligned else 0)
            hidden_spans = list(replayed)          # the served prompt's tokens the client lacks
            mon = STATE.get("monitor")
            if mon is not None and mon_mode in ("shadow", "on") and not micro:
                req_ev = conductor.RequestEvent(
                    cid, messages=body.get("messages") or [],
                    tools=(body.get("tools") or []) if parse_calls else [], tool_choice=tool_mode,
                    n=n_choices, stream=stream, prompt_tokens=int(prompt.numel()),
                    in_think=bool(in_think), max_tokens=max_new,
                    client_budget=client.budget,
                    client=rec.client_kind or "", model=str(model),
                    constrained=gram is not None,
                    profile=plan.res.profile if plan is not None and plan.active else "",
                    cls=str((plan.row or {}).get("class_id") or "")
                    if plan is not None and plan.active else "")
                # one session for the router and the monitor: the router's key when it has one
                sup = rec.sup = mon.begin(
                    mon_mode, req_ev, prompt.tolist(), rec.conv, policy=mon_policy,
                    session_key=plan.session.key if plan is not None and plan.active
                    and plan.session is not None else None)
                if ui is not None:
                    sup.on_event = ui.monitor
                if plan is not None and plan.acts:
                    sup.adopt(plan.acts)            # the router's outcome note (J1)
                if plan is not None and plan.outcome is not None and \
                        plan.outcome["so"]["evaluate"] != "off":
                    # J1 step 6, E-b: the claim check's note carries the turn's evidence
                    sup.evidence = (plan.outcome["so"]["evaluate"],
                                    conductor_outcomes.evidence_text(plan.outcome))
                sup.memo["client_thinking_off"] = bool(client.off)
                if getattr(rec, "loop_escape", None):
                    sup.memo["loop_escape"] = rec.loop_escape       # SRV-475: the escape note
                acts = sup.at_request(req_ev)
                if sup.final_answer:
                    # A1's give-up after the retries (server/argloop.py): this answer is the
                    # sentence, served as a replay is, never generated; the client's tool loop
                    # ends on a plain answer
                    eos_t = tok.eos_token_id if isinstance(tok.eos_token_id, int) \
                        else min(eos_ids(body))
                    final_ids = [int(t) for t in tok(
                        ("\n" + CLOSE_THINK + "\n\n" if in_think else "") + sup.final_answer,
                        add_special_tokens=False).input_ids] + [int(eos_t)]
                    print(f"[argloop] {cid} gave up after the retries: answered without a "
                          f"generation", flush=True)
                if acts:
                    client_ids = prompt.tolist()
                    spliced, in_think = sup.apply_request(client_ids, bool(in_think), acts)
                    prompt = torch.tensor(spliced, dtype=torch.long, device=prompt.device)
                    hidden_prompt += len(spliced) - len(client_ids)
                    ins = _inserted(client_ids, spliced)        # SRV-233: SP4's note
                    if ins is None:
                        hid_at = None
                    elif hid_at is not None:
                        k_ = len(ins[1])
                        hid_at = [(a + k_, b + k_) if a >= ins[0] else (a, b) for a, b in hid_at]
                        hid_at.append((ins[0], ins[0] + k_))
                        s1_len += k_
                    span = (_inserted(client_ids, spliced)
                            if res_ is not None and res_.end_on else None)
                    if span is not None:
                        hidden_spans = hidden_spans + [span]   # the end anchor replays it too
            # what the client sent, without the router's notes or the monitor's: the suffix store
            client_ids = clean_ids or client_ids
            mm = None
            if images:
                try:
                    n_text = int(prompt.numel())
                    text_prompt = prompt
                    prompt, mm = _mm_prompt(prompt, images, rec)
                    if snaps is not None and snaps.pos is not None:
                        snaps.pos += int(prompt.numel()) - n_text   # the images sit before it
                except ValueError as exc:
                    return self._json(400, {"error": {"message": str(exc),
                                                      "type": "invalid_request_error",
                                                      "param": "messages"}})
                if MM_END == "on" and res_ is not None and comp is None and not guest:
                    # SRV-546: the conversation on the NVMe read back, and the end anchor's replay
                    # and alignment, on the expanded key ids (`_mm_end_replay`); SP4's note
                    # (text positions) moves to the rows' coordinates with it
                    try:
                        # ENG-371: a conversation that ran on a side slot and now has an image
                        # (a worker's first screenshot) is carried into slot 0 as a text turn is,
                        # when its images all come after the side slot's end
                        with torch.no_grad():
                            _carry_from_side(list(mm.key_ids), conv_id)
                        _spill_before_replay(list(mm.key_ids), mm_ok=True)
                        mm_old, keys_n = mm, int(prompt.numel())
                        e_before = int(res_.end["at"]) if res_.end is not None else 0
                        if res_.end is not None:
                            prompt, mm, mm_spans, mm_aligned, added = _mm_end_replay(
                                res_, text_prompt, prompt, mm, images, rec)
                        else:
                            mm_spans, mm_aligned, added = [], 0, 0
                        sp4 = []
                        for p_, h_ in hidden_spans:
                            q_ = _mm_text_pos(mm_old, int(p_))
                            sp4.append((q_ + added if q_ >= e_before - added else q_, h_))
                        hidden_spans = list(mm_spans) + sp4
                        if mm_spans or mm_aligned:
                            replayed, aligned = list(mm_spans), mm_aligned
                            hidden_prompt += added
                            if snaps is not None and snaps.pos is not None:
                                snaps.pos += added
                            if plan is not None and plan.row is not None:
                                plan.row["extra"]["end_replayed"] = added
                                if mm_aligned:
                                    plan.row["extra"]["end_aligned"] = mm_aligned
                                plan.row["extra"]["end_mm"] = True
                        elif keys_n >= REREAD_SAY_TOKENS:
                            _say_reread(cid, res_, list(mm.key_ids), plan,
                                        why_mm=STATE.pop("mm_end_why", None))
                    except Exception:                              # noqa: BLE001
                        logbuf.print_exc()      # served as rendered, as before SRV-546
            mm_end = mm
            eos = eos_ids(body)
            n_prompt = int(prompt.numel())
            # the tokens SP4 added are hidden: `prompt_tokens` is what the client sent
            rec.prompt_tokens, rec.thinking = n_prompt - hidden_prompt, bool(in_think)
            rec.hidden_prompt = max(0, int(hidden_prompt))
            rec.hidden_at = _hidden_positions(hid_at, hid_tail, s1_len, n_prompt,
                                              rec.hidden_prompt, plan)
            # FX16: the same ranges without the count check, for the end anchor's record of what
            # the client never sent (the moved-block pass needs the notes' positions, which the
            # usage accounting may decline as `hidden_at_off`)
            rec.note_at = _hidden_positions(hid_at, hid_tail, s1_len, n_prompt,
                                            rec.hidden_prompt, plan, strict=False)
            if head_parts is not None:
                rec.parts = dict(head_parts, images=int(mm.image_tokens) if mm is not None else 0)
                try:
                    rec.harness = _harness_of(plan)
                except Exception:                                  # noqa: BLE001
                    logbuf.print_exc()       # a usage figure must never fail a request
            if comp is not None:
                # what the client sent; the tokens the continuation reused count as cached
                rec.prompt_tokens = rec.prompt_client = comp["client_n"]
                rec.hidden_prompt = 0
            # BUG 2, the second half. The KV buffer is `--max-len` long and the recurrent state is
            # indexed by absolute position, so a generation that runs past the end of it does not
            # degrade -- it raises, part way through a stream whose headers have already gone out.
            # The request is clamped to what the engine can hold, and refused outright when the
            # prompt alone does not fit, which is a 400 the client can read rather than a truncated
            # answer it cannot.
            room = int(STATE["max_len"]) - n_prompt - 1
            if room <= 0:
                rec.stop_detail = "prompt_too_long"
                return self._json(400, {"error": {
                    "message": f"prompt is {n_prompt} tokens and the context is "
                               f"{STATE['max_len']}; nothing is left to generate",
                    "type": "invalid_request_error", "param": "messages"}})
            max_new = max(1, min(max_new, room))
            rec.max_tokens = max_new
            # A client that asked for a budget, in any shape, gets its reasoning closed early
            # enough to leave the answer its reserve of the output limit: the limit never ends
            # its turn inside the block. A request that asked for none is served as before.
            lowered = False
            if json_answer and in_think and not client.asked:
                # SRV-169: a JSON answer gets the answer reserve as a client's budget does, so
                # the thinking cannot spend the whole output limit and cut the JSON (UT1: a
                # 270-token think block, then `{"name": "Margot Ellery", "age": 41, "t` at
                # max_tokens 300)
                closer = len(tok(ThinkBudget.PHRASE, add_special_tokens=False).input_ids)
                cap, reserve = thinking_mod.with_reserve(
                    budget, max_new, STATE.get("answer_reserve", thinking_mod.AUTO), closer)
                lowered = cap != budget
                print(f"[think] {cid} json answer: reserve {reserve} of {max_new} -> cap "
                      f"{cap or 'none'}", flush=True)
                if plan is not None and plan.row is not None:
                    plan.row.setdefault("extra", {})["json_reserve"] = {
                        "reserve": reserve, "max_tokens": max_new, "cap": cap,
                        "lowered": lowered}
                budget = cap
            if client.asked and in_think:
                # the closing phrase is counted in: it streams where no monitor hides it
                closer = len(tok(ThinkBudget.PHRASE, add_special_tokens=False).input_ids)
                cap, reserve = thinking_mod.with_reserve(
                    budget, max_new, STATE.get("answer_reserve", thinking_mod.AUTO), closer)
                lowered = cap != budget
                print(f"[think] {cid} client budget {client.budget} ({client.source}), in force "
                      f"{budget or 'none'}, answer reserve {reserve} of {max_new} -> cap "
                      f"{cap or 'none'}", flush=True)
                if plan is not None and plan.row is not None:
                    plan.row.setdefault("extra", {})["client_budget"] = {
                        "field": client.source, "kind": client.kind, "asked": client.budget,
                        "reserve": reserve, "max_tokens": max_new, "cap": cap,
                        "lowered": lowered}
                budget = cap
            # The cap in force -- the router's for the turn's effort level, the client's, or
            # the reserve's -- closes with the monitor's hidden close (SP1), so ThinkBudget gets
            # none. Without a monitor that acts (or for `n` > 1: SP1 closes once a request) the
            # cap stays ThinkBudget's, its phrase streamed.
            routed = 0
            if (budget and sup is not None and n_choices == 1
                    and ((plan is not None and plan.budget) or client.asked or lowered)):
                task = ("reserve" if lowered else "route" if plan is not None and plan.budget
                        else "client")
                if sup.route_budget(budget, task=task):
                    routed, budget = budget, 0
            # OPS-47: the memory limits, before anything is prefilled and before the response
            # has started. A compaction drops its session's store entries and parked rows first;
            # then the engine is brought under its ceiling and the prefill's workspace under its
            # declared peak, the caches given back cheapest first -- or the request gets a 503.
            compacting = bool(chat and _compacting(plan, body, rec.client_kind))
            mem = STATE.get("mem")
            if mem is not None and not micro:
                memlimits.refresh(STATE)
                if compacting and conv_id:
                    for name in ("state_store", "resident"):
                        obj = STATE.get(name)
                        if obj is not None:
                            obj.drop_session(conv_id)
                try:
                    _gate_holding_end(mem, res_, list(mm.key_ids) if mm is not None
                                      else prompt.tolist(), n_prompt,
                                      mm is not None and MM_END == "on" and comp is None)
                except cache.MemoryRefused as exc:
                    rec.stop_detail = "memory"
                    print(f"[memory] {cid} refused: {exc}", flush=True)
                    return self._busy(503, f"{exc}; retry when the engine is idle", retry=30)
            # the reasoning-loop arms compute only for a request the monitor watches
            arms = sup.arms if sup is not None and STATE.get("think_stall", True) else None
            if sup is not None:
                sup.req.max_tokens = max_new
            think = ThinkBudget(tok, budget, arms=arms)
            rec.live_refs = (think, None)              # read by the live sampler
            prompt_ids = prompt.tolist()
            # SR1: the shadow reader is fed this turn's served prompt (host work only; it reads
            # in the idle window, server/conductor/shadow.py)
            shadow_key, shadow_svc, shadow_cfg = ((None, None, None) if micro else
                                                  _shadow_begin(plan, prompt_ids, n_choices))
            # C1: the disk tier's settings (a new version of the conductor's document applies
            # now) and what it is told about this request
            disk_tag = None
            if STATE.get("disk_cache") is not None and not micro:
                diskcache_mod.refresh(STATE)
                disk_tag = diskcache_mod.request_tag(plan, rec.client_kind, prompt_ids, tok)
            if gram is not None:
                # the constraint joins the penalties as the last logit processor: every decision
                # site already calls `pen`, so the loop is unchanged
                cons = grammar_mod.Constraint(gram, eos, STATE.get("device", "cuda"),
                                              think_end=think.end_id, in_think=in_think)
                pen = cons if pen is None else grammar_mod.LogitChain([pen, cons])

            # The exact-prompt response cache. Greedy decoding is a function of (prompt, params),
            # so an identical request has an identical answer and this is memoisation rather than
            # an approximation. Under a relaxed accept rule the engine is not answering the
            # greedy question at all, and that is the one setting where the key would be lying
            # about what produced the value -- so the cache is not consulted.
            rcache = STATE.get("response_cache")
            rkey, cached_ids = None, None
            if final_ids is not None:
                cached_ids = final_ids              # A1's give-up: the sentence, no engine
            elif comp is not None and comp["prepared"] is not None:
                cached_ids = comp["prepared"]       # the summary written ahead (no engine)
            elif (rcache is not None and not STATE["relax"].on and not sampler.on
                    and lp_top is None and comp is None
                    and not (plan is not None and plan.changed) and not replayed):
                # A sampled answer is not a function of (prompt, params) in any replayable sense
                # without the RNG stream; do not memoise it. A request for log-probabilities needs
                # the rows, and a replay has none.
                # The penalty values are part of the question being memoised: greedy under
                # penalties is a different function, and a key without them would replay an
                # answer a different setting produced (cache-key fix).
                tools_key = hashlib.sha256(json.dumps(
                    [body.get("tools"), body.get("tool_choice")], sort_keys=True).encode()
                    ).hexdigest()[:16] if body.get("tools") else ""
                # The effort level the request resolved to, the client's word and the router's
                # level: two levels can render one prompt and one cap (`high` and `xhigh` at a
                # 16,384-token limit), and a run at one level must not replay the other's answers
                effort_key = (client.effort or "",
                              str(((plan.row or {}).get("effort") if plan is not None else None)
                                  or ""))
                rkey = cache.ResponseCache.key(
                    mm.key_ids if mm is not None else prompt_ids, max_new=max_new, budget=budget,
                    effort=effort_key, stops=tuple(stops),
                    eos=tuple(sorted(eos)), tree=bool(STATE.get("tree")), pen=pen_spec.key(),
                    tools=tools_key, **({"grammar": ("parts:" if gram.side is not None else "")
                                         + gram.pattern} if gram is not None else {}),
                    **({"hidden_cap": routed} if routed else {}),
                    **({"conductor": f"{sup.mode}:{sup.policy.version}"} if sup is not None
                       else {}))
                cached_ids = rcache.get(rkey)
            # This request's own prefill publishes a NEW dict here; a replay publishes none.
            prefill_before = STATE.get("last_prefill")
            lpr = lp_mod.Recorder(lp_top) if lp_top is not None else None

            def open_source(samp, budget_state, guard, recorder, cached):
                """One generation's token source: the replay, or the engine."""
                if cached is not None:
                    if final_ids is None:
                        rec.absorb_response_cache()
                    return rec.track(iter(list(cached)))
                if micro:
                    return rec.track(lane_source(samp, guard, recorder))
                kw = {"lpr": recorder} if recorder is not None else {}
                if mm is not None:
                    kw["mm"] = mm
                    if MM_END == "on" and comp is None:
                        # SRV-546: an end anchor or an NVMe entry kept after images is this
                        # prompt's when it holds every row up to it (the images' keys included):
                        # the rows below it took the rotary these images give them
                        kw["end_mm"] = True
                if sup is not None:
                    kw["sup"] = sup
                if compacting:
                    kw["compacting"] = True
                if snaps is not None:
                    kw["probe"] = snaps
                if disk_tag is not None:
                    kw["disk_tag"] = disk_tag
                if comp is not None:
                    kw["pos_delta"], kw["end_mm"] = comp["record"].delta, True
                if stay:
                    kw["stay"] = True                  # ENG-371: never a guest of the resident
                return rec.track(generate_stream(prompt, max_new, eos, budget_state, conv_id,
                                                 deadline, pen=pen, pstop=guard, sampler=samp,
                                                 on_prefill=watch, **kw))

            def lane_source(samp, guard, recorder):
                """The side lane's tokens (server/preempt.py): served at a safe point
                of the generation that holds the engine, or under the lock when none runs."""
                job = rec.lane_job = preempt.Job(
                    prompt_ids, max_new, eos, sampler=samp, pen=pen, lpr=recorder, pstop=guard,
                    cid=cid, kind=lane_kind)
                rec.lane = "pending"                   # the live view: waiting for a safe point
                me_ = {"rank": RANK_SIDE, "t": time.monotonic(), "side": None}
                waiting = [False]

                def take() -> bool:
                    if not waiting[0]:
                        with QUEUE:
                            WAITERS.append(me_)        # turns waiting for the lock go after it
                        waiting[0] = True
                    return _take(me_, None)

                def release() -> None:
                    with QUEUE:
                        HOLDER["who"] = None
                    LOCK.release()
                    if me_.get("t_hold") is not None:
                        CLOCK.done(RANK_SIDE, time.monotonic() - me_["t_hold"])

                try:
                    for t in STATE["lane"].tokens(job, take=take, release=release,
                                                  gone=self._client_gone):
                        if rec.t_lock is None:
                            rec.t_lock = job.t_start   # the wait for the safe point is queue time
                            rec.lane = job.where       # the live view's lane tag
                            if job.prefill is not None:
                                rec.absorb_prefill(job.prefill)
                        yield t
                finally:
                    if job.state == "running":
                        # a stream ends at its end token, which comes before the engine is given
                        # back: the request's line and counts wait for that (a few ms)
                        job.done.wait(10.0)
                    if waiting[0]:
                        with QUEUE:
                            if me_ in WAITERS:
                                WAITERS.remove(me_)
                    if rec.t_lock is None and job.t_start is not None:
                        rec.t_lock = job.t_start
                    rec.lane = job.where
                    if job.prefill is not None and rec.prompt_n is None:
                        rec.absorb_prefill(job.prefill)
                    if plan is not None and plan.row is not None:
                        plan.row.setdefault("extra", {})["lane"] = {
                            "where": job.where, "wait_ms": round(job.wait_ms or 0.0, 1),
                            "lent_ms": round(job.lent_ms or 0.0, 1),
                            "borrowed_mb": round(job.borrowed_bytes / 2**20, 1)}

            watch = self._prefill_watch(stream, rec)
            rec.prefill_info = watch.info
            source = open_source(sampler, think, pstop, lpr, cached_ids)
            if comp is not None and comp["prepared"] is not None:
                rec.cache_source = "prepared"

            def settle(ids: list[int], finish: str, exc: BaseException | None = None,
                       calls: int = 0, more: tuple = (), prefill: bool = True) -> None:
                """The record's counts, from the ids the engine committed. `more`: the
                earlier choices' ids of an `n > 1` request, whose tokens count too."""
                rec.completion_tokens = len(ids) + sum(len(m) for m in more)
                rec.finish_reason, rec.tool_calls = finish, calls
                if plan is not None and getattr(plan, "compaction", False) and finish == "stop" \
                        and ids:
                    plan.compaction_ids = list(ids)     # the summary: the session's alias key
                if transcribe:
                    rec.out = (ids, in_think)      # SRV-137: decoded by the transcript writer
                if exc is not None:
                    rec.error_type = type(exc).__name__
                if gram is not None and gram.side is not None:
                    # SRV-619: the branch the model's first answer token picked
                    br = gram.branch(ids, think.end_id, in_think)
                    print(f"[structured] {cid} branch={br}", flush=True)
                    if plan is not None and plan.row is not None:
                        plan.row.setdefault("extra", {})["structured_branch"] = br
                if in_think:
                    rec.reasoning_tokens = sum(usage_mod.reasoning_count(
                        x, usage_mod.special_id(tok, "</think>"), think.end_text)
                        for x in (*more, ids))
                if (prefill and cached_ids is None and not micro
                        and STATE.get("last_prefill") is not prefill_before):
                    rec.absorb_prefill(STATE.get("last_prefill"))

            if ui is not None:
                ui.head = len(ui.log)             # the steps before the generation started
            if not stream:
                done: list[dict] = []
                for ci in range(n_choices):
                    if ci:
                        # `n`: the next choice has its own draws (a seeded request, its own
                        # seed, so every choice reproduces), guard, budget and recorder; the same
                        # prompt, rules and deadline. The engine holds one sequence, so choices
                        # are generated one after another, and never replayed from the cache.
                        pstop = (PatternStop(*STATE["pattern_stop"])
                                 if STATE.get("pattern_stop") else None)
                        think = ThinkBudget(tok, budget, arms=arms)
                        rec.live_refs = (think, None)
                        lpr = lp_mod.Recorder(lp_top) if lp_top is not None else None
                        source = open_source(sampler.for_choice(ci), think, pstop, lpr, None)
                    earlier = tuple(d["ids"] for d in done)
                    ids = []
                    try:
                        for t in source:
                            ids.append(t)
                    except ClientGone:
                        settle(ids, "abandoned", more=earlier, prefill=not ci)
                        _log_request(cid, n_prompt, len(ids), "abandoned", t_req, stream=False,
                                     pen=pen_spec, rec=rec, temp=sampler.temperature)
                        raise
                    except Exception as exc:                          # noqa: BLE001
                        # The [req] line and the `errors` count, as the streamed path has them
                        #; `do_POST` still answers the 500 and prints the traceback.
                        settle(ids, "error", exc, more=earlier, prefill=not ci)
                        _log_request(cid, n_prompt, len(ids), "error", t_req, stream=False,
                                     exc=exc, pen=pen_spec, rec=rec, temp=sampler.temperature)
                        raise
                    last_choice = ci == n_choices - 1
                    if last_choice:
                        _finishing(rec, "saving_state")
                    # 2026-10-05: the answer's memory blocks, written and taken out of the text
                    mem_taken = _memory_full(plan, chat, n_choices, ui, ids, in_think, n_prompt,
                                             comp is None and mm is None)
                    if (cached_ids is None or ci) and not micro:
                        _remember(client_ids or prompt_ids, ids, conv_id, mm=mm,
                                  hidden=hidden_spans + mem_taken[1],
                                  delta=comp["record"].delta if comp is not None else 0,
                                  served_hidden=(rec.hidden_at if rec.hidden_at is not None
                                                 else getattr(rec, "note_at", None)))
                        if rkey is not None and not ci and not (sup is not None and sup.acted) \
                                and not mem_taken[0]:
                            rcache.put(rkey, ids, prompt_ids)
                    if shadow_key is not None and not ci:
                        shadow_svc.end_turn(shadow_key, ids)       # SR1
                    text = tok.decode(ids, skip_special_tokens=True)
                    if pstop is not None and pstop.hit:
                        finish = "stop"
                    else:
                        finish = "stop" if (ids and ids[-1] in eos) else (
                            "timeout" if deadline.hit else "length")
                    if answer_side:
                        text, cut = answer_stops(text, stops, in_think)     # SRV-245, SRV-617
                    else:
                        text, cut = _apply_stops(text, stops)
                    if cut:
                        finish = "stop"
                    if mem_taken[0]:
                        head, answer = _reasoning_head(text, in_think)
                        for raw in mem_taken[0]:
                            answer = answer.replace(raw, "", 1)
                        text = head + answer
                    calls, dropped = [], 0
                    if parse_calls:
                        # Only the answer can call a tool: a call the model writes inside
                        # its reasoning is a thought about calling, not a call.
                        head, answer = _reasoning_head(text, in_think)
                        ended = bool(ids) and ids[-1] in eos
                        if STATE.get("tool_format") == "pythonic":
                            answer = rewrite_pythonic(answer, eos=ended)       # SRV-425
                        if sup is not None and sup.holds_calls and last_choice:
                            # SP3: the held calls as the checks decided; what came before a
                            # regeneration (`sup.cuts`, in tokens) lost its blocks
                            cuts = [len(tok.decode(ids[:k], skip_special_tokens=True)) - len(head)
                                    for k in sup.cuts]
                            answer, calls = _sp3_calls(sup, answer, ended, tool_names or None,
                                                       tool_types, [c for c in cuts if c >= 0],
                                                       stopped=cut)
                        else:
                            answer, calls = parse_tool_calls(answer, names=tool_names or None,
                                                             eos=ended, types=tool_types)
                            if sup is not None and last_choice and not sup.checked and calls:
                                # shadow: an answer cut short is still checked, and journaled
                                sup.check_answer(calls, final=True)
                        if refuse_calls and calls:
                            _refused(plan, cid, calls)
                            calls = []
                        if max_calls is not None and len(calls) > max_calls:
                            calls, dropped = calls[:max_calls], len(calls) - max_calls
                        text = head + answer
                        if calls and finish == "stop":
                            finish = "tool_calls"
                    if pstop is not None and pstop.hit:
                        text += GUARD_MARKER
                    if sup is not None and sup.gave_up and last_choice:
                        # GIVE_UP: one visible sentence from the policy, finish stop
                        opened = in_think and CLOSE_THINK not in text
                        text += (CLOSE_THINK + "\n\n" if opened else "") + sup.gave_up
                        finish = "stop"
                    settle(ids, finish, calls=len(calls) + sum(len(d["calls"]) for d in done),
                           more=earlier, prefill=not ci)
                    rec.stop_detail = _stop_detail(finish, ids, eos, cut, pstop)
                    if calls:
                        rec.tool_names = (rec.tool_names or []) + [
                            c["function"]["name"] for c in calls]
                        if transcribe:
                            rec.out_calls = calls
                    if sup is not None and last_choice:
                        _outcome(sup, calls, ids, in_think, finish)
                    _log_request(cid, n_prompt, len(ids), finish, t_req, stream=False,
                                 pen=pen_spec,
                                 pattern=(pstop.label if pstop is not None and pstop.hit else None),
                                 rec=rec, temp=sampler.temperature,
                                 tools=_tools_note(body, len(calls) + dropped, dropped),
                                 sup=sup if last_choice else None)
                    done.append({"ids": ids, "text": text, "finish": finish, "calls": calls,
                                 "lpr": lpr, "pstop": pstop})
                # `usage` with its details, and the top-level `timings` and `metrics`.
                # Open WebUI reads only `usage` on this path; the other two are for the rest.
                fields = rec.fields()
                fmt_lp = lp_mod.Formatter(tok) if lp_top is not None else None
                choices = []
                # SRV-625: a secret from a tool result is not repeated in the answer
                sm = _answer_mask(plan, chat, body)
                for ci, d in enumerate(done):
                    entries = (lp_mod.emitted(d["lpr"].entries, d["ids"], eos)
                               if d["lpr"] is not None else None)
                    if chat:
                        content, reasoning = split_full(d["text"], fmt, in_think=in_think)
                        if plan is not None and plan.active:
                            content, n_stray = drop_stray_close(content, fmt, in_think)  # SRV-187
                            if n_stray and plan.row is not None:
                                plan.row["extra"]["stray_close"] = n_stray
                        if sm is not None:
                            # the answer and its reasoning, never the calls' arguments
                            content = sm.text(content, "content")
                            reasoning = sm.text(reasoning, "reasoning_content")
                        if ui is not None and ui.log and not (plain_lines
                                                               and fmt != "reasoning_content"):
                            # the steps before the generation at the reasoning's head, the rest
                            # at its end (a non-streamed answer has no moment to show them at);
                            # a plain API client's never in its content (SRV-245)
                            content, reasoning = states_mod.into_full(
                                content, reasoning, fmt, in_think, ui,
                                "reasoning" if plain_lines else lines_in)
                        elif ui is not None and ui.log:
                            content += "".join(ui.shown(e) for e in ui.log    # SRV-185's
                                               if e["kind"] in states_mod.MEMORY_KINDS)
                        message = {"role": "assistant", "content": content}
                        if d["calls"]:
                            message["tool_calls"] = d["calls"]
                        if reasoning is not None:
                            message["reasoning_content"] = reasoning
                        choices.append({"index": ci, "finish_reason": d["finish"],
                                        "logprobs": ({"content": fmt_lp.chat(entries)}
                                                     if entries is not None else None),
                                        "message": message})
                    else:
                        choices.append({"index": ci, "finish_reason": d["finish"],
                                        "logprobs": (fmt_lp.legacy(entries)
                                                     if entries is not None else None),
                                        "text": d["text"]})
                _answer_mask_done(plan, cid, sm)
                payload = {"id": cid, "object": "chat.completion" if chat else "text_completion",
                           "created": created, "model": model, "usage": fields["usage"],
                           "choices": choices}
                payload["timings"], payload["metrics"] = fields["timings"], fields["metrics"]
                hit = next((d["pstop"] for d in done
                            if d["pstop"] is not None and d["pstop"].hit), None)
                _finishing(rec, "final_chunk")
                return self._json(200, payload, extra_headers=_guard_headers(hit) + (
                    plan.headers() if plan is not None else ()))

            if self._early is None:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                for k, v in (plan.headers() if plan is not None else ()):
                    self.send_header(k, v)             # X-Tandem and X-Tandem-Profile
                if getattr(self, "_mem_wait", False):
                    self.send_header("X-Tandem-Wait", "memory")      # SRV-305
                self.end_headers()
            self._early_body = True                    # S1: a stream opened early goes on here
            w = self.wfile
            voice = None                               # v3, made below (send() reads it)
            self._streamed = True
            # One piece of text at a time, and every character in it final: `Detokenizer` holds a
            # code point back until all of its bytes have arrived, which is bug 3. `Reasoning` says
            # which field each piece belongs in, which is the second half of bug 1.
            det = Detokenizer(lambda seq: tok.decode(seq, skip_special_tokens=True))
            split = Reasoning(fmt, in_think=in_think, boundary=fmt_boundary)
            # a tail that could still become a stop string waits until it does or cannot;
            # the stop string is never sent, not even the part of it that arrives first.
            stopper = StopStrings(stops)
            # SRV-245: under a conductor plan the stop strings read the answer alone (`AnswerStops`,
            # after the split), and so they do for a plain client on the raw path (SRV-617); every
            # other raw request matches them on the text as the release did
            astop = AnswerStops(stops) if answer_side and stops else None
            if astop is not None:
                stopper = astop.stopper            # one `hit` for the loop and the logs

            # with stop strings, no call streams live: one can cut a block whose arguments were
            # already sent, and the client is left with a call that is not JSON
            tbuf = (ToolCallBuffer(names=tool_names or None, max_calls=max_calls, types=tool_types,
                                   hold=(sup is not None and sup.holds_calls) or refuse_calls,
                                   live=not any(stops))
                    if parse_calls else None)
            # SRV-425: a model that writes Pythonic calls (LFM2: `<|tool_call_start|>[f(a=1)]...`) has
            # each block rewritten into the buffer's JSON tier as it arrives (server/toolcall.py)
            pyc = PythonicCalls() if tbuf is not None and STATE.get("tool_format") == "pythonic" else None
            rec.live_refs = (think, tbuf)
            # 2026-10-05: the model's memory blocks held back from the client and written
            # (server/memblock.py, server/conductor/memory.py); None with the profile's memory off
            mf = [_memory_filter(plan, chat, n_choices, ui)]
            # SRV-187: a bare `</think>` line the model writes in its answer never reaches the
            # client (a conductor plan only: the raw path is the release's, byte for byte)
            stray = [StrayClose() if chat and plan is not None and plan.active else None]
            # SRV-625: a secret from a tool result of the conversation is not repeated in what the
            # client sees (the content after the tool-call buffer, the reasoning); None when the
            # profile's mask is off or no tool result holds one
            amask = _answer_mask(plan, chat, body)
            # The tool-argument loop guard reads the open call's arguments
            aguard = (argloop_mod.ArgLoopGuard(
                context_chars=argloop_mod.longest_message_chars(body.get("messages")))
                if tbuf is not None and STATE.get("argloop", "on") != "off" else None)
            aloop = None                         # its verdict, once it fired
            # BUG 1. The prompt ended inside `<think>`, so the opening tag is already spent and the
            # model will only ever write the closing one. Put it back as the start of `content`.
            # NOT ahead of the loop. `source` is a generator and the prefill runs inside
            # its first `next()`, so a tag sent before the loop reached the client ahead of the
            # model and every client timing its first content delta read the round trip (3 ms on
            # the box, against a 551 ms request). It rides on the first piece of text instead:
            # the client's first content delta is the model's first token, and still opens with
            # the tag.
            opener = [OPEN_THINK + "\n"] if in_think and fmt in ("tags", "both") else []
            #, the same mistake with the other chunk: the role chunk went out before the
            # loop, and a client that stamps TTFT on the first chunk with a `choices` array
            # (vLLM's bench client does, whatever the chunk holds) read the HTTP round trip. It
            # now goes out in the same write as the first chunk after it -- the first text, or the
            # finish chunk of a stream that has none -- and is still the first chunk, unchanged.
            role = ([_chunk(cid, model, created, {"role": "assistant", "content": ""})]
                    if chat and not (self._early is not None and self._early["role"]) else [])

            def put(data: str) -> None:
                w.write(((role.pop() if role else "") + data).encode())

            def mask_flush(key: str) -> None:
                """SRV-625: the field's text ended (a call follows, the reasoning closed, the
                answer ended): what the mask still holds of it goes out, masked."""
                if amask is None or not amask.held(key):
                    return
                rest = amask.finish(key)
                if rest:
                    put(_chunk(cid, model, created, {key: rest},
                               **(lp_take() if lpr is not None else NO_LP)))

            # a chunk carries the log-probabilities of the tokens decided since the last
            # chunk that carried some -- text can lag its tokens (a held character, a stop-string
            # prefix, a tool block), so a chunk may carry none or several; the finish chunk the
            # rest.
            fmt_lp = lp_mod.Formatter(tok) if lpr is not None else None
            lp_at = [0, 0]                       # entries sent, characters of text they covered

            def lp_take() -> dict:
                """`logprobs=` for the next chunk; a request that did not ask passes NO_LP."""
                got = lp_mod.emitted(lpr.entries, ids, eos)[lp_at[0]:]
                lp_at[0] += len(got)
                if chat:
                    return {"logprobs": {"content": fmt_lp.chat(got)}}
                out = fmt_lp.legacy(got, lp_at[1])
                lp_at[1] += sum(len(t) for t in out["tokens"])
                return {"logprobs": out}

            def tools_note() -> str | None:
                if tbuf is None:
                    return _tools_note(body, 0)
                return _tools_note(body, len(tbuf.calls) + tbuf.dropped, tbuf.dropped)

            def send(pairs) -> None:
                for field, piece in pairs:
                    if not piece:
                        continue
                    if stray[0] is not None and field == "content" and not split.in_think:
                        piece = stray[0].push(piece)     # SRV-187: no bare `</think>` line
                        if not piece:
                            continue
                    if mf[0] is not None and field == "content" and not split.in_think:
                        piece = mf[0].feed(piece)        # a memory block stays on the server
                        if not piece:
                            continue
                    if held and field == "content" and not split.in_think:
                        say_held()                       # right after the block closed
                    if opener and field != "reasoning":
                        piece = opener.pop() + piece
                    if tbuf is not None and field == "content":
                        # Content is routed through the buffer so a tool-call block is held back
                        # instead of being shown as raw XML; a recognised call streams
                        # its arguments live as OpenAI deltas through the same feed (the rolex_svg
                        # fix: a whole-file call used to arrive in one lump at the very end).
                        for out_piece in tbuf.feed(pyc.feed(piece) if pyc is not None else piece):
                            if amask is not None:
                                if not split.in_think:
                                    mask_flush("reasoning_content")
                                out_piece = amask.push(out_piece, "content")
                                if not out_piece:
                                    continue
                            if voice is not None:
                                voice.content(out_piece)       # v3: a plan in the answer
                            put(_chunk(cid, model, created, {"content": out_piece},
                                       **(lp_take() if lpr is not None else NO_LP)))
                        deltas = tbuf.drain_deltas()
                        if deltas:
                            mask_flush("content")              # the text before the call ended
                        for d in deltas:
                            put(_chunk(cid, model, created, {"tool_calls": [d]},
                                       **(lp_take() if lpr is not None else NO_LP)))
                        w.flush()
                        continue
                    if chat:
                        key = "reasoning_content" if field == "reasoning" else "content"
                        if amask is not None:
                            if key == "content" and not split.in_think:
                                mask_flush("reasoning_content")
                            piece = amask.push(piece, key)
                            if not piece:
                                continue
                        if voice is not None and key == "content":
                            voice.content(piece)                # v3: a plan in the answer
                        put(_chunk(cid, model, created, {key: piece},
                                   **(lp_take() if lpr is not None else NO_LP)))
                    else:
                        put(_text_chunk(cid, model, created, piece,
                                        **(lp_take() if lpr is not None else NO_LP)))
                    w.flush()

            # S1: a line in the content while the content holds an open `<think>` block (`tags`,
            # `both`) waits for the block to close: outside it there is no place for it that the
            # client sends back as it was (Hermes keeps every block's reasoning, joined with a
            # blank line). Only the newest thinking line waits; the others are out of date.
            held: list[tuple[str, str]] = []

            def say(kind: str, line: str) -> None:
                """One state line, in the profile's channel."""
                if bar is not None and bar.live and kind not in states_mod.MEMORY_KINDS:
                    bar_out([(kind, line)], w.write)    # SRV-237: before the first token
                    w.flush()
                    return
                if plain_lines and kind not in states_mod.MEMORY_KINDS:
                    if fmt == "reasoning_content":
                        send([("reasoning", line)])   # SRV-245: its own field, not the answer
                    return
                if lines_in == "inline":
                    # X1: an empty `tool_calls` delta ends opencode's open reasoning part, the
                    # line opens the next one as its title (states.INLINE); never the tool-call
                    # buffer's, never content
                    put(_chunk(cid, model, created, {"tool_calls": []}))
                    put(_chunk(cid, model, created, {"reasoning_content": states_mod.inline(line)}))
                    w.flush()
                    return
                if lines_in != "content" and kind in states_mod.MEMORY_KINDS:
                    # SRV-185: a note written, changed or forgotten is the user's data: its line
                    # goes into the answer, where every client shows it (`tags` has no place for
                    # a line after the block, and a reasoning line after the answer is not seen)
                    put(_chunk(cid, model, created, {"content": line}))
                    w.flush()
                    return
                if lines_in != "content":
                    send(split.state(line))              # where the reasoning goes now
                    return
                if fmt != "reasoning_content" and split.in_think and not opener:
                    if kind == "thinking":
                        held[:] = [h for h in held if h[0] != "thinking"]
                    held.append((kind, line))
                    return
                # a content delta of its own: not the tool-call buffer's, not the opener's
                put(_chunk(cid, model, created, {"content": line}))
                w.flush()

            def say_held() -> None:
                lines, held[:] = list(held), []
                for kind, line in lines:
                    say(kind, line)

            def tell() -> None:
                """The state lines waiting since the last piece."""
                if ui is not None and ui.pending:
                    for kind, line in ui.take_kinds():
                        say(kind, line)
                if held and not split.in_think:
                    say_held()

            # S1: the progress lines of a long prefill (called by the chunk hook) and a long
            # thinking (from the loop), with the numbers, while the client sees nothing else
            pf_next: list = [None]

            def prefill_line(done: int, total: int, now: float) -> None:
                if pf_next[0] is not None and now < pf_next[0]:
                    return
                w = watch.info.get("window")
                if w is not None and done < w[1]:
                    # VIS-76: inside a layer-major window the newest mark is a chunk of it at the
                    # window's time, ~10 s after the last window's: the rate read 1,024 rows in
                    # 10 s ("about 4 min left" for a 47 s prefill); its last chunk reads it whole
                    return
                tps = activity_mod.prefill_live(rec, 1)["tps_live_avg"]
                if not tps:
                    return
                left, every = (total - done) / tps, float(prog["prefill_every_s"])
                if pf_next[0] is None and left < every:
                    pf_next[0] = float("inf")            # not a long prefill: no line
                    return
                start = int(watch.info.get("start") or 0)
                ui.add("prefill", n=f"{total - start:,}", rate=states_mod.rate(tps),
                       left=states_mod.duration(left))
                tell()
                pf_next[0] = now + every

            if live_lines:
                watch.say = prefill_line
            # v3: the voice has its own thinking lines (and the template only after a silence)
            use_voice = ui is not None and ui.voice is not None and STATE.get("narrator") is not None
            think_at = ({"t0": None, "s": float(prog["thinking_every_s"]),
                         "n": int(prog["thinking_every_tokens"])}
                        if live_lines and in_think and not use_voice else None)
            # N2: `ui.narrator: thoughts` -- the narrator sums up each stretch of thinking from
            # Qwen's own ids (server/conductor/narrator.py Thoughts)
            narr = STATE.get("narrator")
            # SR1: `shadow.narrator: on` -- the summary is read from the session's shadow
            reader = (shadow_svc.thought_slot(shadow_key)
                      if shadow_key is not None and shadow_cfg["narrator"] == "on" else None)
            # v3: one share of Qwen's time for the request's small-model work: the voice first,
            # the shadow's in-turn reading after it (narrator.Share)
            share = narrator_mod.Share(float((ui_doc.get("live") or {}).get(
                "share", voice_mod.DEFAULTS["share"]))) if use_voice else None
            sh_inline = (shadow_svc.inline(shadow_key, budget=share)
                         if shadow_key is not None else None)
            voice = (narr.for_voice(ui, lang=ui.voice, opened=bool(in_think),
                                    close_id=tok.convert_tokens_to_ids(CLOSE_THINK),
                                    decode=lambda xs: tok.decode(xs, skip_special_tokens=True),
                                    share=share, settings=ui_doc.get("live"))
                     if use_voice and stream else None)
            if use_voice and voice is None and live_lines and in_think:
                # the role is off: the templates' thinking lines after all
                think_at = {"t0": None, "s": float(prog["thinking_every_s"]),
                            "n": int(prog["thinking_every_tokens"])}
            thoughts = (narr.for_thoughts(ui, opened=bool(in_think),
                                          close_id=tok.convert_tokens_to_ids(CLOSE_THINK),
                                          decode=lambda xs: tok.decode(xs,
                                                                       skip_special_tokens=True),
                                          reader=reader)
                        if narr is not None and ui is not None
                        and ui_doc.get("narrator") == "thoughts" else None)

            ids: list[int] = []
            finish = "length"
            failed: BaseException | None = None
            # reset per stream; the non-streamed branch returns before this point
            cut = False
            cuts = sup.cuts if sup is not None and tbuf is not None and tbuf.hold else None
            def bar_close() -> None:
                """SRV-237: the status line done, at the first generated token: the tokens read
                and how long, the time waited."""
                marks = rec.pf_marks
                t0 = watch.info.get("t0")
                read = ((int(marks[-1][1]) - int(watch.info.get("start") or 0),
                         time.perf_counter() - t0) if marks and t0 else None)
                ev = bar.close(read=read, waited=(rec.queue_ms or 0.0) / 1000.0)
                if ev is not None:
                    w.write(_chunk(cid, model, created, None, extra={"event": ev}).encode())
                    w.flush()

            try:
                tell()                                   # the router's, before the prefill runs
                for t in source:
                    if bar is not None and bar.live:
                        bar_close()              # SRV-237: the reasoning opens with this token
                    tell()                       # what the monitor did before this token came
                    if cuts and len(ids) >= cuts[0]:
                        # SP3 had the model write this answer's calls again: what the buffer
                        # held of the answer before is dropped, and nothing of it was sent
                        cuts.pop(0)
                        tbuf.cut()
                    ids.append(t)
                    if think_at is not None and split.in_think:
                        now = time.perf_counter()
                        if think_at["t0"] is None:
                            think_at["t0"] = now             # the thinking's first token
                        el = now - think_at["t0"]
                        if el >= think_at["s"] or len(ids) >= think_at["n"]:
                            # N2: the narrator's summary of the thought so far, when it can
                            if thoughts is None or not thoughts.mark(ids, el):
                                ui.add("thinking", elapsed=states_mod.duration(el),
                                       n=f"{len(ids):,}")
                            think_at["s"] = el + float(prog["thinking_every_s"])
                            think_at["n"] = len(ids) + int(prog["thinking_every_tokens"])
                    if t in eos:
                        finish = "stop"
                        break
                    piece = det.push(ids)
                    if not piece:
                        continue                   # a byte-level token that is not a character yet
                    if astop is not None:
                        send(astop.push(split.push(piece)))
                    else:
                        send(split.push(stopper.push(piece)))
                    if stopper.hit:
                        finish, cut = "stop", True
                        break
                    if aguard is not None:
                        blk = tbuf.open_text()
                        v = aguard.feed(len(ids), None if blk is None else len(blk),
                                        tbuf.open_text)
                        if v is not None and _argloop_fired(v, tbuf, sup, rec, cid):
                            aloop, finish, cut = v, "stop", True
                            break
                    if thoughts is not None:
                        thoughts.token(ids, split.in_think)    # N2: between two of Qwen's rounds
                    if voice is not None:
                        voice.token(ids, split.in_think)       # v3: between two of Qwen's rounds
                        if tbuf is not None:
                            voice.tool(ids, tbuf.open_text())
                    if sh_inline is not None:
                        sh_inline.tick(ids)        # SR1: the shadow reads in its share
                _finishing(rec, "flush")                 # the token loop has ended
                if bar is not None and bar.live:
                    bar_close()                          # no token at all
                if not cut:
                    if astop is not None:
                        send(astop.push(split.push(det.flush(ids))) + astop.finish())
                    else:
                        send(split.push(stopper.push(det.flush(ids)) + stopper.finish()))
                    if stopper.hit:
                        finish = "stop"
                if stray[0] is not None:
                    sc, stray[0] = stray[0], None
                    send([("content", sc.finish())])
                    if sc.dropped and plan.row is not None:
                        plan.row["extra"]["stray_close"] = sc.dropped
                if mf[0] is not None:
                    # an unclosed block goes out as the text it was (the honest fallback)
                    mem_filter, mf[0] = mf[0], None
                    send([("content", mem_filter.finish())])
                    mf.append(mem_filter)
                if thoughts is not None:
                    thoughts.finish(ids)                 # N2: what is left, before the calls
                    rec.thoughts = thoughts.totals()
                if voice is not None:
                    voice.finish(ids)                    # v3: what is left, before the calls
                    rec.voice = voice.totals()
                tell()
                if tbuf is not None:
                    # The held text goes out RAW, not through `send()`. `send()` feeds content
                    # back into the same buffer, and what is held still contains the opener: the
                    # block re-opened, the arguments went out a second time at a new index, and
                    # the fallback text was swallowed -- for a bare partial opener ("hello
                    # <tool"), all of it. The sweep that follows carries the calls that were not
                    # streamed live.
                    while cuts and len(ids) >= cuts[0]:
                        cuts.pop(0)                      # regenerated, and nothing came after
                        tbuf.cut()
                    if pyc is not None:
                        # SRV-425: a Pythonic block still open: a call at the end token, text if cut
                        for out_piece in tbuf.feed(pyc.finish(eos=bool(ids) and ids[-1] in eos)):
                            put(_chunk(cid, model, created, {"content": out_piece},
                                       **(lp_take() if lpr is not None else NO_LP)))
                    left, sweep = tbuf.finish(eos=bool(ids) and ids[-1] in eos)
                    if tbuf.cut_live:
                        # 2026-10-05: the client holds the open call as it was streamed; its text is not
                        # sent a second time as content (the next prompt would carry it twice)
                        print(f"[toolcall] {cid} ended inside a live call ({finish}): the block's "
                              "text is not sent again as content", flush=True)
                    if tbuf.hold and refuse_calls and not (sup is not None and sup.holds_calls):
                        # SRV-247: held because the request offered no tools; never sent
                        _refused(plan, cid, [c for c in tbuf.take_held() if "bad" not in c])
                        tbuf.cut_block, sweep = None, []
                    elif tbuf.hold:
                        left += _sp3(sup, tbuf, stopped=stopper.hit)
                        sweep = tbuf.drain_deltas() + sweep
                        tell()                           # the checks of the held calls
                    elif sup is not None and not sup.checked and tbuf.calls:
                        # shadow: an answer cut short is still checked, and journaled
                        sup.check_answer(tbuf.calls, final=True)
                    if refuse_calls and (tbuf.calls or sweep):
                        _refused(plan, cid, tbuf.calls)
                        tbuf.calls, sweep = [], []
                    if amask is not None:
                        # the held text is the answer's too; before a call, the text has ended
                        left = amask.push(left, "content") + (amask.finish("content")
                                                              if sweep else "")
                    if left:
                        put(_chunk(cid, model, created, {"content": left},
                                   **(lp_take() if lpr is not None else NO_LP)))
                    for delta in sweep:
                        put(_chunk(cid, model, created, {"tool_calls": [delta]},
                                   **(lp_take() if lpr is not None else NO_LP)))
                    if left or sweep:
                        w.flush()
                    if tbuf.calls and finish == "stop":
                        finish = "tool_calls"
                    if tbuf.calls:
                        rec.tool_names = [c["function"]["name"] for c in tbuf.calls]
                        if transcribe:
                            rec.out_calls = tbuf.calls
                if aloop is not None:
                    # the stopped call's place: one visible sentence, then the finish chunk
                    put(_chunk(cid, model, created, {"content": "\n\n" + aloop.text})
                        if chat else _text_chunk(cid, model, created, aloop.text))
                    w.flush()
                if pstop is not None and pstop.hit:
                    send([("content", GUARD_MARKER)])
                send(split.finish())
                if amask is not None:
                    mask_flush("reasoning_content")      # SRV-625: the answer ended
                    mask_flush("content")
                    _answer_mask_done(plan, cid, amask)
                tell()
                if sup is not None and sup.gave_up:
                    # GIVE_UP: one visible sentence from the policy, as content, finish stop. An
                    # open reasoning block is closed first, so a client folding on tags shows it.
                    text = sup.gave_up
                    if split.in_think and fmt in ("tags", "both"):
                        text = CLOSE_THINK + "\n\n" + text
                    put(_chunk(cid, model, created, {"content": (opener.pop() if opener else "")
                                                     + text}) if chat
                        else _text_chunk(cid, model, created, text))
                    w.flush()
                    finish = "stop"
                if pstop is not None and pstop.hit and finish == "length":
                    finish = "stop"
                elif deadline.hit and finish == "length":
                    finish = "timeout"
            except (BrokenPipeError, ConnectionResetError):
                # The reader hung up -- a closed pipe and a reset connection are the same event
                # seen from two kernels, and neither is this server's fault. There is nothing to
                # report and nowhere to report it.
                if rec.client_gone_at is None:
                    rec.client_gone_at = time.perf_counter()
                settle(ids, "abandoned", calls=len(tbuf.calls) if tbuf is not None else 0)
                _log_request(cid, n_prompt, len(ids), "abandoned", t_req, stream=True, pen=pen_spec,
                             pattern=(pstop.label if pstop is not None and pstop.hit else None),
                             rec=rec, temp=sampler.temperature, tools=tools_note())
                raise
            except Exception as exc:                                  # noqa: BLE001
                # BUG 2, the third half. The headers of a stream go out before the first token, so
                # an exception raised half way through used to unwind into `do_POST`, which tried
                # to send a 500 -- a JSON body, with its own status line, appended to a live
                # event stream. Every client on earth reads that as a stream that simply stopped.
                # It is logged here, and the stream is CLOSED PROPERLY: the partial text the reader
                # already has, then a finish reason that says what happened.
                failed = exc
                finish = "error"
                logbuf.print_exc()
            mem_taken = _memory_close(plan, mf[-1] if len(mf) > 1 else mf[0], ids, n_prompt,
                                      comp is None and mm is None)
            if cached_ids is None and failed is None and not micro:
                _finishing(rec, "saving_state")
                _remember(client_ids or prompt_ids, ids, conv_id, mm=mm,
                          hidden=hidden_spans + mem_taken[1],
                          delta=comp["record"].delta if comp is not None else 0,
                          served_hidden=(rec.hidden_at if rec.hidden_at is not None
                                                 else getattr(rec, "note_at", None)))
                if rkey is not None and not (sup is not None and sup.acted) and \
                        not mem_taken[0]:
                    rcache.put(rkey, ids, prompt_ids)
            if shadow_key is not None and failed is None:
                shadow_svc.end_turn(shadow_key, ids)           # SR1
            if sh_inline is not None and plan is not None and plan.row is not None:
                plan.row.setdefault("extra", {}).setdefault("shadow", {})["inline"] = \
                    sh_inline.totals()
            settle(ids, finish, failed, calls=len(tbuf.calls) if tbuf is not None else 0)
            rec.stop_detail = _stop_detail(finish, ids, eos, stopper.hit, pstop)
            if aloop is not None:
                rec.stop_detail = "tool_arg_loop"
            if sup is not None:
                _outcome(sup, tbuf.calls if tbuf is not None else [], ids, in_think, finish)
            _log_request(cid, n_prompt, len(ids), finish, t_req, stream=True, exc=failed, pen=pen_spec,
                         pattern=(pstop.label if pstop is not None and pstop.hit else None),
                         rec=rec, temp=sampler.temperature, tools=tools_note(), sup=sup)
            # usage, timings and metrics on exactly one chunk -- the finish chunk by
            # default, the separate `choices: []` chunk when the client asked for include_usage.
            fields = rec.fields() if where in ("finish", "separate") else None
            on_finish = fields if where == "finish" else None
            _finishing(rec, "final_chunk")
            try:
                if opener:
                    # No text at all -- a stop string at the first character, or a failure before
                    # the first token. The block still opens, as the non-streamed answer's does.
                    put(_chunk(cid, model, created, {"content": opener.pop()}))
                if failed is not None and chat:
                    put(_chunk(cid, model, created, {}, finish=finish,
                               error={"message": str(failed), "type": type(failed).__name__},
                               extra=on_finish, **(lp_take() if lpr is not None else NO_LP)))
                else:
                    put(_chunk(cid, model, created, {}, finish=finish, extra=on_finish,
                               **(lp_take() if lpr is not None else NO_LP)) if chat
                        else _text_chunk(cid, model, created, "", finish=finish, extra=on_finish,
                                         **(lp_take() if lpr is not None else NO_LP)))
                if where == "separate":
                    w.write(_chunk(cid, model, created, None, extra=fields).encode())
                w.write(b"data: [DONE]\n\n")
                w.flush()
            except BrokenPipeError:
                pass
        finally:
            if micro:
                # the side lane never held the engine past its job (server/preempt.py)
                with QUEUE:
                    INFLIGHT["lane"] -= 1
                    INFLIGHT["served"] += 1
            else:
                # the allocator's reserve over its slack goes back to the driver, with the answer
                # already sent and before the next request may take the engine (OPS-47)
                if STATE.get("mem") is not None:
                    try:
                        STATE["mem"].done()
                    except Exception:                              # noqa: BLE001
                        logbuf.print_exc()
                # ENG-371: the memory gate's hold of an end anchor ends with its request, whether
                # or not it prefilled (A1's give-up, a prepared summary, a cache hit, a client gone)
                res_h = STATE.get("resident")
                if res_h is not None:
                    res_h.end_hold = False
                # SRV-307: the pages past what this slot keeps go back to the pool, before the
                # slot is given to the next request
                kv_trim_current()
                if (plan is not None and getattr(plan, "compaction", False)
                        and STATE.get("conductor") is not None):
                    # P-M2: a compaction marks its session for the memory writer, and with
                    # `after_compaction: turn` the writer runs now, before the next request
                    try:
                        _writer_after_compaction(plan, conv_id)
                    except Exception:                              # noqa: BLE001
                        logbuf.print_exc()
                elif getattr(getattr(rec, "sup", None), "gave_up", None) and \
                        getattr(STATE.get("conductor"), "memory", None) is not None:
                    STATE["conductor"].memory.writer.mark(conv_id, "give_up")
                if par is not None:
                    if side is None and not micro:
                        # SRV-197: the conversation slot 0 now holds, as its client renders it
                        try:
                            ctx_ = STATE.get("last_ctx")
                            n_ = int(STATE.get("last_prompt_n") or 0)
                            if ctx_ is not None and ctx_ is not ctx_found and n_:
                                # SRV-629: with images, the rows and the view in key space
                                _par_view_keep(_keyed(list(ctx_[:n_]), mm_end),
                                               par_why.get("ids"))
                        except Exception:                          # noqa: BLE001
                            logbuf.print_exc()
                    elif side is not None and not micro and not getattr(side, "no_carry", False):
                        # FX16: the conversation the side slot now holds (its resident prefix)
                        try:
                            ctx_ = STATE.get("last_ctx")
                            n_ = int(STATE.get("last_prompt_n") or 0)
                            if ctx_ is not None and n_:
                                _side_view_keep(side, _keyed(list(ctx_[:n_]), mm_end),
                                                par_why.get("ids"))
                        except Exception:                          # noqa: BLE001
                            logbuf.print_exc()
                    if par_why.get("hop_from") is not None and \
                            (side is None or side.index != par_why["hop_from"]):
                        # SRV-528: what the hop cost, in tokens this slot read again
                        fw = int((STATE.get("last_prefill") or {}).get("forwarded") or 0)
                        with par._lk:
                            par.stats["hop_tokens"] = par.stats.get("hop_tokens", 0) + fw
                    _par_end(par, side, cid)
                if side is None:
                    with QUEUE:
                        HOLDER["who"] = None
                    LOCK.release()
                STATE["last_end_mono"] = time.monotonic()   # the preparer's idle clock
                if me.get("t_hold") is not None:
                    # how long it held the engine, for the waiting line's estimate
                    CLOCK.done(me["rank"], STATE["last_end_mono"] - me["t_hold"])
                    if not me["own"] and rec.client_kind != "internal":
                        _clock_wrote(rec)              # SRV-249: the tokens it wrote, its rate
                with QUEUE:
                    INFLIGHT["running"] -= 1
                    INFLIGHT["served"] += 1


def _memory() -> dict:
    """What a soak run watches for a leak.

    On this board the GPU and the host share one pool and `nvidia-smi` reports N/A for used
    memory, so the numbers that mean anything are the allocator's own -- `allocated` is what the
    engine is holding and `reserved` is what it has taken from the driver and not given back. A
    leak shows as `allocated` climbing across requests; fragmentation shows as `reserved` climbing
    while `allocated` does not.
    """
    out = {}
    mem = STATE.get("mem")
    try:
        out["allocated_gb"] = round(torch.cuda.memory_allocated() / 2**30, 3)
        out["reserved_gb"] = round(torch.cuda.memory_reserved() / 2**30, 3)
        out["max_allocated_gb"] = round((mem.lifetime() if mem is not None
                                         else torch.cuda.max_memory_allocated()) / 2**30, 3)
    except Exception as exc:                                      # noqa: BLE001
        out["error"] = str(exc)
    if mem is not None and mem.on:
        # OPS-47: what ops/roles.sh hands the residency planner -- the engine's declared peak and
        # what it holds now, the decision models (roles of their own) left out
        try:
            alloc, res = mem.memory()
            out.update(ceiling_gb=round(mem.ceiling / 2**30, 3), peak_gb=round(mem.peak / 2**30, 3),
                       engine_reserved_gb=round(res / 2**30, 3),
                       engine_allocated_gb=round(alloc / 2**30, 3))
        except Exception:                                          # noqa: BLE001
            pass
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    out["rss_gb"] = round(int(line.split()[1]) / 2**20, 3)
                    break
    except Exception:
        pass
    # The interpreter singletons' reference counts now (server/refwatch.py); a count
    # that falls from one look to the next is a C extension dropping references (bool_dealloc)
    out["python_refs"] = {name: sys.getrefcount(obj) for name, obj in refwatch_mod.WATCH}
    w = STATE.get("refwatch")
    if w is not None:
        out["python_refs_warnings"] = w.warnings
    return out


def cache_stats() -> dict:
    """What `/v1/cache/stats` answers: what is held, what it costs, and what it bought.

    The byte figures are the real ones -- every snapshot is asked for its own `nbytes` and the
    parts are broken out -- because the interesting question about a state cache on a 121 GiB board
    is not whether it hits, it is what a hit costs to keep. On this model the recurrent state is
    ~150 MB per snapshot whatever the prefix length, and the KV is ~0.8 MB a token; a budget is a
    number of conversations long before it is a number of tokens long.
    """
    eng = STATE.get("engine")
    out = {"model": served_name(), "base_model": STATE.get("model"),
           "session_cache": bool(STATE.get("session_cache")),
           "prefix_cache": bool(STATE.get("prefix_cache")),
           "prefix_chunk": STATE.get("prefix_chunk", 0),
           "last_prefill": STATE.get("last_prefill")}
    store = STATE.get("state_store")
    out["state_store"] = store.report() if store is not None else None
    res = STATE.get("resident")
    out["resident"] = res.report() if res is not None else None
    al = STATE.get("end_align")
    out["end_align"] = al.report() if al is not None else None
    sp = STATE.get("spill")
    out["spill"] = sp.report() if sp is not None else None
    prep = STATE.get("prepared")
    out["compaction"] = (dict(STATE.get("compaction") or {}, **prep.report())
                         if prep is not None else None)
    tk = STATE.get("session_tokens")
    out["session_tokens"] = tk.report() if tk is not None else None
    mem = STATE.get("mem")
    out["memory"] = mem.report() if mem is not None else None
    rcache = STATE.get("response_cache")
    out["response_cache"] = rcache.report() if rcache is not None else None
    suffix = STATE.get("suffix_store")
    out["suffix_store"] = suffix.report() if suffix is not None else None
    disk = STATE.get("disk_cache")
    # the counts only: the entries' previews are the admin's (GET /v1/conductor/memory/disk)
    out["disk_cache"] = disk.report(top=10) if disk is not None else None
    g = getattr(disk, "guard", None) if disk is not None else None
    out["disk_writes"] = g.report() if g is not None else None
    seg = STATE.get("segments")
    out["prompt_tokens_cache"] = seg.report() if seg is not None else None
    tower = STATE.get("vision")
    if tower is not None:
        ic = STATE.get("image_cache")
        out["vision"] = {"tower_bytes": tower.nbytes, **tower.stats,
                         "embed_cache": ic.report() if ic is not None else None}
    if eng is not None:
        out["snapshot_cost"] = eng.cost().snapshot_cost()   # the model's adapter (engine/models/adapter.py)
        out["resident_files"] = eng.resident_files()        # page cache the engine keeps (ENG-239)
    return out


def _text_chunk(cid, model, created, piece, finish=None, extra: dict | None = None,
                logprobs: dict | None = None) -> str:
    body = {"id": cid, "object": "text_completion", "created": created, "model": model,
            "choices": [{"index": 0, "text": piece, "finish_reason": finish,
                         "logprobs": logprobs}]}
    if extra:
        body.update(extra)
    return "data: " + json.dumps(body, ensure_ascii=False) + "\n\n"


def _memory_filter(plan, chat: bool, n_choices: int, ui):
    """The stream's memory filter (server/memblock.py) for a plan whose memory is not off: one
    choice of a chat request; None otherwise (the stream is the release's)."""
    cond = STATE.get("conductor")
    if (cond is None or plan is None or not plan.active or not chat or n_choices != 1
            or getattr(plan, "memory", None) is None):
        return None
    return cond.memory.filter(plan.memory, ui)


def _answer_mask(plan, chat: bool, body: dict):
    """SRV-625: the secret echo mask (server/conductor/redact.py AnswerMask) for a chat request
    whose plan's profile has `secret_mask` on or shadow and whose conversation's tool results hold
    a secret span; None otherwise (the raw path, and every answer with nothing to mask, stays the
    release's byte for byte)."""
    if not chat or plan is None or not plan.active:
        return None
    mode = conductor_schema.profile_secret_mask(getattr(plan.res, "doc", None) or {})
    if mode == "off":
        return None
    spans = redact_mod.tool_spans(body.get("messages"))
    if not spans:
        return None
    return redact_mod.AnswerMask(spans, replace=mode == "on")


def _answer_mask_done(plan, cid: str, sm) -> None:
    """The mask's counts per kind on the request's journal row and in one log line; never a
    value. `extra.secret_mask` when it masked, `extra.secret_mask_shadow` when it only counted."""
    if sm is None or not sm.counts:
        return
    key = "secret_mask" if sm.replace else "secret_mask_shadow"
    if plan.row is not None:
        plan.row.setdefault("extra", {})[key] = dict(sm.counts)
    print(f"[secret_mask] {cid} {'masked' if sm.replace else 'shadow'} "
          + " ".join(f"{k}={v}" for k, v in sorted(sm.counts.items())), flush=True)


def _memory_spans(taken: list[str], ids: list[int], n_prompt: int) -> list:
    """The hidden spans of the blocks the client's copy lacks, at their positions in the
    committed context (a hidden close before them shifts them: found by their tokens)."""
    if not taken:
        return []
    tok = STATE["tok"]
    spans, bad = memblock.token_spans(ids, lambda xs: tok.decode(xs, skip_special_tokens=True),
                                      taken)
    ctx = STATE.get("last_ctx") or []
    out, at = [], n_prompt
    for k0, k1 in spans:
        seq = list(ids[k0:k1])
        p = _locate(ctx, seq, max(at, n_prompt + k0))
        if p is None:
            bad += 1
            continue
        out.append((p, seq))
        at = p + len(seq)
    if bad:
        print(f"[memory] {bad} block(s) not aligned to whole tokens: the next turn prefills "
              "from there", flush=True)
    return out


def _locate(ctx: list[int], seq: list[int], start: int, window: int = 8192) -> int | None:
    n = len(seq)
    for p in range(start, min(len(ctx) - n, start + window) + 1):
        if ctx[p] == seq[0] and ctx[p:p + n] == seq:
            return p
    return None


def _memory_close(plan, mfilter, ids: list[int], n_prompt: int, spans_ok: bool) -> tuple:
    """After a streamed answer: (the blocks kept from the client, their hidden spans), journaled
    on the row. SRV-277: a block the store refused is kept from the client as a written one is,
    so both are hidden spans for the next turn's end alignment."""
    if mfilter is None:
        return [], []
    hidden = list(mfilter.hidden)
    _memory_row(plan, mfilter.failed, getattr(mfilter, "shown", ()))
    return hidden, (_memory_spans(hidden, ids, n_prompt) if spans_ok else [])


def _memory_full(plan, chat, n_choices, ui, ids, in_think, n_prompt, spans_ok) -> tuple:
    """The non-streamed answer's memory blocks: each closed block of the answer written, every
    closed one returned with its hidden span (the text drops them later)."""
    f = _memory_filter(plan, chat, n_choices, ui)
    if f is None:
        return [], []
    text = STATE["tok"].decode(ids, skip_special_tokens=True)
    _, answer = _reasoning_head(text, in_think)
    f.feed(answer)
    f.finish()
    return _memory_close(plan, f, ids, n_prompt, spans_ok)


def _memory_row(plan, failed: list[str], shown=()) -> None:
    ctx = getattr(plan, "memory", None)
    if plan is None or plan.row is None or not ctx:
        return
    info = plan.row["extra"].setdefault("memory", {})
    if ctx.get("written"):
        info["written"] = [{k: v for k, v in w.items() if k != "chars"}
                           for w in ctx["written"]]
    if failed:
        info["failed"] = list(failed)
        hid = [w for w in failed if w not in shown]
        # SRV-277: the reason is the journal's and the log's, never the client's
        print(f"[memory] {plan.row.get('request_id') or '-'}: "
              + (f"{len(hid)} block(s) not written ({', '.join(hid)}), hidden from the answer"
                 if hid else "")
              + ("; " if hid and shown else "")
              + (f"shown as text ({', '.join(shown)}: not a closed block)" if shown else ""),
              flush=True)
    if shown:
        info["shown"] = list(shown)


def _hidden_positions(hid_at, tail: int, s_len: int, n_prompt: int, hidden: int, plan,
                      strict: bool = True):
    """SRV-233: the served prompt's hidden tokens as merged `[(start, end)]`, the closed think
    block at the tail included; None when they are not known (an insertion that was not one run,
    a prompt changed after, or a count that does not add up to the hidden total within the
    whitespace the end alignment moves), and `usage` falls back to the clamp. `strict=False`
    (FX16): the ranges as known, without the count check."""
    if hid_at is None or (strict and s_len != n_prompt):
        return None
    spans = sorted((max(0, a), min(n_prompt, b)) for a, b in hid_at if b > a)
    if tail:
        spans.append((n_prompt - tail, n_prompt))
    out: list[list[int]] = []
    for a, b in sorted(spans):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    total = sum(b - a for a, b in out)
    if strict and abs(total - int(hidden)) > max(64, int(hidden) // 50):
        if plan is not None and plan.row is not None:
            plan.row["extra"]["hidden_at_off"] = [total, int(hidden)]
        return None
    return [(a, b) for a, b in out]


def _refused(plan, cid: str, calls: list) -> None:
    """SRV-247: calls read out of an answer to a request that offered no tools, dropped; the
    journal row names them."""
    names = [str(((c or {}).get("function") or {}).get("name") or "?") for c in calls or ()]
    if not names:
        return
    if plan is not None and plan.row is not None:
        plan.row["extra"].setdefault("calls_refused", []).extend(names[:8])
    print(f"[toolcall] {cid} the request offered no tools: {len(names)} call(s) dropped "
          f"({', '.join(names[:4])})", flush=True)


def _reasoning_head(text: str, in_think: bool) -> tuple[str, str]:
    """`(reasoning block, answer)`: the text through the first `</think>` when the prompt opened
    the block, the rest after it. A block that never closed has no answer yet."""
    if not in_think:
        return "", text
    i = text.find("</think>")
    return (text, "") if i < 0 else (text[:i + len("</think>")], text[i + len("</think>"):])


def _stop_index(text: str, stops: list[str]) -> int | None:
    hits = [text.find(s) for s in stops if s and text.find(s) >= 0]
    return min(hits) if hits else None


def _apply_stops(text: str, stops: list[str]) -> tuple[str, bool]:
    i = _stop_index(text, stops)
    return (text[:i], True) if i is not None else (text, False)


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=None)
    ap.add_argument("--served-model", default="qwen3.8-27b-spark-engine")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--max-len", type=int, default=8192,
                    help="context length. The KV buffer is allocated for all of it up front, so "
                         "this is a memory decision as much as a capability one: 64 KiB a token "
                         "of engine KV plus 20 KiB a token per drafter arm — 27.9 GB of KV and "
                         "53.4 GB of whole stack at 262,144 (measured 2026-09-18, "
                         "tools/mem_audit.py)")
    ap.add_argument("--rep-penalty", type=float, default=1.0,
                    help="deterministic repetition penalty on the target's logits, before the "
                         "argmax (1.0 = off, today's output). Exact under speculation because "
                         "acceptance is argmax equality; see engine/penalty.py. Per request: "
                         "send `repetition_penalty` in the body")
    ap.add_argument("--presence-penalty", type=float, default=0.0,
                    help="flat subtraction from tokens already in the history (0 = off). "
                         "Per request: `presence_penalty` in the body — OpenAI clients and "
                         "Open WebUI already send it, so the day this ships those requests "
                         "change from ignored to honoured")
    ap.add_argument("--frequency-penalty", type=float, default=0.0,
                    help="subtraction per occurrence, on tokens already in the history (0 = off). "
                         "Per request: `frequency_penalty` in the body")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="server default sampling temperature (0 = greedy, the exact path). "
                         "A request that sends temperature > 0 samples -- via engine/sample.py -- "
                         "and decodes without a drafter until rejection sampling exists")
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--no-repeat-ngram", type=int, default=0,
                    help="forbid the token that would complete an n-gram already seen "
                         "(HF's no_repeat_ngram_size; 0 = off, must be >= 2). Deterministic, so "
                         "speculation stays exact; per request: `no_repeat_ngram_size`")
    ap.add_argument("--pattern-stop", default="",
                    help="end a generation that repeats one short pattern forever, as MAX:MIN:COUNT "
                         "(e.g. 64:1:8 -- a block of 1..64 tokens repeated 8 times ends the "
                         "generation with finish_reason stop). The backstop for a greedy loop the "
                         "penalties cannot break, vLLM's RepetitionDetectionParams; empty = off")
    ap.add_argument("--argloop", default="on", choices=("on", "log", "off"),
                    help="the tool-argument loop guard (server/argloop.py): `on` stops a held "
                         "tool call whose arguments loop and tells the model on its next turn; "
                         "`log` only logs where it would have; `off`")
    ap.add_argument("--default-max-tokens", type=int, default=8192,
                    help="what a request that does not send max_tokens gets. It used to be 256 "
                         "and a long answer stopped in the middle of a line")
    ap.add_argument("--reasoning-format", default="tags",
                    choices=("tags", "reasoning_content", "both"),
                    help="how the reasoning block reaches the client: inside <think></think> in "
                         "content (default, and what Open WebUI folds on), as OpenAI-style "
                         "reasoning_content deltas, or both. Overridable per request")
    ap.add_argument("--drafter", default="mtp",
                    choices=("mtp", "router", "merged", "dflash2", "lenrouter", "none"))
    ap.add_argument("--tree", action="store_true",
                    help="verify a draft TREE per step instead of a chain, where the drafter "
                         "builds one; see docs/speculative-decoding.md")
    ap.add_argument("--regenerate", choices=("sample", "off"),
                    default=_S.get("REGENERATE"),
                    help="SRV-241: a regenerate (the same conversation again after it was "
                         "answered, under a TandemLLM plan; or X-Tandem-Regenerate: 1) samples "
                         "with a fresh seed; every other request keeps its own sampling (greedy "
                         "by default). `off`: no detection (server/regenerate.py)")
    ap.add_argument("--regenerate-temperature", type=float,
                    default=float(_S.get("REGEN_TEMPERATURE")),
                    help="a regenerate's temperature (QWEN38_REGEN_TEMPERATURE)")
    ap.add_argument("--regenerate-top-p", type=float,
                    default=float(_S.get("REGEN_TOP_P")),
                    help="a regenerate's top-p (QWEN38_REGEN_TOP_P)")
    ap.add_argument("--regenerate-top-k", type=int,
                    default=int(_S.get("REGEN_TOP_K")),
                    help="a regenerate's top-k (QWEN38_REGEN_TOP_K)")
    ap.add_argument("--sampled-tree", nargs="?", const="det", default="",
                    choices=("", "det", "mixed"),
                    help="let a SAMPLED request keep the tree verify. `det` (the bare "
                         "flag): the greedy request's tree, walked by drawing the target's token "
                         "at each node; `mixed`: the drafter's sampled chain as the spine, accepted "
                         "against its q by rejection sampling, with the lattice's siblings. Off by "
                         "default: sampled requests take the q-aware chain")
    ap.add_argument("--budget", type=int, default=16, help="nodes per tree, anchor included")
    ap.add_argument("--df2-temp", type=float, default=1.0)
    ap.add_argument("--corpus", default=_S.get("CORPUS"),
                    help="suffix store for the lookup drafter, used by --drafter merged")
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--k", type=int, default=0, help="verify block size; 0 = the drafter's depth")
    ap.add_argument("--draft-head", default=None)
    ap.add_argument("--nvfp4", default=None)
    ap.add_argument("--price-table", default=os.environ.get("QSE_PRICE_TABLE", ""),
                    help="FILE[:NAME], the routers' verify/draft/rollback prices for this "
                         "weight set (engine/prices.py); empty = the NVFP4 constants in the code. "
                         "QWEN38_TREE_MS still overrides the tree curve")
    ap.add_argument("--fp8-head", default=None,
                    help="e4m3 lm_head from tools/quant_head.py; halves the head's 2.54 GB in "
                         "both the verify pass and the block drafter's top-k read. `build` (or "
                         "`build:r1,r2,...`) quantises it at load from the checkpoint's bf16 head "
                         "with the served file's ratios; empty = the bf16 head")
    ap.add_argument("--dflash2-blocks", type=int, default=1,
                    help="chained 8-wide draft blocks; 1 proposes 7 tokens, 2 proposes 14")
    ap.add_argument("--dflash2-path", default="greedy", choices=("greedy", "viterbi"))
    ap.add_argument("--dflash2-ckpt", default=None)
    ap.add_argument("--dflash2-ckpt16", default=None,
                    help="the sixteen-wide drafter, for --drafter lenrouter. The router holds both "
                         "checkpoints and picks the block length per step; see engine/lenrouter.py")
    ap.add_argument("--len-fixed", type=int, default=0,
                    help="pin --drafter lenrouter to one block width (8 or 16). 0 lets the "
                         "policy choose, which since phase 9 means --len-latch: one decision a "
                         "request. 8 and 16 measure the fixed baselines "
                         "through the same code")
    ap.add_argument("--len-latch", action=argparse.BooleanOptionalAction, default=True,
                    help="decide the block width ONCE a request -- four wide blocks, up to four "
                         "narrow probes, then no more switching. ON by default since phase 9, "
                         "because switching arms costs 3-18 %% of acceptance depending on the "
                         "workload whether or not the switch was a good idea. "
                         "--no-len-latch restores the per-block router; needs --len-fixed 0 to do "
                         "anything either way")
    ap.add_argument("--learn-cost", action=argparse.BooleanOptionalAction, default=True,
                    help="the length router learns its verify and draft costs from the blocks it "
                         "times, across requests (default). --no-learn-cost keeps the served "
                         "price table: earlier requests no longer move a block's choice, but "
                         "timings inside the request still do, so two runs of the same request "
                         "can part at a near tie (docs/exactness.md, \"Two runs of the same "
                         "request\"). With learning on, the narrator A/B's off runs spread "
                         "37 %% (0.3.24-test: 366 to 504 blocks for the same 2,000 tokens)")
    ap.add_argument("--len-explore", type=int, default=32,
                    help="blocks between forced wide probes when nothing suggests one")
    ap.add_argument("--deep", type=int, default=int(_S.get("DEEP")),
                    help="with --drafter lenrouter --tree: after --deep-after wide blocks in a row "
                         "commit their whole width, propose the lookup drafter's long exact "
                         "continuation of this request's text as one chain of up to this many rows "
                         ". 0 = off. Default from QWEN38_DEEP")
    ap.add_argument("--deep-after", type=int,
                    default=int(_S.get("DEEP_AFTER")),
                    help="full wide blocks in a row before a deep chain (2 keeps it off new "
                         "text). Default from QWEN38_DEEP_AFTER")
    ap.add_argument("--narrow-arm", choices=("on", "off"), default="on",
                    help="with --drafter lenrouter: build the block-8 drafter arm (on, the served "
                         "default) or run the block-16 arm alone (off, SPD-75). Under "
                         "QWEN38_LEN_MODE=wide the narrow arm is released at every request's first "
                         "block and drafts nothing after it, yet holds about 8 GiB at a 262,144 "
                         "window and is synced through every prefill; off refuses any other mode")
    ap.add_argument("--drop-idle", action=argparse.BooleanOptionalAction, default=False,
                    help="release the arm that LOSES the latch for the rest of the request: no "
                         "tap, no sync, and nothing of it in the state snapshot. It stops about "
                         "1.5 ms a block of keeping a cache current for a drafter that will not "
                         "draft again, and takes a third off every snapshot, which is a third "
                         "more entries inside the same cache budget. REFUSES to start with "
                         "--no-len-latch, since the release only happens when the latch closes. "
                         "Off by default until the phase-10 measurement says otherwise")
    ap.add_argument("--relax-tau", type=float, default=1.0,
                    help="LOSSY. Accept a drafted token whose probability is at least this fraction "
                         "of the argmax's. 1.0 (the default) accepts only the target's own "
                         "token: greedy output agrees with plain greedy decoding up to a near tie "
                         "(docs/exactness.md)")
    ap.add_argument("--relax-rank", type=int, default=1,
                    help="LOSSY. Accept a drafted token among the target's top-r. 1 (the "
                         "default) accepts only the target's own token: greedy output agrees with "
                         "plain greedy decoding up to a near tie (docs/exactness.md)")
    ap.add_argument("--think-stall", dest="think_stall", action="store_true", default=True,
                    help="with --conductor shadow or on, run the reasoning-loop arms of the "
                         "policy's reasoning_loop task (a pattern repeated over 128+ tokens, the "
                         "history-wide 16-gram repeat, novelty collapse); the monitor decides "
                         "whether a hit closes the block. --no-think-stall keeps them off. With "
                         "--conductor raw nothing closes a block but a budget")
    ap.add_argument("--no-think-stall", dest="think_stall", action="store_false",
                    help="never run the reasoning-loop arms (the budget still applies)")
    ap.add_argument("--conductor", default="raw", choices=("raw", "off", "shadow", "on"),
                    help="the second-level harness (server/conductor/monitor/). raw, the "
                         "default (off is the same): nothing of it runs and the output is the "
                         "raw engine's. shadow: every rule runs and journals, nothing is done. "
                         "on: each task acts in its policy mode. A request's X-Tandem header "
                         "may lower it, never raise it")
    ap.add_argument("--conductor-policy", default="",
                    help="the monitor's policy file (default ops/supervisor.yaml), validated at "
                         "start against ops/supervisor.schema.json")
    ap.add_argument("--conductor-watch-s", type=float, default=30.0,
                    help="with --conductor shadow or on, how often the stall watchdog's watch "
                         "looks for sessions whose client has gone (S4: client_gone, and NOTIFY "
                         "when its policy enables it)")
    ap.add_argument("--think-budget", type=int, default=0,
                    help="default cap on reasoning tokens per request, 0 = uncapped. A request may "
                         "override it with `max_reasoning_tokens`. When the cap is reached the "
                         "engine closes the reasoning block itself -- see engine/spec.py's "
                         "ThinkBudget -- which CHANGES THE ANSWER and is not a speed trick")
    ap.add_argument("--answer-reserve", default=os.environ.get("QSE_ANSWER_RESERVE", "auto"),
                    help="tokens of the output limit kept for the answer when a client asks for "
                         "a thinking budget in any shape (server/thinking.py): its reasoning "
                         "closes early enough to leave them. auto = min(2048, a quarter of the "
                         "limit), 0 = none (QSE_ANSWER_RESERVE). A request that asks for no "
                         "budget is never touched")
    ap.add_argument("--reasoning-effort", default=None,
                    choices=("none", "off", "low", "medium", "xhigh"),
                    help="default `reasoning_effort` for the chat template. The template's own "
                         "default is xhigh, which is a paragraph asking the model to validate "
                         "assumptions and weigh alternatives before answering. `none` or `off`: "
                         "thinking off for a request that says nothing about thinking (no switch, "
                         "effort word or budget in any shape), as `reasoning_effort: none` in the "
                         "body does")
    ap.add_argument("--cache-budget-gb", type=float, default=2.0,
                    help="RAM budget for the state cache, in GiB (CACHE_GB). One snapshot is the "
                         "48 recurrent states (~150 MB on this model, whatever the prefix length) "
                         "plus the KV of the prefix, so this is a number of CONVERSATIONS, not of "
                         "tokens: 2 GiB holds about three chats under 7k tokens. 0 turns the state "
                         "cache off entirely")
    ap.add_argument("--store-entry-gb", type=float, default=1.0,
                    help="GiB one state-store snapshot may take (STORE_ENTRY_GB), whatever budget "
                         "is free: 1 GiB is about 7k tokens. Room is made before the clone")
    ap.add_argument("--no-session-cache", action="store_true",
                    help="do not keep a conversation's state after its turn. On by default: the "
                         "next turn then re-reads the whole conversation through 64 layers")
    ap.add_argument("--no-prefix-cache", action="store_true",
                    help="do not checkpoint a prefill at chunk boundaries. On by default, which "
                         "is what makes a shared system prompt free from the second request on")
    ap.add_argument("--resident-gb", type=float, default=2.5,
                    help="GiB for the resident prefix's anchors (RESIDENT_GB) -- the recurrent "
                         "state (146.8 MiB) at chunk boundaries of the prompt the KV buffer holds, "
                         "so the next turn of a growing conversation resumes in place, KV not "
                         "copied, at any length (2.5 GiB = 17 anchors). 0 turns it off")
    ap.add_argument("--resident-stash-gb", type=float, default=1.0,
                    help="GiB for the rows a short unrelated request between two turns may "
                         "overwrite, copied aside and back (RESIDENT_STASH_GB; ~104 kB a row: "
                         "1 GiB holds a --resident-guest-max guest and its 512-row slack)")
    ap.add_argument("--resident-guest-max", type=int, default=8192,
                    help="the longest request that is a guest (RESIDENT_GUEST_MAX): opencode's "
                         "title call is 1-3k tokens; a longer one replaces the resident prompt "
                         "with its head pinned")
    ap.add_argument("--resident-tail", type=int, default=4,
                    help="the newest anchors never evicted (the next turn resumes near the end)")
    ap.add_argument("--resident-pin-gb", type=float, default=4.0,
                    help="GiB for the head of the resident prompt a long unrelated request would "
                         "otherwise cut (RESIDENT_PIN_GB): its KV rows and the anchors in it are "
                         "copied aside, and a later request that shares more with that head than "
                         "with the rows in place resumes from it (opencode's compaction request, "
                         "then the next turn with the same tool block). 4 GiB holds about 33k "
                         "tokens. 0 = off")
    ap.add_argument("--resident-park-gb", type=float, default=4.0,
                    help="GiB for one other session's rows (RESIDENT_PARK_GB): when a request of "
                         "another session (x-session-id) replaces the resident prompt, the rows "
                         "past their shared head are copied aside with its newest anchors, and "
                         "that session's next turn swaps them back instead of re-reading them. "
                         "4 GiB holds a session of about 35k tokens past the head. 0 = off")
    ap.add_argument("--end-anchor", default="on", choices=("on", "off"),
                    help="the resident prefix's end anchor (END_ANCHOR, 0.3.14-test): at the end "
                         "of every generation the recurrent state at its last position (147 MiB, "
                         "one per resident prompt), so the next request that starts with the "
                         "prompt and every token the engine wrote resumes there and forwards only "
                         "its new messages, the tokens the monitor hid put back where they sit; "
                         "held to the session cache's greedy gate. off = as 0.3.13-test")
    ap.add_argument("--compaction", default=os.environ.get("QSE_COMPACTION", "continue"),
                    choices=("continue", "client"),
                    help="a client's compaction request (QSE_COMPACTION, 0.3.23-test, "
                         "server/compaction.py): continue = served as the continuation of the "
                         "conversation the engine holds for its session (the prefill reads only "
                         "the instructions); client = as the client sent it (its text history "
                         "read cold)")
    ap.add_argument("--compaction-prepare", default=os.environ.get("QSE_COMPACTION_PREPARE", "on"),
                    choices=("on", "off"),
                    help="write an idle opencode session's summary ahead, so its /compact is "
                         "answered without the engine (QSE_COMPACTION_PREPARE); a request that "
                         "waits stops it at the next chunk or block")
    ap.add_argument("--compaction-idle-s", type=float,
                    default=float(os.environ.get("QSE_COMPACTION_IDLE_S", "30")),
                    help="seconds without a request before a summary is prepared")
    ap.add_argument("--compaction-min-tokens", type=int,
                    default=int(os.environ.get("QSE_COMPACTION_MIN_TOKENS", "32768")),
                    help="the shortest conversation a summary is prepared for")
    ap.add_argument("--end-align", default="on", choices=("on", "off"),
                    help="the end anchor for a copy whose whitespace is not the rows' (END_ALIGN, "
                         "0.3.21-test, server/endalign.py): a follow-up whose copy of an earlier "
                         "generation differs from what the engine wrote in whitespace alone (a "
                         "tool call's trimmed parameter value, the template's newline before "
                         "</think>) is served with the rows' tokens up to the end anchor and "
                         "resumes there; off = the hidden spans' replay alone, as 0.3.20-test")
    ap.add_argument("--mem-ceiling-gib", type=float, default=70.0,
                    help="GiB the engine may hold between requests, allocated plus the "
                         "allocator's reserve (MEM_CEILING_GIB): a cache grows only while the "
                         "engine stays under it less --reserve-slack-gib, and over it the caches "
                         "are given back, cheapest first, before the next prefill. 0 = no limits")
    ap.add_argument("--mem-peak-gib", type=float, default=77.0,
                    help="GiB the engine may hold during a prefill (MEM_PEAK_GIB): the ceiling plus "
                         "the workspace of a 262,144-token prompt. A prefill that would go over it "
                         "gives the caches back first, and gets a 503 with Retry-After when that "
                         "is not enough. The residency planner counts this, not today's free memory")
    ap.add_argument("--reserve-slack-gib", type=float, default=2.0,
                    help="GiB of free blocks the allocator may keep between requests "
                         "(RESERVE_SLACK_GIB); over it they go back to the driver once a request "
                         "has finished, never inside one")
    ap.add_argument("--prompt-token-cache", type=float, default=4.0,
                    help="millions of tokens of tokenised chat messages kept, so a growing "
                         "conversation's prompt is tokenised only where it is new "
                         "(server/tokcache.py). 0 = tokenise every prompt whole")
    ap.add_argument("--session-trim", choices=("on", "off"), default="off",
                    help="cut the engine back to the committed tokens before the session "
                         "snapshot when the last verified block ran past them (end token, stop, "
                         "budget) and its commit is still pending, so the put is not declined. "
                         "A session resume is held to the argmax gate, not to bit equality with "
                         "a cold prefill; off keeps every served text as it was")
    ap.add_argument("--store-boundaries", type=int, default=1,
                    help="chunk boundaries of one prefill the state store keeps: its longest ones "
                         "under the per-entry cap. 0 = every boundary, where one cold prefill's "
                         "nested snapshots fill the budget and evict everything else")
    ap.add_argument("--prefill-heartbeat-s", type=float, default=5.0,
                    help="a streamed request whose prefill has run this long gets an SSE "
                         "comment (': prefill done/total') after every chunk, at most once a "
                         "second, so a client and every proxy between see the stream is alive. "
                         "Clients ignore comments; 0 = never. The engine checks for a client "
                         "that left after every chunk either way")
    ap.add_argument("--live-activity", choices=("on", "off"), default="on",
                    help="/the live activity on /v1/dashboard/live (states, "
                         "prefill progress, timeline, the last 20 stops) at QSE_LIVE_HZ events a "
                         "second while busy. off = contract 1.1 with those fields null at one "
                         "event a second: the kill switch, and the A/B that isolates its cost")
    ap.add_argument("--max-prefill-rows", type=int, default=8192,
                    help="with the prefix cache off, forward a long prompt in chunks of this many "
                         "rows instead of one call; a 131k-token single forward exhausted the "
                         "board. 0 = one call. With the prefix cache on its chunk applies")
    ap.add_argument("--prefix-chunk", type=int, default=1024,
                    help="tokens between prefill checkpoints, and the forward size of EVERY "
                         "prefill while the prefix cache is on -- the two have to agree or a warm "
                         "prefill is not the same arithmetic as a cold one. A CHUNK COSTS A WHOLE "
                         "16.35 GB WEIGHT READ: a 1,724-token prompt is 1.14x at 1024 and 1.57x "
                         "at 256. 1024 is the default because a prompt "
                         "nobody shares pays that and gets nothing. Drop it to 256 if you serve "
                         "one system prompt to many different tails, where the finer grid wins "
                         "back far more than it costs")
    ap.add_argument("--response-cache", action="store_true",
                    help="OPT-IN. Answer an identical (prompt, params) request from memory. "
                         "Greedy decoding is a function so this is exact, but a server that "
                         "answers from a dictionary must never be what a benchmark measures")
    ap.add_argument("--response-cache-mb", type=float, default=256.0)
    ap.add_argument("--response-cache-ttl", type=float, default=3600.0)
    ap.add_argument("--suffix-store", default=_S.get("SUFFIX_STORE"),
        help="directory for the persistent suffix store of what this engine has read and written, "
             "which the lookup drafter reads as a second corpus. Token ids only, never text, "
             "outside this repository, mode 0700. Empty string turns it off")
    ap.add_argument("--suffix-store-readonly", action="store_true",
                    help="read the suffix store and never append to it: a fixed benchmark measured "
                         "against a store of real traffic must not write itself into it")
    ap.add_argument("--suffix-store-mb", type=float, default=192.0,
                    help="cap on the store, in MiB of int32 token ids (192 MiB = 48 M tokens). "
                         "Over the cap the oldest half is forgotten at the next document boundary")
    ap.add_argument("--disk-cache", default="off",
                    help="the disk tier (C1, engine/disk_cache.py): a directory on the NVMe for "
                         "prompt-prefix state -- prefixes prefilled more than once, a session's "
                         "first long request -- under the caches in RAM, or off (the default). "
                         "QSE_DISK_CACHE in serve.env; ops/start.sh passes it to the :8000 service "
                         "only. Its budget, TTL, admission and on/off are the conductor's settings "
                         "document (memory.disk_cache), edited on the Memory screen")
    ap.add_argument("--disk-cache-min-tokens", type=int, default=4096,
                    help="the shortest prefix the disk tier writes an anchor for")
    ap.add_argument("--conductor-defaults", default="builtin", choices=("builtin", "shipped"),
                    help="0.3.22-test (server/conductor/shipped.py): shipped, the live documents "
                         "of 2026-10-03 are stored as version 1 for every document the "
                         "conductor's directory has no version of (ops/start.sh passes it to "
                         "the :8000 service); builtin, the minimal built-ins stay version 0")
    ap.add_argument("--session-spill", default="on", choices=("on", "off"),
                    help="0.3.22-test (engine/spill.py): a live conversation on the NVMe, under the "
                         "disk tier's directory: written whole at a planned stop and read back at "
                         "the next start, written before another client's request evicts it, read "
                         "back at its next turn. Needs --disk-cache; off = 0.3.21-test")
    ap.add_argument("--session-spill-gb", type=float, default=64.0,
                    help="the session spill's own budget on the NVMe (a 256k conversation is "
                         "about 27 GB as it is in the engine, about 16 GB compressed); the disk "
                         "tier's free-space floor holds for both; settings.memory.session_spill "
                         "overrides it live")
    ap.add_argument("--session-spill-codec", default="zstd", choices=("zstd", "off"),
                    help="0.4.3-test: new spill blocks as byte planes compressed with zstd "
                         "(lossless, 1.71x on a real entry; blocks on the disk are read either way)")
    ap.add_argument("--disk-write-budget-gb", type=float, default=200.0,
                    help="0.4.3-test: the engine's NVMe writes a local day past which optional "
                         "writes stop (the tier's, and the spill's but a drain and evictions of "
                         "32k+ tokens; past 1.5x only the drain); 0 = no budget; "
                         "settings.memory.disk_writes.budget_gb overrides it live")
    ap.add_argument("--session-spill-min-tokens", type=int, default=16384,
                    help="the shortest resident conversation an eviction writes to the NVMe "
                         "(shorter ones are the RAM park's)")
    ap.add_argument("--disk-cache-head-anchors", type=int, default=4,
                    help="anchors a session's first long request gets, at the grid lines below "
                         "where its last user message starts (where the next session parts)")
    ap.add_argument("--suffix-store-scope", default="all", choices=("all", "outputs"),
                    help="`all` remembers prompts and answers, `outputs` only what the engine "
                         "wrote. Both stay on the box")
    ap.add_argument("--request-timeout", type=float, default=900.0,
                    help="wall-clock cap on one generation, checked between blocks; 0 disables. "
                         "A request that hits it finishes with reason 'timeout', which is a "
                         "different thing from 'length' and the caller should be told which")
    ap.add_argument("--max-queue", type=int, default=8,
                    help="requests allowed to wait for the engine before new ones get a 503 with "
                         "Retry-After. This engine serves one sequence at a time")
    ap.add_argument("--queue-timeout", type=float, default=120.0,
                    help="how long a request waits for the engine before a 429")
    ap.add_argument("--side-lane", default="on",
                    choices=("on", "decode", "off"),
                    help="0.3.27-test (server/preempt.py): a short request -- a client's side "
                         "request or a raw one, thinking off, within the two caps below -- is "
                         "served at the next safe point of the generation that holds the engine "
                         "(on: between decode blocks and between prefill chunks; decode: between "
                         "decode blocks only), which gets the engine back byte for byte; off: it "
                         "waits in the queue")
    ap.add_argument("--side-lane-prompt-max", type=int, default=preempt.PROMPT_MAX,
                    help="the longest prompt the side lane serves, in tokens")
    ap.add_argument("--side-lane-new-max", type=int, default=preempt.NEW_MAX,
                    help="the most output tokens the side lane serves")
    # parallel requests, phase A (server/slots.py); off (1) builds today's server
    ap.add_argument("--parallel", type=int, default=int(os.environ.get("QSE_PARALLEL", "1") or 1),
                    help="sequence slots: 1 (the default) is one sequence, as always; N builds N - 1 "
                         "side slots beside the engine (each a full KV, recurrent state and "
                         "drafter stack over the one copy of the weights, `--side-max-len` long), "
                         "and two or more decoding requests verify their trees in one pass")
    ap.add_argument("--side-max-len", type=int,
                    default=int(os.environ.get("QSE_SIDE_MAX_LEN", "65536") or 65536),
                    help="a side slot's window in tokens; slot 0 keeps --max-len (262,144). A "
                         "request whose prompt + max_tokens does not fit waits for slot 0")
    ap.add_argument("--parallel-floor-gb", type=float,
                    default=float(os.environ.get("QSE_PARALLEL_FLOOR_GB", "15") or 15),
                    help="admission: MemAvailable less the caches' growth, the requests in flight "
                         "and this request's worst case must stay at or above this; a side slot "
                         "is built only if it does with the slot's bytes added")
    ap.add_argument("--parallel-batch", default=os.environ.get("QSE_PARALLEL_BATCH", "on"),
                    choices=("on", "off"),
                    help="on: decoding slots verify their trees in one joint pass (lockstep "
                         "rounds); off: they take turns, one verify each")
    ap.add_argument("--parallel-draft", default=os.environ.get("QSE_PARALLEL_DRAFT", "on"),
                    choices=("on", "off"),
                    help="with the joint verify on: the decoding slots' draft calls in one packed "
                         "pass over the drafter (on), or one after the other (off)")
    ap.add_argument("--parallel-switch-ms", type=float,
                    default=float(os.environ.get("QSE_PARALLEL_SWITCH_MS", "0") or 0),
                    help="with side slots: the interpreter's thread switch interval in ms (0 keeps "
                         "Python's 5 ms); a woken slot waits up to this long for the GIL")
    ap.add_argument("--parallel-cut", default=os.environ.get("QSE_PARALLEL_CUT", "share"),
                    choices=("share", "joint"),
                    help="how a joint pass's rows are shared: share (each slot cuts its tree to a "
                         "fixed share before it submits: 16 at two slots, 10 at three, 8 at four) or "
                         "joint (the round cuts every slot's whole tree together, best nodes first)")
    ap.add_argument("--parallel-max-rows", type=int, default=32,
                    help="rows of a joint pass at most (32: the row-invariant kernels' range)")
    ap.add_argument("--batch-timeout-ms", type=float, default=2.0,
                    help="a joint round runs without a missing slot once the GPU turn has been "
                         "idle this long")
    ap.add_argument("--turn-quantum-ms", type=float, default=50.0,
                    help="the GPU turn goes to a waiting slot this far behind in GPU time")
    ap.add_argument("--parallel-settle", default="on", choices=("on", "off"),
                    help="synchronize the device before the turn changes hands (the routers' "
                         "timings then hold only their own slot's work)")
    ap.add_argument("--side-precapture", default="off", choices=("on", "off"),
                    help="capture a side slot's verify graphs at start (on, about a minute a slot) or "
                         "at their first use (off: the build takes seconds; each width's first "
                         "lone round on a side slot pays its capture, a fraction of a second)")
    ap.add_argument("--side-graphs", default=os.environ.get("QSE_SIDE_GRAPHS", "on"),
                    choices=("on", "off"),
                    help="a side slot's own verify graphs: on (captured as --side-precapture says; "
                         "most of a slot's constant bytes) or off (the slot verifies eager when it "
                         "decodes alone and holds no graph pool; a joint pass is eager either way)")
    ap.add_argument("--side-slot-const-gb", type=float,
                    default=float(os.environ.get("QSE_SIDE_SLOT_CONST_GB", "0") or 0),
                    help="a side slot's bytes beyond its window's, in the build's estimate; 0 (the "
                         "default) follows --side-graphs: 7.5 GiB with it on (6.9 to 7.5 GiB allocated "
                         "measured for 8,192-token slots, most of it the verify graphs' pools: 121 "
                         "graphs, as slot 0's 5.5 GiB), 1.5 with it off")
    # SRV-307: the shared KV pool (engine/kvpool.py)
    ap.add_argument("--kv-pool", default=os.environ.get("QSE_KV_POOL", "off") or "off",
                    choices=("on", "off"),
                    help="on: every slot's KV window (engine attention KV and the drafters' "
                         "context KV) is reserved as virtual addresses and its pages are committed "
                         "as a conversation's rows are written and given back when it ends, from "
                         "one pool for every slot (CUDA virtual memory management; bit-identical "
                         "to the static buffers). off: each slot allocates its window at load")
    ap.add_argument("--kv-hard-floor-gb", type=float,
                    default=float(os.environ.get("QSE_KV_HARD_FLOOR_GB", "10") or 10),
                    help="with --kv-pool on: a commit never takes MemAvailable under this (GiB); "
                         "the pool gives back caches and idle conversations first, else the "
                         "request is not run (admission keeps requests out at --parallel-floor-gb)")
    ap.add_argument("--kv-margin-rows", type=int,
                    default=int(os.environ.get("QSE_KV_MARGIN_ROWS", "256") or 256),
                    help="with --kv-pool on: rows committed past every write (a masked tile's "
                         "over-read stays on mapped pages)")
    # image input
    ap.add_argument("--vision", default="on", choices=("on", "off"),
                    help="load the checkpoint's vision tower (0.92 GB bf16) and accept image_url "
                         "parts in chat requests; off refuses them with a 400. A checkpoint "
                         "without a tower serves text either way")
    ap.add_argument("--image-max-mb", type=float, default=20.0,
                    help="largest image a request may send or link, after base64 decoding")
    ap.add_argument("--image-max-decode-pixels", type=int, default=64_000_000,
                    help="largest canvas an image may declare, checked before its pixels are "
                         "decoded; the processor then resizes to its own limits")
    ap.add_argument("--image-max-pixels", type=int, default=0,
                    help="the processor's resize target cap in pixels (0: the checkpoint's "
                         "preprocessor_config.json, 16,777,216 for Qwen3.8 = 16,384 rows an image)")
    ap.add_argument("--image-backend", default=os.environ.get("QSE_IMAGE_BACKEND", "torchvision"),
                    choices=("torchvision", "pil"),
                    help="the image processor's backend, pinned so the patches do not depend on "
                         "what is installed (the two differ by up to two 8-bit levels after a "
                         "resize). torchvision (the default, what every served measurement ran) "
                         "needs torchvision 0.28; pil is for an environment without it. A "
                         "backend that is not installed stops the start with the package to "
                         "install")
    ap.add_argument("--max-images", type=int, default=16, help="images one request may send")
    ap.add_argument("--max-image-rows", type=int, default=65536,
                    help="prompt rows all of a request's images may take together (one row per "
                         "2x2 block of 16-pixel patches; a 1024x1024 image is 1024 rows)")
    ap.add_argument("--image-https", default="on", choices=("on", "off"),
                    help="fetch https:// image URLs (off: data: URLs only)")
    ap.add_argument("--image-https-private", default="off", choices=("on", "off"),
                    help="let https image URLs reach loopback, private and link-local "
                         "addresses (off: public hosts only)")
    ap.add_argument("--image-fetch-timeout", type=float, default=15.0,
                    help="seconds for a whole https image download, redirects included")
    ap.add_argument("--image-cache-mb", type=float, default=512.0,
                    help="encoded images kept on the device by content, so a conversation that "
                         "sends an image again does not encode it again")
    ap.add_argument("--image-act-gb", type=float, default=6.0,
                    help="admission: an image whose encode would need more than this (the "
                         "tower's byte math, engine/vision.py) is refused with a 400")
    ap.add_argument("--usage-default", default=os.environ.get("QSE_USAGE_DEFAULT", "on"),
                    choices=("on", "off"),
                    help="a streamed request that sends no stream_options gets usage, "
                         "timings and metrics on its finish chunk (on, the default: Open WebUI asks "
                         "for include_usage only when a model's Usage capability is ticked), or "
                         "nothing, as before (off). include_usage true/false is honoured either "
                         "way. Default from QSE_USAGE_DEFAULT")
    ap.add_argument("--usage-ledger", default="off",
                    help="the SQLite usage ledger's path, or off (the default). One row per "
                         "request, counts and times only, no text. NOT read from the environment: "
                         "ops/start.sh passes serve.env's QSE_USAGE_LEDGER for the :8000 service, "
                         "and every benchmark server stays off")
    ap.add_argument("--conductor-config", default="off",
                    help="the conductor's config directory (versioned profiles, router, roles, "
                         "packages and its journal, server/conductor/), or off (the default). On, "
                         "/v1/models lists `TandemLLM` alone (the raw name is a hidden alias; "
                         "`X-Tandem: raw` serves the raw path) and /v1/conductor/* "
                         "exists; the raw model name is served as before. NOT read from the "
                         "environment: ops/start.sh passes serve.env's QSE_CONDUCTOR_CONFIG for "
                         "the :8000 service, and every benchmark server stays off")
    ap.add_argument("--backends", default="off",
                    help="the proxy's config database (proxy/, docs/proxy.md), or off (the "
                         "default). On, a request whose model routes to a configured backend "
                         "(vLLM, SGLang, any OpenAI-compatible server, OpenAI, DeepSeek) is relayed "
                         "on its handler's thread, never touching the engine's lock or queue; "
                         "/v1/models adds the backends' models and /v1/config/* serves the "
                         "Backends screen. TandemLLM and the raw name stay the engine's. The "
                         "TANDEM_* variables overlay the database (the environment wins). NOT read "
                         "from the environment: ops/start.sh passes serve.env's QSE_BACKENDS")
    ap.add_argument("--host-collector", default="on", choices=("on", "off"),
                    help="the System view's hosts (proxy/hostcollect.py): this machine's memory, "
                         "GPU, CPU, disks and network, and the process behind each backend on it. "
                         "Sampled only when the view asks, at most every 5 s (the process scan "
                         "every 15 s); off leaves system.hosts out")
    ap.add_argument("--usage-retention-days", type=int, default=ledger_mod.MIN_RETENTION_DAYS,
                    help="rows older than this are pruned daily at 04:00; below 400 only with "
                         "QSE_TEST=1")
    ap.add_argument("--trust-loopback", action=argparse.BooleanOptionalAction, default=True,
                    help="a request from loopback with no X-Forwarded-For / X-Real-IP "
                         "header (the box's own watchdog, row3, gate, soak) needs no token for "
                         "/metrics, the full /health and the cache routes once an admin password "
                         "or metrics token is set. The reverse proxy sets both headers, so "
                         "tunnelled traffic never qualifies")
    ap.add_argument("--dashboard-login", default=os.environ.get(auth_mod.LOGIN_ENV,
                                                                 auth_mod.LOGIN_DEFAULT),
                    choices=("on", "off"),
                    help="on, with an admin password (QSE_ADMIN_TOKEN) set: /v1/dashboard/* "
                         "needs the admin token or a session too (the page has no sign-in screen "
                         "yet). off, the default: the dashboard's read API answers without a "
                         "token. With no admin password set every admin route is open and this "
                         "flag changes nothing. Default from QSE_DASHBOARD_LOGIN")
    ap.add_argument("--dashboard-dir", default=None,
                    help="where the dashboard's built files are (default: dashboard/dist in this "
                         "repository); /dashboard/ serves them, or a placeholder when missing")
    ap.add_argument("--setup", action="store_true",
                    help="setup mode (docs/setup.md, `ops/start.sh --setup`): no model of Qwen's "
                         "is loaded; the dashboard, /v1/setup/* and Wren, the setup guide's "
                         "assistant, alone. Every other route answers 503 setup_mode")
    ap.add_argument("--fake-engine", action="store_true",
                    help="the e2e: no weights, no GPU -- a deterministic CPU token source "
                         "(server/fake_engine.py) behind the real handler, auth, ledger, metrics "
                         "and logs. Refuses the production ledger")
    ap.add_argument("--fake-tps", type=float, default=200.0,
                    help="with --fake-engine: tokens a second it writes")
    ap.add_argument("--fake-prefill-tps", type=float, default=0.0,
                    help="with --fake-engine: the prefill rate of a FAKE_SLOW_PREFILL prompt "
                         "(0 = 2,000 tokens a second)")
    ap.add_argument("--fake-load-s", type=float, default=0.0,
                    help="with --fake-engine: a load of this many seconds before the port opens, "
                         "writing the load's phases as the real engine does (the Setup Guide's "
                         "restart screen and its release gate, OPS-175)")
    ap.add_argument("--structured-outputs", action=argparse.BooleanOptionalAction, default=True,
                    help="serve response_format json_object / json_schema and "
                         "structured_outputs (regex, choice, json) as masks over the target's "
                         "rows, exact under speculation. A request without them is untouched; "
                         "--no-structured-outputs refuses them with a 400 as before")
    ap.add_argument("--log-content", action="store_true",
                    help="let exception messages that quote a request into the log. Off by "
                         "default: no line carries prompt, message, tool or answer text")
    ap.add_argument("--log-request-keys", action="store_true",
                    help="/one [body] line per request with the parameter names and "
                         "the scalar values of the non-content ones (model, stream, max_tokens, "
                         "temperature, ...); never messages, prompt, tools or stop strings")
    ap.add_argument("--verbose", action="store_true")
    return ap


def open_ledger(a, *, test: bool | None = None) -> "ledger_mod.Ledger | None":
    """The usage ledger the flags ask for, or None. Refuses a bad configuration."""
    if test is None:
        test = os.environ.get("QSE_TEST") == "1"
    path = None if a.usage_ledger in ("", "off") else a.usage_ledger
    ledger_mod.check_config(path, a.usage_retention_days, test=test)
    if path is None:
        return None
    return ledger_mod.Ledger(path, retention_days=a.usage_retention_days)


def conductor_root(a) -> str | None:
    """The conductor's config directory, or None for off. A fake engine is a test run: it may never
    use the production one."""
    from server.conductor.store import DEFAULT_ROOT
    val = getattr(a, "conductor_config", "off")
    if val in (None, "", "off"):
        return None
    if a.fake_engine and os.path.realpath(os.path.expanduser(val)) == \
            os.path.realpath(os.path.expanduser(DEFAULT_ROOT)):
        raise SystemExit("[conductor] REFUSING: a fake engine may not use the production conductor "
                         "config; give it a directory of its own")
    return val


def _thinking_default(a) -> None:
    """SRV-477: `--reasoning-effort none|off` is a server default of thinking off
    (`a.thinking_default`); the template gets no effort word then."""
    a.thinking_default = "off" if a.reasoning_effort in ("none", "off") else "on"
    if a.thinking_default == "off":
        a.reasoning_effort = None


def main() -> None:
    ap = parser()
    a = ap.parse_args()
    _thinking_default(a)
    # argparse does not check a default against `choices`, and a typo in QSE_DASHBOARD_LOGIN
    # ("true", "ON") must not quietly decide who reads the dashboard
    if a.dashboard_login not in ("on", "off"):
        ap.error(f"--dashboard-login / {auth_mod.LOGIN_ENV} is on or off, not "
                 f"{a.dashboard_login!r}")
    # the tokens come from the environment (ops/start.sh sources secrets.env); a token that
    # is too short stops the start here, before the minutes of loading
    STATE["auth"] = auth_mod.Auth.from_env(trust_loopback=a.trust_loopback,
                                           login=a.dashboard_login == "on")
    # every line from here on also reaches /v1/dashboard/logs; the log file is unchanged
    logbuf.install()
    logbuf.CONFIG["log_content"] = bool(a.log_content)
    if a.setup:
        return _serve_setup(a)
    if a.log_content:
        print("[server] --log-content is ON: exception messages may quote requests in the log",
              flush=True)
    # before the minutes of loading: a ledger configuration that must not start stops here (a
    # fake engine is a test run: it may never write the production ledger)
    led = open_ledger(a, test=True if a.fake_engine else None)
    # and the conductor's policy: a file that does not validate stops the start here too
    STATE["conductor_policy"] = _conductor_policy(a)
    _limits_at_start(a)
    _window_at_start(a)
    if a.fake_engine:
        from server import fake_engine
        fake_engine.load(sys.modules[__name__], a)
    else:
        _load(a)
    _serve(a, led)


def _serve_setup(a) -> None:
    """Setup mode (OPS-93, docs/setup.md): a box with no configured engine yet. Nothing of Qwen's
    loads, nothing of the engine's caches, ledger, conductor or proxy opens; the server answers
    the dashboard, /v1/setup/* and /health (`mode: setup`) and loads Wren alone, which the setup
    guide's apply job unloads before it starts the configured engine in this one's place."""
    from server import setup_api
    ident = identity()
    ident["version"] = ident["version"].split("+")[0]   # no weight profile serves here
    STATE.update(ident)
    STATE.update(setup_mode=True, started=time.time(), args=vars(a), port=a.port,
                 dashboard_dir=a.dashboard_dir, model="setup")
    print(f"[server] {STATE['auth'].describe()}", flush=True)
    print(f"[setup] setup mode {STATE['version']}: the dashboard's #/setup, /v1/setup/* and Wren "
          f"on http://{a.host}:{a.port}; no model of Qwen's loaded", flush=True)
    w = setup_api.wren_of(sys.modules[__name__])
    threading.Thread(target=w.load, name="wren-autoload", daemon=True).start()
    serve_until_drained(EngineServer((a.host, a.port), SetupHandler))


def _limits_at_start(a) -> None:
    """OPS-47: the memory limits the dashboard set (the residency document's `engine` section)
    over the flags, before the load -- the keys that need a restart take effect only here. The
    flags as given stay in STATE["serve_flags"], so the panel can say where each value comes
    from. A document that cannot be read leaves the flags as they are."""
    STATE["serve_flags"] = dict(vars(a))
    try:
        root = conductor_root(a)
    except SystemExit:
        raise
    except Exception:                                              # noqa: BLE001
        root = None
    if root is not None:
        try:
            from server.conductor.store import Store
            doc = Store(root, readonly=True).get("residency")[1]
            for line in memlimits.overrides_at_start(a, doc, len_mode=_S.get("LEN_MODE")):
                print(f"[memory] the dashboard's limit: {line}", flush=True)
        except Exception as exc:                                   # noqa: BLE001
            print(f"[memory] the residency document's limits were not read: {exc}", flush=True)


def _window_at_start(a) -> None:
    """SRV-335: a --max-len above the served profile's largest window (QSE_WEIGHTS, its registry
    entry's `max_window`) stops the start here, before the minutes of loading, naming both numbers.
    The 27B's profiles name no such window; a registry that does not load is said and passed."""
    name = os.environ.get("QSE_WEIGHTS")
    if not name:
        return
    try:
        from server.setup import registry
        why = registry.window_refusal(name, a.max_len)
    except Exception as exc:                                       # noqa: BLE001
        print(f"[server] the model registry was not read ({exc}); --max-len is not checked", flush=True)
        return
    if why:
        raise SystemExit(f"[server] REFUSING to load: {why}")


def _conductor_policy(a) -> "conductor.Policy | None":
    """The monitor's policy for `--conductor shadow|on`, validated; None when raw."""
    mode = conductor.parse_mode(getattr(a, "conductor", "raw"))
    if mode in (None, "raw"):
        return None
    try:
        return conductor.load(getattr(a, "conductor_policy", "") or None)
    except conductor.PolicyError as exc:
        raise SystemExit(f"[conductor] the policy does not load: {exc}")


def _blocking_sync() -> None:
    """Hypothesis 3: the host waits on the GPU by sleeping instead of spinning.

    The loop synchronises twice a round and spends most of a ~95 ms round waiting there; by default
    the CUDA runtime spins a CPU core for it, on a SoC whose CPU and GPU share one power budget. With
    QWEN38_BLOCKING_SYNC=1 the device's primary context -- the one torch uses -- is created with
    CU_CTX_SCHED_BLOCKING_SYNC, set through the driver before torch touches the device. Only how the
    host waits changes, never a value; each wait then costs a wake-up."""
    import ctypes
    cu = ctypes.CDLL("libcuda.so.1")
    dev = ctypes.c_int()
    rc = (cu.cuInit(0), cu.cuDeviceGet(ctypes.byref(dev), 0),
          cu.cuDevicePrimaryCtxSetFlags(dev, 0x04))                  # CU_CTX_SCHED_BLOCKING_SYNC
    flags, active = ctypes.c_uint(), ctypes.c_int()
    cu.cuDevicePrimaryCtxGetState(dev, ctypes.byref(flags), ctypes.byref(active))
    print(f"[server] blocking sync: driver calls {rc}, primary context flags 0x{flags.value:x} "
          f"(active {active.value})", flush=True)
    if any(rc) or not flags.value & 0x04:
        raise SystemExit("[server] QWEN38_BLOCKING_SYNC=1 but the context flag did not take")


def _make_lenrouter(a, eng, max_len: int, corpus_from=None):
    """The served drafter stack (`--drafter lenrouter`) over `eng`, its draft caches `max_len`
    long: slot 0's at the engine's window, a side slot's (`--parallel N`) at its own. Sets
    `a.depth`. `corpus_from`: the lookup drafter whose corpus (and suffix store) a side slot's
    lookup drafter reads too, instead of loading its own copy."""
    from engine.drafters.dflash2 import DFlash2Drafter
    from engine.lenrouter import LengthRouter, NoArm
    if not a.dflash2_ckpt16:
        raise SystemExit("--drafter lenrouter needs --dflash2-ckpt16")
    if a.narrow_arm == "off":
        # SPD-75: the wide arm alone; the router keeps the narrow arm's place released
        small = NoArm(eng, block=8)
    else:
        small = DFlash2Drafter(eng, a.dflash2_ckpt, blocks=1, path=a.dflash2_path,
                               draft_head=a.draft_head, max_len=max_len, block=8)
        small._build()
    large = DFlash2Drafter(eng, a.dflash2_ckpt16, blocks=1, path=a.dflash2_path,
                           draft_head=a.draft_head, max_len=max_len, block=16)
    large._build()
    if a.tree:
        # The combined configuration: each arm is the lookup drafter's tree merged with that
        # arm's own lattice, and the length router chooses the node budget. One NgramDrafter
        # for both arms -- its suffix index is updated in `observe`, and two arms each
        # observing every block would index every token twice.
        from engine.drafters.ngram import NgramDrafter
        from engine.router import MergedRouter, served_tree_table, tree_nodes
        tree_table = served_tree_table(PRICES.get("tree_ms"))
        ng = NgramDrafter(corpus_path=a.corpus if corpus_from is None else "",
                          tokenizer_sha=STATE.get("tokenizer_sha"), min_order=3, max_depth=16,
                          node_budget=large.cfg.block_size - 1, branch_top_k=3,
                          min_expected=0.2, alpha=0.6, corpus_weight=0.5, min_corpus_order=8,
                          verify_base_ms=tree_table[8],
                          verify_per_node_ms=(tree_table[16] - tree_table[8]) / 8)
        if corpus_from is not None:
            ng.corpus = corpus_from.corpus           # the same corpus and suffix store, read-only
        arms = [MergedRouter(ng, head, mtp_depth=head.cfg.block_size - 1,
                             node_budget=tree_nodes(head.cfg.block_size) - 1, mtp_ms_per_token=0.0,
                             head_fixed_ms=PRICES.get("head_fixed_ms", 27.0), adaptive_depth=False,
                             rollback_ms=PRICES.get("rollback_ms", 6.4),
                             verify_ms_table=dict(tree_table), tree_ms_table=dict(tree_table))
                if not isinstance(head, NoArm) else head for head in (small, large)]
        try:
            drafter = LengthRouter(arms[0], arms[1], fixed=a.len_fixed,
                                   explore_period=a.len_explore, tree=True, ngram=ng,
                                   verify_table=PRICES.get("lenrouter_tree_ms"),
                                   draft_table=PRICES.get("lenrouter_draft_ms"),
                                   latch=a.len_latch, drop_idle=a.drop_idle, deep=a.deep,
                                   deep_after=a.deep_after, latch_table=dict(tree_table),
                                   learn_cost=a.learn_cost)
        except ValueError as exc:
            raise SystemExit(f"[server] {exc}")
    else:
        try:
            drafter = LengthRouter(small, large, fixed=a.len_fixed,
                                   explore_period=a.len_explore, latch=a.len_latch,
                                   width_trim=not a.len_latch, drop_idle=a.drop_idle,
                                   learn_cost=a.learn_cost)
        except ValueError as exc:
            raise SystemExit(f"[server] {exc}")
    # The router may propose the wide block on any step, so the loop's cap has to be the wide
    # one; asking it for fewer would silently pin it to the narrow length. With the deep chain
    # it is the deep width, for the same reason.
    a.depth = max(large.cfg.block_size - 1, a.deep - 1 if a.tree else 0)
    return drafter


def _arch_of(model: str | None):
    """The registered architecture of `--model` (engine/models `REGISTRY`); the 27B's when `--model`
    names no checkpoint directory (the 27B's snapshot is resolved later)."""
    from engine.models import REGISTRY, architecture
    if not model or not os.path.isfile(os.path.join(model, "config.json")):
        return REGISTRY["qwen3_5"]
    with open(os.path.join(model, "config.json")) as f:
        return architecture(json.load(f))


def _serving_state(a, eng, tok, drafter, cfg_eos, rcache, **model) -> None:
    """The STATE the request path reads, for every model's loader (SRV-341): the server's own knobs
    from `a`, and in `model` what the loader built -- the tree and sampled-tree mode, the accept rule,
    the device, the caches (or None/False where the model has none yet) and any model-specific record."""
    STATE.update(engine=eng, tok=tok, drafter=drafter, k=a.k or a.depth,
                 model=a.served_model, started=int(time.time()), verbose=a.verbose,
                 max_len=int(a.max_len), default_max_tokens=int(a.default_max_tokens),
                 reasoning_format=a.reasoning_format, request_timeout=float(a.request_timeout),
                 max_queue=int(a.max_queue), queue_timeout=float(a.queue_timeout),
                 draining=False,
                 cfg_eos=cfg_eos, think_budget=a.think_budget, think_stall=a.think_stall,
                 reasoning_effort=a.reasoning_effort,
                 answer_reserve=thinking_mod.parse_reserve(a.answer_reserve),
                 pen_spec=PenaltySpec(a.rep_penalty, a.presence_penalty,
                                     a.frequency_penalty, a.no_repeat_ngram),
                 temperature=a.temperature, top_p=a.top_p, top_k=a.top_k,
                 pattern_stop=(tuple(int(x) for x in a.pattern_stop.split(":"))
                               if a.pattern_stop else None),
                 argloop=a.argloop,
                 prefill_heartbeat=float(a.prefill_heartbeat_s),
                 response_cache=rcache, suffix_scope=a.suffix_store_scope,
                 usage_default=(a.usage_default == "on"), **model)


def _load(a) -> None:
    """The real engine: weights, drafters, caches, the warm-up and the verify graphs (SRV-341, one
    loader for every model). The model's own part -- its weights, engine, tokenizer and drafter --
    comes from the builder its registry entry names (`Arch.serve`, chosen by the checkpoint's config;
    the 27B's is the server's own, `_build_qwen3_5`); what the engine cannot hold yet its adapter's
    capabilities say (`eng.caps`: the caches, the KV pool and side slots), and the rest is every
    model's."""
    if _S.get("BLOCKING_SYNC") == "1":
        _blocking_sync()
    t0 = time.time()
    # the price table first: a bad file must stop the start before minutes of weight loading
    from engine import prices as _prices
    _pname, _pentry = _prices.load(getattr(a, "price_table", ""))
    PRICES.clear()
    PRICES.update(_pentry)
    if _pname:
        STATE["price_table"] = _pname
        print(f"[prices] {_pname}: " + ", ".join(f"{k}={v}" for k, v in _pentry.items()))
    from engine.tokfp import fingerprint as _tok_fp
    # OPS-41: the allocator and MemAvailable after each stage, the residency planner's split
    led = STATE["load_ledger"] = memsplit.LoadLedger()
    led.mark("start")

    def keys(path, weights) -> None:
        STATE["tokenizer_sha"] = _tok_fp(path)      # the suffix stores check it
        # ENG-181: the probe snapshots and a probe file carry it; a probe refuses other weights
        STATE["weights_key"] = probe_mod.weights_key(weights)

    # what a builder may ask of the server: the keys above, the ledger, the log, the KV pool, the
    # vision tower with its processor and caches (`_load_vision`, a builder's own tower)
    ctx = types.SimpleNamespace(keys=keys, mark=led.mark, log=lambda line: print(line, flush=True),
                                kv_pool=lambda: _kv_pool_open(a),
                                vision=lambda path, tower_of=None, config_of=None:
                                _load_vision(a, path, tower_of, config_of))
    m = (_arch_of(a.model).serve_builder() or _build_qwen3_5)(a, ctx)
    eng, tok, drafter = m["engine"], m["tok"], m["drafter"]
    memsplit.PINNED = lambda: memsplit.pinned_of(eng)      # every floor check counts it (SRV-335)
    gen_cfg = os.path.join(m["path"], "generation_config.json")
    cfg_eos = None
    if os.path.isfile(gen_cfg):
        cfg_eos = json.load(open(gen_cfg)).get("eos_token_id")
    # SRV-475: the checkpoint's own sampling, what the loop escape and `agentic_default: card`
    # decode with (none: the regenerate's settings)
    STATE["card_sampling"] = loopescape_mod.card_sampling(m["path"])
    print(f"[sampling] the card's: {STATE['card_sampling'] or 'none (generation_config.json)'}",
          flush=True)
    rcache = (cache.ResponseCache(int(a.response_cache_mb * (1 << 20)), a.response_cache_ttl)
              if a.response_cache else None)
    if eng.caps.caches:
        if eng.caps.anchor_grid:
            a.prefix_chunk = int(eng.caps.anchor_grid)     # the model's own grid (its prefill chunk)
            STATE["anchor_grid"] = a.prefix_chunk          # the memory plan shows it (server/memlimits.py)
        caches = _load_caches(a, eng, tok, drafter, rcache)
        from engine import model as model_mod
        model_mod.WINDOW_YIELD = _window_layer          # SRV-591, SRV-321
    else:
        if not a.no_prefix_cache or not a.no_session_cache or getattr(a, "kv_pool", "off") == "on":
            print("[cache] prefix, session, resident and disk caches and the KV pool are not built for "
                  "this model yet (ENG-157): all off", flush=True)
        caches = {"state_store": None, "session_cache": False, "prefix_cache": False, "session_trim": False,
                  "resident": None, "prefix_chunk": 0, "suffix_store": None}
    STATE["segments"] = (SegmentCache(tok, int(a.prompt_token_cache * 1e6))
                         if a.prompt_token_cache > 0 else None)
    _serving_state(a, eng, tok, drafter, cfg_eos, rcache, device=m["device"], **m["serving"], **caches)
    led.mark("caches")
    print(f"[server] {m['describe']} loaded in {time.time() - t0:.1f}s", flush=True)
    # one warm request, so the first measured one is not paying for Triton autotuning; a drafter
    # that prices its rounds (one with `warming`) decodes WARM_TOKENS through them, learning nothing
    warming = getattr(drafter, "warming", None) if drafter is not None else None
    with torch.no_grad(), (warming() if warming is not None else contextlib.nullcontext()):
        list(generate_stream(tok("warm up the kernels", return_tensors="pt").input_ids[0].to(m["device"]),
                             getattr(drafter, "WARM_TOKENS", 4) if warming is not None else 4, set()))
        # FX17: the prefill shapes a first real request brings (chunk-sized blocks and a tail),
        # so the first requests after a restart do not pay for them (run 3 on 0.4.8-test: the
        # first two chats queued 24 s behind the router's cold questions)
        _warm_prefill(generate_stream, tok, STATE.get("resident"))
    led.mark("warmup")
    # the verify graphs for every width at the first context class, here rather than
    # inside the first requests (one capture is a fraction of a second, thirty are several)
    t_g = time.time()
    with torch.no_grad():
        n_g = eng.precapture(drafter)
    if n_g is not None:
        print(f"[server] verify graphs: {n_g} captured in {time.time() - t_g:.1f}s", flush=True)
    led.mark("graphs")
    _kv_after_load(a)
    if eng.kv.pages is not None:
        # the warm-up's and the captures' rows are nobody's conversation
        eng.kv.pages.trim(_kv_keep_rows(eng, STATE.get("resident")))
    if getattr(a, "side_lane", "off") != "off" and not eng.caps.side_lane:
        a.side_lane = "off"
        print("[side-lane] off: this model's engine is not lent to a short request inside another "
              "one's generation yet (caps.side_lane)", flush=True)
    if int(getattr(a, "parallel", 1) or 1) > 1:
        if eng.caps.side_slots:
            _build_side_slots(a, m["cfg"], m["w"], tok)
            led.mark("side_slots")
        else:
            print(f"[parallel] --parallel {a.parallel}: side slots are not built for this model yet "
                  f"(ENG-241): one slot", flush=True)
    print("[memory] load stages, GiB allocated / taken from MemAvailable: " + ", ".join(
        f"{x['stage']} {_gib(x['allocated'])}/{_gib(x['mem_available_drop'])}"
        for x in led.stages()), flush=True)


def _kv_pool_open(a) -> None:
    """SRV-307: the shared KV pool (engine/kvpool.py) with `--kv-pool on`, before the engine
    allocates its window (a builder asks for it: the 27B's, and any other's through `ctx.kv_pool`)."""
    if getattr(a, "kv_pool", "off") != "on" or kvpool.POOL is not None:
        return
    kvpool.POOL = kvpool.KVPool(hard_floor=int(a.kv_hard_floor_gb * (1 << 30)),
                                margin_rows=int(a.kv_margin_rows), meminfo=memsplit.counted_available,
                                on_event=_kv_pool_event)
    kvpool.OWNER = "slot0"
    if kvpool.Driver.get() is None:
        print("[kv-pool] no CUDA virtual memory management on this device: the pool keeps "
              "only its accounts, every buffer is allocated in full", flush=True)


def _build_qwen3_5(a, ctx) -> dict:
    """The 27B (and every `qwen3_5` checkpoint): its weights (NVFP4, FP8 head), the KV pool, the
    engine, the tokenizer, the vision tower and the drafter `--drafter` names. `ctx` as for a
    registry builder (engine/models `Arch.serve`)."""
    from engine.config import load_config
    from engine.loader import Weights
    from engine.models.qwen3_5.adapter import Qwen38Engine   # with the adapter's cold verbs
    from transformers import AutoTokenizer
    cfg = load_config(a.model)
    ctx.keys(cfg.path, [cfg.path, a.nvfp4, a.fp8_head])
    w = Weights(cfg.path, skip_mtp=a.drafter in ("none", "dflash2", "merged", "lenrouter"),
                nvfp4=a.nvfp4,
                fp8_head=a.fp8_head)
    ctx.mark("weights")
    _kv_pool_open(a)
    eng = Qwen38Engine(cfg, w, max_len=a.max_len)
    ctx.mark("engine")
    tok = AutoTokenizer.from_pretrained(cfg.path)
    _load_vision(a, cfg.path)
    ctx.mark("vision")
    drafter = None
    if a.drafter == "mtp":
        from engine.drafters.mtp import MTPDrafter
        drafter = MTPDrafter(eng, max_len=a.max_len, depth=a.depth, draft_head=a.draft_head)
    elif a.drafter == "router":
        from engine.router import RouterDrafter
        drafter = RouterDrafter(eng, max_len=a.max_len, depth=a.depth)
    elif a.drafter == "merged":
        # The configuration the 11:18 gate passed on, plus the tree: the block drafter priced
        # against the lookup drafter every step, both putting candidates in one verify call.
        from engine.drafters.dflash2 import DFlash2Drafter
        from engine.drafters.ngram import NgramDrafter
        from engine.router import MergedRouter, VERIFY_MS, VERIFY_MS_NVFP4
        table = VERIFY_MS_NVFP4 if a.nvfp4 or _S.get("NVFP4") else VERIFY_MS
        head = DFlash2Drafter(eng, a.dflash2_ckpt, blocks=1, max_len=a.max_len,
                              path=a.dflash2_path, draft_head=a.draft_head)
        head.tree_temp = a.df2_temp
        head._build()
        # `--budget` counts nodes INCLUDING the anchor, because that is what the measured curve is
        # keyed by and where its cliff is: 16 nodes cost 164.4 ms and 17 cost 172. So the drafters
        # get one fewer.
        ng = NgramDrafter(corpus_path=a.corpus, tokenizer_sha=STATE.get("tokenizer_sha"), min_order=3, max_depth=16,
                          node_budget=a.budget - 1, branch_top_k=3, min_expected=0.2,
                          alpha=0.6, corpus_weight=0.5, min_corpus_order=8,
                          verify_base_ms=table[min(table)],
                          verify_per_node_ms=(table[max(table)] - table[min(table)])
                          / (max(table) - min(table)))
        drafter = MergedRouter(ng, head, mtp_depth=head.cfg.block_size - 1,
                               node_budget=a.budget - 1, mtp_ms_per_token=0.0,
                               head_fixed_ms=35.0, adaptive_depth=False,
                               rollback_ms=6.2, verify_ms_table=table)
        a.depth = a.budget - 1
    elif a.drafter == "dflash2":
        from engine.drafters.dflash2 import DFlash2Drafter
        drafter = DFlash2Drafter(eng, a.dflash2_ckpt, blocks=a.dflash2_blocks,
                                 path=a.dflash2_path, draft_head=a.draft_head,
                                 max_len=a.max_len)
        drafter._build()
        # The block width, not `--depth`, is what this drafter proposes per verify pass.
        a.depth = (drafter.cfg.block_size - 1) * a.dflash2_blocks
    elif a.drafter == "lenrouter":
        drafter = _make_lenrouter(a, eng, a.max_len)
    if a.narrow_arm == "off" and a.drafter != "lenrouter":
        raise SystemExit("[server] --narrow-arm off is the length router's (--drafter lenrouter)")
    ctx.mark("drafters")
    relax = Relax(a.relax_tau, a.relax_rank)
    print(f"[server] {w.report()}")
    if relax.on:
        print(f"[server] LOSSY ACCEPT RULE ON: tau={relax.tau} rank={relax.rank}. Output is not "
              f"greedy and not reproducible against a run with tau 1 and rank 1. See "
              f"docs/exactness.md")
    describe = (f"drafter={a.drafter} depth={a.depth} k={a.k or a.depth} "
                f"nvfp4={w.nvfp4_source or 'off'} fp8_head={(w.fp8_head_source if str(w.fp8_head_source).startswith('build') else 'on') if w.fp8_head_source else 'off'} ")
    return {"engine": eng, "tok": tok, "drafter": drafter, "path": cfg.path, "device": "cuda",
            "cfg": cfg, "w": w, "describe": describe,
            "serving": {"tree": bool(a.tree), "sampled_tree": a.sampled_tree, "relax": relax}}


def _load_caches(a, eng, tok, drafter, rcache) -> dict:
    """The serving-time caches (engine/cache.py) of an engine that can hold them (`eng.caps.caches`):
    the memory policy, the state store, the resident prefix and its end anchor, the suffix store.
    Returns their entries of the request-path state."""
    session_on = not a.no_session_cache
    prefix_on = not a.no_prefix_cache
    budget = int(a.cache_budget_gb * (1 << 30))
    if (session_on or prefix_on) and not cache.drafter_is_cacheable(drafter):
        # This drafter's cache is indexed by absolute position and it cannot hand it over. A
        # restored prefix would leave a permanent hole in it and the engine would decode at one
        # token a block, which is a far worse trade than a cold prefill.
        print(f"[cache] drafter {a.drafter} cannot snapshot its own cache; state cache OFF")
        session_on = prefix_on = False
    resident_on = (prefix_on and a.resident_gb > 0 and a.prefix_chunk > 0
                   and cache.resident_capable(drafter))
    # OPS-47: one memory policy for every cache -- the engine's ceiling between requests, its
    # declared peak during a prefill, the reserve it may keep -- and each cache under its own cap
    # (the pin is its own, no longer paid out of the store's budget)
    gib = 1 << 30
    cost = eng.cost()                      # a model's own prefill workspace (SRV-335), else the 27B's
    mem = STATE["mem"] = cache.Budget(int(a.mem_ceiling_gib * gib), int(a.mem_peak_gib * gib),
                                      int(a.reserve_slack_gib * gib),
                                      exclude=lambda: memlimits.role_exclude(STATE),
                                      ws_floor=cost.workspace_floor or cache.WORKSPACE_FLOOR,
                                      ws_per_token=cost.workspace_per_token or cache.WORKSPACE_PER_TOKEN,
                                      ws_cap=cost.workspace_cap, on_event=_memory_event)
    # The roles' pools. Their free segments count in `reserved` and only dropping
    # the pool gives them back: the gate's last level asks the shadow reader to (`release`), and
    # the stale-reserve rule leaves them out of what the engine's allocator holds free
    mem.register("roles", _release_roles)
    mem.foreign_free = lambda: memlimits.role_pool_free(STATE)
    pin = int(a.resident_pin_gb * gib) if resident_on else 0
    store = cache.StateStore(budget, chunk=a.prefix_chunk,
                             max_entry_bytes=int(a.store_entry_gb * gib),
                             per_prefill=a.store_boundaries) if (budget and
                                                                 (session_on or prefix_on)) else None
    # the resident prefix needs the prefill's chunk grid, which only the prefix cache has
    resident = (cache.ResidentPrefix(int(a.resident_gb * gib), a.prefix_chunk,
                                     tail=a.resident_tail,
                                     stash_bytes=int(a.resident_stash_gb * gib),
                                     guest_max=a.resident_guest_max, pin_bytes=pin,
                                     park_bytes=int(a.resident_park_gb * gib), mem=mem,
                                     end=a.end_anchor == "on")
                if resident_on else None)
    if store is not None:
        mem.register("store", store.shed)
    if resident is not None:
        resident.glue = glue_ids(tok)
        if resident.end_on:
            STATE["end_align"] = endalign.EndAlign(
                endalign.token_bytes(tok), on=a.end_align == "on",
                reasoning=(_S.get("END_ALIGN_REASONING") or "on").strip().lower() != "off",
                moved=(_S.get("END_ALIGN_MOVED") or "on").strip().lower() != "off")
            STATE["end_align"].marks = _marks_of(tok)       # FX16: the moved-block pass
        # SRV-546: the guest stash's buffers while no guest's rows wait in them, before anything
        # that holds a conversation (live 10:14:34: the end anchor went for 0.31 GB, then 1.27 GB
        # of idle stash, and the next turn read the chat again)
        mem.register("stash_idle", lambda safe: resident.shed_stash(True))
        mem.register("anchors", resident.shed_anchors)
        mem.register("end", resident.shed_end)
        mem.register("stash", resident.shed_stash)
        mem.register("park", lambda safe=False: _shed_park(resident, safe))
        mem.register("pin", resident.shed_pin)
    suffix = None
    if a.suffix_store:
        try:
            suffix = cache.PersistentSuffixStore(
                a.suffix_store, max_tokens=int(a.suffix_store_mb * (1 << 20)) // 4,
                readonly=a.suffix_store_readonly, tokenizer_sha=STATE.get("tokenizer_sha")).open()
        except ValueError as exc:
            # a store another model's tokenizer wrote (engine/tokfp.py): its ids mean other tokens here
            print(f"[cache] suffix store off: {exc}", flush=True)
    if suffix is not None:
        reader = next((d for d in (drafter, getattr(drafter, "ngram", None),
                                   getattr(drafter, "engram", None))
                       if hasattr(d, "add_store")), None)
        if reader is not None:
            reader.add_store(suffix)
        else:
            print(f"[cache] drafter {a.drafter} has no lookup store; the suffix store is being "
                  f"written but nothing reads it")
    print(f"[cache] session={'on' if session_on else 'off'} "
          f"prefix={'on' if prefix_on else 'off'}/{a.prefix_chunk} "
          f"budget={a.cache_budget_gb:.0f} GiB "
          f"response={'on' if rcache else 'off'} "
          f"resident={(f'{a.resident_gb:g}+{a.resident_stash_gb:g} GiB' if resident else 'off')} "
          f"pin={pin / (1 << 30):g} GiB park={a.resident_park_gb if resident else 0:g} GiB "
          f"end={'on' if resident is not None and resident.end_on else 'off'} "
          f"entry={a.store_entry_gb:g} GiB "
          f"store-boundaries={a.store_boundaries} "
          f"suffix={(suffix.report()['tokens'] if suffix else 0)} tokens")
    print(f"[memory] ceiling {a.mem_ceiling_gib:g} GiB between requests, peak {a.mem_peak_gib:g} "
          f"GiB during a prefill (the workspace at {a.max_len:,} tokens: "
          f"{mem.workspace(a.max_len) / gib:.1f} GiB), reserve slack {a.reserve_slack_gib:g} GiB; "
          f"narrow arm {a.narrow_arm}", flush=True)
    return {"state_store": store, "session_cache": session_on, "prefix_cache": prefix_on,
            "session_trim": (a.session_trim == "on"), "resident": resident,
            "prefix_chunk": cache.prefill_chunk(prefix_on, a.prefix_chunk, a.max_prefill_rows),
            "suffix_store": suffix}


def _par_room() -> int | None:
    """SRV-250: the engine's room to `--mem-peak-gib` under its memory policy (the reservation
    now against the peak), None without a policy. Every cache asks `Budget.admit` before it grows
    and every prefill passes `Budget.gate`, so the caches, the requests in flight and a new one
    take at most this together; the caches' budgets alone counted 6.4 to 8.4 GiB the policy would
    never have let them grow."""
    mem = dict.get(STATE, "mem")
    if mem is None or not getattr(mem, "on", False):
        return None
    _, reserved = mem.memory()
    return max(0, int(mem.peak) - int(reserved))


def _par_growth() -> int:
    """What the long-lived caches may still take: each one's budget less what it holds."""
    g = 0
    store, res, rc = (dict.get(STATE, "state_store"), dict.get(STATE, "resident"),
                      dict.get(STATE, "response_cache"))
    for obj, budget in ((store, "budget"), (rc, "budget")):
        if obj is not None:
            try:
                g += max(0, int(getattr(obj, budget)) - int(getattr(obj, "bytes", 0)))
            except Exception:                                      # noqa: BLE001
                pass
    if res is not None:
        try:
            total = sum(int(getattr(res, k, 0) or 0) for k in ("budget", "stash_budget",
                                                               "pin_budget", "park_budget"))
            g += max(0, total - int(getattr(res, "bytes", 0) or 0))
        except Exception:                                          # noqa: BLE001
            pass
    return g


def _par_sleep(seconds: float) -> None:
    """`time.sleep` for the fake engine (server/fake_engine.py): with side slots, the GPU turn is
    given up while the fake's "forward" sleeps, as a real slot gives it up between blocks."""
    pool = dict.get(STATE, "pool")
    if pool is None or pool.turn is None:
        time.sleep(seconds)
        return
    i = _par_index()
    held = pool.turn.held_by(i)
    if held:
        pool.turn.release(i)
    try:
        time.sleep(seconds)
    finally:
        if held:
            pool.turn.acquire(i)


def _par_fake_slots(a, engine_cls) -> None:
    """`--parallel N` on the fake engine: N - 1 side slots with their own fake engines, the turn,
    and the admission on `FAKE_MEMAVAILABLE_GIB` (none: every request is admitted). No joint verify:
    the fake has no tree to verify."""
    L = int(min(a.side_max_len, a.max_len))
    gib = slots_mod.GiB
    avail = float(os.environ.get("FAKE_MEMAVAILABLE_GIB") or 0)
    meminfo = (lambda: {"MemAvailable": int(avail * gib)}) if avail else (lambda: None)
    adm = slots_mod.Admission(int(a.parallel_floor_gb * gib), meminfo=meminfo, growth=lambda: 0)
    slots = [slots_mod.Slot(0, int(a.max_len), values=None, lock=LOCK)]
    for i in range(1, int(a.parallel)):
        values = {k: None for k in slots_mod.OVERLAY}
        values.update(engine=engine_cls(), drafter=None, max_len=L)
        slots.append(slots_mod.Slot(i, L, values=values))
    turn = slots_mod.Turn(a.turn_quantum_ms / 1e3)
    STATE["pool"] = slots_mod.Pool(slots, turn=turn, meeting=None, admission=adm)
    LOCK.turn = turn
    print(f"[parallel] FAKE: {len(slots)} slots, side window {L}", flush=True)


#: A side slot's bytes beyond its window with `--side-graphs off`: its drafter stack's scratch,
#: the verify scratch and the static buffers, no graph pool (PL1, measured on the box: see
#: docs/parallel-requests.md "What a slot costs").
SIDE_CONST_NO_GRAPHS_GB = 1.5


def _side_const_gb(a) -> float:
    v = getattr(a, "side_slot_const_gb", None)
    if v:
        return float(v)
    return 7.5 if getattr(a, "side_graphs", "on") == "on" else SIDE_CONST_NO_GRAPHS_GB


#: FX17: the warm-up prefill's length: past two chunks of the grid (1,024) plus a tail, the shape
#: of the router's package question (about 2,400 tokens) and of a short chat's first turn
WARM_PREFILL_TOKENS = int(_S.get("WARM_PREFILL_TOKENS") or 2600)


def _warm_prefill(gen, tok, res=None) -> None:
    """A warm-up generation over a WARM_PREFILL_TOKENS-token prompt on the slot the caller bound:
    the prefill kernels at their chunk-sized shapes and the tail, and one block. Never fails a
    start; `QWEN38_WARM_PREFILL_TOKENS=0` skips it."""
    n = int(WARM_PREFILL_TOKENS)
    if n <= 0:
        return
    try:
        import torch
        words = tok("warm up the kernels ", add_special_tokens=False).input_ids
        ids = (words * (n // max(1, len(words)) + 1))[:n]
        t0 = time.time()
        list(gen(torch.tensor(ids, dtype=torch.long).cuda(), 2, set()))
        if res is not None:
            res.clear()                      # the warm-up's rows are nobody's conversation
        print(f"[server] warm-up prefill of {n} tokens in {time.time() - t0:.1f}s", flush=True)
    except Exception as exc:                                       # noqa: BLE001
        print(f"[server] warm-up prefill skipped: {type(exc).__name__}: {exc}", flush=True)


def _side_resident(a, tok):
    """FX16: a side slot's own resident prefix (engine/cache.py `ResidentPrefix`), or None with
    `QSE_SIDE_RESIDENT_GB=0` or when slot 0 has none. The anchors it holds, its stash for a guest
    and its end anchor are sized by `SIDE_RESIDENT_GB` / `SIDE_STASH_GB`; no pin, no park, no
    disk tier and no NVMe spill of its own (`OVERLAY` keeps those None on a side slot). Its
    buffers are given back under the memory policy's levels with slot 0's, the side's after.

    Before 0.4.8 a side slot kept no conversation: the next turn of a chat that ran there was
    carried into slot 0 (`_carry_from_side`), which moved slot 0's conversation out (2026-10-08
    12:12:58: an 83,160-token chat to the NVMe, 4.1 s), and from then on the two chats took turns
    on slot 0 with the side slot idle. With a resident prefix the turn resumes on its own slot
    (`_side_home`) and slot 0's conversation stays where it is."""
    res0 = dict.get(STATE, "resident")
    gb = float(SIDE_RESIDENT_GB)
    if res0 is None or gb <= 0:
        return None
    gib = 1 << 30
    mem = dict.get(STATE, "mem")
    # `guest_frac` 1: every request up to `guest_max` that shares under a chunk with the rows is
    # a guest here (the router's question, a title task, a new short chat), so the conversation
    # the slot holds survives it through the stash whatever its length
    res = cache.ResidentPrefix(int(gb * gib), int(a.prefix_chunk), tail=a.resident_tail,
                               stash_bytes=int(float(SIDE_STASH_GB) * gib),
                               guest_max=a.resident_guest_max, guest_frac=1.0, pin_bytes=0,
                               park_bytes=0, mem=mem, end=res0.end_on)
    res.glue = glue_ids(tok)
    res.keeper = _keep_parent             # ENG-371: a parent's rows to the NVMe before they go
    if mem is not None:
        for level, fn in (("anchors", res.shed_anchors), ("end", res.shed_end),
                          ("stash", res.shed_stash)):
            prev = mem.shedders.get(level)

            def both(safe=False, p=prev, f=fn):
                n = int(p(safe) or 0) if p is not None else 0
                try:
                    return n + int(f(safe) or 0)
                except Exception:                                  # noqa: BLE001
                    return n
            mem.register(level, both)
    return res


def _build_side_slots(a, cfg, w, tok) -> None:
    """Slots 1..N-1 (`--parallel N`): each a `Qwen38Engine` over the same weights at
    `--side-max-len`, with its own drafter stack over the same checkpoints (engine/drafters/dflash2
    caches the weights) and the same lookup corpus and suffix store. A slot is built only if the
    memory floor holds with its bytes added -- computed before anything is allocated -- and is
    warmed and its verify graphs captured before the server answers. The engine's own memory
    policy (engine/cache.py `Budget`) then counts what the side slots took as the engine's: its
    ceiling and peak go up by that much, so slot 0's caches are not shed for it."""
    from engine.models.qwen3_5.adapter import Qwen38Engine   # with the adapter's cold verbs
    n = int(a.parallel)
    # ENG-241: an engine that builds its own side slots (`side_slot`: its state, graphs and drafter over
    # the same weights); else the 27B's, below
    make = getattr(STATE["engine"], "side_slot", None)
    if make is None and a.drafter not in ("lenrouter", "none"):
        raise SystemExit(f"[parallel] --parallel {n} is built for --drafter lenrouter (or none), "
                         f"not {a.drafter}")
    L = int(min(a.side_max_len, a.max_len))
    floor = int(a.parallel_floor_gb * slots_mod.GiB)
    adm = slots_mod.Admission(floor, meminfo=memsplit.counted_meminfo, growth=_par_growth, room=_par_room,
                              committed=lambda o: (kvpool.POOL.owner_bytes(o)
                                                   if kvpool.POOL is not None else 0))
    drafter0 = STATE["drafter"]
    ng0 = getattr(drafter0, "ngram", None)
    slots = [slots_mod.Slot(0, int(a.max_len), values=None, lock=None)]
    a0, r0 = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
    depth0 = a.depth
    paged = kvpool.POOL is not None and kvpool.Driver.get() is not None
    not_built = None
    for i in range(1, n):
        # SRV-307: with the pool a slot's window is addresses; only its rotary table (256 B a
        # token) and its constant are memory at load
        if make is not None:
            per_token, const = STATE["engine"].side_slot_bytes(L, paged)
            need = slots_mod.slot_bytes(L, per_token=per_token, constant=const)
        else:
            need = slots_mod.slot_bytes(L, per_token=256 if paged else 106_752,
                                        constant=int(_side_const_gb(a) * slots_mod.GiB))
        ok, d = adm.check(need, count=False)
        if not ok:
            not_built = (f"slot {i} NOT built (window {L:,}, ~{_gib(need)} GiB): "
                         f"{slots_mod.Admission.describe(d)}")
            print(f"[parallel] {not_built}", flush=True)
            break
        mi0 = slots_mod.read_meminfo() or {}
        a1 = torch.cuda.memory_allocated()
        t0 = time.time()
        kvpool.OWNER = f"slot{i}"
        if make is not None:
            e2, d2 = make(L, drafter0)
        else:
            e2 = Qwen38Engine(cfg, w, max_len=L)
            e2.graphs_off = a.side_graphs == "off"
            d2 = (_make_lenrouter(a, e2, L, corpus_from=ng0) if a.drafter == "lenrouter" else None)
        a.depth = depth0
        values = {k: None for k in slots_mod.OVERLAY}
        values.update(engine=e2, drafter=d2, max_len=L)
        res_i = _side_resident(a, tok)
        if res_i is not None:
            values.update(resident=res_i, end_align=endalign.EndAlign(
                endalign.token_bytes(tok), on=a.end_align == "on",
                reasoning=(_S.get("END_ALIGN_REASONING") or "on").strip().lower() != "off",
                moved=(_S.get("END_ALIGN_MOVED") or "on").strip().lower() != "off"))
            values["end_align"].marks = _marks_of(tok)
        slot = slots_mod.Slot(i, L, values=values)
        slots.append(slot)
        slots_mod.bind(slot)
        try:
            warming = getattr(d2, "warming", None) if d2 is not None else None
            with torch.no_grad(), (warming() if warming is not None else contextlib.nullcontext()):
                list(_generate_stream(tok("warm up the kernels", return_tensors="pt")
                                      .input_ids[0].cuda(),
                                      getattr(d2, "WARM_TOKENS", 4) if warming is not None else 4, set()))
                _warm_prefill(_generate_stream, tok, res_i)    # FX17: the router's question shape
                if make is not None:
                    e2.precapture(d2)          # its class 0 graphs now, not in its first request
                elif a.side_precapture == "on" and e2._graphs_for(2, 0) is not None:
                    e2._graphs.precapture(widths=range(2, 33) if getattr(d2, "calc", False)
                                          else range(2, 17))
            values.update({k: None for k in ("last_ctx", "last_prompt_n", "hidden_spans",
                                             "blocks", "last_prefill")})
            if res_i is not None:
                res_i.clear()                 # the warm-up's rows are nobody's conversation
            if e2.kv.pages is not None:
                e2.kv.length = 0
                e2.kv.pages.trim(0)           # SRV-307: and their pages go back to the pool
        finally:
            slots_mod.bind(None)
            kvpool.OWNER = "slot0"
        torch.cuda.synchronize()
        mi1 = slots_mod.read_meminfo() or {}
        print(f"[parallel] slot {i}: window {L:,}, built in {time.time() - t0:.1f}s, "
              f"+{_gib(torch.cuda.memory_allocated() - a1)} GiB allocated, MemAvailable "
              f"{_gib(mi0.get('MemAvailable'))} -> {_gib(mi1.get('MemAvailable'))} GiB "
              f"(estimate {_gib(need)})", flush=True)
    if len(slots) < 2:
        print(f"[parallel] no side slot fits under the {a.parallel_floor_gb:g} GiB floor: one "
              f"sequence, as without --parallel", flush=True)
        return
    added_a = torch.cuda.memory_allocated() - a0
    added_r = torch.cuda.memory_reserved() - r0
    mem = dict.get(STATE, "mem")
    if mem is not None and mem.on:
        side = max(added_a, added_r)
        mem.fixed += side          # kept when a residency document sets the limits again
        mem.ceiling += side
        mem.peak += side
        print(f"[parallel] memory policy: ceiling and peak +{_gib(max(added_a, added_r))} GiB for "
              f"the side slots ({_gib(mem.ceiling)} / {_gib(mem.peak)} GiB)", flush=True)
    settle = torch.cuda.synchronize if a.parallel_settle == "on" else None
    turn = slots_mod.Turn(a.turn_quantum_ms / 1e3, settle=settle)
    meet = (slots_mod.Meeting(a.parallel_max_rows, a.batch_timeout_ms / 1e3,
                              cut=a.parallel_cut)
            if a.parallel_batch == "on" and STATE["engine"].caps.joint_verify else None)
    timing = os.environ.get("QSE_PARALLEL_TIMING") == "1"
    if meet is not None:
        meet.timing = timing
    dmeet = None
    if meet is not None and a.parallel_draft == "on" and a.drafter == "lenrouter":
        # the packed draft: the decoding slots' draft calls in one pass over the drafter
        from engine.drafters import dflash2 as _df2
        dmeet = slots_mod.Meeting(a.parallel_max_rows, a.batch_timeout_ms / 1e3)
        dmeet.timing = timing
        _df2.DRAFT_MEET = _draft_meet
    pool = slots_mod.Pool(slots, turn=turn, meeting=meet, admission=adm, draft_meeting=dmeet)
    pool.requested, pool.limit_reason = n, not_built       # /health's `max_running_reason`
    slots[0].lock = LOCK
    if a.parallel_switch_ms > 0:
        # a slot woken for its turn waits for the GIL up to the interpreter's switch interval (5 ms
        # by default) while another thread runs Python; the turn changes hands several times a round
        sys.setswitchinterval(a.parallel_switch_ms / 1e3)
    STATE["pool"] = pool
    LOCK.turn = turn
    if PAR_WINDOW_ROWS > 0:
        cache.WINDOW_CAP = _par_window_cap      # SRV-309
    if PAR_LAYER_YIELD and turn is not None:
        WINDOW_TURN[0] = True                           # SRV-321, in `_window_layer`
    print(f"[parallel] {pool.n} slots: slot 0 window {a.max_len:,}, {pool.n - 1} side slot(s) "
          f"window {L:,}; turn quantum {a.turn_quantum_ms:g} ms; joint verify "
          f"{'up to ' + str(a.parallel_max_rows) + ' rows' if meet else 'off'}; packed draft "
          f"{'on' if dmeet is not None else 'off'}; admission floor "
          f"{a.parallel_floor_gb:g} GiB MemAvailable", flush=True)


def _gib(b) -> str:
    return "-" if b is None else f"{b / (1 << 30):.2f}"


def _load_vision(a, path: str, tower_of=None, config_of=None) -> None:
    """The checkpoint's vision tower (bf16, as stored; or `tower_of(vision config)`, the tower a
    model's builder makes in its own weight format, VIS-97), its image processor and the embedding
    cache -- when the checkpoint has a tower and `--vision on`. `config_of(path)`: a builder's own
    vision config reader (Super VL's RADIO tower, ENG-365), whose tower brings its own processor
    (`preprocessor`), warm-up (`warm`) and prompt rows (`context`)."""
    from engine.vision import EmbedCache, VisionTower, load_vision_config
    vcfg = (config_of or load_vision_config)(path) if getattr(a, "vision", "on") == "on" else None
    if vcfg is None:
        print("[vision] off" + (" (the checkpoint has no vision tower)"
                                if getattr(a, "vision", "on") == "on" else ""), flush=True)
        return
    t0 = time.time()
    tower = (tower_of(vcfg) if tower_of is not None
             else VisionTower.from_checkpoint(path, vcfg, device="cuda"))
    if tower is None:
        print("[vision] off (the checkpoint has no vision tower)", flush=True)
        return
    own = getattr(tower, "preprocessor", None)
    pre = (own(int(a.image_max_pixels)) if own is not None else
           images_mod.Preprocessor(path, vcfg.spatial_merge_size,
                                   max_pixels=int(a.image_max_pixels), dtype=tower.dtype,
                                   backend=getattr(a, "image_backend", "torchvision")))
    STATE.update(vision=tower, image_pre=pre,
                 image_cache=EmbedCache(int(a.image_cache_mb * (1 << 20))),
                 image_limits=images_mod.Limits(
                     max_bytes=int(a.image_max_mb * (1 << 20)),
                     max_decode_pixels=int(a.image_max_decode_pixels),
                     max_images=int(a.max_images), timeout_s=float(a.image_fetch_timeout),
                     https=a.image_https == "on",
                     allow_private=a.image_https_private == "on",
                     max_image_rows=int(a.max_image_rows)),
                 image_act_budget=int(a.image_act_gb * (1 << 30)))
    # the kernels' first launch here, not in the first image request: a 4x4-patch image
    with torch.no_grad():
        if hasattr(tower, "warm"):
            tower.warm()
        else:
            m = vcfg.spatial_merge_size * 2
            tower.encode(torch.zeros(m * m, vcfg.patch_dim), (1, m, m))
    tower.stats.update(images=0, patches=0, ms=0.0)
    print(f"[vision] tower {tower.nbytes / 1e9:.2f} GB, {vcfg.depth} blocks, "
          f"{len(tower.t)} tensors; processor {type(pre.proc).__name__} "
          f"({pre.backend} backend, pinned); "
          f"loaded in {time.time() - t0:.1f}s", flush=True)


def _serve(a, led) -> None:
    """What every server does once its engine is loaded, the real one or the fake one."""
    STATE.update(identity())
    STATE["args"] = vars(a)
    STATE["thinking_default"] = getattr(a, "thinking_default", "on")     # SRV-477
    # The side lane (server/preempt.py); the fake engine brings its own runner
    side = getattr(a, "side_lane", "off")
    STATE["lane"] = (preempt.Lane(STATE.get("lane_run") or _lane_run, room=_lane_room,
                                  prompt_max=int(getattr(a, "side_lane_prompt_max",
                                                         preempt.PROMPT_MAX)),
                                  new_max=int(getattr(a, "side_lane_new_max", preempt.NEW_MAX)),
                                  where=side) if side != "off" else None)
    if STATE["lane"] is not None:
        STATE["lane"].gone_exc = ClientGone
    print(f"[side-lane] {side}" + (f": prompts up to {STATE['lane'].prompt_max} tokens, up to "
                                   f"{STATE['lane'].new_max} out, thinking off, raw or a side "
                                   f"request" if STATE["lane"] is not None else ""), flush=True)
    print(f"[server] {STATE['auth'].describe()}", flush=True)
    STATE["log_request_keys"] = bool(a.log_request_keys)
    # the fake engine's writer applies no logit processor, so it cannot honour a constraint
    STATE["structured_outputs"] = bool(getattr(a, "structured_outputs", True)) and not a.fake_engine
    # SRV-241: a regenerate samples (server/regenerate.py), the real engine and the fake alike
    STATE["regenerate"] = (regenerate_mod.Regenerates(
        a.regenerate, a.regenerate_temperature, a.regenerate_top_p, a.regenerate_top_k)
        if a.regenerate != "off" else None)
    # SRV-475: the loop escape's sessions (the card's sampling is read with the weights)
    STATE["escapes"] = loopescape_mod.Escapes()
    STATE["dashboard_dir"] = a.dashboard_dir
    STATE["live_hz"] = live_mod.hz_from_env(os.environ.get("QSE_LIVE_HZ"))
    STATE["live_activity"] = getattr(a, "live_activity", "on") == "on" and STATE["live_hz"] > 0
    STATE["live_windows"] = w = live_mod.windows_from_env(os.environ)
    STATE.pop("live", None)                      # built with the settings above
    pol = STATE.get("conductor_policy")
    mode = conductor.parse_mode(getattr(a, "conductor", "raw"), "raw")
    STATE["monitor"] = conductor.Monitor(mode, pol, STATE["tok"]) if pol is not None else None
    if STATE["monitor"] is not None:
        # the monitor's decisions are counted in Prometheus (SRV-115); its rows go into the
        # conductor's journal, opened below with the ledger
        STATE["monitor"].journal.sinks.append(metrics.on_conductor_event)
        if a.fake_engine:
            # a test run never pushes to a phone, whatever its policy says
            STATE["monitor"].notifier = lambda cfg, title, message: "suppressed: fake engine"
        if mode != "raw":
            STATE["monitor"].start_watch(float(getattr(a, "conductor_watch_s", 30.0)))
    print(f"[conductor] {mode}" + (f", policy {pol.path} version {pol.version}: " + ", ".join(
        f"{t.id}={t.mode}" for t in pol.tasks.values()) if pol is not None else ""), flush=True)
    print(f"[server] live activity {'on' if STATE['live_activity'] else 'off'}, "
          f"{STATE['live_hz'] if STATE['live_activity'] else 1} Hz while busy; live rates from "
          f"decoded token {w['decode_warmup_tokens']} and after prefill chunk "
          f"{w['prefill_warmup_chunks']}, over {w['now_window_s']:g} s", flush=True)
    if led is not None:
        STATE["ledger"] = led.open()
        print(f"[ledger] on: {led.path}, retention {led.retention_days} days", flush=True)
    else:
        print("[ledger] off", flush=True)
    STATE["port"] = a.port
    root = conductor_root(a)
    if root is not None and getattr(a, "conductor_defaults", "builtin") == "shipped":
        # The shipped documents a box has no version of yet, as version 1
        from server.conductor import shipped
        for line in shipped.install_at(root):
            print(f"[conductor] shipped default stored: {line}", flush=True)
    # One location per skill in opencode's skills list (server/skillcanon.py)
    STATE["skill_canon"] = (skillcanon_mod.SkillCanon(os.path.join(root, "skill-locations.json"))
                            if root is not None else None)
    # A token per session on the addresses the runtime notes give the model
    STATE["session_tokens"] = session_tokens_mod.SessionTokens(
        os.path.join(root, "session-tokens.json") if root is not None else None)
    # One journal for the router and the monitor (SRV-115): the table `conductor_events` in the
    # usage ledger's file (schema version 2), written from its own queue thread; with the ledger
    # off it lives in memory only. Opened when either half of the conductor runs.
    journal = None
    if root is not None or STATE["monitor"] is not None:
        from server.conductor.decisions import Journal as CondJournal
        from server.conductor.store import Store
        days = (Store(root, readonly=True).get("settings")[1]["journal"]["retention_days"]
                if root is not None else 30)
        journal = STATE["journal"] = CondJournal(STATE["ledger"].path if led is not None
                                                 else None, retention_days=days)
        if STATE["monitor"] is not None:
            STATE["monitor"].journal.sinks.append(journal.monitor_sink)
        print(f"[conductor] journal: {journal.path or 'in memory (no ledger)'}, "
              f"{days} days", flush=True)
    if root is not None:
        from server.conductor.hook import Conductor
        from server.conductor import schema as cond_schema
        cond_schema.SERVED_NAME = _identity_name()
        from server.conductor import modelfit as cond_fit
        _caps = getattr(STATE.get("engine"), "caps", None)
        print(cond_fit.install(getattr(_caps, "conductor", None)), flush=True)
        cond = STATE["conductor"] = Conductor(
            root, state=STATE, journal=journal,
            monitor_seed=STATE["monitor"].policy.data if STATE["monitor"] is not None else None)
        legacy = os.path.join(cond.root, "decisions.sqlite3")
        if journal.path and os.path.exists(legacy) and not os.path.exists(legacy + ".imported"):
            # the router's rows from before the journal moved into the ledger, copied once
            n = journal.import_legacy(legacy)
            with open(legacy + ".imported", "w") as f:
                f.write(f"{n} rows copied into {journal.path} on {time.ctime()}\n")
            print(f"[conductor] journal: {n} rows imported from {legacy}", flush=True)
        cond.router.session_tokens = STATE["session_tokens"]
        # stored documents a fix must reach (a shipped default reaches only a box without one)
        from server.conductor import migrations as cond_migrations
        cond_migrations.run(cond.store, cond.journal)
        st = cond.store.status()["documents"]
        print(f"[conductor] on: {cond.root} (router v{st['router']['version']}, "
              f"{len(cond.store.names())} profiles); model TandemLLM, {STATE['model']} a hidden "
              f"alias, raw by X-Tandem: raw",
              flush=True)
    else:
        print("[conductor] off: /v1/models lists only the raw engine", flush=True)
    start_proxy(a)
    # The decision models this engine serves itself (ENG-179, SRV-131): None, and nothing of
    # engine/decision imported, with QWEN38_DECIDER_MODELS empty. They load later, through the
    # residency planner: ops/start.sh asks for the on_start ones once /health answers and the page
    # cache is dropped, and a question for an on_demand one starts its load.
    try:
        STATE["deciders"] = decider_api.start(STATE, LOCK, fake=bool(a.fake_engine))
    except ValueError as exc:
        raise SystemExit(f"[deciders] {exc}")
    # The narrator (P1): a small generative model inside the engine that writes the state lines'
    # summaries. None with QWEN38_NARRATOR_MODEL empty; nothing loads here, its first line (a
    # profile with ui.narrator on) starts the load through the residency planner.
    try:
        STATE["narrator"] = narrator_mod.start(STATE, LOCK, fake=bool(a.fake_engine))
    except ValueError as exc:
        raise SystemExit(f"[narrator] {exc}")
    if STATE.get("narrator") is not None and not a.fake_engine:
        threading.Thread(target=_narrator_keeper, name="narrator-keeper", daemon=True).start()
    # SR1: the shadow reader (None with QWEN38_SHADOW_MODEL empty; it loads with the first turn
    # of a profile with shadow.mode on, through the residency planner)
    try:
        STATE["shadow"] = shadow_mod.start(STATE, LOCK, fake=bool(a.fake_engine))
    except ValueError as exc:
        raise SystemExit(f"[shadow] {exc}")
    if STATE.get("shadow") is not None:
        threading.Thread(target=_shadow_loop, name="shadow-reader", daemon=True).start()
    # the small models under memory pressure (server/rolemem.py): unloaded before a conversation
    # is parked or a request waits, loaded again once memory has stayed back
    _roles_start(a)
    # ENG-181: the probe snapshots (None, nothing allocated, with QWEN38_PROBE_SNAPSHOTS off;
    # TandemLLM requests only, so the conductor's server alone) and the router's probe stage
    try:
        STATE["probe"] = probe_mod.start(STATE, journal) if root is not None else None
    except ValueError as exc:
        raise SystemExit(f"[probe] {exc}")
    # C1: the disk tier under the caches in RAM (None, nothing opened, with --disk-cache off)
    STATE["disk_cache"] = diskcache_mod.start(STATE, a, journal)
    # Live conversations on the NVMe (engine/spill.py), and the one the last planned
    # stop wrote read back before the first request
    STATE["spill"] = diskcache_mod.start_spill(STATE, a, STATE["disk_cache"], journal)
    spill_startup()
    # Compaction as a continuation, and the summary prepared ahead
    STATE["compaction"] = dict(compaction_mod.SETTINGS, mode=a.compaction,
                               prepare=a.compaction_prepare, idle_s=a.compaction_idle_s,
                               min_tokens=a.compaction_min_tokens)
    STATE["prepared"] = compaction_mod.Prepared()
    STATE["last_end_mono"] = time.monotonic()
    if not a.fake_engine:
        # the summary written ahead and the memory writer (both need the resident prefix), and an
        # engine's idle graph captures
        threading.Thread(target=_prepare_loop, name="compaction-prepare", daemon=True).start()
    print(f"[compaction] {a.compaction}; prepare {a.compaction_prepare} (idle "
          f"{a.compaction_idle_s:g} s, from {a.compaction_min_tokens} tokens)", flush=True)
    if root is not None:
        STATE["probe_stage"] = probe_mod.ProbeStage(
            weights=STATE.get("weights_key") or "none",
            hidden=int(getattr(getattr(STATE.get("engine"), "cfg", None), "hidden_size", 0) or 0))
    # After the warm-up, so the request that pays for Triton autotuning is not in the histograms.
    metrics.install(sys.modules[__name__])
    live_registry().start()                      # samples only while something is in flight
    # the singletons' reference counts (server/refwatch.py): a C extension that drops their
    # references kills the process at 0 (bool_dealloc, 2026-10-05 13:12); this says so hours ahead
    STATE["refwatch"] = refwatch_mod.start(log=lambda s: print(s, flush=True))
    print(f"[server] listening on http://{a.host}:{a.port}  model {a.served_model}", flush=True)
    print(f"[server] queue max {a.max_queue} wait {a.queue_timeout:.0f}s "
          f"request timeout {a.request_timeout:.0f}s", flush=True)
    serve_until_drained(EngineServer((a.host, a.port), Handler))


class SetupHandler(Handler):
    """Setup mode's routes (`--setup`): the dashboard's files and session, /health, /v1/setup/*
    and an empty /v1/models; every other route answers 503 `setup_mode` with Retry-After."""

    def _setup_only(self, path: str) -> None:
        self._json(503, {"error": {"type": "setup_mode", "code": "setup_mode",
                                   "message": f"this box is in setup mode and serves no model yet "
                                              f"({path}); open /dashboard/#/setup to set it up"}},
                   extra_headers=(("Retry-After", "60"),))

    def do_GET(self):
        raw = self.path.split("?")[0]
        if raw == "/dashboard" or raw.startswith("/dashboard/"):
            return static.serve(self, self.path, STATE.get("dashboard_dir"))
        path = raw.rstrip("/") or "/"
        if path.startswith("/v1/setup/"):
            from server import setup_api
            return setup_api.handle(self, sys.modules[__name__], "GET", path, None)
        if path == "/v1/dashboard/session":
            return self._session("GET")
        if path in ("/health", "/healthz", "/v1/health"):
            w = STATE.get("wren")
            return self._json(503 if STATE.get("draining") else 200, {
                "status": "draining" if STATE.get("draining") else "ok", "mode": "setup",
                "version": STATE.get("version", ""), "status_port": _status_port(),
                "uptime_s": round(time.time() - STATE.get("started", time.time()), 1),
                "assistant": w.state if w is not None else "off"})
        if path == "/v1/models":
            return self._json(200, {"object": "list", "data": []})
        return self._setup_only(path)

    def do_POST(self):
        path = self.path.split("?")[0].rstrip("/")
        try:
            body = self._read()
        except Exception as exc:                                  # noqa: BLE001
            return self._json(400, {"error": {"message": f"bad json: {exc}",
                                              "type": "invalid_request_error"}})
        if path.startswith("/v1/setup/"):
            from server import setup_api
            return setup_api.handle(self, sys.modules[__name__], "POST", path, body)
        if path == "/v1/dashboard/session":
            return self._session("POST", body)
        return self._setup_only(path)

    def do_PUT(self):
        return self._setup_only(self.path.split("?")[0])

    do_PATCH = do_DELETE = do_PUT


class EngineServer(ThreadingHTTPServer):
    """The engine's HTTP server: a listen backlog of 128 (Python's default is 5, and clients that
    connect in the same moment past it get a reset on Linux: the night pass of 2026-10-04 saw 10
    of 64 on the proxy, which had the same default)."""

    request_queue_size = 128


def identity() -> dict:
    """What this process is: the version file, the code hash row3 records, the git commit.

    The served directory is an rsync copy with no `.git`, so the commit comes from `.git-sha`
    when the deploy wrote one.
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = {"version": "", "code_sha256": "", "code_sha": "", "git_sha": ""}
    try:
        out["version"] = open(os.path.join(root, "VERSION")).read().strip()
    except OSError:
        pass
    try:
        from pathlib import Path
        from tools.row3 import code_hash, git_head
        out["code_sha256"] = code_hash(Path(root))
        out["code_sha"] = out["code_sha256"][:16]
        out["git_sha"] = (git_head(Path(root)) or "")[:12]
    except Exception:                                              # noqa: BLE001
        pass
    if not out["git_sha"]:
        try:
            out["git_sha"] = open(os.path.join(root, ".git-sha")).read().strip()[:12]
        except OSError:
            pass
    # the weight profile ops/use-profile.sh started (ops/profiles/<name>.env): it joins the version,
    # so the dashboard's header, /metrics' info and every ledger row (engine_version) carry it
    prof = weights_profile()["name"]
    if prof and out["version"]:
        out["version"] = f"{out['version']}+{prof}"
    return out


def _identity_name() -> str | None:
    """0.8.2: the served model's name for the identity note (server/conductor/schema.py
    `model_name`): the label of the model card whose packs match the weight profile
    (models/<model>/card.json; "Qwen3.8-27B" for exl3-4.5 or nvfp4, the second model's label for its
    packs); None when no card matches (the note's default name)."""
    w = weights_profile()
    try:
        c = model_card.find(w.get("name") or w.get("format"))
    except Exception:                                              # noqa: BLE001
        return None
    return (c or {}).get("label") or None


def _status_port() -> int | None:
    """OPS-175: the status responder's port (QSE_STATUS_PORT, exported from serve.env by ops/start.sh),
    which answers for this server while it restarts; None when there is none."""
    v = os.environ.get("QSE_STATUS_PORT") or ""
    return int(v) if v.isdigit() else None


def weights_profile() -> dict:
    """The weight profile this process serves (QSE_WEIGHTS, exported from ops/profiles/<name>.env):
    its name, the target's weights, the drafters, the draft head and the sampler's nucleus; a loader
    that serves weights the profile's knobs do not name records its own (`weights_info`)."""
    e = os.environ
    exl3 = _S.get("EXL3") or None
    nv = e.get("NV") or _S.get("NVFP4") or None
    out = {"name": e.get("QSE_WEIGHTS") or None,
           "format": "exl3" if exl3 else ("nvfp4" if nv else None),
           "weights": exl3 or nv,
           "hf_repo": e.get("QSE_WEIGHTS_HF") or None,
           "drafters": _S.get("DRAFT_EXL3") or "nvfp4 (ft-b8-v2, ft-b16)",
           "draft_head": _S.get("DRAFT_HEAD_EXL3") or None,
           "sample_nucleus": _S.get("SAMPLE_NUCLEUS")}
    out.update(dict.get(STATE, "weights_info") or {})
    # VIS-95: a profile on a branch of its repo, and the largest window its registry entry names (a
    # pack that fits none: null); the 27B's profiles name neither, and their block stays as it was
    if e.get("QSE_WEIGHTS_REVISION"):
        out["hf_revision"] = e["QSE_WEIGHTS_REVISION"]
    mw = _profile_max_window(out["name"])
    if mw is not False:
        out["max_window"] = mw
    return out


_MAX_WINDOW: dict = {}


def _profile_max_window(name: str | None):
    """The registry's `max_window` of profile `name` (an int, or None for a pack that fits no
    window), or False when the profile names none or the registry does not read; read once."""
    if not name:
        return False
    if name not in _MAX_WINDOW:
        try:
            from server.setup import registry
            p = next((p for m in registry.load_all()[0].values() for p in m["profiles"] if p["id"] == name), None)
            _MAX_WINDOW[name] = p["max_window"] if p is not None and "max_window" in p else False
        except Exception:                                          # noqa: BLE001
            _MAX_WINDOW[name] = False
    return _MAX_WINDOW[name]


def serve_until_drained(httpd) -> None:
    """Serve until SIGTERM/SIGINT, then drain: refuse new work, finish what is running, exit."""

    def _drain(signum, _frame):
        """SIGTERM/SIGINT: stop taking work, let the generation in flight finish, then exit.

        A hard kill in the middle of a verify leaves the caller with a truncated stream and the
        watchdog with no way to tell a crash from a deploy. `draining` makes new requests a 503
        with `Retry-After`, `/health` says `draining` rather than `ok`, and `shutdown()` stops the
        listener -- it has to run off the serving thread or it deadlocks against the loop it is
        stopping. The wait for the work in flight is below, after `serve_forever` returns.
        """
        if STATE.get("draining"):
            return
        STATE["draining"] = True
        print(f"[server] signal {signum}: draining, {INFLIGHT['running']} running, "
              f"{INFLIGHT['waiting']} queued", flush=True)
        threading.Thread(target=httpd.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, _drain)
    signal.signal(signal.SIGINT, _drain)
    try:
        httpd.serve_forever()
    finally:
        # `shutdown()` stops the LISTENER; the handler threads are ThreadingHTTPServer's, and
        # those are daemons. Returning from here used to end the interpreter under the stream the
        # drain had promised to finish -- every graceful stop, every hold's stop, every deploy
        # restart cut the generation in flight. So wait for the work that was admitted:
        # the one running and any already queued for the lock. stop.sh's grace period is the cap
        # on how long a stop may take; this is not a second one.
        while True:
            with QUEUE:
                busy = INFLIGHT["running"] + INFLIGHT["waiting"] + INFLIGHT["lane"]
            if not busy:
                break
            time.sleep(0.2)
        # The usage ledger is the one thing that does persist: what is queued is flushed now,
        # after the last request's row was handed over -- and the conductor's journal with it.
        # The live conversations to the NVMe, read back by the next start
        spill_drain()
        shadow_drain()
        led = STATE.get("ledger")
        if led is not None:
            led.close()
        if STATE.get("disk_cache") is not None:
            STATE["disk_cache"].close()            # what is queued is written, the index saved
        if STATE.get("journal") is not None:
            STATE["journal"].close()
        # The engine holds one sequence and the drafters hold caches indexed by absolute position;
        # neither survives the process, so there is nothing else to persist. What is worth printing is
        # the tally, because a soak run reads it from the last line of the log.
        print(f"[server] stopped. {INFLIGHT}", flush=True)


if __name__ == "__main__":
    main()
