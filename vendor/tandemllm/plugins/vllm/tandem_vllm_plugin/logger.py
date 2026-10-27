"""The stat logger vLLM loads from the `vllm.stat_logger_plugins` entry point.

vLLM v0.30.0 (`vllm/v1/metrics/loggers.py`) builds the logger once per data-parallel engine as
`TandemStatLogger(vllm_config, engine_index)` and calls

    record(scheduler_stats, iteration_stats, mm_cache_stats=None, engine_idx=0)

once per engine step in the API server process. What is read, every field with getattr so that a
renamed field becomes a null, never an exception:

    iteration_stats.finished_requests[]   FinishedRequestStats: request_id (the external id,
        `chatcmpl-<X-Request-Id>` on the chat route), finish_reason, num_prompt_tokens,
        num_cached_tokens, num_generation_tokens, num_preemptions, queued_time, prefill_time,
        decode_time, inference_time, e2e_latency, mean_time_per_output_token, max_tokens_param,
        is_corrupted
    iteration_stats.iteration_timestamp, num_preempted_reqs
    scheduler_stats   num_running_reqs, num_waiting_reqs, kv_cache_usage,
        prefix_cache_stats.{queries, hits}, spec_decoding_stats.{num_drafts, num_draft_tokens,
        num_accepted_tokens, num_accepted_tokens_per_pos}, kv_cache_eviction_events

One item per finished request; the scheduler side is folded into one sample a second (counters
summed, gauges as last seen), so the ring's 300 samples hold five minutes.

IT NEVER STOPS vLLM. If `StatLoggerBase` does not import, this module raises at import, which
vLLM's plugin loader catches and logs; vLLM starts without the logger. Anything that fails inside
`__init__` or `record` is caught, counted and logged at most once a minute; after 100 failures in
a row the logger turns itself off for the rest of the process.
"""

from __future__ import annotations

import logging
import time

from tandem_vllm_plugin import PLUGIN, __version__, _push

log = logging.getLogger("vllm.tandem")

try:
    from vllm.v1.metrics.loggers import StatLoggerBase
except Exception as exc:                                           # noqa: BLE001
    log.warning("tandem: the stat logger is off: vllm.v1.metrics.loggers.StatLoggerBase did not "
                "import (%s: %s); vLLM starts without it", type(exc).__name__, exc)
    raise

SAMPLE_EVERY_S = 1.0
MAX_FAILURES = 100


def _engine_version() -> str:
    try:
        import vllm
        return str(getattr(vllm, "__version__", "") or "")
    except Exception:                                              # noqa: BLE001
        return ""


def pusher() -> "_push.Pusher":
    return _push.shared(plugin=PLUGIN, plugin_version=__version__, engine="vllm",
                        engine_version=_engine_version(), logger="vllm.tandem")


def request_item(f, t_end: float, engine_idx: int) -> dict:
    """One FinishedRequestStats as the proxy reads it."""
    g = getattr
    reason = g(f, "finish_reason", None)
    return {"id": g(f, "request_id", None),
            "finish_reason": None if reason is None else str(reason),
            "prompt_tokens": _push.num(g(f, "num_prompt_tokens", None)),
            "cached_tokens": _push.num(g(f, "num_cached_tokens", None)),
            "completion_tokens": _push.num(g(f, "num_generation_tokens", None)),
            "preemptions": _push.num(g(f, "num_preemptions", None)),
            "queue_ms": _push.ms(g(f, "queued_time", None)),
            "prefill_ms": _push.ms(g(f, "prefill_time", None)),
            "decode_ms": _push.ms(g(f, "decode_time", None)),
            "inference_ms": _push.ms(g(f, "inference_time", None)),
            "e2e_ms": _push.ms(g(f, "e2e_latency", None)),
            "tpot_ms": _push.ms(g(f, "mean_time_per_output_token", None)),
            "max_tokens": _push.num(g(f, "max_tokens_param", None)),
            "corrupted": bool(g(f, "is_corrupted", False)),
            "t_end": t_end, "engine_idx": engine_idx}


class _Window:
    """The scheduler side of one engine, folded over a second."""

    __slots__ = ("t0", "steps", "running", "waiting", "kv_usage", "prefix_queries",
                 "prefix_hits", "preempted", "evictions", "drafts", "draft_tokens", "accepted",
                 "per_pos", "finished")

    def __init__(self, t0: float):
        self.t0 = t0
        self.steps = 0
        self.running = self.waiting = self.kv_usage = None
        self.prefix_queries = self.prefix_hits = 0
        self.preempted = self.evictions = self.finished = 0
        self.drafts = self.draft_tokens = self.accepted = None
        self.per_pos = None

    def fold(self, s, it) -> None:
        g = getattr
        self.steps += 1
        if s is not None:
            self.running = _push.num(g(s, "num_running_reqs", None))
            self.waiting = _push.num(g(s, "num_waiting_reqs", None))
            self.kv_usage = _push.num(g(s, "kv_cache_usage", None))
            pc = g(s, "prefix_cache_stats", None)
            if pc is not None:
                self.prefix_queries += _push.num(g(pc, "queries", 0)) or 0
                self.prefix_hits += _push.num(g(pc, "hits", 0)) or 0
            ev = g(s, "kv_cache_eviction_events", None)
            if ev:
                self.evictions += len(ev)
            sd = g(s, "spec_decoding_stats", None)
            if sd is not None:
                self.drafts = (self.drafts or 0) + (_push.num(g(sd, "num_drafts", 0)) or 0)
                self.draft_tokens = ((self.draft_tokens or 0)
                                     + (_push.num(g(sd, "num_draft_tokens", 0)) or 0))
                self.accepted = ((self.accepted or 0)
                                 + (_push.num(g(sd, "num_accepted_tokens", 0)) or 0))
                pos = g(sd, "num_accepted_tokens_per_pos", None) or []
                if pos:
                    if self.per_pos is None or len(self.per_pos) < len(pos):
                        self.per_pos = (self.per_pos or []) + [0] * (len(pos) - len(
                            self.per_pos or []))
                    for i, v in enumerate(pos):
                        self.per_pos[i] += _push.num(v) or 0
        if it is not None:
            self.preempted += _push.num(g(it, "num_preempted_reqs", 0)) or 0
            self.finished += len(g(it, "finished_requests", None) or ())

    def item(self, t: float, engine_idx: int) -> dict:
        return {"t": t, "span_s": round(t - self.t0, 3), "engine_idx": engine_idx,
                "steps": self.steps, "running": self.running, "waiting": self.waiting,
                "kv_usage": self.kv_usage, "prefix_queries": self.prefix_queries,
                "prefix_hits": self.prefix_hits, "preempted": self.preempted,
                "evictions": self.evictions, "finished": self.finished,
                "spec": None if self.drafts is None else {
                    "drafts": self.drafts, "draft_tokens": self.draft_tokens,
                    "accepted": self.accepted, "accepted_per_pos": self.per_pos or []}}


class TandemStatLogger(StatLoggerBase):
    """Pushes each finished request and a per-second scheduler sample to TandemLLM's proxy."""

    def __init__(self, vllm_config=None, engine_index: int = 0):
        self.engine_index = engine_index
        self._off = False
        self._failures = 0
        self._last_log = -1e9
        self._windows: dict = {}
        try:
            self._push = pusher()
        except Exception as exc:                                   # noqa: BLE001
            self._off = True
            log.warning("tandem: the stat logger is off: %s: %s", type(exc).__name__, exc)

    def record(self, scheduler_stats, iteration_stats, mm_cache_stats=None, engine_idx=0):
        if self._off:
            return
        try:
            now = getattr(iteration_stats, "iteration_timestamp", None) or time.time()
            for f in getattr(iteration_stats, "finished_requests", None) or ():
                self._push.add_request(request_item(f, now, engine_idx))
            w = self._windows.get(engine_idx)
            if w is None:
                w = self._windows[engine_idx] = _Window(now)
            w.fold(scheduler_stats, iteration_stats)
            if now - w.t0 >= SAMPLE_EVERY_S:
                self._push.add_step(w.item(now, engine_idx))
                self._windows[engine_idx] = _Window(now)
            self._failures = 0
        except Exception as exc:                                   # noqa: BLE001
            self._failures += 1
            if self._failures >= MAX_FAILURES:
                self._off = True
            mono = time.monotonic()
            if mono - self._last_log >= 60.0 or self._off:
                self._last_log = mono
                log.warning("tandem: record() failed (%s: %s)%s", type(exc).__name__, exc,
                            "; the logger is off for this process" if self._off else "")

    def log_engine_initialized(self):
        if self._off:
            return
        p = self._push
        where = p.url.split("/v1/collect/")[0] if p.url else None
        log.info("tandem: %s %s on vLLM %s, %s", PLUGIN, __version__, p.engine_version or "?",
                 f"posting to {where}" if where else
                 "no TANDEM_COLLECT_URL: read through GET /tandem/stats only")

    def log(self):
        pass
