"""The two hooks the plugin registers, and what each reads (SGLang v0.5.21, tag e00930c).

`after_collect_metrics(result, tokenizer_manager, state, recv_obj, i)` runs after
`TokenizerManager.collect_metrics(state, recv_obj, i)`. On the request's last output batch
(`state.finished`) it reads, by index `i` into the batch's parallel lists (`BatchStrOutput`,
`managers/io_struct.py`):

    rids[i]                       the request id: the `rid` field of the OpenAI body when the
                                  client (or the proxy) sent one, else SGLang's own uuid
    finished_reasons[i]["type"]   stop, length, abort
    prompt_tokens, completion_tokens, cached_tokens, reasoning_tokens [i]
    cached_tokens_details[i]      device, host (and storage with an L3 tier)
    retraction_counts[i]          how often the scheduler retracted the request
    spec_verify_ct[i], spec_num_correct_drafts[i]   verify passes, accepted drafts
    time_stats[i]                 the scheduler's SchedulerReqTimeStats: queue_duration_s,
                                  forward_entry_time, prefill_finished_time

and from `state.time_stats` (APIServerReqTimeStats, perf_counter seconds): created_time,
first_token_time, finished_time. Every read is guarded: a field that is missing or of another
shape is null in the item.

`after_log_stats(result, collector, stats)` runs after `SchedulerMetricsCollector.log_stats(stats)`
and keeps at most one sample a second per scheduler (`SchedulerStats`,
`observability/metrics_collector.py`).
"""

from __future__ import annotations

import logging
import time

from tandem_sglang_plugin import PLUGIN, __version__, _push

log = logging.getLogger("sglang.tandem")

SAMPLE_EVERY_S = 1.0
MAX_FAILURES = 100

_state = {"off": False, "failures": 0, "last_log": -1e9, "last_step": {}}


def _engine_version() -> str:
    for mod in ("sglang.version", "sglang"):
        try:
            m = __import__(mod, fromlist=["__version__"])
            v = getattr(m, "__version__", None)
            if v:
                return str(v)
        except Exception:                                          # noqa: BLE001
            continue
    return ""


def pusher() -> "_push.Pusher":
    return _push.shared(plugin=PLUGIN, plugin_version=__version__, engine="sglang",
                        engine_version=_engine_version(), logger="sglang.tandem")


def _at(obj, name: str, i: int):
    seq = getattr(obj, name, None)
    if seq is None:
        return None
    try:
        return seq[i]
    except (IndexError, KeyError, TypeError):
        return None


def _field(obj, name: str):
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _span(a, b):
    a, b = _push.num(a), _push.num(b)
    if not a or not b or b < a:
        return None
    return _push.ms(b - a)


def request_item(tm, state, recv_obj, i: int) -> dict:
    fr = _at(recv_obj, "finished_reasons", i)
    reason = _field(fr, "type") if fr is not None else None
    details = _at(recv_obj, "cached_tokens_details", i)
    detail = None
    if details is not None:
        detail = {k: _push.num(_field(details, k)) for k in ("device", "host", "storage")
                  if _push.num(_field(details, k)) is not None} or None
    sched = None
    try:
        ts = getattr(recv_obj, "time_stats", None)
        sched = ts[i] if ts is not None else None
    except Exception:                                              # noqa: BLE001
        sched = None
    api = getattr(state, "time_stats", None)
    created = _field(api, "created_time")
    first = _field(api, "first_token_time")
    finished = _field(api, "finished_time")
    verify = _push.num(_at(recv_obj, "spec_verify_ct", i))
    accepted = _push.num(_at(recv_obj, "spec_num_correct_drafts", i))
    n_draft = None
    if verify:
        k = _push.num(getattr(getattr(tm, "server_args", None), "speculative_num_draft_tokens",
                              None))
        if k:
            n_draft = verify * (k - 1)
    queue_s = _push.num(_field(sched, "queue_duration_s"))
    return {"id": _at(recv_obj, "rids", i),
            "finish_reason": None if reason is None else str(reason),
            "prompt_tokens": _push.num(_at(recv_obj, "prompt_tokens", i)),
            "cached_tokens": _push.num(_at(recv_obj, "cached_tokens", i)),
            "cached_detail": detail,
            "completion_tokens": _push.num(_at(recv_obj, "completion_tokens", i)),
            "reasoning_tokens": _push.num(_at(recv_obj, "reasoning_tokens", i)),
            "preemptions": _push.num(_at(recv_obj, "retraction_counts", i)),
            "queue_ms": _push.ms(queue_s) if queue_s is not None else None,
            "prefill_ms": _span(_field(sched, "forward_entry_time"),
                                _field(sched, "prefill_finished_time")),
            "decode_ms": _span(first, finished),
            "ttft_ms": _span(created, first),
            "e2e_ms": _span(created, finished),
            "spec_verify_ct": verify if verify else None,
            "spec_accepted": accepted if verify else None,
            "spec_draft_tokens": n_draft,
            "t_end": time.time(), "engine_idx": _push.num(_at(recv_obj, "dp_ranks", i)) or 0}


def step_item(collector, stats) -> dict:
    def total(q):
        return _push.num(_field(q, "total")) if not isinstance(q, (int, float)) else q
    labels = getattr(collector, "labels", None) or {}
    g = getattr
    return {"t": time.time(), "engine_idx": labels.get("dp_rank") or 0,
            "tp_rank": labels.get("tp_rank"),
            "running": total(g(stats, "num_running_reqs", None)),
            "waiting": total(g(stats, "num_queue_reqs", None)),
            "kv_usage": _push.num(g(stats, "token_usage", None)),
            "kv_used_tokens": _push.num(g(stats, "kv_used_tokens", None)),
            "kv_available_tokens": _push.num(g(stats, "kv_available_tokens", None)),
            "cache_hit_rate": _push.num(g(stats, "cache_hit_rate", None)),
            "gen_tps": _push.num(g(stats, "gen_throughput", None)),
            "retracted": _push.num(g(stats, "num_retracted_reqs", None)),
            "spec": None if not _push.num(g(stats, "spec_accept_length", None)) else {
                "accept_length": _push.num(g(stats, "spec_accept_length", None)),
                "accept_rate": _push.num(g(stats, "spec_accept_rate", None))}}


def _failed(where: str, exc: Exception) -> None:
    _state["failures"] += 1
    if _state["failures"] >= MAX_FAILURES:
        _state["off"] = True
    now = time.monotonic()
    if now - _state["last_log"] >= 60.0 or _state["off"]:
        _state["last_log"] = now
        log.warning("tandem: %s failed (%s: %s)%s", where, type(exc).__name__, exc,
                    "; the plugin is off for this process" if _state["off"] else "")


def after_collect_metrics(result, tm, state, recv_obj, i, *args, **kwargs):
    if _state["off"]:
        return None
    try:
        if getattr(state, "finished", False):
            pusher().add_request(request_item(tm, state, recv_obj, i))
            _state["failures"] = 0
    except Exception as exc:                                       # noqa: BLE001
        _failed("collect_metrics", exc)
    return None


def after_log_stats(result, collector, stats, *args, **kwargs):
    if _state["off"]:
        return None
    try:
        now = time.monotonic()
        key = id(collector)
        if now - _state["last_step"].get(key, -1e9) >= SAMPLE_EVERY_S:
            _state["last_step"][key] = now
            pusher().add_step(step_item(collector, stats))
        _state["failures"] = 0
    except Exception as exc:                                       # noqa: BLE001
        _failed("log_stats", exc)
    return None
