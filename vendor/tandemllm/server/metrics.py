"""Prometheus metrics for this server, with no dependency and no edit to the decode loop.

`prometheus_client` is not installed on the board and the interpreter there has no pip, so the
exposition format is written out by hand. That is about eighty lines and it is the whole of the
format Prometheus 0.0.4 needs: a `# HELP` line, a `# TYPE` line, and one sample line per label set,
with histograms carrying cumulative `_bucket{le=...}`, a `_sum` and a `_count`.

WHERE THE NUMBERS COME FROM, because it is the part that decides whether they are true.

Nothing here is measured twice. Every counter is taken at a point the server already had:

  * `_log_request` runs once per generation, whatever happened to it, and already knows the prompt
    and completion counts, the finish reason and the wall time. It is the terminal counter.
  * `generate_stream` yields token ids as they are decided, so the gap between the call and the
    first yield is the time to first token and the gaps after it are the inter-token times.
  * the drafter is asked for a block (`propose` / `propose_tree`), told what the target committed
    (`observe`) and told what the verify cost (`on_verify`). Those three are the speculation
    counters, and the loop was calling all three already.

So this module wraps four functions rather than editing them, and `server/app.py` carries one
import and one `install()` call. That is deliberate: phase 9 is editing `app.py` for other reasons
and a hook scattered through its generation loop would collide with every one of those edits. The
cost of the wrappers is two `perf_counter()` calls and a few dictionary increments per block, on a
block that is 133 ms.

WHAT IS NOT HERE. The engine serves one sequence, so there is no batch, no KV utilisation and no
preemption to report; those metrics exist in vLLM because it has a scheduler and this server does
not. docs/roadmap.md says what would have to be true first. `qse_requests_waiting` is
the queue this server does have -- callers waiting for the one engine lock.

Names follow vLLM's set where the quantity is the same one, with a `qse_` prefix. Units are in the
name and they are seconds and bytes, never milliseconds, because that is what the format expects.

    curl -s localhost:8000/metrics
    server/METRICS.md          names, labels, units, an example scrape, Grafana panels
"""

from __future__ import annotations

import threading
import time

# Bumped when a metric name, label or unit changes, and reported as a label of `qse_engine_info`
# so a dashboard can tell which contract it is reading.
VERSION = "0.2.0"

# --------------------------------------------------------------------------- the tiny registry

_LOCK = threading.RLock()


def _escape_help(text: str) -> str:
    return text.replace("\\", "\\\\").replace("\n", "\\n")


def _escape_label(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _fmt(value: float) -> str:
    """Prometheus wants a Go float. An integral value prints without a decimal point.

    `inf` and `nan` are legal in the format and are spelled `+Inf` and `NaN`; a Python `repr`
    spells them `inf` and `nan`, which no scraper accepts.
    """
    v = float(value)
    if v != v:
        return "NaN"
    if v == float("inf"):
        return "+Inf"
    if v == float("-inf"):
        return "-Inf"
    if v == int(v) and abs(v) < 1e15:
        return str(int(v))
    return repr(v)


class _Metric:
    """One metric family. `collect` lets a family be read at scrape time instead of accumulated."""

    kind = "untyped"

    def __init__(self, name: str, help_text: str, labelnames: tuple[str, ...] = (),
                 collect=None):
        self.name = name
        self.help = help_text
        self.labelnames = tuple(labelnames)
        self.collect = collect
        self.values: dict[tuple[str, ...], float] = {}

    # --- label handling ---------------------------------------------------------------------
    def _key(self, labels: dict) -> tuple[str, ...]:
        if set(labels) != set(self.labelnames):
            raise ValueError(f"{self.name} takes labels {self.labelnames}, got {tuple(labels)}")
        return tuple(str(labels[n]) for n in self.labelnames)

    def _label_str(self, key: tuple[str, ...], extra: tuple[tuple[str, str], ...] = ()) -> str:
        pairs = list(zip(self.labelnames, key)) + list(extra)
        if not pairs:
            return ""
        inner = ",".join(f'{n}="{_escape_label(v)}"' for n, v in pairs)
        return "{" + inner + "}"

    # --- reading ----------------------------------------------------------------------------
    def snapshot(self) -> dict[tuple[str, ...], float]:
        if self.collect is None:
            with _LOCK:
                return dict(self.values)
        try:
            got = self.collect()
        except Exception:                                          # noqa: BLE001
            # A gauge that reads the allocator, a cache report or /proc must never be able to
            # break a scrape. A monitoring endpoint that returns 500 when one number is missing
            # is worse than one that is missing one number.
            return {}
        if isinstance(got, dict):
            return {(k if isinstance(k, tuple) else (str(k),)): float(v) for k, v in got.items()}
        return {(): float(got)}

    def render(self) -> list[str]:
        rows = self.snapshot()
        if not rows:
            return []
        out = [f"# HELP {self.name} {_escape_help(self.help)}", f"# TYPE {self.name} {self.kind}"]
        for key in sorted(rows):
            out.append(f"{self.name}{self._label_str(key)} {_fmt(rows[key])}")
        return out


class Counter(_Metric):
    kind = "counter"

    def inc(self, value: float = 1.0, **labels) -> None:
        key = self._key(labels)
        with _LOCK:
            self.values[key] = self.values.get(key, 0.0) + float(value)


class Gauge(_Metric):
    kind = "gauge"

    def set(self, value: float, **labels) -> None:
        key = self._key(labels)
        with _LOCK:
            self.values[key] = float(value)


class Histogram(_Metric):
    """Cumulative buckets, a sum and a count, which is what a histogram is in this format.

    The buckets are upper bounds and they are chosen per metric from what this engine actually
    does: a block is about 133 ms, a verify about 99, a prompt of 256 tokens takes about 760 ms to
    first token. Buckets copied from another server would put every observation in the same bin.
    """

    kind = "histogram"

    def __init__(self, name: str, help_text: str, buckets, labelnames: tuple[str, ...] = ()):
        super().__init__(name, help_text, labelnames)
        b = [float(x) for x in buckets]
        if b != sorted(b) or len(set(b)) != len(b):
            raise ValueError(f"{name}: buckets must be sorted and unique, got {buckets}")
        if b and b[-1] == float("inf"):
            b = b[:-1]
        self.buckets = b
        self.counts: dict[tuple[str, ...], list[float]] = {}
        self.sums: dict[tuple[str, ...], float] = {}
        self.totals: dict[tuple[str, ...], float] = {}

    def observe(self, value: float, **labels) -> None:
        key = self._key(labels)
        v = float(value)
        with _LOCK:
            counts = self.counts.get(key)
            if counts is None:
                counts = [0.0] * (len(self.buckets) + 1)
                self.counts[key] = counts
                self.sums[key] = 0.0
                self.totals[key] = 0.0
            for i, ub in enumerate(self.buckets):
                if v <= ub:
                    counts[i] += 1.0
                    break
            else:
                counts[len(self.buckets)] += 1.0
            self.sums[key] += v
            self.totals[key] += 1.0

    def render(self) -> list[str]:
        with _LOCK:
            keys = sorted(self.counts)
            if not keys:
                return []
            out = [f"# HELP {self.name} {_escape_help(self.help)}",
                   f"# TYPE {self.name} {self.kind}"]
            for key in keys:
                counts = self.counts[key]
                running = 0.0
                for i, ub in enumerate(self.buckets):
                    running += counts[i]
                    le = (("le", _fmt(ub)),)
                    out.append(f"{self.name}_bucket{self._label_str(key, le)} {_fmt(running)}")
                running += counts[len(self.buckets)]
                out.append(f'{self.name}_bucket{self._label_str(key, (("le", "+Inf"),))} '
                           f"{_fmt(running)}")
                out.append(f"{self.name}_sum{self._label_str(key)} {_fmt(self.sums[key])}")
                out.append(f"{self.name}_count{self._label_str(key)} {_fmt(self.totals[key])}")
            return out


class Registry:
    def __init__(self):
        self._metrics: list[_Metric] = []
        self._names: set[str] = set()

    def add(self, metric: _Metric) -> _Metric:
        if metric.name in self._names:
            raise ValueError(f"{metric.name} is registered twice")
        self._names.add(metric.name)
        self._metrics.append(metric)
        return metric

    def render(self) -> str:
        lines: list[str] = []
        for m in self._metrics:
            lines.extend(m.render())
        return "\n".join(lines) + ("\n" if lines else "")

    def reset(self) -> None:
        """Tests only. A served registry never forgets: a counter that resets is not a counter."""
        with _LOCK:
            for m in self._metrics:
                m.values.clear()
                if isinstance(m, Histogram):
                    m.counts.clear()
                    m.sums.clear()
                    m.totals.clear()


REGISTRY = Registry()

# --------------------------------------------------------------------------- the metrics

# Time to first token on this engine is a prefill plus one verify: 761 ms at the median for a
# 256-token prompt, and 526 ms warm for a 1,724-token one out of the prefix cache. The fine end
# is for a response-cache hit, measured at 2.0 ms.
TTFT_BUCKETS = (0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 30.0, 60.0)
# One output token is a whole block divided by what the block committed: 133 ms over 2.6 tokens on
# fresh prose, over 14.9 on a quotation. So the interesting range is 5 ms to 100 ms.
TPOT_BUCKETS = (0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2, 0.5, 1.0)
E2E_BUCKETS = (0.01, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 60.0, 120.0, 300.0, 600.0, 900.0)
# A verify is 98.6 ms at width 16 and a draft 26 ms, both on the shipped configuration.
VERIFY_BUCKETS = (0.02, 0.05, 0.08, 0.09, 0.095, 0.1, 0.11, 0.12, 0.15, 0.2, 0.3, 0.5)
DRAFT_BUCKETS = (0.001, 0.005, 0.01, 0.02, 0.025, 0.03, 0.04, 0.06, 0.1, 0.2, 0.5)
# A block yields at least one token and at most the node budget plus one, which is 17 today.
ACCEPT_BUCKETS = tuple(float(i) for i in range(1, 18))

requests_total = REGISTRY.add(Counter(
    "qse_requests_total",
    "generations that finished, by the reason they finished", ("finish_reason",)))
request_success_total = REGISTRY.add(Counter(
    "qse_request_success_total",
    "generations that finished on the model's own stop token or the caller's token budget"))
prompt_tokens_total = REGISTRY.add(Counter(
    "qse_prompt_tokens_total", "prompt tokens accepted, including tokens served from a cache"))
generation_tokens_total = REGISTRY.add(Counter(
    "qse_generation_tokens_total", "tokens written by the engine"))
errors_total = REGISTRY.add(Counter(
    "qse_errors_total", "exceptions that ended a generation, by exception class", ("type",)))
images_left_out_total = REGISTRY.add(Counter(
    "qse_images_left_out_total",
    "a conversation's earlier images left out of a request over the image caps (server/images.py)"))

time_to_first_token_seconds = REGISTRY.add(Histogram(
    "qse_time_to_first_token_seconds",
    "arrival of the request to its first token, queue wait included", TTFT_BUCKETS))
time_per_output_token_seconds = REGISTRY.add(Histogram(
    "qse_time_per_output_token_seconds",
    "gap between two tokens of one generation. A verified block writes several tokens at once, so "
    "half of these are microseconds and half are a whole block", TPOT_BUCKETS))
e2e_request_latency_seconds = REGISTRY.add(Histogram(
    "qse_e2e_request_latency_seconds",
    "arrival of the request to the end of its stream, queue wait included", E2E_BUCKETS))

spec_decode_num_accepted_tokens_total = REGISTRY.add(Counter(
    "qse_spec_decode_num_accepted_tokens_total",
    "drafted tokens the target agreed with. The token the target writes itself is not counted "
    "here, so accepted over drafted is the draft acceptance rate"))
spec_decode_num_draft_tokens_total = REGISTRY.add(Counter(
    "qse_spec_decode_num_draft_tokens_total", "tokens proposed by the drafter, anchor excluded"))
spec_decode_num_drafts_total = REGISTRY.add(Counter(
    "qse_spec_decode_num_drafts_total", "blocks the drafter proposed"))
spec_accept_per_block = REGISTRY.add(Histogram(
    "qse_spec_accept_per_block",
    "tokens committed by one verified block, the target's own token included", ACCEPT_BUCKETS))
draft_width_chosen_total = REGISTRY.add(Counter(
    "qse_draft_width_chosen_total",
    "blocks proposed at each width, anchor included. The length router chooses 8 or 16",
    ("width",)))
verify_seconds = REGISTRY.add(Histogram(
    "qse_verify_seconds", "one verify pass over a draft block, as the decode loop paid for it",
    VERIFY_BUCKETS))
draft_seconds = REGISTRY.add(Histogram(
    "qse_draft_seconds", "one call to the drafter, lookup and block drafter together",
    DRAFT_BUCKETS))

# --------------------------------------------------------------------------- the conductor
#
# SRV-115: every decision of the conductor, passes included, counted here; the journal
# (server/conductor/decisions.py) keeps the rows. With the conductor raw nothing is counted and
# the families stay at zero. The router's tier 1 is the classifier stages (the decider, Qwen):
# 26.5 ms p50 for the decider and 553 ms for Qwen on the Spark (TandemLLM 7).
TIER1_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 2.0, 5.0, 10.0)

conductor_decisions_total = REGISTRY.add(Counter(
    "qse_conductor_decisions_total",
    "conductor decisions, passes included: the monitor's tasks (D1, E1, C3, ...) and the "
    "router's class (task route)", ("task", "tier", "mode", "decision")))
conductor_actions_total = REGISTRY.add(Counter(
    "qse_conductor_actions_total",
    "decisions the conductor executed, by task and action", ("task", "action")))
conductor_hidden_tokens_total = REGISTRY.add(Counter(
    "qse_conductor_hidden_tokens_total",
    "tokens the model saw that the client did not send or receive (hidden closes, hidden turns, "
    "reminders, router notes new and replayed)"))
conductor_gave_up_total = REGISTRY.add(Counter(
    "qse_conductor_gave_up_total", "turns the conductor ended with its GIVE_UP sentence"))
conductor_tier1_seconds = REGISTRY.add(Histogram(
    "qse_conductor_tier1_seconds",
    "one tier 1 question (a classifier stage: the decider or Qwen), answered or not",
    TIER1_BUCKETS))
conductor_tier1_failures_total = REGISTRY.add(Counter(
    "qse_conductor_tier1_failures_total",
    "tier 1 questions without a usable answer, by reason", ("reason",)))
conductor_package_kept_out_total = REGISTRY.add(Counter(
    "qse_conductor_package_kept_out_total",
    "packages the router kept out of a turn because an offered tool delivers their result "
    "(SRV-615), by why and by mode (act, shadow)", ("why", "mode")))
conductor_reasoning_tokens_cut_total = REGISTRY.add(Counter(
    "qse_conductor_reasoning_tokens_cut_total",
    "reasoning tokens written before a hidden close cut the block (the size of the blocks the "
    "conductor closed)"))

# --------------------------------------------------------------------------- scrape-time gauges
#
# These read the server's own state when Prometheus asks, rather than being accumulated. The
# alternative is a copy of every number kept in step by hand, which is a second source of truth
# and the first thing to go stale.

_SOURCES: dict = {"state": None, "inflight": None, "cache_stats": None, "info": {},
                  "started": time.time()}


def _inflight(field: str):
    def read():
        d = _SOURCES.get("inflight")
        return float(d[field]) if d else 0.0
    return read


def _cache_field(section: str, field: str, scale: float = 1.0):
    """One number out of `app.cache_stats()`, or nothing at all if that cache is off."""
    def read():
        fn = _SOURCES.get("cache_stats")
        if fn is None:
            return {}
        rep = (fn() or {}).get(section)
        if not isinstance(rep, dict) or field not in rep:
            return {}
        return float(rep[field]) * scale
    return read


def _refused():
    """Requests turned away before a generation existed.

    The server keeps one field for both queue refusals -- the 503 when the queue is full and the
    429 when a request waited longer than `--queue-timeout` -- so they are one reason and it says
    so rather than inventing a split it cannot see. `memory` is the 503 a prefill gets when the
    engine would go over its declared peak with every cache given back (OPS-47).
    """
    d = _SOURCES.get("inflight")
    out = {("queue",): float(d.get("refused", 0))} if d else {}
    fn = _SOURCES.get("cache_stats")
    mem = (fn() or {}).get("memory") if fn is not None else None
    if isinstance(mem, dict):
        out[("memory",)] = float(mem.get("refused") or 0)        # OPS-47: the 503 before a wedge
    return out


def _cache_family(field: str, scale: float = 1.0):
    """The same field across every cache that reports it, labelled by cache."""
    def read():
        fn = _SOURCES.get("cache_stats")
        if fn is None:
            return {}
        stats = fn() or {}
        out: dict[tuple[str, ...], float] = {}
        for label, section in (("state", "state_store"), ("response", "response_cache"),
                               ("resident", "resident")):
            rep = stats.get(section)
            if isinstance(rep, dict) and field in rep:
                out[(label,)] = float(rep[field]) * scale
        return out
    return read


def _suffix_bytes():
    fn = _SOURCES.get("cache_stats")
    if fn is None:
        return {}
    rep = (fn() or {}).get("suffix_store")
    if not isinstance(rep, dict) or "tokens" not in rep:
        return {}
    return {("suffix",): float(rep["tokens"]) * 4.0}        # int32 token ids on disk


def _cache_bytes():
    out = _cache_family("bytes")()
    out.update(_suffix_bytes())
    return out


def _gpu_memory():
    try:
        import torch
        return float(torch.cuda.memory_allocated())
    except Exception:                                              # noqa: BLE001
        return {}


def _gpu_reserved():
    try:
        import torch
        return float(torch.cuda.memory_reserved())
    except Exception:                                              # noqa: BLE001
        return {}


def _unified_free():
    """Free memory on the board.

    GB10 shares one pool between the GPU and the host, so `nvidia-smi` reports N/A for used
    memory and the only numbers that mean anything are the driver's own free/total and the
    kernel's MemAvailable. The driver is asked first; /proc/meminfo is the fallback for a process
    that has no CUDA context yet.
    """
    try:
        import torch
        return float(torch.cuda.mem_get_info()[0])
    except Exception:                                              # noqa: BLE001
        pass
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return float(line.split()[1]) * 1024.0
    except Exception:                                              # noqa: BLE001
        pass
    return {}


REGISTRY.add(Gauge("qse_requests_running", "generations the engine is working on",
                   collect=_inflight("running")))
REGISTRY.add(Gauge("qse_requests_waiting", "requests holding a socket and waiting for the engine",
                   collect=_inflight("waiting")))
REGISTRY.add(Counter("qse_requests_refused_total",
                     "requests turned away with a Retry-After before the engine saw them",
                     ("reason",), collect=_refused))
def _cache_hits():
    """Hits by cache, and for the state store by where the snapshot came from: the end of
    a turn (`kind="session"`) or a prefill chunk boundary (`kind="prefix"`). A store that does not
    report the split keeps the one `cache="state"` row it always had."""
    out = _cache_family("hits")()
    fn = _SOURCES.get("cache_stats")
    rep = ((fn() or {}).get("state_store") if fn is not None else None)
    if isinstance(rep, dict) and "hits_session" in rep:
        out.pop(("state",), None)
        for kind in ("session", "prefix"):
            out[("state", kind)] = float(rep.get(f"hits_{kind}", 0))
    return out


class _MixedCounter(Counter):
    """A counter whose rows may carry the optional `kind` label (qse_cache_hits_total)."""

    def _label_str(self, key, extra=()):
        names = ("cache", "kind")[:len(key)]
        pairs = list(zip(names, key)) + list(extra)
        return "{" + ",".join(f'{n}="{_escape_label(v)}"' for n, v in pairs) + "}"


REGISTRY.add(_MixedCounter("qse_cache_hits_total",
                           "prefix lookups answered from a cache; for the state store, by where the "
                           "snapshot came from: kind=session (a turn's end) or prefix (a chunk "
                           "boundary)", ("cache",), collect=_cache_hits))
REGISTRY.add(Counter("qse_cache_misses_total", "prefix lookups that found nothing",
                     ("cache",), collect=_cache_family("misses")))
REGISTRY.add(Counter("qse_cache_evictions_total", "entries dropped to stay inside the byte budget",
                     ("cache",), collect=_cache_family("evictions")))
REGISTRY.add(Gauge("qse_cache_bytes", "bytes held by each cache",
                   ("cache",), collect=_cache_bytes))


def _memory_report() -> tuple | None:
    fn = _SOURCES.get("cache_stats")
    if fn is None:
        return None
    stats = fn() or {}
    return stats.get("memory"), stats.get("resident"), stats.get("state_store")


def _memory_parts():
    """OPS-47: the engine against its limits, by part: what the allocator holds and keeps, the
    caches the policy caps, and the two limits themselves as `part="ceiling"` and `part="peak"`."""
    got = _memory_report()
    if not got or not isinstance(got[0], dict):
        return {}
    mem, res, store = got
    out = {("allocated",): float(mem.get("allocated") or 0),
           ("reserve",): float(mem.get("reserve") or 0),
           ("ceiling",): float(mem.get("ceiling") or 0), ("peak",): float(mem.get("peak") or 0),
           ("prefill_peak",): float(mem.get("prefill_peak") or 0)}
    if isinstance(res, dict):
        for part, key in (("resident_anchors", "anchor_pool_bytes"), ("resident_stash", "stash_bytes"),
                          ("resident_pin", "pin_bytes"), ("resident_park", "park_bytes")):
            out[(part,)] = float(res.get(key) or 0)
    if isinstance(store, dict):
        out[("state_store",)] = float(store.get("bytes") or 0)
    return out


def _memory_shed():
    got = _memory_report()
    if not got or not isinstance(got[0], dict):
        return {}
    return {(lv,): float(n) for lv, n in (got[0].get("shed") or {}).items()}


REGISTRY.add(Gauge("qse_memory_bytes", "the engine's memory against its limits (OPS-47), by part",
                   ("part",), collect=_memory_parts))
REGISTRY.add(Counter("qse_memory_shed_total",
                     "caches given back to keep the engine under its ceiling or its peak, by level",
                     ("level",), collect=_memory_shed))
REGISTRY.add(Gauge("qse_cache_entries", "entries held by each cache",
                   ("cache",), collect=_cache_family("entries")))


def _disk():
    """The disk tier's report (C1) from `/v1/cache/stats`, or None with the tier off."""
    fn = _SOURCES.get("cache_stats")
    rep = (fn() or {}).get("disk_cache") if fn is not None else None
    return rep if isinstance(rep, dict) else None


def _disk_field(field: str):
    def read():
        rep = _disk()
        return {(): float(rep.get(field) or 0)} if rep is not None else {}
    return read


def _disk_by(field: str):
    def read():
        rep = _disk()
        if rep is None:
            return {}
        return {(k,): float(v) for k, v in (rep.get(field) or {}).items()}
    return read


def _disk_bytes():
    rep = _disk()
    if rep is None:
        return {}
    return {(p,): float(rep.get(k) or 0) for p, k in (
        ("entries", "bytes"), ("budget", "budget"), ("free", "free_bytes"), ("floor", "floor"),
        ("staging", "staging_bytes"))}


def _disk_counts():
    rep = _disk()
    if rep is None:
        return {}
    return {("anchors",): float(rep.get("entries") or 0),
            ("chunks",): float(rep.get("chunks") or 0)}


def _disk_writes():
    rep = _disk()
    if rep is None:
        return {}
    return {("chunk",): float(rep.get("writes_chunk") or 0),
            ("anchor",): float(rep.get("writes_anchor") or 0)}


REGISTRY.add(Gauge("qse_disk_cache_bytes", "the disk tier (C1): what its entries hold, its budget, "
                   "the file system's free space and the floor it never crosses, the host staging "
                   "buffer", ("part",), collect=_disk_bytes))
REGISTRY.add(Gauge("qse_disk_cache_entries", "the disk tier's files: anchors (one state each) and "
                   "chunks (1,024 tokens of rows each)", ("kind",), collect=_disk_counts))
REGISTRY.add(Counter("qse_disk_cache_lookups_total", "prefills the disk tier could have shortened "
                     "by a chunk or more", collect=_disk_field("lookups")))
REGISTRY.add(Counter("qse_disk_cache_hits_total", "prefills that started from the disk tier",
                     collect=_disk_field("hits")))
REGISTRY.add(Counter("qse_disk_cache_misses_total", "lookups the disk tier did not answer",
                     collect=_disk_field("misses")))
REGISTRY.add(Counter("qse_disk_cache_tokens_restored_total", "prompt tokens read back from the "
                     "disk instead of prefilled", collect=_disk_field("tokens_restored")))
REGISTRY.add(Counter("qse_disk_cache_read_bytes_total", "bytes a restore read",
                     collect=_disk_field("bytes_read")))
REGISTRY.add(Counter("qse_disk_cache_written_bytes_total", "bytes the writer wrote",
                     collect=_disk_field("bytes_written")))
REGISTRY.add(Counter("qse_disk_cache_writes_total", "files the writer wrote, by kind",
                     ("kind",), collect=_disk_writes))
REGISTRY.add(Counter("qse_disk_cache_evictions_total", "entries the disk tier gave up, by reason "
                     "(budget, floor, ttl, orphan)", ("reason",), collect=_disk_by("evicted")))
REGISTRY.add(Counter("qse_disk_cache_dropped_total", "writes not done, by reason (window: the next "
                     "prefill came first; budget, floor, staging, aborted, too_big)",
                     ("reason",), collect=_disk_by("dropped")))
REGISTRY.add(Counter("qse_disk_cache_failed_restores_total", "restores that failed a check "
                     "(checksum, layout) and took the cold path",
                     collect=_disk_field("failed_restores")))
REGISTRY.add(Gauge("qse_disk_cache_last_restore_seconds", "the last restore's wall time",
                   collect=lambda: ({(): float(_disk()["last_restore_ms"]) / 1e3}
                                    if _disk() and _disk().get("last_restore_ms") else {})))
REGISTRY.add(Counter("qse_cache_collisions_total",
                     "stored prefixes whose 64-bit hash matched and whose tokens did not",
                     ("cache",), collect=_cache_family("rejected_collisions")))
REGISTRY.add(Counter("qse_prefill_tokens_reused_total",
                     "prompt tokens restored from a snapshot instead of forwarded",
                     collect=_cache_field("state_store", "tokens_reused")))
REGISTRY.add(Counter("qse_prefill_tokens_forwarded_total",
                     "prompt tokens run through the 64 layers",
                     collect=_cache_field("state_store", "tokens_forwarded")))
REGISTRY.add(Gauge("qse_gpu_memory_used_bytes", "allocator bytes held by the engine",
                   collect=_gpu_memory))
REGISTRY.add(Gauge("qse_gpu_memory_reserved_bytes",
                   "bytes the allocator has taken from the driver and not given back",
                   collect=_gpu_reserved))
REGISTRY.add(Gauge("qse_unified_memory_free_bytes", "free bytes in the board's one memory pool",
                   collect=_unified_free))
REGISTRY.add(Gauge("qse_engine_uptime_seconds", "seconds since the server started",
                   collect=lambda: time.time() - float(_SOURCES["started"])))
REGISTRY.add(Gauge("qse_engine_info",
                   "always 1. The labels are the configuration the numbers above were produced by",
                   ("version", "model", "drafter", "width", "tree", "nvfp4", "fp8_head",
                    "max_len", "caches"),
                   collect=lambda: ({tuple(_SOURCES["info"].values()): 1.0}
                                    if _SOURCES["info"] else {})))

INFO_LABELS = ("version", "model", "drafter", "width", "tree", "nvfp4", "fp8_head", "max_len",
               "caches")

# --------------------------------------------------------------------------- contract 0.2.0
#
# Every per-request number below comes from `server/usage.py::RequestRecord` at the end of the
# request (`on_record`), not from a hook in the decode loop. Buckets are this engine's numbers: a
# queue that times out at 240 s, a prefill of ~0.4 s at 256 tokens and minutes at 256k, decode
# rates of 30-70 tok/s on new text and 100+ on a quotation.

QUEUE_BUCKETS = (0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 240.0)
PREFILL_BUCKETS = (0.005, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0)
DECODE_TPS_BUCKETS = (5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 100, 130, 160, 200)
PREFILL_TPS_BUCKETS = (100, 250, 500, 1000, 1500, 2000, 2500, 3000, 4000, 6000, 10000)
PROMPT_BUCKETS = (16, 64, 256, 1024, 4096, 16384, 32768, 65536, 131072, 262144)
COMPLETION_BUCKETS = (1, 16, 64, 256, 1024, 4096, 16384, 32768)

# the fixed set of routes for qse_http_requests_total: never the raw path (its cardinality is the
# client's to choose)
ROUTES = (("/v1/chat/completions", "chat"), ("/v1/completions", "completions"),
          ("/v1/models", "models"), ("/metrics", "metrics"), ("/v1/cache/", "cache"),
          ("/v1/dashboard/", "dashboard"), ("/dashboard", "static"))


def route_of(path: str) -> str:
    p = (path or "").split("?")[0]
    if p.rstrip("/") in ("/health", "/healthz", "/v1/health"):
        return "health"
    for prefix, name in ROUTES:
        if p == prefix or p.startswith(prefix if prefix.endswith("/") else prefix + "/") \
                or p.rstrip("/") == prefix.rstrip("/"):
            return name
    return "other"


http_requests_total = REGISTRY.add(Counter(
    "qse_http_requests_total", "HTTP responses by route (a fixed set) and status code",
    ("route", "code")))
prompt_tokens_cached_total = REGISTRY.add(Counter(
    "qse_prompt_tokens_cached_total",
    "prompt tokens restored from the state store (all of them on a response-cache replay)"))
reasoning_tokens_total = REGISTRY.add(Counter(
    "qse_reasoning_tokens_total", "committed tokens inside the reasoning block"))
suffix_store_drafts_total = REGISTRY.add(Counter(
    "qse_suffix_store_drafts_total",
    "proposed blocks for which the persistent suffix store matched the context (hit) or did not "
    "(miss); counted once a block, never once a lookup", ("outcome",)))
request_queue_seconds = REGISTRY.add(Histogram(
    "qse_request_queue_seconds", "arrival to the engine lock, per request", QUEUE_BUCKETS))
request_prefill_seconds = REGISTRY.add(Histogram(
    "qse_request_prefill_seconds", "the engine lock to the first token (template, tokenise, "
    "prefill), per request", PREFILL_BUCKETS))
request_decode_tps = REGISTRY.add(Histogram(
    "qse_request_decode_tokens_per_second",
    "per decoded request: (completion tokens - 1) / (last token - first token) -- the atlas "
    "row's convention, and the `predicted_per_second` a response carries", DECODE_TPS_BUCKETS))
request_prefill_tps = REGISTRY.add(Histogram(
    "qse_request_prefill_tokens_per_second",
    "per request with a forwarded prompt: forwarded tokens / prefill seconds",
    PREFILL_TPS_BUCKETS))
request_prompt_tokens = REGISTRY.add(Histogram(
    "qse_request_prompt_tokens", "prompt tokens per request, cached ones included",
    PROMPT_BUCKETS))
request_completion_tokens = REGISTRY.add(Histogram(
    "qse_request_completion_tokens", "completion tokens per request", COMPLETION_BUCKETS))


def _state_field(key: str):
    def read():
        st = _SOURCES.get("state") or {}
        v = st.get(key)
        return float(v) if isinstance(v, (int, float)) else {}
    return read


def _rss():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) * 1024.0
    except Exception:                                              # noqa: BLE001
        pass
    return {}


def _ledger_stat(key: str):
    def read():
        led = (_SOURCES.get("state") or {}).get("ledger")
        if led is None:
            return {}
        st = led.stats
        return float(st["dropped"] + st["failed"]) if key == "dropped" else float(st[key])
    return read


def _log_subscribers():
    st = _SOURCES.get("state") or {}
    buf = st.get("log_buffer")
    if buf is None:
        try:
            from server import logbuf
            buf = logbuf.BUFFER
        except Exception:                                          # noqa: BLE001
            return {}
    return float(buf.subscribers)


def _build_info():
    b = _SOURCES.get("build") or {}
    return {tuple(b.get(k, "") for k in BUILD_LABELS): 1.0} if b else {}


BUILD_LABELS = ("version", "git_sha", "code_sha256", "flags_sha256")

REGISTRY.add(Gauge("qse_suffix_store_tokens", "token ids in the persistent suffix store",
                   collect=_cache_field("suffix_store", "tokens")))
REGISTRY.add(Gauge("qse_state_store_budget_bytes", "the state cache's byte budget",
                   collect=_cache_field("state_store", "budget")))
REGISTRY.add(Gauge("qse_process_resident_memory_bytes", "resident set of the server process",
                   collect=_rss))
REGISTRY.add(Gauge("qse_engine_start_time_seconds", "when the server started, unix seconds",
                   collect=lambda: float(_SOURCES["started"])))
REGISTRY.add(Gauge("qse_log_subscribers", "open /v1/dashboard/logs streams",
                   collect=_log_subscribers))
REGISTRY.add(Counter("qse_usage_ledger_rows_total", "rows the usage ledger has written",
                     collect=_ledger_stat("rows")))
REGISTRY.add(Counter("qse_usage_ledger_dropped_total",
                     "rows the usage ledger lost: a full queue or a failed write",
                     collect=_ledger_stat("dropped")))
REGISTRY.add(Gauge("qse_build_info",
                   "always 1. version, the git commit, the code hash row3 records "
                   "(tools/row3.py::code_hash) and a hash of the effective flags",
                   BUILD_LABELS, collect=_build_info))

# --------------------------------------------------------------------------- the hooks

_REQ = threading.local()


def render() -> str:
    return REGISTRY.render()


CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"


def serve(handler) -> None:
    """Answer GET /metrics on a `BaseHTTPRequestHandler`.

    Content-Length rather than a chunked body: the server speaks HTTP/1.1 with keep-alive, and a
    body with neither framing hangs the scraper until its own timeout.
    """
    raw = render().encode("utf-8")
    handler.send_response(200)
    handler.send_header("Content-Type", CONTENT_TYPE)
    handler.send_header("Content-Length", str(len(raw)))
    handler.end_headers()
    handler.wfile.write(raw)


def on_request(finish: str, n_prompt: int, n_out: int, seconds: float,
               exc: BaseException | None = None) -> None:
    """One generation ended. Called once, from wherever the server logs its one line per request."""
    requests_total.inc(finish_reason=str(finish))
    prompt_tokens_total.inc(float(n_prompt))
    generation_tokens_total.inc(float(n_out))
    e2e_request_latency_seconds.observe(max(0.0, float(seconds)))
    if finish in ("stop", "length"):
        request_success_total.inc()
    if exc is not None:
        errors_total.inc(type=type(exc).__name__)
    elif finish == "error":
        errors_total.inc(type="unknown")


def on_conductor_event(ev: dict) -> None:
    """A sink of the monitor's journal: one decision counted as it is written, the request's
    passes (`COUNTS`) at its end."""
    d = ev.get("decision")
    if d == "COUNTS":
        for task, tier, mode, decision, n in ev.get("counts") or ():
            conductor_decisions_total.inc(float(n), task=str(task), tier=str(tier),
                                          mode=str(mode), decision=str(decision).lower())
        return
    if d in (None, "OUTCOME", "LABEL"):
        return
    mode = str(ev.get("mode") or "")
    task = str(ev.get("task") or "")
    conductor_decisions_total.inc(task=task, tier=str(ev.get("tier", 0)), mode=mode,
                                  decision=str(d).lower())
    if mode != "act" or d == "PASS":
        return
    conductor_actions_total.inc(task=task, action=str(d))
    hidden = int(ev.get("hidden_tokens") or 0)
    if hidden:
        conductor_hidden_tokens_total.inc(float(hidden))
    if d == "GIVE_UP":
        conductor_gave_up_total.inc()
    if d == "CLOSE_REASONING" and ev.get("at_tokens"):
        conductor_reasoning_tokens_cut_total.inc(float(ev["at_tokens"]))


def on_route(row: dict) -> None:
    """The router's decision for one request, from its journal row at the end of the request."""
    conductor_decisions_total.inc(task="route",
                                  tier="1" if row.get("source") in ("decider", "qwen") else "0",
                                  mode=str(row.get("mode") or ""),
                                  decision=str(row.get("class_id") or "none"))
    for st in row.get("stages") or ():
        if st.get("cached") or str(st.get("error") or "").endswith(("_role_off",
                                                                    "_no_endpoint")):
            continue
        conductor_tier1_seconds.observe(max(0.0, float(st.get("ms") or 0)) / 1e3)
        if st.get("error"):
            conductor_tier1_failures_total.inc(reason=str(st["error"]))
    extra = row.get("extra") or {}
    for mode, got in (("act", (extra.get("package_choice") or {}).get("kept_out")),
                      ("shadow", ((row.get("shadow") or {}).get("tools") or {}).get("kept_out"))):
        for k in got or ():
            conductor_package_kept_out_total.inc(why=str(k.get("why") or ""), mode=mode)
    notes = sum(int(n.get("tokens") or 0) for n in row.get("injected") or ())
    replayed = int((row.get("replayed") or {}).get("tokens") or 0)
    if row.get("injected"):
        conductor_actions_total.inc(task="route", action="PROMPT_TAIL_NOTE")
    elif row.get("acted"):
        conductor_actions_total.inc(task="route", action="SET_THINK_BUDGET")
    if notes + replayed:
        conductor_hidden_tokens_total.inc(float(notes + replayed))


def on_record(rec) -> None:
    """One request is over: the numbers its `RequestRecord` holds that `on_request` does not.

    Called once per request, whatever happened to it, from `server/app.py::_account`. A refusal
    has no generation and so never reached `on_request`; it is counted here as a finish reason.
    """
    if rec.finish_reason == "refused":
        requests_total.inc(finish_reason="refused")
        return
    if rec.prompt_tokens is None:
        return                                       # rejected before the engine: a 400
    prompt_tokens_cached_total.inc(float(rec.cached_tokens or 0))
    reasoning_tokens_total.inc(float(rec.reasoning_tokens or 0))
    request_prompt_tokens.observe(float(rec.prompt_tokens))
    request_completion_tokens.observe(float(rec.completion_tokens))
    if rec.queue_ms is not None:
        request_queue_seconds.observe(rec.queue_ms / 1e3)
    if rec.prompt_ms is not None:
        request_prefill_seconds.observe(rec.prompt_ms / 1e3)
    replay = rec.cache_source in ("response", "prepared")      # server/usage.py REPLAY
    if not replay and rec.finish_reason != "error":
        if rec.decode_tps is not None:
            request_decode_tps.observe(rec.decode_tps)
        if rec.prefill_tps is not None:
            request_prefill_tps.observe(rec.prefill_tps)


def track_http(fn):
    """Wrap `Handler.send_response`: one count per response, by route and status."""
    def wrapped(self, code, message=None):
        try:
            http_requests_total.inc(route=route_of(getattr(self, "path", "")), code=str(int(code)))
        except Exception:                                          # noqa: BLE001
            pass
        return fn(self, code, message)
    wrapped.__name__ = getattr(fn, "__name__", "send_response")
    wrapped.__wrapped__ = fn
    return wrapped


def track_stream(fn):
    """Wrap `generate_stream` so the token times are taken where the tokens are.

    Time to first token is measured from the moment the request arrived, which is what vLLM's
    metric of that name means and what a caller experiences; the queue wait is part of it. The
    arrival time comes from the `_complete` wrapper below, and a generator called from anywhere
    else falls back to its own start.
    """
    def wrapped(*args, **kwargs):
        t0 = getattr(_REQ, "arrived", None) or time.perf_counter()
        first = True
        prev = None
        for tok in fn(*args, **kwargs):
            now = time.perf_counter()
            if first:
                time_to_first_token_seconds.observe(now - t0)
                first = False
            else:
                time_per_output_token_seconds.observe(now - prev)
            prev = now
            yield tok
    wrapped.__name__ = getattr(fn, "__name__", "generate_stream")
    wrapped.__doc__ = fn.__doc__
    wrapped.__wrapped__ = fn
    return wrapped


def track_complete(fn):
    """Stamp the arrival time of one request on its own thread, and count the refusals.

    A refusal returns before any generation exists, so `_log_request` never sees it. The queue
    counters in the server's own `INFLIGHT` dictionary do, which is why the refusal reasons are
    read from there rather than counted here.
    """
    def wrapped(self, *args, **kwargs):
        _REQ.arrived = time.perf_counter()
        _REQ.pending = None
        try:
            return fn(self, *args, **kwargs)
        finally:
            _REQ.arrived = None
    wrapped.__name__ = getattr(fn, "__name__", "_complete")
    wrapped.__doc__ = fn.__doc__
    wrapped.__wrapped__ = fn
    return wrapped


def track_log(fn):
    def wrapped(cid, n_prompt, n_out, finish, t0, *, stream, exc=None, pen=None, pattern=None,
                rec=None, temp=0.0, tools=None, sup=None):
        try:
            on_request(finish, n_prompt, n_out, time.perf_counter() - t0, exc)
        except Exception:                                          # noqa: BLE001
            pass
        extra = {"rec": rec} if rec is not None else {}
        if temp:
            extra["temp"] = temp
        if tools:
            extra["tools"] = tools
        if sup is not None:
            extra["sup"] = sup
        return fn(cid, n_prompt, n_out, finish, t0, stream=stream, exc=exc, pen=pen,
                  pattern=pattern, **extra)
    wrapped.__name__ = getattr(fn, "__name__", "_log_request")
    wrapped.__doc__ = fn.__doc__
    wrapped.__wrapped__ = fn
    return wrapped


def instrument_drafter(drafter):
    """Count what the drafter proposed, what the target kept and what each cost.

    The wrappers are set on the INSTANCE, so they shadow the class methods and survive
    `drafter.reset()`, which the server calls once a request. `hasattr(drafter, "propose_tree")`
    decides the server's tree path, so a drafter that has no tree must not grow one here.

    `observe` is called with single tokens as well -- after a prefill, on a step the drafter
    declined, and for the tokens a reasoning budget forces -- so accepted tokens are counted only
    for an `observe` that follows a proposal, and the pairing is held per thread.
    """
    if drafter is None or getattr(drafter, "_qse_instrumented", False):
        return drafter

    def _count_draft(width: int, n_draft: int) -> None:
        spec_decode_num_drafts_total.inc()
        spec_decode_num_draft_tokens_total.inc(float(n_draft))
        draft_width_chosen_total.inc(width=str(int(width)))
        _REQ.pending = int(n_draft)

    def _record(name, out, seconds, before) -> None:
        draft_seconds.observe(seconds)
        if name == "propose":
            n = len(out) if out else 0
        else:
            n = getattr(out, "n_draft", 0) if out is not None else 0
        if n:
            _count_draft(n + 1, n)
            if before is not None:
                after = getattr(_SOURCES.get("suffix"), "matched", before)
                suffix_store_drafts_total.inc(outcome="hit" if after != before else "miss")

    # a drafter with `propose_tree_steps` has its steps counted instead of `propose_tree`,
    # whose plain call goes through them (counting both would count every draft twice). The
    # launch-first loop stops the steps after the draft launch and streams; that time is not the
    # draft's, so it is not in `draft_seconds`.
    steps_inner = getattr(drafter, "propose_tree_steps", None)
    names = ("propose",) if steps_inner is not None else ("propose", "propose_tree")
    for name in names:
        inner = getattr(drafter, name, None)
        if inner is None:
            continue

        def make(inner=inner, name=name):
            def call(*args, **kwargs):
                before = getattr(_SOURCES.get("suffix"), "matched", None)
                t0 = time.perf_counter()
                out = inner(*args, **kwargs)
                _record(name, out, time.perf_counter() - t0, before)
                return out
            return call
        setattr(drafter, name, make())
    if steps_inner is not None:
        def steps(*args, _inner=steps_inner, **kwargs):
            before = getattr(_SOURCES.get("suffix"), "matched", None)
            gen = _inner(*args, **kwargs)
            spent, t = 0.0, time.perf_counter()
            while True:
                try:
                    next(gen)
                except StopIteration as done:
                    out = done.value
                    break
                spent += time.perf_counter() - t
                yield
                t = time.perf_counter()
            _record("propose_tree", out, spent + time.perf_counter() - t, before)
            return out
        drafter.propose_tree_steps = steps

    inner_observe = getattr(drafter, "observe", None)
    if inner_observe is not None:
        def observe(tokens, _inner=inner_observe):
            pending = getattr(_REQ, "pending", None)
            if pending is not None:
                _REQ.pending = None
                committed = len(tokens)
                spec_accept_per_block.observe(float(committed))
                spec_decode_num_accepted_tokens_total.inc(float(max(0, committed - 1)))
            return _inner(tokens)
        drafter.observe = observe

    inner_verify = getattr(drafter, "on_verify", None)
    if inner_verify is not None:
        def on_verify(width, ms, _inner=inner_verify):
            verify_seconds.observe(float(ms) / 1000.0)
            return _inner(width, ms)
        drafter.on_verify = on_verify
    else:
        drafter.on_verify = lambda width, ms: verify_seconds.observe(float(ms) / 1000.0)

    drafter._qse_instrumented = True
    return drafter


def bind(state=None, inflight=None, cache_stats=None, info: dict | None = None,
         build: dict | None = None, suffix=None) -> None:
    """Point the scrape-time gauges at the server's own objects. Safe to call more than once."""
    if state is not None:
        _SOURCES["state"] = state
        _SOURCES["started"] = state.get("started", _SOURCES["started"])
    if build is not None:
        _SOURCES["build"] = {k: str(build.get(k, "")) for k in BUILD_LABELS}
    if suffix is not None:
        _SOURCES["suffix"] = suffix
    if inflight is not None:
        _SOURCES["inflight"] = inflight
    if cache_stats is not None:
        _SOURCES["cache_stats"] = cache_stats
    if info is not None:
        _SOURCES["info"] = {k: str(info.get(k, "")) for k in INFO_LABELS}


def flags_hash(args: dict, env: dict | None = None) -> str:
    """Sha256 over the effective flags and every QWEN38_* variable: two processes with the same
    hash ran the same configuration. The values themselves are in /v1/dashboard/system."""
    import hashlib
    import json
    import os
    env = os.environ if env is None else env
    blob = json.dumps({"args": {k: str(v) for k, v in sorted(args.items())},
                       "env": {k: v for k, v in sorted(env.items()) if k.startswith("QWEN38_")}},
                      sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def install(app) -> None:
    """Wire the hooks into a running `server/app.py`. One call, from `main()`.

    `app` is the module object -- `sys.modules[__name__]` inside `app.py`, which is `__main__`
    when the server is started as a script. The three functions are rebound as module globals, so
    the calls inside `_complete` find the wrapped versions without `_complete` itself changing.
    """
    if getattr(app, "_qse_installed", False):
        return
    state = getattr(app, "STATE", {})
    app.generate_stream = track_stream(app.generate_stream)
    app._log_request = track_log(app._log_request)
    app.Handler._complete = track_complete(app.Handler._complete)
    if hasattr(app.Handler, "send_response"):
        app.Handler.send_response = track_http(app.Handler.send_response)
    bind(suffix=state.get("suffix_store"))
    instrument_drafter(state.get("drafter"))
    eng = state.get("engine")
    w = getattr(eng, "w", None)
    bind(state, getattr(app, "INFLIGHT", None), getattr(app, "cache_stats", None), info={
        "version": VERSION,
        "model": state.get("model", ""),
        "drafter": type(state.get("drafter")).__name__,
        "width": str(state.get("k", "")),
        "tree": "1" if state.get("tree") else "0",
        "nvfp4": str(getattr(w, "nvfp4_source", "") or ""),
        "fp8_head": "1" if getattr(w, "fp8_head_source", None) else "0",
        "max_len": str(state.get("max_len", "")),
        "caches": ",".join(n for n, on in (("session", state.get("session_cache")),
                                           ("prefix", state.get("prefix_cache")),
                                           ("response", state.get("response_cache")),
                                           ("suffix", state.get("suffix_store"))) if on) or "none",
    }, build={"version": state.get("version", ""), "git_sha": state.get("git_sha", ""),
              "code_sha256": state.get("code_sha256", ""),
              "flags_sha256": flags_hash(state.get("args") or {})})
    app._qse_installed = True

