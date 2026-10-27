"""vLLM and SGLang at layer L1: their own `/metrics` pages read through a name table, and their
discovery routes. No code of ours runs inside either engine.

    watch = Watch(service).start()      one thread; every vllm and sglang backend in turn
    watch.view(bid)                     the backend's `engine` block for the contract
    watch.families()                    the mapped families for the proxy's own /metrics page

WHAT IS READ, verified against the source of the two releases current on 2026-10-04:

    vLLM v0.30.0 (tag ced6857, 2026-09-22): vllm/v1/metrics/loggers.py and
        vllm/v1/spec_decode/metrics.py. Labels `model_name` and `engine` (the data-parallel engine
        index); every counter carries prometheus_client's `_total` suffix on the page. The
        spec-decode families exist only while speculation is configured:
        `vllm:spec_decode_num_drafts_total`, `..._num_draft_tokens_total`,
        `..._num_accepted_tokens_total` and `..._num_accepted_tokens_per_pos_total{position}`.
        `vllm:kv_cache_usage_perc` is a fraction (1 is full), whatever its name says.
    SGLang v0.5.21 (tag e00930c, 2026-10-02): sglang/srt/observability/metrics_collector.py
        (the module moved there from srt/metrics/collector.py). The page exists only with
        `--enable-metrics`; without it `/metrics` answers 404. Scheduler families carry
        `tp_rank`, `pp_rank` and `moe_ep_rank`, and only the first attention-TP rank reports
        unless `--enable-metrics-for-all-schedulers` is set, so a series of any other rank is
        dropped here instead of being counted twice. Tokenizer families carry `is_streaming`.
        Speculation is gauges over SGLang's own logging window (`spec_accept_rate`,
        `spec_accept_length`); there is no accepted-token counter.

THE MAPPING. Each family goes onto the proxy's page as `qse_backend_<name>` with a `backend` label
and nothing else (the engine's own families keep their labels: a relayed backend never enters
them). Series of the same family are summed across engines and data-parallel ranks; a fraction
(KV usage, a hit rate, an acceptance rate) is their mean. A family the page does not have maps to
nothing, and the view says "not reported" for it; a name the table does not know is ignored, so a
new release costs at most a missing tile, never a failed scrape.

THE POLL. Every 15 s per backend on a thread of its own, the page read with a 3 s timeout; a
failure doubles the wait up to 120 s, and a 404 marks the page `off` and is asked again every
120 s. Discovery every 10 minutes: vLLM `GET /version`; SGLang `GET /server_info`
(`/get_server_info` on releases before it). Neither engine's `/health` is used: on SGLang it runs
a one-token generation when the server is idle, and the health poller's `GET /v1/models` answers
the same question for free. The pages of the last 5 minutes are kept for the window figures.
"""

from __future__ import annotations

import collections
import json
import math
import threading
import time
import urllib.error
import urllib.request

from proxy import scrape

VERIFIED = {"vllm": "0.30.0", "sglang": "0.5.21"}

SCRAPE_EVERY_S = 15.0
BACKOFF_MAX_S = 120.0
DISCOVER_EVERY_S = 600.0
TIMEOUT_S = 3.0
WINDOW_S = 300.0
PAGE_LIMIT = 8 * 1024 * 1024

# the rank labels whose other values are copies of rank 0 (SGLang with every scheduler reporting)
RANKS = ("tp_rank", "pp_rank", "moe_ep_rank", "attn_cp_rank", "attn_tp_rank")


class Fam:
    """One mapped family: `qse_backend_<name>` from the first source name the page has.

    kind     counter, gauge or histogram (the type on the proxy's page)
    sources  the engine's sample names, tried in order (a counter's `_total` form first)
    how      sum or mean across the series that remain after `where` and the rank filter
    keep     {engine label: our label} kept on the mapped series (a reason, a position)
    where    {label: value} a series must have to count
    """

    __slots__ = ("name", "kind", "sources", "how", "keep", "where", "help")

    def __init__(self, name, kind, sources, help_text, how="sum", keep=None, where=None):
        self.name, self.kind, self.help = name, kind, help_text
        self.sources = (sources,) if isinstance(sources, str) else tuple(sources)
        self.how, self.keep, self.where = how, dict(keep or {}), dict(where or {})


def _c(name, src, help_text, **kw):
    base = src[:-6] if src.endswith("_total") else src
    return Fam(name, "counter", (base + "_total", base), help_text, **kw)


def _g(name, src, help_text, **kw):
    return Fam(name, "gauge", src, help_text, **kw)


def _h(name, src, help_text, **kw):
    return Fam(name, "histogram", src, help_text, **kw)


VLLM = (
    _g("requests_running", "vllm:num_requests_running", "requests in the engine's batches"),
    _g("requests_waiting", "vllm:num_requests_waiting", "requests waiting to be scheduled"),
    _g("kv_usage", "vllm:kv_cache_usage_perc", "KV cache in use, 0 to 1", how="mean"),
    _g("awake", "vllm:engine_sleep_state", "1 while the engine is awake, 0 asleep", how="mean",
       where={"sleep_state": "awake"}),
    _c("prompt_tokens_total", "vllm:prompt_tokens", "prompt tokens processed"),
    _c("generation_tokens_total", "vllm:generation_tokens", "tokens generated"),
    _c("prompt_tokens_cached_total", "vllm:prompt_tokens_cached",
       "prompt tokens served from a cache (local and external)"),
    _c("prefix_cache_queries_total", "vllm:prefix_cache_queries",
       "prompt tokens looked up in the prefix cache"),
    _c("prefix_cache_hits_total", "vllm:prefix_cache_hits",
       "prompt tokens found in the prefix cache"),
    _c("requests_finished_total", "vllm:request_success", "requests finished, by reason",
       keep={"finished_reason": "reason"}),
    _c("preemptions_total", "vllm:num_preemptions", "requests preempted"),
    _c("corrupted_requests_total", "vllm:corrupted_requests", "requests with NaNs in the logits"),
    _c("spec_drafts_total", "vllm:spec_decode_num_drafts", "speculative drafts"),
    _c("spec_draft_tokens_total", "vllm:spec_decode_num_draft_tokens", "draft tokens proposed"),
    _c("spec_accepted_tokens_total", "vllm:spec_decode_num_accepted_tokens",
       "draft tokens accepted"),
    _c("spec_accepted_tokens_per_pos_total", "vllm:spec_decode_num_accepted_tokens_per_pos",
       "draft tokens accepted, by draft position", keep={"position": "position"}),
    _h("ttft_seconds", "vllm:time_to_first_token_seconds", "time to first token"),
    _h("itl_seconds", "vllm:inter_token_latency_seconds", "time between two output tokens"),
    _h("tpot_seconds", "vllm:request_time_per_output_token_seconds",
       "a request's time per output token"),
    _h("e2e_seconds", "vllm:e2e_request_latency_seconds", "a request from arrival to its end"),
    _h("queue_seconds", "vllm:request_queue_time_seconds", "a request's time waiting"),
    _h("prefill_seconds", "vllm:request_prefill_time_seconds", "a request's prefill"),
    _h("decode_seconds", "vllm:request_decode_time_seconds", "a request's decode"),
    _h("inference_seconds", "vllm:request_inference_time_seconds",
       "a request's time running (prefill and decode)"),
)

SGLANG = (
    _g("requests_running", "sglang:num_running_reqs", "requests in the engine's batches"),
    _g("requests_waiting", "sglang:num_queue_reqs", "requests waiting to be scheduled"),
    _g("kv_usage", "sglang:token_usage", "KV cache in use, 0 to 1", how="mean"),
    _g("kv_tokens", "sglang:num_used_tokens", "KV tokens in use"),
    _g("kv_capacity_tokens", "sglang:max_total_num_tokens", "KV tokens the pool holds"),
    _g("gen_tps", "sglang:gen_throughput", "the engine's own generation rate, tokens a second"),
    _g("cache_hit_rate", "sglang:cache_hit_rate",
       "the engine's own prefix-cache hit rate over its logging window, 0 to 1", how="mean"),
    _c("prompt_tokens_total", "sglang:prompt_tokens_total", "prompt tokens processed"),
    _c("generation_tokens_total", "sglang:generation_tokens_total", "tokens generated"),
    _c("prompt_tokens_cached_total", "sglang:cached_tokens_total",
       "prompt tokens served from a cache (device, host, storage)"),
    _c("requests_finished_total", "sglang:num_requests_total", "requests processed"),
    _c("requests_aborted_total", "sglang:num_aborted_requests_total", "requests aborted"),
    _c("preemptions_total", "sglang:num_retracted_requests_total",
       "requests retracted (SGLang's preemption)"),
    _c("spec_verify_calls_total", "sglang:spec_verify_calls_total", "speculative verify calls"),
    _g("spec_accept_rate", "sglang:spec_accept_rate",
       "accepted over proposed drafts in the engine's last window, 0 to 1", how="mean"),
    _g("spec_accept_length", "sglang:spec_accept_length",
       "tokens a verify step commits in the engine's last window (the bonus token included)",
       how="mean"),
    _g("spec_steps", "sglang:spec_num_steps", "speculative steps configured", how="mean"),
    _g("spec_num_draft_tokens", "sglang:spec_num_draft_tokens", "draft tokens a step proposes",
       how="mean"),
    _h("ttft_seconds", "sglang:time_to_first_token_seconds", "time to first token"),
    _h("itl_seconds", "sglang:inter_token_latency_seconds", "time between two output tokens"),
    _h("e2e_seconds", "sglang:e2e_request_latency_seconds", "a request from arrival to its end"),
    _h("queue_seconds", "sglang:queue_time_seconds", "a request's time waiting"),
)

TABLES = {"vllm": VLLM, "sglang": SGLANG}
PREFIX = "qse_backend_"


# ---------------------------------------------------------------------------------- mapping

def _rank_ok(lab: dict) -> bool:
    for r in RANKS:
        v = lab.get(r)
        if v is not None and v not in ("0", ""):
            return False
    return True


def _where_ok(lab: dict, where: dict) -> bool:
    return all(lab.get(k) == v for k, v in where.items())


def _key(lab: dict, keep: dict) -> tuple:
    return tuple((ours, lab.get(theirs, "")) for theirs, ours in sorted(keep.items()))


def map_page(kind: str, text: str) -> dict:
    """A page of the engine's own families as {name: {"kind", "series": {key: value}}}; a
    histogram's value is {"buckets": [(le, cumulative)], "sum", "count"}. `key` is a tuple of
    (label, value) pairs from `keep` (usually empty). Families the page lacks are absent."""
    table = TABLES.get(kind)
    if not table:
        return {}
    by_name: dict = collections.defaultdict(list)
    for name, lab, v in scrape.parse(text):
        by_name[name].append((lab, v))
    out = {}
    for fam in table:
        if fam.kind == "histogram":
            got = _histogram(fam, by_name)
        else:
            got = _plain(fam, by_name)
        if got:
            out[fam.name] = {"kind": fam.kind, "series": got}
    return out


def _plain(fam: Fam, by_name: dict) -> dict:
    for src in fam.sources:
        rows = by_name.get(src)
        if not rows:
            continue
        groups: dict = collections.defaultdict(list)
        for lab, v in rows:
            if not _rank_ok(lab) or not _where_ok(lab, fam.where):
                continue
            if v is None or (isinstance(v, float) and math.isnan(v)):
                continue
            groups[_key(lab, fam.keep)].append(v)
        if not groups:
            return {}
        return {k: (sum(vs) / len(vs) if fam.how == "mean" else sum(vs))
                for k, vs in groups.items()}
    return {}


def _histogram(fam: Fam, by_name: dict) -> dict:
    src = fam.sources[0]
    buckets: dict = collections.defaultdict(lambda: collections.defaultdict(float))
    sums: dict = collections.defaultdict(float)
    counts: dict = collections.defaultdict(float)
    seen = set()
    for lab, v in by_name.get(src + "_bucket", ()):
        if not _rank_ok(lab) or not _where_ok(lab, fam.where):
            continue
        le = scrape._float(lab.get("le", "+Inf"))
        if le is None:
            continue
        k = _key(lab, fam.keep)
        buckets[k][le] += v
        seen.add(k)
    for suffix, store in (("_sum", sums), ("_count", counts)):
        for lab, v in by_name.get(src + suffix, ()):
            if not _rank_ok(lab) or not _where_ok(lab, fam.where):
                continue
            k = _key(lab, fam.keep)
            store[k] += v
            seen.add(k)
    return {k: {"buckets": sorted(buckets[k].items()), "sum": sums.get(k, 0.0),
                "count": counts.get(k, 0.0)} for k in seen}


def render(pages: list[tuple[str, dict]], info: dict | None = None) -> list[str]:
    """The mapped families of every backend as Prometheus text lines, one HELP and TYPE per
    family. `pages` is [(backend id, mapped page)]; `info` {backend id: (kind, version)} adds
    `qse_backend_info{backend,kind,version} 1`, which tells a reader of the page which table
    the backend's families came from."""
    help_of = {}
    for table in TABLES.values():
        for fam in table:
            help_of.setdefault(fam.name, fam.help)
    names: dict = collections.OrderedDict()
    for bid, page in pages:
        for name, fam in page.items():
            names.setdefault(name, (fam["kind"], []))[1].append((bid, fam["series"]))
    out = []
    if info:
        out.append(f"# HELP {PREFIX}info the engine behind a backend whose own /metrics page is "
                   "read (kind, version from its discovery route)")
        out.append(f"# TYPE {PREFIX}info gauge")
        for bid in sorted(info):
            kind, ver = info[bid]
            lab = _labels((("backend", bid), ("kind", kind), ("version", ver or "")))
            out.append(f"{PREFIX}info{lab} 1")
    for name in sorted(names):
        kind, rows = names[name]
        full = PREFIX + name
        out.append(f"# HELP {full} {help_of.get(name, name)} (from the backend's own /metrics)")
        out.append(f"# TYPE {full} {kind}")
        for bid, series in sorted(rows, key=lambda r: r[0]):
            for key in sorted(series):
                labels = (("backend", bid),) + tuple(key)
                v = series[key]
                if kind != "histogram":
                    out.append(f"{full}{_labels(labels)} {_fmt(v)}")
                    continue
                for le, c in v["buckets"]:
                    out.append(f"{full}_bucket{_labels(labels + (('le', _fmt(le)),))} {_fmt(c)}")
                out.append(f"{full}_sum{_labels(labels)} {_fmt(v['sum'])}")
                out.append(f"{full}_count{_labels(labels)} {_fmt(v['count'])}")
    return out


def _labels(pairs) -> str:
    inner = ",".join(f'{k}="{_esc(str(v))}"' for k, v in pairs)
    return "{" + inner + "}"


def _esc(v: str) -> str:
    return v.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _fmt(v: float) -> str:
    if v == math.inf:
        return "+Inf"
    if v == -math.inf:
        return "-Inf"
    if isinstance(v, float) and math.isnan(v):
        return "NaN"
    if v == int(v) and abs(v) < 1e15:
        return str(int(v))
    return repr(float(v))


# ---------------------------------------------------------------------------------- figures

def value(page: dict | None, name: str, **labels) -> float | None:
    """A counter or gauge summed over the series that match `labels`; None when absent."""
    fam = (page or {}).get(name)
    if not fam or fam["kind"] == "histogram":
        return None
    want = tuple(sorted(labels.items()))
    got = [v for k, v in fam["series"].items()
           if all(dict(k).get(a) == b for a, b in want)]
    return sum(got) if got else None


def histogram(page: dict | None, name: str) -> dict | None:
    fam = (page or {}).get(name)
    if not fam or fam["kind"] != "histogram":
        return None
    buckets: dict = collections.defaultdict(float)
    s = c = 0.0
    for h in fam["series"].values():
        for le, n in h["buckets"]:
            buckets[le] += n
        s += h["sum"]
        c += h["count"]
    return {"buckets": sorted(buckets.items()), "sum": s, "count": c}


def hist_delta(a: dict | None, b: dict | None) -> dict | None:
    if b is None:
        return None
    if a is None:
        return b
    am = dict(a["buckets"])
    return {"buckets": [(le, max(0.0, n - am.get(le, 0.0))) for le, n in b["buckets"]],
            "sum": max(0.0, b["sum"] - a["sum"]), "count": max(0.0, b["count"] - a["count"])}


def quantile(h: dict | None, q: float) -> float | None:
    """Prometheus' histogram_quantile: linear inside the bucket; None without observations."""
    if not h or not h["buckets"]:
        return None
    total = h["buckets"][-1][1]
    if total <= 0:
        return None
    rank = q * total
    prev_le, prev_n = 0.0, 0.0
    for le, n in h["buckets"]:
        if n >= rank:
            if math.isinf(le):
                return prev_le
            inside = n - prev_n
            if inside <= 0:
                return le
            return prev_le + (le - prev_le) * (rank - prev_n) / inside
        prev_le, prev_n = le, n
    return prev_le


def _d(a, b, name, **lab):
    va, vb = value(a, name, **lab), value(b, name, **lab)
    if vb is None:
        return None
    if va is None:
        # a counter the first reading lacks started at 0 in between: SGLang creates its
        # families at the first request (as hist_delta takes a missing histogram)
        if a is None:
            return None
        va = 0.0
    return max(0.0, vb - va)


def _ratio(n, d):
    return None if n is None or not d else n / d


def _r(x, nd=4):
    return None if x is None else round(x, nd)


def figures(kind: str, first: dict | None, last: dict | None, dt: float | None) -> dict:
    """The card's numbers from two mapped pages `dt` seconds apart (the window's first and last).
    Every figure the pages cannot give is None, never 0."""
    cur = last or {}
    win = first is not None and dt is not None and dt > 0
    ttft = hist_delta(histogram(first, "ttft_seconds"), histogram(cur, "ttft_seconds")) \
        if win else None

    def split(name):
        h = hist_delta(histogram(first, name), histogram(cur, name)) if win else None
        if not h or h["count"] <= 0:
            return None
        return {"p50": _r(quantile(h, 0.5)), "p90": _r(quantile(h, 0.9)),
                "count": int(h["count"])}

    gen = _d(first, cur, "generation_tokens_total") if win else None
    prompt = _d(first, cur, "prompt_tokens_total") if win else None
    cached = _d(first, cur, "prompt_tokens_cached_total") if win else None
    out = {
        "running": value(cur, "requests_running"),
        "waiting": value(cur, "requests_waiting"),
        "kv_usage": _r(value(cur, "kv_usage")),
        "kv_tokens": value(cur, "kv_tokens"),
        "kv_capacity_tokens": value(cur, "kv_capacity_tokens"),
        "awake": None if value(cur, "awake") is None else value(cur, "awake") >= 0.5,
        "decode_tps": _r(gen / dt, 2) if gen is not None else None,
        "prompt_tps": _r(prompt / dt, 2) if prompt is not None else None,
        "engine_gen_tps": _r(value(cur, "gen_tps"), 2),
        "cached_share": _r(_ratio(cached, prompt)),
        "prefix_hit_rate": None,
        "engine_cache_hit_rate": _r(value(cur, "cache_hit_rate")),
        "ttft_p50_s": _r(quantile(ttft, 0.5)) if ttft and ttft["count"] > 0 else None,
        "queue_s": split("queue_seconds"),
        "prefill_s": split("prefill_seconds"),
        "decode_s": split("decode_seconds"),
        "e2e_s": split("e2e_seconds"),
        "preemptions": _d(first, cur, "preemptions_total") if win else None,
        "finished": _d(first, cur, "requests_finished_total") if win else None,
        "spec": None,
    }
    if kind == "vllm" and win:
        out["prefix_hit_rate"] = _r(_ratio(_d(first, cur, "prefix_cache_hits_total"),
                                           _d(first, cur, "prefix_cache_queries_total")))
    if kind == "sglang" and win:
        # SGLang has no query counter: its cached-token counter over its prompt counter is the
        # same quantity (tokens found in a cache over tokens asked for)
        out["prefix_hit_rate"] = out["cached_share"]
    if "spec_draft_tokens_total" in cur or "spec_accepted_tokens_total" in cur:
        acc = _d(first, cur, "spec_accepted_tokens_total") if win else None
        drafted = _d(first, cur, "spec_draft_tokens_total") if win else None
        drafts = _d(first, cur, "spec_drafts_total") if win else None
        per_pos = None
        fam = cur.get("spec_accepted_tokens_per_pos_total")
        if fam and win and drafts:
            per_pos = []
            for key in sorted(fam["series"], key=lambda k: int(dict(k).get("position", 0) or 0)):
                pos = dict(key).get("position")
                d = _d(first, cur, "spec_accepted_tokens_per_pos_total", position=pos)
                per_pos.append(_r(_ratio(d, drafts)))
        out["spec"] = {"source": "counters", "acceptance": _r(_ratio(acc, drafted)),
                       # one verify step commits the accepted drafts plus the token it samples
                       "tokens_per_step": _r(None if not drafts or acc is None
                                             else acc / drafts + 1.0),
                       "per_position": per_pos, "drafts": drafts}
    elif "spec_accept_rate" in cur or "spec_accept_length" in cur:
        out["spec"] = {"source": "engine_window", "acceptance": _r(value(cur, "spec_accept_rate")),
                       "tokens_per_step": _r(value(cur, "spec_accept_length")),
                       "per_position": None, "drafts": None}
    return out


# ---------------------------------------------------------------------------------- the poller

def root_of(url: str) -> str:
    """The server's root from a backend URL: `http://h:8000/v1` -> `http://h:8000`."""
    u = (url or "").rstrip("/")
    if u.endswith("/v1"):
        u = u[:-3]
    return u


def _fetch(url: str, key: str | None, timeout: float) -> tuple[int, bytes]:
    req = urllib.request.Request(url, headers={"Accept": "text/plain, application/json"})
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:      # noqa: S310
            return r.status, r.read(PAGE_LIMIT)
    except urllib.error.HTTPError as e:
        return e.code, b""


def _find(obj, key, depth=0):
    """A key anywhere in a nested dict (SGLang's server_info is flat today, grouped before)."""
    if not isinstance(obj, dict) or depth > 3:
        return None
    if key in obj:
        return obj[key]
    for v in obj.values():
        got = _find(v, key, depth + 1)
        if got is not None:
            return got
    return None


def parse_info(kind: str, obj) -> dict:
    """The discovery answer in the card's shape; every field the answer lacks is None."""
    out = {"version": None, "context_length": None, "speculative": None, "cache_report": None,
           "reasoning_parser": None, "metrics_enabled": None}
    if not isinstance(obj, dict):
        return out
    v = obj.get("version")
    out["version"] = str(v) if v is not None else None
    if kind == "sglang":
        algo = _find(obj, "speculative_algorithm")
        out["speculative"] = None if algo in (None, "", "NONE") else {
            "algorithm": algo, "num_steps": _find(obj, "speculative_num_steps"),
            "num_draft_tokens": _find(obj, "speculative_num_draft_tokens")}
        ctx = _find(obj, "context_length")
        out["context_length"] = ctx if isinstance(ctx, int) else None
        cr = _find(obj, "enable_cache_report")
        out["cache_report"] = bool(cr) if cr is not None else None
        rp = _find(obj, "reasoning_parser")
        out["reasoning_parser"] = rp if isinstance(rp, str) and rp else None
        me = _find(obj, "enable_metrics")
        out["metrics_enabled"] = bool(me) if me is not None else None
    return out


class _State:
    __slots__ = ("pages", "state", "error", "at", "due", "fails", "info", "info_at", "info_due",
                 "status", "page_ms", "kind")

    def __init__(self, kind: str = ""):
        self.kind = kind
        self.pages = collections.deque()      # (t, mapped page) for the last WINDOW_S
        self.state = "unknown"                # ok, off (404), error, unknown
        self.error = None
        self.at = None                        # the last good scrape
        self.due = 0.0
        self.fails = 0
        self.info = None
        self.info_at = None
        self.info_due = 0.0
        self.status = None
        self.page_ms = None                   # the last page's parse and map time


class Watch:
    """The L1 poller for every enabled backend of kind vllm or sglang."""

    def __init__(self, service, *, every=SCRAPE_EVERY_S, discover_every=DISCOVER_EVERY_S,
                 timeout=TIMEOUT_S, window=WINDOW_S, clock=time.time, fetch=_fetch):
        self.svc = service
        self.every, self.discover_every = every, discover_every
        self.timeout, self.window = timeout, window
        self.clock, self.fetch = clock, fetch
        self.st: dict[str, _State] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> "Watch":
        self._thread = threading.Thread(target=self._run, name="proxy-engines", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()

    def kick(self, bid: str | None = None) -> None:
        with self._lock:
            for k, s in self.st.items():
                if bid is None or k == bid:
                    s.due = 0.0
                    s.info_due = 0.0
        self._wake.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.tick()
            except Exception as exc:                               # noqa: BLE001
                self.svc.log(f"[proxy] engine scrape failed: {type(exc).__name__}: {exc}")
            self._wake.wait(1.0)
            self._wake.clear()

    def _backends(self) -> list[dict]:
        return [b for b in self.svc.cfg.backends() if b["kind"] in TABLES and b["enabled"]]

    def tick(self) -> None:
        now = self.clock()
        live = set()
        for b in self._backends():
            bid = b["id"]
            live.add(bid)
            with self._lock:
                s = self.st.get(bid)
                if s is None or s.kind != b["kind"]:
                    s = self.st[bid] = _State(b["kind"])
            if now >= s.info_due:
                self.discover(b, s)
            if now >= s.due:
                self.scrape(b, s)
        with self._lock:
            for bid in [k for k in self.st if k not in live]:
                del self.st[bid]

    def _key(self, bid: str) -> str | None:
        try:
            return self.svc.cfg.key_for(bid)
        except Exception:                                          # noqa: BLE001
            return None

    def discover(self, b: dict, s: _State) -> None:
        now = self.clock()
        s.info_due = now + self.discover_every
        root = root_of(b["url"])
        paths = ("/version",) if b["kind"] == "vllm" else ("/server_info", "/get_server_info")
        for p in paths:
            try:
                status, body = self.fetch(root + p, self._key(b["id"]), self.timeout)
            except Exception:                                      # noqa: BLE001
                continue
            if status == 200:
                try:
                    s.info = parse_info(b["kind"], json.loads(body))
                    s.info_at = now
                except ValueError:
                    pass
                return

    def scrape(self, b: dict, s: _State) -> None:
        now = self.clock()
        try:
            status, body = self.fetch(root_of(b["url"]) + "/metrics", self._key(b["id"]),
                                      self.timeout)
        except Exception as exc:                                   # noqa: BLE001
            status, body = None, b""
            err = f"{type(exc).__name__}: {exc}"
        else:
            err = None if status == 200 else f"HTTP {status}"
        s.status = status
        if status == 200:
            t0 = time.perf_counter()
            try:
                page = map_page(b["kind"], body.decode("utf-8", "replace"))
            except Exception as exc:                               # noqa: BLE001
                page, err = None, f"unreadable page: {type(exc).__name__}"
            s.page_ms = round((time.perf_counter() - t0) * 1e3, 3)
            if page is not None:
                with self._lock:
                    s.pages.append((now, page))
                    while s.pages and now - s.pages[0][0] > self.window:
                        s.pages.popleft()
                if s.state != "ok":
                    self.svc.log(f"[proxy] backend {b['id']}: /metrics read "
                                 f"({len(page)} families mapped)")
                s.state, s.error, s.at, s.fails = "ok", None, now, 0
                s.due = now + self.every
                return
        s.fails += 1
        if status == 404:
            if s.state != "off":
                self.svc.log(f"[proxy] backend {b['id']}: no /metrics (404)"
                             + (", start SGLang with --enable-metrics"
                                if b["kind"] == "sglang" else ""))
            s.state = "off"
            s.due = now + BACKOFF_MAX_S
        else:
            if s.state != "error":
                self.svc.log(f"[proxy] backend {b['id']}: /metrics failed: {err}")
            s.state = "error"
            s.due = now + min(BACKOFF_MAX_S, self.every * (2 ** min(s.fails, 4)))
        s.error = err
        with self._lock:
            s.pages.clear()

    # ------------------------------------------------------------------------------ reading
    def latest(self) -> list[tuple[str, dict]]:
        with self._lock:
            return [(bid, s.pages[-1][1]) for bid, s in self.st.items() if s.pages]

    def kinds(self) -> dict:
        """{backend id: (kind, version)} for every backend whose page was read."""
        with self._lock:
            return {bid: (s.kind, (s.info or {}).get("version")) for bid, s in self.st.items()
                    if s.pages}

    def view(self, bid: str, kind: str | None = None) -> dict | None:
        """The `engine` block of a backend's status row; None for a kind without a table."""
        if kind is not None and kind not in TABLES:
            return None
        with self._lock:
            s = self.st.get(bid)
            pages = list(s.pages) if s else []
        if s is None:
            return {"layer": "L1", "metrics": "unknown", "error": None, "scrape_age_s": None,
                    "version": None, "verified": VERIFIED.get(kind or ""), "info": None,
                    "figures": None}
        now = self.clock()
        figs = None
        if pages:
            (t0, p0), (t1, p1) = pages[0], pages[-1]
            figs = figures(kind or "", p0 if t1 > t0 else None, p1, t1 - t0 if t1 > t0 else None)
            figs["window_s"] = round(t1 - t0, 1) if t1 > t0 else None
        info = s.info
        return {"layer": "L1", "metrics": s.state, "error": s.error,
                "scrape_age_s": None if s.at is None else round(now - s.at, 1),
                "version": info.get("version") if info else None,
                "verified": VERIFIED.get(kind or ""), "info": info, "figures": figs}

    def scrape_age(self, bid: str) -> float | None:
        with self._lock:
            s = self.st.get(bid)
            at = s.at if s else None
        return None if at is None else round(self.clock() - at, 1)


class Families:
    """The mapped families on the proxy's /metrics page: a `Registry` member that renders every
    backend's newest page (server/metrics.py calls `render()` at each scrape)."""

    kind = "untyped"

    def __init__(self, watch: Watch):
        self.watch = watch
        self.name = "qse_backend_engine_families"
        self.values: dict = {}

    def render(self) -> list[str]:
        try:
            return render(self.watch.latest(), self.watch.kinds())
        except Exception:                                          # noqa: BLE001
            return []
