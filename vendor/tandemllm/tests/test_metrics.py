"""The metrics endpoint: that the text parses, and that the numbers mean what the names say.

Run: `python -m pytest tests/ -q`, or `python tests/test_metrics.py` with no pytest installed.

No torch, no tokeniser, no board. Two kinds of test live here and they fail for different reasons.
The format tests fail when a scraper would refuse the page -- a missing `# TYPE`, a bucket series
that is not cumulative, a label value with an unescaped quote in it. The semantic tests fail when
the page parses and lies: a success counted for a generation that raised, accepted tokens counted
for an `observe` that followed no proposal.

The drafter here is eleven lines. The one in the engine holds two fine-tuned checkpoints and a
suffix array; what is under test is the pairing of `propose` with `observe`, and a real drafter
would only make that pairing harder to see.
"""

from __future__ import annotations

import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server import metrics as M  # noqa: E402

NAME = r"[a-zA-Z_:][a-zA-Z0-9_:]*"
SAMPLE = re.compile(rf"^(?P<name>{NAME})(?P<labels>\{{.*\}})?\s(?P<value>\S+)$")
LABEL = re.compile(rf'(?P<k>{NAME})="(?P<v>(?:[^"\\]|\\.)*)"')


def _fresh():
    M.REGISTRY.reset()
    M._SOURCES["cache_stats"] = None
    M._SOURCES["inflight"] = None
    M._SOURCES["info"] = {}
    M._REQ.__dict__.clear()


def parse(text: str) -> dict:
    """Read a scrape the way a scraper does, and refuse anything it would refuse.

    Returns {family: {"type": str, "samples": [(name, labels, value)]}}.
    """
    families: dict[str, dict] = {}
    declared: dict[str, str] = {}
    helped: set[str] = set()
    for line in text.split("\n"):
        if not line:
            continue
        if line.startswith("# HELP "):
            name = line[len("# HELP "):].split(" ", 1)[0]
            assert re.fullmatch(NAME, name), f"bad name in HELP: {line}"
            assert name not in helped, f"{name} has two HELP lines"
            helped.add(name)
            continue
        if line.startswith("# TYPE "):
            rest = line[len("# TYPE "):].split(" ")
            assert len(rest) == 2, f"bad TYPE line: {line}"
            name, kind = rest
            assert kind in ("counter", "gauge", "histogram", "summary", "untyped"), line
            assert name not in declared, f"{name} has two TYPE lines"
            declared[name] = kind
            families[name] = {"type": kind, "samples": []}
            continue
        assert not line.startswith("#"), f"unexpected comment: {line}"
        m = SAMPLE.match(line)
        assert m, f"not a sample line: {line!r}"
        name, raw, value = m.group("name"), m.group("labels"), m.group("value")
        labels = {}
        if raw:
            inner = raw[1:-1]
            for lm in LABEL.finditer(inner):
                labels[lm.group("k")] = lm.group("v")
            assert len(labels) == inner.count("=") or not inner, f"bad labels: {line}"
        if value in ("+Inf", "-Inf", "NaN"):
            v = float(value.replace("Inf", "inf").replace("NaN", "nan"))
        else:
            v = float(value)
        base = name
        for suffix in ("_bucket", "_sum", "_count"):
            if name.endswith(suffix) and name[: -len(suffix)] in declared:
                base = name[: -len(suffix)]
                break
        assert base in declared, f"{name} has no # TYPE line"
        families[base]["samples"].append((name, labels, v))
    for name in declared:
        assert name in helped, f"{name} has a TYPE and no HELP"
    return families


# --- the format ---------------------------------------------------------------------------------

def test_an_empty_registry_still_scrapes():
    _fresh()
    text = M.render()
    parse(text)
    assert text.endswith("\n") or text == ""


def test_a_full_scrape_parses():
    _fresh()
    M.on_request("stop", 256, 256, 8.4)
    M.on_request("error", 10, 3, 0.5, exc=ValueError("no"))
    M.time_to_first_token_seconds.observe(0.761)
    M.time_per_output_token_seconds.observe(0.0001)
    M.verify_seconds.observe(0.0986)
    M.draft_seconds.observe(0.0259)
    M.spec_accept_per_block.observe(4.0)
    M.draft_width_chosen_total.inc(width="16")
    M.bind(inflight={"waiting": 2, "running": 1}, info={"version": "0.1.0", "model": "m",
                                                        "drafter": "LengthRouter", "width": "15",
                                                        "tree": "1", "nvfp4": "a.safetensors",
                                                        "fp8_head": "1", "max_len": "32768",
                                                        "caches": "session,prefix"})
    fam = parse(M.render())
    assert fam["qse_requests_total"]["type"] == "counter"
    assert fam["qse_time_to_first_token_seconds"]["type"] == "histogram"
    assert fam["qse_requests_waiting"]["samples"] == [("qse_requests_waiting", {}, 2.0)]


def test_every_documented_metric_is_registered():
    # server/METRICS.md is the contract; a rename that forgets the document fails here.
    want = {
        "qse_requests_total", "qse_requests_running", "qse_requests_waiting",
        "qse_request_success_total", "qse_requests_refused_total", "qse_prompt_tokens_total",
        "qse_generation_tokens_total", "qse_time_to_first_token_seconds",
        "qse_time_per_output_token_seconds", "qse_e2e_request_latency_seconds",
        "qse_spec_decode_num_accepted_tokens_total", "qse_spec_decode_num_draft_tokens_total",
        "qse_spec_decode_num_drafts_total", "qse_spec_accept_per_block",
        "qse_draft_width_chosen_total", "qse_verify_seconds", "qse_draft_seconds",
        "qse_cache_hits_total", "qse_cache_misses_total", "qse_cache_evictions_total",
        "qse_cache_bytes", "qse_cache_entries", "qse_cache_collisions_total",
        "qse_prefill_tokens_reused_total", "qse_prefill_tokens_forwarded_total",
        "qse_gpu_memory_used_bytes", "qse_gpu_memory_reserved_bytes",
        "qse_unified_memory_free_bytes", "qse_engine_uptime_seconds", "qse_errors_total",
        "qse_engine_info", "qse_memory_bytes", "qse_memory_shed_total",
    }
    have = {m.name for m in M.REGISTRY._metrics}
    assert want <= have, f"missing: {sorted(want - have)}"
    assert all(n.startswith("qse_") for n in have)


def test_a_name_is_registered_once():
    r = M.Registry()
    r.add(M.Counter("qse_x", "x"))
    try:
        r.add(M.Gauge("qse_x", "x again"))
    except ValueError:
        return
    raise AssertionError("a second metric of the same name must be refused")


def test_label_values_are_escaped():
    _fresh()
    M.errors_total.inc(type='Bad"\\Name\nwith breaks')
    line = [ln for ln in M.render().split("\n") if ln.startswith("qse_errors_total{")][0]
    assert '\\"' in line and "\\\\" in line and "\\n" in line
    assert "\n" not in line
    fam = parse(M.render())
    assert fam["qse_errors_total"]["samples"][0][2] == 1.0


def test_special_floats_use_the_format_spelling():
    assert M._fmt(float("inf")) == "+Inf"
    assert M._fmt(float("nan")) == "NaN"
    assert M._fmt(3.0) == "3"
    assert M._fmt(0.25) == "0.25"


# --- counters and histograms ---------------------------------------------------------------------

def test_a_counter_accumulates_per_label_set():
    _fresh()
    M.requests_total.inc(finish_reason="stop")
    M.requests_total.inc(finish_reason="stop")
    M.requests_total.inc(finish_reason="length")
    fam = parse(M.render())["qse_requests_total"]
    got = {tuple(sorted(lb.items())): v for _, lb, v in fam["samples"]}
    assert got[(("finish_reason", "stop"),)] == 2.0
    assert got[(("finish_reason", "length"),)] == 1.0


def test_a_counter_refuses_the_wrong_labels():
    _fresh()
    for bad in ({}, {"finish_reason": "stop", "extra": "1"}, {"reason": "stop"}):
        try:
            M.requests_total.inc(**bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad} should not be accepted")


def test_histogram_buckets_are_cumulative_and_end_at_the_count():
    _fresh()
    for v in (0.001, 0.02, 0.1, 0.9, 7.0, 3600.0):
        M.e2e_request_latency_seconds.observe(v)
    fam = parse(M.render())["qse_e2e_request_latency_seconds"]
    buckets = [(float(lb["le"].replace("+Inf", "inf")), v)
               for n, lb, v in fam["samples"] if n.endswith("_bucket")]
    assert buckets == sorted(buckets), "buckets must be emitted in ascending le"
    values = [v for _, v in buckets]
    assert values == sorted(values), "a cumulative series never falls"
    count = [v for n, _, v in fam["samples"] if n.endswith("_count")][0]
    total = [v for n, _, v in fam["samples"] if n.endswith("_sum")][0]
    assert buckets[-1][0] == math.inf and buckets[-1][1] == count == 6.0
    assert abs(total - (0.001 + 0.02 + 0.1 + 0.9 + 7.0 + 3600.0)) < 1e-9
    # an observation above the last finite bound only lands in +Inf
    assert buckets[-2][1] == 5.0


def test_a_histogram_refuses_unsorted_buckets():
    for bad in ((1.0, 0.5), (1.0, 1.0)):
        try:
            M.Histogram("qse_bad", "no", bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad} should not be accepted")


def test_a_gauge_that_raises_does_not_break_the_scrape():
    _fresh()

    def boom():
        raise RuntimeError("the allocator is not there")

    r = M.Registry()
    r.add(M.Gauge("qse_boom", "raises", collect=boom))
    r.add(M.Gauge("qse_fine", "does not", collect=lambda: 7.0))
    text = r.render()
    assert "qse_boom" not in text
    assert "qse_fine 7" in text
    parse(text)


# --- what the numbers mean ----------------------------------------------------------------------

def test_on_request_counts_a_finished_generation():
    _fresh()
    M.on_request("stop", 256, 300, 9.0)
    fam = parse(M.render())
    assert fam["qse_prompt_tokens_total"]["samples"][0][2] == 256.0
    assert fam["qse_generation_tokens_total"]["samples"][0][2] == 300.0
    assert fam["qse_request_success_total"]["samples"][0][2] == 1.0
    assert "qse_errors_total" not in fam


def test_only_stop_and_length_are_successes():
    _fresh()
    for finish in ("stop", "length", "timeout", "abandoned", "error"):
        M.on_request(finish, 1, 1, 0.1)
    fam = parse(M.render())
    assert fam["qse_request_success_total"]["samples"][0][2] == 2.0
    assert len(fam["qse_requests_total"]["samples"]) == 5
    # an `error` finish with no exception object is still an error, under a name that says so
    types = {lb["type"] for _, lb, _ in fam["qse_errors_total"]["samples"]}
    assert types == {"unknown"}


def test_an_exception_is_counted_by_its_class():
    _fresh()
    M.on_request("error", 10, 2, 1.0, exc=KeyError("k"))
    fam = parse(M.render())
    assert fam["qse_errors_total"]["samples"][0][1] == {"type": "KeyError"}


def test_track_stream_measures_the_first_token_and_the_gaps():
    _fresh()

    def gen():
        for t in (1, 2, 3, 4):
            yield t

    assert list(M.track_stream(gen)()) == [1, 2, 3, 4]
    ttft = M.time_to_first_token_seconds
    tpot = M.time_per_output_token_seconds
    assert ttft.totals[()] == 1.0, "one first token per generation"
    assert tpot.totals[()] == 3.0, "three gaps between four tokens"


def test_track_stream_uses_the_arrival_time_when_there_is_one():
    _fresh()
    import time
    M._REQ.arrived = time.perf_counter() - 2.0

    def gen():
        yield 1

    list(M.track_stream(gen)())
    assert M.time_to_first_token_seconds.sums[()] >= 2.0, "the queue wait belongs in TTFT"


class _Drafter:
    """The three calls the decode loop makes, and nothing else."""

    def __init__(self, n_draft: int = 15):
        self.n = n_draft
        self.seen: list = []

    def propose_tree(self, ctx, k):
        return type("T", (), {"n_draft": self.n, "tokens": [0] * (self.n + 1)})()

    def observe(self, tokens):
        self.seen.append(list(tokens))

    def on_verify(self, width, ms):
        self.seen.append(("verify", width, ms))


def test_a_block_is_counted_once_at_its_own_width():
    _fresh()
    d = M.instrument_drafter(_Drafter(15))
    d.propose_tree([1, 2], 15)
    d.on_verify(16, 98.6)
    d.observe([7, 8, 9, 10])                      # three drafted tokens kept plus the target's own
    fam = parse(M.render())
    val = {f["samples"][0][0]: f["samples"][0][2] for f in
           (fam["qse_spec_decode_num_drafts_total"], fam["qse_spec_decode_num_draft_tokens_total"],
            fam["qse_spec_decode_num_accepted_tokens_total"])}
    assert val["qse_spec_decode_num_drafts_total"] == 1.0
    assert val["qse_spec_decode_num_draft_tokens_total"] == 15.0
    assert val["qse_spec_decode_num_accepted_tokens_total"] == 3.0
    assert fam["qse_draft_width_chosen_total"]["samples"][0][1] == {"width": "16"}
    assert abs(M.verify_seconds.sums[()] - 0.0986) < 1e-9
    assert M.spec_accept_per_block.sums[()] == 4.0
    assert d.seen[-1] == [7, 8, 9, 10], "the wrapper still calls the drafter"


def test_a_steps_drafter_is_counted_once_either_way_and_not_for_its_stop():
    """The served router proposes through `propose_tree_steps` (the launch-first loop
    stops it after its draft launch) and its plain `propose_tree` drives the same generator. One
    proposal is one draft whichever way it is called, and `draft_seconds` has no stop in it."""
    import time as _t
    from engine.drafters import run_steps

    class Steps(_Drafter):
        def propose_tree_steps(self, ctx, k):
            yield
            return type("T", (), {"n_draft": self.n, "tokens": [0] * (self.n + 1)})()

        def propose_tree(self, ctx, k):
            return run_steps(self.propose_tree_steps(ctx, k))

    _fresh()
    d = M.instrument_drafter(Steps(7))
    d.propose_tree([1, 2], 7)
    assert M.spec_decode_num_drafts_total.values[()] == 1.0
    g = d.propose_tree_steps([1, 2], 7)
    next(g)
    _t.sleep(0.05)
    try:
        next(g)
    except StopIteration as done:
        assert done.value.n_draft == 7
    assert M.spec_decode_num_drafts_total.values[()] == 2.0
    assert M.spec_decode_num_draft_tokens_total.values[()] == 14.0
    assert M.draft_seconds.sums[()] < 0.02, M.draft_seconds.sums
    assert hasattr(d, "propose_tree")


def test_an_observe_without_a_proposal_is_not_a_block():
    _fresh()
    d = M.instrument_drafter(_Drafter())
    d.observe([5])                                # the prefill's first token
    d.observe([6])                                # a step the drafter declined
    assert M.spec_accept_per_block.totals == {}
    assert M.spec_decode_num_accepted_tokens_total.values == {}


def test_a_declined_proposal_is_not_a_block():
    _fresh()

    class Silent(_Drafter):
        def propose_tree(self, ctx, k):
            return None

    d = M.instrument_drafter(Silent())
    d.propose_tree([1], 15)
    d.observe([4])
    assert M.spec_decode_num_drafts_total.values == {}
    assert M.draft_seconds.totals[()] == 1.0, "a decline still costs drafting time"


def test_instrumenting_twice_does_not_double_count():
    _fresh()
    d = M.instrument_drafter(_Drafter(7))
    M.instrument_drafter(d)
    d.propose_tree([1], 7)
    d.observe([1, 2])
    assert M.spec_decode_num_drafts_total.values[()] == 1.0


def test_a_chain_drafter_does_not_grow_a_tree():
    _fresh()

    class Chain:
        def propose(self, ctx, k):
            return [1, 2, 3]

        def observe(self, tokens):
            pass

    d = M.instrument_drafter(Chain())
    assert not hasattr(d, "propose_tree"), "the server reads this attribute to pick its path"
    d.propose([0], 3)
    d.observe([1, 2])
    assert M.spec_decode_num_draft_tokens_total.values[()] == 3.0
    assert M.draft_width_chosen_total.values[("4",)] == 1.0


def test_refusals_are_read_from_the_server_s_own_queue_counters():
    _fresh()
    M.bind(inflight={"waiting": 8, "running": 1, "refused": 4, "timeouts": 0})
    fam = parse(M.render())
    assert fam["qse_requests_refused_total"]["type"] == "counter"
    assert fam["qse_requests_refused_total"]["samples"] == [
        ("qse_requests_refused_total", {"reason": "queue"}, 4.0)]
    assert fam["qse_requests_waiting"]["samples"][0][2] == 8.0


def test_the_cache_gauges_read_the_server_report():
    _fresh()
    M.bind(cache_stats=lambda: {
        "state_store": {"entries": 15, "bytes": 4_540_000_000, "hits": 9, "misses": 31,
                        "evictions": 0, "rejected_collisions": 0, "tokens_reused": 1536,
                        "tokens_forwarded": 188},
        "response_cache": None,
        "suffix_store": {"tokens": 235_763},
    })
    fam = parse(M.render())
    by_cache = {lb["cache"]: v for _, lb, v in fam["qse_cache_bytes"]["samples"]}
    assert by_cache == {"state": 4_540_000_000.0, "suffix": 235_763.0 * 4}
    assert fam["qse_cache_hits_total"]["samples"] == [("qse_cache_hits_total", {"cache": "state"},
                                                      9.0)]
    assert fam["qse_prefill_tokens_reused_total"]["samples"][0][2] == 1536.0


def test_a_cache_that_is_off_reports_nothing():
    _fresh()
    M.bind(cache_stats=lambda: {"state_store": None, "response_cache": None,
                                "suffix_store": None})
    text = M.render()
    assert "qse_cache_bytes" not in text and "qse_cache_hits_total" not in text


# --- the wiring ---------------------------------------------------------------------------------

class _FakeHandler:
    def __init__(self):
        self.status = None
        self.headers: dict[str, str] = {}
        self.wfile = self

    def send_response(self, code):
        self.status = code

    def send_header(self, k, v):
        self.headers[k] = v

    def end_headers(self):
        pass

    def write(self, raw):
        self.body = raw


def test_serve_sends_a_length_and_the_right_content_type():
    _fresh()
    M.requests_total.inc(finish_reason="stop")
    h = _FakeHandler()
    M.serve(h)
    assert h.status == 200
    assert h.headers["Content-Type"].startswith("text/plain; version=0.0.4")
    assert int(h.headers["Content-Length"]) == len(h.body)
    parse(h.body.decode())


def test_install_wires_the_three_hooks_once():
    _fresh()

    class Handler:
        def _complete(self, body, chat):
            return ("done", getattr(M._REQ, "arrived", None))

    logged: list = []

    class App:
        STATE = {"model": "qwen38-spark-engine", "drafter": _Drafter(15), "k": 15, "tree": True,
                 "max_len": 32768, "session_cache": True, "prefix_cache": True,
                 "response_cache": None, "suffix_store": object(), "started": 1_700_000_000}
        INFLIGHT = {"waiting": 0, "running": 1, "served": 3, "refused": 0, "errors": 0,
                    "timeouts": 0, "abandoned": 0}

        @staticmethod
        def cache_stats():
            return {}

        @staticmethod
        def generate_stream(*a, **k):
            yield from (1, 2)

        @staticmethod
        def _log_request(cid, n_prompt, n_out, finish, t0, *, stream, exc=None, pen=None,
                         pattern=None):
            logged.append((cid, finish, n_out))

    app = App()
    app.Handler = Handler
    M.install(app)
    M.install(app)                                   # a second call must change nothing
    assert app.generate_stream.__wrapped__ is App.__dict__["generate_stream"].__func__

    out, arrived = Handler()._complete({}, chat=True)
    assert out == "done" and arrived is not None, "the arrival time is on the request's own thread"
    list(app.generate_stream())
    app._log_request("id", 8, 2, "stop", __import__("time").perf_counter() - 1.0, stream=True)
    fam = parse(M.render())
    assert logged == [("id", "stop", 2)], "the server's own log line still runs"
    assert fam["qse_requests_total"]["samples"][0][2] == 1.0
    assert fam["qse_time_to_first_token_seconds"]["samples"][-1][2] == 1.0
    info = fam["qse_engine_info"]["samples"][0][1]
    assert info["model"] == "qwen38-spark-engine" and info["tree"] == "1"
    assert info["caches"] == "session,prefix,suffix"
    assert fam["qse_requests_running"]["samples"][0][2] == 1.0


# --- contract 0.2.0 -----------------------------------------------------------------------

class _Rec:
    """The fields of server/usage.py's RequestRecord that on_record reads."""

    def __init__(self, **kw):
        self.finish_reason, self.prompt_tokens, self.cached_tokens = "stop", 100, 0
        self.completion_tokens, self.reasoning_tokens = 50, 0
        self.queue_ms, self.prompt_ms, self.decode_tps, self.prefill_tps = 0.4, 400.0, 45.0, 900.0
        self.cache_source = "none"
        self.__dict__.update(kw)


def _sample(fam, name, **labels):
    base = re.sub(r"_(bucket|sum|count)$", "", name) if name not in fam else name
    for n, lb, v in fam[base]["samples"]:
        if n == name and all(lb.get(k) == str(v2) for k, v2 in labels.items()):
            return v
    return None


def test_token_counters_follow_the_responses():
    _fresh()
    for p, c, cached, reason in ((1959, 412, 1536, 230), (100, 50, 0, 0)):
        M.on_request("stop", p, c, 1.0)
        M.on_record(_Rec(prompt_tokens=p, completion_tokens=c, cached_tokens=cached,
                         reasoning_tokens=reason))
    fam = parse(M.render())
    assert _sample(fam, "qse_prompt_tokens_total") == 2059
    assert _sample(fam, "qse_prompt_tokens_cached_total") == 1536
    assert _sample(fam, "qse_generation_tokens_total") == 462
    assert _sample(fam, "qse_reasoning_tokens_total") == 230
    assert _sample(fam, "qse_request_prompt_tokens_count") == 2


def test_the_per_request_speed_histogram():
    _fresh()
    M.on_record(_Rec(decode_tps=68.26))
    fam = parse(M.render())
    assert _sample(fam, "qse_request_decode_tokens_per_second_bucket", le=60) == 0
    assert _sample(fam, "qse_request_decode_tokens_per_second_bucket", le=70) == 1
    assert _sample(fam, "qse_request_decode_tokens_per_second_sum") == 68.26
    # a replay, an error and a refusal are not speeds; a refusal is a finish reason
    M.on_record(_Rec(decode_tps=99999.0, cache_source="response"))
    M.on_record(_Rec(decode_tps=1.0, finish_reason="error"))
    M.on_record(_Rec(finish_reason="refused", prompt_tokens=None))
    fam = parse(M.render())
    assert _sample(fam, "qse_request_decode_tokens_per_second_count") == 1
    assert _sample(fam, "qse_requests_total", finish_reason="refused") == 1


def test_route_and_status_counters_never_carry_a_raw_path():
    _fresh()

    class H:
        def __init__(self, path):
            self.path = path

    sent = []
    wrapped = M.track_http(lambda self, code, message=None: sent.append(code))
    for path, code in (("/v1/chat/completions", 200), ("/v1/dashboard/summary?tz=UTC", 401),
                       ("/v1/chat/completions", 503), ("/v1/dashboard/logs", 200),
                       ("/dashboard/assets/app-3f9a.js", 200), ("/health", 200),
                       ("/v1/cache/stats", 200), ("/secret/../x?q=1", 404), ("/metrics", 200)):
        wrapped(H(path), code)
    fam = parse(M.render())
    got = {(lb["route"], lb["code"]): v for _, lb, v in fam["qse_http_requests_total"]["samples"]}
    assert got == {("chat", "200"): 1, ("dashboard", "401"): 1, ("chat", "503"): 1,
                   ("dashboard", "200"): 1, ("static", "200"): 1, ("health", "200"): 1,
                   ("cache", "200"): 1, ("other", "404"): 1, ("metrics", "200"): 1}, got
    assert sent == [200, 401, 503, 200, 200, 200, 200, 404, 200]
    assert {lb["route"] for _, lb, _ in fam["qse_http_requests_total"]["samples"]} <= \
        {"chat", "completions", "models", "health", "metrics", "cache", "dashboard", "static",
         "other"}


def test_suffix_store_hits_are_counted_once_a_block():
    _fresh()

    class Store:
        matched = 0

    store = Store()

    class D:
        n = 0

        def propose(self, ctx, k):
            self.n += 1
            if self.n in (1, 3, 4, 8, 11, 15, 19):          # 7 of 20 blocks matched the store
                store.matched += 3                          # (a block may look up several times)
            return [1, 2, 3]

        def observe(self, toks):
            pass

    M.bind(suffix=store)
    d = M.instrument_drafter(D())
    for _ in range(20):
        d.propose([0], 3)
        d.observe([1, 2])
    fam = parse(M.render())
    assert _sample(fam, "qse_suffix_store_drafts_total", outcome="hit") == 7
    assert _sample(fam, "qse_suffix_store_drafts_total", outcome="miss") == 13
    M._SOURCES.pop("suffix", None)


def test_session_and_prefix_hits_are_told_apart():
    _fresh()
    M.bind(cache_stats=lambda: {"state_store": {"hits": 5, "hits_session": 2, "hits_prefix": 3,
                                                "misses": 1, "entries": 2, "bytes": 10},
                                "response_cache": {"hits": 4, "misses": 0},
                                "suffix_store": None})
    fam = parse(M.render())
    rows = {tuple(sorted(lb.items())): v for _, lb, v in fam["qse_cache_hits_total"]["samples"]}
    assert rows == {(("cache", "response"),): 4.0,
                    (("cache", "state"), ("kind", "prefix")): 3.0,
                    (("cache", "state"), ("kind", "session")): 2.0}, rows


def test_build_info_and_the_code_hash_row3_records():
    from pathlib import Path
    from tools.row3 import code_hash
    _fresh()
    root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    h = code_hash(root)
    M.bind(build={"version": "0.1.0-rc4", "git_sha": "7d0b176", "code_sha256": h,
                  "flags_sha256": M.flags_hash({"max_len": 8}, env={"QWEN38_DEEP": "32"})})
    fam = parse(M.render())
    (name, labels, value), = fam["qse_build_info"]["samples"]
    assert value == 1.0 and set(labels) == {"version", "git_sha", "code_sha256", "flags_sha256"}
    assert labels["code_sha256"] == h and len(labels["flags_sha256"]) == 64
    assert M.flags_hash({"max_len": 8}, env={"QWEN38_DEEP": "32"}) != \
        M.flags_hash({"max_len": 8}, env={"QWEN38_DEEP": "0"})
    assert M.flags_hash({"a": 1}, env={"HOME": "/x"}) == M.flags_hash({"a": 1}, env={})
    M._SOURCES.pop("build", None)


def test_the_ledger_and_log_gauges():
    _fresh()

    class Led:
        stats = {"rows": 812, "dropped": 1, "failed": 2}

    class Buf:
        subscribers = 3

    M.bind(state={"ledger": Led(), "log_buffer": Buf(), "started": 1_790_000_000})
    fam = parse(M.render())
    assert _sample(fam, "qse_usage_ledger_rows_total") == 812
    assert _sample(fam, "qse_usage_ledger_dropped_total") == 3
    assert _sample(fam, "qse_log_subscribers") == 3
    assert _sample(fam, "qse_engine_start_time_seconds") == 1_790_000_000
    M._SOURCES["state"] = None


def test_the_full_0_2_0_page_parses_and_labels_stay_small():
    _fresh()
    M.on_request("stop", 256, 256, 8.4)
    M.on_record(_Rec())
    M.on_record(_Rec(finish_reason="refused", prompt_tokens=None))
    M.track_http(lambda self, code, message=None: None)(type("H", (), {"path": "/x"})(), 404)
    M.bind(inflight={"waiting": 0, "running": 0, "refused": 1},
           cache_stats=lambda: {"state_store": {"hits": 1, "hits_session": 1, "hits_prefix": 0,
                                                "budget": 8 << 30, "bytes": 1, "entries": 1},
                                "suffix_store": {"tokens": 10}},
           info={k: "x" for k in M.INFO_LABELS},
           build={"version": "v", "git_sha": "g", "code_sha256": "c", "flags_sha256": "f"})
    text = M.render()
    fam = parse(text)
    assert M.VERSION == "0.2.0"
    for name in ("qse_http_requests_total", "qse_prompt_tokens_cached_total",
                 "qse_reasoning_tokens_total", "qse_request_queue_seconds",
                 "qse_request_prefill_seconds", "qse_request_decode_tokens_per_second",
                 "qse_request_prefill_tokens_per_second", "qse_request_prompt_tokens",
                 "qse_request_completion_tokens", "qse_suffix_store_tokens",
                 "qse_state_store_budget_bytes", "qse_engine_start_time_seconds",
                 "qse_build_info", "qse_log_subscribers"):
        assert name in fam, name
        assert f"# HELP {name} " in text and f"# TYPE {name} " in text, name
    values: dict = {}
    for f in fam.values():
        for _, lb, _ in f["samples"]:
            for k, v in lb.items():
                if k != "le":
                    values.setdefault(k, set()).add(v)
    assert all(len(v) <= 16 for v in values.values()), {k: len(v) for k, v in values.items()}


def _main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    bad = 0
    for fn in fns:
        try:
            fn()
            print(f"  ok  {fn.__name__}")
        except Exception as exc:                                   # noqa: BLE001
            bad += 1
            import traceback
            traceback.print_exc()
            print(f"FAIL  {fn.__name__}: {exc}")
    print(f"\n{len(fns) - bad}/{len(fns)} passed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(_main())
