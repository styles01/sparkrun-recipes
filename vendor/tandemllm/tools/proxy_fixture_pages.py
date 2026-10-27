"""Write the vLLM and SGLang `/metrics` fixtures for the proxy's L1 adapters (P2, P3).

    python tools/proxy_fixture_pages.py [--out tests/fixtures/proxy]

Needs `prometheus_client` (the library both engines expose their pages with; not a dependency of
this repository). Each family is declared as the engine's own code declares it, with the same
name, type, help text, labels and bucket bounds, read from the source of the releases named in
the file names:

    vLLM v0.30.0    vllm/v1/metrics/loggers.py, vllm/v1/spec_decode/metrics.py,
                    vllm/v1/metrics/buckets.py (labels model_name, engine)
    SGLang v0.5.21  sglang/srt/observability/metrics_collector.py (scheduler labels model_name,
                    engine_type, tp_rank, pp_rank, moe_ep_rank; tokenizer labels model_name,
                    engine_type, plus is_streaming where the code adds it)

and rendered by prometheus_client itself, so the text is what the engines' `/metrics` routes
send. Two pages per engine, 15 s apart (`-a`, `-b`), carry a known workload, so the tests can
check every derived figure by hand; `vllm-...-nospec` is a server without speculation. The
night pass (tools/proxy_e2e.sh --engines) records the real pages of both engines beside these.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

VLLM_TTFT = (0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0,
             7.5, 10.0, 20.0, 40.0, 80.0, 160.0, 640.0, 2560.0)
VLLM_ITL = (0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5,
            10.0, 20.0, 40.0, 80.0)
VLLM_LAT = (0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0,
            120.0, 240.0, 480.0, 960.0, 1920.0, 7680.0)
VLLM_PREEMPT = (1, 2, 3, 4, 5, 10, 20)
MODEL = "Qwen/Qwen3.5-0.8B"


def _1_2_5(max_value):
    out, e = [], 0
    while True:
        for m in (1, 2, 5):
            v = m * 10 ** e
            if v > max_value:
                return out
            out.append(v)
        e += 1


# ------------------------------------------------------------------------------------- vLLM

def vllm_page(stage: str, spec: bool = True) -> bytes:
    """stage a: the server after 45 requests; stage b: 15 s later, 3 more requests."""
    reg = CollectorRegistry()
    L = ["model_name", "engine"]
    lv = [MODEL, "0"]
    g = lambda n, d, extra=(): Gauge(n, d, L + list(extra), registry=reg)  # noqa: E731
    c = lambda n, d, extra=(): Counter(n, d, L + list(extra), registry=reg)  # noqa: E731
    h = lambda n, d, b: Histogram(n, d, L, buckets=b, registry=reg)  # noqa: E731
    b = stage == "b"
    g("vllm:num_requests_running", "Number of requests in model execution batches.") \
        .labels(*lv).set(1 if b else 2)
    g("vllm:num_requests_waiting", "Number of requests waiting to be processed.") \
        .labels(*lv).set(0 if b else 1)
    wr = g("vllm:num_requests_waiting_by_reason", "Number of waiting requests by reason. Reason "
           "labels: 'capacity' = waiting for scheduling capacity; 'deferred' = deferred by "
           "transient constraints (LoRA budget, KV transfer, blocked status). Sum of all reasons "
           "equals vllm:num_requests_waiting.", ["reason"])
    wr.labels(*lv, "capacity").set(0 if b else 1)
    wr.labels(*lv, "deferred").set(0)
    sl = g("vllm:engine_sleep_state", "Engine sleep state; awake = 0 means engine is sleeping; "
           "awake = 1 means engine is awake; weights_offloaded = 1 means sleep level 1; "
           "discard_all = 1 means sleep level 2.", ["sleep_state"])
    sl.labels(*lv, "awake").set(1)
    sl.labels(*lv, "weights_offloaded").set(0)
    sl.labels(*lv, "discard_all").set(0)
    g("vllm:kv_cache_usage_perc", "KV-cache usage. 1 means 100 percent usage.") \
        .labels(*lv).set(0.27 if b else 0.31)
    c("vllm:prefix_cache_queries", "Prefix cache queries, in terms of number of queried "
      "tokens.").labels(*lv).inc(53000 if b else 52000)
    c("vllm:prefix_cache_hits", "Prefix cache hits, in terms of number of cached tokens.") \
        .labels(*lv).inc(30800 if b else 30000)
    c("vllm:external_prefix_cache_queries", "External prefix cache queries from KV connector "
      "cross-instance cache sharing, in terms of number of queried tokens.").labels(*lv).inc(0)
    c("vllm:external_prefix_cache_hits", "External prefix cache hits from KV connector "
      "cross-instance cache sharing, in terms of number of cached tokens.").labels(*lv).inc(0)
    c("vllm:num_preemptions", "Cumulative number of preemption from the engine.") \
        .labels(*lv).inc(1)
    c("vllm:prompt_tokens", "Number of prefill tokens processed.") \
        .labels(*lv).inc(53000 if b else 52000)
    src = c("vllm:prompt_tokens_by_source", "Number of prompt tokens by source.", ["source"])
    src.labels(*lv, "local_compute").inc(22200 if b else 22000)
    src.labels(*lv, "local_cache_hit").inc(30800 if b else 30000)
    src.labels(*lv, "external_kv_transfer").inc(0)
    c("vllm:prompt_tokens_cached", "Number of cached prompt tokens (local + external).") \
        .labels(*lv).inc(30800 if b else 30000)
    c("vllm:generation_tokens", "Number of generation tokens processed.") \
        .labels(*lv).inc(9600 if b else 9000)
    ok = c("vllm:request_success", "Count of successfully processed requests.",
           ["finished_reason"])
    for reason, n in (("stop", 43 if b else 40), ("length", 2), ("abort", 1), ("error", 0),
                      ("repetition", 0)):
        ok.labels(*lv, reason).inc(n)
    tok = _1_2_5(32768)
    for name, doc in (("vllm:request_prompt_tokens", "Number of prefill tokens processed."),
                      ("vllm:request_generation_tokens", "Number of generation tokens "
                       "processed."),
                      ("vllm:request_max_num_generation_tokens", "Histogram of maximum number of "
                       "requested generation tokens."),
                      ("vllm:request_params_max_tokens", "Histogram of the max_tokens request "
                       "parameter."),
                      ("vllm:request_prefill_kv_computed_tokens", "Histogram of new KV tokens "
                       "computed during prefill (excluding cached tokens).")):
        hh = h(name, doc, tok)
        for v in (1200,) * (46 if b else 43):
            hh.labels(*lv).observe(v)
    it = h("vllm:iteration_tokens_total", "Histogram of number of tokens per engine_step.",
           (1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384))
    for v in (1, 2, 512) * 30:
        it.labels(*lv).observe(v)
    pn = h("vllm:request_params_n", "Histogram of the n request parameter.", (1, 2, 5, 10, 20))
    for _ in range(46 if b else 43):
        pn.labels(*lv).observe(1)
    # the timing histograms: 43 requests by stage a, three more by stage b
    ttft = h("vllm:time_to_first_token_seconds", "Histogram of time to first token in seconds.",
             VLLM_TTFT)
    itl = h("vllm:inter_token_latency_seconds", "Histogram of inter-token latency in seconds.",
            VLLM_ITL)
    tpot = h("vllm:request_time_per_output_token_seconds", "Histogram of "
             "time_per_output_token_seconds per request.", VLLM_ITL)
    e2e = h("vllm:e2e_request_latency_seconds", "Histogram of e2e request latency in seconds.",
            VLLM_LAT)
    queue = h("vllm:request_queue_time_seconds", "Histogram of time spent in WAITING phase for "
              "request.", VLLM_LAT)
    inf = h("vllm:request_inference_time_seconds", "Histogram of time spent in RUNNING phase "
            "for request.", VLLM_LAT)
    pre = h("vllm:request_prefill_time_seconds", "Histogram of time spent in PREFILL phase for "
            "request.", VLLM_LAT)
    dec = h("vllm:request_decode_time_seconds", "Histogram of time spent in DECODE phase for "
            "request.", VLLM_LAT)
    npre = h("vllm:request_num_preemptions", "Histogram of the number of times a request was "
             "preempted.", VLLM_PREEMPT)
    old = [(0.05, 0.4, 0.2, 2.0, 2.4)] * 43
    new = [(0.09, 0.2, 0.3, 4.0, 4.5), (0.2, 0.6, 0.6, 1.2, 2.0), (0.3, 0.9, 0.9, 3.0, 4.2)]
    for ft, q, p, d, e in old + (new if b else []):
        ttft.labels(*lv).observe(ft)
        queue.labels(*lv).observe(q)
        pre.labels(*lv).observe(p)
        dec.labels(*lv).observe(d)
        inf.labels(*lv).observe(p + d)
        e2e.labels(*lv).observe(e)
        tpot.labels(*lv).observe(0.025)
        npre.labels(*lv).observe(0)
    for _ in range(9600 if b else 9000):
        itl.labels(*lv).observe(0.024)
    if spec:
        c("vllm:spec_decode_num_drafts", "Number of spec decoding drafts.") \
            .labels(*lv).inc(3200 if b else 3000)
        c("vllm:spec_decode_num_draft_tokens", "Number of draft tokens.") \
            .labels(*lv).inc(9600 if b else 9000)
        c("vllm:spec_decode_num_accepted_tokens", "Number of accepted tokens.") \
            .labels(*lv).inc(6750 if b else 6300)
        pp = c("vllm:spec_decode_num_accepted_tokens_per_pos", "Accepted tokens per draft "
               "position.", ["position"])
        for pos, (na, nb) in enumerate(((2700, 2890), (2100, 2250), (1500, 1610))):
            pp.labels(*lv, str(pos)).inc(nb if b else na)
    info = Gauge("vllm:cache_config_info", "Information of the LLMEngine CacheConfig",
                 ["block_size", "cache_dtype", "enable_prefix_caching", "gpu_memory_utilization",
                  "num_gpu_blocks", "engine"], registry=reg)
    info.labels("16", "auto", "True", "0.3", "23040", "0").set(1)
    return generate_latest(reg)


# ------------------------------------------------------------------------------------- SGLang

SG_TTFT = (0.1, 0.2, 0.4, 0.6, 0.8, 1, 2, 4, 6, 8, 10, 20, 40, 60, 80, 100, 200, 400)
SG_E2E = (0.1, 0.2, 0.4, 0.6, 0.8, 1, 2, 4, 6, 8, 10, 20, 40, 60, 80, 100, 200, 400, 600, 1200,
          1800, 2400)
SG_ITL = (0.002, 0.004, 0.006, 0.008, 0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040, 0.060,
          0.080, 0.100, 0.200, 0.400, 0.600, 0.800, 1.000, 2.000, 4.000, 6.000, 8.000)
SG_QUEUE = (0.000, 0.001, 0.005, 0.010, 0.050, 0.100, 0.200, 0.500, 1, 2, 3, 4, 5, 10, 15, 20,
            30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200,
            1400, 1600, 1800, 2000, 2500, 3000)


def sglang_page(stage: str) -> bytes:
    """stage a: the server after 30 requests; b: 15 s later, two more and a busy decode."""
    reg = CollectorRegistry()
    b = stage == "b"
    SL = ["model_name", "engine_type", "tp_rank", "pp_rank", "moe_ep_rank"]
    TL = ["model_name", "engine_type"]
    s0 = [MODEL, "unified", "0", "0", "0"]
    s1 = [MODEL, "unified", "1", "0", "0"]       # a second TP rank: every scheduler reporting
    t = [MODEL, "unified"]
    gs = lambda n, d: Gauge(n, d, SL, registry=reg, multiprocess_mode="mostrecent")  # noqa: E731
    for name, doc, va, vb in (
            ("sglang:num_running_reqs", "The number of running requests.", 3, 2),
            ("sglang:num_queue_reqs", "The number of requests in the waiting queue.", 1, 0),
            ("sglang:num_grammar_queue_reqs", "The number of requests in the grammar waiting "
             "queue.", 0, 0),
            ("sglang:gen_throughput", "The generation throughput (token/s).", 112.0, 118.5),
            ("sglang:cache_hit_rate", "The prefix cache hit rate.", 0.58, 0.61),
            ("sglang:token_usage", "The token usage.", 0.42, 0.4),
            ("sglang:num_used_tokens", "The number of used tokens.", 42000, 40000),
            ("sglang:max_total_num_tokens", "Maximum total number of tokens in the KV cache "
             "pool.", 100000, 100000),
            ("sglang:spec_accept_length", "Mean acceptance length of speculative decoding "
             "(accepted drafts + bonus token per forward).", 3.0, 3.1),
            ("sglang:spec_accept_rate", "Speculative acceptance rate (`accepted drafts / "
             "proposed drafts` in batch).", 0.66, 0.7),
            ("sglang:spec_num_steps", "Currently active speculative_num_steps.", 3, 3),
            ("sglang:spec_num_draft_tokens", "Currently active speculative_num_draft_tokens "
             "(decouples from steps under topk>1).", 4, 4)):
        gg = gs(name, doc)
        gg.labels(*s0).set(vb if b else va)
        gg.labels(*s1).set(vb if b else va)
    Counter("sglang:num_retracted_requests_total", "Total number of retracted requests.", SL,
            registry=reg).labels(*s0).inc(2)
    qt = Histogram("sglang:queue_time_seconds", "Histogram of queueing time in seconds.", SL,
                   buckets=SG_QUEUE, registry=reg)
    for v in [0.02] * 30 + ([0.3, 1.5] if b else []):
        qt.labels(*s0).observe(v)
    pt = Counter("sglang:prompt_tokens_total", "Number of prefill tokens processed.",
                 TL + ["is_streaming"], registry=reg)
    pt.labels(*t, "true").inc(30000 if not b else 31000)
    pt.labels(*t, "false").inc(10000 if not b else 10200)
    gt = Counter("sglang:generation_tokens_total", "Number of generation tokens processed.",
                 TL + ["is_streaming"], registry=reg)
    gt.labels(*t, "true").inc(6000 if not b else 7650)
    gt.labels(*t, "false").inc(1000 if not b else 1150)
    Counter("sglang:spec_verify_calls_total", "Number of speculative decoding verification "
            "calls.", TL, registry=reg).labels(*t).inc(2300 if not b else 2860)
    ct = Counter("sglang:cached_tokens_total", "Number of cached prompt tokens by source "
                 "(device/host/storage).", TL + ["cache_source"], registry=reg)
    ct.labels(*t, "device").inc(22000 if not b else 22780)
    ct.labels(*t, "host").inc(1000 if not b else 1060)
    nr = Counter("sglang:num_requests_total", "Number of requests processed.",
                 TL + ["is_streaming"], registry=reg)
    nr.labels(*t, "true").inc(25 if not b else 27)
    nr.labels(*t, "false").inc(5)
    Counter("sglang:num_aborted_requests_total", "Number of requests aborted.", TL,
            registry=reg).labels(*t).inc(1)
    ttft = Histogram("sglang:time_to_first_token_seconds", "Histogram of time to first token in "
                     "seconds.", TL + ["is_streaming"], buckets=SG_TTFT, registry=reg)
    e2e = Histogram("sglang:e2e_request_latency_seconds", "Histogram of End-to-end request "
                    "latency in seconds", TL, buckets=SG_E2E, registry=reg)
    itl = Histogram("sglang:inter_token_latency_seconds", "Histogram of inter-token latency in "
                    "seconds.", TL, buckets=SG_ITL, registry=reg)
    for v in [0.15] * 30 + ([0.5, 0.9] if b else []):
        ttft.labels(*t, "true").observe(v)
        e2e.labels(*t).observe(v + 3.0)
    for _ in range(7000 if not b else 8800):
        itl.labels(*t).observe(0.009)
    return generate_latest(reg)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--out", default=os.path.join(here, "tests", "fixtures", "proxy"))
    a = ap.parse_args()
    # the `_created` samples carry the creation time: pinned, so the files do not change per run
    time.time = lambda: 1791138000.0
    head = ("# Written by tools/proxy_fixture_pages.py: the families as the release declares "
            "them, rendered by prometheus_client.\n")
    pages = {"vllm-metrics-0.30.0-a.txt": vllm_page("a"),
             "vllm-metrics-0.30.0-b.txt": vllm_page("b"),
             "vllm-metrics-0.30.0-nospec.txt": vllm_page("a", spec=False),
             "sglang-metrics-0.5.21-a.txt": sglang_page("a"),
             "sglang-metrics-0.5.21-b.txt": sglang_page("b")}
    for name, body in pages.items():
        with open(os.path.join(a.out, name), "w") as f:
            f.write(head + body.decode())
        print(name, len(body.splitlines()), "lines")
    # the proxy's own page over the same two readings, and the figures proxy/engines.py derives:
    # the dashboard's test (dashboard/tests/engines.test.ts) must derive the same from the page
    sys.path.insert(0, here)
    from proxy import engines
    mapped = {}
    for stage in ("a", "b"):
        v = engines.map_page("vllm", pages[f"vllm-metrics-0.30.0-{stage}.txt"].decode())
        s = engines.map_page("sglang", pages[f"sglang-metrics-0.5.21-{stage}.txt"].decode())
        mapped[stage] = {"vllm1": v, "sglang1": s}
        lines = engines.render([("vllm1", v), ("sglang1", s)],
                               {"vllm1": ("vllm", "0.30.0"), "sglang1": ("sglang", "0.5.21")})
        with open(os.path.join(a.out, f"proxy-engines-{stage}.txt"), "w") as f:
            f.write("\n".join(lines) + "\n")
    figs = {bid: engines.figures(kind, mapped["a"][bid], mapped["b"][bid], 15.0)
            for bid, kind in (("vllm1", "vllm"), ("sglang1", "sglang"))}
    with open(os.path.join(a.out, "proxy-engines-figures.json"), "w") as f:
        json.dump({"dt_s": 15.0, "figures": figs}, f, indent=1, sort_keys=True)
        f.write("\n")
    print("proxy-engines-{a,b}.txt, proxy-engines-figures.json")


if __name__ == "__main__":
    main()
