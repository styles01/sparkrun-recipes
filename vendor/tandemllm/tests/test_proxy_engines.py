"""The vLLM and SGLang adapters at L1 (proxy/engines.py, P2 and P3): the name tables on pages
shaped as each release sends them, the figures derived from two pages, the mapped page the proxy
serves, the poller (404, back-off, discovery), and the usage of each engine's stream.

The pages (tests/fixtures/proxy/*-metrics-*.txt) are written by tools/proxy_fixture_pages.py from
the families each release declares, rendered by prometheus_client; the two pages of each engine
are 15 s apart with a known workload, so every figure below is checked against a hand count.

Run: python tests/test_proxy_engines.py
"""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from proxy_util import Proxy, fake, run  # noqa: E402

from proxy import adapters as A  # noqa: E402
from proxy import engines as E  # noqa: E402
from proxy import scrape  # noqa: E402
from proxy.adapters.base import CHAT, Stamps  # noqa: E402
from tools import contract_check  # noqa: E402

FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "proxy")


def page(name: str) -> str:
    with open(os.path.join(FIX, name)) as f:
        return f.read()


def near(a, b, tol=1e-6):
    assert a is not None and abs(a - b) <= tol, (a, b)


def _vllm():
    return (E.map_page("vllm", page("vllm-metrics-0.30.0-a.txt")),
            E.map_page("vllm", page("vllm-metrics-0.30.0-b.txt")))


def _sglang():
    return (E.map_page("sglang", page("sglang-metrics-0.5.21-a.txt")),
            E.map_page("sglang", page("sglang-metrics-0.5.21-b.txt")))


# ----------------------------------------------------------------------------------- vLLM (P2)

def test_vllm_table_maps_every_family():
    a, b = _vllm()
    want = {fam.name for fam in E.VLLM} - {"corrupted_requests_total"}   # only with the NaN check
    assert want <= set(b), sorted(want - set(b))
    assert E.value(b, "requests_running") == 1 and E.value(b, "requests_waiting") == 0
    assert E.value(b, "kv_usage") == 0.27
    # request_success keeps its reason
    assert E.value(b, "requests_finished_total", reason="stop") == 43
    assert E.value(b, "requests_finished_total") == 46


def test_vllm_figures_by_hand():
    a, b = _vllm()
    f = E.figures("vllm", a, b, 15.0)
    near(f["prefix_hit_rate"], 0.8)                 # hits +800 over queries +1000
    near(f["cached_share"], 0.8)                    # prompt_tokens_cached +800 over prompt +1000
    near(f["decode_tps"], 40.0)                     # generation +600 in 15 s
    near(f["prompt_tps"], 66.67, 0.01)
    near(f["ttft_p50_s"], 0.175)                    # 0.09, 0.2, 0.3: rank 1.5 in (0.1, 0.25]
    near(f["queue_s"]["p50"], 0.65)                 # 0.2, 0.6, 0.9 on 0.3/0.5/0.8/1.0
    near(f["queue_s"]["p90"], 0.94)
    assert f["queue_s"]["count"] == 3
    assert f["prefill_s"]["count"] == 3 and f["decode_s"]["count"] == 3
    assert f["preemptions"] == 0 and f["finished"] == 3
    assert f["awake"] is True and f["running"] == 1
    s = f["spec"]
    assert s["source"] == "counters"
    near(s["acceptance"], 0.75)                     # accepted +450 over drafted +600
    near(s["tokens_per_step"], 3.25)                # 450 / 200 drafts + the sampled token
    assert s["per_position"] == [0.95, 0.75, 0.55]  # +190, +150, +110 over 200 drafts


def test_vllm_without_speculation():
    p = E.map_page("vllm", page("vllm-metrics-0.30.0-nospec.txt"))
    assert not any(k.startswith("spec_") for k in p)
    f = E.figures("vllm", p, p, 15.0)
    assert f["spec"] is None
    # nothing moved between the two readings: a window figure with no traffic is None, not 0
    assert f["prefix_hit_rate"] is None and f["ttft_p50_s"] is None and f["queue_s"] is None
    assert f["decode_tps"] == 0.0                   # tokens a second over the window: truly 0


def test_one_page_has_no_window_figures():
    _, b = _vllm()
    f = E.figures("vllm", None, b, None)
    assert f["running"] == 1 and f["kv_usage"] == 0.27
    assert f["decode_tps"] is None and f["prefix_hit_rate"] is None and f["ttft_p50_s"] is None
    assert f["spec"]["acceptance"] is None


# ----------------------------------------------------------------------------------- SGLang (P3)

def test_sglang_ranks_counted_once():
    a, b = _sglang()
    # the page has tp_rank 0 and 1 for every scheduler gauge: only rank 0 counts
    assert E.value(b, "requests_running") == 2 and E.value(a, "requests_running") == 3
    assert E.value(b, "kv_tokens") == 40000 and E.value(b, "kv_capacity_tokens") == 100000
    assert E.value(b, "gen_tps") == 118.5


def test_sglang_figures_by_hand():
    a, b = _sglang()
    f = E.figures("sglang", a, b, 15.0)
    near(f["prefix_hit_rate"], 0.7)                 # cached +840 over prompt +1200
    near(f["decode_tps"], 120.0)                    # generation +1800 in 15 s (both stream kinds)
    assert f["engine_gen_tps"] == 118.5 and f["engine_cache_hit_rate"] == 0.61
    near(f["ttft_p50_s"], 0.6)                      # 0.5, 0.9 on 0.4/0.6/0.8/1
    near(f["queue_s"]["p50"], 0.5)                  # 0.3, 1.5 on 0.2/0.5/1/2
    near(f["queue_s"]["p90"], 1.8)
    near(f["e2e_s"]["p50"], 3.0)                    # 3.5, 3.9 inside (2, 4]
    assert f["prefill_s"] is None and f["decode_s"] is None   # SGLang has no such histograms
    assert f["finished"] == 2 and f["preemptions"] == 0
    assert f["spec"] == {"source": "engine_window", "acceptance": 0.7, "tokens_per_step": 3.1,
                         "per_position": None, "drafts": None}


def test_sglang_counters_born_after_the_first_reading():
    """SGLang v0.5.21 creates its scheduler and tokenizer families at their first request: the
    page of a fresh server (the night pass's real one, sglang-metrics-0.5.21-fresh.txt) has none
    of them. A counter missing from the window's first reading and present in its last started
    at 0 in between; before, the window's rates stayed None for 5 minutes after a restart."""
    fresh = E.map_page("sglang", page("sglang-metrics-0.5.21-fresh.txt"))
    assert "generation_tokens_total" not in fresh and "requests_finished_total" not in fresh
    _, b = _sglang()
    f = E.figures("sglang", fresh, b, 15.0)
    near(f["decode_tps"], round(E.value(b, "generation_tokens_total") / 15.0, 2))
    near(f["prompt_tps"], round(E.value(b, "prompt_tokens_total") / 15.0, 2))
    assert f["finished"] == E.value(b, "requests_finished_total")
    near(f["prefix_hit_rate"], round(E.value(b, "prompt_tokens_cached_total")
         / E.value(b, "prompt_tokens_total"), 4))
    # a family missing from the last reading stays None (the page cannot give it)
    g = E.figures("sglang", b, fresh, 15.0)
    assert g["decode_tps"] is None and g["finished"] is None


# ----------------------------------------------------------------------------------- the page

def test_mapped_page_is_prometheus_text():
    a, b = _vllm()
    _, s = _sglang()
    lines = E.render([("vllm1", b), ("sglang1", s)])
    text = "\n".join(lines) + "\n"
    got = scrape.parse(text)
    run_ = [(n, lab, v) for n, lab, v in got if n == "qse_backend_requests_running"]
    assert ("qse_backend_requests_running", {"backend": "vllm1"}, 1.0) in run_
    assert ("qse_backend_requests_running", {"backend": "sglang1"}, 2.0) in run_
    # one HELP and TYPE per family, whichever backend has it
    assert text.count("# TYPE qse_backend_requests_running ") == 1
    # never an engine family: the engine's own tiles must not see a backend
    assert not any(n.startswith(("qse_requests_", "qse_time_to_first")) for n, _, _ in got)
    # a strict reader agrees (prometheus_client's parser, when it is installed)
    try:
        from prometheus_client.parser import text_string_to_metric_families
    except ImportError:
        return
    fams = {f.name: f for f in text_string_to_metric_families(text)}
    assert fams["qse_backend_ttft_seconds"].type == "histogram"
    assert fams["qse_backend_spec_accepted_tokens_per_pos"].type == "counter"


def test_scrape_cost():
    text = page("vllm-metrics-0.30.0-b.txt")
    n = 50
    t0 = time.perf_counter()
    for _ in range(n):
        E.map_page("vllm", text)
    per = (time.perf_counter() - t0) / n * 1e3
    lines = len(text.splitlines())
    # the gate: a 300-line page under 2 ms; this page has about 500 lines
    assert per * 300 / lines < 2.0, f"{per:.2f} ms for {lines} lines"
    print(f"  map_page: {per:.2f} ms for a {lines}-line page")


def test_unknown_families_and_garbage_are_ignored():
    p = E.map_page("vllm", "vllm:new_family_2027_total{engine=\"0\"} 5\nnot a line\n"
                           "vllm:num_requests_running{engine=\"0\"} 2\n")
    assert set(p) == {"requests_running"}
    assert E.map_page("generic", page("vllm-metrics-0.30.0-b.txt")) == {}


# ----------------------------------------------------------------------------------- the poller

class _Cfg:
    def __init__(self, backends):
        self._b = backends

    def backends(self):
        return self._b

    def key_for(self, bid):
        return "sk-fixture-ABCDEFGH1234" if bid == "vllm1" else None


class _Svc:
    def __init__(self, backends):
        self.cfg = _Cfg(backends)
        self.lines = []

    def log(self, msg):
        self.lines.append(msg)


def _backend(bid, kind, url):
    return {"id": bid, "kind": kind, "url": url, "enabled": True}


def test_watch_reads_discovers_and_backs_off():
    clock = [1000.0]
    calls = []
    answers = {
        "http://v:8000/metrics": [(200, page("vllm-metrics-0.30.0-a.txt").encode()),
                                  (200, page("vllm-metrics-0.30.0-b.txt").encode())],
        "http://v:8000/version": [(200, b'{"version": "0.30.0"}')],
        "http://s:30000/metrics": [(404, b"")],
        "http://s:30000/server_info": [(404, b"")],
        "http://s:30000/get_server_info": [(200, json.dumps({
            "version": "0.5.21", "speculative_algorithm": "EAGLE3", "speculative_num_steps": 3,
            "speculative_num_draft_tokens": 4, "context_length": 32768,
            "enable_cache_report": False, "reasoning_parser": "qwen3",
            "enable_metrics": False}).encode())],
    }

    def fetch(url, key, timeout):
        calls.append((url, key))
        q = answers[url]
        return q.pop(0) if len(q) > 1 else q[0]

    svc = _Svc([_backend("vllm1", "vllm", "http://v:8000/v1"),
                _backend("sglang1", "sglang", "http://s:30000/v1"),
                _backend("ds", "deepseek", "https://api.deepseek.com/v1")])
    w = E.Watch(svc, clock=lambda: clock[0], fetch=fetch)
    w.tick()
    v = w.view("vllm1", "vllm")
    assert v["metrics"] == "ok" and v["version"] == "0.30.0" and v["verified"] == "0.30.0"
    assert v["figures"]["running"] == 2 and v["figures"]["decode_tps"] is None  # one page yet
    s = w.view("sglang1", "sglang")
    assert s["metrics"] == "off" and s["figures"] is None
    assert s["info"]["speculative"]["algorithm"] == "EAGLE3"
    assert s["info"]["cache_report"] is False and s["info"]["metrics_enabled"] is False
    assert any("--enable-metrics" in x for x in svc.lines)
    assert w.view("ds", "deepseek") is None
    # the key goes to the backend's own routes and nowhere else; never in a log line
    assert all(k == "sk-fixture-ABCDEFGH1234" for u, k in calls if u.startswith("http://v:"))
    assert not any("sk-fixture" in x for x in svc.lines)
    # 15 s later: the second page gives the window
    clock[0] += 15.0
    w.tick()
    v = w.view("vllm1", "vllm")
    near(v["figures"]["prefix_hit_rate"], 0.8)
    assert v["figures"]["window_s"] == 15.0 and v["scrape_age_s"] == 0.0
    # SGLang's 404 is asked again after 120 s, not every tick
    n404 = sum(1 for u, _ in calls if u == "http://s:30000/metrics")
    clock[0] += 60.0
    w.tick()
    assert sum(1 for u, _ in calls if u == "http://s:30000/metrics") == n404
    clock[0] += 61.0
    w.tick()
    assert sum(1 for u, _ in calls if u == "http://s:30000/metrics") == n404 + 1


def test_watch_error_backoff_and_recovery():
    clock = [0.0]
    state = {"fail": True}

    def fetch(url, key, timeout):
        if url.endswith("/version"):
            return 200, b'{"version": "0.30.0"}'
        if state["fail"]:
            raise ConnectionRefusedError("refused")
        return 200, page("vllm-metrics-0.30.0-a.txt").encode()

    svc = _Svc([_backend("vllm1", "vllm", "http://v:8000/v1")])
    w = E.Watch(svc, clock=lambda: clock[0], fetch=fetch)
    w.tick()
    assert w.view("vllm1", "vllm")["metrics"] == "error"
    due1 = w.st["vllm1"].due
    assert due1 == 30.0                             # 15 s doubled after the first failure
    clock[0] = due1
    w.tick()
    assert w.st["vllm1"].due - clock[0] == 60.0
    state["fail"] = False
    clock[0] = w.st["vllm1"].due
    w.tick()
    v = w.view("vllm1", "vllm")
    assert v["metrics"] == "ok" and v["error"] is None and w.st["vllm1"].due - clock[0] == 15.0
    # a deleted backend leaves the poller
    svc.cfg._b = []
    w.tick()
    assert "vllm1" not in w.st


def test_root_of():
    assert E.root_of("http://h:8000/v1") == "http://h:8000"
    assert E.root_of("http://h:8000/v1/") == "http://h:8000"
    assert E.root_of("http://h:8000") == "http://h:8000"


# ----------------------------------------------------------------------------------- per request

def _fx():
    with open(os.path.join(FIX, "engine-frames.json")) as f:
        return json.load(f)


def _usage(kind, events):
    ad = A.make({"id": "b", "kind": kind, "url": "http://x/v1"})
    st = Stamps()
    for e in events:
        ad.read_frame(CHAT, e.encode(), st, True)
    return ad.reconcile(ad.read_usage(st.usage, CHAT), st), st


def test_vllm_stream_usage():
    fx = _fx()
    u, st = _usage("vllm", fx["vllm_stream_details"])
    assert st.reasoning                             # v0.30.0 streams `delta.reasoning`
    assert u == {"prompt_tokens": 1040, "completion_tokens": 12, "cached_tokens": 1024,
                 "cache_write_tokens": None, "reasoning_tokens": 9}
    ad = A.make({"id": "b", "kind": "vllm", "url": "http://x/v1"})
    u = ad.read_usage(fx["vllm_usage_no_flags"], CHAT)
    assert u["cached_tokens"] is None and u["reasoning_tokens"] is None   # no flags: no counts


def test_sglang_stream_usage():
    fx = _fx()
    u, _ = _usage("sglang", fx["sglang_stream_counted"])
    assert u["reasoning_tokens"] == 508 and u["cached_tokens"] == 640
    # reasoning streamed but not counted (no reasoning parser): SGLang's 0 is not a measurement
    u, st = _usage("sglang", fx["sglang_stream_uncounted"])
    assert st.reasoning and u["reasoning_tokens"] is None and u["cached_tokens"] is None
    # no reasoning at all: the 0 stands
    u, st = _usage("sglang", fx["sglang_stream_no_reasoning"])
    assert not st.reasoning and u["reasoning_tokens"] == 0


def test_kinds_and_capabilities():
    v = A.make({"id": "b", "kind": "vllm", "url": "http://x/v1"})
    s = A.make({"id": "b", "kind": "sglang", "url": "http://x/v1"})
    assert v.kind == "vllm" and s.kind == "sglang"
    assert v.capabilities()["continue_final_message"] and s.capabilities()["choice"] == "regex"
    from proxy import config as C
    assert not C.KINDS["vllm"]["cloud"] and not C.KINDS["sglang"]["cloud"]


# ----------------------------------------------------------------------------------- end to end

def test_proxy_serves_the_engines():
    """The standalone proxy in front of a fake vLLM, a fake SGLang and an SGLang without
    --enable-metrics: the mapped families on its /metrics, the engine block in the system and
    summary bodies (valid against the contract), the Backends API, and requests relayed."""
    sv, pv = fake(models=["Qwen/Qwen3.5-0.8B"], dialect="vllm")
    ss, ps = fake(models=["qwen-s"], dialect="sglang")
    so, po = fake(models=["qwen-off"], dialect="sglang-off")
    env = {"TANDEM_BACKENDS": "vllm1,sglang1,sgoff"}
    for bid, kind, port in (("vllm1", "vllm", pv), ("sglang1", "sglang", ps),
                            ("sgoff", "sglang", po)):
        env[f"TANDEM_BACKEND_{bid.upper()}_KIND"] = kind
        env[f"TANDEM_BACKEND_{bid.upper()}_URL"] = f"http://127.0.0.1:{port}/v1"
    with Proxy(env) as px:
        px.sync()
        w = px.svc.engines
        w.tick()
        for model in ("Qwen/Qwen3.5-0.8B", "qwen-s"):
            st, _, body = px.post("/v1/chat/completions", {
                "model": model, "messages": [{"role": "user", "content": "hi"}]})
            assert st == 200, body
        for s in w.st.values():
            s.due = 0.0
        time.sleep(0.01)
        w.tick()
        _, _, page_ = px.get("/metrics")
        text = page_.decode()
        assert 'qse_backend_kv_usage{backend="vllm1"} 0.12' in text
        assert 'qse_backend_prefix_cache_hits_total{backend="vllm1"} 40' in text, text[:3000]
        assert 'qse_backend_cache_hit_rate{backend="sglang1"} 0.4' in text
        assert 'qse_backend_requests_running{backend="sgoff"}' not in text
        assert 'qse_backend_scrape_age_seconds{backend="vllm1"}' in text
        assert 'qse_backend_info{backend="vllm1",kind="vllm",version="0.30.0"} 1' in text
        assert 'qse_backend_info{backend="sglang1",kind="sglang",version="0.5.21"} 1' in text
        sysb = json.loads(px.get("/v1/dashboard/system")[2])
        rows = {b["id"]: b for b in sysb["proxy"]["backends"]}
        v = rows["vllm1"]["engine"]
        assert v["metrics"] == "ok" and v["version"] == "0.30.0"
        near(v["figures"]["prefix_hit_rate"], 0.4)       # hits +80 over queries +200
        assert rows["sglang1"]["engine"]["metrics"] == "ok"
        near(rows["sglang1"]["engine"]["figures"]["prefix_hit_rate"], 0.4)
        off = rows["sgoff"]["engine"]
        assert off["metrics"] == "off" and off["info"]["metrics_enabled"] is False
        assert rows["vllm1"]["scrape_age_s"] is not None
        for name, path in (("system", "/v1/dashboard/system"),
                           ("summary", "/v1/dashboard/summary")):
            doc = json.loads(px.get(path)[2])
            errs = contract_check.validate(doc, contract_check.load(name))
            assert not errs, (name, errs[:3])
        cfg = json.loads(px.get("/v1/config/backends/vllm1")[2])
        assert cfg["engine"]["metrics"] == "ok" and cfg["kind"] == "vllm"
        assert cfg["capabilities"]["continue_final_message"] is True
        r = px.rows()[-1]
        assert r["backend"] == "sglang1" and r["backend_kind"] == "sglang"
    for srv in (sv, ss, so):
        srv.shutdown()


if __name__ == "__main__":
    run(globals())
