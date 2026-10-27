"""The L1 adapters (P2, P3) against a real vLLM or SGLang server: one pass, minutes, for the night.

    python tools/proxy_engines_live.py --kind vllm --url http://127.0.0.1:8101/v1 \
        --model Qwen/Qwen3.5-0.8B --out results/engines/vllm [--spec]

The server is started by tools/proxy_engines_check.sh (a small model, beside the live engine); this
script only talks to it, through a standalone proxy in this process and directly:

    discovery  vLLM `/version`, SGLang `/server_info`: the version is recorded, and the fixture
               names in tests/fixtures/proxy name the release the tables were checked against
    page       the engine's own /metrics saved as `<kind>-metrics-<version>-live.txt` (the real
               page beside the generated fixtures); every family of the table the release has,
               missing ones listed
    relay      12 requests through the proxy (streamed and not, three sharing a long prefix,
               thinking on), then two reads of the page: the mapped families equal the engine's
               own page read at the same moment (counters, gauges, histogram counts), the
               window's prefix-hit rate is above 0 after the shared prefix, and the ledger rows
               carry the per-request counts the flags promise (cached tokens with
               --enable-prompt-tokens-details or --enable-cache-report, reasoning tokens with
               --reasoning-parser)
    choice     the class question's constraint: vLLM `structured_outputs.choice`, SGLang `regex`;
               the answer is one of the options
    ids        vLLM: a request with `X-Request-Id: <id>` answers with `chatcmpl-<id>`
    spec       with --spec (the server runs n-gram speculation): the spec families exist and the
               acceptance is a number between 0 and 1
    plugin     with --plugin (P6: the server runs tandemllm-vllm-plugin or tandemllm-sglang-plugin,
               posting to this script's proxy on --proxy-port with TANDEM_COLLECT_TOKEN): the
               plugin's batches arrive (the backend reads L2, with the plugin's and the engine's
               versions); every one of the 12 rows gets the engine's split by an exact join; and
               the plugin's per-request sums over the 12 requests equal the engine's own /metrics
               over the same window within 1 % (request count, prompt and generation tokens, and
               the queue, prefill and decode time sums on vLLM, the end-to-end sum on SGLang)

Writes `<out>/report.json` and exits non-zero when a check fails. No paid call, no GPU work of its
own: the server does the work, at most 12 short requests.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "tests")]

from proxy import engines  # noqa: E402
from proxy import scrape  # noqa: E402

NEED = {
    "vllm": ("requests_running", "requests_waiting", "kv_usage", "prompt_tokens_total",
             "generation_tokens_total", "prefix_cache_queries_total", "prefix_cache_hits_total",
             "requests_finished_total", "ttft_seconds", "queue_seconds", "prefill_seconds",
             "decode_seconds", "e2e_seconds"),
    "sglang": ("requests_running", "requests_waiting", "kv_usage", "prompt_tokens_total",
               "generation_tokens_total", "prompt_tokens_cached_total", "requests_finished_total",
               "ttft_seconds", "e2e_seconds", "cache_hit_rate", "gen_tps"),
}
SPEC = {"vllm": ("spec_drafts_total", "spec_draft_tokens_total", "spec_accepted_tokens_total"),
        "sglang": ("spec_accept_rate", "spec_accept_length")}


def http(url, body=None, headers=None, timeout=120):
    data = None if body is None else json.dumps(body).encode()
    h = {"Content-Type": "application/json"}
    h.update(headers or {})
    req = urllib.request.Request(url, data=data, headers=h, method="POST" if data else "GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


class Report:
    def __init__(self):
        self.checks = []

    def check(self, name, ok, **detail):
        self.checks.append({"check": name, "ok": bool(ok), **detail})
        print(("PASS " if ok else "FAIL ") + name + (f"  {detail}" if detail else ""), flush=True)
        return ok

    @property
    def failed(self):
        return [c for c in self.checks if not c["ok"]]


def plugin_checks(rep, px, bid, kind, root, page_before, n_before) -> None:
    """P6: the rows joined, and the plugin's sums against the engine's own page."""
    col = px.svc.collect
    t_end = time.time() + 30
    rows = []
    while time.time() < t_end:
        rows = [r for r in px.rows() if r["backend"] == bid]
        if rows and all(r["l2_join"] for r in rows):
            break
        time.sleep(1.0)
    joins = [r["l2_join"] for r in rows]
    split = [(r["queue_ms"], r["prompt_ms"], r["decode_ms"]) for r in rows]
    rep.check("rows joined (exact)", rows and all(j == "exact" for j in joins), joins=joins,
              split=split[:4])
    b = col.backends.get(bid)
    items = list(b.recent)[n_before:] if b else []
    st, _, after = http(root + "/metrics")
    page_after = engines.map_page(kind, after.decode("utf-8", "replace"))

    def delta(name):
        a0, a1 = engines.value(page_before, name), engines.value(page_after, name)
        return None if a1 is None else a1 - (a0 or 0)

    def hsum(name):
        d = engines.hist_delta(engines.histogram(page_before, name),
                               engines.histogram(page_after, name))
        return None if d is None else d["sum"] * 1e3

    def total(k):
        return sum(it.get(k) or 0 for it in items)
    pairs = [("requests", len(items), delta("requests_finished_total")),
             ("prompt tokens", total("prompt_tokens"), delta("prompt_tokens_total")),
             ("generation tokens", total("completion_tokens"), delta("generation_tokens_total"))]
    if kind == "vllm":
        pairs += [("queue ms", total("queue_ms"), hsum("queue_seconds")),
                  ("prefill ms", total("prefill_ms"), hsum("prefill_seconds")),
                  ("decode ms", total("decode_ms"), hsum("decode_seconds"))]
    else:
        pairs += [("e2e ms", total("e2e_ms"), hsum("e2e_seconds"))]
    out = []
    ok = True
    for name, mine, theirs in pairs:
        rel = (None if not theirs else abs(mine - theirs) / theirs)
        # the plugin rounds each request's times to 0.01 ms (_push.ms): a sum of n of them may
        # be n x 0.005 ms off, which is far over 1 % of a near-zero total such as vLLM's queue
        slack = 0.005 * len(items) if name.endswith(" ms") else 0.0
        good = theirs is not None and (rel is not None and rel <= 0.01 or mine == theirs
                                       or abs(mine - theirs) <= slack)
        ok &= bool(good)
        out.append({"figure": name, "plugin": round(mine, 3), "engine": theirs,
                    "rel": None if rel is None else round(rel, 5)})
    rep.check("plugin sums equal the engine's page (1 %)", ok and len(items) > 0, sums=out,
              items=len(items))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--kind", choices=("vllm", "sglang"), required=True)
    ap.add_argument("--url", required=True, help="the server's base URL with /v1")
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--spec", action="store_true", help="the server runs with speculation")
    ap.add_argument("--plugin", action="store_true",
                    help="the server runs TandemLLM's plugin, posting to --proxy-port")
    ap.add_argument("--proxy-port", type=int, default=0,
                    help="the port of this script's proxy (the plugin's TANDEM_COLLECT_URL)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rep = Report()
    root = engines.root_of(a.url)
    kind = a.kind

    # ---- discovery
    path = "/version" if kind == "vllm" else "/server_info"
    st, _, body = http(root + path)
    info = engines.parse_info(kind, json.loads(body) if st == 200 else None)
    version = info["version"] or "unknown"
    rep.check("discovery", st == 200 and info["version"], version=version,
              verified=engines.VERIFIED[kind], info=info)

    # ---- the page
    st, _, page0 = http(root + "/metrics")
    text0 = page0.decode("utf-8", "replace")
    with open(os.path.join(a.out, f"{kind}-metrics-{version}-live.txt"), "w") as f:
        f.write(text0)
    page0_ok = st == 200

    # ---- through a standalone proxy in this process
    from proxy_util import Proxy                                 # tests/proxy_util.py
    bid = kind + "1"
    env = {"TANDEM_BACKENDS": bid, f"TANDEM_BACKEND_{bid.upper()}_KIND": kind,
           f"TANDEM_BACKEND_{bid.upper()}_URL": a.url}
    if a.plugin:
        env["TANDEM_COLLECT_TOKEN"] = os.environ.get("TANDEM_COLLECT_TOKEN", "")
    with Proxy(env, port=a.proxy_port) as px:
        px.sync()
        w = px.svc.engines
        w.tick()
        if a.plugin:                       # the plugin backs off while no proxy listened
            t_end = time.time() + 60
            while time.time() < t_end and px.svc.collect.view(bid) is None:
                time.sleep(1.0)
            pv = px.svc.collect.view(bid)
            rep.check("plugin posts", pv is not None and pv["state"] == "l2", plugin=pv)
            # the plugin posts once a second: a request that finished before this point (SGLang's
            # own warm-up at start) must be in `recent` before n_before is read, or the sums
            # count it on the plugin's side only (13 against 12 on 2026-10-04)
            time.sleep(3.0)
            st, _, before = http(root + "/metrics")
            page_before = engines.map_page(kind, before.decode("utf-8", "replace"))
            n_before = len(px.svc.collect.backends[bid].recent) if pv else 0
        # about 1,600 tokens: more than two of vLLM's attention blocks for a hybrid model (560
        # tokens on Qwen3.5-0.8B, v0.30.0), so the prefix cache has whole blocks to reuse; at
        # 480 tokens nothing was cached and the cached-token and hit-rate checks read 0
        prefix = ("You review code. " * 400).strip()
        msgs = []
        for i in range(12):
            m = [{"role": "system", "content": prefix},
                 {"role": "user", "content": f"Say one short word about the number {i}."}]
            body = {"model": a.model, "messages": m, "max_tokens": 48, "temperature": 0,
                    "stream": i % 2 == 0}
            if i >= 9:
                body["chat_template_kwargs"] = {"enable_thinking": True}
                body["max_tokens"] = 160
            t0 = time.time()
            stt, _, out = px.post("/v1/chat/completions", body, timeout=180)
            msgs.append({"i": i, "status": stt, "s": round(time.time() - t0, 2)})
        rep.check("relay", all(x["status"] == 200 for x in msgs), requests=msgs)
        if a.plugin:
            plugin_checks(rep, px, bid, kind, root, page_before, n_before)
        rows = [r for r in px.rows() if r["backend"] == bid]
        cached = [r["cached_tokens"] for r in rows]
        reasoning = [r["reasoning_tokens"] for r in rows[9:]]
        rep.check("per-request cached tokens", any(c for c in cached if c), cached=cached,
                  note="vLLM: --enable-prompt-tokens-details; SGLang: --enable-cache-report")
        rep.check("per-request reasoning tokens", any(r for r in reasoning if r),
                  reasoning=reasoning, note="--reasoning-parser qwen3, thinking on")
        # two reads of both pages at the same moment, no traffic in between
        time.sleep(2.0)
        for s in w.st.values():
            s.due = 0.0
        w.tick()
        st, _, own = http(root + "/metrics")
        own_mapped = engines.map_page(kind, own.decode("utf-8", "replace"))
        _, _, proxy_page = px.get("/metrics")
        served = scrape.parse(proxy_page.decode())
        diffs = []
        for name, fam in own_mapped.items():
            if fam["kind"] == "histogram":
                mine = engines.histogram(own_mapped, name)["count"]
                got = sum(v for n, lab, v in served if n == f"qse_backend_{name}_count"
                          and lab.get("backend") == bid)
            else:
                mine = engines.value(own_mapped, name)
                got = sum(v for n, lab, v in served if n == f"qse_backend_{name}"
                          and lab.get("backend") == bid)
            # gauges may move between the two reads; counters and counts must not
            if fam["kind"] != "gauge" and abs((mine or 0) - got) > 1e-6:
                diffs.append({"family": name, "engine": mine, "proxy": got})
        rep.check("proxy page equals the engine's", not diffs, diffs=diffs[:10])
        # the families after traffic: SGLang v0.5.21 creates its scheduler and tokenizer
        # families at the first request, so a fresh server's page (saved above) lacks them
        missing = [n for n in NEED[kind] if n not in own_mapped]
        rep.check("page families", page0_ok and st == 200 and not missing, missing=missing,
                  mapped=sorted(own_mapped), lines=len(own.splitlines()),
                  fresh_page_families=sorted(engines.map_page(kind, text0)))
        view = w.view(bid, kind)
        figs = view["figures"] or {}
        rep.check("window figures", (figs.get("prefix_hit_rate") or 0) > 0
                  and figs.get("ttft_p50_s") is not None and view["metrics"] == "ok",
                  figures=figs)
        if a.spec:
            have = [n for n in SPEC[kind] if n in own_mapped]
            acc = (figs.get("spec") or {}).get("acceptance")
            rep.check("speculation families", len(have) == len(SPEC[kind])
                      and acc is not None and 0 <= acc <= 1, have=have, acceptance=acc)

    # ---- the class question's constraint, straight to the server
    q = {"model": a.model, "max_tokens": 8, "temperature": 0,
         "chat_template_kwargs": {"enable_thinking": False},
         "messages": [{"role": "user", "content": "Is water wet? Answer A for yes, B for no."}]}
    if kind == "vllm":
        q["structured_outputs"] = {"choice": ["A", "B"]}
    else:
        q["regex"] = "(A|B)"
    st, _, out = http(a.url.rstrip("/") + "/chat/completions", q)
    ans = None
    if st == 200:
        ans = json.loads(out)["choices"][0]["message"].get("content")
    rep.check("choice constraint", st == 200 and (ans or "").strip() in ("A", "B"), answer=ans)

    # ---- vLLM takes X-Request-Id as the request's id
    if kind == "vllm":
        rid = f"l1-check-{int(time.time())}"
        st, h, out = http(a.url.rstrip("/") + "/chat/completions",
                          {"model": a.model, "max_tokens": 4,
                           "messages": [{"role": "user", "content": "hi"}]},
                          headers={"X-Request-Id": rid})
        got = json.loads(out).get("id") if st == 200 else None
        rep.check("X-Request-Id adopted", got == f"chatcmpl-{rid}", id=got)

    with open(os.path.join(a.out, "report.json"), "w") as f:
        json.dump({"kind": kind, "version": version, "verified": engines.VERIFIED[kind],
                   "url": a.url, "model": a.model, "checks": rep.checks}, f, indent=1)
    print(f"{len(rep.checks) - len(rep.failed)} of {len(rep.checks)} checks passed")
    return 1 if rep.failed else 0


if __name__ == "__main__":
    sys.exit(main())
