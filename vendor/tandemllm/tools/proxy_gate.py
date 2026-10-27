"""The proxy's integration gate: a live TandemLLM engine behind the standalone proxy.

    python tools/proxy_gate.py --url http://<engine host>:8000/v1 [--n 20] [--key ...] \
        [--out report.json]

Starts the proxy in this process (a temp config, ledger and key file), registers the engine as a
`tandemllm` backend, and sends `--n` small streamed requests with `X-Tandem: raw` (the engine's
plain path, no conductor), one at a time, greedy, thinking off. Each answer carries the engine's
own `usage` and `timings` on its finish chunk; the proxy's ledger row for the same request is
compared with them:

    prompt_tokens, completion_tokens   equal
    decode_tps                         the median of |proxy / engine - 1| within 2 % (--tol), and
                                       each request within 5 % (--tol-one) on its best of
                                       --repeats tries
    ttft_ms                            at or above the engine's ttft_ms (the network and the proxy
                                       are inside it); the difference is reported
    proxy_ms                           p50 under 2 ms

The decode check is relative and repeated because the proxy times the stream as it arrives: over
a LAN a late last chunk moves a 96-token request's rate by a few percent (NP1's issue 4: 1 and 4
of 20 failed a per-request 2 % from the Mac, 20 of 20 passed over loopback on the box). A request
that misses --tol-one is sent again, up to --repeats times in all, and its best try counts; the
gate's 2 % is on the median. The token counts, the status and the TTFT order hold on every try.
Over loopback (the engine on this host, as tools/proxy_e2e.sh runs it on the box) the jitter is
gone; the report says which it was.

Exit 0 when every check holds, 1 otherwise. The requests are small (a 60-token prompt, at most
96 tokens out) and read-only; nothing on the engine changes.
"""

from __future__ import annotations

import argparse
import http.client
import ipaddress
import json
import os
import socket
import statistics
import sys
import time
from urllib.parse import urlsplit

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

from proxy_util import Proxy  # noqa: E402

PROMPT = ("Write the numbers from one to thirty as English words, separated by commas, "
          "and nothing else.")


def _models(url: str, key: str | None) -> list[str]:
    u = urlsplit(url)
    c = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=10)
    c.request("GET", u.path.rstrip("/") + "/models",
              headers={"Authorization": f"Bearer {key}"} if key else {})
    r = c.getresponse()
    return [m["id"] for m in json.loads(r.read())["data"]]


def _finish_fields(stream: bytes) -> dict:
    """The engine's usage and timings from the chunk that carries them."""
    out = {}
    for ev in stream.split(b"\n\n"):
        if not ev.startswith(b"data: {"):
            continue
        obj = json.loads(ev[6:])
        if obj.get("usage"):
            out["usage"] = obj["usage"]
        if obj.get("timings"):
            out["timings"] = obj["timings"]
    return out


def loopback(url: str) -> bool:
    """The engine's host resolves to a loopback address (no network between it and the proxy)."""
    host = urlsplit(url).hostname or ""
    try:
        return all(ipaddress.ip_address(a[4][0]).is_loopback
                   for a in socket.getaddrinfo(host, None))
    except (OSError, ValueError):
        return False


def judge(rows: list, tol: float = 0.02, tol_one: float = 0.05) -> dict:
    """The gate's verdict over the rows (each with `fail`, the strict checks of its last try,
    and `decode_rel` of its best try): every strict check holds, each request's decode within
    `tol_one`, the median within `tol`, the proxy's p50 under 2 ms."""
    pms = [r["proxy"]["proxy_ms"] for r in rows if r["proxy"]["proxy_ms"] is not None]
    rel = [abs(r["decode_rel"]) for r in rows if "decode_rel" in r]
    dt = [r["ttft_delta_ms"] for r in rows if "ttft_delta_ms" in r]
    fails = [(r["i"], r["fail"]) for r in rows if r["fail"]]
    fails += [(r["i"], [f"decode {r['decode_rel']:+.2%} past {tol_one:.0%} in "
                        f"{r.get('tries', 1)} tries"])
              for r in rows if "decode_rel" in r and abs(r["decode_rel"]) > tol_one]
    med = statistics.median(rel) if rel else None
    summary = {"n": len(rows), "fails": len(fails),
               "proxy_ms_p50": statistics.median(pms) if pms else None,
               "proxy_ms_max": max(pms) if pms else None,
               "decode_rel_median": med,
               "decode_rel_max": max(rel) if rel else None,
               "retried": sum(1 for r in rows if r.get("tries", 1) > 1),
               "ttft_delta_ms_p50": statistics.median(dt) if dt else None,
               "ttft_delta_ms_max": max(dt) if dt else None}
    ok = (not fails and len(rel) == len(rows) and med is not None and med <= tol
          and summary["proxy_ms_p50"] is not None and summary["proxy_ms_p50"] < 2.0)
    summary["verdict"] = "PASS" if ok else "FAIL"
    summary["fail_list"] = fails
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="the engine's base URL, ending in /v1")
    ap.add_argument("--key", default=None, help="the engine's bearer, if it needs one")
    ap.add_argument("--model", default=None, help="default: the first id the engine lists")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--max-tokens", type=int, default=96)
    ap.add_argument("--tol", type=float, default=0.02,
                    help="the median request's decode rate against the engine's")
    ap.add_argument("--tol-one", type=float, default=0.05,
                    help="each request's decode rate on its best try")
    ap.add_argument("--repeats", type=int, default=3,
                    help="tries for a request whose decode misses --tol-one")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    lo = loopback(a.url)
    model = a.model or _models(a.url, a.key)[0]
    env = {"TANDEM_BACKENDS": "engine1", "TANDEM_BACKEND_ENGINE1_KIND": "tandemllm",
           "TANDEM_BACKEND_ENGINE1_URL": a.url}
    if a.key:
        env["TANDEM_BACKEND_ENGINE1_KEY"] = a.key
    rows = []
    with Proxy(env) as px:
        px.sync()
        listed = [m["id"] for m in json.loads(px.get("/v1/models")[2])["data"]]
        public = model if model in listed else f"engine1/{model}"
        print(f"[gate] engine {a.url} ({'loopback' if lo else 'over the network'}) model {model} "
              f"(public id {public}); {a.n} requests, raw, greedy, thinking off; decode median "
              f"within {a.tol:.0%}, each within {a.tol_one:.0%} on its best of {a.repeats}",
              flush=True)
        for i in range(a.n):
            tries, best = 0, None
            while True:
                rec = _one(px, public, i, a)
                tries += 1
                if rec["fail"]:
                    best = rec          # a strict check failed: it counts, no repeat
                    break
                if best is None or abs(rec["decode_rel"]) < abs(best["decode_rel"]):
                    best = rec
                if abs(rec["decode_rel"]) <= a.tol_one or tries >= max(1, a.repeats):
                    break
                print(f"[gate] {i:2d} try {tries}: decode {rec['decode_rel']:+.2%}, again",
                      flush=True)
            best["tries"] = tries
            rows.append(best)
            e, p = best["engine"], best["proxy"]
            why = best["fail"] + ([f"decode {best['decode_rel']:+.2%}"]
                                  if abs(best.get("decode_rel", 0)) > a.tol_one else [])
            print(f"[gate] {i:2d} tokens {p['prompt_tokens']}/{p['completion_tokens']} decode "
                  f"engine {e['decode_tps']} proxy {p['decode_tps']} "
                  f"({best.get('decode_rel', 0):+.2%}, try {best['tries']})"
                  f" ttft engine {e['ttft_ms']} proxy {p['ttft_ms']} proxy_ms {p['proxy_ms']}"
                  + (f"  FAIL {why}" if why else ""), flush=True)
    summary = judge(rows, a.tol, a.tol_one)
    summary["loopback"] = lo
    print(f"[gate] {json.dumps({k: v for k, v in summary.items() if k != 'fail_list'})}",
          flush=True)
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"summary": summary, "rows": rows, "at": time.time()}, f, indent=1)
    return 0 if summary["verdict"] == "PASS" else 1


def _one(px, public: str, i: int, a) -> dict:
    """One request through the proxy and its checks; `fail` holds the strict ones (status,
    X-Tandem, token counts, a decode rate, TTFT order), `decode_rel` the proxy's rate against the
    engine's."""
    body = {"model": public, "stream": True, "max_tokens": a.max_tokens,
            "temperature": 0, "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": f"{PROMPT} (run {i})"}]}
    st, hdr, data = px.post("/v1/chat/completions", body, headers={"X-Tandem": "raw"},
                            timeout=300)
    eng = _finish_fields(data)
    row = px.rows()[-1]
    t, u = eng.get("timings") or {}, eng.get("usage") or {}
    rec = {"i": i, "status": st, "x_tandem": hdr.get("X-Tandem"),
           "engine": {"prompt_tokens": u.get("prompt_tokens"),
                      "completion_tokens": u.get("completion_tokens"),
                      "decode_tps": t.get("predicted_per_second"),
                      "ttft_ms": t.get("ttft_ms"), "cache_n": t.get("cache_n")},
           "proxy": {k: row[k] for k in ("prompt_tokens", "completion_tokens",
                                         "decode_tps", "ttft_ms", "proxy_ms",
                                         "usage_source", "backend")}}
    e, p = rec["engine"], rec["proxy"]
    why = []
    if st != 200:
        why.append(f"status {st}")
    if hdr.get("X-Tandem") not in (None, "raw"):
        why.append(f"X-Tandem {hdr.get('X-Tandem')}")
    if (e["prompt_tokens"], e["completion_tokens"]) != (p["prompt_tokens"],
                                                         p["completion_tokens"]):
        why.append("token counts differ")
    if e["decode_tps"] and p["decode_tps"]:
        rec["decode_rel"] = p["decode_tps"] / e["decode_tps"] - 1
    else:
        why.append("no decode rate")
    if e["ttft_ms"] is not None and p["ttft_ms"] is not None:
        rec["ttft_delta_ms"] = p["ttft_ms"] - e["ttft_ms"]
        if rec["ttft_delta_ms"] < 0:
            why.append("proxy TTFT below the engine's")
    rec["fail"] = why
    return rec


if __name__ == "__main__":
    sys.exit(main())
