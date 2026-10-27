"""The 262k memory gate (OPS-47, gate c): the engine's steady state with every cache full, and its
peak during the longest prefill the window allows, against MEM_CEILING_GIB and MEM_PEAK_GIB.

Against a running server (in a hold: it is one engine on :8011 with the served flags):

    python tools/mem_gate.py --base http://127.0.0.1:8011 --label cand --out results/mem_gate

1. idle after the load: the allocator (/health `memory`) and the memory policy's report
   (/v1/cache/stats `memory`);
2. the caches filled the way an opencode night fills them, every request greedy and short:
   session A (a long head and a long body), a title call between two turns (the guest stash),
   session B with the same head (A's rows parked), A again (swapped back, B parked), A's
   compaction (A's head pinned), the turn after it, and a small chat (state store entries);
3. the steady state between requests: reserved against the ceiling, each cache against its cap;
4. one prompt of about 262,000 tokens (the window less the answer): the allocator's peak over its
   forwards (`memory.last_prefill_peak`) against the declared peak, MemAvailable's minimum while
   it ran, and whether the engine shed anything or answered 503;
5. idle again, and two short requests after it: what the first one pays for a reserve that went
   back to the driver at idle (time to first token and decode rate against the second).

Token ids come from the long-prompt set (~/pub-bench/longprompts, `ids-*.npy`), decoded to text
with the served tokenizer: nothing here is anyone's conversation. The report is one JSON file.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.longctx_probe import MemSampler  # noqa: E402

GIB = 1 << 30


def call(base: str, path: str, body: dict | None = None, headers: dict | None = None,
         timeout: float = 1800.0):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base + path, data=data, method="POST" if body else "GET")
    req.add_header("Content-Type", "application/json")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            out = json.loads(r.read() or b"{}")
            return r.status, out, time.perf_counter() - t0
    except urllib.error.HTTPError as e:
        try:
            out = json.loads(e.read() or b"{}")
        except ValueError:
            out = {}
        return e.code, out, time.perf_counter() - t0


def reading(base: str) -> dict:
    """The allocator and the memory policy, between requests."""
    _, health, _ = call(base, "/health")
    _, stats, _ = call(base, "/v1/cache/stats")
    mem = (health or {}).get("memory") or {}
    pol = (stats or {}).get("memory") or {}
    res = (stats or {}).get("resident") or {}
    store = (stats or {}).get("state_store") or {}
    g = lambda b: None if b is None else round(b / GIB, 3)  # noqa: E731
    return {"allocated_gib": mem.get("allocated_gb"), "reserved_gib": mem.get("reserved_gb"),
            "engine_reserved_gib": mem.get("engine_reserved_gb"),
            "max_allocated_gib": mem.get("max_allocated_gb"),
            "ceiling_gib": mem.get("ceiling_gb"), "peak_gib": mem.get("peak_gb"),
            "reserve_gib": g(pol.get("reserve")),
            "caches_gib": {"anchors": g(res.get("anchor_pool_bytes")), "stash": g(res.get("stash_bytes")),
                           "pin": g(res.get("pin_bytes")), "park": g(res.get("park_bytes")),
                           "store": g(store.get("bytes"))},
            "prefill_peak_gib": g(pol.get("prefill_peak")),
            "last_prefill_peak": pol.get("last_prefill_peak"), "last_gate": pol.get("last_gate"),
            "shed": pol.get("shed"), "refused": pol.get("refused"), "empties": pol.get("empties"),
            "last_empty_ms": pol.get("last_empty_ms"),
            "events": [e for e in pol.get("events") or [] if e.get("kind") != "admit_refused"],
            "resident": {k: res.get(k) for k in ("valid", "owner", "park_owner", "park_rows",
                                                  "pin_rows", "parks", "park_hits", "park_swaps",
                                                  "pins", "pin_hits", "switches", "last_switch",
                                                  "refused_anchor", "refused_stash",
                                                  "refused_pin", "refused_park")}}


def chat(base: str, model: str, system: str | None, user: str, session: str, max_tokens: int = 8):
    msgs = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": user}]
    body = {"model": model, "messages": msgs, "max_tokens": max_tokens, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}}
    st, out, s = call(base, "/v1/chat/completions", body, {"x-session-id": session,
                                                             "X-Tandem": "raw"})
    u = (out or {}).get("usage") or {}
    t = (out or {}).get("timings") or {}
    return {"status": st, "s": round(s, 2), "prompt_tokens": u.get("prompt_tokens"),
            "cached": t.get("cache_n", t.get("prompt_cached")), "timings": t,
            "error": (out or {}).get("error")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="http://127.0.0.1:8011")
    ap.add_argument("--model", default="qwen38-spark-engine")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", default="results/mem_gate")
    ap.add_argument("--data", default=os.path.expanduser("~/pub-bench/longprompts"))
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.8-27B",
                    help="the served tokenizer (tools/longctx_probe.py's default)")
    ap.add_argument("--long-tokens", type=int, default=261_600,
                    help="the long prompt's size in tokens of the set (the chat template adds a few)")
    ap.add_argument("--skip-long", action="store_true")
    a = ap.parse_args()

    import numpy as np
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    ids190 = np.load(os.path.join(a.data, "ids-190000.npy"))[0].tolist()
    ids131 = np.load(os.path.join(a.data, "ids-131072.npy"))
    ids65 = np.load(os.path.join(a.data, "ids-65536.npy"))
    dec = lambda x: tok.decode(list(map(int, x)))  # noqa: E731
    head = dec(ids65[2][:8192])                  # code: the tool block and system text
    body_a = dec(ids65[0][:40000])               # prose
    body_b = dec(ids65[1][:20000])               # German prose
    report = {"label": a.label, "base": a.base, "t": time.strftime("%Y-%m-%d %H:%M:%S")}
    report["idle_after_load"] = reading(a.base)
    steps = []
    t_fill = time.perf_counter()
    for name, sysm, user, sess in (
            ("A1", head, body_a, "gate-A"),
            ("title", None, dec(ids131[0][:1500]) + "\n\nA title for this, five words.", "gate-T"),
            ("A2", head, body_a + "\n\n" + dec(ids131[1][:3000]), "gate-A"),
            ("B1", head, body_b, "gate-B"),
            ("A3", head, body_a + "\n\n" + dec(ids131[1][:3000]) + "\n\n" + dec(ids131[2][:2000]),
             "gate-A"),
            ("compaction", None, dec(ids131[0][:30000]), "gate-A"),
            ("after", head, dec(ids131[2][:4000]), "gate-A"),
            ("chat1", None, dec(ids131[1][:2500]), "gate-C"),
            ("chat2", None, dec(ids131[1][:2500]) + "\n\n" + dec(ids131[1][2500:3500]), "gate-C")):
        r = chat(a.base, a.model, sysm, user, sess)
        r["name"] = name
        r["after"] = reading(a.base)
        steps.append(r)
        print(f"[mem_gate] {name}: {r['status']} {r['prompt_tokens']} tokens in {r['s']} s, "
              f"reserved {r['after']['reserved_gib']} GiB, caches {r['after']['caches_gib']}",
              flush=True)
    report["fill"] = steps
    report["fill_s"] = round(time.perf_counter() - t_fill, 1)
    report["steady"] = reading(a.base)
    if not a.skip_long:
        ids = ids190 + ids131[0][:max(0, a.long_tokens - len(ids190))].tolist()
        text = dec(ids)
        with MemSampler() as ms:
            r = chat(a.base, a.model, None, text, "gate-L", max_tokens=8)
        r["mem"] = ms.report()
        r["after"] = reading(a.base)
        report["long"] = r
        print(f"[mem_gate] long: {r['status']} {r['prompt_tokens']} tokens in {r['s']} s; "
              f"prefill peak {r['after']['last_prefill_peak']}; min MemAvailable "
              f"{r['mem']['min_available_gb']} GB", flush=True)
    report["idle_after_long"] = reading(a.base)
    report["after_idle_empty"] = [chat(a.base, a.model, None, dec(ids131[2][:600]) + f"\n\n{i}",
                                       f"gate-S{i}", max_tokens=64) for i in range(2)]
    st = report["steady"]
    long_ = report.get("long") or {}
    lp = (long_.get("after") or {}).get("last_prefill_peak") or {}
    if long_.get("status") != 200 or lp.get("n") != long_.get("prompt_tokens"):
        lp = {}                       # a 503, or a reading of another request: no peak measured
    peak_gib = (lp.get("peak") or 0) / GIB
    report["verdict"] = {
        "steady_reserved_gib": st["engine_reserved_gib"], "ceiling_gib": st["ceiling_gib"],
        "steady_under_ceiling": (st["engine_reserved_gib"] or 0) <= (st["ceiling_gib"] or 0),
        "long_prompt_tokens": (report.get("long") or {}).get("prompt_tokens"),
        "long_status": (report.get("long") or {}).get("status"),
        "prefill_peak_gib": round(peak_gib, 3), "peak_gib": st["peak_gib"],
        "peak_under_declared": bool(lp) and peak_gib <= (st["peak_gib"] or 0),
    }
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"{a.label}.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=1)
    print(f"[mem_gate] {json.dumps(report['verdict'])}\n[mem_gate] wrote {path}", flush=True)
    v = report["verdict"]
    return 0 if v["steady_under_ceiling"] and (a.skip_long or v["peak_under_declared"]) else 1


if __name__ == "__main__":
    sys.exit(main())
