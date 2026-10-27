"""The plugin's cost (P6 gate): decode tokens a second of one engine with and without TandemLLM's plugin.

    python tools/proxy_plugin_ab.py --url http://127.0.0.1:8101/v1 --model Qwen/Qwen3.5-0.8B \
        --arm off --n 20 --out results/engines/ab-vllm-off-1.json
    python tools/proxy_plugin_ab.py --compare A1.json B.json A2.json --out ab-vllm.json

One arm is one server start (tools/proxy_engines_check.sh --plugins --ab starts the engine three
times: without the plugin, with it, without it again). Per arm: 2 warm-up requests, then `--n`
requests one after the other, streamed, temperature 0, the same prompt each time, `max_tokens`
256 with `ignore_eos`, so every request decodes the same number of tokens. A request's decode
rate is (completion tokens - 1) / (last token - first token) at the client; the arm's figure is the
median. Concurrency is 1 on purpose: the logger's work per step is what is measured, and a single
stream is where a per-step cost shows the most.

`--compare` reads the three arms: the gate is |B - mean(A1, A2)| / mean(A1, A2) <= 1 %. When the
two runs without the plugin differ by more than 1 % from each other, the box is noisier than the
gate and the result says `inconclusive` instead of pass or fail. Every limit the engine ran with
goes into the arm's file (`--engine-args`), so no setting hides in a default.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.request

PROMPT = ("Write a long, plain story about a lighthouse keeper who counts ships. Keep going "
          "without a title and without lists.")


def one(url: str, model: str, max_tokens: int) -> dict:
    body = {"model": model, "stream": True, "temperature": 0, "max_tokens": max_tokens,
            "ignore_eos": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": PROMPT}]}
    req = urllib.request.Request(url.rstrip("/") + "/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first = last = None
    n = None
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.strip()
            if not line.startswith(b"data:") or line == b"data: [DONE]":
                continue
            obj = json.loads(line[5:])
            for ch in obj.get("choices") or []:
                d = ch.get("delta") or {}
                if d.get("content") or d.get("reasoning_content") or d.get("reasoning"):
                    now = time.perf_counter()
                    first = first or now
                    last = now
            if isinstance(obj.get("usage"), dict):
                n = obj["usage"].get("completion_tokens")
    tps = (n - 1) / (last - first) if n and n > 1 and first and last and last > first else None
    return {"completion_tokens": n, "ttft_s": None if first is None else round(first - t0, 4),
            "decode_tps": None if tps is None else round(tps, 3)}


def run_arm(a) -> int:
    for _ in range(2):
        one(a.url, a.model, a.max_tokens)
    rows = [one(a.url, a.model, a.max_tokens) for _ in range(a.n)]
    rates = [r["decode_tps"] for r in rows if r["decode_tps"]]
    out = {"arm": a.arm, "url": a.url, "model": a.model, "n": a.n, "max_tokens": a.max_tokens,
           "concurrency": 1, "engine_args": a.engine_args, "requests": rows,
           "decode_tps_median": statistics.median(rates) if rates else None,
           "decode_tps_spread": (round((max(rates) - min(rates)) / statistics.median(rates), 4)
                                 if rates else None)}
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: out[k] for k in ("arm", "decode_tps_median", "decode_tps_spread")}))
    return 0 if len(rates) == a.n else 1


def compare(a) -> int:
    a1, b, a2 = (json.load(open(p)) for p in a.compare)
    ra1, rb, ra2 = (x["decode_tps_median"] for x in (a1, b, a2))
    if not (ra1 and rb and ra2):
        verdict, rel, noise = "fail", None, None
    else:
        base = (ra1 + ra2) / 2
        rel = (rb - base) / base
        noise = abs(ra1 - ra2) / base
        verdict = ("inconclusive" if noise > a.gate else
                   "pass" if abs(rel) <= a.gate else "fail")
    out = {"gate": a.gate, "verdict": verdict, "off_1": ra1, "on": rb, "off_2": ra2,
           "on_vs_off": None if rel is None else round(rel, 5),
           "off_noise": None if noise is None else round(noise, 5),
           "engine_args": b.get("engine_args"), "n": b.get("n"),
           "max_tokens": b.get("max_tokens"), "concurrency": 1}
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out))
    return 0 if verdict in ("pass", "inconclusive") else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url")
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--arm", choices=("off", "on"))
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--engine-args", default="", help="the engine's command line, recorded")
    ap.add_argument("--compare", nargs=3, metavar=("OFF1", "ON", "OFF2"))
    ap.add_argument("--gate", type=float, default=0.01)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.compare:
        return compare(a)
    if not a.url or not a.arm:
        ap.error("--url and --arm, or --compare")
    return run_arm(a)


if __name__ == "__main__":
    sys.exit(main())
