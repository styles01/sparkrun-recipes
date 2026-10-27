"""Where a TTFT difference between two configurations comes from: the prefill, or the first block.

The row's TTFT is the client's time to the first content delta. The server sends the prefill's
token before any speculative block runs (`server/app.py::generate_stream` yields it first), unless
the stream holds it back -- a first token whose bytes are not a whole character waits for the next
token (`server/stream.py::Detokenizer`), and then the first block's draft and verify are inside the
TTFT. So a flag that makes the first block dearer can move TTFT without touching the prefill.

This asks the question directly, on the row's own prompts: for each configuration, one server, the
50 prompts at `max_tokens` 1 (the prefill alone -- no block runs) and 2 (one block after the prefill
token), client clock, streamed, thinking off, temperature 0, as the atlas sends them. Configurations
run in the order given and then reversed (A B C C B A), so drift between servers averages out.

    python tools/ttft_probe.py --env QWEN38_X=1 ... --config base: --config spd41:QWEN38_VERIFY_ROWS=32
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def ttft(port: int, messages: list, max_tokens: int) -> tuple[float, float]:
    """Seconds to the first content delta and to the end of the response."""
    body = {"model": "x", "messages": messages, "max_tokens": max_tokens, "temperature": 0,
            "seed": 42, "stream": True, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first = None
    with urllib.request.urlopen(req, timeout=300) as fh:
        for raw in fh:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            ev = json.loads(line[5:])
            ch = (ev.get("choices") or [{}])[0]
            if first is None and (ch.get("delta") or {}).get("content"):
                first = time.perf_counter() - t0
    return (first if first is not None else float("nan")), time.perf_counter() - t0


def parse_config(spec: str) -> tuple[str, dict]:
    label, _, kvs = spec.partition(":")
    return label, dict(kv.split("=", 1) for kv in kvs.split(",") if kv)


def summarize(results: dict) -> list[str]:
    """Median TTFT per configuration and max_tokens, and each configuration's per-prompt median
    difference against the first one."""
    labels = list(results)
    lines = [f"{'config':<14}{'mt':>4}{'median ms':>11}{'p10':>8}{'p90':>8}{'vs ' + labels[0]:>16}"]
    for lab in labels:
        for mt, per in sorted(results[lab].items()):
            flat = [x for xs in per.values() for x in xs]
            med = statistics.median(flat)
            q = statistics.quantiles(flat, n=10)
            base = results[labels[0]].get(mt, {})
            diffs = [statistics.median(per[p]) - statistics.median(base[p])
                     for p in per if p in base]
            d = statistics.median(diffs) if diffs and lab != labels[0] else 0.0
            lines.append(f"{lab:<14}{mt:>4}{med * 1e3:>11.1f}{q[0] * 1e3:>8.1f}{q[-1] * 1e3:>8.1f}"
                         f"{d * 1e3:>+15.2f}")
    return lines


def main() -> None:
    from tools import row3
    from tools.p1_slots import row_prompts
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", action="append", required=True,
                    help="label:K=V,K=V -- a configuration's environment on top of --env")
    ap.add_argument("--env", action="append", default=[], metavar="K=V")
    ap.add_argument("--server-arg", action="append", default=[])
    ap.add_argument("--max-tokens", default="1,2")
    ap.add_argument("--port", type=int, default=8011)
    ap.add_argument("--atlas", default=os.path.expanduser("~/inf-atlas"))
    ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent.parent)
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    base_env = dict(kv.split("=", 1) for kv in a.env)
    configs = [parse_config(c) for c in a.config]
    order = configs + configs[::-1]
    prompts = row_prompts(a.atlas)
    mts = [int(m) for m in a.max_tokens.split(",")]
    ns = argparse.Namespace(port=a.port, python=row3.DEFAULT_PY, pythonpath=Path.home() / "pylibs",
                            repo=a.repo, max_len=262144, len_fixed=0, budget=16, len_latch=True,
                            nvfp4=row3.DEFAULT_NV, head=row3.DEFAULT_HEAD, store="off",
                            server_arg=list(a.server_arg), start_timeout=600, mem_floor_gb=10.0,
                            tree=True, clean_store="", with_suffix_store=False)
    results: dict = {lab: {} for lab, _ in configs}
    for i, (lab, env) in enumerate(order):
        proc = row3.start_server(ns, dict(base_env, **env),
                                 a.repo / "results" / "ttft" / f"server-{i}-{lab}.log")
        try:
            for p in prompts[:3]:                          # the warm-ups the atlas sends too
                ttft(a.port, p["messages"], 2)
            for mt in mts:
                for p in prompts:
                    t, _ = ttft(a.port, p["messages"], mt)
                    results[lab].setdefault(mt, {}).setdefault(p["name"], []).append(t)
        finally:
            row3.stop_server(proc)
        print(f"[ttft] {lab} pass {1 + (i >= len(configs))} done", flush=True)
    lines = summarize(results)
    print("\n".join(lines))
    if a.json:
        Path(a.json).write_text(json.dumps(results))


if __name__ == "__main__":
    main()
