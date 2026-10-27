#!/usr/bin/env python3
"""Parallel requests on the engine: throughput at 1, 2 and 4 requests at once, each request's text
against its own alone run, and the proof that the requests really decoded together.

    python tools/parallel_bench.py --label par-off --port 8011 --max-len 262144 \\
        --server-arg=--parallel=4 --server-arg=--side-max-len=8192 --levels 1,2,4 --passes 2

It starts its own server through tools/row3.py's `server_cmd` (the row's configuration: raw engine,
caches off, the suffix store off unless `--store clean`), on `--port`, and stops it by PID.

WHAT IT RUNS (thinking off, greedy, the same prompts at every level):

  agreement  every prompt alone, then the prompts in groups of c at once, with logprobs (top 2):
             a request's text must be its alone text. Where it is not, the first differing token
             and the top-2 logprob gap at that position in the alone run are reported -- the
             near-tie rule of docs/exactness.md (a joint round trims each tree to its share, and
             the shape of the verified tree can move a near tie).
  throughput `--passes` times: each prompt alone (c = 1), then groups of c at once. Per request:
             TTFT (send to first token), decode tok/s = (completion tokens - 1) / (end - first
             token), the row's formula. Per group: total tok/s = all completion tokens of the
             group over the span from its first first-token to its last end.
  proof      /health every 0.2 s for the whole run: how many slots were in the barrier's decoding
             set, and the joint rounds by size. A level whose groups never show c slots decoding at
             once and no joint round of c blocks is INVALID (the requests queued).

Every limit knob goes into the report: the server command, the slots, the share table, the barrier
timeout, the admission floor, the store, the passes.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import threading
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from tools import row3  # noqa: E402

PROMPTS = [
    "Explain in detail how inserting a key into a B-tree works, including node splits, with a "
    "worked example on a tree of order 3. Use several paragraphs.",
    "Write a Python module that implements an LRU cache with a per-entry time to live. Include "
    "docstrings, type hints, and a set of unit tests using unittest.",
    "Tell a long story about a lighthouse keeper on a remote island who finds a message in a "
    "bottle, and what happens over the following winter.",
    "Describe the causes, the main events and the consequences of the French Revolution, in a "
    "structured essay with an introduction, sections and a conclusion.",
]


CHAT = [
    "What are the trade-offs between optimistic and pessimistic locking in a database?",
    "Explain how a hash map handles collisions, with two strategies and their costs.",
    "Why does the sky look blue during the day and red at sunset?",
    "Give me a plan for learning to play the piano as an adult, week by week for a month.",
    "What is the difference between a process and a thread, and when would you use each?",
    "Summarise the main ideas of the theory of evolution by natural selection.",
    "How does public-key cryptography let two strangers agree on a secret?",
    "Write a short guide to brewing pour-over coffee, step by step.",
    "What causes inflation, and what tools does a central bank have against it?",
    "Explain gradient descent to someone who knows high-school calculus.",
]


def wide_prompts() -> list[str]:
    """30 prompts as the paper's agreement check mixes them: 10 held-out prose continuations, 10
    held-out code continuations (bench/heldout_*.txt, never in the drafters' training), 10 chat
    questions."""
    root = REPO / "bench"
    out = []
    for name, ask in (("heldout_prose.txt", "Continue this text:\n\n"),
                      ("heldout_code.txt", "Continue this code:\n\n")):
        text = (root / name).read_text()
        step = max(400, len(text) // 10)
        for i in range(10):
            piece = text[i * step:i * step + 600]
            out.append(ask + piece)
    return out + CHAT


def _post(port: int, body: dict, timeout: float = 600.0) -> dict:
    """A streamed chat request: the text, the first-token and end times, the usage, the logprobs."""
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    parts, lps, usage, t_first = [], [], None, None
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            d = json.loads(line[6:])
            if d.get("usage"):
                usage = d["usage"]
            for c in d.get("choices") or []:
                delta = c.get("delta") or {}
                piece = (delta.get("content") or "") + (delta.get("reasoning_content") or "")
                if piece and t_first is None:
                    t_first = time.monotonic()
                parts.append(piece)
                lp = (c.get("logprobs") or {}).get("content")
                if lp:
                    lps.extend(lp)
    t_end = time.monotonic()
    return {"text": "".join(parts), "t0": t0, "t_first": t_first, "t_end": t_end,
            "usage": usage, "logprobs": lps}


def _body(model: str, prompt: str, max_tokens: int, logprobs: bool) -> dict:
    b = {"model": model, "messages": [{"role": "user", "content": prompt}],
         "max_tokens": max_tokens, "temperature": 0, "stream": True,
         "stream_options": {"include_usage": True},
         "chat_template_kwargs": {"enable_thinking": False}}
    if logprobs:
        b.update(logprobs=True, top_logprobs=2)
    return b


def _group(port: int, model: str, prompts: list[str], max_tokens: int, logprobs: bool) -> list:
    out: dict = {}
    errs: list = []
    go = threading.Barrier(len(prompts))

    def run(i):
        try:
            go.wait()
            out[i] = _post(port, _body(model, prompts[i], max_tokens, logprobs))
        except BaseException as exc:                               # noqa: BLE001
            errs.append(exc)

    ths = [threading.Thread(target=run, args=(i,)) for i in range(len(prompts))]
    for th in ths:
        th.start()
    for th in ths:
        th.join()
    if errs:
        raise errs[0]
    return [out[i] for i in range(len(prompts))]


def _rates(rs: list) -> dict:
    per, ttft, n_all = [], [], 0
    for r in rs:
        n = (r["usage"] or {}).get("completion_tokens") or 0
        n_all += n
        if r["t_first"] is not None and n > 1 and r["t_end"] > r["t_first"]:
            per.append((n - 1) / (r["t_end"] - r["t_first"]))
        if r["t_first"] is not None:
            ttft.append((r["t_first"] - r["t0"]) * 1e3)
    firsts = [r["t_first"] for r in rs if r["t_first"] is not None]
    span = (max(r["t_end"] for r in rs) - min(firsts)) if firsts else None
    return {"n": len(rs), "tokens": n_all, "per_request_tps": per,
            "total_tps": (n_all - len(rs)) / span if span else None, "ttft_ms": ttft,
            "span_s": span}


class Poller:
    """/health at 5 Hz: the slots in the barrier's decoding set and the joint rounds."""

    def __init__(self, port: int):
        self.port, self.samples, self._stop = port, [], threading.Event()
        self.th = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=2) as r:
                    p = json.loads(r.read()).get("parallel") or {}
                m = p.get("meeting") or {}
                self.samples.append({"t": time.monotonic(), "decoding": len(p.get("decoding") or []),
                                     "running": p.get("running"),
                                     "joint_rounds": m.get("joint_rounds"),
                                     "joint_hist": m.get("joint_hist") or {},
                                     "timeouts": m.get("timeouts"), "alone": m.get("alone"),
                                     "trimmed": m.get("trimmed"),
                                     "round_ms": m.get("round_ms") or {},
                                     "pass_ms": m.get("pass_ms") or {},
                                     "draft_ms": m.get("draft_ms") or {},
                                     "post_ms": m.get("post_ms") or {},
                                     "verify_wait_ms": m.get("verify_wait_ms") or {},
                                     "accept": m.get("accept") or {},
                                     "kv": [s.get("kv_length") for s in p.get("slots") or []]})
            except Exception:                                      # noqa: BLE001
                pass
            time.sleep(0.2)

    def start(self):
        self.th.start()
        return self

    def stop(self):
        self._stop.set()
        self.th.join(5)

    def window(self, t0: float, t1: float, c: int) -> dict:
        s = [x for x in self.samples if t0 <= x["t"] <= t1]
        if len(s) < 2:
            return {"samples": len(s)}
        a, b = s[0], s[-1]
        hist = {k: (b["joint_hist"].get(k, 0) - a["joint_hist"].get(k, 0))
                for k in set(a["joint_hist"]) | set(b["joint_hist"])}
        # kv lengths growing on c slots between two samples one second apart: c sequences decoding
        grow = 0
        for i in range(len(s)):
            j = next((k for k in range(i, len(s)) if s[k]["t"] - s[i]["t"] >= 1.0), None)
            if j is None:
                break
            ups = sum(1 for x, y in zip(s[i]["kv"], s[j]["kv"])
                      if x is not None and y is not None and y > x)
            grow = max(grow, ups)
        return {"samples": len(s),
                "max_decoding": max(x["decoding"] for x in s),
                "samples_at_c": sum(1 for x in s if x["decoding"] >= c),
                "joint_rounds": (b["joint_rounds"] or 0) - (a["joint_rounds"] or 0),
                "joint_hist": {str(k): v for k, v in sorted(hist.items(), key=lambda kv: str(kv[0]))
                               if v},
                "timeouts": (b["timeouts"] or 0) - (a["timeouts"] or 0),
                "alone": (b["alone"] or 0) - (a["alone"] or 0),
                "trimmed": (b["trimmed"] or 0) - (a["trimmed"] or 0),
                # mean ms in the window: a slot's round (submit to submit) by slots decoding, and a
                # joint pass's GPU time by blocks (QSE_PARALLEL_TIMING=1 only)
                "round_ms": _mean_delta(a["round_ms"], b["round_ms"]),
                "pass_ms": _mean_delta(a["pass_ms"], b["pass_ms"]),
                # a slot's draft (by slots decoding) and its tokens a round (joint<N> or alone)
                "draft_ms": _mean_delta(a["draft_ms"], b["draft_ms"]),
                # after the verify until the next draft (accept, commit, sync, stream, the turn's
                # offer), and the verify call itself as the slot sees it (its wait and the pass)
                "post_ms": _mean_delta(a["post_ms"], b["post_ms"]),
                "verify_wait_ms": _mean_delta(a["verify_wait_ms"], b["verify_wait_ms"]),
                "accept": _mean_delta(a["accept"], b["accept"]),
                "max_slots_growing_1s": grow}


def _mean_delta(a: dict, b: dict) -> dict:
    out = {}
    for k, (ms, n) in b.items():
        ms0, n0 = a.get(k, [0.0, 0])
        if n - n0 > 0:
            out[k] = {"mean": round((ms - ms0) / (n - n0), 2), "n": n - n0}
    return out


def _diverge(a: dict, b: dict) -> dict | None:
    """Where `b`'s text parts from `a`'s, by the alone run's token list (logprobs)."""
    if a["text"] == b["text"]:
        return None
    la, lb = a["logprobs"], b["logprobs"]
    i = next((k for k, (x, y) in enumerate(zip(la, lb)) if x.get("token") != y.get("token")),
             min(len(la), len(lb)))
    gap = None
    if i < len(la):
        top = la[i].get("top_logprobs") or []
        if len(top) >= 2:
            gap = round(top[0]["logprob"] - top[1]["logprob"], 6)
    return {"token_index": i, "alone_tokens": len(la), "together_tokens": len(lb),
            "top2_logprob_gap_alone": gap}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--label", required=True)
    p.add_argument("--port", type=int, default=8011)
    p.add_argument("--repo", type=Path, default=REPO)
    p.add_argument("--python", type=Path, default=row3.DEFAULT_PY)
    p.add_argument("--pythonpath", type=Path, default=row3.HOME / "pylibs")
    p.add_argument("--max-len", type=int, default=262144)
    p.add_argument("--budget", type=int, default=16)
    p.add_argument("--len-fixed", type=int, default=0)
    p.add_argument("--len-latch", action="store_true", default=True)
    p.add_argument("--nvfp4", default=row3.DEFAULT_NV)
    p.add_argument("--head", default=row3.DEFAULT_HEAD)
    p.add_argument("--store", default="off", choices=("off", "clean"))
    p.add_argument("--clean-store", default=str(row3.HOME / "qwen38-suffix-norow-0923"))
    p.add_argument("--server-arg", action="append", default=[])
    p.add_argument("--env", action="append", default=[])
    p.add_argument("--levels", default="1,2,4")
    p.add_argument("--passes", type=int, default=2)
    p.add_argument("--max-tokens", type=int, default=384)
    p.add_argument("--no-agreement", action="store_true")
    p.add_argument("--prompt-set", default="bench", choices=("bench", "wide"),
                   help="bench: the four long-answer prompts; wide: 30 prompts (prose, code, chat) "
                        "for the agreement at the paper's scale")
    p.add_argument("--agreement-only", action="store_true")
    p.add_argument("--start-timeout", type=int, default=600)
    p.add_argument("--mem-floor-gb", type=float, default=10.0)
    p.add_argument("--out-dir", type=Path, default=REPO / "results/parallel")
    a = p.parse_args()
    a.with_suffix_store = False
    global PROMPTS
    if a.prompt_set == "wide":
        PROMPTS = wide_prompts()
    levels = [int(x) for x in a.levels.split(",")]
    env_extra = dict(kv.split("=", 1) for kv in a.env)
    out_dir = a.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    log = out_dir / f"{a.label}.server.log"
    report = {"label": a.label, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
              "server_cmd": row3.server_cmd(a), "env": env_extra, "levels": levels,
              "passes": a.passes, "max_tokens": a.max_tokens, "store": a.store,
              "prompts": len(PROMPTS), "thinking": "off", "temperature": 0}
    proc = row3.start_server(a, env_extra, log)
    poll = None
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{a.port}/v1/models", timeout=10) as r:
            model = json.loads(r.read())["data"][0]["id"]
        with urllib.request.urlopen(f"http://127.0.0.1:{a.port}/health", timeout=10) as r:
            h = json.loads(r.read())
        report["parallel_at_start"] = h.get("parallel")
        report["slots"] = ([{"index": s["index"], "window": s["max_len"]}
                            for s in (h.get("parallel") or {}).get("slots") or []])
        from server import slots as _sl
        report["knobs"] = {"shares": _sl.SHARES, "server_args": a.server_arg}
        poll = Poller(a.port).start()
        # warm-up: the joint shapes' first calls (autotuning) outside the measured windows
        _group(a.port, model, PROMPTS[:1], 32, False)
        for c in levels:
            if c > 1:
                _group(a.port, model, PROMPTS[:c], 48, False)
        # agreement
        if not a.no_agreement:
            alone = [_group(a.port, model, [pr], a.max_tokens, True)[0] for pr in PROMPTS]
            agree = {}
            for c in levels:
                if c == 1:
                    continue
                rows = []
                for g in range(0, len(PROMPTS), c):
                    idx = list(range(g, min(g + c, len(PROMPTS))))
                    t0 = time.monotonic()
                    rs = _group(a.port, model, [PROMPTS[i] for i in idx], a.max_tokens, True)
                    proof = poll.window(t0, time.monotonic(), len(idx))
                    for i, r in zip(idx, rs):
                        d = _diverge(alone[i], r)
                        rows.append({"prompt": i, "identical": d is None, "diverge": d,
                                     "tokens": (r["usage"] or {}).get("completion_tokens")})
                    rows[-1]["proof"] = proof
                agree[str(c)] = {"identical": sum(1 for x in rows if x["identical"]),
                                 "of": len(rows), "rows": rows}
                print(f"[par] agreement c{c}: {agree[str(c)]['identical']}/{len(rows)} identical "
                      f"to alone; " + "; ".join(
                          f"p{x['prompt']} at {x['diverge']['token_index']} gap "
                          f"{x['diverge']['top2_logprob_gap_alone']}"
                          for x in rows if not x["identical"]), flush=True)
            report["agreement"] = agree
            gaps = sorted(x["diverge"]["top2_logprob_gap_alone"] for v in agree.values()
                          for x in v["rows"] if not x["identical"]
                          and x["diverge"]["top2_logprob_gap_alone"] is not None)
            report["agreement_summary"] = {
                c: f"{v['identical']}/{v['of']}" for c, v in agree.items()}
            report["agreement_gaps"] = gaps
            print(f"[par] AGREEMENT {json.dumps(report['agreement_summary'])} gaps (nats) "
                  f"{gaps}", flush=True)
        # throughput
        thr: dict = {str(c): [] for c in levels}
        for ps in range(0 if a.agreement_only else a.passes):
            for c in levels:
                for g in range(0, len(PROMPTS), c):
                    idx = list(range(g, min(g + c, len(PROMPTS))))
                    t0 = time.monotonic()
                    rs = _group(a.port, model, [PROMPTS[i] for i in idx], a.max_tokens, False)
                    st = _rates(rs)
                    st["proof"] = poll.window(t0, time.monotonic(), len(idx))
                    st["pass"], st["prompts"] = ps, idx
                    thr[str(c)].append(st)
                    print(f"[par] pass {ps} c{c} prompts {idx}: total {st['total_tps']:.2f} tok/s, "
                          f"per request {', '.join(f'{x:.1f}' for x in st['per_request_tps'])}, "
                          f"ttft {', '.join(f'{x:.0f}' for x in st['ttft_ms'])} ms; proof "
                          f"{json.dumps(st['proof'])}", flush=True)
        summary = {}
        for c, groups in thr.items():
            tot = [x["total_tps"] for x in groups if x["total_tps"]]
            per = [y for x in groups for y in x["per_request_tps"]]
            tt = [y for x in groups for y in x["ttft_ms"]]
            full = [x for x in groups if len(x["prompts"]) == int(c)]
            valid = all(x["proof"].get("max_decoding", 0) >= int(c)
                        and (int(c) == 1 or sum(v for k, v in x["proof"]["joint_hist"].items()
                                                if int(k) >= 2) > 0)
                        for x in full) if int(c) > 1 else True
            summary[c] = {"total_tps_median": round(statistics.median(tot), 2) if tot else None,
                          "total_tps": [round(x, 2) for x in tot],
                          "per_request_tps_mean": round(statistics.fmean(per), 2) if per else None,
                          "per_request_tps_median": round(statistics.median(per), 2) if per else None,
                          "ttft_ms_p50": round(statistics.median(tt), 1) if tt else None,
                          "concurrency_proven": valid}
        # c=1 total is a single request's rate; the group total of c=1 is the same number
        report["throughput"] = thr
        report["summary"] = summary
        print("[par] SUMMARY " + json.dumps(summary), flush=True)
    finally:
        if poll is not None:
            poll.stop()
            report["health_samples"] = len(poll.samples)
        row3.stop_server(proc)
        path = out_dir / f"{a.label}.json"
        path.write_text(json.dumps(report, indent=1))
        print(f"[par] report {path}", flush=True)


if __name__ == "__main__":
    main()
