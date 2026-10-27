"""Check a live server's per-response usage and timings: exactly once, and consistent.

    python tools/usage_check.py --base http://127.0.0.1:8011 --out results/obs/usage-check.json
    python tools/usage_check.py --base https://your-host.example --model qwen38-spark-engine

Six requests, each small: a streamed chat with no `stream_options` (what Open WebUI's base models
send), one with `include_usage: true`, one with `false`, a non-streamed chat, a streamed chat with
thinking on, and a streamed `/v1/completions`. For each it checks where `usage` / `timings` /
`metrics` arrived (one chunk, the right one), replays the stream through a port of Open WebUI's own
usage merge (the counts it would show), and the arithmetic between the fields. Standard library
only, so it runs from any machine that reaches the server.

The port of Open WebUI 0.11.3 -- backend/open_webui/utils/response.py:13-47 `normalize_usage`,
:100-139 `merge_usage` with `_merge_numeric_usage_map`, and the stream loop at
utils/middleware.py:4952-4962 -- was read from the running container on 2026-09-24.
tests/test_usage.py imports it from here.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from numbers import Number

FIELDS = ("usage", "timings", "metrics")


# ----------------------------------------------------------------- Open WebUI 0.11.3, ported

def owui_normalize(u: dict) -> dict:
    if not u:
        return {}
    inp = u.get("input_tokens") or u.get("prompt_tokens") or u.get("prompt_eval_count")
    if inp is None:
        inp = int(u.get("prompt_n") or 0) + int(u.get("cache_n") or 0)
    out = (u.get("output_tokens") or u.get("completion_tokens") or u.get("eval_count")
           or u.get("predicted_n") or 0)
    total = u.get("total_tokens") or (inp + out)
    r = dict(u)
    r["input_tokens"], r["output_tokens"], r["total_tokens"] = int(inp), int(out), int(total)
    return r


def _num(v) -> bool:
    return isinstance(v, Number) and not isinstance(v, bool)


def _merge_map(cur, inc):
    cur, inc = cur or {}, inc or {}
    r = {**cur, **inc}
    for k in set(cur) | set(inc):
        a, b = cur.get(k, 0), inc.get(k, 0)
        if isinstance(a, dict) or isinstance(b, dict):
            r[k] = _merge_map(a if isinstance(a, dict) else {}, b if isinstance(b, dict) else {})
        elif _num(a) or _num(b):
            r[k] = (a if _num(a) else 0) + (b if _num(b) else 0)
    return r


_SUMMABLE = {"input_tokens", "output_tokens", "total_tokens", "cost", "total_cost", "input_cost",
             "output_cost", "prompt_cost", "completion_cost"}
_DETAILS = ("prompt_tokens_details", "completion_tokens_details", "input_tokens_details",
            "output_tokens_details")


def owui_merge(cur, inc):
    cu = owui_normalize(cur or {}) if cur else {}
    iu = owui_normalize(inc or {}) if inc else {}
    if not iu:
        return cu
    if not cu:
        return iu
    r = {**cu, **iu}
    for k in _SUMMABLE:
        if k in cu or k in iu:
            a, b = cu.get(k, 0), iu.get(k, 0)
            if _num(a) or _num(b):
                r[k] = (a if _num(a) else 0) + (b if _num(b) else 0)
    for k in _DETAILS:
        if isinstance(cu.get(k), dict) or isinstance(iu.get(k), dict):
            r[k] = _merge_map(cu.get(k) if isinstance(cu.get(k), dict) else {},
                              iu.get(k) if isinstance(iu.get(k), dict) else {})
    r["prompt_tokens"] = (iu.get("prompt_tokens") or iu.get("input_tokens")
                          or cu.get("prompt_tokens", 0))
    r["completion_tokens"] = (iu.get("completion_tokens") or iu.get("output_tokens")
                              or cu.get("completion_tokens", 0))
    return r


def owui_replay(chunks) -> dict:
    """What Open WebUI's message `usage` ends up as after reading these stream chunks."""
    u = None
    for data in chunks:
        raw = dict(data.get("usage", {}) or {})
        raw.update(data.get("timings", {}))                # llama.cpp
        if raw:
            u = owui_merge(u, raw)
    return u or {}


# ----------------------------------------------------------------- the checks

def carriers(chunks) -> list[int]:
    return [i for i, c in enumerate(chunks) if any(k in c for k in FIELDS)]


def consistent(fields: dict) -> list[str]:
    """The arithmetic between the fields; an empty list is a pass."""
    bad = []
    u, t, m = fields.get("usage", {}), fields.get("timings", {}), fields.get("metrics", {})
    if u.get("total_tokens") != u.get("prompt_tokens", 0) + u.get("completion_tokens", 0):
        bad.append("total_tokens != prompt + completion")
    if t.get("predicted_n") != u.get("completion_tokens"):
        bad.append("predicted_n != completion_tokens")
    if t.get("cache_n", 0) + t.get("prompt_n", 0) != u.get("prompt_tokens"):
        bad.append("cache_n + prompt_n != prompt_tokens")
    if t.get("cache_n") != u.get("prompt_tokens_details", {}).get("cached_tokens"):
        bad.append("cache_n != cached_tokens")
    if abs(t.get("ttft_ms", 0) - t.get("queue_ms", 0) - t.get("prompt_ms", 0)) > 0.1:
        bad.append("ttft_ms != queue_ms + prompt_ms")
    c = u.get("completion_tokens", 0)
    if c > 1 and t.get("predicted_ms"):
        rate = (c - 1) * 1000.0 / t["predicted_ms"]
        # both are rounded to 0.01: the true divisor lies within 0.005 of the printed one, so the
        # rate can be off by up to 0.005 / (ms - 0.005) of itself -- 100 % when a replay decodes
        # in 0.01 ms, which the old 0.006 / ms (60 %) failed about half the time on the box
        tol = rate * (0.01 + 0.005 / max(t["predicted_ms"] - 0.005, 1e-9)) + 0.02
        if abs(rate - t.get("predicted_per_second", 0)) > tol:
            bad.append(f"predicted_per_second {t.get('predicted_per_second')} != {rate:.2f}")
    r = u.get("completion_tokens_details", {}).get("reasoning_tokens", 0)
    if not 0 <= r <= c or t.get("reasoning_n") != r:
        bad.append("reasoning tokens out of range or disagreeing")
    if t.get("draft_n_accepted", 0) > t.get("draft_n", 0):
        bad.append("draft_n_accepted > draft_n")
    if m.get("time_to_first_token_ms") != t.get("ttft_ms"):
        bad.append("metrics.time_to_first_token_ms != timings.ttft_ms")
    return bad


def _post(base: str, path: str, body: dict, token: str | None, timeout: float):
    req = urllib.request.Request(base.rstrip("/") + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          **({"Authorization": f"Bearer {token}"} if token else {})})
    return urllib.request.urlopen(req, timeout=timeout)


def stream_chunks(resp) -> tuple[list[dict], bool]:
    chunks, done = [], False
    for raw in resp:
        line = raw.decode("utf-8", "replace").rstrip("\r\n")
        if line == "data: [DONE]":
            done = True
        elif line.startswith("data: {"):
            chunks.append(json.loads(line[6:]))
    return chunks, done


CASES = [
    # (name, path, body extra, stream, expected placement)
    ("stream-default", "/v1/chat/completions", {}, True, "finish"),
    ("stream-include-usage", "/v1/chat/completions",
     {"stream_options": {"include_usage": True}}, True, "separate"),
    ("stream-include-usage-false", "/v1/chat/completions",
     {"stream_options": {"include_usage": False}}, True, "none"),
    ("json", "/v1/chat/completions", {}, False, "body"),
    ("stream-thinking", "/v1/chat/completions", {"think": True}, True, "finish"),
    ("completions-stream", "/v1/completions", {}, True, "finish"),
]


def run_case(a, name, path, extra, stream, expect) -> dict:
    extra = dict(extra)
    think = extra.pop("think", False)
    if path.endswith("chat/completions"):
        body = {"model": a.model, "stream": stream, "max_tokens": a.max_tokens, "temperature": 0,
                "messages": [{"role": "user", "content": a.prompt}],
                "chat_template_kwargs": {"enable_thinking": think}}
    else:
        body = {"model": a.model, "stream": stream, "max_tokens": a.max_tokens, "temperature": 0,
                "prompt": a.prompt}
    body.update(extra)
    t0 = time.time()
    out = {"case": name, "expect": expect, "errors": []}
    with _post(a.base, path, body, a.token, a.timeout) as resp:
        if stream:
            chunks, done = stream_chunks(resp)
            where = carriers(chunks)
            out["chunks"], out["done"] = len(chunks), done
            if not done:
                out["errors"].append("no [DONE]")
            if expect == "none":
                if where:
                    out["errors"].append(f"usage on {len(where)} chunks, expected none")
                fields = {}
            else:
                if len(where) != 1:
                    out["errors"].append(f"usage on {len(where)} chunks, expected exactly 1")
                    return out
                c = chunks[where[0]]
                if expect == "finish" and not (c["choices"] and c["choices"][0]["finish_reason"]):
                    out["errors"].append("the carrier is not the finish chunk")
                if expect == "separate" and (c["choices"] != [] or where[0] != len(chunks) - 1):
                    out["errors"].append("the carrier is not the last choices:[] chunk")
                fields = {k: c.get(k) for k in FIELDS}
                seen = owui_replay(chunks)
                out["owui"] = {k: seen.get(k) for k in ("input_tokens", "output_tokens",
                                                         "predicted_per_second", "ttft_ms")}
                if (seen.get("input_tokens") != fields["usage"]["prompt_tokens"]
                        or seen.get("output_tokens") != fields["usage"]["completion_tokens"]):
                    out["errors"].append(f"Open WebUI would show {out['owui']}")
        else:
            payload = json.loads(resp.read())
            fields = {k: payload.get(k) for k in FIELDS}
            if any(v is None for v in fields.values()):
                out["errors"].append("the body lacks one of usage/timings/metrics")
                return out
    out["wall_s"] = round(time.time() - t0, 3)
    if fields:
        out["errors"] += consistent(fields)
        out.update(fields)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="qwen38-spark-engine")
    ap.add_argument("--token", default=None, help="a bearer token, if the route needs one")
    ap.add_argument("--prompt", default="Name three rivers in Europe, one line each.")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--out", default=None, help="write every case, fields included, as JSON")
    a = ap.parse_args()
    results, bad = [], 0
    for case in CASES:
        try:
            r = run_case(a, *case)
        except Exception as exc:                                   # noqa: BLE001
            r = {"case": case[0], "errors": [f"{type(exc).__name__}: {exc}"]}
        results.append(r)
        bad += bool(r["errors"])
        t, u = r.get("timings") or {}, r.get("usage") or {}
        print(f"{'PASS' if not r['errors'] else 'FAIL'}  {r['case']:<28} "
              f"prompt={u.get('prompt_tokens')} completion={u.get('completion_tokens')} "
              f"reasoning={(u.get('completion_tokens_details') or {}).get('reasoning_tokens')} "
              f"cached={t.get('cache_n')} ttft={t.get('ttft_ms')} ms "
              f"decode={t.get('predicted_per_second')} tok/s blocks={t.get('blocks')} "
              f"source={t.get('cache_source')}"
              + (f"  !! {'; '.join(r['errors'])}" if r["errors"] else ""), flush=True)
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"base": a.base, "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                       "cases": results}, f, indent=1)
    print(f"{len(results) - bad}/{len(results)} cases pass")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
