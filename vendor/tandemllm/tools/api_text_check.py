"""Does a server build answer plain requests exactly as another build does? (the api lane's check)

A change to the server's request handling must leave every request that does not use it as it
was, byte for byte. This starts one test server from a checkout (the gate's server command, store
off, caches off), sends a fixed battery twice -- the five bench workloads as greedy JSON, two of
them streamed, one through /v1/completions, one with thinking on, one with tools, one under the
agent lane's penalties, one seeded sampled request -- and writes the answers: text, finish reason,
token count, tool calls without their random ids. `--compare` then holds every file's pass k
against the first file's pass k.

Pass 2 against pass 1 of the same build is the control: greedy output is deterministic up to the
batched-verify tie flips the release itself has (docs/exactness.md), and the lookup drafter has
indexed pass 1's text by pass 2, so a build may differ from itself there -- and every build must
differ in the same way.

    python tools/api_text_check.py --repo ~/qwen38-spark-engine-apibase --out base.json \\
        --server-arg=--sampled-tree=det
    python tools/api_text_check.py --repo ~/qwen38-spark-engine-api --out api.json \\
        --server-arg=--sampled-tree=det
    python tools/api_text_check.py --compare base.json api.json
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

WEATHER = [{"type": "function", "function": {
    "name": "get_weather", "description": "The weather in a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                   "required": ["city"]}}}]
OFF = {"chat_template_kwargs": {"enable_thinking": False}}


def battery() -> list[tuple[str, str, dict]]:
    """(name, path, body) -- only fields every build since rc5 serves."""
    from tools.bench_decode import PROMPTS as BENCH
    out = []
    for name, text in BENCH.items():
        out.append((f"bench/{name}", "/v1/chat/completions",
                    dict(OFF, messages=[{"role": "user", "content": text}], max_tokens=256,
                         temperature=0.0)))
    for name in list(BENCH)[:2]:
        out.append((f"stream/{name}", "/v1/chat/completions",
                    dict(OFF, messages=[{"role": "user", "content": BENCH[name]}], max_tokens=256,
                         temperature=0.0, stream=True)))
    first = next(iter(BENCH.values()))
    out += [
        ("completions", "/v1/completions", {"prompt": first, "max_tokens": 128, "temperature": 0.0}),
        ("thinking", "/v1/chat/completions",
         {"messages": [{"role": "user", "content": "What is 17 times 23? Answer briefly."}],
          "max_tokens": 256, "temperature": 0.0, "reasoning_effort": "low"}),
        ("tools", "/v1/chat/completions",
         dict(OFF, messages=[{"role": "user", "content": "What is the weather in Berlin?"}],
              tools=WEATHER, max_tokens=128, temperature=0.0)),
        ("tools/stream", "/v1/chat/completions",
         dict(OFF, messages=[{"role": "user", "content": "What is the weather in Berlin?"}],
              tools=WEATHER, max_tokens=128, temperature=0.0, stream=True)),
        ("penalties", "/v1/chat/completions",
         dict(OFF, messages=[{"role": "user", "content": first}], max_tokens=128,
              temperature=0.0, no_repeat_ngram_size=4, presence_penalty=0.5)),
        ("seeded", "/v1/chat/completions",
         dict(OFF, messages=[{"role": "user", "content": "Write one sentence about the sea."}],
              max_tokens=64, temperature=0.7, seed=11)),
    ]
    return out


def post(port: int, path: str, body: dict) -> dict:
    """One request, reduced to what a client reads: text, finish, tokens, calls (ids dropped)."""
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as fh:
        raw = fh.read().decode()
    if not body.get("stream"):
        r = json.loads(raw)
        c = r["choices"][0]
        msg = c.get("message") or {}
        calls = [(t["function"]["name"], t["function"]["arguments"])
                 for t in msg.get("tool_calls") or []]
        return {"text": msg.get("content") if "message" in c else c["text"],
                "reasoning": msg.get("reasoning_content"), "finish": c["finish_reason"],
                "tokens": r["usage"]["completion_tokens"], "calls": calls}
    text, finish, tokens, parts = "", None, None, {}
    for line in raw.splitlines():
        if not line.startswith("data: {"):
            continue
        ev = json.loads(line[6:])
        if ev.get("usage"):
            tokens = ev["usage"]["completion_tokens"]
        for c in ev["choices"]:
            d = c.get("delta") or {}
            text += d.get("content") or c.get("text") or ""
            for tc in d.get("tool_calls") or []:
                p = parts.setdefault(tc["index"], ["", ""])
                p[0] += (tc.get("function") or {}).get("name") or ""
                p[1] += (tc.get("function") or {}).get("arguments") or ""
            finish = c.get("finish_reason") or finish
    return {"text": text, "finish": finish, "tokens": tokens,
            "calls": [tuple(p) for _, p in sorted(parts.items())]}


def run(a) -> None:
    from tools import row3
    repo = Path(a.repo).expanduser().resolve()
    ns = argparse.Namespace(port=a.port, python=Path(a.python), pythonpath=Path(a.pythonpath),
                            repo=repo, max_len=a.max_len, len_fixed=0, budget=16,
                            nvfp4=row3.DEFAULT_NV, head=row3.DEFAULT_HEAD,
                            server_arg=["--drop-idle"] + list(a.server_arg), len_latch=True,
                            start_timeout=600, tree=True)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    proc = row3.start_server(ns, {}, out.with_suffix(".log"))
    res = {"repo": str(repo), "code": row3.code_hash(repo)[:16], "server_args": a.server_arg,
           "passes": []}
    try:
        for p in (1, 2):
            got = {name: post(a.port, path, body) for name, path, body in battery()}
            res["passes"].append(got)
            print(f"[text] pass {p}: " + " ".join(f"{n}={v['tokens']}" for n, v in got.items()),
                  flush=True)
    finally:
        row3.stop_server(proc)
    out.write_text(json.dumps(res, indent=1))


def compare(paths: list[str]) -> int:
    """Every build's pass k against the first build's pass k: the verdict. Pass 2 against pass 1
    within a build is the control, reported beside it and not judged -- the lookup drafter indexes
    what pass 1 wrote, so pass 2 verifies other shapes and may flip a one-ulp tie, in every build
    alike."""
    docs = [(p, json.loads(Path(p).read_text())) for p in paths]
    ref = docs[0][1]["passes"]
    bad = 0
    for path, d in docs:
        for i, p in enumerate(d["passes"]):
            diff = [w for w in ref[i] if p.get(w) != ref[i][w]]
            ctrl = [w for w in ref[0] if p.get(w) != ref[0][w]] if i else []
            bad += bool(diff)
            print(f"{Path(path).name:<28} code {d['code']} pass {i + 1}: "
                  f"{'identical' if not diff else 'DIFFERENT ' + ','.join(diff)} to the first "
                  f"build's pass {i + 1}" + (f" (control, vs its own pass 1: "
                                             f"{','.join(ctrl) or 'identical'})" if i else ""))
    print(f"TEXT across builds, pass by pass ({len(ref[0])} requests): "
          f"{'PASS' if not bad else 'FAIL'}")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--server-arg", action="append", default=[])
    ap.add_argument("--port", type=int, default=8011)
    ap.add_argument("--max-len", type=int, default=262144)
    ap.add_argument("--python", default=str(Path("~/recipes/ling3-flash-dgx-spark/.venv/bin/python")
                                          .expanduser()))
    ap.add_argument("--pythonpath", default=str(Path("~/pylibs").expanduser()))
    ap.add_argument("--compare", nargs="+", default=None)
    a = ap.parse_args()
    if a.compare:
        return compare(a.compare)
    if not a.repo or not a.out:
        ap.error("--repo and --out, or --compare")
    run(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
