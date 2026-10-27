"""SRV-87 / ENG-235 on the box: Flash-Next behind server/app.py, checked through the API.

    python tools/fn_server_check.py --url http://127.0.0.1:8011 --out run-mtp.json
    python tools/fn_server_check.py --compare run-none.json run-mtp.json

One set of requests against a running server (stdlib only): /health and /v1/models; greedy answers;
thinking on with the reasoning split; a streamed answer (chunks, the reasoning field, usage); tool
calls with tool_choice auto and forced; a JSON response_format; a presence penalty with a logit_bias
entry; logprobs with top_logprobs 5; sampled requests with a seed (twice each: a seeded request
repeats). Every answer is saved. `--compare` takes two runs of the same build -- the server started
with `--drafter none` (plain decode) and with `--drafter mtp` (speculation) -- and checks that every
greedy and seeded answer is the same text, every tool call the same call, and the logprobs the same
within bf16 noise: the losslessness of speculation through the whole request path.

    python tools/fn_server_check.py --sampled --url ... --out sampled-mtp.json      # ENG-237

`--sampled`: the sampled requests alone, longer -- prose, code and chat after about 1k and 8k tokens of
held-out text (bench/), temperature 0.7, top_p 0.9, top_k 20, min_p 0.05, each with a seed and sent
twice; one more pair of the first case without a seed (two draws, expected to differ). `--compare` of two
such runs checks every seeded answer the same text with and without speculation (the det walk draws by
seed and position, so the draft cannot change a draw); each answer keeps its `timings` (tokens a block,
tok/s).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "Current weather for a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "unit": {"type": "string",
                                                                                         "enum": ["c", "f"]}},
                   "required": ["city"]}}}]

CASES = {
    "greedy_capital": {"messages": [{"role": "user", "content": "What is the capital of Italy? One sentence."}],
                       "max_tokens": 64},
    "greedy_code": {"messages": [{"role": "user", "content": "Write a Python function that returns the n-th "
                                  "Fibonacci number iteratively, with a docstring."}], "max_tokens": 200},
    "greedy_prose": {"messages": [{"role": "user", "content": "Explain in four sentences why the sky is blue."}],
                     "max_tokens": 160},
    "thinking_on": {"messages": [{"role": "user", "content": "Is 391 a prime number? Answer yes or no with the "
                                  "reason."}], "max_tokens": 600,
                    "chat_template_kwargs": {"enable_thinking": True}, "reasoning_effort": "low"},
    "stream_thinking": {"messages": [{"role": "user", "content": "Name three rivers in Europe."}],
                        "max_tokens": 400, "stream": True, "stream_options": {"include_usage": True},
                        "chat_template_kwargs": {"enable_thinking": True}},
    "tool_auto": {"messages": [{"role": "user", "content": "What's the weather in Berlin right now?"}],
                  "tools": TOOLS, "max_tokens": 200},
    "tool_forced": {"messages": [{"role": "user", "content": "Tell me about Paris."}], "tools": TOOLS,
                    "tool_choice": {"type": "function", "function": {"name": "get_weather"}}, "max_tokens": 200},
    "json_format": {"messages": [{"role": "user", "content": "Give a JSON object with the keys name and age for "
                                  "a fictional person."}], "response_format": {"type": "json_object"},
                    "max_tokens": 120},
    "penalty_bias": {"messages": [{"role": "user", "content": "List ten colours, one per line."}],
                     "presence_penalty": 0.8, "logit_bias": {"13": -5}, "max_tokens": 120},
    "logprobs": {"messages": [{"role": "user", "content": "Count from one to ten in words."}],
                 "logprobs": True, "top_logprobs": 5, "max_tokens": 60},
    "sampled_seed": {"messages": [{"role": "user", "content": "Write a two-line poem about the sea."}],
                     "temperature": 0.8, "top_p": 0.95, "seed": 1234, "max_tokens": 80},
    "sampled_seed_code": {"messages": [{"role": "user", "content": "Write a short bash loop that prints 1 to 5."}],
                          "temperature": 0.6, "seed": 99, "max_tokens": 80},
    # ENG-235: three prompts for each transform, the think budget at its cap, a stop string inside a round
    "tool_forced_2": {"messages": [{"role": "user", "content": "I am packing for Oslo."}], "tools": TOOLS,
                      "tool_choice": {"type": "function", "function": {"name": "get_weather"}}, "max_tokens": 200},
    "tool_forced_3": {"messages": [{"role": "user", "content": "Plan a picnic in Rome for tomorrow."}], "tools": TOOLS,
                      "tool_choice": "required", "max_tokens": 200},
    "json_format_2": {"messages": [{"role": "user", "content": "A JSON array of three fruits with their colours."}],
                      "response_format": {"type": "json_object"}, "max_tokens": 150},
    "json_format_3": {"messages": [{"role": "user", "content": "Describe a car as JSON with make, model and year."}],
                      "response_format": {"type": "json_object"}, "max_tokens": 150},
    "penalty_bias_2": {"messages": [{"role": "user", "content": "Write a short paragraph about cats."}],
                       "frequency_penalty": 0.5, "logit_bias": {"279": -3, "264": 2}, "max_tokens": 120},
    "penalty_bias_3": {"messages": [{"role": "user", "content": "Write a haiku about winter."}],
                       "presence_penalty": 1.2, "logit_bias": {"11": -10}, "max_tokens": 80},
    "think_low_cap": {"messages": [{"role": "user", "content": "Find all integer solutions of x^2 + y^2 = 2025 and "
                                    "explain every step of your search carefully."}],
                      "chat_template_kwargs": {"enable_thinking": True}, "reasoning_effort": "low", "max_tokens": 2400},
    "stop_string": {"messages": [{"role": "user", "content": "Count from one to twenty in words, comma separated."}],
                    "stop": ["seven"], "max_tokens": 120},
}
REPEAT = ("sampled_seed", "sampled_seed_code")


def sampled_cases() -> dict:
    """ENG-237: one sampled request a class at about 1k and 8k tokens of context (3.6 characters a token
    of the held-out text), seeded."""
    sys.path.insert(0, ROOT)
    from tools.exl3_fastrun import CHAT, CODE, PROSE
    names = ("heldout_prose.txt", "heldout_code.txt", "calib.txt")
    texts = {n: open(os.path.join(ROOT, "bench", n)).read() for n in names}
    out = {}
    for cls, ask, first in (("prose", PROSE, names[0]), ("code", CODE, names[1]), ("chat", CHAT, names[0])):
        fill = "\n\n".join([texts[first]] + [texts[n] for n in names if n != first])
        for k, chars in (("1k", 3_600), ("8k", 29_000)):
            out[f"sampled_{cls}_{k}"] = {
                "messages": [{"role": "user", "content": (fill * 2)[:chars] + "\n\n" + ask}],
                "temperature": 0.7, "top_p": 0.9, "top_k": 20, "min_p": 0.05, "seed": 4242, "max_tokens": 256,
                "chat_template_kwargs": {"enable_thinking": False}}
    return out


def call(url: str, path: str, body: dict | None = None, timeout: float = 600.0):
    req = urllib.request.Request(url + path, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        if body is not None and body.get("stream"):
            chunks, text, reason, usage, finish = 0, "", "", None, None
            for line in r:
                line = line.decode().strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                ev = json.loads(line[6:])
                if ev.get("usage"):
                    usage = ev["usage"]
                for ch in ev.get("choices", []):
                    d = ch.get("delta", {})
                    text += d.get("content") or ""
                    reason += d.get("reasoning_content") or ""
                    finish = ch.get("finish_reason") or finish
                    chunks += 1
            return {"stream": True, "chunks": chunks, "content": text, "reasoning_content": reason,
                    "usage": usage, "finish_reason": finish}
        return json.loads(r.read())


def run(url: str, cases: dict, repeat) -> dict:
    out = {"health": call(url, "/health"), "models": call(url, "/v1/models"), "answers": {}}
    for name, body in cases.items():
        body = {"model": "TandemLLM", **body}
        t0 = time.time()
        r = call(url, "/v1/chat/completions", body)
        r["_s"] = round(time.time() - t0, 2)
        if name in repeat:
            r["_again"] = call(url, "/v1/chat/completions", body)
        out["answers"][name] = r
        if name == "think_low_cap":
            u = r.get("usage") or {}
            r["_reasoning_tokens"] = (u.get("completion_tokens_details") or {}).get("reasoning_tokens")
        msg = r.get("choices", [{}])[0].get("message", {}) if not r.get("stream") else r
        t = r.get("timings") or {}
        print(f"[check] {name}: {r['_s']} s, finish {r.get('finish_reason') or r.get('choices', [{}])[0].get('finish_reason')}, "
              f"tools {bool(msg.get('tool_calls'))}, reasoning {len(msg.get('reasoning_content') or '')} chars, "
              f"prompt {t.get('prompt_n')}, {t.get('predicted_per_second')} tok/s, {t.get('tokens_per_block')} a block, "
              f"text {(msg.get('content') or '')[:80]!r}", flush=True)
    return out


def unseeded(url: str, body: dict) -> list:
    """Two draws of one request without its seed."""
    body = {"model": "TandemLLM", **{k: v for k, v in body.items() if k != "seed"}}
    return [call(url, "/v1/chat/completions", body) for _ in range(2)]


def _main_text(r: dict) -> tuple:
    if r.get("stream"):
        return (r["content"], r["reasoning_content"], None)
    m = r["choices"][0]["message"]
    calls = [c["function"] for c in (m.get("tool_calls") or [])]        # a call's id is random
    return (m.get("content"), m.get("reasoning_content"), json.dumps(calls, sort_keys=True))


def compare(a: dict, b: dict) -> dict:
    res, ok = {}, True
    for name in a["answers"]:
        ra, rb = a["answers"][name], b["answers"][name]
        same = _main_text(ra) == _main_text(rb)
        row = {"same": same}
        if "_again" in ra:
            row["seed_repeats"] = _main_text(rb) == _main_text(rb["_again"]) and _main_text(ra) == _main_text(ra["_again"])
            same = same and row["seed_repeats"]
        if name == "logprobs":
            la = [t["logprob"] for t in ra["choices"][0]["logprobs"]["content"]]
            lb = [t["logprob"] for t in rb["choices"][0]["logprobs"]["content"]]
            row["logprob_max_diff"] = max((abs(x - y) for x, y in zip(la, lb)), default=0.0)
            row["logprob_tokens"] = [len(la), len(lb)]
        ok = ok and same
        res[name] = row
    for run_, side in ((a, "a"), (b, "b")):
        if run_.get("unseeded"):
            res[f"unseeded_{side}_differ"] = _main_text(run_["unseeded"][0]) != _main_text(run_["unseeded"][1])
    return {"all_same": ok, "cases": res}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8011")
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs=2)
    ap.add_argument("--sampled", action="store_true", help="ENG-237: the long seeded sampled requests alone")
    a = ap.parse_args()
    if a.compare:
        r = compare(json.load(open(a.compare[0])), json.load(open(a.compare[1])))
        print(json.dumps(r, indent=1))
        sys.exit(0 if r["all_same"] else 1)
    if a.sampled:
        cases = sampled_cases()
        out = run(a.url, cases, tuple(cases))
        out["unseeded"] = unseeded(a.url, next(iter(cases.values())))
    else:
        out = run(a.url, CASES, REPEAT)
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
