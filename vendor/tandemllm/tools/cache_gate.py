"""What the serving-time caches are worth, and what they cost in exactness. Against a live server.

Four measurements and one gate, all through `POST /v1/chat/completions` on a running
`server/app.py`, because a cache that only works when the engine is driven from Python is not a
cache this engine has.

  a. multi-turn   -- a four-turn conversation of about 1k tokens a turn, cold then warm. The
                     number is TTFT per turn, and the interesting column is the fourth.
  b. prefix       -- one 2k system prompt and ten different user messages. The first pays for the
                     system prompt and the other nine do not.
  c. exactness    -- the gate. The same request cold and warm, greedy, 64 new tokens: the token
                     ids must be identical. Three prompts, and the run reports the first position
                     where they part if they do.
  d. response     -- the same request twice with `--response-cache` on: hit latency.

Every stage reads `/v1/cache/stats` before and after, so the memory accounting in the table is the
server's own and not this file's arithmetic.

  python tools/cache_gate.py --url http://127.0.0.1:8000 --stage all

The server has to be started with the caches under test. `--stage exact` is the one that gates:
run it before believing any of the others.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request


def post(url: str, path: str, body: dict, timeout: float = 600.0) -> dict:
    req = urllib.request.Request(url + path, method="POST",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def get(url: str, path: str) -> dict:
    with urllib.request.urlopen(url + path, timeout=30) as r:
        return json.loads(r.read())


def stream(url: str, body: dict, timeout: float = 600.0):
    """Yield (t_since_request, piece). The first pair is the time to first token."""
    body = dict(body, stream=True, stream_options={"include_usage": True})
    req = urllib.request.Request(url + "/v1/chat/completions", method="POST",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                return
            obj = json.loads(payload)
            for ch in obj.get("choices") or []:
                piece = (ch.get("delta") or {}).get("content")
                if piece:
                    yield time.perf_counter() - t0, piece


def ttft_ms(url: str, messages: list[dict], max_tokens: int, conv: str | None = None) -> tuple:
    """Time to first token, and the whole answer, for one streamed request."""
    body = {"messages": messages, "max_tokens": max_tokens, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}}
    if conv:
        body["conversation_id"] = conv
    first, text = None, []
    t0 = time.perf_counter()
    for t, piece in stream(url, body):
        if first is None:
            first = t
        text.append(piece)
    return (first or 0.0) * 1e3, (time.perf_counter() - t0) * 1e3, "".join(text)


def ids_of(url: str, messages: list[dict], max_tokens: int) -> tuple[list[int], str]:
    """The answer as text plus its usage, non-streamed -- the form the gate compares."""
    body = {"messages": messages, "max_tokens": max_tokens, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}}
    r = post(url, "/v1/chat/completions", body)
    return r["usage"], r["choices"][0]["message"]["content"]


def words(n: int, seed: int = 0) -> str:
    """Filler with no structure a lookup drafter could exploit, so the prompt costs what it says."""
    import random
    rng = random.Random(seed)
    vocab = ["ledger", "boundary", "recurrent", "convolution", "rollback", "prefill", "verify",
             "accepted", "drafter", "budget", "kernel", "unified", "bandwidth", "snapshot",
             "checkpoint", "greedy", "argmax", "residual", "quantise", "throughput"]
    return " ".join(rng.choice(vocab) for _ in range(n))


def clear(url: str) -> None:
    post(url, "/v1/cache/clear", {})


# ------------------------------------------------------------------ a. multi-turn
def stage_multiturn(url: str, turns: int, per_turn: int, max_tokens: int) -> None:
    print(f"\n### a. multi-turn: {turns} turns, ~{per_turn} words of new user text a turn\n")
    for label in ("cold (cache cleared before every turn)", "warm (cache kept)"):
        clear(url)
        msgs: list[dict] = [{"role": "system", "content": "You are terse."}]
        row = []
        for i in range(turns):
            if label.startswith("cold"):
                clear(url)
            msgs.append({"role": "user",
                         "content": f"Turn {i}. Summarise: " + words(per_turn, seed=i)})
            ms, whole, text = ttft_ms(url, msgs, max_tokens, conv="gate-multiturn")
            st = get(url, "/v1/cache/stats")["last_prefill"]
            row.append((ms, st["reused"], st["forwarded"]))
            msgs.append({"role": "assistant", "content": text})
        print(f"{label:38s} " + "  ".join(
            f"t{i}: {ms:7.1f} ms ({r}+{f})" for i, (ms, r, f) in enumerate(row)))


# ------------------------------------------------------------------ b. prefix cache
def stage_prefix(url: str, system_words: int, n_users: int, max_tokens: int) -> None:
    print(f"\n### b. prefix cache: one ~{system_words}-word system prompt, {n_users} user "
          f"messages\n")
    system = {"role": "system", "content": "Context you must use: " + words(system_words, 99)}
    for label in ("cold (cache cleared before each)", "warm (shared prefix kept)"):
        clear(url)
        out = []
        for i in range(n_users):
            if label.startswith("cold"):
                clear(url)
            msgs = [system, {"role": "user", "content": f"Question {i}: name one word above."}]
            ms, _, _ = ttft_ms(url, msgs, max_tokens)
            st = get(url, "/v1/cache/stats")["last_prefill"]
            out.append((ms, st["reused"], st["forwarded"]))
        first = out[0][0]
        rest = [m for m, _, _ in out[1:]]
        print(f"{label:34s} first {first:8.1f} ms   rest mean {sum(rest)/len(rest):8.1f} ms   "
              f"min {min(rest):8.1f}   reused/forwarded on the last "
              f"{out[-1][1]}/{out[-1][2]}")
    print(json.dumps(get(url, "/v1/cache/stats")["state_store"], indent=2))


# ------------------------------------------------------------------ c. the exactness gate
GATE_PROMPTS = [
    "Explain in three sentences why a recurrent state makes speculative rollback cheap.",
    "Write a Python function that returns the longest common prefix of two lists.",
    "Nenne drei Gruende, warum ein Cache die Ausgabe nicht veraendern darf.",
]


def stage_exact(url: str, max_tokens: int) -> int:
    print(f"\n### c. exactness gate: cold vs warm, greedy, {max_tokens} new tokens\n")
    bad = 0
    for i, p in enumerate(GATE_PROMPTS):
        system = {"role": "system", "content": "Context: " + words(400, 7)}
        msgs = [system, {"role": "user", "content": p}]
        clear(url)
        u1, a = ids_of(url, msgs, max_tokens)
        # warm: the same system prompt is now in the store, reached through a different request
        ids_of(url, [system, {"role": "user", "content": "Say ok."}], 4)
        u2, b = ids_of(url, msgs, max_tokens)
        st = get(url, "/v1/cache/stats")["last_prefill"]
        same = a == b
        bad += 0 if same else 1
        mark = "PASS" if same else "FAIL"
        if not same:
            k = next((j for j in range(min(len(a), len(b))) if a[j] != b[j]), min(len(a), len(b)))
            mark += f"  first difference at character {k}: {a[k:k+40]!r} vs {b[k:k+40]!r}"
        print(f"  prompt {i}  cold {u1['completion_tokens']:4d} tok   warm "
              f"{u2['completion_tokens']:4d} tok   reused {st['reused']}/"
              f"{st['reused'] + st['forwarded']}   {mark}")
    print(f"\n  {len(GATE_PROMPTS) - bad}/{len(GATE_PROMPTS)} identical")
    return bad


# ------------------------------------------------------------------ d. response cache
def stage_response(url: str, max_tokens: int) -> None:
    print(f"\n### d. exact-prompt response cache ({max_tokens} new tokens)\n")
    clear(url)
    msgs = [{"role": "user", "content": "List the first eight prime numbers."}]
    a_ms, a_whole, a_text = ttft_ms(url, msgs, max_tokens)
    b_ms, b_whole, b_text = ttft_ms(url, msgs, max_tokens)
    print(f"  miss  TTFT {a_ms:8.1f} ms   whole {a_whole:8.1f} ms")
    print(f"  hit   TTFT {b_ms:8.1f} ms   whole {b_whole:8.1f} ms   "
          f"speedup {a_whole / max(b_whole, 1e-6):6.1f}x")
    print(f"  identical text: {a_text == b_text}")
    rc = get(url, "/v1/cache/stats")["response_cache"]
    print("  " + json.dumps(rc) if rc else "  response cache is OFF "
          "(start the server with --response-cache)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--stage", default="all",
                    choices=("all", "multiturn", "prefix", "exact", "response", "stats"))
    ap.add_argument("--turns", type=int, default=4)
    ap.add_argument("--per-turn", type=int, default=700)
    ap.add_argument("--system-words", type=int, default=1400)
    ap.add_argument("--users", type=int, default=10)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--gate-tokens", type=int, default=64)
    a = ap.parse_args()

    try:
        get(a.url, "/health")
    except urllib.error.URLError as exc:
        raise SystemExit(f"no server at {a.url}: {exc}")

    if a.stage == "stats":
        print(json.dumps(get(a.url, "/v1/cache/stats"), indent=2))
        return
    bad = 0
    if a.stage in ("all", "exact"):
        bad = stage_exact(a.url, a.gate_tokens)
    if a.stage in ("all", "multiturn"):
        stage_multiturn(a.url, a.turns, a.per_turn, a.max_tokens)
    if a.stage in ("all", "prefix"):
        stage_prefix(a.url, a.system_words, a.users, a.max_tokens)
    if a.stage in ("all", "response"):
        stage_response(a.url, a.max_tokens)
    print("\n### memory\n" + json.dumps(get(a.url, "/v1/cache/stats"), indent=2))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
