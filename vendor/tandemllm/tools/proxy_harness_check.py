"""The harness (proxy P4) against a live engine: a standalone proxy in this process, its own
conductor in a temp directory, the engine registered as a `tandemllm` backend with the harness on.

    python tools/proxy_harness_check.py --url http://<engine host>:8000/v1 [--out result.json]

Three small requests, each at most 64 output tokens, thinking off by the profile's effort or the
side cap; the engine sees them as raw requests (`X-Tandem: raw`), so its own conductor does not run
on top:

    1 a coding-shaped first turn with tools: the class question went to the engine (stage qwen,
      model `<backend>/<model>`), a note was added to the user message, one stream, one [DONE]
    2 the next turn of the same conversation: the note replayed at the same index, and the
      engine's own prefix reuse reported (`cached_tokens` above zero when its caches are on)
    3 a client's title request: capped at 64 tokens with thinking off

Prints one line per check and writes the result as JSON. For the night pass (tools/proxy_e2e.sh)
and a quick smoke before a deploy; it never needs a key and makes no paid call.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "tests")]

from proxy_util import Proxy  # noqa: E402

from proxy import app as proxy_app  # noqa: E402

UA = {"User-Agent": "opencode/1.18.34 ai-sdk/provider-utils/3"}
TOOLS = [{"type": "function", "function": {"name": "read", "description": "Read a file",
                                            "parameters": {"type": "object", "properties": {
                                                "filePath": {"type": "string"}},
                                                "required": ["filePath"]}}}]
SYSTEM = {"role": "system", "content": "You are a coding agent. Working directory: /tmp/harness"}


def frames(raw: bytes) -> list:
    out = []
    for ev in raw.decode("utf-8", "replace").split("\n\n"):
        ev = ev.strip()
        if ev.startswith("data:"):
            d = ev[5:].strip()
            out.append(d if d == "[DONE]" else json.loads(d))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="the engine's base URL, e.g. http://host:8000/v1")
    ap.add_argument("--model", default="TandemLLM")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    env = {"TANDEM_BACKENDS": "engine", "TANDEM_BACKEND_ENGINE_KIND": "tandemllm",
           "TANDEM_BACKEND_ENGINE_URL": a.url, "TANDEM_BACKEND_ENGINE_HARNESS": "on",
           "TANDEM_BACKEND_ENGINE_MODELS": a.model, "TANDEM_BACKEND_ENGINE_MODELS_MODE": "list",
           "TANDEM_BACKEND_ENGINE_HARNESS_PROFILE": "coding"}
    res: dict = {"url": a.url, "at": time.strftime("%Y-%m-%dT%H:%M:%S"), "checks": {}}
    ok_all = True

    def check(name, ok, **info):
        nonlocal ok_all
        ok_all &= bool(ok)
        res["checks"][name] = dict(info, ok=bool(ok))
        print(f"{'PASS' if ok else 'FAIL'} {name} {json.dumps(info)[:300]}", flush=True)

    with Proxy(env=env, conductor=True) as px:
        px.sync()
        u1 = {"role": "user", "content": "Fix the bug in utils.py: add() returns a - b instead of "
                                          "a + b. Edit the file, then run the tests."}
        body = {"model": a.model, "messages": [SYSTEM, u1], "tools": TOOLS, "stream": True,
                "max_tokens": 64, "stream_options": {"include_usage": True}}
        t0 = time.time()
        st, _, out = px.post("/v1/chat/completions", body, headers=UA, timeout=300)
        fr = frames(out)
        cond = proxy_app.STATE["conductor"]
        cond.journal.flush()
        rows = [r for r in cond.journal.rows(limit=10) if r.get("kind") == "route"]
        r1 = rows[0] if rows else {}
        stages = r1.get("stages") or []
        check("first_turn", st == 200 and fr.count("[DONE]") == 1 and
              len({f["id"] for f in fr if f != "[DONE]"}) == 1 and bool(r1.get("injected")),
              status=st, seconds=round(time.time() - t0, 1), class_id=r1.get("class_id"),
              source=r1.get("source"), stages=[(s.get("stage"), s.get("model"), s.get("error"))
                                               for s in stages],
              injected=len(r1.get("injected") or []))
        text = "".join(((c.get("delta") or {}).get("content") or "")
                       for f in fr if f != "[DONE]" for c in f.get("choices") or [])
        msgs = [SYSTEM, u1, {"role": "assistant", "content": text or "ok"},
                {"role": "user", "content": "The tests still fail on add(2, 2). Fix it."}]
        st2, _, out2 = px.post("/v1/chat/completions", dict(body, messages=msgs), headers=UA,
                               timeout=300)
        fr2 = frames(out2)
        usage = [f.get("usage") for f in fr2 if f != "[DONE]" and f.get("usage")]
        cached = ((usage[-1] or {}).get("prompt_tokens_details") or {}).get("cached_tokens") \
            if usage else None
        cond.journal.flush()
        rows = [r for r in cond.journal.rows(limit=10) if r.get("kind") == "route"]
        rep = (rows[0].get("replayed") or {}) if rows else {}
        check("replay", st2 == 200 and (rep.get("count") or 0) >= 1, status=st2,
              replayed=rep, cached_tokens=cached)
        title = {"model": a.model, "max_tokens": 4000, "stream": False, "messages": [
            {"role": "system", "content": "You are a title generator. You output ONLY a thread "
                                          "title. Nothing else."},
            {"role": "user", "content": "Read utils.py and explain add"}]}
        st3, _, out3 = px.post("/v1/chat/completions", title, headers=UA, timeout=120)
        rows = px.rows()
        side = [r for r in rows if r.get("client_kind") != "harness"][-1] if rows else {}
        check("side_title", st3 == 200 and (side.get("completion_tokens") or 0) <= 64,
              status=st3, completion_tokens=side.get("completion_tokens"))
        res["ledger_harness_rows"] = sum(1 for r in rows if r.get("client_kind") == "harness")
    res["ok"] = ok_all
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
