"""OPS-124 / SRV-426: LFM2.5-24B-A2B as the box serves it, checked over HTTP. Needs the GPU (a served model);
nothing here loads or switches anything.

    python tools/lfm_live.py --url http://127.0.0.1:8000 --profile lfm25-exl3-3.5 [--json out.json]
    python tools/lfm_live.py --url ... --health-only               /health alone, no request decodes
    python tools/lfm_live.py --url ... --wait 600                  after a restart: wait for /health first

  health      /health is ok, names the profile (weights.name, the version's suffix) and its load-time
              self-check passed                                      (OPS-123 "it serves on :8000")
  answer      a short chat with thinking off answers, finish stop   (OPS-123 "the model answers")
  reasoning   with thinking on, the thinking is apart from the answer: reasoning_content, or one
              closed <think> block at the head of content when the server's format is tags (SRV-425)
  tools       an opencode-shaped request (a system prompt, several tools, a coding ask) gets OpenAI
              tool_calls with JSON arguments, finish tool_calls, and no Pythonic call text in
              content, streamed and not                              (SRV-425 / SRV-426)
  tool_turn   the tool's result sent back gets words, finish stop

--wait N is the restart scenario (OPS-123 "it comes back after a restart"): run the box's start
command, then this with --wait; /health must come back within N seconds and the checks pass.
Every check prints PASS or FAIL with what it saw; the exit code is the number of failures.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

OFF = {"enable_thinking": False}
LEAKS = ("<|tool_call_start|>", "<|tool_call_end|>", "<tool_call>")
SYSTEM = ("You are a coding agent working in the user's repository. Use the tools to read and change "
          "files; call a tool whenever you need to see a file before you answer.")


def _fn(name, desc, props, req):
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": props, "required": req}}}


TOOLS = [
    _fn("bash", "Run a shell command in the repository", {"command": {"type": "string"},
                                                          "timeout": {"type": "integer"}}, ["command"]),
    _fn("read", "Read a file", {"filePath": {"type": "string"}, "offset": {"type": "integer"},
                                "limit": {"type": "integer"}}, ["filePath"]),
    _fn("edit", "Replace a string in a file", {"filePath": {"type": "string"}, "oldString": {"type": "string"},
                                               "newString": {"type": "string"}}, ["filePath", "oldString", "newString"]),
    _fn("glob", "Find files by pattern", {"pattern": {"type": "string"}}, ["pattern"]),
]
ASK = "Read the file src/main.py and tell me what its main() function does."


def _req(url, path, body=None, timeout=600):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(url + path, data=data, headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(r, timeout=timeout)


def _chat(url, model, body, stream=False):
    body = {"model": model, **body, "stream": stream}
    with _req(url, "/v1/chat/completions", body) as r:
        if not stream:
            d = json.loads(r.read())
            c = d["choices"][0]
            m = c["message"]
            return {"content": m.get("content") or "", "reasoning": m.get("reasoning_content") or "",
                    "calls": [{"id": t.get("id"), "name": t["function"]["name"], "args": t["function"]["arguments"]}
                              for t in m.get("tool_calls") or []], "finish": c["finish_reason"]}
        out = {"content": "", "reasoning": "", "calls": {}, "finish": None}
        for ln in r:
            ln = ln.decode().strip()
            if not ln.startswith("data: {"):
                continue
            for ch in json.loads(ln[6:]).get("choices", []):
                d = ch.get("delta") or {}
                out["content"] += d.get("content") or ""
                out["reasoning"] += d.get("reasoning_content") or ""
                for tc in d.get("tool_calls") or []:
                    c = out["calls"].setdefault(tc["index"], {"id": "", "name": "", "args": ""})
                    c["id"] = tc.get("id") or c["id"]
                    c["name"] += (tc.get("function") or {}).get("name") or ""
                    c["args"] += (tc.get("function") or {}).get("arguments") or ""
                out["finish"] = ch.get("finish_reason") or out["finish"]
        out["calls"] = [out["calls"][k] for k in sorted(out["calls"])]
        return out


def health(url, wait=0.0):
    t0 = time.time()
    while True:
        try:
            with _req(url, "/health", timeout=10) as r:
                d = json.loads(r.read())
            if d.get("status") == "ok":
                return d, time.time() - t0
        except (urllib.error.URLError, OSError, ValueError):
            pass
        if time.time() - t0 > wait:
            return None, time.time() - t0
        time.sleep(2)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--profile", default="", help="the profile /health must name (default: any lfm25-*)")
    ap.add_argument("--health-only", action="store_true")
    ap.add_argument("--wait", type=float, default=0.0, help="seconds to wait for /health (after a restart)")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    results = []

    def check(name, ok, **seen):
        results.append({"check": name, "pass": bool(ok), **seen})
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {json.dumps(seen, ensure_ascii=False)[:400]}", flush=True)

    h, waited = health(a.url, a.wait)
    w = (h or {}).get("weights") or {}
    name = w.get("name") or ""
    want = a.profile or name
    check("health", h is not None and name == want and name.startswith("lfm25-")
          and str(h.get("version", "")).endswith("+" + want) and (w.get("self_check") or {}).get("ok") is True,
          profile=name, version=(h or {}).get("version"), self_check=w.get("self_check"), waited_s=round(waited, 1))
    if h is None or a.health_only:
        return _end(results, a.json)
    model = h.get("model") or "TandemLLM"

    r = _chat(a.url, model, {"messages": [{"role": "user", "content": "What is 17 * 23? Answer with the number."}],
                             "max_tokens": 64, "chat_template_kwargs": OFF})
    check("answer", "391" in r["content"] and r["finish"] == "stop" and not r["reasoning"],
          content=r["content"][:80], finish=r["finish"])

    r = _chat(a.url, model, {"messages": [{"role": "user", "content": "Is 91 a prime number? Answer briefly."}],
                             "max_tokens": 2048})
    if r["reasoning"]:
        apart = "<think>" not in r["content"] and bool(r["content"].strip())
    else:
        head, sep, tail = r["content"].partition("</think>")
        apart = head.lstrip().startswith("<think>") and bool(sep) and bool(tail.strip()) and "<think>" not in tail
    check("reasoning", apart and r["finish"] == "stop", reasoning_chars=len(r["reasoning"]),
          content=r["content"][-120:], finish=r["finish"])

    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": ASK}]
    first = None
    for stream in (False, True):
        r = _chat(a.url, model, {"messages": msgs, "tools": TOOLS, "max_tokens": 4096}, stream=stream)
        calls = r["calls"]
        ok_args = True
        for c in calls:
            try:
                json.loads(c["args"])
            except ValueError:
                ok_args = False
        leak = [x for x in LEAKS if x in r["content"]]
        check("tools_stream" if stream else "tools", bool(calls) and ok_args and r["finish"] == "tool_calls" and not leak
              and all(c["name"] in {t["function"]["name"] for t in TOOLS} for c in calls),
              calls=calls[:3], finish=r["finish"], leak=leak)
        if not stream and calls:
            first = calls[0]
    if first:
        turn = msgs + [{"role": "assistant", "content": None, "tool_calls": [
            {"id": first["id"] or "call_0", "type": "function", "function": {"name": first["name"], "arguments": first["args"]}}]},
            {"role": "tool", "tool_call_id": first["id"] or "call_0",
             "content": "def main():\n    print(sum(range(10)))\n\nif __name__ == '__main__':\n    main()\n"}]
        r = _chat(a.url, model, {"messages": turn, "tools": TOOLS, "max_tokens": 4096})
        check("tool_turn", bool(r["content"].strip()) and r["finish"] in ("stop", "tool_calls")
              and not [x for x in LEAKS if x in r["content"]], content=r["content"][-160:], finish=r["finish"])
    return _end(results, a.json)


def _end(results, path) -> int:
    fails = sum(1 for r in results if not r["pass"])
    print(f"{len(results) - fails} passed" + (f", {fails} FAILED" if fails else ""))
    if path:
        json.dump(results, open(path, "w"), indent=1)
    return fails


if __name__ == "__main__":
    sys.exit(main())
