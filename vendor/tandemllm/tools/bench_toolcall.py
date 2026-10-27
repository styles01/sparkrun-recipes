"""Tool calling, measured. A fixed tool set, a fixed scenario matrix, three numbers.

The client is the official `openai` package, so what is measured is what an OpenAI client receives
-- through its own parsing of the JSON body and of the stream -- not what the server meant to send.
The tools are synthetic and never run: the harness plays the tool itself in the round-trip scenario.

    # a test server (one engine on the box, under the hold), then from any machine with `openai`:
    ssh -N -L 8011:127.0.0.1:8011 <engine-host> &
    python tools/bench_toolcall.py --base http://127.0.0.1:8011/v1 --think off,on \\
        --json results/api/toolcall-<label>.json
    python tools/bench_toolcall.py --read results/api/toolcall-<label>.json

The three numbers, per scenario and over all of them:

  * **parse** -- the answer's calls are exactly the expected NUMBER of calls and no `<tool_call>`
    text is left in the content (a block the parser could not read shows up there);
  * **name** -- the multiset of called functions is the expected one;
  * **args** -- argument fidelity: of the expected parameters, the share whose value is exactly the
    expected string (calls matched to expectations by name, then by best overlap); every miss is
    written down with both values, so a near miss (a trailing newline, a quote) can be read.

`tool_choice` compliance is its own table: over a fixed prompt set, `none` must produce no call,
`required` at least one, a named function only that one; `auto` is reported as the call rate on
the prompts that need a tool and on the ones that do not.

Greedy requests are deterministic, so one run a prompt is the measurement; `--repeats N
--temperature T` repeats each request for rates under sampling.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path


def _tool(name: str, desc: str, **props: str) -> dict:
    return {"type": "function", "function": {
        "name": name, "description": desc,
        "parameters": {"type": "object",
                       "properties": {k: {"type": "string", "description": v}
                                      for k, v in props.items()},
                       "required": list(props)}}}


TOOLS = [
    _tool("write_file", "Write text to a file, replacing it.", path="absolute path",
          content="the whole text of the file"),
    _tool("read_file", "Read a text file.", path="absolute path"),
    _tool("list_dir", "List the entries of a directory.", path="absolute path"),
    _tool("run", "Run a shell command and return its output.", cmd="the command line"),
]


# tools whose parameters are not strings. The Qwen XML format writes every value as text;
# the server returns it as the schema's type, and a client that validates (opencode) refuses the
# text. Used only by the typed/* scenarios, so the string-only scenarios keep their prompt.
TYPED_TOOLS = TOOLS + [
    {"type": "function", "function": {
        "name": "read_lines", "description": "Read a range of lines of a text file.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "absolute path"},
            "offset": {"type": "integer", "description": "first line, 1-based"},
            "limit": {"type": "integer", "description": "how many lines"},
            "follow": {"type": "boolean", "description": "keep reading as the file grows"}},
            "required": ["path", "offset", "limit"]}}},
    {"type": "function", "function": {
        "name": "tag_items", "description": "Attach tags and a weight to a set of items.",
        "parameters": {"type": "object", "properties": {
            "ids": {"type": "array", "items": {"type": "integer"}},
            "tags": {"type": "array", "items": {"type": "string"}},
            "weight": {"type": "number"}}, "required": ["ids", "tags", "weight"]}}},
]


def call(name: str, **args) -> tuple[str, dict]:
    return name, args


# name -> (messages, expected calls, request extras). The prompts say what to call with and give
# every argument literally, so a miss is the engine's (or the model's) and not an ambiguity.
SCENARIOS = {
    "single/read": ([{"role": "user", "content": "Read the file /etc/hostname."}],
                    [call("read_file", path="/etc/hostname")], {}),
    "single/list": ([{"role": "user", "content": "List the directory /var/log."}],
                    [call("list_dir", path="/var/log")], {}),
    "single/run": ([{"role": "user", "content": "Run the command `uname -a`."}],
                   [call("run", cmd="uname -a")], {}),
    "single/write": ([{"role": "user", "content":
                       "Write the text `hello world` (without the backticks) to /tmp/greeting.txt."}],
                     [call("write_file", path="/tmp/greeting.txt", content="hello world")], {}),
    "parallel/write2": ([{"role": "user", "content":
                          "In one reply, create two files: /tmp/a.txt containing `alpha` and "
                          "/tmp/b.txt containing `beta` (without backticks). Call write_file once "
                          "for each file, both in this reply."}],
                        [call("write_file", path="/tmp/a.txt", content="alpha"),
                         call("write_file", path="/tmp/b.txt", content="beta")], {}),
    "parallel/read2": ([{"role": "user", "content":
                         "Read /etc/hosts and /etc/resolv.conf. Call read_file for both files in "
                         "this one reply."}],
                       [call("read_file", path="/etc/hosts"),
                        call("read_file", path="/etc/resolv.conf")], {}),
    "parallel/capped": ([{"role": "user", "content":
                          "Read /etc/hosts and /etc/resolv.conf. Call read_file for both files in "
                          "this one reply."}],
                        [call("read_file", path="/etc/hosts")], {"parallel_tool_calls": False}),
    "json/form": ([{"role": "system", "content":
                    "When you use a function, reply with ONLY a JSON object of the form "
                    '{"name": <function name>, "arguments": {...}} and nothing else, no XML tags.'},
                   {"role": "user", "content": "Read the file /etc/hostname."}],
                  [call("read_file", path="/etc/hostname")], {}),
    "json/answer": ([{"role": "user", "content":
                      'Reply with only a JSON object with the keys "name" and "age" for a person '
                      "called Ann who is 31. Do not call any function."}], [], {}),
    "typed/read_lines": ([{"role": "user", "content":
                           "Read lines 40 to 49 of /var/log/syslog: offset 40, limit 10, and do "
                           "not follow the file."}],
                         [call("read_lines", path="/var/log/syslog", offset=40, limit=10,
                               follow=False)], {"tools": TYPED_TOOLS}),
    "typed/tag_items": ([{"role": "user", "content":
                          "Tag the items 3, 5 and 8 with the tags red and urgent, weight 0.5."}],
                        [call("tag_items", ids=[3, 5, 8], tags=["red", "urgent"], weight=0.5)],
                        {"tools": TYPED_TOOLS}),
}

ROUND_TRIP = {
    "ask": [{"role": "user", "content": "What is the first line of /data/notes.txt? Use a tool."}],
    "expect": call("read_file", path="/data/notes.txt"),
    "result": "The launch code is 7421.",
    "answer_contains": "7421",
}

# (label, prompt, needs a tool) -- the tool_choice compliance set
CHOICE_PROMPTS = [
    ("math", "What is 17 times 3?", False),
    ("hello", "Say hello in French.", False),
    ("define", "In one sentence, what is a hash map?", False),
    ("read", "Show me what is in /etc/hostname.", True),
    ("list", "Which files are in /tmp?", True),
    ("run", "What does `date` print right now?", True),
]
NAMED = "list_dir"


# ------------------------------------------------------------------------------ the client side

def _client(base: str, timeout: float):
    try:
        from openai import OpenAI
    except ImportError:
        raise SystemExit("bench_toolcall needs the `openai` package (pip install openai)")
    return OpenAI(base_url=base, api_key="none", timeout=timeout, max_retries=0)


def ask(client, model: str, messages: list, stream: bool, extra: dict, a) -> dict:
    """One request: the calls and content the client ends up with, and how long it took."""
    extra = dict(extra)
    body = dict(model=model, messages=messages, tools=extra.pop("tools", TOOLS),
                max_tokens=a.max_tokens, temperature=a.temperature, **extra)
    kw = {"extra_body": {"chat_template_kwargs": {"enable_thinking": a.cur_think}}}
    if a.cur_think and a.effort:
        kw["extra_body"]["reasoning_effort"] = a.effort
    t0 = time.perf_counter()
    if not stream:
        r = client.chat.completions.create(**body, **kw)
        c = r.choices[0]
        calls = [(t.function.name, t.function.arguments) for t in c.message.tool_calls or []]
        content, finish = c.message.content or "", c.finish_reason
        usage = r.usage.completion_tokens if r.usage else None
    else:
        parts, content, finish, usage = {}, "", None, None
        for ch in client.chat.completions.create(**body, stream=True, **kw):
            if getattr(ch, "usage", None):
                usage = ch.usage.completion_tokens
            for c in ch.choices:
                content += c.delta.content or ""
                for tc in c.delta.tool_calls or []:
                    p = parts.setdefault(tc.index, {"name": "", "args": ""})
                    if tc.function is not None:
                        p["name"] += tc.function.name or ""
                        p["args"] += tc.function.arguments or ""
                finish = c.finish_reason or finish
        calls = [(p["name"], p["args"]) for _, p in sorted(parts.items())]
    parsed = []
    for name, args in calls:
        try:
            parsed.append((name, json.loads(args) if args else {}))
        except ValueError:
            parsed.append((name, {"<invalid json>": args}))
    return {"calls": parsed, "content": content, "finish": finish, "completion": usage,
            "ms": round((time.perf_counter() - t0) * 1e3, 1)}


def _answer(content: str) -> str:
    """The content after the reasoning block (the `tags` format keeps it in `content`)."""
    i = content.rfind("</think>")
    return content[i + len("</think>"):] if i >= 0 else content


def score(got: list, want: list, content: str) -> dict:
    """Parse / name / args for one answer against its expected calls."""
    parse = len(got) == len(want) and "<tool_call>" not in _answer(content)
    name = Counter(n for n, _ in got) == Counter(n for n, _ in want)
    # match expectations to calls by name, the pairs with the most equal values first
    pairs = sorted(((sum(g[1].get(k) == v for k, v in w[1].items()), i, j)
                    for i, w in enumerate(want) for j, g in enumerate(got) if g[0] == w[0]),
                   reverse=True)
    match, used = {}, set()
    for _, i, j in pairs:
        if i not in match and j not in used:
            match[i] = j
            used.add(j)
    misses, total, exact, by_type = [], 0, 0, {}
    for i, (wname, wargs) in enumerate(want):
        best = got[match[i]] if i in match else None
        for k, v in wargs.items():
            total += 1
            kind = by_type.setdefault(type(v).__name__, [0, 0])
            kind[1] += 1
            have = None if best is None else best[1].get(k)
            # exact means the value AND its JSON type: "40" is not 40, 1 is not True
            if type(have) is type(v) and have == v:
                exact += 1
                kind[0] += 1
            else:
                misses.append({"function": wname, "param": k, "want": v, "got": have})
    return {"parse": parse, "name": name, "args_exact": exact, "args_total": total,
            "by_type": by_type, "misses": misses}


def run_matrix(client, model: str, a) -> dict:
    out = {"scenarios": {}, "round_trip": None, "choice": {}, "malformed": {}}
    for label, (msgs, want, extra) in SCENARIOS.items():
        rows = []
        for stream in (False, True):
            for rep in range(a.repeats):
                r = ask(client, model, msgs, stream, extra, a)
                r.update(score(r["calls"], want, r["content"]), stream=stream, rep=rep)
                r["content"] = r["content"][-400:]
                rows.append(r)
                print(f"  {label:18s} {'stream' if stream else 'json  '} finish={r['finish']:10s} "
                      f"calls={[n for n, _ in r['calls']]} parse={r['parse']} name={r['name']} "
                      f"args={r['args_exact']}/{r['args_total']} {r['ms']:.0f} ms", flush=True)
        out["scenarios"][label] = rows
    # the round trip: call -> the harness plays the tool -> the final answer
    rt = []
    for stream in (False, True):
        first = ask(client, model, ROUND_TRIP["ask"], stream, {}, a)
        s1 = score(first["calls"], [ROUND_TRIP["expect"]], first["content"])
        done = False
        final = None
        if first["calls"]:
            name, args = first["calls"][0]
            msgs = ROUND_TRIP["ask"] + [
                {"role": "assistant", "content": _answer(first["content"]) or None,
                 "tool_calls": [{"id": "call_1", "type": "function",
                                 "function": {"name": name, "arguments": json.dumps(args)}}]},
                {"role": "tool", "tool_call_id": "call_1", "content": ROUND_TRIP["result"]}]
            final = ask(client, model, msgs, stream, {}, a)
            done = (not final["calls"] and final["finish"] == "stop"
                    and ROUND_TRIP["answer_contains"] in _answer(final["content"]))
        rt.append({"stream": stream, "first": {**s1, "calls": first["calls"],
                                               "finish": first["finish"]},
                   "final": None if final is None else {"calls": final["calls"],
                                                        "finish": final["finish"],
                                                        "answer": _answer(final["content"])[-300:]},
                   "completed": done})
        print(f"  round-trip         {'stream' if stream else 'json  '} call={s1['name']} "
              f"completed={done}", flush=True)
    out["round_trip"] = rt
    # malformed / cut blocks: the fallback must keep the text and return no call
    for label, extra in (("stop-inside-block", {"stop": ["</parameter>"]}),):
        rows = []
        for stream in (False, True):
            r = ask(client, model, SCENARIOS["single/read"][0], stream, extra, a)
            ans = _answer(r["content"])
            ok = not r["calls"] and "<tool_call>" in ans
            rows.append({"stream": stream, "finish": r["finish"], "calls": r["calls"],
                         "text_kept": "<tool_call>" in ans, "ok": ok, "tail": ans[-200:]})
            print(f"  malformed/{label:8s} {'stream' if stream else 'json  '} ok={ok} "
                  f"finish={r['finish']}", flush=True)
        out["malformed"][label] = rows
    # tool_choice compliance
    for mode in ("auto", "none", "required", "named"):
        tc = ({"type": "function", "function": {"name": NAMED}} if mode == "named" else mode)
        rows = []
        for label, prompt, needs in CHOICE_PROMPTS:
            for rep in range(a.repeats):
                r = ask(client, model, [{"role": "user", "content": prompt}], False,
                        {"tool_choice": tc}, a)
                names = [n for n, _ in r["calls"]]
                ok = {"auto": True, "none": not names, "required": bool(names),
                      "named": bool(names) and set(names) == {NAMED}}[mode]
                rows.append({"prompt": label, "needs_tool": needs, "calls": names,
                             "finish": r["finish"], "complied": ok, "rep": rep})
                print(f"  choice/{mode:8s} {label:7s} calls={names} complied={ok}", flush=True)
        out["choice"][mode] = rows
    return out


def summarise(res: dict) -> dict:
    """The three numbers per scenario and overall, the compliance rates, the round trip."""
    per, tot = {}, Counter()
    for label, rows in res["scenarios"].items():
        n = len(rows)
        s = {"n": n, "parse": sum(r["parse"] for r in rows) / n,
             "name": sum(r["name"] for r in rows) / n,
             "args": (sum(r["args_exact"] for r in rows) / sum(r["args_total"] for r in rows)
                      if sum(r["args_total"] for r in rows) else None),
             "stream_agrees": all(
                 a["calls"] == b["calls"] for a, b in zip(
                     [r for r in rows if not r["stream"]], [r for r in rows if r["stream"]]))}
        per[label] = s
        tot["n"] += n
        tot["parse"] += sum(r["parse"] for r in rows)
        tot["name"] += sum(r["name"] for r in rows)
        tot["args_exact"] += sum(r["args_exact"] for r in rows)
        tot["args_total"] += sum(r["args_total"] for r in rows)
    comp = {}
    for mode, rows in res["choice"].items():
        if mode == "auto":
            need = [r for r in rows if r["needs_tool"]]
            free = [r for r in rows if not r["needs_tool"]]
            comp[mode] = {"call_rate_when_needed": sum(bool(r["calls"]) for r in need) / len(need),
                          "call_rate_when_not": sum(bool(r["calls"]) for r in free) / len(free)}
        else:
            comp[mode] = {"compliance": sum(r["complied"] for r in rows) / len(rows),
                          "n": len(rows)}
    types: dict = {}
    for rows in res["scenarios"].values():
        for r in rows:
            for k, (e, t) in (r.get("by_type") or {}).items():
                acc = types.setdefault(k, [0, 0])
                acc[0] += e
                acc[1] += t
    return {"scenarios": per,
            "overall": {"n": tot["n"], "parse": tot["parse"] / tot["n"],
                        "name": tot["name"] / tot["n"],
                        "args": tot["args_exact"] / tot["args_total"]},
            "args_by_type": {k: e / t for k, (e, t) in sorted(types.items())},
            "round_trip": sum(r["completed"] for r in res["round_trip"]) / len(res["round_trip"]),
            "malformed_ok": all(r["ok"] for rows in res["malformed"].values() for r in rows),
            "choice": comp}


def show(report: dict) -> None:
    for think, block in report["runs"].items():
        s = block["summary"]
        print(f"\nthinking {think}")
        print(f"  {'scenario':18s} {'n':>3s} {'parse':>6s} {'name':>6s} {'args':>6s}  stream==json")
        for label, v in s["scenarios"].items():
            args = "-" if v["args"] is None else f"{v['args']:.2f}"
            print(f"  {label:18s} {v['n']:3d} {v['parse']:6.2f} {v['name']:6.2f} {args:>6s}  "
                  f"{v['stream_agrees']}")
        o = s["overall"]
        print(f"  {'overall':18s} {o['n']:3d} {o['parse']:6.2f} {o['name']:6.2f} {o['args']:6.2f}")
        if s.get("args_by_type"):
            print("  args by type: " + ", ".join(f"{k} {v:.2f}"
                                                 for k, v in s["args_by_type"].items()))
        print(f"  round trip completed {s['round_trip']:.2f}; malformed fallback ok "
              f"{s['malformed_ok']}")
        for mode, v in s["choice"].items():
            print(f"  tool_choice {mode:9s} {json.dumps(v)}")
        misses = [m for rows in block["results"]["scenarios"].values() for r in rows
                  for m in r["misses"]]
        for m in misses[:12]:
            print(f"  miss {m['function']}.{m['param']}: want {m['want']!r} got {m['got']!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="http://127.0.0.1:8011/v1")
    ap.add_argument("--model", default="")
    ap.add_argument("--think", default="off,on", help="comma-separated: off, on")
    ap.add_argument("--effort", default="low", help="reasoning_effort with thinking on")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--json", default="")
    ap.add_argument("--read", default="", help="print a report this tool wrote")
    a = ap.parse_args()
    if a.read:
        show(json.loads(Path(a.read).read_text()))
        return
    client = _client(a.base, a.timeout)
    model = a.model or client.models.list().data[0].id
    health = {}
    try:
        with urllib.request.urlopen(a.base.rsplit("/v1", 1)[0] + "/health", timeout=10) as r:
            h = json.loads(r.read())
        health = {k: h.get(k) for k in ("model", "max_len", "sampling", "penalty",
                                        "default_max_tokens")}
    except Exception as exc:                                       # noqa: BLE001
        health = {"error": str(exc)}
    report = {"meta": {"base": a.base, "model": model, "args": vars(a), "health": health,
                       "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                       "tools": [t["function"]["name"] for t in TOOLS]}, "runs": {}}
    for think in a.think.split(","):
        a.cur_think = think == "on"
        print(f"[bench_toolcall] thinking {think}", flush=True)
        res = run_matrix(client, model, a)
        report["runs"][think] = {"results": res, "summary": summarise(res)}
    report["meta"]["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(report, indent=1))
        print(f"[bench_toolcall] wrote {a.json}")
    show(report)


if __name__ == "__main__":
    sys.exit(main())
