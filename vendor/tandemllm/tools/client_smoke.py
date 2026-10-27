"""The real-client smoke set: the request shapes the served clients send, each PASS or FAIL.

The row gate measures one shape (the atlas row). rc6 shipped a kernel configuration that broke Open
WebUI's Agent lane (a long prompt with a penalty) and every release before rc7 returned tool-call
values as strings that opencode's validator refused. Before a deploy, this runs the shapes
those clients send, through the official `openai` client, and checks what a client checks:

  * opencode: its nine tools (tests/fixtures/opencode_tools.json), a long system prompt, stream with
    include_usage, `reasoning_format`, `max_tokens` 32000; the calls' arguments must VALIDATE
    against the tool's JSON schema (jsonschema, draft 2020-12), as opencode's validator does;
  * a captured opencode body (`--opencode-body`, the request opencode sent, as JSON), replayed;
  * Open WebUI's Agent lane: stream, include_usage, penalties, no_repeat_ngram_size, a reasoning
    budget, tools, and an ~12k-token prompt (the rc6 crash shape);
  * the OpenAI SDK corners: tool_choice auto / required / named / none, parallel_tool_calls true
    and false, streamed tool-call deltas (id and name first, arguments in chunks that concatenate
    to valid JSON, one index a call), a tool round trip with typed arguments in the history, n=2,
    repetition / presence / frequency penalties, stream with and without usage, long prompts
    (8k, 32k, and one over 100k with `--long`);
  * a streamed 24k prefill: the prefill watch's `: prefill done/total` comment lines come
    before the model's first token (after TandemLLM's step lines, which open the stream while the
    prefill runs), never inside an event, and the stream reads as before.

    python tools/client_smoke.py --base http://127.0.0.1:8011/v1 --json results/api/smoke-<label>.json
    python tools/client_smoke.py --base http://127.0.0.1:8011/v1 --long --opencode-body body.json
    python tools/client_smoke.py --base http://127.0.0.1:8000/v1 --x-tandem raw   # the raw path

A server with the conductor serves one model, `TandemLLM` (the raw name is a hidden alias), so a
check of the raw engine through it sends `--x-tandem raw`: every request then carries
`X-Tandem: raw`, and the first check asserts the response says `X-Tandem: raw` back.

Exit code: the number of failed checks.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OC_TOOLS = os.path.join(HERE, "..", "tests", "fixtures", "opencode_tools.json")

WEATHER = [
    {"type": "function", "function": {
        "name": "get_weather", "description": "Current weather and a forecast for a city.",
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            "days": {"type": "integer", "minimum": 1, "maximum": 7}},
            "required": ["city", "days"]}}},
    {"type": "function", "function": {
        "name": "get_time", "description": "The local time in a city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                       "required": ["city"]}}},
    {"type": "function", "function": {
        "name": "write_file", "description": "Write a text file.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}, "content": {"type": "string"},
            "overwrite": {"type": "boolean"}}, "required": ["path", "content"]}}},
]

FILLER = ("The river valley holds a small town whose mills once ground grain for the whole region; "
          "today the old buildings host workshops, a library and a market on Saturdays. ")


def filler(tokens: int) -> str:
    """About `tokens` tokens of prose, numbered so no two lines repeat."""
    n = max(1, tokens // 37)                     # 37 tokens a line, measured on the served tokenizer
    return "\n".join(f"{i}. {FILLER}" for i in range(n))


#: how TandemLLM's step lines open (server/conductor/states.py MARKER)
STEP = "› "


def _steps_only(ev: dict) -> bool:
    """An event that carries no token of the model's: TandemLLM's step lines (`› reading 24,016
    tokens ...`, S1), a role, nothing, or the usage."""
    for ch in ev.get("choices") or []:
        d = ch.get("delta") or {}
        if d.get("tool_calls"):
            return False
        text = "".join(d.get(k) or "" for k in ("content", "reasoning_content", "reasoning"))
        if any(ln.strip() and not ln.startswith(STEP) for ln in text.split("\n")):
            return False
    return True


def read_heartbeat(lines) -> dict:
    """The prefill heartbeat check's reading of a raw SSE stream, line by line (no line ends).
    `: ...` comments are keep-alives: before the first event, or after events that carry only
    TandemLLM's step lines (S1's progress lines open the stream while the prefill runs); never
    after the model's first token, which comes after the prefill, and never inside an event
    (between its data line and the blank line that ends it). Every data line is JSON and the
    stream ends in [DONE]."""
    comments, events, done, bad = 0, 0, False, []
    spoken = in_event = False
    for line in lines:
        if not line:
            in_event = False
        elif line.startswith(":"):
            comments += 1
            if in_event:
                bad.append("a comment inside an event")
            elif spoken:
                bad.append("a comment after the model's first token")
        elif line.startswith("data: "):
            in_event = True
            if line == "data: [DONE]":
                done = True
                continue
            try:
                ev = json.loads(line[6:])
            except ValueError:
                bad.append(f"not JSON {line[:60]!r}")
                continue
            events += 1
            spoken = spoken or not _steps_only(ev)
        else:
            bad.append(f"unexpected line {line[:60]!r}")
    return {"comments": comments, "events": events, "done": done, "problems": bad}


class Smoke:
    def __init__(self, base: str, key: str, model: str | None, x_tandem: str = "",
                 profile: str = ""):
        import openai
        self.openai = openai
        self.x_tandem = x_tandem
        self.headers = {**({"X-Tandem": x_tandem} if x_tandem else {}),
                        **({"X-Tandem-Profile": profile} if profile else {})}
        self.client = openai.OpenAI(base_url=base, api_key=key or "none", timeout=1800,
                                    max_retries=0, default_headers=self.headers or None)
        self.model = model or self.client.models.list().data[0].id
        self.results: list[dict] = []
        self.last_profile: str | None = None       # X-Tandem-Profile of the last streamed answer
        self.last_mode: str | None = None          # its X-Tandem: on, shadow or raw

    def check(self, name, ok, **seen):
        self.results.append({"check": name, "pass": bool(ok), **seen})
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {json.dumps(seen, ensure_ascii=False)[:400]}",
              flush=True)

    def run(self, name, fn):
        t0 = time.time()
        try:
            fn()
        except Exception as exc:                       # noqa: BLE001 -- a smoke reports, not raises
            self.check(name, False, error=f"{type(exc).__name__}: {str(exc)[:300]}")
        print(f"      ({time.time() - t0:.1f} s)", flush=True)

    # ---------------------------------------------------------------- transports
    def stream(self, **kw):
        """(content, reasoning, calls [(index, id, name, args-text, n_arg_chunks)], finish,
        usage, problems) of one streamed request, read as a client reads the deltas."""
        content, reasoning, finish, usage, problems = "", "", None, None, []
        calls: dict[int, dict] = {}
        usage_chunks = 0
        raw = self.client.chat.completions.with_raw_response.create(model=self.model, stream=True,
                                                                    **kw)
        self.last_profile = raw.headers.get("x-tandem-profile")
        self.last_mode = raw.headers.get("x-tandem")
        for ch in raw.parse():
            if ch.usage is not None:
                usage, usage_chunks = ch.usage, usage_chunks + 1
            for cc in ch.choices:
                d = cc.delta
                content += d.content or ""
                reasoning += getattr(d, "reasoning_content", None) or ""
                for tc in d.tool_calls or []:
                    c = calls.get(tc.index)
                    if c is None:
                        if not tc.id or not (tc.function and tc.function.name):
                            problems.append(f"index {tc.index}: first delta without id/name")
                        c = calls[tc.index] = {"id": tc.id, "name": "", "args": "", "chunks": 0}
                    elif tc.id and tc.id != c["id"]:
                        problems.append(f"index {tc.index}: id changed")
                    if tc.function:
                        c["name"] += tc.function.name or ""
                        if tc.function.arguments:
                            c["args"] += tc.function.arguments
                            c["chunks"] += 1
                finish = cc.finish_reason or finish
        if usage_chunks > 1:
            problems.append(f"{usage_chunks} usage chunks")
        out = [(i, c["id"], c["name"], c["args"], c["chunks"]) for i, c in sorted(calls.items())]
        return content, reasoning, out, finish, usage, problems

    def once(self, **kw):
        r = self.client.chat.completions.create(model=self.model, **kw)
        c = r.choices[0]
        calls = [(i, t.id, t.function.name, t.function.arguments, 1)
                 for i, t in enumerate(c.message.tool_calls or [])]
        return (c.message.content or "", getattr(c.message, "reasoning_content", None) or "",
                calls, c.finish_reason, r.usage, [])


def validate(calls, tools) -> list[str]:
    """What a validating client (opencode, the AI SDK) says about each call: JSON, a known tool,
    and the arguments valid against its schema."""
    import jsonschema
    by = {t["function"]["name"]: t["function"].get("parameters") or {} for t in tools}
    bad = []
    for _, _, name, args, _ in calls:
        if name not in by:
            bad.append(f"{name}: not a tool of the request")
            continue
        try:
            obj = json.loads(args or "{}")
        except ValueError as exc:
            bad.append(f"{name}: arguments are not JSON ({exc})")
            continue
        schema = {k: v for k, v in by[name].items() if k != "$schema"}
        for err in jsonschema.Draft202012Validator(schema).iter_errors(obj):
            bad.append(f"{name}: {'/'.join(map(str, err.path))} {err.message[:120]}")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="http://127.0.0.1:8011/v1")
    ap.add_argument("--key", default=os.environ.get("QSE_API_KEY", ""))
    ap.add_argument("--model", default=None)
    ap.add_argument("--long", action="store_true", help="also the >100k prompt (minutes)")
    ap.add_argument("--opencode-body", default="", help="a captured opencode request (JSON)")
    ap.add_argument("--only", default="", help="comma-separated check prefixes")
    ap.add_argument("--json", default="")
    ap.add_argument("--x-tandem", default="", choices=("", "raw", "shadow", "on"),
                    help="send X-Tandem with every request (raw: the raw path of a conductor "
                         "server) and check the response says the same mode")
    ap.add_argument("--x-tandem-profile", default="",
                    help="send X-Tandem-Profile with every request: the checks under a conductor "
                         "profile (coding: what opencode gets), not the raw path")
    a = ap.parse_args()
    s = Smoke(a.base, a.key, a.model, a.x_tandem, a.x_tandem_profile)
    if a.x_tandem:
        r = s.client.chat.completions.with_raw_response.create(
            model=s.model, max_tokens=4, messages=[{"role": "user", "content": "Say ok."}],
            extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        got = r.headers.get("x-tandem")
        s.check("x_tandem_header", got == a.x_tandem, sent=a.x_tandem, got=got, model=s.model,
                profile=r.headers.get("x-tandem-profile"))
    with open(OC_TOOLS) as f:
        oc_tools = json.load(f)["tools"]
    oc_system = ("You are opencode, an interactive CLI coding agent. Use the tools to act; never "
                 "print a tool call as text.\n" + filler(9000))
    OFF = {"chat_template_kwargs": {"enable_thinking": False}}

    def oc(task, stream=True, **kw):
        body = dict(messages=[{"role": "system", "content": oc_system},
                              {"role": "user", "content": task}],
                    tools=oc_tools, tool_choice="auto", max_tokens=32000,
                    extra_body={"reasoning_format": "reasoning_content"})
        body.update(kw)
        if stream:
            body["stream_options"] = {"include_usage": True}
            return s.stream(**body)
        return s.once(**body)

    def expect_calls(name, res, tools, want_names=None, n=None, finish="tool_calls"):
        content, _, calls, fin, usage, problems = res
        bad = validate(calls, tools) + problems
        if "<tool_call" in content or "<function=" in content:
            bad.append("call text left in the content")
        if want_names is not None and not {c[2] for c in calls} <= set(want_names):
            bad.append(f"names {[c[2] for c in calls]} not in {want_names}")
        if n is not None and len(calls) != n:
            bad.append(f"{len(calls)} calls, expected {n}")
        if not calls:
            bad.append("no call")
        if finish and fin != finish:
            bad.append(f"finish {fin}")
        s.check(name, not bad, calls=[(c[2], c[3][:160]) for c in calls], finish=fin,
                problems=bad, usage=getattr(usage, "total_tokens", None))

    checks = []

    def add(name):
        def deco(fn):
            checks.append((name, fn))
            return fn
        return deco

    @add("opencode: read with offset/limit, streamed, validated")
    def _():
        expect_calls("opencode: read with offset/limit, streamed, validated",
                     oc("Use the read tool to read lines 40 to 49 of /tmp/proj/data/numbers.txt: "
                        "pass offset 40 and limit 10."), oc_tools, ["read"])

    @add("opencode: bash with a timeout, JSON answer, validated")
    def _():
        expect_calls("opencode: bash with a timeout, JSON answer, validated",
                     oc("Run `python3 -c 'print(6*7)'` with the bash tool and a timeout of 20000 "
                        "milliseconds.", stream=False), oc_tools, ["bash"])

    @add("opencode: todowrite list, streamed, validated")
    def _():
        expect_calls("opencode: todowrite list, streamed, validated",
                     oc("Before anything else, make a todo list with the todowrite tool: three items "
                        "(create hello.py, run it, report), the first in_progress."), oc_tools,
                     ["todowrite"])

    @add("opencode: edit with replaceAll, streamed, validated")
    def _():
        expect_calls("opencode: edit with replaceAll, streamed, validated",
                     oc("In /tmp/proj/src/config.py replace every occurrence of localhost with "
                        "127.0.0.1 in ONE edit call with replaceAll set to true."), oc_tools,
                     ["edit", "read"])

    @add("opencode: captured body replayed")
    def _():
        if not a.opencode_body:
            s.check("opencode: captured body replayed (skipped, no --opencode-body)", True)
            return
        with open(a.opencode_body) as f:
            body = json.load(f)
        body.pop("model", None)
        body.pop("stream", None)
        msgs = body.pop("messages")
        msgs[-1] = {"role": "user", "content": "Read only lines 3 to 5 of README.md with the read "
                                               "tool (offset 3, limit 3)."}
        extra = {k: body.pop(k) for k in list(body) if k not in (
            "tools", "tool_choice", "max_tokens", "stream_options", "temperature", "top_p")}
        expect_calls("opencode: captured body replayed", s.stream(messages=msgs, extra_body=extra,
                                                                 **body), body["tools"])

    @add("sdk: tool_choice auto, typed arguments, both transports")
    def _():
        msgs = [{"role": "user", "content": "What's the weather in Paris for the next 3 days, in "
                                            "celsius?"}]
        for label, fn in (("json", s.once), ("stream", s.stream)):
            expect_calls(f"sdk: tool_choice auto, typed arguments ({label})",
                         fn(messages=msgs, tools=WEATHER, max_tokens=2048, extra_body=OFF),
                         WEATHER, ["get_weather"])

    @add("sdk: tool_choice required, streamed")
    def _():
        expect_calls("sdk: tool_choice required, streamed",
                     s.stream(messages=[{"role": "user", "content": "Hi there!"}], tools=WEATHER,
                              tool_choice="required", max_tokens=1024, extra_body=OFF),
                     WEATHER, n=1)

    @add("sdk: tool_choice named")
    def _():
        expect_calls("sdk: tool_choice named (get_time)",
                     s.once(messages=[{"role": "user", "content": "Weather in Rome for 2 days?"}],
                            tools=WEATHER, max_tokens=1024, extra_body=OFF,
                            tool_choice={"type": "function", "function": {"name": "get_time"}}),
                     WEATHER, ["get_time"], n=1)

    @add("sdk: tool_choice none")
    def _():
        content, _, calls, fin, _, _ = s.once(
            messages=[{"role": "user", "content": "Weather in Rome for 2 days?"}], tools=WEATHER,
            tool_choice="none", max_tokens=256, extra_body=OFF)
        s.check("sdk: tool_choice none: no calls, finish stop or length", not calls and fin in (
            "stop", "length"), finish=fin, tail=content[-80:])

    @add("sdk: parallel_tool_calls")
    def _():
        msgs = [{"role": "user", "content": "Get the 2-day weather for Paris AND for Berlin (two "
                                            "get_weather calls)."}]
        res = s.stream(messages=msgs, tools=WEATHER, max_tokens=2048, parallel_tool_calls=True,
                       extra_body=OFF)
        expect_calls("sdk: parallel_tool_calls true: two calls, streamed", res, WEATHER,
                     ["get_weather"], n=2)
        ids = [c[1] for c in res[2]]
        s.check("sdk: parallel calls have distinct ids and indexes", len(set(ids)) == len(ids)
                and [c[0] for c in res[2]] == list(range(len(ids))), ids=ids)
        expect_calls("sdk: parallel_tool_calls false: one call",
                     s.once(messages=msgs, tools=WEATHER, max_tokens=2048,
                            parallel_tool_calls=False, extra_body=OFF), WEATHER, n=1)

    @add("sdk: streamed arguments arrive in chunks")
    def _():
        res = s.stream(messages=[{"role": "user", "content": "Write /tmp/poem.txt with a 12-line "
                                  "poem about rivers (overwrite true)."}], tools=WEATHER,
                       max_tokens=4096, extra_body=OFF)
        expect_calls("sdk: write_file call, streamed", res, WEATHER, ["write_file"])
        chunks = [c[4] for c in res[2]]
        if s.last_profile == "tools" and s.last_mode == "on":
            # the tools profile's tool-call checks act, so SP3 holds a call until it has seen it
            # and sends it whole, one delta a call (docs/server.md, Tool-call checks)
            s.check("sdk: a held call goes out whole, one delta (tools profile)", chunks == [1],
                    chunks=chunks, profile=s.last_profile, mode=s.last_mode)
        else:
            s.check("sdk: a long string argument streams in many chunks",
                    chunks and max(chunks) > 5, chunks=chunks, profile=s.last_profile,
                    mode=s.last_mode)

    @add("sdk: tool round trip with typed arguments in the history")
    def _():
        msgs = [{"role": "user", "content": "What's the weather in Oslo for 2 days?"},
                {"role": "assistant", "content": "", "tool_calls": [{
                    "id": "call_1", "type": "function", "function": {
                        "name": "get_weather", "arguments": json.dumps(
                            {"city": "Oslo", "days": 2, "unit": "celsius"})}}]},
                {"role": "tool", "tool_call_id": "call_1",
                 "content": json.dumps({"today": "4 C, rain", "tomorrow": "6 C, cloudy"})}]
        for label, fn in (("json", s.once), ("stream", s.stream)):
            content, _, calls, fin, _, prob = fn(messages=msgs, tools=WEATHER, max_tokens=512,
                                                 extra_body=OFF)
            s.check(f"sdk: round trip answers from the tool result ({label})",
                    fin == "stop" and not calls and "rain" in content.lower() and not prob,
                    finish=fin, tail=content[-120:])

    @add("sdk: n=2")
    def _():
        r = s.client.chat.completions.create(
            model=s.model, n=2, max_tokens=32, extra_body=OFF,
            messages=[{"role": "user", "content": "Name one colour."}])
        s.check("sdk: n=2 greedy: two identical choices", len(r.choices) == 2 and
                r.choices[0].message.content == r.choices[1].message.content,
                answers=[c.message.content for c in r.choices])
        r = s.client.chat.completions.create(
            model=s.model, n=2, max_tokens=1024, tools=WEATHER, extra_body=OFF,
            messages=[{"role": "user", "content": "Weather in Paris for 2 days?"}])
        calls = [[(0, t.id, t.function.name, t.function.arguments, 1)
                  for t in c.message.tool_calls or []] for c in r.choices]
        bad = [b for cs in calls for b in validate(cs, WEATHER)]
        s.check("sdk: n=2 with tools: both choices call, validated",
                all(calls) and not bad and all(c.finish_reason == "tool_calls" for c in r.choices),
                problems=bad)

    @add("sdk: penalties")
    def _():
        r = s.client.chat.completions.create(
            model=s.model, max_tokens=64, presence_penalty=0.5, frequency_penalty=0.3,
            extra_body=dict(OFF, repetition_penalty=1.05),
            messages=[{"role": "user", "content": "List five fruits, comma separated."}])
        s.check("sdk: repetition + presence + frequency penalties: served",
                r.choices[0].finish_reason in ("stop", "length") and r.choices[0].message.content,
                answer=r.choices[0].message.content)

    @add("stream with and without usage")
    def _():
        for inc in (True, False):
            kw = {"stream_options": {"include_usage": True}} if inc else {}
            content, _, _, fin, usage, prob = s.stream(
                messages=[{"role": "user", "content": "Say ok."}], max_tokens=16,
                extra_body=OFF, **kw)
            # without include_usage the engine still puts usage on the finish chunk (Open
            # WebUI reads it there); what a client must never see is two usage chunks
            s.check(f"stream include_usage={inc}: served, usage at most once",
                    fin in ("stop", "length") and (usage is not None or not inc) and not prob,
                    finish=fin, usage=getattr(usage, "total_tokens", None))

    @add("owui agent: 12k prompt, penalties, tools, streamed")
    def _():
        content, reasoning, calls, fin, usage, prob = s.stream(
            messages=[{"role": "system", "content": "You are a helpful agent.\n" + filler(11700)},
                      {"role": "user", "content": "What is 12 times 12? One line."}],
            stream_options={"include_usage": True}, max_tokens=32768,
            tools=[{"type": "function", "function": {"name": "web_search", "parameters": {
                "type": "object", "properties": {"query": {"type": "string"}}}}}],
            presence_penalty=0.3, frequency_penalty=0.1,
            extra_body={"no_repeat_ngram_size": 4, "max_reasoning_tokens": 1536,
                        "reasoning_effort": "low", "repetition_penalty": 1.05})
        # served is the check (rc6 answered this shape with an error); the penalties may well steer
        # the digits of the answer (measured: "₁₄₄"), which is the penalty's business. On the raw
        # path the answer can end inside the reasoning: 0.11.4 and 0.12.0 both write 9 tokens and
        # no </think> (2026-10-19); the open block was the content until SRV-617 moved a plain
        # client's thinking to reasoning_content
        s.check("owui agent: 12k prompt + penalties + no_repeat + tools: served",
                fin in ("stop", "tool_calls") and usage is not None and not prob
                and (content.split("</think>")[-1].strip() or calls or reasoning.strip()),
                finish=fin, tail=content[-80:], reasoning_tail=reasoning[-80:],
                prompt=getattr(usage, "prompt_tokens", None))

    @add("long prompts")
    def _():
        for toks in (8000, 32000) + ((105000,) if a.long else ()):
            content, _, _, fin, usage, prob = s.stream(
                messages=[{"role": "user", "content": filler(toks) + "\n\nHow many numbered "
                           "lines are above, roughly? One short sentence."}],
                max_tokens=64, stream_options={"include_usage": True}, extra_body=OFF)
            s.check(f"long prompt ~{toks // 1000}k: served", fin in ("stop", "length")
                    and usage is not None and content.strip() and not prob, finish=fin,
                    prompt=getattr(usage, "prompt_tokens", None), tail=content[-60:])

    @add("prefill heartbeat: SSE comments, then a normal stream")
    def _():
        # a streamed prefill past --prefill-heartbeat-s sends `: prefill done/total` comment
        # lines; a client must read the stream as before. Raw bytes here (the SDKs hide comments).
        import urllib.request
        body = {"model": s.model, "stream": True, "max_tokens": 32,
                "stream_options": {"include_usage": True},
                # a first line of its own, so no cached prefix (the long-prompt checks' filler)
                # shortens the prefill below the heartbeat's 5 s
                "messages": [{"role": "user", "content": f"Run {time.time_ns()}.\n" + filler(24000)
                              + "\n\nSay ok."}]}
        body.update(OFF)
        req = urllib.request.Request(a.base.rstrip("/") + "/chat/completions",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json",
                                              "Authorization": f"Bearer {a.key or 'none'}",
                                              **s.headers})
        with urllib.request.urlopen(req, timeout=1800) as r:
            got = read_heartbeat(raw.decode().rstrip("\r\n") for raw in r)
        s.check("prefill heartbeat: comments before the model's first token, the stream intact",
                got["comments"] >= 1 and got["events"] >= 2 and got["done"]
                and not got["problems"], comments=got["comments"], events=got["events"],
                problems=got["problems"][:3])

    only = [x for x in a.only.split(",") if x]
    for name, fn in checks:
        if only and not any(name.startswith(o) for o in only):
            continue
        print(f"--- {name}", flush=True)
        s.run(name, fn)
    fails = [r for r in s.results if not r["pass"]]
    print(f"\n{len(s.results) - len(fails)}/{len(s.results)} passed, {len(fails)} failed")
    if a.json:
        os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
        with open(a.json, "w") as f:
            json.dump({"base": a.base, "model": s.model, "at": time.strftime("%Y-%m-%d %H:%M:%S"),
                       "results": s.results}, f, indent=1)
    return len(fails)


if __name__ == "__main__":
    sys.exit(main())
