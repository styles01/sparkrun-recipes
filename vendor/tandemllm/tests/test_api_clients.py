"""Plain API clients get a clean answer (SRV-245, SRV-247; UX2's normal-user run, 2026-10-07).

On a 0.4.0 test build an OpenAI SDK or curl client that named no `reasoning_format` got the thinking in
`content` as `<think>...</think>`; `stop: ["\\n\\n"]` ended a response at the first blank line of
the reasoning (55 tokens, no answer); a 60k-token prompt's progress lines went into `content`; and
a memory question from a client that offered no tools ended with the model's own `tool` call
(a GET on the memory search) handed over with finish `tool_calls`.

  1  the default under a conductor profile: every client kind gets `reasoning_content` (Hermes
     keeps `tags`); `reasoning_format: tags` keeps the tags; on the raw model a plain API client
     (SDK, curl, an unnamed one) gets `reasoning_content` too, every other kind the server's (SRV-617)
  2  a stop string applies to the answer: one the reasoning holds does not cut it, and a blank
     line does not match the template's blank lines after `</think>`; streamed and not
  3  the answer does not start with the blank lines after the reasoning
  4  state lines never in a plain client's `content`: in `reasoning_content`, or none with tags
  5  a request without tools never gets `tool_calls`: the call is read out of the answer and
     dropped, finish `stop`, the journal row names it; streamed and not; with tools it is a call
  6  SRV-619, SRV-620: tools beside `response_format` (json_schema, json_object) are served, not a
     400: the answer is the bare document (no fences, the thinking in `reasoning_content`) or
     tool calls (one, two, one with `parallel_tool_calls: false`); `tool_choice: none` is the
     document alone, `required` the calls; the constraint the engine gets holds the answer, and
     the branch the answer took is recorded; streamed and not, raw and under the conductor, curl
     and the SDK

Run: python tests/test_api_clients.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_ui_states as tu  # noqa: E402
from engine import grammar as G  # noqa: E402
from server import app  # noqa: E402
from server.conductor import states  # noqa: E402

CURL = {"User-Agent": "curl/8.5.0"}
SDK = {"User-Agent": "OpenAI/Python 2.24.0"}


def ask(text, headers=CURL, stream=True, **extra):
    body = {"model": "TandemLLM", "stream": stream, "max_tokens": 2000,
            "messages": [{"role": "user", "content": text}], **extra}
    return tu._go(body, headers)


def finish_of(out) -> str | None:
    if not out["raw"].lstrip().startswith("data:"):
        return json.loads(out["raw"])["choices"][0]["finish_reason"]
    fin = None
    for ln in out["raw"].splitlines():
        if ln.startswith("data: {"):
            for ch in json.loads(ln[6:]).get("choices") or ():
                fin = ch.get("finish_reason") or fin
    return fin


# ------------------------------------------------------------------ 1 the default

def test_1_sdk_and_curl_get_reasoning_content():
    tu.serve("on")
    for h in (CURL, SDK):
        for stream in (True, False):
            out = ask("Say hello in one short sentence.", h, stream)
            assert "<think>" not in out["content"] and "</think>" not in out["content"], out
            assert "The question has" in out["reasoning"], out["reasoning"][:80]
    tags = ask("Say hello in one short sentence.", CURL, reasoning_format="tags")
    assert tags["content"].startswith("<think>\n") and "</think>" in tags["content"]

def raw(headers, stream=False, **extra):
    return tu._go({"model": "TandemLLM", "stream": stream, "max_tokens": 200,
                   "messages": [{"role": "user", "content": "Say hello."}], **extra},
                  dict(headers, **{"X-Tandem": "raw"}))


def test_1b_raw_path_plain_clients_get_reasoning_content():
    """SRV-617: a benchmark or an SDK on the raw path read `content` as the answer; the
    thinking goes to `reasoning_content` there too. Every other kind keeps the release's tags."""
    tu.serve("on")
    for h in (CURL, SDK, {"User-Agent": "python-httpx/0.28.1"}):
        for stream in (False, True):
            out = raw(h, stream)
            assert "<think>" not in out["content"] and "</think>" not in out["content"], out
            assert "The question has" in out["reasoning"], out["raw"][:200]
    # a client that names a format keeps it
    tags = raw(CURL, reasoning_format="tags")
    assert tags["content"].startswith("<think>") and tags["reasoning"] == "", tags["raw"][:200]
    # Open WebUI folds the tags itself, and the server's own calls parse them: the release's shape
    for h in ({"User-Agent": "Python/3.11 aiohttp/3.9.5"}, {"User-Agent": "tandem-conductor/1"}):
        out = raw(h)
        assert out["content"].startswith("<think>") and out["reasoning"] == "", out["raw"][:200]


def test_1c_raw_path_stops_read_the_answer():
    """SRV-617: a plain client on the raw path gets SRV-245's stop rule with its format: a stop
    string the reasoning holds does not cut the response, and a blank line is not the template's"""
    tu.serve("on")
    for stream in (False, True):
        for stop in (["question"], ["\n\n"]):
            out = tu._go({"model": "TandemLLM", "stream": stream, "max_tokens": 2000, "stop": stop,
                          "messages": [{"role": "user",
                                        "content": "Write two short paragraphs about cats."}]},
                         dict(CURL, **{"X-Tandem": "raw"}))
            assert len(out["content"]) > 20 and "<think>" not in out["content"], (stop, out)
            assert finish_of(out) == "stop", (stop, finish_of(out))


# ------------------------------------------------------------------ 2-3 stops and the answer

def test_2_a_stop_string_applies_to_the_answer():
    tu.serve("on")
    for stream in (True, False):
        # "question" is in the fake's reasoning only; before SRV-245 it cut the response there
        out = ask("Write two short paragraphs about cats.", stream=stream, stop=["question"])
        assert out["content"] and "<think>" not in out["content"], out
        assert finish_of(out) == "stop"
        # a blank line: the template's blank lines after </think> are not the answer's
        nl = ask("Write two short paragraphs about cats.", stream=stream, stop=["\n\n"])
        assert len(nl["content"]) > 20, nl["content"]
    # in the answer, a stop still cuts: the fake's answer is words; cut at its third space
    full = ask("Write two short paragraphs about cats.", stream=False)["content"]
    word = full.split(" ")[2]
    cut = ask("Write two short paragraphs about cats.", stream=False, stop=[" " + word])
    assert full.startswith(cut["content"]) and len(cut["content"]) < len(full), cut["content"]
    # tags asked for: the block whole, the answer cut
    t = ask("Write two short paragraphs about cats.", stream=True, stop=["question"],
            reasoning_format="tags")
    assert "</think>" in t["content"] and t["content"].split("</think>", 1)[1].strip()


def test_3_the_answer_does_not_start_with_blank_lines():
    tu.serve("on")
    for h in (CURL, SDK):
        for stream in (True, False):
            out = ask("Say hello in one short sentence.", h, stream)
            assert out["content"] and not out["content"].startswith("\n"), repr(out["content"][:20])


# ------------------------------------------------------------------ 4 state lines

def test_4_state_lines_never_in_a_plain_clients_content():
    tu.serve("on")
    out = ask("Say hello in one short sentence.")
    assert states.MARKER not in out["content"], out["content"][:200]
    assert states.MARKER in out["reasoning"], "the lines go to reasoning_content"
    tags = ask("Say hello in one short sentence.", reasoning_format="tags")
    assert states.MARKER not in tags["content"], tags["content"][:200]
    flat = ask("Say hello in one short sentence.", stream=False, reasoning_format="tags")
    assert states.MARKER not in flat["content"], flat["content"][:200]


# ------------------------------------------------------------------ 5 no tools, no tool_calls

def test_5_no_tools_no_tool_calls():
    cond, _ = tu.serve("on")
    for stream in (True, False):
        out = ask("What is my UX1 pet? FAKE_CALL_NOTOOLS", stream=stream)
        assert out["calls"] == [] and '"tool_calls"' not in out["raw"], out["raw"][-400:]
        assert "<tool_call>" not in out["content"] and "/v1/memory" not in out["content"]
        assert out["content"].startswith("Let me check my memory"), out["content"]
        assert finish_of(out) == "stop", finish_of(out)
    # with tools offered, the same answer is a call
    TOOL = [{"type": "function", "function": {"name": "tool", "parameters": {
        "type": "object", "properties": {"query": {"type": "string"}}}}}]
    out = ask("What is my UX1 pet? FAKE_CALL_NOTOOLS", tools=TOOL)
    assert out["calls"] and finish_of(out) == "tool_calls", out["raw"][-300:]


# ------------------------------------------------------------------ 6 tools and a schema

WEATHER = {"type": "object", "properties": {"location": {"type": "string"},
                                            "temperature_celsius": {"type": "number"}},
           "required": ["location", "temperature_celsius"], "additionalProperties": False}
JS = {"type": "json_schema", "json_schema": {"name": "weather", "strict": True, "schema": WEATHER}}
TOOLS2 = [{"type": "function", "function": {"name": n, "parameters": {
    "type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}}
    for n in ("search", "get_weather")]
DOC = '{"location": "Tokyo", "temperature_celsius": 28}'
CALL = ("<tool_call>\n<function=search>\n<parameter=query>\nTokyo\n</parameter>\n</function>\n"
        "</tool_call>")
PATHS = {"conductor": {}, "raw": {"X-Tandem": "raw"}}


def serve_schema():
    """The fake engine under the conductor; a one-byte-a-character vocabulary for the masks (the
    fake writes its script, so the constraint is read off the request, not applied)."""
    tu.serve("on")
    from server import fake_engine as fe
    size = app.STATE["engine"].cfg.vocab_size
    ascii_ = [bytes([i]) if 0 < i < 128 else None for i in range(128)]
    app.STATE["grammar_vocab"] = G.Vocab(ascii_, size)
    # the calls vocabulary spells the call markers (single ids in the fake, as in Qwen's)
    marks = ascii_ + [None] * (fe.CALL_CLOSE + 1 - len(ascii_))
    marks[fe.CALL_OPEN], marks[fe.CALL_CLOSE] = b"<tool_call>", b"</tool_call>"
    app.STATE["grammar_vocab_calls"] = G.Vocab(marks, size)


def schema_ask(text, headers, stream, **extra):
    """(the client's view, the Constraint the engine was given)."""
    seen, branches = [], []
    real, real_branch = app.generate_stream, G.Grammar.branch

    def recording(prompt, *a, **k):
        seen.append(k.get("pen"))
        return real(prompt, *a, **k)

    def branch(self, *a, **k):
        branches.append(real_branch(self, *a, **k))
        return branches[-1]

    app.generate_stream, G.Grammar.branch = recording, branch
    try:
        body = dict({"tools": TOOLS2, "response_format": JS}, **extra)
        out = ask(text, headers, stream, **body)
    finally:
        app.generate_stream, G.Grammar.branch = real, real_branch
    out["branches"] = branches
    assert out["head"].startswith("HTTP/1.1 200"), (out["head"], out["raw"][:300])
    cons = None
    for pen in seen:
        for p in (pen.procs if isinstance(pen, G.LogitChain) else [pen]):
            if isinstance(p, G.Constraint):
                cons = p
    return out, cons


def names_of(out):
    return [((c.get("function") or {}).get("name")) for c in out["calls"]
            if (c.get("function") or {}).get("name")]


def _in(cons, text):
    d = cons.g.dfa
    return bool(d.accept[d.walk(d.start, text.encode())])


def test_6_tools_and_a_schema_answer_the_bare_document():
    serve_schema()
    for path, hx in PATHS.items():
        for h in (CURL, SDK):
            for stream in (True, False):
                out, cons = schema_ask("Weather in Tokyo as JSON. FAKE_JSON_ANSWER " + DOC,
                                       dict(h, **hx), stream)
                where = (path, h["User-Agent"], stream)
                assert out["content"] == DOC, (where, out["content"][:200])
                assert out["calls"] == [] and finish_of(out) == "stop", (where, out["raw"][-300:])
                assert "<think>" not in out["content"] and states.MARKER not in out["content"]
                if path == "raw":
                    assert "The question has" in out["reasoning"], (where, out["reasoning"][:80])
                assert cons is not None, (where, "the request was not constrained")
                assert cons.g.vocab is app.STATE["grammar_vocab_calls"], where
                assert cons.g.side is not None and out["branches"] == ["document"], \
                    (where, out["branches"])
                assert _in(cons, "\n\n" + DOC) and _in(cons, CALL) and _in(cons, CALL + "\n" + CALL)
                assert not _in(cons, "```json\n" + DOC + "\n```") and \
                    not _in(cons, "Here it is:\n" + DOC) and \
                    not _in(cons, '{"location": "Tokyo"}'), "fences, prose, a missing key"


def test_6b_tools_and_a_schema_still_call():
    serve_schema()
    one = 'FAKE_CALL_JSON {"name": "search", "arguments": {"query": "Tokyo"}}'
    for path, hx in PATHS.items():
        for stream in (True, False):
            where = (path, stream)
            out, cons = schema_ask("Weather in Tokyo as JSON. " + one, dict(CURL, **hx), stream)
            assert names_of(out) == ["search"] and finish_of(out) == "tool_calls", \
                (where, out["raw"][-300:])
            assert "<tool_call>" not in out["content"] and "{" not in out["content"], out["content"]
            assert _in(cons, "\n\n" + CALL) and out["branches"] == ["calls"], out["branches"]
            # two calls (FAKE_TWO_CALLS writes one to each tool)
            out, cons = schema_ask("Use a tool twice FAKE_TWO_CALLS", dict(CURL, **hx), stream)
            assert names_of(out) == ["search", "get_weather"], (where, names_of(out))
            assert finish_of(out) == "tool_calls"
            out, cons = schema_ask("Use a tool twice FAKE_TWO_CALLS", dict(CURL, **hx), stream,
                                   parallel_tool_calls=False)
            assert names_of(out) == ["search"], (where, names_of(out))
            assert _in(cons, CALL) and not _in(cons, CALL + "\n" + CALL), "one call, the language"


def test_6c_tool_choice_none_and_required():
    serve_schema()
    for path, hx in PATHS.items():
        for stream in (True, False):
            where = (path, stream)
            out, cons = schema_ask("Weather in Tokyo as JSON. FAKE_JSON_ANSWER " + DOC,
                                   dict(CURL, **hx), stream, tool_choice="none")
            assert out["content"] == DOC and out["calls"] == [], (where, out["content"][:200])
            assert _in(cons, DOC) and not _in(cons, CALL), "none: the document alone"
            assert cons.g.side is None and out["branches"] == [], "no branch to pick"
            assert cons.g.vocab is app.STATE["grammar_vocab"]
            for tc in ("required", {"type": "function", "function": {"name": "search"}}):
                out, cons = schema_ask(
                    'Weather in Tokyo. FAKE_CALL_JSON {"name": "search", "arguments": '
                    '{"query": "Tokyo"}}', dict(CURL, **hx), stream, tool_choice=tc)
                assert names_of(out) == ["search"] and finish_of(out) == "tool_calls", \
                    (where, tc, out["raw"][-300:])
                assert _in(cons, CALL) and not _in(cons, DOC), "required/named: the calls alone"


def test_6d_json_object_with_tools():
    serve_schema()
    for path, hx in PATHS.items():
        for stream in (True, False):
            out, cons = schema_ask("Weather in Tokyo as JSON. FAKE_JSON_ANSWER " + DOC,
                                   dict(SDK, **hx), stream, response_format={"type": "json_object"})
            assert out["content"] == DOC and json.loads(out["content"])["temperature_celsius"] == 28
            assert out["calls"] == [] and finish_of(out) == "stop", (path, stream)
            assert _in(cons, DOC) and _in(cons, CALL) and not _in(cons, "```json\n{}\n```")


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"  {name:70s} ok   {fn() or ''}", flush=True)
            passed += 1
    print(f"{passed} passed")
