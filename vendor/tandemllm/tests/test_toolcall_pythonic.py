"""SRV-425: LFM2's Pythonic tool calls (`<|tool_call_start|>[f(a=1)]<|tool_call_end|>`) read as OpenAI
tool_calls through server/toolcall.py's own machinery."""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.toolcall import (PY_CLOSE, PY_OPEN, PythonicCalls, ToolCallBuffer, parse_tool_calls,  # noqa: E402
                             pythonic_calls, rewrite_pythonic)


def _args(calls, i=0):
    return json.loads(calls[i]["function"]["arguments"])


def test_one_call():
    text = f"Let me check.{PY_OPEN}[get_weather(city='Berlin')]{PY_CLOSE}"
    content, calls = parse_tool_calls(rewrite_pythonic(text))
    assert content.strip() == "Let me check."
    assert calls[0]["function"]["name"] == "get_weather" and _args(calls) == {"city": "Berlin"}


def test_values_and_several_calls():
    inner = ("[search(q=\"a, b) c\", limit=3, exact=True, tags=['x', \"y\"], opts={'k': [1, 2.5, None]}), "
             "read_file(path='/tmp/it\\'s.txt', offset=-1, strict=false)]")
    got = pythonic_calls(inner)
    assert [c["name"] for c in got] == ["search", "read_file"]
    assert got[0]["arguments"] == {"q": "a, b) c", "limit": 3, "exact": True, "tags": ["x", "y"],
                                   "opts": {"k": [1, 2.5, None]}}
    assert got[1]["arguments"] == {"path": "/tmp/it's.txt", "offset": -1, "strict": False}
    content, calls = parse_tool_calls(rewrite_pythonic(PY_OPEN + inner + PY_CLOSE))
    assert len(calls) == 2 and _args(calls, 1)["offset"] == -1 and not content.strip()


def test_not_calls_stay_text():
    for inner in ("[get_weather('Berlin')]", "[__import__('os').system('ls')]", "not python (", "[1, 2]",
                  "[f(**kw)]", "[f(x={1, 2})]"):
        text = f"a {PY_OPEN}{inner}{PY_CLOSE} b"
        assert rewrite_pythonic(text) == text, inner
    assert pythonic_calls("[mod.fn(a=1)]")[0]["name"] == "mod.fn"


def test_stream_split_anywhere():
    text = f"Sure.{PY_OPEN}[f(a=1), g(b='x')]{PY_CLOSE} done"
    want = rewrite_pythonic(text)
    for size in (1, 2, 3, 7, 50):
        p = PythonicCalls()
        out = "".join(p.feed(text[i:i + size]) for i in range(0, len(text), size)) + p.finish()
        assert out == want, size
        assert p.blocks == 1


def test_stream_into_the_tool_buffer():
    text = f"Checking.{PY_OPEN}[get_weather(city='Paris', days=2)]{PY_CLOSE}"
    p, tb = PythonicCalls(), ToolCallBuffer(names={"get_weather"})
    content, deltas = "", []
    for i in range(0, len(text), 4):
        content += "".join(tb.feed(p.feed(text[i:i + 4])))
        deltas += tb.drain_deltas()
    content += "".join(tb.feed(p.finish(eos=True)))
    left, sweep = tb.finish(eos=True)
    content += left
    deltas += sweep
    assert content.strip() == "Checking."
    calls = [d for d in deltas if d.get("function", {}).get("name")]
    assert calls[0]["function"]["name"] == "get_weather"
    args = "".join(d["function"].get("arguments", "") for d in deltas)
    assert json.loads(args) == {"city": "Paris", "days": 2}


def test_open_block_at_the_end():
    p = PythonicCalls()
    assert p.feed(f"x{PY_OPEN}[f(a=1)]") == "x"
    assert p.finish(eos=True).startswith("<tool_call>")
    p = PythonicCalls()
    p.feed(f"x{PY_OPEN}[f(a=1)]")
    assert p.finish(eos=False) == f"{PY_OPEN}[f(a=1)]"       # cut by the limit: text
    p = PythonicCalls()
    assert p.feed("tail <|tool_c") == "tail " and p.finish() == "<|tool_c"


if __name__ == "__main__":
    fails = 0
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        try:
            print(f"  {name:<44} ok   {fn() or ''}")
        except AssertionError as e:
            fails += 1
            print(f"  {name:<44} FAIL {e}")
    print(f"{len(tests) - fails} passed" + (f", {fails} FAILED" if fails else ""))
    sys.exit(1 if fails else 0)
