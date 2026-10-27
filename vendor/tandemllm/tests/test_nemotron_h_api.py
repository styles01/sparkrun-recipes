"""NemotronH's chat formats: the checkpoint's own template (Nemotron 3.5 Lightning's) renders thinking on
as an open `<think>` and thinking off as `<think></think>`, and an assistant tool call as the XML
`<tool_call><function=NAME><parameter=K>` form, which the server's default parser reads back to the same
call; through the real chat handler (the fake engine in this process, `tool_format` "xml" as the
NemotronH adapter sets it) a call comes back as tool_calls and the thinking apart from the answer.

Run: python tests/test_nemotron_h_api.py
"""

from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from server import app  # noqa: E402
from server.toolcall import parse_tool_calls  # noqa: E402
from test_toolcheck import Req, serve  # noqa: E402

TEMPLATE = os.path.join(HERE, "fixtures", "nemotron_h", "lightning-chat_template.jinja")
TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "the weather in a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
                   "required": ["city"]}}}]


def render(messages, **kw) -> str | None:
    try:
        from transformers.utils.chat_template_utils import render_jinja_template
    except Exception:  # noqa: BLE001
        return None
    with open(TEMPLATE) as f:
        tpl = f.read()
    out, _ = render_jinja_template(conversations=[messages], chat_template=tpl, **kw)
    return out[0]


def test_template_thinking_on_and_off():
    msgs = [{"role": "user", "content": "hi"}]
    on = render(msgs, add_generation_prompt=True, enable_thinking=True)
    if on is None:
        return "skipped: transformers' jinja renderer missing"
    off = render(msgs, add_generation_prompt=True, enable_thinking=False)
    assert on.endswith("<|im_start|>assistant\n<think>\n"), on[-60:]
    assert off.endswith("<|im_start|>assistant\n<think></think>"), off[-60:]
    assert app.opens_think(on) and not app.opens_think(off)


def test_template_tool_call_round_trips_through_the_parser():
    call = {"name": "get_weather", "arguments": {"city": "Berlin", "days": 3}}
    msgs = [{"role": "user", "content": "weather?"},
            {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": call}]},
            {"role": "tool", "content": "sunny"}]
    text = render(msgs, tools=TOOLS, add_generation_prompt=True, enable_thinking=False)
    if text is None:
        return "skipped: transformers' jinja renderer missing"
    a = text.index("<|im_start|>assistant")
    block = text[a:text.index("<|im_end|>", a)]
    assert "<tool_call>\n<function=get_weather>\n<parameter=city>\nBerlin\n</parameter>" in block, block
    content, calls = parse_tool_calls(block.split("\n", 1)[1], names=["get_weather"],
                                      types={"get_weather": {"days": "integer"}})
    assert len(calls) == 1 and calls[0]["function"]["name"] == "get_weather", calls
    args = json.loads(calls[0]["function"]["arguments"])
    assert args["city"] == "Berlin" and int(args["days"]) == 3, args
    assert "<tool_response>\nsunny\n</tool_response>" in text


def ask(text: str, stream: bool, think: bool = True) -> dict:
    serve("raw")
    app.STATE["tool_format"] = "xml"
    app.STATE["reasoning_format"] = "reasoning_content"
    body = {"messages": [{"role": "user", "content": text}], "stream": stream, "tools": TOOLS,
            "max_tokens": 2000, "chat_template_kwargs": {"enable_thinking": think}}
    head, raw = Req(body).response()
    out = {"content": "", "reasoning": "", "calls": [], "finish": None}
    if not stream:
        m = json.loads(raw)["choices"][0]
        out.update(content=m["message"].get("content") or "", reasoning=m["message"].get("reasoning_content") or "",
                   calls=[c["function"]["name"] for c in m["message"].get("tool_calls") or []],
                   finish=m["finish_reason"])
        return out
    names = {}
    for ln in raw.splitlines():
        if ln.startswith("data: {"):
            for ch in json.loads(ln[6:])["choices"]:
                d = ch["delta"]
                out["content"] += d.get("content") or ""
                out["reasoning"] += d.get("reasoning_content") or ""
                for tc in d.get("tool_calls") or []:
                    names[tc["index"]] = names.get(tc["index"], "") + ((tc.get("function") or {}).get("name") or "")
                out["finish"] = ch["finish_reason"] or out["finish"]
    out["calls"] = [names[i] for i in sorted(names)]
    return out


def test_handler_tool_call_and_reasoning():
    for stream in (False, True):
        out = ask("use the tool for Berlin", stream)
        assert out["finish"] == "tool_calls" and out["calls"] == ["get_weather"], (stream, out)
        assert "<tool_call>" not in out["content"] and out["reasoning"].startswith("The question has")
        out = ask("say something", stream, think=False)
        assert out["reasoning"] == "" and "<think>" not in out["content"] and out["content"].strip()
        assert out["finish"] == "stop"


def test_bolt_profile_names_its_card_and_prices():
    from server import model_card
    from engine.models.nemotron_h import serving
    old = os.environ.get("QSE_WEIGHTS")
    try:
        os.environ["QSE_WEIGHTS"] = "nemotron-bolt-iq2"
        assert serving.model_id("/x") == "nemotron-3.5-bolt-74b-a9b"
        os.environ["QSE_WEIGHTS"] = "nemotron-lightning-bf16"
        assert serving.model_id("/x") == "nemotron-3.5-lightning-30b-a3b"
    finally:
        if old is None:
            os.environ.pop("QSE_WEIGHTS", None)
        else:
            os.environ["QSE_WEIGHTS"] = old
    c = model_card.find("nemotron-bolt-iq2")
    assert c["label"] == "Nemotron 3.5 Bolt 74B-A9B" and not c["capabilities"]["vision"]


if __name__ == "__main__":
    fails = 0
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        try:
            print(f"  {name:<52} ok   {fn() or ''}")
        except AssertionError as e:
            fails += 1
            print(f"  {name:<52} FAIL {e}")
    print(f"{len(tests) - fails} passed" + (f", {fails} FAILED" if fails else ""))
    sys.exit(1 if fails else 0)
