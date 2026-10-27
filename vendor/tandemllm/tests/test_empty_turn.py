"""SRV-111 / SRV-112: a request that offered tools never gets an empty stop.

The test matrix is SRV-111's Gherkin, on the served loop with the random 4-layer model and the
handler (the end token is whatever the model draws at a chosen step, so it can be scripted):

  1  the end token inside the reasoning block: the engine drops it, forwards the hidden close,
     the client receives `</think>` and the continued answer, never the phrase -- every format
  2  the end token after the reasoning with an empty answer: a hidden user turn with the nudge
     and an empty closed reasoning block; no second `<think>` on the wire. Both after the
     engine's own budget closed the block (the end token right after the streamed phrase) and
     with thinking off (the end token as the first token)
  3  two failures end in GIVE_UP, with the stronger nudge in between
  4  a text answer is left alone
  5  no tools, no action
  6  tool_choice none is respected
  7  shadow only records

and n > 1 and a stop string pass too. That the continued answer is a tool call opencode can run
is the fake engine's e2e (tests/test_fake_engine.py, `test_conductor_on_over_the_wire`): this
model writes noise.

Run: python tests/test_empty_turn.py
"""

from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import torch  # noqa: E402

from engine.spec import ThinkBudget  # noqa: E402
from server import app  # noqa: E402
from test_app_loop import Req, serve  # noqa: E402
from test_supervisor_frame import (  # noqa: E402
    CLOSE, OPEN, MonTok, _fields, monitor, policy, quiet,
)
from test_window_edge import FixedDrafter  # noqa: E402

TOOLS = [{"type": "function", "function": {"name": "read"}}]
E1 = {"E1": {"rule": "empty_turn", "mode": "act", "close": "\n\nact now.\n</think>\n\n",
             "nudges": ["call a tool", "call a tool NOW"], "tries": 2}}
PROMPT = [2, 40, 17, 66, 30]


def _natural(prompt, think_on: bool, budget: int = 0, n: int = 30):
    """What the model writes for the chat request, raw: its tokens."""
    serve(FixedDrafter(97, 5, tree_mode=False))
    app.STATE["tok"] = MonTok()
    think = ThinkBudget(MonTok(), budget=budget)
    with quiet():
        return list(app.generate_stream(torch.tensor(prompt), n, set(), think))


def _prompt(think_on: bool) -> list[int]:
    return MonTok().apply_chat_template([{"role": "user", "content": "go"}],
                                        enable_thinking=think_on)["input_ids"][0].tolist()


def _ask(end, mode: str = "on", tasks=E1, stream=True, think_on=True, fmt="tags",
         budget=None, max_tokens=40, **extra):
    """One chat request whose model ends at `end` (a token, or a list of them);
    (monitor, content, reasoning, finish)."""
    serve(FixedDrafter(97, 5, tree_mode=False))
    app.STATE["tok"] = MonTok(ends=end if isinstance(end, list) else [end])
    app.STATE["cfg_eos"] = end
    app.STATE["reasoning_format"] = fmt
    mon = app.STATE["monitor"] = monitor(mode, tasks)
    # SRV-617: the format named, as an unnamed raw client otherwise gets `reasoning_content`
    body = dict({"messages": [{"role": "user", "content": "go"}], "stream": stream,
                 "max_tokens": max_tokens, "tools": TOOLS, "reasoning_format": fmt,
                 "chat_template_kwargs": {"enable_thinking": think_on}}, **extra)
    if budget is not None:
        # the server's own --think-budget: ThinkBudget's streamed phrase. A client's budget
        # closes hidden under an acting monitor (tests/test_client_budgets.py)
        app.STATE["think_budget"] = budget
    with quiet():
        head, raw = Req("/v1/chat/completions", body).response()
    content, reasoning = _fields(raw, stream)
    if stream:
        fin = [json.loads(ln[6:])["choices"][0]["finish_reason"] for ln in raw.splitlines()
               if ln.startswith("data: {") and json.loads(ln[6:])["choices"]][-1]
    else:
        fin = json.loads(raw)["choices"][0]["finish_reason"]
    return mon, content, reasoning, fin


def _first_new(seq, lo: int) -> tuple[int, int]:
    """(position, token) of the first token from `lo` on that did not occur before it."""
    i = next(i for i in range(lo, len(seq)) if seq[i] not in seq[:i])
    return i, seq[i]


def _hidden_in_ctx(text: str) -> bool:
    ids = MonTok().encode(text)
    ctx = list(app.STATE["last_ctx"])
    return any(ctx[i:i + len(ids)] == ids for i in range(len(ctx) - len(ids) + 1))


# ------------------------------------------------------------------ 1: inside the reasoning

def test_1_the_end_token_inside_the_reasoning_block():
    out = _natural(_prompt(True), True)
    _, stop = _first_new(out, 3)                   # the model "ends" inside its reasoning
    for fmt in ("tags", "reasoning_content", "both"):
        for stream in (True, False):
            mon, content, reasoning, fin = _ask(stop, fmt=fmt, stream=stream)
            acts = mon.journal.actions()
            assert acts and acts[0]["reason"] == "empty_turn(case=A)", (fmt, stream, acts)
            assert "act now" not in content + reasoning, (fmt, stream)
            assert _hidden_in_ctx("\n\nact now.\n"), "the close phrase was forwarded"
            if fmt == "reasoning_content":
                assert "</think>" not in content + reasoning and reasoning
            else:
                assert content.count("</think>") == 1, (fmt, content)
                assert content.split("</think>")[1], "the answer continues after the close"


# ------------------------------------------------------------------ 2: after the reasoning

def test_2_an_empty_answer_after_the_clients_own_close():
    """The engine's own budget closes the block (streamed phrase, ThinkBudget's), and the
    model's very next token is its end token: a hidden turn, the answer goes on."""
    out = _natural(_prompt(True), True, budget=3, n=200)
    phrase = MonTok().encode(ThinkBudget.PHRASE)
    k = next(i for i in range(len(out)) if out[i:i + len(phrase)] == phrase) + len(phrase)
    stop = out[k]
    assert stop not in out[:k], "the token after the phrase must be new to end there"
    for stream in (True, False):
        mon, content, reasoning, fin = _ask(stop, stream=stream, budget=3, max_tokens=200)
        acts = mon.journal.actions()
        assert acts and acts[0]["reason"] == "empty_turn(case=B)", acts
        assert acts[0]["text"] == "call a tool"
        assert _hidden_in_ctx("<|im_end|>\n<|im_start|>user\ncall a tool<|im_end|>\n"
                              "<|im_start|>assistant\n<think>\n\n</think>\n\n")
        assert content.count("<think>") == 1, "no second <think> reaches the client"
        assert content.split("</think>")[1].strip(), "the answer goes on after the nudge"


def test_2_an_empty_answer_with_thinking_off():
    out = _natural(_prompt(False), False)
    stop = out[0]                                  # the end token as the first token
    mon, content, _, fin = _ask(stop, think_on=False)
    acts = mon.journal.actions()
    assert acts and acts[0]["reason"] == "empty_turn(case=B)", acts
    assert "<think>" not in content and content.strip(), content


# ------------------------------------------------------------------ 3: GIVE_UP

def test_3_two_failures_end_in_give_up():
    """Every token the model draws ends it: the first act nudges, the second nudges harder,
    the third finish is GIVE_UP -- one sentence, finish stop, journaled gave_up."""
    for stream in (True, False):
        mon, content, _, fin = _ask(list(range(97)), think_on=False, stream=stream)
        evs = mon.journal.events()
        acts = [e for e in evs if e["sp"] == "SP2"]
        assert [(e["decision"], e["text"]) for e in acts] == [
            ("CONTINUE_HIDDEN", "call a tool"), ("CONTINUE_HIDDEN", "call a tool NOW"),
            ("GIVE_UP", policy().texts["give_up"])], acts
        assert content == policy().texts["give_up"] and fin == "stop", (content, fin)
        assert evs[-1]["decision"] == "OUTCOME" and evs[-1]["outcome"] == "gave_up", evs[-1]


# ------------------------------------------------------------------ 4-7: nothing to do

def _answer_stop() -> int:
    """A token the model writes five tokens into its answer, thinking off."""
    out = _natural(_prompt(False), False)
    return _first_new(out, 5)[1]


def test_4_a_text_answer_is_left_alone():
    mon, content, _, fin = _ask(_answer_stop(), think_on=False)
    assert mon.journal.events() == [] and fin == "stop" and content, (content, fin)


def test_5_no_tools_no_action():
    out = _natural(_prompt(False), False)
    mon, content, _, fin = _ask(out[0], think_on=False, tools=[])
    assert mon.journal.events() == [] and content == "" and fin == "stop"


def test_6_tool_choice_none_is_respected():
    out = _natural(_prompt(False), False)
    mon, content, _, fin = _ask(out[0], think_on=False, tool_choice="none")
    assert mon.journal.events() == [] and content == "" and fin == "stop"


def test_7_shadow_only_records():
    out = _natural(_prompt(False), False)
    mon, content, _, fin = _ask(out[0], mode="shadow", think_on=False)
    evs = mon.journal.events()
    assert evs[0]["mode"] == "shadow" and evs[0]["decision"] == "CONTINUE_HIDDEN", evs
    assert mon.journal.actions() == [] and content == "" and fin == "stop"


def test_n_above_one_and_stop_strings_pass():
    out = _natural(_prompt(False), False)
    mon, content, _, fin = _ask(out[0], think_on=False, stream=False, n=2)
    assert mon.journal.actions() == [], "n > 1 never acts"
    # a stop string that matches at once: the client asked for that stop
    mon, content, _, fin = _ask(_answer_stop(), think_on=False, stop=[MonTok().decode(out[:1])])
    assert mon.journal.events() == [] and fin == "stop"


def test_the_outcome_says_what_came_of_the_act():
    out = _natural(_prompt(False), False)
    mon, content, _, fin = _ask(out[0], think_on=False)
    end = mon.journal.events()[-1]
    assert end["decision"] == "OUTCOME" and end["outcome"] in ("text", "empty", "gave_up"), end
    assert end["acts"] >= 1 and end["hidden_tokens"] > 0
    assert len(end["detail"]["resume_ms"]) == end["acts"], end


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:64s} ok")
            passed += 1
    print(f"{passed} passed")
