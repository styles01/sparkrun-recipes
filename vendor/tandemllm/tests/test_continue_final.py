"""FX15 (UX3 W17, 2026-10-08): Open WebUI's "Continue Response" continues the partial answer.

The live case (0.4.6-test, chatcmpl-9ea66357): after Stop, Continue sent the conversation with the
partial answer as the last message (7,200 tokens). `build_prompt` rendered every prompt with a
generation prompt, so the partial was closed (`<|im_end|>`) and a new `<|im_start|>assistant\\n<think>`
started: 77 reasoning tokens, then 1,613 new tokens from the title again; the saved answer held the
story twice.

The test matrix (the real Qwen template, one id per character):

  1  a trailing assistant message with text is continued: the prompt ends with its text, inside its
     own turn (no `<|im_end|>` after it, no new assistant turn), after the template's closed think
     block, so the prompt does not end inside `<think>` and no new reasoning starts
  2  the head of the prompt is the same tokens as the earlier turn's prompt up to the user turn's
     end, so the caches resume as before; the router's note goes at the user turn's end
  3  a trailing assistant with tool calls, with no text, or with `continue_final_message: false`
     (or `add_generation_prompt: true`) renders as before; a last user message renders as before
  4  FX12's regenerate rule: a Continue is not a regenerate
  5  the fake engine's template does the same (the fake-engine suites see a continuation)

Run: python tests/test_continue_final.py
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from test_conductor_router import SYSTEM, setup  # noqa: E402

from server import app  # noqa: E402

USER = "Write a short story about a fox who learns to swim. Give it a title."
PARTIAL = "# The Fox Who Swam\n\nOnce upon a time, at the edge of a cold river, a small red fox"


def _text(body):
    ids, kind, in_think = app.build_prompt(body)
    return app.STATE["tok"].decode(ids.tolist(), skip_special_tokens=False), in_think


def test_a_partial_answer_is_continued_in_its_own_turn():
    setup()
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER},
            {"role": "assistant", "content": PARTIAL}]
    text, in_think = _text({"messages": msgs})
    assert text.endswith(PARTIAL), repr(text[-120:])
    tail = text[text.rindex("<|im_start|>assistant"):]
    assert "<|im_end|>" not in tail and tail.count("<|im_start|>") == 1, repr(tail)
    assert "<think>\n\n</think>\n\n" + PARTIAL in tail, repr(tail)
    assert not in_think, "no new reasoning: the partial is past its think block"
    assert app.continues_final({"messages": msgs})


def test_the_head_is_the_earlier_prompt_and_the_note_goes_at_the_user_turn():
    setup()
    first = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER}]
    t1, _ = _text({"messages": first})
    t2, _ = _text({"messages": first + [{"role": "assistant", "content": PARTIAL}]})
    user_end = t1.rindex("<|im_end|>")
    assert t2[:user_end + len("<|im_end|>")] == t1[:user_end + len("<|im_end|>")]
    from server.conductor import inject
    tok = app.STATE["tok"]
    ids2 = app.build_prompt({"messages": first + [{"role": "assistant", "content": PARTIAL}]})[0]
    at = inject.insertion_point(ids2.tolist(), inject.special_id(tok, inject.IM_END))
    assert tok.decode(ids2.tolist()[:at], skip_special_tokens=False) == t1[:user_end]


def test_other_shapes_render_as_before():
    setup()
    base = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER}]
    gen_tail = "<|im_start|>assistant\n<think>\n"
    for last, body_extra in (
            ({"role": "assistant", "content": "", "tool_calls": [
                {"type": "function", "function": {"name": "f", "arguments": "{}"}}]}, {}),
            ({"role": "assistant", "content": "   "}, {}),
            ({"role": "assistant", "content": PARTIAL}, {"continue_final_message": False}),
            ({"role": "assistant", "content": PARTIAL}, {"add_generation_prompt": True})):
        body = dict(body_extra, messages=base + [last])
        assert not app.continues_final(body), body_extra
        text, in_think = _text(body)
        assert text.endswith(gen_tail) and in_think, (body_extra, repr(text[-80:]))
    text, in_think = _text({"messages": base})
    assert text.endswith(gen_tail) and in_think
    # vLLM's explicit form continues too
    body = {"messages": base + [{"role": "assistant", "content": PARTIAL}],
            "continue_final_message": True, "add_generation_prompt": False}
    assert app.continues_final(body) and _text(body)[0].endswith(PARTIAL)


def test_a_continue_is_not_a_regenerate():
    from server import regenerate
    cont = [{"role": "user", "content": USER}, {"role": "assistant", "content": PARTIAL}]
    assert regenerate.signature(cont) is None, "a Continue has no regenerate signature"
    assert regenerate.signature(cont[:1]) is not None


def test_the_fake_template_continues_too():
    from server.fake_engine import FakeTokenizer
    tok = FakeTokenizer()
    msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "Hel"}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=False, continue_final_message=True)
    text = tok.decode(ids["input_ids"][0].tolist(), skip_special_tokens=False)
    assert text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\nHel"), repr(text)


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:64s} ok")
            passed += 1
    print(f"{passed} passed")
