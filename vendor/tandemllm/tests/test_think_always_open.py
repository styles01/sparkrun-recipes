"""SRV-425: a model whose chat template always opens the reasoning block (LFM2.5's ends every generation
prompt with `<think>`, whatever `enable_thinking` says) still honours a client's thinking off: the
block is closed right after the generation prompt, as the Qwen template's own thinking off renders it,
so the prompt does not end inside `<think>` and the answer starts at once. A client that leaves
thinking on, and the Qwen template in every case, render as before.

Run: python tests/test_think_always_open.py
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import torch  # noqa: E402
from test_conductor_router import QTok, body, by_label, setup, turn  # noqa: E402

from server import app  # noqa: E402

ALWAYS = ("{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"
          "{% if add_generation_prompt %}<|im_start|>assistant\n<think>{% endif %}")


class _Enc(dict):
    """What a Hugging Face tokenizer returns: the ids by key and as an attribute."""

    @property
    def input_ids(self):
        return self["input_ids"]


class AlwaysThinks(QTok):
    """The test tokenizer with a template that ignores enable_thinking, as LFM2.5's does."""

    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        ids = self.encode(text or "")
        return _Enc(input_ids=torch.tensor([ids]) if return_tensors else ids)

    def apply_chat_template(self, messages, add_generation_prompt=True, return_tensors=None,
                            return_dict=False, tools=None, tokenize=True, **kw):
        from transformers.utils.chat_template_utils import render_jinja_template
        text = render_jinja_template([messages], chat_template=ALWAYS, add_generation_prompt=add_generation_prompt)[0][0]
        if not tokenize:
            return text
        return {"input_ids": torch.tensor([self.encode(text)])}


MSGS = [{"role": "user", "content": "Name three rivers."}]


def _render(tok, body, segments=False):
    setup()
    app.STATE["tok"] = tok
    if segments:
        from server.tokcache import SegmentCache
        app.STATE["segments"] = SegmentCache(tok)
    ids, kind, in_think = app.build_prompt(body)
    return tok.decode(ids.tolist(), skip_special_tokens=False), in_think


def test_thinking_off_closes_a_block_the_template_opened():
    for seg in (False, True):
        text, in_think = _render(AlwaysThinks(), {"messages": MSGS, "chat_template_kwargs": {"enable_thinking": False}},
                                 segments=seg)
        assert text.endswith("<|im_start|>assistant\n<think></think>"), repr(text[-60:])
        assert not in_think
        n = app.build_prompt({"messages": MSGS, "chat_template_kwargs": {"enable_thinking": False}}, count=True)
        assert n == len(AlwaysThinks().encode(text))


def test_thinking_on_and_the_qwen_template_are_unchanged():
    text, in_think = _render(AlwaysThinks(), {"messages": MSGS})
    assert text.endswith("<|im_start|>assistant\n<think>") and in_think
    text, in_think = _render(QTok(), {"messages": MSGS, "chat_template_kwargs": {"enable_thinking": False}})
    assert text.endswith("<think>\n\n</think>\n\n") and not in_think, repr(text[-40:])


def test_the_router_closes_the_block_the_template_opened():
    """A class the router serves with thinking off (a summary) closes the bare `<think>` the same
    way, so the conductor's effort reaches a model whose template always opens the block."""
    cond, _ = setup(decider=by_label("summarize", 0.88))
    app.STATE["tok"] = AlwaysThinks()
    msgs = [{"role": "user", "content": "Summarize the text below in five bullet points.\n\n"
                                        + "The engine reads one sequence. " * 20}]
    plan, ids, clean = turn(cond, body(msgs, tools=False))
    tok = AlwaysThinks()
    on, got = tok.decode(clean, skip_special_tokens=False), tok.decode(ids, skip_special_tokens=False)
    assert on.endswith("assistant\n<think>"), repr(on[-40:])
    assert plan.row["thinking"] == "off" and plan.in_think_after is False
    assert got.endswith("assistant\n<think></think>"), repr(got[-40:])
    assert "thinking_off:no_open_tail" not in plan.row["extra"].get("notes", [])


if __name__ == "__main__":
    fails = 0
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        try:
            print(f"  {name:<56} ok   {fn() or ''}")
        except AssertionError as e:
            fails += 1
            print(f"  {name:<56} FAIL {e}")
    print(f"{len(tests) - fails} passed" + (f", {fails} FAILED" if fails else ""))
    sys.exit(1 if fails else 0)
