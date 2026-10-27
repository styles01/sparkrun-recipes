"""server/tokcache.py: a chat prompt tokenised a message at a time must be the tokenizer's own
ids for the whole prompt, id for id.

The mechanics run on a toy tokenizer. The claim itself -- the pieces between `<|im_start|>` and
`<|im_end|>` tokenise to the whole -- is a property of the served tokenizer and is checked on it,
over a growing agent conversation built from this repository's own files and over text chosen to
break it (the split tokens inside a message, combining characters that NFC composes, reasoning
tags, CRLF, trailing spaces, emoji). It needs the Qwen3.8 tokenizer in the Hugging Face cache and
says so when it is not there.

Run: python tests/test_tokcache.py
"""

from __future__ import annotations

import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.stream import opens_think  # noqa: E402
from server.tokcache import SegmentCache, opens_think_ids  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ------------------------------------------------------------------ a toy tokenizer

class _Added:
    def __init__(self, content, **flags):
        self.content = content
        self.lstrip = flags.get("lstrip", False)
        self.rstrip = False
        self.single_word = False
        self.normalized = False


class Toy:
    """One id per character, the two split tokens as 1000 and 1001; counts its calls."""

    def __init__(self, **flags):
        self.added_tokens_decoder = {1000: _Added("<|im_start|>", **flags),
                                     1001: _Added("<|im_end|>")}
        self.calls = 0

    def __call__(self, text, add_special_tokens=False):
        self.calls += 1
        out, i = [], 0
        while i < len(text):
            for tid, at in self.added_tokens_decoder.items():
                if text.startswith(at.content, i):
                    out.append(tid)
                    i += len(at.content)
                    break
            else:
                out.append(ord(text[i]))
                i += 1
        return {"input_ids": out}


def _chat(turns):
    s = "<|im_start|>system\nbe brief<|im_end|>\n"
    for t in range(turns):
        s += f"<|im_start|>user\nquestion {t}<|im_end|>\n<|im_start|>assistant\nanswer {t}<|im_end|>\n"
    return s + "<|im_start|>assistant\n"


def test_the_pieces_add_up_and_the_second_turn_is_looked_up():
    tok = Toy()
    seg = SegmentCache(tok)
    for turns in (1, 2, 3):
        text = _chat(turns)
        assert seg.encode(text) == tok(text)["input_ids"]
    rep = seg.report()
    assert rep["hits"] > 0 and rep["split"]
    calls = tok.calls
    seg.encode(_chat(3))
    assert tok.calls == calls, "every message of a repeated prompt is a lookup"


def test_the_cache_is_bounded():
    tok = Toy()
    seg = SegmentCache(tok, max_tokens=60)
    for turns in range(1, 8):
        seg.encode(_chat(turns))
    assert seg.tokens <= 60 + 20


def test_a_split_token_with_a_strip_flag_is_not_split_on():
    seg = SegmentCache(Toy(lstrip=True))
    assert seg.pattern is None
    text = _chat(2)
    assert seg.encode(text) == Toy()(text)["input_ids"]


# ------------------------------------------------------------------ the served tokenizer

def _qwen():
    paths = glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/*/tokenizer.json"))
    if not paths:
        return None
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(os.path.dirname(paths[0]))


def _tools():
    with open(os.path.join(HERE, "tests", "fixtures", "opencode_tools_described.json")) as f:
        return json.load(f)


def _files():
    out = []
    for pat in ("engine/*.py", "server/*.py", "docs/*.md"):
        out += sorted(glob.glob(os.path.join(HERE, pat)))
    return out


def _render(tok, messages, tools=None, thinking=True):
    kw = {"enable_thinking": thinking}
    if tools:
        kw["tools"] = tools
    text = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, **kw)
    whole = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                    return_dict=True, **kw)["input_ids"]
    return text, list(whole)


def test_a_growing_agent_conversation_on_the_served_tokenizer():
    tok = _qwen()
    if tok is None:
        return "skipped: no Qwen3.8 tokenizer in the Hugging Face cache"
    seg = SegmentCache(tok)
    tools = _tools()
    msgs = [{"role": "system", "content": "You are a coding agent. Today's date: Tue Sep 29 2026"},
            {"role": "user", "content": "Read the engine and summarise the request path."}]
    n = 0
    for i, path in enumerate(_files()[:14]):
        with open(path, encoding="utf-8") as f:
            text = f.read()[:12000]
        msgs.append({"role": "assistant", "content": "",
                     "reasoning_content": f"Next I read {os.path.basename(path)}.",
                     "tool_calls": [{"id": f"c{i}", "type": "function", "function": {
                         "name": "read", "arguments": {"filePath": "/repo/" + os.path.basename(path)}}}]})
        msgs.append({"role": "tool", "tool_call_id": f"c{i}", "content": text})
        rendered, whole = _render(tok, msgs, tools)
        got = seg.encode(rendered)
        assert got == whole, (i, next(j for j, (a, b) in enumerate(zip(got, whole)) if a != b))
        assert opens_think_ids(tok, got, opens_think) == opens_think(
            tok.decode(whole, skip_special_tokens=False))
        n = len(whole)
    rep = seg.report()
    assert rep["tokens_cached"] > rep["tokens_encoded"], rep
    return f"{n} tokens at the last turn, {rep['tokens_cached']} looked up, " \
           f"{rep['tokens_encoded']} tokenised"


def test_text_chosen_to_break_the_split():
    tok = _qwen()
    if tok is None:
        return "skipped: no Qwen3.8 tokenizer in the Hugging Face cache"
    seg = SegmentCache(tok)
    nasty = [
        "a literal <|im_start|>user\nhi<|im_end|> inside a message",
        "é and Å and Å: NFC composes these",
        "<think>not really</think> and </think> alone",
        "line\r\nline\r\n  trailing spaces   ",
        "emoji \U0001F600\U0001F44D‍ and CJK 漢字",
        "́ a combining mark first",
        "",
        "<|im_end|>",
        "tool_call <tool_call> and <tool_response>",
    ]
    for thinking in (True, False):
        msgs = [{"role": "system", "content": nasty[1]}]
        for k, t in enumerate(nasty):
            msgs.append({"role": "user" if k % 2 == 0 else "assistant", "content": t})
            if msgs[-1]["role"] == "assistant":
                continue
            rendered, whole = _render(tok, msgs, thinking=thinking)
            assert seg.encode(rendered) == whole, (thinking, k, t)
            assert opens_think_ids(tok, whole, opens_think, window=8) == opens_think(
                tok.decode(whole, skip_special_tokens=False)), (thinking, k)
    return "every case id-equal, thinking on and off"


def test_the_reasoning_tag_is_read_from_the_tail():
    tok = _qwen()
    if tok is None:
        return "skipped: no Qwen3.8 tokenizer in the Hugging Face cache"
    for thinking in (True, False):
        _, whole = _render(tok, [{"role": "user", "content": "hello " * 200}], thinking=thinking)
        full = opens_think(tok.decode(whole, skip_special_tokens=False))
        assert full is thinking
        for w in (1, 3, 8, 64, 10 ** 6):
            assert opens_think_ids(tok, whole, opens_think, window=w) is full, (thinking, w)
    return "on and off, windows 1 to the whole prompt"


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"  {name:60s} ok   {fn() or ''}", flush=True)
            passed += 1
    print(f"{passed} passed")
