"""Tokenising a chat prompt a message at a time, with the messages the last requests already had
kept.

An agent client sends one conversation that grows by a few hundred tokens a turn, and the server
tokenised all of it again every turn: 130 to 190 ms at 130k to 190k tokens, inside the engine
lock, before the prefill starts. Almost all of that text was tokenised the turn before.

WHY THE PIECES ADD UP TO THE WHOLE. A Hugging Face tokenizer takes its added tokens out of the
text first and runs the normaliser, the pre-tokeniser and the model on the pieces between them,
one piece at a time. `<|im_start|>` and `<|im_end|>` are added tokens with no strip flags and no
normalisation, and the chat template puts one before and after every message. So the ids of the
rendered prompt are the ids of the pieces between those two tokens, in order, with the two
tokens' own ids between them -- which is what `encode` computes, looking each piece up first.
`tests/test_tokcache.py` checks it against the tokenizer's own answer on the served tokenizer.
A tokenizer without those two tokens, or with flags on them, is not split: `encode` then
tokenises the whole text, as before.
"""

from __future__ import annotations

import re
import threading
from collections import OrderedDict

SPLIT_TOKENS = ("<|im_start|>", "<|im_end|>")


def _split_ids(tok) -> dict[str, int]:
    """The split tokens this tokenizer has as plain added tokens, with their ids; {} if any is
    missing or carries a flag that changes how the text around it is cut."""
    added = getattr(tok, "added_tokens_decoder", None) or {}
    found: dict[str, int] = {}
    for tid, at in added.items():
        content = getattr(at, "content", None)
        if content in SPLIT_TOKENS:
            if (getattr(at, "lstrip", False) or getattr(at, "rstrip", False)
                    or getattr(at, "single_word", False) or getattr(at, "normalized", False)):
                return {}
            found[content] = int(tid)
    return found if len(found) == len(SPLIT_TOKENS) else {}


class SegmentCache:
    """`encode(text)`: the tokenizer's ids for `text`, a message at a time, cached by content."""

    def __init__(self, tok, max_tokens: int = 4_000_000):
        self.tok = tok
        self.ids = _split_ids(tok)
        self.pattern = (re.compile("(" + "|".join(re.escape(s) for s in self.ids) + ")")
                        if self.ids else None)
        self.max_tokens = int(max_tokens)
        self._d: OrderedDict[str, list[int]] = OrderedDict()
        self.tokens = 0
        self.stats = {"calls": 0, "pieces": 0, "hits": 0, "tokens_cached": 0,
                      "tokens_encoded": 0}
        self.lock = threading.Lock()

    def _encode_piece(self, text: str) -> list[int]:
        return list(self.tok(text, add_special_tokens=False)["input_ids"])

    def encode(self, text: str) -> list[int]:
        with self.lock:
            self.stats["calls"] += 1
            if self.pattern is None or self.max_tokens <= 0:
                ids = self._encode_piece(text)
                self.stats["tokens_encoded"] += len(ids)
                return ids
            out: list[int] = []
            for part in self.pattern.split(text):
                if not part:
                    continue
                sid = self.ids.get(part)
                if sid is not None:
                    out.append(sid)
                    continue
                self.stats["pieces"] += 1
                got = self._d.get(part)
                if got is not None:
                    self._d.move_to_end(part)
                    self.stats["hits"] += 1
                    self.stats["tokens_cached"] += len(got)
                else:
                    got = self._encode_piece(part)
                    self.stats["tokens_encoded"] += len(got)
                    self._d[part] = got
                    self.tokens += len(got)
                    while self.tokens > self.max_tokens and self._d:
                        _, old = self._d.popitem(last=False)
                        self.tokens -= len(old)
                out.extend(got)
            return out

    def report(self) -> dict:
        with self.lock:
            return {"entries": len(self._d), "tokens": self.tokens, "max_tokens": self.max_tokens,
                    "split": bool(self.pattern), **self.stats}


def opens_think_ids(tok, ids, opens_think, window: int = 64) -> bool:
    """`opens_think(tok.decode(ids))` without decoding the whole prompt.

    The answer is decided by the last reasoning tag in the text. When the decoded tail of the
    prompt holds either tag, the last one in the tail is the last one in the text, so the tail
    answers; a tag cut in half by the window's start is found in neither and so cannot mislead.
    Only a tail with no tag at all needs the whole text."""
    from server.stream import CLOSE_THINK, OPEN_THINK
    tail = tok.decode(list(ids[-window:]), skip_special_tokens=False)
    if OPEN_THINK in tail or CLOSE_THINK in tail:
        return opens_think(tail)
    return opens_think(tok.decode(list(ids), skip_special_tokens=False))
