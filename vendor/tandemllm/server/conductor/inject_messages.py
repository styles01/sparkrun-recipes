"""The router's notes at the message level, for a backend that is not our engine (proxy P4).

The engine places a note as token ids before the last turn's `<|im_end|>` and replays it on every
later turn at the same position while the prompt before it hashes the same (inject.py). A backend
behind the proxy takes messages, not ids, so the same design is kept one level up:

    the note     text appended to the content of the newest user message (the router's own
                 `injection.wrapper`, which begins with two newlines and frames the text as a
                 runtime note the user did not write)
    the replay   per session, (message index, sha256 of the messages up to and including the one
                 the note sits on, the note's text); every later request whose first `index + 1`
                 messages hash the same gets the same text at the same index

So the bytes the backend reads for the head of the conversation are the bytes it read on the turn
the note was added, and its own prefix cache (vLLM, SGLang, a cloud provider's prompt cache) keeps
its match across turns. A client that rewrites its history (a compaction, an edited message)
changes the hash: the note is gone, as on the engine, and the router adds its block again on the
next turn that needs it.

What differs from the engine, and is documented as such: the note is inside the user's message
text rather than before the end-of-turn token, so a model can quote it.

The hash reads `role`, `content`, `name`, `tool_calls` and `tool_call_id` of each message, as
canonical JSON; fields a client adds or drops between turns (cache-control marks, ids of its own)
do not break the replay. Standard library only.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math

KEYS = ("role", "content", "name", "tool_calls", "tool_call_id")
CHARS_PER_TOKEN = 3.5


def est_tokens(text: str) -> int:
    """A note's size in tokens, estimated (the backend's tokenizer is not ours). Used only for the
    replay budget and the journal's `hidden_tokens`, which say `estimated` beside it."""
    return int(math.ceil(len(text or "") / CHARS_PER_TOKEN))


def last_user(messages) -> int | None:
    """The index of the newest user message, the one a note of this turn goes on."""
    if not isinstance(messages, list):
        return None
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if isinstance(m, dict) and m.get("role") == "user":
            return i
    return None


def _canon(m) -> dict:
    if not isinstance(m, dict):
        return {"_": m}
    out = {k: m[k] for k in KEYS if k in m}
    c = out.get("content")
    if isinstance(c, list):
        # a list of parts: the text and the kind of each part, not the client's own marks
        out["content"] = [{k: p[k] for k in ("type", "text", "image_url") if k in p}
                          if isinstance(p, dict) else p for p in c]
    return out


def prefix_hash(messages, index: int) -> str | None:
    """sha256 of the messages `0..index` (inclusive), canonical; None past the end."""
    if not isinstance(messages, list) or index < 0 or index >= len(messages):
        return None
    h = hashlib.sha256()
    for m in messages[:index + 1]:
        h.update(json.dumps(_canon(m), sort_keys=True, ensure_ascii=False,
                            separators=(",", ":")).encode())
        h.update(b"\x1e")
    return h.hexdigest()


def hashes_at(messages, indices) -> dict:
    """index -> prefix_hash, one pass over the messages."""
    want = sorted({i for i in indices if isinstance(i, int) and i >= 0})
    out: dict = {}
    if not isinstance(messages, list) or not want:
        return out
    h = hashlib.sha256()
    j = 0
    for i, m in enumerate(messages):
        if j >= len(want):
            break
        h.update(json.dumps(_canon(m), sort_keys=True, ensure_ascii=False,
                            separators=(",", ":")).encode())
        h.update(b"\x1e")
        while j < len(want) and want[j] == i:
            out[i] = h.copy().hexdigest()
            j += 1
    return out


def append_text(message: dict, text: str) -> dict:
    """A copy of `message` with `text` at the end of its content: a string grows; a list of
    parts gets a text part; no content becomes the text without its leading blank lines."""
    m = dict(message)
    c = m.get("content")
    if isinstance(c, str):
        m["content"] = c + text
    elif isinstance(c, list):
        m["content"] = list(c) + [{"type": "text", "text": text}]
    else:
        m["content"] = text.lstrip("\n")
    return m


def splice(messages: list, notes) -> list:
    """The messages with every `(index, text)` note appended in order; the input is not changed.
    A note whose index is past the end is skipped."""
    out = list(messages)
    for idx, text in notes:
        if isinstance(idx, int) and 0 <= idx < len(out) and isinstance(out[idx], dict):
            out[idx] = append_text(out[idx], text)
    return out


def with_hidden_turn(messages: list, text: str, assistant: dict | None = None) -> list:
    """The conversation plus (the answer so far, when given) and a hidden user turn: E1's
    second request over HTTP."""
    out = copy.copy(list(messages))
    if assistant is not None:
        out.append(assistant)
    out.append({"role": "user", "content": text})
    return out
