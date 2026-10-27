"""Qwen decides with the conversation it already holds (the in-context stage, 2026-10-02).

The digest stages (`classify.py`) read a short text cut out of the user's message: Laya was sure
on about 15 % of follow-ups and 22 % of long prompts, and Qwen's own stage then read 2,000
characters of a conversation it had in full a turn ago. This stage asks Qwen inside that
conversation. Its prompt is the turn's prompt as the client sent it, with the session's earlier
notes replayed where they sit (the prompt the resident prefix holds), cut after the last message's
`<|im_end|>`, plus one hidden user turn with the question and the template's closed think block:

    ...the conversation...<|im_end|>\\n<|im_start|>user\\n{question}<|im_end|>\\n
    <|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n

Each class is a letter, and every letter is one token, so the logits of the prefill's last row are
the whole answer: their softmax over the letters is the constrained distribution, the label its
argmax, the confidence its probability. Nothing is decoded and nothing is written past the prompt.

The cache. The stage runs under the engine lock right before the turn's own prefill, on the same
`cache.prefill` the turn uses; the state store's prefix snapshots are taken only at lines up to
the parting point (`checkpoint_upto`), whose tokens are the turn's, and no session snapshot. It
extends the resident prompt: it resumes at the newest anchor the conversation shares and prefills
the new messages and the question, taking anchors on the 1,024-token grid as any prefill does. The
real turn then shares every token of it up to the insertion point (the last message's
`<|im_end|>`, where the turn's note goes) and resumes at the last grid line at or below it, so it
re-reads at most a grid chunk and its tail instead of everything since the previous prompt. The
rows above the parting point are overwritten by the turn's own prefill, and its state is a grid
prefill's, as exact as any anchor's (`tests/test_context_stage.py` checks the logits and the state
against the turn without the stage, bit for bit, on the random model).

Token work only; the engine side is the server's runner (`server/app.py`).
"""

from __future__ import annotations

import math
import string

from server.conductor import inject

LETTERS = string.ascii_uppercase
# the hidden turn after the conversation, as the template writes a user turn and the generation
# prompt with thinking off; the conversation's own `<|im_end|>` comes before it
TURN = "\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n" + inject.THINK_OFF


class ContextError(ValueError):
    """The stage cannot be asked on this prompt; `why` goes into the journal's stage error."""

    def __init__(self, why: str):
        super().__init__(why)
        self.why = why


def listing(options: dict[str, str]) -> tuple[str, list[str]]:
    """`A: id - description` lines and the letter of each option, in the options' order."""
    ids = list(options)
    if len(ids) > len(LETTERS):
        raise ContextError("too_many_options")
    lines = [f"{LETTERS[i]}: {k} - {options[k]}" for i, k in enumerate(ids)]
    return "\n".join(lines), [LETTERS[i] for i in range(len(ids))]


def letter_ids(tok, letters: list[str]) -> list[int]:
    """The token of each letter; every one must be a single token, all different."""
    out = []
    for c in letters:
        enc = tok(c, add_special_tokens=False).input_ids
        enc = enc.tolist() if hasattr(enc, "tolist") else list(enc)
        if enc and isinstance(enc[0], list):
            enc = enc[0]
        if len(enc) != 1:
            raise ContextError("letter_not_one_token")
        out.append(int(enc[0]))
    if len(set(out)) != len(out):
        raise ContextError("letters_share_a_token")
    return out


def prompt(ids: list[int], tok, question: str) -> tuple[list[int], int]:
    """The stage's prompt and the insertion point it parts from the turn's prompt at.

    `ids`: the turn's prompt as it will be prefilled up to its last message, i.e. the client's
    prompt with the session's earlier notes replayed (`Plan.replayed_ids`). The stage keeps it
    through the last `<|im_end|>` and adds the hidden turn."""
    im_end = inject.special_id(tok, inject.IM_END)
    at = inject.insertion_point(ids, im_end)
    if at is None:
        raise ContextError("no_insertion_point")
    tail = tok(TURN.replace("{question}", question), add_special_tokens=False).input_ids
    tail = tail.tolist() if hasattr(tail, "tolist") else list(tail)
    if tail and isinstance(tail[0], list):
        tail = tail[0]
    return list(ids[:at + 1]) + [int(t) for t in tail], at


def answer(logprobs: list[float], options: list[str]) -> tuple[str, float, dict[str, float]]:
    """The label, its probability and every option's, from the letters' log-probabilities (the
    softmax over the letters alone: the distribution a `choice` grammar leaves)."""
    if len(logprobs) != len(options) or not logprobs:
        raise ContextError("bad_logprobs")
    top = max(logprobs)
    ex = [math.exp(x - top) for x in logprobs]
    z = sum(ex)
    probs = {k: e / z for k, e in zip(options, ex)}
    label = max(options, key=lambda k: probs[k])
    return label, probs[label], probs
