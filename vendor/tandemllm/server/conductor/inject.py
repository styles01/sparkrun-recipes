"""Where hidden text goes into a prompt, and how it stays there across turns.

The Qwen3.8 chat template renders `reasoning_effort` as the first system tokens (xhigh and low add
a sentence, medium adds none), and `enable_thinking: false` drops that sentence too, so changing
either per turn would change the head of a 100k-token prompt and cost a cold prefill. Everything the
router adds therefore goes at the other end: thinking off replaces the generation prompt's
`<think>\n` with the template's own closed block, and a note goes into
the last turn, right before its `<|im_end|>`, after the newest content and before the generation
prompt (`<|im_start|>assistant\\n<think>\\n`). The prefix up to that point is the one the engine
already holds, and `ThinkBudget.start` still sees the prompt's own tail.

The next turn's prompt does not contain the note (the client never saw it), so it would part from
the resident prefix at the note and the engine would re-read the whole previous answer. So a note
is REPLAYED: the session remembers `(pos, hash of the tokens before pos, the note's ids)`, with
`pos` counted in the client's prompt (without notes), and every later prompt whose first `pos`
tokens hash the same gets the same ids at the same place. The prefix then matches the previous
turn's KV through the answer, and the model sees one conversation that does not change under it.
After a compaction the hash no longer matches and the note is gone for good.

Token work only: lists of ints in, lists of ints out. The router decides what; this decides where.
"""

from __future__ import annotations

import hashlib
from array import array

IM_END = "<|im_end|>"
# the generation prompt with thinking on, and the template's own closed block for thinking off
THINK_OPEN = "<think>\n"
THINK_OFF = "<think>\n\n</think>\n\n"
# a template that opens the block with no newline and ignores enable_thinking (LFM2.5): its open
# tail, closed as the server closes it for a client's thinking off (server/app.py build_prompt)
THINK_OPEN_BARE = "<think>"
THINK_OFF_BARE = "<think></think>"


def special_id(tok, text: str) -> int | None:
    try:
        v = tok.convert_tokens_to_ids(text)
    except Exception:                                              # noqa: BLE001
        return None
    unk = getattr(tok, "unk_token_id", None)
    return v if isinstance(v, int) and v >= 0 and v != unk else None


def insertion_point(ids: list[int], im_end: int | None) -> int | None:
    """The index of the prompt's last `<|im_end|>`: the end of the last message. None when the
    prompt has none (a completion request, or a tokenizer without the token)."""
    if im_end is None:
        return None
    for i in range(len(ids) - 1, -1, -1):
        if ids[i] == im_end:
            return i
    return None


def content_row(at: int | None, notes=()) -> int | None:
    """Where the last message's last token sits in the served prompt: `at - 1` (`at` is the
    client's insertion point) shifted by every note the splice put before it; a note at `at`
    itself goes after that token. The probe snapshots read this row (server/probe.py): the same
    with or without the note and the thinking splice this turn adds."""
    if at is None or at < 1:
        return None
    return at - 1 + sum(len(n.ids) for n in notes or () if n.pos < at)


#: FX17: how many of the client's tokens right before a note's insertion point make its anchor
ANCHOR_TAIL = 32
#: the least of a message's own tokens that must match when the message is shorter than the tail
ANCHOR_MIN = 4


def message_anchor(ids: list[int], at: int, marks) -> tuple | None:
    """FX17: a note's place in the prompt that survives a change BEFORE it: the ordinal of the
    user message it closes (counted by the template's `<|im_start|>user` marks) and the last
    `ANCHOR_TAIL` client tokens before the insertion point (the user's own words). Open WebUI moves
    its retrieval block from an earlier user message to its last one on every new message, so the
    prefix hash of every earlier note broke and the router put its whole note in again (5.8k
    tokens a message). None without the marks."""
    if marks is None or at is None:
        return None
    im_s, user, _ = marks
    u = len(user)
    k = 0
    i = 0
    while True:
        try:
            i = ids.index(im_s, i)
        except ValueError:
            break
        if i >= at:
            break
        if list(ids[i + 1:i + 1 + u]) == user:
            k += 1
        i += 1
    return (k, tuple(int(t) for t in ids[max(0, at - ANCHOR_TAIL):at]))


def locate_anchor(ids: list[int], anchor, marks, im_end: int | None) -> int | None:
    """FX17: the insertion point in `ids` for a note with `anchor` (`message_anchor`): the end of
    the ordinal-th user message, when the client tokens before it are the anchor's; else None."""
    if anchor is None or marks is None or im_end is None:
        return None
    k, tail = anchor
    im_s, user, _ = marks
    u = len(user)
    seen, i = 0, 0
    while True:
        try:
            i = ids.index(im_s, i)
        except ValueError:
            return None
        if list(ids[i + 1:i + 1 + u]) == user:
            seen += 1
            if seen == k:
                try:
                    p = ids.index(im_end, i + 1 + u)
                except ValueError:
                    return None
                # the message's end must match the anchor's tail: all of it, or, for a message
                # shorter than the tail (the client's block that left it was part of the tail),
                # all of the message's own content, at least ANCHOR_MIN tokens of it
                content = p - (i + 1 + u)
                m = 0
                while (m < len(tail) and m < p
                       and int(ids[p - 1 - m]) == int(tail[len(tail) - 1 - m])):
                    m += 1
                need = min(len(tail), max(ANCHOR_MIN, content))
                return p if m >= need and m >= 1 else None
        i += 1


def hashes_at(ids: list[int], positions) -> dict[int, str]:
    """sha256 of `ids[:p]` for every `p`, in one pass over the prompt."""
    buf = array("q", ids).tobytes()
    width = array("q").itemsize
    h = hashlib.sha256()
    last, out = 0, {}
    for p in sorted(set(int(x) for x in positions)):
        if p < 0 or p > len(ids):
            continue
        h.update(buf[last * width:p * width])
        last = p
        out[p] = h.copy().hexdigest()[:32]
    return out


def splice(ids: list[int], notes: list[tuple[int, list[int]]]) -> tuple[list[int], list[int]]:
    """`ids` with each `(pos, note_ids)` inserted before original position `pos`; also the final
    start index of each note, in the order given. Notes at one position keep their order."""
    order = sorted(range(len(notes)), key=lambda i: (notes[i][0], i))
    out: list[int] = []
    starts = [0] * len(notes)
    last = 0
    for i in order:
        pos, nid = notes[i]
        out.extend(ids[last:pos])
        starts[i] = len(out)
        out.extend(nid)
        last = pos
    out.extend(ids[last:])
    return out, starts
