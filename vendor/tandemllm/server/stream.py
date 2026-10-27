"""Turning a growing list of token ids into text a client can be handed one piece at a time.

Three things live here rather than in `server/app.py`, and they are here together because all
three are the same mistake: **a token boundary is not a text boundary.** The engine decides tokens
and the protocol carries characters, and everywhere the server assumed those were the same thing it
produced something the reader could see was wrong.

1. **A character can span tokens.** The tokeniser is byte-level, so `🤦🏼` arrives as five tokens
   and the decoder shows a replacement character at three of the five prefixes:

       '🤦'  ->  '🤦�'  ->  '🤦�'  ->  '🤦🏼'

   A stream that emits `decoded[len(emitted):]` at every step sends the `�` the moment it
   appears and then never recovers: the next prefix is the same length, so the piece is empty, and
   the one after that is SHORTER in characters than what has already gone out. The reader is left
   with a permanent `�` where the modifier should be. `Detokenizer` holds the trailing
   incomplete run back until it resolves.

2. **The reasoning block opens in the prompt, not in the output.** The chat template's generation
   prompt ends `<|im_start|>assistant\n<think>\n`, so the model's first generated token is already
   inside the reasoning block and the only tag it ever writes is the closing one. A client that
   folds reasoning on `<think>...</think>` sees a stray `</think>` and folds nothing.
   `opens_think` says when that has happened and the server re-emits the opening tag.

3. **Not every client reads the tags.** `Reasoning` splits the stream at `</think>` so the same
   generation can be delivered as tagged text, as OpenAI-style `reasoning_content` deltas, or as
   both, without the generator knowing which.

Nothing here imports torch: it is string handling, and it is tested as string handling.
"""

from __future__ import annotations

OPEN_THINK = "<think>"
CLOSE_THINK = "</think>"
REPLACEMENT = "�"

#: `reasoning_format` values. `tags` is the default and the reason is in `Reasoning`.
FORMATS = ("tags", "reasoning_content", "both")


def opens_think(rendered_prompt: str) -> bool:
    """Does this rendered prompt leave the model inside an open reasoning block?

    The last tag in the text decides it. `enable_thinking=True` ends the prompt with `<think>\\n`
    and this is true; `enable_thinking=False` ends it with `<think>\\n\\n</think>\\n\\n`, where the
    block is opened and closed inside the prompt, and this is false.
    """
    return rendered_prompt.rfind(OPEN_THINK) > rendered_prompt.rfind(CLOSE_THINK)


def _stable(text: str) -> str:
    """`text` with a trailing run of replacement characters removed.

    Only the TRAILING run, and only because of what produced it: a replacement character in the
    middle of a decoded prefix is a byte sequence that is already known to be broken and will not
    become anything else, while one at the end is a character whose remaining bytes are in the next
    token. Holding back the whole string on account of a bad byte in the middle would stall the
    stream for ever.
    """
    i = len(text)
    while i and text[i - 1] == REPLACEMENT:
        i -= 1
    return text[:i]


class Detokenizer:
    """Incremental text from a decoder that only knows how to decode a whole prefix.

    `decode(ids) -> str` is what the HF tokenisers support. What this adds is the hold-back, and
    the invariant it keeps: **every character this object returns is final.** A piece is never
    retracted, never re-sent, and never contains a character whose bytes have not all arrived.

    `push` is handed every id decided so far, but it decodes only the ids since the last
    CLEAN boundary -- the last call whose text did not end in a replacement character. Until then
    it decoded the whole list on every token: 0.8 ms a call at 8k output tokens, 3.2 ms at 32k,
    four times a round on the round's critical path. The byte-level decoder is a lossy UTF-8
    decode of the ids' bytes, and UTF-8 restarts cleanly after a complete character, so the
    whole text is the text up to the boundary plus the window's text, and every piece is the one
    the whole-list decode gave (tests/test_stream.py checks both against the real tokeniser).
    A boundary is never placed after a trailing replacement character, a literal one included,
    which is also what keeps the text before it free of a run `_stable` would have held back.
    """

    def __init__(self, decode):
        self.decode = decode
        self.start = 0     # ids before this index are decoded and sent, and end on a character
        self.sent = 0      # characters of the window's text already returned

    def push(self, ids: list[int]) -> str:
        """The text decided since the last call. May be empty, which is normal and not an error."""
        window = self.decode(ids[self.start:])
        text = _stable(window)
        piece = text[self.sent:]
        if len(text) == len(window):
            self.start, self.sent = len(ids), 0
        else:
            self.sent = len(text)
        return piece

    def flush(self, ids: list[int]) -> str:
        """The last piece, at end of stream, replacement characters and all.

        At the end there is no next token to complete anything, so whatever is still held back is
        genuinely broken input rather than an unfinished character, and dropping it silently would
        lose real bytes. It goes out as it is.
        """
        window = self.decode(ids[self.start:])
        piece = window[self.sent:]
        self.start, self.sent = len(ids), 0
        return piece


class Reasoning:
    """Where each piece of the stream belongs: the reasoning field, the content field, or both.

    The three formats exist because clients disagree and the disagreement is not settleable:

      * `tags` -- everything goes to `content`, reasoning included, wrapped in
        `<think>...</think>`. This is what the model writes and what Open WebUI folds on, so it is
        the DEFAULT. It is also the only format in which "the first delta starts with `<think>`" is
        a true statement, which is the bug this was written for.
      * `reasoning_content` -- the reasoning goes to `delta.reasoning_content` and the answer to
        `delta.content`, and neither tag appears in either. This is the OpenAI/DeepSeek shape.
      * `both` -- the reasoning is delivered twice: `content` is exactly the `tags` text, tags and
        all, and `reasoning_content` the reasoning alone. Correct for a client that reads one and
        ignores the other, and visibly duplicated in one that renders both, which is why it is not
        the default. (Until its `content` opened the block and never closed it.)

    The split point is the first `</think>`. A piece can straddle it and the tag itself can arrive
    in several pieces, so in the two formats that have to recognise the tag, a tail that could
    still grow into one is held back until it either becomes the tag or cannot.
    """

    def __init__(self, fmt: str = "tags", *, in_think: bool = True, boundary: str = ""):
        if fmt not in FORMATS:
            raise ValueError(f"reasoning_format must be one of {FORMATS}, not {fmt!r}")
        self.fmt = fmt
        self.in_think = bool(in_think)
        # SRV-193: `reasoning_content` for Open WebUI marks where the reasoning ends with a content
        # piece of blank lines (the template's own "\n\n"): Open WebUI closes a round's reasoning
        # item, and times it, at the first content; a round that goes from its thinking straight
        # to a tool call had none and showed "Thought for 0 seconds". A blank message is not shown.
        self.boundary = boundary if fmt == "reasoning_content" else ""
        self.pending = ""
        # `tags` sends every character at once and still has to know where the block ends: the
        # text inside it is labelled `tagged` -- the content FIELD, but not the answer -- so the
        # tool-call buffer only ever reads the answer. `both`'s copy of the block in
        # `content` is labelled the same way. A closing tag split across pieces
        # is found through the last few characters, without holding any of them back.
        self._tail = ""
        # The template writes "</think>\n\n" and those blank lines belong to the tag rather than
        # to the answer. They can arrive in the same piece as the tag or in the next one, so this
        # stays set until the first real character of the answer turns up.
        self._strip_lead = False

    @property
    def splits(self) -> bool:
        """Whether this format has to find the closing tag at all."""
        return self.fmt != "tags"

    def push(self, piece: str) -> list[tuple[str, str]]:
        """`piece` as a list of `(field, text)`, in order. `field` is "content", "reasoning", or
        "tagged": the content field, carrying the reasoning block in `tags` and `both` format."""
        if not piece:
            return []
        if not self.splits and self.in_think:
            return self._tagged(piece)
        if not self.splits or not self.in_think:
            if self._strip_lead:
                piece = piece.lstrip("\n")
                if not piece:
                    return []
                self._strip_lead = False
            return [("content", piece)]
        buf = self.pending + piece
        self.pending = ""
        out: list[tuple[str, str]] = []
        idx = buf.find(CLOSE_THINK)
        if idx >= 0:
            head, tail = buf[:idx], buf[idx + len(CLOSE_THINK):]
            self.in_think = False
            if self.fmt == "both":
                # `content` is the `tags` text: the block closes where the model closed it, and
                # the template's blank lines stay in it, as they do in `tags`.
                if head:
                    out.append(("reasoning", head))
                out.append(("tagged", head + CLOSE_THINK))
                if tail:
                    out.append(("content", tail))
                return out
            if head:
                out.extend(self._reasoning(head))
            if self.boundary:
                out.append(("content", self.boundary))
            self._strip_lead = True
            tail = tail.lstrip("\n")
            if tail:
                self._strip_lead = False
                out.append(("content", tail))
            return out
        hold = _tag_prefix_len(buf)
        if hold:
            buf, self.pending = buf[:len(buf) - hold], buf[len(buf) - hold:]
        if buf:
            out.extend(self._reasoning(buf))
        return out

    def _tagged(self, piece: str) -> list[tuple[str, str]]:
        """`tags` inside the block: everything goes out now, the block's part labelled `tagged`."""
        seen = self._tail + piece
        idx = seen.find(CLOSE_THINK)
        if idx < 0:
            self._tail = seen[-(len(CLOSE_THINK) - 1):]
            return [("tagged", piece)]
        cut = idx + len(CLOSE_THINK) - len(self._tail)      # the tag ends inside this piece
        self.in_think, self._tail = False, ""
        out = [("tagged", piece[:cut])] if cut > 0 else []
        if piece[cut:]:
            out.append(("content", piece[cut:]))
        return out

    def _reasoning(self, text: str) -> list[tuple[str, str]]:
        if self.fmt == "both":
            return [("reasoning", text), ("tagged", text)]
        return [("reasoning", text)]

    def state(self, line: str) -> list[tuple[str, str]]:
        """A TandemLLM state line (server/conductor/states.py, already framed) where the reasoning
        goes now: inside the open block (`tagged` in `tags`, `reasoning` in `reasoning_content`,
        both in `both`), and in the reasoning field only once the block has closed or when the
        prompt opened none (`tags` has no place for it then). Nothing held back moves: the frame
        is exact wherever it lands between two pieces."""
        if self.in_think:
            if self.fmt == "tags":
                return [("tagged", line)]
            return self._reasoning(line)
        return [] if self.fmt == "tags" else [("reasoning", line)]

    def finish(self) -> list[tuple[str, str]]:
        """Whatever is still held back when the generation ends: it never became a tag."""
        if not self.pending:
            return []
        rest, self.pending = self.pending, ""
        return self._reasoning(rest) if self.in_think else [("content", rest)]


class StopStrings:
    """A request's `stop` strings, applied to the stream as the non-streamed answer applies them to
    the whole text: cut at the earliest match, the stop string itself never sent.

    Only the piece that completed a match used to be cut, and everything before it had gone out:
    `stop: ["Stop"]` streamed S, t, o. So a tail that could still grow into a stop string is held
    back -- the shortest one that can, from the earliest position that can -- until it becomes one
    or cannot. A complete match is not the end yet while a longer stop string that started earlier
    is still arriving ("abcd" against ["abcd", "bc"] cuts at a, not at b), because the answer is
    the earliest match in the text and not the first one to complete.

    Nothing before the held tail can ever start a match, so the tail is all that is kept.
    """

    def __init__(self, stops: list[str]):
        self.stops = [s for s in stops if s]
        self.longest = max((len(s) for s in self.stops), default=0)
        self.pending = ""
        self.hit = False

    def push(self, piece: str) -> str:
        """The text that is now safe to send. After a match, `hit` is set, this is the text before
        the stop string, and every later call returns nothing."""
        if not self.stops:
            return piece
        if self.hit:
            return ""
        buf, self.pending = self.pending + piece, ""
        cut = self._match(buf)
        wait = next((p for p in range(max(0, len(buf) - self.longest + 1), len(buf))
                     if any(len(s) > len(buf) - p and s.startswith(buf[p:]) for s in self.stops)),
                    None)
        if cut is not None and (wait is None or wait >= cut):
            self.hit = True
            return buf[:cut]
        keep = len(buf) if wait is None else wait
        self.pending = buf[keep:]
        return buf[:keep]

    def finish(self) -> str:
        """The held tail at the end of the generation, cut at a match that was waiting on a longer
        stop string that never completed."""
        buf, self.pending = self.pending, ""
        if self.hit:
            return ""
        cut = self._match(buf)
        if cut is not None:
            self.hit = True
            return buf[:cut]
        return buf

    def _match(self, text: str) -> int | None:
        hits = [i for i in (text.find(s) for s in self.stops) if i >= 0]
        return min(hits) if hits else None


class AnswerStops:
    """SRV-245 (UX2, 2026-10-07): a request's `stop` strings apply to the ANSWER, never to the
    reasoning. They were matched on the raw text before it was split, so `stop: ["\\n\\n"]` ended a
    response at the first blank line of the thinking: 55 tokens of reasoning, no answer.

    Fed the `(field, text)` pieces `Reasoning.push` returns: only `content` goes through the stop
    strings; `reasoning` and `tagged` (the block in `tags` and `both`) pass as they are. The blank
    lines the template writes after `</think>` (and any an answer starts with) are not matched
    either: they are the tag's, and a stop on a blank line would cut every answer before its
    first word. After a match, `hit` is set and nothing more passes."""

    def __init__(self, stops: list[str]):
        self.stopper = StopStrings(stops)
        self.lead = True                      # the answer has not written a character yet

    @property
    def hit(self) -> bool:
        return self.stopper.hit

    def push(self, parts: list[tuple[str, str]]) -> list[tuple[str, str]]:
        if not self.stopper.stops:
            return parts
        out: list[tuple[str, str]] = []
        for field, text in parts:
            if self.stopper.hit:
                break
            if field != "content":
                out.append((field, text))
                continue
            if self.lead:
                body = text.lstrip("\n")
                if len(body) < len(text):
                    out.append(("content", text[:len(text) - len(body)]))
                if not body:
                    continue
                self.lead, text = False, body
            sent = self.stopper.push(text)
            if sent:
                out.append(("content", sent))
        return out

    def finish(self) -> list[tuple[str, str]]:
        """The held tail of the answer at the end of the generation."""
        tail = self.stopper.finish() if self.stopper.stops else ""
        return [("content", tail)] if tail else []


def answer_stops(text: str, stops: list[str], in_think: bool) -> tuple[str, bool]:
    """SRV-245: a non-streamed generation cut at the first stop string in its answer: the text
    through the first `</think>` (when the prompt opened the block) is never matched, nor the
    blank lines that start the answer. `(text, cut)`."""
    hits = [s for s in stops if s]
    if not hits:
        return text, False
    head = ""
    if in_think:
        i = text.find(CLOSE_THINK)
        if i < 0:
            return text, False                # still thinking: there is no answer to cut
        head, text = text[:i + len(CLOSE_THINK)], text[i + len(CLOSE_THINK):]
    body = text.lstrip("\n")
    lead = text[:len(text) - len(body)]
    found = [j for j in (body.find(s) for s in hits) if j >= 0]
    if not found:
        return head + text, False
    return head + lead + body[:min(found)], True


def _tag_prefix_len(text: str) -> int:
    """How many trailing characters of `text` could still grow into `</think>`."""
    n = min(len(text), len(CLOSE_THINK) - 1)
    for k in range(n, 0, -1):
        if CLOSE_THINK.startswith(text[-k:]):
            return k
    return 0


def split_full(text: str, fmt: str = "tags", *, in_think: bool = True) -> tuple[str, str | None]:
    """The whole generation as `(content, reasoning_content)`, for a non-streamed answer.

    The same three formats as `Reasoning`, and the same reason the opening tag is put back: the
    model never wrote it because the prompt already had.
    """
    if fmt not in FORMATS:
        raise ValueError(f"reasoning_format must be one of {FORMATS}, not {fmt!r}")
    if not in_think:
        return text, None
    idx = text.find(CLOSE_THINK)
    if idx >= 0:
        reasoning, answer = text[:idx], text[idx + len(CLOSE_THINK):].lstrip("\n")
    else:
        # The generation ran out of room inside the reasoning block. There is no answer yet, and
        # saying so by leaving `content` empty is more honest than promoting the reasoning to one.
        reasoning, answer = text, ""
    if fmt == "tags":
        return OPEN_THINK + "\n" + text, None
    if fmt == "both":
        return OPEN_THINK + "\n" + text, reasoning
    return answer, reasoning


class StrayClose:
    """SRV-187 (the UT2 self-test, 2026-10-06): a `</think>` the model writes in its ANSWER, after
    the reasoning block closed, reaches the client as text. Open WebUI's tag parser closes a block
    at the first `</think>` and puts the rest of the text into a message, so a second one shows as
    the literal line `</think>` between two tool rows (the Mario chat, minute 38). The answer is
    filtered here, in `tags` and `reasoning_content` alike: a `</think>` alone on its line (spaces
    around it allowed) is dropped with its newline. Everything else passes as it came, the tag in
    a sentence, in inline code, inside a fenced code block, a `<tool_call>` or a `<tandem_memory>`
    block included, so a program that prints the tag keeps it.

    Streaming: only the start of a line that can still become the tag is held back (at most its
    eight characters and the spaces before them); the first character that cannot belongs to it
    sends the held text on. `dropped` counts the tags taken out (the request's journal row)."""

    TAG = "</think>"
    OPENS = ("<tool_call>", "<tandem_memory")
    CLOSES = ("</tool_call>", "</tandem_memory>")

    def __init__(self):
        self.hold: str | None = ""          # "" at a line start; None inside a line
        self.line = ""                       # the current line as sent (fence and block marks)
        self.fence = False
        self.depth = 0
        self.opened = 0                      # `<think>` lines the answer opened itself
        self.dropped = 0

    def _line_done(self, line: str) -> None:
        if line.strip(" \t") == "<think>" and not self.fence and self.depth == 0:
            self.opened += 1                 # its `</think>` is a pair, not a stray
        if line.lstrip().startswith("```"):
            self.fence = not self.fence
        for o in self.OPENS:
            self.depth += line.count(o)
        for c in self.CLOSES:
            self.depth = max(0, self.depth - line.count(c))

    def push(self, piece: str) -> str:
        out = []
        for ch in piece:
            if self.hold is not None and not self.fence and self.depth == 0:
                h = self.hold + ch
                core = (self.hold if ch == "\n" else h).strip(" \t")
                if ch == "\n" and core == self.TAG and self.opened:
                    self.opened -= 1             # it closes a block the answer opened: keep
                    out.append(h)
                    self.hold, self.line = "", ""
                    continue
                if ch == "\n" and core == self.TAG:
                    self.dropped += 1            # the bare tag and its newline go
                    self.hold, self.line = "", ""
                    continue
                if ch != "\n" and self.TAG.startswith(core):
                    self.hold = h                # it may still become the tag
                    continue
                out.append(h)                    # it cannot: send what was held
                self.line += h
                self.hold = None
                if ch == "\n":
                    self._line_done(self.line)
                    self.line, self.hold = "", ""
                continue
            out.append(ch)
            if ch == "\n":
                self._line_done(self.line)
                self.line, self.hold = "", ""
            else:
                self.line += ch
                if self.hold is not None:
                    self.hold = None
        return "".join(out)

    def finish(self) -> str:
        """The end of the answer: a held bare tag goes, anything else is sent."""
        h, self.hold = self.hold or "", None
        if h.strip(" \t") == self.TAG and not self.opened:
            self.dropped += 1
            return ""
        return h


def drop_stray_close(text: str, fmt: str = "reasoning_content", in_think: bool = False
                     ) -> tuple[str, int]:
    """A non-streamed answer's content through `StrayClose`: (text, tags dropped). In `tags` and
    `both` the content starts with the reasoning block when the prompt opened one; the block's
    own closing tag is the model's split point and stays, the answer after it is filtered."""
    head, rest = "", text
    if fmt in ("tags", "both") and in_think:
        i = text.find(CLOSE_THINK)
        if i < 0:
            return text, 0                   # the block never closed: there is no answer yet
        head, rest = text[:i + len(CLOSE_THINK)], text[i + len(CLOSE_THINK):]
    f = StrayClose()
    out = f.push(rest) + f.finish()
    return head + out, f.dropped
