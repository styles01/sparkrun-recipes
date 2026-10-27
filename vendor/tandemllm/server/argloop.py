"""The tool-argument loop guard: a tool call whose arguments repeat one unit.

On 2026-10-04 at 16:18 Qwen opened a `write` of a 3.5 KB game, wrote 3 KB of it, then the same
97-character row of a tile map 12,866 times: 65,536 tokens, 1.26 MB, 327 s, until the token
limit. The lookup drafter copied the loop (31 of 31 accepted in 2,007 of 2,209 blocks), the call
never closed, nothing was written, and the cut block went back into the next prompt as text. The
reasoning-loop guard (D1) watches the thinking only; nothing watched a call's arguments.

The guard reads the text of the open tool-call block (what follows `<tool_call>`), the way the
stream's `ToolCallBuffer` holds it, every `every` tokens once the block is `min_tokens` long. Two
rules, either fires:

    periodic   the block's last `span_tokens` tokens are one unit of at most `max_period`
               characters repeated, exactly, at least `min_repeats` times; a unit of at least
               `long_period` characters needs only `long_repeats` copies over `long_span_tokens`
               (a long unit repeated three times is a loop, not code). The check
               takes the tail's last `probe` characters, finds where they occurred before (the
               nearest is the smallest period that fits them), and compares the whole span against
               itself shifted by that period: one string compare, at C speed.
    size       the block is longer than `size_tokens` and `size_context` times the longest
               message the request carries (in tokens, estimated at `chars_per_token`), and its
               last `window_chars` characters compress `min_ratio` times or more (zlib): a call far
               bigger than anything the model read, that is mostly repetition.

Thresholds come from `tools/argloop_fp.py` over every write and edit in the opencode database
(the legit set) and the loops found there; the numbers are in docs/server.md, "The tool-argument
loop guard". The guard acts only where nothing of the call has reached the client (the server
holds the calls for the SP3 checks, `ToolCallBuffer.hold`); elsewhere it is logged.

A client that sends the conversation again after a stop (Open WebUI's tool loop, 2026-10-06
06:11) replays the stopped call's arguments: 6,400 tokens of the loop the model then copied, five
times byte for byte. `collapse_history` cuts such a call's repetition out of the history before
the prompt is built (stateless: an argument that does not parse, or a long string value, whose
tail is periodic), leaving its head and one marker; `trailing_stops` reads the markers after the
last user message, the retry streak the router (effort) and the monitor (the note, the give-up
after `GIVE_UP_IDENTICAL` identical stops or `GIVE_UP_ANY` in a row) act on.

No torch, no server state: text and counts in, a verdict out.
"""

from __future__ import annotations

import json
import re
import zlib
from bisect import bisect_right
from dataclasses import dataclass, field

DEFAULTS = {
    "every": 64,             # tokens between checks inside an open block
    "min_tokens": 1024,      # no check before the block is this long
    "span_tokens": 1024,     # periodic: the repeated stretch, in tokens
    "max_period": 4096,      # periodic: the longest unit, in characters
    "min_repeats": 6,        # periodic: the unit at least this many times in the stretch
    "long_period": 256,      # periodic: a unit this long or longer ...
    "long_repeats": 3,       # ... needs only this many copies ...
    "long_span_tokens": 384,  # ... over a stretch of at least this many tokens
    "probe": 256,            # periodic: characters of the tail looked up to find the period
    "size_tokens": 32768,    # size: never below this many tokens in the block
    "size_context": 3.0,     # size: and this many times the longest message the request carries
    "chars_per_token": 3.5,  # size: a message's tokens, estimated
    "window_chars": 16384,   # size: the window whose compression is measured
    "min_ratio": 16.0,       # size: zlib ratio of that window
}


def periodic_tail(text: str, max_period: int, probe: int = 256) -> tuple[int, int]:
    """`(period, span)`: the smallest period the tail's last `probe` characters repeat at, within
    `max_period`, and how many characters at the end of `text` are periodic with it (unit
    included). `(0, 0)` when the tail does not repeat."""
    n = len(text)
    if n < probe * 2:
        return 0, 0
    tail = text[n - probe:]
    j = text.rfind(tail, max(0, n - probe - max_period), n - 1)
    if j < 0:
        return 0, 0
    p = n - probe - j
    # the longest L with text[n-L-p : n-p] == text[n-L : n]; monotone in L, so a binary search
    lo, hi = probe, n - p
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if text[n - mid - p:n - p] == text[n - mid:]:
            lo = mid
        else:
            hi = mid - 1
    return p, lo + p


def ratio(text: str) -> float:
    """zlib's compression ratio of `text` (UTF-8, level 1: the guard's cost, not the best)."""
    raw = text.encode("utf-8", "surrogatepass")
    if not raw:
        return 1.0
    return len(raw) / max(1, len(zlib.compress(raw, 1)))


def longest_message_chars(messages) -> int:
    """The longest text a request's messages carry: a file read, a pasted log, the system."""
    best = 0
    for m in messages or ():
        if not isinstance(m, dict):
            continue
        c = m.get("content")
        if isinstance(c, str):
            best = max(best, len(c))
        elif isinstance(c, list):
            for part in c:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    best = max(best, len(part["text"]))
        for call in m.get("tool_calls") or ():
            fn = call.get("function") if isinstance(call, dict) else None
            if isinstance(fn, dict) and isinstance(fn.get("arguments"), str):
                best = max(best, len(fn["arguments"]))
    return best


@dataclass
class Verdict:
    rule: str                 # periodic | size
    tokens: int               # tokens in the block when it fired
    chars: int                # characters in the block
    period: int = 0           # periodic: the unit's characters
    repeats: float = 0.0      # periodic: span / period
    span_tokens: int = 0      # periodic: the repeated stretch
    piece: str = ""           # periodic: the unit's first characters, from the stretch's start
    ratio: float = 0.0        # size: the window's compression
    detail: dict = field(default_factory=dict)
    text: str = ""                         # the sentence the client got, once it acted

    def as_dict(self) -> dict:
        d = {"rule": self.rule, "tokens": self.tokens, "chars": self.chars}
        if self.rule == "periodic":
            d.update(period=self.period, repeats=round(self.repeats, 1),
                     span_tokens=self.span_tokens, piece=self.piece)
        else:
            d.update(ratio=round(self.ratio, 1))
        d.update(self.detail)
        return d


class ArgLoopGuard:
    """One request's guard. `feed(n_tokens, length, text)` after each token the stream decided:
    `length` is the open block's length in characters (None outside a block) and `text()` gives
    its text, called only at a check. Returns a `Verdict` once, when a rule fires; else None."""

    def __init__(self, cfg: dict | None = None, context_chars: int = 0):
        self.cfg = dict(DEFAULTS, **(cfg or {}))
        self.context_chars = int(context_chars)
        self.fired: Verdict | None = None
        self.checks = 0
        self._start: int | None = None   # the stream's token count when the block opened
        self._offs: list[int] = []       # the block's length in characters after each token
        self._next = 0

    def size_limit(self) -> int:
        c = self.cfg
        return int(max(c["size_tokens"],
                       c["size_context"] * self.context_chars / c["chars_per_token"]))

    def feed(self, n_tokens: int, length: int | None, text) -> Verdict | None:
        if self.fired is not None:
            return None
        if length is None:
            self._start, self._offs = None, []
            return None
        if self._start is None:
            self._start, self._offs = n_tokens - 1, []
            self._next = self.cfg["min_tokens"]
        self._offs.append(int(length))
        k = n_tokens - self._start
        if k < self._next:
            return None
        self._next = k + self.cfg["every"]
        return self.check(text(), k)

    def _tokens_from(self, char_pos: int) -> int:
        """Tokens of the block that end after `char_pos` (the stretch from there to the end)."""
        i = bisect_right(self._offs, char_pos)
        return len(self._offs) - i

    def check(self, text: str, k: int) -> Verdict | None:
        c = self.cfg
        self.checks += 1
        p, span = periodic_tail(text, c["max_period"], c["probe"])
        long_ = p >= c["long_period"]
        reps, need = ((c["long_repeats"], c["long_span_tokens"]) if long_
                      else (c["min_repeats"], c["span_tokens"]))
        if p and span >= reps * p:
            st = self._tokens_from(len(text) - span) if self._offs else k
            if st >= need:
                at = len(text) - span
                self.fired = Verdict("periodic", k, len(text), period=p, repeats=span / p,
                                     span_tokens=st, piece=text[at:at + PIECE_CHARS],
                                     detail={"tier": "long" if long_ else "short"})
                return self.fired
        if k >= self.size_limit():
            r = ratio(text[-c["window_chars"]:])
            if r >= c["min_ratio"]:
                self.fired = Verdict("size", k, len(text), ratio=r,
                                     detail={"limit": self.size_limit()})
                return self.fired
        return None


def stop_text(v: Verdict, name: str | None) -> str:
    """The visible sentence the client gets in place of the call."""
    call = f"`{name}` call" if name else "tool call"
    if v.rule == "periodic":
        why = (f"its arguments repeated the same {v.period}-character piece "
               f"{int(v.repeats)} times in a row")
    else:
        why = (f"its arguments grew past {v.tokens:,} tokens, far larger than anything in the "
               f"conversation, and were mostly repetition")
    return (f"[TandemLLM stopped the {call} after {v.tokens:,} tokens: {why}. Nothing was run "
            f"or written.]")


PIECE_CHARS = 60

NOTE = ("Runtime note: your last tool call was stopped and not run. Its arguments fell into a "
        "loop: {why}. Write the file again without the repetition. Generate repeated structure "
        "(map rows, tables, arrays) with code such as a loop or String.repeat instead of writing "
        "every copy, and split a large file into parts: write the first part, then add the rest "
        "with edits.")

# the retry's note: the history showed the call stopped already; the note names
# the tool and the piece, and after an identical stop says the same output came again
RETRY_NOTE = ("Runtime note: your last {calls} stopped and not run, because the arguments fell "
              "into a loop: {why}.{again} Do not write that piece again copy by copy. Generate "
              "repeated structure (map rows, pixel rows, tables, arrays) with code such as a loop "
              "or String.repeat, or write a small first part of the file now and add the rest "
              "with further calls.")

# the give-up: what the client gets in place of another identical attempt
FINAL = ("[TandemLLM stopped retrying: the last {n} attempts at the `{name}` call fell into the "
         "same loop (one {period}-character piece written over and over) and were stopped "
         "before anything ran. Nothing more is tried automatically. Ask again with a smaller "
         "step, for example: write the file in parts, or generate the repeated rows with a "
         "loop{think}.]")

GIVE_UP_IDENTICAL = 2     # identical stops in a row (same tool, same period) -> the give-up
GIVE_UP_ANY = 3           # any stops in a row of one tool -> the give-up


def _piece(text: str, escaped: bool = False) -> str:
    """A piece's first characters for a note, quoted (a newline as \\n, spaces kept: a row of
    spaces is the piece), never the whole unit. `escaped`: the text is a JSON string's inside
    (a replayed call's raw arguments), read back first where it can be."""
    t = str(text or "")
    if escaped:
        for cut in range(0, 3):
            try:
                t = json.loads('"' + t[:len(t) - cut] + '"')
                break
            except ValueError:
                continue
    return json.dumps(t[:PIECE_CHARS], ensure_ascii=False)


def note_text(v: dict, stops: list | None = None) -> str:
    """The hidden note for the session's next request, from a verdict's `as_dict()`; with the
    history's trailing stops (`trailing_stops`) the retry's note: the tool, the piece, and
    whether the last stops were the same loop again."""
    if not stops:
        if v.get("rule") == "periodic":
            why = (f"the same {v.get('period')}-character piece {int(v.get('repeats', 0))} times "
                   f"after {v.get('tokens', 0):,} tokens")
            if v.get("piece"):
                why += f", beginning {_piece(v['piece'])}"
        else:
            why = f"{v.get('tokens', 0):,} tokens of mostly repeated text"
        return NOTE.replace("{why}", why)
    last = stops[-1]
    name = last.get("name") or (v or {}).get("name") or "tool"
    n = len(stops)
    calls = (f"`{name}` call was" if n == 1 else f"{n} `{name}` calls were")
    piece = (v or {}).get("piece") or last.get("piece")
    why = f"one {last.get('period')}-character piece repeated over and over"
    if piece:
        why += f", beginning {_piece(piece)}"
    ident = identical(stops)
    again = (f" The last {ident} attempts produced the identical loop; change the approach, do "
             f"not retry the same call." if ident >= 2 else "")
    return RETRY_NOTE.replace("{calls}", calls).replace("{why}", why).replace("{again}", again)


def final_text(stops: list, thinking_off: bool = False) -> str:
    """The give-up sentence after `stops` (the history's trailing stops)."""
    last = stops[-1]
    think = ", or turn thinking on (this chat has it off)" if thinking_off else ""
    return (FINAL.replace("{n}", str(len(stops))).replace("{name}", last.get("name") or "tool")
            .replace("{period}", str(last.get("period"))).replace("{think}", think))


# ------------------------------------------------------------------ the replayed history
MARK = ("[TandemLLM stopped this `{name}` call: its arguments repeated one {period}-character "
        "piece {repeats} times, beginning {piece}; {cut:,} characters of the repetition are left "
        "out here. The call never ran.]")
MARK_RE = re.compile(r"\[TandemLLM stopped this `([^`]*)` call: its arguments repeated one "
                     r"(\d+)-character piece")
HISTORY_MIN_CHARS = 1000   # a replayed argument shorter than this is never touched
HISTORY_MIN_SPAN = 512     # the periodic stretch at its end, in characters, at least
HISTORY_REPEATS = 2.5      # and at least this many copies of the unit (a guard stop leaves ~3)
HISTORY_HEAD = 1500        # characters of the call's head kept before the marker
HISTORY_MAX_PERIOD = 8192  # JSON escaping makes a replayed unit longer than the guard's
INVALID_TOOLS = ("invalid",)   # opencode's stand-in for a call that did not parse


def _loop_cut(text: str) -> tuple[str, dict] | None:
    """`text` (a replayed argument, raw or one string value) with its periodic tail cut out and
    the marker in its place, and what was cut; None when it is not a stopped loop."""
    if not isinstance(text, str) or len(text) < HISTORY_MIN_CHARS or MARK_RE.search(text):
        return None
    p, span = periodic_tail(text, HISTORY_MAX_PERIOD, DEFAULTS["probe"])
    if not p or span < max(HISTORY_MIN_SPAN, HISTORY_REPEATS * p):
        return None
    at = len(text) - span
    head = text[:at]
    kept = head if len(head) <= HISTORY_HEAD else head[:HISTORY_HEAD] + " ..."
    info = {"period": p, "repeats": round(span / p, 1), "chars": len(text),
            "cut": len(text) - len(kept), "piece": text[at:at + PIECE_CHARS]}
    return kept, info


def _collapse_value(v, name: str, found: list, escaped: bool = False):
    """A call's arguments (a JSON string, a dict, or a string value inside them) with every
    stopped loop collapsed; `found` gets one entry a collapse."""
    if isinstance(v, dict):
        return {k: _collapse_value(x, name, found) for k, x in v.items()}
    if isinstance(v, list):
        return [_collapse_value(x, name, found) for x in v]
    if not isinstance(v, str):
        return v
    cut = _loop_cut(v)
    if cut is None:
        return v
    kept, info = cut
    found.append(dict(info, name=name))
    return kept + "\n" + MARK.format(name=name, period=info["period"],
                                     repeats=int(info["repeats"]),
                                     piece=_piece(info["piece"], escaped), cut=info["cut"])


def collapse_history(messages) -> tuple[list, list]:
    """(messages, collapsed): the assistant tool calls whose arguments hold a stopped loop --
    arguments that do not parse (a call the guard or the token limit cut while it streamed), or
    a string value inside the parsed arguments of opencode's `invalid` stand-in; a call whose
    arguments parse is a call that closed and is never touched -- with the loop cut down to its
    head and one marker. The
    messages are copied where they change; the client's list is never touched. `collapsed` has
    one entry a cut: the tool, the period, the characters before and after."""
    found: list = []
    out = []
    for m in messages or ():
        calls = m.get("tool_calls") if isinstance(m, dict) else None
        if not calls or m.get("role") != "assistant":
            out.append(m)
            continue
        new_calls, changed = [], False
        for c in calls:
            fn = c.get("function") if isinstance(c, dict) else None
            if not isinstance(fn, dict):
                new_calls.append(c)
                continue
            name = str(fn.get("name") or "tool")
            args = fn.get("arguments")
            n0 = len(found)
            if isinstance(args, str):
                try:
                    parsed = json.loads(args) if args.strip() else None
                except ValueError:
                    parsed = None
                if parsed is None:
                    new = _collapse_value(args, name, found, escaped=True)
                elif name in INVALID_TOOLS:
                    v = _collapse_value(parsed, name, found)
                    new = json.dumps(v, ensure_ascii=False) if len(found) > n0 else args
                else:
                    new = args
            elif name in INVALID_TOOLS:
                new = _collapse_value(args, name, found)
            else:
                new = args
            if len(found) > n0:
                c = dict(c, function=dict(fn, arguments=new))
                changed = True
            new_calls.append(c)
        out.append(dict(m, tool_calls=new_calls) if changed else m)
    return out, found


def _stops_in(v, name: str, acc: list) -> None:
    if isinstance(v, dict):
        for x in v.values():
            _stops_in(x, name, acc)
    elif isinstance(v, list):
        for x in v:
            _stops_in(x, name, acc)
    elif isinstance(v, str):
        m = MARK_RE.search(v)
        if m:
            i = v.find("beginning ", m.end())
            piece = ""
            if i >= 0:
                try:
                    piece = json.JSONDecoder().raw_decode(v, i + 10)[0]
                except ValueError:
                    piece = ""
            acc.append({"name": name, "period": int(m.group(2)), "piece": str(piece)})


def trailing_stops(messages) -> list[dict]:
    """The stopped calls (the markers `collapse_history` left) after the conversation's last
    user message, oldest first, ending at the newest call: a call of the history that was not
    stopped ends the streak. Each: the tool's name and the loop's period."""
    msgs = [m for m in messages or () if isinstance(m, dict)]
    last_user = max((i for i, m in enumerate(msgs) if m.get("role") == "user"), default=-1)
    stops: list = []
    for m in msgs[last_user + 1:]:
        if m.get("role") != "assistant":
            continue
        for c in m.get("tool_calls") or ():
            fn = c.get("function") if isinstance(c, dict) else None
            if not isinstance(fn, dict):
                continue
            acc: list = []
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)          # a parsed call's values (`invalid`)
                except ValueError:
                    pass                             # a cut call: the marker is in the text
            _stops_in(args, str(fn.get("name") or "tool"), acc)
            if acc:
                stops.append(acc[0])
            else:
                stops = []                         # a call that ran ends the streak
    return stops


def identical(stops: list) -> int:
    """How many of the trailing stops, from the newest back, are the same loop (tool, period)."""
    if not stops:
        return 0
    key = (stops[-1].get("name"), stops[-1].get("period"))
    n = 0
    for s in reversed(stops):
        if (s.get("name"), s.get("period")) != key:
            break
        n += 1
    return n


def give_up_due(stops: list) -> bool:
    """The history's trailing stops call for the give-up instead of another attempt."""
    if not stops:
        return False
    same_tool = 0
    for s in reversed(stops):
        if s.get("name") != stops[-1].get("name"):
            break
        same_tool += 1
    return identical(stops) >= GIVE_UP_IDENTICAL or same_tool >= GIVE_UP_ANY
