"""TandemLLM's own steps, shown to the user (`ui.states`, 2026-10-02).

A client shows Qwen's reasoning and its answer. What the conductor does around them -- the class
it picked, a skill it loaded, a cap that closed the reasoning, a tool call it repaired -- was
visible only on the dashboard. With a profile's `ui.states: on` each step is one short line in the
stream, at the moment it happens. Where (`ui.states_channel`, `channel()`): `reasoning`, as every
profile stored before S1 keeps it, puts it in the request's `reasoning_format` -- a
`reasoning_content` delta, or inside the `<think>` block of `tags` (a step after the block closed
has no place in `tags` and is left out there); `content` (S1, the seed profiles' value) sends it as
a content delta of its own, outside the thinking (server/app.py; in `tags` a line said inside the
open block goes out right after `</think>`); `inline` (X1) puts each line in the reasoning as the
title of a reasoning part of its own, in time order, for opencode (INLINE below). The texts are
templates (`TEXTS`), no model and no latency; every line keeps a `summary` field for a clerk-written
one later (not built).

PROGRESS (S1). Three kinds carry the numbers of a wait (`PROGRESS`, `ui.progress`, `progress()`):
the queue (`waiting: opencode's title request runs first, position 1`), a long prefill (`reading
35,189 tokens at ~750 tok/s, about 47 s left`) and a long thinking (`thinking 4 min, 9,800
tokens`). They are always the template, never the narrator's -- but with `ui.narrator: thoughts`
(N2) the long thinking's line is the narrator's summary of the thought so far ("Thinking 4m 10s:
...") and the template only when it could not write one in time; a stretch of thinking that ended
gets its own line (`thought`, "Thought 1m 43s: ...", server/conductor/narrator.py Thoughts).

THE MARKER. Each line goes out framed as

    "\\n› <text>\\n\\n"

`›` (U+203A) and a space open the line, a blank line closes it: one paragraph of its own in a
client that renders markdown (opencode), one plain line in one that does not (Hermes), and the
same in both formats. The frame is self-delimiting, so `strip_text` takes it out exactly wherever
it sits: at the head or the end of the reasoning, between two parts a client joined, or between
two characters of Qwen's own text. Only a framed line whose text is one of the templates is
taken, so a line of Qwen's own that starts with `›` stays. A narrated line is free text: it closes
with a WORD JOINER (U+2060) before the blank line (narrator.frame), and the strip takes a framed
line of any text that ends in it.

INLINE (X1, 2026-10-03). A user report's screenshot of 09:23: with `content`, opencode 1.18.34 shows every
line of a turn in one text block at its top, above the "Thought" blocks they were said between.
How opencode builds its parts (its bundled @ai-sdk/openai-compatible, chunk-9h4pcghr.js, and
SessionProcessor): the provider opens ONE text part per response (`text-start`, id `txt-0`) at the
first content delta and never ends it before the stream does, so every later content delta lands
in that first part; a reasoning delta after a content delta or a `tool_calls` delta opens a NEW
reasoning part (`reasoning-start` again, the processor creates a part with a fresh ascending id),
and parts render in the order they were created. The TUI titles a reasoning part with its first
line when that line is `**title**` followed by a blank line (`+ Thought: <title> · 1m 31s`
collapsed, `Thinking: <title>` while it streams). So `inline`, on a `reasoning_content` stream,
sends each line as a `{"tool_calls": []}` delta (the provider ends the open reasoning part on it
and starts no tool call: the list is empty) and then a reasoning delta `\\n**› <text>**\\n\\n`
(`inline()`): a new part, titled by the line, at the moment it happens; Qwen's reasoning goes on
in that part. No content delta is sent, so the answer's text part opens where the answer starts.
A line said after the answer began opens a part after the text part. `tags` and `both` have no
reasoning parts and send `content`'s deltas (app.py makes the switch).

QWEN NEVER SEES THEM. opencode's chat-completions provider sends every earlier assistant message's
reasoning back as `reasoning_content` (its parts joined with nothing between them), and the Hermes
agent keeps the reasoning on the message it sends back; the Qwen3.8 template renders an assistant
message's `reasoning_content` (trimmed) into the prompt. So the server strips every framed line from
the `reasoning_content` and the string or list content of the assistant messages (where the content
channel's lines come back) before anything reads them (`strip_messages`, in the handler before the
router and before the prompt is rendered): the prompt, and with it the resident prefix and the
notes' positions, are the ones the request would have without the feature. A client that trims the
reasoning loses the frame's outer newlines, so a line at the very start or end is also taken
without them; whatever whitespace is left at an end the template trims. The lines are not tokens:
they are never in `usage`, the suffix store or the response cache key; the transcript keeps them as
a block of their own and the journal row as `extra.states`.
"""

from __future__ import annotations

import re
import time

MARKER = "› "
# SRV-185: the lines about the user's own notes (written, proposed, forgotten). They go into the
# content in every channel but `inline` (whose reasoning parts opencode shows): a client that shows
# no reasoning, or `tags` after its block closed, would drop them, and the user must see a write
MEMORY_KINDS = ("remembered", "proposed", "forgot")
# one line's text: no newline, at most this long
MAX_CHARS = 200
# what each kind of step says; `{...}` are the step's facts
TEXTS = {
    "route": "deciding: {cls}, {effort}",
    "compaction": "compaction seen",
    "package": "retrieving: {pkind} {name}",
    "memory": "reading memory: {n} notes",
    # 2026-10-05: the model's memory block was written (server/conductor/memory.py)
    "remembered": "remembered: {title}",
    "forgot": "forgot: {title}",
    "proposed": "proposed a change to {title}",
    "steer": "steering: {label}",
    "cap": "thinking closed at the {cap} cap",
    "backstop": "thinking closed at the {cap} backstop",
    "loop": "loop detected, steering",
    "close": "closing reasoning",
    "continue": "empty answer, continuing",
    # SRV-199: E1's lift, under the `continue` switch: the calls were written inside the thinking
    "lift": "tool call written in the thinking, sending it",
    "check": "checking the tool call",
    "checks": "checking {n} tool calls",
    "repair": "repairing the tool call",
    "regenerate": "asking for the tool call again",
    "give_up": "giving up",
    # S1: progress, the numbers of a wait the client would otherwise see nothing of; templated,
    # never the narrator's (PROGRESS)
    "queue": "waiting: {reason}, position {n}",
    # the same with the estimate, when the hold times give one (server/queue_wait.py)
    "queue_eta": "waiting: {reason}, position {n}, about {eta}",
    "prefill": "reading {n} tokens at ~{rate} tok/s, about {left} left",
    "thinking": "thinking {elapsed}, {n} tokens",
    # N2: a stretch of thinking that ended, when the narrator (`ui.narrator: thoughts`) could not
    # sum it up in time; its summary is "Thought 1m 43s: ..." (narrator.Thoughts)
    "thought": "thought {elapsed}, {n} tokens",
}
# the profile's per-kind switches (`ui.kinds`) name these; `check` covers `checks`
KINDS = tuple(k for k in TEXTS if k not in ("checks", "queue_eta", "lift")) + ("tool", "plan")
PROGRESS = ("queue", "queue_eta", "prefill", "thinking")
# where the lines go (`ui.states_channel`): the reasoning, as they always did and as every profile
# stored before S1 keeps them, the content, outside the thinking (a new profile's default), or
# inline: each line the title of a reasoning part of its own (X1, INLINE)
CHANNELS = ("reasoning", "content", "inline")
# `ui.progress` when the profile says nothing: on with the lines
PROGRESS_DEFAULT = {"mode": "on", "prefill_every_s": 10, "thinking_every_s": 120,
                    "thinking_every_tokens": 4096}
# who set a cap, in words, for the narrator
CAP_BY = {"route": "the router, for the task's effort", "client": "the client's budget",
          "reserve": "the room the answer needs under max_tokens"}


def schema_label(task: str) -> str | None:
    """A monitor task's label (`D1` -> "reasoning loop"), for the narrator."""
    from server.conductor import schema
    return (schema.TASKS.get(task) or {}).get("label")

# a line the server writes: the marker and one of the templates, nothing else, so a line of
# Qwen's own that happens to start with `›` is never taken
_TEXT = "(?:" + "|".join((r"[^\n]{1,%d}" % MAX_CHARS).join(
    re.escape(part) for part in re.split(r"\{\w+\}", t)) for t in TEXTS.values()) + ")"
# a narrated line: any text on one line, closed by the narrator's sentinel (U+2060); a thought's
# summary (N2) is two short sentences after its duration, so it may run longer than a template
NARRATED_MAX_CHARS = 320
_TEXT = "(?:" + _TEXT + "|[^\n\u2060]{1,%d}\u2060)" % NARRATED_MAX_CHARS
_MID = re.compile(r"\n" + re.escape(MARKER) + _TEXT + r"\n\n")
_HEAD = re.compile(r"\A" + re.escape(MARKER) + _TEXT + r"(?:\n\n|\n?\Z)")
_TAIL = re.compile(r"\n" + re.escape(MARKER) + _TEXT + r"\n?\Z")
# the same line as a reasoning part's title (`inline`): `**` around the marker and the text
_IMID = re.compile(r"\n\*\*" + re.escape(MARKER) + _TEXT + r"\*\*\n\n")
_IHEAD = re.compile(r"\A\*\*" + re.escape(MARKER) + _TEXT + r"\*\*(?:\n\n|\n?\Z)")
_ITAIL = re.compile(r"\n\*\*" + re.escape(MARKER) + _TEXT + r"\*\*\n?\Z")


def channel(ui_doc: dict | None, fmt: str | None = None) -> str:
    """Where a profile's lines go: its `ui.states_channel`, else the reasoning (a profile stored
    before S1). `inline` needs reasoning parts: with `fmt` other than `reasoning_content` it is
    `content`."""
    where = (ui_doc or {}).get("states_channel") or "reasoning"
    if where == "inline" and fmt is not None and fmt != "reasoning_content":
        return "content"
    return where


def progress(ui_doc: dict | None) -> dict:
    """A profile's progress lines with their defaults (`ui.progress`, absent: on)."""
    return {**PROGRESS_DEFAULT, **((ui_doc or {}).get("progress") or {})}


def duration(s: float) -> str:
    """`47 s`, `4 min`: seconds below a minute and a half, whole minutes above."""
    s = max(0.0, float(s))
    return f"{s:.0f} s" if s < 90 else f"{s / 60:.0f} min"


def rate(tps: float) -> str:
    """`750`, `1,230`: a rate to two or three significant figures."""
    return f"{round(float(tps), -1 if tps >= 100 else 0):,.0f}"


def frame(text: str) -> str:
    """One step as it goes out: the marker, the text on one line, a blank line after it."""
    text = " ".join(str(text).split())[:MAX_CHARS]
    return "\n" + MARKER + text + "\n\n"


def inline(line: str) -> str:
    """A framed line (`frame`, or the narrator's) as a reasoning part's title: `\\n**› text**\\n\\n`.
    opencode's title is `**` + no `*` + `**`, so a `*` in the text goes."""
    body = line.strip("\n").replace("*", "")
    return "\n**" + body + "**\n\n"


def strip_text(text: str) -> str:
    """`text` without any framed line: exact wherever the frame is whole, and at the ends also
    when a client trimmed the frame's outer newlines; a reasoning part's title (`inline`) too."""
    if MARKER not in text:
        return text
    prev = None
    while prev != text:
        prev = text
        text = _MID.sub("", text)
        text = _IMID.sub("", text)
        text = _HEAD.sub("", text, count=1)
        text = _IHEAD.sub("", text, count=1)
        text = _TAIL.sub("", text, count=1)
        text = _ITAIL.sub("", text, count=1)
    return text


def strip_messages(messages) -> tuple[list, int]:
    """The messages with every framed line taken out of the assistant messages' reasoning and
    content, and how many characters went. The same list when nothing changes (the common case:
    no assistant message carries the marker)."""
    if not isinstance(messages, list):
        return messages, 0
    out, removed = None, 0
    for i, m in enumerate(messages):
        if not isinstance(m, dict) or m.get("role") != "assistant":
            continue
        new = None
        for key in ("reasoning_content", "reasoning", "content"):
            v = m.get(key)
            if isinstance(v, str) and MARKER in v:
                s = strip_text(v)
                if s != v:
                    new = dict(new or m)
                    new[key] = s
                    removed += len(v) - len(s)
            elif key == "content" and isinstance(v, list):
                parts = None
                for j, p in enumerate(v):
                    t = p.get("text") if isinstance(p, dict) else None
                    if isinstance(t, str) and MARKER in t:
                        s = strip_text(t)
                        if s != t:
                            parts = parts or list(v)
                            parts[j] = dict(p, text=s)
                            removed += len(t) - len(s)
                if parts is not None:
                    new = dict(new or m)
                    new["content"] = parts
        if new is not None:
            out = out or list(messages)
            out[i] = new
    return (out if out is not None else messages), removed


def into_full(content: str, reasoning: str | None, fmt: str, in_think: bool, ui: "States",
              where: str = "reasoning") -> tuple[str, str | None]:
    """A non-streamed answer's `(content, reasoning_content)` with the request's lines. In the
    reasoning: those said before the generation at the head of the reasoning, the rest at its end,
    in the `reasoning_content` field (formats `reasoning_content` and `both`; created when the turn
    did not think) and inside the `<think>` block of the content (`tags` and `both`, when it has
    one). In the content (`where`, S1; `inline` too: an answer that is not streamed has no parts):
    all of them at the head of the content, in their order."""
    if where in ("content", "inline"):
        return "".join(ui.shown(e) for e in ui.log) + content, reasoning
    # SRV-185: the memory lines go after the answer, in the content (as the stream sends them)
    mem = "".join(ui.shown(e) for e in ui.log if e["kind"] in MEMORY_KINDS)
    log = [e for e in ui.log if e["kind"] not in MEMORY_KINDS]
    cut = len(log) if ui.head is None else len([e for e in ui.log[:ui.head]
                                                 if e["kind"] not in MEMORY_KINDS])
    head = "".join(ui.shown(e) for e in log[:cut])
    tail = "".join(ui.shown(e) for e in log[cut:])
    content = content + mem
    if fmt in ("reasoning_content", "both"):
        reasoning = head + (reasoning or "") + tail
    opener = "<think>\n"
    if fmt in ("tags", "both") and in_think and content.startswith(opener):
        end = content.find("</think>")
        end = len(content) if end < 0 else end
        content = opener + head + content[len(opener):end] + tail + content[end:]
    return content, reasoning


def status_clients(spec: str | None = None) -> frozenset:
    """`QWEN38_STATUS_CLIENTS` (`kind,kind`): the client kinds that get the lines said before the
    first generated token as `status` events (`StatusBar`)."""
    if spec is None:
        from engine.settings import SETTINGS as _S
        spec = _S.get("STATUS_CLIENTS")
    return frozenset(k.strip() for k in (spec or "").split(",") if k.strip())


def plain(line: str) -> str:
    """A framed line (`frame`, the narrator's, `inline`'s) as plain text: no marker, no frame, no
    sentinel, the first letter upper case ("Reading 12,874 tokens, about 14 s left")."""
    text = " ".join(str(line).replace("⁠", "").replace("**", "").split())
    if text.startswith(MARKER.strip()):
        text = text[len(MARKER.strip()):].lstrip()
    return text[:1].upper() + text[1:]


class StatusBar:
    """SRV-237 / SRV-249 (UX2, 2026-10-07): Open WebUI times a reasoning block from its first
    `reasoning_content` delta to the first content delta and shows that as "Thought for N
    seconds". The waiting and reading lines went out as reasoning, so a thinking-off summary that
    read 12,874 tokens showed "Thought for 14 seconds" (its block held only the two reading lines)
    and a one-line "17 x 23" that waited 35.8 s for the other request showed "Thought for 34
    seconds".

    Open WebUI 0.11.4 hands a stream chunk that carries an `event` key to its event emitter
    (backend/open_webui/utils/middleware.py: `if 'event' in data: await event_emitter(...)`); an
    event of type `status` is the line under the model's name that tools and web search use: kept
    with the message (`statusHistory`), the newest one shown, shimmering while `done` is false,
    the earlier ones a click away. So for the client kinds in `QWEN38_STATUS_CLIENTS` every line
    said before the model's first generated token (the router's, the packages, the waiting and the
    reading lines) goes out as such an event, and the reasoning block opens with the model's own
    first thought: a turn that does not think shows no "Thought for" at all. At the first token
    (`close`) one last event, `done`, says what the wait was ("Read 12,874 tokens in 14 s").
    Lines after the first token (a cap, a check, a long thinking) stay in the profile's channel.

    The chunk has an empty `choices` list, which every OpenAI client skips; it goes only to the
    client kinds named, and nothing of it is a token (`usage`, the transcript and the prompt never
    see it)."""

    def __init__(self):
        self.live = True                  # until the first generated token
        self.n = 0
        self.last: str | None = None
        self.kinds: set[str] = set()

    def line(self, kind: str, line: str) -> dict | None:
        """The event for one line said before the first token; None once the bar is closed."""
        if not self.live:
            return None
        text = plain(line)
        if not text:
            return None
        self.n += 1
        self.last = text
        self.kinds.add(kind)
        return {"type": "status", "data": {"description": text, "done": False}}

    def close(self, read: tuple[int, float] | None = None,
              waited: float | None = None) -> dict | None:
        """The last event, at the first generated token (or the end of a request that wrote
        none): what the wait was -- the tokens read and how long (`read`, when a reading line was
        shown), the time waited (`waited`, when a waiting line was) -- else the newest line,
        done. None when no line went out (nothing to close) or it was closed already."""
        if not self.live:
            return None
        self.live = False
        if not self.n:
            return None
        parts = []
        if waited is not None and self.kinds & {"queue", "queue_eta"}:
            parts.append(f"waited {duration(waited)}")
        if read is not None and "prefill" in self.kinds and read[0] > 0:
            parts.append(f"read {read[0]:,} tokens in {duration(read[1])}")
        text = plain(", then ".join(parts)) if parts else self.last
        return {"type": "status", "data": {"description": text, "done": True}}


class States:
    """One request's steps: what is waiting to go out, and everything it said. `narrator`: the
    request's narrator (narrator.Lines), called once per line as it goes out; None: templates.
    `voice` (narrator v3, `ui.narrator: live`): the user's language (`en`, `de`); the harness's
    lines then go out in plain words (voice.natural, the sentinel's frame), consecutive packages
    as one line, and the live lines (`live`) beside them. `labels`: class id -> the router's
    label, for the route line in words."""

    def __init__(self, kinds: dict | None = None, narrator=None, voice: str | None = None,
                 labels: dict | None = None):
        self.kinds = kinds or {}
        self.narrator = narrator
        self.voice = voice
        self.labels = labels or {}
        self.pending: list[dict] = []
        self.log: list[dict] = []
        self.head: int | None = None       # how many were said before the generation started
        self.t0 = time.time()

    def on(self, kind: str) -> bool:
        base = {"checks": "check", "queue_eta": "queue"}.get(kind, kind)
        return bool(self.kinds.get(base, True))

    def add(self, kind: str, extra: dict | None = None, *, text_of: str | None = None,
            **facts) -> None:
        """One step: its template filled with `facts`; `extra` are facts for the narrator only;
        `text_of`: another template under this kind's switch (E1's lift under `continue`)."""
        if not self.on(kind):
            return
        try:
            text = TEXTS[text_of or kind].format(**facts)
        except (KeyError, IndexError, ValueError):
            return
        e = {"kind": kind, "text": text, "summary": None,
             "at_ms": int((time.time() - self.t0) * 1000)}
        if self.narrator is not None and kind not in PROGRESS:
            e["facts"] = {**facts, **(extra or {})}
        if self.voice is not None:
            e["f"] = {**facts, **(extra or {})}
        self.log.append(e)
        self.pending.append(e)

    def live(self, kind: str, text: str, info: dict | None = None, **facts) -> None:
        """A live line of narrator v3 (voice.py): its text is final (facts in words and the small
        model's phrase), it goes out in the sentinel's frame, `info` is what happened."""
        if not self.on(kind):
            return
        text = " ".join(str(text).split())
        from server.conductor.narrator import SENTINEL
        text = text.replace(SENTINEL, "")[:NARRATED_MAX_CHARS - 4]
        e = {"kind": kind, "text": text, "summary": text, "narrator": dict(info or {}, live=True),
             "at_ms": int((time.time() - self.t0) * 1000)}
        if facts:
            e["f"] = facts
        self.log.append(e)
        self.pending.append(e)

    def written(self, kind: str, summary: str | None, info: dict, **facts) -> None:
        """A step whose line the narrator already wrote (N2, a thought's summary), or tried to:
        `summary` None goes out as the template, `info` is what happened (journaled)."""
        if not self.on(kind):
            return
        try:
            text = TEXTS[kind].format(**facts)
        except (KeyError, IndexError, ValueError):
            return
        e = {"kind": kind, "text": text, "summary": summary, "narrator": info,
             "at_ms": int((time.time() - self.t0) * 1000)}
        self.log.append(e)
        self.pending.append(e)

    def shown(self, e: dict) -> str:
        """The line as it goes out: the narrator's, written now (once), or the template's (always
        for the progress lines, whose numbers no model rewrites, unless the thought narrator
        wrote one: `written`); with the voice (v3) the template's facts in plain words."""
        if self.voice is not None and "narrator" not in e:
            if "shown" not in e:
                from server.conductor import voice as voice_mod
                f = dict(e.get("f") or {})
                if e["kind"] == "route" and f.get("cls") in self.labels:
                    f["label"] = self.labels[f["cls"]]
                try:
                    e["shown"] = voice_mod.natural(e["kind"], f, self.voice)
                except Exception:                                  # noqa: BLE001
                    e["shown"] = None
            if e.get("shown"):
                from server.conductor.narrator import frame as narrated
                return narrated(e["shown"].replace("*", "\u2217"))
            return frame(e["text"])
        if "narrator" in e:
            if e["summary"]:
                from server.conductor.narrator import frame as narrated
                return narrated(e["summary"])
            return frame(e["text"])
        if self.narrator is None or e["kind"] in PROGRESS:
            return frame(e["text"])
        if "narrator" not in e:
            try:
                e["summary"], e["narrator"] = self.narrator(e)
            except Exception as exc:                               # noqa: BLE001
                e["summary"], e["narrator"] = None, {"why": "error",
                                                     "error": f"{type(exc).__name__}: {exc}"[:200]}
        if e["summary"]:
            from server.conductor.narrator import frame as narrated
            return narrated(e["summary"])
        return frame(e["text"])

    def _out(self) -> list[dict]:
        """The pending entries, taken; with the voice, consecutive packages as one line (the
        others stay in the log, marked `merged`)."""
        out, self.pending = self.pending, []
        if self.voice is None:
            return out
        merged: list[dict] = []
        for e in out:
            prev = merged[-1] if merged else None
            if (e["kind"] == "package" and prev is not None and prev["kind"] == "package"
                    and "narrator" not in e and "narrator" not in prev):
                pf = prev.setdefault("f", {})
                pf.setdefault("names", [pf.get("name")]).append((e.get("f") or {}).get("name"))
                e["merged"] = True
                continue
            merged.append(e)
        return merged

    def take(self) -> list[str]:
        return [self.shown(e) for e in self._out()]

    def take_kinds(self) -> list[tuple[str, str]]:
        """`take` with each line's kind: `(kind, line)`."""
        return [(e["kind"], self.shown(e)) for e in self._out()]

    def last_line(self) -> dict | None:
        """The newest line said (the live view's): its kind, its text as shown, when."""
        for e in reversed(self.log):
            if e.get("merged") or any(e is p for p in self.pending):
                continue
            text = e.get("shown") or e.get("summary") or e["text"]
            return {"kind": e["kind"], "text": text, "at_ms": e["at_ms"]}
        return None

    # ------------------------------------------------------------------ from the conductor
    def route(self, row: dict | None, injected, packages: dict, compaction: bool) -> None:
        """The router's steps, once its plan is applied: a compaction it saw, the class a model
        decided this turn (not a class kept on a tool turn) with its effort, every package a
        note of this turn carries."""
        if row is None:
            return
        if compaction:
            self.add("compaction")
        if row.get("stages") and row.get("class_id"):
            eff = row.get("effort")
            effort = ("thinking off" if eff == "off" else f"{eff} effort") if eff else "default"
            self.add("route", cls=row["class_id"], effort=effort)
        for n in injected or []:
            for p in n.get("parts") or []:
                if p.get("kind") == "package":
                    meta = (packages or {}).get(p.get("name")) or {}
                    self.add("package", pkind=meta.get("kind") or "skill", name=p.get("name"))

    def monitor(self, ev: dict) -> None:
        """A session-monitor decision that acts (the `Supervisor.on_event` hook)."""
        d, task = ev.get("decision"), str(ev.get("task") or "")
        detail = ev.get("detail") or {}
        call = detail.get("call") if isinstance(detail.get("call"), dict) else {}
        # what the narrator may say beyond the template: the tools and files of a check, the
        # call a repair fixed, the reason the monitor gave
        extra = {k: v for k, v in (("tools", detail.get("tools")), ("files", detail.get("files")),
                                   ("tool", call.get("name")),
                                   ("check", schema_label(task) if task else None)) if v}
        why = dict(extra, reason=ev.get("reason")) if ev.get("reason") else extra
        if d == "CHECK":
            n = int(detail.get("calls") or 1)
            self.add("check", extra) if n == 1 else self.add("checks", extra, n=n)
        elif d == "CLOSE_REASONING":
            m = re.match(r"budget\((\d+)\)", str(ev.get("reason") or ""))
            if task in ("route", "client", "reserve") and m:
                self.add("cap", {"set by": CAP_BY[task]}, cap=f"{int(m.group(1)):,}")
            elif task == "D2":
                m = re.search(r"(\d+)", str(ev.get("reason") or ""))
                self.add("backstop", cap=f"{int(m.group(1)):,}") if m else self.add("close")
            elif task in ("D1", "D1b"):
                self.add("loop", extra)
            else:
                self.add("close", extra)
        elif d == "CONTINUE_HIDDEN" and task == "E2":
            # the claim check's note: a steer, not an empty answer
            self.add("steer", label=schema_label(task) or task)
        elif d == "CONTINUE_HIDDEN" and detail.get("case") == "C":
            # SRV-199: the calls the model wrote inside its thinking go out as its answer
            self.add("continue", dict(extra, lift=True, tools=detail.get("calls")), text_of="lift")
        elif d == "CONTINUE_HIDDEN":
            self.add("continue")
        elif d == "REPAIR_CALL":
            self.add("repair", why)
        elif d == "REGENERATE_CALL":
            self.add("regenerate", why)
        elif d == "GIVE_UP":
            self.add("give_up", why)
        elif d == "PROMPT_TAIL_NOTE":
            from server.conductor import schema
            self.add("steer", label=(schema.TASKS.get(task) or {}).get("label") or task)
