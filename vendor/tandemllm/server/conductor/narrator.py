"""The narrator: a small model inside the engine writes the state lines' one-line summaries (P1).

G1's state lines (states.py, `ui.states`) are templates: "deciding: code_edit, high effort",
"checking the tool call". Each line keeps a `summary` field for a model-written one. With a
profile's `ui.narrator: on` the narrator writes it: the event's kind, its template and its facts
(the class and its label, the effort, a cap, the tool names and files of a check, the counts) go
to the small model of the `narrator` role (MiniCPM5-2B, engine/generative), which answers with one
plain line, and that line goes out instead of the template.

THE LIMITS. One line is at most `MAX_WORDS` words and `max_tokens` tokens (24), written inside the
role's latency budget (`latency_budget_ms` in the roles document, 150 ms by default), from the call
to the last token. A request spends at most `REQUEST_LINES` budgets on its lines; after that its
lines are the templates. Whatever fails -- the model not loaded yet (its first line starts the
load), refused by the residency planner, busy, the budget gone before the end token, the token
cap reached, too many words, an error -- the line is the template, and the journal says why.

QWEN IS NEVER BESIDE IT. The narrator runs on the request's own thread when the line goes out:
before the prefill for the router's line, between two of Qwen's rounds for the monitor's, after
the last token for the tool checks. Qwen waits meanwhile and nothing of Qwen runs at the same time
(Harmony 3: during decode the GPU is Qwen's); the engine's lock is held by the request anyway, and
the narrator holds it at most its budget and one step a line. The model's own stream is
synchronised before Qwen goes on (engine/decision/device.py).

THE FRAME. A narrated line goes out as `"\\n› " + text + "\\u2060\\n\\n"`: the template's frame
with a WORD JOINER (U+2060, invisible, not whitespace, so no client's trim and no template's
`|trim` removes it) before the closing newlines. The strip (states.py) takes a framed line of any
text that ends in it, so a narrated line, which no template matches, never reaches Qwen's prompt
either; Qwen does not write U+2060 at the end of a `› ` line of its own.

`roles.narrator.mode`: `on` writes the lines, `shadow` writes them into the journal only (the
client sees the templates: line quality can be read without anyone seeing them), `off` never asks
the model.

THOUGHTS (N2, 2026-10-03). The owner: "when the small model works, for each thinking stretch something
like 'thought about 2 min about X and Y' in two short sentences; the small model does not need the
whole conversation, only the thought text." With a profile's `ui.narrator: thoughts` (the state
lines on; the other lines stay templates) the narrator's model -- Qwen3.5-0.8B or 2B on the engine's
own forward (engine/generative/qwen35.py), the 27B's tokenizer -- reads only the thought: Qwen's
own token ids of it, the last THOUGHT_WINDOW of them, no second tokenisation. It writes at most
THOUGHT_MAX_TOKENS tokens, greedy, thinking off, from a few-shot prompt (THOUGHT_SYSTEM,
THOUGHT_EXAMPLES): what Qwen thought about and decided, and its next step. The duration and the
token count are the engine's: "Thought 1m 43s: planned the level layout and decided on a fixed
timestep; next it writes game.js". When:

  * a stretch of thinking ends (the block closed, by Qwen or by a hidden close, or the turn ended
    inside it): a `thought` line, for a stretch of at least THOUGHT_MIN_TOKENS (a shorter thought
    is short enough to read, and gets no line);
  * a long thinking reaches the progress line's mark (`ui.progress`, every 2 min or 4,096 tokens):
    the `thinking` line is the summary of the thought since the last mark ("Thinking 4m 10s: ...").

THE BUDGET. Qwen waits while the narrator runs (between two rounds, on the request's thread, the
engine's lock held), so its time is rationed: over a request it may hold Qwen at most THOUGHT_SHARE
(3 %) of the time since Qwen's first token, in slices of at most the role's latency budget (150 ms)
each. A long thought has earned enough by its end for the whole summary, which goes out right after
it; a short one is written in slices while Qwen writes the answer, and what is left at the turn's
end gets up to THOUGHT_END_MS (the decode is over then; the finish chunk waits). A summary not done
by then, a refusal, an error, an answer the clean refuses: the template ("thought 1 min, 4,096
tokens"), and the line's `narrator` record says why. The lines are state lines (states.written):
framed with the sentinel, stripped from every prompt, never tokens of the answer or the usage, kept
in the journal row (`extra.states[]`, and `extra.narrator` the request's totals).
"""

from __future__ import annotations

import re
import time

from engine.settings import SETTINGS as _S

MAX_WORDS = 14
MAX_TOKENS = 24
# a request's lines share this many line budgets; the rest are templates
REQUEST_LINES = 4
# the try route's ceilings (the measurement lifts the budget to read what the model writes)
TRY_MAX_BUDGET_MS = 10_000
TRY_MAX_TOKENS = 64
TRY_MAX_EVENTS = 64
SENTINEL = "\u2060"
OFF_WHY = ("this engine serves no narrator: QWEN38_NARRATOR_MODEL is empty (set it in "
           "ops/serve.env, e.g. qwen3.5-0.8b, and restart)")

# The prompt (2026-10-02, read on the box CPU against 12 sample events): without examples and with
# the template beside the event, MiniCPM5-2B copied the template word for word; with the template
# left out, the facts in plain words (a class's label, "very high effort") and three examples it
# writes a line of its own. The system turn is the same for every line, so its rows stay in the
# model's cache and only the event is prefilled.
SYSTEM = ("You turn an event inside an AI coding engine into the status line its user reads while "
          "waiting. Write like a friendly colleague: plain words, present tense, 4 to 12 words, "
          "no quotes, no emoji, no markdown, no full stop. Do not copy the engine's label; say "
          "identifiers in plain words. Use only the facts given; never invent file names, numbers "
          "or reasons.\n\n"
          "Examples:\n"
          "Event: the engine chose what kind of task this is and how hard to think about it\n"
          "Facts: task: UI design; effort: medium effort\n"
          "Line: Treating this as UI design work, with moderate thought\n\n"
          "Event: the engine loaded a skill into the prompt\n"
          "Facts: kind: skill; name: voxel-seasons\n"
          "Line: Pulling in the voxel-seasons skill\n\n"
          "Event: the engine checks the model's tool call before the client runs it\n"
          "Facts: tools: write; files: web/index.html\n"
          "Line: Double-checking the write to web/index.html before it runs")
# the template's fact names, and words, as the model reads them
FACT_NAMES = {"cls": "task", "pkind": "kind", "n": "count"}
EFFORT_WORDS = {"xhigh effort": "very high effort", "thinking off": "no thinking"}

# what each kind of step means, for the model
EVENTS = {
    "route": "the engine chose what kind of task this is and how hard to think about it",
    "compaction": "the long conversation so far was replaced by a short summary to save room",
    "package": "the engine loaded a skill into the prompt",
    "memory": "the engine read notes from its memory",
    "steer": "the engine added a hidden steering note for the model",
    "cap": "the thinking budget ran out, so the reasoning was closed",
    "backstop": "the reasoning reached the hard backstop and was closed",
    "loop": "the reasoning started repeating itself, so the engine steered it",
    "close": "the engine closed the reasoning",
    "continue": "the model stopped without writing an answer, so the engine makes it go on (with "
                "`lift`: it had written its tool calls inside its thinking, and they are sent as "
                "its answer)",
    "check": "the engine checks the model's tool call before the client runs it",
    "checks": "the engine checks the model's tool calls before the client runs them",
    "repair": "a tool call was malformed and the engine fixed it",
    "regenerate": "a tool call was wrong and the model writes it again",
    "give_up": "the engine gave up after the tool call failed again",
}
FACT_CHARS = 120
_BAD = re.compile(r"[\u203a\u2060<>\n]|\|im_|</?think")


def messages(entry: dict, labels: dict | None = None) -> list[dict]:
    """The model's prompt for one line: the fixed system turn and the event with its facts, in
    words (`labels`: class id -> the router's label for it). Facts pass the secret filter."""
    from server.conductor.redact import redact
    facts = []
    for k, v in (entry.get("facts") or {}).items():
        if v is None or v == "" or v == []:
            continue
        v = ", ".join(map(str, v)) if isinstance(v, (list, tuple)) else str(v)
        if k == "cls":
            v = (labels or {}).get(v) or v.replace("_", " ")
        elif k == "effort":
            v = EFFORT_WORDS.get(v, v)
        v = redact(" ".join(v.split()))[0][:FACT_CHARS]
        facts.append(f"{FACT_NAMES.get(k, k)}: {v}")
    kind = entry["kind"]
    user = (f"Event: {EVENTS.get(kind, kind)}\nFacts: {'; '.join(facts) if facts else 'none'}"
            f"\nLine:")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def clean(text: str) -> tuple[str | None, str | None]:
    """(the line, None), or (None, why it cannot be one)."""
    s = (text or "").strip().split("\n", 1)[0].strip()
    if s.lower().startswith("line:"):
        s = s[5:]
    s = s.strip("\"'`*_ ").rstrip(".").strip()
    s = " ".join(s.split())
    if not s:
        return None, "empty"
    if _BAD.search(s):
        return None, "markup"
    if len(s.split()) > MAX_WORDS:
        return None, "words"
    from server.conductor.states import MAX_CHARS
    if len(s) > MAX_CHARS:
        return None, "chars"
    return s, None


def frame(text: str) -> str:
    """A narrated line as it goes out (states.frame plus the sentinel)."""
    from server.conductor.states import MARKER
    return "\n" + MARKER + text + SENTINEL + "\n\n"


class Narrator:
    """The `narrator` role over its slot: one line per state, inside the role's budget."""

    def __init__(self, slot, role=lambda: None, labels=lambda: {}):
        """`role()`: the roles document's `narrator` entry (its mode and latency budget);
        `labels()`: class id -> the router's label."""
        self.slot, self.role, self.labels = slot, role, labels
        self._around = None
        self._voice = None

    def settings(self) -> dict:
        r = self.role() or {}
        return {"mode": r.get("mode", "on"), "budget_ms": int(r.get("latency_budget_ms") or 150)}

    def around(self) -> tuple[list[int], list[int]]:
        """The thought prompt's fixed head and tail in the loaded model's ids (cached per load)."""
        key = (id(self.slot.model), self.slot.loaded_at)
        if self._around is None or self._around[0] != key:
            self._around = (key, self.slot.model.encode_around(thought_messages(), MARK,
                                                                lead=THOUGHT_LEAD))
        return self._around[1]

    def voice_around(self, task: str, lang: str) -> tuple[list[int], list[int]]:
        """v3's prompt head (one for every task and language) and the tail naming `task` and
        `lang`, in the loaded model's ids (cached per load)."""
        from server.conductor import voice
        key = (id(self.slot.model), self.slot.loaded_at)
        if self._voice is None or self._voice[0] != key:
            self._voice = (key, {})
        cache = self._voice[1]
        if (task, lang) not in cache:
            head, tail = self.slot.model.encode_around(voice.prompt(voice.MARK, task, lang),
                                                       voice.MARK)
            first = cache.get("head")
            if first is not None and first != head:
                raise RuntimeError("the voice's prompt head differs between tasks")
            cache["head"] = head
            cache[(task, lang)] = tail
        return cache["head"], cache[(task, lang)]

    def for_voice(self, ui, *, lang: str, opened: bool, close_id: int | None = None,
                  decode=None, share=None, settings: dict | None = None):
        """What one streamed request feeds (`ui.narrator: live`, v3, server/conductor/voice.py);
        None when the role is off or its model cannot be advanced in slices."""
        if self.settings()["mode"] == "off" or not getattr(self.slot.model, "jobs", False):
            return None
        from server.conductor.voice import Voice
        return Voice(self, ui, lang=lang, opened=opened, close_id=close_id, decode=decode,
                     share=share, settings=settings)

    def for_thoughts(self, ui, *, opened: bool, close_id: int | None = None,
                     decode=None, reader=None) -> "Thoughts | None":
        """What one streamed request feeds its ids to (`ui.narrator: thoughts`); None when the
        role is off or its model cannot be advanced in slices. `reader` (SR1, `shadow.narrator:
        on`): the session's shadow (server/conductor/shadow.py `ThoughtSlot`), which reads the
        whole session instead of the last THOUGHT_WINDOW ids of the thought."""
        s = self.settings()
        model = reader.model if reader is not None else self.slot.model
        if s["mode"] == "off" or not getattr(model, "jobs", False):
            return None
        return Thoughts(self, ui, opened=opened, close_id=close_id, decode=decode,
                        budget_ms=s["budget_ms"], shadow=s["mode"] == "shadow", reader=reader)

    def for_request(self) -> "Lines | None":
        """What one request's States calls per line; None when the role is off."""
        s = self.settings()
        if s["mode"] == "off":
            return None
        return Lines(self, s["budget_ms"], shadow=s["mode"] == "shadow")

    def line(self, entry: dict, *, budget_ms: float, max_tokens: int = MAX_TOKENS,
             held: bool = True) -> tuple[str | None, dict]:
        """(the narrated line or None, what happened: `why` None or the reason for the template,
        `ms`, `tokens`, `finish`, the raw text when it was refused)."""
        r = self.slot.run(messages(entry, self.labels()), max_new=max_tokens, budget_ms=budget_ms,
                          held=held)
        info = {k: r[k] for k in ("ms", "tokens", "finish", "prompt_tokens", "reused")
                if k in r}
        if r.get("why"):
            return None, dict(info, why=r["why"], **({"error": r["error"]} if r.get("error")
                                                     else {}))
        if r["finish"] != "stop":
            return None, dict(info, why=r["finish"], raw=r["text"][:200])
        text, why = clean(r["text"])
        if text is None:
            return None, dict(info, why=why, raw=r["text"][:200])
        return text, dict(info, why=None)

    def try_events(self, events: list, *, budget_ms: float | None = None,
                   max_tokens: int | None = None) -> list[dict]:
        """`POST /v1/conductor/narrator/try`: each event's template and line, in the idle window
        (the engine's lock taken per line)."""
        from server.conductor.states import TEXTS
        budget = float(budget_ms or self.settings()["budget_ms"])
        cap = int(max_tokens or MAX_TOKENS)
        out = []
        for ev in events:
            kind, facts = ev.get("kind"), dict(ev.get("facts") or {})
            try:
                text = TEXTS[kind].format(**facts)
            except (KeyError, IndexError, ValueError):
                out.append({"kind": kind, "error": f"no template for {kind!r} with these facts"})
                continue
            entry = {"kind": kind, "text": text, "facts": dict(facts, **(ev.get("extra") or {}))}
            line, info = self.line(entry, budget_ms=budget, max_tokens=cap, held=False)
            out.append({"kind": kind, "template": text, "line": line,
                        "shown": line if line is not None else text, **info})
        return out


class Lines:
    """One request's narrator: the role's budget per line, `REQUEST_LINES` budgets in all."""

    def __init__(self, narrator: Narrator, budget_ms: float, *, shadow: bool = False):
        self.narrator, self.budget_ms, self.shadow = narrator, float(budget_ms), shadow
        self.left_ms = REQUEST_LINES * self.budget_ms

    def __call__(self, entry: dict) -> tuple[str | None, dict]:
        if self.left_ms <= 0:
            return None, {"why": "request_budget"}
        budget = min(self.budget_ms, self.left_ms)
        line, info = self.narrator.line(entry, budget_ms=budget)
        self.left_ms -= float(info.get("ms") or 0.0)
        if self.shadow:
            return None, dict(info, shadow=line)
        return line, info


class Share:
    """A request's small-model time (v3): `share` of the time since Qwen's first token, in ms.
    The voice draws on it first, the shadow reader's in-turn reading after it (shadow.Inline),
    so the two together never hold Qwen longer than the share."""

    def __init__(self, share: float = 0.03, clock=time.perf_counter):
        self.share, self.clock = float(share), clock
        self.t0: float | None = None
        self.spent = 0.0

    def start(self, now: float | None = None) -> None:
        if self.t0 is None:
            self.t0 = self.clock() if now is None else now

    def allow(self, now: float | None = None) -> float:
        if self.t0 is None:
            return 0.0
        now = self.clock() if now is None else now
        return self.share * (now - self.t0) * 1e3 - self.spent

    def spend(self, ms: float) -> None:
        self.spent += float(ms)


# ---------------------------------------------------------------------- thoughts (N2)
# a stretch shorter than this gets no line; the window the model reads; its answer's cap
THOUGHT_MIN_TOKENS = 128
THOUGHT_WINDOW = 1500
THOUGHT_MAX_TOKENS = 40
THOUGHT_MAX_WORDS = 45
# the share of Qwen's time a request's summaries may take, the turn's end allowance, and the
# smallest slice worth starting
THOUGHT_SHARE = 0.03
THOUGHT_END_MS = 500
THOUGHT_MIN_SLICE_MS = 5
MARK = "@@THOUGHT@@"

# The prompt (2026-10-03): the instructions, then three made-up thoughts with their summaries as
# turns of their own, so the model copies the shape (a past-tense verb, the concrete things, the
# decision, "; next it ..."), and the answer begun with "It" (read on the box CPU with real thoughts:
# without it Qwen3.5-0.8B wrote in the thought's own first person, "I will create the TODO list
# ..."). The system turn and the examples are the same for every thought, so the model keeps their
# state and prefills only the thought (qwen35.py `keep`).
THOUGHT_SYSTEM = ("You summarise the private reasoning of an AI assistant for the person waiting on "
                  "it. Write about the assistant in the third person, as \"It\", never as \"I\". "
                  "One or two short sentences, at most 30 words: what it thought about and what it "
                  "decided, then its next step if it names one (\"; next it ...\"). Name the "
                  "concrete things: files, functions, errors, options. Plain words: no quotes, no "
                  "markdown, no lists.")
# the answer is begun for the model: it goes on after "It" with a verb, and the line keeps the rest
THOUGHT_LEAD = "It"
THOUGHT_EXAMPLES = (
    ("The user wants a small platformer in the browser. Let me think about the structure. I need "
     "a canvas, a game loop, the player with gravity and jumping, and some platforms. "
     "requestAnimationFrame gives variable frame times, so the physics would behave differently "
     "on fast and slow machines. A fixed timestep with an accumulator is better: update at 60 Hz, "
     "render as often as the browser allows. Levels: each level an array of strings, # for a "
     "platform tile and P for the start. Three levels are enough. First index.html with the "
     "canvas, then game.js with the loop.",
     "It planned the platformer and decided on a fixed 60 Hz timestep and levels as string "
     "grids; next it writes index.html and game.js"),
    ("The test test_parse_iso fails with ValueError: Invalid isoformat string. The input is "
     "2026-10-03T09:30:00Z. datetime.fromisoformat on Python 3.10 does not accept the trailing Z; "
     "3.11 does. The CI runs 3.10 according to the workflow file. So either bump the CI or handle "
     "the Z. Bumping the CI changes more than needed. Simplest: replace a trailing Z with +00:00 "
     "before parsing in parse_date in utils/dates.py, and check that naive strings still parse. "
     "Let me edit utils/dates.py and run the test again.",
     "It traced the failing test to fromisoformat rejecting a trailing Z on Python 3.10 and chose "
     "to rewrite it as +00:00 in parse_date; next it edits utils/dates.py"),
    ("Three options for the cache: Redis, a local LRU, or a table in the database. The service "
     "runs on one node, so a local LRU avoids a network hop, but it is lost on a restart. Restarts "
     "happen at deploys, maybe twice a week, and a cold cache only costs a few slower requests. "
     "Redis would be one more service to run, and the table is the slowest. So the local LRU, "
     "limited to 10,000 entries. I should mention the restart trade-off in the answer.",
     "It compared Redis, a local LRU and a database table for the cache and picked the LRU, "
     "accepting a cold cache after deploys; next it writes the answer"),
)


def thought_messages(thought: str = MARK) -> list[dict]:
    """The model's prompt for one thought: the instructions, the examples, the thought."""
    msgs = [{"role": "system", "content": THOUGHT_SYSTEM}]
    for t, a in THOUGHT_EXAMPLES:
        msgs += [{"role": "user", "content": "Reasoning:\n" + t},
                 {"role": "assistant", "content": a}]
    return msgs + [{"role": "user", "content": "Reasoning:\n" + thought}]


def span(s: float) -> str:
    """`43s`, `1m 43s`, `12m`, `1h 5m`: a thought's length as the line says it."""
    s = int(round(max(0.0, float(s))))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m {s % 60}s" if s < 600 and s % 60 else f"{s // 60}m"
    return f"{s // 3600}h {s % 3600 // 60}m"


_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def clean_summary(text: str, finish: str | None, room: int) -> tuple[str | None, str | None]:
    """(the summary, None) or (None, why it cannot be one): its first line, at most two sentences;
    an answer cut at the token cap keeps what ends at a full stop or before a `;`; at most
    THOUGHT_MAX_WORDS words and `room` characters."""
    s = next((ln for ln in (text or "").strip().split("\n") if ln.strip()), "").strip()
    for lead in ("summary:", "thought:", "reasoning:"):
        if s.lower().startswith(lead):
            s = s[len(lead):]
    s = " ".join(s.strip("\"'`*_ ").split())
    if finish == "length":
        cut = max(s.rfind(". "), s.rfind("; "), s.rfind("! "), s.rfind("? "))
        s = s if s.endswith((".", "!", "?")) else (s[:cut + 1] if cut > 0 else "")
        s = s.rstrip(";").strip()
    if not s:
        return None, "empty" if finish != "length" else "length"
    if _BAD.search(s):
        return None, "markup"
    s = " ".join(_SENTENCE.split(s)[:2]).rstrip(".").strip()
    if len(s) > 1 and s[0].isupper() and s[1].islower():
        s = s[0].lower() + s[1:]
    if len(s.split()) > THOUGHT_MAX_WORDS:
        return None, "words"
    if len(s) > room:
        return None, "chars"
    return s, None


class Thoughts:
    """One request's thought summaries (`ui.narrator: thoughts`), fed the request's ids as Qwen
    writes them (`token`), the progress line's marks (`mark`) and the turn's end (`finish`)."""

    def __init__(self, narrator: "Narrator", ui, *, opened: bool, close_id: int | None = None,
                 decode=None, budget_ms: float = 150.0, shadow: bool = False,
                 clock=time.perf_counter, log=print, reader=None):
        self.narrator, self.ui = narrator, ui
        # SR1: the session's shadow reads the whole session; else the narrator's own slot
        self.slot = reader if reader is not None else narrator.slot
        self.reader = reader
        self.close_id, self.decode, self.shadow = close_id, decode, shadow
        self.budget_ms, self.clock, self.log = float(budget_ms), clock, log
        self.t0: float | None = None             # Qwen's first token
        self.start: int | None = 0 if opened else None   # the open stretch's first id
        self.t_start: float | None = None
        self.mark_at: int | None = None          # where the last mark's window ended
        self.job: dict | None = None
        self.spent_ms = 0.0                      # what the slices held Qwen, before the end
        self.stats = {"lines": 0, "templates": 0, "none": 0, "superseded": 0, "ms": 0.0,
                      "end_ms": 0.0, "why": {}}
        self.broken: str | None = None

    # ------------------------------------------------------------------ from the token loop
    def token(self, ids: list[int], in_think: bool) -> None:
        """After each of Qwen's tokens: a stretch opens or ends, and a slice runs when the share
        allows it."""
        if self.broken:
            return
        try:
            now = self.clock()
            if self.t0 is None:
                self.t0 = now
            if in_think:
                if self.start is None:
                    self.start, self.t_start = len(ids), now
                elif self.t_start is None:
                    self.t_start = now
            elif self.start is not None:
                self._close(ids, now)
            self._work(now)
        except Exception as exc:                                  # noqa: BLE001
            self._broke(exc)

    def mark(self, ids: list[int], elapsed_s: float) -> bool:
        """The progress line's mark in a long thinking: the summary of the thought since the last
        mark starts (its line is the `thinking` line). False when there is nothing to sum up and
        the caller says the template now."""
        if self.broken or self.start is None or self.job is not None:
            return False
        try:
            lo = max(self.start, self.mark_at or 0)
            self.mark_at = len(ids)
            self._begin("thinking", ids[lo:], len(ids) - self.start, elapsed_s, upto=ids)
            self._work(self.clock())
            return True
        except Exception as exc:                                  # noqa: BLE001
            self._broke(exc)
            return False

    def finish(self, ids: list[int]) -> None:
        """The turn's end (the last token came): a stretch still open ends here, and a summary not
        done yet gets the end allowance, in slices of the role's budget."""
        if self.broken:
            return
        try:
            now = self.clock()
            if self.t0 is None:
                self.t0 = now
            if self.start is not None:
                self._close(ids, now)
            t_end = now + THOUGHT_END_MS / 1000.0
            while self.job is not None:
                left = (t_end - self.clock()) * 1e3
                if left < THOUGHT_MIN_SLICE_MS:
                    self._done("time")
                    break
                ms = self._slice(min(self.budget_ms, left))
                self.stats["end_ms"] += ms
                if ms <= 0 and self.job is not None:
                    self._done("time")
        except Exception as exc:                                  # noqa: BLE001
            self._broke(exc)

    def totals(self) -> dict:
        """The request's numbers for its journal row (`extra.narrator`)."""
        return dict(self.stats, ms=round(self.stats["ms"], 2), end_ms=round(self.stats["end_ms"], 2),
                    share_ms=round(self.spent_ms, 2), **({"broken": self.broken}
                                                         if self.broken else {}))

    # ------------------------------------------------------------------ inside
    def _close(self, ids: list[int], now: float) -> None:
        start, self.start = self.start, None
        t_start = self.t_start if self.t_start is not None else self.t0
        self.t_start = self.mark_at = None
        cut = len(ids)
        if self.close_id is not None:
            tail = ids[start:cut]
            if self.close_id in tail:
                cut = start + len(tail) - 1 - tail[::-1].index(self.close_id)
        n = cut - start
        if self.job is not None and self.job["kind"] == "thinking":
            self.job = None                        # the thought's own summary replaces it
            self.stats["superseded"] += 1
        if n < THOUGHT_MIN_TOKENS:
            self.stats["none"] += 1
            return
        self._begin("thought", ids[start:cut], n, now - t_start, upto=ids[:cut])

    def _begin(self, kind: str, thought: list[int], n: int, elapsed_s: float,
               upto: list[int] | None = None) -> None:
        if self.job is not None:
            self._done("superseded")
        facts = {"elapsed": _duration(elapsed_s), "n": f"{n:,}"}
        prefix = ("Thought " if kind == "thought" else "Thinking ") + span(elapsed_s) + ": "
        base = {"trigger": kind, "thought_tokens": n}
        why = self.slot.ensure()
        if why is not None:
            self.slot.calls[why] += 1
            return self._line(kind, None, dict(base, why=why), facts)
        if self.reader is not None:
            # SR1: the shadow holds the session; the job reads what it misses up to the thought's
            # end and asks for the summary there
            r = self.reader.begin_thought(None, list(upto if upto is not None else thought),
                                          THOUGHT_MAX_TOKENS)
            self.job = {"job": r["job"], "kind": kind, "facts": facts, "prefix": prefix,
                        "base": dict(base, window="session", shadow=True), "t": self.clock()}
            return
        model = self.slot.model
        head, tail = self.narrator.around()
        body = model.text_ids(thought, self.decode)[-THOUGHT_WINDOW:]
        r = self.slot.begin(head + body + tail, max_new=THOUGHT_MAX_TOKENS, keep=len(head))
        if r.get("why"):
            return self._line(kind, None, dict(base, why=r["why"], **(
                {"error": r["error"]} if r.get("error") else {})), facts)
        self.job = {"job": r["job"], "kind": kind, "facts": facts, "prefix": prefix,
                    "base": dict(base, window=len(body)), "t": self.clock()}

    def _work(self, now: float) -> None:
        """Slices while the request's share allows them."""
        while self.job is not None:
            allow = THOUGHT_SHARE * (now - self.t0) * 1e3 - self.spent_ms
            if allow < THOUGHT_MIN_SLICE_MS:
                return
            ms = self._slice(min(self.budget_ms, allow))
            self.spent_ms += ms
            if ms <= 0 and self.job is not None:
                return                             # no progress this round; the next token's
            now = self.clock()

    def _slice(self, budget_ms: float) -> float:
        job = self.job["job"]
        r = self.slot.advance(job, budget_ms=budget_ms)
        ms = float(r.get("ms") or 0.0)
        self.stats["ms"] += ms
        if r.get("why"):
            self._done(r["why"], r.get("error"))
        elif r.get("done"):
            self._done(None)
        return ms

    def _done(self, why: str | None, error: str | None = None) -> None:
        j, self.job = self.job, None
        job = j["job"]
        if why is not None and hasattr(job, "cancel"):
            job.cancel()                     # SR1: a shadow fork puts the session's state back
        info = dict(j["base"], why=why, ms=round(job.ms, 2), slices=job.slices,
                    tokens=len(job.out), prompt_tokens=len(job.ids), reused=job.reused,
                    finish=job.finish, waited_ms=round((self.clock() - j["t"]) * 1e3, 1),
                    **({"error": error} if error else {}))
        line = None
        if why is None:
            # a secret the thought held is never in a status line, nor in the journal
            from server.conductor.redact import redact
            from server.conductor.states import NARRATED_MAX_CHARS
            raw = self.slot.model.job_text(job)
            summary, bad = clean_summary(raw, job.finish, NARRATED_MAX_CHARS - len(j["prefix"]))
            if summary is None:
                info.update(why=bad, raw=redact(raw[:200])[0])
            else:
                line = j["prefix"] + redact(summary)[0]
        self._line(j["kind"], line, info, j["facts"])

    def _line(self, kind: str, line: str | None, info: dict, facts: dict) -> None:
        if self.shadow and line is not None:
            info, line = dict(info, shadow=line), None
        self.stats["lines" if line is not None else "templates"] += 1
        if info.get("why"):
            w = self.stats["why"]
            w[info["why"]] = w.get(info["why"], 0) + 1
        self.ui.written(kind, line, info, **facts)

    def _broke(self, exc: BaseException) -> None:
        self.broken = f"{type(exc).__name__}: {exc}"[:200]
        self.job = None
        self.log(f"[narrator] thoughts off for this request ({self.broken})", flush=True)


def _duration(s: float) -> str:
    from server.conductor.states import duration
    return duration(s)


# ---------------------------------------------------------------------- the server's wiring
def configured() -> str:
    return (_S.get("NARRATOR_MODEL") or "").strip()


def start(state: dict, gate, *, fake: bool = False, log=print):
    """The narrator for this server, or None when `QWEN38_NARRATOR_MODEL` is empty. Nothing
    loads here: the model loads when a profile's first narrated line asks for it."""
    name = configured()
    if not name:
        return None
    from engine import generative
    from engine.generative.slot import Slot
    from server import decider_api
    from server.conductor import residency, schema
    # N2: a Qwen3.5 model compares its tokenizer with the served checkpoint's, to read Qwen's ids
    engine_dir = getattr(getattr(state.get("engine"), "cfg", None), "path", None)
    model = generative.model(name, _S.get("NARRATOR_DIR") or None, allow_fake=fake,
                             engine_dir=engine_dir if isinstance(engine_dir, str) else None)
    device = None
    if not fake:
        from engine.decision.device import Device
        device = Device(_S.get("NARRATOR_DEVICE"))

    def journal(rec):
        cond = state.get("conductor")
        if cond is not None:
            residency.Journal(cond.root).write(rec)

    slot = Slot(model, role="narrator", gate=gate,
                planner=decider_api.planner(state, fake=fake, role="narrator"), device=device,
                log=lambda s: log(s, flush=True),
                contract=lambda f, n: schema.role_errors(f, n, state.get("tokenizer_sha")),
                journal=journal, mem_available=lambda: decider_api._mem_available_gib(fake))
    log(f"[narrator] {name} on {device or 'no device (fake)'}: loads through the residency "
        f"planner when ops/start.sh asks (v3: at the start, then kept loaded) or a profile's "
        f"first line asks", flush=True)
    return Narrator(slot, role=lambda: role_doc(state), labels=lambda: class_labels(state))


def role_doc(state: dict) -> dict:
    """The roles document's `narrator` (the built-in one in a document stored before it)."""
    from server.conductor import schema
    cond = state.get("conductor")
    roles = cond.store.peek("roles")[1] if cond is not None else schema.seed("roles")
    return roles.get("narrator") or schema.seed("roles")["narrator"]


def class_labels(state: dict) -> dict:
    """class id -> label, from the router document."""
    cond = state.get("conductor")
    if cond is None:
        return {}
    return {c["id"]: c["label"] for c in cond.store.peek("router")[1].get("classes") or []}


def status(state: dict) -> dict:
    """`GET /v1/conductor/narrator/status`."""
    n = state.get("narrator")
    if n is None:
        return {"enabled": False, "why": OFF_WHY}
    return {"enabled": True, "why": None, "settings": n.settings(), "slot": n.slot.status(),
            "limits": {"max_words": MAX_WORDS, "max_tokens": MAX_TOKENS,
                       "request_lines": REQUEST_LINES},
            # N2: the thought summaries (`ui.narrator: thoughts`)
            "thoughts": {"jobs": bool(getattr(n.slot.model, "jobs", False)),
                         # True: Qwen's ids go to the model as they are (the same tokenizer)
                         "qwen_ids": (getattr(n.slot.model, "same_ids", "") is None
                                      if getattr(n.slot.model, "jobs", False) else None),
                         "min_tokens": THOUGHT_MIN_TOKENS, "window": THOUGHT_WINDOW,
                         "max_tokens": THOUGHT_MAX_TOKENS, "share": THOUGHT_SHARE,
                         "end_ms": THOUGHT_END_MS}}
