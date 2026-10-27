"""Narrator v3: the voice -- one live line of what Qwen is doing right now (`ui.narrator: live`).

The owner, 2026-10-04: "I need sentences, long and short, based on what it is doing right now." The
state lines (states.py) said what the conductor did ("deciding: code_edit, xhigh effort") and
gave numbers ("thinking 3 min, 8,192 tokens"); narrator v2 (narrator.py Thoughts) summed up a
finished thought. v3 talks while Qwen works. A turn reads like this in opencode, each line the
title of a reasoning part of its own (states.INLINE):

    › code edit, maximum effort; packages coding, testing and 3 more
    › reading 20,745 tokens, about 18 s left
    › thinking 25 s: how the enemies should patrol and when the player can stomp them
    › thinking 50 s: the vector style for the player, gradients and a glowing visor
    › thought 7 min: five files, level, entities, physics, render and game; then a browser test
    › writing mario/render.js: the vector art for the player and the enemies
    › writing mario/render.js, 420 lines so far

WHERE EACH LINE COMES FROM.

  * Facts from the structure, never from a model: the duration ("thinking 3 min"), a tool's
    action and its target ("writing mario/game.js", "running npm test"), the lines a write has
    so far, the queue's place, the prefill's tokens. A tool call's target is read from the open
    block as Qwen writes it (`ToolWatch`), so a file name in a line is always the call's own.
  * The small model (the narrator's slot, Qwen3.5-0.8B on the engine's own forward, the 27B's
    token ids as they are) writes one short phrase after the facts: what the thinking is about
    now (`now`, every `every_s` of thinking from `first_s`), what a finished thought decided
    (`decided`, a stretch of at least `thought_min_s`), what a write is for (`why`), a plan in
    the answer (`plan`). One fixed prompt head serves all four tasks and both languages
    (`SYSTEM`, `EXAMPLES`), so the model keeps its state once per load and prefills only the
    text it reads (a window of Qwen's newest ids) and a short tail naming the task.
  * Whatever the model writes passes `clean`: one phrase, a word count for its task, no markup,
    no first person, no copy of an example, and every name that looks like a file or a path must
    be in the text it read (the model cannot add a file). Near-duplicates of the recent lines
    are dropped (`similar`). A phrase that fails is not said: the line is then the facts alone
    (a tool line) or nothing (a thinking line; the template comes back only after
    `fallback_s` without any line, so a long thinking is never silent).

NEVER NOISY. Nothing for a short turn: no thinking line before `first_s`, no thought line for a
stretch under `thought_min_s`, no tool line for a call that is done within `tool_after_s` of its
target being known (the client shows it itself then), at most one model line per `min_gap_s`.

LANGUAGE. The user's (`lang_of`, the last user message): German gets German facts and a German
phrase (the task's tail says "German"); every other text English.

QWEN'S TIME. The phrases are written in slices on the request's own thread between two of Qwen's
rounds, the device synchronised first, as v2's (Slot.advance). A request's small-model time is
one `Share` (narrator.py) -- `share` (3 %) of the time since Qwen's first token -- which the voice
draws on first and the shadow reader's in-turn reading after it, so the two together stay inside
the decode gate. A line's phrase is short (at most `max_new` tokens, ~150-250 ms on the GPU).

The lines are state lines (states.py `live`): framed with the narrator's sentinel, stripped from
every prompt (Qwen never reads them), never tokens of the answer, journaled in `extra.states[]`
with what the model was given and how long it took, the request's totals in `extra.voice`.
"""

from __future__ import annotations

import re
import time

# ---------------------------------------------------------------------- the settings
# `ui.live` in a profile overrides any of these
DEFAULTS = {
    "first_s": 15.0,        # the first thinking line after this much thinking
    "every_s": 25.0,        # then one every this many seconds
    "min_gap_s": 12.0,      # never two model lines closer than this
    "fallback_s": 120.0,    # a thinking with no line this long gets the template
    "thought_min_s": 10.0,  # a stretch shorter than this gets no `thought` line
    "tool_after_s": 1.5,    # a call done within this of its target gets no line
    "why_wait_s": 6.0,      # a write's line waits this long for its phrase, then goes plain
    "write_every_s": 30.0,  # a long write's progress line
    "share": 0.03,          # the request's small-model time, of Qwen's since its first token
    "end_ms": 400.0,        # the turn's end allowance for a phrase not done yet
}
BUDGET_MS = 150.0           # one slice at most (the role's latency budget)
MIN_SLICE_MS = 5.0
MAX_NEW = 28                # a phrase's tokens
# the windows of Qwen's ids the model reads, per task
WINDOW = {"now": 640, "decided": 900, "why": 260, "plan": 420}
# a long thought's `decided` window: this many ids of its opening, the rest from its end
DECIDED_HEAD = 300
# a `now` phrase the clean refused is tried again after this long, not at the next token
RETRY_S = 8.0
# new thinking ids a `now` line needs since the last one
NOW_MIN_IDS = 120
# a phrase's words, per task
WORDS = {"now": (3, 18), "decided": (3, 24), "why": (2, 14), "plan": (3, 26)}
TASKS = ("now", "decided", "why", "plan")
LANG_NAMES = {"en": "English", "de": "German"}
MARK = "@@TEXT@@"
TRACE = 12

# ---------------------------------------------------------------------- the prompt
# One head for every task and language: the instructions and nine short worked examples. The
# examples are made up (no user's text); the tail names the task and the language.
SYSTEM = (
    "You write the status line a person reads while an AI coding assistant works. You get a piece "
    "of what the assistant is writing right now (its private reasoning, its answer, or a tool "
    "call) and a task. Answer with one short phrase and nothing else.\n"
    "Tasks:\n"
    "now: what it is working out at this moment, 5 to 14 words, without a subject.\n"
    "decided: what it decided, 5 to 18 words, without a subject.\n"
    "why: what the file or command is for, 3 to 10 words.\n"
    "plan: the plan in one line, 5 to 18 words.\n"
    "Use only what the text says. Never invent a file name. No quotes, no markdown, no full stop. "
    "Write in the language the task names.")
EXAMPLES = (
    ("The deploy hangs at 'Waiting for rollout'. kubectl describe shows the new pod in "
     "CrashLoopBackOff: the readiness probe hits /healthz on 8080, but since the env change the "
     "app listens on 3000. So the probe never passes.",
     "now", "en", "why the new pod never becomes ready after the port change"),
    ("def parse_row(line):\n    parts = line.rstrip('\\n').split(';')\n    if len(parts) != 5:\n"
     "        raise ValueError(f'bad row: {line!r}')\n    day = datetime.strptime(parts[0], "
     "'%d.%m.%Y')\n    amount = Decimal(parts[3].replace(',', '.'))",
     "now", "en", "drafting the CSV row parser for German dates and decimal commas"),
    ("So the export: export.py renders each invoice with WeasyPrint from the Jinja template "
     "invoice.html; the totals come from the Invoice model, never recomputed. Then one test that "
     "renders an invoice and checks the total. The template first.",
     "decided", "en", "WeasyPrint from invoice.html, totals from the model; template first, then a test"),
    ("Options: fix the regex or use html.parser. The regex breaks on nested tags, which is the "
     "bug. html.parser is in the standard library. Use it and keep extract_links' signature so "
     "the callers don't change.",
     "decided", "en", "replace the regex with html.parser and keep the signature"),
    (".card { display: grid; grid-template-columns: 64px 1fr; gap: 12px; }\n.card img { "
     "border-radius: 50%; }\n@media (max-width: 480px) { .card { grid-template-columns: 1fr; } }",
     "now", "en", "drafting the card layout and its narrow-screen version"),
    ("Now the rate limit.\nwrite api/ratelimit.py\n# ratelimit.py: a token bucket per API key, 60 "
     "requests a minute\nimport time",
     "why", "en", "a token bucket that limits each API key to 60 requests a minute"),
    ("Let me check whether the migration ran.\nbash psql -c '\\d invoices'",
     "why", "en", "checks that the migration created the invoices table"),
    ("Plan:\n1. models.py: Invoice and Line\n2. invoice.html: the template\n3. export.py: the "
     "PDF\n4. tests\nThen run them.",
     "plan", "en", "models, the invoice template, the PDF export, then tests"),
    ("The login posts to /api/session but no cookie arrives. The frontend runs on :5173 and the "
     "API on :8080, so the request is cross-site, and the cookie is SameSite=Strict. That is why "
     "the browser drops it.",
     "now", "de", "warum das Login-Cookie bei Anfragen von Port 5173 fehlt"),
    ("App.tsx has 600 lines. I'll split it into Board, Toolbar and Sidebar components and move "
     "the state into a reducer in state.ts; the props stay the same.",
     "decided", "de", "App.tsx in Board, Toolbar und Sidebar teilen, den Zustand in einen Reducer"),
    ("Now the export button.\nwrite web/export.js\n// export.js: saves the current table as CSV, "
     "with a header row and semicolons\nexport function toCsv(rows) {",
     "why", "de", "speichert die Tabelle als CSV mit Kopfzeile"),
    ("const queue = [];\nfunction enqueue(job) {\n  queue.push(job);\n  queue.sort((a, b) => "
     "b.priority - a.priority);\n  if (!running) runNext();\n}",
     "now", "de", "entwirft die Warteschlange, die Aufträge nach Priorität abarbeitet"),
)


def _ask(task: str, lang: str) -> str:
    return f"\n\nTask: {task} ({LANG_NAMES.get(lang, 'English')})"


def prompt(text: str = MARK, task: str = "now", lang: str = "en") -> list[dict]:
    """The model's prompt: the instructions, the examples as turns, the text and the task."""
    msgs = [{"role": "system", "content": SYSTEM}]
    for t, k, lg, a in EXAMPLES:
        msgs += [{"role": "user", "content": "Text:\n" + t + _ask(k, lg)},
                 {"role": "assistant", "content": a}]
    return msgs + [{"role": "user", "content": "Text:\n" + text + _ask(task, lang)}]


# ---------------------------------------------------------------------- the language
_DE = re.compile(r"\b(?:und|nicht|ich|ist|das|die|der|ein|eine|mit|für|bitte|auf|auch|wie|was|"
                 r"mach|erstelle|schreib|kannst|soll|sollte|noch|jetzt|dann|aber|oder|dass|"
                 r"wenn|mir|mein|meine|zu|im|den|dem|es|sie|wir|du|habe|hast|warum|zeig|baue|"
                 r"füge|ändere|spiel|bild)\b", re.I)
_EN = re.compile(r"\b(?:and|not|is|the|a|an|with|for|please|on|also|how|what|make|create|"
                 r"write|can|should|now|then|but|or|that|if|me|my|to|in|it|they|we|you|have|"
                 r"why|show|build|add|change|game|use)\b", re.I)


def lang_of(messages) -> str:
    """`de` when the last user message reads German, else `en`."""
    text = ""
    for m in reversed(messages if isinstance(messages, list) else []):
        if not isinstance(m, dict) or m.get("role") != "user":
            continue
        c = m.get("content")
        if isinstance(c, list):
            c = " ".join(str(p.get("text") or "") for p in c if isinstance(p, dict))
        text = str(c or "")
        if text.strip():
            break
    text = re.sub(r"<[^>]{0,200}>", " ", text)[-4000:]
    if re.search(r"[äöüß]", text, re.I):
        de_bonus = 2
    else:
        de_bonus = 0
    de, en = len(_DE.findall(text)) + de_bonus, len(_EN.findall(text))
    return "de" if de >= 2 and de > en else "en"


# ---------------------------------------------------------------------- the facts in words
def dur(s: float, lang: str = "en") -> str:
    """`47 s`, `4 min` (`4 Min` in German)."""
    s = max(0.0, float(s))
    if s < 90:
        return f"{s:.0f} s"
    return f"{s / 60:.0f} " + ("Min" if lang == "de" else "min")


def num(n: int, lang: str = "en") -> str:
    s = f"{int(n):,}"
    return s.replace(",", ".") if lang == "de" else s


def ordinal(n: int, lang: str = "en") -> str:
    if lang == "de":
        return f"Platz {n}"
    suf = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf} in line"


EFFORT = {"en": {"off": "no thinking", "low": "light thinking", "medium": "medium effort",
                 "high": "high effort", "xhigh": "maximum effort"},
          "de": {"off": "ohne Nachdenken", "low": "kurz nachdenken", "medium": "mittlerer Aufwand",
                 "high": "hoher Aufwand", "xhigh": "maximaler Aufwand"}}


def _effort(word: str | None, lang: str) -> str:
    w = str(word or "").replace(" effort", "").strip()
    if w == "thinking off":
        w = "off"
    return EFFORT[lang].get(w, w or ("default effort" if lang == "en" else "Standardaufwand"))


def _names(names: list[str], lang: str, most: int = 4) -> str:
    names = [n for n in names if n]
    if len(names) <= most:
        if len(names) <= 1:
            return "".join(names)
        last = " and " if lang == "en" else " und "
        return ", ".join(names[:-1]) + last + names[-1]
    more = len(names) - most
    return ", ".join(names[:most]) + (f" and {more} more" if lang == "en" else f" und {more} weitere")


#: the waiting line's reasons for TandemLLM's own work (server/app.py OWN_WAIT), in German
OWN_WAIT_DE = {
    "TandemLLM decides how to handle the request first":
        "TandemLLM entscheidet zuerst, wie es die Anfrage bearbeitet",
    "TandemLLM finishes its background work first": "TandemLLM beendet zuerst Hintergrundarbeit",
    "TandemLLM finishes a summary it prepared first":
        "TandemLLM beendet zuerst eine vorbereitete Zusammenfassung",
    "TandemLLM finishes its own work first": "TandemLLM beendet zuerst eigene Arbeit",
}


def natural(kind: str, facts: dict, lang: str = "en") -> str | None:
    """A harness line (states.TEXTS) in plain words and the user's language; None: the template.
    `facts` are the template's facts plus the voice's (`label`, `effort_word`, `names`, ...)."""
    de = lang == "de"
    f = facts or {}
    if kind == "route":
        label = f.get("label") or str(f.get("cls") or "").replace("_", " ")
        if not de and len(label) > 1 and label[0].isupper() and label[1:2].islower():
            label = label[0].lower() + label[1:]
        return f"{label}, {_effort(f.get('effort'), lang)}"
    if kind == "package":
        names = f.get("names") or [f.get("name")]
        return ("Pakete: " if de else "packages ") + _names([str(n) for n in names], lang) \
            if len(names) > 1 else (("lädt " if de else "loading ") + f"{f.get('pkind') or 'skill'} "
                                    f"{names[0]}")
    if kind in ("queue", "queue_eta"):
        n = int(str(f.get("n") or 1).replace(",", ""))
        reason = str(f.get("reason") or "")
        eta = str(f.get("eta") or "") if kind == "queue_eta" else ""
        tail = ((f", noch etwa {eta.replace('min', 'Min')}" if de else f", about {eta} left")
                if eta else "")
        if reason.startswith("another request"):
            return (f"wartet auf eine andere Anfrage, {ordinal(n, lang)}{tail}" if de
                    else f"waiting for another request to finish, {ordinal(n, lang)}{tail}")
        if de and reason in OWN_WAIT_DE:
            return f"wartet: {OWN_WAIT_DE[reason]}, {ordinal(n, lang)}{tail}"
        return (f"wartet: {reason}, {ordinal(n, lang)}{tail}" if de
                else f"waiting while {reason.removesuffix(' first')}, "
                     f"{ordinal(n, lang)}{tail}")
    if kind == "prefill":
        n = str(f.get("n") or "")
        left = str(f.get("left") or "")
        if de:
            return f"liest {n.replace(',', '.')} Tokens, noch etwa {left.replace('min', 'Min')}"
        return f"reading {n} tokens, about {left} left"
    if kind == "thinking":
        return (f"denkt seit {f.get('elapsed_de') or f.get('elapsed')}, {f.get('n')} Tokens" if de
                else f"still thinking, {f.get('elapsed')} so far ({f.get('n')} tokens)")
    if kind == "thought":
        return (f"hat {f.get('elapsed_de') or f.get('elapsed')} nachgedacht ({f.get('n')} Tokens)"
                if de else f"thought for {f.get('elapsed')} ({f.get('n')} tokens)")
    if kind == "compaction":
        return ("das Gespräch wurde zusammengefasst" if de
                else "the conversation was compacted into a summary")
    if kind == "memory":
        return f"liest {f.get('n')} Notizen" if de else f"reading {f.get('n')} notes from memory"
    if kind == "steer":
        return (f"lenkt mit einem Hinweis: {f.get('label')}" if de
                else f"steering with a note: {f.get('label')}")
    if kind == "cap":
        return (f"Denkbudget erreicht ({f.get('cap')} Tokens), schreibt jetzt die Antwort" if de
                else f"thinking budget reached at {f.get('cap')} tokens, answering now")
    if kind == "backstop":
        return (f"Denken bei {f.get('cap')} Tokens beendet" if de
                else f"thinking stopped at the {f.get('cap')}-token backstop")
    if kind == "loop":
        return ("das Denken wiederholt sich, lenkt um" if de
                else "the thinking went in circles, steering it out")
    if kind == "close":
        return "beendet das Denken" if de else "closing the thinking"
    if kind == "continue" and f.get("lift"):
        # SRV-199: E1's lift, the calls written inside the thinking sent as the answer
        what = _names([str(t) for t in f.get("tools") or []], lang)
        return (f"der {what + '-' if what else 'Tool-'}Aufruf stand im Denken, wird gesendet" if de
                else f"the {what + ' ' if what else 'tool '}call was written in the thinking, "
                     f"sending it")
    if kind == "continue":
        return ("keine Antwort, macht weiter" if de
                else "stopped without an answer, asking it to go on")
    if kind in ("check", "checks"):
        tools = f.get("tools") or ([f.get("tool")] if f.get("tool") else [])
        what = _names([str(t) for t in tools], lang) if tools else ""
        if de:
            return f"prüft {what + '-' if what else 'den Tool-'}Aufruf" + ("e" if kind == "checks" else "")
        return f"checking the {what + ' ' if what else 'tool '}call" + ("s" if kind == "checks" else "")
    if kind == "repair":
        return "repariert den Tool-Aufruf" if de else "repairing the tool call"
    if kind == "regenerate":
        return "lässt den Tool-Aufruf neu schreiben" if de else "asking for the tool call again"
    if kind == "give_up":
        return "gibt auf" if de else "giving up"
    return None


# ---------------------------------------------------------------------- tools
# the parameter that names a call's target, in order of preference
TARGET_KEYS = ("filePath", "file_path", "path", "filename", "file", "command", "cmd", "pattern",
               "query", "url", "uri", "description", "prompt")
# parameters whose value is a body being written (lines are counted)
BODY_KEYS = ("content", "newString", "new_string", "contents", "text", "patch", "patchText",
             "code", "body")
VERBS = {
    "en": {"write": "writing", "edit": "editing", "multiedit": "editing", "patch": "patching",
           "apply_patch": "patching", "read": "reading", "bash": "running", "shell": "running",
           "grep": "searching for", "glob": "looking for", "search": "searching for",
           "list": "listing", "ls": "listing", "webfetch": "fetching", "fetch": "fetching",
           "websearch": "searching the web for", "task": "handing off", "todowrite": "updating the to-do list",
           "todoread": "reading the to-do list"},
    "de": {"write": "schreibt", "edit": "ändert", "multiedit": "ändert", "patch": "ändert",
           "apply_patch": "ändert", "read": "liest", "bash": "führt aus:", "shell": "führt aus:",
           "grep": "sucht nach", "glob": "sucht", "search": "sucht nach", "list": "listet",
           "ls": "listet", "webfetch": "lädt", "fetch": "lädt", "websearch": "sucht im Web nach",
           "task": "gibt ab:", "todowrite": "aktualisiert die Aufgabenliste",
           "todoread": "liest die Aufgabenliste"},
}
# browser tools (chrome-devtools_*, playwright_*): the action after the prefix
BROWSER = {
    "en": {"new_page": "opening a page in the browser", "navigate_page": "loading the page",
           "take_screenshot": "taking a screenshot", "take_snapshot": "reading the page",
           "evaluate_script": "running a script in the page",
           "list_console_messages": "reading the browser console",
           "list_network_requests": "reading the network requests", "click": "clicking",
           "fill": "filling a form", "press_key": "pressing a key", "type_text": "typing"},
    "de": {"new_page": "öffnet eine Seite im Browser", "navigate_page": "lädt die Seite",
           "take_screenshot": "macht einen Screenshot", "take_snapshot": "liest die Seite",
           "evaluate_script": "führt ein Skript in der Seite aus",
           "list_console_messages": "liest die Browser-Konsole",
           "list_network_requests": "liest die Netzwerkanfragen", "click": "klickt",
           "fill": "füllt ein Formular", "press_key": "drückt eine Taste", "type_text": "tippt"},
}
WRITE_TOOLS = ("write", "edit", "multiedit", "patch", "apply_patch")
TARGET_CHARS = 64


def short_path(p: str) -> str:
    """The last two parts of a path (`mario/game.js`), a URL without its scheme."""
    p = str(p).strip().strip("\"'")
    if "://" in p:
        p = p.split("://", 1)[1]
        return p if len(p) <= TARGET_CHARS else p[:TARGET_CHARS - 1] + "…"
    parts = [x for x in p.replace("\\", "/").split("/") if x]
    s = "/".join(parts[-2:]) if len(parts) >= 2 else (parts[0] if parts else p)
    return s if len(s) <= TARGET_CHARS else "…" + s[-(TARGET_CHARS - 1):]


def short_command(c: str) -> str:
    """A command's first line, at most TARGET_CHARS characters."""
    first = next((ln.strip() for ln in str(c).splitlines() if ln.strip()), "")
    more = first != str(c).strip()
    if len(first) > TARGET_CHARS:
        return first[:TARGET_CHARS - 1] + "…"
    return first + (" …" if more else "")


def tool_action(name: str, args: dict, lang: str = "en") -> tuple[str, str | None]:
    """(the action in words, the target shown) from a call's name and arguments: facts only."""
    n = str(name or "")
    low = n.lower()
    base = low.split("_", 1)[1] if low.startswith(("chrome-devtools_", "playwright_")) else None
    if base is not None:
        b = base.replace("browser_", "")
        act = BROWSER[lang].get(b) or (("Browser: " if lang == "de" else "browser: ") + b.replace("_", " "))
        url = args.get("url")
        return act, (short_path(url) if url else None)
    verb = VERBS[lang].get(low)
    target = None
    for k in TARGET_KEYS:
        v = args.get(k)
        if isinstance(v, str) and v.strip():
            if k in ("command", "cmd"):
                target = short_command(v)
            elif k in ("pattern", "query", "description", "prompt"):
                target = short_command(v)
            else:
                target = short_path(v)
            break
    if verb is None:
        verb = (f"ruft {n} auf" if lang == "de" else f"calling {n}") if n else "…"
        return verb, target
    return verb, target


def tool_line(verb: str, target: str | None, lang: str) -> str:
    if lang == "de" and verb.endswith(":"):
        return f"{verb} {target}" if target else verb[:-1]
    return f"{verb} {target}" if target else verb


_EXT = (r"(?:js|mjs|cjs|ts|tsx|jsx|py|html|css|scss|json|md|go|rs|java|c|h|cpp|sh|yaml|yml|toml|"
        r"sql|svelte|vue|glsl|txt)")
_FILE = re.compile(r"\b[\w\-]+\." + _EXT + r"\b")
# a header comment that opens with a file name: `// entities.js: ...`, `# tools/run.py`, `/* a.css */`
_HEADER = re.compile(r"^\s*(?://+|#+|/\*+|<!--|\*|--)\s*((?:[\w.\-]+/)*[\w\-]+\." + _EXT + r")\b")


def named_in_body(body: str, before: str) -> str | None:
    """A write whose path comes after its body (opencode's `write`): the file its body names
    itself -- a header comment that opens with a file name (`// entities.js: the enemies`), or a
    first line whose one file name Qwen's words just before the call name too ("Next up are
    entities.js and game.js"). None when neither holds."""
    first = next((ln for ln in body.splitlines() if ln.strip()), "")[:200]
    names = {x.group(0) for x in _FILE.finditer(first)}
    m = _HEADER.match(first)
    if m is not None and len(names) == 1:
        return m.group(1)
    if len(names) != 1:
        return None
    name = names.pop()
    return name if re.search(r"(?<![\w.])" + re.escape(name) + r"\b", before) else None


class ToolWatch:
    """The open tool-call block as Qwen writes it (the buffer's `open_text()`), read incrementally:
    each function's name, its parameters' values once closed, the lines of a body so far. The
    text is the block after `<tool_call>`, in the parameter-XML format (server/toolcall.py)."""

    _FUNC = re.compile(r'<function(?:\s*=\s*([^>\s]+)|\s+name\s*=\s*"([^"]*)")\s*>')
    _PARAM = re.compile(r'<parameter(?:\s*=\s*([^>\s]+)|\s+name\s*=\s*"([^"]*)")\s*>')

    def __init__(self):
        self.calls: list[dict] = []      # {name, args, body_key, lines, t_name, t_target, ...}
        self._text = ""
        self._pos = 0
        self._in: str | None = None      # the open parameter's key
        self._vstart = 0
        self.open = False

    @property
    def current(self) -> dict | None:
        return self.calls[-1] if self.calls and not self.calls[-1].get("closed") else None

    def feed(self, text: str | None, now: float) -> None:
        if text is None:
            if self.open:
                self.open = False
                for c in self.calls:
                    if not c.get("closed"):
                        c["closed"], c["t_closed"] = True, now
            self._text, self._pos, self._in = "", 0, None
            return
        if not self.open or len(text) < len(self._text):
            self.open, self._pos, self._in = True, 0, None
        self._text = text
        while True:
            c = self.current
            if self._in is not None:
                end = text.find("</parameter>", self._pos)
                if end < 0:
                    if c is not None and self._in in BODY_KEYS:
                        safe = max(self._pos, len(text) - 12)
                        c["lines"] += text.count("\n", self._pos, safe)
                        self._pos = safe
                        c["body"] = text[self._vstart:self._vstart + 2000]
                    return
                val = text[self._vstart:end].strip("\n")
                if c is not None:
                    if self._in in BODY_KEYS:
                        c["lines"] += text.count("\n", self._pos, end)
                        c["body"] = val[:2000]
                    else:
                        c["args"][self._in] = val
                        if c.get("t_target") is None and self._in in TARGET_KEYS:
                            c["t_target"] = now
                self._in, self._pos = None, end + len("</parameter>")
                continue
            if c is None:
                m = self._FUNC.search(text, self._pos)
                if m is None:
                    return
                self.calls.append({"name": m.group(1) or m.group(2), "args": {}, "lines": 0,
                                   "body": "", "t_name": now, "t_target": None, "closed": False,
                                   "said": 0, "t_said": None})
                self._pos = m.end()
                continue
            fclose = text.find("</function>", self._pos)
            m = self._PARAM.search(text, self._pos)
            if fclose >= 0 and (m is None or fclose < m.start()):
                c["closed"], c["t_closed"] = True, now
                self._pos = fclose + len("</function>")
                continue
            if m is None:
                return
            self._in = m.group(1) or m.group(2)
            self._pos = self._vstart = m.end()
            if text.startswith("\n", self._pos):
                self._pos = self._vstart = self._pos + 1
            if self._in in BODY_KEYS and c.get("t_body") is None:
                c["t_body"] = now
                c["body_key"] = self._in


# ---------------------------------------------------------------------- the clean
_BAD = re.compile(r"[›⁠<>\n*`#]|\|im_|</?think|^(?:task|text|now|decided|why|plan)\s*:",
                  re.I)
_FIRST_PERSON = re.compile(r"^(?:i|i'm|i'll|i've|let me|let's|we|ich|wir)\b", re.I)
_NAMEY = re.compile(r"[\w\-]+(?:/[\w.\-]+)+|\b[\w\-]+\.(?:[a-z]{1,5})\b", re.I)
_EXAMPLE_ANSWERS = {a.lower() for *_, a in EXAMPLES}
# words a phrase may start with in lower case (a capital there is the model's habit, not a name)
_LOWER_STARTS = {"how", "why", "what", "where", "when", "which", "whether", "the", "a", "an",
                 "its", "their", "five", "four", "three", "two", "one", "six", "seven", "all",
                 "each", "every", "both", "more", "less", "use", "add", "fix", "split", "move",
                 "keep", "replace", "write", "check", "test", "run", "make", "build", "draw",
                 "load", "read", "serve", "open", "start", "first", "then", "next", "only",
                 "no", "new", "small", "simple", "separate", "single", "same", "whole",
                 "fixed", "modern", "vector", "canvas", "game", "player", "level", "enemy",
                 "enemies", "physics", "collision", "rendering", "render", "files", "file"}


def clean(text: str, task: str, source: str, lang: str = "en") -> tuple[str | None, str | None]:
    """(the phrase, None) or (None, why not)."""
    s = next((ln for ln in (text or "").strip().split("\n") if ln.strip()), "").strip()
    s = re.sub(r"^(?:now|decided|why|plan|line|status)\s*[:\-]\s*", "", s, flags=re.I)
    s = " ".join(s.strip("\"'`*_ ").split()).rstrip(".;:,").strip()
    if not s:
        return None, "empty"
    if _BAD.search(s):
        return None, "markup"
    if _FIRST_PERSON.match(s):
        return None, "first_person"
    if s.lower() in _EXAMPLE_ANSWERS or any(similar(s, a) for a in _EXAMPLE_ANSWERS):
        return None, "example"
    lo, hi = WORDS.get(task, (2, 20))
    n = len(s.split())
    if n < lo:
        return None, "short"
    if n > hi:
        return None, "words"
    if lang != "de" and not grounded(s, source):
        return None, "ungrounded"
    if lang == "de" and _english(s):
        return None, "language"
    if lang != "de" and _german(s):
        return None, "language"
    if not numbers_ok(s, source):
        return None, "unsupported_number"
    src = source.lower()
    for m in _NAMEY.finditer(s):
        name = m.group(0).lower().strip(".")
        if name and name not in src and name.split("/")[-1] not in src:
            return None, "unsupported_name"
    if task == "why":
        # a purpose, not a restatement: "to draw the stars" -> "draws the stars" is the model's
        # job; here only the lead words that make it a sentence about the call go
        s = re.sub(r"^(?:to|for|in order to)\s+", "", s, flags=re.I)
        if re.match(r"^(?:why|writes?|writing|creates? the (?:file|html file))\b", s, re.I) \
                or re.search(r"\b(?:the|this|that) (?:\w+ )?file$", s, re.I) \
                or re.search(r"\b[\w\-]+\.(?:js|ts|tsx|jsx|py|html|css|json|md|go|rs)\b", s):
            # the call's target is the line's fact already: a phrase naming a file restates it
            return None, "restates"
        if len(s.split()) < lo:
            return None, "short"
    first = s.split()[0]
    if lang != "de" and first[0].isupper() and first[1:].islower() and (
            first.lower() in _LOWER_STARTS or first.lower() in _GERUND
            or first.lower().endswith("ing")
            or re.search(r"\b" + re.escape(first.lower()) + r"\b", source)):
        s = s[0].lower() + s[1:]
    return s, None


# words a phrase may use that its text need not contain
_GENERIC = {"drafting", "draft", "writing", "write", "code", "checks", "check", "checking",
            "what", "which", "where", "when", "whether", "that", "this", "these", "those",
            "with", "from", "into", "then", "first", "after", "before", "about", "their",
            "them", "they", "make", "makes", "using", "uses", "should", "would", "could",
            "plan", "planning", "decides", "deciding", "working", "works", "out", "version",
            "files", "file", "function", "functions", "logic", "part", "parts", "each", "every",
            "both", "only", "also", "more", "less", "same", "other", "between", "while",
            "still", "does", "doesn't", "isn't", "never", "always", "again", "next", "later",
            "test", "tests", "runs", "run", "fixes", "fix", "adds", "add", "keeps", "keep",
            "handling", "handles", "handle", "setting", "sets", "builds", "build", "goes"}


def _stem(w: str) -> str:
    return w.lower()[:5]


def grounded(phrase: str, source: str) -> bool:
    """Most of the phrase's content words are in the text it read (English phrases): a phrase
    from an example's words, or made up, is not."""
    words = [w for w in re.findall(r"[A-Za-z][A-Za-z'\-]+", phrase)
             if len(w) >= 4 and w.lower() not in _GENERIC]
    if not words:
        return True
    # identifiers count by their parts: createRadialGradient, draw_stars
    split = re.sub(r"([a-z])([A-Z])", r"\1 \2", source).replace("_", " ")
    src = {_stem(w) for w in re.findall(r"[A-Za-z][A-Za-z'\-]+", split)}
    hit = sum(1 for w in words if _stem(w) in src)
    return hit / len(words) >= 0.5


_EN_WORDS = {"the", "and", "with", "for", "of", "to", "is", "are", "in", "on", "from", "by",
             "into", "its", "how", "what", "why", "when", "which", "that", "this", "fix", "add",
             "write", "build", "check", "using", "while"}
_DE_WORDS = {"der", "die", "das", "und", "mit", "für", "von", "zu", "ist", "sind", "im", "in",
             "auf", "aus", "den", "dem", "ein", "eine", "einen", "wie", "was", "warum", "wann",
             "welche", "nach", "bei", "oder", "als", "dass", "sich", "noch"}


def _english(phrase: str) -> bool:
    """A phrase that should be German reads English: its function words are English ones."""
    ws = [w.lower() for w in re.findall(r"[A-Za-zÄÖÜäöüß]+", phrase)]
    en, de = sum(w in _EN_WORDS for w in ws), sum(w in _DE_WORDS for w in ws)
    return en >= 2 and en > de


def _german(phrase: str) -> bool:
    """A phrase that should be English reads German (0.3.28's smoke: "snake.html erstellt, Tests
    mit Laya und Screenshots" for an English user)."""
    ws = [w.lower() for w in re.findall(r"[A-Za-zÄÖÜäöüß]+", phrase)]
    en, de = sum(w in _EN_WORDS for w in ws), sum(w in _DE_WORDS for w in ws)
    return (de >= 2 and de > en) or bool(re.search(r"[äöüß]", phrase) and de >= 1 and de >= en)


def numbers_ok(phrase: str, source: str) -> bool:
    """Every number in the phrase is in the text it read."""
    return all(n in source for n in re.findall(r"\d+(?:[.,]\d+)?", phrase))


_GERUND = {"write": "writing", "fix": "fixing", "design": "designing", "draw": "drawing",
           "check": "checking", "create": "creating", "generate": "generating",
           "debug": "debugging", "find": "finding", "add": "adding", "build": "building",
           "use": "using", "split": "splitting", "make": "making", "test": "testing",
           "run": "running", "update": "updating", "implement": "implementing",
           "handle": "handling", "move": "moving", "set": "setting", "plan": "planning",
           "compute": "computing", "calculate": "calculating", "define": "defining",
           "decide": "deciding", "verify": "verifying", "render": "rendering",
           "refactor": "refactoring", "replace": "replacing", "rewrite": "rewriting",
           "remove": "removing", "rename": "renaming", "read": "reading", "parse": "parsing",
           "draft": "drafting", "outline": "outlining", "place": "placing", "tune": "tuning",
           "adjust": "adjusting", "align": "aligning", "compare": "comparing",
           "store": "storing", "load": "loading", "map": "mapping", "position": "positioning"}


def gerund(phrase: str, lang: str = "en") -> str:
    """`write the level` -> `writing the level`: a thinking line says what is going on."""
    if lang != "en" or not phrase:
        return phrase
    first, _, rest = phrase.partition(" ")
    w = first.lower()
    # the third person too: "drafts the game" -> "drafting the game"
    g = _GERUND.get(w) or (_GERUND.get(w[:-1]) if w.endswith("s") else None) \
        or (_GERUND.get(w[:-2]) if w.endswith("es") else None)
    return (g + (" " + rest if rest else "")) if g else phrase


def _words(s: str) -> set[str]:
    return {w for w in re.findall(r"[\w']+", s.lower()) if len(w) > 2}


def similar(a: str, b: str) -> bool:
    """Near-identical phrases: the same words, mostly."""
    x, y = _words(a), _words(b)
    if not x or not y:
        return a.strip().lower() == b.strip().lower()
    return len(x & y) / len(x | y) >= 0.6


# ---------------------------------------------------------------------- the plan in the answer
_PLAN_HEAD = re.compile(r"(?im)^\s*(?:#+\s*|\*\*)?(?:plan|steps|schritte|vorgehen)\b")
_ITEM = re.compile(r"(?m)^\s*(?:\d+[.)]|[-*•])\s+\S")


def plan_in(text: str) -> bool:
    """A plan: a heading that says so with two items, or three numbered items."""
    items = len(_ITEM.findall(text))
    numbered = len(re.findall(r"(?m)^\s*\d+[.)]\s+\S", text))
    return (bool(_PLAN_HEAD.search(text)) and items >= 2) or numbered >= 3


# ---------------------------------------------------------------------- one request
class Voice:
    """One streamed request's live lines. The token loop calls `token` after each of Qwen's
    tokens, `content` with the answer's visible text, `tool` with the open block (or None), and
    `finish` at the end; lines go out through `ui.live(...)` (states.py)."""

    def __init__(self, narrator, ui, *, lang: str = "en", opened: bool = False,
                 close_id: int | None = None, decode=None, share=None, settings: dict | None = None,
                 clock=time.perf_counter, log=print):
        from server.conductor.narrator import Share
        self.narrator, self.ui, self.lang = narrator, ui, lang if lang in LANG_NAMES else "en"
        self.slot = narrator.slot
        self.close_id, self.decode = close_id, decode
        self.cfg = dict(DEFAULTS, **{k: float(v) for k, v in (settings or {}).items()
                                     if k in DEFAULTS and isinstance(v, (int, float))})
        self.share = share if share is not None else Share(self.cfg["share"], clock=clock)
        self.clock, self.log = clock, log
        self.t0: float | None = None
        # the open stretch of thinking: its first id, its start, the last line said in it
        self.start: int | None = 0 if opened else None
        self.t_start: float | None = None
        self.t_line: float | None = None         # the last line of any kind (the gap)
        self.t_model: float | None = None        # the last model line
        self.now_at = 0                          # ids when the last `now` began
        self.t_try: float | None = None          # when it began (a refused phrase waits RETRY_S)
        self.job: dict | None = None
        self.tools = ToolWatch()
        self.block_at: int | None = None         # ids where the open block began
        self.answer = ""                         # the answer's visible text (a cap)
        self.plan_said = False
        self.said: list[str] = []
        self.trace: list[dict] = []              # the newest jobs' phrases or why not
        self.stretch_phrases: list[str] = []     # the open stretch's `now` phrases
        self.stats = {"lines": 0, "phrases": 0, "plain": 0, "dropped": 0, "ms": 0.0,
                      "end_ms": 0.0, "why": {}}
        self.broken: str | None = None

    # ------------------------------------------------------------------ from the token loop
    def token(self, ids: list[int], in_think: bool) -> None:
        if self.broken:
            return
        try:
            now = self.clock()
            if self.t0 is None:
                self.t0 = now
                self.share.start(now)
            if in_think:
                if self.start is None:
                    self.start, self.t_start = len(ids), now
                elif self.t_start is None:
                    self.t_start = now
                self._thinking(ids, now)
            elif self.start is not None:
                self._closed(ids, now)
            self._work(now)
        except Exception as exc:                                  # noqa: BLE001
            self._broke(exc)

    def content(self, text: str) -> None:
        """The answer's visible text as it streams (not the reasoning, not a tool block)."""
        if self.broken or not text or self.plan_said:
            return
        if len(self.answer) < 6000:
            self.answer += text
        try:
            if "\n\n" in text and plan_in(self.answer):
                tail = self.answer.rstrip().rsplit("\n", 1)[-1]
                if not _ITEM.match(tail):
                    self._plan(self.clock())
        except Exception as exc:                                  # noqa: BLE001
            self._broke(exc)

    def tool(self, ids: list[int], open_text: str | None) -> None:
        """The tool-call buffer's open block (None when none is open), after each token."""
        if self.broken:
            return
        try:
            now = self.clock()
            if open_text is not None and self.block_at is None:
                self.block_at = len(ids)
                if not self.plan_said and plan_in(self.answer):
                    self._plan(now)
            if open_text is None:
                self.block_at = None
            before = [c for c in self.tools.calls if not c.get("closed")]
            self.tools.feed(open_text, now)
            for c in before:
                if c.get("closed"):
                    self._call_closed(c, now)
            c = self.tools.current
            if c is not None:
                self._call_open(c, ids, now)
        except Exception as exc:                                  # noqa: BLE001
            self._broke(exc)

    def finish(self, ids: list[int]) -> None:
        """The turn's end: an open stretch ends, a plan not said yet is read, and a phrase not done
        gets the end allowance."""
        if self.broken:
            return
        try:
            now = self.clock()
            if self.t0 is None:
                self.t0 = now
                self.share.start(now)
            if self.start is not None:
                self._closed(ids, now)
            self.tools.feed(None, now)
            for c in self.tools.calls:
                if c.get("said") == 0 and c.get("waiting"):
                    self._say_call(c, None, {"why": "turn_end"})
            if self.job is None and not self.plan_said and plan_in(self.answer):
                self._plan(now)
            t_end = now + self.cfg["end_ms"] / 1000.0
            while self.job is not None:
                left = (t_end - self.clock()) * 1e3
                if left < MIN_SLICE_MS:
                    self._done("time")
                    break
                ms = self._slice(min(BUDGET_MS, left))
                self.stats["end_ms"] += ms
                if ms <= 0 and self.job is not None:
                    self._done("time")
        except Exception as exc:                                  # noqa: BLE001
            self._broke(exc)

    def totals(self) -> dict:
        return dict(self.stats, ms=round(self.stats["ms"], 2),
                    end_ms=round(self.stats["end_ms"], 2), lang=self.lang,
                    share_ms=round(self.share.spent, 2),
                    **({"broken": self.broken} if self.broken else {}))

    # ------------------------------------------------------------------ thinking
    def _thinking(self, ids: list[int], now: float) -> None:
        el = now - (self.t_start if self.t_start is not None else now)
        last = self.t_line if self.t_line is not None and self.t_line >= (self.t_start or 0) \
            else None
        due = (el >= self.cfg["first_s"] if last is None
               else now - last >= self.cfg["every_s"])
        if due and self.job is None and len(ids) - max(self.start or 0, self.now_at) >= NOW_MIN_IDS \
                and self._gap_ok(now) and (self.t_try is None or now - self.t_try >= RETRY_S):
            lo = max(self.start or 0, len(ids) - WINDOW["now"])
            self.now_at, self.t_try = len(ids), now
            self._begin("now", self._ids(ids[lo:]), {"elapsed": el})
        # never silent: the template's line when nothing was said for `fallback_s`
        quiet_since = last if last is not None else self.t_start
        if quiet_since is not None and now - quiet_since >= self.cfg["fallback_s"] \
                and self.job is None:
            self.ui.live("thinking", natural("thinking", {
                "elapsed": dur(el), "elapsed_de": dur(el, "de"),
                "n": num(len(ids) - (self.start or 0), self.lang)}, self.lang),
                {"why": "fallback"}, n=f"{len(ids) - (self.start or 0):,}", elapsed=dur(el))
            self._said("", now, model=False)

    def _closed(self, ids: list[int], now: float) -> None:
        start, self.start = self.start, None
        t_start = self.t_start if self.t_start is not None else (self.t0 or now)
        self.t_start = None
        cut = len(ids)
        if self.close_id is not None:
            tail = ids[start:cut]
            if self.close_id in tail:
                cut = start + len(tail) - 1 - tail[::-1].index(self.close_id)
        el = now - t_start
        if self.job is not None and self.job["task"] == "now":
            self._drop("superseded")
        if el < self.cfg["thought_min_s"] or cut - start < 32:
            return
        n = cut - start
        steps = [p for p in self.stretch_phrases if p]
        self.stretch_phrases = []
        if n <= WINDOW["decided"]:
            body = self._ids(ids[start:cut])
        elif steps:
            # a long thought that had its lines: what it went through (its own lines, in
            # order), then its end, where it lands
            body = (self._enc("Earlier in this reasoning: " + "; ".join(steps[-8:]) + ".\n[…]\n")
                    + self._ids(ids[cut - (WINDOW["decided"] - DECIDED_HEAD):cut]))
        else:
            # a long thought: its opening (where the plan forms) and its end (where it lands)
            head = DECIDED_HEAD
            body = (self._ids(ids[start:start + head]) + self._enc("\n[…]\n")
                    + self._ids(ids[cut - (WINDOW["decided"] - head):cut]))
        self._begin("decided", body, {"elapsed": el, "n": n})

    # ------------------------------------------------------------------ tools
    def _known(self, c: dict) -> float | None:
        """When the call's line could first say something true: its target (a path, a command)
        is known, or, for a write whose body comes first (opencode's `write`: content before
        filePath), its body began."""
        if c.get("t_target") is not None:
            return c["t_target"]
        if str(c["name"] or "").lower() in WRITE_TOOLS:
            return c.get("t_body")
        return None

    def _call_open(self, c: dict, ids: list[int], now: float) -> None:
        known = self._known(c)
        if known is None or now - known < self.cfg["tool_after_s"]:
            return
        name = str(c["name"] or "").lower()
        if name not in WRITE_TOOLS:
            if not c.get("said"):
                self._say_call(c, None, {"why": None})
            return
        if not c.get("asked"):
            # a write: its phrase reads the words just before the block and the body's start
            body = c.get("body") or ""
            if len(body) >= 240 or now - known >= 3.0:
                c["asked"], c["waiting"] = True, now
                if c["args"].get("filePath") is None and c["args"].get("path") is None:
                    c["named"] = named_in_body(body, self._before(ids))
                if self.job is None and self._room():
                    target = (c["args"].get("filePath") or c["args"].get("path")
                              or c.get("named") or "")
                    call = f"{c['name']} {target}\n{body[:700]}"
                    self._begin("why", self._enc(call), {"call": c})
                else:
                    self._say_call(c, None, {"why": "busy" if self.job is not None else "share"})
            return
        if not c.get("said"):
            if now - c["waiting"] >= self.cfg["why_wait_s"]:
                if self.job is not None and self.job.get("call") is c:
                    self._drop("why_wait")
                self._say_call(c, None, {"why": "why_wait"})
            return
        # a long write: its progress
        if c.get("t_said") is not None and now - c["t_said"] >= self.cfg["write_every_s"] \
                and c.get("lines", 0) >= 20:
            verb, target = tool_action(c["name"], c["args"], self.lang)
            target = target or c.get("named") or ("eine Datei" if self.lang == "de" else "a file")
            n = num(c["lines"], self.lang)
            text = (f"{tool_line(verb, target, 'de')}, bisher {n} Zeilen" if self.lang == "de"
                    else f"{tool_line(verb, target, 'en')}, {n} lines so far")
            self._emit("tool", text, {"progress": True, "lines": c["lines"]}, now, model=False)
            c["t_said"] = now
            c["said"] += 1

    def _before(self, ids: list[int]) -> str:
        """Qwen's words just before the open block (the answer or the reasoning)."""
        at = self.block_at or len(ids)
        try:
            return self._source(self._ids(ids[max(0, at - 80):at]))
        except Exception:                                          # noqa: BLE001
            return ""

    def _call_closed(self, c: dict, now: float) -> None:
        """A call that ended before its line: said plain when it was a long one, else nothing."""
        if c.get("said"):
            return
        known = self._known(c)
        if known is not None and now - known >= self.cfg["tool_after_s"]:
            if self.job is not None and self.job.get("call") is c:
                return                          # its phrase is coming; `_done` says it
            self._say_call(c, None, {"why": "closed"})

    def _say_call(self, c: dict, phrase: str | None, info: dict) -> None:
        if c.get("said"):
            return
        verb, target = tool_action(c["name"], c["args"], self.lang)
        if target is None and str(c["name"] or "").lower() in WRITE_TOOLS:
            target = c.get("named") or ("eine Datei" if self.lang == "de" else "a file")
        text = tool_line(verb, target, self.lang)
        if phrase:
            text += ": " + phrase
        now = self.clock()
        self._emit("tool", text, dict(info, tool=c["name"]), now, model=bool(phrase),
                   phrase=phrase)
        c["said"], c["t_said"] = 1, now

    def _plan(self, now: float) -> None:
        self.plan_said = True
        if self.job is not None or not self._gap_ok(now) or not self._room():
            return
        text = self.answer[-2400:]
        self._begin("plan", self._enc(text), {})

    # ------------------------------------------------------------------ the model's jobs
    def _ids(self, ids: list[int]) -> list[int]:
        return list(self.slot.model.text_ids(ids, self.decode))

    def _enc(self, text: str) -> list[int]:
        return list(self.slot.model.encode_text(text))

    def _room(self) -> bool:
        return self.share.allow(self.clock()) >= MIN_SLICE_MS

    def _gap_ok(self, now: float) -> bool:
        return self.t_model is None or now - self.t_model >= self.cfg["min_gap_s"]

    def _begin(self, task: str, body: list[int], facts: dict) -> None:
        if self.job is not None:
            self._drop("superseded")
        why = self.slot.ensure()
        base = {"task": task, "read": len(body)}
        if why is not None:
            self.slot.calls[why] += 1
            return self._failed(task, facts, dict(base, why=why))
        if not body:
            return self._failed(task, facts, dict(base, why="empty_text"))
        head, tail = self.narrator.voice_around(task, self.lang)
        body = body[-(WINDOW[task] + 16):]
        r = self.slot.begin(head + body + tail, max_new=MAX_NEW, keep=len(head))
        if r.get("why"):
            return self._failed(task, facts, dict(base, why=r["why"],
                                                  **({"error": r["error"]} if r.get("error")
                                                     else {})))
        self.job = {"job": r["job"], "task": task, "facts": facts, "base": base,
                    "t": self.clock(), "call": facts.get("call"),
                    "source": self._source(body)}

    def _source(self, body: list[int]) -> str:
        m = self.slot.model
        try:
            return (getattr(m, "text_of", None) or m.decode)(body)
        except Exception:                                          # noqa: BLE001
            return ""

    def _work(self, now: float) -> None:
        while self.job is not None:
            allow = self.share.allow(now)
            if allow < MIN_SLICE_MS:
                return
            ms = self._slice(min(BUDGET_MS, allow))
            if ms <= 0 and self.job is not None:
                return
            now = self.clock()

    def _slice(self, budget_ms: float) -> float:
        job = self.job["job"]
        r = self.slot.advance(job, budget_ms=budget_ms)
        ms = float(r.get("ms") or 0.0)
        self.stats["ms"] += ms
        self.share.spend(ms)
        if r.get("why"):
            self._done(r["why"], r.get("error"))
        elif r.get("done"):
            self._done(None)
        return ms

    def _drop(self, why: str) -> None:
        j, self.job = self.job, None
        if j is None:
            return
        self.stats["dropped"] += 1
        self._why(why)
        c = j.get("call")
        if c is not None and not c.get("said") and why != "why_wait":
            self._say_call(c, None, {"why": why})

    def _done(self, why: str | None, error: str | None = None) -> None:
        j, self.job = self.job, None
        job = j["job"]
        task, facts = j["task"], j["facts"]
        info = dict(j["base"], why=why, ms=round(job.ms, 2), slices=job.slices,
                    tokens=len(job.out), prompt_tokens=len(job.ids), reused=job.reused,
                    finish=job.finish, waited_ms=round((self.clock() - j["t"]) * 1e3, 1),
                    **({"error": error} if error else {}))
        phrase = None
        if why is None:
            from server.conductor.redact import redact
            raw = self.slot.model.job_text(job)
            phrase, bad = clean(raw, task, j["source"], self.lang)
            if phrase is not None and any(similar(phrase, s) for s in self.said[-4:]):
                phrase, bad = None, "repeat"
            if phrase is None:
                info.update(why=bad, raw=redact(raw[:160])[0])
            else:
                phrase = redact(phrase)[0]
        self.trace.append({"task": task, "phrase": phrase, "why": info.get("why"),
                           **({"raw": info["raw"]} if info.get("raw") else {})})
        del self.trace[:-TRACE]
        if phrase is None:
            return self._failed(task, facts, info)
        now = self.clock()
        if task == "now":
            phrase = gerund(phrase, self.lang)
            el = now - self.t_start if self.t_start is not None else facts.get("elapsed", 0.0)
            if self.start is None:
                return self._failed(task, facts, dict(info, why="closed"))
            self.stretch_phrases.append(phrase)
            text = (f"denkt seit {dur(el, 'de')}: {phrase}" if self.lang == "de"
                    else f"thinking {dur(el)}: {phrase}")
            self._emit("thinking", text, info, now, model=True, phrase=phrase)
        elif task == "decided":
            el = facts.get("elapsed", 0.0)
            text = (f"nach {dur(el, 'de')} Nachdenken: {phrase}" if self.lang == "de"
                    else f"thought {dur(el)}: {phrase}")
            self._emit("thought", text, info, now, model=True, phrase=phrase)
        elif task == "why":
            c = facts.get("call")
            if c is not None and not c.get("said"):
                self._say_call(c, phrase, info)
        elif task == "plan":
            text = f"Plan: {phrase}" if self.lang == "de" else f"plan: {phrase}"
            self._emit("plan", text, info, now, model=True, phrase=phrase)

    def _failed(self, task: str, facts: dict, info: dict) -> None:
        self._why(info.get("why") or "failed")
        c = facts.get("call")
        if task == "why" and c is not None:
            self._say_call(c, None, info)

    def _why(self, why: str) -> None:
        w = self.stats["why"]
        w[why] = w.get(why, 0) + 1

    def _emit(self, kind: str, text: str, info: dict, now: float, *, model: bool,
              phrase: str | None = None) -> None:
        text = text.replace("*", "∗")
        self.ui.live(kind, text, info)
        self.stats["lines"] += 1
        if model:
            self.stats["phrases"] += 1
            self.t_model = now
            self.said.append(phrase or text)
        else:
            self.stats["plain"] += 1
        self._said(text, now, model=model)

    def _said(self, text: str, now: float, *, model: bool) -> None:
        self.t_line = now

    def _broke(self, exc: BaseException) -> None:
        self.broken = f"{type(exc).__name__}: {exc}"[:200]
        if self.job is not None:
            self.job = None
        self.log(f"[voice] off for this request ({self.broken})", flush=True)
