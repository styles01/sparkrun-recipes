"""What a request's structure says, and what the checkers said (SRV-123).

Two kinds of signal, both read without judging what anybody meant:

  * `shape(body)`: roles, tools, message counts, the last user message's text and whether the turn
    only carries tool results (a tool's media in a user message after them included, SRV-161),
    whether it is opencode's compaction request, which skills the client already loaded. These are structural facts; no word list decides anything here (the project's
    rule: intent comes from a model, never from keyword routing).
  * `side_shape(body)`: whether the request is a client's side request (S1): no tools, and the
    client's fixed title or summary instruction opening the system message.
  * `context_line(...)`: the one structured line the router adds to the classifier's digest
    (SRV-153): the previous class, the last tools, the file types, the checker's status and the
    compaction flag. Names and extensions only, never a path or a file's text.
  * `checker_results(text)`: the `RESULT: PASS|FAIL <- reasons` line the demo checkers print
    (`sketchcheck`, `pagodacheck`, `scenecheck`, and the eval suites' `calccheck` and
    `taskcheck`) and the
    numbers in its reasons. Only tool results are read, never assistant text, so a RESULT line
    the model quotes is not a result. A new checker needs one entry in `CHECKERS` (its header
    line), nothing else; the session monitor passes its own list and patterns from its policy.
"""

from __future__ import annotations

import hashlib
import json
import re

# A checker's first line names it and what it checked; the gate id carries that target so a
# checker run on the wrong file is a different gate.
CHECKERS = {
    "sketchcheck": re.compile(r"^sketchcheck\s+(\S+)", re.M),
    "pagodacheck": re.compile(r"^pagodacheck\s+(\S+)", re.M),
    "scenecheck": re.compile(r"^scenecheck\s+(\S+)", re.M),
    "calccheck": re.compile(r"^calccheck\s+(\S+)", re.M),
    "taskcheck": re.compile(r"^taskcheck\s+(\S+)", re.M),
}
_RESULT = re.compile(r"^RESULT:\s*(PASS|FAIL)\b[ \t]*(?:<-[ \t]*(.*))?$", re.M)
_GATE = re.compile(r"^\(([a-z0-9]{1,3})\)\s*")
_NUM = re.compile(r"(-?\d+(?:\.\d+)?)\s*(<=|>=|<|>)\s*(-?\d+(?:\.\d+)?)")
_NAME = re.compile(r"([A-Za-z_][A-Za-z0-9_ .-]*?)\s+(-?\d+(?:\.\d+)?)\s*(?:<=|>=|<|>)")

# opencode's compaction request: no tools, no system text, the history as one user message
# (docs/caches.md). The length floor keeps a short one-line request without a system prompt out.
COMPACTION_MIN_CHARS = 2000
# Opencode 1.18.34 sends its compaction request with its `compaction` agent's prompt as
# the system message (system + the history as one user message, no tools), which the shape above
# rules out: the request was routed as a design turn (7 packages, high effort, 2,116 reasoning
# tokens) and the session never knew it had been compacted. The agent's prompt opens with this
# fixed text; like the side shapes below it is the client's protocol (which call this is), not a
# word that decides a class.
COMPACTION_SHAPES = (
    {"id": "opencode_compaction", "client": "opencode",
     "match": "You are a context summarization agent. You are given a conversation between a "
              "user and an agent."},
)
# SRV-161: a provider without media in tool results gets a tool's images (or PDF) from opencode as
# a user message right after the tool results, captioned with this fixed text and nothing else.
# It is the tool's output, not a person writing: the turn keeps the session's class. The caption
# is the client's protocol, not a word that decides a class; the message's position (after a tool
# result) and its parts (media, no text of its own) decide.
TOOL_MEDIA_CAPTION = "Attached media from tool result:"
_MEDIA_PARTS = ("image_url", "image", "file", "input_file", "input_image")
# S1 (2026-10-02): a client's side request -- the title or summary call it sends to its small model,
# beside the session's turns -- has no tools and opens its system message with the client's own
# fixed instruction. That text is the client's protocol, the way the compaction request's shape and
# the tool-media caption are: it says which call this is, not what anybody meant. Each shape is the
# instruction's first line as the client ships it (opencode 1.18.34's `title` and `summary` agents,
# the Hermes agent 0.21.0's title_generator); the router document's `structural.side_shapes`
# replaces the list. `max_tokens` is the shape's output cap.
SIDE_SHAPES = (
    {"id": "opencode_title", "client": "opencode", "kind": "title", "max_tokens": 64,
     "match": "You are a title generator. You output ONLY a thread title. Nothing else."},
    {"id": "opencode_summary", "client": "opencode", "kind": "summary", "max_tokens": 256,
     "match": "Summarize what was done in this conversation. Write like a pull request "
              "description."},
    {"id": "hermes_title", "client": "Hermes", "kind": "title", "max_tokens": 64,
     "match": "You name chat sessions. Given the user's opening message, write a title that "
              "lets them find this conversation again in a list."},
)
# the arguments that name a file in the clients' tools: opencode's read / edit / write take
# `filePath`, Claude-style tools `file_path`; `path` is a directory as often as a file (grep, glob,
# list), so it counts only when its last component has an extension
FILE_ARGS = ("filePath", "file_path")
# SRV-243 (UX2, 2026-10-07): what the request's tools can do, read off their schemas -- the
# client's protocol, never what anybody wrote. A tool with a `command` (or `cmd`) argument runs a
# shell: opencode's `bash`, Open WebUI's terminal `run_command`, Claude-style `Bash`. A tool with
# a path argument and a content argument writes a file: opencode's `write` and `edit`, the
# terminal's `write_file` and `replace_file_content`. Open WebUI's own `search_web` and
# `fetch_url` do neither. A package names the capabilities its steps need (`requires_tools`) and
# the router leaves it out of a turn whose tools lack one (router.py `_block`).
SHELL_ARGS = ("command", "cmd")
WRITE_PATH_ARGS = ("filePath", "file_path", "path", "filename")
WRITE_BODY_ARGS = ("content", "contents", "text", "newString", "new_string", "replacements",
                   "patch", "diff")
# OPS-39 (the first packages): the render tool of SRV-105 takes the page as an `html` string
# argument (`render(html, frames_at, size)`, `audio_probe(html, seconds)`); the video tool takes
# a scene script and a cue list (`render_video(scenes, cues)`). A package whose steps need one
# (render-tools, explainer-video) names it in `requires_tools` and stays out of a turn without it.
RENDER_ARGS = ("html",)
VIDEO_ARGS = ("scenes", "cues")
CAPS = ("shell", "write", "render", "render_video")
_EXT = re.compile(r"^\.[a-z0-9]{1,8}$")
_TOOL_NAME = re.compile(r"[^A-Za-z0-9_.-]")
_MCP = re.compile(r"^[A-Za-z0-9.]+(?:-[A-Za-z0-9.]+)+_(?=.)")     # `server-name_` of an MCP tool
# SRV-153: the context line's size, the share of the digest it takes from the tail
LINE_MAX = 120
LINE_TOOLS, LINE_EXTS, LINE_CALLS = 3, 4, 8


def _text(content) -> str:
    """The text of a message's content: a string, or the text parts of a list."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content
                       if isinstance(p, dict) and isinstance(p.get("text"), str))
    return ""


def offers_tools(body: dict) -> bool:
    """SRV-621: the request offers at least one function tool (`{"type": "function",
    "function": {"name": ...}}`, or a flat `{"name": ...}` as the template reads it) and its
    `tool_choice` is not "none". router.resolve's tools rule reads it."""
    if not isinstance(body, dict) or body.get("tool_choice") == "none":
        return False
    for t in body.get("tools") or ():
        if not isinstance(t, dict) or t.get("type", "function") != "function":
            continue
        fn = t.get("function") if isinstance(t.get("function"), dict) else t
        if isinstance(fn.get("name"), str) and fn["name"].strip():
            return True
    return False


def tool_caps(body: dict) -> tuple[frozenset, list[str]]:
    """SRV-243: (the capabilities the request's tools give, their names). Nothing with
    `tool_choice: none`. A flat tool (`{"name": ..., "parameters": ...}`) counts as the template
    reads it."""
    if not isinstance(body, dict) or body.get("tool_choice") == "none":
        return frozenset(), []
    caps, names = set(), []
    for t in body.get("tools") or ():
        if not isinstance(t, dict):
            continue
        fn = t.get("function") if isinstance(t.get("function"), dict) else t
        name = fn.get("name")
        if not isinstance(name, str) or not name:
            continue
        names.append(name)
        props = (fn.get("parameters") or {}).get("properties") if isinstance(
            fn.get("parameters"), dict) else None
        keys = set(props) if isinstance(props, dict) else set()
        if any(isinstance((props or {}).get(k), dict) and
               (props[k].get("type") in (None, "string")) for k in SHELL_ARGS if k in keys):
            caps.add("shell")
        if keys & set(WRITE_PATH_ARGS) and keys & set(WRITE_BODY_ARGS):
            caps.add("write")
        if any(isinstance((props or {}).get(k), dict) and
               (props[k].get("type") in (None, "string")) for k in RENDER_ARGS if k in keys):
            caps.add("render")
        if set(VIDEO_ARGS) <= keys:
            caps.add("render_video")
    return frozenset(caps), names


# SRV-623 (C5): what a tool changes outside the conversation, read off its declaration the way
# tool_caps reads it: the parameter names, the operation its name opens with (its description's
# first word when the name states none), an MCP-style `annotations` hint. Never what anybody wrote
# to the model. `send`: a recipient and a message parameter (an email's `to` and `body`, a chat's
# `channel` and `text`); `schedule`: a time and a subject parameter (an event's `title` and
# `date`, a reminder's `message` and `datetime`); `write`: tool_caps' write (a path and a body);
# `create`, `delete`: the operation's verb. A name that opens with a read verb is a read whatever
# its parameters, as is `readOnlyHint: true`; `destructiveHint: true` is a delete. A shell
# (`command`) is none of them: opencode reads through `bash`, often several calls at once.
MUTATIONS = ("send", "schedule", "create", "write", "delete")
RECIPIENT_ARGS = ("to", "recipient", "recipients", "cc", "bcc", "channel", "channel_id",
                  "chat_id", "phone", "phone_number")
MESSAGE_ARGS = ("body", "message", "text", "content")
TIME_ARGS = ("date", "time", "datetime", "start", "start_time", "start_date", "starts_at", "due",
             "due_date", "when", "remind_at", "scheduled_at", "send_at")
SUBJECT_ARGS = ("title", "summary", "subject", "message", "text")
READ_VERBS = ("get", "list", "search", "find", "read", "fetch", "lookup", "look", "query",
              "check", "view", "show", "describe", "count", "browse", "inspect", "retrieve",
              "download", "translate", "calculate", "compute", "convert", "summarize", "preview",
              "validate", "estimate")
VERB_EFFECTS = {
    **{v: "send" for v in ("send", "email", "mail", "notify", "reply", "forward", "post",
                           "publish", "share")},
    **{v: "schedule" for v in ("schedule", "book", "reserve", "remind")},
    **{v: "write" for v in ("write", "edit", "patch", "save", "append")},
    **{v: "create" for v in ("create", "add", "insert", "new", "make", "set", "update", "modify",
                             "change", "put", "move", "rename", "assign", "invite", "submit",
                             "upload", "register", "approve", "merge", "pay", "transfer",
                             "order", "buy", "enable", "disable")},
    **{v: "delete" for v in ("delete", "remove", "rm", "destroy", "drop", "cancel", "purge",
                             "erase", "unlink", "archive", "revoke", "clear")},
}
_WORD = re.compile(r"[A-Za-z]+")
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def _verb(word: str) -> str | None:
    """A known operation verb as `word` states it ("Sends" and "creates" read as their stem)."""
    w = word.lower()
    for cand in (w, w[:-1] if w.endswith("s") else None, w[:-2] if w.endswith("es") else None):
        if cand and (cand in VERB_EFFECTS or cand in READ_VERBS):
            return cand
    return None


def _operation(fn: dict, name: str) -> str | None:
    """The verb a tool's declaration opens with: its name's first word (after an MCP server
    prefix, split at `_ - .` and camelCase), else its description's first word."""
    base = _MCP.sub("", name)
    first = re.split(r"[_\-. ]+", _CAMEL.sub("_", base).strip("_-. "))[0] if base else ""
    got = _verb(first) if first else None
    if got is None and isinstance(fn.get("description"), str):
        m = _WORD.search(fn["description"])
        got = _verb(m.group(0)) if m else None
    return got


def tool_effects(body: dict) -> dict[str, frozenset]:
    """SRV-623: per offered tool, the MUTATIONS its declaration says it makes (empty: a read or
    nothing known). Nothing with `tool_choice: none`; a flat tool counts as the template reads
    it."""
    if not isinstance(body, dict) or body.get("tool_choice") == "none":
        return {}
    out: dict[str, frozenset] = {}
    for t in body.get("tools") or ():
        if not isinstance(t, dict):
            continue
        fn = t.get("function") if isinstance(t.get("function"), dict) else t
        name = fn.get("name")
        if not isinstance(name, str) or not name:
            continue
        hints = fn.get("annotations") if isinstance(fn.get("annotations"), dict) else \
            t.get("annotations") if isinstance(t.get("annotations"), dict) else {}
        verb = _operation(fn, name)
        kinds: set = set()
        if hints.get("readOnlyHint") is True:
            out[name] = frozenset()
            continue
        if hints.get("destructiveHint") is True:
            kinds.add("delete")
        elif verb in READ_VERBS:
            out[name] = frozenset()
            continue
        props = (fn.get("parameters") or {}).get("properties") if isinstance(
            fn.get("parameters"), dict) else None
        keys = set(props) if isinstance(props, dict) else set()
        if keys & set(RECIPIENT_ARGS) and keys & set(MESSAGE_ARGS):
            kinds.add("send")
        if keys & set(TIME_ARGS) and keys & set(SUBJECT_ARGS):
            kinds.add("schedule")
        if keys & set(WRITE_PATH_ARGS) and keys & set(WRITE_BODY_ARGS):
            kinds.add("write")
        if verb in VERB_EFFECTS:
            kinds.add(VERB_EFFECTS[verb])
        out[name] = frozenset(kinds)
    return out


def _images(content) -> int:
    if not isinstance(content, list):
        return 0
    return sum(1 for p in content if isinstance(p, dict) and
               (p.get("type") in ("image_url", "image") or "image_url" in p))


def _media(content) -> int:
    """The media parts of a message: images and files."""
    if not isinstance(content, list):
        return 0
    return sum(1 for p in content if isinstance(p, dict) and
               (p.get("type") in _MEDIA_PARTS or "image_url" in p))


def is_tool_media(msgs: list, i: int) -> bool:
    """SRV-161: message `i` is a user message that only carries what a tool returned: it follows
    tool results (other such messages between them and it are allowed), it has media parts, and
    its text is empty or the client's caption. A user message with text of its own, or one that
    does not follow a tool result (a person attaching an image), is not."""
    m = msgs[i]
    if m.get("role") != "user" or not _media(m.get("content")):
        return False
    if _text(m.get("content")).strip() not in ("", TOOL_MEDIA_CAPTION):
        return False
    j = i - 1
    while j >= 0 and msgs[j].get("role") == "user" and _media(msgs[j].get("content")) and \
            _text(msgs[j].get("content")).strip() in ("", TOOL_MEDIA_CAPTION):
        j -= 1
    return j >= 0 and msgs[j].get("role") == "tool"


def compaction_shape(body: dict, shapes=COMPACTION_SHAPES) -> dict | None:
    """The client compaction shape this request has, or None: no tools offered, a system message
    whose text opens with one shape's `match` (whitespace runs count as one space), and the
    conversation to compact as the one message after it, a user message."""
    if body.get("tools"):
        return None
    msgs = [m for m in (body.get("messages") or []) if isinstance(m, dict)]
    if len(msgs) != 2 or msgs[0].get("role") != "system" or msgs[1].get("role") != "user":
        return None
    text = " ".join(_text(msgs[0].get("content")).split())
    for s in shapes:
        if text.startswith(" ".join(str(s.get("match") or "").split()) or "\0"):
            return s
    return None


def side_shape(body: dict, shapes=SIDE_SHAPES) -> dict | None:
    """The side-request shape this request has, or None: no tools offered, and the first message is
    a system message whose text opens with one shape's `match` (whitespace runs count as one
    space, so a client that rewraps its instruction still matches)."""
    if body.get("tools"):
        return None
    msgs = [m for m in (body.get("messages") or []) if isinstance(m, dict)]
    if not msgs or msgs[0].get("role") != "system":
        return None
    text = " ".join(_text(msgs[0].get("content")).split())
    for s in shapes:
        if text.startswith(" ".join(str(s.get("match") or "").split()) or "\0"):
            return s
    return None


def checker_results(text: str, checkers: dict | None = None, gate=None,
                    reading=None) -> list[dict]:
    """Every RESULT line in one tool result: `{checker, target, verdict, reasons, readings}`,
    each reading `{gate, name, value, op, limit}` with `gate` the checker's own label ("a") or
    the metric name. A tool result may hold two checker runs; each RESULT line pairs with the
    nearest header above it. `checkers` (name -> header pattern with the target as its group),
    `gate` (a reason's gate label) and `reading` (value, comparison, limit as three groups)
    replace the built-in patterns; patterns may be compiled or strings."""
    def rx(p):
        return re.compile(p, re.M) if isinstance(p, str) else p
    checkers = CHECKERS if checkers is None else {k: rx(v) for k, v in checkers.items()}
    gate_rx = _GATE if gate is None else rx(gate)
    num_rx = _NUM if reading is None else rx(reading)
    out = []
    heads = sorted(((m.start(), name, m.group(1)) for name, h in checkers.items()
                    for m in h.finditer(text)))
    for m in _RESULT.finditer(text):
        above = [h for h in heads if h[0] < m.start()]
        checker, target = (above[-1][1], above[-1][2]) if above else ("unknown", "")
        reasons = [r.strip() for r in (m.group(2) or "").split(";") if r.strip()]
        readings = []
        for r in reasons:
            g = gate_rx.match(r)
            n = num_rx.search(r)
            if n is None:
                continue
            name_m = _NAME.search(gate_rx.sub("", r, count=1))
            name = (name_m.group(1).strip() if name_m else "").strip(" :")
            op = n.group(2)
            readings.append({"gate": g.group(1) if g else (name or "value"), "name": name,
                             "value": float(n.group(1)), "op": op,
                             "limit": float(n.group(3))})
        out.append({"checker": checker, "target": target, "verdict": m.group(1),
                    "reasons": reasons, "readings": readings})
    return out


class Shape:
    """The structural facts of one chat request."""

    __slots__ = ("tools", "n_messages", "has_system", "last_role", "tool_turn", "user_text",
                 "user_index", "user_hash", "images", "compaction", "results", "loaded_skills",
                 "system_text", "tool_names_called", "first_hash", "files", "call_files",
                 "last_checker", "max_checks", "tool_media", "n_user", "fresh", "turn1_hash")

    def __repr__(self):
        return (f"Shape(tools={self.tools}, last={self.last_role}, tool_turn={self.tool_turn}, "
                f"compaction={self.compaction}, user_index={self.user_index})")


def _file_of(args) -> str | None:
    """The file a tool call names, if it names one (FILE_ARGS, or a `path` with an extension)."""
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            return None
    if not isinstance(args, dict):
        return None
    for k in FILE_ARGS:
        v = args.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    v = args.get("path")
    if isinstance(v, str) and ext_of(v):
        return v.strip()
    return None


def ext_of(path: str) -> str | None:
    """A file's extension, lower case, or None: `.py`, never the path or the name."""
    base = path.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    if "." not in base.strip("."):
        return None
    ext = "." + base.rsplit(".", 1)[-1].lower()
    return ext if _EXT.match(ext) else None


def _is_tool_response(text: str) -> bool:
    t = text.strip()
    return t.startswith("<tool_response>") and t.endswith("</tool_response>")


def _compaction_marker(body: dict) -> bool | None:
    """SRV-351: `tandem.compaction` in the body (chatd sends it on every request); when present
    the shape guesses below are skipped (server/compaction.py `marker`)."""
    from server.compaction import marker
    return marker(body)


#: SRV-175: the clients whose compaction request is the bare shape (no tools, no system text, the
#: history as one long user message). Any other client's long paste with "summarize" in a fresh
#: chat has that shape too (UT1, 2026-10-06: chatcmpl-6f4facab..., 83,523 characters from an
#: unnamed client, journaled `compaction: served client, why shape`).
BARE_COMPACTION_CLIENTS = ("opencode",)


def shape(body: dict, skill_names: tuple = (), client_kind: str | None = None) -> Shape:
    """The facts `router.py` decides from. `skill_names`: the installed skills, for the exact-name
    check of which ones the client already loaded. `client_kind` (the ledger's): the bare
    compaction shape is a compaction only for a client that sends it (SRV-175)."""
    s = Shape()
    msgs = [m for m in (body.get("messages") or []) if isinstance(m, dict)]
    tools = body.get("tools") or []
    tc = body.get("tool_choice")
    s.tools = bool(tools) and tc != "none"
    s.n_messages = len(msgs)
    s.has_system = bool(msgs) and msgs[0].get("role") == "system"
    s.system_text = _text(msgs[0].get("content")) if s.has_system else ""
    s.last_role = msgs[-1].get("role") if msgs else None
    # the last message a person wrote: a user message that is not only tool responses, nor a
    # tool's media carried in a user message (SRV-161)
    s.user_index, s.user_text, s.images, s.tool_media = None, "", 0, False
    for i in range(len(msgs) - 1, -1, -1):
        m = msgs[i]
        if m.get("role") != "user" or _is_tool_response(_text(m.get("content"))):
            continue
        if is_tool_media(msgs, i):
            s.tool_media = s.tool_media or i == len(msgs) - 1
            continue
        s.user_index, s.user_text = i, _text(m.get("content"))
        s.images = _images(m.get("content"))
        break
    s.tool_turn = s.last_role == "tool" or (
        s.last_role == "user" and s.user_index is not None and s.user_index != len(msgs) - 1)
    s.user_hash = (hashlib.sha256(f"{s.user_index}\x00{s.user_text}".encode()).hexdigest()[:24]
                   if s.user_index is not None else None)
    said = _compaction_marker(body)
    s.compaction = said if said is not None else (
        (client_kind in BARE_COMPACTION_CLIENTS and not s.tools and not s.has_system
         and len(msgs) == 1 and s.last_role == "user"
         and len(s.user_text) >= COMPACTION_MIN_CHARS) or compaction_shape(body) is not None)
    # the checker results that arrived since the model last wrote
    s.results = []
    for m in reversed(msgs):
        if m.get("role") == "assistant":
            break
        if m.get("role") in ("tool", "user"):
            s.results[:0] = checker_results(_text(m.get("content")))
    # skills the client already loaded: a call to its skill tool naming one, exact names only;
    # and the files the calls named, in call order (one entry per call: None when it named none)
    loaded, called, call_files = set(), [], []
    for m in msgs:
        for c in (m.get("tool_calls") or []) if m.get("role") == "assistant" else []:
            fn = (c or {}).get("function") or {}
            name = str(fn.get("name") or "")
            called.append(name)
            call_files.append(_file_of(fn.get("arguments")))
            if "skill" in name.lower():
                args = fn.get("arguments")
                blob = args if isinstance(args, str) else json.dumps(args or {})
                for sk in skill_names:
                    if re.search(rf"(?<![A-Za-z0-9_-]){re.escape(sk)}(?![A-Za-z0-9_-])", blob):
                        loaded.add(sk)
    s.loaded_skills = loaded
    s.tool_names_called = called
    s.call_files = call_files
    s.files = list(dict.fromkeys(f for f in call_files if f))
    # the newest checker result anywhere in the conversation, and the most gates one known
    # checker's RESULT line named (SRV-157's size signal); tool results only, as above
    s.last_checker, s.max_checks = None, None
    for m in msgs:
        if m.get("role") not in ("tool", "user"):
            continue
        t = _text(m.get("content"))
        if "RESULT:" not in t:
            continue
        for r in checker_results(t):
            s.last_checker = (r["checker"], r["verdict"])
            if r["checker"] != "unknown":
                s.max_checks = max(s.max_checks or 0, len(r["reasons"]))
    # the conversation's head: its messages up to and including the first user message, which
    # every turn sends again. It was the first two messages, and a chat with no system message
    # (the raw model's, a Hermes or Open WebUI chat) has one message on its first turn and the
    # first answer as the second on every turn after: turn 2 got another session, turn 1's note
    # was not replayed, and the prompt parted from the resident rows at the note (2026-10-02 22:07,
    # 0.3.14's live check: the follow-up resumed nowhere, `cache_n` 0)
    k = next((i for i, m in enumerate(msgs) if m.get("role") == "user"), 1)
    first = msgs[:k + 1]
    s.first_hash = hashlib.sha256(json.dumps(
        [(m.get("role"), _text(m.get("content"))[:4096]) for m in first],
        ensure_ascii=False).encode()).hexdigest()[:24] if first else None
    # SRV-225: which conversation of that head this is, for a client that sends no
    # id (Open WebUI): how many messages a person wrote (not tool responses, not tool media),
    # whether the request is a conversation's very first turn (its one user message is the last
    # message: a new chat, or that turn regenerated), and from the second user message on a hash
    # of the whole first turn as the client sends it back (the first message, the answer, its
    # tool calls and results), which two chats with the same opening do not share
    people = [i for i, m in enumerate(msgs) if m.get("role") == "user"
              and not _is_tool_response(_text(m.get("content"))) and not is_tool_media(msgs, i)]
    s.n_user = len(people)
    s.fresh = len(people) == 1 and people[0] == len(msgs) - 1
    s.turn1_hash = None
    if len(people) >= 2:
        turn1 = []
        for m in msgs[:people[1]]:
            calls = [((c or {}).get("function") or {}) for c in (m.get("tool_calls") or [])]
            turn1.append((m.get("role"), _text(m.get("content"))[:4096],
                          [(str(f.get("name")), str(f.get("arguments"))[:512]) for f in calls]))
        s.turn1_hash = hashlib.sha256(json.dumps(turn1, ensure_ascii=False, default=str)
                                      .encode()).hexdigest()[:16]
    return s


def skills_named(system_text: str, skill_names) -> list[str]:
    """Installed skills the system prompt names by their exact id (AGENTS.md lists them)."""
    return [n for n in skill_names
            if re.search(rf"(?<![A-Za-z0-9_-]){re.escape(n)}(?![A-Za-z0-9_-])", system_text)]


def digest(text: str, max_chars: int, line: str | None = None) -> str:
    """The text a classifier sees: the whole message when it fits, else its head and tail
    (the task is usually stated first, the latest detail last), marked where it was cut. With a
    context `line` (SRV-153) the line follows on its own last line and its characters come out of
    the tail's share, so the whole stays within `max_chars`."""
    text = text.strip()
    extra = len(line) + 1 if line else 0
    if len(text) + extra <= max_chars:
        out = text
    else:
        head = int(max_chars * 0.7)
        tail = max_chars - head - 7 - extra
        if tail < 0:
            head, tail = max(0, head + tail), 0
        out = text[:head].rstrip() + "\n[...]\n" + (text[-tail:].lstrip() if tail else "")
    return out + "\n" + line if line else out


def context_line(prev: str | None = None, conf: float | None = None, tools=(), exts=(),
                 checker: tuple | None = None, compaction: bool = False) -> str:
    """`prev=code_edit@0.93 tools=read,edit,bash files=.py,.md checker=calccheck:FAIL
    compaction=no`: what the router already knows about the session, for the classifier, and
    `none` for whatever it does not know yet. At most LINE_MAX characters: the last LINE_TOOLS
    tool names (an MCP tool without its server, `take_screenshot` of
    `chrome-devtools_take_screenshot`) and LINE_EXTS extensions, shortened step by step when the
    line runs over."""
    p = "none" if not prev else (f"{prev}@{conf:.2f}" if conf is not None else prev)
    c = (f"{_TOOL_NAME.sub('', str(checker[0]))[-16:] or 'unknown'}:{checker[1]}"
         if checker else "none")
    names = [_TOOL_NAME.sub("", _MCP.sub("", str(n))) for n in list(tools)[-LINE_TOOLS:] if n]
    ok = [e for e in exts if _EXT.match(e or "")]
    line = ""
    for width, n_ext in ((16, LINE_EXTS), (16, 3), (12, 3), (12, 2), (8, 2), (6, 1)):
        t = ",".join(n[:width] for n in names) or "none"
        f = ",".join(ok[:n_ext]) or "none"
        line = f"prev={p} tools={t} files={f} checker={c} compaction={'yes' if compaction else 'no'}"
        if len(line) <= LINE_MAX:
            return line
    return line[:LINE_MAX]


def line_facts(sh: Shape) -> dict:
    """The parts of the context line the request itself carries: the last tool names, the
    extensions of the files the last LINE_CALLS calls named, the newest checker result."""
    exts = []
    for f in sh.call_files[-LINE_CALLS:]:
        e = ext_of(f) if f else None
        if e and e not in exts:
            exts.append(e)
    return {"tools": sh.tool_names_called[-LINE_TOOLS:], "exts": exts,
            "checker": sh.last_checker}
