"""Server-side compaction: a client's compaction served as a continuation of the conversation the
engine holds, and a summary prepared before it is asked for.

THE MEASURED CASES (a user's ~/rrr6 session, opencode 1.18.34, 2026-10-03). A manual /compact at
111,636 tokens: 63,614 prompt tokens read cold (90 s), 2,116 reasoning tokens, 189.8 s in all. An
automatic one at 198,030 tokens (0.3.21: thinking off, no packages): 129,643 prompt tokens read cold
(207 s), 2,139 output tokens, 270 s. Both times the conversation sat resident in the engine, 110k
and 194k tokens of it, and nothing was reused: opencode's request is its compaction agent's prompt
as the system message and ONE user message, "Here is the conversation so far: <conversation>
[User]: ... [Assistant]: ... [Tool result]: ... </conversation>" and the summary instructions with
their template, a text rendering of the history that shares no token with what the engine read.

THE CONTINUATION. The engine holds the session's last generation (`ResidentPrefix.keep_end`, or the
same entry on the NVMe, engine/spill.py): every token the model read and wrote, runtime notes
included. A compaction of that session is served as that conversation plus one user turn: the
client's system text and the instructions after its history (`split`), with a sentence that says
the conversation above is the history they mean. The prefill resumes at the end anchor and reads
only that turn (a few hundred to a few thousand tokens); the model writes the summary from the
rows it already holds, images and all. What is not used: opencode's text rendering of the history
(tool results cut to 2,000 characters, images named, the newest ~15,000 tokens left out because
opencode keeps them verbatim after the summary). So the summary may also cover those newest turns;
they follow it verbatim anyway.

A GUARD. The record must be this conversation: the session id the request carries (opencode's
`x-session-id`) is the record's owner, and the client's history has user messages; the first and
the last of them (their first 120 characters, whitespace folded) must be in the record's text. Any
miss serves the request as the client sent it, as before.

THE PREPARED SUMMARY. A summary still has to be written: ~2,000 tokens at 40-50 tok/s at this
length, 40-60 s. So when the engine is idle (no request running or waiting for `idle_s`, no hold or
box lock), the resident conversation of an opencode session longer than `min_tokens` gets its
summary written in the background, as the continuation above with opencode's own instructions
(the last ones a request of that client brought, else the 1.18.34 text shipped here). A request
waiting for the engine stops it at the next prefill chunk or decode block; the conversation's rows,
end anchor and anchors are left as they were (`scratch`: no anchor taken, nothing kept). A
compaction request of that session at that end anchor, with the same instructions (the system text
and the instructions' last `KEY_TAIL` characters: the template), is answered from the prepared
summary without the engine.

Token ids, hashes and counts only in the logs and the journal; the summary text stays in memory.
"""

from __future__ import annotations

import hashlib
import re
import threading
import time

#: the shape a compaction request's history has (opencode 1.18.34); the router document's
#: `structural.compaction_shapes` may name others
HISTORY_OPEN = "<conversation>"
HISTORY_CLOSE = "</conversation>"
#: the user-message lines of opencode's text rendering of the history
USER_LINE = re.compile(r"^\[User\]: (.+)$", re.M)
GUARD_CHARS = 120
#: what must be equal for a prepared summary to answer a request: the system text, and this much
#: of the end of the instructions (opencode's template and rules, 1,049 characters folded from
#: `<template>` on; the first compaction and a later
#: one differ only before it, in the prior summary they quote)
KEY_TAIL = 1000
LEAD = ("The conversation above, from its first message to the last, is the history to summarize: "
        "where the instructions below speak of a <conversation>, they mean it.")
#: opencode 1.18.34's compaction agent prompt and the instructions of a first compaction, as its
#: requests carry them (captured 2026-10-04); the client's protocol, used only to prepare a summary
#: before a request of that client has brought its own
OPENCODE_SYSTEM = (
    "You are a context summarization agent. You are given a conversation between a user and an "
    "agent. Your goal is to produce a structured summary matching the format specified so another "
    "coding agent can continue the work.\n\nAlways follow the exact output structure requested by "
    "the user prompt. Keep every section, preserve exact file paths and identifiers when known, "
    "and prefer terse bullets over paragraphs.\n\nDo not continue the conversation. Do not respond "
    "to any questions in the conversation. Only output the structured summary in the exact format "
    "requested by the user prompt. Respond in the same language as the conversation.\n")
OPENCODE_INSTRUCTIONS = (
    "Create a new anchored summary from the conversation history in the <conversation> tags above "
    "so another coding agent can continue the work.\n\nOutput exactly the Markdown structure shown "
    "inside <template> and keep the section order unchanged. Do not include the <template> tags in "
    "your response.\n<template>\n## Objective\n- [one or two brief sentences describing what the "
    "user is trying to accomplish]\n\n## Important Details\n- [constraints/preferences, decisions "
    "and why, important facts/assumptions, exact context needed to continue, or \"(none)\"]\n\n## "
    "Work State\n### Completed\n- [finished work, verified facts, or changes made; otherwise "
    "\"(none)\"]\n\n### Active\n- [current work, partial changes, or investigation state; "
    "otherwise \"(none)\"]\n\n### Blocked\n- [blockers, failing commands, or unknowns; otherwise "
    "\"(none)\"]\n\n## Next Move\n1. [immediate concrete action, or \"(none)\"]\n2. [next action "
    "if known, or \"(none)\"]\n\n## Relevant Files\n- [file or directory path: why it matters, or "
    "\"(none)\"]\n</template>\n\nRules:\n- Keep every section, even when empty.\n- Use terse "
    "bullets, not prose paragraphs.\n- Preserve exact file paths, symbols, commands, error "
    "strings, URLs, and identifiers when known.\n- Do not mention the summary process or that "
    "context was compacted.")
SETTINGS = {"mode": "continue", "prepare": "on", "idle_s": 30.0, "min_tokens": 32768,
            "max_tokens": 8192, "prepare_max_tokens": 6144}


def _text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content
                       if isinstance(p, dict) and isinstance(p.get("text"), str))
    return ""


def _fold(s: str) -> str:
    return " ".join(s.split())


def marker(body: dict) -> bool | None:
    """SRV-351: the client's own word, `tandem: {"compaction": true|false}`. True: a compaction
    (chatd sends its real history and one instruction turn, already a continuation of what the
    engine holds); False: not one, whatever its shape; None: not said, the shape decides."""
    t = body.get("tandem")
    v = t.get("compaction") if isinstance(t, dict) else None
    return v if isinstance(v, bool) else None


def split(body: dict) -> dict | None:
    """A compaction request's parts: `system` (its system text), `instructions` (the user text
    after the history), `guard` (the first and last user lines of the history, folded) and
    `history_chars`. None when the request has not that shape, or when it says what it is
    (`marker`): a marked request is never rewritten."""
    if marker(body) is not None:
        return None
    msgs = [m for m in (body.get("messages") or []) if isinstance(m, dict)]
    if body.get("tools") or len(msgs) != 2 or msgs[0].get("role") != "system" \
            or msgs[1].get("role") != "user":
        return None
    system, user = _text(msgs[0].get("content")), _text(msgs[1].get("content"))
    i, j = user.find(HISTORY_OPEN), user.rfind(HISTORY_CLOSE)
    if i < 0 or j <= i:
        return None
    history = user[i + len(HISTORY_OPEN):j]
    instructions = user[j + len(HISTORY_CLOSE):].strip()
    if not system.strip() or not instructions:
        return None
    users = [_fold(m.group(1))[:GUARD_CHARS] for m in USER_LINE.finditer(history)]
    users = [u for u in users if len(u) >= 8]
    guard = [users[0], users[-1]] if users else []
    return {"system": system, "instructions": instructions, "guard": guard,
            "history_chars": len(history)}


def key(system: str, instructions: str) -> str:
    """What a prepared summary must share with the request it answers."""
    h = hashlib.sha256()
    h.update(_fold(system).encode())
    h.update(b"\0")
    h.update(_fold(instructions)[-KEY_TAIL:].encode())
    return h.hexdigest()[:24]


def tokens_sha(tokens) -> str:
    h = hashlib.sha256()
    h.update(",".join(str(int(t)) for t in tokens).encode())
    return h.hexdigest()[:24]


def turn_text(system: str, instructions: str) -> str:
    """The user turn the continuation adds."""
    return f"{system.strip()}\n\n{LEAD}\n\n{instructions.strip()}"


def tail_ids(tok, system: str, instructions: str, think: bool = False) -> list[int]:
    """The ids after the record's last token: the end of the last message, the user turn and the
    generation prompt, as the chat template writes them (thinking off unless `think`)."""
    text = tok.apply_chat_template([{"role": "user", "content": turn_text(system, instructions)}],
                                   add_generation_prompt=True, tokenize=False,
                                   enable_thinking=bool(think))
    k = text.find("<|im_start|>user")
    text = text[k:] if k >= 0 else text
    return list(tok("<|im_end|>\n" + text, add_special_tokens=False).input_ids)


def record_text(tok, tokens, vocab: int) -> str:
    """The record's text for the guard: its token ids decoded, an image's key ids left out."""
    return _fold(tok.decode([int(t) for t in tokens if 0 <= int(t) < vocab],
                            skip_special_tokens=False))


def guard_ok(parts: dict, text: str) -> bool:
    return bool(parts["guard"]) and all(g in text for g in parts["guard"])


class Record:
    """The conversation a compaction continues: its tokens up to its end anchor, where it is
    (`resident` or `spill`), and the rotary offset its images left (`delta`)."""

    def __init__(self, tokens, at: int, where: str, delta: int = 0, mm: bool = False):
        self.tokens, self.at, self.where = list(tokens[:at]), int(at), where
        self.delta, self.mm = int(delta), bool(mm)


def find_record(res, spill, session: str | None) -> Record | None:
    """The session's last generation, from the rows in place or from the NVMe. None when neither
    holds one (no end anchor: a guest in place, a parked or pinned conversation)."""
    if session is None or res is None:
        return None
    end = res.end
    if (res.owner == session and end is not None and res._stash is None
            and int(end["at"]) == res.valid and len(res.tokens) >= res.valid):
        return Record(res.tokens, res.valid, "resident", end.get("delta", 0), end.get("mm", False))
    if spill is not None:
        with spill.lock:
            entries = list(spill.entries.values())
        for e in sorted(entries, key=lambda e: -float(e.man.get("used", 0))):
            end = e.man.get("end")
            if e.man.get("owner") == session and end is not None and int(end["at"]) == e.valid:
                return Record(e.tokens, e.valid, "spill", end.get("delta", 0),
                              end.get("mm", False))
    return None


class Prepared:
    """Summaries written ahead, one per session: at which end anchor, under which instructions."""

    def __init__(self, keep: int = 8):
        self.keep = int(keep)
        self.lock = threading.Lock()
        self.by_session: dict[str, dict] = {}
        self.instructions: dict[str, tuple[str, str]] = {
            "opencode": (OPENCODE_SYSTEM, OPENCODE_INSTRUCTIONS)}
        self.kinds: dict[str, str] = {}             # session -> its client's kind
        self.stats = {"prepared": 0, "served": 0, "stale": 0, "stopped": 0, "yielded": 0,
                      "failed": 0,
                      "too_long": 0, "continued": 0, "client": 0, "guard": 0, "no_record": 0,
                      "last": None}

    def seen(self, session: str | None, kind: str | None) -> None:
        if session and kind:
            with self.lock:
                self.kinds[session] = kind
                while len(self.kinds) > 64:
                    self.kinds.pop(next(iter(self.kinds)))

    def learn(self, kind: str | None, parts: dict) -> None:
        """A compaction request's own instructions, for the next summary prepared for its client
        (the prior summary a later compaction quotes is left out of the key, see `key`)."""
        if kind:
            with self.lock:
                self.instructions[kind] = (parts["system"], parts["instructions"])

    def put(self, session: str, rec: Record, k: str, ids: list[int], info: dict) -> None:
        with self.lock:
            self.by_session[session] = {"at": rec.at, "tsha": tokens_sha(rec.tokens), "key": k,
                                        "ids": list(ids), "t": time.time(), **info}
            while len(self.by_session) > self.keep:
                self.by_session.pop(next(iter(self.by_session)))
            self.stats["prepared"] += 1
            self.stats["last"] = {"tokens": rec.at, "out": len(ids), **info}

    def has(self, session: str, rec: Record) -> bool:
        with self.lock:
            p = self.by_session.get(session)
        return p is not None and p["at"] == rec.at

    def take(self, session: str, rec: Record, k: str) -> list[int] | None:
        """The prepared summary's ids for this request, or None (none, another end anchor, other
        instructions)."""
        with self.lock:
            p = self.by_session.get(session)
        if p is None:
            return None
        if p["at"] != rec.at or p["key"] != k or p["tsha"] != tokens_sha(rec.tokens):
            self.stats["stale"] += 1
            return None
        self.stats["served"] += 1
        return list(p["ids"])

    def report(self) -> dict:
        with self.lock:
            return {**self.stats, "sessions": len(self.by_session),
                    "kinds_with_instructions": sorted(self.instructions)}
