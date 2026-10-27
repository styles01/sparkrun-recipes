"""The background memory writer (2026-10-05, P-M2; the design: Memo "Memory for TandemLLM: design
advice", section 2, "The background write: the hidden turn").

Besides the model's own blocks in its answers, a session's facts are written by a hidden turn of
Qwen itself, thinking off, on the conversation the engine already holds: the resident record plus
one user turn with the writer's instructions (WRITER below) and the notes the scope already has
that match the session (so it updates instead of adding). Its output is parsed with the same
parser as the inline blocks (server/memblock.py) and written by the same store with
`writer: qwen-writer`; a block that fails the schema or a rule is dropped and journaled.

THE TRIGGERS, cheapest first:
  - `compaction`: right after a compaction's summary, the engine holds the whole history; with
    `after_compaction: turn` the writer runs before the engine is given to the next request
    (bounded by `max_tokens` and `turn_deadline_s`), with `idle` it waits for an idle moment like
    the others;
  - `give_up`: a turn that ended in GIVE_UP (the monitor's S4 stall included);
  - `idle`: the session's end, no request for `idle_s` seconds (180 to start, OPS-49's gap
    histogram decides later).
A session is written again only after `min_new_tokens` more of it (the compaction trigger aside).

THE GATE. Before the turn, Laya answers "do these messages contain something to remember" on the
digest of the session's last user messages (about 20 ms); a no at 0.6 or more skips the turn.
Without a decider, or below 0.6, the turn runs (journaled with the gate's answer and how).

THE MODES (`settings.memory.notes.writer.mode`): off; shadow (the turn runs, its blocks are
journaled as counts and nothing is stored; what it would write, each block after `redact()`, is
appended to the user's own shadow log `<notes root>/<user>/shadow/writer-<yyyy-mm-dd>.md`, 0600,
which nothing reads back: the note store reads only `user/`, `projects/` and `sessions/`); on.

ITS OWN NONE (M3, 2026-10-05). After a compaction Qwen answered an empty or a `bash` call (the
session's tools are in the record); 0.3.34 cut that turn and counted it as NONE. `ban_tool_call`
(on by default) makes the call impossible: the engine adds a logit bias of -100 to the
`<tool_call>` token for the writer's turn (`ban_ids`), the proxy sends `tool_choice: "none"`.
`prefill` (empty by default) is text the answer starts with. The cut stays as the backstop.

THE PROXY (M3): proxy/memwriter.py runs the same turn as one request to the backend a session
used, in that backend's idle time, with this module's bookkeeping under keys `px:<session>`.

Each run is one journal row (`kind: memory_notes`, task `writer`): the trigger, the gate's answer,
the seconds, the tokens, the ops written and the refusals with their reason. Never a text. The
dashboard's Notes tab lists them (writer runs).
"""

from __future__ import annotations

import collections
import os
import re
import threading
import time

from server import memblock
from server.conductor import notes as notes_mod
from server.conductor import schema, signals

WRITER = (
    "Memory pass. The runtime asks this, not the user, and the user does not see this turn or your "
    "answer. Read the conversation above and keep what will matter in a later session: what the "
    "user asked you to keep, how they want things done, decisions taken, what went wrong and why, "
    "how to do recurring things in this project, and where the work stands now.\n\n"
    "Write each fact as one block:\n"
    "<tandem_memory op=\"add\" scope=\"project\" kind=\"gotcha\">\n"
    "title: one line, what the note says\n\n"
    "the fact in one to five sentences, with file names, commands and numbers\n"
    "</tandem_memory>\n\n"
    "Scopes: user (about the person), project (about this working directory), session (only this "
    "conversation). Kinds: rule, ruling, gotcha, runbook, program_state (add verify_by: "
    "YYYY-MM-DD), reference, registry.\n"
    "{existing}"
    "Rules: at most {max_blocks} blocks; one fact each; never a password, key or token; nothing "
    "that was true only for one message. This turn has no tools: write the blocks as plain text, "
    "never as a tool call, and nothing else. If nothing is worth keeping, answer NONE.")
EXISTING = ("Notes you already have on these things (to change one, write op=\"update\" "
            "id=\"<id>\" with the new title and body, never a second note):\n{lines}\n\n")
# the gate's words, chosen on :8000's laya-english with 4 sessions that hold a fact and 4 that do not
# (2026-10-05): the first wording said no to "Remember this for the whole project: ..." at 0.77 and
# to 3 of the 4; this one says yes to the 4 and yes to one of the others (a writer turn for nothing
# costs an idle moment, a no costs the session's facts)
GATE_QUESTION = ("Do these messages from the user contain something to remember for later "
                 "sessions: a rule, a preference, a decision, or the cause and fix of a problem?")
GATE_OPTIONS = {"yes": "Yes: at least one thing to remember.",
                "no": "No: only a task or a question."}
DEFAULTS = schema.NOTES["writer"]
TRIGGERS = ("compaction", "give_up", "idle")
RUNS_MAX = 100
USER_TEXTS = 6


_CALL = re.compile(r"<function=tandem_memory>([\s\S]*?)</function>")
_PARAM = re.compile(r"<parameter=([a-z_]+)>\n?([\s\S]*?)\n?</parameter>")
_ATTRS = ("op", "scope", "kind")
_HEADS = ("title", "tags", "id", "reason", "verify_by")


def tool_blocks(text: str) -> list[str]:
    """The `<tool_call><function=tandem_memory><parameter=...>` calls in `text` (Qwen's tool
    form, written with the session's tools in view), each as the `<tandem_memory>` block it
    means: op, scope and kind in the opener; title, tags, id, reason, verify_by as head lines; the
    `body` (or `content`, `text`) parameter as the body. memblock.parse checks the result."""
    out = []
    for m in _CALL.finditer(text):
        p = {k: v.strip() for k, v in _PARAM.findall(m.group(1))}
        attrs = " ".join(f'{k}="{p[k]}"' for k in _ATTRS if p.get(k) and '"' not in p[k])
        heads = "".join(f"{k}: {' '.join(p[k].split())}\n" for k in _HEADS if p.get(k))
        body = p.get("body") or p.get("content") or p.get("text") or ""
        out.append(f"{memblock.OPEN} {attrs}>\n{heads}\n{body}\n{memblock.CLOSE}")
    return out


_FUNC = re.compile(r"<function=([^>\s]*)>")
_OPEN_CALL = re.compile(r"<tool_call>([\s\S]*?)(</tool_call>|$)")


def foreign_call(text: str) -> str | None:
    """The first tool call in the writer's answer that is not a `tandem_memory` call: its
    function's name, "empty" for a call closed with no function; None while every call so far is
    a `tandem_memory` one (or still open before its function). The writer's turn stops there:
    the session's tools are in the record, and after a compaction Qwen answered with an empty or
    a `bash` call (2026-10-05, MEM2's measurement), which writes nothing however long it runs."""
    for m in _OPEN_CALL.finditer(text or ""):
        if _FUNC.search(m.group(1)) is None and m.group(2):
            return "empty"
    # every function, with or without its opener (a tokenizer may drop `<tool_call>` as special)
    for f in _FUNC.finditer(text or ""):
        if f.group(1) != "tandem_memory":
            return f.group(1) or "empty"
    return None


TOOL_CALL_TOKEN = "<tool_call>"
BAN_BIAS = -100.0


def ban_ids(tok) -> dict:
    """{token id: -100} for the `<tool_call>` token of `tok` ({} when it has no such single
    token: then nothing is banned and the cut is the only guard)."""
    conv = getattr(tok, "convert_tokens_to_ids", None)
    if conv is None:
        return {}
    try:
        i = conv(TOOL_CALL_TOKEN)
    except Exception:                                              # noqa: BLE001
        return {}
    unk = getattr(tok, "unk_token_id", None)
    if not isinstance(i, int) or isinstance(i, bool) or i < 0 or (unk is not None and i == unk):
        return {}
    return {i: BAN_BIAS}


SHADOW_DIR = "shadow"


def shadow_log(store, user: str | None, head: str, blocks: list[str]) -> str | None:
    """Append what a shadow run would write to the user's shadow log (module header): one
    heading line, then each block after `redact()`. Returns the file's path, or None."""
    if not user or not blocks:
        return None
    from server.conductor import redact as redact_mod
    try:
        d = os.path.join(store.user_dir(user), SHADOW_DIR)
        os.makedirs(d, mode=0o700, exist_ok=True)
        path = os.path.join(d, f"writer-{time.strftime('%Y-%m-%d')}.md")
        parts = [f"## {time.strftime('%H:%M:%S')} {head}\n"]
        for raw in blocks:
            clean, _ = redact_mod.redact(raw)
            parts.append(clean.strip() + "\n")
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(fd, "a") as f:
            f.write("\n".join(parts) + "\n")
        return path
    except Exception:                                              # noqa: BLE001
        return None


def answer_form(text: str, blocks: list) -> str:
    """What the writer's turn answered: `blocks`; `none` (it said NONE); `tool_call` (a call
    that is not a memory block, cut at its function); `empty`; or `text` (prose, no block)."""
    if blocks:
        return "blocks"
    t = (text or "").strip()
    if not t:
        return "empty"
    if "NONE" in t[:200]:
        return "none"
    if foreign_call(t) is not None or "<tool_call>" in t:
        return "tool_call"
    return "text"


def settings_of(notes_cfg: dict | None) -> dict:
    got = ((notes_cfg or {}).get("writer") or {})
    return {**DEFAULTS, **{k: v for k, v in got.items() if k in DEFAULTS}}


class Writer:
    """The sessions' writer bookkeeping: their recent user messages (for the gate and the
    existing-notes search; in memory only, never written), the record length last written, the
    pending trigger, and the last runs."""

    def __init__(self, memory):
        self.memory = memory
        self.lock = threading.Lock()
        self.by_conv: dict[str, dict] = {}
        self.runs: collections.deque = collections.deque(maxlen=RUNS_MAX)
        self.stats = {"runs": 0, "written": 0, "refused": 0, "gate_no": 0, "stopped": 0,
                      "skipped": 0}

    # ---------------------------------------------------------------- what the requests tell it
    def seen(self, conv: str | None, ctx: dict | None, user_text: str | None) -> None:
        """A request of the session `conv` (the engine's conversation id) with memory context
        `ctx` (Plan.memory)."""
        if not conv or not ctx or ctx.get("mode") not in ("on", "shadow") or ctx.get("read_only"):
            return                          # SRV-173: `X-Tandem-Memory: read` writes nothing
        with self.lock:
            s = self.by_conv.setdefault(conv, {"done_at": 0, "due": None, "texts": collections.deque(
                maxlen=USER_TEXTS)})
            s.update(user=ctx["user"], project=ctx.get("project"), session=ctx["session"],
                     mode=ctx["mode"], t=time.time())
            if user_text and user_text.strip() and ctx.get("from_user"):
                s["texts"].append(user_text.strip()[:1000])
            while len(self.by_conv) > 256:
                self.by_conv.pop(next(iter(self.by_conv)))

    def mark(self, conv: str | None, trigger: str) -> None:
        with self.lock:
            s = self.by_conv.get(conv or "")
            if s is not None and trigger in TRIGGERS:
                s["due"] = trigger

    def state(self, conv: str | None) -> dict | None:
        with self.lock:
            s = self.by_conv.get(conv or "")
            return None if s is None else dict(s, texts=list(s["texts"]))

    # ---------------------------------------------------------------- the turn's parts
    def due(self, conv: str, at: int, idle_s: float, cfg: dict) -> str | None:
        """The trigger the session's record of `at` tokens is due for after `idle_s` idle
        seconds, or None."""
        s = self.state(conv)
        if s is None:
            return None
        if s["due"] in ("compaction", "give_up") and idle_s >= float(cfg["due_idle_s"]):
            return s["due"]
        if idle_s >= float(cfg["idle_s"]) and at - int(s["done_at"]) >= int(cfg["min_new_tokens"]):
            return "idle"
        return None

    def digest(self, conv: str) -> str:
        s = self.state(conv) or {}
        return signals.digest("\n\n".join(s.get("texts") or []), 2000, None)

    def instructions(self, conv: str, cfg: dict) -> tuple[str, list[str]]:
        """The writer's user turn: WRITER with the notes that match the session (their ids)."""
        s = self.state(conv) or {}
        store = self.memory.store
        hits = store.search(s.get("user") or "", self.digest(conv), project=s.get("project"),
                            session=s.get("session"), k=int(cfg["existing"])) \
            if s.get("user") else []
        lines = "\n".join(f"- [{n.id}] {n.kind} ({n.scope}) · {n.title}: "
                          f"{' '.join((n.body or '').split())[:300]}" for n, _ in hits)
        text = WRITER.replace("{existing}", EXISTING.replace("{lines}", lines) if hits else "") \
            .replace("{max_blocks}", str(int(cfg["max_blocks"])))
        return text, [n.id for n, _ in hits]

    def gate(self, router, conv: str) -> tuple[str | None, str]:
        """Laya's yes or no on the session's digest: (answer or None, how)."""
        text = self.digest(conv)
        if not text.strip():
            return None, "no_text"
        try:
            _, router_doc = router.store.peek("router")
            _, roles = router.store.peek("roles")
            clf = router_doc["classifier"]
            stages = [dict(st, enabled=True) for st in clf["stages"] if st["kind"] == "decider"]
            if not stages:
                return None, "no_decider"
            doc = dict(router_doc, classifier=dict(clf, stages=stages))
            ans, st = router.cascade(text, GATE_OPTIONS, doc, roles, question=GATE_QUESTION,
                                     accept=0.6, qid="memory_gate")
        except Exception as exc:                                   # noqa: BLE001
            return None, f"error:{type(exc).__name__}"
        if ans is None:
            return None, (st[-1].get("error") or "below") if st else "none"
        return ans.label, "decider"

    def apply(self, conv: str, text: str, cfg: dict, trigger: str, at: int, info: dict) -> dict:
        """The turn's output: each closed block written (on) or counted (shadow); the run
        journaled and kept for the dashboard. A `tandem_memory` tool call counts as its block
        (the session's tools are in the record, and on the box Qwen wrote the call form)."""
        s = self.state(conv) or {}
        _, blocks = memblock.strip_blocks(text or "")
        calls = tool_blocks(text or "")
        if calls:
            info = dict(info, tool_form=len(calls))
            blocks = blocks + calls
        ops, refused, would = [], [], []
        mode = cfg["mode"]
        for raw in blocks[:int(cfg["max_blocks"])]:
            try:
                op = memblock.parse(raw)
                if op["op"] == "propose":
                    raise notes_mod.Refused("propose", "the writer does not propose packages")
                if mode != "on":
                    ops.append({"op": op["op"], "kind": op.get("kind"), "scope": op.get("scope"),
                                "stored": False})
                    would.append(raw)
                    continue
                res = self.memory.store.write(s["user"], op, writer="qwen-writer",
                                              project=s.get("project"), session=s.get("session"),
                                              request_id=None, from_user=False)
                ops.append({k: res.get(k) for k in ("op", "id", "kind", "scope", "revised")
                            if res.get(k) is not None})
            except memblock.BlockError as exc:
                refused.append(exc.why)
            except notes_mod.Refused as exc:
                refused.append(exc.why)
        if len(blocks) > int(cfg["max_blocks"]):
            refused += ["too_many"] * (len(blocks) - int(cfg["max_blocks"]))
        with self.lock:
            st = self.by_conv.get(conv)
            if st is not None:
                st["done_at"] = at
                st["due"] = None
        self.stats["runs"] += 1
        self.stats["written"] += sum(1 for o in ops if o.get("stored", True))
        self.stats["refused"] += len(refused)
        # an empty answer or a tool call that is not a memory block says what NONE says: nothing
        # to keep (the run's `answer` keeps which; MEM2's open item, 2026-10-05)
        form = answer_form(text, blocks)
        if would and s.get("user"):
            where = " ".join(f"{k} {info[k]}" for k in ("backend", "model") if info.get(k))
            if shadow_log(self.memory.store, s.get("user"),
                          f"{trigger} {where} session {str(s.get('session') or '')[:24]}".replace(
                              "  ", " "), would):
                info = dict(info, shadow_log=len(would))
        run = dict(info, trigger=trigger, mode=mode, ops=ops, refused=refused,
                   none=form in ("none", "empty", "tool_call"), answer=form,
                   session=s.get("session"), user=s.get("user"), ts_ms=int(time.time() * 1000))
        self.runs.appendleft(run)
        self._journal(run)
        return run

    def settle(self, conv: str, at: int) -> None:
        """The session counts as written at `at` (its pending trigger dropped): a run that will
        not happen for this record (the proxy's writer off for the backend, a cap reached)."""
        with self.lock:
            st = self.by_conv.get(conv)
            if st is not None:
                st["done_at"], st["due"] = at, None

    def skipped(self, conv: str, trigger: str, why: str, at: int, info: dict | None = None) -> dict:
        """A run that did not write: the gate said no, it was stopped, or it failed."""
        s = self.state(conv) or {}
        if why == "gate_no":
            self.stats["gate_no"] += 1
            with self.lock:
                st = self.by_conv.get(conv)
                if st is not None:
                    st["done_at"], st["due"] = at, None
        elif why.startswith("stopped") or why.startswith("yielded"):
            self.stats["stopped"] += 1
        else:
            self.stats["skipped"] += 1
        run = dict(info or {}, trigger=trigger, skipped=why, session=s.get("session"),
                   user=s.get("user"), ts_ms=int(time.time() * 1000), ops=[], refused=[])
        self.runs.appendleft(run)
        self._journal(run)
        return run

    def _journal(self, run: dict) -> None:
        j = self.memory.journal
        if j is None:
            return
        j.note("memory_notes", "writer", decision=run.get("skipped") or (
            "wrote" if run.get("ops") else "none"),
            detail={k: v for k, v in run.items() if k not in ("session",)},
            session_key=run.get("session"))
