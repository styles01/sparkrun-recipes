"""The conductor's task router: which profile, which kind of task, and what that turn gets.

Two halves per request, split by the engine lock.

`Router.plan()` runs BEFORE the lock (it may ask the decider, or Qwen through this server's own port,
and must never hold the lock while it does):

  1. Resolve the profile (SRV-117): the server serves one model, `TandemLLM`. Every model name
     resolves to it -- the raw engine's old name is a hidden alias, so no client breaks -- and goes
     to `X-Tandem-Profile`, else the profile mapped to the client kind, else the default.
     `X-Tandem: raw|off|shadow` lowers the mode and never raises it; `X-Tandem: raw` (or the
     locked `raw` / `bench` profile by header) is how benchmarks reach the release's path.
  2. Decide the class of the turn (SRV-129). Structure first: a turn that only carries tool
     results keeps the session's class; opencode's compaction request is `summarize`. A new user
     message goes through the classifier stages (the decider, then Qwen) until one answers at or above
     its threshold. A session changes class only at the stickiness threshold (0.8). Nobody sure:
     the session's class, else the profile's default. No word list decides anything.
  3. Turn the class into a policy: an effort level (budget, thinking on or off, a note), the
     packages (a second choice question when a class has several candidates), and rules.
     Each decision type is `off`, `shadow` or `on` per profile; shadow is journaled only.
     A class with an `effort_by_size` map (SRV-157) picks its effort from the session's size
     (the tokens it added, the files its calls named, the gates a checker named, the turns) under
     the class's effort as the ceiling; the size only rises within a task, and a compaction starts
     it over. The profile's `thinking.by_size` says off, shadow or on.
     The level's budget is a real cap (medium 8,192, high 16,384, xhigh none below the backstop;
     a profile's `thinking.budgets` overrides a level): the server closes the reasoning there
     with the session monitor's hidden close (SP1), so the client sees only the closer.
     A turn whose newest message is a tool result may think at its own level (F2: the router's
     `structural.tool_turn_effort`, a profile's `thinking.tool_turn_effort` over it; `inherit`,
     the default, is the class's): its switch and cap, journaled as `effort_source: tool_turn`.

The classifier's digest (SRV-153) is sized per stage, the decider's to its own `limits()` and
Qwen's to `digest_chars`, head and tail, and carries one context line the router writes from what
it knows: the previous class, the last tools, the file types, the checker's status, a compaction.

`Plan.apply_prompt()` runs UNDER the lock, on the rendered prompt ids: it replays the session's
earlier notes at their positions (server/conductor/inject.py) and adds this turn's note when the
active block is not already in the prompt. `Plan.finish()` writes the outcome.

Everything a plan did is in its journal row (server/conductor/decisions.py).
"""

from __future__ import annotations

import collections
import hashlib
import re
import threading
import time

from server import argloop, thinking
from server.conductor import incontext, inject, libsearch, modelfit, outcomes, schema, signals, transcripts
from server.conductor import memory as memory_mod
from server.conductor import packages as packages_mod
from server import session_tokens as session_tokens_mod
from server.conductor.classify import ASKING, PARENT, Answer, Qwen, decider_for
from server.conductor.store import ConfigError
from engine.settings import SETTINGS as _S

#: FX5: Open WebUI's live working directory at the end of `run_command`'s description
#: (open_webui/utils/tools.py), and what the turn's note says instead (`Router._client_cwd`)
CWD_LINE = re.compile(r"\n\nThe current working directory is: ([^\n]+)\s*\Z")
CWD_NOTE = "The current working directory of the terminal (the run_command tool) is: {path}"
CLIENT_CWD = (_S.get("CLIENT_CWD") or "note").strip().lower()
#: FX5b: `parts`: a class's note leaves out the parts a live note already holds with the same text
#: (`Plan._fresh_block`); `block`: the whole note whenever its key is new
NOTE_DEDUP = (_S.get("NOTE_DEDUP") or "parts").strip().lower()

#: FX17: `on`: a note whose prefix hash broke stays live when its user message still ends with
#: the same client tokens (`inject.locate_anchor`); `off`: the prefix hash alone (0.4.8-test)
NOTE_ANCHOR = (_S.get("NOTE_ANCHOR") or "on").strip().lower()
_MARKS: dict = {}


def _marks(tok):
    """The template's marks for `inject.message_anchor`, once per tokenizer."""
    key = id(tok)
    if key not in _MARKS:
        try:
            im_s = inject.special_id(tok, "<|im_start|>")
            user = list(tok.encode("user", add_special_tokens=False))
            _MARKS[key] = (im_s, user, inject.special_id(tok, inject.IM_END)) if (
                im_s is not None and user) else None
        except Exception:                                          # noqa: BLE001
            _MARKS[key] = None
    return _MARKS[key]
#: SRV-243: `on`: a package whose steps need a tool the request does not offer (a shell, a file
#: write) stays out of the turn, or its `fallback` goes in, and the turn says which tools it has;
#: `off`: the packages chosen as before, whatever the tools
TOOL_FIT = (_S.get("TOOL_FIT") or "on").strip().lower()
# the tool capabilities the tool-fit note is about (`_fit`)
NOTE_CAPS = frozenset({"shell", "write"})
# the package every class that writes files lists (SRV-227): `_below_build`'s "builds"
BUILD_MARK = "files-care"

#: SRV-225: the `h:` heads whose conversations `Router._generation` tells apart, and the
#: conversations kept per head (the oldest go first)
GEN_HEADS_MAX = 256
GEN_PER_HEAD = 8
#: `off`: every request with the same head is one session, as before (the A/B)
SESSION_GENERATIONS = (_S.get("SESSION_GENERATIONS") or "on").strip().lower()

COMPOSITE = "TandemLLM"


class _Skip(Exception):
    """A plan step the model fit leaves out."""


RANK = {"raw": 0, "off": 0, "shadow": 1, "on": 2}
SESSIONS_MAX = 64
TEXT_CACHE_MAX = 512
EFFORT_ORDER = ("off", "low", "medium", "high", "xhigh")
NONE_OPTION = "none"


def _lower(a: str, b: str | None) -> str:
    """The lower of two modes; `off` and `raw` rank the same."""
    if b is None:
        return a
    return a if RANK.get(a, 0) <= RANK.get(b, 0) else ("raw" if b in ("off", "raw") else b)


def _header(headers, name: str) -> str:
    try:
        return (headers.get(name) or "").strip()
    except Exception:                                              # noqa: BLE001
        return ""


class Resolution:
    __slots__ = ("profile", "version", "doc", "mode", "source", "requested")

    def __init__(self, profile, version, doc, mode, source, requested):
        self.profile, self.version, self.doc = profile, version, doc
        self.mode, self.source, self.requested = mode, source, requested

    def headers(self) -> tuple:
        """`X-Tandem-Profile`; `X-Tandem` itself goes out with every completion response
        (server/app.py's `Handler.send_response`), set from this resolution's mode."""
        return (("X-Tandem-Profile", self.profile),)


def is_alias(model: str, raw_model: str) -> bool:
    """The raw engine's old name: served as `TandemLLM`, never listed (SRV-117, 2026-10-01)."""
    return bool(model) and model.lower() == str(raw_model or "").lower() and \
        model.lower() != COMPOSITE.lower()


def _tools_profile(store) -> str | None:
    """SRV-621: the settings' `tools_profile` when the store holds it and it keeps no
    memory (a keyless LAN client is the box's owner to the memory notes, so the rule never sends
    one to a profile that reads or writes them); None: the rule does not apply."""
    try:
        name = schema.tools_profile(store.peek("settings")[1])
        if not name or not store.exists("profiles", name):
            return None
        return name if schema.profile_memory(store.peek("profiles", name)[1]) == "off" else None
    except Exception:                                              # noqa: BLE001
        return None


def resolve(store, model: str, raw_model: str, headers, client_kind: str,
            global_mode: str | None = None, offers_tools: bool = False) -> Resolution:
    """The profile and mode of one request; `ConfigError(400)` for an unknown profile header.
    `model` and `raw_model` are kept in the signature for the callers; neither picks the profile.
    The order: the `X-Tandem-Profile` header, the profile that maps the client kind, then
    (SRV-621) `settings.tools_profile` for a request that offers function tools
    (`offers_tools`, signals.offers_tools), then `settings.default_profile`."""
    requested = _header(headers, "X-Tandem-Profile")
    forced = _header(headers, "X-Tandem").lower()
    # One model (the owner, 2026-10-01): `model` is not consulted. `TandemLLM`, the raw engine's old
    # name (a hidden alias) and any other name resolve the same way; raw is a header, not a name.
    if requested:
        if not store.exists("profiles", requested):
            raise ConfigError(400, f"no conductor profile {requested!r}")
        name, source = requested, "header"
    else:
        name, source = None, "default"
        for n in store.names():
            if n in schema.LOCKED:
                continue
            if client_kind in store.peek("profiles", n)[1].get("clients", []):
                name, source = n, "client"
                break
        if name is None and offers_tools:
            tp = _tools_profile(store)
            if tp is not None:
                name, source = tp, "tools"
        if name is None:
            name = store.peek("settings")[1]["default_profile"]
    v, doc = store.peek("profiles", name)
    mode = doc.get("mode", "raw")
    if forced in ("raw", "off"):
        mode = "raw"
    elif forced == "shadow":
        mode = _lower(mode, "shadow")
    mode = _lower(mode, global_mode)
    return Resolution(name, v, doc, mode, source, requested or None)


class Injection:
    """One note in a session: where it sits in the client's prompt and what it is."""

    __slots__ = ("pos", "hash", "ids", "text", "parts", "kinds", "key", "turn", "at", "row",
                 "shas", "anchor")

    def __init__(self, pos, hsh, ids, text, parts, kinds, key, turn, row, shas=None, anchor=None):
        self.pos, self.hash, self.ids, self.text = pos, hsh, ids, text
        self.parts, self.kinds, self.key, self.turn, self.row = parts, kinds, key, turn, row
        self.shas = shas          # each part's text hash, in `parts`' order (`Plan._fresh_block`)
        self.anchor = anchor      # FX17: `inject.message_anchor`, the note's place by its message
        self.at = time.time()

    def moved_to(self, pos: int, hsh: str) -> "Injection":
        """FX17: the same note at a new insertion point (the client changed the prompt before
        its message without touching the message's end)."""
        n = Injection(pos, hsh, self.ids, self.text, self.parts, self.kinds, self.key, self.turn,
                      self.row, self.shas, self.anchor)
        n.at = self.at
        return n


class Session:
    __slots__ = ("key", "cls", "conf", "source", "since", "turns", "user_hash", "last_row",
                 "notes", "history", "package_pick", "lock", "compacted", "size_tier",
                 "prompt_first", "prompt_last", "evidence", "acts", "cwd")

    def __init__(self, key):
        self.key = key
        self.cls = self.conf = self.source = self.user_hash = self.last_row = None
        self.compacted = False             # a compaction request came; the next class says so
        self.size_tier = None              # the largest size tier this task reached (SRV-157)
        self.prompt_first = self.prompt_last = None   # prompt tokens: first and latest turn
        self.since = time.time()
        self.turns = 0
        self.notes: list[Injection] = []
        self.history: collections.deque = collections.deque(maxlen=64)
        self.package_pick: dict = {}
        self.lock = threading.Lock()
        self.evidence: outcomes.Evidence | None = None   # J1: the snapshots' state
        self.acts: collections.deque = collections.deque(maxlen=64)  # J1: the notes' times
        self.cwd: tuple | None = None      # FX5: (path, note key) of the last working-directory note


def session_key(body: dict, headers, sh: signals.Shape) -> str:
    """A conversation id if the client sends one (opencode sends `x-session-id`; Open WebUI sends
    `X-OpenWebUI-Chat-Id` with ENABLE_FORWARD_USER_INFO_HEADERS, or as a connection header with the
    `{{CHAT_ID}}` placeholder: docs/conductor setup), else a hash of the conversation's head -- its
    messages up to and including the first user message -- which every turn sends again
    (`signals.Shape.first_hash`). Two chats that open with the same message share that hash;
    `Router._generation` tells them apart (SRV-225)."""
    for k in ("conversation_id", "session_id"):
        v = body.get(k)
        if isinstance(v, str) and v:
            return "c:" + v[:120]
    meta = body.get("metadata")
    if isinstance(meta, dict):
        v = meta.get("conversation_id") or meta.get("session_id") or meta.get("chat_id")
        if isinstance(v, str) and v:
            return "c:" + v[:120]
    for h in ("x-session-id", "X-Conversation-Id", "X-OpenWebUI-Chat-Id"):
        v = _header(headers, h)
        if v:
            return "c:" + v[:120]
    return "h:" + (sh.first_hash or "empty")


def _qwen_prompt(router: dict, question: str | None, qid: str, about: str | None,
                 listing: str) -> str:
    """The Qwen stage's system prompt: the class question's (`qwen_prompt`) for the class and, by
    default, for the package question too (its options as the classes); a router's own
    `package_prompt` (FX6) for the package question, with the routed class and the question; any
    other question as it is, with its options."""
    clf = router["classifier"]
    own = schema.package_prompt(router) if qid == "packages" else None
    if question is None or (qid == "packages" and own is None):
        return clf["qwen_prompt"].replace("{classes}", listing)
    if own is not None:
        return (own.replace("{class}", about or "not known")
                .replace("{question}", question).replace("{classes}", listing))
    return f"{question} Answer with the id of the one option that fits best.\n\nOptions:\n{listing}"


def _tool_line(t: dict) -> str:
    """SRV-615: an offered tool for the keep-out question: its name and its description's first
    sentence."""
    fn = t.get("function") if isinstance(t.get("function"), dict) else t
    name = str(fn.get("name") or "?")
    desc = str(fn.get("description") or "").strip()
    first = re.split(r"(?<=[.!?])\s", desc, maxsplit=1)[0][:160] if desc else ""
    return f"{name} ({first})" if first else name


def _about(cls: str, classes: dict) -> str:
    """The routed class as the package prompt names it: `id: label, description`."""
    c = classes.get(cls) or {}
    return f"{cls}: {c.get('label', cls)}, {c.get('description', '')}".rstrip(", ")


def _code_guard(ans: Answer, text: str, options: dict) -> Answer:
    """SRV-189: a code class below `CODE_GUARD.accept` for a request that names nothing of code and
    names a piece of prose becomes the stage's best class outside the code classes, at its own
    probability (journaled as `guard`). Everything else passes as the stage answered."""
    g = schema.CODE_GUARD
    if ans.error is not None or ans.label not in schema.CODE_CLASSES or \
            ans.confidence >= float(g["accept"]):
        return ans
    if re.search(g["cues"], text or "") or not re.search(g["prose"], text or ""):
        return ans
    rest = {k: v for k, v in (ans.probs or {}).items()
            if k in options and k not in schema.CODE_CLASSES}
    if not rest:
        return ans
    best = max(rest, key=rest.get)
    out = Answer(ans.stage, best, rest[best], ans.probs, ans.ms, model=ans.model)
    out.guard = {"was": ans.label, "confidence": round(ans.confidence, 4), "to": best,
                 "why": "no_code_cue"}
    return out


class Plan:
    """What the router decided for one request; `active` is False for raw requests."""

    def __init__(self, res: Resolution):
        self.res = res
        self.active = False
        self.row: dict | None = None
        self.thinking_off = False                    # close the generation prompt's <think>
        self.client_off = False                      # the client turned thinking off (0.3.40)
        self.argloop_retry: dict | None = None       # a retry after stopped looping calls
        self.budget: int | None = None               # None = leave the request's budget
        self.budget_source = "server"
        self.block: dict | None = None               # the note for this turn, if one is due
        self.session: Session | None = None
        self.router_cfg: dict | None = None
        self.decisions: dict = {}
        self.injected: list[dict] = []
        self.clean_ids: list[int] | None = None
        self.changed = False
        self.compaction = False                      # opencode's compaction request
        # S1: a client's side request (its title or summary call): the shape that matched, the
        # output cap and the thinking; None for every other request
        self.side: dict | None = None
        # SRV-137: what the transcript writer keeps of this request -- full, counters or None --
        # and the body it reads its new messages from (a reference, read after the request ended)
        self.transcript: str | None = None
        self.body: dict | None = None
        # the in-context stage, when it is due: Qwen decides the class under the engine lock
        # (`Router.decide_in_context`) and the rest of the plan waits for it
        self.context: dict | None = None
        # ENG-181: the client's insertion point and the notes spliced before it, for the row the
        # probe snapshots read (`probe_row`)
        self._probe_at: tuple | None = None
        # C1: every note in the served prompt as (start, end, kinds, package names), for the disk
        # tier's entries (engine/disk_cache.py names a prefix that holds a package by it)
        self.spans: list[tuple] = []
        # J1: the turn's snapshot (the transcript writer stores it), its outcome, the notes this
        # plan adds besides the class's block (an outcome's note, the cards; each injected once,
        # then replayed), and the acts it took (the monitor's caps count them)
        self.snapshot: dict | None = None
        self.outcome: dict | None = None
        self.judge: list[dict] | None = None          # Laya's questions, after the request
        self.ask: dict | None = None                  # Qwen's, in place before the prefill
        self.extras: list[dict] = []
        self.acts: list[float] = []
        # R1: the plan is being decided under the engine lock (the in-context stage), where no
        # question may go to Qwen through the server's own port
        self.locked = False
        # 2026-10-05 memory (server/conductor/memory.py): the request's user, project and
        # session for the notes, and what its blocks wrote; None with the profile's memory off
        self.user: str | None = None
        self.caps: dict | None = None
        self.memory: dict | None = None
        # SRV-243: what the request's tools can do (signals.tool_caps) and their names
        self.tool_caps: frozenset = frozenset()
        self.tool_names: list[str] = []
        # SRV-621: the profile's `prompt` is `client` (nothing of the plan goes in)
        self.prompt_client = False
        self.t0 = time.perf_counter()

    # ---------------------------------------------------------------- what app.py calls
    def headers(self) -> tuple:
        return self.res.headers()

    def think_budget(self, budget: int) -> int:
        """The request's reasoning budget after the plan: the client's own number always wins."""
        if self.budget is None:
            return budget
        return self.budget

    @property
    def client_word(self) -> str:
        """What a client's effort word is under this profile (the owner, 2026-10-03 12:30):
        `ceiling` (the default), the router may think less than the word's cap and never more;
        `ignore`, the router's level alone (opencode sends `medium` on every turn)."""
        th = ((getattr(self.res, "doc", None) or {}).get("thinking") or {})
        return th.get("client_word", "ceiling")

    @property
    def decides_effort(self) -> bool:
        """The router sets this turn's effort level, so a client's effort word yields to it."""
        return self.active and self.decisions.get("effort") == "on"

    def apply_prompt(self, prompt, tok, in_think: bool):
        """The rendered prompt with the session's notes replayed, this turn's note added and, when
        the plan turned thinking off, the generation prompt's open `<think>` closed. Returns
        (the prompt to prefill, whether it ends inside `<think>`); the same objects when nothing
        changes. A failure here serves the prompt as rendered and says so in the journal row."""
        if not self.active or self.session is None or self.prompt_client:
            # SRV-621: `prompt: client` serves the rendered prompt, the raw path's ids
            return prompt, in_think
        if not self.session.notes and self.block is None and not self.thinking_off and \
                not self.extras:
            return prompt, in_think
        try:
            return self._apply(prompt, tok, in_think)
        except Exception as exc:                                   # noqa: BLE001
            self.row["extra"].setdefault("errors", []).append(f"apply:{type(exc).__name__}")
            print(f"[conductor] the prompt was served as rendered: {type(exc).__name__}: {exc}",
                  flush=True)
            return prompt, in_think

    def _apply(self, prompt, tok, in_think: bool):
        import torch
        sess, cfg = self.session, self.router_cfg or {}
        inj_cfg = cfg.get("injection") or {}
        ids = prompt.tolist()
        self.clean_ids = ids
        tail_ids, closed_ids = None, None
        if self.thinking_off and in_think:
            # the template's own shape for `enable_thinking: false`, spliced at the tail instead of
            # rendered: the head (and its reasoning_effort sentence) stays token for token, and
            # the next turn renders the answer's empty think block the same way
            for open_, off in ((inject.THINK_OPEN, inject.THINK_OFF),
                               (inject.THINK_OPEN_BARE, inject.THINK_OFF_BARE)):
                tail_ids = list(tok(open_, add_special_tokens=False).input_ids)
                closed_ids = list(tok(off, add_special_tokens=False).input_ids)
                if ids[-len(tail_ids):] == tail_ids:
                    break
            else:
                tail_ids = None
                self.row["extra"].setdefault("notes", []).append("thinking_off:no_open_tail")
        im_end = inject.special_id(tok, inject.IM_END)
        at = inject.insertion_point(ids, im_end)
        notes, adds = [], []
        marks = _marks(tok)
        if at is None:
            if sess.notes or self.block is not None or self.extras:
                self.row["extra"].setdefault("errors", []).append("no_insertion_point")
        else:
            with sess.lock:
                live, hashes = self._live(ids, at, marks=marks, im_end=im_end)
                cap = int(inj_cfg.get("max_note_tokens", 8000))
                keys = {n.key for n in live}
                block = self.block
                if block is not None and block["key"] not in keys:
                    block = self._fresh_block(block, live)
                if block is not None and block["key"] not in keys:
                    text = cfg["injection"]["wrapper"].replace("{text}", block["text"])
                    nids = list(tok(text, add_special_tokens=False).input_ids)
                    if len(nids) > cap:
                        self.row["extra"].setdefault("errors", []).append(
                            f"note_too_long:{len(nids)}")
                    else:
                        adds.append((Injection(at, hashes[at], nids, text, block["parts"],
                                               block["kinds"], block["key"], sess.turns,
                                               self.row["id"], shas=block.get("shas"),
                                               anchor=inject.message_anchor(ids, at, marks)),
                                     block["reason"]))
                # J1: the outcome's note and the cards, after the class's block; a card goes in
                # only while no live note carries the same card (the first turn, after a
                # compaction), an outcome's note once
                parts = {(p["kind"], p["name"]) for n in live for p in n.parts}
                for x in self.extras:
                    if x["key"] in keys or (x.get("unless") and tuple(x["unless"]) in parts):
                        continue
                    text = (x.get("wrapper") or cfg["injection"]["wrapper"]).replace(
                        "{text}", x["text"])
                    nids = list(tok(text, add_special_tokens=False).input_ids)
                    if len(nids) > cap:
                        self.row["extra"].setdefault("errors", []).append(
                            f"note_too_long:{len(nids)}")
                        continue
                    adds.append((Injection(at, hashes[at], nids, text, x["parts"], x["kinds"],
                                           x["key"], sess.turns, self.row["id"],
                                           anchor=inject.message_anchor(ids, at, marks)),
                                 x["reason"]))
                    parts |= {(p["kind"], p["name"]) for p in x["parts"]}
                notes = live + [n for n, _ in adds]
                sess.notes = notes
            self._probe_at = (at, notes)
        if not notes and tail_ids is None:
            return prompt, in_think
        out, starts = inject.splice(ids, [(n.pos, n.ids) for n in notes])
        self.spans = [(starts[i], starts[i] + len(n.ids), sorted(n.kinds),
                       [p["name"] for p in (n.parts or []) if p.get("kind") == "package"])
                      for i, n in enumerate(notes)]
        if notes:
            replayed = notes[:len(notes) - len(adds)]
            self.row["replayed"] = {"count": len(replayed),
                                    "tokens": sum(len(n.ids) for n in replayed),
                                    "positions": [n.pos for n in replayed]}
        for i, (new, reason) in enumerate(adds):
            self.injected.append({"pos": new.pos, "at": starts[len(notes) - len(adds) + i],
                                  "tokens": len(new.ids), "text": new.text, "parts": new.parts,
                                  "kinds": sorted(new.kinds), "reason": reason})
            self.row["injected"] = self.injected
            self.row["acted"] = 1
        if tail_ids is not None:
            out = out[:len(out) - len(tail_ids)] + closed_ids
            in_think = False
            self.row["acted"] = 1
        self.changed = True
        self.row["changed"] = 1
        return torch.tensor(out, dtype=prompt.dtype, device=prompt.device), in_think

    def _fresh_block(self, block: dict, live: list) -> dict | None:
        """The class's note without the parts a live note of the session already holds with the
        same text (kind, name and text hash): None when every part is live. A user message whose
        task skill differs from the last one's used to bring the whole note again, the general
        packages included (about 4,800 tokens each time), and the third copy pushed the replayed
        notes over `max_replay_tokens`: the oldest note went, the prompt changed at its position
        and the turn read the conversation after it again (FX5b, 2026-10-06: 35,442 tokens, 49 s,
        in an Open WebUI agent chat). `QWEN38_NOTE_DEDUP=block` keeps the whole note."""
        items = block.get("items")
        if NOTE_DEDUP != "parts" or not items:
            return block
        held = {(p.get("kind"), p.get("name"), h) for n in live
                for p, h in zip(n.parts or (), getattr(n, "shas", None) or ()) if h}
        fresh = [it for it in items if (it["part"]["kind"], it["part"]["name"],
                                        it["sha"]) not in held]
        if len(fresh) == len(items):
            return block
        self.row["extra"].setdefault("notes", []).append(
            "parts_live:" + ",".join(it["part"]["name"] for it in items if it not in fresh))
        if not fresh:
            return None
        key = hashlib.sha256((block["key"] + repr([(it["part"]["kind"], it["part"]["name"],
                                                   it["sha"]) for it in fresh])
                              ).encode()).hexdigest()[:20]
        return dict(block, key=key, text="\n\n".join(it["text"] for it in fresh),
                    parts=[it["part"] for it in fresh], shas=[it["sha"] for it in fresh],
                    kinds={it["kind"] for it in fresh}, items=fresh)

    def _live(self, ids: list[int], at: int, marks=None, im_end=None) -> tuple[list, dict]:
        """The session's earlier notes this prompt still carries the text before (their
        decision types on, the newest within `max_replay_tokens`), and the prefix hashes.
        Called with the session's lock held.

        FX17: a note whose prefix hash broke is kept when its message is still there: the
        ordinal-th user message ends with the same client tokens (`inject.locate_anchor`); it
        moves to that message's new end. Open WebUI's retrieval block leaving an earlier message
        (the 2026-10-08 12:14:30 turn: 4 notes dropped, 8,306 tokens put in again) no longer
        drops the session's notes. `QWEN38_NOTE_ANCHOR=off` is the prefix hash alone."""
        inj_cfg = (self.router_cfg or {}).get("injection") or {}
        sess = self.session
        positions = [n.pos for n in sess.notes if n.pos <= at] + [at]
        hashes = inject.hashes_at(ids, positions)
        live = []
        moved = 0
        for n in sess.notes:
            if n.pos > at or hashes.get(n.pos) != n.hash:
                if NOTE_ANCHOR == "on" and n.anchor is not None:
                    p = inject.locate_anchor(ids, n.anchor, marks, im_end)
                    if p is not None and p <= at:
                        if p not in hashes:
                            hashes.update(inject.hashes_at(ids, [p]))
                        n = n.moved_to(p, hashes[p])
                        moved += 1
                    else:
                        continue                  # the prompt parted before the note
                else:
                    continue                      # the prompt parted before the note
            if not inj_cfg.get("replay", True):
                continue
            if any(self.decisions.get(k) != "on" for k in n.kinds):
                continue                          # its decision type was turned off
            live.append(n)
        budget = int(inj_cfg.get("max_replay_tokens", 16000))
        while live and sum(len(n.ids) for n in live) > budget:
            live.pop(0)                           # the oldest goes first
        if moved and self.row is not None:
            self.row["extra"]["notes_moved"] = moved
        return live, hashes

    def replayed_ids(self, ids: list[int], tok) -> list[int]:
        """The client's prompt with the session's earlier notes replayed where they sit, and no
        note of this turn: the prompt the resident prefix holds up to the last message, which
        the in-context stage asks its question after. Nothing in the session changes."""
        if self.session is None or not self.session.notes:
            return list(ids)
        at = inject.insertion_point(ids, inject.special_id(tok, inject.IM_END))
        if at is None:
            return list(ids)
        with self.session.lock:
            live, _ = self._live(ids, at, marks=_marks(tok),
                                 im_end=inject.special_id(tok, inject.IM_END))
        return inject.splice(ids, [(n.pos, n.ids) for n in live])[0] if live else list(ids)

    def probe_row(self, prompt, tok) -> int | None:
        """The served prompt's position of the last message's last token, after
        `apply_prompt` (server/probe.py): from the insertion point and the notes spliced before
        it, or, for a prompt the plan did not change, from the prompt's own tail."""
        if self._probe_at is not None:
            return inject.content_row(*self._probe_at)
        tail = prompt[-64:].tolist()
        at = inject.insertion_point(tail, inject.special_id(tok, inject.IM_END))
        return None if at is None else inject.content_row(int(prompt.numel()) - len(tail) + at)

    def clean(self, prompt_ids: list[int]) -> list[int]:
        """The prompt as the client sent it, without notes: what the suffix store may keep."""
        return self.clean_ids if self.clean_ids is not None else prompt_ids

    def finish(self, rec, journal) -> None:
        """The turn ended: its outcome goes into the row, the row into the journal, and its prompt
        tokens into the session's size (a compaction request's prompt is not the session's)."""
        if self.row is None:
            return
        pt = getattr(rec, "prompt_tokens", None)
        if self.session is not None and pt and not self.compaction:
            with self.session.lock:
                if self.session.prompt_first is None:
                    self.session.prompt_first = int(pt)
                self.session.prompt_last = int(pt)
        self.row["outcome"] = {
            "status": getattr(rec, "status", None), "finish_reason": getattr(rec, "finish_reason", None),
            "tool_calls": int(getattr(rec, "tool_calls", 0) or 0),
            "tool_names": list(getattr(rec, "tool_names", None) or [])[:8],
            "completion_tokens": getattr(rec, "completion_tokens", None),
            "reasoning_tokens": getattr(rec, "reasoning_tokens", None),
            "prompt_tokens": getattr(rec, "prompt_tokens", None),
            "ttft_ms": round(rec.ttft_ms, 1) if getattr(rec, "ttft_ms", None) else None,
            "total_ms": round(rec.total_ms, 1) if getattr(rec, "total_ms", None) else None,
        }
        ui = getattr(rec, "states", None)
        if ui is not None and ui.log:
            self.row["extra"]["states"] = [
                {"kind": e["kind"], "text": e["text"], "at_ms": e["at_ms"],
                 # P1: the narrator's line and what happened (absent with the narrator off)
                 **({"summary": e["summary"], "narrator": e["narrator"]}
                    if "narrator" in e else {}),
                 # v3: the line in plain words as it went out; a package merged into the one before
                 **({"shown": e["shown"]} if e.get("shown") else {}),
                 **({"merged": True} if e.get("merged") else {})} for e in ui.log]
        if getattr(rec, "thoughts", None):
            # N2: the request's thought summaries in all: lines, templates, the narrator's time
            self.row["extra"]["narrator"] = rec.thoughts
        if getattr(rec, "voice", None):
            # v3: the voice's totals: lines, phrases, plain lines, the small model's time, why not
            self.row["extra"]["voice"] = rec.voice
        self.row.setdefault("acted", 0)
        self.row.setdefault("changed", 0)
        journal.close_row(self.row)


class Router:
    def __init__(self, store, library, journal, *, raw_model=lambda: "qwen38-spark-engine",
                 self_url=lambda: None, server_effort=lambda: None, global_mode=lambda: None,
                 factory=None, probe=lambda: None, monitor_session=lambda key: None,
                 index=None):
        self.store, self.library, self.journal = store, library, journal
        # R1: the package library's search index (server/conductor/libsearch.py)
        self.index = index if index is not None else libsearch.SkillIndex(library)
        # J1: the session monitor's state for a session key (its acts count against the caps)
        self.monitor_session = monitor_session
        self.probe = probe                  # server/probe.py ProbeStage, or None (ENG-181)
        self.raw_model, self.self_url = raw_model, self_url
        self.server_effort, self.global_mode = server_effort, global_mode
        self.factory = factory or self._make
        self._sessions: collections.OrderedDict[str, Session] = collections.OrderedDict()
        # SRV-225: per `h:` head, the conversations that opened with it (`_generation`)
        self._gens: collections.OrderedDict[str, list] = collections.OrderedDict()
        self._texts: collections.OrderedDict = collections.OrderedDict()
        self._adapters: dict = {}
        self._lock = threading.Lock()
        # server/session_tokens.py: the session's token on the addresses the notes
        # give the model (set by the server), and `h:` keys that continue a compacted session
        self.session_tokens = None
        self.aliases = session_tokens_mod.Aliases()
        # proxy P4 (proxy/harness.py): for a request relayed to another backend, the stage that
        # answers instead of the configured one (the backend itself, asked through its own
        # constraint field), or False when the harness switched the class question off there.
        # Called with the stage; None leaves the stage as configured, which is what it answers
        # for the engine's own requests (and unset, without the proxy).
        self.stage_override = None
        # ... and the scope of the class answers the router keeps per text: a backend's answer
        # is never reused for the engine's own request, nor the other way round (None: the
        # engine's)
        self.cache_scope = lambda: None
        # 2026-10-05: the memory notes (server/conductor/memory.py Memory), set by the conductor
        self.memory = None

    # ---------------------------------------------------------------- classifiers
    def _make(self, kind: str, role: dict, timeout_ms: int):
        key = (kind, role.get("endpoint"), role.get("model"), timeout_ms)
        a = self._adapters.get(key)
        if a is not None:
            return a
        endpoint = role.get("endpoint") or ""
        if endpoint == "self" and kind != "decider":
            # Qwen through this server's own port; a decider named `self` is asked in process
            endpoint = self.self_url() or ""
        if not endpoint:
            return None
        a = (decider_for(endpoint, role.get("model") or "auto", timeout_ms) if kind == "decider"
             else Qwen(endpoint, role.get("model") or self.raw_model(), timeout_ms))
        if a is None:
            return None
        self._adapters[key] = a
        return a

    def _stage_adapter(self, stage: dict, roles: dict, qid: str | None = None):
        if self.stage_override is not None:
            got = self.stage_override(stage, roles)
            if got is False:
                return None, "harness_off"
            if got is not None:
                return got, None
        role = roles.get("decider" if stage["kind"] == "decider" else "classifier") or {}
        if role.get("mode", "off") == "off":
            return None, "role_off"
        if stage["kind"] == "decider" and qid:
            # a model the `deciders` document names for this question; `auto` leaves the role's
            _, dec = self.store.peek("deciders")
            named = dec["questions"].get(qid)
            if named and named != "auto":
                role = dict(role, model=named)
        a = self.factory(stage["kind"], role, int(stage["timeout_ms"]))
        return (a, None) if a is not None else (None, "no_endpoint")

    def cascade(self, text: str, options: dict[str, str], router: dict, roles: dict, *,
                question: str | None = None, accept: float | None = None,
                qid: str = "classify", line: str | None = None,
                about: str | None = None) -> tuple:
        """Ask the enabled stages in order until one answers an option at or above its threshold.
        Returns (the accepted Answer or None, every stage's answer as dicts). `text` is the user's
        message; each stage gets its own digest of it (the decider's sized to its `limits()`,
        Qwen's to `digest_chars`), with the context `line` at its end when given. `qid`: the
        question id (`classify`, `packages`) the `deciders` document maps to a decision model.
        The Qwen stage asks the package question with the class question's prompt and the options
        as the classes, unless the router has a `package_prompt` (FX6), which gets `about`, the
        class the turn was routed to."""
        token = ASKING.set(qid)
        try:
            return self._cascade(text, options, router, roles, question, accept, qid, line, about)
        finally:
            ASKING.reset(token)

    def _cascade(self, text, options, router, roles, question, accept, qid, line,
                 about=None) -> tuple:
        clf = router["classifier"]
        chars = int(clf.get("digest_chars", 2000))
        stages, accepted = [], None
        for st in clf["stages"]:
            # ENG-E10: a model that does not answer the class question has the decider in its
            # place, also where the router document leaves the decider stage off (the shipped one)
            served = modelfit.FIT["served_classifier"]
            if not st.get("enabled", True) and (served or st["kind"] != "decider"):
                continue
            if st["kind"] == "qwen" and not served:
                # ENG-E10: the served model's one-id answers with thinking off are no class
                # (the adapter's fit); the decider decides, else the turn falls back
                continue
            a, why = self._stage_adapter(st, roles, qid)
            if a is None:
                stages.append(Answer(st["kind"], error=f"{st['kind']}_{why}").as_dict())
                continue
            if st["kind"] == "decider":
                room = a.state_chars() if hasattr(a, "state_chars") else chars
                ans = a.choice(signals.digest(text, min(chars, room), line),
                               question or clf["question"], options)
            else:
                listing = "\n".join(f"- {k}: {v}" for k, v in options.items())
                ans = a.choice(signals.digest(text, chars, line),
                               _qwen_prompt(router, question, qid, about, listing), options,
                               max_chars=chars)
            if st["kind"] == "qwen" and question is None:
                ans = _code_guard(ans, text, options)    # SRV-189
            d = ans.as_dict()
            if getattr(ans, "guard", None):
                d["guard"] = ans.guard
            need = accept if accept is not None else float(st["accept"])
            d["accept"] = need
            if ans.error is None and ans.label not in options:
                d["error"] = f"{st['kind']}_unknown_label"
            elif ans.error is None and ans.confidence < need and not ans.guard:
                d["below"] = True
            elif ans.error is None:
                accepted = ans
            stages.append(d)
            if accepted is not None:
                break
        return accepted, stages

    # ---------------------------------------------------------------- sessions
    def replayed_for(self, key: str, ids: list[int], tok) -> list[int]:
        """`ids` with session `key`'s notes replayed where its next turn would carry them (the
        cache peek, SRV-349): `Plan.replayed_ids` for a session that exists, every decision type
        taken as on, the ones a turn turns on for its notes' replay as well (memory, library,
        outcomes, identity, client: live on the 27B the peek's prompt was shorter than the turn's,
        and a CPU run through the handler showed the memory notes missing). Nothing is created or changed; an unknown
        session gives `ids` back."""
        with self._lock:
            sess = self._sessions.get(key)
        if sess is None or not sess.notes:
            return list(ids)
        plan = Plan(None)
        plan.active, plan.session = True, sess
        plan.router_cfg = self.store.peek("router")[1]
        with sess.lock:
            kinds = {k for n in sess.notes for k in n.kinds}
        plan.decisions = {d: "on" for d in (*schema.DECISIONS, *sorted(kinds))}
        return plan.replayed_ids(ids, tok)

    def session(self, key: str) -> Session:
        with self._lock:
            s = self._sessions.get(key)
            if s is None:
                s = self._sessions[key] = Session(key)
            self._sessions.move_to_end(key)
            while len(self._sessions) > SESSIONS_MAX:
                self._sessions.popitem(last=False)
            return s

    def _generation(self, base: str, sh: signals.Shape) -> tuple[str, str]:
        """SRV-225: which conversation with the head `base` (an `h:` key: the client
        sent no id) a request belongs to, as `(session key, how)`.

        Open WebUI sends no chat id by default, so every chat that opened with the same message
        (and the same system prompt) was one conductor session: the UX2 run's A01 started a new
        chat whose first decision row already counted 27 turns and 21,216 session tokens of
        earlier chats, with their notes replayed. A head now holds a list of conversations:

          * a conversation's very first turn (`Shape.fresh`: its one user message is the last
            message) while the newest conversation of that head has had a turn starts a new one,
            `<base>#<n>` (`how` = `new`). A first turn regenerated without an id is the same
            request as a new chat that opens the same way; it starts afresh too, which costs it
            nothing the first turn had (the class is decided again the same way, its notes go in
            at the same place);
          * from the second user message on, the request carries a hash of its whole first turn as
            the client sends it back (`Shape.turn1_hash`): the conversation that recorded it, or
            else the newest one that has none yet and is not ahead of this request, records it;
          * anything else (the first turn's tool rounds, a hash no conversation has) goes to the
            newest conversation not ahead of the request in user messages, as before 0.3.49 when
            there was only one.

        A client id (`c:` keys) never comes here. Off (`QWEN38_SESSION_GENERATIONS=off`): `base`."""
        if SESSION_GENERATIONS != "on":
            return base, "head"
        u = int(getattr(sh, "n_user", 0) or 0)
        sig = getattr(sh, "turn1_hash", None)
        with self._lock:
            gens = self._gens.get(base)
            if gens is None:
                gens = self._gens[base] = []
            self._gens.move_to_end(base)
            while len(self._gens) > GEN_HEADS_MAX:
                self._gens.popitem(last=False)
            if not gens:
                gens.append({"key": base, "u": u, "sig": sig, "n": 1})
                return base, "new"
            newest = gens[-1]
            if getattr(sh, "fresh", False):
                s = self._sessions.get(newest["key"])
                if s is None or s.turns == 0:
                    newest.update(u=1, sig=None)
                    return newest["key"], "continued"
                n = newest["n"] + 1
                gens.append({"key": f"{base}#{n}", "u": 1, "sig": None, "n": n})
                del gens[:-GEN_PER_HEAD]
                return gens[-1]["key"], "new"
            if sig is not None:
                for g in reversed(gens):
                    if g["sig"] == sig:
                        g["u"] = max(g["u"], u)
                        return g["key"], "continued"
                free = [g for g in gens if g["sig"] is None and g["u"] <= u]
                if free:
                    g = max(free, key=lambda g: (g["u"], g["n"]))
                    g.update(sig=sig, u=max(g["u"], u))
                    return g["key"], "continued"
            fit = [g for g in gens if g["u"] <= u] or gens
            g = max(fit, key=lambda g: (g["u"], g["n"]))
            g["u"] = max(g["u"], u)
            return g["key"], "continued"

    def sessions_live(self) -> list[dict]:
        with self._lock:
            ss = list(self._sessions.values())
        tk = self.session_tokens
        return [{"key": s.key, "class_id": s.cls, "confidence": s.conf, "source": s.source,
                 "turns": s.turns, "notes": len(s.notes),
                 "note_tokens": sum(len(n.ids) for n in s.notes),
                 **({"calls": tk.calls(s.key)} if tk is not None else {})} for s in ss]

    def compacted_into(self, key: str | None, summary: str) -> None:
        """A compaction of session `key` wrote `summary`: a client without a session id whose
        next head carries it is that session still (server/session_tokens.py `Aliases`)."""
        if key and summary:
            self.aliases.compacted(key, summary)

    def _cached(self, key):
        with self._lock:
            v = self._texts.get(key)
            if v is not None:
                self._texts.move_to_end(key)
            return v

    def _remember(self, key, value) -> None:
        with self._lock:
            self._texts[key] = value
            while len(self._texts) > TEXT_CACHE_MAX:
                self._texts.popitem(last=False)

    # ---------------------------------------------------------------- the plan
    def plan(self, body: dict, headers, *, client_kind: str, request_id: str,
             chat: bool = True, model: str | None = None, dry: bool = False,
             count_tokens=None, caps: dict | None = None, user: str | None = None) -> Plan:
        """`dry`: the test box's run, on a throwaway session and without a journal row.
        `count_tokens()`: the prompt's length in tokens, asked only when the in-context stage
        needs it. `caps` (proxy P4): decision type -> the highest mode it may have for this
        request (the harness's switches per backend and model), and `side` for the side-request
        shapes; None for the engine's own requests."""
        if not dry:
            PARENT.set(request_id)                # VIS-60: the router's question names this turn
        res = resolve(self.store, str(model or body.get("model") or self.raw_model()),
                      self.raw_model(), headers, client_kind, self.global_mode(),
                      offers_tools=chat and signals.offers_tools(body))
        # `{tandem_url}` in a package's text: the address this request came in on
        self.library.at(packages_mod.request_url(headers))
        plan = Plan(res)
        plan.user, plan.caps = user, caps
        plan.memory_opt = memory_mod.opt_of(headers)        # SRV-173: X-Tandem-Memory
        if res.mode == "raw" or not chat:
            return plan
        prof = res.doc
        router_v, router = self.store.peek("router")
        _, packages = self.store.peek("packages")
        _, roles = self.store.peek("roles")
        _, settings = self.store.peek("settings")
        plan.active, plan.router_cfg = True, router
        # SRV-621: `prompt: client`, the prompt goes to the engine as the client sent it
        plan.prompt_client = schema.profile_prompt(prof) == "client"
        dec = {d: _lower(prof["decisions"].get(d, "off"), res.mode) for d in schema.DECISIONS}
        dec = {d: ("off" if m == "raw" else m) for d, m in dec.items()}
        if caps:
            dec = {d: ("off" if _lower(m, caps.get(d)) == "raw" else _lower(m, caps.get(d)))
                   for d, m in dec.items()}
        plan.decisions = dec
        classes = {c["id"]: c for c in router["classes"]}
        sh = signals.shape(body, tuple(packages["packages"]), client_kind=client_kind)
        plan.tool_caps, plan.tool_names = signals.tool_caps(body)
        side = schema.side_requests(router)
        side_cap = (caps or {}).get("side")
        shape = (signals.side_shape(body, side["shapes"])
                 if side["mode"] == "on" and not sh.compaction and side_cap != "off"
                 and not plan.prompt_client else None)
        if shape is not None:
            return self._side(plan, body, headers, sh, shape, router_v, client_kind, request_id,
                              dry, side_cap)
        key = session_key(body, headers, sh)
        sess_how = "id" if key.startswith("c:") else None
        if not dry:
            head = key
            key = self.aliases.resolve(key, body.get("messages") or [],
                                       known=lambda k: k in self._sessions)
            if key == head and key.startswith("h:") and not sh.compaction:
                key, sess_how = self._generation(key, sh)
            elif key != head:
                sess_how = "alias"
        sess = Session("dry") if dry else self.session(key)
        plan.session = sess
        if not dry and self.session_tokens is not None:
            # `{tandem_url}` in this session's notes: its own address, `.../s/<token>`
            self.library.at(session_tokens_mod.with_token(
                packages_mod.request_url(headers) or packages_mod.URL_DEFAULT,
                self.session_tokens.token_for(sess.key)))
        # a compaction under `prompt: client` is served as sent, as on the raw path
        plan.compaction = sh.compaction and not plan.prompt_client
        if not dry:
            plan.body = body
            plan.transcript = None if schema.profile_transcripts(prof) == "off" else \
                transcripts.Transcripts.scope_of(transcripts.settings_of(settings), sh.tools,
                                                 sh.compaction)
        digest_scope = settings["journal"]["digest"]
        keep_text = digest_scope == "all" or (digest_scope == "tools" and sh.tools)
        # SRV-153: the context line, from the session as it stood before this turn
        line = None
        if router["classifier"].get("context_line") and sh.user_text.strip() and \
                not sh.compaction:
            facts = signals.line_facts(sh)
            with sess.lock:
                line = signals.context_line(sess.cls, sess.conf, facts["tools"], facts["exts"],
                                            facts["checker"], sess.compacted)
        row = {"request_id": request_id, "session_key": sess.key, "client_kind": client_kind,
               "model": body.get("model"), "profile": res.profile,
               "profile_version": res.version, "router_version": router_v, "mode": res.mode,
               "decisions": dict(dec), "stages": [], "shadow": None, "injected": [],
               "replayed": None, "acted": 0, "changed": 0, "extra": {},
               "digest": signals.digest(sh.user_text, int(router["classifier"]["digest_chars"]),
                                        line) if keep_text and sh.user_text else None}
        if line:
            row["extra"]["context_line"] = line
        if res.source == "tools":
            row["extra"]["profile_source"] = "tools"        # SRV-621: resolve's rule
        if sess_how is not None:
            # SRV-225: how the session was found: the client's id, a new conversation of its
            # head, one continued, an alias after a compaction
            row["extra"]["session"] = sess_how
        plan.row = row
        # the previous turn's outcome label: the checker result this request carries
        if sh.results and sess.last_row and not dry:
            self.journal.update(sess.last_row, next_checker=sh.results)

        # ---- the class
        t0 = time.perf_counter()
        default = prof["default_class"] if prof["default_class"] in classes else \
            router["classes"][0]["id"]
        cls, conf, source, kept, stages = None, None, None, None, []
        structural = router["structural"]
        if dec["classify"] == "off":
            cls, source = default, "profile"
        elif sh.compaction and structural["compaction"] in classes:
            cls, conf, source = structural["compaction"], 1.0, "structural"
        elif sh.tool_turn and sess.cls in classes and structural["tool_turn"] == "keep":
            # SRV-161: a tool's media in a user message is a tool turn too, journaled as such
            cls, conf, source = sess.cls, sess.conf, "sticky"
            kept = "tool_media" if sh.tool_media else None
        elif sh.user_hash is not None and sh.user_hash == sess.user_hash and sess.cls in classes:
            cls, conf, source = sess.cls, sess.conf, "sticky"
        elif sess.compacted and sess.cls in classes:
            # The first user message after a compaction ("continue", opencode's own
            # "Continue if you have next steps ...") goes on with the work the compaction carried
            # over: the session's class, not a new one read from those few words (2026-10-03, ~/rrr6:
            # "continue" became chat at 0.998 and the design packages never came back). The block
            # is not in the compacted history, so `_apply` adds it again as on a first turn. The
            # message after that is classified as any other.
            cls, conf, source, kept = sess.cls, sess.conf, "sticky", "after_compaction"
            with sess.lock:
                sess.compacted = False
        elif not sh.user_text.strip():
            cls, source, kept = (sess.cls if sess.cls in classes else default), "fallback", \
                "no_text"
        else:
            options = {c: classes[c]["description"] for c in classes}
            with sess.lock:
                sess.compacted = False             # the line said so once
            due = None if dry else self._context_due(router, body, sh, count_tokens)
            if due is not None and self._probe_first(router):
                due = dict(due, probe=True)
            if due is not None:
                # Qwen decides inside the conversation, under the engine lock right before the
                # turn's own prefill (`decide_in_context`); the rest of the plan waits for it
                row["extra"]["context_stage"] = due
                plan.context = {"t0": t0, "options": options, "due": due,
                                "args": (body, router, prof, packages, roles, sh, sess, classes,
                                         default, dec, dry)}
                return plan
            # the deciders document says which model answers; a change there re-asks, and so
            # does another context line
            ck = (router_v, self.store.peek("deciders")[0],
                  hashlib.sha256(sh.user_text.encode()).hexdigest(), line, self.cache_scope())
            hit = self._cached(ck)
            if hit is not None:
                ans, stages = hit
                stages = [dict(s, cached=True) for s in stages]
            else:
                ans, stages = self.cascade(sh.user_text, options, router, roles, line=line)
                if ans is not None or all(s.get("error") is None for s in stages):
                    self._remember(ck, (ans, stages))
            cls, conf, source, kept = self._pick(ans, stages, router, sess, classes, default)
        return self._conclude(plan, t0, cls, conf, source, kept, stages, body, router, prof,
                              packages, roles, sh, sess, classes, default, dec, dry)

    def _side(self, plan, body, headers, sh, shape, router_v, client_kind, request_id,
              dry, cap: str | None = None) -> Plan:
        """A client's side request (S1, `structural.side_requests`): opencode's title or summary
        call, Hermes' title call. No classifier is asked, no package or note is added, the session
        is not touched (its class, turns, notes and size stay the turn's); thinking is off unless
        the client asked for a number of reasoning tokens, and the output is capped by the shape.
        The journal row is `kind: side` with the shape in `extra.side`. A profile in shadow
        journals what it would do and serves the request as the client sent it, as shadow serves
        every turn."""
        res = plan.res
        client = thinking.read(body, strict=False)
        numeric = client.kind == "number" and bool(client.budget)
        info = {"shape": shape["id"], "client": shape.get("client") or "the client",
                "kind": shape["kind"], "max_tokens": int(shape["max_tokens"]),
                "thinking": "client" if numeric else "off"}
        plan.session = Session("side")         # no notes to replay, nothing of the session moves
        row = {"request_id": request_id, "kind": "side",
               "session_key": session_key(body, headers, sh), "client_kind": client_kind, "model": body.get("model"), "profile": res.profile,
               "profile_version": res.version, "router_version": router_v, "mode": res.mode,
               "decisions": dict(plan.decisions), "stages": [], "shadow": None, "injected": [],
               "replayed": None, "acted": 0, "changed": 0, "extra": {"side": info},
               "digest": None, "class_id": None, "confidence": None, "source": "structural",
               "kept": None, "classifier_ms": 0.0, "effort": None, "budget": None,
               "budget_source": "client" if numeric else "server", "thinking": info["thinking"]}
        plan.row = row
        if _lower(res.mode, cap) == "on":
            plan.side = info
            plan.thinking_off = not numeric
            row["acted"] = 1
        else:
            row["shadow"] = {"side": info}
            row["thinking"] = "client" if client.thinking is not None else "on"
        if not dry:
            self.journal.open_row(row)
        return plan

    # ---------------------------------------------------------------- Qwen in the conversation
    def _context_due(self, router, body, sh, count_tokens) -> dict | None:
        """Whether this new user message goes to the in-context stage instead of the digest
        cascade: the stage is on, and the message follows an earlier answer in the conversation
        (`followups`) or the prompt has at least `min_tokens`. A short first message stays the
        decider's. Images go through the stage as the turn's prompt carries them."""
        cs = schema.context_stage(router)
        if cs["mode"] != "on" or count_tokens is None:
            return None
        msgs = [m for m in body.get("messages") or [] if isinstance(m, dict)]
        if cs["followups"] and sh.user_index is not None and any(
                m.get("role") == "assistant" for m in msgs[:sh.user_index]):
            return {"due": "followup"}
        try:
            n = int(count_tokens())
        except Exception:                                          # noqa: BLE001
            return None                     # the render failed: the cascade, as before
        return {"due": "long", "tokens": n} if n >= int(cs["min_tokens"]) else None

    def _probe_first(self, router) -> bool:
        """The probe stage decides first on this plan's lock path: mode on, its file loaded."""
        ps = schema.probe_stage(router)
        st = self.probe()
        return ps["mode"] == "on" and st is not None and st.ready(ps)

    def _probe_stage(self, router, prompt_ids, tok, rows, plan, due) -> dict:
        """The probe on a prefill of the turn's prompt up to its insertion point (`rows(ids)`
        hands back that prefill's last rows, their names and its counts)."""
        ps, st = schema.probe_stage(router), self.probe()
        t0 = time.perf_counter()
        try:
            ids = plan.replayed_ids(prompt_ids, tok)
            at = inject.insertion_point(ids, inject.special_id(tok, inject.IM_END))
            if at is None or at < 1:
                raise incontext.ContextError("no_insertion_point")
            vec, names, info = rows(ids[:at])
            d = st.stage(ps, vec, names, reason=due)
            d.update(prompt_tokens=at, prefilled=int(info.get("forwarded", 0)),
                     reused=int(info.get("reused", 0)))
        except ConnectionError:
            raise
        except Exception as exc:                                   # noqa: BLE001
            d = {"stage": "probe", "label": None, "confidence": 0.0, "probs": {}, "ms": 0.0,
                 "error": "probe_" + getattr(exc, "why", type(exc).__name__), "model": ps["file"],
                 "accept": float(ps["accept"]), "reason": due}
        d["stage_ms"] = round((time.perf_counter() - t0) * 1e3, 2)
        return d

    def decide_in_context(self, plan: "Plan", prompt_ids: list[int], tok, runner,
                          rows=None) -> None:
        """The in-context stage, under the engine lock before the turn's prefill: build its
        prompt from the turn's (the session's notes replayed), let `runner(ids, letter_ids)`
        prefill it and hand back the letters' logits with its counts, and finish the plan from
        the answer. Every failure is the stage's error in the journal, and the class falls back
        as the cascade's does when nobody answered. With the probe stage on (ENG-181) the probe
        answers first, on `rows(ids)`'s prefill up to the insertion point; at or above its
        threshold the question is not asked."""
        ctx, plan.context = plan.context, None
        if ctx is None:
            return
        plan.locked = True
        body, router, prof, packages, roles, sh, sess, classes, default, dec, dry = ctx["args"]
        cs = schema.context_stage(router)
        t0 = time.perf_counter()
        stages = []
        if ctx["due"].get("probe") and rows is not None:
            try:
                pd = self._probe_stage(router, prompt_ids, tok, rows, plan, ctx["due"]["due"])
            except ConnectionError as exc:
                pd = {"stage": "probe", "label": None, "confidence": 0.0, "probs": {}, "ms": 0.0,
                      "error": "probe_client_gone", "model": None, "reason": ctx["due"]["due"]}
                cls, conf, source, kept = self._pick(None, [pd], router, sess, classes, default)
                self._conclude(plan, t0, cls, conf, source, kept, [pd], body, router, prof,
                               packages, roles, sh, sess, classes, default, dec, dry)
                raise exc
            stages.append(pd)
            if pd["error"] is None and not pd.get("below") and pd["label"] in classes:
                ans = Answer("probe", pd["label"], pd["confidence"], pd["probs"])
                cls, conf, source, kept = self._pick(ans, stages, router, sess, classes, default)
                self._conclude(plan, t0, cls, conf, source, kept, stages, body, router, prof,
                               packages, roles, sh, sess, classes, default, dec, dry)
                return
        tq = time.perf_counter()
        stage = {"stage": "qwen_context", "label": None, "confidence": 0.0, "probs": {},
                 "ms": 0.0, "error": None, "model": None, "accept": float(cs["accept"]),
                 "reason": ctx["due"]["due"]}
        ans, gone = None, None
        try:
            text, letters = incontext.listing(ctx["options"])
            ids, at = incontext.prompt(plan.replayed_ids(prompt_ids, tok), tok,
                                       cs["question"].replace("{classes}", text))
            lids = incontext.letter_ids(tok, letters)
            logits, info = runner(ids, lids, at)
            label, conf, probs = incontext.answer(logits, list(ctx["options"]))
            stage.update(label=label, confidence=round(conf, 4),
                         probs={k: round(v, 4) for k, v in probs.items()},
                         prompt_tokens=len(ids), prefilled=int(info.get("forwarded", 0)),
                         reused=int(info.get("reused", 0)), model=info.get("model"))
            if conf < float(cs["accept"]):
                stage["below"] = True
            else:
                ans = Answer("qwen_context", label, conf, probs)
        except ConnectionError as exc:
            # the client left during the stage's prefill: the row is finished, then the request
            # ends as any request whose client left
            stage["error"], gone = "qwen_context_client_gone", exc
        except Exception as exc:                                   # noqa: BLE001
            stage["error"] = "qwen_context_" + getattr(exc, "why", type(exc).__name__)
            if getattr(exc, "why", None) is None:
                print(f"[conductor] the in-context stage failed: {type(exc).__name__}: {exc}",
                      flush=True)
        stage["ms"] = round((time.perf_counter() - tq) * 1e3, 2)
        stages.append(stage)
        cls, conf, source, kept = self._pick(ans, stages, router, sess, classes, default)
        self._conclude(plan, t0, cls, conf, source, kept, stages, body, router, prof, packages,
                       roles, sh, sess, classes, default, dec, dry)
        if gone is not None:
            raise gone

    def settle_context(self, plan: "Plan", why: str = "not_run") -> None:
        """A plan whose in-context stage never ran (the request ended before the engine lock):
        its row gets the session's class, as when nobody answered."""
        ctx, plan.context = plan.context, None
        if ctx is None:
            return
        body, router, prof, packages, roles, sh, sess, classes, default, dec, dry = ctx["args"]
        stages = [{"stage": "qwen_context", "label": None, "confidence": 0.0, "probs": {},
                   "ms": 0.0, "error": f"qwen_context_{why}", "model": None,
                   "reason": ctx["due"]["due"]}]
        cls, conf, source, kept = self._pick(None, stages, router, sess, classes, default)
        self._conclude(plan, time.perf_counter(), cls, conf, source, kept, stages, body, router,
                       prof, packages, roles, sh, sess, classes, default, dec, dry)

    def _pick(self, ans, stages, router, sess, classes, default) -> tuple:
        """The class from the accepted answer (or none): the session's when nobody was sure or
        the answer is below the switch, else the answer's."""
        if ans is None:
            below = self._below_build(stages, router, sess, classes)
            if below is not None:
                return below
            return (sess.cls if sess.cls in classes else default), None, "fallback", (
                "fail_open" if stages and all(s.get("error") for s in stages) else "no_answer")
        if sess.cls in classes and sess.cls != ans.label and \
                ans.confidence < float(router["stickiness"]["switch"]):
            return sess.cls, sess.conf, "sticky", "below_switch"
        return ans.label, ans.confidence, ans.stage, None

    @staticmethod
    def _below_build(stages, router, sess, classes):
        """A session's first classified message whose answer stayed under the accept bar
        (`classifier.below_accept`, mode `build`): the top class when it builds something (its
        policy names `files-care`: code, a page, a scene, a drawing) and leads the second by at
        least `margin`, else None (the default class, as before). The request "Create a new folder and
        create there a pagoda garden" read three_d 0.378 under a router whose chat class had just
        grown (0.464 before) and fell back to chat: no 3D skill, an ASCII pagoda written into a
        file until the argument loop guard stopped it. The class's reading line still says what
        was built when a second build class is close (`_reading`)."""
        rule = (router["classifier"].get("below_accept") or {})
        if rule.get("mode", "default") != "build" or sess.cls in classes:
            return None
        st = next((s for s in reversed(stages) if s.get("probs") and not s.get("error")), None)
        if st is None:
            return None
        top = sorted(st["probs"].items(), key=lambda kv: -kv[1])
        if not top or top[0][0] not in classes:
            return None
        cls, p = top[0]
        # PKG3 (2026-10-07): a class builds when it writes files, and every class that writes
        # files carries `files-care` (SRV-227); since the everyday packages, the prose classes
        # name packages too and still fall back
        if BUILD_MARK not in ((classes[cls].get("policy") or {}).get("packages") or []):
            return None
        lead = p - (top[1][1] if len(top) > 1 else 0.0)
        if lead < float(rule.get("margin", 0.0)):
            return None
        return cls, p, st.get("stage") or "qwen", "below_build"

    def _conclude(self, plan, t0, cls, conf, source, kept, stages, body, router, prof, packages,
                  roles, sh, sess, classes, default, dec, dry) -> Plan:
        """The class is known: the session follows it, and the policy, the effort, the note and
        the journal row come from it."""
        row = plan.row
        row["classifier_ms"] = round((time.perf_counter() - t0) * 1e3, 2)
        row["stages"] = stages
        model_cls, model_conf, model_source = cls, conf, source
        if dec["classify"] == "shadow":
            # journal the class; the policy runs as the profile's default class
            row["shadow"] = {"class_id": cls, "confidence": conf, "source": source}
            cls, source = default, "profile"
        row.update(class_id=cls, confidence=conf, source=source, kept=kept)
        with sess.lock:
            # the session follows the model's class in shadow too, so shadow's stickiness is
            # the one `on` would have
            # a compaction request is its own kind of turn; the work after it goes on in the
            # session's class, so the structural class never becomes the session's
            sess.turns += 1
            if sh.compaction:
                sess.compacted = True
            follow = dec["classify"] != "off" and model_source != "structural"
            # FX6: the reading line goes in on the turn a class starts, not on every later one
            plan.class_new = model_cls != sess.cls
            if follow and model_cls != sess.cls:
                sess.history.append({"at": time.time(), "from": sess.cls, "to": model_cls,
                                     "confidence": model_conf, "source": model_source})
            if follow:
                sess.cls, sess.conf, sess.source = model_cls, model_conf, model_source
            if sh.user_hash is not None:
                sess.user_hash = sh.user_hash

        # ---- the policy
        base = classes[cls]["policy"]
        ov = prof["routing"].get(cls) or {}
        policy = {"effort": ov.get("effort") or base.get("effort") or prof["thinking"]["default"],
                  "packages": list(ov["packages"] if "packages" in ov else base.get("packages", [])),
                  "core": list(ov["core"] if "core" in ov else base.get("core", [])),
                  "rules": list(ov["rules"] if "rules" in ov else base.get("rules", [])),
                  "note": ov.get("note") or base.get("note") or "",
                  "package_sticky": ov.get("package_sticky") or base.get("package_sticky"),
                  "effort_by_size": ov["effort_by_size"] if "effort_by_size" in ov
                  else base.get("effort_by_size")}
        if modelfit.FIT["fallback"] == "lean" and conf is None and source in ("fallback", "sticky"):
            # ENG-E10: nobody decided the class (no answer at the accept bar, or a tool turn of
            # such a session): the profile's default effort, so a short chat does not think at a
            # coding task's xhigh. SRV-534: the class's packages, core and rules stay (the
            # profile's default class on a new session, the session's class after it), and the
            # adapter's fit filters them as for a decided class: with Laya refused, every Chat-tab
            # turn on Super VL went out with no package at all. A tool turn keeps the class of
            # its user message, so the note a run started with stays the same across its steps
            policy.update(effort=prof["thinking"]["default"], effort_by_size=None)
            row["extra"]["model_fit_lean"] = True
        if modelfit.FIT != modelfit.DEFAULTS:
            row["extra"]["model_fit"] = modelfit.summary()
        self._size(plan, router, prof, sh, sess, policy, row)
        if not dry:
            self._observe(plan, body, router, prof, sh, sess, row)
        stops = argloop.trailing_stops(body.get("messages"))
        if stops and dec["effort"] != "off":
            plan.argloop_retry = {"stops": len(stops), "identical": argloop.identical(stops),
                                  "name": stops[-1].get("name")}
        level = self._turn_level(plan, router, prof, sh, policy, row)
        self._effort(plan, body, router, prof, sh, self._outcome_level(plan, level, row), row)
        self._block(plan, body, router, roles, packages, sh, sess, cls, policy, row, classes)
        if not dry:
            self._cards(plan, body, router, prof, packages, sh, cls, row)
            self._outcome_note(plan, body, router, prof, packages, sh, sess, row)
        # R1: the package library (its profile switch off, absent included: nothing at all)
        libsearch.attach(self, plan, body, router, prof, packages, roles, sh, sess, cls, row, dry)
        # 2026-10-05: the memory notes (its profile switch off, absent included: nothing at all)
        try:
            if not modelfit.FIT["memory"]:
                if schema.profile_memory(prof) != "off" and plan.res.mode != "raw" and \
                        getattr(plan, "memory_opt", None) != "off":
                    memory_mod.off_note(plan, sh, row, "model_fit", dry)
                else:
                    row["extra"]["memory"] = {"mode": "off", "why": "model_fit"}
                raise _Skip
            memory_mod.attach(self, plan, body, router, prof, packages, roles, sh, sess, cls, row,
                              dry, plan.user, (plan.caps or {}).get("memory"),
                              opt=getattr(plan, "memory_opt", None))
        except _Skip:
            pass
        except Exception as exc:                                   # noqa: BLE001
            row["extra"].setdefault("errors", []).append(f"memory:{type(exc).__name__}")
            print(f"[conductor] memory failed: {type(exc).__name__}: {exc}", flush=True)
        self._identity(plan, sh, row)
        if not dry:
            self._client_cwd(plan, body, sh, sess, row)
        self._as_sent(plan, row)
        if dry:
            return plan
        self.journal.open_row(row)
        with sess.lock:
            sess.last_row = row["id"]
        return plan

    def explain(self, text: str, profile: str | None = None, *, tools: bool = True,
                messages: list | None = None) -> dict:
        """The dashboard's test box: the whole decision for a pasted prompt, with the exact note,
        touching no session and writing no journal row."""
        body = {"model": COMPOSITE,
                "messages": messages or [{"role": "user", "content": text}]}
        if tools:
            body["tools"] = [{"type": "function", "function": {
                "name": "bash", "description": "run a command",
                "parameters": {"type": "object", "properties": {"command": {"type": "string"}}}}}]
        headers = {"X-Tandem-Profile": profile} if profile else {}
        plan = self.plan(body, headers, client_kind="dashboard", request_id="explain", dry=True)
        out = {"profile": plan.res.profile, "profile_version": plan.res.version,
               "mode": plan.res.mode, "active": plan.active}
        if not plan.active:
            return out
        row = plan.row
        cfg = plan.router_cfg
        note = (cfg["injection"]["wrapper"].replace("{text}", plan.block["text"])
                if plan.block else None)
        out.update({k: row.get(k) for k in ("class_id", "confidence", "source", "kept", "stages",
                                             "classifier_ms", "effort", "budget", "budget_source",
                                             "thinking", "decisions", "shadow", "extra")})
        out["note"] = note
        out["note_parts"] = plan.block["parts"] if plan.block else []
        out["thinking_off"] = plan.thinking_off
        return out

    @staticmethod
    def size_signals(sh: signals.Shape, sess: Session) -> dict:
        """SRV-157's signals, all structural: the prompt tokens the session added after its first
        turn (opencode's own head, about 24k tokens, is not the task's size), the distinct files
        its tool calls named, the most gates one known checker's RESULT line named (None: no
        known checker spoke yet, so it does not vote), and the turn index."""
        added = 0
        if sess.prompt_first is not None and sess.prompt_last is not None:
            added = max(0, sess.prompt_last - sess.prompt_first)
        return {"session_tokens": added, "files": len(sh.files), "checks": sh.max_checks,
                "turns": sess.turns}

    def _size(self, plan, router, prof, sh, sess, policy, row) -> None:
        """SRV-157: the size of the task picks the effort within the class's ceiling. Each signal
        is small below its `small` threshold, medium up to its `medium` one, large above; the
        task's size is the largest any signal reached, and it never falls within the task (a
        compaction starts it over). With the profile's `thinking.by_size` on, the class's
        `effort_by_size` level for that size (never above the class's effort) becomes the turn's
        effort; in shadow it is journaled beside the class's effort, which the turn keeps."""
        mode = (prof.get("thinking") or {}).get("by_size", schema.BY_SIZE_DEFAULT)
        cfg = router["effort"].get("size") or schema.ROUTER["effort"]["size"]
        if sh.compaction:
            if cfg.get("reset_on_compaction", True):
                with sess.lock:
                    sess.size_tier = None
            return
        if mode == "off" or plan.decisions["effort"] == "off":
            return
        sig = self.size_signals(sh, sess)
        tiers = {}
        for name, v in sig.items():
            c = cfg["signals"].get(name)
            if not c or not c.get("enabled", True) or v is None:
                continue
            tiers[name] = 0 if v < c["small"] else 1 if v <= c["medium"] else 2
        now = max(tiers.values(), default=0)
        with sess.lock:
            tier = max(now, sess.size_tier or 0)
            sess.size_tier = tier
        info = {"mode": mode, "size": schema.SIZES[tier], "score": tier,
                "now": schema.SIZES[now],
                "signals": {n: {"value": sig[n], "size": schema.SIZES[t]}
                            for n, t in tiers.items()}}
        by = policy.get("effort_by_size")
        if by:
            ceiling = policy["effort"]
            want = by.get(schema.SIZES[tier]) or ceiling
            picked = want if EFFORT_ORDER.index(want) <= EFFORT_ORDER.index(ceiling) else ceiling
            info.update(ceiling=ceiling, effort=picked, applied=mode == "on")
            if mode == "on":
                policy["effort"] = picked
            else:
                row["shadow"] = dict(row.get("shadow") or {}, size={
                    "size": info["size"], "score": tier, "effort": picked, "ceiling": ceiling})
        row["extra"]["size"] = info

    @staticmethod
    def _turn_level(plan, router, prof, sh, policy, row) -> str:
        """The level this turn thinks at: the class's (after its size), or on a turn whose newest
        message is a tool result (SRV-161's tool media included, never a compaction) the
        `tool_turn_effort` level when one is set (F2). The first turn and every new user message
        keep the class's. Only the thinking switch and the cap follow it; the note is still the
        class's, so the prompt the session's notes sit in does not change."""
        want = schema.tool_turn_effort(router, prof)
        if plan.argloop_retry is not None:
            # A retry after the loop guard stopped the last call thinks at the
            # class's level at least (`_effort` raises it), never the tool turn's lower one
            return policy["effort"]
        if want == "inherit" or not sh.tool_turn or sh.compaction or \
                plan.decisions["effort"] == "off":
            return policy["effort"]
        row["extra"].update(effort_source="tool_turn", class_effort=policy["effort"])
        return want

    def _effort(self, plan, body, router, prof, sh, level, row) -> None:
        dec = plan.decisions["effort"]
        raised = False
        # the client's own number, in any shape (server/thinking.py), always wins; its effort
        # word or switch yields to the level, and an explicit on or off is the client's
        client = thinking.read(body, strict=False)
        client_off = client.thinking is False
        if router["effort"].get("raise_on_fail") and prof["thinking"].get("hints") and \
                level in ("low", "medium", "high") and any(r["verdict"] == "FAIL"
                                                            for r in sh.results):
            level = EFFORT_ORDER[EFFORT_ORDER.index(level) + 1]
            raised = True
        retry = plan.argloop_retry
        if retry is not None:
            # The retry after a stopped looping call is not the same turn again:
            # one level up, at least medium (thinking on), unless the client turned thinking off
            # (its word is a ceiling, the owner 2026-10-03); the stronger note goes in either way
            want = EFFORT_ORDER[min(EFFORT_ORDER.index(level) + 1, len(EFFORT_ORDER) - 1)]
            if EFFORT_ORDER.index(want) < EFFORT_ORDER.index("medium"):
                want = "medium"
            retry["effort"] = ({"from": level, "to": None, "blocked": "client turned thinking off"}
                               if client_off else {"from": level, "to": want})
            if not client_off and want != level:
                level, raised = want, True
            row["extra"]["argloop_retry"] = retry
        lv = router["effort"]["levels"][level]
        # the profile's own cap for the level, else the router's (0 = none below the backstop)
        own = (prof["thinking"].get("budgets") or {}).get(level)
        cap = int(lv["budget"] if own is None else own)
        client_budget = client.budget if client.kind == "number" else None
        client_thinking = client.thinking
        budget, source, thinking_on = None, "server", "on"
        if client_budget is not None:
            source = "client"
        if client_off:
            # The client turned thinking off (OWUI's `reasoning_effort: "none"`,
            # `enable_thinking: false`, ...): the template closes the block, so the row says so
            # instead of the class's level (it said "on" with xhigh's note, 2026-10-06 06:11)
            thinking_on, budget, source = "off_client", None, "client"
            plan.client_off = True
            row["extra"]["client_thinking"] = {
                "off": True, "source": client.source or "enable_thinking", "level": level,
                "why": "client turned thinking off"}
        elif not lv["thinking"]:
            thinking_on = "client" if client_thinking is not None else "off"
        elif client_budget is None and cap > 0:
            budget, source = cap, "class" if own is None else "profile"
        would = {"effort": level, "budget": budget, "budget_source": source,
                 "thinking": thinking_on, "raised_on_fail": raised}
        if dec == "on":
            plan.budget = budget
            plan.budget_source = source
            # the plan closes the block only where the router turned thinking off; a client's
            # off is the template's already (`off_client`), nothing to splice
            plan.thinking_off = thinking_on == "off"
            if budget is not None:
                row["acted"] = 1
                plan.changed = True
            row.update(effort=level, budget=budget, budget_source=source, thinking=thinking_on)
        else:
            row.update(effort=None, budget=None, budget_source="client" if client_budget is not None
                       else "server", thinking="off_client" if client_off
                       else "client" if client_thinking is not None else "on")
            if dec == "shadow":
                row["shadow"] = dict(row.get("shadow") or {}, effort=would)

    def _block(self, plan, body, router, roles, packages, sh, sess, cls, policy, row,
               classes=None) -> None:
        """The note this class wants in the prompt: its effort sentence, the reading line (FX6),
        its packages, its rules. Parts whose decision type is on go into `plan.block`; shadow
        parts are journaled.
        Packages: the general ones and the class's `core` (FX6) go in whole, in the class's
        order; the package question picks among the rest, told which class the turn is."""
        classes = classes or {}
        dec = plan.decisions
        parts: list[tuple[str, dict]] = []                # (decision type, part)
        base_level = policy["effort"]
        note = router["effort"]["levels"][base_level]["note"]
        if dec["effort"] != "off" and note and plan.client_off:
            # No "think carefully" sentence in a turn that cannot think
            row["extra"].setdefault("notes", []).append("effort_note_skipped:client_off")
        elif dec["effort"] != "off" and note and not modelfit.FIT["effort_notes"]:
            # ENG-E10: the level sets the switch and the budget; its sentence stays out
            row["extra"].setdefault("notes", []).append("effort_note_skipped:model_fit")
        elif dec["effort"] != "off" and note:
            parts.append(("effort", {"kind": "effort", "name": base_level, "text": note}))
        reading = self._reading(plan, router, cls, classes, row)
        if reading is not None:
            parts.append(reading)
        # SRV-243: the packages whose steps the request's tools can carry out; a fallback stands
        # in for one that cannot, the rest stay out and the turn says what its tools are
        listed, core_in, fit_note, stand_in = self._fit(plan, router, packages, policy, row)
        # SRV-361: the client's pins and blocks (chatd: a chat's skills control)
        pinned, blocked, unknown = packages_mod.body_packages(body, packages["packages"])
        if blocked:
            listed = [p for p in listed if p not in blocked]
            core_in = [p for p in core_in if p not in blocked]
            stand_in = [p for p in stand_in if p not in blocked]
            row["extra"]["package_blocked"] = sorted(blocked)
        if unknown:
            row["extra"].setdefault("notes", []).append(f"pin_unknown:{','.join(unknown)}")
        mode = "none" if cls in modelfit.FIT["no_package_classes"] else modelfit.FIT["packages"]
        if mode != "all":
            # ENG-E10: the adapter's fit: the class's core alone, or nothing; pins still go in
            keep = set(core_in) if mode == "core" else set()
            dropped = [p for p in listed if p not in keep]
            listed = [p for p in listed if p in keep]
            core_in = [p for p in core_in if p in keep]
            stand_in = [p for p in stand_in if p in keep]
            if dropped:
                row["extra"]["package_fit_dropped"] = dropped
        cands = [p for p in listed if p not in sh.loaded_skills and p not in pinned]
        skipped = [p for p in listed if p in sh.loaded_skills]
        if skipped:
            row["extra"].setdefault("notes", []).append(f"already_loaded:{','.join(skipped)}")
        # a general package (the coding package, 2026-10-02) goes in whole and first; the router
        # picks among the rest, the task skills, as before
        core = [p for p in core_in if p in listed]
        general = [p for p in cands if (packages["packages"].get(p) or {}).get("general")
                   or p in core or p in stand_in]
        cands = [p for p in cands if p not in general]
        if dec["packages"] != "off":
            general = [g for g in general if g in pinned or not self._kept_out(
                plan, body, router, roles, packages, sh, sess, cls, g, row)]
            for g in general:
                text, err = self.library.text(packages, g)
                if text is None:
                    row["extra"].setdefault("notes", []).append(f"{err}:{g}")
                else:
                    parts.append(("packages", {"kind": "package", "name": g, "text": text}))
            flagged = [g for g in general if g not in core]
            if flagged:
                row["extra"]["package_general"] = flagged
            if [g for g in general if g in core]:
                row["extra"]["package_core"] = [g for g in general if g in core]
        if dec["packages"] != "off" and cands:
            chosen, how = None, None
            if len(cands) == 1:
                chosen, how = cands[0], "only_candidate"
            else:
                pk = (cls, sh.user_hash)
                hit = sess.package_pick.get(pk)
                # FX15: a follow-up in the class the session stays in keeps the package the
                # question picked for that class; the class question decides a new kind of
                # request (`plan.class_new`), the package question is not asked again for it
                held = sess.package_pick.get(("class", cls))
                if hit is None and held is not None and held[0] in cands and \
                        not getattr(plan, "class_new", True) and \
                        schema.package_sticky(router, policy) == "class":
                    hit = (held[0], "sticky")
                if hit is not None:
                    chosen, how = hit
                else:
                    opts = {p: packages["packages"][p]["description"] for p in cands
                            if p in packages["packages"]}
                    opts[NONE_OPTION] = "None of these fits the request."
                    ans, st = self.cascade(
                        sh.user_text or "", opts, router, roles,
                        question=router["classifier"]["package_question"],
                        accept=float(router["classifier"]["package_accept"]), qid="packages",
                        about=_about(cls, classes))
                    row["extra"]["package_stages"] = st
                    if ans is not None and ans.label != NONE_OPTION:
                        chosen, how = ans.label, ans.stage
                    else:
                        how = "none" if ans is not None else "no_answer"
                    if len(sess.package_pick) > 32:
                        sess.package_pick.clear()
                    sess.package_pick[pk] = (chosen, how)
                    if chosen:
                        sess.package_pick[("class", cls)] = (chosen, how)
                    else:
                        sess.package_pick.pop(("class", cls), None)
            row["extra"]["package_choice"] = {"candidates": cands, "chosen": chosen, "how": how}
            if chosen and self._kept_out(plan, body, router, roles, packages, sh, sess, cls,
                                         chosen, row):
                chosen = None
            if chosen:
                text, err = self.library.text(packages, chosen)
                if text is None:
                    row["extra"].setdefault("notes", []).append(f"{err}:{chosen}")
                else:
                    parts.append(("packages", {"kind": "package", "name": chosen, "text": text}))
        if dec["packages"] != "off":
            have = {p["name"] for k, p in parts if p.get("kind") == "package"}
            for name in pinned:
                if name in have or name in sh.loaded_skills:
                    continue
                if self._deliverable_tools(plan, body, packages, name) is not None:
                    row["extra"].setdefault("package_choice", {}).setdefault(
                        "kept_out_skipped", []).append({"name": name, "why": "pinned"})
                text, err = self.library.text(packages, name)
                if text is None:
                    row["extra"].setdefault("notes", []).append(f"{err}:{name}")
                else:
                    parts.append(("packages", {"kind": "package", "name": name, "text": text}))
                    row["extra"].setdefault("package_pinned", []).append(name)
        if fit_note and dec["packages"] != "off":
            parts.append(("packages", {"kind": "tools", "name": "fit", "text": fit_note}))
        if dec["rules"] != "off" and (policy["rules"] or policy["note"]) and \
                not modelfit.FIT["rules"]:
            row["extra"].setdefault("notes", []).append("rules_skipped:model_fit")
        elif dec["rules"] != "off" and (policy["rules"] or policy["note"]):
            text = "\n".join(f"- {r}" for r in policy["rules"])
            if policy["note"]:
                text = (text + "\n\n" if text else "") + policy["note"]
            parts.append(("rules", {"kind": "rules", "name": cls, "text": text}))
        on = [(k, p) for k, p in parts if dec[k] == "on"]
        shadow = [(k, p) for k, p in parts if dec[k] == "shadow"]
        if shadow:
            row["shadow"] = dict(row.get("shadow") or {}, notes=[
                {"decision": k, "kind": p["kind"], "name": p["name"], "chars": len(p["text"])}
                for k, p in shadow])
        if not on:
            return
        key = hashlib.sha256(repr([(cls, p["kind"], p["name"],
                                    hashlib.sha256(p["text"].encode()).hexdigest()[:12])
                                   for _, p in on]).encode()).hexdigest()[:20]
        reason = (f"class {cls} ({row.get('source')}"
                  + (f", {row['confidence']:.2f}" if row.get("confidence") is not None else "")
                  + ")")
        # each part with its text's hash: a later note leaves out a part a live note holds
        # (`Plan._fresh_block`)
        items = [{"kind": k, "text": p["text"], "part": {"kind": p["kind"], "name": p["name"]},
                  "sha": hashlib.sha256(p["text"].encode()).hexdigest()[:12]} for k, p in on]
        plan.block = {"key": key, "text": "\n\n".join(p["text"] for _, p in on),
                      "parts": [it["part"] for it in items], "shas": [it["sha"] for it in items],
                      "kinds": {k for k, _ in on}, "reason": reason, "items": items}

    def _deliverable_tools(self, plan, body, packages, name) -> tuple | None:
        """SRV-615: (the package's deliverable, the offered tools whose declaration makes it)
        when the package has a `deliverable` and the request offers client tools; else None.
        The tools come from their declarations (signals.tool_effects), never from words."""
        if TOOL_FIT == "off" or not plan.active or not signals.offers_tools(body or {}):
            return None
        want = packages_mod.Library.deliverable(packages, name)
        if want is None:
            return None
        # a post is sent to a feed: the declarations read `post`, `publish` and `share` as `send`
        kinds = {"post": ("send",)}.get(want, (want,))
        effects = signals.tool_effects(body)
        return want, sorted(n for n, e in effects.items() if set(kinds) & set(e))

    def _kept_out(self, plan, body, router, roles, packages, sh, sess, cls, name, row) -> bool:
        """SRV-615: True when the package `name`, about to go in, stays out because an offered
        tool delivers its result (a write-message draft where send_email is offered: TC-03 of
        tool-eval-bench, 2026-10-18, the model wrote a message to copy and called no tool).
        Only a package with a `deliverable` on a request that offers client tools is asked
        about: a client that forces a function decided already; otherwise, when an offered tool's
        declaration makes the deliverable, one cascade question, the tool or the text itself,
        decides (with no answer the package stays out, `how: schema`); without such a tool the
        package goes in and nothing is asked (Open WebUI's search tools cost no question). A
        follow-up of the same user message (the tool turns) keeps the first decision. The
        router's `tool_fit.deliverable`: `on` acts, `shadow` journals, `off` asks nothing."""
        fit = schema.tool_fit(router)
        mode = fit.get("deliverable", "on")
        if mode == "off":
            return False
        got = self._deliverable_tools(plan, body, packages, name)
        if got is None:
            return False
        want, tools = got
        key = ("kept", cls, sh.user_hash, name)
        rec = sess.package_pick.get(key)
        if rec is None:
            rec = self._keep_out_question(body, router, roles, sh, want, tools, row)
            sess.package_pick[key] = rec
        else:
            rec = dict(rec, held=True)
        if not rec.get("out"):
            if rec.get("how") not in (None, "skipped"):
                row["extra"].setdefault("package_choice", {}).setdefault(
                    "kept_in", []).append({"name": name, **{k: v for k, v in rec.items()
                                                              if k != "out"}})
            return False
        entry = {"name": name, "why": "tool_delivers", "deliverable": want,
                 **{k: v for k, v in rec.items() if k != "out"}}
        if mode == "shadow":
            sh_tools = dict((row.get("shadow") or {}).get("tools") or {})
            sh_tools.setdefault("kept_out", []).append(entry)
            row["shadow"] = dict(row.get("shadow") or {}, tools=sh_tools)
            return False
        row["extra"].setdefault("package_choice", {}).setdefault("kept_out", []).append(entry)
        return True

    def _keep_out_question(self, body, router, roles, sh, want, tools, row) -> dict:
        """SRV-615: the decision for one deliverable package: {out, how, tools, p}."""
        names = [t.get("function", t).get("name") for t in body.get("tools") or ()
                 if isinstance(t, dict) and isinstance(t.get("function", t), dict)]
        choice = body.get("tool_choice")
        if choice == "required" or isinstance(choice, dict):
            # the client forces a function: it decided already
            return {"out": True, "how": "tool_choice", "tools": tools or names[:8], "p": None}
        if not tools:
            # no offered tool makes the deliverable: nothing to ask, the package goes in
            return {"out": False, "how": "skipped", "tools": [], "p": None}
        text = sh.user_text or ""
        if not text.strip():
            return {"out": True, "how": "schema", "tools": tools, "p": None}
        fit = schema.tool_fit(router)
        listing = "; ".join(_tool_line(t) for t in (body.get("tools") or ())[:12]
                            if isinstance(t, dict)) or "none"
        question = fit["deliverable_question"].replace("{tools}", listing)
        ans, stages = self.cascade(text, dict(fit["deliverable_options"]), router, roles,
                                   question=question,
                                   accept=float(fit["deliverable_accept"]),
                                   qid="deliverable")
        row["extra"].setdefault("deliverable_stages", []).append(
            {"deliverable": want, "stages": stages})
        if ans is not None:
            return {"out": ans.label == "tool", "how": ans.stage, "tools": tools,
                    "p": round(float(ans.confidence), 4)}
        # no answer at the bar: the tool whose declaration makes the deliverable decides
        return {"out": True, "how": "schema", "tools": tools, "p": None}

    def fits(self, plan, packages: dict, name: str) -> bool:
        """SRV-243: the request's tools can carry out the package's steps (always with the
        switch off or on a raw plan)."""
        if TOOL_FIT == "off" or not plan.active:
            return True
        return set(self.library.requires(packages, name)) <= set(plan.tool_caps)

    def _fit(self, plan, router, packages, policy, row) -> tuple[list, list, str | None, set]:
        """SRV-243 (UX2, 2026-10-07): a plain Open WebUI chat whose only tools were search_web
        and fetch_url got the testing and Laya packages for a landing page: they tell the model to
        run a shell and write files, so it called `terminal` and `write_file` ('Tool not found'),
        tried to fetch the library's address with fetch_url, wrote the page into a failed call and
        printed it again (518 s for one HTML file; the snake game 423 s, the 3D garden 522 s).
        Here the class's packages are kept only when the request's tools can do what their steps
        need (`packages.Library.requires`); a package with a `fallback` gives way to it; and when
        any had to go, the turn's note says which tools the chat has and to deliver the files in
        the answer (`schema.tool_fit`). Returns (the class's packages, its core, the note or
        None). The router's `tool_fit.mode` `shadow` journals what would have gone; the switch
        `QWEN38_TOOL_FIT=off` keeps the packages as before. The fourth value: the fallbacks that
        stand in for a general package, general in its place."""
        fit = schema.tool_fit(router)
        listed = list(policy["packages"])
        core = list(policy.get("core", []))
        if TOOL_FIT == "off" or fit["mode"] == "off":
            return listed, core, None, set()
        out, dropped, swapped = [], [], {}
        # PKG3 (2026-10-07): the note speaks of the shell and file tools; a package that leaves
        # only for a tool of its own (render-tools without a render tool) leaves without it, or an
        # opencode turn with a shell read "This chat can write files but has no shell"
        noted = False
        for p in listed:
            use = p
            if not self.fits(plan, packages, p):
                if (set(self.library.requires(packages, p)) - set(plan.tool_caps)) & NOTE_CAPS:
                    noted = True
                fb = self.library.fallback(packages, p)
                use = fb if fb and fb in packages["packages"] and \
                    self.fits(plan, packages, fb) else None
                if use is None:
                    dropped.append(p)
                else:
                    swapped[p] = use
            if use is not None and use not in out:
                out.append(use)
        caps = sorted(plan.tool_caps)
        row["extra"]["tools"] = {"offered": plan.tool_names[:12], "caps": caps,
                                 **({"dropped": dropped} if dropped else {}),
                                 **({"fallback": swapped} if swapped else {})}
        if not dropped and not swapped:
            return listed, core, None, set()
        if fit["mode"] == "shadow":
            row["shadow"] = dict(row.get("shadow") or {}, tools={
                "dropped": dropped, "fallback": swapped})
            return listed, core, None, set()
        if not noted:
            core = [swapped.get(p, p) for p in core if p not in dropped]
            stand_in = {fb for p, fb in swapped.items()
                        if (packages["packages"].get(p) or {}).get("general")}
            return out, core, None, stand_in
        names = ", ".join(plan.tool_names[:8])
        tools = f" (its tools: {names})" if names else ""
        key = "note_write" if "write" in caps else "note_none"
        note = fit[key].replace("{tools}", tools)
        core = [swapped.get(p, p) for p in core if p not in dropped]
        stand_in = {fb for p, fb in swapped.items()
                    if (packages["packages"].get(p) or {}).get("general")}
        return out, core, note, stand_in

    def _reading(self, plan, router, cls, classes, row) -> tuple | None:
        """FX6: the reading line, as a `classify` part, or None. A class a model stage picked on
        the turn it starts, below `reading.below`, with another class with a `reading` at
        `reading.other` or more: build the picked one and say so in one short line, never a
        blocking question. The numbers come from the classifier's own answer; no word decides."""
        cfg = schema.reading(router)
        if cfg["mode"] == "off" or plan.decisions.get("classify") != "on":
            return None
        mine = (classes.get(cls) or {}).get("reading")
        conf = row.get("confidence")
        if not mine or not getattr(plan, "class_new", False) or conf is None or \
                row.get("source") not in ("qwen", "decider") or conf >= float(cfg["below"]):
            return None
        # SRV-255: a first message kept on its top build class under the bar reads its own stage
        below_ok = row.get("kept") == "below_build"
        probs = next((s.get("probs") or {} for s in reversed(row.get("stages") or [])
                      if s.get("label") == cls and not s.get("error")
                      and (below_ok or not s.get("below"))), {})
        others = sorted(((p, c) for c, p in probs.items()
                         if c != cls and (classes.get(c) or {}).get("reading")
                         and float(p) >= float(cfg["other"])), reverse=True)
        if not others:
            return None
        p, other = others[0]
        row["extra"]["reading"] = {"class": cls, "confidence": round(float(conf), 4),
                                   "other": other, "other_p": round(float(p), 4),
                                   "mode": cfg["mode"]}
        text = cfg["text"].replace("{reading}", mine).replace(
            "{other}", classes[other]["reading"])
        if cfg["mode"] == "shadow":
            row["shadow"] = dict(row.get("shadow") or {}, reading=text)
            return None
        return ("classify", {"kind": "reading", "name": cls, "text": text})

    # ---------------------------------------------------------------- J1: snapshots, outcomes
    def _observe(self, plan, body, router, prof, sh, sess, row) -> None:
        """The snapshot and the Tier 0 outcome of a turn whose newest message is a tool result
        (server/conductor/outcomes.py), under the profile's `outcomes` switches; all off (every
        profile stored before J1) leaves the plan untouched. A compaction starts the re-read
        watch. A failure is the row's error and the turn goes on as without it."""
        so = schema.profile_outcomes(prof)
        if all(v == "off" for v in so.values()):
            return
        if so["mode"] == "act" or so["cards"] == "on" or so["evaluate"] == "act" or \
                so["retrieve"] == "on":
            plan.decisions["outcomes"] = "on"       # their notes replay
        if not (sh.tool_turn or sh.compaction) or (so["snapshots"] == "off" and
                                                    so["mode"] == "off" and so["judge"] == "off"):
            return
        cfg = schema.outcomes(router)
        now = time.time()
        with sess.lock:
            if sess.evidence is None:
                sess.evidence = outcomes.Evidence(now)
            ev = sess.evidence
            if sh.compaction:
                outcomes.compaction(ev)
                return
            try:
                snap = outcomes.observe(body, sh, ev, cfg, now=now,
                                        keep_media=so["snapshots"] == "on")
                label, why = outcomes.classify(snap, ev, cfg)
                sig = outcomes.settle(ev, label, why)
            except Exception as exc:                               # noqa: BLE001
                row["extra"].setdefault("errors", []).append(f"snapshot:{type(exc).__name__}")
                print(f"[conductor] the snapshot failed: {type(exc).__name__}: {exc}", flush=True)
                return
        snap["outcome"] = {"label": label, "why": why, "mode": so["mode"]}
        if so["judge"] == "shadow":
            from server.conductor import judge
            plan.judge = judge.questions(snap, cfg)        # asked after the request (step 7)
        snap["_narrow_kept"] = snap.pop("_narrow", None) or []
        if so["snapshots"] == "on":
            plan.snapshot = snap
        plan.outcome = {"label": label, "why": why, "sig": sig, "snap": snap, "cfg": cfg,
                        "so": so, "now": now, "effort": None, "note": None,
                        "modes": self._label_modes(so, cfg, label)}
        if so["mode"] != "off":
            row["extra"]["outcome"] = {"label": label, "mode": so["mode"], "why": why[:2],
                                       "ms": snap["ms"]}
        ac = cfg["evaluate"].get("ask") or {}
        if so["evaluate"] != "off" and ac.get("mode", "off") == "shadow" and \
                any(n["q"] == "failure" for n in snap.get("_narrow_kept") or ()):
            # step 6: Qwen asked in place on the narrowed case, under the engine lock before the
            # turn's own prefill (`ask_in_place`), journaled beside the twin
            n = [x for x in snap["_narrow_kept"] if x["q"] == "failure"][-1]
            plan.ask = {"twin": n.get("twin"), "call": n.get("call")}
        snap.pop("_narrow_kept", None)

    # the outcomes the evaluation points (E-c, step 6) and retrieval (step 8) answer
    EVALUATED = ("stuck", "low_quality")

    @staticmethod
    def _label_modes(so, cfg, label) -> dict:
        """What an outcome may do under the profile's switches: `note` and `effort`, each off,
        shadow or act. The plan's step 3 labels (`acting`) follow `outcomes.mode`; stuck and
        low quality follow `evaluate` (the E-c note, xhigh) and `retrieve` (the package)."""
        out = {"note": "off", "effort": "off", "evaluate": "off", "retrieve": "off"}
        mode = so["mode"] if so["mode"] != "off" else "off"
        if label in cfg["acting"]:
            out["note"] = mode
        if cfg["effort"].get(label):
            out["effort"] = mode
        if label in Router.EVALUATED:
            ev = so["evaluate"]
            rt = {"on": "act", "shadow": "shadow"}.get(so["retrieve"], "off")
            out["evaluate"], out["retrieve"] = ev, rt
            best = max((ev, rt, out["note"]), key=("off", "shadow", "act").index)
            out["note"] = best if best != "off" else out["note"]
            if ev != "off":
                out["effort"] = ev
        return out

    def _outcome_level(self, plan, level: str, row) -> str:
        """The effort after the outcome: the outcome's level (never below the turn's), held for
        `raise_turns` tool turns or until a progress. Shadow journals the level it would set."""
        o = plan.outcome
        if o is None or o["so"]["mode"] == "off":
            return level
        cfg, ev = o["cfg"], plan.session.evidence
        lv = cfg["effort"].get(o["label"]) if o["modes"]["effort"] != "off" else None
        if o["modes"]["evaluate"] != "off":
            lv = cfg["evaluate"]["effort"]
        with plan.session.lock:
            if lv:
                ev.raised = {"level": lv, "left": max(0, int(cfg["raise_turns"]) - 1),
                             "act": o["modes"]["effort"] == "act"}
            elif ev.raised and ev.raised["left"] > 0:
                lv = ev.raised["level"]
                ev.raised["left"] -= 1
            else:
                ev.raised = None
        if not lv or EFFORT_ORDER.index(lv) <= EFFORT_ORDER.index(level):
            return level
        applied = (o["modes"]["effort"] == "act" or (ev.raised or {}).get("act", False)) and \
            plan.decisions["effort"] == "on"
        o["effort"] = {"from": level, "to": lv, "applied": applied}
        row["extra"]["outcome"]["effort"] = o["effort"]
        if not applied:
            row["shadow"] = dict(row.get("shadow") or {}, outcome={"effort": lv, "from": level})
            return level
        return lv

    def _acts_free(self, plan, prof, now: float) -> str | None:
        """None when a note may go in under the profile's caps (2 a request, 6 a session hour,
        the monitor's acts of the session counted), else the cap's name."""
        caps = prof["caps"]
        if int(caps["acts_per_request"]) <= len(plan.acts):
            return "per_request"
        sess = plan.session
        mine = sum(1 for t in sess.acts if now - t < 3600.0)
        try:
            ms = self.monitor_session(sess.key)
            theirs = ms.acts_last_hour(now) if ms is not None else 0
        except Exception:                                          # noqa: BLE001
            theirs = 0
        if max(mine, theirs) >= int(caps["acts_per_session_hour"]):
            return "per_session_hour"
        return None

    def _outcome_note(self, plan, body, router, prof, packages, sh, sess, row) -> None:
        """The outcome's note (act: injected once per signature, under the caps; shadow: what
        would have gone in), and the outcome's journal row (`kind: outcome`)."""
        o = plan.outcome
        if o is None or o["so"]["mode"] == "off":
            return
        cfg, label, why, ev = o["cfg"], o["label"], o["why"], sess.evidence
        nmode = o["modes"]["note"]
        note = {"due": nmode != "off" and label in cfg["notes"], "applied": False,
                "mode": nmode}
        if note["due"]:
            with sess.lock:
                seen = o["sig"] in ev.noted
            text = self._note_text(plan, body, router, packages, sh, sess, label, why, o,
                                   cfg, row, note) if not seen else None
            if seen:
                note["skipped"] = "noted"
            elif text is None:
                note["skipped"] = "no_text"
            else:
                note["chars"] = len(text)
                cap = self._acts_free(plan, prof, o["now"]) if nmode == "act" else None
                if cap is not None:
                    note["cap"] = cap
                else:
                    with sess.lock:
                        ev.noted.add(o["sig"])
                    if nmode == "act":
                        plan.extras.append({"key": f"outcome:{row.get('request_id')}:"
                                                   f"{sess.turns}", "text": text,
                                            "parts": [{"kind": "outcome", "name": label}],
                                            "kinds": {"outcomes"},
                                            "reason": f"outcome {label}"})
                        plan.acts.append(o["now"])
                        with sess.lock:
                            sess.acts.append(o["now"])
                        note["applied"] = True
                    else:
                        row["shadow"] = dict(row.get("shadow") or {}, outcome=dict(
                            (row.get("shadow") or {}).get("outcome") or {}, note=label,
                            chars=len(text)))
        o["note"] = note
        row["extra"]["outcome"]["note"] = {k: v for k, v in note.items() if k != "due"} \
            if note["due"] else None
        self.journal.note("outcome", "outcome", decision=label, session_key=sess.key,
                          request_id=row.get("request_id"),
                          detail={"mode": o["so"]["mode"], "why": why, "effort": o["effort"],
                                  "note": note, "class": row.get("class_id"),
                                  "snapshot": outcomes.facts_row(o["snap"])})

    def _note_text(self, plan, body, router, packages, sh, sess, label, why, o, cfg, row,
                   note) -> str | None:
        """The outcome's note: its template filled; for stuck and low quality the E-c re-plan
        note (evaluate) and the retrieved package (retrieve), as their switches say."""
        snap, modes = o["snap"], o["modes"]
        tmpl = cfg["notes"].get(label)
        if label in self.EVALUATED and modes["evaluate"] != "off":
            tmpl = cfg["notes"]["stuck"]                         # E-c: re-plan from the evidence
            if modes["retrieve"] != "off":
                tmpl += "\n\n{package}"
        elif label in self.EVALUATED and modes["retrieve"] == "off":
            tmpl = tmpl if "{package}" not in (tmpl or "") else None
        if not tmpl:
            return None
        fill = {"why": "; ".join(why[:2]) or label.replace("_", " "),
                "since": f"{snap['snapshot'] - (plan.session.evidence.progress_snap or 0)} tool "
                         f"turns"}
        if "{digest}" in tmpl:
            fill["digest"] = outcomes.digest(snap, label, why, cfg)
        if "{card}" in tmpl:
            fill["card"] = self._tool_card(body, cfg, packages, row)
        if "{package}" in tmpl:
            name, text = self._retrieve(router, packages, sh, sess, label, o, cfg, row)
            note["package"] = name
            if text is None:
                return None
            fill["package"] = text
        for k, v in fill.items():
            tmpl = tmpl.replace("{" + k + "}", v)
        return tmpl.strip()

    def _retrieve(self, router, packages, sh, sess, label, o, cfg, row) -> tuple:
        """Step 8: the package that fits the session's work, among `retrieve.packages` (the
        kind skills), by the router's package question (packages.fit: the decider or Qwen picks a
        description; no word decides), once per outcome signature. (name, text) or (None,
        None)."""
        rc = cfg["retrieve"]
        cands = [p for p in rc["packages"] if (packages["packages"].get(p) or {}).get(
            "enabled", False)]
        if not cands:
            return None, None
        pk = ("retrieve", o["sig"])
        hit = sess.package_pick.get(pk)
        if hit is None:
            _, roles = self.store.peek("roles")
            opts = {p: packages["packages"][p]["description"] for p in cands}
            opts[NONE_OPTION] = "None of these fits the work."
            text = sh.user_text or ""
            line = "files=" + ",".join(dict.fromkeys(
                e for e in (signals.ext_of(f) for f in sh.files[-8:]) if e)) or None
            ans, st = self.cascade(text, opts, router, roles, question=rc["question"],
                                   accept=float(router["classifier"]["package_accept"]),
                                   qid="packages", line=line)
            row["extra"]["retrieve_stages"] = st
            hit = ans.label if ans is not None and ans.label != NONE_OPTION else None
            if len(sess.package_pick) > 32:
                sess.package_pick.clear()
            sess.package_pick[pk] = hit
        if hit is None:
            return None, None
        return hit, self.library.text(packages, hit)[0]

    def _tool_card(self, body, cfg, packages, row) -> str:
        client = row.get("client_kind") or "other"
        name = cfg["cards"]["pitfalls"].get(client)
        pit = self.library.text(packages, name)[0] if name else None
        return outcomes.tool_card(body, cfg, pit, client)

    @staticmethod
    def _identity(plan, sh, row) -> None:
        """SRV-191: who the model is, on the turn whose message asks (schema.IDENTITY_ASK), once
        a session; first among the turn's extra notes. Its own decision type, on under a profile
        that is on on every turn, so the note replays where it sat; a shadow profile journals
        it."""
        if plan.res.mode == "on":
            plan.decisions["identity"] = "on"            # the replay of an earlier turn's note
        if sh.compaction or sh.tool_turn or plan.res.mode not in ("on", "shadow") or \
                not re.search(schema.IDENTITY_ASK, sh.user_text or ""):
            return
        text = schema.identity_text()
        if plan.res.mode == "shadow":
            row["shadow"] = dict(row.get("shadow") or {}, identity={"chars": len(text)})
            return
        plan.extras.insert(0, {"key": f"identity:{hashlib.sha256(text.encode()).hexdigest()[:16]}",
                               "text": text, "parts": [{"kind": "identity", "name": "identity"}],
                               "kinds": {"identity"}, "reason": "identity",
                               "unless": ("identity", "identity")})

    @staticmethod
    def _as_sent(plan, row) -> None:
        """SRV-621: under `prompt: client` nothing of the plan reaches the prompt: the
        turn's note, the extra notes (identity, cards, memory lines), an outcome note, a router
        budget and a closed think block are dropped and journaled by kind (`extra.prompt`), and
        the memory notes' part of the turn with them; `apply_prompt` replays no earlier note
        either. The session monitor's acts are not the plan's and stay."""
        if not getattr(plan, "prompt_client", False):
            return
        dropped = []
        if plan.block is not None:
            dropped += sorted({p.get("kind") or "note" for p in plan.block.get("parts") or ()}
                              or {"note"})
        for x in plan.extras:
            dropped += sorted(x.get("kinds") or {"note"})
        if plan.budget is not None:
            dropped.append("budget")
        if plan.thinking_off:
            dropped.append("thinking_off")
        if plan.acts:
            dropped.append("outcome")
        if getattr(plan, "memory", None) is not None:
            dropped.append("memory")
        plan.block, plan.extras, plan.acts, plan.memory = None, [], [], None
        plan.budget, plan.budget_source, plan.thinking_off = None, "server", False
        plan.changed = False
        row["acted"] = 0
        info = {"mode": "client"}
        if dropped:
            info["dropped"] = dropped
        row["extra"]["prompt"] = info

    @staticmethod
    def _client_cwd(plan, body, sh, sess, row) -> None:
        """FX5: Open WebUI's terminal tools end `run_command`'s description with the terminal's
        live working directory (`\n\nThe current working directory is: <path>`, fetched again for
        every user message). The description sits in the system prompt's tool block, so the first
        `cd` of an agent changed the prompt about 4,800 tokens in and the next user message read
        the whole conversation again (41,749 tokens, 56 s, on a 0.4.0 test build). Under a profile
        that is on, the sentence leaves the description and the same sentence goes into the turn as
        a note, when the directory differs from the last one noted (the first turn, a `cd`): the
        tool block stays the same for the session, and the note replays where it sat. Off
        (`QWEN38_CLIENT_CWD=off`), shadow or raw: the tools as sent."""
        if CLIENT_CWD != "note" or plan.res.mode != "on" or sh.compaction or \
                getattr(plan, "prompt_client", False):
            return
        plan.decisions["client"] = "on"                  # the replay of an earlier turn's note
        path = None
        for t in body.get("tools") or ():
            fn = t.get("function") if isinstance(t, dict) else None
            d = fn.get("description") if isinstance(fn, dict) else None
            m = CWD_LINE.search(d) if isinstance(d, str) else None
            if m:
                path = path or m.group(1).strip()
                fn["description"] = d[:m.start()]
        if not path:
            return
        with sess.lock:
            last = sess.cwd
            key = (last[1] if last is not None and last[0] == path
                   else f"cwd:{hashlib.sha256(path.encode()).hexdigest()[:12]}:{sess.turns}")
            sess.cwd = (path, key)
        row["extra"]["client_cwd"] = {"path_chars": len(path), "changed": last is None or
                                      last[0] != path}
        plan.extras.append({"key": key, "text": CWD_NOTE.format(path=path),
                            "parts": [{"kind": "client", "name": "cwd"}], "kinds": {"client"},
                            "reason": "client working directory"})

    def _cards(self, plan, body, router, prof, packages, sh, cls, row) -> None:
        """The client tool card (B1) and the harness card (B2) for a coding class with tools:
        kept live in the session's notes (in again after a compaction dropped them), each once.
        Shadow journals what would go in."""
        so = schema.profile_outcomes(prof)
        if so["cards"] == "off" or not sh.tools or sh.compaction or \
                plan.decisions["packages"] == "off":
            return
        cfg = schema.outcomes(router)
        cc = cfg["cards"]
        if cls not in cc["classes"]:
            return
        items = [("tools", self._tool_card(body, cfg, packages, row))]
        harness, err = self.library.text(packages, cc["harness"])
        if harness is not None and not self.fits(plan, packages, cc["harness"]):
            harness, err = None, "tools_missing"         # SRV-243: it names Laya's routes
        if harness is not None:
            items.append(("harness", harness))
        elif err:
            row["extra"].setdefault("notes", []).append(f"{err}:{cc['harness']}")
        if so["cards"] == "shadow":
            row["shadow"] = dict(row.get("shadow") or {}, cards=[
                {"name": n, "chars": len(t)} for n, t in items])
            return
        for name, text in items:
            plan.extras.insert(0 if name == "harness" else len(plan.extras), {
                "key": f"card:{name}:{hashlib.sha256(text.encode()).hexdigest()[:16]}",
                "text": text, "parts": [{"kind": "card", "name": name}],
                "kinds": {"outcomes"}, "reason": f"card {name}", "unless": ("card", name)})

    def ask_in_place(self, plan: "Plan", prompt_ids: list[int], tok, runner) -> dict | None:
        """Step 6, shadow: Qwen asked in place on the turn's narrowed case (a run's output with
        no exit code and no runner's counts), under the engine lock before the turn's own
        prefill, through the in-context stage's forward (`runner(ids, letter ids, at)`: the
        prompt up to its last message, the session's notes replayed, plus one hidden question;
        the turn then prefills over these rows). The answer is journaled beside the Tier 0 twin
        (`kind: outcome`, task `in_place`); it changes nothing. A failure is the row's error."""
        ask, plan.ask = plan.ask, None
        o = plan.outcome
        if ask is None or o is None:
            return None
        ac = o["cfg"]["evaluate"]["ask"]
        t0 = time.perf_counter()
        d = {"question": "failure", "twin": ask.get("twin"), "call": ask.get("call"),
             "error": None}
        label = "unknown"
        try:
            text, letters = incontext.listing(ac["options"])
            ids, at = incontext.prompt(plan.replayed_ids(prompt_ids, tok), tok,
                                       ac["question"] + "\n\n" + text)
            logits, info = runner(ids, incontext.letter_ids(tok, letters), at)
            got, conf, probs = incontext.answer(logits, list(ac["options"]))
            d.update(confidence=round(conf, 4), probs={k: round(v, 4) for k, v in probs.items()},
                     prefilled=int(info.get("forwarded", 0)), reused=int(info.get("reused", 0)))
            if conf >= float(ac["accept"]):
                label = got
                d["agree"] = (got == ask.get("twin")) if ask.get("twin") else None
        except ConnectionError:
            raise
        except Exception as exc:                                   # noqa: BLE001
            d["error"] = getattr(exc, "why", None) or type(exc).__name__
        d["ms"] = round((time.perf_counter() - t0) * 1e3, 1)
        if plan.row is not None:
            plan.row["extra"].setdefault("outcome", {})["in_place"] = {
                "label": label, "ms": d["ms"], "error": d["error"]}
        return self.journal.note("outcome", "in_place", decision=label, detail=d,
                                 session_key=plan.session.key if plan.session else None,
                                 request_id=(plan.row or {}).get("request_id"))
