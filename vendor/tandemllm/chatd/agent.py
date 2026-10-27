"""The loop's side of the service: parts by step, tool calls, permissions, the request with the
context guard, compaction (SRV-351, SRV-353). Mixed into `chatd.service.Service`; everything here
runs under the conversation's lock when it touches the conversation, like the rest of the service.
"""

from __future__ import annotations

import datetime as _dt
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from chatd import compaction as comp
from chatd import egress, render
from chatd import tools as tools_mod
from chatd.ids import iso, now_ms, ulid
from chatd.plans import PLAN_REFUSAL, PLAN_TOOLS
from chatd.runs import GuardError
from chatd.sandbox import Gone, Unavailable
from chatd.tasks import IGNORED_CALL, SALVAGE_TOKENS, steer_note, ws_digest
from chatd.store import dumps
from chatd.tools import PathError, Result, ToolError

EVENT_OUTPUT_CAP = 64 * 1024        # tool.output per call in events (CONTRACT section 3)
READ_PARALLEL = 4
RESULT_SLACK = 512                  # over a result's cap: a read's marker, a sandbox note
#: SRV-583: the main agent's reads in Agents mode answered from its context when nothing changed
READ_CACHED = ("read_file", "grep", "glob")


def estimated_split(body: dict, ratio: float | None, skills: str | None) -> dict:
    """A cloud model's prompt split: our engine counts the system prompt's
    and the tool schemas' tokens in its usage frame (`tandem.parts`), a provider does not, so
    these are chatd's estimate from the characters at the chat's learned tokens per character.
    The skill index sits inside the system prompt and counts apart. `estimated` says so."""
    r = ratio or comp.DEFAULT_RATIO
    msgs = body.get("messages") or []
    sys_text = "".join(m.get("content") or "" for m in msgs[:1]
                       if m.get("role") == "system" and isinstance(m.get("content"), str))
    tools = body.get("tools") or []
    sk = int(len(skills) * r) if skills and skills in sys_text else 0
    return {"system": max(0, int(len(sys_text) * r) - sk),
            "tools": int(len(json.dumps(tools, ensure_ascii=False)) * r) if tools else 0,
            "skills": sk, "images": comp.measure({"messages": msgs})[1], "estimated": True}


class ApiConflict(Exception):
    pass


def too_long_words(est: int, window: int, path: list[dict], body: dict, ratio,
                   counted: int | None = None) -> str:
    """The request does not fit: say which part is the cause (SRV-487, VIS-169). Live
    (2026-10-15) one 9.5 MB HTML attachment made a request of 2,875,364 tokens, and the error
    told the user to start a new chat. `counted`: the engine's exact count (SRV-639)."""
    size = f"{counted:,} tokens (the engine's count)" if counted else f"about {est:,} tokens"
    head = (f"This request is {size} and the model's window is {window:,}: there is no room "
            f"left for an answer. ")
    atts = [(int(a.get("tokens") or 0), a, m) for m in path if m.get("role") == "user"
            for a in m.get("attachments") or [] if a.get("kind") == "text" and render.inlined(a)]
    big = max(atts, key=lambda x: x[0], default=None)
    if big is not None and big[0] * 2 > est:
        said = " ".join(render.text_of(big[2].get("parts") or []).split())
        said = f' ("{said[:60]}{"..." if len(said) > 60 else ""}")' if said else ""
        return head + (f"The cause is the attachment {big[1].get('name') or 'file'} in your "
                       f"message{said}: about {big[0]:,} tokens. Edit that message and leave "
                       f"the file out, or attach a shorter one.")
    sizes = [(comp.estimate({"messages": [m]}, ratio), m) for m in body.get("messages") or []]
    n, m = max(sizes, key=lambda x: x[0], default=(0, {}))
    role = m.get("role")
    what = ("the system prompt with the skills" if role == "system" else
            "a tool result" if role == "tool" else "one of your messages" if role == "user"
            else "an answer")
    return head + (f"The largest part is {what}, about {n:,} tokens. Edit an earlier message to "
                   f"branch off before it, or start a new chat.")


def st_is_result(status: str | None) -> bool:
    """A call the model gets a result of: it ran (done, failed), not stopped or ignored."""
    return status in (None, "done", "failed")


class ToolCtx:
    """What an executor may use for one call."""

    def __init__(self, svc, run, part: dict):
        self.svc, self.run, self.part = svc, run, part
        self.call_id = part["id"]
        self.run_id = run.id
        self.cid = svc.root_cid(run.conv.id)     # a worker works in its chat's sandbox and files
        self.ws = svc.workspace(self.cid)
        self.engine = svc.engine
        self._out: list[str] = []
        self.quiet = False
        self.oom = False                     # the last exec lost a process at the memory limit
        self.note: str | None = None         # for the result: the sandbox restarted
        self.phase, self.words = "tool", None    # the run's phase while it runs (verify: its own)
        self.agents = bool(getattr(run, "agents", False))  # a main agent in Agents mode (SRV-583)
        # SRV-639: this call's share of the room the step's results may take, in characters
        self.cap: int | None = getattr(run, "result_cap", None)

    def exec(self, cmd: list[str], timeout: float = 600, quiet: bool = False):
        sb = self.svc.sandboxes

        def waiting(why):
            self.run.set_phase("tool", f"Waiting to start the sandbox ({why})")
        sb.ensure(self.cid, cancel=self.run.cancel, on_wait=waiting)
        self.run.set_phase(self.phase, self._running_words())
        self.quiet = quiet
        before = sb.oom_kills(self.cid)
        try:
            code, timed_out = sb.run(self.cid, cmd, timeout=timeout, on_output=self.output,
                                     cancel=self.run.cancel)
        except Gone:
            # gone between the look and the exec: a new one, for this call
            sb.ensure(self.cid, cancel=self.run.cancel, on_wait=waiting)
            before = sb.oom_kills(self.cid)
            code, timed_out = sb.run(self.cid, cmd, timeout=timeout, on_output=self.output,
                                     cancel=self.run.cancel)
        self.note = self.note or sb.take_restarted(self.cid)
        after = sb.oom_kills(self.cid)
        # the cgroup's count when it can be read (a pipe's writer killed at the limit leaves its
        # reader an exit 1); else a SIGKILL chatd did not send: 137 from bash, 9 from setsid when
        # the killed process is the command itself
        self.oom = (after > before if before is not None and after is not None else
                    code in (9, 137) and not timed_out and not self.run.cancel.is_set())
        return code, timed_out

    def _running_words(self) -> str:
        if self.words:
            return self.words
        name = self.part.get("name")
        if name == "bash":
            try:
                cmd = json.loads(self.part.get("input") or "{}").get("command") or ""
            except ValueError:
                cmd = ""
            return f"Running bash: {cmd[:120]}"
        if name == "web_search":
            try:
                q = json.loads(self.part.get("input") or "{}").get("query") or ""
            except ValueError:
                q = ""
            return f"Searching the web for \"{str(q)[:120]}\""
        return f"Running {name}"

    def output(self, text: str) -> None:
        self._out.append(text)
        if not self.quiet:
            self.svc.tool_output(self.run, self.part, text)

    def output_text(self) -> str:
        return "".join(self._out)

    def fetch(self, url: str):
        return egress.fetch(url)

    def search(self, query, max_results=None) -> dict:
        return self.svc.run_web_search(self.cid, query, max_results)

    def present(self, pane: dict) -> None:
        self.svc.emit(self.run.conv, {"type": "pane.open", "pane": pane}, run_id=self.run_id)

    def changed(self, path: str) -> None:
        pass


class AgentMixin:
    # ================================================================== parts by step
    def new_step(self, run) -> None:
        with run.conv.lock:
            self.flush(run)
            if run.open_kind == "thinking":
                self.close_thinking(run)
            run.open_kind, run.open_index = None, None

    def _pending_add(self, run, key: tuple, text: str) -> None:
        run.pend[key] = run.pend.get(key, "") + text

    def append(self, run, kind: str, text: str) -> None:
        conv = run.conv
        with conv.lock:
            if run.open_kind != kind:
                self.flush(run)
                if run.open_kind == "thinking":
                    self.close_thinking(run)
                part = ({"type": "thinking", "text": "", "tokens": None, "done": False}
                        if kind == "thinking" else {"type": "text", "text": ""})
                part["step"] = run.step
                conv.live["parts"].append(part)
                run.open_kind, run.open_index = kind, len(conv.live["parts"]) - 1
                if kind == "thinking":
                    run.think_started = time.monotonic()
                self.emit(conv, {"type": "part.added", "message_id": run.message_id,
                                 "part": dict(part)}, run_id=run.id)
            self._pending_add(run, (run.open_index, "text"), text)
            if time.monotonic() - run.last_flush >= 0.05:
                self.flush(run)

    def add_call(self, run, call_id: str, name: str, **extra) -> int:
        """A tool_call part in this step; `extra`: fields of a call the harness makes itself
        (chatd/verify.py: `harness`, `source`, `input`)."""
        conv = run.conv
        with conv.lock:
            self.flush(run)
            if run.open_kind == "thinking":
                self.close_thinking(run)
            run.open_kind, run.open_index = None, None
            tool = self.tool(name)
            part = {"type": "tool_call", "id": call_id, "name": name,
                    "source": tool.source if tool else {"kind": "builtin"},
                    "risk": tool.risk if tool else None, "input": "", "output": None,
                    "status": "running", "started_at": None, "ended_at": None,
                    "permission": None, "pane": None, "step": run.step, **extra}
            conv.live["parts"].append(part)
            self.emit(conv, {"type": "part.added", "message_id": run.message_id,
                             "part": dict(part)}, run_id=run.id)
            return len(conv.live["parts"]) - 1

    def call_part(self, run, k: int) -> dict:
        return run.conv.live["parts"][k]

    def append_input(self, run, k: int, text: str) -> None:
        with run.conv.lock:
            self._pending_add(run, (k, "input"), text)

    def tool_output(self, run, part: dict, text: str) -> None:
        with run.conv.lock:
            self._pending_add(run, (part["id"], "output"), text)

    def flush(self, run) -> None:
        conv = run.conv
        with conv.lock:
            run.last_flush = time.monotonic()
            if not run.pend or conv.live is None:
                return
            pend, run.pend = run.pend, {}
            parts = conv.live["parts"]
            for (key, field), delta in pend.items():
                if field == "output":
                    part = next((p for p in parts if p.get("id") == key), None)
                    if part is None:
                        continue
                    sent = run.out_sent.get(key, 0)
                    if sent >= EVENT_OUTPUT_CAP:
                        continue
                    delta = delta[:EVENT_OUTPUT_CAP - sent]
                    run.out_sent[key] = sent + len(delta)
                    out = part.get("output") or {"content": "", "is_error": False, "bytes": 0}
                    out["content"] += delta
                    out["bytes"] += len(delta.encode())
                    part["output"] = out
                    self.emit(conv, {"type": "tool.output", "message_id": run.message_id,
                                     "call_id": key, "delta": delta}, run_id=run.id)
                    continue
                part = parts[key]
                if field == "input":
                    part["input"] = (part.get("input") or "") + delta
                else:
                    part["text"] = (part.get("text") or "") + delta
                ev = {"type": "part.delta", "message_id": run.message_id, "index": key,
                      "delta": delta}
                ev["field"] = field
                self.emit(conv, ev, run_id=run.id)

    def drop_step(self, run) -> None:
        """A stream that broke while a call was being written: those calls end `cancelled`, and
        the retry is a step of its own."""
        with run.conv.lock:
            self.flush(run)
            for k in run.step_parts:
                p = run.conv.live["parts"][k]
                p["status"], p["ended_at"] = "cancelled", iso(now_ms())
                self.emit(run.conv, {"type": "tool.done", "message_id": run.message_id,
                                     "call_id": p["id"], "status": "cancelled",
                                     "is_error": False}, run_id=run.id)
            run.step_parts, run.tc_parts = [], {}
        run.step += 1

    def step_empty(self, run) -> bool:
        live = run.conv.live
        if live is None:
            return True
        return not any((p.get("text") or "").strip() or p.get("type") == "tool_call"
                       for p in live["parts"] if p.get("step") == run.step
                       and p.get("type") != "thinking")

    def queued(self, conv) -> bool:
        return self.store.one("SELECT 1 FROM queue WHERE conversation_id = ? LIMIT 1",
                              (conv.id,)) is not None

    # ================================================================== the registry
    def tool(self, name: str):
        t = tools_mod.BY_NAME.get(name)
        if t is not None:
            return t
        mcp = getattr(self, "mcp", None)
        t = mcp.tool(name) if mcp is not None else None
        owui = getattr(self, "owui", None)
        return t if t is not None or owui is None else owui.tool(name)

    def _prefs(self) -> dict:
        return {r["tool"]: dict(r) for r in self.store.all("SELECT * FROM tool_prefs")}

    def available_builtins(self) -> list:
        ok = []
        sandbox = self.sandboxes.available()
        library = self.library_available()
        for t in tools_mod.BUILTINS:
            if t.name in tools_mod.MODE_TOOLS:      # a mode's tool (task), never the picker's
                continue
            if t.sandbox and not sandbox:
                continue
            if t.needs == "library" and not library:
                continue
            ok.append(t)
        return ok

    def tools_for(self, settings: dict, info: dict, cid: str | None = None,
                  listing: bool = True, agents: bool = False) -> list:
        """The chat's tools, in the fixed order: built-ins, MCP servers by name, then imported
        Open WebUI toolkits by name (they need the sandbox). `listing` False: an MCP server whose
        tools were never listed is not started for it (the cache peek's body). A task
        conversation's: the worker's (tools.py WORKER_TOOLS of the chat's enabled ones, nothing
        else). `agents`: the main agent of a run in Agents mode (VIS-E15), with `task`."""
        if not info.get("tools", True):
            return []
        prefs = self._prefs()
        enabled = set(settings.get("tools") or [])
        if cid and (settings.get("plan_mode") or agents or self.todos(cid)):
            enabled.add("todo_write")
        out = [t for t in self.available_builtins() if t.name in enabled and
               (prefs.get(t.name) is None or prefs[t.name]["enabled"] in (None, 1))]
        if any(t.needs == "web_search" for t in out) and \
                self.web_search_settings(settings)["provider"] == "off":
            # SRV-469: no provider for this chat, no search tool in its request
            out = [t for t in out if t.needs != "web_search"]
        if cid and self.root_cid(cid) != cid:
            return [t for t in out if t.name in tools_mod.WORKER_TOOLS]
        # SRV-363: plan mode runs the read tools and the todo list only; since SRV-663 the request
        # keeps every tool (the run refuses the others, `PLAN_REFUSAL`): a tool list that changed
        # with the mode changed the prompt's head, and the turn into plan mode and the turn out
        # of it read the whole conversation again (84,423 tokens cold, live 2026-10-21 03:31)
        if agents:
            # SRV-583: no bash, write_file or edit_file; the todo list and task
            out = [t for t in out if t.name in tools_mod.AGENTS_TOOLS] + [tools_mod.BY_NAME["task"]]
        mcp = getattr(self, "mcp", None)
        if mcp is not None:
            out += mcp.tools_for(settings, cid if listing else None)
        owui = getattr(self, "owui", None)
        if owui is not None and self.sandboxes.available():
            out += [t for t in owui.tools_for()
                    if prefs.get(t.name) is None or prefs[t.name]["enabled"] in (None, 1)]
        return out

    def tool_infos(self) -> list[dict]:
        prefs = self._prefs()
        return [self._info(t, prefs) for t in self.available_builtins() + (
            self.mcp.all_tools() if getattr(self, "mcp", None) else []) + (
            self.owui.all_tools() if getattr(self, "owui", None) else [])]

    def _info(self, t, prefs: dict) -> dict:
        p = prefs.get(t.name) or {}
        default = {"allow": "always", "ask": "ask", "deny": "never"}[t.default]
        return {"id": t.name, "name": t.name, "description": t.description,
                "source": t.source, "enabled": p.get("enabled") in (None, 1),
                "policy": p.get("policy") or default, "risk": t.risk,
                "schema_tokens": len(json.dumps(t.openai())) // 4}

    def set_tool(self, name: str, patch: dict) -> dict:
        if self.tool(name) is None:
            from chatd.service import ApiError
            raise ApiError(404, "not_found", "no such tool")
        cur = self._prefs().get(name) or {}
        enabled = patch.get("enabled", None if cur.get("enabled") is None else bool(cur["enabled"]))
        policy = patch.get("policy", cur.get("policy"))
        if policy not in (None, "ask", "always", "never"):
            from chatd.service import ApiError
            raise ApiError(400, "bad_request", "policy is ask, always or never")
        self.store.run("INSERT OR REPLACE INTO tool_prefs VALUES (?, ?, ?)",
                       (name, None if enabled is None else int(bool(enabled)), policy))
        return self._info(self.tool(name), self._prefs())

    # ================================================================== permissions
    def rules(self, cid: str | None = None) -> list[dict]:
        rows = self.store.all("SELECT * FROM permission_rules ORDER BY created")
        return [{"id": r["id"], "tool": r["tool"], "match": r["match"], "decision": r["decision"],
                 "scope": "chat" if r["conversation_id"] else "all",
                 "conversation_id": r["conversation_id"], "created_at": iso(r["created"])}
                for r in rows if cid is None or r["conversation_id"] in (None, cid)]

    def add_rule(self, tool: str, match: str | None, decision: str, cid: str | None) -> dict:
        """A rule, once: the same tool, match, decision and scope give back the saved one."""
        same = next((r for r in self.rules() if r["tool"] == tool and r["match"] == (match or None)
                     and r["decision"] == decision and r["conversation_id"] == cid), None)
        if same is not None:
            return same
        rid = ulid()
        self.store.run("INSERT INTO permission_rules VALUES (?, ?, ?, ?, ?, ?)",
                       (rid, tool, match or None, decision, cid, now_ms()))
        return next(r for r in self.rules() if r["id"] == rid)

    def delete_rule(self, rid: str) -> None:
        self.store.run("DELETE FROM permission_rules WHERE id = ?", (rid,))

    def _decide(self, cid: str, tool, args: dict, settings: dict) -> tuple[str, str | None]:
        """(allow | ask | deny, the permission mode that allowed it or None): the rules, then the
        chat's mode (SRV-387/SRV-391, tools.decide_mode)."""
        rules = self.rules(cid)
        chat = [r for r in rules if r["conversation_id"] == cid]
        glob = [r for r in rules if r["conversation_id"] is None]
        pref = (self._prefs().get(tool.name) or {}).get("policy")
        mode = self.permission_mode(settings)
        fetched = self._fetched_domains(cid) if tool.name == "web_fetch" and mode == "auto" else ()
        return tools_mod.decide_mode(tool, args, chat, glob, settings, pref, mode, frozenset(fetched))

    def _fetched_domains(self, cid: str) -> set[str]:
        """The domains this chat's web_fetch calls have reached (a call that ran: done or failed
        after it was allowed), which auto mode fetches again without asking."""
        out = set()
        conv = self.conv(cid)
        live = (conv.live or {}).get("parts") or []
        rows = self.store.all("SELECT parts FROM messages WHERE conversation_id = ? AND role = "
                              "'assistant'", (cid,))
        for parts in [json.loads(r["parts"]) for r in rows] + [live]:
            for p in parts:
                if p.get("type") == "tool_call" and p.get("name") == "web_fetch" and \
                        p.get("status") in ("done", "failed"):
                    try:
                        url = json.loads(p.get("input") or "{}").get("url")
                    except ValueError:
                        continue
                    if url:
                        out.add(tools_mod.domain_of(str(url)))
        return out

    def answer_permission(self, cid: str, call_id: str, body: dict) -> None:
        """POST /permissions/{call}: the first answer wins; a second device gets 409."""
        from chatd.service import ApiError
        conv = self.conv(cid)
        run = conv.run
        w = run.waiting.get(call_id) if run is not None else None
        if w is None and run is not None and run.child is not None:
            w = run.child.waiting.get(call_id)          # a worker's prompt, shown in its chat
        if w is None or w["answer"] is not None:
            raise ApiError(409, "already_answered", "this prompt is not open (answered on another "
                                                    "device, or the run ended)")
        decision = body.get("decision")
        if decision not in ("once", "always", "deny"):
            raise ApiError(400, "bad_request", "decision is once, always or deny")
        w["answer"] = {"decision": decision, "scope": body.get("scope") or "chat",
                       "note": str(body.get("note") or "")[:2000]}
        w["event"].set()

    def _permit(self, run, part: dict, tool, args: dict, settings: dict) -> tuple[str, str]:
        """allow, deny (with the model's note) or cancel. Who allowed a call that runs is the
        part's `approved` (and tool.started's): user (the prompt), rule (a rule, an always
        policy, auto-accept edits), auto or yolo (the chat's mode), None (the tool's default). A
        call the mode allowed also leaves a `tool.auto_approved` event in the chat's log."""
        if tool.name == "task":
            return "allow", ""          # each call of its worker is decided on its own
        root = self.root_cid(run.conv.id)               # a worker's rules are its chat's
        d, by = self._decide(root, tool, args, settings)
        if d == "allow" and by:
            conv = run.conv
            with conv.lock:
                part["approved"] = by
                self.emit(conv, {"type": "tool.auto_approved", "message_id": run.message_id,
                                 "call_id": part["id"], "name": tool.name, "mode": by,
                                 "summary": tools_mod.summary(tool.name, args)}, run_id=run.id)
            return "allow", ""
        if d == "allow":
            part["approved"] = "rule" if tool.default != "allow" or \
                (settings.get("policies") or {}).get(tool.name) == "always" else None
            return "allow", ""
        if d == "deny":
            return "deny", "A permission rule denies this call."
        conv = run.conv
        allows = [r for r in self.rules(root) if r["decision"] == "allow"]
        rule = tools_mod.suggested_rule(tool.name, args, allows)
        words = tools_mod.summary(tool.name, args)
        w = {"event": threading.Event(), "answer": None}
        with conv.lock:
            self.flush(run)
            part["status"] = "asking"
            part["permission"] = {"summary": words, "suggested_rule": rule}
            run.waiting[part["id"]] = w
            self.emit(conv, {"type": "tool.asking", "message_id": run.message_id,
                             "call_id": part["id"], "summary": words, "risk": tool.risk,
                             "suggested_rule": rule}, run_id=run.id)
        run.set_phase("permission", f"Waiting for permission: {tool.name}",
                      tool={"call_id": part["id"], "name": tool.name})
        self.emit_global({"type": "tool.asking", "conversation_id": root,
                          "call_id": part["id"], "summary": words})
        self.summary_changed(self.conv(root), status="waiting")
        # the answer can take hours: the run's engine slot goes back meanwhile (live: one chat
        # waiting for a bash prompt held one of the two, and two other chats ran one at a time)
        self.slots.release_if(run)
        w["event"].wait()
        run.waiting.pop(part["id"], None)
        part["permission"] = None
        if w["answer"] is None:
            return "cancel", ""
        a = w["answer"]
        if a["decision"] == "deny":
            note = a["note"]
            return "deny", "The user denied this call." + (f" Their note: {note}" if note else "")
        if a["decision"] == "always":
            scope = None if a["scope"] == "all" else root
            for m in rule["matches"] if "matches" in rule else [rule.get("match")]:
                self.add_rule(rule["tool"], m, "allow", scope)
        part["approved"] = "user"
        return "allow", ""

    # ================================================================== running the calls
    def run_tools(self, run, idxs: list[int]) -> str:
        conv = run.conv
        self.new_step(run)
        if run.ignore_calls:                        # a worker's report-only call: nothing runs
            for k in idxs:
                self._finish_call(run, conv.live["parts"][k], Result(IGNORED_CALL),
                                  status="ignored")
            return "ok"
        with conv.lock:
            settings = self.settings_of(conv.id)
            parts = [conv.live["parts"][k] for k in idxs]
        info = self.model_info(settings.get("model"))
        enabled = {t.name: t for t in self.tools_for(settings, info, conv.id, agents=run.agents)}
        jobs = []
        tasks = 0
        for part in parts:
            tool = enabled.get(part.get("name") or "")
            try:
                args = tools_mod.parse_args(part.get("input") or "")
            except ToolError as exc:
                self._finish_call(run, part, Result(str(exc), is_error=True))
                continue
            if settings.get("plan_mode") and part.get("name") not in PLAN_TOOLS and (
                    tool is not None or self.tool(part.get("name") or "")):
                self._finish_call(run, part, Result(PLAN_REFUSAL, is_error=True))
                continue
            if tool is None and run.agents and part.get("name") in tools_mod.BY_NAME:
                self._finish_call(run, part, Result(tools_mod.AGENTS_REFUSAL, is_error=True))
                continue
            if tool is None:
                self._finish_call(run, part, Result(
                    f"The tool {part.get('name')!r} is not available in this chat. Available: "
                    f"{', '.join(enabled) or 'none'}.", is_error=True))
                continue
            if tool.name == "task":
                tasks += 1
                if tasks > 1:                       # VIS-E15: one worker at a time, in call order
                    self._finish_call(run, part, Result(tools_mod.ONE_TASK, is_error=True))
                    continue
            sig = (tool.name, json.dumps(args, sort_keys=True))
            run.recent_calls.append(sig)
            if len(run.recent_calls) == 3 and len(set(run.recent_calls)) == 1:
                run.notice = (f"The model called {tool.name} with the same arguments three times "
                              f"in a row; the run stopped here.")
                self._finish_call(run, part, Result("Stopped: the same call three times in a row.",
                                                    is_error=True))
                continue
            jobs.append((part, tool, args))
        run.result_cap = self._result_cap(run, info, len(jobs))
        i = 0
        while i < len(jobs):
            if run.cancel.is_set():
                for part, _, _ in jobs[i:]:
                    self._finish_call(run, part, Result("stopped"), status="cancelled")
                return "cancelled"
            if run.sandbox_stuck:
                for part, _, _ in jobs[i:]:
                    self._finish_call(run, part, Result("Not run: the sandbox failed three times "
                                                        "in a row.", is_error=True))
                return "ok"
            part, tool, args = jobs[i]
            group = [jobs[i]]
            if tool.risk == "read" and not tool.sandbox:
                while i + len(group) < len(jobs) and len(group) < READ_PARALLEL:
                    nxt = jobs[i + len(group)]
                    if nxt[1].risk != "read" or nxt[1].sandbox:
                        break
                    group.append(nxt)
            allowed = []
            for part, tool, args in group:
                verdict, note = self._permit(run, part, tool, args, settings)
                if verdict == "cancel":
                    self._finish_call(run, part, Result("stopped"), status="cancelled")
                    return "cancelled"
                if verdict == "deny":
                    self._finish_call(run, part, Result(note), status="denied")
                    continue
                allowed.append((part, tool, args))
            if len(allowed) > 1:
                with ThreadPoolExecutor(READ_PARALLEL) as pool:
                    list(pool.map(lambda j: self._exec_call(run, *j), allowed))
            for j in allowed if len(allowed) == 1 else []:
                self._exec_call(run, *j)
            i += len(group)
        return "cancelled" if run.cancel.is_set() and run.stop_mode else "ok"

    @staticmethod
    def _result_cap(run, info: dict, calls: int) -> int | None:
        """SRV-639: one call's share of what the step's results may add to the next request
        (chatd/compaction.py `step_budget`), in characters. That request holds the last one (the
        engine's count when its usage came) and this call's answer."""
        window = int(info.get("max_len") or 0)
        if not window or not calls:
            return None
        u = (run.usage or {}).get("usage") or {}
        used = int(u.get("prompt_tokens") or 0) or run.request_counted or run.request_est
        used += max(0, run.tokens - run.call_base[0])
        budget = comp.step_budget(window, used, run.conv.memory_limit)
        return comp.result_chars(budget, calls, run.conv.tok_ratio)

    def _exec_call(self, run, part: dict, tool, args: dict) -> None:
        conv = run.conv
        with conv.lock:
            part["status"], part["started_at"] = "running", iso(now_ms())
            self.emit(conv, {"type": "tool.started", "message_id": run.message_id,
                             "call_id": part["id"], "pane": None,
                             "approved": part.get("approved")}, run_id=run.id)
        ctx = ToolCtx(self, run, part)
        if not tool.sandbox:
            run.set_phase("tool", ctx._running_words(), tool={"call_id": part["id"],
                                                               "name": tool.name})
        if tool.risk in ("write", "exec", "task") and not run.snapshotted:
            run.snapshotted = True
            try:
                self.snaps.take(ctx.cid, ctx.ws, run.id, "before")
            except OSError:
                pass
        failure = None                       # the sandbox's own, not the command's
        if run.agents and tool.name in READ_CACHED:
            # SRV-583: the same read of an unchanged workspace whose result is still in the
            # context answers with that call's id (chatd/tasks.py `_read_hit`)
            part["digest"] = ws_digest(ctx.ws)
            hit = self._read_hit(run, part, tool.name, args)
            if hit is not None:
                self._finish_call(run, part, hit, ctx=ctx)
                return
        try:
            res = tool.executor(ctx, args)
        except (ToolError, PathError, Unavailable, egress.Refused) as exc:
            res = Result(str(exc), is_error=True, data=getattr(exc, "data", None))
            if isinstance(exc, Unavailable):
                failure = ((exc.data or {}).get("sandbox") or {}).get("reason") or str(exc)
        except tools_mod.ChangedError:
            res = Result("the file changed meanwhile; read it again", is_error=True)
        except FileNotFoundError:
            res = Result("no such file", is_error=True)
        except Exception as exc:                                    # noqa: BLE001
            self.log(f"[tool] {part['id']} {tool.name} failed: {type(exc).__name__}")
            res = Result(f"the tool failed ({type(exc).__name__})", is_error=True)
            failure = res.content
        if ctx.note:
            res.content = f"{ctx.note}\n{res.content}"
        if tool.sandbox:
            self._sandbox_streak(run, failure, res.content)
        status = res.status or ("cancelled" if run.cancel.is_set() and run.stop_mode else None)
        self._finish_call(run, part, res, status=status, ctx=ctx)

    def _sandbox_streak(self, run, failure: str | None, words: str) -> None:
        """The same sandbox failure three calls in a row ends the run, whatever the calls were
        (the same-call guard does not see a model that changes its command each time)."""
        if failure is None:
            run.sandbox_errors.clear()
            return
        run.sandbox_errors.append(failure)
        if len(run.sandbox_errors) == 3 and len(set(run.sandbox_errors)) == 1 and \
                not run.notice:
            run.sandbox_stuck = True
            run.notice = (f"The sandbox failed three times in a row ({words[:200]}); the run "
                          f"stopped here. Continue goes on from here.")

    def _finish_call(self, run, part: dict, res: Result, status: str | None = None,
                     ctx: ToolCtx | None = None) -> None:
        conv = run.conv
        images = []
        for mime, data in res.images:
            images.append(self.store_blob(conv.id, f"{part['id']}.png", mime, data))
        with conv.lock:
            self.flush(run)
            if run.parent is not None and images:
                # a worker's screenshots by id, for its report's Checks (chatd/tasks.py)
                part["shot"] = f"shot-{1 + sum(1 for p in conv.live['parts'] if p.get('shot'))}"
                res.content += f"\nscreenshot id: {part['shot']}"
            if run.parent is not None and run.notes and st_is_result(status):
                # SRV-585: a steer the user typed meanwhile, appended to the result when it is
                # stored, so every later request renders the same bytes (append-only)
                res.content += "".join("\n\n" + steer_note(t, x) for t, x in run.notes)
                run.notes = []
            ui, cut = tools_mod.for_ui(res.content)
            st = status or ("failed" if res.is_error else "done")
            out = {"content": ui, "is_error": bool(res.is_error) or st == "failed",
                   "bytes": len(res.content.encode()), "truncated": cut,
                   "exit_code": res.exit_code, "images": images}
            if st in ("denied", "cancelled") and not res.content:
                out = None
            part["output"] = out
            limit, edge = (tools_mod.AGENTS_LIMIT_CHARS, tools_mod.AGENTS_EDGE_CHARS) \
                if run.agents else (tools_mod.MODEL_LIMIT_CHARS, tools_mod.MODEL_EDGE_CHARS)
            if run.result_cap and run.result_cap < limit:
                # SRV-639: the step's share of the room left; a read cuts itself to it at whole
                # lines, its marker and a sandbox note go on top
                limit, edge = run.result_cap, run.result_cap // 4
            part["model_content"] = tools_mod.for_model(ctx or ToolCtx(self, run, part), part["id"],
                                                        res.content, limit + RESULT_SLACK, edge)
            part["status"], part["ended_at"] = st, iso(now_ms())
            if res.pane:
                part["pane"] = res.pane
            if out is not None and res.data is not None:
                out["data"] = res.data
            self.emit(conv, {"type": "tool.done", "message_id": run.message_id,
                             "call_id": part["id"], "status": st,
                             "is_error": bool(out and out["is_error"]), "output": out},
                      run_id=run.id)

    # ================================================================== the request
    def _compaction_for(self, cid: str, path: list[dict]) -> tuple | None:
        """The newest compaction on `path`: (its summary, the first message kept, None, its time)
        for one before a user turn; for one inside a long run (boundary `<message id>@<step>`),
        (its summary, that run's user message, (the run's message index, its first step kept),
        its time). The time (ISO 8601): the long results of calls that ended before it are stubs
        in the request (chatd/render.py `result_for`)."""
        ids = {m["id"]: i for i, m in enumerate(path) if m.get("id")}
        best = None
        for r in self.store.all("SELECT * FROM compactions WHERE conversation_id = ? ORDER BY "
                                "created", (cid,)):
            mid, _, step = str(r["boundary_id"]).partition("@")
            if mid not in ids:
                continue
            i = ids[mid]
            if step:
                u = next((k for k in range(i - 1, -1, -1) if path[k]["role"] == "user"), 0)
                best = (r["summary"], u, (i, int(step)), iso(r["created"]))
            else:
                best = (r["summary"], i, None, iso(r["created"]))
        return best

    @staticmethod
    def _step_from(c) -> tuple[int, int] | None:
        return (c[2][0] - c[1], c[2][1]) if c and c[2] else None

    def compactions(self, cid: str, path: list[dict]) -> list[dict]:
        ids = {m["id"] for m in path}
        return [{"id": r["id"], "at": iso(r["created"]), "before_tokens": r["before_tokens"],
                 "after_tokens": r["after_tokens"], "summary": r["summary"],
                 "duration_ms": r["duration_ms"], "boundary_id": r["boundary_id"]}
                for r in self.store.all("SELECT * FROM compactions WHERE conversation_id = ? "
                                        "ORDER BY created", (cid,))
                if str(r["boundary_id"]).partition("@")[0] in ids]

    def _system(self, settings: dict, tools: list, summary: str | None,
                worker: str | None = None, agents: str | None = None) -> str | None:
        """`worker`: a task run's block (chatd/tasks.py `_head`), in place of the chat's prompt;
        `agents`: the main agent's paragraph in Agents mode."""
        prompt = None
        if settings.get("prompt_id") and not worker:
            r = self.store.one("SELECT template FROM prompts WHERE id = ?",
                               (settings["prompt_id"],))
            prompt = r["template"] if r else None
        skills = self.skill_index(settings) if hasattr(self, "skill_index") else None
        today = _dt.date.today().isoformat() if tools else None
        # the plan rule rides on the plan turn's user message (`plans.plan_onto`), not here: the
        # system block is the prompt's head, the same in and out of plan mode (SRV-663)
        return render.system_block(today=today, tools=bool(tools), summary=summary,
                                   skills=skills, prompt=prompt, worker=worker, agents=agents)

    def _request(self, settings: dict, info: dict, tools: list, path: list[dict], c,
                 head: dict | None = None, **kw) -> dict:
        """The chat's request over `path` under the compaction `c` (`_compaction_for`'s tuple, or
        None): the system block with its summary, the messages from its first kept one on, its
        steps and its stubs. `head`: who asks (`_head`: a worker's block)."""
        b = c[1] if c else 0
        ours = info.get("backend") == "engine"
        if "keep_reasoning_from" in kw:
            kw["keep_reasoning_from"] = max(0, kw["keep_reasoning_from"] - b)
        return render.body(settings.get("model") or info.get("id") or "default", settings,
                           path[b:], system=self._system(settings, tools, c[0] if c else None,
                                                         **(head or {})),
                           vision=bool(info.get("vision")), blob=self._blob_of,
                           tools=[t.openai() for t in tools],
                           tandem=self.tandem_field(settings) if ours else None,
                           inline_images=ours, step_from=self._step_from(c),
                           fold_before=c[3] if c else None, window=int(info.get("max_len") or 0),
                           levels=info.get("effort_levels"),
                           effort_off=info.get("effort_off"), **kw)

    def peek_body(self, conv, settings: dict, branch: list[dict]) -> dict:
        """The body of the chat's next call as `request_for` builds it (the system block, the
        tools, the effort, the skills field), over the branch as it is: what the engine's cache
        peek renders (SRV-349). The head must be the next request's, or the peek finds nothing."""
        if settings.get("plan_mode") and settings.get("effort") not in ("xhigh",):
            settings = {**settings, "effort": "high"}
        info = self.model_info(settings.get("model"))
        tools = self.tools_for(settings, info, conv.id, listing=False,
                               agents=bool(settings.get("agents")))
        return self._request(settings, info, tools, branch, self._compaction_for(conv.id, branch),
                             self._head(conv, None, settings))

    def request_for(self, run) -> tuple[bytes, dict]:
        """The body of the run's next call, after the context guard (chatd/compaction.py)."""
        conv = run.conv
        if run.step and run.conv.live is not None:
            self.new_step(run)
        with conv.lock:
            settings = self.settings_of(conv.id)
            msgs = self._messages(conv.id)
            path = [msgs[m] for m in self._path(msgs, run.user_message_id)]
            live = conv.live
            if live is not None and live["parts"]:
                # with its id: a compaction inside the run names its steps by it
                path.append({"id": run.message_id, "role": "assistant",
                             "parts": [dict(p) for p in live["parts"]
                                                             if p.get("step", 1) < run.step or
                                                             run.continue_final]})
        if settings.get("plan_mode") and settings.get("effort") not in ("xhigh",):
            settings = {**settings, "effort": "high"}           # a plan turn thinks hard
        user = next((m for m in reversed(path) if m.get("role") == "user"), None)
        if user is not None and not render.text_of(user.get("parts") or []).strip() and \
                not user.get("attachments"):
            # never a turn with an empty user message (the 2026-10-12 empty request)
            raise GuardError("The message is empty: nothing was sent to the model. Write the "
                             "message again, or attach a file.")
        info = self.model_info(settings.get("model"))
        tools = self.tools_for(settings, info, conv.id, agents=run.agents)
        head = self._head(conv, run, settings)
        schemas = [t.openai() for t in tools]
        window = int(info.get("max_len") or 0)
        ours = info.get("backend") == "engine"
        user_i = next((i for i in range(len(path) - 1, -1, -1) if path[i]["role"] == "user"), 0)

        def build(keep_from: int = 0) -> dict:
            return self._request(settings, info, tools, path,
                                 self._compaction_for(conv.id, path), head,
                                 continue_final=run.continue_final, keep_reasoning_from=keep_from,
                                 tail=run.tail)
        if ours and run.step <= 1 and not run.remark:
            gone = self.missing_pins(settings)
            if gone:
                run.remark = (f"The pinned skill{'s' if len(gone) > 1 else ''} "
                              f"{', '.join(gone)} {'are' if len(gone) > 1 else 'is'} no longer "
                              f"in the library; the request went without.")
        body = build()
        est = comp.estimate(body, conv.tok_ratio)
        at = comp.compact_at(window, conv.memory_limit)
        # `compact_next`: the engine refused this call for memory twice, or as over the window
        # (runs.py `_call`)
        force, run.compact_next = run.compact_next, False
        run.compact_cut = False
        if (force or at and est > at) and run.compacted != run.step:
            run.compacted = run.step            # once a step: a long run compacts again later
            if self.compact(conv, run, [m for m in path if m.get("id")], tools=tools):
                run.compact_cut = True
                body = build()
                est = comp.estimate(body, conv.tok_ratio)
        # SRV-639: the estimate holds what the engine counted last at its ratio, and what came
        # since (a tool result) at the same ratio: `margin` is room for that part reading denser
        # (#18: 4,612 tokens over the estimate, with 256 left for it)
        margin = comp.margin(comp.measure(body)[0], conv.tok_counted, conv.tok_ratio)
        if window and est + margin + comp.MIN_ANSWER > window:
            body = build(keep_from=user_i)
            est = comp.estimate(body, conv.tok_ratio)
            margin = comp.margin(comp.measure(body)[0], conv.tok_counted, conv.tok_ratio)
        need, run.request_counted = est + margin, None
        if window and ours and need + comp.MIN_ANSWER > window:
            # at the edge our engine counts the request exactly, with the model's template
            run.request_counted = self.count_tokens(body)
            if run.request_counted:
                need = run.request_counted + comp.MIN_MARGIN
        # refused here: a request the engine counted over, or one whose estimate alone is over;
        # one that only may be over (a backend that cannot count it) goes with max_tokens below
        # its margin, and an over-window 400 then compacts (runs.py `_call`)
        if window and (need if run.request_counted else est) + comp.MIN_ANSWER > window:
            raise GuardError(too_long_words(est, window, path, body, conv.tok_ratio,
                                            counted=run.request_counted))
        cap = int(info.get("max_output") or 0)
        if window and (ours or cap):
            # a backend's model without a known output limit gets none: its window is no limit
            # (DeepSeek refuses a max_tokens above 393,216 in a 1,048,576 window)
            room = window - need - 256
            body["max_tokens"] = max(256, min(room, cap) if cap else room)
        elif cap:
            # VIS-227: a relay that lists no window (llama.cpp, Ollama, a manual model) still gets
            # the operator's output limit
            body["max_tokens"] = cap
        if run.salvage:                             # a worker's report-only call (chatd/tasks.py)
            body["max_tokens"] = min(int(body.get("max_tokens") or SALVAGE_TOKENS), SALVAGE_TOKENS)
        run.tail = None                             # a tail goes with one call only
        # the text's characters and the images' tokens apart: the ratio is learned on text only
        run.request_chars, run.request_images = comp.measure(body)
        run.request_est = est
        run.backend = info.get("backend") or "engine"
        if run.backend == "cloud":
            run.split_est = estimated_split(body, conv.tok_ratio, self.skill_index(settings)
                                            if hasattr(self, "skill_index") else None)
        if run.parent is not None:                  # the chat's Context panel: the worker's split
            self.worker_split(run, body, path)
        self.store.run("UPDATE runs SET request = ?, state = 'running', step = ? WHERE id = ?",
                       (dumps({"messages": len(body["messages"]), "tools": len(schemas),
                               "continue": run.continue_final, "estimate": est}),
                        run.step, run.id))
        return render.encode(body), self.headers_of(conv)

    def count_tokens(self, body: dict) -> int | None:
        """The engine's exact count of a request's prompt (`POST /v1/tokenize`, SRV-347: the
        prompt the same request gets, rendered with the model's template), or None when it cannot
        tell (an older engine, a body its template refuses)."""
        q = {k: body[k] for k in ("model", "messages", "tools", "chat_template_kwargs")
             if body.get(k)}
        st, obj = self.engine.post_json("/v1/tokenize", q, timeout=60.0)
        n = obj.get("tokens") if st == 200 and isinstance(obj, dict) else None
        return n if isinstance(n, int) and n > 0 else None

    # ================================================================== compaction
    def compact(self, conv, run=None, path: list[dict] | None = None, force: bool = False,
                tools: list | None = None) -> bool:
        """Summarize the earlier part of the conversation (chatd/compaction.py): everything before
        the last two user turns, or a long run's steps but its last ones, as deep as it takes for
        the next request to land under TARGET_SHARE of the guard's limit."""
        with conv.lock:
            row = self._row(conv.id)
            settings = self.settings_of(conv.id)
            if path is None:
                msgs = self._messages(conv.id)
                path = [msgs[m] for m in self._path(msgs, row["head_id"])]
        info = self.model_info(settings.get("model"))
        window = int(info.get("max_len") or 0)
        if tools is None:
            tools = self.tools_for(settings, info, conv.id, listing=False,
                                   agents=run.agents if run is not None else
                                   bool(settings.get("agents")))
        head = self._head(conv, run, settings)
        ratio = conv.tok_ratio
        prior = self._compaction_for(conv.id, path)
        done = prior[2] if prior else None             # a long run's compaction: (index, step)
        # the cuts, shallow to deep: before the last two user turns; a long run's steps but its
        # last KEEP_STEPS, ..., but its last one (its user message stays)
        cuts: list[tuple[int, tuple[int, int] | None]] = []
        b = comp.boundary(path)
        if b is not None and not (prior is not None and (done[0] if done else prior[1]) >= b):
            cuts.append((b, None))
        prior_step = done[1] if done and done[0] == len(path) - 1 else 0
        for keep in range(comp.KEEP_STEPS, 0, -1):
            sb = comp.step_boundary(path, prior_step, keep)
            if sb is not None:
                cuts.append((sb[0], sb))
        if not cuts:
            return False
        before = comp.estimate(self._request(settings, info, tools, path, prior, head), ratio)
        # the first cut whose request (with a summary at its longest) lands under the target,
        # else the deepest. Live (2026-10-17) a compaction at 162k tokens kept 138,673, three
        # steps of shader and JS texts, and the engine read 131,505 of them again for a 15 % cut
        target = int(comp.compact_at(window, conv.memory_limit) * comp.TARGET_SHARE)
        now = iso(now_ms())
        for b, sb in cuts:
            after = comp.estimate(self._request(settings, info, tools, path,
                                                self._cut(path, b, sb, None, now), head), ratio)
            if not target or after + comp.SUMMARY_TOKENS <= target:
                break
        start = prior[1] if prior else 0
        system = render.system_block(today=None, tools=False, summary=prior[0] if prior else None)
        msgs: list[dict] = [{"role": "system", "content": system}] if system else []
        fold = prior[3] if prior else None
        end = b if sb is None else sb[0]
        gone = render.all_folded(path[start:end + 1])     # file texts: on disk, not summarized
        for k in range(start, end):
            msgs.extend(render.message(path[k], False, self._blob_of, fold_before=fold,
                                       from_step=done[1] if done and done[0] == k else 0,
                                       folded=gone))
        if sb is not None:
            i, keep_step = sb
            run_msg = dict(path[i], parts=[p for p in path[i].get("parts") or []
                                           if int(p.get("step") or 1) < keep_step])
            msgs.extend(render.message(run_msg, False, self._blob_of, fold_before=fold,
                                       from_step=done[1] if done and done[0] == i else 0,
                                       folded=gone))
        asked = comp.estimate({"messages": msgs}, ratio)
        # a chat the engine refused for memory: the summary's request stays under the guard's
        # share of the refused one, or the engine refuses it too
        cap = comp.compact_at(window, conv.memory_limit) if conv.memory_limit else window
        if cap and asked + comp.SUMMARY_TOKENS + 512 > cap:
            msgs = comp.shrink(msgs, int((cap - comp.SUMMARY_TOKENS - 1024) /
                                         (ratio or comp.DEFAULT_RATIO)))
        msgs.append({"role": "user", "content": comp.INSTRUCTIONS})
        body = {"model": settings.get("model") or info.get("id") or "default", "messages": msgs,
                "stream": False, "max_tokens": comp.SUMMARY_TOKENS,
                "chat_template_kwargs": {"enable_thinking": False}}
        if info.get("backend") == "engine":
            body["tandem"] = {"compaction": True}
        if run is not None:
            run.set_phase("prefill", "Compacting the conversation")
        t0 = time.monotonic()
        st, obj = self.engine.post_json("/v1/chat/completions", body,
                                        self.headers_of(conv), timeout=900.0)
        if st != 200 or not isinstance(obj, dict):
            self.log(f"[compact] {conv.id} failed ({st})")
            return False
        try:
            text = obj["choices"][0]["message"].get("content") or ""
        except (KeyError, IndexError, TypeError, AttributeError):
            return False
        if "</think>" in text:
            text = text.split("</think>", 1)[1]
        text = text.strip()
        if len(text) < 40:
            return False
        asked = int((obj.get("usage") or {}).get("prompt_tokens") or asked)
        created = now_ms()
        # before and after: the chat's request, as the guard counts it, without and with this one
        after = comp.estimate(self._request(settings, info, tools, path,
                                            self._cut(path, b, sb, text, iso(created)), head),
                              ratio)
        bid = path[b]["id"] if sb is None else f"{path[sb[0]]['id']}@{sb[1]}"
        cid = ulid()
        ms = int((time.monotonic() - t0) * 1000)
        with conv.lock:
            self.store.run("INSERT INTO compactions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                           (cid, conv.id, bid, text, before, after, created, ms,
                            run.id if run else None))
            self.emit(conv, {"type": "compaction", "compaction": {
                "id": cid, "at": iso(created), "before_tokens": before,
                "after_tokens": after, "summary": text, "duration_ms": ms,
                "boundary_id": bid}})
        self.log(f"[compact] {conv.id} {before} -> {after} (summary call {asked} tokens) in {ms} ms")
        return True

    @staticmethod
    def _cut(path: list[dict], b: int, sb: tuple[int, int] | None, summary: str | None,
             at: str) -> tuple:
        """A compaction at a cut as `_compaction_for` gives one: before message `b`, or a long
        run's steps before `sb` (its message index, its first step kept)."""
        if sb is None:
            return summary, b, None, at
        u = next((k for k in range(sb[0] - 1, -1, -1) if path[k]["role"] == "user"), 0)
        return summary, u, sb, at

    def compact_now(self, cid: str) -> None:
        from chatd.service import ApiError
        conv = self.conv(cid)
        if conv.run is not None and conv.run.active:
            raise ApiError(409, "run_active", "a run is on; compaction happens before its next "
                                              "model call when it is needed")
        threading.Thread(target=self._compact_bg, args=(conv,), daemon=True).start()

    def _compact_bg(self, conv) -> None:
        if not self.compact(conv, force=True):
            self.emit(conv, {"type": "context.changed", "context": conv.context}, persist=False) \
                if conv.context else None
