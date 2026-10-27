"""The monitor and the per-request supervisor: modes, caps, texts to token ids, the journal.

`Monitor` is the server's: its mode (`--conductor`), its policy, the sessions and the journal.
`Monitor.begin` gives a request its `Supervisor`, or None when the request runs raw. The server's
loop calls the supervisor at the four safe points:

    SP4  sup.at_request(ev)          -> [Action]   PROMPT_TAIL_NOTE, SET_THINK_BUDGET
    SP1  sup.at_block(ev, room)      -> Action?    CLOSE_REASONING, GIVE_UP
    SP2  sup.at_finish(ev, room)     -> Action?    CONTINUE_HIDDEN, REGENERATE_CALL, GIVE_UP
    SP3  sup.check_answer(items)     -> Action?    REGENERATE_CALL, GIVE_UP (repairs are kept)

None means PASS. An action returned is to be executed: the mode, the task's own mode and the caps
are already applied, the texts are token ids (`Action.ids`, `Action.visible`), and the journal
has the event. In shadow every decision is journaled and nothing is returned. The one exception
is E2's claim check (claimcheck.py): its SP2 action comes back with nothing to forward and
`detail["ask"]`, in shadow too, because its decision needs Qwen's answers; the loop asks them in
place and `settle_claim` gives the note to forward, or None to take the end token after all.

SP4 also runs the session-level tasks (S1, S2, S4: sessiontasks.py), which read what a request
carries about the turns before it. The one thing that cannot wait for a request is a client that
has gone: `Monitor.sweep` (every 30 s on the watch's thread, `start_watch`) journals
`client_gone` for a session idle past the stall watchdog's limit after its last turn ended, and
sends NOTIFY through `notifier` only when that task acts and its `notify.enabled` is on.

SP3 runs at the end token, inside SP2: the answer's calls (and its blocks that do not parse) are
all held, every SP3 task checks every one, and then either all are released (with the repairs
`release_call` / `release_block` hand out) or none is: one REGENERATE_CALL for the whole answer,
a hidden turn with the reminders after the dropped end token, and the model writes the set again.
`cuts` records where (in tokens the client was sent) each regeneration began, so the server drops
what it held of the answer before it. An answer that ends without its end token (the token limit,
a stop string) is checked by `check_answer(..., final=True)`: nothing can be written again, so a
check that would regenerate gives up instead.

Caps (policy `caps`): at most `acts_per_request` acts a request and `acts_per_session_hour` a
session, and -- unless `same_action_in_a_row` -- never the same hidden action (kind and text)
twice in a row; beyond that the act becomes GIVE_UP, one visible sentence and finish `stop`.
SP4's actions and the budget close they cause are not capped: they run once a request.

The splices at SP4 work on token ids at the prompt's tail only, so the resident prefix and the
state store stay valid: a note goes into the last message, before its turn end; thinking off
replaces the generation prompt's open block with the closed empty one the template writes for
`enable_thinking: false`. Neither touches the system turn or `reasoning_effort`.
"""

from __future__ import annotations

import dataclasses
import json
import threading
import time

from server.conductor.monitor import claimcheck, notify, tier0
from server.conductor.monitor.journal import Journal
from server.conductor.monitor.observe import RequestEvent
from server.conductor.monitor.policy import (
    CLOSE_REASONING, CONTINUE_HIDDEN, GIVE_UP, PASS, PROMPT_TAIL_NOTE, REGENERATE_CALL,
    REPAIR_CALL, SAFE_POINTS, SET_THINK_BUDGET, Action, Policy, lower,
)
from server.conductor.monitor.observe import ToolCallEvent
from server.conductor.monitor.session import Sessions, args_hash


class Monitor:
    """The server's session monitor."""

    def __init__(self, mode: str, policy: Policy, tokenizer, journal: Journal | None = None):
        self.mode = mode
        self.tok = tokenizer
        self.journal = journal or Journal()
        self.policy = policy
        self.sessions = Sessions(**policy.session)
        self._ids: dict[str, list[int]] = {}
        # NOTIFY's sender (notify.send); a test or the fake engine puts its own here
        self.notifier = notify.send
        self._compacted: tuple[str, float] | None = None
        self._watch: threading.Thread | None = None
        self._stop = threading.Event()

    def reload(self, policy: Policy) -> None:
        """A new policy for the requests that start from now on; running ones keep theirs."""
        self.sessions.max = int(policy.session["max_sessions"])
        self.sessions.window = float(policy.session["join_window_s"])
        self.sessions.lineage = int(policy.session["lineage_tokens"])
        self.policy = policy
        self._ids.clear()

    def mode_for(self, header, n: int = 1) -> str:
        """The request's mode: `X-Tandem` lowers the server's; `n > 1` never acts."""
        m = lower(self.mode, header)
        return lower(m, "shadow") if n > 1 else m

    def ids(self, text: str) -> list[int]:
        """`text` as token ids, the way the served tokenizer reads it; cached per policy."""
        got = self._ids.get(text)
        if got is None:
            got = [int(t) for t in self.tok(text, add_special_tokens=False).input_ids]
            if len(self._ids) < 512:
                self._ids[text] = got
        return list(got)

    def begin(self, mode: str, req: RequestEvent, prompt_ids, conv_id: str | None,
              now: float | None = None, policy: Policy | None = None,
              session_key: str | None = None) -> "Supervisor | None":
        """The request's supervisor, or None when it runs raw. `policy`: the request's own (the
        conductor's profile), else the server's. `session_key`: the conductor router's key for
        the request, so one session is one session in the journal for the router and the
        monitor alike; without one the monitor keys it itself."""
        if mode == "raw":
            return None
        now = time.time() if now is None else now
        policy = policy or self.policy
        if session_key:
            key, how = session_key, "conductor"
        else:
            key, how = self.sessions.key_for(conv_id, prompt_ids, req.tools_offered,
                                             req.messages, now)
        session = self.sessions.touch(key, now)
        session.record_results(req.messages, now)
        from server.conductor import signals
        shape = signals.shape({"messages": req.messages, "tools": req.tools,
                               "tool_choice": req.tool_choice},
                              client_kind=getattr(req, "client", None) or None)
        self._compaction(session, shape, now)
        # the checker result this request carries is the outcome label of the session's
        # previous request (SRV-115 next_checker), written once whoever writes it first
        if session.last_request and shape.results:
            self.journal.write({"request_id": session.last_request, "decision": "LABEL",
                                "session": key, "next_checker": shape.results})
        session.last_request = req.request_id
        claimcheck.label(session, policy, self.journal, now)
        return Supervisor(self, policy, mode, req, session, how, now=now, shape=shape)

    def _compaction(self, session, shape, now: float) -> None:
        """Mark opencode's compaction request on its session (S2 reads `last_compaction`). A
        client without a conversation id starts a new key after it (its prompt shares nothing
        with the old one): the first such session within the join window takes the mark."""
        if shape.compaction:
            session.last_compaction = now
            self._compacted = (session.key, now)
            return
        if self._compacted is None or session.requests != 1:
            return
        key, at = self._compacted
        if key != session.key and now - at <= self.sessions.window and \
                not session.key.startswith(("conv:", "c:")) and not key.startswith(("conv:",
                                                                                    "c:")):
            session.last_compaction = at
            self._compacted = None

    # ------------------------------------------------------------ the watch (S4, out of band)
    def turn_ended(self, sup: "Supervisor", now: float | None = None) -> None:
        """The request ended (every way out): the watch counts the client's idle time from
        here."""
        sup.session.ended_at = time.time() if now is None else now

    def sweep(self, now: float | None = None) -> list[dict]:
        """Every session whose client has gone: no request for the stall watchdog's `minutes`
        after its last turn ended. One `client_gone` event a turn end; NOTIFY (sent through
        `notifier`) only when its task acts and `notify.enabled`. The events written."""
        now = time.time() if now is None else now
        out = []
        for s in self.sessions.states():
            w, idle = s.watch, s.idle(now)
            if not w or idle is None or idle < w["minutes"] * 60.0 or s.gone_at == s.ended_at:
                continue
            s.gone_at = s.ended_at
            cfg, mode = w["notify"], w["mode"]
            minutes = round(idle / 60.0, 1)
            fields = {"session": s.key, "client": w["client"] or "-",
                      "profile": w["profile"] or "-", "minutes": f"{minutes:g}",
                      "requests": str(s.requests)}
            detail = {"idle_minutes": minutes, "requests": s.requests,
                      "notify": "off" if not cfg["enabled"] else None}
            decision = "NOTIFY" if cfg["enabled"] else PASS
            if cfg["enabled"] and mode == "act":
                detail["notify"] = self.notifier(cfg, _fill(cfg["title"], fields),
                                                 _fill(cfg["message"], fields))
            elif cfg["enabled"]:
                detail["notify"] = "shadow: not sent"
            out.append(self.journal.write({
                "request_id": s.last_request, "session": s.key, "sp": "watch",
                "task": w["task"], "rule": "stall_watchdog", "tier": 0, "mode": mode,
                "decision": decision, "reason": f"client_gone({minutes:g} min)",
                "detail": detail, "policy_version": w["policy_version"],
                "client": w["client"], "profile": w["profile"]}))
        return out

    def start_watch(self, every_s: float = 30.0) -> None:
        """The watch's thread: `sweep` every `every_s` seconds. A daemon; `stop_watch` ends it."""
        if self._watch is not None:
            return

        def run():
            while not self._stop.wait(every_s):
                try:
                    self.sweep()
                except Exception as exc:                           # noqa: BLE001
                    print(f"[conductor] watch failed: {type(exc).__name__}: {exc}", flush=True)

        self._stop.clear()
        self._watch = threading.Thread(target=run, name="conductor-watch", daemon=True)
        self._watch.start()

    def stop_watch(self) -> None:
        self._stop.set()
        if self._watch is not None:
            self._watch.join(5.0)
        self._watch = None

    def watching(self, now: float | None = None) -> list[dict]:
        """What the session tasks hold per session, for the dashboard's Monitor screen: the
        minutes since progress, the gates and their readings, a compaction being watched, the
        client's idle time and whether a push would go out."""
        now = time.time() if now is None else now
        out = []
        for s in reversed(self.sessions.states()):
            if not s.tasks and not s.watch:
                continue
            row = {"session": s.key, "requests": s.requests, "last_seen": s.last_seen,
                   "ended_at": s.ended_at,
                   "running": s.ended_at is None or s.ended_at < s.last_seen,
                   "note_at": s.note_at,
                   "idle_minutes": None if s.idle(now) is None else round(s.idle(now) / 60, 1),
                   "client_gone": s.gone_at is not None and s.gone_at == s.ended_at,
                   "watch": None, "gates": [], "compaction": None, "progress": None}
            if s.watch:
                row["watch"] = {"task": s.watch["task"], "minutes": s.watch["minutes"],
                                "mode": s.watch["mode"], "profile": s.watch["profile"],
                                "client": s.watch["client"],
                                "notify": bool(s.watch["notify"]["enabled"])}
            for tid, st in s.tasks.items():
                for g in (st.get("gates") or {}).values():
                    if g["runs"]:
                        last = g["runs"][-1][0]
                        row["gates"].append({"task": tid, "checker": g["checker"],
                                             "gate": g["gate"], "op": last["op"],
                                             "limit": last["limit"],
                                             "values": [r[0]["value"] for r in g["runs"]],
                                             "noted": bool(g["noted"])})
                if "progress_at" in st:
                    row["progress"] = {"task": tid, "last": st.get("progress"),
                                       "minutes": round((now - st["progress_at"]) / 60.0, 1)}
                if "compaction" in st:
                    row["compaction"] = {"task": tid, "count": st.get("count", 0),
                                         "watching": bool(st.get("watching")),
                                         "steps": len(st.get("steps") or ())}
            out.append(row)
        return out


class Supervisor:
    """One request: its policy snapshot, mode, session and counters."""

    def __init__(self, monitor: Monitor, policy: Policy, mode: str, req: RequestEvent,
                 session, how: str = "", now: float | None = None, shape=None):
        self.monitor, self.policy, self.mode, self.req = monitor, policy, mode, req
        self.session, self.session_how = session, how
        self.now = time.time() if now is None else now     # the request's start (SP4's clock)
        self.shape = shape                  # signals.shape of the request's messages
        self.journal = monitor.journal
        self.acts = 0                       # executed acts that count against the caps
        self.task_acts: dict[str, int] = {}
        self.hidden_tokens = 0
        self.acted = False                  # anything executed that changes the output
        self.gave_up: str | None = None     # the GIVE_UP sentence, once given up
        # SRV-199: the token ids of the calls the output ended with inside its reasoning (SP2's
        # `FinishEvent.calls`), set by the server before `at_finish`; E1's case C forwards them
        self.lift_ids: list[int] | None = None
        # SP2's events held while the block's tokens before the end token still go out (server)
        self.held_events: list | None = None
        self.think_budget: int | None = None
        self.resume_ms: list[float] = []    # SP2: end token to the continued answer's first token
        self.memo: dict = {}                # the rules' own per-request bookkeeping
        self.final_answer: str | None = None  # A1's give-up: the answer instead of a generation
        self.cuts: list[int] = []           # SP3: tokens sent before each regeneration began
        self.checked = False                # SP3 ran over the answer that is ending now
        self.forwarding: Action | None = None    # a hidden forward is running (the live view)
        self.forwarded_at: float | None = None   # ... and when the last one ended
        self.last: dict | None = None       # the newest decision and action (the live view)
        self.last_action: dict | None = None
        self.shadows = 0                    # shadow decisions that were not a pass
        self._repairs: dict = {}
        # SP3: the items of the answer being checked (C5 reads the whole set at one call)
        self.answer_items: list = []
        self._passes: dict = {}             # (task, tier, mode) -> checks that found nothing
        self._budget: Action | None = None
        self._last: tuple | None = None
        self._shadowed: set = set()
        self._at = {sp: policy.at(sp) for sp in SAFE_POINTS}
        self._tools: tuple | None = None
        # E2: the loop sets `can_ask` when it can ask a pending question in place (no logprobs
        # asked); `_pending` is the claim check waiting for its answers (`settle_claim`)
        self.can_ask = False
        self._pending: tuple | None = None
        # J1 step 6, E-b: (mode, the turn's evidence digest) the claim check's note carries in
        # act; shadow journals its size. None: the note as the policy writes it
        self.evidence: tuple | None = None
        self.events: list[dict] = []
        # every decision that acts, and each check of an answer's held calls, as it happens: the
        # user-visible states' hook (server/conductor/states.py); None when nobody listens
        self.on_event = None

    def adopt(self, times: list[float]) -> None:
        """Acts the router took for this request before SP4 (J1: an outcome's note): they count
        against this request's and the session's caps like the monitor's own."""
        for t in times:
            self.acts += 1
            self.session.record_act(t)

    # ------------------------------------------------------------ what the loop asks
    @property
    def holds_calls(self) -> bool:
        """Hold tool calls until SP3 has seen them: only when a tool-call task can act, and never
        for a constrained request (the grammar writes its calls; SRV-35 stays as it is)."""
        return (self.mode == "on" and not self.req.constrained
                and any(t.mode == "act" and self._may_act(t) for t in self._at["SP3"]))

    def _may_act(self, task) -> bool:
        """A tool-call task that can act on this request. C3 (`repeated_call`) can only fire
        when the turn's history already ends with the repeated call (SRV-473): on every other
        request it cannot act, so it holds nothing and the calls stream live as before. C5
        (`sequential_mutations`, SRV-623) only when an offered tool mutates."""
        if task.rule not in ("repeated_call", "sequential_mutations"):
            return True
        got = self.memo.get(("may_act", task.id))
        if got is None:
            from server.conductor.monitor.toolcheck import can_repeat, can_sequence
            can = can_repeat if task.rule == "repeated_call" else can_sequence
            got = self.memo[("may_act", task.id)] = bool(can(self, task))
        return got

    def _tool_meta(self) -> tuple:
        if self._tools is None:
            from server.toolcall import schema_types
            fns = [t.get("function") for t in self.req.tools or () if isinstance(t, dict)]
            fns = [f for f in fns if isinstance(f, dict) and isinstance(f.get("name"), str)]
            self._tools = ([f["name"] for f in fns], schema_types(self.req.tools) or None,
                           {f["name"]: f.get("parameters") for f in fns
                            if isinstance(f.get("parameters"), dict)})
        return self._tools

    @property
    def tool_names(self) -> list[str]:
        return self._tool_meta()[0]

    @property
    def tool_types(self) -> dict | None:
        """Per function, the parameters whose schema is not a string (`toolcall.schema_types`)."""
        return self._tool_meta()[1]

    @property
    def arms(self) -> dict | None:
        """The ThinkBudget detector arms of the first reasoning-loop task that is not off."""
        for t in self._at["SP1"]:
            if t.rule == "reasoning_loop":
                return t["arms"]
        return None

    @property
    def watches_blocks(self) -> bool:
        return bool(self._at["SP1"]) or self.think_budget is not None

    # ------------------------------------------------------------ the safe points
    def at_request(self, ev: RequestEvent) -> list[Action]:
        """SP4. The actions to apply to the prompt before it is prefilled."""
        out = []
        note = self._argloop_note(ev)
        if note is None and not self.final_answer:
            note = self._escape_note(ev)
        if note is not None:
            out.append(note)
        for task in self._at["SP4"]:
            act = tier0.RULES[task.rule].fn(ev, task, self)
            if act is None:
                continue
            act.task = act.task or task.id
            eff = self._effective(task)
            if act.kind == PASS or eff != "act":
                self._write("SP4", task, act, eff)
                continue
            if act.kind == PROMPT_TAIL_NOTE:
                act.ids = self.monitor.ids(self.policy.templates["tail_note"]
                                           .replace("{text}", act.text))
            elif act.kind == SET_THINK_BUDGET and act.budget == 0:
                act.ids = self.monitor.ids(self.policy.templates["think_off"])
            elif act.kind == SET_THINK_BUDGET and (act.budget or 0) > 0:
                if not self._closing(act):          # the budget's close needs its closer
                    self._write("SP4", task, act, "act", cap="no_closer", decision=PASS)
                    continue
                self.think_budget, self._budget = act.budget, act
            else:
                continue
            out.append(act)
        return out

    def argloop_stopped(self, verdict, name: str | None) -> None:
        """The tool-argument loop guard stopped this request's open call (server/argloop.py):
        journaled as A1's GIVE_UP with the sentence the client got, and the session keeps the
        verdict for its next request's hidden note (`_argloop_note`)."""
        d = dict(verdict.as_dict(), name=name)
        act = Action(GIVE_UP, reason=f"tool_arg_loop({d['rule']})", text=verdict.text,
                     detail=d, task="A1")
        self.acted = True
        self._write("SP1", None, act, "act", at=verdict.tokens)
        self.session.tasks["A1"] = {"pending": d, "at": self.now}

    def _argloop_note(self, ev: RequestEvent) -> Action | None:
        """SP4: the session's last call was stopped by the loop guard, or the history the client
        sent again ends in calls it stopped (`argloop.trailing_stops`: the markers the handler's
        `collapse_history` left) -> the hidden note at the end of this prompt, once; it names the
        tool and the piece, and after an identical stop says so. After `GIVE_UP_IDENTICAL`
        identical stops in a row, or `GIVE_UP_ANY` of one tool (the Open WebUI retry
        storm of 2026-10-06: five byte-identical attempts), A1's GIVE_UP instead: no generation,
        the client gets `argloop.final_text` and its tool loop ends. Not on a request without
        tools (a title, a compaction); a new user message starts the streak over."""
        st = self.session.tasks.get("A1")
        if not ev.tools_offered or ev.n != 1:
            return None
        from server import argloop
        stops = argloop.trailing_stops(ev.messages)
        pending = st.pop("pending", None) if st else None
        if pending is None and not stops:
            return None
        if argloop.give_up_due(stops):
            text = argloop.final_text(stops, thinking_off=self.memo.get("client_thinking_off",
                                                                         False))
            d = {"stops": len(stops), "identical": argloop.identical(stops),
                 "name": stops[-1].get("name"), "period": stops[-1].get("period")}
            act = Action(GIVE_UP, reason=f"tool_arg_loop_retries({len(stops)})", text=text,
                         detail=d, task="A1")
            if self.mode != "on":
                self._write("SP4", None, act, "shadow")
                return None
            self.acted = True
            self.final_answer = text
            self._write("SP4", None, act, "act")
            return None
        d = dict(pending or {}, stops=len(stops), identical=argloop.identical(stops))
        act = Action(PROMPT_TAIL_NOTE, reason=f"tool_arg_loop_note({d.get('rule', 'history')})",
                     text=argloop.note_text(pending or {}, stops), detail=d, task="A1")
        if self.mode != "on":
            self._write("SP4", None, act, "shadow")
            return None
        act.ids = self.monitor.ids(self.policy.templates["tail_note"].replace("{text}", act.text))
        self.memo["session_note"] = "A1"
        return act

    # the loop escape's note when the policy's C3 has no `escape_texts` (a stored monitor
    # document from before SRV-475)
    ESCAPE_TEXTS = ("You ran `{name}` {count} times in a row and got the same result each time. "
                    "Running it again will not change that. Try another approach: read what the "
                    "result says, change the arguments or the tool, or fix the cause first.",)

    def _escape_note(self, ev: RequestEvent) -> Action | None:
        """SP4: the loop escape (SRV-475, server/conductor/loopescape.py) holds for this request
        -> the hidden note at the end of the prompt with the sampled turn: which call repeated,
        and to take another path. Set by the handler in `memo["loop_escape"]`; journaled as C3's
        PROMPT_TAIL_NOTE with the escape's detail."""
        esc = self.memo.get("loop_escape")
        if not esc or not ev.tools_offered or ev.n != 1:
            return None
        task = self.policy.tasks.get("C3")
        texts = (task.get("escape_texts") if task is not None else None) or self.ESCAPE_TEXTS
        text = (texts[min(int(esc.get("nth", 0)), len(texts) - 1)]
                .replace("{name}", str(esc.get("name") or "the tool"))
                .replace("{count}", str(esc.get("count") or 2)))
        act = Action(PROMPT_TAIL_NOTE, reason=f"loop_escape({esc.get('why')})", text=text,
                     detail=dict(esc), task="C3")
        if self.mode != "on":
            self._write("SP4", None, act, "shadow")
            return None
        act.ids = self.monitor.ids(self.policy.templates["tail_note"].replace("{text}", act.text))
        self.memo["session_note"] = "C3"
        return act

    def route_budget(self, budget: int, task: str = "route") -> bool:
        """A reasoning cap for this turn, closed at SP1 with the hidden close as a
        SET_THINK_BUDGET is, under the vendor's phrase (`texts.vendor_close`): the client sees the
        closer and what follows it. `task` says whose cap it is in the journal: `route` the
        router's (its effort level's budget), `client` the client's own (server/thinking.py),
        `reserve` one the answer reserve lowered. A smaller cap an SP4 task set stays. False when
        this supervisor cannot act (shadow) or the policy has no usable close text; the caller
        then leaves the cap to ThinkBudget's streamed phrase."""
        text = self.policy.texts.get("vendor_close")
        if self.mode != "on" or budget <= 0 or not text:
            return False
        act = Action(SET_THINK_BUDGET, reason=f"{task}({budget})", budget=budget, text=text,
                     task=task)
        if not self._closing(act):
            return False
        if self.think_budget is None or budget < self.think_budget:
            self.think_budget, self._budget = budget, act
        return True

    def apply_request(self, prompt: list[int], in_think: bool, actions: list[Action]
                      ) -> tuple[list[int], bool]:
        """SP4's actions applied to the prompt's tail: `(prompt ids, still inside <think>)`.
        Each action is journaled here, applied or not (`cap=no_shape` when the prompt does not
        have the shape it needs)."""
        t = self.policy.templates
        for act in actions:
            hidden, cap = 0, None
            if act.kind == PROMPT_TAIL_NOTE:
                got = tail_note(prompt, act.ids, self.monitor.ids(t["turn_start"]),
                                self.monitor.ids(t["turn_end"]))
                hidden = len(act.ids)
            elif act.budget == 0:
                got = think_off(prompt, self.monitor.ids(t["think_open"]), act.ids)
                hidden = len(act.ids) - len(self.monitor.ids(t["think_open"]))
            else:
                got = prompt                     # a budget closes at SP1; nothing to splice
            if got is None:
                cap, hidden = "no_shape", 0
            else:
                prompt = got
                self.acted = True
                self.hidden_tokens += max(0, hidden)
                if act.kind == SET_THINK_BUDGET and act.budget == 0:
                    in_think = False
            self._write("SP4", None, act, "act", hidden=max(0, hidden), cap=cap,
                        decision=PASS if cap else None)
        return prompt, in_think

    def at_block(self, ev, room: int) -> Action | None:
        """SP1, after a block inside the reasoning. A SET_THINK_BUDGET cap closes first."""
        if (self._budget is not None and ev.inside
                and ev.reasoning_tokens >= self.think_budget):
            act, self._budget = self._budget, None
            close = Action(CLOSE_REASONING, reason=f"budget({self.think_budget})", text=act.text,
                           task=act.task, detail={"at": ev.reasoning_tokens})
            if self._closing(close) and self._room(close, room, "SP1", None):
                self.hidden_tokens += close.hidden
                self._write("SP1", None, close, "act", hidden=close.hidden,
                            at=ev.reasoning_tokens)
                return close
        return self._decide("SP1", ev, room)

    def at_finish(self, ev, room: int) -> Action | None:
        """SP2, an end token before it reaches the client. An answer with tool calls is SP3's
        first: its calls are checked here, where the model can still write them again."""
        if (not ev.inside and ev.answer and self._at["SP3"] and self.req.tools_offered
                and self.req.n == 1):
            from server.toolcall import answer_items
            items = answer_items(ev.answer, self.tool_names or None, eos=True,
                                 types=self.tool_types)
            if items:
                act = self.check_answer(items, room=room, pos=ev.pos)
                if act is not None:
                    return act
        return self._decide("SP2", ev, room)

    def check_answer(self, items: list[dict], final: bool = False, room=None,
                     pos: int | None = None) -> Action | None:
        """SP3: every call and every block that does not parse of the answer that ended, before
        any is released (`toolcall.answer_items` reads them). Repairs are kept for the release.
        Findings that act become one REGENERATE_CALL for the whole answer (texts joined), or
        GIVE_UP when the caps say so, or when `final` (no end token: nothing can be written
        again). None: release them all as they are."""
        from server.toolcall import diagnose
        self.checked = True
        tasks = self._at["SP3"]
        if not tasks or not items:
            return None
        if self.holds_calls:
            self._tell({"decision": "CHECK", "task": "SP3",
                        "detail": {"calls": len(items), **_call_facts(items)}})
        schemas = self._tool_meta()[2]
        findings: list[tuple] = []
        wrote = len(self.events)
        queue = self.answer_items = list(items)
        i = 0
        while i < len(queue):
            it = queue[i]
            if "function" in it:
                fn = it["function"]
                ev = ToolCallEvent(fn["name"], fn["arguments"], i, block=it.get("_block", ""),
                                   schema=schemas.get(fn["name"]), final=final)
            else:
                blk = it["bad"] if "bad" in it else it.get("cut", "")
                ev = ToolCallEvent("", "", i, block=blk, error=diagnose(blk), cut="cut" in it,
                                   stopped=bool(it.get("stop")), final=final)
            i += 1
            for task in tasks:
                act = tier0.RULES[task.rule].fn(ev, task, self)
                eff = self._effective(task)
                if eff == "act" and self.req.constrained:
                    eff = "shadow"
                if act is None:
                    self._tally(task, eff, "PASS")
                    continue
                act.task = task.id
                if act.detail.get("mode") in ("off", "shadow") and eff == "act":
                    eff = act.detail["mode"]
                if eff == "off":
                    continue
                if act.kind == PASS:
                    self._write("SP3", task, act, eff, call=ev)
                    continue
                if eff != "act":
                    self._write("SP3", task, act, "shadow", call=ev)
                    continue
                if act.kind == REPAIR_CALL:
                    if ev.error is not None:
                        self._repairs[("bad", ev.block)] = act.detail["calls"]
                        queue.extend(act.detail["calls"])   # its calls are checked as well
                    else:
                        self._repairs[("call", ev.name, ev.arguments)] = act.detail["arguments"]
                    self.acted = True
                    self._write("SP3", task, act, "act", call=ev)
                    if ev.error is not None:
                        break
                    # the tasks after this one check the call as it will go out
                    ev = dataclasses.replace(ev, arguments=act.detail["arguments"], block="")
                    continue
                findings.append((task, act, ev))
        if not findings:
            if len(self.events) == wrote and self.policy.journal["tool_passes"]:
                self._write("SP3", None, Action(PASS, reason=f"checked({len(items)})",
                                                detail={"calls": sum("function" in x
                                                                     for x in items),
                                                        "blocks": len(items)}),
                            "act" if self.holds_calls else "shadow", quiet=True)
            return None
        task, first, _ = findings[0]
        found = [{"task": t.id, "rule": t.rule, "decision": a.kind, "reason": a.reason,
                  "call": e.name or None, "detail": a.detail} for t, a, e in findings]
        texts = list(dict.fromkeys(a.text for _, a, _ in findings if a.text))
        # C5's findings are about order, not about a call that cannot run: past a cap they go
        # out as they are (journaled), never a give-up
        release = all(a.detail.get("release") for _, a, _ in findings)
        give_up = final or any(a.kind == GIVE_UP for _, a, _ in findings)
        if not give_up:
            act = Action(REGENERATE_CALL, reason="; ".join(a.reason for _, a, _ in findings),
                         text="\n\n".join(texts), task=task.id,
                         detail={"findings": found, "calls": len(items)})
            cap = self._cap(act)
            if cap is None:
                act.ids = self.monitor.ids(self.policy.templates["regen_turn"]
                                           .replace("{text}", act.text))
                act.visible = []
                if not self._room(act, room, "SP3", task):
                    return None
                self._count(act)
                for t, _, _ in findings[1:]:
                    if t.id != task.id:
                        self.task_acts[t.id] = self.task_acts.get(t.id, 0) + 1
                self.hidden_tokens += act.hidden
                self._write("SP3", task, act, "act", hidden=act.hidden, pos=pos)
                self.checked = False                 # the next answer is checked again
                return act
            if release:
                self._write("SP3", task, Action(PASS, reason=f"released({cap})", task=task.id,
                                                detail={"findings": found, "cap": cap}),
                            "act", cap=cap)
                return None
            give_up = cap
        act = Action(GIVE_UP, reason=first.reason, task=task.id,
                     detail={"findings": found,
                             "cap": give_up if isinstance(give_up, str) else
                             ("ended" if final and first.kind != GIVE_UP else None)})
        act.text = self.policy.texts["give_up"]
        self.gave_up = act.text
        self.acted = True
        self._write("SP3", task, act, "act", cap=act.detail.get("cap"))
        return act

    def at_tool_call(self, ev) -> Action | None:
        """SP3 for one call alone: `check_answer` of an answer that holds only `ev`."""
        item = ({"function": {"name": ev.name, "arguments": ev.arguments}} if ev.error is None
                else {"cut": ev.block} if ev.cut else {"bad": ev.block, "closed": True})
        return self.check_answer([item], final=ev.final)

    def release_call(self, call: dict) -> dict:
        """A held call as it goes out: with SP3's repaired arguments, else unchanged."""
        fn = call.get("function") or {}
        got = self._repairs.get(("call", fn.get("name"), fn.get("arguments")))
        call = {k: v for k, v in call.items() if not k.startswith("_")}
        return call if got is None else dict(call, function=dict(fn, arguments=got))

    def release_block(self, inner: str) -> list[dict] | None:
        """The calls a held block that does not parse goes out as (SP3 repaired it), or None:
        it goes out as the content it is."""
        return self._repairs.get(("bad", inner))

    # ------------------------------------------------------------ after the request
    def finish(self, outcome: str, **detail) -> None:
        """The request's outcome, journaled once when anything was decided, and the checks that
        found nothing, counted (`COUNTS`: the metrics' passes; never a journal row)."""
        if self._passes:
            self.journal.write(dict(self._base("end"), decision="COUNTS",
                                    counts=[list(k) + [n] for k, n in self._passes.items()]))
            self._passes = {}
        if not self.events:
            return
        self.journal.write(dict(self._base("end"), decision="OUTCOME", outcome=outcome,
                                acts=self.acts, hidden_tokens=self.hidden_tokens,
                                detail=detail))

    def mode_of(self, task) -> str:
        """The task's mode for this request: off, shadow or act."""
        return self._effective(task)

    # ------------------------------------------------------------ internals
    def _effective(self, task) -> str:
        """The task's mode under the request's: shadow lowers every task to shadow."""
        if task.mode == "off":
            return "off"
        return "shadow" if self.mode == "shadow" else task.mode

    def _tally(self, task, mode: str, decision: str) -> None:
        k = (task.id, 0, mode, decision)
        self._passes[k] = self._passes.get(k, 0) + 1

    def _decide(self, sp: str, ev, room) -> Action | None:
        for task in self._at[sp]:
            act = tier0.RULES[task.rule].fn(ev, task, self)
            if act is None:
                self._tally(task, self._effective(task), "PASS")
                continue
            act.task = task.id
            at = getattr(ev, "reasoning_tokens", None)
            if act.detail.get("ask"):
                if self._ask(sp, task, act, room, getattr(ev, "pos", None)):
                    return act
                continue
            if act.kind == PASS:
                self._write(sp, task, act, self._effective(task), at=at)
                continue
            if self._effective(task) != "act":
                # what would have been done, once a request: an arm that fired stays fired, and
                # the same decision at every later block is not a new one
                if (task.id, act.kind) not in self._shadowed:
                    self._shadowed.add((task.id, act.kind))
                    self._write(sp, task, act, "shadow", at=at)
                continue
            if act.kind != GIVE_UP:
                cap = self._cap(act)
                if cap is None:
                    if not self._resolve(act) or not self._room(act, room, sp, task):
                        continue
                    self._count(act)
                    self.hidden_tokens += act.hidden
                    self._write(sp, task, act, "act", hidden=act.hidden, at=at,
                                pos=getattr(ev, "pos", None))
                    return act
                act = Action(GIVE_UP, reason=act.reason, task=task.id,
                             detail=dict(act.detail, cap=cap))
            act.text = self.policy.texts["give_up"]
            self.gave_up = act.text
            self.acted = True
            self._write(sp, task, act, "act", at=at, cap=act.detail.get("cap"))
            return act
        return None

    def _ask(self, sp: str, task, act: Action, room, pos) -> bool:
        """A decision that needs Qwen's answer first (E2, the claim check): True hands `act` back
        to the loop with nothing to forward, so the loop asks the questions where the KV holds
        the whole context and calls `settle_claim`, in shadow as well (the answers are what
        shadow journals). False, journaled as a PASS, when the loop cannot ask (`can_ask`: a
        request that asked for logprobs) or the caps leave no act."""
        eff = self._effective(task)
        facts = {k: v for k, v in act.detail.items() if k != "ask"}
        cap = self._cap(act) if eff == "act" else None
        if not self.can_ask or cap is not None:
            self._write(sp, task, Action(PASS, reason="claim_check(not_asked)" if cap is None
                                         else "claim_check(cap)", detail=facts, task=task.id),
                        eff, cap=cap, tier=1)
            return False
        self._pending = (sp, task, facts, eff, room, pos)
        act.ids, act.visible = [], []
        return True

    def settle_claim(self, asker) -> Action | None:
        """The pending claim check (E2): its questions asked through `asker(ids, letter ids) ->
        the letters' logits`, which the loop runs where the KV holds the whole context
        (server/app.py `_force_close`), and its decision: the hidden note to forward, or None to
        take the end token as it was drawn (no claim, supported, below a threshold, shadow, the
        caps, no room, a failure: fail-open). Every way out is journaled."""
        sp, task, facts, eff, room, pos = self._pending
        self._pending = None
        from server.conductor import incontext
        tmpl, tok = self.policy.templates["hidden_turn"], self.monitor.tok
        t0 = time.perf_counter()

        def ids_of(text: str) -> list[int]:
            # not through `Monitor.ids`: a question names this session's tools, so it is not cached
            return [int(t) for t in tok(tmpl.replace("{text}", text),
                                        add_special_tokens=False).input_ids]

        try:
            got = claimcheck.ask(task, facts, asker, ids_of,
                                 lambda letters: incontext.letter_ids(tok, letters))
        except Exception as exc:                                   # noqa: BLE001
            why = getattr(exc, "why", None) or type(exc).__name__
            self._write(sp, task, Action(PASS, reason=f"claim_check(error:{why})",
                                         detail=dict(facts, ms=_ms(t0)), task=task.id),
                        eff, tier=1, pos=pos)
            return None
        detail = dict(facts, claim=got["claim"], evidence=got["evidence"], ms=_ms(t0))
        ans = got["answer"]
        if ans is None:
            self._write(sp, task, Action(PASS, reason=f"claim_check({got['why']})", detail=detail,
                                         task=task.id), eff, tier=1, pos=pos)
            return None
        text = task["notes"][ans]
        if self.evidence is not None:
            # E-b: the note carries the turn's evidence (the runtime's snapshot digest)
            detail["digest"] = {"mode": self.evidence[0], "chars": len(self.evidence[1])}
            if self.evidence[0] == "act":
                text = text + "\n\n" + self.evidence[1]
        act = Action(CONTINUE_HIDDEN, reason=f"claim_check({ans})", text=text,
                     detail=dict(detail, answer=ans), task=task.id)
        cap = self._cap(act) if eff == "act" else None
        if cap is not None:
            self._write(sp, task, act, "act", cap=cap, decision=PASS, tier=1, pos=pos)
            return None
        if eff != "act":
            claimcheck.opened(self, task, self.req.request_id, time.time(), eff, ans)
            self._write(sp, task, act, "shadow", tier=1, pos=pos)
            return None
        act.ids, act.visible = ids_of(act.text), []
        if not self._room(act, room, sp, task):
            return None
        claimcheck.opened(self, task, self.req.request_id, time.time(), eff, ans)
        self._count(act)
        self.hidden_tokens += act.hidden
        self._write(sp, task, act, "act", hidden=act.hidden, tier=1, pos=pos)
        return act

    def _cap(self, act: Action) -> str | None:
        caps = self.policy.caps
        if self.acts >= caps["acts_per_request"]:
            return "per_request"
        if self.session.acts_last_hour(time.time()) >= caps["acts_per_session_hour"]:
            return "per_session_hour"
        if not caps["same_action_in_a_row"] and self._last == (act.kind, act.text):
            return "same_in_a_row"
        return None

    def _count(self, act: Action) -> None:
        self.acts += 1
        self.task_acts[act.task] = self.task_acts.get(act.task, 0) + 1
        self.session.record_act(time.time())
        self._last = (act.kind, act.text)
        self.acted = True

    def _closing(self, act: Action) -> bool:
        """A hidden close: the phrase up to the closer is hidden, the closer on is visible."""
        close = self.policy.templates["think_close"]
        i = act.text.find(close)
        if i < 0:
            return False
        head = self.monitor.ids(act.text[:i]) if i else []
        act.visible = self.monitor.ids(act.text[i:])
        act.ids = head + act.visible
        return True

    def _resolve(self, act: Action) -> bool:
        """The token ids of an SP1/SP2 action; False when its text cannot be used."""
        if act.kind == CONTINUE_HIDDEN and act.detail.get("case") == "C":
            # SRV-199's lift: the model's own call tokens, visible, after the closer when the
            # reasoning is still open (the newline before it hidden)
            lift = [int(t) for t in (self.lift_ids or ())]
            if not lift:
                return False
            if act.text:
                if not self._closing(act):
                    return False
            else:
                act.ids, act.visible = [], []
            act.ids, act.visible = act.ids + lift, act.visible + lift
            return True
        if act.kind == CONTINUE_HIDDEN and act.detail.get("case") == "B":
            act.ids = self.monitor.ids(self.policy.templates["hidden_turn"]
                                       .replace("{text}", act.text))
            act.visible = []
            return True
        if act.kind in (CLOSE_REASONING, CONTINUE_HIDDEN):
            return self._closing(act)
        return True

    def _room(self, act: Action, room, sp: str, task) -> bool:
        """The hidden tokens and one more fit in the context; else PASS, journaled."""
        if room is None or len(act.ids) + 2 <= room:
            return True
        self._write(sp, task, act, "act", cap="no_room", decision=PASS)
        return False

    def _base(self, sp: str) -> dict:
        return {"request_id": self.req.request_id, "session": self.session.key, "sp": sp,
                "policy_version": self.policy.version}

    def _write(self, sp: str, task, act: Action, mode: str, hidden: int = 0, at=None,
               cap=None, decision=None, call=None, pos=None, quiet: bool = False,
               tier: int = 0) -> None:
        detail = act.detail
        if call is not None:
            detail = dict(detail, call={"name": call.name or None, "index": call.index,
                                        "args_hash": args_hash(call.arguments) if call.name
                                        else None, "error": call.error})
        ev = dict(self._base(sp), task=act.task or (task.id if task else sp),
                  rule=task.rule if task else "", tier=tier, mode=mode,
                  decision=decision or act.kind, reason=act.reason, score=act.score,
                  detail=detail, text=act.text, hidden_tokens=hidden, at_tokens=at, cap=cap,
                  pos=pos, client=self.req.client, profile=self.req.profile)
        if quiet:
            ev["quiet"] = True
        self.events.append(self.journal.write(ev))
        # the live view's banner says what was done, by which task, with its score and where
        self.last = {"task": ev["task"], "decision": ev["decision"], "mode": mode,
                     "reason": act.reason, "score": act.score, "tier": ev["tier"],
                     "at_tokens": at, "at": ev["ts"]}
        if mode == "act" and ev["decision"] not in (PASS, None):
            self.last_action = dict(self.last)
            self._tell(ev)
        elif mode == "shadow" and ev["decision"] not in (PASS, None):
            self.shadows += 1               # what an act would have done: the outlined badge

    def _tell(self, ev: dict) -> None:
        if self.on_event is None:
            return
        try:
            self.on_event(ev)
        except Exception as exc:                                   # noqa: BLE001
            print(f"[conductor] a state line failed: {type(exc).__name__}: {exc}", flush=True)


# the argument names opencode's and Hermes' file tools put a path in
_PATH_KEYS = ("filePath", "file_path", "path", "file")


def _call_facts(items: list[dict]) -> dict:
    """The tool names and file paths of an answer's calls, for the state line's narrator (only
    the `on_event` hook sees them; the journal does not)."""
    tools, files = [], []
    for it in items:
        fn = it.get("function") if isinstance(it, dict) else None
        if not isinstance(fn, dict):
            continue
        if fn.get("name"):
            tools.append(str(fn["name"]))
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError:
                args = None
        if isinstance(args, dict):
            p = next((args[k] for k in _PATH_KEYS if isinstance(args.get(k), str)), None)
            if p:
                files.append("/".join(p.rstrip("/").split("/")[-2:]))
    return {"tools": tools[:4], "files": files[:4]}


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1e3, 1)


def _fill(text: str, fields: dict) -> str:
    for k, v in fields.items():
        text = text.replace("{" + k + "}", v)
    return text


# ------------------------------------------------------------------ SP4 splices on token ids

def _find_last(seq: list[int], sub: list[int], end: int | None = None) -> int:
    """The start of the last occurrence of `sub` in `seq[:end]`, or -1."""
    end = len(seq) if end is None else end
    k = len(sub)
    for i in range(end - k, -1, -1):
        if seq[i:i + k] == sub:
            return i
    return -1


def tail_note(prompt: list[int], note: list[int], turn_start: list[int],
              turn_end: list[int]) -> list[int] | None:
    """`prompt` with `note` at the end of the last message, just before its turn end -- the
    turn end that precedes the generation prompt's turn start. None when the shape is not there
    (a completions prompt, a template without turns)."""
    s = _find_last(prompt, turn_start)
    if s < 0:
        return None
    e = _find_last(prompt, turn_end, s)
    if e < 0:
        return None
    return prompt[:e] + list(note) + prompt[e:]


def think_off(prompt: list[int], think_open: list[int], think_off_ids: list[int]
              ) -> list[int] | None:
    """`prompt` with the generation prompt's open reasoning block replaced by the closed empty
    one; None when the prompt does not end in an open block."""
    k = len(think_open)
    if k == 0 or prompt[-k:] != list(think_open):
        return None
    return prompt[:-k] + list(think_off_ids)
