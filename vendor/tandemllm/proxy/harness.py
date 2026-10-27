"""The harness in front of any backend (proxy P4): the conductor's features over HTTP.

`server/app.py` runs the conductor around the engine at six points. The proxy runs the same
conductor (the engine's own object in the embedded mode, one of its own in the standalone proxy)
around a relayed request at the same points: the plan before the relay, the request body before
the upstream send, the stream in the middle, the follow-up requests at its end, the ledger and
journal rows at the finish. The router, the packages, the library, the session tokens, the
transcripts, the journal and the monitor's rules are the existing modules, called unchanged; what
is new is the message-level form of the parts that touch tokens:

    the note        appended to the newest user message, replayed per session at the same message
                    index while the messages up to it hash the same
                    (server/conductor/inject_messages.py)
    the effort      the backend's own switch (`Adapter.thinking`): `chat_template_kwargs`
                    (vLLM, SGLang, any Qwen-family server), `reasoning_effort` (OpenAI's reasoning
                    models), `thinking.type` (DeepSeek); a budget is a word, never a hard cap
    the class       the router's question asked of the backend itself, with its own constraint
                    field (`Adapter.choice_body`), or of a `classifier` role whose endpoint names a
                    registered backend (`backend:<id>/<model>`)
    the monitor     E1 (the empty turn) and the tool-call regenerate as second requests whose
                    stream the client gets as the continuation of the first (one id, one role
                    delta, one [DONE]); D1 and D2 (the reasoning loop, the backstop) read the
                    streamed words (server/conductor/monitor/tier0_words.py) and close the
                    reasoning with a second request that continues it (`continue_final_message`)
                    where the backend has that; the tool-argument guard (server/argloop.py) stops
                    the stream
    compaction      the TandemLLM section appended to the compaction instruction

SWITCHES. A backend's `harness.mode` (off, shadow, on) and per model an override of the mode and
of each feature group. `off` is the P1 relay, byte for byte: nothing here runs. The defaults per
group (`DEFAULTS`) follow the P4 note: free features on everywhere, the paid ones (packages on
every turn, E1, the regenerate) off on a cloud backend until a model is switched on.

Every act is journaled as the engine's monitor journals it (the same `conductor_events` rows), and
every extra request's tokens and cost go into the journal row's detail (the in-turn ones) or a
ledger row of its own with the client `harness` (the class question), so the Backends screen can
show what each feature cost.
"""

from __future__ import annotations

import collections
import contextvars
import json
import math
import re
import threading
import time
import uuid

from proxy.adapters.base import CHAT, sse_data
from server.conductor import inject_messages as im

GROUPS = ("class", "effort", "packages", "side", "loops", "e1", "tools", "compaction",
          "states", "memory")
RANK = {"off": 0, "shadow": 1, "on": 2}
# the defaults by kind (P4 note, "Switches and defaults"): local engines, cloud APIs
DEFAULTS = {
    False: {"class": "on", "effort": "on", "packages": "on", "side": "on", "loops": "on",
            "e1": "on", "tools": "on", "compaction": "on", "states": "on", "memory": "on"},
    # memory (2026-10-05): a cloud model writes notes (shadow: the card and the writes, nothing
    # read back) until it is switched on per model; the profile's own switch still decides
    True: {"class": "on", "effort": "on", "packages": "off", "side": "on", "loops": "on",
           "e1": "off", "tools": "shadow", "compaction": "on", "states": "on",
           "memory": "shadow"},
}
# what each group costs or loses on a backend that is not our engine, one sentence each, by the
# backend's capabilities (shown next to the switch)
NOTES_MAX = 256
CLASS_TIMEOUT_MS = 8000
E1_TRIES = 2
REGEN_MAX = 2
CURRENT: contextvars.ContextVar = contextvars.ContextVar("proxy_turn", default=None)


def lower(a: str, b: str) -> str:
    return a if RANK.get(a, 0) <= RANK.get(b, 0) else b


def switches(backend: dict, model_row: dict | None) -> dict:
    """The effective switches of one model on one backend: `mode` and every group, each `off`,
    `shadow` or `on`, a group never above the mode."""
    mode = (backend.get("harness") or {}).get("mode") or "off"
    feats = (model_row or {}).get("features") or {}
    if feats.get("mode") not in (None, "inherit"):
        mode = feats["mode"]
    out = {"mode": mode}
    base = DEFAULTS[bool(backend.get("cloud"))]
    for g in GROUPS:
        v = feats.get(g, "inherit")
        v = base[g] if v in (None, "inherit") else v
        out[g] = lower(v, mode)
    return out


def degradations(backend: dict, caps: dict) -> dict:
    """group -> one sentence: what the feature costs or loses on this backend."""
    cloud = bool(backend.get("cloud"))
    sw = caps.get("thinking_switch", "none")
    effort = {
        "chat_template_kwargs": "Thinking on or off through chat_template_kwargs; an effort "
                                "level is a word, never a hard cap.",
        "reasoning_effort": "Effort through reasoning_effort, on the o-series and gpt-5 models "
                            "only; no hard cap.",
        "thinking_type": "Thinking on or off only (thinking.type); no budget, no cap.",
    }.get(sw, "This backend has no thinking switch: the effort is journaled as unsupported.")
    choice = caps.get("choice", "text")
    klass = ("One short extra request to this backend per new user message"
             + (" (a few hundred prompt tokens, billed)" if cloud else "")
             + ("; answered as plain text and parsed, no constraint field" if choice == "text"
                else f"; constrained by {choice}") + ".")
    cache = bool(caps.get("cache_counts"))
    packages = ("The note sits at the end of the user message, so the model can quote it; it is "
                "replayed at the same message index, so "
                + ("the backend's prompt cache keeps its match" if cache else
                   "a backend with a prefix cache keeps its match; without one every turn reads "
                   "the note again")
                + (", and it costs input tokens every turn." if cloud else "."))
    loops = ("Loops are read from the streamed words (thresholds measured for words); "
             + ("a close is a second request that continues the reasoning."
                if caps.get("continue_final_message") else
                "this backend cannot continue an assistant message, so a reasoning loop is "
                "journaled, never closed. The tool-argument guard stops the stream."))
    return {
        "class": klass,
        "effort": effort,
        "packages": packages,
        "side": "No loss: the cap and thinking off go into the request, which saves tokens.",
        "loops": loops,
        "e1": "An empty answer costs one more full request" + (", billed." if cloud else "."),
        "tools": "The checks are free; a regenerate is one more full request"
                 + (", billed" if cloud else "") + ", and the client never sees the first one.",
        "compaction": "The summary keeps the TandemLLM section; the backend reads the client's "
                      "whole history cold (the engine's continuation is engine-only).",
        "states": "Template lines unless a narrator role names a backend; no prefill progress "
                  "(the backend's prefill is not visible).",
        "memory": ("The memory card and the notes sit at the end of the user message"
                   + (" and cost input tokens on every new message" if cloud else "")
                   + "; the model's blocks are taken out of the stream and written. A streamed "
                     "answer only. The background writer is one more request to this backend "
                   + ("when a session ends or compacts, off on a paid model unless switched on, "
                      "within the backend's daily cap." if cloud else
                      "in its idle time, when a session ends or compacts; a client request "
                      "stops it.")),
    }


class MsgNote:
    __slots__ = ("idx", "hash", "text", "key", "kinds", "parts", "tokens", "at", "shas")

    def __init__(self, idx, hsh, text, key, kinds, parts, shas=None):
        self.idx, self.hash, self.text, self.key = idx, hsh, text, key
        self.kinds, self.parts = set(kinds or ()), list(parts or [])
        self.shas = list(shas or [])     # each part's text hash (router.Plan._fresh_block)
        self.tokens = im.est_tokens(text)
        self.at = time.time()


class _Headers:
    """The client's headers with the harness's overrides (the backend's profile, a lowered
    mode), read the way the router reads them."""

    def __init__(self, base, over: dict):
        self.base, self.over = base, {k.lower(): v for k, v in over.items()}

    def get(self, name, default=None):
        v = self.over.get(name.lower())
        if v is not None:
            return v
        try:
            return self.base.get(name, default)
        except Exception:                                          # noqa: BLE001
            return default

    def items(self):
        out = dict(self.base.items()) if hasattr(self.base, "items") else {}
        out.update(self.over)
        return out.items()


class _Sup:
    """What the SP3 rules (server/conductor/monitor/toolcheck.py) read of a supervisor."""

    def __init__(self, messages, tools):
        self.memo: dict = {}
        self.task_acts: dict = {}
        self.req = type("Req", (), {"messages": messages or [], "tools_offered": bool(tools),
                                    "n": 1, "max_tokens": None})()
        self.tool_types = None


# ---------------------------------------------------------------------------------------------
class Harness:
    """One proxy's harness: the conductor it runs, the message-level notes per session, the
    pending notes (the argument guard's), and the requests it asks of backends itself."""

    def __init__(self, svc, conductor, monitor_base=None, log=None):
        self.svc = svc
        self.cond = conductor
        self.base = monitor_base
        self.log = log or (lambda m: None)
        self._notes: collections.OrderedDict = collections.OrderedDict()
        self._pending: dict = {}
        self._lock = threading.Lock()
        self.stats = collections.Counter()
        self.writer = None
        if conductor is not None:
            conductor.router.stage_override = self._stage
            conductor.router.cache_scope = self._scope
            if getattr(conductor, "memory", None) is not None:
                # M3 (2026-10-05): the background memory writer as a request to the session's
                # own backend, in its idle time (proxy/memwriter.py)
                from proxy.memwriter import ProxyWriter
                self.writer = ProxyWriter(self)

    def stop(self) -> None:
        if self.writer is not None:
            self.writer.stop()

    # ----------------------------------------------------------------- the router's stages
    @staticmethod
    def _scope():
        t = CURRENT.get()
        return None if t is None else f"{t.backend['id']}/{t.route.model}"

    def _stage(self, stage: dict, roles: dict):
        t = CURRENT.get()
        if t is None:
            return None                       # the engine's own request: as configured
        if stage["kind"] == "decider":
            return None                       # Laya stays whatever the role says
        if t.sw["class"] == "off":
            return False
        role = roles.get("classifier") or {}
        ep = str(role.get("endpoint") or "")
        if ep.startswith("backend:") and role.get("mode", "off") != "off":
            bid, _, model = ep[8:].partition("/")
            b = self.svc.table.backend_for(bid) or self.svc.cfg.backend(bid)
            if b is not None:
                return BackendQuestion(self, b, model or t.route.model,
                                       int(stage.get("timeout_ms") or CLASS_TIMEOUT_MS), t)
        return BackendQuestion(self, t.backend, t.route.model,
                               int(stage.get("timeout_ms") or CLASS_TIMEOUT_MS), t)

    # ----------------------------------------------------------------- one request
    def begin(self, handler, path: str, body: dict, route, rec):
        """The request's Turn, or None when the harness does not run for it: the backend's mode
        (or the model's) is off, it is not a chat completion, or no conductor is configured. With
        None the relay is P1's, byte for byte."""
        if self.cond is None or path != CHAT or not isinstance(body, dict) or \
                not isinstance(body.get("messages"), list) or route is None:
            return None
        backend = self.svc.table.backend_for(route.backend)
        if backend is None:
            return None
        row = (self.svc.table.snapshot()["by_backend"].get(route.backend) or {}).get(route.model)
        sw = switches(backend, row)
        if sw["mode"] == "off":
            return None
        try:
            if int(body.get("n") or 1) != 1:
                return None
        except (TypeError, ValueError):
            return None
        t = Turn(self, handler, path, body, route, backend, sw, rec)
        try:
            t.plan_it()
        except Exception as exc:                                   # noqa: BLE001
            # a conductor failure relays the request as P1 does, and says so
            self.log(f"[harness] {rec.request_id}: plan failed, relayed as is: "
                     f"{type(exc).__name__}: {exc}")
            self.stats["plan_failed"] += 1
            return None
        if t.plan is None or not t.plan.active:
            t.finish_raw()
            return None
        return t

    # ----------------------------------------------------------------- the notes per session
    def notes_of(self, key: str) -> list:
        with self._lock:
            got = self._notes.get(key)
            if got is not None:
                self._notes.move_to_end(key)
            return list(got or [])

    def keep_notes(self, key: str, notes: list) -> None:
        with self._lock:
            self._notes[key] = list(notes)
            self._notes.move_to_end(key)
            while len(self._notes) > NOTES_MAX:
                self._notes.popitem(last=False)

    def pend(self, key: str, what: str, value) -> None:
        with self._lock:
            self._pending.setdefault(key, {})[what] = value

    def take_pending(self, key: str, what: str):
        with self._lock:
            return (self._pending.get(key) or {}).pop(what, None)

    def peek_pending(self, key: str) -> dict:
        with self._lock:
            return dict(self._pending.get(key) or {})

    # ----------------------------------------------------------------- requests of its own
    def post(self, backend: dict, adapter, body: dict, timeout_s: float,
             extra_headers=()) -> tuple[int, object, float]:
        """One non-streamed request to a backend through its pool: (status, the JSON answer or
        None, ms). Never raises."""
        from proxy import http1
        t0 = time.perf_counter()
        try:
            key = self.svc.cfg.key_for(backend["id"])
            pool = self.svc.relay.pool(backend)
            data = json.dumps(body, ensure_ascii=False).encode()
            headers = adapter.headers(key) + list(extra_headers)
            conn = pool.get()
            try:
                conn.send("POST", pool.path(CHAT), headers, data)
                status, _, _ = conn.read_head(timeout_s)
                raw = conn.read_all(timeout_s)
                pool.put(conn)
            except Exception:
                conn.close()
                raise
            try:
                obj = json.loads(raw or b"null")
            except ValueError:
                obj = None
            return status, obj, (time.perf_counter() - t0) * 1e3
        except Exception as exc:                                   # noqa: BLE001
            why = "timeout" if isinstance(exc, http1.Timeout) else type(exc).__name__
            return 0, {"_error": why}, (time.perf_counter() - t0) * 1e3

    def side_row(self, parent, backend: dict, adapter, model: str, public: str, what: str,
                 obj, status: int, ms: float, n: int = 1) -> dict:
        """A ledger row for a request the harness asked by itself (the class question): client
        `harness`, its own tokens and cost. Returns the normalised usage and the cost."""
        from proxy.measure import RelayRecord
        rec = RelayRecord(f"{parent.request_id}-{what}{n if n > 1 else ''}", "chat", False)
        rec.model, rec.model_served, rec.public_model = public, public, public
        rec.client_kind, rec.client_id = "harness", f"harness:{what}"
        rec.backend, rec.backend_kind, rec.backend_model = backend["id"], backend["kind"], model
        rec.user = getattr(parent, "user", None)
        rec.status = status or 502
        u = adapter.read_usage((obj or {}).get("usage") if isinstance(obj, dict) else None)
        rec.set_usage(u)
        rec.cost_micros = adapter.price(u, self.svc.price_of(backend["id"], model), rec.ts)
        rec.finish_reason = "stop" if status == 200 else "error"
        rec.end()
        self.svc.account(rec)
        return {"usage": u, "cost_micros": rec.cost_micros}

    # ----------------------------------------------------------------- the GUI's figures
    def feature_costs(self, bid: str, since_ms: int) -> dict:
        """model -> group -> {"requests", "cost_micros"} since `since_ms`: the class questions
        (ledger rows of the client `harness`) and the in-turn second requests (the journal's
        monitor rows, their detail's cost)."""
        out: dict = {}
        led = self.svc.ledger
        if led is not None:
            try:
                for model, cid, n, cost in led.reader().execute(
                        "SELECT backend_model, client_id, COUNT(*), SUM(cost_micros) FROM requests "
                        "WHERE backend = ? AND client_kind = 'harness' AND ts_ms >= ? "
                        "GROUP BY backend_model, client_id", (bid, since_ms)):
                    g = {"harness:class": "class", "harness:writer": "memory"}.get(cid, "class")
                    d = out.setdefault(model, {}).setdefault(g, {"requests": 0,
                                                                 "cost_micros": None})
                    d["requests"] += int(n)
                    if cost is not None:
                        d["cost_micros"] = (d["cost_micros"] or 0) + int(cost)
            except Exception:                                      # noqa: BLE001
                pass
        j = getattr(self.cond, "journal", None)
        if j is not None and getattr(j, "path", None):
            try:
                import sqlite3
                con = sqlite3.connect(f"file:{j.path}?mode=ro", uri=True, timeout=1.0)
                try:
                    for task, detail in con.execute(
                            "SELECT task, detail FROM conductor_events WHERE kind = 'monitor' AND "
                            "ts_ms >= ? AND detail LIKE ?", (since_ms, f'%"backend": "{bid}"%')):
                        try:
                            d = json.loads(detail or "{}")
                        except ValueError:
                            continue
                        if d.get("backend") != bid or "extra_request" not in d:
                            continue
                        g = {"E1": "e1", "D1": "loops", "D2": "loops"}.get(task, "tools")
                        x = out.setdefault(d.get("model") or "?", {}).setdefault(
                            g, {"requests": 0, "cost_micros": None})
                        x["requests"] += 1
                        c = (d.get("extra_request") or {}).get("cost_micros")
                        if c is not None:
                            x["cost_micros"] = (x["cost_micros"] or 0) + int(c)
                finally:
                    con.close()
            except Exception:                                      # noqa: BLE001
                pass
        return out


# ---------------------------------------------------------------------------------------------
class BackendQuestion:
    """The router's classifier stage, asked of a backend: the `Qwen` client's `choice` over the
    backend's own constraint field (`Adapter.choice_body`). Its answer reads as the engine's
    stage `qwen`; the model is `<backend>/<model>`."""

    def __init__(self, hx: Harness, backend: dict, model: str, timeout_ms: int, turn):
        self.hx, self.backend, self.model = hx, backend, model
        self.timeout_ms, self.turn = int(timeout_ms), turn

    def choice(self, state: str, prompt: str, options: dict, max_chars: int = 2000):
        from server.conductor.classify import Answer
        ids = list(options)
        adapter = self.hx.svc.adapter(self.backend)
        body, how = adapter.choice_body(prompt, state[:max_chars], ids, self.model)
        status, obj, ms = self.hx.post(self.backend, adapter, body, self.timeout_ms / 1000.0,
                                       adapter.choice_headers())
        who = f"{self.backend['id']}/{self.model}"
        if self.turn is not None:
            self.turn.questions += 1
            self.hx.side_row(self.turn.rec, self.backend, adapter, self.model,
                             self.turn.route.public, "class", obj, status, ms,
                             n=self.turn.questions)
        if status != 200 or not isinstance(obj, dict):
            why = (obj or {}).get("_error") if isinstance(obj, dict) else None
            return Answer("qwen", ms=ms, error=f"qwen_{'timeout' if why == 'timeout' else ('http_' + str(status) if status else 'unreachable')}",
                          model=who)
        ch = (obj.get("choices") or [{}])[0] or {}
        text = str((ch.get("message") or {}).get("content") or "")
        label = parse_choice(text, ids, how)
        toks = ((ch.get("logprobs") or {}).get("content")) or []
        conf = 0.5
        if toks and how != "json":
            try:
                conf = math.exp(sum(float(t.get("logprob", 0.0)) for t in toks))
            except (TypeError, ValueError, OverflowError):
                conf = 0.5
        if label is None:
            return Answer("qwen", None, 0.0, {}, ms, error="qwen_unknown_label", model=who)
        return Answer("qwen", label, conf, {label: conf}, ms, model=who)


def parse_choice(text: str, ids: list[str], how: str) -> str | None:
    """The option a backend answered: the JSON field, the constrained text, or plain text in
    which exactly one option id stands as a word."""
    t = (text or "").strip()
    if how == "json":
        try:
            v = json.loads(t).get("choice")
        except (ValueError, AttributeError):
            v = None
        return v if v in ids else None
    bare = t.strip("`'\". \n").split("\n")[0].strip("`'\". ")
    if bare in ids:
        return bare
    found = [i for i in ids if re.search(r"(?<![\w-])" + re.escape(i) + r"(?![\w-])", t)]
    return found[0] if len(found) == 1 else None


# ---------------------------------------------------------------------------------------------
class Turn:
    """One relayed chat request under the harness: its plan, the body it sends, what the stream
    said, what the monitor did."""

    def __init__(self, hx: Harness, handler, path, body, route, backend, sw, rec):
        self.hx, self.handler, self.path, self.route = hx, handler, path, route
        self.backend, self.sw, self.rec = backend, sw, rec
        self.client_body = body
        self.stream = bool(body.get("stream"))
        opts = body.get("stream_options")
        self.client_usage = isinstance(opts, dict) and opts.get("include_usage") is True
        self.tools = bool(body.get("tools"))
        self.messages = body.get("messages") or []
        self.plan = None
        self.policy = None
        self.notes_new: list = []
        self.notes_live: list = []
        self.body_base: dict | None = None     # the client's body after the plan (before the
        self.sent: dict | None = None          # adapter's switches), and the last body sent
        self.adapter = None
        self.upstream_model = None
        self.effort_applied: str | None = None
        self.events: list[dict] = []
        self.acts = 0
        self.hidden_tokens = 0
        self.states = None
        self.lines: list[bytes] = []
        self.usages: list = []                 # normalised usage of each request, in order
        self.extra: list[dict] = []            # the second requests: what, usage, cost
        self._reset_stream()
        self.ident: dict | None = None         # id, model, created of the first chunk
        self.done_sent = False
        self.e1_tries = 0
        self.regens = 0
        self.closed = False
        self.stop: str | None = None           # why the upstream read stopped early
        self.argloop = None
        self.follow: str | None = None         # the follow-up in flight: e1, regen, close
        self.repaired: dict = {}
        self.word_cfg = None
        self.words = None
        self.d1_seen = False
        self.d2_seen = False
        self.arg_guard = None
        self._arg_idx = None
        self._arg_n = 0
        self.a1_note = None
        self.mem = None                        # 2026-10-05: the memory blocks' filter
        self.questions = 0
        self.close_text = None
        self._pending_extra = None
        self.t0 = time.perf_counter()

    def _reset_stream(self) -> None:
        self.reasoning: list[str] = []
        self.content: list[str] = []
        self.calls: dict = {}                  # index -> {"id", "name", "args"}
        self.open_call = None
        self.fin = None
        self.in_tag = False                    # inside <think> in the content
        self.held_tools: list[bytes] = []
        self.held_end: list[bytes] = []        # the finish frame(s)
        self.held_usage: tuple | None = None   # (bytes, obj) of the usage frame the client asked
        self.held_done: bytes | None = None

    # ----------------------------------------------------------------- the plan
    def plan_it(self) -> None:
        hx, rec = self.hx, self.rec
        cond = hx.cond
        prof = (self.backend.get("harness") or {}).get("profile")
        over = {}
        try:
            client_prof = self.handler.headers.get("X-Tandem-Profile")
        except Exception:                                          # noqa: BLE001
            client_prof = None
        if prof and not client_prof:
            over["X-Tandem-Profile"] = prof
        if self.sw["mode"] == "shadow":
            over["X-Tandem"] = "shadow"
        headers = _Headers(self.handler.headers, over)
        self.hdrs = headers
        # the state lines a client sends back are taken out before anything reads the messages
        from server.conductor import states as states_mod
        msgs, stripped = states_mod.strip_messages(self.messages)
        body = self.client_body
        if stripped:
            body = dict(body, messages=msgs)
            self.messages = msgs
        caps = {"classify": self.sw["class"], "effort": self.sw["effort"],
                "packages": self.sw["packages"], "rules": self.sw["packages"],
                "side": self.sw["side"],
                # 2026-10-05: the memory notes, of the proxy key's user (none: nothing)
                "memory": self.sw.get("memory", "off") if self.stream else "off"}
        token = CURRENT.set(self)
        try:
            plan = cond.router.plan(body, headers, client_kind=rec.client_kind or "unknown",
                                    request_id=rec.request_id, chat=True, caps=caps,
                                    user=getattr(rec, "user", None) or None)
        finally:
            CURRENT.reset(token)
        self.plan = plan
        rec.plan = plan
        rec.profile, rec.profile_version = plan.res.profile, plan.res.version
        rec.conductor_mode = plan.res.mode
        if not plan.active:
            return
        if plan.row is not None:
            plan.row["extra"]["proxy"] = {"backend": self.backend["id"],
                                          "kind": self.backend["kind"],
                                          "model": self.route.model, "switches": dict(self.sw)}
            if stripped:
                plan.row["extra"]["states_stripped"] = stripped
        self.body_base = body
        if hx.base is not None:
            try:
                self.policy = cond.monitor_policy(plan, hx.base)
            except Exception:                                      # noqa: BLE001
                self.policy = None
        from server.conductor.monitor import tier0_words
        self.word_cfg = tier0_words.words_cfg(self.policy.data if self.policy is not None
                                              else None)
        if self._task_mode("D1") != "off" or self._task_mode("D2") != "off":
            self.words = tier0_words.WordLoop(self.word_cfg)
        if self._task_mode("A1") != "off" and self.tools:
            from server import argloop
            self.arg_guard = argloop.ArgLoopGuard(
                context_chars=argloop.longest_message_chars(self.messages))
        self._notes()
        self._states()
        if getattr(plan, "memory", None) is not None and getattr(cond, "memory", None) is not None:
            # 2026-10-05: the notes that went in count as used; the model's blocks are taken out
            # of the stream and written (server/memblock.py)
            from server.conductor import memory as memory_mod
            memory_mod.mark_injected(cond.memory, plan)
            self.mem = cond.memory.filter(plan.memory, self.states)

    def _mem_flush(self) -> bytes:
        """The memory filter's rest (an unclosed block goes out as the text it was), once, and
        what its blocks did on the journal row."""
        f, self.mem = self.mem, None
        if f is None:
            return b""
        rest = f.finish()
        plan = self.plan
        if plan is not None and plan.row is not None and plan.memory:
            info = plan.row["extra"].setdefault("memory", {})
            if plan.memory.get("written"):
                info["written"] = [{k: v for k, v in w.items() if k != "chars"}
                                   for w in plan.memory["written"]]
            if f.failed:
                info["failed"] = list(f.failed)
        if rest:
            self.content.append(rest)
            return self._chunk({"content": rest})
        return b""

    def finish_raw(self) -> None:
        """A plan that came back raw (a profile in raw, `X-Tandem: raw`): the ledger's columns."""
        self.rec.conductor_acts = 0
        self.rec.conductor_hidden_tokens = 0
        self.rec.conductor_acted = 0

    def _task_mode(self, tid: str) -> str:
        """A monitor task's mode for this request: the profile's policy, lowered by the group's
        switch (`on` is the task's own mode, `shadow` at most shadow, `off` off)."""
        group = {"D1": "loops", "D2": "loops", "A1": "loops", "E1": "e1"}.get(tid, "tools")
        g = self.sw.get(group, "off")
        if g == "off" or self.plan is None or self.plan.res.mode == "raw":
            return "off"
        if tid == "A1":
            mode = "act"
        elif self.policy is None or tid not in self.policy.tasks:
            return "off"
        else:
            mode = self.policy.tasks[tid].mode
        if mode == "off":
            return "off"
        if g == "shadow" or self.plan.res.mode == "shadow":
            return "shadow"
        return mode

    # ----------------------------------------------------------------- the notes
    def _notes(self) -> None:
        plan, hx = self.plan, self.hx
        sess = plan.session
        if sess is None or plan.side is not None:
            return
        cfg = plan.router_cfg or {}
        inj = cfg.get("injection") or {}
        wrapper = inj.get("wrapper") or "\n\n{text}"
        at = im.last_user(self.messages)
        old = hx.notes_of(sess.key)
        keep_all = bool(inj.get("replay", True))
        hashes = im.hashes_at(self.messages, [n.idx for n in old] + ([at] if at is not None
                                                                     else []))
        live = []
        for n in old:
            if hashes.get(n.idx) != n.hash or not keep_all:
                continue
            if any(plan.decisions.get(k) != "on" for k in n.kinds):
                continue
            live.append(n)
        budget = int(inj.get("max_replay_tokens", 16000))
        while live and sum(n.tokens for n in live) > budget:
            live.pop(0)
        adds = []
        if at is not None:
            cap = int(inj.get("max_note_tokens", 8000))
            keys = {n.key for n in live}
            parts = {(p.get("kind"), p.get("name")) for n in live for p in n.parts}
            todo = []
            blk = plan.block
            if blk is not None and blk["key"] not in keys:
                # only the parts no live note holds with the same text (the engine's rule,
                # server/conductor/router.py `Plan._fresh_block`)
                blk = plan._fresh_block(blk, live)
            if blk is not None and blk["key"] not in keys:
                todo.append((blk, blk.get("reason")))
            for x in plan.extras:
                if x["key"] in keys or (x.get("unless") and tuple(x["unless"]) in parts):
                    continue
                todo.append((x, x.get("reason")))
            for blk, reason in todo:
                text = (blk.get("wrapper") or wrapper).replace("{text}", blk["text"])
                if im.est_tokens(text) > cap:
                    plan.row["extra"].setdefault("errors", []).append(
                        f"note_too_long:{im.est_tokens(text)}")
                    continue
                n = MsgNote(at, hashes.get(at), text, blk["key"], blk.get("kinds"),
                            blk.get("parts"), blk.get("shas"))
                adds.append((n, reason))
                parts |= {(p.get("kind"), p.get("name")) for p in n.parts}
        notes = live + [n for n, _ in adds]
        hx.keep_notes(sess.key, notes)
        self.notes_live, self.notes_new = live, adds
        # the argument guard's note for this session's next request (A1), once, not replayed
        pend = hx.take_pending(sess.key, "argloop") if self.tools else None
        if pend is not None and at is not None and self._task_mode("A1") != "off":
            from server import argloop
            text = "\n\n" + argloop.note_text(pend)
            mode = self._task_mode("A1")
            ev = self._journal("SP4", "A1", "tool_arg_loop", "PROMPT_TAIL_NOTE",
                               f"tool_arg_loop_note({pend.get('rule')})", mode, detail=pend,
                               text=text.strip(), hidden=im.est_tokens(text), pos=at)
            if mode == "act":
                self.a1_note = (at, text)
                self.acts += 1
                self.hidden_tokens += ev["hidden_tokens"]
        row = plan.row
        if live:
            row["replayed"] = {"count": len(live), "tokens": sum(n.tokens for n in live),
                               "positions": [n.idx for n in live], "estimated": True}
        for n, reason in adds:
            plan.injected.append({"pos": n.idx, "at": n.idx, "tokens": n.tokens,
                                  "text": n.text, "parts": n.parts, "kinds": sorted(n.kinds),
                                  "reason": reason, "unit": "message", "estimated": True})
            row["injected"] = plan.injected
            row["acted"] = 1
        if live or adds:
            plan.changed = True
            row["changed"] = 1

    def _states(self) -> None:
        plan = self.plan
        ui = (plan.res.doc or {}).get("ui") or {}
        if self.sw["states"] != "on" or ui.get("states") != "on" or not self.stream or \
                plan.res.mode != "on" or plan.side is not None:
            return
        from server.conductor import states as states_mod
        _, packages = self.hx.cond.store.peek("packages")
        self.states = states_mod.States(ui.get("kinds") or {}, narrator=self._narrator(ui))
        self.states.route(plan.row, plan.injected, packages.get("packages") or {},
                          plan.compaction)

    def _narrator(self, ui: dict):
        """A narrator role whose endpoint names a backend (`backend:<id>/<model>`): the lines
        are written there inside the role's latency budget; else the templates."""
        if ui.get("narrator") not in ("on",):
            return None
        _, roles = self.hx.cond.store.peek("roles")
        role = roles.get("narrator") or {}
        ep = str(role.get("endpoint") or "")
        if role.get("mode") != "on" or not ep.startswith("backend:"):
            return None
        bid, _, model = ep[8:].partition("/")
        b = self.hx.svc.cfg.backend(bid)
        if b is None or not model:
            return None
        return BackendNarrator(self.hx, b, model, float(role.get("latency_budget_ms") or 150))

    # ----------------------------------------------------------------- the body
    def upstream(self, body: dict, adapter, upstream_model: str) -> tuple[dict, bool]:
        """The body for one try on one backend (a fallback may be another kind): the notes,
        the thinking switch, the side request's cap, the compaction section. (body, changed)."""
        plan = self.plan
        self.adapter, self.upstream_model = adapter, upstream_model
        base = self.body_base or body
        changed = base is not body
        msgs = base.get("messages") or []
        notes = [(n.idx, n.text) for n in self.notes_live] + \
                [(n.idx, n.text) for n, _ in self.notes_new]
        if self.a1_note is not None:
            notes.append(self.a1_note)
        if plan.compaction and self.sw["compaction"] != "off":
            sec = self._compaction_section()
            at = im.last_user(msgs)
            if sec and at is not None:
                if self.sw["compaction"] == "on" and plan.res.mode == "on":
                    notes.append((at, sec))
                plan.row["extra"]["compaction"] = {"section": self.sw["compaction"] == "on" and
                                                   plan.res.mode == "on",
                                                   "history": "client_history",
                                                   "tokens": im.est_tokens(sec)}
        out = dict(base)
        if notes:
            out["messages"] = im.splice(msgs, notes)
            changed = True
        on = not plan.thinking_off
        level = plan.row.get("effort") if plan.row else None
        if plan.side is not None:
            on = plan.side.get("thinking") == "client"
            cap = int(plan.side["max_tokens"])
            for f in ("max_tokens", "max_completion_tokens"):
                v = out.get(f)
                if isinstance(v, int) and not isinstance(v, bool) and v <= cap:
                    break
            else:
                f = "max_completion_tokens" if "max_completion_tokens" in out else "max_tokens"
                out[f] = cap
                changed = True
        if plan.side is not None or plan.decisions.get("effort") == "on":
            out2, applied = adapter.thinking(out, on, None if on is False and plan.side
                                             else level, upstream_model)
            if out2 is not out:
                out, changed = out2, True
            self.effort_applied = applied
            if plan.row is not None:
                plan.row["extra"]["effort_switch"] = applied
        self.sent = out
        return out, changed

    def headers(self, adapter, headers: list) -> list:
        """The upstream headers under the harness: another TandemLLM gets `X-Tandem: raw`, so
        its own conductor does not run a second time over what this one already did."""
        if adapter.kind != "tandemllm":
            return headers
        return [(k, v) for k, v in headers if k.lower() != "x-tandem"] + [("X-Tandem", "raw")]

    def _compaction_section(self) -> str:
        """The TandemLLM section the compaction's summary must keep: the packages in force, the
        session's address, what the session fetched, the checks still open."""
        plan, hx = self.plan, self.hx
        names = []
        for n in self.notes_live + [n for n, _ in self.notes_new]:
            for p in n.parts:
                if p.get("kind") == "package" and p.get("name") not in names:
                    names.append(p.get("name"))
        lines = []
        if names:
            lines.append("- packages in force: " + ", ".join(names))
        reg = hx.cond.router.session_tokens
        key = plan.session.key if plan.session is not None else None
        if reg is not None and key:
            from server import session_tokens as st_mod
            from server.conductor import packages as packages_mod
            url = packages_mod.request_url(self.hdrs) or packages_mod.URL_DEFAULT
            lines.append("- this session's runtime address: "
                         + st_mod.with_token(url, reg.token_for(key)))
            calls = reg.calls(key) or {}
            if calls:
                lines.append("- fetched this session: " + ", ".join(
                    f"{k} {v}" for k, v in sorted(calls.items())))
        if key and hx.peek_pending(key):
            lines.append("- open checks: " + ", ".join(sorted(hx.peek_pending(key))))
        if not lines:
            return ""
        return ("\n\n<runtime-note>\nKeep this section in the summary as it is, under the "
                "heading \"TandemLLM\"; the runtime that serves you needs it after the "
                "compaction.\n" + "\n".join(lines) + "\n</runtime-note>")

    # ----------------------------------------------------------------- the stream
    @property
    def hold_end(self) -> bool:
        """Hold the finish frame, the usage frame and [DONE] until the turn is settled: E1 or a
        check may still answer with a second request."""
        if not self.stream or not self.tools:
            return False
        return self._task_mode("E1") == "act" or self.hold_tools

    @property
    def hold_tools(self) -> bool:
        if not self.stream or not self.tools:
            return False
        return any(self._task_mode(t) == "act" for t in ("C1", "C2", "C3", "C4"))

    def feed(self, event: bytes, data_part: bytes, drop: bool) -> bytes:
        """One upstream SSE event (with its separator): what goes to the client now."""
        data = sse_data(data_part)
        if data is None:
            return event if not drop else b""
        if data.strip() == b"[DONE]":
            if self.hold_end and not self.done_sent:
                self.held_done = event
                return b""
            self.done_sent = True
            return self._mem_flush() + self._lines() + event
        try:
            obj = json.loads(data)
        except ValueError:
            return event if not drop else b""
        if not isinstance(obj, dict):
            return event if not drop else b""
        rewrite = False
        if self.follow is not None and self.ident is not None:
            rewrite = self._as_first(obj)
            if rewrite is None:
                return b""                   # the second request's role delta
        elif self.ident is None and obj.get("id"):
            self.ident = {"id": obj.get("id"), "model": obj.get("model"),
                          "created": obj.get("created"),
                          "object": obj.get("object") or "chat.completion.chunk"}
        if drop:
            return b""
        if self.mem is not None and not self.in_tag:
            # the model's memory blocks stay on the proxy (server/memblock.py)
            for ch in obj.get("choices") or []:
                d = ch.get("delta") if isinstance(ch, dict) else None
                if isinstance(d, dict) and isinstance(d.get("content"), str) and d["content"]:
                    got = self.mem.feed(d["content"])
                    if got != d["content"]:
                        d["content"] = got
                        rewrite = True
        out = (json.dumps(obj, ensure_ascii=False).encode() if rewrite else None)
        ev = (b"data: " + out + b"\n\n") if out is not None else event
        has_tool = False
        fin = None
        stopped_before = self.stop is not None
        for ch in obj.get("choices") or []:
            if not isinstance(ch, dict):
                continue
            d = ch.get("delta") if isinstance(ch.get("delta"), dict) else {}
            self._read_delta(d)
            if d.get("tool_calls"):
                has_tool = True
            if ch.get("finish_reason"):
                fin = ch["finish_reason"]
                self.fin = fin
        if self.stop is not None:
            if not stopped_before and self.stop in ("D1", "D2"):
                return ev + self._lines()      # the piece that tipped it, then the close
            return self._lines()
        usage_only = not obj.get("choices") and isinstance(obj.get("usage"), dict)
        if usage_only and self.hold_end:
            self.held_usage = (ev, obj)
            return self._lines()
        if has_tool and self.hold_tools and fin is None:
            self.held_tools.append(ev)
            return self._lines()
        if fin is not None and self.hold_end:
            if has_tool and self.hold_tools:
                self.held_tools.append(ev)
            else:
                self.held_end.append(ev)
            return self._lines()
        return ev + self._lines()

    def _as_first(self, obj: dict):
        """A follow-up's chunk as a continuation of the first answer: its id, model and created;
        no second role delta. True when changed, None to drop the frame."""
        changed = False
        for k in ("id", "model", "created"):
            if self.ident.get(k) is not None and obj.get(k) != self.ident[k]:
                obj[k] = self.ident[k]
                changed = True
        chs = obj.get("choices") or []
        for ch in chs:
            if isinstance(ch, dict) and isinstance(ch.get("delta"), dict) and \
                    "role" in ch["delta"]:
                d = dict(ch["delta"])
                d.pop("role", None)
                if not any(v for v in d.values()) and not ch.get("finish_reason") and \
                        not obj.get("usage"):
                    return None
                ch["delta"] = d
                changed = True
        return changed

    def _read_delta(self, d: dict) -> None:
        for f in ("reasoning_content", "reasoning"):
            v = d.get(f)
            if isinstance(v, str) and v:
                self.reasoning.append(v)
                self._watch_reasoning(v)
        c = d.get("content")
        if isinstance(c, str) and c:
            self._read_content(c)
        tc = d.get("tool_calls")
        if isinstance(tc, list):
            for i, call in enumerate(tc):
                if not isinstance(call, dict):
                    continue
                idx = call.get("index", i)
                e = self.calls.setdefault(idx, {"id": None, "name": "", "args": ""})
                if call.get("id"):
                    e["id"] = call["id"]
                fn = call.get("function") or {}
                if fn.get("name"):
                    e["name"] = fn["name"]
                a = fn.get("arguments")
                if isinstance(a, str) and a:
                    e["args"] += a
                    self.open_call = idx
                    self._watch_args(idx, e, a)

    def _read_content(self, c: str) -> None:
        """Content text; `<think>...</think>` inside it is reasoning (a backend without a
        reasoning parser)."""
        while c:
            if self.in_tag:
                j = c.find("</think>")
                if j < 0:
                    self.reasoning.append(c)
                    self._watch_reasoning(c)
                    return
                self.reasoning.append(c[:j])
                self._watch_reasoning(c[:j])
                self.in_tag = False
                c = c[j + 8:]
                continue
            j = c.find("<think>")
            if j < 0:
                self.content.append(c)
                return
            self.content.append(c[:j])
            self.in_tag = True
            c = c[j + 7:]

    # ----------------------------------------------------------------- D1, D2, the guard
    def _watch_reasoning(self, text: str) -> None:
        if self.words is None or self.closed or self.stop is not None or self.follow == "close":
            return
        why = self.words.feed(text)
        if why is not None and not self.d1_seen:
            self.d1_seen = True
            self._loop_verdict("D1", "reasoning_loop", why)
            return
        d2 = self._task_mode("D2")
        if d2 != "off" and not self.d2_seen and self.policy is not None:
            t = self.policy.tasks["D2"]
            mt = self.client_body.get("max_tokens") or self.client_body.get(
                "max_completion_tokens")
            limit = int(t["cap"])
            if isinstance(mt, int) and mt > int(t["reserve"]):
                limit = min(limit, mt - int(t["reserve"]))
            est = im.est_tokens("".join(self.reasoning))
            if est >= limit:
                self.d2_seen = True
                self._loop_verdict("D2", "reasoning_backstop", f"backstop({limit})")

    def _loop_verdict(self, tid: str, rule: str, why: str) -> None:
        mode = self._task_mode(tid)
        caps = self.adapter.capabilities() if self.adapter is not None else {}
        detail = {"words": self.words.words if self.words else None,
                  "reasoning_tokens_est": im.est_tokens("".join(self.reasoning)),
                  "backend": self.backend["id"], "model": self.route.model}
        reason = why
        if mode == "act" and not caps.get("continue_final_message"):
            mode, detail["shadow_why"] = "shadow", "no_continue_final_message"
        elif mode == "act" and tid == "D1" and not (self.word_cfg or {}).get("calibrated"):
            mode, detail["shadow_why"] = "shadow", "words_uncalibrated"
        close = None
        if self.policy is not None and tid in self.policy.tasks:
            close = self.policy.tasks[tid].get("close")
        self._journal("SP1", tid, rule, "CLOSE_REASONING", reason, mode, detail=detail,
                      text=close, at=detail["reasoning_tokens_est"],
                      score=self.words.share if (self.words and tid == "D1") else None)
        if mode == "act" and close:
            self.stop = tid
            self.close_text = close
            if self.states is not None:
                self.states.monitor({"decision": "CLOSE_REASONING", "task": tid,
                                     "reason": reason})

    def _watch_args(self, idx, e: dict, piece: str) -> None:
        g = self.arg_guard
        if g is None or g.fired is not None or self.stop is not None:
            return
        if self._arg_idx != idx:
            g.feed(self._arg_n, None, None)            # the previous call's block ended
            self._arg_idx = idx
        # the guard counts tokens; a backend's frames are not tokens: one per 3.5 characters,
        # the same estimate the guard's size rule uses
        n = max(1, int(round(len(piece) / 3.5)))
        base = len(e["args"]) - len(piece)
        v = None
        for k in range(n):
            self._arg_n += 1
            v = g.feed(self._arg_n, base + int(round(len(piece) * (k + 1) / n)),
                       lambda: e["args"])
            if v is not None:
                break
        if v is None:
            return
        from server import argloop
        mode = self._task_mode("A1")
        d = dict(v.as_dict(), name=e["name"] or None, backend=self.backend["id"],
                 model=self.route.model, streamed=not self.hold_tools)
        text = argloop.stop_text(v, e["name"] or None)
        self._journal("SP1", "A1", "tool_arg_loop", "GIVE_UP", f"tool_arg_loop({v.rule})",
                      mode, detail=d, text=text, at=v.tokens)
        self.rec.argloop = dict(d, acted=mode == "act")
        if mode == "act":
            self.stop = "argloop"
            self.argloop = (v, e["name"], text, idx)
            if self.plan is not None and self.plan.session is not None:
                self.hx.pend(self.plan.session.key, "argloop", v.as_dict())
            if self.states is not None:
                self.states.monitor({"decision": "GIVE_UP", "task": "A1",
                                     "reason": f"tool_arg_loop({v.rule})"})

    # ----------------------------------------------------------------- the end of a stream
    def follow_up(self) -> dict | None:
        """After an upstream stream ended (or stopped): the next request's body, or None when
        the turn is settled. The usage of the stream that ended is already in `usages`."""
        if self.stop in ("D1", "D2") and not self.closed:
            tid = self.stop
            self.closed = True
            self.stop = None
            body = self.adapter.continue_body(self.sent or self.body_base,
                                              "<think>\n" + "".join(self.reasoning)
                                              + self.close_text)
            if body is None:
                return None
            self.follow = "close"
            self._pending_extra = (tid, None)
            self._reset_after_close()
            self.acts += 1
            self.hidden_tokens += im.est_tokens(self.close_text)
            return body
        if self.stop == "argloop":
            return None
        if self.words is not None:
            self.words.flush()
        regen = self._checks()
        if regen is not None:
            return regen
        return self._e1()

    def _reset_after_close(self) -> None:
        self.held_tools, self.held_end, self.held_usage, self.held_done = [], [], None, None

    def _e1(self) -> dict | None:
        mode = self._task_mode("E1")
        if mode == "off" or not self.tools or self.calls or "".join(self.content).strip():
            return None
        if self.fin not in (None, "stop", "eos"):
            return None
        t = self.policy.tasks.get("E1") if self.policy is not None else None
        if t is None:
            return None
        nudges = t["nudges"]
        if self.e1_tries >= int(t["tries"]):
            self._journal("SP2", "E1", "empty_turn", "GIVE_UP",
                          f"empty_turn(tries={self.e1_tries})", mode,
                          detail={"case": "B", "backend": self.backend["id"],
                                  "model": self.route.model})
            return None
        text = nudges[min(self.e1_tries, len(nudges) - 1)]
        ev = self._journal("SP2", "E1", "empty_turn", "CONTINUE_HIDDEN", "empty_turn(case=B)",
                           mode, detail={"case": "B", "act": self.e1_tries + 1,
                                         "backend": self.backend["id"],
                                         "model": self.route.model},
                           text=text, hidden=im.est_tokens(text))
        if mode != "act":
            return None
        self.e1_tries += 1
        self.acts += 1
        self.hidden_tokens += ev["hidden_tokens"]
        if self.states is not None:
            self.states.monitor({"decision": "CONTINUE_HIDDEN", "task": "E1"})
        body = dict(self.sent or self.body_base)
        body["messages"] = im.with_hidden_turn(body.get("messages") or [], text)
        body, _ = self.adapter.thinking(body, False, "off", self.upstream_model)
        self.follow = "e1"
        self._pending_extra = ("E1", ev)
        self._reset_stream_keep()
        return body

    def _reset_stream_keep(self) -> None:
        """A second request's answer replaces the first's: the held frames go (their usage is
        counted), the texts start over."""
        self.held_tools, self.held_end, self.held_usage, self.held_done = [], [], None, None
        self.calls, self.content, self.fin, self.open_call = {}, [], None, None

    def _checks(self) -> dict | None:
        """SP3: the calls of the answer through C1 to C4 (the engine's rules). With a call held
        and a check acting: a repair is applied to what goes out, a regenerate is a second
        request with the reminder as the calls' result."""
        if not self.calls or self.policy is None:
            return None
        modes = {t: self._task_mode(t) for t in ("C1", "C2", "C3", "C4")}
        if all(m == "off" for m in modes.values()):
            return None
        from server.conductor.monitor import tier0, toolcheck  # noqa: F401
        from server.conductor.monitor.observe import ToolCallEvent
        from server.conductor.monitor.policy import PASS, REGENERATE_CALL, REPAIR_CALL
        schemas = {}
        for tl in self.client_body.get("tools") or []:
            fn = (tl or {}).get("function") or {}
            if fn.get("name"):
                schemas[fn["name"]] = fn.get("parameters")
        sup = _Sup(self.messages, self.tools)
        regen = None
        for idx in sorted(self.calls, key=lambda x: (str(type(x)), x)):
            c = self.calls[idx]
            args, err = c["args"], None
            try:
                json.loads(args or "{}")
            except ValueError as exc:
                fixed = repair_json(args)
                if fixed is not None:
                    self.repaired[idx] = fixed
                    self._journal("SP3", "C4", "malformed_call", "REPAIR_CALL",
                                  "repair(json)", modes["C4"],
                                  detail={"call": {"name": c["name"], "index": idx},
                                          "fixed": ["json"], "backend": self.backend["id"],
                                          "model": self.route.model})
                    args = fixed
                else:
                    err = f"arguments are not valid JSON ({exc.msg})"
            ev = ToolCallEvent(name=c["name"] if err is None else "", arguments=args,
                               index=int(idx) if isinstance(idx, int) else 0, block="",
                               error=err, schema=schemas.get(c["name"]))
            for tid, mode in modes.items():
                if mode == "off" or tid not in self.policy.tasks:
                    continue
                task = self.policy.tasks[tid]
                try:
                    act = tier0.RULES[task.rule].fn(ev, task, sup)
                except Exception as exc:                           # noqa: BLE001
                    self.hx.log(f"[harness] {tid} failed: {type(exc).__name__}: {exc}")
                    continue
                if act is None:
                    continue
                m = mode
                if act.detail.get("mode") in ("off", "shadow") and m == "act":
                    m = "shadow" if act.detail.get("mode") == "shadow" else "off"
                if m == "off":
                    continue
                detail = dict(act.detail, call={"name": c["name"] or None, "index": idx},
                              backend=self.backend["id"], model=self.route.model)
                if act.kind == REPAIR_CALL and m == "act" and "arguments" in act.detail:
                    self.repaired[idx] = act.detail["arguments"]
                self._journal("SP3", tid, task.rule, act.kind, act.reason, m, detail=detail,
                              text=act.text or None)
                if act.kind == REGENERATE_CALL and m == "act" and regen is None and \
                        self.hold_tools:
                    regen = (tid, act)
                if act.kind not in (PASS,):
                    sup.task_acts[tid] = sup.task_acts.get(tid, 0) + 1
        if regen is None or self.regens >= REGEN_MAX:
            return None
        tid, act = regen
        self.regens += 1
        self.acts += 1
        self.hidden_tokens += im.est_tokens(act.text)
        if self.states is not None:
            self.states.monitor({"decision": "REGENERATE_CALL", "task": tid,
                                 "reason": act.reason})
        calls = [{"id": c["id"] or f"call_{i}", "type": "function",
                  "function": {"name": c["name"], "arguments": c["args"]}}
                 for i, c in sorted(self.calls.items(), key=lambda kv: str(kv[0]))]
        body = dict(self.sent or self.body_base)
        msgs = list(body.get("messages") or [])
        content = "".join(self.content) or None
        msgs.append({"role": "assistant", "content": content, "tool_calls": calls})
        for c in calls:
            msgs.append({"role": "tool", "tool_call_id": c["id"], "content": act.text})
        body["messages"] = msgs
        self.follow = "regen"
        self._pending_extra = (tid, None)
        self.repaired = {}
        self._reset_stream_keep()
        return body

    def extra_done(self, usage: dict | None, cost) -> None:
        """A second request ended: its tokens and cost go into the act's journal detail."""
        what = self._pending_extra
        self.extra.append({"what": self.follow, "usage": usage, "cost_micros": cost})
        if what is not None:
            tid = what[0]
            # one row of its own, so the Backends screen can sum what each feature cost
            self._journal("SP2" if tid == "E1" else ("SP1" if tid in ("D1", "D2") else "SP3"),
                          tid, "extra_request", "PASS", f"extra_request({self.follow})", "act",
                          detail={"backend": self.backend["id"], "model": self.route.model,
                                  "extra_request": {"follow": self.follow, "usage": usage,
                                                    "cost_micros": cost}}, quiet=True)
        self._pending_extra = None

    def follow_failed(self, why: str) -> None:
        self._journal("SP2", "harness", "extra_request", "PASS", f"follow_up_failed({why})",
                      "shadow", detail={"backend": self.backend["id"], "follow": self.follow})

    def tail(self) -> bytes:
        """What is still owed to the client once the turn is settled: the lines, the calls held
        (repaired where a check repaired them), the finish frame, the usage frame the client
        asked for (the sum of every request, when there was more than one), [DONE]."""
        out = bytearray(self._mem_flush() + self._lines())
        if self.stop == "argloop" and self.argloop is not None:
            v, name, text, idx = self.argloop
            out += self._chunk({"content": "\n\n" + text})
            out += self._chunk({}, finish="stop")
            self.done_sent = False
        else:
            if self.held_tools:
                if self.repaired:
                    out += self._calls_chunk()
                    # the finish frame of the held tool frames
                    out += self._chunk({}, finish=self.fin or "tool_calls")
                else:
                    for ev in self.held_tools:
                        out += ev
            for ev in self.held_end:
                out += ev
            if self.stop in ("D1", "D2") or (self.closed and self.fin is None):
                out += self._chunk({}, finish="stop")
        if self.held_usage is not None and self.client_usage:
            ev, obj = self.held_usage
            if len(self.usages) > 1:
                tot = self.total_usage_raw()
                if tot is not None:
                    obj = dict(obj, usage=tot)
                    if self.ident:
                        obj["id"] = self.ident["id"]
                    ev = b"data: " + json.dumps(obj, ensure_ascii=False).encode() + b"\n\n"
            out += ev
        if not self.done_sent:
            out += self.held_done or b"data: [DONE]\n\n"
            self.done_sent = True
        return bytes(out)

    def _calls_chunk(self) -> bytes:
        calls = []
        for i, (idx, c) in enumerate(sorted(self.calls.items(), key=lambda kv: str(kv[0]))):
            calls.append({"index": idx if isinstance(idx, int) else i,
                          "id": c["id"] or f"call_{i}", "type": "function",
                          "function": {"name": c["name"],
                                       "arguments": self.repaired.get(idx, c["args"])}})
        return self._chunk({"tool_calls": calls})

    def _chunk(self, delta: dict, finish=None) -> bytes:
        ident = self.ident or {"id": self.rec.request_id, "model": self.route.public,
                               "created": int(time.time()), "object": "chat.completion.chunk"}
        obj = {"id": ident["id"], "object": ident.get("object") or "chat.completion.chunk",
               "created": ident.get("created"), "model": ident.get("model"),
               "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
        return b"data: " + json.dumps(obj, ensure_ascii=False).encode() + b"\n\n"

    def _lines(self) -> bytes:
        """The state lines due now, as chunks of their own (after the first chunk only)."""
        st = self.states
        if st is None or not st.pending or self.ident is None:
            return b""
        from server.conductor import states as states_mod
        ui = (self.plan.res.doc or {}).get("ui") or {}
        caps = self.adapter.capabilities() if self.adapter is not None else {}
        fmt = "reasoning_content" if caps.get("reasoning_field") in ("reasoning_content",
                                                                     "reasoning") else "tags"
        where = states_mod.channel(ui, fmt)
        if fmt != "reasoning_content" and where == "reasoning":
            where = "content"
        out = bytearray()
        for line in st.take():
            if where == "inline":
                out += self._chunk({"tool_calls": []})
                out += self._chunk({"reasoning_content": states_mod.inline(line)})
            elif where == "reasoning":
                out += self._chunk({"reasoning_content": line})
            else:
                out += self._chunk({"content": line})
        return bytes(out)

    def total_usage(self) -> dict | None:
        """The normalised usage of every request summed; a field is None when any request did
        not report it (measured or null)."""
        if not self.usages:
            return None
        out = {}
        for k in ("prompt_tokens", "completion_tokens", "cached_tokens", "cache_write_tokens",
                  "reasoning_tokens"):
            vals = [u.get(k) if u else None for u in self.usages]
            out[k] = None if any(v is None for v in vals) else sum(vals)
        if out["prompt_tokens"] is None and out["completion_tokens"] is None:
            return None
        return out

    def total_usage_raw(self) -> dict | None:
        """The summed usage in the OpenAI shape, for the client's usage frame."""
        t = self.total_usage()
        if t is None or t["prompt_tokens"] is None or t["completion_tokens"] is None:
            return None
        u = {"prompt_tokens": t["prompt_tokens"], "completion_tokens": t["completion_tokens"],
             "total_tokens": t["prompt_tokens"] + t["completion_tokens"]}
        if t["cached_tokens"] is not None:
            u["prompt_tokens_details"] = {"cached_tokens": t["cached_tokens"]}
        if t["reasoning_tokens"] is not None:
            u["completion_tokens_details"] = {"reasoning_tokens": t["reasoning_tokens"]}
        return u

    # ----------------------------------------------------------------- the journal
    def _journal(self, sp, task, rule, decision, reason, mode, *, detail=None, text=None,
                 hidden=0, at=None, pos=None, score=None, quiet=False) -> dict:
        plan = self.plan
        ev = {"ts": time.time(), "request_id": self.rec.request_id,
              "session": plan.session.key if plan is not None and plan.session is not None
              else None, "sp": sp, "task": task, "rule": rule, "tier": 0,
              "mode": "act" if mode == "act" else "shadow", "decision": decision,
              "reason": reason, "score": score, "detail": detail or {}, "text": text,
              "hidden_tokens": int(hidden or 0), "at_tokens": at, "pos": pos,
              "client": self.rec.client_kind,
              "profile": plan.res.profile if plan is not None else None}
        if quiet:
            ev["quiet"] = True
        self.events.append(ev)
        try:
            self.hx.cond.journal.monitor_sink(ev)
        except Exception:                                          # noqa: BLE001
            pass
        if not quiet:
            self.hx.log(f"[harness] {self.rec.request_id} {sp} task={task} mode={ev['mode']} "
                        f"decision={decision} reason={reason}")
        return ev

    # ----------------------------------------------------------------- the finish
    def finish(self) -> None:
        """The turn ended (the record is complete): the router's journal row, the session's
        size, the transcript, the ledger's conductor columns."""
        rec, plan = self.rec, self.plan
        rec.conductor_acts = self.acts
        notes = sum(n.tokens for n in self.notes_live) + sum(n.tokens for n, _ in
                                                             self.notes_new)
        rec.conductor_hidden_tokens = self.hidden_tokens + notes
        rec.conductor_acted = int(bool(self.acts or (plan is not None and plan.changed)
                                       or self.effort_applied not in (None, "client",
                                                                      "unsupported",
                                                                      "enable_thinking=default",
                                                                      "thinking=default")))
        rec.out_text = ("".join(self.content), "".join(self.reasoning))
        rec.out_calls = [{"id": c["id"], "type": "function",
                          "function": {"name": c["name"], "arguments": c["args"]}}
                         for _, c in sorted(self.calls.items(), key=lambda kv: str(kv[0]))]
        rec.harness_events = self.events
        if self.states is not None:
            rec.states = self.states
        if plan is None or not plan.active:
            return
        cond = self.hx.cond
        try:
            if self.extra:
                plan.row["extra"]["extra_requests"] = self.extra
            plan.finish(rec, cond.journal)
        except Exception as exc:                                   # noqa: BLE001
            import traceback
            self.hx.log(f"[harness] journal row failed: {type(exc).__name__}: {exc} "
                        + traceback.format_exc()[-600:])
        if plan.transcript is not None:
            cond.transcripts.submit(plan, rec)
        if self.hx.writer is not None:
            try:
                self.hx.writer.seen(self)
            except Exception as exc:                               # noqa: BLE001
                self.hx.log(f"[harness] memory writer bookkeeping failed: "
                            f"{type(exc).__name__}: {exc}")
        if plan.judge:
            try:
                cond.judge.submit(plan)
            except Exception:                                      # noqa: BLE001
                pass


class BackendNarrator:
    """The `narrator` role over a backend: one state line written by a small model there, within
    the role's latency budget; the template when it is late or refused (narrator.clean)."""

    def __init__(self, hx: Harness, backend: dict, model: str, budget_ms: float):
        self.hx, self.backend, self.model, self.budget_ms = hx, backend, model, budget_ms

    def __call__(self, entry: dict):
        from server.conductor import narrator as nar
        adapter = self.hx.svc.adapter(self.backend)
        body = {"model": self.model, "stream": False, "temperature": 0,
                "max_tokens": nar.MAX_TOKENS, "messages": nar.messages(entry)}
        body, _ = adapter.thinking(body, False, "off", self.model)
        status, obj, ms = self.hx.post(self.backend, adapter, body, self.budget_ms / 1000.0)
        info = {"backend": self.backend["id"], "model": self.model, "ms": round(ms, 1)}
        if status != 200 or not isinstance(obj, dict):
            return None, dict(info, why="late" if (obj or {}).get("_error") == "timeout"
                              else "error")
        text = str((((obj.get("choices") or [{}])[0] or {}).get("message") or {})
                   .get("content") or "")
        line, why = nar.clean(text)
        return line, dict(info, **({"why": why} if why else {}))


def repair_json(text: str) -> str | None:
    """A deterministic fix of tool-call arguments that are almost JSON: trailing commas, Python
    literals, single-quoted keys and strings without a quote inside. None when it is a guess."""
    t = (text or "").strip()
    if not t:
        return "{}"
    cands = []
    s = re.sub(r",\s*([}\]])", r"\1", t)
    cands.append(s)
    s2 = re.sub(r"\bTrue\b", "true", re.sub(r"\bFalse\b", "false", re.sub(r"\bNone\b", "null",
                                                                         s)))
    cands.append(s2)
    if '"' not in s2:
        cands.append(s2.replace("'", '"'))
    if t.count("{") == t.count("}") + 1:
        cands.append(s2 + "}")
    for c in cands:
        try:
            v = json.loads(c)
        except ValueError:
            continue
        if isinstance(v, dict):
            return json.dumps(v, ensure_ascii=False)
    return None


def new_id() -> str:
    return uuid.uuid4().hex[:16]
