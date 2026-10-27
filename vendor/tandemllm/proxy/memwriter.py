"""The proxy's background memory writer (M3, 2026-10-05; the design: Memo "Memory for TandemLLM:
design advice", section 2, "The background write"; an open item of a live deploy and
verification).

The engine writes a session's notes with a hidden turn on the KV it holds
(server/conductor/memwriter.py, `app.memory_once`). A backend behind the proxy (vLLM, SGLang, any
OpenAI-compatible server, a cloud API) holds no conversation the proxy can continue, so here the
same turn is one more request to the backend the session used: the messages of the session's
last request (as sent: notes and all, so a prefix cache matches), its answer, and the writer's
user turn (`memwriter.WRITER` with the notes the scope already has). Its answer goes through the
engine writer's own `apply`: the same parser, store, journal row (`kind: memory_notes`, task
`writer`) and the dashboard's writer runs, under the key `px:<session>`, with `backend`, `model`,
`cost_micros` and `via: proxy` in the run.

WHEN. A session is seen at the end of each streamed request whose plan carries memory (the
`memory` group on or shadow, a proxy user). The triggers are the engine's: `compaction` (the
client's compaction request in the same session: the session's last turn before it is kept as
the record, and the turn runs `due_idle_s` after it) and `idle` (no request in the session for `idle_s`, with
`min_new_tokens` more of it since the last pass). A look every `TICK_S` seconds runs at most one
writer request at a time, and only while the backend has no client request in flight. On a local
backend a client request that arrives while the writer streams stops it (the connection is
closed, so vLLM and SGLang abort it; the session stays due). A cloud request is not stopped: its
prompt is billed once it is sent.

THE SWITCHES, per backend in its harness document (`TANDEM_BACKEND_<ID>_WRITER*`):

    writer               auto (default): on, except off on a paid cloud model (a cloud backend
                         whose model has no price, or a price above zero); off; shadow; on
    writer_usd_day       a paid model's spend cap a day (default 0.25 USD); a run whose estimate
                         (the prompt at the input price, `max_tokens` at the output price) would
                         pass it is refused, journaled `cap_cost`
    writer_runs_day      runs a day (default 48), journaled `cap_runs` past it
    writer_prompt_tokens the prompt's size at most (default 32,000 estimated tokens): the oldest
                         messages after the system prompt are dropped (`trimmed` in the run)

The effective mode is the lower of the backend's and `settings.memory.notes.writer.mode`; a paid
model with no price is refused (`no_price`): its cap could not be checked. The day's runs and
spend are read back from the ledger (client `harness:writer`) after a restart.

SHADOW: the request runs, nothing is stored; the journal counts the ops, and the blocks it would
have written go to the user's shadow log (memwriter.shadow_log), which nothing reads back.

ITS OWN NONE: `ban_tool_call` sends `tool_choice: "none"` (the session's tools stay in the body,
so the prompt's prefix is the one the backend cached); a call it writes anyway (a `<tool_call>`
in the text, a structured call) ends the stream at its function's name and the run says NONE.
`prefill` starts the answer where the backend can continue an assistant message (vLLM, SGLang).
"""

from __future__ import annotations

import collections
import json
import threading
import time
import uuid

from proxy.adapters.base import CHAT
from server.conductor import inject_messages as im
from server.conductor import memwriter as mw

KEY_PREFIX = "px:"
CLIENT_ID = "harness:writer"
TICK_S = 2.0
SESSIONS_MAX = 256
DEADLINE_S = 300.0
DEFAULTS = {"writer": "auto", "writer_usd_day": 0.25, "writer_runs_day": 48,
            "writer_prompt_tokens": 32000}
_RANK = {"off": 0, "shadow": 1, "on": 2}


def _lower(a: str, b: str) -> str:
    return a if _RANK.get(a, 0) <= _RANK.get(b, 0) else b


def backend_cfg(backend: dict) -> dict:
    """The writer's switches of a backend (its harness document over DEFAULTS); an explicit
    `writer_usd_day: null` is no spend cap."""
    h = backend.get("harness") or {}
    out = dict(DEFAULTS)
    for k in DEFAULTS:
        if k in h and (h[k] is not None or k == "writer_usd_day"):
            out[k] = h[k]
    return out


def paid(backend: dict, price: dict | None) -> bool:
    """A cloud backend's model costs money unless its price says 0 in and 0 out."""
    if not backend.get("cloud"):
        return False
    if not price:
        return True
    try:
        return not (float(price.get("input")) == 0.0 and float(price.get("output")) == 0.0)
    except (TypeError, ValueError):
        return True


def resolve(backend: dict, price: dict | None, settings_mode: str) -> tuple[str, str]:
    """(the writer's mode for this backend's model, why): the backend's switch (auto resolved),
    lowered by the settings document's writer mode."""
    b = backend_cfg(backend)["writer"]
    why = "backend"
    if b == "auto":
        b, why = ("off", "paid_cloud") if paid(backend, price) else ("on", "auto")
    mode = _lower(b, settings_mode or "off")
    if mode != b:
        why = "settings"
    return mode, why


def shadow_log_hint(memory) -> str | None:
    """Where a shadow run's would-be blocks go, as a pattern (`<user>`, `<yyyy-mm-dd>`), the home
    directory shortened to ~; None without a notes store."""
    import os
    store = getattr(memory, "store", None)
    root = getattr(store, "dir", None)
    if not root:
        return None
    p = os.path.join(root, "<user>", mw.SHADOW_DIR, "writer-<yyyy-mm-dd>.md")
    home = os.path.expanduser("~")
    return "~" + p[len(home):] if p.startswith(home + os.sep) else p


def _text_of(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(str(p.get("text") or "") for p in content if isinstance(p, dict))
    return ""


def est_tokens(msgs: list) -> int:
    n = 0
    for m in msgs or []:
        if not isinstance(m, dict):
            continue
        n += 4 + im.est_tokens(_text_of(m.get("content")))
        for c in m.get("tool_calls") or []:
            fn = (c or {}).get("function") or {}
            n += im.est_tokens(str(fn.get("name") or "") + str(fn.get("arguments") or ""))
    return n


def trim(msgs: list, cap: int) -> tuple[list, int]:
    """`msgs` within `cap` estimated tokens: the system prompt kept, the oldest messages after it
    dropped, and then whatever no longer starts at a user message (a tool result without its
    call). (messages, how many were dropped)."""
    msgs = list(msgs or [])
    head = msgs[:1] if msgs and isinstance(msgs[0], dict) and msgs[0].get("role") in (
        "system", "developer") else []
    rest = msgs[len(head):]
    dropped = 0
    while rest and est_tokens(head + rest) > cap:
        rest.pop(0)
        dropped += 1
    if dropped:
        while rest and not (isinstance(rest[0], dict) and rest[0].get("role") == "user"):
            rest.pop(0)
            dropped += 1
    return head + rest, dropped


def record_of(turn) -> tuple[list, list | None] | None:
    """What the writer continues for this turn: (messages, tools): the messages it sent (the
    notes in place, so the backend's prefix cache matches) and its answer's text (its tool calls
    left out: a session that stops at a call has no result for it)."""
    sent = getattr(turn, "sent", None) or getattr(turn, "body_base", None) or turn.client_body
    msgs = [m for m in (sent.get("messages") or []) if isinstance(m, dict)]
    tools = sent.get("tools") if isinstance(sent.get("tools"), list) and sent.get("tools") \
        else None
    answer = "".join(turn.content or [])
    if answer.strip():
        msgs = msgs + [{"role": "assistant", "content": answer}]
    if not msgs:
        return None
    return msgs, tools


class Session:
    __slots__ = ("key", "bid", "model", "public", "msgs", "tools", "est", "at", "t", "pending",
                 "user", "kind")

    def __init__(self, key: str):
        self.key = key
        self.bid = self.model = self.public = self.user = self.kind = None
        self.msgs: list = []
        self.tools = None
        self.est = 0
        self.at = 0
        self.t = time.monotonic()
        self.pending = None


class _Parent:
    def __init__(self, user):
        self.request_id = f"px-writer-{uuid.uuid4().hex[:12]}"
        self.user = user


class ProxyWriter:
    """One per harness: the sessions the proxy relayed with memory, the day's runs and spend per
    backend, and the look that runs the writer's request."""

    def __init__(self, hx, auto: bool = True):
        self.hx = hx
        self.auto = auto
        self.sessions: collections.OrderedDict[str, Session] = collections.OrderedDict()
        self.lock = threading.Lock()
        self.spend: dict = {}
        self.busy = False
        self.stats = collections.Counter()
        self.last: str | None = None
        self._thread = None
        self._stop = threading.Event()

    # ---------------------------------------------------------------- pieces
    @property
    def memory(self):
        return getattr(self.hx.cond, "memory", None)

    def cfg(self) -> dict:
        return mw.settings_of(self.memory.cfg())

    def _log(self, msg: str) -> None:
        try:
            self.hx.log(msg)
        except Exception:                                          # noqa: BLE001
            pass

    # ---------------------------------------------------------------- what the turns tell it
    def seen(self, turn) -> None:
        """The end of a relayed turn (Turn.finish)."""
        mem = self.memory
        plan = turn.plan
        if mem is None or plan is None or not plan.active:
            return
        if (turn.rec.status or 200) != 200:
            return
        if plan.compaction:
            self._compaction(turn)
            return
        ctx = getattr(plan, "memory", None)
        if not ctx or ctx.get("mode") not in ("on", "shadow") or not ctx.get("session"):
            return
        rec = record_of(turn)
        if rec is None:
            return
        key = KEY_PREFIX + ctx["session"]
        mem.writer.seen(key, ctx, ctx.get("text"))
        self._keep(key, turn, rec, ctx.get("user"))
        self._start()

    def _keep(self, key, turn, rec, user) -> None:
        msgs, tools = rec
        est = est_tokens(msgs)
        with self.lock:
            s = self.sessions.get(key)
            if s is None:
                s = self.sessions[key] = Session(key)
            self.sessions.move_to_end(key)
            # `at` only grows: what the session added since its last request (a compaction's
            # shorter history adds nothing), so `min_new_tokens` reads as on the engine
            s.at += max(0, est - s.est)
            s.est = est
            s.bid, s.model = turn.route.backend, turn.upstream_model or turn.route.model
            s.public, s.kind = turn.route.public, turn.backend.get("kind")
            if user:
                s.user = user
            s.msgs, s.tools = msgs, tools
            s.t = time.monotonic()
            while len(self.sessions) > SESSIONS_MAX:
                self.sessions.popitem(last=False)

    def _compaction(self, turn) -> None:
        """The client's compaction request (opencode: its summarization prompt and the history
        as one user message). The record is the session's last turn as it was sent, kept apart
        (`pending`) so the turns after the compaction do not replace it: it is the history the
        backend cached, as messages."""
        sess = turn.plan.session
        if sess is None:
            return
        key = KEY_PREFIX + sess.key
        mem = self.memory
        if mem.writer.state(key) is None:
            return                      # never seen with memory: nothing of it is the writer's
        with self.lock:
            s = self.sessions.get(key)
            if s is None or not s.msgs:
                return
            s.pending = (list(s.msgs), s.tools)
            s.t = time.monotonic()
        mem.writer.mark(key, "compaction")
        self._start()

    # ---------------------------------------------------------------- the look
    def _start(self) -> None:
        if not self.auto or self._thread is not None:
            return
        with self.lock:
            if self._thread is not None:
                return
            self._thread = threading.Thread(target=self._loop, name="proxy-memwriter",
                                            daemon=True)
            self._thread.start()

    def _loop(self) -> None:
        while not self._stop.wait(TICK_S):
            try:
                self.last = self.tick()
            except Exception as exc:                               # noqa: BLE001
                self._log(f"[memory] proxy writer look failed: {type(exc).__name__}: {exc}")

    def stop(self) -> None:
        self._stop.set()
        t = self._thread
        if t is not None:
            t.join(timeout=5.0)

    def tick(self) -> str:
        """One look: the first due session whose backend is idle gets its writer request.
        Returns what happened (`off`, `not_due`, `busy`, or the run's outcome)."""
        mem = self.memory
        if mem is None:
            return "off"
        cfg = self.cfg()
        if cfg["mode"] == "off":
            return "off"
        if self.busy:
            return "busy"
        now = time.monotonic()
        with self.lock:
            cands = list(self.sessions.values())
        got = "not_due"
        for s in cands:
            if mem.writer.state(s.key) is None:
                continue
            trig = mem.writer.due(s.key, s.at, now - s.t, cfg)
            if trig is None:
                continue
            if trig == "compaction" and cfg["after_compaction"] == "off":
                mem.writer.mark(s.key, "idle")
                continue
            backend = self.hx.svc.table.backend_for(s.bid)
            if backend is None or not backend.get("enabled", True):
                mem.writer.settle(s.key, s.at)
                continue
            if self.hx.svc.relay.inflight.count(s.bid) > 0:
                got = "busy"
                continue
            self.busy = True
            try:
                return self.run(s, trig, cfg, backend)
            finally:
                self.busy = False
        return got

    # ---------------------------------------------------------------- the day's spend
    def _day(self, bid: str) -> dict:
        day = time.strftime("%Y-%m-%d")
        k = (bid, day)
        with self.lock:
            d = self.spend.get(k)
        if d is not None:
            return d
        d = {"runs": 0, "cost_micros": 0}
        led = getattr(self.hx.svc, "ledger", None)
        if led is not None:
            try:
                t = time.localtime()
                midnight = int(time.mktime((t.tm_year, t.tm_mon, t.tm_mday, 0, 0, 0, 0, 0, -1))
                               * 1000)
                n, cost = led.reader().execute(
                    "SELECT COUNT(*), SUM(cost_micros) FROM requests WHERE backend = ? AND "
                    "client_id = ? AND ts_ms >= ?", (bid, CLIENT_ID, midnight)).fetchone()
                d = {"runs": int(n or 0), "cost_micros": int(cost or 0)}
            except Exception:                                      # noqa: BLE001
                pass
        with self.lock:
            self.spend = {kk: v for kk, v in self.spend.items() if kk[1] == day}
            self.spend.setdefault(k, d)
            return self.spend[k]

    def today(self, bid: str) -> dict:
        return dict(self._day(bid))

    # ---------------------------------------------------------------- one run
    def run(self, s: Session, trigger: str, cfg: dict, backend: dict) -> str:
        mem = self.memory
        svc = self.hx.svc
        bid = backend["id"]
        price = svc.price_of(bid, s.model)
        mode, why = resolve(backend, price, cfg["mode"])
        bc = backend_cfg(backend)
        info = {"via": "proxy", "backend": bid, "model": s.model, "how": why}
        if mode == "off":
            # the switch says no for this backend's model: nothing is asked and nothing journaled
            mem.writer.settle(s.key, s.at)
            if trigger == "compaction":
                s.pending = None
            self.stats["off"] += 1
            return "off"
        day = self._day(bid)
        if day["runs"] >= int(bc["writer_runs_day"]):
            return self._refuse(s, trigger, "cap_runs", dict(info, runs_day=day["runs"]))
        gate = {"answer": None, "how": "off"}
        if cfg["gate"]:
            t0 = time.perf_counter()
            ans, how = mem.writer.gate(self.hx.cond.router, s.key)
            gate = {"answer": ans, "how": how, "ms": round((time.perf_counter() - t0) * 1e3, 1)}
            if ans == "no":
                mem.writer.skipped(s.key, trigger, "gate_no", s.at, dict(info, gate=gate))
                if trigger == "compaction":
                    s.pending = None
                return "gate_no"
        msgs, tools = (s.pending if trigger == "compaction" and s.pending is not None
                       else (s.msgs, s.tools))
        text, existing = mem.writer.instructions(s.key, cfg)
        cap = int(bc["writer_prompt_tokens"]) - im.est_tokens(text) - 8
        msgs, dropped = trim(msgs, cap)
        info.update(gate=gate, existing=len(existing), record=s.at)
        if dropped:
            info["trimmed"] = dropped
        if not msgs:
            return self._refuse(s, trigger, "too_long", info)
        adapter = svc.adapter(backend)
        body, how = self.body(adapter, s.model, msgs, tools, text, cfg)
        info.update(how_body=how)
        est_prompt = est_tokens(body["messages"])
        info["prompt_est"] = est_prompt
        if paid(backend, price):
            if not price or price.get("input") is None or price.get("output") is None:
                return self._refuse(s, trigger, "no_price", info)
            est = adapter.cost({"prompt_tokens": est_prompt, "completion_tokens":
                                int(cfg["max_tokens"]), "cached_tokens": 0}, price) or 0
            capm = bc["writer_usd_day"]
            info["cost_est_micros"] = est
            if capm is not None and day["cost_micros"] + est > float(capm) * 1e6:
                return self._refuse(s, trigger, "cap_cost",
                                    dict(info, spent_micros=day["cost_micros"]))
        stop = (lambda: False) if backend.get("cloud") else \
            (lambda: svc.relay.inflight.count(bid) > 0)
        t0 = time.perf_counter()
        status, out, usage_raw, meta = self.stream(backend, adapter, body, stop)
        ms = (time.perf_counter() - t0) * 1e3
        side = self.hx.side_row(_Parent(s.user), backend, adapter, s.model, s.public or s.model,
                                "writer", {"usage": usage_raw} if usage_raw else None, status, ms)
        cost = side.get("cost_micros")
        if cost is None and info.get("cost_est_micros"):
            info["cost_estimated"] = True            # a paid model's answer without its usage
        with self.lock:
            day["runs"] += 1
            day["cost_micros"] += int(cost if cost is not None else
                                      info.get("cost_est_micros") or 0)
        u = side.get("usage") or {}
        info.update(ms=round(ms), tokens=u.get("completion_tokens"),
                    prompt_tokens=u.get("prompt_tokens"), cached_tokens=u.get("cached_tokens"),
                    cost_micros=cost, **meta)
        if meta.get("yielded"):
            self.stats["yielded"] += 1
            mem.writer.skipped(s.key, trigger, "yielded:request", s.at, info)
            self._log(f"[memory] proxy writer {bid}/{s.model} {s.key[3:19]}: {trigger} yielded "
                      f"to a request after {ms:.0f} ms")
            return "yielded"
        if status != 200:
            return self._refuse(s, trigger, f"http_{status}" if status else
                                f"error:{meta.get('error') or 'unreachable'}", info)
        run = mem.writer.apply(s.key, out, dict(cfg, mode=mode), trigger, s.at, info)
        if trigger == "compaction":
            s.pending = None
        self.stats["runs"] += 1
        said = (f"{len(run['ops'])} op(s)" + (" (shadow)" if mode != "on" else "")
                if run.get("ops") else
                "NONE" if run.get("answer") == "none" else
                f"NONE ({run.get('answer')}{', cut at ' + meta['cut'] if meta.get('cut') else ''})"
                if run.get("none") else f"no block ({run.get('answer')})")
        self._log(f"[memory] proxy writer {bid}/{s.model} {s.key[3:19]}: {trigger} {mode}: {said}"
                  f", {u.get('prompt_tokens') or est_prompt} prompt, "
                  f"{u.get('completion_tokens') or 0} out, {ms:.0f} ms"
                  + (f", {cost} micro-USD" if cost else ""))
        return "wrote"

    def _refuse(self, s: Session, trigger: str, why: str, info: dict) -> str:
        """A run that will not be asked for this record: journaled once, the session settled."""
        mem = self.memory
        self.stats[why] += 1
        mem.writer.skipped(s.key, trigger, why, s.at, info)
        mem.writer.settle(s.key, s.at)
        if trigger == "compaction":
            s.pending = None
        self._log(f"[memory] proxy writer {info.get('backend')}/{info.get('model')} "
                  f"{s.key[3:19]}: {trigger} refused: {why}")
        return why

    # ---------------------------------------------------------------- the request
    @staticmethod
    def body(adapter, model: str, msgs: list, tools, text: str, cfg: dict) -> tuple[dict, str]:
        """The writer's request: the record, the writer's user turn, thinking off, greedy,
        `max_tokens`, streamed with its usage; the session's tools kept with `tool_choice: none`
        (`ban_tool_call`), or left out; `prefill` as the assistant's start where the backend can
        continue one. (body, what was applied)."""
        body = {"model": model, "messages": list(msgs) + [{"role": "user", "content": text}],
                "stream": True, "stream_options": {"include_usage": True}, "temperature": 0,
                "max_tokens": int(cfg["max_tokens"])}
        how = []
        if tools:
            body["tools"] = tools
            if cfg.get("ban_tool_call", True):
                body["tool_choice"] = "none"
                how.append("tool_choice=none")
        body, applied = adapter.thinking(body, False, "off", model)
        how.append(applied)
        pre = str(cfg.get("prefill") or "")
        if pre:
            cont = adapter.continue_body(body, pre)
            if cont is not None:
                body = cont
                how.append("prefill")
            else:
                how.append("prefill=unsupported")
        return body, ",".join(how)

    def stream(self, backend: dict, adapter, body: dict, stop) -> tuple[int, str, dict | None,
                                                                       dict]:
        """The writer's request, streamed: (status, the answer's text, the usage object, meta:
        finish, cut, yielded, error). It stops at a tool call that is not a memory block and,
        with `stop()` true, at the next piece (the connection is closed)."""
        from proxy import http1
        svc = self.hx.svc
        meta: dict = {}
        pre = ""
        if body.get("continue_final_message"):
            pre = str((body["messages"][-1] or {}).get("content") or "")
        parts: list[str] = [pre] if pre else []
        calls: dict = {}
        usage = None
        to = backend.get("timeouts") or {}
        first = float(to.get("first_byte") or 60.0)
        idle = float(to.get("idle") or 60.0)
        t_end = time.monotonic() + DEADLINE_S
        conn = None
        try:
            key = svc.cfg.key_for(backend["id"])
            pool = svc.relay.pool(backend)
            data = json.dumps(body, ensure_ascii=False).encode()
            conn = pool.get()
            conn.send("POST", pool.path(CHAT), adapter.headers(key), data)
            status, _, _ = conn.read_head(first)
            if status != 200:
                conn.read_all(idle)
                return status, "", None, meta
            buf = b""
            done = False
            for piece in conn.body(idle):
                buf += piece.replace(b"\r\n", b"\n")
                while b"\n\n" in buf and not done:
                    ev, buf = buf.split(b"\n\n", 1)
                    for line in ev.split(b"\n"):
                        if not line.startswith(b"data:"):
                            continue
                        d = line[5:].strip()
                        if d == b"[DONE]":
                            done = True
                            break
                        try:
                            obj = json.loads(d)
                        except ValueError:
                            continue
                        if isinstance(obj.get("usage"), dict):
                            usage = obj["usage"]
                        for ch in obj.get("choices") or []:
                            dl = ch.get("delta") or {}
                            if isinstance(dl.get("content"), str):
                                parts.append(dl["content"])
                            for c in dl.get("tool_calls") or []:
                                i = c.get("index", 0)
                                cc = calls.setdefault(i, {"name": "", "args": ""})
                                fn = c.get("function") or {}
                                cc["name"] += fn.get("name") or ""
                                cc["args"] += fn.get("arguments") or ""
                            if ch.get("finish_reason"):
                                meta["finish"] = ch["finish_reason"]
                cut = mw.foreign_call("".join(parts))
                if cut is None:
                    cut = next((c["name"] or "empty" for c in calls.values()
                                if c["name"] and c["name"] != "tandem_memory"), None)
                if cut is not None:
                    meta.update(cut=cut, finish="tool_call")
                    break
                if done:
                    break
                if stop():
                    meta["yielded"] = True
                    break
                if time.monotonic() > t_end:
                    meta["finish"] = "deadline"
                    break
            text = "".join(parts)
            for c in calls.values():
                if c["name"] == "tandem_memory":
                    try:
                        args = json.loads(c["args"] or "{}")
                    except ValueError:
                        args = {}
                    if isinstance(args, dict):
                        text += ("\n<tool_call>\n<function=tandem_memory>\n" + "".join(
                            f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in args.items()
                            if isinstance(v, (str, int, float))) + "</function>\n</tool_call>")
            return 200, text, usage, meta
        except Exception as exc:                                   # noqa: BLE001
            meta["error"] = "timeout" if isinstance(exc, http1.Timeout) else type(exc).__name__
            return 0, "".join(parts), usage, meta
        finally:
            if conn is not None:
                conn.close()
