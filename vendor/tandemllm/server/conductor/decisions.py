"""The conductor's journal: every decision of the router and of the session monitor, what it hid
from the client, and what came of it. One journal for both (SRV-115).

Two kinds of row in one table, `conductor_events` (schema in server/ledger.py, version 2):

  * `route`: one per `TandemLLM` request the router saw, shadow and on alike -- the class, its
    confidence, which stage decided (and every stage's answer and time), the effort set and where
    the budget came from, every note with its exact text, token position and reason, the notes
    replayed, what shadow would have done, and the turn's outcome (finish, tool calls, reasoning
    tokens, time). The dashboard's "right / wrong" buttons fill `label`.
  * `monitor`: one per decision of the session monitor (server/conductor/monitor/): the task
    (D1, E1, C3, ...), its tier and mode, the rule and why, the action, the hidden text and its
    tokens and position, the outcome of the request it acted in. It arrives through the
    monitor journal's `sinks` (`monitor_sink`); the request's passes are counted, not stored.

The next request of a session fills `next_checker` of the previous request's rows with the
checker's RESULT line and numbers. That is the honesty record (the session export lists every
hidden text the model saw) and the flywheel's data (TandemLLM 5).

Storage mirrors the usage ledger, and in production IS the usage ledger's file: stdlib sqlite3,
WAL, a queue and one writer thread, so nothing here can slow or fail a request; a full queue
drops the item and counts it. Every text is stored once in `conductor_texts`, by its hash. The
live feed reads two rings (the last 500 router rows, the last 2,000 monitor rows), which also hold
the requests still running. Rows older than `retention_days` (30) go at start and nightly at
04:00, and the texts no younger row used with them. The user's text (`digest`) is kept only for
requests with tools unless the settings widen it; chats without tools keep counters (a design
decision).
"""

from __future__ import annotations

import collections
import hashlib
import json
import os
import queue
import sqlite3
import threading
import time
import uuid

from server.ledger import CONDUCTOR_DDL, next_local

JSON_COLS = ("stages", "decisions", "injected", "replayed", "shadow", "outcome", "next_checker",
             "extra", "detail", "probabilities")
COLS = ("id", "ts_ms", "request_id", "session_key", "kind", "task", "tier", "mode", "sp", "rule",
        "trigger", "score", "decision", "action", "cap", "tier1_model", "question_id",
        "probabilities", "tier1_ms", "hidden_text_id", "hidden_tokens", "tokens_at_cut",
        "position", "outcome", "next_checker", "policy_version", "client_kind", "model",
        "profile", "profile_version", "router_version", "class_id", "confidence", "source", "kept",
        "stages", "classifier_ms", "effort", "budget", "budget_source", "thinking", "decisions",
        "injected", "replayed", "shadow", "digest", "label", "label_note", "labelled_at", "acted",
        "changed", "detail", "extra")
# what a monitor row is, as the API gives it (contract decisions.schema.json#/$defs/event)
EVENT_KEYS = ("id", "ts_ms", "request_id", "session_key", "kind", "task", "tier", "mode", "sp",
              "rule", "trigger", "score", "decision", "action", "cap", "hidden_text_id",
              "hidden_tokens", "tokens_at_cut", "position", "outcome", "next_checker",
              "policy_version", "client_kind", "profile", "detail", "label", "label_note",
              "labelled_at", "text")
RING = 500
EVENTS_RING = 2000
# the router's rows (`rows(kind="router")`): its decision for a turn, and a client's side request
# it served (S1); the Overview's and the sessions' figures read the turns alone
ROUTER_KINDS = ("route", "side")
QUEUE_MAX = 5000
# what a monitor decision's journal row keeps of its detail: never a call's arguments (they are
# the user's files and commands), only their count and names
_DROP = ("arguments", "calls", "t_end")


def new_id() -> str:
    return uuid.uuid4().hex[:16]


def text_id(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _detail(d) -> dict | None:
    if not isinstance(d, dict):
        return None
    out = {k: v for k, v in d.items() if k not in _DROP}
    if isinstance(d.get("calls"), list):
        out["calls"] = [((c or {}).get("function") or {}).get("name") for c in d["calls"]]
    fins = out.get("findings")
    if isinstance(fins, list):
        # the shallow copy shares the findings list: rewriting f["detail"] in place
        # edited the caller's own event, not just the redacted copy
        out["findings"] = [dict(f, detail=_detail(f["detail"]))
                           if isinstance(f, dict) and isinstance(f.get("detail"), dict)
                           else f for f in fins]
    return out


def route_fields(row: dict) -> dict:
    """The common columns of a router row, from its own: task `route`, tier 1 when a model
    decided, the class as the decision, the action it took, the first note's text id."""
    if row.get("kind") == "side":
        # S1: a client's side request, served by its shape: no class, no note
        side = (row.get("extra") or {}).get("side") or {}
        return {"kind": "side", "task": "route", "tier": 0, "decision": side.get("shape"),
                "action": "SIDE_REQUEST" if row.get("acted") else None, "trigger": "structural",
                "score": None, "tier1_model": None, "question_id": None, "probabilities": None,
                "tier1_ms": None, "hidden_text_id": None, "hidden_tokens": 0, "position": None,
                "cap": side.get("max_tokens")}
    stages = row.get("stages") or []
    won = next((s for s in stages if not s.get("error") and not s.get("below")), None)
    inj = row.get("injected") or []
    action = ("PROMPT_TAIL_NOTE" if inj else
              "SET_THINK_BUDGET" if row.get("acted") else None)
    return {"kind": "side" if row.get("kind") == "side" else "route", "task": "route",
            "tier": 1 if row.get("source") in ("decider", "qwen") else 0,
            "decision": row.get("class_id"), "action": action, "trigger": row.get("source"),
            "score": row.get("confidence"),
            "tier1_model": (won or {}).get("model") if won else None,
            "question_id": "classify" if stages else None,
            "probabilities": (won or {}).get("probs") if won else None,
            "tier1_ms": row.get("classifier_ms") if stages else None,
            "hidden_text_id": text_id(inj[0]["text"]) if inj and inj[0].get("text") else None,
            "hidden_tokens": sum(int(n.get("tokens") or 0) for n in inj),
            "position": inj[0].get("pos") if inj else None}


def monitor_row(ev: dict) -> dict:
    """A monitor event as a journal row (`text` stays on the ring's copy only)."""
    mode, decision = ev.get("mode"), ev.get("decision")
    text = ev.get("text") or None
    return {"id": new_id(), "ts_ms": int(float(ev.get("ts") or time.time()) * 1000),
            "request_id": ev.get("request_id"), "session_key": ev.get("session"),
            "kind": "monitor", "task": ev.get("task"), "tier": ev.get("tier", 0), "mode": mode,
            "sp": ev.get("sp"), "rule": ev.get("rule") or None, "trigger": ev.get("reason"),
            "score": ev.get("score"), "decision": decision,
            "action": decision if mode == "act" and decision not in (None, "PASS") else None,
            "cap": ev.get("cap"), "hidden_text_id": text_id(text) if text else None,
            "hidden_tokens": int(ev.get("hidden_tokens") or 0),
            "tokens_at_cut": ev.get("at_tokens"), "position": ev.get("pos"),
            "policy_version": ev.get("policy_version"), "client_kind": ev.get("client") or None,
            "profile": ev.get("profile") or None, "detail": _detail(ev.get("detail")),
            "text": text}


class Journal:
    def __init__(self, path: str | None, retention_days: int = 30):
        self.path = os.path.realpath(os.path.expanduser(path)) if path else None
        self.retention_days = int(retention_days)
        self.ring: collections.OrderedDict[str, dict] = collections.OrderedDict()
        self.events_ring: collections.deque = collections.deque(maxlen=EVENTS_RING)
        self._lock = threading.Lock()
        self._q: queue.Queue = queue.Queue(QUEUE_MAX)
        self.dropped = 0
        self.errors = 0
        self.monitor_rows = 0
        self._thread = None
        self._stop = threading.Event()
        self._next_prune = 0.0
        if self.path:
            os.makedirs(os.path.dirname(self.path), mode=0o700, exist_ok=True)
            conn = self._connect()
            conn.executescript(CONDUCTOR_DDL)
            conn.commit()
            conn.close()
            self._thread = threading.Thread(target=self._run, name="conductor-journal",
                                            daemon=True)
            self._thread.start()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=5.0, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    # ------------------------------------------------------------------ the router's rows
    def open_row(self, row: dict) -> dict:
        """A decision made at request start: visible in the live feed at once."""
        row.setdefault("id", new_id())
        row.setdefault("ts_ms", int(time.time() * 1000))
        row.setdefault("kind", "route")
        row.setdefault("task", "route")
        with self._lock:
            self.ring[row["id"]] = row
            while len(self.ring) > RING:
                self.ring.popitem(last=False)
        return row

    def close_row(self, row: dict) -> None:
        """The turn ended: the outcome is in the row; write it, and every note's text."""
        row.update(route_fields(row))
        for n in row.get("injected") or ():
            if n.get("text"):
                self._submit(("text", text_id(n["text"]), n["text"], row["ts_ms"]))
        self._submit(("upsert", dict(row)))

    def update(self, rid: str, **fields) -> bool:
        """`next_checker` or a human label on an earlier row, the router's or the monitor's."""
        with self._lock:
            r = self.ring.get(rid) or next((e for e in self.events_ring if e["id"] == rid), None)
            if r is not None:
                r.update(fields)
        self._submit(("update", rid, fields))
        return True

    def update_request(self, request_id: str, **fields) -> None:
        """Fill fields that are still empty on every row of one request: its outcome when it
        ends, the checker result the next request brought. Never overwrites."""
        if not request_id:
            return
        with self._lock:
            for r in list(self.ring.values()) + list(self.events_ring):
                if r.get("request_id") == request_id:
                    for k, v in fields.items():
                        if r.get(k) is None:
                            r[k] = v
        self._submit(("fill", request_id, fields))

    # ------------------------------------------------------------------ the monitor's events
    def monitor_sink(self, ev: dict) -> None:
        """`Journal.sinks` of the session monitor: a decision becomes a row; the request's
        outcome and a later checker result fill its rows; the pass counts are not rows."""
        decision = ev.get("decision")
        if decision == "COUNTS":
            return
        if decision == "OUTCOME":
            self.update_request(ev.get("request_id"), outcome={
                "kind": ev.get("outcome"), "acts": ev.get("acts"),
                "hidden_tokens": ev.get("hidden_tokens"),
                "finish": (ev.get("detail") or {}).get("finish")})
            return
        if decision == "LABEL":
            self.update_request(ev.get("request_id"), next_checker=ev.get("next_checker"))
            return
        row = monitor_row(ev)
        ev["journal_id"] = row["id"]          # the transcript names the row (SRV-137)
        with self._lock:
            self.events_ring.append(row)
        self.monitor_rows += 1
        if row["text"]:
            self._submit(("text", row["hidden_text_id"], row["text"], row["ts_ms"]))
        self._submit(("upsert", {k: v for k, v in row.items() if k != "text"}))

    # ------------------------------------------------------------------ the runtime's own rows
    def note(self, kind: str, task: str, *, decision: str | None = None,
             detail: dict | None = None, session_key: str | None = None,
             request_id: str | None = None) -> dict:
        """A row that is neither the router's nor the monitor's: `config` (a stored document
        loaded with defaults, SRV-133) or `memory` (a transcript block's redaction counts, a
        failed write, the prune, SRV-137). Names, counts and reasons only, never a text."""
        row = {"id": new_id(), "ts_ms": int(time.time() * 1000), "kind": kind, "task": task,
               "decision": decision, "detail": detail, "session_key": session_key,
               "request_id": request_id}
        self._submit(("upsert", row))
        return row

    # ------------------------------------------------------------------ the writer
    def _submit(self, item) -> None:
        if self.path is None:
            return
        try:
            self._q.put_nowait(item)
        except queue.Full:
            self.dropped += 1

    @staticmethod
    def _enc(c: str, v):
        return json.dumps(v, ensure_ascii=False) if c in JSON_COLS and v is not None else v

    def _apply(self, conn: sqlite3.Connection, it) -> None:
        if it[0] == "upsert":
            r = it[1]
            conn.execute(f"INSERT OR REPLACE INTO conductor_events ({', '.join(COLS)}) "
                         f"VALUES ({', '.join('?' * len(COLS))})",
                         tuple(self._enc(c, r.get(c)) for c in COLS))
        elif it[0] == "text":
            _, tid, text, ts = it
            conn.execute("INSERT INTO conductor_texts (id, text, chars, first_ms, last_ms) "
                         "VALUES (?, ?, ?, ?, ?) ON CONFLICT(id) DO UPDATE SET "
                         "last_ms = MAX(last_ms, excluded.last_ms)",
                         (tid, text, len(text), ts, ts))
        else:
            _, key, fields = it
            fields = {k: v for k, v in fields.items() if k in COLS}
            if not fields:
                return
            if it[0] == "update":
                conn.execute(f"UPDATE conductor_events SET "
                             f"{', '.join(f'{k} = ?' for k in fields)} WHERE id = ?",
                             tuple(self._enc(k, v) for k, v in fields.items()) + (key,))
            else:                                    # fill: only what is still empty
                for k, v in fields.items():
                    conn.execute(f"UPDATE conductor_events SET {k} = ? WHERE request_id = ? "
                                 f"AND {k} IS NULL", (self._enc(k, v), key))

    def _run(self) -> None:
        conn = self._connect()
        while not self._stop.is_set():
            try:
                item = self._q.get(timeout=0.5)
            except queue.Empty:
                item = None
            batch = [item] if item is not None else []
            while len(batch) < 200:
                try:
                    batch.append(self._q.get_nowait())
                except queue.Empty:
                    break
            if batch:
                try:
                    for it in batch:
                        self._apply(conn, it)
                    conn.commit()
                except sqlite3.Error as exc:
                    self.errors += 1
                    print(f"[conductor] journal write failed: {exc}", flush=True)
                    try:
                        conn.rollback()
                    except sqlite3.Error:
                        pass
                except Exception as exc:                       # noqa: BLE001
                    # a bug in the shaping of one item used to take the thread down with it:
                    # every record after that hit a full queue and dropped, and nothing said
                    # the writer was gone. One bad batch is counted and the loop goes on —
                    # rolled back, so its half-applied rows can't ride the next commit in
                    self.errors += 1
                    print(f"[conductor] journal batch skipped: {type(exc).__name__}: {exc}",
                          flush=True)
                    try:
                        conn.rollback()
                    except sqlite3.Error:
                        pass
                finally:
                    for _ in batch:
                        self._q.task_done()
            if time.time() >= self._next_prune:
                # once at start, then nightly at 04:00 local, as the ledger's own pass
                self._next_prune = next_local(4, time.time())
                try:
                    self.prune(conn)
                except sqlite3.Error:
                    self.errors += 1
        conn.close()

    def prune(self, conn: sqlite3.Connection, now: float | None = None) -> int:
        """Rows older than `retention_days`, and the texts no younger row used. Nightly."""
        cut = int(((time.time() if now is None else now) - self.retention_days * 86400) * 1000)
        n = conn.execute("DELETE FROM conductor_events WHERE ts_ms < ?", (cut,)).rowcount
        conn.execute("DELETE FROM conductor_texts WHERE last_ms < ?", (cut,))
        conn.commit()
        return n

    def flush(self, timeout: float = 5.0) -> None:
        end = time.time() + timeout
        while self._q.unfinished_tasks and time.time() < end:
            time.sleep(0.02)

    def close(self) -> None:
        self.flush()
        self._stop.set()

    def import_legacy(self, path: str) -> int:
        """Copy the router's rows from its old file (`decisions.sqlite3`, table
        `route_decisions`, before the journal moved into the ledger) once; rows already here
        are kept. Returns the rows copied; the old file is left as it is."""
        if not self.path or not os.path.exists(path):
            return 0
        src = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            cols = [r[1] for r in src.execute("PRAGMA table_info(route_decisions)")]
            if not cols:
                return 0
            rows = [dict(zip(cols, r)) for r in src.execute(
                f"SELECT {', '.join(cols)} FROM route_decisions")]
        except sqlite3.Error:
            return 0
        finally:
            src.close()
        conn = self._connect()
        try:
            n = 0
            for r in rows:
                for c in JSON_COLS:
                    if isinstance(r.get(c), str):
                        try:
                            r[c] = json.loads(r[c])
                        except ValueError:
                            pass
                r.update(route_fields(r))
                for note in r.get("injected") or ():
                    if note.get("text"):
                        self._apply(conn, ("text", text_id(note["text"]), note["text"],
                                           r["ts_ms"]))
                cur = conn.execute(f"INSERT OR IGNORE INTO conductor_events ({', '.join(COLS)}) "
                                   f"VALUES ({', '.join('?' * len(COLS))})",
                                   tuple(self._enc(c, r.get(c)) for c in COLS))
                n += cur.rowcount
            conn.commit()
            return n
        finally:
            conn.close()

    # ------------------------------------------------------------------ reads
    @staticmethod
    def _decode(r: sqlite3.Row | tuple) -> dict:
        d = dict(zip(COLS, r))
        for c in JSON_COLS:
            if isinstance(d.get(c), str):
                try:
                    d[c] = json.loads(d[c])
                except ValueError:
                    pass
        return d

    @staticmethod
    def _shape(r: dict) -> dict:
        """A monitor row as the API gives it: its own keys, the text attached; a router row as
        it is."""
        if r.get("kind") != "monitor":
            return r
        return {k: r.get(k) for k in EVENT_KEYS}

    def _texts(self, rows: list[dict]) -> None:
        """Attach `text` (the hidden text) to monitor rows read from the file."""
        want = {r["hidden_text_id"] for r in rows
                if r.get("kind") == "monitor" and r.get("hidden_text_id") and "text" not in r}
        if not want or not self.path:
            return
        got: dict[str, str] = {}
        try:
            conn = self._connect()
            ids = sorted(want)
            for i in range(0, len(ids), 500):
                part = ids[i:i + 500]
                got.update(conn.execute(f"SELECT id, text FROM conductor_texts WHERE id IN "
                                        f"({', '.join('?' * len(part))})", part).fetchall())
            conn.close()
        except sqlite3.Error:
            return
        for r in rows:
            if r.get("kind") == "monitor" and "text" not in r:
                r["text"] = got.get(r.get("hidden_text_id"))

    def rows(self, *, since_ms: int = 0, limit: int = 100, session: str | None = None,
             before_ms: int | None = None, kind: str | None = "route",
             request_ids=None, sessions: list[str] | None = None,
             texts: bool = True) -> list[dict]:
        """Newest first: the rings (running and recent) merged with the file. `kind`: route (the
        router's turns), side (a client's side requests, S1), router (both), monitor, or None for
        all; `request_ids`: only these requests' rows; `sessions`: a batch of session keys (the
        sessions page reads one window for a page of them) — a page's "-" is the null group;
        `texts`: attach the hidden texts of the monitor rows (False when the caller only
        aggregates and never shows one — a week of monitor rows is the bulk of the bytes)."""
        want = set(request_ids) if request_ids is not None else None
        kinds = ROUTER_KINDS if kind == "router" else (kind,)
        with self._lock:
            ring = ([dict(r) for r in self.ring.values()
                     if kind is None or r.get("kind", "route") in kinds]) + \
                   ([dict(r) for r in self.events_ring] if kind in ("monitor", None) else [])
        ring = [r for r in ring if r["ts_ms"] >= since_ms and
                (session is None or r.get("session_key") == session) and
                (sessions is None or (r.get("session_key") or "-") in set(sessions)) and
                (before_ms is None or r["ts_ms"] < before_ms) and
                (want is None or r.get("request_id") in want)]
        seen = {r["id"] for r in ring}
        disk: list[dict] = []
        if self.path and (want is None or want):
            q = "SELECT " + ", ".join(COLS) + " FROM conductor_events WHERE ts_ms >= ?"
            args: list = [since_ms]
            if kind is not None:
                q += f" AND kind IN ({', '.join('?' * len(kinds))})"
                args += list(kinds)
            if session is not None:
                q += " AND session_key = ?"
                args.append(session)
            if sessions is not None:
                keys = [k for k in sessions if k != "-"]
                ors = []
                if keys:
                    ors.append(f"session_key IN ({', '.join('?' * len(keys))})")
                    args += keys
                if "-" in sessions:
                    ors.append("(session_key IS NULL OR session_key = '-')")
                q += " AND (" + (" OR ".join(ors) if ors else "0") + ")"
            if before_ms is not None:
                q += " AND ts_ms < ?"
                args.append(before_ms)
            if want is not None:
                ids = sorted(want)[:900]
                q += f" AND request_id IN ({', '.join('?' * len(ids))})"
                args += ids
            q += " ORDER BY ts_ms DESC LIMIT ?"
            args.append(int(limit))
            try:
                conn = self._connect()
                disk = [self._decode(r) for r in conn.execute(q, args)]
                conn.close()
            except sqlite3.Error:
                disk = []
        out = ring + [r for r in disk if r["id"] not in seen]
        out.sort(key=lambda r: r["ts_ms"], reverse=True)
        out = out[:limit]
        if texts:
            self._texts(out)
        return [self._shape(r) for r in out]

    def recent_sessions(self, *, since_ms: int = 0, limit: int = 200) -> list[str]:
        """The keys of the sessions with a row of the week, newest first — the sessions page's
        first stage: a GROUP BY of two columns, not the 300k rows a global `rows` read would
        materialise to say the same (the 2026-10-16 audit). The rings first, so a session whose
        newest row is still unwritten counts at its newest."""
        last: dict[str, int] = {}
        with self._lock:
            live = list(self.ring.values()) + list(self.events_ring)
        for r in live:
            if r["ts_ms"] >= since_ms and r.get("kind", "route") in ("route", "monitor"):
                k = r.get("session_key") or "-"
                last[k] = max(last.get(k, 0), r["ts_ms"])
        if self.path:
            try:
                conn = self._connect()
                for k, ts in conn.execute(
                        "SELECT session_key, MAX(ts_ms) FROM conductor_events "
                        "WHERE ts_ms >= ? AND kind IN ('route', 'monitor') "
                        "GROUP BY session_key", (since_ms,)):
                    k = k or "-"
                    last[k] = max(last.get(k, 0), ts)
                conn.close()
            except sqlite3.Error:
                pass
        return [k for k, _ in sorted(last.items(), key=lambda kv: (-kv[1], kv[0]))][:limit]

    def with_events(self, rows: list[dict], events: list[dict] | None = None) -> list[dict]:
        """Router rows with the monitor's events of the same request attached (`events`);
        `events`: the ones already read, else they are looked up."""
        ids = {r.get("request_id") for r in rows if r.get("request_id")}
        evs = events if events is not None else (
            self.rows(kind="monitor", request_ids=ids, limit=20000) if ids else [])
        by: dict[str, list] = collections.defaultdict(list)
        for e in sorted(evs, key=lambda x: x["ts_ms"]):
            by[e.get("request_id")].append(e)
        return [dict(r, events=by.get(r.get("request_id"), [])) for r in rows]

    def get(self, rid: str) -> dict | None:
        with self._lock:
            if rid in self.ring:
                return dict(self.ring[rid])
            for r in self.events_ring:
                if r["id"] == rid:
                    return self._shape(r)
        if not self.path:
            return None
        try:
            conn = self._connect()
            r = conn.execute("SELECT " + ", ".join(COLS) + " FROM conductor_events WHERE id = ?",
                             (rid,)).fetchone()
            conn.close()
        except sqlite3.Error:
            return None
        if not r:
            return None
        out = self._decode(r)
        self._texts([out])
        return self._shape(out)

    def info(self) -> dict:
        return {"path": self.path, "ring": len(self.ring), "events_ring": len(self.events_ring),
                "queue": self._q.qsize(), "dropped": self.dropped, "errors": self.errors,
                "writer_alive": self._thread is not None and self._thread.is_alive(),
                "monitor_rows": self.monitor_rows, "retention_days": self.retention_days,
                "table": "conductor_events"}


# ------------------------------------------------------------------ the monitor's aggregates
def family(task: str | None) -> str:
    """The task family the Overview's ribbon colours by (TandemLLM 4): thinking, end of turn,
    tool calls, session; `other` for a task the catalogue does not know."""
    from server.conductor import schema
    meta = schema.TASKS.get(task or "")
    return meta["family"] if meta else "other"


def is_act(e: dict) -> bool:
    return e.get("mode") == "act" and e.get("decision") not in (None, "PASS")


def monitor_stats(events: list[dict], *, hours: int = 24, now_ms: int | None = None) -> dict:
    """The session monitor's tiles, ribbon, last actions and shadow precision."""
    now_ms = now_ms or int(time.time() * 1000)
    since = now_ms - hours * 3600_000
    evs = [e for e in events if e["ts_ms"] >= since]
    acts = [e for e in evs if is_act(e)]
    shadow = [e for e in evs if e.get("mode") == "shadow" and e.get("decision") != "PASS"]
    by_task: dict[str, dict] = {}
    for e in evs:
        t = by_task.setdefault(e.get("task") or "-", {"task": e.get("task") or "-",
                                                      "family": family(e.get("task")),
                                                      "acts": 0, "shadow": 0, "pass": 0})
        t["acts" if is_act(e) else "pass" if e.get("decision") == "PASS" else "shadow"] += 1
    hourly = []
    for h in range(hours):
        lo = since + h * 3600_000
        inh = [e for e in acts if lo <= e["ts_ms"] < lo + 3600_000]
        hourly.append({"start_ms": lo, "acts": len(inh),
                       "by_family": dict(collections.Counter(family(e.get("task")) for e in inh)),
                       "shadow": sum(1 for e in shadow if lo <= e["ts_ms"] < lo + 3600_000)})
    prec: dict[tuple, dict] = {}
    for e in evs:
        if e.get("decision") == "PASS" or e.get("mode") not in ("shadow", "act"):
            continue
        # proxy P4: a decision on a relayed request names its backend; the engine's say null
        be = (e.get("detail") or {}).get("backend") if isinstance(e.get("detail"), dict) else None
        p = prec.setdefault((e.get("task"), e.get("mode"), be),
                            {"task": e.get("task"), "mode": e.get("mode"), "backend": be,
                             "decisions": 0, "labelled": 0, "agree": 0})
        p["decisions"] += 1
        if e.get("label") in ("right", "wrong"):
            p["labelled"] += 1
            p["agree"] += e["label"] == "right"
    for p in prec.values():
        p["precision"] = round(p["agree"] / p["labelled"], 4) if p["labelled"] else None
    return {
        "decisions": len(evs), "acts": len(acts), "shadow": len(shadow),
        "gave_up": sum(1 for e in acts if e.get("decision") == "GIVE_UP"),
        "hidden_tokens": sum(int(e.get("hidden_tokens") or 0) for e in acts),
        "by_task": sorted(by_task.values(), key=lambda t: t["task"]),
        "by_action": dict(collections.Counter(e.get("decision") for e in acts)),
        "hourly": hourly,
        "last_actions": [{k: e.get(k) for k in ("id", "ts_ms", "request_id", "session_key",
                                                "task", "decision", "trigger", "outcome",
                                                "hidden_tokens", "cap")}
                         for e in sorted(acts, key=lambda x: x["ts_ms"], reverse=True)[:20]],
        "precision": sorted(prec.values(), key=lambda p: (str(p["task"]), p["mode"],
                                                          str(p["backend"] or ""))),
    }



# ------------------------------------------------------------------ aggregates
def _pct(vals: list[float], q: float) -> float | None:
    if not vals:
        return None
    v = sorted(vals)
    i = min(len(v) - 1, max(0, int(round(q * (len(v) - 1)))))
    return round(v[i], 2)


def stats(rows: list[dict], *, hours: int = 24, now_ms: int | None = None,
          events: list[dict] | None = None) -> dict:
    """The Overview's tiles and ribbon, from the router's rows of the last `hours`, and the
    session monitor's block from its `events` (None: no monitor on this server)."""
    now_ms = now_ms or int(time.time() * 1000)
    since = now_ms - hours * 3600_000
    rows = [r for r in rows if r["ts_ms"] >= since]
    by_class: dict[str, int] = collections.Counter(r.get("class_id") or "none" for r in rows)
    by_source = collections.Counter(r.get("source") or "none" for r in rows)
    by_effort = collections.Counter(r.get("effort") or "none" for r in rows)
    stage_ms: dict[str, list[float]] = collections.defaultdict(list)
    fails: dict[str, int] = collections.Counter()
    calls: dict[str, int] = collections.Counter()
    for r in rows:
        for s in r.get("stages") or []:
            # a stage answered from the text cache asked nothing, and a stage whose role is off was
            # never asked: neither is a call, a latency or a failure (the classifier tile read
            # "0 / 0 ms" with the decider off on 2026-10-01)
            if s.get("cached") or str(s.get("error") or "").endswith("role_off"):
                continue
            calls[s["stage"]] += 1
            stage_ms[s["stage"]].append(float(s.get("ms") or 0))
            if s.get("error"):
                fails[s["error"]] += 1
    acted = [r for r in rows if r.get("acted")]
    shadow = [r for r in rows if r.get("shadow")]
    injected_tokens = sum(sum(n.get("tokens") or 0 for n in (r.get("injected") or []))
                          for r in rows)
    hourly = []
    for h in range(hours):
        lo = since + h * 3600_000
        hi = lo + 3600_000
        inh = [r for r in rows if lo <= r["ts_ms"] < hi]
        hourly.append({"start_ms": lo, "total": len(inh),
                       "by_class": dict(collections.Counter(r.get("class_id") or "none"
                                                            for r in inh)),
                       "acted": sum(1 for r in inh if r.get("acted")),
                       "shadow": sum(1 for r in inh if r.get("shadow"))})
    labelled = [r for r in rows if r.get("label") in ("right", "wrong")]
    decided = [r for r in rows if r.get("source") in ("decider", "qwen")]
    # the promotion rule's input: per profile and decision type that ran in shadow, how many of
    # its decisions a person labelled and how many of those were right
    prec: dict[tuple, dict] = {}
    for r in rows:
        for d, m in (r.get("decisions") or {}).items():
            if m != "shadow":
                continue
            be = ((r.get("extra") or {}).get("proxy") or {}).get("backend")
            p = prec.setdefault((r.get("profile"), d, be), {"profile": r.get("profile"),
                                                            "decision": d, "backend": be,
                                                            "decisions": 0, "labelled": 0,
                                                            "agree": 0})
            p["decisions"] += 1
            if r.get("label") in ("right", "wrong"):
                p["labelled"] += 1
                p["agree"] += r["label"] == "right"
    for p in prec.values():
        p["precision"] = round(p["agree"] / p["labelled"], 4) if p["labelled"] else None
    classes: dict[str, dict] = {}
    for r in rows:
        c = classes.setdefault(r.get("class_id") or "none", {"class_id": r.get("class_id") or "none",
                                                             "decisions": 0, "labelled": 0,
                                                             "agree": 0})
        c["decisions"] += 1
        if r.get("label") in ("right", "wrong"):
            c["labelled"] += 1
            c["agree"] += r["label"] == "right"
    return {
        "hours": hours, "decisions": len(rows), "acted": len(acted), "shadow": len(shadow),
        "changed": sum(1 for r in rows if r.get("changed")),
        "injections": sum(1 for r in rows if r.get("injected")),
        "injected_tokens": injected_tokens,
        "by_class": dict(by_class), "by_source": dict(by_source), "by_effort": dict(by_effort),
        "classifier": {k: {"calls": calls[k], "p50_ms": _pct(v, 0.5), "p95_ms": _pct(v, 0.95)}
                       for k, v in stage_ms.items()},
        "failures": dict(fails),
        "fail_open_rate": round(sum(1 for r in rows if r.get("kept") in
                                    ("fail_open", "no_answer")) / len(rows), 4) if rows else None,
        "model_decided": len(decided),
        "labels": {"right": sum(1 for r in labelled if r["label"] == "right"),
                   "wrong": sum(1 for r in labelled if r["label"] == "wrong")},
        "precision": sorted(prec.values(), key=lambda p: (str(p["profile"]), p["decision"],
                                                          str(p["backend"] or ""))),
        "classes": sorted(classes.values(), key=lambda c: -c["decisions"]),
        "hourly": hourly,
        # the session monitor's counts (SRV-115); None when no monitor runs on this server
        "monitor": monitor_stats(events, hours=hours, now_ms=now_ms) if events is not None
        else None,
    }


def _session(out: dict, r: dict) -> dict:
    k = r.get("session_key") or "-"
    return out.setdefault(k, {"key": k, "client_kind": r.get("client_kind"),
                              "profile": r.get("profile"), "started_ms": r["ts_ms"],
                              "turns": 0, "notes": 0, "note_tokens": 0, "classes": {},
                              "class_id": None, "last_ms": r["ts_ms"], "last_checker": None,
                              "labels": 0, "acts": 0, "gave_ups": 0, "shadow": 0,
                              "interventions": 0, "requests": set()})


def sessions(rows: list[dict], events: list[dict] = ()) -> list[dict]:
    """One line per session: client, profile, the class now, turns, notes, the monitor's acts
    and gave-ups, the interventions (every hidden text, router and monitor), last checker."""
    out: dict[str, dict] = {}
    for e in sorted(events, key=lambda x: x["ts_ms"]):
        s = _session(out, e)
        s["requests"].add(e.get("request_id"))
        s["started_ms"] = min(s["started_ms"], e["ts_ms"])
        s["last_ms"] = max(s["last_ms"], e["ts_ms"])
        s["client_kind"] = s["client_kind"] or e.get("client_kind")
        if is_act(e):
            s["acts"] += 1
            s["gave_ups"] += e.get("decision") == "GIVE_UP"
            s["interventions"] += 1
        elif e.get("mode") == "shadow" and e.get("decision") != "PASS":
            s["shadow"] += 1
        if e.get("next_checker"):
            s["last_checker"] = (e["next_checker"] or [{}])[-1].get("verdict")
    for r in sorted(rows, key=lambda x: x["ts_ms"]):
        s = _session(out, r)
        s["requests"].add(r.get("request_id"))
        s["started_ms"] = min(s["started_ms"], r["ts_ms"])
        s["client_kind"] = r.get("client_kind") or s["client_kind"]
        s["turns"] += 1
        s["last_ms"] = max(s["last_ms"], r["ts_ms"])
        s["class_id"] = r.get("class_id")
        s["profile"] = r.get("profile") or s["profile"]
        s["classes"][r.get("class_id") or "none"] = s["classes"].get(r.get("class_id") or "none",
                                                                     0) + 1
        for n in r.get("injected") or []:
            s["notes"] += 1
            s["interventions"] += 1
            s["note_tokens"] += n.get("tokens") or 0
        if r.get("next_checker"):
            s["last_checker"] = (r["next_checker"] or [{}])[-1].get("verdict")
        if r.get("label"):
            s["labels"] += 1
    for s in out.values():
        s["turns"] = max(s["turns"], len(s.pop("requests") - {None}))
    return sorted(out.values(), key=lambda s: s["last_ms"], reverse=True)


def interventions(rows: list[dict], events: list[dict] = ()) -> list[dict]:
    """Everything the conductor changed in a session, oldest first: each router note (its exact
    text, where it went, why) and each act of the monitor (a hidden close, a hidden turn, a
    regeneration with its reminder, a repair, a give-up), with what came of the request."""
    out = []
    for r in rows:
        for n in r.get("injected") or []:
            out.append({"source": "router", "id": r["id"], "ts_ms": r["ts_ms"],
                        "request_id": r.get("request_id"), "task": "route",
                        "action": "PROMPT_TAIL_NOTE", "reason": n.get("reason"),
                        "text": n.get("text"), "tokens": n.get("tokens"), "pos": n.get("pos"),
                        "hidden": True, "parts": n.get("parts"),
                        "outcome": r.get("outcome"), "next_checker": r.get("next_checker")})
    for e in events:
        if not is_act(e):
            continue
        out.append({"source": "monitor", "id": e["id"], "ts_ms": e["ts_ms"],
                    "request_id": e.get("request_id"), "task": e.get("task"),
                    "action": e.get("decision"), "reason": e.get("trigger"),
                    "text": e.get("text"), "tokens": int(e.get("hidden_tokens") or 0),
                    "pos": e.get("position"), "hidden": bool(e.get("hidden_tokens")),
                    "tokens_at_cut": e.get("tokens_at_cut"), "cap": e.get("cap"),
                    "outcome": e.get("outcome"), "next_checker": e.get("next_checker")})
    out.sort(key=lambda x: x["ts_ms"])
    return out


def export(key: str, rows: list[dict], events: list[dict] = ()) -> dict:
    """The session export as JSON: every intervention, the turn count, the header line."""
    turns = len({x.get("request_id") for x in list(rows) + list(events)} - {None}) or len(rows)
    items = interventions(rows, events)
    return {"session": key, "turns": turns, "interventions": len(items),
            "hidden_texts": sum(1 for i in items if i["hidden"]),
            "summary": f"{len(items)} intervention{'s' if len(items) != 1 else ''} in {turns} "
                       f"turn{'s' if turns != 1 else ''}",
            "injections": items}


def _outcome_words(o) -> str:
    if not isinstance(o, dict):
        return "not known (the request did not finish)"
    if "kind" in o:
        return f"{o.get('kind') or '?'} (finish {o.get('finish') or '?'})"
    return (f"{o.get('finish_reason')}, {o.get('tool_calls', 0)} tool calls, "
            f"{o.get('reasoning_tokens')} reasoning tokens")


_ACTION_WORDS = {"CLOSE_REASONING": "closed the reasoning", "CONTINUE_HIDDEN":
                 "continued the turn", "REGENERATE_CALL": "had the tool calls written again",
                 "REPAIR_CALL": "repaired a tool call", "GIVE_UP": "gave up the turn",
                 "PROMPT_TAIL_NOTE": "added a note", "SET_THINK_BUDGET": "set the thinking budget"}


def export_md(key: str, rows: list[dict], events: list[dict] = ()) -> str:
    """The honest record of a session: every hidden text the model saw, in order, and every
    other thing the conductor changed (a repair, a give-up), with what came of it."""
    rows = sorted(rows, key=lambda x: x["ts_ms"])
    exp = export(key, rows, events)
    lines = [f"# Session {key}", "",
             f"{exp['summary']}. The client's transcript does not contain the hidden texts "
             f"below; the model saw each at the position given (tokens: a router note counted "
             f"in the client's prompt, a monitor text in the whole sequence) and a router note "
             f"on every later turn of the session until a compaction.", ""]
    mon = [i for i in exp["injections"] if i["source"] == "monitor"]
    if mon:
        lines += ["## The session monitor", ""]
        for i in mon:
            t = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(i["ts_ms"] / 1000)) + "Z"
            what = _ACTION_WORDS.get(i["action"], i["action"])
            lines.append(f"### {t} · {i['task']} {what} · request {i['request_id']}")
            lines.append(f"Because {i['reason']}" + (f" (cap {i['cap']})" if i.get("cap") else "")
                         + ". Outcome: " + _outcome_words(i.get("outcome")) + "." +
                         (f" Next checker: {i['next_checker'][-1].get('verdict')}."
                          if i.get("next_checker") else ""))
            if i["hidden"]:
                lines += ["", f"Hidden text at token {i.get('pos')} ({i['tokens']} tokens):", "",
                          "```text", i.get("text") or "", "```"]
            elif i.get("text"):
                lines += ["", "Visible text:", "", "```text", i["text"], "```"]
            lines.append("")
        if rows:
            lines += ["## The router", ""]
    for r in rows:
        t = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(r["ts_ms"] / 1000)) + "Z"
        lines.append(f"## {t} · {r.get('class_id')} ({r.get('source')}, "
                     f"{(r.get('confidence') or 0):.2f}) · effort {r.get('effort')} · "
                     f"{r.get('mode')}")
        out = r.get("outcome") or {}
        lines.append(f"Outcome: {out.get('finish_reason')}, {out.get('tool_calls', 0)} tool "
                     f"calls, {out.get('reasoning_tokens')} reasoning tokens.")
        for n in r.get("injected") or []:
            lines += ["", f"Injected at token {n.get('pos')} ({n.get('tokens')} tokens), "
                          f"because {n.get('reason')}:", "", "```text", n.get("text", ""), "```"]
        if r.get("replayed"):
            rp = r["replayed"]
            lines.append(f"Replayed {rp.get('count', 0)} earlier note(s), "
                         f"{rp.get('tokens', 0)} tokens.")
        if r.get("shadow"):
            lines.append(f"Shadow (not applied): {json.dumps(r['shadow'], ensure_ascii=False)}")
        lines.append("")
    return "\n".join(lines)
