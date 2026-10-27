"""The usage ledger: one SQLite row per request, kept for at least 400 days, with no text.

Prometheus on this cluster keeps 15 days and stores series, not requests; a year of "what did I
spend" needs its own store. This is it: stdlib `sqlite3`, one file, WAL, one writer thread.

WHAT A ROW IS. `server/usage.py::RequestRecord.row()` -- counts, timings, the finish reason, the
cache source, a client id. Never the prompt, the answer, a tool name or argument, a conversation
id, the `user` field, an IP address or a raw token. (The conductor's journal shares the file and
not this rule: its two tables hold the texts the conductor itself hid from the client, and are
kept 30 days and left out of the weekly backups. See SCHEMA VERSION 2 below.) `client_id` is
`k:` + the first 12 hex digits of sha256(the bearer token), or `anon`; `client_kind` is one of a
fixed set, read from the User-Agent. Refused requests (503/429) and rejected ones (400) are rows
too, with null token fields, so the request count is honest.

IT NEVER SLOWS OR FAILS A REQUEST. `submit()` is a `put_nowait` on a bounded queue (10,000); a full
queue drops the row and counts it. The writer commits in batches at most every second. A disk or
SQLite error drops that batch, is counted, and is logged at most once a minute. The drain on
SIGTERM calls `close()`, which flushes what is queued.

OFF BY DEFAULT, AND NEVER FROM THE ENVIRONMENT. The server takes the path from `--usage-ledger`
only. `ops/serve.env` sets `QSE_USAGE_LEDGER` and `ops/start.sh` turns it into the flag for the
:8000 service; the server itself does not read the variable, because `ops/gate.sh` exports all of
serve.env before it launches row3, verify_spec and the rest, each with a server of its own, and
none of those may write the operator's ledger. A test or e2e run whose path resolves to the production
file is refused outright.

SCHEMA VERSION 2 (SRV-115), migrated in place when a version-1 file opens, idempotent, every row
kept. `requests` gains the conductor's columns: `conductor_acts` (the session monitor's executed
acts), `conductor_hidden_tokens` (tokens the model saw that the client did not send),
`conductor_acted` (1 when anything the conductor did changed what the model saw or wrote: speed
views leave those requests out). Two new tables are the conductor's journal, one for the router
and the monitor alike (server/conductor/decisions.py writes them from its own queue thread):
`conductor_events`, one row per decision, and `conductor_texts`, every text the conductor
injected or wrote, once, by its hash. Their retention is the journal's own (30 days), and the
weekly backup drops both, so a copy never outlives it.

SCHEMA VERSION 3 (the proxy, P1), additive, migrated in place like v2: `requests` gains the
columns of a request relayed to another backend (`proxy/`): `backend` (its id), `backend_kind`,
`backend_model` (the id sent upstream), `public_model` (the id the client sent), `usage_source`
(`backend` when the backend reported `usage`, `none` when it did not), `cache_write_tokens`,
`cost_micros` (micro-dollars at the model's price when the row was written; null without a
price), `proxy_ms` (the proxy's own time: arrival to the upstream send, plus the delay of the
first frame), `retries`, `fallback_from` (the backend tried first, when another one answered),
`upstream_status` and `user` (the label of the proxy key the request carried). An engine row has
them null, except `user`. A relayed row has null `prompt_ms`, `prefill_tps`, `blocks`,
`draft_tokens`, `draft_accepted` and `cache_source`: the proxy cannot see them, and a number it
cannot see is null, never an estimate.

SCHEMA VERSION 4 (the proxy, P6), additive, migrated in place: `preemptions`, `engine_finish` (the
engine's own finish reason), `corrupted` (vLLM's NaN check) and `l2_join` (`exact` or
`approximate`) on a relayed row whose backend runs TandemLLM's plugin (plugins/). The same rows
then also carry the engine's own `queue_ms`, `prompt_ms` (prefill), `decode_ms`, `decode_tps`,
`prefill_tps` and `cached_tokens`, and on SGLang `blocks` (verify passes), `draft_tokens` and
`draft_accepted`. The plugin's numbers usually arrive after the row: `amend()` sets them on the
row in place, on the writer thread, after every row queued before it.

SCHEMA VERSION 6 (VIS-60): `parent_request`, the request an internal row served. A
request whose agent starts with `tandem-` (the router's question to the engine, `tandem-conductor/1`)
is `client_kind = 'internal'`; the dashboard leaves those rows out of every view unless asked. The
migration relabels the router's question rows written before (client `other`, a model other than
TandemLLM, `max_tokens` 24, at most 24 tokens out) once, and records the count in
`meta.relabel_internal`.

SCHEMA VERSION 7 (SRV-197): `slot`, the engine's sequence slot a request ran on with
`--parallel N` (0 the main slot with the resident conversation, 1 and up a side slot); null without
parallel requests, for a side-lane job and on a relayed row.

SCHEMA VERSION 8 (SRV-233, SRV-250): `engine_cached`, the prompt tokens the engine
read from a cache, the router's hidden notes included (`cached_tokens` is the client's tokens
among them since SRV-233); `forwarded_tokens`, what the prefill forwarded, hidden tokens included;
`wait_mem_ms`, how long the parallel memory admission held the request (inside `queue_ms`).

SCHEMA VERSION 9 (SRV-241): `sampling`, how the request sampled: null for greedy (the default),
`client` when the client sent its own temperature, `regenerate` when TandemLLM sampled a regenerate
(server/regenerate.py), `loop_escape` or `card` when it sampled with the model card's settings
(server/conductor/loopescape.py, SRV-475). The dashboard's speed sample leaves the regenerate,
loop-escape and card rows out.

SCHEMA VERSION 10 (VIS-207): `system_fingerprint`, the backend configuration a relayed answer names
(OpenAI's and DeepSeek's `system_fingerprint`, the first one a response carries); null for the
engine's own rows and for a backend that sends none. A model id such as `deepseek-flash` stays
while the provider swaps the snapshot behind it; a new fingerprint in the rows shows the swap.

RETENTION. Rows older than `--usage-retention-days` (400; less is refused outside tests) are
deleted once a day at 04:00 local, then `PRAGMA incremental_vacuum`. Once an ISO week, an online
backup (the sqlite3 backup API) goes to `backup/ledger-<year>-W<week>.sqlite3` beside the file,
the newest 8 kept.

    sqlite3 ~/.qwen38-spark-engine/usage/ledger.sqlite3 \
        "select datetime(ts_ms/1000,'unixepoch','localtime'), finish_reason, prompt_tokens,
                completion_tokens, round(decode_tps,1) from requests order by id desc limit 10"
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import os
import queue
import sqlite3
import threading
import time

from server.conductor import signals

PRODUCTION = "~/.qwen38-spark-engine/usage/ledger.sqlite3"
SCHEMA_VERSION = 10
MIN_RETENTION_DAYS = 400
QUEUE_MAX = 10_000
BACKUP_KEEP = 8

COLUMNS = ("ts_ms", "request_id", "model", "client_id", "client_kind", "endpoint", "stream",
           "status", "finish_reason", "prompt_tokens", "cached_tokens", "completion_tokens",
           "reasoning_tokens", "queue_ms", "prompt_ms", "ttft_ms", "decode_ms", "total_ms",
           "decode_tps", "prefill_tps", "blocks", "draft_tokens", "draft_accepted", "tool_calls",
           "thinking", "cache_source", "max_tokens", "error_type", "engine_version", "code_sha",
           "model_served", "profile", "profile_version", "conductor_mode", "conductor_acts",
           "conductor_hidden_tokens", "conductor_acted", "lane", "backend", "backend_kind",
           "backend_model", "public_model", "usage_source", "cache_write_tokens", "cost_micros",
           "proxy_ms", "retries", "fallback_from", "upstream_status", "user", "preemptions",
           "engine_finish", "corrupted", "l2_join", "wait_own_ms", "parent_request",
           "slot", "engine_cached", "forwarded_tokens", "wait_mem_ms", "sampling",
           "system_fingerprint")

# The conductor's columns (SRV-117, then SRV-115's three of schema version 2), added in place to a
# ledger that lacks them. Additive and idempotent: an old row reads null in each.
ADDED = (("model_served", "TEXT"), ("profile", "TEXT"), ("profile_version", "INTEGER"),
         ("conductor_mode", "TEXT"), ("conductor_acts", "INTEGER"),
         ("conductor_hidden_tokens", "INTEGER"), ("conductor_acted", "INTEGER"),
         # Served in the side lane (server/preempt.py) -- `decode` or `prefill` (at a
         # safe point of another request's generation) or `self` (under the lock, none running)
         ("lane", "TEXT"),
         # schema version 3 (the proxy, P1): a request relayed to another backend
         ("backend", "TEXT"), ("backend_kind", "TEXT"), ("backend_model", "TEXT"),
         ("public_model", "TEXT"), ("usage_source", "TEXT"), ("cache_write_tokens", "INTEGER"),
         ("cost_micros", "INTEGER"), ("proxy_ms", "REAL"), ("retries", "INTEGER"),
         ("fallback_from", "TEXT"), ("upstream_status", "INTEGER"), ("user", "TEXT"),
         # schema version 4 (the proxy, P6): what a plugin inside vLLM or SGLang reported
         ("preemptions", "INTEGER"), ("engine_finish", "TEXT"), ("corrupted", "INTEGER"),
         ("l2_join", "TEXT"),
         # schema version 5 (SRV-177): ms a request waited behind TandemLLM's own
         # work (the shadow or prepare job, the conductor's question)
         ("wait_own_ms", "REAL"),
         # schema version 6 (VIS-60): the request an internal row served (the
         # router's question to the engine names the turn it routes)
         ("parent_request", "TEXT"),
         # schema version 7 (SRV-197): the sequence slot (`--parallel N`)
         ("slot", "INTEGER"),
         # schema version 8 (SRV-233, SRV-250): the engine's own cache figure and
         # forwarded tokens (hidden notes included), the memory admission's wait
         ("engine_cached", "INTEGER"), ("forwarded_tokens", "INTEGER"), ("wait_mem_ms", "REAL"),
         # schema version 9 (SRV-241): how the request sampled (null greedy, `client`,
         # `regenerate`)
         ("sampling", "TEXT"),
         # schema version 10 (VIS-207): a relayed answer's `system_fingerprint`
         ("system_fingerprint", "TEXT"))

# the columns `Ledger.amend` may set on a row already written (proxy/collect_api.py)
AMENDABLE = ("queue_ms", "prompt_ms", "decode_ms", "decode_tps", "prefill_tps", "cached_tokens",
             "preemptions", "engine_finish", "corrupted", "blocks", "draft_tokens",
             "draft_accepted", "l2_join")

# Schema version 2: the conductor's journal (server/conductor/decisions.py). `kind` is `route`
# (the task router's decision for a request: its class, effort, notes; `task` = route) or
# `monitor` (one decision of the session monitor: `task` D1, E1, C3, ...). The router's own
# fields sit beside the common ones; JSON columns hold JSON text.
CONDUCTOR_DDL = """
CREATE TABLE IF NOT EXISTS conductor_events (
  id TEXT PRIMARY KEY, ts_ms INTEGER NOT NULL, request_id TEXT, session_key TEXT,
  kind TEXT NOT NULL, task TEXT, tier INTEGER, mode TEXT, sp TEXT, rule TEXT, trigger TEXT,
  score REAL, decision TEXT, action TEXT, cap TEXT, tier1_model TEXT, question_id TEXT,
  probabilities TEXT, tier1_ms REAL, hidden_text_id TEXT, hidden_tokens INTEGER,
  tokens_at_cut INTEGER, position INTEGER, outcome TEXT, next_checker TEXT,
  policy_version INTEGER, client_kind TEXT, model TEXT, profile TEXT, profile_version INTEGER,
  router_version INTEGER, class_id TEXT, confidence REAL, source TEXT, kept TEXT, stages TEXT,
  classifier_ms REAL, effort TEXT, budget INTEGER, budget_source TEXT, thinking TEXT,
  decisions TEXT, injected TEXT, replayed TEXT, shadow TEXT, digest TEXT, label TEXT,
  label_note TEXT, labelled_at INTEGER, acted INTEGER, changed INTEGER, detail TEXT,
  extra TEXT);
CREATE INDEX IF NOT EXISTS conductor_events_ts ON conductor_events(ts_ms);
CREATE INDEX IF NOT EXISTS conductor_events_session ON conductor_events(session_key, ts_ms);
CREATE INDEX IF NOT EXISTS conductor_events_request ON conductor_events(request_id);
CREATE TABLE IF NOT EXISTS conductor_texts (
  id TEXT PRIMARY KEY, text TEXT NOT NULL, chars INTEGER NOT NULL, first_ms INTEGER NOT NULL,
  last_ms INTEGER NOT NULL);
"""
CONDUCTOR_TABLES = ("conductor_events", "conductor_texts")

# Schema v1, exactly as the design note §2.3 has it. Forward-only migrations key on
# meta.schema_version (`migrate`): v2 is the conductor's columns and journal, v3 the proxy's
# columns (and an index for the backend filter).
DDL = """
CREATE TABLE IF NOT EXISTS requests (
  id INTEGER PRIMARY KEY, ts_ms INTEGER NOT NULL, request_id TEXT NOT NULL,
  model TEXT NOT NULL, client_id TEXT NOT NULL, client_kind TEXT NOT NULL,
  endpoint TEXT NOT NULL, stream INTEGER NOT NULL, status INTEGER NOT NULL, finish_reason TEXT,
  prompt_tokens INTEGER, cached_tokens INTEGER, completion_tokens INTEGER, reasoning_tokens INTEGER,
  queue_ms REAL, prompt_ms REAL, ttft_ms REAL, decode_ms REAL, total_ms REAL,
  decode_tps REAL, prefill_tps REAL, blocks INTEGER, draft_tokens INTEGER, draft_accepted INTEGER,
  tool_calls INTEGER NOT NULL DEFAULT 0, thinking INTEGER NOT NULL DEFAULT 0,
  cache_source TEXT, max_tokens INTEGER, error_type TEXT, engine_version TEXT, code_sha TEXT);
CREATE INDEX IF NOT EXISTS requests_ts ON requests(ts_ms);
CREATE INDEX IF NOT EXISTS requests_dim_ts ON requests(model, client_id, ts_ms);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""

KINDS = ("open-webui", "openai-sdk", "opencode", "hermes", "curl", "dashboard", "chat-app",
         "other", "internal")
# VIS-60: TandemLLM's own calls to its own port (the router's question to the engine,
# `tandem-conductor/1`; `tandem-engine/...`, `tandem-proxy/...`) are `internal`: the dashboard
# leaves them out of the Usage totals, the request counter, the filters and the Requests list
# unless asked (`internal=1`), and the row names the request it served (`parent_request`)
INTERNAL_UA = "tandem-"
PARENT_HEADER = "X-Tandem-Parent"
# the router's question before VIS-60, read off the box's ledger (2026-10-06): client `other`, the
# raw model name, `max_tokens` 24, at most 24 tokens written. Schema 6 relabels these rows once
# (`relabel_internal`), the one step of a migration that touches rows: they are the server's own
_OLD_INTERNAL = ("client_kind = 'other' AND max_tokens = 24 AND model IS NOT NULL "
                 "AND model != 'TandemLLM' AND COALESCE(completion_tokens, 0) <= 24")
# D3: the Hermes agent (NousResearch hermes-agent 0.21.0) talks to a custom endpoint
# through the OpenAI Python SDK and adds no header of its own (`OpenAI/Python 2.24.0`, read off
# its venv on the Pi, 2026-10-03), so its requests are told by their shape: the tools it offers
# name its own skill and session tools, which no other client here offers (its `hermes-cli`
# toolset: skill_view, skills_list, skill_manage, session_search, delegate_task beside terminal,
# read_file, patch ...), at least two of them; or its title call's fixed instruction (S1's
# `hermes_title` side shape), which carries no tools; or its built-in identity opening the system
# message (a SOUL.md replaces it, the tools stay). Only the SDK's agent or none is re-read: a
# client that names itself keeps its kind
HERMES_TOOLS = frozenset({"skill_view", "skills_list", "skill_manage", "session_search",
                          "delegate_task"})
HERMES_TOOLS_MIN = 2
HERMES_IDENTITY = "You are Hermes Agent, built by Nous Research."


def resolve(path: str) -> str:
    return os.path.realpath(os.path.expanduser(path))


def is_production(path: str) -> bool:
    return resolve(path) == resolve(PRODUCTION)


def check_config(path: str | None, retention_days: int, *, test: bool) -> None:
    """Refuse the two configurations that must never start: a test writing the real ledger, and
    a retention shorter than the year-plus the Usage view promises."""
    if path and test and is_production(path):
        raise SystemExit(f"[ledger] REFUSING: a test run may not write the production ledger "
                         f"({resolve(PRODUCTION)}); give it a path of its own")
    if path and retention_days < MIN_RETENTION_DAYS and not test:
        raise SystemExit(f"[ledger] --usage-retention-days {retention_days} is below the "
                         f"{MIN_RETENTION_DAYS}-day minimum (a year of history plus a margin); "
                         f"shorter retention is for tests only (QSE_TEST=1)")


def client_of(headers, body: dict | None = None) -> tuple[str, str]:
    """`(client_id, client_kind)` of a request, with nothing personal in either.

    The id is a hash prefix of the bearer token -- Open WebUI's connection key, the dashboard's
    own -- so two clients are told apart without the token ever being stored; the token table
    replaces it with a name later. The kind is a fixed set read from the User-Agent: Open WebUI's
    outbound client is aiohttp's default agent (`Python/3.x aiohttp/3.x`), the only aiohttp caller
    here; a browser cannot set a User-Agent from `fetch`, so the dashboard is recognised by the
    `Referer` a same-origin fetch from `/dashboard/` carries (or an explicit `X-QSE-Client:
    dashboard`). opencode names itself first in its agent (`opencode/<version> ai-sdk/...`).
    With the request's `body`, an OpenAI SDK's or an unnamed agent's request with the Hermes
    agent's shape is `hermes` (`is_hermes`).
    """
    def get(name):
        try:
            return headers.get(name) or ""
        except Exception:                                          # noqa: BLE001
            return ""
    auth = get("Authorization").strip()
    token = auth[7:].strip() if auth[:7].lower() == "bearer " else ""
    cid = "k:" + hashlib.sha256(token.encode()).hexdigest()[:12] if token else "anon"
    ua = get("User-Agent").lower()
    referer = get("Referer")
    from_dashboard = "/dashboard/" in referer.split("?")[0] if referer else False
    if ua.startswith(INTERNAL_UA):
        kind = "internal"
    elif get("X-QSE-Client").strip().lower() == "chat-app":
        kind = "chat-app"                          # chatd, the Chat tab's agent loop (SRV-351)
    elif (get("X-QSE-Client").strip().lower() == "dashboard" or "qse-dashboard" in ua
            or from_dashboard):
        kind = "dashboard"
    elif "open-webui" in ua or "openwebui" in ua or "aiohttp" in ua:
        kind = "open-webui"
    elif ua.startswith("opencode/"):
        # opencode 1.18.32 on 2026-09-27, read off a real request: `opencode/1.18.32
        # ai-sdk/provider-utils/4.0.23 runtime/bun/1.3.14`
        kind = "opencode"
    elif ua.startswith("openai/") or "openai/python" in ua or "openai/js" in ua:
        kind = "openai-sdk"
    elif ua.startswith("curl/"):
        kind = "curl"
    else:
        kind = "other"
    if kind in ("openai-sdk", "other") and isinstance(body, dict) and is_hermes(body):
        kind = "hermes"
    return cid, kind


def is_hermes(body: dict) -> bool:
    """The Hermes agent's request shape (HERMES_TOOLS above): its tools, its title call, or its
    identity at the head of the system message."""
    names = set()
    for t in body.get("tools") or []:
        fn = t.get("function") if isinstance(t, dict) else None
        if isinstance(fn, dict) and isinstance(fn.get("name"), str):
            names.add(fn["name"])
    if len(names & HERMES_TOOLS) >= HERMES_TOOLS_MIN:
        return True
    shape = signals.side_shape(body)
    if shape is not None and shape.get("client") == "Hermes":
        return True
    msgs = body.get("messages") or []
    if msgs and isinstance(msgs[0], dict) and msgs[0].get("role") == "system":
        return signals._text(msgs[0].get("content")).lstrip().startswith(HERMES_IDENTITY)
    return False


def migrate(conn: sqlite3.Connection, now_ms: int | None = None) -> int:
    """Bring a ledger to SCHEMA_VERSION in place and return the version it had (0: new). Every
    step is idempotent and additive -- tables and columns are created when missing, no row is
    touched -- so a crash half way and a second run are both harmless."""
    conn.executescript(DDL)
    have = {r[1] for r in conn.execute("PRAGMA table_info(requests)")}
    for name, kind in ADDED:
        if name not in have:
            conn.execute(f"ALTER TABLE requests ADD COLUMN {name} {kind}")
    conn.executescript(CONDUCTOR_DDL)
    conn.execute("CREATE INDEX IF NOT EXISTS requests_backend_ts ON requests(backend, ts_ms)")
    # FX14 (2026-10-08): the dashboard's filter lists answer from these by seeks instead of reading
    # every row on each usage call (dimensions(): VIS-60's "internal only" pairs and the proxy's
    # users cost 230 ms of a 1.1 s 365-day query over 500k rows). Indexes only, no row touched
    conn.execute("CREATE INDEX IF NOT EXISTS requests_dim_kind ON requests(model, client_id, client_kind)")
    conn.execute("CREATE INDEX IF NOT EXISTS requests_user ON requests(user) WHERE user IS NOT NULL")
    got = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
    was = int(got[0]) if got else 0
    if was and was < 6 and conn.execute(
            "SELECT 1 FROM meta WHERE key='relabel_internal'").fetchone() is None:
        n = conn.execute(f"UPDATE requests SET client_kind = 'internal' WHERE {_OLD_INTERNAL}"
                         ).rowcount
        conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES ('relabel_internal', ?)",
                     (str(n),))
    if was < SCHEMA_VERSION:
        rows = [("schema_version", str(SCHEMA_VERSION))]
        if was:
            rows.append((f"migrated_v{was}_to_v{SCHEMA_VERSION}",
                         str(now_ms if now_ms is not None else int(time.time() * 1000))))
        conn.executemany("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", rows)
    conn.commit()
    return was


def next_local(hour: int, now: float) -> float:
    """The next `hour`:00 local time strictly after `now` (epoch seconds)."""
    t = _dt.datetime.fromtimestamp(now).astimezone()
    cand = t.replace(hour=hour, minute=0, second=0, microsecond=0)
    if cand.timestamp() <= now:
        cand = (t + _dt.timedelta(days=1)).replace(hour=hour, minute=0, second=0, microsecond=0)
    return cand.timestamp()


class Ledger:
    """The writer. One per server; `open()` starts its thread, `close()` flushes and stops it."""

    def __init__(self, path: str, *, retention_days: int = MIN_RETENTION_DAYS,
                 queue_max: int = QUEUE_MAX, flush_s: float = 1.0, backup_keep: int = BACKUP_KEEP,
                 clock=time.time, log=None):
        self.path = resolve(path)
        self.retention_days = int(retention_days)
        self.flush_s = float(flush_s)
        self.backup_keep = int(backup_keep)
        self.clock = clock
        self.log = log or (lambda msg: print(msg, flush=True))
        self._q: queue.Queue = queue.Queue(maxsize=int(queue_max))
        self.stats = {"rows": 0, "dropped": 0, "failed": 0, "pruned": 0, "backups": 0}
        self._stop = threading.Event()
        self._closing = threading.Event()
        self._thread: threading.Thread | None = None
        self._conn: sqlite3.Connection | None = None
        self._last_error_log = 0.0
        self._next_prune = 0.0
        self._local = threading.local()
        self.since_ms: int | None = None
        self.schema_was: int | None = None      # the version the file had when it opened

    # ----------------------------------------------------------------- lifecycle
    def open(self) -> "Ledger":
        os.makedirs(os.path.dirname(self.path), mode=0o700, exist_ok=True)
        self._conn = self._connect()
        row = self._conn.execute("SELECT value FROM meta WHERE key='created_at'").fetchone()
        if row is None:
            now_ms = int(self.clock() * 1000)
            self._conn.executemany("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
                                   [("schema_version", str(SCHEMA_VERSION)),
                                    ("created_at", str(now_ms))])
            self._conn.commit()
        oldest = self._conn.execute("SELECT MIN(ts_ms) FROM requests").fetchone()[0]
        created = int(self._conn.execute(
            "SELECT value FROM meta WHERE key='created_at'").fetchone()[0])
        self.since_ms = min(created, oldest) if oldest is not None else created
        self._next_prune = next_local(4, self.clock())
        self._thread = threading.Thread(target=self._run, name="usage-ledger", daemon=True)
        self._thread.start()
        return self

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, check_same_thread=False, timeout=5.0)
        # auto_vacuum has to be set before the first table exists to take effect
        conn.execute("PRAGMA auto_vacuum=INCREMENTAL")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        self.schema_was = migrate(conn, int(self.clock() * 1000))
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass
        return conn

    def close(self, timeout: float = 10.0) -> None:
        """Flush what is queued, then stop. Called by the SIGTERM drain."""
        self._closing.set()
        self.flush(timeout)
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)
        if self._conn is not None:
            try:
                self._conn.close()
            except sqlite3.Error:
                pass
            self._conn = None

    def flush(self, timeout: float = 10.0) -> bool:
        """Wait until every submitted row is committed or given up on."""
        end = time.monotonic() + timeout
        while self._q.unfinished_tasks and time.monotonic() < end:
            time.sleep(0.01)
        return not self._q.unfinished_tasks

    # ----------------------------------------------------------------- writing
    def submit(self, row: dict) -> bool:
        """Hand a row to the writer. Never blocks: a full queue drops it and counts it."""
        try:
            self._q.put_nowait(row)
            return True
        except queue.Full:
            self.stats["dropped"] += 1
            return False

    def amend(self, request_id: str, fields: dict, since_ms: int = 0) -> bool:
        """Set measured columns on a row already submitted (the plugins' numbers, P6). Queued
        behind the rows, so a row submitted before it is written first. Never blocks."""
        f = {k: v for k, v in fields.items() if k in AMENDABLE}
        if not f or not request_id:
            return False
        try:
            self._q.put_nowait(("amend", str(request_id), f, int(since_ms or 0)))
            return True
        except queue.Full:
            self.stats["dropped"] += 1
            return False

    @property
    def queued(self) -> int:
        return self._q.qsize()

    def _run(self) -> None:
        batch: list[dict] = []
        last = time.monotonic()
        while True:
            try:
                batch.append(self._q.get(timeout=0.25))
            except queue.Empty:
                pass
            now = time.monotonic()
            if batch and (now - last >= self.flush_s or len(batch) >= 1000 or self._stop.is_set()
                          or self._q.empty()):
                self._write(batch)
                for _ in batch:
                    self._q.task_done()
                batch = []
                last = now
            if self.clock() >= self._next_prune:
                self.maintain()
            if self._stop.is_set() and self._q.empty() and not batch:
                return

    def _write(self, batch: list) -> None:
        rows = [r for r in batch if isinstance(r, dict)]
        amends = [a for a in batch if isinstance(a, tuple)]
        try:
            if self._conn is None:
                self._conn = self._connect()
            if rows:
                self._conn.executemany(
                    f"INSERT INTO requests ({', '.join(COLUMNS)}) VALUES "
                    f"({', '.join('?' * len(COLUMNS))})",
                    [tuple(r.get(c) for c in COLUMNS) for r in rows])
            for _, rid, f, since in amends:
                cols = sorted(f)
                self._conn.execute(
                    f"UPDATE requests SET {', '.join(c + ' = ?' for c in cols)} "
                    f"WHERE ts_ms >= ? AND request_id = ?",
                    [f[c] for c in cols] + [since, rid])
            self._conn.commit()
            self.stats["rows"] += len(rows)
            self.stats["amended"] = self.stats.get("amended", 0) + len(amends)
        except (sqlite3.Error, OSError) as exc:
            self.stats["failed"] += len(batch)
            try:
                self._conn.rollback()
            except Exception:                                      # noqa: BLE001
                self._conn = None
            self._error(f"write failed, {len(batch)} rows lost: {type(exc).__name__}: {exc}")

    def _error(self, msg: str) -> None:
        now = self.clock()
        if now - self._last_error_log >= 60.0 or not self._last_error_log:
            self._last_error_log = now
            self.log(f"[ledger] {msg} ({self.stats['failed']} lost, {self.stats['dropped']} "
                     f"dropped so far; logged at most once a minute)")

    # ----------------------------------------------------------------- retention and backup
    def maintain(self) -> None:
        """The daily 04:00 pass: prune, vacuum, the weekly backup. Runs on the writer thread."""
        self._next_prune = next_local(4, self.clock())
        try:
            n = self.prune()
            if n:
                self.log(f"[ledger] pruned {n} rows older than {self.retention_days} days")
            self.backup()
        except (sqlite3.Error, OSError) as exc:
            self._error(f"maintenance failed: {type(exc).__name__}: {exc}")

    def prune(self, now: float | None = None) -> int:
        if self._conn is None:
            self._conn = self._connect()
        cutoff = int(((self.clock() if now is None else now) - self.retention_days * 86400) * 1000)
        cur = self._conn.execute("DELETE FROM requests WHERE ts_ms < ?", (cutoff,))
        self._conn.commit()
        self._conn.execute("PRAGMA incremental_vacuum")
        self._conn.commit()
        self.stats["pruned"] += cur.rowcount
        return cur.rowcount

    def backup(self, now: float | None = None) -> str | None:
        """This ISO week's online backup, unless it exists; the newest `backup_keep` kept."""
        if self._conn is None:
            self._conn = self._connect()
        y, w, _ = _dt.date.fromtimestamp(self.clock() if now is None else now).isocalendar()
        d = os.path.join(os.path.dirname(self.path), "backup")
        os.makedirs(d, mode=0o700, exist_ok=True)
        dest = os.path.join(d, f"ledger-{y}-W{w:02d}.sqlite3")
        if os.path.exists(dest):
            return None
        tmp = dest + ".part"
        out = sqlite3.connect(tmp)
        try:
            self._conn.backup(out)
            # the conductor's journal is kept 30 days; a backup kept 8 weeks must not hold it
            for t in CONDUCTOR_TABLES:
                out.execute(f"DROP TABLE IF EXISTS {t}")
            out.commit()
            out.execute("VACUUM")
        finally:
            out.close()
        os.replace(tmp, dest)
        os.chmod(dest, 0o600)
        self.stats["backups"] += 1
        old = sorted(f for f in os.listdir(d) if f.startswith("ledger-") and f.endswith(".sqlite3"))
        for f in old[:-self.backup_keep] if self.backup_keep else []:
            os.remove(os.path.join(d, f))
        return dest

    # ----------------------------------------------------------------- reading
    def reader(self) -> sqlite3.Connection:
        """A read-only connection for the calling thread (the dashboard API's handlers)."""
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, check_same_thread=False, timeout=5.0)
            conn.execute("PRAGMA query_only=1")
            self._local.conn = conn
        return conn

    def info(self) -> dict:
        """Size, age and health, for /v1/dashboard/summary and /system and for /metrics."""
        size = 0
        for suffix in ("", "-wal"):
            try:
                size += os.path.getsize(self.path + suffix)
            except OSError:
                pass
        try:
            rows, oldest = self.reader().execute(
                "SELECT COUNT(*), MIN(ts_ms) FROM requests").fetchone()
        except sqlite3.Error:
            rows, oldest = None, None
        return {"enabled": True, "path": self.path, "rows": rows, "bytes": size,
                "schema_version": SCHEMA_VERSION,
                "oldest_ms": oldest, "since_ms": self.since_ms,
                "retention_days": self.retention_days, "queue": self.queued,
                "dropped": self.stats["dropped"] + self.stats["failed"],
                "written": self.stats["rows"]}
