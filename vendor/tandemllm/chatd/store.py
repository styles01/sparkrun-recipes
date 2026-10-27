"""The chat's one SQLite file: `<data>/chat.sqlite3`, WAL, the schema of the plan note Chat 2.

    conversations   one row a chat: its settings and pane layout (JSON), the head of the branch on
                    screen, `version` (optimistic concurrency for PATCH) and `last_seq`
    messages        the message tree (`parent_id`); `parts` and `stats` are JSON; a device's
                    `client_msg_id` is unique per conversation, so a retried send is one message
    runs            one row a run: its state and the last request it sent (for Continue)
    events          the event log, `(conversation_id, seq)`; ephemeral events are never stored
    global_events   the list-level stream (`gseq`)
    queue           messages typed while a run works (the steer queue)
    tasks           the tasks a main agent gave its workers (v8; a worker's transcript is the
                    messages of its hidden `task` conversation)
    folders, attachments, and the phase-2 tables (artifacts, permission_rules, mcp_servers,
    user_tools, presets, prompts, terminals), created now so their migrations stay additive

Migrations are forward-only: `MIGRATIONS[n]` takes a file at version n-1 to n inside one
transaction, and `meta.schema_version` records where a file is. A file newer than this code is
refused (a downgrade must not half-read it). One connection, shared under a lock: the writes are
small (one event row each, about 20 a second per running chat) and SQLite serialises them anyway.

Backups: `backup()` writes `VACUUM INTO backups/chat-YYYYMMDD.sqlite3` once a day and keeps 7.
"""

from __future__ import annotations

import datetime as _dt
import glob
import json
import os
import sqlite3
import threading

SCHEMA_V1 = """
CREATE TABLE conversations (
  id TEXT PRIMARY KEY, title TEXT NOT NULL, title_auto INTEGER NOT NULL DEFAULT 1,
  created INTEGER NOT NULL, updated INTEGER NOT NULL,
  model TEXT, preset_id TEXT, pinned INTEGER NOT NULL DEFAULT 0, archived INTEGER NOT NULL DEFAULT 0,
  deleted INTEGER, folder_id TEXT, head_id TEXT,
  settings TEXT NOT NULL, ui TEXT NOT NULL,
  version INTEGER NOT NULL DEFAULT 1, last_seq INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE folders (id TEXT PRIMARY KEY, name TEXT NOT NULL, created INTEGER NOT NULL);
CREATE TABLE messages (
  id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL REFERENCES conversations(id),
  parent_id TEXT REFERENCES messages(id), role TEXT NOT NULL, created INTEGER NOT NULL,
  status TEXT NOT NULL, parts TEXT NOT NULL, attachments TEXT,
  run_id TEXT, client_msg_id TEXT, stats TEXT,
  UNIQUE (conversation_id, client_msg_id)
);
CREATE INDEX messages_parent ON messages(conversation_id, parent_id);
CREATE TABLE runs (
  id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL, user_message_id TEXT,
  assistant_message_id TEXT, state TEXT NOT NULL, step INTEGER NOT NULL DEFAULT 0,
  max_steps INTEGER, model TEXT, started INTEGER, ended INTEGER, error TEXT,
  started_by TEXT, request TEXT
);
CREATE INDEX runs_state ON runs(state);
CREATE TABLE events (
  conversation_id TEXT NOT NULL, seq INTEGER NOT NULL, ts INTEGER NOT NULL,
  run_id TEXT, type TEXT NOT NULL, data TEXT NOT NULL, PRIMARY KEY (conversation_id, seq)
) WITHOUT ROWID;
CREATE TABLE global_events (gseq INTEGER PRIMARY KEY, ts INTEGER NOT NULL, type TEXT NOT NULL,
                            data TEXT NOT NULL);
CREATE TABLE queue (
  id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL, text TEXT NOT NULL,
  client_msg_id TEXT NOT NULL, created INTEGER NOT NULL,
  UNIQUE (conversation_id, client_msg_id)
);
CREATE TABLE attachments (id TEXT PRIMARY KEY, conversation_id TEXT, name TEXT, kind TEXT,
                          mime TEXT, bytes INTEGER, sha256 TEXT, tokens INTEGER, created INTEGER);
CREATE TABLE artifacts (id TEXT, version INTEGER, conversation_id TEXT, message_id TEXT, title TEXT,
                        mime TEXT, sha256 TEXT, PRIMARY KEY (id, version));
CREATE TABLE permission_rules (id TEXT PRIMARY KEY, tool TEXT, match TEXT, decision TEXT,
                               conversation_id TEXT, created INTEGER);
CREATE TABLE mcp_servers (id TEXT PRIMARY KEY, name TEXT, transport TEXT, command TEXT, url TEXT,
                          env_enc BLOB, enabled_global INTEGER, tool_allow TEXT, tool_deny TEXT);
CREATE TABLE user_tools (id TEXT PRIMARY KEY, name TEXT, kind TEXT, source TEXT, spec TEXT,
                         created INTEGER);
CREATE TABLE presets (id TEXT PRIMARY KEY, name TEXT, model TEXT, system_prompt TEXT, settings TEXT);
CREATE TABLE prompts (id TEXT PRIMARY KEY, command TEXT UNIQUE, title TEXT, template TEXT);
CREATE TABLE terminals (id TEXT PRIMARY KEY, conversation_id TEXT, title TEXT, owner TEXT,
                        created INTEGER, closed INTEGER);
"""

# v2 (phase 2): compactions, the global tool switches, and each sandbox's last use
SCHEMA_V2 = """
CREATE TABLE compactions (
  id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL, boundary_id TEXT NOT NULL,
  summary TEXT NOT NULL, before_tokens INTEGER, after_tokens INTEGER, created INTEGER NOT NULL,
  duration_ms INTEGER, run_id TEXT
);
CREATE INDEX compactions_conv ON compactions(conversation_id, created);
CREATE TABLE tool_prefs (tool TEXT PRIMARY KEY, enabled INTEGER, policy TEXT);
CREATE TABLE sandboxes (conversation_id TEXT PRIMARY KEY, created INTEGER, last_used INTEGER,
                        removed INTEGER);
"""

# v3: MCP servers' arguments, headers (encrypted), the names of their secret keys, a timeout,
# the last tool list and the state (chatd/mcp.py)
SCHEMA_V3 = """
ALTER TABLE mcp_servers ADD COLUMN args TEXT;
ALTER TABLE mcp_servers ADD COLUMN headers_enc BLOB;
ALTER TABLE mcp_servers ADD COLUMN env_keys TEXT;
ALTER TABLE mcp_servers ADD COLUMN header_keys TEXT;
ALTER TABLE mcp_servers ADD COLUMN timeout_s REAL;
ALTER TABLE mcp_servers ADD COLUMN tools_json TEXT;
ALTER TABLE mcp_servers ADD COLUMN updated INTEGER;
ALTER TABLE mcp_servers ADD COLUMN status TEXT;
ALTER TABLE mcp_servers ADD COLUMN error TEXT
"""

# v4 (phases 3 and 4): imported Open WebUI toolkits fill v1's user_tools (`name` is the toolkit's
# slug, `kind` "owui"; chatd/owui.py), MCP servers' prompts and resources, and each chat's todo
# list (plan mode, SRV-363)
SCHEMA_V4 = """
ALTER TABLE user_tools ADD COLUMN title TEXT;
ALTER TABLE user_tools ADD COLUMN valves_enc BLOB;
ALTER TABLE user_tools ADD COLUMN enabled INTEGER NOT NULL DEFAULT 1;
ALTER TABLE user_tools ADD COLUMN updated INTEGER;
ALTER TABLE mcp_servers ADD COLUMN prompts_json TEXT;
ALTER TABLE mcp_servers ADD COLUMN resources_json TEXT;
CREATE UNIQUE INDEX user_tools_name ON user_tools(name);
CREATE TABLE todos (
    conversation_id TEXT PRIMARY KEY,
    items TEXT NOT NULL,
    updated INTEGER NOT NULL
)
"""

# v5 (SRV-387/SRV-391): the permission mode each run ran in (its badge in the thread)
SCHEMA_V5 = """
ALTER TABLE runs ADD COLUMN permission_mode TEXT
"""

# v6 (2026-10-13): one permission rule per tool, match, decision and scope. A bash prompt offered
# only its first part's prefix, which never covered a chained command, and every "always allow"
# saved it again (the live chat 01M4TJG422BS6CS816PJR837A6 held 11 rows of bash "cd"); the oldest
# row of each stays
SCHEMA_V6 = """
DELETE FROM permission_rules WHERE rowid NOT IN (
  SELECT MIN(rowid) FROM permission_rules
  GROUP BY tool, IFNULL(match, ''), decision, IFNULL(conversation_id, ''))
"""

# v7 (2026-10-16, VIS-203): no conductor profile in a chat or a preset. The composer's picker listed
# every profile (opencode's, Open WebUI's, A/B tests, header-only ones) and a pick went to the engine
# as X-Tandem-Profile; the Chat tab runs under chat-app, by its client kind
SCHEMA_V7 = """
UPDATE conversations SET settings = json_set(settings, '$.profile', NULL)
  WHERE json_valid(settings) AND json_extract(settings, '$.profile') IS NOT NULL;
UPDATE presets SET settings = json_set(settings, '$.profile', NULL)
  WHERE json_valid(settings) AND json_extract(settings, '$.profile') IS NOT NULL
"""

# v8 (VIS-E15, SRV-579): workers. A task the main agent gives runs on a conversation of its own
# (`kind` task, `parent_id` the chat), hidden from the list; `tasks` is one row a task: its brief,
# its state, its figures, the report chatd parsed and chatd's check (chatd/tasks.py)
SCHEMA_V8 = """
ALTER TABLE conversations ADD COLUMN kind TEXT NOT NULL DEFAULT 'chat';
ALTER TABLE conversations ADD COLUMN parent_id TEXT;
CREATE INDEX conversations_parent ON conversations(parent_id);
CREATE TABLE tasks (
  id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL, run_id TEXT, call_id TEXT,
  n INTEGER NOT NULL, attempt INTEGER NOT NULL DEFAULT 1, title TEXT NOT NULL, brief TEXT NOT NULL,
  expect TEXT NOT NULL, task_conversation_id TEXT NOT NULL, status TEXT NOT NULL,
  steps INTEGER NOT NULL DEFAULT 0, max_steps INTEGER, prompt_tokens INTEGER NOT NULL DEFAULT 0,
  completion_tokens INTEGER NOT NULL DEFAULT 0, cached_tokens INTEGER NOT NULL DEFAULT 0,
  started INTEGER, ended INTEGER, report TEXT, report_source TEXT, check_text TEXT, error TEXT
);
CREATE UNIQUE INDEX tasks_n ON tasks(conversation_id, n);
CREATE INDEX tasks_call ON tasks(conversation_id, call_id)
"""

# version n: the SQL that takes a file from n-1 to n
MIGRATIONS: dict[int, str] = {1: SCHEMA_V1, 2: SCHEMA_V2, 3: SCHEMA_V3, 4: SCHEMA_V4, 5: SCHEMA_V5,
                              6: SCHEMA_V6, 7: SCHEMA_V7, 8: SCHEMA_V8}
BACKUPS_KEPT = 7


class StoreError(RuntimeError):
    pass


class Store:
    def __init__(self, path: str, migrations: dict[int, str] | None = None):
        self.path = path
        self.migrations = dict(MIGRATIONS if migrations is None else migrations)
        if path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        with self.lock:
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.execute("PRAGMA synchronous=NORMAL")
            self.db.execute("PRAGMA busy_timeout=5000")
            self.db.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
        if path != ":memory:":
            try:
                os.chmod(path, 0o600)
            except OSError:
                pass
        self.migrated: list[int] = self.migrate()

    @property
    def schema_version(self) -> int:
        row = self.db.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
        return int(row[0]) if row else 0

    def migrate(self) -> list[int]:
        """Forward-only: every migration above the file's version, each in its own transaction.
        Returns the versions applied (empty when the file was current)."""
        target = max(self.migrations)
        with self.lock:
            have = self.schema_version
            if have > target:
                raise StoreError(f"{self.path} is at schema version {have}, newer than this chatd "
                                 f"({target}); refusing to read it")
            done = []
            for v in range(have + 1, target + 1):
                self.db.execute("BEGIN IMMEDIATE")
                try:
                    for stmt in _statements(self.migrations[v]):
                        self.db.execute(stmt)
                    self.db.execute("INSERT OR REPLACE INTO meta VALUES ('schema_version', ?)",
                                    (str(v),))
                    self.db.execute("COMMIT")
                except Exception:
                    self.db.execute("ROLLBACK")
                    raise
                done.append(v)
            return done

    # ------------------------------------------------------------------ helpers
    def one(self, sql: str, args=()) -> sqlite3.Row | None:
        with self.lock:
            return self.db.execute(sql, args).fetchone()

    def all(self, sql: str, args=()) -> list[sqlite3.Row]:
        with self.lock:
            return self.db.execute(sql, args).fetchall()

    def run(self, sql: str, args=()) -> int:
        with self.lock:
            return self.db.execute(sql, args).rowcount

    def tx(self):
        """`with store.tx():` one write transaction under the lock."""
        return _Tx(self)

    # ------------------------------------------------------------------ backups
    def backup(self, folder: str, today: str | None = None) -> str | None:
        """Today's `VACUUM INTO` copy, once a day; keeps the newest BACKUPS_KEPT. Returns the path
        written, None when today's copy exists already."""
        os.makedirs(folder, exist_ok=True)
        day = today or _dt.date.today().strftime("%Y%m%d")
        out = os.path.join(folder, f"chat-{day}.sqlite3")
        if os.path.exists(out):
            return None
        with self.lock:
            self.db.execute("VACUUM INTO ?", (out,))
        try:
            os.chmod(out, 0o600)
        except OSError:
            pass
        for old in sorted(glob.glob(os.path.join(folder, "chat-*.sqlite3")))[:-BACKUPS_KEPT]:
            os.remove(old)
        return out


class _Tx:
    def __init__(self, store: Store):
        self.s = store

    def __enter__(self):
        self.s.lock.acquire()
        try:
            self.s.db.execute("BEGIN IMMEDIATE")
        except Exception:
            # BEGIN failed: __exit__ will never run, so release the lock here or every later
            # transaction in this process deadlocks on it forever (issue #8).
            self.s.lock.release()
            raise
        return self.s.db

    def __exit__(self, kind, exc, tb):
        try:
            self.s.db.execute("COMMIT" if kind is None else "ROLLBACK")
        finally:
            self.s.lock.release()
        return False


def _statements(sql: str) -> list[str]:
    return [s.strip() for s in sql.split(";") if s.strip()]


def dumps(obj) -> str:
    """The JSON the store and the wire use: compact, UTF-8, key order as built."""
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
