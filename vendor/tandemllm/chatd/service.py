"""Conversations, the message tree, the event log and the runs: the logic behind every route.

THE ONE RULE OF CONSISTENCY. Every change to a conversation happens under that conversation's lock
together with the event that describes it (`emit`: the next `seq`, the row in `events`, then every
subscriber). A snapshot (`snapshot()`) is read under the same lock and carries the `seq` it
includes, so a client that applies the events after that `seq` to the snapshot has exactly the
state of a client that saw every event (the reducer in dashboard/src/chat/store.ts is the same code
for both). The streaming assistant message lives in memory (`Conv.live`) while its run is on and
is written to its row once a second and at the end; the snapshot reads the memory copy.

THE TREE. An edit makes a sibling user message under the same parent; a regenerate makes a sibling
assistant message under the same user message; `head_id` names the leaf on screen and the branch
is the path from the root to it. Switching a branch moves the head to the newest leaf under the
chosen sibling. Message ids are ULIDs, so siblings sort by the order they were made.

THE QUEUE (steer). A message sent while a run works is queued (`queue.changed`); when the run ends
the first queued message becomes the next user message and starts the next run. Send now stops the
current answer (its text stays) and the queued message goes in at once.
"""

from __future__ import annotations

import email.parser
import email.policy
import hashlib
import json
import os
import secrets
import shutil
import sys
import threading
import traceback
import time

from chatd import attach as attach_mod
from chatd import compaction as comp
from chatd import context as ctx_mod
from chatd import render
from chatd import tools as tools_mod
from chatd.agent import AgentMixin
from chatd.panes import PanesMixin
from chatd.plans import PlansMixin
from chatd.preview import PreviewMixin
from chatd.skills import SkillsMixin
from chatd.tasks import TasksMixin
from chatd.terminals import TerminalsMixin
from chatd.verify import VerifyMixin
from chatd.verify import summary as verify_summary
from chatd import websearch
from chatd.websearch import WebSearchMixin
from chatd.engine import Engine, LiveTail
from chatd.sandbox import Sandboxes
from chatd.snapshots import Snapshots
from chatd.workspace import PathError, Workspace
from chatd.events import RESET_BEHIND, Fanout, Subscriber, as_json
from chatd.ids import iso, now_ms, ulid
from chatd.runs import FLUSH_S, Run
from chatd.store import Store, dumps

DEFAULT_SETTINGS = {
    "model": "", "profile": None, "effort": "auto", "thinking": True,
    "tools": [], "packages": [], "packages_blocked": [], "packages_mode": "auto", "mcp": [],
    "prompt_id": None, "policies": {}, "auto_accept_edits": False,
    # model calls a run may make before it stops for Continue: 50 ended the 2026-10-17 pagoda run
    # (15 files, one or two a call) while it was still fixing its first errors, before any check
    "max_steps": 100,
    "sandbox_network": "on", "plan_mode": False,
    # SRV-387/SRV-391: ask | auto | yolo (chatd/tools.py decide_mode); null: the default mode
    # (GET/PUT /permissions/mode), which a new chat copies when it is created
    "permission_mode": None,
    # SRV-469: null = the global web search settings; else some of provider, max_results,
    # safe_search, language, time_range (chatd/websearch.py)
    "web_search": None,
    # SRV-532: a run that built or changed a web page or app checks it in a browser before it
    # ends (chatd/verify.py)
    "verify": True,
    # VIS-E15: Agents mode (the main agent plans and gives the file work to workers, chatd/tasks.py),
    # off: today's single loop, byte for byte; a worker's model calls; the tasks one run may give
    "agents": False, "task_max_steps": 40, "task_max_per_run": 8,
}
# `max`: a provider's own top level (DeepSeek: low, high, max), offered for its models only
EFFORTS = ("auto", "low", "medium", "high", "xhigh", "max")
EMPTY_UI = {"panes": [], "active": None, "split": 0.6}
TITLE_NEW = "New chat"
MAX_ATTACHMENT_BYTES = 50 * 1024 * 1024
TRASH_KEEP_MS = 7 * 86_400_000
_STATUS_BY_OUTCOME = {"completed": "done", "cancelled": "stopped",
                      "interrupted": "interrupted", "failed": "error"}
DELTA_KEEP_MS = 7 * 86_400_000
CAPS_TTL_S = 30.0
PERSIST_S = 1.0
METRICS_S = 1.0
CONTEXT_RUN_S = 1.0
CONTEXT_IDLE_S = 10.0
SANDBOX_S = 10.0          # a running sandbox's memory and disk go out with the context this often
PEEK_TTL_S = 30.0

#: SRV-546: a prefill that reads this many tokens again names it a re-read, with the engine's reason
REREAD_WORDS_TOKENS = 8192
#: VIS-E15: the events of a task conversation that go to its chat's log, with `task_id`: the
#: worker's transcript as it streams, its prompts and the panes it opens
TASK_FORWARD = ("message.created", "part.added", "part.delta", "part.done", "tool.asking",
                "tool.started", "tool.output", "tool.done", "tool.auto_approved", "pane.open")


def prefill_words(pre: dict) -> str:
    """The Now line of a prefill, from the engine's live row: what is read again and why (the
    engine's words, `prefill.why`), what is new beside what the cache held, or the prompt."""
    total = int(pre.get("total") or 0)
    cached = int(pre.get("cached") or 0)
    new = max(0, total - cached)
    why = pre.get("why")
    if why and new >= REREAD_WORDS_TOKENS:
        return f"Re-reading {new:,} of {total:,} tokens: {why}"
    if cached:
        return f"Reading {new:,} new tokens ({cached:,} cached)"
    return f"Prefilling {total:,} tokens"


class ApiError(Exception):
    def __init__(self, status: int, kind: str, message: str, extra: dict | None = None):
        super().__init__(message)
        self.status, self.kind, self.extra = status, kind, extra or {}

    def body(self) -> dict:
        return {"error": {"type": self.kind, "message": str(self)}, **self.extra}


class Conv:
    """A conversation's live state: its lock, its subscribers, the run, the streaming message."""

    def __init__(self, cid: str, last_seq: int):
        self.id = cid
        self.lock = threading.RLock()
        self.fan = Fanout()
        self.last_seq = last_seq
        self.run: Run | None = None
        self.live: dict | None = None            # the streaming assistant message
        self.persisted_at = 0.0
        self.context: dict | None = None
        self.context_sent_at = 0.0
        self.sandbox_sent: tuple[float, dict | None] = (0.0, None)
        self.peek: tuple[float, dict | None] | None = None
        self.tok_ratio: float | None = None       # tokens per request character, learned
        self.tok_counted = 0                      # the characters of the request it was learned on
        # chatd's estimate of the smallest request the engine refused for memory (runs.py): the
        # context guard compacts under it from then on (compaction.compact_at)
        self.memory_limit: int | None = None
        # VIS-E15: a task conversation's chat and its tasks row; its events go to the chat's log
        self.parent_id: str | None = None
        self.task_id: str | None = None


class Slots:
    """At most `n` runs talk to the engine at once (QSE_CHAT_MAX_RUNS, default 2)."""

    def __init__(self, n: int):
        self.n = max(1, n)
        self.holders: set = set()
        self.cv = threading.Condition()

    def acquire(self, run, timeout: float) -> bool:
        run = getattr(run, "slot_owner", run)          # a worker runs in its parent's slot
        with self.cv:
            if run in self.holders:
                return True
            if len(self.holders) >= self.n:
                self.cv.wait(timeout)
            if len(self.holders) >= self.n:
                return False
            self.holders.add(run)
            return True

    def release_if(self, run) -> None:
        run = getattr(run, "slot_owner", run)
        with self.cv:
            if run in self.holders:
                self.holders.discard(run)
                self.cv.notify_all()


def device_label(ua: str | None) -> str | None:
    """A browser and its system in words ("Chrome on macOS"): the Now line's "started on ...".
    Nothing that identifies a person."""
    if not ua:
        return None
    u = ua.lower()
    browser = ("Edge" if "edg/" in u else "Firefox" if "firefox" in u else "Chrome"
               if "chrome" in u or "crios" in u else "Safari" if "safari" in u else None)
    system = ("iPhone" if "iphone" in u else "iPad" if "ipad" in u else "Android"
              if "android" in u else "macOS" if "mac os" in u else "Windows" if "windows" in u
              else "Linux" if "linux" in u else None)
    if browser and system:
        return f"{browser} on {system}"
    return browser or system or "another device"


class Service(AgentMixin, PanesMixin, PreviewMixin, TerminalsMixin, SkillsMixin, PlansMixin,
              WebSearchMixin, VerifyMixin, TasksMixin):
    def __init__(self, data_dir: str, engine_url: str, *, engine_key: str | None = None,
                 max_runs: int = 2, retry_window_s: float = 600.0, titles: bool = True,
                 log=None, store: Store | None = None, engine: Engine | None = None,
                 sandboxes: Sandboxes | None = None):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.blobs = os.path.join(data_dir, "blobs")
        self.store = store or Store(os.path.join(data_dir, "chat.sqlite3"))
        self.engine = engine or Engine(engine_url, engine_key)
        self.retry_window_s = retry_window_s
        self.titles = titles
        self._log = log or (lambda line: print(line, file=sys.stderr, flush=True))
        self.slots = Slots(max_runs)
        self.convs: dict[str, Conv] = {}
        self.convs_lock = threading.Lock()
        self.gfan = Fanout()
        self.glock = threading.Lock()
        row = self.store.one("SELECT MAX(gseq) FROM global_events")
        self.gseq = int(row[0] or 0) if row else 0
        self._caps: tuple[float, dict] | None = None
        self._facts: tuple[float, dict] = (0.0, {})
        self._refresh_lock = threading.Lock()
        self._first_lock = threading.Lock()
        self._refreshing = False
        self._no_peek_until = 0.0
        self.live = LiveTail(self.engine, self._runs_on)
        self.stopping = threading.Event()
        self.sandboxes = sandboxes or Sandboxes(data_dir, log=self.log,
                                                on_change=self._sandbox_changed,
                                                network_mode=self._network_mode,
                                                release=self.engine.release_memory)
        self.snaps = Snapshots(os.path.join(data_dir, "snapshots"))
        self._library: tuple[float, bool] | None = None
        from chatd.mcp import Mcp
        from chatd.owui import Owui
        self.mcp = Mcp(self, os.path.dirname(os.path.abspath(data_dir)))
        self.owui = Owui(self)
        self.interrupted_at_start = self._interrupt_leftovers()
        self.ticker = threading.Thread(target=self._tick, name="chatd-ticker", daemon=True)
        self.sweeper = threading.Thread(target=self._sweep, name="chatd-sweeper", daemon=True)

    def start(self) -> None:
        self.ticker.start()
        self.sweeper.start()
        n = self.sandboxes.adopt()
        if n:
            self.log(f"[sandbox] {n} running sandbox(es) adopted")
        self.reap_sandboxes()
        threading.Thread(target=self.sandboxes.guard, name="chatd-sandbox-guard",
                         daemon=True).start()
        if self.mcp is not None:
            self.mcp.start()

    def log(self, line: str) -> None:
        self._log(line)

    # ================================================================== sandbox glue
    def reap_sandboxes(self) -> list[str]:
        """At start: the containers adopt() found whose chat is gone (its container and folders
        go) or in the trash (its container goes; the files wait for the purge or a restore)."""
        out = []
        for cid in list(self.sandboxes.boxes):
            row = self.store.one("SELECT deleted FROM conversations WHERE id = ?", (cid,))
            if row is not None and row["deleted"] is None:
                continue
            gone = row is None
            self.sandboxes.remove(cid, files=gone)
            with self.sandboxes.lock:
                self.sandboxes.boxes.pop(cid, None)
            if gone:
                shutil.rmtree(os.path.join(self.data_dir, "snapshots", cid), ignore_errors=True)
            self.log(f"[sandbox] {cid}: removed the container of a chat that is "
                     f"{'gone (and its folders)' if gone else 'in the trash'}")
            out.append(cid)
        return out

    def workspace(self, cid: str) -> Workspace:
        """The chat's /workspace; a task conversation's is its chat's (one sandbox, one tree)."""
        ws = Workspace(self.sandboxes.paths(self.root_cid(cid))[0])
        ws.ensure()
        return ws

    def root_cid(self, cid: str) -> str:
        """The chat a task conversation belongs to; any other conversation is its own (VIS-E15).
        A row's parent never changes, so the answer is kept."""
        roots = self.__dict__.setdefault("_roots", {})
        if cid not in roots:
            r = self.store.one("SELECT parent_id FROM conversations WHERE id = ?", (cid,))
            if r is None:
                return cid
            roots[cid] = r["parent_id"] or cid
        return roots[cid]

    def settings_of(self, cid: str) -> dict:
        """A conversation's settings with the defaults; a task conversation has none of its own:
        its chat's (the model, the permission mode, the tools, the sandbox network)."""
        row = self._row(self.root_cid(cid))
        return {**DEFAULT_SETTINGS, **json.loads(row["settings"])}

    def _network_mode(self, cid: str) -> str:
        r = self.store.one("SELECT settings FROM conversations WHERE id = ?", (self.root_cid(cid),))
        s = json.loads(r["settings"]) if r else {}
        return "off" if s.get("sandbox_network") == "off" else "on"

    def _sandbox_changed(self, cid: str, status: dict) -> None:
        try:
            conv = self.conv(cid)
        except ApiError:
            return
        self.emit(conv, {"type": "sandbox.changed", "sandbox": status})

    def library_available(self) -> bool:
        """Whether the library's tools go into requests. A check that gets no answer (a busy
        engine: the 3 s timeout, a reset) keeps the last answer (SRV-663): the tool list is the
        prompt's head, and a flap dropped library_search and library_read from one request of a
        196,472-token run on Ornith, which read it all again twice (2026-10-21 05:29, 05:42).
        Only an answer of the engine's (any status) changes it."""
        if self._library is None or time.monotonic() - self._library[0] > 300:
            st, _ = self.engine.get_json("/v1/library", timeout=3.0)
            if st == 0 and self._library is not None:
                self._library = (time.monotonic(), self._library[1])
            else:
                self._library = (time.monotonic(), st == 200)
        return self._library[1]

    def store_blob(self, cid: str | None, name: str, mime: str, data: bytes) -> dict:
        """Bytes into the blob store and an attachment row; the Attachment."""
        sha = hashlib.sha256(data).hexdigest()
        path = os.path.join(self.blobs, sha[:2], sha)
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = path + f".tmp{secrets.token_hex(4)}"
            with open(tmp, "wb") as fh:
                fh.write(data)
            os.replace(tmp, path)
        kind = ("image" if mime.startswith("image/") else "pdf" if mime == "application/pdf"
                else "text" if mime.startswith("text/") or name.lower().endswith(
                    (".md", ".txt", ".py", ".ts", ".js", ".json", ".yaml", ".yml", ".toml",
                     ".csv")) else "file")
        att_id = ulid()
        tokens = ctx_mod.estimate_tokens(data.decode("utf-8", "replace")) if kind == "text" else None
        self.store.run("INSERT INTO attachments VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                       (att_id, cid, name, kind, mime, len(data), sha, tokens, now_ms()))
        return _attachment(att_id, name, kind, mime, len(data), tokens)

    # ================================================================== conversations: state
    def conv(self, cid: str) -> Conv:
        with self.convs_lock:
            c = self.convs.get(cid)
            if c is not None:
                return c
            row = self.store.one("SELECT last_seq FROM conversations WHERE id = ? AND deleted IS "
                                 "NULL", (cid,))
            if row is None:
                raise ApiError(404, "not_found", "no such conversation")
            c = self.convs[cid] = Conv(cid, int(row[0]))
            if self.root_cid(cid) != cid:
                c.parent_id = self.root_cid(cid)
                t = self.store.one("SELECT id FROM tasks WHERE task_conversation_id = ?", (cid,))
                c.task_id = t["id"] if t else None
            return c

    def _row(self, cid: str):
        row = self.store.one("SELECT * FROM conversations WHERE id = ? AND deleted IS NULL", (cid,))
        if row is None:
            raise ApiError(404, "not_found", "no such conversation")
        return row

    def emit(self, conv: Conv, ev: dict, *, run_id: str | None = None,
             persist: bool = True) -> int | None:
        """One event: stored with the next seq (unless ephemeral), then to every subscriber. A task
        conversation keeps no log: its transcript's events go to its chat's with `task_id`, the
        rest (its run's own run.*, context, usage) stay inside chatd (`task.*` says them)."""
        if conv.parent_id is not None:
            if ev["type"] not in TASK_FORWARD:
                return None
            return self.emit(self.conv(conv.parent_id), {**ev, "task_id": conv.task_id},
                             run_id=run_id, persist=persist)
        data = as_json(ev)
        with conv.lock:
            seq = None
            if persist:
                seq = conv.last_seq + 1
                with self.store.tx() as db:
                    db.execute("INSERT INTO events VALUES (?, ?, ?, ?, ?, ?)",
                               (conv.id, seq, now_ms(), run_id, ev["type"], data))
                    db.execute("UPDATE conversations SET last_seq = ? WHERE id = ?",
                               (seq, conv.id))
                conv.last_seq = seq
            conv.fan.send(seq, ev["type"], data)
            return seq

    def emit_global(self, ev: dict) -> None:
        data = as_json(ev)
        with self.glock:
            self.gseq += 1
            self.store.run("INSERT INTO global_events VALUES (?, ?, ?, ?)",
                           (self.gseq, now_ms(), ev["type"], data))
            self.gfan.send(self.gseq, ev["type"], data)

    def _messages(self, cid: str) -> dict[str, dict]:
        out = {}
        for r in self.store.all("SELECT m.*, r.permission_mode FROM messages m LEFT JOIN runs r "
                                "ON r.id = m.run_id WHERE m.conversation_id = ? ORDER BY m.id",
                                (cid,)):
            out[r["id"]] = {"id": r["id"], "role": r["role"], "created": r["created"],
                            "parent_id": r["parent_id"], "run_id": r["run_id"],
                            "permission_mode": r["permission_mode"],
                            "status": r["status"], "parts": json.loads(r["parts"]),
                            "attachments": json.loads(r["attachments"]) if r["attachments"] else None,
                            "stats": json.loads(r["stats"]) if r["stats"] else None,
                            "client_msg_id": r["client_msg_id"]}
        return out

    @staticmethod
    def _children(msgs: dict[str, dict]) -> dict[str | None, list[str]]:
        kids: dict[str | None, list[str]] = {}
        for mid in sorted(msgs):
            kids.setdefault(msgs[mid]["parent_id"], []).append(mid)
        return kids

    @staticmethod
    def _path(msgs: dict[str, dict], head: str | None) -> list[str]:
        out = []
        while head and head in msgs:
            out.append(head)
            head = msgs[head]["parent_id"]
        return list(reversed(out))

    def _branch(self, conv: Conv, head: str | None) -> tuple[list[dict], dict[str, dict]]:
        """The branch's messages, root to head, the streaming one from memory."""
        msgs = self._messages(conv.id)
        if conv.live is not None and conv.live["id"] in msgs:
            msgs[conv.live["id"]] = {**msgs[conv.live["id"]], "parts": conv.live["parts"],
                                     "status": conv.live["status"]}
        return [msgs[m] for m in self._path(msgs, head)], msgs

    @staticmethod
    def _wire_message(m: dict, kids: dict) -> dict:
        sib = kids.get(m["parent_id"], [])
        out = {"id": m["id"], "role": m["role"], "created_at": iso(m["created"]),
               "parent_id": m["parent_id"], "run_id": m["run_id"], "parts": m["parts"],
               "status": m["status"]}
        if m.get("attachments"):
            out["attachments"] = m["attachments"]
        if len(sib) > 1:
            out["siblings"] = {"index": sib.index(m["id"]), "count": len(sib)}
        if m.get("stats"):
            out["stats"] = m["stats"]
        if m.get("permission_mode"):
            out["permission_mode"] = m["permission_mode"]      # the run's mode: the thread's badge
        return out

    def _status(self, conv: Conv, branch: list[dict]) -> str:
        if conv.run is not None and conv.run.active:
            child = conv.run.child
            return "waiting" if conv.run.phase == "permission" or (
                child is not None and child.phase == "permission") else "running"
        if branch and branch[-1]["role"] == "assistant" and branch[-1]["status"] in (
                "error", "interrupted"):
            # a run that stopped at its own step cap is no error: its Continue card goes on
            # from there (live 2026-10-21, a 100-call run's header said "error")
            last = conv.run if conv.run is not None and conv.run.message_id == branch[-1]["id"] \
                else None
            stats = branch[-1].get("stats") or {}
            if isinstance(stats, str):                  # the rail's row: the column as stored
                try:
                    stats = json.loads(stats)
                except ValueError:
                    stats = {}
            if branch[-1]["status"] == "interrupted" and (
                    (stats or {}).get("interrupted") == "limit" or
                    getattr(last, "interrupt", None) == "limit"):
                return "idle"
            return "error"
        return "idle"

    def _summary(self, conv: Conv, row, branch: list[dict], preview: str | None = None) -> dict:
        """`branch`: the branch's messages (parts optional); `preview`: the last user message's
        text when the branch carries no parts."""
        if preview is None:
            last_user = next((m for m in reversed(branch) if m["role"] == "user"), None)
            preview = render.text_of(last_user["parts"]) if last_user else ""
        preview = next((ln for ln in preview.split("\n") if ln.strip()), "")[:120]
        settings = json.loads(row["settings"])
        return {"id": row["id"], "title": row["title"], "created_at": iso(row["created"]),
                "updated_at": iso(row["updated"]), "pinned": bool(row["pinned"]),
                "archived": bool(row["archived"]), "folder_id": row["folder_id"],
                "status": self._status(conv, branch), "preview": preview,
                "model": settings.get("model") or row["model"] or "",
                "profile": settings.get("profile"),
                "turns": sum(1 for m in branch if m["role"] == "assistant"),
                "now": conv.run.now if conv.run is not None and conv.run.active else None,
                "version": int(row["version"]), "unread": False}

    def summary(self, cid: str) -> dict:
        """The rail's row: the branch walked up from the head without its parts (a recursive
        query), and the parts of the last user message only."""
        conv = self.conv(cid)
        with conv.lock:
            row = self._row(cid)
            metas = [dict(r) for r in self.store.all(
                "WITH RECURSIVE p(id, parent_id, role, status, stats, depth) AS ("
                "SELECT id, parent_id, role, status, stats, 0 FROM messages WHERE id = ? AND "
                "conversation_id = ? UNION ALL SELECT m.id, m.parent_id, m.role, m.status, NULL, "
                "p.depth + 1 FROM messages m JOIN p ON m.id = p.parent_id) "
                "SELECT id, role, status, stats FROM p ORDER BY depth DESC", (row["head_id"], cid))]
            if conv.live is not None and metas and metas[-1]["id"] == conv.live["id"]:
                metas[-1]["status"] = conv.live["status"]
            last_user = next((m for m in reversed(metas) if m["role"] == "user"), None)
            preview = ""
            if last_user is not None:
                r = self.store.one("SELECT parts FROM messages WHERE id = ?", (last_user["id"],))
                preview = render.text_of(json.loads(r["parts"])) if r else ""
            return self._summary(conv, row, metas, preview)

    def snapshot(self, cid: str, viewer: str | None = None) -> dict:
        """`GET /conversations/{id}`: the Conversation with the `seq` it includes."""
        conv = self.conv(cid)
        with conv.lock:
            row = self._row(cid)
            branch, msgs = self._branch(conv, row["head_id"])
            kids = self._children(msgs)
            out = self._summary(conv, row, branch)
            settings = {**DEFAULT_SETTINGS, **json.loads(row["settings"])}
            run = conv.run.as_dict(viewer) if conv.run is not None and conv.run.active else None
            if conv.context is None:
                conv.context = self._context(conv, settings, branch)
            out.update({
                "messages": [self._wire_message(m, kids) for m in branch],
                "head_id": row["head_id"], "settings": settings, "run": run,
                "context": conv.context, "sandbox": self.sandboxes.status(cid),
                "compactions": self.compactions(cid, branch),
                "usage": _usage(branch),
                "queue": self._queue(cid), "ui": json.loads(row["ui"]),
                "todos": self.todos(cid),
                "seq": str(conv.last_seq),
            })
        if run is None:
            self._maybe_peek(conv, settings, branch)
        return out

    def summary_changed(self, conv: Conv, **patch) -> None:
        """`conversation.updated` on the global stream (the rail); the conversation's own stream
        learns the same from its run events. A task conversation is not on the rail."""
        if conv.parent_id is not None:
            return
        self.emit_global({"type": "conversation.updated", "patch": {"id": conv.id, **patch}})

    # ================================================================== conversations: routes
    def list(self, q: str = "", archived: bool = False, folder: str | None = None) -> dict:
        sql = "SELECT id FROM conversations WHERE deleted IS NULL AND kind = 'chat' AND archived = ?"
        args: list = [1 if archived else 0]
        if folder:
            sql += " AND folder_id = ?"
            args.append(folder)
        if q:
            like = f"%{q.lower()}%"
            sql += (" AND (LOWER(title) LIKE ? OR id IN (SELECT conversation_id FROM messages "
                    "WHERE LOWER(parts) LIKE ?))")
            args += [like, like]
        sql += " ORDER BY updated DESC LIMIT 500"
        out = []
        for r in self.store.all(sql, args):
            try:
                out.append(self.summary(r["id"]))
            except ApiError:
                continue
        folders = [{"id": f["id"], "name": f["name"]}
                   for f in self.store.all("SELECT * FROM folders ORDER BY created")]
        return {"conversations": out, "folders": folders}

    def create(self, body: dict, viewer: str | None = None) -> dict:
        caps = self.capabilities()
        settings = {**caps["defaults"],
                    **_clean_settings(self.preset_settings(body.get("preset_id"))),
                    **_clean_settings(body.get("settings") or {})}
        settings["permission_mode"] = settings.get("permission_mode") or self.permission_default()
        cid = ulid()
        t = now_ms()
        self.store.run("INSERT INTO conversations (id, title, created, updated, model, settings, "
                       "ui) VALUES (?, ?, ?, ?, ?, ?, ?)",
                       (cid, TITLE_NEW, t, t, settings.get("model"), dumps(settings),
                        dumps(EMPTY_UI)))
        self.log(f"[conv] {cid} created")
        snap = self.snapshot(cid, viewer)
        self.emit_global({"type": "conversation.created",
                          "conversation": {k: snap[k] for k in _SUMMARY_KEYS}})
        return snap

    def patch(self, cid: str, body: dict) -> dict:
        conv = self.conv(cid)
        with conv.lock:
            row = self._row(cid)
            fields = {k: body[k] for k in ("title", "pinned", "archived", "folder_id")
                      if k in body}
            ui = body.get("ui")
            # a ui-only patch is last-write-wins (CONTRACT section 8, request 3): two devices
            # opening a pane at once is normal, not a conflict
            if fields and body.get("version") is not None and int(body["version"]) != int(
                    row["version"]):
                raise ApiError(409, "version_conflict", "the conversation changed on another "
                               "device", {"current": self.summary(cid)})
            if ui is not None:
                ui = _clean_ui(ui)
                self.store.run("UPDATE conversations SET ui = ? WHERE id = ?", (dumps(ui), cid))
                self.emit(conv, {"type": "ui.changed", "ui": ui})
            if fields:
                sets, args = [], []
                if "title" in fields:
                    title = str(fields["title"] or "").strip()[:200]
                    if title:
                        sets += ["title = ?", "title_auto = 0"]
                        args.append(title)
                for k in ("pinned", "archived"):
                    if k in fields:
                        sets.append(f"{k} = ?")
                        args.append(1 if fields[k] else 0)
                if "folder_id" in fields:
                    sets.append("folder_id = ?")
                    args.append(fields["folder_id"] or None)
                sets += ["version = version + 1", "updated = ?"]
                args += [now_ms(), cid]
                self.store.run(f"UPDATE conversations SET {', '.join(sets)} WHERE id = ?", args)
                s = self.summary(cid)
                patch = {k: s[k] for k in ("title", "pinned", "archived", "folder_id", "version",
                                           "updated_at")}
                self.emit(conv, {"type": "conversation.updated", "patch": patch})
                self.summary_changed(conv, **patch)
            return self.summary(cid)

    def delete(self, cid: str) -> None:
        conv = self.conv(cid)
        run = conv.run
        if run is not None and run.active:
            run.stop("cancel")
            if run.thread is not None:
                run.thread.join(10)
        kids = [r["id"] for r in self.store.all("SELECT id FROM conversations WHERE parent_id = ?",
                                                (cid,))]
        t = now_ms()
        self.store.run("UPDATE conversations SET deleted = ? WHERE id = ? OR parent_id = ?",
                       (t, cid, cid))
        self.store.run("DELETE FROM queue WHERE conversation_id = ?", (cid,))
        with self.convs_lock:
            for k in [cid, *kids]:
                self.convs.pop(k, None)
        with conv.lock:
            for s in list(conv.fan.subs):
                s.close()
        # the chat's sandbox stops with it (its files stay for a restore; the purge removes them)
        self.sandboxes.stop(cid, "deleted")
        self.log(f"[conv] {cid} deleted (trash for 7 days)")
        self.emit_global({"type": "conversation.deleted", "id": cid})

    def restore(self, cid: str) -> dict:
        """POST /conversations/{id}/restore: back from the trash. Its sandbox starts again on its
        next tool call, as any stopped one does."""
        n = self.store.run("UPDATE conversations SET deleted = NULL WHERE id = ? AND deleted IS NOT "
                           "NULL AND kind = 'chat'", (cid,))
        if not n:
            raise ApiError(404, "not_found", "no such conversation in the trash")
        self.store.run("UPDATE conversations SET deleted = NULL WHERE parent_id = ?", (cid,))
        self.log(f"[conv] {cid} restored from the trash")
        snap = self.snapshot(cid)
        self.emit_global({"type": "conversation.created",
                          "conversation": {k: snap[k] for k in _SUMMARY_KEYS}})
        return snap

    def purge(self, cid: str) -> None:
        """The trash's end (7 days, or delete forever): the rows, the sandbox's container and
        the chat's folders (workspace, home, egress, snapshots)."""
        if self.store.one("SELECT deleted FROM conversations WHERE id = ?", (cid,)) is None:
            raise ApiError(404, "not_found", "no such conversation")
        if self.store.one("SELECT 1 FROM conversations WHERE id = ? AND deleted IS NULL", (cid,)):
            self.delete(cid)
        kids = [r["id"] for r in self.store.all("SELECT id FROM conversations WHERE parent_id = ?",
                                                (cid,))]
        with self.store.tx() as db:
            for c in [cid, *kids]:                  # its task conversations go with it (VIS-E15)
                for table in ("events", "messages", "runs", "queue", "todos", "compactions",
                              "sandboxes", "terminals", "tasks"):
                    db.execute(f"DELETE FROM {table} WHERE conversation_id = ?", (c,))
                db.execute("DELETE FROM permission_rules WHERE conversation_id = ?", (c,))
                db.execute("DELETE FROM conversations WHERE id = ?", (c,))
        self.sandboxes.remove(cid, files=True)
        with self.sandboxes.lock:
            self.sandboxes.boxes.pop(cid, None)
        shutil.rmtree(os.path.join(self.data_dir, "snapshots", cid), ignore_errors=True)
        self.log(f"[conv] {cid} purged: its rows, its sandbox and its folders")

    def create_folder(self, name: str) -> dict:
        f = {"id": ulid(), "name": (name or "").strip()[:80] or "Folder"}
        self.store.run("INSERT INTO folders VALUES (?, ?, ?)", (f["id"], f["name"], now_ms()))
        return f

    # ================================================================== permission modes
    def permission_default(self) -> str:
        """The mode a new chat gets (and a chat from before the modes runs in): auto unless the
        user chose another (chatd's meta table)."""
        r = self.store.one("SELECT value FROM meta WHERE key = 'permission_mode_default'")
        return r["value"] if r and r["value"] in tools_mod.MODES else tools_mod.DEFAULT_MODE

    def set_permission_default(self, body: dict) -> dict:
        mode = (body or {}).get("default")
        if mode not in tools_mod.MODES:
            raise ApiError(400, "bad_request", f"default must be one of {', '.join(tools_mod.MODES)}")
        self.store.run("INSERT OR REPLACE INTO meta VALUES ('permission_mode_default', ?)", (mode,))
        self._caps = None                       # the capabilities' defaults name it
        self.emit_global({"type": "permission_mode.default", "default": mode})
        return self.permission_modes()

    def patch_defaults(self, body: dict) -> dict:
        """PATCH /defaults {permission_mode, agents}: the settings a new chat starts with that the
        user sets; answers the capabilities' defaults."""
        clean = _clean_settings(body)
        other = sorted(k for k in clean if k not in ("permission_mode", "agents"))
        if other or not clean:
            raise ApiError(400, "bad_request", "only permission_mode and agents can be set as "
                                               "defaults")
        if "permission_mode" in clean:
            if clean["permission_mode"] is None:
                raise ApiError(400, "bad_request", f"permission_mode must be one of {', '.join(tools_mod.MODES)}")
            self.set_permission_default({"default": clean["permission_mode"]})
        if "agents" in clean:                   # SRV-583: off until the eval rows exist
            self.store.run("INSERT OR REPLACE INTO meta VALUES ('agents_default', ?)",
                           ("1" if clean["agents"] else "0",))
            self._caps = None
        return self.capabilities()["defaults"]

    def agents_default(self) -> bool:
        r = self.store.one("SELECT value FROM meta WHERE key = 'agents_default'")
        return bool(r and r["value"] == "1")

    def permission_modes(self) -> dict:
        """GET /permissions/mode: the default and the modes there are."""
        return {"default": self.permission_default(), "modes": list(tools_mod.MODES)}

    def permission_mode(self, settings: dict) -> str:
        """The mode a chat's calls are decided in: its own, else the default."""
        m = settings.get("permission_mode")
        return m if m in tools_mod.MODES else self.permission_default()

    def chat_permission_mode(self, cid: str) -> dict:
        """GET /conversations/{id}/permission-mode."""
        row = self._row(cid)
        settings = {**DEFAULT_SETTINGS, **json.loads(row["settings"])}
        return {"mode": settings.get("permission_mode"), "effective": self.permission_mode(settings),
                "default": self.permission_default()}

    def set_chat_permission_mode(self, cid: str, body: dict) -> dict:
        """PUT /conversations/{id}/permission-mode {mode}: from the next call on, a running run
        included (its badge follows: `run.mode`)."""
        mode = (body or {}).get("mode")
        if mode not in tools_mod.MODES:
            raise ApiError(400, "bad_request", f"mode must be one of {', '.join(tools_mod.MODES)}")
        self.update_settings(cid, {"permission_mode": mode})
        return self.chat_permission_mode(cid)

    def update_settings(self, cid: str, patch: dict) -> dict:
        conv = self.conv(cid)
        with conv.lock:
            row = self._row(cid)
            settings = {**DEFAULT_SETTINGS, **json.loads(row["settings"])}
            clean = _clean_settings(patch)
            if "policies" in clean:
                clean["policies"] = {**settings.get("policies", {}), **clean["policies"]}
            settings.update(clean)
            self.store.run("UPDATE conversations SET settings = ?, model = ? WHERE id = ?",
                           (dumps(settings), settings.get("model"), cid))
            self.emit(conv, {"type": "settings.changed", "settings": settings})
            run = conv.run
            if run is not None and run.active and run.permission_mode != self.permission_mode(settings):
                run.permission_mode = self.permission_mode(settings)
                self.store.run("UPDATE runs SET permission_mode = ? WHERE id = ?",
                               (run.permission_mode, run.id))
                self.emit(conv, {"type": "run.mode", "run_id": run.id,
                                 "permission_mode": run.permission_mode}, run_id=run.id)
            branch, _ = self._branch(conv, row["head_id"])
            conv.context = self._context(conv, settings, branch)
            self._send_context(conv, force=True)
            return settings

    # ================================================================== the turn
    def send(self, cid: str, body: dict, viewer: str | None = None) -> dict:
        cmid = _client_id(body)
        text = _text(body, bool(body.get("attachments")))
        conv = self.conv(cid)
        with conv.lock:
            dup = self._dup(cid, cmid)
            if dup is not None:
                return dup
            if conv.run is not None and conv.run.active:
                # a second device sent while this chat runs: it waits in the queue (SRV-345)
                return {"queued": self._enqueue(conv, cmid, text)}
            row = self._row(cid)
            parent = body.get("parent_id") or row["head_id"]
            atts = self._attachments(cid, body.get("attachments") or [])
            self._place(conv, row, parent, atts)
            um = self._insert_message(cid, parent, "user", [{"type": "text", "text": text}],
                                      "done", client_msg_id=cmid, attachments=atts or None)
            self._title_from(conv, row, text)
            self.emit(conv, {"type": "message.created", "message": self._wire_message(um, {})})
            run = self._start_run(conv, um["id"], viewer)
            return {"message_id": um["id"], "run_id": run.id}

    def _dup(self, cid: str, cmid: str) -> dict | None:
        """The first answer to a repeated client_msg_id (a phone that retried after a drop)."""
        m = self.store.one("SELECT id FROM messages WHERE conversation_id = ? AND client_msg_id = ?",
                           (cid, cmid))
        if m is not None:
            r = self.store.one("SELECT id FROM runs WHERE user_message_id = ? ORDER BY started "
                               "LIMIT 1", (m["id"],))
            return {"message_id": m["id"], "run_id": r["id"] if r else None}
        q = self.store.one("SELECT * FROM queue WHERE conversation_id = ? AND client_msg_id = ?",
                           (cid, cmid))
        if q is not None:
            return {"queued": _queued(q)}
        return None

    def _title_from(self, conv: Conv, row, text: str) -> None:
        if row["title"] != TITLE_NEW or not row["title_auto"]:
            return
        first = next((ln for ln in text.split("\n") if ln.strip()), "")
        title = first.replace("#", "").replace("*", "").replace("`", "").strip()[:60]
        if not title:
            return
        self.store.run("UPDATE conversations SET title = ?, version = version + 1 WHERE id = ?",
                       (title, conv.id))
        v = self.store.one("SELECT version FROM conversations WHERE id = ?", (conv.id,))[0]
        self.emit(conv, {"type": "conversation.updated", "patch": {"title": title, "version": v}})
        self.summary_changed(conv, title=title, version=v)

    def _insert_message(self, cid: str, parent: str | None, role: str, parts: list, status: str,
                        *, client_msg_id: str | None = None, run_id: str | None = None,
                        attachments: list | None = None) -> dict:
        m = {"id": ulid(), "role": role, "created": now_ms(), "parent_id": parent,
             "run_id": run_id, "status": status, "parts": parts, "attachments": attachments,
             "stats": None, "client_msg_id": client_msg_id}
        self.store.run("INSERT INTO messages (id, conversation_id, parent_id, role, created, "
                       "status, parts, attachments, run_id, client_msg_id) VALUES "
                       "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                       (m["id"], cid, parent, role, m["created"], status, dumps(parts),
                        dumps(attachments) if attachments else None, run_id, client_msg_id))
        self.store.run("UPDATE conversations SET head_id = ?, updated = ? WHERE id = ?",
                       (m["id"], m["created"], cid))
        return m

    def _start_run(self, conv: Conv, user_mid: str, viewer: str | None, *,
                   continue_mid: str | None = None, parent: Run | None = None,
                   task: dict | None = None) -> Run:
        """Under the conversation's lock: the assistant message, the run row, `run.started`,
        `message.created`, then the thread. `parent` and `task`: a worker's run on its task
        conversation (chatd/tasks.py), at most the task's model calls, Continue included."""
        settings = self.settings_of(conv.id)
        if continue_mid:
            mid = continue_mid
            m = self._messages(conv.id)[mid]
            conv.live = {"id": mid, "parts": m["parts"], "status": "streaming"}
            self.store.run("UPDATE messages SET status = 'streaming' WHERE id = ?", (mid,))
            self.store.run("UPDATE conversations SET head_id = ? WHERE id = ?", (mid, conv.id))
        else:
            self.todos_onto(conv.id, user_mid)
            self.plan_onto(conv.id, user_mid)
            m = self._insert_message(conv.id, user_mid, "assistant", [], "streaming")
            mid = m["id"]
            conv.live = {"id": mid, "parts": [], "status": "streaming"}
        steps = int(task["max_steps"]) if task is not None else \
            int(settings.get("max_steps") or DEFAULT_SETTINGS["max_steps"])
        run = Run(self, conv, user_message_id=user_mid, message_id=mid, max_steps=steps,
                  started_by=viewer, continue_final=bool(continue_mid), parent=parent, task=task)
        run.permission_mode = self.permission_mode(settings)
        if parent is None:
            # read once: a switch during the run applies to the next one (the prompt's head)
            run.agents = bool(settings.get("agents"))
        if continue_mid and conv.live["parts"] and (parent is not None or run.agents):
            self._cut_calls(conv, run)
        if continue_mid and conv.live["parts"]:
            last = conv.live["parts"][-1]
            run.step = max(int(p.get("step") or 1) for p in conv.live["parts"])
            # a cut text goes on in its own step; after tool results the next call is a new step
            run.continue_final = last.get("type") in ("text", "thinking")
            run.resume_same_step = run.continue_final
            run.max_steps = steps if task is not None else run.step + steps
        self.store.run("UPDATE messages SET run_id = ? WHERE id = ?", (run.id, mid))
        self.store.run("INSERT INTO runs (id, conversation_id, user_message_id, "
                       "assistant_message_id, state, max_steps, model, started, started_by, "
                       "permission_mode) VALUES (?, ?, ?, ?, 'queued', ?, ?, ?, ?, ?)",
                       (run.id, conv.id, user_mid, mid, run.max_steps, settings.get("model"),
                        run.started_ms, viewer, run.permission_mode))
        conv.run = run
        run.metrics["thinking_budget"] = self._budget(settings)
        self.emit(conv, {"type": "run.started", "run": run.as_dict(viewer)}, run_id=run.id)
        if not continue_mid:
            msgs = self._messages(conv.id)
            self.emit(conv, {"type": "message.created",
                             "message": self._wire_message(msgs[mid], self._children(msgs))},
                      run_id=run.id)
        if conv.parent_id is None:
            self.emit_global({"type": "run.started", "conversation_id": conv.id,
                              "run_id": run.id})
        self.summary_changed(conv, status="running", now=run.now)
        self.log(f"[run] {run.id} started conv={conv.id}")
        self.live.ensure()
        run.start()
        return run

    def _cut_calls(self, conv: Conv, run: Run) -> None:
        """Continue of a worker, or of a main agent in Agents mode (SRV-585): a call the stop cut
        short is never run again. A write, a command or an MCP call reads that its effect is
        unknown; a task call whose worker was cut is the one call that goes on (its worker
        resumes inside it, chatd/tasks.py)."""
        parts = conv.live["parts"]
        for k, p in enumerate(parts):
            if p.get("type") != "tool_call" or p.get("status") not in (
                    "running", "asking", "cancelled", "interrupted"):
                continue
            if p.get("name") == "task":
                r = self.store.one("SELECT 1 FROM tasks WHERE conversation_id = ? AND call_id = ?",
                                   (conv.id, p["id"]))
                if r is not None and k == max(i for i, q in enumerate(parts)
                                              if q.get("type") == "tool_call"):
                    run.resume_part = k
                continue
            if p.get("status") == "interrupted":
                continue
            risky = p.get("name") in ("bash", "write_file", "edit_file") or \
                str(p.get("name") or "").startswith("mcp__")
            words = tools_mod.INTERRUPTED_CALL if risky else tools_mod.INTERRUPTED_READ
            before = p.get("model_content") or ""
            p["model_content"] = (before + "\n" if before else "") + words
            p["status"], p["permission"] = "interrupted", None
            p["ended_at"] = p.get("ended_at") or iso(now_ms())
            self.emit(conv, {"type": "tool.done", "message_id": conv.live["id"],
                             "call_id": p["id"], "status": "interrupted", "is_error": True})
        self._persist_live(conv)

    def _budget(self, settings: dict) -> int | None:
        """The thinking budget in force, from the model's card (effort levels with their caps)."""
        if not settings.get("thinking", True):
            return 0
        eff = settings.get("effort") or "auto"
        if eff == "auto":
            return None
        levels = (self.model_info(settings.get("model")).get("effort_budgets") or {})
        b = levels.get(eff)
        return int(b) if b else None

    def close_thinking(self, run: Run, tokens: int | None = None) -> None:
        conv = run.conv
        with conv.lock:
            self.flush(run)
            if run.open_kind != "thinking" or conv.live is None:
                return
            part = conv.live["parts"][run.open_index]
            part["done"] = True
            part["tokens"] = tokens
            if run.think_started is not None:
                part["duration_ms"] = int((time.monotonic() - run.think_started) * 1000)
            self.emit(conv, {"type": "part.done", "message_id": run.message_id,
                             "index": run.open_index, "tokens": tokens}, run_id=run.id)
            run.open_kind = None

    def _persist_live(self, conv: Conv) -> None:
        if conv.live is not None:
            self.store.run("UPDATE messages SET parts = ?, status = ? WHERE id = ?",
                           (dumps(conv.live["parts"]), conv.live["status"], conv.live["id"]))
            conv.persisted_at = time.monotonic()

    def learn_ratio(self, conv: Conv, prompt_tokens: int, request_chars: int,
                    image_tokens: int = 0) -> None:
        r = __import__("chatd.compaction", fromlist=["learn"]).learn(prompt_tokens, request_chars,
                                                                     image_tokens)
        if r is not None:
            conv.tok_ratio, conv.tok_counted = r, request_chars

    def finish_guard(self, run: Run, outcome: str, exc: Exception) -> None:
        """The last line for a finish_run that broke halfway (runs.py `_main`). Forces the
        three facts everything downstream depends on -- the runs row terminal, the assistant
        message off 'streaming', the conversation released -- then drains the queue. Every
        step is guarded on its own, because the state that broke finish_run may still be
        broken; this must never raise, it is the only thing between the run thread and a
        permanently wedged conversation."""
        conv = run.conv
        status = _STATUS_BY_OUTCOME.get(outcome, "error")
        # the reason finish_run broke is the whole story of a wedged chat — a name only
        # hides it; keep the tail, it is a crash path and crashes want stacks
        self.log(f"[run] {run.id}: finish_run raised {type(exc).__name__}; forcing "
                 f"{outcome}\n{traceback.format_exc(limit=8)}")
        try:
            self.store.run("UPDATE runs SET state = ?, ended = ?, error = ?, step = ? "
                           "WHERE id = ?",
                           (outcome, now_ms(), run.error, run.step, run.id))
        except Exception as exc2:                                       # noqa: BLE001
            self.log(f"[run] {run.id}: the forced runs write failed too: {type(exc2).__name__}")
        try:
            # only the rows the run itself left streaming -- the guard must not touch a
            # message some other run owns
            self.store.run("UPDATE messages SET status = ? WHERE id = ? "
                           "AND status = 'streaming'", (status, run.message_id))
        except Exception as exc2:                                       # noqa: BLE001
            self.log(f"[run] {run.id}: the forced message write failed: {type(exc2).__name__}")
        with conv.lock:                                                 # memory: cannot fail
            run.state = outcome                                         # clears run.active
            if conv.run is run:
                conv.run = None                                         # the message gate opens
            live = conv.live
            if live is not None and live["id"] == run.message_id:
                live["status"] = status                                 # what finish_run would have
                                                                        # finalized
                # finish_run cancels open tool cards before persisting; skipping that
                # here left an 'asking' permission card with no run to answer it —
                # unresolved forever
                for p in live["parts"]:
                    if (p.get("type") == "tool_call"
                            and p.get("status") in ("running", "asking")):
                        p["status"], p["ended_at"] = "cancelled", iso(now_ms())
                        p["permission"] = None
                try:
                    self._persist_live(conv)                            # the cancelled state sticks
                except Exception as exc2:                               # noqa: BLE001
                    self.log(f"[run] {run.id}: forced persist failed: {type(exc2).__name__}")
            conv.live = None
        try:
            # the client deserves a frame at the moment of the flip, not a diff on refetch
            self.emit(conv, {"type": "run.error", "run_id": run.id,
                             "message_id": run.message_id,
                             "message": run.error or f"forced: {outcome}", "retryable": True},
                      run_id=run.id)
        except Exception as exc2:                                       # noqa: BLE001
            self.log(f"[run] {run.id}: the forced run.error failed: {type(exc2).__name__}")
        try:
            self._drain(conv, run.started_by)                           # the queue must not starve
        except Exception as exc2:                                       # noqa: BLE001
            self.log(f"[run] {run.id}: the forced drain failed: {type(exc2).__name__}")

    def finish_run(self, run: Run, outcome: str) -> None:
        """The run's end: the message's final parts, status and figures; `run.done` (or
        `run.error`); then the queue and the title."""
        conv = run.conv
        if run.snapshotted:
            try:
                self.snaps.take(self.root_cid(conv.id), self.workspace(conv.id), run.id, "after")
            except OSError:
                pass
        status = _STATUS_BY_OUTCOME.get(outcome, "error")
        finish = run.finish_reason or {"done": "stop", "stopped": "stopped",
                                       "interrupted": "error", "error": "error"}[status]
        if status == "stopped":
            finish = "stopped"
        stats = run.stats(finish)
        if status == "interrupted":
            # the Continue card's words (VIS-197): it said "the engine restarted" for every one
            stats["interrupted"] = run.interrupt or "engine"
        with conv.lock:
            if run.carry:
                self.append(run, "thinking" if run.in_tag else "text", run.carry)
                run.carry = ""
            self.flush(run)
            if run.open_kind == "thinking":
                self.close_thinking(run, stats.get("reasoning_tokens") or None)
            elif conv.live is not None and (run.usage or {}).get("usage"):
                # a thinking part closed before the usage came: the engine's count, when the
                # call had one thinking part (phase 1's calls do)
                idx = [i for i, p in enumerate(conv.live["parts"]) if p["type"] == "thinking"]
                if len(idx) == 1 and conv.live["parts"][idx[0]].get("tokens") is None and \
                        stats.get("reasoning_tokens"):
                    conv.live["parts"][idx[0]]["tokens"] = stats["reasoning_tokens"]
                    self.emit(conv, {"type": "part.done", "message_id": run.message_id,
                                     "index": idx[0], "tokens": stats["reasoning_tokens"]},
                              run_id=run.id)
            run.state = outcome
            if conv.live is not None and conv.live["id"] == run.message_id:
                for p in conv.live["parts"]:            # a call the stop cut short
                    if p.get("type") == "tool_call" and p.get("status") in ("running", "asking"):
                        p["status"], p["ended_at"], p["permission"] = "cancelled", iso(now_ms()), \
                            None
                        self.emit(conv, {"type": "tool.done", "message_id": run.message_id,
                                         "call_id": p["id"], "status": "cancelled",
                                         "is_error": False}, run_id=run.id)
                for text in (run.notice, run.remark):
                    if not text or status == "error":
                        continue
                    note = {"type": "error", "message": text, "retryable": False,
                            "step": run.step}
                    conv.live["parts"].append(note)
                    self.emit(conv, {"type": "part.added", "message_id": run.message_id,
                                     "part": note}, run_id=run.id)
                said = verify_summary(run.verify, conv.live["parts"]) if status == "done" \
                    else None
                if said is not None:                # what the verify phase checked, and what not
                    said["step"] = run.step
                    conv.live["parts"].append(said)
                    self.emit(conv, {"type": "part.added", "message_id": run.message_id,
                                     "part": said}, run_id=run.id)
                conv.live["status"] = status
                if status == "error":
                    conv.live["parts"].append({"type": "error", "message": run.error or "failed",
                                               "retryable": True})
                self._persist_live(conv)
            # the harness's picks and the prompt's split stay with the answer, so the Context
            # panel still has them after chatd restarts (they lived in memory only)
            frame = (run.usage or {}).get("tandem") or {}
            if frame.get("harness") or frame.get("parts"):
                stats["tandem"] = {"harness": frame.get("harness"), "parts": frame.get("parts")}
            elif run.split_est is not None:         # a relayed call's: chatd's estimate
                stats["tandem"] = {"harness": None, "parts": run.split_est}
            self.store.run("UPDATE messages SET stats = ? WHERE id = ?",
                           (dumps(stats), run.message_id))
            self.store.run("UPDATE runs SET state = ?, ended = ?, error = ?, step = ? WHERE id = ?",
                           (outcome, now_ms(), run.error, run.step, run.id))
            self.store.run("UPDATE conversations SET updated = ? WHERE id = ?", (now_ms(), conv.id))
            if status == "error":
                self.emit(conv, {"type": "run.error", "run_id": run.id,
                                 "message_id": run.message_id, "message": run.error or "failed",
                                 "retryable": True}, run_id=run.id)
            else:
                self.emit(conv, {"type": "run.done", "run_id": run.id,
                                 "message_id": run.message_id, "stats": stats,
                                 "status": status}, run_id=run.id)
            conv.live = None
            if conv.run is run:
                conv.run = None
            if run.parent is not None:
                # a worker: its tasks row, its report and `task.done` on the chat (chatd/tasks.py);
                # no title, no queue (the chat's steers are its parent's), not on the rail
                self.log(f"[run] {run.id} {outcome} task={run.task['id']} conv={conv.parent_id} "
                         f"tokens={stats['completion_tokens']} finish={finish}")
                self.task_ended(run, outcome, stats)
                return
            row = self._row(conv.id)
            branch, _ = self._branch(conv, row["head_id"])
            self.emit(conv, {"type": "usage.changed", "usage": _usage(branch)})
            settings = {**DEFAULT_SETTINGS, **json.loads(row["settings"])}
            conv.peek = None
            # SRV-349: the usage frame's `tandem` -- the harness's choices and the split by part
            tandem = (run.usage or {}).get("tandem") or {}
            conv.context = self._context(conv, settings, branch, harness=tandem.get("harness"),
                                         split=tandem.get("parts"))
            self._send_context(conv, force=True)
            self.summary_changed(conv, status=self._status(conv, branch), now=None,
                                 updated_at=iso(now_ms()))
        self.emit_global({"type": "run.done", "conversation_id": conv.id, "run_id": run.id,
                          "status": status})
        self.log(f"[run] {run.id} {outcome} conv={conv.id} tokens={stats['completion_tokens']} "
                 f"finish={finish}")
        if outcome == "completed" and self.titles:
            threading.Thread(target=self._title, args=(conv.id,), daemon=True).start()
        if run.stop_mode not in ("cancel", "shutdown"):
            self._drain(conv, run.started_by)

    # ---- the queue
    def _queue(self, cid: str) -> list[dict]:
        return [_queued(q) for q in self.store.all(
            "SELECT * FROM queue WHERE conversation_id = ? ORDER BY created, id", (cid,))]

    def _enqueue(self, conv: Conv, cmid: str, text: str) -> dict:
        q = {"id": ulid(), "text": text, "created": now_ms(), "client_msg_id": cmid}
        child = conv.run.child if conv.run is not None else None
        if child is not None and child.active:
            # a steer while a worker runs: the main agent reads it after the report, as any
            # steer; the worker sees it as a note on its next tool result (SRV-585)
            child.notes.append((time.strftime("%H:%M"), text))
        self.store.run("INSERT INTO queue VALUES (?, ?, ?, ?, ?)",
                       (q["id"], conv.id, text, cmid, q["created"]))
        self.emit(conv, {"type": "queue.changed", "queue": self._queue(conv.id)})
        return {"id": q["id"], "text": text, "created_at": iso(q["created"]), "client_msg_id": cmid}

    def queue(self, cid: str, body: dict, viewer: str | None = None) -> dict:
        cmid = _client_id(body)
        conv = self.conv(cid)
        with conv.lock:
            dup = self._dup(cid, cmid)
            if dup is not None:
                return dup.get("queued") or dup
            q = self._enqueue(conv, cmid, _text(body, False))
            idle = conv.run is None or not conv.run.active
        if idle:
            self._drain(conv, viewer)
        return q

    def unqueue(self, cid: str, qid: str) -> None:
        conv = self.conv(cid)
        with conv.lock:
            if self.store.run("DELETE FROM queue WHERE id = ? AND conversation_id = ?",
                              (qid, cid)):
                self.emit(conv, {"type": "queue.changed", "queue": self._queue(cid)})

    def send_now(self, cid: str, qid: str, viewer: str | None = None) -> None:
        """Stop the current generation (its text stays), then this queued message goes in."""
        conv = self.conv(cid)
        with conv.lock:
            q = self.store.one("SELECT * FROM queue WHERE id = ? AND conversation_id = ?",
                               (qid, cid))
            if q is None:
                raise ApiError(404, "not_found", "no such queued message")
            first = self.store.one("SELECT MIN(created) FROM queue WHERE conversation_id = ?",
                                   (cid,))[0]
            self.store.run("UPDATE queue SET created = ? WHERE id = ?", (int(first) - 1, qid))
            self.emit(conv, {"type": "queue.changed", "queue": self._queue(cid)})
            run = conv.run
        if run is not None and run.active:
            run.stop("now")            # finish_run drains the queue, this message first
        else:
            self._drain(conv, viewer)

    def _drain(self, conv: Conv, viewer: str | None) -> None:
        with conv.lock:
            if conv.run is not None and conv.run.active:
                return
            q = self.store.one("SELECT * FROM queue WHERE conversation_id = ? ORDER BY created, id "
                               "LIMIT 1", (conv.id,))
            if q is None:
                return
            try:
                self._row(conv.id)
            except ApiError:
                return
            self.store.run("DELETE FROM queue WHERE id = ?", (q["id"],))
            self.emit(conv, {"type": "queue.changed", "queue": self._queue(conv.id)})
            self.send(conv.id, {"client_msg_id": q["client_msg_id"], "text": q["text"]}, viewer)

    # ---- stop, continue, regenerate, edit, branch
    def stop(self, cid: str) -> None:
        conv = self.conv(cid)
        run = conv.run
        if run is not None and run.active:
            self.log(f"[run] {run.id} stop asked")
            run.stop("stop")

    def _take_over(self, conv: Conv) -> None:
        """An edit or a regenerate ends the run that is on first (no queue drain)."""
        run = conv.run
        if run is not None and run.active:
            run.stop("cancel")
            if run.thread is not None and run.thread is not threading.current_thread():
                run.thread.join(15)

    def continue_run(self, cid: str, viewer: str | None = None) -> dict:
        conv = self.conv(cid)
        with conv.lock:
            if conv.run is not None and conv.run.active:
                raise ApiError(409, "run_active", "a run is on")
            row = self._row(cid)
            branch, _ = self._branch(conv, row["head_id"])
            last = branch[-1] if branch else None
            if last is None or last["role"] != "assistant" or last["status"] != "interrupted":
                raise ApiError(409, "nothing_to_continue", "the last answer was not interrupted")
            run = self._start_run(conv, last["parent_id"], viewer, continue_mid=last["id"])
            return {"message_id": last["id"], "run_id": run.id}

    def regenerate(self, cid: str, mid: str, viewer: str | None = None) -> dict:
        conv = self.conv(cid)
        self._take_over(conv)
        with conv.lock:
            msgs = self._messages(cid)
            m = msgs.get(mid)
            if m is None:
                raise ApiError(404, "not_found", "no such message")
            user_mid = m["parent_id"] if m["role"] == "assistant" else m["id"]
            if user_mid is None:
                raise ApiError(400, "bad_request", "nothing to answer")
            self.store.run("UPDATE conversations SET head_id = ? WHERE id = ?", (user_mid, cid))
            self.emit(conv, {"type": "branch.changed", "head_id": user_mid})
            run = self._start_run(conv, user_mid, viewer)
            return {"message_id": run.message_id, "run_id": run.id}

    def edit(self, cid: str, mid: str, body: dict, viewer: str | None = None) -> dict:
        cmid = _client_id(body)
        text = _text(body, False)
        conv = self.conv(cid)
        with conv.lock:
            dup = self._dup(cid, cmid)
            if dup is not None:
                return dup
        self._take_over(conv)
        with conv.lock:
            m = self._messages(cid).get(mid)
            if m is None or m["role"] != "user":
                raise ApiError(404, "not_found", "no such user message")
            um = self._insert_message(cid, m["parent_id"], "user",
                                      [{"type": "text", "text": text}],
                                      "done", client_msg_id=cmid, attachments=m.get("attachments"))
            self.emit(conv, {"type": "branch.changed", "head_id": um["id"]})
            msgs = self._messages(cid)
            self.emit(conv, {"type": "message.created",
                             "message": self._wire_message(um, self._children(msgs))})
            run = self._start_run(conv, um["id"], viewer)
            return {"message_id": um["id"], "run_id": run.id}

    def switch_branch(self, cid: str, body: dict, viewer: str | None = None) -> dict:
        conv = self.conv(cid)
        with conv.lock:
            if conv.run is not None and conv.run.active:
                raise ApiError(409, "run_active", "a run is on; switch branches when it ends")
            msgs = self._messages(cid)
            m = msgs.get(str(body.get("message_id") or ""))
            if m is None:
                raise ApiError(404, "not_found", "no such message")
            kids = self._children(msgs)
            sib = kids.get(m["parent_id"], [])
            pick = sib[max(0, min(len(sib) - 1, int(body.get("index") or 0)))]
            head = pick
            while kids.get(head):
                head = kids[head][-1]
            self.store.run("UPDATE conversations SET head_id = ? WHERE id = ?", (head, cid))
            self.emit(conv, {"type": "branch.changed", "head_id": head})
            conv.context = None
        return self.snapshot(cid, viewer)

    # ================================================================== the ticker
    def _runs_on(self) -> bool:
        with self.convs_lock:
            return any(c.run is not None and c.run.active for c in self.convs.values())

    def _tick(self) -> None:
        last_slow = 0.0
        while not self.stopping.is_set():
            time.sleep(FLUSH_S)
            with self.convs_lock:
                convs = list(self.convs.values())
            slow = time.monotonic() - last_slow >= METRICS_S
            if slow:
                last_slow = time.monotonic()
            if slow:
                try:
                    self.close_idle_terminals()
                except Exception as exc:                            # noqa: BLE001
                    self.log(f"[tick] terminals {type(exc).__name__}")
            for conv in convs:
                run = conv.run
                try:
                    if run is not None and run.active:
                        self.flush(run)
                        if slow:
                            self._metrics(conv, run)
                            if time.monotonic() - conv.persisted_at >= PERSIST_S:
                                with conv.lock:
                                    self._persist_live(conv)
                    elif slow and len(conv.fan) and \
                            time.monotonic() - conv.context_sent_at >= CONTEXT_IDLE_S:
                        self._send_context(conv)
                except Exception as exc:                            # noqa: BLE001
                    self.log(f"[tick] {conv.id} {type(exc).__name__}")

    def _metrics(self, conv: Conv, run: Run) -> None:
        """Once a second: `run.metrics` (ephemeral) from the run's live row, else chatd's counts;
        prefill progress and the queue place move the Now line too. The live row is this model
        call's: a run of several calls (a tool call each) has a finished row of each earlier call
        in the snapshot for 30 s, and one of those read live (2026-10-16) said "not tracked" for
        the speed and the old call's tokens. Tokens and thinking are the run's: the earlier
        calls' plus this one's. Tokens per round and acceptance between the rows' figures come
        from the run's finished calls (their usage frames)."""
        row = self.live.row(self.session_of(conv), run.request_id)
        if row is not None and row.get("phase") == "done" and \
                row.get("request_id") not in run.request_ids:
            row = None                              # another run's call, or a previous turn's
        if row is not None and (row.get("backend") or run.backend != "engine"):
            # a relayed call's row (a cloud model's) counts no tokens (proxy/measure.py: a frame
            # is not a token), and its zeros overwrote chatd's own counts: live 2026-10-17 a
            # DeepSeek turn read 0 thinking and "not tracked" for a minute of thinking
            row = None
        act = (row or {}).get("activity") or {}
        dec = act.get("decode") or {}
        pre = act.get("prefill") or {}
        turn = run.turn_figures()
        if row is not None:
            live = row.get("phase") != "done"
            tps = dec.get("tps_now") if live and (dec.get("tokens") or 0) > 50 else None
            if tps is None and live and row.get("decode_live_now_tps") is not None:
                tps = row["decode_live_now_tps"]
            run.metrics["decode_tps"] = tps
            run.metrics["tokens_per_round"] = (dec.get("tokens_per_round_now") or
                                               row.get("tokens_per_block") if live else None) \
                or turn["tokens_per_round"]
            # the live row has no acceptance rate (`accept_mean` is a mean length, not a share):
            # the run's finished calls' share, from their usage frames
            run.metrics["acceptance"] = turn["acceptance"]
            if row.get("thinking_budget") is not None:
                run.metrics["thinking_budget"] = row["thinking_budget"]
            if live and dec.get("tokens") is not None and run.usage is None:
                run.tokens = run.call_base[0] + int(dec["tokens"])
                run.thinking_tokens = run.call_base[1] + int(dec.get("thinking_tokens") or 0)
            state = act.get("state")
            # a row polled before the first token came says prefilling after it: the call has
            # left its prefill once its phase did
            if state == "prefilling" and pre.get("total") and run.phase in ("queued", "prefill"):
                run.prefill = {"done": int(pre.get("done") or 0), "total": int(pre["total"]),
                               "tps": pre.get("tps_now"), "eta_ms": pre.get("eta_ms"),
                               "cached": int(pre.get("cached") or 0), "why": pre.get("why")}
                run.set_phase("prefill", prefill_words(pre))
            elif state == "queued" and run.phase in ("queued", "prefill"):
                place = (act.get("queue") or {}).get("place")
                run.set_phase("queued", act.get("label") or "Queued",
                              queue_position=int(place) if place else None)
        else:
            run.metrics["decode_tps"] = run.local_decode_tps()
            run.metrics["tokens_per_round"] = turn["tokens_per_round"]
            run.metrics["acceptance"] = turn["acceptance"]
        run.metrics["ttft_ms"] = (row.get("ttft_ms") if row is not None and
                                  row.get("phase") != "done" and row.get("ttft_ms") is not None
                                  else run.live_ttft_ms())
        run.tps_history.append(run.metrics["decode_tps"])
        del run.tps_history[:-120]
        if conv.parent_id is not None:              # a worker: `task.metrics` on its chat
            self.task_metrics(run)
            return
        self.emit(conv, {"type": "run.metrics", "run_id": run.id,
                         "decode_tps": run.metrics["decode_tps"],
                         "tokens_per_round": run.metrics["tokens_per_round"],
                         "acceptance": run.metrics["acceptance"], "tokens": run.tokens,
                         "thinking_used": run.thinking_tokens,
                         "thinking_call": run.thinking_tokens - run.call_base[1],
                         "thinking_budget": run.metrics["thinking_budget"],
                         "ttft_ms": run.metrics["ttft_ms"],
                         "prefill": run.prefill}, persist=False)
        if time.monotonic() - conv.context_sent_at >= CONTEXT_RUN_S:
            with conv.lock:
                row_ = self._row(conv.id)
                settings = {**DEFAULT_SETTINGS, **json.loads(row_["settings"])}
                branch, _ = self._branch(conv, row_["head_id"])
                conv.context = self._context(conv, settings, branch, live_row=row, run=run)
            self._send_context(conv)

    def _send_context(self, conv: Conv, force: bool = False) -> None:
        if conv.context is None or conv.parent_id is not None:
            return
        if not force and not len(conv.fan):
            return
        conv.context_sent_at = time.monotonic()
        self.emit(conv, {"type": "context.changed", "context": conv.context}, persist=False)
        # the sandbox's figures went out only when its state changed: live (2026-10-16) the panel
        # said 20 MB and 24 KB of disk through an npm install of 51 MB
        at, sent = conv.sandbox_sent
        if time.monotonic() - at >= SANDBOX_S and self.sandboxes.box(conv.id).state == "running":
            st = self.sandboxes.status(conv.id)
            conv.sandbox_sent = (time.monotonic(), st)
            if st != sent:
                self.emit(conv, {"type": "sandbox.changed", "sandbox": st}, persist=False)

    # ================================================================== the context report
    def _context(self, conv: Conv, settings: dict, branch: list[dict], *, live_row=None,
                 run: Run | None = None, harness: dict | None = None,
                 split: dict | None = None) -> dict:
        info = self.model_info(settings.get("model"))
        stats = next((m["stats"] for m in reversed(branch)
                      if m["role"] == "assistant" and m.get("stats")), None)
        files = 0
        for m in branch:
            for a in m.get("attachments") or []:
                files += (int(a.get("tokens") or 0) if render.inlined(a) else
                          attach_mod.LINE_TOKENS)
        cp = self._compaction_for(conv.id, branch) if hasattr(self, "_compaction_for") else None
        step_from = (cp[2][0], cp[2][1]) if cp and cp[2] else None
        files += ctx_mod.file_tokens(branch, cp[1] if cp else 0, step_from, cp[3] if cp else None,
                                     int(info.get("max_len") or 0))
        # the run in flight counts from its own calls, whoever asks (a snapshot too)
        on = run or (conv.run if conv.run is not None and conv.run.active else None)
        tokens = _branch_tokens(branch, on, step_from)
        if on is not None and on.step_usage:
            # its last call's cache, not the run's before Continue (live 2026-10-17: "178,858 warm
            # from the end", the figure of step 100 while step 114 was refused)
            last = on.step_usage[-1]
            stats = dict(stats or {}, prompt_tokens=last.get("prompt_tokens"),
                         cached_tokens=last.get("cached_tokens"),
                         cache_source=last.get("cache_source"))
        # the turn's thinking the next request carries, the call in flight's too: what
        # `_branch_tokens` left out
        k = next((i for i in range(len(branch) - 1, -1, -1) if branch[i]["role"] == "assistant"),
                 None)
        thinking = None
        if k is not None:
            usage = _usage_of(branch[k], on)
            if on is not None and branch[k].get("id") == on.message_id and \
                    on.step not in {u.get("step") for u in usage}:
                usage.append({"step": on.step,
                              "reasoning_tokens": on.thinking_tokens - on.call_base[1]})
            thinking = _turn_thinking(branch[k], usage, step_from[1] if step_from and
                                      step_from[0] == k else 1,
                                      total=(branch[k].get("stats") or {}).get("reasoning_tokens"))
        if harness is None and conv.context is not None:
            harness = conv.context.get("harness")
        if split is None and conv.context is not None:
            split = conv.context.get("split")
        kept = (stats or {}).get("tandem") or {}
        if harness is None:
            harness = kept.get("harness")
        if split is None:
            split = kept.get("parts")
        peek = conv.peek[1] if conv.peek is not None else None
        if run is not None and run.backend != "engine":
            # a relayed call (a cloud model): no live row counts its prompt; chatd's estimate of
            # the request in flight, the split it estimated, and the provider's cache as the last
            # finished call's usage said
            split = run.split_est or split
            last = run.step_usage[-1] if run.step_usage else {}
            live_row = {"prompt_tokens": run.request_est or None,
                        "cached_tokens": min(int(last.get("cached_tokens") or 0),
                                             run.request_est or 0),
                        "cache_source": "provider"} if run.request_est else None
        out = ctx_mod.report(model=info, settings=settings, branch_tokens=tokens,
                             files_tokens=files, stats=stats,
                             run=run.as_dict() if run is not None else None, row=live_row,
                             facts=self.facts(), peek=peek, thinking=thinking,
                             harness=harness if isinstance(harness, dict) else None,
                             split=split if isinstance(split, dict) else None,
                             session=render.session_id(conv.id))
        if conv.memory_limit:                       # where the guard compacts this chat now
            out["compact_at"] = comp.compact_at(out["window"], conv.memory_limit)
        out["skills"] = self.skills_report(settings)
        # the running workers' own contexts, by task (VIS-E15)
        out["workers"] = [self.worker_context(w) for w in self.running_workers(on)]
        out["last_worker"] = self.last_worker(conv.id) if not out["workers"] else None
        return out

    def _maybe_peek(self, conv: Conv, settings: dict, branch: list[dict]) -> None:
        """On attach: where the chat's cache lives (`POST /v1/cache/peek`, SRV-349), and a prewarm
        when it would come back from the NVMe spill or the disk tier and the engine is idle. A
        server without the routes answers 404 and the panel keeps the last request's figures."""
        if not branch or conv.parent_id is not None or time.monotonic() < self._no_peek_until or (
                conv.peek is not None and time.monotonic() - conv.peek[0] < PEEK_TTL_S):
            return
        conv.peek = (time.monotonic(), None)

        def work():
            # the next request's body, head and all: its system block and tools are the rows'
            req = {k: v for k, v in self.peek_body(conv, settings, branch).items()
                   if k not in ("stream", "stream_options")}
            req["session"] = render.session_id(conv.id)
            st, obj = self.engine.post_json("/v1/cache/peek", req, timeout=5.0)
            if st in (404, 405):
                self._no_peek_until = time.monotonic() + 600.0
            if st != 200 or not isinstance(obj, dict):
                return
            conv.peek = (time.monotonic(), obj)
            if obj.get("tier") in ("nvme_spill", "disk_tier"):
                self.engine.post_json("/v1/cache/prewarm", req, timeout=5.0)
            with conv.lock:
                conv.context = self._context(conv, settings, branch)
            self._send_context(conv, force=True)
        threading.Thread(target=work, name="chatd-peek", daemon=True).start()

    # ================================================================== the model and the server
    def facts(self) -> dict:
        """The engine's facts, at most CAPS_TTL_S old. Only the first call waits for the engine;
        after it a stale copy is answered at once and refreshed in the background, so a snapshot
        never waits on a slow engine while it holds a conversation's lock."""
        self._refresh()
        return self._facts[1]

    def _refresh(self, wait: bool = False) -> None:
        if self._caps is None:
            with self._first_lock:
                if self._caps is None:
                    self._refreshing = True
                    self._refill()
            return
        with self._refresh_lock:
            fresh = time.monotonic() - self._caps[0] < (
                CAPS_TTL_S if self._caps[1].get("models") else 3.0)
            if fresh or self._refreshing:
                return
            self._refreshing = True
        if wait:
            self._refill()
        else:
            threading.Thread(target=self._refill, name="chatd-caps", daemon=True).start()

    def _refill(self) -> None:
        try:
            f = self.engine.facts()
            self._facts = (time.monotonic(), f)
            self._caps = (time.monotonic(), self._build_caps(f))
        finally:
            self._refreshing = False

    def capabilities(self, *, password_set: bool = False, signed_in: bool = True) -> dict:
        """`GET /capabilities` (also the dashboard's probe)."""
        self._refresh()
        caps = {k: v for k, v in self._caps[1].items() if not k.startswith("_")}
        caps["session"] = {"password_set": password_set, "signed_in": signed_in,
                           "sandbox": "ready" if self.sandboxes.available() else "not_installed"}
        caps["defaults"] = {**caps["defaults"],
                            "tools": [t.name for t in self.available_builtins()]}
        caps["commands"] = self.commands()
        ws = self.web_search_global()
        caps["web_search"] = {"provider": ws["provider"], "providers": list(websearch.PROVIDERS)}
        return caps

    def _build_caps(self, facts: dict) -> dict:
        models = self.engine.models()
        eng = facts.get("engine") or {}
        out = []
        for m in models:
            card = m.get("tandem") if isinstance(m.get("tandem"), dict) else {}
            c = card.get("capabilities") or {}
            ours = m.get("owned_by") in ("local", "tandemllm") or card.get("backend") == "engine"
            w = eng.get("weights") if ours else None
            levels = c.get("effort_levels") or []
            cloud = bool(card.get("cloud")) or card.get("kind") in ("openai", "anthropic", "deepseek",
                                                                    "cloud")
            out.append({
                "id": m["id"], "name": _model_name(m, card, ours, cloud),
                "weights": card.get("pack") or (w.get("name") if isinstance(w, dict) else "") or "",
                "max_len": int(card.get("context_length") or (eng.get("max_len") if ours else 0)
                               or 0),
                # the conductor's profiles are each client's harness (opencode, Open WebUI,
                # Hermes, tests); the Chat tab always runs under its own, chat-app
                "profiles": [],
                "backend": "engine" if ours else "cloud" if cloud else "proxy",
                "capabilities": {
                    "thinking": bool(c.get("thinking", False)),
                    "effort_levels": [x for x in levels if x != "off"],
                    # a provider's default level (DeepSeek's `high`), null for our engine
                    "effort_default": c.get("effort_default"),
                    "tools": bool(c.get("tools", True)), "vision": bool(c.get("vision", False)),
                    "structured": bool(c.get("structured", False))},
                "loaded": bool(card.get("loaded", True)),
                # chatd's own use, not part of the contract's list
                # the weight profile the engine serves (QSE_WEIGHTS, ops/use-profile.sh)
                "_profile": (w.get("name") if isinstance(w, dict) else None) or card.get("pack"),
                "_max_output": card.get("max_output") or (eng.get("default_max_tokens")
                                                         if ours else None),
                "_drafter": (w.get("drafters") if isinstance(w, dict) else None)
                            or eng.get("drafter"),
                "_effort_budgets": card.get("effort_budgets") or {},
                # the word a provider takes to turn thinking off (DeepSeek: reasoning_effort none)
                "_effort_off": c.get("effort_off"),
            })
        first = out[0] if out else None
        levels = (first or {}).get("capabilities", {}).get("effort_levels") or []
        defaults = dict(DEFAULT_SETTINGS)
        defaults["model"] = first["id"] if first else ""
        defaults["permission_mode"] = self.permission_default()
        defaults["agents"] = self.agents_default()
        return {
            "models": [{k: v for k, v in m.items() if not k.startswith("_")} for m in out],
            "_models": out,
            "efforts": ["auto"] + [x for x in EFFORTS[1:-1] if x in levels],
            "commands": [{"command": "clear", "description": "start a new chat"}],
            "defaults": defaults,
        }

    def model_info(self, model_id: str | None) -> dict:
        self._refresh()
        models = self._caps[1].get("_models") or []
        m = next((x for x in models if x["id"] == model_id), None) or (models[0] if models else {})
        return {"id": m.get("id") or model_id or "", "max_len": m.get("max_len") or 0,
                "weights": m.get("weights") or "", "drafter": m.get("_drafter"),
                "profile": m.get("_profile"),
                "backend": m.get("backend") or "engine",
                "max_output": m.get("_max_output"),
                "vision": (m.get("capabilities") or {}).get("vision", False),
                "tools": (m.get("capabilities") or {}).get("tools", True),
                "effort_budgets": m.get("_effort_budgets") or {},
                "effort_levels": (m.get("capabilities") or {}).get("effort_levels") or [],
                "effort_off": m.get("_effort_off")}

    def takes_slot(self, run) -> bool:
        """Whether the run's next call needs one of the run slots: a cloud model's does not, it
        runs at the provider, not on the box (live 2026-10-17: a DeepSeek chat waited behind two
        engine runs). Read per call: the chat's model can change between two calls of a run."""
        try:
            settings = self.settings_of(run.conv.id)
            return self.model_info(settings.get("model")).get("backend") != "cloud"
        except Exception:                                           # noqa: BLE001
            return True

    # ================================================================== attachments
    def upload(self, content_type: str, body: bytes) -> dict:
        if len(body) > MAX_ATTACHMENT_BYTES + 65536:
            raise ApiError(413, "too_large", "an attachment may be at most 50 MB")
        msg = email.parser.BytesParser(policy=email.policy.HTTP).parsebytes(
            b"Content-Type: " + content_type.encode() + b"\r\n\r\n" + body)
        part = None
        for p in msg.iter_parts() if msg.is_multipart() else []:
            if p.get_param("name", header="content-disposition") == "file":
                part = p
                break
        if part is None:
            raise ApiError(400, "bad_request", "send the file as multipart field `file`")
        data = part.get_payload(decode=True) or b""
        if len(data) > MAX_ATTACHMENT_BYTES:
            raise ApiError(413, "too_large", "an attachment may be at most 50 MB")
        name = os.path.basename(part.get_filename() or "file")[:200]
        mime = part.get_content_type() or "application/octet-stream"
        return self.store_blob(None, name, mime, data)

    def import_owui(self, content_type: str, body: bytes) -> dict:
        from chatd.skills import multipart_file
        try:
            name, data = multipart_file(content_type, body)
            source = data.decode("utf-8")
        except UnicodeDecodeError:
            raise ApiError(400, "bad_tools_file", "the file is not UTF-8 text") from None
        except ValueError as exc:
            raise ApiError(400, "bad_tools_file", str(exc)) from None
        # a .py file, or Open WebUI's JSON export of its Workspace tools ([{id, name, content}])
        try:
            export = json.loads(source) if source.lstrip()[:1] in ("[", "{") else None
        except ValueError:
            export = None
        if export is not None:
            items = export if isinstance(export, list) else [export]
            entries = [(str(x.get("id") or x.get("name") or "tools"), x["content"]) for x in items
                       if isinstance(x, dict) and isinstance(x.get("content"), str)]
            if not entries:
                raise ApiError(400, "bad_tools_file", "a JSON export needs entries with `content`")
        else:
            entries = [(name, source)]
        kits = [self.owui.add(n, src) for n, src in entries]
        names = {t for k in kits for t in k["tools"]}
        return {"toolkit": kits[0], "toolkits": kits, "valves": kits[0]["valves"],
                "tools": [i for i in self.tool_infos() if i["id"] in names]}

    def attachment(self, att_id: str) -> tuple[str, bytes]:
        r = self.store.one("SELECT * FROM attachments WHERE id = ?", (att_id,))
        if r is None:
            raise ApiError(404, "not_found", "no such attachment")
        with open(os.path.join(self.blobs, r["sha256"][:2], r["sha256"]), "rb") as fh:
            return r["mime"], fh.read()

    def _attachments(self, cid: str, ids: list) -> list[dict]:
        out = []
        for att_id in ids:
            r = self.store.one("SELECT * FROM attachments WHERE id = ?", (str(att_id),))
            if r is None:
                raise ApiError(400, "bad_request", "an attachment was not uploaded")
            self.store.run("UPDATE attachments SET conversation_id = ? WHERE id = ?", (cid, r["id"]))
            out.append(_attachment(r["id"], r["name"], r["kind"], r["mime"], r["bytes"],
                                   r["tokens"]))
        return out

    def _blob_of(self, att: dict) -> bytes | None:
        try:
            return self.attachment(att["id"])[1]
        except (ApiError, OSError, KeyError):
            return None

    def _place(self, conv: Conv, row, parent: str | None, atts: list[dict]) -> None:
        """SRV-487, when a message is sent: each attachment is saved in the chat's workspace under
        uploads/ (`path`; `unplaced`, why not, when it cannot be: the message is sent all the
        same), and a text file goes into the prompt whole only while the message's
        text files stay within a quarter of the window's free room, at most INLINE_MAX_TOKENS
        (`inline`); past it the prompt gets one line with its name, size and path
        (chatd/attach.py, chatd/render.py)."""
        if not atts:
            return
        settings = {**DEFAULT_SETTINGS, **json.loads(row["settings"])}
        window = int(self.model_info(settings.get("model")).get("max_len") or 0)
        branch, _ = self._branch(conv, parent)          # an edit goes on the parent's branch
        left = attach_mod.budget(window, _branch_tokens(branch))
        ws_dir = self.workspace(conv.id).root
        for a in atts:
            try:
                _, data = self.attachment(a["id"])
            except (ApiError, OSError):
                continue
            try:
                a["path"] = attach_mod.save_upload(ws_dir, a.get("name") or "file", data)
            except (PathError, OSError) as exc:
                # the sandbox owns uploads/ (a symlink there is refused): the message goes
                # without the file in the workspace, and the prompt says why (chatd/attach.py)
                a["unplaced"] = attach_mod.unplaced_words(exc)
                self.log(f"[attach] {a['id']} not saved under uploads/: {a['unplaced']}")
            if a.get("kind") != "text":
                continue
            tokens = int(a.get("tokens") or 0)
            a["inline"] = tokens <= left
            if a["inline"]:
                left -= tokens
            else:
                where = f"at {a['path']}" if a.get("path") else f"not saved: {a.get('unplaced')}"
                self.log(f"[attach] {a['id']} {tokens} tokens over the budget: one line, the file "
                         f"{where}")

    # ================================================================== the stream
    def attach(self, cid: str, after: int | None) -> tuple[Subscriber | None, list]:
        """A subscriber and the events to replay first; (None, []) means `reset` (too far behind,
        ahead of the log, or the log has a hole where old deltas were deleted)."""
        conv = self.conv(cid)
        with conv.lock:
            last = conv.last_seq
            if after is None:
                after = last
            if after > last or last - after > RESET_BEHIND:
                return None, []
            sub = Subscriber()
            conv.fan.add(sub)
        rows = self.store.all("SELECT seq, type, data FROM events WHERE conversation_id = ? AND "
                              "seq > ? AND seq <= ? ORDER BY seq", (cid, after, last))
        if len(rows) != last - after:
            with conv.lock:
                conv.fan.remove(sub)
            return None, []
        return sub, [(r["seq"], r["type"], r["data"]) for r in rows]

    def detach(self, cid: str, sub: Subscriber) -> None:
        with self.convs_lock:
            conv = self.convs.get(cid)
        if conv is not None:
            with conv.lock:
                conv.fan.remove(sub)

    def attach_global(self, after: int | None) -> tuple[Subscriber | None, list]:
        with self.glock:
            last = self.gseq
            if after is None:
                after = last
            if after > last or last - after > RESET_BEHIND:
                return None, []
            sub = Subscriber()
            self.gfan.add(sub)
        rows = self.store.all("SELECT gseq, type, data FROM global_events WHERE gseq > ? AND "
                              "gseq <= ? ORDER BY gseq", (after, last))
        return sub, [(r["gseq"], r["type"], r["data"]) for r in rows]

    # ================================================================== start, title, sweep
    def _interrupt_leftovers(self) -> int:
        """A chatd that stopped with runs on: every such run is `interrupted` now, its message keeps
        the text it had written, and Continue resumes it."""
        n = 0
        # a run whose worker was running: `task` (Continue resumes the worker first, chatd/tasks.py)
        tasked = {r["run_id"] for r in self.store.all("SELECT run_id FROM tasks WHERE status = "
                                                      "'running'")}
        for r in self.store.all("SELECT * FROM runs WHERE state IN ('queued', 'running')"):
            n += 1
            try:
                conv = self.conv(r["conversation_id"])
            except ApiError:
                self.store.run("UPDATE runs SET state = 'interrupted' WHERE id = ?", (r["id"],))
                continue
            why = "task" if r["id"] in tasked else "restart"
            with conv.lock:
                stats = {**_empty_stats("error"), "interrupted": why}
                # a Continued message keeps its earlier run's figures, with this reason
                self.store.run("UPDATE messages SET status = 'interrupted', stats = CASE WHEN "
                               "stats IS NULL THEN ? ELSE json_set(stats, '$.interrupted', "
                               "?) END WHERE id = ? AND status = 'streaming'",
                               (dumps(stats), why, r["assistant_message_id"]))
                self.store.run("UPDATE runs SET state = 'interrupted', ended = ? WHERE id = ?",
                               (now_ms(), r["id"]))
                self.emit(conv, {"type": "run.done", "run_id": r["id"],
                                 "message_id": r["assistant_message_id"],
                                 "stats": stats, "status": "interrupted"},
                          run_id=r["id"])
        self.store.run("UPDATE tasks SET status = 'interrupted' WHERE status = 'running'")
        if n:
            self.log(f"[start] {n} run(s) left running by the last chatd are interrupted")
        return n

    def _title(self, cid: str) -> None:
        """After the first answer: one short call names the chat (thinking off). The first line of
        the first message stays the title when the call fails or a person renamed the chat."""
        row = self.store.one("SELECT * FROM conversations WHERE id = ? AND deleted IS NULL", (cid,))
        if row is None or not row["title_auto"] or row["kind"] != "chat":
            return
        msgs = self._messages(cid)
        branch = [msgs[m] for m in self._path(msgs, row["head_id"])]
        if sum(1 for m in branch if m["role"] == "assistant") != 1:
            return
        user = render.text_of(branch[0]["parts"])[:2000]
        answer = render.text_of(branch[-1]["parts"])[:1000]
        settings = json.loads(row["settings"])
        body = {"model": settings.get("model") or "default", "stream": False, "max_tokens": 48,
                "chat_template_kwargs": {"enable_thinking": False},
                "messages": [{"role": "system", "content": "You name chats. You never continue "
                              "or answer them."},
                             {"role": "user", "content": "Here is the start of a chat between a "
                              "user and an assistant, between the lines.\n-----\nUser: "
                              f"{user[:600]}\nAssistant: {answer[:400]}\n-----\nReply with a "
                              "title for this chat of at most six words, and nothing else."}]}
        st, obj = self.engine.post_json("/v1/chat/completions", body,
                                        # raw: no routing, no packages, no state lines; a short
                                        # thinking-off call the engine serves on its side lane
                                        {"x-session-id": f"chat-title-{cid}",
                                         "X-QSE-Client": "chat-app", "X-Tandem": "raw"},
                                        timeout=60.0)
        if st != 200 or not isinstance(obj, dict):
            return
        try:
            text = obj["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            return
        if "</think>" in text:
            text = text.split("</think>", 1)[1]
        title = next((ln for ln in text.strip().split("\n") if ln.strip()), "").strip(" \"'.#*`")
        if not title or len(title.split()) > 8 or len(title) > 60 or "<think>" in title:
            return                       # not a title: the first line of the message stays
        conv = self.conv(cid)
        with conv.lock:
            cur = self.store.one("SELECT title_auto FROM conversations WHERE id = ?", (cid,))
            if cur is None or not cur[0]:
                return
            self.store.run("UPDATE conversations SET title = ?, version = version + 1 WHERE id = ?",
                           (title[:60], cid))
            v = self.store.one("SELECT version FROM conversations WHERE id = ?", (cid,))[0]
            self.emit(conv, {"type": "conversation.updated", "patch": {"title": title[:60],
                                                                       "version": v}})
            self.summary_changed(conv, title=title[:60], version=v)

    def _sweep(self) -> None:
        """Hourly: the trash after 7 days, delta events of final messages after 7 days, the daily
        backup."""
        while not self.stopping.wait(5.0 if not hasattr(self, "_swept") else 3600.0):
            self._swept = True
            try:
                self.sweep_once()
            except Exception as exc:                                # noqa: BLE001
                self.log(f"[sweep] {type(exc).__name__}")

    def sweep_once(self, now: int | None = None) -> None:
        now = now or now_ms()
        for r in self.store.all("SELECT id FROM conversations WHERE deleted IS NOT NULL AND "
                                "deleted < ? AND kind = 'chat'", (now - TRASH_KEEP_MS,)):
            self.purge(r["id"])
        self.store.run("DELETE FROM events WHERE type IN ('part.delta', 'tool.output') AND ts < ?",
                       (now - DELTA_KEEP_MS,))
        self.store.run("DELETE FROM global_events WHERE ts < ?", (now - DELTA_KEEP_MS,))
        if self.store.path != ":memory:":
            self.store.backup(os.path.join(self.data_dir, "backups"))

    def shutdown(self) -> None:
        """chatd stops: every run ends `interrupted` (Continue resumes it after the restart)."""
        self.stopping.set()
        self.sandboxes.stopping.set()
        with self.convs_lock:
            runs = [c.run for c in self.convs.values() if c.run is not None and c.run.active]
        for r in runs:
            r.stop("shutdown")
        for r in runs:
            if r.thread is not None:
                r.thread.join(10)


# ---------------------------------------------------------------------- helpers
_SUMMARY_KEYS = ("id", "title", "created_at", "updated_at", "pinned", "archived", "folder_id",
                 "status", "preview", "model", "profile", "turns", "now", "version", "unread")
_SETTINGS_KEYS = set(DEFAULT_SETTINGS)


def _text(body: dict, attachments: bool) -> str:
    """A message's text. An empty one goes nowhere unless files ride with it: the model would get
    a user turn with nothing in it (2026-10-12, a client that sent `content` instead of `text`)."""
    text = body.get("text")
    if text is not None and not isinstance(text, str):
        raise ApiError(400, "bad_request", "text must be a string")
    text = text or ""
    if not text.strip() and not attachments:
        hint = " (the field is `text`, not `content`)" if body.get("content") else ""
        raise ApiError(400, "empty_message", f"a message needs text or an attachment{hint}")
    return text


def _client_id(body: dict) -> str:
    cmid = body.get("client_msg_id")
    if not isinstance(cmid, str) or not cmid or len(cmid) > 80:
        raise ApiError(400, "bad_request", "client_msg_id is required (a ULID made on the device)")
    return cmid


def _model_name(m: dict, card: dict, ours: bool, cloud: bool) -> str:
    """The picker's name of a model: the engine's own label; a backend's model by its provider
    name, else its id (the card's label names the backend, which several models share)."""
    if ours:
        return card.get("label") or m["id"]
    name = card.get("name") or m["id"]
    return f"{name} (cloud)" if cloud else name


def _clean_settings(patch: dict) -> dict:
    if not isinstance(patch, dict):
        raise ApiError(400, "bad_request", "settings must be an object")
    out = {k: v for k, v in patch.items() if k in _SETTINGS_KEYS}
    if "profile" in out:
        out["profile"] = None           # the Chat tab's runs take the chat-app profile
    if "effort" in out and out["effort"] not in EFFORTS:
        raise ApiError(400, "bad_request", f"effort must be one of {', '.join(EFFORTS)}")
    if "max_steps" in out:
        out["max_steps"] = max(1, min(200, int(out["max_steps"] or DEFAULT_SETTINGS["max_steps"])))
    if "verify" in out:
        out["verify"] = bool(out["verify"])
    if "agents" in out:
        out["agents"] = bool(out["agents"])
    for k, hi in (("task_max_steps", 200), ("task_max_per_run", 32)):
        if k in out:
            out[k] = max(1, min(hi, int(out[k] or DEFAULT_SETTINGS[k])))
    if "permission_mode" in out and out["permission_mode"] not in (None, *tools_mod.MODES):
        raise ApiError(400, "bad_request", f"permission_mode must be one of {', '.join(tools_mod.MODES)}")
    if "web_search" in out:
        try:
            out["web_search"] = websearch.clean_override(out["web_search"])
        except websearch.Invalid as exc:
            raise ApiError(400, "bad_request", str(exc), {"field": exc.field}) from None
    return out


def _clean_ui(ui) -> dict:
    """The pane layout; a pane the client sent without an id keeps its provisional `kind:ref`
    (CONTRACT section 8, request 2), so every device sees the same ids."""
    if not isinstance(ui, dict):
        raise ApiError(400, "bad_request", "ui must be an object")
    panes = []
    for p in ui.get("panes") or []:
        if isinstance(p, dict) and p.get("kind"):
            q = {k: p[k] for k in ("id", "kind", "ref", "title", "pinned") if k in p}
            q["ref"] = str(q.get("ref") or "")
            q["id"] = str(q.get("id") or f"{q['kind']}:{q['ref']}")
            panes.append(q)
    split = ui.get("split", 0.6)
    try:
        split = max(0.35, min(0.75, float(split)))
    except (TypeError, ValueError):
        split = 0.6
    return {"panes": panes, "active": ui.get("active"), "split": split}


def _queued(q) -> dict:
    return {"id": q["id"], "text": q["text"], "created_at": iso(q["created"]),
            "client_msg_id": q["client_msg_id"]}


def _attachment(att_id, name, kind, mime, size, tokens) -> dict:
    url = f"/v1/chat-app/attachments/{att_id}"
    # an image's thumbnail on every device is the file itself, from this origin
    out = {"id": att_id, "name": name, "kind": kind, "bytes": int(size), "mime": mime,
           "url": url, "tokens": tokens,
           "preview": url if kind == "image" and size <= 5 * 1024 * 1024 else None}
    if kind == "text" and tokens is not None:
        # what the composer can say before the send places it (SRV-487, VIS-169)
        out["inline"] = int(tokens) <= attach_mod.INLINE_MAX_TOKENS
    return out


def _empty_stats(finish: str) -> dict:
    return {"finish": finish, "ttft_ms": None, "decode_tps": None, "total_ms": None,
            "prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0, "cached_tokens": 0,
            "tokens_per_round": None, "acceptance": None}


def _usage(branch: list[dict]) -> dict:
    out = {"turns": 0, "prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0,
           "cached_tokens": 0, "tool_calls": 0, "wall_s": 0, "cost_usd": None,
           "workers": {"count": 0, "prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0}}
    wall = 0.0
    for m in branch:
        s = m.get("stats")
        if m["role"] != "assistant":
            continue
        # every call the model made, whatever became of it (a stopped turn has no stats)
        out["tool_calls"] += sum(1 for p in m.get("parts") or [] if p.get("type") == "tool_call")
        if not s:
            continue
        out["turns"] += 1
        for k in ("prompt_tokens", "completion_tokens", "reasoning_tokens", "cached_tokens"):
            out[k] += int(s.get(k) or 0)
        w = s.get("workers") or {}
        out["workers"]["count"] += int(w.get("count") or 0)
        for k in ("prompt_tokens", "completion_tokens", "cached_tokens"):   # its workers' calls
            out[k] += int(w.get(k) or 0)
            out["workers"][k] += int(w.get(k) or 0)
        if s.get("cost_usd") is not None:         # a cloud model's turns, at the proxy's prices
            out["cost_usd"] = round((out["cost_usd"] or 0) + float(s["cost_usd"]), 6)
        wall += float(s.get("total_ms") or 0) / 1000
    out["wall_s"] = round(wall)
    return out


def _usage_of(m: dict, run=None) -> list[dict]:
    """An assistant message's calls' counts by step: its stats', then the run in flight's on it,
    whose calls are in no stats until it ends (the stats are an earlier run's: Continue goes on in
    the same message)."""
    usage = list((m.get("stats") or {}).get("step_usage") or [])
    if run is not None and m.get("id") == run.message_id:
        usage += run.step_usage
    by = {int(u.get("step") or 0): u for u in usage}
    return [by[k] for k in sorted(by)]


def _turn_thinking(m: dict, usage: list[dict], from_step: int = 1, upto: int | None = None,
                   total: int | None = None) -> int:
    """The thinking of an agent turn that its next request carries: its steps from `from_step`
    (a compaction inside the run took the ones before it into the summary) to `upto`, each by its
    call's count (`usage`), else estimated from its text; `total`, the turn's own count, for a
    turn saved before the calls' counts."""
    exact = {int(u["step"]): int(u["reasoning_tokens"]) for u in usage
             if u.get("step") and u.get("reasoning_tokens") is not None}
    if not exact and from_step <= 1 and total is not None:
        return int(total)
    chars: dict[int, int] = {}
    for p in m.get("parts") or []:
        if p.get("type") == "thinking":
            k = int(p.get("step") or 1)
            chars[k] = chars.get(k, 0) + len(p.get("text") or "")
    return sum(exact[k] if k in exact else (chars[k] + 3) // 4 for k in set(chars) | set(exact)
               if k >= from_step and (upto is None or k <= upto))


def _after_step(m: dict, step: int) -> int:
    """What the next request carries of an agent turn after its call at `step`, estimated: that
    step's tool results, the later steps' text, calls and results (their thinking apart)."""
    n = 0
    for p in m.get("parts") or []:
        k = int(p.get("step") or 1)
        if k < step:
            continue
        if p.get("type") == "tool_call":
            n += ctx_mod.estimate_tokens(render.result_text(p))
            if k > step:
                n += ctx_mod.estimate_tokens(p.get("input") or "")
        elif p.get("type") == "text" and k > step:
            n += ctx_mod.estimate_tokens(p.get("text") or "")
    return n


def _branch_tokens(branch: list[dict], run=None,
                   step_from: tuple[int, int] | None = None) -> int:
    """What the next request carries: the last answered request's prompt and answer (the engine's
    own counts), plus an estimate for what came after it, less the turn's thinking (the panel
    counts it apart, `_turn_thinking`). `run`: the run in flight; `step_from`: (message index,
    step) of a compaction inside a run. Live (2026-10-17) the panel said 45,778 tokens while the
    engine got 215,247: the message's stats were those of the run before a Continue, the steps
    since were in none, and the whole turn's thinking came off, that of the 42 steps in the
    summary too."""
    base, since = 0, 0
    for i, m in enumerate(branch):
        s = m.get("stats")
        usage = _usage_of(m, run) if m["role"] == "assistant" else []
        if m["role"] == "assistant" and (usage or s and s.get("prompt_tokens")):
            # the last call's prompt (the earlier calls' answers are in it) and answer, less the
            # turn's thinking; a turn's completion_tokens are every call's (live 2026-10-16: an
            # agent turn of 13,270 counted ~11,000 tokens twice)
            last = usage[-1] if usage else {}
            reasoning = int((s or {}).get("reasoning_tokens") or 0)
            since = 0
            if last.get("completion_tokens") is not None:
                step = int(last.get("step") or 0)
                frm = step_from[1] if step_from and step_from[0] == i else 1
                prompt = int(last.get("prompt_tokens") or (s or {}).get("prompt_tokens") or 0)
                base = prompt + int(last["completion_tokens"]) - _turn_thinking(
                    m, usage, frm, step or None, reasoning)
                since = _after_step(m, step) if step else 0
            elif int(s.get("steps") or 1) > 1:     # a turn saved before the calls' own counts
                base = int(s["prompt_tokens"]) + ctx_mod.estimate_tokens(
                    render.text_of(render.steps(m.get("parts") or [])[-1]))
            else:
                base = int(s["prompt_tokens"]) + int(s.get("completion_tokens") or 0) - reasoning
            base = max(0, base)
        else:
            since += ctx_mod.estimate_tokens(render.text_of(m.get("parts") or []))
    return base + since
