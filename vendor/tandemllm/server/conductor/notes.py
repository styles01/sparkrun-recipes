"""Memory notes: the facts the model keeps between turns and sessions, as files (2026-10-05).

The design: Memo "Memory for TandemLLM: design advice" (2026-10-05), built on Harmony 5's file
format and the transcripts' root (server/conductor/transcripts.py). One note is one markdown file
with a frontmatter, one fact per file. The files are the truth: the dashboard reads and writes
them, the index (`INDEX.md`) and the search are generated from them, and a person can read or edit
them by hand.

THE LAYOUT, under the memory root (beside the transcripts, `~/.qwen38-spark-engine/memory` on the
box, `0600` files, refused inside a git work tree as the transcripts are):

    notes/<user>/
      INDEX.md                               generated: one line per live note, by scope and kind
      user/<kind>-<slug>--<id>.md            about the person: preferences, rulings, answers
      projects/<name>-<hash8>/<kind>-...md   about one working directory
      sessions/<session dir>/<kind>-...md    true for one conversation only
      revisions/<id>/<n>.md                  every earlier body of a note
      archive/<yyyy-mm>/<file>               archived notes with their reason (30 days)
      usage.json                             injections per note and day (counts, no text)
    proposals/<name>/<n>.md                  P-M3: package proposals (server/conductor/proposals.py)

`<user>` is `owner` on the engine (one person on the LAN); a request relayed by the proxy names
the proxy user. A request without a user reads and writes nothing.

THE RULES, in code (refusals, never sentences in the prompt): a title of at most 120 characters,
a body of at most 400 words, 20 notes a session, 500 live notes a user; a write over a cap is
refused, never clipped. Every title and body passes `redact()`; a body that is mostly a secret
(under 20 characters left) is refused. A `rule` or `ruling` about the user needs a user message in
the turn (not a tool result). The model writes `provisional` notes only; ratify, pin and delete
are the dashboard's. Before an `add` lands, the same scope is searched: the normalised title, then
BM25's top 5 over title and body with a word-overlap similarity at or above `dedupe_threshold`;
a hit turns the add into a revision of the hit (old body under `revisions/`, a changelog line). A
model revision of a ratified note sends it back to provisional. Forgetting is `archive`; the
nightly job deletes archives older than 30 days.

Standard library only, like the rest of the conductor. Thread-safe: one lock a store.
"""

from __future__ import annotations

import calendar
import hashlib
import io
import json
import os
import re
import secrets
import shutil
import threading
import time
import zipfile

from server.conductor import redact as redact_mod
from server.conductor.libsearch import TextIndex, words
from server.conductor.transcripts import dir_name, in_repository

KINDS = ("rule", "ruling", "gotcha", "runbook", "program_state", "reference", "registry",
         "archive")
# the dashboard's plain words for the kinds
KIND_WORDS = {"rule": "how I want things done", "ruling": "a decision that was taken",
              "gotcha": "what went wrong once", "runbook": "how to do a thing",
              "program_state": "where the work stands", "reference": "a fact to look up",
              "registry": "a list of things", "archive": "kept for the record"}
SCOPES = ("user", "project", "session")
STATUSES = ("provisional", "ratified")
WRITERS = ("qwen-inline", "qwen-writer", "api", "human")
OPS = ("add", "update", "archive")
CAPS = {"title_chars": 120, "body_words": 400, "session_notes": 20, "user_notes": 500,
        "index_lines": 40, "tags": 8}
ARCHIVE_DAYS = 30
SECRET_LEFT = 20
WEIGHTS = {"title": 3, "tags": 2, "body": 1}
RECHECK_S = 1.0
_SLUG = re.compile(r"[^a-z0-9]+")
_ID = re.compile(r"^[0-9a-f]{12}$")
_USER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_FRONT = re.compile(r"\A---\n(.*?)\n---\n?", re.S)
CHANGELOG = "## Changelog"
LIST_FIELDS = ("tags", "sources")


class Refused(ValueError):
    """A write the store does not take; `why` is the journal's word (schema, cap, secret, ...)."""

    def __init__(self, why: str, message: str):
        super().__init__(message)
        self.why = why


def new_id(clock=time.time) -> str:
    """12 hex characters, time first: short enough for the model to copy back."""
    return f"{int(clock()) & 0xffffffff:08x}{secrets.token_hex(2)}"


def slug(text: str, n: int = 48) -> str:
    s = _SLUG.sub("-", (text or "").lower()).strip("-")
    return (s[:n].rstrip("-") or "note")


def norm_title(text: str) -> str:
    return " ".join(words(text or ""))


def project_dir(path: str) -> str:
    """A working directory as a directory name: its last part and a hash of the whole path."""
    p = (path or "").rstrip("/")
    base = slug(os.path.basename(p) or "root", 32)
    return f"{base}-{hashlib.sha256(p.encode()).hexdigest()[:8]}"


def user_id(label: str | None) -> str | None:
    """A user's directory name: the label itself when it is a safe name, else its slug and a hash
    of it (a proxy key's label may hold spaces or quotes; two labels never share a directory)."""
    if not label:
        return None
    if valid_user(label):
        return label
    return f"{slug(label, 40)}-{hashlib.sha256(label.encode()).hexdigest()[:8]}"


def valid_user(user: str | None) -> bool:
    return bool(user) and bool(_USER.match(user)) and user not in (".", "..")


def similarity(a_title: str, a_body: str, b_title: str, b_body: str) -> float:
    """Word overlap of two notes: the titles' Jaccard and the bodies' Jaccard, the higher counting
    twice (a reworded body with the same title, or the same body under a new title, is a hit)."""
    def jac(x: str, y: str) -> float:
        a, b = set(words(x)), set(words(y))
        return len(a & b) / len(a | b) if a and b else 0.0
    t, b = jac(a_title, b_title), jac(a_body, b_body)
    return (2 * max(t, b) + min(t, b)) / 3


def _now_iso(t: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))


def _write_private(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), mode=0o700, exist_ok=True)
    tmp = path + ".tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(text)
    os.replace(tmp, path)


# ------------------------------------------------------------------------------ the file format
class Note:
    __slots__ = ("id", "title", "kind", "scope", "project", "session", "status", "writer",
                 "confidence", "tags", "created", "updated", "verify_by", "pinned", "sources",
                 "body", "changelog", "path", "archived", "reason", "revisions")

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))
        self.tags = list(self.tags or [])
        self.sources = list(self.sources or [])
        self.changelog = list(self.changelog or [])
        self.pinned = bool(self.pinned)
        self.revisions = int(self.revisions or 0)

    def meta(self) -> dict:
        """The frontmatter as a dict (what the list views show; no body)."""
        out = {k: getattr(self, k) for k in ("id", "title", "kind", "scope", "project", "session",
                                              "status", "writer", "confidence", "tags", "created",
                                              "updated", "verify_by", "pinned", "sources",
                                              "revisions")}
        if self.archived:
            out.update(archived=self.archived, reason=self.reason)
        return out

    def full(self) -> dict:
        return dict(self.meta(), body=self.body, changelog=list(self.changelog))

    def stale(self, today: str) -> bool:
        return self.kind == "program_state" and bool(self.verify_by) and self.verify_by < today

    def render(self) -> str:
        lines = ["---"]
        for k in ("id", "title", "kind", "scope", "project", "session", "status", "writer",
                  "confidence", "tags", "created", "updated", "verify_by", "pinned", "sources",
                  "revisions", "archived", "reason"):
            v = getattr(self, k)
            if v is None or v == [] or v == "" or (k in ("pinned",) and not v) or \
                    (k == "revisions" and not v):
                continue
            if k in LIST_FIELDS:
                v = ", ".join(str(x) for x in v)
            elif isinstance(v, bool):
                v = "true" if v else "false"
            lines.append(f"{k}: {str(v).replace(chr(10), ' ')}")
        lines.append("---")
        out = "\n".join(lines) + "\n" + (self.body or "").strip() + "\n"
        if self.changelog:
            out += "\n" + CHANGELOG + "\n" + "\n".join(f"- {c}" for c in self.changelog) + "\n"
        return out

    @classmethod
    def parse(cls, text: str, path: str | None = None) -> "Note | None":
        m = _FRONT.match(text or "")
        if not m:
            return None
        kw: dict = {"path": path}
        for line in m.group(1).splitlines():
            k, sep, v = line.partition(":")
            k, v = k.strip(), v.strip()
            if not sep or k not in cls.__slots__:
                continue
            if k in LIST_FIELDS:
                kw[k] = [x.strip() for x in v.split(",") if x.strip()]
            elif k == "pinned":
                kw[k] = v.lower() == "true"
            elif k == "confidence":
                try:
                    kw[k] = float(v)
                except ValueError:
                    pass
            elif k == "revisions":
                kw[k] = int(v) if v.isdigit() else 0
            else:
                kw[k] = v
        rest = text[m.end():]
        i = rest.find("\n" + CHANGELOG + "\n")
        if rest.startswith(CHANGELOG + "\n"):
            i = -1
            body, log = "", rest[len(CHANGELOG) + 1:]
        elif i >= 0:
            body, log = rest[:i], rest[i + len(CHANGELOG) + 2:]
        else:
            body, log = rest, ""
        kw["body"] = body.strip()
        kw["changelog"] = [ln[2:].strip() for ln in log.splitlines() if ln.startswith("- ")]
        if not kw.get("id") or not _ID.match(kw["id"]):
            return None
        return cls(**kw)


# ------------------------------------------------------------------------------ the store
class _UserCache:
    __slots__ = ("sig", "checked", "notes", "index", "order")

    def __init__(self):
        self.sig, self.checked = None, -1e9
        self.notes: dict[str, Note] = {}
        self.index = TextIndex()
        self.order: list[str] = []


class NoteStore:
    """The notes of every user under `<root>/notes`. `settings()`: `settings.memory.notes`
    (server/conductor/schema.py NOTES), read on every write for the dedupe threshold."""

    def __init__(self, root: str, *, settings=lambda: {}, journal=None, clock=time.time,
                 mono=time.monotonic):
        self.root = os.path.realpath(os.path.expanduser(root))
        self.dir = os.path.join(self.root, "notes")
        self.settings, self.journal, self.clock, self.mono = settings, journal, clock, mono
        self.repo = in_repository(self.root)
        self._lock = threading.RLock()
        self._users: dict[str, _UserCache] = {}
        self._usage: dict[str, dict] = {}
        self._usage_dirty: set = set()
        self.stats = {"writes": 0, "refused": 0, "revised": 0, "archived": 0, "redactions": 0,
                      "searches": 0, "doubted": 0}
        if self.repo:
            print(f"[memory] REFUSING notes: {self.root} is inside the git work tree "
                  f"{self.repo}; memories never go into a repository", flush=True)

    # ---------------------------------------------------------------- paths and loading
    def user_dir(self, user: str) -> str:
        if not valid_user(user):
            raise Refused("user", f"not a user name: {user!r}")
        return os.path.join(self.dir, user)

    def users(self) -> list[str]:
        try:
            return sorted(u for u in os.listdir(self.dir) if valid_user(u) and
                          os.path.isdir(os.path.join(self.dir, u)))
        except OSError:
            return []

    def _scope_dir(self, user: str, scope: str, project: str | None, session: str | None) -> str:
        base = self.user_dir(user)
        if scope == "user":
            return os.path.join(base, "user")
        if scope == "project":
            return os.path.join(base, "projects", project_dir(project or ""))
        return os.path.join(base, "sessions", dir_name(session or ""))

    def _live_files(self, user: str) -> list[str]:
        base = self.user_dir(user)
        out = []
        for sub in ("user", "projects", "sessions"):
            top = os.path.join(base, sub)
            if not os.path.isdir(top):
                continue
            for dp, _, fs in os.walk(top):
                out += [os.path.join(dp, f) for f in fs if f.endswith(".md")]
        return sorted(out)

    def _cache(self, user: str) -> _UserCache:
        """The user's live notes and their index, rebuilt when a file moved (looked at once a
        second at most)."""
        with self._lock:
            c = self._users.get(user)
            if c is None:
                c = self._users[user] = _UserCache()
            now = self.mono()
            if c.sig is not None and now - c.checked < RECHECK_S:
                return c
            c.checked = now
            files = self._live_files(user)
            sig = []
            for f in files:
                try:
                    st = os.stat(f)
                    sig.append((f, st.st_mtime_ns, st.st_size))
                except OSError:
                    pass
            sig = tuple(sig)
            if sig == c.sig:
                return c
            notes = {}
            for f, _, _ in sig:
                try:
                    with open(f) as fh:
                        n = Note.parse(fh.read(), f)
                except OSError:
                    n = None
                if n is not None:
                    notes[n.id] = n
            c.notes = notes
            c.order = sorted(notes)
            c.index.build([{"title": notes[i].title or "", "tags": " ".join(notes[i].tags),
                            "body": notes[i].body or ""} for i in c.order], WEIGHTS)
            c.sig = sig
            return c

    def _touch(self, user: str) -> None:
        with self._lock:
            c = self._users.get(user)
            if c is not None:
                c.sig, c.checked = None, -1e9

    # ---------------------------------------------------------------- reads
    def notes(self, user: str) -> list[Note]:
        if not valid_user(user) or self.repo:
            return []
        return [self._cache(user).notes[i] for i in self._cache(user).order]

    def get(self, user: str, nid: str) -> Note | None:
        if not valid_user(user) or not _ID.match(nid or ""):
            return None
        return self._cache(user).notes.get(nid)

    def visible(self, n: Note, project: str | None, session: str | None) -> bool:
        """Whether a note belongs to a request of this project and session."""
        if n.scope == "user":
            return True
        if n.scope == "project":
            return bool(project) and (n.project or "").rstrip("/") == project.rstrip("/")
        return bool(session) and n.session == session

    def search(self, user: str, query: str, *, project: str | None = None,
               session: str | None = None, k: int = 3, statuses=STATUSES, scopes=SCOPES,
               all_scopes: bool = False, today: str | None = None) -> list[tuple[Note, float]]:
        """BM25 over the notes this request may see (or every note of the user, `all_scopes`:
        the dashboard's search), live statuses only, stale `program_state` notes left out. A
        session note ranks above a project note of the same score, a project note above a user
        note."""
        if not valid_user(user) or self.repo or not (query or "").strip():
            return []
        c = self._cache(user)
        self.stats["searches"] += 1
        today = today or time.strftime("%Y-%m-%d", time.gmtime(self.clock()))
        sc = c.index.scores(query)
        rank = {"session": 2, "project": 1, "user": 0}
        out = []
        for i, s in enumerate(sc):
            if s <= 0:
                continue
            n = c.notes[c.order[i]]
            if n.status not in statuses or n.scope not in scopes or n.stale(today):
                continue
            if not all_scopes and not self.visible(n, project, session):
                continue
            out.append((n, s))
        out.sort(key=lambda x: (-round(x[1], 6), -rank.get(x[0].scope, 0), x[0].id))
        return out[:max(0, int(k))]

    def tree(self, user: str) -> dict:
        """The dashboard's tree: scopes with their counts and provisional counts."""
        today = time.strftime("%Y-%m-%d", time.gmtime(self.clock()))
        nodes: dict[str, dict] = {}
        for n in self.notes(user):
            key = ("user" if n.scope == "user" else
                   f"projects/{project_dir(n.project or '')}" if n.scope == "project" else
                   f"sessions/{dir_name(n.session or '')}")
            d = nodes.setdefault(key, {"node": key, "scope": n.scope,
                                       "label": (n.project if n.scope == "project" else
                                                 n.session if n.scope == "session" else "user"),
                                       "count": 0, "provisional": 0, "stale": 0})
            d["count"] += 1
            d["provisional"] += n.status == "provisional"
            d["stale"] += n.stale(today)
        return {"user": user, "nodes": sorted(nodes.values(), key=lambda d: (
            SCOPES.index(d["scope"]), d["node"]))}

    def node_of(self, n: Note) -> str:
        return ("user" if n.scope == "user" else
                f"projects/{project_dir(n.project or '')}" if n.scope == "project" else
                f"sessions/{dir_name(n.session or '')}")

    # ---------------------------------------------------------------- the index
    def index_lines(self, user: str, project: str | None, session: str | None,
                    max_lines: int = CAPS["index_lines"]) -> list[tuple[str, list[str]]]:
        """(scope heading, lines) for the notes this request may see: pinned first, then ratified,
        then the newest; at most `max_lines` a scope, the rest counted."""
        today = time.strftime("%Y-%m-%d", time.gmtime(self.clock()))
        by: dict[str, list[Note]] = {s: [] for s in SCOPES}
        for n in self.notes(user):
            if n.status in STATUSES and not n.stale(today) and self.visible(n, project, session):
                by[n.scope].append(n)
        out = []
        for scope in SCOPES:
            ns = sorted(by[scope], key=lambda n: (not n.pinned, n.status != "ratified",
                                                  -(_ts(n.updated))))
            if not ns:
                continue
            lines = [line_of(n) for n in ns[:max_lines]]
            if len(ns) > max_lines:
                lines.append(f"... and {len(ns) - max_lines} more; search them")
            out.append((scope, lines))
        return out

    def write_index(self, user: str) -> str | None:
        """INDEX.md of every live note of the user (all projects and sessions), regenerated."""
        if self.repo or not valid_user(user):
            return None
        ns = self.notes(user)
        lines = [f"# Memory index ({user})", "",
                 f"Generated {_now_iso(self.clock())}; {len(ns)} live notes. Edit the notes, "
                 "not this file.", ""]
        groups: dict[str, list[Note]] = {}
        for n in ns:
            groups.setdefault(self.node_of(n), []).append(n)
        for node in sorted(groups, key=lambda k: (not k.startswith("user"), k)):
            lines.append(f"## {node}")
            for n in sorted(groups[node], key=lambda n: (n.kind or "", n.title or "")):
                lines.append(line_of(n))
            lines.append("")
        path = os.path.join(self.user_dir(user), "INDEX.md")
        _write_private(path, "\n".join(lines))
        return path

    # ---------------------------------------------------------------- the write path
    def write(self, user: str, op: dict, *, writer: str, project: str | None = None,
              session: str | None = None, request_id: str | None = None,
              from_user: bool = True) -> dict:
        """One op (add, update, archive) from the model's block or the model's route. Returns
        `{op, id, title, scope, kind, revised?}` or raises `Refused`."""
        if self.repo:
            raise Refused("refused_root", "the memory root is inside a git work tree")
        if not valid_user(user):
            raise Refused("no_user", "no user for this request: nothing is written")
        kind_op = (op.get("op") or "add").strip()
        if kind_op not in OPS:
            raise Refused("schema", f"op: one of {', '.join(OPS)}")
        with self._lock:
            if kind_op == "archive":
                n = self._own(user, op.get("id"), project, session)
                reason = (op.get("reason") or "").strip() or "archived by the model"
                if n.status == "ratified" and writer != "human":
                    # the user confirmed it: the model's doubt makes it provisional again, with the
                    # reason in its changelog, and the user archives it on the Notes tab. A
                    # fresh session that checked the files archived the demo's ratified note
                    #; ratify, pin and delete are the dashboard's
                    self._doubt(user, n, reason, writer)
                    self.stats["doubted"] += 1
                    return {"op": "archive", "id": n.id, "title": n.title, "scope": n.scope,
                            "kind": n.kind, "kept": "provisional"}
                self.archive(user, n.id, reason, by=writer)
                self.stats["archived"] += 1
                return {"op": "archive", "id": n.id, "title": n.title, "scope": n.scope,
                        "kind": n.kind}
            title, body, counts = self._clean(op)
            kind = (op.get("kind") or "").strip()
            scope = (op.get("scope") or "").strip()
            tags = _tags(op.get("tags"))
            if kind_op == "update":
                n = self._own(user, op.get("id"), project, session)
                kind = kind or n.kind
                if kind not in KINDS:
                    raise Refused("schema", f"kind: one of {', '.join(KINDS)}")
                self._rule_check(kind, n.scope, from_user)
                self._revise(user, n, title or n.title, body, kind, tags or n.tags, writer,
                             request_id, "updated")
                self._done(counts)
                return {"op": "update", "id": n.id, "title": n.title, "scope": n.scope,
                        "kind": n.kind, "redacted": counts}
            if kind not in KINDS:
                raise Refused("schema", f"kind: one of {', '.join(KINDS)}")
            if scope not in SCOPES:
                raise Refused("schema", f"scope: one of {', '.join(SCOPES)}")
            if not title:
                raise Refused("schema", "title: one line, what the note says")
            rescoped = None
            if scope == "project" and not project:
                # SRV-277 (2026-10-08): a conversation without a working directory (every Open
                # WebUI chat, a curl script) has no project, and the note used to be refused and
                # shown to the user as markup. TandemLLM serves one user and a "remember" from
                # any client is read back in the next conversation (SRV-173), so the note lands
                # in the user's scope, which every later conversation sees; a session note
                # would be gone in the next chat. The result and the journal say `rescoped`.
                scope, rescoped = "user", "project"
            if scope == "session" and not session:
                raise Refused("no_session", "no session for this request")
            self._rule_check(kind, scope, from_user)
            hit = self._dedupe(user, scope, project, session, title, body)
            if hit is not None:
                self._revise(user, hit, title, body, kind, tags or hit.tags, writer, request_id,
                             "revised (same fact written again)")
                self.stats["revised"] += 1
                self._done(counts)
                return {"op": "add", "id": hit.id, "title": hit.title, "scope": hit.scope,
                        "kind": hit.kind, "revised": True, "redacted": counts,
                        **({"rescoped": rescoped} if rescoped else {})}
            live = self.notes(user)
            if len(live) >= CAPS["user_notes"]:
                raise Refused("cap", f"{CAPS['user_notes']} live notes: archive some first")
            if scope == "session" and sum(1 for n in live if n.scope == "session" and
                                          n.session == session) >= CAPS["session_notes"]:
                raise Refused("cap", f"{CAPS['session_notes']} notes in this session")
            now = self.clock()
            n = Note(id=new_id(self.clock), title=title, kind=kind, scope=scope,
                     project=project if scope == "project" else None,
                     session=session if scope == "session" else None,
                     status="ratified" if writer == "human" else "provisional", writer=writer,
                     tags=tags, created=_now_iso(now), updated=_now_iso(now),
                     verify_by=_date(op.get("verify_by")), body=body,
                     sources=[request_id] if request_id else [],
                     changelog=[f"{_now_iso(now)[:16].replace('T', ' ')} created by {writer}"
                                + (f" (request {request_id})" if request_id else "")])
            d = self._scope_dir(user, scope, project, session)
            n.path = os.path.join(d, f"{kind}-{slug(title)}--{n.id}.md")
            _write_private(n.path, n.render())
            self._touch(user)
            self.stats["writes"] += 1
            self._done(counts)
            out = {"op": "add", "id": n.id, "title": n.title, "scope": scope, "kind": kind,
                   "redacted": counts}
            if rescoped:
                out["rescoped"] = rescoped
            return out

    def _done(self, counts: dict) -> None:
        if counts:
            self.stats["redactions"] += sum(counts.values())

    def _clean(self, op: dict) -> tuple[str, str, dict]:
        """Title and body through the caps and the secret refusal."""
        title = " ".join(str(op.get("title") or "").split())
        body = str(op.get("body") or "").strip()
        if len(title) > CAPS["title_chars"]:
            raise Refused("cap", f"title: at most {CAPS['title_chars']} characters "
                                 f"(this one has {len(title)})")
        nw = len(body.split())
        if nw > CAPS["body_words"]:
            raise Refused("cap", f"body: at most {CAPS['body_words']} words (this one has {nw})")
        if not body:
            raise Refused("schema", "body: the fact itself, at least one sentence")
        title, c1 = redact_mod.redact(title)
        body, c2 = redact_mod.redact(body)
        counts = redact_mod.merge(dict(c1), c2)
        if counts:
            left = re.sub(r"\[redacted:[a-z_]+\]", "", body)
            if len(left.strip()) < SECRET_LEFT:
                raise Refused("secret", "the body is mostly a secret; secrets are never kept")
        return title, body, counts

    @staticmethod
    def _rule_check(kind: str, scope: str, from_user: bool) -> None:
        if kind in ("rule", "ruling") and scope == "user" and not from_user:
            raise Refused("rule_from_tool", "a rule about the user needs the user's own message "
                                            "in this turn, not a tool result")

    def _own(self, user: str, nid, project, session) -> Note:
        n = self.get(user, str(nid or "").strip())
        if n is None or not self.visible(n, project, session):
            raise Refused("unknown_id", f"no note {nid!r} in this conversation's memory")
        return n

    def _dedupe(self, user, scope, project, session, title, body) -> Note | None:
        nt = norm_title(title)
        same = [n for n in self.notes(user) if n.scope == scope and
                self.visible(n, project, session) and n.status in STATUSES]
        for n in same:
            if norm_title(n.title) == nt and nt:
                return n
        thr = float((self.settings() or {}).get("dedupe_threshold", 0.6))
        hits = self.search(user, f"{title}\n{body}", project=project, session=session, k=5,
                           scopes=(scope,))
        best, bs = None, 0.0
        for n, _ in hits:
            s = similarity(title, body, n.title or "", n.body or "")
            if s >= thr and s > bs:
                best, bs = n, s
        return best

    def _revise(self, user, n: Note, title, body, kind, tags, writer, request_id, what) -> None:
        self._keep_revision(user, n)
        now = _now_iso(self.clock())
        changed = (body.strip() != (n.body or "").strip()) or title != n.title
        n.title, n.body, n.kind, n.tags = title, body, kind, list(tags)[:CAPS["tags"]]
        n.updated, n.writer = now, writer
        if writer == "human":
            n.status = "ratified"
        elif n.status == "ratified" and changed:
            n.status = "provisional"
        if request_id and request_id not in n.sources:
            n.sources = (n.sources + [request_id])[-8:]
        n.changelog.append(f"{now[:16].replace('T', ' ')} {what} by {writer}"
                           + (f" (request {request_id})" if request_id else ""))
        _write_private(n.path, n.render())
        self._touch(user)

    def _keep_revision(self, user: str, n: Note) -> None:
        n.revisions = int(n.revisions or 0) + 1
        d = os.path.join(self.user_dir(user), "revisions", n.id)
        _write_private(os.path.join(d, f"{n.revisions}.md"), n.render())

    # ---------------------------------------------------------------- the dashboard's writes
    def edit(self, user: str, nid: str, fields: dict) -> Note:
        """A person's edit: a new revision, `ratified`, writer `human`."""
        with self._lock:
            n = self.get(user, nid)
            if n is None:
                raise Refused("unknown_id", f"no note {nid!r}")
            title, body, counts = self._clean({"title": fields.get("title", n.title),
                                               "body": fields.get("body", n.body)})
            kind = fields.get("kind") or n.kind
            if kind not in KINDS:
                raise Refused("schema", f"kind: one of {', '.join(KINDS)}")
            if "verify_by" in fields:
                n.verify_by = _date(fields.get("verify_by"))
            self._revise(user, n, title, body, kind,
                         _tags(fields["tags"]) if "tags" in fields else n.tags, "human", None,
                         "edited")
            self._done(counts)
            return n

    def set_status(self, user: str, nid: str, status: str | None = None,
                   pinned: bool | None = None) -> Note:
        with self._lock:
            n = self.get(user, nid)
            if n is None:
                raise Refused("unknown_id", f"no note {nid!r}")
            now = _now_iso(self.clock())
            if status is not None:
                if status not in STATUSES:
                    raise Refused("schema", f"status: one of {', '.join(STATUSES)}")
                n.status = status
                n.changelog.append(f"{now[:16].replace('T', ' ')} {status} by human")
            if pinned is not None:
                if pinned and not (n.status == "ratified" and n.kind in ("rule", "ruling")):
                    raise Refused("schema", "only a ratified rule or ruling can be pinned")
                n.pinned = bool(pinned)
                n.changelog.append(f"{now[:16].replace('T', ' ')} "
                                   f"{'pinned' if pinned else 'unpinned'} by human")
            n.updated = now
            _write_private(n.path, n.render())
            self._touch(user)
            return n

    def _doubt(self, user: str, n: Note, reason: str, writer: str) -> None:
        now = _now_iso(self.clock())
        n.status = "provisional"
        n.changelog.append(f"{now[:16].replace('T', ' ')} provisional again: {writer} asked to "
                           f"archive it ({' '.join(reason.split())[:200]})")
        n.updated = now
        _write_private(n.path, n.render())
        self._touch(user)

    def archive(self, user: str, nid: str, reason: str, by: str = "human") -> dict:
        with self._lock:
            n = self.get(user, nid)
            if n is None:
                raise Refused("unknown_id", f"no note {nid!r}")
            now = self.clock()
            n.archived, n.reason = _now_iso(now), " ".join(str(reason).split())[:200]
            n.changelog.append(f"{_now_iso(now)[:16].replace('T', ' ')} archived by {by}")
            d = os.path.join(self.user_dir(user), "archive", time.strftime("%Y-%m",
                                                                           time.gmtime(now)))
            dest = os.path.join(d, os.path.basename(n.path))
            _write_private(dest, n.render())
            try:
                os.remove(n.path)
            except OSError:
                pass
            self._touch(user)
            return {"id": n.id, "archived": n.archived, "reason": n.reason}

    def delete(self, user: str, nid: str) -> bool:
        """At once: the file, its revisions, and an archived copy."""
        with self._lock:
            n = self.get(user, nid)
            gone = False
            if n is not None:
                try:
                    os.remove(n.path)
                    gone = True
                except OSError:
                    pass
            for a in self.archived(user):
                if a.id == nid:
                    try:
                        os.remove(a.path)
                        gone = True
                    except OSError:
                        pass
            shutil.rmtree(os.path.join(self.user_dir(user), "revisions", nid),
                          ignore_errors=True)
            self._touch(user)
            return gone

    def archived(self, user: str) -> list[Note]:
        d = os.path.join(self.user_dir(user), "archive")
        out = []
        for dp, _, fs in os.walk(d):
            for f in fs:
                if f.endswith(".md"):
                    p = os.path.join(dp, f)
                    try:
                        with open(p) as fh:
                            n = Note.parse(fh.read(), p)
                    except OSError:
                        n = None
                    if n is not None:
                        out.append(n)
        return sorted(out, key=lambda n: n.archived or "", reverse=True)

    def revisions(self, user: str, nid: str) -> list[dict]:
        d = os.path.join(self.user_dir(user), "revisions", nid)
        out = []
        try:
            names = sorted(os.listdir(d), key=lambda f: int(f.split(".")[0]) if
                           f.split(".")[0].isdigit() else 0)
        except OSError:
            return []
        for f in names:
            try:
                with open(os.path.join(d, f)) as fh:
                    n = Note.parse(fh.read())
            except OSError:
                n = None
            if n is not None:
                out.append({"n": int(f.split(".")[0]), "title": n.title, "body": n.body,
                            "kind": n.kind, "status": n.status, "writer": n.writer,
                            "updated": n.updated})
        return out

    def export(self, user: str) -> bytes:
        """A zip of the user's directory as it is on disk."""
        base = self.user_dir(user)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for dp, _, fs in os.walk(base):
                for f in fs:
                    if f.endswith(".tmp"):
                        continue
                    p = os.path.join(dp, f)
                    z.write(p, os.path.relpath(p, os.path.dirname(base)))
        return buf.getvalue()

    def delete_user(self, user: str) -> bool:
        base = self.user_dir(user)
        with self._lock:
            ok = os.path.isdir(base)
            shutil.rmtree(base, ignore_errors=True)
            self._users.pop(user, None)
            self._usage.pop(user, None)
            return ok

    # ---------------------------------------------------------------- usage (injections)
    def _usage_of(self, user: str) -> dict:
        u = self._usage.get(user)
        if u is None:
            try:
                with open(os.path.join(self.user_dir(user), "usage.json")) as f:
                    u = json.load(f)
            except (OSError, ValueError):
                u = {}
            self._usage[user] = u
        return u

    def mark_used(self, user: str, ids, what: str = "body") -> None:
        if not valid_user(user) or self.repo or not ids:
            return
        day = time.strftime("%Y-%m-%d", time.gmtime(self.clock()))
        with self._lock:
            u = self._usage_of(user)
            for i in ids:
                d = u.setdefault(i, {})
                d[day] = int(d.get(day, 0)) + 1
            self._usage_dirty.add(user)

    def uses(self, user: str, nid: str, days: int = 7) -> dict:
        cut = time.strftime("%Y-%m-%d", time.gmtime(self.clock() - days * 86400))
        with self._lock:
            d = self._usage_of(user).get(nid) or {}
        return {k: v for k, v in sorted(d.items()) if k >= cut}

    def flush_usage(self) -> None:
        with self._lock:
            dirty, self._usage_dirty = self._usage_dirty, set()
            for user in dirty:
                u = self._usage.get(user) or {}
                cut = time.strftime("%Y-%m-%d", time.gmtime(self.clock() - 90 * 86400))
                u = {i: {k: v for k, v in d.items() if k >= cut} for i, d in u.items()}
                self._usage[user] = u
                try:
                    _write_private(os.path.join(self.user_dir(user), "usage.json"),
                                   json.dumps(u, separators=(",", ":")))
                except OSError:
                    pass

    # ---------------------------------------------------------------- the nightly job
    def nightly(self, now: float | None = None) -> dict:
        """04:00 beside the transcripts' prune: archives older than 30 days deleted, duplicates
        listed (the dashboard shows them; a clerk merges them when one is loaded), stale
        `program_state` counted, INDEX.md regenerated, the cap checked. Counts only."""
        now = self.clock() if now is None else now
        today = time.strftime("%Y-%m-%d", time.gmtime(now))
        out = {"users": 0, "deleted": 0, "stale": 0, "duplicates": 0, "over_cap": 0}
        cut = now - ARCHIVE_DAYS * 86400
        for user in self.users():
            out["users"] += 1
            for a in self.archived(user):
                if _ts(a.archived) < cut:
                    try:
                        os.remove(a.path)
                        shutil.rmtree(os.path.join(self.user_dir(user), "revisions", a.id),
                                      ignore_errors=True)
                        out["deleted"] += 1
                    except OSError:
                        pass
            ns = self.notes(user)
            out["stale"] += sum(1 for n in ns if n.stale(today))
            out["duplicates"] += len(self.duplicates(user))
            out["over_cap"] += len(ns) > CAPS["user_notes"]
            self.write_index(user)
        self.flush_usage()
        if self.journal is not None:
            self.journal.note("memory_notes", "nightly", decision="done", detail=out)
        return out

    def duplicates(self, user: str) -> list[tuple[str, str, float]]:
        """Pairs of live notes in the same scope whose similarity is at or above the threshold."""
        thr = float((self.settings() or {}).get("dedupe_threshold", 0.6))
        ns = self.notes(user)
        out = []
        for i, a in enumerate(ns):
            for b in ns[i + 1:]:
                if a.scope != b.scope or a.project != b.project or a.session != b.session:
                    continue
                s = similarity(a.title or "", a.body or "", b.title or "", b.body or "")
                if s >= thr:
                    out.append((a.id, b.id, round(s, 3)))
        return out

    def info(self) -> dict:
        return {"root": self.dir, "refused": self.repo, "users": self.users(), **self.stats}


def line_of(n: Note) -> str:
    flag = " pinned" if n.pinned else ""
    return f"- [{n.id}] {n.kind} · {n.title} ({n.status}{flag}, {(n.updated or '')[:10]})"


def _ts(iso: str | None) -> float:
    try:
        return float(calendar.timegm(time.strptime(iso or "", "%Y-%m-%dT%H:%M:%SZ")))
    except ValueError:
        return 0.0


def _date(v) -> str | None:
    v = str(v or "").strip()
    return v if re.match(r"^\d{4}-\d{2}-\d{2}$", v) else None


def _tags(v) -> list[str]:
    if isinstance(v, str):
        v = v.split(",")
    out = []
    for t in v or []:
        t = slug(str(t), 32)
        if t and t != "note" and t not in out:
            out.append(t)
    return out[:CAPS["tags"]]
