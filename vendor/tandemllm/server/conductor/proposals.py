"""Packages the model maintains: its proposals, waiting for a click (2026-10-05, P-M3; the design:
Memo "Memory for TandemLLM: design advice", section 6).

The same block as a memory note, another policy. The model writes

    <tandem_memory op="propose" kind="skill" name="style-watercolor" mode="append">
    title: a Sources section and what to do without search
    reason: the last two watercolor sessions searched for references the skill does not name

    ## Sources
    ...
    </tandem_memory>

and the engine keeps it as a proposal: never applied by the model. A memory is one user's,
provisional and read back at once, because a wrong preference costs one turn; a package is every
session's, and a wrong edit makes the model worse for all of them. So a proposal waits in
`<memory root>/proposals/<name>/<n>.md` with its evidence (the session, the request, the J1
outcomes of that session that showed a gap; "no evidence" when the session has none), and the
Packages screen's Proposals tab shows the diff against the live text. Apply is the packages
document's versioned PUT (the package becomes inline with the new text; its old version stays, and
the Packages screen's restore brings it back); Reject keeps the file with its reason.

`mode`: `replace` (the body is the package's whole new text) or `append` (the body is a section
added after the live text). A proposal remembers the live text's hash; at Apply a replacement of a
text that changed since is refused (409), an append is applied to the text as it is then.

Caps: a body of at most 2,000 characters (the block's own limit), 20 open proposals a package,
the secret refusal on the body. Journaled as `memory_notes` task `propose`, `apply`, `reject`
with names, numbers and counts only.
"""

from __future__ import annotations

import hashlib
import os
import re
import threading
import time

from server.conductor import redact as redact_mod
from server.conductor.notes import Refused, _write_private, _now_iso

MODES = ("replace", "append")
STATUSES = ("open", "applied", "rejected")
OPEN_MAX = 20
_NAME = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,79}$")
_FRONT = re.compile(r"\A---\n(.*?)\n---\n", re.S)
TEXT_HEAD = "## Proposed text\n"
BODY_HEAD = "## As written\n"


def sha(text: str) -> str:
    return hashlib.sha256((text or "").encode()).hexdigest()[:16]


class Proposals:
    """`root`: the memory root (the notes' sibling directory `proposals/`); `store`: the
    conductor's document store (the packages document); `library`: its text resolver."""

    def __init__(self, root: str, store, library, journal=None, clock=time.time):
        self.dir = os.path.join(os.path.realpath(os.path.expanduser(root)), "proposals")
        self.store, self.library, self.journal, self.clock = store, library, journal, clock
        self._lock = threading.Lock()

    # ---------------------------------------------------------------- files
    def _path(self, name: str, n: int) -> str:
        return os.path.join(self.dir, name, f"{int(n)}.md")

    def _read(self, path: str) -> dict | None:
        try:
            with open(path) as f:
                text = f.read()
        except OSError:
            return None
        m = _FRONT.match(text)
        if not m:
            return None
        meta: dict = {}
        for line in m.group(1).splitlines():
            k, sep, v = line.partition(":")
            if sep:
                meta[k.strip()] = v.strip()
        rest = text[m.end():]
        i, j = rest.find(TEXT_HEAD), rest.find(BODY_HEAD)
        proposed = rest[i + len(TEXT_HEAD):j].strip("\n") if i >= 0 and j > i else ""
        body = rest[j + len(BODY_HEAD):].strip("\n") if j >= 0 else ""
        meta["n"] = int(meta.get("n") or 0)
        for k in ("base_version", "applied_version"):
            if meta.get(k):
                meta[k] = int(meta[k])
        meta["evidence"] = [e.strip() for e in (meta.get("evidence") or "").split(",")
                            if e.strip()]
        return dict(meta, proposed=proposed, body=body)

    def _write(self, p: dict) -> None:
        keys = ("n", "name", "kind", "mode", "title", "reason", "status", "writer", "created",
                "updated", "session", "request_id", "base_version", "base_sha", "evidence",
                "applied_version", "rejected_reason", "user")
        lines = ["---"]
        for k in keys:
            v = p.get(k)
            if v is None or v == "" or v == []:
                continue
            if isinstance(v, list):
                v = ", ".join(str(x) for x in v)
            lines.append(f"{k}: {' '.join(str(v).split())}")
        lines.append("---")
        text = "\n".join(lines) + "\n" + TEXT_HEAD + (p.get("proposed") or "") + "\n\n" + \
            BODY_HEAD + (p.get("body") or "") + "\n"
        _write_private(self._path(p["name"], p["n"]), text)

    def all(self) -> list[dict]:
        out = []
        try:
            names = sorted(os.listdir(self.dir))
        except OSError:
            return []
        for name in names:
            d = os.path.join(self.dir, name)
            if not os.path.isdir(d):
                continue
            for f in os.listdir(d):
                if f.endswith(".md") and f[:-3].isdigit():
                    p = self._read(os.path.join(d, f))
                    if p is not None:
                        out.append(p)
        out.sort(key=lambda p: (p.get("created") or ""), reverse=True)
        return out

    def get(self, name: str, n: int) -> dict | None:
        if not _NAME.match(name or ""):
            return None
        return self._read(self._path(name, n))

    # ---------------------------------------------------------------- the live text
    def live(self, name: str) -> tuple[int, dict, str | None]:
        """(packages version, document, the package's text now; None when it has none)."""
        v, doc = self.store.get("packages")
        if name not in (doc.get("packages") or {}):
            return v, doc, None
        text, _ = self.library.text(doc, name)
        return v, doc, text

    # ---------------------------------------------------------------- the model's op
    def propose(self, op: dict, *, user: str | None, session: str | None,
                request_id: str | None, writer: str = "qwen-inline", evidence=None) -> dict:
        name = (op.get("name") or "").strip()
        if not _NAME.match(name):
            raise Refused("schema", "name: the package's name")
        mode = (op.get("mode") or "replace").strip()
        if mode not in MODES:
            raise Refused("schema", f"mode: one of {', '.join(MODES)}")
        body = (op.get("body") or "").strip()
        if not body:
            raise Refused("schema", "body: the new text or the section to add")
        body, counts = redact_mod.redact(body)
        if counts:
            raise Refused("secret", "a proposal never carries a secret")
        title = " ".join(str(op.get("title") or "").split())[:120] or f"a change to {name}"
        v, _, text = self.live(name)
        if text is None:
            raise Refused("unknown_package", f"no package {name!r} with a text to change")
        with self._lock:
            mine = [p for p in self.all() if p["name"] == name]
            if sum(1 for p in mine if p.get("status") == "open") >= OPEN_MAX:
                raise Refused("cap", f"{OPEN_MAX} open proposals for {name} already")
            n = max([p["n"] for p in mine] + [0]) + 1
            now = _now_iso(self.clock())
            proposed = body if mode == "replace" else text.rstrip() + "\n\n" + body
            p = {"n": n, "name": name, "kind": (op.get("kind") or "skill").strip(), "mode": mode,
                 "title": title, "reason": " ".join(str(op.get("reason") or "").split())[:400],
                 "status": "open", "writer": writer, "created": now, "updated": now,
                 "session": session, "request_id": request_id, "base_version": v,
                 "base_sha": sha(text), "evidence": list(evidence or []), "user": user,
                 "proposed": proposed, "body": body}
            self._write(p)
        self._note("propose", name, n, {"mode": mode, "evidence": len(p["evidence"])})
        return {"op": "propose", "name": name, "n": n, "title": title}

    # ---------------------------------------------------------------- the dashboard's clicks
    def apply(self, name: str, n: int, author: str) -> dict:
        """The packages document's versioned PUT with the proposed text (the package inline);
        409 (`Refused("stale")`) when a replacement's base text changed since."""
        with self._lock:
            p = self.get(name, n)
            if p is None:
                raise Refused("unknown_id", f"no proposal {name}#{n}")
            if p.get("status") != "open":
                raise Refused("closed", f"proposal {name}#{n} is {p.get('status')}")
            v, doc, text = self.live(name)
            if text is None:
                raise Refused("unknown_package", f"no package {name!r}")
            new = p["proposed"]
            if sha(text) != p.get("base_sha"):
                if p.get("mode") != "append":
                    raise Refused("stale", f"{name} changed since the proposal; reject it or "
                                           "let the model propose again")
                new = text.rstrip() + "\n\n" + p["body"]
            pk = dict(doc["packages"][name])
            pk.update(source="inline", body=new)
            pk.pop("body_from", None)
            est = int(len(new) / 3.5) + 16
            if est > int(pk.get("max_tokens") or 0):
                pk["max_tokens"] = min(32768, est)
            doc = dict(doc, packages=dict(doc["packages"], **{name: pk}))
            ver = self.store.put("packages", None, doc, author=author, base=v,
                                 note=f"proposal {name}#{n}: {p['title']}"[:200])
            p.update(status="applied", applied_version=ver, updated=_now_iso(self.clock()))
            self._write(p)
        self._note("apply", name, n, {"version": ver})
        return {"name": name, "n": n, "version": ver, "previous": v}

    def reject(self, name: str, n: int, reason: str) -> dict:
        with self._lock:
            p = self.get(name, n)
            if p is None:
                raise Refused("unknown_id", f"no proposal {name}#{n}")
            if p.get("status") != "open":
                raise Refused("closed", f"proposal {name}#{n} is {p.get('status')}")
            p.update(status="rejected", rejected_reason=" ".join(str(reason).split())[:200],
                     updated=_now_iso(self.clock()))
            self._write(p)
        self._note("reject", name, n, {})
        return {"name": name, "n": n, "status": "rejected"}

    def view(self, p: dict) -> dict:
        """A proposal with the live text it would replace and whether it still applies."""
        _, _, text = self.live(p["name"])
        stale = text is not None and sha(text) != p.get("base_sha")
        return dict(p, live=text, stale=stale,
                    applies=p.get("status") == "open" and text is not None and
                    (not stale or p.get("mode") == "append"),
                    evidence_word=("no evidence" if not p.get("evidence") else
                                   f"{len(p['evidence'])} outcome row"
                                   f"{'' if len(p['evidence']) == 1 else 's'}"))

    def _note(self, task: str, name: str, n: int, detail: dict) -> None:
        if self.journal is not None:
            self.journal.note("memory_notes", task, decision=task,
                              detail=dict(detail, name=name, n=n))
