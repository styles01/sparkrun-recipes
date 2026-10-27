"""What a run changed in the chat's files, and its undo (SRV-357, Chat 5).

Before a run's first write or command, and when the run ends, chatd records the workspace as a
manifest {path: blob hash} with each file's bytes in a content-addressed store
(`<data>/snapshots/<conversation>/objects/<hash>`; an unchanged file is stored once, whatever the
number of runs). The diff pane is `diff(run)`: per file the status (added, modified, deleted), the
line counts and a unified patch. `undo(run)` puts every file the run touched back as it was
before, and is itself recorded, so an undo can be undone.

The plan put these in git refs inside the workspace. They live beside it instead: running git on the
host inside a tree the agent controls would run its `.git/config` (core.fsmonitor, hooks) as
chatd's user, and running it in the container would start a sandbox for every write. Pure Python
on the host reads the files as data only, through the workspace's symlink-safe fds
(chatd/workspace.py), and writes them back the same way. Skipped: .git, node_modules, .tandem
and files over 5 MB (listed with their size, not restorable).
"""

from __future__ import annotations

import difflib
import json
import os

from chatd.workspace import PathError, Workspace, etag

MAX_FILE = 5 * 1024 * 1024
MAX_FILES = 20_000


class Snapshots:
    def __init__(self, root: str):
        self.root = root

    def _dir(self, cid: str) -> str:
        return os.path.join(self.root, cid)

    def _obj(self, cid: str, h: str) -> str:
        return os.path.join(self._dir(cid), "objects", h[:2], h)

    def _manifest(self, cid: str, run: str, which: str) -> str:
        return os.path.join(self._dir(cid), "runs", run, which + ".json")

    def has(self, cid: str, run: str, which: str) -> bool:
        return os.path.exists(self._manifest(cid, run, which))

    def take(self, cid: str, ws: Workspace, run: str, which: str) -> dict:
        """Record the tree; returns the manifest {path: hash or "big:<size>"}."""
        man = {}
        for n, full in enumerate(ws.walk()):
            if n >= MAX_FILES:
                break
            rel = os.path.relpath(full, ws.root)
            # opened through the workspace's fds (O_NOFOLLOW, a regular file by its fstat): a
            # file the sandbox swaps for a symlink after the walk is skipped, not followed
            try:
                fh, st = ws.dir.open(rel)
                with fh:
                    if st.st_size > MAX_FILE:
                        man[rel] = f"big:{st.st_size}"
                        continue
                    data = fh.read(MAX_FILE + 1)
            except (PathError, OSError):
                continue
            if len(data) > MAX_FILE:
                man[rel] = f"big:{len(data)}"
                continue
            h = etag(data)
            p = self._obj(cid, h)
            if not os.path.exists(p):
                os.makedirs(os.path.dirname(p), exist_ok=True)
                with open(p + ".tmp", "wb") as fh:
                    fh.write(data)
                os.replace(p + ".tmp", p)
            man[rel] = h
        path = self._manifest(cid, run, which)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(man, fh)
        return man

    def load(self, cid: str, run: str, which: str) -> dict | None:
        try:
            with open(self._manifest(cid, run, which)) as fh:
                return json.load(fh)
        except (OSError, ValueError):
            return None

    def _text(self, cid: str, h: str | None) -> list[str] | None:
        if not h:
            return []
        if h.startswith("big:"):
            return None
        try:
            with open(self._obj(cid, h), "rb") as fh:
                data = fh.read()
        except OSError:
            return None
        if b"\0" in data[:4096]:
            return None
        return data.decode("utf-8", "replace").splitlines(keepends=True)

    def _size(self, cid: str, h: str | None) -> int:
        """A recorded file's size: 0 for none (deleted), else the object's or a big one's."""
        if not h:
            return 0
        if h.startswith("big:"):
            return int(h[4:])
        try:
            return os.path.getsize(self._obj(cid, h))
        except OSError:
            return 0

    def diff(self, cid: str, run: str) -> dict:
        """{files: [{path, status, additions, deletions, patch}]} between the run's before and
        after (after = the tree now while the run has not ended); a binary or big file's entry
        has its size after (`bytes`)."""
        before = self.load(cid, run, "before")
        after = self.load(cid, run, "after")
        if before is None:
            return {"files": []}
        if after is None:
            return {"files": [], "pending": True}
        out = []
        for path in sorted(set(before) | set(after)):
            a, b = before.get(path), after.get(path)
            if a == b:
                continue
            status = "added" if a is None else "deleted" if b is None else "modified"
            la, lb = self._text(cid, a), self._text(cid, b)
            if la is None or lb is None:
                out.append({"path": "/workspace/" + path, "status": status, "additions": 0,
                            "deletions": 0, "patch": "Binary or large file changed\n",
                            "bytes": self._size(cid, b)})
                continue
            patch = "".join(difflib.unified_diff(la, lb, f"a/{path}", f"b/{path}", n=3))
            adds = sum(1 for ln in patch.splitlines() if ln.startswith("+") and
                       not ln.startswith("+++"))
            dels = sum(1 for ln in patch.splitlines() if ln.startswith("-") and
                       not ln.startswith("---"))
            out.append({"path": "/workspace/" + path, "status": status, "additions": adds,
                        "deletions": dels, "patch": patch})
        return {"files": out}

    def patch_text(self, cid: str, run: str) -> str:
        return "".join(f["patch"] if f["patch"].startswith("---") else
                       f"--- a/{f['path'][11:]}\n+++ b/{f['path'][11:]}\n{f['patch']}"
                       for f in self.diff(cid, run)["files"])

    def restore(self, cid: str, ws: Workspace, run: str, undo_id: str) -> int:
        """The workspace as it was before `run` (rewind): files made since go, changed ones come
        back; the state before the rewind is recorded as run `undo_id`. Returns how many files
        changed. A file too big to record is left as it is."""
        target = self.load(cid, run, "before")
        if target is None:
            raise ValueError("this run has no recorded start")
        now = self.take(cid, ws, undo_id, "before")
        n = 0
        for path in set(now) | set(target):
            a, b = target.get(path), now.get(path)
            if a == b or (a or "").startswith("big:") or (b or "").startswith("big:"):
                continue
            try:
                if a is None:
                    ws.delete(path)
                else:
                    with open(self._obj(cid, a), "rb") as fh:
                        ws.write(path, fh.read())
                n += 1
            except (OSError, ValueError):
                pass
        self.take(cid, ws, undo_id, "after")
        return n

    def undo(self, cid: str, ws: Workspace, run: str, undo_id: str) -> int:
        """Put the files the run changed back as they were before it; returns how many. The
        state before the undo is recorded as run `undo_id` (before), and after it (after)."""
        before = self.load(cid, run, "before")
        after = self.load(cid, run, "after")
        if before is None or after is None:
            raise ValueError("this run has no recorded change")
        self.take(cid, ws, undo_id, "before")
        n = 0
        for path in set(before) | set(after):
            a, b = before.get(path), after.get(path)
            if a == b:
                continue
            if a is None:
                try:
                    ws.delete(path)
                    n += 1
                except (OSError, ValueError):
                    pass
            elif not a.startswith("big:"):
                with open(self._obj(cid, a), "rb") as fh:
                    ws.write(path, fh.read())
                n += 1
        self.take(cid, ws, undo_id, "after")
        return n
