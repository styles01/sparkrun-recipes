"""A deleted chat's sandbox (live check of 2026-10-13: a deleted chat's container kept running and its
egress, homes, snapshots and workspaces folders stayed): delete stops the container, restore brings
the chat back and its sandbox starts again on the next call, the purge (delete forever, or the trash
after 7 days) removes the container and the folders, and chatd's start removes the containers of
chats that are gone or in the trash. CPU only: the sandbox is test_chatd_agent's fake, the container
engine a recorder.

Run: python tests/test_chatd_sandbox_lifecycle.py
"""

from __future__ import annotations

import os

from chatd_util import run_all
from test_chatd_agent import calls, chat_with_sandbox, parts_of

from chatd.service import TRASH_KEEP_MS


class _Engine:
    """The container engine's API, recorded: (method, path)."""

    def __init__(self):
        self.calls = []

    def call(self, method, path, body=None, timeout=60.0):
        self.calls.append((method, path))
        return 204, None


def _chat():
    c = chat_with_sandbox()
    c.svc.sandboxes.engine = _Engine()
    return c


def _folders(c, cid):
    return [os.path.join(c.data, d, cid) for d in ("workspaces", "homes", "egress", "snapshots")]


def _use_the_sandbox(c, cid):
    """A bash call (the sandbox starts) and a write (a snapshot is taken): every folder exists."""
    c.send(cid, calls(("write_file", {"path": "a.txt", "content": "x"}), ("bash", {"command": "ls"})))
    s = c.wait_idle(cid, 20)
    assert [p["status"] for p in parts_of(s) if p["type"] == "tool_call"] == ["done", "done"], s
    assert c.svc.sandboxes.box(cid).state == "running"
    for d in _folders(c, cid):
        assert os.path.isdir(d), d


def test_delete_stops_the_sandbox_and_restore_starts_it_again():
    c = _chat()
    try:
        cid = c.new()
        _use_the_sandbox(c, cid)
        c.ok("DELETE", f"/conversations/{cid}")
        b = c.svc.sandboxes.box(cid)
        assert b.state == "stopped" and b.stopped_reason == "deleted"
        assert all(os.path.isdir(d) for d in _folders(c, cid))          # the files wait
        assert c.call("GET", f"/conversations/{cid}")[0] == 404
        assert not any(cid in str(x) for x in c.ok("GET", "/conversations"))
        # restore: the chat is back with its files, and its next call starts the sandbox again
        snap = c.ok("POST", f"/conversations/{cid}/restore")
        assert snap["id"] == cid and open(os.path.join(c.data, "workspaces", cid, "a.txt")).read() == "x"
        c.send(cid, calls(("bash", {"command": "ls"})))
        c.wait_idle(cid, 20)
        assert c.svc.sandboxes.box(cid).state == "running"
        assert c.call("POST", f"/conversations/{cid}/restore")[0] == 404    # not in the trash
    finally:
        c.close()


def test_delete_forever_removes_the_container_and_the_folders():
    c = _chat()
    try:
        cid, other = c.new(), c.new()
        _use_the_sandbox(c, cid)
        _use_the_sandbox(c, other)
        c.ok("DELETE", f"/conversations/{cid}?forever=1")
        assert ("DELETE", f"/containers/fake-{cid}?force=1") in c.svc.sandboxes.engine.calls
        assert not any(os.path.exists(d) for d in _folders(c, cid)), _folders(c, cid)
        assert c.svc.store.one("SELECT 1 FROM conversations WHERE id = ?", (cid,)) is None
        assert cid not in c.svc.sandboxes.boxes
        assert c.call("POST", f"/conversations/{cid}/restore")[0] == 404
        # the other chat keeps everything
        assert all(os.path.isdir(d) for d in _folders(c, other))
        assert c.svc.sandboxes.box(other).state == "running"
    finally:
        c.close()


def test_the_trash_after_seven_days_is_the_same_purge():
    c = _chat()
    try:
        cid = c.new()
        _use_the_sandbox(c, cid)
        c.ok("DELETE", f"/conversations/{cid}")
        deleted = c.svc.store.one("SELECT deleted FROM conversations WHERE id = ?", (cid,))["deleted"]
        c.svc.sweep_once(now=deleted + TRASH_KEEP_MS - 1000)            # not yet
        assert all(os.path.isdir(d) for d in _folders(c, cid))
        c.svc.sweep_once(now=deleted + TRASH_KEEP_MS + 1000)
        assert not any(os.path.exists(d) for d in _folders(c, cid))
        assert ("DELETE", f"/containers/fake-{cid}?force=1") in c.svc.sandboxes.engine.calls
        assert c.svc.store.one("SELECT 1 FROM conversations WHERE id = ?", (cid,)) is None
    finally:
        c.close()


def test_the_start_removes_the_containers_of_gone_and_trashed_chats():
    c = _chat()
    try:
        live, trashed = c.new(), c.new()
        for cid in (live, trashed):
            _use_the_sandbox(c, cid)
        c.ok("DELETE", f"/conversations/{trashed}")
        gone = "01GONEGONEGONEGONEGONEGONE"                   # a container left by a chat that is gone
        for d in _folders(c, gone):
            os.makedirs(d)
        b = c.svc.sandboxes.box(gone)
        b.container, b.state = "fake-gone", "running"
        c.svc.sandboxes.box(trashed).container = "fake-trashed"
        c.lines.clear()
        removed = c.svc.reap_sandboxes()
        assert sorted(removed) == sorted([gone, trashed]), removed
        names = [p for m, p in c.svc.sandboxes.engine.calls if m == "DELETE"]
        assert "/containers/fake-gone?force=1" in names            # by its id: its own only
        assert "/containers/fake-trashed?force=1" in names
        assert not any(os.path.exists(d) for d in _folders(c, gone))          # gone: the folders go
        assert all(os.path.isdir(d) for d in _folders(c, trashed))            # trashed: they wait
        assert c.svc.sandboxes.box(live).state == "running" and live not in removed
        logged = [x for x in c.lines if x.startswith("[sandbox]") and "removed the container" in x]
        assert len(logged) == 2 and any("gone (and its folders)" in x for x in logged), c.lines
        assert c.svc.reap_sandboxes() == []
    finally:
        c.close()


if __name__ == "__main__":
    run_all(globals())
