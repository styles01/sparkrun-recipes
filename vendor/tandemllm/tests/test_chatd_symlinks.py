"""chatd's host-side reads and writes in the folders the sandbox writes into (SEC-1): the chat's
workspace (/workspace) and its home (/home/agent) are bind-mounted read-write, so the sandbox can
put a symlink anywhere in them, before chatd looks or between its check and its open. No access
may follow one. CPU only; the real container case is tests/test_sandbox_box.py.

    1  save_upload with uploads/ a symlink out of the workspace: nothing lands outside
    2  save_upload with a dangling symlink at the file name: the next free name, nothing outside
    3  save_upload of a name of 200 non-ASCII characters: cut to 255 bytes, the extension kept
    4  a message whose attachment cannot be placed is sent all the same; the prompt says why
    5  a name too long for the file system no longer fails the send
    6  Open WebUI: .tandem/owui a symlink out of the home: refused, nothing written outside
    7  Open WebUI: symlinks planted at run.py, <toolkit>.py and req-<call id>.json: replaced,
       their targets untouched, the call answers
    8  a call id from the backend that is a path (`../..`) is replaced before it names a file
    9  snapshot take: a symlinked file is not recorded; a file swapped for a symlink between the
       walk and the open is not read
   10  Workspace write, read, delete, list: a folder swapped for a symlink right after the check
       (the first file access) does not redirect the access
   11  a FIFO in the workspace does not block take, grep or read
   12  a text file over the budget that cannot be placed is one line; the send does not fail

Every case fails on chatd before the fix and passes after it.

Run: python tests/test_chatd_symlinks.py
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading

from chatd_util import Chat, run_all
from test_chatd_attach import upload
from test_chatd_owui import fixture, owui_chat

from chatd import attach
from chatd import workspace as workspace_mod
from chatd.snapshots import Snapshots
from chatd.workspace import Workspace


def dirs():
    base = tempfile.mkdtemp(prefix="chatd-links-")
    ws, outside = os.path.join(base, "ws"), os.path.join(base, "outside")
    os.makedirs(ws)
    os.makedirs(outside)
    return base, ws, outside


def tree(d: str) -> list[str]:
    return sorted(os.path.relpath(os.path.join(p, f), d) for p, _, fs in os.walk(d) for f in fs)


# ---------------------------------------------------------------------- uploads
def test_an_upload_into_a_symlinked_uploads_folder_stays_in_the_workspace():
    base, ws, outside = dirs()
    try:
        os.symlink(outside, os.path.join(ws, "uploads"))
        try:
            got = attach.save_upload(ws, "notes.txt", b"the user's file")
        except (ValueError, OSError) as exc:
            got = exc
        assert tree(outside) == [], (tree(outside), got)
        assert isinstance(got, workspace_mod.PathError) and "symlink" in str(got), got
    finally:
        shutil.rmtree(base)


def test_a_dangling_symlink_at_the_name_is_not_followed():
    base, ws, outside = dirs()
    try:
        os.makedirs(os.path.join(ws, "uploads"))
        os.symlink(os.path.join(outside, "planted.txt"), os.path.join(ws, "uploads", "notes.txt"))
        got = attach.save_upload(ws, "notes.txt", b"the user's file")
        assert tree(outside) == [], tree(outside)
        assert got == "/workspace/uploads/notes-2.txt", got
        with open(os.path.join(ws, "uploads", "notes-2.txt"), "rb") as fh:
            assert fh.read() == b"the user's file"
        assert os.path.islink(os.path.join(ws, "uploads", "notes.txt"))
    finally:
        shutil.rmtree(base)


def test_a_long_non_ascii_name_is_cut_by_bytes_and_keeps_its_extension():
    base, ws, _ = dirs()
    try:
        name = "ä" * 200 + ".txt"                                  # 404 bytes of UTF-8
        got = attach.save_upload(ws, name, b"x")
        pick = os.path.basename(got)
        assert len(pick.encode()) <= 255 and pick.endswith(".txt"), (len(pick.encode()), pick)
        assert pick == "ä" * 125 + ".txt", pick                     # 250 + 4 bytes
        assert os.path.isfile(os.path.join(ws, "uploads", pick))
        again = os.path.basename(attach.save_upload(ws, name, b"y"))
        assert again == "ä" * 124 + "-2.txt" and len(again.encode()) <= 255, again
        assert attach.fit_name("ab€", "") == "ab€"
        assert attach.fit_name("€" * 100, ".md") == "€" * 84 + ".md"   # never half a character
    finally:
        shutil.rmtree(base)


def test_a_message_whose_attachment_cannot_be_placed_is_sent_and_says_why():
    c = Chat()
    outside = tempfile.mkdtemp(prefix="chatd-outside-")
    try:
        cid = c.new()
        ws = os.path.join(c.data, "workspaces", cid)
        os.makedirs(ws, exist_ok=True)
        os.symlink(outside, os.path.join(ws, "uploads"))
        att = upload(c, "notes.txt", "text/plain", b"alpha beta\n")
        st, out, _ = c.call("POST", f"/conversations/{cid}/messages",
                            {"client_msg_id": "u1", "text": "read this",
                             "attachments": [att["id"]]})
        assert st in (200, 202), (st, out)
        s = c.wait_idle(cid)
        assert s["messages"][-1]["status"] == "done", s["messages"][-1]
        assert tree(outside) == [], tree(outside)
        a = s["messages"][0]["attachments"][0]
        assert "path" not in a and "symlink" in a["unplaced"], a
        sent = c.last_body()["messages"][-1]["content"]
        assert sent == ("read this\n\n```notes.txt\nalpha beta\n\n```\n\n[Attached file notes.txt: it "
                        "could not be saved to /workspace/uploads/ (/workspace/uploads/notes.txt: a "
                        "symlink on the way is not followed), so the sandbox's tools do not have "
                        "it.]"), sent
        assert c.svc.attachment(att["id"])[1] == b"alpha beta\n"            # the blob stays
    finally:
        c.close()
        shutil.rmtree(outside)


def test_a_text_file_over_the_budget_that_cannot_be_placed_is_sent_as_one_line():
    c = Chat()
    try:
        cid = c.new()
        ws = os.path.join(c.data, "workspaces", cid)
        os.makedirs(ws, exist_ok=True)
        open(os.path.join(ws, "uploads"), "w").close()                  # a file, not a folder
        att = upload(c, "big.txt", "text/plain", ("x" * 99 + "\n").encode() * 2_000)
        assert att["tokens"] > 32_000, att
        st, out, _ = c.call("POST", f"/conversations/{cid}/messages",
                            {"client_msg_id": "u1", "text": "read this",
                             "attachments": [att["id"]]})
        assert st in (200, 202), (st, out)
        s = c.wait_idle(cid)
        a = s["messages"][0]["attachments"][0]
        assert "path" not in a and a["unplaced"] and a["inline"] is False, a
        sent = c.last_body()["messages"][-1]["content"]
        assert "```" not in sent and "It could not be saved to /workspace/uploads/" in sent, sent
    finally:
        c.close()


def test_a_name_too_long_for_the_file_system_does_not_fail_the_send():
    c = Chat()
    try:
        cid = c.new()
        att = upload(c, "ö" * 190 + ".md", "text/markdown", b"# notes\n")     # 383 bytes
        st, out, _ = c.call("POST", f"/conversations/{cid}/messages",
                            {"client_msg_id": "u1", "text": "here", "attachments": [att["id"]]})
        assert st in (200, 202), (st, out)
        s = c.wait_idle(cid)
        a = s["messages"][0]["attachments"][0]
        pick = a["path"].rsplit("/", 1)[1]
        assert a["path"].startswith("/workspace/uploads/") and pick.endswith(".md"), a
        assert len(pick.encode()) <= 255, len(pick.encode())
        assert os.path.isfile(os.path.join(c.data, "workspaces", cid, "uploads", pick))
    finally:
        c.close()


# ---------------------------------------------------------------------- Open WebUI's runner
def _owui_call(c, cid, call_id=None):
    call = {"name": "owui__word_tools__word_count", "arguments": {"text": "a b a", "unique": True}}
    if call_id:
        call["id"] = call_id
    c.send(cid, "FAKE_CALLS " + json.dumps([call]))
    s = c.wait_idle(cid, 60)
    return next(p for p in s["messages"][-1]["parts"] if p["type"] == "tool_call")


def _owui_setup():
    c = owui_chat()
    c.svc.owui.add("notes.py", fixture("notes.py"))
    c.svc.add_rule("owui__word_tools__*", None, "allow", None)
    return c


def test_owui_refuses_a_symlinked_runner_folder():
    c = _owui_setup()
    outside = tempfile.mkdtemp(prefix="chatd-outside-")
    try:
        cid = c.new()
        _, home, _ = c.svc.sandboxes.paths(cid)
        os.makedirs(os.path.join(home, ".tandem"))
        os.symlink(outside, os.path.join(home, ".tandem", "owui"))
        p = _owui_call(c, cid)
        assert tree(outside) == [], tree(outside)
        assert p["status"] == "failed" and "symlink" in p["output"]["content"], p
    finally:
        c.close()
        shutil.rmtree(outside)


def test_owui_replaces_planted_symlinks_and_leaves_their_targets_alone():
    c = _owui_setup()
    outside = tempfile.mkdtemp(prefix="chatd-outside-")
    try:
        cid = c.new()
        _, home, _ = c.svc.sandboxes.paths(cid)
        d = os.path.join(home, ".tandem", "owui")
        os.makedirs(d)
        for name in ("run.py", "word_tools.py", "req-call_planted.json"):
            target = os.path.join(outside, name + ".victim")
            with open(target, "w") as fh:
                fh.write("the host's own file\n")
            os.symlink(target, os.path.join(d, name))
        p = _owui_call(c, cid, "call_planted")
        for name in sorted(os.listdir(outside)):
            with open(os.path.join(outside, name)) as fh:
                assert fh.read() == "the host's own file\n", name
        assert sorted(os.listdir(outside)) == ["req-call_planted.json.victim", "run.py.victim",
                                               "word_tools.py.victim"]
        assert p["status"] == "done" and p["output"]["content"] == '{"words": 2}', p
        assert not os.path.islink(os.path.join(d, "run.py"))
        assert not os.path.islink(os.path.join(d, "word_tools.py"))
    finally:
        c.close()
        shutil.rmtree(outside)


def test_a_call_id_that_is_a_path_is_replaced():
    c = _owui_setup()
    try:
        cid = c.new()
        _, home, _ = c.svc.sandboxes.paths(cid)
        d = os.path.join(home, ".tandem", "owui")
        os.makedirs(os.path.join(d, "req-.."))                      # the sandbox's part
        p = _owui_call(c, cid, "../../../../../escaped")
        # req-../../../../../escaped.json is <data>/homes/escaped.json, outside the home
        assert not os.path.exists(os.path.join(os.path.dirname(home), "escaped.json"))
        assert p["id"].startswith("call_") and "/" not in p["id"], p["id"]
        assert p["status"] == "done", p
    finally:
        c.close()


# ---------------------------------------------------------------------- snapshots
def test_a_snapshot_never_reads_through_a_symlink():
    base, ws_dir, outside = dirs()
    try:
        secret = os.path.join(outside, "secret.txt")
        with open(secret, "w") as fh:
            fh.write("SECRET of the host\n")
        ws = Workspace(ws_dir)
        with open(os.path.join(ws_dir, "a.txt"), "w") as fh:
            fh.write("inside\n")
        os.symlink(secret, os.path.join(ws_dir, "linked.txt"))       # a symlink before take
        with open(os.path.join(ws_dir, "b.txt"), "w") as fh:
            fh.write("swapped later\n")
        snaps = Snapshots(os.path.join(base, "snapshots"))
        real_walk = ws.walk

        def walk_then_swap(start=workspace_mod.MOUNT):
            for p in real_walk(start):
                if p.endswith("b.txt"):                              # the race: after the walk
                    os.unlink(p)
                    os.symlink(secret, p)
                yield p
        ws.walk = walk_then_swap
        man = snaps.take("c1", ws, "r1", "before")
        assert "linked.txt" not in man and "b.txt" not in man, man
        assert "a.txt" in man, man
        stored = b"".join(open(os.path.join(dp, f), "rb").read()
                          for dp, _, fs in os.walk(os.path.join(base, "snapshots")) for f in fs)
        assert b"SECRET" not in stored
    finally:
        shutil.rmtree(base)


# ---------------------------------------------------------------------- the workspace's own race
class SwapOnAccess:
    """chatd.workspace's `os` with one change: the first call of a name in `on` swaps the folder
    `victim` for a symlink to `target` before it runs, as the sandbox can between a check and the
    access it guards."""

    def __init__(self, victim: str, target: str, on: set):
        self.victim, self.target, self.on, self.done = victim, target, on, False

    def __getattr__(self, name):
        attr = getattr(os, name)
        if name not in self.on:
            return attr

        def hooked(*a, **k):
            if not self.done:
                self.done = True
                os.rename(self.victim, self.victim + ".real")
                os.symlink(self.target, self.victim)
            return attr(*a, **k)
        return hooked


def _race(on: set, act):
    base, ws_dir, outside = dirs()
    os.makedirs(os.path.join(ws_dir, "sub"))
    for d, text in ((ws_dir + "/sub", "inside\n"), (outside, "SECRET of the host\n")):
        with open(os.path.join(d, "secret.txt"), "w") as fh:
            fh.write(text)
    ws = Workspace(ws_dir)
    workspace_mod.os = SwapOnAccess(os.path.join(ws_dir, "sub"), outside, on)
    try:
        try:
            got = act(ws)
        except (ValueError, OSError) as exc:
            got = exc
    finally:
        workspace_mod.os = os
    return base, outside, got


def test_a_write_does_not_follow_a_folder_swapped_after_the_check():
    base, outside, got = _race({"open"}, lambda ws: ws.write("sub/evil.txt", b"pwned"))
    try:
        assert tree(outside) == ["secret.txt"], (tree(outside), got)
        assert isinstance(got, workspace_mod.PathError), got
    finally:
        shutil.rmtree(base)


def test_a_read_does_not_follow_a_folder_swapped_after_the_check():
    base, _, got = _race({"open"}, lambda ws: ws.read("sub/secret.txt"))
    try:
        assert got != b"SECRET of the host\n", got
        assert isinstance(got, workspace_mod.PathError), got
    finally:
        shutil.rmtree(base)


def test_a_delete_does_not_follow_a_folder_swapped_after_the_check():
    base, outside, got = _race({"open", "unlink"}, lambda ws: ws.delete("sub/secret.txt"))
    try:
        assert tree(outside) == ["secret.txt"], (tree(outside), got)
    finally:
        shutil.rmtree(base)


def test_a_listing_does_not_follow_a_folder_swapped_after_the_check():
    base, _, got = _race({"open", "scandir"}, lambda ws: ws.list("sub"))
    try:
        assert isinstance(got, workspace_mod.PathError), got
    finally:
        shutil.rmtree(base)


def test_a_fifo_blocks_neither_a_snapshot_nor_grep_nor_a_read():
    base, ws_dir, _ = dirs()
    fifo = os.path.join(ws_dir, "pipe")
    os.mkfifo(fifo)
    with open(os.path.join(ws_dir, "a.txt"), "w") as fh:
        fh.write("alpha\n")
    ws = Workspace(ws_dir)
    out = {}

    def work():
        out["take"] = Snapshots(os.path.join(base, "snapshots")).take("c1", ws, "r1", "before")
        out["grep"] = ws.grep("alpha")
        try:
            ws.read("pipe")
        except (ValueError, OSError) as exc:
            out["read"] = exc
    t = threading.Thread(target=work, daemon=True)
    t.start()
    t.join(10)
    stuck = t.is_alive()
    while t.is_alive():                                  # let a blocked open see a writer and end
        try:
            os.close(os.open(fifo, os.O_WRONLY | os.O_NONBLOCK))
        except OSError:
            pass
        t.join(0.2)
    try:
        assert not stuck, "an open of the FIFO blocked"
        assert list(out["take"]) == ["a.txt"], out
        assert out["grep"] == ["/workspace/a.txt:1: alpha"], out
        assert isinstance(out["read"], workspace_mod.PathError), out
    finally:
        shutil.rmtree(base)


if __name__ == "__main__":
    run_all(globals())
