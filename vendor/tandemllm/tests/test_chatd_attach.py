"""SRV-487: no attachment is ever inlined past the window, and a request that does not fit names
its cause. CPU only, the fake engine.

THE CASE (live, 2026-10-15, chat 01M4ZSH3DEEX2G3X11YK1RND30): a Medium article saved as HTML,
9,455,990 bytes, its figures embedded as base64 `data:` URIs, went into the prompt whole (2,363,997
tokens) and the run failed with "Start a new chat". Now every attachment is saved under the chat's
/workspace/uploads/, a text file over the budget is one line with its path, and the error names
the attachment.

Run: python tests/test_chatd_attach.py
"""

from __future__ import annotations

import base64
import json
import os

from chatd_util import Chat, run_all

from chatd import attach, render
from chatd.agent import too_long_words


def upload(c: Chat, name: str, mime: str, data: bytes) -> dict:
    b = "XyZ"
    raw = (f"--{b}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{name}\"\r\n"
           f"Content-Type: {mime}\r\n\r\n").encode() + data + f"\r\n--{b}--\r\n".encode()
    return c.ok("POST", "/attachments", raw=raw,
                headers={"Content-Type": f"multipart/form-data; boundary={b}"})


def test_the_budget_is_a_quarter_of_the_free_room_at_most_32k():
    assert attach.budget(262_144, 0) == 32_000
    assert attach.budget(262_144, 200_000) == 15_536
    assert attach.budget(262_144, 300_000) == 0
    assert attach.budget(0, 10**9) == 32_000                # the window unknown


def test_an_html_page_with_embedded_images_goes_as_one_line_and_a_file():
    c = Chat()
    try:
        cid = c.new()
        blob = base64.b64encode(os.urandom(600_000)).decode()
        page = (f"<html><body><h1>TandemLLM and StairCut</h1><img src=\"data:image/gif;base64,"
                f"{blob}\"></body></html>").encode()
        att = upload(c, "TandemLLM-and-StairCut.html", "text/html", page)
        assert att["kind"] == "text" and att["tokens"] > 200_000 and att["inline"] is False
        c.ok("POST", f"/conversations/{cid}/messages",
             {"client_msg_id": "a1", "text": "write an X post about this article",
              "attachments": [att["id"]]})
        s = c.wait_idle(cid)
        assert s["messages"][-1]["status"] == "done", s["messages"][-1]
        sent = c.last_body()["messages"][-1]["content"]
        assert sent == ("write an X post about this article\n\n[Attached file TandemLLM-and-"
                        f"StairCut.html: 781 KB, about {att['tokens']:,} tokens: too long to "
                        "include. It is saved at /workspace/uploads/TandemLLM-and-StairCut.html.]")
        a = s["messages"][0]["attachments"][0]
        assert a["inline"] is False and a["path"] == "/workspace/uploads/TandemLLM-and-StairCut.html"
        assert open(os.path.join(c.data, "workspaces", cid, "uploads",
                                 "TandemLLM-and-StairCut.html"), "rb").read() == page
        assert any("[attach]" in ln and "over the budget" in ln for ln in c.lines)
        assert str(att["tokens"]) not in json.dumps(c.snap(cid)["context"]), \
            "the line counts in the window, not the file"
        # the next turn renders the same bytes for that message (the engine's prefix cache)
        c.ok("POST", f"/conversations/{cid}/messages", {"client_msg_id": "a2", "text": "shorter"})
        c.wait_idle(cid)
        assert [m for m in c.last_body()["messages"] if m["role"] == "user"][0]["content"] == sent
    finally:
        c.close()


def test_a_small_file_is_inlined_as_before_and_saved_too():
    c = Chat()
    try:
        cid = c.new()
        att = upload(c, "notes.txt", "text/plain", b"line one\nline two")
        assert att["inline"] is True
        img = upload(c, "shot.png", "image/png", b"\x89PNG\r\n\x1a\nfake")
        c.ok("POST", f"/conversations/{cid}/messages",
             {"client_msg_id": "a1", "text": "read it", "attachments": [att["id"], img["id"]]})
        s = c.wait_idle(cid)
        assert c.last_body()["messages"][-1]["content"] == \
            "read it\n\n```notes.txt\nline one\nline two\n```"
        got = s["messages"][0]["attachments"]
        assert [g.get("path") for g in got] == ["/workspace/uploads/notes.txt",
                                               "/workspace/uploads/shot.png"]
        assert got[0]["inline"] is True and "inline" not in got[1]
        up = os.path.join(c.data, "workspaces", cid, "uploads")
        assert sorted(os.listdir(up)) == ["notes.txt", "shot.png"]
    finally:
        c.close()


def test_two_attachments_share_one_budget_and_names_never_clash():
    c = Chat()
    try:
        cid = c.new()
        body = ("x" * 99 + "\n").encode() * 1_000            # 25,000 tokens each
        a1 = upload(c, "notes.txt", "text/plain", body)
        a2 = upload(c, "notes.txt", "text/plain", body)
        c.ok("POST", f"/conversations/{cid}/messages",
             {"client_msg_id": "a1", "text": "compare", "attachments": [a1["id"], a2["id"]]})
        s = c.wait_idle(cid)
        got = s["messages"][0]["attachments"]
        assert [g["inline"] for g in got] == [True, False]
        assert [g["path"] for g in got] == ["/workspace/uploads/notes.txt",
                                           "/workspace/uploads/notes-2.txt"]
    finally:
        c.close()


def test_an_edit_measures_the_budget_on_the_branch_it_goes_on():
    c = Chat()
    try:
        real = c.svc.model_info
        c.svc.model_info = lambda mid: {**real(mid), "max_len": 65_536}
        cid = c.new()
        c.send(cid, "hi")
        first = c.wait_idle(cid)["messages"][-1]["id"]
        c.send(cid, "word " * 30_000)                       # the head's branch: ~30,000 tokens
        c.wait_idle(cid)
        att = upload(c, "notes.txt", "text/plain", ("x" * 99 + "\n").encode() * 480)
        assert 8_848 < att["tokens"] < 16_356, att          # over the head's budget, not the edit's
        c.ok("POST", f"/conversations/{cid}/messages",
             {"client_msg_id": "e1", "text": "read this", "parent_id": first,
              "attachments": [att["id"]]})
        s = c.wait_idle(cid)
        assert s["messages"][-2]["attachments"][0]["inline"] is True, s["messages"][-2]
        assert "```notes.txt\n" in c.last_body()["messages"][-1]["content"]
    finally:
        c.close()


def test_an_attachment_from_before_renders_as_it_did():
    data = b"line one\nline two"
    old = {"id": "a", "name": "notes.txt", "kind": "text", "tokens": 5}
    msg = {"role": "user", "parts": [{"type": "text", "text": "read it"}], "attachments": [old]}
    assert render._user(msg, False, lambda a: data)["content"] == \
        "read it\n\n```notes.txt\nline one\nline two\n```"
    huge = dict(old, tokens=2_363_997, bytes=9_455_990)
    out = render._user(dict(msg, attachments=[huge]), False, lambda a: data)["content"]
    assert out == ("read it\n\n[Attached file notes.txt: 9.0 MB, about 2,363,997 tokens: too long "
                   "to include. It was attached before attachments were saved to the workspace.]")


def test_a_request_that_does_not_fit_names_the_attachment():
    user = {"role": "user", "parts": [{"type": "text", "text": "Here is the article of medium"}],
            "attachments": [{"name": "TandemLLM-and-StairCut.html", "kind": "text",
                             "tokens": 2_363_997, "inline": True}]}
    w = too_long_words(2_875_364, 262_144, [user], {"messages": []}, None)
    assert w == ("This request is about 2,875,364 tokens and the model's window is 262,144: there "
                 "is no room left for an answer. The cause is the attachment TandemLLM-and-"
                 "StairCut.html in your message (\"Here is the article of medium\"): about "
                 "2,363,997 tokens. Edit that message and leave the file out, or attach a "
                 "shorter one.")
    assert "Start a new chat" not in w
    body = {"messages": [{"role": "system", "content": "s"},
                         {"role": "tool", "tool_call_id": "c", "content": "x" * 400_000}]}
    w = too_long_words(270_000, 262_144, [], body, None)
    assert "The largest part is a tool result, about 1" in w and "branch off before it" in w


if __name__ == "__main__":
    run_all(globals())
