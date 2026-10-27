"""The Memory screen's API over the session transcripts (VIS-48, QA VIS-49): the session list, one
transcript paged at 200 blocks, a transcript read while it is written, a session without
meta.json, the keep toggle the prune honours, the exports, and the admin token on every memory
route, reads included.

The Gherkin of VIS-48 on the server's side:
  1 sessions list        test_1_* (five sessions: date, client, turns, acts, checker, keep)
  2 transcript viewer    test_2_* (blocks with their header fields; hidden notes keep their
                          journal rows; a compaction block is marked; 3,000 blocks page at 200)
  3 keep toggle          test_3_* (meta.json.keep, and the prune leaves the session)
  4 phone layout         the dashboard's tests (dashboard/tests/memory.test.ts, e2e/memory.spec.ts)
All text is synthetic.

Run: python tests/test_memory_api.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ["TZ"] = "Europe/Berlin"
time.tzset()

from server.conductor import transcripts as T  # noqa: E402
from tests.test_transcripts import (TS, _plan, _rec, _turn_messages, _writer)  # noqa: E402


def _session(w, key, n=1, ts=TS, result="ok", **plan_kw):
    d = None
    for i in range(n):
        d = w.write(w.job(_plan(_turn_messages(f"{result}\nrun {i}"), key=key, **plan_kw),
                          _rec(f"</think>\n\nanswer {i}", ts=ts + i, finish="stop")))
    return d


# ------------------------------------------------------------------------- 1 the list
def test_1_five_sessions_list_newest_first_with_their_counts():
    w = _writer()
    for i in range(5):
        _session(w, f"c:ses_{i}", n=i + 1, ts=TS + i * 3600,
                 result="calccheck minicalc\nRESULT: PASS" if i == 4 else "ok")
    rows = w.sessions()
    assert [r["session"] for r in rows] == [f"c:ses_{i}" for i in (4, 3, 2, 1, 0)]
    top = rows[0]
    assert top["counts"]["blocks"] == 5 and top["client"] == "opencode" and top["keep"] is False
    assert top["checkers"]["PASS"] == 5 and top["checkers"]["last"] == "calccheck minicalc PASS"
    assert top["month"] == "2026-09" and top["has_meta"] is True


def test_1_a_session_without_meta_json_is_listed_with_unknown_counts():
    w = _writer()
    d = _session(w, "c:lost", n=2)
    os.remove(os.path.join(d, "meta.json"))
    row = next(r for r in w.sessions() if r["dir"] == "c-lost")
    assert row["counts"] is None and row["session"] is None and row["has_meta"] is False
    got = w.read("c:lost")
    assert [b["turn"] for b in got["blocks"]] == [1, 2], "split at the block headers"


# ------------------------------------------------------------------------- 2 the viewer
def test_2_blocks_carry_their_header_fields():
    w = _writer()
    d = w.write(w.job(_plan(_turn_messages("ok")),
                      _rec("</think>\n\n## Goal\n### user\nnot a heading of ours")))
    w.write(w.job(_plan([{"role": "user", "content": "x" * 3000}], compaction=True, tools=False,
                        row={"class_id": "summarize", "effort": None}),
                  _rec("</think>\n\nsummary", finish="stop")))
    got = w.read("c:ses_test")
    b1, b2 = got["blocks"]
    assert (b1["turn"], b1["class_id"], b1["effort"], b1["reasoning_tokens"], b1["compaction"]) == \
        (1, "code_edit", "xhigh", 1234, False)
    assert b1["time"] == "2026-09-21 16:13:20" and b1["request_id"].startswith("chatcmpl-")
    # a message line that reads like one of the block's own headings is escaped, and the
    # offsets keep the block whole
    assert "## Goal\n\\### user\nnot a heading of ours" in b1["text"], b1["text"]
    assert b2["compaction"] and b2["turn"] == 2
    assert "offsets" not in got["session"]
    assert os.path.exists(d)


def test_2_three_thousand_blocks_page_at_two_hundred():
    w = _writer()
    d = _session(w, "c:long", n=1)
    meta = T.read_meta(d)
    text = open(os.path.join(d, "transcript.md")).read()
    with open(os.path.join(d, "transcript.md"), "a") as f:
        for i in range(2, 3001):
            meta["offsets"].append(f.tell())
            f.write(text.replace("## turn 1 ·", f"## turn {i} ·"))
        meta["bytes"] = f.tell()
    T.write_meta(d, meta)
    last = w.read("c:long")
    assert (last["page"], last["pages"], last["total"], len(last["blocks"])) == (15, 15, 3000, 200)
    assert last["blocks"][-1]["turn"] == 3000
    first = w.read("c:long", page=1)
    assert [b["turn"] for b in first["blocks"]][:3] == [1, 2, 3] and len(first["blocks"]) == 200
    assert w.read("c:long", page=99)["page"] == 15


def test_2_a_transcript_being_written_is_read_to_its_last_complete_block():
    w = _writer()
    d = _session(w, "c:busy", n=2)
    with open(os.path.join(d, "transcript.md"), "a") as f:
        f.write("## turn 3 · 2026-09-21 16:13:22 · request chatcmpl-half · class code")
    got = w.read("c:busy")
    assert [b["turn"] for b in got["blocks"]] == [1, 2]
    _, text = w.export("c:busy")
    assert "chatcmpl-half" not in text and text.count("## turn ") == 2


# ------------------------------------------------------------------------- 3 keep
def test_3_keep_is_written_to_meta_json_and_the_prune_leaves_the_session():
    w = _writer()
    d = _session(w, "c:keepme")
    assert w.set_keep("c:keepme", True) == {"session": "c:keepme", "keep": True}
    assert T.read_meta(d)["keep"] is True
    assert w.journal.rows[-1]["task"] == "keep" and w.journal.rows[-1]["decision"] == "on"
    assert w.prune(TS + 400 * 86400) == {"deleted": 0, "kept": 1} and os.path.exists(d)
    w.set_keep("c:keepme", False)
    assert w.prune(TS + 400 * 86400) == {"deleted": 1, "kept": 0} and not os.path.exists(d)
    assert w.set_keep("c:nobody", True) is None


def test_3_keep_and_a_block_do_not_lose_each_other():
    w = _writer()
    _session(w, "c:race")
    w.set_keep("c:race", True)
    d = _session(w, "c:race")
    meta = T.read_meta(d)
    assert meta["keep"] is True and meta["counts"]["blocks"] == 2


# ------------------------------------------------------------------------- the fake engine
def test_the_memory_routes_on_the_fake_engine():
    from tests.test_conductor_e2e import PROXY, TOOLS, _call, _check, _port, _server, _wait
    root = tempfile.mkdtemp(prefix="qse-mem-")
    cdir = os.path.join(root, "conductor")
    port = _port()
    proc, _ = _server(port, cdir)
    try:
        _wait(port, proc)
        for i, s in enumerate(("ses_a", "ses_b")):
            st, _, _ = _call(port, "POST", "/v1/chat/completions",
                             {"model": "TandemLLM", "tools": TOOLS, "max_tokens": 200,
                              "messages": [{"role": "user", "content": f"run the tool {i}"}]},
                             headers={"x-session-id": s})
            assert st == 200
        deadline = time.time() + 10
        lst = None
        while time.time() < deadline:
            st, _, lst = _call(port, "GET", "/v1/conductor/memory/sessions")
            # a first block is on disk a moment before its meta.json: wait for both
            if st == 200 and sum(s["has_meta"] for s in lst["sessions"]) == 2:
                break
            time.sleep(0.2)
        _check(lst, "memory-sessions")
        assert {s["session"] for s in lst["sessions"]} == {"c:ses_a", "c:ses_b"}, lst
        assert lst["writer"]["blocks"] == 2 and lst["writer"]["settings"]["scope"] == "tools"
        st, _, one = _call(port, "GET", "/v1/conductor/memory/sessions/c:ses_a")
        assert st == 200, one
        _check(one, "memory-session")
        assert one["blocks"][0]["turn"] == 1 and "### tool call" in one["blocks"][0]["text"]
        st, _, k = _call(port, "PATCH", "/v1/conductor/memory/sessions/c:ses_a/keep",
                         {"keep": True})
        assert st == 200 and k["keep"] is True, k
        _check(k, "memory-keep")
        st, _, md = _call(port, "GET", "/v1/conductor/memory/sessions/c:ses_a/export?format=md",
                          raw=True)
        assert st == 200 and md.decode().startswith("## turn 1 · ")
        st, _, js = _call(port, "GET", "/v1/conductor/memory/sessions/c:ses_a/export?format=json")
        _check(js, "memory-session")
        st, _, ex = _call(port, "GET", "/v1/conductor/sessions/c:ses_a/export?format=json")
        assert st == 200 and ex["transcript"] == {
            "blocks": 1, "export": "/v1/conductor/memory/sessions/c:ses_a/export"}
        _check(ex, "export")
        assert _call(port, "GET", "/v1/conductor/memory/sessions/c:nobody")[0] == 404
        assert _call(port, "PATCH", "/v1/conductor/memory/sessions/c:ses_a/keep",
                     {"keep": "yes"})[0] == 400
        # the transcripts need the admin token from off the box, reads included; the session
        # list of the journal follows the dashboard's rule as before
        for path in ("/v1/conductor/memory/sessions", "/v1/conductor/memory/sessions/c:ses_a",
                     "/v1/conductor/memory/sessions/c:ses_a/export?format=md"):
            st, _, err = _call(port, "GET", path, headers=PROXY, token=None)
            assert st == 401, (path, st, err)
        assert _call(port, "PATCH", "/v1/conductor/memory/sessions/c:ses_a/keep", {"keep": False},
                     headers=PROXY, token=None)[0] == 401
        assert _call(port, "GET", "/v1/conductor/memory/sessions", headers=PROXY)[0] == 200
    finally:
        proc.kill()
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:64s} ok")
            passed += 1
    print(f"{passed} passed")
