"""Memory notes through the real handler (2026-10-05, P-M1): the fake engine in process, a
conductor, opencode-shaped requests. Needs CPU torch (the box's CPU suite).

The test matrix:

  1  memory off (the seeds; every profile stored before it): a block the model writes goes out
     as text, nothing is written, the prompt carries no memory note
                                                        test_off_leaves_the_block_alone
  2  on, streamed: the block is not in the client's content, the note is on disk (provisional,
     qwen-inline), the state line says "remembered: short answers", the journal row's extra
     names the write without its text                  test_on_streamed_write
  3  on, a block before a tool call: the block is held back, the call goes out whole
                                                        test_on_block_before_a_call
  4  SRV-277: a refused block is hidden from the answer (streamed and not), an unclosed block
     goes out as text; nothing is written           test_refused_hidden_unclosed_goes_out
  5  non-streamed: the content without the block, the note on disk
                                                        test_non_streamed_write
  6  the next turns: the card and the index go in once at the tail and replay at their place on
     the tool turn after it (the prompt starts with the one before it up to the notes' end);
     the bodies go in on a new user message, never on a tool turn
                                                        test_the_reads_and_their_replay
  7  the hidden span of a taken block: its tokens found in the committed context
                                                        test_hidden_span
  8  shadow: the card and the write, no index and no bodies
                                                        test_shadow
  9  CLIP1's fresh session (2026-10-05 21:35, live): a ratified project note, opencode's
     title request, then the question in a new session. The body goes in on the first turn
     beside the index, in its own frame (the user's notes: no "do not mention it", answer from
     the note and name it), and the stats count it as a body (they read body_notes 0 before)
                                                        test_fresh_session_recall

 10  SRV-471 (QA SRV-472): the Chat tab (client `chat-app`, the shipped profile): the turn's row
     and its usage frame carry the memory part (mode on, the index's notes); the card says the
     notes are its memory between chats and never to deny one; "remember ..." writes a note a
     new chat's index names; `memory: off` on purpose: the off note, once; absent: nothing
                                                        test_chat_tab_*

Run: python tests/test_memory_router.py
"""

from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from test_ui_states import OPENCODE_UA, TOOLS, _go, lines, serve  # noqa: E402

from server import app  # noqa: E402

SYSTEM = "You are opencode.\nWorking directory: /home/user/pagoda\nPlatform: linux"
BLOCK_OPEN = "<tandem_memory"


def setup(memory="on", states="on"):
    cond, seen = serve(states)
    for name in ("coding", "chat"):
        _, doc = cond.store.get("profiles", name)
        if memory is not None:
            doc["memory"] = memory
        cond.store.put("profiles", name, doc, note="memory", author="test")
    return cond, seen


def ask(text, history=(), stream=True, sid="ses_mem1", headers=None, **extra):
    body = {"model": "TandemLLM", "stream": stream, "max_tokens": 2000, "tools": TOOLS,
            "reasoning_effort": "medium", "reasoning_format": "reasoning_content",
            "messages": [{"role": "system", "content": SYSTEM}, *history,
                         {"role": "user", "content": text}], **extra}
    return _go(body, {"User-Agent": OPENCODE_UA, "x-session-id": sid, **(headers or {})})


def test_off_leaves_the_block_alone():
    cond, seen = setup(memory=None)
    out = ask("Please keep it short. FAKE_MEMORY")
    assert BLOCK_OPEN in out["content"], out["content"]
    assert cond.notes.notes("owner") == []
    text = app.STATE["tok"].decode(seen[-1], skip_special_tokens=False)
    assert "Read one in full" not in text and "<tandem_memory op=" not in text


def test_on_streamed_write():
    cond, seen = setup()
    out = ask("Please keep it short. FAKE_MEMORY")
    assert BLOCK_OPEN not in out["content"] and out["content"].strip(), out["content"]
    ns = cond.notes.notes("owner")
    assert len(ns) == 1 and ns[0].title == "short answers" and ns[0].status == "provisional"
    assert ns[0].writer == "qwen-inline" and ns[0].scope == "user"
    assert "remembered: short answers" in lines(out["reasoning"]) + lines(out["content"]), out
    text = app.STATE["tok"].decode(seen[-1], skip_special_tokens=False)
    assert "<tandem_memory op=" in text                      # the card went in


def test_on_block_before_a_call():
    cond, _ = setup()
    call = json.dumps({"name": "bash", "arguments": {"command": "ls -la"}})
    out = ask("Keep it short. FAKE_MEMORY_PROJECT FAKE_CALL_JSON " + call)
    assert BLOCK_OPEN not in out["content"], out["content"]
    args = "".join((c.get("function") or {}).get("arguments") or "" for c in out["calls"])
    assert json.loads(args) == {"command": "ls -la"}, out["calls"]
    ns = cond.notes.notes("owner")
    assert len(ns) == 1 and ns[0].scope == "project" and ns[0].project == "/home/user/pagoda"


def test_refused_hidden_unclosed_goes_out():
    cond, _ = setup()
    out = ask("Keep it short. FAKE_MEMORY_BAD")
    assert BLOCK_OPEN not in out["content"] and "kind=\"poem\"" not in out["content"], out["content"]
    assert out["content"].strip(), "the answer around the block still goes out"
    out = ask("Keep it short, no stream. FAKE_MEMORY_BAD", stream=False)
    assert BLOCK_OPEN not in out["content"] and out["content"].strip(), out["content"]
    out = ask("Keep it short again. FAKE_MEMORY_OPEN")
    assert BLOCK_OPEN in out["content"]
    assert cond.notes.notes("owner") == []


def test_non_streamed_write():
    cond, _ = setup()
    out = ask("Please keep it short. FAKE_MEMORY", stream=False)
    assert BLOCK_OPEN not in out["content"] and out["content"].strip()
    assert [n.title for n in cond.notes.notes("owner")] == ["short answers"]


def _tail_notes(text):
    return text.count("<runtime-note>")


def test_the_reads_and_their_replay():
    cond, seen = setup()
    cond.notes.write("owner", {"op": "add", "scope": "project", "kind": "gotcha",
                               "title": "the summer scene checker gate",
                               "body": "Gate (a) wants the summer scene within 1.0 of night 1."},
                     writer="human", project="/home/user/pagoda")
    q1 = "How is the summer scene checked by the gate?"
    a1 = ask(q1, sid="ses_mem2")
    p1 = app.STATE["tok"].decode(seen[-1], skip_special_tokens=False)
    assert "Read one in full" in p1 and "within 1.0 of night 1" in p1, p1[-1500:]
    hist = [{"role": "user", "content": q1},
            {"role": "assistant", "content": a1["content"], "reasoning_content": a1["reasoning"],
             "tool_calls": [{"id": "c1", "type": "function", "function": {
                 "name": "bash", "arguments": "{\"command\": \"ls\"}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "a.js b.js"}]
    body = {"model": "TandemLLM", "stream": True, "max_tokens": 2000, "tools": TOOLS,
            "reasoning_effort": "medium", "reasoning_format": "reasoning_content",
            "messages": [{"role": "system", "content": SYSTEM}, *hist]}
    _go(body, {"User-Agent": OPENCODE_UA, "x-session-id": "ses_mem2"})
    p2 = app.STATE["tok"].decode(seen[-1], skip_special_tokens=False)
    # replayed at their place: the second prompt starts with the first up to the notes' end
    end1 = p1.rindex("</runtime-note>") + len("</runtime-note>")
    assert p2.startswith(p1[:end1]), (p1[end1 - 300:end1], p2[end1 - 300:end1 + 100])
    assert p2.count("Read one in full") == 1 and p2.count("within 1.0 of night 1") == 1


def test_hidden_span():
    cond, seen = setup()
    ask("Please keep it short. FAKE_MEMORY", sid="ses_mem3")
    tok = app.STATE["tok"]
    ctx = app.STATE.get("last_ctx") or []
    from server import memblock
    n_prompt = len(seen[-1])
    gen = list(ctx[n_prompt:])
    _, found = memblock.strip_blocks(tok.decode(gen, skip_special_tokens=True))
    assert ctx and found, "the fake engine kept the committed context and the block"
    spans = app._memory_spans(found, gen, n_prompt)
    assert spans and tok.decode(spans[0][1], skip_special_tokens=True).strip() == \
        found[0].strip(), spans


def test_shadow():
    cond, seen = setup(memory="shadow")
    cond.notes.write("owner", {"op": "add", "scope": "user", "kind": "reference",
                               "title": "the summer scene checker gate",
                               "body": "Gate (a) wants the summer scene within 1.0 of night 1."},
                     writer="human")
    out = ask("How is the summer scene checked? FAKE_MEMORY", sid="ses_mem4")
    assert BLOCK_OPEN not in out["content"]
    p = app.STATE["tok"].decode(seen[-1], skip_special_tokens=False)
    assert "<tandem_memory op=" in p and "Read one in full" not in p and "within 1.0" not in p
    assert len(cond.notes.notes("owner")) == 2


def test_fresh_session_recall():
    from test_side_requests import opencode_title
    from server.conductor import memory as memory_mod
    cond, seen = setup()
    proj = "/tmp/demo/project"
    sysm = f"You are opencode.\nWorking directory: {proj}\nPlatform: linux"
    w = cond.notes.write("owner", {"op": "add", "scope": "project", "kind": "gotcha",
                                   "title": "build requires node 20",
                                   "body": "The project build needs node 20. Use `nvm use 20` (or "
                                           "equivalent) before running build commands."},
                         writer="qwen-inline", project=proj, from_user=True)
    cond.notes.set_status("owner", w["id"], "ratified")
    q = "Which node version does the build of this project need?"
    hdr = {"User-Agent": OPENCODE_UA, "x-session-id": "ses_clip13"}
    _go(opencode_title(q), hdr)
    before = dict(cond.memory.stats)
    body = {"model": "TandemLLM", "stream": True, "max_tokens": 2000, "tools": TOOLS,
            "reasoning_effort": "medium", "reasoning_format": "reasoning_content",
            "messages": [{"role": "system", "content": sysm}, {"role": "user", "content": q}]}
    _go(body, hdr)
    p = app.STATE["tok"].decode(seen[-1], skip_special_tokens=False)
    tail = p[p.rindex(q):]
    # the index and the body, both at the tail of the first turn's prompt
    assert "Read one in full" in tail and "nvm use 20" in tail, tail[-2000:]
    frame = tail[tail.index(memory_mod.BODIES_HEAD) - 400:]
    frame = frame[frame.rindex("<runtime-note>"):]
    assert "do not mention it" not in frame.split(memory_mod.BODIES_HEAD)[0], frame
    assert "answer from the note" in frame and "ratified" in frame
    # every other note keeps the router's wrapper
    assert tail.count("do not mention it") == tail.count("<runtime-note>") - 1, tail
    st = cond.memory.stats
    assert st["body_notes"] - before["body_notes"] == 1, (before, st)
    assert st["index_notes"] - before["index_notes"] == 1, (before, st)


def test_a_request_can_keep_out_of_memory():
    """SRV-173 (UT1, 2026-10-06; a design decision the same day: one user, every client is the
    user `owner`). A curl test's "remember ..." went into that user's notes, which is right for the one
    user; a test or a script that must not touch them says so per request. `X-Tandem-Memory: off`:
    no card, no read, no write (the journal row says why); `read`: the reads, no card, and a block
    the model writes anyway is hidden and dropped; no header: the profile's switch, as before."""
    cond, seen = setup()
    out = ask("Please keep it short. FAKE_MEMORY", sid="ses_off", headers={"X-Tandem-Memory": "off"})
    assert cond.notes.notes("owner") == [], "off: nothing written"
    text = app.STATE["tok"].decode(seen[-1], skip_special_tokens=False)
    assert "<tandem_memory op=" not in text, "off: no card"
    cond.journal.flush()
    row = cond.journal.rows(session="c:ses_off", limit=1)[0]
    assert (row.get("extra") or {}).get("memory") == {"mode": "off", "why": "header"}, row
    cond.notes.write("owner", {"op": "add", "scope": "user", "kind": "rule",
                               "title": "metric units", "body": "The user wants metric units."},
                     writer="human")
    out = ask("Which units do I prefer, metric? FAKE_MEMORY", sid="ses_read",
              headers={"X-Tandem-Memory": "read"})
    text = app.STATE["tok"].decode(seen[-1], skip_special_tokens=False)
    assert "The user wants metric units." in text, "read: the bodies go in"
    assert "<tandem_memory op=" not in text, "read: no card"
    assert BLOCK_OPEN not in out["content"], "read: the block is hidden"
    assert "remembered:" not in lines(out["reasoning"]) + lines(out["content"]), out
    assert [n.title for n in cond.notes.notes("owner")] == ["metric units"], "read: nothing added"
    out = ask("Please keep it short. FAKE_MEMORY", sid="ses_on", headers={"X-Tandem-Memory": "x"})
    assert "short answers" in [n.title for n in cond.notes.notes("owner")], "other: the profile's"


class _Headers(dict):
    """Request headers as the handler reads them: names in any case."""

    def get(self, k, d=None):
        return super().get(k.lower(), d) if isinstance(k, str) else d


def test_the_cache_peek_renders_the_memory_notes_the_next_turn_carries():
    # SRV-349 on the 27B (live, 2026-10-13): under the conductor the peek's prompt was shorter than
    # the turn's -- the session's memory notes (and the other kinds a turn turns on for its replay)
    # did not replay, so its count of tokens and of cached rows was off
    cond, seen = setup()
    cond.notes.write("owner", {"op": "add", "scope": "project", "kind": "gotcha",
                               "title": "the summer scene checker gate",
                               "body": "Gate (a) wants the summer scene within 1.0 of night 1."},
                     writer="human", project="/home/user/pagoda")
    hist = []
    for i, q in enumerate(("How is the summer scene checked by the gate?",
                           "And the winter scene, briefly?", "Which file holds the gate?")):
        body = {"model": "TandemLLM", "stream": True, "max_tokens": 2000, "tools": TOOLS,
                "reasoning_effort": "medium", "reasoning_format": "reasoning_content",
                "messages": [{"role": "system", "content": SYSTEM}, *hist,
                             {"role": "user", "content": q}]}
        peek, _ = app._peek_ids(dict(body), _Headers({"user-agent": OPENCODE_UA,
                                                      "x-session-id": "ses_peek"}))
        out = _go(body, {"User-Agent": OPENCODE_UA, "x-session-id": "ses_peek"})
        if i:                       # the first turn's own note is the router's to choose
            assert peek == seen[-1], (i, len(peek), len(seen[-1]))
            assert any(k == "memory" for n in cond.router._sessions["c:ses_peek"].notes
                       for k in n.kinds)
        hist += [{"role": "user", "content": q},
                 {"role": "assistant", "content": out["content"],
                  "reasoning_content": out["reasoning"]}]


# ---------------------------------------------------------------------- SRV-471: the Chat tab
CHAT_APP = {"X-QSE-Client": "chat-app", "User-Agent": "tandemllm-chatd/0.9"}


def _chat_app(memory=...):
    """The engine with the shipped chat-app profile (memory as given; `...`: as shipped)."""
    import copy
    from server.conductor import schema
    cond, seen = setup(memory=None)
    doc = copy.deepcopy(schema.SHIPPED_PROFILES["chat-app"])
    if memory is None:
        doc.pop("memory", None)
    elif memory is not ...:
        doc["memory"] = memory
    cond.store.put("profiles", "chat-app", doc, note="shipped chat-app", author="test")
    return cond, seen


def chat_ask(text, sid, history=()):
    body = {"model": "TandemLLM", "stream": True, "max_tokens": 2000,
            "reasoning_effort": "medium", "reasoning_format": "reasoning_content",
            "messages": [{"role": "system", "content": "You are a helpful assistant."}, *history,
                         {"role": "user", "content": text}]}
    return _go(body, {**CHAT_APP, "x-session-id": sid})


def _row(cond, sid):
    cond.journal.flush()
    return cond.journal.rows(session="c:" + sid, limit=1)[0]


def test_chat_tab_turn_sees_the_memory_index_and_reports_it():
    cond, seen = _chat_app()
    for t in ("garden: tomatoes in the south bed", "units: metric"):
        cond.notes.write("owner", {"op": "add", "scope": "user", "kind": "reference",
                                   "title": t, "body": f"The user noted {t}."}, writer="human")
    out = chat_ask("Do you know that I have a daughter?", "chat-a1")
    row = _row(cond, "chat-a1")
    assert row["profile"] == "chat-app", row
    mem = (row.get("extra") or {}).get("memory") or {}
    assert mem.get("mode") == "on" and mem["index"]["notes"] == 2, row.get("extra")
    text = app.STATE["tok"].decode(seen[-1], skip_special_tokens=False)
    assert "Your memory: notes the runtime keeps" in text and "garden: tomatoes" in text, \
        "the index went in"
    assert "your memory between conversations" in text and \
        "never say that you have no memory between chats" in text, "the card's memory line"
    part = out["usage"]["tandem"]["harness"]["memory"]
    assert part["mode"] == "on" and part["index_notes"] == 2 and part["index_tokens"] > 0, part
    assert part["card_tokens"] > 0 and part["why"] is None


def test_chat_tab_remember_then_a_new_chat_finds_it():
    cond, seen = _chat_app()
    out = chat_ask("Remember that I like short answers. FAKE_MEMORY", "chat-b1")
    assert BLOCK_OPEN not in out["content"]
    ns = cond.notes.notes("owner")
    assert [n.title for n in ns] == ["short answers"] and ns[0].status == "provisional"
    chat_ask("How should you answer me?", "chat-b2")
    text = app.STATE["tok"].decode(seen[-1], skip_special_tokens=False)
    assert "short answers" in text.split("How should you answer me?")[-1], "the new chat's index"


def test_chat_tab_memory_off_says_so_once_and_absent_adds_nothing():
    from server.conductor import memory as memory_mod
    cond, seen = _chat_app(memory="off")
    out = chat_ask("Do you remember my name?", "chat-c1")
    text = app.STATE["tok"].decode(seen[-1], skip_special_tokens=False)
    assert "memory is off in this profile" in text and "<tandem_memory op=" not in text
    assert (_row(cond, "chat-c1").get("extra") or {}).get("memory") == \
        {"mode": "off", "why": "profile"}
    assert out["usage"]["tandem"]["harness"]["memory"]["why"] == "profile"
    hist = [{"role": "user", "content": "Do you remember my name?"},
            {"role": "assistant", "content": out["content"]}]
    chat_ask("And my city?", "chat-c1", hist)
    text = app.STATE["tok"].decode(seen[-1], skip_special_tokens=False)
    assert text.count(memory_mod.OFF_NOTE[:40]) == 1, "the note once a session, replayed"
    cond, seen = _chat_app(memory=None)
    out = chat_ask("Do you remember my name?", "chat-d1")
    text = app.STATE["tok"].decode(seen[-1], skip_special_tokens=False)
    assert memory_mod.OFF_NOTE[:40] not in text and "<tandem_memory op=" not in text
    assert (_row(cond, "chat-d1").get("extra") or {}).get("memory") is None
    assert out["usage"]["tandem"]["harness"]["memory"]["why"] == "unset"


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:60s} ok")
            passed += 1
    print(f"{passed} passed")
