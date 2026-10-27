"""Per-session transcripts on the NVMe (SRV-137, QA SRV-138): the block format against golden
files, the compaction block, the tools-only scope, the prune with keep, the secret-pattern
refusal over 23 synthetic secret shapes, a write that fails, and the files staying under the
memory root with mode 0600.

The Gherkin of SRV-137 is the matrix:
  1 one block per request              test_1_* (golden file; hidden notes with their journal
                                        rows, the clip with its size, the checker line, meta.json)
  2 compaction block                   test_2_* (golden file)
  3 tools-only scope                   test_3_* (unit; the fake engine: a chat without tools
                                        writes counters, a request with tools a block)
  4 prune and keep                     test_4_* (faked times)
  5 secrets are never stored           test_5_* (every shape redacted and counted; on the fake
                                        engine a grep of memory/ finds none of the values and the
                                        journal holds the counts, never a value)
  6 a write failure does not fail the request   test_6_* (unit; the fake engine with a read-only
                                        memory directory)
All text here is synthetic. Golden files: tests/fixtures/transcripts/*.md; regenerate one by
deleting it and running this file (it writes the file and fails once, so a change is a diff to read).

Run: python tests/test_transcripts.py
"""

from __future__ import annotations

import json
import os
import random
import re
import shutil
import sqlite3
import stat
import sys
import tempfile
import time
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ["TZ"] = "Europe/Berlin"
time.tzset()

from server.conductor import redact, schema, transcripts as T  # noqa: E402

FIX = os.path.join(ROOT, "tests", "fixtures", "transcripts")
TS = 1790000000.0                                   # 2026-09-21 16:13:20 in Berlin
SYSTEM = "You are opencode.\n<env>\n  Working directory: /work/minicalc\n</env>"


# ------------------------------------------------------------------------- helpers
class Tok:
    """One id per character, as the fake engine's tokenizer."""

    @staticmethod
    def decode(ids, skip_special_tokens=False):
        return "".join(chr(i) for i in ids)


class Journal:
    def __init__(self):
        self.rows = []

    def note(self, kind, task, **kw):
        self.rows.append(dict(kw, kind=kind, task=task))


def _writer(root=None, **settings) -> T.Transcripts:
    root = root or os.path.join(tempfile.mkdtemp(prefix="mem-"), "memory")
    doc = {"memory": {"transcripts": dict(T.DEFAULTS, **settings)}}
    return T.Transcripts(root, settings=lambda: doc, journal=Journal(), tok=lambda: Tok)


def _plan(messages, *, key="c:ses_test", transcript="full", compaction=False, injected=(),
          row=None, tools=True):
    body = {"model": "TandemLLM", "messages": messages}
    if tools:
        body["tools"] = [{"type": "function", "function": {"name": "bash"}}]
    return types.SimpleNamespace(
        body=body, transcript=transcript, compaction=compaction, injected=list(injected),
        session=types.SimpleNamespace(key=key), res=types.SimpleNamespace(profile="coding"),
        row=dict({"id": "9f3c2a1b0c4d5e6f", "class_id": "code_edit", "effort": "xhigh",
                  "replayed": {"count": 2, "tokens": 300}}, **(row or {})))


def _rec(output: str, *, calls=(), in_think=True, events=(), rid="chatcmpl-0123456789abcdef01234567",
         finish="tool_calls", ts=TS):
    return types.SimpleNamespace(
        ts=ts, request_id=rid, client_kind="opencode", out=([ord(c) for c in output], in_think),
        out_calls=list(calls), finish_reason=finish, prompt_tokens=52311,
        completion_tokens=len(output), reasoning_tokens=1234,
        sup=types.SimpleNamespace(events=list(events)) if events else None)


def _golden(name: str, text: str) -> None:
    path = os.path.join(FIX, name)
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write(text)
        raise AssertionError(f"wrote {path}; read it, then run again")
    with open(path) as f:
        want = f.read()
    assert text == want, f"{name} differs from its golden file:\n{text[:2000]}"


def _secrets(seed: int = 7) -> list[dict]:
    """The fixture's shapes with their values filled in (seeded, synthetic)."""
    rng = random.Random(seed)
    sets = {"A": "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
            "U": "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
            "L": "abcdefghijklmnopqrstuvwxyz0123456789",
            "B": "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"}

    def fill(t: str) -> str:
        return re.sub(r"\{([AULB]):(\d+)\}|\{p\}",
                      lambda m: "p" if m.group(0) == "{p}" else
                      "".join(rng.choice(sets[m.group(1)]) for _ in range(int(m.group(2)))), t)

    with open(os.path.join(FIX, "secret-shapes.json")) as f:
        shapes = json.load(f)["shapes"]
    out = []
    for s in shapes:
        secret = fill(s["secret"])
        out.append({"kind": s["kind"], "secret": secret,
                    "text": fill(s["text"]).replace("{secret}", secret)})
    return out


def _turn_messages(result: str) -> list[dict]:
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": "Add a power operator to the calculator and run the checks."},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function",
             "function": {"name": "bash", "arguments": "{\"command\": \"python check.py\"}"}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": result}]


EDIT = {"id": "call_2", "type": "function", "function": {
    "name": "edit", "arguments": "{\"filePath\": \"calc.py\", \"oldString\": \"a\", \"newString\": \"b\"}"}}
RESULT = ("calccheck minicalc\nRESULT: FAIL <- (a) 3 of 5 cases pass; (b) power missing\n"
          + "".join(f"case {i}: 2 ** {i} expected {2 ** i}\n" for i in range(400)))


# ------------------------------------------------------------------------- 1 one block per request
def test_1_one_block_per_request_matches_the_golden_file():
    w = _writer()
    events = [{"mode": "act", "task": "D1", "decision": "CLOSE_REASONING", "text": "Stop and act.",
               "hidden_tokens": 24, "pos": None, "journal_id": "ab12cd34ef56ab12"},
              {"mode": "shadow", "task": "C1", "decision": "BLOCK_CALL"}]
    out = ("I keep checking the same case again. " * 30 + "</think>\n\nI will add the operator.")
    plan = _plan(_turn_messages(RESULT), injected=[{"pos": 20480, "tokens": 182, "parts": [
        {"kind": "effort", "name": "xhigh"}, {"kind": "rules", "name": "code_edit"}]}])
    d = w.write(w.job(plan, _rec(out, calls=[EDIT], events=events)))
    text = open(os.path.join(d, "transcript.md")).read()
    _golden("block-tools.md", text)
    assert "I keep checking" in text, "the D1-cut loop keeps its start and end"
    assert text.count("I keep checking the same case again.") < 30
    meta = json.load(open(os.path.join(d, "meta.json")))
    assert meta["session"] == "c:ses_test" and meta["project"] == "/work/minicalc"
    assert meta["counts"]["blocks"] == 1 and meta["counts"]["clipped"] == 1
    assert meta["counts"]["hidden_notes"] == 2 and meta["counts"]["acts"] == 1
    assert meta["checkers"] == {"FAIL": 1, "last": "calccheck minicalc FAIL"}
    assert meta["bytes"] == len(text.encode()) and meta["keep"] is False
    assert d == os.path.join(w.root, "sessions", "2026-09", "c-ses_test")
    # a second request of the session appends the next block after the first one, in the same
    # directory, from its own new messages only
    msgs = _turn_messages(RESULT) + [
        {"role": "assistant", "content": "I will add the operator.", "tool_calls": [EDIT]},
        {"role": "tool", "tool_call_id": "call_2", "content": "Edit applied."}]
    d2 = w.write(w.job(_plan(msgs), _rec("</think>\n\nDone: the power operator works.",
                                         finish="stop", ts=TS + 40 * 86400)))
    assert d2 == d
    text2 = open(os.path.join(d, "transcript.md")).read()
    assert text2.startswith(text) and "## turn 2 ·" in text2
    second = text2[len(text):]
    assert "### tool result (edit, 13 chars)\nEdit applied." in second and "RESULT" not in second
    assert json.load(open(os.path.join(d, "meta.json")))["counts"]["blocks"] == 2


def test_1_the_files_are_0600_under_the_memory_root():
    w = _writer()
    d = w.write(w.job(_plan(_turn_messages("ok")), _rec("</think>\n\nok", finish="stop")))
    for f in ("transcript.md", "meta.json"):
        assert stat.S_IMODE(os.stat(os.path.join(d, f)).st_mode) == 0o600, f
    for p in (w.root, w.sessions_dir, d):
        assert stat.S_IMODE(os.stat(p).st_mode) == 0o700, p
    assert os.path.realpath(d).startswith(os.path.realpath(w.root) + os.sep)


def test_1_the_first_turn_writes_the_user_message_and_a_whole_answer():
    w = _writer()
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": [{"type": "text", "text": "What is in this picture?"},
                                         {"type": "image_url", "image_url": {"url": "data:,"}}]}]
    d = w.write(w.job(_plan(msgs), _rec("short thought</think>\n\nA red pagoda.", finish="stop")))
    text = open(os.path.join(d, "transcript.md")).read()
    assert "### user\nWhat is in this picture?\n[image]" in text
    assert "### assistant\nA red pagoda." in text and "short thought" not in text


def test_new_messages_and_names():
    msgs = _turn_messages("x")
    assert [m["role"] for m in T.new_messages(msgs)] == ["tool"]
    assert [m["role"] for m in T.new_messages(msgs[:2])] == ["user"]
    assert T.tool_names(msgs) == {"call_1": "bash"}
    assert T.dir_name("c:ses_ab12") == "c-ses_ab12"
    a, b = T.dir_name("c:a/b"), T.dir_name("c:a_b")
    assert a != b and "/" not in a and a.startswith("c-a_b-")
    assert T.split_reasoning("think</think>\n\nanswer", True) == ("answer", "think")
    assert T.split_reasoning("ran out", True) == ("", "ran out")
    assert T.split_reasoning("<think>\n\n</think>\n\nanswer", False) == ("answer", "\n\n")


# ------------------------------------------------------------------------- 2 compaction
def test_2_a_compaction_request_is_one_block_with_its_summary():
    w = _writer()
    history = "user: build a calculator\nassistant: ...\n" * 200
    plan = _plan([{"role": "user", "content": history + "Summarize the conversation."}],
                 compaction=True, row={"class_id": "summarize", "effort": None, "replayed": None},
                 tools=False)
    rec = _rec("</think>\n\n## Goal\nA calculator with a power operator.\n## Done\nParser, tests.",
               finish="stop")
    rec.prompt_tokens, rec.completion_tokens = 120200, 4100
    d = w.write(w.job(plan, rec))
    text = open(os.path.join(d, "transcript.md")).read()
    _golden("block-compaction.md", text)
    assert "build a calculator" not in text, "the history itself is not copied again"
    assert json.load(open(os.path.join(d, "meta.json")))["counts"]["compactions"] == 1


# ------------------------------------------------------------------------- 3 tools-only scope
def test_3_the_scope_decides_full_counters_or_nothing():
    tools = dict(T.DEFAULTS)
    assert T.Transcripts.scope_of(tools, True, False) == "full"
    assert T.Transcripts.scope_of(tools, False, True) == "full", "a compaction is written"
    assert T.Transcripts.scope_of(tools, False, False) == "counters"
    assert T.Transcripts.scope_of(dict(tools, scope="all"), False, False) == "full"
    assert T.Transcripts.scope_of(dict(tools, scope="none"), True, False) is None
    assert schema.seed("settings")["memory"]["transcripts"]["scope"] == "tools"
    assert T.settings_of({"journal": {}}) == T.DEFAULTS, "a settings version from before"


def test_3_a_chat_without_tools_changes_only_the_counters():
    w = _writer()
    plan = _plan([{"role": "user", "content": "hello there"}], transcript="counters", tools=False)
    d = w.write(w.job(plan, _rec("</think>\n\nhi", finish="stop")))
    assert not os.path.exists(os.path.join(d, "transcript.md"))
    meta = json.load(open(os.path.join(d, "meta.json")))
    assert meta["counts"]["requests"] == 1 and meta["counts"]["untranscribed"] == 1
    assert meta["counts"]["blocks"] == 0 and "hello" not in json.dumps(meta)


# ------------------------------------------------------------------------- 4 prune and keep
def test_4_the_prune_deletes_old_sessions_and_keeps_the_kept():
    w = _writer()
    now = TS + 31 * 86400
    old = w.write(w.job(_plan(_turn_messages("a"), key="c:old"), _rec("</think>\n\na")))
    kept = w.write(w.job(_plan(_turn_messages("b"), key="c:kept"), _rec("</think>\n\nb")))
    new = w.write(w.job(_plan(_turn_messages("c"), key="c:new"),
                        _rec("</think>\n\nc", ts=now - 3600)))
    meta = T.read_meta(kept)
    meta["keep"] = True
    T.write_meta(kept, meta)
    got = w.prune(now)
    assert got == {"deleted": 1, "kept": 1}
    assert not os.path.exists(old) and os.path.exists(kept) and os.path.exists(new)
    assert w.journal.rows[-1]["kind"] == "memory" and w.journal.rows[-1]["detail"] == {
        "deleted": 1, "kept": 1, "retention_days": 30}
    # a new block of a pruned session starts a directory of its own month
    d = w.write(w.job(_plan(_turn_messages("d"), key="c:old"), _rec("</think>\n\nd", ts=now)))
    assert os.path.basename(os.path.dirname(d)) == "2026-10"


# ------------------------------------------------------------------------- 5 secrets
def test_5_every_secret_shape_is_redacted_and_counted():
    shapes = _secrets()
    assert len(shapes) >= 20 and len({s["kind"] for s in shapes}) >= 18
    for s in shapes:
        out, counts = redact.redact(s["text"])
        assert s["secret"] not in out, (s["kind"], out)
        assert f"[redacted:{s['kind']}]" in out and counts.get(s["kind"], 0) >= 1, (s["kind"], out)
    # code that only names a secret is left alone
    for code in ("token = tok(text)", "password: str", "MY_TOKEN=$TOKEN", "def secret(): pass",
                 "--password=$PW", "AKIA is a prefix"):
        assert redact.redact(code) == (code, {}), code


def test_5_a_block_never_holds_a_secret_and_the_journal_holds_counts():
    shapes = _secrets()
    w = _writer()
    big = "\n".join(s["text"] for s in shapes)
    d = w.write(w.job(_plan(_turn_messages(big)),
                      _rec("</think>\n\nThe key is " + shapes[9]["secret"], finish="stop")))
    blob = b"".join(open(os.path.join(dp, f), "rb").read()
                    for dp, _, fs in os.walk(w.root) for f in fs)
    for s in shapes:
        for line in s["secret"].splitlines():
            if len(line) >= 8:
                assert line.encode() not in blob, s["kind"]
    text = open(os.path.join(d, "transcript.md")).read()
    n = len(shapes) + 1
    assert re.search(rf"^redactions: {n} \(", text, re.M), text[-800:]
    row = w.journal.rows[-1]
    assert row["decision"] == "block" and row["detail"]["redactions"] == n
    assert sum(row["detail"]["redacted"].values()) == n
    assert not any(s["secret"] in json.dumps(w.journal.rows) for s in shapes)


def test_5_a_key_across_the_clip_is_redacted_before_the_clip():
    w = _writer(clip_chars=300)
    key = _secrets()[0]["secret"]
    d = w.write(w.job(_plan(_turn_messages("x" * 250 + key)), _rec("</think>\n\nok")))
    text = open(os.path.join(d, "transcript.md")).read()
    assert "-----BEGIN RSA" not in text and "[redacted:private_key]" in text


# ------------------------------------------------------------------------- 6 a failed write
def test_6_a_write_that_fails_is_journaled_and_raises_nothing():
    w = _writer()
    os.makedirs(w.root, mode=0o500)
    try:
        assert w.write(w.job(_plan(_turn_messages("a")), _rec("</think>\n\na"))) is None
    finally:
        os.chmod(w.root, 0o700)
    assert w.stats["failed"] == 1
    row = w.journal.rows[-1]
    assert (row["kind"], row["task"], row["decision"]) == ("memory", "transcript", "write_failed")
    assert "Permission" in row["detail"]["error"]


def test_6_a_root_inside_a_repository_is_refused():
    repo = tempfile.mkdtemp(prefix="repo-")
    os.makedirs(os.path.join(repo, ".git"))
    w = _writer(os.path.join(repo, "memory"))
    assert w.repo == os.path.realpath(repo)
    assert not w.submit(_plan(_turn_messages("a")), _rec("a"))
    assert not os.path.exists(os.path.join(repo, "memory"))


def test_the_memory_root_is_the_engine_home_or_inside_the_config():
    from server.conductor.store import DEFAULT_ROOT
    assert T.memory_root(DEFAULT_ROOT) == os.path.realpath(os.path.expanduser(
        "~/.qwen38-spark-engine/memory"))
    tmp = tempfile.mkdtemp(prefix="cond-")
    assert T.memory_root(tmp) == os.path.join(os.path.realpath(tmp), "memory")


def test_the_queue_writes_off_the_request_thread():
    w = _writer()
    plan = _plan(_turn_messages("queued"))
    assert w.submit(plan, _rec("</think>\n\nqueued"))
    w.flush()
    assert w.stats["blocks"] == 1 and w._thread is not None
    assert plan.transcript == "full"


def test_the_weekly_backup_holds_no_transcript_and_no_memory_row():
    from server.conductor.decisions import Journal as CondJournal
    from server.ledger import Ledger
    home = tempfile.mkdtemp(prefix="home-")
    led = Ledger(os.path.join(home, "usage", "ledger.sqlite3"))
    led.open()
    j = CondJournal(led.path)
    w = T.Transcripts(os.path.join(home, "memory"), settings=lambda: {}, journal=j,
                      tok=lambda: Tok)
    w.write(w.job(_plan(_turn_messages("a")), _rec("</think>\n\na")))
    j.flush()
    dest = led.backup()
    led.close()
    j.close()
    assert all(f.startswith("ledger-") for f in os.listdir(os.path.dirname(dest)))
    conn = sqlite3.connect(dest)
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    conn.close()
    assert "conductor_events" not in tables


# ------------------------------------------------------------------------- the fake engine
def _fake(port, cdir):
    from tests.test_conductor_e2e import _server, _wait
    proc, ledger = _server(port, cdir)
    _wait(port, proc)
    return proc, ledger


def _chat(port, messages, tools=True, session="ses_e2e"):
    from tests.test_conductor_e2e import TOOLS, _call
    body = {"model": "TandemLLM", "messages": messages, "max_tokens": 300}
    if tools:
        body["tools"] = TOOLS
    return _call(port, "POST", "/v1/chat/completions", body, headers={"x-session-id": session})


def _memory_rows(ledger):
    conn = sqlite3.connect(ledger)
    rows = conn.execute("SELECT task, decision, session_key, detail FROM conductor_events WHERE "
                        "kind = 'memory'").fetchall()
    conn.close()
    return [(t, d, s, json.loads(x) if x else None) for t, d, s, x in rows]


def test_3_5_on_the_fake_engine_scope_and_secrets():
    from tests.test_conductor_e2e import _port
    root = tempfile.mkdtemp(prefix="qse-mem-")
    cdir = os.path.join(root, "conductor")
    port = _port()
    proc, ledger = _fake(port, cdir)
    shapes = _secrets(11)
    try:
        st, _, r = _chat(port, [{"role": "user", "content": "hello, just chatting"}], tools=False,
                         session="ses_chat")
        assert st == 200, r
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": "Read the config and run the tool."},
                {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function",
                 "function": {"name": "bash", "arguments": "{\"command\": \"cat .env\"}"}}]},
                {"role": "tool", "tool_call_id": "c1",
                 "content": shapes[2]["text"] + "\n" + shapes[9]["text"] + "\n" + shapes[0]["text"]}]
        st, _, r = _chat(port, msgs)
        assert st == 200 and r["choices"][0]["message"].get("tool_calls"), r
        mem = os.path.join(cdir, "memory", "sessions")
        deadline = time.time() + 10
        while time.time() < deadline and len(_memory_rows(ledger)) < 1:
            time.sleep(0.2)
        months = os.listdir(mem)
        chat = os.path.join(mem, months[0], "c-ses_chat")
        work = os.path.join(mem, months[0], "c-ses_e2e")
        assert not os.path.exists(os.path.join(chat, "transcript.md"))
        assert json.load(open(os.path.join(chat, "meta.json")))["counts"]["untranscribed"] == 1
        text = open(os.path.join(work, "transcript.md")).read()
        assert text.startswith("## turn 1 · ") and "### tool result (bash, " in text
        assert "### tool call (bash)" in text and "### runtime" in text
        assert re.search(r"^redactions: 3 \(api_key 1, bearer 1, private_key 1\)$", text, re.M), text
        blob = b"".join(open(os.path.join(dp, f), "rb").read()
                        for dp, _, fs in os.walk(os.path.join(cdir, "memory")) for f in fs)
        for s in (shapes[2], shapes[9], shapes[0]):
            for line in s["secret"].splitlines():
                assert line.encode() not in blob, s["kind"]
        rows = [r for r in _memory_rows(ledger) if r[1] == "block"]
        assert rows and rows[0][2] == "c:ses_e2e" and rows[0][3]["redactions"] == 3
        conn = sqlite3.connect(ledger)
        dump = "\n".join(str(x) for x in conn.execute("SELECT * FROM conductor_events"))
        conn.close()
        assert shapes[2]["secret"] not in dump and shapes[9]["secret"] not in dump
        # the raw path never writes a transcript
        from tests.test_conductor_e2e import TOOLS, _call
        st, _, _ = _call(port, "POST", "/v1/chat/completions",
                         {"model": "TandemLLM", "messages": msgs, "tools": TOOLS,
                          "max_tokens": 50}, headers={"x-session-id": "ses_raw", "X-Tandem": "raw"})
        assert st == 200
        time.sleep(0.5)
        assert not os.path.exists(os.path.join(mem, months[0], "c-ses_raw"))
        # nothing of the memory left the conductor's directory
        for dp, _, fs in os.walk(root):
            for f in fs:
                if f in ("transcript.md", "meta.json"):
                    assert os.path.realpath(dp).startswith(os.path.realpath(cdir)), dp
    finally:
        proc.kill()
        shutil.rmtree(root, ignore_errors=True)


def test_6_on_the_fake_engine_a_read_only_memory_directory_fails_nothing():
    from tests.test_conductor_e2e import _port, _stream
    root = tempfile.mkdtemp(prefix="qse-mem-")
    cdir = os.path.join(root, "conductor")
    os.makedirs(os.path.join(cdir, "memory"), mode=0o500)
    port = _port()
    proc, ledger = _fake(port, cdir)
    try:
        text, hdrs, usage = _stream(port, [{"role": "user", "content": "run the tool please"}],
                                    headers={"x-session-id": "ses_ro"})
        assert usage and usage["completion_tokens"] > 0 and text, (text, usage)
        deadline = time.time() + 10
        while time.time() < deadline and not _memory_rows(ledger):
            time.sleep(0.2)
        rows = _memory_rows(ledger)
        assert rows and rows[0][:2] == ("transcript", "write_failed"), rows
        assert not os.listdir(os.path.join(cdir, "memory"))
    finally:
        os.chmod(os.path.join(cdir, "memory"), 0o700)
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
