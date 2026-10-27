"""The turn snapshot (J1 step 1, the plan of 2026-10-02, section 1.1): at the start of every
request whose newest message is a tool result, the router keeps what the harness saw of the turn
before it, and the transcript writer stores it beside the block.

The test matrix (the plan's step 1: "pure observation; flag-off identity; under 2 ms a request";
"a file per tool turn, sizes within limits, host time; nothing else changes"):

  1  flag off (no `outcomes` section, or every switch off): the plans, the rows and the served
     prompt ids of a whole session are the same as without the section; no snapshot, no row
     field                                              test_flag_off_is_identical
  2  snapshots on alone changes no prompt id, no effort, no note: pure observation
                                                        test_snapshots_on_change_nothing_served
  3  the fields of 1.1: calls (8, results clipped at 1,200), files (path, bytes, hash), the last
     write and run, the screenshot's hash and size, the console's counts and error lines, the
     checker lines, exit codes, a test runner's counts, the state dump and its diff, a log tail
                                                        test_the_fields
  4  the limits: 8 calls, 20 files, 4 KB of state, 32 KB a file (tails clipped first)
                                                        test_the_limits
  5  the writer: snapshots/turn-<n>.json beside the block n, 0600, the images under shots/
     (the newest 5 kept), one line in the block's runtime section, meta.json's count; a block
     without a snapshot is byte-identical to before (the golden files of test_transcripts)
                                                        test_the_writer
  6  host time: observe() on the two Mario sessions of 2026-10-02 (with a screenshot attached
     after every take_screenshot) under 2 ms a request (median), each request's own time
     journaled                                           test_host_time
  7  a session first seen mid-way counts only what came after its last user message; a call
     is taken once                                       test_first_sight_and_once

Run: python tests/test_snapshots.py
"""

from __future__ import annotations

import base64
import json
import os
import stat
import statistics
import struct
import sys
import tempfile
import types
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from test_conductor_router import QTok, by_label, setup  # noqa: E402

from server import app  # noqa: E402
from server.conductor import outcomes, schema, signals, transcripts  # noqa: E402

SYSTEM = "You are opencode.\n<env>\n  Working directory: /work/game\n</env>"
TASK = "Build a small canvas game in game.html and test it in the browser."
NAMES = ("read", "write", "edit", "bash", "chrome-devtools_new_page",
         "chrome-devtools_take_screenshot", "chrome-devtools_list_console_messages",
         "chrome-devtools_evaluate_script", "chrome-devtools_press_key")
TOOLS = [{"type": "function", "function": {
    "name": n, "description": f"the {n} tool",
    "parameters": {"type": "object",
                   "properties": {"filePath": {"type": "string"}, "content": {"type": "string"},
                                  "command": {"type": "string"}, "url": {"type": "string"},
                                  "function": {"type": "string"}, "key": {"type": "string"}},
                   "required": {"read": ["filePath"], "write": ["filePath", "content"],
                                "edit": ["filePath"], "bash": ["command"]}.get(n, [])}}}
    for n in NAMES]


def png(w: int, h: int, salt: bytes = b"") -> str:
    """A small real PNG of w x h (one grey row repeated), base64; `salt` changes its bytes."""
    raw = b"".join(b"\x00" + bytes([len(salt) * 37 % 256]) * w for _ in range(h))
    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d))
    data = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"tEXt", b"c\x00" + salt)
            + chunk(b"IEND", b""))
    return base64.b64encode(data).decode()


def media(b64: str) -> dict:
    return {"role": "user", "content": [
        {"type": "text", "text": signals.TOOL_MEDIA_CAPTION},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}}]}


class S:
    """A session the tests drive: calls with their results, in order."""

    def __init__(self):
        self.msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": TASK}]
        self.n = 0

    def step(self, *calls, shot: str | None = None) -> list:
        """One assistant step: `calls` are (name, args, result); `shot`: an image after them."""
        tc, res = [], []
        for name, args, text in calls:
            self.n += 1
            cid = f"call_{self.n}"
            tc.append({"id": cid, "type": "function",
                       "function": {"name": name, "arguments": json.dumps(args)}})
            res.append({"role": "tool", "tool_call_id": cid, "content": text})
        self.msgs += [{"role": "assistant", "content": "", "tool_calls": tc}] + res
        if shot is not None:
            self.msgs.append(media(shot))
        return list(self.msgs)


def body(msgs) -> dict:
    return {"model": "TandemLLM", "messages": list(msgs), "tools": TOOLS}


def fresh(switches: dict | None, profile="coding"):
    cond = setup(decider=by_label("code_edit", 0.91))[0]
    if switches is not None:
        _, doc = cond.store.get("profiles", profile)
        doc["outcomes"] = dict(switches)
        cond.store.put("profiles", profile, doc, note=f"outcomes {switches}", author="t")
    return cond


def plan(cond, msgs, sid="ses-snap", rid="r"):
    return cond.plan(body(msgs), {"x-session-id": sid}, client_kind="opencode", request_id=rid,
                     chat=True)


def served(cond, msgs, sid="ses-snap"):
    """plan, render, apply: (plan, the served prompt ids)."""
    b = body(msgs)
    p = cond.plan(b, {"x-session-id": sid}, client_kind="opencode", request_id="r", chat=True)
    prompt, _, in_think = app.build_prompt(b)
    out, _ = p.apply_prompt(prompt, app.STATE["tok"], in_think)
    return p, out.tolist()


def text_session() -> list[list]:
    """A text-only session: every request's messages, for the identity tests."""
    s = S()
    out = [list(s.msgs)]
    out.append(s.step(("bash", {"command": "ls -la /work/game"}, "total 0")))
    out.append(s.step(("write", {"filePath": "/work/game/game.html", "content": "<canvas>"},
                       "Wrote file successfully.")))
    out.append(s.step(("bash", {"command": "npm test"},
                       "Tests: 1 failed, 4 passed, 5 total\nexit code 1")))
    out.append(s.step(("edit", {"filePath": "/work/game/game.js", "oldString": "a",
                                "newString": "b"}, "Edit applied successfully.")))
    out.append(s.step(("bash", {"command": "npm test"},
                       "Tests: 5 passed, 5 total\nexit code 0")))
    return out


def _plan_view(p) -> dict:
    r = p.row
    return {k: r.get(k) for k in ("class_id", "effort", "budget", "budget_source", "thinking",
                                  "decisions", "shadow", "injected", "acted", "changed")} | {
        "extra": {k: v for k, v in r["extra"].items()}, "budget": p.budget,
        "thinking_off": p.thinking_off, "extras": p.extras}


# ------------------------------------------------------------------ 1

def test_flag_off_is_identical():
    turns = text_session()
    base = fresh(None)
    off = fresh({k: "off" for k in schema.PROFILE_OUTCOMES})
    for i, msgs in enumerate(turns):
        pa, ia = served(base, msgs)
        pb, ib = served(off, msgs)
        assert ia == ib, f"turn {i}: the same prompt ids"
        va, vb = _plan_view(pa), _plan_view(pb)
        assert va == vb, (i, va, vb)
        assert "outcome" not in pa.row["extra"] and pa.snapshot is None and pa.outcome is None
        assert pa.extras == [] and pa.acts == []
    assert base.router._sessions["c:ses-snap"].evidence is None, "no evidence kept with it off"
    assert schema.seed("profiles", "coding").get("outcomes") is None, "the seeds stay off"
    assert schema.check("router", schema.seed("router")) == []
    assert "outcomes" not in schema.seed("router")


# ------------------------------------------------------------------ 2

def test_snapshots_on_change_nothing_served():
    turns = text_session()
    base, on = fresh(None), fresh({"snapshots": "on"})
    for i, msgs in enumerate(turns):
        pa, ia = served(base, msgs)
        pb, ib = served(on, msgs)
        assert ia == ib, f"turn {i}: the same prompt ids"
        assert _plan_view(pa) == _plan_view(pb), i
        assert (pb.snapshot is not None) == (i > 0), f"turn {i}: a snapshot on a tool turn only"
    assert "outcome" not in pb.row["extra"], "mode off: the route row is unchanged"


# ------------------------------------------------------------------ 3

def test_the_fields():
    cond = fresh({"snapshots": "on"})
    s = S()
    plan(cond, s.msgs)
    shot = png(1280, 720)
    s.step(("write", {"filePath": "/work/game/game.html", "content": "<canvas id=c></canvas>"},
            "Wrote file successfully."))
    p = plan(cond, s.msgs)
    snap = p.snapshot
    assert snap["files"] == [{"path": "/work/game/game.html", "bytes": 22,
                              "hash": outcomes._h("<canvas id=c></canvas>")}], snap["files"]
    assert snap["last_write"] == 0 and snap["last_run"] is None
    msgs = s.step(("chrome-devtools_new_page", {"url": "file:///work/game/game.html"},
                   "## Pages\n1: game"),
                  ("chrome-devtools_take_screenshot", {}, "Took a screenshot."), shot=shot)
    snap = plan(cond, msgs).snapshot
    assert snap["shots"][0]["sha"] == outcomes._h(shot) and (snap["shots"][0]["w"],
                                                               snap["shots"][0]["h"]) == (1280, 720)
    assert snap["shots"][0]["call"] == "chrome-devtools_take_screenshot"
    assert snap["last_run"] == 1 and [c["name"] for c in snap["calls"]] == [
        "chrome-devtools_new_page", "chrome-devtools_take_screenshot"]
    assert snap["_media"][0][0] == snap["shots"][0]["sha"], "the bytes go to the writer"
    msgs = s.step(("chrome-devtools_list_console_messages", {},
                   "## Console messages\nmsgid=1 [error] Uncaught ReferenceError: player is not "
                   "defined (0 args)\nmsgid=2 [log] ready (1 args)\nmsgid=3 [warn] slow (0 args)"))
    snap = plan(cond, msgs).snapshot
    assert snap["console"]["counts"] == {"error": 1, "log": 1, "warn": 1}, snap["console"]
    assert snap["console"]["errors"][0].startswith("msgid=1 [error] Uncaught ReferenceError")
    state1 = {"tandem_state": 1, "player": {"x": 40, "y": 300}, "lives": 3, "level": 1}
    state2 = {"tandem_state": 1, "player": {"x": 96, "y": 300}, "lives": 3}
    msgs = s.step(("chrome-devtools_press_key", {"key": "ArrowRight"}, "pressed"),
                  ("chrome-devtools_evaluate_script", {"function": "() => window.__state()"},
                   "Script ran on page and returned:\n```json\n" + json.dumps(state1) + "\n```"))
    snap = plan(cond, msgs).snapshot
    assert snap["state"] == state1 and snap["state_diff"] == {"changed": {}, "gone": []}
    msgs = s.step(("chrome-devtools_evaluate_script", {"function": "() => window.__state()"},
                   "returned:\n" + json.dumps(state2)))
    snap = plan(cond, msgs).snapshot
    assert snap["state_diff"] == {"changed": {"player.x": [40, 96]}, "gone": ["level"]}, \
        snap["state_diff"]
    msgs = s.step(("bash", {"command": "cd /work/game && npm test 2>&1 | tail -5"},
                   "Tests: 2 failed, 9 passed, 11 total\nexit code 1"),
                  ("bash", {"command": "tail -n 60 ./.tandem/run.log"},
                   "\n".join(f"line {i}" for i in range(100))),
                  ("bash", {"command": "python check.py"},
                   "calccheck src/calc.py\nRESULT: FAIL <- (a) error 0.4 > 0.1"))
    snap = plan(cond, msgs).snapshot
    assert snap["exits"] == [{"call": "bash", "code": 1}]
    assert snap["tests"] == [{"failed": 2, "passed": 9, "call": "bash"}], snap["tests"]
    assert snap["logs"][0]["tail"].splitlines() == [f"line {i}" for i in range(40, 100)]
    assert snap["checkers"][0]["verdict"] == "FAIL" and \
        snap["checkers"][0]["readings"][0]["value"] == 0.4
    assert snap["calls"][0]["args"]["command"].startswith("cd /work/game")
    json.dumps({k: v for k, v in snap.items() if not k.startswith("_")})   # plain JSON


# ------------------------------------------------------------------ 4

def test_the_limits():
    cond = fresh({"snapshots": "on"})
    s = S()
    plan(cond, s.msgs)
    big = "x" * 50_000
    s.step(*[("write", {"filePath": f"/work/game/f{i}.js", "content": big[:i + 1]},
              "Wrote file successfully.") for i in range(25)])
    msgs = s.step(*[("bash", {"command": f"echo {i}", "pad": big}, f"out {i}\n" + big)
                    for i in range(12)])
    snap = plan(cond, msgs).snapshot
    assert len(snap["calls"]) == 8 and snap["calls"][-1]["args"]["command"] == "echo 11"
    assert all(len(c["result"]) <= 1200 for c in snap["calls"])
    assert all(len(c["args"]["pad"]) <= 240 for c in snap["calls"]), "long strings clipped"
    cond2 = fresh({"snapshots": "on"})
    s2 = S()
    plan(cond2, s2.msgs, sid="ses-lim")
    msgs = s2.step(*[("write", {"filePath": f"/work/game/f{i}.js", "content": "a"},
                      "Wrote file successfully.") for i in range(25)])
    snap = plan(cond2, msgs, sid="ses-lim").snapshot
    assert len(snap["files"]) == 20
    st = {"tandem_state": 1, "rows": ["y" * 100] * 80}
    msgs = s2.step(("bash", {"command": "node dump.js"}, json.dumps(st)))
    snap = plan(cond2, msgs, sid="ses-lim").snapshot
    assert snap["state"] == {"tandem_state": 1, "clipped": True}, "over 4 KB: the marker only"
    fat = outcomes.Snapshot(snap)
    fat["calls"] = [{"name": "bash", "args": {}, "chars": 9, "result": "r" * 1200}] * 8
    fat["logs"] = [{"call": "bash", "tail": "l" * 30_000}]
    data = outcomes.to_json(fat, 32768)
    assert len(data) <= 32768 and json.loads(data)["logs"][0]["tail"] == "l" * 400


# ------------------------------------------------------------------ 5

def _rec(rid="chatcmpl-1"):
    return types.SimpleNamespace(ts=1790000000.0, request_id=rid, client_kind="opencode",
                                 out=([ord(c) for c in "ok"], False), out_calls=[],
                                 finish_reason="tool_calls", prompt_tokens=1000,
                                 completion_tokens=2, reasoning_tokens=40, sup=None)


def test_the_writer():
    root = os.path.join(tempfile.mkdtemp(prefix="mem-"), "memory")
    doc = {"memory": {"transcripts": dict(transcripts.DEFAULTS)}}
    w = transcripts.Transcripts(root, settings=lambda: doc, journal=None,
                                tok=lambda: types.SimpleNamespace(
                                    decode=lambda ids, skip_special_tokens=False:
                                    "".join(chr(i) for i in ids)))
    cond = fresh({"snapshots": "on"})
    s = S()
    p = plan(cond, s.msgs)
    w.write(w.job(p, _rec("chatcmpl-0")))                    # the first turn: no snapshot
    d = w.find("c:ses-snap")
    assert not os.path.exists(os.path.join(d, "snapshots"))
    shots = [png(64, 32, bytes([i])) for i in range(7)]
    for i in range(7):
        msgs = s.step(("chrome-devtools_take_screenshot", {}, "Took a screenshot."),
                      shot=shots[i])
        p = plan(cond, msgs)
        w.write(w.job(p, _rec(f"chatcmpl-{i + 1}")))
    names = sorted(os.listdir(os.path.join(d, "snapshots")))
    assert names == sorted(f"turn-{n}.json" for n in range(2, 9)), names
    for n in names:
        path = os.path.join(d, "snapshots", n)
        assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
        assert os.path.getsize(path) <= 32768
    kept = os.listdir(os.path.join(d, "shots"))
    assert len(kept) == 5, kept
    assert all(stat.S_IMODE(os.stat(os.path.join(d, "shots", k)).st_mode) == 0o600 for k in kept)
    last = json.load(open(os.path.join(d, "snapshots", "turn-8.json")))
    assert last["shots"][0]["file"].startswith("shots/turn-8-0-") and "_media" not in last
    assert last["turn"]["finish"] == "tool_calls" and last["turn"]["reasoning_tokens"] == 40
    with open(os.path.join(d, "shots", last["shots"][0]["file"][6:]), "rb") as f:
        assert base64.b64encode(f.read()).decode() == shots[6]
    text = open(os.path.join(d, "transcript.md")).read()
    assert text.count("snapshot: snapshots/turn-") == 7 and "snapshot: snapshots/turn-8.json" \
        in text and "· outcome" not in text, "mode off: the line names the file only"
    meta = transcripts.read_meta(d)
    assert meta["counts"]["snapshots"] == 7 and meta["counts"]["blocks"] == 8
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tools"))
    import contract_check
    root = os.path.join(os.path.dirname(HERE), "docs", "contract", "conductor-v1")
    for name in ("memory-sessions", "memory-session"):
        sch = contract_check.load(name, root)
        assert not contract_check.validate(meta["counts"], sch["$defs"]["counts"], root=root), name
    first = transcripts.read_blocks(d, meta["bytes"], meta["offsets"])[0]["text"]
    assert "snapshot:" not in first
    # the secret-pattern refusal: a key in a result never reaches the snapshot file
    key = "sk-" + "Q7v" * 10
    pem = "-----BEGIN PRIVATE KEY-----\nMIIEv" + "Q" * 60 + "\n-----END PRIVATE KEY-----"
    msgs = s.step(("bash", {"command": f"echo {key}"}, f"{key}\n{pem}\nAuthorization: Bearer "
                                                        f"{'z' * 30}"))
    p = plan(cond, msgs)
    w.write(w.job(p, _rec("chatcmpl-9")))
    raw = open(os.path.join(d, "snapshots", "turn-9.json")).read()
    assert key not in raw and "MIIEv" not in raw and "z" * 30 not in raw, raw[:600]
    assert "[redacted:" in raw and json.loads(raw)["calls"][0]["name"] == "bash"


# ------------------------------------------------------------------ 6

def mario_sessions():
    fx = json.load(open(os.path.join(HERE, "fixtures", "claim_check_mario.json")))
    out = []
    for ses in fx["sessions"]:
        msgs, k = [], 0
        for m in ses["messages"]:
            if m["role"] == "user" and m.get("content") == signals.TOOL_MEDIA_CAPTION:
                k += 1
                m = media(png(1280, 720, bytes([k % 3])))
            msgs.append(m)
        out.append((ses, msgs))
    return out


def test_host_time():
    times = []
    for ses, msgs in mario_sessions():
        tools = [{"type": "function", "function": {"name": n, "parameters": {}}}
                 for n in ses["tools"]]
        sh_ok = 0
        for _ in range(5):
            ev = outcomes.Evidence(0.0)
            cfg = schema.outcomes({})
            for i in range(3, len(msgs) + 1):
                if msgs[i - 1]["role"] not in ("tool", "user"):
                    continue
                b = {"messages": msgs[:i], "tools": tools}
                sh = signals.shape(b)
                if not sh.tool_turn:
                    continue
                snap = outcomes.observe(b, sh, ev, cfg, now=float(i), keep_media=True)
                times.append(snap["ms"])
                sh_ok += 1
        assert sh_ok > 40, sh_ok
    med = statistics.median(times)
    print(f"    observe(): median {med:.3f} ms, p90 {sorted(times)[int(len(times) * 0.9)]:.3f} ms "
          f"over {len(times)} requests")
    assert med < 2.0, med


# ------------------------------------------------------------------ 7

def test_first_sight_and_once():
    cond = fresh({"snapshots": "on"})
    s = S()
    s.step(("bash", {"command": "echo old"}, "old"))
    s.msgs += [{"role": "assistant", "content": "Done."},
               {"role": "user", "content": "Now add a score."}]
    msgs = s.step(("bash", {"command": "echo new"}, "new"))
    snap = plan(cond, msgs).snapshot
    assert [c["args"]["command"] for c in snap["calls"]] == ["echo new"], snap["calls"]
    snap = plan(cond, msgs, rid="retry").snapshot
    assert snap["calls"] == [] and snap["new_calls"] == 0, "a retried request takes nothing twice"


if __name__ == "__main__":
    app.STATE.setdefault("tok", QTok())
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:68s} ok")
            passed += 1
    print(f"{passed} passed")
