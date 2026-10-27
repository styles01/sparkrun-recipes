"""The classifier's richer digest (SRV-153, QA SRV-154): the user message plus one context line.

SRV-153's Gherkin, row by row:
  1. same budget: a 6,000-token message's digest plus the line fits the decider's `limits()` (the
     state reaches the decider whole, nothing cut after it), the line is at most 120 characters
     and Qwen's digest stays within `digest_chars`;
  2. the line's contents: after a turn classified code_edit at 0.93 whose calls were read, edit
     and bash on .py and .md files and whose checker said calccheck FAIL, the decider reads
     `prev=code_edit@0.93 tools=read,edit,bash files=.py,.md checker=calccheck:FAIL compaction=no`;
  3. the accuracy report: tools/conductor_eval.py prints accuracy and p50 per set for the decider
     alone, the cascade and the router's stickiness, under today's digest and with the line, on
     the short, the long and the follow-up sets (a fake decider here; the real figures are the
     box's, in the Memo note).
Plus the edge cases: a first message says prev=none, a session without tools says tools=none, a
checker unknown to signals says unknown, the line never carries a path or a file's text, a
compaction says compaction=yes once; shadow mode journals the digest used; the switch in the
router document turns the line off; another line is another cache entry; the fixture sets hold
what the ticket asks (60 long prompts of 1,500 to 8,000 tokens, 12 per class).

Run: python tests/test_classifier_digest.py
"""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from test_conductor_router import SYSTEM, QTok, body, turn  # noqa: E402

from server import app  # noqa: E402
from server.conductor import classify, schema, signals  # noqa: E402
from server.conductor.hook import Conductor  # noqa: E402
from tools import conductor_eval  # noqa: E402

FIX = os.path.join(HERE, "fixtures", "conductor")
LAYA = {"context_tokens": 512, "question_tokens": 192, "max_options": 20}
GOLDEN = ("prev=code_edit@0.93 tools=read,edit,bash files=.py,.md checker=calccheck:FAIL "
          "compaction=no")


class FakeDecider(classify.Decider):
    """A decider with Laya's limits that keeps every state it was given; `fn(state, options)`
    answers (label, confidence)."""

    def __init__(self, fn, limits=None):
        self.fn, self.lim, self.states = fn, dict(limits or LAYA), []

    def limits(self):
        return dict(self.lim)

    def ask(self, state, questions):
        self.states.append(state)
        q = questions["q"]
        label, conf = self.fn(state, q["criteria"])
        return {"model": "fake", "answers": {"q": {"choice": label, "probabilities": {
            label: conf}, "confidence": conf}}}


class FakeQwen:
    def __init__(self, fn):
        self.fn, self.states = fn, []

    def choice(self, state, prompt, options, max_chars=2000):
        self.states.append(state[:max_chars])
        label, conf = self.fn(state, options)
        return classify.Answer("qwen", label, conf, {label: conf}, 1.0)


def setup(decider_fn=None, qwen_fn=None, context_line=True):
    app.STATE.clear()
    app.STATE.update(tok=QTok(), device="cpu", reasoning_effort="medium", segments=None,
                     model="qwen38-spark-engine")
    dec = FakeDecider(decider_fn or (lambda s, o: ("code_edit", 0.93)))
    qwen = FakeQwen(qwen_fn or (lambda s, o: ("code_edit", 0.9)))
    cond = Conductor(tempfile.mkdtemp(prefix="digest-"), state=app.STATE, journal=False,
                     factory=lambda kind, role, timeout: dec if kind == "decider" else qwen)
    _, r = cond.store.get("router")
    if r["classifier"]["context_line"] != context_line:
        r["classifier"]["context_line"] = context_line
        cond.store.put("router", None, r, note=f"line {context_line}", author="t")
    return cond, dec, qwen


def call(i, name, args):
    return {"id": f"c{i}", "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


def session_with_tools(user2="continue"):
    """turn 1: a code request; turn 2 its tool calls and results; then a new user message."""
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": "Build minicalc as the SPEC says."}]
    work = [{"role": "assistant", "content": "", "tool_calls": [
                call(1, "read", {"filePath": "/home/k/minicalc/minicalc/parser.py"})]},
            {"role": "tool", "tool_call_id": "c1", "content": "def parse(): ..."},
            {"role": "assistant", "content": "", "tool_calls": [
                call(2, "read", {"filePath": "/home/k/minicalc/SPEC.md"}),
                call(3, "edit", {"filePath": "/home/k/minicalc/README.md", "oldString": "a",
                                 "newString": "b"}),
                call(4, "bash", {"command": "python3 check.py"})]},
            {"role": "tool", "tool_call_id": "c2", "content": "# minicalc\nthe spec"},
            {"role": "tool", "tool_call_id": "c3", "content": "ok"},
            {"role": "tool", "tool_call_id": "c4",
             "content": "calccheck minicalc\n  [vars] env\nRESULT: FAIL <- (a) vars failed 1 > 0"},
            {"role": "assistant", "content": "The vars group still fails."},
            {"role": "user", "content": user2}]
    return msgs, msgs + work


# ------------------------------------------------------------------ row 1: the same budget

def test_the_digest_and_the_line_fit_the_deciders_limits():
    big = ("The parser rejects a valid expression and we need it fixed. " * 400).strip()
    assert len(big) > 6000 * 3.5, "about 6,000 tokens"
    cond, dec, qwen = setup(decider_fn=lambda s, o: ("code_edit", 0.93 if len(s) < 500 else 0.5),
                            qwen_fn=lambda s, o: ("debug", 0.2))
    first, _ = session_with_tools()
    turn(cond, body(first))                          # the session's first class: code_edit 0.93
    dec.states.clear()
    msgs = first + [{"role": "assistant", "content": "ok"}, {"role": "user", "content": big}]
    plan, _, _ = turn(cond, body(msgs))
    state = dec.states[-1]
    room = (LAYA["context_tokens"] - LAYA["question_tokens"]) * classify.CHARS_PER_TOKEN
    assert len(state) <= room == dec.state_chars(), (len(state), room)
    line = plan.row["extra"]["context_line"]
    assert state.endswith("\n" + line) and len(line) <= signals.LINE_MAX
    assert state.startswith(big[:200]), "the head is the message's"
    assert "[...]" in state and big[-40:] in state, "the tail survives too"
    # Qwen's digest is its own, at digest_chars, with the same line
    assert len(qwen.states[-1]) <= 2000 and qwen.states[-1].endswith("\n" + line)
    # the digest itself, at every budget the schema allows: never over, the line whole
    for chars in (200, 500, 1120, 2000, 20000):
        for text in ("short", big):
            d = signals.digest(text, chars, GOLDEN)
            assert len(d) <= chars and d.endswith("\n" + GOLDEN), (chars, len(d))


def test_a_long_tool_list_still_fits_120_characters():
    line = signals.context_line(
        "three_d", 0.91, ["chrome-devtools_evaluate_script", "chrome-devtools_navigate_page",
                          "chrome-devtools_take_screenshot"], [".js", ".html", ".css", ".png"],
        ("scenecheck", "FAIL"), True)
    assert len(line) <= signals.LINE_MAX, len(line)
    assert line.startswith("prev=three_d@0.91 tools=") and line.endswith("compaction=yes")
    assert "chrome-devtools" not in line, "an MCP tool goes without its server's name"
    worst = signals.context_line("a" * 40, 0.5, ["x" * 60] * 3, [".abcdefgh"] * 9,
                                 ("y" * 60, "FAIL"), False)
    assert len(worst) <= signals.LINE_MAX


# ------------------------------------------------------------------ row 2: the line's contents

def test_the_line_reads_the_session():
    cond, dec, _ = setup()
    first, msgs = session_with_tools()
    turn(cond, body(first))
    plan, _, _ = turn(cond, body(msgs))
    assert dec.states[-1] == "continue\n" + GOLDEN, dec.states[-1]
    assert plan.row["extra"]["context_line"] == GOLDEN
    assert signals.context_line("code_edit", 0.93, ["read", "edit", "bash"], [".py", ".md"],
                                ("calccheck", "FAIL"), False) == GOLDEN


def test_the_edge_cases_of_the_line():
    cond, dec, _ = setup()
    # a first message: nothing known yet
    plan, _, _ = turn(cond, body([{"role": "user", "content": "Build minicalc."}]),
                      headers={"x-session-id": "first"})
    assert plan.row["extra"]["context_line"] == ("prev=none tools=none files=none checker=none "
                                                 "compaction=no")
    # a session with a class but no tools yet
    msgs = [{"role": "user", "content": "Build minicalc."}, {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "go on"}]
    plan, _, _ = turn(cond, body(msgs), headers={"x-session-id": "first"})
    assert plan.row["extra"]["context_line"].startswith("prev=code_edit@0.93 tools=none files=none")
    # a checker signals does not know: its name is unknown
    _, msgs = session_with_tools()
    msgs[7]["content"] = "mycheck v2\nRESULT: PASS"
    plan, _, _ = turn(cond, body(msgs), headers={"x-session-id": "unk"})
    assert "checker=unknown:PASS" in plan.row["extra"]["context_line"]
    # never a path, a file name or a file's text
    line = plan.row["extra"]["context_line"]
    for leak in ("/", "minicalc", "SPEC", "parser", "README", "def parse", "check.py"):
        assert leak not in line, (leak, line)
    assert signals.ext_of("/a/b/Makefile") is None and signals.ext_of("x.TAR.GZ") == ".gz"
    assert signals.ext_of("weird.name with space") is None


def test_a_compaction_says_so_once():
    cond, dec, _ = setup()
    first, msgs = session_with_tools()
    h = {"x-session-id": "comp"}
    turn(cond, body(first), headers=h)
    summary = [{"role": "user", "content": "Summarize the conversation so far. " * 80}]
    plan, _, _ = turn(cond, {"model": "TandemLLM", "messages": summary}, headers=h)
    assert plan.row["source"] == "structural" and "context_line" not in plan.row["extra"]
    after = [{"role": "system", "content": SYSTEM},
             {"role": "user", "content": "Here is the summary. Continue with the next stage."}]
    plan, _, _ = turn(cond, body(after), headers=h)
    assert plan.row["extra"]["context_line"].endswith("compaction=yes")
    after += [{"role": "assistant", "content": "ok"}, {"role": "user", "content": "go on"}]
    plan, _, _ = turn(cond, body(after), headers=h)
    assert plan.row["extra"]["context_line"].endswith("compaction=no"), "once"


# ------------------------------------------------------------------ the switch, cache, shadow

def test_the_switch_turns_the_line_off():
    assert schema.ROUTER["classifier"]["context_line"] is False, "off until it passes the ship rule"
    cond, dec, _ = setup(context_line=False)
    first, msgs = session_with_tools()
    turn(cond, body(first))
    plan, _, _ = turn(cond, body(msgs))
    assert dec.states[-1] == "continue" and "context_line" not in plan.row["extra"]


def test_another_line_is_another_cache_entry():
    cond, dec, _ = setup()
    turn(cond, body([{"role": "user", "content": "continue"}]), headers={"x-session-id": "a"})
    turn(cond, body([{"role": "user", "content": "continue"}]), headers={"x-session-id": "b"})
    assert len(dec.states) == 1, "the same text and the same line: the cache answers"
    _, msgs = session_with_tools()
    turn(cond, body(msgs), headers={"x-session-id": "c"})
    assert len(dec.states) == 2 and dec.states[-1].endswith("checker=calccheck:FAIL compaction=no")


def test_shadow_journals_the_digest_used():
    cond, dec, _ = setup()
    _, doc = cond.store.get("profiles", "coding")
    doc["decisions"]["classify"] = "shadow"
    cond.store.put("profiles", "coding", doc, note="classify shadow", author="t")
    first, msgs = session_with_tools()
    turn(cond, body(first))
    plan, _, _ = turn(cond, body(msgs))
    assert plan.row["shadow"]["class_id"] == "code_edit"
    assert plan.row["digest"] == "continue\n" + GOLDEN, plan.row["digest"]
    assert plan.row["extra"]["context_line"] == GOLDEN


# ------------------------------------------------------------------ row 3: the accuracy report

def _sets():
    load = lambda n: [json.loads(x) for x in open(os.path.join(FIX, n)) if x.strip()]
    return {"short": load("task_prompts.jsonl"), "long": load("long_prompts.jsonl"),
            "followups": load("followups.jsonl")}


def test_the_fixture_sets():
    s = _sets()
    long = s["long"]
    assert len(long) == 60 and len(s["short"]) == 216
    by = {}
    for r in long:
        by.setdefault(r["label"], []).append(r)
        assert 1500 <= r["tokens"] <= 8000, r["tokens"]
    assert sorted(by) == ["code_edit", "debug", "design_ui", "plan", "summarize"]
    assert all(len(v) == 12 for v in by.values())
    assert {r["place"] for r in long} == {"head", "tail", "middle"}
    labels = {c["id"] for c in schema.ROUTER["classes"]}
    for r in s["followups"]:
        assert r["label"] in labels and r["context"]["prev"] in labels
    assert sum(r["kind"] == "switch" for r in s["followups"]) >= 10


def _reader(state, options):
    """A fake decider that only knows the session: sure of `prev` when the line names it, else
    unsure; a long message's request is found only when it is in the state it was given."""
    for c in options:
        if f"prev={c}@" in state:
            return c, 0.9
    return "chat", 0.3


def test_the_accuracy_report_with_and_without_the_line():
    sets = _sets()
    small = {"followups": sets["followups"], "long": sets["long"][:6]}
    router = schema.seed("router")
    dec = FakeDecider(_reader)
    got = conductor_eval.report(small, ["today", "line"], router, dec, None)
    fu_today = got["variants"]["today"]["followups"]["stages"]
    fu_line = got["variants"]["line"]["followups"]["stages"]
    assert set(fu_line) == {"decider", "cascade", "router"}
    stay = sum(r["kind"] == "stay" for r in sets["followups"])
    assert fu_today["decider"]["correct"] == 0 and fu_line["decider"]["correct"] == stay
    for key in ("accuracy", "answered", "accuracy_answered", "top1", "p50_ms", "p95_ms",
                "confusion"):
        assert key in fu_line["decider"]
    # the router keeps the previous class when nobody is sure: today's follow-ups stay right
    assert fu_today["router"]["correct"] == stay
    rows = got["rows"]["line:followups"]
    assert rows[0]["line"].startswith("prev=") and len(rows) == len(sets["followups"])
    # the long set: today the decider read the front of a 2,000-character digest, so its tail
    # (and the cut mark) never reached it; with the line it reads head, tail and the line
    for r in got["rows"]["today:long"] + got["rows"]["line:long"]:
        assert r["chars"] > 2000
    today_long = dec.states[len(sets["followups"]):len(sets["followups"]) + 6]
    line_long = dec.states[-6:]
    assert all(len(s) <= dec.state_chars() and "[...]" not in s for s in today_long)
    assert all(len(s) <= dec.state_chars() and "[...]" in s and s.endswith("compaction=no")
               for s in line_long)


def test_the_report_file_keeps_the_compare_screen_shape():
    tmp = tempfile.mkdtemp(prefix="clf-")
    fu = os.path.join(tmp, "fu.jsonl")
    with open(fu, "w") as f:
        for r in _sets()["followups"][:12]:
            f.write(json.dumps(r) + "\n")
    out = os.path.join(tmp, "rep.json")
    classify.register_backend("self", lambda model, timeout_ms: FakeDecider(_reader))
    try:
        with redirect_stdout(io.StringIO()) as buf:
            assert conductor_eval.main(["--set", fu, "--decider", "self", "--digest",
                                        "today,line", "--out", out]) == 0
    finally:
        classify.BACKENDS.pop("self", None)
    rep = json.load(open(out))
    assert rep["kind"] == "classifier" and rep["digest"] == "today"
    assert rep["set"]["n"] == 12 and set(rep["stages"]) == {"decider", "cascade", "router"}
    assert set(rep["variants"]) == {"today", "line"}
    assert rep["config"]["decider_chars"] == 1120
    printed = buf.getvalue()
    assert "today" in printed and "line" in printed and "router" in printed
    # `router` is the router's own digest: the line when the document has it on
    with redirect_stdout(io.StringIO()):
        classify.register_backend("self", lambda model, timeout_ms: FakeDecider(_reader))
        try:
            conductor_eval.main(["--set", fu, "--decider", "self", "--out", out])
        finally:
            classify.BACKENDS.pop("self", None)
    assert json.load(open(out))["digest"] == "sized", "the seed has the line off"
    cdir = tempfile.mkdtemp(prefix="clf-cfg-")
    cond, _, _ = setup()                                   # a store with the line on
    import shutil
    shutil.copytree(cond.store.root, cdir, dirs_exist_ok=True)
    with redirect_stdout(io.StringIO()):
        classify.register_backend("self", lambda model, timeout_ms: FakeDecider(_reader))
        try:
            conductor_eval.main(["--set", fu, "--decider", "self", "--config", cdir, "--out", out])
        finally:
            classify.BACKENDS.pop("self", None)
    assert json.load(open(out))["digest"] == "line"


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:64s} ok")
            passed += 1
    print(f"{passed} passed")
