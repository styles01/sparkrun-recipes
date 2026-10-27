"""Narrator v3, the voice: live lines of what Qwen is doing right now (`ui.narrator: live`).

The owner, 2026-10-04: "I need sentences, long and short, based on what it is doing right now."

The test matrix:

  the facts in words (server/conductor/voice.py):
    1  the user's language from the last user message      test_the_language
    2  the harness's lines in plain words, English and German; packages merged into one line
                                                           test_the_harness_in_words
    3  a tool call read as Qwen writes it: the name, the target once its parameter closes, a
       body's lines, a body before the path (opencode's write), two functions in one block;
       the action in words from the structure alone        test_the_tool_watch
    4  the clean: a phrase or why not -- markup, first person, a copied example, a file name or a
       number the text does not hold, a phrase whose words are not the text's, the word count;
       the gerund of a thinking line                       test_the_clean
  one request's voice (the fake model, a fake clock):
    5  thinking: the first line at `first_s`, then every `every_s`, the timer the engine's;
       nothing for a short thinking; the thought's line at its end; a phrase like the last one
       is not said                                         test_thinking_lines
    6  a write: its line after `tool_after_s` with the phrase; plain when the model fails; no line
       for a call done at once; a long write's progress; the file named by the body when the
       path comes last                                     test_tool_lines
    7  the share: the voice never holds Qwen past its share of the time since the first token, and
       the shadow's in-turn reading gets only what is left (narrator.Share)
                                                           test_the_share
    8  a model that fails: the template's thinking line after `fallback_s`, never silence
                                                           test_never_silent
    9  a plan in the answer gets a line, once              test_a_plan
  the handler (the fake engine in process, a conductor; test_ui_states' serve):
   10  opencode's stream (inline): the live lines as reasoning titles, the router's line in words;
       the answer, the calls and the usage are the stream's without the voice; the journal row
       keeps every line with `shown` and the totals in `extra.voice`
                                                           test_the_stream_*
   11  the lines never reach Qwen: turn 2 with turn 1's reasoning sent back (as opencode joins its
       parts, and trimmed) reaches the engine as the prompt without them
                                                           test_the_lines_never_reach_qwen
   12  a German user gets German lines                     test_a_german_user
   13  the voice off (`thoughts`, `off`, no slot) is byte for byte the stream it was
                                                           test_off_is_unchanged
   14  opencode's own parser (node and the opencode binary, when this machine has them): a line
       said while a write streams is a reasoning part after the tool part, titled, and the
       message opencode sends back strips to the one without the lines
                                                           test_opencodes_parser
   15  the contract: ui.narrator `live`, ui.live, the kinds   test_the_contract

Run: python tests/test_voice.py
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from engine.generative.fake import FakeGenerative, HEAD, TAIL  # noqa: E402
from engine.generative.slot import Slot  # noqa: E402
from server.conductor import narrator as narr_mod, schema, states, voice  # noqa: E402

M = states.MARKER
SENT = narr_mod.SENTINEL


# ------------------------------------------------------------------ helpers
class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def body_of(ids):
    text = "".join(chr(i) for i in ids if 0 <= i < 0x110000)
    return text.split(HEAD, 1)[-1].split(TAIL, 1)[0]


def first_words(n=6):
    """A fake phrase: the last `n` words of the text the model read (grounded by construction)."""
    def f(ids):
        words = re.findall(r"[A-Za-z]+", body_of(ids))
        return " ".join(words[-n:]) if words else "nothing"
    return f


def make(summary=None, *, lang="en", opened=False, settings=None, fail=False, delay_s=0.0,
         clock=None):
    m = FakeGenerative(summary=summary or first_words())
    m.fail = fail
    m.delay_s = delay_s
    clock = clock or Clock()
    m.clock = clock
    m.sleep = lambda s: setattr(clock, "t", clock.t + s)
    slot = Slot(m, role="narrator", gate=None, log=lambda s: None, clock=clock)
    slot.load(wait=True)
    n = narr_mod.Narrator(slot)
    ui = states.States({}, voice=lang)
    v = n.for_voice(ui, lang=lang, opened=opened, close_id=1, settings=settings or {})
    v.clock = clock
    v.share.clock = clock
    return v, ui, m, clock


def said(ui):
    return [e.get("summary") or e.get("shown") or e["text"] for e in ui._out()]


def think(v, clock, ids, text, *, per_token_s=0.05):
    for ch in text:
        clock.t += per_token_s
        ids.append(ord(ch))
        v.token(ids, True)


# ------------------------------------------------------------------ 1-4 the facts in words
def test_the_language():
    de = [{"role": "user", "content": "Erstelle bitte ein Mario-Spiel im Browser und teste es"}]
    en = [{"role": "user", "content": "create mario game on browser, use modern figures"}]
    assert voice.lang_of(de) == "de" and voice.lang_of(en) == "en"
    parts = [{"role": "user", "content": [{"type": "text", "text": "Warum ist das Spiel so "
                                           "langsam? Kannst du es schneller machen"}]}]
    assert voice.lang_of(parts) == "de"
    assert voice.lang_of([]) == "en" and voice.lang_of(None) == "en"
    # the last user message counts
    assert voice.lang_of(de + [{"role": "assistant", "content": "ok"}] + en) == "en"


def test_the_harness_in_words():
    ui = states.States({}, voice="en", labels={"code_edit": "code edit"})
    ui.route({"stages": ["classify"], "class_id": "code_edit", "effort": "xhigh"},
             [{"parts": [{"kind": "package", "name": n} for n in
                         ("coding", "testing", "laya-testing", "laya-playtest", "laya-scripts")]}],
             {}, False)
    ui.add("queue", reason="another request is running", n=2)
    ui.add("prefill", n="20,745", rate="660", left="18 s")
    out = ui.take_kinds()
    texts = [line.strip("\n")[len(M):].rstrip(SENT) for _, line in out]
    assert texts == ["code edit, maximum effort",
                     "packages coding, testing, laya-testing, laya-playtest and 1 more",
                     "waiting for another request to finish, 2nd in line",
                     "reading 20,745 tokens, about 18 s left"], texts
    # every one is framed with the sentinel, so the strip takes it out of any text
    for _, line in out:
        assert line.endswith(SENT + "\n\n") and states.strip_text("a" + line + "b") == "ab"
    # the log keeps the templates, the merged package marked, the shown text beside them
    assert ui.log[0]["text"] == "deciding: code_edit, xhigh effort" and ui.log[0]["shown"]
    assert sum(1 for e in ui.log if e.get("merged")) == 4
    de = states.States({}, voice="de")
    de.add("queue", reason="another request is running", n=1)
    de.add("prefill", n="15,134", rate="1,000", left="2 min")
    assert [ln.strip("\n")[len(M):].rstrip(SENT) for ln in de.take()] == [
        "wartet auf eine andere Anfrage, Platz 1", "liest 15.134 Tokens, noch etwa 2 Min"]
    # without the voice: the templates, byte for byte as before
    plain = states.States({})
    plain.add("queue", reason="another request is running", n=1)
    assert plain.take() == [states.frame("waiting: another request is running, position 1")]
    # the newest line for the live view
    assert de.last_line()["text"] == "liest 15.134 Tokens, noch etwa 2 Min"


def test_the_tool_watch():
    w = voice.ToolWatch()
    block = ("\n<function=bash>\n<parameter=command>\nnpm test -- --run\n</parameter>\n"
             "<parameter=description>\nrun the tests\n</parameter>\n</function>\n"
             "<function=write>\n<parameter=content>\n// game.js: the loop\nconst a = 1;\n"
             "const b = 2;\n</parameter>\n<parameter=filePath>\n/Users/k/x/mario/game.js\n"
             "</parameter>\n</function>\n")
    for i in range(1, len(block) + 1, 3):
        w.feed(block[:i], i / 100)
    w.feed(block, 9.0)
    w.feed(None, 10.0)
    a, b = w.calls
    assert a["name"] == "bash" and a["args"]["command"] == "npm test -- --run"
    assert a["t_target"] is not None and a["closed"]
    assert b["name"] == "write" and b["lines"] == 3 and b["body"].startswith("// game.js")
    assert b["t_body"] < b["t_target"] and b["args"]["filePath"].endswith("mario/game.js")
    assert voice.tool_action("bash", a["args"]) == ("running", "npm test -- --run")
    assert voice.tool_action("write", b["args"]) == ("writing", "mario/game.js")
    assert voice.tool_action("read", {"filePath": "/a/b/c/level.js"}, "de") == ("liest", "c/level.js")
    assert voice.tool_line(*voice.tool_action("bash", {"command": "ls"}, "de"), "de") == \
        "führt aus: ls"
    assert voice.tool_action("chrome-devtools_take_screenshot", {})[0] == "taking a screenshot"
    assert voice.short_command("a\nb") == "a …"
    assert voice.named_in_body("// entities.js: enemies\nx", "Next: entities.js and game.js") == \
        "entities.js"
    assert voice.named_in_body("const x = 'entities.js'", "Next: game.js") is None
    # a header comment that opens with the file's name is the body naming itself
    assert voice.named_in_body("// entities.js: enemies", "Next: game.js") == "entities.js"
    assert voice.named_in_body("/* hooks/usePreloader.ts */\nexport", "") == "hooks/usePreloader.ts"
    assert voice.named_in_body("<!-- index.html -->", "") == "index.html"
    assert voice.named_in_body("// a.js and b.js", "a.js b.js") is None


def test_the_clean():
    src = "The enemies patrol between walls; a stomp squashes them. drawCoins() spins 12 coins."
    ok = lambda s, task="now": voice.clean(s, task, src)  # noqa: E731
    assert ok("How the enemies patrol and when a stomp squashes them") == \
        ("how the enemies patrol and when a stomp squashes them", None)
    assert ok("I will check the enemies") == (None, "first_person")
    assert ok("**enemies** patrol") == (None, "markup")
    assert ok("enemies patrol in render.js") == (None, "unsupported_name")
    assert ok("spins 40 coins") == (None, "unsupported_number")
    assert ok("spins 12 coins with drawCoins")[1] is None
    assert ok("database migrations and cookie headers") == (None, "ungrounded")
    assert ok("enemies") == (None, "short")
    assert ok(" ".join(["enemies"] * 30)) == (None, "words")
    assert ok(voice.EXAMPLES[0][3]) == (None, "example")
    assert ok("Now: enemies patrol between walls")[0] == "enemies patrol between walls"
    assert voice.gerund("fix the stomp") == "fixing the stomp"
    assert voice.gerund("fix the stomp", "de") == "fix the stomp"
    assert voice.gerund("drafts the snake game") == "drafting the snake game"
    assert voice.gerund("fixes the stomp") == "fixing the stomp"
    # an English user's line in German is refused
    assert voice.clean("snake.html erstellt, Tests mit Laya und Screenshots", "plan",
                       "snake.html tests Laya screenshots erstellt") == (None, "language")
    # German phrases are not checked word by word against an English thought (they are
    # translated), but names and numbers still are
    assert voice.clean("wie die Gegner patrouillieren", "now", src, "de")[1] is None
    assert voice.clean("Gegner in render.js", "now", src, "de") == (None, "unsupported_name")
    assert voice.clean("how the enemies patrol and turn", "now", src, "de") == (None, "language")
    # a write's phrase that only names the file again says nothing the line does not
    wsrc = "write game.js // game.js: the loop, the coins and the enemies"
    assert voice.clean("fixes the game.js file", "why", wsrc) == (None, "restates")
    assert voice.clean("to run the loop, the coins and the enemies", "why", wsrc)[0] == \
        "run the loop, the coins and the enemies"


# ------------------------------------------------------------------ 5-9 one request
FAST = {"first_s": 15, "every_s": 25, "min_gap_s": 12, "thought_min_s": 10, "share": 0.03}
PARA = ("The enemies patrol left and right and turn at the walls and at ledges. A stomp from "
        "above squashes an enemy and bounces the player. Coins spin and the flag ends the "
        "level. ")


def test_thinking_lines():
    v, ui, m, clock = make(settings=FAST)
    ids = [0]
    v.token(ids, False)                       # Qwen's first token: the share starts
    out = []
    think(v, clock, ids, PARA * 12)           # ~2,000 tokens at 20 a second: ~100 s
    out += said(ui)
    assert out, "a long thinking says something"
    assert all(x.startswith("thinking ") for x in out), out
    # the first after first_s, then about every every_s (never closer than min_gap_s)
    ats = [e["at_ms"] for e in ui.log]
    assert len(out) >= 2 and all(":" in x for x in out)
    assert re.match(r"thinking (1[5-9]|2\d) s: ", out[0]), out[0]
    # the thought's end: its decision
    ids.append(1)
    clock.t += 0.05
    v.token(ids, False)
    end = said(ui)
    assert len(end) == 1 and end[0].startswith("thought ") and ": " in end[0], end
    assert v.totals()["lines"] == len(out) + 1
    del ats
    # a short thinking: nothing at all
    v, ui, m, clock = make(settings=FAST)
    ids = [0]
    v.token(ids, False)
    think(v, clock, ids, PARA * 2, per_token_s=0.02)     # 340 tokens, ~7 s
    clock.t = 9.0
    ids.append(1)
    v.token(ids, False)
    v.finish(ids)
    assert said(ui) == [], ui.log
    # the same phrase twice is said once
    v, ui, m, clock = make(summary=lambda ids: "the enemies patrol left and right", settings=FAST)
    ids = [0]
    v.token(ids, False)
    think(v, clock, ids, PARA * 12)
    got = said(ui)
    assert len(got) == 1 and v.totals()["why"].get("repeat"), (got, v.totals())


def test_tool_lines():
    blk = ("\n<function=write>\n<parameter=content>\n// render.js: the vector art for the player\n"
           + "ctx.fill();\n" * 120 + "</parameter>\n<parameter=filePath>\n/u/k/mario/render.js\n"
           "</parameter>\n</function>\n")
    before = "Next up is render.js, the modern vector art."

    def run(v, clock, block, per_char_s=0.01):
        ids = [ord(c) for c in before]
        v.token(ids, False)
        v.content(before)
        for i in range(1, len(block) + 1):
            clock.t += per_char_s
            ids.append(ord(block[i - 1]))
            v.token(ids, False)
            v.tool(ids, block[:i])
        v.tool(ids, None)
        v.finish(ids)

    v, ui, m, clock = make(summary=lambda ids: "the vector art for the player", settings=FAST)
    run(v, clock, blk)
    got = said(ui)
    # the path comes last: the body's first line names the file and the words before it too
    assert got[0] == "writing render.js: the vector art for the player", got
    # the job read the call alone: the tool and the body, not the conversation
    seen = body_of(m.jobs_seen[-1])
    assert seen.startswith("write ") and "ctx.fill" in seen and "Next up" not in seen
    # a model that fails: the facts alone
    v, ui, m, clock = make(fail=True, settings=FAST)
    run(v, clock, blk)
    assert said(ui)[0] == "writing render.js", ui.log
    # a call done at once: no line (the client shows it itself)
    v, ui, m, clock = make(settings=FAST)
    run(v, clock, "\n<function=bash>\n<parameter=command>\nls\n</parameter>\n</function>\n",
        per_char_s=0.001)
    assert said(ui) == []
    # a long bash command still streaming after tool_after_s: the action from the structure
    v, ui, m, clock = make(settings=FAST)
    run(v, clock, "\n<function=bash>\n<parameter=command>\nnpm test\n</parameter>\n"
        "<parameter=description>\n" + "x" * 400 + "\n</parameter>\n</function>\n", per_char_s=0.01)
    assert said(ui) == ["running npm test"]
    # a long write's progress, from the lines it has
    v, ui, m, clock = make(summary=lambda ids: "the vector art for the player",
                           settings=dict(FAST, write_every_s=5))
    run(v, clock, blk, per_char_s=0.01)
    got = said(ui)
    assert any(re.match(r"writing render\.js, \d+ lines so far", x) for x in got), got


def test_the_share():
    clock = Clock()
    v, ui, m, clock = make(delay_s=0.04, settings=dict(FAST, share=0.03), clock=clock)
    ids = [0]
    v.token(ids, False)
    think(v, clock, ids, PARA * 12)
    tot = v.totals()
    assert tot["lines"] >= 1
    # what the model held Qwen, against what the share allowed at the end
    allowed = 0.03 * (clock.t - v.share.t0) * 1e3
    assert v.share.spent <= allowed + voice.BUDGET_MS, (v.share.spent, allowed)
    # the shadow's in-turn reading takes what is left, at most
    from server.conductor import shadow as shadow_mod
    s = narr_mod.Share(0.03, clock=clock)
    s.start(0.0)
    s.spend(29.0)
    inl = shadow_mod.Inline(type("Svc", (), {"clock": clock})(), "k", share=0.02, slice_ms=150,
                            budget=s)
    clock.t = 1.0
    inl.t0 = 0.0
    inl.est = 5.0
    # 1 s in: the share allows 30 ms, the voice spent 29: 1 ms left, below one chunk
    assert s.allow(clock.t) == 1.0
    inl.tick([1, 2, 3])                       # returns before touching the service
    assert inl.stats["slices"] == 0


def test_never_silent():
    v, ui, m, clock = make(fail=True, settings=dict(FAST, fallback_s=30))
    ids = [0]
    v.token(ids, False)
    think(v, clock, ids, PARA * 12)
    got = said(ui)
    assert got and all(x.startswith("still thinking, ") for x in got), got


def test_a_plan():
    v, ui, m, clock = make(summary=lambda ids: "index.html, level.js and game.js, then a test",
                           settings=FAST)
    ids = [0]
    v.token(ids, False)
    clock.t = 30.0
    text = ("Plan:\n1. `mario/index.html`: the canvas\n2. `mario/level.js`: the tiles\n"
            "3. `mario/game.js`: the loop\n\nNow the files.\n\n")
    for ch in text:
        ids.append(ord(ch))
        v.token(ids, False)
        v.content(ch)
    v.finish(ids)
    assert said(ui) == ["plan: index.html, level.js and game.js, then a test"], ui.log
    # once: more text later says nothing new
    v.content("\n\n1. a\n2. b\n3. c\n\nx")
    assert said(ui) == []


# ------------------------------------------------------------------ 10-14 the handler
LIVE = {"first_s": 0, "every_s": 0, "min_gap_s": 0, "thought_min_s": 0, "tool_after_s": 0,
        "why_wait_s": 60, "share": 1.0, "end_ms": 5000}
WRITE_TOOLS = [{"type": "function", "function": {"name": "write", "parameters": {
    "type": "object", "properties": {"content": {"type": "string"},
                                     "filePath": {"type": "string"}},
    "required": ["content", "filePath"]}}}]


def _serve(narrator="live", slot=None, live=None, channel="inline"):
    import test_ui_states as tu
    from server import app
    cond, seen = tu.serve("on")
    from test_narrator import narrator_on
    narrator_on(cond)               # SRV-534: the role is off by default
    for name in ("coding", "chat"):
        _, doc = cond.store.get("profiles", name)
        u = dict(doc["ui"], states_channel=channel)
        if narrator is not None:
            u["narrator"] = narrator
        if live is not None:
            u["live"] = live
        doc["ui"] = u
        cond.store.put("profiles", name, doc, note="voice", author="test")
    app.STATE.pop("narrator", None)
    if slot is not None:
        app.STATE["narrator"] = narr_mod.Narrator(
            slot, role=lambda: narr_mod.role_doc(app.STATE),
            labels=lambda: narr_mod.class_labels(app.STATE))
    return cond, seen


def fake_slot(summary=None):
    m = FakeGenerative(summary=summary or first_words())
    slot = Slot(m, role="narrator", gate=threading.Lock(), log=lambda s: None)
    slot.load(wait=True)
    return slot, m


def _write_call():
    return ("FAKE_CALL_JSON " + json.dumps({"name": "write", "arguments": {
        "content": "// render.js: the vector art for the player\n" + "ctx.fill();\n" * 40,
        "filePath": "/u/k/mario/render.js"}}))


def titles(raw):
    import test_state_inline as ti
    return ti.said(raw)


def test_the_stream_carries_live_lines():
    import test_ui_states as tu
    _serve("off")
    off = tu.opencode("Write it. " + _write_call(), tools=WRITE_TOOLS)
    s, m = fake_slot(lambda ids: "the vector art for the player"
                     if "ctx.fill" in body_of(ids) else first_words()(ids))
    cond, _ = _serve("live", s, live=LIVE)
    loop = [t.rstrip(SENT) for t in titles(tu.opencode("FAKE_LOOP think", tools=WRITE_TOOLS)["raw"])]
    assert loop[0].endswith("effort") and not loop[0][0].isupper(), loop
    assert any(t.startswith("thinking ") for t in loop), loop
    assert any(t.startswith("thought ") for t in loop), loop
    on = tu.opencode("Write it. " + _write_call(), tools=WRITE_TOOLS)
    got = [t.rstrip(SENT) for t in titles(on["raw"])]
    # the path comes after the body: the body's header comment names the file
    assert "writing render.js: the vector art for the player" in got, got
    # the same answer, calls and usage as with the voice off
    assert states.strip_text(on["reasoning"]) == states.strip_text(off["reasoning"])
    assert on["content"] == off["content"]
    args = lambda r: "".join((c.get("function") or {}).get("arguments") or "" for c in r["calls"])  # noqa: E731
    assert args(on) == args(off)
    tu.same_usage(on, off)
    rid = json.loads(on["raw"].splitlines()[0][6:])["id"]
    row = [r for r in cond.journal.ring.values() if r.get("request_id") == rid][0]
    st = row["extra"]["states"]
    assert any(e.get("narrator", {}).get("live") for e in st)
    assert any(e.get("shown") for e in st if e["kind"] == "route")
    assert row["extra"]["voice"]["lines"] == 2, row["extra"]["voice"]


def test_the_lines_never_reach_qwen():
    import test_ui_states as tu
    import test_state_inline as ti
    s, _ = fake_slot()
    prompts = {}
    for shape in ("clean", "parts", "trimmed"):
        _, seen = _serve("live", s, live=LIVE)
        first = tu.opencode("Write it. " + _write_call(), tools=WRITE_TOOLS)
        assert SENT in first["reasoning"]
        msg = ti.message(ti.parts(first["raw"]))
        pick = {"parts": lambda x: x, "trimmed": lambda x: x.strip(),
                "clean": states.strip_text}[shape]
        asst = {"role": "assistant", "content": msg["content"],
                "reasoning_content": pick(msg["reasoning_content"])}
        seen.clear()
        tu.opencode("Now add a test.", tools=WRITE_TOOLS,
                    history=[{"role": "user", "content": "Write it. " + _write_call()}, asst])
        prompts[shape] = seen[-1]
    assert prompts["parts"] == prompts["clean"] == prompts["trimmed"]


def test_a_german_user():
    import test_ui_states as tu
    s, _ = fake_slot()
    _serve("live", s, live=LIVE)
    got = []
    for text in ("FAKE_LOOP Bitte schreib das Spiel neu und teste es danach",
                 "Bitte schreib das Spiel neu und teste es danach " + _write_call()):
        got += [t.rstrip(SENT) for t in titles(tu.opencode(text, tools=WRITE_TOOLS)["raw"])]
    assert any(t.startswith("denkt seit ") for t in got), got
    assert any(t.startswith("schreibt ") for t in got), got
    assert not any(t.startswith(("thinking ", "writing ")) for t in got), got


def test_off_is_unchanged():
    import test_ui_states as tu
    _serve(None)
    base = tu._strip_ids(tu.opencode("FAKE_LOOP x", tools=WRITE_TOOLS)["raw"])
    for narrator, with_slot in (("off", True), (None, True), ("thoughts", False)):
        s = m = None
        if with_slot:
            s, m = fake_slot()
        _serve(narrator, s)
        assert tu._strip_ids(tu.opencode("FAKE_LOOP x", tools=WRITE_TOOLS)["raw"]) == base, \
            narrator
        assert m is None or not m.jobs_seen


def _node():
    for c in [os.environ.get("NODE"), shutil.which("node")]:
        if c and os.access(c, os.X_OK):
            return c
    return None


def test_opencodes_parser():
    import test_ui_states as tu
    import test_state_inline as ti
    node = _node()
    binary = os.environ.get("OPENCODE_BIN") or os.path.expanduser("~/.opencode/bin/opencode")
    if node is None or not os.path.exists(binary):
        print("    (skipped: no node or no opencode binary on this machine)")
        return
    from tools import opencode_bundle
    work = tempfile.mkdtemp(prefix="oc-voice-")
    entry = opencode_bundle.extract(binary, os.path.join(work, "provider"))
    s, _ = fake_slot(lambda ids: "the vector art for the player"
                     if "ctx.fill" in body_of(ids) else first_words()(ids))
    _serve("live", s, live=dict(LIVE, write_every_s=0.0001))
    raw = tu.opencode("Write it. " + _write_call(), tools=WRITE_TOOLS)["raw"]
    sse = os.path.join(work, "voice.sse")
    with open(sse, "w") as f:
        f.write(raw)
    r = subprocess.run([node, os.path.join(ROOT, "tools", "opencode_parts.mjs"), entry, sse],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-2000:]
    got = json.loads(r.stdout)
    want = ti.parts(raw)
    assert [(p["type"], p.get("title")) for p in got["parts"]] == \
        [(p["type"], p.get("title")) for p in want]
    kinds = [p["type"] for p in got["parts"]]
    tool_at = kinds.index("tool")
    after = [p.get("title") or "" for p in got["parts"][tool_at + 1:]]
    # the write's lines come after its tool part, each a titled reasoning part, in order
    assert after and all(t.startswith(M + "writing ") for t in after), after
    titled = [p["title"] for p in got["parts"] if p.get("title")]
    assert titled == [M + x for x in ti.said(raw)]
    # what opencode sends back strips to the reasoning without a single line
    msg = got["message"]
    assert M not in states.strip_text(msg["reasoning_content"])
    print(f"    node: {len(got['parts'])} parts, {kinds.count('reasoning')} reasoning, "
          f"tool at {tool_at}, after it {after}")


def test_the_contract():
    ui_s = schema.SCHEMAS["profiles"]["properties"]["ui"]["properties"]
    assert "live" in ui_s["narrator"]["enum"]
    assert set(ui_s["live"]["properties"]) == set(voice.DEFAULTS)
    assert {"tool", "plan", "thinking", "thought"} <= set(ui_s["kinds"]["properties"])
    st = narr_mod.Share(0.03, clock=lambda: 10.0)
    assert st.allow() == 0.0
    st.start(0.0)
    assert abs(st.allow() - 300.0) < 1e-9


if __name__ == "__main__":
    passed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:60s} ok")
            passed += 1
    print(f"{passed} passed")
