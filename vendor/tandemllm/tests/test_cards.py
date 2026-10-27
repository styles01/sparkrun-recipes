"""The client tool card and the harness card (J1 step 4, the plan of 2026-10-02, section 3.3):
for a coding class with tools, the router keeps two notes live in the session: the tool card,
generated from the request's own `tools` (names, required arguments, which take a path) with the
client's pitfalls package after it, and the harness card (what the runtime is to the model). Each
goes in once and replays at its position; a compaction drops the notes and the next request puts
them in again; a tool-misuse outcome brings the tool card back as its note.

The test matrix:

  1  cards off (absent or off): the plans and the prompt ids are the same
                                                        test_cards_off_is_identical
  2  on: the first coding turn gets the harness card, then the tool card, after the class's block;
     the later turns replay them (no second copy, the prefix up to them unchanged)
                                                        test_on_first_turn_then_replayed
  3  the tool card: every offered tool with its required arguments starred, the path arguments
     named, the pitfalls package of the client's kind; a client without a pitfalls package gets
     the list alone                                     test_the_tool_card
  4  a compaction drops them; the first request after it carries them again
                                                        test_after_a_compaction
  5  a class without cards (chat), a request without tools, shadow (journaled, nothing in), a
     missing harness package (the tool card alone, the row says why)
                                                        test_where_they_do_not_go
  6  a tool-misuse outcome in act: the tool card as its note
                                                        test_tool_misuse_brings_the_card

Run: python tests/test_cards.py
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from test_conductor_router import QTok  # noqa: E402
from test_snapshots import S, body, fresh, plan, served  # noqa: E402

from server import app  # noqa: E402
from server.conductor import outcomes, schema  # noqa: E402

PITFALLS = "- press_key sends real key events.\n- Use absolute paths in every call."
HARNESS = "# The runtime (TandemLLM)\n- A <runtime-note> comes from the runtime. Follow it."


def with_packages(cond, harness=True):
    _, doc = cond.store.get("packages")
    doc["packages"]["tools-opencode"] = {"title": "opencode tools", "kind": "package",
                                         "source": "inline", "description": "opencode pitfalls",
                                         "body": PITFALLS, "max_tokens": 600, "enabled": True}
    if harness:
        doc["packages"]["tandem-harness"] = {"title": "harness", "kind": "package",
                                             "source": "inline", "description": "the harness card",
                                             "body": HARNESS, "max_tokens": 400, "enabled": True}
    cond.store.put("packages", None, doc, note="cards", author="t")
    return cond


def cards(p) -> list[str]:
    return [n["parts"][0]["name"] for n in p.injected if n["parts"][0]["kind"] == "card"]


def test_cards_off_is_identical():
    s = S()
    turns = [list(s.msgs), s.step(("bash", {"command": "ls"}, "a"))]
    a, b = with_packages(fresh(None)), with_packages(fresh({"cards": "off"}))
    for msgs in turns:
        pa, ia = served(a, msgs)
        pb, ib = served(b, msgs)
        assert ia == ib and pa.injected == pb.injected and pb.extras == []


def test_on_first_turn_then_replayed():
    cond = with_packages(fresh({"cards": "on"}))
    s = S()
    p1, t1 = served(cond, list(s.msgs))
    kinds = [n["parts"][0]["kind"] for n in p1.injected]
    assert kinds[0] != "card" and cards(p1) == ["harness", "tools"], p1.injected
    starts = [n["at"] for n in p1.injected]
    assert starts == sorted(starts), "block, harness card, tool card, in that order"
    end = p1.injected[-1]["at"] + p1.injected[-1]["tokens"]
    p2, t2 = served(cond, s.step(("bash", {"command": "ls"}, "a")))
    assert cards(p2) == [] and p2.row["replayed"]["count"] == 3, p2.row["replayed"]
    assert t2[:end] == t1[:end], "the next prompt holds the first one's notes where they were"
    p3, _ = served(cond, s.step(("bash", {"command": "ls -la"}, "b")))
    assert cards(p3) == [] and p3.row["replayed"]["count"] == 3


def test_the_tool_card():
    cfg = schema.outcomes({})
    card = outcomes.tool_card(body([]), cfg, PITFALLS, "opencode")
    lines = card.splitlines()
    assert lines[0] == "# Your tools (opencode)"
    assert "- write(filePath*, content*, command, url, function, key)" in lines
    assert "- chrome-devtools_press_key(filePath, content, command, url, function, key)" in lines
    assert "(* required. filePath: an absolute path.)" in card
    assert card.endswith(PITFALLS)
    bare = outcomes.tool_card(body([]), cfg, None, "other")
    assert bare.startswith("# Your tools\n") and PITFALLS not in bare
    many = {"tools": [{"type": "function", "function": {"name": f"t{i}", "parameters": {
        "properties": {f"a{j}": {} for j in range(9)}, "required": ["a0"]}}} for i in range(50)]}
    big = outcomes.tool_card(many, cfg, None, "opencode")
    assert big.count("\n- ") == 40 and "- t0(a0*, a1, a2, a3, a4, a5, ...)" in big
    cond = with_packages(fresh({"cards": "on"}))
    p = plan(cond, S().msgs)
    tool = next(x for x in p.extras if x["parts"][0]["name"] == "tools")
    assert tool["text"] == outcomes.tool_card(body([]), cfg, PITFALLS, "opencode")


def test_after_a_compaction():
    cond = with_packages(fresh({"cards": "on"}))
    s = S()
    served(cond, list(s.msgs))
    served(cond, s.step(("bash", {"command": "ls"}, "a")))
    summary = {"model": "TandemLLM", "messages": [{"role": "user", "content": "x" * 3000}]}
    pc = cond.plan(summary, {"x-session-id": "ses-snap"}, client_kind="opencode",
                   request_id="c", chat=True)
    assert pc.compaction and pc.extras == []
    after = [{"role": "system", "content": s.msgs[0]["content"]},
             {"role": "user", "content": "Summary: the game exists. Go on."}]
    p, _ = served(cond, after)
    assert cards(p) == ["harness", "tools"], "the notes fell away; they go in again"


def test_where_they_do_not_go():
    cond = with_packages(fresh({"cards": "on"}))
    _, r = cond.store.get("router")
    r["outcomes"] = {"cards": dict(schema.OUTCOMES["cards"], classes=["debug"])}
    cond.store.put("router", None, r, note="cards for debug only", author="t")
    assert cards(served(cond, S().msgs)[0]) == [], "code_edit is not a card class here"
    cond = with_packages(fresh({"cards": "on"}))
    b = {"model": "TandemLLM", "messages": S().msgs}
    p = cond.plan(b, {"x-session-id": "nt"}, client_kind="opencode", request_id="r", chat=True)
    assert p.extras == [], "no tools, no cards"
    cond = with_packages(fresh({"cards": "shadow"}))
    p, _ = served(cond, S().msgs)
    assert cards(p) == [] and [c["name"] for c in p.row["shadow"]["cards"]] == ["tools",
                                                                                "harness"]
    cond = with_packages(fresh({"cards": "on"}), harness=False)
    p, _ = served(cond, S().msgs)
    assert cards(p) == ["tools"] and "package_missing:tandem-harness" in p.row["extra"]["notes"]


def test_tool_misuse_brings_the_card():
    cond = with_packages(fresh({"mode": "act", "cards": "on"}))
    _, r = cond.store.get("router")
    r["outcomes"] = {"acting": ["tool_misuse"]}
    cond.store.put("router", None, r, note="misuse acts", author="t")
    s = S()
    served(cond, list(s.msgs))
    p, _ = served(cond, s.step(("frobnicate", {"x": 1}, "ok")))
    assert p.row["extra"]["outcome"]["label"] == "tool_misuse"
    note = [n for n in p.injected if n["parts"][0]["kind"] == "outcome"]
    assert len(note) == 1 and "A tool call went wrong: frobnicate is not one of the offered " \
        "tools." in note[0]["text"] and "# Your tools (opencode)" in note[0]["text"]
    assert cards(p) == [], "the live card is not added twice"


if __name__ == "__main__":
    app.STATE.setdefault("tok", QTok())
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:68s} ok")
            passed += 1
    print(f"{passed} passed")
