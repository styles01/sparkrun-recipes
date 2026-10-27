"""A per-turn-kind effort (F2, 2026-10-02): a tool-result turn may think at its own level.

On spiral, the runs whose tool-result turns ran at chat/low under a 1,024 cap (the SRV-161 bug,
since fixed) reached PASS fastest. The router document's `structural.tool_turn_effort`
(inherit, off, low, medium, high, xhigh; a profile's `thinking.tool_turn_effort` over it) gives a
turn whose newest message is a tool result that level's thinking switch and cap, journaled as
`extra.effort_source = tool_turn` with the class's level beside it; the first turn and every new
user message keep the class's. The note stays the class's, so the prompt is the same token for
token. The test matrix:

  1  inherit (absent, the router's or the profile's) plans every turn as before, and journals
     no effort source                                  test_inherit_is_identical
  2  low on tool turns: the level, the 1,024 cap, the journal, the same prompt ids
                                                       test_low_on_tool_turns
  3  the first turn and a new user message keep the class's effort
                                                       (in 2) and test_a_new_user_message_*
  4  SRV-161's tool media turns are tool turns        test_tool_media_turns_are_covered
  5  a client's own number wins, its effort word yields
                                                       test_the_clients_number_wins
  6  the profile overrides the router, both ways, and its budget for the level applies
                                                       test_the_profile_overrides_the_router
  7  off closes the thinking; a checker FAIL raises the level as before; a compaction and the
     effort decision off are untouched; shadow journals the level it would set
                                                       test_off_raise_compaction_and_modes
  8  the schemas take the levels and refuse anything else
                                                       test_the_schemas

Run: python tests/test_tool_turn_effort.py
"""

from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from test_conductor_router import CODE, SYSTEM, body, by_label, setup, turn  # noqa: E402
from test_tool_media_turn import SHAPES  # noqa: E402

from server.conductor import schema  # noqa: E402


def router_level(cond, level):
    _, r = cond.store.get("router")
    if level is None:
        r["structural"].pop("tool_turn_effort", None)
    else:
        r["structural"]["tool_turn_effort"] = level
    cond.store.put("router", None, r, note=f"tool turns {level}", author="t")


def profile_set(cond, path, value, profile="coding"):
    _, doc = cond.store.get("profiles", profile)
    node = doc
    for k in path[:-1]:
        node = node.setdefault(k, {})
    node[path[-1]] = value
    cond.store.put("profiles", profile, doc, note=f"{'.'.join(path)} {value}", author="t")


def call(i, name="bash", args=None):
    return {"role": "assistant", "content": "", "tool_calls": [
        {"id": f"c{i}", "type": "function",
         "function": {"name": name, "arguments": json.dumps(args or {"command": "ls"})}}]}


def result(i, text="ok"):
    return {"role": "tool", "tool_call_id": f"c{i}", "content": text}


# a code_edit session: the first message, two tool turns, a tool's media, a new user message,
# one more tool turn
SESSION = [
    ("user", [{"role": "system", "content": SYSTEM}, {"role": "user", "content": CODE}]),
    ("tool", [call(11), result(11)]),
    ("tool", [call(12, "edit", {"filePath": "src/lexer.ts"}), result(12, "edited")]),
    ("media", SHAPES["read a shot"]()),
    ("user", [{"role": "assistant", "content": "Done: every token has its line."},
              {"role": "user", "content": "Now add the column number too."}]),
    ("tool", [call(17), result(17)]),
]


def run(cond, sid="ses-f2", **kw):
    """Every turn of SESSION: (kind, what the plan set, the prompt ids served)."""
    msgs, out = [], []
    for kind, add in SESSION:
        msgs += add
        p, ids, _ = turn(cond, body(list(msgs), **kw), headers={"x-session-id": sid})
        r = p.row
        out.append((kind, {"class": r["class_id"], "effort": r["effort"], "budget": p.budget,
                           "budget_source": r["budget_source"], "thinking": r["thinking"],
                           "thinking_off": p.thinking_off,
                           "extra": {k: v for k, v in r["extra"].items()}}, ids))
    return out


def fresh(**kw):
    return setup(decider=by_label("code_edit", 0.91), **kw)[0]


# ------------------------------------------------------------------ 1

def test_inherit_is_identical():
    base = run(fresh())
    assert [b[1]["effort"] for b in base] == ["xhigh"] * 6, base
    assert all("effort_source" not in b[1]["extra"] for b in base)
    for how in ("router", "profile", "both"):
        cond = fresh()
        if how in ("router", "both"):
            router_level(cond, "inherit")
        if how in ("profile", "both"):
            profile_set(cond, ["thinking", "tool_turn_effort"], "inherit")
        got = run(cond)
        assert [(k, d) for k, d, _ in got] == [(k, d) for k, d, _ in base], how
        assert [ids for *_, ids in got] == [ids for *_, ids in base], f"{how}: the same prompts"


# ------------------------------------------------------------------ 2 and 3

def test_low_on_tool_turns():
    base = run(fresh())
    cond = fresh()
    router_level(cond, "low")
    got = run(cond)
    for (kind, d, ids), (_, b, bids) in zip(got, base):
        assert d["class"] == "code_edit", d
        assert ids == bids, f"{kind}: the note is the class's, the prompt token for token"
        if kind == "user":
            assert d == b, f"a person's turn keeps the class's effort: {d}"
            assert d["effort"] == "xhigh" and d["budget"] is None
        else:
            assert (d["effort"], d["budget"], d["budget_source"], d["thinking"],
                    d["thinking_off"]) == ("low", 1024, "class", "on", False), (kind, d)
            assert (d["extra"]["effort_source"], d["extra"]["class_effort"]) == \
                ("tool_turn", "xhigh"), d


def test_a_new_user_message_after_tool_turns_keeps_the_class():
    """A new class decided on a person's message after tool turns sets that turn's effort; the
    tool turns after it are low again."""
    cond, _ = setup(decider=lambda t, o: ("research", 0.95) if "compare" in t else
                    ("code_edit", 0.91))
    router_level(cond, "low")
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": CODE},
            call(1), result(1),
            {"role": "assistant", "content": "Done."},
            {"role": "user", "content": "Now compare two lexer designs for me."}]
    p, _, _ = turn(cond, body(msgs), headers={"x-session-id": "n"})
    assert (p.row["class_id"], p.row["effort"], p.budget) == ("research", "medium", 8192), p.row
    assert "effort_source" not in p.row["extra"]
    msgs += [call(2), result(2)]
    p, _, _ = turn(cond, body(msgs), headers={"x-session-id": "n"})
    assert (p.row["class_id"], p.row["effort"], p.budget, p.row["extra"]["class_effort"]) == \
        ("research", "low", 1024, "medium"), p.row


# ------------------------------------------------------------------ 4

def test_tool_media_turns_are_covered():
    for name, tail in SHAPES.items():
        cond = fresh()
        router_level(cond, "low")
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": CODE}]
        turn(cond, body(list(msgs)), headers={"x-session-id": "m"})
        msgs += tail()
        p, _, _ = turn(cond, body(list(msgs)), headers={"x-session-id": "m"})
        assert p.row["kept"] == "tool_media", (name, p.row)
        assert p.row["extra"]["effort_source"] == "tool_turn", (name, p.row)
        # a checker FAIL among the results raises low one level, as it raised the class's
        want = ("medium", 8192) if "FAIL" in json.dumps(tail()) else ("low", 1024)
        assert (p.row["effort"], p.budget) == want, (name, p.row)


# ------------------------------------------------------------------ 5

def test_the_clients_number_wins():
    for kw in ({"max_reasoning_tokens": 777}, {"reasoning": {"max_tokens": 777}},
               {"thinking": {"type": "enabled", "budget_tokens": 777}}):
        cond = fresh()
        router_level(cond, "low")
        got = run(cond, **kw)
        for kind, d, _ in got:
            assert (d["budget"], d["budget_source"]) == (None, "client"), (kw, kind, d)
        p_tool = got[1][1]
        assert p_tool["effort"] == "low" and p_tool["extra"]["effort_source"] == "tool_turn"
    # an effort word yields to the turn's level, as it yields to the class's
    cond = fresh()
    router_level(cond, "low")
    got = run(cond, reasoning_effort="medium")
    assert [(d["effort"], d["budget"]) for _, d, _ in got] == [
        ("xhigh", None), ("low", 1024), ("low", 1024), ("low", 1024), ("xhigh", None),
        ("low", 1024)], got


# ------------------------------------------------------------------ 6

def test_the_profile_overrides_the_router():
    cond = fresh()
    router_level(cond, "low")
    profile_set(cond, ["thinking", "tool_turn_effort"], "inherit")
    assert all(d["effort"] == "xhigh" and "effort_source" not in d["extra"]
               for _, d, _ in run(cond)), "the profile's inherit is the class's"
    cond = fresh()
    profile_set(cond, ["thinking", "tool_turn_effort"], "medium")
    assert [d["effort"] for _, d, _ in run(cond)] == ["xhigh", "medium", "medium", "medium",
                                                      "xhigh", "medium"]
    # another profile (no field) keeps the router's inherit
    p, _, _ = turn(cond, body([{"role": "user", "content": CODE}, call(11), result(11)]),
                   headers={"x-session-id": "o", "X-Tandem-Profile": "overnight"})
    assert p.row["effort"] == "xhigh", p.row
    # the profile's own cap for the level
    cond = fresh()
    profile_set(cond, ["thinking", "tool_turn_effort"], "low")
    profile_set(cond, ["thinking", "budgets"], {"low": 512})
    got = run(cond)
    assert (got[1][1]["effort"], got[1][1]["budget"], got[1][1]["budget_source"]) == \
        ("low", 512, "profile"), got[1]


# ------------------------------------------------------------------ 7

def test_off_raise_compaction_and_modes():
    cond = fresh()
    router_level(cond, "off")
    got = run(cond)
    tool = got[1][1]
    assert (tool["effort"], tool["budget"], tool["thinking"], tool["thinking_off"]) == \
        ("off", None, "off", True), tool
    assert got[0][1]["thinking_off"] is False
    # off closes the generation prompt's <think> on the tool turn and nowhere else
    assert got[1][2] != run(fresh())[1][2]
    # a compaction is not a tool turn
    cond = fresh()
    router_level(cond, "low")
    p, _, _ = turn(cond, body([{"role": "user", "content": "x " * 3000}], tools=False),
                   headers={"x-session-id": "c"})
    assert p.row["class_id"] == "summarize" and "effort_source" not in p.row["extra"], p.row
    # effort decision off: nothing is set or journaled
    cond = fresh()
    router_level(cond, "low")
    profile_set(cond, ["decisions", "effort"], "off")
    for _, d, _ in run(cond):
        assert (d["effort"], d["budget"]) == (None, None) and "effort_source" not in d["extra"]
    # shadow: the turn runs as the server's, the journal holds the level it would set
    cond = fresh()
    router_level(cond, "low")
    profile_set(cond, ["decisions", "effort"], "shadow")
    msgs = [{"role": "user", "content": CODE}, call(1), result(1)]
    p, _, _ = turn(cond, body(msgs), headers={"x-session-id": "s"})
    assert p.budget is None and p.row["effort"] is None
    assert (p.row["shadow"]["effort"]["effort"], p.row["shadow"]["effort"]["budget"],
            p.row["extra"]["effort_source"]) == ("low", 1024, "tool_turn"), p.row


# ------------------------------------------------------------------ 8

def test_the_schemas():
    assert schema.TOOL_TURN_EFFORTS == ("inherit", "off", "low", "medium", "high", "xhigh")
    r = schema.seed("router")
    assert "tool_turn_effort" not in r["structural"], "the seed is unchanged: inherit"
    assert schema.check("router", r) == []
    for lv in schema.TOOL_TURN_EFFORTS:
        r["structural"]["tool_turn_effort"] = lv
        assert schema.check("router", r) == [], lv
    r["structural"]["tool_turn_effort"] = "max"
    assert any("tool_turn_effort" in e for e in schema.check("router", r))
    p = schema.seed("profiles", "overnight")
    assert "tool_turn_effort" not in p["thinking"]
    p["thinking"]["tool_turn_effort"] = "low"
    assert schema.check("profiles", p, name="overnight") == []
    p["thinking"]["tool_turn_effort"] = "minimal"
    assert any("tool_turn_effort" in e for e in schema.check("profiles", p, name="overnight"))
    assert schema.tool_turn_effort({}, {}) == "inherit"
    assert schema.tool_turn_effort({"structural": {"tool_turn_effort": "low"}},
                                   {"thinking": {}}) == "low"
    assert schema.tool_turn_effort({"structural": {"tool_turn_effort": "low"}},
                                   {"thinking": {"tool_turn_effort": "inherit"}}) == "inherit"
    assert schema.meta()["tool_turn_efforts"] == list(schema.TOOL_TURN_EFFORTS)


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:68s} ok")
            passed += 1
    print(f"{passed} passed")
