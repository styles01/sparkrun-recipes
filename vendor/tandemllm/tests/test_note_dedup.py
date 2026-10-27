"""A class's note carries only the parts no live note of the session holds (QWEN38_NOTE_DEDUP).

Live on a 0.4.0 test build (an Open WebUI agent chat, 2026-10-06): every user message of the coding class
whose task skill differed from the last one's (test-lib, none, test-cli) brought the whole note
again, the general packages included (about 4,800 tokens each time). The third copy pushed the
replayed notes over `max_replay_tokens`, the oldest note went, the prompt changed at its position
(12,292) and the turn read 35,442 tokens again (49 s). Replayed on the live engine (one build,
n = 1): the fifth code message of a 20,000-token conversation read it all again (cached share
0.000, 60.7 s).

  1. a new user message whose task skill differs gets a note with that skill alone: the general
     package and the effort sentence are live already and stay where they are;
  2. a skill whose text changed counts as new; every part live: no note at all;
  3. over many such messages the first note keeps its place: the replayed notes stay under the
     budget and the prompt never changes before the newest message (with `block`, the 0.3.47
     behaviour, the oldest note goes and the first position moves).

Run: python tests/test_note_dedup.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_conductor_router as T  # noqa: E402
from server.conductor import router as router_mod  # noqa: E402

CORE = "# Coding core\n\n" + "Run after every step, and read the file before you change it.\n" * 40
SKILLS = {"lib-skill": "library", "cli-skill": "command-line", "api-skill": "endpoint",
          "data-skill": "csv"}
ASKS = {"lib-skill": "Write a small library module slug.py with a slugify function and tests.",
        "cli-skill": "Now write a command-line tool that reads stdin and prints the slug.",
        "api-skill": "Add an HTTP endpoint that returns the slug as JSON, with a test.",
        "data-skill": "Write a script that slugifies the name column of a csv file."}


def _pick(text, options):
    if "none" in options:                    # the package question: the skill the text asks for
        for name, word in SKILLS.items():
            if word in text and name in options:
                return name, 0.9
        return "none", 0.9
    return "code_edit", 0.95


def _setup(budget=None):
    cond, fakes = T.setup(decider=_pick)
    _, p = cond.store.get("packages")
    p["packages"]["coding"] = {"title": "Coding", "kind": "package", "source": "inline",
                               "description": "How programming work must happen.", "body": CORE,
                               "max_tokens": 3000, "enabled": True, "general": True}
    for n, w in SKILLS.items():
        p["packages"][n] = {"title": n, "kind": "skill", "source": "inline",
                            "description": f"Testing a {w} piece of code.",
                            "body": f"# Testing a {w}\n\n1. Run it on real input.",
                            "max_tokens": 3000, "enabled": True}
    cond.store.put("packages", None, p, note="coding general", author="t")
    _, r = cond.store.get("router")
    for c in r["classes"]:
        if c["id"] == "code_edit":
            c["policy"]["packages"] = ["coding"] + list(SKILLS)
    if budget is not None:
        r["injection"]["max_replay_tokens"] = budget
    cond.store.put("router", None, r, note="skills on code_edit", author="t")
    return cond, p


def _conversation(cond, skills):
    msgs = [{"role": "system", "content": T.SYSTEM}]
    out = []
    for i, s in enumerate(skills):
        msgs.append({"role": "user", "content": ASKS[s]})
        plan, ids, _ = T.turn(cond, T.body(list(msgs)), headers={"x-session-id": "dedup"})
        out.append((plan, ids))
        msgs.append({"role": "assistant", "content": f"Done with step {i}."})
    return out


def _parts(plan):
    return [p["name"] for n in plan.injected for p in n["parts"]]


def test_a_new_skill_brings_only_itself():
    cond, _ = _setup()
    (p1, ids1), (p2, ids2), (p3, ids3) = _conversation(cond, ["lib-skill", "cli-skill",
                                                               "lib-skill"])
    assert "coding" in _parts(p1) and "lib-skill" in _parts(p1), _parts(p1)
    assert _parts(p2) == ["cli-skill"], _parts(p2)
    assert any(n.startswith("parts_live:") and "coding" in n for n in p2.row["extra"]["notes"])
    # every part of turn 3's block is live (turn 1's note holds lib-skill): no note
    assert not p3.injected, p3.injected
    assert T.text_of(ids3).count("# Coding core") == 1
    assert p3.row["replayed"]["count"] == 2


def test_a_changed_text_is_new_and_the_old_mode_keeps_the_whole_note():
    cond, p = _setup()
    (p1, _), = _conversation(cond, ["lib-skill"])
    p["packages"]["coding"]["body"] = CORE + "\nOne more rule.\n"
    cond.store.put("packages", None, p, note="coding edited", author="t")
    msgs = [{"role": "system", "content": T.SYSTEM}, {"role": "user", "content": ASKS["lib-skill"]},
            {"role": "assistant", "content": "Done with step 0."},
            {"role": "user", "content": ASKS["cli-skill"]}]
    p2, _, _ = T.turn(cond, T.body(msgs), headers={"x-session-id": "dedup"})
    assert _parts(p2) == ["coding", "cli-skill"], _parts(p2)
    old = router_mod.NOTE_DEDUP
    router_mod.NOTE_DEDUP = "block"
    try:
        cond, _ = _setup()
        (q1, _), (q2, _) = _conversation(cond, ["lib-skill", "cli-skill"])
        assert "coding" in _parts(q2) and "cli-skill" in _parts(q2), _parts(q2)
    finally:
        router_mod.NOTE_DEDUP = old


def test_the_first_note_keeps_its_place_under_the_budget():
    # a budget that holds the first note and a few skills, not three copies of the core
    budget = len(CORE) * 2 + 1500
    for mode in ("parts", "block"):
        old = router_mod.NOTE_DEDUP
        router_mod.NOTE_DEDUP = mode
        try:
            cond, _ = _setup(budget=budget)
            turns = _conversation(cond, ["lib-skill", "cli-skill", "api-skill", "data-skill",
                                         "lib-skill"])
        finally:
            router_mod.NOTE_DEDUP = old
        first = turns[1][0].row["replayed"]["positions"][0]
        last = turns[-1][0].row["replayed"]["positions"]
        if mode == "parts":
            assert last[0] == first, (first, last)
            for (_, a), (_, b) in zip(turns, turns[1:]):
                # the served prompt never changes before the first note's position
                assert b[:first] == a[:first]
            assert T.text_of(turns[-1][1]).count("# Coding core") == 1
        else:
            assert last[0] != first, "the 0.3.47 mode evicts the oldest note (the bug)"


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:64s} ok")
            passed += 1
    print(f"{passed} passed")
