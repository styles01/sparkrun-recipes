"""opencode's synced skills as their first-seen copies (server/skillcanon.py).

The case D3 measured: the same 32-entry skills list, 11 entries in two synced folders, the copy
opencode keeps flipping between sessions. After the fix every session's system message is the
first session's, byte for byte; nothing outside a synced entry changes; a folder unseen for a week
gives way; the choice survives a restart (its file).

Run: python tests/test_skillcanon.py
"""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server import skillcanon  # noqa: E402

A, B = "952c69fc-aaaa_8845c1fe-bbbb", "3f210873-cccc_20d46271-dddd"


def _list(choice: str, desc_b: str = "Make art.") -> str:
    out = ["You are opencode.\n<available_skills>"]
    names = ["algorithmic-art", "brand", "kb-dentist", "docx"]
    for name, c in zip(names, choice):
        if name == "kb-dentist":
            loc = "/Users/x/.claude/skills/kb-dentist/SKILL.md"
            desc = "Dentist knowledge."
        else:
            loc = f"/Users/x/.claude/skills/synced/{A if c == 'A' else B}/{name}/SKILL.md"
            desc = desc_b if (name == "algorithmic-art" and c == "B") else "Make art." \
                if name == "algorithmic-art" else f"{name} skill."
        out.append(f"  <skill>\n    <name>{name}</name>\n    <description>{desc}</description>\n"
                   f"    <location>{loc}</location>\n  </skill>")
    out.append("</available_skills>\n<env>cwd</env>")
    return "\n".join(out)


def _body(text, parts=False):
    content = [{"type": "text", "text": text}] if parts else text
    return {"messages": [{"role": "system", "content": content},
                         {"role": "user", "content": "hi <skill>"}]}


def test_every_session_gets_the_first_sessions_copies():
    with tempfile.TemporaryDirectory() as d:
        c = skillcanon.SkillCanon(os.path.join(d, "s.json"))
        first = _list("AAAA")
        b0, n0 = c.apply(_body(first))
        assert n0 == 0 and b0["messages"][0]["content"] == first
        for flip in ("BABA", "ABBB", "BBAB"):
            got, n = c.apply(_body(_list(flip, desc_b="Make art, v2.")))
            assert got["messages"][0]["content"] == first, flip
            assert n == flip[:2].count("B") + flip[3:].count("B")
            assert got["messages"][1]["content"] == "hi <skill>"
        got, _ = c.apply(_body(_list("BBBB"), parts=True))
        assert got["messages"][0]["content"][0]["text"] == first
        # a restart: the file holds the choice
        c2 = skillcanon.SkillCanon(os.path.join(d, "s.json"))
        assert c2.apply(_body(_list("BBBB")))[0]["messages"][0]["content"] == first


def test_a_folder_unseen_for_a_week_gives_way_and_nothing_else_changes():
    with tempfile.TemporaryDirectory() as d:
        t = [1_000_000.0]
        c = skillcanon.SkillCanon(os.path.join(d, "s.json"), clock=lambda: t[0])
        c.apply(_body(_list("AAAA")))
        t[0] += 8 * 86400
        b, n = c.apply(_body(_list("BBBB")))
        assert n == 0 and b["messages"][0]["content"] == _list("BBBB")
        plain = {"messages": [{"role": "system", "content": "no skills here"}]}
        assert c.apply(plain)[0] is plain


def test_a_copy_outside_synced_is_canonical_too_and_an_edit_is_learned():
    """The captured case, a removed skill's copy under `.trash/` against the synced
    one, alternating between two opencode processes; the chosen copy's own description edit is
    learned, never undone; `<built-in>` entries are left alone."""
    def lst(where, desc="Brand rules."):
        loc = (f"/Users/x/.claude/skills/synced/{A}/brand/SKILL.md" if where == "S" else
               "/Users/x/.claude/skills/.trash/1791021397616-4386-xyRphr/brand/SKILL.md")
        return ("<available_skills>\n  <skill>\n    <name>brand</name>\n    <description>"
                f"{desc}</description>\n    <location>{loc}</location>\n  </skill>\n"
                "  <skill>\n    <name>customize-opencode</name>\n    <description>Built in."
                "</description>\n    <location>&lt;built-in&gt;</location>\n  </skill>\n"
                "</available_skills>")
    with tempfile.TemporaryDirectory() as d:
        c = skillcanon.SkillCanon(os.path.join(d, "s.json"))
        assert c.apply(_body(lst("T")))[1] == 0
        got, n = c.apply(_body(lst("S", "Brand rules, synced.")))
        assert n == 1 and got["messages"][0]["content"] == lst("T")
        # the chosen copy's description edited: learned, and served as edited
        got, n = c.apply(_body(lst("T", "Brand rules v2.")))
        assert n == 0 and got["messages"][0]["content"] == lst("T", "Brand rules v2.")
        got, n = c.apply(_body(lst("S", "Brand rules, synced.")))
        assert n == 1 and got["messages"][0]["content"] == lst("T", "Brand rules v2.")
        # a 0.3.22 file (folder ids, no `loc`) is relearned, never misread
        with open(os.path.join(d, "old.json"), "w") as f:
            f.write('{"skills": {"brand": {"dir": "%s", "desc": "x", "seen": {}}}}' % A)
        c3 = skillcanon.SkillCanon(os.path.join(d, "old.json"))
        assert c3.apply(_body(lst("S")))[1] == 0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
