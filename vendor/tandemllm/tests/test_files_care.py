"""SRV-227 (QA SRV-228): a new folder never replaces the user's work.

The live case (the UX2 run of 2026-10-07, A01 on a 0.4.0 test build): asked 'Create a new folder and
create there a pagoda garden', the Open WebUI agent ran `mkdir -p /home/user/pagoda-garden` on a
folder that existed with the user's own page and wrote a new index.html over it without reading
it; the answer never said a file was replaced. The tools run in the client (Open WebUI's
terminal), so the engine cannot stop the write; what it controls is what the model is told. Every
class that writes files now carries the general package `files-care`, whose rules say what a new
name is, that a file is read before it changes, that a replaced file is copied first, and that the
answer says so. Guidance, not routing: no word of the request decides anything here.

  1  the shipped router lists files-care on every class that writes files (code_edit, debug,
     design_ui, drawing, three_d) and on no other; the package is general, enabled, and its text
     carries the four rules                           test_shipped_documents
  2  the pagoda turn of an Open WebUI agent (profile chat: packages on, rules off) gets the text
     in its note beside the 3D skill; the same documents without the package do not
                                                      test_the_pagoda_turn_carries_the_rules
  3  a later user message of the same session does not bring it again (part-level dedup)
                                                      test_not_injected_again

Run: python tests/test_files_care.py
"""

from __future__ import annotations

import copy
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from test_conductor_router import COMPOSITE, setup, turn  # noqa: E402

# the Open WebUI terminal's tools as it sends them (a shell and a file write: SRV-243 keeps
# files-care, which needs a file write, only for a chat that can write files)
TOOLS = [{"type": "function", "function": {"name": n, "description": n, "parameters": {
    "type": "object", "properties": {a: {"type": "string"} for a in args}}}}
    for n, args in (("run_command", ("command", "cwd")), ("write_file", ("path", "content")),
                    ("read_file", ("path",)))]

from server.conductor import schema  # noqa: E402

PAGODA = "Create a new folder and create there a pagoda garden"
WRITERS = {"code_edit", "debug", "design_ui", "drawing", "three_d"}
RULES = ("does not exist yet", "mkdir -p", "Never write over a file you have not read",
         "copy it first")


def test_shipped_documents():
    pk = schema.SHIPPED_PACKAGES["packages"]["files-care"]
    assert pk["general"] and pk["enabled"] and pk["kind"] == "package"
    for r in RULES:
        assert r in pk["body"], r
    for c in schema.SHIPPED_ROUTER["classes"]:
        has = "files-care" in c["policy"]["packages"]
        assert has == (c["id"] in WRITERS), c["id"]
    for kind, name, doc in __import__("server.conductor.shipped", fromlist=["x"]).documents():
        if kind == "router":
            assert schema.check(kind, doc, packages=schema.SHIPPED_PACKAGES) == []


def _cond(with_package=True):
    """A conductor with the shipped router and packages (the inline 3D skill, as FX6's tests),
    the class question answering three_d."""
    from test_conductor_router import by_label
    cond, fakes = setup(decider=by_label("three_d", 0.93))
    p = copy.deepcopy(schema.SHIPPED_PACKAGES)
    for n, v in p["packages"].items():
        # short texts (the test's tokenizer is one id per character; the note's limit is 8,000)
        if n != "files-care":
            p["packages"][n] = dict(v, source="inline", body=f"# {n}\n\nThe {n} skill.")
    r = copy.deepcopy(schema.SHIPPED_ROUTER)
    if not with_package:
        for c in r["classes"]:
            c["policy"]["packages"] = [x for x in c["policy"]["packages"] if x != "files-care"]
    r["classifier"] = cond.store.get("router")[1]["classifier"]
    cond.store.put("packages", None, p, note="shipped", author="t")
    cond.store.put("router", None, r, note="shipped", author="t")
    # Open WebUI's profile as a new box ships it (packages on, rules off)
    cond.store.put("profiles", "chat", copy.deepcopy(schema.SHIPPED_PROFILES["chat"]),
                   note="shipped", author="t")
    return cond


def _ask(cond, msgs, sid="owui-agent"):
    b = {"model": COMPOSITE, "tools": TOOLS, "messages": msgs}
    plan, ids, _ = turn(cond, b, headers={"X-OpenWebUI-Chat-Id": sid}, kind="open-webui")
    return plan


def _note(plan) -> str:
    return "".join(n["text"] for n in plan.injected)


def test_the_pagoda_turn_carries_the_rules():
    cond = _cond()
    plan = _ask(cond, [{"role": "system", "content": "You are an agent."},
                       {"role": "user", "content": PAGODA}])
    assert plan.row["profile"] == "chat" and plan.row["class_id"] == "three_d", plan.row
    text = _note(plan)
    for r in RULES:
        assert r in text, r
    assert "# threejs-scene" in text, "beside the class's core skill"
    # the documents of nothing about the user's files
    old = _ask(_cond(with_package=False), [{"role": "system", "content": "You are an agent."},
                                           {"role": "user", "content": PAGODA}], sid="old")
    assert old.row["class_id"] == "three_d" and not any(r in _note(old) for r in RULES)


def test_not_injected_again():
    cond = _cond()
    m1 = [{"role": "system", "content": "You are an agent."}, {"role": "user", "content": PAGODA}]
    p1 = _ask(cond, m1, sid="again")
    assert "copy it first" in _note(p1)
    m2 = m1 + [{"role": "assistant", "content": "Built pagoda-garden-2."},
               {"role": "user", "content": "Add a koi pond."}]
    p2 = _ask(cond, m2, sid="again")
    assert "copy it first" not in _note(p2), "the part sits in the conversation already"


if __name__ == "__main__":
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            n += 1
            print(f"ok  {name}")
    print(f"{n} passed")
