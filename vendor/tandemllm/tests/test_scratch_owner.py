"""FX15 (UX3, 2026-10-08): the testing package deletes only the scratch folder its own run made.

The live case: PKG3's A01 run (2026-10-07 23:45, opencode in an existing project
/home/user/pago) wrote "let me not touch pago's files" and then ran `rm -rf /home/user/pago/.tandem`:
the testing package's rule "A {tandem_scratch} left by an earlier run that was cut off: ... delete it
before you start" and its last step `rm -rf {tandem_scratch}`. That `.tandem` held the user's own test
kit (run.mjs, 18:43). Not recoverable.

The test matrix:

  1  `{tandem_scratch}` is the conversation's own folder, `.tandem/<session token>`
     (`.tandem/session` without one); QWEN38_TANDEM_SCRATCH still names another place
  2  the shipped testing text: the create command marks the folder only when it makes it, the
     delete command needs the mark, a leftover is reused (no delete at the start); files-care says
     never delete what this conversation did not create
  3  the commands themselves, run by bash in a project that already has a `.tandem` with the
     user's files: the user's files stay; the run's own folder is made, marked and deleted; a
     folder that was there before (even at the scratch path) is never deleted
  4  the box's stored text migrates (edit 34) and a second run changes nothing

Run: python tests/test_scratch_owner.py
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from server.conductor import packages as P  # noqa: E402
from tools import packages_migrate as M  # noqa: E402

URL = "http://127.0.0.1:8000/s/Ab3xYz9_kLmn"


def _text(name):
    with open(os.path.join(ROOT, "packages", "inline", name)) as f:
        return f.read()


def _commands():
    t = P.with_url(_text("testing.md"), URL)
    make = re.search(r"Create it once with `(.+?)`, so git never lists it", t).group(1)
    drop = re.search(r"delete it only when this run made it: `(.+?)`", t).group(1)
    return make, drop


def _sh(cmd, cwd):
    r = subprocess.run(["bash", "-c", cmd], cwd=cwd, capture_output=True, text=True, timeout=30)
    return r.returncode


def test_the_scratch_is_the_conversations_own_folder():
    old = os.environ.pop(P.SCRATCH_ENV, None)
    try:
        assert P.scratch_for(URL) == ".tandem/Ab3xYz9_kLmn"
        assert P.scratch_for(None) == ".tandem/session"
        os.environ[P.SCRATCH_ENV] = "/tmp/tandem-{token}"
        assert P.scratch_for(URL) == "/tmp/tandem-Ab3xYz9_kLmn"
    finally:
        os.environ.pop(P.SCRATCH_ENV, None)
        if old is not None:
            os.environ[P.SCRATCH_ENV] = old


def test_the_shipped_rules():
    t = _text("testing.md")
    assert "delete it before you start" not in t
    assert M.SCRATCH_RULE_V3 in t and M.LEFTOVER_NEW in t
    assert M.NO_DELETE_RULE in _text("files-care.md")


def test_the_users_tandem_folder_survives_the_run():
    make, drop = _commands()
    with tempfile.TemporaryDirectory() as proj:
        # the user's own .tandem, as in /home/user/pago: a test kit of theirs
        os.makedirs(os.path.join(proj, ".tandem"))
        with open(os.path.join(proj, ".tandem", "run.mjs"), "w") as f:
            f.write("// the user's kit\n")
        assert _sh(make, proj) == 0
        mine = os.path.join(proj, ".tandem", "Ab3xYz9_kLmn")
        assert os.path.isfile(os.path.join(mine, ".made-by-this-run"))
        with open(os.path.join(mine, "run.log"), "w") as f:
            f.write("ok\n")
        _sh(drop, proj)
        assert not os.path.exists(mine), "the run's own folder goes"
        assert os.path.isfile(os.path.join(proj, ".tandem", "run.mjs")), "the user's kit stays"


def test_a_folder_that_was_there_is_never_deleted():
    make, drop = _commands()
    with tempfile.TemporaryDirectory() as proj:
        there = os.path.join(proj, ".tandem", "Ab3xYz9_kLmn")
        os.makedirs(there)
        with open(os.path.join(there, "notes.txt"), "w") as f:
            f.write("the user's\n")
        assert _sh(make, proj) == 0
        assert not os.path.exists(os.path.join(there, ".made-by-this-run")), "not marked: not made"
        _sh(drop, proj)
        assert os.path.isfile(os.path.join(there, "notes.txt"))
    # the old rule deleted it (the A01 command), which is what this replaces
    with tempfile.TemporaryDirectory() as proj:
        os.makedirs(os.path.join(proj, ".tandem"))
        open(os.path.join(proj, ".tandem", "run.mjs"), "w").close()
        old = re.search(r"delete it \(`(rm -rf \{tandem_scratch\})`\)", M.SCRATCH_RULE).group(1)
        _sh(P.with_url(old, None).replace(".tandem/session", ".tandem"), proj)
        assert not os.path.exists(os.path.join(proj, ".tandem", "run.mjs"))


def test_the_box_text_migrates_once():
    with open(os.path.join(ROOT, "packages", "seed.json")) as f:
        doc = json.load(f)
    old_body = _text("testing.md").replace(M.SCRATCH_RULE_V3, M.SCRATCH_RULE).replace(
        M.LEFTOVER_NEW, M.LEFTOVER_OLD)
    fc = _text("files-care.md").replace("\n" + M.NO_DELETE_RULE, "")
    doc["packages"]["testing"] = dict(doc["packages"]["testing"], body=old_body)
    doc["packages"]["files-care"] = dict(doc["packages"]["files-care"], body=fc)
    for p in doc["packages"].values():
        p.pop("body_from", None)
        p.setdefault("body", "")
    new, rep = M.migrate_doc(doc)
    assert rep["testing"] == ["testing: scratch marked, deleted only when made",
                              "testing: a leftover is reused, never deleted"], rep
    assert rep["files-care"] == ["files-care: never delete what this conversation did not create"]
    assert new["packages"]["testing"]["body"] == _text("testing.md")
    assert new["packages"]["files-care"]["body"] == _text("files-care.md")
    assert M.migrate_doc(new)[1] == {}, "a second run changes nothing"


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:64s} ok")
            passed += 1
    print(f"{passed} passed")
