"""tools/library_install.py (R1): the dry run writes nothing; --apply copies the library under the
destination, registers every skill as a library package (and the card), adds the library
directory to `skill_dirs`, and sets a profile's switch, each one new version through the server's
own API; a second run copies nothing; a text the server would cut stops the install.

Run: python tests/test_library_install.py
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
import tempfile
import threading
from http.server import ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_auth import ADMIN, _setup  # noqa: E402  (first: the CPU environment)
from test_libsearch import SKILLS, make_library, skill_md  # noqa: E402

from server import app  # noqa: E402
from server.conductor.hook import Conductor  # noqa: E402
from tools import library_install as LI  # noqa: E402

CARD = ("---\nname: tandem-library\ndescription: How to search the TandemLLM skill library.\n---\n"
        "# The skill library\n\ncurl -s \"$L/search?q=...\"\n")


def run(*args) -> tuple[int, str]:
    out = io.StringIO()
    argv = sys.argv
    sys.argv = ["library_install.py", *args]
    try:
        with contextlib.redirect_stdout(out):
            try:
                rc = LI.main()
            except SystemExit as exc:
                rc = exc.code if isinstance(exc.code, int) else 1
                print(exc.code)
    finally:
        sys.argv = argv
    return rc, out.getvalue()


def test_dry_run_then_apply():
    _setup()
    cond = Conductor(tempfile.mkdtemp(prefix="cond-"), state=app.STATE, journal=False)
    app.STATE["conductor"] = cond
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), app.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    src = make_library()
    card = os.path.join(tempfile.mkdtemp(), "SKILL.md")
    with open(card, "w") as f:
        f.write(CARD)
    dest = tempfile.mkdtemp(prefix="skills-")
    os.environ["QSE_ADMIN_TOKEN"] = ADMIN
    common = ["--src", src, "--card", card, "--dest", dest, "--base", base]
    try:
        v0 = cond.store.get("packages")[0]
        p0 = cond.store.get("profiles", "coding")[0]
        rc, out = run(*common, "--profile", "coding", "--library", "shadow")
        assert rc == 0, out
        assert f"{len(SKILLS)} skills" in out and "files: new 8, changed 0, same 0" in out, out
        assert "valid;" in out and "library off -> shadow" in out and "dry run" in out
        assert os.listdir(dest) == [] and cond.store.get("packages")[0] == v0
        assert cond.store.get("profiles", "coding")[0] == p0
        assert ADMIN not in out, "the token is never printed"
        rc, out = run(*common, "--profile", "coding", "--library", "shadow", "--apply")
        assert rc == 0, out
        assert sorted(os.listdir(os.path.join(dest, "library"))) == sorted(
            list(SKILLS) + ["INDEX.md"])
        assert os.path.isfile(os.path.join(dest, "tandem-library", "SKILL.md"))
        v, doc = cond.store.get("packages")
        assert v == v0 + 1 and doc["skill_dirs"][-2:] == [dest + "/library", dest]
        for n in SKILLS:
            p = doc["packages"][n]
            assert p["library"] and p["enabled"] and p["source"] == f"skill:{n}"
            assert p["description"] == SKILLS[n][0] and p["max_tokens"] == LI.MAX_TOKENS
            text, err = cond.library.text(doc, n)
            assert err is None and text == f"# {n}\n\n{SKILLS[n][1]}", (n, err, text)
        assert cond.library.text(doc, "tandem-library")[0].startswith("# The skill library")
        assert "library" not in doc["packages"]["tandem-library"]
        assert cond.store.get("profiles", "coding")[1]["library"] == "shadow"
        # a second run: the files are the same, a disabled skill stays disabled
        doc["packages"]["git-workflow"]["enabled"] = False
        cond.store.put("packages", None, doc, note="off", author="t")
        rc, out = run(*common, "--apply")
        assert rc == 0 and "files: new 0, changed 0, same 8" in out, out
        assert cond.store.get("packages")[1]["packages"]["git-workflow"]["enabled"] is False
        # a text the server would cut stops everything
        os.makedirs(os.path.join(src, "huge"))
        with open(os.path.join(src, "huge", "SKILL.md"), "w") as f:
            f.write(skill_md("huge", "A skill far too long for its cap.", "word " * 600, "ref"))
        v = cond.store.get("packages")[0]
        rc, out = run(*common, "--apply")
        assert rc != 0 and "would cut" in out and cond.store.get("packages")[0] == v, out
    finally:
        httpd.shutdown()
        os.environ.pop("QSE_ADMIN_TOKEN", None)


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:68s} ok")
            passed += 1
    print(f"{passed} passed")
