"""tools/playtest_install.py (X1): the dry run stores nothing; --apply stores laya-playtest (an
inline general library package: its core goes into the prompt, the reference is read on request),
the new texts of laya-testing and test-web, laya-playtest right after laya-testing on every class
that lists it, and a profile's `ui.states_channel: inline`, each one new version through the
server's own API; a second run stores nothing; a core the server would cut stops the install.

Run: python tests/test_playtest_install.py
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

from server import app  # noqa: E402
from server.conductor.hook import Conductor  # noqa: E402
from tools import playtest_install as PI  # noqa: E402

PLAYTEST = ("---\nname: laya-playtest\ndescription: Laya plays the software you built.\n---\n"
            "# Laya plays your build\n\nFetch the kit, expose the state, run 3 episodes.\n\n"
            "## Reference\n\n### A browser game\nwindow.tandem_state = () => ({x})\n")
LT = "---\nname: laya-testing\ndescription: Laya judges test evidence.\n---\n# Laya\n\nAsk it.\n"


def run(*args) -> tuple[int, str]:
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        try:
            rc = PI.main(list(args))
        except SystemExit as exc:
            rc = exc.code if isinstance(exc.code, int) else 1
            print(exc.code)
    return rc, out.getvalue()


def tree(playtest=PLAYTEST):
    d = tempfile.mkdtemp(prefix="x1-pk-")
    for name, text in (("laya-playtest", playtest), ("laya-testing", LT + "- And it plays (laya-playtest).\n"),
                       ("test-web", "---\nname: test-web\ndescription: Web tests.\n---\n# Web\n\n8. Laya plays it.\n")):
        os.makedirs(os.path.join(d, name))
        with open(os.path.join(d, name, "SKILL.md"), "w") as f:
            f.write(text)
    return d


def test_dry_run_apply_again():
    _setup()
    cond = Conductor(tempfile.mkdtemp(prefix="cond-"), state=app.STATE, journal=False)
    app.STATE["conductor"] = cond
    # what the served engine has: L1's packages and laya-testing on two classes
    _, pk = cond.store.get("packages")
    for n, body in (("laya-testing", "# Laya\n\nAsk it."), ("test-web", "# Web")):
        pk["packages"][n] = {"title": n, "kind": "package", "source": "inline",
                             "description": "the old description", "body": body, "max_tokens": 1200,
                             "enabled": True, "general": True}
    cond.store.put("packages", None, pk, note="L1", author="t")
    _, rt = cond.store.get("router")
    for c in rt["classes"]:
        if c["id"] in ("code_edit", "debug"):
            c["policy"]["packages"] = ["laya-testing"] + [p for p in c["policy"].get("packages", [])
                                                          if p != "laya-testing"]
    cond.store.put("router", None, rt, note="L1", author="t")
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), app.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    os.environ["QSE_ADMIN_TOKEN"] = ADMIN
    common = ["--tree", tree(), "--base", base, "--packages", "--attach", "--inline", "coding"]
    try:
        vs = lambda: (cond.store.get("packages")[0], cond.store.get("router")[0],   # noqa: E731
                      cond.store.get("profiles", "coding")[0])
        v0 = vs()
        rc, out = run(*common)
        assert rc == 0, out
        assert "valid once the packages version above is stored" in out, out
        assert "-> inline" in out and "checked only" in out and vs() == v0, out
        assert ADMIN not in out, "the token is never printed"
        rc, out = run(*common, "--apply")
        assert rc == 0, out
        assert vs() == (v0[0] + 1, v0[1] + 1, v0[2] + 1), (vs(), v0)
        doc = cond.store.get("packages")[1]
        p = doc["packages"]["laya-playtest"]
        assert p["library"] and p["general"] and p["source"] == "inline" and p["enabled"], p
        text, err = cond.library.text(doc, "laya-playtest")
        assert err is None and text == ("# Laya plays your build\n\nFetch the kit, expose the "
                                        "state, run 3 episodes."), text
        assert doc["packages"]["laya-testing"]["body"].endswith("(laya-playtest).")
        assert doc["packages"]["laya-testing"]["max_tokens"] == 1200, "other fields kept"
        assert doc["packages"]["test-web"]["body"].endswith("Laya plays it.")
        for c in cond.store.get("router")[1]["classes"]:
            pk = c["policy"].get("packages", [])
            if "laya-testing" in pk:
                assert pk[pk.index("laya-testing") + 1] == "laya-playtest", (c["id"], pk)
            else:
                assert "laya-playtest" not in pk
        assert cond.store.get("profiles", "coding")[1]["ui"]["states_channel"] == "inline"
        # the reference is read on request, through the library route
        import json
        import urllib.request
        with urllib.request.urlopen(base + "/v1/library/laya-playtest", timeout=10) as r:
            got = json.load(r)
        assert got["reference"].startswith("## Reference") and "tandem_state" in got["reference"]
        # again: nothing to change, nothing stored
        rc, out = run(*common, "--apply")
        assert rc == 0 and out.count("nothing to change") == 3 and vs() == (v0[0] + 1, v0[1] + 1,
                                                                             v0[2] + 1), out
        # a core the server would cut stops everything
        rc, out = run("--tree", tree(PLAYTEST.replace("run 3 episodes.", "word " * 400)),
                      "--base", base, "--packages", "--apply")
        assert rc != 0 and "would be cut" in out and vs()[0] == v0[0] + 1, out
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
