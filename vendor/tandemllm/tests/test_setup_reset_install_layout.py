"""The reset levels never remove the secrets file the engine reads (OPS-163, QA OPS-164).

An install keeps its secrets in DIR/state/secrets.env (QSE_SECRETS) and its ledger in
DIR/state/usage (QSE_USAGE_LEDGER); install.sh moves nothing else, so the caches, the conductor and
the memory notes stay under ~/.qwen38-spark-engine on an install too. The reset's guard named only
~/.qwen38-spark-engine/secrets.env. tests/test_setup_reset.py's box (default layout) with an
install-style DIR/state under the same temporary HOME, on the CPU; never the box's own files.

The test matrix, the Gherkin of OPS-163:

  1  protected() holds the real path of QSE_SECRETS's file and of its folder at every level; the
     secrets file follows ops/start.sh's order (QSE_SECRETS, $QSE_STATE_DIR/secrets.env, default)
                                                              test_the_guard_names_the_installs_secrets
  2  an item at DIR/state or a parent of it is refused "it holds something that is never removed"
     at levels 2 to 5; the guard before the fix let it pass  test_an_item_holding_the_secrets_is_refused
  3  every level run with its confirmation: DIR/state/secrets.env and ~/.qwen38-spark-engine/secrets.env
     keep their hashes; the ledger in DIR/state/usage and a conductor folder under DIR/state stay at
     levels 3 and 4 and go at level 5                         test_every_level_keeps_both_secrets_files
  4  the items an install's levels 3 and 5 look in: the ledger from serve.env (DIR/state/usage),
     the conductor, notes and caches under ~/.qwen38-spark-engine, where the template puts them
                                                              test_the_items_follow_the_install
  5  a symbolic link at DIR/state/secrets.env: the link and its target are guarded and both survive
                                                              test_a_symlinked_secrets_file
  6  no QSE_SECRETS: ~/.qwen38-spark-engine/secrets.env is guarded as before
                                                              test_the_default_layout_is_unchanged

Run: python tests/test_setup_reset_install_layout.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_setup_apply import Box, _api, _smoke_ok, Recorder, _sel  # noqa: E402
from test_setup_reset import _hash, _home, _put, _setup_app, M  # noqa: E402

from server.setup import apply, reset  # noqa: E402

WORDS = {3: "delete caches", 4: "delete weights", 5: "delete history"}


def _install(box: Box, conductor: bool = False) -> dict:
    """DIR/state as install.sh lays it out, QSE_SECRETS and QSE_USAGE_LEDGER in serve.env and in
    this process's environment (start.sh exports serve.env to the server)."""
    d = os.path.join(box.home, "TandemLLM")
    p = {"dir": d, "state": os.path.join(d, "state"), "secrets": os.path.join(d, "state", "secrets.env"),
         "ledger": os.path.join(d, "state", "usage", "ledger.sqlite3"),
         "conductor": os.path.join(d, "state", "conductor")}
    _put(p["secrets"], 300)
    _put(p["ledger"], 4000)
    lines = f"QSE_SECRETS='{p['secrets']}'\nQSE_USAGE_LEDGER='{p['ledger']}'\n"
    if conductor:
        _put(os.path.join(p["conductor"], "router", "v1.json"), 500)
        lines += f"QSE_CONDUCTOR_CONFIG='{p['conductor']}'\n"
    with open(os.path.join(box.ops, "serve.env"), "a") as fh:
        fh.write(lines)
    os.environ["QSE_SECRETS"] = p["secrets"]
    return p


def test_the_guard_names_the_installs_secrets():
    with Box() as box:
        p = _install(box)
        real = os.path.realpath
        for n in (2, 3, 4, 5):
            guard = reset.protected(M, None, n)
            assert real(p["secrets"]) in guard and real(p["state"]) in guard, n
            assert real(os.path.join(box.home, ".qwen38-spark-engine", "secrets.env")) in guard
        assert real(os.path.dirname(p["ledger"])) in reset.protected(M, None, 3)
        assert real(os.path.dirname(p["ledger"])) not in reset.protected(M, None, 5)
        # the secrets file, in ops/start.sh's order
        assert apply.secrets_file() == p["secrets"]
        os.environ.pop("QSE_SECRETS")
        os.environ["QSE_STATE_DIR"] = os.path.join(box.home, "st")
        try:
            assert apply.secrets_file() == os.path.join(box.home, "st", "secrets.env")
            os.environ.pop("QSE_STATE_DIR")
            assert apply.secrets_file() == os.path.join(box.home, ".qwen38-spark-engine", "secrets.env")
        finally:
            os.environ.pop("QSE_STATE_DIR", None)
        # serve.env's QSE_SECRETS is guarded even when this process's environment lacks it
        assert real(p["secrets"]) in reset.protected(M, None, 5)


def test_an_item_holding_the_secrets_is_refused():
    with Box() as box:
        p = _install(box)
        for n in (2, 3, 4, 5):
            guard = reset.protected(M, None, n)
            for item in (p["state"], p["dir"], p["secrets"]):
                assert reset._allowed(item, guard, n) == "it holds something that is never removed", (n, item)
            # the ledger at level 5 sits inside the install's state folder and may go
        assert reset._allowed(p["ledger"], reset.protected(M, None, 5), 5) is None
        assert reset._allowed(p["ledger"], reset.protected(M, None, 3), 3) is not None
        # the negative control: the guard as it was (the default state folder only) lets them pass
        saved = reset.secrets_files, reset.state_dirs, reset._env
        reset.secrets_files = lambda: [os.path.join(reset.state_dir(), "secrets.env")]
        reset.state_dirs = lambda: [os.path.realpath(reset.state_dir())]
        reset._env = lambda keys: {}
        try:
            guard = reset.protected(M, None, 5)
            assert reset._allowed(p["state"], guard, 5) is None
            assert reset._allowed(p["secrets"], guard, 5) is None
        finally:
            reset.secrets_files, reset.state_dirs, reset._env = saved


def test_every_level_keeps_both_secrets_files():
    with Box() as box:
        home = _home(box)                                      # ~/.qwen38-spark-engine/secrets.env too
        p = _install(box, conductor=True)
        _setup_app()
        both = [p["secrets"], home["secrets"]]
        keep = _hash(both)
        # level 2: a job of the apply runner (test_setup_reset.test_level_2)
        apply.write_overlay(M, _sel(slots=2))
        real = apply.spawn
        apply.spawn = lambda jid: 1
        try:
            code, b = _api("POST", "/v1/setup/reset", {"level": 2, "confirm": True})
        finally:
            apply.spawn = real
        assert code == 202, b
        assert apply.Runner(b["job_id"], cmd=Recorder(), http=_smoke_ok("exl3-4.5"), ops=box.ops).main() == 0
        assert _hash(both) == keep
        for n in (3, 4, 5):
            code, b = _api("POST", "/v1/setup/reset", {"level": n, "confirm": WORDS[n]})
            assert code == 200, (n, b)
            assert _hash(both) == keep, n
            if n < 5:
                assert os.path.exists(p["ledger"]) and os.path.isdir(p["conductor"]), n
        assert b["ok"], b
        assert not os.path.exists(p["ledger"]) and not os.path.exists(p["conductor"])
        assert os.path.isfile(p["secrets"]) and os.path.isfile(home["secrets"])


def test_the_items_follow_the_install():
    with Box() as box:
        _home(box)
        p = _install(box)
        _setup_app()
        code, b = _api("GET", "/v1/setup/reset")
        assert code == 200, b
        lv = {x["level"]: x for x in b["levels"]}
        l5 = [i["path"] for i in lv[5]["items"]]
        assert l5 == ["~/TandemLLM/state/usage/ledger.sqlite3", "~/.qwen38-spark-engine/conductor",
                      "~/.qwen38-spark-engine/memory"], l5
        assert not any(i.get("refused") for i in lv[5]["items"]), lv[5]["items"]
        l3 = [i["path"] for i in lv[3]["items"]]
        assert l3 and all(x.startswith("~/.qwen38-spark-engine/") for x in l3), l3
        assert "secrets" not in str([lv[n]["items"] for n in (2, 3, 4, 5)])
        assert os.path.isfile(p["secrets"])


def test_a_symlinked_secrets_file():
    with Box() as box:
        _home(box)
        p = _install(box)
        target = os.path.join(box.home, "vault", "secrets.env")
        _put(target, 200)
        os.remove(p["secrets"])
        os.symlink(target, p["secrets"])
        _setup_app()
        guard = reset.protected(M, None, 5)
        assert os.path.realpath(target) in guard
        assert reset._allowed(p["secrets"], guard, 5) == "it holds something that is never removed"
        keep = _hash([target])
        code, b = _api("POST", "/v1/setup/reset", {"level": 5, "confirm": WORDS[5]})
        assert code == 200 and b["ok"], b
        assert os.path.islink(p["secrets"]) and _hash([target]) == keep


def test_the_default_layout_is_unchanged():
    with Box() as box:
        home = _home(box)
        os.environ.pop("QSE_SECRETS", None)
        _setup_app()
        for n in (2, 3, 4, 5):
            guard = reset.protected(M, None, n)
            assert os.path.realpath(home["secrets"]) in guard, n
        assert reset.state_dirs() == [os.path.realpath(os.path.join(box.home, ".qwen38-spark-engine"))]
        keep = _hash([home["secrets"]])
        code, b = _api("POST", "/v1/setup/reset", {"level": 5, "confirm": WORDS[5]})
        assert code == 200 and b["ok"] and _hash([home["secrets"]]) == keep, b


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:60s} ok")
            passed += 1
    print(f"{passed} passed")
