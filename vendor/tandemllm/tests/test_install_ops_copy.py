"""An installed box has the ops scripts start.sh and the documented cron line use (OPS-161, QA OPS-162).

install.sh copies the release's service scripts into DIR/run/ops, beside the serve.env it renders.
roles.sh (start.sh runs it after /health), watchdog.sh (the cron line of docs/operations.md) and
hold.sh were missing, so every start on an installed box printed `roles.sh: No such file or
directory`. The list is checked against every "$HERE/<name>" the copied scripts call or source; the
install checks run install.sh into an empty HOME with tests/test_install_control.py's stand-ins and
stub engine on a free port (Linux only), never the box's engine.

The test matrix, the Gherkin of OPS-161:

  1  the copy list and the sourcing agree: every "$HERE/<name>" that start.sh, stop.sh, roles.sh,
     watchdog.sh, hold.sh and use-profile.sh call is in install.sh's copy list (serve.env is the
     rendered one); the list before the fix misses roles.sh  test_the_copy_list_and_the_calls_agree
  2  docs/operations.md and install.sh's last message name run/ops/watchdog.sh for an installed box
                                                              test_the_docs_name_the_installed_paths
  3  a start on an installed box runs run/ops/roles.sh up: no "No such file or directory", no "a role
     did not load"; with roles.sh removed the old message comes back; roles.sh up alone exits 0
                                                              test_a_start_runs_roles_sh
  4  the documented cron line, run/ops/watchdog.sh: it reads run/ops/serve.env (this box's port and
     logs under DIR/src), leaves a healthy engine alone and restarts a dead one through run/ops
                                                              test_the_watchdog_on_an_installed_box
  5  run/ops/hold.sh: the pause where start.sh and the watchdog look (DIR/src/.watchdog.off), roles.sh
     down before the stop, the restore through run/ops/start.sh
                                                              test_hold_on_an_installed_box
  6  a re-run replaces an older copy, mode 755                test_a_rerun_refreshes_the_copies

Run: python tests/test_install_ops_copy.py   (3 to 6 need Linux: install.sh stops anywhere else)
"""
from __future__ import annotations

import os
import re
import stat
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tests"))

import test_install_control as I  # noqa: E402

CALLERS = ("start.sh", "stop.sh", "roles.sh", "watchdog.sh", "hold.sh", "use-profile.sh")
RENDERED = {"serve.env"}            # install.sh writes it; never copied
BEFORE = ["start.sh", "stop.sh", "engines.sh", "use-profile.sh", "make-secrets.sh", "engine_status.py",
          "fetch_weights.py"]


def copy_list(text: str | None = None) -> list[str]:
    text = text or open(os.path.join(ROOT, "install.sh")).read()
    m = re.search(r'for s in ([^;]*); do install -m 755 "\$SRC/ops/\$s"', text)
    return m.group(1).replace("\\", " ").split()


def here_calls(names) -> dict[str, set[str]]:
    """script -> the files it reaches through "$HERE/<name>" (not "$HERE/.." or deeper paths)."""
    out = {}
    for n in names:
        if not n.endswith(".sh"):
            continue
        text = open(os.path.join(ROOT, "ops", n)).read()
        out[n] = {m for m in re.findall(r'\$(?:\{HERE\}|HERE)/([\w][\w.-]*)(?![\w./-])', text)}
    return out


def missing(listed: list[str]) -> dict[str, set[str]]:
    have = set(listed) | RENDERED
    return {k: v - have for k, v in here_calls(listed).items() if v - have}


def test_the_copy_list_and_the_calls_agree():
    listed = copy_list()
    assert set(CALLERS) <= set(listed), listed
    assert {"roles.sh", "watchdog.sh", "hold.sh"} <= set(listed)
    assert "chatd.sh" not in listed and "gate.sh" not in listed      # documented in install.sh
    assert missing(listed) == {}, missing(listed)
    assert "roles.sh" in here_calls(["start.sh"])["start.sh"]
    for n in listed:
        assert os.path.isfile(os.path.join(ROOT, "ops", n)), n
    # the negative control: the list before OPS-161 misses what start.sh and hold.sh call
    gone = missing(BEFORE)
    assert "roles.sh" in gone.get("start.sh", set()), gone


def test_the_docs_name_the_installed_paths():
    ops = open(os.path.join(ROOT, "docs", "operations.md")).read()
    assert "    * * * * * bash $HOME/TandemLLM/run/ops/watchdog.sh\n" in ops
    for n in copy_list():
        assert f"`{n}`" in ops, n
    text = open(os.path.join(ROOT, "install.sh")).read()
    done = text[text.index("${B}Done.${N}"):]
    assert "Watchdog:  bash $RUN/ops/watchdog.sh" in done
    assert "chatd.sh runs from the checkout" in text and "gate.sh is the release gate" in text


# ---------------------------------------------------------------- an installed box (Linux)
_BOX: dict = {}


def _box() -> dict | None:
    if "b" not in _BOX:
        _BOX["b"] = I.install("ops-copy", False, I.free_port())
    return _BOX["b"]


def _skip() -> bool:
    if _box() is None:
        print("    (skipped: install.sh runs on Linux only -- run it on the box)")
        return True
    return False


def _stop(box: dict, env: dict) -> None:
    p = os.path.join(box["dir"], "src", "logs", "engine.pid")
    if os.path.exists(p):
        try:
            pid = I.adopt(box, env)
        except (OSError, ValueError):
            return
        I.tandem(box, env, "stop", "20")
        if not I._gone(pid):
            os.kill(pid, 9)


def test_a_start_runs_roles_sh():
    if _skip():
        return
    box = _box()
    env = I.stub_engine(box)
    ops = os.path.join(box["dir"], "run", "ops")
    for n in copy_list():
        st = os.stat(os.path.join(ops, n))
        assert stat.S_IMODE(st.st_mode) == 0o755, n
    r = subprocess.run(["bash", os.path.join(ops, "roles.sh"), "up"], capture_output=True, text=True, env=env, timeout=60)
    assert r.returncode == 0 and "[roles] none to load (ROLES_AUTOLOAD is empty)" in r.stdout, r.stdout + r.stderr
    try:
        r = I.tandem(box, env, "start")
        out = r.stdout + r.stderr
        assert r.returncode == 0, out
        assert "No such file or directory" not in out and "a role did not load" not in out, out
        assert "[roles] none to load" in out, out
    finally:
        _stop(box, env)
    # the negative control: start.sh without run/ops/roles.sh has the old message back
    os.rename(os.path.join(ops, "roles.sh"), os.path.join(ops, "roles.sh.away"))
    try:
        r = subprocess.run(["bash", os.path.join(ops, "start.sh")], capture_output=True, text=True, env=env,
                           timeout=120, stdin=subprocess.DEVNULL)
        out = r.stdout + r.stderr
        assert "roles.sh: No such file or directory" in out and "[start] a role did not load" in out, out
    finally:
        _stop(box, env)
    # and tandem start puts it back first (OPS-171: tandem doctor copies a missing run/ops script)
    try:
        r = I.tandem(box, env, "start")
        out = r.stdout + r.stderr
        assert "[doctor] repaired: run/ops/roles.sh was missing: copied from src/ops" in out, out
        assert "No such file or directory" not in out and os.path.exists(os.path.join(ops, "roles.sh")), out
    finally:
        _stop(box, env)
        os.unlink(os.path.join(ops, "roles.sh.away"))


def test_the_watchdog_on_an_installed_box():
    if _skip():
        return
    box = _box()
    env = dict(I.stub_engine(box), RESTART_SETTLE_S="0", WATCHDOG_FAILS="1")
    src = os.path.join(box["dir"], "src")
    wd = os.path.join(box["dir"], "run", "ops", "watchdog.sh")
    log = os.path.join(src, "logs", "watchdog.log")
    try:
        assert I.tandem(box, env, "start").returncode == 0
        I.adopt(box, env)
        # the cron line: no environment of serve.env's, this box's port from run/ops/serve.env
        r = subprocess.run(["bash", wd], capture_output=True, text=True, env=env, timeout=60)
        assert r.returncode == 0, r.stdout + r.stderr
        assert open(os.path.join(src, "logs", "watchdog.fails")).read().strip() == "0"
        assert not os.path.exists(os.path.join(box["home"], "qwen38-spark-engine"))   # the template's REPO
        # a dead engine: restarted through run/ops (stop.sh, start.sh)
        _stop(box, env)
        r = subprocess.run(["bash", wd], capture_output=True, text=True, env=env, timeout=120)
        text = open(log).read()
        assert "health check failed (code 000), 1/1" in text and "restarted" in text, text
        assert I.tandem(box, env, "status").returncode == 0
    finally:
        _stop(box, env)


def test_hold_on_an_installed_box():
    if _skip():
        return
    box = _box()
    env = dict(I.stub_engine(box), HOLD_ENGINE_WAIT="10", HOLD_REFRESH="2")
    src = os.path.join(box["dir"], "src")
    probe = os.path.join(box["home"], "hold-probe")
    try:
        assert I.tandem(box, env, "start").returncode == 0
        I.adopt(box, env)
        cmd = f'[ -f {src}/.watchdog.off ] && echo paused > {probe}; pwd >> {probe}'
        r = subprocess.run(["bash", os.path.join(box["dir"], "run", "ops", "hold.sh"), "1", "--", "bash", "-c", cmd],
                           capture_output=True, text=True, env=env, timeout=180)
        out = r.stdout + r.stderr
        assert r.returncode == 0, out
        assert open(probe).read().splitlines() == ["paused", src], open(probe).read()
        assert "[stop] stopped" in out and "[hold] restarting the service" in out, out
        assert "No such file or directory" not in out, out
        assert not os.path.exists(os.path.join(src, ".watchdog.off"))
        assert not os.path.exists(os.path.join(box["dir"], "run", ".watchdog.off"))
        time.sleep(1)
        assert I.tandem(box, env, "status").returncode == 0, "the hold restored the service"
    finally:
        _stop(box, env)


def test_a_rerun_refreshes_the_copies():
    if _skip():
        return
    box = _box()
    ops = os.path.join(box["dir"], "run", "ops")
    for n in ("roles.sh", "watchdog.sh", "hold.sh"):
        with open(os.path.join(ops, n), "w") as fh:
            fh.write("# an older copy\n")
        os.chmod(os.path.join(ops, n), 0o600)
    # the stub engine is a local change of the checkout, which the installer refuses to update
    subprocess.run(["git", "-C", os.path.join(box["dir"], "src"), "checkout", "--", "server/app.py"], check=True)
    r = subprocess.run(box["args"], env=box["env"], capture_output=True, text=True, timeout=600,
                       stdin=subprocess.DEVNULL)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    for n in copy_list():
        p = os.path.join(ops, n)
        assert open(p, "rb").read() == open(os.path.join(ROOT, "ops", n), "rb").read(), n
        assert stat.S_IMODE(os.stat(p).st_mode) == 0o755, n


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:60s} ok")
            passed += 1
    print(f"{passed} passed")
