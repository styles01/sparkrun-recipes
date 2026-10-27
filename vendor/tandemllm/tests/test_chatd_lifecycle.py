"""chatd starts and stops with the engine (OPS-155, QA OPS-156; GitHub issue #3).

ops/start.sh, ops/stop.sh and ops/watchdog.sh run for real in tests/test_ops_scripts.py's sandbox
(fake curl, a fake process table, a fake engine), with a stand-in ops/chatd.sh that keeps its state
in a file and writes every call to the markers. bin/tandem is the one install.sh writes: its
`write_control` is taken from install.sh and run against stand-in start.sh, stop.sh and chatd.sh.
No process of the box is seen or signalled; ops/chatd.sh itself is tests/test_chatd_sh.py's.

The test matrix, the Gherkin of OPS-155 (the scratch install on the Spark covers the rest: a fresh
install's 200, the run that ends `interrupted`, the cron boot, the apply job, systemd, uninstall):

  1  start.sh starts chatd after /health, and the start that finds a healthy engine starts it too
                                                         test_start_starts_chatd_after_health
  2  setup mode does not start chatd                     test_setup_mode_does_not_start_chatd
  3  a hold's restart (HOLD_RESTART=1) does not start it test_a_holds_restart_does_not_start_chatd
  4  QSE_CHATD=off: not started, said so                 test_the_opt_out
  5  a chatd that cannot start: a warning, the engine start still exits 0
                                                         test_a_failing_chatd_never_fails_the_engine_start
  6  stop.sh never touches chatd (watchdog, holds, the setup guide's switches call it)
                                                         test_stop_sh_leaves_chatd_alone
  7  the watchdog starts a dead chatd, at most 3 times an hour, logs it, never restarts the engine
     for it; nothing in setup mode or with QSE_CHATD=off  test_the_watchdog_starts_a_dead_chatd
  8  tandem start | stop | restart | status | chat ...   test_tandem_start_stop_restart_status_and_chat
  9  the systemd unit runs tandem start and tandem stop  test_the_systemd_unit_stops_both
  10 the install's last message, the setup message and the re-run message
                                                         test_the_install_says_what_it_did

Run: python tests/test_chatd_lifecycle.py
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tests"))

import test_ops_scripts as T  # noqa: E402

# ops/chatd.sh's stand-in: state in $CHAT_STATE (a file), every call in the markers, with whether
# the engine answered at that moment (the fake engine touches <repo>/up)
FAKE_CHATD = """#!/bin/bash
up=no; [ -f "{repo}/up" ] && up=yes
echo "chatd.sh $1 engine-up=$up" >> "$MARKERS"
case "$1" in
  start) [ "${{FAKE_CHATD_FAIL:-0}}" = 1 ] && {{ echo "[chatd] FAILED: port 8016 is taken"; exit 1; }}
         touch "$CHAT_STATE"; echo "[chatd] started" ;;
  stop) rm -f "$CHAT_STATE"; echo "[chatd] stopped" ;;
  restart) touch "$CHAT_STATE"; echo "[chatd] stopped"; echo "[chatd] started" ;;
  status) [ -f "$CHAT_STATE" ] && {{ echo "[chatd] running: pid 4242, http://127.0.0.1:8016, chatd 0.1.0"; exit 0; }}
          echo "[chatd] not running"; exit 1 ;;
esac
"""

ENGINE_ENV = ("SERVED_MODEL=t\nMAX_LEN=64\nDEFAULT_MAX_TOKENS=8\nREASONING_FORMAT=tags\n"
              "REASONING_EFFORT=medium\nLEN_FIXED=0\nLEN_LATCH=1\nBUDGET=16\nCORPUS=x\n"
              "CKPT8=x\nCKPT16=x\nNV=x\nHEAD=x\nCACHE_GB=0\nREQUEST_TIMEOUT=1\n"
              "MAX_QUEUE=1\nQUEUE_TIMEOUT=1\nFUSE_PROJ=0\nQWEN38_DF2_TREE_MODE=paths\n"
              "QWEN38_TREE_ALIAS_STATE=0\n")


def _write(path: str, text: str, mode: int = 0o755) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(text)
    os.chmod(path, mode)


def _chat_box(scripts: list[str]) -> str:
    repo = T._box(scripts)
    _write(os.path.join(repo, "ops", "chatd.sh"), FAKE_CHATD.format(repo=repo))
    if not shutil.which("setsid"):                 # macOS has none; start.sh launches with it
        _write(os.path.join(repo, "bin", "setsid"), '#!/bin/bash\nexec "$@"\n')
    return repo


def _env(repo: str, **env) -> dict:
    return T._env(repo, CHAT_STATE=os.path.join(repo, "chat-up"), **env)


def _marks(repo: str) -> list[str]:
    p = os.path.join(repo, "markers")
    return open(p).read().splitlines() if os.path.exists(p) else []


def _chat_calls(repo: str) -> list[str]:
    return [m for m in _marks(repo) if m.startswith("chatd.sh")]


def _start(repo: str, *args, extra: str = "", **env) -> subprocess.CompletedProcess:
    """start.sh with a fake engine that answers /health once it runs."""
    engine = os.path.join(repo, "fake-engine.sh")
    _write(engine, f"#!/bin/bash\necho \"engine $*\" >> {repo}/markers\ntouch {repo}/up\nsleep 30\n")
    env_file = os.path.join(repo, "ops", "serve.env")
    body = open(env_file).read()
    with open(env_file, "w") as fh:
        fh.write(body.replace("PY=/bin/false", f"PY={engine}").replace("PORT=8000", "PORT=18997")
                 + ENGINE_ENV + extra)
    _write(os.path.join(repo, "bin", "curl"), f"#!/bin/bash\n[ -f {repo}/up ] || exit 22\nexit 0\n")
    try:
        return subprocess.run(["bash", os.path.join(repo, "ops", "start.sh"), *args],
                              capture_output=True, text=True, env=_env(repo, **env), timeout=120)
    finally:
        subprocess.run(["pkill", "-f", engine])


def test_start_starts_chatd_after_health():
    repo = _chat_box(["start.sh"])
    r = _start(repo)
    assert r.returncode == 0, r.stdout + r.stderr
    assert _chat_calls(repo) == ["chatd.sh status engine-up=yes", "chatd.sh start engine-up=yes"], _marks(repo)
    assert "[chatd] started" in r.stdout
    marks = _marks(repo)
    assert marks.index("chatd.sh start engine-up=yes") > [i for i, m in enumerate(marks) if m.startswith("engine ")][0]
    # a start that finds the engine healthy starts a chatd that is not running, and leaves a running one
    os.remove(os.path.join(repo, "markers"))
    os.remove(os.path.join(repo, "chat-up"))
    r = T._run(repo, "start.sh", FAKE_HEALTHY="1", CHAT_STATE=os.path.join(repo, "chat-up"))
    assert r.returncode == 0 and "already healthy" in r.stdout, r.stdout + r.stderr
    assert _chat_calls(repo)[-1].startswith("chatd.sh start"), _marks(repo)
    os.remove(os.path.join(repo, "markers"))
    r = T._run(repo, "start.sh", FAKE_HEALTHY="1", CHAT_STATE=os.path.join(repo, "chat-up"))
    assert [c.split()[1] for c in _chat_calls(repo)] == ["status"], _marks(repo)


def test_setup_mode_does_not_start_chatd():
    repo = _chat_box(["start.sh"])
    r = _start(repo, "--setup")
    assert r.returncode == 0 and "setup mode answers" in r.stdout, r.stdout + r.stderr
    assert _chat_calls(repo) == [], _marks(repo)
    # the marker the installer writes: a healthy setup-mode server is left without chatd too
    os.makedirs(os.path.join(repo, "run"), exist_ok=True)
    open(os.path.join(repo, "run", "setup-pending"), "w").write("HOST=127.0.0.1\n")
    r = T._run(repo, "start.sh", FAKE_HEALTHY="1", CHAT_STATE=os.path.join(repo, "chat-up"))
    assert r.returncode == 0 and _chat_calls(repo) == [], _marks(repo)
    # the setup guide's apply job removes the marker and runs start.sh (use-profile.sh): chatd starts
    os.remove(os.path.join(repo, "run", "setup-pending"))
    r = T._run(repo, "start.sh", FAKE_HEALTHY="1", CHAT_STATE=os.path.join(repo, "chat-up"))
    assert _chat_calls(repo)[-1].startswith("chatd.sh start"), _marks(repo)


def test_a_holds_restart_does_not_start_chatd():
    repo = _chat_box(["start.sh"])
    r = _start(repo, HOLD_RESTART="1")
    assert r.returncode == 0, r.stdout + r.stderr
    assert _chat_calls(repo) == [], _marks(repo)


def test_the_opt_out():
    repo = _chat_box(["start.sh"])
    r = _start(repo, extra="QSE_CHATD=off\n")
    assert r.returncode == 0 and "[start] chatd: off (QSE_CHATD=off)" in r.stdout, r.stdout
    assert _chat_calls(repo) == [], _marks(repo)


def test_a_failing_chatd_never_fails_the_engine_start():
    repo = _chat_box(["start.sh"])
    r = _start(repo, FAKE_CHATD_FAIL="1")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "port 8016 is taken" in r.stdout and "[start] WARNING: chatd did not start" in r.stdout, r.stdout
    assert "logs/chatd.log" in r.stdout


def test_stop_sh_leaves_chatd_alone():
    repo = _chat_box(["stop.sh"])
    pid = T._sleeper()
    T._proc(repo, pid, T._engine_argv(8000))
    open(os.path.join(repo, "chat-up"), "w").close()
    try:
        r = T._run(repo, "stop.sh", "5", CHAT_STATE=os.path.join(repo, "chat-up"))
        assert r.returncode == 0 and "stopped" in r.stdout, r.stdout + r.stderr
        assert _chat_calls(repo) == [] and os.path.exists(os.path.join(repo, "chat-up"))
        assert "chatd" not in open(os.path.join(ROOT, "ops", "stop.sh")).read()
    finally:
        if T._alive(pid):
            os.kill(pid, 9)


def test_the_watchdog_starts_a_dead_chatd():
    repo = _chat_box(["watchdog.sh"])
    state = os.path.join(repo, "chat-up")

    def wd(**env):
        return T._run(repo, "watchdog.sh", FAKE_HTTP_CODE="200", CHAT_STATE=state, **env)
    assert wd().returncode == 0
    assert _chat_calls(repo)[-1].startswith("chatd.sh start"), _marks(repo)
    assert "chatd was not running; started it (1/3 this hour)" in T._wd_log(repo), T._wd_log(repo)
    # running: nothing, and the count is cleared
    wd()
    assert _chat_calls(repo)[-1].startswith("chatd.sh status")
    assert not os.path.exists(os.path.join(repo, "logs", "watchdog.chatd"))
    # a chatd that does not stay up: three starts this hour, then one "giving up" line
    for _ in range(5):
        os.remove(state) if os.path.exists(state) else None
        wd(FAKE_CHATD_FAIL="1")
    starts = [c for c in _chat_calls(repo) if c.startswith("chatd.sh start")]
    assert len(starts) == 1 + 3, _marks(repo)
    log = T._wd_log(repo)
    assert log.count("did not start (") == 3 and log.count("giving up until the next hour") == 1, log
    assert not [m for m in _marks(repo) if m.startswith(("stop.sh", "start.sh"))], _marks(repo)
    # setup mode and the opt-out: nothing
    os.remove(os.path.join(repo, "logs", "watchdog.chatd"))
    n = len(_chat_calls(repo))
    os.makedirs(os.path.join(repo, "run"))
    open(os.path.join(repo, "run", "setup-pending"), "w").close()
    wd()
    os.remove(os.path.join(repo, "run", "setup-pending"))
    with open(os.path.join(repo, "ops", "serve.env"), "a") as fh:
        fh.write("QSE_CHATD=off\n")
    wd()
    assert len(_chat_calls(repo)) == n, _marks(repo)
    # an engine that does not answer is the engine's business: no chatd call
    T._run(repo, "watchdog.sh", FAKE_HTTP_CODE="000", CHAT_STATE=state)
    assert len(_chat_calls(repo)) == n


# ------------------------------------------------------------------ bin/tandem (install.sh's write_control)
def _tandem() -> tuple[str, str, dict]:
    """A TANDEM_DIR with bin/tandem as install.sh writes it, stand-in scripts beside it."""
    d = tempfile.mkdtemp(prefix="tandem-ctl-")
    src, run = os.path.join(d, "src"), os.path.join(d, "run")
    marks = os.path.join(d, "markers")
    for name in ("start.sh", "stop.sh"):
        _write(os.path.join(run, "ops", name),
               f'#!/bin/bash\necho "{name} $*" >> "$MARKERS"\n'
               + (f"touch {d}/up\n" if name == "start.sh" else f"rm -f {d}/up\n"))
    _write(os.path.join(src, "ops", "chatd.sh"), FAKE_CHATD.format(repo=d))
    _write(os.path.join(d, "py"), f'#!/bin/bash\necho "py $*" >> "$MARKERS"\n')
    _write(os.path.join(d, "fakebin", "curl"), f"#!/bin/bash\n[ -f {d}/up ] || exit 7\necho '{{}}'\n")
    _write(os.path.join(run, "ops", "serve.env"),
           f"REPO={src}\nPY={d}/py\nPORT=18995\nNV=a,b\nHEAD=h\nCKPT8=c\nCKPT16=c\nQWEN38_MODEL=m\n"
           f"SERVED_MODEL=t\nif [ -f {run}/local.env ]; then . {run}/local.env; fi\n", 0o644)
    text = open(os.path.join(ROOT, "install.sh")).read()
    fn = text[text.index("write_control() {"):text.index("\nCTL\n}\n") + len("\nCTL\n}\n")]
    script = subprocess.run(["bash", "-c", fn + '\nDIR="$1"; write_control', "-", d], capture_output=True,
                            text=True, check=True).stdout
    _write(os.path.join(d, "bin", "tandem"), script)
    env = dict(os.environ, MARKERS=marks, CHAT_STATE=os.path.join(d, "chat-up"), TANDEM_SKIP_MEMCHECK="1",
               PATH=os.path.join(d, "fakebin") + os.pathsep + os.environ["PATH"])
    return d, marks, env


def _ctl(d: str, env: dict, *args) -> subprocess.CompletedProcess:
    return subprocess.run([os.path.join(d, "bin", "tandem"), *args], capture_output=True, text=True,
                          env=env, timeout=60, stdin=subprocess.DEVNULL)


def _lines(path: str) -> list[str]:
    if not os.path.exists(path):
        return []
    out = open(path).read().splitlines()
    open(path, "w").close()
    return [ln for ln in out if not ln.startswith("py ")]


def test_tandem_start_stop_restart_status_and_chat():
    d, marks, env = _tandem()
    r = _ctl(d, env, "start")
    assert r.returncode == 0 and "[chatd] started" in r.stdout, r.stdout + r.stderr
    assert _lines(marks) == ["start.sh ", "chatd.sh status engine-up=yes", "chatd.sh start engine-up=yes"]
    # healthy already: chatd is started if it is not running, a running one is left alone
    r = _ctl(d, env, "start")
    assert "already healthy" in r.stdout and _lines(marks) == ["chatd.sh status engine-up=yes"]
    # restart: the engine only
    r = _ctl(d, env, "restart")
    assert r.returncode == 0, r.stdout + r.stderr
    assert _lines(marks) == ["stop.sh 60", "start.sh ", "chatd.sh status engine-up=yes"]
    # status: the engine, then one chatd line
    r = _ctl(d, env, "status")
    assert r.returncode == 0 and r.stdout.splitlines()[-1].startswith("[chatd] running: pid 4242"), r.stdout
    _lines(marks)
    # stop: chatd first, then the engine
    r = _ctl(d, env, "stop", "120")
    assert r.returncode == 0 and _lines(marks) == ["chatd.sh stop engine-up=yes", "stop.sh 120"], r.stdout
    r = _ctl(d, env, "status")
    assert r.returncode == 1 and r.stdout.splitlines()[-1] == "[chatd] not running", r.stdout
    _lines(marks)
    # tandem chat ...
    for sub in ("start", "status", "restart", "stop"):
        r = _ctl(d, env, "chat", sub)
        assert r.returncode == 0, (sub, r.stdout + r.stderr)
    assert [m.split()[1] for m in _lines(marks)] == ["start", "status", "restart", "stop"]
    assert _ctl(d, env, "chat", "status").returncode == 1
    r = _ctl(d, env, "chat", "nope")
    assert r.returncode == 2 and "chat start | stop | restart | status | logs" in r.stderr
    r = _ctl(d, env, "help")
    assert "chat start|stop|restart|status|logs" in r.stderr, r.stderr
    # QSE_CHATD=off: start leaves chatd alone, status says chat is off
    _write(os.path.join(d, "run", "local.env"), "QSE_CHATD=off\n", 0o644)
    _lines(marks)
    r = _ctl(d, env, "start")
    assert r.returncode == 0 and _lines(marks) == ["start.sh "], r.stdout
    r = _ctl(d, env, "status")
    assert r.stdout.splitlines()[-1] == "[tandem] chat: off (QSE_CHATD=off)", r.stdout
    os.remove(os.path.join(d, "run", "local.env"))
    # setup mode (the installer's marker): tandem start does not start chatd
    _ctl(d, env, "stop")
    os.makedirs(os.path.join(d, "src", "run"))
    open(os.path.join(d, "src", "run", "setup-pending"), "w").close()
    _lines(marks)
    _ctl(d, env, "start")
    assert not [m for m in _lines(marks) if m.startswith("chatd.sh start")]
    # a chatd that cannot start: a warning, tandem start still exits 0
    os.remove(os.path.join(d, "src", "run", "setup-pending"))
    _ctl(d, env, "stop")
    r = _ctl(d, dict(env, FAKE_CHATD_FAIL="1"), "start")
    assert r.returncode == 0 and "WARNING: chatd did not start" in r.stderr, r.stdout + r.stderr


def test_the_systemd_unit_stops_both():
    text = open(os.path.join(ROOT, "install.sh")).read()
    unit = text[text.index('cat > "$UNIT" <<EOF'):]
    unit = unit[:unit.index("\nEOF\n")]
    assert "ExecStart=/bin/bash $DIR/bin/tandem start" in unit and "ExecStop=/bin/bash $DIR/bin/tandem stop 120" in unit
    assert "systemctl restart restarts chatd" in unit
    # bin/tandem stop stops chatd (test_tandem_...); --uninstall runs it and chatd.sh stop as well
    un = text[text.index('if [ "$UNINSTALL" = 1 ]; then'):text.index("printf '%sTandemLLM installer%s")]
    assert '"$DIR/bin/tandem" stop 120' in un and 'bash "$SRC/ops/chatd.sh" stop' in un


def test_the_install_says_what_it_did():
    text = open(os.path.join(ROOT, "install.sh")).read()
    done = text[text.index("${B}Done.${N}"):]
    assert "Control:   $DIR/bin/tandem start | stop | restart | status | smoke | logs | doctor | chat" in done
    assert re.search(r"Chat: +http://\$API_HOST:\$PORT/dashboard/#/chat +chatd starts and stops with the engine", done)
    assert "($DIR/bin/tandem chat status)" in done and "The sandbox is optional" in done
    assert "the chat answers and uses the file tools; bash $SRC/ops/sandbox/install.sh adds the rest" in done
    setup = text[text.index("no model is loaded yet. Open the Setup Guide"):text.index('if [ "$START" != 1 ]; then\n    info "skipped (--no-start). Start it with')]
    assert "The Chat tab starts with the engine once the setup is done (%s/bin/tandem chat status)" in setup
    assert "chat_note" in setup and done.rstrip().endswith("chat_note")
    assert "chatd was running before this update and keeps the old code until: %s/bin/tandem chat restart" in text


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:60s} ok")
            passed += 1
    print(f"{passed} passed")
