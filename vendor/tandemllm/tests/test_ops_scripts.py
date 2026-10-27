"""The two supervisor scripts, run for real against fake `curl`/`pgrep` (OPS, the pre-merge review).

Both faults here are the same shape: a file that outlives the thing it describes. `logs/loading.since`
outlived the loader whose load it was timing, and `.watchdog.off` outlived the hold that armed it --
and in both cases the survivor condemns whatever comes next, which on this box means a restart in
the middle of a 29 GB load, or no service at all until a timeout that is measured in tens of minutes.

`ops/watchdog.sh` and `ops/start.sh` are copied into a temporary directory beside a `serve.env` that
points at it, so the real scripts run with nothing real behind them: `curl`, `pgrep` and (on macOS,
where `stat -c` does not exist) `stat` come from a fake bin on the PATH, and `stop.sh`/`start.sh`
leave a marker instead of touching a board.

The process table is a fake too: `ops/engines.sh` reads `$ENGINE_PROC/<pid>/cmdline` and
`/exe`, and the sandbox points it at a directory of its own, so no test ever sees -- or signals --
the box's real engine. The fake `pgrep -f` searches the same table, which is how the scripts found
engines before.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time

OPS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ops")

SERVE_ENV = """HOME_DIR={repo}
REPO={repo}
PY=/bin/false
PYTHONPATH={repo}
HOST=127.0.0.1
PORT=8000
WATCHDOG_FAILS=3
LOADING_MAX=900
WATCHDOG_PAUSE_MAX=2400
ENGINE_PROC={repo}/proc
"""

FAKE_CURL = """#!/bin/bash
# `-w '%{http_code}'` asks for the code on stdout -- no newline, and when nothing answers the real
# curl still prints 000 and exits 7; `-sf` asks for an exit status.
case " $* " in
  *" -w "*) printf '%s' "${FAKE_HTTP_CODE:-000}"; [ "${FAKE_HTTP_CODE:-000}" = "000" ] && exit 7;;
  *) [ "${FAKE_HEALTHY:-0}" = "1" ] || exit 22;;
esac
exit 0
"""

FAKE_PGREP = """#!/bin/bash
# `pgrep -f PATTERN` over the sandbox's process table: every pid whose command line matches.
found=1
for d in "$ENGINE_PROC"/[0-9]*; do
  [ -f "$d/cmdline" ] || continue
  if tr '\\0' ' ' < "$d/cmdline" | grep -qE -- "${!#}"; then echo "${d##*/}"; found=0; fi
done
exit $found
"""

FAKE_STAT = """#!/bin/bash
# GNU `stat -c %Y` on a box that only has the BSD one.
[ "$1" = "-c" ] && exec /usr/bin/stat -f %m "$3"
exec /usr/bin/stat "$@"
"""

MARKER = """#!/bin/bash
echo "$(basename "$0") $*" >> "$MARKERS"
"""


def _box(scripts: list[str]) -> str:
    """A temporary repo with the real scripts, a serve.env pointing at it, and fakes on the PATH."""
    repo = tempfile.mkdtemp(prefix="opsbox-")
    ops = os.path.join(repo, "ops")
    os.makedirs(os.path.join(repo, "logs"))
    os.makedirs(ops)
    os.makedirs(os.path.join(repo, "proc"))
    for name in scripts + ["engines.sh", "roles.sh"]:
        shutil.copy(os.path.join(OPS, name), os.path.join(ops, name))
    with open(os.path.join(ops, "serve.env"), "w") as fh:
        fh.write(SERVE_ENV.format(repo=repo))
    binary = os.path.join(repo, "bin")
    os.makedirs(binary)
    fakes = {"curl": FAKE_CURL, "pgrep": FAKE_PGREP, "stop.sh": MARKER, "start.sh": MARKER}
    if sys.platform == "darwin":
        fakes["stat"] = FAKE_STAT
    for name, body in fakes.items():
        path = os.path.join(ops if name.endswith(".sh") else binary, name)
        if os.path.exists(path):
            continue                       # the script under test, not a fake
        with open(path, "w") as fh:
            fh.write(body)
        os.chmod(path, 0o755)
    return repo


def _env(repo: str, **env) -> dict:
    e = dict(os.environ)
    e["PATH"] = os.path.join(repo, "bin") + os.pathsep + e["PATH"]
    e["MARKERS"] = os.path.join(repo, "markers")
    e["ENGINE_PROC"] = os.path.join(repo, "proc")
    e["BOX_LOCK"] = os.path.join(repo, "box.flock")    # never the box's real lock
    e["RESTART_SETTLE_S"] = "0"                        # OPS-41's settle, on a test's clock
    e.update({k: str(v) for k, v in env.items()})
    return e


def _run(repo: str, script: str, *args, **env) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", os.path.join(repo, "ops", script), *args],
                          capture_output=True, text=True, env=_env(repo, **env))


VENV_PY = "/home/u/.venv/bin/python"
PY_EXE = "/home/u/.local/share/uv/python/cpython-3.11.16/bin/python3.11"


def _engine_argv(port: int | None = 8000) -> list[str]:
    """What start.sh, the systemd unit and row3 launch."""
    return ([VENV_PY, "-u", "server/app.py", "--host", "127.0.0.1"]
            + (["--port", str(port)] if port is not None else []) + ["--max-len", "262144"])


def _proc(repo: str, pid: int, argv: list[str], exe: str = PY_EXE) -> None:
    """One entry of the sandbox's process table: `cmdline` NUL-separated, `exe` a link."""
    d = os.path.join(repo, "proc", str(pid))
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "cmdline"), "wb") as fh:
        fh.write(b"".join(a.encode() + b"\0" for a in argv))
    if exe:
        os.symlink(exe, os.path.join(d, "exe"))


def _sleeper() -> int:
    """A real process to be signalled -- what the table says it is, is up to the test. Detached,
    as start.sh's engine is: a child of this test would linger as a zombie after the signal, and
    `kill -0` in stop.sh would call it alive."""
    return int(subprocess.run(["bash", "-c", "sleep 300 >/dev/null 2>&1 & echo $!"],
                              capture_output=True, text=True).stdout)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _since(repo: str) -> str:
    path = os.path.join(repo, "logs", "loading.since")
    return open(path).read().strip() if os.path.exists(path) else ""


def _wd_log(repo: str) -> str:
    path = os.path.join(repo, "logs", "watchdog.log")
    return open(path).read() if os.path.exists(path) else ""


def test_a_dead_loaders_timestamp_does_not_condemn_the_next_one():
    # The fault: `loading.since` was removed only on a 200. A loader that died before it ever
    # answered left its start time behind, the next one inherited an age past LOADING_MAX on its
    # very first check, and three strikes later the watchdog restarted a server mid-load.
    repo = _box(["watchdog.sh"])
    stale = int(time.time()) - 5000
    with open(os.path.join(repo, "logs", "loading.since"), "w") as fh:
        fh.write(f"111 {stale}\n")
    _proc(repo, 222, _engine_argv())
    r = _run(repo, "watchdog.sh")
    assert r.returncode == 0, r.stderr
    assert "loading" in _wd_log(repo) and "treating as hung" not in _wd_log(repo)
    pid, since = _since(repo).split()
    assert pid == "222" and int(since) > stale, "the new loader times its own load"
    assert not os.path.exists(os.path.join(repo, "markers")), "nothing was restarted"


def test_the_same_loader_keeps_its_first_attempt():
    # The reset is on the pid, not on every check: a loader that has been silent for ten minutes
    # must still be counted from when IT started, or LOADING_MAX never elapses.
    repo = _box(["watchdog.sh"])
    started = int(time.time()) - 300
    with open(os.path.join(repo, "logs", "loading.since"), "w") as fh:
        fh.write(f"222 {started}\n")
    _proc(repo, 222, _engine_argv())
    assert _run(repo, "watchdog.sh").returncode == 0
    assert _since(repo) == f"222 {started}"
    assert "loading (code 000, 3" in _wd_log(repo), _wd_log(repo)


def test_the_restart_path_clears_the_timestamp():
    repo = _box(["watchdog.sh"])
    with open(os.path.join(repo, "logs", "loading.since"), "w") as fh:
        fh.write(f"222 {int(time.time()) - 5000}\n")
    with open(os.path.join(repo, "logs", "watchdog.fails"), "w") as fh:
        fh.write("2\n")                                   # two strikes already
    _proc(repo, 222, _engine_argv())
    r = _run(repo, "watchdog.sh")
    assert r.returncode == 0, r.stderr
    markers = open(os.path.join(repo, "markers")).read()
    assert "stop.sh" in markers and "start.sh" in markers
    assert _since(repo) == "", "the restarted engine must not inherit the hung one's clock"


def test_a_pause_file_that_outlived_its_hold_does_not_block_the_reboot_start():
    # `.watchdog.off` cannot survive a hold, but it survives a reboot, and the @reboot line then
    # refused to start the service for up to WATCHDOG_PAUSE_MAX with nobody holding anything.
    repo = _box(["start.sh"])
    pause = os.path.join(repo, ".watchdog.off")
    with open(pause, "w") as fh:
        fh.write("bench: row3\n")
    old = time.time() - 3000
    os.utime(pause, (old, old))
    r = _run(repo, "start.sh", FAKE_HEALTHY="1")
    assert r.returncode == 0, r.stderr + r.stdout
    assert "older than the longest hold" in r.stdout, r.stdout
    assert not os.path.exists(pause), "the stale pause file is removed, not stepped around"


def test_a_live_hold_still_refuses_and_says_so_with_a_non_zero_status():
    repo = _box(["start.sh"])
    pause = os.path.join(repo, ".watchdog.off")
    with open(pause, "w") as fh:
        fh.write("bench: row3\n")
    r = _run(repo, "start.sh", FAKE_HEALTHY="1")
    assert r.returncode != 0, "a refusal cron cannot see is a refusal nobody notices"
    assert "a hold owns the board" in r.stdout
    assert os.path.exists(pause), "a live hold's file is left alone"


def test_a_hold_longer_than_the_pause_limit_keeps_the_watchdog_out():
    """The watchdog ignores a pause file older than WATCHDOG_PAUSE_MAX (2400 s), and on
    2026-09-18 a hand-made hold outlived it: :8000 came back beside a test server mid-gate.
    `ops/hold.sh`'s keeper refreshes the file for the life of the hold, so the limit never bites
    a live hold. The same scripts on a compressed clock: pause limit 3 s, refresh 1 s, a held
    command that runs 6 s and asks the watchdog -- with the service silent and one strike to
    restart -- at the end of it. The watchdog must stay out."""
    repo = _box(["hold.sh", "watchdog.sh"])
    env_file = os.path.join(repo, "ops", "serve.env")
    with open(env_file) as fh:
        body = fh.read()
    with open(env_file, "w") as fh:          # serve.env is sourced, so the knobs live there
        fh.write(body.replace("WATCHDOG_PAUSE_MAX=2400", "WATCHDOG_PAUSE_MAX=3")
                     .replace("WATCHDOG_FAILS=3", "WATCHDOG_FAILS=1"))
    probe = os.path.join(repo, "probe.sh")
    with open(probe, "w") as fh:
        fh.write(f"sleep 6\nbash {repo}/ops/watchdog.sh\n")
    r = subprocess.run(
        ["bash", os.path.join(repo, "ops", "hold.sh"), "1", "--", "bash", probe],
        capture_output=True, text=True,
        env=_env(repo, HOLD_REFRESH=1, FAKE_HTTP_CODE="000"))
    assert r.returncode == 0, r.stdout + r.stderr
    log = _wd_log(repo)
    assert "ignoring it" not in log and "restarting" not in log, log
    markers = open(os.path.join(repo, "markers")).read().split("\n")
    assert [m.split()[0] for m in markers if m] == ["stop.sh", "start.sh"], markers
    assert not os.path.exists(os.path.join(repo, ".watchdog.off")), "the hold removes its pause"


def test_a_pause_nobody_refreshes_still_ages_out():
    """The other half: the limit is the safety for a pause whose hold is gone, and it stays."""
    repo = _box(["watchdog.sh"])
    pause = os.path.join(repo, ".watchdog.off")
    with open(pause, "w") as fh:
        fh.write("hold\n")
    old = time.time() - 3000
    os.utime(pause, (old, old))
    with open(os.path.join(repo, "logs", "watchdog.fails"), "w") as fh:
        fh.write("2\n")                                   # two strikes already
    r = _run(repo, "watchdog.sh", FAKE_HTTP_CODE="000")
    assert r.returncode == 0, r.stderr
    log = _wd_log(repo)
    assert "ignoring it" in log and "restarting" in log, log

def test_the_engine_does_not_inherit_the_box_lock():
    """Holds run as `flock ~/.qwen38-box.flock bash ops/hold.sh...`, and flock hands its
    lock descriptor to the command unless told `-o`. It was inherited all the way down -- hold.sh,
    start.sh, `setsid nohup` -- into the RESTARTED :8000 engine, which then held the box lock for
    its whole life, and the next agent's flock waited on a server that never exits. start.sh must
    launch the engine with nothing open but stdin/stdout/stderr, whatever its caller held."""
    if not os.path.isdir("/proc/self/fd") or shutil.which("flock") is None:
        print("    (skipped: needs /proc and flock -- run it on the box)")
        return
    repo = _box(["start.sh"])
    env_file = os.path.join(repo, "ops", "serve.env")
    engine = os.path.join(repo, "fake-engine.sh")
    with open(engine, "w") as fh:
        fh.write(f"#!/bin/bash\nls -l /proc/$$/fd > {repo}/engine-fds\ntouch {repo}/up\nsleep 30\n")
    os.chmod(engine, 0o755)
    with open(env_file) as fh:
        body = fh.read()
    with open(env_file, "w") as fh:
        fh.write(body.replace("PY=/bin/false", f"PY={engine}").replace("PORT=8000", "PORT=18999"))
        fh.write("SERVED_MODEL=t\nMAX_LEN=64\nDEFAULT_MAX_TOKENS=8\nREASONING_FORMAT=tags\n"
                 "REASONING_EFFORT=medium\nLEN_FIXED=0\nLEN_LATCH=1\nBUDGET=16\nCORPUS=x\n"
                 "CKPT8=x\nCKPT16=x\nNV=x\nHEAD=x\nCACHE_GB=0\nREQUEST_TIMEOUT=1\n"
                 "MAX_QUEUE=1\nQUEUE_TIMEOUT=1\nFUSE_PROJ=0\nQWEN38_DF2_TREE_MODE=paths\n"
                 "QWEN38_TREE_ALIAS_STATE=0\n")
    with open(os.path.join(repo, "bin", "curl"), "w") as fh:     # healthy once the engine is up
        fh.write(f"#!/bin/bash\n[ -f {repo}/up ] || exit 22\nexit 0\n")
    lock = os.path.join(repo, "box.flock")
    e = _env(repo)
    r = subprocess.run(["flock", lock, "bash", os.path.join(repo, "ops", "start.sh")],
                       capture_output=True, text=True, env=e, timeout=120)
    try:
        assert r.returncode == 0, r.stdout + r.stderr
        fds = open(os.path.join(repo, "engine-fds")).read()
        assert "box.flock" not in fds, f"the engine holds the box lock:\n{fds}"
        free = subprocess.run(["flock", "-n", lock, "true"], timeout=10)
        assert free.returncode == 0, "the lock must be free once start.sh has returned"
    finally:
        subprocess.run(["pkill", "-f", engine])

def _start_with_fake_engine(drop: str, extra: str = "",
                           procs=()) -> tuple[subprocess.CompletedProcess, str]:
    repo = _box(["start.sh"])
    for pid, argv, exe in procs:
        _proc(repo, pid, argv, exe)
    env_file = os.path.join(repo, "ops", "serve.env")
    engine = os.path.join(repo, "fake-engine.sh")
    # the same python runs the engine and the page-cache tool: the tool's call is recorded
    with open(engine, "w") as fh:
        fh.write(f"#!/bin/bash\ncase \"$1\" in *drop_page_cache.py) echo \"drop $*\" >> {repo}/markers; "
                 f"echo '[dropcache] 1 files'; exit 0;; esac\necho \"engine $*\" >> {repo}/markers\n"
                 f"touch {repo}/up\nsleep 30\n")
    os.chmod(engine, 0o755)
    with open(env_file) as fh:
        body = fh.read()
    with open(env_file, "w") as fh:
        fh.write(body.replace("PY=/bin/false", f"PY={engine}").replace("PORT=8000", "PORT=18998"))
        fh.write("SERVED_MODEL=t\nMAX_LEN=64\nDEFAULT_MAX_TOKENS=8\nREASONING_FORMAT=tags\n"
                 "REASONING_EFFORT=medium\nLEN_FIXED=0\nLEN_LATCH=1\nBUDGET=16\nCORPUS=x\n"
                 "CKPT8=x\nCKPT16=x\nNV=x\nHEAD=x\nCACHE_GB=0\nREQUEST_TIMEOUT=1\n"
                 "MAX_QUEUE=1\nQUEUE_TIMEOUT=1\nFUSE_PROJ=0\nQWEN38_DF2_TREE_MODE=paths\n"
                 f"QWEN38_TREE_ALIAS_STATE=0\nDROP_PAGE_CACHE={drop}\n{extra}")
    with open(os.path.join(repo, "bin", "curl"), "w") as fh:     # healthy once the engine is up
        fh.write(f"#!/bin/bash\n[ -f {repo}/up ] || exit 22\nexit 0\n")
    os.chmod(os.path.join(repo, "bin", "curl"), 0o755)
    try:
        r = subprocess.run(["bash", os.path.join(repo, "ops", "start.sh")], capture_output=True,
                           text=True, env=_env(repo), timeout=120)
    finally:
        subprocess.run(["pkill", "-f", engine])
    marks = os.path.join(repo, "markers")
    return r, (open(marks).read() if os.path.exists(marks) else "")


def test_the_service_drops_the_weights_page_cache_once_healthy():
    """~52 GB of the board is the page cache of the weight files, which the
    engine never reads again after its load; an 8,192-row prefill then took MemFree to 1 GB and the
    driver logged NV_ERR_NO_MEMORY, and with the cache dropped the same request left 53 GB free and
    logged nothing. start.sh drops it once the service answers, when serve.env says so."""
    r, marks = _start_with_fake_engine("1")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "healthy" in r.stdout and "drop " in marks and "tools/drop_page_cache.py" in marks, (r.stdout, marks)
    assert "[dropcache]" in r.stdout, r.stdout
    r, marks = _start_with_fake_engine("0")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "drop " not in marks, marks


def test_the_sampled_tree_mode_reaches_the_engine():
    """Serve.env's SAMPLED_TREE picks how a sampled request verifies -- det (or the old
    1) the greedy request's tree, mixed the sampled spine; unset, the q-aware chain (no flag)."""
    for value, want in (("det", "--sampled-tree=det"), ("1", "--sampled-tree=det"),
                        ("mixed", "--sampled-tree=mixed"), (None, None)):
        r, marks = _start_with_fake_engine("0", f"SAMPLED_TREE={value}\n" if value else "")
        assert r.returncode == 0, r.stdout + r.stderr
        argv = next(line for line in marks.splitlines() if line.startswith("engine "))
        if want is None:
            assert "--sampled-tree" not in argv, argv
        else:
            assert want in argv.split(), (value, argv)


def test_the_memory_limits_and_the_narrow_arm_reach_the_engine():
    """OPS-47 / SPD-75: each memory key serve.env sets becomes the server's flag; a key it does not
    set is the server's own default; NARROW_ARM=0 is --narrow-arm off and 1 passes nothing."""
    keys = ("MEM_CEILING_GIB=70\nMEM_PEAK_GIB=77\nRESERVE_SLACK_GIB=2\nRESIDENT_GB=2.5\n"
            "RESIDENT_STASH_GB=1.0\nRESIDENT_GUEST_MAX=8192\nRESIDENT_PIN_GB=4.0\n"
            "RESIDENT_PARK_GB=4\nSTORE_ENTRY_GB=1\n")
    r, marks = _start_with_fake_engine("0", keys + "NARROW_ARM=0\n")
    assert r.returncode == 0, r.stdout + r.stderr
    argv = next(line for line in marks.splitlines() if line.startswith("engine ")).split()
    for flag, value in (("--mem-ceiling-gib", "70"), ("--mem-peak-gib", "77"),
                        ("--reserve-slack-gib", "2"), ("--resident-gb", "2.5"),
                        ("--resident-stash-gb", "1.0"), ("--resident-guest-max", "8192"),
                        ("--resident-pin-gb", "4.0"), ("--resident-park-gb", "4"),
                        ("--store-entry-gb", "1"), ("--narrow-arm", "off")):
        assert argv[argv.index(flag) + 1] == value, (flag, argv)
    r, marks = _start_with_fake_engine("0", "NARROW_ARM=1\n")
    argv = next(line for line in marks.splitlines() if line.startswith("engine "))
    assert "--narrow-arm" not in argv and "--mem-ceiling-gib" not in argv, argv


def test_the_disk_tier_reaches_the_service_only_when_serve_env_names_it():
    """C1: QSE_DISK_CACHE becomes --disk-cache on the :8000 service; without it the flag says
    off, and a row or gate server (which never runs start.sh) has no tier either way."""
    r, marks = _start_with_fake_engine("0", "QSE_DISK_CACHE=/srv/disk-cache\n")
    assert r.returncode == 0, r.stdout + r.stderr
    argv = next(line for line in marks.splitlines() if line.startswith("engine ")).split()
    assert argv[argv.index("--disk-cache") + 1] == "/srv/disk-cache", argv
    r, marks = _start_with_fake_engine("0", "")
    argv = next(line for line in marks.splitlines() if line.startswith("engine ")).split()
    assert argv[argv.index("--disk-cache") + 1] == "off", argv


def test_a_profile_of_another_model_replaces_the_27b_model_flags():
    """OPS-99: QSE_ENGINE_FLAGS (a Flash-Next profile: its pack and its MTP drafter) take the place of
    the 27B's model and drafter flags, the shared ones stay; unset, the 27B's argv is the one it was."""
    r, marks = _start_with_fake_engine("0", 'QSE_ENGINE_FLAGS="--model /p/flashnext-packs/3.05 --drafter mtp"\n')
    assert r.returncode == 0, r.stdout + r.stderr
    argv = next(line for line in marks.splitlines() if line.startswith("engine ")).split()
    assert argv[argv.index("--model") + 1] == "/p/flashnext-packs/3.05" and argv.count("--drafter") == 1, argv
    assert argv[argv.index("--drafter") + 1] == "mtp", argv
    for gone in ("--nvfp4", "--fp8-head", "--dflash2-ckpt", "--tree", "--len-fixed", "--len-latch"):
        assert gone not in argv, (gone, argv)
    assert argv[argv.index("--corpus") + 1] == "x", argv          # the lookup's corpus (ENG-229)
    for kept in ("--served-model", "--max-len", "--cache-budget-gb", "--conductor", "--disk-cache"):
        assert kept in argv, (kept, argv)
    r, marks = _start_with_fake_engine("0", "")
    argv = next(line for line in marks.splitlines() if line.startswith("engine ")).split()
    i = argv.index("--drafter")
    assert argv[i:i + 7] == ["--drafter", "lenrouter", "--len-fixed", "0", "--len-latch", "--dflash2-path",
                             "greedy"], argv
    j = argv.index("--tree")
    assert argv[j:j + 15] == ["--tree", "--budget", "16", "--corpus", "x", "--dflash2-ckpt", "x",
                              "--dflash2-ckpt16", "x", "--nvfp4", "x", "--fp8-head", "x",
                              "--cache-budget-gb", "0"], argv
    assert "--model" not in argv, argv


def test_the_served_memory_limits_are_the_owners():
    """The design decisions of 2026-10-01 (OPS-47), and the context window stays 262,144."""
    env = dict(line.split("=", 1) for line in open(os.path.join(OPS, "serve.env")).read().splitlines()
               if "=" in line and not line.startswith("#"))
    want = {"MAX_LEN": "262144", "MEM_CEILING_GIB": "70", "MEM_PEAK_GIB": "77",
            "RESERVE_SLACK_GIB": "2", "RESIDENT_GB": "2.5", "RESIDENT_STASH_GB": "1.0",
            "RESIDENT_GUEST_MAX": "8192", "RESIDENT_PIN_GB": "4.0", "RESIDENT_PARK_GB": "4",
            "CACHE_GB": "2", "STORE_ENTRY_GB": "1", "NARROW_ARM": "1"}
    assert {k: env.get(k) for k in want} == want


def test_the_served_configuration_walks_sampled_requests_on_the_tree():
    """Adopted: serve.env sets SAMPLED_TREE=det, which start.sh turns into
    --sampled-tree=det (the test above)."""
    lines = open(os.path.join(OPS, "serve.env")).read().splitlines()
    assert "SAMPLED_TREE=det" in lines, "serve.env must set SAMPLED_TREE=det"


def test_a_killed_hold_stops_its_command_before_it_restarts_the_service():
    """A hold that is itself signalled -- an ssh session dropped, a tool timeout, a
    `kill` -- ran its EXIT trap and restarted :8000 while the command it was holding for kept
    running: a row3 with its own engine on :8001, beside the restarted service, which is the
    two-engine state that wedges this box. The hold must stop the command's whole process
    group first, and only then bring the service back."""
    if not os.path.isdir("/proc/self"):
        print("    (skipped: needs /proc -- run it on the box)")
        return
    import signal
    repo = _box(["hold.sh"])
    tag = f"{os.getpid()}.25"                              # a sleep only this test runs
    with open(os.path.join(repo, "ops", "start.sh"), "w") as fh:
        fh.write(f"#!/bin/bash\n/usr/bin/pgrep -f 'sleep {tag}' >/dev/null "
                 f"&& echo 'start.sh BESIDE-THE-COMMAND' >> $MARKERS "
                 f"|| echo 'start.sh alone' >> $MARKERS\n")
    held = os.path.join(repo, "held.sh")
    with open(held, "w") as fh:                            # a command with a child, like row3
        fh.write(f"#!/bin/bash\nsleep {tag} &\nwait\n")
    e = _env(repo)
    for sig in (signal.SIGTERM, signal.SIGHUP):
        if os.path.exists(e["MARKERS"]):
            os.remove(e["MARKERS"])
        hold = subprocess.Popen(["bash", os.path.join(repo, "ops", "hold.sh"), "1", "--",
                                 "bash", held], env=e, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL, start_new_session=True)
        time.sleep(1.5)
        hold.send_signal(sig)
        hold.wait(timeout=30)
        time.sleep(0.5)
        left = subprocess.run(["/usr/bin/pgrep", "-f", f"sleep {tag}"], capture_output=True)
        subprocess.run(["/usr/bin/pkill", "-f", f"sleep {tag}"])
        markers = open(e["MARKERS"]).read()
        assert "start.sh alone" in markers, f"{sig.name}: {markers}"
        assert left.returncode != 0, f"{sig.name}: the held command outlived its hold"
        assert not os.path.exists(os.path.join(repo, ".watchdog.off"))


def test_the_restarted_service_does_not_inherit_the_holds_descriptors():
    """A hold run as `flock LOCKFILE hold.sh ...` has the box lock on an open descriptor, and the
    service it restarts used to inherit it: on 2026-09-23 at 09:24 the restarted :8000 engine held
    the lock, and every later `flock` waited on the operator's engine. The restart runs with nothing
    above stdio open. Here the lock is fd 9 and fd 5, and the fake start.sh reports what it got."""
    repo = _box(["hold.sh"])
    start = os.path.join(repo, "ops", "start.sh")
    with open(start, "w") as fh:
        fh.write('#!/bin/bash\nfor fd in 5 9; do { true >&$fd; } 2>/dev/null && '
                 'echo "start.sh inherited fd $fd" >> "$MARKERS"; done\n'
                 'echo "start.sh ran" >> "$MARKERS"\n')
    os.chmod(start, 0o755)
    lock = os.path.join(repo, "box.flock")
    e = _env(repo)
    out = open(os.path.join(repo, "hold.out"), "w")
    r = subprocess.run(["bash", "-c", f'exec 9>"{lock}" 5>"{lock}.2"; '
                        f'bash "{repo}/ops/hold.sh" 1 -- true'],
                       stdout=out, stderr=subprocess.STDOUT, env=e, timeout=60)
    # to a file, not a pipe: the hold's keeper loop leaves a `sleep 60` behind it holding any pipe
    assert r.returncode == 0, open(out.name).read()
    marks = open(e["MARKERS"]).read()
    assert "start.sh ran" in marks, marks
    assert "inherited" not in marks, marks


def test_a_signalled_hold_under_flock_o_restores_alone_and_inside_the_lock():
    """Under the `flock -o`. With -o the lock lives in flock's own process and nowhere
    below it, so the order that matters is: the held command is gone before start.sh runs, and the
    lock is still taken while start.sh runs -- flock waits on hold.sh, so a signalled hold's stop
    and restore both happen inside it. The signal goes to hold.sh, as a `kill` of the hold does."""
    if not os.path.isdir("/proc/self") or shutil.which("flock") is None:
        print("    (skipped: needs /proc and flock -- run it on the box)")
        return
    import signal
    repo = _box(["hold.sh"])
    lock = os.path.join(repo, "box.flock")
    tag = f"{os.getpid()}.35"
    with open(os.path.join(repo, "ops", "start.sh"), "w") as fh:
        fh.write(f"#!/bin/bash\n/usr/bin/pgrep -f 'sleep {tag}' >/dev/null "
                 f"&& echo 'start.sh BESIDE-THE-COMMAND' >> $MARKERS "
                 f"|| echo 'start.sh alone' >> $MARKERS\n"
                 f"flock -n {lock} true && echo 'lock FREE during the restore' >> $MARKERS "
                 f"|| echo 'lock held during the restore' >> $MARKERS\n")
    held = os.path.join(repo, "held.sh")
    with open(held, "w") as fh:
        fh.write(f"#!/bin/bash\nsleep {tag} &\nwait\n")
    e = _env(repo)
    fl = subprocess.Popen(["flock", "-o", lock, "bash", os.path.join(repo, "ops", "hold.sh"), "1",
                           "--", "bash", held], env=e, stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL, start_new_session=True)
    try:
        time.sleep(1.5)
        hold = subprocess.run(["/usr/bin/pgrep", "-P", str(fl.pid)], capture_output=True, text=True)
        assert hold.stdout.split(), "hold.sh is not running under flock"
        os.kill(int(hold.stdout.split()[0]), signal.SIGTERM)
        fl.wait(timeout=30)
        time.sleep(0.5)
        left = subprocess.run(["/usr/bin/pgrep", "-f", f"sleep {tag}"], capture_output=True)
        markers = open(e["MARKERS"]).read()
        assert "start.sh alone" in markers, markers
        assert "lock held during the restore" in markers, markers
        assert left.returncode != 0, "the held command outlived its hold"
        assert subprocess.run(["flock", "-n", lock, "true"], timeout=10).returncode == 0
    finally:
        subprocess.run(["/usr/bin/pkill", "-f", f"sleep {tag}"])


def test_a_plain_flock_hold_frees_the_lock_when_it_returns():
    """The plain form, `flock LOCK hold.sh ...`, puts the lock on a descriptor every child of the
    hold inherits. The engine no longer gets it (both scripts), and neither may the keeper:
    `kill $KEEPER` ends the loop but not the `sleep 60` inside it, which held the lock for up to a
    minute after the hold had returned, and the next agent's hold waited on a sleep."""
    if shutil.which("flock") is None:
        print("    (skipped: needs flock -- run it on the box)")
        return
    repo = _box(["hold.sh"])
    lock = os.path.join(repo, "box.flock")
    e = _env(repo)
    out = open(os.path.join(repo, "hold.out"), "w")
    r = subprocess.run(["flock", lock, "bash", os.path.join(repo, "ops", "hold.sh"), "1", "--",
                        "true"], stdout=out, stderr=subprocess.STDOUT, env=e, timeout=60)
    assert r.returncode == 0, open(out.name).read()
    free = subprocess.run(["flock", "-n", lock, "true"], timeout=10)
    assert free.returncode == 0, "the lock must be free as soon as the hold has returned"


# ------------------------------------------------------------------

LOOKALIKES = [
    # the 2026-09-23 22:47 lock holder
    (["flock", "-o", "/home/u/.qwen38-box.flock", "bash", "-c",
      "cd ~/e && ~/.venv/bin/python -u server/app.py --host 0.0.0.0 --port 8000 --tree"],
     "/usr/bin/flock"),
    (["bash", "-c", "python -u server/app.py --host 127.0.0.1 --port 8000"], "/usr/bin/bash"),
    ([VENV_PY, "-c", "import runpy; runpy.run_path('server/app.py') # --host a --port 8000"],
     PY_EXE),
    ([VENV_PY, "-u", "tools/row3.py", "--server", "server/app.py", "--host", "a", "--port", "8000"],
     PY_EXE),
    (["grep", "server/app.py --host .* --port 8000"], "/usr/bin/grep"),
    # an engine's argument vector under something that is not Python (`exec -a`)
    (_engine_argv(), "/usr/bin/bash"),
]


def _engines(repo: str, *port: str) -> list[str]:
    r = subprocess.run(["bash", "-c", f'. "{repo}/ops/engines.sh"; engine_pids {" ".join(port)}'],
                       capture_output=True, text=True, env=_env(repo))
    assert r.returncode == 0 and not r.stderr, r.stderr
    return r.stdout.split()


def test_only_a_python_running_server_app_py_is_an_engine():
    """The executable is Python and the script it runs is server/app.py -- relative, or a
    path that ends in it, after interpreter options; the port is `--port`, `--port=`, or app.py's
    default 8000. Every look-alike of 2026-09-23, and a zombie (empty cmdline), is not one."""
    repo = _box([])
    _proc(repo, 1001, _engine_argv(8000))
    _proc(repo, 1002, [VENV_PY, "/home/u/e/server/app.py", "--port=8011"])
    _proc(repo, 1003, [VENV_PY, "-u", "server/app.py", "--fake-engine"])
    for i, (argv, exe) in enumerate(LOOKALIKES):
        _proc(repo, 2001 + i, argv, exe)
    _proc(repo, 3001, [], PY_EXE)
    _proc(repo, 3002, _engine_argv(8000), "")                  # exe unreadable: another user's
    assert _engines(repo) == ["1001", "1002", "1003"]
    assert _engines(repo, "8000") == ["1001", "1003"]
    assert _engines(repo, "8011") == ["1002"]
    assert _engines(repo, "8001") == []


def test_stop_signals_the_engine_and_nothing_that_only_mentions_it():
    """Scenario 1. The engine on :8000 and every look-alike alive: stop.sh signals the
    engine alone. (Before, the lock holder got SIGTERM and the box lock went with it.) An engine
    on another port is not the service and is left alone too. No pid file is involved, so a
    service an older start.sh launched is still found."""
    repo = _box(["stop.sh"])
    procs = [_sleeper() for _ in range(len(LOOKALIKES) + 2)]
    try:
        engine, other, rest = procs[0], procs[1], procs[2:]
        _proc(repo, engine, _engine_argv(8000))
        _proc(repo, other, _engine_argv(8011))
        for p, (argv, exe) in zip(rest, LOOKALIKES):
            _proc(repo, p, argv, exe)
        r = _run(repo, "stop.sh", "5")
        assert r.returncode == 0, r.stdout + r.stderr
        assert f"SIGTERM {engine} " in r.stdout and "[stop] stopped" in r.stdout, r.stdout
        assert not _alive(engine)
        assert [p for p in rest + [other] if _alive(p)] == rest + [other], r.stdout
    finally:
        for p in procs:
            if _alive(p):
                os.kill(p, 9)


def test_stop_with_only_a_look_alike_stops_nothing():
    repo = _box(["stop.sh"])
    p = _sleeper()
    try:
        _proc(repo, p, *LOOKALIKES[0])
        r = _run(repo, "stop.sh", "5")
        assert "nothing on :8000" in r.stdout and _alive(p), r.stdout
    finally:
        os.kill(p, 9)


def test_a_hold_runs_and_restores_with_a_look_alike_alive():
    """Scenario 2. A process whose command line mentions server/app.py is alive (the lock
    holder of 22:47) and no engine is: the hold runs its command and restarts the service. Before,
    it refused both, and :8000 stayed down."""
    repo = _box(["hold.sh"])
    _proc(repo, 4242, *LOOKALIKES[0])
    r = subprocess.run(["bash", os.path.join(repo, "ops", "hold.sh"), "1", "--", "true"],
                       capture_output=True, text=True, env=_env(repo, HOLD_ENGINE_WAIT=3),
                       timeout=120)
    assert r.returncode == 0 and "REFUSING" not in r.stdout, r.stdout + r.stderr
    markers = open(os.path.join(repo, "markers")).read().split()
    assert markers[::2] == ["stop.sh", "start.sh"], markers


def test_a_hold_still_refuses_to_restore_beside_a_real_second_engine():
    """Scenario 3. The command left an engine alive on :8011 (row3's port): the restore
    waits, then refuses to start :8000 beside it and names the pid (one engine at a time)."""
    repo = _box(["hold.sh"])
    left = os.path.join(repo, "leave-engine.py")
    with open(left, "w") as fh:
        fh.write(f"""import os
d = os.path.join({os.path.join(repo, "proc")!r}, "5151")
os.makedirs(d)
open(os.path.join(d, "cmdline"), "wb").write(b"\\0".join(a.encode() for a in {_engine_argv(8011)!r}) + b"\\0")
os.symlink({PY_EXE!r}, os.path.join(d, "exe"))
""")
    r = subprocess.run(["bash", os.path.join(repo, "ops", "hold.sh"), "1", "--",
                        sys.executable, left],
                       capture_output=True, text=True, env=_env(repo, HOLD_ENGINE_WAIT=2),
                       timeout=120)
    assert "REFUSING to restart the service: an engine is still alive: 5151" in r.stdout, r.stdout
    markers = open(os.path.join(repo, "markers")).read()
    assert "start.sh" not in markers, markers
    assert not os.path.exists(os.path.join(repo, ".watchdog.off"))


def test_a_hold_refuses_to_start_its_command_beside_an_engine_stop_sh_left():
    repo = _box(["hold.sh"])
    _proc(repo, 6161, _engine_argv(8011))
    r = subprocess.run(["bash", os.path.join(repo, "ops", "hold.sh"), "1", "--", "true"],
                       capture_output=True, text=True, env=_env(repo, HOLD_ENGINE_WAIT=1),
                       timeout=120)
    assert r.returncode == 1 and "still alive after stop.sh: 6161" in r.stdout, r.stdout


def test_the_watchdog_does_not_take_a_look_alike_for_a_loader():
    """In the watchdog: a silent :8000 with only a look-alike alive is not "loading, leave
    it alone" -- the strikes count and the third restarts the service."""
    repo = _box(["watchdog.sh"])
    _proc(repo, 4242, *LOOKALIKES[0])
    with open(os.path.join(repo, "logs", "watchdog.fails"), "w") as fh:
        fh.write("2\n")
    r = _run(repo, "watchdog.sh", FAKE_HTTP_CODE="000")
    assert r.returncode == 0, r.stderr
    assert "loading" not in _wd_log(repo) and "restarting" in _wd_log(repo), _wd_log(repo)


def test_engines_on_the_real_process_table():
    """The same rule over the real /proc, with a real Python running a script called
    server/app.py and a real look-alike. Opt-in (QSE_OPS_LIVE=1), and run inside a hold: until
    is deployed, the served hold.sh takes either process for an engine."""
    if os.environ.get("QSE_OPS_LIVE") != "1" or not os.path.isdir("/proc/self"):
        print("    (skipped: set QSE_OPS_LIVE=1, on the box, inside a hold)")
        return
    repo = _box([])
    os.makedirs(os.path.join(repo, "server"))
    with open(os.path.join(repo, "server", "app.py"), "w") as fh:
        fh.write("import time\ntime.sleep(60)\n")
    port = str(18000 + os.getpid() % 1000)
    eng = subprocess.Popen([sys.executable, "-u", "server/app.py", "--host", "127.0.0.1",
                            "--port", port], cwd=repo)
    # two commands, so bash does not exec the sleep and its command line keeps the text
    look = subprocess.Popen(["bash", "-c", f"sleep 60; : server/app.py --host 0.0.0.0 --port {port}"])
    try:
        time.sleep(0.5)
        env = _env(repo)
        del env["ENGINE_PROC"]
        r = subprocess.run(["bash", "-c", f'. "{repo}/ops/engines.sh"; engine_pids {port}'],
                           capture_output=True, text=True, env=env)
        assert r.stdout.split() == [str(eng.pid)], (r.stdout, eng.pid, look.pid)
    finally:
        eng.kill()
        subprocess.run(["pkill", "-P", str(look.pid)])
        look.kill()


# ------------------------------------------------------------------ before the cron lines come back

def _fails(repo: str) -> str:
    path = os.path.join(repo, "logs", "watchdog.fails")
    return open(path).read().strip() if os.path.exists(path) else ""


def _strikes(repo: str, n: int) -> None:
    with open(os.path.join(repo, "logs", "watchdog.fails"), "w") as fh:
        fh.write(f"{n}\n")


def test_the_watchdog_logs_the_code_curl_gave():
    """Curl prints 000 itself when nothing answers, and exits non-zero; `|| echo 000` after it made
    every failed check of 2026-09-19 read `code 000000`."""
    repo = _box(["watchdog.sh"])
    assert _run(repo, "watchdog.sh", FAKE_HTTP_CODE="000").returncode == 0
    assert "health check failed (code 000), 1/3" in _wd_log(repo), _wd_log(repo)
    assert _run(repo, "watchdog.sh", FAKE_HTTP_CODE="502").returncode == 0
    assert "health check failed (code 502), 2/3" in _wd_log(repo), _wd_log(repo)


def test_a_fresh_pause_keeps_the_watchdog_silent_whatever_the_service_says():
    """Hold.sh arms `.watchdog.off` first thing: with :8000 silent and two strikes already, the
    watchdog neither counts nor restarts nor writes a line."""
    repo = _box(["watchdog.sh"])
    open(os.path.join(repo, ".watchdog.off"), "w").close()
    _strikes(repo, 2)
    for _ in range(3):
        assert _run(repo, "watchdog.sh", FAKE_HTTP_CODE="000").returncode == 0
    assert _wd_log(repo) == "" and _fails(repo) == "2"
    assert not os.path.exists(os.path.join(repo, "markers")), "nothing was restarted"


def _hold_lock(lock: str) -> subprocess.Popen:
    """Another process holding the box lock, as a hold's `flock` does; returns once it is held."""
    open(lock, "a").close()
    p = subprocess.Popen(["flock", lock, "sleep", "60"], start_new_session=True)
    end = time.time() + 5
    while time.time() < end and subprocess.run(["flock", "-n", lock, "true"]).returncode == 0:
        time.sleep(0.05)
    return p


def test_the_watchdog_leaves_the_board_to_whoever_holds_the_box_lock():
    """A hold owns the board from the moment its flock returns, and the pause file only from hold.sh's
    first line; a lock holder that is not hold.sh at all (a CPU job under the lock, a hand) arms no
    pause. The watchdog asks the lock itself: held, it neither counts nor restarts, and it says why
    once the service is not answering. Healthy, it never touches the lock (no agent waits on it)."""
    if shutil.which("flock") is None:
        print("    (skipped: needs flock -- run it on the box)")
        return
    repo = _box(["watchdog.sh"])
    lock = os.path.join(repo, "box.flock")
    holder = _hold_lock(lock)
    try:
        assert _run(repo, "watchdog.sh", FAKE_HTTP_CODE="200").returncode == 0
        assert _wd_log(repo) == "", "a healthy minute writes nothing"
        _strikes(repo, 2)
        assert _run(repo, "watchdog.sh", FAKE_HTTP_CODE="000").returncode == 0
        assert "box lock held (code 000), leaving it alone" in _wd_log(repo), _wd_log(repo)
        assert _fails(repo) == "0" and not os.path.exists(os.path.join(repo, "markers"))
    finally:
        os.killpg(holder.pid, 9)                      # flock and the sleep that inherited the lock
        holder.wait()
    _strikes(repo, 2)                                 # the lock is free: the third strike restarts
    assert _run(repo, "watchdog.sh", FAKE_HTTP_CODE="000").returncode == 0
    markers = open(os.path.join(repo, "markers")).read().split()
    assert markers[::2] == ["stop.sh", "start.sh"], markers


def test_the_watchdog_restarts_inside_the_box_lock():
    """The restart holds the lock from the check to the end of start.sh, so a hold cannot begin
    between them (and stop the engine the watchdog is loading); the lock is free when it returns."""
    if shutil.which("flock") is None:
        print("    (skipped: needs flock -- run it on the box)")
        return
    repo = _box(["watchdog.sh"])
    lock = os.path.join(repo, "box.flock")
    open(lock, "w").close()
    with open(os.path.join(repo, "ops", "start.sh"), "w") as fh:
        fh.write(f"#!/bin/bash\nflock -n {lock} true && echo 'start.sh lock FREE' >> $MARKERS "
                 f"|| echo 'start.sh lock held' >> $MARKERS\n")
    _strikes(repo, 2)
    assert _run(repo, "watchdog.sh", FAKE_HTTP_CODE="000").returncode == 0
    assert "start.sh lock held" in open(os.path.join(repo, "markers")).read()
    assert subprocess.run(["flock", "-n", lock, "true"]).returncode == 0


def test_a_cold_boot_is_left_to_load():
    """And the @reboot line together, one tick a minute. The @reboot line sleeps 90 s and then
    start.sh launches the engine; the first two ticks find nothing (strikes 1 and 2), and from the
    third the engine exists: it is LOADING, the strikes reset and nothing is restarted until it has
    been silent for LOADING_MAX. (Why 90 s is safe: it is less than the WATCHDOG_FAILS - 1 = 2
    minutes the third strike needs, so the engine always exists by then.) Past LOADING_MAX it is
    hung, and the strikes count again to a restart."""
    repo = _box(["watchdog.sh"])
    for n in (1, 2):
        assert _run(repo, "watchdog.sh", FAKE_HTTP_CODE="000").returncode == 0
        assert _fails(repo) == str(n)
    _proc(repo, 777, _engine_argv())                  # start.sh from the @reboot line
    for _ in range(3):
        assert _run(repo, "watchdog.sh", FAKE_HTTP_CODE="000").returncode == 0
        assert _fails(repo) == "0"
    assert _wd_log(repo).count("loading (code 000") == 3
    assert not os.path.exists(os.path.join(repo, "markers")), "a loading engine is left alone"
    with open(os.path.join(repo, "logs", "loading.since"), "w") as fh:
        fh.write(f"777 {int(time.time()) - 901}\n")   # LOADING_MAX (900 s) has passed
    for n in (1, 2, 3):
        assert _run(repo, "watchdog.sh", FAKE_HTTP_CODE="000").returncode == 0
    log = _wd_log(repo)
    assert log.count("treating as hung") == 3 and "restarting" in log, log
    markers = open(os.path.join(repo, "markers")).read().split()
    assert markers[::2] == ["stop.sh", "start.sh"], markers


def test_start_refuses_beside_a_loading_engine_or_one_on_another_port():
    """The server binds its port only once the weights are in, so for the minutes of a load neither
    /health nor `ss` sees it: start.sh launched a second engine beside it. And an engine on another
    port (a probe server a signalled hold left behind) was no reason to refuse either, so the watchdog
    would start :8000 beside it. One engine at a time: start.sh refuses, non-zero, naming the pid."""
    for port in (8000, 8011):
        repo = _box(["start.sh"])
        _proc(repo, 4321, _engine_argv(port))
        r = _run(repo, "start.sh")
        assert r.returncode == 1, r.stdout + r.stderr
        assert "REFUSING: an engine is already alive (pid 4321)" in r.stdout, r.stdout
        assert not os.path.exists(os.path.join(repo, "logs", "engine.pid")), "nothing was launched"


def test_start_is_not_stopped_by_a_look_alike():
    """In start.sh: a process that only mentions server/app.py (the lock holder of 22:47) is
    not an engine, and the service starts."""
    r, _ = _start_with_fake_engine("0", procs=[(4242, *LOOKALIKES[0])])
    assert r.returncode == 0 and "healthy" in r.stdout, r.stdout + r.stderr

FAKE_GATE_PY = """#!/bin/bash
# gate.sh's $PY: the code hash for `-c`, and tools/block_ab.py recorded and answered.
for x in "$@"; do [ "$x" = "-c" ] && { echo 0123456789abcdef0123; exit 0; }; done
case " $* " in
  *" tools/block_ab.py "*)
    echo "py $*" >> "$MARKERS"
    out=""; label=""; prev=""
    for x in "$@"; do
      [ "$prev" = "--out" ] && out="$x"; [ "$prev" = "--label" ] && label="$x"; prev="$x"
    done
    printf '## block A/B %s (fake)\n\nruns and verdicts\n' "$label" > "$out/$label-stub.md"
    echo "[block-ab] $label (better): fake"
    exit "${FAKE_BA_RC:-0}";;
esac
exit 0
"""


def _gate_box() -> str:
    repo = _box(["gate.sh"])
    with open(os.path.join(repo, "ops", "serve.env"), "a") as fh:
        fh.write("NV=/nv\nHEAD=/head\nCKPT8=/ck8\nCKPT16=/ck16\nCORPUS=/corpus\nFUSE_PROJ=0\n"
                 "QWEN38_VERIFY_GRAPH=1\n")
    py = os.path.join(repo, "bin", "fakepy")
    with open(py, "w") as fh:
        fh.write(FAKE_GATE_PY)
    os.chmod(py, 0o755)
    return repo


def test_the_gate_runs_the_block_ab_with_the_base_and_every_candidate():
    """--block-ab puts the served configuration first and each candidate after it, passes
    the pairs, workloads and the greedy check, and carries the tool's stub into the gate's report."""
    repo = _gate_box()
    skip = ["--skip-suite", "--skip-gpu", "--skip-identity", "--skip-lossless", "--skip-row"]
    r = _run(repo, "gate.sh", "cal", *skip, "--block-ab", "kr1=tools.nvfp4_skinny:WIDE_KR=1;PF=2",
             "--block-ab", "null=engine.model:TREE_ALIAS_STATE=0", "--block-pairs", "4",
             GATE_PY=os.path.join(repo, "bin", "fakepy"))
    assert r.returncode == 0, r.stdout + r.stderr
    call = open(os.path.join(repo, "markers")).read()
    assert "tools/block_ab.py --label cal" in call, call
    assert call.index("--state base=") < call.index("--state kr1=tools.nvfp4_skinny:WIDE_KR=1;PF=2") \
        < call.index("--state null=engine.model:TREE_ALIAS_STATE=0"), call
    for want in ("--pairs 4", "--workloads prose,chat,code", "--rule better", "--greedy-check",
                 "--ckpt8 /ck8", "--ckpt16 /ck16", "--precapture 32"):
        assert want in call, (want, call)
    ledger = open(os.path.join(repo, "results", "gate", "cal", "report.md")).read()
    assert "5a block A/B (better): PASS" in ledger and "## block A/B cal (fake)" in ledger, ledger
    assert "GATE cal: every step PASS" in ledger


def test_a_failing_block_ab_stops_the_gate():
    repo = _gate_box()
    skip = ["--skip-suite", "--skip-gpu", "--skip-identity", "--skip-lossless", "--skip-row"]
    r = _run(repo, "gate.sh", "cal2", *skip, "--block-ab", "c=m:A=1", "--block-rule", "noworse",
             GATE_PY=os.path.join(repo, "bin", "fakepy"), FAKE_BA_RC=1)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "--rule noworse" in open(os.path.join(repo, "markers")).read()
    ledger = open(os.path.join(repo, "results", "gate", "cal2", "report.md")).read()
    assert "5a block A/B (noworse): FAIL" in ledger and "ABORTED at 5a block A/B" in ledger, ledger


def test_the_scripts_parse():
    for name in ("watchdog.sh", "start.sh", "stop.sh", "hold.sh", "engines.sh", "gate.sh",
                 "roles.sh", "chatd.sh", "make-secrets.sh"):
        r = subprocess.run(["bash", "-n", os.path.join(OPS, name)], capture_output=True, text=True)
        assert r.returncode == 0, f"{name}: {r.stderr}"


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:60s} ok")
            passed += 1
    print(f"{passed} passed")
