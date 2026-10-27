"""bin/tandem on an install with no ops/active-profile (OPS-159, QA OPS-160).

install.sh writes bin/tandem with `set -euo pipefail` and never writes ops/active-profile; serve.env's
profile line read the file with a bare `$(cat ...)`, whose failure under set -e ended bin/tandem with
status 1 and no message. The template's line tolerates a missing file now, and bin/tandem names the
file and line of a config command that fails. The tests source the real template and run the
bin/tandem install.sh writes (its `write_control`); the install checks run install.sh into an empty
HOME with tests/test_fresh_install.py's stand-ins (Linux only) and a stub server/app.py on a free port,
never the box's engine (a process table of their own, ENGINE_PROC).

The test matrix, the Gherkin of OPS-159:

  1  the template sourced under set -euo pipefail loads with no active-profile, an empty one, one
     naming a profile with no env file and a community/ name with none; an existing profile (nvfp4,
     a cold Flash-Next one, a community/<id> file) applies as before; the unfixed line exits 1 and
     prints nothing                                          test_the_template_loads_without_a_profile
  2  bin/tandem status, env, logs 5 and stop reach their own code on such a config; start runs
     start.sh and exits 0 once /health answers               test_every_subcommand_reaches_its_code
  3  a config command that fails: bin/tandem prints the file and the line and exits 1; the old
     script (no report) exits 1 silently on the same config  test_a_failing_config_says_why
  4  a plain install (no --setup) and a --setup install: run/ops/serve.env loads under set -euo
     pipefail with no active-profile                          test_the_rendered_configs_load
  5  a plain install, then bin/tandem start, status, env, logs and stop against a stub engine on a
     free port; and the same with --setup (setup mode answers /health)
                                                              test_a_plain_install_starts
                                                              test_a_setup_install_starts
  6  a re-run of install.sh rewrites an old serve.env and bin/tandem with the fix
                                                              test_a_rerun_rewrites_the_config
  7  start drops the page cache of the weight paths serve.env names only: setup mode's empty ones
     are not passed (an empty one walked the working directory)
                                                              test_start_drops_only_named_paths

Run: python tests/test_install_control.py   (4 to 6 need Linux: install.sh stops anywhere else)
"""
from __future__ import annotations

import json
import os
import platform
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tests"))

import test_chatd_lifecycle as C  # noqa: E402
import test_fresh_install as F  # noqa: E402

TEMPLATE = os.path.join(ROOT, "ops", "serve.env")
FIXED = '$(cat "${REPO}/ops/active-profile" 2>/dev/null || true)}'
UNFIXED = '$(cat "${REPO}/ops/active-profile" 2>/dev/null)}'
LOAD = 'set -euo pipefail; set -a; . "$1"; set +a; echo "loaded QSE_PROFILE_NAME=$QSE_PROFILE_NAME NV=$NV"'


def _template_box(text: str | None = None) -> tuple[str, str, dict]:
    """A repo with the release's profiles and the template (REPO pointing at it), in a HOME of its own."""
    d = tempfile.mkdtemp(prefix="install-ctl-")
    repo = os.path.join(d, "repo")
    shutil.copytree(os.path.join(ROOT, "ops", "profiles"), os.path.join(repo, "ops", "profiles"))
    env_file = os.path.join(repo, "ops", "serve.env")
    text = (text or open(TEMPLATE).read()).replace("REPO=${HOME}/qwen38-spark-engine", f"REPO={repo}")
    with open(env_file, "w") as fh:
        fh.write(text)
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": os.path.join(d, "home")}
    return repo, env_file, env


def _load(env_file: str, env: dict) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", "-c", LOAD, "-", env_file], capture_output=True, text=True, env=env,
                          timeout=30)


def _profile(repo: str, text: str | None) -> None:
    p = os.path.join(repo, "ops", "active-profile")
    if text is None:
        if os.path.exists(p):
            os.remove(p)
    else:
        with open(p, "w") as fh:
            fh.write(text)


def test_the_template_loads_without_a_profile():
    assert open(TEMPLATE).read().count(FIXED) == 1, "serve.env's profile line"
    repo, env_file, env = _template_box()
    base_nv = None
    for case in (None, "", "nope\n", "community/nope\n"):
        _profile(repo, case)
        r = _load(env_file, env)
        assert r.returncode == 0 and "loaded" in r.stdout, (case, r.stdout + r.stderr)
        nv = r.stdout.split(" NV=")[1]
        base_nv = base_nv or nv
        assert nv == base_nv, (case, nv)                 # the template's own values
    # an existing profile applies, as before
    _profile(repo, "nvfp4\n")
    r = _load(env_file, env)
    assert r.returncode == 0 and "QSE_PROFILE_NAME=nvfp4 " in r.stdout, r.stdout + r.stderr
    # a cold Flash-Next profile: its own `$([ cold = warm ] && ...)` failed the same way
    _profile(repo, "flashnext-4.05\n")
    r = subprocess.run(["bash", "-c", LOAD + '; echo "KEEP=[$QSE_PAGE_CACHE_KEEP] TABLE=$QWEN38_FN_TABLE"',
                        "-", env_file], capture_output=True, text=True, env=env, timeout=30)
    assert r.returncode == 0 and "KEEP=[] TABLE=cold" in r.stdout, r.stdout + r.stderr
    # a community/<id> profile reads its file
    cdir = os.path.join(repo, "run", "community")
    os.makedirs(cdir)
    with open(os.path.join(cdir, "x.env"), "w") as fh:
        fh.write("NV=from-community\n")
    _profile(repo, "community/x\n")
    r = _load(env_file, env)
    assert r.returncode == 0 and r.stdout.strip().endswith("NV=from-community"), r.stdout + r.stderr
    # QSE_PROFILE_OVERRIDE skips the file
    r = _load(env_file, dict(env, QSE_PROFILE_OVERRIDE="nvfp4"))
    assert r.returncode == 0 and "QSE_PROFILE_NAME=nvfp4 " in r.stdout
    # the negative control: the unfixed line dies silently without the file
    repo, env_file, env = _template_box(open(TEMPLATE).read().replace(FIXED, UNFIXED))
    r = _load(env_file, env)
    assert r.returncode == 1 and r.stdout == "" and r.stderr == "", r.stdout + r.stderr


def _ctl_box(serve_env_extra: str = "", old_control: bool = False) -> tuple[str, str, dict]:
    """test_chatd_lifecycle's bin/tandem box, with the template's profile block in its serve.env
    (no ops/active-profile under REPO) and `extra` lines after it."""
    d, marks, env = C._tandem()
    tmpl = open(TEMPLATE).read()
    block = tmpl[tmpl.index("# Weight profile (ops/use-profile.sh)"):]
    block = block[:block.index("\nfi\n") + 4]
    serve = os.path.join(d, "run", "ops", "serve.env")
    with open(serve, "a") as fh:
        fh.write(block + serve_env_extra)
    if old_control:
        script = open(os.path.join(d, "bin", "tandem")).read()
        script = re.sub(r"\n# a command that fails in the config.*?\ntrap '[^\n]*' ERR\n", "\n", script, flags=re.S)
        script = script.replace("\ntrap - ERR\n", "\n")
        assert " ERR" not in script
        C._write(os.path.join(d, "bin", "tandem"), script)
    return d, marks, env


def logs(ctl: str, env: dict) -> str:
    """`tandem logs 5` for a second and a half; it follows the log (tail -f), so its process group
    is ended."""
    p = subprocess.Popen([ctl, "logs", "5"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         env=env, start_new_session=True)
    time.sleep(1.5)
    os.killpg(p.pid, signal.SIGTERM)
    return p.communicate(timeout=10)[0]


def test_every_subcommand_reaches_its_code():
    d, marks, env = _ctl_box()
    assert not os.path.exists(os.path.join(d, "src", "ops", "active-profile"))
    r = C._ctl(d, env, "status")
    assert r.returncode == 1 and r.stdout.startswith("[tandem] not running"), r.stdout + r.stderr
    r = C._ctl(d, env, "env")
    assert r.returncode == 0 and "QSE_PROFILE_NAME=" in r.stdout, r.stdout + r.stderr
    os.makedirs(os.path.join(d, "src", "logs"))
    with open(os.path.join(d, "src", "logs", "engine.log"), "w") as fh:
        fh.write("".join(f"line {i}\n" for i in range(10)))
    out = logs(os.path.join(d, "bin", "tandem"), env)
    assert out.splitlines() == [f"line {i}" for i in range(5, 10)], out
    C._lines(marks)
    r = C._ctl(d, env, "start")
    assert r.returncode == 0 and "[tandem] healthy" in r.stdout, r.stdout + r.stderr
    assert C._lines(marks)[0] == "start.sh "
    r = C._ctl(d, env, "stop")
    assert r.returncode == 0 and C._lines(marks)[-1] == "stop.sh 60", r.stdout + r.stderr
    # an empty active-profile, or one naming no env file: status still loads the config
    for text in ("", "nope\n"):
        _profile(os.path.join(d, "src"), text)
        r = C._ctl(d, env, "status")
        assert r.returncode == 1 and r.stdout.startswith("[tandem] not running"), (text, r.stdout + r.stderr)


def test_a_failing_config_says_why():
    extra = "A=1\nB=$(false)\n"
    d, _, env = _ctl_box(extra)
    serve = os.path.join(d, "run", "ops", "serve.env")
    line = open(serve).read().splitlines().index("B=$(false)") + 1
    for cmd in ("start", "status", "env"):
        r = C._ctl(d, env, cmd)
        assert r.returncode == 1 and f"{serve}: line {line} failed" in r.stderr, (cmd, r.stdout + r.stderr)
        assert "the config did not load" in r.stderr and "[tandem] healthy" not in r.stdout
    # a file the config sources (local.env) is named by its own path and line
    d, _, env = _ctl_box()
    local = os.path.join(d, "run", "local.env")
    C._write(local, "X=1\nY=$(false)\n", 0o644)
    r = C._ctl(d, env, "status")
    assert r.returncode == 1 and f"{local}: line 2 failed" in r.stderr, r.stdout + r.stderr
    # the negative control: the script without the report, on the same config, says nothing
    d, _, env = _ctl_box(extra, old_control=True)
    r = C._ctl(d, env, "start")
    assert r.returncode == 1 and r.stdout == "" and r.stderr == "", r.stdout + r.stderr


def test_start_drops_only_named_paths():
    """Setup mode names no weights: an empty path reached drop_page_cache.py as "", which walked the
    working directory (the scratch install on the Spark dropped the cache of 1.07 M files in $HOME)."""
    d, marks, env = _ctl_box("NV=\nHEAD=\nCKPT8=\nCKPT16=\n")
    open(marks, "w").close()
    r = C._ctl(d, env, "start")
    assert r.returncode == 0, r.stdout + r.stderr
    calls = [ln for ln in open(marks).read().splitlines() if ln.startswith("py ")]
    assert calls == [f"py {d}/src/tools/drop_page_cache.py m"], calls
    C._ctl(d, env, "stop")
    # nothing named: no call at all (with no arguments the tool drops its default paths)
    with open(os.path.join(d, "run", "ops", "serve.env"), "a") as fh:
        fh.write("QWEN38_MODEL=\n")
    open(marks, "w").close()
    r = C._ctl(d, env, "start")
    assert r.returncode == 0 and not [ln for ln in open(marks).read().splitlines() if ln.startswith("py ")], \
        open(marks).read()


# ---------------------------------------------------------------- install.sh into an empty HOME (Linux)
# the venv's Python for a plain install: test_fresh_install's, and snapshot_download answers with a
# folder of stand-in files in this HOME's HF cache (no network)
_PLAIN_PY = """#!/bin/bash
if [ "$1" = -c ] && [[ "$2" == *version_info* ]]; then exit 0; fi
if [ "$1" = - ]; then
    s="$(cat)"
    case "$s" in
      *"import torch"*) echo "    ok torch (stand-in)"; exit 0 ;;
      *"snapshot_download(repo, allow_patterns"*)
        d="$HOME/.cache/huggingface/hub/models--${{2//\\//--}}/snapshots/stand-in"; mkdir -p "$d"
        for f in config.json tokenizer.json outside.safetensors mlp.safetensors gdn.safetensors \\
                 attn.safetensors head-fp8.safetensors model.safetensors; do echo x > "$d/$f"; done
        echo "$d"; exit 0 ;;
    esac
    shift; exec {py} -c "$s" "$@"
fi
exec {py} "$@"
"""
# huggingface.co's model API answers 200; everything else is the real curl
_CURL = """#!/bin/bash
case "$*" in *huggingface.co/api/models/*) printf 200; exit 0 ;; esac
exec {curl} "$@"
"""
# server/app.py's stand-in: /health and /v1/models on --host/--port, SIGTERM ends it
STUB_APP = r'''
import http.server, json, signal, sys
a = sys.argv[1:]
host = a[a.index("--host") + 1] if "--host" in a else "127.0.0.1"
port = int(a[a.index("--port") + 1])
mode = "setup" if "--setup" in a else "serve"
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body = {"/health": {"status": "ok", "mode": mode},
                "/v1/models": {"data": [{"id": "stub"}]}}.get(self.path.split("?")[0])
        self.send_response(200 if body else 404)
        self.end_headers()
        self.wfile.write(json.dumps(body or {}).encode())
    def log_message(self, *args):
        pass
signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
print("stub engine:", mode, port, flush=True)
http.server.HTTPServer((host, port), H).serve_forever()
'''


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def install(name: str, setup: bool, port: int, extra: tuple = ()) -> dict | None:
    """install.sh [--setup] --no-start into a HOME of its own on test_fresh_install's release and
    stand-ins; a plain install's downloads are stand-in files. None off Linux."""
    if platform.system() != "Linux" or F._box() is None:
        return None
    rel, stubs, ref, tmp = (F._BOX["_"][k] for k in ("rel", "stubs", "ref", "tmp"))
    home = os.path.join(tmp, name)
    d = os.path.join(home, "TandemLLM")
    py = F._PY if setup else _PLAIN_PY
    F._write(os.path.join(d, "venv/bin/python"), py.format(py=sys.executable).encode())
    os.chmod(os.path.join(d, "venv/bin/python"), 0o755)
    for f in ("config.json", "tokenizer.json"):
        F._write(os.path.join(home, ".qwen38-spark-engine/models/qwen3.5-2b", f), b"{}")
    mine = os.path.join(home, "stubs")
    F._write(os.path.join(mine, "curl"), _CURL.format(curl=shutil.which("curl")).encode())
    os.chmod(os.path.join(mine, "curl"), 0o755)
    env = {"HOME": home, "PATH": mine + ":" + os.path.join(stubs, "bin") + ":" + os.environ.get("PATH", "/usr/bin:/bin"),
           "CUDA_HOME": os.path.join(stubs, "cuda"), "LANG": "C.UTF-8"}
    args = ["bash", os.path.join(ROOT, "install.sh"), "--no-start", "--yes", "--no-corpus", "--dir", d,
            "--repo", rel, "--ref", ref, "--port", str(port), "--host", "127.0.0.1", *extra]
    if setup:
        args.append("--setup")
    r = subprocess.run(args, env=env, capture_output=True, text=True, timeout=600, stdin=subprocess.DEVNULL)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    # no status responder in these boxes (OPS-175 gives every install one): it outlives the stub
    # engine's stop on purpose, and the tests leave no process behind
    with open(os.path.join(d, "run", "local.env"), "a") as fh:
        fh.write("QSE_STATUS_PORT=\n")
    return {"home": home, "dir": d, "env": env, "out": r.stdout, "port": port, "args": args}


def test_the_rendered_configs_load():
    for setup in (False, True):
        box = install("ctl-render-" + ("setup" if setup else "plain"), setup, free_port())
        if box is None:
            print("    (skipped: install.sh runs on Linux only -- run it on the box)")
            return
        src = os.path.join(box["dir"], "src")
        assert not os.path.exists(os.path.join(src, "ops", "active-profile"))
        assert os.path.exists(os.path.join(src, "run", "setup-pending")) == setup
        r = _load(os.path.join(box["dir"], "run/ops/serve.env"), box["env"])
        assert r.returncode == 0 and "loaded QSE_PROFILE_NAME= " in r.stdout, (setup, r.stdout + r.stderr)


def stub_engine(box: dict) -> dict:
    """server/app.py of the install's checkout replaced by STUB_APP; the environment bin/tandem runs
    in: a process table of its own (the box's engine is never seen), no memory check, no chatd."""
    with open(os.path.join(box["dir"], "src", "server", "app.py"), "w") as fh:
        fh.write(STUB_APP)
    proc = os.path.join(box["home"], "proc")
    os.makedirs(proc, exist_ok=True)
    return dict(box["env"], ENGINE_PROC=proc, TANDEM_SKIP_MEMCHECK="1", QSE_CHATD="off")


def adopt(box: dict, env: dict) -> int:
    """The stub engine's pid, entered in the test's process table so stop.sh finds it."""
    pid = int(open(os.path.join(box["dir"], "src", "logs", "engine.pid")).read())
    link = os.path.join(env["ENGINE_PROC"], str(pid))
    if not os.path.lexists(link):
        os.symlink(f"/proc/{pid}", link)
    return pid


def tandem(box: dict, env: dict, *args, timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run([os.path.join(box["dir"], "bin", "tandem"), *args], capture_output=True, text=True,
                          env=env, timeout=timeout, stdin=subprocess.DEVNULL)


def _gone(pid: int) -> bool:
    for _ in range(50):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.1)
    return False


def _cycle(box: dict, mode: str) -> None:
    env = stub_engine(box)
    pid = None
    try:
        r = tandem(box, env, "start")
        assert r.returncode == 0, r.stdout + r.stderr
        pid = adopt(box, env)
        r = tandem(box, env, "status")
        assert r.returncode == 0 and f"[tandem] healthy on :{box['port']}" in r.stdout, r.stdout + r.stderr
        assert json.loads(subprocess.run(["curl", "-s", f"http://127.0.0.1:{box['port']}/health"],
                                         capture_output=True, text=True).stdout)["mode"] == mode
        r = tandem(box, env, "env")
        assert r.returncode == 0 and f"PORT={box['port']}" in r.stdout
        assert f"stub engine: {mode} {box['port']}" in logs(os.path.join(box["dir"], "bin", "tandem"), env)
        r = tandem(box, env, "stop", "20")
        assert r.returncode == 0 and "[stop] stopped" in r.stdout, r.stdout + r.stderr
        assert _gone(pid), "the stub engine is still alive"
        r = tandem(box, env, "status")
        assert r.returncode == 1 and "[tandem] not running" in r.stdout
    finally:
        if pid is not None and not _gone(pid):
            os.kill(pid, signal.SIGKILL)


def test_a_plain_install_starts():
    box = install("ctl-plain", False, free_port())
    if box is None:
        print("    (skipped: install.sh runs on Linux only -- run it on the box)")
        return
    _cycle(box, "serve")
    log = open(os.path.join(box["dir"], "src", "logs", "engine.log")).read()
    assert "--nvfp4" in log or "stub engine: serve" in log


def test_a_setup_install_starts():
    box = install("ctl-setup", True, free_port())
    if box is None:
        print("    (skipped: install.sh runs on Linux only -- run it on the box)")
        return
    _cycle(box, "setup")


def test_a_rerun_rewrites_the_config():
    port = free_port()
    box = install("ctl-rerun", True, port)
    if box is None:
        print("    (skipped: install.sh runs on Linux only -- run it on the box)")
        return
    serve = os.path.join(box["dir"], "run/ops/serve.env")
    ctl = os.path.join(box["dir"], "bin/tandem")
    # a box installed before the fix: the old line and the old script
    old = open(serve).read().replace(FIXED, UNFIXED)
    with open(serve, "w") as fh:
        fh.write(old)
    with open(ctl, "w") as fh:
        fh.write(re.sub(r"\ntrap '[^\n]*' ERR\n", "\n", open(ctl).read()))
    assert _load(serve, box["env"]).returncode == 1
    r = subprocess.run(box["args"], env=box["env"], capture_output=True, text=True,
                       timeout=600, stdin=subprocess.DEVNULL)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert FIXED in open(serve).read() and _load(serve, box["env"]).returncode == 0
    assert "' ERR" in open(ctl).read()


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:60s} ok")
            passed += 1
    print(f"{passed} passed")
