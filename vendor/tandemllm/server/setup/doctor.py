"""tandem doctor (OPS-171): check an install or a checkout, repair what is safe to repair, report the rest.

A normal user has nobody to fix a box. On 2026-10-17 a Setup Guide install failed because a deploy had
written ops/start.sh with mode 644; nothing on the box could say so or put it right. The doctor runs
before every Setup Guide apply (the job's check step, server/setup/apply.py) and every `tandem start`
(bin/tandem, written by install.sh), and on request:

    python -m server.setup.doctor [--check] [--json] [--quiet] [--profile P] [--need-gb N]
    bash ~/TandemLLM/bin/tandem doctor [--check]

THE CHECKS, and what is repaired (only file modes, the run/ops copies, pid files, the hold pause and
the setup marker; secrets, weights and configuration are reported, never changed):

    modes        every file of ops/executables.txt (git's 100755 list) is executable    chmod +x
    copies       an install's run/ops has every script install.sh copies (OPS-161)   a missing one
                 copied from src/ops, 755; one that differs from src/ops is reported (re-run install)
    control      an install's bin/tandem is there and executable                       chmod 755
    config       serve.env sources under `set -u` (as start.sh does), the line that fails named
    secrets      the secrets file the engine reads (OPS-153: QSE_SECRETS, $QSE_STATE_DIR, the
                 default) is mode 600                                                  chmod 600
                 one in another place than the engine reads is reported
    python       the engine's interpreter (serve.env PY) runs; chatd's (CHATD_PY, else PY) runs,
                 unless QSE_CHATD=off
    pids         logs/engine.pid and chatd's pid file name a live process of theirs      removed
    pause        a hold's .watchdog.off older than WATCHDOG_PAUSE_MAX (2400 s)          removed
    marker       run/setup-pending names a HOST start.sh takes                rewritten 127.0.0.1
    disk         room for the chosen weights (--need-gb, or --profile's missing download) and the
                 60 GB margin use-profile.sh keeps

Each run appends what it found and did to logs/doctor.log. Exit status 0: nothing left to report.
Standard library only, except --profile, which reads the model registry.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#: install.sh's copy list (tests/test_doctor.py keeps them equal)
COPIES = ("start.sh", "stop.sh", "engines.sh", "use-profile.sh", "make-secrets.sh", "engine_status.py",
          "fetch_weights.py", "roles.sh", "watchdog.sh", "hold.sh")
MANIFEST = os.path.join("ops", "executables.txt")
PAUSE_MAX_S = 2400
DISK_MARGIN_GB = 60
HOST_OK = re.compile(r"^[0-9.:a-fA-F]+$")


class Report:
    def __init__(self):
        self.checks: list[dict] = []

    def add(self, cid: str, state: str, text: str) -> None:
        self.checks.append({"id": cid, "state": state, "text": text})

    @property
    def repaired(self) -> list[str]:
        return [c["text"] for c in self.checks if c["state"] == "repaired"]

    @property
    def problems(self) -> list[str]:
        return [c["text"] for c in self.checks if c["state"] == "problem"]

    def body(self) -> dict:
        return {"ok": not self.problems, "repaired": self.repaired, "problems": self.problems,
                "checks": self.checks, "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def layout(root: str) -> dict:
    """The tree's shape: an install (DIR/src with DIR/.tandemllm-install: the scripts and serve.env in
    DIR/run/ops, the control script in DIR/bin) or a checkout (ops/ beside the code)."""
    root = os.path.realpath(root)
    up = os.path.dirname(root)
    if os.path.isfile(os.path.join(up, ".tandemllm-install")):
        return {"root": root, "install": up, "ops": os.path.join(up, "run", "ops"),
                "control": os.path.join(up, "bin", "tandem")}
    return {"root": root, "install": None, "ops": os.path.join(root, "ops"), "control": None}


def _executable(mode: int) -> int:
    """`mode` with x wherever r is: u, g and o each as far as they may read."""
    return mode | ((mode & 0o444) >> 2)


def _make_executable(p: str, rep: Report, repair: bool, rel: str, cid: str = "modes") -> None:
    try:
        st = os.stat(p)
    except OSError:
        rep.add(cid, "problem", f"{rel} is missing")
        return
    if st.st_mode & stat.S_IXUSR:
        return
    if not repair:
        rep.add(cid, "problem", f"{rel} lost its execute bit (mode {stat.S_IMODE(st.st_mode):o})")
        return
    try:
        os.chmod(p, _executable(stat.S_IMODE(st.st_mode)))
        rep.add(cid, "repaired", f"{rel}: execute bit restored (was {stat.S_IMODE(st.st_mode):o})")
    except OSError as exc:
        rep.add(cid, "problem", f"{rel} lost its execute bit and cannot be changed ({exc.strerror})")


def check_modes(lay: dict, rep: Report, repair: bool) -> None:
    try:
        with open(os.path.join(lay["root"], MANIFEST)) as fh:
            files = [ln.strip() for ln in fh if ln.strip() and not ln.startswith("#")]
    except OSError:
        rep.add("modes", "problem", f"{MANIFEST} is missing: the release's file modes cannot be checked")
        return
    before = len(rep.checks)
    for rel in files:
        if os.path.exists(os.path.join(lay["root"], rel)):
            _make_executable(os.path.join(lay["root"], rel), rep, repair, rel)
    if len(rep.checks) == before:
        rep.add("modes", "ok", f"{len(files)} executable files")


def check_copies(lay: dict, rep: Report, repair: bool) -> None:
    if not lay["install"]:
        return
    src, run = os.path.join(lay["root"], "ops"), lay["ops"]
    before = len(rep.checks)
    for s in COPIES:
        a, b = os.path.join(src, s), os.path.join(run, s)
        rel = f"run/ops/{s}"
        if not os.path.exists(b):
            if not os.path.exists(a):
                rep.add("copies", "problem", f"{rel} is missing, and so is src/ops/{s}")
            elif repair:
                os.makedirs(run, exist_ok=True)
                shutil.copyfile(a, b)
                os.chmod(b, 0o755)
                rep.add("copies", "repaired", f"{rel} was missing: copied from src/ops")
            else:
                rep.add("copies", "problem", f"{rel} is missing")
            continue
        try:
            same = open(a, "rb").read() == open(b, "rb").read()
        except OSError:
            same = True
        if not same:
            rep.add("copies", "problem", f"{rel} differs from src/ops/{s}: run the installer again")
        _make_executable(b, rep, repair, rel, "copies")
    if lay["control"]:
        if os.path.exists(lay["control"]):
            _make_executable(lay["control"], rep, repair, "bin/tandem", "control")
        else:
            rep.add("control", "problem", "bin/tandem is missing: run the installer again")
    if len(rep.checks) == before:
        rep.add("copies", "ok", f"run/ops has the {len(COPIES)} scripts")


def serve_env(lay: dict, keys: list[str], timeout: float = 15) -> tuple[dict, str | None]:
    """serve.env's values for `keys`, sourced under `set -u` as start.sh does, and the error (the
    shell's own words) when it does not source."""
    envf = os.path.join(lay["ops"], "serve.env")
    if not os.path.isfile(envf):
        return {}, f"{envf} is missing"
    script = ('set -u; set -a; . "$ENVF" || exit 3; set +a; for k in "$@"; do '
              'if [ -n "${!k+x}" ]; then printf "%s\\t%s\\n" "$k" "${!k}"; fi; done')
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": os.path.expanduser("~"), "ENVF": envf}
    try:
        r = subprocess.run(["bash", "-c", script, "-", *keys], capture_output=True, text=True,
                           timeout=timeout, env=env)
    except (OSError, subprocess.SubprocessError) as exc:
        return {}, str(exc)
    out = {}
    for line in r.stdout.splitlines():
        k, _, v = line.partition("\t")
        out[k] = v
    err = None
    if r.returncode != 0:
        err = (r.stderr.strip().splitlines() or [f"exit {r.returncode}"])[-1]
    return out, err


def check_config(lay: dict, rep: Report, env: dict, err: str | None) -> None:
    if err:
        rep.add("config", "problem", f"serve.env does not load: {err}")
    else:
        rep.add("config", "ok", "serve.env loads")


def secrets_paths(env: dict) -> tuple[str, list[str]]:
    """The file the engine reads (start.sh's order) and the other places a secrets file may sit."""
    home = os.path.expanduser("~")
    default = os.path.join(home, ".qwen38-spark-engine", "secrets.env")
    want = os.environ.get("QSE_SECRETS") or env.get("QSE_SECRETS") or (
        os.path.join(env["QSE_STATE_DIR"], "secrets.env") if env.get("QSE_STATE_DIR") else default)
    want = os.path.expanduser(want)
    others = [p for p in {default, os.path.expanduser(env.get("QSE_SECRETS") or default)} if p != want]
    return want, others


def check_secrets(rep: Report, env: dict, repair: bool) -> None:
    want, others = secrets_paths(env)
    if os.path.exists(want):
        mode = stat.S_IMODE(os.stat(want).st_mode)
        if mode & 0o077:
            if repair:
                os.chmod(want, 0o600)
                rep.add("secrets", "repaired", f"{want}: mode {mode:o} -> 600")
            else:
                rep.add("secrets", "problem", f"{want} is readable by others (mode {mode:o})")
        else:
            rep.add("secrets", "ok", f"the engine reads {want}")
        return
    stray = [p for p in others if os.path.exists(p)]
    if stray:
        rep.add("secrets", "problem", f"the engine reads {want}, which is missing; a secrets file sits at "
                                      f"{stray[0]} instead (move it there, or set QSE_SECRETS)")
    else:
        rep.add("secrets", "ok", "no secrets file (no admin password, no metrics token)")


def _runs(py: str, code: str = "import sys", timeout: float = 30) -> str | None:
    if not py:
        return "not set"
    exe = shutil.which(py) if os.sep not in py else py
    if not exe or not os.path.exists(exe):
        return f"{py} does not exist"
    try:
        r = subprocess.run([exe, "-c", code], capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as exc:
        return f"{py} does not run ({exc})"
    return None if r.returncode == 0 else f"{py} fails: {(r.stderr.strip().splitlines() or ['?'])[-1]}"


def check_python(rep: Report, env: dict) -> None:
    why = _runs(env.get("PY", ""), "import sys; assert sys.version_info >= (3, 11), sys.version")
    rep.add("python", "problem" if why else "ok", f"the engine's interpreter: {why}" if why else
            f"the engine's interpreter runs ({env.get('PY')})")
    if (os.environ.get("QSE_CHATD") or env.get("QSE_CHATD") or "on") == "off":
        return
    py = os.environ.get("CHATD_PY") or env.get("CHATD_PY") or env.get("PY", "")
    why = _runs(py, "import sys; assert sys.version_info >= (3, 9), sys.version")
    if why:
        rep.add("python", "problem", f"chatd's interpreter: {why}")


def _cmdline(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as fh:
            return fh.read().replace(b"\0", b" ").decode("utf-8", "replace")
    except OSError:
        try:
            return subprocess.run(["ps", "-o", "args=", "-p", str(pid)], capture_output=True, text=True,
                                  timeout=5).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _stale_pid(path: str, marker: str, rel: str, rep: Report, repair: bool) -> None:
    try:
        pid = int(open(path).read().strip() or 0)
    except (OSError, ValueError):
        if os.path.exists(path):
            if repair:
                os.unlink(path)
                rep.add("pids", "repaired", f"{rel} held no pid: removed")
            else:
                rep.add("pids", "problem", f"{rel} holds no pid")
        return
    if pid > 0 and _alive(pid) and marker in _cmdline(pid):
        return
    if repair:
        os.unlink(path)
        rep.add("pids", "repaired", f"{rel} named pid {pid}, which is not running it: removed")
    else:
        rep.add("pids", "problem", f"{rel} names pid {pid}, which is not running it")


def check_pids(lay: dict, rep: Report, env: dict, repair: bool) -> None:
    n = len(rep.checks)
    _stale_pid(os.path.join(lay["root"], "logs", "engine.pid"), "server/app.py", "logs/engine.pid", rep, repair)
    data = os.path.expanduser(os.environ.get("QSE_CHAT_DATA") or env.get("QSE_CHAT_DATA") or "~/.tandemllm/chat")
    _stale_pid(os.path.join(data, "chatd.pid"), "chatd", "chatd's pid file", rep, repair)
    pause = os.path.join(lay["root"], ".watchdog.off")
    if os.path.exists(pause):
        age = time.time() - os.stat(pause).st_mtime
        limit = float(os.environ.get("WATCHDOG_PAUSE_MAX") or PAUSE_MAX_S)
        if age >= limit:
            if repair:
                os.unlink(pause)
                rep.add("pause", "repaired", f"a hold's pause of {age / 3600:.1f} h ago was left behind: removed")
            else:
                rep.add("pause", "problem", f"a hold's pause of {age / 3600:.1f} h ago was left behind")
        else:
            rep.add("pause", "problem", "a hold owns the board now (.watchdog.off): starts wait for it")
    if len(rep.checks) == n:
        rep.add("pids", "ok", "no stale pid file")


def check_marker(lay: dict, rep: Report, repair: bool) -> None:
    p = os.path.join(lay["root"], "run", "setup-pending")
    if not os.path.lexists(p):
        return
    if not os.path.isfile(p):
        rep.add("marker", "problem", "run/setup-pending is not a file")
        return
    host = None
    for ln in open(p, errors="replace").read().splitlines():
        if ln.startswith("HOST="):
            host = ln[5:].strip()
            break
    if host is None or HOST_OK.match(host):
        rep.add("marker", "ok", f"setup mode on {host or '127.0.0.1'} until the setup is done")
        return
    if repair:
        with open(p, "w") as fh:
            fh.write("HOST=127.0.0.1\n")
        rep.add("marker", "repaired", f"run/setup-pending named the address {host!r}, which start.sh refuses: "
                                      f"now 127.0.0.1")
    else:
        rep.add("marker", "problem", f"run/setup-pending names the address {host!r}, which start.sh refuses")


def free_gb(path: str) -> float:
    while not os.path.exists(path):
        path = os.path.dirname(path)
    return shutil.disk_usage(path).free / 1e9


def check_disk(rep: Report, need_gb: float | None, where: str) -> None:
    if not need_gb:
        return
    free = free_gb(where)
    if free < need_gb + DISK_MARGIN_GB:
        rep.add("disk", "problem", f"the download needs {need_gb:g} GB and {DISK_MARGIN_GB} GB left after it; "
                                   f"the disk has {free:.0f} GB free")
    else:
        rep.add("disk", "ok", f"{free:.0f} GB free for a {need_gb:g} GB download")


def profile_need_gb(profile: str) -> float:
    """The GB the profile still has to download (the plan's figure): 0 when its weights are here."""
    from server.setup import device, registry
    for m in registry.load_all()[0].values():
        prof = registry.profile(m, profile)
        if prof is None:
            continue
        disk = device.on_disk(m, fresh=True).get(profile, {})
        return 0.0 if disk.get("weights") else float(prof.get("disk_gb") or 0.0)
    return 0.0


def run(root: str = ROOT, *, repair: bool = True, need_gb: float | None = None, log: bool = True) -> dict:
    lay = layout(root)
    rep = Report()
    check_modes(lay, rep, repair)
    check_copies(lay, rep, repair)
    env, err = serve_env(lay, ["PY", "QSE_SECRETS", "QSE_STATE_DIR", "QSE_CHATD", "CHATD_PY", "QSE_CHAT_DATA",
                               "REPO"])
    check_config(lay, rep, env, err)
    check_secrets(rep, env, repair)
    check_python(rep, env)
    check_pids(lay, rep, env, repair)
    check_marker(lay, rep, repair)
    check_disk(rep, need_gb, os.path.expanduser("~"))
    body = rep.body()
    if log:
        try:
            os.makedirs(os.path.join(lay["root"], "logs"), exist_ok=True)
            with open(os.path.join(lay["root"], "logs", "doctor.log"), "a") as fh:
                for c in body["checks"]:
                    if c["state"] != "ok":
                        fh.write(f"{body['at']} {c['state']}: {c['text']}\n")
                fh.write(f"{body['at']} {'ok' if body['ok'] else 'problems'}: {len(body['repaired'])} repaired, "
                         f"{len(body['problems'])} to look at, {len(body['checks'])} checks\n")
        except OSError:
            pass
    return body


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tandem doctor", description="check this install and repair what is safe")
    ap.add_argument("--check", action="store_true", help="report only, change nothing")
    ap.add_argument("--json", action="store_true", help="the report as JSON")
    ap.add_argument("--quiet", action="store_true", help="print only what was repaired and what is wrong")
    ap.add_argument("--profile", help="check the disk for this weight profile's download")
    ap.add_argument("--need-gb", type=float, help="check the disk for a download of this many GB")
    ap.add_argument("--root", default=ROOT, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    need = a.need_gb
    if a.profile and need is None:
        try:
            need = profile_need_gb(a.profile)
        except Exception as exc:                                  # noqa: BLE001
            print(f"[doctor] the registry did not load ({type(exc).__name__}): no disk check for {a.profile}")
    body = run(a.root, repair=not a.check, need_gb=need)
    if a.json:
        print(json.dumps(body, indent=1))
    else:
        for c in body["checks"]:
            if c["state"] == "ok" and a.quiet:
                continue
            print(f"[doctor] {c['state'] if c['state'] != 'ok' else 'ok'}: {c['text']}")
        if not a.quiet or not body["ok"]:
            print(f"[doctor] {len(body['repaired'])} repaired, {len(body['problems'])} to look at"
                  + ("" if body["ok"] else "; details in logs/doctor.log"))
    return 0 if body["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
