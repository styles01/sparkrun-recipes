"""Our scripts call each other through bash, never through the execute bit (OPS-167, QA OPS-168).

On 2026-10-17 a Setup Guide install failed at "Restart the engine": `ops/use-profile.sh: line 172:
ops/start.sh: Permission denied`. use-profile.sh ran `"$HERE/start.sh"` as a program, and a deploy had
written start.sh with mode 644 (git has 100755). Every call of an in-repo script now goes through
`bash` (or the interpreter), and this file fails on a new direct call: in the shell scripts (a small
tokenizer finds the words at command position outside quotes), in the Python code's subprocess
argument vectors, in the systemd units' Exec lines and in the documented command lines (cron lines and
code blocks). The scripts themselves run for real at mode 644 in the last two tests.

The test matrix, the Gherkin of OPS-167:

  1  a switch with start.sh (and stop.sh) at mode 644 succeeds    test_a_switch_with_mode_644
  2  the watchdog restarts through stop.sh and start.sh at 644    test_the_watchdog_with_mode_644
  3  a new direct call is caught, with its file and line: the tokenizer's negative controls, and the
     tree has none                                                test_the_tokenizer, test_no_direct_call_in_the_tree
  4  the systemd units and the cron lines use bash               test_units_and_documented_lines

Run: python tests/test_no_exec_bit.py   (1 and 2 need Linux: the scripts read /proc and flock)
"""
from __future__ import annotations

import ast
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _files() -> list[str]:
    out = subprocess.run(["git", "-C", ROOT, "ls-files"], capture_output=True, text=True).stdout.split()
    if not out:                       # a copy without .git (the fresh-install tests' release)
        for d, _, names in os.walk(ROOT):
            if "node_modules" in d or "/.git" in d:
                continue
            out += [os.path.relpath(os.path.join(d, n), ROOT) for n in names]
    return [f for f in out if os.path.isfile(os.path.join(ROOT, f))]


FILES = _files()
#: the scripts of this repository by name, and the control script install.sh writes
SCRIPTS = {os.path.basename(f) for f in FILES if f.endswith(".sh")} | {"tandem"}
KEYWORDS = {"if", "then", "do", "else", "elif", "while", "until", "!", "{", "time", "exec", "nohup",
            "setsid", "command", "builtin", "env", "sudo", "nice", "ionice", "xargs"}
REDIR = re.compile(r"^\d*(<<<|<<-?|>>|<>|>&|<&|>\||>|<)(.*)$")
ASSIGN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\[[^]]*\])?\+?=")


def _is_script(word: str) -> bool:
    """A word that names one of our scripts by a path: `"$HERE/start.sh"`, `$DIR/bin/tandem`."""
    w = word.replace('"', "").replace("'", "")
    if "/" not in w:
        return False
    base = w.rstrip("/").rsplit("/", 1)[-1]
    return base in SCRIPTS and (base != "tandem" or w.endswith("bin/tandem"))


def commands(line: str) -> list[str]:
    """The words at command position of one logical shell line, outside quotes; `$(` opens a new
    command, inside double quotes too."""
    out: list[str] = []
    stack: list[str | None] = []       # the quote state a `$(` interrupted
    quote: str | None = None
    word, cmdpos, skip = "", True, 0   # skip: words that are an operator's argument
    timeout_args = 0

    def end_word():
        nonlocal word, cmdpos, skip, timeout_args
        w = word
        word = ""
        if not w:
            return
        if skip:
            skip -= 1
            return
        m = REDIR.match(w)
        if m:
            if not m.group(2):
                skip = 1               # the target is the next word
            return
        if not cmdpos:
            return
        if timeout_args:
            if w.startswith("-"):
                return
            timeout_args -= 1
            return
        if ASSIGN.match(w) or w in KEYWORDS:
            return
        if w in ("timeout", "flock"):
            timeout_args = 1
            return
        out.append(w)
        cmdpos = False

    i, n = 0, len(line)
    while i < n:
        c = line[i]
        if quote == "'":
            word += c
            if c == "'":
                quote = None
            i += 1
            continue
        if c == "\\" and i + 1 < n:
            word += line[i:i + 2]
            i += 2
            continue
        if c == "$" and line[i + 1:i + 3] == "((":
            j = line.find("))", i)
            word += line[i:(j + 2 if j >= 0 else n)]
            i = j + 2 if j >= 0 else n
            continue
        if c == "$" and line[i + 1:i + 2] == "(":
            stack.append(quote)
            quote = None
            if word:
                word += "$("            # part of the word around it: the inner command is its own
            saved_word, saved_pos = word, cmdpos
            word, cmdpos = "", True
            stack.append((saved_word, saved_pos))  # type: ignore[arg-type]
            i += 2
            continue
        if quote == '"':
            word += c
            if c == '"':
                quote = None
            i += 1
            continue
        if c in "'\"":
            quote = c
            word += c
            i += 1
            continue
        if c == "#" and not word:
            break
        if c == ")" and stack:
            end_word()
            saved_word, saved_pos = stack.pop()       # type: ignore[misc]
            quote = stack.pop()
            word, cmdpos = saved_word + ")", saved_pos
            i += 1
            continue
        if c in " \t":
            end_word()
            i += 1
            continue
        if c in ";&|()":
            end_word()
            cmdpos = True
            skip = 0
            i += 1
            continue
        word += c
        i += 1
    end_word()
    return out


#: a heredoc whose body is a script install.sh writes (bin/tandem); any other heredoc is data
SCRIPT_HEREDOCS = {"CTL"}
HEREDOC = re.compile(r"<<-?\s*['\"]?([A-Za-z_][A-Za-z0-9_]*)['\"]?")


def _logical_lines(text: str):
    """(line number, text) of each logical line: continuations joined, the bodies of data heredocs
    (messages, Python, unit files) and Exec lines of a unit left out (test 4 reads those)."""
    buf, start, until = "", 0, None
    for no, raw in enumerate(text.splitlines(), 1):
        if until is not None:
            if raw.strip() == until[0]:
                until = None
            elif until[1]:
                yield no, raw
            continue
        if not buf:
            start = no
        if raw.endswith("\\") and not raw.endswith("\\\\"):
            buf += raw[:-1] + " "
            continue
        buf += raw
        if not re.match(r"^\s*Exec\w*=", buf):
            yield start, buf
        m = HEREDOC.search(buf) if "<<<" not in buf else None
        if m:
            until = (m.group(1), m.group(1) in SCRIPT_HEREDOCS)
        buf = ""


def direct_calls(text: str, name: str) -> list[str]:
    return [f"{name}:{no}: {w}  ({ln.strip()[:120]})" for no, ln in _logical_lines(text)
            for w in commands(ln) if _is_script(w)]


def _python_calls(path: str) -> list[str]:
    try:
        tree = ast.parse(open(os.path.join(ROOT, path)).read())
    except (SyntaxError, UnicodeDecodeError):
        return []
    found = []

    def names_script(node) -> bool:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return _is_script(node.value) or (node.value.endswith(".sh") and "/" not in node.value
                                              and node.value in SCRIPTS)
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "join" and node.args:
            return names_script(node.args[-1])
        if isinstance(node, ast.JoinedStr) and node.values:
            return names_script(node.values[-1])
        return False

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        fn = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
        if fn not in ("run", "Popen", "call", "check_call", "check_output", "execv", "execvp", "execvpe",
                      "spawnv", "cmd"):
            continue
        a = node.args[0]
        if isinstance(a, (ast.List, ast.Tuple)) and a.elts and names_script(a.elts[0]):
            found.append(f"{path}:{node.lineno}: argv[0] is a script")
    return found


SHELL = [f for f in FILES if (f.endswith(".sh") or f == "install.sh") and not f.startswith("tests/")]
PYTHON = [f for f in FILES if f.endswith(".py") and not f.startswith(("tests/", "manual/"))]


def test_the_tokenizer():
    hits = lambda s: [w for w in commands(s) if _is_script(w)]   # noqa: E731
    # the incident's line and its kin: caught
    assert hits('    "$HERE/start.sh" < /dev/null 9>&-') == ['"$HERE/start.sh"']
    assert hits('STATUS_REASON="x" "$HERE/stop.sh" 120 9>&-') == ['"$HERE/stop.sh"']
    assert hits('"$HERE/stop.sh" 30 >> "$WD" 2>&1') == ['"$HERE/stop.sh"']
    assert hits('x && $RUN/ops/start.sh || y') == ["$RUN/ops/start.sh"]
    assert hits('if "$DIR/bin/tandem" start; then :; fi') == ['"$DIR/bin/tandem"']
    assert hits('R="$("$HERE/use-profile.sh" --list)"') == ['"$HERE/use-profile.sh"']
    assert hits('nohup setsid "$HERE/watchdog.sh" &') == ['"$HERE/watchdog.sh"']
    assert hits('timeout 30 "$HERE/stop.sh"') == ['"$HERE/stop.sh"']
    # through bash, sourced, named in a string or as an argument: not calls
    for ok in ('bash "$HERE/start.sh" < /dev/null', '. "$HERE/engines.sh"', 'source "$HERE/engines.sh"',
               'say "dry run: lock; $HERE/stop.sh 120; then $HERE/start.sh"', '[ -f "$REPO/ops/chatd.sh" ] || return 0',
               'CHATD="$REPO/ops/chatd.sh"', 'install -m 755 "$SRC/ops/$s" "$RUN/ops/$s"',
               'STATUS_REASON="x" bash "$HERE/stop.sh" 120', 'run bash "$DIR/bin/tandem" stop 120',
               'grep -q "$DIR/bin/tandem" "$UNIT"', 'echo "$(cat "$HERE/start.sh")"'):
        assert hits(ok) == [], (ok, hits(ok))
    # a continuation line and a unit's Exec line
    assert direct_calls('foo && \\\n  "$HERE/stop.sh" 30\n', "x") == \
        ['x:1: "$HERE/stop.sh"  (foo &&    "$HERE/stop.sh" 30)']
    assert direct_calls("ExecStart=$DIR/bin/tandem start\n", "u") == []
    assert direct_calls("cat <<EOF\n  Watchdog:  $RUN/ops/watchdog.sh\nEOF\n", "m") == []
    assert len(direct_calls("cat <<'CTL'\nstop) \"$RUN/ops/stop.sh\" 60 ;;\nCTL\n", "c")) == 1
    # Python: an argv whose first word is a script
    with tempfile.NamedTemporaryFile("w", suffix=".py", dir=ROOT, delete=False) as fh:
        fh.write('import os, subprocess\nsubprocess.run([os.path.join(OPS, "start.sh")])\n'
                 'subprocess.Popen(["bash", os.path.join(OPS, "stop.sh")])\n')
    try:
        got = _python_calls(os.path.relpath(fh.name, ROOT))
        assert len(got) == 1 and ":2:" in got[0], got
    finally:
        os.unlink(fh.name)


def test_no_direct_call_in_the_tree():
    found = []
    for f in SHELL:
        found += direct_calls(open(os.path.join(ROOT, f)).read(), f)
    for f in PYTHON:
        found += _python_calls(f)
    assert not found, "call these through bash:\n" + "\n".join(found)
    # the incident's file is among those read, and its calls are through bash now
    assert "ops/use-profile.sh" in SHELL and 'bash "$HERE/start.sh" < /dev/null 9>&-' in \
        open(os.path.join(ROOT, "ops/use-profile.sh")).read()


def _doc_lines():
    """(file, line no, text) of every command line the docs show: cron lines, lines of fenced code
    blocks and of LaTeX verbatim blocks, and indented (code) lines of the markdown files."""
    docs = [f for f in FILES if f.startswith("docs/") and f.endswith(".md")] + \
        [f for f in ("README.md", "EARLY-ACCESS.md") if f in FILES] + \
        [f for f in FILES if f.startswith("manual/chapters/") and f.endswith(".tex")]
    for f in docs:
        code = False
        for no, ln in enumerate(open(os.path.join(ROOT, f)).read().splitlines(), 1):
            s = ln.strip()
            if s.startswith("```") or re.match(r"\\(begin|end)\{(verbatim|lstlisting|minted|Verbatim)", s):
                code = not code
                continue
            if code or (f.endswith(".md") and ln.startswith("    ") and not s.startswith(("-", "|", "*"))) \
                    or re.match(r"^\s*(@reboot|\*\s+\*\s+\*\s+\*\s+\*)", ln):
                yield f, no, ln


def test_units_and_documented_lines():
    # systemd: an Exec line never runs one of our scripts as a program
    units = [f for f in FILES if f.endswith(".service")] + ["install.sh"]
    bad = []
    for f in units:
        for no, ln in enumerate(open(os.path.join(ROOT, f)).read().splitlines(), 1):
            m = re.match(r"^\s*Exec\w*=-?(\S+)", ln)
            if m and _is_script(m.group(1)):
                bad.append(f"{f}:{no}: {ln.strip()}")
    assert not bad, bad
    unit = open(os.path.join(ROOT, "install.sh")).read()
    assert "ExecStart=/bin/bash $DIR/bin/tandem start\nExecStop=/bin/bash $DIR/bin/tandem stop 120" in unit
    # the docs: a command line runs our scripts through bash
    bad = []
    for f, no, ln in _doc_lines():
        cron = re.match(r"^\s*(@reboot|\*\s+\*\s+\*\s+\*\s+\*)\s+(.*)$", ln)
        text = cron.group(2) if cron else ln
        # bin/tandem is the user's own command in the docs' examples; a cron line runs it unattended
        bad += [f"{f}:{no}: {w}" for w in commands(text) if _is_script(w)
                and (cron or not w.replace('"', "").endswith("bin/tandem"))]
    assert not bad, "documented commands that need `bash`:\n" + "\n".join(bad)
    ops = open(os.path.join(ROOT, "docs", "operations.md")).read()
    assert "    * * * * * bash $HOME/TandemLLM/run/ops/watchdog.sh\n" in ops
    assert "    @reboot sleep 90 && bash $HOME/<checkout>/ops/start.sh" in ops


# ---------------------------------------------------------------- the scripts at mode 644 (Linux)
def _linux() -> bool:
    if platform.system() != "Linux":
        print("    (skipped: the ops scripts read /proc and use flock -- run it on the box)")
        return False
    return True


def test_a_switch_with_mode_644():
    """use-profile.sh's switch against stub start.sh and stop.sh written without their execute bits
    (the incident: start.sh 644): the switch runs both through bash and succeeds; the old line, a
    direct call, fails the same way the box did (the negative control)."""
    if not _linux():
        return
    repo, env = _switch_box()
    for s in ("start.sh", "stop.sh"):
        os.chmod(os.path.join(repo, "ops", s), 0o644)
    r = subprocess.run(["bash", os.path.join(repo, "ops", "use-profile.sh"), "exl3-4.5"], env=env,
                       capture_output=True, text=True, timeout=120, stdin=subprocess.DEVNULL)
    out = r.stdout + r.stderr
    assert r.returncode == 0 and "OK: exl3-4.5 live" in out and "Permission denied" not in out, out
    assert open(os.path.join(repo, "ops", "active-profile")).read().strip() == "exl3-4.5"
    # the negative control: the line before OPS-167
    up = os.path.join(repo, "ops", "use-profile.sh")
    text = open(up).read()
    with open(up, "w") as fh:
        fh.write(text.replace('    bash "$HERE/start.sh" < /dev/null 9>&-\n}', '    "$HERE/start.sh" < /dev/null 9>&-\n}'))
    r = subprocess.run(["bash", up, "exl3-6.0"], env=env, capture_output=True, text=True, timeout=120,
                       stdin=subprocess.DEVNULL)
    assert r.returncode != 0 and "start.sh: Permission denied" in r.stdout + r.stderr, r.stdout + r.stderr
    shutil.rmtree(os.path.dirname(repo), ignore_errors=True)


def _switch_box() -> tuple[str, dict]:
    """A checkout in a temporary folder whose switch needs no engine: the real use-profile.sh,
    serve.env and profiles; start.sh and stop.sh stubs that mark what ran; an engine stand-in for the
    smoke (a /health that names the active profile and a chat that answers); two EXL3 profiles'
    weights and every asset on disk (stand-in files)."""
    tmp = tempfile.mkdtemp(prefix="exec-bit-")
    repo, home = os.path.join(tmp, "repo"), os.path.join(tmp, "home")
    ops = os.path.join(repo, "ops")
    os.makedirs(home)
    shutil.copytree(os.path.join(ROOT, "ops", "profiles"), os.path.join(ops, "profiles"))
    for f in ("use-profile.sh", "fetch_weights.py", "engines.sh"):
        shutil.copy2(os.path.join(ROOT, "ops", f), os.path.join(ops, f))
    for d in ("server", "models", "engine", "packages"):          # read only: the registry's code
        os.symlink(os.path.join(ROOT, d), os.path.join(repo, d))
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    env_text = open(os.path.join(ROOT, "ops", "serve.env")).read() \
        .replace("REPO=${HOME}/qwen38-spark-engine", f"REPO={repo}").replace("PORT=8000", f"PORT={port}")
    env_text = re.sub(r"(?m)^PY=.*$", f"PY={sys.executable}", env_text)
    with open(os.path.join(ops, "serve.env"), "w") as fh:
        fh.write(env_text)
    # the engine stand-in: start.sh starts it (in the background, its pid in the box's run/), stop.sh
    # stops it; /health names the profile start.sh was started for
    stub = os.path.join(tmp, "engine.py")
    with open(stub, "w") as fh:
        fh.write(STUB_ENGINE)
    with open(os.path.join(ops, "start.sh"), "w") as fh:
        fh.write(f'#!/bin/bash\nHERE="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"\n'
                 f'set -a; . "$HERE/serve.env"; set +a\n'
                 f'setsid {sys.executable} {stub} {port} "$QSE_WEIGHTS" > {tmp}/engine.log 2>&1 < /dev/null &\n'
                 f'echo $! > {tmp}/engine.pid\n'
                 f'for i in $(seq 1 50); do curl -sf -m 1 http://127.0.0.1:{port}/health >/dev/null && exit 0; sleep 0.2; done\nexit 1\n')
    with open(os.path.join(ops, "stop.sh"), "w") as fh:
        fh.write(f'#!/bin/bash\n[ -f {tmp}/engine.pid ] && kill "$(cat {tmp}/engine.pid)" 2>/dev/null; '
                 f'rm -f {tmp}/engine.pid; sleep 0.3; echo "[stop] stopped"\n')
    sys.path.insert(0, ROOT)
    from server.setup import registry
    m = registry.get("qwen3.8-27b")
    for a in m["assets"]:
        for p in [a["path"], *a.get("also", [])]:
            p = p.replace("${HOME}", home).replace("${REPO}", repo).replace(
                "${HF_HUB}", os.path.join(home, ".cache", "huggingface", "hub")).replace("*", "snap")
            os.makedirs(os.path.dirname(p), exist_ok=True)
            open(p, "w").write("x")
    for bpw in ("4.5", "6.0"):
        d = os.path.join(home, "exl3-quants", bpw)
        os.makedirs(d)
        open(os.path.join(d, "config.json"), "w").write("{}")
        open(os.path.join(d, "model.safetensors"), "w").write("w")
    with open(os.path.join(ops, "active-profile"), "w") as fh:
        fh.write("exl3-6.0\n")
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": home, "LANG": "C.UTF-8",
           "MEM_NEED_GB": "0", "DISK_MARGIN_GB": "0", "BOX_LOCK": os.path.join(tmp, "box.flock"),
           "ENGINE_PROC": os.path.join(tmp, "proc"), "STATUS_PY": "true"}
    os.makedirs(env["ENGINE_PROC"])
    return repo, env


STUB_ENGINE = r'''
import http.server, json, signal, sys
port, name = int(sys.argv[1]), sys.argv[2]
class H(http.server.BaseHTTPRequestHandler):
    def _j(self, body):
        b = json.dumps(body).encode(); self.send_response(200)
        self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b)))
        self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        self._j({"status": "ok", "weights": {"name": name}})
    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        self._j({"choices": [{"message": {"content": "ready"}}]})
    def log_message(self, *a):
        pass
signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
http.server.HTTPServer(("127.0.0.1", port), H).serve_forever()
'''


def test_the_watchdog_with_mode_644():
    """watchdog.sh after three failed checks, with stop.sh and start.sh at mode 644: it restarts
    through bash and logs "restarted"."""
    if not _linux():
        return
    tmp = tempfile.mkdtemp(prefix="exec-bit-wd-")
    ops = os.path.join(tmp, "ops")
    os.makedirs(ops)
    shutil.copy2(os.path.join(ROOT, "ops", "watchdog.sh"), ops)
    shutil.copy2(os.path.join(ROOT, "ops", "engines.sh"), ops)
    marks = os.path.join(tmp, "marks")
    with open(os.path.join(ops, "serve.env"), "w") as fh:
        fh.write(f"REPO={tmp}\nPORT=1\nQSE_CHATD=off\n")
    for s in ("stop.sh", "start.sh"):
        with open(os.path.join(ops, s), "w") as fh:
            fh.write(f'#!/bin/bash\necho "{s} $*" >> {marks}\n')
        os.chmod(os.path.join(ops, s), 0o644)
    os.makedirs(os.path.join(tmp, "logs"))
    open(os.path.join(tmp, "logs", "watchdog.fails"), "w").write("2\n")
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": tmp, "ENGINE_PROC": os.path.join(tmp, "proc"),
           "RESTART_SETTLE_S": "0", "BOX_LOCK": os.path.join(tmp, "box.flock")}
    os.makedirs(env["ENGINE_PROC"])
    r = subprocess.run(["bash", os.path.join(ops, "watchdog.sh")], env=env, capture_output=True, text=True,
                       timeout=60)
    log = open(os.path.join(tmp, "logs", "watchdog.log")).read()
    assert r.returncode == 0 and "restarted" in log and "RESTART FAILED" not in log, log + r.stderr
    assert open(marks).read().splitlines() == ["stop.sh 30", "start.sh "], open(marks).read()
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            t0 = time.time()
            fn()
            print(f"  {name:60s} ok ({time.time() - t0:.1f} s)")
            passed += 1
    print(f"{passed} passed")
