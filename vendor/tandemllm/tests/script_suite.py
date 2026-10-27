"""The CPU suite in script mode, with a count of what each file ran.

    python tests/script_suite.py tests/test_x.py [args]   one file, its test functions counted
    python tests/script_suite.py --all [--out DIR] [--timeout S] [files...]
                                                         every tests/test_*.py, one process each

A file ran its tests when its `test_*` functions were called (counted here: the module's names are
wrapped as they are defined) or when it says so itself (`N passed` or unittest's `Ran N tests`,
N > 0, the last such line: a module that checks at import, tests/test_relax.py; a TestCase module). A file that exits 0 having run none FAILS:
until 0.13 (SRV-661) sixteen files had no `__main__` entry, ran nothing on the box and were counted as passed
(tests/scriptmode.py is their runner now); a file that runs none on purpose prints `skipped: <why>`
and is listed as skipped. `--all` runs each file in a clean environment (no
serve.env, no QWEN38_*, CUDA_VISIBLE_DEVICES empty), prints a line a file and a total, and exits 1
when any file failed or ran nothing. Standard library only.
"""

from __future__ import annotations

import functools
import glob
import os
import re
import subprocess
import sys
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
COUNT_LINE = "[script-suite] test functions run: "
#: a file that runs nothing here on purpose says so on a line of its own: `skipped: <why>` (a GPU
#: kernel test on the CPU); the suite lists it apart, never as passed
SKIPPED = re.compile(r"(?mi)^skipped:?\s*(.*)$")
PASSED = re.compile(r"(?m)^\s*(\d+) passed\b|^Ran (\d+) tests? in ")


class _Counted(dict):
    """A module namespace that wraps each test function of `path` as it is defined."""

    def __init__(self, path: str, counter: list, module, **kw):
        super().__init__(**kw)
        self.path, self.counter, self.module = path, counter, module
        module.__dict__.update(kw)

    def __setitem__(self, key, value):
        if (isinstance(key, str) and key.startswith("test_") and isinstance(value, types.FunctionType)
                and value.__code__.co_filename == self.path):
            value = self._wrap(value)
        super().__setitem__(key, value)
        # `sys.modules["__main__"]` is the module as the file defines it, as it is under a plain
        # `python file.py`: unittest.main() finds the TestCase classes there, pickle the classes
        self.module.__dict__[key] = value

    def __delitem__(self, key):
        super().__delitem__(key)
        self.module.__dict__.pop(key, None)

    def _wrap(self, fn):
        counter = self.counter

        @functools.wraps(fn)
        def counted(*a, **kw):
            counter[0] += 1
            return fn(*a, **kw)
        return counted


def run_one(path: str, argv: list) -> int:
    path = os.path.abspath(path)
    counter = [0]
    sys.argv = [path] + argv
    sys.path.insert(0, os.path.dirname(path))
    with open(path, "rb") as fh:
        code = compile(fh.read(), path, "exec")
    mod = types.ModuleType("__main__")
    g = _Counted(path, counter, mod, __name__="__main__", __file__=path, __builtins__=__builtins__,
                 __package__=None, __spec__=None)
    sys.modules["__main__"] = mod
    rc = 0
    try:
        exec(code, g)
    except SystemExit as e:
        rc = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    finally:
        sys.stdout.flush()
        print(f"{COUNT_LINE}{counter[0]}", flush=True)
    return rc


def ran(log: str) -> int:
    """How many tests a file's log says it ran: the counted calls, else its own `N passed`."""
    n = 0
    for line in log.splitlines():
        if line.startswith(COUNT_LINE):
            n = int(line[len(COUNT_LINE):] or 0)
    said = [int(a or b) for a, b in PASSED.findall(log)]     # `N passed`, unittest's `Ran N tests`
    return max(n, said[-1] if said else 0)


def run_all(args: list) -> int:
    out, timeout, files = None, 900, []
    it = iter(args)
    for a in it:
        if a == "--out":
            out = next(it)
        elif a == "--timeout":
            timeout = int(next(it))
        else:
            files.append(a)
    files = files or sorted(glob.glob(os.path.join(ROOT, "tests", "test_*.py")))
    if out:
        os.makedirs(out, exist_ok=True)
    env = {k: v for k, v in os.environ.items() if k in ("HOME", "PATH", "LANG", "TMPDIR", "USER")}
    env.update(PYTHONPATH=os.pathsep.join(p for p in (ROOT, os.environ.get("SUITE_PYTHONPATH")) if p),
               CUDA_VISIBLE_DEVICES="", TZ="Europe/Berlin", PYTHONDONTWRITEBYTECODE="1")
    bad, skipped, total = [], [], 0
    for f in files:
        t0 = time.time()
        try:
            p = subprocess.run([sys.executable, "-u", __file__, f], cwd=ROOT, env=env, timeout=timeout,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            rc, log = p.returncode, p.stdout
        except subprocess.TimeoutExpired as e:
            rc, log = 124, (e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
        n = ran(log)
        total += n
        skip = SKIPPED.search(log) if rc == 0 and n == 0 else None
        verdict = ("ok" if rc == 0 and n > 0 else
                   f"SKIPPED ({skip.group(1).strip()[:80]})" if skip else
                   "FAIL (ran no test)" if rc == 0 else f"FAIL (exit {rc})")
        if skip:
            skipped.append(os.path.relpath(f, ROOT))
        elif verdict != "ok":
            bad.append(os.path.relpath(f, ROOT))
        if out:
            with open(os.path.join(out, os.path.basename(f)[:-3] + ".log"), "w") as fh:
                fh.write(log)
        print(f"{os.path.relpath(f, ROOT):56s} {n:5d} tests {time.time() - t0:7.1f} s  {verdict}", flush=True)
    print(f"files={len(files)} ok={len(files) - len(bad) - len(skipped)} fail={len(bad)} "
          f"skipped={len(skipped)} tests={total}"
          + (" -- " + " ".join(bad) if bad else "") + (" -- skipped: " + " ".join(skipped) if skipped else ""),
          flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    if sys.argv[1] == "--all":
        sys.exit(run_all(sys.argv[2:]))
    sys.exit(run_one(sys.argv[1], sys.argv[2:]))
