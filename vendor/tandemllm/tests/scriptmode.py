"""Script mode for a test module written for pytest: `python tests/test_x.py` runs its tests.

Every tests/test_*.py is a standalone script on the box (CONTRIBUTING.md; ops/gate.sh, the release
suite), and a module without a `__main__` entry ran nothing there while the suite counted it as
passed (16 files in 0.13.0's tree, SRV-661, among them the Ornith trunk's and the chatd file-text tests).
A module ends with

    if __name__ == "__main__":
        from scriptmode import run
        run(globals())

and `run` calls each `test_*` function in name order, with the few pytest fixtures these modules
take (`tmp_path`, `monkeypatch`, `capsys`; an argument with a default keeps it), prints one line a
test and `N passed`, and exits non-zero on a failure or when it ran no test at all. A test that
raises pytest's Skipped (or `scriptmode.Skip`) is counted as skipped.
"""

from __future__ import annotations

import inspect
import io
import os
import pathlib
import shutil
import sys
import tempfile
import traceback
from contextlib import redirect_stderr, redirect_stdout


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    # pytest puts the checkout on the path (tests/conftest.py's rootdir); a script has its folder
    sys.path.insert(1, ROOT)
import conftest  # noqa: E402,F401  (the sandbox pins conftest sets for pytest, for lazy imports)


class Skip(Exception):
    """Raise to skip a test in script mode."""


class MonkeyPatch:
    """pytest's monkeypatch, the part these modules use: setattr, setenv, delenv, setitem,
    delattr, chdir, syspath_prepend; undone after the test."""

    def __init__(self):
        self._undo: list = []

    def setattr(self, target, name, value=None, raising=True):
        if isinstance(target, str) and value is None:
            mod, _, attr = target.rpartition(".")
            target, name, value = __import__(mod, fromlist=[attr]), attr, name
        old = getattr(target, name, None)
        had = hasattr(target, name)
        if raising and not had:
            raise AttributeError(name)
        setattr(target, name, value)
        self._undo.append(lambda: setattr(target, name, old) if had else delattr(target, name))

    def delattr(self, target, name, raising=True):
        if not hasattr(target, name):
            if raising:
                raise AttributeError(name)
            return
        old = getattr(target, name)
        delattr(target, name)
        self._undo.append(lambda: setattr(target, name, old))

    def setitem(self, d, key, value):
        had, old = key in d, d.get(key)
        d[key] = value
        self._undo.append(lambda: d.__setitem__(key, old) if had else d.pop(key, None))

    def setenv(self, name, value, prepend=None):
        self.setitem(os.environ, name, str(value))

    def delenv(self, name, raising=True):
        if name not in os.environ:
            if raising:
                raise KeyError(name)
            return
        old = os.environ.pop(name)
        self._undo.append(lambda: os.environ.__setitem__(name, old))

    def chdir(self, path):
        old = os.getcwd()
        os.chdir(path)
        self._undo.append(lambda: os.chdir(old))

    def syspath_prepend(self, path):
        sys.path.insert(0, str(path))
        self._undo.append(lambda: sys.path.remove(str(path)) if str(path) in sys.path else None)

    def undo(self):
        while self._undo:
            self._undo.pop()()


class _CapSys:
    def __init__(self):
        self.out, self.err = io.StringIO(), io.StringIO()

    def readouterr(self):
        o, e = self.out.getvalue(), self.err.getvalue()
        self.out.seek(0), self.out.truncate(), self.err.seek(0), self.err.truncate()
        return type("Captured", (), {"out": o, "err": e})()


def _skipped(exc: BaseException) -> bool:
    return isinstance(exc, Skip) or type(exc).__name__ in ("Skipped", "SkipTest")


def run(globs: dict) -> None:
    passed = skipped = failed = 0
    for name, fn in sorted(globs.items()):
        if not (name.startswith("test_") and inspect.isfunction(fn)):
            continue
        kw, tmp, mp, cap = {}, None, None, None
        for p in inspect.signature(fn).parameters.values():
            if p.default is not inspect.Parameter.empty:
                continue
            if p.name == "tmp_path":
                tmp = tempfile.mkdtemp(prefix="scriptmode-")
                kw[p.name] = pathlib.Path(tmp)
            elif p.name == "monkeypatch":
                kw[p.name] = mp = MonkeyPatch()
            elif p.name == "capsys":
                kw[p.name] = cap = _CapSys()
            else:
                raise SystemExit(f"{name}: fixture {p.name!r} is not available in script mode")
        try:
            if cap is not None:
                with redirect_stdout(cap.out), redirect_stderr(cap.err):
                    fn(**kw)
            else:
                fn(**kw)
            passed += 1
            print(f"  {name:72s} ok", flush=True)
        except BaseException as exc:                               # noqa: BLE001
            if _skipped(exc):
                skipped += 1
                print(f"  {name:72s} skipped: {exc}", flush=True)
            elif isinstance(exc, KeyboardInterrupt):
                raise
            else:
                failed += 1
                print(f"  {name:72s} FAIL", flush=True)
                traceback.print_exc()
        finally:
            if mp is not None:
                mp.undo()
            if tmp is not None:
                shutil.rmtree(tmp, ignore_errors=True)
    print(f"{passed} passed" + (f", {skipped} skipped" if skipped else "")
          + (f", {failed} FAILED" if failed else ""), flush=True)
    if failed or passed + skipped == 0:
        if passed + skipped == 0:
            print("no test ran", flush=True)
        sys.exit(1)
