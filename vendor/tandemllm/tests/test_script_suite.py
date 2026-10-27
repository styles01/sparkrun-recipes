"""SRV-661: every tests/test_*.py runs its tests as a script, and the suite fails a file that ran none.

Until 0.13 sixteen files (the Ornith trunk's, chatd's file-text and panel tests among them) had no
`__main__` entry: `python tests/<file>` defined their tests, ran none, exited 0, and the box's
script-mode suite counted them as passed.

  1  every test module has a script entry (or checks at import and says `N passed`)
  2  tests/script_suite.py counts the test functions a file ran; a file that exits 0 having run
     none fails, a file that says `N passed` itself counts
  3  tests/scriptmode.py runs a pytest-style module: fixtures, skips, a failure exits 1, no test
     exits 1
"""

from __future__ import annotations

import glob
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SUITE = os.path.join(HERE, "script_suite.py")
#: modules that check at import time and print their own `N passed`
AT_IMPORT = {"test_relax.py"}


def test_every_test_module_has_a_script_entry():
    missing = []
    for path in sorted(glob.glob(os.path.join(HERE, "test_*.py"))):
        src = open(path, encoding="utf-8").read()
        if 'if __name__ == "__main__":' not in src and os.path.basename(path) not in AT_IMPORT:
            missing.append(os.path.basename(path))
    assert not missing, f"no script entry (end with scriptmode.run(globals())): {missing}"


def _suite(*files):
    return subprocess.run([sys.executable, SUITE, "--all", *files], capture_output=True, text=True,
                          timeout=120)


def _module(d, name, body):
    p = os.path.join(d, name)
    with open(p, "w") as fh:
        fh.write(body)
    return p


def test_the_suite_fails_a_file_that_ran_no_test_and_counts_the_others():
    with tempfile.TemporaryDirectory() as d:
        none = _module(d, "test_none.py", "def test_a():\n    assert True\n")
        some = _module(d, "test_some.py", "def test_a():\n    assert True\n\n\ndef test_b():\n"
                       "    assert 1\n\n\nif __name__ == '__main__':\n    test_a()\n    test_b()\n")
        said = _module(d, "test_said.py", "print('7 passed')\n")
        case = _module(d, "test_case.py", "import unittest\n\n\nclass T(unittest.TestCase):\n"
                       "    def test_a(self):\n        pass\n\n    def test_b(self):\n"
                       "        pass\n\n\nif __name__ == '__main__':\n    unittest.main()\n")
        gpu = _module(d, "test_gpu.py", "print('skipped: Triton on a GPU')\n")
        r = _suite(none, some, said, case, gpu)
        out = r.stdout
        assert r.returncode == 1, out
        line = {ln.split()[0].split("/")[-1]: ln for ln in out.splitlines() if ln.endswith(
            ("ok", "test)", ")"))}
        assert "0 tests" in line["test_none.py"] and "FAIL (ran no test)" in line["test_none.py"], out
        assert "2 tests" in line["test_some.py"] and line["test_some.py"].endswith("ok"), out
        assert "7 tests" in line["test_said.py"] and line["test_said.py"].endswith("ok"), out
        # a TestCase module: unittest finds its classes in `__main__` as under a plain python
        assert "2 tests" in line["test_case.py"] and line["test_case.py"].endswith("ok"), out
        # a file that says it skipped on purpose is listed apart, neither passed nor failed
        assert "0 tests" in line["test_gpu.py"] and "SKIPPED (Triton on a GPU)" in line["test_gpu.py"], out
        assert out.rstrip().splitlines()[-1].startswith("files=5 ok=3 fail=1 skipped=1 tests=11"), out


def test_scriptmode_runs_a_pytest_module_with_its_fixtures():
    with tempfile.TemporaryDirectory() as d:
        src = (f"import sys, os\nsys.path.insert(0, {HERE!r})\n"
               "def test_tmp(tmp_path):\n    (tmp_path / 'x').write_text('1')\n"
               "def test_env(monkeypatch):\n    monkeypatch.setenv('SCRIPTMODE_X', '1')\n"
               "    assert os.environ['SCRIPTMODE_X'] == '1'\n"
               "def test_after():\n    assert 'SCRIPTMODE_X' not in os.environ\n"
               "def test_skip():\n    import unittest\n    raise unittest.SkipTest('no')\n"
               "if __name__ == '__main__':\n    from scriptmode import run\n    run(globals())\n")
        p = _module(d, "test_fix.py", src)
        r = subprocess.run([sys.executable, p], capture_output=True, text=True, timeout=60)
        assert r.returncode == 0 and "3 passed, 1 skipped" in r.stdout, r.stdout + r.stderr
        bad = _module(d, "test_bad.py", src.replace("assert 'SCRIPTMODE_X' not in",
                                                    "assert 'SCRIPTMODE_X' in"))
        r = subprocess.run([sys.executable, bad], capture_output=True, text=True, timeout=60)
        assert r.returncode == 1 and "3 passed" not in r.stdout and "1 FAILED" in r.stdout, r.stdout
        empty = _module(d, "test_empty.py", src.split("def test_tmp")[0]
                        + "if __name__ == '__main__':\n    from scriptmode import run\n"
                        "    run(globals())\n")
        r = subprocess.run([sys.executable, empty], capture_output=True, text=True, timeout=60)
        assert r.returncode == 1 and "no test ran" in r.stdout, r.stdout


if __name__ == "__main__":
    from scriptmode import run
    run(globals())
