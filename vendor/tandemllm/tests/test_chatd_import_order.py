"""The tests' sandbox pins hold whatever pytest collects first (tests/conftest.py).

A test module that imports chatd before tests/chatd_util.py sets QSE_SANDBOX_MEM_GB=1 (as
tests/test_sandbox_box.py did), collected first, left chatd/sandbox.py at its 4 GiB default and
six limit tests in tests/test_chatd_agent.py failed. A fresh pytest collects
tests/chatd_import_first.py, such a module, before them.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_chatd_imported_before_chatd_util_keeps_the_tests_limit():
    import importlib.util
    import unittest
    if importlib.util.find_spec("pytest") is None:
        # it asks pytest's collection order; a box without pytest runs the scripts one by one
        raise unittest.SkipTest("pytest is not installed")
    env = {k: v for k, v in os.environ.items() if not k.startswith("QSE_SANDBOX_")}
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                        "tests/chatd_import_first.py", "tests/test_chatd_agent.py",
                        "-k", "memory_guard or killed_at_the_limit"],
                       cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)
    # 0: every selected test passed (5 would be none selected, 1 a failure)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-2000:]
    assert re.search(r"\b[1-9]\d* passed", r.stdout), r.stdout[-500:]


def test_the_limit_in_this_session():
    from chatd import sandbox as sb_mod
    assert sb_mod.LIMIT_BYTES == sb_mod.GiB
    assert sb_mod.SOCKET != "/var/run/docker.sock" or os.environ.get("QSE_SANDBOX_BOX") == "1"


if __name__ == "__main__":
    from scriptmode import run
    run(globals())
