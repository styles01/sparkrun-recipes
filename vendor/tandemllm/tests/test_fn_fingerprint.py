"""ENG-221 / ENG-E8: the model layer and the shared methods leave the 27B's disk-tier fingerprint as it was.

`engine/models/`, `engine/methods/` and the knobs owned under them are left out of the 27B's
fingerprint (engine/disk_cache.py `_engine_files`, `_NOT_27B_KNOBS`), so adding Flash-Next sweeps no
saved chat. Proof: the fingerprint of this tree equals the fingerprint of a copy without
`engine/models/` and `engine/methods/` and without those knobs, for the tier (no skip) and for the
session spill (engine/spill.py's NOT_ARITHMETIC skip). A model's own fingerprint is v3's
(tests/test_fp_v3.py).
"""

from __future__ import annotations

import ast
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import disk_cache as dc  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTRA = {"weights": "exl3-6.0", "flags": {"X": "1"}}


def _without_models(dst: str) -> None:
    shutil.copytree(os.path.join(ROOT, "engine"), os.path.join(dst, "engine"),
                    ignore=shutil.ignore_patterns("__pycache__", "models", "methods"))
    shutil.copytree(os.path.join(ROOT, "tools"), os.path.join(dst, "tools"),
                    ignore=shutil.ignore_patterns("__pycache__"))
    p = os.path.join(dst, "engine", "settings.py")
    src = open(p).read()
    tree = ast.parse(src)
    lines = src.splitlines(keepends=True)
    drop = []
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", "") == "KNOBS":
            for k, v in zip(node.value.keys, node.value.values):
                if ast.literal_eval(v)[1].startswith(("engine/models/", "tools/fn_")):
                    drop.append(k.lineno)
    with open(p, "w") as f:
        f.writelines(l for i, l in enumerate(lines, 1) if i not in drop)


def test_27b_fingerprint_unchanged_by_the_model_layer():
    assert os.path.isdir(os.path.join(ROOT, "engine", "models", "flashnext"))
    assert os.path.isfile(os.path.join(ROOT, "engine", "methods", "walk.py"))
    with tempfile.TemporaryDirectory() as d:
        _without_models(d)
        assert not os.path.exists(os.path.join(d, "engine", "models"))
        assert not os.path.exists(os.path.join(d, "engine", "methods"))
        from engine.spill import NOT_ARITHMETIC
        for skip in ((), NOT_ARITHMETIC):
            a = dc.fingerprint(ROOT, EXTRA, skip=skip)
            b = dc.fingerprint(d, EXTRA, skip=skip)
            assert a == b, (skip, a[:12], b[:12])
    return f"tier and spill fingerprints equal without engine/models and engine/methods ({a[:12]})"


if __name__ == "__main__":
    fails = 0
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        try:
            print(f"  {name:<50} ok   {fn() or ''}")
        except AssertionError as e:
            fails += 1
            print(f"  {name:<50} FAIL {e}")
    print(f"{len(tests) - fails} passed" + (f", {fails} FAILED" if fails else ""))
    sys.exit(1 if fails else 0)
