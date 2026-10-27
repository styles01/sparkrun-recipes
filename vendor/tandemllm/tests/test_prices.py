"""/121: price tables per weight set.

No table must leave every router price exactly as the code has it (the served NVFP4 build); a table
must reach the tree curve, the length router's priors and the merged router's constants; the
environment's QWEN38_TREE_MS must still win; a bad file must fail loudly.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import prices  # noqa: E402
from engine.router import SERVED_TREE_MS, served_tree_table  # noqa: E402

FP8 = {"tree_ms": {"8": 150.0, "16": 152.0, "24": 170.0, "32": 181.0},
       "lenrouter_tree_ms": {"8": 150.0, "16": 152.0, "32": 181.0},
       "lenrouter_draft_ms": {"8": 13.0, "16": 14.0}, "head_fixed_ms": 30.0, "rollback_ms": 7.0,
       "measured": "test"}


def _file(doc) -> str:
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump(doc, f)
    return path


def test_no_table_is_identity():
    os.environ.pop("QWEN38_TREE_MS", None)
    assert prices.load(None) == (None, {}) and prices.load("") == (None, {})
    assert served_tree_table() == SERVED_TREE_MS
    assert served_tree_table(None) == SERVED_TREE_MS
    return "no table: the code's constants"


def test_table_reaches_the_curves():
    os.environ.pop("QWEN38_TREE_MS", None)
    path = _file({"fp8-plain": FP8})
    name, e = prices.load(path)
    assert name == "fp8-plain"
    assert e["tree_ms"] == {8: 150.0, 16: 152.0, 24: 170.0, 32: 181.0}
    assert e["lenrouter_draft_ms"] == {8: 13.0, 16: 14.0}
    assert e["head_fixed_ms"] == 30.0 and e["rollback_ms"] == 7.0
    assert served_tree_table(e["tree_ms"]) == e["tree_ms"]
    from engine.lenrouter import LengthRouter  # noqa: F401  (the keyword names the server passes)
    import inspect
    sig = inspect.signature(LengthRouter.__init__).parameters
    assert "verify_table" in sig and "draft_table" in sig
    return "tree curve, length-router priors, merged constants"


def test_env_wins():
    os.environ["QWEN38_TREE_MS"] = "8:77.9,16:78.9,24:87.0,32:97.0"
    try:
        _, e = prices.load(_file({"fp8-plain": FP8}))
        assert served_tree_table(e["tree_ms"]) == {8: 77.9, 16: 78.9, 24: 87.0, 32: 97.0}
    finally:
        os.environ.pop("QWEN38_TREE_MS", None)
    return "QWEN38_TREE_MS beats the table"


def test_names_and_errors():
    two = _file({"a": FP8, "b": FP8})
    assert prices.load(two + ":b")[0] == "b"
    bad = [two,                                           # two entries, no name
           two + ":c",                                     # unknown name
           _file({"x": {"tree_ms": {"8": 1.0}}}),          # tree_ms without 16
           _file({"x": {"tree_ms": {"8": -1, "16": 2}}}),  # negative time
           _file({"x": {"verify": {}}}),                   # unknown key
           _file([])]
    for spec in bad:
        try:
            prices.load(spec)
        except (ValueError, KeyError):
            continue
        raise AssertionError(f"{spec} accepted")
    return f"{len(bad)} bad tables refused"


if __name__ == "__main__":
    fails = 0
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        try:
            print(f"  {name:<32} ok   {fn() or ''}")
        except AssertionError as e:
            fails += 1
            print(f"  {name:<32} FAIL {e}")
    print(f"{len(tests) - fails} passed" + (f", {fails} FAILED" if fails else ""))
    sys.exit(1 if fails else 0)
