"""ENG-243: fingerprint v3, the arithmetic manifest (engine/methods/manifest.py).

  * the 27B's v3 is unchanged by an edit to method code (speculation, sampling, the caches, the shared
    methods, Flash-Next's package and kernels) that sweeps v2;
  * it changes with an edit to a file that computes rows, a `tools/` kernel they import, the default
    of a knob such a file owns, and the content of a tile table a knob names (v2 hashed only the name);
  * Flash-Next's v3 changes with its kernels and its tile tables, not with its server adapter;
  * the manifest holds every engine module `engine/model.py` imports, transitively, and is pinned;
  * a tile table both models read moves both; the two models' fingerprints differ, so neither reads
    the other's rows;
  * the adoption record: the first start under v3 adopts v2's files of the same tree (and what v2
    itself read) once, in a directory that already has a v2 record; a restart keeps it, a later
    arithmetic change gets nothing.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import test_forward_tree as T  # noqa: E402,F401  (the CPU environment before the engine imports)
from engine import disk_cache as dc  # noqa: E402
from engine.methods import manifest as mf  # noqa: E402
from engine.models.flashnext import serving  # noqa: E402
from engine.models.qwen3_5 import adapter  # noqa: E402
from engine.spill import NOT_ARITHMETIC  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTRA = {"weights": "w", "env": {}}
M27, MFN = adapter.manifest(), serving.manifest()


def _copy(d: str) -> str:
    for sub in ("engine", "tools", "ops"):
        shutil.copytree(os.path.join(ROOT, sub), os.path.join(d, sub),
                        ignore=shutil.ignore_patterns("__pycache__", "*.npy", "monitoring", "profiles"))
    return d


def _touch(d: str, rel: str, text: str = "\n# an edit\n") -> None:
    with open(os.path.join(d, rel), "a") as f:
        f.write(text)


def _fps(d: str, env=None) -> tuple:
    saved = dict(os.environ)
    try:
        if env:
            os.environ.update(env)
        return (mf.fingerprint(d, M27, EXTRA), mf.fingerprint(d, MFN, EXTRA),
                dc.fingerprint(d, EXTRA, skip=NOT_ARITHMETIC))
    finally:
        os.environ.clear()
        os.environ.update(saved)


def test_method_edits_leave_the_27b_v3():
    with tempfile.TemporaryDirectory() as d:
        _copy(d)
        a27, afn, av2 = _fps(d)
        for rel in ("engine/spec.py", "engine/sample.py", "engine/router.py", "engine/cache.py",
                    "engine/methods/walk.py", "engine/models/flashnext/serving.py"):
            _touch(d, rel)
        b27, bfn, bv2 = _fps(d)
        assert b27 == a27 and bfn == afn, "a method edit moved a v3"
        assert bv2 != av2                         # v2 swept every saved conversation for engine/spec.py
        _touch(d, "tools/fn_kernels.py")
        _touch(d, "ops/fn-exl3-tiles.json", " ")
        c27, cfn, _ = _fps(d)
        assert c27 == a27 and cfn != bfn, "Flash-Next's kernels and tiles are its own"
    return f"27B v3 {a27[:12]} kept through 6 method edits and Flash-Next's kernels (v2 moved)"


def test_arithmetic_edits_move_the_27b_v3():
    with tempfile.TemporaryDirectory() as d:
        _copy(d)
        tiles = os.path.join(d, "ops", "skinny-tiles.json")
        env = {"QWEN38_SKINNY_TILES": tiles}
        base, _, v2 = _fps(d, env)
        seen = {base}
        _touch(d, "ops/skinny-tiles.json", " ")       # the table's content, same name
        t27, _, tv2 = _fps(d, env)
        assert t27 not in seen and tv2 == v2, "v3 reads the table; v2 only its name"
        seen.add(t27)
        for rel in ("engine/model.py", "tools/gdn_kernels.py", "engine/drafters/mtp.py"):
            _touch(d, rel)
            got = _fps(d, env)[0]
            assert got not in seen, rel
            seen.add(got)
        p = os.path.join(d, "engine", "settings.py")
        src = open(p).read()
        own = re.search(r"'(\w+)': \('([^']*)', 'engine/model.py'\)", src)
        meth = re.search(r"'(\w+)': \('([^']*)', 'engine/router.py'\)", src)
        before = _fps(d, env)[0]
        open(p, "w").write(src.replace(meth.group(0), meth.group(0).replace(f"('{meth.group(2)}'", "('77'")))
        assert _fps(d, env)[0] == before, "a method's knob default"
        open(p, "w").write(src.replace(own.group(0), own.group(0).replace(f"('{own.group(2)}'", "('77'")))
        assert _fps(d, env)[0] != before, "an engine/model.py knob default"
    return "table content, model.py, a kernel, a cached drafter, an owned knob default: each moves v3"


def test_the_27b_manifest_is_pinned():
    assert M27.modules == (
        "engine/config.py", "engine/drafters/__init__.py", "engine/drafters/dflash2.py",
        "engine/drafters/draft_graph.py", "engine/drafters/dspark.py", "engine/drafters/mtp.py",
        "engine/gdn.py", "engine/kvpool.py", "engine/linear.py", "engine/loader.py", "engine/model.py",
        "engine/settings.py", "engine/tree.py", "engine/verify_graph.py", "engine/vision.py"), M27.modules
    assert M27.data == () and M27.arch == "qwen3_5"
    tools = sorted(os.path.relpath(p, ROOT) for p in mf.files(ROOT, M27)[len(M27.modules):])
    assert "tools/nvfp4_skinny.py" in tools and not any(t.startswith("tools/fn_") for t in tools), tools
    return f"{len(M27.modules)} engine modules and {len(tools)} tools/ kernels (docs/caches.md)"


def test_a_shared_tile_table_moves_both_and_the_models_never_share_one():
    with tempfile.TemporaryDirectory() as d:
        _copy(d)
        a27, afn, _ = _fps(d)
        assert a27 != afn, "two models, one fingerprint"
        p = os.path.join(d, "ops", "exl3-tiles-fused.json")
        doc = json.load(open(p))
        k = next(iter(doc))
        doc[k] = {"changed": True} if not isinstance(doc[k], dict) else {**doc[k], "changed": True}
        json.dump(doc, open(p, "w"))
        b27, bfn, _ = _fps(d)
        assert b27 != a27 and bfn != afn, "one entry of the shared table"
    return "ops/exl3-tiles-fused.json in both manifests; the 27B's and Flash-Next's fingerprints differ"


def test_the_manifest_holds_what_model_py_imports():
    pat = re.compile(r"^\s*(?:from\s+engine(?:\.([\w.]+))?\s+import\s+([\w\s,()]+)|import\s+engine\.([\w.]+))", re.M)
    seen, todo = set(), ["engine/model.py"]
    while todo:
        rel = todo.pop()
        if rel in seen:
            continue
        seen.add(rel)
        for m in pat.finditer(open(os.path.join(ROOT, rel)).read()):
            if m.group(3):
                names = [m.group(3)]
            else:
                base = m.group(1)
                subs = [n.strip().split(" as ")[0] for n in m.group(2).replace("(", " ").replace(")", " ").split(",")]
                names = ([base] if base else []) + [f"{base}.{n}" if base else n for n in subs if n]
            for n in names:
                f = os.path.join("engine", *n.split("."))
                for c in (f + ".py", os.path.join(f, "__init__.py")):
                    if os.path.isfile(os.path.join(ROOT, c)):
                        todo.append(c)
    missing = sorted(seen - set(M27.modules) - {os.path.join("engine", n) for n in NOT_ARITHMETIC})
    assert not missing, missing
    return f"{len(seen)} modules imported from engine/model.py, all in the 27B's manifest"


def test_v3_adopts_v2_once_per_formula():
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, dc.EQUIV_FILE), "w") as f:
            json.dump({"v2-of-0.4.6": ["v1"], "v2-now": ["v2-of-0.4.6"]}, f)   # a box since 0.4.6
        assert dc.equivalents(d, "v3", "v2-now", formula=mf.FORMULA) == ("v2-now", "v2-of-0.4.6")
        doc = json.load(open(os.path.join(d, dc.EQUIV_FILE)))
        assert doc["formulas"] == {mf.FORMULA: "v3"} and doc["v2-of-0.4.6"] == ["v1"]
        assert dc.equivalents(d, "v3", "v2 after an edit", formula=mf.FORMULA) == ("v2-now", "v2-of-0.4.6")
        assert dc.equivalents(d, "v3 after a kernel change", "v2-now", formula=mf.FORMULA) == ()
        assert dc.equivalents(d, "v2-of-0.4.6", "v1") == ("v1",)        # the v2 record as it was
    with tempfile.TemporaryDirectory() as d:
        assert dc.equivalents(d, "v3", "v2", formula=mf.FORMULA) == ("v2",)          # a fresh directory
    return "adopted once with what v2 read; restart keeps it; a later change gets nothing"


if __name__ == "__main__":
    fails = 0
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        try:
            print(f"  {name:<46} ok   {fn() or ''}")
        except AssertionError as e:
            fails += 1
            print(f"  {name:<46} FAIL {e}")
    print(f"{len(tests) - fails} passed" + (f", {fails} FAILED" if fails else ""))
    sys.exit(1 if fails else 0)
