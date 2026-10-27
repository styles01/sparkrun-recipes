"""/124: every QWEN38_* knob the engine reads goes through engine/settings.py.

Pins the seam, not a behaviour: (1) no module of the engine's scope (engine/, server/, the kernel
and linear modules of tools/) reads a QWEN38_* variable any other way than `SETTINGS.get`; (2) every
name read is in the registry and every registry entry is read somewhere; (3) `SETTINGS` views the
live environment (so a value set before an import, or before a per-call read, is seen exactly as the
old `os.environ.get` saw it); (4) a second EngineSettings over another mapping or another prefix is
independent of the process's.
"""

from __future__ import annotations

import glob
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from engine.settings import KNOBS, SETTINGS, EngineSettings  # noqa: E402

KERNEL = re.compile(r"_kernels|fp8_linear|nvfp4_linear|exl3_linear|nvfp4_skinny|nvfp4_verify_tiles|head_gemv|"
                    r"attn_|small_linear|loop_sync")
# reads that are a tool's own __main__ bench knobs, not the engine's
ALLOWED = {("tools/gdn_wy_kernels.py", "QWEN38_WY_GRID"), ("tools/gdn_wy_kernels.py", "QWEN38_WY_SIZES")}


def scope() -> list[str]:
    fs = glob.glob(os.path.join(ROOT, "engine", "**", "*.py"), recursive=True)
    fs += glob.glob(os.path.join(ROOT, "server", "**", "*.py"), recursive=True)
    fs += [f for f in glob.glob(os.path.join(ROOT, "tools", "*.py")) if KERNEL.search(os.path.basename(f))]
    return sorted(f for f in fs if not f.endswith("engine/settings.py"))


def test_no_stray_reads():
    raw = re.compile(r"os\.(environ|getenv)[^\n]*QWEN38_[A-Z0-9_]+|QWEN38_[A-Z0-9_]+[\"']\s*in\s+os\.environ")
    bad = []
    for f in scope():
        rel = os.path.relpath(f, ROOT)
        for i, line in enumerate(open(f), 1):
            if raw.search(line):
                name = re.search(r"QWEN38_[A-Z0-9_]+", line).group(0)
                if (rel, name) not in ALLOWED:
                    bad.append(f"{rel}:{i}: {line.strip()[:90]}")
    assert not bad, "QWEN38_* read around engine/settings.py:\n" + "\n".join(bad)
    return f"{len(scope())} modules, every QWEN38_* read through SETTINGS"


def _environ_reads(tree):
    """(line, name) for every read of the process environment in a module: os.environ.get(...),
    os.getenv(...), os.environ[...] and `... in os.environ`, with the name a string literal, an
    f-string's literal head, or a module constant bound to a string. The line regex above misses a
    call split over two lines (QWEN38_MEMORY_PROXIES, 2026-10-05) and a name held in a constant
    (QWEN38_TANDEM_URL / _SCRATCH, SRV-163/171); this does not."""
    import ast
    consts = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) \
                and isinstance(node.value.value, str):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    consts[t.id] = node.value.value

    def is_environ(n):
        return isinstance(n, ast.Attribute) and n.attr == "environ" \
            and isinstance(n.value, ast.Name) and n.value.id == "os"

    def name_of(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, str):
            return n.value
        if isinstance(n, ast.Name):
            return consts.get(n.id)
        if isinstance(n, ast.JoinedStr) and n.values and isinstance(n.values[0], ast.Constant):
            return str(n.values[0].value)
        return None

    for n in ast.walk(tree):
        arg = None
        if isinstance(n, ast.Call) and n.args:
            f = n.func
            if (isinstance(f, ast.Attribute) and f.attr == "get" and is_environ(f.value)) or \
                    (isinstance(f, ast.Attribute) and f.attr == "getenv"
                     and isinstance(f.value, ast.Name) and f.value.id == "os"):
                arg = n.args[0]
        elif isinstance(n, ast.Subscript) and is_environ(n.value) and isinstance(n.ctx, ast.Load):
            arg = n.slice
        elif isinstance(n, ast.Compare) and len(n.ops) == 1 and isinstance(n.ops[0], (ast.In, ast.NotIn)) \
                and is_environ(n.comparators[0]):
            arg = n.left
        name = name_of(arg) if arg is not None else None
        if name and name.startswith("QWEN38_"):
            yield n.lineno, name


def test_no_hidden_reads():
    import ast
    bad = []
    for f in scope():
        rel = os.path.relpath(f, ROOT)
        for line, name in _environ_reads(ast.parse(open(f).read(), f)):
            if (rel, name) not in ALLOWED:
                bad.append(f"{rel}:{line}: {name}")
    assert not bad, "QWEN38_* read around engine/settings.py (split call or constant name):\n" + "\n".join(bad)
    return "no QWEN38_* environment read hides from the line check"


def test_registry_matches_reads():
    used = set()
    for f in scope():
        used |= set(re.findall(r"_S\.get\(\s*\"([A-Z0-9_]+)\"", open(f).read()))
    computed = {"TREE_NODES", "TREE_NODES_NARROW", "SKINNY_TILES_B", "SKINNY_TILES_C", "SKINNY_TILES_WIDE_B"}
    missing = used - set(KNOBS)
    unused = set(KNOBS) - used - computed
    assert not missing, f"read but not registered: {sorted(missing)}"
    assert not unused, f"registered but never read: {sorted(unused)}"
    for k, (d, where) in KNOBS.items():
        assert d is None or isinstance(d, str), (k, d)
        assert os.path.isfile(os.path.join(ROOT, where)), (k, where)
    return f"{len(KNOBS)} knobs, all read, all registered"


def test_live_environment():
    os.environ.pop("QWEN38_FUSE_PROJ", None)
    assert SETTINGS.get("FUSE_PROJ") == KNOBS["FUSE_PROJ"][0]
    os.environ["QWEN38_FUSE_PROJ"] = "1"
    try:
        assert SETTINGS.get("FUSE_PROJ") == "1" and SETTINGS.is_set("FUSE_PROJ")
    finally:
        os.environ.pop("QWEN38_FUSE_PROJ")
    try:
        SETTINGS.get("NOT_A_KNOB")
    except KeyError:
        pass
    else:
        raise AssertionError("an unregistered knob was read")
    return "reads the live environment; an unknown name is an error"


def test_second_configuration_and_prefix():
    other = EngineSettings(env={"QSE_FUSE_PROJ": "1", "QSE_VERIFY_ROWS": "32"}, prefix="QSE_")
    assert other.get("FUSE_PROJ") == "1" and other.get("VERIFY_ROWS") == "32"
    assert other.get("TREE_MS") == KNOBS["TREE_MS"][0]
    d = other.describe()
    assert d["FUSE_PROJ"]["set"] and not d["TREE_MS"]["set"] and len(d) == len(KNOBS)
    return "a mapping with prefix QSE_ is its own configuration; describe() lists every knob"


def test_scripts_start_without_the_repo_on_the_path():
    """The engine's entry points run as scripts (`python server/app.py`, row3's and ops/start.sh's way), where only
    the script's own directory is on sys.path until the script inserts the repo root. An import of engine.settings
    above that insert broke the server on 2026-09-27 (row3: ModuleNotFoundError: No module named 'engine'); the CPU
    suite runs with the repo on PYTHONPATH and could not see it, so this drops the repo from the path on purpose."""
    import subprocess
    keep = [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep)
            if p and os.path.abspath(p) != ROOT]
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(keep), CUDA_VISIBLE_DEVICES="")
    out = []
    for script in ("server/app.py", "tools/row3.py", "tools/fp8_probe.py", "tools/quant_head.py"):
        r = subprocess.run([sys.executable, script, "--help"], cwd=ROOT, env=env, capture_output=True,
                           text=True, timeout=300)
        assert r.returncode == 0, f"{script} --help: rc {r.returncode}\n{r.stderr[-600:]}"
        out.append(script)
    return "script mode starts: " + ", ".join(out)


if __name__ == "__main__":
    fails = 0
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        try:
            print(f"  {name:<40} ok   {fn() or ''}")
        except AssertionError as e:
            fails += 1
            print(f"  {name:<40} FAIL {e}")
    print(f"{len(tests) - fails} passed" + (f", {fails} FAILED" if fails else ""))
    sys.exit(1 if fails else 0)
