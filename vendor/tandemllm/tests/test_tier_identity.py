"""rc-0.8.0: the disk tier's identity by content, and 0.7.2's files adopted under the RC.

THE CASE (live, 2026-10-13): the 27B moved from ~/tandem-live (0.7.2) to ~/tandem-rc080 swept all 29
files of its tier (2.97 GB). tools/tier_fp_inputs.py rebuilt both fingerprints exactly (d302e0dd...,
cd8a07b0...) and found twelve inputs apart, none of them arithmetic: four drafter checkpoint files and
four table knobs by path, and four shadow-reader knobs main's serve.env added after 0.7.2.

  1  fp4 hashes the extra by content: a tree in another folder, a copied checkpoint and the shadow
     knobs keep it; a changed checkpoint byte, table or row knob does not
  2  0.7.2's inputs adopt under the RC: in place, and from another folder with QSE_DISK_ADOPT_FROM
  3  the RC's own fp3 files (Flash-Next's tier and spill on the box) adopt on the same tree

CPU only. Run: python tests/test_tier_identity.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import disk_cache as dc  # noqa: E402
from engine import spill as sp  # noqa: E402
from engine.methods import manifest as mf  # noqa: E402
from server import diskcache  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M = mf.Manifest(arch="qwen3_5", modules=("engine/config.py", "engine/tree.py"))


def _tree(base: str, name: str, mtime: float | None = None, byte: int = 7) -> str:
    """A tree folder with a drafter checkpoint and a StairCut table."""
    t = os.path.join(base, name)
    os.makedirs(os.path.join(t, "train", "ft-b8"))
    os.makedirs(os.path.join(t, "ops"))
    with open(os.path.join(t, "train", "ft-b8", "config.json"), "w") as f:
        f.write('{"block": 8}')
    with open(os.path.join(t, "train", "ft-b8", "model.safetensors"), "wb") as f:
        f.write(bytes([byte]) * 4096 + b"weights")
    with open(os.path.join(t, "ops", "stair.json"), "w") as f:
        f.write('{"chain": [1, 2, 3]}')
    if mtime is not None:
        for d, _, fs in os.walk(t):
            for x in fs:
                os.utime(os.path.join(d, x), (mtime, mtime))
    return t


def _args(t: str):
    return types.SimpleNamespace(model=None, nvfp4="/nonexistent/mlp.safetensors", fp8_head=None,
                                 head=None, dflash2_ckpt=os.path.join(t, "train", "ft-b8"),
                                 dflash2_ckpt16=None, drafter="lenrouter", prefix_chunk=1024,
                                 narrow_arm="on")


class Env:
    """QWEN38_* as a tree's serve.env sets them (the process's other variables kept)."""

    def __init__(self, t: str, added: bool, **more):
        self.env = {"QWEN38_STAIR_TABLES": os.path.join(t, "ops", "stair.json"),
                    "QWEN38_TREE_NODES": "24", "QWEN38_SHADOW_MODEL": "qwen3.5-0.8b",
                    **({k: v for k, v in diskcache.ADDED_AFTER_072.items()} if added else {}),
                    **more}

    def __enter__(self):
        self.saved = {k: v for k, v in os.environ.items() if k.startswith("QWEN38_")}
        for k in self.saved:
            del os.environ[k]
        os.environ.update(self.env)

    def __exit__(self, *exc):
        for k in [k for k in os.environ if k.startswith("QWEN38_")]:
            del os.environ[k]
        os.environ.update(self.saved)


def _v2(extra):
    """0.7.2's tier fingerprint of these inputs."""
    return dc.fingerprint(ROOT, extra, skip=sp.NOT_ARITHMETIC)


# ------------------------------------------------------------------ 1. by content

def test_a_moved_tree_a_copied_checkpoint_and_the_shadow_knobs_keep_the_content_extra():
    d = tempfile.mkdtemp(prefix="qse-tid-")
    try:
        old, new = _tree(d, "tandem-live", mtime=1.7e9), _tree(d, "tandem-rc080", mtime=1.8e9)
        with Env(old, added=False):
            a = diskcache.fingerprint_extra(_args(old))
            pa = diskcache.fingerprint_extra_paths(_args(old))
        with Env(new, added=True):
            b = diskcache.fingerprint_extra(_args(new))
            pb = diskcache.fingerprint_extra_paths(_args(new))
        assert a == b, "another folder, another mtime, the shadow knobs: the same rows"
        assert pa != pb, "the extra by path is where the files are"
        assert mf.fingerprint(ROOT, M, a) == mf.fingerprint(ROOT, M, b)
        assert not any(k.startswith("QWEN38_SHADOW") for k in b["env"])
        assert b["files"][0] == "missing" and isinstance(b["files"][1], dict)
        # what does change a row still does
        with open(os.path.join(new, "train", "ft-b8", "model.safetensors"), "r+b") as f:
            f.write(b"\x00")
        diskcache._IDS.clear()
        with Env(new, added=True):
            c = diskcache.fingerprint_extra(_args(new))
        assert c != b, "a changed checkpoint byte"
        with open(os.path.join(new, "ops", "stair.json"), "w") as f:
            f.write('{"chain": [1, 2, 4]}')
        with Env(new, added=True):
            assert diskcache.fingerprint_extra(_args(new))["env"]["QWEN38_STAIR_TABLES"] != \
                b["env"]["QWEN38_STAIR_TABLES"], "a changed table"
        with Env(new, added=True, QWEN38_TREE_NODES="16"):
            assert diskcache.fingerprint_extra(_args(new))["env"] != c["env"], "a row knob"
        return "moved, copied, shadow knobs: equal; a checkpoint byte, a table, a row knob: not"
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_a_file_past_the_sample_size_is_known_by_its_sampled_bytes():
    d = tempfile.mkdtemp(prefix="qse-tid-")
    real = diskcache.SAMPLE_OVER
    try:
        diskcache.SAMPLE_OVER = 1 << 16
        p = os.path.join(d, "big.safetensors")
        with open(p, "wb") as f:
            f.write(os.urandom(3 << 20))
        a = diskcache.content_id(p)
        shutil.copy(p, p + ".copy")
        assert diskcache.content_id(p + ".copy") == a
        with open(p, "r+b") as f:
            f.write(b"\x00\x01")             # in the first MiB
        os.utime(p, (time.time() + 5, time.time() + 5))
        assert diskcache.content_id(p) != a
        return f"{a.split(':')[0]} bytes, read in 18 places"
    finally:
        diskcache.SAMPLE_OVER = real
        shutil.rmtree(d, ignore_errors=True)


# ------------------------------------------------------------------ 2. 0.7.2's files

def _adopted(tier: str, v2_old: str, legacies: list) -> tuple:
    # what 0.7.2's own start left in the directory: its v2 and what v2 read (its fingerprint_legacy)
    with open(os.path.join(tier, dc.EQUIV_FILE), "w") as f:
        json.dump({v2_old: ["0.4.6's"]}, f)
    return dc.equivalents(tier, "the RC's fp4", legacies, formula=mf.FORMULA)


def test_0_7_2s_tier_inputs_adopt_under_the_rc():
    d = tempfile.mkdtemp(prefix="qse-tid-")
    try:
        old, new = _tree(d, "tandem-live", mtime=1.7e9), _tree(d, "tandem-rc080", mtime=1.7e9)
        with Env(old, added=False):
            v2_old = _v2(diskcache.fingerprint_extra_paths(_args(old)))
        # in place: 0.7.2's folder, 0.8.0's serve.env with the four new lines
        with Env(old, added=True):
            leg = diskcache.legacy_fingerprints(ROOT, M, _args(old), tree=old)
        tier = os.path.join(d, "tier-a")
        os.makedirs(tier)
        also = _adopted(tier, v2_old, leg)
        assert v2_old in also and "0.4.6's" in also, also
        # moved to another folder: only with the folder it came from
        with Env(new, added=True):
            leg_new = diskcache.legacy_fingerprints(ROOT, M, _args(new), tree=new)
            leg_from = diskcache.legacy_fingerprints(ROOT, M, _args(new), [old], tree=new)
        assert v2_old not in leg_new
        tier = os.path.join(d, "tier-b")
        os.makedirs(tier)
        assert v2_old in _adopted(tier, v2_old, leg_from)
        with open(os.path.join(tier, dc.EQUIV_FILE)) as f:
            assert json.load(f)["formulas"] == {mf.FORMULA: "the RC's fp4"}
        # a checkpoint that differs is not the same rows: no adoption
        other = _tree(d, "elsewhere", mtime=1.7e9, byte=9)
        with Env(other, added=True):
            leg_other = diskcache.legacy_fingerprints(ROOT, M, _args(other), [old], tree=other)
        assert v2_old not in leg_other
        return f"0.7.2's {v2_old[:12]} adopted in place and from its old folder"
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_the_rcs_own_fp3_files_adopt_on_the_same_tree():
    d = tempfile.mkdtemp(prefix="qse-tid-")
    try:
        t = _tree(d, "tandem-rc080", mtime=1.7e9)
        with Env(t, added=True):
            fp3 = mf.fingerprint(ROOT, M, diskcache.fingerprint_extra_paths(_args(t)), formula="fp3")
            fp4 = mf.fingerprint(ROOT, M, diskcache.fingerprint_extra(_args(t)))
            leg = diskcache.legacy_fingerprints(ROOT, M, _args(t), tree=t)
        assert fp3 in leg and fp4 != fp3
        tier = os.path.join(d, "tier")
        os.makedirs(tier)
        with open(os.path.join(tier, dc.EQUIV_FILE), "w") as f:
            json.dump({fp3: ["v2"], "formulas": {"fp3": fp3}}, f)
        also = dc.equivalents(tier, fp4, leg, formula=mf.FORMULA)
        assert fp3 in also and "v2" in also
        assert dc.equivalents(tier, fp4, ["anything"], formula=mf.FORMULA) == also, "adopted once"
        return f"fp3 {fp3[:12]} read under fp4 {fp4[:12]}"
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ------------------------------------------------------------------ 3. NemotronH's GGUF

def test_nemotron_hs_gguf_and_its_loader_are_in_the_fingerprint():
    """A Bolt profile serves the GGUF QSE_NH_GGUF names (or one in the model folder), loaded by
    gguf_load.py and tools/iq2_linear.py: a changed byte in any of them is another fingerprint."""
    from engine.models.nemotron_h import serving
    names = {os.path.relpath(p, ROOT) for p in mf.files(ROOT, serving.manifest())}
    for f in ("engine/models/nemotron_h/gguf.py", "engine/models/nemotron_h/gguf_load.py", "tools/iq2_linear.py"):
        assert f in names, f"{f} is not in NemotronH's manifest"
    d = tempfile.mkdtemp(prefix="qse-tid-")
    saved = os.environ.pop("QSE_NH_GGUF", None)
    try:
        cfg, here, there = (os.path.join(d, x) for x in ("config", "here", "there"))
        for x in (cfg, here, there):
            os.makedirs(x)
        with open(os.path.join(cfg, "config.json"), "w") as f:
            f.write('{"model_type": "nemotron_h"}')
        for x in (here, there):
            with open(os.path.join(x, "bolt.gguf"), "wb") as f:
                f.write(b"GGUF" + bytes(4096))
        a = types.SimpleNamespace(model=cfg, prefix_chunk=1024)
        os.environ["QSE_NH_GGUF"] = os.path.join(here, "bolt.gguf")
        x = diskcache.fingerprint_extra(a)
        os.environ["QSE_NH_GGUF"] = os.path.join(there, "bolt.gguf")
        assert diskcache.fingerprint_extra(a) == x, "the same GGUF in another folder"
        with open(os.path.join(there, "bolt.gguf"), "r+b") as f:
            f.write(b"GGUF\x01")
        assert diskcache.fingerprint_extra(a) != x, "another GGUF beside the config folder"
        del os.environ["QSE_NH_GGUF"]
        shutil.copy(os.path.join(here, "bolt.gguf"), cfg)
        y = diskcache.fingerprint_extra(a)
        shutil.copy(os.path.join(there, "bolt.gguf"), cfg)
        assert diskcache.fingerprint_extra(a) != y, "another GGUF in the model folder"
        return "the loader, the IQ2 kernel and the GGUF's bytes count"
    finally:
        os.environ.pop("QSE_NH_GGUF", None)
        if saved is not None:
            os.environ["QSE_NH_GGUF"] = saved
        shutil.rmtree(d, ignore_errors=True)


def test_the_shared_vision_code_nemotron_hs_image_rows_import_is_in_the_fingerprint():
    """Super VL's tower (nemotron_h/vision.py) takes its attention, image context and embed cache from
    engine/vision.py: a change there changes image rows, so every engine module it imports counts."""
    import ast
    from engine.models.nemotron_h import serving
    names = {os.path.relpath(p, ROOT) for p in mf.files(ROOT, serving.manifest())}
    with open(os.path.join(ROOT, "engine/models/nemotron_h/vision.py")) as f:
        tree = ast.parse(f.read())
    used = {n.module.replace(".", "/") + ".py" for n in ast.walk(tree)
            if isinstance(n, ast.ImportFrom) and n.module and n.module.startswith("engine.")}
    assert "engine/vision.py" in used
    for f in sorted(used):
        assert f in names, f"{f} is not in NemotronH's manifest"
    return ", ".join(sorted(used))


# ------------------------------------------------------------------ 4. what NemotronH's files import (ENG-379)

def _engine_imports(read, start) -> dict:
    """Every engine/ module the files in `start` import, at the top or inside a function, and what those
    import in turn: module -> the first file found importing it. `read(rel)` is a file's text."""
    import ast
    seen, todo, via = set(), list(start), {}
    while todo:
        rel = todo.pop()
        if rel in seen:
            continue
        seen.add(rel)
        for n in ast.walk(ast.parse(read(rel))):
            if isinstance(n, ast.Import):
                names = [a.name for a in n.names]
            elif isinstance(n, ast.ImportFrom) and not n.level and n.module:
                names = [n.module] + [f"{n.module}.{a.name}" for a in n.names]
            else:
                continue
            for name in (x for x in names if x.split(".")[0] == "engine"):
                f = os.path.join(*name.split("."))
                hit = next((c for c in (f + ".py", os.path.join(f, "__init__.py"))
                            if os.path.isfile(os.path.join(ROOT, c))), None)
                if hit is not None:
                    via.setdefault(hit, rel)
                    todo.append(hit)
    return {m: via[m] for m in seen if m in via}


def _read(rel: str) -> str:
    with open(os.path.join(ROOT, rel)) as f:
        return f.read()


def test_every_engine_module_nemotron_h_imports_is_listed_or_left_out_with_its_reason():
    """kvpool.py (inside new_state), capture_lock.py and config.py were imported by NemotronH's files and
    in neither list: a change to one that changed rows would have kept the old rows. Each import is now
    in MODULES or in LEFT_OUT with why it computes no row, and a new one fails here."""
    from engine.models.nemotron_h import serving
    found = _engine_imports(_read, serving.MODULES)
    stray = sorted(f"{m} (imported by {by})" for m, by in found.items()
                   if m not in serving.MODULES and m not in serving.LEFT_OUT)
    assert not stray, f"neither in MODULES nor in LEFT_OUT: {stray}"
    assert not set(serving.MODULES) & set(serving.LEFT_OUT)
    assert all(len(why) > 4 for why in serving.LEFT_OUT.values()), "each one says why"
    stale = sorted(set(serving.LEFT_OUT) - set(found))
    assert not stale, f"left out but imported by none: {stale}"
    assert {"engine/kvpool.py", "engine/capture_lock.py", "engine/config.py"} <= set(serving.LEFT_OUT)
    eng = "engine/models/nemotron_h/engine.py"

    def with_new_import(rel):
        return _read(rel) + "\n\ndef _later():\n    from engine import loader\n" if rel == eng else _read(rel)
    assert _engine_imports(with_new_import, serving.MODULES).get("engine/loader.py") == eng, \
        "an import inside a function counts"
    return f"{len(found)} engine/ modules imported, {len(serving.LEFT_OUT)} of them left out"


def test_nemotron_hs_manifest_is_pinned_and_a_left_out_module_moves_no_fingerprint():
    from engine.models.nemotron_h import serving
    assert serving.MODULES == (
        "engine/models/nemotron_h/__init__.py", "engine/models/nemotron_h/config.py",
        "engine/models/nemotron_h/engine.py", "engine/models/nemotron_h/gguf.py",
        "engine/models/nemotron_h/gguf_load.py", "engine/models/nemotron_h/kernels.py",
        "engine/models/nemotron_h/load.py", "engine/models/nemotron_h/pack.py",
        "engine/models/nemotron_h/ref.py", "engine/models/nemotron_h/vision.py",
        "engine/models/lfm2_moe/kernels.py", "engine/models/lfm2_moe/pack.py", "engine/settings.py",
        "engine/vision.py"), serving.MODULES
    m = serving.manifest()
    with tempfile.TemporaryDirectory() as d:
        for sub in ("engine", "tools"):
            shutil.copytree(os.path.join(ROOT, sub), os.path.join(d, sub),
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.so", "*.cu",
                                                          "*.safetensors", "*.bin"))
        fp = mf.fingerprint(d, m, {})
        for rel in serving.LEFT_OUT:
            with open(os.path.join(d, rel), "a") as f:
                f.write("\n# one more line\n")
            assert mf.fingerprint(d, m, {}) == fp, f"{rel} moved NemotronH's fingerprint"
        for rel in serving.MODULES:
            if rel == dc.SETTINGS_FILE:        # counts by its knobs' defaults, not its bytes
                continue
            with open(os.path.join(d, rel), "rb") as f:
                was = f.read()
            with open(os.path.join(d, rel), "ab") as f:
                f.write(b"\n# one more line\n")
            assert mf.fingerprint(d, m, {}) != fp, f"{rel} did not move NemotronH's fingerprint"
            with open(os.path.join(d, rel), "wb") as f:
                f.write(was)
        assert mf.fingerprint(d, m, {}) == fp
    return f"{len(serving.LEFT_OUT)} left out keep it, {len(serving.MODULES) - 1} listed move it"


if __name__ == "__main__":
    fails = 0
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        try:
            print(f"  {name:<72} ok   {fn() or ''}", flush=True)
        except AssertionError as e:
            fails += 1
            print(f"  {name:<72} FAIL {e}", flush=True)
    print(f"{len(tests) - fails} passed" + (f", {fails} FAILED" if fails else ""))
    sys.exit(1 if fails else 0)
