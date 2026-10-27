"""FX15 (2026-10-08): what the disk tier's fingerprint reads, and the one adoption of the files the
formula before it wrote.

THE INCIDENT. 0.4.6-test changed nothing under `engine/` but `engine/settings.py`, where seven
server knobs (the regenerate rule, the memory proxies, a package's address) were registered. The
fingerprint hashed every engine module's bytes, so that alone swept 13 shared heads (12.4 GB) and
13 spilled conversations (29.6 GB) at the 01:44 start; UX3's first Open WebUI chat re-read its
6.7k-token head cold (TTFT 14.3 s against 2.7 s). At the same time the fingerprint never read the
`tools/` kernels the engine imports, so a kernel change kept rows another kernel wrote.

THE RULES pinned here: a registry edit for a knob the server reads keeps the fingerprint; a changed
default of a knob the engine or a kernel reads changes it, and so does any imported `tools/`
module; the first start under the new formula adopts the files of the old one on the same tree,
reads them bit for bit as a cold prefill, and records that, so a later build reads them only under
the same fingerprint.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import disk_cache as dc  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP = ("cache.py", "disk_cache.py", "spill.py", "capture_lock.py")


def _tree(d):
    """A copy of the repository's `engine/` and `tools/` modules under `d`."""
    for sub in ("engine", "tools"):
        shutil.copytree(os.path.join(ROOT, sub), os.path.join(d, sub),
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.so", "*.cu",
                                                      "*.safetensors", "*.bin"))
    return d


def _edit(path, old, new):
    with open(path) as f:
        s = f.read()
    assert old in s, (path, old)
    with open(path, "w") as f:
        f.write(s.replace(old, new, 1))


def test_registering_a_server_knob_keeps_the_fingerprint():
    with tempfile.TemporaryDirectory() as d:
        _tree(d)
        a, la = dc.fingerprint(d, {}, skip=SKIP), dc.fingerprint_legacy(d, {}, skip=SKIP)
        _edit(os.path.join(d, "engine", "settings.py"), "KNOBS: dict[str, tuple] = {",
              "KNOBS: dict[str, tuple] = {\n    # a server feature's switch\n"
              "    'SOME_SERVER_SWITCH': ('on', 'server/app.py'),")
        assert dc.fingerprint(d, {}, skip=SKIP) == a
        # the formula before FX15 swept everything for this (0.4.6-test, 01:44)
        assert dc.fingerprint_legacy(d, {}, skip=SKIP) != la


def test_an_arithmetic_default_changes_the_fingerprint():
    with tempfile.TemporaryDirectory() as d:
        _tree(d)
        a = dc.fingerprint(d, {}, skip=SKIP)
        _edit(os.path.join(d, "engine", "settings.py"), "'WY_PREC': ('ieee',",
              "'WY_PREC': ('tf32',")
        assert dc.fingerprint(d, {}, skip=SKIP) != a


def test_an_imported_kernel_changes_the_fingerprint_and_an_unused_tool_does_not():
    with tempfile.TemporaryDirectory() as d:
        _tree(d)
        mods = [os.path.basename(p) for p in dc._tools_modules(d, dc._engine_files(d, SKIP))]
        assert "gdn_prefill_kernels.py" in mods and "nvfp4_linear.py" in mods
        a, la = dc.fingerprint(d, {}, skip=SKIP), dc.fingerprint_legacy(d, {}, skip=SKIP)
        with open(os.path.join(d, "tools", "gdn_prefill_kernels.py"), "a") as f:
            f.write("\n# another tile\n")
        assert dc.fingerprint(d, {}, skip=SKIP) != a
        assert dc.fingerprint_legacy(d, {}, skip=SKIP) == la        # the gap the old one had
        b = dc.fingerprint(d, {}, skip=SKIP)
        with open(os.path.join(d, "tools", "fx15_unused_helper.py"), "w") as f:
            f.write("X = 1\n")
        assert dc.fingerprint(d, {}, skip=SKIP) == b


def test_the_first_start_adopts_the_old_formula_once_and_records_it():
    with tempfile.TemporaryDirectory() as d:
        assert dc.equivalents(d, "new", "old") == ("old",)
        with open(os.path.join(d, dc.EQUIV_FILE)) as f:
            assert json.load(f) == {"new": ["old"]}
        # a restart of the same build: the record, whatever the old formula says now
        assert dc.equivalents(d, "new", "old after a registry edit") == ("old",)
        # a build with other arithmetic: nothing of the record, and no new adoption
        assert dc.equivalents(d, "newer", "old") == ()
    with tempfile.TemporaryDirectory() as d:
        assert dc.equivalents(d, "same", "same") == ()


# ------------------------------------------------------------------ the tier and the spill read them

def test_the_tier_reads_adopted_files_bit_for_bit_and_a_later_formula_sweeps_them():
    import test_forward_tree as T  # noqa: F401  (the CPU environment before the engine imports)
    from engine import cache
    from test_cache import fresh, tokens
    from test_disk_cache import Pos, exact, files, run, tier

    x = tokens(120, seed=91)
    with tempfile.TemporaryDirectory() as d:
        t = tier(d, fp="the 0.4.6 formula", settings={"admit_after": 1})
        run(fresh(), x, t, Pos())
        t.close()
        n_files = len(files(d, "chunks")) + len(files(d, "anchors"))
        assert n_files > 0
        also = dc.equivalents(d, "the FX15 formula", "the 0.4.6 formula")
        t2 = tier(d, fp="the FX15 formula", also=also, settings={"admit_after": 1})
        assert t2.stats["swept_fingerprint"] == 0 and t2.report()["entries"] == 1
        eng, dr = fresh(), Pos()
        lg, reused, kind = run(eng, x + [3, 4], t2, dr)
        assert (kind, reused) == ("disk", 112)
        exact(eng, lg, x + [3, 4], dr)
        assert t2.counts[(112, cache.prefix_hashes(x)[112])] >= 1
        t2.close()
        # a restart of the same build reads them through the record
        t3 = tier(d, fp="the FX15 formula",
                  also=dc.equivalents(d, "the FX15 formula", "anything"),
                  settings={"admit_after": 1})
        assert t3.stats["swept_fingerprint"] == 0 and t3.report()["entries"] == 1
        t3.close()
        # a build with other arithmetic sweeps them
        t4 = tier(d, fp="a kernel change", also=dc.equivalents(d, "a kernel change", "x"),
                  settings={"admit_after": 1})
        assert t4.stats["swept_fingerprint"] == n_files and t4.report()["entries"] == 0
        t4.close()


def test_the_session_spill_reads_adopted_entries():
    import test_forward_tree as T  # noqa: F401
    from engine import spill
    from server import app
    from test_session_spill import _spill
    from test_end_anchor import _state, turn
    from test_cache import tokens

    d = tempfile.mkdtemp(prefix="qse-spill-")
    try:
        res = _state()
        sp, t = _spill(d, res, fp="the 0.4.6 formula")
        turn(tokens(200, seed=92), 20, session="A")
        sp.save_resident(res, app.STATE["engine"], None, "drain")
        assert len(sp.entries) == 1
        t.close()
        sp2 = spill.SessionSpill(os.path.join(d, "tier", "sessions"), chunk=sp.chunk,
                                 fingerprint="the FX15 formula", also=("the 0.4.6 formula",),
                                 settings={"budget_gb": 1.0, "min_tokens": 64}).open()
        assert len(sp2.entries) == 1 and sp2.stats["swept"] == 0
        sp3 = spill.SessionSpill(os.path.join(d, "tier", "sessions"), chunk=sp.chunk,
                                 fingerprint="a kernel change",
                                 settings={"budget_gb": 1.0, "min_tokens": 64}).open()
        assert not sp3.entries and sp3.stats["swept"] == 1
    finally:
        shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:72s} ok")
            passed += 1
    print(f"{passed} passed")
