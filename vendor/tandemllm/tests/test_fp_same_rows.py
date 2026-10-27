"""0.8.2: a module whose bytes changed without changing a stored row keeps the saved conversations
(engine/methods/manifest.py `SAME_ROWS`). CPU only.

#1 (0xWhiteMage, a6f2142c) changed engine/loader.py to load a standalone EXL3 pack. The file is in
the 27B's v3 manifest (engine/models/qwen3_5/adapter.py MODULES) and in v2's engine walk, so its
bytes alone changed every 27B fingerprint, and the next 27B start would have swept every saved chat
(on testers' boxes too), although a 27B profile (QWEN38_EXL3 set, or NVFP4) takes the old path.

  * the entry names this tree's loader.py, and with it the 27B's v2 code fingerprint is 0.7.2's
    (9ac3d8a94c8a9199, the number the release notes give since 0.7.2);
  * a fingerprint with `same` reads the replaced bytes: v2 and v3 of the new tree equal the old
    tree's; without it they differ;
  * the legacy list holds the old tree's fp4 (its extra by content, as a start computes it), and a
    record the old tree's start left adopts it, with what it had adopted, once, under a new key;
  * a directory whose fingerprint did not change (Flash-Next's: loader.py is not in its manifest)
    keeps what it had adopted when the new key adopts again.

ENG-341: NemotronH's LiLiCorr draft head (engine/drafters/lilicorr.py) is in no 27B manifest, but v2
walks every module under engine/, so it moved v2 of the 27B off 0.9.x's. v2 leaves it out
(engine/disk_cache.py `_NOT_27B_FILES`); and SPD-125's new fn_kernels.py bytes, read one step back,
gave a tree that never shipped for Flash-Next and LFM2.5, so their 0.9.x files would not be adopted:

  * the 27B's v2 and v3 code fingerprints are 0.9.x's, on the bytes and with SAME_ROWS;
  * a change to lilicorr.py moves no 27B fingerprint;
  * Flash-Next's and LFM2.5's legacy fingerprints hold their 0.9.x fp4 (`same_variants`);
  * a tree two releases on adopts the files the release before it wrote, and that one's.

OPS-215: ENG-383 made tools/fn_kernels.py's router take up to 512 experts (it refused any count but 512),
so Flash-Next's, LFM2.5's and NemotronH's fingerprints moved off 0.12.2's with no entry. At 512 experts the
router launches as before, and the entry is in:

  * Flash-Next's, LFM2.5's and NemotronH's legacy fingerprints hold 0.12.2's fp4.

ENG-381: ENG-363 (0.11.2) changed engine/models/lfm2_moe/engine.py and kernels.py with no entry, so
LFM2.5's legacy fingerprints reached neither 0.9.x's fp4 nor 0.10.0-0.11.1's. The rows are the same
(tools/lfm_attn_rows_check.py), and the entries are in:

  * LFM2.5's legacy fingerprints hold 0.10.0-0.11.1's fp4 too;
  * an edit to an LFM2.5 module with no entry reaches neither.

Run: python tests/test_fp_same_rows.py
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test_fp_equivalents import _edit, _tree as _code  # noqa: E402
from test_tier_identity import Env, _args, _tree  # noqa: E402

from engine import disk_cache as dc  # noqa: E402
from engine import spill as sp  # noqa: E402
from engine.methods import manifest as mf  # noqa: E402
from engine.models.qwen3_5 import adapter  # noqa: E402
from engine.models.flashnext import serving as fn_serving  # noqa: E402
from server import diskcache  # noqa: E402

from engine.models.lfm2_moe import serving as lfm_serving  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOADER = "engine/loader.py"
LILICORR = "engine/drafters/lilicorr.py"
M27 = adapter.manifest()
#: the code fingerprints of 0.9.0-0.9.2 (extra {}), the first 16 hex digits: v2 on the bytes and with
#: SAME_ROWS (0.7.2's), and fp4 by model on the bytes
V2_09, V2_072 = "a0151bf64448bd92", "9ac3d8a94c8a9199"
FP4_09 = {"27b": "432dc8e260606f4c", "flashnext": "cc8c32668aa6aa65", "lfm25": "aa8db59af76a1c98"}
#: LFM2.5's fp4 of 0.10.0-0.11.1 (SPD-125's fn_kernels.py, lfm2_moe before ENG-363)
FP4_LFM25_011 = "9049446866c5e4fc"
#: OPS-215: the fp4 of 0.12.2 by model (extra {}), before ENG-383's router took other expert counts
FP4_0122 = {"flashnext": "dac45dfe10281b1e", "lfm25": "899978df1e8b979b", "nemotron_h": "8cfead8de4befc19"}


def _sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def test_the_entry_names_this_loader_and_keeps_the_27bs_code_fingerprint():
    now = _sha(os.path.join(ROOT, LOADER))
    assert (LOADER, now) in mf.SAME_ROWS, "SAME_ROWS names another loader.py: update it"
    assert LOADER in M27.modules
    code = dc.fingerprint(ROOT, {}, skip=sp.NOT_ARITHMETIC)
    same = dc.fingerprint(ROOT, {}, skip=sp.NOT_ARITHMETIC, same=mf.SAME_ROWS)
    assert same.startswith(V2_072), same
    assert not code.startswith(V2_072), "the loader's bytes count without the entry"
    assert mf.adoption() != mf.FORMULA and mf.adoption().startswith(mf.FORMULA + "+same-")
    return f"v2 code {code[:16]} on the bytes, {same[:16]} with SAME_ROWS (0.7.2's)"


#: SRV-619: engine/grammar.py changed after 0.9 without changing a stored row (a mask on the sampled
#: token); it is in no model's manifest, so fp4 is untouched, and v2's engine walk reads its 0.9 bytes
AFTER_09 = {k: v for k, v in mf.SAME_ROWS.items() if k[0] == "engine/grammar.py"}


def test_the_27bs_code_fingerprints_are_0_9s():
    v2 = dc.fingerprint(ROOT, {}, skip=sp.NOT_ARITHMETIC, same=AFTER_09)
    v3 = mf.fingerprint(ROOT, M27, {})
    assert v2.startswith(V2_09), v2
    assert v3.startswith(FP4_09["27b"]), v3
    return f"v2 {v2[:16]}, fp4 {v3[:16]}"


def test_lilicorr_is_in_no_27b_fingerprint():
    assert os.path.isfile(os.path.join(ROOT, LILICORR))
    with tempfile.TemporaryDirectory() as d:
        t = _code(d)
        before = (dc.fingerprint(t, {}, skip=sp.NOT_ARITHMETIC), dc.fingerprint_legacy(t, {}, skip=sp.NOT_ARITHMETIC),
                  mf.fingerprint(t, M27, {}))
        _edit(os.path.join(t, LILICORR), "from __future__", "# another draft\nfrom __future__")
        assert (dc.fingerprint(t, {}, skip=sp.NOT_ARITHMETIC), dc.fingerprint_legacy(t, {}, skip=sp.NOT_ARITHMETIC),
                mf.fingerprint(t, M27, {})) == before
        # a module of the 27B's own under engine/drafters/ still counts
        _edit(os.path.join(t, "engine", "drafters", "ngram.py"), "from __future__", "# x\nfrom __future__")
        assert dc.fingerprint(t, {}, skip=sp.NOT_ARITHMETIC) != before[0]


def test_flash_next_and_lfm25_adopt_their_0_9_files():
    out = []
    for name, m in (("flashnext", fn_serving.manifest()), ("lfm25", lfm_serving.manifest())):
        now = mf.fingerprint(ROOT, m, {})
        assert not now.startswith(FP4_09[name]), "fn_kernels.py's bytes are 0.9.x's: drop its SPD-125 entry"
        one = mf.fingerprint(ROOT, m, {}, same=mf.SAME_ROWS)
        assert not one.startswith(FP4_09[name]), "SAME_ROWS alone reads exl3_linear.py as 0.8.2's"
        got = [mf.fingerprint(ROOT, m, {}, same=s) for s in mf.same_variants(ROOT, m)]
        assert any(f.startswith(FP4_09[name]) for f in got), (name, [f[:16] for f in got])
        out.append(f"{name} {len(got)} variants")
    return ", ".join(out)


def test_lfm25_adopts_0_11_1s_files_and_an_edit_without_an_entry_breaks_it():
    m = lfm_serving.manifest()
    got = [mf.fingerprint(ROOT, m, {}, same=s)[:16] for s in mf.same_variants(ROOT, m)]
    assert FP4_LFM25_011 in got and FP4_09["lfm25"] in got, ("lfm25", got)
    with tempfile.TemporaryDirectory() as d:
        t = _code(d)
        _edit(os.path.join(t, "engine", "models", "lfm2_moe", "kernels.py"), "from __future__",
              "# x\nfrom __future__")
        edited = [mf.fingerprint(t, m, {}, same=s)[:16] for s in mf.same_variants(t, m)]
    assert FP4_LFM25_011 not in edited and FP4_09["lfm25"] not in edited, ("lfm25", edited)
    return f"0.9.x {FP4_09['lfm25']} and 0.11.1 {FP4_LFM25_011} among {len(got)} variants"


def test_a_tree_two_releases_on_adopts_the_release_befores_files():
    d = tempfile.mkdtemp(prefix="qse-same2-")
    real = mf.SAME_ROWS
    try:
        # as 0.9.0 (exl3_linear.py and fn_kernels.py) and SPD-125 (fn_kernels.py again) did
        r0 = _code(os.path.join(d, "r0"))
        r1, r2 = os.path.join(d, "r1"), os.path.join(d, "r2")
        tree = os.path.join("engine", "tree.py")
        shutil.copytree(r0, r1)
        _edit(os.path.join(r1, LOADER), "class Weights", "# release 1\nclass Weights")
        _edit(os.path.join(r1, tree), "from __future__", "# release 1\nfrom __future__")
        shutil.copytree(r1, r2)
        _edit(os.path.join(r2, tree), "from __future__", "# release 2\nfrom __future__")
        sha = lambda r, p: _sha(os.path.join(r, p))  # noqa: E731
        t = _tree(d, "box", mtime=1.7e9)
        tier = os.path.join(d, "tier")
        os.makedirs(tier)
        with Env(t, added=True):
            extra = diskcache.fingerprint_extra(_args(t))
            fp = {r: mf.fingerprint(r, M27, extra) for r in (r0, r1, r2)}
            # release 1's start: its new bytes read as release 0's, release 0's files adopted
            mf.SAME_ROWS = {(LOADER, sha(r1, LOADER)): sha(r0, LOADER), (tree, sha(r1, tree)): sha(r0, tree)}
            assert fp[r0] in diskcache.legacy_fingerprints(r1, M27, _args(t), tree=t)
            dc.equivalents(tier, fp[r1], diskcache.legacy_fingerprints(r1, M27, _args(t), tree=t),
                           formula=mf.adoption())
            # release 2: tree.py's bytes again; release 1's entries stay
            mf.SAME_ROWS = {(LOADER, sha(r1, LOADER)): sha(r0, LOADER), (tree, sha(r1, tree)): sha(r0, tree),
                            (tree, sha(r2, tree)): sha(r1, tree)}
            leg = diskcache.legacy_fingerprints(r2, M27, _args(t), tree=t)
            one = mf.fingerprint(r2, M27, extra, same=mf.SAME_ROWS)
        assert one not in (fp[r0], fp[r1]), "SAME_ROWS alone reads a tree no release had"
        assert fp[r1] in leg, "release 1's own files"
        also = dc.equivalents(tier, fp[r2], leg, formula=mf.adoption())
        assert {fp[r0], fp[r1]} <= set(also), also
        return f"{fp[r2][:12]} adopts {fp[r1][:12]} and {fp[r0][:12]}"
    finally:
        mf.SAME_ROWS = real
        shutil.rmtree(d, ignore_errors=True)


def test_flash_next_does_not_read_the_loader():
    files = {os.path.relpath(p, ROOT) for p in mf.files(ROOT, fn_serving.manifest())}
    assert LOADER not in files, files


def test_a_fingerprint_with_same_reads_the_replaced_bytes():
    with tempfile.TemporaryDirectory() as d:
        old, new = _code(os.path.join(d, "old")), _code(os.path.join(d, "new"))
        _edit(os.path.join(new, LOADER), "class Weights", "# a standalone pack's path\nclass Weights")
        same = {(LOADER, _sha(os.path.join(new, LOADER))): _sha(os.path.join(old, LOADER))}
        x = {"weights": "a", "grid": 1024}
        v2 = lambda t, s=None: dc.fingerprint(t, x, skip=sp.NOT_ARITHMETIC, same=s)  # noqa: E731
        assert v2(new) != v2(old) and v2(new, same) == v2(old)
        assert mf.fingerprint(new, M27, x) != mf.fingerprint(old, M27, x)
        assert mf.fingerprint(new, M27, x, same=same) == mf.fingerprint(old, M27, x)
        assert mf.fingerprint(new, M27, x, formula="fp3", same=same) == \
            mf.fingerprint(old, M27, x, formula="fp3")
        # another file's change is not covered by the entry
        _edit(os.path.join(new, "engine", "tree.py"), "from __future__", "# x\nfrom __future__")
        assert mf.fingerprint(new, M27, x, same=same) != mf.fingerprint(old, M27, x)


def test_the_old_trees_fp4_is_adopted_once_with_its_chain():
    d = tempfile.mkdtemp(prefix="qse-same-")
    real = mf.SAME_ROWS
    try:
        old, new = _code(os.path.join(d, "old")), _code(os.path.join(d, "new"))
        _edit(os.path.join(new, LOADER), "class Weights", "# a standalone pack's path\nclass Weights")
        mf.SAME_ROWS = {(LOADER, _sha(os.path.join(new, LOADER))): _sha(os.path.join(old, LOADER))}
        t = _tree(d, "box", mtime=1.7e9)
        tier = os.path.join(d, "tier")
        os.makedirs(tier)
        with Env(t, added=True):
            extra = diskcache.fingerprint_extra(_args(t))
            fp_old, fp_new = mf.fingerprint(old, M27, extra), mf.fingerprint(new, M27, extra)
            leg = diskcache.legacy_fingerprints(new, M27, _args(t), tree=t)
        assert fp_old != fp_new and fp_old in leg
        # what the old tree's first fp4 start left: its fp4, and what that one adopted (0.7.2's)
        with open(os.path.join(tier, dc.EQUIV_FILE), "w") as f:
            json.dump({"formulas": {mf.FORMULA: fp_old}, fp_old: ["0.7.2's v2", "the RC's fp3"]}, f)
        also = dc.equivalents(tier, fp_new, leg, formula=mf.adoption())
        assert {fp_old, "0.7.2's v2", "the RC's fp3"} <= set(also), also
        assert dc.equivalents(tier, fp_new, ["anything"], formula=mf.adoption()) == also, "once"
        # without the entry (the formula's key alone) nothing is adopted: the files would be swept
        with open(os.path.join(tier, dc.EQUIV_FILE), "w") as f:
            json.dump({"formulas": {mf.FORMULA: fp_old}, fp_old: ["0.7.2's v2"]}, f)
        assert dc.equivalents(tier, fp_new, leg, formula=mf.FORMULA) == ()
        return f"{fp_new[:12]} adopts {fp_old[:12]} and its {len(also) - 1} older ones"
    finally:
        mf.SAME_ROWS = real
        shutil.rmtree(d, ignore_errors=True)


def test_an_unchanged_fingerprint_keeps_what_it_adopted():
    with tempfile.TemporaryDirectory() as tier:
        with open(os.path.join(tier, dc.EQUIV_FILE), "w") as f:
            json.dump({"formulas": {mf.FORMULA: "flash-next"}, "flash-next": ["the RC's fp3", "v2"]}, f)
        also = dc.equivalents(tier, "flash-next", ["flash-next", "v2 on the old loader"],
                              formula=mf.adoption())
        assert set(also) == {"the RC's fp3", "v2", "v2 on the old loader"}, also


def test_flash_next_lfm25_and_nemotron_h_adopt_0_12_2s_files():
    from engine.models.nemotron_h import serving as nh_serving
    out = []
    for name, m in (("flashnext", fn_serving.manifest()), ("lfm25", lfm_serving.manifest()),
                    ("nemotron_h", nh_serving.manifest())):
        got = [mf.fingerprint(ROOT, m, {}, same=s) for s in mf.same_variants(ROOT, m)]
        assert any(g.startswith(FP4_0122[name]) for g in got), (name, [g[:16] for g in got])
        out.append(f"{name} {FP4_0122[name]}")
    return ", ".join(out)


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
