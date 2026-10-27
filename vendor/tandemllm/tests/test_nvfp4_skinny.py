"""The skinny CUDA kernel's routing and its tile table, without a board.

The kernel's own arithmetic is checked on the board by `tools/nvfp4_skinny.py --check` (against the
exact product and against v2, and row independence at every row count). What can be checked here is
the part that would be a silent correctness bug rather than a slow kernel: the losslessness gate
compares a greedy run decoding at M = 1 with a speculative one verifying 8 or 16 rows, so every
decode-side row count must land on the same kernel with the same K split, and the split must be a
function of the weight's shape alone.
"""

from __future__ import annotations

import importlib
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

SHAPES = [(17408, 5120), (5120, 17408), (10240, 5120), (6144, 5120), (5120, 6144),
          (12288, 5120), (1024, 5120), (34816, 5120), (14336, 5120), (16384, 5120)]


def _reload(**env):
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        from tools import nvfp4_skinny
        return importlib.reload(nvfp4_skinny)
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_off_by_default_and_the_switch_switches():
    sk = _reload()
    assert not any(sk.use_skinny(m) for m in (1, 8, 16, 32)), "on without the flag"
    sk = _reload(QWEN38_NVFP4_SKINNY="1")
    assert all(sk.use_skinny(m) for m in (1, 2, 8, 15, 16, 17, 32)), "flag on, a row count off"
    assert not sk.use_skinny(33) and not sk.use_skinny(0), "outside 1..32"
    _reload()
    return "off by default; on for 1..32 with QWEN38_NVFP4_SKINNY=1, never above"


def test_programmatic_dependent_launch_is_off_by_default():
    """QWEN38_SKINNY_PDL, and the norms release their dependents only when it is on."""
    sk = _reload()
    from tools import norm_kernels
    assert not sk.PDL and not norm_kernels._pdl()
    sk = _reload(QWEN38_NVFP4_SKINNY="1", QWEN38_SKINNY_PDL="1")
    try:
        assert sk.PDL and norm_kernels._pdl()
    finally:
        _reload()
    return "off by default; on only with the skinny kernel on"


def test_one_row_and_a_verified_block_take_the_same_kernel():
    from tools import nvfp4_linear
    sk = _reload(QWEN38_NVFP4_SKINNY="1")
    try:
        routed = {m: nvfp4_linear.use_skinny(m) for m in (1, 2, 4, 8, 11, 14, 15, 16, 24, 32)}
        assert set(routed.values()) == {True}, routed
    finally:
        _reload()
    return "M in 1..32 all on the skinny kernel when it is on"


def test_the_k_split_is_a_function_of_the_shape_only():
    """Gave `pick` a row count, for the 17..32-row tile table; the K split -- a row's
    summation order -- must still depend on the shape alone, at every row count."""
    sk = _reload()
    for table in (None, os.path.join(ROOT, "ops/skinny-tiles.json")):
        sk = _reload(**({"QWEN38_SKINNY_TILES": table} if table else {}))
        for (n, k) in SHAPES:
            assert len({sk.pick(n, k, m)["wk"] for m in range(1, 33)}) == 1, (n, k)
            cfg = sk.pick(n, k)
            assert set(cfg) - {"il", "kr"} == {"nt", "wk", "pf", "minb"}, cfg
            assert cfg["nt"] in (1, 2, 4, 8, 16) and cfg["wk"] in (1, 2, 4, 8, 16), cfg
            assert cfg["pf"] in (0, 1, 2) and cfg["minb"] in (1, 2), cfg
            # 512 threads at two CTAs an SM exists only for the 8-row tile
            assert not (cfg["wk"] == 16 and cfg["minb"] == 2 and cfg["nt"] != 1), cfg
            assert cfg.get("il", 0) in (0, 1), cfg
    _reload()
    return f"{len(SHAPES)} shapes, complete configs, one K split for 1..32 rows, served table included"


def test_the_second_table_is_off_until_the_switch_and_names_only_its_shapes():
    """The in-process A/B: QWEN38_SKINNY_TILES_B loads a second table and `ALT` routes the
    shapes it names through it; every other shape, and every shape with the switch off, keeps the
    first table's tile."""
    import json
    import tempfile
    alt = os.path.join(tempfile.mkdtemp(), "alt.json")
    json.dump({"5120x17408": {"nt": 1, "wk": 16, "pf": 2, "il": 1}}, open(alt, "w"))
    first = os.path.join(ROOT, "ops/skinny-tiles.json")
    sk = _reload(QWEN38_SKINNY_TILES=first, QWEN38_SKINNY_TILES_B=alt)
    try:
        before = {shp: sk.pick(*shp) for shp in SHAPES}
        assert not sk.ALT
        sk.ALT = True
        assert sk.pick(5120, 17408) == {"nt": 1, "wk": 16, "pf": 2, "minb": 1, "il": 1}
        assert all(sk.pick(*shp) == before[shp] for shp in SHAPES if shp != (5120, 17408))
        sk.ALT = False
        assert all(sk.pick(*shp) == before[shp] for shp in SHAPES)
    finally:
        _reload()
    return "ALT off = the first table; on = the second table's shapes only"


def test_the_wide_table_changes_the_tile_past_sixteen_rows_and_never_the_k_split():
    """QWEN38_SKINNY_TILES_WIDE is read for 17..32 rows only, and an entry that splits K
    differently from the shape's base tile is refused at load."""
    import json
    import tempfile
    wide = os.path.join(tempfile.mkdtemp(), "wide.json")
    json.dump({"17408x5120": {"nt": 4, "wk": 16, "pf": 1},          # same K split: taken
               "10240x5120": {"nt": 4, "wk": 8, "pf": 2}}, open(wide, "w"))  # other split: refused
    first = os.path.join(ROOT, "ops/skinny-tiles.json")
    sk = _reload(QWEN38_SKINNY_TILES=first)
    base = {shp: sk.pick(*shp) for shp in SHAPES}
    sk = _reload(QWEN38_SKINNY_TILES=first, QWEN38_SKINNY_TILES_WIDE=wide)
    try:
        for m in (1, 8, 16):
            assert all(sk.pick(*shp, m) == base[shp] for shp in SHAPES), m
        assert sk.pick(17408, 5120, 17) == {"nt": 4, "wk": 16, "pf": 1, "minb": 1, "il": 0}
        assert sk.pick(17408, 5120, 32)["nt"] == 4
        assert sk.pick(10240, 5120, 32) == base[(10240, 5120)], "the other K split was not refused"
        assert all(sk.pick(*shp, 32) == base[shp] for shp in SHAPES if shp != (17408, 5120))
    finally:
        _reload()
    return "1..16 rows = the base table; 17..32 = the wide entry; a wide entry with another K split refused"


def test_the_kr1_wide_table_loads_whole_and_orders_by_register_past_sixteen_rows():
    """Ops/skinny-tiles-wide-kr1.json keeps every target shape (none refused for its K split),
    puts the six at nt4:wk16:pf0 with the register-sequential order past 16 rows, and leaves 1..16 rows
    and the drafter's shape on their tiles."""
    first = os.path.join(ROOT, "ops/skinny-tiles.json")
    sk = _reload(QWEN38_SKINNY_TILES=first)
    base = {shp: sk.pick(*shp) for shp in SHAPES}
    sk = _reload(QWEN38_SKINNY_TILES=first,
                 QWEN38_SKINNY_TILES_WIDE=os.path.join(ROOT, "ops/skinny-tiles-wide-kr1.json"))
    try:
        target = SHAPES[:6]
        assert all(shp in sk._WIDE for shp in target), "an entry was refused at load"
        for m in (1, 8, 16):
            assert all(sk.pick(*shp, m) == base[shp] for shp in SHAPES), m
        for m in (17, 24, 32):
            for shp in target:
                assert sk.pick(*shp, m) == {"nt": 4, "wk": 16, "pf": 0, "minb": 1, "il": 0, "kr": 1}, (shp, m)
                assert sk.pick(*shp, m)["wk"] == base[shp]["wk"]
        assert "kr" not in sk.pick(1024, 5120, 24)
    finally:
        _reload()
    return "6 target shapes at nt4:pf0:kr1 for 17..32 rows, same K split; 1..16 rows unchanged"


def test_the_second_wide_table_is_off_until_the_switch_and_keeps_the_k_split():
    """QWEN38_SKINNY_TILES_WIDE_B and `WIDE_B` route the shapes it names past sixteen rows
    only while the switch is on; 1..16 rows never see it; an entry with another K split is refused."""
    import json
    import tempfile
    wb = os.path.join(tempfile.mkdtemp(), "wide_b.json")
    json.dump({"17408x5120": {"nt": 4, "wk": 16, "pf": 0, "kr": 1, "ldw": 1},
               "10240x5120": {"nt": 4, "wk": 8, "pf": 0}}, open(wb, "w"))
    first = os.path.join(ROOT, "ops/skinny-tiles.json")
    wide = os.path.join(ROOT, "ops/skinny-tiles-wide.json")
    sk = _reload(QWEN38_SKINNY_TILES=first, QWEN38_SKINNY_TILES_WIDE=wide,
                 QWEN38_SKINNY_TILES_WIDE_B=wb)
    try:
        before = {(shp, m): sk.pick(*shp, m) for shp in SHAPES for m in (1, 16, 17, 24, 32)}
        assert not sk.WIDE_B and (10240, 5120) not in sk._WIDE_B, "the other K split was taken"
        sk.WIDE_B = True
        assert sk.pick(17408, 5120, 24) == {"nt": 4, "wk": 16, "pf": 0, "minb": 1, "il": 0, "kr": 1,
                                            "ldw": 1}
        for (shp, m), cfg in before.items():
            if shp == (17408, 5120) and m > 16:
                continue
            assert sk.pick(*shp, m) == cfg, (shp, m)
        sk.WIDE_B = False
        assert all(sk.pick(*shp, m) == cfg for (shp, m), cfg in before.items())
    finally:
        _reload()
    return "WIDE_B off = the served tables; on = its shapes past 16 rows only; another K split refused"


def test_each_weight_load_hint_is_its_own_module_built_once():
    """Flipping `LDW` in a process picks the module built for that hint, builds each hint
    once, and flipping back returns the first module, not a rebuild."""
    sk = _reload()
    built = []

    class Fake:
        def __init__(self, name):
            self.name = name

    import torch.utils.cpp_extension as ce
    orig = ce.load_inline
    ce.load_inline = lambda name, **kw: built.append(name) or Fake(name)
    try:
        sk.LDW = 0
        m0 = sk._module()
        sk.LDW = 1
        m1 = sk._module()
        sk.LDW = 0
        again = sk._module()
        sk.LDW = 1
        assert sk._module() is m1
        # a tile entry's own hint (the 17..32-row tile alone): that hint's module, LDW's untouched
        sk.LDW = 0
        assert sk._module(1) is m1 and sk._module() is m0 and sk._module(0) is m0
        m3 = sk._module(3)
        assert m3 is not m0 and m3 is not m1 and sk._module() is m0
    finally:
        ce.load_inline = orig
        _reload()
    assert m0 is again and m0 is not m1 and m0.name != m1.name
    assert len(built) == 3, built
    return "hints -> modules, each built once, flipping back reuses; an entry's own hint leaves LDW's"


def test_an_explicit_tile_does_not_inherit_the_tables_order_or_layout():
    """With kr1 as the served 17..32-row table, a caller that names its own tile (nt / wk / pf, a
    test or a bench) must get that tile, not kr1's register order, a slice-serial layout or the
    entry's own hint; a caller that names nothing gets the table entry whole."""
    first = os.path.join(ROOT, "ops/skinny-tiles.json")
    import json
    import tempfile
    wide = os.path.join(tempfile.mkdtemp(), "wide.json")
    json.dump({"17408x5120": {"nt": 4, "wk": 16, "pf": 0, "kr": 1, "ldw": 1}}, open(wide, "w"))
    sk = _reload(QWEN38_SKINNY_TILES=first, QWEN38_SKINNY_TILES_WIDE=wide)
    calls = []

    class Mod:
        def skinny(self, *a):
            calls.append(("skinny", a[6:]))

        def skinny_ser(self, *a):
            calls.append(("ser", a[6:]))

    class W:
        N, K, s2, sizes = 17408, 5120, 1.0, None

        def __init__(self):
            import torch
            self.w = torch.zeros(1, dtype=torch.uint8)
            self.s = torch.zeros(1, dtype=torch.float8_e4m3fn)

    import torch
    orig = sk._module
    sk._module = lambda ldw=None: (calls.append(("ldw", ldw)), Mod())[1]
    sk.SRUN = 0
    try:
        x = torch.zeros(24, 5120, dtype=torch.bfloat16)
        out = torch.zeros(24, 17408, dtype=torch.bfloat16)
        sk.nvfp4_matmul_skinny(x, W(), out=out)                    # the table: kr1 and its hint
        sk.nvfp4_matmul_skinny(x, W(), out=out, nt=1, wk=16, pf=0)  # an explicit tile: kr0, LDW's
        sk.nvfp4_matmul_skinny(x, W(), out=out, nt=4, wk=16, pf=0, kr=1)
    finally:
        sk._module = orig
        _reload()
    # the kernel's arguments from nt on: (nt, wk, pf, minb, il, pdl, kr, spw)
    assert calls[0] == ("ldw", 1) and calls[1][0] == "skinny" and calls[1][1][-2] == 1, calls
    assert calls[2] == ("ldw", None) and calls[3][1][:3] == (1, 16, 0) and calls[3][1][-2] == 0, calls
    assert calls[5][1][-2] == 1, calls
    return "the table's entry whole; an explicit tile without the entry's kr and hint; kr asked for, kr given"


def test_the_served_environment_turns_the_scale_runs_on_and_the_code_default_stays_off():
    """Adopted in phase5: ops/serve.env sets QWEN38_SKINNY_SRUN=1, which the module reads; without it
    (a test, a tool, the gate's clean environment) the kernel reads the stored scales as before."""
    lines = [ln.strip() for ln in open(os.path.join(ROOT, "ops/serve.env")) if not ln.lstrip().startswith("#")]
    assert "QWEN38_SKINNY_SRUN=1" in lines
    assert _reload(QWEN38_SKINNY_SRUN="1").SRUN == 1
    os.environ.pop("QWEN38_SKINNY_SRUN", None)
    assert _reload().SRUN == 0
    return "serve.env: SRUN=1; the code default 0"


def test_the_scale_runs_are_a_permutation_of_the_scales():
    """Run (G, q) is rows 16G..16G+15, bytes 8q..8q+7 of each, 128 contiguous bytes; every
    scale byte lands exactly once where the kernel's address formula reads it, rows past N (up to
    a multiple of 16) are zero, and nothing else is in the copy."""
    import torch
    sk = _reload()
    g = torch.Generator().manual_seed(52)
    for N, K in ((16, 128), (24, 256), (1000, 5120), (5120, 17408), (8, 1152)):
        s = torch.randint(0, 256, (N, K // 16), generator=g, dtype=torch.uint8)
        run = sk.scale_runs(s)
        KQ = K // 128
        n16 = (N + 15) // 16 * 16
        assert run.numel() == n16 * K // 16, (N, K, run.numel())
        r = torch.arange(N)[:, None, None]
        q = torch.arange(KQ)[None, :, None]
        b = torch.arange(8)[None, None, :]
        idx = (((r // 16) * KQ + q) * 16 + r % 16) * 8 + b          # the kernel's formula
        assert torch.equal(run[idx.reshape(-1)].view(N, KQ * 8), s), (N, K)
        seen = torch.zeros(run.numel(), dtype=torch.bool)
        seen[idx.reshape(-1)] = True
        assert int(seen.sum()) == N * KQ * 8 and not run[~seen].any(), (N, K)
    return "5 shapes (odd N, one K step, the down shape): every byte where the kernel reads it"


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"  {name:56s} ok   {fn() or ''}")
            passed += 1
    print(f"{passed} passed")
