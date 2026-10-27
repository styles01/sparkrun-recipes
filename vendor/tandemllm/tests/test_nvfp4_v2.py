"""The v2 W4A16 kernel's routing, without a board.

Three properties, and the first is the one that would be a silent correctness bug rather than a
slow kernel. This engine's gate is bit-level equality between a speculative greedy run and a
non-speculative one; the non-speculative one decodes at M = 1 and the speculative one at the block
width. If the two land on different kernels they land on different fp32 accumulation orders, the
outputs differ in the last bf16 bit, and `tools/verify_spec.py` reports a speculation failure for
an arithmetic reason. So M = 1 and the block widths have to be routed the same way, whatever the
flag says -- which for the shipped default means all of them on v2.

The other two are the ones that took the board down at 17:29 on 2026-09-17 in a different form: a
config that reaches the launch missing a key, and a switch that does not switch.
"""

from __future__ import annotations

import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REQUIRED = ("block_m", "block_n", "split_k", "num_warps", "num_stages")
# the seven projection shapes a verify step reads, and two that are not in the table
SHAPES = [(17408, 5120), (5120, 17408), (10240, 5120), (6144, 5120), (5120, 6144),
          (12288, 5120), (1024, 5120), (5120, 5120), (248320, 5120)]


def _reload(**env):
    """Re-import the module under a given environment, and put the environment back."""
    old = {k: os.environ.get(k) for k in env}
    os.environ.update({k: v for k, v in env.items()})
    try:
        from tools import nvfp4_linear_v2
        return importlib.reload(nvfp4_linear_v2)
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_one_row_and_a_verified_block_take_the_same_kernel():
    v2 = _reload(QWEN38_NVFP4_V2="1")
    widths = [1, 2, 4, 8, 11, 14, 15, 16, 24, 32]
    routed = {m: v2.use_v2(m) for m in widths}
    assert len(set(routed.values())) == 1, (
        f"M = 1 and the block widths are routed to different kernels: {routed}")
    assert routed[1] is True, "the shipped default has to be v2 at every decode-side row count"
    _reload()
    return f"M in {widths} all on one kernel"


def test_every_shape_gets_a_complete_config_at_every_row_count():
    v2 = _reload()
    bad = []
    for (n, k) in SHAPES:
        for m in (1, 2, 8, 15, 16, 17, 24, 31, 32):
            cfg = v2.pick_config_v2(n, k, m)
            missing = [key for key in REQUIRED if key not in cfg]
            if missing:
                bad.append((n, k, m, missing))
    assert not bad, f"incomplete configs: {bad[:6]}"
    return f"{len(SHAPES)} shapes x 9 row counts, every key present"


def test_no_measured_tile_carries_split_k():
    """`split_k` is the M curve, so an entry that keeps it is a regression, not a tuning choice.

    A split-K launch writes SPLIT_K x M x N fp32 partials and reads them back to sum; the surcharge
    is linear in M and it is what made both kernels lose a third of their rate between one row and
    sixteen. Every measured entry dropped it and the table is only allowed to hold winners.
    """
    v2 = _reload()
    offenders = {shape: cfg for shape, cfg in v2._V2_CONFIG.items() if cfg.get("split_k", 1) != 1}
    assert not offenders, f"split_k survived in the table: {offenders}"
    assert v2._V2_FALLBACK["split_k"] == 1, "the fallback has to drop it too"
    return f"{len(v2._V2_CONFIG)} measured shapes, none split over K"


def test_the_switch_switches_and_the_range_is_honoured():
    assert _reload(QWEN38_NVFP4_V2="0").use_v2(14) is False, "'0' did not turn v2 off"
    assert _reload(QWEN38_NVFP4_V2="all").use_v2(4096) is True, "'all' did not mean all"
    win = _reload(QWEN38_NVFP4_V2="1", QWEN38_NVFP4_V2_MIN="4", QWEN38_NVFP4_V2_MAX="8")
    assert [win.use_v2(m) for m in (3, 4, 8, 9)] == [False, True, True, False], "range ignored"
    v2 = _reload()
    assert v2.use_v2(14) is True, "the default did not come back"
    return "off, all, an explicit range, and the default restored"


def test_a_whole_table_override_reaches_every_shape():
    """The in-engine A/B needs one tile on every shape, including the ones with no entry."""
    v2 = _reload(QWEN38_NVFP4_V2_BN="128", QWEN38_NVFP4_V2_W="8")
    for (n, k) in SHAPES:
        cfg = v2.pick_config_v2(n, k, 14)
        assert cfg["block_n"] == 128 and cfg["num_warps"] == 8, (n, k, cfg)
    v2 = _reload()
    assert v2.pick_config_v2(17408, 5120, 14)["block_n"] == 64, "the override outlived its env"
    return "override applies to measured and unmeasured shapes, and does not stick"


def _reload_v1(**env):
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        from tools import nvfp4_linear
        return importlib.reload(nvfp4_linear)
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_the_prefill_tile_owns_only_the_band_between_decode_and_unpack():
    """On, rows 33..UNTIL-1 take v2 with the prefill tile; the decode band stays on its
    kernel and the unpack path keeps everything from UNTIL up. Off, nothing moves."""
    on = _reload_v1(QWEN38_NVFP4_PREFILL_V2="1", QWEN38_NVFP4_PREFILL_V2_UNTIL="1024")
    assert [on.prefill_v2(m) for m in (1, 16, 32)] == [False] * 3
    assert all(on.prefill_v2(m) for m in (33, 64, 256, 512, 1023))
    assert not on.prefill_v2(1024) and not on.prefill_v2(8192)
    off = _reload_v1(QWEN38_NVFP4_PREFILL_V2="0")
    assert not any(off.prefill_v2(m) for m in (33, 256, 512, 1023))
    _reload_v1()
    return "33..1023 on the prefill tile, nothing when off"


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"  {name:58s} ok   {fn() or ''}")
            passed += 1
    print(f"{passed} passed")
