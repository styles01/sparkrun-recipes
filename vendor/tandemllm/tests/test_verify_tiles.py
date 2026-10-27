"""A tile registered for the verify row counts must be a COMPLETE config.

`set_config` replaces a bucket rather than updating it, so a dict that is missing one key does not
fail where it is written -- it fails in `nvfp4_matmul`'s launch, with a `KeyError`, on the first
projection of the first forward. That is the server's warm-up, so the failure takes the board down
for every track on it. It did, at 17:29 on 2026-09-17, for eleven minutes.

The fix is that `tools/nvfp4_verify_tiles.py` overlays its measured numbers onto the fallback rather
than replacing it. The test is that whatever comes back out of `pick_config` has every key the
launch reads, at every row count the engine can ask for -- which is the property that was actually
violated, rather than the particular key that happened to be missing.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# every key `nvfp4_matmul` reads out of a config before it launches
REQUIRED = ("block_m", "block_n", "split_k", "num_warps", "num_stages")


def test_every_registered_tile_is_a_complete_config():
    from tools import nvfp4_verify_tiles as vt
    from tools.nvfp4_linear import pick_config

    vt.register()
    shapes = list(vt.MEASURED) + [(5120, 5120), (248320, 5120)]     # registered, and not
    bad = []
    for (n, k) in shapes:
        for m in (1, 2, 8, 15, 16, 17, 24, 31, 32, 33, 48, 64, 96, 128, 129, 256, 512):
            cfg = pick_config(n, k, m)
            missing = [key for key in REQUIRED if key not in cfg]
            if missing:
                bad.append((n, k, m, missing))
    assert not bad, f"incomplete configs: {bad[:6]}"
    return f"{len(shapes)} shapes x 17 row counts, every key present"


def test_the_measured_numbers_survive_the_overlay():
    """The point of the overlay is to add keys, not to lose the ones that were measured."""
    from tools import nvfp4_verify_tiles as vt
    from tools.nvfp4_linear import pick_config

    vt.register()
    cfg = pick_config(17408, 5120, 24)              # decode bucket
    assert cfg["block_n"] == 32, cfg
    assert cfg["split_k"] == 1, cfg
    assert cfg["num_warps"] == 4, cfg
    cfg = pick_config(5120, 17408, 64)              # mid bucket
    assert cfg["block_n"] == 128 and cfg["split_k"] == 2, cfg
    return "the measured tiles come back out of pick_config"


def test_registration_can_be_switched_off():
    """`QWEN38_VERIFY_TILES=0` has to restore the chooser, or the staircase cannot be re-measured
    with and without in one session."""
    from tools import nvfp4_verify_tiles as vt
    old = os.environ.get("QWEN38_VERIFY_TILES")
    os.environ["QWEN38_VERIFY_TILES"] = "0"
    try:
        vt._DONE = False
        vt.register()
        assert vt._DONE is False, "register() ran with the switch off"
    finally:
        if old is None:
            os.environ.pop("QWEN38_VERIFY_TILES", None)
        else:
            os.environ["QWEN38_VERIFY_TILES"] = old
        vt._DONE = False
        vt.register()
    return "the switch is honoured and registration is restored afterwards"


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"  {name:52s} ok   {fn() or ''}")
            passed += 1
    print(f"{passed} passed")
