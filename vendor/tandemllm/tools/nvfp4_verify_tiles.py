"""Tile shapes for the W4A16 kernel at VERIFY row counts, measured rather than guessed.

The verify staircase -- 16 nodes 143.9 ms, 24 nodes 215.3 -- looked like a property of the hardware:
the FP4 kernel does sixteen rows of tensor-core work whatever it is asked for, so a seventeenth row
costs a second tile. `tools/nvfp4_wide_probe.py` measured it at 17:19 and that reading was wrong.
Most of the step is the TILE THE CHOOSER HANDS OUT, and the chooser was tuned for one row:

    gate/up  N=17408 K=5120        down  N=5120 K=17408
      M   default   best    tile          M   default   best    tile
     16     0.383   0.338   m16 n32 w8   16     0.371   0.371   m16 n16 k8 w1
     24     0.545   0.371   m32 n32 w4   24     0.530   0.436   m32 n128 k2 w4
     32     0.647   0.435   m32 n64 w4   32     0.593   0.450   m32 n128 k2 w4
     64     0.534   0.506   m64 n32 w4   64     0.594   0.569   m64 n128 k2 w4
    128     1.210   0.784  m128 n64 w4  128     1.123   0.883   m128 n64 k1 w4

At 24 rows the right tile is **32 % faster** than the default, and the gap between 16 rows and 24
falls from 0.162 ms a projection to 0.033. Across 64 layers and three projections that is the
difference between a 71 ms staircase and an 8 ms one.

Registered through `set_config`, which is the interface `tools/sweep_nvfp4_shapes.py` writes and
`pick_config` reads, so nothing in `tools/nvfp4_linear.py` is edited -- that file is track A's and
its M = 1 path is untouched. `pick_config` forces `block_m` itself in the decode bucket (16 to 16
rows, 32 above), which happens to be what was measured, so only the N tile, the K split and the warp
count are set here.
"""

from __future__ import annotations

import os
if os.path.dirname(os.path.dirname(os.path.abspath(__file__))) not in __import__("sys").path:
    # run as a script from tools/: the repo root, appended (lowest priority), for engine.settings
    __import__("sys").path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.settings import SETTINGS as _S  # noqa: E402  (every QWEN38_* knob)
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.nvfp4_linear import _FALLBACK, set_config  # noqa: E402

# (N, K) -> bucket -> tile. "decode" covers M <= 32, "mid" covers 33..128 (engine/../pick_config).
# `set_config` REPLACES a bucket rather than updating it, so a config registered here has to be
# COMPLETE: every key `nvfp4_matmul` reads must be present or the launch dies with a KeyError, and
# it dies in the server's warm-up, which takes the board down for everybody. Registering a partial
# dict is what happened at 17:29 and it blocked every atlas row on the box until 17:40.
#
# So nothing here is registered raw. `register()` overlays these onto `_FALLBACK[bucket]`, which is
# the complete key set by construction, and `tests/test_verify_tiles.py` asserts that what comes
# back out of `pick_config` has every key the launch reads, at every M the engine can ask for. A
# measured tile is a few numbers; the rest of the config is not this file's business to restate.
MEASURED = {
    (17408, 5120): {                       # gate_proj and up_proj
        "decode": {"block_n": 32, "split_k": 1, "num_warps": 4},
        "mid": {"block_m": 64, "block_n": 32, "split_k": 1, "num_warps": 4},
    },
    (5120, 17408): {                       # down_proj: tall and thin, wants a wide N and a K split
        "decode": {"block_n": 128, "split_k": 2, "num_warps": 4},
        "mid": {"block_m": 64, "block_n": 128, "split_k": 2, "num_warps": 4},
    },
}

_DONE = False


def register() -> None:
    """Install the measured tiles. Idempotent, and a no-op when switched off.

    `QWEN38_VERIFY_TILES=0` restores whatever the chooser had, which is how the staircase gets
    re-measured with and without in one session.
    """
    global _DONE
    if _DONE or _S.get("VERIFY_TILES") != "1":
        return
    for (n, k), buckets in MEASURED.items():
        for bucket, cfg in buckets.items():
            full = dict(_FALLBACK[bucket])          # the complete key set
            full.update(cfg)                        # the measured tile over it
            set_config(n, k, bucket, full)
    _DONE = True


register()
