"""CPU test for tools/profile_decode.py's KV sizing: every sweep setting must fit in the buffer.

On 2026-09-23 `--sweep-two-stream off,on,off,on,off,on` died at its third setting with the
"verify block overruns the KV window", because the buffer was sized for the fused sweep only.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.profile_decode import STEP_MEDIAN_TOKENS, room_for  # noqa: E402


def _args(**kw):
    base = dict(prompt_len=256, steps=32, warmup=6, sweep_fused=False, sweep_two_stream="")
    base.update(kw)
    return SimpleNamespace(**base)


def test_the_two_stream_sweep_fits():
    a = _args(sweep_two_stream="off,on,off,on,off,on")
    used = a.prompt_len + a.warmup + a.steps + 6 * STEP_MEDIAN_TOKENS
    assert room_for(a) >= used


def test_no_sweep_is_what_it_was():
    assert room_for(_args()) == 256 + 32 + 6 + 64
    assert room_for(_args(sweep_fused=True)) == 256 + 32 + 6 + 64 + 240


if __name__ == "__main__":
    test_the_two_stream_sweep_fits()
    test_no_sweep_is_what_it_was()
    print("2/2 passed")
