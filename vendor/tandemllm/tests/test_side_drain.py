"""rc-0.8.0: a chat on a side slot survives a planned restart.

THE CASE (live, 2026-10-12): with three slots, a chat whose last turn ran on a side slot came back
cold after a restart: the drain wrote slot 0's conversation and the parked one, nothing of the side
slots, and the next turn prefilled its 24,679 tokens again.

  1  the drain writes each side slot's conversation too (reason `drain-side`), and the start reads
     back slot 0's (`drained()` takes `drain` only)
  2  after the restart that chat's next turn waits for slot 0 (`_par_side_need`: `wait=spill`),
     whose prefill reads it from the NVMe, and writes the same tokens as with no restart

On the random 4-layer model with the CPU pool of tests/test_side_home.py.

Run: python tests/test_side_drain.py
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import test_forward_tree as T  # noqa: E402,F401  (the CPU environment before the engine imports)

from server import app  # noqa: E402
from test_end_anchor import TOOL, turn  # noqa: E402
from test_par_continue import _need  # noqa: E402
from test_session_spill import _spill  # noqa: E402
from test_side_carry import R_PROMPT, S_PROMPT, side_turn  # noqa: E402
from test_side_home import _pool_home, _reference  # noqa: E402


def test_a_side_slots_chat_is_drained_and_comes_back_through_slot_0():
    _, ref2 = _reference()
    d = tempfile.mkdtemp(prefix="qse-side-drain-")
    old = app.SIDE_CARRY
    try:
        app.SIDE_CARRY = "off"                      # the spill, not the carry, brings it back
        res, pool, side, side_res = _pool_home()
        sp, t = _spill(d, res)
        turn(R_PROMPT, 40, session="R")
        out1, _, _ = side_turn(pool, S_PROMPT, 60, "S")
        e_side = side_res.end["at"]
        app.spill_drain()
        reasons = sorted((x.man["owner"], x.man["reason"]) for x in sp.entries.values())
        assert reasons == [("R", "drain"), ("S", "drain-side")], reasons
        t.close()
        # a restart: new slots over the same weights, the spill reopened, slot 0's conversation read back
        res2, pool2, side2, side_res2 = _pool_home()
        sp2, t2 = _spill(d, res2)
        app.spill_startup()
        assert res2.owner == "R" and side_res2.valid == 0
        nxt = S_PROMPT + out1 + TOOL
        need, why = _need(nxt, max_new=24)
        assert (need is None or why.get("slot0_first")) and why.get("wait") == "spill", why
        assert why.get("spill") == e_side, why
        out2, pre2, _ = turn(nxt, 24, session="S")
        assert pre2["kind"] == "spill" and pre2["reused"] == e_side, pre2
        assert out2 == ref2, "the same tokens as the chat with no restart"
        t2.close()
        return f"slot 1's chat ({e_side} tokens) drained, back through slot 0 from the spill"
    finally:
        app.SIDE_CARRY = old
        shutil.rmtree(d, ignore_errors=True)


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
