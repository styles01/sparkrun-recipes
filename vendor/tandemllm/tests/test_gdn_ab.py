"""QWEN38_GDN_AB: the gate projections through one fixed-order kernel, and the flag-off path as before.

The kernel is Triton; on a CPU the test replaces it with the arithmetic it stands for and checks the
plumbing on the random four-layer model: with the flag the two gate projections are one call over
the concatenated weight, split back into `a` and `b`; without it the engine is bit-identical to what
it was. The kernel's own numbers are `tools/small_linear.py check()` on the board.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import test_forward_tree as T  # noqa: E402

import torch  # noqa: E402

import engine.model as M  # noqa: E402
from tools import small_linear as SL  # noqa: E402


def test_the_gate_projections_split_the_concatenated_call():
    eng, _ = T.build(seed=8)
    calls = []

    def stand_in(x, w):
        calls.append(tuple(w.shape))
        return torch.nn.functional.linear(x, w)

    old, SL.small_linear = SL.small_linear, stand_in
    M.GDN_AB = True
    try:
        flat = torch.randn(5, eng.cfg.hidden_size)
        a, b = eng._gate_inputs(flat, "layers.0")
    finally:
        SL.small_linear, M.GDN_AB = old, False
    wa = eng.w.norm("layers.0.linear_attn.in_proj_a.weight")
    wb = eng.w.norm("layers.0.linear_attn.in_proj_b.weight")
    assert calls == [(wa.shape[0] + wb.shape[0], wa.shape[1])], calls
    assert torch.allclose(a, flat @ wa.T) and torch.allclose(b, flat @ wb.T)
    a2, b2 = eng._gate_inputs(flat, "layers.0")
    assert torch.equal(a2, torch.nn.functional.linear(flat, wa)), "flag off changed"
    assert torch.equal(b2, torch.nn.functional.linear(flat, wb)), "flag off changed"
    return "one call over [a; b], split back; flag off is the two F.linear calls, bit for bit"


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"  {name:56s} ok   {fn() or ''}")
            passed += 1
    print(f"{passed} passed")
