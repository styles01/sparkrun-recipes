"""The residual add fused into the norm that reads it must not move a number.

`Qwen38Engine._forward_addnorm` restructures the layer loop: layer l's input norm is computed by
layer l - 1's second add, the final norm by the last one. The kernel is Triton, so here both fused
functions are replaced by the torch arithmetic they stand for; what is checked is the loop -- which
weight goes with which add, what the drafter's tap sees, the two hidden states the engine keeps --
against the unfused forward on the random four-layer model, bit for bit.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import test_forward_tree as T  # noqa: E402

import torch  # noqa: E402

import engine.model as M  # noqa: E402
from tools import norm_kernels as NK  # noqa: E402


def _torch_norm(x, weight, eps):
    out = x.float()
    out = out * torch.rsqrt(out.pow(2).mean(-1, keepdim=True) + eps)
    return (out * (1.0 + weight.float())).type_as(x)


def _torch_gated(x, gate, weight, eps):
    dt = x.dtype
    h = x.float()
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)
    h = weight * h.to(dt)
    h = h * torch.nn.functional.silu(gate.float())
    return h.to(dt)


def _torch_add_norm(res, x, weight, eps):
    h = res + x
    return h, _torch_norm(h, weight, eps)


def _run(fused: bool):
    taps = []
    old = (NK.rms_norm, NK.add_rms_norm, NK.rms_norm_gated, M.FUSED["norm"], M.FUSED_ADDNORM)
    NK.rms_norm, NK.add_rms_norm, NK.rms_norm_gated = _torch_norm, _torch_add_norm, _torch_gated
    M.FUSED["norm"], M.FUSED_ADDNORM = True, fused
    try:
        eng, pos = T.build(seed=6)
        eng.tap = lambda h: taps.append(h.clone())
        with torch.no_grad():
            lg = eng.forward_block(torch.tensor([4, 8, 15, 16, 23]), start=pos)
        return lg, taps, eng.hidden_pre_norm.clone(), eng.hidden_post_norm.clone(), \
            eng.state.S.clone()
    finally:
        (NK.rms_norm, NK.add_rms_norm, NK.rms_norm_gated, M.FUSED["norm"],
         M.FUSED_ADDNORM) = old


def test_the_fused_loop_is_the_same_forward():
    a, b = _run(False), _run(True)
    assert torch.equal(a[0], b[0]), "logits"
    assert len(a[1]) == len(b[1]) == 5 and all(torch.equal(x, y) for x, y in zip(a[1], b[1])), \
        "the tap saw different residual streams"
    assert torch.equal(a[2], b[2]) and torch.equal(a[3], b[3]), "hidden states"
    assert torch.equal(a[4], b[4]), "recurrent state"
    return "logits, five taps, both hidden states and the state bit-identical"


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"  {name:56s} ok   {fn() or ''}")
            passed += 1
    print(f"{passed} passed")
