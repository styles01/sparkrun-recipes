"""ENG-198/200: the EXL3 weight format on the CPU (the kernels' own gate is tools/exl3_check.py).

The pure-torch decode is the yardstick the CUDA kernels are checked against bit for bit on the
board, so it is pinned here: the tile layout is the tensor core's B fragment, a known bit string
decodes to the codebook value written out by hand, the reference product equals x @ W for the
dense weight the rotations define, group members are views that read like their own blocks, and
the loader's EXL3 discovery names exactly the projections and the head (never the MTP layer).
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.linear import Linear  # noqa: E402
from tools import exl3_linear as X  # noqa: E402


def _block(N=256, K=256, kb=3, seed=0):
    return X.random_block(N, K, kb, device="cpu", seed=seed)


def test_tile_layout_is_the_mma_b_fragment():
    perm = X._tile_perm()
    assert sorted(perm.tolist()) == list(range(256))
    for lane in range(32):
        g, t = lane // 4, lane % 4
        # B fragment of m16n8k16: b0 = k (2t, 2t+1) at column g, b1 = k (2t+8, 2t+9)
        want = [(2 * t, g), (2 * t + 1, g), (2 * t + 8, g), (2 * t + 9, g),
                (2 * t, g + 8), (2 * t + 1, g + 8), (2 * t + 8, g + 8), (2 * t + 9, g + 8)]
        got = [divmod(int(perm[lane * 8 + j]), 16) for j in range(8)]
        assert got == want, (lane, got, want)
    return "positions 8l..8l+7 = lane l's two B fragments"


def test_known_bits_decode_to_the_codebook():
    # all-zero bits: every window is 0, 0 * mult = 0, byte sum 0, h = 1024
    t = torch.zeros(1, 1, 16 * 2, dtype=torch.int16)
    w = X.decode_trellis(t)
    k_inv = torch.tensor(0x1eee, dtype=torch.int16).view(torch.float16).float()
    k_bias = torch.tensor(0xc931 - 0x10000, dtype=torch.int16).view(torch.float16).float()
    want = (1024.0 * k_inv + k_bias).to(torch.float16)
    assert torch.equal(w, torch.full((16, 16), float(want), dtype=torch.float16))
    # one window by hand: K = 4, position 0 ends at bit 4 -> the window is the tile's last 12 bits
    # followed by its first 4 (it wraps). Set word 0's top nibble to 0xA: position 0 = 0x000A.
    t = torch.zeros(1, 1, 64, dtype=torch.int16)
    t.view(torch.int32)[0, 0, 0] = 0xA0000000 - (1 << 32)
    w = X.decode_trellis(t)
    x = (0xA * X.MUL1) & 0xFFFFFFFF
    s = sum((x >> (8 * i)) & 0xFF for i in range(4))
    want = ((1024 + s) * k_inv + k_bias).to(torch.float16)
    assert w[0, 0] == want, (float(w[0, 0]), float(want))
    return f"zero bits -> {float(w[1, 1]):.4f}, window 0xA -> {float(want):.4f}"


def test_reference_product_is_the_dense_weight():
    w = _block(256, 384, 4, seed=1)
    x = (torch.randn(5, 384) * 2).to(torch.bfloat16)
    ref = X.reference_matmul(x, w).float()
    dense = x.float() @ w.dequant().float().T
    rel = ((ref - dense).abs().max() / dense.abs().max()).item()
    assert rel < 2e-2, rel                # the fp16 input rounding and the bf16 weight copy
    assert isinstance(w, Linear) and w.shape == (256, 384)
    return f"reference vs x @ W: rel {rel:.2e}"


def test_group_members_are_views_and_read_alike():
    a, b = _block(256, 256, 2, seed=2), _block(128, 256, 2, seed=3)
    da, db = a.dequant(), b.dequant()
    grp = X.EXL3Block.group([a, b], ["a", "b"])
    assert grp.sizes == [256, 128] and grp.N == 384 and grp.bounds == (2, 1 << 30)
    ma, mb = grp.member(0), grp.member(1)
    assert ma.t.data_ptr() == grp.t.data_ptr() and not mb.t.is_contiguous()
    assert torch.equal(ma.dequant(), da) and torch.equal(mb.dequant(), db)
    x = torch.randn(3, 256).to(torch.bfloat16)
    y = X.reference_matmul(x, grp)
    assert torch.equal(y[:, :256], X.reference_matmul(x, a))
    assert torch.equal(y[:, 256:], X.reference_matmul(x, b))
    return "group = members side by side; members are views"


def test_every_rate_and_no_other():
    for kb in X.KS:
        assert _block(128, 128, kb).kb == kb
    for bad in (1, 9):
        try:
            X.EXL3Block(torch.zeros(8, 8, 16 * bad, dtype=torch.int16), torch.ones(128), torch.ones(128))
        except AssertionError:
            continue
        raise AssertionError(f"K={bad} accepted")
    return f"K {X.KS} accepted, 1 and 9 refused"


def test_loader_discovers_projections_and_head_not_mtp():
    from safetensors.torch import save_file
    from engine.loader import exl3_bases
    with tempfile.TemporaryDirectory() as d:
        t = {}
        for name in ("model.language_model.layers.0.mlp.gate_proj", "lm_head",
                     "mtp.layers.0.mlp.gate_proj"):
            t[f"{name}.trellis"] = torch.zeros(8, 8, 32, dtype=torch.int16)
        t["model.language_model.layers.0.input_layernorm.weight"] = torch.zeros(4)
        save_file(t, os.path.join(d, "model.safetensors"))
        json.dump({"weight_map": {k: "model.safetensors" for k in t}},
                  open(os.path.join(d, "model.safetensors.index.json"), "w"))
        got = exl3_bases(d)
    assert got == {"layers.0.mlp.gate_proj", "lm_head"}, got
    return "projections + head, never mtp.*"


def test_flag_off_reads_nothing():
    from engine.settings import KNOBS
    assert KNOBS["EXL3"][0] is None and KNOBS["EXL3_FUSE"][0] == "1"
    return "QWEN38_EXL3 unset by default"


if __name__ == "__main__":
    fails = 0
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        try:
            print(f"  {name:<44} ok   {fn() or ''}")
        except AssertionError as e:
            fails += 1
            print(f"  {name:<44} FAIL {e}")
    print(f"{len(tests) - fails} passed" + (f", {fails} FAILED" if fails else ""))
    sys.exit(1 if fails else 0)
