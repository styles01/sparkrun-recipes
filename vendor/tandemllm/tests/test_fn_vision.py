"""Flash-Next's vision tower and the image rows of its prompts (VIS-97), on the host.

  1  `Proj`: F.linear over an EXL3 projection (one block, or q, k, v side by side) is the linear
     layer over the dequantised weights                                     test_proj_is_a_linear
  2  the tower over EXL3 projections equals engine/vision.py's tower on the same weights dequantised
     (q, k and v apart vs a fused qkv), within the bf16 rounding of the two arithmetics
                                                                            test_exl3_tower_equals_the_dense_one
  3  `_rot` of a text state: the 1-D table at slot + delta                  test_rot_text
  4  `_rot` with images: a range holding image rows gets transformers' interleaved three-axis rows
     (height at frequencies 1, 4, .., 31, width at 2, 5, .., 29, temporal elsewhere), its text rows
     the table's rows bit for bit; a text range after an image the table at index + its delta; rows
     past the prompt slot + `st.delta`                                      test_rot_images
  5  `_pre_ctl` of a chunk: every row's rotary row and every completed block's pooling row are the
     rows of its slot (a chunk that starts inside a block included)        test_pre_ctl_images
  6  `_image_rows`: the tower's rows over the placeholders, split across chunks
                                                                            test_image_rows
  7  `_stage_qsa`: the own blocks' rotary rows move by delta, their slots do not
                                                                            test_stage_qsa_delta
  8  FlashNextServing `mm` / `pos_delta`: on the state and every draft head's state
                                                                            test_serving_fans_out
  9  a pack whose config keeps the vision config but which carries no tower (4.05) has none
                                                                            test_a_pack_without_a_tower

Run: python tests/test_fn_vision.py
"""

from __future__ import annotations

import os
import sys
import types

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.models.flashnext.engine import FlashNextEngine  # noqa: E402
from engine.models.flashnext.ref import FNState  # noqa: E402
from engine.models.flashnext.vision import Proj, tower_of  # noqa: E402
from engine.vision import Image, MMContext, VisionConfig, VisionTower, expand  # noqa: E402
from tools.exl3_linear import EXL3Block  # noqa: E402

G = torch.Generator().manual_seed(11)
IMG = 248056                       # the image placeholder


def _block(n: int, k: int) -> EXL3Block:
    t = torch.randint(-32768, 32767, (k // 16, n // 16, 16 * 3), generator=G, dtype=torch.int32).to(torch.int16)
    suh = torch.where(torch.rand(k, generator=G) < 0.5, -1.0, 1.0).half() * 0.05
    svh = torch.where(torch.rand(n, generator=G) < 0.5, -1.0, 1.0).half() * 0.05
    return EXL3Block(t, suh, svh, out_fp32=True)


def test_proj_is_a_linear():
    a, b = _block(128, 256), _block(256, 256)
    x = torch.randn(5, 256, generator=G).bfloat16()
    bias = torch.randn(384, generator=G).bfloat16()
    got = F.linear(x, Proj([a, b]), bias)
    want = (x.float() @ torch.cat([a.dequant(), b.dequant()]).float().T + bias.float()).bfloat16()
    rel = ((got.float() - want.float()).norm() / want.float().norm()).item()
    assert got.shape == (5, 384) and got.dtype == torch.bfloat16 and rel < 1e-2, rel
    try:
        torch.matmul(x, Proj([a]))
    except TypeError:
        pass
    else:
        raise AssertionError("an EXL3 projection took matmul")
    return f"F.linear over two blocks vs the dequantised linear: {rel:.1e}; matmul refused"


def _cfg() -> VisionConfig:
    return VisionConfig(depth=2, hidden_size=128, num_heads=2, intermediate_size=256, patch_size=2,
                        temporal_patch_size=2, spatial_merge_size=2, in_channels=3, out_hidden_size=128,
                        num_position_embeddings=16)


def test_exl3_tower_equals_the_dense_one():
    c = _cfg()
    D, I = c.hidden_size, c.intermediate_size
    plain, blocks = {}, {}
    plain["patch_embed.proj.weight"] = torch.randn(D, 3, 2, 2, 2, generator=G) * 0.2
    plain["patch_embed.proj.bias"] = torch.randn(D, generator=G) * 0.1
    plain["pos_embed.weight"] = torch.randn(16, D, generator=G) * 0.1
    plain["merger.norm.weight"] = 1 + torch.randn(D, generator=G) * 0.1
    plain["merger.norm.bias"] = torch.randn(D, generator=G) * 0.1
    shapes = {"attn.q_proj": (D, D), "attn.k_proj": (D, D), "attn.v_proj": (D, D), "attn.proj": (D, D),
              "mlp.linear_fc1": (I, D), "mlp.linear_fc2": (D, I)}
    for i in range(c.depth):
        p = f"blocks.{i}"
        for n in ("norm1", "norm2"):
            plain[f"{p}.{n}.weight"] = 1 + torch.randn(D, generator=G) * 0.1
            plain[f"{p}.{n}.bias"] = torch.randn(D, generator=G) * 0.1
        for n, (N, K) in shapes.items():
            blocks[f"{p}.{n}"] = _block(N, K)
            plain[f"{p}.{n}.bias"] = torch.randn(N, generator=G) * 0.05
    blocks["merger.linear_fc1"] = _block(4 * D, 4 * D)
    blocks["merger.linear_fc2"] = _block(c.out_hidden_size, 4 * D)
    plain["merger.linear_fc1.bias"] = torch.randn(4 * D, generator=G) * 0.05
    plain["merger.linear_fc2.bias"] = torch.randn(c.out_hidden_size, generator=G) * 0.05
    fn = tower_of(c, dict(plain), blocks, device="cpu")
    dense = {k: v for k, v in plain.items() if not any(f".{n}." in k for n in ("q_proj", "k_proj", "v_proj"))}
    for name, b in blocks.items():
        if not name.endswith(("q_proj", "k_proj", "v_proj")):
            dense[f"{name}.weight"] = b.dequant()
    for i in range(c.depth):
        p = f"blocks.{i}.attn"
        dense[f"{p}.qkv.weight"] = torch.cat([blocks[f"{p}.{n}"].dequant() for n in ("q_proj", "k_proj", "v_proj")])
        dense[f"{p}.qkv.bias"] = torch.cat([plain[f"{p}.{n}.bias"] for n in ("q_proj", "k_proj", "v_proj")])
    ref = VisionTower(c, dense, device="cpu")
    grid = (1, 4, 4)
    px = torch.randn(16, c.patch_dim, generator=G)
    a, b = fn.encode(px, grid).float(), ref.encode(px, grid).float()
    rel = ((a - b).norm() / b.norm()).item()
    assert a.shape == (4, c.out_hidden_size) and rel < 2e-2, rel
    assert ref.nbytes > fn.nbytes > 0 and fn.activation_bytes(16) > ref.activation_bytes(16)
    return f"EXL3 tower vs the dense one on its dequantised weights: relative difference {rel:.1e}"


# ----------------------------------------------------------------------------- the language model
class _Cfg:
    rotary_dim = 64
    mrope_section = [11, 11, 10]
    has_indexer = True
    indexer_compress_ratio = 4
    linear_conv_kernel_dim = 4
    ple_conv_kernel_size = 4
    ngram_size = 3


def _eng(max_len: int = 4096):
    from engine.models.flashnext.ref import FlashNextRef
    inv = 1.0 / (1e7 ** (torch.arange(0, 64, 2, dtype=torch.float32) / 64))
    e = types.SimpleNamespace(cfg=_Cfg(), inv_freq=inv, dtype=torch.bfloat16, device="cpu")
    e.rope_cos, e.rope_sin = FlashNextRef.rope(e, torch.arange(max_len))
    for name in ("_rot", "_image_rows", "_pre_ctl", "qsa_class", "_stage_qsa"):
        setattr(e, name, getattr(FlashNextEngine, name).__get__(e))
    return e


def _want(inv: torch.Tensor, p3: torch.Tensor):
    """transformers' interleaved recomposition, written out from its source: p3 [3, L] -> cos, sin."""
    f = p3.float()[:, :, None] * inv[None, None, :]                    # [3, L, F]
    thw = f[0].clone()
    for d, off in ((1, 1), (2, 2)):
        idx = slice(off, (11, 11, 10)[d] * 3, 3)
        thw[..., idx] = f[d][..., idx]
    emb = torch.cat([thw, thw], -1)
    return emb.cos().bfloat16(), emb.sin().bfloat16()


def _mm(grids, text=(5, 7, 9)):
    """A prompt of text, an image, text, an image, text (the placeholders expanded), its MMContext over
    a tower whose rows say which image and row they are."""
    ids = [1] * text[0] + [IMG] + [2] * text[1] + [IMG] + [3] * text[2]
    imgs = [Image(pixel_values=torch.zeros(t * h * w, 1), grid=(t, h, w), digest=bytes([j]) * 16)
            for j, (t, h, w) in enumerate(grids)]
    full, spans = expand(ids, imgs, IMG, 2)
    tower = types.SimpleNamespace(cfg=types.SimpleNamespace(spatial_merge_size=2), encoded=[])

    def encode(px, grid):
        tower.encoded.append(grid)
        j = [im.grid for im in imgs].index(grid)
        n = px.shape[0] // 4
        return (1000 * (j + 1) + torch.arange(n, dtype=torch.float32))[:, None].repeat(1, 8)
    tower.encode = encode
    return full, MMContext(full, spans, imgs, tower, _Cfg()), tower


def test_rot_text():
    e = _eng()
    st = FNState()
    cos, sin, off = e._rot(st, 10, 50)
    assert cos is e.rope_cos and sin is e.rope_sin and off == 0
    st.delta = -37
    assert e._rot(st, 100, 120)[2] == -37
    return "a text state reads the 1-D table at slot + delta"


def test_rot_images():
    e = _eng()
    full, mm, _ = _mm([(1, 8, 12), (1, 6, 4)])
    st = FNState(mm=mm, delta=mm.delta)
    n = mm.n
    s0, s1 = mm.spans
    cos, sin, off = e._rot(st, 0, n + 6)                   # the whole prompt and six rows past it
    p3 = torch.cat([mm.pos3, (torch.arange(n, n + 6) + mm.delta).view(1, -1).expand(3, -1)], 1)
    wc, ws = _want(e.inv_freq, p3)
    assert off == 0 and torch.equal(cos, wc) and torch.equal(sin, ws)
    text = [r for r in range(n + 6) if not (s0.start <= r < s0.start + s0.n or s1.start <= r < s1.start + s1.n)]
    for r in text:                                         # a text row: the 1-D table's, bit for bit
        q = int(p3[0, r])
        assert torch.equal(cos[r], e.rope_cos[q]) and torch.equal(sin[r], e.rope_sin[q]), r
    assert not torch.equal(cos[s0.start + 13], e.rope_cos[int(p3[0, s0.start + 13])])   # an image row is not
    a = s0.start + s0.n + 1                                # text between the images
    c2, _, off2 = e._rot(st, a, a + 4)
    assert c2 is e.rope_cos and off2 == int(mm.pos3[0, a]) - a < 0
    c3, _, off3 = e._rot(st, n + 2, n + 40)                # past the prompt
    assert c3 is e.rope_cos and off3 == mm.delta
    c4, _, off4 = e._rot(st, s1.start - 2, s1.start + 3)   # a range that starts in text, ends in an image
    assert off4 == -(s1.start - 2) and torch.equal(c4, wc[s1.start - 2:s1.start + 3])
    return (f"{n} prompt rows, 2 images: three-axis rows = transformers'; text rows = the table's; "
            f"delta {mm.delta}")


def test_pre_ctl_images():
    e = _eng()
    full, mm, _ = _mm([(1, 16, 20), (1, 8, 8)], text=(6, 11, 13))
    st = FNState(mm=mm, delta=mm.delta)
    n = mm.n
    p3 = torch.cat([mm.pos3, (torch.arange(n, n + 64) + mm.delta).view(1, -1).expand(3, -1)], 1)
    wc, ws = _want(e.inv_freq, p3)
    checked = 0
    for L0, T in ((0, 37), (37, 50), (87, n - 87 + 1), (n + 1, 9), (3, 1), (n - 3, 3)):
        P = e._pre_ctl(L0, T, st)
        assert P["pos"].tolist() == list(range(L0, L0 + T))
        for m in range(T):
            r = int(P["rpos"][m])
            assert torch.equal(P["cos"][r], wc[L0 + m]) and torch.equal(P["sin"][r], ws[L0 + m]), (L0, m)
            checked += 1
        for j, b in enumerate(range(*P["pool"])):
            assert P["pool_slots"][j].tolist() == [4 * b + k for k in range(4)]
            assert torch.equal(P["cos"][int(P["pool_pos"][j])], wc[4 * b]), (L0, b)
    return f"{checked} rows and their blocks: each reads its slot's rotary row"


def test_image_rows():
    e = _eng()
    full, mm, tower = _mm([(1, 4, 6), (1, 2, 2)])
    st = FNState(mm=mm, delta=mm.delta)
    out = []
    for L0 in range(0, mm.n, 5):                            # chunks of 5 cut through both images
        T = min(5, mm.n - L0)
        out.append(e._image_rows(torch.full((T, 8), -1.0), st, L0))
    rows = torch.cat(out)[:, 0].tolist()
    want = [-1.0] * mm.n
    for s in mm.spans:
        for k in range(s.n):
            want[s.start + k] = 1000.0 * (s.image + 1) + k
    assert rows == want and len(tower.encoded) == 2         # each image encoded once
    return f"{sum(s.n for s in mm.spans)} image rows over their placeholders across chunks of 5"


def test_stage_qsa_delta():
    from tools import fn_seq_kernels as SK
    e = _eng()

    def h():
        return {"actl": torch.zeros(3, 4, dtype=torch.int32), "oslot": torch.zeros(3, SK.OWN * 4, dtype=torch.int32),
                "opos": torch.zeros(3, SK.OWN, dtype=torch.int32), "qctl": torch.zeros(3, 4, dtype=torch.int32),
                "tail": torch.zeros(3, 3, dtype=torch.int32), "pidx": torch.zeros(1, dtype=torch.int64)}
    L = 3001
    pos = L + np.arange(3)
    anc = np.tile(L + np.arange(3), (3, 1))
    a, b = h(), h()
    e._stage_qsa(a, L, L // 4, pos, anc, 3)
    e._stage_qsa(b, L, L // 4, pos, anc, 3, -411)
    for k in a:
        if k == "opos":
            assert torch.equal(b[k], a[k] - 411)
        else:
            assert torch.equal(a[k], b[k]), k
    return "opos moves by delta; slots, counts, tails and the pool row do not"


def test_serving_fans_out():
    from engine.models.flashnext.serving import FlashNextServing
    s = object.__new__(FlashNextServing)
    s.st = FNState()
    s.heads = {"mtp": types.SimpleNamespace(st=FNState())}
    mm = object()
    s.mm, s.pos_delta = mm, -12
    assert s.mm is mm and s.pos_delta == -12
    assert s.heads["mtp"].st.mm is mm and s.heads["mtp"].st.delta == -12
    s.mm, s.pos_delta = None, 0
    assert s.st.mm is None and s.heads["mtp"].st.delta == 0
    return "the state's and the MTP head's"


def test_a_pack_without_a_tower():
    import json
    import tempfile

    from engine.models.flashnext.vision import load_tower
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "config.json"), "w") as f:
            json.dump({"vision_config": {"depth": 2, "hidden_size": 128, "num_heads": 2, "intermediate_size": 256,
                                         "patch_size": 2, "out_hidden_size": 128, "num_position_embeddings": 16}}, f)
        assert load_tower(types.SimpleNamespace(vision={}), d, device="cpu") is None
    return "no model.visual.* tensors: no tower (the server says vision off)"


if __name__ == "__main__":
    fails = 0
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        try:
            print(f"  {name:<45} ok   {fn() or ''}")
        except Exception as exc:                                   # noqa: BLE001
            fails += 1
            import traceback
            traceback.print_exc()
            print(f"  {name:<45} FAIL ({exc})")
    print(f"{len(tests) - fails} passed" + (f", {fails} FAILED" if fails else ""))
    sys.exit(1 if fails else 0)
