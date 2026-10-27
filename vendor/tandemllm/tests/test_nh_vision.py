"""Super VL's image path (ENG-365): the RADIO tower and projector, the processor, the `<img>` rows of a prompt,
and the NemotronH engine's prefill with image rows.

On the host, with tiny random weights:
  1  `target_patches`: the reference processor's grid for a table of sizes (tiny, wide, tall, huge), recorded
     from the checkpoint's own `_compute_target_patches` on the Spark          test_target_patches_follow_the_reference
     and a very long, thin image (a side the reference floors to 0) at least one shuffle cell a side
                     test_a_very_long_thin_image_gets_a_grid_of_at_least_one_shuffle_cell_a_side
  2  `pixels`: no resize at the grid's size, the CLIP normalisation            test_pixels_normalise_and_keep_the_grid
  3  `patchify`: RADIO's `_image_to_patches` order                             test_patchify_is_radios_order
  4  `positions`: the reference's square-then-crop up to a 247-patch side; past it the separable form, equal to
     the square's crop to fp32 rounding, and a 2 x 6,656 grid needs no square  test_positions_equal_the_square
  5  `pixel_shuffle`: row (i, j) is patches (2i, 2j), (2i, 2j+1), (2i+1, 2j), (2i+1, 2j+1)
                                                                               test_pixel_shuffle_layout
  6  the tower: prefix tokens dropped, the final LayerNorm only with MTP, rows of the trunk's width
                                                                               test_tower_shapes_and_final_norm
  7  `expand`: `<img>`, a row each, `</img>`; a count mismatch refused          test_expand_tags_and_spans
  8  `NhMMContext`: splice across chunk boundaries, encode once, key ids by content
                                                                               test_splice_across_chunks
  9  `load_radio_config`: Lightning has none; SwiGLU, ps v1 and layer scale refused
                                                                               test_config_refusals
 10  `activation_bytes`: bounded for the largest and the widest grid          test_activation_bytes
 11  the Super VL template writes one `<image>` per image part, in user and tool messages
                                                                               test_template_placeholders
 12  the server's `_vision_context` takes the tower's own `context`; `_load_vision` its config reader,
     processor and warm-up                                                     test_server_takes_the_towers_own_path
On a GPU, or on the CPU under TRITON_INTERPRET=1 (the tiny NemotronH engine of test_nemotron_h_engine.py):
 13  a prompt whose image rows are the embedding rows of some tokens prefills to the bits of the prompt with
     those tokens, split inside the image (`NemotronHServing.forward` twice); a one-row forward on an image row
     takes the prefill; a text request leaves no trace                         test_engine_prefill_splices_image_rows
 14  the caches (engine/cache.py `prefill` with a state store) key on the image's content: another image under
     the same placeholders resumes at most to the image, the same image past it, bit for bit
                                                                               test_caches_key_on_image_content
With the checkpoint on the box (QSE_NH_VISION_SNAPSHOT, else the Spark's Super VL EXL3 4.0 folder; skipped
elsewhere), its own remote code as the reference:
 15  pixel values and grid identical to the checkpoint's processor; the prompt ids of a chat with an image
     equal its processor's `input_ids`                                         test_real_processor_and_prompt
 16  the tower and projector against the checkpoint's `RadioModel` and vision projector, built small with
     random weights, in fp32                                                   test_real_modeling_code_small

Run: python tests/test_nh_vision.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import types

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.models.nemotron_h import vision as NV  # noqa: E402
from engine.vision import EmbedCache, Image, digest_of  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(HERE, "fixtures", "nemotron_h", "super-vl-chat_template.jinja")
SNAP = os.environ.get("QSE_NH_VISION_SNAPSHOT") or os.path.expanduser(
    "~/models/nemotron-3.5-super-vl-120b-a12b-exl3-4.0")
G = torch.Generator().manual_seed(7)


def cfg(**kw) -> NV.RadioConfig:
    base = dict(depth=2, hidden_size=32, num_heads=4, intermediate_size=128, patch_size=4, max_img_size=32,
                prefix_tokens=3, layer_norm_eps=1e-6, out_hidden_size=24, projector_hidden_size=48, downsample=2,
                final_norm=True, image_token_id=18, image_start_id=19, image_end_id=20, min_patches=16,
                max_patches=64, max_model_len=64, norm_mean=(0.48145466, 0.4578275, 0.40821073),
                norm_std=(0.26862954, 0.26130258, 0.27577711))
    base.update(kw)
    return NV.RadioConfig(**base)


def weights(c: NV.RadioConfig, scale: float = 0.2) -> dict:
    def r(*shape):
        return torch.randn(*shape, generator=G) * scale
    D, I, P = c.hidden_size, c.intermediate_size, c.projector_hidden_size
    t = {"model.patch_generator.cls_token.token": r(c.prefix_tokens, D),
         "model.patch_generator.embedder.weight": r(D, c.patch_dim),
         "model.patch_generator.pos_embed": r(1, c.side * c.side, D),
         "mlp1.0.weight": 1 + r(D * c.downsample ** 2), "mlp1.1.weight": r(P, D * c.downsample ** 2),
         "mlp1.3.weight": r(c.out_hidden_size, P)}
    if c.final_norm:
        t["vision_final_layernorm.weight"], t["vision_final_layernorm.bias"] = 1 + r(D), r(D)
    for i in range(c.depth):
        p = f"model.blocks.{i}."
        t.update({p + "norm1.weight": 1 + r(D), p + "norm1.bias": r(D), p + "attn.qkv.weight": r(3 * D, D),
                  p + "attn.qkv.bias": r(3 * D), p + "attn.proj.weight": r(D, D), p + "attn.proj.bias": r(D),
                  p + "norm2.weight": 1 + r(D), p + "norm2.bias": r(D), p + "mlp.fc1.weight": r(I, D),
                  p + "mlp.fc1.bias": r(I), p + "mlp.fc2.weight": r(D, I), p + "mlp.fc2.bias": r(D)})
    return t


def tower(c: NV.RadioConfig | None = None, dtype=torch.float32) -> NV.RadioTower:
    c = c or cfg()
    return NV.RadioTower(c, weights(c), device="cpu", dtype=dtype)


def image(c: NV.RadioConfig, rows: int, cols: int, seed: int = 0) -> Image:
    pv = torch.randn(rows * cols, c.patch_dim, generator=torch.Generator().manual_seed(seed))
    return Image(pixel_values=pv, grid=(1, rows, cols), digest=digest_of(pv, (1, rows, cols)), source="data")


# ------------------------------------------------------------------ 1-3. the processor
#: (w, h) -> the reference's (w_patches, h_patches), Super VL's processor (patch 16, 1,024..13,312 patches),
#: recorded 2026-10-17 on the Spark from image_processing_nemotron_h_omni.py `_compute_target_patches`
REFERENCE_GRIDS = {(1, 1): (32, 32), (16, 16): (32, 32), (100, 37): (50, 22), (640, 480): (40, 30),
                   (1280, 800): (80, 50), (800, 1280): (50, 80), (1920, 1080): (120, 68), (3840, 2160): (154, 86),
                   (4000, 3000): (132, 100), (8000, 40): (500, 4), (40, 8000): (4, 500), (512, 512): (32, 32),
                   (1024, 1024): (64, 64), (1000, 333): (64, 22), (437, 300): (40, 28), (2560, 1600): (146, 90),
                   (24, 900): (6, 172), (17, 17): (32, 32)}


def super_vl() -> NV.RadioConfig:
    return cfg(depth=32, hidden_size=1280, num_heads=16, intermediate_size=5120, patch_size=16, max_img_size=2048,
               prefix_tokens=10, out_hidden_size=4096, projector_hidden_size=20480, min_patches=1024,
               max_patches=13312, max_model_len=16384)


def test_target_patches_follow_the_reference():
    c = super_vl()
    b = NV.image_budget(c)
    assert b == 13312
    bad = {wh: (NV.target_patches(*wh, c, b), want) for wh, want in REFERENCE_GRIDS.items()
           if NV.target_patches(*wh, c, b) != want}
    assert not bad, bad
    # every grid even on both sides (the shuffle), inside the budget
    for wh in REFERENCE_GRIDS:
        w, h = NV.target_patches(*wh, c, b)
        assert w % 2 == 0 and h % 2 == 0 and w * h <= b, (wh, w, h)
    assert NV.image_budget(c, max_pixels=1024 * 1024) == 4096
    return f"{len(REFERENCE_GRIDS)} sizes, the reference's grids; 1280x800 -> 80x50 (1,000 rows)"


def test_a_very_long_thin_image_gets_a_grid_of_at_least_one_shuffle_cell_a_side():
    c = super_vl()
    # 16 x 40,000: the reference floors the width to 0 patches -- a division by zero above the minimum
    # budget, a 0-pixel resize at it
    for wh, b in (((16, 40000), 1100), ((16, 40000), 1024), ((40000, 16), 1100), ((1, 300000), 13312)):
        w, h = NV.target_patches(*wh, c, b)
        assert w >= c.downsample and h >= c.downsample and w % 2 == 0 and h % 2 == 0, (wh, b, w, h)
    from PIL import Image as PILImage
    small = cfg(patch_size=4, min_patches=16, max_patches=64)
    px = NV.pixels(PILImage.new("RGB", (1, 400)), small, NV.image_budget(small))
    wp, hp = NV.target_patches(1, 400, small, NV.image_budget(small))
    assert px.shape == (3, hp * 4, wp * 4) and wp == 2, (px.shape, wp, hp)
    return f"16x40000 at 1,100 patches -> {NV.target_patches(16, 40000, c, 1100)}; 1x400 -> {wp}x{hp}"


def test_pixels_normalise_and_keep_the_grid():
    from PIL import Image as PILImage
    c = cfg(patch_size=4, min_patches=16, max_patches=64)
    b = NV.image_budget(c)
    pil = PILImage.new("RGB", (16, 16), (255, 0, 128))
    px = NV.pixels(pil, c, b)
    assert px.shape == (3, 16, 16) and px.dtype == torch.float32
    want = [(255 / 255.0 - c.norm_mean[0]) / c.norm_std[0], (0 - c.norm_mean[1]) / c.norm_std[1],
            (128 / 255.0 - c.norm_mean[2]) / c.norm_std[2]]
    for ch in range(3):
        assert torch.allclose(px[ch], torch.full((16, 16), want[ch]), atol=1e-6), (ch, px[ch, 0, 0], want[ch])
    big = NV.pixels(PILImage.new("RGB", (200, 37)), c, b)          # resized to the grid
    wp, hp = NV.target_patches(200, 37, c, b)
    assert big.shape == (3, hp * 4, wp * 4)
    pre = NV.Preprocessor(c, dtype=torch.bfloat16)
    im = pre(pil.convert("L"), "data")                              # any mode: converted to RGB
    assert im.pixel_values.dtype == torch.bfloat16 and im.grid == (1, 4, 4)
    assert im.pixel_values.shape == (16, c.patch_dim) and im.tokens(pre.merge) == 4
    return f"constant image -> the CLIP-normalised values; 200x37 -> {wp}x{hp} patches"


def test_patchify_is_radios_order():
    px = torch.randn(3, 8, 12, generator=G)
    got = NV.patchify(px, 4)
    # transformers' RadioPatchEmbeddings._image_to_patches, batch 1
    b = px.unsqueeze(0).reshape(1, 3, 2, 4, 3, 4).permute(0, 2, 4, 1, 3, 5).reshape(1, 6, 48)[0]
    assert torch.equal(got, b)
    assert torch.equal(got[4], px[:, 4:8, 4:8].reshape(-1))         # patch (row 1, col 1)
    return "rows outer, channels then pixels inside a patch"


# ------------------------------------------------------------------ 4-6. the tower
def test_positions_equal_the_square():
    t = tower(cfg(max_img_size=64, patch_size=4))                     # a 16 x 16 table
    worst = 0.0
    for rows, cols in ((4, 4), (8, 30), (30, 8), (2, 40), (40, 2), (16, 16), (6, 10), (24, 24)):
        a, b = t.separable_positions(rows, cols), t.square_positions(rows, cols)
        assert a.shape == (rows * cols, 32)
        err = ((a - b).abs().max() / b.abs().max()).item()
        worst = max(worst, err)
        assert err < 1e-6, (rows, cols, err)
        assert torch.equal(t.positions(rows, cols), b)                 # a small square: the reference's own
    c = super_vl()
    big = NV.RadioTower.__new__(NV.RadioTower)
    big.cfg = c
    calls = []
    big.square_positions = lambda r, k: calls.append(("square", r, k))
    big.separable_positions = lambda r, k: calls.append(("separable", r, k))
    big.positions(64, 188)                                              # 3000 x 1000
    big.positions(2, 6656)
    assert calls == [("square", 64, 188), ("separable", 2, 6656)], calls
    assert big.activation_bytes(2 * 6656) < 2 << 30
    return (f"separable = the square's crop to {worst:.1e} on 8 grids; up to a 247-patch side the square "
            f"itself; a 2 x 6,656 grid budgets < 2 GiB")


def test_pixel_shuffle_layout():
    rows, cols, C = 4, 6, 3
    f = torch.arange(rows * cols * C, dtype=torch.float32).view(rows * cols, C)
    out = NV.pixel_shuffle(f, rows, cols, 2)
    assert out.shape == (rows * cols // 4, 4 * C)
    x = f.view(rows, cols, C)
    for i in range(rows // 2):
        for j in range(cols // 2):
            want = torch.cat([x[2 * i, 2 * j], x[2 * i, 2 * j + 1], x[2 * i + 1, 2 * j], x[2 * i + 1, 2 * j + 1]])
            assert torch.equal(out[i * (cols // 2) + j], want), (i, j)
    return "row (i, j) = patches (2i,2j) (2i,2j+1) (2i+1,2j) (2i+1,2j+1)"


def test_tower_shapes_and_final_norm():
    c = cfg()
    t = tower(c)
    im = image(c, 4, 6)
    out = t.encode(im.pixel_values, im.grid)
    assert out.shape == (6, c.out_hidden_size) and torch.isfinite(out).all()
    f = t.features(im.pixel_values, im.grid)
    assert f.shape == (24, c.hidden_size)                            # the 3 prefix tokens dropped
    t2 = NV.RadioTower(cfg(final_norm=False), {k: v for k, v in t.t.items() if "final_layernorm" not in k},
                       device="cpu", dtype=torch.float32)
    assert torch.equal(t2.features(im.pixel_values, im.grid), f)
    assert not torch.allclose(t2.project(f, 4, 6), out)              # the LayerNorm is applied with MTP only
    try:
        NV.RadioTower(c, {k: v for k, v in t.t.items() if k != "mlp1.3.weight"}, device="cpu")
    except RuntimeError as e:
        assert "missing" in str(e)
    else:
        raise AssertionError("a tower without mlp1.3 loaded")
    try:
        NV.RadioTower(c, {**t.t, "model.norm.weight": torch.ones(32)}, device="cpu")
    except RuntimeError as e:
        assert "does not read" in str(e)
    else:
        raise AssertionError("an unknown tower tensor was accepted")
    assert t.stats["images"] == 1 and t.stats["patches"] == 24
    assert NV.RadioTower.name_of("vision_model.radio_model.model.blocks.3.attn.qkv.weight") == "model.blocks.3.attn.qkv.weight"
    assert NV.RadioTower.name_of("mlp1.1.weight") == "mlp1.1.weight"
    assert NV.RadioTower.name_of("vision_projector.vision_final_layernorm.bias") == "vision_final_layernorm.bias"
    assert NV.RadioTower.name_of("vision_model.radio_model.model.patch_generator.video_embedder.weight") is None
    assert NV.RadioTower.name_of("language_model.backbone.layers.0.norm.weight") is None
    return "4x6 patches -> 6 rows of 24; final norm only with MTP; missing/extra tensors refused"


# ------------------------------------------------------------------ 7-8. the prompt
def test_expand_tags_and_spans():
    c = cfg()
    a, b = image(c, 4, 4, 1), image(c, 4, 6, 2)
    ids = [1, 2, 18, 3, 18, 4]
    full, spans = NV.expand(ids, [a, b], c)
    assert full == [1, 2, 19] + [18] * 4 + [20, 3, 19] + [18] * 6 + [20, 4], full
    assert [(s.start, s.n, s.image) for s in spans] == [(3, 4, 0), (10, 6, 1)]
    try:
        NV.expand([1, 18, 2], [a, b], c)
    except ValueError as e:
        assert "1 image placeholders for 2 images" in str(e)
    else:
        raise AssertionError("a mismatch expanded")
    return "two images: <img> rows </img> each, spans at the rows"


class FakeTower:
    """A tower whose rows are given: `rows[digest]`."""

    def __init__(self, c, rows: dict):
        self.cfg, self.rows, self.calls = c, rows, 0

    def encode(self, pv, grid):
        self.calls += 1
        return self.rows[digest_of(pv, grid)]


def test_splice_across_chunks():
    c = cfg()
    a = image(c, 4, 4, 3)
    rows = torch.randn(4, 8, generator=G)
    ft = FakeTower(c, {a.digest: rows})
    ids = [5, 18, 6, 7]
    full, spans = NV.expand(ids, [a], c)
    mm = NV.NhMMContext(full, spans, [a], ft, cache=EmbedCache(1 << 20))
    assert mm.delta == 0 and mm.n == len(full) == 9 and mm.image_tokens == 4
    emb = torch.zeros(len(full), 8)
    out = torch.cat([mm.splice(emb[0:3].clone(), 0), mm.splice(emb[3:5].clone(), 3), mm.splice(emb[5:].clone(), 5)])
    assert torch.equal(out[2:6], rows) and not out[:2].any() and not out[6:].any()
    assert ft.calls == 1 and mm.encoded == 1
    assert [mm.at(i) for i in range(9)] == [False, False, True, True, True, True, False, False, False]
    assert mm.key_ids[:2] == full[:2] and mm.key_ids[6:] == full[6:]
    assert len(set(mm.key_ids[2:6])) == 1 and mm.key_ids[2] >= 1 << 40
    b = image(c, 4, 4, 4)
    mb = NV.NhMMContext(*NV.expand(ids, [b], c), [b], ft)
    assert mb.key_ids != mm.key_ids and NV.expand(ids, [b], c)[0] == full
    # a new request with the same image: the embedding cache, not the tower
    m2 = NV.NhMMContext(full, spans, [a], ft, cache=mm.cache)
    m2.splice(torch.zeros(9, 8), 0)
    assert ft.calls == 1 and m2.encoded == 0
    try:
        mm.splice(torch.zeros(3, 8), 7)
    except ValueError:
        pass
    else:
        raise AssertionError("a forward past the prompt spliced")
    try:
        mm.rows(None, None, 0, 1)
    except TypeError:
        pass
    else:
        raise AssertionError("NhMMContext gave rotary rows")
    return "3 chunks splice the 4 rows; encoded once; key ids by content"


# ------------------------------------------------------------------ 9-10. config and bytes
def _config_dir(d: str, vision: dict | None, top: dict | None = None) -> str:
    raw = {"llm_config": {"hidden_size": 64, "num_nextn_predict_layers": 1}, "projector_hidden_size": 128,
           "downsample_ratio": 0.5, "img_context_token_id": 18, "norm_mean": [0.5] * 3, "norm_std": [0.5] * 3,
           **(top or {})}
    if vision is not None:
        raw["vision_config"] = {"model_type": "radio", "hidden_size": 32, "num_hidden_layers": 2,
                                "num_attention_heads": 4, "patch_size": 4, **vision}
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump(raw, f)
    with open(os.path.join(d, "tokenizer.json"), "w") as f:
        json.dump({"added_tokens": [{"id": 18, "content": "<image>"}, {"id": 19, "content": "<img>"},
                                    {"id": 20, "content": "</img>"}]}, f)
    return d


def test_config_refusals():
    with tempfile.TemporaryDirectory() as d:
        _config_dir(d, None)
        assert NV.load_radio_config(d) is None                        # Lightning: no tower
        c = NV.load_radio_config(_config_dir(d, {}))
        assert (c.image_token_id, c.image_start_id, c.image_end_id, c.final_norm, c.downsample) == (18, 19, 20, True, 2)
        assert c.prefix_tokens == 10 and c.intermediate_size == 128 and c.out_hidden_size == 64
        for v, top, what in (({"use_swiglu_ffn": True}, None, "use_swiglu_ffn"), ({"hidden_act": "silu"}, None, "hidden_act"),
                             ({}, {"ps_version": "v1"}, "pixel shuffle"), ({"layerscale_value": 0.1}, None, "layer scale")):
            try:
                NV.load_radio_config(_config_dir(d, v, top))
            except NotImplementedError as e:
                assert what in str(e), e
            else:
                raise AssertionError(f"{what} loaded")
        assert NV.load_tower(_config_dir(d, {}), device="cpu") is None  # a config, no tower tensors
        with open(os.path.join(d, "model.safetensors.index.json"), "w") as f:  # an index naming absent shards
            json.dump({"weight_map": {"mlp1.1.weight": "model-00009-of-00009.safetensors"}}, f)
        assert NV.load_tower(d, device="cpu") is None
        assert NV.load_radio_config(os.path.join(d, "model.gguf")) is None  # a GGUF file, not a folder
    return "no vision_config: None; SwiGLU, silu, ps v1, layer scale refused; no tensors: no tower"


def test_activation_bytes():
    t = NV.RadioTower.__new__(NV.RadioTower)
    t.cfg = super_vl()
    per = t.activation_bytes(1) - t.activation_bytes(0)
    assert per > 2 * 5120 * 2                                         # fc1's output and its gelu, bf16
    big = t.activation_bytes(13312)
    assert big < 2 << 30, big
    return f"{per} B a patch; the largest image {big / 2**30:.2f} GiB"


# ------------------------------------------------------------------ 11-12. template and server
def test_template_placeholders():
    try:
        from transformers.utils.chat_template_utils import render_jinja_template
    except Exception:  # noqa: BLE001
        return "skipped: transformers' jinja renderer missing"
    with open(TEMPLATE) as f:
        tpl = f.read()
    url = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
    msgs = [{"role": "user", "content": [{"type": "text", "text": "What is this?"}, url]},
            {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {
                "name": "render_page", "arguments": {"path": "index.html"}}}]},
            {"role": "tool", "content": [{"type": "text", "text": "rendered"}, url]}]
    out, _ = render_jinja_template(conversations=[msgs], chat_template=tpl, add_generation_prompt=True,
                                   enable_thinking=False)
    text = out[0]
    assert text.count("<image>") == 2, text
    assert "What is this?\n<image><|im_end|>" in text
    assert "<tool_response>\nrendered\n<image>\n</tool_response>" in text, text
    return "one <image> per image part: the user's and the tool response's"


def test_server_takes_the_towers_own_path():
    from server import app
    c = cfg()
    t = tower(c)
    keep = {k: app.STATE.get(k) for k in ("vision", "image_cache", "engine")}
    try:
        app.STATE.update(vision=t, image_cache=EmbedCache(1 << 20), engine=types.SimpleNamespace(cfg=None))
        im = image(c, 4, 4, 9)
        ids, mm = app._vision_context([1, 18, 2], [im], app.STATE["engine"])
        assert ids == [1, 19, 18, 18, 18, 18, 20, 2] and isinstance(mm, NV.NhMMContext)
        assert mm.cache is app.STATE["image_cache"]
        with tempfile.TemporaryDirectory() as d:
            a = types.SimpleNamespace(vision="on", image_max_pixels=0, image_cache_mb=1, image_max_mb=1,
                                      image_max_decode_pixels=1 << 20, max_images=4, image_fetch_timeout=1,
                                      image_https="off", image_https_private="off", max_image_rows=4096,
                                      image_act_gb=1)
            _config_dir(d, {})
            app._load_vision(a, d, lambda vcfg: t, NV.load_radio_config)
            assert app.STATE["vision"] is t and isinstance(app.STATE["image_pre"], NV.Preprocessor)
            assert t.stats["images"] == 0                                # the warm-up's image is not counted
            a.vision = "off"
            app.STATE["vision"] = None
            app._load_vision(a, d, lambda vcfg: t, NV.load_radio_config)
            assert app.STATE["vision"] is None
    finally:
        app.STATE.update(keep)
    return "_vision_context: the tower's <img> rows; _load_vision: its config, processor and warm-up"


# ------------------------------------------------------------------ 13. the engine
def test_engine_prefill_splices_image_rows():
    from tests.test_nemotron_h_engine import DEV, SKIP, tiny
    if SKIP:
        return SKIP
    from engine.models.nemotron_h.serving import NemotronHServing
    _, eng, _ = tiny()
    c = cfg(out_hidden_size=128)
    E = eng.W["embeddings.weight"]
    swap = [31, 32, 33, 34, 35, 36]                                   # the tokens whose rows the image carries
    im = image(c, 4, 6, 11)
    ft = FakeTower(c, {im.digest: E[torch.tensor(swap, device=E.device)].clone()})
    ids = [5, 9, 3, 18, 17, 22, 3, 40, 41, 12]
    full, spans = NV.expand(ids, [im], c)
    plain = list(full)
    s = spans[0]
    plain[s.start:s.start + s.n] = swap
    cut = s.start + 2                                                  # a chunk boundary inside the image
    a = NemotronHServing(eng, 256)
    a.forward(torch.tensor(plain[:cut], device=DEV), 0)
    la = a.forward(torch.tensor(plain[cut:], device=DEV), cut)
    b = NemotronHServing(eng, 256)
    b.mm = NV.NhMMContext(full, spans, [im], ft)
    b.forward(torch.tensor(full[:cut], device=DEV), 0)
    lb = b.forward(torch.tensor(full[cut:], device=DEV), cut)
    assert torch.equal(la, lb), (la - lb).abs().max()
    assert all(torch.equal(x, y) for x, y in zip(list(a.st.ssm.values()), list(b.st.ssm.values())))
    # a one-row forward on an image row takes the prefill (the decode graph reads token embeddings)
    c2 = NemotronHServing(eng, 256)
    c2.mm = NV.NhMMContext(full, spans, [im], ft)
    c2.forward(torch.tensor(full[:s.start + 1], device=DEV), 0)
    c2.forward(torch.tensor(full[s.start + 1:s.start + 2], device=DEV), s.start + 1)
    lc = c2.forward(torch.tensor(full[s.start + 2:], device=DEV), s.start + 2)
    assert c2.stats["decode_steps"] == 0
    assert torch.equal(lc.argmax(), la.argmax())
    # rows forwarded past the prompt in one call (a forced reasoning close, a hidden note), with
    # the request's images still attached: plain rows, as on a text request
    tail = [44, 45, 46]
    a.forward(torch.tensor(tail, device=DEV), len(plain))
    lt = b.forward(torch.tensor(tail, device=DEV), len(full))
    assert torch.equal(a.forward(torch.tensor([7], device=DEV), len(plain) + 3),
                       b.forward(torch.tensor([7], device=DEV), len(full) + 3)), lt
    # a text request after it: no image rows left anywhere
    d = NemotronHServing(eng, 256)
    d.forward(torch.tensor(plain[:cut], device=DEV), 0)
    ld = d.forward(torch.tensor(plain[cut:], device=DEV), cut)
    assert torch.equal(ld, la)
    return f"image rows = token rows, split at {cut}: logits and SSM states bit for bit ({DEV})"


def test_caches_key_on_image_content():
    from tests.test_nemotron_h_engine import DEV, SKIP, tiny
    if SKIP:
        return SKIP
    from engine import cache
    from engine.models.nemotron_h.serving import NemotronHServing
    _, eng, _ = tiny()
    c = cfg(out_hidden_size=128)
    E = eng.W["embeddings.weight"]
    x, y = image(c, 4, 6, 21), image(c, 4, 6, 22)
    ft = FakeTower(c, {x.digest: E[31:37].clone(), y.digest: E[41:47].clone()})
    ids = list(range(20, 30)) + [18] + list(range(40, 52))

    def run(img, store):
        srv = NemotronHServing(eng, 256)
        full, spans = NV.expand(ids, [img], c)
        srv.mm = NV.NhMMContext(full, spans, [img], ft)
        with torch.no_grad():
            lg, reused, _ = cache.prefill(srv, None, full, DEV, store=store, chunk=8, checkpoint=True,
                                          key_ids=srv.mm.key_ids)
        return lg, reused, srv.mm

    store = cache.StateStore(1 << 30, chunk=8)
    lx, r0, mx = run(x, store)
    first = mx.spans[0].start
    assert r0 == 0
    ly, r1, my = run(y, store)                                         # same placeholders, other pixels
    assert r1 <= first, (r1, first)
    lyc, _, _ = run(y, None)
    assert torch.equal(ly, lyc), (ly - lyc).abs().max()
    lx2, r2, _ = run(x, store)                                         # the same image: past it
    assert r2 > first + mx.spans[0].n, (r2, first)
    assert torch.equal(lx2, lx) and not torch.equal(lx, ly)
    assert mx.key_ids != my.key_ids
    return f"other image: reused {r1} (image at {first}); same image: reused {r2}, logits bit-identical ({DEV})"


# ------------------------------------------------------------------ 15-16. the real checkpoint's files
def _have_snapshot() -> bool:
    return os.path.isfile(os.path.join(SNAP, "image_processing_nemotron_h_omni.py"))


def test_real_processor_and_prompt():
    """On a box with the checkpoint (its processor, template and tokenizer files only): our pixel values,
    grid and prompt ids are the checkpoint's own processor's."""
    if not _have_snapshot():
        return "SKIP (no Super VL checkpoint on this machine)"
    import base64
    import io

    from PIL import Image as PILImage
    from transformers import AutoProcessor, AutoTokenizer

    from server import app
    from server import images as images_mod
    c = NV.load_radio_config(SNAP)
    proc = AutoProcessor.from_pretrained(SNAP, trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(SNAP)
    out = []
    for seed, (w, h) in enumerate(((437, 300), (1280, 800), (3000, 1000))):
        g = torch.Generator().manual_seed(seed)
        arr = (torch.rand(h, w, 3, generator=g) * 255).to(torch.uint8).numpy()
        buf = io.BytesIO()
        PILImage.fromarray(arr).save(buf, format="PNG")
        url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        body = {"messages": [{"role": "user", "content": [{"type": "text", "text": "Describe this image."},
                                                          {"type": "image_url", "image_url": {"url": url}}]}],
                "chat_template_kwargs": {"enable_thinking": False}}
        pil = PILImage.open(io.BytesIO(buf.getvalue())).convert("RGB")
        ims = images_mod.collect(body, images_mod.Limits(), NV.Preprocessor(c))
        ref = proc.image_processor(images=[pil], return_tensors="pt")
        rpx = ref["pixel_values"][0]
        rows, cols = rpx.shape[1] // 16, rpx.shape[2] // 16
        assert ims[0].grid == (1, rows, cols), (ims[0].grid, rpx.shape)
        assert torch.equal(ims[0].pixel_values, NV.patchify(rpx.float(), 16)), "pixel values differ"
        app.STATE.update(tok=tok, device="cpu")
        ids, _, _ = app.build_prompt(body)
        full, spans = NV.expand(ids.tolist(), ims, c)
        text = tok.apply_chat_template(body["messages"], add_generation_prompt=True, tokenize=False,
                                       enable_thinking=False)
        want = proc(text=[text], images=[pil], return_tensors="pt")["input_ids"][0].tolist()
        assert full == want, "prompt ids differ from the processor's"
        out.append(f"{w}x{h} -> {cols}x{rows} patches, {spans[0].n} rows")
    return "identical pixels and ids: " + "; ".join(out)


def test_real_modeling_code_small():
    """The checkpoint's own `RadioModel` and vision projector (remote code), built small with random weights:
    ours on the same weights in fp32 within float rounding."""
    if not _have_snapshot():
        return "SKIP (no Super VL checkpoint on this machine)"
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    RadioModel = get_class_from_dynamic_module("modeling_radio.RadioModel", SNAP)
    RadioConfig = get_class_from_dynamic_module("configuration_radio.RadioConfig", SNAP)
    Projector = get_class_from_dynamic_module(
        "modeling_nemotron_h_omni.NemotronH_Omni_Reasoning_V3VisionProjector", SNAP)
    rc = RadioConfig(hidden_size=64, num_hidden_layers=2, num_attention_heads=4, patch_size=4, max_img_size=64,
                     num_cls_tokens=3, num_registers=7, video_temporal_patch_size=None)
    rc._attn_implementation = "sdpa"
    ref = RadioModel(rc).eval()
    ref.make_preprocessor_external()
    ns = types.SimpleNamespace(downsample_ratio=0.5, video_temporal_patch_size=2, ps_version="v2",
                               vit_hidden_size=64, projector_hidden_size=96,
                               llm_config=types.SimpleNamespace(hidden_size=48, num_nextn_predict_layers=1),
                               vision_config=types.SimpleNamespace(layer_norm_eps=1e-6))
    proj = Projector(ns).eval()
    with torch.no_grad():
        for n, p in list(ref.named_parameters()) + list(proj.named_parameters()):
            if "layer_scale" in n:
                continue                                              # C-RADIO has none: 1.0, as loaded
            p.copy_(torch.randn(p.shape, generator=G) * 0.2 + (1.0 if "norm" in n and n.endswith("weight") else 0.0))
    sd = {**{k: v for k, v in ref.state_dict().items()}, **{f"P.{k}": v for k, v in proj.state_dict().items()}}
    t = {"model.patch_generator.cls_token.token": sd["embeddings.cls_register_token"],
         "model.patch_generator.embedder.weight": sd["embeddings.patch_projection.weight"],
         "model.patch_generator.pos_embed": sd["embeddings.position_embedding"],
         "vision_final_layernorm.weight": sd["P.vision_final_layernorm.weight"],
         "vision_final_layernorm.bias": sd["P.vision_final_layernorm.bias"],
         "mlp1.0.weight": sd["P.mlp1.norm.weight"], "mlp1.1.weight": sd["P.mlp1.linear1.weight"],
         "mlp1.3.weight": sd["P.mlp1.linear2.weight"]}
    for i in range(2):
        r, p = f"encoder.layer.{i}.", f"model.blocks.{i}."
        a = r + "attention.attention."
        t.update({p + "norm1.weight": sd[r + "norm1.weight"], p + "norm1.bias": sd[r + "norm1.bias"],
                  p + "attn.qkv.weight": torch.cat([sd[a + x + ".weight"] for x in ("query", "key", "value")]),
                  p + "attn.qkv.bias": torch.cat([sd[a + x + ".bias"] for x in ("query", "key", "value")]),
                  p + "attn.proj.weight": sd[r + "attention.output.dense.weight"],
                  p + "attn.proj.bias": sd[r + "attention.output.dense.bias"],
                  p + "norm2.weight": sd[r + "norm2.weight"], p + "norm2.bias": sd[r + "norm2.bias"],
                  p + "mlp.fc1.weight": sd[r + "mlp.fc1.weight"], p + "mlp.fc1.bias": sd[r + "mlp.fc1.bias"],
                  p + "mlp.fc2.weight": sd[r + "mlp.fc2.weight"], p + "mlp.fc2.bias": sd[r + "mlp.fc2.bias"]})
    c = cfg(hidden_size=64, num_heads=4, intermediate_size=256, patch_size=4, max_img_size=64, prefix_tokens=10,
            out_hidden_size=48, projector_hidden_size=96)
    ours = NV.RadioTower(c, t, device="cpu", dtype=torch.float32)
    worst = 0.0
    for rows, cols in ((4, 6), (8, 8), (2, 10)):
        px = torch.randn(1, 3, rows * 4, cols * 4, generator=G)
        with torch.no_grad():
            want = proj(px, ref)
        got = ours.encode(NV.patchify(px[0], 4), (1, rows, cols))
        err = ((got - want[0]).abs().max() / want.abs().max()).item()
        worst = max(worst, err)
        assert err < 1e-5, (rows, cols, err)
    return f"3 grids, relative max |diff| {worst:.1e} against the checkpoint's modeling code (fp32)"


if __name__ == "__main__":
    fails = 0
    tests = [(n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        try:
            print(f"  {name:<45} ok   {fn() or ''}", flush=True)
        except Exception as exc:                                   # noqa: BLE001
            fails += 1
            import traceback
            traceback.print_exc()
            print(f"  {name:<45} FAIL ({exc})", flush=True)
    print(f"{len(tests) - fails} passed" + (f", {fails} FAILED" if fails else ""))
    sys.exit(1 if fails else 0)
