"""Image input, checked against the published implementation on a random model.

The model is a tiny Qwen3.5-family vision-language checkpoint with random weights, built by
transformers (`Qwen3_5ForConditionalGeneration`), written to disk as a checkpoint and read back by
this engine's own loaders. Everything the 27 B model does with an image is in the structure, not
the size: the ViT with its learned position grid and 2-D rotary, the patch merger, the image rows
in the prompt, the three-axis interleaved rotary, the offset every later row carries.

What is checked, against transformers (fp32 on a CPU):

  1. the tower's output for an image;
  2. the rotary coordinates of a prompt with images (`get_rope_index`);
  3. the logits of the whole prompt, one call and chunked across the image;
  4. the continuation after the prompt -- one token at a time, as a verified block and as a tree --
     which is where `pos_delta` lives;
  5. greedy generation, token for token;

and, without a reference, the properties that make the rest safe:

  6. a text row's three-axis rotary is the engine's 1-D table row, bit for bit, and a prompt chunk
     without image rows is computed exactly as a text request computes it;
  7. the caches key on the image's content: the same prefix with another image misses at the
     image, the same image hits, and a restored state continues bit-identically;
  8. placeholder expansion and its errors.

Run: python tests/test_vision.py     (needs transformers with qwen3_5; skips the reference checks
otherwise)
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

for _k in ("NORM", "GDN", "HEAD", "ATTN", "GDNBLOCK", "GDNTREE", "GDNPRE"):
    os.environ.setdefault(f"QWEN38_FUSED_{_k}", "0")
os.environ.setdefault("QWEN38_TREE_CHAIN_DELEGATE", "0")

import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from engine import cache  # noqa: E402
from engine import vision as V  # noqa: E402

try:
    from transformers.models.qwen3_5 import modeling_qwen3_5 as HFM
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
    HAVE_HF = True
except Exception:  # noqa: BLE001
    HAVE_HF = False

IMG, VS, VE = 90, 91, 92          # image placeholder, vision start, vision end (tiny vocab 97)
DT = torch.float32


# ------------------------------------------------------------------ the tiny checkpoint
def hf_config():
    text = dict(hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4,
                num_key_value_heads=2, head_dim=16, vocab_size=97, rms_norm_eps=1e-6,
                max_position_embeddings=512,
                layer_types=["linear_attention", "linear_attention", "linear_attention",
                             "full_attention"],
                linear_conv_kernel_dim=4, linear_key_head_dim=8, linear_value_head_dim=8,
                linear_num_key_heads=2, linear_num_value_heads=4, mtp_num_hidden_layers=0,
                bos_token_id=0, eos_token_id=1, tie_word_embeddings=False,
                rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                                 "partial_rotary_factor": 0.5, "mrope_section": [2, 1, 1],
                                 "mrope_interleaved": True})
    vis = dict(depth=2, hidden_size=32, num_heads=2, intermediate_size=64, patch_size=4,
               temporal_patch_size=2, spatial_merge_size=2, in_channels=3, out_hidden_size=64,
               num_position_embeddings=16, hidden_act="gelu_pytorch_tanh",
               deepstack_visual_indexes=[])
    return dict(text_config=text, vision_config=vis, image_token_id=IMG, video_token_id=93,
                vision_start_token_id=VS, vision_end_token_id=VE, tie_word_embeddings=False)


_CACHE: dict = {}


def reference():
    """(transformers model, checkpoint dir), built once: random weights, fp32, written to disk
    under the checkpoint's own tensor names."""
    if "ref" in _CACHE:
        return _CACHE["ref"]
    from safetensors.torch import save_file
    raw = hf_config()
    torch.manual_seed(0)
    cfg = Qwen3_5Config(**raw)
    cfg._attn_implementation = "sdpa"
    model = HFM.Qwen3_5ForConditionalGeneration(cfg).to(DT).eval()
    with torch.no_grad():
        # random, not the init's zeros/ones, so every norm and bias is exercised
        for n, p in model.named_parameters():
            p.copy_(torch.randn_like(p) * (0.5 if p.dim() == 1 else 0.15))
            if n.endswith("A_log"):
                p.copy_(torch.rand_like(p) * 0.5)
    d = tempfile.mkdtemp(prefix="tiny-vl-")
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump(raw, f)
    sd = {k: v.contiguous() for k, v in model.state_dict().items() if "rotary" not in k}
    save_file(sd, os.path.join(d, "model.safetensors"))
    _CACHE["ref"] = (model, d)
    return _CACHE["ref"]


def engine_for(d, max_len=256):
    from engine.config import load_config
    from engine.loader import Weights
    from engine.models.qwen3_5.adapter import Qwen38Engine
    cfg = load_config(d)
    w = Weights(d, device="cpu", skip_mtp=True, nvfp4="", fp8_head="")
    eng = Qwen38Engine(cfg, w, max_len=max_len, device="cpu")
    eng.kv.k = eng.kv.k.to(DT)
    eng.kv.v = eng.kv.v.to(DT)
    eng.state.conv = eng.state.conv.to(DT)
    tower = V.VisionTower.from_checkpoint(d, device="cpu", dtype=DT)
    return eng, tower


def image(seed: int, grid=(1, 4, 6), vcfg=None) -> V.Image:
    g = torch.Generator().manual_seed(seed)
    t, h, w = grid
    pv = torch.randn(t * h * w, 3 * 2 * 4 * 4, generator=g)
    return V.Image(pv, grid, V.digest_of(pv, grid), "data")


def prompt(images, before=(5, 6, 7, 8, 9, 10, 11), after=(12, 13, 14, 15)):
    ids = list(before)
    for _ in images:
        ids += [VS, IMG, VE]
    ids += list(after)
    return ids


def hf_inputs(ids, images):
    x = torch.tensor([ids])
    mm = (x == IMG).int()
    pv = torch.cat([im.pixel_values for im in images]) if images else None
    grid = torch.tensor([list(im.grid) for im in images]) if images else None
    return x, mm, pv, grid


def mm_context(eng, tower, ids1, images, **kw):
    full, spans = V.expand(ids1, images, IMG, tower.cfg.spatial_merge_size)
    return full, V.MMContext(full, spans, images, tower, eng.cfg, **kw)


def close(a, b, tol=2e-4):
    d = (a.float() - b.float()).abs().max().item()
    s = b.float().abs().max().item()
    return d <= tol * max(1.0, s), d


def need_hf():
    if not HAVE_HF:
        return "SKIP (transformers without qwen3_5)"
    return None


# ------------------------------------------------------------------ 1. the tower
def test_tower_matches_reference():
    if need_hf():
        return need_hf()
    model, d = reference()
    _, tower = engine_for(d)
    out = []
    for grid in ((1, 4, 6), (1, 8, 4), (1, 2, 2)):
        im = image(11, grid)
        with torch.no_grad():
            ref = model.model.visual(im.pixel_values, grid_thw=torch.tensor([list(grid)]))
            ref = getattr(ref, "pooler_output", ref)
            if isinstance(ref, (list, tuple)):
                ref = torch.cat(list(ref))
            mine = tower.encode(im.pixel_values, grid)
        ok, dd = close(mine, ref, 1e-5)
        assert mine.shape == ref.shape, (mine.shape, ref.shape)
        assert ok, f"grid {grid}: max diff {dd}"
        out.append(f"{grid}:{dd:.1e}")
    return "tower == reference: " + " ".join(out)


def test_patches_kept_in_the_tower_dtype_encode_the_same():
    """The server keeps an image's patches in the tower's dtype (bf16) while the request waits;
    the tower's first op casts to that dtype anyway, so the rows are the same bits."""
    _, d = reference() if HAVE_HF else (None, None)
    if d is None:
        return need_hf()
    tower = V.VisionTower.from_checkpoint(d, device="cpu", dtype=torch.bfloat16)
    im = image(12, (1, 4, 6))
    with torch.no_grad():
        a = tower.encode(im.pixel_values, im.grid)
        b = tower.encode(im.pixel_values.to(torch.bfloat16), im.grid)
    assert torch.equal(a, b)
    return f"{tuple(a.shape)} rows bit-identical from fp32 and bf16 patches"


# ------------------------------------------------------------------ 2. rotary coordinates
def test_rope_positions_match_reference():
    if need_hf():
        return need_hf()
    model, _ = reference()
    ims = [image(1, (1, 4, 6)), image(2, (1, 8, 4))]
    ids1 = prompt(ims, before=(5, 6, 7), after=(8, 9))
    ids1 = ids1[:6] + [20, 21] + ids1[6:]            # text between the two images
    full, spans = V.expand(ids1, ims, IMG, 2)
    pos, delta = V.rope_positions(len(full), spans, [im.grid for im in ims], 2)
    x, mm, _, grid = hf_inputs(full, ims)
    ref, rdelta = model.model.get_rope_index(x, mm, image_grid_thw=grid)
    assert torch.equal(pos, ref[:, 0]), (pos, ref[:, 0])
    assert delta == int(rdelta), (delta, rdelta)
    return f"{len(full)} rows, delta {delta}, identical"


def test_mrope_rows_match_reference_rotary():
    if need_hf():
        return need_hf()
    model, d = reference()
    eng, _ = engine_for(d)
    ims = [image(1, (1, 4, 6))]
    full, spans = V.expand(prompt(ims), ims, IMG, 2)
    pos, _ = V.rope_positions(len(full), spans, [im.grid for im in ims], 2)
    rot = model.model.language_model.rotary_emb
    x = torch.zeros(1, 1, dtype=torch.bfloat16)
    cos_r, sin_r = rot(x, pos[:, None, :])
    axes = torch.tensor(V.mrope_axes(eng.cfg.rotary_dim, eng.cfg.mrope_section))
    cos, sin = V.mrope_cos_sin(pos, eng.rope_inv_freq(), axes)
    assert torch.equal(cos, cos_r[0]) and torch.equal(sin, sin_r[0]), \
        ((cos.float() - cos_r[0].float()).abs().max(), )
    return f"{pos.shape[1]} rows bit-identical to the reference rotary"


# ------------------------------------------------------------------ 3/4/5. the language model
def _hf_logits(model, full, ims):
    x, mm, pv, grid = hf_inputs(full, ims)
    with torch.no_grad():
        return model(input_ids=x, pixel_values=pv, image_grid_thw=grid,
                     mm_token_type_ids=mm).logits[0]


def test_prompt_logits_match_reference():
    if need_hf():
        return need_hf()
    model, d = reference()
    ims = [image(3, (1, 4, 6)), image(4, (1, 4, 4))]
    ids1 = prompt(ims)
    # the floor: this engine against the reference on a TEXT prompt of the same length. The two
    # compute the delta rule by different algorithms (chunked here, the reference's own there), so
    # fp32 logits differ in the fourth decimal before any image is involved.
    n_full = len(V.expand(ids1, ims, IMG, 2)[0])
    base_eng, _ = engine_for(d)
    text = [3 + (i * 7) % 80 for i in range(n_full)]
    with torch.no_grad():
        base = (base_eng.forward(torch.tensor(text), start=0)[0]
                - model(input_ids=torch.tensor([text])).logits[0]).abs().max().item()
    notes = [f"text floor {base:.1e}"]
    for chunk in (0, 5, 16):
        eng, tower = engine_for(d)
        full, mm = mm_context(eng, tower, ids1, ims)
        ref = _hf_logits(model, full, ims)
        eng.mm, eng.pos_delta = mm, mm.delta
        with torch.no_grad():
            if chunk == 0:
                mine = eng.forward(torch.tensor(full), start=0)[0]
            else:
                parts = []
                for i in range(0, len(full), chunk):
                    blk = torch.tensor(full[i:i + chunk])
                    parts.append(eng.forward(blk, start=i)[0])
                mine = torch.cat(parts)
        dd = (mine - ref).abs().max().item()
        assert dd <= max(3 * base, 1e-4), f"chunk {chunk}: max logit diff {dd} (text floor {base})"
        assert torch.equal(mine.argmax(-1), ref.argmax(-1)), f"chunk {chunk}: argmax differs"
        notes.append(f"chunk {chunk or 'all'} {dd:.1e}")
    # the images mattered: the same prompt with the placeholders' own embeddings is far off
    eng, _ = engine_for(d)
    with torch.no_grad():
        blind = eng.forward(torch.tensor(full), start=0)[0]
    assert (blind - ref).abs().max().item() > 100 * max(base, 1e-5), "images made no difference"
    return "prompt logits vs reference: " + ", ".join(notes)


def test_continuation_uses_the_offset():
    """After the prompt every row's rotary is index + delta: one token at a time, a verified
    block and a tree all agree with the reference's teacher-forced logits over the whole text."""
    if need_hf():
        return need_hf()
    from engine.tree import DraftTree  # noqa: F401  (the tree path is exercised via forward_tree)
    model, d = reference()
    ims = [image(5, (1, 6, 4))]
    ids1 = prompt(ims)
    cont = [30, 31, 32, 33, 34]
    eng, tower = engine_for(d)
    full, mm = mm_context(eng, tower, ids1, ims)
    ref = _hf_logits(model, full + cont, ims)
    n = len(full)
    assert mm.delta < 0, mm.delta
    notes = []
    # one token at a time
    eng.mm, eng.pos_delta = mm, mm.delta
    with torch.no_grad():
        eng.forward(torch.tensor(full), start=0, last_only=True)
        steps = [eng.forward(torch.tensor([t]), start=n + i)[0, -1] for i, t in enumerate(cont)]
    ok, dd = close(torch.stack(steps), ref[n:n + len(cont)])
    assert ok, f"single steps: {dd}"
    notes.append(f"steps {dd:.1e}")
    # a verified block
    eng2, _ = engine_for(d)
    eng2.mm, eng2.pos_delta = mm, mm.delta
    with torch.no_grad():
        eng2.forward(torch.tensor(full), start=0, last_only=True)
        blk = eng2.forward_block(torch.tensor(cont), start=n)
    ok, dd = close(blk, ref[n:n + len(cont)])
    assert ok, f"block: {dd}"
    notes.append(f"block {dd:.1e}")
    # a tree whose first branch is the continuation: node rows at depth d take n + d + delta
    eng3, _ = engine_for(d)
    eng3.mm, eng3.pos_delta = mm, mm.delta
    toks = cont[:3] + [40, 41]
    parents = (-1, 0, 1, 0, 3)                 # 30 -> 31 -> 32, and 30 -> 40 -> 41
    with torch.no_grad():
        eng3.forward(torch.tensor(full), start=0, last_only=True)
        lg = eng3.forward_tree(torch.tensor(toks), parents, start=n)
    ok, dd = close(lg[:3], ref[n:n + 3])
    assert ok, f"tree: {dd}"
    notes.append(f"tree {dd:.1e}")
    # and without the offset the rows would be wrong: the check can fail
    eng4, _ = engine_for(d)
    eng4.mm, eng4.pos_delta = mm, 0
    with torch.no_grad():
        eng4.forward(torch.tensor(full), start=0, last_only=True)
        bad = eng4.forward_block(torch.tensor(cont), start=n)
    assert not close(bad, ref[n:n + len(cont)])[0], "the offset made no difference; test is blind"
    return ", ".join(notes) + " (delta %d; without it the rows differ)" % mm.delta


def test_greedy_generation_matches_reference():
    if need_hf():
        return need_hf()
    model, d = reference()
    ims = [image(7, (1, 4, 6))]
    ids1 = prompt(ims)
    eng, tower = engine_for(d)
    full, mm = mm_context(eng, tower, ids1, ims)
    x, mmt, pv, grid = hf_inputs(full, ims)
    with torch.no_grad():
        ref = model.generate(input_ids=x, pixel_values=pv, image_grid_thw=grid,
                             mm_token_type_ids=mmt, attention_mask=torch.ones_like(x),
                             max_new_tokens=12, do_sample=False, eos_token_id=None,
                             pad_token_id=0)[0, len(full):].tolist()
    eng.mm, eng.pos_delta = mm, mm.delta
    out = []
    with torch.no_grad():
        lg = eng.forward(torch.tensor(full), start=0, last_only=True)[0, -1]
        pos = len(full)
        for _ in range(12):
            t = int(lg.argmax())
            out.append(t)
            lg = eng.forward(torch.tensor([t]), start=pos)[0, -1]
            pos += 1
    assert out == ref, (out, ref)
    return f"12 greedy tokens identical: {out}"


# ------------------------------------------------------------------ 6. the text path
def test_text_rows_are_the_table():
    """A row with three equal coordinates gets the engine's own table row, bit for bit."""
    if need_hf():
        return need_hf()
    _, d = reference()
    eng, _ = engine_for(d)
    pos = torch.arange(0, 200)
    cos_t, sin_t = eng.rope(pos)
    axes = torch.tensor(V.mrope_axes(eng.cfg.rotary_dim, eng.cfg.mrope_section))
    cos, sin = V.mrope_cos_sin(pos.view(1, -1).expand(3, -1), eng.rope_inv_freq(), axes)
    assert torch.equal(cos, cos_t) and torch.equal(sin, sin_t)
    return "200 positions bit-identical"


def test_text_chunks_of_an_image_request_are_text():
    """The chunk before the image, run under the image request's context, writes exactly what a
    text request writes: same logits, same KV, same recurrent state, bit for bit."""
    if need_hf():
        return need_hf()
    _, d = reference()
    ims = [image(9, (1, 4, 6))]
    before = tuple(range(20, 36))
    ids1 = prompt(ims, before=before)
    a, tower = engine_for(d)
    full, mm = mm_context(a, tower, ids1, ims)
    b, _ = engine_for(d)
    a.mm, a.pos_delta = mm, mm.delta
    with torch.no_grad():
        la = a.forward(torch.tensor(full[:16]), start=0)
        lb = b.forward(torch.tensor(full[:16]), start=0)
    assert mm.encoded == 0, "an image was encoded for a chunk without image rows"
    assert torch.equal(la, lb)
    assert torch.equal(a.state.S, b.state.S) and torch.equal(a.state.conv, b.state.conv)
    assert torch.equal(a.kv.k[:, :, :, :16], b.kv.k[:, :, :, :16])
    return "logits, KV and state bit-identical; no image encoded"


def test_text_request_leaves_no_trace():
    """`mm=None, pos_delta=0` is the text path: an engine that served an image request and is
    then handed a text request computes what a fresh engine computes."""
    if need_hf():
        return need_hf()
    _, d = reference()
    ims = [image(9, (1, 4, 6))]
    a, tower = engine_for(d)
    full, mm = mm_context(a, tower, prompt(ims), ims)
    a.mm, a.pos_delta = mm, mm.delta
    with torch.no_grad():
        a.forward(torch.tensor(full), start=0, last_only=True)
    text = list(range(3, 30))
    b, _ = engine_for(d)
    a.mm, a.pos_delta = None, 0                       # what generate_stream sets for text
    with torch.no_grad():
        a.reset()
        la = a.forward(torch.tensor(text), start=0)
        lb = b.forward(torch.tensor(text), start=0)
        sa = a.forward(torch.tensor([5, 6]), start=len(text))
        sb = b.forward(torch.tensor([5, 6]), start=len(text))
    assert torch.equal(la, lb) and torch.equal(sa, sb)
    return "bit-identical to a fresh engine"


def test_a_verify_never_reads_the_images():
    """A verify block or tree at start 0 -- what a CUDA-graph capture runs, in the middle of an
    image request -- is not a prompt chunk: with the request's images attached it computes what
    it computes without them. (The first box run captured a graph during an image request, the
    capture took the image path, and a gather in it ran out of bounds.)"""
    if need_hf():
        return need_hf()
    _, d = reference()
    ims = [image(9, (1, 4, 6))]
    a, tower = engine_for(d)
    full, mm = mm_context(a, tower, prompt(ims, before=(5,)), ims)   # image rows from row 2
    assert mm.spans[0].start == 2
    b, _ = engine_for(d)
    toks = torch.tensor([5, 6, 7, 8])
    with torch.no_grad():
        for eng in (a, b):
            eng.forward(torch.tensor([3, 4]), start=0)
        a.mm = mm                                  # attached, pos_delta 0: only mm is in question
        la = a.forward_block(toks, start=0)
        lb = b.forward_block(toks, start=0)
        ta = a.forward_tree(toks, (-1, 0, 1, 0), start=0)
        tb = b.forward_tree(toks, (-1, 0, 1, 0), start=0)
    assert mm.encoded == 0, "a verify encoded an image"
    assert torch.equal(la, lb) and torch.equal(ta, tb)
    return "block and tree at start 0 ignore the images"


# ------------------------------------------------------------------ 7. the caches
def test_cache_keys_on_image_content():
    if need_hf():
        return need_hf()
    _, d = reference()
    before = tuple(range(20, 30))
    x, y = image(21, (1, 4, 6)), image(22, (1, 4, 6))
    ids1 = prompt([x], before=before, after=tuple(range(40, 52)))
    chunk = 8

    def run(eng, tower, img, store):
        full, mm = mm_context(eng, tower, ids1, [img])
        eng.mm, eng.pos_delta = mm, mm.delta
        with torch.no_grad():
            lg, reused, fwd = cache.prefill(eng, None, full, "cpu", store=store, chunk=chunk,
                                            checkpoint=True, key_ids=mm.key_ids)
        return lg, reused, mm

    store = cache.StateStore(1 << 30, chunk=chunk)
    e1, t1 = engine_for(d)
    lx, r0, mx = run(e1, t1, x, store)
    assert r0 == 0
    # same URL, other pixels: the placeholders are identical, the key ids are not
    full_y, _ = V.expand(ids1, [y], IMG, 2)
    assert full_y == V.expand(ids1, [x], IMG, 2)[0]
    e2, t2 = engine_for(d)
    ly, r1, my = run(e2, t2, y, store)
    first_image_row = my.spans[0].start
    assert r1 <= first_image_row, (r1, first_image_row)
    cold_y, _ = engine_for(d)
    lyc, _, _ = run(cold_y, t2, y, None)
    assert torch.equal(ly, lyc) or close(ly, lyc, 1e-6)[0]
    # the same image again: restored past the image, and the same logits as the cold run
    e3, t3 = engine_for(d)
    lx2, r2, _ = run(e3, t3, x, store)
    assert r2 > first_image_row + 1, (r2, first_image_row)
    assert torch.equal(lx2, lx), (lx2 - lx).abs().max()
    assert mx.key_ids != my.key_ids
    return (f"other image: reused {r1} (image at {first_image_row}); same image: reused {r2}, "
            f"logits bit-identical")


def test_resident_prefix_keys_on_image_content():
    if need_hf():
        return need_hf()
    _, d = reference()
    x, y = image(31, (1, 4, 6)), image(32, (1, 4, 6))
    ids1 = prompt([x], before=tuple(range(20, 30)), after=tuple(range(40, 60)))
    eng, tower = engine_for(d)
    res = cache.ResidentPrefix(1 << 26, 8)
    outs = []
    for img in (x, y, x):
        full, mm = mm_context(eng, tower, ids1, [img])
        eng.mm, eng.pos_delta = mm, mm.delta
        with torch.no_grad():
            lg, reused, _ = cache.prefill(eng, None, full, "cpu", chunk=8, resident=res,
                                          key_ids=mm.key_ids)
        outs.append((lg, reused, mm.spans[0].start))
    (lx, r0, s), (ly, r1, _), (lx2, r2, _) = outs
    assert r0 == 0 and r1 <= s and r2 <= s, (r0, r1, r2, s)
    assert not torch.equal(lx, ly)
    assert torch.equal(lx, lx2) or close(lx, lx2, 1e-6)[0]
    return f"resident reuse stops at the image ({r1}, {r2} <= {s})"


def test_embed_cache_skips_the_tower():
    if need_hf():
        return need_hf()
    _, d = reference()
    x = image(41, (1, 4, 6))
    ec = V.EmbedCache(1 << 20)
    eng, tower = engine_for(d)
    for k in range(2):
        full, mm = mm_context(eng, tower, prompt([x]), [x], cache=ec)
        eng.mm, eng.pos_delta = mm, mm.delta
        eng.reset()
        with torch.no_grad():
            eng.forward(torch.tensor(full), start=0, last_only=True)
        assert mm.encoded == (1 if k == 0 else 0), (k, mm.encoded)
    assert ec.stats["hits"] == 1
    return "second request: 0 images encoded"


def test_encode_hook_reports_progress():
    if need_hf():
        return need_hf()
    _, d = reference()
    ims = [image(51, (1, 4, 6)), image(52, (1, 4, 4))]
    seen = []
    eng, tower = engine_for(d)
    full, mm = mm_context(eng, tower, prompt(ims), ims,
                          on_encode=lambda i, n: seen.append((i, n)))
    eng.mm, eng.pos_delta = mm, mm.delta
    with torch.no_grad():
        eng.forward(torch.tensor(full), start=0, last_only=True)
    assert seen == [(1, 2), (None, None), (2, 2), (None, None)], seen
    return "encoding image 1 of 2, 2 of 2"


# ------------------------------------------------------------------ 8. placeholders
def test_expand_and_its_errors():
    ims = [image(1, (1, 4, 6)), image(2, (1, 2, 2))]
    ids = [1, VS, IMG, VE, 2, VS, IMG, VE, 3]
    full, spans = V.expand(ids, ims, IMG, 2)
    assert [s.n for s in spans] == [6, 1]
    assert full == [1, VS] + [IMG] * 6 + [VE, 2, VS, IMG, VE, 3]
    assert spans[0].start == 2 and spans[1].start == 11
    for bad in ([1, 2, 3], [IMG, IMG, IMG]):
        try:
            V.expand(bad, ims, IMG, 2)
        except ValueError:
            continue
        raise AssertionError("a placeholder/image count mismatch was accepted")
    return "rows per image = t*h*w/4; mismatch refused"


def test_digest_of_bf16_patches():
    """The server keeps patches in bf16 (numpy has no bf16): the digest reads their bytes, and
    it differs from the float32 digest of the same values (the dtype is part of the key)."""
    pv = torch.randn(8, 1536, generator=torch.Generator().manual_seed(3))
    b = pv.to(torch.bfloat16)
    d1, d2 = V.digest_of(b, (1, 2, 4)), V.digest_of(b.clone(), (1, 2, 4))
    assert d1 == d2 and len(d1) == 16
    assert d1 != V.digest_of(pv, (1, 2, 4))
    b2 = b.clone()
    b2[0, 0] = b2[0, 0] + 1
    assert V.digest_of(b2, (1, 2, 4)) != d1
    return "bf16 digest stable, changes with one value"


def test_key_ids_follow_content_not_placement():
    a, b = image(1, (1, 4, 6)), image(1, (1, 4, 6))
    c = image(2, (1, 4, 6))
    assert a.digest == b.digest and a.digest != c.digest
    e = image(1, (1, 6, 4))                       # same pixels, other grid: another image
    assert e.digest != a.digest
    assert V.key_id(a.digest) >= V.KEY_BASE
    return "digest follows pixels and grid"


def test_tower_byte_math_is_an_upper_bound():
    """The admission estimate is larger than what the tower allocates (measured on the tiny
    tower with the CPU allocator's peak is not available; checked as a closed form instead)."""
    cfg = V.VisionConfig(depth=27, hidden_size=1152, num_heads=16, intermediate_size=4304,
                         patch_size=16, temporal_patch_size=2, spatial_merge_size=2,
                         in_channels=3, out_hidden_size=5120, num_position_embeddings=2304)
    t = V.VisionTower.__new__(V.VisionTower)
    t.cfg = cfg
    per = t.activation_bytes(1)
    # the widest single tensors per patch: the MLP intermediate (bf16) and q,k in fp32
    assert per > 2 * 4304 + 2 * 1152 * 4
    big = t.activation_bytes(65536)                 # the processor's largest image
    assert big < 8 << 30, big
    return f"{per} B a patch; the largest image {big / 2**30:.2f} GiB"


# ------------------------------------------------------------------ 9. the real checkpoint's files
def test_real_template_and_processor_match_the_published_processor():
    """On a box with the checkpoint (tokenizer and processor files only, no weights read): the
    server's prompt -- the chat template's one placeholder per image, expanded here -- and its
    patches are exactly what the published processor produces for the same conversation."""
    try:
        from engine.config import resolve_snapshot
        snap = resolve_snapshot(None)
    except Exception:  # noqa: BLE001
        return "SKIP (no checkpoint on this machine)"
    import base64
    import io
    from PIL import Image as PILImage
    from transformers import AutoProcessor, AutoTokenizer
    from server import app
    from server import images as images_mod
    g = torch.Generator().manual_seed(5)
    arr = (torch.rand(300, 437, 3, generator=g) * 255).to(torch.uint8).numpy()
    buf = io.BytesIO()
    PILImage.fromarray(arr).save(buf, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    body = {"messages": [{"role": "system", "content": "Be brief."},
                         {"role": "user", "content": [
                             {"type": "text", "text": "What is in this picture?"},
                             {"type": "image_url", "image_url": {"url": url}}]}]}
    tok = AutoTokenizer.from_pretrained(snap)
    app.STATE.update(tok=tok, device="cpu")
    ids, _, _ = app.build_prompt(body)
    vcfg = V.load_vision_config(snap)
    pre = images_mod.Preprocessor(snap, vcfg.spatial_merge_size)
    ims = images_mod.collect(body, images_mod.Limits(), pre)
    full, spans = V.expand(ids.tolist(), ims, vcfg.image_token_id, vcfg.spatial_merge_size)
    proc = AutoProcessor.from_pretrained(snap)
    text = tok.apply_chat_template(body["messages"], add_generation_prompt=True, tokenize=False,
                                   enable_thinking=True)
    pil = PILImage.open(io.BytesIO(buf.getvalue())).convert("RGB")
    ref = proc(text=[text], images=[pil], return_tensors="pt")
    assert full == ref["input_ids"][0].tolist(), "prompt ids differ from the processor's"
    assert tuple(ref["image_grid_thw"][0].tolist()) == ims[0].grid
    assert torch.equal(ims[0].pixel_values, ref["pixel_values"].float())
    # the server's own configuration: patches kept in the tower's dtype
    pre16 = images_mod.Preprocessor(snap, vcfg.spatial_merge_size, dtype=torch.bfloat16)
    ims16 = images_mod.collect(body, images_mod.Limits(), pre16)
    assert ims16[0].pixel_values.dtype == torch.bfloat16
    assert torch.equal(ims16[0].pixel_values, ref["pixel_values"].to(torch.bfloat16))
    assert ims16[0].grid == ims[0].grid
    return (f"{len(full)} ids and {tuple(ims[0].pixel_values.shape)} patches identical "
            f"(grid {ims[0].grid}, {spans[0].n} image rows)")


# ------------------------------------------------------------------ run
if __name__ == "__main__":
    fails = 0
    tests = [(n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        try:
            print(f"  {name:<46} ok   {fn() or ''}", flush=True)
        except AssertionError as e:
            fails += 1
            print(f"  {name:<46} FAIL {e}", flush=True)
    print(f"{len(tests) - fails} passed" + (f", {fails} FAILED" if fails else ""))
    sys.exit(1 if fails else 0)
