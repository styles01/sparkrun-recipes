"""ENG-365: Super VL's image path against the checkpoint's own implementation, on the real weights, on the board.

    python tools/nh_vision_check.py tower --model ~/models/nemotron-3.5-super-vl-120b-a12b-exl3-4.0 --out t.pt
    ops/hold.sh 15 -- python tools/nh_vision_check.py lm --model <same> --ref t.pt --out l.json

tower  no trunk (the tower and projector: 1.6 GiB in bf16, 3.1 in fp32 beside it). Three synthetic pictures
       (drawn here, nobody's data: shapes, a scene, a dark app screen with text) through our processor and the
       checkpoint's (`image_processing_nemotron_h_omni.py`, remote code): the pixel values must be identical.
       Then the checkpoint's own `RadioModel` and vision projector (its modeling files, remote code) on the
       checkpoint's tensors in fp32 (the gold, from fp32 pixels) and in bf16 (the reference as it runs: its
       pixels cast to bf16), against our tower (engine/models/nemotron_h/vision.py, bf16): max |diff| and
       cosine of the image rows. Bar: ours within 1.5x the reference bf16 run's own distance to the gold.
       With `--nvfp4 <dir>` the tower's tensors are compared byte for byte with NVIDIA's ModelOpt checkpoint
       (the pack copies them; both come from the BF16 checkpoint). `--out` keeps the reference's bf16 rows and
       the pictures for `lm`.

lm     the trunk (the EXL3 pack: ~58 GiB; the box's engine stopped, ops/hold.sh). Each picture with "Describe
       this image." (thinking off): the prompt ids against the checkpoint's processor's `input_ids` (identical);
       then the engine (NemotronHServing, prefill in 4,096-row chunks, the decode graph) decodes `--new` greedy
       tokens twice, once with our tower's rows and once with the reference's bf16 rows spliced over the same
       placeholders. The two texts must be the same, or part only where the run with ours has a near tie
       (its top-2 logit gap at the first differing position is reported). A full BF16 reference of the trunk
       does not fit the board (240 GB); the trunk's own distance to BF16 is the pack's (KL 0.049, docs).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.models.nemotron_h import vision as NV  # noqa: E402

QUESTION = "Describe this image."


def pictures() -> list:
    """Three synthetic pictures, the same on every run: shapes on white, a simple scene, a dark app screen."""
    from PIL import Image, ImageDraw
    a = Image.new("RGB", (640, 480), (255, 255, 255))
    d = ImageDraw.Draw(a)
    d.ellipse([60, 120, 260, 320], fill=(220, 30, 30))
    d.rectangle([330, 140, 500, 310], fill=(30, 70, 220))
    d.polygon([(560, 330), (470, 450), (630, 450)], fill=(40, 170, 60))
    b = Image.new("RGB", (1024, 768))
    d = ImageDraw.Draw(b)
    for y in range(500):
        d.line([(0, y), (1023, y)], fill=(90 + y // 8, 150 + y // 10, 235))
    d.rectangle([0, 500, 1023, 767], fill=(60, 150, 50))
    d.ellipse([820, 60, 940, 180], fill=(250, 210, 40))
    d.rectangle([300, 330, 560, 560], fill=(150, 90, 50))
    d.polygon([(270, 330), (430, 200), (590, 330)], fill=(200, 30, 30))
    d.rectangle([400, 450, 460, 560], fill=(80, 50, 30))
    c = Image.new("RGB", (1280, 800), (24, 26, 34))
    d = ImageDraw.Draw(c)
    d.rectangle([0, 0, 1279, 60], fill=(40, 44, 56))
    d.text((24, 20), "Garden settings", fill=(235, 235, 240))
    d.rounded_rectangle([520, 360, 760, 440], radius=20, fill=(240, 120, 40))
    d.text((610, 392), "START", fill=(255, 255, 255))
    d.ellipse([140, 520, 260, 640], fill=(70, 200, 120))
    return [("shapes", a), ("scene", b), ("app", c)]


def reference(snap: str, dtype, dev):
    """The checkpoint's `RadioModel` and vision projector (its own modeling files) on its tensors."""
    from safetensors import safe_open
    from transformers import AutoConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    cfg = AutoConfig.from_pretrained(snap, trust_remote_code=True)
    cfg.vision_config._attn_implementation = "sdpa"
    Radio = get_class_from_dynamic_module("modeling_radio.RadioModel", snap)
    Proj = get_class_from_dynamic_module("modeling_nemotron_h_omni.NemotronH_Omni_Reasoning_V3VisionProjector", snap)
    radio, proj = Radio(cfg.vision_config), Proj(cfg)
    radio.make_preprocessor_external()
    with open(os.path.join(snap, "model.safetensors.index.json")) as f:
        wm = json.load(f)["weight_map"]
    raw = {}
    for name in sorted({v for k, v in wm.items() if k.startswith(("vision_model.", "vision_projector.", "mlp1."))}):
        with safe_open(os.path.join(snap, name), framework="pt", device="cpu") as f:
            for k in f.keys():
                if k.startswith(("vision_model.", "vision_projector.", "mlp1.")):
                    raw[k] = f.get_tensor(k)
    R = "vision_model.radio_model.model."
    sd = {"embeddings.cls_register_token": raw[R + "patch_generator.cls_token.token"],
          "embeddings.patch_projection.weight": raw[R + "patch_generator.embedder.weight"],
          "embeddings.position_embedding": raw[R + "patch_generator.pos_embed"],
          "embeddings.video_patch_projection.weight": raw[R + "patch_generator.video_embedder.weight"],
          "summary_idxs": raw["vision_model.summary_idxs"]}
    D = cfg.vision_config.hidden_size
    for i in range(cfg.vision_config.num_hidden_layers):
        b, e = f"{R}blocks.{i}.", f"encoder.layer.{i}."
        q, k, v = raw[b + "attn.qkv.weight"].split(D)
        qb, kb, vb = raw[b + "attn.qkv.bias"].split(D)
        a = e + "attention.attention."
        sd.update({a + "query.weight": q, a + "key.weight": k, a + "value.weight": v,
                   a + "query.bias": qb, a + "key.bias": kb, a + "value.bias": vb,
                   e + "attention.output.dense.weight": raw[b + "attn.proj.weight"],
                   e + "attention.output.dense.bias": raw[b + "attn.proj.bias"]})
        for n in ("norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias", "mlp.fc1.weight", "mlp.fc1.bias",
                  "mlp.fc2.weight", "mlp.fc2.bias"):
            sd[e + n] = raw[b + n]
    missing, unexpected = radio.load_state_dict(sd, strict=False)
    assert not unexpected and all("layer_scale" in m for m in missing), (missing, unexpected)
    proj.load_state_dict({"vision_final_layernorm.weight": raw["vision_projector.vision_final_layernorm.weight"],
                          "vision_final_layernorm.bias": raw["vision_projector.vision_final_layernorm.bias"],
                          "mlp1.norm.weight": raw["mlp1.0.weight"], "mlp1.linear1.weight": raw["mlp1.1.weight"],
                          "mlp1.linear2.weight": raw["mlp1.3.weight"]})
    radio.config.torch_dtype = dtype
    return radio.to(dev, dtype).eval(), proj.to(dev, dtype).eval(), cfg


def stats(x: torch.Tensor, y: torch.Tensor) -> dict:
    x, y = x.float(), y.float()
    cos_rows = torch.nn.functional.cosine_similarity(x, y, dim=-1)
    return {"max_abs": round((x - y).abs().max().item(), 5), "ref_max_abs": round(y.abs().max().item(), 3),
            "cos": round(torch.nn.functional.cosine_similarity(x.flatten(), y.flatten(), dim=0).item(), 6),
            "cos_row_min": round(cos_rows.min().item(), 6),
            "rel": round(((x - y).norm() / y.norm()).item(), 6)}


@torch.no_grad()
def check_tower(a) -> dict:
    from transformers import AutoImageProcessor
    dev = "cuda"
    t0 = time.time()
    ours = NV.load_tower(a.model, device=dev)
    load_s = time.time() - t0
    print(f"[tower] ours: {ours.nbytes / 2**30:.3f} GiB of tensors, {len(ours.t)} tensors, {load_s:.1f} s; "
          f"allocated {torch.cuda.memory_allocated() / 2**30:.3f} GiB", flush=True)
    out = {"tower_gib": ours.nbytes / 2**30, "load_s": round(load_s, 1), "images": {}}
    if a.nvfp4:
        from safetensors import safe_open
        with open(os.path.join(a.nvfp4, "model.safetensors.index.json")) as f:
            wm = json.load(f)["weight_map"]
        same, n = 0, 0
        for name in sorted({v for k, v in wm.items() if NV.RadioTower.name_of(k) is not None}):
            with safe_open(os.path.join(a.nvfp4, name), framework="pt", device="cpu") as f:
                for k in f.keys():
                    tn = NV.RadioTower.name_of(k)
                    if tn is not None:
                        n += 1
                        same += int(torch.equal(f.get_tensor(k).to(dev), ours.t[tn]))
        out["nvfp4_identical"] = f"{same}/{n}"
        print(f"[tower] tensors identical to NVIDIA's ModelOpt checkpoint: {same} of {n}", flush=True)
    gold_r, gold_p, cfg = reference(a.model, torch.float32, dev)
    half_r, half_p, _ = reference(a.model, torch.bfloat16, dev)
    rproc = AutoImageProcessor.from_pretrained(a.model, trust_remote_code=True)
    pre = ours.preprocessor()
    keep = {}
    for name, pil in pictures():
        im = pre(pil, "data")
        rpx = rproc(images=[pil], return_tensors="pt")["pixel_values"][0]
        _, rows, cols = im.grid
        same_px = torch.equal(im.pixel_values, NV.patchify(rpx.float(), 16).to(im.pixel_values.dtype))
        px = rpx.unsqueeze(0).to(dev)
        # the gold: every step in fp32, from the fp32 pixels
        g = gold_p._project(gold_r(px.float()).features, rows, cols)[0]
        h = half_p(px, half_r)[0]                                  # the reference as it runs: bf16
        t1 = time.time()
        o = ours.encode(im.pixel_values, im.grid)
        torch.cuda.synchronize()
        ms = (time.time() - t1) * 1e3
        r = {"size": list(pil.size), "grid": [rows, cols], "rows": o.shape[0], "pixels_identical": same_px,
             "ours_vs_ref_bf16": stats(o, h), "ours_vs_gold": stats(o, g), "ref_bf16_vs_gold": stats(h, g),
             "encode_ms": round(ms, 1)}
        r["bar_ok"] = r["ours_vs_gold"]["rel"] <= 1.5 * r["ref_bf16_vs_gold"]["rel"]
        out["images"][name] = r
        keep[name] = h.cpu()
        print(f"[tower] {name} {pil.size} -> {cols}x{rows} patches, {o.shape[0]} rows, pixels identical "
              f"{same_px}; ours vs ref bf16 {r['ours_vs_ref_bf16']}; ours vs gold {r['ours_vs_gold']['rel']} "
              f"(ref bf16 vs gold {r['ref_bf16_vs_gold']['rel']}): {'PASS' if r['bar_ok'] else 'FAIL'}", flush=True)
    sep = ours.separable_positions(4, 300)
    sq = ours.square_positions(4, 300)
    out["separable_positions_max_abs"] = (sep.float() - sq.float()).abs().max().item()
    print(f"[tower] separable positions vs the square at 4x300 (cuda): max |diff| "
          f"{out['separable_positions_max_abs']}", flush=True)
    out["peak_gib"] = torch.cuda.max_memory_allocated() / 2**30
    if a.out:
        torch.save({"rows": keep, "report": out}, a.out)
    return out


@torch.no_grad()
def check_lm(a) -> dict:
    from transformers import AutoProcessor, AutoTokenizer

    from engine.models.nemotron_h.serving import build
    dev = "cuda"
    ref = torch.load(a.ref) if a.ref else None
    tok = AutoTokenizer.from_pretrained(a.model)
    proc = AutoProcessor.from_pretrained(a.model, trust_remote_code=True)
    t0 = time.time()
    srv, info = build(a.model, a.max_len, tok, check=False)
    print(f"[lm] engine loaded in {time.time() - t0:.0f} s", flush=True)
    before = torch.cuda.memory_allocated()
    tower = NV.load_tower(a.model, device=dev)
    tower.warm()
    torch.cuda.synchronize()
    tower_gib = (torch.cuda.memory_allocated() - before) / 2**30
    print(f"[lm] tower: {tower_gib:.3f} GiB allocated beside the trunk", flush=True)
    eos = {int(tok.convert_tokens_to_ids("<|im_end|>"))}

    class Given:
        def __init__(self, rows):
            self.cfg, self.rows = tower.cfg, rows

        def encode(self, pv, grid):
            return self.rows.to(dev)

    def greedy(full, spans, im, tw) -> tuple[list[int], list[float]]:
        srv.restart()
        srv.mm = NV.NhMMContext(full, spans, [im], tw)
        lg = srv.forward(torch.tensor(full, device=dev), 0)[0, -1].float()
        srv.mm.release()
        toks, gaps = [], []
        pos = len(full)
        for _ in range(a.new):
            top = lg.topk(2)
            t = int(top.indices[0])
            toks.append(t)
            gaps.append(round((top.values[0] - top.values[1]).item(), 4))
            if t in eos:
                break
            lg = srv.forward(torch.tensor([t], device=dev), pos)[0, -1].float()
            pos += 1
        srv.mm = None
        return toks, gaps

    out = {"tower_gib_beside_trunk": round(tower_gib, 3), "images": {}}
    pre = tower.preprocessor()
    for name, pil in pictures():
        msgs = [{"role": "user", "content": [{"type": "text", "text": QUESTION},
                                             {"type": "image_url", "image_url": {"url": "x"}}]}]
        text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        ids = tok(text, add_special_tokens=False)["input_ids"]
        im = pre(pil, "data")
        full, spans = NV.expand(ids, [im], tower.cfg)
        want = proc(text=[text], images=[pil], return_tensors="pt")["input_ids"][0].tolist()
        a_toks, a_gaps = greedy(full, spans, im, tower)
        r = {"prompt_ids_identical": full == want, "prompt": len(full), "rows": spans[0].n,
             "ours": tok.decode(a_toks, skip_special_tokens=True)}
        if ref is not None:
            b_toks, _ = greedy(full, spans, im, Given(ref["rows"][name]))
            k = next((i for i, (x, y) in enumerate(zip(a_toks, b_toks)) if x != y), None)
            if k is None and len(a_toks) != len(b_toks):
                k = min(len(a_toks), len(b_toks))
            r.update({"ref_rows": tok.decode(b_toks, skip_special_tokens=True), "identical": k is None,
                      "first_diff": k, "gap_at_diff": a_gaps[k] if k is not None and k < len(a_gaps) else None,
                      "tokens": [len(a_toks), len(b_toks)]})
        out["images"][name] = r
        print(f"[lm] {name}: prompt ids identical {r['prompt_ids_identical']} ({len(full)} ids, {spans[0].n} rows)\n"
              f"     ours:     {r['ours']!r}", flush=True)
        if ref is not None:
            print(f"     ref rows: {r['ref_rows']!r}\n     identical {r['identical']}, first diff {r['first_diff']}, "
                  f"gap there {r['gap_at_diff']}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("tower", "lm"))
    ap.add_argument("--model", required=True, help="a Super VL checkpoint directory (EXL3 pack or ModelOpt)")
    ap.add_argument("--nvfp4", help="tower: NVIDIA's ModelOpt checkpoint, its tower compared byte for byte")
    ap.add_argument("--ref", help="lm: the tower phase's --out (the reference's rows)")
    ap.add_argument("--new", type=int, default=96, help="lm: greedy tokens a run")
    ap.add_argument("--max-len", type=int, default=16384)
    ap.add_argument("--out")
    a = ap.parse_args()
    out = check_tower(a) if a.what == "tower" else check_lm(a)
    if a.what == "lm" and a.out:
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1, default=str)[:6000])


if __name__ == "__main__":
    main()
