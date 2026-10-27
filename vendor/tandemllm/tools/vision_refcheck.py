"""The engine's image path against the published implementation, on the real checkpoint.

Two phases, two processes, because one model at a time fits on the board (and the box rule is one
engine at a time):

    python tools/vision_refcheck.py dump   --out /tmp/vref.pt     # transformers, bf16
    python tools/vision_refcheck.py engine --ref /tmp/vref.pt     # this engine, fp8 as stored

The image set is drawn here, deterministically (shapes, a sign, a bar chart, a grid of digits, a
simple scene; five sizes, so five grids), so the check needs no download and no one's pictures.

The reference is `Qwen3_5ForConditionalGeneration` with the language model's fp8 blocks dequantised
to bf16 exactly as the format defines (the same numbers the engine's kernels read) and the vision
tower as stored (bf16). Per image it records the tower's output, a greedy continuation, and the
teacher-forced logits over that continuation. The engine phase reports, per image:

  * tower     max |diff| and cosine of the image rows against the reference's;
  * logits    tools/refcheck.py's report over the continuation (argmax on confident positions,
              top-5 overlap, KL), the same gate the text path is held to (>= 99 % where gap >= 1);
  * greedy    the engine's own greedy continuation, plain decoding, against the reference's:
              identical tokens, or the first position they part and the reference's top-2 gap
              there (a near-tie is where two correct implementations may part).

`--nvfp4/--head` run the engine phase on the served weight set instead (then the logits compare
the quantised model with the bf16 reference: informative, not a gate).
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

QUESTIONS = (
    ("shapes", (640, 480), "Which shapes are in this image, and what colour is each one?"),
    ("sign", (800, 300), "What does the sign say?"),
    ("bars", (512, 512), "Which bar is the tallest, and which is the shortest?"),
    ("digits", (384, 384), "Read the digits row by row."),
    ("scene", (1024, 768), "Describe this picture in one sentence."),
)


def draw(kind: str, size: tuple[int, int]):
    """One test image, the same pixels every time."""
    from PIL import Image, ImageDraw, ImageFont
    w, h = size
    img = Image.new("RGB", size, (255, 255, 255))
    d = ImageDraw.Draw(img)

    def font(px):
        for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",):
            if os.path.isfile(path):
                return ImageFont.truetype(path, px)
        return ImageFont.load_default(size=px)
    if kind == "shapes":
        d.ellipse((40, 60, 240, 260), fill=(220, 30, 30))
        d.rectangle((300, 80, 480, 260), fill=(30, 60, 220))
        d.polygon([(520, 400), (620, 400), (570, 290)], fill=(30, 170, 60))
    elif kind == "sign":
        img = Image.new("RGB", size, (20, 90, 40))
        d = ImageDraw.Draw(img)
        d.rectangle((10, 10, w - 10, h - 10), outline=(255, 255, 255), width=8)
        d.text((w // 2, h // 2), "OPEN 24 HOURS", fill=(255, 255, 255), font=font(80),
               anchor="mm")
    elif kind == "bars":
        vals = {"A": 120, "B": 340, "C": 60, "D": 220}
        for i, (k, v) in enumerate(vals.items()):
            x = 60 + i * 110
            d.rectangle((x, 460 - v, x + 70, 460), fill=(70, 110, 200))
            d.text((x + 35, 485), k, fill=(0, 0, 0), font=font(28), anchor="mm")
        d.line((40, 460, 490, 460), fill=(0, 0, 0), width=3)
    elif kind == "digits":
        rows = ("7 2 9", "4 1 8", "3 6 5")
        for r, line in enumerate(rows):
            d.text((w // 2, 70 + r * 120), line, fill=(0, 0, 0), font=font(72), anchor="mm")
    elif kind == "scene":
        for y in range(h):
            t = y / h
            d.line((0, y, w, y), fill=(int(120 + 100 * t), int(170 + 60 * t), 255))
        d.rectangle((0, int(h * 0.65), w, h), fill=(60, 150, 50))
        d.ellipse((w - 220, 60, w - 100, 180), fill=(255, 220, 40))
        d.rectangle((300, 330, 560, 520), fill=(200, 60, 40))
        d.polygon([(280, 330), (580, 330), (430, 220)], fill=(120, 40, 30))
        d.rectangle((410, 430, 460, 520), fill=(90, 50, 20))
    return img


def build_inputs(snap: str, kind: str, size, question: str):
    """The processor's prompt for one image and its question (thinking off: a direct answer)."""
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(snap)
    img = draw(kind, size)
    messages = [{"role": "user", "content": [{"type": "image"},
                                             {"type": "text", "text": question}]}]
    text = proc.tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                              tokenize=False, enable_thinking=False)
    enc = proc(text=[text], images=[img], return_tensors="pt")
    return img, enc


def do_dump(a) -> None:
    from transformers import AutoConfig
    from transformers.models.qwen3_5 import modeling_qwen3_5 as HFM
    from safetensors import safe_open
    from engine.config import load_config
    from engine.loader import Weights
    cfg = load_config(a.model)
    snap = cfg.path
    hf = AutoConfig.from_pretrained(snap)
    hf._attn_implementation = "sdpa"
    hf.text_config._attn_implementation = "sdpa"
    hf.vision_config._attn_implementation = "sdpa"
    with torch.device("meta"):
        model = HFM.Qwen3_5ForConditionalGeneration(hf)
    t0 = time.time()
    w = Weights(snap, device=a.device, skip_mtp=True, nvfp4="", fp8_head="")
    sd: dict[str, torch.Tensor] = {}
    for name, t in list(w.t.items()):
        sd["lm_head.weight" if name == "lm_head.weight" else "model.language_model." + name] = t
    for base in sorted(w.q):
        blk = w.q.pop(base)
        sd["model.language_model." + base + ".weight"] = blk.dequant().to(torch.bfloat16)
        blk.w = None
    torch.cuda.empty_cache()
    with safe_open(os.path.join(snap, "outside.safetensors"), framework="pt",
                   device=a.device) as f:
        for k in f.keys():
            if k.startswith("model.visual."):
                sd[k] = f.get_tensor(k)
    missing, _ = model.load_state_dict(sd, strict=False, assign=True)
    missing = [m for m in missing if "rotary" not in m and "inv_freq" not in m]
    if missing:
        raise RuntimeError(f"reference missing weights: {missing[:8]} ({len(missing)} total)")
    model.model.language_model.rotary_emb = HFM.Qwen3_5TextRotaryEmbedding(
        hf.text_config, device=a.device)
    vc = hf.vision_config
    try:
        model.model.visual.rotary_pos_emb = HFM.Qwen3_5VisionRotaryEmbedding(
            vc.hidden_size // vc.num_heads // 2).to(a.device)
    except TypeError:
        model.model.visual.rotary_pos_emb = HFM.Qwen3_5VisionRotaryEmbedding(vc, device=a.device)
    model.eval()
    del sd, w
    print(f"[dump] reference ready in {time.time() - t0:.0f}s, "
          f"{torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)
    out = []
    for kind, size, q in QUESTIONS[:a.n]:
        _, enc = build_inputs(snap, kind, size, q)
        enc = {k: v.to(a.device) for k, v in enc.items()}
        if "mm_token_type_ids" not in enc:
            enc["mm_token_type_ids"] = (enc["input_ids"] == hf.image_token_id).int()
        t1 = time.time()
        with torch.no_grad():
            tower = model.model.get_image_features(enc["pixel_values"],
                                                   enc["image_grid_thw"]).pooler_output
            tower = torch.cat(list(tower)) if isinstance(tower, (list, tuple)) else tower
            gen = model.generate(**enc, max_new_tokens=a.gen, do_sample=False)
            n = enc["input_ids"].shape[1]
            cont = gen[0, n:]
            full = gen[:, :].clone()
            mm = torch.cat([enc["mm_token_type_ids"],
                            torch.zeros(1, cont.numel(), dtype=enc["mm_token_type_ids"].dtype,
                                        device=a.device)], dim=1)
            lg = model(input_ids=full, pixel_values=enc["pixel_values"],
                       image_grid_thw=enc["image_grid_thw"], mm_token_type_ids=mm,
                       attention_mask=torch.ones_like(full)).logits[0, n - 1:n - 1 + cont.numel()]
        text = model_text(snap, cont.tolist())
        print(f"[dump] {kind:7s} {tuple(enc['image_grid_thw'][0].tolist())} prompt {n} "
              f"({time.time() - t1:.0f}s): {text!r}", flush=True)
        out.append({"kind": kind, "size": size, "question": q,
                    "ids": enc["input_ids"][0].cpu(), "grid": enc["image_grid_thw"][0].cpu(),
                    "pixel_values": enc["pixel_values"].float().cpu(),
                    "tower": tower.to(torch.bfloat16).cpu(), "cont": cont.cpu(),
                    "logits": lg.float().cpu(), "text": text})
    torch.save({"snapshot": snap, "items": out, "gen": a.gen}, a.out)
    print(f"[dump] wrote {a.out}", flush=True)


def model_text(snap, ids):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(snap).decode(ids, skip_special_tokens=True)


def do_engine(a) -> None:
    from engine.config import load_config
    from engine.loader import Weights
    from engine.model import Qwen38Engine
    from engine import vision as V
    from tools.refcheck import print_report, report
    ref = torch.load(a.ref, map_location="cpu")
    cfg = load_config(a.model)
    t0 = time.time()
    w = Weights(cfg.path, device=a.device, skip_mtp=True, nvfp4=a.nvfp4, fp8_head=a.head)
    tower = V.VisionTower.from_checkpoint(cfg.path, device=a.device)
    vcfg = tower.cfg
    longest = max(it["ids"].numel() for it in ref["items"]) + ref["gen"] + 64
    eng = Qwen38Engine(cfg, w, max_len=max(4096, longest), device=a.device)
    print(f"[engine] loaded in {time.time() - t0:.0f}s ({w.report()}; tower "
          f"{tower.nbytes / 1e9:.2f} GB)", flush=True)
    gate_ok, rows = True, []
    for it in ref["items"]:
        grid = tuple(int(x) for x in it["grid"].tolist())
        img = V.Image(it["pixel_values"], grid, V.digest_of(it["pixel_values"], grid), "ref")
        ids = it["ids"].tolist()
        # the processor's prompt already has the rows expanded: one span per image
        start = ids.index(vcfg.image_token_id)
        spans = [V.Span(start, img.tokens(vcfg.spatial_merge_size), 0)]
        mm = V.MMContext(ids, spans, [img], tower, cfg)
        with torch.no_grad():
            mine = tower.encode(img.pixel_values, grid).float().cpu()
        theirs = it["tower"].float()
        tdiff = (mine - theirs).abs().max().item()
        tcos = torch.nn.functional.cosine_similarity(mine, theirs, dim=-1).min().item()
        cont = it["cont"].tolist()
        # teacher-forced over the reference's continuation
        eng.reset()
        eng.mm, eng.pos_delta = mm, mm.delta
        with torch.no_grad():
            full = torch.tensor(ids + cont, device=a.device)
            # the prompt as the server prefills it, then the continuation as one block
            logits = eng.forward(full[:len(ids)], start=0)[0, -1:]
            rest = eng.forward(full[len(ids):len(ids) + len(cont) - 1], start=len(ids))[0]
            tf = torch.cat([logits, rest]).float().cpu()
        r = report(tf, it["logits"], f"{it['kind']}: engine vs reference, teacher-forced")
        # the engine's own greedy continuation, plain decoding
        eng.reset()
        out = []
        with torch.no_grad():
            lg = eng.forward(torch.tensor(ids, device=a.device), start=0, last_only=True)[0, -1]
            pos = len(ids)
            for _ in range(len(cont)):
                t = int(lg.argmax())
                out.append(t)
                lg = eng.forward(torch.tensor([t], device=a.device), start=pos)[0, -1]
                pos += 1
        part = next((i for i, (x, y) in enumerate(zip(out, cont)) if x != y), None)
        gap = None
        if part is not None:
            top2 = it["logits"][part].topk(2).values
            gap = (top2[0] - top2[1]).item()
        conf = r["gap1.0"][0]
        ok = conf >= 0.99 and tcos > 0.99
        gate_ok &= ok
        rows.append((it["kind"], grid, tdiff, tcos, r["argmax"], conf, r["kl"], part, gap,
                     model_text(cfg.path, out)))
        print_report(r)
        print(f"    tower max|diff| {tdiff:.4f}  min cosine {tcos:.6f}")
        print(f"    greedy: {'identical ' + str(len(cont)) + ' tokens' if part is None else f'parts at {part} (reference top-2 gap {gap:.3f})'}")
        print(f"    engine: {rows[-1][-1]!r}\n    ref:    {it['text']!r}", flush=True)
    eng.mm, eng.pos_delta = None, 0
    print("\nimage    grid          tower|d|  cos       argmax  conf>=1  KL        greedy")
    for k, g, td, tc, am, cf, kl, part, gap, _ in rows:
        gs = "identical" if part is None else f"parts@{part} gap {gap:.2f}"
        print(f"{k:8s} {str(g):13s} {td:8.4f}  {tc:.6f}  {am * 100:6.2f}  {cf * 100:6.2f}  "
              f"{kl:.5f}  {gs}")
    served = bool(a.nvfp4 or a.head)
    verdict = ("INFO (quantised weights against the bf16 reference: not a gate)" if served
               else ("PASS" if gate_ok else "FAIL"))
    print(f"\nGATE  tower cosine > 0.99 and argmax on confident positions >= 99 %   {verdict}")
    if a.json:
        import json
        with open(a.json, "w") as f:
            json.dump([dict(zip(("kind", "grid", "tower_maxdiff", "tower_mincos", "argmax",
                                 "argmax_confident", "kl", "greedy_parts_at", "gap", "text"), r))
                       for r in rows], f, indent=1)
    sys.exit(0 if (gate_ok or served) else 1)


def do_images(a) -> None:
    os.makedirs(a.out, exist_ok=True)
    for kind, size, q in QUESTIONS:
        p = os.path.join(a.out, f"{kind}.png")
        draw(kind, size).save(p)
        print(p, size, q)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump")
    d.add_argument("--out", required=True)
    d.add_argument("--gen", type=int, default=48)
    d.add_argument("--n", type=int, default=len(QUESTIONS))
    e = sub.add_parser("engine")
    e.add_argument("--ref", required=True)
    e.add_argument("--nvfp4", default="")
    e.add_argument("--head", default="")
    e.add_argument("--json", default="")
    i = sub.add_parser("images")
    i.add_argument("--out", required=True)
    for p in (d, e):
        p.add_argument("--model", default=None)
        p.add_argument("--device", default="cuda")
    a = ap.parse_args()
    {"dump": do_dump, "engine": do_engine, "images": do_images}[a.cmd](a)


if __name__ == "__main__":
    main()
