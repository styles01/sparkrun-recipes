"""VIS-97: Flash-Next's vision against the official implementation, on a real pack, on the board.

    PYTHONPATH=<repo>:~/fn-ref/site python tools/fn_vision_check.py tower --pack ~/fn2/packs/3.05 --out t.json
    PYTHONPATH=<repo>:~/fn-ref/site python tools/fn_vision_check.py lm --pack ~/fn2/packs/3.05 --out l.json

tower  no trunk (about a gigabyte). The pack's padding: fc1's padded rows and bias are zero in every
       block. Then synthetic images (drawn here, nobody's data) through the server's preprocessing
       (server/images.py `Preprocessor`, the pack's own processor): our tower
       (engine/models/flashnext/vision.py on the EXL3 kernels) against transformers' `Qwen4ExpVisionModel`
       built on the same weights dequantised, in fp32 (the gold) and in bf16. Bar: ours within 1.5x the
       bf16 model's own distance to the gold.

lm     the trunk (about 48 GiB). Two prompts with an image -- one under 2,051 rows (every row dense) and
       one past it (the sparse attention pools blocks of image keys) -- and a text prompt as long as the
       larger (the noise floor). The engine prefills through FlashNextServing with the request's
       MMContext, as the server does, in two forwards split inside the image (a chunk boundary in the
       image rows, off a block boundary), then decodes `--new` greedy tokens on its decode graphs; the
       same prefill again, then those tokens verified as chain blocks of 4 (verify and commit with the
       rotary offset): every pick must be the decode's token. transformers' layers
       (tools/fn_refcheck.py `official_layer`, the engine's weights dequantised, the experts one at a
       time) run the prompt and the generated tokens at once, with the positions of transformers' own
       `get_rope_index` (checked equal to engine/vision.py `rope_positions`) and the same image rows
       (our tower's: `tower` checks the tower). The text rows after the image (the generated ones
       included): KL(official || engine) and top-1 agreement; the image rows, whose next-token guesses
       mean nothing: the relative distance of the streams after the last layer. Bars against the text
       floor: text KL within 1.5x, top-1 agreement within 5 points. `--noise` runs the official layers
       in fp32 too: the engine's streams must sit within 1.5x the official bf16 run's distance to them,
       on the image rows and on the text rows. `--control` runs the official layers with 1-D positions
       on the larger image prompt: its text KL must be 3x the engine's or more (the check sees the
       rotary).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.models.flashnext.pack import PackIndex  # noqa: E402
from engine.models.flashnext.vision import Proj, load_tower  # noqa: E402

QUESTION = "Describe the shapes, their colours and where they are in this picture."


def mem_available_gib() -> float:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / (1 << 20)
    return 1e9


def picture(w: int, h: int, seed: int):
    """A synthetic picture: a two-colour gradient and a few filled shapes."""
    import random

    from PIL import Image, ImageDraw
    rng = random.Random(seed)
    im = Image.new("RGB", (w, h))
    px = im.load()
    a, b = [rng.randrange(256) for _ in range(3)], [rng.randrange(256) for _ in range(3)]
    for y in range(h):
        t = y / max(h - 1, 1)
        row = tuple(int(a[k] * (1 - t) + b[k] * t) for k in range(3))
        for x in range(w):
            px[x, y] = row
    d = ImageDraw.Draw(im)
    for _ in range(6):
        x0, y0 = rng.randrange(w * 3 // 4), rng.randrange(h * 3 // 4)
        x1, y1 = x0 + rng.randrange(w // 10, w // 3), y0 + rng.randrange(h // 10, h // 3)
        fill = tuple(rng.randrange(256) for _ in range(3))
        (d.ellipse if rng.random() < 0.5 else d.rectangle)([x0, y0, x1, y1], fill=fill)
    return im


def dequant(t: dict, name: str) -> torch.Tensor:
    """A tower weight as the official model takes it: a `Proj` dequantised (the blocks' rows joined),
    a tensor as it is."""
    v = t[name]
    if isinstance(v, Proj):
        return torch.cat([b.dequant() for b in v.blocks]).float()
    return v.float()


def official_tower(tower, vcfg_raw: dict, dtype, dev):
    """transformers' `Qwen4ExpVisionModel` on the tower's weights dequantised, fc1 and fc2 cut back from
    the converter's 4,352 padded columns to the model's 4,304; the parameters in `dtype`, the rotary's
    buffers fp32 as a checkpoint loaded in `dtype` keeps them."""
    from transformers.models.qwen4_exp import modeling_qwen4_exp as m
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpVisionConfig
    cfg = Qwen4ExpVisionConfig(**{k: v for k, v in vcfg_raw.items() if k != "model_type"})
    cfg._attn_implementation = "eager"
    model = m.Qwen4ExpVisionModel(cfg)
    I = cfg.intermediate_size
    sd = {}
    for name, p in model.state_dict().items():
        w = dequant(tower.t, name)
        if name.endswith(("mlp.linear_fc1.weight", "mlp.linear_fc1.bias")):
            w = w[:I]
        elif name.endswith("mlp.linear_fc2.weight"):
            w = w[:, :I]
        sd[name] = w.reshape(p.shape).cpu()
    model.load_state_dict(sd)
    for p in model.parameters():
        p.data = p.data.to(dtype)
    return model.to(dev).eval()


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


@torch.no_grad()
def check_tower(a) -> dict:
    from server import images as images_mod
    dev = "cuda"
    idx = PackIndex.open(a.pack, table=False)
    tower = load_tower(idx, a.pack, device=dev)
    raw = json.load(open(os.path.join(a.pack, "config.json")))["vision_config"]
    I = raw["intermediate_size"]
    pad = []
    for i in range(tower.cfg.depth):
        fc1 = tower.t[f"blocks.{i}.mlp.linear_fc1.weight"].blocks[0]
        w = fc1.dequant()
        pad.append(max(w[I:].abs().max().item(), tower.t[f"blocks.{i}.mlp.linear_fc1.bias"][I:].abs().max().item()))
    print(f"[tower] fc1 padded rows and bias: max |value| {max(pad)} over {len(pad)} blocks", flush=True)
    gold = official_tower(tower, raw, torch.float32, dev)
    half = official_tower(tower, raw, torch.bfloat16, dev)
    pre = images_mod.Preprocessor(a.pack, tower.cfg.spatial_merge_size, dtype=tower.dtype)
    rows = []
    for k, (w, h) in enumerate(((448, 336), (640, 640), (1024, 768), (333, 517))):
        img = pre(picture(w, h, k))
        grid = torch.tensor([img.grid], device=dev)
        ours = tower.encode(img.pixel_values, img.grid)
        g = gold(img.pixel_values.to(dev).float(), grid).pooler_output
        hb = half(img.pixel_values.to(dev), grid).pooler_output
        r = {"size": [w, h], "grid": list(img.grid), "rows": ours.shape[0], "ours_vs_gold": round(rel(ours, g), 5),
             "bf16_vs_gold": round(rel(hb, g), 5), "ours_vs_bf16": round(rel(ours, hb), 5)}
        r["ok"] = r["ours_vs_gold"] <= 1.5 * r["bf16_vs_gold"]
        rows.append(r)
        print(json.dumps(r), flush=True)
    return {"padding_max_abs": max(pad), "images": rows, "tower_bytes": tower.nbytes,
            "ok": max(pad) == 0 and all(r["ok"] for r in rows)}


# --------------------------------------------------------------------------------------------- lm
def official_positions(full: list[int], images, merge: int, image_id: int):
    """transformers' own `get_rope_index` over the expanded prompt: [3, T] and the delta."""
    import types

    from transformers.models.qwen4_exp import modeling_qwen4_exp as m
    ids = torch.tensor([full])
    me = types.SimpleNamespace(config=types.SimpleNamespace(vision_config=types.SimpleNamespace(
        spatial_merge_size=merge)))
    me.get_vision_position_ids = lambda *a, **k: m.Qwen4ExpModel.get_vision_position_ids(me, *a, **k)
    pos, delta = m.Qwen4ExpModel.get_rope_index(me, ids, mm_token_type_ids=(ids == image_id).int(),
                                                image_grid_thw=torch.tensor([im.grid for im in images]))
    return pos[:, 0], int(delta.view(-1)[0])


@torch.no_grad()
def official_logits(eng, tcfg, ids: torch.Tensor, emb: torch.Tensor, pos3: torch.Tensor, rows: slice,
                    dev, dtype, head: bool = True):
    """transformers' layers over the whole sequence (`emb` [T, H] the input rows, `pos3` [3, T] their
    positions), layer by layer on the engine's weights in `dtype`: log-probabilities of `rows` [R, vocab]
    fp32 (None without `head`) and their streams after the last layer [R, hc H]."""
    from transformers.models.qwen4_exp import modeling_qwen4_exp as m

    from tools.fn_refcheck import official_layer
    c = eng.cfg
    T = ids.numel()
    rot = m.Qwen4ExpTextRotaryEmbedding(tcfg).to(dev)
    cos, sin = rot(torch.zeros(1, T, dtype=dtype, device=dev), pos3.view(3, 1, T).to(dev))
    tril = torch.tril(torch.ones(T, T, dtype=torch.bool, device=dev)).view(1, 1, T, T)
    mask4 = torch.where(tril, torch.zeros((), dtype=dtype, device=dev),
                        torch.tensor(torch.finfo(dtype).min, dtype=dtype, device=dev))
    h = emb.to(dev, dtype).repeat(1, c.hc_count)[None]
    for li in range(c.num_hidden_layers):
        L = official_layer(tcfg, li, eng, dtype, dev)
        ex = L.mlp.experts
        del ex.forward, ex.gate_up_proj, ex.down_proj         # the class's forward over lazy experts
        ex.gate_up_proj = LazyExperts(eng.banks[li], True, dtype)
        ex.down_proj = LazyExperts(eng.banks[li], False, dtype)
        out = L(h, position_embeddings=(cos, sin), attention_mask=mask4, conv_mask=None, past_key_values=None,
                ple_input_ids=ids[None].to(dev))
        h = out[0] if isinstance(out, tuple) else out
        del L
        torch.cuda.empty_cache()
    if not head:
        return None, h[0][rows]
    x = eng.mixer(h[0][rows], "hyper_connection_mixer", combine=False)
    head = eng.W["lm_head.weight"].dequant()
    lg = torch.cat([torch.nn.functional.linear(x[i:i + 256], head).float() for i in range(0, x.shape[0], 256)])
    del head
    return torch.log_softmax(lg, -1), h[0][rows]


class LazyExperts:
    """A layer's routed experts where transformers' `Qwen4ExpTextExperts.forward` indexes them
    (`gate_up_proj[e]`, `down_proj[e]`): each dequantised when its turn comes, so a prompt that routes
    to every expert never holds the layer's bank in bf16 (tools/fn_refcheck.py stacks the experts a
    short prompt hits; 2,000 rows hit all 512, 15 GB a layer with the stacking's copies)."""

    def __init__(self, bank, gate_up: bool, dtype):
        self.bank, self.gate_up, self.dtype = bank, gate_up, dtype

    def __getitem__(self, e) -> torch.Tensor:
        e = int(e)
        if self.gate_up:
            return torch.cat([self.bank.block(e, 0).dequant(), self.bank.block(e, 1).dequant()]).to(self.dtype)
        return self.bank.block(e, 2).dequant().to(self.dtype)


def compare(ref_lp: torch.Tensor, eng_lg: torch.Tensor) -> dict:
    """KL(official || engine) per row, top-1 agreement, the official margin where the top-1 differs."""
    lq = torch.log_softmax(eng_lg.float(), -1)
    kl = (ref_lp.exp() * (ref_lp - lq)).sum(-1)
    v, i = torch.topk(ref_lp, 2, dim=-1)
    agree = i[:, 0] == eng_lg.argmax(-1)
    margins = (v[:, 0] - v[:, 1])[~agree]
    return {"rows": int(kl.numel()), "kl_mean": round(kl.mean().item(), 5), "kl_p99": round(kl.quantile(0.99).item(), 5),
            "kl_max": round(kl.max().item(), 5), "top1_agree": round(agree.float().mean().item(), 4),
            "disagree_margins": [round(x, 3) for x in margins.tolist()[:20]]}


@torch.no_grad()
def check_lm(a) -> dict:
    from transformers import AutoTokenizer
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig

    from engine.models.flashnext.serving import build
    from engine.vision import MMContext, expand
    from server import images as images_mod
    from tools.fn_generate import class_prompt
    from tools.fn_prefill_check import head_logits
    dev, dtype = "cuda", torch.bfloat16
    tok = AutoTokenizer.from_pretrained(a.pack)
    t0 = time.time()
    eng, _, info = build(a.pack, 8192, tok, spec=False)
    fn, c = eng.fn, eng.cfg
    tower = load_tower(PackIndex.open(a.pack, table=False), a.pack, device=dev)
    print(f"[lm] loaded in {time.time() - t0:.0f} s, MemAvailable {mem_available_gib():.1f} GiB", flush=True)
    raw = json.load(open(os.path.join(a.pack, "config.json")))["text_config"]
    tcfg = Qwen4ExpTextConfig(**{k: v for k, v in raw.items() if k not in ("model_type", "mtp")})
    tcfg._attn_implementation = "eager"
    pre = images_mod.Preprocessor(a.pack, tower.cfg.spatial_merge_size, dtype=tower.dtype)
    merge, image_id = tower.cfg.spatial_merge_size, tower.cfg.image_token_id

    def chat(content) -> list[int]:
        text = tok.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
        return tok(text, add_special_tokens=False)["input_ids"]

    def prefill(full: list[int], mm, split: int):
        """The prompt in two forwards (split at `split`); its streams and the last row's logits."""
        eng.restart()
        eng.mm, eng.pos_delta = mm, (mm.delta if mm is not None else 0)
        lg = eng.forward(torch.tensor(full[:split]), 0)
        stacks = [fn.prompt_stack]
        lg = eng.forward(torch.tensor(full[split:]), split)
        return torch.cat(stacks + [fn.prompt_stack]), lg

    def run(name: str, full: list[int], mm, control: bool = False) -> dict:
        """The engine's prefill, greedy decode and the same tokens verified; the official pass; the rows."""
        n = len(full)
        split = (mm.spans[0].start + 37) if mm is not None else n // 2
        stack, lg = prefill(full, mm, split)
        gen, gen_lg, gen_h = [int(lg.view(-1).argmax())], [], []
        for j in range(a.new - 1):
            lg = eng.forward(torch.tensor([gen[-1]]), n + j)
            gen_lg.append(lg.view(-1).float().clone())
            gen_h.append(eng.hidden_post_norm.view(1, -1).clone())
            gen.append(int(lg.view(-1).argmax()))
        # the same prefill, then the decode's tokens as chain blocks of 4: every pick is the decode's token
        _, lg0 = prefill(full, mm, split)
        picks, at = [int(lg0.view(-1).argmax())], n
        for b0 in range(0, len(gen) - 1, 4):
            block = gen[b0:b0 + 4]
            eng.forward_tree(torch.tensor(block), list(range(-1, len(block) - 1)), at)
            picks += [int(x) for x in eng.picks.tolist()]
            eng.commit_tree(list(range(len(block))))
            at += len(block)
        bad = [j for j in range(len(gen)) if picks[j] != gen[j]]
        ids = torch.tensor(full + gen[:-1])
        T = ids.numel()
        eng_lg = torch.cat([torch.cat([head_logits(fn, stack[i:i + 64]) for i in range(0, n, 64)]),
                            torch.stack(gen_lg)])
        eng_h = torch.cat([stack] + gen_h)
        emb = fn.embed(ids.to(dev))
        image = torch.zeros(T, dtype=torch.bool)
        if mm is not None:
            for s in mm.spans:
                emb[s.start:s.start + s.n] = mm.embeds(s.image).to(emb.dtype)
                image[s.start:s.start + s.n] = True
            pos3, delta = official_positions(full, mm.images, merge, image_id)
            assert torch.equal(pos3, mm.pos3) and delta == mm.delta, "get_rope_index != rope_positions"
            pos3 = torch.cat([pos3, (torch.arange(n, T) + delta).view(1, -1).expand(3, -1)], 1)
        else:
            pos3 = torch.arange(T).view(1, -1).expand(3, -1)
        if control:
            pos3 = torch.arange(T).view(1, -1).expand(3, -1)
        t1 = time.time()
        ref, ref_h = official_logits(fn, tcfg, ids, emb, pos3, slice(0, T), dev, dtype)
        # text rows: after the first image (all of the floor's), the generated ones included
        text = ~image & (torch.arange(T) >= (mm.spans[0].start if mm is not None else 0))

        def rel_rows(x, y):
            return ((x.float() - y.float()).norm(dim=-1) / y.float().norm(dim=-1)).cpu()
        dh = rel_rows(eng_h, ref_h)
        noise = None
        if a.noise and not control:
            # the bf16 noise floor of the streams: the official layers in fp32, against which the engine and
            # the official bf16 run are both measured
            _, ref32_h = official_logits(fn, tcfg, ids, emb.float(), pos3, slice(0, T), dev, torch.float32, head=False)
            e32, b32 = rel_rows(eng_h, ref32_h), rel_rows(ref_h, ref32_h)
            noise = {k: {"engine": round(e32[m].mean().item(), 5), "official_bf16": round(b32[m].mean().item(), 5)}
                     for k, m in (("text", text), ("image", image)) if m.any()}
            del ref32_h
        res = {"name": name, "prompt_rows": n, "image_rows": int(image.sum()), "text_rows": int(text.sum()),
               "delta": mm.delta if mm is not None else 0, "split": split, "verify_picks_equal_decode": not bad,
               "verify_first_difference": bad[:1], "generated": tok.decode(gen),
               "text": compare(ref[text.to(dev)], eng_lg[text.to(dev)]),
               "streams_rel_text": round(dh[text].mean().item(), 5),
               "streams_rel_image": round(dh[image].mean().item(), 5) if mm is not None else None,
               "image_rows_top1_agree": round((ref[image.to(dev)].argmax(-1) == eng_lg[image.to(dev)].argmax(-1))
                                              .float().mean().item(), 4) if mm is not None else None,
               "streams_vs_fp32": noise, "official_s": round(time.time() - t1),
               "mem_avail_gib": round(mem_available_gib(), 1)}
        print(json.dumps(res), flush=True)
        return res

    out = {"pack": a.pack, "new": a.new, "load": info["load"], "runs": []}
    small, large = picture(448, 320, 1), picture(1664, 1344, 2)    # 140 and 2,184 image rows
    prompts = []
    for name, pic in (("image dense", small), ("image sparse", large)):
        img = pre(pic)
        ids = chat([{"type": "image"}, {"type": "text", "text": QUESTION}])
        full, spans = expand(ids, [img], image_id, merge)
        prompts.append((name, full, MMContext(full, spans, [img], tower, c)))
    for name, full, mm in prompts:
        out["runs"].append(run(name, full, mm))
    floor = class_prompt(tok, "prose", len(prompts[-1][1]))
    out["runs"].append(run("text floor", floor, None))
    fl = out["runs"][2]
    out["ok"] = all(r["verify_picks_equal_decode"] for r in out["runs"]) and all(
        r["text"]["kl_mean"] <= 1.5 * fl["text"]["kl_mean"] + 1e-3
        and r["text"]["top1_agree"] >= fl["text"]["top1_agree"] - 0.05 for r in out["runs"][:2])
    if a.noise:
        out["ok"] = out["ok"] and all(v["engine"] <= 1.5 * v["official_bf16"] for r in out["runs"][:3]
                                      for v in r["streams_vs_fp32"].values())
    if a.control:
        name, full, mm = prompts[-1]
        r = run(name + ", official with 1-D positions", full, mm, control=True)
        r["sees_the_rotary"] = r["text"]["kl_mean"] >= 3 * out["runs"][1]["text"]["kl_mean"]
        out["runs"].append(r)
        out["ok"] = out["ok"] and r["sees_the_rotary"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("tower", "lm"))
    ap.add_argument("--pack", required=True)
    ap.add_argument("--new", type=int, default=24, help="greedy tokens the engine decodes (lm)")
    ap.add_argument("--control", action="store_true", help="lm: also the official layers with 1-D positions")
    ap.add_argument("--noise", action="store_true", help="lm: also the official layers in fp32 (the streams' bf16 noise)")
    ap.add_argument("--out")
    a = ap.parse_args()
    a.pack = os.path.expanduser(a.pack)
    res = check_tower(a) if a.what == "tower" else check_lm(a)
    print(f"[{a.what}] {'PASS' if res['ok'] else 'FAIL'}", flush=True)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
