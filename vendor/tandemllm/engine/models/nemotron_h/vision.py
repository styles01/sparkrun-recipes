"""Super VL's image path (ENG-365): the checkpoint's RADIO tower and projector, the processor, and the image
rows of a prompt.

An Omni checkpoint (`NemotronH_Omni_Reasoning_V3`) carries a C-RADIO ViT-H/16 (`vision_model.radio_model.*`: 32
blocks of hidden 1,280, 16 heads of 80, MLP 5,120 with exact gelu, a learned position table of 128 x 128 for
2,048-pixel images, 10 prefix tokens: 3 class tokens and 7 registers) and its projector (`vision_projector.
vision_final_layernorm` where the trunk has an MTP head, then `mlp1`: RMSNorm over 5,120, a linear to 20,480,
relu^2, a linear to the trunk's 4,096), all BF16 (1.57 GiB in every Super VL checkpoint: the EXL3 packs copy
it unquantised, NVIDIA's ModelOpt checkpoint keeps it BF16). The pack loader skips these names; this module
loads them and runs them with the reference's ops in the reference's order (transformers' `RadioModel` and
`NemotronH_Omni_Reasoning_V3VisionProjector`, the checkpoint's own modeling files): bf16 weights and
activations, the norms' and the position table's fp32 where the reference computes in fp32.

THE PROCESSOR (`Preprocessor`, the checkpoint's `image_processing_nemotron_h_omni.py` without its remote code):
one tile an image at a dynamic resolution. The 16-pixel patch grid is chosen as the reference's
`_compute_target_patches` chooses it -- the image's patch counts rounded up, scaled down to fit 13,312
patches, up to reach 1,024, each side made even for the pixel shuffle -- then the image is resized to that
grid with torch's antialiased bicubic in fp32 (not PIL's: the reference says why) and normalised with the
CLIP mean and std. The patches are kept as the tower reads them, `[h * w, 3 * 16 * 16]` in the tower's dtype,
so the server's row counts (`Image.tokens(2)`: h * w / 4 rows) and its admission check (`activation_bytes`
of `pixel_values.shape[0]` patches) read them as they read the 27B's.

THE PROMPT (`expand`): the chat template writes `<image>` (id 18) for an image part; the processor writes
`<img>`, then `<image>` once per row, then `</img>`. A row is one 2 x 2 block of patches after the pixel
shuffle: 256 rows for the smallest image, 3,328 for the largest, 1,000 for a 1,280 x 800 screenshot.

THE ROWS (`NhMMContext`): the trunk has no positional encoding (Mamba2 and NoPE attention), so an image row
is only an embedding: the projector's row written over the placeholder's at the prompt chunk that holds it
(transformers' `masked_scatter` into `inputs_embeds`); positions do not move (`delta` 0) and every text row of
the prompt is computed exactly as without the image. It is engine/vision.py's `MMContext` (the embedding
cache, the encode-on-first-chunk, the caches' `key_ids` from the image's digest) without its three-axis
rotary.

Video, the sound tower and the RADIO summary (`summary_idxs`) are not served.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from engine.vision import EmbedCache, Image, MMContext, Span, _sdpa, digest_of, key_id

RADIO = "vision_model.radio_model."
PROJ = "vision_projector."
MLP1 = "mlp1."
#: tensors of the tower the image path does not read: the video patch embedder, the input conditioner (the
#: processor normalises; the reference detaches it with `make_preprocessor_external`), the summary indices
#: the position table resampled as the reference does (the square, then the crop) up to this many bytes
SQUARE_BYTES = 300 << 20
UNUSED = (RADIO + "model.patch_generator.video_embedder.weight", RADIO + "input_conditioner.norm_mean",
          RADIO + "input_conditioner.norm_std", "vision_model.summary_idxs")


# ------------------------------------------------------------------------------------ config
@dataclass
class RadioConfig:
    depth: int
    hidden_size: int
    num_heads: int
    intermediate_size: int
    patch_size: int
    max_img_size: int
    prefix_tokens: int                  # class tokens + registers, dropped after the last block
    layer_norm_eps: float
    out_hidden_size: int                # the trunk's hidden size
    projector_hidden_size: int
    downsample: int                     # the pixel shuffle's factor (1 / downsample_ratio)
    final_norm: bool                    # vision_final_layernorm before the projector (a trunk with MTP)
    image_token_id: int
    image_start_id: int
    image_end_id: int
    min_patches: int
    max_patches: int
    max_model_len: int
    norm_mean: tuple
    norm_std: tuple
    in_channels: int = 3

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def patch_dim(self) -> int:
        return self.in_channels * self.patch_size ** 2

    @property
    def spatial_merge_size(self) -> int:
        """The server's name for the rows' merge (server/images.py `Image.tokens`)."""
        return self.downsample

    @property
    def side(self) -> int:
        return self.max_img_size // self.patch_size


def _added_token(snapshot: str, text: str) -> int:
    with open(os.path.join(snapshot, "tokenizer.json")) as f:
        for t in json.load(f).get("added_tokens", []):
            if t.get("content") == text:
                return int(t["id"])
    raise RuntimeError(f"{snapshot}: the tokenizer has no {text!r} token")


def load_radio_config(snapshot: str) -> RadioConfig | None:
    """The tower's configuration from an Omni checkpoint's config.json and preprocessor_config.json; None for a
    checkpoint without a RADIO tower (Lightning, a GGUF file). What the engine does not implement -- a SwiGLU
    MLP, the v1 pixel shuffle, an activation other than gelu -- stops the load with its name."""
    if not os.path.isfile(os.path.join(snapshot, "config.json")):
        return None
    with open(os.path.join(snapshot, "config.json")) as f:
        raw = json.load(f)
    v = raw.get("vision_config")
    if not v or v.get("model_type") != "radio":
        return None
    for k, want in (("use_swiglu_ffn", False), ("hidden_act", "gelu"), ("qkv_bias", True)):
        if v.get(k, want) != want:
            raise NotImplementedError(f"RADIO {k}={v.get(k)!r} is not implemented (the engine runs {want!r})")
    if raw.get("ps_version", "v2") != "v2":
        raise NotImplementedError(f"pixel shuffle {raw.get('ps_version')!r} is not implemented (v2)")
    if float(v.get("layerscale_value", 1.0)) != 1.0:
        raise NotImplementedError("RADIO layer scale is not implemented (C-RADIO has none)")
    pre = {}
    p = os.path.join(snapshot, "preprocessor_config.json")
    if os.path.isfile(p):
        with open(p) as f:
            pre = json.load(f)
    llm = raw.get("llm_config") or {}
    hidden = int(v["hidden_size"])
    return RadioConfig(
        depth=int(v["num_hidden_layers"]), hidden_size=hidden, num_heads=int(v["num_attention_heads"]),
        intermediate_size=int(hidden * float(v.get("mlp_ratio", 4.0))), patch_size=int(v["patch_size"]),
        max_img_size=int(v.get("max_img_size", 2048)),
        prefix_tokens=int(v.get("num_cls_tokens", 3)) + int(v.get("num_registers", 7)),
        layer_norm_eps=float(v.get("layer_norm_eps", 1e-6)), out_hidden_size=int(llm["hidden_size"]),
        projector_hidden_size=int(raw["projector_hidden_size"]),
        downsample=int(round(1.0 / float(raw.get("downsample_ratio", 0.5)))),
        final_norm=int(llm.get("num_nextn_predict_layers") or 0) > 0,
        image_token_id=int(raw.get("img_context_token_id", _added_token(snapshot, "<image>"))),
        image_start_id=_added_token(snapshot, "<img>"), image_end_id=_added_token(snapshot, "</img>"),
        min_patches=int(pre.get("min_num_patches", v.get("min_num_patches", 1024))),
        max_patches=int(pre.get("max_num_patches", v.get("max_num_patches", 13312))),
        max_model_len=int(pre.get("max_model_len", 16384)),
        norm_mean=tuple(pre.get("norm_mean", raw.get("norm_mean"))),
        norm_std=tuple(pre.get("norm_std", raw.get("norm_std"))))


# ------------------------------------------------------------------------------------ the processor
def target_patches(w: int, h: int, cfg: RadioConfig, budget: int) -> tuple[int, int]:
    """`(w_patches, h_patches)` of an image of `w x h` pixels with `budget` patches available: the reference's
    `_compute_target_patches`, step for step (Python's `round`, which rounds half to even, included)."""
    ps, d = cfg.patch_size, cfg.downsample
    ch = round(h / ps + 0.5)
    cw = round(w / ps + 0.5)
    factor = min(math.sqrt(budget / (ch * cw)), 1.0)
    th = max(math.floor(factor * ch), 1)      # a side the reference floors to 0 (a very long, thin
    tw = max(math.floor(factor * cw), 1)      # image) divides by zero below or resizes to 0 pixels
    if budget > cfg.min_patches and th * tw < cfg.min_patches:
        up = math.sqrt(cfg.min_patches / (th * tw))
        th = math.ceil(up * th)
        tw = math.ceil(up * tw)
    r = th % d
    if r:
        th = th + d - r if (th + d - r) * tw <= budget else max(d, th - r)
    r = tw % d
    if r:
        tw = tw + d - r if th * (tw + d - r) <= budget else max(d, tw - r)
    return tw, th


def image_budget(cfg: RadioConfig, n_images: int = 1, max_pixels: int = 0) -> int:
    """Patches one image may take: the reference's per-image budget (the window's tokens times the shuffle's
    factor squared, at least the minimum for each image, at most `max_num_patches`); `max_pixels` (the
    server's `--image-max-pixels`) lowers the cap."""
    budget = max((cfg.max_model_len - 4) * cfg.downsample ** 2, cfg.min_patches * n_images)
    cap = cfg.max_patches if cfg.max_patches > 0 else budget
    if max_pixels:
        cap = min(cap, max(cfg.min_patches, int(max_pixels) // cfg.patch_size ** 2))
    return max(min(budget, cap), cfg.min_patches)


def pixels(pil, cfg: RadioConfig, budget: int) -> torch.Tensor:
    """An RGB image as the reference's processor writes it: `[3, H, W]` fp32, resized to its patch grid with
    torch's antialiased bicubic and normalised."""
    import numpy as np
    wp, hp = target_patches(pil.width, pil.height, cfg, budget)
    tw, th = wp * cfg.patch_size, hp * cfg.patch_size
    t = torch.from_numpy(np.array(pil, dtype=np.uint8)).permute(2, 0, 1).unsqueeze(0).to(dtype=torch.float32)
    if t.shape[-2] != th or t.shape[-1] != tw:
        t = F.interpolate(t, size=(th, tw), mode="bicubic", align_corners=False, antialias=True)
    mean = torch.tensor(cfg.norm_mean).view(1, 3, 1, 1)
    std = torch.tensor(cfg.norm_std).view(1, 3, 1, 1)
    return ((t / 255.0 - mean) / std).squeeze(0)


def patchify(px: torch.Tensor, patch: int) -> torch.Tensor:
    """`[3, H, W]` -> `[rows * cols, 3 * patch * patch]`, rows outer: RADIO's `_image_to_patches`."""
    c, h, w = px.shape
    r, k = h // patch, w // patch
    return px.reshape(c, r, patch, k, patch).permute(1, 3, 0, 2, 4).reshape(r * k, c * patch * patch)


class Preprocessor:
    """The server's image processor for a RADIO tower (server/images.py's interface: `merge`, `backend`,
    `__call__(pil, source) -> Image`): the patches in the tower's dtype (its first op casts to it, as the
    reference's `pixel_values.to(torch_dtype)`), the grid `(1, rows, cols)` in patches, the digest of both."""

    backend = "native"

    def __init__(self, cfg: RadioConfig, max_pixels: int = 0, dtype=None):
        self.cfg = cfg
        self.merge = cfg.downsample
        self.dtype = dtype
        self.budget = image_budget(cfg, 1, max_pixels)
        self.proc = self

    def __call__(self, pil, source: str = "") -> Image:
        if pil.mode != "RGB":
            pil = pil.convert("RGB")
        px = pixels(pil, self.cfg, self.budget)
        pv = patchify(px, self.cfg.patch_size)
        pv = (pv if self.dtype is None else pv.to(self.dtype)).contiguous()
        grid = (1, px.shape[1] // self.cfg.patch_size, px.shape[2] // self.cfg.patch_size)
        return Image(pixel_values=pv, grid=grid, digest=digest_of(pv, grid), source=source)


# ------------------------------------------------------------------------------------ the prompt
def expand(ids: list[int], images: list[Image], cfg: RadioConfig) -> tuple[list[int], list[Span]]:
    """Each image's one `<image>` as the processor writes it: `<img>`, one `<image>` a row, `</img>`. Returns
    the expanded ids and where each image's rows are."""
    pads = [i for i, t in enumerate(ids) if t == cfg.image_token_id]
    if len(pads) != len(images):
        raise ValueError(f"the prompt has {len(pads)} image placeholders for {len(images)} images")
    out: list[int] = []
    spans: list[Span] = []
    prev = 0
    for j, (i, img) in enumerate(zip(pads, images)):
        out.extend(ids[prev:i])
        out.append(cfg.image_start_id)
        n = img.tokens(cfg.downsample)
        spans.append(Span(len(out), n, j))
        out.extend([cfg.image_token_id] * n)
        out.append(cfg.image_end_id)
        prev = i + 1
    out.extend(ids[prev:])
    return out, spans


class NhMMContext(MMContext):
    """One request's images as NemotronH's prompt chunks read them (`NemotronHServing.mm`): `splice(h, start)`
    writes the image rows of prompt rows `[start, start + T)` over their placeholders' embeddings. No rotary:
    positions are the rows' own (`delta` 0). The embedding cache, the encode on the first chunk that needs an
    image, `key_ids` and `release` are engine/vision.py's `MMContext`'s."""

    def __init__(self, ids: list[int], spans: list[Span], images: list[Image], tower: "RadioTower", *,
                 cache: EmbedCache | None = None, on_encode=None):
        self.n = len(ids)
        self.spans = spans
        self.images = images
        self.tower = tower
        self.merge = tower.cfg.downsample
        self.delta = 0
        self.cache = cache
        self.on_encode = on_encode
        self._emb: dict[int, torch.Tensor] = {}
        self.encoded = 0
        self.key_ids = list(ids)
        for s in spans:
            self.key_ids[s.start:s.start + s.n] = [key_id(images[s.image].digest)] * s.n

    def at(self, row: int) -> bool:
        """Whether prompt row `row` is an image row."""
        return any(s.start <= row < s.start + s.n for s in self.spans)

    def splice(self, h: torch.Tensor, start: int) -> torch.Tensor:
        """`h` [T, H], the embeddings of prompt rows start.., with the images' rows over their placeholders'
        (in place)."""
        T = h.shape[0]
        if start + T > self.n:
            raise ValueError(f"forward past the prompt ({start + T} > {self.n}) with images attached")
        for s in self.spans:
            a, b = max(s.start, start), min(s.start + s.n, start + T)
            if a < b:
                h[a - start:b - start] = self.embeds(s.image)[a - s.start:b - s.start].to(h.dtype)
        return h

    def rows(self, eng, h, start, T):
        raise TypeError("NemotronH reads its image rows through `splice`; it has no rotary rows")


# ------------------------------------------------------------------------------------ the tower
def pixel_shuffle(f: torch.Tensor, rows: int, cols: int, d: int) -> torch.Tensor:
    """`[rows * cols, C]` patch features -> `[rows * cols / d^2, d^2 C]`: the reference's `pixel_shuffle`
    (ps_version v2), its views and permutes in its order. Row (i, j) is the patches (d i + m, d j + k) side
    by side, m outer, k inner."""
    x = f.reshape(1, rows, cols, -1)
    n, a, b, ch = x.size()
    x = x.view(n, a, b // d, ch * d).permute(0, 2, 1, 3).contiguous()
    x = x.view(n, b // d, a // d, ch * d * d).permute(0, 2, 1, 3).contiguous()
    return x.reshape(-1, x.shape[-1])


class RadioTower:
    """The RADIO ViT and the projector, over plain tensors under the checkpoint's names less their prefix
    (`model.blocks.0.attn.qkv.weight`, `vision_final_layernorm.weight`, `mlp1.1.weight`)."""

    def __init__(self, cfg: RadioConfig, tensors: dict[str, torch.Tensor], device="cuda",
                 dtype: torch.dtype = torch.bfloat16):
        self.cfg = cfg
        self.device = device
        self.dtype = dtype
        self.t = {k: v.to(device=device, dtype=dtype) for k, v in tensors.items()}
        missing = [k for k in self._required() if k not in self.t]
        if missing:
            raise RuntimeError(f"RADIO tower is missing {len(missing)} tensors, e.g. {missing[:3]}")
        extra = sorted(set(self.t) - set(self._required()))
        if extra:
            raise RuntimeError(f"RADIO tower has {len(extra)} tensors the engine does not read, e.g. {extra[:3]}")
        self.stats = {"images": 0, "patches": 0, "ms": 0.0}

    def _required(self) -> list[str]:
        c = self.cfg
        out = ["model.patch_generator.cls_token.token", "model.patch_generator.embedder.weight",
               "model.patch_generator.pos_embed", "mlp1.0.weight", "mlp1.1.weight", "mlp1.3.weight"]
        if c.final_norm:
            out += ["vision_final_layernorm.weight", "vision_final_layernorm.bias"]
        for i in range(c.depth):
            p = f"model.blocks.{i}."
            out += [p + n for n in ("norm1.weight", "norm1.bias", "attn.qkv.weight", "attn.qkv.bias",
                                    "attn.proj.weight", "attn.proj.bias", "norm2.weight", "norm2.bias",
                                    "mlp.fc1.weight", "mlp.fc1.bias", "mlp.fc2.weight", "mlp.fc2.bias")]
        return out

    @staticmethod
    def name_of(key: str) -> str | None:
        """The tower's name of a checkpoint key, or None for one it does not read."""
        if key in UNUSED:
            return None
        for pre in (RADIO, PROJ):
            if key.startswith(pre):
                return key[len(pre):]
        return key if key.startswith(MLP1) else None

    @classmethod
    def from_checkpoint(cls, snapshot: str, cfg: RadioConfig | None = None, device="cuda",
                        dtype: torch.dtype = torch.bfloat16) -> "RadioTower | None":
        """The tower's tensors read from the checkpoint's safetensors (the shards that hold them, by the index),
        straight to `device`; None when the checkpoint has a RADIO config and no tower tensors (a GGUF's folder
        of config and tokenizer, whose index may name shards it does not hold)."""
        from safetensors import safe_open
        cfg = cfg or load_radio_config(snapshot)
        if cfg is None:
            raise RuntimeError(f"{snapshot}: no RADIO vision_config")
        idx = os.path.join(snapshot, "model.safetensors.index.json")
        if os.path.isfile(idx):
            with open(idx) as f:
                wm = json.load(f)["weight_map"]
            files = sorted(v for v in {v for k, v in wm.items() if cls.name_of(k) is not None}
                           if os.path.isfile(os.path.join(snapshot, v)))
        else:
            files = sorted(n for n in os.listdir(snapshot) if n.endswith(".safetensors"))
        tensors: dict[str, torch.Tensor] = {}
        for name in files:
            with safe_open(os.path.join(snapshot, name), framework="pt", device="cpu") as f:
                for key in f.keys():
                    n = cls.name_of(key)
                    if n is not None:
                        tensors[n] = f.get_tensor(key)
        if not tensors:
            return None
        return cls(cfg, tensors, device=device, dtype=dtype)

    @property
    def nbytes(self) -> int:
        return sum(v.numel() * v.element_size() for v in self.t.values())

    # --- byte math for the admission check -----------------------------------------
    def activation_bytes(self, n_patches: int) -> int:
        """Peak bytes one image of `n_patches` patches needs, upper bound. Per patch row of a block, bf16: the
        residual, its norm, the qkv, q, k and v transposed, the attention's output and projection, fc1's output
        and its gelu, fc2's (19 x D); the patches in fp32 and bf16; the position rows in fp32 and bf16 and the
        resampled table (`positions`: the square in fp32 and bf16, or a pass under 256 MB). Per row after the shuffle (a quarter of the patches): the
        RMSNorm's fp32 copy, linear1's output and its relu^2 (bf16) and the output row. Attention runs on the
        fused backends."""
        c = self.cfg
        n = int(n_patches)
        rows = n // c.downsample ** 2 + 1
        return (n + c.prefix_tokens) * 2 * 19 * c.hidden_size + n * c.patch_dim * 6 + n * c.hidden_size * 6 \
            + SQUARE_BYTES * 3 // 2 \
            + rows * (4 * c.hidden_size * c.downsample ** 2 + 4 * c.projector_hidden_size + 2 * c.out_hidden_size)

    # --- the forward --------------------------------------------------------------------------
    def positions(self, rows: int, cols: int) -> torch.Tensor:
        """The position table at a `rows x cols` grid `[rows * cols, D]`: RADIO's cropped position embedding
        (`_interpolate_position_embedding`: bilinear in fp32 to the longer side squared, then cropped). Up to
        SQUARE_BYTES of square (a longer side of 247 patches on Super VL: every image the processor makes
        with an aspect ratio up to about 4.5:1) as the reference computes it; past that without the square,
        which for a 2 x 6,656 grid would be 218 GB in fp32 (`separable_positions`)."""
        c = self.cfg
        m = max(rows, cols)
        if m * m * c.hidden_size * 4 <= SQUARE_BYTES:
            return self.square_positions(rows, cols)
        return self.separable_positions(rows, cols)

    def separable_positions(self, rows: int, cols: int) -> torch.Tensor:
        """`positions` without the square: bilinear is separable, and along an axis whose size does not change
        its weights are 1 and 0, so resampling one axis, cropping it, then the other gives the square's crop
        to fp32 rounding (tests/test_nh_vision.py); the longer axis goes first and the channels in chunks."""
        c = self.cfg
        pos = self.t["model.patch_generator.pos_embed"].reshape(1, c.side, c.side, -1).permute(0, 3, 1, 2)
        m = max(rows, cols)
        step = max(1, (256 << 20) // (4 * c.side * m))          # the first pass's output under 256 MB
        out = []
        for a in range(0, c.hidden_size, step):
            p = pos[:, a:a + step].float()
            if cols >= rows:
                p = F.interpolate(p, size=(m, c.side), mode="bilinear", align_corners=False)[:, :, :rows]
                p = F.interpolate(p, size=(rows, m), mode="bilinear", align_corners=False)[..., :cols]
            else:
                p = F.interpolate(p, size=(c.side, m), mode="bilinear", align_corners=False)[..., :cols]
                p = F.interpolate(p, size=(m, cols), mode="bilinear", align_corners=False)[:, :, :rows]
            out.append(p.to(self.dtype))
        return torch.cat(out, dim=1).flatten(2).permute(0, 2, 1)[0]

    def square_positions(self, rows: int, cols: int) -> torch.Tensor:
        """`positions` as the reference writes it: the square, then the crop."""
        c = self.cfg
        pos = self.t["model.patch_generator.pos_embed"].reshape(1, c.side, c.side, -1).permute(0, 3, 1, 2)
        m = max(rows, cols)
        pos = F.interpolate(pos.float(), size=(m, m), mode="bilinear", align_corners=False).to(self.dtype)
        return pos[..., :rows, :cols].flatten(2).permute(0, 2, 1)[0]

    def _block(self, x: torch.Tensor, i: int) -> torch.Tensor:
        c, t = self.cfg, self.t
        p = f"model.blocks.{i}."
        S = x.shape[0]
        y = F.layer_norm(x, (c.hidden_size,), t[p + "norm1.weight"], t[p + "norm1.bias"], c.layer_norm_eps)
        qkv = F.linear(y, t[p + "attn.qkv.weight"], t[p + "attn.qkv.bias"])
        q, k, v = (z.view(1, S, c.num_heads, c.head_dim).transpose(1, 2) for z in qkv.split(c.hidden_size, -1))
        o = _sdpa(q, k, v, c.head_dim ** -0.5).transpose(1, 2).reshape(S, c.hidden_size)
        x = x + F.linear(o, t[p + "attn.proj.weight"], t[p + "attn.proj.bias"])
        y = F.layer_norm(x, (c.hidden_size,), t[p + "norm2.weight"], t[p + "norm2.bias"], c.layer_norm_eps)
        y = F.gelu(F.linear(y, t[p + "mlp.fc1.weight"], t[p + "mlp.fc1.bias"]))
        return x + F.linear(y, t[p + "mlp.fc2.weight"], t[p + "mlp.fc2.bias"])

    def features(self, pixel_values: torch.Tensor, grid: tuple[int, int, int]) -> torch.Tensor:
        """The ViT's patch features `[rows * cols, D]` (RADIO's `features`: the prefix tokens dropped)."""
        c, t = self.cfg, self.t
        _, rows, cols = (int(g) for g in grid)
        if pixel_values.shape != (rows * cols, c.patch_dim):
            raise ValueError(f"{tuple(pixel_values.shape)} patches for a {rows}x{cols} grid")
        x = F.linear(pixel_values.to(device=self.device, dtype=self.dtype), t["model.patch_generator.embedder.weight"])
        x = x + self.positions(rows, cols)
        x = torch.cat([t["model.patch_generator.cls_token.token"], x], dim=0)
        for i in range(c.depth):
            x = self._block(x, i)
        return x[c.prefix_tokens:]

    def project(self, f: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
        """The projector over the features: the final LayerNorm (a trunk with MTP), the v2 pixel shuffle of 2 x 2
        patches into one row of 4 x D, mlp1 (RMSNorm in fp32, linear, relu^2, linear) -> `[rows * cols / 4,
        out_hidden]`."""
        c, t = self.cfg, self.t
        if c.final_norm:
            f = F.layer_norm(f, (c.hidden_size,), t["vision_final_layernorm.weight"],
                             t["vision_final_layernorm.bias"], c.layer_norm_eps)
        x = pixel_shuffle(f, rows, cols, c.downsample)
        h = x.float()
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-5)
        x = (t["mlp1.0.weight"].float() * h).to(self.dtype)
        x = torch.square(F.relu(F.linear(x, t["mlp1.1.weight"])))
        return F.linear(x, t["mlp1.3.weight"])

    @torch.no_grad()
    def encode(self, pixel_values: torch.Tensor, grid: tuple[int, int, int]) -> torch.Tensor:
        """One image: `pixel_values` `[rows * cols, patch_dim]` (`Preprocessor`), `grid` `(1, rows, cols)` in
        patches. Returns its rows `[rows * cols / 4, out_hidden]`."""
        t0 = time.perf_counter()
        _, rows, cols = (int(g) for g in grid)
        out = self.project(self.features(pixel_values, grid), rows, cols)
        self.stats["images"] += 1
        self.stats["patches"] += rows * cols
        self.stats["ms"] += (time.perf_counter() - t0) * 1e3
        return out

    # --- what the server asks of a model's own tower (server/app.py `_load_vision`) ---------------
    def preprocessor(self, max_pixels: int = 0) -> Preprocessor:
        return Preprocessor(self.cfg, max_pixels=max_pixels, dtype=self.dtype)

    def warm(self) -> None:
        """The kernels' first launch at load, not in the first image request: a 4 x 4-patch image."""
        self.encode(torch.zeros(16, self.cfg.patch_dim), (1, 4, 4))

    def context(self, ids: list[int], images: list[Image], cache: EmbedCache | None = None,
                on_encode=None) -> tuple[list[int], NhMMContext]:
        """A request's prompt with its images' rows (`expand`) and its `NhMMContext`."""
        full, spans = expand(ids, images, self.cfg)
        return full, NhMMContext(full, spans, images, self, cache=cache, on_encode=on_encode)


def load_tower(snapshot: str, device="cuda") -> RadioTower | None:
    """The RADIO tower of a NemotronH checkpoint, or None for one without (`ctx.vision`'s `tower_of`)."""
    cfg = load_radio_config(snapshot)
    return None if cfg is None else RadioTower.from_checkpoint(snapshot, cfg, device=device)
