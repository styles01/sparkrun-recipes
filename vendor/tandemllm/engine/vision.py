"""The checkpoint's vision tower, and the image half of a prompt.

The checkpoint is a vision-language model. Its language model is what the rest of this engine
serves; next to it, under `model.visual.`, sits a 27-block ViT (0.92 GB, all bf16 -- the FP8
checkpoint leaves the whole tower unquantised) that turns an image into rows of the language
model's own hidden size. This module loads that tower and runs it, and it computes what the
language model needs to read those rows: which prompt positions they take, and which rotary
position each one carries.

Three rules, and each is the reason for a choice below:

  * **Weights as the checkpoint stores them.** bf16 weights, bf16 activations, the same torch ops
    in the same order as the reference implementation (transformers' `Qwen3_5VisionModel`). No
    kernel here quantises anything. What leaves the tower is what the reference computes, to the
    rounding of the GEMM library.
  * **The text path does not change.** Nothing in this file runs for a request without an image.
    The engine reads `Engine.mm` (None for text) once per forward and `Engine.pos_delta` (0 for
    text) where it builds positions; with both at their defaults every forward is the one it was.
  * **A cache key is the image, not its URL.** An image's rows are placeholder tokens in the
    prompt, and a placeholder says nothing about which image it stands for. The caches compare
    `key_ids`, where each image's placeholders are replaced by an id derived from a digest of the
    preprocessed pixels, so the same URL with new content misses and the same content under two
    URLs hits.

THE POSITIONS. Qwen3.x uses a three-axis rotary (mRoPE): every position has a temporal, a height
and a width coordinate, and the rotary frequencies are dealt out among the three axes, interleaved
(`mrope_section` = [11, 11, 10] of the 32 frequencies). A text token has three equal coordinates,
which makes it ordinary RoPE -- that is why the engine's text path can use a 1-D table. An image of
`h x w` merged patches starting at position `s` takes `h * w` prompt rows whose coordinates are
`(s, s + row, s + col)`, and the text after it resumes at `s + max(h, w)`. So after an image the
rotary position of a row is its index plus a (negative) constant, `delta`, which is what
`Engine.pos_delta` holds for the rest of the request.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

VISUAL_PREFIXES = ("model.visual.", "visual.")

#: Placeholder rows get key ids at and above this, which no tokenizer reaches.
KEY_BASE = 1 << 40


# ------------------------------------------------------------------------------------ config
@dataclass
class VisionConfig:
    depth: int
    hidden_size: int
    num_heads: int
    intermediate_size: int
    patch_size: int
    temporal_patch_size: int
    spatial_merge_size: int
    in_channels: int
    out_hidden_size: int
    num_position_embeddings: int
    hidden_act: str = "gelu_pytorch_tanh"
    rope_theta: float = 10000.0
    image_token_id: int = -1
    video_token_id: int = -1
    vision_start_token_id: int = -1
    vision_end_token_id: int = -1
    deepstack: list[int] = field(default_factory=list)

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def patch_dim(self) -> int:
        return self.in_channels * self.temporal_patch_size * self.patch_size ** 2

    @property
    def merge_unit(self) -> int:
        return self.spatial_merge_size ** 2


def load_vision_config(snapshot: str) -> VisionConfig | None:
    """The checkpoint's `vision_config`, or None for a checkpoint without a vision tower."""
    with open(os.path.join(snapshot, "config.json")) as f:
        raw = json.load(f)
    v = raw.get("vision_config")
    if not v:
        return None
    rp = v.get("rope_parameters") or {}
    return VisionConfig(
        depth=v["depth"], hidden_size=v["hidden_size"], num_heads=v["num_heads"],
        intermediate_size=v["intermediate_size"], patch_size=v["patch_size"],
        temporal_patch_size=v.get("temporal_patch_size", 2),
        spatial_merge_size=v.get("spatial_merge_size", 2), in_channels=v.get("in_channels", 3),
        out_hidden_size=v["out_hidden_size"],
        num_position_embeddings=v["num_position_embeddings"],
        hidden_act=v.get("hidden_act", "gelu_pytorch_tanh"),
        rope_theta=float(rp.get("rope_theta", v.get("rope_theta", 10000.0))),
        image_token_id=int(raw.get("image_token_id", -1)),
        video_token_id=int(raw.get("video_token_id", -1)),
        vision_start_token_id=int(raw.get("vision_start_token_id", -1)),
        vision_end_token_id=int(raw.get("vision_end_token_id", -1)),
        deepstack=list(v.get("deepstack_visual_indexes") or []))


# ------------------------------------------------------------------------------------ geometry
def patch_positions(grid: tuple[int, int, int], merge: int, device="cpu") -> torch.Tensor:
    """`[t*h*w, 2]` (row, col) of every patch, in the tower's order: merge blocks of `m x m`
    patches, block-major, repeated per frame (transformers' `get_vision_position_ids`)."""
    t, h, w = grid
    hp, wp = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device),
                            indexing="ij")
    shape = (h // merge, merge, w // merge, merge)
    hp = hp.reshape(shape).transpose(1, 2).flatten()
    wp = wp.reshape(shape).transpose(1, 2).flatten()
    return torch.stack([hp, wp], dim=-1).repeat(t, 1)


def bilinear_taps(grid: tuple[int, int, int], side: int, merge: int, device="cpu"):
    """The four corners and weights that resample the square learned position table to this
    grid, in the tower's patch order: `([4, N] long, [4, N] float32)`. Bilinear with the corners
    aligned (`linspace(0, side - 1, n)`), as transformers' `get_vision_bilinear_indices_and_weights`."""
    t, h, w = grid
    hg = torch.linspace(0, side - 1, h, device=device)
    wg = torch.linspace(0, side - 1, w, device=device)
    hf, wf = hg.int(), wg.int()
    hc, wc = (hf + 1).clamp(max=side - 1), (wf + 1).clamp(max=side - 1)
    hfr, wfr = hg - hf, wg - wf
    hfo, hco = hf * side, hc * side
    idx = [(hfo[:, None] + wf[None, :]).flatten(), (hfo[:, None] + wc[None, :]).flatten(),
           (hco[:, None] + wf[None, :]).flatten(), (hco[:, None] + wc[None, :]).flatten()]
    wts = [((1 - hfr)[:, None] * (1 - wfr)[None, :]).flatten(),
           ((1 - hfr)[:, None] * wfr[None, :]).flatten(),
           (hfr[:, None] * (1 - wfr)[None, :]).flatten(),
           (hfr[:, None] * wfr[None, :]).flatten()]
    hi = torch.arange(h, device=device).view(h // merge, merge)
    wi = torch.arange(w, device=device).view(w // merge, merge)
    order = (hi[:, :, None, None] * w + wi[None, None, :, :]).transpose(1, 2).flatten().repeat(t)
    return (torch.stack([i[order] for i in idx]), torch.stack([x[order] for x in wts]))


def image_tokens(grid: tuple[int, int, int], merge: int) -> int:
    t, h, w = grid
    return t * h * w // (merge * merge)


# ------------------------------------------------------------------------------------ the tower
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    a, b = x.chunk(2, dim=-1)
    return torch.cat((-b, a), dim=-1)


class VisionTower:
    """The ViT and its patch merger, over plain tensors under the checkpoint's own names."""

    def __init__(self, cfg: VisionConfig, tensors: dict[str, torch.Tensor], device="cuda",
                 dtype: torch.dtype = torch.bfloat16):
        if cfg.deepstack:
            # A checkpoint whose tower feeds intermediate layers into the language model; the
            # served one has none (`deepstack_visual_indexes: []`). Refuse rather than drop them.
            raise NotImplementedError("deepstack vision layers are not supported")
        self.cfg = cfg
        self.device = device
        self.dtype = dtype
        self.t = {k: v.to(device=device, dtype=dtype) for k, v in tensors.items()}
        missing = [k for k in self._required() if k not in self.t]
        if missing:
            raise RuntimeError(f"vision tower is missing {len(missing)} tensors, e.g. {missing[:3]}")
        dim = cfg.head_dim // 2
        # computed on the host and moved, as the reference's buffer is
        self.inv_freq = (1.0 / (cfg.rope_theta ** (
            torch.arange(0, dim, 2, dtype=torch.float) / dim))).to(device)
        self.side = int(cfg.num_position_embeddings ** 0.5)
        self.stats = {"images": 0, "patches": 0, "ms": 0.0}

    def _required(self) -> list[str]:
        out = ["patch_embed.proj.weight", "patch_embed.proj.bias", "pos_embed.weight",
               "merger.norm.weight", "merger.norm.bias", "merger.linear_fc1.weight",
               "merger.linear_fc1.bias", "merger.linear_fc2.weight", "merger.linear_fc2.bias"]
        for i in range(self.cfg.depth):
            p = f"blocks.{i}"
            out += [f"{p}.{n}" for n in ("norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias",
                                         "attn.qkv.weight", "attn.qkv.bias", "attn.proj.weight",
                                         "attn.proj.bias", "mlp.linear_fc1.weight",
                                         "mlp.linear_fc1.bias", "mlp.linear_fc2.weight",
                                         "mlp.linear_fc2.bias")]
        return out

    @classmethod
    def from_checkpoint(cls, snapshot: str, cfg: VisionConfig | None = None, device="cuda",
                        dtype: torch.dtype = torch.bfloat16):
        """Read `model.visual.*` from the checkpoint's safetensors (the language-model loader skips
        these names), straight to `device`."""
        from safetensors import safe_open
        cfg = cfg or load_vision_config(snapshot)
        if cfg is None:
            raise RuntimeError(f"{snapshot}: no vision_config")
        idx = os.path.join(snapshot, "model.safetensors.index.json")
        if os.path.isfile(idx):
            with open(idx) as f:
                wm = json.load(f)["weight_map"]
            files = sorted({v for k, v in wm.items() if k.startswith(VISUAL_PREFIXES)})
        else:
            files = sorted(n for n in os.listdir(snapshot) if n.endswith(".safetensors"))
        tensors: dict[str, torch.Tensor] = {}
        for name in files:
            with safe_open(os.path.join(snapshot, name), framework="pt", device="cpu") as f:
                for key in f.keys():
                    for pre in VISUAL_PREFIXES:
                        if key.startswith(pre):
                            tensors[key[len(pre):]] = f.get_tensor(key)
                            break
        return cls(cfg, tensors, device=device, dtype=dtype)

    @property
    def nbytes(self) -> int:
        return sum(v.numel() * v.element_size() for v in self.t.values())

    # --- byte math for the admission check -----------------------------------------
    def activation_bytes(self, n_patches: int) -> int:
        """Peak bytes one image of `n_patches` patches needs inside the tower, upper bound.

        Per patch, live at the widest point of a block: the residual, its norm, the attention
        output and the projection (4 x D, bf16), the qkv (3 x D, bf16), q and k in fp32 for the
        rotary plus their bf16 copies (2 x D x (4 + 2)), and the MLP's two intermediate tensors
        (2 x I, bf16); plus the input patches in fp32 and bf16. Attention runs on the fused
        backends (no score matrix). The merger reads a quarter of the rows and is smaller."""
        c = self.cfg
        d, i = c.hidden_size, c.intermediate_size
        per = 2 * (4 * d + 3 * d + 2 * i) + 2 * d * 6 + c.patch_dim * 6
        return int(n_patches) * per

    # --- the forward --------------------------------------------------------------------------
    def _attn(self, x: torch.Tensor, p: str, cos, sin, seglen: int) -> torch.Tensor:
        c = self.cfg
        S = x.shape[0]
        qkv = F.linear(x, self.t[f"{p}.attn.qkv.weight"], self.t[f"{p}.attn.qkv.bias"])
        q, k, v = qkv.reshape(S, 3, c.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        # rotary in fp32, as the reference
        qd, kd = q.dtype, k.dtype
        qf, kf = q.float(), k.float()
        cf, sf = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
        q = ((qf * cf) + (_rotate_half(qf) * sf)).to(qd)
        k = ((kf * cf) + (_rotate_half(kf) * sf)).to(kd)
        q = q.transpose(0, 1).unsqueeze(0)
        k = k.transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)
        scale = c.head_dim ** -0.5
        outs = []
        # one attention per frame, as the reference's `cu_seqlens` split
        for qs, ks, vs in zip(q.split(seglen, dim=2), k.split(seglen, dim=2),
                              v.split(seglen, dim=2)):
            o = _sdpa(qs, ks, vs, scale)
            outs.append(o.transpose(1, 2).contiguous())
        o = torch.cat(outs, dim=1).reshape(S, -1).contiguous()
        return F.linear(o, self.t[f"{p}.attn.proj.weight"], self.t[f"{p}.attn.proj.bias"])

    def _mlp(self, x: torch.Tensor, p: str) -> torch.Tensor:
        h = F.linear(x, self.t[f"{p}.mlp.linear_fc1.weight"], self.t[f"{p}.mlp.linear_fc1.bias"])
        if self.cfg.hidden_act == "gelu_pytorch_tanh":
            h = F.gelu(h, approximate="tanh")
        elif self.cfg.hidden_act in ("gelu", "gelu_new_exact"):
            h = F.gelu(h)
        elif self.cfg.hidden_act == "silu":
            h = F.silu(h)
        else:
            raise NotImplementedError(f"vision hidden_act {self.cfg.hidden_act}")
        return F.linear(h, self.t[f"{p}.mlp.linear_fc2.weight"], self.t[f"{p}.mlp.linear_fc2.bias"])

    @torch.no_grad()
    def encode(self, pixel_values: torch.Tensor, grid: tuple[int, int, int]) -> torch.Tensor:
        """One image: `pixel_values` `[t*h*w, patch_dim]` as the checkpoint's image processor
        writes them, `grid` its `(t, h, w)` in patches. Returns `[t*h*w / merge^2, out_hidden]`."""
        import time
        t0 = time.perf_counter()
        c = self.cfg
        t, h, w = (int(g) for g in grid)
        n = t * h * w
        if pixel_values.shape[0] != n:
            raise ValueError(f"{pixel_values.shape[0]} patches for a {t}x{h}x{w} grid")
        dev = self.device
        x = pixel_values.to(dev).view(-1, c.in_channels, c.temporal_patch_size, c.patch_size,
                                      c.patch_size)
        k = (c.temporal_patch_size, c.patch_size, c.patch_size)
        x = F.conv3d(x.to(self.dtype), self.t["patch_embed.proj.weight"],
                     self.t["patch_embed.proj.bias"], stride=k).view(-1, c.hidden_size)
        idx, wts = bilinear_taps((t, h, w), self.side, c.spatial_merge_size, device=dev)
        pos = (F.embedding(idx, self.t["pos_embed.weight"]) * wts[:, :, None]).sum(0)
        x = x + pos.to(x.dtype)
        pid = patch_positions((t, h, w), c.spatial_merge_size, device=dev)
        rot = (pid.unsqueeze(-1) * self.inv_freq).flatten(1)
        emb = torch.cat((rot, rot), dim=-1)
        cos, sin = emb.cos(), emb.sin()
        eps = 1e-6
        for i in range(c.depth):
            p = f"blocks.{i}"
            y = F.layer_norm(x, (c.hidden_size,), self.t[f"{p}.norm1.weight"],
                             self.t[f"{p}.norm1.bias"], eps)
            x = x + self._attn(y, p, cos, sin, h * w)
            y = F.layer_norm(x, (c.hidden_size,), self.t[f"{p}.norm2.weight"],
                             self.t[f"{p}.norm2.bias"], eps)
            x = x + self._mlp(y, p)
        x = F.layer_norm(x, (c.hidden_size,), self.t["merger.norm.weight"],
                         self.t["merger.norm.bias"], eps).view(-1, c.hidden_size * c.merge_unit)
        x = F.linear(x, self.t["merger.linear_fc1.weight"], self.t["merger.linear_fc1.bias"])
        x = F.gelu(x)
        x = F.linear(x, self.t["merger.linear_fc2.weight"], self.t["merger.linear_fc2.bias"])
        self.stats["images"] += 1
        self.stats["patches"] += n
        self.stats["ms"] += (time.perf_counter() - t0) * 1e3
        return x


def _sdpa(q, k, v, scale):
    """Non-causal attention over one frame. On a GPU only the fused backends are allowed: the
    unfused one would write a [heads, S, S] score matrix, 137 GB for the largest image the
    processor admits, and on this board that is not an error but a wedge."""
    if q.is_cuda:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION,
                          SDPBackend.CUDNN_ATTENTION]):
            return F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0,
                                                  is_causal=False, scale=scale)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0,
                                          is_causal=False, scale=scale)


# ------------------------------------------------------------------------------------ images
@dataclass
class Image:
    """One preprocessed image: the processor's patches, its grid and the digest of both."""
    pixel_values: torch.Tensor          # [t*h*w, patch_dim] on the host (float32, or the tower's dtype)
    grid: tuple[int, int, int]
    digest: bytes
    source: str = ""                    # "data" or "https", for logs (never the URL)

    def tokens(self, merge: int) -> int:
        return image_tokens(self.grid, merge)


def digest_of(pixel_values: torch.Tensor, grid) -> bytes:
    """The content key: the preprocessed patches and the grid, never the URL."""
    h = hashlib.blake2b(digest_size=16)
    h.update(repr(tuple(int(g) for g in grid)).encode())
    h.update(str(pixel_values.dtype).encode())
    x = pixel_values.detach().to("cpu").contiguous()
    if x.dtype == torch.bfloat16:
        x = x.view(torch.int16)          # numpy has no bfloat16; the bytes are the same
    h.update(x.numpy().tobytes())
    return h.digest()


def key_id(digest: bytes) -> int:
    return KEY_BASE + int.from_bytes(digest[:5], "little")


# ------------------------------------------------------------------------------------ the prompt
@dataclass
class Span:
    start: int                          # first placeholder row in the prompt
    n: int                              # placeholder rows
    image: int                          # index into the request's images


def expand(ids: list[int], images: list[Image], image_token_id: int,
           merge: int) -> tuple[list[int], list[Span]]:
    """The chat template writes ONE placeholder per image; the processor's prompt has one per
    merged patch. Returns the expanded ids and where each image's rows are."""
    pads = [i for i, t in enumerate(ids) if t == image_token_id]
    if len(pads) != len(images):
        raise ValueError(f"the prompt has {len(pads)} image placeholders for {len(images)} images")
    out: list[int] = []
    spans: list[Span] = []
    prev = 0
    for j, (i, img) in enumerate(zip(pads, images)):
        out.extend(ids[prev:i])
        n = img.tokens(merge)
        spans.append(Span(len(out), n, j))
        out.extend([image_token_id] * n)
        prev = i + 1
    out.extend(ids[prev:])
    return out, spans


def rope_positions(n: int, spans: list[Span], grids: list[tuple[int, int, int]],
                   merge: int) -> tuple[torch.Tensor, int]:
    """`([3, n] long, delta)`: every prompt row's (t, h, w) rotary coordinates and the offset the
    rows after the prompt carry (position = index + delta), as transformers' `get_rope_index`."""
    parts: list[torch.Tensor] = []
    cur = 0            # next rotary position
    row = 0            # next prompt row
    for s in spans:
        if s.start > row:
            L = s.start - row
            parts.append(torch.arange(L).view(1, -1).expand(3, -1) + cur)
            cur += L
        t, h, w = grids[s.image]
        lt, lh, lw = t, h // merge, w // merge
        if lt * lh * lw != s.n:
            raise ValueError(f"span of {s.n} rows for a {lt}x{lh}x{lw} merged grid")
        pt = torch.arange(lt).repeat_interleave(lh * lw) + cur
        ph = (torch.arange(lh) + cur).repeat_interleave(lw).repeat(lt)
        pw = (torch.arange(lw) + cur).repeat(lh * lt)
        parts.append(torch.stack([pt, ph, pw], dim=0))
        cur += max(h, w) // merge
        row = s.start + s.n
    if n > row:
        parts.append(torch.arange(n - row).view(1, -1).expand(3, -1) + cur)
    pos = torch.cat(parts, dim=1) if parts else torch.zeros(3, 0, dtype=torch.long)
    delta = int(pos.max()) + 1 - n if n else 0
    return pos.long(), delta


def mrope_axes(rotary_dim: int, section: list[int]) -> list[int]:
    """Which axis (0 = t, 1 = h, 2 = w) each rotary frequency reads: the interleaved layout,
    transformers' `recomposition_frequencies` (h at 1, 4, ... below 3 * section[1], w at 2, 5, ...
    below 3 * section[2], t everywhere else)."""
    nf = rotary_dim // 2
    axes = [0] * nf
    for dim, offset in ((1, 1), (2, 2)):
        for j in range(offset, min(section[dim] * 3, nf), 3):
            axes[j] = dim
    return axes


def mrope_cos_sin(pos3: torch.Tensor, inv: torch.Tensor, axes: torch.Tensor,
                  dtype=torch.bfloat16) -> tuple[torch.Tensor, torch.Tensor]:
    """Cos and sin `[T, rotary_dim]` for rows with three coordinates. For a row whose three
    coordinates are equal this is the engine's 1-D table row bit for bit: the same fp32 product
    of the same two numbers, then the same cos and cast."""
    f = pos3.to(torch.float32)[:, :, None] * inv[None, None, :]          # [3, T, F]
    T = pos3.shape[1]
    sel = axes.view(1, -1).expand(T, -1)                                   # [T, F]
    ff = f.permute(1, 2, 0).gather(2, sel.unsqueeze(-1)).squeeze(-1)       # [T, F]
    emb = torch.cat([ff, ff], dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


class EmbedCache:
    """Encoded images by digest, least-recently-used out, under a byte budget: the rows of an
    image a conversation sends again are not encoded again."""

    def __init__(self, budget_bytes: int):
        from collections import OrderedDict
        self.budget = int(budget_bytes)
        self._d: "OrderedDict[bytes, torch.Tensor]" = OrderedDict()
        self.bytes = 0
        self.stats = {"hits": 0, "misses": 0, "evictions": 0}

    def get(self, digest: bytes):
        x = self._d.get(digest)
        if x is None:
            self.stats["misses"] += 1
            return None
        self._d.move_to_end(digest)
        self.stats["hits"] += 1
        return x

    def put(self, digest: bytes, x: torch.Tensor) -> None:
        b = x.numel() * x.element_size()
        if b > self.budget:
            return
        old = self._d.pop(digest, None)
        if old is not None:
            self.bytes -= old.numel() * old.element_size()
        self._d[digest] = x
        self.bytes += b
        while self.bytes > self.budget and self._d:
            _, o = self._d.popitem(last=False)
            self.bytes -= o.numel() * o.element_size()
            self.stats["evictions"] += 1

    def report(self) -> dict:
        return {"entries": len(self._d), "bytes": self.bytes, "budget": self.budget, **self.stats}


class MMContext:
    """One request's images, as the engine's forward reads them (`Engine.mm`).

    `rows(eng, h, start, T)` is called by the forward for a prompt chunk: it writes the image rows'
    embeddings over the placeholders' and says which rotary each row takes. A chunk without image
    rows keeps the engine's own 1-D table (its coordinates are equal, so that table is exact), so
    such a chunk is computed exactly as a text request computes it. Images are encoded when the
    first chunk that needs them runs -- a prefix the caches restore never pays for its images.
    """

    def __init__(self, ids: list[int], spans: list[Span], images: list[Image], tower: VisionTower,
                 eng_cfg, *, cache: EmbedCache | None = None, on_encode=None):
        self.n = len(ids)
        self.spans = spans
        self.images = images
        self.tower = tower
        self.merge = tower.cfg.spatial_merge_size
        self.pos3, self.delta = rope_positions(self.n, spans, [im.grid for im in images],
                                               self.merge)
        self.axes = mrope_axes(eng_cfg.rotary_dim, list(eng_cfg.mrope_section))
        self.cache = cache
        self.on_encode = on_encode
        self._emb: dict[int, torch.Tensor] = {}
        self.encoded = 0                 # images this request ran the tower for
        self.key_ids = list(ids)
        for s in spans:
            kid = key_id(images[s.image].digest)
            self.key_ids[s.start:s.start + s.n] = [kid] * s.n
        self._dev_pos = None
        self._dev_axes = None

    @property
    def image_tokens(self) -> int:
        return sum(s.n for s in self.spans)

    def release(self) -> None:
        """Drop this request's references to its encoded rows once the prompt is prefilled (the
        embedding cache keeps what its budget allows)."""
        self._emb.clear()

    def embeds(self, j: int) -> torch.Tensor:
        x = self._emb.get(j)
        if x is not None:
            return x
        img = self.images[j]
        x = self.cache.get(img.digest) if self.cache is not None else None
        if x is None:
            if self.on_encode is not None:
                self.on_encode(j + 1, len(self.images))
            try:
                x = self.tower.encode(img.pixel_values, img.grid)
            finally:
                if self.on_encode is not None:
                    self.on_encode(None, None)
            self.encoded += 1
            if self.cache is not None:
                self.cache.put(img.digest, x)
        self._emb[j] = x
        return x

    def rows(self, eng, h: torch.Tensor, start: int, T: int):
        """`(h, positions, rope_rows)` for prompt rows `[start, start + T)`; `rope_rows` is None
        when the chunk holds no image row."""
        end = start + T
        if end > self.n:
            raise ValueError(f"forward past the prompt ({end} > {self.n}) with images attached")
        dev = h.device
        if self._dev_pos is None:
            self._dev_pos = self.pos3.to(dev)
            self._dev_axes = torch.tensor(self.axes, dtype=torch.long, device=dev)
        hit = [s for s in self.spans if s.start < end and s.start + s.n > start]
        pos3 = self._dev_pos[:, start:end]
        positions = pos3[0]
        if not hit:
            return h, positions, None
        h = h.clone()
        for s in hit:
            x = self.embeds(s.image)
            a, b = max(s.start, start), min(s.start + s.n, end)
            h[0, a - start:b - start] = x[a - s.start:b - s.start].to(h.dtype)
        inv = eng.rope_inv_freq()
        cos, sin = mrope_cos_sin(pos3, inv, self._dev_axes)
        return h, positions, (cos, sin)
