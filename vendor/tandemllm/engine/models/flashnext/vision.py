"""Flash-Next's vision tower (VIS-97).

The pack carries the tower beside the language model (`model.visual.*`, 987 tensors): the 27-block
ViT of engine/vision.py (hidden 1,152, 16 heads, patch 16, temporal patch 2, merge 2, out 2,560, no
deepstack), its projections quantised to EXL3 by turboderp's converter (5 bits in 3.05, 4 in 2.05, 6
in 5.05 and 6.05; the attention's q, k and v apart), the norms, biases, the patch embedding and the
learned position table as plain tensors. The pack also keeps the attention's fused bf16 `qkv`; it is
not read (the converter's q, k and v are what the quantised tower is). The converter pads the MLP's
4,304 columns to 4,352: the padded rows of fc1 and its bias are stored as zeros, so the padded outputs
are gelu(0) = 0 and fc2's padded columns add nothing (tools/fn_vision_check.py checks it on the pack).

`FlashNextVisionTower` is engine/vision.py's tower, its code unchanged (that file is in the 27B's
cache fingerprint): each projection's weight in `self.t` is a `Proj`, which the tower's
`F.linear(x, weight, bias)` hands to tools/exl3_linear through torch's `__torch_function__` protocol
(any row count; the outputs in fp32, the bias added there -- bf16, as every tensor of the tower --
then bf16 as a linear layer's output; the fused qkv is q, k and v side by side). The rest of the
arithmetic is the 27B tower's, which is transformers' (`Qwen4ExpVisionModel` matches
`Qwen3_5VisionModel` step for step). The image rows and their three-axis rotary positions in the
language model are engine/models/flashnext/engine.py's (`FlashNextEngine._rot`).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from engine.vision import VisionConfig, VisionTower, load_vision_config

PREFIX = "visual."
LINEARS = ("attn.qkv", "attn.proj", "mlp.linear_fc1", "mlp.linear_fc2")


class Proj:
    """An EXL3 projection where the tower reads a weight: `F.linear(x, proj, bias)` is the blocks'
    outputs side by side (one block, or q, k and v for the fused qkv) plus the bias."""

    def __init__(self, blocks: list):
        self.blocks = blocks

    @property
    def N(self) -> int:
        return sum(b.N for b in self.blocks)

    @property
    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for b in self.blocks for t in (b.t, b.suh, b.svh))

    def to(self, *args, **kwargs) -> "Proj":
        return self                         # the blocks sit where the pack index put them

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if func is not F.linear:
            raise TypeError(f"an EXL3 projection takes F.linear only, not {getattr(func, '__name__', func)}")
        x, w, bias = (list(args) + [(kwargs or {}).get("bias")])[:3]
        from tools.exl3_linear import exl3_matmul
        x = x.contiguous()
        y = torch.cat([exl3_matmul(x, b) for b in w.blocks], dim=-1) if len(w.blocks) > 1 \
            else exl3_matmul(x, w.blocks[0])
        return (y + bias.float()).to(x.dtype)


class FlashNextVisionTower(VisionTower):
    """The tower over a pack: plain tensors and `Proj` weights in `self.t`, under the 27B's names."""

    @property
    def nbytes(self) -> int:
        return sum(v.nbytes if isinstance(v, Proj) else v.numel() * v.element_size() for v in self.t.values())

    def activation_bytes(self, n_patches: int) -> int:
        """engine/vision.py's bound plus the widest projection's fp32 output and its biased copy (fc1's
        4,352 padded columns; the merger runs on a quarter of the rows)."""
        n = max(v.N for k, v in self.t.items() if isinstance(v, Proj) and ".mlp." in k)
        return super().activation_bytes(n_patches) + int(n_patches) * 8 * n


def tower_of(cfg: VisionConfig, tensors: dict, blocks: dict, device="cuda") -> FlashNextVisionTower:
    """The tower from the pack's plain tensors (biases and all, by base name) and its EXL3 blocks (by
    projection: `blocks.{i}.attn.q_proj`, .., `merger.linear_fc2`): q, k and v as the fused qkv with
    their biases joined, every other projection as itself."""
    t = dict(tensors)
    for i in range(cfg.depth):
        p = f"blocks.{i}.attn."
        qkv = [p + n for n in ("q_proj", "k_proj", "v_proj")]
        t[p + "qkv.weight"] = Proj([blocks[n] for n in qkv])
        t[p + "qkv.bias"] = torch.cat([t.pop(n + ".bias") for n in qkv])
        for n in LINEARS[1:]:
            t[f"blocks.{i}.{n}.weight"] = Proj([blocks[f"blocks.{i}.{n}"]])
    for n in ("merger.linear_fc1", "merger.linear_fc2"):
        t[n + ".weight"] = Proj([blocks[n]])
    return FlashNextVisionTower(cfg, t, device=device)


def load_tower(idx, pack: str, device="cuda") -> FlashNextVisionTower | None:
    """The tower of a pack (`PackIndex` `idx`, its directory `pack`), or None for a pack without one (a
    vision config but no `model.visual.*` tensors in its files)."""
    cfg = load_vision_config(pack)
    if cfg is None or not idx.vision:
        return None
    blocks, tensors = {}, {}
    fused = tuple(f"blocks.{i}.attn.qkv." for i in range(cfg.depth))
    for n in idx.vision:
        base = n[len(PREFIX):]
        if base.endswith(".trellis"):
            b = base[: -len(".trellis")]
            blocks[b] = idx.exl3(PREFIX + b, device, out_fp32=True)
        elif not base.endswith((".suh", ".svh", ".mul1")) and not base.startswith(fused):
            tensors[base] = idx.read(n, "cpu")
    return tower_of(cfg, tensors, blocks, device=device)
