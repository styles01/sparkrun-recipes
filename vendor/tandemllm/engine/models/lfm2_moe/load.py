"""Load LFM2 MoE weights onto the device, in the forms the engine multiplies by (ENG-285).

  W[name]          the reference's canonical names (engine/models/lfm2_moe/ref.py): EXL3Blocks for
                   the pack's EXL3 projections, bf16 tensors for the rest (an fp16 copy the converter
                   stored is cast back to bf16: the norms, the conv taps, the embedding)
  W[fused]         what the engine launches once instead of two or three times:
                     layers.L.self_attn.qkv           q | k | v (one EXL3 group, or one bf16 weight)
                     layers.L.feed_forward.w13        w1 | w3 of a dense layer
  W["lm_head.weight"]  the pack's head, else the embedding (tied: the same tensor, no copy)
  experts[L]       the routed experts of MoE layer L: an `ExpertsEXL3` (each projection's trellis,
                   suh, svh, for the bank kernels) or an `ExpertsBF16` (stacked [E, I, H] / [E, H, I])

The router's `gate.weight` stays in its stored precision as bf16 and `expert_bias` in fp32 (the
reference adds it in fp32).
"""

from __future__ import annotations

import os
import time

import torch

from engine.models.lfm2_moe.pack import PackIndex, projections


class ExpertsBF16:
    """The experts of one layer, stacked: w1 and w3 [E, I, H] (in one tensor [E, 2 I, H], w1 first),
    w2 [E, H, I], bf16."""

    def __init__(self, w13: torch.Tensor, w2: torch.Tensor):
        self.w13, self.w2 = w13, w2
        self.E, self.I2, self.H = w13.shape

    @property
    def nbytes(self) -> int:
        return (self.w13.numel() + self.w2.numel()) * 2

    def expert(self, e: int):
        I = self.I2 // 2
        return self.w13[e, :I], self.w13[e, I:], self.w2[e]


class ExpertsEXL3:
    """The experts of one layer as EXL3Blocks [(w1, w3, w2)], for the MoE bank kernels and the CPU
    reference (`tools/exl3_linear.reference_matmul`)."""

    def __init__(self, blocks: list):
        self.blocks = blocks
        self.E = len(blocks)
        self._bank = None

    @property
    def nbytes(self) -> int:
        return sum(b.nbytes for e in self.blocks for b in e)

    def expert(self, e: int):
        return self.blocks[e]

    def bank(self, device):
        """The layer's experts as the bank kernels read them (tools/fn_kernels.MoEBank: gate, up,
        down of each expert; no shared expert)."""
        if self._bank is None:
            from tools.fn_kernels import MoEBank
            self._bank = MoEBank([tuple((b.t, b.suh[0], b.svh) for b in e) for e in self.blocks], device)
        return self._bank


def _plain(idx: PackIndex, name: str, device, dtype=torch.bfloat16) -> torch.Tensor:
    t = idx.read(name)
    return t.to(device=device, dtype=dtype if t.is_floating_point() else t.dtype).contiguous()


def _proj(idx: PackIndex, base: str, device, dtype, out_fp32: bool = False):
    if idx.formats.get(base) == "exl3":
        return idx.exl3(base, device, out_fp32=out_fp32)
    return _plain(idx, base + ".weight", device, dtype)


#: LFM2.5's own splits of tools/fn_kernels.dense for its EXL3 shapes (tools/fn_tiles.py --dense; a shape not in
#: the file takes the kernel's default split). Flash-Next's shapes are not among them, so its table is untouched.
DENSE_SPLITS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
                            "ops", "lfm25-dense-splits.json")


def apply_splits(path: str = DENSE_SPLITS) -> int:
    """The dense splits of the file into tools/fn_kernels.DENSE_SPLIT; returns the entries applied."""
    import json
    if not os.path.isfile(path):
        return 0
    from tools import fn_kernels as FK
    n = 0
    for k, v in json.load(open(path)).items():
        if k.startswith("_"):
            continue
        N, Kin, kb = (int(x) for x in k.split("x"))
        FK.DENSE_SPLIT[(N, Kin, kb)] = (int(v["S"]), int(v["warps"]))
        n += 1
    return n


def load(path: str, device="cuda", dtype=torch.bfloat16, log=print, idx: PackIndex | None = None):
    """(idx, W, experts, timing) of the checkpoint or pack at `path`."""
    from tools.exl3_linear import EXL3Block
    t0 = time.time()
    idx = idx or PackIndex.open(path)
    idx.audit()
    if idx.is_exl3:
        idx.check_mul1()
    c = idx.cfg
    timing = {"index_s": round(time.time() - t0, 2), "notes": list(idx.notes)}
    if idx.is_exl3 and str(device).startswith("cuda"):
        timing["dense_splits"] = apply_splits()
    W: dict = {}
    t1 = time.time()
    for base, _, _ in projections(c, experts=False):
        W[base + ".weight"] = _proj(idx, base, device, dtype)
    from engine.models.lfm2_moe.pack import plain_names
    for name, _ in plain_names(c):
        W[name] = _plain(idx, name, device, torch.float32 if name.endswith("expert_bias") else dtype)
    tied = idx.formats.get("lm_head") == "tied"
    W["lm_head.weight"] = W["embed_tokens.weight"] if tied else _proj(idx, "lm_head", device, dtype, out_fp32=True)
    timing["dense_s"] = round(time.time() - t1, 2)
    # fused launches
    for li in range(c.num_hidden_layers):
        p = f"layers.{li}."
        if c.is_attention(li):
            a = p + "self_attn."
            ws = [W[a + f"{x}_proj.weight"] for x in ("q", "k", "v")]
            if all(isinstance(w, EXL3Block) for w in ws) and len({w.kb for w in ws}) == 1:
                g = EXL3Block.group(ws, ["q", "k", "v"])
                W[a + "qkv"] = g
                for i, x in enumerate(("q", "k", "v")):
                    W[a + f"{x}_proj.weight"] = g.member(i)
            elif all(torch.is_tensor(w) for w in ws):
                W[a + "qkv"] = torch.cat(ws).contiguous()
        if not c.is_moe(li):
            f = p + "feed_forward."
            w1, w3 = W[f + "w1.weight"], W[f + "w3.weight"]
            if isinstance(w1, EXL3Block) and isinstance(w3, EXL3Block) and w1.kb == w3.kb:
                g = EXL3Block.group([w1, w3], ["w1", "w3"])
                W[f + "w13"] = g
                W[f + "w1.weight"], W[f + "w3.weight"] = g.member(0), g.member(1)
            elif torch.is_tensor(w1) and torch.is_tensor(w3):
                W[f + "w13"] = torch.cat([w1, w3]).contiguous()
    t1 = time.time()
    experts: dict = {}
    for li in c.moe_layers:
        f = f"layers.{li}.feed_forward.experts."
        if idx.formats.get(f + "0.w1") == "exl3":
            experts[li] = ExpertsEXL3([tuple(idx.exl3(f + f"{e}.{w}", device) for w in ("w1", "w3", "w2"))
                                       for e in range(c.num_experts)])
        else:
            I, H = c.moe_intermediate_size, c.hidden_size
            w13 = torch.empty(c.num_experts, 2 * I, H, dtype=dtype, device=device)
            w2 = torch.empty(c.num_experts, H, I, dtype=dtype, device=device)
            for e in range(c.num_experts):
                w13[e, :I].copy_(idx.read(f + f"{e}.w1.weight"))
                w13[e, I:].copy_(idx.read(f + f"{e}.w3.weight"))
                w2[e].copy_(idx.read(f + f"{e}.w2.weight"))
            experts[li] = ExpertsBF16(w13, w2)
        if log and li % 8 == 7:
            log(f"[lfm-load] experts through layer {li}: {time.time() - t1:.1f} s")
    timing["experts_s"] = round(time.time() - t1, 2)
    timing["total_s"] = round(time.time() - t0, 2)
    if log:
        log(f"[lfm-load] {path}: {timing}")
    return idx, W, experts, timing
