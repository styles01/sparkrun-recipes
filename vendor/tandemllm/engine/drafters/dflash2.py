"""DFlash2: a block drafter that proposes a whole masked block in one non-causal pass.

The module is `z-lab/Qwen3.8-27B-DFlash2`: five non-causal decoder layers of its own (hidden 5120,
32 query / 8 KV heads, head_dim 128, intermediate 17408, rope_theta 1e7, full 128-dim rotary), a
fusion projection `fc` over five of the target's hidden states, a dynamic depthwise convolution
wrapped around each sublayer, and a bigram candidate selector. It has no embedding table and no
head: the noise block is embedded with the *target's* `embed_tokens` and the draft hidden states are
turned into tokens by the *target's* `lm_head`.

WHICH TARGET HIDDEN STATES, AND FROM WHERE
------------------------------------------
`dflash_config.target_layer_ids = [5, 19, 33, 47, 61]`, and those ids are consumed by the serving
stack as *capture points at the entry of that layer*, not as that layer's output. The target model
marks the listed layers and each marked layer captures inside
`LayerCommunicator.prepare_attn_and_capture_last_layer_outputs`, whose body is:

    hidden_states, residual = self.prepare_attn(hidden_states, residual, ...)
    if captured_last_layer_outputs is not None:
        gathered_last_layer_output = self._communicate_simple_fn(hidden_states=residual, ...)
        ...
        captured_last_layer_outputs.append(gathered_last_layer_output)

`prepare_attn` is the fused add-norm: it folds the incoming `hidden_states` into `residual` and
writes `input_layernorm(residual)` into `hidden_states`. What is appended is **`residual`** -- the
residual stream *entering* layer L, i.e. the output of layer L-1 *after* its MLP residual add and
*before* layer L's `input_layernorm`. The name of the helper says it: "capture_last_layer_outputs".

So the five tensors this drafter needs are the residual stream entering target layers
5, 19, 33, 47 and 61, concatenated in that (ascending) order into [N, 5 * 5120] and fed to `fc`.

In this engine that is exactly what `Qwen38Engine.tap` already hands out. `forward()` calls
`self.tap(h[0].detach())` once with the embedding -- the residual stream entering layer 0 -- and
once at the end of every layer, after `h = res + self.mlp(x, p)` -- the residual stream entering the
next layer. Tap invocation number j is therefore "the residual stream entering layer j", and
invocations 5, 19, 33, 47, 61 are the five tensors, with no host-side change at all. See
`# REQUIRED HOST CHANGES` below for the one thing that is *not* required but would be tidier.

THE BLOCK
---------
`block_size = 8`. The block the draft module runs on is

    ids       = [anchor, MASK, MASK, MASK, MASK, MASK, MASK, MASK]      (mask id 248070)
    positions = [p,      p+1,  p+2,  p+3,  p+4,  p+5,  p+6,  p+7]

where `anchor` is the last token the target decided and `p` is its position -- the serving worker
builds precisely this (`block_ids.fill_(mask_token_id); block_ids[:, 0].copy_(bonus_tokens)`,
`positions = prefix_lens + [0..block_size-1]`).

Row j of the module's output predicts the token at position p+j, so **row 0 is dead** (its token is
the anchor, already known) and one block yields `block_size - 1 = 7` proposals, not 8:

    draft_hidden = draft_hidden.view(bs, block_size, -1)
    pred_hidden  = draft_hidden[:, 1:, :]            # [bs, block_size-1, H]
    ...
    draft_tokens[:, 0].copy_(block_ids[:, 0]); draft_tokens[:, 1:].copy_(draft_next)

That matches `engine/spec.py` one for one: the verify pass runs `[tok] + draft` with
`len(draft) == 7`, and an all-accept block yields 7 drafted tokens plus the bonus.

CONTEXT ENTERS AS KV, NOT AS TOKENS
-----------------------------------
`target_hidden` is projected once, `hidden_norm(fc(target_hidden))`, and that single tensor is what
every one of the five layers projects its context K and V from -- the HF reference passes the same
`target_hidden` into every layer, and the serving worker materialises it into the draft KV cache
once per committed token:

    ctx_hidden = self.draft_model.project_target_hidden(target_hidden)
    for layer in self.draft_model.layers:
        k, v = layer.self_attn.kv_proj_only(ctx_hidden)
        k = attn.apply_k_norm(k); k = attn.apply_k_rope(positions, k)

So the context costs `k_proj + v_proj` per layer per committed token, paid in `sync()`, and the
draft pass itself only ever projects Q from the eight noise rows. Q comes from the noise block, K
and V from context concatenated with the noise block, which is the KV injection the reference
names.

Attention is **non-causal**: `config.is_causal = false` maps to `AttentionType.ENCODER_ONLY`, so
inside the block every row sees every other row, including later ones -- that is what makes this a
block drafter rather than seven chained steps. All five layers are `sliding_attention` with
`sliding_window = 2048`, applied to the context side.

NORM CONVENTION -- THE THING THAT IS EASY TO GET WRONG
------------------------------------------------------
The target uses `GemmaRMSNorm`, `normalize(x) * (1 + w)`, which is what `engine.model.rms_norm`
implements. The draft checkpoint does **not**: it is a Qwen3 draft (`Qwen3RMSNorm` in the HF
reference, sglang's plain `RMSNorm` whose weight is initialised to `ones`) and computes
`normalize(x) * w`. Every norm in this file therefore uses the local `_rms`, never the engine's.
`tools/dflash2_probe.py` checks this empirically: the checkpoint's norm weights sit near 1.0, which
only makes sense for the plain convention.

COST OF ONE 8-WIDE DRAFT (7 proposals), bf16, exact
---------------------------------------------------
    5 layers x 332,974,336 params                     1,664,871,680
    fc  [5120, 25600]                                   131,072,000
    hidden_norm + norm                                       10,240
                                                     --------------
    backbone                                          1,795,953,920 params = 3,591,907,840 B
    selector hidden_projection [256, 5120]                            2,621,440 B
    selector codebook rows actually touched (gathered, not read whole):
        7 slots x (16 successor + 16 predecessor) rows x 256 x 2 B =      114,688 B
    target lm_head [248320, 5120] bf16, read ONCE for all 7 rows   2,542,796,800 B
                                                                   --------------
    total                                                           6,137,440,768 B = 6.14 GB

    -> 22.5 ms at the board's 273 GB/s, 32.3 ms at the 190 GB/s a real kernel reaches; 3.2 ms and
       4.6 ms per proposal. `tools/dflash2_probe.py` prints this table from the checkpoint itself.

The whole checkpoint is 1,924,404,480 params / 3.849 GB, of which 254.3 MB is the two codebooks --
resident, and read 2,217x more sparsely than their size suggests.

Against the prediction head: MTP-3 moves 9.1 GB for 3 proposals, this moves 6.14 GB
for 7. Against a 26.93 GB verify step it is 23 % of a step for up to 8 accepted tokens.

THE SELECTOR DOES NOT REPLACE THE HEAD READ
-------------------------------------------
It is an inference-time device -- the serving worker runs it on every draft, greedy and sampling
alike -- but it *consumes* the head's output instead of avoiding it. `compute_candidates` runs the
full head and then takes the top k:

    def _project_candidate_logits(hidden, lm_head, *, num_org, use_quant_head):
        weight = lm_head.weight
        return torch.matmul(hidden.to(weight.dtype), weight[:num_org].T)
    ...
    def _radix_topk(scores, k):
        # The selector's largest single cost: it reads the whole logits tensor.

and the codebooks are then *gathered* by candidate id, 32 rows of 256 per slot. So the whole
2,542,796,800 B of `lm_head` is read either way; the selector adds 2,621,440 B for
`hidden_projection` and 114,688 B of gathers, 0.04 % of the draft, and buys acceptance rather than
bytes: it rescores the 16 candidates per slot against the previous slot's choice through a rank-256
bigram factorisation and walks the resulting lattice.

The 33 ms / 6 ms difference is a *different* lever and it is still available: replacing the target's
head with a reduced-vocabulary draft head, the way `MTPDrafter` already can, takes the draft from
6.14 GB to 3.76 GB (32k fp8 rows = 167,772,160 B), 13.8 ms at 273 GB/s. The selector is unaffected -- it
indexes its codebooks by global token id, so a reduced head only has to map its rows back. That
path is wired here behind `draft_head=`, off by default until `tools/draft_head.py` exists.

WHAT IS NOT IN THE REFERENCE
----------------------------
`blocks=2` (two chained 8-wide drafts, 14 proposals) has no counterpart in the serving stack. The
worker runs exactly one draft forward per verify; the only supported way to propose more is a larger
`block_size` (`speculative_num_draft_tokens` overrides the checkpoint's 8 and `set_block_size`
re-blocks the convolutions). The chaining implemented below is therefore this engine's own, and its
shape is stated where it is built.
"""

# REQUIRED HOST CHANGES
# ---------------------
# None. `Qwen38Engine.tap` already delivers the five tensors this drafter needs, at the point it
# needs them, and `engine/spec.py` already calls `sync(tokens, hidden, first_pos)` after every
# verified block with `first_pos` equal to the position of `tokens[0]`. This drafter ignores
# `sync`'s `hidden` argument -- that one is the post-final-norm hidden, which is what the MTP head
# wants and not what DFlash2 wants -- and reads its own tap buffer instead.
#
# Two changes would be improvements rather than requirements, and neither is made here:
#
#   1. `eng.tap` is a single slot, so two drafters that both need it cannot be resident at once
#      (`engine/router.py` switches between drafters, and a future tap consumer would silently
#      steal this one). The patch, if that day comes:
#
#        --- a/engine/model.py
#        +++ b/engine/model.py
#        @@ class Qwen38Engine.__init__
#        -        self.tap = None  # set to a callable to receive every layer's hidden state
#        +        self.taps: list = []   # callables, each receives (index, hidden) per layer
#        +        self.tap = None        # kept: the single-consumer form, called first
#        @@ Qwen38Engine.forward
#        -        if self.tap is not None:
#        -            self.tap(h[0].detach())
#        +        self._emit_tap(0, h)
#        ...and at the end of the layer loop `self._emit_tap(layer + 1, h)`, with
#        +    def _emit_tap(self, index: int, h: torch.Tensor) -> None:
#        +        if self.tap is None and not self.taps:
#        +            return
#        +        row = h[0].detach()
#        +        if self.tap is not None:
#        +            self.tap(row)
#        +        for fn in self.taps:
#        +            fn(index, row)
#
#      Passing the index explicitly would also remove this file's need to count invocations.
#
#   2. `forward()` taps all 65 layers unconditionally once `tap` is set. Handing the engine the set
#      of wanted indices (`eng.tap_layers = {5, 19, 33, 47, 61}`) would let it skip the 60 calls
#      whose result is dropped. They are `detach()` views, so the cost today is 60 Python calls per
#      forward and no memory traffic -- not worth a host edit on its own.

from __future__ import annotations

import glob
import json
import os
from engine.settings import SETTINGS as _S  # noqa: E402  (every QWEN38_* knob)

import torch
import torch.nn.functional as F
from safetensors import safe_open

from . import Drafter, run_steps

DEFAULT_CKPT = os.path.expanduser(
    "~/.cache/huggingface/hub/models--z-lab--Qwen3.8-27B-DFlash2/snapshots")


# ----------------------------------------------------------------------------- checkpoint

def resolve_checkpoint(path: str | None = None) -> str:
    """Accept a snapshot directory, a `snapshots` directory, or nothing at all."""
    path = path or _S.get("DFLASH2") or DEFAULT_CKPT
    path = os.path.expanduser(path)
    if os.path.isfile(os.path.join(path, "config.json")):
        return path
    hits = sorted(glob.glob(os.path.join(path, "*", "config.json")))
    if not hits:
        raise FileNotFoundError(f"no config.json under {path}")
    return os.path.dirname(hits[0])


class DFlash2Config:
    """The draft module's own geometry. Nothing here is read from the target."""

    def __init__(self, raw: dict):
        d = raw["dflash_config"]
        self.hidden_size = int(raw["hidden_size"])
        self.intermediate_size = int(raw["intermediate_size"])
        self.num_hidden_layers = int(raw["num_hidden_layers"])
        self.num_attention_heads = int(raw["num_attention_heads"])
        self.num_key_value_heads = int(raw["num_key_value_heads"])
        self.head_dim = int(raw.get("head_dim", self.hidden_size // self.num_attention_heads))
        self.vocab_size = int(raw["vocab_size"])
        self.rms_norm_eps = float(raw["rms_norm_eps"])
        self.rope_theta = float(raw.get("rope_parameters", {}).get("rope_theta", 1e7))
        self.layer_types = list(raw.get("layer_types", ["full_attention"] * self.num_hidden_layers))
        self.sliding_window = int(raw.get("sliding_window") or 0) or None
        # `is_causal: false` -> AttentionType.ENCODER_ONLY in the serving stack.
        self.is_causal = bool(raw.get("is_causal", False))
        # z-lab's DFlash2 carries `block_size` inside `dflash_config`; the SpecForge-derived
        # checkpoints (DSpark) carry it at the top level, where their base class reads it.
        bs = d.get("block_size", raw.get("block_size"))
        if bs is None:
            raise KeyError("block_size is in neither dflash_config nor the top level")
        self.block_size = int(bs)
        self.conv_kernel_size = int(d.get("conv_kernel_size", 0))
        self.conv_group_size = int(d.get("conv_group_size", 0))
        self.mask_token_id = int(d["mask_token_id"])
        self.selector_rank = int(d.get("selector_rank", 0))
        self.selector_top_k = int(d.get("selector_top_k", 0))
        self.target_layer_ids = [int(x) for x in d["target_layer_ids"]]
        self.output_multiplier = float(d.get("output_multiplier", 1.0))
        cap = float(d.get("final_logit_softcapping") or 0.0)
        self.final_logit_softcapping = cap if cap > 0 else None

    # --- derived ---
    @property
    def num_groups(self) -> int:
        return self.hidden_size // self.conv_group_size

    @property
    def kv_dim(self) -> int:
        return self.num_key_value_heads * self.head_dim

    @property
    def q_dim(self) -> int:
        return self.num_attention_heads * self.head_dim

    def is_sliding(self, layer: int) -> bool:
        return self.layer_types[layer] == "sliding_attention"

    def expected_tensors(self) -> dict[str, tuple[int, ...]]:
        """Every tensor this drafter reads, with the shape it must have."""
        h, hd = self.hidden_size, self.head_dim
        out: dict[str, tuple[int, ...]] = {
            "fc.weight": (h, len(self.target_layer_ids) * h),
            "hidden_norm.weight": (h,),
            "norm.weight": (h,),
        }
        if self.selector_rank:
            out["candidate_selector.hidden_projection.weight"] = (self.selector_rank, h)
            out["candidate_selector.predecessor_codebook"] = (self.vocab_size, self.selector_rank)
            out["candidate_selector.successor_codebook"] = (self.vocab_size, self.selector_rank)
        for i in range(self.num_hidden_layers):
            p = f"layers.{i}"
            out[f"{p}.input_layernorm.weight"] = (h,)
            out[f"{p}.post_attention_layernorm.weight"] = (h,)
            out[f"{p}.self_attn.q_proj.weight"] = (self.q_dim, h)
            out[f"{p}.self_attn.k_proj.weight"] = (self.kv_dim, h)
            out[f"{p}.self_attn.v_proj.weight"] = (self.kv_dim, h)
            out[f"{p}.self_attn.o_proj.weight"] = (h, self.q_dim)
            out[f"{p}.self_attn.q_norm.weight"] = (hd,)
            out[f"{p}.self_attn.k_norm.weight"] = (hd,)
            out[f"{p}.mlp.gate_proj.weight"] = (self.intermediate_size, h)
            out[f"{p}.mlp.up_proj.weight"] = (self.intermediate_size, h)
            out[f"{p}.mlp.down_proj.weight"] = (h, self.intermediate_size)
            if self.conv_kernel_size:
                taps, g = self.conv_kernel_size, self.num_groups
                for conv in ("attention_conv", "mlp_conv"):
                    out[f"{p}.{conv}.base_kernel"] = (2, taps, h)
                    out[f"{p}.{conv}.kernel_projection.weight"] = (2 * taps * g, h)
        return out


def load_config(path: str | None = None) -> tuple[DFlash2Config, str]:
    snap = resolve_checkpoint(path)
    with open(os.path.join(snap, "config.json")) as f:
        raw = json.load(f)
    return DFlash2Config(raw), snap


_WEIGHTS: dict[tuple[str, str, torch.dtype], dict[str, torch.Tensor]] = {}

# The drafter's five layers are read in bf16 on every draft call -- about 3.1 GB
# of its 4.9 GB, the rest being the target's e4m3 head -- while the target's own projections are
# NVFP4. With this on, the attention and MLP projections of the drafter are quantised to NVFP4 at
# load (tools/quant_nvfp4.quantize_clipped, no activation weighting) and read through the same
# W4A16 kernel. A drafter only proposes, so the output cannot change; acceptance can, and is what
# decides it. Off by default.
DRAFT_NVFP4 = _S.get("DRAFT_NVFP4") == "1"
# The drafter's vocabulary head in NVFP4: 0.72 GB a block instead of the e4m3
# head's 1.27, read once a draft call. Lossless for the output by construction -- the drafter only
# proposes and the target's own head verifies -- so what it can cost is acceptance, which the row
# measures. One copy per engine, shared by both arms; read at call time so an A/B can flip it.
DRAFT_HEAD_NVFP4 = _S.get("DRAFT_HEAD_NVFP4") == "1"
# The context projection `fc` [5120, 25600] in NVFP4 as well: quantised
# the backbone's seven projections and left this one bf16, and it is read on every sync -- every
# block -- 262 MB for three or four committed rows. 74 MB in NVFP4. Drafter only, so lossless for the
# output; read at call time, quantised on first use.
DRAFT_FC_NVFP4 = _S.get("DRAFT_FC_NVFP4") == "1"
# The greedy walk read the device once per slot (`int(local[e, idx])`), the
# caller once more per token (`int(x)` over the walked ids), and `propose_tree` once more for the
# candidate table: about 33 device-to-host synchronisations a draft call where one will do. The
# walk is the same argmaxes, taken on the device, brought over in ONE copy with the candidates.
HOST_WALK = _S.get("HOST_WALK") == "1"
# The draft call served from a CUDA graph (engine/drafters/draft_graph.py):
# the context window gathered at device indices and masked past the committed context, the
# anchor, position and length on the device. Needs QWEN38_HOST_WALK (the walk after the replay).
DRAFT_GRAPH = _S.get("DRAFT_GRAPH") == "1"
_NVFP4_HEADS: dict = {}
# The drafter's vocabulary head from another EXL3 checkpoint (a lower-K trellis head, e.g. the 3.0
# quant's K4 head at 0.64 GB against the 4.5 target's K6 at 0.96): drafter only, so the output stays
# exact and what it can cost is acceptance. Unset = the target's own head, as before.
DRAFT_HEAD_EXL3 = _S.get("DRAFT_HEAD_EXL3") or ""
_EXL3_HEADS: dict = {}


def exl3_draft_head(eng):
    """The `lm_head` EXL3 block of DRAFT_HEAD_EXL3's checkpoint, read once an engine."""
    key = (id(eng), DRAFT_HEAD_EXL3)
    if key not in _EXL3_HEADS:
        from safetensors import safe_open
        from engine.loader import _exl3_files
        from tools.exl3_linear import read_block
        blk = None
        for fpath in _exl3_files(DRAFT_HEAD_EXL3):
            with safe_open(fpath, framework="pt", device=str(eng.device)) as f:
                bases = {k[: -len(".trellis")] for k in f.keys() if k.endswith(".trellis")}
                for base in bases:
                    if base.endswith("lm_head"):
                        blk = read_block(f, base, out_fp32=True)
        if blk is None:
            raise RuntimeError(f"no lm_head trellis in {DRAFT_HEAD_EXL3}")
        print(f"[drafter] head from {DRAFT_HEAD_EXL3}: K{blk.kb}, {blk.nbytes / 1e9:.2f} GB")
        _EXL3_HEADS[key] = blk
    return _EXL3_HEADS[key]


def nvfp4_head(eng):
    key = id(eng)
    if key not in _NVFP4_HEADS:
        from tools.head_gemv import head_to_nvfp4
        _NVFP4_HEADS[key] = head_to_nvfp4(eng.w.norm("lm_head.weight"))
    return _NVFP4_HEADS[key]


_PROJ = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
         "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
_QUANTISED: dict[str, dict] = {}


def quantise_projections(w: dict, n_layers: int, key: str | None = None) -> dict:
    """A copy of `w` with every layer's attention and MLP projection as an NVFP4 block."""
    if key is not None and key in _QUANTISED:
        return _QUANTISED[key]
    from tools.quant_nvfp4 import quantize_clipped
    out = dict(w)
    for i in range(n_layers):
        for proj in _PROJ:
            name = f"layers.{i}.{proj}.weight"
            out[name] = quantize_clipped(w[name].float(), None)
    if key is not None:
        _QUANTISED[key] = out
    return out


# The drafter's projections (and its context fc) as EXL3 blocks from a file written by
# tools/drafter_exl3_quant.py (ENG-215): drafter only, so the output stays exact and what the bits
# can cost is acceptance. Unset = the NVFP4 / bf16 projections as before (the fallback).
DRAFT_EXL3 = _S.get("DRAFT_EXL3") or ""
_EXL3_PROJ: dict = {}


def exl3_projections(w: dict, path: str, key: str) -> dict:
    """A copy of `w` with every projection of `path` as an EXL3 block (the fc as `fc.exl3`)."""
    if key in _EXL3_PROJ:
        return _EXL3_PROJ[key]
    from tools.exl3_linear import read_block
    out = dict(w)
    dev = w["fc.weight"].device
    n, nb = 0, 0
    with safe_open(path, framework="pt", device=str(dev)) as f:
        bases = sorted({k[: -len(".trellis")] for k in f.keys() if k.endswith(".trellis")})
        for base in bases:
            blk = read_block(f, base)
            target = "fc.exl3" if base == "fc" else f"{base}.weight"
            if base != "fc":
                old = w[target]
                assert tuple(blk.shape) == tuple(old.shape), (base, blk.shape, old.shape)
            out[target] = blk
            n += 1
            nb += blk.nbytes
    # q|k|v and gate|up as one launch each where the members share the rate (the members become
    # views of the group, which sum in the group's order alone too)
    from tools.exl3_linear import EXL3Block
    fused = 0
    for i in range(1 + max(int(k.split(".")[1]) for k in out if k.startswith("layers."))):
        for gname, members in ((f"layers.{i}.self_attn.qkv", ("self_attn.q_proj", "self_attn.k_proj",
                                                               "self_attn.v_proj")),
                               (f"layers.{i}.mlp.gate_up", ("mlp.gate_proj", "mlp.up_proj"))):
            blocks = [out.get(f"layers.{i}.{m}.weight") for m in members]
            if not all(isinstance(b, EXL3Block) for b in blocks) or len({b.kb for b in blocks}) != 1:
                continue
            grp = EXL3Block.group(blocks, list(members))
            out[gname] = grp
            for j, m in enumerate(members):
                out[f"layers.{i}.{m}.weight"] = grp.member(j)
            fused += 1
    print(f"[drafter] {n} EXL3 projections from {path}: {nb / 1e9:.3f} GB, {fused} fused groups")
    _EXL3_PROJ[key] = out
    return out


def _lin(x: torch.Tensor, w) -> torch.Tensor:
    """`F.linear` for a bf16 weight, the W4A16 kernel for an NVFP4 one, the trellis kernel for an
    EXL3 one."""
    if isinstance(w, torch.Tensor):
        return F.linear(x, w)
    if type(w).__name__ == "EXL3Block":
        return w.matmul(x.contiguous() if x.dim() == 2 else x)
    from tools.nvfp4_linear import nvfp4_matmul
    return nvfp4_matmul(x.reshape(-1, x.shape[-1]), w).view(*x.shape[:-1], w.N)


def load_weights(snapshot: str, device: str = "cuda",
                 dtype: torch.dtype = torch.bfloat16) -> dict[str, torch.Tensor]:
    """Read the single-file checkpoint straight onto `device`. Nothing is allocated at import.

    Cached by (snapshot, device, dtype): a bench that compares four configurations of this drafter
    builds four drafters, and four private copies of 3.85 GB would not fit beside a 27 B target.
    The tensors are read-only here, so sharing them is safe; what is per-drafter is the KV buffer
    and the tap state.
    """
    key = (snapshot, str(device), dtype)
    if key in _WEIGHTS:
        return _WEIGHTS[key]
    out: dict[str, torch.Tensor] = {}
    for path in sorted(glob.glob(os.path.join(snapshot, "*.safetensors"))):
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                t = f.get_tensor(key)
                if t.is_floating_point() and t.dtype != dtype:
                    t = t.to(dtype)
                out[key] = t.to(device) if device != "cpu" else t
    if not out:
        raise FileNotFoundError(f"no safetensors under {snapshot}")
    _WEIGHTS[key] = out
    return out


# ----------------------------------------------------------------------------- primitives

def _rms(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """Qwen3 RMS norm: `normalize(x) * w`. NOT the target's `(1 + w)` convention."""
    d = x.dtype
    y = x.float()
    y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + eps)
    return y.to(d) * w


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    a, b = x.chunk(2, dim=-1)
    return torch.cat([-b, a], dim=-1)


def _rope_tables(positions: torch.Tensor, dim: int, theta: float,
                 dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Neox-style full rotary over all `dim` of the head; the draft has no partial factor.

    Distinct from the target's rotary in this engine, which is partial at 0.25 over head_dim 256.
    `get_rope(head_dim, rotary_dim=head_dim, base=1e7, is_neox_style=True)` in the serving stack.
    """
    inv = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32,
                                        device=positions.device) / dim))
    f = positions.float()[:, None] * inv[None, :]
    emb = torch.cat([f, f], dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """`x` is [B, heads, T, head_dim]; `cos`/`sin` are [T, head_dim]."""
    return x * cos[None, None] + _rotate_half(x) * sin[None, None]


def _grouped_conv(h: torch.Tensor, delta: torch.Tensor, base: torch.Tensor,
                  num_groups: int, group_size: int, taps: int,
                  block_pos: torch.Tensor) -> torch.Tensor:
    """One dynamic depthwise K-tap convolution along the token axis, inside a block.

    Port of `_grouped_conv` in the serving model, operation for operation:

        blocks = hidden_states.unflatten(-1, (num_groups, group_size))
        coefficients = base.view(1, taps, num_groups, group_size) + delta.unsqueeze(-1)
        out = coefficients[:, 0] * blocks
        position = arange(T) & (block_size - 1)
        for tap in range(1, taps):
            shifted = F.pad(blocks[:-tap], (0, 0, 0, 0, tap, 0))
            out = out + coefficients[:, tap] * shifted * (position >= tap).view(-1, 1, 1)
        return out.flatten(-2)

    Three details that are the whole thing:

      * grouping -- `unflatten(-1, (num_groups, group_size))` splits the 5120 channels into 320
        contiguous runs of 16, so channel c is in group c // 16. `base_kernel` is per *channel*
        [2, taps, 5120]; the projected `delta` is per *group* [T, taps, 320] and is broadcast
        across the 16 channels of its group. `kernel_projection` emits 2 * taps * num_groups =
        1280 features laid out [side][tap][group], which is the reshape order used here.
      * direction -- `F.pad(blocks[:-tap], (..., tap, 0))` pads the *front* of the token axis, so
        `shifted[t] = blocks[t - tap]`. The convolution looks backwards: it is causal.
      * blocking -- `position` is the index *within* the block, and the `position >= tap` mask kills
        every tap that would reach across a block boundary. A block never sees the one before it.

    The coefficient used at tap t belongs to the *current* token: the kernel is dynamic per output
    row, not a shared filter.
    """
    blocks = h.unflatten(-1, (num_groups, group_size))
    coef = base.view(1, taps, num_groups, group_size) + delta.unsqueeze(-1)
    out = coef[:, 0] * blocks
    for tap in range(1, taps):
        shifted = F.pad(blocks[:-tap], (0, 0, 0, 0, tap, 0))
        out = out + coef[:, tap] * shifted * (block_pos >= tap).view(-1, 1, 1)
    return out.flatten(-2)


class _Conv:
    """`prepare` convolves a sublayer's input and hands back the kernel `finish` applies to its
    output -- both halves from one projection of the input, which is why `kernel_projection` emits
    two sides."""

    __slots__ = ("base", "kp", "taps", "groups", "gsize")

    def __init__(self, base: torch.Tensor, kp: torch.Tensor, taps: int, groups: int, gsize: int):
        self.base, self.kp = base, kp
        self.taps, self.groups, self.gsize = taps, groups, gsize

    def prepare(self, h: torch.Tensor, block_pos: torch.Tensor):
        coef = F.linear(h, self.kp).reshape(h.shape[0], 2, self.taps, self.groups)
        return (_grouped_conv(h, coef[:, 0], self.base[0], self.groups, self.gsize,
                              self.taps, block_pos),
                coef[:, 1])

    def finish(self, h: torch.Tensor, coef: torch.Tensor, block_pos: torch.Tensor):
        return _grouped_conv(h, coef, self.base[1], self.groups, self.gsize, self.taps, block_pos)


# ----------------------------------------------------------------------------- the module

class DFlash2Module:
    """The draft module itself. Knows nothing about the engine, so the probe can drive it."""

    def __init__(self, cfg: DFlash2Config, w: dict[str, torch.Tensor]):
        self.cfg = cfg
        self.w = w
        self.convs: list[tuple[_Conv, _Conv] | None] = []
        for i in range(cfg.num_hidden_layers):
            if not cfg.conv_kernel_size:
                self.convs.append(None)
                continue
            p = f"layers.{i}"
            self.convs.append((
                _Conv(w[f"{p}.attention_conv.base_kernel"],
                      w[f"{p}.attention_conv.kernel_projection.weight"],
                      cfg.conv_kernel_size, cfg.num_groups, cfg.conv_group_size),
                _Conv(w[f"{p}.mlp_conv.base_kernel"],
                      w[f"{p}.mlp_conv.kernel_projection.weight"],
                      cfg.conv_kernel_size, cfg.num_groups, cfg.conv_group_size),
            ))

    def rope(self, positions: torch.Tensor, dim: int,
             dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        """The rotary this checkpoint was trained with. Plain NTK-free RoPE here; a subclass whose
        config asks for a scaled rotary overrides it, and getting this wrong is silent -- the
        drafter still returns tokens, they are just the wrong ones."""
        return _rope_tables(positions, dim, self.cfg.rope_theta, dtype)

    @property
    def device(self) -> torch.device:
        return self.w["fc.weight"].device

    @property
    def dtype(self) -> torch.dtype:
        return self.w["fc.weight"].dtype

    # ---- context ---------------------------------------------------------------
    def project_context(self, target_hidden: torch.Tensor) -> torch.Tensor:
        """`hidden_norm(fc(concat of the five captured hidden states))`, [N, 5*H] -> [N, H]."""
        cfg = self.cfg
        expected = len(cfg.target_layer_ids) * cfg.hidden_size
        if target_hidden.ndim != 2 or target_hidden.shape[-1] != expected:
            raise ValueError(f"target_hidden must be [N, {expected}], got "
                             f"{tuple(target_hidden.shape)}")
        if "fc.exl3" in self.w:
            return _rms(_lin(target_hidden, self.w["fc.exl3"]), self.w["hidden_norm.weight"],
                        cfg.rms_norm_eps)
        if DRAFT_FC_NVFP4:
            fc = self.w.get("fc.nvfp4")
            if fc is None:
                from tools.quant_nvfp4 import quantize_clipped
                fc = self.w["fc.nvfp4"] = quantize_clipped(self.w["fc.weight"].float(), None)
            return _rms(_lin(target_hidden, fc), self.w["hidden_norm.weight"], cfg.rms_norm_eps)
        return _rms(F.linear(target_hidden, self.w["fc.weight"]),
                    self.w["hidden_norm.weight"], cfg.rms_norm_eps)

    def context_kv(self, ctx_hidden: torch.Tensor,
                   positions: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Per layer, the K and V the context contributes: `k_proj`/`v_proj` of the *same*
        projected context hidden, K through `k_norm` and then RoPE at its absolute position.

        Returns [n_layers] of (k, v), each [n_kv_heads, N, head_dim].
        """
        cfg = self.cfg
        n, hd, nkv = ctx_hidden.shape[0], cfg.head_dim, cfg.num_key_value_heads
        cos, sin = self.rope(positions, hd, ctx_hidden.dtype)
        out = []
        for i in range(cfg.num_hidden_layers):
            p = f"layers.{i}.self_attn"
            k = _lin(ctx_hidden, self.w[f"{p}.k_proj.weight"]).view(n, nkv, hd)
            k = _rms(k, self.w[f"{p}.k_norm.weight"], cfg.rms_norm_eps)
            v = _lin(ctx_hidden, self.w[f"{p}.v_proj.weight"]).view(n, nkv, hd)
            k = _apply_rope(k.transpose(0, 1)[None], cos, sin)[0]
            out.append((k, v.transpose(0, 1)))
        return out

    # ---- the block pass --------------------------------------------------------
    def forward_block(self, noise_emb: torch.Tensor, positions: torch.Tensor,
                      ctx_kv: list[tuple[torch.Tensor, torch.Tensor]] | None,
                      ctx_positions: torch.Tensor | None,
                      block_size: int | None = None, return_kv: bool = False,
                      masks: tuple[torch.Tensor | None, torch.Tensor | None] | None = None):
        """One pass of the five layers over `noise_emb` [T, H]. Returns `norm(h)`, [T, H].

        `ctx_kv[i]` is (k, v) with k already normed and RoPE'd -- the draft KV cache. `positions`
        are the block's absolute positions; `ctx_positions` are the context's, used only to build
        the sliding-window mask.

        With `return_kv`, also returns the block's own per-layer (k, v) -- normed, RoPE'd, shaped
        [n_kv_heads, T, head_dim] -- so a chained second block can attend to this one.

        `masks` overrides the pair `_masks` would build, as (full-attention, sliding). Serving never
        passes it: one block at a time needs nothing the default does not do. Training does, because
        it packs many blocks of the same sequence into one pass and those blocks must not see each
        other -- the pass is non-causal inside a block by design, and that is exactly what would
        leak between two blocks sharing a tensor.
        """
        cfg = self.cfg
        bs = block_size or cfg.block_size
        t = noise_emb.shape[0]
        hd, nh, nkv = cfg.head_dim, cfg.num_attention_heads, cfg.num_key_value_heads
        rep = nh // nkv
        dev = noise_emb.device
        block_pos = torch.arange(t, device=dev) % bs
        cos, sin = self.rope(positions, hd, noise_emb.dtype)
        if masks is None:
            masks = self._masks(positions, ctx_positions, t)

        h = noise_emb
        block_kv: list[tuple[torch.Tensor, torch.Tensor]] = []
        for i in range(cfg.num_hidden_layers):
            p = f"layers.{i}"
            conv = self.convs[i]
            res = h
            x = _rms(h, self.w[f"{p}.input_layernorm.weight"], cfg.rms_norm_eps)
            akernel = None
            if conv is not None:
                x, akernel = conv[0].prepare(x, block_pos)

            qkv = self.w.get(f"{p}.self_attn.qkv")
            if qkv is not None:
                qy, ky, vy = _lin(x, qkv).split(qkv.sizes, dim=-1)
                q, k, v = qy.reshape(t, nh, hd), ky.reshape(t, nkv, hd), vy.reshape(t, nkv, hd)
            else:
                q = _lin(x, self.w[f"{p}.self_attn.q_proj.weight"]).view(t, nh, hd)
                k = _lin(x, self.w[f"{p}.self_attn.k_proj.weight"]).view(t, nkv, hd)
                v = _lin(x, self.w[f"{p}.self_attn.v_proj.weight"]).view(t, nkv, hd)
            q = _rms(q, self.w[f"{p}.self_attn.q_norm.weight"], cfg.rms_norm_eps)
            k = _rms(k, self.w[f"{p}.self_attn.k_norm.weight"], cfg.rms_norm_eps)
            q = _apply_rope(q.transpose(0, 1)[None], cos, sin)
            k = _apply_rope(k.transpose(0, 1)[None], cos, sin)
            v = v.transpose(0, 1)[None]
            if return_kv:
                block_kv.append((k[0], v[0]))
            if ctx_kv is not None:
                ck, cv = ctx_kv[i]
                k = torch.cat([ck[None], k], dim=2)
                v = torch.cat([cv[None], v], dim=2)
            o = F.scaled_dot_product_attention(q, k.repeat_interleave(rep, dim=1),
                                               v.repeat_interleave(rep, dim=1),
                                               attn_mask=masks[1] if cfg.is_sliding(i) else masks[0])
            o = o.transpose(1, 2).reshape(t, -1)
            o = _lin(o, self.w[f"{p}.self_attn.o_proj.weight"])
            if conv is not None:
                o = conv[0].finish(o, akernel, block_pos)
            h = res + o

            res = h
            x = _rms(h, self.w[f"{p}.post_attention_layernorm.weight"], cfg.rms_norm_eps)
            mkernel = None
            if conv is not None:
                x, mkernel = conv[1].prepare(x, block_pos)
            gu = self.w.get(f"{p}.mlp.gate_up")
            if gu is not None:
                g, u = _lin(x, gu).split(gu.sizes, dim=-1)
            else:
                g = _lin(x, self.w[f"{p}.mlp.gate_proj.weight"])
                u = _lin(x, self.w[f"{p}.mlp.up_proj.weight"])
            x = _lin(F.silu(g) * u, self.w[f"{p}.mlp.down_proj.weight"])
            if conv is not None:
                x = conv[1].finish(x, mkernel, block_pos)
            h = res + x

        out = _rms(h, self.w["norm.weight"], cfg.rms_norm_eps)
        return (out, block_kv) if return_kv else out

    def _forward_block_steps(self, noise_emb: torch.Tensor, positions: torch.Tensor,
                             ctx_kv: list[tuple[torch.Tensor, torch.Tensor]] | None,
                             ctx_positions: torch.Tensor | None, masks=None):
        """`forward_block` (no `return_kv`, the checkpoint's block size) as a generator: each
        NVFP4 projection is handed out as `(weight key, rows)` so `forward_block_multi` can run
        one call for several slots' blocks (the packed draft of parallel requests); a bf16 weight runs here, per
        block. Every other statement is `forward_block`'s, in its order."""
        cfg = self.cfg
        bs = cfg.block_size
        t = noise_emb.shape[0]
        hd, nh, nkv = cfg.head_dim, cfg.num_attention_heads, cfg.num_key_value_heads
        rep = nh // nkv
        dev = noise_emb.device
        block_pos = torch.arange(t, device=dev) % bs
        cos, sin = self.rope(positions, hd, noise_emb.dtype)
        if masks is None:
            masks = self._masks(positions, ctx_positions, t)

        def lin(x, key):
            w = self.w[key]
            if isinstance(w, torch.Tensor):
                return _lin(x, w)
            y = yield key, x
            return y

        h = noise_emb
        for i in range(cfg.num_hidden_layers):
            p = f"layers.{i}"
            conv = self.convs[i]
            res = h
            x = _rms(h, self.w[f"{p}.input_layernorm.weight"], cfg.rms_norm_eps)
            akernel = None
            if conv is not None:
                x, akernel = conv[0].prepare(x, block_pos)
            q = (yield from lin(x, f"{p}.self_attn.q_proj.weight")).view(t, nh, hd)
            q = _rms(q, self.w[f"{p}.self_attn.q_norm.weight"], cfg.rms_norm_eps)
            k = (yield from lin(x, f"{p}.self_attn.k_proj.weight")).view(t, nkv, hd)
            k = _rms(k, self.w[f"{p}.self_attn.k_norm.weight"], cfg.rms_norm_eps)
            v = (yield from lin(x, f"{p}.self_attn.v_proj.weight")).view(t, nkv, hd)
            q = _apply_rope(q.transpose(0, 1)[None], cos, sin)
            k = _apply_rope(k.transpose(0, 1)[None], cos, sin)
            v = v.transpose(0, 1)[None]
            if ctx_kv is not None:
                ck, cv = ctx_kv[i]
                k = torch.cat([ck[None], k], dim=2)
                v = torch.cat([cv[None], v], dim=2)
            o = F.scaled_dot_product_attention(q, k.repeat_interleave(rep, dim=1),
                                               v.repeat_interleave(rep, dim=1),
                                               attn_mask=masks[1] if cfg.is_sliding(i) else masks[0])
            o = o.transpose(1, 2).reshape(t, -1)
            o = yield from lin(o, f"{p}.self_attn.o_proj.weight")
            if conv is not None:
                o = conv[0].finish(o, akernel, block_pos)
            h = res + o

            res = h
            x = _rms(h, self.w[f"{p}.post_attention_layernorm.weight"], cfg.rms_norm_eps)
            mkernel = None
            if conv is not None:
                x, mkernel = conv[1].prepare(x, block_pos)
            g = yield from lin(x, f"{p}.mlp.gate_proj.weight")
            u = yield from lin(x, f"{p}.mlp.up_proj.weight")
            x = yield from lin(F.silu(g) * u, f"{p}.mlp.down_proj.weight")
            if conv is not None:
                x = conv[1].finish(x, mkernel, block_pos)
            h = res + x
        return _rms(h, self.w["norm.weight"], cfg.rms_norm_eps)

    def _masks(self, positions: torch.Tensor, ctx_positions: torch.Tensor | None,
               t: int) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """(full-attention mask, sliding mask). `None` means "attend to everything".

        Non-causal: nothing here masks a later position out, because `is_causal` is false and every
        row of the block is supposed to see every other row. The only thing masked is the context
        beyond the 2048-wide window of the five `sliding_attention` layers.
        """
        cfg = self.cfg
        if ctx_positions is None or ctx_positions.numel() == 0:
            return None, None
        if cfg.sliding_window is None:
            return None, None
        allpos = torch.cat([ctx_positions, positions])
        delta = positions[:, None] - allpos[None, :]
        # A key at distance >= window behind the query is out of the window; the block's own rows
        # are at distance <= block_size and always inside it.
        return None, delta < cfg.sliding_window

    # ---- candidate selection ---------------------------------------------------
    def unary_candidates(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Top-k over the head's output. The head read has already happened by this point --
        `logits` is [P, V] and reading it whole is, in the reference's own words, "the selector's
        largest single cost"."""
        k = self.cfg.selector_top_k
        vals, ids = torch.topk(logits.float(), k, dim=-1)
        if self.cfg.output_multiplier != 1.0:
            vals = vals * self.cfg.output_multiplier
        if self.cfg.final_logit_softcapping is not None:
            cap = self.cfg.final_logit_softcapping
            vals = torch.tanh(vals / cap) * cap
        return ids.long(), vals

    def lattice(self, pred_hidden: torch.Tensor, candidate_ids: torch.Tensor,
                unary: torch.Tensor, anchor_id: int) -> torch.Tensor:
        """Score[e, p, c] = unary[e, c] + <A[pred[e, p]] * project(h[e]), B[c]>

        `pred[0, :]` is the verified anchor repeated across the k predecessor slots; `pred[e, :]`
        for e > 0 is the previous slot's candidate list. A is `predecessor_codebook`, B is
        `successor_codebook`, both [vocab, 256], both *gathered* by id -- 32 rows per slot, never
        read whole.
        """
        w, k = self.w, self.cfg.selector_top_k
        hidden = F.linear(pred_hidden, w["candidate_selector.hidden_projection.weight"])
        keys = w["candidate_selector.successor_codebook"][candidate_ids]            # [L, k, r]
        if torch.is_tensor(anchor_id):
            # a device scalar (graph): no read back to the host
            anchor = anchor_id.reshape(1, 1).expand(1, k)
        else:
            anchor = torch.full((1, k), int(anchor_id), dtype=torch.long,
                                device=candidate_ids.device)
        pred_ids = torch.cat([anchor, candidate_ids[:-1]], dim=0)                   # [L, k]
        preds = w["candidate_selector.predecessor_codebook"][pred_ids]              # [L, k, r]
        pair = (preds.float() * hidden.float()[:, None, :])
        return unary[:, None, :] + torch.einsum("lpr,lcr->lpc", pair, keys.float())

    @staticmethod
    def walk(candidate_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """The greedy path through the lattice -- `sample_path` with `greedy_mask` all true.

        Slot 0 takes `scores[0, 0].argmax()` (every predecessor row of slot 0 is the same anchor),
        and each later slot takes the argmax of the row selected by the previous slot's index. A
        chain walk, not a full Viterbi: the reference does exactly this.
        """
        idx = int(scores[0, 0].argmax())
        path = [idx]
        local = scores[1:].argmax(dim=-1)            # [L-1, k]
        for e in range(local.shape[0]):
            idx = int(local[e, idx])
            path.append(idx)
        sel = torch.tensor(path, dtype=torch.long, device=candidate_ids.device)
        return candidate_ids.gather(-1, sel[:, None])[:, 0]

    @staticmethod
    def walk_host(candidate_ids: torch.Tensor, scores: torch.Tensor) -> tuple[list[int], list]:
        """`walk`, with every argmax taken on the device and brought over in one copy together
        with the candidate table. Returns (token ids, candidates [L][k]) as Python lists."""
        L, k = candidate_ids.shape
        return DFlash2Module._walk_blob(DFlash2Module._blob(candidate_ids, scores).tolist(), L, k)

    @staticmethod
    def walk_host_logp(candidate_ids: torch.Tensor, scores: torch.Tensor,
                       logp: torch.Tensor) -> tuple[list[int], list, list]:
        """`walk_host` plus the lattice's log-probabilities, in ONE synchronisation: both
        are copied into pinned memory behind the draft and the host waits once, where it used to
        read the walk and then launch the log-softmax and read again. The same numbers."""
        L, k = candidate_ids.shape
        blob = DFlash2Module._blob(candidate_ids, scores)
        if not blob.is_cuda:
            return DFlash2Module._walk_blob(blob.tolist(), L, k) + (logp.tolist(),)
        bh = torch.empty(blob.shape, dtype=blob.dtype, pin_memory=True)
        lh = torch.empty(logp.shape, dtype=logp.dtype, pin_memory=True)
        bh.copy_(blob, non_blocking=True)
        lh.copy_(logp, non_blocking=True)
        torch.cuda.current_stream().synchronize()
        return DFlash2Module._walk_blob(bh.tolist(), L, k) + (lh.tolist(),)

    @staticmethod
    def _blob(candidate_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        return torch.cat([scores[0, 0].argmax().view(1), scores[1:].argmax(dim=-1).reshape(-1),
                          candidate_ids.reshape(-1).long()])

    @staticmethod
    def _walk_blob(blob: list[int], L: int, k: int) -> tuple[list[int], list]:
        local = blob[1:1 + (L - 1) * k]
        cand = [blob[1 + (L - 1) * k + l * k: 1 + (L - 1) * k + (l + 1) * k] for l in range(L)]
        idx = blob[0]
        toks = [cand[0][idx]]
        for e in range(L - 1):
            idx = local[e * k + idx]
            toks.append(cand[e + 1][idx])
        return toks, cand

    @staticmethod
    def viterbi(candidate_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """The *maximising* path through the same lattice.

        `walk` above is what the reference does and it is not the argmax of the selector's own
        objective. That objective is a first-order chain,

            score(t_0..t_{L-1}) = sum_l [ unary(l, t_l) + pair(t_{l-1}, t_l) ]

        and `scores[l, p, c]` already holds `unary(l, c) + pair(candidate p of slot l-1, c)`, so
        the maximiser is Viterbi over L slots and k candidates: L x k x k = 7 x 16 x 16 = 1,792
        additions over numbers that are already in registers. The pairwise term was computed for
        every (p, c) pair whether or not the greedy walk looked at it, so this costs no extra
        memory traffic at all -- it reads a tensor the greedy walk also builds and then throws
        away 15/16 of.

        It cannot change what the engine outputs. A different draft is still verified token by
        token against the target's own argmax; a better draft is only accepted further.
        """
        dp = scores[0, 0].clone()                     # [k]  slot 0: every predecessor is the anchor
        back: list[torch.Tensor] = []
        for l in range(1, scores.shape[0]):
            total = dp[:, None] + scores[l]           # [k_pred, k_cand]
            best, arg = total.max(dim=0)              # over the predecessor
            dp = best
            back.append(arg)
        idx = int(dp.argmax())
        path = [idx]
        for arg in reversed(back):
            idx = int(arg[idx])
            path.insert(0, idx)
        sel = torch.tensor(path, dtype=torch.long, device=candidate_ids.device)
        return candidate_ids.gather(-1, sel[:, None])[:, 0]


# ----------------------------------------------------------------------------- the drafter

def forward_block_multi(module: "DFlash2Module", items: list) -> list:
    """The packed draft of parallel requests (server/slots.py): `forward_block` for several slots' blocks over ONE module, each NVFP4
    projection one call over every block's rows (rows are independent of their neighbours on the
    served skinny kernel up to 32 rows: docs/kernels.md); attention, the convolutions and the
    norms per block, as alone. `items` = [(noise_emb, positions, ctx_kv, ctx_positions, masks)].
    Returns each block's `norm(h)`."""
    gens = [module._forward_block_steps(*it) for it in items]
    reqs: list = [None] * len(gens)
    done: list = [None] * len(gens)
    live = []
    for j, g in enumerate(gens):
        try:
            reqs[j] = next(g)
            live.append(j)
        except StopIteration as fin:
            done[j] = fin.value
    while live:
        groups: dict = {}
        for j in live:
            groups.setdefault(reqs[j][0], []).append(j)
        answers: dict = {}
        for key, js in groups.items():
            xs = [reqs[j][1] for j in js]
            y = _lin(torch.cat(xs, dim=0) if len(xs) > 1 else xs[0], module.w[key])
            for j, part in zip(js, y.split([x.shape[0] for x in xs], dim=0)):
                answers[j] = part
        nxt = []
        for j in live:
            try:
                reqs[j] = gens[j].send(answers[j])
                nxt.append(j)
            except StopIteration as fin:
                done[j] = fin.value
        live = nxt
    return done


#: The server's hook for the packed draft -- a callable (drafter, anchor, pos0) -> the draft
#: graph's outputs, or None to draft alone. None outside a server with side slots (`--parallel N`).
DRAFT_MEET = None


class DFlash2Drafter(Drafter):
    """Wires `DFlash2Module` to `Qwen38Engine` through the tap and the drafter interface."""

    name = "dflash2"
    wants_rows = True          # engine/spec.py hands it the accepted rows of the tap after a tree

    def requires(self) -> dict:
        """The taps (hidden states of these target layers, this width), the target's
        embedding (the draft block's noise rows) and its head (the draft logits)."""
        return {"hidden_size": self.cfg.hidden_size, "tap_layers": list(self.cfg.target_layer_ids),
                "tensors": ("embed_tokens.weight", "lm_head.weight")}

    def __init__(self, eng, ckpt: str | None = None, *, blocks: int = 1,
                 selector: bool = True, draft_head: str | None = None,
                 max_len: int | None = None, path: str = "greedy", tap: str = "entry",
                 block: int | None = None):
        """`block` overrides the checkpoint's block length.

        The module is non-causal inside its block and its rotary is absolute, so nothing in the
        architecture ties it to eight: a block of sixteen is sixteen rows, fifteen of them masked,
        at positions p..p+15. What ties it to eight is TRAINING -- the released weights were trained
        at eight and their top-1 falls off past slot 7 -- so a longer block is only interesting with
        a drafter fine-tuned at that length, or with a verify that reads more than the top-1 of each
        slot. `tools/train_dflash2.py --block` trains one.
        """
        self.eng = eng
        # Which hidden state `target_layer_ids` names. The header argues for the residual stream
        # ENTERING the listed layer, from the serving stack's capture point. The alternative
        # reading -- the listed layer's OUTPUT -- is one index along, and it is the one thing the
        # CPU probe cannot settle, because the probe feeds synthetic hidden states. So it is a
        # flag, and acceptance decides it.
        if tap not in ("entry", "output"):
            raise ValueError(f"tap must be entry or output, not {tap!r}")
        self.tap_mode = tap
        if path not in ("greedy", "viterbi"):
            raise ValueError(f"path must be greedy or viterbi, not {path!r}")
        self.path = path
        self._lattice: tuple[torch.Tensor, torch.Tensor] | None = None
        # The selector's scores are a head logit plus a bilinear term, not a calibrated
        # distribution. A softmax of them is the cheapest thing that is monotone in the right
        # direction; the temperature is the one knob that says how much to believe it, and the
        # simulator turns it against what the target really wrote.
        self.tree_temp = float(_S.get("DF2_TEMP"))
        # "nodes" spends the budget best-first over marginals (DDTree); "paths" spends it on whole
        # branches. See `engine/tree.py` -- an alternative node only pays if it has descendants.
        #
        # STAYS "paths" in the release candidate, and the honest reason is that nothing has
        # measured the difference yet.
        #
        # Track B at 13:34 had "nodes" ahead by 3 % on the five-workload mean. Re-measured under
        # the v2 kernel and the latch (phase 9, 02:38) that collapsed to 0.2 % -- ahead on all five
        # and behind on none, which looked like enough. The row then read 57.99 / 27.59 against the
        # previous config's 53.74 / 30.42, which looked like a median regression, and it is NOT:
        # re-running the previous config unchanged gave 59.06 / 28.39. **Two runs of the same
        # engine differ by 10 % on the mean and 7 % on the median**, because the latch's own
        # decisions are made from timing measurements and the row is fifty requests of an adaptive
        # policy, not fifty of a fixed one.
        #
        # So the row cannot resolve 0.2 %, and neither number above convicts or clears this flag.
        # It stays at the value the release candidate was soaked on, and it goes to the `speed`
        # branch to be measured properly -- several rows of each arm, and the spread reported.
        self.tree_mode = _S.get("DF2_TREE_MODE")
        self.cfg, self.snapshot = load_config(ckpt)
        if block:
            self.cfg.block_size = int(block)
        if self.cfg.hidden_size != eng.cfg.hidden_size:
            raise ValueError(f"draft hidden {self.cfg.hidden_size} != target "
                             f"{eng.cfg.hidden_size}")
        for lid in self.cfg.target_layer_ids:
            if lid >= eng.cfg.num_hidden_layers:
                raise ValueError(f"target_layer_ids has {lid}, target has "
                                 f"{eng.cfg.num_hidden_layers} layers")
        self.blocks = int(blocks)
        self.use_selector = bool(selector) and bool(self.cfg.selector_rank)
        self.max_len = max_len or eng.max_len
        from engine.drafters import check_target
        check_target(self, eng)                            # the rest of what it reads
        self.module: DFlash2Module | None = None          # built on first use, never at import
        self._w: dict[str, torch.Tensor] | None = None

        # Optional reduced-vocabulary head, exactly as MTPDrafter takes one. The selector indexes
        # its codebooks by global id, so the head's row->id map is applied before the lattice.
        self.head = None
        self.head_index = None
        draft_head = draft_head if draft_head is not None else _S.get("DRAFT_HEAD")
        self._draft_head_path = draft_head

        # The draft's own KV cache, one entry per committed target position.
        self._ck: torch.Tensor | None = None
        self._cv: torch.Tensor | None = None
        self.ctx_len = 0

        # Sampled drafting. The serving loop calls `drafter.sampler = sampler` when the
        # request samples: proposals are then drawn from this head's own distribution under the
        # request's profile instead of the greedy selector walk, and `last_q` carries one
        # distribution row per proposed token for the verify's q-aware accept. None on the greedy
        # path, where nothing changes.
        self.sampler = None
        self.last_q: list[torch.Tensor] | None = None
        # `propose_tree` on a sampled request asks `_tokens_from` for the lattice beside the
        # sample, and leaves the deterministic tree the same lattice builds for the router to price
        self._want_lattice = False
        self.last_det_tree = None
        # Draft temperature for sampled drafting: the draft's proposal distribution is
        # `softmax(logits / (draft_temp * request_temp))` because the request's profile is applied
        # on top. A cooler draft is SHARPER, and since the accept is `min(1, p(d)/q(d))`, a sharp
        # q makes acceptance approach `p(d)` -- the target's own mass on the draft token. Measured
        # 2026-09-19: at the request temperature the draft's spread over wrong tokens cost ~44 %
        # per-position acceptance (4.0 tok/block); the lever is here, not in the verify.
        self.draft_temp = float(_S.get("DRAFT_TEMP"))

        # Tap state: `_tap_i` counts invocations of `eng.tap` within one `eng.forward`.
        self._tap_i = 0
        self._tap_rows: list[torch.Tensor] = []
        shift = 0 if self.tap_mode == "entry" else 1
        self._want = {lid + shift: j for j, lid in enumerate(sorted(self.cfg.target_layer_ids))}
        self._n_taps = eng.cfg.num_hidden_layers + 1
        self.stats = {"calls": 0, "proposed": 0}
        self.attach()

    def set_sampling(self, sampler) -> None:
        """Sample proposals from this head when the request samples."""
        self.sampler = None if sampler is None or not getattr(sampler, "on", False) else sampler
        self.last_q = None

    # ---- lazy build ------------------------------------------------------------
    def _build(self) -> DFlash2Module:
        if self.module is not None:
            if self._ck is None:
                self._alloc_cache()          # released by `release`; the weights stayed
            return self.module
        self._w = load_weights(self.snapshot, self.eng.device)
        missing = [n for n in self.cfg.expected_tensors() if n not in self._w]
        if missing:
            raise RuntimeError(f"draft checkpoint is missing {len(missing)} tensors, "
                               f"first: {missing[:3]}")
        if DRAFT_EXL3 and os.path.isfile(self._exl3_file()):
            self._w = exl3_projections(self._w, os.path.expanduser(self._exl3_file()),
                                       key=f"{self.snapshot}|{self._exl3_file()}")
        elif DRAFT_NVFP4:
            self._w = quantise_projections(self._w, self.cfg.num_hidden_layers, key=self.snapshot)
        self.module = DFlash2Module(self.cfg, self._w)
        self._alloc_cache()
        if self._draft_head_path:
            from tools.draft_head import load_draft_head
            self.head, self.head_index = load_draft_head(self._draft_head_path, self.eng.device)
        return self.module

    def _exl3_file(self) -> str:
        """QWEN38_DRAFT_EXL3: one file, or a directory holding `<checkpoint dir name>.safetensors`
        for each arm (the length router builds both); an arm whose file is not there keeps its
        NVFP4 projections."""
        p = os.path.expanduser(DRAFT_EXL3)
        if os.path.isdir(p):
            return os.path.join(p, os.path.basename(os.path.normpath(self.snapshot)) + ".safetensors")
        return p

    def _alloc_cache(self) -> None:
        cfg = self.cfg
        shape = (cfg.num_hidden_layers, cfg.num_key_value_heads, self.max_len, cfg.head_dim)
        pages = getattr(getattr(self.eng, "kv", None), "pages", None)
        if pages is not None and self.max_len <= self.eng.kv.max_len:
            # SRV-307: the target's paged set grows and trims this arm's context KV with its own
            # rows (a context row is a position the target has forwarded)
            from engine import kvpool as _kvp
            self._ck = pages.add(_kvp.PagedBuffer(shape, 2, torch.bfloat16, self.eng.device,
                                                  owner=pages.owner, name="draft.k")).tensor
            self._cv = pages.add(_kvp.PagedBuffer(shape, 2, torch.bfloat16, self.eng.device,
                                                  owner=pages.owner, name="draft.v")).tensor
            return
        self._ck = torch.zeros(*shape, dtype=torch.bfloat16, device=self.eng.device)
        self._cv = torch.zeros_like(self._ck)

    def release(self, free_cache: bool = False) -> None:
        """Take this drafter out of service for the rest of the request, keeping its weights.

        The length router calls this on the arm that loses the latch. What it does is set
        `ctx_len` to zero, so that anything which does ask this arm to propose gets a DECLINE
        rather than a draft conditioned on positions that stopped being written -- and drop the
        tapped rows it was holding alive between steps.

        **What it deliberately does NOT do is free anything**, and both halves of that are
        arithmetic rather than caution.

        The checkpoint stays because the latch is a belief about the TEXT: `reset` throws it away
        and the next request needs both arms from its first block, where reloading would be paid in
        seconds against a 761 ms time to first token.

        The draft KV stays because freeing it is a cost with no benefit. It is
        `num_layers x kv_heads x max_len x head_dim` in two bf16 tensors, 20.0 kiB a token of
        CONTEXT -- 655 MB an arm at the shipped `--max-len 32768` -- and it goes back to a caching
        allocator pool that nothing else on this board can spend: the state store's budget is set
        by `--cache-budget-gb` at startup, not by what is free. What it costs to give back is a
        `torch.zeros` over 655 MB, twice, at the next request's first sync: about 6 ms of that
        request's time to first token, every request, for memory nobody asked for. It also makes
        `/health`'s `allocated` oscillate by 655 MB a request, which is the one number that is
        supposed to mean a leak.

        `free_cache=True` does it anyway, for measuring that claim rather than believing it.
        `_build` reallocates lazily, so an arm freed and then asked to sync comes back EMPTY rather
        than stale, which is the same decline as above and not a hole.
        """
        if free_cache:
            self._ck = None
            self._cv = None
        self.ctx_len = 0
        self._tap_rows = []

    # ---- tap -------------------------------------------------------------------
    def attach(self) -> None:
        """Install the tap. Overwrites whatever else was using `eng.tap` -- see the header."""
        self.eng.tap = self._on_tap

    def detach(self) -> None:
        if self.eng.tap is self._on_tap:
            self.eng.tap = None

    def _on_tap(self, h: torch.Tensor) -> None:
        i = self._tap_i
        if i == 0:
            self._tap_rows = [None] * len(self._want)          # type: ignore[list-item]
        j = self._want.get(i)
        if j is not None:
            self._tap_rows[j] = h
        self._tap_i = (i + 1) % self._n_taps

    def tap_indices(self) -> set:
        """The tap invocations `_on_tap` keeps (engine/model.py `prefill_window`)."""
        return set(self._want)

    def tapped(self):
        """(the target layer ids, the tap mode, their rows of the last forward), the rows one
        `[positions, hidden]` tensor a layer in sorted id order; None before a forward or once
        released. Read-only: the probe snapshots copy from them (server/probe.py)."""
        rows = self._tap_rows
        if not rows or any(r is None for r in rows):
            return None
        return sorted(self.cfg.target_layer_ids), self.tap_mode, rows

    # ---- drafter interface -----------------------------------------------------
    def reset(self) -> None:
        self.ctx_len = 0
        self._tap_i = 0
        self._tap_rows = []

    # ---- serving-time state cache ----------------------------------------------------------
    def state_snapshot(self):
        """The draft KV for the positions committed so far, sliced and cloned.

        `engine/cache.py` restores a target state without forwarding the prefix that produced it,
        so nothing writes this cache for those positions. That would leave a hole, and this cache
        is indexed by absolute position: a hole is permanent and the drafter declines for ever
        after. Measured on the board it is 20.0 kB a token -- 40 MB over a 2,000-token prompt,
        against 151 MB of recurrent state and 131 MB of target KV in the same snapshot. Not the
        rounding error it was assumed to be when this was written, and still the cheapest quarter
        of a snapshot: without it the prefill is warm and the decode is cold.
        """
        n = int(self.ctx_len)
        if self._ck is None or n == 0:
            return ("dflash2", 0, None, None)
        return ("dflash2", n, self._ck[:, :, :n].clone(), self._cv[:, :, :n].clone())

    def snapshot_bytes_per_token(self) -> int:
        """What `state_snapshot` clones per committed position: this arm's draft K and V."""
        if self._ck is None:
            return 0
        return 2 * self._ck[:, :, :1].numel() * self._ck.element_size()

    def state_restore(self, snap) -> None:
        kind, n, ck, cv = snap
        if kind != "dflash2":
            raise ValueError(f"not a dflash2 snapshot: {kind!r}")
        if n:
            self._build()
            from engine import kvpool as _kvp
            _kvp.ensure_for(self._ck, n)      # SRV-307
            self._ck[:, :, :n] = ck
            self._cv[:, :, :n] = cv
        self.ctx_len = int(n)

    def state_resume(self, n: int) -> None:
        """The draft KV rows below `n` are already in place, written by an earlier prefill of the
        same tokens (`engine/cache.py::ResidentPrefix`): pick up from there without a copy."""
        n = int(n)
        if n:
            self._build()
        self.ctx_len = n
        self._tap_rows = []

    def kv_views(self) -> list:
        """The position-indexed buffers a request writes, as (tensor, position axis)."""
        return [(self._ck, 2), (self._cv, 2)] if self._ck is not None else []

    def sync(self, tokens: list[int], hidden: torch.Tensor, first_pos: int,
             rows: list[int] | None = None) -> None:
        """Materialise the draft KV for the positions the target has just committed.

        `hidden` -- the target's post-final-norm hidden, which `engine/spec.py` hands every drafter
        -- is deliberately unused: DFlash2 wants the five mid-stack residual streams, which arrived
        through the tap during the same forward.

        Only `len(tokens)` rows are written, and `ctx_len` is set to `first_pos + len(tokens)`, so
        the KV of a position the verify pass rejected is never in the cache. This is the same
        failure the MTP head has a note about -- conditioning a draft on a rejected position --
        expressed as a cache length instead of a hidden state.
        """
        m = self._build()
        n = len(tokens)
        if n == 0:
            return
        taps = [r for r in self._tap_rows if r is not None]
        if len(taps) != len(self._want):
            raise RuntimeError(f"tap delivered {len(taps)} of {len(self._want)} target hidden "
                               f"states; is another consumer holding eng.tap?")
        if rows is None and taps[0].shape[0] < n:
            raise RuntimeError(f"tap holds {taps[0].shape[0]} positions, sync wants {n}")
        with torch.no_grad():
            if rows is None:
                # A chain block: the accepted tokens are the first n rows of the pass.
                picked = [r[:n] for r in taps]
            else:
                # A tree block: the tap holds one row per NODE in DFS order, and only the rows on
                # the accepted path are states the target actually committed to. Feeding it the
                # first n instead would condition the next draft on a branch that was rejected --
                # the 12:05 failure in the ledger, in its tree form.
                if len(rows) != n:
                    raise RuntimeError(f"{len(rows)} rows for {n} tokens")
                from engine.model import h2d
                sel = h2d(rows, torch.long, taps[0].device)
                picked = [r[sel] for r in taps]
            fused = torch.cat(picked, dim=-1)
            ctx_hidden = m.project_context(fused.to(m.dtype))
            positions = torch.arange(first_pos, first_pos + n, device=self.eng.device)
            for i, (k, v) in enumerate(m.context_kv(ctx_hidden, positions)):
                self._ck[i, :, first_pos:first_pos + n] = k
                self._cv[i, :, first_pos:first_pos + n] = v
        self.ctx_len = first_pos + n

    def propose(self, context: list[int], k: int) -> list[int]:
        return run_steps(self._propose_steps(context, k))

    def _propose_steps(self, context: list[int], k: int, logp: bool = False):
        """`propose` as a generator that stops once the draft is on the device. `logp`:
        also bring back the lattice's log-probabilities, which only the tree reads."""
        self.last_q = None
        if k <= 0 or self.ctx_len == 0:
            return []
        m = self._build()
        cfg = self.cfg
        bs = cfg.block_size
        base = len(context) - 1                    # the anchor's absolute position
        if base != self.ctx_len:
            # The anchor is the first position the draft cache does not cover; anything else means
            # a sync was missed and the context would be conditioned on the wrong prefix.
            return []
        if base + bs * self.blocks >= self.max_len:
            return []
        self.stats["calls"] += 1
        out: list[int] = []
        anchor = int(context[-1])
        carry: list[tuple[torch.Tensor, torch.Tensor]] | None = None
        carry_pos: torch.Tensor | None = None
        if self.sampler is not None and getattr(self.sampler, "on", False):
            self.last_q = []
        with torch.no_grad():
            for b in range(self.blocks):
                pos0 = base + b * (bs - 1)
                toks, carry, carry_pos = yield from self._one_block_steps(m, anchor, pos0, carry,
                                                                          carry_pos, logp)
                out.extend(toks)
                if len(out) >= k:
                    break
                anchor = toks[-1]
        out = out[:k]
        if self.last_q is not None:
            self.last_q = self.last_q[:len(out)]
        self.stats["proposed"] += len(out)
        return out

    # ---- one 8-wide block ------------------------------------------------------
    def _one_block_steps(self, m: DFlash2Module, anchor: int, pos0: int,
                         carry: list[tuple[torch.Tensor, torch.Tensor]] | None,
                         carry_pos: torch.Tensor | None, logp: bool = False):
        """Build `[anchor, MASK x (block_size-1)]` at positions `pos0 .. pos0+block_size-1`, run
        the module, turn rows 1.. into tokens, and hand back the block's own K/V for a chained
        second block.

        The chain (`blocks=2`) is this engine's own and has no counterpart in the serving stack,
        which runs one draft forward per verify; there the only way to propose more than
        `block_size - 1` tokens is to raise `block_size` itself. What is carried here is rows
        0..block_size-2 of this block's per-layer K and V -- not row block_size-1, whose position
        the next block's anchor re-occupies and re-derives from the token this block predicted for
        it. The context `fc`/`hidden_norm` tensor is *not* extended: the draft cannot produce
        target hidden states for tokens the target has not seen, so the second block's new
        conditioning is its anchor token, its positions, and this block's attention keys -- and
        nothing else. Expect it to accept worse than the first block. That is the honest shape of
        chaining a block drafter with no target pass in between, and the reason `blocks` defaults
        to 1 until a measurement says otherwise.
        """
        cfg = self.cfg
        bs = cfg.block_size
        dev = self.eng.device
        if carry is None and DRAFT_GRAPH and HOST_WALK and torch.cuda.is_available():
            from engine.drafters.draft_graph import DraftGraph
            if DraftGraph.eligible(self):
                # parallel requests: with other slots decoding, this draft call joins their packed
                # draft (`DRAFT_MEET`, server/app.py); alone, or outside such a server, the graph
                out = DRAFT_MEET(self, anchor, pos0) if DRAFT_MEET is not None else None
                if out is None:
                    out = self.graph_run(anchor, pos0)
                # the draft is queued; a caller with work of its own does it now
                yield
                cand, scores = out[0], out[1]
                self._lattice = (cand, scores)
                self._logp_host = None
                if len(out) > 2 and logp:
                    toks, self._cand_host, self._logp_host = m.walk_host_logp(cand, scores, out[2])
                else:
                    toks, self._cand_host = m.walk_host(cand, scores)
                return toks, None, None
        ids = torch.full((bs,), cfg.mask_token_id, dtype=torch.long, device=dev)
        ids[0] = anchor
        noise = F.embedding(ids, self.eng.w.norm("embed_tokens.weight")).to(m.dtype)
        positions = torch.arange(pos0, pos0 + bs, device=dev)

        lo = 0
        if cfg.sliding_window is not None:
            lo = max(0, pos0 - cfg.sliding_window + 1)
        ctx_pos = torch.arange(lo, self.ctx_len, device=dev)
        ctx_kv = [(self._ck[i, :, lo:self.ctx_len], self._cv[i, :, lo:self.ctx_len])
                  for i in range(cfg.num_hidden_layers)]
        if carry is not None:
            # Target-derived keys first, then the earlier blocks' own: the key order has to match
            # `ctx_pos`, which the sliding-window mask is built from.
            ctx_kv = [(torch.cat([k, c[0]], dim=1), torch.cat([v, c[1]], dim=1))
                      for c, (k, v) in zip(carry, ctx_kv)]
            ctx_pos = torch.cat([ctx_pos, carry_pos])

        want_kv = self.blocks > 1
        result = m.forward_block(noise, positions, ctx_kv, ctx_pos, return_kv=want_kv)
        hidden, block_kv = result if want_kv else (result, None)
        pred = hidden[1:]                                        # row 0 is the anchor: dead
        tokens = self._tokens_from(m, pred, anchor, first=pos0 + 1)
        if not want_kv:
            return tokens, None, None
        keep = bs - 1
        new_carry = [(k[:, :keep], v[:, :keep]) for k, v in block_kv]
        new_pos = positions[:keep]
        if carry is not None:
            new_carry = [(torch.cat([c[0], k], dim=1), torch.cat([c[1], v], dim=1))
                         for c, (k, v) in zip(carry, new_carry)]
            new_pos = torch.cat([carry_pos, new_pos])
        return tokens, new_carry, new_pos

    def graph_run(self, anchor: int, pos0: int):
        """The lone draft call from this drafter's graph."""
        from engine.drafters.draft_graph import DraftGraph
        g = getattr(self, "_graph", None)
        if g is None or g.ck_ptr != self._ck.data_ptr():
            g = self._graph = DraftGraph(self)
        return g.run(anchor, pos0)

    def _tokens_from(self, m: DFlash2Module, pred: torch.Tensor, anchor: int,
                     first: int = 0) -> list[int]:
        """Rows 1.. of the block through the target's head, in ONE call over all of them.
        `first` is the sequence position row 1 proposes for."""
        from engine.model import head_logits, linear
        self._lattice = None
        if self.head is not None:
            logits = linear(pred, self.head)
        elif DRAFT_HEAD_NVFP4:
            logits = linear(pred, nvfp4_head(self.eng))
        elif DRAFT_HEAD_EXL3:
            logits = head_logits(pred, exl3_draft_head(self.eng))
        else:
            # The same 2.54 GB the verify step reads, read again for seven rows. It goes through
            # the engine's own head kernel for the same reason the verify path does.
            logits = head_logits(pred, self.eng.w.norm("lm_head.weight"))
        if self.sampler is not None and getattr(self.sampler, "on", False):
            # q-aware drafting: draw each row's token from this head's own distribution
            # under the request's profile and carry q for the verify's min(1, p/q) accept. The
            # selector's walk is a greedy policy over greedy-tuned scores, so it is not used when
            # sampling; the head's own distribution is the drafter's real proposal distribution.
            dt = getattr(self.sampler, "draft_temperature", None) or self.draft_temp
            rows = self.sampler.probs_rows(logits if dt == 1.0 else logits / dt)
            ids: list[int] = []
            if self.last_q is None:
                self.last_q = []
            from engine import sample as _smp
            batch = (_smp.BATCH and not self.sampler.coupled and self.head_index is None
                     and hasattr(self.sampler, "draws"))
            if batch:
                # every row's proposal in one device draw and one copy (engine/sample.py BATCH)
                drawn = self.sampler.draws(rows).tolist()
            for r in range(rows.shape[0]):
                row = rows[r]
                if batch:
                    ids.append(int(drawn[r]))
                    self.last_q.append(row)
                    continue
                t = self.sampler.pick(row) if not self.sampler.coupled else None
                if self.head_index is not None:
                    # A reduced draft head: q lives on the reduced vocabulary and must be
                    # scattered into the full one so p and q are rows over the same support.
                    full = torch.zeros(self.eng.cfg.vocab_size, dtype=row.dtype, device=row.device)
                    full[self.head_index] = row
                    row = full
                    t = int(self.head_index[t]) if t is not None else None
                if t is None:
                    # A seeded request: propose under the SAME position-keyed noise the
                    # target will draw with, over the full vocabulary, so a proposal that agrees
                    # with the target's draw is accepted and one that does not costs nothing.
                    t = self.sampler.pick_at(row, first + r)
                ids.append(int(t))
                self.last_q.append(row)
            if self._want_lattice and self.use_selector:
                # a sampled request's tree -- the spine is the sample above, the siblings
                # and the shape come from the same block's lattice (`propose_tree`)
                cand, unary = m.unary_candidates(logits)
                if self.head_index is not None:
                    cand = self.head_index[cand].long()
                self._lattice = (cand, m.lattice(pred, cand, unary, anchor))
                self._cand_host = None
            return ids
        if not self.use_selector:
            ids = logits.argmax(-1)
            if self.head_index is not None:
                ids = self.head_index[ids]
            return [int(x) for x in ids]
        cand, unary = m.unary_candidates(logits)
        if self.head_index is not None:
            cand = self.head_index[cand].long()
        scores = m.lattice(pred, cand, unary, anchor)
        # The lattice is [slots, k predecessors, k candidates] = 7 x 16 x 16 numbers the greedy walk
        # builds in full and then reads 7 of. A tree verify can afford to read the rest; keeping it
        # here costs one 7 KB copy and no extra memory traffic on the weights at all.
        self._lattice = (cand, scores)
        self._cand_host = None
        if HOST_WALK and self.path == "greedy":
            toks, self._cand_host = m.walk_host(cand, scores)
            return toks
        walk = m.viterbi if self.path == "viterbi" else m.walk
        return [int(x) for x in walk(cand, scores)]

    # ---- the tree -------------------------------------------------------------
    def propose_tree(self, context: list[int], budget: int = 16, **_):
        return run_steps(self.propose_tree_steps(context, budget))

    def propose_tree_steps(self, context: list[int], budget: int = 16, **_):
        """The same block drafted as a TREE: the greedy path, then the best nodes around it.

        The drafter already computes a distribution at every slot and 16 candidates per slot, and a
        chain throws away 15 of every 16. On this board a node costs about 2 ms against a 145-175 ms
        step, so the question is not whether width is worth it but how much of it to buy.

        The construction is best-first over path probability under a node budget, seeded with the
        released greedy walk. The seeding is not a detail: the ledger's 10:28 entry measured the
        exact maximiser of the selector's own score -- Viterbi -- accepting 3.22 tokens a block
        against greedy's 4.30, because what a verify pays for is the expected accepted PREFIX, in
        which slot 0 multiplies every later term, and a maximiser will trade slot 0 away for a
        better total. Greedy's slot 0 is the head's own top-1, the single most reliable signal in
        the lattice, so the greedy path goes in first and the budget buys alternatives around it.

        Best-first is exactly right for the rest: a node's path probability is its parent's times a
        conditional, so priorities fall monotonically down any path, and popping in descending order
        yields the highest-probability ancestor-closed set of nodes there is -- which is the set
        that maximises expected accepted length for a given budget.
        """
        from engine.tree import DraftTree, lattice_paths, lattice_tree, level_quota, spine_tree

        anchor = int(context[-1])
        # a request that samples (the server's `--sampled-tree mixed`) gets the SAMPLED
        # chain as the tree's spine with its q rows, and deterministic siblings from the lattice
        sampled = self.sampler is not None and getattr(self.sampler, "on", False)
        self._want_lattice = sampled
        self._logp_host = None
        try:
            chain = yield from self._propose_steps(context, self.cfg.block_size - 1, logp=True)
        finally:
            self._want_lattice = False
        qrows = self.last_q if sampled else None
        self.last_q = None                     # the tree carries q itself (tree.q), not last_q
        self.last_det_tree = None
        if not chain:
            return None
        build = lattice_paths if self.tree_mode == "paths" else lattice_tree
        if self._lattice is None or budget <= 0:
            if sampled:
                self.last_det_tree = DraftTree.chain(anchor, chain, source="df2-greedy")
                return spine_tree(anchor, chain, qrows, [[t] for t in chain],
                                  [[[0.0]]] * len(chain), [])
            return DraftTree.chain(anchor, chain, source="df2-greedy")
        cand_t, scores_t = self._lattice
        cand = (self._cand_host if HOST_WALK and getattr(self, "_cand_host", None) is not None
                else cand_t.tolist())                            # [L][k]
        logp = self._logp_host
        if logp is None:
            logp = torch.log_softmax(scores_t.float() / self.tree_temp, dim=-1).tolist()
        if sampled:
            # the deterministic tree this lattice builds -- its own greedy walk first, as a greedy
            # request's -- fixes the shape (nodes a level) before anything looks at the sample, and
            # is what the router prices the head's proposal on (engine/router.py)
            greedy, row = [], 0
            for l in range(len(cand)):
                row = max(range(len(cand[l])), key=lambda c, l=l, row=row: logp[l][row][c])
                greedy.append(row)
            det = build(anchor, cand, logp, greedy, budget)
            self.last_det_tree = det
            return spine_tree(anchor, chain, qrows, cand, logp, level_quota(det))
        greedy = [cand[l].index(chain[l]) if chain[l] in cand[l] else 0
                  for l in range(min(len(cand), len(chain)))]
        return build(anchor, cand, logp, greedy, budget)

    # ---- accounting ------------------------------------------------------------
    def draft_bytes(self, head_bytes: int | None = None) -> dict[str, float]:
        """Bytes one block draft reads. The number drafters are compared by -- ARCHITECTURE 1.4."""
        cfg = self.cfg
        exp = cfg.expected_tensors()
        backbone = sum(_numel(s) for n, s in exp.items()
                       if not n.startswith("candidate_selector.")) * 2
        proj = _numel(exp["candidate_selector.hidden_projection.weight"]) * 2 \
            if cfg.selector_rank else 0
        slots = cfg.block_size - 1
        gather = slots * 2 * cfg.selector_top_k * cfg.selector_rank * 2 if cfg.selector_rank else 0
        if head_bytes is None:
            h = self.eng.w.norm("lm_head.weight")
            head_bytes = h.numel() * h.element_size()
        sel = (proj + gather) if self.use_selector else 0
        return {"backbone_B": backbone, "selector_B": sel, "head_B": head_bytes,
                "total_B": backbone + sel + head_bytes, "proposals": slots * self.blocks}


def _numel(shape: tuple[int, ...]) -> int:
    n = 1
    for d in shape:
        n *= d
    return n
