"""DFlash2 for the Qwen3.5 MoE trunk (ENG-391): a block drafter as a draft head of the shared tree drafter
(engine/methods/mtp_drafter.py), reading its context from the engine's taps (`Qwen35MoeEngine.set_taps`).

The checkpoint (`DFlash2DraftModel`, BF16; e.g. Ornith-1.5-122B's: 6 sliding-window layers of hidden 3,072,
32 query / 8 KV heads of 128, block 16, the grouped dynamic convolution of kernel 2 and group 16, a rank-256
candidate selector over the top 16, mask 248,077, 8 target layers) is engine/drafters/dflash2.py's family:

  context   for every committed target token, the outputs of the target layers `target_layer_ids`
            (the residual rows entering the next layers, `TAP_SHIFT`), concatenated -> fc -> RMSNorm hidden_norm = g; each layer's
            K = rope(k_norm(k_proj g)) and V = v_proj g at the token's position, kept in a ring of
            `sliding_window` positions (a block row sees the context keys within the window)
  block     [anchor, MASK x 15] at positions n .. n + 15, the target's embedding, non-causal inside the
            block; per layer RMSNorm, the convolution's input side, q/k/v with q/k norms, RoPE (all 128
            dims), attention over the context keys and the block, o_proj, the convolution's output side,
            the residual; the same around the SiLU MLP; the final norm; the target's LM head on rows 1..15
  selector  per slot the head's top 16 candidates and their logits; a candidate c after predecessor p
            scores logit(c) + <predecessor_codebook[p] * hidden_projection(row), successor_codebook[c]>,
            slot 1's predecessor being the anchor; the greedy walk takes slot 1's best and each next
            slot's best given the previous choice (the reference's `walk`)

The block, the head, the selector and the walk are engine/drafters/lilicorr.py's draft machinery (its
block is DFlash2's with the window ring; its head object keeps the ring and captures the draft as one
CUDA graph); the selector replaces LiLiCorr's lattice. A draft only proposes: every drafted token is
verified by the trunk, so the draft's arithmetic (bf16, torch) never reaches the text.
"""

from __future__ import annotations

import os

import torch

from engine.drafters.dflash2 import DFlash2Config, _Conv, load_weights
from engine.drafters.lilicorr import LiLiCorrHead, LiLiCorrModel


class _RingConfig:
    """DFlash2Config with the fields the ring machinery reads."""

    def __init__(self, c: DFlash2Config):
        if c.is_causal or not c.sliding_window or any(t != "sliding_attention" for t in c.layer_types):
            raise ValueError("DFlash2 head: a non-causal draft with every layer on one sliding window only")
        if not c.selector_rank or not c.selector_top_k:
            raise ValueError("DFlash2 head: the candidate selector is required")
        if c.output_multiplier != 1.0 or c.final_logit_softcapping:
            raise ValueError("DFlash2 head: output multiplier / soft-capping not implemented")
        for k in ("hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "head_dim",
                  "intermediate_size", "rms_norm_eps", "rope_theta", "vocab_size", "block_size", "mask_token_id",
                  "target_layer_ids", "conv_kernel_size", "conv_group_size", "selector_rank", "selector_top_k"):
            setattr(self, k, getattr(c, k))
        self.window = int(c.sliding_window)
        self.sinks = False

    @property
    def slots(self) -> int:
        return self.block_size - 1

    @property
    def num_groups(self) -> int:
        return self.hidden_size // self.conv_group_size if self.conv_group_size else 0


class DFlash2RingModel(LiLiCorrModel):
    """The draft's arithmetic over its weights, the target's embedding and head (rows -> fp32 logits)."""

    def __init__(self, cfg: _RingConfig, W: dict, embed: torch.Tensor, head, kv_from: dict | None = None,
                 ids_map: torch.Tensor | None = None):
        """`kv_from`: the bf16 weights the context's K and V are projected with (when `W`'s block
        projections are quantised); `ids_map`: the vocabulary ids of the head's rows (a hot head)."""
        self.cfg, self.W, self.embed, self.head, self.ids_map = cfg, W, embed, head, ids_map
        c = cfg
        self.convs = []
        for i in range(c.num_hidden_layers):
            p = f"layers.{i}"
            self.convs.append(None if not c.conv_kernel_size else (
                _Conv(W[f"{p}.attention_conv.base_kernel"], W[f"{p}.attention_conv.kernel_projection.weight"],
                      c.conv_kernel_size, c.num_groups, c.conv_group_size),
                _Conv(W[f"{p}.mlp_conv.base_kernel"], W[f"{p}.mlp_conv.kernel_projection.weight"],
                      c.conv_kernel_size, c.num_groups, c.conv_group_size)))
        kw = kv_from if kv_from is not None else W
        self.kv_w = torch.cat([kw[f"layers.{i}.self_attn.{x}_proj.weight"]
                               for i in range(c.num_hidden_layers) for x in "kv"]).contiguous()
        self.sink = [None] * c.num_hidden_layers

    def selector(self, rows: torch.Tensor, ids: torch.Tensor, unary: torch.Tensor,
                 anchor: torch.Tensor) -> torch.Tensor:
        """scores [slots, k_pred, k_cand]: unary + <A[pred] * project(row), B[cand]> (DFlash2Module.lattice)."""
        W, k = self.W, self.cfg.selector_top_k
        hidden = torch.nn.functional.linear(rows, W["candidate_selector.hidden_projection.weight"])
        keys = W["candidate_selector.successor_codebook"][ids]
        pred_ids = torch.cat([anchor.reshape(1, 1).expand(1, k), ids[:-1]], dim=0)
        preds = W["candidate_selector.predecessor_codebook"][pred_ids]
        pair = preds.float() * hidden.float()[:, None, :]
        return unary[:, None, :] + torch.einsum("lpr,lcr->lpc", pair, keys.float())

    def draft(self, anchor, pos0, ck, cv, ctx_ok, anchor_g=None, anchor_ok=1.0) -> torch.Tensor:
        """[slots, 4] on the device: per slot (token, its softmax over the slot's row of scores, the
        second token, its softmax), the greedy walk through the selector's scores."""
        hid = self.block(anchor, pos0, ck, cv, ctx_ok)
        rows = hid[1:]
        top, ids = torch.topk(self.head(rows).float(), self.cfg.selector_top_k, dim=-1)
        if self.ids_map is not None:
            ids = self.ids_map[ids]
        sc = self.selector(rows, ids, top, anchor)
        return self.walk(ids, sc[0, 0], sc[1:])


class DFlash2Head(LiLiCorrHead):
    """The DFlash2 draft as a head of the tree drafter: depth block - 1, its context from the engine's taps."""


from engine.settings import SETTINGS as _S  # noqa: E402

#: the draft's attention and MLP projections in NVFP4 (quantised at load, `dflash2.quantise_projections`): a
#: draft reads 0.4 GB instead of 1.4; drafter only, so the text cannot change, acceptance can
NVFP4 = _S.get("Q35M_DFLASH2_NVFP4") == "1"
#: the draft's head over the hot vocabulary (the engine's `set_draft_head`, int8 rows) instead of the target's
#: EXL3 head (0.57 GB a draft)
HOT = _S.get("Q35M_DFLASH2_HOT") == "1"

#: where a target layer id is read: 0 = the residual entering that layer, 1 = the layer's output (the residual
#: entering the next one, `hidden_states[id + 1]` of transformers' output_hidden_states). Measured on Ornith 4.0
#: (tools/q35m_dflash2_check.py, the gate's mean run of kept drafted tokens on three prompts, 2026-10-20):
#: 1.57 / 7.67 / 1.57 at 0, 1.62 / 9.14 / 2.05 at 1
TAP_SHIFT = 1


def build_head(srv, path: str, log=print) -> DFlash2Head:
    """The checkpoint at `path` over the serving engine's embedding and head (the engine's taps set first,
    `Qwen35MoeEngine.set_taps(target_layer_ids, sliding_window)`)."""
    import json
    with open(os.path.join(path, "config.json")) as f:
        raw = json.load(f)
    if "DFlash2DraftModel" not in (raw.get("architectures") or []):
        raise ValueError(f"{path}: not a DFlash2 draft ({raw.get('architectures')})")
    cfg = _RingConfig(DFlash2Config(raw))
    eng = srv.fn
    if cfg.hidden_size != eng.cfg.hidden_size or cfg.vocab_size != eng.cfg.vocab_size:
        raise ValueError(f"DFlash2 hidden {cfg.hidden_size} / vocab {cfg.vocab_size} against the target's "
                         f"{eng.cfg.hidden_size} / {eng.cfg.vocab_size}")
    if tuple(eng.taps) != tuple(x + TAP_SHIFT for x in cfg.target_layer_ids):
        raise ValueError(f"the engine taps {eng.taps}, the draft reads {cfg.target_layer_ids}")
    W = load_weights(path, str(eng.device))
    Wb = W
    if NVFP4:
        from engine.drafters.dflash2 import quantise_projections
        W = quantise_projections(W, cfg.num_hidden_layers)
    ids_map = None
    if HOT and getattr(eng, "draft_head", None) is not None:
        from tools import fn_kernels as K
        dh, ids_map = eng.draft_head, eng.draft_ids

        def head(rows):
            return K.gemv(rows.contiguous(), dh)
    else:
        hw = eng.W["lm_head.weight"]

        def head(rows):
            return eng.lin(rows.contiguous(), hw).float()
    model = DFlash2RingModel(cfg, W, eng.W["embed_tokens.weight"], head, kv_from=Wb, ids_map=ids_map)
    log(f"[dflash2] {path}: {cfg.num_hidden_layers} layers, block {cfg.block_size}, window {cfg.window}, "
        f"taps {cfg.target_layer_ids}, selector top {cfg.selector_top_k} rank {cfg.selector_rank}, "
        f"projections {'NVFP4' if NVFP4 else 'BF16'}, head {'hot ' + str(ids_map.numel()) if ids_map is not None else 'the target'}'s")
    return DFlash2Head(srv, model)


def config_of(path: str) -> dict:
    import json
    with open(os.path.join(path, "config.json")) as f:
        raw = json.load(f)
    c = DFlash2Config(raw)
    return {"target_layer_ids": [x + TAP_SHIFT for x in c.target_layer_ids], "window": int(c.sliding_window or 0)}
