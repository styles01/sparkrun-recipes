"""DSpark: the same block-drafter shape as DFlash2, a different backbone, and two extra heads.

`Doopeworld/Qwen3.8-27B-DSpark-vLLM`. The published claim is 16-18 % more accepted tokens a block
than DFlash on the same target, and the reason to care here
is that the median of this engine's row is fresh prose, where the shipped drafter accepts 2.7 of a
sixteen-token block and everything phase 6 bought landed on the other half of the row.

WHAT IT SHARES WITH DFlash2, AND WHY THIS FILE IS SHORT
-------------------------------------------------------
The reference (`dspark.py` / `dflash.py` in the checkpoint) makes `DSparkDraftModel` a subclass of
SpecForge's `DFlashDraftModel`, and that base is the same construction `engine/drafters/dflash2.py`
already implements: no embedding and no head of its own, a `fc` over five captured target hidden
states through `hidden_norm`, five non-causal decoder layers whose K and V come from BOTH the
projected context and the noise block, and a block of `[anchor, MASK, MASK, ...]` whose rows
1.. are the proposals. So the module here is `DFlash2Module` with three things changed and the
drafter is `DFlash2Drafter` with two.

WHAT IS DIFFERENT, ALL OF IT LOAD-BEARING
-----------------------------------------
1. **No dynamic convolution and no candidate selector.** The z-lab DFlash2 checkpoint wraps every
   sublayer in a dynamic depthwise convolution and carries a bigram lattice over 16 candidates a
   slot; this one has neither. `DFlash2Config` already reads that off the config (`conv_kernel_size`
   and `selector_rank` both absent -> 0), and both code paths already test for it, so nothing here
   has to say so. What it costs is the tree: with no lattice there are no alternatives to put in
   one, and `propose_tree` returns a chain. On this board that is a real loss -- phase 6 measured
   the tree at +40.8 % on the row -- and it is why DSpark is offered as a THIRD ARM rather than a
   replacement: the length router can send fresh prose to it and reproduction to the tree.

2. **YaRN rotary.** `rope_parameters.rope_type = "yarn"`, factor 32 over an original 8,192-position
   window, `beta_fast` 32 and `beta_slow` 1. This is not a long-context detail that can be skipped
   at short lengths: YaRN rescales the inverse frequencies at EVERY position and multiplies cos and
   sin by an attention factor, so a drafter run with plain RoPE is not a slightly worse drafter, it
   is a drafter reading its own positions wrong. Implemented in `_yarn_inv_freq` against the HF
   `_compute_yarn_parameters` it was trained under.

3. **Block size 7**, so six proposals a block against DFlash2's seven at block 8 and fifteen at
   block 16, and a different mask token (248077).

4. **Two heads the DFlash2 checkpoint does not have.**

   * `markov_head` -- a rank-256 learned bigram bias, `markov_w2(markov_w1(prev_token))`, added to
     the draft logits. It conditions slot j on the token at slot j-1, which at training time is
     teacher-forced and at inference time is whatever the drafter itself proposed there. The
     reference's own `spec_generate` does not apply it (it is the DFlash base's loop), so the
     inference-time reading is this file's: run the block once unbiased, take the proposals, and
     use them as the previous tokens for one refinement pass. That is the natural fixed-point step
     and it costs one read of `markov_w2` -- 127 MB, about 0.6 ms -- for the whole block rather
     than one per slot. `--markov 0` turns it off and the A/B says what it was worth.

   * `confidence_head` -- one linear over [hidden, markov latent] predicting whether each slot will
     be accepted. Nothing here consumes it yet; it is computed and exposed as `last_conf` because
     an adaptive block length is exactly what the length router already is, and the router's
     current evidence is the arm's own accept history rather than a per-block prediction. Wiring
     it is a measurement, not a port, and it is left for one.

NOTHING HERE CAN CHANGE WHAT THE ENGINE WRITES. A drafter proposes; the target verifies every
proposal against its own argmax. A DSpark block that is wrong costs a rollback and nothing else,
which is why this file has no losslessness argument of its own -- `tools/verify_spec.py` runs the
same gate against it that every other drafter passes.
"""

from __future__ import annotations

import math
import os
from engine.settings import SETTINGS as _S  # noqa: E402  (every QWEN38_* knob)

import torch
import torch.nn.functional as F

from engine.drafters.dflash2 import (DFlash2Config, DFlash2Drafter, DFlash2Module, _rotate_half,
                                     load_weights, resolve_checkpoint)

DEFAULT_REPO = "Doopeworld/Qwen3.8-27B-DSpark-vLLM"


def _yarn_inv_freq(dim: int, base: float, factor: float, orig_max: int,
                   beta_fast: float, beta_slow: float,
                   device) -> tuple[torch.Tensor, float]:
    """The inverse frequencies and the attention factor YaRN trains with.

    Ported from `transformers.modeling_rope_utils._compute_yarn_parameters`. NTK-by-parts: the
    dimensions that complete many rotations inside the original window are left alone
    (extrapolated), the ones that complete less than one are scaled down by `factor`
    (interpolated), and between the two correction points there is a linear ramp.
    """
    pos_freqs = base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
    extrapolation = 1.0 / pos_freqs
    interpolation = 1.0 / (factor * pos_freqs)

    def correction_dim(rot: float) -> float:
        return (dim * math.log(orig_max / (rot * 2 * math.pi))) / (2 * math.log(base))

    low = math.floor(correction_dim(beta_fast))
    high = math.ceil(correction_dim(beta_slow))
    low, high = max(low, 0), min(high, dim - 1)
    if low == high:
        high += 0.001
    ramp = (torch.arange(dim // 2, dtype=torch.float32, device=device) - low) / (high - low)
    ramp = ramp.clamp(0.0, 1.0)
    ext_factor = 1.0 - ramp
    inv = interpolation * (1 - ext_factor) + extrapolation * ext_factor
    return inv, 0.1 * math.log(factor) + 1.0


class DSparkConfig(DFlash2Config):
    """`DFlash2Config` plus the YaRN parameters and the two head sizes.

    Everything else it reads off the same keys: this checkpoint is a `Qwen3Config` with the same
    `dflash_config` block, and the fields DFlash2 has that DSpark does not -- the convolution and
    the selector -- read as absent and switch their code paths off on their own.
    """

    def __init__(self, raw: dict):
        super().__init__(raw)
        rp = raw.get("rope_parameters", {}) or {}
        self.rope_type = str(rp.get("rope_type", "default"))
        self.rope_factor = float(rp.get("factor", 1.0))
        self.rope_orig_max = int(rp.get("original_max_position_embeddings", 8192))
        self.rope_beta_fast = float(rp.get("beta_fast", 32.0))
        self.rope_beta_slow = float(rp.get("beta_slow", 1.0))
        self.markov_rank = int(raw.get("markov_rank", 0))
        self.markov_head_type = str(raw.get("markov_head_type", "vanilla"))
        self.enable_confidence_head = bool(raw.get("enable_confidence_head", False))
        self.confidence_head_with_markov = bool(raw.get("confidence_head_with_markov", True))

    def expected_tensors(self) -> dict[str, tuple[int, ...]]:
        out = super().expected_tensors()
        if self.markov_rank:
            out["markov_head.markov_w1.weight"] = (self.vocab_size, self.markov_rank)
            out["markov_head.markov_w2.weight"] = (self.vocab_size, self.markov_rank)
        if self.enable_confidence_head:
            d = self.hidden_size + (self.markov_rank if self.confidence_head_with_markov else 0)
            out["confidence_head.proj.weight"] = (1, d)
            out["confidence_head.proj.bias"] = (1,)
        return out


class DSparkModule(DFlash2Module):
    """`DFlash2Module` with the YaRN rotary and the two heads."""

    def __init__(self, cfg: DSparkConfig, w: dict[str, torch.Tensor]):
        super().__init__(cfg, w)
        self._rope_cache: dict[tuple[int, int, str], tuple[torch.Tensor, torch.Tensor]] = {}

    def rope(self, positions: torch.Tensor, dim: int, dtype: torch.dtype):
        cfg = self.cfg
        if cfg.rope_type != "yarn" or _S.get("DSPARK_NO_YARN") == "1":
            return super().rope(positions, dim, dtype)
        inv, att = _yarn_inv_freq(dim, cfg.rope_theta, cfg.rope_factor, cfg.rope_orig_max,
                                  cfg.rope_beta_fast, cfg.rope_beta_slow, positions.device)
        f = positions.float()[:, None] * inv[None, :]
        emb = torch.cat([f, f], dim=-1)
        # The attention factor is part of the rotary in YaRN, not a separate softmax scale: HF
        # folds it into cos and sin, and so does this.
        return (emb.cos() * att).to(dtype), (emb.sin() * att).to(dtype)

    # --- the heads ---------------------------------------------------------------------------

    def markov_latent(self, prev_ids: torch.Tensor) -> torch.Tensor:
        """`markov_w1[prev]`: one gathered row of 256 per slot, 512 bytes."""
        return F.embedding(prev_ids, self.w["markov_head.markov_w1.weight"])

    def markov_bias(self, latent: torch.Tensor) -> torch.Tensor:
        """`markov_w2 @ latent`: one read of a [248320, 256] matrix for the whole block."""
        return F.linear(latent, self.w["markov_head.markov_w2.weight"])

    def confidence(self, pred_hidden: torch.Tensor,
                   latent: torch.Tensor | None) -> torch.Tensor:
        """Predicted acceptance probability per slot, in [0, 1]."""
        feats = pred_hidden if latent is None else torch.cat([pred_hidden, latent], dim=-1)
        z = F.linear(feats.float(), self.w["confidence_head.proj.weight"].float(),
                     self.w["confidence_head.proj.bias"].float())
        return torch.sigmoid(z[:, 0])


class DSparkDrafter(DFlash2Drafter):
    """`DFlash2Drafter` wired to `DSparkModule`.

    Everything the loop touches -- the tap, the draft KV cache, `sync`, `observe`, the snapshot
    pair -- is the base class's and is unchanged, because the two checkpoints consume the target
    in exactly the same way: five captured residual streams and nothing else. What this overrides
    is which module gets built and how a block's hidden rows become tokens.
    """

    name = "dspark"

    def __init__(self, eng, ckpt: str | None = None, *, markov: bool = True,
                 max_len: int | None = None, draft_head: str | None = None,
                 block: int | None = None, tap: str = "output"):
        # `selector=False` is not a choice: this checkpoint has no selector tensors, and the base
        # class already reads `selector_rank = 0` off the config. Passing it makes that explicit
        # and keeps `use_selector` false even if a future config carries the key.
        super().__init__(eng, ckpt or _S.get("DSPARK", DEFAULT_REPO),
                         blocks=1, selector=False, draft_head=draft_head,
                         max_len=max_len, path="greedy", tap=tap, block=block)
        self.markov = bool(markov)
        self.last_conf: list[float] | None = None

    # The base class parses the config in `__init__` before this class can intervene, so the
    # DSpark-only fields are re-read here off the same file. One parse, two views.
    def _reparse(self) -> None:
        import json
        with open(os.path.join(self.snapshot, "config.json")) as f:
            raw = json.load(f)
        self.cfg = DSparkConfig(raw)

    def _build(self):
        if self.module is not None:
            return self.module
        self._reparse()
        if self.cfg.markov_head_type != "vanilla":
            raise ValueError(f"only the vanilla markov head is implemented, not "
                             f"{self.cfg.markov_head_type!r}")
        self._w = load_weights(self.snapshot, self.eng.device)
        missing = [n for n in self.cfg.expected_tensors() if n not in self._w]
        if missing:
            raise RuntimeError(f"DSpark checkpoint is missing {len(missing)} tensors, "
                               f"first: {missing[:3]}")
        self.module = DSparkModule(self.cfg, self._w)
        cfg = self.cfg
        self._ck = torch.zeros(cfg.num_hidden_layers, cfg.num_key_value_heads, self.max_len,
                               cfg.head_dim, dtype=torch.bfloat16, device=self.eng.device)
        self._cv = torch.zeros_like(self._ck)
        if self._draft_head_path:
            from tools.draft_head import load_draft_head
            self.head, self.head_index = load_draft_head(self._draft_head_path, self.eng.device)
        return self.module

    def _tokens_from(self, m, pred: torch.Tensor, anchor: int, first: int = 0) -> list[int]:
        """Rows 1.. of the block through the target's head, then the bigram bias, then argmax.

        The bias needs the token at the PREVIOUS slot, and at inference that is whatever this
        drafter proposes there. So the unbiased argmax is taken first and used as the previous
        tokens for one refinement pass -- one read of `markov_w2` for the whole block. Iterating
        to a fixed point would cost a read per iteration and the second iteration moved nothing
        in the A/B, so one is what ships.
        """
        from engine.model import head_logits, linear
        self._lattice = None
        if self.head is not None:
            logits = linear(pred, self.head)
        else:
            logits = head_logits(pred, self.eng.w.norm("lm_head.weight"))
        ids = logits.argmax(-1)
        if self.head_index is not None:
            ids = self.head_index[ids]
        latent = None
        if self.markov and self.cfg.markov_rank:
            prev = torch.cat([torch.tensor([anchor], device=ids.device, dtype=ids.dtype),
                              ids[:-1]])
            latent = m.markov_latent(prev)
            # `latent` is the checkpoint's dtype and `logits` may be fp32 off the fp8 head, so
            # the bias is computed in the weight's dtype and promoted, never the other way.
            biased = logits.float() + m.markov_bias(latent).float()
            ids = biased.argmax(-1)
            if self.head_index is not None:
                ids = self.head_index[ids]
        if self.cfg.enable_confidence_head:
            lat = latent if self.cfg.confidence_head_with_markov else None
            if lat is None and self.cfg.confidence_head_with_markov:
                lat = m.markov_latent(torch.cat([
                    torch.tensor([anchor], device=ids.device, dtype=ids.dtype), ids[:-1]]))
            self.last_conf = m.confidence(pred, lat).tolist()
        return [int(x) for x in ids]

    def propose_tree(self, context: list[int], budget: int = 16, **_):
        """A chain, always: with no candidate selector there is no second-best token to branch to.

        Returned as a `DraftTree` rather than `None` so that a `MergedRouter` can still graft the
        lookup drafter's tree onto it -- which is where a DSpark arm's width would come from.
        """
        from engine.tree import DraftTree
        chain = self.propose(context, self.cfg.block_size - 1)
        if not chain:
            return None
        return DraftTree.chain(int(context[-1]), chain, source="dspark")

    def draft_bytes(self, head_bytes: int | None = None) -> dict[str, float]:
        cfg = self.cfg
        exp = cfg.expected_tensors()
        backbone = sum(_prod(s) for n, s in exp.items()
                       if not n.startswith(("markov_head.", "confidence_head."))) * 2
        markov = (_prod(exp["markov_head.markov_w2.weight"]) * 2
                  if (self.markov and cfg.markov_rank) else 0)
        if head_bytes is None:
            h = self.eng.w.norm("lm_head.weight")
            head_bytes = h.numel() * h.element_size()
        return {"backbone_B": backbone, "markov_B": markov, "head_B": head_bytes,
                "total_B": backbone + markov + head_bytes,
                "proposals": cfg.block_size - 1}


def _prod(shape) -> int:
    n = 1
    for d in shape:
        n *= d
    return n


def resolve(path: str | None = None) -> str:
    return resolve_checkpoint(path or _S.get("DSPARK", DEFAULT_REPO))
