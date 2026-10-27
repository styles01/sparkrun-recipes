"""The model adapter's data types (ENG-221 contract, ENG-E8 layering): what a model tells the shared
methods about itself, as data the methods read instead of branching on the model.

An adapter is the engine object of one architecture with the verbs the server's loop and the caches
call (forward, forward_tree, forward_block, commit_tree, rollback_to, picks, hidden_post_norm, kv, state,
settle, discard_pending, trim_pending, reset, precapture), `caps` (`Capabilities`), `cost()` returning a
`CostModel` and `manifest()` (what its stored rows depend on, engine/methods/manifest.py). Hot verbs are the
engine's own bound methods, never wrapped; cold ones may be installed on the engine class by the model's
adapter module (`engine/models/qwen3_5/adapter.py` for the 27B, whose `engine/model.py` keeps its bytes).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Capabilities:
    """What the shared server may ask of a model's engine beyond the loop's verbs (SRV-341): the
    server turns off, and says so, what a model cannot hold yet."""

    caches: bool                # the prefix, session, resident and disk caches and the KV pool
    side_slots: bool            # `--parallel` side slots over the same weights
    anchor_grid: int | None = None   # the caches' grid (restore points, chunk files); None: --prefix-chunk
    # SRV-339: the shadow reader's outputs (the narrator's thought lines, the router's verdicts) were
    # validated on this model's reasoning; False: the reader is not started and the status says why
    shadow: bool = True
    # joint verify rounds: the decoding slots' blocks in one pass (`forward_tree_multi`); False: the
    # slots take turns, a whole round each (ENG-241 step 2, until the multi-sequence kernels)
    joint_verify: bool = True
    # the side lane (server/preempt.py): a short request served inside another one's generation on
    # the lent engine (`engine/cache.py` Borrow, which copies the 27B's pending commit); False: the
    # lane is off and such a request waits or takes an idle side slot
    side_lane: bool = True
    # a decoding slot's plain step (no draft this round) joins the round's joint pass as a one-row block,
    # so N slots taking plain steps read the weights once a round (ENG-241; a model with no draft head,
    # LFM2.5); False: a plain step runs alone, as before
    joint_plain: bool = False
    # ENG-E10: how the conductor's router fits this model (server/conductor/modelfit.py DEFAULTS
    # names the keys); None: the router as its documents say, written for Qwen3.8
    conductor: dict | None = None


@dataclass(frozen=True)
class CostModel:
    """Bytes the caches, the memory governor and the dashboard price a model by, from its state layout."""

    section_bytes: int              # the fixed state a snapshot copies whole
    view_bytes_per_token: int       # the position-indexed buffers a snapshot slices, a token
    parts: dict = field(default_factory=dict)   # section bytes by kind ("recurrent", "conv", ...)
    # a prefill's transient above what the engine held (the memory policy's `workspace`: the larger
    # of the floor and a line through zero); None: engine/cache.py's figures, measured on the 27B
    workspace_floor: int | None = None
    workspace_per_token: int | None = None
    workspace_cap: int | None = None   # where the workspace stops growing; None: it does not

    def snapshot_cost(self) -> dict:
        """The dashboard's `cache.snapshot_cost` row (contract 1.3 keys)."""
        return {"recurrent_bytes": int(self.parts.get("recurrent", 0)), "conv_bytes": int(self.parts.get("conv", 0)),
                "kv_bytes_per_token": int(self.view_bytes_per_token)}
