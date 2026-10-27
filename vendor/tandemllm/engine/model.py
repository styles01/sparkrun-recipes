"""The reference forward: correct, readable, and slow enough to be trusted.

Everything faster in this engine is checked against this file. It follows the published
implementation operation for operation, including where a cast to fp32 happens and which of the two
RMS norm conventions a given norm uses, because those are the details that silently cost accuracy:

  * `rms_norm`      -- input / post-attention / final / q / k -- is `normalise(x) * (1 + weight)`
  * `rms_norm_gated` -- the linear-attention output norm      -- is `weight * normalise(x) * silu(z)`

The weights are never dequantised into memory. Projections read the checkpoint's fp8 codes through
`tools.fp8_linear`, which applies the 128x128 block scale to the fp32 accumulator.
"""

from __future__ import annotations

import os
from engine.settings import SETTINGS as _S  # noqa: E402  (every QWEN38_* knob)
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import gdn  # noqa: E402
from engine.config import TextConfig  # noqa: E402
from engine.loader import Weights  # noqa: E402
from tools.fp8_linear import FP8Block, FP8Group, fp8_matmul  # noqa: E402
from tools.gdn_prefill_kernels import fused_prefill_refusal  # noqa: E402
from tools.head_gemv import FP8Head  # noqa: E402
from tools.exl3_linear import EXL3Block  # noqa: E402
from tools import exl3_linear as _x3  # noqa: E402
from tools.nvfp4_linear import NVFP4Block, nvfp4_matmul  # noqa: E402
from tools.nvfp4_linear_v2 import nvfp4_matmul_group  # noqa: E402
from tools import nvfp4_verify_tiles as _verify_tiles  # noqa: E402,F401  (registers on import)


# Fused kernels replacing parts of the reference forward. Off unless asked for: each one is a
# different arithmetic ORDER for the same quantity, so each has to earn its place against the gates
# (argmax agreement, greedy losslessness) and not only against a stopwatch. `tools/*_kernels.py`
# and `tools/head_gemv.py` each carry a `check()` against the function they replace.
# Four of the five are ON as of 11:55: each one is faster in the engine's own step or verify curve,
# and the whole set passes the losslessness gate of 10:07. `attn` is off because indexing the KV
# groups saves 0.6 ms at one token and costs 2.4 ms at eight. Set any of these to 0 to compare.
#: Skip the recurrent-state clone on a TREE verify, where nothing advances it; see `forward_tree`.
#:
#: OFF in the release candidate, and not because it failed anything. It was predicted at 1.7 ms
#: of a 134 ms block and it is not: measured at
#: 134.613 -> 134.344 ms, about 0.27 ms, which is inside the spread of repeated runs of the same
#: configuration. What it does have is the gates -- the losslessness gate PASSES with it on, and
#: `test_forward_tree`, `test_tree` and `test_cache` are identical either way -- and it removes
#: 151 MB of allocation and 302 MB of traffic per block that the tree path had no use for. It
#: would ship on being unnecessary work rather than on a stopwatch reading. It went into the RC
#: together with `tree_mode = nodes`, the row read differently, and both were reverted -- and then
#: re-running the reverted configuration unchanged read differently again, by more than either
#: change was worth. The row's run-to-run spread is 10 % of its mean when the policy is adaptive,
#: so it cannot resolve a change this size and did not. This is off in the RC because the RC was
#: soaked without it, not because anything caught it, and it goes to the `speed` branch where it
#: can be measured against several rows rather than one.
TREE_ALIAS_STATE = _S.get("TREE_ALIAS_STATE") == "1"

FUSED = {
    "norm": _S.get("FUSED_NORM") == "1",
    "gdn": _S.get("FUSED_GDN") == "1",
    "head": _S.get("FUSED_HEAD") == "1",
    "attn": _S.get("FUSED_ATTN") == "1",
    "gdnblock": _S.get("FUSED_GDNBLOCK") == "1",
    "gdnpre": _S.get("FUSED_GDNPRE") == "1",
    # The tree's counterpart to gdnblock. Same bet, same shape of kernel, one file over:
    # tools/gdn_tree_kernels.py walks the DFS pre-order carrying one factor per DEPTH, which is the
    # node's own ancestry, so the entry tile is read once as it is for a chain.
    "gdntree": _S.get("FUSED_GDNTREE") == "1",
    # The prefill counterpart. A different bet from the three above: those replace launches over
    # tiny tensors, this one replaces chunk-shaped fp32 TEMPORARIES -- about six gigabytes a layer
    # at 8k -- and the serial loop that walks them. Off by default until it is gated and measured.
    "gdnprefill": _S.get("FUSED_GDNPREFILL") == "1",
}

# "1" rank-k rollback, "0" the replay it replaces, "check" both with the difference recorded.
RANKK = _S.get("RANKK")

# The block's commit in one kernel (tools/gdn_commit_kernels.py), and no copy of
# the recurrent state on either side of it. Without it a chain verify clones the 151 MB state, walks
# it forward in place, and a partial accept -- nearly every block on new text -- copies the clone
# back over all of it and then rebuilds every layer from the clone with eight torch kernels a layer.
# With it the chain pass reads the entry state and writes its final state into a spare buffer
# (`GDNState.swap`), so the entry is still there for the commit, which reads it once and writes the
# live state once for all 48 layers. A tree verify, which never advances the state, commits in
# place. Off by default: the rank-k sum is taken in a different order from torch's matmul.
FUSED_COMMIT = _S.get("FUSED_COMMIT") == "1"

# The tree recurrence checked the tree's depth with `int(depths.max())`, a
# device-to-host read, in every one of the 48 linear-attention layers of every tree verify: 48
# synchronisations a block, each one draining the queue and leaving the GPU idle while the host
# launches the next layer. The tree's depths are already on the host (`TreeCtx.depth_list`).
TREE_HOST_DEPTH = _S.get("TREE_HOST_DEPTH") == "1"

# The recurrent half of a linear-attention layer over a verify block in four
# kernels (tools/gdn_verify_kernels.py) instead of about forty launches: the convolution with its
# SiLU, the gates, and the recurrence reading key heads by index and writing the commit's factors
# directly. Its convolution and `beta` round the way the decode step's do. Needs the rank-k
# commit (QWEN38_RANKK=1, the default), since it keeps no replay record.
FUSED_GDNVERIFY = _S.get("FUSED_GDNVERIFY") == "1"

# Each residual add and the RMS norm that reads it, in one launch
# (tools/norm_kernels.py::add_rms_norm): 128 adds a forward disappear, the numbers do not change.
FUSED_ADDNORM = _S.get("FUSED_ADDNORM") == "1"

# Serve the verify from CUDA graphs (engine/verify_graph.py): a replayed graph
# has no per-launch gaps. Needs the fused commit and GDN verify mixer, the decode-attention kernel
# and a bf16 KV cache; without them the eager verify runs.
VERIFY_GRAPH = _S.get("VERIFY_GRAPH") == "1"
#: SRV-321: () -> None, called by a layer-major window (`prefill_window`) between two layers, with
#: the EXL3 slab cache empty and off; the server sets it so a prefill on one slot offers the GPU
#: turn to the slots that decode every layer, not only after the window (an 8,192-row window is
#: one ~10 s pass, and three decoding chats wrote nothing for those 10 s). Whatever runs in it runs
#: on other engines: this window's ops, their order and their inputs stay the same, bit for bit.
WINDOW_YIELD = None

# Fold the block's commit into the next block's verify. Without it a commit is a
# second pass over the 151 MB recurrent state after every verify (read the entry, add the accepted
# rows' rank-k update, write the live state: 2.1 ms a block) and the next verify reads the state
# again. With it the commit is RECORDED (`_pend`: the accepted rows, and which of two static factor
# buffers the verify wrote), the live buffer keeps the entry, and the next verify's recurrence
# applies the record to each state tile in registers before its first row and writes the tile back
# once -- the commit kernel's arithmetic, op for op. The row count reaches the kernel through the
# device, so a captured verify graph serves blocks with and without a pending commit. A chain no
# longer writes its walked state: a full accept is a pending commit of every row. Anything else that
# reads the state -- a decode step, a prefill, a snapshot -- applies the record first with the
# commit kernel (`_settle`). Needs the fused commit and the fused GDN verify mixer; a full accept
# then takes the rank-k form rather than the walk, the arithmetic every partial accept already takes.
COMMIT_IN_VERIFY = _S.get("COMMIT_IN_VERIFY") == "1"

# The most rows a verify takes the fast path at: the fused GDN verify mixer for a
# chain, the fold, the verify graphs. 16 is the code as it was -- a 17-row chain fell to the chunked
# recurrence and every verify past 16 rows lost its graph and its fold, which is the cliff a wider
# tree and the deep chain paid. 32 raises all three together; the kernels behind
# them loop over the rows and were never limited to 16, only their callers were.
VERIFY_ROWS = int(_S.get("VERIFY_ROWS"))

# The served loop's host-to-device copies without a synchronisation, and fewer
# read-backs a round. `torch.tensor(list, device=cuda)` copies from pageable memory and PyTorch then
# synchronises the stream, so the host waited for every queued kernel five or six times a round
# (the block's tokens, the accepted path twice, a new tree shape's three tables) before it could
# queue the next launch. From pinned memory the copy is queued behind them (`h2d`). Also: the
# verify graph takes the argmax of its own logits, and the draft graph the lattice's
# log-probabilities, which the draft reads back together with the walk (one wait, not two). The
# same values everywhere; only when the host waits changes. Measured and left off (tools/loop_sync.py):
# 6.5 -> 2.0 synchronisations a round, ms a round unchanged
# -- the removed waits were on a queue that was already empty.
HOST_ASYNC = _S.get("HOST_ASYNC") == "1"


def h2d(values, dtype: torch.dtype, device) -> torch.Tensor:
    """A Python list as a device tensor. With HOST_ASYNC on a GPU: through pinned memory, queued
    behind the work already on the stream. The caching host allocator keeps the pinned block until
    the copy has run, so the list can go out of scope at once."""
    t = torch.tensor(values, dtype=dtype)
    if not HOST_ASYNC or torch.device(device).type != "cuda":
        return t.to(device)
    return t.pin_memory().to(device, non_blocking=True)


# An attention layer's q and k norms and partial rotary in two launches
# (tools/attn_prep.py) instead of about seventeen: the same arithmetic in the same order, bit for bit.
FUSED_ATTN_PREP = _S.get("FUSED_ATTN_PREP") == "1"

# The GDN gate inputs `a | b` through one fixed-order kernel (tools/small_linear.py) instead of two
# library GEMMs whose algorithm can differ inside a graph capture: the same bits in the eager
# verify, the graphed verify and the decode step.
GDN_AB = _S.get("GDN_AB") == "1"

# A chain-shaped tree is a chain, and the chain is 12.5 ms cheaper because it has a kernel the tree
# cannot use. So `forward_tree` hands one to `forward_block` and a drafter takes the cheaper price
# by proposing a line. Off only in the tests that have to exercise the tree path on a chain shape,
# where the delegation would make the comparison a tautology.
TREE_CHAIN_DELEGATE = _S.get("TREE_CHAIN_DELEGATE") == "1"

# The chunked delta rule's blocking on a prefill. The reference uses 64. It is a blocking choice,
# not a semantic one: the chunk loop is serial in Tp/chunk, and the intra-chunk work grows with the
# square of the chunk, so the best value is a measurement. See tools/profile_prefill.py --gdn-chunk.
GDN_PREFILL_CHUNK = int(_S.get("GDN_CHUNK"))
GDN_PREFILL_NOREP = _S.get("GDN_PREFILL_NOREP") == "1"

# The fused prefill pair is the one fused path that RAISES instead of degrading when its
# preconditions are not met -- it is built for chunk 64 and it needs Triton -- and it raises inside
# a prefill, which at 16k is minutes of work already spent. Ask once, here, and keep the reference
# chunked delta rule when the answer is no: it honours any chunk and its arithmetic is what the
# fused pair is checked against, so nothing about the numbers changes.
GDNPREFILL_REFUSAL = fused_prefill_refusal(GDN_PREFILL_CHUNK) if FUSED["gdnprefill"] else ""
if GDNPREFILL_REFUSAL:
    print(f"[engine] QWEN38_FUSED_GDNPREFILL=1 but {GDNPREFILL_REFUSAL}; prefills take the "
          f"reference chunked delta rule", flush=True)
FUSED_GDNPREFILL = FUSED["gdnprefill"] and not GDNPREFILL_REFUSAL

# Index the KV groups instead of materialising them from this many rows up.
GQA_FROM = int(_S.get("GQA_FROM"))

# Below GQA_FROM rows -- every decode step and every verify block -- attend with
# tools/attn_kernels.py, which reads each cached key and value once for the six query heads that
# share it, instead of SDPA over a `repeat_interleave`d copy of the whole context. Off by default:
# it is a different arithmetic order from the SDPA path, so it is quality-gated, not bit-gated.
DECODE_ATTN = _S.get("DECODE_ATTN") == "1"
# ... and the cache itself in e4m3 with one fp32 scale per (head, token): half the bytes of the
# bf16 cache, read by the same kernel. Implies DECODE_ATTN, which is the only reader of the codes;
# a prefill reads its own rows in bf16 and the cached ones dequantised.
KV_FP8 = _S.get("KV_FP8") == "1"
DECODE_ATTN = DECODE_ATTN or KV_FP8

# Tell SDPA a prefill is causal instead of handing it a [T, T] boolean. Set to 0 for the
# materialised mask the engine used until phase 4, which is the control this is measured against.
PREFILL_CAUSAL = _S.get("PREFILL_CAUSAL") == "1"

# The same argument one step further, for a prefill CHUNK. `engine/cache.py` splits a prefill so it
# can checkpoint, and every chunk after the first starts at `start > 0`, where the mask wanted is
# the causal triangle offset by `start` -- the BOTTOM-RIGHT alignment, which `is_causal` is not.
# Handing SDPA a materialised [T, ctx] boolean instead takes it off its fused backend and it was
# measured at 1.7-1.9x on a cold prefill. `causal_lower_right` says
# the same thing in a form the kernel keeps its backend for.
#
# From this many rows up, and no lower: a speculative verify block is 8 or 16 rows and it keeps the
# materialised mask it has been measured with all day, so no other track's number moves. 0 turns
# this off and restores the boolean everywhere, which is the control.
CHUNK_LOWER_RIGHT_FROM = int(_S.get("LOWER_RIGHT_FROM"))
ROLLBACK_DIFF: list = []

# Two streams over the projections that read the same activation and do not read each other's
# output. The engine reads its weights at 169 GB/s where a cold free-running chain of the same
# tensors reads 194, and phases 8 and 9 retired two explanations of the 25 between them: it is not
# launch overhead (144 launches removed bought 1.0 %, so a launch is about 7 microseconds) and it is
# not occupancy (fusing the widest pairs doubled the programs per launch and bought the same 1 %).
# The hypothesis left is that ONE weight read at a time does not keep enough loads in flight, which
# a second stream tests directly: two independent reads, issued together, no arithmetic changed and
# no kernel rewritten. Each pair is fenced with events on both sides, so the order of everything
# around it is exactly what it was.
#
# This is an alternative to `QWEN38_FUSE_PROJ`, not a companion: a fused group is already one
# launch, and there is nothing left to overlap it with.
TWO_STREAM = _S.get("TWO_STREAM") == "1"
_SIDE_STREAM: "torch.cuda.Stream | None" = None


def par2(first, second, reads=()):
    """`first()` on this stream and `second()` on a side one, joined before either is read.

    Both callables must read only tensors that already exist on the current stream and must not read
    each other's output; `reads` names the ones the side callable reads. `record_stream` on both
    sides is not decoration: the caching allocator is per stream, so a tensor produced on the side
    stream and freed on the main one can be handed out again while the side stream is still writing
    it, and an activation the main stream frees can be handed out while the side stream still reads.
    """
    global _SIDE_STREAM
    if not (TWO_STREAM and torch.cuda.is_available()):
        return first(), second()
    if _SIDE_STREAM is None:
        _SIDE_STREAM = torch.cuda.Stream()
    cur, side = torch.cuda.current_stream(), _SIDE_STREAM
    for t in reads:
        t.record_stream(side)
    ev = torch.cuda.Event()
    ev.record(cur)
    side.wait_event(ev)
    with torch.cuda.stream(side):
        b = second()
    a = first()
    done = torch.cuda.Event()
    done.record(side)
    cur.wait_event(done)
    if isinstance(b, tuple):
        for t in b:
            t.record_stream(cur)
    else:
        b.record_stream(cur)
    return a, b


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    if FUSED["norm"]:
        from tools.norm_kernels import rms_norm as fused
        return fused(x, weight, eps)
    out = x.float()
    out = out * torch.rsqrt(out.pow(2).mean(-1, keepdim=True) + eps)
    return (out * (1.0 + weight.float())).type_as(x)


def rms_norm_gated(x: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor,
                   eps: float) -> torch.Tensor:
    if FUSED["norm"]:
        from tools.norm_kernels import rms_norm_gated as fused
        return fused(x, gate, weight, eps)
    dt = x.dtype
    h = x.float()
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)
    h = weight * h.to(dt)
    h = h * F.silu(gate.float())
    return h.to(dt)


def head_logits(h: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """The vocabulary projection. The largest single read in the step -- see tools/head_gemv.py.

    ONE ROW ONLY. The kernel is a GEMV: it broadcasts the activation against the weight tile and
    accumulates in fp32 registers, which is the right shape at M = 1 and the wrong one at M = 8,
    where it does eight times the scalar arithmetic and never touches a tensor core. Measured at
    11:26: with this routed at M = 8 the verify pass went from 184 ms a block to 451. The library's
    GEMM owns everything above one row.
    """
    m = h.shape[-2] if h.dim() > 1 else 1
    if isinstance(weight, FP8Head):
        # e4m3 codes with a scale per vocabulary row: half the bytes, fp32 logits, and its own
        # kernel at both row counts -- a GEMV at one row and a GEMM above it, for the reason in the
        # docstring above.
        from tools.head_gemv import head_matmul_fp8
        return head_matmul_fp8(h, weight).view(*h.shape[:-1], weight.N)
    if FUSED["head"] and m == 1 and not isinstance(weight, (FP8Block, NVFP4Block, EXL3Block)):
        from tools.head_gemv import head_matmul
        return head_matmul(h, weight, bm=1).view(*h.shape[:-1], weight.shape[0])
    return linear(h, weight)


def matmul_group(x: torch.Tensor, g) -> torch.Tensor:
    """One launch for a fused projection group: an NVFP4Group, an FP8Group, or an EXL3 group."""
    if isinstance(g, FP8Group):
        return fp8_matmul(x, g)
    if isinstance(g, EXL3Block):
        return g.matmul(x)
    return nvfp4_matmul_group(x, g)


def linear(x: torch.Tensor, w) -> torch.Tensor:
    """`x @ w^T` for any weight format: a plain tensor is `F.linear`; every stored format
    (FP8Block, NVFP4Block, FP8Head, BF16Block, ...) implements the Linear interface's `matmul`,
    so a new format is a class and not another branch here (engine/linear.py)."""
    if isinstance(w, torch.Tensor):
        return F.linear(x, w)
    return w.matmul(x)


class KVCache:
    """One sequence, one contiguous buffer per attention layer, an append pointer per layer.

    Rolling a rejected speculative block back is the pointer moving back; nothing is copied.
    """

    def __init__(self, cfg: TextConfig, max_len: int, device: str, dtype=torch.bfloat16,
                 fp8: bool = False, paged: bool | None = None):
        n = len(cfg.attention_layers)
        self.slot = {l: i for i, l in enumerate(cfg.attention_layers)}
        self.fp8 = fp8
        # SRV-307 (the shared KV pool, engine/kvpool.py): with a pool, the window is reserved as
        # addresses and its pages are committed as rows are written (`ensure`); the tensors have
        # the static buffer's exact shape, dtype and strides, so every kernel and graph is the
        # same. `paged` None: the pool decides (on for a window of PAGED_MIN_ROWS and more).
        from engine import kvpool as _kvp
        self.pages = None
        if paged is None:
            paged = (_kvp.POOL is not None and not fp8 and int(max_len) >= _kvp.PAGED_MIN_ROWS
                     and str(device).startswith("cuda"))
        shape = (n, 1, cfg.num_key_value_heads, max_len, cfg.head_dim)
        if paged:
            self.pages = _kvp.PagedSet(_kvp.OWNER)
            self.k = self.pages.add(_kvp.PagedBuffer(shape, 3, dtype, device, owner=_kvp.OWNER,
                                                     name="kv.k")).tensor
            self.v = self.pages.add(_kvp.PagedBuffer(shape, 3, dtype, device, owner=_kvp.OWNER,
                                                     name="kv.v")).tensor
        else:
            self.k = torch.zeros(*shape, dtype=torch.float8_e4m3fn if fp8 else dtype,
                                 device=device)
            self.v = torch.zeros_like(self.k)
        # one fp32 scale per (head, token) when the codes are e4m3; see tools/attn_kernels.py
        self.ks = (torch.zeros(n, 1, cfg.num_key_value_heads, max_len, dtype=torch.float32,
                               device=device) if fp8 else None)
        self.vs = torch.zeros_like(self.ks) if fp8 else None
        self.length = 0
        self.max_len = max_len

    def ensure(self, rows: int) -> None:
        """SRV-307: commit the pages that hold rows [0, rows) (plus the pool's margin) before a
        write reaches them. One int compare when they are; nothing without a pool. Never inside
        a CUDA graph capture: the zeroing of fresh pages would be captured, and the caller
        committed the rows before it captured."""
        ps = self.pages
        if ps is None or rows + ps.margin() <= ps.rows:
            return
        if ps.on_device() and torch.cuda.is_current_stream_capturing():
            return
        ps.ensure(rows)

    def append(self, layer: int, k: torch.Tensor, v: torch.Tensor, start: int) -> tuple:
        i = self.slot[layer]
        t = k.shape[2]
        # a verify block that starts inside the room clamp and crosses `max_len` used to
        # die here as a torch shape mismatch (2026-09-18, twice: 16 rows into 12, 8 into 6),
        # because the loops clamp on OUTPUT tokens while this write counts ROWS -- a chain block
        # is `1 + len(draft)` rows (the anchor's own), a tree is all its nodes, and a tree's
        # rejected rows are written past `length` and merely never read. The loops now clamp on
        # rows; this guard is the second line of defence, so a path that misses the clamp fails
        # with a sentence rather than a traceback the log cannot attribute.
        if start + t > self.max_len:
            raise RuntimeError(
                f"verify block overruns the KV window: rows [{start}, {start + t}) "
                f"into a {self.max_len}-row buffer")
        if self.fp8:
            from tools.attn_kernels import quantize_kv
            kc, ksc = quantize_kv(k)
            vc, vsc = quantize_kv(v)
            self.k[i, :, :, start:start + t] = kc
            self.v[i, :, :, start:start + t] = vc
            self.ks[i, :, :, start:start + t] = ksc
            self.vs[i, :, :, start:start + t] = vsc
        else:
            self.k[i, :, :, start:start + t] = k
            self.v[i, :, :, start:start + t] = v
        return self.k[i, :, :, :start + t], self.v[i, :, :, :start + t]

    def scales(self, layer: int, n: int) -> tuple:
        """The e4m3 scales of the first `n` tokens, or (None, None) for a bf16 cache."""
        if not self.fp8:
            return None, None
        i = self.slot[layer]
        return self.ks[i, :, :, :n], self.vs[i, :, :, :n]

    def dequant(self, layer: int, n: int) -> tuple:
        """The first `n` cached tokens in bf16, for a prefill chunk that attends to them."""
        i = self.slot[layer]
        k = (self.k[i, :, :, :n].float() * self.ks[i, :, :, :n, None]).to(torch.bfloat16)
        v = (self.v[i, :, :, :n].float() * self.vs[i, :, :, :n, None]).to(torch.bfloat16)
        return k, v


class GDNState:
    """The recurrent and convolution state of every linear-attention layer, one sequence."""

    def __init__(self, cfg: TextConfig, device: str):
        n = len(cfg.linear_layers)
        self.slot = {l: i for i, l in enumerate(cfg.linear_layers)}
        self.S = torch.zeros(n, 1, cfg.linear_num_value_heads, cfg.linear_key_head_dim,
                             cfg.linear_value_head_dim, dtype=torch.float32, device=device)
        self.conv = torch.zeros(n, 1, cfg.conv_dim, cfg.linear_conv_kernel_dim - 1,
                                dtype=torch.bfloat16, device=device)
        self.primed = False
        self._spare: torch.Tensor | None = None

    def swap(self) -> torch.Tensor:
        """Make a spare buffer the live recurrent state and return the one it replaces.

        Nothing is copied: the returned tensor still holds the state as it was, which is what a
        chain verify needs to keep for its commit, and the pass writes its final state into the
        new live one. The two buffers trade places every block.
        """
        if self._spare is None:
            self._spare = torch.empty_like(self.S)
        entry, self.S, self._spare = self.S, self._spare, self.S
        return entry

    def clone(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.S.clone(), self.conv.clone()

    def restore(self, saved: tuple[torch.Tensor, torch.Tensor]) -> None:
        self.S.copy_(saved[0])
        self.conv.copy_(saved[1])

    @property
    def nbytes(self) -> int:
        return self.S.numel() * 4 + self.conv.numel() * 2


class BlockTrace:
    """What a speculative block has to remember so a partial accept can be undone.

    The recurrent state after k of B verified tokens was never materialised -- the chunked form
    produces outputs for every position and one final state. Snapshotting every prefix would move
    151 MB per position; recomputing the layer would re-read every weight. What is kept instead is
    the entry state and the per-token projections the recurrence consumes, which are three orders of
    magnitude smaller, and the replay then touches no weights at all.

    When every drafted token is accepted -- the case a good drafter produces most of the time --
    the final state the verify pass already wrote is correct and none of this is read.
    """

    def __init__(self):
        self.S_entry: torch.Tensor | None = None
        self.conv_entry: torch.Tensor | None = None
        self.layers: dict[int, tuple] = {}
        # The rank-k form: per layer, the normalised keys, the chunk's pseudo-values and its
        # cumulative gate. With these a partial accept is a weighted sum, not a replay -- see
        # `gdn.chunk_gated_delta_rule(return_factors=True)` for why they do not depend on how much
        # of the block is kept. `layers` stays as the fallback for a block that spans two chunks.
        self.factors: dict[int, tuple] = {}
        # Where the block was written, so a partial accept can put `kv.length` back.
        self.start = 0
        # the static factor buffers this verify wrote (a folding verify), else None
        self.fold_par: int | None = None

    @property
    def nbytes(self) -> int:
        n = 0
        for t in self.layers.values():
            n += sum(x.numel() * x.element_size() for x in t if x is not None)
        for x in (self.S_entry, self.conv_entry):
            if x is not None:
                n += x.numel() * x.element_size()
        return n


class TreeCtx:
    """Everything a tree verify needs that depends only on the tree's SHAPE, built once and cached.

    A draft tree changes three things in the forward pass and nothing else: which nodes a node may
    attend to (`anc_incl`), which columns its convolution reads (`conv_idx`), and what position it
    occupies for RoPE (`depths`, since a node's position is its depth, not its index). All three are
    functions of the parent array alone, so two steps that happen to draft the same shape -- which
    with a fixed node budget is most steps -- reuse them.
    """

    _cache: dict = {}

    def __init__(self, parents: tuple[int, ...], device: str, width: int):
        from engine.tree import DraftTree
        t = DraftTree(tokens=[0] * len(parents), parents=list(parents))
        n = len(parents)
        self.parents = parents
        self.n = n
        m = t.ancestor_mask()
        self.anc_incl = h2d(m, torch.bool, device)
        self.anc_strict = self.anc_incl & ~torch.eye(n, dtype=torch.bool, device=device)
        self.conv_idx = h2d(t.conv_windows(width), torch.long, device)
        self.depth_list = t.depths()
        self.depths = h2d(self.depth_list, torch.long, device)
        self.is_chain = list(parents) == [-1] + list(range(n - 1))

    @classmethod
    def get(cls, parents, device: str, width: int) -> "TreeCtx":
        key = (tuple(parents), device, width)
        ctx = cls._cache.get(key)
        if ctx is None:
            ctx = cls(tuple(parents), device, width)
            if len(cls._cache) > 256:
                cls._cache.clear()
            cls._cache[key] = ctx
        return ctx


class Qwen38Engine:
    def __init__(self, cfg: TextConfig, w: Weights, max_len: int = 8192, device: str = "cuda",
                 paged: bool | None = None):
        self.cfg = cfg
        self.w = w
        self.device = device
        self.max_len = max_len
        self.kv = KVCache(cfg, max_len, device, fp8=KV_FP8, paged=paged)
        self.state = GDNState(cfg, device)
        self._tril: dict[int, torch.Tensor] = {}
        self._rope_cache: tuple[torch.Tensor, torch.Tensor] | None = None
        # a request with images. `mm` (engine/vision.py MMContext) supplies the image
        # rows of the prompt and their three-axis rotary; `pos_delta` is the rotary offset of every
        # row after the prompt (index + delta). None and 0 for text, which leaves every forward
        # exactly as it was.
        self.mm = None
        self.pos_delta = 0
        self._rope_rows: tuple[torch.Tensor, torch.Tensor] | None = None
        self._trace: "BlockTrace | None" = None
        self.hidden_pre_norm: torch.Tensor | None = None
        self.hidden_post_norm: torch.Tensor | None = None
        self.tap = None  # set to a callable to receive every layer's hidden state
        self.trace: BlockTrace | None = None  # set during a speculative block verify
        self.tree: TreeCtx | None = None      # set during a speculative TREE verify
        self._tree: TreeCtx | None = None     # the one the last forward_tree ran
        self._tree_start = 0
        self._tree_is_chain = False
        self._gv = None               # set while a graph-safe verify body is being run/captured
        self._graphs = None           # engine.verify_graph.VerifyGraphs, built on first use
        self.graphs_off = False       # True: every verify eager, no graph pool (a side slot's choice)
        self.picks = None             # the last tree verify's argmax, when a graph took it
        self._pending_walk = False    # a chain's walked state waits in the scratch buffer
        # Once the verify graphs have been on, the recurrent state never changes buffers again (a
        # captured graph holds its address): a chain verify walks into this scratch buffer instead
        # of swapping, graphed or not. Allocated on first use.
        self._scratch_S: torch.Tensor | None = None
        self._walk_scratch = False    # the current chain trace walks into _scratch_S
        self._ab_cat: dict = {}       # layer prefix -> [in_proj_a; in_proj_b] for QWEN38_GDN_AB
        # the commit waiting to be applied to `state.S` in place -- ("static", parity, rows)
        # after a folding verify, which wrote one of two static sets of factor buffers, or
        # ("trace", factors by layer, rows) after any other -- the parity the next folding verify
        # writes, and the pending rows and their count on the device
        self._pend: tuple | None = None
        # what `trim_pending` needs to cut that commit short: (the pending record, the block's
        # entry convolution tails, its pre-convolution projections); stale once `_pend` changes
        self._pend_ctx: tuple | None = None
        self._fac: list | None = None
        self._par = 0
        self._fold_par: int | None = None      # while a folding verify runs, is captured or replayed
        self._prows: torch.Tensor | None = None
        self._pn: torch.Tensor | None = None
        self._pstage: torch.Tensor | None = None

    # ---------------------------------------------------------------- rotary
    def rope(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Partial rotary over the first `rotary_dim` of each head.

        The published model carries an interleaved 3-axis mRoPE for image and video grids. For text
        the three position rows are identical, so the interleave rewrites each row with a copy of
        itself and the result is ordinary RoPE, which is what this table holds. A prompt chunk that
        holds image rows takes its rows from engine/vision.py `mrope_cos_sin` instead,
        and every row after such a prompt reads this table at index + `pos_delta`.
        """
        if self._rope_cache is None:
            # Built once for every position the engine can reach, then gathered. The table was
            # being rebuilt from scratch on every attention layer of every step -- eight small
            # kernels, sixteen times a token, over numbers that never change. The values are
            # identical: the same product, computed for all rows at once instead of one row.
            dim = self.cfg.rotary_dim
            inv = 1.0 / (self.cfg.rope_theta ** (
                torch.arange(0, dim, 2, dtype=torch.float32, device=self.device) / dim))
            t = torch.arange(self.max_len + 64, dtype=torch.float32, device=self.device)
            freqs = t[:, None] * inv[None, :]
            emb = torch.cat([freqs, freqs], dim=-1)
            self._rope_cache = (emb.cos().to(torch.bfloat16), emb.sin().to(torch.bfloat16))
        cos, sin = self._rope_cache
        return cos[positions], sin[positions]

    def rope_inv_freq(self) -> torch.Tensor:
        """The inverse frequencies `rope()`'s table is built from, by the same expression on the
        same device, so a row computed from them (engine/vision.py `mrope_cos_sin`) equals the
        table's row bit for bit when its three coordinates are equal."""
        dim = self.cfg.rotary_dim
        return 1.0 / (self.cfg.rope_theta ** (
            torch.arange(0, dim, 2, dtype=torch.float32, device=self.device) / dim))

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        a, b = x.chunk(2, dim=-1)
        return torch.cat([-b, a], dim=-1)

    def apply_rope(self, q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor,
                   sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        d = cos.shape[-1]
        c = cos[None, None, :, :]
        s = sin[None, None, :, :]
        qr, qp = q[..., :d], q[..., d:]
        kr, kp = k[..., :d], k[..., d:]
        q = torch.cat([qr * c + self._rotate_half(qr) * s, qp], dim=-1)
        k = torch.cat([kr * c + self._rotate_half(kr) * s, kp], dim=-1)
        return q, k

    # ---------------------------------------------------------------- blocks
    def mlp(self, h: torch.Tensor, p: str) -> torch.Tensor:
        g = self.w.group(f"{p}.mlp.gate_up")
        if g is not None:
            # One launch for both halves. They read the same activation and neither reads the
            # other's output, so the only thing that made them two kernels was that they are two
            # names -- and two kernels is 272 programs apiece on 48 SMs, twice, with the board
            # draining between them.
            wd = self.w.proj(f"{p}.mlp.down_proj")
            if (_x3.MLP_BRIDGE and isinstance(g, EXL3Block) and isinstance(wd, EXL3Block)
                    and h.is_cuda and h.numel() // h.shape[-1] > _x3.CHUNK_UNTIL):
                # a prefill MLP: gate|up's fp32 products straight into down's rotated input
                x2 = h.reshape(-1, h.shape[-1])
                if x2.stride(1) != 1 or x2.stride(0) % 4 or x2.data_ptr() % 8:
                    x2 = x2.contiguous()
                out = _x3.exl3_mlp_slab(x2, g, wd)
                if out is not None:
                    return out.view(*h.shape[:-1], -1)
            y = matmul_group(h.reshape(-1, h.shape[-1]), g)
            if (isinstance(wd, EXL3Block) and _x3.SILU_FUSED and y.is_cuda
                    and g.sizes[0] == g.sizes[1] == wd.K):
                # silu(gate) * up inside the down projection's input rotation: the same bits
                return _x3.exl3_down_silu(y, wd).view(*h.shape[:-1], -1)
            gate, up = y.split(g.sizes, dim=-1)
            act = F.silu(gate) * up
            return linear(act, wd).view(*h.shape[:-1], -1)
        if TWO_STREAM:
            wg, wu = self.w.proj(f"{p}.mlp.gate_proj"), self.w.proj(f"{p}.mlp.up_proj")
            gate, up = par2(lambda: linear(h, wg), lambda: linear(h, wu), reads=(h,))
        else:
            # The flag is checked HERE, not inside `par2`, because the call site is what pays the
            # overhead: two lambda allocations and a call per projection per token measured ~7 % of
            # a decode step across 48 layers (interleaved A/B, 2026-09-20 night), and a closed
            # flag must cost exactly nothing.
            gate = linear(h, self.w.proj(f"{p}.mlp.gate_proj"))
            up = linear(h, self.w.proj(f"{p}.mlp.up_proj"))
        return linear(F.silu(gate) * up, self.w.proj(f"{p}.mlp.down_proj"))

    def _attn_mask(self, T: int, ctx: int, start: int, device) -> torch.Tensor:
        """What each of the T rows may attend to across the whole `ctx`-long cache.

        Straight-line decoding wants the causal triangle offset by `start`. A tree wants the
        committed prefix for everybody and, inside the block, each node's own ancestors -- the same
        relation the linear layers get, expressed as an attention mask. The KV cache is written in
        DFS order, so column `start + j` is node j and the block half of the mask is exactly
        `ancestor_incl`.
        """
        if self.tree is None:
            return torch.ones(T, ctx, dtype=torch.bool, device=device).tril(start)
        m = torch.zeros(T, ctx, dtype=torch.bool, device=device)
        if start:
            m[:, :start] = True
        m[:, start:start + self.tree.n] = self.tree.anc_incl
        return m

    def attention(self, h: torch.Tensor, p: str, layer: int, start: int,
                  positions: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        B, T, _ = h.shape
        grp = self.w.group(f"{p}.self_attn.qkv")
        if grp is not None:
            # q is twelve times the size of k or v, and k and v were each putting sixteen programs
            # on the board -- a launch that cannot fill it at any tiling. Together they are one.
            y = matmul_group(h.reshape(-1, h.shape[-1]), grp)
            qy, ky, vy = y.split(grp.sizes, dim=-1)
            # a column slice is not contiguous, and `reshape` is where the copy is paid: 459 kB a
            # layer at the block's row count, which is 0.04 ms a step across all sixteen
            qg = qy.reshape(B, T, cfg.num_attention_heads, cfg.head_dim * 2)
            kk_ = ky.reshape(B, T, cfg.num_key_value_heads, cfg.head_dim)
            vv_ = vy.reshape(B, T, cfg.num_key_value_heads, cfg.head_dim)
        else:
            # q is twelve times the size of k and v together, so the split that balances the two
            # streams is q against the pair of them.
            if TWO_STREAM:
                wq = self.w.proj(f"{p}.self_attn.q_proj")
                wk, wv = self.w.proj(f"{p}.self_attn.k_proj"), self.w.proj(f"{p}.self_attn.v_proj")
                qy, (ky, vy) = par2(lambda: linear(h, wq),
                                    lambda: (linear(h, wk), linear(h, wv)), reads=(h,))
            else:
                qy = linear(h, self.w.proj(f"{p}.self_attn.q_proj"))
                ky = linear(h, self.w.proj(f"{p}.self_attn.k_proj"))
                vy = linear(h, self.w.proj(f"{p}.self_attn.v_proj"))
            qg = qy.view(B, T, cfg.num_attention_heads, cfg.head_dim * 2)
            kk_ = ky.view(B, T, cfg.num_key_value_heads, cfg.head_dim)
            vv_ = vy.view(B, T, cfg.num_key_value_heads, cfg.head_dim)
        q, gate = qg.chunk(2, dim=-1)
        gate = gate.reshape(B, T, -1)
        v = vv_.transpose(1, 2)
        if FUSED_ATTN_PREP and FUSED["norm"] and B == 1 and q.is_cuda and self._rope_rows is None:
            from tools.attn_prep import attn_prep
            if self._rope_cache is None:
                self.rope(positions[:1])
            q, k = attn_prep(q[0], kk_[0], self.w.norm(f"{p}.self_attn.q_norm.weight"),
                             self.w.norm(f"{p}.self_attn.k_norm.weight"), *self._rope_cache,
                             positions, cfg.rms_norm_eps)
        else:
            q = rms_norm(q, self.w.norm(f"{p}.self_attn.q_norm.weight"),
                         cfg.rms_norm_eps).transpose(1, 2)
            k = rms_norm(kk_, self.w.norm(f"{p}.self_attn.k_norm.weight"),
                         cfg.rms_norm_eps).transpose(1, 2)
            rr = self._rope_rows
            cos, sin = self.rope(positions) if rr is None else rr
            q, k = self.apply_rope(q, k, cos, sin)
        rep = cfg.num_attention_heads // cfg.num_key_value_heads
        if self._gv is not None:
            # the graph-safe verify: a scatter at the device slots, the whole cache, the length
            # from the device
            from tools.attn_kernels import decode_attention_dev
            i = self.kv.slot[layer]
            slots = self._gv.slots[T]
            self.kv.k[i].index_copy_(2, slots, k.to(self.kv.k.dtype))
            self.kv.v[i].index_copy_(2, slots, v.to(self.kv.v.dtype))
            if self.tree is not None:
                bm = self.tree.anc_incl
            else:
                bm = self._tril.get(T)
                if bm is None:
                    bm = self._tril[T] = torch.ones(T, T, dtype=torch.bool,
                                                    device=h.device).tril()
            o = decode_attention_dev(q, self.kv.k[i], self.kv.v[i], self._gv.lenp, bm,
                                     self._gv.max_lc)
            o = o.transpose(1, 2).reshape(B, T, -1)
            o = o * torch.sigmoid(gate)
            return linear(o, self.w.proj(f"{p}.self_attn.o_proj"))
        kk, vv = self.kv.append(layer, k, v, start)
        if DECODE_ATTN and T < GQA_FROM:
            from tools.attn_kernels import decode_attention
            if self.tree is not None:
                bm = self.tree.anc_incl
            else:
                bm = self._tril.get(T)
                if bm is None:
                    bm = self._tril[T] = torch.ones(T, T, dtype=torch.bool,
                                                    device=h.device).tril()
            ks, vs = self.kv.scales(layer, start + T)
            o = decode_attention(q, kk, vv, start, bm, ks=ks, vs=vs)
            o = o.transpose(1, 2).reshape(B, T, -1)
            o = o * torch.sigmoid(gate)
            return linear(o, self.w.proj(f"{p}.self_attn.o_proj"))
        if self.kv.fp8:
            # A prefill: its own rows in bf16 as computed, the cached ones dequantised.
            if start:
                pk, pv = self.kv.dequant(layer, start)
                kk, vv = torch.cat([pk, k], dim=2), torch.cat([pv, v], dim=2)
            else:
                kk, vv = k, v
        # A prefill is the one case where the mask is the plain causal triangle over the whole
        # context, and saying so instead of handing the kernel a [T, T] boolean is not a
        # micro-optimisation: a materialised mask takes SDPA off its fused backend, and the
        # fallback writes a [24, T, T] score matrix -- 3.2 GB at T = 8,192, per layer, sixteen
        # times. `is_causal` aligns to the top-left, which is what `start == 0` means.
        causal = PREFILL_CAUSAL and T > 1 and self.tree is None and start == 0
        if causal or T == 1:
            mask = None
        elif (self.tree is None and CHUNK_LOWER_RIGHT_FROM
                and T >= CHUNK_LOWER_RIGHT_FROM):
            from torch.nn.attention.bias import causal_lower_right
            mask = causal_lower_right(T, kk.shape[2])
        else:
            mask = self._attn_mask(T, kk.shape[2], start, h.device)
        if FUSED["attn"] or T >= GQA_FROM:
            # `repeat_interleave` materialises the whole context six times over, once per query
            # group: at 4k of context that is 200 MB written and read again per token across the
            # sixteen attention layers, for data the kernel can index instead. `enable_gqa` lets
            # it index. It loses 2.4 ms on a verify block of eight (11:26) and wins on a long
            # sequence, where the copy is 100 MB a layer, so the row count decides.
            o = F.scaled_dot_product_attention(q, kk, vv, attn_mask=mask, is_causal=causal,
                                               enable_gqa=True)
        else:
            kk = kk.repeat_interleave(rep, dim=1)
            vv = vv.repeat_interleave(rep, dim=1)
            o = F.scaled_dot_product_attention(q, kk, vv, attn_mask=mask, is_causal=causal)
        o = o.transpose(1, 2).reshape(B, T, -1)
        o = o * torch.sigmoid(gate)
        return linear(o, self.w.proj(f"{p}.self_attn.o_proj"))

    def linear_attention(self, h: torch.Tensor, p: str, layer: int,
                         use_state: bool) -> torch.Tensor:
        cfg = self.cfg
        B, T, _ = h.shape
        i = self.state.slot[layer]
        if (T == 1 and use_state and self.tree is None and self.trace is None
                and FUSED["gdn"] and FUSED["gdnpre"]):
            return self._linear_attention_decode(h, p, i)
        if (FUSED_GDNVERIFY and RANKK == "1" and use_state and B == 1 and self.trace is not None
                and ((self.tree is not None and FUSED["gdntree"] and T <= 64)
                     or (self.tree is None and FUSED["gdnblock"] and T <= VERIFY_ROWS))):
            return self._linear_attention_verify(h, p, layer, i)
        grp = self.w.group(f"{p}.linear_attn.qkvz")
        z_pre = None
        if grp is not None:
            y = matmul_group(h.reshape(-1, h.shape[-1]), grp)
            qkv_y, z_y = y.split(grp.sizes, dim=-1)
            mixed = qkv_y.reshape(B, T, -1).transpose(1, 2)
            z_pre = z_y.reshape(B, T, -1)
        else:
            # `z` is read at the far end of this method, after the convolution and the recurrence
            # have run, but it depends on nothing but `h`. Computing it HERE is what lets it share
            # the board with the projection it used to queue behind.
            if TWO_STREAM:
                wqkv, wz = (self.w.proj(f"{p}.linear_attn.in_proj_qkv"),
                            self.w.proj(f"{p}.linear_attn.in_proj_z"))
                qkv_y, z_y = par2(lambda: linear(h, wqkv), lambda: linear(h, wz), reads=(h,))
                mixed = qkv_y.transpose(1, 2)
                z_pre = z_y
            else:
                mixed = linear(h, self.w.proj(f"{p}.linear_attn.in_proj_qkv")).transpose(1, 2)
        raw = mixed if self.trace is not None else None
        cw = self.w.norm(f"{p}.linear_attn.conv1d.weight").squeeze(1)
        if self.tree is not None:
            # A tree has no single successor convolution state, so this does NOT advance the one in
            # `self.state`; `commit_tree` writes the accepted path's tail instead.
            mixed = gdn.conv_tree(mixed, self.state.conv[i], cw, self.tree.conv_idx)
        elif use_state:
            mixed = gdn.conv_update(mixed, self.state.conv[i], cw)
        else:
            mixed, new_conv = gdn.conv_prefill(mixed, cw)
            self.state.conv[i].copy_(new_conv)
        mixed = mixed.transpose(1, 2)
        q, k, v = mixed.split([cfg.key_dim, cfg.key_dim, cfg.value_dim], dim=-1)
        q = q.view(B, T, cfg.linear_num_key_heads, cfg.linear_key_head_dim)
        k = k.view(B, T, cfg.linear_num_key_heads, cfg.linear_key_head_dim)
        v = v.view(B, T, cfg.linear_num_value_heads, cfg.linear_value_head_dim)
        z = (z_pre if z_pre is not None
             else linear(h, self.w.proj(f"{p}.linear_attn.in_proj_z"))).view(
            B, T, cfg.linear_num_value_heads, cfg.linear_value_head_dim)
        b = linear(h, self.w.norm(f"{p}.linear_attn.in_proj_b.weight"))
        a = linear(h, self.w.norm(f"{p}.linear_attn.in_proj_a.weight"))
        beta = b.sigmoid()
        A_log = self.w.norm(f"{p}.linear_attn.A_log")
        dt_bias = self.w.norm(f"{p}.linear_attn.dt_bias")
        g = -A_log.float().exp() * F.softplus(a.float() + dt_bias.float())
        rep = cfg.num_v_per_k
        # the fused prefill reads q and k at key head h // rep itself (GDN_PREFILL_NOREP): the
        # widened copies and two thirds of the l2 norms are not made; every value is the same
        norep = (GDN_PREFILL_NOREP and FUSED_GDNPREFILL and self.tree is None and self.trace is None
                 and T >= GDN_PREFILL_CHUNK and B == 1)
        if rep > 1 and not norep:
            q = q.repeat_interleave(rep, dim=2)
            k = k.repeat_interleave(rep, dim=2)
        if self.trace is not None:
            # q, k and v are post-convolution and post repeat-interleave, so a prefix of them is
            # exactly what the recurrence for that prefix consumes; `raw` is what the convolution
            # state has to be rebuilt from.
            if FUSED_COMMIT and RANKK == "1":
                # the fused commit reads the factors and the raw projections, nothing else, and
                # none of these tensors is written again: references, not six copies a layer
                self.trace.layers[layer] = (raw[0], None, None, None, None, None)
            else:
                self.trace.layers[layer] = (raw[0].clone(), q.clone(), k.clone(), v.clone(),
                                            g.clone(), beta.clone())
        if self.tree is not None and FUSED["gdntree"] and use_state and T <= 64:
            # The state tile is loaded once and the node's ancestry is carried in registers, one
            # factor per depth. Everything the chunked path returns, the same three buffers.
            from tools.gdn_tree_kernels import fused_tree_step
            o, delta, gc = fused_tree_step(q, k, v, g, beta, self.tree.depths,
                                           self.state.S[i],
                                           max_depth=(max(self.tree.depth_list)
                                                      if TREE_HOST_DEPTH else None))
            self.trace.factors[layer] = (
                gdn.l2norm(k.float(), dim=-1).transpose(1, 2).contiguous(),
                delta.transpose(1, 2).contiguous(),
                gc.transpose(1, 2).contiguous())
        elif self.tree is not None:
            # The state is deliberately left at its entry value: the chunked form's final state sums
            # over every node, which on a tree mixes branches that never coexist. The accepted
            # path's state is reconstructed from the factors in `commit_tree`.
            o, _, fac = gdn.chunk_gated_delta_rule(
                q, k, v, g, beta, self.state.S[i], chunk_size=T, return_factors=True,
                tree=(self.tree.anc_incl, self.tree.anc_strict))
            if fac is None:
                raise RuntimeError("tree verify did not get its factors back")
            self.trace.factors[layer] = fac
        elif use_state and T == 1 and FUSED["gdn"]:
            from tools.gdn_kernels import fused_decode_step
            o = fused_decode_step(q, k, v, g, beta, self.state.S[i])
        elif use_state and T == 1:
            o, _ = gdn.recurrent_gated_delta_rule(q, k, v, g, beta, self.state.S[i])
        elif self.trace is not None and FUSED["gdnblock"] and use_state and T <= 16:
            # A verify block is eight tokens, and eight tokens of this recurrence fit in registers.
            # The chunked form exists for sequences that do not fit: it turns the recurrence into
            # matrix work, pays a serial matrix inverse for the intra-chunk term, and reads the
            # state several times. Walking the state forward eight times in one kernel reads it
            # once. The per-token update vectors come back with it, so the rank-k rollback needs
            # nothing extra.
            from tools.gdn_kernels import fused_block_step
            if FUSED_COMMIT:
                # the entry state stays where `forward_block` left it; the walk lands in the spare
                o, delta = fused_block_step(q, k, v, g, beta, self.trace.S_entry[i],
                                            out_state=self.state.S[i])
            else:
                o, delta = fused_block_step(q, k, v, g, beta, self.state.S[i])
            self.trace.factors[layer] = (
                gdn.l2norm(k.float(), dim=-1).transpose(1, 2).contiguous(),
                delta.transpose(1, 2).contiguous(),
                g.float().cumsum(dim=1).transpose(1, 2).contiguous())
        elif self.trace is not None:
            s_in = (self.trace.S_entry[i] if FUSED_COMMIT else self.state.S[i]) if use_state \
                else None
            o, S, fac = gdn.chunk_gated_delta_rule(
                q, k, v, g, beta, s_in,
                chunk_size=GDN_PREFILL_CHUNK if T > GDN_PREFILL_CHUNK else max(2, T),
                return_factors=True)
            self.state.S[i].copy_(S)
            if fac is not None:
                self.trace.factors[layer] = fac
        elif FUSED_GDNPREFILL and T >= GDN_PREFILL_CHUNK and B == 1:
            # A prefill is the one call where the reference's whole-tensor form is expensive: at 8k
            # it writes six gigabytes of chunk-shaped fp32 temporaries per layer and runs a serial
            # loop of 128 iterations, for 71 GFLOP of arithmetic. The fused pair keeps the same
            # arithmetic in the same order and writes two of those tensors instead of nine.
            from tools.gdn_prefill_kernels import fused_chunk_prefill
            o, S = fused_chunk_prefill(q, k, v, g, beta,
                                       self.state.S[i] if use_state else None,
                                       chunk_size=GDN_PREFILL_CHUNK)
            self.state.S[i].copy_(S)
        else:
            o, S = gdn.chunk_gated_delta_rule(
                q, k, v, g, beta, self.state.S[i] if use_state else None,
                chunk_size=GDN_PREFILL_CHUNK if T > GDN_PREFILL_CHUNK else max(2, T))
            self.state.S[i].copy_(S)
        o = rms_norm_gated(o.reshape(-1, cfg.linear_value_head_dim),
                           z.reshape(-1, cfg.linear_value_head_dim),
                           self.w.norm(f"{p}.linear_attn.norm.weight"), cfg.rms_norm_eps)
        o = o.view(B, T, -1)
        return linear(o, self.w.proj(f"{p}.linear_attn.out_proj"))

    def _linear_attention_verify(self, h: torch.Tensor, p: str, layer: int,
                                 i: int) -> torch.Tensor:
        """A verify block's mixer: the projections, then tools/gdn_verify_kernels.py.

        Same inputs and outputs as the general path above for a chain (the conv state advanced,
        the walked state written) or a tree (neither), with the trace holding the factors and the
        raw projections -- which is all the rank-k commit reads.
        """
        from tools.gdn_verify_kernels import verify_mixer
        cfg = self.cfg
        B, T, _ = h.shape
        flat = h.reshape(-1, h.shape[-1])
        grp = self.w.group(f"{p}.linear_attn.qkvz")
        if grp is not None:
            mixed, z = matmul_group(flat, grp).split(grp.sizes, dim=-1)
        else:
            mixed = linear(flat, self.w.proj(f"{p}.linear_attn.in_proj_qkv"))
            z = linear(flat, self.w.proj(f"{p}.linear_attn.in_proj_z"))
        a, b = self._gate_inputs(flat, p)
        tree = self.tree
        chain_swap = tree is None and FUSED_COMMIT and not self._walk_scratch
        pend, store, fac_out = None, True, None
        fp = self.trace.fold_par
        if fp is not None:
            # the factors into this parity's static buffers, the pending commit (if the
            # device says there is one) from the other's
            fac, other = self._fac[fp], self._fac[1 - fp]
            fac_out = (fac[0][i], fac[1][i], fac[2][i])
            pend = (other[0][i], other[1][i], other[2][i], self._prows, self._pn)
        if tree is None and fp is not None:
            # the entry is the live buffer (the pending commit lands in it), and the walk is not
            # stored -- this block's own commit, full or partial, will be pending too
            s_in, s_out, store = self.state.S[i], None, False
        elif tree is None and self._walk_scratch:
            # graph-compatible chain: the entry stays the live buffer, the walk goes to
            # the scratch buffer, whose address every captured graph also holds
            s_in, s_out = self.state.S[i], self._scratch_S[i]
        else:
            s_in = self.trace.S_entry[i] if chain_swap else self.state.S[i]
            s_out = self.state.S[i] if chain_swap else None
        o, fac, _ = verify_mixer(
            mixed, self.state.conv[i], self.w.norm(f"{p}.linear_attn.conv1d.weight").squeeze(1),
            a, b, self.w.norm(f"{p}.linear_attn.A_log"), self.w.norm(f"{p}.linear_attn.dt_bias"),
            s_in,
            key_dim=cfg.key_dim, key_heads=cfg.linear_num_key_heads,
            value_heads=cfg.linear_num_value_heads, head_k=cfg.linear_key_head_dim,
            head_v=cfg.linear_value_head_dim,
            window=tree.conv_idx if tree is not None else None,
            depths=tree.depths if tree is not None else None,
            max_depth=max(tree.depth_list) if tree is not None else 0,
            out_state=s_out, pend=pend, store_state=store, fac_out=fac_out,
            anc=tree.anc_incl if tree is not None else None)
        self.trace.factors[layer] = fac
        # the pre-convolution projections, [C, T] as the commit reads them; a view, never written
        self.trace.layers[layer] = (mixed.t(), None, None, None, None, None)
        o = rms_norm_gated(o.reshape(-1, cfg.linear_value_head_dim),
                           z.reshape(-1, cfg.linear_value_head_dim),
                           self.w.norm(f"{p}.linear_attn.norm.weight"), cfg.rms_norm_eps)
        return linear(o.view(B, T, -1), self.w.proj(f"{p}.linear_attn.out_proj"))

    def _gate_inputs(self, flat: torch.Tensor, p: str) -> tuple[torch.Tensor, torch.Tensor]:
        """The two 48-wide gate projections, `a` and `b`, of `flat` [T, H]."""
        if GDN_AB:
            from tools.small_linear import small_linear
            w = self._ab_cat.get(p)
            if w is None:
                w = self._ab_cat[p] = torch.cat(
                    [self.w.norm(f"{p}.linear_attn.in_proj_a.weight"),
                     self.w.norm(f"{p}.linear_attn.in_proj_b.weight")]).contiguous()
            ab = small_linear(flat, w)
            n = w.shape[0] // 2
            return ab[:, :n], ab[:, n:]
        return (linear(flat, self.w.norm(f"{p}.linear_attn.in_proj_a.weight")),
                linear(flat, self.w.norm(f"{p}.linear_attn.in_proj_b.weight")))

    def _linear_attention_decode(self, h: torch.Tensor, p: str, i: int) -> torch.Tensor:
        """The T = 1 mixer with the glue in two kernels instead of a dozen.

        Same arithmetic as the general path above, in the same order; what is gone is the launches.
        The convolution, its SiLU and the state shift are one kernel, `g` and `beta` are one more,
        and the key side is not widened to forty-eight heads at all -- the recurrence kernel indexes
        it. Nine launches where the general path issues about twenty, on a mixer whose useful work
        is a 64.9 MB weight read.

        It runs only when there is no trace and no tree: a verify pass needs the pre-convolution
        projections kept for the rollback, and a tree has no single convolution successor.
        """
        from tools.gdn_kernels import decode_pre, fused_decode_step
        cfg = self.cfg
        if TWO_STREAM:
            wqkv, wz = (self.w.proj(f"{p}.linear_attn.in_proj_qkv"),
                        self.w.proj(f"{p}.linear_attn.in_proj_z"))
            mixed_y, z_y = par2(lambda: linear(h, wqkv), lambda: linear(h, wz), reads=(h,))
        else:
            mixed_y = linear(h, self.w.proj(f"{p}.linear_attn.in_proj_qkv"))
            z_y = linear(h, self.w.proj(f"{p}.linear_attn.in_proj_z"))
        mixed = mixed_y.reshape(-1)
        z = z_y.view(1, 1, cfg.linear_num_value_heads, cfg.linear_value_head_dim)
        if GDN_AB:
            a, b = self._gate_inputs(h.reshape(1, -1), p)
            a, b = a.reshape(-1), b.reshape(-1)
        else:
            b = linear(h, self.w.norm(f"{p}.linear_attn.in_proj_b.weight")).reshape(-1)
            a = linear(h, self.w.norm(f"{p}.linear_attn.in_proj_a.weight")).reshape(-1)
        qkv, g, beta = decode_pre(
            mixed, self.state.conv[i],
            self.w.norm(f"{p}.linear_attn.conv1d.weight").squeeze(1),
            a, b, self.w.norm(f"{p}.linear_attn.A_log"),
            self.w.norm(f"{p}.linear_attn.dt_bias"))
        kd, vd = cfg.key_dim, cfg.value_dim
        q = qkv[:kd].view(1, 1, cfg.linear_num_key_heads, cfg.linear_key_head_dim)
        k = qkv[kd:2 * kd].view(1, 1, cfg.linear_num_key_heads, cfg.linear_key_head_dim)
        v = qkv[2 * kd:2 * kd + vd].view(1, 1, cfg.linear_num_value_heads,
                                         cfg.linear_value_head_dim)
        o = fused_decode_step(q, k, v, g, beta, self.state.S[i], rep=cfg.num_v_per_k)
        o = rms_norm_gated(o.reshape(-1, cfg.linear_value_head_dim),
                           z.reshape(-1, cfg.linear_value_head_dim),
                           self.w.norm(f"{p}.linear_attn.norm.weight"), cfg.rms_norm_eps)
        return linear(o.view(1, 1, -1), self.w.proj(f"{p}.linear_attn.out_proj"))

    # ---------------------------------------------------------------- forward
    def forward(self, tokens: torch.Tensor, start: int = 0, *,
                last_only: bool = False) -> torch.Tensor:
        T = tokens.numel()
        self.kv.ensure(start + T)                # SRV-307: the rows this forward writes
        if self.trace is None and self._gv is None:
            self._settle()
        h = F.embedding(tokens.view(1, T), self.w.norm("embed_tokens.weight"))
        mm = self.mm
        # A prompt chunk only: a verify (a block, a tree, a graph's body, which a capture runs at
        # start 0 whatever the request) is never a prompt row, and must not read the images
        if (mm is not None and start < mm.n and self._gv is None and self.trace is None
                and self.tree is None):
            return self._forward_mm(mm, h, start, T, last_only)
        if self.tap is not None:
            self.tap(h[0].detach())
        # A tree node's POSITION is its depth, not its index: two siblings are both the token
        # after their parent and both carry that position, which is what makes a tree a set of
        # alternative continuations rather than a longer sequence. `pos_delta` is 0 unless the
        # prompt held an image.
        if self._gv is not None:
            positions = self._gv.pos[T]
        else:
            s0 = start + self.pos_delta
            positions = (s0 + self.tree.depths) if self.tree is not None else \
                torch.arange(s0, s0 + T, device=self.device)
        return self._forward_layers(h, start, positions, T, last_only)

    def _forward_mm(self, mm, h, start: int, T: int, last_only: bool) -> torch.Tensor:
        """A prompt chunk of a request with images: the image rows' embeddings and,
        when the chunk holds any, their three-axis rotary, then the same layers."""
        h, positions, rows = mm.rows(self, h, start, T)
        if self.tap is not None:
            self.tap(h[0].detach())
        self._rope_rows = rows
        try:
            return self._forward_layers(h, start, positions, T, last_only)
        finally:
            self._rope_rows = None

    def _forward_layers(self, h, start: int, positions, T: int, last_only: bool) -> torch.Tensor:
        cfg = self.cfg
        use_state = self.state.primed
        if FUSED_ADDNORM and FUSED["norm"]:
            return self._forward_addnorm(h, start, positions, use_state, T, last_only)
        for layer in range(cfg.num_hidden_layers):
            p = f"layers.{layer}"
            res = h
            x = rms_norm(h, self.w.norm(f"{p}.input_layernorm.weight"), cfg.rms_norm_eps)
            if cfg.is_linear(layer):
                x = self.linear_attention(x, p, layer, use_state)
            else:
                x = self.attention(x, p, layer, start, positions)
            h = res + x
            res = h
            x = rms_norm(h, self.w.norm(f"{p}.post_attention_layernorm.weight"), cfg.rms_norm_eps)
            h = res + self.mlp(x, p)
            if self.tap is not None:
                self.tap(h[0].detach())
        self.state.primed = True
        self.kv.length = start + T
        self.hidden_pre_norm = h
        h = rms_norm(h, self.w.norm("norm.weight"), cfg.rms_norm_eps)
        self.hidden_post_norm = h
        if last_only:
            h = h[:, -1:]
        return head_logits(h, self.w.norm("lm_head.weight"))

    def _forward_addnorm(self, h, start, positions, use_state, T, last_only):
        """`forward`'s layer loop with every residual add fused into the norm that reads it.

        The same tensors in the same order: the input norm of layer l + 1 and the final norm read
        the sum the add wrote, so the add's rounding is where it was. The drafter's tap still
        receives each layer's residual stream, which the fused kernel writes."""
        from tools.norm_kernels import add_rms_norm
        cfg = self.cfg
        n = cfg.num_hidden_layers
        x = rms_norm(h, self.w.norm("layers.0.input_layernorm.weight"), cfg.rms_norm_eps)
        for layer in range(n):
            p = f"layers.{layer}"
            if cfg.is_linear(layer):
                a = self.linear_attention(x, p, layer, use_state)
            else:
                a = self.attention(x, p, layer, start, positions)
            h, x = add_rms_norm(h, a, self.w.norm(f"{p}.post_attention_layernorm.weight"),
                                cfg.rms_norm_eps)
            m = self.mlp(x, p)
            nxt = (f"layers.{layer + 1}.input_layernorm.weight" if layer + 1 < n
                   else "norm.weight")
            h, x = add_rms_norm(h, m, self.w.norm(nxt), cfg.rms_norm_eps)
            if self.tap is not None:
                self.tap(h[0].detach())
        self.state.primed = True
        self.kv.length = start + T
        self.hidden_pre_norm = h
        self.hidden_post_norm = x
        if last_only:
            x = x[:, -1:]
        return head_logits(x, self.w.norm("lm_head.weight"))

    def window_ok(self, start: int) -> bool:
        """Can `prefill_window` run prompt rows from `start` on: a plain text prefill with the
        fused add-norm loop, and a tap whose rows it can hand on (or none)."""
        if not (FUSED_ADDNORM and FUSED["norm"]):
            return False
        if self.trace is not None or self.tree is not None or self._gv is not None:
            return False
        if self.mm is not None and start < self.mm.n:
            return False
        return self.tap is None or self.tap_indices() is not None

    def tap_indices(self):
        """The tap invocations (0 = the embedding, l + 1 = layer l's output) the tap's owner keeps,
        or None when it cannot say."""
        owner = getattr(self.tap, "__self__", None)
        fn = getattr(owner, "tap_indices", None)
        return None if fn is None else fn()

    def prefill_window(self, toks: list, starts: list, bufs: list | None = None):
        """Consecutive prompt chunks, LAYER-major: every chunk passes layer l before any passes
        layer l + 1, so an EXL3 projection decodes its fp16 slab once a window instead of once a
        chunk (`tools.exl3_linear` slab cache). Each chunk runs exactly the operations its own
        `forward` would (same row count, same entry state, same kernels), in another order: bit
        for bit the chunk-major forwards.

        `bufs[c]` (for c < len(toks) - 1): an (S, conv) pair that receives the recurrent state
        at the end of chunk c, which a chunk-major loop would have had in `state` at that line.
        Returns (the last row's logits of the last chunk, each chunk's post-norm hidden, each
        chunk's kept tap rows as {invocation: rows})."""
        from tools.norm_kernels import add_rms_norm
        try:
            from tools import exl3_linear as _x3
        except Exception:                                          # noqa: BLE001
            _x3 = None
        cfg = self.cfg
        n = cfg.num_hidden_layers
        if self.trace is None and self._gv is None:
            self._settle()
        keep = self.tap_indices() if self.tap is not None else set()
        keep = set(keep or ())
        C = len(toks)
        if C:
            self.kv.ensure(int(starts[-1]) + int(toks[-1].numel()))   # SRV-307: its rows
        primed0 = self.state.primed
        emb = self.w.norm("embed_tokens.weight")
        w0 = self.w.norm("layers.0.input_layernorm.weight")
        hs, xs, pos, taps = [], [], [], []
        for c in range(C):
            T = toks[c].numel()
            h = F.embedding(toks[c].view(1, T), emb)
            taps.append({0: h[0].detach()} if 0 in keep else {})
            s0 = starts[c] + self.pos_delta
            pos.append(torch.arange(s0, s0 + T, device=self.device))
            hs.append(h)
            xs.append(rms_norm(h, w0, cfg.rms_norm_eps))
        if _x3 is not None:
            _x3.slab_cache_on()
        try:
            for layer in range(n):
                p = f"layers.{layer}"
                lin = cfg.is_linear(layer)
                si = self.state.slot[layer] if lin else None
                wpost = self.w.norm(f"{p}.post_attention_layernorm.weight")
                wnext = self.w.norm(f"layers.{layer + 1}.input_layernorm.weight" if layer + 1 < n
                                    else "norm.weight")
                for c in range(C):
                    if lin:
                        a = self.linear_attention(xs[c], p, layer, primed0 or c > 0)
                        if bufs is not None and c < C - 1:
                            bufs[c][0][si].copy_(self.state.S[si])
                            bufs[c][1][si].copy_(self.state.conv[si])
                    else:
                        a = self.attention(xs[c], p, layer, starts[c], pos[c])
                    h, x = add_rms_norm(hs[c], a, wpost, cfg.rms_norm_eps)
                    m = self.mlp(x, p)
                    hs[c], xs[c] = add_rms_norm(h, m, wnext, cfg.rms_norm_eps)
                    if layer + 1 in keep:
                        taps[c][layer + 1] = hs[c][0].detach()
                if _x3 is not None:
                    _x3.slab_cache_clear()
                if WINDOW_YIELD is not None and layer + 1 < n:
                    # the slab cache is empty here; off while another slot's forwards run, so they
                    # take their own kernels (a decode's few rows never read a prefill's slab)
                    if _x3 is not None:
                        _x3.slab_cache_off()
                    WINDOW_YIELD()
                    if _x3 is not None:
                        _x3.slab_cache_on()
        finally:
            if _x3 is not None:
                _x3.slab_cache_off()
        self.state.primed = True
        self.kv.length = starts[-1] + toks[-1].numel()
        self.hidden_pre_norm = hs[-1]
        self.hidden_post_norm = xs[-1]
        return head_logits(xs[-1][:, -1:], self.w.norm("lm_head.weight")), xs, taps

    def replay_taps(self, kept: dict) -> None:
        """Hand the tap one chunk's rows in a forward's order: one invocation per layer and the
        embedding, the kept rows where the owner keeps them."""
        if self.tap is None:
            return
        for i in range(self.cfg.num_hidden_layers + 1):
            self.tap(kept.get(i))

    def forward_block(self, tokens: torch.Tensor, start: int) -> torch.Tensor:
        """Verify a block of tokens in one pass, keeping what a rollback would need.

        Returns logits for every position: `logits[i]` is the distribution for the token that
        follows `tokens[i]`.
        """
        T = tokens.numel()
        self.kv.ensure(start + T)                # SRV-307
        if self._graphs_on():
            self._scratch()
        fold = self._folds(T, tree=False)
        scratch = (not fold and self._scratch_S is not None and T <= VERIFY_ROWS and FUSED["gdnblock"]
                   and FUSED_GDNVERIFY and FUSED_COMMIT and RANKK == "1")
        if fold:
            self._fold_begin()
        else:
            self._settle()
        if (scratch or fold) and self._graphs_for(T, start) is not None:
            lg = self._graphs.run("chain", tokens, start)
            if fold:
                self._fold_end(list(range(T)))
            else:
                self._pending_walk = True
            return lg
        self._walk_scratch = scratch
        self.trace = BlockTrace()
        if fold:
            # the live buffer is the entry; the verify applies the pending commit to it
            self.trace.fold_par = self._fold_par
            self.trace.S_entry = self.state.S
        elif scratch:
            self.trace.S_entry = self.state.S               # the walk goes to _scratch_S
        elif FUSED_COMMIT and self._scratch_S is None:
            self.trace.S_entry = self.state.swap()
        else:
            self.trace.S_entry = self.state.S.clone()
        self.trace.conv_entry = self.state.conv.clone()
        try:
            logits = self.forward(tokens, start=start)
        finally:
            trace, self.trace = self.trace, None
            self._walk_scratch = False
        trace.start = start
        self._trace = trace
        self._pending_walk = scratch
        if fold:
            self._fold_end(list(range(T)))
        return logits[0]

    @staticmethod
    def _graphs_on() -> bool:
        return (VERIFY_GRAPH and FUSED_COMMIT and FUSED_GDNVERIFY and DECODE_ATTN and not KV_FP8
                and torch.cuda.is_available())

    def _scratch(self) -> torch.Tensor:
        if self._scratch_S is None:
            self._scratch_S = torch.empty_like(self.state.S)
        return self._scratch_S

    def _graphs_for(self, T: int, start: int):
        """The verify graphs, when this block can be served from one. An engine built with
        `graphs_off` (a side slot under `--side-graphs off`) verifies eager, as every block past
        the graphs' context range already does, and holds no graph pool."""
        if getattr(self, "graphs_off", False) or not self._graphs_on():
            return None
        self._scratch()
        if self._graphs is None:
            from engine.verify_graph import VerifyGraphs
            self._graphs = VerifyGraphs(self)
        return self._graphs if self._graphs.eligible(T, start) else None

    def _settle(self) -> None:
        """A chain verify accepted in full and never committed: its walked state is the state.
        And a pending commit is applied."""
        if self._pending_walk:
            self.state.S.copy_(self._scratch_S)
            self._pending_walk = False
        if self._pend is not None:
            self._flush()

    def _folds(self, T: int, tree: bool) -> bool:
        """Whether this verify takes the fused GDN mixer in every layer, so it can apply a pending
        commit itself: `linear_attention`'s routing, and at most the VERIFY_ROWS rows the
        static factor buffers hold."""
        if not (COMMIT_IN_VERIFY and FUSED_COMMIT and FUSED_GDNVERIFY and RANKK == "1"
                and self.state.primed and torch.cuda.is_available() and T <= VERIFY_ROWS):
            return False
        return FUSED["gdntree"] if tree else FUSED["gdnblock"]

    def _fold_buffers(self) -> list:
        if self._fac is None:
            cfg, dev, f32 = self.cfg, self.device, torch.float32
            L, H = len(cfg.linear_layers), cfg.linear_num_value_heads
            dk, dv = cfg.linear_key_head_dim, cfg.linear_value_head_dim
            n = VERIFY_ROWS
            self._fac = [(torch.zeros(L, H, n, dk, dtype=f32, device=dev),
                          torch.zeros(L, H, n, dv, dtype=f32, device=dev),
                          torch.zeros(L, H, n, dtype=f32, device=dev)) for _ in range(2)]
            self._prows = torch.zeros(n, dtype=torch.int32, device=dev)
            self._pn = torch.zeros(1, dtype=torch.int32, device=dev)
            self._pstage = torch.zeros(n + 1, dtype=torch.int32).pin_memory()
        return self._fac

    def _fold_begin(self) -> None:
        """Before a folding verify: settle a walked chain, apply a pending commit the verify cannot
        read (one recorded from a trace), hand the device the one it can -- its rows and count, 0
        when there is none -- and fix the parity this verify writes."""
        self._fold_buffers()
        if self._pending_walk:
            self.state.S.copy_(self._scratch_S)
            self._pending_walk = False
        if self._pend is not None and self._pend[0] != "static":
            self._flush()
        rows = self._pend[2] if self._pend is not None else []
        self._pend = None
        st = self._pstage
        st[0] = len(rows)
        if rows:
            st[1:1 + len(rows)] = torch.tensor(rows, dtype=torch.int32)
        self._pn.copy_(st[:1], non_blocking=True)
        self._prows.copy_(st[1:], non_blocking=True)
        self._fold_par = self._par

    def _fold_end(self, rows: list[int] | None) -> None:
        """After a folding verify: its factors are the next pending commit's -- a chain's full
        accept until `rollback_to` says otherwise, a tree's nothing until `commit_tree` -- and the
        next verify writes the other buffers."""
        p = self._fold_par
        self._fold_par = None
        self._par = 1 - p
        self._pend = ("static", p, rows) if rows else None

    def committed_state(self) -> torch.Tensor:
        """The recurrent state as the engine means it: `state.S` with a pending commit applied, in
        a copy -- for tools that compare states without disturbing the pending record."""
        S = self.state.S.clone()
        if self._pending_walk:
            S.copy_(self._scratch_S)
        if self._pend is not None:
            self._flush(S, keep=True)
        return S

    def _flush(self, S: torch.Tensor | None = None, keep: bool = False) -> None:
        """Apply the pending commit to `state.S` (or to `S`) in place with the commit kernel."""
        from tools.gdn_commit_kernels import fused_commit
        kind, what, rows = self._pend
        if not keep:
            self._pend = None
        S = self.state.S if S is None else S
        if kind == "static":
            kk, u, gc = self._fac[what]
        else:
            per = [what[layer] for layer in self.cfg.linear_layers]
            kk = torch.stack([f[0][0] for f in per])
            u = torch.stack([f[1][0] for f in per])
            gc = torch.stack([f[2][0] for f in per])
        fused_commit(S, S, kk, u, gc, rows)

    def forward_tree(self, tokens: torch.Tensor, parents, start: int) -> torch.Tensor:
        """Verify a whole draft TREE in one forward pass.

        `tokens` and `parents` are `engine.tree.DraftTree` in DFS pre-order: node 0 is the anchor,
        every other node is a drafted continuation of its parent, and every parent index is lower
        than its child's. `logits[i]` is what the target would write after the path root..i.

        The step costs what a chain of the same length costs -- the weights are read once either way
        -- so on this board the tree is nearly free width: the measured NVFP4 curve is 126 ms at one
        row, 145 ms from two to four, 153 ms at eight and 175 ms at sixteen, while a chain of the
        same sixteen nodes can only ever accept along one line.

        Nothing is committed here. `commit_tree` takes the path the caller accepted.
        """
        n = len(parents)
        if tokens.numel() != n:
            raise ValueError(f"{tokens.numel()} tokens for {n} parents")
        if n < 2:
            raise ValueError("a tree verify needs an anchor and at least one draft")
        self.kv.ensure(start + n)                # SRV-307: graph or eager, the rows it writes
        ctx = TreeCtx.get(parents, self.device, self.cfg.linear_conv_kernel_dim)
        self.picks = None
        delegate = ctx.is_chain and TREE_CHAIN_DELEGATE
        fold = not delegate and self._folds(n, tree=True)
        if fold:
            self._fold_begin()
        elif not delegate:
            self._settle()
        if delegate:
            # A chain-shaped tree IS a chain, and the chain has a kernel this path cannot use:
            # `fused_block_step` walks the recurrence in registers where the tree carries a factor
            # per depth, and the difference was measured at 12.5 ms a block.
            # So a drafter that proposes a line gets the line's price, and the router can fall back
            # to a chain by proposing one rather than by knowing anything about kernels.
            self._tree = ctx
            self._tree_start = start
            self._tree_is_chain = True
            return self.forward_block(tokens, start)
        self._tree_is_chain = False
        if FUSED["gdntree"] and RANKK == "1" and self._graphs_for(n, start) is not None:
            lg = self._graphs.run("tree", tokens, start, ctx)
            self.picks = self._graphs.last_picks
            if fold:
                self._fold_end(None)
            self._tree = ctx
            self._tree_start = start
            return lg
        self.trace = BlockTrace()
        # A TREE verify never advances the recurrent state, so there is nothing for the clone to
        # protect against. Both tree paths say so in their own words -- `fused_tree_step`: "the
        # entry state is NOT advanced: a tree has as many final states as it has leaves", and the
        # chunked one leaves it deliberately because its final state sums over nodes that never
        # coexist -- and `commit_tree` is what reconstructs the accepted path's state afterwards.
        # The clone is 302 MB a block, about 1.7 ms of 134.
        #
        # What it costs to skip: `S_entry` then ALIASES `state.S`. `commit_tree` reads `S_entry[i]`
        # and writes `state.S[i]` in the same iteration, which is safe because the right-hand side
        # is evaluated into a new tensor before the copy and no iteration reads another's index;
        # `rollback` becomes a self-copy, which is a no-op and is the right answer for a state that
        # was never changed. The chain path keeps its clone, where `fused_block_step` really does
        # walk the state forward in place.
        self.trace.S_entry = (self.state.S if TREE_ALIAS_STATE or FUSED_COMMIT
                              else self.state.S.clone())
        self.trace.conv_entry = self.state.conv.clone()
        if fold:
            # the tree's recurrence applies the pending commit to the live buffer, which
            # is this tree's entry (a tree never advances the state)
            self.trace.fold_par = self._fold_par
        self.tree = ctx
        try:
            logits = self.forward(tokens, start=start)
        finally:
            self.tree = None
            trace, self.trace = self.trace, None
        if fold:
            self._fold_end(None)
        self._trace = trace
        self._tree = ctx
        self._tree_start = start
        return logits[0]

    # ------------------------------------------------- parallel requests (lockstep verify rounds)
    # `forward_tree_multi` below verifies the tree blocks of several engines -- sequence slots over
    # ONE copy of the weights -- in one pass: every projection's weights are read once for all of
    # them. The methods here are `forward_tree`'s eager path cut at its projections, so the batch
    # can run one call per weight over every engine's rows. Nothing here runs unless the server was
    # started with `--parallel N` and two or more requests decode at once; `forward_tree` itself is
    # untouched.

    def tree_batchable(self, parents) -> bool:
        """Whether this tree block can share a pass with other engines' blocks: a real tree on the
        fused verify path, at most VERIFY_ROWS rows. A chain-shaped block keeps its own chain
        kernel (`forward_tree` delegates it to `forward_block`), so it verifies alone."""
        n = len(parents)
        if n < 2 or n > VERIFY_ROWS:
            return False
        ctx = TreeCtx.get(parents, self.device, self.cfg.linear_conv_kernel_dim)
        return not (ctx.is_chain and TREE_CHAIN_DELEGATE)

    def _tree_begin(self, tokens: torch.Tensor, parents, start: int):
        """`forward_tree` up to its `forward` call, on the eager path (a joint pass has no graph):
        the same statements in the same order."""
        n = len(parents)
        if tokens.numel() != n:
            raise ValueError(f"{tokens.numel()} tokens for {n} parents")
        if n < 2:
            raise ValueError("a tree verify needs an anchor and at least one draft")
        self.kv.ensure(start + n)                # SRV-307: a joint pass's rows
        ctx = TreeCtx.get(parents, self.device, self.cfg.linear_conv_kernel_dim)
        self.picks = None
        fold = self._folds(n, tree=True)
        if fold:
            self._fold_begin()
        else:
            self._settle()
        self._tree_is_chain = False
        self.trace = BlockTrace()
        self.trace.S_entry = (self.state.S if TREE_ALIAS_STATE or FUSED_COMMIT
                              else self.state.S.clone())
        self.trace.conv_entry = self.state.conv.clone()
        if fold:
            self.trace.fold_par = self._fold_par
        self.tree = ctx
        return ctx, fold, start

    def _tree_abort(self) -> None:
        self.tree = None
        self.trace = None

    def _tree_end(self, st, logits: torch.Tensor) -> torch.Tensor:
        """`forward_tree` after its `forward` call."""
        ctx, fold, start = st
        self.tree = None
        trace, self.trace = self.trace, None
        if fold:
            self._fold_end(None)
        self._trace = trace
        self._tree = ctx
        self._tree_start = start
        return logits[0]

    def _gdn_verify_ok(self, T: int, use_state: bool, B: int) -> bool:
        """`linear_attention`'s own test for the fused verify mixer, for a tree block."""
        return bool(FUSED_GDNVERIFY and RANKK == "1" and use_state and B == 1
                    and self.trace is not None and self.tree is not None
                    and FUSED["gdntree"] and T <= 64)

    def _gdn_verify_steps(self, h: torch.Tensor, p: str, layer: int, i: int):
        """`_linear_attention_verify` for a tree block, as a generator: the input and output
        projections are handed out as `(kind, weight name, rows)` requests; the gate inputs, the
        convolution, the recurrence and the gated norm are this engine's own. Same statements in
        the same order otherwise."""
        from tools.gdn_verify_kernels import verify_mixer
        cfg = self.cfg
        B, T, _ = h.shape
        flat = h.reshape(-1, h.shape[-1])
        grp = self.w.group(f"{p}.linear_attn.qkvz")
        if grp is not None:
            y = yield "grp", f"{p}.linear_attn.qkvz", flat
            mixed, z = y.split(grp.sizes, dim=-1)
        else:
            mixed = yield "lin", f"{p}.linear_attn.in_proj_qkv", flat
            z = yield "lin", f"{p}.linear_attn.in_proj_z", flat
        a, b = self._gate_inputs(flat, p)
        tree = self.tree
        pend, store, fac_out = None, True, None
        fp = self.trace.fold_par
        if fp is not None:
            fac, other = self._fac[fp], self._fac[1 - fp]
            fac_out = (fac[0][i], fac[1][i], fac[2][i])
            pend = (other[0][i], other[1][i], other[2][i], self._prows, self._pn)
        s_in, s_out = self.state.S[i], None
        o, fac, _ = verify_mixer(
            mixed, self.state.conv[i], self.w.norm(f"{p}.linear_attn.conv1d.weight").squeeze(1),
            a, b, self.w.norm(f"{p}.linear_attn.A_log"), self.w.norm(f"{p}.linear_attn.dt_bias"),
            s_in,
            key_dim=cfg.key_dim, key_heads=cfg.linear_num_key_heads,
            value_heads=cfg.linear_num_value_heads, head_k=cfg.linear_key_head_dim,
            head_v=cfg.linear_value_head_dim,
            window=tree.conv_idx, depths=tree.depths, max_depth=max(tree.depth_list),
            out_state=s_out, pend=pend, store_state=store, fac_out=fac_out, anc=tree.anc_incl)
        self.trace.factors[layer] = fac
        self.trace.layers[layer] = (mixed.t(), None, None, None, None, None)
        o = rms_norm_gated(o.reshape(-1, cfg.linear_value_head_dim),
                           z.reshape(-1, cfg.linear_value_head_dim),
                           self.w.norm(f"{p}.linear_attn.norm.weight"), cfg.rms_norm_eps)
        out = yield "lin", f"{p}.linear_attn.out_proj", o.view(B * T, -1)
        return out.view(B, T, -1)

    def _attention_steps(self, h: torch.Tensor, p: str, layer: int, start: int,
                         positions: torch.Tensor):
        """`attention` on its decode-attention verify path (T < GQA_FROM, no graph) as a
        generator: the q/k/v and o projections are handed out as requests; the norms, the rotary,
        the KV append and the attention over this engine's own cache are its own."""
        cfg = self.cfg
        B, T, _ = h.shape
        flat = h.reshape(-1, h.shape[-1])
        grp = self.w.group(f"{p}.self_attn.qkv")
        if grp is not None:
            y = yield "grp", f"{p}.self_attn.qkv", flat
            qy, ky, vy = y.split(grp.sizes, dim=-1)
            qg = qy.reshape(B, T, cfg.num_attention_heads, cfg.head_dim * 2)
            kk_ = ky.reshape(B, T, cfg.num_key_value_heads, cfg.head_dim)
            vv_ = vy.reshape(B, T, cfg.num_key_value_heads, cfg.head_dim)
        else:
            qy = yield "lin", f"{p}.self_attn.q_proj", flat
            ky = yield "lin", f"{p}.self_attn.k_proj", flat
            vy = yield "lin", f"{p}.self_attn.v_proj", flat
            qg = qy.view(B, T, cfg.num_attention_heads, cfg.head_dim * 2)
            kk_ = ky.view(B, T, cfg.num_key_value_heads, cfg.head_dim)
            vv_ = vy.view(B, T, cfg.num_key_value_heads, cfg.head_dim)
        q, gate = qg.chunk(2, dim=-1)
        gate = gate.reshape(B, T, -1)
        v = vv_.transpose(1, 2)
        if FUSED_ATTN_PREP and FUSED["norm"] and B == 1 and q.is_cuda and self._rope_rows is None:
            from tools.attn_prep import attn_prep
            if self._rope_cache is None:
                self.rope(positions[:1])
            q, k = attn_prep(q[0], kk_[0], self.w.norm(f"{p}.self_attn.q_norm.weight"),
                             self.w.norm(f"{p}.self_attn.k_norm.weight"), *self._rope_cache,
                             positions, cfg.rms_norm_eps)
        else:
            q = rms_norm(q, self.w.norm(f"{p}.self_attn.q_norm.weight"),
                         cfg.rms_norm_eps).transpose(1, 2)
            k = rms_norm(kk_, self.w.norm(f"{p}.self_attn.k_norm.weight"),
                         cfg.rms_norm_eps).transpose(1, 2)
            rr = self._rope_rows
            cos, sin = self.rope(positions) if rr is None else rr
            q, k = self.apply_rope(q, k, cos, sin)
        kk, vv = self.kv.append(layer, k, v, start)
        from tools.attn_kernels import decode_attention
        bm = self.tree.anc_incl
        ks, vs = self.kv.scales(layer, start + T)
        o = decode_attention(q, kk, vv, start, bm, ks=ks, vs=vs)
        o = o.transpose(1, 2).reshape(B, T, -1)
        o = o * torch.sigmoid(gate)
        out = yield "lin", f"{p}.self_attn.o_proj", o.reshape(B * T, -1)
        return out.view(B, T, -1)

    def _mixer_steps(self, x, p, layer, start, positions, use_state, T):
        """A layer's mixer for `_forward_steps`: the step version where the served verify path
        applies (its projections batchable), else the plain call (nothing handed out)."""
        cfg = self.cfg
        if cfg.is_linear(layer):
            i = self.state.slot[layer]
            if self._gdn_verify_ok(T, use_state, x.shape[0]):
                return (yield from self._gdn_verify_steps(x, p, layer, i))
            return self.linear_attention(x, p, layer, use_state)
        if DECODE_ATTN and T < GQA_FROM and self._gv is None and x.is_cuda:
            return (yield from self._attention_steps(x, p, layer, start, positions))
        return self.attention(x, p, layer, start, positions)

    def _forward_steps(self, tokens: torch.Tensor, start: int):
        """`forward` of a tree block (eager, `last_only` off) as a generator: it yields
        `(kind, weight name, rows)` where a projection, the MLP or the head would run and takes
        the result back, so `forward_tree_multi` can run one call for several engines. Every other
        statement is `forward`'s, `_forward_layers`' and `_forward_addnorm`'s, in their order;
        returns the logits."""
        cfg = self.cfg
        T = tokens.numel()
        self.kv.ensure(start + T)                # SRV-307
        h = F.embedding(tokens.view(1, T), self.w.norm("embed_tokens.weight"))
        if self.tap is not None:
            self.tap(h[0].detach())
        s0 = start + self.pos_delta
        positions = s0 + self.tree.depths
        use_state = self.state.primed
        n = cfg.num_hidden_layers
        if FUSED_ADDNORM and FUSED["norm"]:
            from tools.norm_kernels import add_rms_norm
            x = rms_norm(h, self.w.norm("layers.0.input_layernorm.weight"), cfg.rms_norm_eps)
            for layer in range(n):
                p = f"layers.{layer}"
                a = yield from self._mixer_steps(x, p, layer, start, positions, use_state, T)
                h, x = add_rms_norm(h, a, self.w.norm(f"{p}.post_attention_layernorm.weight"),
                                    cfg.rms_norm_eps)
                m = yield "mlp", p, x
                nxt = (f"layers.{layer + 1}.input_layernorm.weight" if layer + 1 < n
                       else "norm.weight")
                h, x = add_rms_norm(h, m, self.w.norm(nxt), cfg.rms_norm_eps)
                if self.tap is not None:
                    self.tap(h[0].detach())
            self.state.primed = True
            self.kv.length = start + T
            self.hidden_pre_norm = h
            self.hidden_post_norm = x
            return (yield "head", "lm_head.weight", x)
        for layer in range(n):
            p = f"layers.{layer}"
            res = h
            x = rms_norm(h, self.w.norm(f"{p}.input_layernorm.weight"), cfg.rms_norm_eps)
            x = yield from self._mixer_steps(x, p, layer, start, positions, use_state, T)
            h = res + x
            res = h
            x = rms_norm(h, self.w.norm(f"{p}.post_attention_layernorm.weight"), cfg.rms_norm_eps)
            m = yield "mlp", p, x
            h = res + m
            if self.tap is not None:
                self.tap(h[0].detach())
        self.state.primed = True
        self.kv.length = start + T
        self.hidden_pre_norm = h
        h = rms_norm(h, self.w.norm("norm.weight"), cfg.rms_norm_eps)
        self.hidden_post_norm = h
        return (yield "head", "lm_head.weight", h)

    def commit_tree(self, path: list[int]) -> None:
        """Make the engine's state what verifying `path` alone as a chain would have left.

        `path` is node indices from the anchor down, ascending, starting at 0. For the 48 linear
        layers this is the SpecLA identity: within one chunk the pseudo-values `u_t` depend only on
        the tokens above t on its own branch, so the state after any path is

            S_path = exp(gc[last]) . S_entry + SUM_{t in path} exp(gc[last] - gc[t]) . k_t (x) u_t

        which is one state read and a [Dk, L] x [L, Dv] product per layer, over factors the verify
        pass already computed. It is the rank-k rollback of `rollback_to` with the prefix replaced
        by a gather, and it costs the same whether the path is the whole tree or one node of it.

        For the 16 attention layers the cache holds every node at its DFS slot, so committing is a
        gather of the accepted rows down to `start .. start + L - 1`; everything past that is
        overwritten by the next block and never read, because `kv.length` says so.
        """
        trace, ctx = self._trace, self._tree
        if trace is None or ctx is None:
            raise RuntimeError("commit_tree without a preceding forward_tree")
        if self._tree_is_chain:
            # the path of a chain is a prefix, so the chain rollback is the whole of the commit
            self.rollback_to(len(path))
            self._tree = None
            return
        if not path or path[0] != 0 or any(b <= a for a, b in zip(path, path[1:])):
            raise ValueError(f"path must start at the anchor and ascend: {path}")
        start, width, L = self._tree_start, self.cfg.linear_conv_kernel_dim, len(path)
        if FUSED_COMMIT and len(trace.factors) == len(self.cfg.linear_layers):
            self._fused_commit(trace, path)
            if path != list(range(L)):
                sel = start + h2d(path, torch.long, self.device)
                self.kv.k[..., start:start + L, :] = self.kv.k[..., sel, :]
                self.kv.v[..., start:start + L, :] = self.kv.v[..., sel, :]
                if self.kv.fp8:
                    self.kv.ks[..., start:start + L] = self.kv.ks[..., sel]
                    self.kv.vs[..., start:start + L] = self.kv.vs[..., sel]
            self.kv.length = start + L
            self._trace = None
            self._tree = None
            return
        idx = h2d(path, torch.long, self.device)
        last = path[-1]
        for layer, (kk, u, gc) in trace.factors.items():
            i = self.state.slot[layer]
            gl = gc[:, :, last]                                        # [B, H]
            w = torch.exp(gl[..., None] - gc[:, :, idx])               # [B, H, L]
            kw = (kk[:, :, idx] * w[..., None]).transpose(-1, -2)      # [B, H, Dk, L]
            self.state.S[i].copy_(gl[..., None, None].exp() * trace.S_entry[i]
                                  + kw @ u[:, :, idx])
            raw = trace.layers[layer][0]                               # [C, n], pre-convolution
            self.state.conv[i].copy_(gdn.conv_tail(raw[None], trace.conv_entry[i], idx, width))
        # A path that is already a DFS prefix needs no gather: its rows are the rows they would be
        # copied to. That is every block of a chain-shaped tree, accepted in full or not.
        if path != list(range(L)):
            sel = start + idx
            self.kv.k[..., start:start + L, :] = self.kv.k[..., sel, :]
            self.kv.v[..., start:start + L, :] = self.kv.v[..., sel, :]
            if self.kv.fp8:
                self.kv.ks[..., start:start + L] = self.kv.ks[..., sel]
                self.kv.vs[..., start:start + L] = self.kv.vs[..., sel]
        self.kv.length = start + L
        self._trace = None
        self._tree = None

    def accept_tree(self, tree, picks: list[int]) -> tuple[list[int], list[int]]:
        """Walk the tree along what the target actually wrote. Returns (path, new tokens).

        Greedy verification of a tree is a walk, not a comparison: at each node take the target's
        own argmax and descend into the child carrying that token, stopping where there is none.
        The tokens gained are the path's own plus the target's token at the node it stopped at, so a
        tree yields at least one token exactly as a chain does.
        """
        node, path = 0, [0]
        while True:
            want = picks[node]
            nxt = next((c for c in range(node + 1, len(tree.tokens))
                        if tree.parents[c] == node and tree.tokens[c] == want), None)
            if nxt is None:
                break
            path.append(nxt)
            node = nxt
        new = [tree.tokens[i] for i in path[1:]] + [picks[node]]
        return path, new

    def rollback_to(self, keep: int) -> None:
        """Make the state what it would have been after only the first `keep` tokens of the block.

        No weight is read: the entry state and the projections cached during the verify pass are
        everything the recurrence needs. The convolution state is rebuilt from the same prefix of
        the pre-convolution projections.
        """
        trace = self._trace
        if trace is None:
            raise RuntimeError("rollback_to without a preceding forward_block")
        # The KV rows past the kept prefix hold the rejected drafts. The next block overwrites
        # them, so decoding never cared; a snapshot does, because `capture` takes `kv.length` rows
        # and the recurrent state below is the kept prefix's. Left at the block's end, a
        # generation that stopped on a block rejected at its last slot stored an entry whose key
        # was one token longer than what its state had seen.
        self.kv.length = trace.start + keep
        width = self.cfg.linear_conv_kernel_dim
        self._pending_walk = False          # the commit rebuilds the state from the entry
        if (FUSED_COMMIT and RANKK == "1"
                and len(trace.factors) == len(trace.layers) == len(self.cfg.linear_layers)):
            # every layer is rebuilt from the entry state, so nothing is copied back first
            self._fused_commit(trace, list(range(keep)))
            return
        self.state.S.copy_(trace.S_entry)
        if trace.factors and len(trace.factors) == len(trace.layers) and RANKK != "0":
            # Rank-k: one state read and one [Dk, keep] x [keep, Dv] product per layer. The replay
            # below re-runs the chunked recurrence, whose intra-chunk inverse is a serial loop, and
            # the ledger charges it 22-23 ms on every block that takes a rollback.
            for layer, (kk, u, gc) in trace.factors.items():
                i = self.state.slot[layer]
                gk = gc[:, :, keep - 1]                                   # [B, H]
                w = torch.exp(gk[..., None] - gc[:, :, :keep])            # [B, H, keep]
                kw = (kk[:, :, :keep] * w[..., None]).transpose(-1, -2)   # [B, H, Dk, keep]
                S_rank = gk[..., None, None].exp() * trace.S_entry[i] + kw @ u[:, :, :keep]
                if RANKK == "check":
                    q, k2, v2, g2, b2 = trace.layers[layer][1:]
                    _, S_replay = gdn.chunk_gated_delta_rule(
                        q[:, :keep], k2[:, :keep], v2[:, :keep], g2[:, :keep], b2[:, :keep],
                        trace.S_entry[i], chunk_size=max(2, min(64, keep)))
                    d = (S_rank - S_replay).abs().max().item()
                    ROLLBACK_DIFF.append((layer, keep, d,
                                          S_replay.abs().max().item()))
                self.state.S[i].copy_(S_rank)
                raw = trace.layers[layer][0]
                joined = torch.cat([trace.conv_entry[i], raw[None, :, :keep]], dim=-1)
                self.state.conv[i].copy_(joined[:, :, -(width - 1):])
            return
        # The chunk size is a blocking choice, not a semantic one, and the chunked form pays a
        # serial loop of `chunk_size - 1` small operations for its intra-chunk inverse. A verified
        # block is a dozen tokens, so blocking it at 64 runs that loop 63 times per layer, 3,000
        # times per rollback, for nothing. Block it at the prefix length instead.
        chunk = max(2, min(64, keep))
        for layer, (raw, q, k, v, g, beta) in trace.layers.items():
            i = self.state.slot[layer]
            _, S = gdn.chunk_gated_delta_rule(q[:, :keep], k[:, :keep], v[:, :keep],
                                              g[:, :keep], beta[:, :keep],
                                              trace.S_entry[i], chunk_size=chunk)
            self.state.S[i].copy_(S)
            joined = torch.cat([trace.conv_entry[i], raw[None, :, :keep]], dim=-1)
            self.state.conv[i].copy_(joined[:, :, -(width - 1):])

    def _fused_commit(self, trace: "BlockTrace", rows: list[int]) -> None:
        """The recurrent and convolution state after `rows` of the verified block, all layers at
        once, from the entry state the trace kept (tools/gdn_commit_kernels.py)."""
        from tools.gdn_commit_kernels import conv_commit, fused_commit
        if COMMIT_IN_VERIFY and trace.S_entry is self.state.S:
            # the recurrent state's commit waits for the next verify (or `_settle`); the
            # entry is the live buffer, so nothing reads the state in between. The convolution
            # tails are small and are committed now.
            self._pend = (("static", trace.fold_par, list(rows)) if trace.fold_par is not None
                          else ("trace", trace.factors, list(rows)))
            raw = torch.stack([trace.layers[layer][0] for layer in self.cfg.linear_layers])
            conv_commit(trace.conv_entry, self.state.conv, raw, rows)
            self._pend_ctx = (self._pend, trace.conv_entry, raw)
            return
        facs = [trace.factors[layer] for layer in self.cfg.linear_layers]
        kk = torch.stack([f[0][0] for f in facs])
        u = torch.stack([f[1][0] for f in facs])
        gc = torch.stack([f[2][0] for f in facs])
        fused_commit(trace.S_entry, self.state.S, kk, u, gc, rows)
        raw = torch.stack([trace.layers[layer][0] for layer in self.cfg.linear_layers])
        conv_commit(trace.conv_entry, self.state.conv, raw, rows)

    def trim_pending(self, n: int) -> bool:
        """Bring the engine back to length `n`, inside the last verified block, when that block's
        commit is still pending (QWEN38_COMMIT_IN_VERIFY): the recurrent state has not absorbed
        the block yet, so the commit is cut to the path rows below `n` and the convolution tails
        are rebuilt from the block's entry tails, as a commit of the shorter path does. A
        generation that stops part way through an accepted path -- at an end token, a stop, the
        token budget, or a client that stopped reading -- has `kv.length` ahead of the tokens it
        committed, and a snapshot of it has no key. False, and nothing changed, when the state
        has absorbed rows past `n` or `n` is below the block."""
        n = int(n)
        if n == self.kv.length:
            return True
        ctx = self._pend_ctx
        if (n > self.kv.length or self._pending_walk or ctx is None or self._pend is None
                or ctx[0] is not self._pend):
            return False
        from tools.gdn_commit_kernels import conv_commit
        _, conv_entry, raw = ctx
        kind, what, rows = self._pend
        start = self.kv.length - len(rows)
        m = n - start
        if m < 0:
            return False
        if m == 0:
            self._pend = None
            self.state.conv.copy_(conv_entry)
        else:
            self._pend = (kind, what, list(rows[:m]))
            conv_commit(conv_entry, self.state.conv, raw, rows[:m])
        self._pend_ctx = (self._pend, conv_entry, raw) if self._pend is not None else None
        self.kv.length = n
        return True

    def reset(self) -> None:
        self.state.S.zero_()
        self.state.conv.zero_()
        self.state.primed = False
        self.kv.length = 0
        self._pend = None
        self._pend_ctx = None


def _run_request(eng, kind: str, name: str, x: torch.Tensor) -> torch.Tensor:
    """One batched request of `forward_tree_multi` over the rows `x` of every engine."""
    if kind == "mlp":
        return eng.mlp(x, name)
    if kind == "head":
        return head_logits(x, eng.w.norm(name))
    if kind == "grp":
        return matmul_group(x, eng.w.group(name))
    return linear(x, eng.w.proj(name))


def forward_tree_multi(items: list) -> list:
    """Verify the tree blocks of several engines -- sequence slots over ONE copy of the weights --
    with every projection's weights read once for all of them (parallel requests, lockstep rounds).

    `items` = [(engine, tokens, parents, start)], every block `tree_batchable`. Each engine's
    forward is a generator (`_forward_steps`) that hands out its projections -- the MLP, the head,
    and on the fused verify path the recurrent and attention layers' input and output projections
    -- as `(kind, weight name, rows)`. Requests for the same weight run as ONE call over all the
    engines' rows and are split back. The convolutions, recurrences, attention over each engine's
    own KV, the norms and the commits stay each engine's own, at its own row count. Returns each
    engine's logits, as `forward_tree` would. Every matrix-product kernel fixes its reduction order
    by the weight's shape, never by the row count, up to 32 rows (docs/kernels.md), so a row's
    projection is the bits it is alone; the caller keeps the joint pass at or under 32 rows."""
    if not items:
        return []
    sts, gens = [], []
    try:
        for eng, tokens, parents, start in items:
            sts.append(eng._tree_begin(tokens, parents, start))
            gens.append(eng._forward_steps(tokens, start))
        reqs = [next(g) for g in gens]
        eng0 = items[0][0]
        done: list = [None] * len(gens)
        live = list(range(len(gens)))
        while live:
            groups: dict = {}
            for j in live:
                kind, name, _ = reqs[j]
                groups.setdefault((kind, name), []).append(j)
            answers: dict = {}
            for (kind, name), js in groups.items():
                xs = [reqs[j][2] for j in js]
                if len(xs) == 1:
                    answers[js[0]] = _run_request(eng0, kind, name, xs[0])
                    continue
                # the MLP and the head take [1, T, H] (their row axis is 1), the projections [T, H]
                dim = 1 if kind in ("mlp", "head") else 0
                out = _run_request(eng0, kind, name, torch.cat(xs, dim=dim))
                for j, part in zip(js, out.split([x.shape[dim] for x in xs], dim=dim)):
                    answers[j] = part
            nxt = []
            for j in live:
                try:
                    reqs[j] = gens[j].send(answers[j])
                    nxt.append(j)
                except StopIteration as fin:
                    done[j] = fin.value
            live = nxt
        return [eng._tree_end(st, lg) for (eng, *_), st, lg in zip(items, sts, done)]
    except BaseException:
        for eng, *_ in items:
            eng._tree_abort()
        raise


# the family's engine under a family-neutral name; `Qwen38Engine` stays for every caller.
Engine = Qwen38Engine
