"""The verify pass served from CUDA graphs.

`tools/graph_bound.py` measured the prize: the same sixteen-row verify replayed from a captured
graph took 89.2 ms against 97.3 eager, because a graph removes the per-launch gaps on the device --
about 2 us each, several thousand a block -- which a Python host and a native one pay alike. A
verify cannot simply be replayed as captured, though: its position, its context length and, for a
tree, its shape change every block, and a graph bakes every launch argument in. So the verify is
made graph-safe first:

  * the block's positions and `[start, start + T]` live in static device buffers, filled before a
    replay; the rotary tables are gathered at those positions;
  * the KV append is a scatter at the positions (`index_copy_`), and attention reads the whole
    cache buffer with the length from the device (`tools/attn_kernels.py::decode_attention_dev`);
  * a tree's ancestor mask, convolution windows and depths are static buffers per row count;
  * a chain's walked recurrent state goes into a static scratch buffer, the entry state is the live
    one, and the commit rebuilds the live state in place (`tools/gdn_commit_kernels.py`). If the
    caller accepts the whole chain and never commits, the next engine call copies the walked state
    in (`Qwen38Engine._settle`), which is what the eager path would have left.

One graph per (chain or tree, rows, context class), captured lazily into one shared memory pool --
the graphs never run concurrently -- after an eager warm-up of the same graph-safe body. The context
class is the next power of two of the context, from 1,024 to 32,768: past that the attention chunk
grows and the eager path, which cuts the context elsewhere, keeps the verify. Everything the eager
path leaves on the engine is restored after a replay from what the capture recorded: the trace, the
hidden states, and the drafter's tap rows, which are replayed into the tap in order.

Requires the fused commit and the fused GDN verify mixer (the graph-safe recurrence is theirs), the
decode-attention kernel, and a bf16 KV cache; otherwise the eager path runs, unchanged.
"""

from __future__ import annotations

import os
from engine.settings import SETTINGS as _S  # noqa: E402  (every QWEN38_* knob)

import torch

from engine import capture_lock  # noqa: E402

# The longest context a verify is served from a graph at. 32,768 is the code as it was; the 32k
# probe's prompt (32,795 tokens) was one past it and every verify of it ran eager.
MAX_CTX = int(_S.get("GRAPH_MAX_CTX"))
MIN_CLASS = 1024


class StaticTree:
    """The attributes of `engine.model.TreeCtx` the forward reads, in buffers a graph can keep."""

    def __init__(self, n: int, width: int, device):
        self.n = n
        self.anc_incl = torch.zeros(n, n, dtype=torch.bool, device=device)
        self.anc_strict = torch.zeros(n, n, dtype=torch.bool, device=device)
        self.conv_idx = torch.zeros(n, width, dtype=torch.long, device=device)
        self.depths = torch.zeros(n, dtype=torch.long, device=device)
        self.depth_list = [0] * n
        self.is_chain = False

    def load(self, ctx) -> None:
        self.anc_incl.copy_(ctx.anc_incl)
        self.anc_strict.copy_(ctx.anc_strict)
        self.conv_idx.copy_(ctx.conv_idx)
        self.depths.copy_(ctx.depths)
        self.depth_list = list(ctx.depth_list)


class Captured:
    """One graph and everything its capture left behind that the engine needs after a replay."""

    __slots__ = ("graph", "logits", "picks", "trace", "hidden_pre", "hidden_post", "taps")

    def __init__(self):
        self.graph = torch.cuda.CUDAGraph()
        self.logits = None
        self.picks = None
        self.trace = None
        self.hidden_pre = None
        self.hidden_post = None
        self.taps: list[torch.Tensor] = []


class VerifyGraphs:
    def __init__(self, eng):
        self.eng = eng
        dev = eng.device
        self.pool = torch.cuda.graph_pool_handle()
        self.stream = torch.cuda.Stream()
        self.graphs: dict = {}
        self.tokens = {}                          # T -> static token buffer
        self.pos = {}                             # T -> static rotary positions
        self.slots = {}                           # T -> static KV slots (DFS order for a tree)
        self.trees = {}                           # T -> StaticTree
        self.lenp = torch.zeros(2, dtype=torch.int32, device=dev)
        self.lenp_host = torch.zeros(2, dtype=torch.int32).pin_memory()
        self.max_lc = 0                           # the class being captured or replayed
        self.last_picks = None                    # the replayed graph's argmax, or None
        self.stats = {"captured": 0, "replayed": 0, "eager": 0}

    @staticmethod
    def signature() -> tuple:
        """The flags a captured graph baked in: a graph captured under one set is never replayed
        under another (an in-process A/B flips them)."""
        import engine.model as M
        from tools import nvfp4_skinny
        from tools import gdn_verify_kernels as V
        from tools import head_gemv as H
        from tools import attn_kernels as AK
        return (AK.STAGES3, nvfp4_skinny.SKINNY, nvfp4_skinny.ALT, nvfp4_skinny.ALT2, nvfp4_skinny.PDL,
                M.FUSED_ADDNORM,
                M.TREE_HOST_DEPTH,
                M.FUSED["norm"], V.ONE_WARP, V.WARPS, M.GDN_AB, M.FUSED_ATTN_PREP, V.WY,
                V.WY_FUSED, V.WY_MAXT, V.WY_CHAIN_MAXT, V.WY_FUSED_MAXT, V.CONV_BLOCK,
                V.CONV_WARPS, M.HOST_ASYNC, V.WY_KC, nvfp4_skinny.LDW, nvfp4_skinny.WIDE_B,
                V.TREE_PF, H.HEAD_BN, H.HEAD_WARPS, H.HEAD_STAGES, nvfp4_skinny.WIDE_LDW)

    @staticmethod
    def ctx_class(lc: int) -> int:
        c = MIN_CLASS
        while c < lc:
            c *= 2
        return c

    def eligible(self, T: int, start: int) -> bool:
        import engine.model as M
        return 2 <= T <= M.VERIFY_ROWS and start + T <= MAX_CTX

    def _buffers(self, T: int):
        if T not in self.tokens:
            dev = self.eng.device
            self.tokens[T] = torch.zeros(T, dtype=torch.long, device=dev)
            self.pos[T] = torch.zeros(T, dtype=torch.long, device=dev)
            self.slots[T] = torch.zeros(T, dtype=torch.long, device=dev)
            self.trees[T] = StaticTree(T, self.eng.cfg.linear_conv_kernel_dim, dev)
        return self.tokens[T], self.pos[T], self.trees[T]

    def run(self, kind: str, tokens: torch.Tensor, start: int, ctx=None):
        """Verify `tokens` at `start` from a graph; returns the logits [T, V]."""
        eng = self.eng
        T = tokens.numel()
        tok, pos, tree = self._buffers(T)
        tok.copy_(tokens.view(-1))
        self.lenp_host[0] = start
        self.lenp_host[1] = start + T
        self.lenp.copy_(self.lenp_host, non_blocking=True)
        # a tree node's POSITION is its depth, its KV SLOT its index: the two differ for a tree
        torch.arange(start, start + T, device=eng.device, out=self.slots[T])
        # after a prompt with images a row's rotary position is its index plus
        # `pos_delta`; 0 for text, where this is what it always was
        d = eng.pos_delta
        if kind == "tree":
            tree.load(ctx)
            torch.add(tree.depths, start + d, out=pos)
        elif d:
            torch.arange(start + d, start + d + T, device=eng.device, out=pos)
        else:
            pos.copy_(self.slots[T])
        cls = self.ctx_class(start + T)
        # a folding verify writes one of two static factor sets: one graph per parity
        key = (kind, T, cls, self.signature(), eng._fold_par)
        cap = self.graphs.get(key)
        if cap is None:
            cap = self._capture(kind, T, cls, tree)
            self.graphs[key] = cap
        cap.graph.replay()
        self.stats["replayed"] += 1
        self._restore(cap, start, T)
        self.last_picks = cap.picks
        return cap.logits

    def precapture(self, widths=range(2, 17), cls: int = MIN_CLASS) -> int:
        """Capture the chain and tree graphs for these row counts at one context class, so the
        first requests do not pay for it. Returns how many were captured. Needs a primed state
        (a prefill has run): the graph-safe body is the verify of an existing sequence."""
        eng = self.eng
        from engine.model import TreeCtx
        primed, n = eng.state.primed, 0
        eng.state.primed = True
        # a pending commit is applied first -- the captures below write both static
        # factor sets -- and the graphs of both parities are captured
        eng._settle()
        pars = (0, 1) if eng._folds(2, tree=False) else (None,)
        if pars[0] is not None:
            eng._fold_buffers()
            eng._pn.zero_()
        try:
            for T in widths:
                tok, pos, tree = self._buffers(T)
                # any tree of T nodes that is not a chain: the anchor with two children, the second
                # a line down to depth 15 at most (the walk kernel carries 16 levels), the rest
                # more children of the anchor (row counts past 17 are captured too)
                parents = ((-1, 0, 0) + tuple(range(2, min(T - 1, 16))) if T >= 3 else (-1, 0))
                parents += (0,) * (T - len(parents))
                if T >= 3:
                    tree.load(TreeCtx.get(parents, eng.device, eng.cfg.linear_conv_kernel_dim))
                for kind in (("chain", "tree") if T >= 3 else ("chain",)):
                    for par in pars:
                        key = (kind, T, cls, self.signature(), par)
                        if key in self.graphs:
                            continue
                        self.lenp_host[0], self.lenp_host[1] = 0, T
                        self.lenp.copy_(self.lenp_host)
                        torch.arange(0, T, device=eng.device, out=self.slots[T])
                        pos.copy_(self.slots[T])
                        eng._fold_par = par
                        self.graphs[key] = self._capture(kind, T, cls, tree)
                        n += 1
        finally:
            eng.state.primed = primed
            eng._fold_par = None
        torch.cuda.synchronize()
        return n

    def _body(self, kind: str, T: int, tree):
        """The graph-safe verify, run eagerly to warm up and then under capture."""
        eng = self.eng
        from engine.model import BlockTrace
        eng.trace = BlockTrace()
        eng.trace.S_entry = eng.state.S
        eng.trace.conv_entry = eng.state.conv.clone()
        eng.trace.fold_par = eng._fold_par          # no walk to store, factors static
        eng._gv = self
        eng._walk_scratch = kind == "chain" and eng._fold_par is None
        if kind == "tree":
            eng.tree = tree
        try:
            logits = eng.forward(self.tokens[T], start=0)
        finally:
            eng._gv = None
            eng._walk_scratch = False
            eng.tree = None
            trace, eng.trace = eng.trace, None
        return logits[0], trace

    def _capture(self, kind: str, T: int, cls: int, tree) -> Captured:
        import engine.model as M
        eng = self.eng
        self.max_lc = cls
        cap = Captured()
        conv_keep = eng.state.conv.clone()
        tap = eng.tap
        # the warm-up runs the body for real, and a pending commit it applied here would be
        # applied again by the replay; the device's count is 0 until the capture is done
        pn_keep = eng._pn.clone() if eng._fold_par is not None else None
        if pn_keep is not None:
            eng._pn.zero_()
        # warm-up: the same body, eagerly, so every kernel in it is compiled before the capture.
        # It advances a chain's conv state in place and writes the KV rows the replay will write
        # again; the conv state is put back, the state and the KV need nothing.
        eng.tap = None
        self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            self._body(kind, T, tree)
        torch.cuda.current_stream().wait_stream(self.stream)
        eng.state.conv.copy_(conv_keep)
        taps: list[torch.Tensor] = []
        eng.tap = taps.append if tap is not None else None
        try:
            # no other thread's CUDA call may run inside a capture (engine/capture_lock.py)
            with capture_lock.LOCK, torch.cuda.graph(cap.graph, pool=self.pool,
                                                     stream=self.stream):
                logits, trace = self._body(kind, T, tree)
                # the greedy picks inside the graph, the same argmax kernel the loop ran
                picks = logits.argmax(-1) if M.HOST_ASYNC else None
        finally:
            eng.tap = tap
        eng.state.conv.copy_(conv_keep)
        if pn_keep is not None:
            eng._pn.copy_(pn_keep)
        cap.logits, cap.picks, cap.trace = logits, picks, trace
        cap.hidden_pre, cap.hidden_post = eng.hidden_pre_norm, eng.hidden_post_norm
        cap.taps = taps
        self.stats["captured"] += 1
        return cap

    def _restore(self, cap: Captured, start: int, T: int) -> None:
        eng = self.eng
        eng._trace = cap.trace
        cap.trace.start = start         # captured at 0; a partial accept puts kv.length back from it
        eng.hidden_pre_norm, eng.hidden_post_norm = cap.hidden_pre, cap.hidden_post
        eng.state.primed = True
        eng.kv.length = start + T
        if eng.tap is not None:
            for h in cap.taps:
                eng.tap(h)
