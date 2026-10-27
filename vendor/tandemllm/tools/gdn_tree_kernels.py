"""The gated delta rule over a draft TREE, with the state tile never leaving registers.

`tools/gdn_kernels.py::fused_block_step` (track A's) walks a chain of up to sixteen tokens forward
in registers: the `[DK, BV]` tile of the state is loaded once, updated T times and stored once, so
verifying sixteen tokens costs the state traffic of verifying one. A tree cannot use it, because at
node t the state to update is the one after t's PARENT, not after t-1, and the chain's walk has
already overwritten it. That single fact is what made a tree verify cost 12.5 ms a block more than
a chain of the same width -- the whole of the tree's deficit.

The way out is the factor identity this engine already relies on for its rollback:

    S_t = exp(gc_t) . S_0  +  SUM_{j in path(t)} exp(gc_t - gc_j) . k_j (x) u_j

Under DFS pre-order, when the walk reaches a node at depth d, the last node it visited at each
depth 0..d-1 IS that node's ancestor. So carrying one `(k, u, gc)` per DEPTH -- a stack sixteen deep
-- is carrying exactly the node's own path, and the parent's state is a rank-d reconstruction from
the entry tile. No pops, no unwinding, no division by a decay, and the entry tile is still read once.

The cost is register arithmetic, not memory: a chain node does one rank-1 update, a tree node does a
rank-d rebuild with d at most the tree's depth. The kernel is memory-bound either way, which is the
bet this file is making.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except ImportError:                                            # pragma: no cover
    HAVE_TRITON = False


if HAVE_TRITON:

    @triton.jit
    def _gdn_tree_step(Q, K, V, G, BETA, DEPTH, S, OUT, DELTA, GC, T,
                       s_qt, s_vt, s_gt, s_h, s_k, s_v,
                       DK: tl.constexpr, DV: tl.constexpr, BV: tl.constexpr,
                       MAXD: tl.constexpr, EPS: tl.constexpr, SCALE: tl.constexpr):
        h = tl.program_id(0)
        vb = tl.program_id(1)
        ok = tl.arange(0, DK)
        ov = vb * BV + tl.arange(0, BV)
        sp = S + h * s_h + ok[:, None] * s_k + ov[None, :] * s_v
        s0 = tl.load(sp)                                    # the entry tile, read once
        d_idx = tl.arange(0, MAXD)
        # one (k, u, gc) per DEPTH: under DFS pre-order this is the current node's own ancestry
        st_k = tl.zeros([MAXD, DK], dtype=tl.float32)
        st_u = tl.zeros([MAXD, BV], dtype=tl.float32)
        st_g = tl.zeros([MAXD], dtype=tl.float32)
        for t in range(T):
            d = tl.load(DEPTH + t)
            q = tl.load(Q + t * s_qt + h * DK + ok).to(tl.float32)
            k = tl.load(K + t * s_qt + h * DK + ok).to(tl.float32)
            q = q * tl.rsqrt(tl.sum(q * q) + EPS) * SCALE
            k = k * tl.rsqrt(tl.sum(k * k) + EPS)
            v = tl.load(V + t * s_vt + h * DV + ov).to(tl.float32)
            g = tl.load(G + t * s_gt + h).to(tl.float32)
            beta = tl.load(BETA + t * s_gt + h).to(tl.float32)
            # the parent's cumulative gate; zero at the anchor, which has no ancestors
            anc = d_idx < d
            gp = tl.sum(tl.where(d_idx == d - 1, st_g, 0.0))
            # rebuild the parent's state from the entry tile and the ancestors on the stack
            w = tl.where(anc, tl.exp(gp - st_g), 0.0)
            kw = st_k * w[:, None]                          # [MAXD, DK]
            s = tl.exp(gp) * s0 + tl.dot(tl.trans(kw), st_u)
            # and then it is the chain's own step
            s = s * tl.exp(g)
            kv = tl.sum(s * k[:, None], axis=0)
            delta = (v - kv) * beta
            s = s + k[:, None] * delta[None, :]
            gc = gp + g
            out = tl.sum(s * q[:, None], axis=0)
            tl.store(OUT + t * s_vt + h * DV + ov, out.to(OUT.dtype.element_ty))
            tl.store(DELTA + t * s_vt + h * DV + ov, delta)
            if vb == 0:
                tl.store(GC + t * s_gt + h, gc)
            # push this node at its own depth
            here = (d_idx == d)
            st_k = tl.where(here[:, None], k[None, :], st_k)
            st_u = tl.where(here[:, None], delta[None, :], st_u)
            st_g = tl.where(here, gc, st_g)


MAXD = 16


def fused_tree_step(query, key, value, g, beta, depths, state, *, bv: int = 16,
                    max_depth: int | None = None):
    """The recurrence over a draft tree, one kernel per layer.

    `depths[t]` is node t's depth, the anchor being 0, and the nodes must be in DFS pre-order --
    the same invariant `engine/tree.py` has carried since 09:27, now load-bearing in a third place.

    Returns `out`, `delta` (the per-node rank-1 update vectors, which are the `u` of the factor
    identity) and `gc`, the cumulative gate ALONG EACH PATH rather than along the sequence. Those
    three plus the normalised keys are exactly what `Qwen38Engine.commit_tree` reads, so the fused
    path and the chunked path hand back the same buffer.

    The entry state is NOT advanced: a tree has as many final states as it has leaves.
    """
    if not HAVE_TRITON:
        raise RuntimeError("triton is not available")
    B, T, H, Dk = query.shape
    Dv = value.shape[-1]
    if B != 1:
        raise ValueError(f"fused_tree_step is a single sequence, got B={B}")
    if Dv % bv:
        raise ValueError(f"value head dim {Dv} is not a multiple of bv={bv}")
    # `max_depth` from the caller's host-side copy of the tree: `int(depths.max())` reads the
    # device, which is a synchronisation, and a verify calls this 48 times
    deepest = int(depths.max()) if max_depth is None else int(max_depth)
    if deepest >= MAXD:
        raise ValueError(f"tree is {deepest + 1} deep, kernel carries {MAXD}")
    q = query.reshape(T, H, Dk).contiguous()
    k = key.reshape(T, H, Dk).contiguous()
    v = value.reshape(T, H, Dv).contiguous()
    gg = g.reshape(T, H).contiguous().float()
    bb = beta.reshape(T, H).contiguous().float()
    dd = depths.reshape(T).contiguous().to(torch.int32)
    S = state.reshape(H, Dk, Dv)
    out = torch.empty(T, H, Dv, dtype=query.dtype, device=query.device)
    delta = torch.empty(T, H, Dv, dtype=torch.float32, device=query.device)
    gc = torch.empty(T, H, dtype=torch.float32, device=query.device)
    _gdn_tree_step[(H, Dv // bv)](
        q, k, v, gg, bb, dd, S, out, delta, gc, T,
        H * Dk, H * Dv, H,
        S.stride(0), S.stride(1), S.stride(2),
        DK=Dk, DV=Dv, BV=bv, MAXD=MAXD, EPS=1e-6, SCALE=Dk ** -0.5,
        num_warps=4,
    )
    return out.view(1, T, H, Dv), delta.view(1, T, H, Dv), gc.view(1, T, H)


def check(H: int = 48, Dk: int = 128, Dv: int = 128, *, bv: int = 16, seed: int = 0,
          device: str = "cuda") -> dict:
    """Against `engine.gdn.chunk_gated_delta_rule(tree=...)`, on a real branching tree."""
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from engine import gdn
    from engine.tree import DraftTree

    torch.manual_seed(seed)
    tree = DraftTree(tokens=[0] * 12,
                     parents=[-1, 0, 1, 2, 3, 1, 5, 0, 7, 8, 7, 0])
    tree.check()
    T = len(tree.tokens)
    q = torch.randn(1, T, H, Dk, device=device, dtype=torch.bfloat16)
    k = torch.randn(1, T, H, Dk, device=device, dtype=torch.bfloat16)
    v = torch.randn(1, T, H, Dv, device=device, dtype=torch.bfloat16)
    g = -torch.rand(1, T, H, device=device) * 0.5
    beta = torch.rand(1, T, H, device=device)
    S = torch.randn(1, H, Dk, Dv, device=device) * 0.1
    inc = torch.tensor(tree.ancestor_mask(), dtype=torch.bool, device=device)
    strict = inc & ~torch.eye(T, dtype=torch.bool, device=device)
    o_ref, _, fac = gdn.chunk_gated_delta_rule(q, k, v, g, beta, S, chunk_size=T,
                                               return_factors=True, tree=(inc, strict))
    depths = torch.tensor(tree.depths(), device=device)
    o, delta, gc = fused_tree_step(q, k, v, g, beta, depths, S.clone(), bv=bv)
    k_ref, u_ref, gc_ref = fac
    return {
        "out": (o_ref.float() - o.float()).abs().max().item(),
        "out_absmax": o_ref.float().abs().max().item(),
        "u": (u_ref.transpose(1, 2).float() - delta.float()).abs().max().item(),
        "u_absmax": u_ref.float().abs().max().item(),
        "gc": (gc_ref.transpose(1, 2).float() - gc.float()).abs().max().item(),
        "depth": max(tree.depths()),
    }


def bench(H: int = 48, Dk: int = 128, Dv: int = 128, *, bv: int = 16, iters: int = 200,
          device: str = "cuda", sizes=(8, 16, 32, 64)) -> None:
    """The tree kernel against the chain kernel at the same node count, which is the comparison."""
    import os
    import sys
    import time
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from tools.gdn_kernels import fused_block_step

    print(f"{'nodes':>6} {'chain us':>10} {'tree us':>10} {'ratio':>7}  (per layer, H={H})")
    for T in sizes:
        q = torch.randn(1, T, H, Dk, device=device, dtype=torch.bfloat16)
        k = torch.randn(1, T, H, Dk, device=device, dtype=torch.bfloat16)
        v = torch.randn(1, T, H, Dv, device=device, dtype=torch.bfloat16)
        g = -torch.rand(1, T, H, device=device) * 0.5
        beta = torch.rand(1, T, H, device=device)
        S = torch.randn(1, H, Dk, Dv, device=device) * 0.1
        # a balanced binary tree of T nodes: the shape a budget actually buys
        depths = torch.tensor([max(0, (i + 1).bit_length() - 1) for i in range(T)], device=device)
        for _ in range(5):
            fused_block_step(q, k, v, g, beta, S.clone(), bv=bv)
            fused_tree_step(q, k, v, g, beta, depths, S.clone(), bv=bv)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fused_block_step(q, k, v, g, beta, S, bv=bv)
        torch.cuda.synchronize()
        a = (time.perf_counter() - t0) / iters * 1e6
        t0 = time.perf_counter()
        for _ in range(iters):
            fused_tree_step(q, k, v, g, beta, depths, S, bv=bv)
        torch.cuda.synchronize()
        b = (time.perf_counter() - t0) / iters * 1e6
        print(f"{T:6d} {a:10.1f} {b:10.1f} {b / a:7.2f}")


if __name__ == "__main__":
    import sys
    if "--bench" in sys.argv:
        bench()
    else:
        r = check()
        print(f"out |d| {r['out']:.3e} (absmax {r['out_absmax']:.2f})   "
              f"u |d| {r['u']:.3e} (absmax {r['u_absmax']:.2f})   "
              f"gc |d| {r['gc']:.3e}   tree depth {r['depth']}")
