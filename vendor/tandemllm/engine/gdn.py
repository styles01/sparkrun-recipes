"""The gated delta rule and its depthwise causal convolution.

`flash-linear-attention` and `causal_conv1d` are not installed on this board and the interpreter
has no pip, so both are written here. That is not a workaround: the decode path needs a recurrent
step it can capture in a CUDA graph, and block verification needs to run the recurrence over an
arbitrary prefix of a verified block starting from a saved state, and neither is something the
library kernels expose.

Both functions here follow the reference implementation exactly, including the fp32 casts, the
l2 normalisation of q and k, the `1/sqrt(head_dim)` scale applied after it, and the chunk size 64
of the chunked form. `state_shape` is the contract the rest of the engine holds them to.
"""

from __future__ import annotations

import os
from engine.settings import SETTINGS as _S  # noqa: E402  (every QWEN38_* knob)

import torch
import torch.nn.functional as F

# The UT transform's matrix inverse in one `solve_triangular` instead of a serial substitution.
# On by default; set QWEN38_UT_INVERSE=0 to get the loop back and compare.
UT_INVERSE = _S.get("UT_INVERSE") == "1"

# The arithmetic precision of the chunked form's own matmuls, ON A PREFILL ONLY -- fp32, tf32 or
# bf16. The tensors stay fp32 either way and so does the recurrent state; what changes is which
# units do the products. A decode step never reaches this function, and a block verify reaches it
# with eight rows, where the arithmetic is not the cost -- so the mode applies only above
# `PREFILL_MM_FROM` rows and never to a tree.
PREFILL_MM = _S.get("GDN_MM")
PREFILL_MM_FROM = int(_S.get("GDN_MM_FROM"))


def _mm(a: torch.Tensor, b: torch.Tensor, mode: str) -> torch.Tensor:
    """One matmul of the chunked form, at the arithmetic precision the caller asked for.

    The reference keeps every tensor in fp32 and so does this file, because the recurrent state has
    to: the state is a sum over the whole sequence and rounding it is the failure the research note
    records for a bf16 state. The *products*, though, are a different question. Each one is a small
    matrix over one chunk, its inputs are l2-normalised or gated into a bounded range, and its
    result is either re-scaled immediately or accumulated into the fp32 state. On a prefill there
    are 11.6 MFLOP of them per head per chunk, 3.4 TFLOP over a 8k pass, and in true fp32 they run
    on the CUDA cores while the tensor cores idle.

    `tf32` keeps the tensors fp32 and lets the tensor cores do the product at 10 mantissa bits.
    `bf16` casts the operands, which halves the read as well. Both are gated on held-out KL against
    this same function in `fp32`, which is the only reason either is selectable rather than shipped.
    """
    if mode == "bf16":
        return torch.matmul(a.to(torch.bfloat16), b.to(torch.bfloat16)).float()
    if mode == "tf32":
        with _tf32_ctx():
            return torch.matmul(a, b)
    return torch.matmul(a, b)


class _tf32_ctx:
    """`allow_tf32` is a global flag, not a context; this makes it one, and restores it."""

    def __enter__(self):
        self.prev = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = True
        return self

    def __exit__(self, *exc):
        torch.backends.cuda.matmul.allow_tf32 = self.prev
        return False


def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def conv_update(x: torch.Tensor, conv_state: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """One or more steps of the depthwise causal conv, advancing `conv_state` in place.

    x           [B, C, T]      the new projections
    conv_state  [B, C, W - 1 + T'] the last W-1 inputs; updated to the last W-1 of the concatenation
    weight      [C, W]
    """
    width = weight.shape[-1]
    joined = torch.cat([conv_state, x], dim=-1)
    conv_state.copy_(joined[:, :, -(width - 1):] if width > 1 else joined[:, :, :0])
    out = F.conv1d(joined, weight.unsqueeze(1), None, padding=0, groups=x.shape[1])
    return F.silu(out[:, :, -x.shape[-1]:]).to(x.dtype)


def conv_prefill(x: torch.Tensor, weight: torch.Tensor,
                 conv_state: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """The same convolution over a fresh sequence; returns the output and the new conv state."""
    width = weight.shape[-1]
    padded = F.pad(x, (width - 1, 0))
    out = F.silu(F.conv1d(padded, weight.unsqueeze(1), None, padding=0, groups=x.shape[1]))
    new_state = padded[:, :, -(width - 1):].contiguous() if width > 1 else padded[:, :, :0]
    return out.to(x.dtype), new_state


def conv_tree(x: torch.Tensor, conv_state: torch.Tensor, weight: torch.Tensor,
              window: torch.Tensor) -> torch.Tensor:
    """The same depthwise causal convolution, over a draft TREE instead of a chain.

    The convolution of width W at a node reads that node and its W-1 predecessors. In a chain those
    are the W-1 columns to its left; in a tree they are its ancestors, which in DFS pre-order are
    *not* adjacent. `window` is the gather that says so: row i holds W column indices into
    `cat([conv_state, x], -1)`, oldest first, with the node itself last, falling back to the
    convolution state's columns where the path runs out of tree. `engine.tree.conv_windows` builds
    it from the parent array alone, once per tree shape.

    x           [B, C, T]      the block's raw projections, DFS pre-order
    conv_state  [B, C, W - 1]  the last W-1 inputs before the block; NOT advanced here -- a tree has
                               no single successor state, so the commit does it for the accepted path
    weight      [C, W]
    window      [T, W]
    """
    B, C, T = x.shape
    W = window.shape[-1]
    joined = torch.cat([conv_state, x], dim=-1)               # [B, C, W-1+T]
    win = joined[:, :, window]                                # [B, C, T, W]
    # The same kernel the chain path runs, not a hand-written weighted sum. `F.conv1d` on a bf16
    # input accumulates its own way and rounds its own way, and a sum written in fp32 here differs
    # from it by about one bf16 ulp -- which on this stack is not noise, it is a different answer
    # that propagates through 48 layers. Each window is presented as its own batch item with the
    # convolution's own receptive field, so the arithmetic is the reference's, gathered.
    flat = win.permute(0, 2, 1, 3).reshape(B * T, C, W)
    out = F.conv1d(flat, weight.unsqueeze(1), None, padding=0, groups=C)   # [B*T, C, 1]
    out = out.view(B, T, C).permute(0, 2, 1)
    return F.silu(out).to(x.dtype)


def conv_tail(x: torch.Tensor, conv_state: torch.Tensor, path: torch.Tensor,
              width: int) -> torch.Tensor:
    """The convolution state after committing to `path`: the last W-1 raw inputs along it."""
    joined = torch.cat([conv_state, x[:, :, path]], dim=-1)
    return joined[:, :, -(width - 1):] if width > 1 else joined[:, :, :0]


def recurrent_gated_delta_rule(query, key, value, g, beta, state):
    """Token-at-a-time. Shapes [B, T, H, D]; `state` [B, H, Dk, Dv] fp32, advanced in place."""
    dtype = query.dtype
    query = l2norm(query.float(), dim=-1)
    key = l2norm(key.float(), dim=-1)
    query = query * (query.shape[-1] ** -0.5)
    value = value.float()
    beta = beta.float()
    g = g.float()
    B, T, H, Dv = value.shape
    out = torch.empty(B, T, H, Dv, dtype=torch.float32, device=value.device)
    for i in range(T):
        q_t = query[:, i]              # [B, H, Dk]
        k_t = key[:, i]
        v_t = value[:, i]              # [B, H, Dv]
        state.mul_(g[:, i].exp()[:, :, None, None])
        kv = torch.einsum("bhkv,bhk->bhv", state, k_t)
        delta = (v_t - kv) * beta[:, i][:, :, None]
        state.add_(k_t[:, :, :, None] * delta[:, :, None, :])
        out[:, i] = torch.einsum("bhkv,bhk->bhv", state, q_t)
    return out.to(dtype), state


def chunk_gated_delta_rule(query, key, value, g, beta, state=None, chunk_size: int = 64,
                           output_final_state: bool = True, return_factors: bool = False,
                           tree: tuple[torch.Tensor, torch.Tensor] | None = None,
                           mm: str | None = None):
    """Chunked form, for prefill and for verifying a block of drafted tokens.

    Shapes in [B, T, H, D], state [B, H, Dk, Dv] fp32. A copy of `state` is used, never the caller's
    tensor, so the entry state of a speculative block survives the verify pass unchanged.

    `return_factors` additionally hands back the three tensors a *partial* accept needs, and is only
    honoured when the whole call is one chunk -- which every speculative verify is, because the
    caller blocks at the block length. They are the normalised keys, the chunk's cumulative gate,
    and `u`, the pseudo-values the WY transform produces. Their point is this identity:

        S_after_n = exp(gc[n-1]) . S_entry  +  SUM_{t<n} exp(gc[n-1] - gc[t]) . k_t (x) u_t

    `u_t` depends on `S_entry` and on the tokens up to `t` only -- `attn` is unit lower triangular,
    so nothing in row t reads a later row -- and therefore **it does not depend on how many tokens
    of the block are eventually kept**. The state after any prefix is a weighted sum of factors the
    verify pass has already computed, which is a state read and a [Dk, n] x [n, Dv] product per
    layer instead of re-running the recurrence.

    `tree` turns the block into a DRAFT TREE. It is `(ancestor_incl, ancestor_strict)`, two [T, T]
    boolean matrices where row i marks the nodes i may read: its ancestors and itself, and its
    ancestors alone. Three things in the chain form are prefix operations, and each becomes the same
    operation over the ancestor relation instead:

      * the cumulative gate `gc` is a prefix sum along the sequence; on a tree it is the sum along
        the path from the root, which is `ancestor_incl @ g`;
      * the decay between two positions, `exp(gc[i] - gc[j])`, is only meaningful when j is on i's
        path, so it is masked to `ancestor_incl` -- masked *before* the exponential, because
        `gc[i] - gc[j]` for an unrelated j is unbounded above;
      * the UT transform's strictly-lower-triangular `attn` becomes strictly-ancestor-masked.

    That last one is why `engine/tree.py` stores nodes in DFS pre-order and nothing else. The
    ancestor relation is then a subset of the strict lower triangle, so `I - attn` is still unit
    lower triangular and `solve_triangular` is still the right solver -- the tree is a masking
    change to the kernel, not a new algorithm. Because `(I - A)^-1 = I + A + A^2 + ... ` and every
    power of an ancestor-masked matrix is ancestor-masked, row t of the inverse reads only t's own
    path; so `u_t` still depends on nothing below t on its branch, and the factor identity above
    holds for **every root-to-node path in the tree**, which is what makes the commit a gather.

    A tree call must be a single chunk -- the identity is per chunk -- so `chunk_size` must be at
    least T, and no final state is meaningful (a tree has as many final states as leaves). Callers
    take the path they accepted out of the factors.
    """
    if mm is None:
        mm = PREFILL_MM if (tree is None and query.shape[1] > PREFILL_MM_FROM) else "fp32"
    if tree is not None:
        T_in = query.shape[1]
        if chunk_size < T_in:
            chunk_size = T_in
        if chunk_size != T_in:
            raise ValueError(f"a tree verify must be one chunk: T={T_in}, chunk={chunk_size}")
    dtype = query.dtype
    query = l2norm(query, dim=-1, eps=1e-6)
    key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [x.transpose(1, 2).contiguous().to(torch.float32)
                                  for x in (query, key, value, beta, g)]
    B, H, T, Dk = key.shape
    Dv = value.shape[-1]
    pad = (chunk_size - T % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad))
    key = F.pad(key, (0, 0, 0, pad))
    value = F.pad(value, (0, 0, 0, pad))
    beta = F.pad(beta, (0, pad))
    g = F.pad(g, (0, pad))
    Tp = T + pad
    query = query * (Dk ** -0.5)

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [x.reshape(B, H, -1, chunk_size, x.shape[-1])
                                         for x in (query, key, value, k_beta, v_beta)]
    g = g.reshape(B, H, -1, chunk_size)
    if tree is None:
        mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool,
                                     device=query.device), 0)
        g = g.cumsum(dim=-1)
        decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    else:
        anc_incl, anc_strict = tree
        mask = ~anc_strict
        keep = anc_incl.to(g.dtype)
        # gc[i] = sum of g over the path root..i. On a chain this is exactly `cumsum`.
        g = g @ keep.transpose(0, 1)
        diff = g.unsqueeze(-1) - g.unsqueeze(-2)
        decay_mask = diff.masked_fill(~anc_incl, 0).exp().float() * keep.float()
    attn = -(_mm(k_beta, key.transpose(-1, -2), mm) * decay_mask).masked_fill(mask, 0)
    eye = torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    if UT_INVERSE:
        # The loop below is forward substitution: it inverts the unit lower triangular matrix
        # `I - attn` one row at a time. Each iteration is four small operations over [B, H, nc, i],
        # so a chunk of 8 costs 28 kernel launches per layer and a chunk of 64 costs 252, times 48
        # linear layers. On a block verify that is about 1,300 launches over tensors of a few
        # kilobytes; on a prefill of 256 tokens it is over twelve thousand. It is the same matrix
        # inverse either way, and `solve_triangular` does it in one call.
        attn = torch.linalg.solve_triangular(eye - attn, eye.expand_as(attn), upper=False,
                                             unitriangular=True, left=True)
    else:
        for i in range(1, chunk_size):
            row = attn[..., i, :i].clone()
            sub = attn[..., :i, :i].clone()
            attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
        attn = attn + eye
    value = _mm(attn, v_beta, mm)
    k_cumdecay = _mm(attn, k_beta * g.exp().unsqueeze(-1), mm)
    S = (torch.zeros(B, H, Dk, Dv, dtype=torch.float32, device=value.device)
         if state is None else state.clone().float())
    out = torch.zeros_like(value)
    mask = (torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), 1)
            if tree is None else ~tree[0])
    for i in range(Tp // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        a = (_mm(q_i, k_i.transpose(-1, -2), mm) * decay_mask[:, :, i]).masked_fill_(mask, 0)
        v_prime = _mm(k_cumdecay[:, :, i], S, mm)
        v_new = v_i - v_prime          # `u`: the chunk's pseudo-values, causal in t
        inter = _mm(q_i * g[:, :, i, :, None].exp(), S, mm)
        out[:, :, i] = inter + _mm(a, v_new, mm)
        S = (S * g[:, :, i, -1, None, None].exp()
             + _mm((k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2),
                   v_new, mm))
    out = out.reshape(B, H, -1, Dv)[:, :, :T].transpose(1, 2).contiguous().to(dtype)
    if return_factors:
        if Tp // chunk_size != 1:
            factors = None          # the identity above is per chunk; this call is not one chunk
        else:
            factors = (key.reshape(B, H, Tp, Dk)[:, :, :T].contiguous(),
                       v_new.reshape(B, H, Tp, Dv)[:, :, :T].contiguous(),
                       g.reshape(B, H, Tp)[:, :, :T].contiguous())
        return out, (S if output_final_state else None), factors
    return out, (S if output_final_state else None)
