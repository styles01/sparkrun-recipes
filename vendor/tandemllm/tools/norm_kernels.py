"""The two RMS norms as single kernels.

`engine/model.py` writes both norms the way the published implementation writes them, as a sequence
of torch operations, because that is how they were checked against it. At one token per step the
sequence is the cost: `rms_norm` alone is a cast, a square, a mean, an add, a reciprocal square
root, two multiplies and a cast back -- eight launches over 5,120 numbers, 20 KB, which no launch
can amortise. The engine runs 129 of them per token and the ledger's breakdown charges 5.1 ms for
it, an effective 0.5 GB/s.

There are two conventions in this model and they are not interchangeable:

    rms_norm         normalise(x) * (1 + w)        input / post-attention / final / q / k
    rms_norm_gated   w * normalise(x) * silu(z)    the linear-attention output norm

Both are written here with the same fp32 cast points and the same operation order as the reference,
so the difference against it is reduction order alone. `check()` reports that difference; the
engine-level gates decide whether it may ship.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import gdc_launch_dependents
    HAVE_TRITON = True
except ImportError:                                            # pragma: no cover
    HAVE_TRITON = False


def _pdl() -> bool:
    """Release the next kernel early -- only when the projections are launched to wait."""
    from tools import nvfp4_skinny
    return nvfp4_skinny.PDL and nvfp4_skinny.SKINNY


if HAVE_TRITON:

    @triton.jit
    def _rms_norm(X, W, Y, N, EPS: tl.constexpr, BLOCK: tl.constexpr, PDL: tl.constexpr = False):
        if PDL:
            # the projection that reads this norm may launch now and fetch its weights
            gdc_launch_dependents()
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        m = cols < N
        x = tl.load(X + row * N + cols, mask=m, other=0.0).to(tl.float32)
        rstd = tl.rsqrt(tl.sum(x * x) / N + EPS)
        w = tl.load(W + cols, mask=m, other=0.0).to(tl.float32)
        y = x * rstd * (1.0 + w)
        tl.store(Y + row * N + cols, y.to(Y.dtype.element_ty), mask=m)

    @triton.jit
    def _add_rms_norm(R, X, W, H, Y, N, EPS: tl.constexpr, BLOCK: tl.constexpr,
                      PDL: tl.constexpr = False):
        """H = R + X rounded to H's dtype, then `_rms_norm` of the ROUNDED sum: the two launches it
        replaces, in one, with the same arithmetic in the same order."""
        if PDL:
            gdc_launch_dependents()
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        m = cols < N
        r = tl.load(R + row * N + cols, mask=m, other=0.0).to(tl.float32)
        a = tl.load(X + row * N + cols, mask=m, other=0.0).to(tl.float32)
        h = (r + a).to(H.dtype.element_ty)
        tl.store(H + row * N + cols, h, mask=m)
        x = h.to(tl.float32)
        rstd = tl.rsqrt(tl.sum(x * x) / N + EPS)
        w = tl.load(W + cols, mask=m, other=0.0).to(tl.float32)
        y = x * rstd * (1.0 + w)
        tl.store(Y + row * N + cols, y.to(Y.dtype.element_ty), mask=m)

    @triton.jit
    def _rms_norm_gated(X, Z, W, Y, N, EPS: tl.constexpr, BLOCK: tl.constexpr,
                        PDL: tl.constexpr = False):
        if PDL:
            gdc_launch_dependents()
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        m = cols < N
        x = tl.load(X + row * N + cols, mask=m, other=0.0).to(tl.float32)
        rstd = tl.rsqrt(tl.sum(x * x) / N + EPS)
        # The reference rounds to the input dtype BEFORE the weight multiply, does that multiply in
        # bf16, and only then promotes to fp32 for the gate. Not the obvious order, and matching it
        # is the difference between a drop-in and a change of numerics.
        w = tl.load(W + cols, mask=m, other=0.0).to(Y.dtype.element_ty)
        h = (w * (x * rstd).to(Y.dtype.element_ty)).to(tl.float32)
        z = tl.load(Z + row * N + cols, mask=m, other=0.0).to(tl.float32)
        y = h * (z * tl.sigmoid(z))
        tl.store(Y + row * N + cols, y.to(Y.dtype.element_ty), mask=m)


def _block(n: int) -> int:
    b = 1
    while b < n:
        b *= 2
    return b


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """`normalise(x) * (1 + weight)`, fused. Drop-in for `engine.model.rms_norm`."""
    n = x.shape[-1]
    flat = x.reshape(-1, n)
    if not flat.is_contiguous():
        flat = flat.contiguous()
    y = torch.empty_like(flat)
    _rms_norm[(flat.shape[0],)](flat, weight, y, n, EPS=eps, BLOCK=_block(n),
                                PDL=_pdl(), num_warps=8 if n >= 4096 else 4)
    return y.view_as(x)


def add_rms_norm(res: torch.Tensor, x: torch.Tensor, weight: torch.Tensor,
                 eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    """`h = res + x; (h, rms_norm(h))` in one launch, bit-identical to the two it replaces."""
    n = x.shape[-1]
    r = res.reshape(-1, n).contiguous()
    a = x.reshape(-1, n).contiguous()
    h = torch.empty_like(r)
    y = torch.empty_like(r)
    _add_rms_norm[(r.shape[0],)](r, a, weight, h, y, n, EPS=eps, BLOCK=_block(n),
                                 PDL=_pdl(), num_warps=8 if n >= 4096 else 4)
    return h.view_as(res), y.view_as(res)


def rms_norm_gated(x: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor,
                   eps: float) -> torch.Tensor:
    """`weight * normalise(x) * silu(gate)`, fused. Drop-in for `engine.model.rms_norm_gated`."""
    n = x.shape[-1]
    flat = x.reshape(-1, n).contiguous()
    g = gate.reshape(-1, n).contiguous()
    y = torch.empty_like(flat)
    _rms_norm_gated[(flat.shape[0],)](flat, g, weight, y, n, EPS=eps, BLOCK=_block(n),
                                      PDL=_pdl(), num_warps=8 if n >= 4096 else 4)
    return y.view_as(x)


# --------------------------------------------------------------------------- the gate

def check_add(device: str = "cuda", rows=(1, 8, 16, 260)) -> list[str]:
    """`add_rms_norm` against `res + x` and `rms_norm` of it: bit-identical or it is not a drop-in."""
    out = []
    for m in rows:
        res = torch.randn(m, 5120, device=device).to(torch.bfloat16)
        x = torch.randn(m, 5120, device=device).to(torch.bfloat16)
        w = (torch.randn(5120, device=device) * 0.1).to(torch.bfloat16)
        h_ref = res + x
        y_ref = rms_norm(h_ref, w, 1e-6)
        h, y = add_rms_norm(res, x, w, 1e-6)
        assert torch.equal(h, h_ref) and torch.equal(y, y_ref), m
        out.append(f"add_rms_norm rows {m}: sum and norm bit-identical")
    return out


def check(device: str = "cuda") -> list[str]:
    from engine.model import rms_norm as ref, rms_norm_gated as ref_g
    lines = []
    for m, n in ((1, 5120), (8, 5120), (16, 5120), (1, 256), (48, 128)):
        x = torch.randn(1, m, n, device=device, dtype=torch.bfloat16)
        w = torch.randn(n, device=device, dtype=torch.bfloat16) * 0.1
        a, b = ref(x, w, 1e-6), rms_norm(x, w, 1e-6)
        d = float((a.float() - b.float()).abs().max())
        rel = d / max(float(a.float().abs().max()), 1e-9)
        lines.append(f"rms_norm       [{m:3d}, {n:5d}]  absmax {d:.3e}  rel {rel:.2e}")
    for m, n in ((48, 128), (384, 128), (720, 128)):
        x = torch.randn(m, n, device=device, dtype=torch.bfloat16)
        z = torch.randn(m, n, device=device, dtype=torch.bfloat16)
        w = torch.randn(n, device=device, dtype=torch.bfloat16) * 0.1
        a, b = ref_g(x, z, w, 1e-6), rms_norm_gated(x, z, w, 1e-6)
        d = float((a.float() - b.float()).abs().max())
        rel = d / max(float(a.float().abs().max()), 1e-9)
        lines.append(f"rms_norm_gated [{m:3d}, {n:5d}]  absmax {d:.3e}  rel {rel:.2e}")
    return lines


def bench(device: str = "cuda", iters: int = 500) -> list[str]:
    import time
    from engine.model import rms_norm as ref
    out = []
    x = torch.randn(1, 1, 5120, device=device, dtype=torch.bfloat16)
    w = torch.randn(5120, device=device, dtype=torch.bfloat16) * 0.1
    for name, fn in (("reference", lambda: ref(x, w, 1e-6)),
                     ("fused", lambda: rms_norm(x, w, 1e-6))):
        for _ in range(50):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        us = (time.perf_counter() - t0) / iters * 1e6
        out.append(f"{name:10s} {us:7.2f} us per 5120-wide norm   "
                   f"({us * 129 / 1000:.2f} ms for the step's 129)")
    return out


if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for line in check():
        print(line)
    print()
    for line in bench():
        print(line)
