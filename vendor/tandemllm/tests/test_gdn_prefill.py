"""The two pieces of new mathematics in the fused prefill path, on a CPU.

`tools/gdn_prefill_kernels.py` cannot be tested here -- it is Triton and it needs the board. What
CAN be tested without one is the reasoning the kernel is built on, and both parts of it are the
kind that look obviously true and are the reason a kernel is silently wrong:

  1. the UT transform's inverse as a DOUBLING SERIES rather than a forward substitution, and
  2. what the reference does to the PADDED TAIL of a sequence that is not a multiple of the chunk,
     which is what the kernel's masked loads have to reproduce.

Both are checked against `engine/gdn.py` and `torch.linalg.solve_triangular`, which are what the
shipped path uses.
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import gdn                                                     # noqa: E402
from tools.gdn_prefill_kernels import (                                    # noqa: E402
    CHUNK, HAVE_TRITON, STEPS, fused_prefill_refusal,
)

FAILED = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global FAILED
    print(f"{'ok  ' if ok else 'FAIL'} {name}{('   ' + detail) if detail else ''}")
    if not ok:
        FAILED += 1


def doubling_inverse(A: torch.Tensor, steps: int) -> torch.Tensor:
    """What the kernel does: S_1 = I, then S_2n = S_n + A^n . S_n, `steps` times."""
    eye = torch.eye(A.shape[-1], dtype=A.dtype, device=A.device).expand_as(A).contiguous()
    inv, P = eye.clone(), A.clone()
    for _ in range(steps):
        inv = inv + P @ inv
        P = P @ P
    return inv


def test_the_doubling_series_is_the_triangular_solve():
    torch.manual_seed(0)
    A = torch.randn(8, CHUNK, CHUNK, dtype=torch.float32).tril(-1) * 0.3
    eye = torch.eye(CHUNK).expand_as(A)
    ref = torch.linalg.solve_triangular(eye - A, eye.contiguous(), upper=False,
                                        unitriangular=True, left=True)
    got = doubling_inverse(A, STEPS)
    d = (got - ref).abs().max().item()
    check("test_the_doubling_series_is_the_triangular_solve", d < 1e-4,
          f"max |diff| {d:.2e} over 8 matrices of {CHUNK}x{CHUNK}")


def test_the_series_terminates_exactly_at_the_chunk():
    """A^C = 0 for a strictly lower triangular C x C matrix, so SUM_{p<C} A^p is the whole inverse
    and not a truncation. One doubling too few is a different matrix; this asserts both halves."""
    torch.manual_seed(1)
    A = torch.randn(1, CHUNK, CHUNK, dtype=torch.float64).tril(-1)
    P = torch.eye(CHUNK, dtype=torch.float64)[None]
    for _ in range(CHUNK):
        P = P @ A
    check("test_the_series_terminates_exactly_at_the_chunk", P.abs().max().item() == 0.0,
          f"A^{CHUNK} is exactly zero")
    # And the series really does need all of them. On a random matrix it does not look that way --
    # the high powers underflow and five doublings read the same as six -- so the witness is the
    # shift matrix, whose p-th power is the p-th subdiagonal and never decays. Five doublings reach
    # A^31 and miss the other thirty-two terms exactly.
    shift = torch.zeros(1, CHUNK, CHUNK, dtype=torch.float64)
    shift[0, torch.arange(1, CHUNK), torch.arange(CHUNK - 1)] = 1.0
    eye = torch.eye(CHUNK, dtype=torch.float64)[None]
    ref = torch.linalg.solve_triangular(eye - shift, eye.contiguous(), upper=False,
                                        unitriangular=True, left=True)
    full = doubling_inverse(shift, STEPS)
    short = doubling_inverse(shift, STEPS - 1)
    check("test_one_doubling_short_is_a_different_matrix",
          (full - ref).abs().max().item() == 0.0 and (short - ref).abs().max().item() == 1.0,
          f"{STEPS} doublings exact, {STEPS - 1} of them miss "
          f"{int((short - ref).abs().sum().item())} of the {CHUNK * (CHUNK + 1) // 2} entries")


def test_the_padded_tail_carries_the_last_real_gate():
    """`F.pad` puts zeros in g, so the chunk-local cumulative gate is CONSTANT across the padding
    at the last real row's value -- which is the value the state's decay for that chunk uses. A
    kernel that loads a padded row's gate as zero would decay the state by exp(0) instead."""
    T, C = 100, CHUNK
    g = -torch.rand(T) * 0.05
    pad = (C - T % C) % C
    gp = torch.nn.functional.pad(g, (0, pad)).reshape(-1, C).cumsum(-1)
    last_real_in_chunk_1 = gp[1, T - C - 1]
    check("test_the_padded_tail_carries_the_last_real_gate",
          torch.allclose(gp[1, -1], last_real_in_chunk_1),
          f"chunk 1 ends at {gp[1, -1]:.6f}, its last real row is {last_real_in_chunk_1:.6f}")


def test_the_padded_rows_contribute_nothing_to_the_output():
    """The same sequence, run alone and run with junk appended past its length, must give the same
    answer for its own rows and the same final state -- otherwise the kernel's masked stores are
    hiding a state that the padding moved."""
    torch.manual_seed(2)
    T, H, Dk, Dv = 100, 4, 128, 128
    q = torch.randn(1, T, H, Dk, dtype=torch.bfloat16) * 0.5
    k = torch.randn(1, T, H, Dk, dtype=torch.bfloat16) * 0.5
    v = torch.randn(1, T, H, Dv, dtype=torch.bfloat16) * 0.5
    beta = torch.rand(1, T, H, dtype=torch.bfloat16)
    g = -torch.rand(1, T, H, dtype=torch.bfloat16) * 0.05
    S0 = torch.randn(1, H, Dk, Dv) * 0.01
    o1, S1 = gdn.chunk_gated_delta_rule(q, k, v, g, beta, S0, chunk_size=CHUNK)
    # the same call with the tail explicitly zeroed to a chunk boundary, which is what `F.pad` does
    pad = (CHUNK - T % CHUNK) % CHUNK
    z = lambda x, n=pad: torch.cat([x, torch.zeros_like(x[:, :n])], dim=1)  # noqa: E731
    o2, S2 = gdn.chunk_gated_delta_rule(z(q), z(k), z(v), z(g), z(beta), S0, chunk_size=CHUNK)
    do = (o1.float() - o2[:, :T].float()).abs().max().item()
    ds = (S1 - S2).abs().max().item()
    check("test_the_padded_rows_contribute_nothing_to_the_output", do == 0.0 and ds == 0.0,
          f"output {do:.1e}, final state {ds:.1e} over {pad} padded rows")


def test_the_decay_mask_is_lower_inclusive_and_attn_is_strictly_lower():
    """The kernel writes the two masks by hand. They are the reference's, and they are different
    from each other by exactly the diagonal."""
    C = 8
    gc = -torch.arange(C).float() * 0.1
    ref = ((gc[:, None] - gc[None, :]).tril().exp().float()).tril()
    i = torch.arange(C)
    lower = i[:, None] >= i[None, :]
    got = torch.where(lower, torch.where(lower, gc[:, None] - gc[None, :],
                                         torch.zeros(())).exp(), torch.zeros(()))
    check("test_the_decay_mask_is_lower_inclusive", torch.equal(ref, got),
          "diagonal is exp(0) = 1 and the strict upper triangle is 0")
    x = torch.randn(C, C)
    ref_attn = -(x * ref).masked_fill(torch.triu(torch.ones(C, C, dtype=torch.bool), 0), 0)
    got_attn = torch.where(i[:, None] > i[None, :], -(x * got), torch.zeros(()))
    check("test_attn_is_strictly_lower", torch.equal(ref_attn, got_attn),
          "the UT transform's matrix has a zero diagonal")


def test_the_kernel_refuses_a_chunk_it_was_not_built_for():
    """`QWEN38_FUSED_GDNPREFILL=1` is the shipped default and `QWEN38_GDN_CHUNK` is a documented
    knob. The kernel is built for chunk 64 and RAISES on anything else, and on a box without
    Triton -- mid-prefill, after the request has already paid for one. The engine asks first and
    falls back to the reference chunked delta rule, which honours any chunk."""
    assert fused_prefill_refusal(CHUNK, have_triton=True) == ""
    assert "triton" in fused_prefill_refusal(CHUNK, have_triton=False)
    for chunk in (32, 128, 2048):
        why = fused_prefill_refusal(chunk, have_triton=True)
        assert str(chunk) in why and str(CHUNK) in why, why
    assert fused_prefill_refusal(128, have_triton=False) != ""
    # and the default argument is this machine's own answer: a refusal exactly when Triton is
    # missing. It said "this CPU has no Triton" until 2026-09-23, which held on a laptop and failed
    # on the box, where the CPU suite runs beside an installed Triton.
    assert (fused_prefill_refusal(CHUNK) != "") == (not HAVE_TRITON)


if __name__ == "__main__":
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        fn()
    print(f"{FAILED} failed")
    sys.exit(1 if FAILED else 0)
