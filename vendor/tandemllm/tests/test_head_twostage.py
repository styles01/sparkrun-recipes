"""Stage 0: the two-stage head's bound is sound, and the candidate set is what it says.

The bound is the whole exactness argument, so it is tested where it is tight: a quantised copy whose
error is the largest the bound allows in one direction, near-ties at the top, and rank 16. The
e4m3 and NVFP4 heads are stand-ins here (a random head and a perturbed copy); the tool itself runs
the real pair on the board.
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _pair(n=512, k=256, noise=0.05, seed=0):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(n, k, generator=g)
    q = w + noise * torch.randn(n, k, generator=g)
    return w, q


def test_the_bound_holds_on_every_row_and_rank():
    from tools.head_twostage import candidates, error_norms, radius, sound
    w, q = _pair()
    en = error_norms(w, q, rows=100)
    h = torch.randn(200, 256, generator=torch.Generator().manual_seed(1)) * 3
    s, s_hat = h @ w.T, h @ q.T
    for mode in ("row", "group"):
        rad = radius(h, en, mode)
        assert ((s - s_hat).abs() <= rad).all(), mode
        for k in (1, 16):
            m = candidates(s_hat, rad, k)
            assert sound(s, m, k) == 0, (mode, k)
            assert (m.sum(dim=1) >= k).all()


def test_the_group_bound_is_never_looser_than_the_row_bound():
    from tools.head_twostage import error_norms, radius
    w, q = _pair(seed=2)
    en = error_norms(w, q)
    h = torch.randn(50, 256)
    assert (radius(h, en, "group") <= radius(h, en, "row") * (1 + 1e-5)).all()


def test_an_aligned_error_reaches_the_row_bound():
    """H parallel to one row's error: Cauchy-Schwarz is attained, so the bound cannot be cut."""
    from tools.head_twostage import error_norms, radius
    w, q = _pair(n=8, k=64, seed=3)
    en = error_norms(w, q)
    h = (w[5] - q[5]).unsqueeze(0) * 7
    realised = ((h @ w.T) - (h @ q.T)).abs()[0, 5]
    assert realised <= radius(h, en, "row")[0, 5]
    assert realised >= 0.999 * (h.norm() * en["e"][5])


def test_near_ties_keep_both_and_a_clear_winner_keeps_one():
    from tools.head_twostage import candidates
    s_hat = torch.tensor([[10.0, 9.99, 3.0, -1.0], [10.0, 2.0, 1.0, 0.0]])
    rad = torch.full_like(s_hat, 0.1)
    m = candidates(s_hat, rad, 1)
    assert m[0].tolist() == [True, True, False, False]
    assert m[1].tolist() == [True, False, False, False]


def test_a_miss_is_counted():
    from tools.head_twostage import sound
    s = torch.tensor([[1.0, 5.0, 2.0]])
    assert sound(s, torch.tensor([[True, False, True]]), 1) == 1
    assert sound(s, torch.tensor([[False, True, False]]), 1) == 0


def test_the_nvfp4_rows_are_the_exact_product_not_the_bf16_one():
    from tools.head_twostage import nvfp4_rows
    from tools.nvfp4_linear import quantize_to_nvfp4
    w = torch.randn(32, 128, generator=torch.Generator().manual_seed(4)) * 0.037
    nv = quantize_to_nvfp4(w)
    exact = nvfp4_rows(nv, 0, 32)
    assert torch.equal(exact.to(torch.bfloat16), nv.dequant()), "same values, before the rounding"
    assert (exact - w).abs().max() < w.abs().max() / 4


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"  {name:56s} ok   {fn() or ''}")
            passed += 1
    print(f"{passed} passed")
