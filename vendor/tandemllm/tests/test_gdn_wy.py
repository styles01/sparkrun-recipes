"""The WY form of the GDN verify recurrence is the sequential walk's mathematics.

On the CPU, in float64: `tools/gdn_wy_kernels.py::wy_math` -- the same steps the two kernels take (the
path gate from the ancestor mask, (I + A)^-1 from the 8-row diagonal blocks and the block-lower
polynomial, d, o) -- against the row-by-row walk, for chains and for DFS trees of every shape the
served budgets produce, and the block inverse against a plain inverse. The board's kernels are checked
against the same walk in `tests/gpu/test_gdn_wy_gpu.py`.
"""

from __future__ import annotations

import os
import random
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _walk(q, k, v, g, beta, S0, parents):
    """The sequential kernels' order: each node from its parent's state (a chain: the row before)."""
    states, o, d = [], [], []
    for t in range(q.shape[0]):
        s = (S0 if parents[t] < 0 else states[parents[t]]) * g[t].exp()[:, None, None]
        kv = torch.einsum("hk,hkv->hv", k[t], s)
        dt = (v[t] - kv) * beta[t][:, None]
        s = s + k[t][:, :, None] * dt[:, None, :]
        o.append(torch.einsum("hk,hkv->hv", q[t], s))
        d.append(dt)
        states.append(s)
    return torch.stack(o), torch.stack(d, dim=1)


def _case(T, parents, H=3, dk=16, dv=8, seed=0, corr=0.0):
    from engine.tree import DraftTree
    g_ = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g_, dtype=torch.float64)  # noqa: E731
    k = r(T, H, dk) + corr * r(1, H, dk)
    k = k / k.norm(dim=-1, keepdim=True)
    q = r(T, H, dk)
    q = q / q.norm(dim=-1, keepdim=True) * dk ** -0.5
    v = r(T, H, dv)
    g = -torch.rand(T, H, generator=g_, dtype=torch.float64) * 0.5
    beta = torch.sigmoid(r(T, H))
    S0 = r(H, dk, dv) * 0.1
    anc = torch.tensor(DraftTree(tokens=[0] * T, parents=parents).ancestor_mask())
    return q, k, v, g, beta, S0, anc


def _check(T, parents, **kw):
    from tools.gdn_wy_kernels import wy_math
    q, k, v, g, beta, S0, anc = _case(T, parents, **kw)
    o_ref, d_ref = _walk(q, k, v, g, beta, S0, parents)
    o, d, _ = wy_math(q, k, v, g, beta, S0, anc)
    rel = lambda a, b: float((a - b).abs().max() / b.abs().max())  # noqa: E731
    return max(rel(o, o_ref), rel(d, d_ref))


def _random_tree(n, rng):
    parents, path = [-1], [0]
    for i in range(1, n):
        path = path[:rng.randint(1, len(path))]
        parents.append(path[-1])
        path.append(i)
    return parents


def test_chains_of_every_length_are_the_walk():
    worst = max(_check(T, [-1] + list(range(T - 1)), seed=T) for T in range(1, 33))
    assert worst < 1e-12, worst
    return f"chains of 1..32 rows: worst rel {worst:.1e} against the float64 walk"


def test_random_trees_are_the_walk_and_siblings_do_not_see_each_other():
    rng = random.Random(4)
    worst = 0.0
    for n in (2, 5, 9, 16, 17, 24, 31, 32):
        for s in range(3):
            worst = max(worst, _check(n, _random_tree(n, rng), seed=100 * n + s))
    assert worst < 1e-12, worst
    return f"24 random DFS trees of 2..32 nodes: worst rel {worst:.1e}"


def test_correlated_keys_stay_exact():
    """One common direction in every key: the case that makes the doubling series blow up."""
    worst = max(_check(32, p, corr=4.0, seed=7) for p in ([-1] + list(range(31)),
                                                         _random_tree(32, random.Random(9))))
    assert worst < 1e-10, worst
    return f"keys sharing one direction, chain and tree of 32: worst rel {worst:.1e}"


def test_the_block_inverse_is_the_inverse():
    from tools.gdn_wy_kernels import unit_lower_inverse
    g_ = torch.Generator().manual_seed(1)
    for n, b in ((16, 8), (16, 16), (32, 8), (32, 16), (32, 32)):
        L = torch.tril(torch.randn(2, n, n, generator=g_, dtype=torch.float64), -1) * 0.7
        ref = torch.linalg.inv(torch.eye(n, dtype=torch.float64) - L)
        assert torch.allclose(unit_lower_inverse(L, b), ref, rtol=0, atol=1e-9), (n, b)
    return "n 16/32 with blocks of 8/16/32 against torch.linalg.inv"


def test_the_sliced_products_are_the_normalised_products():
    """At a 32-row tile the prep sums the Gram products and the norms' squares over slices of
    32 key channels of the RAW rows and scales by the norms afterwards -- the same numbers as
    normalising first (float64), whatever the slice width."""
    from tools.gdn_wy_kernels import sliced_products
    g_ = torch.Generator().manual_seed(3)
    for T, kc in ((24, 32), (32, 32), (17, 16), (32, 64)):
        q = torch.randn(T, 128, generator=g_, dtype=torch.float64) * 3.0
        k = torch.randn(T, 128, generator=g_, dtype=torch.float64) + 2.0
        kkt, qkt, kn, qn = sliced_products(q, k, kc, 128 ** -0.5)
        kr = k / torch.sqrt((k * k).sum(1, keepdim=True) + 1e-6)
        qr = q / torch.sqrt((q * q).sum(1, keepdim=True) + 1e-6) * 128 ** -0.5
        for a, b in ((kkt, kr @ kr.t()), (qkt, qr @ kr.t()), (kn, kr), (qn, qr)):
            assert torch.allclose(a, b, rtol=1e-12, atol=1e-14), (T, kc)
    return "rows of 17..32, slices of 16/32/64: K K^T, Q K^T and the normalised rows to 1e-12"


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"  {name:56s} ok   {fn() or ''}")
            passed += 1
    print(f"{passed} passed")
