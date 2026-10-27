"""On the board: the WY recurrence against a float64 walk and against the sequential kernels.

`tools/gdn_wy_kernels.py` computes the verify block's recurrence with every row at once. It is the
same mathematics as `_block_step` / `_tree_step` in another order, so the yardstick is a float64
walk of the same inputs: the per-row updates (the commit's factors) and a chain's walked state must be
as close to it as the sequential kernels are, the output as close as its bf16 rounding allows; the
state carried over 1,024 tokens must stay within 1e-6 of the sequential kernel's; and a pending
commit applied in the recurrence must be the commit kernel's bits.
"""

from __future__ import annotations

import os
import random
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from engine.tree import DraftTree  # noqa: E402
from tools import gdn_commit_kernels as CK  # noqa: E402
from tools import gdn_verify_kernels as VK  # noqa: E402
from tools import gdn_wy_kernels as WK  # noqa: E402


def _tree(n, rng, depth_cap=15):
    parents, path = [-1], [0]
    for i in range(1, n):
        path = path[:min(rng.randint(1, len(path)), depth_cap)]
        parents.append(path[-1])
        path.append(i)
    t = DraftTree(tokens=[0] * n, parents=parents)
    t.check()
    return t


def test_wy_is_as_close_to_float64_as_the_walk():
    rng = random.Random(3)
    worst = {}
    cases = [(n, None, 0.0) for n in (2, 3, 8, 9, 16, 17, 24, 32)]
    cases += [(n, _tree(n, rng), 0.0) for n in (4, 9, 16, 24, 32)]
    cases += [(16, None, 3.0), (32, _tree(32, rng), 3.0)]
    for (n, tree, corr), fused in [(c, f) for c in cases for f in (False, True)]:
        r = WK.compare(n, tree, seed=n, corr=corr, fused=fused)
        assert r["u wy"] < 1e-6, (n, tree is not None, fused, r)
        assert r.get("S wy", 0.0) < 2e-6, (n, r)
        assert r["o wy"] <= max(1.2 * r["o seq"], 5e-3), (n, r)
        assert r["conv"] == 0.0, (n, r)                     # a chain's conv state: exact
        if not fused:
            assert r["kk"] < 1e-6 and r["gc"] < 1e-6, (n, r)
        for k, v in r.items():
            worst[k] = max(worst.get(k, 0.0), v)
    return "chains 2..32, 5 trees, correlated keys, conv+gates apart and fused: " + "  ".join(
        f"{k} {v:.1e}" for k, v in worst.items()
        if k in ("u seq", "u wy", "S wy", "o wy", "conv rows differ"))


def test_a_chain_that_stores_its_state_keeps_the_walk():
    """The WY form declines a chain that writes its walked state (the S_T product spills at a
    32-row tile; the served chain never stores): the call is the sequential kernel's, bit for bit."""
    x = WK._inputs(24, torch.Generator(device="cuda").manual_seed(2))
    outs = []
    for wy in (False, True):
        y = {k: v.clone() for k, v in x.items()}
        so = torch.empty_like(y["state"])
        o, fac, _ = VK.verify_mixer(**y, **WK.KW, out_state=so, wy=wy)
        outs.append((o, fac, so))
    (o0, f0, s0), (o1, f1, s1) = outs
    assert torch.equal(o0, o1) and torch.equal(s0, s1)
    assert all(torch.equal(a, b) for a, b in zip(f0, f1))
    return "chain of 24 with out_state: WY on == WY off, bit for bit"


def test_the_state_over_1024_tokens():
    d = WK.drift(blocks=64, T=16)
    assert d < 1e-6, d
    return f"64 chain blocks of 16, WY against sequential: {d:.1e}"


class _sliced:
    """The key-channel slices at a 32-row tile (QWEN38_GDNV_WY_KC) for the life of a check."""

    def __init__(self, kc: int):
        self.kc = kc

    def __enter__(self):
        self.keep = VK.WY_KC
        VK.WY_KC = self.kc

    def __exit__(self, *exc):
        VK.WY_KC = self.keep


def test_the_sliced_kernels_are_as_close_to_float64_as_the_walk():
    """At 17..32 rows, q, k and the state in 32-channel slices -- the same yardstick as
    the kernels: u within 1e-6 of float64, the output at its bf16 rounding, the conv state exact.
    One case sits above 1e-6 in the WY form itself, sliced or not: a 32-row chain of correlated keys,
    fused (u 1.1e-6 whole, 1.3e-6 sliced over three seeds, hold 3 of 2026-09-25; the walk 9e-8). There
    the slices are held to the whole-key kernel instead: within 1.5x of its u."""
    rng = random.Random(11)
    worst = {}
    cases = [(n, None, 0.0) for n in (17, 24, 31, 32)]
    cases += [(n, _tree(n, rng), 0.0) for n in (17, 24, 24, 32)]
    cases += [(24, _tree(24, rng), 3.0), (32, None, 3.0)]
    for (n, tree, corr), fused in [(c, f) for c in cases for f in (False, True)]:
        with _sliced(32):
            r = WK.compare(n, tree, seed=100 + n, corr=corr, fused=fused)
        bound = 1e-6
        if r["u wy"] >= bound:
            with _sliced(0):
                whole = WK.compare(n, tree, seed=100 + n, corr=corr, fused=fused)
            bound = max(bound, 1.5 * whole["u wy"])
            worst["u whole"] = max(worst.get("u whole", 0.0), whole["u wy"])
        assert r["u wy"] < bound, (n, tree is not None, fused, bound, r)
        assert r.get("S wy", 0.0) < 2e-6, (n, r)
        assert r["o wy"] <= max(1.2 * r["o seq"], 5e-3), (n, r)
        assert r["conv"] == 0.0, (n, r)
        if not fused:
            assert r["kk"] < 1e-6 and r["gc"] < 1e-6, (n, r)
        for k, v in r.items():
            worst[k] = max(worst.get(k, 0.0), v)
    return "slices of 32 at 17..32 rows, chains and trees, apart and fused: " + "  ".join(
        f"{k} {v:.1e}" for k, v in worst.items() if k in ("u seq", "u wy", "u whole", "S wy", "o wy"))


def test_a_16_row_tile_is_never_sliced():
    """QWEN38_GDNV_WY_KC leaves every block of <= 16 rows on the kernels shipped: bit for bit."""
    rng = random.Random(12)
    for n, tree in ((16, None), (9, None), (16, _tree(16, rng)), (12, _tree(12, rng))):
        x = WK._inputs(n, torch.Generator(device="cuda").manual_seed(n))
        ta = WK._tree_args(tree) if tree is not None else {}
        for fused in (False, True):
            outs = []
            for kc in (0, 32):
                y = {k: v.clone() for k, v in x.items()}
                with _sliced(kc), WK.every_size():
                    o, fac, _ = VK.verify_mixer(**y, **WK.KW, **ta, wy=True, fused=fused,
                                                store_state=False)
                outs.append((o, fac, y["state"], y["conv_state"]))
            (o0, f0, s0, c0), (o1, f1, s1, c1) = outs
            assert torch.equal(o0, o1) and torch.equal(s0, s1) and torch.equal(c0, c1), (n, fused)
            assert all(torch.equal(a, b) for a, b in zip(f0, f1)), (n, fused)
    return "chains of 9 and 16, trees of 12 and 16, apart and fused: KC 32 == KC 0, bit for bit"


def test_a_pending_commit_into_a_sliced_verify_is_the_commit_kernel_bit_for_bit():
    """The apply writes the pending commit back a slice at a time with `_pending`'s arithmetic:
    the state, outputs and factors are commit-then-verify's bits at 24 and 32 rows."""
    rng = random.Random(13)
    cases = ((2, _tree(24, rng)), (16, _tree(24, rng)), (7, _tree(32, rng)), (16, None))
    with _sliced(32):
        n = _pending_commit_cases(cases, nxt_chain=24)
    return f"{n} commits of 2..16 rows into 24/32-row sliced verifies: state, outputs, factors identical"


def test_a_pending_commit_is_the_commit_kernel_bit_for_bit():
    """(a) commit kernel, then the WY verify; (b) the WY verify with the commit pending: the state it
    writes back, its outputs and its factors are the same bits (commits of 2+ rows; a one-row commit
    is the commit kernel's P = 1 specialisation, the documented ulp)."""
    rng = random.Random(8)
    cases = ((2, None), (7, None), (16, None), (5, _tree(9, rng)), (16, _tree(24, rng)))
    checked = _pending_commit_cases(cases)
    return (f"{checked} commits of 2..16 rows into chain and tree verifies, conv+gates apart and "
            f"fused: state, outputs, factors and conv state bit-identical")


def _pending_commit_cases(cases, nxt_chain: int = 16) -> int:
    kw = WK.KW
    gen = torch.Generator(device="cuda").manual_seed(5)
    n_prev = 16
    checked = 0
    size = WK.every_size()
    size.__enter__()
    for (rows_n, nxt_tree), fused in [(c, f) for c in cases for f in (False, True)]:
        prev = WK._inputs(n_prev, gen)
        scratch = torch.empty_like(prev["state"])
        _, (pk, pu, pg), _ = VK.verify_mixer(**prev, **kw, out_state=scratch, wy=True)
        entry = prev["state"]
        rows = list(range(rows_n))
        nt = len(nxt_tree.parents) if nxt_tree is not None else nxt_chain
        nxt = WK._inputs(nt, gen)
        ta = WK._tree_args(nxt_tree) if nxt_tree is not None else {}
        Sa = entry.clone()
        CK.fused_commit(Sa[None], Sa[None], pk, pu, pg, rows)
        committed = Sa.clone()
        ia = {k: v.clone() for k, v in nxt.items() if k != "state"}
        oa, fa, _ = VK.verify_mixer(**ia, state=Sa, **ta, **kw, wy=True, fused=fused,
                                    store_state=False)
        Sb = entry.clone()
        ib = {k: v.clone() for k, v in nxt.items() if k != "state"}
        pend = (pk[0], pu[0], pg[0], torch.tensor(rows, dtype=torch.int32, device="cuda"),
                torch.tensor([len(rows)], dtype=torch.int32, device="cuda"))
        ob, fb, _ = VK.verify_mixer(**ib, state=Sb, **ta, **kw, wy=True, fused=fused,
                                    pend=pend, store_state=False)
        assert torch.equal(Sb, committed), (rows_n, (Sb - committed).abs().max().item())
        assert torch.equal(oa, ob) and all(torch.equal(x, y) for x, y in zip(fa, fb)), rows_n
        assert torch.equal(ia["conv_state"], ib["conv_state"]), rows_n
        checked += 1
    size.__exit__()
    return checked


def test_the_served_thresholds_route_by_rows():
    """QWEN38_GDNV_WY_MAXT / _CHAIN_MAXT: past them a block walks -- its outputs are the walk's bits."""
    old = (VK.WY_MAXT, VK.WY_CHAIN_MAXT, VK.WY_FUSED_MAXT)
    rng = random.Random(4)
    try:
        VK.WY_MAXT = VK.WY_CHAIN_MAXT = VK.WY_FUSED_MAXT = 16
        for n, tree in ((24, None), (24, _tree(24, rng)), (17, None)):
            x = WK._inputs(n, torch.Generator(device="cuda").manual_seed(n))
            ta = WK._tree_args(tree) if tree is not None else {}
            outs = []
            for wy in (False, True):
                y = {k: v.clone() for k, v in x.items()}
                o, fac, _ = VK.verify_mixer(**y, **WK.KW, **ta, wy=wy, fused=wy, store_state=False)
                outs.append((o, fac))
            assert torch.equal(outs[0][0], outs[1][0]), n
            assert all(torch.equal(a, b) for a, b in zip(outs[0][1], outs[1][1])), n
    finally:
        VK.WY_MAXT, VK.WY_CHAIN_MAXT, VK.WY_FUSED_MAXT = old
    return "thresholds 16: 17- and 24-row chains and a 24-node tree take the walk, bit for bit"


def test_a_tree_without_its_mask_is_refused():
    x = WK._inputs(4, torch.Generator(device="cuda").manual_seed(1))
    t = DraftTree(tokens=[0] * 4, parents=[-1, 0, 0, 1])
    ta = WK._tree_args(t)
    ta.pop("anc")
    try:
        with WK.every_size():
            VK.verify_mixer(**x, **WK.KW, **ta, wy=True)
    except ValueError as e:
        assert "ancestor mask" in str(e)
    else:
        raise AssertionError("a tree without its ancestor mask must be refused")
    return "refused with a ValueError"


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"  {name:56s} ok   {fn() or ''}")
            passed += 1
    print(f"{passed} passed")
