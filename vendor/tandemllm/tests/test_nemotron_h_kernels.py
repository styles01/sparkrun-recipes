"""NemotronH's row kernels against their torch twins, and the row invariance speculation needs: a row
of an M-row call equals the one-row call bit for bit (the Mamba2 conv and SSM: a chain block's row
equals the decode steps that reach it, each writing its state). On the GPU, or on the CPU under
TRITON_INTERPRET=1."""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.models.nemotron_h import kernels as K  # noqa: E402

SKIP = None if (K.HAVE_TRITON and (torch.cuda.is_available() or os.environ.get("TRITON_INTERPRET"))) else \
    "skipped: Triton on a GPU, or TRITON_INTERPRET=1 (the CPU interpreter)"
DEV = "cuda" if torch.cuda.is_available() and not os.environ.get("TRITON_INTERPRET") else "cpu"
# the CPU interpreter truncates bf16 casts (a step of 2^-8 relative); the GPU rounds
TOL = 3e-3 if DEV == "cuda" else 5e-2
g = torch.Generator().manual_seed(0)


def rnd(*shape, s=1.0, dtype=torch.bfloat16):
    return (torch.randn(*shape, generator=g) * s).to(dtype).to(DEV)


def close(a, b, tol=TOL, what=""):
    a, b = a.float(), b.float()
    err = ((a - b).abs() / (b.abs() + 1.0)).max().item()
    assert err < tol, f"{what}: {err}"


# Lightning's Mamba2 shapes scaled down: heads 8 x 16, state 24 (not a power of two, as Bolt's 96), groups 2
HH, HD, N, G, KT = 8, 16, 24, 2, 4
I = HH * HD
CD = I + 2 * G * N
PD = I + CD + HH


def _mamba_inputs(M):
    proj = rnd(M, PD)
    wc, bc = rnd(CD, KT, s=0.5), rnd(CD, s=0.1)
    state = rnd(KT - 1, CD)
    S0 = rnd(HH, HD, N, s=0.3, dtype=torch.float32)
    A = -torch.exp(rnd(HH, s=0.5, dtype=torch.float32))
    dtb = rnd(HH, s=0.5, dtype=torch.float32)
    Dp = 1 + rnd(HH, s=0.1, dtype=torch.float32)
    return proj, wc, bc, state, S0, A, dtb, Dp


def _anc(parents):
    paths = []
    for i, p in enumerate(parents):
        paths.append([i] if p < 0 else paths[p] + [i])
    anc = torch.zeros(len(parents), K.ANC, dtype=torch.int32)
    for i, pa in enumerate(paths):
        anc[i, :len(pa)] = torch.tensor(pa)
    return anc.to(DEV), torch.tensor([len(p) for p in paths], dtype=torch.int32, device=DEV)


def test_conv_and_ssm_twins_on_a_tree():
    parents = [-1, 0, 1, 0, 3, 4, 1, 6]
    M = len(parents)
    proj, wc, bc, state, S0, A, dtb, Dp = _mamba_inputs(M)
    src = torch.tensor(K.window_src(parents, KT), dtype=torch.int32, device=DEV)
    xbc = K.conv_rows(proj, I, state, wc, bc, src)
    close(xbc, K.conv_rows_reference(proj, I, state, wc, bc, src), what="conv")
    anc, nanc = _anc(parents)
    y = K.ssm_rows(proj, I + CD, xbc, anc, nanc, S0, A, dtb, Dp, G)
    yr, _ = K.ssm_rows_reference(proj, I + CD, xbc, anc, nanc, S0, A, dtb, Dp, G)
    close(y, yr, what="ssm")


def test_a_chain_row_is_the_decode_steps_bit_for_bit():
    for M in (1, 2, 7, 32):
        proj, wc, bc, state, S0, A, dtb, Dp = _mamba_inputs(M)
        src = torch.tensor(K.window_src(list(range(-1, M - 1)), KT), dtype=torch.int32, device=DEV)
        xbc = K.conv_rows(proj, I, state, wc, bc, src)
        anc, nanc = _anc(list(range(-1, M - 1)))
        y = K.ssm_rows(proj, I + CD, xbc, anc, nanc, S0, A, dtb, Dp, G)
        st, S = state.clone(), S0.clone()
        src1 = torch.tensor(K.window_src([-1], KT), dtype=torch.int32, device=DEV)
        a1, n1 = _anc([-1])
        for r in range(M):
            x1 = K.conv_rows(proj[r:r + 1], I, st, wc, bc, src1, write=True)
            assert torch.equal(x1[0], xbc[r]), (M, r, "conv")
            y1 = K.ssm_rows(proj[r:r + 1], I + CD, x1, a1, n1, S, A, dtb, Dp, G, write=True)
            assert torch.equal(y1[0], y[r]), (M, r, "ssm")
        # the commit of the whole chain: the walk over the path, written, is the decode steps' state
        Sc = S0.clone()
        pa = torch.zeros(1, K.ANC, dtype=torch.int32, device=DEV)
        pa[0, :M] = torch.arange(M)
        K.ssm_rows(proj, I + CD, xbc, pa, torch.tensor([M], dtype=torch.int32, device=DEV), Sc, A, dtb, Dp, G,
                   write=True)
        assert torch.equal(Sc, S), (M, "commit")


def test_a_tree_row_is_its_paths_decode_steps_bit_for_bit():
    """A block walks the SSM once a head (`_ssm_walk`): a row continuing the row before it goes on
    from that row's state, a branch walks its path from S0 again. Every row is the decode steps along
    its path, bit for bit."""
    parents = [-1, 0, 1, 0, 3, 4, 1, 6, 7, 0]
    M = len(parents)
    proj, wc, bc, state, S0, A, dtb, Dp = _mamba_inputs(M)
    src = torch.tensor(K.window_src(parents, KT), dtype=torch.int32, device=DEV)
    xbc = K.conv_rows(proj, I, state, wc, bc, src)
    anc, nanc = _anc(parents)
    y = K.ssm_rows(proj, I + CD, xbc, anc, nanc, S0, A, dtb, Dp, G)
    a1, n1 = _anc([-1])
    for r in range(M):
        S = S0.clone()
        for a in anc[r, :int(nanc[r])].tolist():
            y1 = K.ssm_rows(proj[a:a + 1], I + CD, xbc[a:a + 1], a1, n1, S, A, dtb, Dp, G, write=True)
        assert torch.equal(y1[0], y[r]), r


def test_gated_norm_add_rms_router():
    M = 9
    y, z, w = rnd(M, I), rnd(M, I), 1 + rnd(I, s=0.1)
    out = K.gated_norm(y, z, w, G, 1e-5)
    close(out, K.gated_norm_reference(y, z, w, G, 1e-5), what="gated_norm")
    for r in (0, 4, 8):
        assert torch.equal(K.gated_norm(y[r:r + 1], z[r:r + 1], w, G, 1e-5)[0], out[r])
    h, d, wn = rnd(M, 96), rnd(M, 96), 1 + rnd(96, s=0.1)
    hn, nn = K.add_rms(h, d, wn, 1e-5)
    hr, nr = K.add_rms_reference(h, d, wn, 1e-5)
    if DEV == "cuda":
        assert torch.equal(hn, hr)
    close(hn, hr, what="add")
    close(nn, nr, what="add_rms")
    for E, k in ((12, 3), (40, 7)):
        lg = rnd(M, E, s=2.0, dtype=torch.float32)
        lg[1, 3] = lg[1, 5]                            # a tie: the lower id
        bias = rnd(E, s=0.05, dtype=torch.float32)
        bias[3] = bias[5]
        ids, wts = K.router(lg, bias, k, 2.5)
        ri, rw = K.router_reference(lg, bias, k, 2.5)
        assert torch.equal(ids.cpu(), ri.cpu()), (E, k)
        close(wts, rw, 1e-6, what="router")
        for r in (0, 1, 8):
            i1, w1 = K.router(lg[r:r + 1], bias, k, 2.5)
            assert torch.equal(i1[0], ids[r]) and torch.equal(w1[0], wts[r])


def test_moe_relu2_and_combine():
    M, S, E, Ii, KIN = 5, 3, 6, 64, 96
    x = rnd(M, KIN)
    wu, wd = rnd(E, Ii, KIN, s=0.1), rnd(E, KIN, Ii, s=0.1)
    ids = torch.stack([torch.randperm(E, generator=g)[:S] for _ in range(M)]).to(torch.int32).to(DEV)
    wts = rnd(M, S, dtype=torch.float32).abs()
    yd = K.moe_relu2(x, ids, wu, wd)
    close(yd, K.moe_relu2_reference(x, ids, wu, wd), what="moe")
    out = K.combine(yd, ids, wts)
    close(out, K.combine_reference(yd, ids, wts), what="combine")
    for r in range(M):
        y1 = K.moe_relu2(x[r:r + 1], ids[r:r + 1], wu, wd)
        assert torch.equal(y1, yd[r * S:(r + 1) * S])
        assert torch.equal(K.combine(y1, ids[r:r + 1], wts[r:r + 1])[0], out[r])


if __name__ == "__main__":
    if SKIP:
        print(SKIP)
        sys.exit(0)
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok {name}")
    print("all nemotron_h kernel tests passed")
