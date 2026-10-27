"""On the board (ENG-227): Flash-Next's sequence kernels (tools/fn_seq_kernels.py) and the row form of
attn_prep / gate_mul (tools/fn_kernels.py).

The property under test is the one lossless speculation needs: a row of a verify block -- chain or
tree, its predecessors in the block or in the committed state -- is the decode step's row at that
position bit for bit. For each kernel: the block's rows equal one-row calls replayed along each row's
own path (the decode step's walk), the commit walk leaves the decode walk's state, and the one-row
call equals a torch reference within its bf16/fp32 rounding.

Run on the box: python tests/gpu/test_fn_seq_kernels_gpu.py   (under flock ~/.qwen38-box.flock)
"""

from __future__ import annotations

import os
import random
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from tools import fn_kernels as K  # noqa: E402
from tools import fn_seq_kernels as SK  # noqa: E402

DEV = "cuda"
H, KH, D = 48, 16, 128                    # GDN value heads, key heads, head dim
KD = KH * D
C = 2 * KD + H * D                        # the conv channels: q | k | v
TREE = [-1, 0, 1, 2, 3, 2, 1, 6, 0, 8, 8, 10]  # DFS pre-order, siblings at depths 1, 2, 3


def _g(seed):
    return torch.Generator(device=DEV).manual_seed(seed)


def _paths(parents):
    out = []
    for i, p in enumerate(parents):
        out.append([i] if p < 0 else out[p] + [i])
    return out


def test_tree_helpers():
    ctl, n = SK.tree_ctl(TREE)
    assert ctl == [(-2, 0), (-1, 1), (-1, 2), (-1, -1), (-1, -1), (2, -1), (1, -1), (-1, -1), (0, 3),
                   (-1, -1), (3, -1), (-1, -1)] and n == 4, ctl
    src = SK.window_src(TREE, (3, 2, 1), 3)
    assert src[0] == [-1, -2, -3] and src[3] == [0, 1, 2] and src[5] == [0, 1, 2] and src[1] == [-2, -3, 0], src
    assert src[11] == [0, 8, 10] and src[9] == [-3, 0, 8], src
    return f"ctl and windows of a {len(TREE)}-row tree"


def _gdn_inputs(M, seed):
    g = _g(seed)
    qkv = torch.randn(M, C, generator=g, device=DEV).to(torch.bfloat16)
    gg = -torch.rand(M, H, generator=g, device=DEV) * 0.5
    beta = torch.rand(M, H, generator=g, device=DEV)
    S = torch.randn(1, H, D, D, generator=g, device=DEV) * 0.1
    return qkv, gg, beta, S


def test_gdn_rows_tree_equals_decode_walk():
    M = len(TREE)
    qkv, gg, beta, S = _gdn_inputs(M, 1)
    stk = torch.empty(31, H, D, D, device=DEV)
    ctl = torch.tensor(SK.tree_ctl(TREE)[0], dtype=torch.int32, device=DEV)
    out = SK.gdn_rows(qkv, KD, gg, beta, S.clone(), stk, ctl, 3, False)
    one = torch.tensor([[-2, -1]], dtype=torch.int32, device=DEV)
    for i, pa in enumerate(_paths(TREE)):
        s = S.clone()
        for r in pa:                                   # the decode step along the row's path
            o = SK.gdn_rows(qkv[r:r + 1], KD, gg[r:r + 1], beta[r:r + 1], s, stk, one, 3, True)
        assert torch.equal(o[0], out[i]), i
    # the commit walk (a chain over the path, written) leaves the decode walk's state
    pa = _paths(TREE)[9]
    idx = torch.tensor(pa, device=DEV)
    chain = torch.tensor([[-2, -1]] + [[-1, -1]] * (len(pa) - 1), dtype=torch.int32, device=DEV)
    s1 = S.clone()
    SK.gdn_rows(qkv[idx], KD, gg[idx], beta[idx], s1, stk, chain, 3, True)
    s2 = S.clone()
    for r in pa:
        SK.gdn_rows(qkv[r:r + 1], KD, gg[r:r + 1], beta[r:r + 1], s2, stk, one, 3, True)
    assert torch.equal(s1, s2)
    ref, _ = SK.gdn_rows_reference(qkv, KD, gg, beta, S, TREE, 3)
    rel = ((out.float() - ref).abs().max() / ref.abs().max()).item()
    assert rel < 1e-2, rel
    return f"{M}-row tree = decode walk per row, commit walk = decode state, vs fp32 rel {rel:.1e}"


def test_gdn_rows_parked_commit_folded():
    """Parked rows applied by the next call (the folded commit) leave the state of a separate commit
    walk, and the call's rows after them equal the rows after a separate commit, bit for bit."""
    M = len(TREE)
    qkv, gg, beta, S = _gdn_inputs(M + 5, 3)
    stk = torch.empty(31, H, D, D, device=DEV)
    ctl = torch.tensor(SK.tree_ctl(TREE)[0], dtype=torch.int32, device=DEV)
    chain = torch.tensor([[-2, -1]] + [[-1, -1]] * 4, dtype=torch.int32, device=DEV)
    pre = (qkv[M:].contiguous(), gg[M:].contiguous(), beta[M:].contiguous())
    s1 = S.clone()
    SK.gdn_rows(pre[0], KD, pre[1], pre[2], s1, stk, chain, 3, True)            # the commit walk
    o1 = SK.gdn_rows(qkv[:M], KD, gg[:M], beta[:M], s1.clone(), stk, ctl, 3, False)
    s2 = S.clone()
    n = torch.tensor([5], dtype=torch.int32, device=DEV)
    o2 = SK.gdn_rows(qkv[:M], KD, gg[:M], beta[:M], s2, stk, ctl, 3, False, pre=pre + (n,))
    assert torch.equal(s1, s2) and torch.equal(o1, o2)
    z = torch.tensor([0], dtype=torch.int32, device=DEV)                       # nothing parked: no change
    s3 = S.clone()
    o3 = SK.gdn_rows(qkv[:M], KD, gg[:M], beta[:M], s3, stk, ctl, 3, False, pre=pre + (z,))
    assert torch.equal(s3, S) and torch.equal(o3, SK.gdn_rows(qkv[:M], KD, gg[:M], beta[:M], S.clone(), stk, ctl, 3, False))
    return "5 parked rows folded into a 12-row tree call = commit walk then the call, bit for bit"


def test_gdn_pre_windows_equal_decode_shift():
    M = len(TREE)
    g = _g(2)
    x = torch.randn(M, C + 64, generator=g, device=DEV).to(torch.bfloat16)
    cs = torch.randn(1, C, 3, generator=g, device=DEV).to(torch.bfloat16)
    w = (torch.randn(C, 4, generator=g, device=DEV) * 0.3).to(torch.bfloat16)
    ab = torch.randn(M, 2 * H, generator=g, device=DEV).to(torch.bfloat16)
    alog = torch.randn(H, generator=g, device=DEV)
    dtb = torch.randn(H, generator=g, device=DEV)
    src = torch.tensor(SK.window_src(TREE, (3, 2, 1), 3), dtype=torch.int32, device=DEV)
    y, gg, beta = SK.gdn_pre(x, cs.clone(), w, src, ab, alog, dtb)
    one = torch.tensor([[-1, -2, -3]], dtype=torch.int32, device=DEV)
    for i, pa in enumerate(_paths(TREE)):
        s = cs.clone()
        for r in pa:
            yo, go, bo = SK.gdn_pre(x[r:r + 1], s, w, one, ab[r:r + 1], alog, dtb, shift=True)
        assert torch.equal(yo[0], y[i]) and torch.equal(go[0], gg[i]) and torch.equal(bo[0], beta[i]), i
    # one row against torch: conv over [state, x], silu; the gates
    j = torch.cat([cs[0].float(), x[0, :C].float()[:, None]], 1)
    acc = (j * w.float()).sum(1)
    want = torch.nn.functional.silu(acc)
    dy = (y[0].float() - want).abs().max().item()
    assert dy <= want.abs().max().item() * 2 ** -7, dy
    gw = -alog.exp() * torch.nn.functional.softplus(ab[0, :H].float() + dtb)
    assert torch.allclose(gg[0], gw, rtol=1e-5, atol=1e-6)
    assert torch.allclose(beta[0], torch.sigmoid(ab[0, H:].float()), rtol=1e-5, atol=1e-6)
    return "tree windows = the decode shift along each path; one row vs torch within a bf16 step"


def test_ple_gate_and_conv_rows():
    M, HC, HD = 6, 4, 2560
    Cp = HC * HD
    g = _g(3)
    key = torch.randn(M, Cp, generator=g, device=DEV).to(torch.bfloat16)
    qry = torch.randn(M, Cp, generator=g, device=DEV).to(torch.bfloat16)
    val = torch.randn(M, HD, generator=g, device=DEV).to(torch.bfloat16)
    gv = SK.ple_gate(key, qry, val, HC)
    for m in range(M):
        assert torch.equal(SK.ple_gate(key[m:m + 1], qry[m:m + 1], val[m:m + 1], HC)[0], gv[m]), m
    # the reference's bf16 ops
    k4, q4 = key.view(M, HC, HD), qry.view(M, HC, HD)
    gate = (k4 * q4).sum(-1, keepdim=True) / HD ** 0.5
    gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
    want = (torch.sigmoid(gate) * val.unsqueeze(-2)).flatten(-2)
    steps = ((gv.float() - want.float()).abs() / want.float().abs().clamp_min(1e-3)).max().item()
    assert steps < 3 * 2 ** -7, steps
    # the conv: a chain of M rows over a state of 9, against one-row calls with the shifted state
    par = list(range(-1, M - 1))
    ps = torch.randn(1, Cp, 9, generator=g, device=DEV).to(torch.bfloat16)
    wc = (torch.randn(Cp, 4, generator=g, device=DEV) * 0.3).to(torch.bfloat16)
    gvn = torch.randn(M, Cp, generator=g, device=DEV).to(torch.bfloat16)
    src = torch.tensor(SK.window_src(par, (9, 6, 3), 9), dtype=torch.int32, device=DEV)
    y = SK.ple_conv(gvn, gv, ps, wc, src)
    one = torch.tensor([[-1, -4, -7]], dtype=torch.int32, device=DEV)
    s = ps.clone()
    for m in range(M):
        assert torch.equal(SK.ple_conv(gvn[m:m + 1], gv[m:m + 1], s, wc, one)[0], y[m]), m
        s = torch.cat([s[..., 1:], gvn[m:m + 1].T.unsqueeze(0)], -1)
    joined = torch.cat([ps, gvn[:1].T.unsqueeze(0)], -1)                  # row 0 against F.conv1d
    ref = torch.nn.functional.conv1d(joined, wc.view(Cp, 1, 4), groups=Cp, dilation=3)
    ref = gv[0] + torch.nn.functional.silu(ref[0, :, -1])
    dd = (y[0].float() - ref.float()).abs().max().item()
    assert dd <= 2 * ref.float().abs().max().item() * 2 ** -7, dd
    return f"gate and conv rows = one-row calls; gate vs bf16 torch {steps:.1e} rel, conv vs F.conv1d {dd:.1e}"


def _attn_setup(L, M, seed):
    g = _g(seed)
    HQ, HK, AD, cap = 24, 2, 256, 4096
    kc = torch.randn(HK, cap, AD, generator=g, device=DEV).to(torch.bfloat16)
    vc = torch.randn(HK, cap, AD, generator=g, device=DEV).to(torch.bfloat16)
    q = torch.randn(M, HQ, AD, generator=g, device=DEV).to(torch.bfloat16)
    return q, kc, vc


def _ctl(parents, L):
    M = len(parents)
    paths = _paths(parents)
    ctl = torch.zeros(M, 4, dtype=torch.int32)
    anc = torch.zeros(M, SK.ANC, dtype=torch.int32)
    for i, pa in enumerate(paths):
        ctl[i, :3] = torch.tensor([L + len(pa), L, 1])
        anc[i, :len(pa)] = torch.tensor([L + a for a in pa])
    return ctl.to(DEV), anc.to(DEV)


def test_attn_rows_tree_equals_decode_and_reference():
    out = []
    for L in (5, 300, 1500, 2030):
        M = len(TREE)
        q, kc, vc = _attn_setup(L, M, L)
        ctl, anc = _ctl(TREE, L)
        sel = torch.zeros(M, SK.NSEL, dtype=torch.int32, device=DEV)
        o = SK.attn_rows(q, kc, vc, ctl, anc, sel)
        # decode: each row alone, its path copied to contiguous positions L.. of a second cache
        for i, pa in enumerate(_paths(TREE)):
            k2, v2 = kc.clone(), vc.clone()
            for d, r in enumerate(pa):
                k2[:, L + d], v2[:, L + d] = kc[:, L + r], vc[:, L + r]
            p = L + len(pa) - 1
            c1 = torch.tensor([[p + 1, p, 1, 0]], dtype=torch.int32, device=DEV)
            a1 = torch.zeros(1, SK.ANC, dtype=torch.int32, device=DEV)
            a1[0, 0] = p
            o1 = SK.attn_rows(q[i:i + 1], k2, v2, c1, a1, sel[:1])
            assert torch.equal(o1[0], o[i]), (L, i)
        ref = SK.attn_rows_reference(q, kc, vc, ctl, anc, sel)
        rel = ((o.float() - ref).abs().max() / ref.abs().max()).item()
        assert rel < 2e-2, (L, rel)
        out.append(f"L{L} rel {rel:.1e}")
    return "tree rows = decode rows at their positions; " + ", ".join(out)


def test_attn_rows_selected_positions():
    M, L = 3, 3000
    q, kc, vc = _attn_setup(L, M, 9)
    ctl, anc = _ctl([-1, 0, 1], L)
    sel = torch.zeros(M, SK.NSEL, dtype=torch.int32, device=DEV)
    rng = random.Random(4)
    for t in range(M):
        p = L + t
        blocks = sorted(rng.sample(range((p + 1) // 4), 512))
        pos = [4 * b + j for b in blocks for j in range(4)] + list(range(4 * ((p + 1) // 4), p + 1))
        sel[t, :len(pos)] = torch.tensor(pos, dtype=torch.int32)
        ctl[t, 0], ctl[t, 2] = len(pos), 0
    o = SK.attn_rows(q, kc, vc, ctl, anc, sel)
    ref = SK.attn_rows_reference(q, kc, vc, ctl, anc, sel)
    rel = ((o.float() - ref).abs().max() / ref.abs().max()).item()
    assert rel < 2e-2, rel
    for t in range(M):                                 # a row alone gives its bits
        assert torch.equal(SK.attn_rows(q[t:t + 1], kc, vc, ctl[t:t + 1], anc[t:t + 1], sel[t:t + 1])[0], o[t])
    return f"QSA-selected keys vs fp32 rel {rel:.1e}, rows = one-row calls"


def test_attn_prep_and_gate_mul_rows():
    g = _g(41)
    Hq, Hk, AD, cap, rot, ihd, M = 24, 2, 256, 64, 64, 128, 5
    qkv = torch.randn(M, Hq * 2 * AD + 2 * Hk * AD, generator=g, device=DEV).to(torch.bfloat16)
    iqk = torch.randn(M, 5 * ihd, generator=g, device=DEV).to(torch.bfloat16)
    qw = (torch.randn(AD, generator=g, device=DEV) * 0.1).to(torch.bfloat16)
    kw = (torch.randn(AD, generator=g, device=DEV) * 0.1).to(torch.bfloat16)
    ang = torch.arange(cap, device=DEV)[:, None].float() * torch.rand(rot // 2, generator=g, device=DEV)[None]
    cos = torch.cat([ang.cos(), ang.cos()], -1).to(torch.bfloat16)
    sin = torch.cat([ang.sin(), ang.sin()], -1).to(torch.bfloat16)
    pos = torch.tensor([10, 11, 12, 11, 12], device=DEV)
    slot = torch.tensor([10, 11, 12, 13, 14], device=DEV)
    kc = torch.zeros(Hk, cap, AD, dtype=torch.bfloat16, device=DEV)
    vc, ikc = torch.zeros_like(kc), torch.zeros(cap, ihd, dtype=torch.bfloat16, device=DEV)
    q = K.attn_prep(qkv, iqk, pos, qw, kw, cos, sin, kc, vc, ikc, Hq, Hk, 4 * ihd, 1e-6, slot=slot)
    for m in range(M):
        k1, v1, i1 = torch.zeros_like(kc), torch.zeros_like(vc), torch.zeros_like(ikc)
        q1 = K.attn_prep(qkv[m:m + 1], iqk[m:m + 1], pos[m:m + 1], qw, kw, cos, sin, k1, v1, i1, Hq, Hk, 4 * ihd, 1e-6)
        p, s = int(pos[m]), int(slot[m])
        assert torch.equal(q1[0], q[m]) and torch.equal(k1[:, p], kc[:, s]) and torch.equal(v1[:, p], vc[:, s])
        assert torch.equal(i1[p], ikc[s])
    o = torch.randn(M, Hq, AD, generator=g, device=DEV).to(torch.bfloat16)
    y = K.gate_mul(o, qkv, AD)
    for m in range(M):
        assert torch.equal(K.gate_mul(o[m:m + 1], qkv[m:m + 1], AD)[0], y[m])
    return "rows (rope at the position, cache at the slot) = one-row calls; gate rows = one-row calls"


def _rope_tables(n, rot=64):
    inv = 1.0 / (1e7 ** (torch.arange(0, rot, 2, device=DEV).float() / rot))
    ang = torch.arange(n, device=DEV)[:, None].float() * inv[None]
    return torch.cat([ang.cos(), ang.cos()], -1).to(torch.bfloat16), torch.cat([ang.sin(), ang.sin()], -1).to(torch.bfloat16)


def test_qsa_pool_scores_and_selection():
    """Pooled keys within a bf16 step of the reference's arithmetic; scores rows = one-row calls and
    the own blocks equal the shared pool's for the same keys; the selection over 3,000 random keys
    picks the reference's blocks."""
    g = _g(11)
    cap, L, NH, M = 4096, 3000, 4, 3
    eps = 1e-6
    ikc = torch.randn(cap, 128, generator=g, device=DEV).to(torch.bfloat16)
    kln = (torch.randn(128, generator=g, device=DEV) * 0.1).to(torch.bfloat16)
    cos, sin = _rope_tables(cap)
    nb = L // 4
    b = torch.arange(nb, dtype=torch.int32, device=DEV)
    slots = (b[:, None] * 4 + torch.arange(4, dtype=torch.int32, device=DEV)).contiguous()
    pool = SK.qsa_pool(ikc, slots, b * 4, kln, cos, sin, eps)
    gm = ikc[: nb * 4].view(nb, 4, 128).float().mean(1).to(torch.bfloat16).float()
    gm = (gm * torch.rsqrt(gm.pow(2).mean(-1, keepdim=True) + eps) * (1 + kln.float())).to(torch.bfloat16).float()
    c, sn = cos[b.long() * 4].float(), sin[b.long() * 4].float()
    xr = gm[:, :64]
    rot = torch.cat([-xr[:, 32:], xr[:, :32]], -1)
    ref = torch.cat([((xr * c).to(torch.bfloat16).float() + (rot * sn).to(torch.bfloat16).float()), gm[:, 64:]], -1)
    steps = ((pool.float() - ref).abs() / ref.abs().clamp_min(1e-2)).max().item()
    assert steps < 3 * 2 ** -7, steps
    iq = torch.randn(M, NH, 128, generator=g, device=DEV).to(torch.bfloat16)
    # rows at positions L - 1 + t: own blocks from block (L - 1) // 4, pooled from the same slots
    own0 = (L - 1) // 4
    ob = own0 + torch.arange(SK.OWN, device=DEV, dtype=torch.int32)
    os_ = (ob[:, None] * 4 + torch.arange(4, dtype=torch.int32, device=DEV)).contiguous()
    own = SK.qsa_pool(ikc, os_, ob * 4, kln, cos, sin, eps).unsqueeze(0).expand(M, -1, -1).contiguous()
    qctl = torch.zeros(M, 4, dtype=torch.int32, device=DEV)
    for t in range(M):
        qctl[t, 0], qctl[t, 1] = (L - 1 + t + 1) // 4, own0
    NB = 1024
    sc = SK.qsa_scores(iq, pool, own, qctl, NB)
    for t in range(M):
        assert torch.equal(SK.qsa_scores(iq[t:t + 1], pool, own[t:t + 1], qctl[t:t + 1], NB)[0], sc[t]), t
    shared = qctl.clone()
    shared[:, 1] = 10 ** 6                              # every block from the shared pool
    full = SK.qsa_scores(iq, pool, own, shared, NB)
    assert torch.equal(full, sc), "own blocks pooled from the same keys must score as the shared pool"
    sel = torch.zeros(M, SK.NSEL, dtype=torch.int32, device=DEV)
    tail = torch.stack([qctl[:, 0] * 4 + k for k in range(3)], 1).int().contiguous()
    SK.qsa_select(sc, tail, sel)
    overlap = []
    for t in range(M):
        p = L - 1 + t
        want = SK.qsa_reference(iq[t], ikc, kln, cos, sin, p, eps)
        got = sel[t, :len(want)].long()
        assert torch.equal(got[2048:], want[2048:]), (t, got[2048:], want[2048:])      # the tail
        overlap.append(len(set(got[:2048].tolist()) & set(want[:2048].tolist())) / 2048)
    assert min(overlap) >= 0.99, overlap
    return f"pool within {steps:.1e} rel of the reference; score rows = one-row calls; selection overlap {min(overlap):.4f}"


def test_attn_prep_indexer_query():
    g = _g(42)
    Hq, Hk, AD, cap, ihd, M = 24, 2, 256, 64, 128, 4
    qkv = torch.randn(M, Hq * 2 * AD + 2 * Hk * AD, generator=g, device=DEV).to(torch.bfloat16)
    iqk = torch.randn(M, 5 * ihd, generator=g, device=DEV).to(torch.bfloat16)
    qw = kw = (torch.randn(AD, generator=g, device=DEV) * 0.1).to(torch.bfloat16)
    qln = (torch.randn(ihd, generator=g, device=DEV) * 0.1).to(torch.bfloat16)
    cos, sin = _rope_tables(cap)
    pos = torch.tensor([3, 9, 9, 20], device=DEV)
    kc = torch.zeros(Hk, cap, AD, dtype=torch.bfloat16, device=DEV)
    vc, ikc = torch.zeros_like(kc), torch.zeros(cap, ihd, dtype=torch.bfloat16, device=DEV)
    iq = torch.zeros(M, 4, ihd, dtype=torch.bfloat16, device=DEV)
    K.attn_prep(qkv, iqk, pos, qw, kw, cos, sin, kc, vc, ikc, Hq, Hk, 4 * ihd, 1e-6, qln=qln, iq=iq)
    from engine.models.flashnext.ref import apply_rope, rms
    want = apply_rope(rms(iqk[:, :4 * ihd].view(M, 4, ihd), qln, 1e-6), cos[pos][:, None], sin[pos][:, None])
    steps = ((iq.float() - want.float()).abs() / want.float().abs().clamp_min(1e-2)).max().item()
    assert steps < 3 * 2 ** -7, steps
    for m in range(M):
        one = torch.zeros(1, 4, ihd, dtype=torch.bfloat16, device=DEV)
        K.attn_prep(qkv[m:m + 1], iqk[m:m + 1], pos[m:m + 1], qw, kw, cos, sin, kc.clone(), vc.clone(), ikc.clone(),
                    Hq, Hk, 4 * ihd, 1e-6, qln=qln, iq=one)
        assert torch.equal(one[0], iq[m]), m
    return f"indexer query (norm 1 + w, rope at the position) within {steps:.1e} rel of torch; rows = one-row calls"


def test_qsa_topk_kernel_equals_torch():
    """The one-kernel selection = the torch selection (stable top-512 with ties, ascending, tail)."""
    g = _g(13)
    out = []
    for NB, M in ((1024, 3), (2048, 2), (8192, 4)):
        sc = (torch.randn(M, NB, generator=g, device=DEV) * 4).round()            # many exact ties
        nblk = torch.randint(513, NB + 1, (M,), generator=g, device=DEV)
        sc[torch.arange(NB, device=DEV)[None, :] >= nblk[:, None]] = float("-inf")
        tail = (nblk[:, None] * 4 + torch.arange(3, device=DEV)[None, :]).int().contiguous()
        a = torch.zeros(M, SK.NSEL, dtype=torch.int32, device=DEV)
        b = torch.zeros_like(a)
        SK.qsa_select(sc, tail, a)
        SK.qsa_select_torch(sc, tail, b)
        assert torch.equal(a, b), NB
        out.append(NB)
    return f"bit-equal to the torch sorts at {out} blocks with ties"


TESTS = [test_tree_helpers, test_gdn_rows_tree_equals_decode_walk, test_gdn_rows_parked_commit_folded, test_gdn_pre_windows_equal_decode_shift,
         test_ple_gate_and_conv_rows, test_attn_rows_tree_equals_decode_and_reference,
         test_attn_rows_selected_positions, test_attn_prep_and_gate_mul_rows, test_qsa_pool_scores_and_selection,
         test_attn_prep_indexer_query, test_qsa_topk_kernel_equals_torch]


if __name__ == "__main__":
    bad = 0
    for t in TESTS:
        try:
            print(f"PASS {t.__name__}: {t()}", flush=True)
        except Exception as e:                          # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f"FAIL {t.__name__}: {e!r}", flush=True)
            bad += 1
    print(f"{len(TESTS) - bad}/{len(TESTS)} passed")
    sys.exit(1 if bad else 0)
