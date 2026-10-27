"""ENG-391: the Qwen3.5 MoE row attention (engine/models/qwen3_5_moe/kernels.py) against its torch twin, row
invariance (a row of an M-row call is the one-row call's row bit for bit), class invariance (more 512-key
programs never change a row) and 64 value heads' worth of query heads per KV head (16). On the GPU, or the CPU
under TRITON_INTERPRET=1; skipped without Triton."""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.models.qwen3_5_moe import kernels as K  # noqa: E402

SKIP = None if (K.HAVE_TRITON and (torch.cuda.is_available() or os.environ.get("TRITON_INTERPRET"))) else \
    "skipped: Triton on a GPU, or TRITON_INTERPRET=1"
DEV = "cuda" if torch.cuda.is_available() and not os.environ.get("TRITON_INTERPRET") else "cpu"


def test_classes():
    assert K.classes(262144) == [4, 8, 16, 32, 64, 128, 256, 512]
    assert K.class_of(1, 262144) == 4 and K.class_of(2049, 262144) == 8 and K.class_of(262144, 262144) == 512


def test_rows_attention():
    if SKIP:
        return SKIP
    g = torch.Generator().manual_seed(0)
    HQ, HK, D, cap, L = 32, 2, 256, 3000, 2100
    kc = (torch.randn(HK, cap, D, generator=g) * 0.5).to(torch.bfloat16).to(DEV)
    vc = torch.randn(HK, cap, D, generator=g).to(torch.bfloat16).to(DEV)
    # a tree of 4 rows after L committed rows: row 0 (depth 0), rows 1, 2 its children, row 3 row 2's child
    parents, depth = [-1, 0, 0, 2], [0, 1, 1, 2]
    M = 4
    q = torch.randn(M, HQ, D, generator=g).to(torch.bfloat16).to(DEV)
    ctl = torch.zeros(M, 4, dtype=torch.int32)
    anc = torch.zeros(M, 33, dtype=torch.int32)
    for i in range(M):
        path, j = [], i
        while j >= 0:
            path.insert(0, j)
            j = parents[j]
        ctl[i, 0], ctl[i, 1] = L + depth[i] + 1, L
        anc[i, :len(path)] = torch.tensor([L + p for p in path])
    ctl, anc = ctl.to(DEV), anc.to(DEV)
    o = K.attn_rows(q, kc, vc, ctl, anc, K.class_of(L + 3, 8192))
    ref = K.attn_rows_reference(q, kc, vc, ctl, anc)
    assert (o.float() - ref).abs().max() < 2e-2 * ref.abs().max()
    for i in range(M):
        one = K.attn_rows(q[i:i + 1], kc, vc, ctl[i:i + 1], anc[i:i + 1], K.class_of(L + 3, 8192))
        assert torch.equal(one[0], o[i]), i
    wide = K.attn_rows(q, kc, vc, ctl, anc, 16)
    assert torch.equal(wide, o)


if __name__ == "__main__":
    from scriptmode import run
    run(globals())
