"""ENG-391: the DFlash2 head's selector (engine/models/qwen3_5_moe/dflash2.py) against the 27B drafter's lattice
(engine/drafters/dflash2.py `DFlash2Module.lattice`) on random tensors, and the greedy walk's path through it; a hot
head's row ids map back to vocabulary ids before the selector reads its codebooks."""

from __future__ import annotations

import os
import sys
import types

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.drafters.dflash2 import DFlash2Module  # noqa: E402
from engine.models.qwen3_5_moe.dflash2 import DFlash2RingModel  # noqa: E402


def _model(V=50, H=16, r=8, k=4):
    g = torch.Generator().manual_seed(0)
    W = {"candidate_selector.hidden_projection.weight": torch.randn(r, H, generator=g),
         "candidate_selector.predecessor_codebook": torch.randn(V, r, generator=g),
         "candidate_selector.successor_codebook": torch.randn(V, r, generator=g)}
    m = DFlash2RingModel.__new__(DFlash2RingModel)
    m.W, m.cfg, m.ids_map = W, types.SimpleNamespace(selector_top_k=k), None
    ref = DFlash2Module.__new__(DFlash2Module)
    ref.w, ref.cfg = W, types.SimpleNamespace(selector_top_k=k)
    return m, ref, g


def test_selector_is_the_lattice():
    m, ref, g = _model()
    rows = torch.randn(6, 16, generator=g)
    logits = torch.randn(6, 50, generator=g)
    top, ids = torch.topk(logits, 4, dim=-1)
    got = m.selector(rows, ids, top, torch.tensor([7]))
    want = ref.lattice(rows, ids, top, 7)
    assert torch.allclose(got, want, atol=1e-5)
    walk = m.walk(ids, got[0, 0], got[1:])
    assert walk[:, 0].long().tolist() == DFlash2Module.walk(ids, want).tolist()


def test_hot_ids_map_back():
    m, ref, g = _model()
    hot = torch.tensor([3, 9, 11, 20, 21, 30, 41, 49])
    rows = torch.randn(3, 16, generator=g)
    hl = torch.randn(3, 8, generator=g)
    m.ids_map = hot
    m.block = lambda *a, **k: torch.cat([torch.zeros(1, 16), rows])
    m.head = lambda r: hl
    out = m.draft(torch.tensor([5]), 0, None, None, None)
    top, idx = torch.topk(hl, 4, dim=-1)
    want = DFlash2Module.walk(hot[idx], ref.lattice(rows, hot[idx], top, 5))
    assert out[:, 0].long().tolist() == want.tolist()


if __name__ == "__main__":
    from scriptmode import run
    run(globals())
