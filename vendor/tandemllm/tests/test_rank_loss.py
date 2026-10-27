"""RND-37 / RND-38: the drafter's lattice ranking term, on a CPU with a tiny random drafter.

    1  the ranking term is the cross-entropy over the engine's own `lattice()` scores on the true
       predecessor's row, slot by slot, counted only while the true path stays in the lattice
                                                    test_rank_term_is_the_engines_lattice_order
    2  a slot whose true token is not among the candidates ends the counted path
                                                    test_counting_stops_where_the_path_leaves
    3  --rank-loss 0 is today's objective: the shared-logits call gives the same loss bit for bit,
       and a plain run builds the plain AdamW              test_rank_zero_is_todays_loss
    4  a training step moves only the codebook rows it gathered; every other row is bit-identical
                                                    test_untouched_codebook_rows_stay_bit_identical
    5  --train rank selects the selector, fc, the norms and the last two layers, nothing else
                                                    test_rank_mode_selects_the_top_and_the_selector
    6  the in-loop tree eval and a short ranking run work end to end on the tiny module
                                                    test_tree_eval_and_a_short_run

Run: python tests/test_rank_loss.py
"""

from __future__ import annotations

import argparse
import os
import random
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine.drafters.dflash2 import DFlash2Config, DFlash2Module  # noqa: E402
from tools import train_dflash2 as T  # noqa: E402

V, H, K = 400, 32, 4


def tiny(block: int = 8, seed: int = 0, layers: int = 3, v: int = V, k: int = K):
    raw = {"hidden_size": H, "intermediate_size": 64, "num_hidden_layers": layers,
           "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8, "vocab_size": v,
           "rms_norm_eps": 1e-6, "layer_types": ["sliding_attention"] * layers,
           "sliding_window": 2048,
           "dflash_config": {"block_size": block, "conv_kernel_size": 2, "conv_group_size": 8,
                             "mask_token_id": v - 1, "selector_rank": 8, "selector_top_k": k,
                             "target_layer_ids": [1, 2, 3, 4, 5]}}
    cfg = DFlash2Config(raw)
    g = torch.Generator().manual_seed(seed)
    w = {}
    for name, shape in cfg.expected_tensors().items():
        w[name] = torch.ones(shape) if name.endswith("norm.weight") else \
            torch.randn(shape, generator=g) * 0.3
    embed = torch.randn(v, H, generator=g)
    head = torch.randn(v, H, generator=g)
    return DFlash2Module(cfg, w), embed, head


class FakeSample:
    def __init__(self, ids, label, n_fused=0, name="s", topic="prose", gen_start=1):
        self.ids = torch.as_tensor(ids, dtype=torch.long)
        self.label = torch.as_tensor(label, dtype=torch.long)
        self.name, self.topic, self.kind, self.gen_start = name, topic, "gen", gen_start
        self.klass, self.split = topic, "heldout"
        n = self.ids.numel()
        g = torch.Generator().manual_seed(7)
        self.fused = torch.randn(n, 5 * H, generator=g) if n_fused else torch.zeros(n, 5 * H)
        self.top_ids = torch.randint(0, 40, (n, 4), generator=g)
        self.top_lp = torch.log_softmax(torch.randn(n, 4, generator=g), -1)

    def __len__(self):
        return int(self.ids.numel())


def _setup(seed=0, b=3, l=7, out_of_lattice=None):
    """pred/logits for `b` blocks of `l` slots, and labels built FROM the candidates so the true
    path is in the lattice (except at `out_of_lattice = (block, slot)`)."""
    m, embed, head = tiny(seed=seed)
    g = torch.Generator().manual_seed(seed + 1)
    pred = torch.randn(b, l, H, generator=g)
    logits = F.linear(pred.reshape(-1, H), head).float()
    cand, _ = m.unary_candidates(logits)
    cand = cand.view(b, l, K)
    n = 200
    anchors = torch.tensor([10 + 40 * i for i in range(b)])
    ids = torch.randint(0, V - 1, (n,), generator=g)
    label = torch.randint(0, V - 1, (n,), generator=g)
    for bi, a in enumerate(anchors.tolist()):
        for j in range(l):
            c = (bi + j) % K
            label[a + j] = int(cand[bi, j, c])
            if out_of_lattice == (bi, j):
                bad = [t for t in range(V - 1) if t not in cand[bi, j].tolist()][0]
                label[a + j] = bad
    return m, embed, head, pred, logits, cand, anchors, FakeSample(ids, label)


def _expected(m, pred, logits, cand, anchors, s):
    """The ranking CE from the ENGINE's lattice(), row by row, the slow obvious way."""
    b, l, _ = pred.shape
    terms, top1 = [], 0
    _, unary = m.unary_candidates(logits)
    unary = unary.view(b, l, K)
    for bi, a in enumerate(anchors.tolist()):
        sc = m.lattice(pred[bi], cand[bi], unary[bi], int(s.ids[a]))      # [l, k, k]
        row = 0
        for j in range(l):
            tt = int(s.label[a + j])
            row_c = cand[bi, j].tolist()
            if tt not in row_c:
                break
            c = row_c.index(tt)
            terms.append(F.cross_entropy(sc[j, row][None], torch.tensor([c])))
            top1 += int(int(sc[j, row].argmax()) == c)
            row = c
    return torch.stack(terms).mean(), len(terms), top1


def test_rank_term_is_the_engines_lattice_order():
    for seed in range(3):
        m, embed, head, pred, logits, cand, anchors, s = _setup(seed=seed)
        got, n, t1 = T.rank_loss_of(m, pred, logits, s, anchors, "cpu")
        want, n2, t2 = _expected(m, pred, logits, cand, anchors, s)
        assert n == n2 == 3 * 7, (n, n2)
        assert t1 == t2
        assert torch.allclose(got, want, rtol=1e-5, atol=1e-6), (float(got), float(want))


def test_counting_stops_where_the_path_leaves():
    m, embed, head, pred, logits, cand, anchors, s = _setup(seed=4, out_of_lattice=(1, 2))
    got, n, _ = T.rank_loss_of(m, pred, logits, s, anchors, "cpu")
    want, n2, _ = _expected(m, pred, logits, cand, anchors, s)
    assert n == n2 == 7 + 2 + 7, (n, n2)          # block 1 stops before its slot 2
    assert torch.allclose(got, want, rtol=1e-5, atol=1e-6)


def test_rank_zero_is_todays_loss():
    m, embed, head, pred, logits, cand, anchors, s = _setup(seed=5)
    a1, h1 = T.loss_of(pred, head, s, anchors, 0.5, "cpu")
    a2, h2 = T.loss_of(pred, head, s, anchors, 0.5, "cpu", logits=logits)
    assert torch.equal(a1, a2) and torch.equal(h1, h2)
    # a plain run keeps the plain optimiser (the Optim wrapper is only built for ranking runs)
    w = {k: v.clone().requires_grad_(True) for k, v in m.w.items()}
    params = [w[k] for k in T.select_params(w, m.cfg, "all")]
    assert not any(k.startswith("candidate_selector.") for k in T.select_params(w, m.cfg, "all"))
    assert params


def _args(**kw):
    a = argparse.Namespace(steps=3, warmup=1, gen_weight=1.0, blocks=4, kl=0.5, clip=1.0,
                           eval_every=0, budget_min=0.0, state_every=0, rank_loss=1.0,
                           selector_lr=1e-2, wd=0.0, eval_tree="0", eval_blocks=4, block=0,
                           train="rank", state_dtype="bf16", state_max_overhead=0.05)
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def test_untouched_codebook_rows_stay_bit_identical():
    m, embed, head, pred, logits, cand, anchors, s = _setup(seed=6)
    w = m.w
    for k in list(w):
        w[k] = w[k].detach().clone()
    params = T.arm(w, m.cfg, "rank")
    before = {k: w[k].detach().clone() for k in (T.PRED_CB, T.SUCC_CB)}
    opt = T.Optim(w, params, 1e-3, 1e-2, 0.0)
    lg = F.linear(pred.reshape(-1, H), head).float()
    pred_l = pred.clone().requires_grad_(False)
    loss, n, _ = T.rank_loss_of(m, pred_l, lg, s, anchors, "cpu")
    loss.backward()
    opt.clip(1.0)
    opt.step()
    touched_pred = set(s.ids[anchors].tolist())
    b, l = pred.shape[:2]
    for bi, a in enumerate(anchors.tolist()):
        touched_pred |= set(s.label[a:a + l - 1].tolist())
    touched_succ = set(cand.reshape(-1).tolist())
    for name, touched in ((T.PRED_CB, touched_pred), (T.SUCC_CB, touched_succ)):
        after = w[name].detach()
        for r in range(V):
            same = torch.equal(after[r], before[name][r])
            if r in touched:
                assert not same, (name, r, "a gathered row did not move")
            else:
                assert same, (name, r, "an untouched row moved")


def test_rank_mode_selects_the_top_and_the_selector():
    m, _, _ = tiny(layers=4)
    sel = set(T.select_params(m.w, m.cfg, "rank"))
    assert {T.HPROJ, T.PRED_CB, T.SUCC_CB, "fc.weight", "norm.weight", "hidden_norm.weight"} <= sel
    assert all(not k.startswith(("layers.0.", "layers.1.")) for k in sel)
    assert any(k.startswith("layers.2.") for k in sel) and any(k.startswith("layers.3.") for k in sel)
    assert set(T.select_params(m.w, m.cfg, "ranksel")) == {T.HPROJ, T.PRED_CB, T.SUCC_CB}
    sel3 = set(T.select_params(m.w, m.cfg, "rank3"))
    assert any(k.startswith("layers.1.") for k in sel3)


def test_tree_eval_and_a_short_run():
    # a small vocabulary with a wide candidate list, so a random label is usually in the lattice
    v, k = 80, 64
    m, embed, head = tiny(seed=8, v=v, k=k)
    for kk in list(m.w):
        m.w[kk] = m.w[kk].detach().clone().float()
    g = torch.Generator().manual_seed(3)
    n = 120
    ids = torch.randint(0, v - 1, (n,), generator=g)
    label = torch.cat([ids[1:], ids[:1]])

    class S(FakeSample):
        pass
    held = [S(ids, label, n_fused=1, name="h1", gen_start=20),
            S(ids, label, n_fused=1, name="think-prose-00", gen_start=20)]
    ev = T.tree_acceptance(m, embed, head, held, "cpu", budget=8, block=8, max_blocks=6)
    assert 1.0 <= ev["ALL"] <= 8.0 and "think" in ev and "prose" in ev
    assert ev["blocks"] > 0
    # a short ranking run: three steps on the tiny module; gathered rows move, the rest stay put
    before = {kk: m.w[kk].clone() for kk in (T.PRED_CB, T.SUCC_CB)}
    train = [S(ids, label, n_fused=1, name="t1", gen_start=20)]
    params = T.arm(m.w, m.cfg, "rank")
    a = _args()
    T.train_loop(m, m.w, params, embed, head, train, [], held, a, "cpu", 1e-3, random.Random(0),
                 cfg=m.cfg, block=8, mode="rank")
    moved = sum(int(not torch.equal(m.w[T.PRED_CB][r], before[T.PRED_CB][r])) for r in range(v))
    assert 0 < moved < v, moved
    # the selector alone: the backbone stays bit for bit, the selector moves
    m2, embed2, head2 = tiny(seed=8, v=v, k=k)
    for kk in list(m2.w):
        m2.w[kk] = m2.w[kk].detach().clone().float()
    frozen = {kk: t.clone() for kk, t in m2.w.items() if not kk.startswith("candidate_selector.")}
    hp = m2.w[T.HPROJ].clone()
    T.train_loop(m2, m2.w, T.arm(m2.w, m2.cfg, "ranksel"), embed2, head2, train, [], held, _args(),
                 "cpu", 1e-3, random.Random(0), cfg=m2.cfg, block=8, mode="ranksel")
    assert all(torch.equal(m2.w[kk], t) for kk, t in frozen.items())
    assert not torch.equal(m2.w[T.HPROJ], hp)
    never = set(range(v)) - set(ids.tolist()) - set(label.tolist())
    for r in never:                                   # a token that never occurs is never a predecessor
        assert torch.equal(m.w[T.PRED_CB][r], before[T.PRED_CB][r]), r


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok {name}")
