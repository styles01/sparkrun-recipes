"""ENG-287: the LFM2 MoE board engine on a tiny BF16 model: the row step against the reference forward,
a verify block's rows against the decode step's (bit for bit), commit and rollback, the chunked prefill.
On the GPU, or on the CPU under TRITON_INTERPRET=1 (whose bf16 casts truncate: the reference comparison
is then within a few bf16 steps; the engine against itself is bit for bit either way)."""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.models.lfm2_moe import kernels as K  # noqa: E402
from engine.models.lfm2_moe.config import config_from_dict  # noqa: E402
from engine.models.lfm2_moe.load import ExpertsBF16  # noqa: E402
from engine.models.lfm2_moe.ref import Lfm2MoeRef, chain  # noqa: E402
from tests.test_lfm2_moe_ref import TINY, _random_weights  # noqa: E402

SKIP = None if (K.HAVE_TRITON and (torch.cuda.is_available() or os.environ.get("TRITON_INTERPRET"))) else "skipped: Triton on a GPU, or TRITON_INTERPRET=1 (the CPU interpreter)"
DEV = "cuda" if torch.cuda.is_available() and not os.environ.get("TRITON_INTERPRET") else "cpu"
TINY_B = {**TINY, "vocab_size": 256, "hidden_size": 256, "intermediate_size": 256, "moe_intermediate_size": 128,
          "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 64, "num_experts": 4, "num_experts_per_tok": 2}
PROMPT = torch.tensor([5, 9, 3, 17, 22, 3, 40, 41, 12, 7, 63, 2, 30, 88])


_TINY = []


def tiny():
    if _TINY:
        return _TINY[0]
    from engine.models.lfm2_moe.engine import Lfm2MoeEngine
    cfg = config_from_dict(TINY_B, "tiny")
    W0 = _random_weights(cfg)
    W = {k: (v.float() if k.endswith("expert_bias") else v.to(torch.bfloat16)).to(DEV) for k, v in W0.items()
         if ".experts." not in k}
    experts = {}
    for li in cfg.moe_layers:
        f = f"layers.{li}.feed_forward.experts."
        w13 = torch.stack([torch.cat([W0[f + f"{e}.w1.weight"], W0[f + f"{e}.w3.weight"]]) for e in range(cfg.num_experts)])
        w2 = torch.stack([W0[f + f"{e}.w2.weight"] for e in range(cfg.num_experts)])
        experts[li] = ExpertsBF16(w13.to(torch.bfloat16).to(DEV).contiguous(), w2.to(torch.bfloat16).to(DEV).contiguous())
    for li in range(cfg.num_hidden_layers):
        p = f"layers.{li}."
        if cfg.is_attention(li):
            a = p + "self_attn."
            W[a + "qkv"] = torch.cat([W[a + f"{x}_proj.weight"] for x in ("q", "k", "v")]).contiguous()
        if not cfg.is_moe(li):
            f = p + "feed_forward."
            W[f + "w13"] = torch.cat([W[f + "w1.weight"], W[f + "w3.weight"]]).contiguous()
    W["lm_head.weight"] = W["embed_tokens.weight"]
    eng = Lfm2MoeEngine(cfg, W, experts, device=DEV, max_len=128)
    ref = Lfm2MoeRef(cfg, {k: v.float().cpu() for k, v in W0.items()})
    _TINY.append((cfg, eng, ref))
    return _TINY[0]


def _decode(eng, st, toks):
    out = []
    for t in toks:
        eng.stage(t, st)
        lg = eng.step(st).clone()
        st.length += 1
        out.append(lg[0])
    return torch.stack(out)


def test_prefill_and_decode_near_the_reference():
    if SKIP:
        return SKIP
    cfg, eng, ref = tiny()
    st = eng.new_state(128)
    lg = eng.prefill(PROMPT, st)
    rs = ref.new_state(64)
    want = ref.prefill(PROMPT, rs)[-1]
    assert (lg[0].cpu() - want).abs().max() < 0.06 * want.abs().max()
    got = _decode(eng, st, [11, 50, 3]).cpu()
    for i, t in enumerate((11, 50, 3)):
        w = ref.decode(t, rs)[0]
        assert (got[i] - w).abs().max() < 0.08 * w.abs().max(), i


def test_verify_rows_are_the_decode_steps():
    if SKIP:
        return SKIP
    cfg, eng, ref = tiny()
    st = eng.new_state(128)
    eng.prefill(PROMPT, st)
    L = st.length
    conv0 = {li: t.clone() for li, t in st.conv.items()}
    toks = [7, 19, 23, 44, 31, 60]
    want = _decode(eng, st, toks)
    st.length = L
    for li, t in conv0.items():
        st.conv[li].copy_(t)
    eng.stage_rows(toks, chain(len(toks)), st)
    got = eng.rows_step(st, len(toks))
    assert torch.equal(got, want)
    # a tree: rows 0 -> {1 -> 2, 3}; row 3's path is (7, 19, 44)
    eng.stage_rows([7, 19, 23, 44], [-1, 0, 1, 1], st)
    tree = eng.rows_step(st, 4).clone()
    assert torch.equal(tree[:3], want[:3])
    s2 = eng.new_state(128)
    eng.prefill(PROMPT, s2)
    assert torch.equal(_decode(eng, s2, [7, 19, 44])[2], tree[3])


def test_commit_a_moved_path_and_a_rollback():
    if SKIP:
        return SKIP
    cfg, eng, ref = tiny()
    st = eng.new_state(128)
    eng.prefill(PROMPT, st)
    alone = eng.new_state(128)
    eng.prefill(PROMPT, alone)
    eng.verify([7, 19, 23, 44], [-1, 0, 1, 1], st)
    eng.commit(st, [0, 1, 3])
    _decode(eng, alone, [7, 19, 44])
    assert st.length == alone.length
    for li in st.conv:
        assert torch.equal(st.conv[li], alone.conv[li])
    assert torch.equal(_decode(eng, st, [5]), _decode(eng, alone, [5]))
    eng.verify([9, 10, 11, 12], chain(4), st)
    eng.commit(st, [0, 1])
    _decode(eng, alone, [9, 10])
    assert torch.equal(_decode(eng, st, [60]), _decode(eng, alone, [60]))


def test_a_joint_pass_is_each_states_lone_verify():
    """ENG-241 for LFM2: two states' blocks in one pass give each state the bits of its lone verify, and
    each commits its own block."""
    if SKIP:
        return SKIP
    cfg, eng, ref = tiny()
    a, b = eng.new_state(128), eng.new_state(128)
    eng.prefill(PROMPT, a)
    eng.prefill(PROMPT[:9], b)
    blocks = [([7, 19, 23, 44], [-1, 0, 1, 1]), ([5, 6, 7], [-1, 0, 1])]
    lone = []
    for (toks, par), st in zip(blocks, (a, b)):
        eng.verify(toks, par, st)
        lone.append(eng._rb[len(toks)]["logits"][:len(toks)].clone())
    rbs = eng.verify_multi([(toks, par, st) for (toks, par), st in zip(blocks, (a, b))])
    for rb, want, (toks, _) in zip(rbs, lone, blocks):
        assert torch.equal(rb["logits"][:len(toks)], want)
    eng.commit(a, [0, 1, 3])
    eng.commit(b, [0, 1])
    alone_a, alone_b = eng.new_state(128), eng.new_state(128)
    eng.prefill(PROMPT, alone_a)
    eng.prefill(PROMPT[:9], alone_b)
    _decode(eng, alone_a, [7, 19, 44])
    _decode(eng, alone_b, [5, 6])
    assert torch.equal(_decode(eng, a, [3]), _decode(eng, alone_a, [3]))
    assert torch.equal(_decode(eng, b, [3]), _decode(eng, alone_b, [3]))
    return "7 rows of 2 states: each its lone bits; each commit its own block"


def test_a_joint_decode_step_is_each_states_own():
    """Several states' decode tokens in one pass (`decode_multi`, eager here): each row is that state's
    lone decode step bit for bit, and each state advances as its own step would move it."""
    if SKIP:
        return SKIP
    cfg, eng, ref = tiny()
    states, alone = [eng.new_state(128) for _ in range(3)], [eng.new_state(128) for _ in range(3)]
    for st, al, n in zip(states, alone, (14, 9, 5)):
        eng.prefill(PROMPT[:n], st)
        eng.prefill(PROMPT[:n], al)
    for tok in (7, 19):
        rbs = eng.decode_multi([(tok, st, f"s{j}") for j, st in enumerate(states)])
        for j, (st, al, rb) in enumerate(zip(states, alone, rbs)):
            want = _decode(eng, al, [tok])
            assert torch.equal(rb["logits"][:1], want), j
            st.length += 1
    for st, al in zip(states, alone):
        assert st.length == al.length and all(torch.equal(st.conv[i], al.conv[i]) for i in st.conv)
    return "3 states, 2 steps: each its own decode step's bits and state"


def test_self_check_passes():
    if SKIP:
        return SKIP
    cfg, eng, ref = tiny()
    res = eng.self_check(PROMPT.tolist(), steps=6, log=None)
    assert res["ok"], res


if __name__ == "__main__":
    fails = 0
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        try:
            print(f"  {name:<44} ok   {fn() or ''}")
        except AssertionError as e:
            fails += 1
            print(f"  {name:<44} FAIL {e}")
    print(f"{len(tests) - fails} passed" + (f", {fails} FAILED" if fails else ""))
    sys.exit(1 if fails else 0)
