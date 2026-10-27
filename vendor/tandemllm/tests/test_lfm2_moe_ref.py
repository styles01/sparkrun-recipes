"""ENG-283: the LFM2 MoE reference forward against the official implementation, and its block algebra.

A tiny random LFM2 MoE (6 layers: conv, conv, attention, conv, conv, attention; the first 2 dense,
8 experts top 2 with a random expert bias) is built with `transformers`' `Lfm2MoeForCausalLM` in fp32;
the engine's reference (`engine/models/lfm2_moe/ref.py`) runs on the same weights. Where transformers
has no `lfm2_moe` the comparison tests are skipped and say so. The block tests (a verify tree equals
the sequential decode of each row's path, a partly kept chain continues exactly) need no transformers.
"""

from __future__ import annotations

import copy
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.models import architecture  # noqa: E402
from engine.models.lfm2_moe.config import config_from_dict  # noqa: E402
from engine.models.lfm2_moe.ref import Lfm2MoeRef, chain  # noqa: E402

TINY = dict(
    model_type="lfm2_moe", architectures=["Lfm2MoeForCausalLM"], vocab_size=96, hidden_size=32,
    intermediate_size=48, moe_intermediate_size=16, num_hidden_layers=6, num_attention_heads=4,
    num_key_value_heads=2, head_dim=8, norm_eps=1e-5, max_position_embeddings=256, conv_bias=False,
    conv_L_cache=3, num_dense_layers=2, num_experts=8, num_experts_per_tok=2, use_expert_bias=True,
    routed_scaling_factor=1.0, norm_topk_prob=True, tie_word_embeddings=True,
    layer_types=["conv", "conv", "full_attention", "conv", "conv", "full_attention"],
    rope_parameters={"rope_type": "default", "rope_theta": 1e6}, eos_token_id=2, bos_token_id=1, pad_token_id=0,
)
# the fp32 reference equals transformers within this bound (summation-order noise over 6 layers)
BOUND = 1e-4
PROMPT = torch.tensor([5, 9, 3, 17, 22, 3, 40, 41, 12, 7, 63, 2, 30, 88])


def _random_weights(cfg, seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    H, I, E, MI = cfg.hidden_size, cfg.intermediate_size, cfg.num_experts, cfg.moe_intermediate_size

    def r(*shape, s=0.2):
        return torch.randn(*shape, generator=g) * s
    W = {"embed_tokens.weight": r(cfg.vocab_size, H, s=1.0), "embedding_norm.weight": 1 + r(H, s=0.1)}
    for li in range(cfg.num_hidden_layers):
        p = f"layers.{li}."
        W[p + "operator_norm.weight"] = 1 + r(H, s=0.1)
        W[p + "ffn_norm.weight"] = 1 + r(H, s=0.1)
        if cfg.is_attention(li):
            a = p + "self_attn."
            W[a + "q_proj.weight"] = r(cfg.q_dim, H)
            W[a + "k_proj.weight"] = r(cfg.kv_dim, H)
            W[a + "v_proj.weight"] = r(cfg.kv_dim, H)
            W[a + "out_proj.weight"] = r(H, cfg.q_dim)
            W[a + "q_layernorm.weight"] = 1 + r(cfg.head_dim, s=0.1)
            W[a + "k_layernorm.weight"] = 1 + r(cfg.head_dim, s=0.1)
        else:
            W[p + "conv.in_proj.weight"] = r(3 * H, H)
            W[p + "conv.conv.weight"] = r(H, 1, cfg.conv_L_cache, s=0.5)
            W[p + "conv.out_proj.weight"] = r(H, H)
        f = p + "feed_forward."
        if cfg.is_moe(li):
            W[f + "gate.weight"] = r(E, H, s=0.5)
            W[f + "expert_bias"] = r(E, s=0.3)
            for e in range(E):
                W[f + f"experts.{e}.w1.weight"] = r(MI, H)
                W[f + f"experts.{e}.w3.weight"] = r(MI, H)
                W[f + f"experts.{e}.w2.weight"] = r(H, MI)
        else:
            W[f + "w1.weight"] = r(I, H)
            W[f + "w3.weight"] = r(I, H)
            W[f + "w2.weight"] = r(H, I)
    return W


def _official(cfg, W):
    try:
        from transformers.models.lfm2_moe import modeling_lfm2_moe as m
        from transformers.models.lfm2_moe.configuration_lfm2_moe import Lfm2MoeConfig
    except Exception:
        return None
    tc = Lfm2MoeConfig(**{k: v for k, v in TINY.items() if k not in ("model_type", "architectures")})
    tc._attn_implementation = "eager"
    model = m.Lfm2MoeForCausalLM(tc).eval()
    sd = {}
    for name, t in W.items():
        if ".experts." in name:
            continue
        sd["model." + name] = t
    for li in cfg.moe_layers:
        f = f"layers.{li}.feed_forward.experts."
        sd[f"model.{f}gate_up_proj"] = torch.stack([torch.cat([W[f + f"{e}.w1.weight"], W[f + f"{e}.w3.weight"]])
                                                    for e in range(cfg.num_experts)])
        sd[f"model.{f}down_proj"] = torch.stack([W[f + f"{e}.w2.weight"] for e in range(cfg.num_experts)])
    sd["lm_head.weight"] = W["embed_tokens.weight"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not unexpected, unexpected
    assert not [k for k in missing if k != "lm_head.weight"], missing
    return model


_TINY = []


def tiny():
    if not _TINY:
        cfg = config_from_dict(TINY, "tiny")
        W = _random_weights(cfg)
        _TINY.append((cfg, W, Lfm2MoeRef(cfg, W)))
    return _TINY[0]


def _raises(fn, *needles):
    try:
        fn()
    except ValueError as e:
        for n in needles:
            assert n in str(e), (n, str(e))
        return
    raise AssertionError(f"no ValueError (wanted {needles})")


def test_architecture_selects_lfm2_moe():
    assert architecture(TINY).name == "lfm2_moe"
    assert architecture({"architectures": ["Qwen4ExpForCausalLM"]}).name == "qwen4_exp"


def test_unknown_field_stops_the_load():
    _raises(lambda: config_from_dict({**TINY, "shiny_new_switch": 1}), "does not know: shiny_new_switch")
    _raises(lambda: config_from_dict({**TINY, "conv_bias": True}), "conv_bias")


def test_lfm25_24b_config():
    """The real model's config.json (revision 1769516a) reads, with its derived sizes."""
    raw = json.load(open(os.path.join(os.path.dirname(__file__), "data", "lfm2.5-24b-a2b.config.json")))
    c = config_from_dict(raw, "x", {"eos_token_id": [124900]})
    assert c.attention_layers == list(range(2, 40, 4))
    assert len(c.conv_layers) == 30 and c.moe_layers == list(range(2, 40))
    assert (c.q_dim, c.kv_dim, c.gqa) == (2048, 512, 4)
    assert c.kv_bytes_per_token() == 20_480
    assert c.conv_state_bytes() == 30 * 2 * 2048 * 2
    assert c.rope_theta == 1e7 and c.eos_token_ids == [124900]


def test_prefill_and_decode_match_transformers():
    cfg, W, ref = tiny()
    model = _official(cfg, W)
    if model is None:
        return "skipped: transformers has no lfm2_moe here"
    with torch.no_grad():
        want = model(PROMPT[None]).logits[0]
    st = ref.new_state(64)
    got = ref.prefill(PROMPT, st)
    worst = (got - want).abs().max().item()
    assert worst < BOUND, worst
    # decode three tokens one at a time against transformers' cache
    from transformers import DynamicCache
    cache = DynamicCache(config=model.config)
    with torch.no_grad():
        out = model(PROMPT[None], past_key_values=cache, use_cache=True)
    for tok in (11, 50, 3):
        with torch.no_grad():
            out = model(torch.tensor([[tok]]), past_key_values=out.past_key_values, use_cache=True)
        got = ref.decode(tok, st)
        d = (got[0] - out.logits[0, -1]).abs().max().item()
        assert d < BOUND, d
        worst = max(worst, d)
    return f"max |ref - transformers| {worst:.1e} (prefill + 3 decode steps)"


def test_verify_tree_equals_sequential_decode():
    cfg, W, ref = tiny()
    st = ref.new_state(64)
    ref.prefill(PROMPT, st)
    tokens = torch.tensor([7, 19, 23, 44, 31, 60])
    parents = [-1, 0, 1, 1, 0, 4]                # 0 -> {1 -> {2, 3}, 4 -> 5}
    snap = copy.deepcopy(st)
    lg = ref.rows(tokens, parents, st)
    assert st.length == snap.length and all(torch.equal(st.conv[i], snap.conv[i]) for i in st.conv)
    paths = {0: [0], 1: [0, 1], 2: [0, 1, 2], 3: [0, 1, 3], 4: [0, 4], 5: [0, 4, 5]}
    for row, path in paths.items():
        s2 = copy.deepcopy(snap)
        for r in path:
            want = ref.decode(int(tokens[r]), s2)
        assert (lg[row] - want[0]).abs().max().item() < 1e-5, row


def test_commit_of_a_path_continues_exactly():
    cfg, W, ref = tiny()
    st = ref.new_state(64)
    ref.prefill(PROMPT, st)
    alone = copy.deepcopy(st)
    tokens = torch.tensor([7, 19, 23, 44, 31])
    ref.rows(tokens, [-1, 0, 1, 1, 0], st)
    ref.commit(st, [0, 1, 3])                   # a path that leaves the chain: its KV rows move
    for t in (7, 19, 44):
        ref.decode(t, alone)
    assert st.length == alone.length
    for li in st.conv:
        assert torch.allclose(st.conv[li], alone.conv[li], atol=1e-6)
    a, b = ref.decode(5, st), ref.decode(5, alone)
    assert (a - b).abs().max().item() < 1e-5


def test_chain_kept_in_part():
    """A chain block of 5 rows of which 2 are kept (the rollback of a partly accepted draft)."""
    cfg, W, ref = tiny()
    st = ref.new_state(64)
    ref.prefill(PROMPT, st)
    alone = copy.deepcopy(st)
    ref.rows(torch.tensor([7, 19, 23, 44, 31]), chain(5), st)
    ref.commit(st, [0, 1])
    ref.decode(7, alone)
    ref.decode(19, alone)
    a, b = ref.decode(60, st), ref.decode(60, alone)
    assert (a - b).abs().max().item() < 1e-5


def test_commit_refuses_a_non_path():
    cfg, W, ref = tiny()
    st = ref.new_state(64)
    ref.prefill(PROMPT, st)
    ref.rows(torch.tensor([7, 19, 23]), [-1, 0, 0], st)
    _raises(lambda: ref.commit(st, [0, 1, 2]), "not a root path")


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
