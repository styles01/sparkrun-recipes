"""ENG-135/136 (ported from branch `flashnext`): the Flash-Next reference forward against the official implementation.

A tiny random Flash-Next (3 GDN + 1 attention layers, 8 experts top-2, 4 hyper-connection
streams, PLE on layer 2, a QSA indexer with a budget of 8 tokens so a 14-token prompt reaches the
sparse path) is built with `transformers`' `Qwen4ExpForCausalLM` in fp32 and random weights; the
engine's reference (`engine/models/flashnext/ref.py`) runs on the same weights. The official code is
`transformers` >= 5.17 (`models/qwen4_exp`); where it is not installed the comparison tests are
skipped and say so (the box's engine interpreter has an older release; FN_TRANSFORMERS may point
at a directory that holds a newer `transformers` package).
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if os.environ.get("FN_TRANSFORMERS"):
    sys.path.insert(0, os.environ["FN_TRANSFORMERS"])

from engine.models.flashnext.config import config_from_dict  # noqa: E402

from engine.models.flashnext.ple import DenseTable  # noqa: E402
from engine.models.flashnext.ref import ExpertBank, FlashNextRef, FNState  # noqa: E402

TINY = dict(
    model_type="qwen4_exp_text", vocab_size=64, hidden_size=32, num_hidden_layers=4,
    num_attention_heads=4, num_key_value_heads=2, head_dim=16, hidden_act="silu",
    max_position_embeddings=256, rms_norm_eps=1e-6,
    layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
    linear_conv_kernel_dim=4, linear_key_head_dim=8, linear_value_head_dim=8,
    linear_num_key_heads=2, linear_num_value_heads=4, output_gate_type="sigmoid",
    num_experts=8, num_experts_per_tok=2, moe_intermediate_size=16,
    shared_expert_intermediate_size=16, intermediate_size=64,
    hc_count=4, hc_lowrank=8, ple_layer_ids=[2], ple_embed_dim=32, ple_conv_kernel_size=4,
    ngram_size=3, heads_per_ngram=8, ngram_vocab_size_base=101,
    make_ngram_vocab_size_divisible_by=16, split_ngram_parts=1,
    indexer_n_heads=2, indexer_kv_heads=1, indexer_head_dim=8, indexer_budget=8,
    indexer_compress_ratio=2, eos_token_id=3, bos_token_id=3, pad_token_id=None,
    tie_word_embeddings=False,
    rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25,
                     "mrope_section": [1, 1, 0], "mrope_interleaved": True},
)
# ENG-135: the fp32 reference equals transformers within this bound (fp32 summation-order noise
# over 4 layers of random weights; measured 3-4e-6)
BOUND = 5e-6
PROMPT = torch.tensor([5, 9, 3, 17, 22, 3, 3, 40, 41, 12, 7, 63, 2, 30])


def _official():
    try:
        from transformers.models.qwen4_exp import modeling_qwen4_exp as m  # noqa: F401
        from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
    except Exception:
        return None
    torch.manual_seed(0)
    cfg = Qwen4ExpTextConfig(**{k: v for k, v in TINY.items() if k != "model_type"})
    cfg._attn_implementation = "eager"
    model = m.Qwen4ExpForCausalLM(cfg).float().eval()
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n.endswith("A_log"):
                p.copy_(torch.rand(p.shape, generator=g) * 2 - 1)
            elif n.endswith("dt_bias"):
                p.copy_(torch.rand(p.shape, generator=g))
            else:
                p.copy_(torch.randn(p.shape, generator=g) * (0.3 if p.dim() == 1 else 0.2))
    return model


def weights_from_official(model) -> tuple[dict, dict]:
    sd = model.state_dict()
    W, tables = {}, {}
    for k, v in sd.items():
        name = k[len("model."):] if k.startswith("model.") else k
        if name.endswith("mlp.experts.gate_up_proj"):
            base = name[: -len("gate_up_proj")]
            inter = v.shape[1] // 2
            W[base + "gate_proj"] = ExpertBank(dense=v[:, :inter].clone())
            W[base + "up_proj"] = ExpertBank(dense=v[:, inter:].clone())
        elif name.endswith("mlp.experts.down_proj"):
            W[name] = ExpertBank(dense=v.clone())
        elif name.endswith("ngram_embedding.weight"):
            li = int(name.split(".")[1])
            tables[li] = DenseTable(v.clone())
        else:
            W[name] = v.clone()
    return W, tables


def _pair():
    model = _official()
    if model is None:
        return None, None
    W, tables = weights_from_official(model)
    cfg = config_from_dict(dict(TINY))
    return model, FlashNextRef(cfg, W, tables, dtype=torch.float32)


def test_logits_match_official():
    model, ref = _pair()
    if model is None:
        return "skipped: transformers without qwen4_exp"
    with torch.no_grad():
        want = model(PROMPT[None]).logits[0].float()
    got = ref.forward(PROMPT, FNState())
    err = (got - want).abs().max().item()
    assert err < BOUND, f"max abs err {err}"
    assert torch.equal(got.argmax(-1), want.argmax(-1))
    return f"14 tokens (sparse QSA past token 9), max abs err {err:.1e}"


def test_layer_by_layer_match_official():
    model, ref = _pair()
    if model is None:
        return "skipped: transformers without qwen4_exp"
    outs = []
    hooks = [layer.register_forward_hook(lambda m, i, o: outs.append(o[0] if isinstance(o, tuple) else o))
             for layer in model.model.layers]
    with torch.no_grad():
        model(PROMPT[None])
    for h in hooks:
        h.remove()
    trace = []
    ref.forward(PROMPT, FNState(), trace=trace)
    errs = [(a[0] - b).abs().max().item() for a, b in zip(outs, trace)]
    assert len(errs) == 4 and max(errs) < BOUND, errs
    return "per-layer max abs err " + " ".join(f"{e:.1e}" for e in errs)


def test_sparse_path_is_exercised():
    model, ref = _pair()
    if model is None:
        return "skipped: transformers without qwen4_exp"
    dense = FlashNextRef(config_from_dict(dict(TINY, indexer_budget=1 << 20)), ref.W, ref.tables,
                         dtype=torch.float32)
    a = ref.forward(PROMPT, FNState())
    b = dense.forward(PROMPT, FNState())
    early = (a[:9] - b[:9]).abs().max().item()
    late = (a[9:] - b[9:]).abs().max().item()
    assert early < 1e-6 and late > 1e-3, (early, late)
    return f"dense below the budget ({early:.0e}), selection changes later rows ({late:.1e})"


def test_qsa_mask_equals_per_query_selection():
    model, ref = _pair()
    if model is None:
        return "skipped: transformers without qwen4_exp"
    T = 41
    iq = torch.randn(T, 2, 8, generator=torch.Generator().manual_seed(3))
    raw = torch.randn(T, 8, generator=torch.Generator().manual_seed(4))
    m = ref.qsa_mask(iq, raw, 0, 3)
    cos, sin = ref.rope(torch.arange(T))
    for t in range(T):
        sel = ref.qsa_select(iq[t], raw, cos, sin, t, 3)
        want = torch.zeros(T, dtype=torch.bool)
        if sel is None:
            want[: t + 1] = True
        else:
            want[sel] = True
        assert torch.equal(m[t] & (torch.arange(T) <= t), want), f"query {t}"
    return f"{T} queries, budget 8 of up to {T} tokens"


def test_long_prompt_matches_official():
    """48 tokens (sparse QSA from token 9), prefilled in chunks and decoded one at a time. The
    official `torch.topk` leaves ties unspecified; relu makes exact-zero block scores common on
    random weights, so a query whose selection has a tie at the cutoff may pick other blocks. Those
    positions are found with the reference's own scores and are the only ones allowed to differ."""
    model, ref = _pair()
    if model is None:
        return "skipped: transformers without qwen4_exp"
    import engine.models.flashnext.ref as R
    ids = torch.randint(0, 64, (48,), generator=torch.Generator().manual_seed(5))
    ids[[10, 11, 30]] = 3
    with torch.no_grad():
        want = model(ids[None]).logits[0].float()
    ties = set()
    orig = R.topk_stable

    def spy(x, k):
        if x.dim() == 2 and x.shape[-1] > k:
            srt = torch.sort(x, -1, descending=True).values
            ties.update((((srt[:, k - 1] == srt[:, k]) & torch.isfinite(srt[:, k])).nonzero().flatten()).tolist())
        return orig(x, k)
    R.topk_stable = spy
    try:
        ref.forward(ids, FNState())
    finally:
        R.topk_stable = orig
    st = FNState(cap=64)
    got = torch.cat([ref.forward(ids[:20], st), ref.forward(ids[20:37], st)] +
                    [ref.forward(ids[t:t + 1], st) for t in range(37, 48)])
    keep = [t for t in range(48) if t not in ties]
    err = (got[keep] - want[keep]).abs().max().item()
    assert err < BOUND and torch.equal(got[keep].argmax(-1), want[keep].argmax(-1)), err
    return f"48 tokens in chunks 20+17+11x1, max abs err {err:.1e}; tie positions excluded: {sorted(ties)}"


def test_long_block_attention_in_query_blocks():
    """A 600-token block (attention computed in query blocks of 256) equals the same tokens in
    blocks of 200 (fp32)."""
    model, ref = _pair()
    if model is None:
        return "skipped: transformers without qwen4_exp"
    ids = torch.randint(0, 64, (600,), generator=torch.Generator().manual_seed(7))
    a = ref.forward(ids, FNState(cap=640))
    st = FNState(cap=640)
    b = torch.cat([ref.forward(ids[i:i + 200], st) for i in range(0, 600, 200)])
    err = (a - b).abs().max().item()
    assert err < 1e-3, err
    return f"max abs err {err:.1e}"


def test_incremental_equals_full():
    model, ref = _pair()
    if model is None:
        # no official model needed for this one: random weights of the same shapes would do, but
        # the official init is the easiest source of a complete weight set
        return "skipped: transformers without qwen4_exp"
    full = ref.forward(PROMPT, FNState())
    st = FNState()
    parts = [ref.forward(PROMPT[:5], st)]
    for t in range(5, len(PROMPT)):
        parts.append(ref.forward(PROMPT[t:t + 1], st))
    inc = torch.cat(parts)
    err = (inc - full).abs().max().item()
    assert err < 1e-4, err
    assert torch.equal(inc.argmax(-1), full.argmax(-1))
    return f"prefill 5 + 9 single steps vs one pass, max abs err {err:.1e}"


def test_bf16_argmax_matches_fp32():
    model, ref = _pair()
    if model is None:
        return "skipped: transformers without qwen4_exp"
    W, tables = weights_from_official(model)
    ref16 = FlashNextRef(ref.cfg, W, tables, dtype=torch.bfloat16)
    a = ref.forward(PROMPT, FNState())
    b = ref16.forward(PROMPT, FNState())
    same = a.argmax(-1) == b.argmax(-1)
    agree = same.float().mean().item()
    # a random tiny model has near ties everywhere; a position whose fp32 margin between its two
    # best tokens is clear of bf16 noise (8 bf16 steps of the row's largest logit) must agree
    top2 = torch.topk(a, 2, dim=-1).values
    clear = (top2[:, 0] - top2[:, 1]) > 8 * 2 ** -7 * a.abs().max(-1).values
    assert clear.any() and bool(same[clear].all()), (same.tolist(), clear.tolist())
    return f"bf16 vs fp32 argmax agreement {agree:.2f} overall, 1.00 on the {int(clear.sum())} clear positions"


if __name__ == "__main__":
    fails = 0
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        try:
            print(f"  {name:<36} ok   {fn() or ''}")
        except AssertionError as e:
            fails += 1
            print(f"  {name:<36} FAIL {e}")
    print(f"{len(tests) - fails} passed" + (f", {fails} FAILED" if fails else ""))
    sys.exit(1 if fails else 0)
