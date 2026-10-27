"""The NemotronH reference forward against the official implementation, and its block algebra.

Tiny random NemotronH models (Mamba2, MoE and NoPE attention blocks; a Lightning-like one without the
latent pair, a Super-like one with it, and a Puzzle/Bolt-like one with per-layer top k, expert widths
and expert counts) are built with `transformers`' `NemotronHForCausalLM` in fp32 (its torch path, one SSD
chunk: transformers' multi-chunk torch path disagrees with its own one-chunk result by 3e-2 on these
weights, so the chunked form is held to the row path in tests/test_nemotron_h_engine.py instead; the
per-layer MoE blocks as the Puzzle's remote code builds them,
one config a block); the engine's reference (`engine/models/nemotron_h/ref.py`) runs on the same
weights. Where transformers has no `nemotron_h` the comparison tests are skipped and say so. The block
tests (a verify tree equals the sequential decode of each row's path, a partly kept tree continues
exactly) need no transformers.
"""

from __future__ import annotations

import copy
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.models.nemotron_h.config import config_from_dict  # noqa: E402
from engine.models.nemotron_h.ref import NemotronHRef, chain  # noqa: E402

TINY = dict(
    model_type="nemotron_h", architectures=["NemotronHForCausalLM"], vocab_size=96, hidden_size=32,
    num_attention_heads=4, num_key_value_heads=2, head_dim=8, layer_norm_epsilon=1e-5,
    max_position_embeddings=256, mamba_num_heads=8, mamba_head_dim=4, ssm_state_size=6, n_groups=2,
    conv_kernel=4, chunk_size=128, n_routed_experts=8, num_experts_per_tok=2, moe_intermediate_size=12,
    moe_shared_expert_intermediate_size=16, moe_latent_size=None, routed_scaling_factor=2.5,
    layers_block_type=["mamba", "moe", "attention", "mamba", "moe", "mamba", "attention", "moe"],
    num_nextn_predict_layers=0, tie_word_embeddings=False, eos_token_id=2, bos_token_id=1, pad_token_id=0,
    mlp_hidden_act="relu2", mamba_hidden_act="silu", use_conv_bias=True, mamba_proj_bias=False, use_bias=False,
    n_group=1, topk_group=1, norm_topk_prob=True, residual_in_fp32=False, attention_bias=False, mlp_bias=False,
)
LATENT = {**TINY, "moe_latent_size": 16}
# Puzzle / Bolt: no global k or width; each MoE block its own k, width and (Bolt) expert count
PUZZLE = {k: v for k, v in LATENT.items() if k not in ("layers_block_type", "num_experts_per_tok",
                                                      "moe_intermediate_size")}
PUZZLE.update(model_type="nemotron_h_puzzle", architectures=["NemotronHPuzzleForCausalLM"], block_configs=[
    {"block_type": "mamba"}, {"block_type": "moe", "num_experts_per_tok": 2, "moe_intermediate_size": 12},
    {"block_type": "attention"}, {"block_type": "mamba"},
    {"block_type": "moe", "num_experts_per_tok": 3, "moe_intermediate_size": 20, "n_routed_experts": 6},
    {"block_type": "mamba"}, {"block_type": "attention"},
    {"block_type": "moe", "num_experts_per_tok": 1, "moe_intermediate_size": 8, "n_routed_experts": 4}])
for b in PUZZLE["block_configs"]:
    if b["block_type"] == "moe":
        b.setdefault("n_routed_experts", 8)
# fp32 summation-order noise over 8 layers (measured up to 5.3e-6 on the Puzzle-like model)
BOUND = 2e-5
PROMPT = torch.tensor([5, 9, 3, 17, 22, 3, 40, 41, 12, 7, 63, 2, 30, 88])


def random_weights(cfg, seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    H = cfg.hidden_size

    def r(*shape, s=0.2):
        return torch.randn(*shape, generator=g) * s
    W = {"embeddings.weight": r(cfg.vocab_size, H, s=1.0), "norm_f.weight": 1 + r(H, s=0.1),
         "lm_head.weight": r(cfg.vocab_size, H)}
    for li, t in enumerate(cfg.layer_types):
        p = f"layers.{li}."
        W[p + "norm.weight"] = 1 + r(H, s=0.1)
        m = p + "mixer."
        if t == "mamba":
            W[m + "in_proj.weight"] = r(cfg.in_proj_dim, H)
            W[m + "conv1d.weight"] = r(cfg.conv_dim, 1, cfg.conv_kernel, s=0.4)
            W[m + "conv1d.bias"] = r(cfg.conv_dim, s=0.1)
            W[m + "dt_bias"] = 0.5 + torch.rand(cfg.mamba_num_heads, generator=g)
            W[m + "A_log"] = torch.log(torch.arange(1, cfg.mamba_num_heads + 1, dtype=torch.float32)) + r(
                cfg.mamba_num_heads, s=0.1)
            W[m + "D"] = 1 + r(cfg.mamba_num_heads, s=0.1)
            W[m + "norm.weight"] = 1 + r(cfg.intermediate, s=0.1)
            W[m + "out_proj.weight"] = r(H, cfg.intermediate)
        elif t == "attention":
            W[m + "q_proj.weight"] = r(cfg.q_dim, H)
            W[m + "k_proj.weight"] = r(cfg.kv_dim, H)
            W[m + "v_proj.weight"] = r(cfg.kv_dim, H)
            W[m + "o_proj.weight"] = r(H, cfg.q_dim)
        else:
            E, I, X = cfg.moe_experts[li], cfg.moe_ffn[li], cfg.expert_in
            W[m + "gate.weight"] = r(E, H, s=0.5)
            W[m + "gate.e_score_correction_bias"] = r(E, s=0.05)
            for e in range(E):
                W[m + f"experts.{e}.up_proj.weight"] = r(I, X)
                W[m + f"experts.{e}.down_proj.weight"] = r(X, I)
            W[m + "shared_experts.up_proj.weight"] = r(cfg.shared_ffn, H)
            W[m + "shared_experts.down_proj.weight"] = r(H, cfg.shared_ffn)
            if cfg.moe_latent_size:
                W[m + "fc1_latent_proj.weight"] = r(cfg.moe_latent_size, H)
                W[m + "fc2_latent_proj.weight"] = r(H, cfg.moe_latent_size)
    return W


def hf_model(raw: dict, cfg, W: dict):
    """transformers' NemotronHForCausalLM on W (fp32, eager, the torch Mamba path), or None."""
    try:
        from transformers.models.nemotron_h import configuration_nemotron_h as hc
        from transformers.models.nemotron_h import modeling_nemotron_h as hm
    except Exception:  # noqa: BLE001
        return None
    base = {k: v for k, v in raw.items() if k not in ("block_configs", "architectures", "model_type")}
    base.update(layers_block_type=list(cfg.layer_types), use_mamba_kernels=False,
                num_experts_per_tok=base.get("num_experts_per_tok", 2), moe_intermediate_size=base.get(
                    "moe_intermediate_size", 8))
    hcfg = hc.NemotronHConfig(**base)
    hcfg._attn_implementation = "eager"
    torch.manual_seed(0)
    model = hm.NemotronHForCausalLM(hcfg).float().eval()
    for li in cfg.moe_layers:
        # the Puzzle's remote code builds each block from its own config: the same here for the MoE mixer
        lc = copy.deepcopy(hcfg)
        lc.num_experts_per_tok, lc.moe_intermediate_size, lc.n_routed_experts = (
            cfg.moe_k[li], cfg.moe_ffn[li], cfg.moe_experts[li])
        model.model.layers[li].mixer = hm.NemotronHMoE(lc, layer_idx=li).float()
    sd = {}
    for k, v in W.items():
        if ".experts." in k:
            continue
        sd[("model." + k) if not k.startswith("lm_head") else k] = v
    for li in cfg.moe_layers:
        m = f"layers.{li}.mixer."
        E = cfg.moe_experts[li]
        sd[f"model.{m}experts.up_proj"] = torch.stack([W[m + f"experts.{e}.up_proj.weight"] for e in range(E)])
        sd[f"model.{m}experts.down_proj"] = torch.stack([W[m + f"experts.{e}.down_proj.weight"] for e in range(E)])
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [k for k in missing if not k.endswith("inv_freq")]
    assert not missing and not unexpected, (missing, unexpected)
    return model


def _models(raw: dict, seed: int = 0):
    cfg = config_from_dict(raw, "tiny")
    W = random_weights(cfg, seed)
    return cfg, W, NemotronHRef(cfg, W)


def _vs_hf(raw: dict, name: str) -> str | None:
    cfg, W, ref = _models(raw)
    hf = hf_model(raw, cfg, W)
    if hf is None:
        return "skipped: transformers has no nemotron_h"
    with torch.no_grad():
        want = hf(PROMPT[None]).logits[0].float()
    st = ref.new_state(64)
    got = ref.prefill(PROMPT, st)
    err = (got - want).abs().max().item()
    assert err < BOUND, f"{name}: prefill vs transformers {err}"
    # the cached step: decode the next tokens one at a time, each against transformers' full forward
    toks = PROMPT.tolist()
    for _ in range(4):
        nxt = int(got[-1].argmax())
        toks.append(nxt)
        got = ref.decode(nxt, st)
        with torch.no_grad():
            w2 = hf(torch.tensor([toks])).logits[0, -1].float()
        err = (got[0] - w2).abs().max().item()
        assert err < BOUND, f"{name}: decode step at {len(toks)} vs transformers {err}"
    return None


def test_lightning_like_vs_transformers():
    return _vs_hf(TINY, "lightning-like")


def test_latent_vs_transformers():
    return _vs_hf(LATENT, "latent")


def test_puzzle_per_layer_vs_transformers():
    return _vs_hf(PUZZLE, "puzzle")


def test_tree_rows_are_their_paths_decoded():
    for raw in (TINY, PUZZLE):
        cfg, W, ref = _models(raw, 1)
        st = ref.new_state(64)
        ref.prefill(PROMPT, st)
        L = st.length
        snap = (dict((k, v.clone()) for k, v in st.conv.items()), dict((k, v.clone()) for k, v in st.ssm.items()))
        tokens = torch.tensor([7, 11, 13, 17, 19, 23, 29])
        parents = [-1, 0, 1, 0, 3, 4, 1]
        got = ref.rows(tokens, parents, st)
        paths = [[0], [0, 1], [0, 1, 2], [0, 3], [0, 3, 4], [0, 3, 4, 5], [0, 1, 6]]
        for i, pa in enumerate(paths):
            s2 = ref.new_state(64)
            for d in (s2.conv, s2.ssm):
                pass
            for li in st.conv:
                s2.conv[li].copy_(snap[0][li])
                s2.ssm[li].copy_(snap[1][li])
            for li in st.k:
                s2.k[li][:, :L] = st.k[li][:, :L]
                s2.v[li][:, :L] = st.v[li][:, :L]
            s2.length = L
            lg = ref.rows(tokens[pa], chain(len(pa)), s2)
            err = (lg[-1] - got[i]).abs().max().item()
            assert err < 1e-5, (raw["model_type"], i, err)


def test_commit_a_moved_path_continues_exactly():
    cfg, W, ref = _models(TINY, 2)
    st = ref.new_state(64)
    ref.prefill(PROMPT, st)
    tokens = torch.tensor([7, 11, 13, 17, 19])
    parents = [-1, 0, 0, 2, 3]
    ref.rows(tokens, parents, st)
    ref.commit(st, [0, 2, 3])
    a = ref.decode(31, st)
    st2 = ref.new_state(64)
    ref.prefill(torch.cat([PROMPT, torch.tensor([7, 13, 17, 31])]), st2)
    b = ref.rows(torch.tensor([5]), [-1], st2)
    st3 = ref.new_state(64)
    c = ref.prefill(torch.cat([PROMPT, torch.tensor([7, 13, 17, 31])]), st3)
    err = (a[0] - c[-1]).abs().max().item()
    assert err < 1e-5, err
    assert b.shape[0] == 1


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            r = fn()
            print(f"{r or 'ok'} {name}")
    print("all nemotron_h ref tests passed")
