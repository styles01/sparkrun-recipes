"""ENG-383: the Qwen3.5 MoE reference forward (engine/models/qwen3_5_moe/ref.py) against transformers'
Qwen3_5MoeForCausalLM on a tiny random config, fp32 on the CPU: logits of a prompt in one pass, then the
same prompt as a prefill and token-by-token decode steps (the recurrent and KV state carried).

    python tools/q35m_refcheck.py [--shape ornith|minicpm]   (the shape's head and expert ratios, tiny widths)
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.models.flashnext.ref import ExpertBank, FNState  # noqa: E402
from engine.models.qwen3_5_moe.config import config_from_dict  # noqa: E402
from engine.models.qwen3_5_moe.ref import Qwen35MoeRef  # noqa: E402

SHAPES = {
    # (heads, kv heads, linear key heads, linear value heads, experts, top-k)
    "ornith": (8, 2, 2, 8, 16, 4),
    "minicpm": (4, 2, 2, 4, 16, 4),
}


def tiny_config(shape: str) -> dict:
    hq, hk, lk, lv, E, k = SHAPES[shape]
    return dict(model_type="qwen3_5_moe_text", vocab_size=97, hidden_size=64, num_hidden_layers=4,
                num_attention_heads=hq, num_key_value_heads=hk, head_dim=32, hidden_act="silu",
                max_position_embeddings=512, rms_norm_eps=1e-6,
                layer_types=["linear_attention"] * 3 + ["full_attention"], linear_conv_kernel_dim=4,
                linear_key_head_dim=16, linear_value_head_dim=16, linear_num_key_heads=lk,
                linear_num_value_heads=lv, num_experts=E, num_experts_per_tok=k, moe_intermediate_size=24,
                shared_expert_intermediate_size=32, eos_token_id=3, tie_word_embeddings=False,
                mtp_num_hidden_layers=0, attn_output_gate=True,
                rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25,
                                 "mrope_section": [2, 1, 1], "mrope_interleaved": True})


def weights_of(model) -> dict:
    """transformers' state dict under the reference's canonical names; the fused expert banks as
    per-expert dense banks (gate rows first, as the converter splits them)."""
    W = {}
    sd = {k: v.detach() for k, v in model.state_dict().items()}
    for k, v in sd.items():
        name = k[len("model."):] if k.startswith("model.") else k
        if ".mlp.experts.gate_up_proj" in name:
            p = name[: -len("gate_up_proj")]
            I = v.shape[1] // 2
            W[p + "gate_proj"] = ExpertBank(dense=v[:, :I].contiguous())
            W[p + "up_proj"] = ExpertBank(dense=v[:, I:].contiguous())
        elif ".mlp.experts.down_proj" in name:
            W[name] = ExpertBank(dense=v)
        else:
            W[name] = v
    return W


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="ornith", choices=sorted(SHAPES))
    ap.add_argument("--tokens", type=int, default=40)
    a = ap.parse_args()
    from transformers import Qwen3_5MoeForCausalLM, Qwen3_5MoeTextConfig
    torch.manual_seed(0)
    raw = tiny_config(a.shape)
    tcfg = Qwen3_5MoeTextConfig(**{k: v for k, v in raw.items() if k != "model_type"})
    tcfg._attn_implementation = "eager"
    model = Qwen3_5MoeForCausalLM(tcfg).float().eval()
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "norm" in n and "linear_attn.norm" not in n:
                p.normal_(0, 0.1)                   # (1 + w) norms: weights near 0
            elif n.endswith("A_log"):
                p.uniform_(-1, 1)
            else:
                p.normal_(0, 0.15)
    cfg = config_from_dict(raw)
    ref = Qwen35MoeRef(cfg, weights_of(model), dtype=torch.float32, device="cpu")
    ids = torch.randint(5, raw["vocab_size"], (a.tokens,))
    with torch.no_grad():
        want = model(ids[None]).logits[0].float()
    got = ref.forward(ids, FNState())
    d1 = (got - want).abs().max().item()
    # prefill half, decode the rest one token at a time (the recurrent rule's single-row path)
    st = FNState()
    h = a.tokens // 2
    rows = [ref.forward(ids[:h], st)]
    for t in range(h, a.tokens):
        rows.append(ref.forward(ids[t:t + 1], st))
    d2 = (torch.cat(rows) - want).abs().max().item()
    scale = want.abs().max().item()
    top1 = (got.argmax(-1) == want.argmax(-1)).float().mean().item()
    ok = d1 < 1e-3 * max(1.0, scale) and d2 < 1e-3 * max(1.0, scale) and top1 == 1.0
    print(f"[q35m-refcheck] shape {a.shape}: one pass max|diff| {d1:.2e}, prefill+decode {d2:.2e} "
          f"(logit scale {scale:.2f}), top-1 {top1 * 100:.1f} % -> {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
