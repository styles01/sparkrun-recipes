"""A BF16 checkpoint loads through the same loader and runs through the Linear interface.

The FP8 checkpoint stores every projection as e4m3 codes + scales (`FP8Block`); the BF16 checkpoint
of the same model stores the same names as plain bf16 matrices. The loader now wraps those as
`BF16Block`s, so `Weights.proj()` answers for them and `model.linear` multiplies through
`BF16Block.matmul` (`F.linear`). Checked on a tiny random checkpoint written to disk the way the BF16
release lays it out: the projections become BF16Blocks, everything else stays a plain tensor, and a
forward over the loaded weights is bit-identical to the same forward over the plain tensors.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

for _k in ("NORM", "GDN", "HEAD", "ATTN", "GDNBLOCK", "GDNTREE"):
    os.environ.setdefault(f"QWEN38_FUSED_{_k}", "0")
os.environ.setdefault("QWEN38_TREE_CHAIN_DELEGATE", "0")

import torch
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from engine.linear import BF16Block  # noqa: E402
from engine.loader import LM_PREFIX, Weights  # noqa: E402
from engine.model import Qwen38Engine  # noqa: E402
from test_refactor_gate import TinyWeights, config  # noqa: E402


def _checkpoint(cfg, plain) -> str:
    d = tempfile.mkdtemp()
    text = {"hidden_size": cfg.hidden_size, "intermediate_size": cfg.intermediate_size,
            "num_hidden_layers": cfg.num_hidden_layers, "num_attention_heads": cfg.num_attention_heads,
            "num_key_value_heads": cfg.num_key_value_heads, "head_dim": cfg.head_dim,
            "vocab_size": cfg.vocab_size, "rms_norm_eps": cfg.rms_norm_eps,
            "max_position_embeddings": cfg.max_position_embeddings, "layer_types": cfg.layer_types,
            "linear_conv_kernel_dim": cfg.linear_conv_kernel_dim, "linear_key_head_dim": cfg.linear_key_head_dim,
            "linear_value_head_dim": cfg.linear_value_head_dim, "linear_num_key_heads": cfg.linear_num_key_heads,
            "linear_num_value_heads": cfg.linear_num_value_heads,
            "rope_parameters": {"rope_theta": cfg.rope_theta, "partial_rotary_factor": cfg.partial_rotary_factor,
                                "mrope_section": cfg.mrope_section}}
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump({"text_config": text, "tie_word_embeddings": False}, f)
    out = {}
    for name, t in plain.items():
        key = name if name == "lm_head.weight" else LM_PREFIX + name
        if not name.endswith(".weight") and t.dim() == 2:     # FakeWeights keeps projections without ".weight"
            key += ".weight"
        out[key] = t.contiguous()
    out["model.visual.patch_embed.proj.weight"] = torch.zeros(4, 4, dtype=torch.bfloat16)
    save_file(out, os.path.join(d, "model-00001-of-00001.safetensors"))
    return d


def test_bf16_checkpoint_loads_and_runs():
    cfg = config("fp8")                         # dims that are multiples of 128, as the real model's
    tw = TinyWeights(cfg, "bf16", seed=3)       # fp32 tensors; the checkpoint is bf16
    plain = {k: v.to(torch.bfloat16) for k, v in tw.t.items()}
    w = Weights(_checkpoint(cfg, plain), device="cpu", skip_mtp=True, nvfp4="", fp8_head="")
    blocks = [k for k, v in w.q.items() if isinstance(v, BF16Block)]
    n_proj = sum(1 for k in plain if not k.endswith(".weight") and plain[k].dim() == 2)
    assert len(blocks) == n_proj and len(w.q) == n_proj, (len(blocks), n_proj, len(w.q))
    assert "lm_head.weight" in w.t and "embed_tokens.weight" in w.t and not any("visual" in k for k in w.t)

    class Plain:                                # the same bf16 tensors as plain tensors
        t = plain
        def norm(self, n): return self.t[n]
        def proj(self, n): return self.t[n]
        def group(self, n): return None

    ids = torch.arange(1, 13)
    outs = []
    for weights in (w, Plain()):
        eng = Qwen38Engine(cfg, weights, max_len=64, device="cpu")
        with torch.no_grad():
            outs.append(eng.forward(ids, start=0, last_only=True))
    assert torch.isfinite(outs[0].float()).all()
    assert torch.equal(outs[0].view(torch.int16), outs[1].view(torch.int16)), "BF16Block forward != plain forward"
    return f"{len(blocks)} projections as BF16Block; logits bit-identical to the plain-tensor forward"


if __name__ == "__main__":
    fails = 0
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        try:
            print(f"  {name:<40} ok   {fn() or ''}")
        except AssertionError as e:
            fails += 1
            print(f"  {name:<40} FAIL {e}")
    print(f"{len(tests) - fails} passed" + (f", {fails} FAILED" if fails else ""))
    sys.exit(1 if fails else 0)
