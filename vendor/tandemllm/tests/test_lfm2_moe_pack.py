"""ENG-285: LFM2 MoE weights from a BF16 checkpoint and from an EXL3 pack (synthetic tiny files, CPU).

A tiny model is written as a BF16 checkpoint (the checkpoint's own names, `model.` prefix, an index
over two shards) and as an EXL3 pack (random trellises of `tools/exl3_linear.random_block`, the mul1
codebook, the router and norms plain, a quantised head). The index reads both, the audit names what a
broken pack lacks, and the loaded weights run the reference forward: the BF16 checkpoint exactly as the
weights it was written from, the EXL3 pack as its own dequantised weights within bf16 noise.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from safetensors.torch import save_file  # noqa: E402

from engine.models.lfm2_moe.config import config_from_dict  # noqa: E402
from engine.models.lfm2_moe.load import load  # noqa: E402
from engine.models.lfm2_moe.pack import MUL1, PackIndex, projections, summary  # noqa: E402
from engine.models.lfm2_moe.ref import Lfm2MoeRef  # noqa: E402
from tests.test_lfm2_moe_ref import TINY, _random_weights, _raises  # noqa: E402

# EXL3 needs 128-wide in and out features
TINY_EXL3 = {**TINY, "vocab_size": 256, "hidden_size": 128, "intermediate_size": 256, "moe_intermediate_size": 128,
             "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 64, "num_hidden_layers": 4,
             "num_dense_layers": 1, "num_experts": 4, "num_experts_per_tok": 2,
             "layer_types": ["conv", "full_attention", "conv", "full_attention"]}
PROMPT = torch.tensor([5, 9, 3, 17, 22, 3, 40, 41, 12, 7])


_TMP = []


def _tmp() -> str:
    """A fresh directory for one test's files (removed at exit)."""
    d = tempfile.mkdtemp(prefix="lfm-pack-")
    _TMP.append(d)
    return d


def _write(d, tensors: dict, raw: dict, shards: int = 2, quant: dict | None = None):
    os.makedirs(d, exist_ok=True)
    names = sorted(tensors)
    wm = {}
    for s in range(shards):
        part = {k: tensors[k].contiguous() for k in names[s::shards]}
        fn = f"model-{s + 1:05d}-of-{shards:05d}.safetensors"
        save_file(part, os.path.join(d, fn))
        wm.update({k: fn for k in part})
    json.dump({"metadata": {}, "weight_map": wm}, open(os.path.join(d, "model.safetensors.index.json"), "w"))
    json.dump({**raw, **({"quantization_config": quant} if quant else {})}, open(os.path.join(d, "config.json"), "w"))


def _bf16_ckpt(d):
    cfg = config_from_dict(TINY)
    W = {k: v.to(torch.bfloat16) for k, v in _random_weights(cfg).items()}
    W = {k: (v.float() if k.endswith("expert_bias") else v) for k, v in W.items()}
    _write(d, {"model." + k: v for k, v in W.items()}, TINY)
    return cfg, W


def _exl3_pack(d, half: str | None = None, codebook: str = "mul1", halves: bool = False):
    from tools.exl3_linear import random_block
    cfg = config_from_dict(TINY_EXL3)
    plain = {k: v for k, v in _random_weights(cfg).items()
             if not any(k == b + ".weight" for b, _, _ in projections(cfg))}
    out = {"model." + k: (v.float() if k.endswith("expert_bias") else v.to(torch.float16)) for k, v in plain.items()}
    blocks = {}
    for i, (base, N, K) in enumerate(projections(cfg) + [("lm_head", cfg.vocab_size, cfg.hidden_size)]):
        # q, k and v at one rate (one fused launch); the rest mixed 2..4
        kb = 4 if base == "lm_head" else (3 if base.endswith(("q_proj", "k_proj", "v_proj")) else 2 + i % 3)
        if halves and base != "lm_head":
            # exllamav3 1.6.0's half-integer rates beside whole ones (1.5, 2.5 and 3.5 as 15, 25, 35)
            kb = 35 if base.endswith(("q_proj", "k_proj", "v_proj")) else (15, 25, 35, 2, 3, 4)[i % 6]
        b = random_block(N, K, kb, device="cpu", seed=i)
        blocks[base] = b
        key = base if base == "lm_head" else "model." + base
        t = b.t
        if base == half:
            t = torch.zeros(t.shape[0], t.shape[1], 16 * 4 + 8, dtype=torch.int16)        # 4.5: no kernel
        out[key + ".trellis"] = t
        out[key + ".suh"] = b.suh.view(-1)
        out[key + ".svh"] = b.svh
        if codebook == "mul1":
            out[key + ".mul1"] = torch.tensor([MUL1 - (1 << 32)], dtype=torch.int32)
        elif codebook == "mcg":
            out[key + ".mcg"] = torch.tensor([0], dtype=torch.int32)
    _write(d, out, TINY_EXL3, quant={"quant_method": "exl3", "bits": 3.0})
    return cfg, plain, blocks


def test_bf16_checkpoint_loads_and_runs_as_written():
    d = os.path.join(_tmp(), "bf16")
    cfg, W = _bf16_ckpt(d)
    idx = PackIndex.open(d)
    idx.audit()
    assert not idx.is_exl3 and set(idx.formats.values()) == {"bf16", "tied"}
    _, LW, experts, _ = load(d, device="cpu", dtype=torch.float32, log=None, idx=idx)
    a = Lfm2MoeRef(cfg, {k: v.float() for k, v in W.items()})
    b = Lfm2MoeRef(cfg, LW, experts=experts)
    sa, sb = a.new_state(32), b.new_state(32)
    assert torch.equal(a.prefill(PROMPT, sa), b.prefill(PROMPT, sb))
    db = idx.decode_bytes(context=16)
    assert db["total"] == sum(db["parts"].values()) and db["parts"]["kv_read"] == 16 * cfg.kv_bytes_per_token(2)
    assert "q_proj" not in json.dumps(summary(idx)["formats"])


def test_exl3_pack_audits_and_runs():
    d = os.path.join(_tmp(), "exl3")
    cfg, plain, blocks = _exl3_pack(d)
    idx = PackIndex.open(d)
    notes = idx.audit()
    idx.check_mul1()
    assert idx.is_exl3 and set(idx.formats.values()) == {"exl3"} and notes
    h = idx.k_histogram()
    assert set(h) >= {"experts.w1", "experts.w2", "experts.w3", "head"} and h["head"] == {4.0: 1}
    _, LW, experts, _ = load(d, device="cpu", dtype=torch.float32, log=None, idx=idx)
    assert LW["layers.1.self_attn.qkv"].sizes == [256, 128, 128]
    # the same model with every projection dequantised
    W = {k: v.float() for k, v in plain.items()}
    for base, b in blocks.items():
        W[base + ".weight"] = b.dequant().float()
    a = Lfm2MoeRef(cfg, W)
    b = Lfm2MoeRef(cfg, LW, experts=experts)
    la, lb = a.prefill(PROMPT, a.new_state(32)), b.prefill(PROMPT, b.new_state(32))
    rel = float((la - lb).abs().max() / la.abs().max())
    assert rel < 0.05, rel
    return f"EXL3 pack vs its dequantised weights: max rel {rel:.3f}"


def test_half_integer_rates_audit_and_run():
    """ENG-285: a pack with 1.5, 2.5 and 3.5 bits a weight beside whole rates (exllamav3 1.6.0 below
    4 bpw) loads, and runs as its dequantised weights (the reference decode of the 16 K + 8 tiles)."""
    d = os.path.join(_tmp(), "exl3-half")
    cfg, plain, blocks = _exl3_pack(d, halves=True)
    idx = PackIndex.open(d)
    idx.audit()
    h = idx.k_histogram()
    rates = {k for v in h.values() for k in v}
    assert {1.5, 2.5, 3.5} <= rates, h
    _, LW, experts, _ = load(d, device="cpu", dtype=torch.float32, log=None, idx=idx)
    assert LW["layers.1.self_attn.qkv"].kb == 35
    W = {k: v.float() for k, v in plain.items()}
    for base, b in blocks.items():
        W[base + ".weight"] = b.dequant().float()
    a = Lfm2MoeRef(cfg, W)
    b = Lfm2MoeRef(cfg, LW, experts=experts)
    la, lb = a.prefill(PROMPT, a.new_state(32)), b.prefill(PROMPT, b.new_state(32))
    rel = float((la - lb).abs().max() / la.abs().max())
    assert rel < 0.05, rel
    return f"rates {sorted(rates)}: max rel {rel:.3f}"


def test_exl3_pack_the_kernels_cannot_decode():
    for what, match in (("half", "4.5 bits a weight"), ("mcg", "mcg codebook"), ("3inst", "3inst codebook")):
        d = os.path.join(_tmp(), what)
        if what == "half":
            _exl3_pack(d, half="layers.2.conv.in_proj")
        else:
            _exl3_pack(d, codebook=what)
        _raises(lambda: PackIndex.open(d).audit(), match)


def test_missing_and_truncated():
    d = os.path.join(_tmp(), "bf16")
    _bf16_ckpt(d)
    wm = json.load(open(os.path.join(d, "model.safetensors.index.json")))
    wm["weight_map"].pop("model.layers.3.conv.out_proj.weight")
    json.dump(wm, open(os.path.join(d, "model.safetensors.index.json"), "w"))
    _raises(lambda: PackIndex.open(d).audit(), "layers.3.conv.out_proj is missing")
    fn = os.path.join(d, "model-00001-of-00002.safetensors")
    with open(fn, "r+b") as f:
        f.truncate(os.path.getsize(fn) - 10)
    _raises(lambda: PackIndex.open(d), "truncated or corrupt")


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
    for d in _TMP:
        shutil.rmtree(d, ignore_errors=True)
    sys.exit(1 if fails else 0)
