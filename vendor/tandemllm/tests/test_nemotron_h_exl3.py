"""NemotronH from an EXL3 pack (ENG-E12): a tiny Bolt-like pack written on the CPU the way exllamav3 with the
Puzzle/Bolt patch writes one (the Omni wrapper's `language_model.` keys and quantization_config, every
projection a mul1 trellis with its channel factors, the router, norms, conv taps and embedding in fp16, a
vision tensor the reader skips) audits clean, loads every layer kind as EXL3 (Mamba2 in_proj / out_proj,
attention q | k | v as one group and o_proj, the latent pair, the shared expert, the routed experts as a
`Relu2Bank`, the head with fp32 logits), and runs in the engine near the same model with every projection
dequantised. A wrong trellis shape, the mcg codebook and a rate the kernels do not take stop with the name.

Engine checks run on the GPU or under TRITON_INTERPRET=1 (the CPU interpreter); the EXL3 products are then
tools/exl3_linear.py's torch reference."""

from __future__ import annotations

import json
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.models.lfm2_moe.pack import MUL1  # noqa: E402
from engine.models.nemotron_h import kernels as K  # noqa: E402
from engine.models.nemotron_h.config import config_from_dict  # noqa: E402
from engine.models.nemotron_h.load import load  # noqa: E402
from engine.models.nemotron_h.pack import PackIndex, expected, summary  # noqa: E402
from tests.test_nemotron_h_ref import PUZZLE, random_weights  # noqa: E402

SKIP = None if (K.HAVE_TRITON and (torch.cuda.is_available() or os.environ.get("TRITON_INTERPRET"))) else \
    "skipped: Triton on a GPU, or TRITON_INTERPRET=1 (the CPU interpreter)"
DEV = "cuda" if torch.cuda.is_available() and not os.environ.get("TRITON_INTERPRET") else "cpu"

# Every projection a multiple of 128 both ways (EXL3's Hadamard blocks): in_proj 128 + 248 + 8 = 384
TEXT = {**PUZZLE, "vocab_size": 256, "hidden_size": 128, "num_attention_heads": 4, "num_key_value_heads": 2,
        "head_dim": 64, "mamba_num_heads": 8, "mamba_head_dim": 16, "ssm_state_size": 30, "n_groups": 2,
        "moe_shared_expert_intermediate_size": 128, "moe_latent_size": 128, "chunk_size": 8,
        "block_configs": [dict(b, moe_intermediate_size=128 * (i % 2 + 1)) if b["block_type"] == "moe" else b
                          for i, b in enumerate(PUZZLE["block_configs"])]}
OMNI = {"architectures": ["NemotronH_Omni_Reasoning_V3"], "model_type": "nemotron_h_omni", "llm_config": TEXT,
        "eos_token_id": 11, "vision_config": {}, "tie_word_embeddings": False,
        "quantization_config": {"quant_method": "exl3", "version": "1.6.2", "bits": 2.0, "head_bits": 6}}
PROMPT = torch.tensor([5, 9, 3, 17, 22, 3, 40, 41, 12, 7, 63, 2, 30, 88, 101, 7, 9, 200, 31])


def _projections(cfg) -> list[tuple[str, int, int]]:
    """(base, N, K) of every projection the pack quantises: the router stays fp16, as exllamav3 keeps it."""
    out = []
    for name, shape in expected(cfg, mtp=False):
        if len(shape) == 2 and name.endswith("_proj.weight") or name == "lm_head.weight":
            out.append((name[:-len(".weight")], shape[0], shape[1]))
    return out


def _kb(base: str, i: int) -> int:
    # bits as the converter gives them at 2.0 -hq: experts 2, the select projections 4, the head 6; one
    # half-integer rate among the experts and q | k | v at one rate (one fused launch)
    if base == "lm_head":
        return 6
    if ".experts." in base:
        return 25 if i % 7 == 0 else 2
    return 4


def _write_pack(d: str, half_kb: int | None = None, codebook: str = "mul1", bad_shape: str | None = None):
    from safetensors.torch import save_file
    from tools.exl3_linear import random_block
    cfg = config_from_dict(OMNI, d)
    W = random_weights(cfg)
    projs = _projections(cfg)
    bases = {b for b, _, _ in projs}
    out = {}
    for k, v in W.items():
        if k.endswith(".weight") and k[:-len(".weight")] in bases:
            continue
        key = "language_model." + (k if k.startswith("lm_head") else "backbone." + k)
        out[key] = v.float() if k.endswith(("e_score_correction_bias", "A_log", "dt_bias", ".D")) else v.to(torch.float16)
    blocks = {}
    for i, (base, N, Kin) in enumerate(projs):
        kb = half_kb if (half_kb is not None and i == 3) else _kb(base, i)
        b = random_block(N, Kin, kb, device="cpu", seed=i)
        blocks[base] = b
        key = "language_model." + (base if base == "lm_head" else "backbone." + base)
        t = b.t
        if base == bad_shape:
            t = t[:, :-1].contiguous()
        out[key + ".trellis"] = t
        out[key + ".suh"] = b.suh.view(-1)
        out[key + ".svh"] = b.svh
        out[key + (".mul1" if codebook == "mul1" else ".mcg")] = torch.tensor([MUL1 - (1 << 32)], dtype=torch.int32)
    out["vision_model.radio_model.model.blocks.0.attn.qkv.weight"] = torch.zeros(8, 8, dtype=torch.bfloat16)
    keys = sorted(out)
    wm = {}
    for s in range(2):
        fn = f"model-{s + 1:05d}-of-00002.safetensors"
        save_file({k: out[k].contiguous() for k in keys[s::2]}, os.path.join(d, fn))
        wm.update({k: fn for k in keys[s::2]})
    with open(os.path.join(d, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {}, "weight_map": wm}, f)
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump(OMNI, f)
    return cfg, W, blocks


def _raises(fn, *needles):
    try:
        fn()
    except ValueError as e:
        for n in needles:
            assert n in str(e), (n, str(e))
        return
    raise AssertionError(f"no error naming {needles}")


def test_pack_audits_and_loads_every_layer_kind_as_exl3():
    from tools.exl3_linear import EXL3Block
    from tools.exl3_relu2 import Relu2Bank
    with tempfile.TemporaryDirectory() as d:
        cfg, W, blocks = _write_pack(d)
        idx = PackIndex.open(d)
        notes = idx.audit()
        idx.check_mul1()
        assert idx.is_exl3 and idx.skipped == 1 and idx.cfg.eos_token_id == 11
        n_exl3 = sum(1 for f in idx.formats.values() if f == "exl3")
        assert n_exl3 == len(blocks) and idx.formats["layers.1.mixer.gate"] == "f16", notes
        assert "EXL3 bits" in notes[-1]
        # bytes a token: a projection's trellis and factors, the experts at each layer's mean
        db = idx.decode_bytes()
        b = blocks["layers.0.mixer.in_proj"]
        assert idx.nbytes("layers.0.mixer.in_proj.weight") == b.t.numel() * 2 + (b.K + b.N) * 2
        assert db["total"] == sum(db["parts"].values()) and db["parts"]["experts"] > 0
        h = blocks["lm_head"]
        assert db["parts"]["head"] == h.t.numel() * 2 + (h.K + h.N) * 2          # the pack's head, not the embedding
        assert summary(idx)["quant"]["bits"] == 2.0
        _, LW, experts, timing = load(d, "cpu", log=None, idx=idx)
        for base in ("layers.0.mixer.in_proj", "layers.0.mixer.out_proj", "layers.1.mixer.fc1_latent_proj",
                     "layers.1.mixer.fc2_latent_proj", "layers.1.mixer.shared_experts.up_proj",
                     "layers.1.mixer.shared_experts.down_proj", "layers.2.mixer.o_proj"):
            assert isinstance(LW[base + ".weight"], EXL3Block), base
        assert LW["layers.2.mixer.qkv"].sizes == [cfg.q_dim, cfg.kv_dim, cfg.kv_dim]
        assert LW["lm_head.weight"].out_fp32 and LW["layers.1.mixer.gate.weight"].dtype == torch.bfloat16
        for li in cfg.moe_layers:
            assert isinstance(experts[li], Relu2Bank) and experts[li].E == cfg.moe_experts[li]
            assert (experts[li].I, experts[li].X) == (cfg.moe_ffn[li], cfg.moe_latent_size)


def test_flat_pack_with_a_stored_head_under_a_tied_config():
    """ENG-377: a community pack's layout (PR #63): a flat NemotronH config with `tie_word_embeddings` true and the
    quantization_config at the top, `backbone.*` keys, no tower, and the head stored in fp16 beside the embedding.
    The stored head is read (the tie dropped, an audit note); without it the embedding is the head."""
    from safetensors.torch import load_file, save_file
    for stored in (True, False):
        with tempfile.TemporaryDirectory() as d:
            cfg, W, blocks = _write_pack(d)
            flat = {**TEXT, "architectures": ["NemotronHForCausalLM"], "model_type": "nemotron_h", "eos_token_id": 11,
                    "tie_word_embeddings": True, "quantization_config": dict(OMNI["quantization_config"], head_bits=16)}
            head = torch.randn(cfg.vocab_size, cfg.hidden_size, generator=torch.Generator().manual_seed(7)).to(torch.float16)
            for fn in sorted(f for f in os.listdir(d) if f.endswith(".safetensors")):
                t = load_file(os.path.join(d, fn))
                t = {k[len("language_model."):]: v for k, v in t.items() if k.startswith("language_model.")
                     and not k.startswith("language_model.lm_head.")}
                save_file(t, os.path.join(d, fn))
            if stored:
                save_file({"lm_head.weight": head}, os.path.join(d, "model-head.safetensors"))
            wm = {}
            for fn in sorted(f for f in os.listdir(d) if f.endswith(".safetensors")):
                wm.update({k: fn for k in load_file(os.path.join(d, fn))})
            json.dump({"metadata": {}, "weight_map": wm}, open(os.path.join(d, "model.safetensors.index.json"), "w"))
            json.dump(flat, open(os.path.join(d, "config.json"), "w"))
            idx = PackIndex.open(d)
            notes = idx.audit()
            assert idx.skipped == 0 and idx.is_exl3 and idx.stored_head is stored, notes
            assert idx.cfg.tie_word_embeddings is (not stored)
            assert any("stored head is read" in n for n in notes) is stored, notes
            _, LW, _, _ = load(d, "cpu", log=None, idx=idx)
            if stored:
                assert idx.formats["lm_head"] == "f16"
                assert torch.equal(LW["lm_head.weight"], head.to(torch.bfloat16))
                assert not torch.equal(LW["lm_head.weight"], LW["embeddings.weight"])
            else:
                assert LW["lm_head.weight"] is LW["embeddings.weight"]


def test_bad_packs_stop_with_the_name():
    for kw, needles in (({"bad_shape": "layers.0.mixer.in_proj"}, ("layers.0.mixer.in_proj.trellis", "does not fit")),
                        ({"codebook": "mcg"}, ("mul1",)),
                        ({"half_kb": 45}, ("4.5 bits",))):
        with tempfile.TemporaryDirectory() as d:
            if kw.get("half_kb") == 45:
                # 4.5 bits: a 16 * 4 + 8 wide tile, a rate no kernel decodes (random_block refuses it, so write by hand)
                cfg, W, blocks = _write_pack(d)
                from safetensors.torch import load_file, save_file
                for fn in sorted(f for f in os.listdir(d) if f.endswith(".safetensors")):
                    t = load_file(os.path.join(d, fn))
                    key = "language_model.backbone.layers.0.mixer.out_proj.trellis"
                    if key in t:
                        t[key] = torch.zeros(t[key].shape[0], t[key].shape[1], 72, dtype=torch.int16)
                        save_file(t, os.path.join(d, fn))
            else:
                _write_pack(d, **kw)
            _raises(lambda: PackIndex.open(d).audit(), *needles)


def _dequantised(cfg, W, blocks):
    """The model with every projection dequantised (bf16 values, fp32 arithmetic): the yardstick."""
    from engine.models.nemotron_h.ref import NemotronHRef
    f32 = ("e_score_correction_bias", "A_log", "dt_bias", ".D")
    R = {k: (v.float() if k.endswith(f32) else v.to(torch.bfloat16).float()) for k, v in W.items()}
    for base, b in blocks.items():
        R[base + ".weight"] = b.dequant().float()
    return NemotronHRef(cfg, R)


def test_relu2_bank_is_the_experts_arithmetic():
    """`Relu2Bank.moe_relu2` on the CPU: each pair's bf16(relu(bf16(up x))^2) through down, against the
    dequantised experts (the EXL3 product rounds its rotated input to fp16: near, not equal)."""
    from tools.exl3_linear import random_block
    from tools.exl3_relu2 import Relu2Bank
    E, I, Xw = 4, 256, 128
    bank = Relu2Bank([(random_block(I, Xw, 2 if e % 2 else 25, device="cpu", seed=e),
                       random_block(Xw, I, 3, device="cpu", seed=10 + e)) for e in range(E)])
    g = torch.Generator().manual_seed(0)
    x = (torch.randn(3, Xw, generator=g) * 4).to(torch.bfloat16)
    ids = torch.tensor([[0, 3], [2, 2], [1, 0]], dtype=torch.int32)
    yd = bank.moe_relu2(x, ids)
    assert yd.shape == (6, Xw) and yd.dtype == torch.float32
    assert torch.equal(yd, yd.to(torch.bfloat16).float())                  # bf16 values
    for p, e in enumerate(ids.reshape(-1).tolist()):
        up, down = bank.expert(e)
        a = torch.relu(x[p // 2:p // 2 + 1].float() @ up.dequant().float().T).square()
        want = a @ down.dequant().float().T
        err = (yd[p] - want[0]).abs().max() / want.abs().max()
        assert err < 0.03, (p, e, float(err))


def test_engine_runs_the_pack_near_the_dequantised_model():
    if SKIP:
        return SKIP
    from engine.models.nemotron_h.engine import NemotronHEngine
    tol = 0.08 if DEV == "cuda" else 0.3
    with tempfile.TemporaryDirectory() as d:
        cfg, W, blocks = _write_pack(d)
        idx, LW, experts, _ = load(d, DEV, log=None)
        eng = NemotronHEngine(idx.cfg, LW, experts, device=DEV, max_len=256)
        ref = _dequantised(idx.cfg, W, blocks)
        st = eng.new_state(256)
        lg = eng.prefill(PROMPT, st)[0].float().cpu()
        rs = ref.new_state(64)
        want = ref.prefill(PROMPT, rs)[-1]
        assert (lg - want).abs().max() < tol * want.abs().max(), float((lg - want).abs().max() / want.abs().max())
        for t in (11, 50, 3):
            eng.stage(t, st)
            got = eng.step(st)[0].float().cpu()
            st.length += 1
            w = ref.decode(t, rs)[0]
            assert (got - w).abs().max() < tol * w.abs().max(), t
        # a verify block's rows are the decode steps' and the commit gives their states
        res = eng.self_check(PROMPT.tolist(), steps=6, log=None)
        assert res["ok"], res


if __name__ == "__main__":
    fails = 0
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        try:
            r = fn()
            print(f"  {name:<60} ok   {r or ''}")
        except Exception as e:  # noqa: BLE001
            fails += 1
            import traceback
            traceback.print_exc()
            print(f"  {name:<60} FAIL {type(e).__name__}: {e}")
    print(f"{len(tests) - fails} passed" + (f", {fails} failed" if fails else ""))
    sys.exit(1 if fails else 0)
