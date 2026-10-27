"""NemotronH from NVIDIA's ModelOpt checkpoint (Nemotron 3.5 Super VL's NVFP4 release): a tiny Super-VL-like
Omni checkpoint written the way ModelOpt 0.37 MIXED_PRECISION writes one -- routed and shared experts NVFP4
(uint8 codes, e4m3 group scales, a tensor scale, an input scale), the Mamba2 and attention projections and the
head FP8 (e4m3 codes, a tensor scale, an input scale), the latent pair, router, norms and embedding BF16, the MTP
head's attention in BF16 with KV scales and its experts NVFP4, a vision projector the reader skips -- audits
clean, loads each projection in its format (`FP8Linear`, `NVFP4Wide`, `NVFP4Bank`), and runs in the engine near
the same model with every projection dequantised; a verify block's rows are the decode steps'. The bank and the
FP8 rows kernel are held to their references and to their own one-row results.

Engine checks run on the GPU or under TRITON_INTERPRET=1 (the CPU interpreter; the NVFP4 and FP8 products are
then the torch references)."""

from __future__ import annotations

import json
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.models.nemotron_h import kernels as K  # noqa: E402
from engine.models.nemotron_h.config import config_from_dict  # noqa: E402
from engine.models.nemotron_h.load import load  # noqa: E402
from engine.models.nemotron_h.pack import PackIndex, expected  # noqa: E402
from tests.test_nemotron_h_ref import LATENT, random_weights  # noqa: E402

SKIP = None if (K.HAVE_TRITON and (torch.cuda.is_available() or os.environ.get("TRITON_INTERPRET"))) else \
    "skipped: Triton on a GPU, or TRITON_INTERPRET=1 (the CPU interpreter)"
DEV = "cuda" if torch.cuda.is_available() and not os.environ.get("TRITON_INTERPRET") else "cpu"

# Super VL's shape at toy size: uniform experts, a latent MoE, an MTP head (attention + MoE); every K a multiple
# of 128 (NVFP4's K step): in_proj 128 + 248 + 8, inner 128
TEXT = {**{k: v for k, v in LATENT.items() if k != "num_nextn_predict_layers"},
        "vocab_size": 256, "hidden_size": 128, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 64,
        "mamba_num_heads": 8, "mamba_head_dim": 16, "ssm_state_size": 30, "n_groups": 2, "chunk_size": 8,
        "n_routed_experts": 8, "num_experts_per_tok": 3, "moe_intermediate_size": 128, "moe_latent_size": 128,
        "moe_shared_expert_intermediate_size": 128, "routed_scaling_factor": 2.5,
        "layers_block_type": ["mamba", "moe", "attention", "mamba", "moe", "mamba", "attention", "moe"],
        "num_nextn_predict_layers": 1, "mtp_layers_block_type": ["attention", "moe"]}
OMNI = {"architectures": ["NemotronH_Omni_Reasoning_V3"], "model_type": "nemotron_h_omni", "llm_config": TEXT,
        "eos_token_id": 11, "vision_config": {}, "sound_config": None, "tie_word_embeddings": False}
QUANT = {"producer": {"name": "modelopt", "version": "0.37.0"},
         "quantization": {"quant_algo": "MIXED_PRECISION", "kv_cache_quant_algo": "FP8"}}
PROMPT = torch.tensor([5, 9, 3, 17, 22, 3, 40, 41, 12, 7, 63, 2, 30, 88, 101, 7, 9, 200, 31])


def _fmt(name: str) -> str | None:
    """The format ModelOpt gives a projection of Super VL (None: BF16)."""
    if not name.endswith(".weight") or "latent" in name or ".gate." in name or "embeddings" in name or "norm" in name:
        return None
    if ".experts." in name or "shared_experts" in name:
        return "nvfp4"
    if name.startswith("mtp."):
        return None                                        # the MTP head's attention and eh_proj stay BF16
    if name == "lm_head.weight" or any(f"mixer.{p}_proj" in name for p in ("in", "out", "q", "k", "v", "o")):
        return "fp8"
    return None


def _fp8(w: torch.Tensor):
    s = w.abs().max().float() / 448.0
    return (w.float() / s).to(torch.float8_e4m3fn), s


def _write(d: str, bad: str | None = None):
    from safetensors.torch import save_file
    from tools.nvfp4_linear import quantize_to_nvfp4
    cfg = config_from_dict(OMNI, d)
    W = random_weights(cfg)
    g = torch.Generator().manual_seed(7)
    for name, shape in expected(cfg, mtp=True):
        if name not in W:
            W[name] = (1 + 0.1 * torch.randn(*shape, generator=g)) if name.endswith("norm.weight") else \
                (0.05 * torch.randn(*shape, generator=g) if name.endswith("bias") else 0.2 * torch.randn(*shape, generator=g))
    # a decisive router: the correction bias (the choice only) far apart, so no near tie flips an expert between
    # two arithmetics of the same weights (random gates gave ties, and one flipped choice is a different model)
    for k in W:
        if k.endswith("gate.e_score_correction_bias"):
            W[k] = torch.arange(W[k].numel(), dtype=torch.float32) * 2.0
    out, deq = {}, {}
    for k, v in W.items():
        key = "language_model." + (k if k.startswith(("lm_head", "mtp.")) else "backbone." + k)
        base = key[:-len(".weight")]
        f = _fmt(k)
        if f == "nvfp4":
            b = quantize_to_nvfp4(v.float())
            s = b.s if bad != k else b.s[:, :-1]
            out.update({key: b.w, base + ".weight_scale": s, base + ".weight_scale_2": torch.tensor(b.s2),
                        base + ".input_scale": torch.tensor(1.0)})
            deq[k] = b.dequant().float()
        elif f == "fp8":
            c, s = _fp8(v)
            out.update({key: c, base + ".weight_scale": s.reshape(()), base + ".input_scale": torch.tensor([1.0])})
            deq[k] = c.float() * s
        else:
            out[key] = v.float() if k.endswith(("e_score_correction_bias", "A_log", "dt_bias", ".D")) else v.to(torch.bfloat16)
            deq[k] = out[key].float()
    out["language_model.mtp.layers.0.mixer.k_proj.k_scale"] = torch.tensor([1.0])
    out["language_model.mtp.layers.0.mixer.v_proj.v_scale"] = torch.tensor([1.0])
    out["vision_projector.vision_final_layernorm.weight"] = torch.ones(8, dtype=torch.bfloat16)
    keys = sorted(out)
    wm = {}
    for s in range(2):
        fn = f"model-{s + 1:05d}-of-00002.safetensors"
        save_file({k: out[k].contiguous() for k in keys[s::2]}, os.path.join(d, fn))
        wm.update({k: fn for k in keys[s::2]})
    for name, obj in (("model.safetensors.index.json", {"metadata": {}, "weight_map": wm}),
                      ("config.json", OMNI), ("hf_quant_config.json", QUANT)):
        with open(os.path.join(d, name), "w") as f:
            json.dump(obj, f)
    return cfg, W, deq


def test_checkpoint_audits_and_loads_each_format():
    from tools.nvfp4_relu2 import FP8Linear, NVFP4Bank, NVFP4Wide
    with tempfile.TemporaryDirectory() as d:
        cfg, W, deq = _write(d)
        idx = PackIndex.open(d)
        notes = idx.audit()
        assert idx.has_mtp and idx.skipped == 1 and idx.cfg.eos_token_id == 11
        assert idx.formats["layers.0.mixer.in_proj"] == "fp8" and idx.formats["lm_head"] == "fp8"
        assert idx.formats["layers.1.mixer.experts.0.up_proj"] == "nvfp4"
        assert idx.formats["layers.1.mixer.fc1_latent_proj"] == "bf16"
        assert idx.formats["mtp.layers.0.mixer.q_proj"] == "bf16" and idx.formats["mtp.layers.1.mixer.experts.3.down_proj"] == "nvfp4"
        assert any("ModelOpt MIXED_PRECISION" in n for n in notes), notes
        # bytes a token: an FP8 projection's codes, an NVFP4 one's codes and group scales
        assert idx.nbytes("layers.0.mixer.in_proj.weight") == cfg.in_proj_dim * cfg.hidden_size + 4
        assert idx.nbytes("layers.1.mixer.shared_experts.up_proj.weight") == cfg.shared_ffn * cfg.hidden_size * 9 // 16
        _, LW, experts, _ = load(d, "cpu", log=None, idx=idx)
        assert isinstance(LW["layers.0.mixer.in_proj.weight"], FP8Linear) and isinstance(LW["lm_head.weight"], FP8Linear)
        up, down = LW["layers.1.mixer.shared_experts.up_proj.weight"], LW["layers.1.mixer.shared_experts.down_proj.weight"]
        assert isinstance(up, NVFP4Wide) and isinstance(down, NVFP4Wide) and down.wide and not up.wide
        assert "layers.2.mixer.qkv" not in LW                     # three FP8 tensors, three scales: not fused
        for li in list(cfg.moe_layers) + [("mtp", 1)]:
            e = experts[li]
            assert isinstance(e, NVFP4Bank) and (e.E, e.I, e.X) == (8, 128, 128), li
        # the loaded weights are the checkpoint's values
        assert torch.equal(LW["layers.0.mixer.in_proj.weight"].dequant().cpu(), deq["layers.0.mixer.in_proj.weight"])
        u0, d0 = experts[1].expert(0)
        assert torch.equal(u0.float(), deq["layers.1.mixer.experts.0.up_proj.weight"].to(torch.bfloat16).float())
        assert torch.equal(d0.float(), deq["layers.1.mixer.experts.0.down_proj.weight"].to(torch.bfloat16).float())


def test_bad_scales_stop_with_the_name():
    with tempfile.TemporaryDirectory() as d:
        _write(d, bad="layers.4.mixer.experts.2.up_proj.weight")
        try:
            PackIndex.open(d).audit()
        except ValueError as e:
            assert "layers.4.mixer.experts.2.up_proj" in str(e) and "does not fit" in str(e), str(e)
            return
        raise AssertionError("a wrong NVFP4 scale table passed the audit")


def test_bank_is_the_experts_arithmetic_and_row_invariant():
    """`NVFP4Bank.moe_relu2`: every pair bf16(relu(bf16(up x))^2) through down, bf16 values, against the
    dequantised experts in fp32 (the kernel's sums run in another order: near); on the GPU each row of a
    32-row call is the row's own call bit for bit."""
    from tools.nvfp4_linear import quantize_to_nvfp4
    from tools.nvfp4_relu2 import NVFP4Bank
    E, I, X = 6, 256, 128
    g = torch.Generator().manual_seed(1)
    ups = [quantize_to_nvfp4(torch.randn(I, X, generator=g) * 0.2) for _ in range(E)]
    downs = [quantize_to_nvfp4(torch.randn(X, I, generator=g) * 0.2) for _ in range(E)]
    bank = NVFP4Bank.stack([((u.w, u.s, u.s2), (dn.w, dn.s, dn.s2)) for u, dn in zip(ups, downs)], DEV)
    x = (torch.randn(32, X, generator=g) * 8).to(torch.bfloat16).to(DEV)     # relu^2 past fp16's range
    ids = torch.randint(0, E, (32, 3), generator=g, dtype=torch.int32).to(DEV)
    yd = bank.moe_relu2(x, ids).cpu()
    assert torch.equal(yd, yd.to(torch.bfloat16).float())
    for p, e in enumerate(ids.reshape(-1).tolist()):
        a = torch.relu(ups[e].dequant().float() @ x[p // 3].float().cpu()).to(torch.bfloat16).float().square()
        want = downs[e].dequant().float() @ a.to(torch.bfloat16).float()
        err = (yd[p] - want).abs().max() / want.abs().max()
        assert err < 0.02, (p, e, float(err))
    assert torch.isfinite(yd).all()
    if DEV == "cuda":
        for r in (0, 7, 31):
            one = bank.moe_relu2(x[r:r + 1], ids[r:r + 1]).cpu()
            assert torch.equal(one, yd[3 * r:3 * r + 3]), r


def test_fp8_rows_are_row_invariant_and_near_the_reference():
    from tools.nvfp4_relu2 import FP8Linear
    g = torch.Generator().manual_seed(2)
    for N, Kk in ((384, 128), (128, 256), (1000, 192)):
        c, s = _fp8(torch.randn(N, Kk, generator=g) * 0.1)
        w = FP8Linear(c.to(DEV), s)
        x = torch.randn(32, Kk, generator=g).to(torch.bfloat16).to(DEV)
        want = x.float().cpu() @ (c.float() * s).T
        for f32 in (False, True):
            y = w.matmul(x, f32=f32).float().cpu()
            assert (y - want).abs().max() < 0.02 * want.abs().max(), (N, Kk, f32)
            if DEV == "cuda":
                for m in (1, 5, 16, 17):
                    assert torch.equal(w.matmul(x[:m], f32=f32).float().cpu(), y[:m]), (N, Kk, m, f32)


def _dequantised(cfg, W, deq):
    from engine.models.nemotron_h.ref import NemotronHRef
    f32 = ("e_score_correction_bias", "A_log", "dt_bias", ".D")
    R = {k: (v.float() if k.endswith(f32) else v.to(torch.bfloat16).float()) for k, v in W.items()}
    R.update({k: v.float() for k, v in deq.items() if k in R})
    return NemotronHRef(cfg, R)


def test_engine_runs_the_checkpoint_near_the_dequantised_model():
    if SKIP:
        return SKIP
    from engine.models.nemotron_h.engine import NemotronHEngine
    tol = 0.08 if DEV == "cuda" else 0.3
    with tempfile.TemporaryDirectory() as d:
        cfg, W, deq = _write(d)
        idx, LW, experts, _ = load(d, DEV, mtp=False, log=None)
        eng = NemotronHEngine(idx.cfg, LW, experts, device=DEV, max_len=256)
        ref = _dequantised(idx.cfg, W, deq)
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
