"""A NemotronH GGUF in llama.cpp's layout (`nemotron_h_moe`), written on the CPU with gguf-py from a tiny
random model (IQ2_XXS experts and a 2-D IQ2_XXS weight from random blocks, a Q8_0 projection, Bolt's
NVFP4 out_proj with its tensor scale, the rest F32/BF16 in llama.cpp's own Mamba2 forms): the reader's config equals the model's, the loader's engine
answers near the same model with the packed weights decoded by the twins (bf16 at load), and its
load-time self-check passes (verify rows = decode steps bit for bit on the packed kernels). Skipped
without gguf-py (with NVFP4: llama.cpp's, after GGML_TYPE_NVFP4 = 40), or without a GPU / TRITON_INTERPRET=1 for the engine part."""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.models.nemotron_h import kernels as K  # noqa: E402
from engine.models.nemotron_h.config import config_from_dict  # noqa: E402
from tests.test_iq2_kernels import random_iq2  # noqa: E402
from tests.test_nemotron_h_ref import TINY, random_weights  # noqa: E402
from tools import iq2_linear as Q  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() and not os.environ.get("TRITON_INTERPRET") else "cpu"
KERNELS = K.HAVE_TRITON and (torch.cuda.is_available() or os.environ.get("TRITON_INTERPRET"))
RAW = {**TINY, "vocab_size": 256, "hidden_size": 256, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 64,
       "mamba_num_heads": 4, "mamba_head_dim": 16, "ssm_state_size": 24, "n_groups": 2, "n_routed_experts": 4,
       "moe_intermediate_size": 256, "moe_shared_expert_intermediate_size": 256, "num_experts_per_tok": 2}
PROMPT = [5, 9, 3, 17, 22, 3, 40, 41, 12, 7, 63, 2, 30, 88]


def random_q8(N: int, K_: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    nb = N * K_ // 32
    b = rng.integers(-40, 40, size=(nb, 34)).astype(np.int8).view(np.uint8)
    b[:, :2] = (rng.random(nb) * 0.004 + 0.001).astype(np.float16).view(np.uint8).reshape(nb, 2)
    return b.reshape(-1)


def random_nvfp4_small(N: int, K_: int, seed: int) -> np.ndarray:
    """NVFP4 blocks with sub-block scales between 0.125 and 0.47 (UE4M3 bytes 0x28..0x3F)."""
    rng = np.random.default_rng(seed)
    b = rng.integers(0, 256, size=(N * K_ // 64, 36), dtype=np.uint8)
    b[:, :4] = rng.integers(0x28, 0x40, size=(b.shape[0], 4), dtype=np.uint8)
    return b.reshape(-1)


NVFP4_SCALE = 0.02


def write_gguf(path: str):
    """(cfg, W with every weight as the engine should read it) of the file written to `path`."""
    import gguf
    cfg = config_from_dict(RAW, "tiny")
    W = random_weights(cfg)
    w = gguf.GGUFWriter(path, "nemotron_h_moe")
    a = "nemotron_h_moe."
    n = cfg.num_hidden_layers
    w.add_uint32(a + "block_count", n)
    w.add_uint32(a + "context_length", 4096)
    w.add_uint32(a + "embedding_length", cfg.hidden_size)
    w.add_array(a + "feed_forward_length", [cfg.moe_ffn.get(i, 0) for i in range(n)])
    w.add_uint32(a + "attention.head_count", cfg.num_attention_heads)
    w.add_array(a + "attention.head_count_kv", [cfg.num_key_value_heads if t == "attention" else 0 for t in cfg.layer_types])
    w.add_float32(a + "attention.layer_norm_rms_epsilon", cfg.rms_norm_eps)
    w.add_uint32(a + "attention.key_length", cfg.head_dim)
    w.add_uint32(a + "expert_count", 4)
    w.add_uint32(a + "expert_used_count", 2)
    w.add_uint32(a + "vocab_size", cfg.vocab_size)
    w.add_uint32(a + "ssm.conv_kernel", cfg.conv_kernel)
    w.add_uint32(a + "ssm.state_size", cfg.ssm_state_size)
    w.add_uint32(a + "ssm.group_count", cfg.n_groups)
    w.add_uint32(a + "ssm.inner_size", cfg.intermediate)
    w.add_uint32(a + "ssm.time_step_rank", cfg.mamba_num_heads)
    w.add_uint32(a + "expert_shared_feed_forward_length", cfg.shared_ffn)
    w.add_bool(a + "expert_weights_norm", True)
    w.add_float32(a + "expert_weights_scale", cfg.routed_scaling_factor)
    w.add_uint32("tokenizer.ggml.eos_token_id", 2)
    T = gguf.GGMLQuantizationType
    want = {}

    def put(name, canon, x: torch.Tensor, keep=None):
        w.add_tensor(name, x.float().numpy())
        want[canon] = keep if keep is not None else x

    def put_q(name, canon, raw, N, K_, qtype, deq):
        w.add_tensor(name, raw.reshape(N, -1), raw_dtype=qtype)            # the shape in bytes a row
        want[canon] = deq(raw, N, K_)
    put("token_embd.weight", "embeddings.weight", W["embeddings.weight"])
    put("output_norm.weight", "norm_f.weight", W["norm_f.weight"])
    put("output.weight", "lm_head.weight", W["lm_head.weight"])
    for li, t in enumerate(cfg.layer_types):
        b, m = f"blk.{li}.", f"layers.{li}.mixer."
        put(b + "attn_norm.weight", f"layers.{li}.norm.weight", W[f"layers.{li}.norm.weight"])
        if t == "mamba":
            put_q(b + "ssm_in.weight", m + "in_proj.weight", random_q8(cfg.in_proj_dim, cfg.hidden_size, li),
                  cfg.in_proj_dim, cfg.hidden_size, T.Q8_0, Q.dequant_q8_0)
            put(b + "ssm_conv1d.weight", m + "conv1d.weight", W[m + "conv1d.weight"].reshape(cfg.conv_dim, -1),
                W[m + "conv1d.weight"])
            put(b + "ssm_conv1d.bias", m + "conv1d.bias", W[m + "conv1d.bias"])
            put(b + "ssm_dt.bias", m + "dt_bias", W[m + "dt_bias"])
            A = -torch.exp(W[m + "A_log"])
            put(b + "ssm_a", m + "A", A.reshape(-1, 1), A)
            put(b + "ssm_d", m + "D", W[m + "D"].reshape(-1, 1), W[m + "D"])
            put(b + "ssm_norm.weight", m + "norm.weight", W[m + "norm.weight"].reshape(cfg.n_groups, -1),
                W[m + "norm.weight"])
            put_q(b + "ssm_out.weight", m + "out_proj.weight", random_nvfp4_small(cfg.hidden_size, cfg.intermediate, li),
                  cfg.hidden_size, cfg.intermediate, T.NVFP4, lambda r, N, K_: Q.dequant_nvfp4(r, N, K_) * NVFP4_SCALE)
            w.add_tensor(b + "ssm_out.scale", np.array([NVFP4_SCALE], dtype=np.float32))
        elif t == "attention":
            for x, y in (("attn_q", "q_proj"), ("attn_k", "k_proj"), ("attn_v", "v_proj"), ("attn_output", "o_proj")):
                put(b + x + ".weight", m + y + ".weight", W[m + y + ".weight"])
        else:
            E, I, X = cfg.moe_experts[li], cfg.moe_ffn[li], cfg.expert_in
            put(b + "ffn_gate_inp.weight", m + "gate.weight", W[m + "gate.weight"])
            put(b + "exp_probs_b.bias", m + "gate.e_score_correction_bias", W[m + "gate.e_score_correction_bias"])
            up, down = random_iq2(E * I, X, seed=100 + li), random_iq2(E * X, I, seed=200 + li)
            w.add_tensor(b + "ffn_up_exps.weight", up.reshape(E, I, -1), raw_dtype=T.IQ2_XXS)
            w.add_tensor(b + "ffn_down_exps.weight", down.reshape(E, X, -1), raw_dtype=T.IQ2_XXS)
            want[(li, "up")] = Q.dequant_iq2_xxs(up, E * I, X).view(E, I, X)
            want[(li, "down")] = Q.dequant_iq2_xxs(down, E * X, I).view(E, X, I)
            put_q(b + "ffn_up_shexp.weight", m + "shared_experts.up_proj.weight", random_iq2(cfg.shared_ffn, cfg.hidden_size, li),
                  cfg.shared_ffn, cfg.hidden_size, T.IQ2_XXS, Q.dequant_iq2_xxs)
            put(b + "ffn_down_shexp.weight", m + "shared_experts.down_proj.weight", W[m + "shared_experts.down_proj.weight"])
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return cfg, want


def test_gguf_reads_and_serves_near_its_twin():
    try:
        import gguf
        gguf.GGMLQuantizationType.NVFP4
    except Exception:  # noqa: BLE001
        return "skipped: gguf-py missing or without NVFP4"
    from engine.models.nemotron_h.gguf_load import config_from_gguf, load_gguf
    from engine.models.nemotron_h.gguf import GGUFFile
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "tiny.gguf")
        cfg, want = write_gguf(p)
        g = GGUFFile.open(p)
        c2 = config_from_gguf(g)
        for f in ("layer_types", "moe_k", "moe_ffn", "moe_experts", "hidden_size", "mamba_num_heads", "mamba_head_dim",
                  "ssm_state_size", "n_groups", "num_key_value_heads", "head_dim", "shared_ffn"):
            assert getattr(c2, f) == getattr(cfg, f), f
        assert g.types() == {"F32": g.types()["F32"], "Q8_0": 3, "IQ2_XXS": 9, "NVFP4": 3}, g.types()
        assert g.canonical_names()["layers.0.mixer.out_proj.scale"] == "blk.0.ssm_out.scale" and not g.unmapped()
        if not KERNELS:
            return "config and types ok; engine skipped: Triton on a GPU, or TRITON_INTERPRET=1"
        from engine.models.nemotron_h.engine import NemotronHEngine
        from engine.models.nemotron_h.load import ExpertsRelu2
        c3, W, experts, timing, _ = load_gguf(p, DEV, log=None)
        assert timing["packed"] == 3 * 2 + 3 + 3 + 3, timing
        o = W["layers.0.mixer.out_proj.weight"]
        assert isinstance(o, Q.NVFP4Linear) and o.scale == np.float32(NVFP4_SCALE)
        eng = NemotronHEngine(c3, W, experts, device=DEV, max_len=256)
        res = eng.self_check(PROMPT, steps=8, log=None)
        assert res["ok"], res
        # the twin: every packed weight decoded to bf16 at load, the rest as the file holds it
        W2 = {}
        for k, v in want.items():
            if isinstance(k, str):
                W2[k] = v.float().to(DEV) if k.endswith(("bias", ".A", ".D", "dt_bias")) else v.to(torch.bfloat16).to(DEV)
        for li in c3.attention_layers:
            m = f"layers.{li}.mixer."
            W2[m + "qkv"] = torch.cat([W2[m + f"{x}_proj.weight"] for x in "qkv"]).contiguous()
        ex2 = {li: ExpertsRelu2(want[(li, "up")].to(torch.bfloat16).to(DEV).contiguous(),
                                want[(li, "down")].to(torch.bfloat16).to(DEV).contiguous()) for li in c3.moe_layers}
        twin = NemotronHEngine(c3, W2, ex2, device=DEV, max_len=256)
        a, b = eng.new_state(256), twin.new_state(256)
        la = eng.prefill(torch.tensor(PROMPT), a)[0].cpu()
        lb = twin.prefill(torch.tensor(PROMPT), b)[0].cpu()
        rel = ((la - lb).abs().max() / lb.abs().max()).item()
        assert rel < (0.05 if DEV == "cuda" else 0.2), rel
    return f"self-check {res['rows']} rows bit for bit; vs the bf16 twin rel {rel:.3f}"


def header_gguf(path: str, dims: tuple, ty: int = 0, alignment: int | None = None, data: int = 64) -> None:
    """A GGUF of one tensor written by hand (no gguf-py): the header, the table and `data` bytes."""
    import struct

    def s(x: str) -> bytes:
        return struct.pack("<Q", len(x)) + x.encode()
    kv = [] if alignment is None else [s("general.alignment") + struct.pack("<II", 4, alignment)]
    b = b"GGUF" + struct.pack("<IQQ", 3, 1, len(kv)) + b"".join(kv)
    b += s("t") + struct.pack(f"<I{len(dims)}Q", len(dims), *dims) + struct.pack("<IQ", ty, 0)
    with open(path, "wb") as f:
        f.write(b + bytes(-len(b) % 32 + data))


def test_a_header_whose_sizes_wrap_or_whose_alignment_is_not_a_power_of_two_is_refused():
    from engine.models.nemotron_h.gguf import GGUFFile
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "h.gguf")
        header_gguf(p, (4, 4))
        assert GGUFFile.open(p).tensors["t"].nbytes == 64
        # np.prod over these wraps to 0 and to 2 values: under the file's size, so they passed as not truncated
        for dims in ((2 ** 32, 2 ** 32), (3, 6148914691236517206)):
            header_gguf(p, dims)
            try:
                GGUFFile.open(p)
            except ValueError as e:
                assert "truncated" in str(e), e
            else:
                raise AssertionError(f"{dims} opened")
        for a in (0, 24):
            header_gguf(p, (4, 4), alignment=a)
            try:
                GGUFFile.open(p)
            except ValueError as e:
                assert "general.alignment" in str(e), e
            else:
                raise AssertionError(f"alignment {a} opened")
    return "2**32 x 2**32 and 3 x 6148914691236517206 truncated; alignment 0 and 24 refused"


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
