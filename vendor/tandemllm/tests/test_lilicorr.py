"""LiLiCorr (SPD-123) on tiny random models: the NemotronH engine's taps (a verify row's taps are the decode
step's, the ring after a commit is the ring after the decode steps, a prompt chunk's rows land at their
positions, the graphs write what the eager steps write, the self-check holds them), the drafter's
arithmetic against an fp32 reference written here from vLLM's `LiLiCorrHead.score` / `forward` and
llama.cpp's `src/models/dflash.cpp` (the block with its window, sinks and dynamic convolution, the lattice,
the walk), its NVFP4 checkpoint reader, and the head's context ring (incremental feed, the window, a new
sequence). On the GPU, or on the CPU under TRITON_INTERPRET=1 (the engine's kernels)."""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.models.nemotron_h import kernels as K  # noqa: E402
from engine.models.nemotron_h.config import config_from_dict  # noqa: E402
from engine.models.nemotron_h.load import ExpertsRelu2  # noqa: E402
from engine.models.nemotron_h.ref import chain  # noqa: E402
from tests.test_nemotron_h_engine import DEV, PROMPT, SKIP, TINY_B  # noqa: E402
from tests.test_nemotron_h_ref import random_weights  # noqa: E402

TAPS = (1, 3, 6)
WINDOW = 16


def _engine():
    """The tiny Lightning-like engine with taps at TAPS and a ring for a window of WINDOW rows."""
    from engine.models.nemotron_h.engine import NemotronHEngine
    cfg = config_from_dict(TINY_B, "tiny-taps")
    W0 = random_weights(cfg)
    f32 = ("e_score_correction_bias", "A_log", "dt_bias", ".D")
    W = {k: (v.float() if k.endswith(f32) else v.to(torch.bfloat16)).to(DEV) for k, v in W0.items()
         if ".experts." not in k or "shared_experts" in k}
    experts = {}
    for li in cfg.moe_layers:
        m = f"layers.{li}.mixer.experts."
        E = cfg.moe_experts[li]
        experts[li] = ExpertsRelu2(
            torch.stack([W0[m + f"{e}.up_proj.weight"] for e in range(E)]).to(torch.bfloat16).to(DEV).contiguous(),
            torch.stack([W0[m + f"{e}.down_proj.weight"] for e in range(E)]).to(torch.bfloat16).to(DEV).contiguous())
    for li in cfg.attention_layers:
        m = f"layers.{li}.mixer."
        W[m + "qkv"] = torch.cat([W[m + f"{x}_proj.weight"] for x in "qkv"]).contiguous()
    eng = NemotronHEngine(cfg, W, experts, device=DEV, max_len=256)
    eng.set_taps(TAPS, WINDOW)
    return cfg, eng


_E: dict = {}


def engine():
    if "e" not in _E:
        _E["e"] = _engine()
    return _E["e"]


def _decode(eng, st, toks):
    taps = []
    for t in toks:
        eng.stage(t, st)
        eng.step(st)
        st.length += 1
        taps.append(eng._rb[1]["taps"][0].clone())
    return torch.stack(taps)


def _snap(st):
    return [t.clone() for t in st.conv.values()] + [t.clone() for t in st.ssm.values()]


def _put(st, snap):
    for t, s in zip(list(st.conv.values()) + list(st.ssm.values()), snap):
        t.copy_(s)


# ----------------------------------------------------------------------------------- the engine's taps
def test_set_taps_refuses_late_and_bad_layers():
    if SKIP:
        return SKIP
    cfg, eng = engine()
    try:
        eng.set_taps(TAPS, WINDOW)
    except RuntimeError:
        pass
    else:
        raise AssertionError("set_taps after the row buffers exist must refuse")
    from engine.models.nemotron_h.engine import NemotronHEngine
    e2 = NemotronHEngine(cfg, eng.W, eng.experts, device=DEV, max_len=64)
    for bad in ((0, 3), (3, cfg.num_hidden_layers)):
        try:
            e2.set_taps(bad, WINDOW)
        except ValueError:
            continue
        raise AssertionError(f"taps {bad} must refuse")
    return "late and out-of-range taps refused"


def test_verify_taps_and_the_committed_ring_are_the_decode_steps():
    if SKIP:
        return SKIP
    cfg, eng = engine()
    st = eng.new_state(256)
    assert st.ring.shape == (WINDOW + 64, len(TAPS), cfg.hidden_size)
    eng.prefill(PROMPT, st)
    L = st.length
    s0, r0 = _snap(st), st.ring.clone()
    toks = [7, 19, 23, 44, 31, 60]
    want = _decode(eng, st, toks)
    ring_after = st.ring.clone()
    st.length = L
    _put(st, s0)
    st.ring.copy_(r0)
    eng.stage_rows(toks, chain(len(toks)), st)
    eng.rows_step(st, len(toks))
    assert torch.equal(eng._rb[len(toks)]["taps"][:len(toks)], want)
    eng.commit(st, list(range(len(toks))))
    assert torch.equal(st.ring, ring_after)
    # a moved path: rows 0 -> {1 -> 2, 3}, keep (0, 1, 3): the ring holds the path's taps at L .. L + 2
    st2 = eng.new_state(256)
    eng.prefill(PROMPT, st2)
    eng.verify([7, 19, 23, 44], [-1, 0, 1, 1], st2)
    eng.commit(st2, [0, 1, 3])
    alone = eng.new_state(256)
    eng.prefill(PROMPT, alone)
    t3 = _decode(eng, alone, [7, 19, 44])
    R = st2.ring.shape[0]
    for j in range(3):
        assert torch.equal(st2.ring[(L + j) % R], t3[j]), j
    return f"{len(toks)} verify rows' taps and the committed ring: the decode steps' bits"


def test_taps_are_the_residual_entering_the_layer():
    """A tap is the residual stream entering its layer: the embedding row plus every earlier layer's
    output (the reference forward's per-layer sums, near: the engine's kernels round as bf16)."""
    if SKIP:
        return SKIP
    cfg, eng = engine()
    st = eng.new_state(256)
    eng.prefill(PROMPT[:1], st)
    got = _decode(eng, st, [7])[0].float().cpu()
    # the residual before layer li from a hooked eager run of the same row
    seen = {}
    orig = K.add_rms

    def spy(h, d, w, eps, h_out=None, n_out=None):
        out = orig(h, d, w, eps, h_out, n_out)
        seen.setdefault("h", []).append(out[0].clone())
        return out
    K.add_rms = spy
    try:
        st2 = eng.new_state(256)
        eng.prefill(PROMPT[:1], st2)
        seen.clear()
        eng.stage(7, st2)
        eng.step(st2)
    finally:
        K.add_rms = orig
    hs = seen["h"]                                       # entering layer 0, 1, ..., then the final sum
    for k, li in enumerate(TAPS):
        assert torch.equal(got[k], hs[li][0].float().cpu()), li
    return "taps = h entering layers 1, 3, 6"


def test_prefill_lands_the_chunks_last_rows_in_the_ring():
    if SKIP:
        return SKIP
    cfg, eng = engine()
    g = torch.Generator().manual_seed(3)
    prompt = torch.randint(0, cfg.vocab_size, (100,), generator=g)
    a = eng.new_state(256)
    eng.prefill(prompt, a)
    b = eng.new_state(256)
    eng.prefill(prompt[:1], b)
    taps = _decode(eng, b, prompt[1:].tolist())          # rows 1 .. 99 by the decode step
    R = a.ring.shape[0]
    tol = 0.05 if DEV == "cuda" else 0.3
    for p in range(100 - R + 1, 100):
        x, y = a.ring[p % R].float(), taps[p - 1].float()
        assert (x - y).abs().max() <= tol * y.abs().max() + 1e-3, p
    return f"positions {100 - R + 1}..99 at p mod {R}, near the decode path"


def test_graphs_write_the_eager_taps():
    if SKIP or DEV != "cuda":
        return SKIP or "skipped: CUDA graphs need the GPU"
    from engine.models.nemotron_h.engine import DecodeGraph
    cfg, eng = engine()
    a, b = eng.new_state(256), eng.new_state(256)
    eng.prefill(PROMPT, a)
    eng.prefill(PROMPT, b)
    dg = DecodeGraph(eng, a)
    for t in (7, 19, 44):
        dg(t)
        _decode(eng, b, [t])
    assert torch.equal(a.ring, b.ring)
    vg, cg = {}, {}
    for _ in range(2):
        eng.verify([5, 6, 7, 8], [-1, 0, 1, 1], a, graphs=vg)
        eng.verify([5, 6, 7, 8], [-1, 0, 1, 1], b)
        eng.commit(a, [0, 1, 3], graphs=cg)
        eng.commit(b, [0, 1, 3])
        assert torch.equal(a.ring, b.ring)
    return "decode, verify and commit graphs: the eager ring"


def test_joint_decode_writes_each_ring():
    if SKIP:
        return SKIP
    cfg, eng = engine()
    states, alone = [eng.new_state(256) for _ in range(2)], [eng.new_state(256) for _ in range(2)]
    for st, al, n in zip(states, alone, (14, 9)):
        eng.prefill(PROMPT[:n], st)
        eng.prefill(PROMPT[:n], al)
    for tok in (7, 19):
        eng.decode_multi([(tok, st, f"j{j}") for j, st in enumerate(states)])
        for st, al in zip(states, alone):
            st.length += 1
            _decode(eng, al, [tok])
    for st, al in zip(states, alone):
        assert torch.equal(st.ring, al.ring)
    return "2 states: each ring its own decode steps'"


def test_self_check_holds_the_taps():
    if SKIP:
        return SKIP
    cfg, eng = engine()
    res = eng.self_check(PROMPT.tolist(), steps=6, log=None)
    assert res["ok"] and res["tap_diff"] is None, res
    return "PASS with tap_diff None"


def test_serving_moves_the_taps_generation():
    if SKIP:
        return SKIP
    from engine.models.nemotron_h.serving import NemotronHServing
    cfg, eng = engine()
    srv = NemotronHServing(eng, 256)
    g0 = srv.st.tap_gen
    srv.forward(PROMPT, 0)
    srv.kv.length = 10                                   # a restore: the rows below 10 may be another's
    assert srv.st.tap_from == 10 and srv.st.tap_gen == g0 + 1
    srv.kv.length = 0                                    # a new sequence
    assert srv.st.tap_from == 0 and srv.st.tap_gen == g0 + 2
    return "restore and restart move tap_gen"


# ----------------------------------------------------------------------------------- the drafter
LC_CFG = {
    "architectures": ["LiLiCorrDraftModel"], "model_type": "qwen3", "block_size": 8,
    "dflash_config": {"mask_token_id": 9, "target_layer_ids": list(TAPS), "use_swa": True, "swa_window_size": WINDOW,
                      "causal": False, "attention_sink_bias": True, "projector_type": "lilicorr",
                      "lilicorr_enabled": True, "lilicorr_candidate_topk": 4, "lilicorr_hidden_size": 64,
                      "lilicorr_num_layers": 2, "lilicorr_num_heads": 4, "lilicorr_mlp_ratio": 2.0,
                      "lilicorr_factor_dim": 32, "lilicorr_vector_eps": 1e-4, "lilicorr_logit_scale": 8.0,
                      "conv_kernel_size": 2, "conv_group_size": 16},
    "hidden_size": 128, "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 32,
    "intermediate_size": 256, "hidden_act": "silu", "rms_norm_eps": 1e-6, "vocab_size": 256,
    "rope_theta": 10000.0, "rope_scaling": None,
}
#: the linears the checkpoint stores as NVFP4 (K a multiple of 128): fc, the layers' MLPs, three of the head's
NVFP4 = ("fc.weight", "lilicorr.token_proj.weight", "lilicorr.pass_hidden_proj.weight", "lilicorr.context_proj.weight")


def _nvfp4(k: str) -> bool:
    return k in NVFP4 or (k.startswith("layers.") and ".mlp." in k and k.endswith(".weight"))


def lc_weights(seed: int = 1) -> dict:
    """fp32 weights of the checkpoint's layout (names as in model.safetensors)."""
    from engine.drafters.lilicorr import LiLiCorrConfig
    c = LiLiCorrConfig(LC_CFG)
    g = torch.Generator().manual_seed(seed)

    def r(*shape, s=0.1):
        return torch.randn(*shape, generator=g) * s
    H, D, Hc, T = c.hidden_size, c.head_dim, c.lc_hidden, c.topk
    W = {"fc.weight": r(H, len(TAPS) * H, s=0.05), "hidden_norm.weight": 1 + r(H), "norm.weight": 1 + r(H)}
    for i in range(c.num_hidden_layers):
        p = f"layers.{i}."
        W.update({p + "input_layernorm.weight": 1 + r(H), p + "post_attention_layernorm.weight": 1 + r(H),
                  p + "self_attn.q_proj.weight": r(c.num_attention_heads * D, H),
                  p + "self_attn.k_proj.weight": r(c.num_key_value_heads * D, H),
                  p + "self_attn.v_proj.weight": r(c.num_key_value_heads * D, H),
                  p + "self_attn.o_proj.weight": r(H, c.num_attention_heads * D),
                  p + "self_attn.q_norm.weight": 1 + r(D), p + "self_attn.k_norm.weight": 1 + r(D),
                  p + "self_attn.attention_sink_bias": r(c.num_attention_heads, s=1.0),
                  p + "mlp.gate_proj.weight": r(c.intermediate_size, H), p + "mlp.up_proj.weight": r(c.intermediate_size, H),
                  p + "mlp.down_proj.weight": r(H, c.intermediate_size, s=0.05)})
        for conv in ("attention_conv", "mlp_conv"):
            W[p + conv + ".base_kernel"] = r(2, c.conv_kernel_size, H, s=0.5)
            W[p + conv + ".kernel_projection.weight"] = r(2 * c.conv_kernel_size * c.num_groups, H)
    q = "lilicorr."
    W.update({q + "token_proj.weight": r(Hc, H), q + "token_proj.bias": r(Hc),
              q + "pass_hidden_proj.weight": r(Hc, H), q + "pass_hidden_proj.bias": r(Hc),
              q + "context_proj.weight": r(Hc, H), q + "context_proj.bias": r(Hc),
              q + "feature_norm.weight": 1 + r(5), q + "feature_norm.bias": r(5),
              q + "feature_mlp.up_proj.weight": r(Hc, 5, s=0.5), q + "feature_mlp.up_proj.bias": r(Hc),
              q + "feature_mlp.down_proj.weight": r(Hc, Hc), q + "feature_mlp.down_proj.bias": r(Hc),
              q + "slot_embedding": r(1, 1, c.slots, 1, Hc), q + "rank_embedding": r(1, 1, 1, T, Hc),
              q + "relative_slot_bias": r(c.lc_heads, 2 * c.block_size - 1, s=0.5), q + "same_slot_bias": r(c.lc_heads, s=0.5),
              q + "output_norm.weight": 1 + r(Hc), q + "anchor_norm.weight": 1 + r(Hc),
              q + "factor_input_proj.weight": r(Hc, 3 * Hc), q + "factor_input_proj.bias": r(Hc),
              q + "out_head.weight": r(c.factor_dim, Hc), q + "out_head.bias": r(c.factor_dim),
              q + "in_head.weight": r(c.factor_dim, Hc), q + "in_head.bias": r(c.factor_dim),
              q + "anchor_out_head.weight": r(c.factor_dim, Hc), q + "anchor_out_head.bias": r(c.factor_dim)})
    for j in range(c.lc_layers):
        p = f"lilicorr.layers.{j}."
        W.update({p + "attn_norm.weight": 1 + r(Hc), p + "mlp_norm.weight": 1 + r(Hc),
                  p + "attn.in_proj_weight": r(3 * Hc, Hc), p + "attn.in_proj_bias": r(3 * Hc),
                  p + "attn.out_proj.weight": r(Hc, Hc), p + "attn.out_proj.bias": r(Hc),
                  p + "mlp.up_proj.weight": r(2 * Hc, Hc), p + "mlp.up_proj.bias": r(2 * Hc),
                  p + "mlp.down_proj.weight": r(Hc, 2 * Hc), p + "mlp.down_proj.bias": r(Hc)})
    return W


def _rope(x, pos, theta):
    """Neox RoPE over all of x's last dim at integer positions `pos` (x [..., T, D])."""
    D = x.shape[-1]
    inv = 1.0 / theta ** (torch.arange(0, D, 2, dtype=torch.float64) / D)
    f = pos.double()[:, None] * inv[None]
    cos, sin = torch.cat([f, f], -1).cos().float(), torch.cat([f, f], -1).sin().float()
    a, b = x[..., :D // 2], x[..., D // 2:]
    return x * cos + torch.cat([-b, a], -1) * sin


def _rmsn(x, w, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def ref_block(W, c, embed, anchor, n, ctx_pos, ctx_taps):
    """The block in fp32 (llama.cpp's graph): context rows at `ctx_pos` with their taps; every query row
    sees the context keys within its window and the whole block; a sink logit a head."""
    B, nh, nkv, D = c.block_size, c.num_attention_heads, c.num_key_value_heads, c.head_dim
    gsz, taps = c.conv_group_size, c.conv_kernel_size
    g = _rmsn(ctx_taps.reshape(len(ctx_pos), -1) @ W["fc.weight"].T, W["hidden_norm.weight"])
    x = embed[torch.tensor([anchor] + [c.mask_token_id] * (B - 1))]
    qpos = torch.arange(n, n + B)

    def conv(h, coef, base):
        # out[t] = sum over taps of (coef[t, tap, group(ch)] + base[tap, ch]) * h[t - tap], inside the block
        out = torch.zeros_like(h)
        for t in range(B):
            for tap in range(taps):
                if t - tap >= 0:
                    w = coef[t, tap].repeat_interleave(gsz) + base[tap]
                    out[t] += w * h[t - tap]
        return out
    for i in range(c.num_hidden_layers):
        p = f"layers.{i}."
        xn = _rmsn(x, W[p + "input_layernorm.weight"])
        coef = (xn @ W[p + "attention_conv.kernel_projection.weight"].T).view(B, 2, taps, -1)
        base = W[p + "attention_conv.base_kernel"]
        xn = conv(xn, coef[:, 0], base[0])
        q = _rmsn((xn @ W[p + "self_attn.q_proj.weight"].T).view(B, nh, D), W[p + "self_attn.q_norm.weight"])
        k = _rmsn((xn @ W[p + "self_attn.k_proj.weight"].T).view(B, nkv, D), W[p + "self_attn.k_norm.weight"])
        v = (xn @ W[p + "self_attn.v_proj.weight"].T).view(B, nkv, D)
        ck = _rmsn((g @ W[p + "self_attn.k_proj.weight"].T).view(-1, nkv, D), W[p + "self_attn.k_norm.weight"])
        cv = (g @ W[p + "self_attn.v_proj.weight"].T).view(-1, nkv, D)
        q = _rope(q.transpose(0, 1), qpos, c.rope_theta)                       # [nh, B, D]
        keys = torch.cat([_rope(ck.transpose(0, 1), ctx_pos, c.rope_theta), _rope(k.transpose(0, 1), qpos, c.rope_theta)], 1)
        vals = torch.cat([cv.transpose(0, 1), v.transpose(0, 1)], 1)
        kpos = torch.cat([ctx_pos, qpos])
        is_ctx = torch.cat([torch.ones(len(ctx_pos), dtype=torch.bool), torch.zeros(B, dtype=torch.bool)])
        o = torch.zeros(B, nh, D)
        for hh in range(nh):
            kv = hh // (nh // nkv)
            for r in range(B):
                vis = ~is_ctx | (qpos[r] - kpos < c.window)
                s = (keys[kv][vis] @ q[hh, r]) / math.sqrt(D)
                s = torch.cat([s, W[p + "self_attn.attention_sink_bias"][hh].view(1)])
                pr = torch.softmax(s, 0)[:-1]
                o[r, hh] = pr @ vals[kv][vis]
        o = o.reshape(B, -1) @ W[p + "self_attn.o_proj.weight"].T
        o = conv(o, coef[:, 1], base[1])
        x = x + o
        xn = _rmsn(x, W[p + "post_attention_layernorm.weight"])
        coef = (xn @ W[p + "mlp_conv.kernel_projection.weight"].T).view(B, 2, taps, -1)
        base = W[p + "mlp_conv.base_kernel"]
        xn = conv(xn, coef[:, 0], base[0])
        y = (F.silu(xn @ W[p + "mlp.gate_proj.weight"].T) * (xn @ W[p + "mlp.up_proj.weight"].T)) @ W[p + "mlp.down_proj.weight"].T
        x = x + conv(y, coef[:, 1], base[1])
    return _rmsn(x, W["norm.weight"]), g


def ref_lattice(W, c, embed, ids, logp, pass_hidden, anchor_g):
    """vLLM's `LiLiCorrHead.score` and `forward` for one sequence, fp32, the concatenated factor input."""
    S, T, Hc = ids.shape[0], c.topk, c.lc_hidden
    q = "lilicorr."

    def lin(x, name):
        return x @ W[name + ".weight"].T + W[name + ".bias"]
    x = lin(embed[ids], q + "token_proj") + lin(pass_hidden, q + "pass_hidden_proj").unsqueeze(-2)
    rank = torch.arange(T, dtype=torch.float32) / (T - 1)
    feats = torch.stack([logp, logp.exp(), logp - logp.max(-1, keepdim=True).values,
                         rank.expand_as(logp), (torch.arange(T) == 0).float().expand_as(logp)], -1)
    feats = F.layer_norm(feats, (5,), W[q + "feature_norm.weight"], W[q + "feature_norm.bias"])
    x = x + lin(F.silu(lin(feats, q + "feature_mlp.up_proj")), q + "feature_mlp.down_proj")
    x = x + W[q + "slot_embedding"][0, 0, :S] + W[q + "rank_embedding"][0, 0]
    x = x.reshape(S * T, Hc)
    n = S * T
    bias = torch.zeros(c.lc_heads, n, n)
    for a in range(n):
        for b in range(n):
            sa, sb = a // T, b // T
            bias[:, a, b] = W[q + "relative_slot_bias"][:, sa - sb + c.block_size - 1] + (W[q + "same_slot_bias"] if sa == sb else 0)
    hd = Hc // c.lc_heads
    for j in range(c.lc_layers):
        p = f"{q}layers.{j}."
        y = _rmsn(x, W[p + "attn_norm.weight"])
        qkv = y @ W[p + "attn.in_proj_weight"].T + W[p + "attn.in_proj_bias"]
        qq, kk, vv = (t.view(n, c.lc_heads, hd).transpose(0, 1) for t in qkv.chunk(3, -1))
        att = torch.softmax(qq @ kk.transpose(-1, -2) / math.sqrt(hd) + bias, -1) @ vv
        x = x + lin(att.transpose(0, 1).reshape(n, Hc), p + "attn.out_proj")
        x = x + lin(F.silu(lin(_rmsn(x, W[p + "mlp_norm.weight"]), p + "mlp.up_proj")), p + "mlp.down_proj")
    x = _rmsn(x, W[q + "output_norm.weight"]).view(S, T, Hc)
    a = _rmsn(lin(anchor_g.view(1, -1), q + "context_proj"), W[q + "anchor_norm.weight"])[0]
    fh = F.silu(lin(torch.cat([x, a.expand_as(x), x * a], -1), q + "factor_input_proj"))
    out = F.normalize(lin(fh, q + "out_head"), dim=-1, eps=c.vector_eps)
    inn = F.normalize(lin(fh, q + "in_head"), dim=-1, eps=c.vector_eps)
    ao = F.normalize(lin(a, q + "anchor_out_head"), dim=-1, eps=c.vector_eps)
    start = c.logit_scale * (inn[0] * ao).sum(-1)
    pairs = c.logit_scale * torch.einsum("sif,sjf->sij", out[:-1], inn[1:])
    return start, pairs


def ref_walk(ids, start, pairs):
    """llama.cpp's host walk: slot 1's best start, then the best given the previous pick."""
    pick = int(start.argmax())
    toks = [int(ids[0, pick])]
    for s in range(pairs.shape[0]):
        pick = int(pairs[s, pick].argmax())
        toks.append(int(ids[s + 1, pick]))
    return toks


def _model(W, dtype, nvfp4: bool):
    from engine.drafters.lilicorr import LiLiCorrConfig, LiLiCorrModel
    from tools.nvfp4_linear import quantize_to_nvfp4
    c = LiLiCorrConfig(LC_CFG)
    g = torch.Generator().manual_seed(9)
    embed = torch.randn(c.vocab_size, c.hidden_size, generator=g)
    head_w = torch.randn(c.vocab_size, c.hidden_size, generator=g) * 0.3
    Wd, Wref = {}, dict(W)
    for k, v in W.items():
        if nvfp4 and _nvfp4(k):
            q = quantize_to_nvfp4(v.to(DEV))
            Wd[k] = q
            Wref[k] = q.dequant().float().cpu()
        else:
            Wd[k] = v.to(DEV, dtype)
    hw = head_w.to(DEV, dtype)
    model = LiLiCorrModel(c, Wd, embed.to(DEV, dtype), lambda x: F.linear(x, hw).float())
    return c, model, Wref, embed.to(dtype).float(), head_w.to(dtype).float()


def _ctx(c, n, seed=4):
    g = torch.Generator().manual_seed(seed)
    pos = torch.arange(max(0, n - c.window + 1), n)
    return pos, torch.randn(len(pos), len(TAPS), c.hidden_size, generator=g)


def test_block_and_lattice_match_the_fp32_reference():
    """The model in fp32 against the reference: the block's rows, the lattice's scores and the walk."""
    if SKIP:
        return SKIP
    torch.backends.cuda.matmul.allow_tf32 = False         # fp32 products as fp32 (the reference's)
    torch.backends.cudnn.allow_tf32 = False
    W = lc_weights()
    c, m, Wref, embed, head_w = _model(W, torch.float32, nvfp4=False)
    n = 23                                       # past the window: rows 8 .. 22 of the context are seen
    pos, taps = _ctx(c, n)
    want, g = ref_block(Wref, c, embed, 5, n, pos, taps)
    gd, k, v = m.context(taps.to(DEV), pos.to(DEV))
    assert torch.allclose(gd.cpu(), g, atol=1e-5, rtol=1e-4), (gd.cpu() - g).abs().max()
    ok = (n + torch.arange(c.block_size)[:, None] - pos[None, :]) < c.window
    got = m.block(torch.tensor([5], device=DEV), torch.tensor(n, device=DEV), k, v, ok.to(DEV))
    assert torch.allclose(got.cpu(), want, atol=2e-5, rtol=1e-4), (got.cpu() - want).abs().max()
    lg = want[1:] @ head_w.T
    ids, logp = m.candidates(lg.to(DEV))
    rlp, rid = torch.topk(lg, c.topk, -1)
    assert torch.equal(ids.cpu(), rid)
    rlp = rlp - torch.logsumexp(lg, -1, keepdim=True)
    start, pairs = m.lattice(ids, logp, want[1:].to(DEV), g[-1].to(DEV))
    rs, rp = ref_lattice(Wref, c, embed, rid, rlp, want[1:], g[-1])
    assert torch.allclose(start.cpu(), rs, atol=1e-4) and torch.allclose(pairs.cpu(), rp, atol=1e-4), \
        ((start.cpu() - rs).abs().max(), (pairs.cpu() - rp).abs().max())
    walk = m.walk(ids, start, pairs).cpu()
    assert walk[:, 0].long().tolist() == ref_walk(rid, rs, rp)
    return f"block within 2e-5, scores within 1e-4, walk {walk[:, 0].long().tolist()}"


def test_bf16_nvfp4_model_near_the_reference():
    """The served form (bf16, the NVFP4 linears on their kernel) against the fp32 reference on the
    dequantised weights: the block's rows and the scores near."""
    if SKIP:
        return SKIP
    W = lc_weights()
    c, m, Wref, embed, head_w = _model(W, torch.bfloat16, nvfp4=True)
    n = 12
    pos, taps = _ctx(c, n)
    want, g = ref_block(Wref, c, embed, 5, n, pos, taps)
    gd, k, v = m.context(taps.to(DEV, torch.bfloat16), pos.to(DEV))
    ok = (n + torch.arange(c.block_size)[:, None] - pos[None, :]) < c.window
    got = m.block(torch.tensor([5], device=DEV), torch.tensor(n, device=DEV), k, v, ok.to(DEV)).float().cpu()
    err = float((got - want).abs().max() / want.abs().max())
    assert err < 0.05, err
    ids, logp = m.candidates(F.linear(got.to(DEV)[1:].to(torch.bfloat16), head_w.to(DEV, torch.bfloat16)).float())
    start, pairs = m.lattice(ids, logp, got[1:].to(DEV, torch.bfloat16), gd[-1])
    rs, rp = ref_lattice(Wref, c, embed, ids.cpu(), logp.cpu(), got[1:], g[-1])
    e2 = float(max((start.cpu() - rs).abs().max(), (pairs.cpu() - rp).abs().max()))
    assert e2 < 0.3, e2                                 # scores span +-8 (cosines x 8)
    return f"block rel err {err:.4f}, scores abs err {e2:.4f} (logit scale {c.logit_scale})"


def test_checkpoint_reader_takes_modelopt_nvfp4():
    """A checkpoint written as ModelOpt writes NVFP4 W4A16 (uint8 codes, e4m3 group scales, fp32 tensor
    scale) reads back as NVFP4 blocks with the same values; the config parses; a file missing is named."""
    if SKIP:
        return SKIP
    from safetensors.torch import save_file

    from engine.drafters.lilicorr import load_config, load_weights, missing_files
    from tools.nvfp4_linear import quantize_to_nvfp4
    W = lc_weights()
    out = {}
    for k, v in W.items():
        if _nvfp4(k):
            q = quantize_to_nvfp4(v)
            base = k[:-len("weight")]
            out[k] = q.w.cpu()
            out[base + "weight_scale"] = q.s.cpu()
            out[base + "weight_scale_2"] = torch.tensor(q.s2, dtype=torch.float32)
        else:
            out[k] = v.to(torch.bfloat16)
    with tempfile.TemporaryDirectory() as d:
        assert missing_files(d) == ["config.json", "model.safetensors"]
        save_file(out, os.path.join(d, "model.safetensors"))
        json.dump(LC_CFG, open(os.path.join(d, "config.json"), "w"))
        assert missing_files(d) == []
        cfg = load_config(d)
        got = load_weights(d, DEV)
    assert cfg.target_layer_ids == list(TAPS) and cfg.slots == 7 and cfg.window == WINDOW
    fc = got["fc.weight"]
    assert type(fc).__name__ == "NVFP4Block" and "fc.weight_scale" not in got
    assert torch.equal(fc.dequant().cpu(), quantize_to_nvfp4(W["fc.weight"]).dequant().cpu())
    assert got["norm.weight"].dtype == torch.bfloat16
    return f"{sum(type(v).__name__ == 'NVFP4Block' for v in got.values())} NVFP4 blocks, the rest bf16"


def test_wide_activations_stay_finite_on_nvfp4():
    """The NVFP4 kernels multiply in fp16: a row past its range (the draft's MLP inputs reach 1e5) is
    scaled by a power of two around the product, finite and near the bf16 product; a row in range is
    the plain call's bits."""
    if SKIP or DEV != "cuda":
        return SKIP or "skipped: the NVFP4 kernels need the GPU"
    from engine.drafters.lilicorr import _lin, _lin_wide
    from tools.nvfp4_linear import quantize_to_nvfp4
    g = torch.Generator().manual_seed(2)
    w = quantize_to_nvfp4((torch.randn(256, 512, generator=g) * 0.02).to(DEV))
    x = torch.randn(8, 512, generator=g).to(DEV, torch.bfloat16)
    x[3, 7] = 3e5
    x[5] *= 1e3
    y = _lin_wide(x, w)
    ref = x.float() @ w.dequant().float().T
    assert torch.isfinite(y).all()
    assert float((y.float() - ref).abs().max() / ref.abs().max()) < 0.02
    assert torch.equal(y[[0, 1, 2, 4, 6, 7]], _lin(x[[0, 1, 2, 4, 6, 7]], w))
    return "rows past 2^15 scaled by a power of two; rows in range untouched"


class _Srv:
    def __init__(self, st):
        self.st = st


class _St:
    def __init__(self, ring, tap_from=0):
        self.ring, self.tap_from, self.tap_gen = ring, tap_from, 0


def test_head_feeds_its_ring_by_position_and_window():
    """The head's ring: a draft at n reads the committed rows n - window + 1 .. n - 1 from the engine's
    ring (positions modulo its rows), fed incrementally or at once alike; a new sequence (tap_gen) and
    rows before `tap_from` are not read."""
    if SKIP:
        return SKIP
    from engine.drafters.lilicorr import LiLiCorrHead
    W = lc_weights()
    c, m, _, _, _ = _model(W, torch.float32, nvfp4=False)
    R = WINDOW + 64
    g = torch.Generator().manual_seed(7)
    taps = torch.randn(200, len(TAPS), c.hidden_size, generator=g).to(DEV)
    ring = torch.zeros(R, len(TAPS), c.hidden_size, device=DEV)
    ring[torch.arange(200 - R, 200) % R] = taps[200 - R:]
    st = _St(ring)
    a = LiLiCorrHead(_Srv(st), m)
    a.draft_at(3, 150)                                   # an earlier draft: its rows stay in the ring
    got = a.draft_at(3, 190).clone()                     # (the graph's output buffer: the next draft's too)
    b = LiLiCorrHead(_Srv(st), m)
    once = b.draft_at(3, 190).clone()
    assert torch.allclose(got, once, atol=1e-5), (got - once).abs().max()
    # the same draft from the model directly: the window's rows only
    pos = torch.arange(190 - WINDOW + 1, 190)
    gd, k, v = m.context(taps[pos.to(DEV)], pos.to(DEV))
    ok = (190 + torch.arange(c.block_size)[:, None] - pos[None, :]) < WINDOW
    direct = m.draft(torch.tensor([3], device=DEV), torch.tensor(190, device=DEV), k, v, ok.to(DEV), gd[-1])
    assert torch.allclose(got, direct, atol=1e-5), (got - direct).abs().max()
    # a restore at 185: the rows below are not this sequence's; the draft reads 185 .. 189 only
    st.tap_from, st.tap_gen = 185, st.tap_gen + 1
    got2 = a.draft_at(3, 190).clone()
    pos = torch.arange(185, 190)
    gd, k, v = m.context(taps[pos.to(DEV)], pos.to(DEV))
    ok = (190 + torch.arange(c.block_size)[:, None] - pos[None, :]) < WINDOW
    direct2 = m.draft(torch.tensor([3], device=DEV), torch.tensor(190, device=DEV), k, v, ok.to(DEV), gd[-1])
    assert torch.allclose(got2, direct2, atol=1e-5)
    # 135 .. 149 for the first draft, 175 .. 189 (the window) for the second, 185 .. 189 after the restore
    assert a.stats["context_rows"] == 15 + 15 + 5, a.stats
    return f"walk {got[:, 0].long().tolist()}; after a restore {got2[:, 0].long().tolist()}"


def test_head_interface_counts_rows_and_reads_tokens():
    """What the tree drafter asks: `prime` / `launch` count the fed rows, `read` returns per depth the
    walk's token and the row's second with their softmaxes (descending), `resume` sets the count."""
    if SKIP:
        return SKIP
    from engine.drafters.lilicorr import LiLiCorrHead
    from types import SimpleNamespace
    W = lc_weights()
    c, m, _, _, _ = _model(W, torch.float32, nvfp4=False)
    ring = torch.randn(WINDOW + 64, len(TAPS), c.hidden_size, device=DEV)
    st = _St(ring)
    srv = SimpleNamespace(st=st, kv=SimpleNamespace(length=40))
    h = LiLiCorrHead(srv, m)
    h.prime(None, list(range(30)))
    h.launch(None, [1] * 9 + [5])
    tok, prob = h.read()
    assert h.st.length == 40 and h.depth == 7 and len(tok) == 7
    assert all(len(t) == 2 and p[0] >= p[1] for t, p in zip(tok, prob))
    h.resume(12)
    assert h.st.length == 12
    if DEV == "cuda":
        # the captured draft against the eager one on the same rings and input
        e = LiLiCorrHead(srv, m, graph=False)
        want = e.draft_at(5, 40).clone()
        got = h.draft_at(5, 40)
        assert h.captures == 1 and torch.allclose(got, want, atol=1e-5), (got - want).abs().max()
    return f"tokens {[t[0] for t in tok]}"


def test_serving_drafts_a_round_with_the_head_and_a_long_lookup():
    """The NemotronH adapter's tree drafter with LiLiCorr: a round after a prompt drafts the head's block
    beside a lookup of the longer minimum match; QSE_NH_LOOKUP=off drafts with the head alone; a side
    slot gets its own head over the same weights and slot 0's lookup."""
    if SKIP:
        return SKIP
    from engine.drafters.lilicorr import LiLiCorrConfig, LiLiCorrModel
    from engine.models.nemotron_h.serving import LOOK_MIN_ORDER_HEAD, NemotronHServing, lookup_drafter
    cfg, eng = engine()
    c = LiLiCorrConfig(LC_CFG)
    model = LiLiCorrModel(c, {k: v.to(DEV, torch.bfloat16) for k, v in lc_weights().items()}, eng.W["embeddings.weight"],
                          eng.head)
    srv = NemotronHServing(eng, 256)
    head = srv.add_lilicorr(model=model)
    quiet = lambda m: None                               # noqa: E731
    d = lookup_drafter(srv, "", None, log=quiet, head=head)
    assert d.head is head and d.ngram.min_order == LOOK_MIN_ORDER_HEAD
    srv.kv.length = 0
    lg = srv.forward(PROMPT, 0)
    d.reset()
    d.prime(PROMPT.tolist())
    d.sync(PROMPT.tolist(), srv.hidden_post_norm[0], 0)
    anchor = int(lg[0, -1].argmax())
    tree = d.propose_tree(PROMPT.tolist() + [anchor], 16)
    assert tree is not None and tree.tokens[0] == anchor and len(tree.tokens) == 1 + head.depth, tree.tokens
    # the prompt's rows within the window, from the engine's ring
    assert head.stats["context_rows"] == min(len(PROMPT), WINDOW - 1), head.stats
    e2, d2 = srv.side_slot(256, d)
    assert d2.head is not head and d2.head.m is model and d2.ngram.min_order == LOOK_MIN_ORDER_HEAD
    os.environ["QSE_NH_LOOKUP"] = "off"
    try:
        d3 = lookup_drafter(srv, "", None, log=quiet, head=head)
    finally:
        del os.environ["QSE_NH_LOOKUP"]
    assert d3.ngram is None and srv.side_slot(256, d3)[1].ngram is None
    return f"a round of {len(tree.tokens) - 1} drafted tokens; lookup min order {LOOK_MIN_ORDER_HEAD}"


def test_build_leaves_the_lookup_alone_and_says_why():
    """`serving.build` with LiLiCorr: missing files, or a draft that fails the load-time gate (random
    weights agree with no trunk), leave no head (the lookup drafts alone), the reason in the log and in
    the info /health reports; the engine's self-check still passes."""
    if SKIP:
        return SKIP
    from types import SimpleNamespace

    from safetensors.torch import save_file

    import engine.models.nemotron_h.load as L
    from engine.models.nemotron_h import serving as SV
    from tools.nvfp4_linear import quantize_to_nvfp4
    cfg, eng = engine()

    class Tok:
        def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
            return msgs[0]["content"]

        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [3 + ord(ch) % 200 for ch in text]}
    real = L.load_any
    L.load_any = lambda path, device, mtp=False, log=None: (SimpleNamespace(cfg=cfg, notes={}, has_mtp=False), eng.W,
                                                             eng.experts, {})
    logs = []
    try:
        srv, info = SV.build("tiny", 256, Tok(), log=logs.append, lilicorr="/nonexistent/lilicorr")
        assert info["self_check"]["ok"] and "lilicorr" not in srv.heads
        assert info["lilicorr"]["ok"] is False and "files missing" in info["lilicorr"]["why"], info["lilicorr"]
        assert any("the lookup drafts alone" in m for m in logs)
        assert SV._lc_info(info["lilicorr"])["why"].startswith("LiLiCorr files missing")
        with tempfile.TemporaryDirectory() as d:
            out = {}
            for k, v in lc_weights().items():
                if _nvfp4(k):
                    q = quantize_to_nvfp4(v)
                    out[k], out[k[:-len("weight")] + "weight_scale"] = q.w.cpu(), q.s.cpu()
                    out[k[:-len("weight")] + "weight_scale_2"] = torch.tensor(q.s2, dtype=torch.float32)
                else:
                    out[k] = v.to(torch.bfloat16)
            save_file(out, os.path.join(d, "model.safetensors"))
            json.dump(LC_CFG, open(os.path.join(d, "config.json"), "w"))
            logs.clear()
            srv, info = SV.build("tiny", 256, Tok(), log=logs.append, lilicorr=d)
        lc = info["lilicorr"]
        assert lc["ok"] is False and lc["gate"]["mean_run"] < SV.LILICORR_GATE and "gate" in lc["why"], lc
        assert "lilicorr" not in srv.heads and srv.fn.taps == TAPS
        assert any("[lilicorr] gate FAIL" in m for m in logs) and any("the lookup drafts alone" in m for m in logs)
    finally:
        L.load_any = real
    return f"missing files and a failed gate (mean run {lc['gate']['mean_run']}): no head, reason logged"


if __name__ == "__main__":
    fails = 0
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        try:
            print(f"  {name:<62} ok   {fn() or ''}")
        except AssertionError as e:
            fails += 1
            print(f"  {name:<62} FAIL {e}")
    print(f"{len(tests) - fails} passed" + (f", {fails} FAILED" if fails else ""))
    sys.exit(1 if fails else 0)
