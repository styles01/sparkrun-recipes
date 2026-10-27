"""LiLiCorr: NVIDIA's block drafter for Nemotron 3.5 Bolt (nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr,
arXiv 2608.20530), as a draft head of the shared tree drafter (engine/methods/mtp_drafter.py).

The checkpoint (`LiLiCorrDraftModel`, ModelOpt NVFP4 W4A16, group 16) is a DFlash2-style backbone and a
two-layer candidate-lattice head over the target's own embedding and LM head:

  context   for every committed target token, the residual rows entering Bolt's layers 1, 18, 35, 51, 68
            and 85 (llama.cpp's `t_layer_inp`, `NemotronHEngine.set_taps`), concatenated [6 x 4096] ->
            fc (NVFP4) -> RMSNorm hidden_norm = g; each of the five layers' K = rope(k_norm(k_proj g)) and
            V = v_proj g at the token's position, kept in a window ring of 1,024 positions
  block     [anchor, MASK x 7] (mask 990) at positions n .. n+7, the target's embedding; per layer:
            RMSNorm, the grouped dynamic convolution's input side (kernel 2, group 16; dflash2._Conv),
            q/k/v, q_norm/k_norm, RoPE (theta 10,000, all 128 dims), attention of every block row over
            the context keys within its window (query - key < 1,024) and all 8 block rows, with a learned
            sink logit a head in the softmax's denominator (`attention_sink_bias`), o_proj, the output
            side of the convolution, the residual; the same around the SiLU MLP (12,288, NVFP4); the
            final norm; the target's LM head on rows 1..7
  lattice   per slot the top 8 candidates (descending), features [log p, p, log p - log p_top, rank / 7,
            rank == 0] -> LayerNorm(5) -> MLP 5 -> 1,024 -> 1,024; token states token_proj(embedding),
            pass states pass_hidden_proj(the slot's final row), slot and rank embeddings; two
            pre-norm transformer layers over the 56 nodes with a learned per-head bias by slot distance;
            output_norm; the anchor context_proj(g of the token before the anchor) -> anchor_norm;
            factor = silu(factor_input_proj([x, a, x * a])) -> L2-normalised out_head / in_head vectors;
            score(slot 1, j) = 8 <anchor_out_head(a), in_1j>, score(slot s, j | slot s-1, i) =
            8 <out_(s-1)i, in_sj>
  walk      greedy, left to right, on the GPU: slot 1's best start, then each slot's best given the
            previous choice; the draft is 7 tokens, each with the softmax of its row of scores as its
            confidence (llama.cpp's p_min reads the same)

Arithmetic and names follow vLLM's `LiLiCorrForCausalLM` (Apache-2.0, adapted from SGLang #37462) and
llama.cpp's `src/models/dflash.cpp` + `common/speculative.cpp` (MIT, #30506); the window and the sinks
follow llama.cpp's SWA mask and `ggml_soft_max_add_sinks`.

The draft reads its context from the engine's tap ring (`NhState.ring`) by position, not from the rows
the tree drafter hands over (those are the trunk's final rows, which the lookup's bookkeeping counts):
a draft anchored at n computes g, K and V for the committed rows it has not read yet within the window
(n - 1,023 .. n - 1) and drops its ring when the engine's rows below the length stop being this
sequence's (`tap_gen`: a restart, a cache restore). Rows it never read change a draft, never the text:
every drafted token is verified by the trunk, so a draft's arithmetic (bf16, torch) never reaches it.
"""

from __future__ import annotations

import json
import math
import os

import torch
import torch.nn.functional as F

from engine.drafters.dflash2 import _Conv, _apply_rope, _lin, _rms, _rope_tables

#: the files of a LiLiCorr checkpoint directory
FILES = ("config.json", "model.safetensors")


class LiLiCorrConfig:
    """The checkpoint's config.json, the fields the draft reads; anything else it asks for is refused."""

    def __init__(self, raw: dict):
        d = raw.get("dflash_config") or {}
        if "LiLiCorrDraftModel" not in (raw.get("architectures") or []) or not d.get("lilicorr_enabled"):
            raise ValueError(f"not a LiLiCorr draft: architectures {raw.get('architectures')}")
        if d.get("projector_type", "lilicorr") != "lilicorr" or d.get("causal", False):
            raise ValueError(f"LiLiCorr projector {d.get('projector_type')!r}, causal {d.get('causal')}: unsupported")
        if raw.get("hidden_act", "silu") != "silu" or raw.get("rope_scaling"):
            raise ValueError("LiLiCorr: silu MLP and plain RoPE only")
        self.hidden_size = int(raw["hidden_size"])
        self.num_hidden_layers = int(raw["num_hidden_layers"])
        self.num_attention_heads = int(raw["num_attention_heads"])
        self.num_key_value_heads = int(raw["num_key_value_heads"])
        self.head_dim = int(raw.get("head_dim") or self.hidden_size // self.num_attention_heads)
        self.intermediate_size = int(raw["intermediate_size"])
        self.rms_norm_eps = float(raw["rms_norm_eps"])
        self.rope_theta = float(raw.get("rope_theta", 10000.0))
        self.vocab_size = int(raw["vocab_size"])
        self.block_size = int(raw.get("block_size") or d["block_size"])
        self.mask_token_id = int(d["mask_token_id"])
        self.target_layer_ids = [int(x) for x in d["target_layer_ids"]]
        self.window = int(d.get("swa_window_size") or raw.get("sliding_window")) if d.get("use_swa", True) else 0
        self.sinks = bool(d.get("attention_sink_bias", raw.get("attention_sink_bias", False)))
        self.conv_kernel_size = int(d.get("conv_kernel_size") or 0)
        self.conv_group_size = int(d.get("conv_group_size") or 0)
        self.topk = int(d["lilicorr_candidate_topk"])
        self.lc_hidden = int(d.get("lilicorr_hidden_size") or self.hidden_size)
        self.lc_layers = int(d["lilicorr_num_layers"])
        self.lc_heads = int(d["lilicorr_num_heads"])
        self.lc_mlp = int(self.lc_hidden * float(d.get("lilicorr_mlp_ratio", 2.0)))
        self.factor_dim = int(d.get("lilicorr_factor_dim") or self.lc_hidden)
        self.vector_eps = float(d.get("lilicorr_vector_eps", 1e-4))
        self.logit_scale = float(d.get("lilicorr_logit_scale", 1.0))
        # Super VL's drafter carries its own LM head (NVFP4, the target's vocabulary); Bolt's borrows the target's
        self.own_head = bool(raw.get("has_own_lm_head", False))
        if self.own_head and int(raw.get("draft_vocab_size") or self.vocab_size) != self.vocab_size:
            raise ValueError("LiLiCorr with its own head over a draft vocabulary: unsupported")
        if not self.window:
            raise ValueError("LiLiCorr without a sliding window: the context ring needs one")
        if self.topk < 1 or self.topk > 16 or self.topk & (self.topk - 1):
            raise ValueError(f"LiLiCorr candidate_topk {self.topk}: a power of two up to 16")
        if bool(self.conv_kernel_size) != bool(self.conv_group_size):
            raise ValueError("LiLiCorr convolution: kernel and group size together")

    @property
    def slots(self) -> int:
        return self.block_size - 1

    @property
    def num_groups(self) -> int:
        return self.hidden_size // self.conv_group_size if self.conv_group_size else 0


def load_config(path: str) -> LiLiCorrConfig:
    with open(os.path.join(path, "config.json")) as f:
        return LiLiCorrConfig(json.load(f))


def missing_files(path: str | None) -> list[str]:
    if not path or not os.path.isdir(path):
        return [path or "(no path)"]
    return [f for f in FILES if not os.path.isfile(os.path.join(path, f))]


def load_weights(path: str, device: str = "cuda") -> dict:
    """The checkpoint's tensors: a ModelOpt NVFP4 linear (`weight` uint8, `weight_scale` e4m3,
    `weight_scale_2` fp32) as one `NVFP4Block` under its weight's name, every other tensor bf16."""
    from safetensors import safe_open

    from tools.nvfp4_linear import NVFP4Block
    raw = {}
    with safe_open(os.path.join(path, "model.safetensors"), framework="pt", device="cpu") as f:
        for k in f.keys():
            raw[k] = f.get_tensor(k)
    W = {}
    for k, t in raw.items():
        if k.endswith((".weight_scale", ".weight_scale_2", ".input_scale")):
            continue
        if t.dtype == torch.uint8:
            base = k[:-len("weight")]
            if base + "weight_scale" not in raw or base + "weight_scale_2" not in raw:
                raise ValueError(f"{k}: an NVFP4 weight without its scales")
            W[k] = NVFP4Block(t.to(device), raw[base + "weight_scale"].to(device), raw[base + "weight_scale_2"].float())
        else:
            W[k] = t.to(device=device, dtype=torch.bfloat16)
    return W


#: the largest activation the NVFP4 kernels take as it is: they multiply in fp16 (max 65,504)
F16_SAFE = 2.0 ** 15


def _lin_wide(x: torch.Tensor, w) -> torch.Tensor:
    """`_lin` for an input that can leave fp16's range (the taps, the MLPs' inputs: the draft's
    activations reach 1e5 where the trunk's normed ones never do): an NVFP4 weight's kernel multiplies
    in fp16, so a row whose largest value is past F16_SAFE is scaled down by a power of two first and
    the product scaled back (both exact); a bf16 weight runs as it is."""
    if isinstance(w, torch.Tensor):
        return _lin(x, w)
    s = torch.exp2(torch.ceil(torch.log2(x.abs().amax(-1, keepdim=True).float() / F16_SAFE)).clamp_min(0.0))
    return (_lin((x / s).to(x.dtype), w) * s).to(x.dtype)


def _linb(x: torch.Tensor, W: dict, name: str) -> torch.Tensor:
    """A linear layer of the checkpoint with its bias when it has one."""
    y = _lin(x, W[name + ".weight"])
    b = W.get(name + ".bias")
    return y if b is None else y + b


class LiLiCorrModel:
    """The draft's arithmetic over its weights `W`, the target's embedding `embed` [vocab, H] and LM head
    `head` (rows [M, H] -> logits [M, vocab] fp32). Knows nothing of the engine, so tests and tools can
    drive it."""

    def __init__(self, cfg: LiLiCorrConfig, W: dict, embed: torch.Tensor, head):
        self.cfg, self.W, self.embed, self.head = cfg, W, embed, head
        c = cfg
        self.convs = []
        for i in range(c.num_hidden_layers):
            p = f"layers.{i}"
            self.convs.append(None if not c.conv_kernel_size else (
                _Conv(W[f"{p}.attention_conv.base_kernel"], W[f"{p}.attention_conv.kernel_projection.weight"],
                      c.conv_kernel_size, c.num_groups, c.conv_group_size),
                _Conv(W[f"{p}.mlp_conv.base_kernel"], W[f"{p}.mlp_conv.kernel_projection.weight"],
                      c.conv_kernel_size, c.num_groups, c.conv_group_size)))
        # every layer's K and V of a context row in one product (the context is written a row at a time)
        self.kv_w = torch.cat([W[f"layers.{i}.self_attn.{x}_proj.weight"]
                               for i in range(c.num_hidden_layers) for x in "kv"]).contiguous()
        self.sink = [W[f"layers.{i}.self_attn.attention_sink_bias"].float() if c.sinks else None
                     for i in range(c.num_hidden_layers)]
        self.attn_bias = self._lattice_bias()
        T = c.topk
        self.rank_frac = (torch.arange(T, dtype=torch.float32) / max(T - 1, 1)).to(self.device)
        self.is_top1 = (torch.arange(T) == 0).float().to(self.device)
        h = c.lc_hidden
        fw = W["lilicorr.factor_input_proj.weight"]
        self.factor_w = (fw[:, :h].contiguous(), fw[:, h:2 * h].contiguous(), fw[:, 2 * h:].contiguous())

    @property
    def device(self) -> torch.device:
        return self.embed.device

    def _lattice_bias(self) -> torch.Tensor:
        """The head's attention bias [heads, nodes, nodes]: by the slot distance (offset block - 1) plus a
        same-slot term (vLLM's `_build_attention_bias`)."""
        c = self.cfg
        slot = torch.arange(c.slots, device=self.device).repeat_interleave(c.topk)
        rel = slot[:, None] - slot[None, :]
        bias = self.W["lilicorr.relative_slot_bias"][:, rel + c.block_size - 1]
        same = (slot[:, None] == slot[None, :]).to(bias.dtype)
        return (bias + same[None] * self.W["lilicorr.same_slot_bias"].view(-1, 1, 1)).contiguous()

    # ---- the context ---------------------------------------------------------------------------
    def context(self, taps: torch.Tensor, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Rows of the target's taps [n, taps, H] at `positions` -> (g [n, H], K, V [layers, kv heads, n,
        head_dim]): K normed and roped, as the block reads them."""
        c = self.cfg
        n = taps.shape[0]
        g = _rms(_lin_wide(taps.reshape(n, -1), self.W["fc.weight"]), self.W["hidden_norm.weight"], c.rms_norm_eps)
        kv = F.linear(g, self.kv_w).view(n, c.num_hidden_layers, 2, c.num_key_value_heads, c.head_dim)
        k = torch.stack([_rms(kv[:, i, 0], self.W[f"layers.{i}.self_attn.k_norm.weight"], c.rms_norm_eps)
                         for i in range(c.num_hidden_layers)])                  # [L, n, kv, D]
        cos, sin = _rope_tables(positions, c.head_dim, c.rope_theta, g.dtype)
        k = _apply_rope(k.transpose(1, 2), cos, sin)                            # [L, kv, n, D]
        v = kv[:, :, 1].permute(1, 2, 0, 3)
        return g, k, v

    # ---- the block -----------------------------------------------------------------------------
    def block(self, anchor: torch.Tensor, pos0: torch.Tensor, ck: torch.Tensor, cv: torch.Tensor,
              ctx_ok: torch.Tensor) -> torch.Tensor:
        """The block [anchor, MASK x (block - 1)] at positions pos0 .. pos0 + block - 1 over the context
        keys `ck`, `cv` [layers, kv heads, C, D] where `ctx_ok` [block, C] (bool) says which key each row
        sees. Returns the final normed rows [block, H]."""
        c = self.cfg
        B, nh, nkv, D = c.block_size, c.num_attention_heads, c.num_key_value_heads, c.head_dim
        dev = self.device
        ids = torch.full((B,), c.mask_token_id, dtype=torch.long, device=dev)
        ids[:1] = anchor
        h = self.embed.index_select(0, ids)
        bpos = torch.arange(B, device=dev)
        cos, sin = _rope_tables(pos0 + bpos, D, c.rope_theta, h.dtype)
        # the score mask: the context keys a row sees, then every row of the block, then the sink
        keep = torch.cat([ctx_ok, torch.ones(B, B, dtype=torch.bool, device=dev)], 1)
        neg = torch.zeros(keep.shape, dtype=torch.float32, device=dev).masked_fill_(~keep, float("-inf"))
        eps = c.rms_norm_eps
        for i in range(c.num_hidden_layers):
            p = f"layers.{i}"
            conv = self.convs[i]
            x = _rms(h, self.W[f"{p}.input_layernorm.weight"], eps)
            ak = None
            if conv is not None:
                x, ak = conv[0].prepare(x, bpos)
            q = _lin(x, self.W[f"{p}.self_attn.q_proj.weight"]).view(B, nh, D)
            k = _lin(x, self.W[f"{p}.self_attn.k_proj.weight"]).view(B, nkv, D)
            v = _lin(x, self.W[f"{p}.self_attn.v_proj.weight"]).view(B, nkv, D)
            q = _apply_rope(_rms(q, self.W[f"{p}.self_attn.q_norm.weight"], eps).transpose(0, 1)[None], cos, sin)[0]
            k = _apply_rope(_rms(k, self.W[f"{p}.self_attn.k_norm.weight"], eps).transpose(0, 1)[None], cos, sin)[0]
            keys = torch.cat([ck[i], k], 1)                                     # [kv, C + B, D]
            vals = torch.cat([cv[i], v.transpose(0, 1)], 1)
            o = self._attend(q, keys, vals, neg, self.sink[i])                  # [B, nh * D]
            o = _lin(o, self.W[f"{p}.self_attn.o_proj.weight"])
            if conv is not None:
                o = conv[0].finish(o, ak, bpos)
            h = h + o
            x = _rms(h, self.W[f"{p}.post_attention_layernorm.weight"], eps)
            mk = None
            if conv is not None:
                x, mk = conv[1].prepare(x, bpos)
            x = _lin_wide(F.silu(_lin_wide(x, self.W[f"{p}.mlp.gate_proj.weight"]))
                          * _lin_wide(x, self.W[f"{p}.mlp.up_proj.weight"]), self.W[f"{p}.mlp.down_proj.weight"])
            if conv is not None:
                x = conv[1].finish(x, mk, bpos)
            h = h + x
        return _rms(h, self.W["norm.weight"], eps)

    def _attend(self, q, keys, vals, neg, sink) -> torch.Tensor:
        """q [heads, B, D] over keys/vals [kv heads, K, D] (GQA), the additive mask `neg` [B, K], a sink
        logit a head (or none) in the softmax's denominator; fp32. -> [B, heads * D]."""
        nh, B, D = q.shape
        rep = nh // keys.shape[0]
        s = torch.einsum("grbd,gkd->grbk", q.float().view(-1, rep, B, D), keys.float()) / math.sqrt(D)
        s = s.reshape(nh, B, -1) + neg
        if sink is not None:
            s = torch.cat([s, sink.view(nh, 1, 1).expand(nh, B, 1)], -1)
        p = torch.softmax(s, -1)[..., :keys.shape[1]]
        o = torch.einsum("grbk,gkd->grbd", p.view(-1, rep, B, keys.shape[1]), vals.float())
        return o.reshape(nh, B, D).transpose(0, 1).reshape(B, nh * D).to(q.dtype)

    # ---- the lattice ---------------------------------------------------------------------------
    def candidates(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Each slot's top-k ids and their log-probabilities (descending) from its logits [slots, vocab]."""
        top, ids = torch.topk(logits.float(), self.cfg.topk, dim=-1)
        return ids, top - torch.logsumexp(logits.float(), -1, keepdim=True)

    def lattice(self, ids: torch.Tensor, logp: torch.Tensor, pass_hidden: torch.Tensor, anchor_g: torch.Tensor,
                anchor_ok: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
        """(start [k], pairs [slots - 1, k, k]) in logit units: vLLM's `LiLiCorrHead.forward` for one
        sequence. ids/logp [slots, k], pass_hidden [slots, H] (the block's final rows 1..), anchor_g [H]
        (g of the token before the anchor)."""
        c, W = self.cfg, self.W
        S, T, Hc = ids.shape[0], c.topk, c.lc_hidden
        eps = c.rms_norm_eps
        dt = W["lilicorr.slot_embedding"].dtype
        tok = self.embed.index_select(0, ids.reshape(-1)).to(dt)
        x = (_linb(tok, W, "lilicorr.token_proj") if c.lc_hidden != c.hidden_size else tok).view(S, T, Hc)
        x = x + _linb(pass_hidden.to(dt), W, "lilicorr.pass_hidden_proj").unsqueeze(1)
        lp = logp.float()
        feat = torch.stack([lp, lp.exp(), lp - lp.max(-1, keepdim=True).values,
                            self.rank_frac.expand_as(lp), self.is_top1.expand_as(lp)], -1).to(dt)
        feat = F.layer_norm(feat, (5,), W["lilicorr.feature_norm.weight"], W["lilicorr.feature_norm.bias"])
        x = x + _linb(F.silu(_linb(feat, W, "lilicorr.feature_mlp.up_proj")), W, "lilicorr.feature_mlp.down_proj")
        x = x + W["lilicorr.slot_embedding"][0, 0, :S] + W["lilicorr.rank_embedding"][0, 0]
        x = x.reshape(S * T, Hc)
        a = _linb(anchor_g.view(1, -1).to(dt), W, "lilicorr.context_proj") * anchor_ok
        n = S * T
        bias = self.attn_bias[:, :n, :n]
        hd = Hc // c.lc_heads
        for j in range(c.lc_layers):
            p = f"lilicorr.layers.{j}"
            y = _rms(x, W[f"{p}.attn_norm.weight"], eps)
            qkv = F.linear(y, W[f"{p}.attn.in_proj_weight"], W[f"{p}.attn.in_proj_bias"])
            q, k, v = (t.view(n, c.lc_heads, hd).transpose(0, 1) for t in qkv.chunk(3, -1))
            o = F.scaled_dot_product_attention(q[None], k[None], v[None], attn_mask=bias[None])[0]
            x = x + _linb(o.transpose(0, 1).reshape(n, Hc), W, f"{p}.attn.out_proj")
            y = _rms(x, W[f"{p}.mlp_norm.weight"], eps)
            y = F.silu(_linb(y, W, f"{p}.mlp.up_proj"))
            x = x + _lin_wide(y, W[f"{p}.mlp.down_proj.weight"]) + W[f"{p}.mlp.down_proj.bias"]
        x = _rms(x, W["lilicorr.output_norm.weight"], eps).view(S, T, Hc)
        a = _rms(a, W["lilicorr.anchor_norm.weight"], eps)
        ws, wa, wx = self.factor_w
        pre = F.linear(x, ws, W["lilicorr.factor_input_proj.bias"]) + F.linear(a, wa) + F.linear(x * a, wx)
        fh = F.silu(pre)
        out = F.normalize(_linb(fh, W, "lilicorr.out_head"), dim=-1, eps=c.vector_eps)
        inn = F.normalize(_linb(fh, W, "lilicorr.in_head"), dim=-1, eps=c.vector_eps)
        ao = F.normalize(_linb(a, W, "lilicorr.anchor_out_head"), dim=-1, eps=c.vector_eps)[0]
        start = (inn[0] @ ao).float() * c.logit_scale
        pairs = torch.matmul(out[:-1], inn[1:].transpose(-1, -2)).float() * c.logit_scale
        return start, pairs

    @staticmethod
    def walk(ids: torch.Tensor, start: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
        """The greedy walk on the device: [slots, 4] fp32 rows (token, its score's softmax, the row's
        second token, its softmax) -- the first slot from `start`, each next one from the row of `pairs`
        its predecessor picks."""
        S, T = ids.shape
        rows = [start]
        pick = start.argmax().view(1)
        picks = [pick]
        for s in range(S - 1):
            r = pairs[s].index_select(0, pick)[0]
            pick = r.argmax().view(1)
            rows.append(r)
            picks.append(pick)
        sc = torch.stack(rows)                                                  # [S, k]
        pr = torch.softmax(sc, -1)
        top2 = torch.topk(pr, 2, -1)
        tok = ids.gather(1, top2.indices)
        return torch.stack([tok[:, 0].float(), top2.values[:, 0], tok[:, 1].float(), top2.values[:, 1]], 1)

    def draft(self, anchor, pos0, ck, cv, ctx_ok, anchor_g, anchor_ok=1.0) -> torch.Tensor:
        """The block, the target's head on its rows 1.., the candidates, the lattice and the walk:
        [slots, 4] on the device (`walk`)."""
        hid = self.block(anchor, pos0, ck, cv, ctx_ok)
        ids, logp = self.candidates(self.head(hid[1:]))
        start, pairs = self.lattice(ids, logp, hid[1:], anchor_g, anchor_ok)
        return self.walk(ids, start, pairs)


class LiLiCorrHead:
    """LiLiCorr as a draft head of the shared tree drafter (`MTPTreeDrafter`): `depth` 7, the rows it
    is handed are counted (the drafter's bookkeeping) and the context is read from the serving engine's
    tap ring. `srv`: the NemotronH serving engine (`st`, `kv.length`, `fn` with the taps set).

    The draft (the block, the head, the lattice, the walk) is one CUDA graph over the head's rings and
    a staged input (the anchor, its position, the anchor row's slot, which context slots each block row
    sees), captured at the first draft (the load-time gate's); the context's rows run eagerly before it
    (their count varies)."""

    def __init__(self, srv, model: LiLiCorrModel, graph: bool = True):
        self.srv, self.m = srv, model
        c = model.cfg
        self.depth = c.slots
        self.W = c.window
        dev, dt = model.device, model.embed.dtype
        self.k = torch.zeros(c.num_hidden_layers, c.num_key_value_heads, self.W, c.head_dim, dtype=dt, device=dev)
        self.v = torch.zeros_like(self.k)
        self.g = torch.zeros(self.W, c.hidden_size, dtype=dt, device=dev)
        self.tag = torch.full((self.W,), -1, dtype=torch.long)       # the position a ring slot holds (host)
        self.fed = 0                                                  # every position below is in the ring
        self.gen = None
        self.st = type("LiLiCorrState", (), {"length": 0})()
        self.captures = 0
        self._out = None
        self._n = 0
        self.stats = {"drafts": 0, "context_rows": 0}
        # the staged input: int64 (anchor, position, anchor row's slot), fp32 (the anchor row is this
        # sequence's), bool [block, window] (the context slots a block row sees)
        pin = dev.type == "cuda"
        self._h = {"i": torch.zeros(3, dtype=torch.long, pin_memory=pin), "f": torch.zeros(1, pin_memory=pin),
                   "ok": torch.zeros(c.block_size, self.W, dtype=torch.bool, pin_memory=pin)}
        self._d = {k: torch.zeros_like(v, device=dev) for k, v in self._h.items()}
        self._copied = torch.cuda.Event() if pin else None
        self.graph = None if not (graph and pin) else False           # False: not captured yet

    # --- the drafter's interface ------------------------------------------------------------------
    def reset(self) -> None:
        self.st.length = 0

    def prime(self, rows, nexts) -> None:
        self.st.length += len(nexts)

    def views(self) -> list:
        return []

    def resume(self, n: int) -> None:
        self.st.length = int(n)

    def capture_all(self) -> None:
        pass

    @torch.no_grad()
    def launch(self, rows, nexts: list[int]) -> None:
        """The draft anchored at the round's anchor (`nexts[-1]`, at the engine's length n): the
        context brought up to n - 1, the block, the lattice and the walk, left on the device for
        `read`."""
        self._n = len(nexts)
        self._out = self.draft_at(int(nexts[-1]), int(self.srv.kv.length))

    def read(self) -> tuple[list, list]:
        self.st.length += self._n
        out = self._out.tolist()
        return [[int(r[0]), int(r[2])] for r in out], [[r[1], r[3]] for r in out]

    # --- the context and the draft ----------------------------------------------------------------
    def draft_at(self, anchor: int, n: int) -> torch.Tensor:
        """[slots, 4] on the device (`LiLiCorrModel.walk`) for `anchor` at position n."""
        self.feed(n)
        tag = self.tag
        ok = (tag >= max(n - self.W + 1, 0)) & (tag < n)              # a slot this draft may read at all
        # row i (position n + i) sees a context key at p when n + i - p < window
        pos = torch.arange(self.m.cfg.block_size)[:, None] + n
        prev = (n - 1) % self.W
        if self._copied is not None:
            self._copied.synchronize()                                # the last draft's copy has left
        h = self._h
        h["ok"].copy_(ok[None, :] & (pos - tag[None, :] < self.W))
        h["i"][0], h["i"][1], h["i"][2] = anchor, n, prev
        h["f"][0] = 1.0 if n > 0 and int(tag[prev]) == n - 1 else 0.0
        for k, t in h.items():
            self._d[k].copy_(t, non_blocking=True)
        if self._copied is not None:
            self._copied.record()
        self.stats["drafts"] += 1
        if self.graph is None:
            return self._draft()
        if self.graph is False:
            self._capture()
        self.graph.replay()
        return self._out

    def _draft(self) -> torch.Tensor:
        d = self._d
        return self.m.draft(d["i"][:1], d["i"][1], self.k, self.v, d["ok"], self.g.index_select(0, d["i"][2:])[0],
                            d["f"][0])

    def _capture(self) -> None:
        from engine import capture_lock
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                self._draft()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with capture_lock.LOCK, torch.cuda.graph(g):
            self._out = self._draft()
        torch.cuda.synchronize()
        self.graph = g
        self.captures += 1

    def feed(self, n: int) -> None:
        """g, K and V of the committed rows below n this ring does not hold yet, from the engine's tap
        ring: the rows within the window that the engine still has from this sequence."""
        st = self.srv.st
        if (id(st), st.tap_gen) != self.gen:
            self.gen = (id(st), st.tap_gen)
            self.tag.fill_(-1)
            self.fed = 0
        ring = st.ring
        R = ring.shape[0]
        a = max(self.fed, n - self.W + 1, st.tap_from, n - R, 0)
        if a < n:
            pos = torch.arange(a, n)
            dev = self.m.device
            g, k, v = self.m.context(ring.index_select(0, (pos % R).to(dev)), pos.to(dev))
            slot = (pos % self.W).to(dev)
            self.g.index_copy_(0, slot, g)
            self.k.index_copy_(2, slot, k)
            self.v.index_copy_(2, slot, v)
            self.tag[pos % self.W] = pos
            self.stats["context_rows"] += n - a
        self.fed = max(self.fed, n)


def build_head(srv, path: str, log=print) -> LiLiCorrHead:
    """The checkpoint at `path` over the serving engine's embedding and head (its taps set first:
    `NemotronHEngine.set_taps` before any state)."""
    cfg = load_config(path)
    eng = srv.fn
    if cfg.hidden_size != eng.cfg.hidden_size or cfg.vocab_size != eng.cfg.vocab_size:
        raise ValueError(f"LiLiCorr hidden {cfg.hidden_size} / vocab {cfg.vocab_size} against the target's "
                         f"{eng.cfg.hidden_size} / {eng.cfg.vocab_size}")
    if tuple(eng.taps) != tuple(cfg.target_layer_ids):
        raise ValueError(f"the engine taps {eng.taps}, the draft reads {cfg.target_layer_ids}")
    W = load_weights(path, str(eng.device))
    head = eng.head
    if cfg.own_head:
        if "lm_head.weight" not in W:
            raise ValueError("LiLiCorr config says has_own_lm_head, the checkpoint has no lm_head.weight")
        hw = W["lm_head.weight"]
        head = (lambda rows: F.linear(rows, hw).float()) if isinstance(hw, torch.Tensor) else \
            (lambda rows: _lin_wide(rows, hw).float())
    model = LiLiCorrModel(cfg, W, eng.W["embeddings.weight"], head)
    log(f"[lilicorr] {path}: {cfg.num_hidden_layers} layers, block {cfg.block_size}, window {cfg.window}, "
        f"taps {cfg.target_layer_ids}, lattice top {cfg.topk}, head {'its own' if cfg.own_head else 'the target'}'s")
    return LiLiCorrHead(srv, model)


def gate(eng, head: LiLiCorrHead, st, prompt: list[int], steps: int = 40, every: int = 4) -> dict:
    """The load-time agreement gate: the trunk's greedy decode of `steps` tokens after `prompt` (the
    engine's own decode step, taps on), then a draft at every `every`-th position against the tokens
    the trunk wrote: the mean run of drafted tokens equal to them (the tokens a greedy verify would
    keep)."""
    lg = eng.prefill(torch.tensor(prompt), st)
    seq = list(prompt) + [int(lg[0].argmax())]
    for _ in range(steps):
        eng.stage(seq[-1], st)
        out = eng.step(st)
        st.length += 1
        seq.append(int(out[0].argmax()))
    srv_st, head.srv = head.srv, type("GateSrv", (), {"st": st})()
    runs = []
    try:
        for t in range(len(prompt), len(prompt) + steps - head.depth, every):
            # the anchor seq[t] at position t; the trunk's next tokens seq[t + 1 ..]
            got = head.draft_at(seq[t], t)[:, 0].long().tolist()
            r = 0
            while r < len(got) and got[r] == seq[t + 1 + r]:
                r += 1
            runs.append(r)
    finally:
        head.srv = srv_st
        head.gen = None
    return {"positions": len(runs), "mean_run": round(sum(runs) / max(len(runs), 1), 3), "runs": runs}
