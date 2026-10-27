"""The Flash-Next forward in plain torch: slow, exact, the reference every kernel is checked against.

It follows the official implementation (`transformers` 5.17, `models/qwen4_exp`) operation for
operation, on the weights of the served pack: an EXL3 projection is multiplied by its dequantised
weight (`EXL3Block.dequant()`, bf16 of the fp32 rotation product), a mixer by its int8 rows or its
fp16 weight (`pack.mixer_weight`), the n-gram rows as `ple.EXL3NGramTable` decodes them. Activations
stay in the compute dtype (bf16 when serving; fp32 in the tight CPU test against `transformers`);
norms, softmaxes, the router and the recurrent state run in fp32 as in the reference. Ported from
branch `flashnext` (1a6b91ec, 5ac0e35a, 36ef4e3c, ef246906, 87522636), where it read Mia's NVFP4
checkpoint.

One step of the model, per layer (hc = 4 residual streams of `hidden`):

    h [T, 4*H]
    h += PLE(h, tokens)                         (layer 1 only: hashed n-gram rows)
    x, h0, w = mixer_attn(h)                    x = mean_s(sigmoid-mix_s * norm(h)_s)  [T, H]
    h = h0 + GDN_or_attention(x) (x) w          inject into every stream with weight w[s]
    x, h0, w = mixer_mlp(h)
    h = h0 + MoE(x) (x) w
    ...
    logits = head(mixer_final(h))               no final norm: the last mixer collapses streams

Differences from the reference, all deliberate and all value-preserving:
  * top-k ties (router, QSA blocks) resolve to the lower index (a stable sort), so the choice
    never depends on the launch shape; the reference's `torch.topk` leaves ties unspecified.
  * batch size is 1 (the engine serves one sequence per state).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from engine.gdn import chunk_gated_delta_rule, conv_update, recurrent_gated_delta_rule
from engine.models.flashnext.config import FlashNextConfig
from engine.models.flashnext.ple import NGramHash
from engine.models.state import SectionState


# ------------------------------------------------------------------------------ primitives
def lin(x: torch.Tensor, w) -> torch.Tensor:
    """x @ w^T for a plain weight (cast to x's dtype), an EXL3 projection (its dequantised weight)
    or int8 mixer rows (`pack.Int8Rows`, fp32 products)."""
    if isinstance(w, torch.Tensor):
        return F.linear(x, w if w.dtype == x.dtype else w.to(x.dtype))
    if hasattr(w, "bf16_cached"):                      # tools.exl3_linear.EXL3Block
        wd = w.bf16_cached()
        return F.linear(x, wd if wd.dtype == x.dtype else wd.to(x.dtype))
    return w.matmul(x)


class ExpertBank:
    """The routed experts of one layer and projection: `dense` [E, N, K] (tests, a BF16
    checkpoint) or `blocks` {e: EXL3Block} (a pack). `weight(e)` is expert e's [N, K] weight."""

    def __init__(self, dense: torch.Tensor | None = None, blocks: dict | None = None):
        self.dense, self.blocks = dense, blocks

    def weight(self, e: int, dtype) -> torch.Tensor:
        if self.dense is not None:
            return self.dense[e].to(dtype)
        return self.blocks[e].dequant().to(dtype)


def rms(x: torch.Tensor, w: torch.Tensor | None, eps: float, group: int | None = None,
        plus_one: bool = True) -> torch.Tensor:
    """RMS norm in fp32, weight (1 + w) (`Qwen4ExpTextRMSNorm`), optionally per group."""
    h = x.float()
    if group is not None:
        h = h.reshape(*h.shape[:-1], -1, group)
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)
    if group is not None:
        h = h.flatten(-2)
    if w is not None:
        h = h * ((1.0 + w.float()) if plus_one else w.float())
    return h.to(x.dtype)


def topk_stable(x: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-k along the last dim; equal values resolve to the lower index."""
    v, i = torch.sort(x, dim=-1, descending=True, stable=True)
    return v[..., :k], i[..., :k]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    a, b = x.chunk(2, dim=-1)
    return torch.cat([-b, a], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x [..., T, D] (heads before T) or [T, heads, D] handled by the caller's broadcasting."""
    d = cos.shape[-1]
    xr, xp = x[..., :d], x[..., d:]
    return torch.cat([xr * cos + rotate_half(xr) * sin, xp], dim=-1)


MTP = -1     # the layer index the MTP layer's state and weights go by (weights under `mtp.layers.0.`)


# ------------------------------------------------------------------------------ state
@dataclass
class FNState:
    """Everything a continuation needs, for batch size 1."""
    length: int = 0
    k: dict = field(default_factory=dict)        # attn layer -> [kvh, L, D]
    v: dict = field(default_factory=dict)
    ik: dict = field(default_factory=dict)       # attn layer -> [L, index_head_dim] raw indexer keys
    conv: dict = field(default_factory=dict)     # gdn layer -> [1, conv_dim, W-1]
    S: dict = field(default_factory=dict)        # gdn layer -> [1, Hv, Dk, Dv] fp32
    ple_ctx: dict = field(default_factory=dict)  # ple layer -> [1, ngram_size-1] tokens
    ple_conv: dict = field(default_factory=dict) # ple layer -> [1, hc_hidden, (K-1)*dilation]
    cap: int = 0        # > 0: KV and indexer keys live in preallocated buffers of `cap` tokens
    pool: dict = field(default_factory=dict)     # attn layer -> [blocks, index_head_dim] pooled QSA keys (engine)
    pool_n: dict = field(default_factory=dict)   # attn layer -> blocks computed
    pages: object = None  # the KV pool's paged set of k, v and ik (engine/kvpool.py), or None: static buffers
    pool_ratio: int = 4
    record: bool = False  # the row-exact engine keeps the recurrent state after every row of a block
    rows: dict = field(default_factory=dict)   # ("S"|"conv"|"ctx"|"pconv", layer) -> [state after row t]
    # VIS-97: the request's images (engine/vision.py MMContext: their rows, the prompt's three-axis
    # rotary coordinates) and the rotary offset of the rows after the prompt (position = slot + delta);
    # the server sets both per request (FlashNextServing `mm`, `pos_delta`)
    mm: object = None
    delta: int = 0

    def reset_inplace(self) -> None:
        """A fresh sequence in the SAME tensors (a captured decode graph holds them)."""
        self.length = 0
        for d in (self.S, self.conv, self.ple_conv):
            for v in d.values():
                v.zero_()
        self.ple_ctx = {}
        self.rows = {}
        self.record = False
        self.pool_n = {}

    def clamp_pool(self) -> None:
        """Pooled QSA keys of blocks past the committed length are stale after a rewind."""
        for li in list(self.pool_n):
            self.pool_n[li] = min(self.pool_n[li], self.length // self.pool_ratio)

    def clone(self) -> "FNState":
        c = lambda d: {k: v.clone() for k, v in d.items()}   # noqa: E731
        return FNState(self.length, c(self.k), c(self.v), c(self.ik), c(self.conv), c(self.S),
                       c(self.ple_ctx), c(self.ple_conv), self.cap, c(self.pool), dict(self.pool_n),
                       pool_ratio=self.pool_ratio, mm=self.mm, delta=self.delta)


class FNModelState(SectionState):
    """FNState as a ModelState (engine/models/state.py): per GDN layer S and conv, per PLE layer
    its dilated conv state and its token context. The KV rows and the indexer keys are kept by
    length (the caches truncate them); the pooled QSA keys are a cache of those and are clamped to
    the length on restore, so blocks past it are recomputed."""

    def __init__(self, st: "FNState"):
        self.st = st

    def sections(self) -> dict:
        st = self.st
        out = {}
        for li in sorted(st.S):
            out[f"gdn.S.{li}"] = st.S[li]
        for li in sorted(st.conv):
            out[f"gdn.conv.{li}"] = st.conv[li]
        for li in sorted(st.ple_conv):
            out[f"ple.conv.{li}"] = st.ple_conv[li]
        for li in sorted(st.ple_ctx):
            out[f"ple.ctx.{li}"] = st.ple_ctx[li]
        return out

    def restore(self, snap) -> None:
        super().restore(snap)
        self.st.clamp_pool()


# ------------------------------------------------------------------------------ the model
class FlashNextRef:
    """`W` maps canonical names (language-model prefix removed) to tensors, EXL3Blocks, Int8Rows
    and ExpertBanks; `tables` maps a PLE layer index to its table backend."""

    def __init__(self, cfg: FlashNextConfig, W: dict, tables: dict, dtype=torch.bfloat16,
                 device="cpu"):
        self.cfg, self.W, self.tables = cfg, W, tables
        self.dtype, self.device = dtype, device
        self.hash = {}
        for li, pi in cfg.ple_layers.items():
            stored = {k: W[f"layers.{li}.ple.ple_embedding.{k}"] for k in
                      ("layer_multipliers", "ngram_heads_vocab_sizes", "ngram_heads_offsets")
                      if f"layers.{li}.ple.ple_embedding.{k}" in W}
            tab = tables.get(li)
            if hasattr(tab, "multipliers"):        # an EXL3 table carries the hash parameters
                stored = {"layer_multipliers": tab.multipliers, "ngram_heads_vocab_sizes": tab.sizes,
                          "ngram_heads_offsets": tab.offsets}
            self.hash[li] = NGramHash.from_config(cfg, pi, stored)
        self.route_trace = None         # a list to record (layer, expert ids [T, k]) per MoE call
        dim = cfg.rotary_dim
        self.inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))

    @staticmethod
    def pre(li: int) -> str:
        return "mtp.layers.0." if li == MTP else f"layers.{li}."

    # --- primitives (FlashNextEngine replaces these with kernels; the arithmetic contract is the same)
    def lin(self, x, w):
        return lin(x, w)

    def rms(self, x, w, eps, group=None, plus_one=True):
        return rms(x, w, eps, group=group, plus_one=plus_one)

    def delta_rule(self, q, k, v, g, beta, S):
        """The gated delta rule over the block: token by token for one row, chunked for more."""
        if q.shape[1] == 1:
            return recurrent_gated_delta_rule(q, k, v, g, beta, S.clone())
        return chunk_gated_delta_rule(q, k, v, g, beta, S)

    def experts(self, x, idx, wts, G, U, Dn):
        """Routed experts, as the reference: ascending expert id, bf16 index_add."""
        out = torch.zeros_like(x)
        for e in torch.unique(idx).tolist():
            rows, slot = torch.where(idx == e)
            xs = x[rows]
            h = F.silu(F.linear(xs, G.weight(e, x.dtype))) * F.linear(xs, U.weight(e, x.dtype))
            y = F.linear(h, Dn.weight(e, x.dtype)) * wts[rows, slot, None]
            out.index_add_(0, rows, y.to(out.dtype))
        return out

    # --- helpers
    def rope(self, positions: torch.Tensor):
        freqs = positions.float()[:, None].cpu() * self.inv_freq[None, :]
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().to(self.dtype).to(self.device), emb.sin().to(self.dtype).to(self.device)

    def rope_at(self, positions: torch.Tensor):
        """cos/sin at these positions (the engine gathers them from a device table)."""
        return self.rope(positions)

    def w(self, name: str):
        return self.W[name]

    # --- hyper-connections
    def mixer(self, h: torch.Tensor, p: str, combine: bool = True):
        c = self.cfg
        n = self.rms(h, self.w(p + ".hc_norm.weight"), c.rms_norm_eps, group=c.hidden_size)
        mix = F.silu(self.lin(n, self.w(p + ".input_mix_weight_down.weight")) / c.hc_count)
        mix = torch.sigmoid(self.lin(mix, self.w(p + ".input_mix_weight_up.weight")))
        mix = mix.unflatten(-1, (c.hc_count, c.hidden_size))
        x = (mix * n.unflatten(-1, (c.hc_count, c.hidden_size))).mean(dim=-2)
        if not combine:
            return x
        inj = 2 * torch.sigmoid(self.lin(n, self.w(p + ".block_inject_weight.weight")) / c.hc_count)
        return x, h, inj

    @staticmethod
    def inject(h0: torch.Tensor, out: torch.Tensor, inj: torch.Tensor) -> torch.Tensor:
        return h0 + (out.unsqueeze(-2) * inj.unsqueeze(-1)).flatten(-2)

    # --- PLE
    def ple_ids(self, tokens: torch.Tensor, li: int, st: FNState) -> torch.Tensor:
        """The table rows of `tokens` ([T, heads], on the host); advances the hash context."""
        hsh = self.hash[li]
        ctx = st.ple_ctx.get(li)
        if ctx is None:
            ctx = torch.full((1, hsh.context_len), self.cfg.eos_token_id, dtype=torch.long)
        ids = hsh.ids(tokens.view(1, -1).cpu(), ctx.cpu())[0]
        st.ple_ctx[li] = hsh.next_context(tokens.view(1, -1).cpu(), ctx.cpu())
        return ids

    def ple(self, h: torch.Tensor, tokens: torch.Tensor, li: int, st: FNState) -> torch.Tensor:
        ids = self.ple_ids(tokens, li, st)                                  # [T, heads]
        emb = self.tables[li].gather(ids, dtype=self.dtype).to(self.device).flatten(-2)  # [T, E]
        return self.ple_apply(h, emb, li, st)

    def ple_apply(self, h: torch.Tensor, emb: torch.Tensor, li: int, st: FNState) -> torch.Tensor:
        """The PLE branch from its gathered rows `emb` [T, E] (no host work: a graph can hold it)."""
        c = self.cfg
        p = self.pre(li) + "ple."
        key = self.rms(self.lin(emb, self.w(p + "key_proj.weight")), self.w(p + "norm_key.weight"),
                  c.rms_norm_eps, group=c.hidden_size).unflatten(-1, (c.hc_count, c.hidden_size))
        value = self.lin(emb, self.w(p + "value_proj.weight"))
        query = self.rms(h, self.w(p + "norm_query.weight"), c.rms_norm_eps,
                    group=c.hidden_size).unflatten(-1, (c.hc_count, c.hidden_size))
        gate = (key * query).sum(dim=-1, keepdim=True) / math.sqrt(c.hidden_size)
        gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        gv = torch.sigmoid(gate) * value.unsqueeze(-2)                      # [T, hc, H]
        gv_normed = self.rms(gv.flatten(-2), self.w(p + "norm_conv.weight"), c.rms_norm_eps,
                        group=c.hidden_size)
        gv = gv.flatten(-2)
        # dilated depthwise causal conv with a (K-1)*dilation state, then SiLU
        K, dil = c.ple_conv_kernel_size, c.ngram_size
        slen = (K - 1) * dil
        xin = gv_normed.T.unsqueeze(0)                                      # [1, C, T]
        prev = st.ple_conv.get(li)
        if prev is None:
            prev = torch.zeros(1, xin.shape[1], slen, dtype=xin.dtype, device=xin.device)
        joined = torch.cat([prev, xin], dim=-1)
        if li in st.ple_conv:
            st.ple_conv[li].copy_(joined[:, :, -slen:])
        else:
            st.ple_conv[li] = joined[:, :, -slen:].clone()
        wconv = self.w(p + "conv1d.weight")                                 # [C, 1, K]
        out = F.conv1d(joined, wconv.to(xin.dtype), None, groups=xin.shape[1], dilation=dil)
        out = F.silu(out[:, :, -xin.shape[-1]:])[0].T                        # [T, C]
        return gv + out

    # --- GDN
    def gdn(self, x: torch.Tensor, li: int, st: FNState) -> torch.Tensor:
        c = self.cfg
        p = self.pre(li) + "linear_attn."
        T = x.shape[0]
        qkv = self.lin(x, self.w(p + "in_proj_qkv.weight"))                      # [T, conv_dim]
        z = self.lin(x, self.w(p + "in_proj_z.weight")).view(T, -1, c.linear_value_head_dim)
        b = self.lin(x, self.w(p + "in_proj_b.weight"))
        a = self.lin(x, self.w(p + "in_proj_a.weight"))
        cs = st.conv.get(li)
        if cs is None:
            cs = torch.zeros(1, c.conv_dim, c.linear_conv_kernel_dim - 1, dtype=x.dtype, device=x.device)
            st.conv[li] = cs
        wconv = self.w(p + "conv1d.weight").reshape(c.conv_dim, -1).to(x.dtype)
        mixed = conv_update(qkv.T.unsqueeze(0), cs, wconv)[0].T              # [T, conv_dim]
        q, k, v = torch.split(mixed, [c.key_dim, c.key_dim, c.value_dim], dim=-1)
        q = q.reshape(1, T, -1, c.linear_key_head_dim)
        k = k.reshape(1, T, -1, c.linear_key_head_dim)
        v = v.reshape(1, T, -1, c.linear_value_head_dim)
        beta = torch.sigmoid(b).view(1, T, -1)
        g = (-self.w(p + "A_log").float().exp() * F.softplus(a.float() + self.w(p + "dt_bias"))).view(1, T, -1)
        if c.num_v_per_k > 1:
            q = q.repeat_interleave(c.num_v_per_k, dim=2)
            k = k.repeat_interleave(c.num_v_per_k, dim=2)
        S = st.S.get(li)
        if S is None:
            S = torch.zeros(1, c.linear_num_value_heads, c.linear_key_head_dim,
                            c.linear_value_head_dim, dtype=torch.float32, device=x.device)
        out, S = self.delta_rule(q, k, v, g, beta, S)
        if li in st.S and st.S[li] is not S:
            st.S[li].copy_(S)                  # in place: a captured decode graph holds this tensor
        else:
            st.S[li] = S
        out = out.reshape(-1, c.linear_value_head_dim)
        h = out.float()
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + c.rms_norm_eps)
        h = self.w(p + "norm.weight") * h.to(out.dtype)
        act = torch.sigmoid if c.output_gate_type == "sigmoid" else F.silu
        h = (h * act(z.reshape(-1, c.linear_value_head_dim).float())).to(out.dtype)
        return self.lin(h.reshape(T, -1), self.w(p + "out_proj.weight"))

    # --- attention + QSA
    def qsa_select(self, q_idx: torch.Tensor, raw_keys: torch.Tensor, cos_all, sin_all,
                   q_pos: int, li: int) -> torch.Tensor | None:
        """Visible token indices the indexer keeps for one query at absolute position `q_pos`,
        or None when every visible token is kept (the exact-dense case)."""
        c = self.cfg
        n_vis = q_pos + 1
        ratio = c.indexer_compress_ratio
        nblk = n_vis // ratio
        topk = c.indexer_budget // ratio
        if nblk <= topk:
            return None
        p = self.pre(li) + "self_attn.indexer."
        groups = raw_keys[: nblk * ratio].view(nblk, ratio, -1)
        pooled = groups.float().mean(dim=1).to(raw_keys.dtype)
        pooled = self.rms(pooled, self.w(p + "k_layernorm.weight"), c.rms_norm_eps)
        starts = torch.arange(nblk, device=raw_keys.device) * ratio
        pooled = apply_rope(pooled, cos_all[starts], sin_all[starts])
        scores = (q_idx.float() @ pooled.float().T)                          # [heads, nblk]
        scores = torch.relu(scores).sum(dim=0) / math.sqrt(c.indexer_head_dim)
        _, sel = topk_stable(scores, topk)
        tok = (sel.unsqueeze(-1) * ratio + torch.arange(ratio, device=sel.device)).flatten()
        tail = torch.arange(nblk * ratio, n_vis, device=sel.device)
        return torch.cat([tok, tail])

    def qsa_mask(self, iq: torch.Tensor, raw_keys: torch.Tensor, start: int, li: int) -> torch.Tensor | None:
        """The indexer's selection for T queries at positions start..start+T-1 as a [T, L] bool
        mask over the cached tokens, or None while every query still sees at most budget + tail
        tokens (exact dense attention). Same selection as `qsa_select` per query, all at once."""
        c = self.cfg
        ratio = c.indexer_compress_ratio
        topk = c.indexer_budget // ratio
        T = iq.shape[0]
        L = raw_keys.shape[0]
        dev = raw_keys.device
        qpos = torch.arange(start, start + T, device=dev)
        nblk = (qpos + 1) // ratio
        NB = int(nblk.max())
        if NB <= topk:
            return None
        p = self.pre(li) + "self_attn.indexer."
        pooled = raw_keys[: NB * ratio].view(NB, ratio, -1).float().mean(dim=1).to(raw_keys.dtype)
        pooled = self.rms(pooled, self.w(p + "k_layernorm.weight"), c.rms_norm_eps)
        cs, sn = self.rope_at(torch.arange(NB, device=dev) * ratio)
        pooled = apply_rope(pooled, cs, sn)
        scores = torch.einsum("thd,bd->thb", iq.float(), pooled.float())
        scores = torch.relu(scores).sum(dim=1) / math.sqrt(c.indexer_head_dim)      # [T, NB]
        blk = torch.arange(NB, device=dev)
        scores = scores.masked_fill(blk[None, :] >= nblk[:, None], float("-inf"))
        _, sel = topk_stable(scores, topk)                                          # [T, topk]
        tok = (sel.unsqueeze(-1) * ratio + torch.arange(ratio, device=dev)).flatten(1)
        mask = torch.zeros(T, L, dtype=torch.bool, device=dev)
        mask.scatter_(1, tok, True)
        pos = torch.arange(L, device=dev)[None, :]
        mask |= (pos >= (nblk * ratio)[:, None]) & (pos <= qpos[:, None])           # the tail
        dense = nblk <= topk
        mask[dense] = True
        return mask

    def attention(self, x: torch.Tensor, li: int, st: FNState, positions: torch.Tensor) -> torch.Tensor:
        c = self.cfg
        p = self.pre(li) + "self_attn."
        T = x.shape[0]
        D = c.head_dim
        qg = self.lin(x, self.w(p + "q_proj.weight")).view(T, -1, 2 * D)
        q, gate = qg[..., :D], qg[..., D:].reshape(T, -1)
        q = self.rms(q, self.w(p + "q_norm.weight"), c.rms_norm_eps)             # [T, Hq, D]
        k = self.rms(self.lin(x, self.w(p + "k_proj.weight")).view(T, -1, D), self.w(p + "k_norm.weight"), c.rms_norm_eps)
        v = self.lin(x, self.w(p + "v_proj.weight")).view(T, -1, D)
        cos, sin = self.rope(positions)                                      # [T, rot]
        q = apply_rope(q, cos[:, None], sin[:, None])
        k = apply_rope(k, cos[:, None], sin[:, None])
        start = st.length
        L = start + T
        if st.cap:
            if L > st.cap:
                raise ValueError(f"context {L} exceeds the state's capacity {st.cap}")
            if li not in st.k:
                st.k[li] = torch.empty(k.shape[1], st.cap, D, dtype=k.dtype, device=k.device)
                st.v[li] = torch.empty(v.shape[1], st.cap, D, dtype=v.dtype, device=v.device)
            st.k[li][:, start:L] = k.transpose(0, 1)
            st.v[li][:, start:L] = v.transpose(0, 1)
            kc, vc = st.k[li][:, :L], st.v[li][:, :L]
        else:
            kc = torch.cat([st.k[li], k.transpose(0, 1)], dim=1) if li in st.k else k.transpose(0, 1).contiguous()
            vc = torch.cat([st.v[li], v.transpose(0, 1)], dim=1) if li in st.v else v.transpose(0, 1).contiguous()
            st.k[li], st.v[li] = kc, vc
        # QSA indexer: raw keys always cached; selection only matters past the budget
        sel_mask = None
        if c.has_indexer:
            ip = p + "indexer."
            qk = self.lin(x, self.w(ip + "index_qk_proj.weight"))
            nh, hd = c.indexer_n_heads, c.indexer_head_dim
            iq = qk[:, : nh * hd].view(T, nh, hd)
            ikey = qk[:, nh * hd:]
            iq = self.rms(iq, self.w(ip + "q_layernorm.weight"), c.rms_norm_eps)
            iq = apply_rope(iq, cos[:, None], sin[:, None])
            if st.cap:
                if li not in st.ik:
                    st.ik[li] = torch.empty(st.cap, ikey.shape[1], dtype=ikey.dtype, device=ikey.device)
                st.ik[li][start:L] = ikey
                raw = st.ik[li][:L]
            else:
                st.ik[li] = torch.cat([st.ik[li], ikey], dim=0) if li in st.ik else ikey.clone()
                raw = st.ik[li]
            sel_mask = self.qsa_mask(iq, raw, start, li)
        rep = c.num_attention_heads // c.num_key_value_heads
        kk = kc.repeat_interleave(rep, dim=0)                                # [Hq, L, D]
        vv = vc.repeat_interleave(rep, dim=0)
        qh = q.transpose(0, 1)                                               # [Hq, T, D]
        outs = []
        # queries in blocks of 256: the scores of a 2k-row prefill chunk against a 64k context
        # would otherwise be 24 x 2k x 64k values; each query row is computed the same either way
        for q0 in range(0, T, 256):
            q1 = min(q0 + 256, T)
            Lb = start + q1                                                  # keys this block can see
            scores = torch.matmul(qh[:, q0:q1], kk[:, :Lb].transpose(1, 2)) * (D ** -0.5)
            qpos = torch.arange(start + q0, start + q1, device=x.device)
            mask = torch.arange(Lb, device=x.device)[None, :] <= qpos[:, None]
            if sel_mask is not None:
                mask &= sel_mask[q0:q1, :Lb]
            scores = scores.masked_fill(~mask[None], torch.finfo(scores.dtype).min)
            probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
            outs.append(torch.matmul(probs, vv[:, :Lb]))
        out = torch.cat(outs, dim=1).transpose(0, 1).reshape(T, -1)        # [T, Hq*D]
        out = out * torch.sigmoid(gate)
        return self.lin(out, self.w(p + "o_proj.weight"))

    # --- MoE
    def route(self, x: torch.Tensor, li: int):
        c = self.cfg
        logits = self.lin(x, self.w(self.pre(li) + "mlp.gate.weight"))
        probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
        top, idx = topk_stable(probs, c.num_experts_per_tok)
        if c.norm_topk_prob:
            top = top / top.sum(dim=-1, keepdim=True)
        if self.route_trace is not None:
            self.route_trace.append((li, idx.detach()))
        return top.to(logits.dtype), idx

    def moe(self, x: torch.Tensor, li: int) -> torch.Tensor:
        p = self.pre(li) + "mlp."
        wts, idx = self.route(x, li)
        G, U, Dn = self.w(p + "experts.gate_proj"), self.w(p + "experts.up_proj"), self.w(p + "experts.down_proj")
        out = self.experts(x, idx, wts, G, U, Dn)
        sp = p + "shared_expert."
        sh = self.lin(F.silu(self.lin(x, self.w(sp + "gate_proj.weight"))) * self.lin(x, self.w(sp + "up_proj.weight")),
                 self.w(sp + "down_proj.weight"))
        sh = torch.sigmoid(self.lin(x, self.w(p + "shared_expert_gate.weight"))) * sh
        return out + sh

    # --- whole model
    def embed(self, tokens: torch.Tensor) -> torch.Tensor:
        e = self.w("embed_tokens.weight")[tokens.to(self.w("embed_tokens.weight").device)]
        return e.to(self.dtype).to(self.device)

    def layer(self, h: torch.Tensor, li: int, tokens: torch.Tensor, st: FNState,
              positions: torch.Tensor) -> torch.Tensor:
        if li in self.hash:
            h = h + self.ple(h, tokens, li, st)
        x, h0, inj = self.mixer(h, self.pre(li) + "attn_hyper_connection")
        if li != MTP and self.cfg.is_linear(li):
            out = self.gdn(x, li, st)
        else:
            out = self.attention(x, li, st, positions)
        h = self.inject(h0, out, inj)
        x, h0, inj = self.mixer(h, self.pre(li) + "mlp_hyper_connection")
        return self.inject(h0, self.moe(x, li), inj)

    @torch.no_grad()
    def forward(self, tokens: torch.Tensor, st: FNState, layers=None, trace: list | None = None,
                head: str = "all") -> torch.Tensor | None:
        """tokens [T] -> logits [T, vocab] (fp32); advances `st` by T. `head`: "all" rows, the
        "last" row only (a prefill's final chunk), or "none" (a prefill chunk whose logits nobody
        reads: the 1.27 GB head is not read)."""
        tokens = tokens.view(-1)
        T = tokens.shape[0]
        positions = torch.arange(st.length, st.length + T, device=self.device)
        h = self.embed(tokens).repeat(1, self.cfg.hc_count)
        for li in (layers if layers is not None else range(self.cfg.num_hidden_layers)):
            h = self.layer(h, li, tokens, st, positions)
            if trace is not None:
                trace.append(h)
        st.length += T
        self.stack = h                  # the pre-final-mixer streams: the MTP layer's input
        if head == "none":
            return None
        x = self.mixer(h[-1:] if head == "last" else h, "hyper_connection_mixer", combine=False)
        return self.lin(x, self.w("lm_head.weight")).float()

    # --- MTP
    @torch.no_grad()
    def mtp(self, stack: torch.Tensor, next_tokens: torch.Tensor, st: FNState, head=None) -> tuple[torch.Tensor, torch.Tensor]:
        """One MTP step over T positions: `stack` [T, hc*H] is the target's (or the previous MTP
        step's) pre-final-mixer streams at positions st.length.., `next_tokens` [T] the tokens
        that follow them. Returns (logits [T, vocab] for the token after next, the MTP layer's
        own pre-mixer streams for the next draft step) and advances the MTP state `st`.

        The input fusion follows the vLLM and SGLang Qwen4Exp MTP (they agree): one RMS norm
        (1 + w) over all hc*H values of the streams, `fc_hidden` shared by every stream, plus
        `fc_embedding(norm(embed(token)))` added to every stream; then one full-attention layer
        with QSA, MoE and hyper-connections, its own final mixer, and the shared head."""
        c = self.cfg
        next_tokens = next_tokens.view(-1)
        T = next_tokens.shape[0]
        e = self.rms(self.embed(next_tokens), self.w("mtp.pre_fc_norm_embedding.weight"), c.rms_norm_eps)
        e = self.lin(e, self.w("mtp.fc_embedding.weight"))
        hs = self.rms(stack, self.w("mtp.pre_fc_norm_hidden.weight"), c.rms_norm_eps)
        hs = self.lin(hs.view(T, c.hc_count, c.hidden_size), self.w("mtp.fc_hidden.weight"))
        h = (hs + e.unsqueeze(-2)).flatten(-2)
        positions = torch.arange(st.length, st.length + T, device=self.device)
        h = self.layer(h, MTP, next_tokens, st, positions)
        st.length += T
        x = self.mixer(h, "mtp.hyper_connection_mixer", combine=False)
        if head == "x":                   # the final mixer's output, for a caller that owns the head
            return x, h
        if head is not None:              # a drafter's own head (e.g. over a hot vocabulary)
            return head(x), h
        return self.lin(x, self.w("lm_head.weight")).float(), h
