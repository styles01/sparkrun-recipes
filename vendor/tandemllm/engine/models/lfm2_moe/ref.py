"""The LFM2 MoE forward in torch, operation for operation (ENG-283): the model the board code is held to.

`Lfm2MoeRef` runs a block of rows through the model, where a block is a prompt chunk (a chain), one
decode token, or a draft tree in DFS pre-order with its parents. Every row sees the cache's rows
[0, length) and its own ancestors in the block; the short conv of a row reads the inputs of its two
predecessors on its own path (the state's last inputs where the path is shorter). So the row of a
verify block computes what decoding that row's path one token at a time computes, and `commit(path)`
keeps a path:

  conv state   the last `conv_L_cache - 1` inputs B * x along the kept path (copies of the block's
               own rows, nothing recomputed): the rollback of a short conv is exact by construction
  KV rows      the path's rows moved down to positions length.., the rest left past the length

The arithmetic follows `transformers.models.lfm2_moe.modeling_lfm2_moe` (eager attention) with the
same dtype casts: the RMS norm in fp32, cast back, then times plain `w`; the rotary tables cast to the
activations' dtype; softmax in fp32; the router's sigmoid in the logits' dtype with the expert bias
(fp32) added for the choice only, the chosen sigmoids renormalised with + 1e-6; each expert's output
times its weight, added in ascending expert id. The tests (tests/test_lfm2_moe_ref.py) hold it to
transformers on a tiny random-weight model in fp32.

Weights are a dict of canonical names (the checkpoint's without its `model.` prefix):

  embed_tokens.weight, embedding_norm.weight, lm_head.weight (absent: the embedding, tied)
  layers.L.operator_norm.weight, layers.L.ffn_norm.weight
  layers.L.conv.in_proj.weight [3 H, H], layers.L.conv.conv.weight [H, 1, K], layers.L.conv.out_proj.weight
  layers.L.self_attn.{q,k,v,out}_proj.weight, layers.L.self_attn.{q,k}_layernorm.weight
  layers.L.feed_forward.w1/w2/w3.weight                          (dense layers)
  layers.L.feed_forward.gate.weight [E, H], .expert_bias [E], .experts.E.w1/w2/w3.weight  (MoE layers)

`lin(x, w)` is the one place a weight is multiplied, so the board engine (engine.py) routes it to its
kernels and keeps the rest.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


@dataclass
class LfmState:
    """One sequence: the short conv's last inputs per conv layer [conv_L_cache - 1, H] (oldest first),
    K and V per attention layer [kv heads, cap, head_dim], the length."""
    cap: int
    length: int = 0
    conv: dict = field(default_factory=dict)
    k: dict = field(default_factory=dict)
    v: dict = field(default_factory=dict)
    pages: object = None        # the KV pool's paged set of the K and V rows (engine/kvpool.py), or None
    block: tuple | None = None  # the board engine's last verify block: (tokens, parents, paths, row buffers key)

    def reset_inplace(self) -> None:
        """A new sequence in the same buffers: the conv state zeroed, the length 0 (KV rows past the
        length are never read)."""
        self.length = 0
        for t in self.conv.values():
            t.zero_()


def tree_paths(parents: list[int]) -> list[list[int]]:
    """Each row's path from the block's root to itself (rows of the block, DFS pre-order)."""
    paths: list[list[int]] = []
    for i, p in enumerate(parents):
        if p >= i:
            raise ValueError(f"row {i}: parent {p} is not before it (DFS pre-order)")
        paths.append([i] if p < 0 else paths[p] + [i])
    return paths


def chain(n: int) -> list[int]:
    return list(range(-1, n - 1))


class Lfm2MoeRef:
    def __init__(self, cfg, W: dict, dtype=torch.float32, device="cpu", experts: dict | None = None):
        self.cfg, self.W, self.dtype, self.device = cfg, W, dtype, torch.device(device)
        # MoE layer -> its experts as loaded (engine/models/lfm2_moe/load.py); None: the W names
        self.experts = experts or {}
        D = cfg.head_dim
        self.inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, D, 2, dtype=torch.float32) / D))
        self.rec: dict = {}          # what a commit needs of the last block: each conv layer's B * x rows
        self._block = None           # (parents, paths) of the last block

    # ------------------------------------------------------------------------ helpers
    @staticmethod
    def pre(li: int) -> str:
        return f"layers.{li}."

    def w(self, name: str) -> torch.Tensor:
        return self.W[name]

    def head_weight(self):
        return self.W.get("lm_head.weight", self.W["embed_tokens.weight"])

    def lin(self, x: torch.Tensor, w) -> torch.Tensor:
        if not torch.is_tensor(w):
            # an EXL3Block (tools/exl3_linear: its torch reference on the CPU)
            return w.matmul(x.to(torch.bfloat16)).to(x.dtype if not w.out_fp32 else torch.float32)
        return F.linear(x, w)

    def rms(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        dt = x.dtype
        h = x.float()
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.cfg.rms_norm_eps)
        return w * h.to(dt)

    def rope(self, positions: torch.Tensor):
        """cos, sin [T, head_dim] in the activations' dtype (NEOX: the two halves share a frequency)."""
        f = positions.float()[:, None] * self.inv_freq[None, :].to(positions.device)
        emb = torch.cat([f, f], dim=-1)
        return emb.cos().to(self.dtype), emb.sin().to(self.dtype)

    @staticmethod
    def rotate_half(x):
        h = x.shape[-1] // 2
        return torch.cat([-x[..., h:], x[..., :h]], dim=-1)

    def new_state(self, cap: int) -> LfmState:
        c = self.cfg
        st = LfmState(cap=cap)
        for li in range(c.num_hidden_layers):
            if c.is_attention(li):
                st.k[li] = torch.zeros(c.num_key_value_heads, cap, c.head_dim, dtype=self.dtype, device=self.device)
                st.v[li] = torch.zeros_like(st.k[li])
            else:
                st.conv[li] = torch.zeros(c.conv_state_len, c.hidden_size, dtype=self.dtype, device=self.device)
        return st

    # ------------------------------------------------------------------------ the layers
    def conv(self, h: torch.Tensor, li: int, st: LfmState, paths: list[list[int]]) -> torch.Tensor:
        """The double-gated short conv of the block's rows [M, H]."""
        c = self.cfg
        p = self.pre(li) + "conv."
        H, S = c.hidden_size, c.conv_state_len
        bcx = self.lin(h, self.w(p + "in_proj.weight"))
        B, C, x = bcx[:, :H], bcx[:, H:2 * H], bcx[:, 2 * H:]
        bx = B * x
        self.rec[li] = bx
        wc = self.w(p + "conv.weight").reshape(H, -1)              # [H, K], tap K - 1 = the row itself
        state = st.conv[li]
        out = []
        for i, pa in enumerate(paths):
            # the row's window: the state's inputs, then the block's rows on its path, the last S + 1
            win = torch.cat([state, bx[pa]], 0)[-(S + 1):]           # [S + 1, H] oldest first
            out.append((win.to(wc.dtype) * wc.T).sum(0))
        y = C * torch.stack(out).to(h.dtype)
        return self.lin(y, self.w(p + "out_proj.weight"))

    def attention(self, h: torch.Tensor, li: int, st: LfmState, paths: list[list[int]],
                  positions: torch.Tensor) -> torch.Tensor:
        c = self.cfg
        p = self.pre(li) + "self_attn."
        M, D, Hq, Hk = h.shape[0], c.head_dim, c.num_attention_heads, c.num_key_value_heads
        L = st.length
        q = self.rms(self.lin(h, self.w(p + "q_proj.weight")).view(M, Hq, D), self.w(p + "q_layernorm.weight"))
        k = self.rms(self.lin(h, self.w(p + "k_proj.weight")).view(M, Hk, D), self.w(p + "k_layernorm.weight"))
        v = self.lin(h, self.w(p + "v_proj.weight")).view(M, Hk, D)
        cos, sin = self.rope(positions)
        cos, sin = cos[:, None, :], sin[:, None, :]
        q = q * cos + self.rotate_half(q) * sin
        k = k * cos + self.rotate_half(k) * sin
        # the block's keys and values at slots L.. (a verify block's sit past the length until a commit)
        st.k[li][:, L:L + M] = k.transpose(0, 1)
        st.v[li][:, L:L + M] = v.transpose(0, 1)
        scale = D ** -0.5
        out = torch.empty(M, Hq, D, dtype=h.dtype, device=h.device)
        for i, pa in enumerate(paths):
            slots = torch.cat([torch.arange(L, device=h.device), L + torch.tensor(pa, device=h.device)])
            ks = st.k[li][:, slots].repeat_interleave(c.gqa, dim=0)      # [Hq, n, D]
            vs = st.v[li][:, slots].repeat_interleave(c.gqa, dim=0)
            s = torch.einsum("hd,hnd->hn", q[i], ks) * scale
            a = torch.softmax(s, dim=-1, dtype=torch.float32).to(h.dtype)
            out[i] = torch.einsum("hn,hnd->hd", a, vs)
        return self.lin(out.reshape(M, Hq * D), self.w(p + "out_proj.weight"))

    def mlp(self, h: torch.Tensor, li: int) -> torch.Tensor:
        p = self.pre(li) + "feed_forward."
        return self.lin(F.silu(self.lin(h, self.w(p + "w1.weight"))) * self.lin(h, self.w(p + "w3.weight")),
                        self.w(p + "w2.weight"))

    def route(self, h: torch.Tensor, li: int):
        """(ids [M, k] int64, weights [M, k] in the activations' dtype)."""
        c = self.cfg
        p = self.pre(li) + "feed_forward."
        logits = self.lin(h, self.w(p + "gate.weight"))
        s = logits.sigmoid()
        if c.use_expert_bias:
            _, ids = torch.topk(s + self.w(p + "expert_bias"), k=c.num_experts_per_tok, dim=-1)
            wts = torch.gather(s, 1, ids).type_as(logits)
        else:
            wts, ids = torch.topk(s, k=c.num_experts_per_tok, dim=-1)
        wts = wts / (wts.sum(dim=-1, keepdim=True) + 1e-6)
        return ids, wts * c.routed_scaling_factor

    def expert(self, x: torch.Tensor, li: int, e: int) -> torch.Tensor:
        if li in self.experts:
            w1, w3, w2 = self.experts[li].expert(e)
        else:
            p = f"{self.pre(li)}feed_forward.experts.{e}."
            w1, w3, w2 = self.w(p + "w1.weight"), self.w(p + "w3.weight"), self.w(p + "w2.weight")
        return self.lin(F.silu(self.lin(x, w1)) * self.lin(x, w3), w2)

    def moe(self, h: torch.Tensor, li: int) -> torch.Tensor:
        ids, wts = self.route(h, li)
        out = torch.zeros_like(h)
        for e in sorted(set(ids.view(-1).tolist())):
            rows, slot = torch.where(ids == e)
            y = self.expert(h[rows], li, e) * wts[rows, slot, None]
            out.index_add_(0, rows, y.to(out.dtype))
        return out

    # ------------------------------------------------------------------------ a block
    @torch.no_grad()
    def rows(self, tokens: torch.Tensor, parents: list[int], st: LfmState) -> torch.Tensor:
        """Logits [M, vocab] (fp32) of a block at the state's length; nothing in `st` moves but the
        block's own KV rows past the length (`commit` keeps a path)."""
        c = self.cfg
        tokens = tokens.view(-1).to(self.device)
        M = tokens.numel()
        if st.length + M > st.cap:
            raise ValueError(f"{st.length + M} rows exceed the state's capacity {st.cap}")
        paths = tree_paths(parents)
        positions = st.length + torch.tensor([len(pa) - 1 for pa in paths], device=self.device)
        self.rec = {}
        self._block = (list(parents), paths)
        h = self.W["embed_tokens.weight"][tokens].to(self.dtype)
        for li in range(c.num_hidden_layers):
            p = self.pre(li)
            n = self.rms(h, self.w(p + "operator_norm.weight"))
            if c.is_attention(li):
                h = h + self.attention(n, li, st, paths, positions)
            else:
                h = h + self.conv(n, li, st, paths)
            n = self.rms(h, self.w(p + "ffn_norm.weight"))
            h = h + (self.moe(n, li) if c.is_moe(li) else self.mlp(n, li))
        h = self.rms(h, self.w("embedding_norm.weight"))
        return self.lin(h, self.head_weight()).float()

    @torch.no_grad()
    def commit(self, st: LfmState, path: list[int]) -> None:
        """Keep the rows `path` (row 0 first, each the next one's parent) of the last block."""
        parents, paths = self._block
        if path and (path[0] != 0 or any(parents[b] != a for a, b in zip(path, path[1:]))):
            raise ValueError(f"{path} is not a root path of the last block")
        L, n = st.length, len(path)
        idx = torch.tensor(path, dtype=torch.long, device=self.device)
        for li, bx in self.rec.items():
            S = self.cfg.conv_state_len
            st.conv[li].copy_(torch.cat([st.conv[li], bx[idx]], 0)[-S:])
        if any(r != j for j, r in enumerate(path)):
            src, dst = L + idx, L + torch.arange(n, device=self.device)
            for li in st.k:
                st.k[li][:, dst] = st.k[li][:, src]
                st.v[li][:, dst] = st.v[li][:, src]
        st.length = L + n
        self._block, self.rec = None, {}

    def prefill(self, tokens: torch.Tensor, st: LfmState) -> torch.Tensor:
        """A prompt chunk: logits [T, vocab] of every row; the state advances."""
        n = tokens.numel()
        lg = self.rows(tokens, chain(n), st)
        self.commit(st, list(range(n)))
        return lg

    def decode(self, token: int, st: LfmState) -> torch.Tensor:
        return self.prefill(torch.tensor([int(token)]), st)
