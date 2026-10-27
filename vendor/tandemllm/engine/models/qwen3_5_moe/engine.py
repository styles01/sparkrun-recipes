"""The Qwen3.5 MoE trunk on the board (ENG-383, ENG-391): the reference forward with its weight reads on
the kernels, a row step -- one decode token, or a verify block of up to 32 draft rows -- that reads its
inputs from device buffers so a CUDA graph holds it, and the prompt on the prefill kernels.

The machinery is Flash-Next's (engine/models/flashnext/engine.py), whose GDN, attention-projection and
expert kernels serve this family as they are: the staging of a row's tokens, positions, cache slots,
ancestors and GDN conv windows; the GDN rows with the commit folded into the next call; the expert
bank with the shared expert as its last expert; the decode, verify, commit and MTP chain graphs. What
this class changes is what Flash-Next has and this family has not -- one residual stream with RMS
pre-norms instead of four streams and hyper-connection mixers, dense attention at every context length
instead of the sparse indexer -- and the MTP layer's input fusion:

  norms           tools/fn_kernels.hc_norm (one group of H: the reference's RMS norm, weight 1 + w)
  residual + norm tools/fn_kernels.inject_norm / combine_inject_norm with one stream and a zero inject
                  logit: inj = 2 sigmoid(0) = 1 exactly, so h = bf16(h0 + out) and the next norm in one
                  launch (the MoE's combine in the same launch)
  attention       tools/fn_kernels.attn_prep (q/k norms, partial rotary, the cache writes), then this
                  package's `kernels.attn_rows` over the row's visible keys in 512-key programs; the
                  context class (`qsa_class`) is the program count, a decode graph each
  prompt chunks   the same projections, GDN and expert kernels as Flash-Next's prefill; attention by
                  SDPA over the cache with the lower-right causal mask

Every part of a row is a row-invariant kernel, so a verified row is the decode step's row bit for bit
and greedy speculation writes the plain greedy text (the self-check of the expert kernels gates it).
"""

from __future__ import annotations

import numpy as np
import torch

from engine.models.flashnext.engine import (CAPTURE_ROWS, DENSE_MAX, CommitGraph, DecodeGraph,  # noqa: F401
                                            FlashNextEngine, MTPChainGraph, VerifyGraph)
from engine.models.flashnext.ref import MTP
from engine.models.qwen3_5_moe import kernels as AK
from engine.models.qwen3_5_moe.ref import Qwen35MoeRef


class Qwen35MoeEngine(FlashNextEngine):

    #: the tapped layers (`set_taps`) and the ring's rows
    taps: tuple = ()
    ring_rows = 0

    def __init__(self, cfg, W: dict, banks: dict, device="cuda", max_len: int = 8192):
        super().__init__(cfg, W, {}, banks, device=device, max_len=max_len)
        self._zd: dict = {}
        self._tapcol: dict = {}

    # ------------------------------------------------------------------------- taps (a block drafter's context)
    def set_taps(self, layers, window: int) -> None:
        """Keep the residual rows entering `layers` (a block drafter's context, DFlash2): every state
        gets a ring of `window` + CAPTURE_ROWS rows by position; a decode step writes its row, a commit
        its path's rows, a prompt chunk its last rows. Before any row buffers or graphs exist."""
        if self._rb:
            raise RuntimeError("set_taps after the row buffers exist: their graphs would not write the taps")
        layers = tuple(int(x) for x in layers)
        if any(not 1 <= li < self.cfg.num_hidden_layers for li in layers):
            raise ValueError(f"taps {layers}: layers 1..{self.cfg.num_hidden_layers - 1}")
        self.taps = layers
        self._tapcol = {li: k for k, li in enumerate(layers)}
        self.ring_rows = int(window) + CAPTURE_ROWS

    def new_state(self, cap: int | None = None, paged: bool = False):
        st = super().new_state(cap, paged)
        st.ring = (torch.zeros(self.ring_rows, len(self.taps), self.cfg.hidden_size, dtype=self.dtype,
                               device=self.device) if self.taps else None)
        st.tap_from, st.tap_gen = 0, 0
        return st

    def reset(self, st) -> None:
        super().reset(st)
        st.tap_from, st.tap_gen = 0, getattr(st, "tap_gen", 0) + 1

    def build_pool(self, st, layers=None) -> None:
        """No pooled indexer keys: this family's attention is dense (Flash-Next's pool is its sparse
        indexer's; a cache restore and the MTP head's resume call this)."""
        if self.cfg.has_indexer:
            super().build_pool(st, layers)

    def _rows_buffers(self, M: int, key=None) -> dict:
        rb = super()._rows_buffers(M, key)
        if self.taps and "taps" not in rb:
            rb["taps"] = torch.zeros(M, len(self.taps), self.cfg.hidden_size, dtype=self.dtype, device=self.device)
        return rb

    def stage(self, token: int, st) -> None:
        """Host work of one decode step: the token, its position, slot and attention control, and its
        row in the taps ring (`pidx`, the indexer's field, which this family does not use)."""
        rb = self._rows_buffers(1)
        rb["copied"].synchronize()
        h, L = rb["h"], st.length
        h["tok"].numpy()[0] = token
        h["pos"].numpy()[0] = L + st.delta
        h["slot"].numpy()[0] = L
        h["anc"].numpy()[0, 0] = L
        self._stage_qsa(h, L, 0, np.array([L]), h["anc"].numpy(), 1, st.delta)
        if self.taps:
            h["pidx"].numpy()[0] = L % self.ring_rows
        rb["dbuf"].copy_(rb["hbuf"], non_blocking=True)
        rb["copied"].record()

    def commit_dev(self, st, key) -> None:
        super().commit_dev(st, key)
        if self.taps and st.ring is not None:
            d = self._commit_bufs(key)["d"]
            src = self._rb[key[0]]["taps"].index_select(0, d["idx"])
            st.ring.index_copy_(0, d["dst"] % self.ring_rows, src)

    # the reference's pieces this family uses (FlashNextEngine's base is Flash-Next's reference)
    norm = Qwen35MoeRef.norm
    layer = Qwen35MoeRef.layer
    mtp = Qwen35MoeRef.mtp

    # ------------------------------------------------------------------------- primitives
    def lin(self, x, w):
        from engine.models.qwen3_5_moe.load import SplitGroup
        if isinstance(w, SplitGroup):
            return torch.cat([super(Qwen35MoeEngine, self).lin(x, b) for b in w.blocks], dim=-1)
        return super().lin(x, w)

    def _zero(self, M: int) -> torch.Tensor:
        """The inject logits of M rows: zeros (fp32 [M, 1]), so the inject is the plain residual add."""
        z = self._zd.get(M)
        if z is None:
            z = self._zd[M] = torch.zeros(M, 1, dtype=torch.float32, device=self.device)
        return z

    def qsa_class(self, last: int) -> int:
        """The attention class of a call whose last position is `last`: the 512-key programs that cover
        its visible keys (`kernels.class_of`); a decode, verify and MTP graph is captured per class."""
        return AK.class_of(last + 1, self.max_len)

    def _site_norm(self, li: int):
        if li >= self.cfg.num_hidden_layers:
            return self.W["norm.weight"]
        return self.W[self.pre(li) + "input_layernorm.weight"]

    # ------------------------------------------------------------------------- the trunk
    def trunk(self, h, tokens, st, positions, step: bool = False, pre: bool = False):
        """Every layer: the mixer on the normed rows, the residual add and the post-norm in one launch,
        the MoE, its combine, the residual add and the next layer's norm (the final norm after the last)
        in one launch. Returns the post-final-norm rows n [M, H] (also `self.stack`: the MTP input)."""
        from tools import fn_kernels as K
        c = self.cfg
        H, eps = c.hidden_size, c.rms_norm_eps
        M = h.shape[0]
        zd = self._zero(M)
        n = K.hc_norm(h, self._site_norm(0), H, eps)
        for li in range(c.num_hidden_layers):
            p = self.pre(li)
            k = self._tapcol.get(li)
            if k is not None:
                if step:
                    self._cur["taps"][:, k].copy_(h)
                elif st.ring is not None:
                    R = self.ring_rows
                    j0 = max(0, M - R)
                    rdst = torch.arange(st.length + j0, st.length + M, device=h.device) % R
                    st.ring[:, k].index_copy_(0, rdst, h[j0:])
            if c.is_linear(li):
                out = self._gdn1(n, li, st) if step else (self._gdn_pre(n, li, st) if pre else self.gdn(n, li, st))
            else:
                out = self._attn1(n, li, st) if step else (self._attn_pre(n, li, st) if pre else
                                                           self.attention(n, li, st, positions))
            h, n = K.inject_norm(h, out, zd, 0, self.W[p + "post_attention_layernorm.weight"], 1, eps)
            nxt = self._site_norm(li + 1)
            if pre:
                h, n = K.inject_norm(h, self._moe_pre(n, li), zd, 0, nxt, 1, eps)
            elif M <= self.MOE_ROWS:
                yd, ids, wts = self.moe_parts(n, li)
                if step and not self._write:
                    self.rec[("ids", li)] = ids
                h, n = K.combine_inject_norm(h, yd, ids, wts, zd, 0, nxt, 1, eps)
            else:
                h, n = K.inject_norm(h, self.moe(n, li), zd, 0, nxt, 1, eps)
        self.stack = n
        return n

    def final(self, n):
        """The head over the post-final-norm rows: fp32 logits."""
        return self.lin(n, self.W["lm_head.weight"]).float()

    def rows_step(self, st, M: int, write: bool = False, cls: int = 0) -> torch.Tensor:
        """M rows from the device buffers (`stage` / `stage_rows` first) through the trunk and the head
        (FlashNextEngine.rows_step's contract: logits [M, vocab] fp32 in a static buffer, the greedy
        tokens in rows["next"] and its pinned copy; `write`: a decode step, the states advance)."""
        from tools import fn_kernels as K
        rb = self._rows_buffers(M)
        self._cur, self._write, self._cls = rb, write, cls
        self.rec = {}
        h = self.W["embed_tokens.weight"].index_select(0, rb["d"]["tok"])
        n = self.trunk(h, None, st, None, step=True)
        if DENSE_MAX:
            K.dense(n, self.W["lm_head.weight"], out=rb["logits"])
        else:
            from tools.exl3_linear import exl3_matmul
            exl3_matmul(n, self.W["lm_head.weight"], out=rb["logits"])
        torch.argmax(rb["logits"], dim=-1, out=rb["next"])
        rb["next_h"].copy_(rb["next"], non_blocking=True)
        if write and self.taps and st.ring is not None:
            st.ring.index_copy_(0, rb["d"]["pidx"], rb["taps"])
        if self.fold:
            self._pend_n.zero_()
        self.rec["stack"] = self.stack
        return rb["logits"]

    def verify_multi(self, items: list) -> list:
        raise NotImplementedError("joint verify passes are not built for the Qwen3.5 MoE trunk yet")

    # ------------------------------------------------------------------------- attention
    def _attn1_core(self, qkv, iqk, li, st):
        """Attention of the rows past q|k|v: the prep launch (norms, partial rotary at each row's
        position, K and V into the cache at each row's slot), the row attention over each row's
        visible keys at the call's class, the output gate."""
        from tools import fn_kernels as K
        c, d = self.cfg, self._cur["d"]
        p = self.pre(li) + "self_attn."
        q = K.attn_prep(qkv, None, d["pos"], self.W[p + "q_norm.weight"], self.W[p + "k_norm.weight"],
                        self.rope_cos, self.rope_sin, st.k[li], st.v[li], None, c.num_attention_heads,
                        c.num_key_value_heads, 0, c.rms_norm_eps, slot=d["slot"])
        o = AK.attn_rows(q, st.k[li], st.v[li], d["actl"], d["anc"], self._cls, bufs=self._cur["attn"])
        return K.gate_mul(o, qkv, c.head_dim)

    def _attn_pre(self, x, li, st):
        """Attention of a prompt chunk: q|k|v, the prep kernel (the chunk's K and V into the cache at
        their positions), SDPA over the cache's rows [0, L + T) with the lower-right causal mask (query
        blocks of 1,024 rows), the output gate, o_proj."""
        import torch.nn.functional as F
        from torch.nn.attention.bias import causal_lower_right

        from tools import fn_kernels as K
        c, P = self.cfg, self._pre
        p = self.pre(li) + "self_attn."
        T, L = x.shape[0], st.length
        D, Hq, Hk = c.head_dim, c.num_attention_heads, c.num_key_value_heads
        qkv = self.lin(x, self.W[p + "qkv"])
        q = K.attn_prep(qkv, None, P["rpos"], self.W[p + "q_norm.weight"], self.W[p + "k_norm.weight"],
                        P["cos"], P["sin"], st.k[li], st.v[li], None, Hq, Hk, 0, c.rms_norm_eps, slot=P["pos"])
        o = torch.empty(T, Hq, D, dtype=self.dtype, device=x.device)
        rep = Hq // Hk
        for q0 in range(0, T, 1024):
            q1 = min(q0 + 1024, T)
            k = st.k[li][:, :L + q1].unsqueeze(0)
            v = st.v[li][:, :L + q1].unsqueeze(0)
            qh = q[q0:q1].transpose(0, 1).unsqueeze(0)
            oo = F.scaled_dot_product_attention(qh, k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1),
                                                attn_mask=causal_lower_right(q1 - q0, L + q1))
            o[q0:q1] = oo[0].transpose(0, 1)
        y = K.gate_mul(o, qkv, D)
        return self.lin(y, self.W[p + "o_proj.weight"])

    # ------------------------------------------------------------------------- MTP
    def _mtp_in(self, tok_emb, stack_in):
        """The MTP layer's input: fc([norm_e(embed) ; norm_h(stack)])."""
        from tools import fn_kernels as K
        c = self.cfg
        H, eps = c.hidden_size, c.rms_norm_eps
        e = K.hc_norm(tok_emb.contiguous(), self.W["mtp.pre_fc_norm_embedding.weight"], H, eps)
        hs = K.hc_norm(stack_in.contiguous(), self.W["mtp.pre_fc_norm_hidden.weight"], H, eps)
        return self.lin(torch.cat([e, hs], dim=-1), self.W["mtp.fc.weight"])

    def mtp_rows(self, st_m, key, stack_in: torch.Tensor, cls: int = 0):
        """The MTP layer over the rows of buffers `key` (`stage_mtp`), from `stack_in` [M, H] (the trunk's
        post-norm rows, or the previous step's y): the input fusion, its attention on the MTP cache, its
        MoE (bank -1), the MTP norm. Returns (y [M, H] for the head, y for the next step)."""
        from tools import fn_kernels as K
        c = self.cfg
        eps = c.rms_norm_eps
        M = stack_in.shape[0]
        rb = self._rows_buffers(M, key)
        self._cur, self._write, self.rec, self._cls = rb, False, {}, cls
        x = self._mtp_in(self.W["embed_tokens.weight"].index_select(0, rb["d"]["tok"]), stack_in)
        p = self.pre(MTP)
        zd = self._zero(M)
        n = K.hc_norm(x, self.W[p + "input_layernorm.weight"], c.hidden_size, eps)
        out = self._attn1(n, MTP, st_m)
        h, n = K.inject_norm(x, out, zd, 0, self.W[p + "post_attention_layernorm.weight"], 1, eps)
        yd, ids, wts = self.moe_parts(n, MTP)
        _, y = K.combine_inject_norm(h, yd, ids, wts, zd, 0, self.W["mtp.norm.weight"], 1, eps)
        return y, y

    @torch.no_grad()
    def mtp_pre(self, stack: torch.Tensor, next_tokens: torch.Tensor, st_m) -> torch.Tensor:
        """The MTP layer over a chunk of committed rows on the prefill kernels; the cache advances.
        Returns y [T, H]."""
        from tools import fn_kernels as K
        c = self.cfg
        eps = c.rms_norm_eps
        T = next_tokens.numel()
        self._pre = self._pre_ctl(st_m.length, T, st_m)
        x = self._mtp_in(self.embed(next_tokens.view(-1)), stack.to(self.dtype))
        p = self.pre(MTP)
        zd = self._zero(T)
        n = K.hc_norm(x, self.W[p + "input_layernorm.weight"], c.hidden_size, eps)
        out = self._attn_pre(n, MTP, st_m)
        h, n = K.inject_norm(x, out, zd, 0, self.W[p + "post_attention_layernorm.weight"], 1, eps)
        _, y = K.inject_norm(h, self._moe_pre(n, MTP), zd, 0, self.W["mtp.norm.weight"], 1, eps)
        st_m.length += T
        self._pre = None
        return y
