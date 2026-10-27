"""Flash-Next on the board (ENG-221, ENG-227): the reference forward with its weight reads on the
kernels, and a row step -- one decode token, or a verify block of up to 32 draft rows -- that reads
its inputs from device buffers, so a CUDA graph can hold it.

`FlashNextRef` (ref.py) is the model, operation for operation, checked against `transformers`. This
class keeps its structure and routes what reads weights to kernels over the pack's formats:

  EXL3 projections and the head     tools/fn_kernels.dense (up to 32 rows), tools/exl3_linear above
  mixers (int8 rows or fp16)        tools/fn_kernels.gemv (down and inject in one weight) + mix_up
  router, routed and shared experts tools/fn_kernels.router + moe (one bank, the shared expert last)
  GDN a/b, PLE projections (fp16)   tools/fn_kernels.gemv
  GDN conv, gates and recurrence    tools/fn_seq_kernels.gdn_pre + gdn_rows (a row's path)
  PLE gate and dilated conv         tools/fn_seq_kernels.ple_gate + ple_conv (a row's path)
  attention                         tools/fn_seq_kernels.attn_rows (a row's visible keys)

`forward()` (the reference's) serves a prompt chunk of any length. `rows_step()` is M rows whose
tokens, positions, cache slots, ancestors and PLE rows sit in device buffers (`stage` for a decode
token, `stage_rows` for a draft block in DFS pre-order); `DecodeGraph` and `VerifyGraph` capture it.
Every part of a row is computed with an instruction sequence that does not depend on the other rows
or on where its predecessors sit, so the row of a verify block is the decode step's row at that
position bit for bit: a greedy speculative run writes the plain greedy text (ENG-227). A decode step
advances the state in place; a verify block writes nothing but KV rows past the length and keeps what
`commit` needs, which then walks the accepted path's GDN rows again from the entry state (the decode
arithmetic), takes the conv windows from the path and moves the path's KV rows into place.

The row step covers the dense attention regime: a context of up to `indexer_budget + 3` tokens
(2,051), where QSA keeps every token. Past it a decode step takes the reference's attention (eager,
slow, exact) and a verify block is refused; the selected-key path is ENG-147's work.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from engine import capture_lock
from engine.settings import SETTINGS as _S
from engine.models.flashnext.ref import MTP, FlashNextRef, FNState
from engine.models.flashnext.ple import decode_rows


# rows up to which an EXL3 projection runs on tools/fn_kernels.dense (the kernel's limit);
# QWEN38_FN_DENSE=0: the library path of tools/exl3_linear for every row count (an A/B)
DENSE_MAX = 32 if _S.get("FN_DENSE") == "1" else 0
# ENG-225: a prompt chunk through the prefill kernels (tools/fn_prefill_kernels.py: experts grouped by
# expert, mixers and attention on many rows) or the reference forward ("ref"), and the chunk length
PREFILL = _S.get("FN_PREFILL")
PREFILL_CHUNK = int(_S.get("FN_PREFILL_CHUNK"))


class FlashNextEngine(FlashNextRef):

    def __init__(self, cfg, W: dict, tables: dict, banks: dict, device="cuda", max_len: int = 8192):
        super().__init__(cfg, W, tables, dtype=torch.bfloat16, device=device)
        self.banks = banks
        self.max_len = max_len
        cos, sin = FlashNextRef.rope(self, torch.arange(max_len))
        self.rope_cos, self.rope_sin = cos, sin                          # [max_len, rotary] bf16
        self.dense_limit = (cfg.indexer_budget + cfg.indexer_compress_ratio - 1) if cfg.has_indexer else 1 << 60
        self._rb: dict = {}                                              # rows buffers by row count
        self._order1 = torch.empty(0, dtype=torch.int32, device=device)    # one row: launch order
        self.rec: dict = {}                                              # what a commit needs
        self.host_ms: dict = {}                                          # host staging time by part
        # the commit folded into the next call (QWEN38_FN_FOLD_COMMIT): the accepted path's GDN rows,
        # parked a layer, applied by the next verify block or decode step to each state tile as it
        # loads it (bit for bit the commit's own walk); `_pend_owner` is the state they belong to
        self.fold = _S.get("FN_FOLD_COMMIT") == "1"
        self._pend: dict = {}
        self._pend_n = torch.zeros(max(len(cfg.linear_layers), 1), dtype=torch.int32, device=device)
        self._lin_idx = {li: i for i, li in enumerate(cfg.linear_layers)}
        self._pend_owner = None
        self._joint: dict = {}          # id(state) -> (block, rec, class) of its block in the last joint pass
        self._mtp_key = 0               # whose MTP chain buffers the next draft uses (a head's `key`)
        self.joint_timing = None        # a dict: `verify_multi` adds each part's GPU ms (a measurement)
        # the sequence kernels' weight forms: GDN conv taps [C, 4] bf16, A_log and dt_bias fp32, the
        # PLE conv taps rounded to bf16 (the reference casts them to the activations' dtype)
        self._wc: dict = {}
        for li in cfg.linear_layers:
            a = self.pre(li) + "linear_attn."
            if a + "conv1d.weight" not in W:
                continue
            self._wc[li] = W[a + "conv1d.weight"].reshape(cfg.conv_dim, -1).to(torch.bfloat16).contiguous()
            self._wc[("alog", li)] = W[a + "A_log"].float().contiguous()
            self._wc[("dtb", li)] = W[a + "dt_bias"].float().contiguous()
        for li in cfg.ple_layers:
            w = self.pre(li) + "ple.conv1d.weight"
            if w in W:
                self._wc[("ple", li)] = W[w].reshape(W[w].shape[0], -1).to(torch.bfloat16).contiguous()

    # ------------------------------------------------------------------------- state
    def new_state(self, cap: int | None = None, paged: bool = False) -> FNState:
        """A state with every tensor the step writes allocated up front (a graph holds them). `paged`
        with the KV pool on (engine/kvpool.py, ENG-241): the K, V and indexer-key buffers are reserved
        as addresses and committed as rows are written -- by the caller, before every write
        (`FlashNextServing._ensure`) -- with the same shapes, strides and addresses for every kernel and
        graph, so the bits are the static buffers'."""
        from engine import kvpool as _kvp
        c = self.cfg
        cap = cap or self.max_len
        st = FNState(cap=cap)
        if (paged and _kvp.POOL is not None and cap >= _kvp.PAGED_MIN_ROWS
                and str(self.device).startswith("cuda")):
            st.pages = _kvp.PagedSet(_kvp.OWNER)
        dev = self.device
        for li in range(c.num_hidden_layers):
            if c.is_linear(li):
                st.S[li] = torch.zeros(1, c.linear_num_value_heads, c.linear_key_head_dim,
                                       c.linear_value_head_dim, dtype=torch.float32, device=dev)
                st.conv[li] = torch.zeros(1, c.conv_dim, c.linear_conv_kernel_dim - 1, dtype=self.dtype, device=dev)
            else:
                st.k[li] = self._rows_tensor(st, (c.num_key_value_heads, cap, c.head_dim), 1, f"k.{li}")
                st.v[li] = self._rows_tensor(st, (c.num_key_value_heads, cap, c.head_dim), 1, f"v.{li}")
                if c.has_indexer:
                    st.ik[li] = self._rows_tensor(st, (cap, c.indexer_head_dim), 0, f"ik.{li}")
                    st.pool[li] = self._pool_tensor(cap)
        for li in c.ple_layers:
            st.ple_conv[li] = torch.zeros(1, c.hc_hidden, (c.ple_conv_kernel_size - 1) * c.ngram_size,
                                          dtype=self.dtype, device=dev)
        return st

    def _rows_tensor(self, st: FNState, shape: tuple, axis: int, name: str) -> torch.Tensor:
        """A buffer indexed by position on `axis`: paged in the state's set, else zeros."""
        if st.pages is None:
            return torch.zeros(*shape, dtype=self.dtype, device=self.device)
        from engine import kvpool as _kvp
        return st.pages.add(_kvp.PagedBuffer(shape, axis, self.dtype, self.device, owner=st.pages.owner,
                                             name=f"fn.{name}")).tensor

    def _pool_tensor(self, cap: int) -> torch.Tensor:
        """The pooled indexer keys of a cache of `cap` tokens: a row a block of 4, kept up to date by
        every decode step and commit (ENG-147), plus the rows a commit may write past the last block."""
        from tools import fn_seq_kernels as SK
        return torch.zeros(cap // self.cfg.indexer_compress_ratio + SK.OWN + 1, self.cfg.indexer_head_dim,
                           dtype=self.dtype, device=self.device)

    @torch.no_grad()
    def build_pool(self, st: FNState, layers=None) -> None:
        """Pool every complete block of the state's indexer keys (after a prefill, which the reference
        forward runs without the pool). The same kernel as a step's, so the bits are the step's."""
        from tools import fn_seq_kernels as SK
        c = self.cfg
        nb = st.length // c.indexer_compress_ratio
        if not c.has_indexer or nb == 0:
            return
        dev = self.device
        b = torch.arange(nb, dtype=torch.int32, device=dev)
        slots = (b[:, None] * 4 + torch.arange(4, dtype=torch.int32, device=dev)).contiguous()
        cos, sin, off = self._rot(st, 0, nb * 4)
        for li in (layers if layers is not None else c.attention_layers):
            p = self.pre(li) + "self_attn.indexer."
            SK.qsa_pool(st.ik[li], slots, b * 4 + off, self.W[p + "k_layernorm.weight"], cos, sin,
                        c.rms_norm_eps, out=st.pool[li][:nb])

    def reset(self, st: FNState) -> None:
        if self._pend_owner is st:
            self._pend_n.zero_()
            self._pend_owner = None
        st.reset_inplace()
        for d in (st.k, st.v, st.ik, st.pool):
            for v in d.values():
                v.zero_()

    # ------------------------------------------------------------------------- primitives
    def lin(self, x, w):
        from tools import fn_kernels as K
        from tools.exl3_linear import EXL3Block, exl3_matmul
        if isinstance(w, EXL3Block):
            lead = x.shape[:-1]
            x2 = x.reshape(-1, x.shape[-1]).contiguous()
            # up to 32 rows (decode, verify): Flash-Next's dense kernel (rows invariant, PDL); a
            # prompt chunk: the library path of tools/exl3_linear
            y = K.dense(x2, w) if x2.shape[0] <= DENSE_MAX else exl3_matmul(x2, w)
            return y.view(*lead, w.N)
        if hasattr(w, "q") or (isinstance(w, torch.Tensor) and w.dtype == torch.float16 and w.dim() == 2):
            lead = x.shape[:-1]
            y = K.gemv(x.reshape(-1, x.shape[-1]).contiguous(), w)
            return y.to(x.dtype).view(*lead, -1)
        return super().lin(x, w)

    def rope(self, positions: torch.Tensor):
        p = positions.to(self.rope_cos.device).long()
        return self.rope_cos.index_select(0, p), self.rope_sin.index_select(0, p)

    def rope_at(self, positions: torch.Tensor):
        return self.rope(positions)

    def _rot(self, st: FNState, a: int, b: int):
        """The rotary rows of cache slots [a, b) of `st` (VIS-97): (cos, sin, off), slot s reads row
        s + off. A text row reads the 1-D table at its position, slot + delta: 0 before any image, the
        prompt's `pos3` after one, `st.delta` past the prompt (transformers' `get_rope_index`). A range
        that holds image rows gets rows of its own: engine/vision.py `mrope_cos_sin` of the three
        coordinates, on the host as the table is made, so a text row among them is the table's row bit
        for bit."""
        mm = st.mm
        if mm is None:
            return self.rope_cos, self.rope_sin, st.delta
        if not any(s.start < b and s.start + s.n > a for s in mm.spans):
            return self.rope_cos, self.rope_sin, (int(mm.pos3[0, a]) - a if a < mm.n else st.delta)
        from engine.vision import mrope_axes, mrope_cos_sin
        c = self.cfg
        p3 = mm.pos3[:, a:b]
        if b > mm.n:
            tail = torch.arange(max(a, mm.n), b) + st.delta
            p3 = torch.cat([p3, tail.view(1, -1).expand(3, -1)], dim=1)
        axes = torch.tensor(mrope_axes(c.rotary_dim, list(c.mrope_section)))
        cos, sin = mrope_cos_sin(p3, self.inv_freq, axes, dtype=self.dtype)
        return cos.to(self.device), sin.to(self.device), -a

    def _image_rows(self, e: torch.Tensor, st: FNState, L0: int) -> torch.Tensor:
        """A chunk's embeddings [T, H] at slots L0.. with the images' rows over their placeholders'
        (transformers' `masked_scatter` of the image features into `inputs_embeds`); an image is encoded
        when the first chunk that needs it runs (engine/vision.py `MMContext.embeds`)."""
        mm, T = st.mm, e.shape[0]
        for s in mm.spans:
            a, b = max(s.start, L0), min(s.start + s.n, L0 + T)
            if a < b:
                e[a - L0:b - L0] = mm.embeds(s.image)[a - s.start:b - s.start].to(e.dtype)
        return e

    def delta_rule(self, q, k, v, g, beta, S):
        if q.shape[1] > 64:
            from tools.gdn_prefill_kernels import CHUNK, fused_chunk_prefill, fused_prefill_refusal
            if not fused_prefill_refusal(CHUNK):
                return fused_chunk_prefill(q, k, v, g, beta, S, chunk_size=CHUNK)
        return super().delta_rule(q, k, v, g, beta, S)

    def mixer(self, h, p, combine=True):
        """norm, down (+ inject) in one GEMV, then mix_up: the reference's mixer with each bf16
        rounding where `lin` makes it."""
        from tools import fn_kernels as K
        c = self.cfg
        n = self.rms(h, self.w(p + ".hc_norm.weight"), c.rms_norm_eps, group=c.hidden_size)
        di = K.gemv(n, self.W[p + ".down_inject"])
        x = K.mix_up(di, n, self.W[p + ".input_mix_weight_up.weight"], c.hc_count)
        if not combine:
            return x
        R = self.W[p + ".rank"]
        inj = 2 * torch.sigmoid(di[:, R:].to(self.dtype) / c.hc_count)
        return x, h, inj

    def moe(self, x, li):
        from tools import fn_kernels as K
        c = self.cfg
        out = torch.empty_like(x)
        step = 2048
        for r0 in range(0, x.shape[0], step):
            xs = x[r0:r0 + step]
            logits = K.gemv(xs, self.W[self.pre(li) + "mlp.router"])
            ids, wts = K.router(logits, c.num_experts, c.num_experts_per_tok)
            if self.route_trace is not None:
                self.route_trace.append((li, ids[:, :-1].long()))
            # one row: any launch order is the same arithmetic, so skip the sort
            order = self._order1 if xs.shape[0] == 1 else None
            K.moe(xs, ids, wts, self.banks[li], order=order, out=out[r0:r0 + step])
        return out

    def moe_parts(self, x, li):
        """The router and the experts of every (row, slot) pair, without the combine (the trunk
        folds it into the next inject): (yd [M S, H] fp32, ids, wts)."""
        from tools import fn_kernels as K
        c = self.cfg
        logits = K.gemv(x, self.W[self.pre(li) + "mlp.router"])
        ids, wts = K.router(logits, c.num_experts, c.num_experts_per_tok)
        if self.route_trace is not None:
            self.route_trace.append((li, ids[:, :-1].long()))
        yd = K.moe_experts(x, ids, self.banks[li], order=self._order1 if x.shape[0] == 1 else None)
        if self._moe_record is not None:
            self._moe_record.append((li, x.clone(), ids.clone(), wts.clone(), yd.clone()))
        return yd, ids, wts

    _moe_record = None
    spec_ok = False            # the load-time self-check passed (speculation may run)
    MOE_ROWS = 2048            # rows the MoE takes in one call (11 pairs a row on the grid's y axis)
    TREE_ROWS = (0, 1, 2, 3, 5, 6, 8, 11)
    CHAIN_WIDTHS = (2, 4, 8, 12, 16, 32)

    @torch.no_grad()
    def self_check(self, prompt: list[int], steps: int = 32, log=print) -> dict:
        """ENG-223's load-time check of the expert kernel's verify twins on real routing: decode
        `steps` tokens one at a time (the decode step), recording every layer's MoE input, ids,
        weights and every pair's expert output; then run each layer's rows again as chain calls of
        every verify width in CHAIN_WIDTHS (the first rows) and an 8-row tree-shaped call (rows
        TREE_ROWS, out of order), with the router recomputed from the recorded inputs, and compare
        every row bit for bit (the
        combine is per row and per column, so equal pairs give an equal sum). Any difference
        leaves `spec_ok` False and the log names the layer and the row."""
        from tools import fn_kernels as K
        c = self.cfg
        st = self.new_state(self.max_len)
        logits = self.prefill(torch.tensor(prompt), st)
        tok = int(logits[0].argmax())
        self._moe_record = rec = []
        try:
            for _ in range(steps):
                tok = int(self.decode_eager(tok, st)[0].argmax())
        finally:
            self._moe_record = None
        by_layer: dict = {}
        for li, x, ids, wts, out in rec:
            by_layer.setdefault(li, []).append((x, ids, wts, out))
        bad = None
        checked = 0
        S = c.num_experts_per_tok + 1
        for li, rows in sorted(by_layer.items()):
            X = torch.cat([r[0] for r in rows])
            I = torch.cat([r[1] for r in rows])
            Wt = torch.cat([r[2] for r in rows])
            Y = torch.cat([r[3] for r in rows]).view(len(rows), S, -1)        # each row's pairs
            ids, wts = K.router(K.gemv(X, self.W[self.pre(li) + "mlp.router"]), c.num_experts, c.num_experts_per_tok)
            calls = [(f"chain{w}", list(range(w))) for w in self.CHAIN_WIDTHS if w <= len(rows)]
            calls.append(("tree", [t for t in self.TREE_ROWS[::-1] if t < len(rows)]))
            for name, sel in calls:
                sl = torch.tensor(sel, device=X.device)
                got = K.moe_experts(X[sl].contiguous(), I[sl].contiguous(), self.banks[li]).view(len(sel), S, -1)
                for j, t in enumerate(sel):
                    checked += 1
                    if bad is None and not (torch.equal(got[j], Y[t]) and torch.equal(ids[t], I[t]) and torch.equal(wts[t], Wt[t])):
                        bad = {"layer": li, "row": t, "call": name}
        self.spec_ok = bad is None
        res = {"ok": self.spec_ok, "steps": steps, "layers": len(by_layer), "rows_checked": checked, "first_diff": bad}
        if log:
            log(f"[fn-selfcheck] {'PASS' if self.spec_ok else 'FAIL: speculation stays off'}: {res}")
        return res

    # ------------------------------------------------------------------------- the trunk
    def _mix(self, n, p):
        """The mixer from its normed streams: (x [M, H], d [M, R + hc] fp32 = down and inject)."""
        from tools import fn_kernels as K
        di = K.gemv(n, self.W[p + ".down_inject"])
        return K.mix_up(di, n, self.W[p + ".input_mix_weight_up.weight"], self.cfg.hc_count), di

    def _site_norm(self, li: int):
        """The norm weight of layer li's first mixer (the final mixer's past the last layer)."""
        if li >= self.cfg.num_hidden_layers:
            return self.W["hyper_connection_mixer.hc_norm.weight"]
        return self.W[self.pre(li) + "attn_hyper_connection.hc_norm.weight"]

    def trunk(self, h, tokens, st, positions, step: bool = False, pre: bool = False):
        """Every layer: each inject writes the streams and, in the same launch, the next site's norm
        (not in front of a PLE layer, which changes the streams first). Returns the final mixer's
        normed streams n [M, hc H]; the streams themselves are `self.stack`. `pre`: a prompt chunk
        through the prefill kernels (ENG-225)."""
        from tools import fn_kernels as K
        c = self.cfg
        H, hc, eps = c.hidden_size, c.hc_count, c.rms_norm_eps
        mix = self._mix_pre if pre else self._mix
        n = None
        for li in range(c.num_hidden_layers):
            p = self.pre(li)
            if li in c.ple_layers:
                if step:
                    h = h + self._ple_step(h, li, st)
                else:
                    h = h + (self._ple_pre(h, tokens, li, st) if pre else self.ple(h, tokens, li, st))
                n = None
            if n is None:
                n = K.hc_norm(h, self._site_norm(li), H, eps)
            x, di = mix(n, p + "attn_hyper_connection")
            if c.is_linear(li):
                out = self._gdn1(x, li, st) if step else (self._gdn_pre(x, li, st) if pre else self.gdn(x, li, st))
            else:
                out = self._attn1(x, li, st) if step else (self._attn_pre(x, li, st) if pre else
                                                           self.attention(x, li, st, positions))
            h, n = K.inject_norm(h, out, di, self.W[p + "attn_hyper_connection.rank"],
                                 self.W[p + "mlp_hyper_connection.hc_norm.weight"], hc, eps)
            x, di = mix(n, p + "mlp_hyper_connection")
            nxt = None if li + 1 in c.ple_layers else self._site_norm(li + 1)
            R = self.W[p + "mlp_hyper_connection.rank"]
            if pre:
                h, n = K.inject_norm(h, self._moe_pre(x, li), di, R, nxt, hc, eps)
            elif x.shape[0] <= self.MOE_ROWS:
                yd, ids, wts = self.moe_parts(x, li)
                if step and not self._write:
                    self.rec[("ids", li)] = ids            # the experts a verify block reads (ENG-155 prices)
                h, n = K.combine_inject_norm(h, yd, ids, wts, di, R, nxt, hc, eps)
            else:
                h, n = K.inject_norm(h, self.moe(x, li), di, R, nxt, hc, eps)
        self.stack = h
        if n is None:
            n = K.hc_norm(h, self._site_norm(c.num_hidden_layers), H, eps)
        return n

    def final(self, n):
        """The final mixer (no inject) from its normed streams, and the head: fp32 logits."""
        from tools import fn_kernels as K
        d = K.gemv(n, self.W["hyper_connection_mixer.down_inject"])
        x = K.mix_up(d, n, self.W["hyper_connection_mixer.input_mix_weight_up.weight"], self.cfg.hc_count)
        return self.lin(x, self.W["lm_head.weight"]).float()

    @torch.no_grad()
    def forward(self, tokens, st, layers=None, trace=None, head="all"):
        """A prompt chunk (or any block) through the fused trunk; the reference's forward for a
        layer subset or a trace."""
        if layers is not None or trace is not None:
            return super().forward(tokens, st, layers=layers, trace=trace, head=head)
        tokens = tokens.view(-1).to(self.device)
        T = tokens.shape[0]
        positions = torch.arange(st.length, st.length + T, device=self.device)
        h = self.embed(tokens).repeat(1, self.cfg.hc_count)
        n = self.trunk(h, tokens, st, positions)
        st.length += T
        if head == "none":
            return None
        return self.final(n[-1:] if head == "last" else n)

    # ------------------------------------------------------------------------- prompt chunks (ENG-225)
    @torch.no_grad()
    def forward_pre(self, tokens, st, head="all"):
        """A prompt chunk through the prefill kernels: the trunk's arithmetic shared across rows (experts
        grouped by expert, mixers and the router on the library GEMM, attention over each row's visible
        keys with the QSA selection on the kernels, the GDN conv and gates in one launch, the chunked
        delta rule). Agrees with `forward` on the next token, not bit for bit (tools/fn_prefill_check.py).
        Advances `st` as `forward` does: caches, recurrent and conv states, the pooled indexer keys."""
        tokens = tokens.view(-1).to(self.device)
        T = tokens.shape[0]
        self._pre = self._pre_ctl(st.length, T, st)
        e = self.embed(tokens)
        if st.mm is not None:
            e = self._image_rows(e, st, st.length)
        h = e.repeat(1, self.cfg.hc_count)
        n = self.trunk(h, tokens, st, None, pre=True)
        st.length += T
        self._pre = None
        if head == "none":
            return None
        return self.final(n[-1:] if head == "last" else n)

    def _pre_ctl(self, L0: int, T: int, st: FNState) -> dict:
        """What every layer of a chunk of `st` at positions L0..L0+T-1 shares: the positions, their
        rotary rows (`cos`, `sin` read at `rpos` by the rows and at `pool_pos` by the blocks the chunk
        completes, `_rot`), the GDN conv and PLE conv taps along the chain (`gdn_pre` / `ple_conv`
        sources), each row's attention control (`attn_rows`: visible keys, no ancestors, dense flag),
        and past 2,051 tokens the QSA inputs (`qsa_scores_rows` / `qsa_select`: complete blocks, all in
        the shared pool; the tail)."""
        from tools import fn_seq_kernels as SK
        c, dev = self.cfg, self.device
        pos = torch.arange(L0, L0 + T, device=dev)
        m = torch.arange(T, device=dev)[:, None]

        def chain(dists, slen):
            d = torch.tensor(dists, device=dev)[None, :]
            return torch.where(m - d >= 0, m - d, -1 - (slen - d + m)).to(torch.int32).contiguous()
        out = {"pos": pos, "csrc": chain((3, 2, 1), c.linear_conv_kernel_dim - 1),
               "psrc": chain((9, 6, 3), (c.ple_conv_kernel_size - 1) * c.ngram_size), "bufs": {}}
        r = c.indexer_compress_ratio if c.has_indexer else 4
        cos, sin, off = self._rot(st, L0 - L0 % r, L0 + T)
        out.update(cos=cos, sin=sin, rpos=pos + off)
        nblk = (pos + 1) // r
        dense = (nblk <= SK.TOPK) | (not c.has_indexer)
        ctl = torch.zeros(T, 4, dtype=torch.int32, device=dev)
        ctl[:, 0] = torch.where(dense, pos + 1, 4 * SK.TOPK + pos + 1 - r * nblk)
        ctl[:, 1] = 1 << 30                      # no row's keys come from ancestors: every key is cached
        ctl[:, 2] = dense
        out.update(ctl=ctl, anc=torch.zeros(T, 1, dtype=torch.int32, device=dev),
                   sel=torch.zeros(T, SK.NSEL, dtype=torch.int32, device=dev), cls=self.qsa_class(L0 + T - 1))
        if c.has_indexer:
            b = torch.arange(L0 // r, (L0 + T) // r, dtype=torch.int32, device=dev)
            out.update(nblk=nblk, tail=(r * nblk[:, None] + torch.arange(3, device=dev)[None, :]).to(torch.int32),
                       pool=(L0 // r, (L0 + T) // r), pool_slots=(b[:, None] * r + torch.arange(r, dtype=torch.int32,
                                                                                               device=dev)).contiguous(),
                       pool_pos=(b * r + off).contiguous())
        return out

    def _mix_pre(self, n, p):
        """The mixer of a chunk: down and inject on the library GEMM (fp32), the up projection of
        a = bf16(silu(bf16(d) / hc)) likewise, the stream mean with mix_up's roundings."""
        from tools import fn_prefill_kernels as PK
        c = self.cfg
        di = PK.mm_rows(n, self.W[p + ".down_inject"])
        R = self.W[p + ".rank"]
        a = torch.nn.functional.silu(di[:, :R].to(self.dtype).float() / c.hc_count).to(self.dtype)
        return PK.mix_tail(PK.mm_rows(a, self.W[p + ".input_mix_weight_up.weight"]), n, c.hc_count), di

    def _moe_pre(self, x, li):
        """The MoE of a chunk: the router (library GEMM, the router kernel), every pair's expert output
        grouped by expert (`moe_grouped`), the combine (`moe_combine`: one CTA a row's columns reads
        each pair's output once; the decode path folds it into the inject, four times a row) -> [M, H]."""
        from tools import fn_kernels as K
        from tools import fn_prefill_kernels as PK
        c = self.cfg
        ids, wts = K.router(PK.mm_rows(x, self.W[self.pre(li) + "mlp.router"]), c.num_experts, c.num_experts_per_tok)
        if self.route_trace is not None:
            self.route_trace.append((li, ids[:, :-1].long()))
        return K.moe_combine(PK.moe_grouped(x, ids, self.banks[li]), ids, wts)

    def _attn_pre(self, x, li, st):
        """Attention of a chunk: q|k|v and the indexer projection, the prep kernel (norms, rope, the
        chunk's keys, values and indexer keys into the caches at their positions), the chunk's complete
        blocks pooled into the shared pool, past 2,051 tokens each row's block scores and its top 512,
        attention over each row's visible keys (`attn_rows`), the output gate, o_proj."""
        from tools import fn_kernels as K
        from tools import fn_prefill_kernels as PK
        from tools import fn_seq_kernels as SK
        c, P = self.cfg, self._pre
        p = self.pre(li) + "self_attn."
        ip = p + "indexer."
        qkv = self.lin(x, self.W[p + "qkv"])
        iqk = self.lin(x, self.W[ip + "index_qk_proj.weight"]) if c.has_indexer else None
        cls = P["cls"] if c.has_indexer else 0
        iq = None
        if cls:
            iq = P["bufs"].get("iq")
            if iq is None:
                iq = P["bufs"]["iq"] = torch.empty(x.shape[0], c.indexer_n_heads, c.indexer_head_dim,
                                                   dtype=self.dtype, device=x.device)
        q = K.attn_prep(qkv, iqk, P["rpos"], self.W[p + "q_norm.weight"], self.W[p + "k_norm.weight"],
                        P["cos"], P["sin"], st.k[li], st.v[li], st.ik.get(li), c.num_attention_heads,
                        c.num_key_value_heads, c.indexer_n_heads * c.indexer_head_dim if c.has_indexer else 0,
                        c.rms_norm_eps, slot=P["pos"], qln=self.W[ip + "q_layernorm.weight"] if cls else None, iq=iq)
        if c.has_indexer and P["pool"][1] > P["pool"][0]:
            SK.qsa_pool(st.ik[li], P["pool_slots"], P["pool_pos"], self.W[ip + "k_layernorm.weight"], P["cos"],
                        P["sin"], c.rms_norm_eps, out=st.pool[li][P["pool"][0]:P["pool"][1]])
        if cls:
            sc = PK.qsa_scores_rows(iq, st.pool[li], P["nblk"], cls)
            SK.qsa_select(sc, P["tail"], P["sel"])
            del sc
        o = SK.attn_rows(q, st.k[li], st.v[li], P["ctl"], P["anc"], P["sel"], bufs=P["bufs"])
        y = K.gate_mul(o, qkv, c.head_dim)
        return self.lin(y, self.W[p + "o_proj.weight"])

    def _gdn_pre(self, x, li, st):
        """The GDN layer of a chunk: qkv|z (EXL3), a|b on the library GEMM, the conv over the chain and
        the gates in one launch (`gdn_pre`), the chunked delta rule (`delta_rule`), the gated norm,
        out_proj. The conv state keeps the chunk's last three inputs."""
        from tools import fn_kernels as K
        from tools import fn_prefill_kernels as PK
        from tools import fn_seq_kernels as SK
        from tools.gdn_prefill_kernels import CHUNK as GDN_CHUNK, fused_chunk_prefill, fused_prefill_refusal
        c = self.cfg
        a = self.pre(li) + "linear_attn."
        T = x.shape[0]
        C = c.conv_dim
        qkvz = self.lin(x, self.W[a + "qkvz"])
        ab = PK.mm_rows(x, self.W[a + "ab"]).to(self.dtype)
        y, g, beta = SK.gdn_pre(qkvz, st.conv[li], self._wc[li], self._pre["csrc"], ab, self._wc[("alog", li)],
                                self._wc[("dtb", li)])
        w = c.linear_conv_kernel_dim - 1
        hist = qkvz[-w:, :C].T if T >= w else torch.cat([st.conv[li][0], qkvz[:, :C].T], 1)[:, -w:]
        st.conv[li][0].copy_(hist)
        kd = c.key_dim
        q = y[:, :kd].view(1, T, -1, c.linear_key_head_dim)
        k = y[:, kd:2 * kd].view(1, T, -1, c.linear_key_head_dim)
        v = y[:, 2 * kd:].view(1, T, -1, c.linear_value_head_dim)
        H = c.linear_num_value_heads
        g, beta = g.view(1, T, H), beta.view(1, T, H)
        if T > 64 and not fused_prefill_refusal(GDN_CHUNK):
            # the key heads as they are (the kernel maps value heads to them) and value blocks of 16:
            # 10.2 vs 15.4 ms a layer at 4,096 rows (2026-10-11), the output within one bf16 step
            o, S = fused_chunk_prefill(q, k, v, g, beta, st.S[li], chunk_size=GDN_CHUNK, bv=16)
        else:
            if c.num_v_per_k > 1:
                q = q.repeat_interleave(c.num_v_per_k, dim=2)
                k = k.repeat_interleave(c.num_v_per_k, dim=2)
            o, S = self.delta_rule(q, k, v, g, beta, st.S[li])
        st.S[li].copy_(S)
        z = qkvz[:, C:]
        hh = K.gated_norm(o.reshape(-1, c.linear_value_head_dim).to(self.dtype), z.reshape(-1, c.linear_value_head_dim),
                          self.W[a + "norm.weight"], c.rms_norm_eps, c.output_gate_type == "sigmoid")
        return self.lin(hh.view(T, -1), self.W[a + "out_proj.weight"])

    def _ple_pre(self, h, tokens, li, st):
        """PLE of a chunk: the n-gram rows hashed on the host (the reference's `ple_ids`, which advances
        the context), read from the table's page cache as the decode step reads them and decoded on the
        device, the projections on the library GEMM, then the step's kernels (gate, norms, the dilated
        conv over the chain); the conv state keeps the last 9."""
        from tools import fn_kernels as K
        from tools import fn_prefill_kernels as PK
        from tools import fn_seq_kernels as SK
        c = self.cfg
        p = self.pre(li) + "ple."
        H, eps = c.hidden_size, c.rms_norm_eps
        T = h.shape[0]
        tab, pd = self.tables[li], self._ple_dev()
        ids = self.ple_ids(tokens, li, st).reshape(-1).numpy()
        packed = torch.from_numpy(tab.rows_of(ids)).to(self.device)
        heads = torch.from_numpy(np.searchsorted(tab.offsets_np, ids, side="right") - 1).to(self.device)
        emb = decode_rows(packed, heads, pd["bias"], tab.K, pd["codebook"]).to(self.dtype).reshape(T, -1)
        key = K.hc_norm(PK.mm_rows(emb, self.W[p + "key_proj.weight"]).to(self.dtype), self.W[p + "norm_key.weight"], H, eps)
        val = PK.mm_rows(emb, self.W[p + "value_proj.weight"]).to(self.dtype)
        qry = K.hc_norm(h, self.W[p + "norm_query.weight"], H, eps)
        gv = SK.ple_gate(key, qry, val, c.hc_count)
        gvn = K.hc_norm(gv, self.W[p + "norm_conv.weight"], H, eps)
        out = SK.ple_conv(gvn, gv, st.ple_conv[li], self._wc[("ple", li)], self._pre["psrc"])
        ps = st.ple_conv[li]
        w = ps.shape[-1]
        ps[0].copy_(gvn[-w:].T if T >= w else torch.cat([ps[0], gvn.T], 1)[:, -w:])
        return out

    # ------------------------------------------------------------------------- rows: decode and verify
    def _rows_buffers(self, M: int, key=None) -> dict:
        """The device inputs of an M-row call and their pinned host image (one H2D copy a call):

          tok, pos, slot      int64 [M]: the token, its position (block start + depth) and its cache
                              slot (block start + row; equal to the position in a chain)
          heads, packed       the PLE rows: hash head int64 [M nh], packed table row int16 [M nh, words]
          actl                int32 [M, 4]: visible keys, block start, dense flag (attn_rows)
          anc                 int32 [M, ANC]: the cache slot of the row's ancestor at each depth
          gctl                int32 [M, 2]: the GDN tile's (load, save) (gdn_rows)
          csrc, psrc          int32 [M, 3]: where the GDN conv taps (distance 3, 2, 1) and the PLE
                              conv taps (9, 6, 3) live: a row of the block or a state column
          qctl                int32 [M, 4]: QSA complete blocks, first own block (ENG-147)
          oslot, opos         int32 [M, OWN, 4], [M, OWN]: the key slots and first position of the
                              row's own blocks (those holding positions of the round)
          tail                int32 [M, 3]: the positions past the row's last complete block
          pidx                int64 [1]: the pool row a decode step writes (its own block)
        """
        key = M if key is None else key
        rb = self._rb.get(key)
        if rb is not None:
            return rb
        from tools import fn_seq_kernels as SK
        c, dev = self.cfg, self.device
        tab = next(iter(self.tables.values())) if self.tables else None
        nh = c.ngram_heads
        words = tab.words if tab is not None else 1
        n64 = M * (3 + nh) + 1
        w32 = (("actl", 4), ("anc", SK.ANC), ("gctl", 2), ("csrc", 3), ("psrc", 3), ("qctl", 4),
               ("oslot", SK.OWN * 4), ("opos", SK.OWN), ("tail", 3))
        n32 = M * sum(w for _, w in w32)
        n32 += n32 % 2                                          # the int16 rows start 8-byte aligned
        nb = 8 * n64 + 4 * n32 + 2 * M * nh * words

        def views(b):
            i64 = b[: 8 * n64].view(torch.int64)
            i32 = b[8 * n64: 8 * n64 + 4 * n32].view(torch.int32)
            v = {"tok": i64[:M], "pos": i64[M:2 * M], "slot": i64[2 * M:3 * M], "heads": i64[3 * M:n64 - 1],
                 "pidx": i64[n64 - 1:]}
            k = 0
            for name, w in w32:
                v[name] = i32[k:k + M * w].view(M, w)
                k += M * w
            v["packed"] = b[8 * n64 + 4 * n32:].view(torch.int16).view(M * nh, words)
            return v
        hbuf = torch.zeros(nb, dtype=torch.uint8, pin_memory=True)
        dbuf = torch.zeros(nb, dtype=torch.uint8, device=dev)
        copied = torch.cuda.Event()
        copied.record()
        rb = {"M": M, "hbuf": hbuf, "dbuf": dbuf, "h": views(hbuf), "d": views(dbuf), "copied": copied,
              "logits": torch.zeros(M, c.vocab_size, dtype=torch.float32, device=dev),
              "next": torch.zeros(M, dtype=torch.long, device=dev),
              "next_h": torch.zeros(M, dtype=torch.long, pin_memory=True), "attn": {}}
        rb["d"]["sel"] = torch.zeros(M, SK.NSEL, dtype=torch.int32, device=dev)
        if c.has_indexer:
            rb["d"]["iq"] = torch.zeros(M, c.indexer_n_heads, c.indexer_head_dim, dtype=self.dtype, device=dev)
        if M == 1:                                   # a decode step: everything but the position is fixed
            h = rb["h"]
            h["gctl"].numpy()[:] = (-2, -1)
            h["csrc"].numpy()[:] = (-1, -2, -3)
            h["psrc"].numpy()[:] = (-1, -4, -7)
        self._rb[key] = rb
        return rb

    def _ple_dev(self) -> dict:
        if not hasattr(self, "_pled"):
            tab = next(iter(self.tables.values()))
            self._pled = {"bias": tab.bias.to(self.device), "codebook": tab.codebook(self.device)}
        return self._pled

    def stage(self, token: int, st: FNState) -> None:
        """Host work of one decode step (before the step or its graph runs): the token, its position,
        its PLE rows (hash with the state's context, which advances, and gather from the table)."""
        rb = self._rows_buffers(1)
        rb["copied"].synchronize()             # the image is the previous call's copy source until then
        h, L = rb["h"], st.length
        h["tok"].numpy()[0] = token
        h["pos"].numpy()[0] = L + st.delta
        h["slot"].numpy()[0] = L
        h["anc"].numpy()[0, 0] = L
        self._stage_qsa(h, L, L // self.cfg.indexer_compress_ratio if self.cfg.has_indexer else 0,
                        np.array([L]), h["anc"].numpy(), 1, st.delta)
        for li in self.cfg.ple_layers:
            hsh = self.hash[li]
            ctx = self._ple_ctx(st, li)
            self._ple_fill(h, 0, li, hsh.ids_one(token, ctx))
            st.ple_ctx[li] = torch.tensor([(ctx + [token])[-hsh.context_len:]], dtype=torch.long)
        rb["dbuf"].copy_(rb["hbuf"], non_blocking=True)
        rb["copied"].record()

    def _stage_qsa(self, h: dict, L: int, own0: int, pos: np.ndarray, anc: np.ndarray, n: int,
                   delta: int = 0) -> None:
        """The attention control of n rows at positions `pos`: positions below L are cache slots,
        from L on the row's ancestor at that depth (`anc`). Rows whose complete blocks fit the budget
        see every position (dense); the others see their selected blocks and their tail (QSA). The
        row's own blocks start at block own0 (the shared pool holds the blocks before it); each is
        pooled with the rotary row of its first slot + `delta` (the state's, VIS-97: a round's own
        blocks start at most 3 rows before it, in text after any image -- the chat template ends a
        prompt with the assistant's header)."""
        from tools import fn_seq_kernels as SK
        c = self.cfg
        r = c.indexer_compress_ratio if c.has_indexer else 4
        nblk = (pos + 1) // r
        dense = (nblk <= SK.TOPK) | (not c.has_indexer)
        actl = h["actl"].numpy()
        actl[:n, 0] = np.where(dense, pos + 1, 4 * SK.TOPK + pos + 1 - r * nblk)
        actl[:n, 1] = L
        actl[:n, 2] = dense
        if not c.has_indexer:
            return
        q = (r * (own0 + np.arange(SK.OWN)))[:, None] + np.arange(r)[None, :]          # [OWN, 4] positions
        ai = anc[:n][:, np.clip(q - L, 0, anc.shape[1] - 1)]                         # [n, OWN, 4]
        slots = np.where(q[None] < L, q[None], np.where(q[None] <= pos[:, None, None], ai, 0))
        h["oslot"].numpy()[:n] = slots.reshape(n, -1)
        h["opos"].numpy()[:n] = r * (own0 + np.arange(SK.OWN)) + delta
        qc = h["qctl"].numpy()
        qc[:n, 0] = nblk
        qc[:n, 1] = own0
        h["tail"].numpy()[:n] = r * nblk[:, None] + np.arange(3)[None, :]
        h["pidx"].numpy()[0] = own0

    def qsa_class(self, last: int) -> int:
        """The context class of a call whose last position is `last`: 0 while every row is dense
        (no indexer work), else the score width in blocks (a power of two, at least 1,024)."""
        from tools import fn_seq_kernels as SK
        c = self.cfg
        if not c.has_indexer or (last + 1) // c.indexer_compress_ratio <= SK.TOPK:
            return 0
        nb = 1024
        while nb * c.indexer_compress_ratio < last + 1:
            nb *= 2
        return nb

    @staticmethod
    def path_ple_ids(hsh, ctx0: list[int], tokens: list[int], paths: list[list[int]]) -> list[list[int]]:
        """The n-gram row ids of every row of a block: each row hashed after the tokens on its own
        path (its ancestors, then the state's context `ctx0` before the block)."""
        n = hsh.context_len
        return [hsh.ids_one(tokens[pa[-1]], (ctx0 + [tokens[a] for a in pa[:-1]])[-n:]) for pa in paths]

    def _ple_ctx(self, st: FNState, li: int) -> list[int]:
        ctx = st.ple_ctx.get(li)
        return [self.cfg.eos_token_id] * self.hash[li].context_len if ctx is None else ctx.view(-1).tolist()

    def _ple_fill(self, h: dict, row: int, li: int, ids: list[int]) -> None:
        tab = self.tables[li]
        nh = self.cfg.ngram_heads
        h["packed"].numpy()[row * nh:(row + 1) * nh] = tab.rows_of(np.asarray(ids, dtype=np.int64))
        h["heads"].numpy()[row * nh:(row + 1) * nh] = np.searchsorted(tab.offsets_np, ids, side="right") - 1

    def stage_rows(self, tokens: list[int], parents: list[int], st: FNState, key=None) -> None:
        """Host work of a verify block: `tokens` in DFS pre-order with `parents` (-1 for the first
        row, which sits at the state's length). Nothing in `st` changes; `commit` keeps a path. `key`:
        the rows buffers' own key (a joint pass stages each state's block in its own, `verify_multi`)."""
        from tools import fn_seq_kernels as SK
        M = len(tokens)
        rb = self._rows_buffers(M, key)
        rb["copied"].synchronize()
        h, L = rb["h"], st.length
        depth, paths = [], []
        for i, p in enumerate(parents):
            depth.append(0 if p < 0 else depth[p] + 1)
            paths.append([i] if p < 0 else paths[p] + [i])
        d = np.asarray(depth, dtype=np.int64)
        h["tok"].numpy()[:] = tokens
        h["pos"].numpy()[:] = L + d + st.delta
        h["slot"].numpy()[:] = L + np.arange(M)
        anc = h["anc"].numpy()
        for i, pa in enumerate(paths):
            anc[i, :len(pa)] = L + np.asarray(pa)
        self._stage_qsa(h, L, L // self.cfg.indexer_compress_ratio if self.cfg.has_indexer else 0, L + d, anc, M,
                        st.delta)
        ctl, _ = SK.tree_ctl(parents)
        h["gctl"].numpy()[:] = ctl
        h["csrc"].numpy()[:] = SK.window_src(parents, (3, 2, 1), 3)
        h["psrc"].numpy()[:] = SK.window_src(parents, (9, 6, 3), 9)
        for li in self.cfg.ple_layers:
            hsh = self.hash[li]
            for i, ids in enumerate(self.path_ple_ids(hsh, self._ple_ctx(st, li), tokens, paths)):
                self._ple_fill(h, i, li, ids)
        rb["dbuf"].copy_(rb["hbuf"], non_blocking=True)
        rb["copied"].record()
        self._block = (list(tokens), list(parents), paths)

    def rows_step(self, st: FNState, M: int, write: bool = False, cls: int = 0) -> torch.Tensor:
        """M rows from the device buffers (`stage` / `stage_rows` first) through the trunk and the
        head. `write` (one row, a decode step): the recurrent and conv states and the pooled indexer
        keys advance in place. Otherwise nothing but the KV rows at the block's slots is written, and
        what a commit needs is kept in `self.rec`. `cls` (`qsa_class`): 0 = every row dense, else the
        indexer selects each sparse row's blocks over that many block scores. Returns logits [M, vocab]
        fp32 (a static buffer) and leaves the greedy tokens in rows["next"] and its pinned host copy.
        `st.length` is the caller's."""
        from tools import fn_kernels as K
        c = self.cfg
        rb = self._rows_buffers(M)
        self._cur, self._write, self._cls = rb, write, cls
        self.rec = {}
        d = rb["d"]
        h = self.W["embed_tokens.weight"].index_select(0, d["tok"]).repeat(1, c.hc_count)
        n = self.trunk(h, None, st, None, step=True)
        dd = K.gemv(n, self.W["hyper_connection_mixer.down_inject"])
        x = K.mix_up(dd, n, self.W["hyper_connection_mixer.input_mix_weight_up.weight"], c.hc_count)
        if DENSE_MAX:
            K.dense(x, self.W["lm_head.weight"], out=rb["logits"])
        else:
            from tools.exl3_linear import exl3_matmul
            exl3_matmul(x, self.W["lm_head.weight"], out=rb["logits"])
        # the greedy tokens, and their copy into pinned host memory: the host reads them after a sync
        torch.argmax(rb["logits"], dim=-1, out=rb["next"])
        rb["next_h"].copy_(rb["next"], non_blocking=True)
        if self.fold:
            self._pend_n.zero_()                   # every GDN layer applied its parked rows
        self.rec["stack"] = self.stack
        return rb["logits"]

    def step(self, st: FNState, cls: int | None = None) -> torch.Tensor:
        """One decode token from the device buffers (`stage` first); the state advances in place except
        `st.length`, which the caller moves. Returns logits [1, vocab] fp32 (a static buffer)."""
        return self.rows_step(st, 1, write=True, cls=self.qsa_class(st.length) if cls is None else cls)

    @property
    def g(self) -> dict:
        """The decode step's buffers (the one-row call's)."""
        return self._rb.get(1, {})

    @torch.no_grad()
    def verify(self, tokens: list[int], parents: list[int], st: FNState, graphs: dict | None = None) -> list[int]:
        """The target's greedy token at every row of a draft block (DFS pre-order, `parents`), eager or
        from `graphs` (M -> VerifyGraph). Each row's token is the decode step's at that position bit
        for bit. Nothing in `st` moves until `commit`."""
        M = len(tokens)
        depth = [0] * M
        for i, p in enumerate(parents):
            depth[i] = 0 if p < 0 else depth[p] + 1
        cls = self.qsa_class(st.length + max(depth))
        self.settle(st)
        self._joint.pop(id(st), None)
        gr = None
        if graphs is not None:
            gr = graphs.get((M, cls))
            if gr is None:
                gr = graphs[(M, cls)] = VerifyGraph(self, st, M, cls)
                gr.capture()
        t0 = time.perf_counter()
        self.stage_rows(tokens, parents, st)
        self.host_ms["stage_rows"] = self.host_ms.get("stage_rows", 0.0) + 1e3 * (time.perf_counter() - t0)
        self._vcls = cls
        if gr is not None:
            gr.replay()
        else:
            self.rows_step(st, M, cls=cls)
        self._pend_owner = None
        torch.cuda.current_stream().synchronize()
        return self._rb[M]["next_h"].tolist()

    @torch.no_grad()
    def verify_multi(self, items: list) -> list:
        """ENG-241 step 3: the verify blocks of several states (sequence slots over these weights) in
        one pass: [(tokens, parents, st)] -> per state (logits [M, vocab] fp32, its greedy tokens on the
        device, its streams [M, hc H]). The input and output projections of every layer, the experts and
        the head run once over all the states' rows (their weights read once); the PLE, the norms and
        mixers, the GDN recurrence and the attention run per state on its own rows and caches, as a
        lone verify runs them. The projections and the experts give a row the bits it gets alone up to
        32 rows (the self-check's widths), so each state's tokens are its lone verify's; the caller
        keeps a pass at or under 32 rows. Parked GDN rows are applied first (`flush`): each state's
        walk then starts from its own state. Eager; each state commits its block (`commit`)."""
        from tools import fn_kernels as K
        c = self.cfg
        H, hc, eps = c.hidden_size, c.hc_count, c.rms_norm_eps
        S = c.num_experts_per_tok + 1
        jt = self.joint_timing
        marks = []

        def mark(name):
            # QWEN38 joint-pass timing (tools/fn_joint_check.py --timing): an event at each part's start
            if jt is not None:
                ev = torch.cuda.Event(enable_timing=True)
                ev.record()
                marks.append((name, ev))
        mark("stage")
        self.flush()
        cx = []
        for j, (tokens, parents, st) in enumerate(items):
            M = len(tokens)
            depth = [0] * M
            for i, p in enumerate(parents):
                depth[i] = 0 if p < 0 else depth[p] + 1
            self.stage_rows(tokens, parents, st, key=("joint", j, M))
            cx.append({"st": st, "M": M, "cls": self.qsa_class(st.length + max(depth)), "block": self._block,
                       "rb": self._rows_buffers(M, ("joint", j, M)), "rec": {}})
        cuts = [0]
        for x in cx:
            cuts.append(cuts[-1] + x["M"])
        rows = [slice(cuts[j], cuts[j + 1]) for j in range(len(cx))]

        def use(x):
            self._cur, self._write, self._cls, self.rec = x["rb"], False, x["cls"], x["rec"]
        mark("mix")
        hs = [self.W["embed_tokens.weight"].index_select(0, x["rb"]["d"]["tok"]).repeat(1, hc) for x in cx]
        ns = [None] * len(cx)
        for li in range(c.num_hidden_layers):
            p = self.pre(li)
            mark("mix")
            xs, dis = [], []
            for j, x in enumerate(cx):
                use(x)
                if li in c.ple_layers:
                    hs[j] = hs[j] + self._ple_step(hs[j], li, x["st"])
                    ns[j] = None
                if ns[j] is None:
                    ns[j] = K.hc_norm(hs[j], self._site_norm(li), H, eps)
                xa, di = self._mix(ns[j], p + "attn_hyper_connection")
                xs.append(xa)
                dis.append(di)
            X = torch.cat(xs)
            mark("proj")
            if c.is_linear(li):
                a = p + "linear_attn."
                qkvz, ab = self.lin(X, self.W[a + "qkvz"]), K.gemv_bf16(X, self.W[a + "ab"])
                mark("gdn")
                parts = []
                for j, x in enumerate(cx):
                    use(x)
                    parts.append(self._gdn1_core(qkvz[rows[j]], ab[rows[j]], li, x["st"]))
                mark("proj")
                out = self.lin(torch.cat(parts), self.W[a + "out_proj.weight"])
            else:
                a = p + "self_attn."
                qkv = self.lin(X, self.W[a + "qkv"])
                iqk = self.lin(X, self.W[a + "indexer.index_qk_proj.weight"]) if c.has_indexer else None
                mark("attn")
                parts = []
                for j, x in enumerate(cx):
                    use(x)
                    parts.append(self._attn1_core(qkv[rows[j]], iqk[rows[j]] if iqk is not None else None, li,
                                                  x["st"]))
                mark("proj")
                out = self.lin(torch.cat(parts), self.W[a + "o_proj.weight"])
            mark("mix")
            xs, dis2 = [], []
            for j in range(len(cx)):
                hs[j], ns[j] = K.inject_norm(hs[j], out[rows[j]], dis[j], self.W[p + "attn_hyper_connection.rank"],
                                             self.W[p + "mlp_hyper_connection.hc_norm.weight"], hc, eps)
                xm, di = self._mix(ns[j], p + "mlp_hyper_connection")
                xs.append(xm)
                dis2.append(di)
            nxt = None if li + 1 in c.ple_layers else self._site_norm(li + 1)
            R = self.W[p + "mlp_hyper_connection.rank"]
            mark("moe")
            yd, ids, wts = self.moe_parts(torch.cat(xs), li)
            mark("mix")
            for j, x in enumerate(cx):
                r = rows[j]
                x["rec"][("ids", li)] = ids[r]
                hs[j], ns[j] = K.combine_inject_norm(hs[j], yd[r.start * S:r.stop * S], ids[r], wts[r], dis2[j], R,
                                                     nxt, hc, eps)
        mark("head")
        heads = []
        for j, x in enumerate(cx):
            if ns[j] is None:
                ns[j] = K.hc_norm(hs[j], self._site_norm(c.num_hidden_layers), H, eps)
            dd = K.gemv(ns[j], self.W["hyper_connection_mixer.down_inject"])
            heads.append(K.mix_up(dd, ns[j], self.W["hyper_connection_mixer.input_mix_weight_up.weight"], hc))
            x["rec"]["stack"] = hs[j]
        Xh = torch.cat(heads)
        if DENSE_MAX:
            logits = K.dense(Xh, self.W["lm_head.weight"])
        else:
            from tools.exl3_linear import exl3_matmul
            logits = exl3_matmul(Xh, self.W["lm_head.weight"])
        out = []
        for j, x in enumerate(cx):
            rb = x["rb"]
            rb["logits"][:x["M"]].copy_(logits[rows[j]])
            torch.argmax(rb["logits"][:x["M"]], dim=-1, out=rb["next"][:x["M"]])
            self._joint[id(x["st"])] = (x["block"], x["rec"], x["cls"])
            out.append((rb["logits"][:x["M"]], rb["next"][:x["M"]], hs[j]))
        self._pend_owner = None
        if jt is not None:
            mark("end")
            torch.cuda.synchronize()
            for (name, a), (_, b) in zip(marks, marks[1:]):
                jt[name] = jt.get(name, 0.0) + a.elapsed_time(b)
            jt["passes"] = jt.get("passes", 0) + 1
        return out

    @torch.no_grad()
    def commit(self, st: FNState, path: list[int], graphs: dict | None = None) -> torch.Tensor:
        """Keep the rows `path` (a root-to-node path of the last verify block, row 0 first): the GDN
        states walked again over the path's rows from the round's entry state (the decode step's
        arithmetic, written in place), the conv windows and the PLE conv state taken from the path,
        the path's KV rows moved down to their positions, the PLE context advanced, the length moved.
        `graphs` ((rows, path length, moved) -> CommitGraph): from a captured graph, which reads the
        verify graph's own intermediates. Returns the path's streams [len(path), hc H] (the MTP
        layer's input; a static buffer with graphs)."""
        joint = self._joint.pop(id(st), None)
        if joint is not None:
            # a block of a joint pass (`verify_multi`): its own block and intermediates, committed
            # eagerly (a commit graph reads the lone verify graph's intermediates)
            self._block, self.rec, self._vcls = joint
            graphs = None
        tokens = self._block[0]
        self.settle(st)
        t0 = time.perf_counter()
        key = self.stage_commit(st, path)
        self.host_ms["stage_commit"] = self.host_ms.get("stage_commit", 0.0) + 1e3 * (time.perf_counter() - t0)
        if graphs is not None:
            g = graphs.get(key)
            if g is None:
                g = graphs[key] = CommitGraph(self, st, key)
                g.capture()
            g.graph.replay()
        else:
            self.commit_dev(st, key)
        for li in self.cfg.ple_layers:
            hsh = self.hash[li]
            st.ple_ctx[li] = torch.tensor([(self._ple_ctx(st, li) + [tokens[i] for i in path])[-hsh.context_len:]],
                                          dtype=torch.long)
        st.length += len(path)
        if self.fold:
            self._pend_owner = st
        return self._commit_bufs(key)["stack"]

    def _commit_bufs(self, key) -> dict:
        """A commit's device inputs (path rows, KV source and destination slots, the conv tails) and
        its pinned host image; the path's streams as the output."""
        cb = self._rb.get(("commit",) + key)
        if cb is not None:
            return cb
        from tools import fn_seq_kernels as SK
        c, dev = self.cfg, self.device
        M, n = key[0], key[1]
        w9 = (c.ple_conv_kernel_size - 1) * c.ngram_size
        sizes = (("idx", n), ("src", n), ("dst", n), ("c3", c.linear_conv_kernel_dim - 1), ("c9", w9))
        tot = sum(w for _, w in sizes)
        hbuf = torch.zeros(tot, dtype=torch.int64, pin_memory=True)
        dbuf = torch.zeros(tot, dtype=torch.int64, device=dev)

        def views(b):
            out, k = {}, 0
            for name, w in sizes:
                out[name] = b[k:k + w]
                k += w
            return out
        copied = torch.cuda.Event()
        copied.record()
        cb = {"hbuf": hbuf, "dbuf": dbuf, "h": views(hbuf), "d": views(dbuf), "copied": copied,
              "stack": torch.zeros(n, c.hc_count * c.hidden_size, dtype=self.dtype, device=dev)}
        # the pool rows a commit refreshes: the blocks from the round's first block on (int32 slots
        # and first positions, int64 pool rows), from the committed cache after the moves
        cb["h32"] = torch.zeros(SK.OWN * 5, dtype=torch.int32, pin_memory=True)
        cb["d32"] = torch.zeros(SK.OWN * 5, dtype=torch.int32, device=dev)
        cb["hpi"] = torch.zeros(SK.OWN, dtype=torch.int64, pin_memory=True)
        cb["dpi"] = torch.zeros(SK.OWN, dtype=torch.int64, device=dev)
        self._rb[("commit",) + key] = cb
        return cb

    def stage_commit(self, st: FNState, path: list[int]) -> tuple:
        from tools import fn_seq_kernels as SK
        M, n = len(self._block[0]), len(path)
        moved = any(path[j] != j for j in range(n))
        key = (M, n, moved, getattr(self, "_vcls", 0))
        cb = self._commit_bufs(key)
        cb["copied"].synchronize()
        h, L = cb["h"], st.length
        h["idx"].numpy()[:] = path
        h["src"].numpy()[:] = [L + r for r in path]
        h["dst"].numpy()[:] = L + np.arange(n)
        h["c3"].numpy()[:] = self._tail_cols(path, h["c3"].numel())
        h["c9"].numpy()[:] = self._tail_cols(path, h["c9"].numel())
        if self.cfg.has_indexer:
            r = self.cfg.indexer_compress_ratio
            b = L // r + np.arange(SK.OWN)
            h32 = cb["h32"].numpy()
            h32[:SK.OWN * 4] = np.clip((r * b)[:, None] + np.arange(r)[None, :], 0, st.cap - 1).reshape(-1)
            h32[SK.OWN * 4:] = r * b + st.delta
            cb["hpi"].numpy()[:] = np.clip(b, 0, next(iter(st.pool.values())).shape[0] - 1)
            cb["d32"].copy_(cb["h32"], non_blocking=True)
            cb["dpi"].copy_(cb["hpi"], non_blocking=True)
        cb["dbuf"].copy_(cb["hbuf"], non_blocking=True)
        cb["copied"].record()
        return key

    def commit_dev(self, st: FNState, key) -> None:
        """The device half of `commit` (capturable): every input from `_commit_bufs(key)`."""
        from tools import fn_seq_kernels as SK
        c, rec = self.cfg, self.rec
        cb = self._commit_bufs(key)
        d = cb["d"]
        idx = d["idx"]
        chain = self._chain_ctl(key[1])
        for li in c.linear_layers:
            qkvz, y, gg, beta = rec[li]
            if self.fold:
                py, pg, pb = self._pend_bufs(li)
                torch.index_select(y, 0, idx, out=py[:key[1]])
                torch.index_select(gg, 0, idx, out=pg[:key[1]])
                torch.index_select(beta, 0, idx, out=pb[:key[1]])
            else:
                SK.gdn_rows(y.index_select(0, idx), c.key_dim, gg.index_select(0, idx), beta.index_select(0, idx),
                            st.S[li], self._scratch(), chain, c.num_v_per_k, True)
            hist = torch.cat([st.conv[li][0], qkvz[:, :c.conv_dim].T], 1)
            st.conv[li][0].copy_(hist.index_select(1, d["c3"]))
        if key[2]:
            for li in c.attention_layers:
                st.k[li].index_copy_(1, d["dst"], st.k[li].index_select(1, d["src"]))
                st.v[li].index_copy_(1, d["dst"], st.v[li].index_select(1, d["src"]))
                if li in st.ik:
                    st.ik[li].index_copy_(0, d["dst"], st.ik[li].index_select(0, d["src"]))
        if c.has_indexer:
            # the blocks the round touched, pooled again from the committed keys (ENG-147)
            slots, posb = cb["d32"][:SK.OWN * 4].view(SK.OWN, 4), cb["d32"][SK.OWN * 4:]
            for li in c.attention_layers:
                own = SK.qsa_pool(st.ik[li], slots, posb, self.W[self.pre(li) + "self_attn.indexer.k_layernorm.weight"],
                                  self.rope_cos, self.rope_sin, c.rms_norm_eps)
                st.pool[li].index_copy_(0, cb["dpi"], own)
        for li in c.ple_layers:
            hist = torch.cat([st.ple_conv[li][0], rec[("ple", li)].T], 1)
            st.ple_conv[li][0].copy_(hist.index_select(1, d["c9"]))
        torch.index_select(rec["stack"], 0, idx, out=cb["stack"])
        if self.fold:
            self._pend_n.fill_(key[1])

    def _pend_bufs(self, li: int):
        b = self._pend.get(li)
        if b is None:
            c = self.cfg
            b = self._pend[li] = (torch.zeros(33, c.conv_dim, dtype=self.dtype, device=self.device),
                                  torch.zeros(33, c.linear_num_value_heads, dtype=torch.float32, device=self.device),
                                  torch.zeros(33, c.linear_num_value_heads, dtype=torch.float32, device=self.device))
        return b

    def _parked(self, li: int):
        """The parked rows of GDN layer li for its next kernel call (None without the fold)."""
        if not self.fold:
            return None
        i = self._lin_idx[li]
        return self._pend_bufs(li) + (self._pend_n[i:i + 1],)

    @torch.no_grad()
    def flush(self) -> None:
        """Apply the parked GDN rows to their state now (before anything that reads the state other
        than a verify block or a decode step: the reference forward, a snapshot)."""
        st = self._pend_owner
        if st is None:
            return
        from tools import fn_seq_kernels as SK
        c = self.cfg
        z = self._chain_ctl(1)[:0]
        for li in c.linear_layers:
            py = self._pend_bufs(li)[0]
            SK.gdn_rows(py[:0], c.key_dim, self._pend_bufs(li)[1][:0], self._pend_bufs(li)[2][:0], st.S[li],
                        self._scratch(), z, c.num_v_per_k, False,
                        out=torch.empty(0, c.linear_num_value_heads, 128, dtype=torch.bfloat16, device=self.device),
                        pre=self._parked(li))
        self._pend_n.zero_()
        self._pend_owner = None

    def settle(self, st: FNState) -> None:
        """Before a call on `st`: parked rows of another state are applied to that state first."""
        if self._pend_owner is not None and self._pend_owner is not st:
            self.flush()

    @staticmethod
    def _tail_cols(path: list[int], w: int) -> list[int]:
        """The last `w` positions along `path` as columns of [state (w columns) | the block's rows]."""
        return (list(range(w)) + [w + r for r in path])[-w:]

    def _chain_ctl(self, n: int) -> torch.Tensor:
        if not hasattr(self, "_chains"):
            self._chains = {}
        if n not in self._chains:
            self._chains[n] = torch.tensor([[-2, -1]] + [[-1, -1]] * (n - 1), dtype=torch.int32, device=self.device)
        return self._chains[n]

    def _scratch(self) -> torch.Tensor:
        """The GDN tiles a tree block parks (one slot a branching row, at most 31); allocated once."""
        if not hasattr(self, "_stk"):
            c = self.cfg
            self._stk = torch.empty(31, c.linear_num_value_heads, c.linear_key_head_dim, c.linear_value_head_dim,
                                    dtype=torch.float32, device=self.device)
        return self._stk

    def _ple_step(self, h, li, st):
        """PLE of the rows: the gathered n-gram rows decoded, key/value projections, the gate, the norm,
        the dilated conv over the row's path (`psrc`), all row-invariant."""
        from tools import fn_kernels as K
        from tools import fn_seq_kernels as SK
        c, d = self.cfg, self._cur["d"]
        p = self.pre(li) + "ple."
        H, eps = c.hidden_size, c.rms_norm_eps
        tab, pd = self.tables[li], self._ple_dev()
        M = h.shape[0]
        emb = decode_rows(d["packed"], d["heads"], pd["bias"], tab.K, pd["codebook"]).to(self.dtype).reshape(M, -1)
        key = K.hc_norm(K.gemv_bf16(emb, self.W[p + "key_proj.weight"]), self.W[p + "norm_key.weight"], H, eps)
        val = K.gemv_bf16(emb, self.W[p + "value_proj.weight"])
        qry = K.hc_norm(h, self.W[p + "norm_query.weight"], H, eps)
        gv = SK.ple_gate(key, qry, val, c.hc_count)
        gvn = K.hc_norm(gv, self.W[p + "norm_conv.weight"], H, eps)
        out = SK.ple_conv(gvn, gv, st.ple_conv[li], self._wc[("ple", li)], d["psrc"])
        if self._write:
            ps = st.ple_conv[li]
            ps.copy_(torch.cat([ps[..., 1:], gvn.T.unsqueeze(0)], -1))
        else:
            self.rec[("ple", li)] = gvn
        return out

    def _gdn1(self, x, li, st):
        """The GDN layer of the rows: qkv|z and a|b, the conv over each row's window and the gates
        (`csrc`), the delta rule over the block's tree (`gctl`), the gated norm, out_proj."""
        from tools import fn_kernels as K
        p = self.pre(li) + "linear_attn."
        qkvz = self.lin(x, self.W[p + "qkvz"])
        ab = K.gemv_bf16(x, self.W[p + "ab"])
        return self.lin(self._gdn1_core(qkvz, ab, li, st), self.W[p + "out_proj.weight"])

    def _gdn1_core(self, qkvz, ab, li, st):
        """`_gdn1` past its input projections and before out_proj: the rows of one state (a joint
        pass runs the projections over every state's rows at once, `verify_multi`)."""
        from tools import fn_kernels as K
        from tools import fn_seq_kernels as SK
        c, d = self.cfg, self._cur["d"]
        p = self.pre(li) + "linear_attn."
        M = qkvz.shape[0]
        y, gg, beta = SK.gdn_pre(qkvz, st.conv[li], self._wc[li], d["csrc"], ab, self._wc[("alog", li)],
                                 self._wc[("dtb", li)], shift=self._write)
        o = SK.gdn_rows(y, c.key_dim, gg, beta, st.S[li], self._scratch(), d["gctl"], c.num_v_per_k, self._write,
                        pre=self._parked(li))
        if not self._write:
            self.rec[li] = (qkvz, y, gg, beta)
        z = qkvz[:, c.conv_dim:]
        hh = K.gated_norm(o.view(-1, c.linear_value_head_dim), z.reshape(-1, c.linear_value_head_dim),
                          self.W[p + "norm.weight"], c.rms_norm_eps, c.output_gate_type == "sigmoid")
        return hh.view(M, -1)

    def _attn1(self, x, li, st):
        """Attention of the rows: q|k|v and the indexer projection, one prep launch (norms, rope at
        each row's position, cache writes at each row's slot), attention over each row's visible keys
        (`actl`, `anc`), the output gate, o_proj."""
        c = self.cfg
        p = self.pre(li) + "self_attn."
        qkv = self.lin(x, self.W[p + "qkv"])
        iqk = self.lin(x, self.W[p + "indexer.index_qk_proj.weight"]) if c.has_indexer else None
        return self.lin(self._attn1_core(qkv, iqk, li, st), self.W[p + "o_proj.weight"])

    def _attn1_core(self, qkv, iqk, li, st):
        """`_attn1` past its input projections and before o_proj: the rows of one state."""
        from tools import fn_kernels as K
        from tools import fn_seq_kernels as SK
        c, d = self.cfg, self._cur["d"]
        p = self.pre(li) + "self_attn."
        D, Hq, Hk = c.head_dim, c.num_attention_heads, c.num_key_value_heads
        M = qkv.shape[0]
        cls = self._cls if c.has_indexer else 0
        ip = p + "indexer."
        q = K.attn_prep(qkv, iqk, d["pos"], self.W[p + "q_norm.weight"], self.W[p + "k_norm.weight"],
                        self.rope_cos, self.rope_sin, st.k[li], st.v[li], st.ik.get(li), Hq, Hk,
                        c.indexer_n_heads * c.indexer_head_dim if c.has_indexer else 0, c.rms_norm_eps,
                        slot=d["slot"], qln=self.W[ip + "q_layernorm.weight"] if cls else None,
                        iq=d["iq"] if cls else None)
        if c.has_indexer and (cls or self._write):
            # the pooled keys of the row's own blocks (from its own path); a decode step also keeps
            # its block in the shared pool (garbage until the block is complete, then its value)
            own = SK.qsa_pool(st.ik[li], d["oslot"].view(-1, 4), d["opos"].view(-1), self.W[ip + "k_layernorm.weight"],
                              self.rope_cos, self.rope_sin, c.rms_norm_eps)
            if self._write:
                st.pool[li].index_copy_(0, d["pidx"], own[:1])
            if cls:
                sc = self._cur.get(("sc", cls))
                if sc is None:
                    sc = self._cur[("sc", cls)] = torch.empty(M, cls, dtype=torch.float32, device=qkv.device)
                SK.qsa_scores(d["iq"], st.pool[li], own.view(M, SK.OWN, -1), d["qctl"], cls, out=sc)
                SK.qsa_select(sc, d["tail"], d["sel"])
        o = SK.attn_rows(q, st.k[li], st.v[li], d["actl"], d["anc"], d["sel"], bufs=self._cur["attn"])
        return K.gate_mul(o, qkv, D)

    # ------------------------------------------------------------------------- MTP (ENG-153)
    @torch.no_grad()
    def mtp_feed(self, stack: torch.Tensor, next_tokens: torch.Tensor, st_m: FNState):
        """The MTP layer over committed rows: `stack` [T, hc H] (the trunk's streams at positions
        st_m.length..) and the tokens that follow them. Returns (logits [T, vocab] of the token after
        next, the MTP layer's own streams for a deeper draft). The reference's `mtp` on the engine's
        primitives (EXL3 projections, int8 mixers, the expert kernels); eager."""
        return self.mtp(stack, next_tokens.to(self.device), st_m)

    @torch.no_grad()
    def mtp_chain(self, stack_row: torch.Tensor, next_token: int, st_m: FNState, depth: int) -> list[int]:
        """A greedy draft chain of `depth` tokens from one committed row and its successor: the
        first feed is real and stays in the MTP layer's cache; the deeper feeds are drafts and are
        rolled back (the cache is kept by length)."""
        logits, h = self.mtp_feed(stack_row, torch.tensor([next_token]), st_m)
        keep = st_m.length
        out = [int(logits[-1].argmax())]
        for _ in range(depth - 1):
            logits, h = self.mtp_feed(h[-1:], torch.tensor([out[-1]]), st_m)
            out.append(int(logits[-1].argmax()))
        st_m.length = keep
        st_m.clamp_pool()
        return out

    def new_mtp_state(self, cap: int | None = None, pages=None) -> FNState:
        """The MTP layer's cache (KV and indexer keys of its one attention layer), allocated up front;
        with `pages` (the trunk state's paged set) its rows are paged with the trunk's: the head never
        holds a row past the trunk's."""
        c, cap = self.cfg, cap or self.max_len
        st = FNState(cap=cap, pages=pages)
        st.k[MTP] = self._rows_tensor(st, (c.num_key_value_heads, cap, c.head_dim), 1, "k.mtp")
        st.v[MTP] = self._rows_tensor(st, (c.num_key_value_heads, cap, c.head_dim), 1, "v.mtp")
        if c.has_indexer:
            st.ik[MTP] = self._rows_tensor(st, (cap, c.indexer_head_dim), 0, "ik.mtp")
            st.pool[MTP] = self._pool_tensor(cap)
        return st

    def mtp_rows(self, st_m: FNState, key, stack_in: torch.Tensor, cls: int = 0):
        """The MTP layer over the rows of buffers `key` (tokens, positions, slots, ancestors staged by
        `stage_mtp`), from the streams `stack_in` [M, hc H]: the reference's input fusion (rms of the
        token's embedding -> fc_embedding, rms of all hc H streams -> fc_hidden per stream, added),
        its attention layer through the trunk's row kernels on the MTP cache, the MoE (bank -1), its
        own final mixer. Returns (x [M, H], the layer's streams [M, hc H]); the head is the caller's."""
        from tools import fn_kernels as K
        c = self.cfg
        H, hc, eps = c.hidden_size, c.hc_count, c.rms_norm_eps
        M = stack_in.shape[0]
        rb = self._rows_buffers(M, key)
        self._cur, self._write, self.rec, self._cls = rb, False, {}, cls
        d = rb["d"]
        e = self.rms(self.W["embed_tokens.weight"].index_select(0, d["tok"]).to(self.dtype),
                     self.W["mtp.pre_fc_norm_embedding.weight"], eps)
        e = self.lin(e, self.W["mtp.fc_embedding.weight"])
        hs = self.rms(stack_in, self.W["mtp.pre_fc_norm_hidden.weight"], eps)
        hs = self.lin(hs.reshape(M * hc, H), self.W["mtp.fc_hidden.weight"]).view(M, hc, H)
        h = (hs + e.unsqueeze(1)).flatten(1)
        p = self.pre(MTP)
        n = K.hc_norm(h, self.W[p + "attn_hyper_connection.hc_norm.weight"], H, eps)
        x, di = self._mix(n, p + "attn_hyper_connection")
        out = self._attn1(x, MTP, st_m)
        h, n = K.inject_norm(h, out, di, self.W[p + "attn_hyper_connection.rank"],
                             self.W[p + "mlp_hyper_connection.hc_norm.weight"], hc, eps)
        x, di = self._mix(n, p + "mlp_hyper_connection")
        yd, ids, wts = self.moe_parts(x, MTP)
        h, n = K.combine_inject_norm(h, yd, ids, wts, di, self.W[p + "mlp_hyper_connection.rank"],
                                     self.W["mtp.hyper_connection_mixer.hc_norm.weight"], hc, eps)
        dd = K.gemv(n, self.W["mtp.hyper_connection_mixer.down_inject"])
        x = K.mix_up(dd, n, self.W["mtp.hyper_connection_mixer.input_mix_weight_up.weight"], hc)
        return x, h

    def stage_mtp(self, st_m: FNState, nexts: list[int], depth: int) -> int:
        """Host work of a greedy chain draft at st_m.length: the feed of n = len(nexts) committed rows
        (their streams copied into the static input on the device by the caller), then depth - 1
        one-row draft steps after them, each at its own position (cache slot = position), and the
        pool rows the feed completes. Returns the context class of the draft (`qsa_class`)."""
        from tools import fn_seq_kernels as SK
        c = self.cfg
        n, L = len(nexts), st_m.length
        r = c.indexer_compress_ratio if c.has_indexer else 4
        cls = self.qsa_class(L + n + depth - 1)
        for j in range(depth):
            M = n if j == 0 else 1
            rb = self._rows_buffers(M, ("mtp", self._mtp_key, j, M))
            rb["copied"].synchronize()
            h = rb["h"]
            p0 = L if j == 0 else L + n + j - 1
            pos = p0 + np.arange(M)
            if j == 0:
                h["tok"].numpy()[:] = nexts
            h["pos"].numpy()[:] = pos + st_m.delta
            h["slot"].numpy()[:] = pos
            anc = h["anc"].numpy()
            for i in range(M):
                anc[i, :i + 1] = p0 + np.arange(i + 1)
            # the feed's rows own the blocks from its first; a draft step's shared pool ends where the
            # feed's committed rows end (the pool is refreshed after the feed)
            self._stage_qsa(h, p0, (L if j == 0 else L + n) // r, pos, anc, M, st_m.delta)
            rb["dbuf"].copy_(rb["hbuf"], non_blocking=True)
            rb["copied"].record()
        if c.has_indexer:
            b = self._mtp_bufs(depth)
            blk = L // r + np.arange(SK.OWN)
            h32 = b["h32"].numpy()
            h32[:SK.OWN * 4] = np.clip((r * blk)[:, None] + np.arange(r)[None, :], 0, st_m.cap - 1).reshape(-1)
            h32[SK.OWN * 4:] = r * blk + st_m.delta
            b["hpi"].numpy()[:] = np.clip(blk, 0, st_m.pool[MTP].shape[0] - 1)
            b["d32"].copy_(b["h32"], non_blocking=True)
            b["dpi"].copy_(b["hpi"], non_blocking=True)
        return cls

    def _mtp_bufs(self, depth: int) -> dict:
        """The MTP chain's static buffers: the committed rows' streams (input), and per draft step the
        top-2 tokens and their probabilities under the draft head (output, with pinned copies); one set
        a head (`_mtp_key`): a side slot's draft must not land in another slot's buffers between its
        launch and its read (ENG-241)."""
        bs = self.__dict__.setdefault("_mtpbs", {})
        b = bs.get(self._mtp_key)
        if b is None:
            c = self.cfg
            b = bs[self._mtp_key] = {"stack": torch.zeros(33, c.hc_count * c.hidden_size, dtype=self.dtype, device=self.device),
                              "tok": torch.zeros(32, 2, dtype=torch.long, device=self.device),
                              "prob": torch.zeros(32, 2, dtype=torch.float32, device=self.device),
                              "tok_h": torch.zeros(32, 2, dtype=torch.long, pin_memory=True),
                              "prob_h": torch.zeros(32, 2, dtype=torch.float32, pin_memory=True),
                              "logits": torch.zeros(1, c.vocab_size, dtype=torch.float32, device=self.device)}
            from tools import fn_seq_kernels as SK
            b["h32"] = torch.zeros(SK.OWN * 5, dtype=torch.int32, pin_memory=True)
            b["d32"] = torch.zeros(SK.OWN * 5, dtype=torch.int32, device=self.device)
            b["hpi"] = torch.zeros(SK.OWN, dtype=torch.int64, pin_memory=True)
            b["dpi"] = torch.zeros(SK.OWN, dtype=torch.int64, device=self.device)
        assert depth <= 32
        if self.draft_head is not None and ("hot", self.draft_ids.numel()) not in b:
            b[("hot", self.draft_ids.numel())] = torch.zeros(1, self.draft_ids.numel(), dtype=torch.float32,
                                                             device=self.device)
        return b

    draft_head = None          # Int8Rows over a hot vocabulary (`set_draft_head`), else the full head
    draft_ids = None

    def set_draft_head(self, ids) -> None:
        """Draft over a hot vocabulary `ids` (token ids; None: the full head): the head's rows,
        dequantised and stored as int8 rows with an fp32 scale (`tools.fn_kernels.gemv`), read in place
        of the full EXL3 head by the MTP drafts. Only the drafts change; the verify reads the served
        head. A head once built is kept for the process (captured draft graphs point into it)."""
        if ids is None:
            self.draft_head = self.draft_ids = None
            return
        from engine.models.flashnext.pack import Int8Rows
        heads = self.__dict__.setdefault("_draft_heads", {})
        key = hash(bytes(memoryview(np.ascontiguousarray(np.asarray(ids, dtype=np.int64)))))
        if key not in heads:
            idx = torch.as_tensor(np.asarray(ids, dtype=np.int64), device=self.device)
            w = self.W["lm_head.weight"].dequant()
            heads[key] = (Int8Rows.of(w.index_select(0, idx).half().contiguous()), idx)
            del w
            torch.cuda.empty_cache()
        self.draft_head, self.draft_ids = heads[key]

    def mtp_chain_dev(self, st_m: FNState, n: int, depth: int, cls: int = 0) -> None:
        """A greedy chain of `depth` drafts on the device (`stage_mtp` first, the n committed rows'
        streams in `_mtp_bufs()["stack"]`): the feed, then each draft's token and streams into the
        next step without the host. Per step the top-2 tokens and their probabilities under the draft
        head land in `tok` / `prob` and their pinned copies (the second token is a sibling candidate)."""
        from tools import fn_kernels as K
        from tools import fn_seq_kernels as SK
        c = self.cfg
        b = self._mtp_bufs(depth)
        x, h = self.mtp_rows(st_m, ("mtp", self._mtp_key, 0, n), b["stack"][:n], cls)
        if c.has_indexer:                     # the blocks the feed completed, into the MTP pool
            ip = self.pre(MTP) + "self_attn.indexer."
            own = SK.qsa_pool(st_m.ik[MTP], b["d32"][:SK.OWN * 4].view(SK.OWN, 4), b["d32"][SK.OWN * 4:],
                              self.W[ip + "k_layernorm.weight"], self.rope_cos, self.rope_sin, c.rms_norm_eps)
            st_m.pool[MTP].index_copy_(0, b["dpi"], own)
        for j in range(depth):
            if self.draft_head is not None:
                lg = b[("hot", self.draft_ids.numel())]
                K.gemv(x[-1:], self.draft_head, out=lg)
            else:
                lg = b["logits"]
                K.dense(x[-1:], self.W["lm_head.weight"], out=lg)
            vals, ids = torch.topk(lg[0], 2)
            z = torch.exp(lg[0] - vals[0]).sum()
            b["prob"][j].copy_(torch.exp(vals - vals[0]) / z)
            b["tok"][j].copy_(self.draft_ids.index_select(0, ids) if self.draft_head is not None else ids)
            if j + 1 == depth:
                break
            self._rows_buffers(1, ("mtp", self._mtp_key, j + 1, 1))["d"]["tok"].copy_(b["tok"][j, :1])
            x, h = self.mtp_rows(st_m, ("mtp", self._mtp_key, j + 1, 1), h[-1:], cls)
        b["tok_h"].copy_(b["tok"], non_blocking=True)
        b["prob_h"].copy_(b["prob"], non_blocking=True)

    # ------------------------------------------------------------------------- generation
    prefill_path = PREFILL           # "kernels" (ENG-225) or "ref"

    @torch.no_grad()
    def prefill(self, ids: torch.Tensor, st: FNState, chunk: int | None = None, keep_stack: bool = False,
                path: str | None = None) -> torch.Tensor:
        """The prompt in chunks; logits of the last row. `path`: "kernels" (`forward_pre`, ENG-225) or
        "ref" (the reference's batched forward); default `prefill_path`. `keep_stack`: every row's
        pre-final-mixer streams in `self.prompt_stack` [T, hc H] (the MTP layer's input; `self.stack`
        holds the last chunk's only)."""
        self.flush()
        path = path or self.prefill_path
        if path != "kernels" and (st.mm is not None or st.delta):
            raise NotImplementedError("images on the reference prefill path (QWEN38_FN_PREFILL=ref): the "
                                      "prefill kernels take them")
        fwd = self.forward_pre if path == "kernels" else self.forward
        chunk = chunk or (PREFILL_CHUNK if path == "kernels" else 2048)
        out = None
        stacks = []
        for i in range(0, ids.numel(), chunk):
            last = i + chunk >= ids.numel()
            out = fwd(ids[i:i + chunk].to(self.device), st, head="last" if last else "none")
            if keep_stack:
                stacks.append(self.stack)
            if st.length > 4096:
                # the reference attention's score blocks grow with the context; cached blocks of one
                # chunk fit no request of the next (a 32k prompt held tens of GB, 2026-10-11)
                torch.cuda.empty_cache()
        self.prompt_stack = torch.cat(stacks) if keep_stack else None
        if st.pool and path != "kernels":            # the kernel path pools every chunk's blocks itself
            self.build_pool(st)
        return out[-1:]

    @torch.no_grad()
    def mtp_prime(self, stack: torch.Tensor, next_tokens: list[int], st_m: FNState, path: str | None = None) -> None:
        """The MTP layer's cache over committed rows: `stack` [T, hc H] (the trunk's streams at
        positions st_m.length..) and the tokens that follow them, in chunks; on the prefill kernels
        (`mtp_pre`) or the reference's `mtp_feed` with the pool built after."""
        path = path or self.prefill_path
        T = len(next_tokens)
        chunk = PREFILL_CHUNK if path == "kernels" else 2048
        for i in range(0, T, chunk):
            j = min(i + chunk, T)
            nt = torch.tensor(next_tokens[i:j])
            if path == "kernels":
                self.mtp_pre(stack[i:j], nt, st_m)
            else:
                self.mtp_feed(stack[i:j], nt, st_m)
            if st_m.length > 4096:
                torch.cuda.empty_cache()           # as `prefill`
        if path != "kernels":
            self.build_pool(st_m, layers=[MTP])

    @torch.no_grad()
    def mtp_pre(self, stack: torch.Tensor, next_tokens: torch.Tensor, st_m: FNState) -> torch.Tensor:
        """The MTP layer over a chunk of committed rows on the prefill kernels: the reference's input
        fusion, its attention and MoE through `_attn_pre` / `_moe_pre`; the cache advances. Returns the
        layer's streams [T, hc H] (the final mixer and the head are the drafts' business)."""
        from tools import fn_kernels as K
        c = self.cfg
        H, hc, eps = c.hidden_size, c.hc_count, c.rms_norm_eps
        T = next_tokens.numel()
        self._pre = self._pre_ctl(st_m.length, T, st_m)
        e = self.rms(self.embed(next_tokens.view(-1)), self.W["mtp.pre_fc_norm_embedding.weight"], eps)
        e = self.lin(e, self.W["mtp.fc_embedding.weight"])
        hs = self.rms(stack, self.W["mtp.pre_fc_norm_hidden.weight"], eps)
        hs = self.lin(hs.reshape(T * hc, H), self.W["mtp.fc_hidden.weight"]).view(T, hc, H)
        h = (hs + e.unsqueeze(1)).flatten(1)
        p = self.pre(MTP)
        n = K.hc_norm(h, self.W[p + "attn_hyper_connection.hc_norm.weight"], H, eps)
        x, di = self._mix_pre(n, p + "attn_hyper_connection")
        out = self._attn_pre(x, MTP, st_m)
        h, n = K.inject_norm(h, out, di, self.W[p + "attn_hyper_connection.rank"],
                             self.W[p + "mlp_hyper_connection.hc_norm.weight"], hc, eps)
        x, di = self._mix_pre(n, p + "mlp_hyper_connection")
        h, _ = K.inject_norm(h, self._moe_pre(x, MTP), di, self.W[p + "mlp_hyper_connection.rank"], None, hc, eps)
        st_m.length += T
        self._pre = None
        return h

    @torch.no_grad()
    def decode_eager(self, token: int, st: FNState) -> torch.Tensor:
        """One token through `step` (no graph)."""
        self.settle(st)
        self.stage(token, st)
        out = self.step(st)
        self._pend_owner = None
        st.length += 1
        return out


class DecodeGraph:
    """`FlashNextEngine.step` captured once for a state and a context class (`qsa_class`: dense, or the
    width of the indexer's block scores); `__call__(token)` stages the token (host), replays the
    class's graph (captured when the length first enters it), and moves the length. `max_lc` is kept
    for the callers; the step reads the length from the device."""

    def __init__(self, eng: FlashNextEngine, st: FNState, max_lc: int = 2048):
        self.eng, self.st, self.max_lc = eng, st, max_lc
        eng._rows_buffers(1)
        self.graphs: dict = {}
        self.stacks: dict = {}                 # class -> the graph's streams output [1, hc H]
        self.graph = None
        self.captures = 0                      # classes captured lazily by a step

    def capture(self, cls: int | None = None) -> None:
        eng, st = self.eng, self.st
        cls = eng.qsa_class(st.length) if cls is None else cls
        eng.settle(st)
        saved = _state_saved(eng, st)
        saved_n = eng._pend_n.clone()
        pctx = dict(st.ple_ctx)
        length = st.length
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                eng.stage(0, st)
                eng.step(st, cls)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with capture_lock.LOCK, torch.cuda.graph(g):
            eng.step(st, cls)
        self.stacks[cls] = eng.stack
        torch.cuda.synchronize()
        _state_restore(saved)                           # the warm-up steps wrote the state: undo
        eng._pend_n.copy_(saved_n)
        st.ple_ctx.clear()
        st.ple_ctx.update(pctx)
        st.length = length
        self.graphs[cls] = self.graph = g

    def _ready(self) -> None:
        cls = self.eng.qsa_class(self.st.length)
        if cls not in self.graphs:
            self.capture(cls)
            self.captures += 1              # this step's time is the capture's (a price skips it)
        self.graph = self.graphs[cls]

    @torch.no_grad()
    def greedy(self, token: int) -> int:
        """One greedy step: the next token, read from the graph's own argmax (no extra launch)."""
        eng, st = self.eng, self.st
        self._ready()
        eng.settle(st)
        t0 = time.perf_counter()
        eng.stage(token, st)
        eng.host_ms["stage"] = eng.host_ms.get("stage", 0.0) + 1e3 * (time.perf_counter() - t0)
        self.graph.replay()
        eng._pend_owner = None
        st.length += 1
        torch.cuda.current_stream().synchronize()
        return int(eng.g["next_h"][0])

    @torch.no_grad()
    def __call__(self, token: int) -> torch.Tensor:
        eng, st = self.eng, self.st
        self._ready()
        eng.settle(st)
        eng.stage(token, st)
        self.graph.replay()
        eng._pend_owner = None
        st.length += 1
        return eng.g["logits"]


class VerifyGraph:
    """`FlashNextEngine.rows_step` of an M-row verify block captured once for a state: every block
    of M rows (chain or tree; positions, slots, ancestors and the GDN tile's walk are device inputs)
    replays it. `replay()` after `stage_rows`; the commit then reads the graph's own intermediates."""

    def __init__(self, eng: FlashNextEngine, st: FNState, M: int, cls: int = 0):
        self.eng, self.st, self.M, self.cls = eng, st, M, cls
        self.graph = None
        self.rec: dict = {}

    def capture(self) -> None:
        eng, st, M = self.eng, self.st, self.M
        eng.stage_rows([0] * M, list(range(-1, M - 1)), st)   # writes only KV rows past the length
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                eng.rows_step(st, M, cls=self.cls)
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with capture_lock.LOCK, torch.cuda.graph(self.graph):
            eng.rows_step(st, M, cls=self.cls)
        torch.cuda.synchronize()
        self.rec = eng.rec

    def replay(self) -> None:
        self.graph.replay()
        self.eng.rec = self.rec


class CommitGraph:
    """`FlashNextEngine.commit_dev` for a verify block of M rows and a path of n rows (moved or not),
    captured once after that block's VerifyGraph (it reads the verify graph's intermediates). The
    warm-up runs write the state: it is saved and restored around them."""

    def __init__(self, eng: FlashNextEngine, st: FNState, key: tuple):
        self.eng, self.st, self.key = eng, st, key
        self.graph = None

    def capture(self) -> None:
        eng, st = self.eng, self.st
        saved = _state_saved(eng, st)
        saved_n = eng._pend_n.clone()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            eng.commit_dev(st, self.key)
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with capture_lock.LOCK, torch.cuda.graph(self.graph):
            eng.commit_dev(st, self.key)
        torch.cuda.synchronize()
        _state_restore(saved)
        eng._pend_n.copy_(saved_n)


class MTPChainGraph:
    """`FlashNextEngine.mtp_chain_dev` for a feed of n rows and a chain depth, captured once for an MTP
    cache: the whole draft (feed, depth - 1 deeper steps, every head and argmax) is one replay."""

    def __init__(self, eng: FlashNextEngine, st_m: FNState, n: int, depth: int, cls: int = 0):
        self.eng, self.st_m, self.n, self.depth, self.cls = eng, st_m, n, depth, cls
        self.graph = None

    def capture(self) -> None:
        eng, st_m = self.eng, self.st_m
        # writes only cache rows past the length, and pool rows the next feed writes again
        eng.stage_mtp(st_m, [0] * self.n, self.depth)
        saved = {k: v.clone() for k, v in st_m.pool.items()}
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                eng.mtp_chain_dev(st_m, self.n, self.depth, self.cls)
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with capture_lock.LOCK, torch.cuda.graph(self.graph):
            eng.mtp_chain_dev(st_m, self.n, self.depth, self.cls)
        torch.cuda.synchronize()
        for k, v in st_m.pool.items():
            v.copy_(saved[k])


#: the rows past the length a captured step, verify block or commit may write (a block is at most 33 rows)
CAPTURE_ROWS = 64


def _state_saved(eng: FlashNextEngine, st: FNState) -> list:
    """What a capture's warm-up may write, cloned: the recurrent sections whole, the KV and indexer-key
    rows from the length on (`CAPTURE_ROWS`) and the pooled-key blocks from the length's block on.
    Cloning the whole buffers took 8 GB at a 262,144-token window, and with the KV pool the rows past
    the committed ones are not mapped at all (ENG-241)."""
    L = st.length
    r = eng.cfg.indexer_compress_ratio if eng.cfg.has_indexer else 4
    out = []
    for name, t in _state_tensors(st).items():
        kind = name.split(".", 1)[0]
        if kind in ("k", "v"):
            out.append((t, 1, L, min(CAPTURE_ROWS, t.shape[1] - L)))
        elif kind == "ik":
            out.append((t, 0, L, min(CAPTURE_ROWS, t.shape[0] - L)))
        elif kind == "pool":
            b = L // r
            out.append((t, 0, b, min(CAPTURE_ROWS // r + 2, t.shape[0] - b)))
        else:
            out.append((t, 0, 0, t.shape[0]))
    return [(t, ax, lo, n, t.narrow(ax, lo, n).clone()) for t, ax, lo, n in out if n > 0]


def _state_restore(saved: list) -> None:
    for t, ax, lo, n, x in saved:
        t.narrow(ax, lo, n).copy_(x)


def _state_tensors(st: FNState) -> dict:
    out = {}
    for name, d in (("S", st.S), ("conv", st.conv), ("k", st.k), ("v", st.v), ("ik", st.ik), ("pconv", st.ple_conv),
                    ("pool", st.pool)):
        for li, t in d.items():
            out[f"{name}.{li}"] = t
    return out
