"""LFM2 MoE on the board (ENG-287): the reference forward with its weight reads on the kernels, a row
step that serves a decode token and a verify block of up to 32 rows with the same per-row arithmetic,
and the step as a CUDA graph.

`Lfm2MoeRef` (ref.py) is the model, checked against `transformers`. This class keeps its structure:

  rows_step(st, M)   M rows whose tokens, positions, cache slots, attention control, ancestors and
                     conv windows sit in device buffers (`stage` for a decode token, `stage_rows` for
                     a draft block in DFS pre-order), every part on a row-invariant kernel:
                       projections        tools/fn_kernels.dense (EXL3, up to 32 rows) or
                                          kernels.linear_rows (BF16)
                       norms, short conv, q/k prep, attention, router   kernels.py
                       experts            the bank kernels of tools/fn_kernels (EXL3) or
                                          kernels.moe_bf16, combined by fn_kernels.moe_combine
                     so the row of a verify block is the decode step's row at that position bit for
                     bit, and a greedy speculative run writes the plain greedy text. A decode step
                     shifts the conv state in place; a verify block writes nothing but its KV rows
                     past the length and keeps each conv layer's B * x rows for the commit.
  commit(st, path)   the conv states from the path's last rows (copies), the path's KV rows moved
                     down, the length moved (`commit_dev` is capturable)
  prefill(tokens)    a prompt in chunks of PREFILL_CHUNK on batched kernels (cuBLAS or the EXL3
                     library GEMM, the grouped EXL3 experts, SDPA with a lower-right causal mask):
                     the next token agrees with the row path, the bits do not

`DecodeGraph` captures the decode step once per attention class (kernels.classes: the number of
512-key programs, which never changes a row's bits) and `VerifyGraph` an M-row block.
"""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F

from engine import capture_lock
from engine.models.lfm2_moe import kernels as K
from engine.models.lfm2_moe.ref import Lfm2MoeRef, LfmState
from engine.settings import SETTINGS as _S

#: rows a prompt chunk takes (the caches' anchor grid, `serving.Lfm2MoeServing.caps`)
PREFILL_CHUNK = 4096
#: rows up to which a projection runs on the row-invariant kernels (their limit, the verify width)
ROWS_MAX = 32
#: rows a captured step, verify block or commit may write past the length (a block is at most 33 rows)
CAPTURE_ROWS = 64
#: the bit widths tools/fn_kernels.dense decodes (the half-integer rates as ten times the rate)
DENSE_K = (2, 3, 4, 5, 6, 8, 15, 25, 35)


def _is_exl3(w) -> bool:
    return not torch.is_tensor(w)


class _NoEvent:
    """A CPU engine's stand-in for a CUDA event (the tests under TRITON_INTERPRET=1)."""

    def record(self) -> None:
        pass

    def synchronize(self) -> None:
        pass


def _event(dev):
    return torch.cuda.Event() if dev.type == "cuda" else _NoEvent()


def _sync(dev) -> None:
    if dev.type == "cuda":
        torch.cuda.current_stream().synchronize()


class Lfm2MoeEngine(Lfm2MoeRef):

    def __init__(self, cfg, W: dict, experts: dict, device="cuda", max_len: int = 32768):
        super().__init__(cfg, W, dtype=torch.bfloat16, device=device, experts=experts)
        c = cfg
        self.max_len = max_len
        cos, sin = Lfm2MoeRef.rope(self, torch.arange(max_len, device=self.device))
        self.rope_cos, self.rope_sin = cos.contiguous(), sin.contiguous()        # [max_len, D] bf16
        self.classes = K.classes(max_len)
        self._rb: dict = {}
        self.rec: dict = {}
        self.host_ms: dict = {}
        self._wc = {li: W[self.pre(li) + "conv.conv.weight"].reshape(c.hidden_size, -1).to(torch.bfloat16).contiguous()
                    for li in c.conv_layers}
        self._gate = {li: W[self.pre(li) + "feed_forward.gate.weight"].to(torch.bfloat16).contiguous()
                      for li in c.moe_layers}
        # the router's 64 x 2048 weight on tools/fn_kernels.gemv (K split over a CTA's warps, rows
        # invariant) where fp16 holds it exactly; a 16-row tile GEMM has 4 programs for 64 outputs
        self._gate16 = {}
        for li, g in self._gate.items():
            h = g.to(torch.float16)
            if self.device.type == "cuda" and torch.equal(h.to(torch.bfloat16), g):
                self._gate16[li] = h.contiguous()
        self._bias = {li: W[self.pre(li) + "feed_forward.expert_bias"].float().contiguous()
                      for li in c.moe_layers if c.use_expert_bias}
        self.exl3 = any(_is_exl3(w) for w in W.values())
        self.banks = {li: e.bank(self.device) for li, e in experts.items() if hasattr(e, "bank")} \
            if str(self.device).startswith("cuda") else {}

    # ------------------------------------------------------------------------- state
    def new_state(self, cap: int | None = None, paged: bool = False) -> LfmState:
        """A state with every tensor the step writes allocated up front (a graph holds them). `paged`
        with the KV pool on (engine/kvpool.py, ENG-241): K and V are reserved as addresses and committed
        as rows are written, by the caller before every write (`Lfm2MoeServing._ensure`)."""
        from engine import kvpool as _kvp
        c = self.cfg
        cap = cap or self.max_len
        st = LfmState(cap=cap)
        if paged and _kvp.POOL is not None and cap >= _kvp.PAGED_MIN_ROWS and str(self.device).startswith("cuda"):
            st.pages = _kvp.PagedSet(_kvp.OWNER)
        for li in range(c.num_hidden_layers):
            if c.is_attention(li):
                for kind, d in (("k", st.k), ("v", st.v)):
                    shape = (c.num_key_value_heads, cap, c.head_dim)
                    if st.pages is None:
                        d[li] = torch.zeros(*shape, dtype=self.dtype, device=self.device)
                    else:
                        d[li] = st.pages.add(_kvp.PagedBuffer(shape, 1, self.dtype, self.device, owner=st.pages.owner,
                                                              name=f"lfm.{kind}.{li}")).tensor
            else:
                st.conv[li] = torch.zeros(c.conv_state_len, c.hidden_size, dtype=self.dtype, device=self.device)
        return st

    def reset(self, st: LfmState) -> None:
        st.reset_inplace()

    # ------------------------------------------------------------------------- primitives
    def lin(self, x: torch.Tensor, w) -> torch.Tensor:
        from tools import fn_kernels as FK
        from tools.exl3_linear import exl3_matmul
        x2 = x.reshape(-1, x.shape[-1])
        if not x2.is_contiguous():
            x2 = x2.contiguous()
        if _is_exl3(w):
            # fn_kernels.dense takes K 2..6 and 8; a K-7 tensor (a 6.0 pack's) takes the library's skinny
            # kernel, whose split is the shape's too (row-invariant up to 32 rows, tools/exl3_linear.py)
            y = FK.dense(x2, w) if x2.shape[0] <= ROWS_MAX and w.kb in DENSE_K else exl3_matmul(x2, w)
            return y.view(*x.shape[:-1], w.N)
        if x2.shape[0] <= ROWS_MAX:
            return K.linear_rows(x2, w).view(*x.shape[:-1], w.shape[0])
        return F.linear(x2, w).view(*x.shape[:-1], w.shape[0])

    def _fused(self, x: torch.Tensor, p: str, name: str, parts: tuple) -> torch.Tensor:
        """The fused projection `p + name` (one launch), or its parts one after the other where the
        loader could not fuse them (EXL3 members at different bit widths)."""
        w = self.W.get(p + name)
        if w is not None:
            return self.lin(x, w)
        return torch.cat([self.lin(x, self.W[p + q + ".weight"]) for q in parts], -1)

    def head(self, n: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        """Logits [M, vocab] fp32 of the final normed rows (a static `out` in a graph)."""
        from tools import fn_kernels as FK
        from tools.exl3_linear import exl3_matmul
        w = self.head_weight()
        M = n.shape[0]
        if _is_exl3(w):
            if M <= ROWS_MAX and w.kb in DENSE_K:
                return FK.dense(n, w, out=out)
            return exl3_matmul(n, w, out=out)
        if M <= ROWS_MAX:
            return K.linear_rows(n, w, out=out, f32=True)
        y = F.linear(n, w).float()
        return y if out is None else out.copy_(y)

    def moe_rows(self, n: torch.Tensor, li: int) -> torch.Tensor:
        """The MoE of M <= 32 rows: the router, every (row, slot) pair through its expert, the combine
        in ascending expert id."""
        from tools import fn_kernels as FK
        c = self.cfg
        g16 = self._gate16.get(li)
        logits = FK.gemv(n, g16) if g16 is not None else K.linear_rows(n, self._gate[li], f32=True)
        ids, wts = K.router(logits, self._bias.get(li), c.num_experts_per_tok, c.routed_scaling_factor)
        if li in self.banks:
            yd = FK.moe_experts(n, ids, self.banks[li], order=self._order1 if n.shape[0] == 1 else None)
        else:
            e = self.experts[li]
            yd = K.moe_bf16(n, ids, e.w13, e.w2)
        if self.route_trace is not None:
            self.route_trace.append((li, ids.long()))
        return FK.moe_combine(yd, ids, wts) if self.device.type == "cuda" else K.combine_reference(yd, ids, wts)

    route_trace = None

    @property
    def _order1(self):
        if not hasattr(self, "_o1"):
            self._o1 = torch.empty(0, dtype=torch.int32, device=self.device)
        return self._o1

    # ------------------------------------------------------------------------- rows: decode and verify
    def _rows_buffers(self, M: int, key=None) -> dict:
        """The device inputs of an M-row call and their pinned host image (one H2D copy a call):

          tok, pos, slot   int64 [M]: the token, its position (block start + depth), its cache slot
          ctl              int32 [M, 2]: visible keys, block start
          anc              int32 [M, ANC]: the cache slot of the row's ancestor at each depth
          csrc             int32 [M, KT - 1]: where the row's earlier conv inputs live (a row of the
                           block, or -1 - a state column)
        and the outputs (logits, greedy tokens and their pinned copy, the final normed rows) and each
        conv layer's B * x rows (what a commit keeps)."""
        key = M if key is None else key
        rb = self._rb.get(key)
        if rb is not None:
            return rb
        c, dev = self.cfg, self.device
        S = c.conv_state_len
        n64 = 3 * M
        n32 = M * (2 + K.ANC + S)
        n32 += n32 % 2
        nb = 8 * n64 + 4 * n32

        def views(b):
            i64 = b[:8 * n64].view(torch.int64)
            i32 = b[8 * n64:].view(torch.int32)
            v = {"tok": i64[:M], "pos": i64[M:2 * M], "slot": i64[2 * M:3 * M]}
            k = 0
            for name, w in (("ctl", 2), ("anc", K.ANC), ("csrc", S)):
                v[name] = i32[k:k + M * w].view(M, w)
                k += M * w
            return v
        pin = dev.type == "cuda"
        hbuf = torch.zeros(nb, dtype=torch.uint8, pin_memory=pin)
        dbuf = torch.zeros(nb, dtype=torch.uint8, device=dev)
        copied = _event(dev)
        copied.record()
        rb = {"M": M, "hbuf": hbuf, "dbuf": dbuf, "h": views(hbuf), "d": views(dbuf), "copied": copied,
              "logits": torch.zeros(M, c.vocab_size, dtype=torch.float32, device=dev),
              "next": torch.zeros(M, dtype=torch.long, device=dev),
              "next_h": torch.zeros(M, dtype=torch.long, pin_memory=pin),
              "hidden": torch.zeros(M, c.hidden_size, dtype=self.dtype, device=dev),
              "bx": {li: torch.zeros(M, c.hidden_size, dtype=self.dtype, device=dev) for li in c.conv_layers},
              "attn": {}}
        if M == 1:
            rb["h"]["csrc"].numpy()[:] = K.window_src([-1], c.conv_L_cache)[0]
        self._rb[key] = rb
        return rb

    @staticmethod
    def _key(slot, M: int):
        """The row buffers' key of an M-row call of a sequence slot (None: slot 0's, keyed by M alone)."""
        return M if slot is None else (slot, M)

    def stage(self, token: int, st: LfmState, slot=None) -> None:
        """Host work of one decode step: the token, its position and slot, its attention control."""
        rb = self._rows_buffers(1, self._key(slot, 1))
        rb["copied"].synchronize()            # the image is the previous call's copy source until then
        h, L = rb["h"], st.length
        h["tok"].numpy()[0] = token
        h["pos"].numpy()[0] = L
        h["slot"].numpy()[0] = L
        h["ctl"].numpy()[0] = (L + 1, L)
        h["anc"].numpy()[0, 0] = L
        rb["dbuf"].copy_(rb["hbuf"], non_blocking=True)
        rb["copied"].record()

    def stage_rows(self, tokens: list[int], parents: list[int], st: LfmState, key=None) -> None:
        """Host work of a verify block: `tokens` in DFS pre-order with `parents` (-1 for the first
        row, which sits at the state's length). Nothing in `st` changes; `commit` keeps a path."""
        M = len(tokens)
        if M > ROWS_MAX + 1:
            raise ValueError(f"a verify block of {M} rows: the row kernels take {ROWS_MAX + 1} at most")
        rb = self._rows_buffers(M, key)
        rb["copied"].synchronize()
        h, L = rb["h"], st.length
        paths = []
        for i, p in enumerate(parents):
            if p >= i:
                raise ValueError(f"row {i}: parent {p} is not before it (DFS pre-order)")
            paths.append([i] if p < 0 else paths[p] + [i])
        d = np.asarray([len(pa) - 1 for pa in paths], dtype=np.int64)
        h["tok"].numpy()[:] = tokens
        h["pos"].numpy()[:] = L + d
        h["slot"].numpy()[:] = L + np.arange(M)
        h["ctl"].numpy()[:, 0] = L + d + 1
        h["ctl"].numpy()[:, 1] = L
        anc = h["anc"].numpy()
        for i, pa in enumerate(paths):
            anc[i, :len(pa)] = L + np.asarray(pa)
        h["csrc"].numpy()[:] = K.window_src(parents, self.cfg.conv_L_cache)
        rb["dbuf"].copy_(rb["hbuf"], non_blocking=True)
        rb["copied"].record()
        # what a commit needs: the block, and the buffers whose B * x rows it keeps
        st.block = (list(tokens), list(parents), paths, M if key is None else key)

    def rows_step(self, st: LfmState, M: int, write: bool = False, ns: int | None = None, key=None) -> torch.Tensor:
        """M rows from the device buffers (`stage` / `stage_rows` first) through every layer and the head.
        `write` (one row, a decode step): the conv states shift in place. Otherwise only the block's KV
        rows past the length are written and each conv layer's B * x rows are kept (`rb["bx"]`). `ns`:
        the attention class (kernels.class_of). Returns logits [M, vocab] fp32 (a static buffer) and leaves
        the greedy tokens in rb["next"] and its pinned host copy; `st.length` is the caller's."""
        rb = self._rows_buffers(M, key)
        self._rows([(st, rb, M, ns or K.class_of(st.length + M, st.cap))], write)
        return rb["logits"]

    def _rows(self, parts: list, write: bool = False) -> None:
        """The rows of one or more states [(state, its row buffers, rows, attention class)] in one pass:
        the projections, the experts and the head over every row at once (their weights read once), the
        short conv and attention per state on its own rows and caches. Every kernel is row-invariant, so a
        state's rows get the bits of its lone call. Each part's logits, greedy tokens and final rows land
        in its buffers."""
        c = self.cfg
        eps = c.rms_norm_eps
        cuts = [0]
        for _, _, M, _ in parts:
            cuts.append(cuts[-1] + M)
        R = cuts[-1]
        sl = [slice(cuts[j], cuts[j + 1]) for j in range(len(parts))]
        one = len(parts) == 1
        tok = parts[0][1]["d"]["tok"] if one else torch.cat([rb["d"]["tok"] for _, rb, _, _ in parts])
        h = self.W["embed_tokens.weight"].index_select(0, tok)
        h, n = K.add_rms(h, None, self.W["layers.0.operator_norm.weight"], eps)
        for li in range(c.num_hidden_layers):
            p = self.pre(li)
            if c.is_attention(li):
                a = p + "self_attn."
                qkv = self._fused(n, a, "qkv", ("q_proj", "k_proj", "v_proj"))
                o = torch.empty(R, c.num_attention_heads, c.head_dim, dtype=self.dtype, device=self.device)
                for (st, rb, M, ns), r in zip(parts, sl):
                    d = rb["d"]
                    q = K.qk_prep(qkv[r], d["pos"], d["slot"], self.W[a + "q_layernorm.weight"],
                                  self.W[a + "k_layernorm.weight"], self.rope_cos, self.rope_sin, st.k[li], st.v[li],
                                  c.num_attention_heads, c.num_key_value_heads, eps)
                    # the split buffers of this class: a graph of another class holds its own, so a
                    # capture never frees what an earlier graph replays into
                    K.attn_rows(q, st.k[li], st.v[li], d["ctl"], d["anc"], ns, bufs=rb["attn"].setdefault(ns, {}),
                                out=o[r])
                out = self.lin(o.view(R, -1), self.W[a + "out_proj.weight"])
            else:
                bcx = self.lin(n, self.W[p + "conv.in_proj.weight"])
                y = torch.empty(R, c.hidden_size, dtype=self.dtype, device=self.device)
                for (st, rb, M, ns), r in zip(parts, sl):
                    K.conv_rows(bcx[r], st.conv[li], self._wc[li], rb["d"]["csrc"], rb["bx"][li], y=y[r], write=write)
                out = self.lin(y, self.W[p + "conv.out_proj.weight"])
            h, n = K.add_rms(h, out, self.W[p + "ffn_norm.weight"], eps)
            if c.is_moe(li):
                m = self.moe_rows(n, li)
            else:
                f = p + "feed_forward."
                gu = self._fused(n, f, "w13", ("w1", "w3"))
                I = c.intermediate_size
                m = self.lin(F.silu(gu[:, :I]) * gu[:, I:], self.W[f + "w2.weight"])
            nxt = self.W[self.pre(li + 1) + "operator_norm.weight"] if li + 1 < c.num_hidden_layers \
                else self.W["embedding_norm.weight"]
            h, n = K.add_rms(h, m, nxt, eps)
        if one:
            rb = parts[0][1]
            rb["hidden"].copy_(n)
            self.head(n, out=rb["logits"])
        else:
            logits = self.head(n)
            for (_, rb, _, _), r in zip(parts, sl):
                rb["hidden"].copy_(n[r])
                rb["logits"].copy_(logits[r])
        for _, rb, _, _ in parts:
            torch.argmax(rb["logits"], dim=-1, out=rb["next"])
            rb["next_h"].copy_(rb["next"], non_blocking=True)

    def step(self, st: LfmState, ns: int | None = None, slot=None) -> torch.Tensor:
        """One decode token from the device buffers (`stage` first): the conv states shift in place, the
        KV row lands at the length; `st.length` is the caller's. Logits [1, vocab] fp32 (static)."""
        return self.rows_step(st, 1, write=True, ns=ns, key=self._key(slot, 1))

    @property
    def g(self) -> dict:
        return self._rb.get(1, {})

    @torch.no_grad()
    def verify(self, tokens: list[int], parents: list[int], st: LfmState, graphs: dict | None = None,
               slot=None) -> list[int]:
        """The greedy token at every row of a draft block (DFS pre-order), eager or from `graphs`
        ((M, class) -> VerifyGraph). Nothing in `st` moves until `commit`. `slot`: a side slot's own
        row buffers (None: slot 0's)."""
        M = len(tokens)
        key = self._key(slot, M)
        ns = K.class_of(st.length + M, st.cap)
        gr = None
        if graphs is not None:
            gr = graphs.get((M, ns))
            if gr is None:
                gr = graphs[(M, ns)] = VerifyGraph(self, st, M, ns, key)
                gr.capture()
        t0 = time.perf_counter()
        self.stage_rows(tokens, parents, st, key=key)
        self.host_ms["stage_rows"] = self.host_ms.get("stage_rows", 0.0) + 1e3 * (time.perf_counter() - t0)
        if gr is not None:
            gr.replay()
        else:
            self.rows_step(st, M, ns=ns, key=key)
        self._width = M
        _sync(self.device)
        return self._rb[key]["next_h"].tolist()

    @torch.no_grad()
    def verify_multi(self, items: list) -> list:
        """ENG-241's joint pass for LFM2: the verify blocks of several states (sequence slots over these
        weights) [(tokens, parents, state)] in one pass of at most 32 rows (`_rows`): each state's
        greedy tokens are its lone verify's, bit for bit. Eager; each state commits its block
        (`commit`). Returns per state its row buffers (logits, next, hidden)."""
        if sum(len(t) for t, _, _ in items) > ROWS_MAX:
            raise ValueError(f"a joint pass of {sum(len(t) for t, _, _ in items)} rows: at most {ROWS_MAX}")
        parts = []
        for j, (tokens, parents, st) in enumerate(items):
            M = len(tokens)
            key = ("joint", j, M)
            self.stage_rows(tokens, parents, st, key=key)
            parts.append((st, self._rows_buffers(M, key), M, K.class_of(st.length + M, st.cap)))
        self._rows(parts)
        _sync(self.device)
        return [rb for _, rb, _, _ in parts]

    @torch.no_grad()
    def decode_multi(self, items: list, graphs: dict | None = None) -> list:
        """One decode token of each of several states [(token, state, slot)] in one pass (`_rows`, the
        weights read once), every state advanced in place as by its own decode step (its conv states
        shift, its KV row lands at its length; `st.length` is the caller's), each row the bits of that
        state's lone decode step. From `graphs` (engine/models/graph_lru.py: (slot, class) of every state ->
        JointDecodeGraph, the most recent kept), else eager. Returns each state's row buffers (logits, next,
        hidden)."""
        parts = []
        for token, st, slot in items:
            self.stage(token, st, slot)
            parts.append((st, self._rows_buffers(1, self._key(slot, 1)), 1, K.class_of(st.length + 1, st.cap)))
        if graphs is not None:
            key = tuple((slot, ns) for (_, _, slot), (_, _, _, ns) in zip(items, parts))
            # a capture's warm-ups read the rows staged above
            g = graphs.take(key, lambda: JointDecodeGraph(self, parts).captured())
            g.graph.replay()
        else:
            self._rows(parts, write=True)
        _sync(self.device)
        return [rb for _, rb, _, _ in parts]

    @torch.no_grad()
    def commit(self, st: LfmState, path: list[int], graphs: dict | None = None) -> None:
        """Keep the rows `path` (row 0 first, each the next one's parent) of the last verify block: each
        conv state from its last inputs along the path, the path's KV rows moved to their positions,
        the length moved. `graphs`: (M, n, moved) -> CommitGraph."""
        tokens, parents, _, rbkey = st.block
        if path and (path[0] != 0 or any(parents[b] != a for a, b in zip(path, path[1:]))):
            raise ValueError(f"{path} is not a root path of the last block")
        key = self.stage_commit(st, path)
        if rbkey != len(tokens):
            graphs = None                    # a joint pass's block: its own buffers, committed eagerly
        if graphs is not None and path:
            g = graphs.get(key)
            if g is None:
                g = graphs[key] = CommitGraph(self, st, key)
                g.capture()
            g.graph.replay()
        elif path:
            self.commit_dev(st, key, rbkey)
        st.length += len(path)

    def _commit_bufs(self, key) -> dict:
        cb = self._rb.get(("commit",) + key)
        if cb is not None:
            return cb
        c, dev = self.cfg, self.device
        M, n = key[0], key[1]
        S = c.conv_state_len
        sizes = (("idx", n), ("src", n), ("dst", n), ("cols", S))
        tot = sum(w for _, w in sizes)
        hbuf = torch.zeros(tot, dtype=torch.int64, pin_memory=dev.type == "cuda")
        dbuf = torch.zeros(tot, dtype=torch.int64, device=dev)

        def views(b):
            out, k = {}, 0
            for name, w in sizes:
                out[name] = b[k:k + w]
                k += w
            return out
        copied = _event(dev)
        copied.record()
        cb = {"hbuf": hbuf, "dbuf": dbuf, "h": views(hbuf), "d": views(dbuf), "copied": copied}
        self._rb[("commit",) + key] = cb
        return cb

    def stage_commit(self, st: LfmState, path: list[int]) -> tuple:
        M, n = len(st.block[0]), len(path)
        moved = any(path[j] != j for j in range(n))
        key = (M, n, moved)
        if not path:
            return key
        cb = self._commit_bufs(key)
        cb["copied"].synchronize()
        h, L = cb["h"], st.length
        S = self.cfg.conv_state_len
        h["idx"].numpy()[:] = path
        h["src"].numpy()[:] = [L + r for r in path]
        h["dst"].numpy()[:] = L + np.arange(n)
        # the last S inputs along the path as columns of [state (S columns) | the block's rows]
        h["cols"].numpy()[:] = (list(range(S)) + [S + r for r in path])[-S:]
        cb["dbuf"].copy_(cb["hbuf"], non_blocking=True)
        cb["copied"].record()
        return key

    def commit_dev(self, st: LfmState, key, rbkey=None) -> None:
        """The device half of `commit` (capturable): every input from `_commit_bufs(key)`, the B * x rows
        from the block's row buffers (`rbkey`, the lone verify's of M rows by default)."""
        c = self.cfg
        d = self._commit_bufs(key)["d"]
        bx = self._rb[key[0] if rbkey is None else rbkey]["bx"]
        for li in c.conv_layers:
            st.conv[li].copy_(torch.cat([st.conv[li], bx[li]], 0).index_select(0, d["cols"]))
        if key[2]:
            for li in c.attention_layers:
                st.k[li].index_copy_(1, d["dst"], st.k[li].index_select(1, d["src"]))
                st.v[li].index_copy_(1, d["dst"], st.v[li].index_select(1, d["src"]))

    # ------------------------------------------------------------------------- prompt chunks
    @torch.no_grad()
    def prefill(self, tokens: torch.Tensor, st: LfmState, head: str = "last") -> torch.Tensor | None:
        """A prompt (any length) in chunks of PREFILL_CHUNK: the state advances. Logits of the last row
        [1, vocab] fp32 (`head="all"`: of every row of the last chunk; "none": None)."""
        tokens = tokens.view(-1).to(self.device)
        out, rows = None, []
        for a in range(0, tokens.numel(), PREFILL_CHUNK):
            chunk = tokens[a:a + PREFILL_CHUNK]
            last = a + PREFILL_CHUNK >= tokens.numel()
            out = self.forward_chunk(chunk, st, head=head if last else "none")
            rows.append(self.prompt_hidden)
        # every row's final normed state (what a drafter syncs from), not the last chunk's alone
        self.prompt_hidden = rows[0] if len(rows) == 1 else torch.cat(rows)
        return out

    def forward_chunk(self, tokens: torch.Tensor, st: LfmState, head: str = "last") -> torch.Tensor | None:
        from tools import fn_kernels as FK
        c = self.cfg
        T, L, eps = tokens.numel(), st.length, c.rms_norm_eps
        if L + T > st.cap:
            raise ValueError(f"{L + T} rows exceed the state's capacity {st.cap}")
        dev = self.device
        pos = torch.arange(L, L + T, device=dev)
        S = c.conv_state_len
        i = torch.arange(T, device=dev)[:, None]
        dist = torch.arange(S, 0, -1, device=dev)[None, :]
        csrc = torch.where(i - dist >= 0, i - dist, -1 - (S - (dist - i))).to(torch.int32).contiguous()
        h = self.W["embed_tokens.weight"].index_select(0, tokens)
        h, n = K.add_rms(h, None, self.W["layers.0.operator_norm.weight"], eps)
        bx = torch.empty(T, c.hidden_size, dtype=self.dtype, device=dev)
        for li in range(c.num_hidden_layers):
            p = self.pre(li)
            if c.is_attention(li):
                out = self._attn_chunk(n, li, st, pos)
            else:
                bcx = self.lin(n, self.W[p + "conv.in_proj.weight"])
                y = K.conv_rows(bcx, st.conv[li], self._wc[li], csrc, bx)
                st.conv[li].copy_(torch.cat([st.conv[li], bx], 0)[-S:])
                out = self.lin(y, self.W[p + "conv.out_proj.weight"])
            h, n = K.add_rms(h, out, self.W[p + "ffn_norm.weight"], eps)
            if c.is_moe(li):
                logits = F.linear(n, self._gate[li]).float()
                ids, wts = K.router(logits, self._bias.get(li), c.num_experts_per_tok, c.routed_scaling_factor)
                if li in self.banks:
                    from tools import fn_prefill_kernels as PK
                    yd = PK.moe_grouped(n, ids, self.banks[li])
                else:
                    yd = self._moe_bf16_chunk(n, ids, li)
                m = FK.moe_combine(yd, ids, wts) if dev.type == "cuda" else K.combine_reference(yd, ids, wts)
            else:
                f = p + "feed_forward."
                gu = self._fused(n, f, "w13", ("w1", "w3"))
                I = c.intermediate_size
                m = self.lin(F.silu(gu[:, :I]) * gu[:, I:], self.W[f + "w2.weight"])
            nxt = self.W[self.pre(li + 1) + "operator_norm.weight"] if li + 1 < c.num_hidden_layers \
                else self.W["embedding_norm.weight"]
            h, n = K.add_rms(h, m, nxt, eps)
        st.length = L + T
        self.prompt_hidden = n
        if head == "none":
            return None
        return self.head(n[-1:] if head == "last" else n)

    def _attn_chunk(self, n, li, st, pos):
        """Attention of a prompt chunk: q|k|v, the prep kernel (norms, rotary, the chunk's K and V into
        the cache at their positions), SDPA over the cache's rows [0, L + T) with the lower-right
        causal mask, out_proj."""
        from torch.nn.attention.bias import causal_lower_right
        c = self.cfg
        a = self.pre(li) + "self_attn."
        T, L = n.shape[0], int(pos[0])
        qkv = self._fused(n, a, "qkv", ("q_proj", "k_proj", "v_proj"))
        q = K.qk_prep(qkv, pos, pos, self.W[a + "q_layernorm.weight"], self.W[a + "k_layernorm.weight"],
                      self.rope_cos, self.rope_sin, st.k[li], st.v[li], c.num_attention_heads,
                      c.num_key_value_heads, c.rms_norm_eps)
        k = st.k[li][:, :L + T].unsqueeze(0)
        v = st.v[li][:, :L + T].unsqueeze(0)
        qh = q.transpose(0, 1).unsqueeze(0)                                  # [1, Hq, T, D]
        o = F.scaled_dot_product_attention(qh, k.repeat_interleave(c.gqa, 1), v.repeat_interleave(c.gqa, 1),
                                           attn_mask=causal_lower_right(T, L + T))
        o = o[0].transpose(0, 1).reshape(T, -1)
        return self.lin(o, self.W[a + "out_proj.weight"])

    def _moe_bf16_chunk(self, n, ids, li):
        """The BF16 experts of a chunk: each expert once over its rows (cuBLAS) -> yd [T S, H] fp32."""
        e = self.experts[li]
        T, S = ids.shape
        I = e.I2 // 2
        flat = ids.reshape(-1).long()
        yd = torch.empty(T * S, n.shape[1], dtype=torch.float32, device=n.device)
        order = torch.argsort(flat, stable=True)
        counts = torch.bincount(flat, minlength=e.E).tolist()
        o = 0
        for ex, cnt in enumerate(counts):
            if not cnt:
                continue
            pairs = order[o:o + cnt]
            o += cnt
            x = n.index_select(0, pairs // S)
            gu = F.linear(x, e.w13[ex])
            yd[pairs] = F.linear(F.silu(gu[:, :I]) * gu[:, I:], e.w2[ex]).float()
        return yd

    # ------------------------------------------------------------------------- the load-time check
    @torch.no_grad()
    def self_check(self, prompt: list[int], steps: int = 24, log=print) -> dict:
        """ENG-287's load-time check that a verify block's rows are the decode step's: decode `steps`
        tokens one at a time from a prefill, then verify the same tokens as one chain block from the
        prefill's state (and an 8-row tree whose rows are prefixes of the chain), and compare every
        row's logits bit for bit. Any difference leaves `spec_ok` False (speculation off)."""
        st = self.new_state(min(self.max_len, len(prompt) + steps + CAPTURE_ROWS))
        lg = self.prefill(torch.tensor(prompt), st)
        snap = {li: t.clone() for li, t in st.conv.items()}
        L = st.length
        toks, rows = [int(lg[0].argmax())], []
        for _ in range(min(steps, ROWS_MAX)):
            self.stage(toks[-1], st)
            out = self.step(st)
            st.length += 1
            rows.append(out[0].clone())
            toks.append(int(out[0].argmax()))
        st.length = L
        for li, t in snap.items():
            st.conv[li].copy_(t)
        M = len(rows)
        self.stage_rows(toks[:M], list(range(-1, M - 1)), st)
        got = self.rows_step(st, M)
        bad = next((i for i in range(M) if not torch.equal(got[i], rows[i])), None)
        self.spec_ok = bad is None
        res = {"ok": self.spec_ok, "rows": M, "first_diff": bad}
        if log:
            log(f"[lfm-selfcheck] {'PASS' if self.spec_ok else 'FAIL: speculation stays off'}: {res}")
        return res

    spec_ok = False


class DecodeGraph:
    """`Lfm2MoeEngine.step` captured once for a state and an attention class; `__call__(token)` stages
    the token, replays the class's graph (captured when the length first enters it) and moves the
    length."""

    def __init__(self, eng: Lfm2MoeEngine, st: LfmState, slot=None):
        self.eng, self.st, self.slot = eng, st, slot
        self.rb = eng._rows_buffers(1, eng._key(slot, 1))
        self.graphs: dict = {}
        self.graph = None
        self.captures = 0

    def capture(self, ns: int) -> None:
        eng, st = self.eng, self.st
        saved = _state_saved(eng, st)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                eng.stage(0, st, self.slot)
                eng.step(st, ns, self.slot)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with capture_lock.LOCK, torch.cuda.graph(g):
            eng.step(st, ns, self.slot)
        torch.cuda.synchronize()
        _state_restore(saved)                       # the warm-up steps shifted the conv states: undo
        self.graphs[ns] = g

    def _ready(self) -> None:
        ns = K.class_of(self.st.length + 1, self.st.cap)
        if ns not in self.graphs:
            self.capture(ns)
            self.captures += 1
        self.graph = self.graphs[ns]

    @torch.no_grad()
    def __call__(self, token: int) -> torch.Tensor:
        eng, st = self.eng, self.st
        self._ready()
        eng.stage(token, st, self.slot)
        self.graph.replay()
        st.length += 1
        return self.rb["logits"]

    @torch.no_grad()
    def greedy(self, token: int) -> int:
        self(token)
        torch.cuda.current_stream().synchronize()
        return int(self.rb["next_h"][0])


class JointDecodeGraph:
    """`Lfm2MoeEngine._rows` of several states' decode tokens (write=True, a row each) captured once for
    that set of states and their attention classes; each state's tokens are staged in its own decode
    buffers, so a replay reads them where its lone DecodeGraph does. The warm-up runs shift the conv
    states and write KV rows: each state is saved and restored around them."""

    def __init__(self, eng: Lfm2MoeEngine, parts: list):
        self.eng, self.parts = eng, parts
        self.graph = None

    def capture(self) -> None:
        eng, parts = self.eng, self.parts
        saved = [x for st, _, _, _ in parts for x in _state_saved(eng, st)]
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                eng._rows(parts, write=True)
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with capture_lock.LOCK, torch.cuda.graph(self.graph):
            eng._rows(parts, write=True)
        torch.cuda.synchronize()
        _state_restore(saved)

    def captured(self) -> "JointDecodeGraph":
        self.capture()
        return self


class VerifyGraph:
    """`Lfm2MoeEngine.rows_step` of an M-row verify block captured once for a state and a class: every
    block of M rows (chain or tree) replays it; `replay()` after `stage_rows`."""

    def __init__(self, eng: Lfm2MoeEngine, st: LfmState, M: int, ns: int, key=None):
        self.eng, self.st, self.M, self.ns, self.key = eng, st, M, ns, M if key is None else key
        self.graph = None

    def capture(self) -> None:
        eng, st, M = self.eng, self.st, self.M
        eng.stage_rows([0] * M, list(range(-1, M - 1)), st, key=self.key)   # writes only KV rows past the length
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                eng.rows_step(st, M, ns=self.ns, key=self.key)
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with capture_lock.LOCK, torch.cuda.graph(self.graph):
            eng.rows_step(st, M, ns=self.ns, key=self.key)
        torch.cuda.synchronize()

    def replay(self) -> None:
        self.graph.replay()


class CommitGraph:
    """`Lfm2MoeEngine.commit_dev` for a verify block of M rows and a path of n rows (moved or not); the
    warm-up runs write the state, which is saved and restored around them."""

    def __init__(self, eng: Lfm2MoeEngine, st: LfmState, key: tuple):
        self.eng, self.st, self.key = eng, st, key
        self.graph = None

    def capture(self) -> None:
        eng, st = self.eng, self.st
        saved = _state_saved(eng, st)
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


def _state_saved(eng: Lfm2MoeEngine, st: LfmState) -> list:
    """What a capture's warm-up may write, cloned: the conv states whole, the K and V rows from the
    length on (`CAPTURE_ROWS`)."""
    L = st.length
    out = [(t, 0, 0, t.shape[0]) for t in st.conv.values()]
    for d in (st.k, st.v):
        for t in d.values():
            out.append((t, 1, L, min(CAPTURE_ROWS, t.shape[1] - L)))
    return [(t, ax, lo, n, t.narrow(ax, lo, n).clone()) for t, ax, lo, n in out if n > 0]


def _state_restore(saved: list) -> None:
    for t, ax, lo, n, x in saved:
        t.narrow(ax, lo, n).copy_(x)
