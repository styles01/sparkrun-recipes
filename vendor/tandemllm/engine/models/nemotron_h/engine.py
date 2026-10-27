"""NemotronH on the board: the reference forward with its weight reads on the kernels, a row step that
serves a decode token and a verify block of up to 32 rows with the same per-row arithmetic, and the
step as a CUDA graph.

`NemotronHRef` (ref.py) is the model, checked against `transformers`. This class keeps its structure:

  rows_step(st, M)   M rows whose tokens, positions, cache slots, attention control, ancestors and
                     conv windows sit in device buffers (`stage` for a decode token, `stage_rows` for
                     a draft block in DFS pre-order), every part on a row-invariant kernel
                     (kernels.py: projections on 16-row tiles, the Mamba2 conv and SSM walk, the gated
                     norm, attention over fixed key chunks, the router, the relu^2 experts and their
                     ordered combine), so the row of a verify block is the decode step's row at that
                     position bit for bit. A decode step writes its conv and SSM states in place; a
                     verify block writes nothing but its KV rows past the length and keeps each Mamba
                     layer's in_proj rows and conv outputs for the commit.
  commit(st, path)   the conv states from the path's last xBC inputs (copies), each SSM state walked
                     along the path by the decode step's own kernel (the same fp32 products in the same
                     order: the decode steps' state bit for bit), the path's KV rows moved down, the
                     length moved (`commit_dev` is capturable)
  prefill(tokens)    a prompt in chunks of PREFILL_CHUNK on batched kernels (cuBLAS, the chunked SSD
                     form in fp32 torch matmuls, SDPA with a lower-right causal mask, the experts
                     grouped): the next token agrees with the row path, the bits do not; a prompt's
                     image rows (Super VL, vision.py) replace their placeholders' embeddings here

`DecodeGraph` captures the decode step once per attention class (kernels.classes) and `VerifyGraph` an
M-row block.

Taps (`set_taps`, a block drafter's context, SPD-123): the residual stream entering each tapped layer
(llama.cpp's `t_layer_inp`, the output of the layer before after its residual add) of every row lands in
the row buffers ("taps"), inside the graphs; a committed row's taps go to the state's ring (`st.ring`, by
position modulo its rows): a decode step's in the step, a verify block's kept path's in the commit, a
prompt chunk's last rows in the chunk. They are copies of values the rows compute anyway, so the rows'
bits do not change.
"""

from __future__ import annotations

import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from engine import capture_lock
from engine.models.nemotron_h import kernels as K
from engine.models.nemotron_h.ref import NemotronHRef, NhState, softplus

#: rows a prompt chunk takes (the caches' anchor grid)
PREFILL_CHUNK = 4096
#: rows up to which a projection runs on the row-invariant kernels (their limit, the verify width)
ROWS_MAX = 32
#: rows a captured step, verify block or commit may write past the length (a block is at most 33 rows)
CAPTURE_ROWS = 64
#: the bit widths tools/fn_kernels.dense decodes (the half-integer rates as ten times the rate)
DENSE_K = (2, 3, 4, 5, 6, 8, 15, 25, 35)


#: SPD-125: the EXL3 projections of decode steps and verify blocks on tools/fn_kernels.dense_rows (each trellis
#: tile decoded once for 16 rows, the cluster's partials added by every rank; every byte the one-row `dense`
#: call's); "off" = `dense`, one CTA a row
DENSE_ROWS = os.environ.get("QSE_NH_DENSE_ROWS", "on") != "off"


def _exl3_mm(x2: torch.Tensor, w, out: torch.Tensor | None = None) -> torch.Tensor:
    """An EXL3 projection (ENG-E12): tools/fn_kernels.dense_rows up to 32 rows (`dense`'s bytes, row-invariant,
    DENSE_ROWS), the library's GEMM above that or at a rate dense does not take; on the CPU the torch reference a
    row at a time (row-invariant as the kernels are: a verify block's rows are the decode step's)."""
    from tools import fn_kernels as FK
    from tools.exl3_linear import exl3_matmul
    if not x2.is_cuda:
        y = torch.cat([exl3_matmul(x2[i:i + 1], w) for i in range(x2.shape[0])])
        return y if out is None else out.copy_(y)
    if x2.shape[0] <= ROWS_MAX and w.kb in DENSE_K:
        return (FK.dense_rows if DENSE_ROWS else FK.dense)(x2, w, out=out)
    return exl3_matmul(x2, w, out=out)


def _is_exl3(w) -> bool:
    from tools.exl3_linear import EXL3Block
    return isinstance(w, EXL3Block)


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


class NemotronHEngine(NemotronHRef):

    def __init__(self, cfg, W: dict, experts: dict, device="cuda", max_len: int = 32768):
        super().__init__(cfg, W, dtype=torch.bfloat16, device=device, experts=experts)
        c = cfg
        self.max_len = max_len
        self.classes = K.classes(max_len)
        self._rb: dict = {}
        self._attn: dict = {}
        self.host_ms: dict = {}
        m = {li: self.pre(li) + "mixer." for li in range(c.num_hidden_layers)}
        self._wc = {li: W[m[li] + "conv1d.weight"].reshape(c.conv_dim, -1).to(torch.bfloat16).contiguous()
                    for li in c.mamba_layers}
        # A itself where the loader gave it (a GGUF stores -exp(A_log)), else from A_log
        self._A = {li: (W[m[li] + "A"].float() if m[li] + "A" in W else -torch.exp(W[m[li] + "A_log"].float())).contiguous()
                   for li in c.mamba_layers}
        self._f32 = {(li, n): W[m[li] + n].float().contiguous() for li in c.mamba_layers for n in ("dt_bias", "D")}
        self._f32.update({(li, "bias"): W[m[li] + "gate.e_score_correction_bias"].float().contiguous()
                          for li in c.moe_layers})

    #: the tapped layers (`set_taps`) and each one's column in the taps; the ring's rows
    taps: tuple = ()
    _tap_at: dict = {}
    ring_rows = 0

    def set_taps(self, layers, window: int) -> None:
        """Keep the residual rows entering `layers` for a drafter that reads the last `window` committed
        rows: every state gets a ring of `window` + CAPTURE_ROWS rows (a verify block's commit lands
        between two drafts). Before any state, row buffer or graph exists: they are sized and captured
        with the taps."""
        if self._rb:
            raise RuntimeError("set_taps after the row buffers exist: their graphs would not write the taps")
        layers = tuple(int(x) for x in layers)
        if any(not 0 < li < self.cfg.num_hidden_layers for li in layers):
            raise ValueError(f"taps {layers}: a layer of 1..{self.cfg.num_hidden_layers - 1} (the embedding is not one)")
        self.taps = layers
        self._tap_at = {li: k for k, li in enumerate(layers)}
        self.ring_rows = int(window) + CAPTURE_ROWS

    # ------------------------------------------------------------------------- state
    def new_state(self, cap: int | None = None, paged: bool = False) -> NhState:
        """A state with every tensor the step writes allocated up front (a graph holds them). `paged`
        with the KV pool on (engine/kvpool.py): K and V are reserved as addresses and committed as rows
        are written, by the caller before every write."""
        from engine import kvpool as _kvp
        c = self.cfg
        cap = cap or self.max_len
        st = NhState(cap=cap)
        if self.taps:
            st.ring = torch.zeros(self.ring_rows, len(self.taps), c.hidden_size, dtype=self.dtype, device=self.device)
        if paged and _kvp.POOL is not None and cap >= _kvp.PAGED_MIN_ROWS and str(self.device).startswith("cuda"):
            st.pages = _kvp.PagedSet(_kvp.OWNER)
        for li, t in enumerate(c.layer_types):
            if t == "attention":
                for kind, d in (("k", st.k), ("v", st.v)):
                    shape = (c.num_key_value_heads, cap, c.head_dim)
                    if st.pages is None:
                        d[li] = torch.zeros(*shape, dtype=self.dtype, device=self.device)
                    else:
                        d[li] = st.pages.add(_kvp.PagedBuffer(shape, 1, self.dtype, self.device, owner=st.pages.owner,
                                                              name=f"nh.{kind}.{li}")).tensor
            elif t == "mamba":
                st.conv[li] = torch.zeros(c.conv_state_len, c.conv_dim, dtype=self.dtype, device=self.device)
                st.ssm[li] = torch.zeros(c.mamba_num_heads, c.mamba_head_dim, c.ssm_state_size, dtype=torch.float32,
                                         device=self.device)
        return st

    def reset(self, st: NhState) -> None:
        st.reset_inplace()

    # ------------------------------------------------------------------------- primitives
    def lin(self, x: torch.Tensor, w, out: torch.Tensor | None = None) -> torch.Tensor:
        x2 = x.reshape(-1, x.shape[-1])
        if not x2.is_contiguous():
            x2 = x2.contiguous()
        if _is_exl3(w):
            return _exl3_mm(x2, w, out=out).view(*x.shape[:-1], w.N)
        if not torch.is_tensor(w):
            # a packed weight (tools/iq2_linear: IQ2Linear, NVFP4Linear): its own row-invariant kernel at any row count
            return w.matmul(x2, out=out).view(*x.shape[:-1], w.shape[0])
        if x2.shape[0] <= ROWS_MAX:
            return K.linear_rows(x2, w, out=out).view(*x.shape[:-1], w.shape[0])
        y = F.linear(x2, w)
        return y if out is None else out.copy_(y)

    def qkv(self, n: torch.Tensor, m: str, mul) -> torch.Tensor:
        """q | k | v of an attention block: the fused weight in one launch, or the three where the loader
        could not fuse them (packed weights)."""
        w = self.W.get(m + "qkv")
        if w is not None:
            return mul(n, w)
        return torch.cat([mul(n, self.W[m + f"{x}_proj.weight"]) for x in "qkv"], -1)

    @staticmethod
    def mm(x: torch.Tensor, w) -> torch.Tensor:
        """A prompt chunk's projection: cuBLAS for a bf16 weight, the packed weight's own kernel else."""
        return F.linear(x, w) if torch.is_tensor(w) else w.matmul(x)

    def head(self, n: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        """Logits [M, vocab] fp32 of the final normed rows (a static `out` in a graph)."""
        w = self.head_weight()
        if _is_exl3(w):
            return _exl3_mm(n, w, out=out)          # an fp32 head block (load.py: out_fp32)
        if not torch.is_tensor(w):
            return w.matmul(n, out=out, f32=True)
        if n.shape[0] <= ROWS_MAX:
            return K.linear_rows(n, w, out=out, f32=True)
        y = F.linear(n, w).float()
        return y if out is None else out.copy_(y)

    def moe_rows(self, n: torch.Tensor, li) -> torch.Tensor:
        """The MoE block of M <= 32 rows: the router, every (row, slot) pair through its expert, the
        ordered combine, the latent pair, the shared expert."""
        c = self.cfg
        if isinstance(li, tuple):                    # the MTP head's block ("mtp", j)
            p = f"mtp.layers.{li[1]}.mixer."
            k = c.mtp_moe[li[1]][0]
            bias = self._f32.get(li)
            if bias is None:
                bias = self._f32[li] = self.W[p + "gate.e_score_correction_bias"].float().contiguous()
        else:
            p = self.pre(li) + "mixer."
            k = c.moe_k[li]
            bias = self._f32[(li, "bias")]
        logits = K.linear_rows(n, self.W[p + "gate.weight"], f32=True)
        ids, wts = K.router(logits, bias, k, c.routed_scaling_factor)
        x = self.lin(n, self.W[p + "fc1_latent_proj.weight"]) if c.moe_latent_size else n
        e = self.experts[li]
        yd = e.moe_relu2(x, ids) if hasattr(e, "moe_relu2") else K.moe_relu2(x, ids, e.up, e.down)
        m = K.combine(yd, ids, wts)
        if c.moe_latent_size:
            m = self.lin(m, self.W[p + "fc2_latent_proj.weight"])
        if self.route_trace is not None:
            self.route_trace.append((li, ids.long()))
        s = self.lin(F.relu(self.lin(n, self.W[p + "shared_experts.up_proj.weight"])).square(),
                     self.W[p + "shared_experts.down_proj.weight"])
        return m + s

    route_trace = None

    # ------------------------------------------------------------------------- rows: decode and verify
    def _rows_buffers(self, M: int, key=None) -> dict:
        """The device inputs of an M-row call and their pinned host image (one H2D copy a call):

          tok, pos, slot   int64 [M]: the token, its position (block start + depth), its cache slot
          ring             int64 [M]: a decode step's row in the state's taps ring
          ctl              int32 [M, 2]: visible keys, block start
          anc              int32 [M, ANC]: the cache slot of the row's ancestor at each depth
          csrc             int32 [M, KT - 1]: where the row's earlier conv inputs live (a row of the
                           block, or -1 - a state column)
          manc, mn         int32 [M, ANC], [M]: the block rows of the row's path (root first), its length
        and the outputs (logits, greedy tokens and their pinned copy, the final normed rows, the taps) and
        each Mamba layer's in_proj rows and conv outputs (what a commit keeps)."""
        key = M if key is None else key
        rb = self._rb.get(key)
        if rb is not None:
            return rb
        c, dev = self.cfg, self.device
        S = c.conv_state_len
        n64 = 4 * M
        n32 = M * (2 + K.ANC + S + K.ANC + 1)
        n32 += n32 % 2
        nb = 8 * n64 + 4 * n32

        def views(b):
            i64 = b[:8 * n64].view(torch.int64)
            i32 = b[8 * n64:].view(torch.int32)
            v = {"tok": i64[:M], "pos": i64[M:2 * M], "slot": i64[2 * M:3 * M], "ring": i64[3 * M:4 * M]}
            k = 0
            for name, w in (("ctl", 2), ("anc", K.ANC), ("csrc", S), ("manc", K.ANC), ("mn", 1)):
                v[name] = i32[k:k + M * w].view(M, w)
                k += M * w
            v["mn"] = v["mn"].view(M)
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
              "proj": {li: torch.zeros(M, c.in_proj_dim, dtype=self.dtype, device=dev) for li in c.mamba_layers},
              "xbc": {li: torch.zeros(M, c.conv_dim, dtype=self.dtype, device=dev) for li in c.mamba_layers},
              "tag": key[:-1] if isinstance(key, tuple) else None}
        if self.taps:
            rb["taps"] = torch.zeros(M, len(self.taps), c.hidden_size, dtype=self.dtype, device=dev)
        if M == 1:
            h = rb["h"]
            h["csrc"].numpy()[:] = K.window_src([-1], c.conv_kernel)[0]
            h["manc"].numpy()[0, 0] = 0
            h["mn"].numpy()[0] = 1
        self._rb[key] = rb
        return rb

    def _attn_bufs(self, tag, M: int, ns: int) -> dict:
        """ENG-363: attention's partials of an M-row call at class `ns`, one set for the row counts of a
        power-of-two bucket of a slot (`tag`) instead of one a row count: a set is M x 32 MiB at class
        2,048 (a 1,048,576-token window), and the 33 verify widths held up to 17 GiB. Each call writes
        the partials it reads, so calls of one slot share them one after another."""
        rows = min(1 << (M - 1).bit_length(), ROWS_MAX + 1)
        return self._attn.setdefault((tag, ns, rows), {"rows": rows})

    @staticmethod
    def _key(slot, M: int):
        """The row buffers' key of an M-row call of a sequence slot (None: slot 0's, keyed by M alone)."""
        return M if slot is None else (slot, M)

    def stage(self, token: int, st: NhState, slot=None) -> None:
        """Host work of one decode step: the token, its position and slot, its attention control."""
        rb = self._rows_buffers(1, self._key(slot, 1))
        rb["copied"].synchronize()            # the image is the previous call's copy source until then
        h, L = rb["h"], st.length
        h["tok"].numpy()[0] = token
        h["pos"].numpy()[0] = L
        h["slot"].numpy()[0] = L
        h["ctl"].numpy()[0] = (L + 1, L)
        h["anc"].numpy()[0, 0] = L
        if self.taps:
            h["ring"].numpy()[0] = L % self.ring_rows
        rb["dbuf"].copy_(rb["hbuf"], non_blocking=True)
        rb["copied"].record()

    def stage_rows(self, tokens: list[int], parents: list[int], st: NhState, key=None) -> None:
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
        anc, manc = h["anc"].numpy(), h["manc"].numpy()
        for i, pa in enumerate(paths):
            anc[i, :len(pa)] = L + np.asarray(pa)
            manc[i, :len(pa)] = pa
        h["mn"].numpy()[:] = d + 1
        h["csrc"].numpy()[:] = K.window_src(parents, self.cfg.conv_kernel)
        rb["dbuf"].copy_(rb["hbuf"], non_blocking=True)
        rb["copied"].record()
        st.block = (list(tokens), list(parents), paths, M if key is None else key)

    def rows_step(self, st: NhState, M: int, write: bool = False, ns: int | None = None, key=None) -> torch.Tensor:
        """M rows from the device buffers (`stage` / `stage_rows` first) through every layer and the head.
        `write` (one row, a decode step): the conv and SSM states advance in place. Otherwise only the
        block's KV rows past the length are written and each Mamba layer's rows are kept for the commit.
        `ns`: the attention class (kernels.class_of). Returns logits [M, vocab] fp32 (a static buffer)
        and leaves the greedy tokens in rb["next"] and its pinned host copy; `st.length` is the caller's."""
        rb = self._rows_buffers(M, key)
        self._rows([(st, rb, M, ns or K.class_of(st.length + M, st.cap))], write)
        return rb["logits"]

    def _rows(self, parts: list, write: bool = False) -> None:
        """The rows of one or more states [(state, its row buffers, rows, attention class)] in one pass:
        the projections, the experts and the head over every row at once (their weights read once), the
        conv, the SSM walk and attention per state on its own rows and states. Every kernel is
        row-invariant, so a state's rows get the bits of its lone call."""
        c = self.cfg
        eps = c.rms_norm_eps
        cuts = [0]
        for _, _, M, _ in parts:
            cuts.append(cuts[-1] + M)
        R = cuts[-1]
        sl = [slice(cuts[j], cuts[j + 1]) for j in range(len(parts))]
        one = len(parts) == 1
        tok = parts[0][1]["d"]["tok"] if one else torch.cat([rb["d"]["tok"] for _, rb, _, _ in parts])
        h = self.W["embeddings.weight"].index_select(0, tok)
        h, n = K.add_rms(h, None, self.W["layers.0.norm.weight"], eps)
        I, CD = c.intermediate, c.conv_dim
        for li, t in enumerate(c.layer_types):
            m = self.pre(li) + "mixer."
            k = self._tap_at.get(li)
            if k is not None:
                for (_, rb, _, _), r in zip(parts, sl):
                    rb["taps"][:, k].copy_(h if one else h[r])
            if t == "mamba":
                if one:
                    P = self.lin(n, self.W[m + "in_proj.weight"], out=parts[0][1]["proj"][li])
                else:
                    P = self.lin(n, self.W[m + "in_proj.weight"])
                y = torch.empty(R, I, dtype=self.dtype, device=self.device)
                for (st, rb, M, ns), r in zip(parts, sl):
                    pr = rb["proj"][li]
                    if not one:
                        pr.copy_(P[r])
                    d = rb["d"]
                    xb = K.conv_rows(pr, I, st.conv[li], self._wc[li], self.W[m + "conv1d.bias"], d["csrc"],
                                     y=rb["xbc"][li], write=write)
                    K.ssm_rows(pr, I + CD, xb, d["manc"], d["mn"], st.ssm[li], self._A[li], self._f32[(li, "dt_bias")],
                               self._f32[(li, "D")], c.n_groups, y=y[r], write=write)
                g = K.gated_norm(y, P[:, :I], self.W[m + "norm.weight"], c.n_groups, eps)
                out = self.lin(g, self.W[m + "out_proj.weight"])
            elif t == "attention":
                qkv = self.qkv(n, m, self.lin)
                Hq, Hk, D = c.num_attention_heads, c.num_key_value_heads, c.head_dim
                q = qkv[:, :Hq * D].reshape(R, Hq, D).contiguous()     # a view would keep qkv's row stride
                o = torch.empty(R, Hq, D, dtype=self.dtype, device=self.device)
                for (st, rb, M, ns), r in zip(parts, sl):
                    d = rb["d"]
                    kv = qkv[r]
                    st.k[li].index_copy_(1, d["slot"], kv[:, Hq * D:(Hq + Hk) * D].reshape(M, Hk, D).transpose(0, 1))
                    st.v[li].index_copy_(1, d["slot"], kv[:, (Hq + Hk) * D:].reshape(M, Hk, D).transpose(0, 1))
                    K.attn_rows(q[r], st.k[li], st.v[li], d["ctl"], d["anc"], ns, bufs=self._attn_bufs(rb["tag"], M, ns),
                                out=o[r])
                out = self.lin(o.view(R, -1), self.W[m + "o_proj.weight"])
            else:
                out = self.moe_rows(n, li)
            nxt = self.W[self.pre(li + 1) + "norm.weight"] if li + 1 < c.num_hidden_layers else self.W["norm_f.weight"]
            h, n = K.add_rms(h, out, nxt, eps)
        if one:
            rb = parts[0][1]
            rb["hidden"].copy_(n)
            self.head(n, out=rb["logits"])
        else:
            logits = self.head(n)
            for (_, rb, _, _), r in zip(parts, sl):
                rb["hidden"].copy_(n[r])
                rb["logits"].copy_(logits[r])
        for st, rb, _, _ in parts:
            torch.argmax(rb["logits"], dim=-1, out=rb["next"])
            rb["next_h"].copy_(rb["next"], non_blocking=True)
            if write and st.ring is not None:
                st.ring.index_copy_(0, rb["d"]["ring"], rb["taps"])

    def step(self, st: NhState, ns: int | None = None, slot=None) -> torch.Tensor:
        """One decode token from the device buffers (`stage` first): the conv and SSM states advance in
        place, the KV row lands at the length; `st.length` is the caller's. Logits [1, vocab] fp32."""
        return self.rows_step(st, 1, write=True, ns=ns, key=self._key(slot, 1))

    @torch.no_grad()
    def verify(self, tokens: list[int], parents: list[int], st: NhState, graphs: dict | None = None,
               slot=None) -> list[int]:
        """The greedy token at every row of a draft block (DFS pre-order), eager or from `graphs`
        ((M, class) -> VerifyGraph). Nothing in `st` moves until `commit`."""
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
        _sync(self.device)
        return self._rb[key]["next_h"].tolist()

    @torch.no_grad()
    def verify_multi(self, items: list) -> list:
        """The verify blocks of several states (sequence slots over these weights) [(tokens, parents,
        state)] in one pass of at most 32 rows: each state's rows are its lone verify's, bit for bit.
        Eager; each state commits its block (`commit`). Returns per state its row buffers."""
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
        """One decode token of each of several states [(token, state, slot)] in one pass, every state
        advanced in place as by its own decode step, each row the bits of that state's lone decode step.
        From `graphs` (engine/models/graph_lru.py: (slot, class) of every state -> JointDecodeGraph, the
        most recent kept), else eager."""
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
    def commit(self, st: NhState, path: list[int], graphs: dict | None = None) -> None:
        """Keep the rows `path` (row 0 first, each the next one's parent) of the last verify block: the
        conv states from the path's last inputs, each SSM state walked along the path, the path's KV
        rows moved to their positions, the length moved. `graphs`: (M, n, moved) -> CommitGraph."""
        tokens, parents, _, rbkey = st.block
        if path and (path[0] != 0 or any(parents[b] != a for a, b in zip(path, path[1:]))):
            raise ValueError(f"{path} is not a root path of the last block")
        key = self.stage_commit(st, path)
        if rbkey != len(tokens):
            graphs = None                    # a joint pass's or a side slot's block: committed eagerly
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
        n = key[1]
        S = c.conv_state_len
        sizes = (("idx", n), ("src", n), ("dst", n), ("cols", S), ("ring", n))
        tot = sum(w for _, w in sizes)
        pin = dev.type == "cuda"
        hbuf = torch.zeros(tot, dtype=torch.int64, pin_memory=pin)
        dbuf = torch.zeros(tot, dtype=torch.int64, device=dev)
        h32 = torch.zeros(K.ANC + 1, dtype=torch.int32, pin_memory=pin)
        d32 = torch.zeros(K.ANC + 1, dtype=torch.int32, device=dev)

        def views(b, b32):
            out, k = {}, 0
            for name, w in sizes:
                out[name] = b[k:k + w]
                k += w
            out["path"] = b32[:K.ANC].view(1, K.ANC)
            out["pn"] = b32[K.ANC:]
            return out
        copied = _event(dev)
        copied.record()
        cb = {"hbuf": hbuf, "dbuf": dbuf, "h32": h32, "d32": d32, "h": views(hbuf, h32), "d": views(dbuf, d32),
              "copied": copied, "y": torch.zeros(1, c.intermediate, dtype=self.dtype, device=dev)}
        self._rb[("commit",) + key] = cb
        return cb

    def stage_commit(self, st: NhState, path: list[int]) -> tuple:
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
        if self.taps:
            h["ring"].numpy()[:] = (L + np.arange(n)) % self.ring_rows
        # the last S inputs along the path as columns of [state (S columns) | the block's rows]
        h["cols"].numpy()[:] = (list(range(S)) + [S + r for r in path])[-S:]
        h["path"].numpy()[0, :n] = path
        h["pn"].numpy()[0] = n
        cb["dbuf"].copy_(cb["hbuf"], non_blocking=True)
        cb["d32"].copy_(cb["h32"], non_blocking=True)
        cb["copied"].record()
        return key

    def commit_dev(self, st: NhState, key, rbkey=None) -> None:
        """The device half of `commit` (capturable): every input from `_commit_bufs(key)`, the block's
        Mamba rows from its row buffers (`rbkey`, the lone verify's of M rows by default)."""
        c = self.cfg
        cb = self._commit_bufs(key)
        d = cb["d"]
        rb = self._rb[key[0] if rbkey is None else rbkey]
        I, CD = c.intermediate, c.conv_dim
        for li in c.mamba_layers:
            pr = rb["proj"][li]
            st.conv[li].copy_(torch.cat([st.conv[li], pr[:, I:I + CD]], 0).index_select(0, d["cols"]))
            K.ssm_rows(pr, I + CD, rb["xbc"][li], d["path"], d["pn"], st.ssm[li], self._A[li],
                       self._f32[(li, "dt_bias")], self._f32[(li, "D")], c.n_groups, y=cb["y"], write=True)
        if key[2]:
            for li in c.attention_layers:
                st.k[li].index_copy_(1, d["dst"], st.k[li].index_select(1, d["src"]))
                st.v[li].index_copy_(1, d["dst"], st.v[li].index_select(1, d["src"]))
        if st.ring is not None:
            st.ring.index_copy_(0, d["ring"], rb["taps"].index_select(0, d["idx"]))

    # ------------------------------------------------------------------------- prompt chunks
    @torch.no_grad()
    def prefill(self, tokens: torch.Tensor, st: NhState, head: str = "last", mm=None) -> torch.Tensor | None:
        """A prompt (any length) in chunks of PREFILL_CHUNK: the state advances. Logits of the last row
        [1, vocab] fp32 (`head="all"`: of every row of the last chunk; "none": None). `mm`: the request's
        images (vision.py `NhMMContext`), whose rows replace their placeholders' embeddings."""
        tokens = tokens.view(-1).to(self.device)
        out, rows = None, []
        for a in range(0, tokens.numel(), PREFILL_CHUNK):
            chunk = tokens[a:a + PREFILL_CHUNK]
            last = a + PREFILL_CHUNK >= tokens.numel()
            out = self.forward_chunk(chunk, st, head=head if last else "none", mm=mm)
            rows.append(self.prompt_hidden)
        self.prompt_hidden = rows[0] if len(rows) == 1 else torch.cat(rows)
        return out

    def forward_chunk(self, tokens: torch.Tensor, st: NhState, head: str = "last", mm=None) -> torch.Tensor | None:
        c = self.cfg
        T, L, eps = tokens.numel(), st.length, c.rms_norm_eps
        if L + T > st.cap:
            raise ValueError(f"{L + T} rows exceed the state's capacity {st.cap}")
        h = self.W["embeddings.weight"].index_select(0, tokens)
        if mm is not None:
            h = mm.splice(h, L)                           # the chunk's image rows (no positions to move)
        h, n = self._add_rms(h, None, self.W["layers.0.norm.weight"])
        if st.ring is not None:
            j0 = max(0, T - self.ring_rows)               # the chunk's rows the ring keeps
            rdst = torch.arange(L + j0, L + T, device=self.device) % self.ring_rows
        for li, t in enumerate(c.layer_types):
            k = self._tap_at.get(li)
            if k is not None and st.ring is not None:
                st.ring[:, k].index_copy_(0, rdst, h[j0:])
            if t == "mamba":
                out = self._mamba_chunk(n, li, st)
            elif t == "attention":
                out = self._attn_chunk(n, li, st, L)
            else:
                out = self._moe_chunk(n, li)
            nxt = self.W[self.pre(li + 1) + "norm.weight"] if li + 1 < c.num_hidden_layers else self.W["norm_f.weight"]
            h, n = self._add_rms(h, out, nxt)
        st.length = L + T
        self.prompt_hidden = n
        if head == "none":
            return None
        return self.head(n[-1:] if head == "last" else n)

    def _add_rms(self, h, d, w):
        return K.add_rms(h, d, w, self.cfg.rms_norm_eps)

    def _mamba_chunk(self, n, li, st):
        """A prompt chunk's Mamba2 block: the conv over [state | chunk] in fp32, the chunked SSD form
        (chunk `cfg.chunk_size`) in fp32 matmuls from the state, the gated norm, out_proj."""
        c = self.cfg
        m = self.pre(li) + "mixer."
        T = n.shape[0]
        I, CD, Hh, P, N, G = c.intermediate, c.conv_dim, c.mamba_num_heads, c.mamba_head_dim, c.ssm_state_size, c.n_groups
        proj = self.mm(n, self.W[m + "in_proj.weight"])
        z, xr, dtr = proj[:, :I], proj[:, I:I + CD], proj[:, I + CD:]
        S_ = c.conv_state_len
        win = torch.cat([st.conv[li], xr], 0)                               # [S + T, CD]
        conv = F.conv1d(win.T.float()[None], self._wc[li].float()[:, None, :], self.W[m + "conv1d.bias"].float(),
                        groups=CD)[0].T                                     # [T, CD]
        xbc = F.silu(conv).to(self.dtype)
        st.conv[li].copy_(win[-S_:])
        x = xbc[:, :I].float().view(T, Hh, P)
        rep = Hh // G
        B = xbc[:, I:I + G * N].float().view(T, G, N)
        C = xbc[:, I + G * N:].float().view(T, G, N)
        dt = softplus(dtr.float() + self._f32[(li, "dt_bias")])            # [T, H]
        A = self._A[li]
        y, S = ssd_chunked(x, dt, A, B, C, st.ssm[li], rep, c.chunk_size)
        y = y + self._f32[(li, "D")][None, :, None] * x
        st.ssm[li].copy_(S)
        y = y.reshape(T, I).to(self.dtype)
        g = (y.float() * F.silu(z.float())).view(T, G, -1)
        g = g * torch.rsqrt(g.pow(2).mean(-1, keepdim=True) + c.rms_norm_eps)
        g = (g.view(T, I) * self.W[m + "norm.weight"].float()).to(self.dtype)
        return self.mm(g, self.W[m + "out_proj.weight"])

    def _attn_chunk(self, n, li, st, L):
        """Attention of a prompt chunk: q|k|v, the chunk's K and V into the cache at their positions,
        SDPA over the cache's rows [0, L + T) with the lower-right causal mask, o_proj (NoPE)."""
        from torch.nn.attention.bias import causal_lower_right
        c = self.cfg
        m = self.pre(li) + "mixer."
        T = n.shape[0]
        Hq, Hk, D = c.num_attention_heads, c.num_key_value_heads, c.head_dim
        qkv = self.qkv(n, m, self.mm)
        q = qkv[:, :Hq * D].reshape(T, Hq, D)
        st.k[li][:, L:L + T] = qkv[:, Hq * D:(Hq + Hk) * D].reshape(T, Hk, D).transpose(0, 1)
        st.v[li][:, L:L + T] = qkv[:, (Hq + Hk) * D:].reshape(T, Hk, D).transpose(0, 1)
        k = st.k[li][:, :L + T].unsqueeze(0)
        v = st.v[li][:, :L + T].unsqueeze(0)
        # GQA inside SDPA: the same results as K and V repeated to every query head, without the copies (at
        # 262,144 tokens the copies grew with every chunk and the allocator kept each size: 100 GiB reserved)
        o = F.scaled_dot_product_attention(q.transpose(0, 1).unsqueeze(0), k, v, attn_mask=causal_lower_right(T, L + T),
                                           enable_gqa=True)
        return self.mm(o[0].transpose(0, 1).reshape(T, -1), self.W[m + "o_proj.weight"])

    def _moe_chunk(self, n, li):
        """A prompt chunk's MoE block: fp32 router logits, the router kernel, each expert once over its
        rows (cuBLAS; an IQ2 bank's expert decoded to bf16 first, an EXL3 expert on its own GEMM), the
        ordered combine, the latent pair, the shared expert."""
        c = self.cfg
        m = self.pre(li) + "mixer."
        logits = F.linear(n.float(), self.W[m + "gate.weight"].float())
        bias = self._f32[(li, "bias")]
        ids, wts = K.router(logits, bias, c.moe_k[li], c.routed_scaling_factor)
        x = self.mm(n, self.W[m + "fc1_latent_proj.weight"]) if c.moe_latent_size else n
        e = self.experts[li]
        T, S = ids.shape
        flat = ids.reshape(-1).long()
        yd = torch.empty(T * S, x.shape[1], dtype=torch.float32, device=n.device)
        order = torch.argsort(flat, stable=True)
        counts = torch.bincount(flat, minlength=e.E).tolist()
        o = 0
        for ex, cnt in enumerate(counts):
            if not cnt:
                continue
            pairs = order[o:o + cnt]
            o += cnt
            xe = x.index_select(0, pairs // S)
            up, down = e.expert(ex)                  # bf16 (an IQ2Bank's decoded for the chunk) or EXL3 blocks
            yd[pairs] = self.mm(F.relu(self.mm(xe, up)).square(), down).float()
        out = K.combine(yd, ids, wts)
        if c.moe_latent_size:
            out = self.mm(out, self.W[m + "fc2_latent_proj.weight"])
        s = self.mm(F.relu(self.mm(n, self.W[m + "shared_experts.up_proj.weight"])).square(),
                    self.W[m + "shared_experts.down_proj.weight"])
        return out + s

    # ------------------------------------------------------------------------- the load-time check
    @torch.no_grad()
    def self_check(self, prompt: list[int], steps: int = 24, log=print) -> dict:
        """The load-time check that a verify block's rows are the decode step's: decode `steps` tokens
        one at a time from a prefill, then verify the same tokens as one chain block from the prefill's
        state and compare every row's logits bit for bit; then commit the chain and compare the SSM
        and conv states with the decode steps'. With taps (`set_taps`) a verify row's taps are held to
        the decode step's and the ring after the commit to the ring after the steps. Any difference
        leaves `spec_ok` False (speculation off) and names the first row or layer."""
        st = self.new_state(min(self.max_len, len(prompt) + steps + CAPTURE_ROWS))
        lg = self.prefill(torch.tensor(prompt), st)
        snap = _snapshot(st)
        L = st.length
        toks, rows, taps = [int(lg[0].argmax())], [], []
        for _ in range(min(steps, ROWS_MAX)):
            self.stage(toks[-1], st)
            out = self.step(st)
            st.length += 1
            rows.append(out[0].clone())
            toks.append(int(out[0].argmax()))
            if self.taps:
                taps.append(self._rb[1]["taps"][0].clone())
        after = _snapshot(st)
        ring = st.ring.clone() if st.ring is not None else None
        st.length = L
        _restore(st, snap)
        M = len(rows)
        self.stage_rows(toks[:M], list(range(-1, M - 1)), st)
        got = self.rows_step(st, M)
        bad = next((i for i in range(M) if not torch.equal(got[i], rows[i])), None)
        tap_bad = next((i for i in range(len(taps)) if not torch.equal(self._rb[M]["taps"][i], taps[i])), None)
        layer = None
        if bad is None:
            self.commit(st, list(range(M)))
            for k, (a, b) in enumerate(zip(_snapshot(st), after)):
                if not torch.equal(a, b):
                    layer = sorted(st.conv)[k % len(st.conv)] if st.conv else k
                    break
            if ring is not None and not torch.equal(st.ring, ring):
                tap_bad = "ring"
        self.spec_ok = bad is None and layer is None and tap_bad is None
        res = {"ok": self.spec_ok, "rows": M, "first_diff": bad, "state_diff_layer": layer}
        if self.taps:
            res["tap_diff"] = tap_bad
        if log:
            log(f"[nh-selfcheck] {'PASS' if self.spec_ok else 'FAIL: speculation stays off'}: {res}")
        return res

    spec_ok = False


def ssd_chunked(x, dt, A, B, C, S0, rep: int, Q: int):
    """The chunked SSD form of the recurrence S_t = S_{t-1} exp(dt_t A) + dt_t x_t B_t, y_t = S_t C_t,
    in fp32 from S0. x [T, H, P], dt [T, H], A [H], B / C [T, G, N] (G groups of `rep` heads), S0 [H, P, N].
    Returns (y [T, H, P] without the D term, the final state). Rows past T are padded with dt = 0
    (no decay, no input), so the final state is the state at T."""
    T, H, P = x.shape
    N = B.shape[-1]
    pad = (-T) % Q
    if pad:
        x = F.pad(x, (0, 0, 0, 0, 0, pad))
        dt = F.pad(dt, (0, 0, 0, pad))
        B = F.pad(B, (0, 0, 0, 0, 0, pad))
        C = F.pad(C, (0, 0, 0, 0, 0, pad))
    nc = (T + pad) // Q
    x, dt = x.view(nc, Q, H, P), dt.view(nc, Q, H)
    B, C = B.view(nc, Q, -1, N), C.view(nc, Q, -1, N)
    G = B.shape[2]
    a = dt * A[None, None, :]                                         # [c, Q, H]
    acum = torch.cumsum(a, dim=1)
    # intra-chunk: y_i = sum_{j <= i} exp(acum_i - acum_j) (C_i . B_j) dt_j x_j
    seg = acum[:, :, None, :] - acum[:, None, :, :]                   # [c, i, j, H]
    tri = torch.tril(torch.ones(Q, Q, dtype=torch.bool, device=x.device))
    Lm = torch.where(tri[None, :, :, None], torch.exp(seg), torch.zeros((), device=x.device))
    cb = torch.einsum("cign,cjgn->cgij", C, B)                        # [c, G, i, j]
    cb = cb.repeat_interleave(rep, 1)                                 # [c, H, i, j]
    Mx = cb * Lm.permute(0, 3, 1, 2)                                  # [c, H, i, j]
    xdt = x * dt[..., None]                                           # [c, Q, H, P]
    y = torch.einsum("chij,cjhp->cihp", Mx, xdt)
    # each chunk's own contribution to the state at its end
    decay = torch.exp(acum[:, -1:, :] - acum)                         # [c, Q, H]
    Bh = B.repeat_interleave(rep, 2)                                  # [c, Q, H, N]
    states = torch.einsum("cjh,cjhp,cjhn->chpn", decay, xdt, Bh)
    # across chunks: the state entering each chunk
    tot = torch.exp(acum[:, -1, :])                                   # [c, H]
    S = S0.clone()
    ins = []
    for k in range(nc):
        ins.append(S)
        S = S * tot[k][:, None, None] + states[k]
    Sin = torch.stack(ins)                                            # [c, H, P, N]
    Ch = C.repeat_interleave(rep, 2)                                  # [c, Q, H, N]
    y = y + torch.einsum("cihn,chpn->cihp", Ch, Sin) * torch.exp(acum)[..., None]
    return y.reshape(nc * Q, H, P)[:T], S


class DecodeGraph:
    """`NemotronHEngine.step` captured once for a state and an attention class; `__call__(token)` stages
    the token, replays the class's graph (captured when the length first enters it) and moves the
    length."""

    def __init__(self, eng: NemotronHEngine, st: NhState, slot=None):
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
        _state_restore(saved)                       # the warm-up steps advanced the states: undo
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
    """`NemotronHEngine._rows` of several states' decode tokens (write=True, a row each) captured once
    for that set of states and their attention classes; each state is saved and restored around the
    warm-up runs."""

    def __init__(self, eng: NemotronHEngine, parts: list):
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
    """`NemotronHEngine.rows_step` of an M-row verify block captured once for a state and a class: every
    block of M rows (chain or tree) replays it; `replay()` after `stage_rows`."""

    def __init__(self, eng: NemotronHEngine, st: NhState, M: int, ns: int, key=None):
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
    """`NemotronHEngine.commit_dev` for a verify block of M rows and a path of n rows (moved or not); the
    warm-up runs write the state, which is saved and restored around them."""

    def __init__(self, eng: NemotronHEngine, st: NhState, key: tuple):
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


def _snapshot(st: NhState) -> list:
    return [t.clone() for t in st.conv.values()] + [t.clone() for t in st.ssm.values()]


def _restore(st: NhState, snap: list) -> None:
    for t, s in zip(list(st.conv.values()) + list(st.ssm.values()), snap):
        t.copy_(s)


def _state_saved(eng: NemotronHEngine, st: NhState) -> list:
    """What a capture's warm-up may write, cloned: the conv and SSM states whole, the K and V rows from
    the length on (`CAPTURE_ROWS`)."""
    L = st.length
    out = [(t, 0, 0, t.shape[0]) for t in list(st.conv.values()) + list(st.ssm.values())]
    for d in (st.k, st.v):
        for t in d.values():
            out.append((t, 1, L, min(CAPTURE_ROWS, t.shape[1] - L)))
    return [(t, ax, lo, n, t.narrow(ax, lo, n).clone()) for t, ax, lo, n in out if n > 0]


def _state_restore(saved: list) -> None:
    for t, ax, lo, n, x in saved:
        t.narrow(ax, lo, n).copy_(x)
