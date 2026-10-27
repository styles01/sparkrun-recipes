"""The NemotronH forward in torch, operation for operation: the model the board code is held to.

`NemotronHRef` runs a block of rows through the model, where a block is a prompt chunk (a chain), one
decode token, or a draft tree in DFS pre-order with its parents. Every row sees the cache's rows
[0, length) and its own ancestors in the block. A Mamba2 row reads, along its own path:

  conv    the last `conv_kernel - 1` xBC inputs before it (the state's where the path is shorter)
  SSM     the state S0 at the block's start, advanced one step per row of its path:
            dt = softplus(dt_raw + dt_bias)            per head, fp32
            S  = S * exp(dt * A) + dt * x (outer) B    per head [head_dim, state], fp32, B of its group
            y  = S C + D x                              rounded to the activations' dtype

so a verify row computes what decoding its path one token at a time computes, and `commit(path)`
replays the same steps along the kept path into the state (the path identity of the recurrence: the
same products in the same order, nothing approximated). A block writes nothing to the state but its
KV rows past the length.

The arithmetic follows `transformers.models.nemotron_h` and the CUDA path the checkpoints were run
with (mamba_ssm's `selective_state_update`, `causal_conv1d_update`, the gated `rmsnorm_fn`):

  RMS norm      fp32, times plain `w`, cast back
  gated norm    y * silu(z) in fp32, RMS over groups of inner / n_groups, times `w` in fp32, one
                rounding (`norm_before_gate=False`)
  dt            no clamp (`time_step_limit` [0, inf]; transformers' torch path clamps at
                `time_step_min`, its CUDA path and vLLM do not)
  attention     no positional encoding (NoPE), scale 1/sqrt(head_dim), softmax in fp32
  router        logits fp32 from fp32 weights, sigmoid, the correction bias for the choice only, top k
                (ties to the lower id), the chosen sigmoids / (sum + 1e-20) x routed_scaling_factor
  experts       relu(up x)^2, down; the weighted sum in fp32, rounded once; latent: fc1_latent before,
                fc2_latent after; plus the shared expert on the block's input

The tests (tests/test_nemotron_h_ref.py) hold it to transformers on tiny random models in fp32.

Weights are a dict of canonical names (the checkpoint's without its `backbone.` prefix):

  embeddings.weight, norm_f.weight, lm_head.weight (absent and tied: the embedding)
  layers.L.norm.weight
  layers.L.mixer.{in_proj, out_proj}.weight, .conv1d.{weight [conv_dim, 1, K], bias}, .dt_bias,
      .A_log, .D, .norm.weight                                                        (mamba)
  layers.L.mixer.{q,k,v,o}_proj.weight                                                (attention)
  layers.L.mixer.gate.weight [E, H], .gate.e_score_correction_bias [E],
      .experts.E.{up,down}_proj.weight, .shared_experts.{up,down}_proj.weight,
      .fc1_latent_proj.weight, .fc2_latent_proj.weight (latent only)                  (moe)

`lin(x, w)` is the one place a weight is multiplied, so the board engine routes it to its kernels.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


@dataclass
class NhState:
    """One sequence: per Mamba layer the conv's last inputs [conv_kernel - 1, conv_dim] (oldest first)
    and the SSM state [heads, head_dim, state] fp32; K and V per attention layer [kv heads, cap, D]."""
    cap: int
    length: int = 0
    conv: dict = field(default_factory=dict)
    ssm: dict = field(default_factory=dict)
    k: dict = field(default_factory=dict)
    v: dict = field(default_factory=dict)
    pages: object = None        # the KV pool's paged set (engine/kvpool.py), or None
    block: tuple | None = None  # the board engine's last verify block: (tokens, parents, paths, row buffers key)
    # the board engine's taps (`NemotronHEngine.set_taps`): the residual rows entering the tapped layers of
    # the last committed rows, [ring rows, taps, hidden] by position modulo the ring; rows from `tap_from` on
    # are this sequence's, and `tap_gen` moves whenever the rows below the length stop being them
    ring: object = None
    tap_from: int = 0
    tap_gen: int = 0

    def reset_inplace(self) -> None:
        """A new sequence in the same buffers: conv and SSM states zeroed, the length 0."""
        self.length = 0
        self.tap_from = 0
        self.tap_gen += 1
        for t in self.conv.values():
            t.zero_()
        for t in self.ssm.values():
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


def softplus(x: torch.Tensor) -> torch.Tensor:
    """mamba_ssm's softplus: log(1 + exp(x)), x itself above 20."""
    return torch.where(x <= 20.0, torch.log1p(torch.exp(x)), x)


class NemotronHRef:
    def __init__(self, cfg, W: dict, dtype=torch.float32, device="cpu", experts: dict | None = None):
        self.cfg, self.W, self.dtype, self.device = cfg, W, dtype, torch.device(device)
        # MoE layer -> its experts as loaded (load.py); absent: the W names
        self.experts = experts or {}
        self.rec: dict = {}          # what a commit needs of the last block, per Mamba layer
        self._block = None           # (parents, paths) of the last block

    # ------------------------------------------------------------------------ helpers
    @staticmethod
    def pre(li: int) -> str:
        return f"layers.{li}."

    def w(self, name: str) -> torch.Tensor:
        return self.W[name]

    def head_weight(self):
        return self.W.get("lm_head.weight", self.W["embeddings.weight"])

    def lin(self, x: torch.Tensor, w) -> torch.Tensor:
        return F.linear(x, w)

    def rms(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        dt = x.dtype
        h = x.float()
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.cfg.rms_norm_eps)
        return (w.float() * h).to(dt)

    def gated_norm(self, y: torch.Tensor, z: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        c = self.cfg
        x = y.float() * F.silu(z.float())
        g = x.view(*x.shape[:-1], c.n_groups, -1)
        g = g * torch.rsqrt(g.pow(2).mean(-1, keepdim=True) + c.rms_norm_eps)
        return (g.view_as(x) * w.float()).to(y.dtype)

    def new_state(self, cap: int) -> NhState:
        c = self.cfg
        st = NhState(cap=cap)
        for li, t in enumerate(c.layer_types):
            if t == "attention":
                st.k[li] = torch.zeros(c.num_key_value_heads, cap, c.head_dim, dtype=self.dtype, device=self.device)
                st.v[li] = torch.zeros_like(st.k[li])
            elif t == "mamba":
                st.conv[li] = torch.zeros(c.conv_state_len, c.conv_dim, dtype=self.dtype, device=self.device)
                st.ssm[li] = torch.zeros(c.mamba_num_heads, c.mamba_head_dim, c.ssm_state_size, dtype=torch.float32,
                                         device=self.device)
        return st

    # ------------------------------------------------------------------------ Mamba2
    def ssm_step(self, S: torch.Tensor, x: torch.Tensor, B: torch.Tensor, dt: torch.Tensor, li: int) -> torch.Tensor:
        """One step of the recurrence: S [heads, hd, N] fp32, x [heads, hd], B [groups, N], dt [heads]
        (softplus applied) -> the next S."""
        c = self.cfg
        A = -torch.exp(self.w(self.pre(li) + "mixer.A_log").float())
        rep = c.mamba_num_heads // c.n_groups
        Bh = B.float().repeat_interleave(rep, 0)                                   # [heads, N]
        dA = torch.exp(dt * A)                                                     # [heads]
        return S * dA[:, None, None] + (dt[:, None] * x.float())[:, :, None] * Bh[:, None, :]

    def ssm_out(self, S: torch.Tensor, x: torch.Tensor, C: torch.Tensor, li: int) -> torch.Tensor:
        c = self.cfg
        rep = c.mamba_num_heads // c.n_groups
        Ch = C.float().repeat_interleave(rep, 0)
        D = self.w(self.pre(li) + "mixer.D").float()
        return (S * Ch[:, None, :]).sum(-1) + D[:, None] * x.float()               # [heads, hd]

    def split_proj(self, proj: torch.Tensor):
        c = self.cfg
        I, CD = c.intermediate, c.conv_dim
        return proj[:, :I], proj[:, I:I + CD], proj[:, I + CD:]

    def split_xbc(self, xbc: torch.Tensor):
        c = self.cfg
        I, G = c.intermediate, c.n_groups * c.ssm_state_size
        x = xbc[..., :I].reshape(*xbc.shape[:-1], c.mamba_num_heads, c.mamba_head_dim)
        B = xbc[..., I:I + G].reshape(*xbc.shape[:-1], c.n_groups, c.ssm_state_size)
        C = xbc[..., I + G:].reshape(*xbc.shape[:-1], c.n_groups, c.ssm_state_size)
        return x, B, C

    def dt_of(self, dt_raw: torch.Tensor, li: int) -> torch.Tensor:
        return softplus(dt_raw.float() + self.w(self.pre(li) + "mixer.dt_bias").float())

    def mamba(self, h: torch.Tensor, li: int, st: NhState, paths: list[list[int]]) -> torch.Tensor:
        c = self.cfg
        p = self.pre(li) + "mixer."
        S_ = c.conv_state_len
        proj = self.lin(h, self.w(p + "in_proj.weight"))
        z, xbc_raw, dt_raw = self.split_proj(proj)
        wc = self.w(p + "conv1d.weight").reshape(c.conv_dim, -1)                   # [conv_dim, K]
        bc = self.w(p + "conv1d.bias")
        state = st.conv[li]
        conv = []
        for pa in paths:
            win = torch.cat([state, xbc_raw[pa]], 0)[-(S_ + 1):]                    # [K, conv_dim] oldest first
            conv.append(F.silu((win.float() * wc.T.float()).sum(0) + bc.float()).to(h.dtype))
        xbc = torch.stack(conv)
        x, B, C = self.split_xbc(xbc)
        dt = self.dt_of(dt_raw, li)
        ys = []
        for i, pa in enumerate(paths):
            S = st.ssm[li].clone()
            for a in pa:
                S = self.ssm_step(S, x[a], B[a], dt[a], li)
            ys.append(self.ssm_out(S, x[i], C[i], li).to(h.dtype))
        y = torch.stack(ys).reshape(len(paths), -1)
        self.rec[li] = (xbc_raw, x, B, dt)
        return self.lin(self.gated_norm(y, z, self.w(p + "norm.weight")), self.w(p + "out_proj.weight"))

    # ------------------------------------------------------------------------ attention
    def attention(self, h: torch.Tensor, li: int, st: NhState, paths: list[list[int]]) -> torch.Tensor:
        c = self.cfg
        p = self.pre(li) + "mixer."
        M, D, Hq, Hk = h.shape[0], c.head_dim, c.num_attention_heads, c.num_key_value_heads
        L = st.length
        q = self.lin(h, self.w(p + "q_proj.weight")).view(M, Hq, D)
        k = self.lin(h, self.w(p + "k_proj.weight")).view(M, Hk, D)
        v = self.lin(h, self.w(p + "v_proj.weight")).view(M, Hk, D)
        st.k[li][:, L:L + M] = k.transpose(0, 1)
        st.v[li][:, L:L + M] = v.transpose(0, 1)
        out = torch.empty(M, Hq, D, dtype=h.dtype, device=h.device)
        for i, pa in enumerate(paths):
            slots = torch.cat([torch.arange(L, device=h.device), L + torch.tensor(pa, device=h.device)])
            ks = st.k[li][:, slots].repeat_interleave(c.gqa, dim=0)
            vs = st.v[li][:, slots].repeat_interleave(c.gqa, dim=0)
            s = torch.einsum("hd,hnd->hn", q[i], ks) * D ** -0.5
            a = torch.softmax(s, dim=-1, dtype=torch.float32).to(h.dtype)
            out[i] = torch.einsum("hn,hnd->hd", a, vs)
        return self.lin(out.reshape(M, Hq * D), self.w(p + "o_proj.weight"))

    # ------------------------------------------------------------------------ MoE
    def route(self, h: torch.Tensor, li: int, k: int | None = None, prefix: str | None = None):
        """(ids [M, k] int64, weights [M, k] fp32)."""
        c = self.cfg
        p = (prefix or self.pre(li)) + "mixer.gate."
        k = k or c.moe_k[li]
        logits = F.linear(h.float(), self.w(p + "weight").float())
        s = logits.sigmoid()
        choice = s + self.w(p + "e_score_correction_bias").float()
        ids = torch.sort(choice, dim=-1, descending=True, stable=True).indices[:, :k].contiguous()
        wts = torch.gather(s, 1, ids)
        wts = wts / (wts.sum(dim=-1, keepdim=True) + 1e-20)
        return ids, wts * c.routed_scaling_factor

    def relu2_mlp(self, x: torch.Tensor, up, down) -> torch.Tensor:
        return self.lin(F.relu(self.lin(x, up)).square(), down)

    def expert_weights(self, li: int, e: int, prefix: str | None = None):
        if prefix is None and li in self.experts:
            return self.experts[li].expert(e)
        p = f"{prefix or self.pre(li)}mixer.experts.{e}."
        return self.w(p + "up_proj.weight"), self.w(p + "down_proj.weight")

    def moe(self, h: torch.Tensor, li: int, k: int | None = None, prefix: str | None = None) -> torch.Tensor:
        c = self.cfg
        p = (prefix or self.pre(li)) + "mixer."
        ids, wts = self.route(h, li, k, prefix)
        x = self.lin(h, self.w(p + "fc1_latent_proj.weight")) if c.moe_latent_size else h
        acc = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
        for e in sorted(set(ids.view(-1).tolist())):
            rows, slot = torch.where(ids == e)
            up, down = self.expert_weights(li, e, prefix)
            y = self.relu2_mlp(x[rows], up, down)
            acc.index_add_(0, rows, y.float() * wts[rows, slot, None])
        out = acc.to(h.dtype)
        if c.moe_latent_size:
            out = self.lin(out, self.w(p + "fc2_latent_proj.weight"))
        return out + self.relu2_mlp(h, self.w(p + "shared_experts.up_proj.weight"),
                                    self.w(p + "shared_experts.down_proj.weight"))

    # ------------------------------------------------------------------------ a block
    @torch.no_grad()
    def rows(self, tokens: torch.Tensor, parents: list[int], st: NhState, hidden: bool = False) -> torch.Tensor:
        """Logits [M, vocab] (fp32) of a block at the state's length; nothing in `st` moves but the
        block's own KV rows past the length (`commit` keeps a path). `hidden`: also keep the final
        normed rows in `self.last_hidden`."""
        c = self.cfg
        tokens = tokens.view(-1).to(self.device)
        M = tokens.numel()
        if st.length + M > st.cap:
            raise ValueError(f"{st.length + M} rows exceed the state's capacity {st.cap}")
        paths = tree_paths(parents)
        self.rec = {}
        self._block = (list(parents), paths)
        h = self.W["embeddings.weight"][tokens].to(self.dtype)
        for li, t in enumerate(c.layer_types):
            n = self.rms(h, self.w(self.pre(li) + "norm.weight"))
            if t == "mamba":
                h = h + self.mamba(n, li, st, paths)
            elif t == "attention":
                h = h + self.attention(n, li, st, paths)
            else:
                h = h + self.moe(n, li)
        h = self.rms(h, self.w("norm_f.weight"))
        self.last_hidden = h
        return self.lin(h, self.head_weight()).float()

    @torch.no_grad()
    def commit(self, st: NhState, path: list[int]) -> None:
        """Keep the rows `path` (row 0 first, each the next one's parent) of the last block."""
        parents, paths = self._block
        if path and (path[0] != 0 or any(parents[b] != a for a, b in zip(path, path[1:]))):
            raise ValueError(f"{path} is not a root path of the last block")
        L, n = st.length, len(path)
        idx = torch.tensor(path, dtype=torch.long, device=self.device)
        S_ = self.cfg.conv_state_len
        for li, (xbc_raw, x, B, dt) in self.rec.items():
            st.conv[li].copy_(torch.cat([st.conv[li], xbc_raw[idx]], 0)[-S_:])
            S = st.ssm[li].clone()
            for a in path:
                S = self.ssm_step(S, x[a], B[a], dt[a], li)
            st.ssm[li].copy_(S)
        if any(r != j for j, r in enumerate(path)):
            src, dst = L + idx, L + torch.arange(n, device=self.device)
            for li in st.k:
                st.k[li][:, dst] = st.k[li][:, src]
                st.v[li][:, dst] = st.v[li][:, src]
        st.length = L + n
        self._block, self.rec = None, {}

    def prefill(self, tokens: torch.Tensor, st: NhState) -> torch.Tensor:
        """A prompt chunk: logits [T, vocab] of every row; the state advances."""
        n = tokens.numel()
        lg = self.rows(tokens, chain(n), st)
        self.commit(st, list(range(n)))
        return lg

    def decode(self, token: int, st: NhState) -> torch.Tensor:
        return self.prefill(torch.tensor([int(token)]), st)
