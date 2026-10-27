"""NemotronH's MTP head as a draft head: what the shared MTP drafter (engine/methods/mtp_drafter.py)
asks of a model's head, over the checkpoint's `mtp.layers.*` (Megatron's multi-token prediction).

One step of the head reads the trunk's final normed row h_t and the next token t+1:

  x = eh_proj([enorm(embed(t+1)) | hnorm(h_t)])        2H -> H, the embedding first
  x = x + attention(norm_0(x))                         NoPE GQA over the head's own KV cache
  x = x + moe(norm_1(x))                               the latent MoE block with its own router
  o = final_layernorm(x); logits = lm_head(o)          the trunk's head: the token at t+2

One native depth; deeper drafts iterate the step on its own output (o and the drafted token), as
exllamav3's `nemotronh_mtp.py` does. The head's KV is a view of the caches (`views`), so a restore
or a rollback covers it. It drafts only: every drafted token is verified by the trunk, so the head's
arithmetic (torch, eager) never reaches the text.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn.functional as F

from engine.models.nemotron_h import kernels as K

MTP_ROWS_KERNELS = 32


class NemotronHMTPHead:
    def __init__(self, eng, depth: int, cap: int):
        c = eng.cfg
        if c.mtp_layer_types != ["attention", "moe"]:
            raise ValueError(f"NemotronH MTP blocks {c.mtp_layer_types}: the head runs attention + moe")
        if "mtp.layers.0.eh_proj.weight" not in eng.W:
            raise ValueError("the checkpoint's MTP head was not loaded (load(..., mtp=True))")
        self.e, self.cfg, self.depth = eng, c, int(depth)
        shape = (c.num_key_value_heads, cap, c.head_dim)
        self.st = SimpleNamespace(length=0, cap=cap, k=torch.zeros(*shape, dtype=eng.dtype, device=eng.device),
                                  v=torch.zeros(*shape, dtype=eng.dtype, device=eng.device))
        self.captures = 0
        self._n = 0
        self._res = ([], [])
        self.bias = eng.W["mtp.layers.1.mixer.gate.e_score_correction_bias"].float().contiguous()

    # --- the drafter's interface ------------------------------------------------------------------
    def reset(self) -> None:
        self.st.length = 0

    def views(self) -> list:
        return [(self.st.k, 1), (self.st.v, 1)]

    def resume(self, n: int) -> None:
        self.st.length = int(n)

    def capture_all(self) -> None:
        pass

    @torch.no_grad()
    def prime(self, rows: torch.Tensor, nexts: list[int]) -> None:
        """Rows [n, H] (the trunk's final normed rows) and the token after each into the head's cache."""
        n = len(nexts)
        for a in range(0, n, 4096):
            self._step(rows[a:a + 4096], nexts[a:a + 4096], self.st.length + a)
        self.st.length += n

    @torch.no_grad()
    def launch(self, rows: torch.Tensor, nexts: list[int]) -> None:
        """Feed rows [n, H] with their next tokens, then draft `depth` tokens from the last row: per
        depth the top two tokens and their probabilities (`read`). Draft steps write the head's KV past
        its length only."""
        n = len(nexts)
        L = self.st.length
        o = self._step(rows, nexts, L)[-1:]
        toks, probs = [], []
        for d in range(self.depth):
            lg = self._logits(o)
            p = torch.softmax(lg, -1)
            pv, ti = torch.topk(p, 2, dim=-1)
            t2, p2 = ti[0].tolist(), pv[0].tolist()
            toks.append(t2)
            probs.append(p2)
            if d + 1 < self.depth:
                o = self._step(o, [t2[0]], L + n + d)
        self._n = n
        self._res = (toks, probs)

    def read(self) -> tuple[list, list]:
        self.st.length += self._n
        return self._res

    # --- the head's forward -----------------------------------------------------------------------
    def _logits(self, o: torch.Tensor) -> torch.Tensor:
        return self.e.mm(o, self.e.head_weight()).float()

    def _step(self, h: torch.Tensor, toks: list[int], pos0: int) -> torch.Tensor:
        """Rows at positions pos0.. of the head's sequence: their final normed outputs [n, H]."""
        e, c, W = self.e, self.cfg, self.e.W
        T = len(toks)
        if pos0 + T > self.st.cap:
            raise ValueError(f"the MTP head's cache holds {self.st.cap} rows, not {pos0 + T}")
        ids = torch.tensor(toks, dtype=torch.long, device=e.device)
        emb = W["embeddings.weight"].index_select(0, ids)
        x = e.mm(torch.cat([e.rms(emb, W["mtp.layers.0.enorm.weight"]), e.rms(h.to(e.dtype), W["mtp.layers.0.hnorm.weight"])], -1),
                 W["mtp.layers.0.eh_proj.weight"])
        # block 0: NoPE attention over the head's cache
        m = "mtp.layers.0.mixer."
        n0 = e.rms(x, W["mtp.layers.0.norm.weight"])
        Hq, Hk, D = c.num_attention_heads, c.num_key_value_heads, c.head_dim
        qkv = e.qkv(n0, m, e.mm)
        q = qkv[:, :Hq * D].reshape(T, Hq, D)
        self.st.k[:, pos0:pos0 + T] = qkv[:, Hq * D:(Hq + Hk) * D].reshape(T, Hk, D).transpose(0, 1)
        self.st.v[:, pos0:pos0 + T] = qkv[:, (Hq + Hk) * D:].reshape(T, Hk, D).transpose(0, 1)
        k = self.st.k[:, :pos0 + T].unsqueeze(0).repeat_interleave(c.gqa, 1)
        v = self.st.v[:, :pos0 + T].unsqueeze(0).repeat_interleave(c.gqa, 1)
        if T == 1:
            att = F.scaled_dot_product_attention(q.transpose(0, 1).unsqueeze(0), k, v)
        else:
            from torch.nn.attention.bias import causal_lower_right
            att = F.scaled_dot_product_attention(q.transpose(0, 1).unsqueeze(0), k, v,
                                                 attn_mask=causal_lower_right(T, pos0 + T))
        x = x + e.mm(att[0].transpose(0, 1).reshape(T, -1), W[m + "o_proj.weight"])
        # block 1: the MoE
        n1 = e.rms(x, W["mtp.layers.1.norm.weight"])
        x = x + (e.moe_rows(n1, ("mtp", 1)) if T <= MTP_ROWS_KERNELS else self._moe_many(n1))
        return e.rms(x, W["mtp.layers.1.final_layernorm.weight"])

    def _moe_many(self, n: torch.Tensor) -> torch.Tensor:
        """The head's MoE over many rows (a prime's backlog): each expert once over its rows."""
        e, c, W = self.e, self.cfg, self.e.W
        m = "mtp.layers.1.mixer."
        k = c.mtp_moe[1][0]
        logits = F.linear(n.float(), W[m + "gate.weight"].float())
        ids, wts = K.router(logits, self.bias, k, c.routed_scaling_factor)
        x = e.mm(n, W[m + "fc1_latent_proj.weight"]) if c.moe_latent_size else n
        ex = e.experts[("mtp", 1)]
        T, S = ids.shape
        flat = ids.reshape(-1).long()
        yd = torch.empty(T * S, x.shape[1], dtype=torch.float32, device=n.device)
        order = torch.argsort(flat, stable=True)
        counts = [] if hasattr(ex, "moe_relu2") else torch.bincount(flat, minlength=ex.E).tolist()
        if hasattr(ex, "moe_relu2"):
            yd = ex.moe_relu2(x, ids)
        o = 0
        for j, cnt in enumerate(counts):
            if cnt:
                pairs = order[o:o + cnt]
                o += cnt
                yd[pairs] = F.linear(F.relu(F.linear(x.index_select(0, pairs // S), ex.up[j])).square(),
                                     ex.down[j]).float()
        out = K.combine(yd, ids, wts)
        if c.moe_latent_size:
            out = e.mm(out, W[m + "fc2_latent_proj.weight"])
        s = e.mm(F.relu(e.mm(n, W[m + "shared_experts.up_proj.weight"])).square(),
                 W[m + "shared_experts.down_proj.weight"])
        return out + s
