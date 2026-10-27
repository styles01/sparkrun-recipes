"""The multi-token-prediction head that ships inside the checkpoint.

`mtp.safetensors` holds one full-attention decoder layer of the same shape as the target's own,
a projection `fc` from twice the hidden width down to it, and three norms. The wiring, which no
released modeling file for this checkpoint implements but a serving stack does, is:

    x = fc( concat( pre_fc_norm_embedding(embed(t_next)),
                    pre_fc_norm_hidden(h_prev) ) )
    x = decoder_layer(x, position)
    logits = lm_head(norm(x))

`h_prev` is the target model's final hidden state at the position before `t_next`. Chaining the
head's own output back in as `h_prev` drafts a second and third token without the target running
at all.

Whether `h_prev` should be taken before or after the target's final norm is not stated anywhere
this could be read from, so both are implemented and selected by measurement (`hidden="post"` or
`"pre"`). Getting it wrong cannot corrupt anything -- the verify step rejects a bad draft the same
as a good one -- it can only cost acceptance, which is exactly what the measurement reads.

Cost, which is the thing to watch: each drafted token reads the head's 0.48 GB and then the
target's 2.54 GB `lm_head`, so a three-token draft moves 9.1 GB against the verify step's 26.9 GB.
The head is cheap and the head's *head* is not.
"""

from __future__ import annotations

import os
from engine.settings import SETTINGS as _S  # noqa: E402  (every QWEN38_* knob)

import torch
import torch.nn.functional as F

from engine.model import KVCache, linear, rms_norm
from . import Drafter


class _MTPCache:
    """One sequence of keys and values for the head's single attention layer."""

    def __init__(self, cfg, max_len: int, device: str):
        self.k = torch.zeros(1, cfg.num_key_value_heads, max_len, cfg.head_dim,
                             dtype=torch.bfloat16, device=device)
        self.v = torch.zeros_like(self.k)
        self.length = 0

    def append(self, k: torch.Tensor, v: torch.Tensor, start: int):
        t = k.shape[2]
        self.k[:, :, start:start + t] = k
        self.v[:, :, start:start + t] = v
        return self.k[:, :, :start + t], self.v[:, :, :start + t]


class MTPDrafter(Drafter):
    name = "mtp"

    def requires(self) -> dict:
        """The checkpoint's own MTP layer, the target's embedding and head."""
        return {"hidden_size": self.cfg.hidden_size,
                "tensors": ("mtp.fc.weight", "embed_tokens.weight", "lm_head.weight")}

    def __init__(self, eng, max_len: int = 4096, hidden: str = "post", depth: int | None = None,
                 draft_head: str | None = None):
        self.eng = eng
        self.cfg = eng.cfg
        self.w = eng.w
        self.hidden = hidden
        self.depth = depth
        # A reduced-vocabulary head for the draft only. It cannot change what the engine outputs --
        # the verify step recomputes the target's own distribution over the whole vocabulary and
        # rejects anything that disagrees -- so the only thing a wrong draft head costs is
        # acceptance. See tools/draft_head.py for the byte arithmetic.
        self.head = None
        self.head_index = None
        draft_head = draft_head if draft_head is not None else _S.get("DRAFT_HEAD")
        if draft_head:
            from tools.draft_head import load_draft_head
            self.head, self.head_index = load_draft_head(draft_head, eng.device)
            print(f"[draft-head] {self.head.N} rows, {self.head.nbytes / 1e6:.1f} MB "
                  f"(full head 2543 MB)")
        self.cache = _MTPCache(self.cfg, max_len, eng.device)
        self.stats = {"calls": 0, "proposed": 0}
        self.synced = 0
        self.h_last: torch.Tensor | None = None
        if "mtp.fc.weight" not in self.w.t:
            raise RuntimeError("the checkpoint's mtp weights were not loaded "
                               "(Weights(..., skip_mtp=True))")

    def reset(self) -> None:
        self.cache.length = 0
        self.synced = 0
        self.h_last = None

    # ---- serving-time state cache ----------------------------------------------------------
    def state_snapshot(self):
        """The head's own KV, its fill length, and the last hidden state it was handed.

        `engine/cache.py` can restore the target's state without forwarding the prefix, and this
        cache is indexed by absolute position: without this it would be handed a hole. One
        attention layer over the prefix is 4 kB a token against 151 MB of recurrent state and
        65.5 kB a token of target KV in the same snapshot.
        """
        n = int(self.cache.length)
        return ("mtp", n, self.cache.k[:, :, :n].clone() if n else None,
                self.cache.v[:, :, :n].clone() if n else None,
                int(self.synced), None if self.h_last is None else self.h_last.clone())

    def state_restore(self, snap) -> None:
        kind, n, k, v, synced, h_last = snap
        if kind != "mtp":
            raise ValueError(f"not an mtp snapshot: {kind!r}")
        if n:
            self.cache.k[:, :, :n] = k
            self.cache.v[:, :, :n] = v
        self.cache.length = n
        self.synced = synced
        self.h_last = h_last

    def sync(self, tokens: list[int], hidden: torch.Tensor, first_pos: int) -> None:
        """Fill the head's own cache for positions the target has just decided.

        Slot p of the head's attention corresponds to the token at position p being embedded
        together with the target's hidden state at p - 1. After a verified block those hidden states
        exist and are the target's own, so filling the slots from them -- rather than leaving
        whatever the last speculative pass wrote there -- keeps the head conditioned on what
        actually happened. One pass over the head's 0.48 GB, batched over every new position.
        """
        w, cfg = self.w, self.cfg
        n = len(tokens)
        if n < 2:
            return
        ids = torch.tensor([tokens[1:]], device=self.eng.device)          # positions first_pos+1..
        h_prev = hidden[None, :n - 1]                                      # positions first_pos..
        with torch.no_grad():
            emb = F.embedding(ids, w.norm("embed_tokens.weight"))
            a = rms_norm(emb, w.norm("mtp.pre_fc_norm_embedding.weight"), cfg.rms_norm_eps)
            b = rms_norm(h_prev, w.norm("mtp.pre_fc_norm_hidden.weight"), cfg.rms_norm_eps)
            x = linear(torch.cat([a, b], dim=-1), w.norm("mtp.fc.weight"))
            self._layer(x, first_pos + 1, causal=True)
        self.synced = first_pos + n
        # The hidden state to condition the next draft on is the one at the LAST ACCEPTED position,
        # which after a partial accept is not the last row of the verify pass -- that row belongs to
        # a token that was rejected. Conditioning on a rejected position costs acceptance on every
        # block after a rejection, silently, which is exactly the kind of thing only a measurement
        # over a long enough generation shows.
        self.h_last = hidden[n - 1:n]

    def _layer(self, x: torch.Tensor, position: int, causal: bool = False) -> torch.Tensor:
        cfg, w, p = self.cfg, self.w, "mtp.layers.0"
        B, T, _ = x.shape
        res = x
        h = rms_norm(x, w.norm(f"{p}.input_layernorm.weight"), cfg.rms_norm_eps)
        qg = linear(h, w.proj(f"{p}.self_attn.q_proj")).view(B, T, cfg.num_attention_heads,
                                                             cfg.head_dim * 2)
        q, gate = qg.chunk(2, dim=-1)
        gate = gate.reshape(B, T, -1)
        q = rms_norm(q, w.norm(f"{p}.self_attn.q_norm.weight"), cfg.rms_norm_eps).transpose(1, 2)
        k = linear(h, w.proj(f"{p}.self_attn.k_proj")).view(B, T, cfg.num_key_value_heads,
                                                            cfg.head_dim)
        k = rms_norm(k, w.norm(f"{p}.self_attn.k_norm.weight"), cfg.rms_norm_eps).transpose(1, 2)
        v = linear(h, w.proj(f"{p}.self_attn.v_proj")).view(
            B, T, cfg.num_key_value_heads, cfg.head_dim).transpose(1, 2)
        # the rotary position is the target's: index + `pos_delta` after a prompt with images
        # (0 for text); the cache slot and the mask stay the index
        r0 = position + getattr(self.eng, "pos_delta", 0)
        cos, sin = self.eng.rope(torch.arange(r0, r0 + T, device=x.device))
        q, k = self.eng.apply_rope(q, k, cos, sin)
        kk, vv = self.cache.append(k, v, position)
        rep = cfg.num_attention_heads // cfg.num_key_value_heads
        mask = None
        if causal and T > 1:
            mask = torch.ones(T, kk.shape[2], dtype=torch.bool, device=x.device).tril(position)
        o = F.scaled_dot_product_attention(q, kk.repeat_interleave(rep, dim=1),
                                           vv.repeat_interleave(rep, dim=1), attn_mask=mask)
        o = o.transpose(1, 2).reshape(B, T, -1) * torch.sigmoid(gate)
        h = res + linear(o, w.proj(f"{p}.self_attn.o_proj"))
        res = h
        y = rms_norm(h, w.norm(f"{p}.post_attention_layernorm.weight"), cfg.rms_norm_eps)
        gate_p = linear(y, w.proj(f"{p}.mlp.gate_proj"))
        up = linear(y, w.proj(f"{p}.mlp.up_proj"))
        return res + linear(F.silu(gate_p) * up, w.proj(f"{p}.mlp.down_proj"))

    def propose(self, context: list[int], k: int) -> list[int]:
        eng, w, cfg = self.eng, self.w, self.cfg
        src = self.h_last
        if src is None or k <= 0:
            return []
        depth = min(k, self.depth or k)
        self.stats["calls"] += 1
        h_prev = src[None]                        # [1, 1, hidden] at the last ACCEPTED position
        tok = torch.tensor([[context[-1]]], device=eng.device)
        position = len(context) - 1
        self.cache.length = position
        out: list[int] = []
        with torch.no_grad():
            for _ in range(depth):
                emb = F.embedding(tok, w.norm("embed_tokens.weight"))
                a = rms_norm(emb, w.norm("mtp.pre_fc_norm_embedding.weight"), cfg.rms_norm_eps)
                b = rms_norm(h_prev, w.norm("mtp.pre_fc_norm_hidden.weight"), cfg.rms_norm_eps)
                x = linear(torch.cat([a, b], dim=-1), w.norm("mtp.fc.weight"))
                x = self._layer(x, position)
                y = rms_norm(x, w.norm("mtp.norm.weight"), cfg.rms_norm_eps)
                if self.head is not None:
                    logits = linear(y, self.head)
                    nxt = int(self.head_index[int(logits[0, -1].argmax())])
                else:
                    logits = linear(y, w.norm("lm_head.weight"))
                    nxt = int(logits[0, -1].argmax())
                out.append(nxt)
                tok = torch.tensor([[nxt]], device=eng.device)
                h_prev = y if self.hidden == "post" else x
                position += 1
        self.stats["proposed"] += len(out)
        return out
