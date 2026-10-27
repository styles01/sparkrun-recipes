"""A Llama-architecture decoder in plain PyTorch: eager, one sequence, greedy, a KV cache of its own.

MiniCPM5-2B and -1B are `LlamaForCausalLM` checkpoints (MiniCPM5-2B's config.json: 42 layers,
hidden 2,048, 16 query heads and 2 KV heads of 128, MLP 6,144, rope theta 5e6 without scaling, an
untied head, vocabulary 130,560, BF16). This file is the whole forward: no transformers model, no
SGLang or vLLM, nothing beside the engine but these tensors.

The arithmetic is transformers' `LlamaForCausalLM` (5.x) step for step, so that greedy text is the
reference's on the same weights (tests/test_narrator.py compares the logits with transformers' own
model on a random tiny checkpoint; tools/narrator_reference.py the real checkpoint's greedy text):
RMSNorm in fp32 with the weight applied after the cast back, the rotary cos and sin computed in fp32
and cast to the activations' dtype, attention through `scaled_dot_product_attention` with the GQA
groups, the SiLU-gated MLP, the head on the last row only (the argmax does not need the others).

The cache: K and V of every layer, preallocated once at `max_ctx` rows (MiniCPM5-2B at 1,024 rows:
42 x 2 x 2 heads x 128 x 1,024 x 2 bytes = 44 MB), and the ids its rows hold. A new prompt reuses
the rows of the longest common prefix with them (the narrator's system prompt is the same for
every line, so only the event is prefilled) and writes from there on.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class Config:
    hidden: int
    layers: int
    heads: int
    kv_heads: int
    head_dim: int
    mlp: int
    vocab: int
    eps: float
    theta: float
    tied: bool

    @classmethod
    def read(cls, path: str) -> "Config":
        with open(os.path.join(path, "config.json")) as f:
            c = json.load(f)
        arch = c.get("architectures") or []
        if c.get("model_type") != "llama" and "LlamaForCausalLM" not in arch:
            raise ValueError(f"{path}: not a Llama-architecture checkpoint ({c.get('model_type')})")
        if c.get("rope_scaling"):
            raise ValueError(f"{path}: rope_scaling {c['rope_scaling']} is not implemented here")
        if c.get("attention_bias") or c.get("mlp_bias"):
            raise ValueError(f"{path}: biases in attention or MLP are not implemented here")
        heads = int(c["num_attention_heads"])
        return cls(hidden=int(c["hidden_size"]), layers=int(c["num_hidden_layers"]), heads=heads,
                   kv_heads=int(c.get("num_key_value_heads") or heads),
                   head_dim=int(c.get("head_dim") or int(c["hidden_size"]) // heads),
                   mlp=int(c["intermediate_size"]), vocab=int(c["vocab_size"]),
                   eps=float(c.get("rms_norm_eps", 1e-6)),
                   theta=float(c.get("rope_theta", 10000.0)),
                   tied=bool(c.get("tie_word_embeddings", False)))


def _norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    dt = x.dtype
    h = x.to(torch.float32)
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)
    return w * h.to(dt)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    a, b = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-b, a), dim=-1)


class Llama:
    """The weights (a dict of tensors on one device) and one sequence's cache."""

    def __init__(self, cfg: Config, weights: dict[str, torch.Tensor], max_ctx: int):
        self.cfg, self.w, self.max_ctx = cfg, weights, int(max_ctx)
        any_w = weights["model.embed_tokens.weight"]
        self.device, self.dtype = any_w.device, any_w.dtype
        self.head = weights["model.embed_tokens.weight"] if cfg.tied else weights["lm_head.weight"]
        shape = (1, cfg.kv_heads, self.max_ctx, cfg.head_dim)
        self.k = [torch.zeros(shape, dtype=self.dtype, device=self.device)
                  for _ in range(cfg.layers)]
        self.v = [torch.zeros(shape, dtype=self.dtype, device=self.device)
                  for _ in range(cfg.layers)]
        self.inv_freq = 1.0 / (cfg.theta ** (torch.arange(0, cfg.head_dim, 2, dtype=torch.int64)
                                             .to(device=self.device, dtype=torch.float32)
                                             / cfg.head_dim))
        self.ids: list[int] = []            # the ids whose rows the cache holds, in order

    def nbytes(self) -> int:
        seen, n = set(), 0
        for t in [*self.w.values(), *self.k, *self.v]:
            if t.data_ptr() not in seen:
                seen.add(t.data_ptr())
                n += t.numel() * t.element_size()
        return n

    def _rope(self, pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        freqs = pos.to(torch.float32)[:, None] * self.inv_freq[None, :]
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(self.dtype), emb.sin().to(self.dtype)

    def forward(self, ids: list[int], start: int) -> torch.Tensor:
        """Rows `start .. start+len(ids)` written into the cache; the logits of the last row
        (fp32, one vector of the vocabulary)."""
        cfg, w = self.cfg, self.w
        n = len(ids)
        end = start + n
        if end > self.max_ctx:
            raise ValueError(f"{end} rows do not fit the cache's {self.max_ctx}")
        x = F.embedding(torch.tensor([ids], device=self.device), w["model.embed_tokens.weight"])
        cos, sin = self._rope(torch.arange(start, end, device=self.device))
        cos, sin = cos[None, None], sin[None, None]
        mask, causal = None, False
        if n > 1 and start == 0:
            causal = True
        elif n > 1:
            mask = (torch.arange(end, device=self.device)[None, :]
                    <= torch.arange(start, end, device=self.device)[:, None])
        for i in range(cfg.layers):
            p = f"model.layers.{i}."
            h = _norm(x, w[p + "input_layernorm.weight"], cfg.eps)
            q = F.linear(h, w[p + "self_attn.q_proj.weight"]).view(1, n, cfg.heads, cfg.head_dim)
            k = F.linear(h, w[p + "self_attn.k_proj.weight"]).view(1, n, cfg.kv_heads,
                                                                   cfg.head_dim)
            v = F.linear(h, w[p + "self_attn.v_proj.weight"]).view(1, n, cfg.kv_heads,
                                                                   cfg.head_dim)
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            q = q * cos + _rotate_half(q) * sin
            k = k * cos + _rotate_half(k) * sin
            self.k[i][:, :, start:end] = k
            self.v[i][:, :, start:end] = v
            a = F.scaled_dot_product_attention(
                q, self.k[i][:, :, :end], self.v[i][:, :, :end], attn_mask=mask,
                is_causal=causal, scale=cfg.head_dim ** -0.5,
                enable_gqa=cfg.kv_heads != cfg.heads)
            a = a.transpose(1, 2).reshape(1, n, cfg.heads * cfg.head_dim)
            x = x + F.linear(a, w[p + "self_attn.o_proj.weight"])
            h = _norm(x, w[p + "post_attention_layernorm.weight"], cfg.eps)
            g = F.linear(h, w[p + "mlp.gate_proj.weight"])
            u = F.linear(h, w[p + "mlp.up_proj.weight"])
            x = x + F.linear(F.silu(g) * u, w[p + "mlp.down_proj.weight"])
        h = _norm(x[:, -1:], w["model.norm.weight"], cfg.eps)
        return F.linear(h, self.head)[0, -1].float()

    def generate(self, ids: list[int], max_new: int, eos, deadline: float | None = None,
                 clock=time.monotonic) -> dict:
        """Greedy: up to `max_new` tokens after `ids`, stopping at an id in `eos` (not returned),
        at the token cap, or once `clock()` passed `deadline` (checked before every step, so a
        step that started in time finishes). `finish`: stop, length or time."""
        if not ids:
            raise ValueError("an empty prompt")
        if len(ids) + max_new > self.max_ctx:
            raise ValueError(f"a prompt of {len(ids)} tokens and {max_new} new ones do not fit "
                             f"the cache's {self.max_ctx} rows")
        out: list[int] = []
        if deadline is not None and clock() >= deadline:
            return {"ids": out, "finish": "time", "prefilled": 0, "reused": 0}
        keep = 0
        for a, b in zip(self.ids, ids[:-1]):          # at least one row is read again
            if a != b:
                break
            keep += 1
        self.ids = self.ids[:keep]
        logits = self.forward(ids[keep:], keep)
        self.ids = list(ids)
        finish = "length"
        while True:
            t = int(torch.argmax(logits))
            if t in eos:
                finish = "stop"
                break
            out.append(t)
            if len(out) >= max_new:
                break
            if deadline is not None and clock() >= deadline:
                finish = "time"
                break
            logits = self.forward([t], len(self.ids))
            self.ids.append(t)
        return {"ids": out, "finish": finish, "prefilled": len(ids) - keep, "reused": keep}
