# This file is from the Laya SDK 0.3.4 (`laya/common.py`, class `DecisionModel`), Copyright
# Convai Innovations, licensed under the Apache License 2.0 (engine/decision/laya/LICENSE-APACHE-2.0;
# NOTICE at the repository root). Changes: renamed `LayaNet` (the engine's `DecisionModel` is the
# runtime interface), the training-only `detach_encoder` path and checkpointing flag removed. The
# module tree, and so every state-dict key the checkpoints store, is unchanged.
"""Laya's network: a bidirectional encoder plus the typed decision head.

The encoder's last hidden state gets a per-type embedding, goes through 2 norm-first transformer
layers (4d feed-forward), and the output at each option's `[MASK]` marker is scored by a small MLP
(LayerNorm, Linear, GELU, Linear) into one logit per option; markers past the option count are
masked to -1e4. The act head reads the `[CLS]` output with four features of the option
distribution (top-1, top-1 minus top-2, normalised entropy, option count / 255) in fp32.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LayaNet(nn.Module):
    def __init__(self, encoder: nn.Module, head_layers: int = 2, n_act: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        d = encoder.config.hidden_size
        nhead = max(1, d // 64)
        layer = nn.TransformerEncoderLayer(d, nhead, 4 * d, dropout, batch_first=True,
                                           norm_first=True)
        self.head = (nn.TransformerEncoder(layer, head_layers, enable_nested_tensor=False)
                     if head_layers > 0 else None)
        self.type_emb = nn.Embedding(3, d)
        self.scorer = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.act_head = nn.Sequential(nn.Linear(d + 4, 256), nn.GELU(), nn.Linear(256, n_act))
        self.register_buffer("temperature", torch.ones(3))

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        h = h + self.type_emb(qtype)[:, None, :]
        if self.head is not None:
            pad = ~attention_mask.bool()
            for layer in self.head.layers:
                h = layer(h, src_key_padding_mask=pad)
        idx = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
        m = torch.gather(h, 1, idx)
        logits = self.scorer(m).squeeze(-1).float()
        logits = logits.masked_fill(~marker_mask, -1e4)

        p = torch.softmax(logits.detach(), -1)
        k = marker_mask.sum(-1).clamp(min=2).float()
        ent = -(p * torch.log(p.clamp_min(1e-9))).sum(-1) / torch.log(k)
        top2 = p.topk(2, -1).values
        feats = torch.stack([top2[:, 0], top2[:, 0] - top2[:, 1], ent, k / 255.0], -1)
        pooled = h[:, 0].float()
        act_logits = self.act_head(torch.cat([pooled, feats], -1))
        return logits, act_logits
