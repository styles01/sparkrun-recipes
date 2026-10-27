"""The Qwen3.5 MoE forward in plain torch: slow, exact, the reference the engine is checked against.

It follows `transformers.models.qwen3_5_moe` (5.x) operation for operation. The mixers and the MoE are
the ones Flash-Next's reference already implements (engine/models/flashnext/ref.py: the GDN layer with
its gated norm, the gated attention with partial rotary, the softmax router with renormalised top-k and
the sigmoid-gated shared expert); what differs is the residual stream around them, one stream with
pre-norms:

    h = embed(tokens)
    per layer:  h = h + mixer(rms(h) (1 + w_in))          GDN or gated attention
                h = h + moe(rms(h) (1 + w_post))
    logits = head(rms(h) (1 + w_norm))

and the MTP layer (one attention layer with its MoE):

    x = fc([rms(embed(t_next)) (1 + w_e) ; rms(h_prev) (1 + w_h)])
    x = layer(x);  y = rms(x) (1 + w_mtp_norm);  logits = head(y)

`h_prev` is the trunk's post-final-norm row (`stack`), and a deeper draft step feeds the MTP layer's own
`y` back (engine/drafters/mtp.py's "post" choice for the 27B, the same family).
"""

from __future__ import annotations

import torch

from engine.models.flashnext.ref import MTP, FlashNextRef, FNState


class Qwen35MoeRef(FlashNextRef):
    """`W` maps canonical names (language-model prefix removed) to tensors, EXL3Blocks and ExpertBanks."""

    def __init__(self, cfg, W: dict, dtype=torch.bfloat16, device="cpu"):
        super().__init__(cfg, W, {}, dtype=dtype, device=device)

    def norm(self, h: torch.Tensor, name: str) -> torch.Tensor:
        return self.rms(h, self.w(name), self.cfg.rms_norm_eps)

    def layer(self, h, li, tokens, st, positions):
        p = self.pre(li)
        x = self.norm(h, p + "input_layernorm.weight")
        if li != MTP and self.cfg.is_linear(li):
            out = self.gdn(x, li, st)
        else:
            out = self.attention(x, li, st, positions)
        h = h + out
        return h + self.moe(self.norm(h, p + "post_attention_layernorm.weight"), li)

    @torch.no_grad()
    def forward(self, tokens: torch.Tensor, st: FNState, layers=None, trace: list | None = None,
                head: str = "all") -> torch.Tensor | None:
        """tokens [T] -> logits [T, vocab] (fp32); advances `st` by T. `trace` receives every layer's
        output; `stack` is the post-final-norm rows (the MTP layer's input)."""
        tokens = tokens.view(-1)
        T = tokens.shape[0]
        positions = torch.arange(st.length, st.length + T, device=self.device)
        h = self.embed(tokens)
        for li in (layers if layers is not None else range(self.cfg.num_hidden_layers)):
            h = self.layer(h, li, tokens, st, positions)
            if trace is not None:
                trace.append(h)
        st.length += T
        n = self.norm(h, "norm.weight")
        self.stack = n
        if head == "none":
            return None
        x = n[-1:] if head == "last" else n
        return self.lin(x, self.w("lm_head.weight")).float()

    @torch.no_grad()
    def mtp(self, stack: torch.Tensor, next_tokens: torch.Tensor, st: FNState, head=None):
        """One MTP step over T positions: `stack` [T, H] the trunk's post-norm rows (or the previous MTP
        step's `y`) at positions st.length.., `next_tokens` [T] the tokens after them. Returns (logits
        [T, vocab] of the token after next, y [T, H] for a deeper step); advances `st`."""
        next_tokens = next_tokens.view(-1)
        T = next_tokens.shape[0]
        e = self.norm(self.embed(next_tokens), "mtp.pre_fc_norm_embedding.weight")
        hs = self.norm(stack.to(self.dtype), "mtp.pre_fc_norm_hidden.weight")
        x = self.lin(torch.cat([e, hs], dim=-1), self.w("mtp.fc.weight"))
        positions = torch.arange(st.length, st.length + T, device=self.device)
        x = self.layer(x, MTP, next_tokens, st, positions)
        st.length += T
        y = self.norm(x, "mtp.norm.weight")
        if head == "x":
            return y, y
        if head is not None:
            return head(y), y
        return self.lin(y, self.w("lm_head.weight")).float(), y
