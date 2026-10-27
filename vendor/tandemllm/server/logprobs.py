"""`logprobs`: the log-probability of every emitted token, and its alternatives.

What is reported is the row the token was CHOSEN from, as the loop had it at the decision: the
target's logits after the deterministic logit processors (penalties, the no-repeat rule,
`logit_bias`) and before temperature, top-p, top-k and min_p -- the model's distribution under
the request's rules, not the sampling filter's. A verified block hands its rows over at once: in a
chain, row i is the token at position i; in a tree, the rows of the accepted path. The engine's own
tokens -- the phrase a reasoning budget forces -- were not chosen by the model at all, and carry
logprob 0 with themselves as the only alternative.

The recorder costs nothing unless a request asks: the loop checks for one per decision site, and
only a recording request pays for a log-softmax over its rows and one host copy a block.

Token text and bytes: the text is the token decoded on its own; the bytes are the token's own
bytes, read through the byte-level alphabet, so a token that holds half a UTF-8 character reports
the half (its text then shows the replacement character).
"""

from __future__ import annotations

import torch

FLOOR = -9999.0          # what a -inf log-probability is reported as (JSON has no infinity)


class Recorder:
    """`(token, logprob, [(alternative, logprob), ...])` per committed token, in emission order."""

    def __init__(self, top: int):
        self.top = int(top)
        self.entries: list[tuple[int, float, list[tuple[int, float]]]] = []

    @torch.no_grad()
    def rows(self, lg: torch.Tensor, toks: list[int]) -> None:
        """`lg[i]` is the row `toks[i]` was chosen from."""
        n = len(toks)
        lp = torch.log_softmax(lg[:n].float(), dim=-1)
        idx = torch.tensor([int(t) for t in toks], dtype=torch.long, device=lp.device)
        chosen = lp.gather(1, idx[:, None])[:, 0].tolist()
        if self.top:
            v, i = lp.topk(self.top, dim=-1)
            alts = [list(zip(ii, vv)) for ii, vv in zip(i.tolist(), v.tolist())]
        else:
            alts = [[] for _ in range(n)]
        self.entries.extend((int(t), c, a) for t, c, a in zip(toks, chosen, alts))

    def forced(self, toks: list[int]) -> None:
        """Tokens the engine wrote itself: probability one under the engine's rule."""
        self.entries.extend((int(t), 0.0, [(int(t), 0.0)] if self.top else []) for t in toks)


def _byte_decoder() -> dict[str, int]:
    """The byte-level BPE alphabet, inverted: each printable stand-in character -> its byte."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) \
        + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


_DECODER = _byte_decoder()


class Formatter:
    """Token text/bytes for a tokenizer, cached per id (a stream asks for the same ids often)."""

    def __init__(self, tok):
        self.tok = tok
        self.cache: dict[int, tuple[str, list[int]]] = {}

    def token(self, tid: int) -> tuple[str, list[int]]:
        got = self.cache.get(tid)
        if got is not None:
            return got
        raw = None
        conv = getattr(self.tok, "convert_ids_to_tokens", None)
        if conv is not None:
            piece = conv(int(tid))
            if isinstance(piece, str) and piece and all(c in _DECODER for c in piece):
                raw = bytes(_DECODER[c] for c in piece)
        if raw is None:
            raw = self.tok.decode([int(tid)], skip_special_tokens=False).encode("utf-8")
        got = (raw.decode("utf-8", errors="replace"), list(raw))
        if len(self.cache) < 65536:
            self.cache[tid] = got
        return got

    def _alt(self, tid: int, lp: float) -> dict:
        text, raw = self.token(tid)
        return {"token": text, "logprob": max(lp, FLOOR), "bytes": raw}

    def chat(self, entries) -> list[dict]:
        """The chat shape: `choices[].logprobs.content`."""
        out = []
        for tid, lp, alts in entries:
            e = self._alt(tid, lp)
            e["top_logprobs"] = [self._alt(a, v) for a, v in alts]
            out.append(e)
        return out

    def legacy(self, entries, offset: int = 0) -> dict:
        """The /v1/completions shape; `offset` is where in the text the first token starts."""
        tokens, lps, tops, offs = [], [], [], []
        for tid, lp, alts in entries:
            text, _ = self.token(tid)
            tokens.append(text)
            lps.append(max(lp, FLOOR))
            top = {}
            for a, v in alts:
                top.setdefault(self.token(a)[0], max(v, FLOOR))
            tops.append(top)
            offs.append(offset)
            offset += len(text)
        return {"tokens": tokens, "token_logprobs": lps, "top_logprobs": tops,
                "text_offset": offs}


def emitted(entries: list, ids: list[int], eos: set[int]) -> list:
    """The entries of the tokens a client sees: one per id, the end token left out."""
    n = len(ids) - (1 if ids and ids[-1] in eos else 0)
    return entries[:max(0, n)]
