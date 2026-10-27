"""The server log's per-request lines, read back: both factors of a request's speed, and where in
its blocks the draft went wrong.

Speed on this engine is committed tokens a block over block time. Since 2026-09-24 every `[req]`
line the server prints carries the decode loop's own count (`server/app.py::BlockStats`):

    [req] <id> json prompt=256 completion=256 finish=length 7712 ms 33.07 tok/s \
        blocks=78 committed=255 decode_ms=7289.4 accept=15:0x20,1x14,4x2|7:0x3,6x1

`blocks` is every forward of the decode loop (a verify, a declined single step, a forced close),
`committed` the tokens past the prefill's, `decode_ms` first token to last block. `accept` is the
first-miss histogram: per block that had a draft, `depth:accepted x count` -- how many draft tokens
the block could have accepted (a chain's length, a tree's depth) and how many it did. This module
is the one parser for all of it (`tools/row3.py`, `tools/accept_hist.py`).
"""

from __future__ import annotations

import re

REQ = re.compile(r"\[req\] (\S+) (?:stream|json) prompt=(\d+) completion=(\d+) finish=(\S+) "
                 r"(\d+) ms [\d.]+ tok/s(.*)$", re.M)
BLOCKS = re.compile(r" blocks=(\d+) committed=(\d+) decode_ms=([\d.]+) accept=(\S+)")
# a sampled request says its temperature; a greedy line has no field
TEMP = re.compile(r" temp=([\d.e+-]+)")


def parse_hist(token: str) -> dict[int, dict[int, int]]:
    """`8:2x5,8x1|16:4x2` -> {8: {2: 5, 8: 1}, 16: {4: 2}}; `-` -> {}."""
    out: dict[int, dict[int, int]] = {}
    if token == "-":
        return out
    for part in token.split("|"):
        key, cells = part.split(":", 1)
        out[int(key)] = {int(c): int(n) for c, n in (x.split("x") for x in cells.split(","))}
    return out


def add_hist(dst: dict[int, dict[int, int]], src: dict[int, dict[int, int]]) -> dict:
    for key, h in src.items():
        d = dst.setdefault(key, {})
        for c, n in h.items():
            d[c] = d.get(c, 0) + n
    return dst


def parse_requests(text: str) -> list[dict]:
    """One record per `[req]` line, in log order. A line from a server older than the block
    count, or a request that ended in its prefill, carries `blocks: None` -- not zero."""
    out = []
    for m in REQ.finditer(text):
        rec = {"cid": m.group(1), "prompt": int(m.group(2)), "completion": int(m.group(3)),
               "finish": m.group(4), "ms": int(m.group(5)), "blocks": None}
        t = TEMP.search(m.group(6))
        rec["temp"] = float(t.group(1)) if t is not None else 0.0
        b = BLOCKS.search(m.group(6))
        if b is not None and int(b.group(1)) > 0:
            rec.update(blocks=int(b.group(1)), committed=int(b.group(2)),
                       decode_ms=float(b.group(3)), accept=parse_hist(b.group(4)))
        out.append(rec)
    return out


def factors(reqs: list[dict]) -> dict:
    """Tokens a block and milliseconds a block over a set of requests.

    `tok_blk` / `ms_blk` pool the requests (all committed tokens over all blocks, all decode time
    over all blocks) -- the row's block, as `tools/block_budget.py` defines it. `tok_blk_p50` /
    `ms_blk_p50` are the medians of the per-request values, beside the row's p50. Empty when no
    request carried the count (a `--drafter none` server, an old log)."""
    have = [r for r in reqs if r.get("blocks")]
    if not have:
        return {}
    blocks = sum(r["blocks"] for r in have)
    per_tok = sorted(r["committed"] / r["blocks"] for r in have)
    per_ms = sorted(r["decode_ms"] / r["blocks"] for r in have)
    acc: dict[int, dict[int, int]] = {}
    for r in have:
        add_hist(acc, r["accept"])
    return {"requests": len(have), "blocks": blocks,
            "tok_blk": sum(r["committed"] for r in have) / blocks,
            "ms_blk": sum(r["decode_ms"] for r in have) / blocks,
            "tok_blk_p50": _median(per_tok), "ms_blk_p50": _median(per_ms),
            "accept": acc}


def curve(acc: dict[int, dict[int, int]], slots: int = 15) -> list[dict]:
    """A_i = P(draft slot i accepted | slots 1..i-1 were), censored: a block counts at slot i only
    if it HAD a slot i (depth >= i). Returns, per slot, the rate and the blocks it rests on."""
    out = []
    for i in range(1, slots + 1):
        reached = hit = 0
        for depth, h in acc.items():
            if depth < i:
                continue
            for a, n in h.items():
                if a >= i - 1:
                    reached += n
                    if a >= i:
                        hit += n
        out.append({"slot": i, "rate": hit / reached if reached else None, "n": reached})
    return out


def first_miss(acc: dict[int, dict[int, int]]) -> dict[int, int]:
    """How many blocks' first miss landed at each slot (1-based); a block that accepted its whole
    depth is counted under `depth + 1` (it never missed)."""
    out: dict[int, int] = {}
    for _depth, h in acc.items():
        for a, n in h.items():
            out[a + 1] = out.get(a + 1, 0) + n
    return dict(sorted(out.items()))


def _median(xs: list[float]) -> float:
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])
