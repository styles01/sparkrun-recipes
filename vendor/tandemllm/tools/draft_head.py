"""A small head for the drafter, so a proposed token stops costing 2.54 GB.

The prediction head in the checkpoint has no output head of its own, so every token it drafts is
0.48 GB of head plus a full pass over the target's `lm_head`: 248,320 x 5120 in bf16, 2.54 GB, or
9.4 % of a whole verify step, per drafted token. A three-deep draft spends 9.1 GB against the
verify step's 19.4 GB. That is the most expensive thing in the engine per unit of work it does.

It is also the easiest to remove, because a draft head cannot change what the engine outputs. The
verify step recomputes the target's own distribution over the full vocabulary and rejects anything
that disagrees, so a draft head that is wrong costs acceptance and nothing else. That makes it the
one place where an approximation needs no quality gate, only a throughput measurement.

What this builds: the rows of `lm_head` for the V most frequent tokens of a corpus, quantised to
NVFP4, plus the index that maps a row back to its token id. At V = 32,768 that is 94 MB instead of
2,543 MB -- 27x -- and the drafted token costs 0.57 GB instead of 3.02 GB.

The tokens left out are the tail. A tail token the drafter cannot propose is a token the block
stops at, and the verify step emits it; the cost is one shorter block, not a wrong output.

    python tools/draft_head.py --out ~/nvfp4/draft-head.safetensors --vocab 32768
"""

from __future__ import annotations

import argparse
import collections
import glob
import os
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.config import load_config  # noqa: E402
from tools.nvfp4_linear import NVFP4Block, quantize_to_nvfp4  # noqa: E402

# Enough text to rank a vocabulary: every Python file and package description in the interpreter
# this engine runs under. It is code-heavy, which matches what the engine is asked to generate,
# and it costs nothing to read.
DEFAULT_CORPUS = [
    "~/recipes/ling3-flash-dgx-spark/.venv/lib/python3.11/site-packages/transformers/models/*/modeling_*.py",
    "~/recipes/ling3-flash-dgx-spark/.venv/lib/python3.11/site-packages/*/*.py",
    "~/recipes/ling3-flash-dgx-spark/.venv/lib/python3.11/site-packages/*.dist-info/METADATA",
]


def collect_text(patterns: list[str], limit_mb: float) -> str:
    out = []
    total = 0
    budget = int(limit_mb * 1e6)
    for pat in patterns:
        for path in sorted(glob.glob(os.path.expanduser(pat))):
            try:
                with open(path, encoding="utf-8", errors="ignore") as f:
                    piece = f.read(200_000)
            except OSError:
                continue
            out.append(piece)
            total += len(piece)
            if total >= budget:
                return "".join(out)
    return "".join(out)


def rank_tokens(tok, text: str, vocab: int, chunk: int = 400_000) -> torch.Tensor:
    counts = collections.Counter()
    for i in range(0, len(text), chunk):
        ids = tok(text[i:i + chunk], add_special_tokens=False).input_ids
        counts.update(ids)
    ranked = [t for t, _ in counts.most_common()]
    keep = list(ranked[:vocab])
    seen = set(keep)
    # every special token, whatever its frequency: a draft that cannot propose an end-of-turn
    # token stops one block short at every turn boundary.
    for t in getattr(tok, "all_special_ids", []) or []:
        if t not in seen:
            keep.append(t)
            seen.add(t)
    # fill to a multiple of 128 with the next most frequent, so the kernel's N tiles are exact
    i = vocab
    while len(keep) % 128 or len(keep) < vocab:
        if i >= len(ranked):
            break
        if ranked[i] not in seen:
            keep.append(ranked[i])
            seen.add(ranked[i])
        i += 1
    cover = sum(counts[t] for t in keep) / max(1, sum(counts.values()))
    print(f"[rank] {len(counts)} distinct tokens in the corpus, keeping {len(keep)}, "
          f"covering {cover * 100:.3f} % of occurrences")
    return torch.tensor(sorted(keep), dtype=torch.int32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vocab", type=int, default=32768)
    ap.add_argument("--corpus-mb", type=float, default=24.0)
    ap.add_argument("--corpus", nargs="*", default=None)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    cfg = load_config(args.model)
    tok = AutoTokenizer.from_pretrained(cfg.path)
    text = collect_text(args.corpus or DEFAULT_CORPUS, args.corpus_mb)
    print(f"[corpus] {len(text) / 1e6:.1f} MB")
    index = rank_tokens(tok, text, args.vocab)

    head = None
    for path in sorted(glob.glob(os.path.join(cfg.path, "*.safetensors"))):
        with safe_open(path, framework="pt", device="cuda") as f:
            for key in f.keys():
                if key.endswith("lm_head.weight"):
                    head = f.get_tensor(key)
                    break
        if head is not None:
            break
    if head is None:
        raise SystemExit("lm_head.weight not found")
    rows = head[index.to(head.device).long()].contiguous()
    blk = quantize_to_nvfp4(rows.to(torch.bfloat16))
    del head, rows
    torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    save_file({"draft_head.weight": blk.w.cpu(),
               "draft_head.weight_scale": blk.s.cpu(),
               "draft_head.weight_scale_2": torch.tensor(blk.s2, dtype=torch.float32),
               "draft_head.index": index},
              args.out, metadata={"format": "nvfp4", "group": "16", "vocab": str(len(index))})
    full = 248320 * 5120 * 2
    print(f"[head] {len(index)} rows, {blk.nbytes / 1e6:.1f} MB against the full head's "
          f"{full / 1e6:.0f} MB ({full / blk.nbytes:.1f}x) -> {args.out}")


def load_draft_head(path: str, device: str = "cuda") -> tuple[NVFP4Block, torch.Tensor]:
    with safe_open(os.path.expanduser(path), framework="pt", device=device) as f:
        blk = NVFP4Block(f.get_tensor("draft_head.weight"),
                         f.get_tensor("draft_head.weight_scale"),
                         f.get_tensor("draft_head.weight_scale_2").float().item())
        index = f.get_tensor("draft_head.index").long()
    return blk, index


if __name__ == "__main__":
    main()
