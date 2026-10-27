"""How far do a verify block's rows drift from plain decoding's, and does a constraint add to it?

`tools/verify_spec.py` compares tokens; this compares the rows the tokens were chosen from. Plain
greedy decodes one token a forward; the adversarial drafter makes every token come from row 0 of a
`k + 1`-row verify block followed by a rollback -- the same arithmetic the served loop's chain uses,
at its most frequent. Both runs record their RAW rows (a recorder placed before any constraint in
the processor chain), and for every position where the two runs still share their prefix the tool
reports the largest difference over the vocabulary and between the two leading tokens.

Run with and without `--constraint` on the same prompt: a mask sets -inf on the tokens it forbids
and leaves every other logit as the forward wrote it, so if the constraint only changes WHICH race
is run, the raw drift is the same size with and without it, and the flip at the first divergent
position is visible in the raw rows themselves. That is the isolating test for a divergence the
lossless gate reports under a constraint.

    python tools/row_drift.py --chat --prompt prose --new 96 --k 15 --constraint json_object \\
        --json results/api/drift-prose-json.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class RawRows:
    """A logit processor that changes nothing and keeps a CPU copy of the rows it is shown."""

    def __init__(self):
        self.blocks: list[tuple[str, torch.Tensor]] = []
        self.mask = True

    def seed(self, ids):
        self.blocks = []

    def commit(self, ids):
        pass

    def apply_single(self, row):
        self.blocks.append(("single", row.float().cpu().unsqueeze(0)))

    def apply_chain(self, lg, draft):
        self.blocks.append(("chain", lg.float().cpu()))

    def apply_tree(self, lg, tree):
        raise SystemExit("row_drift compares chains")


def rows_by_position(rec: RawRows, per_block: list[int]) -> list[torch.Tensor]:
    """The row each committed token was chosen from, in order: a single step's row, or the first
    `per_block[b]` rows of verify block b (row j decides the block's token j)."""
    out, b = [], 0
    for kind, rows in rec.blocks:
        if kind == "single":
            out.append(rows[0])
        else:
            out += [rows[j] for j in range(per_block[b])]
            b += 1
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--chat", action="store_true")
    ap.add_argument("--prompt", default="prose")
    ap.add_argument("--new", type=int, default=96)
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--constraint", default="", help="json_object or a regex; empty = none")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    from engine import grammar as G
    from engine.drafters.fixed import AdversarialDrafter
    from engine.spec import generate_greedy, generate_spec
    from tools.verify_spec import PROMPTS, bf16_ulp, build, encode
    cfg, _, eng, tok = build(a)
    ids = encode(tok, PROMPTS[a.prompt], a.chat, a.device, think=not a.constraint)

    def chain(rec):
        if not a.constraint:
            return rec
        pattern = G.json_object_regex(3) if a.constraint == "json_object" else a.constraint
        eos = {tok.eos_token_id} if isinstance(tok.eos_token_id, int) else set()
        gen = os.path.join(cfg.path, "generation_config.json")
        if os.path.isfile(gen):
            e = json.load(open(gen)).get("eos_token_id")
            eos |= set(e) if isinstance(e, list) else ({e} if isinstance(e, int) else set())
        gram = G.Grammar(pattern, G.Vocab.from_tokenizer(tok, cfg.vocab_size))
        return G.LogitChain([rec, G.Constraint(gram, eos, a.device)])

    rg, rs = RawRows(), RawRows()
    base, _ = generate_greedy(eng, ids, a.new, pen=chain(rg))
    adv = AdversarialDrafter(cfg.vocab_size, seed=1)
    got, st = generate_spec(eng, ids, a.new, adv, a.k, pen=chain(rs))
    Rg = rows_by_position(rg, [])
    Rs = rows_by_position(rs, st.per_block)
    d = next((i for i, (x, y) in enumerate(zip(base, got)) if x != y), None)
    upto = min(len(Rg), len(Rs), d + 1 if d is not None else len(base))
    rows = []
    for p in range(upto):
        g, s = Rg[p], Rs[p]
        top = torch.topk(g, 2).indices.tolist()
        rows.append({"pos": p, "token": base[p],
                     "max_abs": float((g - s).abs().max()),
                     "top1": float(g[top[0]]), "gap_greedy": float(g[top[0]] - g[top[1]]),
                     "gap_spec": float(s[top[0]] - s[top[1]])})
    out = {"prompt": a.prompt, "constraint": a.constraint or None, "k": a.k, "new": a.new,
           "divergence": d, "positions": rows,
           "max_abs_drift": max(r["max_abs"] for r in rows),
           "max_abs_drift_first_20": max(r["max_abs"] for r in rows[:20])}
    if d is not None and d < len(Rs):
        x, y = base[d], got[d]
        g, s = Rg[d], Rs[d]
        ulp = bf16_ulp(float(g[x]))
        out["flip"] = {"pos": d, "greedy": [x, tok.decode([x])], "spec": [y, tok.decode([y])],
                       "raw_greedy_row": {"greedy_tok": float(g[x]), "spec_tok": float(g[y])},
                       "raw_spec_row": {"greedy_tok": float(s[x]), "spec_tok": float(s[y])},
                       "raw_argmax_greedy_row": int(g.argmax()),
                       "raw_argmax_spec_row": int(s.argmax()),
                       "ulp": ulp,
                       "greedy_margin_ulp": float(g[x] - g[y]) / ulp,
                       "spec_margin_ulp": float(s[y] - s[x]) / ulp}
    print(json.dumps({k: v for k, v in out.items() if k != "positions"}, indent=1))
    worst = sorted(rows, key=lambda r: -r["max_abs"])[:5]
    for r in worst:
        print(f"  pos {r['pos']:3d} max |drift| {r['max_abs']:.4f} "
              f"(top1 {r['top1']:.2f}, one ulp {bf16_ulp(r['top1']):.4f})")
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
