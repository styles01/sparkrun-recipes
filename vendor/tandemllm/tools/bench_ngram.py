"""Measure the lookup drafter on the board, against whatever the best drafter is that day.

`tools/sim_draft.py` answers what a drafter accepts. It cannot answer what a step costs, and the
step cost is moving: the MLPs went to NVFP4 and the draft head was trimmed while this drafter was
being written. So the comparison has to be made in one process, on one weight set, in one sitting,
or it is comparing two afternoons.

The gate this run exists to settle:

    the mean over the five workloads must beat the best configuration that does not use the lookup
    drafter, measured here, and it must not lose more than 3 % on any single workload.

Everything about the run that is not the drafter is held fixed: the same five prompts as
`tools/bench_decode.py`, the same greedy decoding, the same number of new tokens. `--new 512` by
default rather than 128, because the 12:05 ledger entry is a bug that was invisible at 48 tokens
and halved acceptance at 128. A speculative decoder cannot be judged on a short generation.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.config import load_config  # noqa: E402
from engine.drafters.engram import EngramDrafter  # noqa: E402
from engine.drafters.mtp import MTPDrafter  # noqa: E402
from engine.drafters.ngram import NgramDrafter  # noqa: E402
from engine.loader import Weights  # noqa: E402
from engine.model import Qwen38Engine  # noqa: E402
from engine.router import (MTP_MS_PER_TOKEN, MTP_MS_PER_TOKEN_TRIMMED,  # noqa: E402
                           ROLLBACK_MS, ROLLBACK_MS_NVFP4, VERIFY_MS, VERIFY_MS_NVFP4,
                           MergedRouter)
from engine.spec import generate_greedy, generate_spec, generate_spec_tree  # noqa: E402
from tools.bench_decode import PROMPTS  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--new", type=int, default=512)
    ap.add_argument("--corpus", default=os.environ.get("QWEN38_CORPUS", ""))
    ap.add_argument("--mtp-depth", default="3", help="comma separated depths for the head")
    ap.add_argument("--depth", type=int, default=16, help="max lookup draft length")
    ap.add_argument("--budget", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--min-expected", type=float, default=0.6)
    ap.add_argument("--branch-top-k", type=int, default=3)
    ap.add_argument("--min-order", type=int, default=3)
    ap.add_argument("--min-corpus-order", type=int, default=5)
    ap.add_argument("--corpus-weight", type=float, default=0.5)
    ap.add_argument("--curve", default=None, choices=["fp8", "nvfp4"],
                    help="which measured verify curve the drafter and router price against; "
                         "defaults to nvfp4 when QWEN38_NVFP4 is set")
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--engram-v1", action="store_true", help="also run the first suffix memory")
    ap.add_argument("--dflash2", default=None,
                    help="comma separated block counts for the released block drafter; it is the "
                         "other candidate for best-without-the-lookup-drafter, so the gate is only "
                         "honest with it in the run when it works")
    ap.add_argument("--router-head", default="mtp", choices=["mtp", "dflash2"],
                    help="which neural drafter the router prices the lookup drafter against")
    ap.add_argument("--head-ms", type=float, default=None,
                    help="drafting cost per proposed token; defaults to the trimmed-head figure "
                         "when QWEN38_DRAFT_HEAD is set")
    ap.add_argument("--rollback-ms", type=float, default=None,
                    help="cost of one state replay; defaults to the figure for the weight set")
    ap.add_argument("--dflash2-ms", type=float, default=35.0,
                    help="cost of one block-drafter call; it is fixed per call, not per token "
                         "(35 ms a block against a 153 ms verify)")
    ap.add_argument("--dflash2-block", type=int, default=8)
    ap.add_argument("--think", action="store_true",
                    help="let the model reason before answering; off by default, because the bench "
                         "row this program is measured against runs with thinking off, and because "
                         "128 tokens of reasoning is 128 tokens the edit and quote regimes never "
                         "reach")
    ap.add_argument("--tree", action="store_true",
                    help="also run the tree-verify configurations: the block drafter's lattice as "
                         "a tree, and the router merging it with the lookup drafter's tree")
    ap.add_argument("--tree-budget", default="16",
                    type=lambda v: [int(x) for x in str(v).split(",") if x],
                    help="nodes per tree, anchor included; the measured curve is 145 ms at 4 and "
                         "175 at 16, so this is the knob the whole track turns")
    ap.add_argument("--df2-temp", type=float, default=1.0)
    ap.add_argument("--tree-mode", default="paths,nodes",
                    help="how the block drafter spends its node budget: whole branches (paths) or "
                         "best-first over marginals (nodes). Comma separated to run both.")
    ap.add_argument("--only", default=None)
    a = ap.parse_args()

    cfg = load_config(a.model)
    w = Weights(cfg.path, device=a.device, skip_mtp=False)
    eng = Qwen38Engine(cfg, w, max_len=a.max_len, device=a.device)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.path)
    eos = cfg.eos_token_ids
    depths = [int(x) for x in a.mtp_depth.split(",") if x]
    curve = a.curve or ("nvfp4" if os.environ.get("QWEN38_NVFP4") else "fp8")
    table = VERIFY_MS_NVFP4 if curve == "nvfp4" else VERIFY_MS
    keys = sorted(table)
    base_ms = table[keys[0]]
    per_node_ms = (table[keys[-1]] - table[keys[0]]) / (keys[-1] - keys[0])
    head_ms = a.head_ms if a.head_ms is not None else (
        MTP_MS_PER_TOKEN_TRIMMED if os.environ.get("QWEN38_DRAFT_HEAD") else MTP_MS_PER_TOKEN)
    rollback_ms = a.rollback_ms if a.rollback_ms is not None else (
        ROLLBACK_MS_NVFP4 if curve == "nvfp4" else ROLLBACK_MS)
    print(f"pricing against the {curve} verify curve: {base_ms:.1f} ms + {per_node_ms:.3f} ms/node, "
          f"head {head_ms:.1f} ms/token, rollback {rollback_ms:.1f} ms")

    def make_ngram() -> NgramDrafter:
        return NgramDrafter(corpus_path=a.corpus, min_order=a.min_order, max_depth=a.depth,
                            node_budget=a.budget, branch_top_k=a.branch_top_k,
                            min_expected=a.min_expected, alpha=a.alpha,
                            corpus_weight=a.corpus_weight, min_corpus_order=a.min_corpus_order,
                            verify_base_ms=base_ms, verify_per_node_ms=per_node_ms)

    if a.corpus:
        probe = make_ngram()
        if probe.corpus is None:
            print(f"warning: no corpus store at {a.corpus}; the drafter is local-only")
        else:
            print(f"corpus store: {probe.corpus.n:,} tokens from {a.corpus}")

    rows: list[tuple[str, str, float, float, float, int]] = []
    for name, text in PROMPTS.items():
        if a.only and name not in a.only.split(","):
            continue
        ids = tok(tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                          add_generation_prompt=True,
                                          enable_thinking=a.think),
                  return_tensors="pt").input_ids[0].to(a.device)
        print(f"\n### {name}  ({ids.numel()} prompt tokens)")

        def record(label: str, st) -> None:
            print("   ", st.line(label))
            rows.append((name, label, st.tok_s, st.accept_len, st.accept_rate, st.tokens))

        if a.baseline:
            _, st = generate_greedy(eng, ids, a.new, eos)
            record("none", st)
        if a.engram_v1:
            _, st = generate_spec(eng, ids, a.new, EngramDrafter(), 8, eos)
            record("engram-v1", st)
        for d in depths:
            md = MTPDrafter(eng, max_len=a.max_len, depth=d)
            _, st = generate_spec(eng, ids, a.new, md, d, eos)
            record(f"mtp d={d}", st)
        ng = make_ngram()
        _, st = generate_spec(eng, ids, a.new, ng, a.depth, eos)
        record("ngram", st)
        print(f"      fired {ng.stats['fired']}/{ng.stats['calls']} "
              f"({100 * ng.stats['fired'] / max(ng.stats['calls'], 1):.1f} %), "
              f"orders {dict(sorted(ng.stats['order_hist'].items(), reverse=True))}, "
              f"sources {ng.stats['source_hist']}")
        for b in ([int(x) for x in a.dflash2.split(",")] if a.dflash2 else []):
            from engine.drafters.dflash2 import DFlash2Drafter
            dd = DFlash2Drafter(eng, blocks=b, max_len=a.max_len)
            _, st = generate_spec(eng, ids, a.new, dd, 8 * b, eos)
            record(f"dflash2 b={b}", st)
        if a.tree:
            from engine.drafters.dflash2 import DFlash2Drafter
            modes = [m for m in a.tree_mode.split(",") if m]
            for mode in modes:
                for b in a.tree_budget:
                    dd = DFlash2Drafter(eng, blocks=1, max_len=a.max_len)
                    dd.tree_temp, dd.tree_mode = a.df2_temp, mode
                    _, st = generate_spec_tree(eng, ids, a.new, dd, b - 1, eos)
                    record(f"df2-{mode} b={b}", st)
            for b in a.tree_budget:
                head = DFlash2Drafter(eng, blocks=1, max_len=a.max_len)
                head.tree_temp, head.tree_mode = a.df2_temp, modes[0]
                rt = MergedRouter(make_ngram(), head, mtp_depth=a.dflash2_block,
                                  node_budget=b - 1, mtp_ms_per_token=0.0,
                                  head_fixed_ms=a.dflash2_ms, adaptive_depth=False,
                                  rollback_ms=rollback_ms, verify_ms_table=table)
                _, st = generate_spec_tree(eng, ids, a.new, rt, b - 1, eos)
                record(f"router-tree b={b}", st)
                print(f"      chose ngram {rt.stats.get('ngram', 0)}x, head "
                      f"{rt.stats.get('mtp', 0)}x, merged {rt.stats.get('merged', 0)}x, "
                      f"head skipped {rt.stats.get('head_skipped', 0)}x, "
                      f"calibration lookup {rt.calib.value:.2f} head {rt.calib_head.value:.2f}")
        for d in depths:
            if a.router_head == "dflash2":
                from engine.drafters.dflash2 import DFlash2Drafter
                head, head_depth = DFlash2Drafter(eng, blocks=1, max_len=a.max_len), a.dflash2_block
                kw = dict(mtp_ms_per_token=0.0, head_fixed_ms=a.dflash2_ms, adaptive_depth=False)
            else:
                head, head_depth = MTPDrafter(eng, max_len=a.max_len, depth=d), d
                kw = dict(mtp_ms_per_token=head_ms)
            rt = MergedRouter(make_ngram(), head, mtp_depth=head_depth, node_budget=a.budget,
                              rollback_ms=rollback_ms, verify_ms_table=table, **kw)
            _, st = generate_spec(eng, ids, a.new, rt, a.depth, eos)
            record(f"router/{a.router_head} d={head_depth}", st)
            print(f"      chose ngram {rt.stats['ngram']}x ({rt.stats['ngram_tokens']} tok), "
                  f"head {rt.stats['mtp']}x ({rt.stats['mtp_tokens']} tok), "
                  f"calibration {rt.calib.value:.2f}")

    print("\n" + "=" * 86)
    print(f"{'workload':10s} {'drafter':14s} {'tok/s':>8} {'acc/block':>10} {'draft acc':>10} "
          f"{'tok':>6}")
    for name, label, tps, acc, rate, n in rows:
        print(f"{name:10s} {label:14s} {tps:8.2f} {acc:10.2f} {rate * 100:9.1f}% {n:6d}")

    labels: list[str] = []
    for _, label, *_ in rows:
        if label not in labels:
            labels.append(label)
    workloads = sorted({r[0] for r in rows})
    print(f"\n{'drafter':14s} " + " ".join(f"{k:>8}" for k in workloads) + f" {'mean':>8}")
    means: dict[str, float] = {}
    per: dict[str, dict[str, float]] = {}
    for label in labels:
        vals = {n: t for n, la, t, *_ in rows if la == label}
        per[label] = vals
        row = [vals.get(k, 0.0) for k in workloads]
        means[label] = statistics.fmean(row) if row else 0.0
        print(f"{label:14s} " + " ".join(f"{v:8.2f}" for v in row) + f" {means[label]:8.2f}")

    lookup = [la for la in labels if la.startswith(("ngram", "router", "engram"))]
    # `df2-tree` is not a lookup-drafter configuration, so it sits on the baseline side of the gate:
    # the thing a merged tree has to beat is the best configuration without the lookup drafter, and
    # once the block drafter can propose a tree that is the block drafter's tree.
    other = [la for la in labels if la not in lookup and la != "none"]
    if lookup and other:
        best_other = max(other, key=lambda la: means[la])
        print(f"\ngate: best configuration without the lookup drafter is {best_other} "
              f"at {means[best_other]:.2f} tok/s mean")
        for la in lookup:
            worst = min(((per[la].get(k, 0.0) / per[best_other][k] - 1.0) * 100, k)
                        for k in workloads if per[best_other].get(k))
            verdict = "PASS" if means[la] > means[best_other] and worst[0] > -3.0 else "fail"
            print(f"  {la:14s} mean {means[la]:6.2f} "
                  f"({(means[la] / means[best_other] - 1) * 100:+5.1f} %), "
                  f"worst workload {worst[1]} {worst[0]:+5.1f} %  {verdict}")


if __name__ == "__main__":
    main()
