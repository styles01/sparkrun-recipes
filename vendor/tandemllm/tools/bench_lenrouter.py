"""Fixed eight, fixed sixteen, and the router, on the five workloads, in one process.

The comparison this tool exists for is the one phase 5 could not make: the 17:14 table was three
separate runs of `bench_decode.py`, and the router has to be measured against both fixed lengths on
the same weights, the same prompts and the same allocation history or the comparison is with a
different afternoon.

Both fixed baselines run through `LengthRouter` with `fixed=` set rather than through
`DFlash2Drafter` directly, so the three rows differ in the routing policy and in nothing else --
not in which object holds the tap, not in how many drafters are resident, not in the cost of
keeping two draft caches current. A baseline that does not pay a cost the routed run pays would
flatter the router by exactly that cost.

**Thinking is off.** The template's default is on, the atlas row sends `enable_thinking: false`, and
the difference between the two regimes reached 72 % on the same prompt with the same drafter.
Every number here is the regime the row is measured in.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.config import load_config  # noqa: E402
from engine.drafters.dflash2 import DFlash2Drafter  # noqa: E402
from engine.lenrouter import LengthRouter  # noqa: E402
from engine.loader import Weights  # noqa: E402
from engine.model import Qwen38Engine  # noqa: E402
from engine.spec import generate_spec, generate_spec_tree  # noqa: E402
from tools.bench_decode import PROMPTS  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--new", type=int, default=256)
    ap.add_argument("--ckpt8", required=True)
    ap.add_argument("--ckpt16", required=True)
    ap.add_argument("--nvfp4", default=None)
    ap.add_argument("--fp8-head", default=None)
    ap.add_argument("--only", default=None)
    ap.add_argument("--configs", default="fixed8,fixed16,router",
                    help="any of fixed8, fixed16, router, drop, nodes, alias, both, mix3, mix4, "
                         "and Phase 2's knob sets. "
                         "`drop` releases the arm that loses the latch; `nodes`, `alias` and "
                         "`both` are the two lossless tree flags the atlas row cannot resolve, "
                         "paired against `router` in this process rather than across afternoons. "
                         "Phase 2 (2026-09-24): a label of '+'-joined knobs is the served router "
                         "with those knobs -- `deep` (32 rows after two full blocks), "
                         "`wN` / `nN` (the wide / narrow arm's tree budget, N nodes), "
                         "`nodes` / `paths` and `tNN` (the builder and the selector "
                         "temperature NN/10), `aN` (the wide tree only after N committed tokens), "
                         "e.g. `n16+w24+nodes+a32`; a budget past 16 and `deep` need "
                         "QWEN38_VERIFY_ROWS=32")
    ap.add_argument("--latch", action="store_true",
                    help="one width decision a request instead of one a block; see "
                         "engine/lenrouter.py::_choose_latched")
    ap.add_argument("--explore", type=int, default=32)
    ap.add_argument("--no-trim", action="store_true",
                    help="turn off the per-step width choice made from the wide drafter's own "
                         "lattice, leaving only the choice of which drafter runs")
    ap.add_argument("--tree", action="store_true",
                    help="verify a TREE per step instead of a chain. Each arm becomes a "
                         "MergedRouter -- the lookup drafter's tree and the block drafter's "
                         "lattice in one node set -- and the length router chooses the node budget")
    ap.add_argument("--corpus", default=os.environ.get("QWEN38_CORPUS", ""))
    ap.add_argument("--snapshot-bytes", action="store_true",
                    help="after each configuration, capture the state snapshot the serving cache "
                         "would store for this sequence and print its parts. This is where the "
                         "phase-10 idle drop shows up that the five workloads cannot see: an arm "
                         "released at the latch is not in the snapshot, and the snapshot is what "
                         "decides how many entries fit in the cache budget")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--repeat", type=int, default=1,
                    help="run the whole sweep this many times. A paired comparison has a "
                         "run-to-run spread of its own, and one repeat of it decides nothing for "
                         "the same reason one atlas row does not")
    a = ap.parse_args()

    wanted_early = [c.strip() for c in a.configs.split(",") if c.strip()]
    if "drop" in wanted_early and not a.latch:
        raise SystemExit("--configs drop needs --latch: the arm is released when the latch closes, "
                         "and without the latch the `drop` column would be a copy of `router`")

    cfg = load_config(a.model)
    w = Weights(cfg.path, device=a.device, skip_mtp=True, nvfp4=a.nvfp4, fp8_head=a.fp8_head)
    eng = Qwen38Engine(cfg, w, max_len=a.max_len, device=a.device)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.path)
    eos = cfg.eos_token_ids
    print(f"[bench] {w.report()}")

    small = DFlash2Drafter(eng, os.path.expanduser(a.ckpt8), blocks=1, max_len=a.max_len, block=8)
    small._build()
    large = DFlash2Drafter(eng, os.path.expanduser(a.ckpt16), blocks=1, max_len=a.max_len, block=16)
    large._build()

    def build_router():
        if not a.tree:
            return LengthRouter(small, large, explore_period=a.explore,
                                width_trim=(not a.no_trim) and not a.latch, latch=a.latch)
        from engine.drafters.ngram import NgramDrafter
        from engine.router import MergedRouter, served_tree_table, tree_nodes
        # Track B's measured NVFP4 tiles, not the 11:47 curve: 16 nodes verify in 129.2 ms where
        # they used to take 164.4, and 24 is a bucket the kernel pays a whole second tile for.
        tree_table = served_tree_table()
        ng = NgramDrafter(corpus_path=a.corpus, min_order=3, max_depth=16, node_budget=15,
                          branch_top_k=3, min_expected=0.2, alpha=0.6, corpus_weight=0.5,
                          min_corpus_order=8, verify_base_ms=tree_table[8],
                          verify_per_node_ms=(tree_table[16] - tree_table[8]) / 8)
        arms = []
        for head, budget in ((small, small.cfg.block_size - 1),
                             (large, large.cfg.block_size - 1)):
            arms.append(MergedRouter(ng, head, mtp_depth=budget,
                                     node_budget=tree_nodes(head.cfg.block_size) - 1,
                                     mtp_ms_per_token=0.0, head_fixed_ms=27.0,
                                     adaptive_depth=False, rollback_ms=6.4,
                                     verify_ms_table=dict(tree_table),
                                     tree_ms_table=dict(tree_table)))
        return LengthRouter(arms[0], arms[1], explore_period=a.explore, tree=True, ngram=ng,
                            latch=a.latch, latch_table=dict(tree_table))

    run_one = generate_spec_tree if a.tree else generate_spec

    # Warm up BOTH widths before anything is timed, through the verify path this run will use, and
    # then throw the router away.
    #
    # The first run of the first process autotunes Triton, and the first table this tool printed
    # says how much that is worth: the `prose` row, which ran first, read a verify of 260.2 ms and a
    # draft of 68.5 ms against the 115.5 and 26.8 every later row settled at, and its fixed-8
    # figure came out at 11.95 tok/s against phase 5's 18.53 on the same prompt with the same
    # checkpoint. A warm-up is not a nicety here: the first configuration measured was 35 % slower
    # than itself, and the router looked 35 % better than a baseline that was paying for the
    # compiler. The ledger has this trap from phase 4, on prefill, and it is the same one.
    warm = build_router()
    warm.learn_cost = False
    warm_ids = tok(tok.apply_chat_template([{"role": "user", "content": PROMPTS["prose"]}],
                                           tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False),
                   return_tensors="pt").input_ids[0].to(a.device)
    for width in (8, 16):
        warm.fixed = width
        run_one(eng, warm_ids, 48, warm, large.cfg.block_size - 1, eos)
    print("[bench] warmed both widths, 2 x 48 tokens; the cost estimates start from here")

    router = build_router()

    wanted = [c.strip() for c in a.configs.split(",") if c.strip()]
    rows = []
    for rep in range(1, a.repeat + 1):
        if a.repeat > 1:
          print(f"\n########## repeat {rep} of {a.repeat}")
        for name, text in PROMPTS.items():
          if a.only and name != a.only:
              continue
          ids = tok(tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                            add_generation_prompt=True, enable_thinking=False),
                    return_tensors="pt").input_ids[0].to(a.device)
          print(f"\n### {name}  ({ids.numel()} prompt tokens, thinking off)")
          # Every even repeat runs the configurations in the opposite order. Within a workload the
          # configurations run back to back in a fixed sequence, and a difference of a fraction of a
          # per cent is exactly the size of an order effect -- the first run after a prompt change
          # pays for whatever the last one left in cache. Reversing on alternate repeats makes the
          # order a thing that averages out instead of a thing that adds.
          named = ("fixed8", "fixed16", "router", "drop", "nodes", "alias", "both", "mix3", "mix4")
          order = [c for c in (("fixed8", 8), ("fixed16", 16), ("router", 0),
                               ("drop", 0), ("nodes", 0), ("alias", 0), ("both", 0),
                               ("mix3", 0), ("mix4", 0)) if c[0] in wanted]
          order += [(c, 0) for c in wanted if c not in named]
          if rep % 2 == 0:
              order.reverse()
          for label, fixed in order:
              router.fixed = fixed
              # The two lossless flags phase 9 could not resolve with one row each, paired here
              # instead. Both are worth a fraction of a per cent -- `nodes` read +0.2 % on this same
              # bench and the clone skip took the block 134.613 -> 134.344 ms -- and the atlas row's
              # own run-to-run spread is 10 %, so the row can never settle either of them however
              # many times it is run. A paired comparison in one process on the same prompts can.
              import engine.model as _M
              knobs = phase2_knobs(label)
              mode = "nodes" if label in ("nodes", "both") else knobs["mode"]
              for _h in (router.head_small, router.head_large):
                  _h.tree_mode = mode
              _M.TREE_ALIAS_STATE = label in ("alias", "both")
              # `drop` is `router` with the arm that loses the latch released: same policy, same
              # decisions, one drafter stops being kept current. Paired in one process against
              # `router` because the thing it is worth is about a per cent of a block, and a per cent
              # does not survive being measured on two afternoons.
              router.drop_idle = label == "drop" or "drop" in label.split("+")
              # `lp`: the latch priced on the tree each arm will run (QWEN38_LATCH_PRICE)
              router.latch_price = "lp" in label.split("+")
              if router.latch_price:
                  router.stats.setdefault("price", "-")
              # `mixN` pins the width on a fixed schedule -- one narrow block in every N, chosen by
              # the block counter and nothing else. It isolates the COST OF SWITCHING from the cost
              # of choosing badly: it switches as often as the router does and it knows nothing.
              router.mix_period = int(label[3:]) if label.startswith("mix") else 0
              # Phase 2: the label's knobs, the served router otherwise, restored for the next label
              router.deep = knobs["deep"]
              if a.tree:
                  router.large.node_budget = router.large.head_budget = knobs["wide"] - 1
                  router.small.node_budget = router.small.head_budget = knobs["narrow"] - 1
                  router.wide_budget = knobs["wide"] - 1
                  router.tree_wide_after = knobs["after"]
              for _h in (router.head_small, router.head_large):
                  _h.tree_temp = knobs["temp"]
              router.attach()
              # the loop's depth cap is the deep width when the deep chain is on, as the server's
              k_loop = max(large.cfg.block_size - 1, router.deep - 1)
              _, st = run_one(eng, ids, a.new, router, k_loop, eos)
              print("   ", st.line(label))
              print("     ", router.report())
              blk_ms = st.decode_s * 1e3 / st.blocks if st.blocks else 0.0
              snap_parts, snap_bytes = None, 0
              if a.snapshot_bytes:
                  from engine.cache import capture
                  snap = capture(eng, router)
                  snap_parts, snap_bytes = snap.parts(), snap.nbytes
                  gb = 1024 ** 3
                  print("      snapshot " + "  ".join(f"{k} {v / gb:.3f} GB"
                                                      for k, v in snap_parts.items())
                        + f"  total {snap_bytes / gb:.3f} GB"
                        + f"  -> {int(24 * gb // snap_bytes) if snap_bytes else 0} entries in 24 GiB")
              rows.append({"repeat": rep, "workload": name, "config": label, "tok_s": st.tok_s,
                           "accept_len": st.accept_len, "accept_rate": st.accept_rate,
                           "blocks": st.blocks, "tokens": st.tokens,
                           "small": router.stats["small"], "large": router.stats["large"],
                           "trims": router.stats["trims"], "forced": router.stats["forced"],
                           "width_hist": dict(router.stats["width_hist"]),
                           "latched": router.stats["latched"], "idle": router.stats["idle"],
                           "tree_mode": mode, "alias_state": bool(_M.TREE_ALIAS_STATE),
                           "block_ms": blk_ms, "draft_ms": st.draft_s * 1e3 / max(st.blocks, 1),
                           "snapshot_bytes": snap_bytes, "snapshot_parts": snap_parts})
              # `generate_spec` calls `reset()` at the top of every run, which clears the arms, the
              # ceiling rate and the calibration and keeps the learned costs -- so each row here
              # starts from the same cold policy a request gets, and the cost constants improve
              # across the sweep exactly as they would in a long-lived server.

    hdr = [c for c in ("fixed8", "fixed16", "router", "drop", "nodes", "alias", "both",
                       "mix3", "mix4") if c in wanted]
    hdr += [c for c in wanted if c not in hdr]
    _summary(rows, hdr, a.repeat)
    if a.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(a.json_out)), exist_ok=True)
        json.dump(rows, open(a.json_out, "w"), indent=1)
        print(f"[bench] -> {a.json_out}")


def phase2_knobs(label: str) -> dict:
    """A label's knobs (see --configs): '+'-joined `deep`, `wN`, `nN`, `nodes`, `paths`, `tNN`;
    anything else in the label is left at the served value."""
    from engine.router import tree_nodes
    k = {"deep": 0, "wide": tree_nodes(16), "narrow": tree_nodes(8),
         "after": int(os.environ.get("QWEN38_TREE_WIDE_AFTER", "0") or 0),
         "mode": os.environ.get("QWEN38_DF2_TREE_MODE", "paths"),
         "temp": float(os.environ.get("QWEN38_DF2_TEMP", "1.0"))}
    for tok in label.split("+"):
        if tok == "deep":
            k["deep"] = 32
        elif tok in ("nodes", "paths"):
            k["mode"] = tok
        elif len(tok) > 1 and tok[0] in "wn" and tok[1:].isdigit():
            k["wide" if tok[0] == "w" else "narrow"] = int(tok[1:])
        elif len(tok) > 1 and tok[0] == "t" and tok[1:].isdigit():
            k["temp"] = int(tok[1:]) / 10.0
        elif len(tok) > 1 and tok[0] == "a" and tok[1:].isdigit():
            k["after"] = int(tok[1:])
    return k


def _summary(rows, hdr, repeats: int) -> None:
    """The table, and -- when the sweep was repeated -- the paired comparison it was repeated for.

    A repeat is not a second opinion about the mean. It is the only way to know whether a
    difference of a fraction of a per cent between two configurations survives the noise of the
    instrument that measured it, which is the phase-9 trap stated for a bench instead of a row.
    """
    import statistics
    print("\n" + "=" * 96)
    print(f"{'workload':10s} " + " ".join(f"{h:>18s}" for h in hdr) + "   best fixed   router vs")
    means = {h: [] for h in hdr}
    for name in dict.fromkeys(r["workload"] for r in rows):
        cells, by = [], {}
        for h in hdr:
            hits = [x for x in rows if x["workload"] == name and x["config"] == h]
            r = hits[0] if hits else None
            if r is not None:
                r = dict(r, tok_s=statistics.median(x["tok_s"] for x in hits),
                         accept_len=statistics.median(x["accept_len"] for x in hits))
            by[h] = r
            cells.append(f"{r['tok_s']:8.2f}/{r['accept_len']:5.2f}" if r else " " * 14)
            if r:
                means[h].append(r["tok_s"])
        fixed = [by[h]["tok_s"] for h in ("fixed8", "fixed16") if by.get(h)]
        best = max(fixed) if fixed else 0.0
        rel = (by["router"]["tok_s"] / best - 1) * 100 if by.get("router") and best else 0.0
        print(f"{name:10s} " + " ".join(f"{c:>18s}" for c in cells)
              + f"   {best:8.2f}   {rel:+6.2f} %")
    print("-" * 96)
    print(f"{'mean':10s} " + " ".join(
        f"{sum(means[h]) / len(means[h]):>18.2f}" if means[h] else " " * 18 for h in hdr))
    print("  (each cell is the MEDIAN of the repeats, tok/s / accepted per block)"
          if repeats > 1 else "")

    if "router" not in hdr or repeats < 2:
        return
    # The paired part. For every other configuration, compare it with `router` on each workload
    # separately -- same prompts, same process, same allocation history -- and say how many
    # workloads it won, beside the spread of `router` against ITSELF across the repeats. A
    # configuration ahead on every workload is a result even when the difference is small; one
    # ahead by more than the noise on none of them is not.
    print("\n" + "-" * 96)
    print(f"{'vs router':10s} {'mean delta':>12s} {'won':>6s} {'lost':>6s} "
          f"{'worst wl':>10s} {'router self-spread':>20s}")
    self_spread = []
    for name in dict.fromkeys(r["workload"] for r in rows):
        xs = [x["tok_s"] for x in rows if x["workload"] == name and x["config"] == "router"]
        if len(xs) > 1 and statistics.median(xs):
            self_spread.append(100.0 * (max(xs) - min(xs)) / statistics.median(xs))
    spread = max(self_spread) if self_spread else 0.0
    for h in hdr:
        if h == "router":
            continue
        deltas = {}
        for name in dict.fromkeys(r["workload"] for r in rows):
            b = [x["tok_s"] for x in rows if x["workload"] == name and x["config"] == "router"]
            o = [x["tok_s"] for x in rows if x["workload"] == name and x["config"] == h]
            if b and o:
                deltas[name] = 100.0 * (statistics.median(o) / statistics.median(b) - 1)
        if not deltas:
            continue
        won = sum(1 for v in deltas.values() if v > 0)
        lost = sum(1 for v in deltas.values() if v < 0)
        worst = min(deltas, key=deltas.get)
        print(f"{h:10s} {statistics.fmean(deltas.values()):>+11.2f}% {won:>6d} {lost:>6d} "
              f"{worst[:10]:>10s} {spread:>19.2f}%")


if __name__ == "__main__":
    main()
