"""The E1 table: what each layer-drop configuration costs and what it would buy.

Reads what `tools/e1_layer_drop.py` wrote and prints one row per configuration, so the verdict is
produced by arithmetic over recorded JSON rather than by hand.

The tok/s column is the ledger's cost model, `tau / (M + s.N + d)`, with two changes made honestly:

  * `M` is the **measured** step time of that configuration, not the byte model's prediction. The
    byte model is printed next to it so the two can be compared, which is the point of measuring.
  * `tau` is the baseline's calibrated 3.14 scaled by the RATIO of the distribution-matching
    ceilings, `E[acc at p1] + 1`, between the pruned configuration and L = 0. The absolute ceiling
    is an upper bound no drafter on this board reaches; its ratio is the part that transfers.

    python tools/e1_report.py --out results/e1
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SLOPE_MS = 1.896
DRAFT_MS = 17.0
NODES = 16
TAU0 = 3.14
BANDWIDTH_GB_S = 163.0
KILL_P1 = 0.75
KILL_REP = 0.10


def ms(entry: dict, key: int) -> float:
    m = entry.get("ms", {})
    return float(m.get(str(key), m.get(key, float("nan"))))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results/e1")
    ap.add_argument("--group", default="prose")
    ap.add_argument("--samples", type=int, default=0,
                    help="decode this many free generations per configuration, for reading")
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    ev = json.load(open(os.path.join(args.out, "eval.json")))
    tm = json.load(open(os.path.join(args.out, "time.json")))
    gpath = os.path.join(args.out, "gen.json")
    gn = json.load(open(gpath)) if os.path.exists(gpath) else {}
    order = list(ev.keys())

    base = ev["L0"][args.group]
    tau_ref = base["acc16"] + 1.0

    print(f"E1 -- layer drop, group '{args.group}', {base['sequences']} held-out sequences, "
          f"{base['positions']} positions\n")
    head = (f"{'config':<9}{'L':>3}{'GB':>7}{'step ms':>9}{'model ms':>9}{'v(8) ms':>9}"
            f"{'p1':>7}{'p1(t)':>7}{'argmax':>8}{'acc16':>7}{'tau':>6}{'rep8':>7}"
            f"{'gen tok':>8}{'tok/s':>7}")
    print(head)
    rows = []
    for name in order:
        e = ev[name]
        g = e.get(args.group)
        if g is None:
            continue
        t = tm.get(name, {})
        step = ms(t, 1)
        v8 = ms(t, 8)
        tau = TAU0 * (g["acc16"] + 1.0) / tau_ref
        toks = tau / ((step + SLOPE_MS * NODES + DRAFT_MS) / 1000.0)
        entry = gn.get(name, {})
        rep = entry.get("rep8", float("nan"))
        # A generation that stopped after six tokens cannot repeat itself, so the length is part of
        # the collapse reading and not a footnote to it.
        lens = [s["tokens"] for s in entry.get("samples", [])]
        mean_len = sum(lens) / len(lens) if lens else float("nan")
        model_ms = e["bytes_gb"] / BANDWIDTH_GB_S * 1000.0
        print(f"{name:<9}{e['drops']:>3}{e['bytes_gb']:>7.2f}{step:>9.2f}{model_ms:>9.2f}"
              f"{v8:>9.2f}{g['mean_p1']:>7.3f}{g['mean_p_teacher']:>7.3f}"
              f"{g['argmax_agreement']:>8.3f}{g['acc16']:>7.2f}{tau:>6.2f}"
              f"{rep * 100:>6.1f}%{mean_len:>8.0f}{toks:>7.1f}")
        rows.append((name, e, g, tau, toks, rep))

    print(f"\nkill rule (RESEARCH-PRUNE-DISTILL-0917 section 6): p1 < {KILL_P1} or "
          f"repetition > {KILL_REP * 100:.0f} % at L = 8")
    for name, e, g, tau, toks, rep in rows:
        if e["drops"] != 8:
            continue
        verdict = "CLOSED" if (g["mean_p1"] < KILL_P1 or (rep == rep and rep > KILL_REP)) else "open"
        print(f"  {name:<9} p1 {g['mean_p1']:.3f}  rep8 {rep * 100:.1f}%  -> branch {verdict}")

    if args.samples:
        from transformers import AutoTokenizer  # noqa: PLC0415
        from engine.config import load_config  # noqa: PLC0415
        tok = AutoTokenizer.from_pretrained(load_config(args.model).path)
        print("\n--- free generations, read them ---")
        for name in order:
            for s in gn.get(name, {}).get("samples", [])[: args.samples]:
                text = tok.decode(s["ids"], skip_special_tokens=True)
                print(f"\n[{name} / {s['topic']} / {s['tokens']} tok / rep8 "
                      f"{s['rep8'] * 100:.1f}%]\n{text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
