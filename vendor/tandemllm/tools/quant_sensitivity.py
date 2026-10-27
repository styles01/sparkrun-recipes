"""Which projections are worth four bits: a cost per unit, and the mix that spends bytes best.

A unit is one fusion group of one layer (`quant_nvfp4.UNITS`: the MLP's gate and up, its down
projection, the GDN input projections, the GDN output, the attention q/k/v, the attention output).
`measure` loads the reference checkpoint once, holds every NVFP4 unit beside it, and for each unit
alone swaps it in, runs a text the quantiser never saw and the gate never reads, and records how far
the next-token distribution moved:

    KL(reference || unit in NVFP4), over the reference's 256 most likely tokens at every position

(the reference's top 256 carry nearly all of its mass; storing them costs 12 MB where full logits
would cost 8 GB), with the change in loss and the bytes the unit saves beside it. One pass with
every unit swapped at once checks how far the single-unit costs add up.

`mix` turns the costs into a weight file. For a budget of extra bytes a step, it keeps in FP8 the
units with the largest cost per byte kept, until the budget is spent, and writes every other unit's
NVFP4 tensors into one file the engine loads like any other (`--nvfp4 mix.safetensors`). The
quality gate, not this estimate, decides whether a mix ships.

    python tools/quant_sensitivity.py measure --model <bf16> --nvfp4 mlp.st,gdn.st,attn.st \
        --corpus sens-code.txt:4096,sens-prose.txt:4096 --out sens.json
    python tools/quant_sensitivity.py mix --sens sens.json --nvfp4 mlp.st,gdn.st,attn.st \
        --keep-gb 1.0 --out mix-1gb.safetensors
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.quant_nvfp4 import UNITS, nvfp4_bytes, read_corpus  # noqa: E402

TOPK = 256


def units_of(names) -> dict[str, list[str]]:
    """{`layers.5.mlp.down`: [`layers.5.mlp.down_proj`]} for every unit whose members are all in
    `names`."""
    names = set(names)
    layers = sorted({int(n.split(".")[1]) for n in names})
    out = {}
    for layer in layers:
        for unit, members in UNITS.items():
            full = [f"layers.{layer}.{m}" for m in members]
            if all(f in names for f in full):
                out[f"layers.{layer}.{unit}"] = full
    return out


def load_blocks(spec: str, device: str) -> dict:
    from tools.nvfp4_linear import NVFP4Block
    blocks = {}
    for path in spec.split(","):
        path = os.path.expanduser(path.strip())
        if not path:
            continue
        with safe_open(path, framework="pt", device=device) as f:
            keys = set(f.keys())
            for k in keys:
                if not k.endswith(".weight_scale_2"):
                    continue
                base = k[: -len(".weight_scale_2")]
                blocks[base] = NVFP4Block(f.get_tensor(f"{base}.weight"),
                                          f.get_tensor(f"{base}.weight_scale"),
                                          f.get_tensor(k).float().item())
    return blocks


def fp8_bytes(N: int, K: int) -> int:
    return N * K + -(-N // 128) * -(-K // 128) * 2


def _pass(eng, chunks, ref=None):
    """Teacher-forced over every chunk. Without `ref`: the reference's top-k and NLL. With it: the
    top-k KL against it and the NLL."""
    kl_sum, nll_sum, n = 0.0, 0.0, 0
    tops = []
    for ci, ids in enumerate(chunks):
        eng.reset()
        with torch.no_grad():
            logits = eng.forward(ids.to(eng.device), start=0)[0].float()
        m = ids.numel() - 1
        lp = F.log_softmax(logits[:m], dim=-1)
        tgt = ids[1:].to(eng.device)
        nll_sum += float(-lp.gather(1, tgt[:, None]).sum())
        n += m
        if ref is None:
            v, i = lp.topk(TOPK, dim=-1)
            tops.append((v, i))
        else:
            rv, ri = ref[ci]
            q = lp.gather(1, ri)
            kl_sum += float((rv.exp() * (rv - q)).sum())
        del logits, lp
    return (tops if ref is None else kl_sum / n), nll_sum / n


def cmd_measure(args) -> None:
    from engine.config import load_config
    from engine.loader import Weights
    from engine.model import Qwen38Engine

    cfg = load_config(args.model)

    def tok():
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(cfg.path)
    w = Weights(cfg.path, device=args.device, skip_mtp=True, nvfp4="", fp8_head="")
    eng = Qwen38Engine(cfg, w, max_len=args.chunk + 16, device=args.device)
    q = load_blocks(args.nvfp4, eng.device)
    units = units_of(q)
    if args.layers:
        units = {u: m for u, m in units.items() if int(u.split(".")[1]) < args.layers}
    chunks = []
    for _, ids in read_corpus(args.corpus, tok, 4096):
        chunks += [ids[s:s + args.chunk] for s in range(0, ids.numel(), args.chunk)
                   if ids[s:s + args.chunk].numel() >= 64]
    print(f"[sens] {len(units)} units, {len(chunks)} chunks, "
          f"{sum(c.numel() for c in chunks)} tokens", flush=True)
    t0 = time.time()
    ref, ref_nll = _pass(eng, chunks)
    print(f"[ref] NLL {ref_nll:.4f}  ({time.time() - t0:.1f} s)", flush=True)
    rows = []
    for ui, (unit, members) in enumerate(units.items()):
        saved = {m: w.q[m] for m in members}
        for m in members:
            w.q[m] = q[m]
        kl, nll = _pass(eng, chunks, ref)
        for m in members:
            w.q[m] = saved[m]
        b8 = sum(fp8_bytes(*q[m].shape) for m in members)
        b4 = sum(nvfp4_bytes(*q[m].shape) for m in members)
        rows.append({"unit": unit, "layer": int(unit.split(".")[1]), "kind": unit.split(".", 2)[2],
                     "kl": kl, "dnll": nll - ref_nll, "bytes_fp8": b8, "bytes_nvfp4": b4})
        if ui % 16 == 0 or ui == len(units) - 1:
            print(f"  {ui + 1:3d}/{len(units)} {unit:28s} KL {kl:.6f}  dNLL {nll - ref_nll:+.5f}  "
                  f"({time.time() - t0:.0f} s)", flush=True)
    # every unit at once: how far the single-unit costs add up
    saved = {m: w.q[m] for ms in units.values() for m in ms}
    for m in saved:
        w.q[m] = q[m]
    kl_all, nll_all = _pass(eng, chunks, ref)
    for m, b in saved.items():
        w.q[m] = b
    out = {"model": cfg.path, "nvfp4": args.nvfp4, "corpus": args.corpus, "topk": TOPK,
           "tokens": sum(c.numel() for c in chunks), "ref_nll": ref_nll,
           "all": {"kl": kl_all, "dnll": nll_all - ref_nll,
                   "sum_unit_kl": sum(r["kl"] for r in rows)},
           "units": rows}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"[all] KL {kl_all:.6f} (sum of units {out['all']['sum_unit_kl']:.6f})  "
          f"dNLL {nll_all - ref_nll:+.5f}  -> {args.out}")
    by_kind: dict[str, list[float]] = {}
    for r in rows:
        by_kind.setdefault(r["kind"], []).append(r["kl"])
    for k, v in sorted(by_kind.items()):
        print(f"  {k:12s} {len(v):3d} units  KL sum {sum(v):.6f}  max {max(v):.6f}")


def choose_keep(rows: list[dict], keep_bytes: float) -> list[dict]:
    """Greedy knapsack: the units with the most KL per extra byte stay in FP8 until the budget is
    spent. Greedy is within one unit of optimal here, and a unit is at most 0.14 GB."""
    ranked = sorted(rows, key=lambda r: r["kl"] / (r["bytes_fp8"] - r["bytes_nvfp4"]), reverse=True)
    keep, spent = [], 0
    for r in ranked:
        extra = r["bytes_fp8"] - r["bytes_nvfp4"]
        if spent + extra > keep_bytes:
            continue
        keep.append(r)
        spent += extra
    return keep


def cmd_mix(args) -> None:
    sens = json.load(open(args.sens))
    rows = sens["units"]
    keep = choose_keep(rows, args.keep_gb * 1e9)
    keep_units = {r["unit"] for r in keep}
    members_kept = set()
    for u in keep_units:
        layer, kind = u.split(".")[1], u.split(".", 2)[2]
        members_kept |= {f"layers.{layer}.{m}" for m in UNITS[kind]}
    tensors = {}
    for path in args.nvfp4.split(","):
        with safe_open(os.path.expanduser(path.strip()), framework="pt", device="cpu") as f:
            meta = f.metadata() or {}
            for k in f.keys():
                base = k.rsplit(".", 1)[0]
                if base.endswith(".weight"):
                    base = base[: -len(".weight")]
                if base in members_kept:
                    continue
                tensors[k] = f.get_tensor(k)
    extra = sum(r["bytes_fp8"] - r["bytes_nvfp4"] for r in keep)
    if not tensors:
        print(f"[mix] keep {len(keep)} units: that is every unit, nothing left in NVFP4; no file")
        return
    kl_full = sum(r["kl"] for r in rows)
    kl_mix = kl_full - sum(r["kl"] for r in keep)
    kinds: dict[str, int] = {}
    for r in keep:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    save_file(tensors, args.out, metadata={
        "format": "nvfp4", "group": "16", "mode": meta.get("mode", "?"), "targets": "mix",
        "source": meta.get("source", "?"), "kept_fp8": ",".join(sorted(keep_units)),
        "keep_gb": f"{args.keep_gb:g}"})
    summary = {"keep_gb": args.keep_gb, "extra_bytes": extra, "kept_units": sorted(keep_units),
               "kept_by_kind": kinds, "est_kl_full": kl_full, "est_kl_mix": kl_mix,
               "projections_nvfp4": len(tensors) // 3, "file": args.out}
    with open(os.path.splitext(args.out)[0] + ".json", "w") as f:
        json.dump(summary, f, indent=1)
    print(f"[mix] keep {len(keep)} units in FP8 ({extra / 1e9:.2f} GB a step): {kinds}; "
          f"estimated KL {kl_full:.5f} -> {kl_mix:.5f}; {len(tensors) // 3} NVFP4 projections "
          f"-> {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("measure")
    m.add_argument("--model", default=None)
    m.add_argument("--nvfp4", required=True, help="the NVFP4 files whose units are measured")
    m.add_argument("--corpus", required=True, help="file[:tokens][,...] the gate never reads")
    m.add_argument("--chunk", type=int, default=2048)
    m.add_argument("--layers", type=int, default=0)
    m.add_argument("--device", default="cuda")
    m.add_argument("--out", required=True)
    m.set_defaults(fn=cmd_measure)
    x = sub.add_parser("mix")
    x.add_argument("--sens", required=True)
    x.add_argument("--nvfp4", required=True)
    x.add_argument("--keep-gb", type=float, required=True,
                   help="extra bytes a step spent on keeping units in FP8")
    x.add_argument("--out", required=True)
    x.set_defaults(fn=cmd_mix)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
