"""Quantise `lm_head` to e4m3 with one scale per vocabulary row, and gate the result.

`lm_head` is 248,320 x 5,120 in bf16 -- 2.54 GB, 13 % of an NVFP4 step's read set. It is read once
by the verify pass and once more by the block drafter, which takes `topk` over the full vocabulary,
so at a block of eight it is 10.8 + 10.8 ms of a 161 ms block. At e4m3 it is 1.27 GB.

Unlike the reduced-vocabulary draft head this **changes what the engine emits**: the verify pass's
own distribution comes out of this tensor, so it takes the full quality gate -- held-out NLL on
prose and code against the bf16 head on identical tokens, and argmax agreement restricted to the
positions where the bf16 head is confident.

    python tools/quant_head.py build --out ~/nvfp4/head-fp8.safetensors
    python tools/quant_head.py gate  --head ~/nvfp4/head-fp8.safetensors --tokens 2048
    python tools/quant_head.py bench --head ~/nvfp4/head-fp8.safetensors
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.config import resolve_snapshot  # noqa: E402
from tools.head_gemv import FP8Head, head_matmul_fp8, quantize_head_fp8  # noqa: E402


def read_head(snapshot: str, device: str) -> torch.Tensor:
    for name in sorted(os.listdir(snapshot)):
        if not name.endswith(".safetensors"):
            continue
        with safe_open(os.path.join(snapshot, name), framework="pt", device=device) as f:
            for key in f.keys():
                if key.endswith("lm_head.weight"):
                    return f.get_tensor(key)
    raise SystemExit("lm_head.weight not found in the snapshot")


def load_head_fp8(path: str, device: str = "cuda") -> FP8Head:
    with safe_open(os.path.expanduser(path), framework="pt", device=device) as f:
        return FP8Head(f.get_tensor("lm_head.weight"), f.get_tensor("lm_head.weight_scale"))


def cmd_build(args) -> None:
    snapshot = resolve_snapshot(args.model)
    w = read_head(snapshot, args.device)
    print(f"[head] {tuple(w.shape)} {w.dtype}  {w.numel() * w.element_size() / 1e9:.2f} GB")
    ratios = tuple(float(x) for x in args.ratios.split(","))
    t0 = time.time()
    head = quantize_head_fp8(w, ratios=ratios)
    print(f"[quant] ratios {ratios}  {time.time() - t0:.1f} s  "
          f"{head.nbytes / 1e9:.3f} GB  ({head.nbytes / (w.numel() * 2):.3f}x)")
    deq = head.dequant()
    d = (deq.float() - w.float()).abs()
    rel = float(d.max() / w.float().abs().max())
    print(f"[error] max |dw| {float(d.max()):.5f}  mean |dw| {float(d.mean()):.6f}  "
          f"relative to the largest weight {rel:.5f}")
    del deq, d
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    save_file({"lm_head.weight": head.w.cpu(), "lm_head.weight_scale": head.s.cpu()},
              args.out, metadata={"format": "fp8-e4m3", "scale": "per-row",
                                  "ratios": args.ratios, "source": snapshot})
    print(f"[out] {args.out}")


def cmd_bench(args) -> None:
    """The kernel against the library on the bf16 head, at the two row counts that occur."""
    head = load_head_fp8(args.head, args.device)
    ref = head.dequant()
    out = []
    for M in (1, 7, 8, 16):
        x = torch.randn(M, head.K, device=args.device, dtype=torch.bfloat16) * 0.05
        a = torch.nn.functional.linear(x, ref).float()
        b = head_matmul_fp8(x, head)
        agree = int((a.argmax(-1) == b.argmax(-1)).sum())

        def timed(fn, n=30):
            for _ in range(5):
                fn()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(n):
                fn()
            torch.cuda.synchronize()
            return (time.perf_counter() - t0) / n * 1e3

        ms_ref = timed(lambda: torch.nn.functional.linear(x, ref))
        ms_fp8 = timed(lambda: head_matmul_fp8(x, head))
        bref = ref.numel() * 2
        bfp8 = head.nbytes
        out.append(f"M={M:2d}  bf16 F.linear {ms_ref:6.2f} ms {bref / ms_ref / 1e6:6.1f} GB/s   "
                   f"fp8 kernel {ms_fp8:6.2f} ms {bfp8 / ms_fp8 / 1e6:6.1f} GB/s   "
                   f"argmax same {agree}/{M}  max|dlogit| {float((a - b).abs().max()):.4f}")
    for line in out:
        print(line)


def cmd_gate(args) -> None:
    """Held-out NLL and argmax agreement, bf16 head against fp8 head, same tokens, one process."""
    import torch.nn.functional as F
    from transformers import AutoTokenizer
    from engine.config import load_config
    from engine.loader import Weights
    from engine.model import Qwen38Engine

    HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    corpora = {"prose": os.path.join(HERE, "bench", "heldout_prose.txt"),
               "code": os.path.join(HERE, "bench", "heldout_code.txt")}

    cfg = load_config(args.model)
    tok = AutoTokenizer.from_pretrained(cfg.path)
    w = Weights(cfg.path, skip_mtp=True, nvfp4=args.nvfp4)
    eng = Qwen38Engine(cfg, w, max_len=max(args.chunk, 2048) + 64)
    bf16_head = w.t["lm_head.weight"]
    fp8_head = load_head_fp8(args.head, eng.device)

    ids = {n: tok(open(p).read(), return_tensors="pt").input_ids[0][: args.tokens]
           for n, p in corpora.items()}
    res: dict[str, dict] = {}
    for tag, head in (("bf16", bf16_head), ("fp8", fp8_head)):
        w.t["lm_head.weight"] = head
        res[tag] = {}
        for name, seq in ids.items():
            nll, n, argmax, gap = 0.0, 0, [], []
            eng.reset()
            start = 0
            for c0 in range(0, seq.numel() - 1, args.chunk):
                piece = seq[c0:c0 + args.chunk]
                logits = eng.forward(piece.to(eng.device), start=start)[0].float()
                start += piece.numel()
                tgt = seq[c0 + 1:c0 + 1 + piece.numel()].to(eng.device)
                m = tgt.numel()
                lp = F.log_softmax(logits[:m], dim=-1)
                nll += float(-lp.gather(1, tgt[:, None]).sum())
                n += m
                top2 = logits[:m].topk(2, dim=-1)
                argmax.append(top2.indices[:, 0].cpu())
                gap.append((top2.values[:, 0] - top2.values[:, 1]).cpu())
                del logits, lp, top2
            res[tag][name] = {"nll": nll / n, "n": n, "argmax": torch.cat(argmax),
                              "gap": torch.cat(gap)}
            print(f"  {tag:4s} {name:5s} NLL {nll / n:.4f} over {n} tokens")
    w.t["lm_head.weight"] = bf16_head

    print("\n--- delta, fp8 head against the bf16 head, same tokens")
    worst = -1e9
    for name in ids:
        a, b = res["bf16"][name], res["fp8"][name]
        agree = (a["argmax"] == b["argmax"]).float()
        conf = a["gap"] >= 1.0
        d = b["nll"] - a["nll"]
        worst = max(worst, d)
        print(f"  {name:5s} NLL {a['nll']:.4f} -> {b['nll']:.4f}  delta {d:+.4f} nats   "
              f"argmax {float(agree.mean()):.4f} "
              f"(confident positions {float(agree[conf].mean()):.4f}, n={int(conf.sum())})")
    print(f"  worst delta {worst:+.4f} nats   gate <= +0.05   "
          f"{'PASS' if worst <= 0.05 else 'FAIL'}")


def cmd_same(args) -> None:
    """The head `--fp8-head build` makes at load, against a head file, byte for byte."""
    from engine.loader import Weights, parse_head_build
    snapshot = resolve_snapshot(args.model)
    ratios = parse_head_build(args.spec)
    if ratios is None:
        raise SystemExit(f"--spec {args.spec!r} is not a build spec")
    w = Weights.__new__(Weights)          # the head path only; the rest of the checkpoint stays on disk
    w.device = args.device
    w.t = {"lm_head.weight": read_head(snapshot, args.device)}
    w.bytes_other = 0
    w.fp8_head_source = None
    t0 = time.time()
    w.build_fp8_head(ratios)
    built = w.t["lm_head.weight"]
    ms = (time.time() - t0) * 1e3
    ref = load_head_fp8(args.head, args.device)
    same_codes = built.w.view(torch.uint8) == ref.w.view(torch.uint8)
    rows_codes = int((~same_codes).any(dim=1).sum())
    rows_scale = int((built.s != ref.s).sum())
    ok = rows_codes == 0 and rows_scale == 0
    print(f"[same] built at load ({ms:.0f} ms, ratios {ratios}) vs {args.head}: "
          f"rows with other codes {rows_codes}, rows with another scale {rows_scale} of {ref.N}  "
          f"{'IDENTICAL' if ok else 'DIFFERENT'}")
    raise SystemExit(0 if ok else 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build")
    b.add_argument("--model", default=None)
    b.add_argument("--out", required=True)
    b.add_argument("--device", default="cuda")
    b.add_argument("--ratios", default="1.0", help="scale multipliers to search, comma separated")
    b.set_defaults(fn=cmd_build)

    n = sub.add_parser("bench")
    n.add_argument("--head", required=True)
    n.add_argument("--device", default="cuda")
    n.set_defaults(fn=cmd_bench)

    g = sub.add_parser("gate")
    g.add_argument("--model", default=None)
    g.add_argument("--head", required=True)
    g.add_argument("--nvfp4", default=None)
    g.add_argument("--tokens", type=int, default=2048)
    g.add_argument("--chunk", type=int, default=1024)
    g.set_defaults(fn=cmd_gate)

    s = sub.add_parser("same", help="the head built at load (--fp8-head build) == a head file?")
    s.add_argument("--model", default=None)
    s.add_argument("--head", required=True)
    s.add_argument("--spec", default="build")
    s.add_argument("--device", default="cuda")
    s.set_defaults(fn=cmd_same)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
