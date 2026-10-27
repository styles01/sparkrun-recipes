"""Per-second bandwidth of a decode-shaped GEMV, logged long enough to see a state change.

There is a documented pathology on this board where a bf16 GEMV alternates between roughly a third
of the board's bandwidth and nearly all of it, on a period of seconds, with the reported clocks and
throttle reasons unchanged throughout. A run that samples both states averages to the middle, and
the middle is close to what this engine's step reports, so every number in the ledger is suspect
until it is known which state the board was in.

This logs one measurement per second for as long as it is asked to, on the same shape and the same
kernel the engine's `lm_head` uses, and prints the distribution rather than the mean. What it is
looking for is bimodality, not a value.

    python tools/bw_probe.py --seconds 600
    taskset -c 0-4,10-14 python tools/bw_probe.py --seconds 600     # host off the performance cores
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seconds", type=int, default=300)
    ap.add_argument("--n", type=int, default=248320)
    ap.add_argument("--k", type=int, default=5120)
    ap.add_argument("--batch", type=int, default=30, help="calls per sample")
    args = ap.parse_args()
    dev = "cuda"
    w = torch.randn(args.n, args.k, dtype=torch.bfloat16, device=dev)
    x = torch.randn(1, args.k, dtype=torch.bfloat16, device=dev)
    nbytes = w.numel() * 2
    for _ in range(5):
        torch.nn.functional.linear(x, w)
    torch.cuda.synchronize()
    samples = []
    t_end = time.time() + args.seconds
    print(f"[probe] bf16 GEMV N={args.n} K={args.k}, {nbytes / 1e9:.2f} GB per call", flush=True)
    # CUDA is asynchronous, so a host-side "run for 0.9 s then synchronise" loop queues far more
    # work than 0.9 s of it and averages over the whole queue -- which is exactly long enough to
    # average a flip away. A fixed, small batch with a synchronise after it samples a window of
    # about a tenth of the flip's shortest reported period.
    batch = args.batch
    while time.time() < t_end:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(batch):
            torch.nn.functional.linear(x, w)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        gbs = nbytes * batch / dt * 1e-9
        time.sleep(max(0.0, 0.5 - dt))
        samples.append(gbs)
        print(f"  {time.strftime('%H:%M:%S')}  {gbs:7.1f} GB/s   ({batch} calls)", flush=True)
    s = sorted(samples)
    lo = [v for v in samples if v < (min(samples) + max(samples)) / 2]
    hi = [v for v in samples if v >= (min(samples) + max(samples)) / 2]
    print(f"\n[summary] n={len(samples)}  min {s[0]:.1f}  p10 {s[len(s) // 10]:.1f}  "
          f"median {statistics.median(s):.1f}  p90 {s[-max(1, len(s) // 10)]:.1f}  max {s[-1]:.1f}")
    print(f"[summary] below midpoint: {len(lo)} samples, mean {statistics.mean(lo):.1f} GB/s"
          if lo else "[summary] below midpoint: none")
    print(f"[summary] above midpoint: {len(hi)} samples, mean {statistics.mean(hi):.1f} GB/s"
          if hi else "[summary] above midpoint: none")
    spread = (max(samples) - min(samples)) / statistics.median(samples)
    print(f"[summary] spread {spread * 100:.1f} % of the median -- "
          f"{'BIMODAL, the flip is present' if spread > 0.5 else 'single state'}")


if __name__ == "__main__":
    main()
