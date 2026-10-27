"""The same kernel on a COLD working set, which is the only way this engine ever reads a weight.

`tools/nvfp4_wide_probe.py` times one projection in a loop. One projection is 17 to 50 MB and this
board has a system-level cache in that neighbourhood, so the loop measures a kernel reading a
weight that is already resident -- and it reported 216-266 GB/s, which is 79-98 % of a 273 GB/s
board and should have been the giveaway. The verify pass reads 15 GB once; nothing is resident.

This walks a chain of DISTINCT weights whose total exceeds any cache, so every byte is a cold DRAM
read, and it reports the rate at each M and for each candidate tile. It answers two questions the
warm probe cannot: what this board actually gives a W4A16 kernel on cold bytes, and whether the M
dependence the warm probe found survives.

2026-09-23, the parallelism gate. A batched verify of N sequences at width 16 is 17 N rows
through the same weights, and the parallel-requests plan (docs/roadmap.md) stands or falls on whether the
rate per byte holds from 17 rows to 272. Three additions answer that without touching the engine:

  --impl shipped   whatever `nvfp4_matmul` does at that M today (v2 to 32 rows, v1's mid/prefill
                   tiles to 511, unpack + library GEMM from 512), so the curve has a "today" line
  --rowcheck S     every row of the M-row product against the same rows computed S at a time on
                   the path a single sequence takes. A batch is lossless only if a sequence's rows
                   come out of it bit for bit what they would have been alone; this is that
                   invariant at the kernel, before any engine code exists to break it
  --head           the e4m3 vocabulary projection instead of an NVFP4 shape: one 1.27 GB read a
                   verify whatever M is, and an fp32 [M, 248320] write that is NOT constant in M
"""
import argparse, json, os, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.nvfp4_linear import nvfp4_matmul, pick_config  # noqa: E402
from tools.nvfp4_linear_v2 import nvfp4_matmul_v2  # noqa: E402
from tools.quant_nvfp4 import quantize_clipped  # noqa: E402
from tools.head_gemv import FP8Head, head_matmul_fp8  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--N", type=int, default=17408)
ap.add_argument("--K", type=int, default=5120)
ap.add_argument("--gb", type=float, default=3.0, help="working set, GB")
ap.add_argument("--rows", default="1,2,4,8,14,16")
ap.add_argument("--reps", type=int, default=6)
ap.add_argument("--impl", default="v1", choices=["v1", "v2", "shipped"],
                help="which kernel the candidates below are run on; `shipped` is the engine's "
                     "own dispatch at that M and takes no tiles")
ap.add_argument("--tiles", default="",
                help="explicit candidates, 'n32:k1:w4:s3[:d2][:p1],...'; empty = the shipped "
                     "decode tile plus whatever pick_config returns")
ap.add_argument("--rowcheck", type=int, default=0,
                help="compare every row against the same rows computed this many at a time on "
                     "the shipped path (17 = one sequence's verify at width 16). 0 = off")
ap.add_argument("--head", action="store_true",
                help="time the e4m3 vocabulary projection (N=248320, K=5120) instead")
ap.add_argument("--json", default="", help="append one record per (M, tile) to this JSONL file")
a = ap.parse_args()


def parse_tiles(spec):
    """'n32:k1:w4:s3:d2:p1' -> a kwargs dict for the matmul under test."""
    out = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        cfg = {"block_m": 16, "num_stages": 3}
        for f in item.split(":"):
            key = {"m": "block_m", "n": "block_n", "k": "split_k", "w": "num_warps",
                   "s": "num_stages", "d": "dots", "p": "prefetch"}[f[0]]
            cfg[key] = int(f[1:])
        out.append((item, cfg))
    return out


def make_weight():
    if a.head:
        # A random e4m3 head with the served one's shape and a per-row scale; the codes' values do
        # not change what a read costs, and a real head is one tensor, not a chain.
        codes = (torch.randn(a.N, a.K, device="cuda") * 0.5).to(torch.float8_e4m3fn)
        return FP8Head(codes, torch.full((a.N,), 0.01, device="cuda", dtype=torch.float32))
    return quantize_clipped(torch.randn(a.N, a.K, device="cuda", dtype=torch.float32) * 0.02, None)


def rowcheck(x, w, cfg, solo):
    """Max |d| and the count of elements that differ between the M-row product and the same rows
    produced `solo` at a time on the shipped path."""
    y = run(x, w, cfg).float()
    ref = torch.cat([run(x[i:i + solo], w, None).float() for i in range(0, x.shape[0], solo)])
    d = (y - ref).abs()
    return float(d.max()), int((d > 0).sum()), d.numel()


def run(x, w, cfg):
    if a.head:
        return head_matmul_fp8(x, w)
    if cfg is None or a.impl == "shipped":
        return nvfp4_matmul(x, w)
    return mm(x, w, **cfg)


if a.head:
    a.N = 248320 if a.N == 17408 else a.N
one = make_weight()
nb = one.nbytes
n = max(2, int(a.gb * 1e9 / nb))
ws = [one] + [make_weight() for _ in range(n - 1)]
total = n * nb
print(f"N={a.N} K={a.K}  {nb / 1e6:.1f} MB each x {n} = {total / 1e9:.2f} GB working set")
mm = nvfp4_matmul if a.impl == "v1" else nvfp4_matmul_v2
kind = "head-fp8" if a.head else a.impl
print(f"impl = {kind}")
print(f"{'M':>4} {'tile':>28} {'ms/chain':>10} {'GB/s':>8} {'TFLOP/s':>8}"
      + (f" {'rowcheck vs ' + str(a.rowcheck) + ' at a time':>34}" if a.rowcheck else ""))

DECODE = {"block_m": 16, "block_n": 16, "split_k": 8, "num_warps": 1, "num_stages": 3}
for M in [int(x) for x in a.rows.split(",")]:
    x = torch.randn(M, a.K, device="cuda", dtype=torch.bfloat16)
    if a.head or a.impl == "shipped":
        cands = [(kind, None)]
    elif a.tiles:
        cands = parse_tiles(a.tiles)
        for _, c in cands:
            c["block_m"] = 16 if M <= 16 else max(32, c["block_m"])
    else:
        cands = [("decode m16 n16 k8 w1", DECODE)]
        blk = pick_config(a.N, a.K, M)
        if blk != DECODE:
            cands.append((f"block m{blk['block_m']} n{blk['block_n']} "
                          f"k{blk['split_k']} w{blk['num_warps']}", blk))
    for name, cfg in cands:
        try:
            for w in ws:                               # warm the instruction cache, not the data
                run(x, w, cfg)
        except Exception as e:                         # a tile that does not fit shared memory
            print(f"{M:4d} {name:>28} {'n/a':>10}  {type(e).__name__}", flush=True)
            continue
        best = 1e9
        for _ in range(a.reps):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for w in ws:
                run(x, w, cfg)
            torch.cuda.synchronize()
            best = min(best, time.perf_counter() - t0)
        tflops = 2.0 * M * a.N * a.K * n / best / 1e12
        line = f"{M:4d} {name:>28} {best * 1e3:10.2f} {total / best / 1e9:8.1f} {tflops:8.2f}"
        rec = {"N": a.N, "K": a.K, "M": M, "impl": kind, "tile": name, "ms_chain": best * 1e3,
               "gbps": total / best / 1e9, "tflops": tflops, "ws_gb": total / 1e9, "n": n,
               "mb_each": nb / 1e6}
        if a.rowcheck:
            dmax, ndiff, numel = rowcheck(x, one, cfg, a.rowcheck)
            line += f"   max|d| {dmax:.3e}  differ {ndiff}/{numel}"
            rec.update(rowcheck=a.rowcheck, max_abs_diff=dmax, n_diff=ndiff, numel=numel)
        print(line, flush=True)
        if a.json:
            with open(a.json, "a") as f:
                f.write(json.dumps(rec) + "\n")
