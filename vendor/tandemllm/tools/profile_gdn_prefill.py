"""Where the chunked delta rule's prefill milliseconds actually go, stage by stage.

`tools/profile_prefill.py` says the GDN mixer is 128 ms of an 8k pass and 55 % of the whole thing.
It does not say which part of the mixer, and the last two attempts to speed it up were aimed at the
wrong part on exactly that evidence: the blocking was swept (64 is best, 128 is a tie, 256 is
worse) and the arithmetic precision was swept (`tf32` takes 15 % off the mixer and makes the pass
slower), and both were negative. The arithmetic is 11.6 MFLOP a head a chunk -- 71 GFLOP a layer at
8k, which is 2.3 ms on this board's fp32 units. The other 126 ms are not arithmetic.

So this tool times the stages instead, with CUDA events, on the real shapes of one GDN layer
(B = 1, H = 48 value heads, Dk = Dv = 128, chunk 64) and synthetic tensors -- no weights, no
loader, seconds to run -- and prints beside each stage the fp32 bytes it moves and what those
bytes cost at the board's peak. A stage far above its byte floor is launch-bound; a stage at its
byte floor is doing real traffic and the only way to make it cheaper is to move fewer bytes.

The stages are the function's own, in order, and the loop is timed both as a whole and per
iteration group, because the loop is the only serial part: 128 iterations at 8k, about a dozen
kernels each, 48 layers deep.

    python tools/profile_gdn_prefill.py --lens 256,2048,8192
    python tools/profile_gdn_prefill.py --lens 8192 --chunk 64,128
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import gdn  # noqa: E402

PEAK_GBPS = 273.0


class Stage:
    """One named stage, timed with CUDA events and carrying the bytes it claims to move."""

    def __init__(self, log: list, name: str, gb: float):
        self.log, self.name, self.gb = log, name, gb

    def __enter__(self):
        self.a = torch.cuda.Event(enable_timing=True)
        self.b = torch.cuda.Event(enable_timing=True)
        self.a.record()
        return self

    def __exit__(self, *exc):
        self.b.record()
        self.log.append((self.name, self.a, self.b, self.gb))
        return False


def staged(query, key, value, g, beta, state, chunk_size, log, launches):
    """`engine.gdn.chunk_gated_delta_rule` with the same operations in the same order, split into
    timed stages. Any divergence from the shipped function is a bug in this tool; `--check` runs
    both and compares."""
    fp32 = 4

    def gb(*shape_counts):
        return sum(n * fp32 for n in shape_counts) / 1e9

    dtype = query.dtype
    B, T, H, Dk = query.shape
    Dv = value.shape[-1]
    nq = B * H * T * Dk
    nv = B * H * T * Dv

    with Stage(log, "l2norm q,k", gb(2 * (2 * nq))):
        query = gdn.l2norm(query, dim=-1, eps=1e-6)
        key = gdn.l2norm(key, dim=-1, eps=1e-6)
    launches["l2norm q,k"] = 2 * 5

    with Stage(log, "transpose+fp32 cast", gb(2 * (2 * nq + nv), 2 * 2 * B * H * T)):
        query, key, value, beta, g = [x.transpose(1, 2).contiguous().to(torch.float32)
                                      for x in (query, key, value, beta, g)]
    launches["transpose+fp32 cast"] = 5

    pad = (chunk_size - T % chunk_size) % chunk_size
    with Stage(log, "pad+scale+k_beta,v_beta", gb(2 * nq, 2 * nv, 2 * nq, 2 * nv)):
        query = F.pad(query, (0, 0, 0, pad))
        key = F.pad(key, (0, 0, 0, pad))
        value = F.pad(value, (0, 0, 0, pad))
        beta = F.pad(beta, (0, pad))
        g = F.pad(g, (0, pad))
        Tp = T + pad
        query = query * (Dk ** -0.5)
        v_beta = value * beta.unsqueeze(-1)
        k_beta = key * beta.unsqueeze(-1)
        query, key, value, k_beta, v_beta = [x.reshape(B, H, -1, chunk_size, x.shape[-1])
                                             for x in (query, key, value, k_beta, v_beta)]
        g = g.reshape(B, H, -1, chunk_size)
    launches["pad+scale+k_beta,v_beta"] = 9

    nc = Tp // chunk_size
    ncc = B * H * nc * chunk_size * chunk_size          # one [B,H,nc,C,C] tensor
    with Stage(log, "cumsum+decay_mask", gb(5 * ncc)):
        mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool,
                                     device=query.device), 0)
        g = g.cumsum(dim=-1)
        decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    launches["cumsum+decay_mask"] = 6

    with Stage(log, "attn = k_beta@k^T", gb(2 * nq, ncc, 3 * ncc)):
        attn = -(torch.matmul(k_beta, key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    launches["attn = k_beta@k^T"] = 3

    eye = torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    with Stage(log, "solve_triangular (UT)", gb(3 * ncc)):
        attn = torch.linalg.solve_triangular(eye - attn, eye.expand_as(attn), upper=False,
                                             unitriangular=True, left=True)
    launches["solve_triangular (UT)"] = 2

    with Stage(log, "value = attn@v_beta", gb(ncc, 2 * nv)):
        value = torch.matmul(attn, v_beta)
    launches["value = attn@v_beta"] = 1

    with Stage(log, "k_cumdecay = attn@(k_beta e^g)", gb(ncc, 4 * nq)):
        k_cumdecay = torch.matmul(attn, k_beta * g.exp().unsqueeze(-1))
    launches["k_cumdecay = attn@(k_beta e^g)"] = 3

    S = (torch.zeros(B, H, Dk, Dv, dtype=torch.float32, device=value.device)
         if state is None else state.clone().float())
    out = torch.zeros_like(value)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), 1)
    # Per iteration: two [C,Dk] reads, one [C,C] write and re-read, three [C,Dv], the state twice.
    per_it = ((2 * chunk_size * Dk + 4 * chunk_size * chunk_size + 8 * chunk_size * Dv
               + 4 * Dk * Dv) * B * H)
    with Stage(log, f"the serial chunk loop ({nc} iterations)", gb(nc * per_it)):
        for i in range(nc):
            q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
            a = (torch.matmul(q_i, k_i.transpose(-1, -2)) * decay_mask[:, :, i]).masked_fill_(mask, 0)
            v_prime = torch.matmul(k_cumdecay[:, :, i], S)
            v_new = v_i - v_prime
            inter = torch.matmul(q_i * g[:, :, i, :, None].exp(), S)
            out[:, :, i] = inter + torch.matmul(a, v_new)
            S = (S * g[:, :, i, -1, None, None].exp()
                 + torch.matmul((k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None])
                                .transpose(-1, -2), v_new))
    launches[f"the serial chunk loop ({nc} iterations)"] = nc * 16

    with Stage(log, "reshape+bf16 cast out", gb(nv, nv // 2)):
        out = out.reshape(B, H, -1, Dv)[:, :, :T].transpose(1, 2).contiguous().to(dtype)
    launches["reshape+bf16 cast out"] = 2
    return out, S


def run(T: int, chunk: int, reps: int, device: str, seed: int = 0):
    torch.manual_seed(seed)
    H, Dk, Dv = 48, 128, 128
    q = torch.randn(1, T, H, Dk, device=device, dtype=torch.bfloat16) * 0.5
    k = torch.randn(1, T, H, Dk, device=device, dtype=torch.bfloat16) * 0.5
    v = torch.randn(1, T, H, Dv, device=device, dtype=torch.bfloat16) * 0.5
    beta = torch.rand(1, T, H, device=device, dtype=torch.bfloat16)
    g = -torch.rand(1, T, H, device=device, dtype=torch.bfloat16) * 0.05
    S0 = torch.zeros(1, H, Dk, Dv, device=device, dtype=torch.float32)
    return q, k, v, g, beta, S0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lens", default="256,2048,8192")
    ap.add_argument("--chunk", default="64")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--check", action="store_true",
                    help="assert this tool's staged copy equals engine/gdn.py's function")
    a = ap.parse_args()

    print(f"one GDN layer: B=1, H=48 value heads, Dk=Dv=128, fp32 inside, peak {PEAK_GBPS:.0f} GB/s")
    for T in [int(x) for x in a.lens.split(",")]:
        for chunk in [int(x) for x in a.chunk.split(",")]:
            q, k, v, g, beta, S0 = run(T, chunk, a.reps, a.device)
            with torch.no_grad():
                ref, Sref = gdn.chunk_gated_delta_rule(q, k, v, g, beta, S0, chunk_size=chunk)
                torch.cuda.synchronize()
                if a.check:
                    log, launches = [], {}
                    got, Sgot = staged(q, k, v, g, beta, S0, chunk, log, launches)
                    d = (got.float() - ref.float()).abs().max().item()
                    dS = (Sgot - Sref).abs().max().item()
                    print(f"  [check] T={T} chunk={chunk}  max |out| diff {d:.3e}  "
                          f"max |S| diff {dS:.3e}")
                # the shipped function, whole, for the stage sum to be read against
                torch.cuda.synchronize()
                ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
                ev[0].record()
                for _ in range(a.reps):
                    gdn.chunk_gated_delta_rule(q, k, v, g, beta, S0, chunk_size=chunk)
                ev[1].record()
                torch.cuda.synchronize()
                whole = ev[0].elapsed_time(ev[1]) / a.reps

                totals: dict[str, list[float]] = {}
                bytes_of: dict[str, float] = {}
                launches: dict[str, int] = {}
                for _ in range(a.reps):
                    log: list = []
                    staged(q, k, v, g, beta, S0, chunk, log, launches)
                    torch.cuda.synchronize()
                    for name, ea, eb, gbv in log:
                        totals.setdefault(name, []).append(ea.elapsed_time(eb))
                        bytes_of[name] = gbv

            print(f"\n[T={T}  chunk={chunk}]  the shipped function: {whole:8.3f} ms   "
                  f"({T / (whole * 48 / 1e3):.0f} tok/s if the whole pass were 48 of these)")
            print(f"    {'stage':34s}{'ms':>9}{'GB':>8}{'floor ms':>10}{'x floor':>9}"
                  f"{'launches':>10}")
            tot = 0.0
            for name in totals:
                ms = sum(totals[name]) / len(totals[name])
                tot += ms
                floor = bytes_of[name] / PEAK_GBPS * 1e3
                print(f"    {name:34s}{ms:9.3f}{bytes_of[name]:8.3f}{floor:10.3f}"
                      f"{(ms / floor if floor else 0):9.1f}{launches.get(name, 0):10d}")
            print(f"    {'sum of the stages':34s}{tot:9.3f}"
                  f"{sum(bytes_of.values()):8.3f}"
                  f"{sum(bytes_of.values()) / PEAK_GBPS * 1e3:10.3f}"
                  f"{'':9}{sum(launches.values()):10d}")


if __name__ == "__main__":
    main()
