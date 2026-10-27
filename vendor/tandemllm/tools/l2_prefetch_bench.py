"""The first measurement: what a projection gains when its weights are already in L2.

Between a GDN layer's projections the mixer runs for ~0.1 ms with DRAM mostly idle. A graph branch that
prefetches the next projection's bytes into L2 during that time is worth building only if the
projection, run with its bytes resident, is at least ~30 us faster than cold -- and only if the lines
survive what the mixer itself streams in between. This measures exactly that, on the served skinny
kernel and tile, one projection at a time (every time is a CUDA-event median over --reps):

    cold        L2 flushed (a 96 MB read), then the projection
    warm        the projection's own bytes read once (w, s, the activation), then the projection
    warm+state  the same, then a stream over a mixer-sized buffer (read + write, --state-mb), then
                the projection -- the lines the mixer's state traffic leaves
    prefetch    `prefetch.global.L2::evict_last` over the bytes (a 64-CTA kernel), then the projection;
                and the prefetch kernel's own time

    python tools/l2_prefetch_bench.py --shapes 5120:6144,6144:5120,10240:5120 --rows 16 --state-mb 3.1
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_PF = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <stdint.h>

__global__ void l2_prefetch(const uint8_t* __restrict__ p, size_t n) {
    size_t step = (size_t)gridDim.x * blockDim.x * 128;
    for (size_t o = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * 128; o < n; o += step)
        asm volatile("prefetch.global.L2::evict_last [%0];" :: "l"(p + o));
}

void prefetch(torch::Tensor t, int64_t ctas) {
    l2_prefetch<<<(int)ctas, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const uint8_t*>(t.data_ptr()), (size_t)t.numel() * t.element_size());
}
"""
_MOD = None


def _prefetch_mod():
    global _MOD
    if _MOD is None:
        from torch.utils.cpp_extension import load_inline
        venv_bin = os.path.dirname(sys.executable)                # ninja lives there, as in nvfp4_skinny
        if venv_bin not in os.environ.get("PATH", "").split(os.pathsep):
            os.environ["PATH"] = venv_bin + os.pathsep + os.environ.get("PATH", "")
        _MOD = load_inline(name="qwen38_l2_prefetch", cpp_sources=["void prefetch(torch::Tensor t, int64_t ctas);"],
                           cuda_sources=[_PF], functions=["prefetch"],
                           extra_cuda_cflags=["-O3", "-gencode=arch=compute_121a,code=sm_121a"])
    return _MOD


def timed(fn, reps: int) -> float:
    """Median over `reps` of the CUDA time of fn()'s LAST launch sequence (setup inside fn is
    excluded by recording the events around `fn()`'s returned callable)."""
    out = []
    for _ in range(reps):
        run = fn()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        run()
        b.record()
        b.synchronize()
        out.append(a.elapsed_time(b) * 1e3)
    return statistics.median(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shapes", default="5120:6144,6144:5120,10240:5120,17408:5120,5120:17408")
    ap.add_argument("--rows", type=int, default=16)
    ap.add_argument("--state-mb", type=float, default=3.1)
    ap.add_argument("--reps", type=int, default=21)
    ap.add_argument("--ctas", type=int, default=64)
    a = ap.parse_args()
    from tools.nvfp4_linear import quantize_to_nvfp4
    from tools import nvfp4_skinny as SK
    from tools.nvfp4_skinny import nvfp4_matmul_skinny
    flush = torch.empty(96 * 2 ** 20 // 4, dtype=torch.float32, device="cuda")
    state = torch.randn(int(a.state_mb * 2 ** 20 / 4), dtype=torch.float32, device="cuda")
    pf = _prefetch_mod()
    print(f"rows {a.rows}, state {a.state_mb} MB, {a.reps} reps, prefetch {a.ctas} CTAs x 256")
    print(f"{'shape':>12} {'MB':>6} {'cold us':>9} {'warm':>8} {'warm+state':>11} {'prefetch':>9} "
          f"{'pf kernel':>10} {'saved us':>9}")
    for shp in a.shapes.split(","):
        N, K = (int(v) for v in shp.split(":"))
        w = quantize_to_nvfp4(torch.randn(N, K, device="cuda") * 0.02)
        x = torch.randn(a.rows, K, device="cuda").to(torch.bfloat16)
        out = torch.empty(a.rows, N, dtype=torch.bfloat16, device="cuda")
        nvfp4_matmul_skinny(x, w, out=out)
        # the scale bytes the kernel reads: the run copy (made by the call above) when it is on
        sc = w._srun if SK.SRUN else w.s
        mb = (w.w.numel() + sc.numel()) / 2 ** 20

        def gemm():
            return lambda: nvfp4_matmul_skinny(x, w, out=out)

        def cold():
            flush.zero_()
            return gemm()

        def warm():
            flush.zero_()
            w.w.view(torch.int32).sum(); sc.view(torch.uint8).sum()
            return gemm()

        def warm_state():
            flush.zero_()
            w.w.view(torch.int32).sum(); sc.view(torch.uint8).sum()
            state.mul_(1.0000001)
            return gemm()

        def prefetched():
            flush.zero_()
            pf.prefetch(w.w, a.ctas); pf.prefetch(sc, a.ctas)
            torch.cuda.synchronize()
            return gemm()

        def pf_only():
            flush.zero_()
            return lambda: (pf.prefetch(w.w, a.ctas), pf.prefetch(sc, a.ctas))

        t = [timed(f, a.reps) for f in (cold, warm, warm_state, prefetched, pf_only)]
        print(f"{N}x{K:>6} {mb:6.1f} {t[0]:9.1f} {t[1]:8.1f} {t[2]:11.1f} {t[3]:9.1f} {t[4]:10.1f} "
              f"{t[0] - t[3]:9.1f}", flush=True)
        del w


if __name__ == "__main__":
    main()
