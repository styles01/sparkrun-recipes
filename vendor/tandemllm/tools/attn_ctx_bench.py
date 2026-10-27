"""Verify attention (tools/attn_kernels.py decode_attention) at 8k-100k context: usage attn_ctx_bench.py OUT.json.
Verify attention at long context, no model: 16 layers of bf16 KV [4 heads, ctx, 256] (a ring the size of the
model's, so nothing sits in L2), T query rows of 24 heads, a tree-shaped mask; decode_attention per layer as the
engine launches it, the split kernel alone and the split + combine, CUDA events, median of 9. Reports ms over the
16 layers and the KV bytes' GB/s; and the partials' bytes (ns x R x 4 x 256 fp32 written + read)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json, math, torch
import tools.attn_kernels as A
torch.manual_seed(0)
out = {}
for ctx in (8192, 32768, 65536, 100000):
    L = 16
    ks = [torch.randn(1, 4, ctx, 256, device="cuda", dtype=torch.bfloat16) for _ in range(L)]
    vs = [torch.randn(1, 4, ctx, 256, device="cuda", dtype=torch.bfloat16) for _ in range(L)]
    for T in (16, 24, 32):
        q = torch.randn(1, 24, T, 256, device="cuda", dtype=torch.bfloat16)
        par = [-1] + [max(0, i - 1 - (i % 3 == 0)) for i in range(1, T)]
        m = torch.zeros(T, T, dtype=torch.bool, device="cuda")
        for i in range(T):
            j = i
            while j >= 0:
                m[i, j] = True; j = par[j]
        start = ctx - T
        def run():
            for l in range(L):
                A.decode_attention(q, ks[l], vs[l], start, m)
        for _ in range(3): run()
        ts = []
        for _ in range(9):
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            torch.cuda.synchronize(); s.record(); run(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
        ts.sort(); ms = ts[4]
        bm, groups, ns, chunk = A.pick_launch(T, 6, ctx)
        kvb = L * 4 * ctx * 256 * 2 * 2
        pb = L * 2 * ns * 6 * T * 4 * 256 * 4
        # the split kernel alone (combine skipped): time the split launch the same way
        from torch.profiler import profile, ProfilerActivity
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            run(); torch.cuda.synchronize()
        k = {e.key: (getattr(e, "self_device_time_total", 0) or getattr(e, "self_cuda_time_total", 0)) / 1e3 for e in prof.key_averages()}
        sp = sum(v for kk, v in k.items() if "attn_split" in kk); cb = sum(v for kk, v in k.items() if "attn_combine" in kk)
        r = {"ms16": round(ms, 3), "kv_GBs": round(kvb / ms / 1e6, 1), "split_ms": round(sp, 3), "combine_ms": round(cb, 3),
             "ns": ns, "chunk": chunk, "partials_MB": round(pb / 1e6, 1)}
        out[f"{ctx}/{T}"] = r
        print(f"[attn] ctx {ctx:6d} T {T:2d}: {ms:6.2f} ms / 16 layers, KV {kvb/1e9:.2f} GB at {r['kv_GBs']} GB/s; split {sp:.2f} combine {cb:.2f} ms; ns {ns} chunk {chunk}, partials {pb/1e6:.0f} MB", flush=True)
    del ks, vs; torch.cuda.empty_cache()
json.dump(out, open(sys.argv[1], "w"), indent=1)
