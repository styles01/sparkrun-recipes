# 122B DFlash Memory Audit — Benchmark Results

**Date:** July 13, 2026
**Hardware:** DGX Spark (GB10, 121GB unified memory)
**Model:** Qwen 3.5 122B A10B (MoE, 10B active/token)
**Quant:** INT4+FP8 hybrid (dense profile)

## Test Results

### DFlash n=4 (GMU 0.75)

| Metric | Value |
|---|---|
| Model weights | 63.85 GiB |
| CUDA graph memory | **0.63 GiB** |
| Available KV cache | 19.15 GiB |
| GPU KV cache size | 361,258 tokens |
| Max concurrency @ 262K | 1.38x |
| Lanes @ 100K | ~3.6x (fits 3 lanes easily) |
| Lanes @ 120K | ~3.0x (exactly 3 lanes) |
| Free RAM | ~37GB |
| Boot time | ~5 min |
| Speed | TBD (need benchmark) |

### DFlash n=6 (GMU 0.83, CTX 165K) — TESTED

| Metric | Value |
|---|---|
| Model weights | 63.85 GiB |
| CUDA graph memory | 0.68 GiB |
| Available KV cache | 27.92 GiB |
| GPU KV cache size | 458,423 tokens |
| Max concurrency @ 165K | 2.78x (short of 3 lanes) |
| Free RAM | ~16GB |
| Boot time | ~5 min |
| Speed | TBD (need benchmark) |

### DFlash n=8 (GMU 0.88, CTX 165K) — DATA MAY BE CONTAMINATED

n=8 boot was killed by n=6 launch. Logs may be from n=6 container, not n=8.
Reported: graphs 0.68GB, KV 27.92GB, 458K tokens, 2.78x @ 165K, 17GB free.
**Treat these as n=6 numbers, not n=8.** Need clean retest for n=8.

## Complete Comparison Table

| n | GMU | CTX | Weights | Graphs | KV GB | KV Tokens | Concurrency | Free RAM |
|---|---|---|---|---|---|---|---|---|
| 4 | 0.75 | 262K | 63.85 | 0.63 | 19.15 | 361,258 | 1.38x @ 262K | 37GB |
| 6 | 0.83 | 165K | 63.85 | 0.68 | 27.92 | 458,423 | 2.78x @ 165K | 16GB |
| 12 | 0.75 | 262K | 63.85 | 0.47 | 19.27 | 307,200 | 1.17x @ 262K | 25GB |
| 8 | 0.88 | 165K | ??? | ??? | ??? | ??? | ??? | ??? (contaminated) |

## Drafter + Overhead Cost (Derived)

| n | GMU | Budget | - Weights - Graphs | KV allocated | Drafter+OH |
|---|---|---|---|---|---|
| 4 | 0.75 | 91GB | 26.5GB | 19.15GB | 7.35GB |
| 6 | 0.83 | 100.4GB | 35.9GB | 27.92GB | 7.98GB |
| 12 | 0.75 | 91GB | 26.5GB | 19.27GB | 7.23GB |

Drafter+OH is roughly 7-8GB, relatively stable across n values. The variation is more from GMU scaling (overhead grows with budget) than from n.

## KV Per Token

| n | KV GB | KV Tokens | KB/token |
|---|---|---|---|
| 4 | 19.15 | 361,258 | 55.3 |
| 6 | 27.92 | 458,423 | 63.0 |
| 12 | 19.27 | 307,200 | 65.2 |

KV per token varies 55-65KB. Use 63KB for estimates at GMU 0.83+.

## KEY FINDING: DFlash Graph Cost is Irrelevant

Both n=4 and n=12 have sub-1GB CUDA graph cost. n=12 is actually CHEAPER (0.47 vs 0.63GB).

The difference in KV tokens (361K n=4 vs 307K n=12) is from the drafter's activation memory scaling with n — the 0.8B drafter needs more working memory at n=12.

| n | Graphs | KV tokens | Free RAM | Lanes @ 100K |
|---|---|---|---|---|
| 4 | 0.63GB | 361,258 | ~37GB | 3.6x |
| 12 | 0.47GB | 307,200 | ~25GB | 3.07x |

n=12: max speed (82 tok/s) but tighter headroom (25GB)
n=4: more headroom (37GB) and more KV (361K) but lower speed (TBD)

## Key Finding So Far

**DFlash CUDA graphs are TINY — 0.63GB at n=4.**

This completely invalidates my earlier estimates of 15-50GB graph cost. I was wrong. DFlash's graph architecture is fundamentally different from MTP:

| Spec Decoding | Graph Cost | Why |
|---|---|---|
| 35B MTP k=3 | 21GB | MTP captures full forward graphs for 4 tokens × 6 batch sizes × 40 layers |
| 122B DFlash n=4 | 0.63GB | DFlash uses a separate drafter model — main model graphs are single-token, drafter has its own small graphs |

DFlash's drafter is a separate 0.8B model with its own graph space. The main 122B model only captures single-token-forward graphs regardless of n. The drafter predicts n tokens but its graphs are tiny (0.8B model).

This means **DFlash n value does NOT significantly affect CUDA graph memory**. n=12 should have similar graph cost to n=4.

## Comparison: 35B MTP vs 122B DFlash Graph Architecture

| Factor | 35B MTP k=3 | 122B DFlash n=4 |
|---|---|---|
| Graph approach | Single model, multi-token forward graphs | Separate drafter model, single-token main graphs |
| Main model graphs | 4 tokens/forward × 6 sizes × 40 layers = 21GB | 1 token/forward × batch sizes × layers = ~0.6GB |
| Drafter graphs | (integrated, no separate) | 0.8B drafter, tiny |
| Total graph cost | 21GB | 0.63GB |

## What This Means

If n=12 also shows ~0.63GB graphs (expected), then:
- 122B at n=12, GMU 0.75 = 64 + 0.63 + 19 KV = ~84GB used, ~37GB free
- 82 tok/s on code + 30GB+ headroom + 3 lanes at 120K
- **The holy grail: max speed + co-location**

## Pending

- [ ] n=12 graph cost (testing now)
- [ ] n=4 speed benchmark
- [ ] n=12 speed benchmark  
- [ ] n=4 prose acceptance (vs n=12's 9%)
- [ ] n=12 prose acceptance (confirm still 9%)
- [ ] Optimal max-model-len for 3 lanes (120K? 128K?)