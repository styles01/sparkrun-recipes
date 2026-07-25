# DFlash Paper Findings — Best Practices for Agentic Workloads

**Source:** arXiv:2602.06036 (DFlash: Block Diffusion for Flash Speculative Decoding, ICML 2026)
**Date:** July 13, 2026
**Relevance:** Our 122B DFlash deployment on DGX Spark

## Key Findings

### 1. Block Size Generalization
- Models trained at block size 16 generalize well to smaller inference-time block sizes
- Our drafter was trained at BS=16; running at n=6 is fine (37% of training BS)
- Reverse doesn't hold: training at BS=8 and inferring at BS=16 doesn't work

### 2. Concurrency Reduces Speedup
| Concurrency | Code Speedup | Chat Speedup |
|---|---|---|
| 1 | 4.2-5.1x | 2.2-3.2x |
| 4 | 3.6-4.5x | — |
| 8 | 3.2-4.5x | — |
| 16 | 2.7-3.9x | — |
| 32 | 2.4-2.9x | — |

At 3 concurrent agents, we're in the 4x code / 2-3x chat range.

### 3. CRITICAL: Smaller n Is Better Under Concurrent Load

> "In practical serving scenarios, large blocks can increase verification cost under compute-bound settings (e.g., large batch sizes); reducing the block size in such cases can therefore yield better overall speedup."

With 3 concurrent agents (compute-bound), large n wastes verification compute on rejected drafts. Smaller n = less wasted compute = better effective speedup.

### 4. Chat/Prose Always Gets Lower Acceptance
- Code: 4-5x speedup, τ 5-8 (high acceptance)
- Chat (MT-Bench): 2-2.5x speedup, τ 2-3 (lower acceptance)
- Our 20-39% acceptance on chat is exactly what the paper predicts

### 5. Block Size Mismatch Ablation (Table 8)
| Train BS | Test BS | Math500 | HumanEval | MT-Bench |
|---|---|---|---|---|
| 16 | 16 | 4.64x / 6.33τ | 3.96x / 5.29τ | 2.45x / 3.09τ |
| 8 | 8 | 3.97x / 5.21τ | 3.53x / 4.61τ | 2.22x / 3.29τ |
| 16 | 8 | ~4.0x / ~5.2τ | ~3.5x / ~4.6τ | ~2.2x / ~3.3τ |

BS=16 model at inference BS=8 ≈ native BS=8 model. Our BS=16 model at n=6 should perform similarly.

## Recommendation for Our Workload

**Current: n=6 at 3 concurrent agents**
- Code: ~35-39% acceptance, ~40 tok/s with 2 concurrent
- Chat: ~20-31% acceptance, ~25 tok/s

**Potential: n=4 at 3 concurrent agents**
- Less wasted verification compute under load
- Code acceptance might drop to ~30% but effective speed could be same or better
- Chat acceptance might increase to ~25% (less wasted drafts)
- More KV cache headroom (361K tokens at n=4 vs 307K at n=12 at same GMU)
- Paper says this is the right direction for concurrent serving

**n=3 might be too aggressive** — diminishing returns, drafter barely does anything

## Our Tested Data Points
| n | GMU | CTX | KV Tokens | Code Accept | Chat Accept | Free RAM |
|---|---|---|---|---|---|---|
| 4 | 0.75 | 262K | 361,258 | TBD | TBD | 37GB |
| 6 | 0.83 | 165K | 458,423 | 35-39% | 20-31% | 16GB |
| 12 | 0.75 | 262K | 307,200 | 45% | 9% | 25GB |

## TODO
- [ ] Test n=4 at GMU 0.83, 150K, 3 lanes — compare effective speed under 3-agent load
- [ ] Benchmark n=4 vs n=6 on code AND chat with same concurrency
- [ ] If n=4 is better under load, lock it as production