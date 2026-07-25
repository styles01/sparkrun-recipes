# SM121-Optimized vLLM v0.25.0 — Research Lead

**Source:** https://x.com/mr_r0b0t/status/2077077113211949223
**Author:** mr_r0b0t (r0b0tlab on GitHub)
**Date observed:** July 14, 2026

## The Signal

SM121-optimized vLLM v0.25.0 serving Qwen 3.6 35B NVFP4 on DGX Spark (GB10) at 93.05 tok/s with MTP=2.

## Key Repos

| Repo | Stars | Relevance |
|---|---|---|
| `r0b0tlab/vllm-v0250-cu130-sm121` | 1 fork | **vLLM v0.25.0 container built for SM121 (GB10)** — potential upgrade for all our models |
| `r0b0tlab/qwen36-35b-a3b-nvfp4-sm121-vllm` | 5 | 35B recipe for SM121-optimized vLLM |
| `r0b0tlab/qwen36-35b-a3b-nvfp4-fast-sm121-vllm` | 6 | "fast" variant of 35B recipe |
| `r0b0tlab/hermes-concurrent-agents` | 71 | **Concurrent Hermes agents on GB10** — directly relevant to our multi-agent setup |
| `r0b0tlab/deepseek-v4-flash-nvfp4-gb10-benchmark` | 10 | DS4 in NVFP4 on Spark (we use FP8) |
| `r0b0tlab/nvidia-qwen-3.6-27B-sm121-nvfp4` | 5 | 27B NVFP4 on SM121 |
| `r0b0tlab/qwen36-35b-a3b-nvfp4-gb10-native-mtp` | 13 | Original 35B MTP reproducibility pack |

## Comparison to Our Setup

| | Our 35B | Their 35B |
|---|---|---|
| vLLM | v0.24.0 | **v0.25.0 SM121-optimized** |
| MTP | k=3 | k=2 |
| Speed | 102.8 tok/s | 93.05 tok/s |
| CUDA graphs | 21GB | Unknown |

We're already faster at k=3 (102.8 vs 93.05), but the v0.25.0 SM121 container could improve both 35B AND 122B.

## Action Items

1. **Test v0.25.0 SM121 container** with our 122B n=4 config — see if decode speed improves
2. **Test v0.25.0 SM121 container** with our 35B MTP k=3 config — compare to our 102.8 tok/s
3. **Review `hermes-concurrent-agents`** repo — they may have patterns we can adopt for multi-agent orchestration on GB10
4. **Review DS4 NVFP4 benchmark** — they got DS4 working in NVFP4, potentially smaller than our FP8 (227GB)
5. **Check if v0.25.0 fixes** any of the bugs we patched (PR #48375 Mamba, flashinfer_b12x)