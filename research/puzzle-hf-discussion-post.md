# Puzzle-75B NVFP4 MTP on DGX Spark: 8 attempts, 7 OOM crashes — can't replicate moranilt's 32 tok/s

## TL;DR

Puzzle-75B A9B NVFP4 runs fine on DGX Spark (GB10, 121GB unified memory, SM121) in eager mode without MTP — 18.9 tok/s, 44.49GB weights, 724K KV tokens, 23GB free RAM. But every MTP config (k=2, k=3) OOMs during FlashInfer autotune / CUDA graph compile, regardless of container version, GMU, or flags. Can't replicate @moranilt's 32.2 tok/s with MTP k=3 from discussion #3.

## What works

**Enforce-eager, no MTP:**
```
Container: ghcr.io/aeon-7/aeon-vllm-ultimate:2026-06-18-v0.23.0-dflashfix
Entry: /usr/local/bin/vllm serve

vllm serve nvidia/NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-NVFP4 \
  --served-model-name puzzle-75b \
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 1 --trust-remote-code \
  --quantization fp4 --moe-backend marlin \
  --kv-cache-dtype fp8 \
  --mamba-backend flashinfer \
  --mamba-ssm-cache-dtype float16 \
  --enable-mamba-cache-stochastic-rounding \
  --mamba-cache-philox-rounds 5 \
  --enable-expert-parallel \
  --async-scheduling --enable-chunked-prefill \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser nemotron_v3 \
  --enforce-eager \
  --gpu-memory-utilization 0.73 \
  --max-model-len 160000 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 8192 \
  --load-format fastsafetensors
```

Result: 44.49GB weights, 724,266 KV tokens, 4.53x concurrency at 160K, 23GB free RAM, 18.9 tok/s. Stable.

## What doesn't work — every MTP attempt OOMs

| # | Container | MTP | Eager | drop_caches | Mamba flags | Result |
|---|---|---|---|---|---|---|
| 1 | vLLM v0.24.0 | k=3 | ❌ | ❌ | ❌ | OOM during compile |
| 2 | vLLM v0.24.0 | k=3 | ❌ | ❌ | ❌ | OOM (lower GMU 0.70) |
| 3 | vLLM v0.24.0 | none | ❌ | ❌ | ❌ | OOM (v0.24.0 autotune alone crashes) |
| 4 | AEON v0.23.0 | k=3 | ❌ | ❌ | ✅ | OOM during compile |
| 5 | AEON v0.23.0 | none | ✅ | ❌ | ✅ | ✅ 18.9 tok/s |
| 6 | AEON v0.23.0 | k=2 | ✅ | ❌ | ✅ | OOM (eager doesn't stop autotune) |
| 7 | vLLM v0.25.0 | k=2 | ❌ | ✅ | ✅+align | OOM (#3738 regression?) |
| 8 | AEON v0.23.0 | k=2 | ❌ | ✅ | ✅+align | OOM during autotune |

All MTP attempts peak at 121GB (full unified memory) during FlashInfer MoE autotune + CUDA graph capture, then crash. Even with `--enforce-eager` (attempt 6), FlashInfer autotune still runs and still OOMs.

## Root causes identified (from research)

**1. FlashInfer MoE autotune workspace on SM121**
FlashInfer FP4 MoE autotune allocates huge workspace tensors testing kernel configurations. On SM121 (GB10), the FP4 kernels don't fit in shared memory — CUTLASS exceeds SMEM, cuDNN unsupported, CuTe DSL missing SM121. The autotune workspace alone can be ~20GB. This is NOT the drafter's weight footprint.

**2. FlashInfer PR #3738 regression in v0.25.0**
A FlashInfer library change narrowed NVFP4 MoE autotune workspace to FP8-activation family, breaking native FP4+FP4 on SM121. @moranilt's v0.23.0 image predates this — likely why their config works and v0.25.0 doesn't.

**3. Drafter inherits wrong MoE backend**
The MTP draft head's MoE is unquantized BF16 but silently inherits the outer `--moe-backend`. NVIDIA's official fix: set `"moe_backend":"triton"` INSIDE `--speculative-config`.

**4. Mamba cache mode crash**
`mamba_cache_mode=all` auto-activates with `--enable-prefix-caching` and crashes with MTP. Fix: `--mamba-cache-mode align` OR remove `--enable-prefix-caching`.

## What we haven't tried yet

- `--compilation-config '{"cudagraph_mode":"piecewise"}'` — reduces CUDA graph memory from 13GB to 5GB
- `VLLM_NVFP4_GEMM_BACKEND=flashinfer-cutlass` — @rmagur1203's working backend (different from marlin)
- eugr's patched Docker (FlashInfer #3738 fix, commit 562ed29, 2 days old)
- Removing `--enable-prefix-caching` entirely (NVIDIA's official approach)

## Comparison to other models on same hardware

| Model | Weights | Compile time | Peak memory | Settled memory | MTP | Result |
|---|---|---|---|---|---|---|
| Qwen 122B DFlash n=4 | 67GB | 2 min | ~104GB | 104GB | ✅ DFlash | Works |
| Qwen 35B MTP k=3 | 22GB | 3 min | ~76GB | 76GB | ✅ MTP | Works |
| Step 3.7 Flash | 99GB | N/A (llama.cpp) | 111GB | 111GB | N/A | Works |
| Puzzle-75B (no MTP) | 44GB | N/A (eager) | 98GB | 98GB | ❌ | Works |
| Puzzle-75B (MTP) | 44GB | — | 121GB+ | — | ❌ | OOM |

A 44GB model eating more memory than a 67GB model during compile is the red flag. The hybrid Mamba2 + LatentMoE + Attention architecture has three compute patterns, each needing separate autotune passes. MTP adds a fourth. The autotune workspace doesn't get freed before CUDA graph capture starts — peak memory = weights + drafter + autotune workspace + graph buffers, all alive simultaneously.

## Environment

- Hardware: NVIDIA DGX Spark (GB10 Grace Blackwell, 121GB unified memory, SM121)
- OS: Ubuntu 24.04 (aarch64)
- Docker containers tested: vllm/vllm-openai:v0.24.0, vllm/vllm-openai:v0.25.0, ghcr.io/aeon-7/aeon-vllm-ultimate:2026-06-18-v0.23.0-dflashfix
- Model: nvidia/NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-NVFP4 (50GB on disk)
- Env vars: OMP_NUM_THREADS=4, CUDA_MANAGED_FORCE_DEVICE_ALLOC=1, PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True, VLLM_ALLOW_LONG_MAX_MODEL_LEN=1, VLLM_USE_FLASHINFER_MOE_FP4=0, VLLM_MARLIN_USE_ATOMIC_ADD=1, VLLM_NVFP4_GEMM_BACKEND=marlin, VLLM_FLASHINFER_ALLREDUCE_BACKEND=trtllm

## Questions for the community

1. Has anyone else gotten MTP working with Puzzle on a single Spark? Not clustered, not multi-GPU.
2. Is `--compilation-config '{"cudagraph_mode":"piecewise"}'` the missing piece? Has anyone tried it?
3. Does eugr's FlashInfer #3738 patch (commit 562ed29) fix the autotune workspace OOM?
4. @moranilt — what Docker image exactly did you use? The stock vllm/vllm-openai:v0.23.0 or a custom build? Did you run drop_caches?

Happy to provide full docker logs from any of the 8 attempts on request.