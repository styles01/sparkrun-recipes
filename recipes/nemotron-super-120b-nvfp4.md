# Nemotron Super 120B NVFP4 — Spark Arena Recipe

**Source:** https://spark-arena.com/benchmark/3f73ed2d-73b2-446a-bb5f-d64d7da041eb
**Author:** Seth Hobson
**Hardware:** NVIDIA DGX Spark (GB10, single node)
**Claimed speed:** 30+ tok/s (21.66 at single-stream baseline)
**Date observed:** July 14, 2026

## Recipe (from Spark Arena)

```yaml
model: nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4
name: Nemotron-3-Super-NVFP4-Marlin-MTP
container: vllm/vllm-openai:v0.20.0-aarch64-cu130-ubuntu2404
solo_only: true

command: |
  vllm serve nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 \
    --served-model-name nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 nemotron-3-super \
    --tensor-parallel-size 1 \
    --port 8000 --host 0.0.0.0 \
    --max-model-len 131072 \
    --max-num-seqs 10 \
    --max-num-batched-tokens 32768 \
    --gpu-memory-utilization 0.85 \
    --quantization fp4 \
    --moe-backend marlin \
    --kv-cache-dtype fp8 \
    --mamba-ssm-cache-dtype float32 \
    --async-scheduling \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --reasoning-parser nemotron_v3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --speculative-config '{"method":"mtp","num_speculative_tokens":1,"moe_backend":"triton"}' \
    --trust-remote-code

env:
  OMP_NUM_THREADS: '4'
  CUDA_MANAGED_FORCE_DEVICE_ALLOC: '1'
  PYTORCH_CUDA_ALLOC_CONF: expandable_segments:True
  VLLM_ALLOW_LONG_MAX_MODEL_LEN: '1'
  VLLM_USE_FLASHINFER_MOE_FP4: '0'
  VLLM_MARLIN_USE_ATOMIC_ADD: '1'
  VLLM_NVFP4_GEMM_BACKEND: marlin
```

## Key Observations

- **NVFP4 quant** — same format as our Qwen 35B, 67GB on disk
- **MTP k=1** — very conservative, could try k=3 like our 35B
- **GMU 0.85** — higher than our 122B (0.83)
- **128K context** — could push higher (1M native)
- **10 max seqs** — way more concurrency than our 3
- **v0.20.0 container** — older vLLM, might need newer for our patches
- **marlin MoE backend** — same as our 35B
- **nemotron_v3 reasoning parser** — native reasoning support
- **qwen3_coder tool parser** — tool calling works

## Our Tweaks to Test

1. **MTP k=3** instead of k=1 — should boost speed significantly
2. **GMU 0.83** instead of 0.85 — match our 122B config for fair comparison
3. **150K context** instead of 128K — match our production config
4. **3 lanes** instead of 10 — match our actual usage pattern
5. **v0.24.0 container** — our tested version with Spark patches
6. **PR #48375 Mamba patch** — might be needed (hybrid Mamba model like 35B)

## Intelligence Comparison (from model cards)

| Benchmark | Qwen 122B | Nemotron Super 120B | Winner |
|---|---|---|---|
| MMLU-Pro | 86.70 | 83.73 | Qwen +3 |
| AIME25 | 90.36 | 90.21 | Tie |
| GPQA | 86.60 | 79.23 | Qwen +7 |
| HLE | 25.30 | 18.26 | Qwen +7 |
| HMMT Feb25 | 91.40 | 93.67 | **Nemotron +2.3** |
| HMMT w/tools | 89.55 | 94.73 | **Nemotron +5.2** |
| SciCode | 42.00 | 42.05 | Tie |
| Context | 262K | **1M** | Nemotron |

Nemotron wins on math (HMMT) and context length. Qwen wins on reasoning (GPQA, HLE) and general knowledge (MMLU-Pro).

## Disk Size: 67GB (same as Qwen 122B)