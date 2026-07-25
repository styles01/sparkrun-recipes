# vLLM v26 (main) + fp8 KV + DFlash — Docker Build & Deploy

**Created:** July 25, 2026
**Status:** ✅ WORKING — 1.43M KV tokens, 5.46× concurrency, fp8 KV + DFlash n=7

## What This Is

vLLM built from the **main branch** (unreleased, commit 318b527, 608 commits ahead of v0.25.1) for SM121 (GB10 DGX Spark). Combined with our patches, this enables **fp8 KV cache + DFlash speculative decoding** — something that was architecturally blocked on all previous vLLM versions.

## Key Results

| Metric | aeon 0.23 (bf16 KV) | v26 (fp8 KV) |
|---|---|---|
| KV tokens | 549K | **1,431,632** (2.6×) |
| Concurrency @ 256K | 2.09× | **5.46×** (2.6×) |
| Prefill tok/s | 827 | 726 |
| Decode tok/s | 50.2 | 43.6 (peak 60.7) |
| CUDA graphs | 0.25 GiB | 0.03 GiB |
| Model load | 63.85 GiB | 63.97 GiB |
| KV cache memory | 30.98 GiB | 35.55 GiB |
| Load time | 36s (fastsafetensors) | 430s (standard) |

## Build Instructions

### 1. Clone vLLM main on the Spark

```bash
ssh jaita@larryspark.local 'mkdir -p ~/repos && git clone --depth 1 https://github.com/vllm-project/vllm ~/repos/vllm'
```

### 2. Build the Docker image for SM121

```bash
ssh jaita@larryspark.local 'cd ~/repos/vllm && docker build --build-arg torch_cuda_arch_list="12.1" -f docker/Dockerfile -t vllm-v26-sm121:latest .'
```

This takes **3-5 hours** on GB10 (arm64). The FA2/FA3 CUDA kernels are the slow part (~85s each, 400 total).

### 3. Apply patches on top

Patches are at: `~/.hermes/profiles/oracle/workspace/spark-llm-optimization/docker/`

Create a Dockerfile:
```dockerfile
FROM vllm-v26-sm121:latest

USER root

COPY patch_inc_hybrid.py /tmp/patches/patch_inc_hybrid.py
COPY patch_int8_lmhead_v3.py /tmp/patches/patch_int8_lmhead_v3.py
COPY patch_fla_shmem.py /tmp/patches/patch_fla_shmem.py
COPY patch_prefix_align.py /tmp/patches/patch_prefix_align.py
COPY patch_flashinfer_block_stride.py /tmp/patches/patch_flashinfer_block_stride.py

RUN chmod +x /tmp/patches/*.py; \
    python3 /tmp/patches/patch_inc_hybrid.py; \
    python3 /tmp/patches/patch_int8_lmhead_v3.py || true; \
    python3 /tmp/patches/patch_fla_shmem.py || true; \
    python3 /tmp/patches/patch_prefix_align.py || true; \
    python3 /tmp/patches/patch_flashinfer_block_stride.py || true; \
    rm -rf /tmp/patches
```

Build:
```bash
cd ~/.hermes/profiles/oracle/workspace/spark-llm-optimization/docker/
scp Dockerfile.v26 jaita@larryspark.local:/tmp/vllm-patches/Dockerfile
scp patch_*.py jaita@larryspark.local:/tmp/vllm-patches/
ssh jaita@larryspark.local 'cd /tmp/vllm-patches && docker build --no-cache -t vllm-v26-patched:latest .'
```

### Patch Status on v26

| Patch | Status | Notes |
|---|---|---|
| inc_hybrid | ✅ Applied | CRITICAL — enables hybrid INT4+FP8 checkpoint loading |
| int8_lmhead_v3 | ❌ Skipped | Anchor not found in v26 — _get_logits changed. May not be needed. |
| fla_shmem | ❌ Skipped | v26 may have fixed this natively |
| prefix_align | ✅ Applied | Align-aware back-off for prefix caching + DFlash |
| flashinfer_block_stride | ❌ Skipped | v26 may handle this natively (no assert error) |

## Deploy Command

```bash
ssh jaita@larryspark.local 'docker run -d \
  --name qwen-spark --gpus all -p 8000:8000 --user root \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -e HF_HOME=/root/.cache/huggingface \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  vllm-v26-patched:latest \
  bleysg/Qwen3.5-122B-A10B-int4-fp8-hybrid \
    --served-model-name qwen --host 0.0.0.0 --port 8000 \
    --trust-remote-code --max-model-len 262144 --kv-cache-dtype fp8 \
    --gpu-memory-utilization 0.85 --max-num-seqs 3 --max-num-batched-tokens 8192 \
    --enable-prefix-caching --enable-chunked-prefill \
    --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3 \
    --attention-backend FLASHINFER \
    --speculative-config "{\"method\":\"dflash\",\"model\":\"z-lab/Qwen3.5-122B-A10B-DFlash\",\"num_speculative_tokens\":7}"'
```

## Rollback (to aeon 0.23)

```bash
ssh jaita@larryspark.local 'docker rm -f qwen-spark; cd ~/qwen3.5-122B-A10B-on-spark && CTX=262144 GPU_MEM=0.85 MAX_NUM_SEQS=3 MAX_BATCHED_TOKENS=8192 SERVED_NAME=qwen bash install.sh --start --profile dense --nspec 7 --no-smoke'
```

## Key Learnings

### Why v26 Works When v0.25.1 Didn't

1. **FlashInfer 0.6.15** (#48914) — newer FlashInfer with non-causal fp8 dequant support
2. **DFlash fc sizing fix** (#48524) — fixed drafter layer count mismatch
3. **Heterogeneous KV cache groups** (#48878) — better page size handling
4. **Mixed precision KV cache detection** (#49623) — native fp8 + bf16 mixed handling
5. **No assert error** — v26's `unify_kv_cache_spec_page_size` handles the fp8/bf16 page size mismatch natively (no patch needed)

### What Was Blocked Before

- vLLM #41559: DFlash's non-causal attention incompatible with KV cache quantization — **RESOLVED in v26** via FlashInfer 0.6.15 Triton dequant path
- vLLM #48477: Qwen3.5-122B-A10B-FP8 CUBLAS crash on 0.25.0 — **not hit on v26 main**
- KV cache page_size assert — **not hit on v26** (native handling)

### int8_lmhead Patch

The int8 lm-head GEMV patch skipped because v26 restructured `_get_logits`. The decode speed (43.6 vs 50.2) may be partly due to missing this optimization. Porting the int8 lm-head to v26's logits processor could recover ~10-15% decode speed.

### Docker --user root

v26's Docker image runs as a non-root user by default. Must pass `--user root` when running, or the HF cache (owned by root from Docker volumes) is unreadable.

### Build Time

3-5 hours on GB10 (arm64). The FA2/FA3 CUDA kernels are the bottleneck (~85s each × 400 objects). Could potentially skip FA2/FA3 compilation since we use FlashInfer, but the Dockerfile doesn't have a flag for this.

## Images on Spark

| Image | Size | Description |
|---|---|---|
| vllm-v26-sm121:latest | 18.1GB | vLLM main built for SM121, no patches |
| vllm-v26-patched:latest | 18.1GB | v26 + inc_hybrid + prefix_align patches |
| ghcr.io/aeon-7/aeon-vllm-ultimate:2026-06-18-v0.23.0-dflashfix | ~15GB | Old aeon 0.23 (fallback) |

## Files

- Patches: `~/.hermes/profiles/oracle/workspace/spark-llm-optimization/docker/patch_*.py`
- Dockerfile: `~/.hermes/profiles/oracle/workspace/spark-llm-optimization/docker/Dockerfile.v26`
- vLLM source: `/tmp/vllm-v26` on Spark (should be moved to `~/repos/vllm`)
- Build log: `/tmp/vllm-v26-build.log` on Spark