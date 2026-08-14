# Recipe: Qwen 3.8 27B NVFP4

**Status:** ✅ Production (triton_attn + MTP k=2, stable with concurrency)
**Served name:** `qwen3.8-27b`
**Docker image:** `vllm/vllm-openai:nightly-aarch64` (v0.27.2rc1.dev77+)
**Tool calling:** ✅ `--enable-auto-tool-choice --tool-call-parser qwen3_coder`
**Reasoning parser:** `qwen3`
**Vision:** ✅ Multimodal (image_token_id present, vision_config)

> **Recipe contract:** [`recipes/qwen-38-27b.yaml`](../recipes/qwen-38-27b.yaml)
> **Reference:** [MiaAI-Lab/Qwen3.8-27B-DGX-Spark-RTX-6000](https://github.com/MiaAI-Lab/Qwen3.8-27B-DGX-Spark-RTX-6000)

## Model Location on Spark

```
~/models/hf/hub/models--unsloth--Qwen3.8-27B-NVFP4/snapshots/b0d9f9de93a9e98df9b1dd41ba444ab1139b1ab3/
```
~22 GB, NVFP4 (compressed-tensors). MTP draft head built into checkpoint (no separate repo).

> ⚠️ **Do NOT use `Qwen/Qwen3.8-27B-FP8` (54 GB)** — it has no calibrated FP8 KV scheme → bf16 KV → OOM during MTP. This was our root crash cause. Use the NVFP4 checkpoint.

## Model Specs

| Spec | Value |
|---|---|
| Architecture | `Qwen3_5ForConditionalGeneration` (hybrid Mamba/Gated DeltaNet + full attention) |
| Layers | 64 (48 linear attention + 16 full attention, interval 4) |
| Hidden | 5120 |
| Heads | 24, KV heads 4, head_dim 256 |
| Context | 262,144 (256K), extendable to 1M via YaRN |
| Vocab | 248,320 |
| Quant | NVFP4 (compressed-tensors) |
| Size | ~22 GB safetensors |
| MTP | Built-in (mtp_num_hidden_layers=1, mtp_use_dedicated_embeddings=false) |
| Acceptance | ~45-85% (varies by load) |

## CRITICAL: triton_attn Required (NOT flashinfer)

**FlashInfer + MTP crashes on SM121 (DGX Spark) with `cudaErrorIllegalAddress`.**

FlashInfer's MTP speculative decode path corrupts the Gated DeltaNet (GDN) recurrent state cache under concurrency (c≥5). The crash happens in `kv_cache_coordinator.py` where the EAGLE/MTP peek-and-drop path overreads Mamba/GDN recurrent state groups.

**Fix:** Use `--attention-backend triton_attn`. Only the 16 full-attention layers use this backend; the 48 GDN layers use their own Triton/FLA kernel unaffected by the bug.

- **GitHub #37754:** FlashInfer + MTP crashes on SM121 — confirmed fix: triton_attn
- **MiaAI-Lab:** Confirmed working with triton_attn + nightly-aarch64, no enforce-eager, MTP k=2

Note: FlashInfer may still appear in logs for the full-attention layers — this is expected. The GDN layers use their own kernel.

## CRITICAL: NO Mamba Patch

**Do NOT apply the PR #48375 Mamba patch to Qwen 3.8.**

- The patch was designed for Qwen 3.6's code layout (~48 layers)
- Qwen 3.8 has 64 layers with a different MTP implementation
- vLLM has **native MTP support** for Qwen 3.8 — auto-detects MTP model in checkpoint
- vLLM logs: *"Detected MTP model. Sharing target model embedding/lm_head weights with the draft model."*
- Applying the patch corrupts the native MTP support and causes `EngineDeadError`

## Docker Image: nightly-aarch64

Use `vllm/vllm-openai:nightly-aarch64` (not `v0.27.1` or `latest`).

- Nightly version: `v0.27.2rc1.dev77+gac7509e2b`
- Contains GDN speculative decoding optimizations (#48577), Mamba hybrid fixes (#49291)
- The `latest` tag resolves to v0.27.1 which lacks some fixes
- Set `CUTE_DSL_ARCH=sm_121a` for GB10 cutlass kernels (Mia Lab reference)

## Locked Config (Production)

```bash
docker run -d --name qwen38-spark \
  --gpus all --network host --ipc host --shm-size 32gb \
  --privileged \
  -e CUTE_DSL_ARCH=sm_121a \
  -e VLLM_FLOAT32_MATMUL_PRECISION=high \
  -v ~/models/hf:/root/.cache/huggingface \
  --entrypoint vllm \
  vllm/vllm-openai:nightly-aarch64 \
  serve unsloth/Qwen3.8-27B-NVFP4 \
    --served-model-name qwen3.8-27b \
    --host 0.0.0.0 --port 8000 \
    --trust-remote-code \
    --max-model-len 262144 \
    --max-num-seqs 4 \
    --max-num-batched-tokens 8192 \
    --gpu-memory-utilization 0.84 \
    --quantization compressed-tensors \
    --attention-backend triton_attn \
    --load-format fastsafetensors \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder \
    --enable-auto-tool-choice \
    --async-scheduling \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
    -tp 1
```

> ⚠️ **`--max-num-batched-tokens 8192` is REQUIRED** — 32768 crashes under concurrency (GDN/Mamba cache alignment). This is the single most important flag.

## What Was Tried and Failed

| Config | Result | Root Cause |
|---|---|---|
| flashinfer + MTP k=7 + Mamba patch | EngineDeadError on 2nd request | Mamba patch corrupts native MTP |
| flashinfer + MTP k=3, no patch | cudaErrorIllegalAddress at c≥5 | FlashInfer GDN MTP crash |
| flashinfer + MTP k=3 + enforce-eager | Stable but ~11 tok/s | No CUDA graphs = slow |
| flashinfer + MTP k=3 + no prefix caching | Still crashes | Prefix caching not the only issue |
| **triton_attn + MTP k=2 + nightly** | **✅ Stable with concurrency** | **triton_attn avoids GDN crash** |

## Variants (alternative configs — NOT sparkrun contract, reference only)

These are alternative serving configs documented for reference. The recipe itself uses the single locked config above (GMU 0.84, 4 seqs, 8192 batch tokens).

| Variant | GMU | Seqs | Description |
|---|---|---|---|
| **Production (recipe default)** | 0.84 | 4 | Mia Lab config — max throughput, MTP k=2 |
| **Concurrent stack** | 0.50 | 8 | Leaves ~80 GB free for ComfyUI + GNOME |
| **erdaltoprak** | 0.50 | 4 | Pinned image SHA `c96082d3...`, `--kv-cache-dtype fp8`, MTP k=2 — GMU 0.50 is a deliberate CUDA-graph headroom choice (his OOM workaround). See `ERDALTOPRAK-QWEN38-CONFIG.md`. |
| **SGLang FP8** | 0.85 | — | `lmsysorg/sglang:latest` + `Qwen/Qwen3.8-27B-FP8`, flashinfer backend. UNVERIFIED on SM121 (cookbook recipe is x86-only). |
| **Radix DSpark** | — | — | Radix + SGLang + DSpark beat NVFP4 MTP single-stream (171 tok/s on 2× PRO 6000). Not yet tested on Spark. |

## Key Differences from Qwen 3.6

| | Qwen 3.6 27B | Qwen 3.8 27B |
|---|---|---|
| Layers | ~48 | 64 |
| MTP | Requires PR #48375 patch | Native (no patch needed) |
| Attention backend | flashinfer | **triton_attn** (flashinfer crashes) |
| enforce-eager | Not needed | Not needed (with triton_attn) |
| vLLM version | v0.24.0+ | nightly-aarch64 (v0.27.2rc1) |
| Container | ghcr.io/styles01/vllm-v26-patched | vllm/vllm-openai:nightly-aarch64 |
| Image | Custom patched | Stock nightly — no patching needed |

## vLLM Announcement Reference

vLLM announced Day-0 support for Qwen 3.8-27B ([post](https://x.com/vllm_project/status/2088287539979559068)):
- MTP draft head included in checkpoint (no separate model)
- ~84.8% acceptance rate (FP8)
- Requires vLLM nightly + transformers 5.8.0+
- Verified on GB300 (BF16/FP8 at TP=4, NVFP4 at TP=1)