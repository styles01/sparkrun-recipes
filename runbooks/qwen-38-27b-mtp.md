# Recipe: Qwen 3.8 27B NVFP4 + MTP n=3 (GB10)

**Status:** ✅ VALIDATED — running on Spark since 2026-08-20
**Served name:** `qwen38-27b`
**Docker image:** `ghcr.io/styles01/qwen38-mtp3:latest` (derived from drowzeys GB10 build)
**Spec decode:** MTP n=3 (multi-token prediction)
**KV cache:** fp8

> **Recipe contract:** [`recipes/qwen-38-27b-nvfp4-mtp.yaml`](../recipes/qwen-38-27b-nvfp4-mtp.yaml)
> **Original source:** [@0xBakeer tweet](https://x.com/0xBakeer/status/2089090318905774558)
> **Forked from:** drowzeys `keys-vllm-027-gb10-qwen38:mtp3-20260813` (repo now gone — image preserved)

## Validated Performance

| Metric | Value |
|---|---|
| Aggregate tok/s | ~96 (decode + prefill) |
| Single-stream tok/s | ~31.7 |
| KV concurrency multiplier | 6.45× |
| GMU | 0.55 |
| Context | 262,144 tokens |
| Lanes (max_num_seqs) | 4 |

## Locked Config (Production)

```bash
docker run -d --name qwen38 --gpus all --ipc host --network host \
  -v ~/models/hf/hub/models--unsloth--Qwen3.8-27B-NVFP4:/models \
  -v ~/vllm-cache:/root/.cache/vllm \
  -e FLASHINFER_CUDA_ARCH_LIST=12.1a \
  -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  ghcr.io/styles01/qwen38-mtp3:latest \
  vllm serve /models/snapshots/b0d9f9de93a9e98df9b1dd41ba444ab1139b1ab3 \
    --served-model-name qwen38-27b --host 0.0.0.0 --port 8000 \
    --max-model-len 262144 --kv-cache-dtype fp8 \
    --gpu-memory-utilization 0.55 --max-num-seqs 4 \
    --max-num-batched-tokens 8192 \
    --enable-prefix-caching --enable-chunked-prefill \
    --enable-flashinfer-autotune --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

> ⚠️ **`--kv-cache-dtype fp8`** — validated on this image. Do NOT use bf16 unless switching to DSpark (which forces FLASH_ATTN).
> ⚠️ **`--enable-prefix-caching`** — 14-22× speedup on shared prefixes (e.g., multi-turn chat with system prompt).
> ⚠️ **`--enable-chunked-prefill`** — required for long context prefill without OOM.

## What Changed from drowzeys Original

| | drowzeys original | Our fork |
|---|---|---|
| MTP n | 3 | 3 (unchanged) |
| KV cache | fp8 | fp8 (unchanged) |
| GMU | 0.55 | 0.55 (unchanged) |
| max_num_batched_tokens | 8192 | 8192 (unchanged) |
| Container | `ghcr.io/drowzeys/keys-vllm-027-gb10-qwen38:mtp3-20260813` | `ghcr.io/styles01/qwen38-mtp3:latest` |
| Status | Repo deleted | Preserved and republished |

## Comparison with Other Flavors

| Config | Single-stream | Aggregate | Notes |
|---|---|---|---|
| **MTP n=3 (this)** | ~31.7 tok/s | ~96 tok/s | **VALIDATED, stable, fp8 KV** |
| DSpark k=14 (experimental) | ~75 tok/s target | unknown | bf16 KV, requires FLASH_ATTN |
| MTP n=12 (experimental) | collapsed | 9% accept | Too aggressive for prose |

## Caveats

- Speculative decoding is output-preserving (full model verifies every draft token).
- NVFP4 quantization benefit vanishes under high concurrency (~0% at 16 streams).
- This is the **canonical daily-driver config** for Qwen 3.8 27B on GB10.
- DSpark variant exists but is experimental — this MTP config is the validated path.

## Historical Note

The original `ghcr.io/drowzeys/keys-vllm-027-gb10-qwen38:mtp3-20260813` image was published by drowzeys (MiaAI Lab) and contained vLLM 0.27.1 with MTP support, FlashInfer MoE backend, and GB10-specific optimizations. When the drowzeys GitHub repo was deleted, this image became the primary source. We have republished it as `ghcr.io/styles01/qwen38-mtp3:latest` for community preservation.
