# Recipe: Qwen 3.8 27B NVFP4 + DSpark (GB10)

**Status:** ⚠️ EXPERIMENTAL — DSpark drafter for ~75 tok/s single-stream
**Served name:** `qwen38-27b`
**Docker image:** `ghcr.io/drowzeys/keys-vllm-027-gb10-qwen38:mtp3-20260813` (drowzeys GB10 build — enables Spark Arena)
**Spec decode:** DSpark drafter (k=14), NOT MTP
**KV cache:** bf16 (REQUIRED — DSpark forces FLASH_ATTN which rejects fp8)

> **Recipe contract:** [`recipes/qwen-38-27b-nvfp4-dspark.yaml`](../recipes/qwen-38-27b-nvfp4-dspark.yaml)
> **Source:** [@0xBakeer tweet](https://x.com/0xBakeer/status/2089090318905774558) — 75 tok/s single-stream on same model/hardware

## CRITICAL: bf16 KV cache (NOT fp8)

**The DSpark drafter forces the FLASH_ATTN attention backend, which REJECTS fp8 KV cache.** The drowzeys image forces fp8 KV by default → crash:
```
ValueError: Selected backend AttentionBackendEnum.FLASH_ATTN is not valid for this configuration. Reason: ['kv_cache_dtype not supported']
```
**Fix:** explicitly set `--kv-cache-dtype bf16` to override the forced fp8. This is the single most important flag for this recipe.

## DSpark Drafter

- **Model:** `Doopeworld/Qwen3.8-27B-DSpark-vLLM` (2.7GB, 1.36B params, 5-layer DFlash + Markov head)
- **Location on Spark:** `~/models/Doopeworld-Qwen3.8-27B-DSpark-vLLM/`
- **k=14** for single-stream latency (75 tok/s); **k=7** for aggregate throughput
- `draft_sample_method: "probabilistic"` — ~23% faster than greedy

## Locked Config (Experimental)

```bash
docker run -d --name qwen38 --gpus all --ipc host --network host \
  -v ~/models/hf/hub/models--unsloth--Qwen3.8-27B-NVFP4:/models \
  -v ~/models/Doopeworld-Qwen3.8-27B-DSpark-vLLM:/models-draft \
  -v ~/vllm-cache:/root/.cache/vllm \
  -e FLASHINFER_CUDA_ARCH_LIST=12.1a -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  ghcr.io/drowzeys/keys-vllm-027-gb10-qwen38:mtp3-20260813 \
  vllm serve /models/snapshots/b0d9f9de93a9e98df9b1dd41ba444ab1139b1ab3 \
    --served-model-name qwen38-27b --host 0.0.0.0 --port 8000 \
    --max-model-len 262144 --kv-cache-dtype bf16 --gpu-memory-utilization 0.55 \
    --max-num-seqs 4 --max-num-batched-tokens 16384 \
    --enable-prefix-caching \
    --enable-flashinfer-autotune --enable-auto-tool-choice --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 \
    --speculative-config '{"method":"dspark","model":"/models-draft","num_speculative_tokens":14,"draft_sample_method":"probabilistic"}'
```

> ⚠️ **`--kv-cache-dtype bf16` is REQUIRED** — DSpark forces FLASH_ATTN which rejects fp8.
> ⚠️ **`--max-num-batched-tokens 16384`** — required for k=14 draft slots (default 2048 too small).
> ⚠️ **`--enable-prefix-caching`** — vLLM silently disables it for this hybrid model; worth 14-22x on shared prefixes.

## Comparison

| Config | Single-stream | Notes |
|---|---|---|
| MTP n=3 (canonical) | ~31.7 tok/s | Validated, fp8 KV |
| **DSpark k=14 (this)** | **~75 tok/s target** | bf16 KV, prefix caching |

## Caveats

- Speculative decoding is output-preserving (full model verifies every draft token).
- Quantization benefit (NVFP4 vs FP8) vanishes under high concurrency (~0% at 16 streams).
- The drowzeys image forces fp8 KV — must override with bf16 for DSpark.
- If DSpark fails on drowzeys, fallback is the official `vllm/vllm-openai:v0.27.1-aarch64` (what the author used).
