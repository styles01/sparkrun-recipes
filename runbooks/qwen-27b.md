# Recipe: Qwen 3.8 27B FP8

**Status:** ✅ Production (enforce-eager + MTP k=3, stable)
**Served name:** `qwen3.8-27b`
**Docker image:** `vllm/vllm-openai:v0.27.1` (stock — also tagged `latest`)
**Tool calling:** ✅ `--enable-auto-tool-choice --tool-call-parser qwen3_coder`
**Reasoning parser:** `qwen3`
**Vision:** ✅ Multimodal (image_token_id present, vision_config)

> **Recipe contract:** [`recipes/qwen-27b.yaml`](../recipes/qwen-27b.yaml)

## Model Location on Spark

```
~/models/hf/hub/models--Qwen--Qwen3.8-27B-FP8/snapshots/017b9c7af6b5689d5dd426a76e0bc077eb5ca20a/
```
~29 GB, 3 safetensors shards. MTP draft head built into checkpoint (no separate repo).

## Model Specs

| Spec | Value |
|---|---|
| Architecture | `Qwen3_5ForConditionalGeneration` (hybrid Mamba/Gated DeltaNet + full attention) |
| Layers | 64 (48 linear attention + 16 full attention, interval 4) |
| Hidden | 5120 |
| Heads | 24, KV heads 4, head_dim 256 |
| Context | 262,144 (256K), extendable to 1M via YaRN |
| Vocab | 248,320 |
| Quant | FP8 (compressed-tensors) |
| Size | ~29 GB safetensors |
| MTP | Built-in (mtp_num_hidden_layers=1, mtp_use_dedicated_embeddings=false) |
| Acceptance | ~84.8% (FP8, per vLLM announcement) |

## CRITICAL: Disable Prefix Caching + 2048 Batch Tokens

**MTP crashes at concurrency >= 5 when prefix caching is enabled.**

The EAGLE/MTP peek-and-drop path in `kv_cache_coordinator.py` overreads Mamba/GDN recurrent state groups because `MambaManager.find_longest_cache_hit` never reads `drop_eagle_block` (GitHub #50630, PR #47861 unmerged). With prefix caching OFF, the peek-and-drop path is never invoked, avoiding the `cudaErrorIllegalAddress`.

Also: `--max-num-batched-tokens 2048` (not 32768) — GDN/Mamba cache alignment constraint. Default 8192 is too large and triggers illegal memory access under concurrency.

- **GitHub #37754:** FlashInfer + MTP crashes on SM121 (DGX Spark) — confirmed fix: disable prefix caching
- **vLLM Recipes Qwen3.5.md:** "Enable MTP-1 speculative decoding and disable prefix caching"
- **StepCodex:** "Disabling prefix caching also works (~31 tok/s). The bug only manifests when prefix caching=on"
- **PR #47861:** Full fix (MambaManager capability check) — still unmerged

## CRITICAL: enforce-eager Required

**CUDA graph capture crashes MTP on Qwen 3.8's hybrid Mamba architecture.**

The `cudaErrorIllegalAddress` error occurs when CUDA graphs replay MTP draft tokens through the Gated DeltaNet (linear attention) layers. The recurrent state cache layout doesn't match the graph's captured memory addresses.

- **GitHub issue #38643:** FLA linear attention tensor format mismatch
- **PR #34571:** CUDA graph capture on hybrid models (not yet merged)
- **Fix:** `--enforce-eager` disables CUDA graphs entirely. MTP works perfectly without graphs.
- **Tradeoff:** ~10-20% throughput cost vs CUDA graphs, but MTP gives 3x+ speedup. Net positive.

## CRITICAL: NO Mamba Patch

**Do NOT apply the PR #48375 Mamba patch (`patch_mamba_drop_eagle.sh`) to Qwen 3.8.**

- The patch was designed for Qwen 3.6's code layout (~48 layers)
- Qwen 3.8 has 64 layers with a different MTP implementation
- vLLM v0.27.1 has **native MTP support** for Qwen 3.8 — it auto-detects the MTP model in the checkpoint
- vLLM logs: *"Detected MTP model. Sharing target model embedding/lm_head weights with the draft model."*
- Applying the patch corrupts the native MTP support and causes `EngineDeadError`

## Locked Config (Production)

```bash
docker run -d --name qwen38-spark \
  --gpus all --network host --ipc host --shm-size 32gb \
  --entrypoint "" \
  -v ~/models/hf:/cache/huggingface \
  vllm/vllm-openai:v0.27.1 \
  vllm serve /cache/huggingface/hub/models--Qwen--Qwen3.8-27B-FP8/snapshots/017b9c7af6b5689d5dd426a76e0bc077eb5ca20a \
    --served-model-name qwen3.8-27b \
    --host 0.0.0.0 --port 8000 \
    --trust-remote-code \
    --max-model-len 262144 \
    --max-num-seqs 5 \
    --max-num-batched-tokens 32768 \
    --gpu-memory-utilization 0.79 \
    --kv-cache-dtype fp8 \
    --attention-backend flashinfer \
    --moe-backend marlin \
    --load-format fastsafetensors \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder \
    --enable-auto-tool-choice \
    --async-scheduling \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --enforce-eager \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
    -tp 1
```

## Variants

| Variant | GMU | Seqs | Description |
|---|---|---|---|
| `recipe_production` | 0.79 | 5 | Default — max throughput, headless |
| `recipe_concurrent` | 0.50 | 8 | Leaves ~80 GB free for ComfyUI + GNOME |

## Troubleshooting

### EngineDeadError with MTP
- **Cause:** CUDA graph capture corrupts MTP on hybrid Mamba architecture
- **Fix:** Add `--enforce-eager` (already in recipe)
- **DO NOT apply Mamba patch** — it makes the crash worse

### cudaErrorIllegalAddress
- **Cause:** CUDA graph replay hits invalid memory in linear attention layers
- **Fix:** `--enforce-eager` (disables all CUDA graphs)

### Engine dies under sustained load (no MTP)
- **Cause:** Same CUDA graph issue, triggered by heavy concurrent load
- **Fix:** `--enforce-eager` (MTP + eager mode is stable)

### Slow generation (~20 tok/s without MTP)
- **Cause:** No speculative decoding
- **Fix:** Enable MTP k=3 with `--enforce-eager`. Expected ~60-80 tok/s with MTP.

## Key Differences from Qwen 3.6

| | Qwen 3.6 27B | Qwen 3.8 27B |
|---|---|---|
| Layers | ~48 | 64 |
| MTP | Requires PR #48375 patch | Native (no patch needed) |
| CUDA graphs | ✅ Works with MTP | ❌ Crashes with MTP |
| enforce-eager | Not needed | **Required** |
| vLLM version | v0.24.0+ | v0.27.1 (latest) |
| Container | ghcr.io/styles01/vllm-v26-patched | vllm/vllm-openai:v0.27.1 (stock) |
| Image | Custom patched | Stock — no patching needed |

## vLLM Announcement Reference

vLLM announced Day-0 support for Qwen 3.8-27B ([post](https://x.com/vllm_project/status/2088287539979559068)):
- MTP draft head included in checkpoint (no separate model)
- ~84.8% acceptance rate (FP8)
- Requires vLLM nightly + transformers 5.8.0+
- Verified on GB300 (BF16/FP8 at TP=4, NVFP4 at TP=1)