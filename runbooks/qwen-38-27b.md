# Recipe: Qwen 3.8 27B NVFP4 (GB10)

**Status:** ✅ Production (drowzeys GB10 build + MTP n=3, flashinfer autotune)
**Served name:** `qwen38-27b`
**Docker image:** `ghcr.io/drowzeys/keys-vllm-027-gb10-qwen38:mtp3-20260813` (eugr spark-vllm-b12x GB10 build)
**Tool calling:** ✅ `--enable-auto-tool-choice --tool-call-parser qwen3_coder`
**Reasoning parser:** `qwen3`
**Spec decode:** MTP n=3 (built-in draft head)

> **Recipe contract:** [`recipes/qwen-38-27b-nvfp4-gb10.yaml`](../recipes/qwen-38-27b-nvfp4-gb10.yaml)
> **Source:** [drowzeys/keys-vLLm.0.27-Qwen3.8-NVFP4-MTP3-Single-DGX-Spark](https://github.com/drowzeys/keys-vLLm.0.27-Qwen3.8-NVFP4-MTP3-Single-DGX-Spark)

## CRITICAL: GB10-specific image (NOT stock vLLM)

**Stock `vllm/vllm-openai` has NO NVFP4 kernels for Blackwell sm_121a (GB10).** Every stock-vLLM attempt crashed. The drowzeys/eugr build provides `FlashInferCutlassNvFp4LinearKernel` for NVFP4 GEMM on GB10.

- Image: `ghcr.io/drowzeys/keys-vllm-027-gb10-qwen38:mtp3-20260813`
- Fallback: `eugr/spark-vllm-b12x:nightly-20260813`
- Env: `FLASHINFER_CUDA_ARCH_LIST=12.1a`, `FLASHINFER_DISABLE_VERSION_CHECK=1`, `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`

## Model Location on Spark

```
~/models-local-qwen38/Qwen3.8-27B-NVFP4/
```
~22 GB, NVFP4 (compressed-tensors). MTP draft head built into checkpoint (no separate repo).

> ⚠️ **Do NOT use `Qwen/Qwen3.8-27B-FP8`** — crashes with stock vLLM (no GB10 NVFP4 kernels). Use the NVFP4 checkpoint.

## Model Specs

| Spec | Value |
|---|---|
| Architecture | `Qwen3_5ForConditionalGeneration` (hybrid Mamba/Gated DeltaNet + full attention) |
| Layers | 64 (48 linear attention + 16 full attention, interval 4) |
| Hidden | 5120 |
| Heads | 24, KV heads 4, head_dim 256 |
| Context | 262,144 (256K), extendable to 1M via YaRN |
| Quant | NVFP4 (compressed-tensors) |
| Size | ~22 GB safetensors |
| MTP | Built-in (mtp_num_hidden_layers=1) |
| Acceptance | ~43-85% (varies by load) |

## Locked Config (Production)

```bash
docker run -d --name qwen38 --gpus all --ipc=host --network host \
  -v ~/models-local-qwen38:/models \
  -e FLASHINFER_CUDA_ARCH_LIST=12.1a \
  -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  ghcr.io/drowzeys/keys-vllm-027-gb10-qwen38:mtp3-20260813 \
  vllm serve /models/Qwen3.8-27B-NVFP4 --served-model-name qwen38-27b \
    --host 0.0.0.0 --port 8000 \
    --max-model-len 262144 --kv-cache-dtype fp8 --gpu-memory-utilization 0.90 \
    --max-num-seqs 4 \
    --reasoning-parser qwen3 \
    --enable-flashinfer-autotune --enable-auto-tool-choice --tool-call-parser qwen3_coder \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

> ⚠️ **MTP n is capped at 3** (model has exactly 1 MTP layer). Depth >=4 crashes.
> ⚠️ **`--tool-call-parser qwen3_coder` is REQUIRED** — Qwen 3.8 emits XML-formatted tool calls; `hermes` parser can't parse them (returns no tool_calls). This was a key fix.
> ⚠️ **`--reasoning-parser qwen3` is REQUIRED** — splits thinking into a separate `reasoning` field so content stays clean.

## Reasoning / Thinking (IMPORTANT)

Qwen 3.8 is a **native thinking model** — thinking is ON by default and is the point (it makes the model smarter). **Do NOT disable thinking** (`enable_thinking: false`). If responses are slow because the model over-thinks, **raise the client timeout** — not the thinking.

- Reasoning effort is controlled per-request via `chat_template_kwargs.reasoning_effort` (values: `xhigh`/`medium`/`low`/`none`). Default in the chat template is `xhigh`.
- For agent frameworks (Hermes/Loca), set `stale_timeout_seconds` high (e.g. 600) so non-streaming subagent calls don't time out during long thinking.

## What Was Tried and Failed

| Config | Result | Root Cause |
|---|---|---|
| stock vLLM + NVFP4 | Crash | No GB10 NVFP4 kernels |
| flashinfer + MTP k=7 + Mamba patch | EngineDeadError | Mamba patch corrupts native MTP |
| flashinfer + MTP k=3, stock | cudaErrorIllegalAddress at c≥5 | FlashInfer GDN MTP crash (stock) |
| **drowzeys GB10 + MTP n=3** | **✅ Stable, 31.7 tok/s single-stream** | **GB10 NVFP4 kernels + native MTP** |

## Key Differences from Qwen 3.6

| | Qwen 3.6 27B | Qwen 3.8 27B |
|---|---|---|
| Layers | ~48 | 64 |
| MTP | Requires PR #48375 patch | Native (no patch needed) |
| Attention backend | flashinfer | flashinfer (GB10 build) |
| vLLM version | v0.24.0+ | GB10 build (v0.27.2rc1.dev88) |
| Container | ghcr.io/styles01/vllm-v26-patched | ghcr.io/drowzeys/keys-vllm-027-gb10-qwen38 |
| Image | Custom patched | GB10-specific (NVFP4 kernels) |
