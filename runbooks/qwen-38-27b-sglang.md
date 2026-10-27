# Recipe: Qwen 3.8 27B NVFP4 on SGLang (GB10)

**Status:** ✅ Production-ready — three engines measured on-device
**Served name:** `qwen3.8-27b-sglang`
**Port:** 8888
**Image:** `lmsysorg/sglang:qwen38-27b` (pinned, ships modelopt + NVFP4 kernels)
**Hardware:** NVIDIA DGX Spark / GB10 (aarch64, SM121, 128 GB unified memory)

> **Recipe contract:** [`recipes/qwen-38-27b-nvfp4-sglang.yaml`](../recipes/qwen-38-27b-nvfp4-sglang.yaml)
> **Source:** [MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark](https://github.com/MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark)
> **Original author:** Mia's AI Lab ([@MiaAI_lab](https://x.com/MiaAI_lab))
> **Model:** [RadixArk/Qwen3.8-27B-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-27B-NVFP4) (~22 GB, NVFP4 W4A4)

## Three Engines — Choose Your Weapon

This repo ships **three swap-in speculative decode engines** on the same base image. All numbers were measured on the same DGX Spark box (same hardware, same model, different days — cross-engine deltas are indicative, not a race).

| Engine | Script | Best for | Code | Short Chat (T1 think) | Long Essay | Aggregate (c=16) |
|---|---|---|---|---|---|---|---|
| **EAGLE/MTP** | `start.sh` | Long-form writing | 34.5 | 21.0 | **24.1** | — |
| **DSpark** | `start-dspark.sh` | Code, agents, tools | **51.5** | 23.2 | 18.3 | — |
| **DFlash2** | `start-dflash.sh` | Code + essay (best overall) | 50.9 | **66.6** | **25.4** | **227.6** |

> ⚠️ DFlash2 short-chat numbers are from `completion_tokens` (server-reported), not event-counting. DFlash2 batches ~3.75 tokens per SSE event — naive event-counting reads ~9 "tok/s" while real is 66.6.

### Quick selection guide

- **Code-heavy agents / tools** → DSpark (51.5 tok/s, block-7 is the peak)
- **Long-form writing / essays** → MTP (24.1 tok/s, holds prose quality)
- **Best of both / mixed workload** → DFlash2 (50.9 code, 25.4 essay, 66.6 chat)
- **Maximum aggregate throughput** → DFlash2 (227.6 tok/s at 16 concurrent, validated)

## CRITICAL: GB10-specific image (NOT stock SGLang)

**Stock `lmsysorg/sglang:latest` does NOT have the GB10 NVFP4 kernels.** The `qwen38-27b` tag is a model-specific build from the SGLang cookbook with modelopt + NVFP4 support for sm_121a baked in.

- Image: `lmsysorg/sglang:qwen38-27b` (multi-arch, includes arm64)
- DFlash2: requires derived image (auto-built from `patch/` on first run — see [DFlash2 section](#dflash2))
- Env: `FLASHINFER_CUDA_ARCH_LIST=12.1a`, `FLASHINFER_DISABLE_VERSION_CHECK=1`
- CPU pinning: ten 3.9 GHz Cortex-X5 cores (`--cpuset-cpus 5-9,15-19`) — measured +2-7% decode vs letting scheduler land on 2.8 GHz A725 efficiency cores

## Model Location on Spark

```
~/models/hf/hub/models--RadixArk--Qwen3.8-27B-NVFP4/
```
~22 GB, NVFP4 W4A4 (NOT the unsloth checkpoint — this is the SGLang-optimized variant).

> ⚠️ **Do NOT use `unsloth/Qwen3.8-27B-NVFP4`** with this recipe — the RadixArk checkpoint has calibrated FP8 KV scales and a quantized lm_head that SGLang expects.

## Locked Config (Production)

### EAGLE/MTP (default)

```bash
docker run -d --name sglang-qwen38 --restart unless-stopped \
  --gpus all --ipc host --network host \
  --shm-size 32g \
  --cpuset-cpus "5-9,15-19" \
  -v ~/models/hf/hub/models--RadixArk--Qwen3.8-27B-NVFP4:/models \
  -v ~/vllm-cache:/root/.cache/vllm \
  -e FLASHINFER_CUDA_ARCH_LIST=12.1a \
  -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  lmsysorg/sglang:qwen38-27b \
  python3 -m sglang.launch_server \
    --model-path /models \
    --served-model-name qwen3.8-27b-sglang \
    --host 0.0.0.0 --port 8888 \
    --speculative-algorithm EAGLE \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --attention-backend flashinfer \
    --kv-cache-dtype fp8_e4m3 \
    --mem-fraction-static 0.90 \
    --context-length 262144 \
    --max-running-requests 10 \
    --mamba-ssm-dtype bfloat16 \
    --chunked-prefill-size 8192 \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder \
    --trust-remote-code
```

### DSpark (code peak)

```bash
docker run -d --name sglang-qwen38 --restart unless-stopped \
  --gpus all --ipc host --network host \
  --shm-size 32g \
  --cpuset-cpus "5-9,15-19" \
  -v ~/models/hf/hub/models--RadixArk--Qwen3.8-27B-NVFP4:/models \
  -v ~/models/hf/hub/models--RadixArk--Qwen3.8-27B-DSpark:/models-draft \
  -v ~/vllm-cache:/root/.cache/vllm \
  -e FLASHINFER_CUDA_ARCH_LIST=12.1a \
  -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  lmsysorg/sglang:qwen38-27b \
  python3 -m sglang.launch_server \
    --model-path /models \
    --served-model-name qwen3.8-27b-sglang \
    --host 0.0.0.0 --port 8888 \
    --speculative-algorithm EAGLE \
    --speculative-draft-model-path /models-draft \
    --speculative-dspark-block-size 7 \
    --speculative-num-draft-tokens 8 \
    --attention-backend flashinfer \
    --kv-cache-dtype fp8_e4m3 \
    --mem-fraction-static 0.90 \
    --context-length 262144 \
    --max-running-requests 10 \
    --mamba-ssm-dtype bfloat16 \
    --chunked-prefill-size 8192 \
    --num-continuous-decode-steps 2 \
    --torch-compile \
    --enable-decode-graph-caps \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder \
    --trust-remote-code
```

> ⚠️ **DSpark draft model:** `RadixArk/Qwen3.8-27B-DSpark` (~2.7 GB). Downloaded automatically on first run if missing.
> ⚠️ **Block-7 is the code peak** — block-5 trades -16% code for +8% prose if you want it.
> ⚠️ **DSpark cannot use YaRN / context > 262K** — the YaRN override leaks into draft config and crashes.

### DFlash2 (code + essay win)

```bash
# NOTE: DFlash2 requires a derived image. Auto-builds from patch/ on first run.
# If you already have lmsysorg/sglang:qwen38-27b-dflash2, use it directly.
# Otherwise, the start-dflash.sh script from the source repo builds it.

# After building the image:
docker run -d --name sglang-qwen38 --restart unless-stopped \
  --gpus all --ipc host --network host \
  --shm-size 32g \
  --cpuset-cpus "5-9,15-19" \
  -v ~/models/hf/hub/models--RadixArk--Qwen3.8-27B-NVFP4:/models \
  -v ~/models/hf/hub/models--z-lab--Qwen3.8-27B-DFlash2:/models-draft-dflash2 \
  -v ~/vllm-cache:/root/.cache/vllm \
  -e FLASHINFER_CUDA_ARCH_LIST=12.1a \
  -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  lmsysorg/sglang:qwen38-27b-dflash2 \
  python3 -m sglang.launch_server \
    --model-path /models \
    --served-model-name qwen3.8-27b-sglang \
    --host 0.0.0.0 --port 8888 \
    --speculative-algorithm DFLASH \
    --speculative-draft-model-path /models-draft-dflash2 \
    --speculative-num-draft-tokens 8 \
    --attention-backend flashinfer \
    --kv-cache-dtype fp8_e4m3 \
    --mem-fraction-static 0.90 \
    --context-length 262144 \
    --max-running-requests 10 \
    --mamba-ssm-dtype bfloat16 \
    --chunked-prefill-size 8192 \
    --torch-compile \
    --enable-decode-graph-caps \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder \
    --trust-remote-code
```

> ⚠️ **DFlash2 draft model:** `z-lab/Qwen3.8-27B-DFlash2` or `incoai/Qwen3.8-27B-DFlash2` (~2.6 GB). Trained against bf16 base; works with NVFP4 target via quantized-head selector.
> ⚠️ **Crash history (fixed):** Old dequant-once head handling materialized dense NVFP4 lm_head (~2.5-5 GB) at graph capture, hard-rebooting the box. Fixed with in-place `lm_head.quant_method.apply`. Use `mem-fraction-static 0.90` (not 0.95).
> ⚠️ **DFlash2 SSE streams batch ~3.75 tokens/event** — always count `completion_tokens`, never raw SSE events.

## Validated Performance (Measured on this box)

### Per-engine, per-workload

| Probe | DSpark (block-7) | MTP (EAGLE 3/1/4) | DFlash2 (NVFP4) |
|---|---|---|---|
| Code — LRUCache + small test | **51.5 tok/s** | 34.5 tok/s | 50.9 tok/s |
| Short chat — T=1 thinking ON | 23.2 tok/s | 21.0 tok/s | **66.6 tok/s** |
| Long essay — Babbage → GPUs | 18.3 tok/s | **24.1 tok/s** | **25.4 tok/s** |

### DFlash2 concurrency ladder (2026-08-19, post-fix boot)

| Streams | TTFT | Aggregate tok/s | Per-stream tok/s |
|---|---|---|---|
| 1 | 127 ms | 56.6 | 56.6 |
| 2 | 202 ms | 58.4 | 42.4 |
| 4 | 224 ms | 111.6 | 33.4 |
| 8 | 280 ms | 184.9 | 30.8 |
| 16 | 4.18 s | **227.6** | 28.2 |

> One boot, one fixture — indicative, not a guarantee. Replicate before relying on it. Per-stream degrades gracefully (56.6 → 28.2 tok/s); aggregate scales to 227.6 at 16 concurrent.

### Wall-time benchmarks (includes prefill, different clock)

| Engine | Thinking | Non-thinking | Tool-call | TTFT (16K warm) |
|---|---|---|---|---|
| MTP | 17.2–20.5 tok/s | 21.6–22.7 | 26–28 | ~8.3 s |
| DFlash2 | 18.8–19.7 tok/s | 22.8–24.1 | 27.3–30.8 | ~8.2 s |

## Long Context & Concurrency (up to 1M, 10 concurrent)

| You want | `YARN` | `CONTEXT_LENGTH` | `MAX_CONCURRENT_REQUESTS` | YaRN factor |
|---|---|---|---|---|
| 1M + 10 concurrent | 1 | 1000000 | 10 | 4.0 |
| 512K + 10 concurrent | 1 | 524288 | 10 | 2.0 |
| 768K + 10 concurrent | 1 | 786432 | 10 | 3.0 |
| Native 262K + 10 concurrent | 0 | 262144 | 10 | — |
| 1M + 2 concurrent | 1 | 1000000 | 2 | 4.0 |

- **YaRN only works with MTP** — DSpark and DFlash2 crash with `AttributeError: 'PreTrainedConfig' object has no attribute 'max_position_embeddings'`
- **Above 262K you must set `YARN=1`** (1M auto-enables even with `YARN=0`)
- **Hardware bound:** one KV token ≈ 32.8 KB, 1M sequence ≈ 33 GB, pool ≈ 75 GB → ~2 full 1M requests run simultaneously regardless of `MAX_CONCURRENT_REQUESTS`

## Key Settings (measured, not guessed)

| Setting | Value | Why |
|---|---|---|
| Mem fraction | 0.90 (DSpark/DFlash2), 0.95 (MTP) | 0.95 + DFlash2 wedged the box once; 0.90 validated safe |
| Chunked prefill | 8192 | cookbook default 2048 was suboptimal |
| GDN dtype | bfloat16 | cookbook float32 was -3% |
| Mamba pool | concurrency × 4 slots | `extra_buffer_lazy` + overlap scheduler; verified in build's `kv_cache_configurator` |
| CPU pinning | `5-9,15-19` | X5 cores only; A725 cores cost 2-7% decode |
| Torch compile | ON (DSpark/DFlash2 only) | +decode graphs; prefill graphs OFF |
| Attention backend | flashinfer | triton fallback if spec decode errors at boot |

## Reasoning / Thinking (IMPORTANT)

Qwen 3.8 is a **native thinking model** — thinking is ON by default. Do NOT disable it.

- `--reasoning-parser qwen3` surfaces `thinking…` as `reasoning_content` (not inline)
- Per-request control: `chat_template_kwargs.reasoning_effort=xhigh|medium|low` (xhigh default)
- Sampling defaults from checkpoint: `temperature=1.0, top_p=0.95, top_k=20`
- For agent frameworks (Hermes/Loca), set `stale_timeout_seconds` high (e.g., 600)

## Tool Calling

SGLang needs no extra flag (unlike vLLM's `--enable-auto-tool-choice`):
- `--tool-call-parser qwen3_coder` decodes `<tool_call><function=…>` payload
- Just send `tools` in the request — no additional server flags
- The `hermes` parser expects a different payload and will not parse

## API Usage

OpenAI-compatible: `http://127.0.0.1:8888/v1` (model: `qwen3.8-27b-sglang`)
Anthropic-compatible: `http://127.0.0.1:8888/v1/messages` (Claude Code: `ANTHROPIC_BASE_URL=http://127.0.0.1:8888`)

## Comparison with vLLM Recipe

| | vLLM (qwen-38-27b.yaml) | SGLang (this) |
|---|---|---|
| Image | `ghcr.io/styles01/qwen38-mtp3:latest` | `lmsysorg/sglang:qwen38-27b` |
| Engine | vLLM 0.27.1 | SGLang (cookbook build) |
| Model | `unsloth/Qwen3.8-27B-NVFP4` | `RadixArk/Qwen3.8-27B-NVFP4` |
| Spec decode | MTP n=3 only | EAGLE/MTP, DSpark, DFlash2 |
| Single-stream | ~31.7 tok/s | 34.5–51.5 (engine-dependent) |
| Aggregate | ~96 tok/s (c=4) | 227.6 (DFlash2, c=16) |
| Tool calling | `--enable-auto-tool-choice` | Built-in (no flag) |
| Port | 8000 | 8888 |
| Daily driver | ✅ Yes (validated, stable) | 🧪 Yes (measured, more options) |

## Credits

- **Mia's AI Lab** ([@MiaAI_lab](https://x.com/MiaAI_lab)) — original repo, all measurements, three-engine comparison
- **SGLang cookbook** — [Qwen3.8-27B DGX Spark recipe](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-27B)
- **RadixArk** — NVFP4 checkpoint + DSpark draft model
- **z-lab / incoai** — DFlash2 block-diffusion drafter
- **hasso5703** — DSpark-on-GB10 config foundation
