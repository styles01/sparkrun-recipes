# MedGemma 27B vLLM Serve Recipe — DGX Spark (GB10, 121GB Unified VRAM)

## Model
- **HF Model ID:** `google/medgemma-27b-text-it`
- **Architecture:** Gemma3 text (`gemma3_text`) — standard transformer with RoPE, GQA, sliding-window attention (1024), tied embeddings
- **Gated:** Yes — requires Hugging Face login + acceptance of Health AI Developer Foundations license
- **Size:** 11 safetensors shards (~54 GB in BF16)

## vLLM Compatibility
- **Status:** Supported in vLLM 0.24.0+ (feature request [#34042](https://github.com/vllm-project/vllm/issues/34042) closed as completed)
- **Model class:** `Gemma3ForCausalLM`
- **Dtype restriction:** vLLM blocks `float16` for `gemma3_text` due to numerical instability; use **`bfloat16`** (or `float32`)

## Quantization Options

| Method | Weight Size | vLLM Flag | Notes |
|--------|-------------|-----------|-------|
| **BF16 (stock)** | ~54 GB | `--dtype bfloat16` | Highest quality, fits comfortably on 121 GB |
| **FP8 dynamic** | ~27 GB | `--quantization fp8 --dtype bfloat16` | Best throughput/quality trade-off on Blackwell (SM100) GB10 |
| **GGUF Q4_K_M** | ~14–17 GB | `--quantization gguf` | Community GGUFs exist; use if DRAM is tight |

- No official FP8/NVFP4 checkpoint from Google exists yet.
- DGX Spark GB10 (Blackwell/SM100) **does support FP8**, so dynamic FP8 quantization via `--quantization fp8` is the recommended path for maximum throughput.

## Prerequisites

```bash
# 1. Log in to Hugging Face and accept the Health AI Developer Foundations license
#    for google/medgemma-27b-text-it
huggingface-cli login

# 2. Stop GNOME to free ~6–8 GB of unified memory
sudo systemctl stop gdm3   # or equivalent on your Spark

# 3. Let FlashInfer autotune once at conservative GMU, then cache it
#    (see serve commands below)
```

## Serve Commands

### Option A — BF16 (Highest Quality, ~40–60 tok/s estimated)
```bash
vllm serve google/medgemma-27b-text-it \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 131072 \
  --tensor-parallel-size 1 \
  --swap-space 4 \
  --enable-chunked-prefill \
  --max-num-seqs 256
```

### Option B — FP8 Dynamic (Recommended, ~80–120 tok/s estimated)
```bash
vllm serve google/medgemma-27b-text-it \
  --dtype bfloat16 \
  --quantization fp8 \
  --gpu-memory-utilization 0.93 \
  --max-model-len 131072 \
  --tensor-parallel-size 1 \
  --swap-space 4 \
  --enable-chunked-prefill \
  --max-num-seqs 512
```

### Option C — GGUF Q4_K_M (Lowest Memory, ~14–17 GB)
Download a community GGUF (e.g. `unsloth/medgemma-27b-text-it-GGUF`) and serve:
```bash
vllm serve unsloth/medgemma-27b-text-it-GGUF \
  --quantization gguf \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 131072 \
  --tensor-parallel-size 1
```

## Recommended Parameters for Spark

| Parameter | Value | Reason |
|-----------|-------|--------|
| `--tensor-parallel-size` | `1` | Spark has only one GPU |
| `--dtype` | `bfloat16` | Required by vLLM for Gemma3 (blocks float16) |
| `--gpu-memory-utilization` | `0.90–0.93` | 121 GB unified; leave ~10 GB for system/FlashInfer overhead |
| `--max-model-len` | `131072` | Gemma3 family supports 128K context |
| `--enable-chunked-prefill` | (default in v1) | Improves throughput on long-context models |
| `--max-num-seqs` | `256–512` | Tune based on expected concurrency |

## FlashInfer Autotuning Caveat
FlashInfer triggers an autotuning pass on first run that can spike memory. On Spark:
1. Run the serve command once with **lower** `--gpu-memory-utilization 0.80`.
2. Let it complete autotuning (it will cache results to `~/.cache/flashinfer/`).
3. Stop the server, then restart at higher GMU (`0.90–0.93`).

## Expected Performance (Estimates)

| Quantization | Context | Throughput Estimate | VRAM Used |
|--------------|---------|---------------------|------------|
| BF16 | 128K | **40–60 tok/s** | ~90–100 GB |
| FP8 dynamic | 128K | **80–120 tok/s** | ~50–60 GB |
| GGUF Q4_K_M | 128K | **60–90 tok/s** | ~25–35 GB |

> These are rough estimates based on Gemma3 27B on Blackwell-class hardware. Actual numbers depend on batch size, prompt length, and whether chunked prefill is active. Benchmark with your workload.

## Known Issues / Caveats
1. **Gated model** — You must authenticate with `huggingface-cli login` and accept the license before downloading weights.
2. **No NVFP4 native checkpoint** — If you want NVFP4, you must quantize the BF16 weights yourself (not trivial for 54 GB).
3. **float16 blocked** — vLLM will raise an error if you try `--dtype float16`; use `bfloat16`.
4. **Unified memory** — Spark uses unified CPU/GPU memory. The GB10 has 121 GB total shared. Running the desktop (GNOME) consumes ~6–8 GB; stopping it frees that memory for the model.
5. **Sliding window** — Gemma3 uses per-layer sliding window (1024 tokens) for local-attention layers; global attention is used for others. This is handled automatically by vLLM’s Gemma3 attention backend.

## Quick Verification
```bash
# Test a simple medical prompt
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/medgemma-27b-text-it",
    "prompt": "A 45-year-old male presents with chest pain. What are the differential diagnoses?",
    "max_tokens": 256,
    "temperature": 0.7
  }'
```
