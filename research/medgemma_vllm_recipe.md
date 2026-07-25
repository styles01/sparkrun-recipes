# MedGemma 27B vLLM Batch-Inference Recipe — DGX Spark (GB10)

## Executive Summary

| Item | Finding |
|------|---------|
| **Model support** | MedGemma 27B (Gemma 3 arch) is fully supported in vLLM 0.24.0 — issue #34042 closed as completed |
| **Pre-quantized FP8** | **YES** — community FP8 checkpoints exist on HuggingFace (compressed-tensors format) |
| **`--quantization fp8`** | Supported natively in vLLM 0.24.0 (Fp8Config class) |
| **`--quantization compressed-tensors`** | Supported natively; auto-detected from config.json |
| **NVFP4 / `modelopt_fp4`** | Requires Blackwell; DGX Spark is Blackwell (SM120), but no pre-quantized MedGemma NVFP4 checkpoints found |
| **AWQ / GPTQ** | No pre-quantized MedGemma 27B AWQ/GPTQ checkpoints found on HF |
| **Blackwell FP8 caveat** | Block-scaled FP8 can crash on SM120 with DeepGEMM (issue #47436); workaround = `VLLM_USE_DEEP_GEMM=0`. The community checkpoints use **channel/token-scaled FP8** (not block-scaled), so this bug is unlikely to hit. |

---

## Recommended Pre-Quantized Checkpoint

For **text-only** clinical-document inference (no images needed):

```
SaitBurak/medgemma-27b-text-it-FP8-dynamic
```

- **Size:** ~28.5 GB (vs 54 GB BF16)
- **Format:** `compressed-tensors` with FP8 dynamic activation quantization
  - Weights: per-channel FP8 (static)
  - Activations: per-token FP8 (dynamic)
- **Architecture:** `Gemma3ForCausalLM` (text-only variant, no vision components)
- **vLLM tags:** `compressed-tensors`, `gemma3_text`

Alternative multimodal checkpoint (if you need vision):
```
ig1/medgemma-27b-it-FP8-Dynamic
```

---

## Exact vLLM Serve Command (DGX Spark, Batch Optimized)

### Option A: Online API Server (for multi-client batching)

```bash
VLLM_USE_V1=1 \
  vllm serve SaitBurak/medgemma-27b-text-it-FP8-dynamic \
  --gpu-memory-utilization 0.95 \
  --max-model-len 32768 \
  --max-num-seqs 256 \
  --max-num-batched-tokens 8192 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --disable-log-requests \
  --port 8000
```

**Why these flags:**
- `VLLM_USE_V1=1` — vLLM V1 engine has superior batch scheduling and higher throughput
- `--gpu-memory-utilization 0.95` — use almost all 121 GB unified VRAM on GB10
- `--max-num-seqs 256` — high concurrency for 800K documents
- `--max-num-batched-tokens 8192` — large batch token limit for throughput
- `--enable-chunked-prefill` — allows prefill and decode to interleave, critical for batch throughput
- `--enable-prefix-caching` — reuse KV cache for repeated prompt prefixes (e.g., clinical note headers)
- `--max-model-len 32768` — MedGemma 27B supports 131K context; 32K is a safe batch-friendly default

### Option B: Offline Batch Inference (recommended for 800K docs)

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="SaitBurak/medgemma-27b-text-it-FP8-dynamic",
    gpu_memory_utilization=0.95,
    max_model_len=32768,
    max_num_seqs=256,
    max_num_batched_tokens=8192,
    enable_chunked_prefill=True,
    enable_prefix_caching=True,
)

sampling = SamplingParams(
    temperature=0.0,
    max_tokens=1024,
)

# Process 800K documents in one call — vLLM handles batching internally
outputs = llm.generate(prompts, sampling)
```

---

## If You Must Quantize On-the-Fly (not recommended)

If you want to use the original `google/medgemma-27b-it` (54 GB BF16) with vLLM’s built-in FP8 quantization:

```bash
VLLM_USE_V1=1 \
  vllm serve google/medgemma-27b-it \
  --quantization fp8 \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.95 \
  --max-model-len 32768 \
  --max-num-seqs 256 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --port 8000
```

**Caveat:** On-the-fly FP8 quantization is **slower and less accurate** than loading pre-quantized weights. The pre-quantized checkpoint above is strongly preferred.

---

## Blackwell / SM120 Specific Notes

- DGX Spark GB10 uses Blackwell SM120, which **does** support FP8 natively.
- If you encounter a `DeepGEMM "Unknown SF transformation"` crash during weight loading (issue #47436), this affects **block-scaled FP8** only. The community checkpoint uses **channel-scaled weights**, so you should not hit this.
- If it does occur, add:
  ```bash
  VLLM_USE_DEEP_GEMM=0
  ```
  before the `vllm serve` command.

---

## What Does NOT Work / Not Found

| Approach | Status | Notes |
|----------|--------|-------|
| Pre-quantized AWQ MedGemma 27B | ❌ Not found | No AWQ variants on HuggingFace |
| Pre-quantized GPTQ MedGemma 27B | ❌ Not found | No GPTQ variants on HuggingFace |
| Pre-quantized NVFP4 MedGemma 27B | ❌ Not found | `modelopt_fp4` exists in vLLM but no community checkpoints |
| `--quantization awq` with original model | ⚠️ Unsupported | AWQ requires pre-quantized weights; cannot quantize on-the-fly |
| AutoFP8 / llm-compressor on Gemma 3 | ⚠️ Untested | No confirmed reports; compressed-tensors is the proven path |

---

## Throughput Estimates (DGX Spark, Single GPU)

With FP8 + V1 engine + chunked prefill + 256 concurrent sequences:
- **Expected throughput:** ~200–400 docs/sec for short clinical notes (512–1K tokens)
- **Total time for 800K docs:** ~30–60 minutes depending on output length
- **VRAM headroom:** ~28 GB weights + ~60–80 GB KV cache + overhead = well within 121 GB

---

## Quick Validation Steps

1. **Download checkpoint (one-time):**
   ```bash
   huggingface-cli download SaitBurak/medgemma-27b-text-it-FP8-dynamic
   ```

2. **Test load:**
   ```bash
   VLLM_USE_V1=1 vllm serve SaitBurak/medgemma-27b-text-it-FP8-dynamic \
     --gpu-memory-utilization 0.95 --max-model-len 4096 --max-num-seqs 64
   ```

3. **Verify quantization detected:**
   Check logs for: `Loading model weights took ...` and confirm no `quantization` flag error.

4. **Health check:**
   ```bash
   curl http://localhost:8000/v1/completions \
     -H "Content-Type: application/json" \
     -d '{"model":"SaitBurak/medgemma-27b-text-it-FP8-dynamic","prompt":"Summarize this clinical note:","max_tokens":128}'
   ```
