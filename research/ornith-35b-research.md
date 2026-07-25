# Ornith-1.0-35B Research — DGX Spark Deployment Feasibility

**Date:** 2026-07-24
**Researcher:** Oracle (subagent)
**Target:** `azampatti/Ornith-1.0-35B-int4-AutoRound-SAR` on HuggingFace

---

## 1. What Is Ornith-1.0-35B?

**Ornith-1.0** is a family of open-source **agentic coding** models from **DeepReinforce AI** (HuggingFace: `deepreinforce-ai`). Released in 2025, the family includes:
- 9B Dense
- 31B Dense
- **35B MoE** ← target of this research
- 397B MoE

Key innovation: **Self-Improving Training Framework** — uses RL to jointly optimize the scaffold (search trajectory) and solution rollouts, discovering better coding agent trajectories.

**Ornith-1.0-35B** is the lightweight MoE member designed for single-GPU deployment. It is a **reasoning model** (opens with `<think>...</think>` before answering) and supports **tool calling** and **extended thinking**.

### Benchmarks (Ornith-1.0-35B vs competitors)
| Benchmark | Ornith-35B | Qwen3.5-35B | Qwen3.6-35B | Gemma4-31B | Qwen3.5-397B |
|---|---|---|---|---|---|
| Terminal-Bench 2.1 (Terminus-2) | **64.2** | 41.4 | 52.5 | 42.1 | 53.5 |
| SWE-bench Verified | **75.6** | 70 | 73.4 | 52 | 76.4 |
| SWE-bench Pro | **50.4** | 44.6 | 49.5 | 35.7 | 51.6 |
| NL2Repo | **34.6** | 20.5 | 29.4 | 15.5 | 36.8 |
| ClawEval Avg | **69.8** | 65.4 | 68.7 | 48.5 | 70.7 |

Ornith-35B **beats Qwen3.5-397B** on Terminal-Bench and matches it on SWE-bench, despite being ~11x smaller.

### License: **MIT** (globally accessible, no regional restrictions)

---

## 2. Architecture

| Property | Value |
|---|---|
| **Base architecture** | Qwen3.5-MoE (post-trained on top of Qwen 3.5) |
| **HF model_type** | `qwen3_5_moe` |
| **HF architectures** | `Qwen3_5MoeForConditionalGeneration` |
| **Total parameters** | ~35B |
| **Active parameters per token** | ~3B (A3B = 3B Active) |
| **MoE or Dense** | **MoE** — 256 experts, 8 experts per token + 1 shared expert |
| **Hidden size** | 2,048 |
| **Num layers** | 40 |
| **Attention heads** | 16 |
| **KV heads** | 2 |
| **Head dim** | 256 |
| **Vocab size** | 248,320 |
| **Multimodal** | Yes — includes vision config (Qwen3.5-MoE-Vision), image/video token support |
| **Attention pattern** | Hybrid: full attention every 4th layer, linear attention otherwise (30/40 layers are linear attention) |
| **Linear attention config** | linear_key_head_dim=128, linear_num_key_heads=16, linear_num_value_heads=32, linear_value_head_dim=128, linear_conv_kernel_dim=4 |
| **RoPE** | mrope (multimodal RoPE), partial_rotary_factor=0.25, rope_theta=10000000, mrope_section=[11,11,10] |
| **MTP heads** | Yes — `mtp_num_hidden_layers: 1`, `mtp_use_dedicated_embeddings: false` |
| **Shared expert** | shared_expert_intermediate_size=512, shared expert gates preserved in FP16 in quantized version |
| **Context length** | **262,144 tokens** (max_position_embeddings=262144) |

---

## 3. Quantization (int4 AutoRound)

The target model `azampatti/Ornith-1.0-35B-int4-AutoRound-SAR` uses:

| Property | Value |
|---|---|
| **Quant method** | `auto-round` (Intel AutoRound) |
| **AutoRound version** | 0.14.3 |
| **Bits** | 4 (W4A16 — 4-bit weights, 16-bit activations) |
| **Group size** | 128 |
| **Symmetric** | Yes |
| **Packing format** | `auto_round:auto_gptq` |
| **Calibration dataset** | NVIDIA OpenCodeInstruct |
| **Calibration settings** | batch_size=8, iters=1000, nsamples=512, seqlen=2048 |
| **Quantization tool** | [spark-auto-round](https://github.com/whpthomas/spark-auto-round) v0.14.3 (GB10-optimized wrapper) |
| **Model size (quantized)** | ~20 GB |
| **Block quantized** | `model.language_model.layers` |

### Critical detail: Shared expert gates kept in FP16
All 40 layers' `shared_expert_gate` modules are explicitly kept at **16-bit float** (not quantized) to preserve MoE routing accuracy. The MTP head is also kept at 16-bit float.

### Quantization Quality
| Metric | Value |
|---|---|
| Peak RAM (during quantization) | 111.31 GB |
| Peak VRAM | 28.00 GB |
| Layers Passed | 22/40 (55%) |
| Layers Warning | 18/40 (45%) — cosine similarity 0.986-0.990 for layers 22-39 |

---

## 4. Speculative Decoding Support

**Yes — MTP (Multi-Token Prediction) is supported.**

- The base model has `mtp_num_hidden_layers: 1` in config.json
- The azampatti quantized variant explicitly states **"Includes MTP Support"** and **"MTP Heads Available"**
- The MTP head is preserved in FP16 in the quantized version (not quantized)
- The model card provides vLLM `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'`

### Performance with MTP
| Metric | Value |
|---|---|
| Throughput (no MTP) | ~60-65 t/s |
| **Throughput (with MTP)** | **~80-91 t/s** (~35-40% speedup) |
| Latency (TTFT) | ~100-200ms |

This is **not** DFlash or EAGLE — it's native MTP speculative decoding built into the model architecture.

---

## 5. Context Length

**262,144 tokens** (256K context)

The azampatti variant's model card recommends:
- Basic serving: `--max-model-len 196608` (192K — conservative)
- Production serving: `--max-model-len 262144` (full 256K)

---

## 6. Existing sparkrun / Community Recipes

### sparkrun
**No sparkrun recipe exists for Ornith-1.0-35B.** Search for "ornith" returned zero results.

However, closely related recipes exist for the **Qwen3.5/3.6-35B-A3B** architecture (same base model family):

| Recipe | Runtime | TP | Model | Notes |
|---|---|---|---|---|
| `@official/qwen3.6-35b-a3b-fp8-mtp-vllm` | vllm-distributed | 1 | Qwen/Qwen3.6-35B-A3B-FP8 | **MTP enabled**, closest analog |
| `@official/qwen3.6-35b-a3b-fp8-vllm` | vllm-distributed | 1 | Qwen/Qwen3.6-35B-A3B-FP8 | No MTP |
| `@sparkrun-transitional/qwen3.5-35b-a3b-bf16-sglang` | sglang | 1 | Qwen/Qwen3.5-35B-A3B | BF16, sglang |
| `@sparkrun-transitional/qwen3.5-35b-a3b-fp8-sglang` | sglang | 1 | Qwen/Qwen3.5-35B-A3B-FP8 | FP8, sglang |
| `@eugr/qwen3.5-35b-a3b-fp8` | vllm-ray | 2 | Qwen/Qwen3.5-35B-A3B-FP8 | 2-node |
| `@eugr/qwen3.6-35b-a3b-fp8-dflash` | vllm-ray | 2 | Qwen/Qwen3.6-35B-A3B-FP8 | DFlash spec decode |
| `@eugr/qwen3.6-35b-a3b-nvfp4` | vllm-ray | 2 | nvidia/Qwen3.6-35B-A3B-NVFP4 | NVFP4, 0.4 GPU mem |

### GitHub Community Recipes

1. **[Bitbull-Ideas/vllm.ornith-1.0-35b_spark](https://github.com/Bitbull-Ideas/vllm.ornith-1.0-35b_spark)**
   - Stars: 0 (very new)
   - FP8 variant (not int4) using `deepreinforce-ai/Ornith-1.0-35B-FP8`
   - Validated on real DGX Spark with vLLM 0.24.0
   - Uses `--language-model-only`, `--enforce-eager`, GPU_MEM_UTIL=0.5
   - 262K context, MAX_NUM_SEQS=3
   - Systemd autostart setup included
   - Tool calling: `qwen3_xml` parser
   - **Key finding**: GGUF path via vllm-gguf-plugin was rejected — Qwen3.5-MoE incompatibilities

2. **[picopapaya/sakamakismile-ornith1.0-35b-nvfp4-vllm](https://github.com/picopapaya/sakamakismile-ornith1.0-35b-nvfp4-vllm)**
   - Community NVFP4 requantization, Docker-based
   - **Critical warning**: Concurrency-4 with long generations **wedges the server** (throughput collapses to 0 tok/s, server unresponsive, health check stays "healthy")
   - Decode speed: 61.9 tok/s @ concurrency 1, 108.2 tok/s aggregate @ concurrency 2
   - GPU_MEM_UTIL=0.5 fails to start if other containers are running (need 0.4)
   - ~10 min first-boot startup (weight download + JIT compilation for SM_121a)

3. **[picopapaya/sakamakismile-ornith1.0-35b-nvfp4-sglang](https://github.com/picopapaya/sakamakismile-ornith1.0-35b-nvfp4-sglang)**
   - Same NVFP4 weights, SGLang engine
   - ~59 tok/s single-request decode

4. **[whpthomas/Ornith-1.0-35B-int4-AutoRound](https://huggingface.co/whpthomas/Ornith-1.0-35B-int4-AutoRound)** (HuggingFace)
   - The "original" int4 AutoRound quant by the spark-auto-round author
   - 1,192 downloads
   - Same settings as azampatti's variant but **no MTP support** documented and shorter context (196,608 vs 262,144)

5. **[eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker)**
   - The community Docker setup for vLLM on DGX Spark (single or multi-node)
   - Used by the sparkrun ecosystem
   - Supports fastsafetensors, InstantTensor loading, Ray distributed mode

---

## 7. DGX Spark Fit Analysis (121GB unified memory, GB10 sm_121)

### Model: azampatti/Ornith-1.0-35B-int4-AutoRound-SAR

| Factor | Value | Fits? |
|---|---|---|
| Model weights (int4) | ~20 GB | ✅ Yes |
| Active parameters per token | ~3B | ✅ Yes (very efficient) |
| Context 262K KV cache (FP8) | ~10 GB | ✅ Yes |
| GPU memory utilization | 0.55 recommended | ✅ ~67 GB of 121 GB |
| Total estimated VRAM | ~30-35 GB (weights + KV) | ✅ Plenty of headroom |

**Verdict: YES, fits comfortably on a single DGX Spark.**

The int4 quantized model at ~20GB is significantly smaller than:
- FP8 variant (~36 GB)
- BF16 original (~70 GB)
- NVFP4 community quant (~21.9 GB)

With `gpu_memory_utilization=0.55`, you get ~67 GB usable, of which ~20 GB is weights, leaving ~47 GB for KV cache — enough for 262K context with substantial concurrency headroom.

### Comparison to closest sparkrun recipe (Qwen3.6-35B-A3B-FP8-MTP)
The official FP8 recipe estimates:
- Model weights: 34.89 GB (FP8)
- KV cache: 10.00 GB (262K, FP8)
- Total per-GPU: 44.89 GB
- Max context tokens: 1,622,913 (6.2x the 262K max_model_len)

The int4 Ornith variant should use **~43% less memory** for weights (20 GB vs 35 GB), leaving even more room for KV cache and concurrency.

---

## 8. Recommended vLLM Configuration

### Required vLLM version: ≥ 0.19.1
The base model card specifies:
- Transformers ≥ 5.8.1
- **vLLM ≥ 0.19.1**
- SGLang ≥ 0.5.9

The Bitbull FP8 repo validated with vLLM 0.24.0.

### Basic Serving (from model card)
```bash
vllm serve azampatti/Ornith-1.0-35B-int4-AutoRound-SAR \
  --served-model-name ornith-1.0-35b \
  --max-model-len 196608 \
  --gpu-memory-utilization 0.55 \
  --load-format auto \
  --attention-backend flashinfer \
  --moe-backend marlin \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

### Production Configuration (DGX Spark optimized)
```bash
vllm serve azampatti/Ornith-1.0-35B-int4-AutoRound-SAR \
  --served-model-name ornith-1.0-35b \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.55 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 8 \
  --optimization-level 3 \
  --performance-mode throughput \
  --load-format instanttensor \
  --attention-backend flashinfer \
  --moe-backend marlin \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --default-chat-template-kwargs '{"preserve_thinking":true}' \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --generation-config auto \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
  --override-generation-config '{"temperature":0.6,"top_p":0.95,"top_k":-1,"min_p":0.0,"presence_penalty":0.0,"repetition_penalty":1.0}'
```

### Environment Variables (DGX Spark)
```bash
export TORCH_MATMUL_PRECISION=high
export NVIDIA_FORWARD_COMPAT=1
export NVIDIA_DISABLE_REQUIRE=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_MARLIN_USE_ATOMIC_ADD=1
export FLASHINFER_DISABLE_VERSION_CHECK=1
```

### Notes on tool-call parser
- The azampatti model card recommends `qwen3_coder` for tool calls
- The base model (deepreinforce-ai) recommends `qwen3_xml`
- The Bitbull FP8 repo uses `qwen3_xml`
- **Recommendation**: Try `qwen3_coder` first (per quantized model card), fall back to `qwen3_xml` if tool calling fails

---

## 9. Who Is azampatti?

**Aldo Zampatti** — HuggingFace user `azampatti`

- HF profile: https://huggingface.co/azampatti
- Has only **2 models** on HuggingFace:
  1. `azampatti/Ornith-1.0-35B-int4-AutoRound-SAR` (958 downloads)
  2. `azampatti/Qwen3.6-35B-A3B-int4-AutoRound-OC` (108 downloads)
- Both quantized using spark-auto-round (whpthomas's tool)
- The Qwen3.6 model card explicitly credits whpthomas: *"All Credit goes to https://github.com/whpthomas/spark-auto-round"*
- The "SAR" suffix likely stands for **S**park **A**uto**R**ound

### Related: whpthomas (the quantization tool author)
- GitHub: https://github.com/whpthomas/spark-auto-round
- HuggingFace: `whpthomas` — also has `Ornith-1.0-35B-int4-AutoRound` (1,192 downloads)
- Created the GB10-optimized AutoRound quantization wrapper
- The azampatti variant appears to be a derivative of whpthomas's work with added MTP head support and extended context

### Related: deepreinforce-ai (the base model creator)
- HuggingFace: `deepreinforce-ai` (DeepReinforce)
- Models: Ornith-1.0-9B, Ornith-1.0-35B, Ornith-1.0-397B (all variants: BF16, FP8, GGUF)
- Website: https://deep-reinforce.com/ornith.html
- Total downloads across all models: millions (35B-GGUF alone has 2.8M downloads)

---

## 10. NVIDIA Forum & Reddit Mentions

### NVIDIA DGX Spark Forum
- **No results found** for "ornith" or "ornith 35b DGX spark" on forums.developer.nvidia.com
- The Ornith model family appears to be too new for NVIDIA forum discussions

### Reddit (r/LocalLLaMA)
- Reddit search returned no parseable results (likely rate-limited or blocked)
- The web_search tool was unavailable (Firecrawl not configured)
- **Status: Unable to verify Reddit mentions via available tools**

### GitHub Activity
Significant GitHub activity found:
- 192-star repo: `ARahim3/mlx-dspark` — MLX speculative decoding port supporting Ornith-1.0
- 67-star repo: `AEON-7/Ornith-1.0-35B-AEON-Ultimate-Uncensored` — uncensored variant
- Multiple community deployment repos targeting DGX Spark specifically

---

## 11. All Known Ornith-1.0-35B Quantization Variants

| HF Repo | Format | Downloads | Notes |
|---|---|---|---|
| `deepreinforce-ai/Ornith-1.0-35B` | BF16 | 2,135,681 | Original |
| `deepreinforce-ai/Ornith-1.0-35B-FP8` | FP8 | 813,141 | Official FP8 |
| `deepreinforce-ai/Ornith-1.0-35B-GGUF` | GGUF | 2,810,304 | Official GGUF |
| `sakamakismile/Ornith-1.0-35B-NVFP4` | NVFP4 | 464,864 | Community, DGX Spark tested |
| `azampatti/Ornith-1.0-35B-int4-AutoRound-SAR` | int4 AutoRound | 958 | **Target — has MTP** |
| `whpthomas/Ornith-1.0-35B-int4-AutoRound` | int4 AutoRound | 1,192 | Original int4, no MTP |
| `cyburn/Ornith-1.0-35B-int4-AutoRound` | int4 AutoRound | 3,742 | Another int4 variant |
| `cyankiwi/Ornith-1.0-35B-AWQ-INT4` | AWQ INT4 | 30,144 | AWQ format |
| `XReyRobert/Ornith-1.0-35B-GPTQ-Pro-FOEM-4bit-g128-ns256` | GPTQ | 6,675 | GPTQ format |
| `AEON-7/Ornith-1.0-35B-AEON-Ultimate-Uncensored-NVFP4` | NVFP4 | 49,667 | Uncensored NVFP4 |
| `OsaurusAI/Ornith-1.0-35B-MXFP8` | MXFP8 | 4,156 | MXFP8 format |

---

## 12. Summary & Recommendations

### Feasibility: ✅ YES — Ornith-1.0-35B-int4-AutoRound-SAR is an excellent fit for DGX Spark

**Why this specific variant is the best choice:**
1. **MTP support** — 35-40% throughput improvement via speculative decoding (80-91 t/s vs 60-65 t/s)
2. **Smallest footprint** — ~20GB vs 36GB (FP8) or 70GB (BF16), leaving maximum KV cache space
3. **DGX Spark-optimized** — quantized specifically on GB10 using spark-auto-round
4. **Full 256K context** — unlike whpthomas variant (192K)
5. **Shared expert gates in FP16** — preserves MoE routing accuracy
6. **MIT license** — no restrictions

### Recommended approach:
1. Use the azampatti model card's **production vLLM config** (Section 8 above)
2. Set `gpu_memory_utilization=0.55` initially (can raise to 0.8 if stable)
3. Enable MTP with 3 speculative tokens
4. Use `flashinfer` attention backend + `marlin` MoE backend
5. Set the 6 environment variables listed in the model card
6. Use `eugr/spark-vllm-docker` as the container base

### Risks / Caveats:
1. **Quantization quality is moderate** — 45% of layers have warnings (cosine sim 0.986-0.990 for deep layers). This is typical for MoE int4 but expect slight degradation vs FP8
2. **No sparkrun recipe exists** — would need to create one (closest analog: `@official/qwen3.6-35b-a3b-fp8-mtp-vllm`)
3. **Concurrency stability** — the NVFP4 sibling repo showed concurrency-4 hangs; similar issues *might* affect int4 at high concurrency, though the architecture is the same and the issue may be vLLM-version-specific
4. **Tool-call parser ambiguity** — model card says `qwen3_coder`, base model says `qwen3_xml`; test both
5. **Very new model** — 958 downloads, minimal community validation on DGX Spark
6. **No official NVIDIA NVFP4 build** — only community requantizations exist
7. **web_search unavailable** — could not fully verify Reddit/forum discussions

### Suggested sparkrun recipe (if creating one):
```
Name: @oracle/ornith-1.0-35b-int4-autoround-mtp-vllm
Runtime: vllm-distributed
Model: azampatti/Ornith-1.0-35B-int4-AutoRound-SAR
TP: 1
Nodes: 1
GPU Mem: 0.55
Context: 262144
Speculative: {"method": "mtp", "num_speculative_tokens": 3}
Attention: flashinfer
MoE backend: marlin
Tool parser: qwen3_coder
Reasoning parser: qwen3
```