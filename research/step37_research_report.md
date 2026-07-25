# StepFun Step 3.7 Flash — Research Report for DGX Spark Deployment

**Date:** 2026-07-18
**Target Hardware:** NVIDIA DGX Spark (GB10 ARM64, sm_121, 121GB unified memory)
**Sources:** HuggingFace model card, GGUF README, official StepFun blog, GitHub issues (vLLM 59, llama.cpp 11), Docker Hub tags

---

## 1. What is StepFun Step 3.7 Flash?

- **HuggingFace ID:** `stepfun-ai/Step-3.7-Flash` (Apache-2.0)
- **GGUF repo:** `stepfun-ai/Step-3.7-Flash-GGUF` (Apache-2.0)
- **GitHub:** `stepfun-ai/Step-3.7-Flash`
- **Type:** Sparse Mixture-of-Experts (MoE) vision-language model
- **Total params:** 198B (196B language backbone + 1.8B vision encoder, perception_encoder arch)
- **Active params/token:** ~11B (MoE top-8 of 288 experts)
- **Context window:** 256K tokens (max_position_embeddings: 262144)
- **Architecture:** `Step3p7ForConditionalGeneration` (custom_code, trust_remote_code required)
  - Text backbone: `Step3p5ForCausalLM` — 45 layers, hidden_size 4096, GQA (64 heads, 8 groups, head_dim 128)
  - MoE on layers 3-44 (288 experts, top-8, sigmoid router, scaling 3.0, shared expert dim 1280)
  - Sliding window 512 + full attention pattern (1 full + 3 sliding, repeating)
  - RoPE scaling: llama3-type, factor 2.0, original max 131072
  - MTP (Multi-Token Prediction): 3 next-N predict layers for speculative decoding
  - Vision: perception_encoder, image_size 728, patch 14, 47 layers, 16 heads
- **Capabilities:** Native multimodal (image + text), tool calling, multi-step reasoning, agentic workflows, coding (SWE-Bench PRO 56.3, 2nd place), search/visual grounding (SimpleVQA 79.2 #1, V* 95.3)
- **Reasoning levels:** 3 selectable (low/medium/high) to trade speed vs depth
- **Throughput claim:** up to 400 tokens/sec (server-side)
- **Languages:** en, zh, ja, ko, ar, hi, de, fr, es, ru (multilingual)
- **Custom code:** Requires `transformers>=5.0` and `trust_remote_code=True`

## 2. Can it run on NVIDIA DGX Spark (GB10, 121GB)?

**YES — officially supported and benchmarked.**

From the official model card (Section 4):
> "For local and workstation scenarios, it can also run on high-memory devices such as **NVIDIA DGX Station**, AMD Ryzen AI Max+ 395-based systems, and Mac Studio / Macbook Pro devices with at least 128GB unified memory."

From the GGUF README:
> "With 128 GB of unified memory (Mac Studio, **DGX Spark**, Ryzen AI Max+ 395, etc.), you can privately host Step-3.7-Flash: Q4 quants and below run at full 256K context with high precision."

**DGX Spark is explicitly named as a target platform.** The 121GB unified memory on GB10 fits Q4_K_S (112GB), IQ4_XS (105GB), Q3_K_L (103GB), Q3_K_M (94GB), and IQ3_XXS (76GB) quants.

### Official DGX Spark (GB10, 128GB) Benchmarks (llama.cpp)

**Q4_K_S** (context up to 131300):
| N_KV | TG t/s | PP t/s | Total t/s |
|------|--------|--------|-----------|
| 128 | 24.82 | — | 24.82 |
| 2176 | 26.08 | 255.33 | 168.31 |
| 8320 | 24.76 | 753.89 | 518.86 |
| 16512 | 20.60 | 557.49 | 463.78 |
| 32896 | 18.47 | 624.14 | 553.50 |
| 65664 | 16.48 | 583.47 | 546.79 |
| 131072 | 13.02 | 465.66 | 450.37 |

**IQ4_XS** (context up to 262272, full 256K):
| N_KV | TG t/s | PP t/s | Total t/s |
|------|--------|--------|-----------|
| 128 | 23.85 | — | 23.85 |
| 8320 | 22.01 | 653.73 | 453.46 |
| 32896 | 19.60 | 630.44 | 562.25 |
| 131072 | 12.47 | 438.74 | 424.58 |
| 262144 | 8.61 | 283.44 | 279.09 |

**Q3_K_L** (context up to 262272, full 256K):
| N_KV | TG t/s | PP t/s | Total t/s |
|------|--------|--------|-----------|
| 128 | 21.52 | — | 21.52 |
| 32896 | 18.98 | 596.25 | 533.15 |
| 131072 | 11.87 | 415.57 | 402.23 |
| 262144 | 8.22 | 288.00 | 283.30 |

**Note:** The GGUF README lists "NVIDIA DGX Spark (GB10, 128 GB unified memory)" — the user's 121GB unit is slightly below the 128GB reference but Q3_K_L (103GB) and IQ4_XS (105GB) fit comfortably with runtime overhead ~7GB. Q4_K_S (112GB) is tight but feasible.

## 3. Deployment Recipes

### 3.1 vLLM (RECOMMENDED — dedicated ARM64 Docker images exist)

**Docker Hub has purpose-built ARM64 images:**
- `vllm/vllm-openai:stepfun37-arm64-cu130` (10.2GB) — **CUDA 13.0, ARM64**
- `vllm/vllm-openai:stepfun37-arm64-cu129` (12.2GB) — CUDA 12.9, ARM64
- Also: `stepfun37` (generic), `stepfun37-x86_64-cu130/cu129`

Since the DGX Spark runs PyTorch 2.13+cu130 on GB10 (sm_121), the **`stepfun37-arm64-cu130`** image is the best match.

**vLLM launch (FP8):**
```bash
vllm serve <MODEL_PATH_OR_HF_ID> \
  --served-model-name step3p7-flash \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --disable-cascade-attn \
  --reasoning-parser step3p5 \
  --enable-auto-tool-choice \
  --tool-call-parser step3p5 \
  --speculative_config '{"method": "mtp", "num_speculative_tokens": 3}' \
  --trust-remote-code
```

**vLLM launch (BF16):**
```bash
vllm serve <MODEL_PATH_OR_HF_ID> \
  --served-model-name step3p7-flash-bf16 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --disable-cascade-attn \
  --reasoning-parser step3p5 \
  --enable-auto-tool-choice \
  --tool-call-parser step3p5 \
  --speculative_config '{"method": "mtp", "num_speculative_tokens": 3}' \
  --trust-remote-code
```

**vLLM launch (NVFP4 — requires modelopt, FP8 KV cache):**
```bash
python3 -m vllm.entrypoints.openai.api_server \
  --host 0.0.0.0 --port ${PORT} \
  --model stepfun-ai/Step-3.7-Flash-NVFP4 \
  --served-model-name step3p7 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.9 \
  --enable-expert-parallel \
  --trust-remote-code \
  --quantization modelopt \
  --kv-cache-dtype fp8 \
  --max-model-len 8192 \
  --reasoning-parser step3p5 \
  --enable-auto-tool-choice \
  --tool-call-parser step3p5 \
  --async-scheduling
```

**Important flags for this model:**
- `--enable-expert-parallel` (MoE expert parallelism)
- `--disable-cascade-attn` (required, breaks with cascade)
- `--reasoning-parser step3p5` and `--tool-call-parser step3p5` (custom parsers)
- `--speculative_config '{"method":"mtp","num_speculative_tokens":3}'` (MTP speculative decoding)
- `--trust-remote-code` (custom model code)

### 3.2 SGLang

Dedicated Docker: `lmsysorg/sglang:dev-step-3.7-flash`

**BF16:**
```bash
sglang serve --model-path stepfun-ai/Step-3.7-Flash \
  --tp 8 --reasoning-parser step3p5 --tool-call-parser step3p5 \
  --enable-multimodal \
  --speculative-algorithm EAGLE --speculative-num-steps 3 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  --enable-multi-layer-eagle --trust-remote-code \
  --host 0.0.0.0 --port 8000
```

**FP8** (model `stepfun-ai/Step-3.7-Flash-FP8`): add `--ep 4`

**NVFP4** (model `stepfun-ai/Step-3.7-Flash-NVFP4`): add `--tp 4 --ep 4 --moe-runner-backend flashinfer_trtllm --kv-cache-dtype fp8_e4m3 --quantization modelopt_fp4 --attention-backend trtllm_mha`

**Note for Blackwell GPUs:** `--mm-attention-backend fa4` may be used.

### 3.3 llama.cpp (BEST for 121GB unified memory — DGX-Spark build recipe in official README)

**Clone the StepFun fork (NOT mainline — has step3.7-specific branch):**
```bash
git clone https://github.com/stepfun-ai/llama.cpp.git
cd llama.cpp
git checkout -b step3.7 origin/step3.7
```

**Build for DGX-Spark (CUDA):**
```bash
cmake -S . -B build-cuda \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DGGML_CUDA_GRAPHS=ON \
  -DGGML_CUDA_FORCE_MMQ=ON \
  -DLLAMA_OPENSSL=OFF \
  -DLLAMA_BUILD_COMMON=ON \
  -DLLAMA_BUILD_TOOLS=ON \
  -DLLAMA_BUILD_SERVER=ON \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_TESTS=OFF
cmake --build build-cuda -j8
```

**Run (text + vision server):**
```bash
./build/bin/llama-server \
  -m Step-3.7-flash-Q4_K_S.gguf \
  --mmproj mmproj-Step-3.7-flash-f16.gguf \
  -c 32768 -ngl 99 -fa on \
  --host 0.0.0.0 --port 8080
```

**Benchmark:**
```bash
./build/bin/llama-batched-bench -m Step-3.7-flash-Q4_K_S.gguf \
  -c 131300 -b 2048 -ub 1024 \
  -npp 0,2048,8192,16384,32768,65536,131072 -ntg 128 -npl 1
```

### 3.4 Transformers (debug only — not for throughput)

Requires `transformers>=5.0`. Uses `AutoProcessor`/`AutoModelForCausalLM` with `trust_remote_code=True`, `device_map="auto"`.

## 4. HuggingFace Model IDs & Download

| Repo | ID | Format | Size |
|------|----|--------|------|
| BF16 (reference) | `stepfun-ai/Step-3.7-Flash` | safetensors | ~394GB |
| FP8 | `stepfun-ai/Step-3.7-Flash-FP8` | safetensors | ~110GB |
| NVFP4 | `stepfun-ai/Step-3.7-Flash-NVFP4` | safetensors | ~60GB |
| GGUF | `stepfun-ai/Step-3.7-Flash-GGUF` | gguf | see below |

**GGUF quant files** (imatrix-calibrated):
| File | Quant | Size |
|------|-------|------|
| `Step-3.7-flash-BF16.gguf` | BF16 | 394 GB |
| `Step-3.7-flash-Q8_0.gguf` | Q8_0 | 209 GB |
| `Step-3.7-flash-Q4_K_S.gguf` | Q4_K_S | 112 GB |
| `Step-3.7-flash-IQ4_XS.gguf` | IQ4_XS | 105 GB |
| `Step-3.7-flash-Q3_K_L.gguf` | Q3_K_L | 103 GB |
| `Step-3.7-flash-Q3_K_M.gguf` | Q3_K_M | 94 GB |
| `Step-3.7-flash-IQ3_XXS.gguf` | IQ3_XXS | 76 GB |
| `mmproj-Step-3.7-flash-f16.gguf` | F16 (vision projector) | 4 GB |

**Download (GGUF, recommended for 121GB DGX Spark):**
```bash
huggingface-cli download stepfun-ai/Step-3.7-Flash-GGUF \
  Step-3.7-flash-IQ4_XS.gguf mmproj-Step-3.7-flash-f16.gguf \
  --local-dir ./Step-3.7-Flash-GGUF
```

**Download (vLLM FP8):**
```bash
huggingface-cli download stepfun-ai/Step-3.7-Flash-FP8 --local-dir ./Step-3.7-Flash-FP8
```

## 5. Reddit / Community Discussions

**Reddit access was blocked** (Cloudflare 403 on all endpoints: JSON API, old.reddit, redlib proxy, jina reader). Search engines (Google, Bing, Brave) also triggered CAPTCHAs. Therefore I could NOT retrieve r/LocalLLaMA or r/MachineLearning threads directly. However:

- **Brave autocomplete** confirms "stepfun step 3.7 flash reddit" is a highly-searched query, indicating active Reddit discussion exists.
- **HuggingFace Community** (stepfun-ai/Step-3.7-Flash) has 15 discussions, including:
  - "How to turn off thinking" (9 comments)
  - "Tool calling bug in the official API"
  - "Benchmarks of reasoning levels?"
  - "The long-context model encountered an error"
  - "Its multimodal capabilities are pretty good"

**GitHub issues serve as the best community signal (since Reddit was inaccessible):**

### vLLM (59 issues matching "step 3.7") — notable:
- **[Model] Support Step-3.7-Flash** #43859 (closed, merged) — initial support PR
- **[Bug] MTP drafter weight loader assumes hidden_dim=2048** #44836 (closed) — NVFP4 checkpoint has 4096
- **[Bug] MTP speculative decoding fails on NVFP4** #44087 (closed) — drafter quantizes mtp_block
- **[Feature] FlashInfer-CUTLASS NVFP4 MoE lacks SWIGLUSTEP activation** #48921 (open) — blocks NVFP4 at TP=8

### llama.cpp (11 issues) — notable:
- **#24181** "Step 3.7 Flash gets stuck in reasoning trying to make tool calls" — **79 comments** (most active thread; Mac Metal)
- **#24254** "mtmd: Step-3.7-Flash decodes images in 80-token batches; slow processing" (open, 8 comments)
- **#24257** "Crash with Step 3.7 flash" (open, 4 comments)
- **#24259** "infinite <<<<<... in Step 3.7 flash's reasoning" (open, 7 comments)
- **#25129** "MTP breaks multimodality — inconsistent sequence positions" (open, 2 comments)
- **#24486** "TP: Stepfun 3.7 does not work with uneven splits" (open, 5 comments) — CUDA, 10×5060Ti
- **#24218** "--spec-draft-p-min + --no-mmap crashes on Blackwell" (open) — **Blackwell-relevant**
- **#25144** "speculative: fix MTP draft crash on vision inputs" (open PR)
- **PR #24340** "Support Step3.5/3.7 flash mtp3" (closed, merged)

## 6. Special Flags / Configs for ARM64 / CUDA (GB10 sm_121)

### vLLM on GB10
- Use the **`vllm/vllm-openai:stepfun37-arm64-cu130`** Docker image — purpose-built for ARM64 + CUDA 13.0. This is the single most important finding: ARM64 binaries are prebuilt.
- The model needs `--enable-expert-parallel` (288 experts benefit from EP), `--disable-cascade-attn`, and the step3p5 parsers.
- MTP speculative decoding (`--speculative_config '{"method":"mtp","num_speculative_tokens":3}'`) gives the throughput boost.
- TP=8 is recommended in the docs; on a single GB10 die you may need TP=1 or TP=2 with expert parallel — test empirically. The NVFP4 recipe uses TP=4.

### llama.cpp on DGX Spark (official build recipe)
- Must use the **`stepfun-ai/llama.cpp` fork**, branch `step3.7` (NOT mainline ggml-org — the step3.7 branch has the model arch).
- Build flags: `-DGGML_CUDA=ON -DGGML_CUDA_GRAPHS=ON -DGGML_CUDA_FORCE_MMQ=ON` (`GGML_CUDA_FORCE_MMQ` is GB10-relevant since Blackwell-class sm_121 has limited native FP8 MMA in some kernels).
- Run with `-fa on` (flash attention), `-ngl 99` (all layers on GPU), `-b 2048 -ub 1024`.
- **Blackwell-specific open bug (#24218):** `--spec-draft-p-min + --no-mmap` crashes on Blackwell — avoid combining these on GB10.

### Known GB10 / Blackwell gotchas
- NVFP4 quantization via vLLM has an open issue (#48921): FlashInfer-CUTLASS NVFP4 MoE lacks SWIGLUSTEP activation, blocking NVFP4 at TP=8. If using NVFP4, check this issue's status.
- MTP + multimodal has open bugs in both vLLM (#44087) and llama.cpp (#25129, #25144) — vision + speculative decoding is the least stable path.
- Reasoning-loop bug (llama.cpp #24181, 79 comments) is the most-reported local issue — model gets stuck in `<` patterns during tool-calling/reasoning. Workaround: use the autoparser carefully or disable MTP for agentic loads.

## 7. Recommendation for this DGX Spark (121GB)

**Best path: llama.cpp with IQ4_XS or Q3_K_L GGUF.**
- Fits comfortably in 121GB (105GB or 103GB + 7GB overhead + 4GB mmproj = ~116GB).
- Full 256K context supported per official benchmarks.
- Official DGX Spark build recipe and benchmarks exist.
- Expected TG throughput: ~22-24 t/s at short context, ~12 t/s at 131K, ~8.6 t/s at 256K.

**Alternative: vLLM with FP8 via the `stepfun37-arm64-cu130` Docker image.**
- Higher throughput (up to 400 t/s claimed with MTP) but requires TP≥4 and the FP8 checkpoint (~110GB).
- Risk: FP8 + MTP has known bugs on this architecture; verify #44087 and #48921 are resolved before committing.
- 121GB is tight for FP8 (110GB) + KV cache + activations — may need to limit `--max-model-len`.

**Avoid:** NVFP4 on GB10 (open vLLM blocker #48921), BF16 (394GB — does not fit), Q8_0 GGUF (209GB — does not fit).