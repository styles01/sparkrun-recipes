# Step 3.7 Flash — Updates Since Flowtivity Blog Article (May 30, 2026)

**Research date:** July 15, 2026
**Article date:** May 30, 2026
**Model release:** May 29, 2026

## Executive Summary

The Step 3.7 Flash ecosystem has evolved dramatically since the Flowtivity article. **vLLM support is now official and production-ready**, with dedicated Docker images, MTP speculative decoding, and NVFP4 quantization support all available. The article's llama.cpp-only approach has been superseded by a full multi-backend deployment story. However, significant bugs remain in vLLM's MTP implementation on NVFP4, particularly affecting DGX Spark users.

---

## 1. vLLM Support — YES, Officially Added

### Official vLLM Support Merged
- **PR #43859** "[Model] Support Step-3.7-Flash" was **merged into vLLM main on May 28, 2026** — 2 days BEFORE the article.
  - Author: `ltd0924` (StepFun employee, from `stepfun-ai:feat/step3p7` branch)
  - 696 additions, 25 deletions, 15 files changed
  - Reviewers included core vLLM maintainers (WoosukKwon, ywang96, etc.)
- **StepFun has a vLLM fork** at `github.com/stepfun-ai/vllm` (45 stars, 3 PRs)
- The model card has been **updated with full vLLM deployment instructions** including:
  - `vllm serve` commands for BF16, FP8, and NVFP4 models
  - `--speculative_config '{"method": "mtp", "num_speculative_tokens": 3}'` for MTP
  - `--reasoning-parser step3p5` and `--tool-call-parser step3p5` for reasoning/tool support
  - `--enable-expert-parallel` for MoE expert parallelism
  - `--disable-cascade-attn` flag
  - `--async-scheduling` for NVFP4 mode

### Prebuilt vLLM Docker Image
- **`vllm/vllm-openai:stepfun37`** — StepFun's official prebuilt vLLM Docker image with Step 3.7 support
- Available via `docker pull vllm/vllm-openai:stepfun37`
- This is the recommended way to run Step 3.7 Flash on vLLM

### SGLang Support Also Added
- **`lmsysorg/sglang:dev-step-3.7-flash`** — Dedicated SGLang Docker image
- SGLang uses EAGLE speculative decoding (instead of MTP) with `--speculative-algorithm EAGLE`
- Supports `--enable-multi-layer-eagle` for multi-layer speculative decoding
- NVFP4 mode uses `--moe-runner-backend flashinfer_trtllm` and `--attention-backend trtllm_mha`

---

## 2. vLLM Container/Recipe for GB10/SM121 — YES

### NVIDIA Community Docker for DGX Spark
- **NVIDIA's Community Docker** has been updated with official Step 3.7 Flash recipes
- NVIDIA forum thread: "Step-3.7-Flash is supported in community Docker on DGX Spark!" (posted May 28, 2026)
- Posted by `eugr_nv` (NVIDIA Moderator)
- **4.3k views, 79 likes, 27 users participating** as of July 2026
- The model **requires at least two Sparks in a cluster**
- Both FP8 and NVFP4 checkpoints are supported; **NVFP4 recommended** due to memory constraints

### Community Docker Commands
```bash
# Update and build fresh container
git pull
./build-and-copy.sh --cleanup -c

# NVFP4 version (recommended)
./hf-download.sh stepfun-ai/Step-3.7-Flash-NVFP4 -c
./run-recipe.sh step-3.7-flash-nvfp4 --no-ray

# FP8 version
./hf-download.sh stepfun-ai/Step-3.7-Flash-FP8 -c
./run-recipe.sh step-3.7-flash-fp8 --no-ray
```
- `--no-ray` is **required** for FP8 to fit with full context

### vLLM NVFP4 Launch Flags (from model card)
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

---

## 3. New Quantizations

### Official Quantizations from StepFun
| Model | Format | Size | HuggingFace ID |
|-------|--------|------|----------------|
| BF16 | Safetensors | 201B | `stepfun-ai/Step-3.7-Flash` |
| FP8 | Safetensors | ~197B | `stepfun-ai/Step-3.7-Flash-FP8` |
| NVFP4 | Safetensors | 104B | `stepfun-ai/Step-3.7-Flash-NVFP4` |
| GGUF (various) | GGUF | 197B | `stepfun-ai/Step-3.7-Flash-GGUF` |

### Official GGUF Quantizations Available
- Q4_K_S: 111.5 GB
- IQ4_XS: 104.99 GB (used in the Flowtivity article)
- Q3_K_L: 102.5 GB
- Multimodal Projector (FP16): 3.97 GB
- Runtime overhead: ~7 GB
- Minimum memory: 120 GB

### Community Quantizations (30+ models on HuggingFace)
- **`jcbtc/Step-3.7-Flash-ROCmFPX-Q3-QualityPlus`** — ROCm FPX Q3 quantization (updated 1 day ago, 3.27k downloads)
- **`0xSero/Step-3.7-Flash-148B`** — REAP-pruned NVFP4 (148B, 95GB, 212/288 experts kept)
- **`0xSero/Step-3.7-Flash-173B`** — Less aggressively pruned REAP variant
- **`osmapi/Step-3.7-Flash-MXFP4-mlx`** — MXFP4 for Apple MLX
- **`osmapi/Step-3.7-Flash-OptiQ-3.7bpw-mlx`** — OptiQ 3.7 bpw for MLX
- **`mlx-community/Step-3.7-Flash-4bit`**, `6bit`, `8bit` — MLX quantizations
- **`unsloth/Step-3.7-Flash-GGUF`** — Unsloth's GGUF builds (32.4k downloads)
- **`tarruda/Step-3.7-Flash-GGUF`** — Community GGUF (4.8k downloads, updated 6 days ago)
- **`bartowski/Step-3.7-Flash-GGUF`** — Bartowski's GGUF builds
- **`AesSedai/Step-3.7-Flash-GGUF`** — Another community GGUF
- **`mudler/Step-3.7-Flash-APEX-GGUF`** — APEX GGUF variant
- **`meshllm/Step-3.7-Flash-UD-Q4_K_XL-layers`** — Layer-distributed GGUF
- **`Aminlight/Step-3.7-Flash-oQ6-mtp`** — Q6 with MTP support
- **`tcclaviger/Step-3.7-Flash-240REAP`** — REAP-pruned variant (240 experts)
- **`AEON-7/Step-3.7-Flash-AEON-Ultimate-Abliterated-BF16`** — Abliterated BF16
- Various JANGQ-AI pruned variants (JANG_2L, JANG_K, JANGTQ_K)
- **Total: 80 models** on HuggingFace matching "step-3.7-flash"

### No INT4+FP8 Hybrid Quantization Found
- No hybrid INT4+FP8 quantization has been released as of July 2026
- NVFP4 (4-bit float) is the closest to this concept and is officially supported

---

## 4. MTP / Speculative Decoding — YES, with Caveats

### vLLM MTP Support
- **MTP (Multi-Token Prediction) is officially supported** in vLLM for Step 3.7 Flash
- Config: `--speculative_config '{"method": "mtp", "num_speculative_tokens": 3}'`
- Uses vLLM's `Step3p5MTP` and `Step3p5MTPProposer` classes
- The model architecture code lives in `vllm/model_executor/models/step3p5_mtp.py`

### SGLang EAGLE Support
- SGLang uses EAGLE speculative decoding instead of MTP
- `--speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4`
- Supports `--enable-multi-layer-eagle`

### CRITICAL BUG: MTP on NVFP4 is Broken
**Two major bugs discovered in vLLM's MTP implementation for NVFP4:**

#### Bug 1: MTP drafter quantizes mtp_block incorrectly (Issue #44087)
- **Status: Closed** (fix merged)
- Opened: May 30, 2026 by `choiceoh`
- The MTP drafter builds `mtp_block` and `shared_head` using the model's NVFP4 quant config even when MTP weights are unquantized (BF16)
- This causes `AssertionError` (weight shape mismatch) — packed FP4 vs BF16 shapes
- `exclude_modules` in `hf_quant_config.json` cannot reach MTP layers because they're not target-model modules
- **Fix was merged** — the drafter now skips quantization for MTP block when checkpoint tensors are unquantized

#### Bug 2: MTP drafter assumes wrong hidden_dim (Issue #44836)
- **Status: Closed as not planned**
- Opened: Jun 8, 2026 by `evehour`
- `step3p5_mtp.py`'s `VocabParallelEmbedding` parameter assumes `hidden_dim=2048` but the NVFP4 checkpoint has `hidden_dim=4096`
- Error: `RuntimeError: The size of tensor a (2048) must match the size of tensor b (4096) at non-singleton dimension 1`
- The stock `stepfun-ai/Step-3.7-Flash-NVFP4` export **ships no MTP weights at all** — the 3 nextn layers were dropped during quantization
- **Reproducible by multiple users in the DGX Spark community**
- Referenced in NVIDIA forum thread (post #49) and HuggingFace discussions (#4)

### Community Solution: MTP Draft Model
**`Hikari07jp/Step-3.7-Flash-MTP-draft`** — A community-created MTP draft model
- Extracts the 3 MTP/nextn layers from the BF16 release (they're tiny, ~5.9GB)
- Packaged as a vLLM-loadable draft with `model_type: step3p7` and `num_nextn_predict_layers > 0`
- Drop-in: point vLLM's `--speculative-config` at this directory
- Works with `vllm/vllm-openai:stepfun37` Docker image

#### MTP Draft Benchmarks (2× RTX PRO 6000 Blackwell, SM120, TP=2)
- **Acceptance rate: ~0.80** in production traffic
- K=1 is the sweet spot (K=2/K=3 lose to draft cost)
- Single-stream decode speedups:
  - Free-form: +17.5% (106.8 → 125.5 tok/s)
  - Code: +25.3% (106.7 → 133.7 tok/s)
  - Japanese: +20.9% (107.0 → 129.3 tok/s)
  - Tool-call: +26.6% (106.9 → 135.4 tok/s)
- Speedup grows with context length (base becomes more memory-bound):
  - 1K context: +20% (K=1)
  - 8K context: +22%
  - 32K context: +22%
  - 128K context: +28%

#### MTP Draft Usage
```bash
docker run -d --gpus all --ipc=host --shm-size=64g --network host \
  -v /path/to/Step-3.7-Flash-NVFP4:/model:ro \
  -v /path/to/Step-3.7-Flash-MTP-draft:/draft:ro \
  vllm/vllm-openai:stepfun37 \
  /model \
    --served-model-name step3p7 --port 8000 \
    --trust-remote-code --tensor-parallel-size 2 --enable-expert-parallel \
    --quantization modelopt --kv-cache-dtype fp8 \
    --max-model-len 262144 --gpu-memory-utilization 0.92 --async-scheduling \
    --speculative-config '{"method":"mtp","model":"/draft","num_speculative_tokens":1}'
```

---

## 5. HuggingFace Safetensors Versions — YES, Available Since Launch

The **original BF16 model** (`stepfun-ai/Step-3.7-Flash`) was released in **safetensors format** from day one:
- 201B params, tagged `Safetensors` and `Transformers`
- Tensor types: BF16, F32
- 142,771 downloads/month

Additional safetensors versions:
- **FP8**: `stepfun-ai/Step-3.7-Flash-FP8` — safetensors, fp8 precision
- **NVFP4**: `stepfun-ai/Step-3.7-Flash-NVFP4` — safetensors, 8-bit precision, modelopt
  - 104B params, tensor types: F32, BF16, F8_E4M3, U8
  - 61,438 downloads/month
- **Pruned NVFP4**: `0xSero/Step-3.7-Flash-148B` — safetensors, 95GB on-disk

All of these are directly vLLM-compatible (vLLM requires safetensors, not GGUF).

---

## 6. vLLM vs llama.cpp Benchmarks

### No direct vLLM-vs-llama.cpp comparison benchmark has been published
The Flowtivity article only benchmarked llama.cpp. No community benchmark directly comparing vLLM and llama.cpp on the same hardware has been found.

### vLLM Performance Data Available
- The MTP draft model page includes vLLM benchmarks on 2× RTX PRO 6000 (SM120):
  - Base NVFP4 (no speculation): ~107 tok/s single-stream
  - With MTP K=1: ~125-135 tok/s (+17-27%)
- NVIDIA Community Docker recipes are designed for clustered DGX Spark (2+ Sparks)
- The model card claims up to 400 tok/s throughput (likely on larger GPU clusters)

### llama.cpp Performance (from article)
- IQ4_XS GGUF (105GB) on llama.cpp
- DGX Spark with CUDA graphs + MMQ
- Specific tok/s numbers from the article

---

## 7. Community Recipes for DGX Spark

### NVIDIA Community Docker (Primary)
- **Repository**: NVIDIA's Community Docker for DGX Spark
- **Forum thread**: https://forums.developer.nvidia.com/t/step-3-7-flash-is-supported-in-community-docker-on-dgx-spark/371652
- **Recipes**: `step-3.7-flash-nvfp4` and `step-3.7-flash-fp8`
- **Requirements**: At least 2 Sparks clustered
- NVFP4 recommended for memory constraints
- `--no-ray` required for FP8 with full context

### StepFun's llama.cpp Fork
- **Repository**: `github.com/stepfun-ai/llama.cpp` (branch: `step3.7`)
- Official StepFun fork with Step 3.7 support
- Build instructions for Mac, DGX-Spark, and AMD Windows included in model card
- DGX-Spark build flags: `-DGGML_CUDA=ON -DGGML_CUDA_GRAPHS=ON -DGGML_CUDA_FORCE_MMQ=ON`

### StepFun's vLLM Fork
- **Repository**: `github.com/stepfun-ai/vllm` (45 stars)
- 3 pull requests

### Community Docker Image for DGX Spark
- `eugr/spark-vllm-docker` — Community vLLM Docker for DGX Spark (referenced in HF discussions)

---

## 8. Updates from StepFun

### Model Card Significantly Updated
The original model card (May 29) has been expanded with:
- **Section 4: Availability, Deployment, and Ecosystem** — now lists vLLM, SGLang, Transformers, llama.cpp as supported backends
- **Section 6: Local Deployment** — full instructions for vLLM (6.1), SGLang (6.2), Transformers (6.3), llama.cpp (6.4)
- **NVIDIA NIM** — Step 3.7 Flash available as NVIDIA NIM inference microservice
- **Megatron ecosystem** — StepFun model support landed in NVIDIA Megatron Core and Megatron Bridge
- **Agent platforms** — mentions Hermes Agent, OpenClaw, Kilo Code
- **Pricing** — $0.20/M input (cache miss), $0.04/M input (cache hit), $1.15/M output
- Available on: StepFun Open Platform, OpenRouter, NVIDIA NIM, DeepInfra, Fireworks AI, Modal

### Official Repositories
- `stepfun-ai/Step-3.7-Flash` GitHub repo (274 stars, 23 forks)
- `stepfun-ai/llama.cpp` fork (branch: `step3.7`)
- `stepfun-ai/vllm` fork
- Collection on HuggingFace: "Step-3.7-Flash" (6 items, updated May 30)

### Transformers 5.0 Required
- The model card notes: "Deployment of this model requires transformers 5.0 or later"

---

## 9. Model Updates / Fine-tunes

### No Updated Base Model
- The base model has not been updated or fine-tuned since release
- BF16 model last updated: Jun 2, 2026 (minor config/card updates)
- NVFP4 model last updated: Jun 1, 2026
- FP8 model exists but has fewer downloads (16 likes vs 413 for BF16)

### Community Fine-tunes and Variants
- **7 finetunes** listed in the model tree
- **36 quantizations** in the model tree
- `dealignai/Step-3.7-Flash-JANG_2L-CRACK`
- `JANGQ-AI/Step-3.7-Flash-JANG_2L`, `JANG_K`, `JANGTQ_K` — Pruned variants
- `OsaurusAI/Step-3.7-Flash-JANG_2L`, `JANG_K` — More pruned variants
- `anerjy/Step-3.7-Flash-MLX-adapter` — MLX adapter
- `AEON-7/Step-3.7-Flash-AEON-Ultimate-Abliterated-BF16` — Abliterated version
- `Aminlight/Step-3.7-Flash-oQ6-mtp` — Q6 with MTP

---

## 10. Issues, Bugs, and Limitations Discovered

### Critical Bugs
1. **MTP + NVFP4 = Broken** (vLLM #44087, #44836)
   - The official NVFP4 checkpoint ships **no MTP weights** (3 nextn layers dropped during quantization)
   - vLLM's MTP drafter has hardcoded `hidden_dim=2048` but the checkpoint uses `4096`
   - vLLM's MTP drafter incorrectly applies NVFP4 quant config to unquantized MTP weights
   - **Workaround**: Use `Hikari07jp/Step-3.7-Flash-MTP-draft` (community-extracted BF16 MTP layers)
   - Bug #44087 was fixed (merged), Bug #44836 was closed as "not planned"

2. **`libtorch_cuda.so` missing** (NVIDIA forum)
   - Community Docker build issue, fixed by `./build-and-copy.sh --cleanup -c`
   - Related to GitHub API rate limiting causing stale flashinfer wheels

### Known Limitations
1. **Requires 2+ DGX Sparks** — Cannot run on a single Spark (even NVFP4 at 104B exceeds single-Spark memory with full context)
2. **FP8 needs `--no-ray`** — Required to fit with full 256K context
3. **NVFP4 max context limited** — Model card shows `--max-model-len 8192` for NVFP4 vLLM (vs 262144 for BF16/FP8)
4. **`custom_code` required** — Model uses `trust_remote_code=True` in all backends
5. **No `torch.compile` support** — Warning: "model does not support torch.compile" seen in error logs
6. **Transformers 5.0+ required** — Won't work with older transformers versions

### Community Discussion Topics (HuggingFace)
- "Tool calling bug in the official API" — reported by `tarruda`
- "How to turn off thinking" — 9 comments, community discussion about reasoning levels
- "Supers Found in model" — 9 comments, community findings
- "Can we add something like Qwen's preserve_thinking to the chat template?" — feature request
- "The long-context model encountered an error" — long-context bug report
- "Benchmarks of reasoning levels?" — community requesting per-level benchmarks

### vLLM Roadmap Q3 2026
- vLLM has published a Q3 2026 roadmap (Issue #48168, opened July 10, 2026)
- May include further Step 3.7 optimizations

### Additional vLLM PRs/Issues Related to Step 3.7
- `[Bugfix][MM][CG] Enable dual-path ViT CUDA graph for Step3-VL` — ViT CUDA graph fix
- `[MM][CG] Enable encoder Cudagraph for Step3VL` — Encoder cudagraph support
- `[CI] Fix "test_vit_cudagraph_[image|video][step3_vl]" failure` — CI fix
- `[Rust Frontend] Add seed_oss and step3p5 reasoning parsers` — Rust frontend parser support
- `[Bugfix] Fix weight loading issues caused by #41184` — General weight loading fix
- `[Core] Release cached device memory under pressure on UMA GPUs during weight loading` — UMA optimization (relevant for DGX Spark)
- `[New Model][Nvidia] Add SM12x support for DeepSeek V4 Flash` — SM12x (DGX Spark architecture) support improvements

---

## Summary: What Changed Since the Article

| Question | Answer |
|----------|--------|
| Has vLLM support been added? | **YES** — Official, merged May 28 (before article), with Docker image |
| vLLM container/recipe for GB10/SM121? | **YES** — NVIDIA Community Docker + `vllm/vllm-openai:stepfun37` |
| New quantizations (NVFP4, INT4+FP8)? | **NVFP4 YES** (official), FP8 YES (official). No INT4+FP8 hybrid. 30+ community quants |
| MTP/speculative decoding added? | **YES** — Official MTP in vLLM, EAGLE in SGLang. But **broken on NVFP4** — use community draft model |
| Anyone benchmarked on vLLM? | **Partial** — MTP draft has vLLM benchmarks, no direct vLLM-vs-llama.cpp comparison |
| HuggingFace safetensors versions? | **YES** — BF16, FP8, NVFP4 all in safetensors since launch |
| Community recipes for DGX Spark? | **YES** — NVIDIA Community Docker with official recipes |
| Updates from StepFun? | **YES** — Model card expanded, vLLM/SGLang Docker images, NIM support, Megatron integration |
| Model updated or fine-tuned? | **NO** base model update. 7 community finetunes, REAP-pruned variants |
| Issues/bugs discovered? | **YES** — MTP+NVFP4 broken (2 bugs), requires 2+ Sparks, NVFP4 context limited to 8K in vLLM |

---

## Key Links

- **Main model**: https://huggingface.co/stepfun-ai/Step-3.7-Flash
- **NVFP4 model**: https://huggingface.co/stepfun-ai/Step-3.7-Flash-NVFP4
- **FP8 model**: https://huggingface.co/stepfun-ai/Step-3.7-Flash-FP8
- **GGUF model**: https://huggingface.co/stepfun-ai/Step-3.7-Flash-GGUF
- **MTP draft (community)**: https://huggingface.co/Hikari07jp/Step-3.7-Flash-MTP-draft
- **REAP-pruned 148B**: https://huggingface.co/0xSero/Step-3.7-Flash-148B
- **vLLM support PR**: https://github.com/vllm-project/vllm/pull/43859
- **MTP NVFP4 bug #1**: https://github.com/vllm-project/vllm/issues/44087
- **MTP NVFP4 bug #2**: https://github.com/vllm-project/vllm/issues/44836
- **NVIDIA forum thread**: https://forums.developer.nvidia.com/t/step-3-7-flash-is-supported-in-community-docker-on-dgx-spark/371652
- **StepFun GitHub**: https://github.com/stepfun-ai
- **StepFun llama.cpp fork**: https://github.com/stepfun-ai/llama.cpp (branch: step3.7)
- **StepFun vLLM fork**: https://github.com/stepfun-ai/vllm
- **HF discussion (MTP bug)**: https://huggingface.co/stepfun-ai/Step-3.7-Flash-NVFP4/discussions/4
- **HF community discussions**: https://huggingface.co/stepfun-ai/Step-3.7-Flash/discussions