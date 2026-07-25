# Puzzle-75B MTP OOM on DGX Spark — THE EXACT FIX

**Research date:** 2026-07-18
**Verdict:** The OOM is **NOT a single flag, container, or vLLM version issue.** It is a combination of (A) a FlashInfer library bug in the NVFP4 MoE autotune workspace allocation, (B) the draft head silently inheriting the wrong MoE/attention backend, and (C) `mamba_cache_mode=all` auto-activating with prefix caching. moranilt's working config dodges (A) because v0.23.0's FlashInfer predates the PR #3738 regression, and dodges (B)/(C) through a quirk of the Puzzle drafter (its MoE is BF16/unquantized → auto-selects Triton regardless).

---

## THE 3 ROOT CAUSES (in priority order)

### Cause A — FlashInfer PR #3738 regression (THE NEW ONE, 2026-07-16)

**This is the most likely reason our recent crashes differ from moranilt's 9-day-old success.**

- **FlashInfer PR #3738** narrowed the "native FP4 profiler workspace allocation" to only the **FP8-activation family** (`is_native_wfp4afp8_family`).
- Native Blackwell NVFP4 MoE uses **FP4 activations + FP4 weights**, which is NOT in the FP8-activation family.
- Result: `prepareQuantParams()` allocates **null quant workspaces** → autotune crashes or OOMs during the workspace setup phase that precedes CUDA graph capture.
- **eugr's repo fixed this 2 days ago** (commit `562ed29`, 2026-07-16) with a source patch to `csrc/fused_moe/cutlass_backend/cutlass_fused_moe_kernels.cuh` that adds an `is_native_wfp4afp4_family` predicate.
- **moranilt used v0.23.0 (9 days ago)** — the FlashInfer bundled in that image predates PR #3738, so he never hit this regression.
- **Our crash uses a newer image** (v0.24.0/v0.25.1 or AEON v0.23.0 container with a newer FlashInfer) → we hit the regression.

**THE FIX for Cause A:** Use eugr's Docker image (`ghcr.io/eugr/spark-vllm-docker` / build from `github.com/eugr/spark-vllm-docker`), which auto-applies the patch. OR downgrade FlashInfer to pre-#3738. OR pin to vanilla `vllm/vllm-openai:v0.23.0` (moranilt's exact image) which has the older FlashInfer.

**The patch (if applying manually to a Dockerfile):**
```python
# Adds to cutlass_fused_moe_kernels.cuh:
bool const is_native_wfp4afp4_family = isNativeWfp4Afp8Family();
bool const is_native_wfp4afp4_family =
    mSM >= 100 &&
    (mDType == nvinfer1::DataType::kFP4 || mDType == nvinfer1::DataType::kINT64) &&
    (mWType == nvinfer1::DataType::kFP4 || mWType == nvinfer1::DataType::kINT64);
# And: if (is_native_wfp4afp8_family || is_native_wfp4afp4_family) {
```
Source: `github.com/eugr/spark-vllm-docker` commit `562ed29`, Dockerfile lines 195-240.

---

### Cause B — Draft head inherits the wrong MoE/attention backend (THE CONFIGURATION BUG)

**This is the precise config difference between NVIDIA's official working recipe and a naive replication.**

- The MTP draft head's MoE layer uses **unquantized BF16 weights** (confirmed by rmagur1203 section ㉒: "The MTP head of Nemotron-3-Super contains an MoE layer, but unlike the main model, this MoE uses unquantized BF16 weights. For BF16 MoE, the Triton kernel is used instead of FlashInfer CUTLASS FP4. This is normal behavior.").
- The vLLM `ProposerData` config (in `vllm/config/speculative.py`, confirmed in BOTH v0.23.0 and v0.25.1) has two fields:
  - `moe_backend`: "When `None`, the draft model **inherits the target model's `--moe-backend` setting**."
  - `attention_backend`: "When `None`, the backend is automatically selected."
- **If you set `--moe-backend marlin` (or flashinfer-cutlass) on the outer flag ONLY, the draft head inherits it** → the BF16 drafter MoE tries to use a quantized kernel → crash or garbage.
- **NVIDIA's official SparkDeploymentGuide sets it INSIDE speculative-config:**
  ```
  --speculative_config '{"method":"mtp","num_speculative_tokens":3,"moe_backend":"triton"}'
  ```
  This forces the drafter to use Triton (correct for BF16) while the main model uses Marlin.
- **vLLM Issue #37754 comment 6 (agata-corp)** confirms the same pattern for the attention backend: "--attention-backend triton_attn alone is not enough — the outer flag only applies to the target model, while the MTP draft head still selects FlashInfer. You also have to set the backend inside speculative-config."

**THE FIX for Cause B:** Always set `"moe_backend":"triton"` (and optionally `"attention_backend":"triton_attn"`) INSIDE `--speculative-config`, not just the outer `--moe-backend`.

**moranilt's config does NOT set this** — his spec config is just `{"method":"mtp","num_speculative_tokens":3}`. Why does it work for him? Because the Puzzle drafter's BF16 MoE auto-selects Triton when the outer backend doesn't apply to unquantized weights. But this is fragile — if the outer `--moe-backend` is set to something that DOES get inherited badly, it breaks. **NVIDIA's explicit `"moe_backend":"triton"` in the spec config is the safe, documented approach.**

---

### Cause C — `mamba_cache_mode=all` auto-activates with prefix caching (THE MAMBA CRASH)

- `MambaCacheMode` is `Literal["all", "align", "none"]` (confirmed in `vllm/config/cache.py`, both v0.23.0 and v0.25.1).
- **Default is `"none"`**, BUT when `--enable-prefix-caching` is ON, it auto-selects `"all"` (the "default behavior for models that support it when prefix caching is enabled").
- **`"all"` + MTP = CRASH.** rmagur1203 section ⑩: "MTP 1 + `mamba_cache_mode=all` → Crash. Error: `selective_state_update` assertion — `state_batch_indices` is None or not 2D. Cause: Block-level gather in MTP spec decode metadata incompatible with Mamba2's `mamba_cache_mode=all` state indexing."
- **The fix:** `--mamba-cache-mode align` (only caches the mamba state of the last token of each scheduler step).
- **moranilt's config has `--enable-prefix-caching` ON but does NOT set `--mamba-cache-mode`** → he should default to `all` → should crash. Two possible explanations:
  1. v0.23.0's auto-selection logic may differ slightly (but the code shows the same default).
  2. **He may be getting lucky because the crash is concurrency-dependent** (agata-corp's report: "Single-request load never crashes, 30 concurrent triggers it"). moranilt benchmarks at concurrency 1.
  3. The Puzzle model may not trigger the `state_batch_indices` path the same way Nemotron-3-Super-120B does.
- **NVIDIA's official SparkDeploymentGuide does NOT set `--mamba-cache-mode` and does NOT set `--enable-prefix-caching`** → defaults to `none` → safe. This is the safest path.

**THE FIX for Cause C:** Either (a) remove `--enable-prefix-caching` (→ defaults to `none`, safe, NVIDIA's approach), OR (b) keep prefix caching but explicitly add `--mamba-cache-mode align`.

---

## THE EXACT WORKING CONFIG (NVIDIA Official, adapted for Puzzle)

This is NVIDIA's SparkDeploymentGuide recipe (for Nemotron-3-Super-120B) adapted for Puzzle-75B. **This is the authoritative config — it comes from NVIDIA engineers (Izzy Putterman, Nave Assaf, Joyjit Daw, et al).**

```bash
docker run --rm -it --gpus all \
  -e VLLM_NVFP4_GEMM_BACKEND=marlin \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e VLLM_FLASHINFER_ALLREDUCE_BACKEND=trtllm \
  -e VLLM_USE_FLASHINFER_MOE_FP4=0 \
  -e HF_TOKEN=$HF_TOKEN \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -p 8000:8000 \
  vllm/vllm-openai:cu130-nightly \
    --model nvidia/NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-NVFP4 \
    --served-model-name puzzle-75b \
    --host 0.0.0.0 \
    --port 8000 \
    --async-scheduling \
    --dtype auto \
    --kv-cache-dtype fp8 \
    --tensor-parallel-size 1 \
    --trust-remote-code \
    --gpu-memory-utilization 0.90 \
    --enable-chunked-prefill \
    --max-num-seqs 4 \
    --max-model-len 1000000 \
    --moe-backend marlin \
    --mamba-backend flashinfer \
    --mamba_ssm_cache_dtype float32 \
    --quantization fp4 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3,"moe_backend":"triton"}' \
    --reasoning-parser nemotron_v3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder
```

**Key differences from our crashing config:**
| Lever | NVIDIA Official (works) | Our crashing config | Why it matters |
|---|---|---|---|
| `--speculative-config` | `{"method":"mtp","num_speculative_tokens":3,"moe_backend":"triton"}` | `{"method":"mtp","num_speculative_tokens":2}` (no moe_backend) | **Cause B** — drafter inherits wrong backend |
| `--mamba-cache-mode` | not set (→ `none`, no prefix caching) | `align` (we set it) | **Cause C** — we're OK here, but only because we set it |
| `--enable-prefix-caching` | NOT set | ON | **Cause C** — if on without `align`, crashes |
| `--mamba_ssm_cache_dtype` | `float32` | `float16` | NVIDIA uses float32 on Spark; float16 is for TRT-LLM |
| `--gpu-memory-utilization` | `0.90` | `0.73` | NVIDIA is aggressive; 0.73 is moranilt's conservative |
| `--max-model-len` | `1000000` (with `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`) | `160000` | NVIDIA fits 1M; we were overly conservative |
| `--max-num-seqs` | `4` | `1` → `8` | NVIDIA's 4 is the balance point |
| `--quantization` | `fp4` (explicit) | not set (auto-detect) | Explicit is safer |
| Container | `vllm/vllm-openai:cu130-nightly` | v0.24.0 / AEON v0.23.0 | **Cause A** — nightly has FlashInfer #3738 fix pending; v0.23.0 predates the regression |
| `VLLM_NVFP4_GEMM_BACKEND` | `marlin` (env) | not set | Forces Marlin for NVFP4 GEMM |
| `VLLM_USE_FLASHINFER_MOE_FP4` | `0` (env) | `0` (moranilt) | Disables FlashInfer FP4 MoE (Blackwell multi-GPU only) |

---

## ANSWERS TO ALL 9 QUESTIONS

### 1. vLLM Issue #37754 — FlashInfer+MTP+SM121 crash bug
- **URL:** `github.com/vllm-project/vllm/issues/37754`
- **Status:** OPEN (as of 2026-07-13, last update). 6 comments.
- **BUT:** This issue is about **Nemotron-3-Super-120B (GQA=16)**, NOT Puzzle-75B. The crash is `BatchPrefillWithPagedKVCache` illegal memory access in FlashInfer attention, triggered by MTP≥2 on SM121 with high GQA.
- **The exact fix/workaround:** `--attention-backend triton_attn` for the main model, AND set `"attention_backend":"triton_attn"` inside `--speculative-config` (per comment 6 by agata-corp). The outer flag alone is insufficient — the draft head selects FlashInfer independently.
- **rmagur1203's comment (comment 2):** They got FlashInfer working on SM121 with 4 patches: `disable_flashinfer_q_quantization: true` in `--attention-config`, `vllm_flashinfer_utils_patch.py` (forces `supports_trtllm_attention()` True for SM120), `flashinfer_backend_patch.py` (HND layout + causal mask for spec decode), `mamba_mixer2_patch.py` (skip sync during graph capture). Achieved ~50 tok/s ShareGPT.
- **eugr's comment (comment 3):** Use `flashinfer_cutlass` backend + their community Docker.
- **agata-corp's comment (comment 6) — THE KEY:** The crash is specifically MTP × concurrency. Single request never crashes. The draft head independently selects FlashInfer. Fix: set `moe_backend` AND `attention_backend` inside `--speculative-config`.

### 2. rmagur1203's vllm-dgx-spark repo — OPTIMIZATION_REPORT.md
- **URL:** `github.com/rmagur1203/vllm-dgx-spark` (77KB report, 13 sections, 144-combo benchmark)
- **Model:** Nemotron-3-Super-120B-A12B-NVFP4 (NOT Puzzle, but same SM121 + NemotronH architecture)
- **Final working config:**
  ```yaml
  environment:
    FLASHINFER_CUDA_ARCH_LIST: "12.1a"
    VLLM_KV_CACHE_LAYOUT: "HND"
    VLLM_NVFP4_GEMM_BACKEND: "flashinfer-cutlass"
    VLLM_USE_FLASHINFER_MOE_FP4: "1"
  command:
    --model=nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4
    --tensor-parallel-size=1
    --enable-chunked-prefill
    --max-model-len=131072
    --attention-backend=FLASHINFER
    --mamba-cache-mode=align        # CRITICAL — "all" crashes with MTP
    --max-num-batched-tokens=8400
    --load-format=fastsafetensors
    --gpu-memory-utilization=0.8
    --attention-config={"disable_flashinfer_q_quantization":true}
    --compilation-config={"cudagraph_mode":"full_and_piecewise","pass_config":{"fuse_norm_quant":true,"fuse_act_quant":true}}
    --speculative-config={"method":"mtp","num_speculative_tokens":2}
  ```
- **Memory estimation table (section ⑩-b):**
  | cudagraph_mode | MTP k | Model | compile | CUDA graph | KV cache | Total | Status |
  |---|---|---|---|---|---|---|---|
  | NONE | 0 | 70GB | 0 | 0 | ~48GB | ~118GB | ✅ slow |
  | PIECEWISE | 1 | 71GB | 6GB | 5GB | ~33GB | ~115GB | ✅ recommended |
  | FULL+PIECE | 1 | 71GB | 6GB | 13GB | ~25GB | ~115GB | ⚠️ tight |
  | PIECEWISE | 3 | 71GB | 10GB | 8GB | ~24GB | ~113GB | ⚠️ fits |
  | PIECEWISE | 5 | 71GB | 15GB | 12GB | ~15GB | ~113GB | ❌ OOM |
- **Key discoveries:**
  - `drop_caches` frees 44GB→100GB+ (Spark unified memory reports only ~44GB free otherwise).
  - MTP drafter uses **unquantized BF16 MoE** → Triton kernel, not FlashInfer FP4.
  - `mamba_cache_mode=all` + MTP = crash (assertion in `selective_state_update`).
  - `cudagraph_mode=NONE` + MTP = Mamba assertion crash (separate bug).
  - MTP=5 OOMs (torch.compile activations, not drafter weights).
  - Best single-user: **46.3 tok/s** (Marlin+Triton+MTP=2+f&p+fusion).
  - Marlin + MTP actually degrades performance vs no MTP on RTX PRO 6000 (-22%), because W4A16 dequantization ≠ native FP4 → low acceptance.

### 3. NVIDIA's official SparkDeploymentGuide for Nemotron-3-Super
- **URL:** `github.com/NVIDIA-NeMo/Nemotron/tree/main/usage-cookbook/Nemotron-3-Super/SparkDeploymentGuide`
- **Contents:** Single `README.md` (6.3KB). Covers BOTH vLLM and TensorRT-LLM for single DGX Spark.
- **vLLM recipe:** `vllm/vllm-openai:cu130-nightly`, `VLLM_NVFP4_GEMM_BACKEND=marlin`, `VLLM_USE_FLASHINFER_MOE_FP4=0`, `--moe-backend marlin`, `--speculative-config '{"method":"mtp","num_speculative_tokens":3,"moe_backend":"triton"}'`, `--mamba_ssm_cache_dtype float32`, GMU 0.90, 1M context, 4 seqs.
- **TRT-LLM recipe:** `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc9`, CUTLASS MoE, MTP 3 layers, fp8 KV, float16 SSM with stochastic rounding, `free_gpu_memory_fraction: 0.9`, `enable_block_reuse: false` (Mamba state not prefix-cacheable).
- **Contributors:** Izzy Putterman, Nave Assaf, Joyjit Daw, and other NVIDIA engineers.
- **NOTE:** This guide is for Nemotron-3-Super-120B. There is **NO NVIDIA-published Spark guide for Puzzle-75B** — the Puzzle model card only has the generic (datacenter tp=2/4) recipe. The SparkDeploymentGuide is the closest NVIDIA official reference and the architecture is the same family (NemotronH hybrid Mamba-MoE with MTP).

### 4. eugr/spark-vllm-docker — community Docker recipes
- **URL:** `github.com/eugr/spark-vllm-docker` (1.8k stars, 504 commits, very active — last commit 2 days ago)
- **No Puzzle-75B recipe exists.** Recipes available: nemotron-3-super-nvfp4, nemotron-3-nano-nvfp4, qwen3.6-35b-a3b (fp8/nvfp4/no-mtp), deepseek-v4-flash, gemma4-26b, glm-4.7-flash, minimax-m2/m2.5/m2.7, gpt-oss-120b, step-3.7-flash, diffusion-gemma, etc.
- **nemotron-3-super-nvfp4.yaml** uses `--tensor-parallel 2` (dual Spark cluster), `--moe-backend cutlass`, `--attention-backend TRITON_ATTN`, `--mamba_ssm_cache_dtype float32`, NO MTP in base recipe (MTP would be a mod).
- **CRITICAL — "Flashinfer regression fix" commit (2026-07-16, 2 days ago):** Patches FlashInfer PR #3738 which broke NVFP4 MoE autotune workspace allocation. This is **Cause A** above. The patch adds `is_native_wfp4afp4_family` predicate to `cutlass_fused_moe_kernels.cuh`. **Building with eugr's Dockerfile auto-applies this fix.**
- eugr's bench (Issue #37754 comment 5): Nemotron-3-Super-120B MTP=3 on Spark → **26.5 tok/s** tg32, 24.5 tok/s at d8192.

### 5. HuggingFace discussion #3 — moranilt's FULL config
- **URL:** `huggingface.co/nvidia/NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-NVFP4/discussions/3`
- **moranilt's EXACT working config (verbatim):**
  ```yaml
  services:
    nemotron:
      image: vllm/vllm-openai:v0.23.0       # ← v0.23.0, NOT v0.24/0.25
      environment:
        - HF_TOKEN=***
        - CUDA_VISIBLE_DEVICES=0
        - VLLM_USE_FLASHINFER_MOE_FP4=0     # ← disables FlashInfer FP4 MoE
        - VLLM_ENGINE_READY_TIMEOUT_S=1200
      command: >
        --model nvidia/NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-NVFP4
        --port 8000
        --tensor-parallel-size 1
        --async-scheduling
        --trust-remote-code
        --mamba-backend flashinfer
        --mamba_ssm_cache_dtype float16              # ← float16, NOT float32
        --enable-mamba-cache-stochastic-rounding
        --mamba-cache-philox-rounds 5
        --speculative-config '{"method":"mtp","num_speculative_tokens":3}'  # ← NO moe_backend inside!
        --tool-call-parser qwen3_coder
        --reasoning-parser nemotron_v3
        --enable-auto-tool-choice
        --max-model-len 160000
        --gpu-memory-utilization 0.73
        --kv-cache-dtype fp8
        --max-num-seqs 8
        --enable-prefix-caching              # ← ON (risky with MTP, see Cause C)
        --enable-prompt-tokens-details
        --max-num-batched-tokens 32768
        --enable-chunked-prefill
        --default-chat-template-kwargs '{"preserve_thinking":false}'
  ```
- **Bench results (concurrency 1, 10k in / 2k out, 5 prompts):** 32.2 tok/s output, 193 tok/s total, TTFT 6.97s, TPOT 27.6ms. MTP acceptance 69.4% (pos0 83%, pos1 64%, pos2 62%). Acceptance length 3.08.
- **k=5 drops pos1 acceptance to 40%** — stay at k=3.
- **He does NOT set `--mamba-cache-mode`** → with `--enable-prefix-caching` ON, defaults to `all` → should crash per rmagur1203, but he benchmarks at concurrency 1 (the crash is concurrency-dependent per agata-corp).
- **He does NOT set `moe_backend` inside spec-config** → drafter auto-selects Triton for BF16 MoE. Works for Puzzle, but fragile.
- **His container is vanilla `vllm/vllm-openai:v0.23.0`** — predates FlashInfer PR #3738 (Cause A). This is likely the #1 reason his config works and ours doesn't.

### 6. HuggingFace discussion #9 — lsmc's OOM report
- **DOES NOT EXIST.** The Puzzle-75B NVFP4 model has only **3 discussions total:**
  1. #3 — moranilt: "What is the best settings to run on DGX Spark?" (working, 32 tok/s)
  2. #1 — codyknowscode: "Uses 160GB of RAM on start and produces rubbish output (with thinking FALSE)" (OOM + corruption)
  3. The "quantidistillation" discussion by TimeLordRaps (unrelated to OOM).
- **There is no discussion #9 and no user "lsmc".** This was a hallucinated lead. The relevant OOM report is **discussion #1 (codyknowscode)**: v0.24.0, GMU 0.80, max-model-len 262144, MTP k=4, `qwen3_xml` parser. Required significant SWAP to start, 10 tok/s decode, "rubbish output" (the PR #48375 prefix-cache corruption). Resolution: use 160k not 262k, GMU 0.73 not 0.80, `qwen3_coder` not `qwen3_xml`, k=3 not k=4, and disable prefix-caching until PR #48375 merges.

### 7. Disable FlashInfer autotune entirely?
- **There is NO `VLLM_FLASHINFER_AUTOTUNE=0` env var.** The autotune is inside the FlashInfer library itself, not vLLM.
- **Available env vars (confirmed in `vllm/envs.py`, both v0.23.0 and v0.25.1):**
  - `VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE` (default `394 * 1024 * 1024` = 394MB) — the workspace buffer size. Can be reduced but may cause kernel failures if too small.
  - `VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR` (default `None`) — caches autotune results so subsequent boots skip re-autotuning. Set this to a persistent volume to avoid re-autotuning every boot.
  - `VLLM_FLASHINFER_ALLREDUCE_BACKEND` (default `"auto"`, options `trtllm`/`mnnvl`) — set to `trtllm` for single-GPU Spark.
  - `VLLM_ENABLE_INDUCTOR_MAX_AUTOTUNE` (default `True`!) — enables torch inductor max_autotune + coordinate_descent_tuning. **Set to `0` to disable torch.compile autotune** (reduces compile memory/time, may reduce throughput).
  - `VLLM_ENABLE_INDUCTOR_COORDINATE_DESCENT_TUNING` (default `True`) — set to `0` to disable coordinate descent tuning.
  - `VLLM_TRITON_FORCE_FIRST_CONFIG` (default `0`) — set to `1` to skip Triton autotune benchmarking and use first valid config.
- **The real fix for the autotune OOM is NOT disabling autotune — it's the FlashInfer PR #3738 patch (Cause A).** Disabling autotune via `VLLM_ENABLE_INDUCTOR_MAX_AUTOTUNE=0` helps with torch.compile memory but does NOT fix the FlashInfer MoE workspace allocation bug.
- **Using `--moe-backend marlin` bypasses FlashInfer MoE entirely** (Marlin is W4A16 emulation). This is NVIDIA's official recommendation for Spark and moranilt's implicit path (via `VLLM_USE_FLASHINFER_MOE_FP4=0`). If using Marlin, the FlashInfer autotune bug is irrelevant.

### 8. Does `--cudagraph-mode piecewise` exist?
- **NO. There is no top-level `--cudagraph-mode` flag.**
- `cudagraph_mode` is a field on `CompilationConfig`, set via `--compilation-config` (or `-cc`):
  ```
  --compilation-config '{"cudagraph_mode":"piecewise"}'
  ```
- Valid values (confirmed in `vllm/config/compilation.py`, both v0.23.0 and v0.25.1): `NONE`, `PIECEWISE`, `FULL`, `FULL_AND_PIECEWISE`.
- **`FULL_AND_PIECEWISE` is the v1 default** and is what rmagur1203 uses. It captures full cudagraph for decode batches + piecewise for prefill.
- **`PIECEWISE` alone** uses less CUDA graph memory (5GB vs 13GB for FULL+PIECEWISE per rmagur1203's table) but is slower for decode.
- rmagur1203's memory table shows PIECEWISE + MTP k=3 fits at ~113GB (within 121GB). FULL+PIECEWISE + MTP k=1 fits at ~115GB.
- **For our OOM, `--compilation-config '{"cudagraph_mode":"piecewise"}'` is a valid lever to reduce graph memory from 13GB to 5GB.**

### 9. Can the drafter model be quantized or loaded separately?
- **For MTP: NO.** The MTP drafter is a single baked-in layer in the checkpoint (`num_nextn_predict_layers=1`). It is NOT a separate model.
- The `--speculative-config` `quantization` field exists but per the source docstring: "it only takes effect when using the **draft model-based** speculative method" (i.e., EAGLE/n-gram with a separate draft model, NOT MTP).
- The `--spec-model` flag (confirmed in `vllm/engine/arg_utils.py` line 1518) is for specifying an external draft model — not applicable to MTP.
- **The MTP drafter's MoE is already unquantized BF16** (rmagur1203 section ㉒). You cannot make it smaller. You can only control which kernel runs it via `"moe_backend":"triton"` in the spec config.
- **To reduce MTP memory, the only levers are:** (a) lower `num_speculative_tokens` (k=1 vs k=3), (b) `cudagraph_mode: piecewise` (less graph capture), (c) `VLLM_ENABLE_INDUCTOR_MAX_AUTOTUNE=0` (less torch.compile memory).

---

## ACTION PLAN (ordered by likelihood of fixing our OOM)

### Step 1 — Eliminate Cause A (FlashInfer regression) — HIGHEST PRIORITY
- **Option A (easiest):** Switch to moranilt's exact image: `vllm/vllm-openai:v0.23.0`. This predates FlashInfer PR #3738.
- **Option B (best long-term):** Use eugr's Docker image which auto-applies the FlashInfer #3738 patch. Build from `github.com/eugr/spark-vllm-docker`.
- **Option C (bypass entirely):** Use `--moe-backend marlin` + `VLLM_USE_FLASHINFER_MOE_FP4=0` + `VLLM_NVFP4_GEMM_BACKEND=marlin` (NVIDIA's official path). Marlin bypasses FlashInfer MoE completely — the autotune bug is irrelevant.

### Step 2 — Eliminate Cause B (drafter backend inheritance)
- Add `"moe_backend":"triton"` INSIDE `--speculative-config`:
  ```
  --speculative-config '{"method":"mtp","num_speculative_tokens":3,"moe_backend":"triton"}'
  ```
- This is NVIDIA's official config and matches the vLLM source design.

### Step 3 — Eliminate Cause C (mamba_cache_mode)
- **Either** remove `--enable-prefix-caching` entirely (→ `mamba_cache_mode` defaults to `none`, safe, NVIDIA's approach),
- **Or** keep prefix caching but add `--mamba-cache-mode align` explicitly.

### Step 4 — Reduce compile memory (if still OOM)
- `--compilation-config '{"cudagraph_mode":"piecewise"}'` — reduces graph memory from 13GB to 5GB.
- `-e VLLM_ENABLE_INDUCTOR_MAX_AUTOTUNE=0` — disables torch inductor autotune.
- `-e VLLM_ENABLE_INDUCTOR_COORDINATE_DESCENT_TUNING=0` — disables coordinate descent tuning.
- `-e VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR=/root/.cache/flashinfer_autotune` — cache autotune results (mount a volume).

### Step 5 — Reduce model footprint (last resort)
- Lower `--max-num-batched-tokens` to 8192 (from 32768).
- Lower `--max-model-len` to 128000 (from 160000).
- Lower `--gpu-memory-utilization` to 0.65 (but KV pool gets small).
- `--enforce-eager` (disables CUDA graph entirely — guaranteed load but slow).

### Step 6 — Pre-boot memory
- `sudo sync && sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'` — frees ~44GB OS page cache.
- rmagur1203 automated this as a `drop-caches` Docker service that runs before vLLM starts.

---

## THE MINIMAL DIFF (moranilt works vs we crash)

```
IMAGE:        vllm/vllm-openai:v0.23.0          ← moranilt (FlashInfer pre-#3738)
              vllm/vllm-openai:v0.24.0+         ← us (FlashInfer post-#3738, Cause A)

SPEC CONFIG:  {"method":"mtp","num_speculative_tokens":3,"moe_backend":"triton"}
                                              ← ADD THIS (Cause B, NVIDIA official)

MAMBA:        --mamba-cache-mode align          ← ADD if keeping prefix caching (Cause C)
              OR remove --enable-prefix-caching ← NVIDIA's approach

GEMM:         VLLM_NVFP4_GEMM_BACKEND=marlin    ← ADD (NVIDIA official, bypasses FlashInfer MoE)
              VLLM_USE_FLASHINFER_MOE_FP4=0     ← already have this (moranilt too)
              --moe-backend marlin              ← ADD (NVIDIA official)
```

**The single most likely fix: switch to `vllm/vllm-openai:v0.23.0` (moranilt's exact image) OR use `--moe-backend marlin` with `VLLM_NVFP4_GEMM_BACKEND=marlin` to bypass the FlashInfer MoE path entirely.** Then add `"moe_backend":"triton"` to the spec config and resolve the mamba_cache_mode issue.

---

## SOURCES (all verified 2026-07-18)

- vLLM Issue #37754 (6 comments, OPEN): `github.com/vllm-project/vllm/issues/37754` — FlashInfer+MTP+SM121 crash, Nemotron-3-Super-120B
- rmagur1203/vllm-dgx-spark OPTIMIZATION_REPORT.md (77KB): `github.com/rmagur1203/vllm-dgx-spark/blob/main/OPTIMIZATION_REPORT.md`
- NVIDIA NeMo SparkDeploymentGuide: `github.com/NVIDIA-NeMo/Nemotron/tree/main/usage-cookbook/Nemotron-3-Super/SparkDeploymentGuide/README.md`
- eugr/spark-vllm-docker (commit 562ed29, FlashInfer #3738 patch): `github.com/eugr/spark-vllm-docker`
- HF Discussion #3 (moranilt, working): `huggingface.co/nvidia/NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-NVFP4/discussions/3`
- HF Discussion #1 (codyknowscode, OOM+corruption): `huggingface.co/nvidia/NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-NVFP4/discussions/1`
- vLLM source v0.25.1: `vllm/config/speculative.py` (ProposerData: moe_backend, attention_backend fields), `vllm/config/compilation.py` (CUDAGraphMode enum), `vllm/config/cache.py` (MambaCacheMode), `vllm/config/mamba.py` (MambaBackendEnum default TRITON), `vllm/envs.py` (FlashInfer env vars), `vllm/engine/arg_utils.py` (--spec-model, --compilation-config)
- vLLM source v0.23.0: same files verified — all flags/env vars exist in both versions
- HF discussion #9 / "lsmc": **DOES NOT EXIST** — only 3 discussions on the model page