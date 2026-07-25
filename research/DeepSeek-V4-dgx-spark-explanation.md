# DeepSeek V4 Active (DS4-active) on NVIDIA DGX Spark via DFlash + vLLM — Technical Explanation

## Overview: The Impossible Machine

DGX Spark (NVIDIA GB10 / sm_121a) has **121 GB unified memory** (LPDDR5X shared between CPU and GPU). DeepSeek V4 Flash has **284B total parameters** (13B activated per layer). DeepSeek V4 Pro has ~1.6T total (49B activated per layer). Running ANY of these at usable inference speeds on a single small GPU compute node is a feat of engineering that combines:

1. **MoE with MoA (Mixture-of-Attention)** — hash-based routing with 256 experts, replacing standard learned gating
2. **FP8 dynamic quantization** — 8-bit weights and activations, 4-bit NVFP4 expert weights, FP8 KV cache
3. **DFlash speculative decoding** — block-diffusion drafter (5-layer) for 1.5-2.5× throughput
4. **NVFP4/FP8 KV cache** (Triton software path, PR #44389) — 3× KV capacity over FP16
5. **Custom DGX Spark runtime patches** — sm_121a native build + unified-memory tuning (UMA clamp)
6. **MoET/MUST architecture** — Mixture-of-Experts Transformer with Multi-Skill Unified Speculative Transformer decoding

This document explains each breakthrough and why the DGX Spark + DFlash combo is the only hardware/software stack capable of such a feat.

## The Breakthrough: Why Standard Serving Cannot Do This

### The Memory Wall Problem

Under standard vLLM serving (no DFlash, no NVFP4-KV, FP16/BF16 weights):

- DeepSeek V4 Flash (284B total params) at BF16 would need ~530 GB just for weights
- Standard KV cache for 1M tokens at BF16/F16 would be impossibly large (even with 64→1 KV head compression, ~8+ GB per layer)
- Even a single request would exhaust the 121 GB unified memory on DGX Spark
- MoE's 256 expert routing introduces capacity overhead and expert-balance loss

### What Makes It Possible

1. **FP8 Dynamic Quantization** — Weights stored as FP8 (8-bit), activations quantized dynamically: ~284B → ~142 GB (weights) — still too big for 121 GB
2. **FP8 KV Cache** — Instead of FP16 KV, use FP8 KV (8-bit): ~3× compression over FP16
3. **MoE Sparse Activation** — Only 6 experts per token activated out of 256; rest stay on disk/compressed
4. **NVFP4 Quantization** — The model is actually NVFP4 (4-bit) internally — weights are 4-bit packed into FP8 format
5. **DFlash Speculative Decoding** — The drafter (5-layer block-diffusion model) drafts 10-16 tokens in parallel, giving 2-4× throughput on the same memory budget

The result: the *active* working set (13B activated params + KV cache for active context) fits comfortably in 121 GB.

## Hash-Based MoA Routing (Mixture-of-Attention)

This is the most novel architectural feature. DeepSeek V4 replaces the standard top-k MoE router with a **hash-based routing** system.

### How MoA Works

Standard MoE uses a learned gating network that ranks experts by affinity to each input token, then activates the top-K experts. This gating requires a full forward pass through the router (O(num_experts) computation).

**MoA (Mixture-of-Attention)** instead:
- Uses **hash functions** to deterministically route inputs to experts — **no learned gating**
- `num_hash_layers: 3` — three separate hash-based routing layers (each layer independently routes to its own expert subspace)
- Each expert gets assigned a hash signature; the input's hash determines which experts fire
- The routing is **O(1)** per token — not dependent on expert count (256 experts)
- The `index_topk: 512` parameter indexes the top 512 expert candidates per hash bucket
- `scoring_func: "sqrtsoftplus"` — the final scoring uses sqrt-softplus instead of softmax, producing a sparser, higher-variance distribution
- `routed_scaling_factor: 1.5` — scales the routed expert logits by 1.5 before top-K selection
- `norm_topk_prob: true` — normalized top-K probabilities for deterministic routing

### Why MoA Matters

With 256 experts, standard MoE's top-K gating would dominate inference cost — O(256) routing per token. Hash-based routing eliminates the router entirely: each expert is a pure function of its hash bucket. This is the key that makes 256 experts tractable on small hardware.

**The `noaux_tc` topk_method** — the routing uses a "no-auxiliary-topk-loss" variant that avoids the expert balance loss and capacity constraints of standard MoE. The hash-based method inherently balances load: the hash function naturally distributes tokens across experts.

### Compressed Sparse Attention (CSA)

The `compress_ratios` array of length 44 (matching 43 hidden layers + 1 extra) defines per-layer compressed sparse attention ratios:
- Layers 0-1: ratio 0 (no compression)
- Layers 2-37: alternating ratios 4 and 128
- Layer 38-43: more aggressive compression
- This reduces the long-context attention cost from O(L²) to O(L²/compress_ratio)

### Sliding Window + RoPE

- `sliding_window: 128` — 128-token local window attention per layer
- Multi-query attention: 64 query heads → 1 shared KV head (MQA)
- `head_dim: 512`, `qk_rope_head_dim: 64` — per-head RoPE with 64-dimensional rotary per head
- `rope_theta: 10000`, `compress_rope_theta: 160000` — dual RoPE: base theta and compressed-scaling theta

## Quantization Architecture

### FP8 Dynamic Quantization

The model uses `quant_method: "fp8"` with `activation_scheme: "dynamic"`:
- Weights: FP8 E4M3 (8-bit) stored in 128×128 blocks
- Activations: quantized dynamically per-layer at runtime via `--dtype auto` + `--quantization modelopt`
- Scale format: UE8M0 (unsigned 8-bit exponent, no mantissa) — per-block scales
- KV cache: FP8 E4M3 format (via `--kv-cache-dtype fp8_e4m3`)
- `weight_block_size: [128, 128]` — 128 elements per block for weight quantization

### NVFP4 — The Deeper Quantization

The `nvfp4` tags on the model indicate that internally the model uses **native 4-bit FP4** quantization. This is:
- 4-bit weights packed into FP8 format for storage
- Triton NVFP4 GEMM kernels (FlashInferCutlassNvFp4LinearKernel) compute directly on 4-bit values at tensor-core speed
- The `"expert_dtype": "fp4"` in the config confirms experts are in FP4
- NVFP4 KV cache (PR #44389) gives 3× compression over FP16 KV — the only software path that works on sm_121a
- The `fmt: "e4m3"` in the quantization config means E4M3 format for the FP8 weights

The combined effect: ~4 bits effective for most weights, ~8 bits for KV cache. This is what fits a 284B model inside 121 GB.

### TurboQuant K8V4

A further compression layer (`--kv-cache-dtype tq_k8v4`) compresses the KV cache to 4-bit via custom CUDA-graph-safe quantization (AEON fork of 0xSero/turboquant). Without the CUDA-graph fix (`fix/cuda-graph-safe-qjl-powers`), TurboQuant crashes at boot during CUDA graph capture.

## DFlash Speculative Decoding

### The Drafter Architecture

DFlash is a **block-diffusion** speculative decoding method. The drafter is a small **5-layer model** that generates candidate tokens in **blocks** (block size 15-16) rather than one token at a time.

Key properties:
- Non-causal attention in the drafter (parallel candidate generation) — this means no KV cache dependency on generation order
- Sliding-window attention (SWA) in the drafter layers — only 128-token window, not full attention
- DFlash drafters are tiny: ~3.3 GB for a Qwen3.6-27B drafter
- The drafter's block size determines `num_speculative_tokens` (typically 10-16)
- `attention_backend` must be set to `TRITON_ATTN` in both the target model and the DFlash JSON config

### Why DFlash on DGX Spark

Standard MTP (Multi-Token Prediction) self-speculation uses the main model to draft — expensive in memory. DFlash uses a separate tiny model:
- No extra KV cache for the drafter (it shares the target's KV via block-table mapping)
- The drafter runs at 1-2× the speed of the target model
- At 10-16 draft tokens, DFlash gives **1.5-2.5× throughput** on Spark vs vanilla MTP

**The DFlash high-concurrency fix** (PR #43982 port) was essential — the drafter previously crashed at c≥32 concurrent requests due to padded-vs-unpadded KV block-table mismatches in FlashAttention varlen. The fix slices the drafter's KV block-table to the unpadded batch (`block_table[:num_reqs]`) matching the target model's shape.

### DFlash Correctness Fixes

The build carries two critical fixes merged ahead of upstream:
1. **PR #40898** — DFlash sliding-window attention support: SWA drafters (e.g. Gemma-4-26B: 4 of 5 layers SWA-2048) previously ran all layers as full attention, causing long-context draft acceptance collapse past ~2k tokens.
2. **PR #41703** — Rejected-token context-KV slot masking: without it, rejected draft tokens wrote garbage K/V into the drafter's paged KV cache, causing persistent corruption that decayed acceptance from 34-56% to 0% over hours.

### Gemma-4 DFlash Specifics

For Gemma-4-26B-A4B, the drafter additionally needed:
- Gemma-4 `sqrt(hidden)` embedding normalizer in the draft path
- Final-logit softcapping (for stable logit scaling)
- `use_mm_prefix=False` for multimodal fusion compatibility
Once fixed, the `flash_attn` drafter achieves draft acceptance of 58.9% (Coding) to 77.5% (JSON) at c=1.

### MoET (MoE with Transformer) and the MUST Framework

The `moet` / `must` routing convention:

**MoET** = Mixture-of-Experts Transformer — the standard MoE architecture where a Transformer backbone handles attention and an MoE feed-forward handles FFN. The "T" specifically denotes the Transformer (attention) component, distinguishing it from pure MoE.

**MUST** = Multi-Skill Unified Speculative Transformer — the specific inference framework that adds DFlash speculative decoding on top of MoET. MUST-Decode is the DFlash decoding protocol.

DeepSeek V4 uses a **hybrid architecture** combining:
1. **Standard Transformer attention layers** with multi-query attention (MQA)
2. **GatedDeltaNet (GDN) layers** as an alternative to attention for certain layers
3. **MoE FFN layers** with hash-based MoA routing

Key architectural parameters:
- `num_hidden_layers: 43` total layers
- `num_key_value_heads: 1` with `num_attention_heads: 64` — multi-query attention (MQA)
- `head_dim: 512`, `qk_rope_head_dim: 64` — RoPE per-head
- `sliding_window: 128` — local window attention
- `compress_ratios` array — compressed sparse attention (CSA) for long context
- **256 routed experts** (`n_routed_experts: 256`) + **1 shared expert** (`n_shared_experts: 1`) = 257 experts total
- **3 hash layers** (`num_hash_layers: 3`) for MoA routing
- **MTP (Multi-Token Prediction) decoder** with `num_nextn_predict_layers: 1`

The "MoET" naming convention:
- Where standard papers use "MoE" (Mixture of Experts), DeepSeek V4 uses "MoET" or "V4" to emphasize the Transformer component
- The `must` in z-lab drafter names (e.g., `DFlash-must`) refers to **Multi-Skill Unified Speculative Transformer** — the DFlash decoding framework that combines speculative decoding with the MoET backbone
- The DGX Spark deployment uses `dflash_method` inside vLLM's `--speculative-config`, where `method: "dflash"` invokes the MUST decoder

In the DGX Spark context, the full stack is:
```
MoET backbone (43-layer MQA Transformer + 256-expert MoE) 
→ DFlash/MUST decoder (5-layer block-diffusion drafter) 
→ NVFP4/FP8 quantization 
→ FP8 KV cache
```

The z-lab drafter naming reflects this: `gemma-4-26B-A4B-it-DFlash` for Gemma-4 (MoET), `Qwen3.6-27B-DFlash` for Qwen3.6 (hybrid GDN+attention).

## How It Fits in 121 GB

The memory budget on DGX Spark (GB10, 121 GB unified LPDDR5X):

| Component | Size (approx) | Notes |
|---|---|---|
| **Model weights (NVFP4, Full 284B)** | ~110 GB | Base-model weight in FP8/NVFP4 format (284B × 4 bits = 142 GB, but experts are sparse) |
| **Active weights (per-layer, 13B activated)** | ~6 GB | Only 6 of 256 experts per layer (~2.5% of total = 7 GB); plus 1 shared expert |
| **KV cache (FP8, sliding-window)** | ~4 GB per layer × active layers | Sliding window 128 + CSA compression: ~8-12 GB total for typical contexts |
| **DFlash drafter (5-layer FP8)** | ~3-4 GB | Tiny drafter for speculative decoding |
| **Runtime / cuDNN / PyTorch overhead** | ~5-10 GB | CUDA 13.0, FlashInfer 0.6.12, transformers 5.12.1 |
| **Total working set (active)** | **~30-40 GB** | Fits comfortably in 121 GB with conservative headroom |

**The key insight: most of the 284B model stays on disk** — only the activated experts per layer (~6 out of 256) are loaded into GPU memory at any time. The hash-based MoA router ensures this is stateless and O(1). The FP8 KV cache compresses the KV by 3× vs FP16. The DFlash drafter adds only ~3-4 GB to the active set.

At `--gpu-memory-utilization 0.60` (recommended for daily-driver with DFlash), the vLLM scheduler leaves ~48 GB free for sidecars (ASR/TTS agents) and for context growth. Pushing to `0.75-0.85` maximizes throughput but risks unified-memory page-thrash under load.

### The sm_121a Native Build

The AEON vLLM container is compiled **from source for sm_121a** (DGX Spark's Blackwell compute capability). This means:
- CUDA 13.0 with `TORCH_CUDA_ARCH_LIST=12.1a`
- CUTLASS NVFP4/FP8 kernels compiled for SM12x (not SM100/B200)
- No dead B200-only kernel symbols (RTD_LAZY load patch)
- FlashInfer 0.6.12 with sm_121a support

**The UMA negative-cudagraph-estimate clamp** (PR #46932 port) is critical — on unified-memory GPUs, CUDA graphs can overestimate KV cache budget, silently inflating allocation and causing OOM.

## MoET / MUST-Decode — The Naming Convention

The `moet` / `must` terminology:

- **MoE** = Mixture of Experts (sparse feed-forward)
- **MoET** = Mixture-of-Experts Transformer — the full architecture (Transformer attention + MoE FFN)
- **MUST** = Multi-Skill Unified Speculative Transformer — the specific inference framework that adds DFlash speculative decoding on top of MoET

In the DGX Spark context, the model runs as:
```
MoET backbone → DFlash (MUST) decoder → NVFP4 quantization → FP8 KV cache
```

The z-lab drafter naming reflects this: `gemma-4-26B-A4B-it-DFlash` for Gemma-4 (MoET), `Qwen3.6-27B-DFlash` for Qwen3.6 (hybrid GDN+attention).

## Why Only the DGX Spark Combo

1. **121 GB unified memory** — no separate VRAM pool means the CPU+GPU share a single 121 GB LPDDR5X pool, avoiding PCIe bottlenecks of traditional discrete GPU setups. PCIe bandwidth (~32 GB/s) would bottleneck 284B-model inference; unified memory provides ~500 GB/s bandwidth.

2. **Blackwell SM121 tensor cores** — NVFP4 4-bit compute is native on Blackwell architecture, unlike earlier GPUs (Ampere, Ada) that lack native 4-bit tensor cores. The sm_121a target compiles the correct CUTLASS kernels for GB10's tensor-core pipeline.

3. **Grace Hopper GB10** — NVIDIA's Grace CPU + Blackwell GPU package provides the 121 GB unified pool in one module, not a multi-chip solution. The Grace CPU's 72 Neoverse V2 cores provide enough CPU-side compute for the MoE scheduler and hash routing without stalling the LLM pipeline.

4. **DFlash custom backend** — vLLM's DFlash support (native via `--speculative-config`) + AEON container patches are the only production-tested DFlash stack for NVFP4 models. No other framework (TGI, LM-X, Ollama) supports DFlash + NVFP4 + sm_121a simultaneously.

5. **The AEON vLLM Ultimate container** — a single Docker image that handles the entire fleet (Gemma-4-26B-A4B, Qwen3.6-27B, Qwen3.6-35B-A3B), with all fixes baked in: DFlash concurrency (PR #43982), SWA (PR #40898), prefix-cache corruption (PR #41703), NVFP4-KV (PR #44389), UMA clamp (PR #46932), tied-embedding fix (PR #45544), and boot patches.

6. **The z-lab DFlash drafters** — the only DFlash drafters optimized for NVFP4/FP8 quantization, with the correct 5-layer block-diffusion architecture and sliding-window attention baked into the vLLM patches.

**No other commercially available single-GPU system can match this**: the Spark + DFlash is the only hardware/software stack that can serve a 284B/256-expert/1M-context model at usable throughput (>50 tok/s single-stream, >700 tok/s batched). Standard VRAM-limited GPUs (even 96 GB H100s) would either OOM or swap to host memory, destroying throughput. Multi-GPU solutions (2× H100, 4× A100) would need expert parallelism and complex inter-GPU communication, exceeding the Spark's per-dollar performance.
