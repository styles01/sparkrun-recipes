# Qwen 3.5 122B DFlash on DGX Spark — n=7 DFlash + GMU 0.85 Recipe

**Forum:** NVIDIA DGX Spark Community
**Author:** styles01
**Date:** July 24, 2026

---

I've been running Qwen 3.5 122B-A10B as my daily driver on a DGX Spark (GB10, 121GB unified memory). After also deploying Laguna S 2.1 and learning from community benchmarks (BlackwellBoy's 20-cell DFlash sweep), I've updated my Qwen recipe with two optimizations that meaningfully improve the experience.

## Hardware

- NVIDIA DGX Spark (GB10, sm_121, 121GB unified memory)
- vLLM via Docker (`ghcr.io/aeon-7/aeon-vllm-ultimate:2026-06-18-v0.23.0-dflashfix`)

## Model

- **Target:** `bleysg/Qwen3.5-122B-A10B-int4-fp8-hybrid` (67GB, hybrid INT4+FP8 + int8 lm-head)
- **Drafter:** `z-lab/Qwen3.5-122B-A10B-DFlash` (1.5GB, EAGLE-style)
- **Architecture:** 122B total, 10B active per token, 40 layers (hybrid Mamba/attention), MoE

## What Changed

### 1. DFlash n=7 (was n=12)

This is the fix for DFlash's biggest weakness: **prose collapse**.

At n=12, DFlash drafts 12 tokens in parallel. On code and tool calls, the drafter predicts 7+ consecutive tokens — excellent, 82.8 tok/s. But on prose (creative writing, philosophical discussion, general conversation), acceptance collapses to 9% and throughput drops to ~15 tok/s — *slower than no-spec baseline*.

Dropping to n=7 fixes this. The draft is shorter, so the drafter doesn't need to predict as far ahead. BlackwellBoy independently confirmed k=7 as the optimum across a 20-cell sweep (k ∈ {5,6,7,8,9} × seqs ∈ {4,8,16,32}) — it peaked at every concurrency level. Poolside's own benchmarks show k=7 gives ~5 accepted tokens/pass with 2.3-3.7× speedup.

**Key finding from the sweep:** k≥8 collapses hard. DFlash per-position acceptance drops to ~0 past position 3, so deeper drafts burn compute for nothing. k=7 is the ceiling.

### 2. GMU 0.85 (was 0.82)

Pushing GPU memory utilization from 0.82 to 0.85 gives more KV pool. KV tokens went from 426K to 549K (+29%), concurrency from 1.62× to 2.09× at 256K.

## Final Config

```bash
CTX=262144 GPU_MEM=0.85 MAX_NUM_SEQS=3 MAX_BATCHED_TOKENS=8192 \
SERVED_NAME=qwen bash install.sh --start --profile dense --nspec 7
```

| Parameter | Value |
|---|---|
| GPU memory utilization | 0.85 |
| KV cache dtype | bf16 |
| DFlash n | 7 |
| Attention backend | flash_attn |
| Max model len | 262,144 (256K) |
| Max concurrent seqs | 3 |
| Max batched tokens | 8192 |
| Prefix caching | ON (with align-aware hash_block_size fix) |
| Load format | fastsafetensors |
| Tool parser | qwen3_xml |
| Reasoning parser | qwen3 |

## Verified Performance (llama-benchy)

| Test | tok/s | TTFT |
|---|---|---|
| Prefill (pp512) | **827 tok/s** | 635ms |
| Decode (tg128) | **50.2 tok/s** (peak 58.3) | — |

## Memory Breakdown

| Component | Value |
|---|---|
| Model weights | 63.85 GiB |
| KV cache pool | 30.98 GiB (549,001 tokens) |
| Concurrency at 256K | 2.09× |
| Load time | **36 seconds** (fastsafetensors) |
| Free RAM | ~15 GB |

## What I Learned From Laguna

I also ran Poolside's Laguna S 2.1 (118B MoE, NVFP4) on the same Spark. It's a better coding model (70.2% vs 49.4% on Terminal-Bench 2.1) with the best tool calling I've seen locally. But it invents facts under pressure — 3 confirmed fabrications in a grounding eval vs Qwen's zero across 240 runs.

The Laguna deployment taught me:
- k=7 is the DFlash sweet spot (independently confirmed by multiple benchmarkers)
- fp8 KV cache is safe and effective on GB10 (for non-DFlash models)
- FlashInfer has lower graph cost than flash_attn on sm_121

I attempted to port the Qwen-specific patches to vLLM 0.25.1 to enable fp8 KV cache. The patches worked — model loaded, fp8 KV accepted, DFlash resolved. But fp8 KV + DFlash is architecturally blocked by vLLM issue #41559: DFlash's non-causal attention is incompatible with KV cache quantization. No backend supports non-causal + fp8. This is an upstream issue, not a local patch problem.

## Why Qwen Over Laguna

Laguna is a better coding model. Qwen has better grounding discipline. For a daily driver touching real agent workloads, grounding matters more than benchmark scores. Qwen has never fabricated in 240 grounding runs; Laguna had 3 confirmed fabrications. That's the property that actually matters for autonomous agents.

## Credits

- **entrpi/qwen3.5-122B-A10B-on-spark** — the base deployment repo, Docker image, and serve script with all the Qwen-specific patches
- **bleysg** — the hybrid INT4+FP8 dense checkpoint
- **z-lab** — the DFlash drafter
- **BlackwellBoy** (@Blackwellboy on X) — the 20-cell DFlash sweep that confirmed k=7
- **Poolside** — their benchmark data validated k=7 acceptance rates
- **r/LocalLLaMA** community — the private agentic eval that confirmed Qwen's grounding discipline

## Patches Applied (inside container)

The serve script applies several runtime monkeypatches before `vllm serve`:

1. `patch_fla_shmem.py` — FLA sm121 big-tile shmem fix (prefill/TTFT)
2. `patch_unify2.py` — scale-block unify for DFlash compatibility
3. `patch_prefix_align.py` — align-aware hash_block_size for prefix caching + DFlash
4. `patch_inc_hybrid.py` — hybrid INT4+FP8 dense-expert dispatch (dense profile)
5. `patch_int8_lmhead_v3.py` — int8 lm-head GEMV (dense profile)

These are Qwen 122B-specific and required for DFlash + prefix caching to work together. Don't skip them.

Happy to answer questions or share the full serve script.