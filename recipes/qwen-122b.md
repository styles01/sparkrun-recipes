# Recipe: Qwen 3.5 122B DFlash (Updated July 24, 2026)

**Status:** ✅ PROD — n=7 + GMU 0.85 on aeon vLLM 0.23.0
**Served name:** `qwen`
**Docker image:** `ghcr.io/aeon-7/aeon-vllm-ultimate:2026-06-18-v0.23.0-dflashfix`
**Tool calling:** ✅ `--enable-auto-tool-choice --tool-call-parser qwen3_xml`
**Reasoning parser:** `qwen3`

## What Changed (July 24, 2026)

Applied lessons from Laguna S 2.1 deployment and community benchmarks.

| Setting | Before | After | Why |
|---|---|---|---|
| DFlash n | 12 | **7** | n=12 collapses on prose (9% acceptance, 15 tok/s). n=7 is the confirmed sweet spot (BlackwellBoy's 20-cell sweep + Poolside benchmarks). k≥8 collapses hard. |
| GMU | 0.82 | **0.85** | More KV headroom. 549K tokens (up from 426K, +29%). |
| KV cache dtype | bf16 | bf16 | fp8 KV + DFlash blocked by vLLM issue #41559 (non-causal attention incompatible with KV quantization). Not a local patch issue. |

## Model Location on Spark

```
~/.cache/huggingface/hub/models--bleysg--Qwen3.5-122B-A10B-int4-fp8-hybrid  (~67GB)
~/.cache/huggingface/hub/models--z-lab--Qwen3.5-122B-A10B-DFlash            (~1.5GB drafter)
```

## Config

| Parameter | Value |
|---|---|
| GPU mem util | 0.85 |
| KV cache dtype | bf16 (default) |
| DFlash n | 7 |
| Attention backend | flash_attn (target), FLASH_ATTN (drafter) |
| Max model len | 262,144 (256K native) |
| Max seqs | 3 |
| Max batched tokens | 8192 |
| Prefix caching | ON (with align-aware hash_block_size fix) |
| Load format | fastsafetensors |
| Served name | qwen |
| Port | 8000 |
| Tool parser | qwen3_xml |
| Reasoning parser | qwen3 |

## Memory & Performance (verified July 24, 2026)

| Metric | Value |
|---|---|
| Model load | 63.85 GiB, **36 seconds** (fastsafetensors) |
| KV cache | 30.98 GiB → **549,001 tokens** |
| Concurrency | **2.09× at 256K** |
| Prefill (pp512) | **827 tok/s**, 635ms TTFT |
| Decode (tg128) | **50.2 tok/s** (peak 58.3) |
| Free RAM | ~15 GB |

## Start Command

```bash
ssh jaita@larryspark.local 'cd ~/qwen3.5-122B-A10B-on-spark && \
  CTX=262144 GPU_MEM=0.85 MAX_NUM_SEQS=3 MAX_BATCHED_TOKENS=8192 \
  SERVED_NAME=qwen bash install.sh --start --profile dense --nspec 7 --no-smoke'
```

## Stop Command

```bash
ssh jaita@larryspark.local 'docker rm -f qwen-spark'
```

## Why n=7 (not n=12)

n=12 DFlash on Qwen 122B:
- Code: 7.7 accepted tokens/step → 82.8 tok/s ✅
- Prose: 2.1 accepted tokens/step → 15 tok/s ❌ (slower than no-spec baseline)
- Root cause: drafter can't predict 12 tokens ahead on open-ended prose

n=7 DFlash:
- Stable across workloads — no prose collapse
- BlackwellBoy's 20-cell sweep confirmed k=7 peaks at every concurrency level
- Poolside's benchmarks: ~5 accepted tokens/pass, 2.3-3.7× speedup
- k≥8 collapses hard (acceptance drops to ~0 past position 3)

## Why NOT fp8 KV

vLLM issue #41559 (OPEN): DFlash's non-causal attention requirement is incompatible with ALL KV cache quantization. No backend supports non-causal + fp8. This is an upstream vLLM architectural issue, not a local patch problem.

We attempted to port the aeon patches to vLLM 0.25.1 to enable fp8 KV. The patches worked (model loaded, fp8 KV accepted) but:
1. CUDA graphs were slower than eager mode for this workload
2. The KV pool with correct DFlash scaling was 561K (similar to aeon's 549K)
3. Load time was 415s vs 36s with aeon's fastsafetensors

Conclusion: aeon 0.23.0 with bf16 KV is the better config today. Revisit fp8 KV when vLLM adds non-causal fp8 support.

## Comparison to Previous Config

| Metric | n=12, GMU 0.82 (old) | n=7, GMU 0.85 (new) |
|---|---|---|
| KV tokens | 426K | **549K (+29%)** |
| Concurrency @ 256K | 1.62× | **2.09×** |
| Decode tok/s | 82.8 (code only) | 50.2 (stable across workloads) |
| Load time | ~8 min | **36 seconds** |
| Prose collapse | ❌ (9% accept, 15 tok/s) | ✅ fixed |

## Patches Applied (inside container)

1. `patch_fla_shmem.py` — FLA sm121 big-tile shmem fix (prefill/TTFT)
2. `patch_unify2.py` — scale-block unify for DFlash compatibility
3. `patch_prefix_align.py` — align-aware hash_block_size for prefix caching + DFlash
4. `patch_inc_hybrid.py` — hybrid INT4+FP8 dense-expert dispatch (dense profile)
5. `patch_int8_lmhead_v3.py` — int8 lm-head GEMV (dense profile)

## Serve Script

`~/qwen3.5-122B-A10B-on-spark/runtime/serve.sh` — driven by install.sh

## Research References

- BlackwellBoy DFlash sweep: https://x.com/Blackwellboy/status/2080309756317577509
- Poolside k=7 acceptance data: https://huggingface.co/poolside/Laguna-S-2.1-DFlash-NVFP4
- r/LocalLLaMA agentic eval: https://www.reddit.com/r/LocalLLaMA/comments/1v2ua8g/
- vLLM fp8 KV + DFlash blocker: https://github.com/vllm-project/vllm/issues/41559
- vLLM DFlash mixed attention: https://github.com/vllm-project/vllm/issues/40898
- Entrpi repo (base deployment): https://github.com/entrpi/qwen3.5-122B-A10B-on-spark
- Full fp8 KV research: `research/vllm-025-dflash-kv-cache-research.md`