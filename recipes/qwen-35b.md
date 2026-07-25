# Recipe: Qwen 3.6 35B NVFP4

**Status:** Community tested (Recipe A — 104 benchmark runs). Our Recipe C untested.
**Served name:** `qwen35b`
**Docker image:** `vllm/vllm-openai:v0.24.0`
**Tool calling:** ✅ `--enable-auto-tool-choice --tool-call-parser qwen3_xml`
**Reasoning parser:** `qwen3`

## Model Location on Spark

```
~/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/*/
```
~18GB, 3 safetensors shards. Already downloaded.

## MTP Bug — FIXED with PR #48375 (July 11, 2026)

**MTP k=3 + prefix caching on Qwen 3.6 (hybrid Mamba) corrupted the recurrent state cache.**

Known vLLM bug (issues #43559, #47194, #47087, PR #48375 opened July 11 2026).
- `MambaManager.find_longest_cache_hit()` ignored `drop_eagle_block` → corrupted state persisted
- Spread via prefix caching → every request sharing prefix restored from corrupted state
- Symptoms: malformed XML tool calls, accuracy drops 94%→75%, garbage tokens

**Fix applied: PR #48375 patch mounted into container at startup.**
- Patch file: `~/patch_mamba_drop_eagle.sh` on Spark
- Applied before vLLM starts via `--entrypoint bash -c "bash /patch... && vllm serve..."`
- 3-line change: `if drop_eagle_block and max_num_blocks > 0: max_num_blocks -= 1`
- MTP k=3 + prefix caching now both ON with correct behavior
- When PR #48375 merges upstream, remove the patch from the switch script

**This gives us: ~100 tok/s + correct tool calls + prefix caching. Best of all worlds.**

## Locked Config (Recipe A — MTP REMOVED, July 11 2026)

```bash
docker run --gpus all -d --name qwen35b-spark \
  --network host --ipc host \
  -e VLLM_MARLIN_USE_ATOMIC_ADD=1 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:v0.24.0 \
  nvidia/Qwen3.6-35B-A3B-NVFP4 \
  --served-model-name qwen35b \
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 1 --trust-remote-code \
  --kv-cache-dtype fp8 \
  --attention-backend flashinfer \
  --moe-backend marlin \
  --gpu-memory-utilization 0.65 \
  --max-model-len 262144 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 32768 \
  --enable-chunked-prefill \
  --async-scheduling \
  --enable-prefix-caching \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3,"moe_backend":"triton"}' \
  --load-format fastsafetensors \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_xml \
  --enable-auto-tool-choice
```

## Performance (Recipe A, community benchmarked)

| Metric | Value |
|---|---|
| Single-user tok/s | 109.3 |
| c2 aggregate | 151.6 |
| c5 aggregate | 175.5 (peak 276) |
| c10 aggregate | 165 (peak 292) |
| 8K context, c1 | 113.2 tok/s |
| 16K context, c2 | 178.3 tok/s |
| Prefill (2K, c1) | 5529 tok/s, TTFT 373ms |

## Recipe C — Headless Maxout (UNTESTED, our addition)

Same as Recipe A but:
- GMU 0.85 (headless, GNOME stopped)
- max-num-seqs 32 (max concurrency)
- max-num-batched-tokens 16384
- `--no-async-scheduling` (DocAI finding)
- `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1`
- Custom chat template from MiaAI-Lab repo (vision + thinking + tools)

## Notes

- First boot: FlashInfer autotune ~15 min (cached after)
- TP=2 fails on GB10 (single GPU) — always TP=1
- For long-context prefill (8K+ inputs, 4+ concurrent): remove `--speculative-config`
- If GNOME running: use GMU 0.4 (Mia's default)
- MiaAI-Lab repo NOT yet cloned on Spark — needed for Recipe B/C chat template
  - `git clone https://github.com/MiaAI-Lab/Qwen3.6-35B-A3B-NVFP4-vLLM.git`

## A/B Test Plan (Pending)

Three recipes to compare on our Spark:

### Test A: Recipe A (current — LOCKED, tested)
- Model: `nvidia/Qwen3.6-35B-A3B-NVFP4`
- GMU 0.65, 4 seqs, MTP k=3, flashinfer, marlin, fp8 KV
- Result: **102.8 tok/s single-stream, 71% MTP acceptance**

### Test B: AEON-7 Heretic + DFlash n=11
- Model: `AEON-7/Qwen3.6-35B-A3B-heretic-NVFP4`
- Drafter: `z-lab/Qwen3.6-35B-A3B-DFlash`
- Docker: `ghcr.io/aeon-7/aeon-vllm-ultimate:latest`
- GMU 0.85, DFlash n=11, triton_attn, fp8_e4m3 KV
- qwen3_coder, qwen3
- Expected: 116 tok/s single, 575 tok/s peak (C64), prose 78 tok/s
- Needs: model + drafter download, aeon-vllm image (may already have it from 122B)

### Test C: Recipe A+ (emX0r-style, MTP k=3, optimized)
- Same model: `nvidia/Qwen3.6-35B-A3B-NVFP4`
- Bump: GMU 0.80, 6-8 seqs, `VLLM_MARLIN_USE_ATOMIC_ADD=1`
- Keep: MTP k=3, flashinfer, marlin, fp8 KV, qwen3_xml
- Expected: ~100 tok/s single, better concurrency (270→320 tok/s at 4-6 streams)

### Why A/B Test
- Test B uses DFlash n=11 — which on 35B does NOT collapse on prose (78 tok/s vs 122B's 15 tok/s)
- Test B uses AEON-7's heretic quant — may be higher quality
- Test C is the safe optimization of our current recipe — more KV headroom, more concurrency
- Need to measure all three on: single-stream tok/s, prose tok/s, tool call success, Loca agent loop

## Economic Case

At 109 tok/s single-user with 4 concurrent, this model could replace Ollama for most agent workloads.
- Ollama subscription: $150/mo → $20/mo (downgrade)
- Concurrency: 4 agents on Spark vs 1 on Ollama
- Speed: 109 tok/s vs ~14 tok/s (Ollama 122B)
- Tradeoff: slightly less smart than 122B (91.0 vs 122B's frontier-level)