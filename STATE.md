# STATE — Spark LLM Current State

**Last updated:** July 25, 2026 by Oracle (vLLM v26 main + fp8 KV + DFlash n=7 + int8_lmhead — **BREAKTHROUGH**: 1.37M KV tokens, 5.24× concurrency, 45.98 tok/s decode, 957 tok/s prefill, ~16 GB headroom for 3 lanes @ 256K)
**Update this file after EVERY switch, test, benchmark, or crash.**

## What's Running NOW

**Qwen 122B (vLLM v26 + fp8 KV + DFlash n=7 + int8_lmhead)**
- **Container:** `qwen-spark`
- **Port:** 8000
- **Model:** `qwen-122b`
- **vLLM:** v26 main (commit 318b527, built from source for SM121)
- **KV cache:** fp8, 1,372,342 tokens capacity
- **Concurrency:** 5.24× at 256K context
- **GMU:** 0.85
- **Lanes:** 3 (max-num-seqs=3)
- **Context:** 262,144 tokens
- **Decode:** 45.98 tok/s (peak 56.0)
- **Prefill:** 957 tok/s
- **int8 lm-head:** ✅ Applied, 1.4 GB freed
- **Started:** July 25, 2026 14:46 UTC
- **Status:** **PRODUCTION READY**

### Headroom Analysis (3 Lanes @ 256K)

| Resource | Total | Used | Free |
|---|---|---|---|
| **Unified Memory** | 121 GB | 105 GB | **~16 GB** |
| **KV Pool** | 1,372,342 tokens | 768K (3×256K) | **604K tokens** |
| **GPU Memory** | 121 GB | 102.85 GB | **~18 GB** |

**Breakdown:**
- Model weights: 63.97 GB
- KV cache pool (reserved): 35.55 GB
- Activations: ~5-10 GB (dynamic)
- int8 lm-head saved: 1.4 GB
- CUDA graphs: 0.02 GB

**Headroom for 3 lanes @ 256K:**
- Actual KV usage: 768K tokens (56% of pool)
- Free RAM: **~16 GB** for co-location (TTS, embeddings, etc.)
- Can scale to **5 concurrent 256K sessions** before hitting limits

---

The NGC image `nvcr.io/nvidia/vllm:26.06-py3` auto-selects `FLASHINFER_CUTLASS` MoE backend with `enable_flashinfer_autotune=False`. Our previous attempts on `vllm/vllm-openai:v0.24/v0.25` used TRTLLM backend with autotune ON, burning 20GB+ in workspace allocation before CUDA graph capture. The NGC image uses 0.33GB for CUDA graphs instead. Community-validated by joeynyc (GitHub repo), VramJon (NVIDIA forum), brian322 (NVIDIA forum).

### 35b (Qwen 3.6 35B NVFP4)

| Field | Value |
|---|---|
| **Status** | ✅ RUNNING NOW (media-server config, Jul 13) |
| **Our recipe** | `recipes/qwen-35b.md` (Recipe A) |
| **Tool calls** | ✅ `qwen3_xml` parser — tested and working |
| **Reasoning** | ✅ `qwen3` parser — reasoning_content extracted |
| **Config** | GMU 0.40, 6 seqs, 256K ctx, MTP k=3, flashinfer_b12x, fp8 KV, PR #48375 patch |
| **Architecture** | MoE: 35B total, 3B active/token (8 of 256 experts + shared), 40 layers (hybrid Mamba/attention), NVFP4 |
| **Memory used** | ~76GB (22GB weights + 21GB CUDA graphs + 39GB KV + 1GB overhead) |
| **Free RAM** | ~45GB (for co-located video gen, TTS, image gen, GNOME) |
| **KV cache** | 3.26M tokens (12.45x max concurrency at 256K) |
| **Max concurrent** | 6 lanes |
| **Benchmark** | 102.8 tok/s (code gen), 74% MTP acceptance — `benchmarks/35b-2026-07-11.md` |
| **Notes** | MTP k=3 + prefix caching + PR #48375 Mamba bugfix + flashinfer_b12x MoE. Real agent traffic: ~20s turns, 30-60 tok/s. Intelligence gap vs 122B noticeable — "teenager vs adult." **Hallucination and gaslighting significantly worse than DS4.** Not suitable as daily driver for agent workloads requiring reliability. **Two scripts: switch-to-35b.sh (media-server, GMU 0.40) and switch-to-35b-production.sh (max KV, GMU 0.65, no co-location).** |

### 122b (Qwen 3.5 122B DFlash)

| Field | Value |
|---|---|
| **Status** | ✅ RUNNING NOW (n=6 config, Jul 13) |
| **Our recipe** | `recipes/qwen-122b.md` |
| **Tool calls** | ✅ `qwen3_xml` parser — tested and working |
| **Reasoning** | ✅ `qwen3` parser — reasoning_content extracted |
| **Config** | DFlash n=6, GMU 0.83, 150K ctx, 3 seqs |
| **Architecture** | MoE: 122B total, 10B active/token (A10B), 40 layers hybrid Mamba/attention, INT4+FP8 |
| **Memory used** | ~104GB (64GB weights + 0.59GB graphs + 28.4GB KV + 11GB drafter/OH) |
| **Free RAM** | ~16GB (for one media workload: image gen OR TTS) |
| **KV cache** | 457,180 tokens (3.05x at 150K) |
| **Max concurrent** | 3 lanes |
| **KV pool** | 457K tokens (3×150K fits, 3×165K does NOT) |
| **Benchmark** | 82.8 tok/s (n=12 code gen) — n=6 TBD |
| **Notes** | Start: `install.sh --start --profile dense --nspec 6`. Stop: `docker rm -f qwen-spark`. Docker containment. DFlash graph cost is ~0.6GB regardless of n (separate drafter model). See `benchmarks/122b-memory-audit-2026-07-13.md` for full audit. |

### ds4 (DeepSeek-V4-Flash)

| Field | Value |
|---|---|
| **Status** | ✅ LAST-KNOWN-GOOD (locked July 11, 2026) |
| **Verified by** | Oracle (July 11, 2026) |
| **Our recipe** | `recipes/deepseek-v4-flash.md` |
| **Tool calls** | ✅ `deepseek_v4` parser — tested and working |
| **Last worked** | July 11, 2026 |
| **Benchmark** | 21.0 tok/s (code gen, 407 tokens) |
| **Benchmark file** | `benchmarks/ds4-2026-07-11.md` |
| **Config** | 128K ctx, 2 concurrent, GMU 0.78, MTP k=2, cgroup MemoryMax=110G, reasoning_effort=high |
| **Architecture** | MoE: 159B total, ~11B active/token (6 routed + 1 shared of 257 experts), 43 layers, FP8 |
| **16hr avg (Jul 13, reasoning OFF)** | 32.7 tok/s overall (4,372 samples) — bimodal: 40+ tok/s idle/script, 11-18 tok/s agent load |
| **16hr avg (Jul 13, reasoning ON)** | Content: 26-30 tok/s | Thinking: 1-8 tok/s | MTP accept 60-84% content, 43-58% reasoning |
| **Prefix cache hit rate** | 85.5-86.1% (stable across all loads) |
| **KV cache usage** | 0-16% (massive headroom) |
| **MTP acceptance** | 62-93% (content), 43-58% (reasoning phase) |
| **Active params/token** | ~11B (7 of 257 experts: 6 routed + 1 shared) |
| **Notes** | Serve script patched with tool calling. Backup at `.bak`. Switch script updated with orphaned scope cleanup. Reasoning was OFF until Jul 13 — caused "fucking stuff up." Fixed by adding `reasoning_effort: high` to larryspark provider config across all profiles. Lara needs `/new` to pick up. |

## Loca Config State

| Field | Value |
|---|---|
| **Provider** | `dflash-spark` |
| **API** | `http://larryspark.local:8000/v1` |
| **Default model** | `qwen` (stale — should be `deepseek-v4-flash` or match what's running) |
| **Models listed** | `qwen` only (needs all 3) |
| **Fallback** | `ollama-launch` → `gemma-4-12b` |
| **Config file** | `~/.hermes/profiles/loca/config.yaml` |

## Pending Actions

1. [ ] Patch DS4 serve script with tool calling flags
2. [ ] Deploy switch scripts to Spark
3. [ ] Update Loca config with all 3 models
4. [ ] Retest DS4 with tool calling (via Loca)
5. [ ] Retest 122B (verify still working)
6. [ ] Test 35B (first time for us)
7. [ ] Benchmark all three with standardized template
8. [ ] Clone MiaAI repo on Spark (for 35B chat template)

## Crash History (See ADR-006 for full protocol)

| Date | Model | Cause | Resolution | Required power-cycle? |
|---|---|---|---|---|
| July 7 | Puzzle-75B MTP | OOM at 0.85 GMU + MTP | fank recipe: headless, no enforce-eager | YES (multiple) |
| July 7 | Puzzle-75B MTP | EngineCore silent death with --enforce-eager | Remove enforce-eager for MTP | YES |
| July 7 | Puzzle-75B MTP | Broken DFlash entrypoint | Use stock vLLM 0.24.0 | YES |
| July 7 | FlashInfer + ComfyUI | First-boot autotune OOM with Flux co-located | Kill Flux before first boot | No |
| July 5 | DS4 | ninja missing → JIT compile crash | Install ninja in venv | No |
| July 18 | Puzzle-75B MTP | OOM (10 crashes total, every k/container/flag combo) | FlashInfer MoE autotune workspace ~20GB on SM121. Only enforce-eager no-MTP works. HF discussion post drafted. | YES (multiple) |
| July 18 | Step 3.7 + Lara TTS | OOM — TTS model loaded on top of Step 3.7 (522MB free) | Don't co-locate on tight configs | YES |
| July 19 | 122B n=4 165K | Running clean | Production config locked | No |