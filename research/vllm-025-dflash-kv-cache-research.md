# vLLM 0.25.x DFlash + fp8 KV Cache Research

**Date:** 2026-07-24
**Goal:** Find community solutions for running Qwen 3.5 122B with DFlash + fp8 KV cache on vLLM 0.25.x, specifically the `unify_kv_cache_spec_page_size` assert issue.

---

## TL;DR — Key Findings

1. **DFlash + fp8 KV is fundamentally broken on vLLM ≤ 0.25.x.** Issue [#41559](https://github.com/vllm-project/vllm/issues/41559) documents that DFlash's non-causal attention requirement is incompatible with ALL KV cache quantization (fp8, turboquant) because no attention backend supports non-causal + fp8 simultaneously. This was reported on v0.20.0 and remains **OPEN** as of 2026-07-24.

2. **PR #43081 (merged 2026-06-22) partially fixed this** by adding `supports_non_causal()` and a `_noncausal_prefill_wrapper` to FlashInfer, but **only for SM100 (trtllm-gen path)**. On SM120/SM121 (GB10/DGX Spark), the fp8→bf16 dequant path is NOT wired in, causing illegal memory access. Reported by user Pablohassan in #41559 comments.

3. **vLLM 0.25.0/0.25.1 does NOT have native DFlash support for Qwen 3.5 that doesn't need page_size patches.** The DFlash Bring-Up Tracker [#46105](https://github.com/vllm-project/vllm/issues/46105) shows "Hybrid DFlash" (SWA + non-causal full attention) is still **unchecked**. The Qwen 3.5 DSpark support PRs [#47377](https://github.com/vllm-project/vllm/pull/47377) / [#47390](https://github.com/vllm-project/vllm/pull/47390) are **open and unmerged**.

4. **The Entrpi repo (our repo) uses vLLM 0.23, NOT 0.25.x.** The README explicitly states: "Engine: vLLM 0.23, sm121 build with the DFlash PRs, via the prebuilt image `ghcr.io/aeon-7/aeon-vllm-ultimate:2026-06-18-v0.23.0-dflashfix`." There are **no 0.25.x updates** in the repo — the latest commits are about serving-mode docs, prefix caching, and KV pool reclamation, all on 0.23.

5. **Qwen3.5-122B-A10B-FP8 crashes on 0.25.0/nightly** regardless of DFlash. Issue [#48477](https://github.com/vllm-project/vllm/issues/48477) reports `CUBLAS_STATUS_EXECUTION_FAILED` at profile_run on 0.25.0 and nightly, while 0.24.0 works. This is a separate regression that blocks FP8 hybrid GDN + DFlash together on any post-0.24 version.

6. **The Entrpi repo's patches (`patch_unify2.py` + `patch_prefix_align.py`) are the ONLY community recipe** for DFlash on Qwen 3.5 122B hybrid GDN. They are designed for vLLM 0.23.0 and target `kv_cache_utils.py` specifically. No one else has publicly solved this for 0.25.x.

7. **The albond repo** (`albond/DGX_Spark_Qwen3.5-122B-A10B-AR-INT4`) uses vLLM 0.19.1 with MTP-2 (not DFlash) and TurboQuant KV cache (4x). It does NOT use DFlash and does NOT target 0.25.x.

---

## Detailed Findings by Source

### 1. NVIDIA DGX Spark Forums

**URL checked:** https://forums.developer.nvidia.com/c/accelerated-computing/dgx-spark-gb10/dgx-spark-gb10/721

- Forum is a standard Discourse instance. Latest topics: NemoClaw (March 2026), Performance FAQ (Feb 2026), Performance blog (Nov 2025), Power Clarification, Welcome post.
- **No topics about Qwen 3.5 122B DFlash, fp8 KV cache, or vLLM 0.25.x KV cache page size issues found** in the main category listing.
- The forum search API returned no results for "Qwen 122B DFlash vLLM fp8".
- The Spark Arena leaderboard is referenced from the forum, and forum users (eugr, dbsci, raphael.amorim) are the Spark Arena maintainers.

### 2. Spark Arena Leaderboard

**URL checked:** https://spark-arena.com/leaderboard

- The leaderboard is a Next.js client-rendered app. The HTML payload contains only the shell — actual benchmark data loads via client-side JavaScript (`LeaderboardDataLoader` component), which cannot be extracted via curl.
- **The leaderboard exists and is active** (powered by spark-vllm-docker, llama-benchy, sparkrun). Maintainers: Drew Botwinick (dbsci), Eugene Rakhmatulin (eugr), Raphael Amorim.
- Related repos: [spark-vllm-docker](https://github.com/eugr/spark-vllm-docker), [llama-benchy](https://github.com/eugr/llama-benchy), [sparkrun](https://github.com/spark-arena/sparkrun).
- **Could not extract specific Qwen 122B entries or configs** due to client-side rendering. Would need a headless browser to render the data.
- The leaderboard benchmarks are for DGX Spark specifically, using spark-vllm-docker images. These images are based on vLLM but the exact version and patches are in the spark-vllm-docker repo (not checked in detail).

### 3. Entrpi/qwen3.5-122B-A10B-on-spark (Our Repo)

**URL checked:** https://github.com/Entrpi/qwen3.5-122B-A10B-on-spark

**Current state (as of 2026-07-24):**
- Uses **vLLM 0.23.0** (image `ghcr.io/aeon-7/aeon-vllm-ultimate:2026-06-18-v0.23.0-dflashfix`)
- **No 0.25.x port exists.** Latest commits are about serving-mode docs, prefix caching with DFlash, KV pool reclamation (376k→427k), int8 lm-head, load speed. All on 0.23.
- Has one closed issue: "KV Cache dtype (why not fp8?)" — answered with `ValueError: Selected backend AttentionBackendEnum.FLASH_ATTN is not valid for this configuration. Reason: ['kv_cache_dtype not supported']` — confirming fp8 KV is blocked by FlashAttn's non-causal incompatibility.

**The two KV cache patches (designed for vLLM 0.23.0, NOT 0.25.x):**

#### patch_unify2.py — The `unify_kv_cache_spec_page_size` assert fix

**Root cause documented in the patch:**
- With `--mamba-block-size` set, vLLM's hybrid alignment makes target mamba page == target attention page by PADDING the mamba page (e.g. "+0.54%").
- That padded value becomes `max_page_size`.
- The DFlash drafter's attention page is smaller; `unify_kv_cache_spec_page_size` scales its `block_size` by `ratio = max_page_size // layer_page_size`.
- Because `max_page_size` is a *padded* (non block-size-linear) number, the scaled page lands just under it, and `assert new_spec.page_size_bytes == max_page_size` fires.

**The fix:** Instead of removing the assert, the patch keeps the scaled `block_size` AND pads the `<1%` remainder via `page_size_padded=max_page_size`. This mirrors what `get_kv_cache_groups` already does for `HiddenStateCacheSpec` layers.

**Critical insight:** The OLD patch (which just did `replace(layer_spec, page_size_padded=max_page_size)` without the block_size scaling) DROPPED the scaling, leaving the drafter at block_size=16 behind a max-sized physical page. This mis-strided the drafter KV and pinned mean accept length at ~1.47 (garbage drafts). The corrected patch keeps both.

**This is EXACTLY our "1.13M tokens → 561K tokens" problem.** The first successful run had only the assert removed (via the old simple approach), which gave the large KV pool but with mis-strided drafts. The "more patches" that reduced KV to 561K likely changed the unification behavior — possibly by using `pop()` instead of `max()` for `get_uniform_page_size`, which changes which page size is the "max" and thus the scaling ratio.

#### patch_prefix_align.py — The hash_block_size divisibility fix

**Root cause:** The DFlash drafter's attention layers carry a ~2x larger KV page than the target. Page-size unification scales the TARGET's mamba + attention blocks UP (2240 → 4480) to match the drafter's page. `resolve_kv_cache_block_sizes` then sees a MambaSpec whose block_size (4480) != cache_config.block_size (2240) and takes its back-off branch, forcing hash_block_size = LCM = 4480. The drafter group (block 2240) is not divisible by 4480, so `HybridKVCacheCoordinator.__init__` aborts.

**The fix:** The back-off only makes sense for non-align mamba. The patch checks `mamba_cache_mode != "align"` before triggering the back-off, allowing the GCD path (2240) which divides every group.

**serve.sh configuration:**
- `--attention-backend flash_attn` (target) — the DFlash drafter forces `FLASH_ATTN` via `attention_backend` in speculative config
- `--gpu-memory-utilization 0.82` (validated, ~14 GiB free on 128 GB GB10)
- `--max-model-len 262144`, `--max-num-seqs 3`, `--max-num-batched-tokens 8192`
- `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0` (reclaims ~0.6 GiB from CUDA graph over-estimate)
- Prefix caching ON by default (requires patch_prefix_align.py)
- **No `--kv-cache-dtype fp8`** — the serve script does NOT pass fp8 KV because FlashAttn rejects it for non-causal

### 4. vLLM Issue #40898 — DFlash Mixed Attention (SWA)

**URL checked:** https://github.com/vllm-project/vllm/issues/40898

This is actually a **PR** (not an issue): "[Spec Decode] Add Sliding Window Attention support to DFlash drafter"

**Status:** Closed (merged)

**Key contents:**
- Adds SWA support to DFlash draft models with mixed `sliding_attention` / `full_attention` layers
- Uses per-KV-group DFlash block tables and slot mappings while keeping the target/draft raw KV tensor shared
- Avoids the EAGLE cache-drop path for DFlash (where draft context K/V is pre-written by the target model)
- Test result on Qwen3.5-122B-A10B + DFlash (TP4, 15 spec tokens): 5,988,876 KV tokens, 521.33 tok/s, acceptance 42.23%, acceptance length 7.33

**The tracking issue #46105 says this PR is STALE and needs to be rewritten for MRV2.** Quote: "This PR is stale and needs to be rewritten in MRV2. @benchislett is leading the rework."

### 5. vLLM Issue #41559 — DFlash incompatible with ALL KV cache quantization

**THE most critical finding.** [Issue #41559](https://github.com/vllm-project/vllm/issues/41559)

**Status:** OPEN (reported on v0.20.0, still open as of 2026-07-24)

**Core problem:** DFlash requires non-causal attention (`causal=False`) for draft cross-attention. Every attention backend either:
- Rejects non-causal attention entirely (FlashInfer, Triton), OR
- Rejects KV-quant dtypes when non-causal is set (FlashAttn, FlexAttention)

**Compatibility matrix (v0.20.0):**

| Backend | Non-causal support | fp8 KV + non-causal |
|---|---|---|
| FLASH_ATTN | Yes | Rejected |
| FLASHINFER | No | N/A |
| TRITON | No | N/A |
| FLEX_ATTENTION | Yes | Rejected |
| TURBOQUANT | N/A | Hardcoded `causal=True` |

**Key comment from benchislett (vLLM DFlash maintainer):** PR #39995 adds FlashInfer FP8 KV Cache support for non-causal, but "needs some polish."

**Key comment from Pablohassan (SM121/GB10 specific):** On SM121 (GB10/DGX Spark), fp8 KV + non-causal draft crashes with `CUDA error: an illegal memory access` — the fp8→bf16 dequant is only wired into the SM100 trtllm-gen path, not the native non-causal prefill wrapper used on SM120/121.

**Key comment from MidasMining:** DFlash + TurboQuant KV works on llama.cpp's ggml backend (not vLLM) — proving it's not architecturally blocked, just vLLM implementation gaps.

### 6. vLLM Issue #48477 — Qwen3.5-122B-A10B-FP8 crashes on 0.25.0

[Issue #48477](https://github.com/vllm-project/vllm/issues/48477)

**Status:** OPEN

**Core problem:** `Qwen/Qwen3.5-122B-A10B-FP8` crashes at startup on 0.25.0 and nightly with `CUBLAS_STATUS_EXECUTION_FAILED` at profile_run. 0.24.0 works fine.

**Version matrix:**

| Version | Plain FP8 serve (TP2) | DFlash on hybrid GDN |
|---|---|---|
| 0.24.0 | ✅ works | ❌ broken (pre-#47914) |
| 0.25.0 | ❌ startup crash | ❌ (release cut before fixes) |
| nightly | ❌ CUBLAS crash | ✅ verified on Qwen3.5-35B (not 122B) |

**Comment from zcyyyds-test:** The crash is NOT FP8-specific — bf16 Qwen3.5-122B-A10B also crashes on nightly with TP>1. The regression discriminates on the 122B target (layer count 48 vs 35B's 40, hidden 3072 vs 2048, TP>1), not on quantization.

**This means:** Even if we solve the KV cache page size issue, 0.25.0/0.25.1 likely can't serve the 122B FP8 model at all due to this separate CUBLAS regression.

### 7. vLLM Issue #46105 — DFlash Bring-Up Tracker

[Issue #46105](https://github.com/vllm-project/vllm/issues/46105)

**Status:** OPEN (tracking issue)

**Key checkboxes:**
- [x] "Standard" DFlash: non-causal, all full attention layers (Qwen3-8B-DFlash)
- [ ] "Hybrid" DFlash: SWA + non-causal full attention (Gemma4 DFlash) — **NOT DONE**
- [x] "MiMo-style" DFlash: all SWA layers (PR #46104 merged)
- [x] "Speculators-style" DFlash: all SWA, all causal

**Attention backend support status:**
- [x] FlashInfer (default cutlass backend) supports non-causal prefills — **merged via #43081**. "This enables FP8-KV Cache + DFlash for many use-cases."
- [ ] TRTLLM pending decode-optimized kernels
- [ ] CuteDSL not plugged into vLLM
- [ ] XQA not integrated

**Critical note from the tracker:** "#40898 adds many patches to get hybrid DFlash working in MRV1. This PR is stale and needs to be rewritten in MRV2."

**Comment from zenprocess (production user on GB10):** Running hybrid DFlash on vLLM `0.23.1rc1.dev701` (post-#46104), Qwen3.6-35B-A3B-FP8, on GB10 SM121. They adapted #40898 onto the post-#46104 tree by hand. Quote: "the port removes [patches that are now upstream]." This is the closest to our setup but on 35B not 122B.

### 8. vLLM Issue #49701 — page_size_padded fallback for non-CUDA backends

[Issue #49701](https://github.com/vllm-project/vllm/issues/49701)

**Status:** OPEN

**Relevant to us:** This issue directly references `unify_kv_cache_spec_page_size()` in `vllm/v1/core/kv_cache_utils.py` and the `NotImplementedError` for backends without `indexes_kv_by_block_stride = True`. The fix adds an `elif isinstance(layer_spec, AttentionSpec)` fallback that sets `page_size_padded=max_page_size` — the SAME pattern used in our patch_unify2.py and the existing MambaSpec path.

### 9. vLLM Issue #41657 — Fix KV cache tensor sharing across block sizes

[Issue #41657](https://github.com/vllm-project/vllm/issues/41657)

**Status:** Closed (merged)

**Root cause:** KV cache groups with different `block_size` values could share the same raw `KVCacheTensor` when `page_size_bytes` matched, but equal `page_size_bytes` is not enough — groups with different `block_size` can map distinct logical blocks to the same physical slot (e.g., `2 * 16 == 1 * 32`).

**Fix:** Only share raw `KVCacheTensor` among groups with the same `block_size`.

**Discussion insight from benchislett:** "The problem all along was the different KV layouts for different attention backends. The different `block_size` shouldn't make much difference. When using DFlash Speculative Decoding, because we require an attention backend that supports non-causal attention, we allow the drafter layers and target model layers to have a different attention backend."

**Discussion insight from heheda12345:** "Will forcing the same backend name break linear attention models? These models have some layers with AttentionBackend and other layers with Mamba-like backends."

### 10. Related Open PRs/Issues for Qwen 3.5 + DFlash

- [#47377](https://github.com/vllm-project/vllm/pull/47377) — Add DSpark support for Qwen3.5 target models (OPEN)
- [#47390](https://github.com/vllm-project/vllm/pull/47390) — fix: support Qwen3.5 family for DSpark (OPEN)
- [#48381](https://github.com/vllm-project/vllm/pull/48381) — DSpark: store draft KV cache in model dtype for MLA-only cache layouts (OPEN)
- [#48392](https://github.com/vllm-project/vllm/pull/48392) — DFlash/DSpark draft support under decode context parallelism (OPEN)
- [#49293](https://github.com/vllm-project/vllm/pull/49293) — Keep model-dtype query for FlashInfer builders serving non-causal attention (OPEN)
- [#48380](https://github.com/vllm-project/vllm/issues/48380) — DSpark/DFlash fails with `--kv-cache-dtype fp8_ds_mla`: draft inherits MLA-only cache layout (OPEN)

### 11. albond/DGX_Spark_Qwen3.5-122B-A10B-AR-INT4

**URL checked:** https://github.com/albond/DGX_Spark_Qwen3.5-122B-A10B-AR-INT4

- Uses **vLLM 0.19.1** (not 0.25.x)
- Uses **MTP-2** speculative decoding (not DFlash)
- Has **TurboQuant KV Cache** support (TQ 4x KV cache, 256K context)
- Achieves 52 tok/s with INT4+FP8 hybrid, INT8 LM head, MTP-2
- **Does NOT use DFlash and does NOT target 0.25.x** — not directly applicable to our problem
- The Entrpi repo's README credits this as the base recipe: "Building on the albond recipe"

---

## Analysis: Why We Got 1.13M Tokens Once But Can't Reproduce

Based on the Entrpi repo's patch_unify2.py documentation and the vLLM issue findings, here's the likely explanation:

### The 1.13M Token Run (First Success)

The FIRST successful run had:
- Only the `new_spec.page_size_bytes == max_page_size` assert removed
- The prefix_align patch applied
- No other assert changes

This corresponds to the **OLD simple patch approach**: `replace(layer_spec, page_size_padded=max_page_size)` which DROPPED the block_size scaling. This gives:
- Large KV pool (because the page size unification succeeds without the assert blocking)
- **BUT mis-strided drafter KV** (block_size=16 behind a max-sized physical page)

The 1.13M tokens came from the pool being allocated with the wrong (unscaled) block size — the drafter's blocks were smaller than they should have been, so more "blocks" fit in the pool. But the drafts were garbage (accept length ~1.47 per the Entrpi docs).

### The 561K Token Runs (Subsequent)

When "more patches" were added (likely the corrected patch_unify2.py approach that KEEPS the scaled block_size AND pads the remainder), the block size scaling was restored. This:
- Makes drafts correct (accept length ~7.7)
- But the scaled block_size (e.g., 4480 vs 2240) means each block takes more memory
- Result: fewer total blocks → fewer KV tokens → 561K instead of 1.13M

### The `pop()` vs `max()` Issue

If `get_uniform_page_size` returns `pop()` instead of `max()`, it picks a different page size as the "uniform" value. This changes:
- Which group's page size becomes `max_page_size`
- The scaling ratio applied to other groups
- The total KV pool size

Using `pop()` might return the drafter's page size (smaller) as the "max", meaning no scaling is needed → larger pool but potentially mis-strided. Using `max()` returns the padded target page → scaling needed → smaller pool but correct strides.

---

## Config Recommendations

### For vLLM 0.23.0 (Current Working Setup — Entrpi Repo)

**This is the ONLY validated path.** Use the Entrpi repo as-is:
- Image: `ghcr.io/aeon-7/aeon-vllm-ultimate:2026-06-18-v0.23.0-dflashfix`
- Apply `patch_unify2.py` (corrected: scaled-block + pad-remainder)
- Apply `patch_prefix_align.py` (align-aware back-off)
- **Do NOT use `--kv-cache-dtype fp8`** — it's incompatible with DFlash's non-causal attention on FlashAttn
- KV pool: ~426K tokens (dense) / ~457K tokens (dflash) at gpu-mem 0.82
- Expected throughput: ~81 tok/s on agent traffic (DFlash n=12)

### For vLLM 0.25.1 (Target — Currently Blocked)

**Direct upgrade to 0.25.1 is NOT viable** for Qwen 3.5 122B + DFlash + fp8 KV because:

1. **FP8 KV + DFlash is fundamentally incompatible** (#41559) — no attention backend supports non-causal + fp8 on SM121
2. **Qwen3.5-122B-A10B-FP8 crashes on 0.25.0/0.25.1** (#48477) — separate CUBLAS regression
3. **Hybrid DFlash support is incomplete** (#46105) — "Hybrid" DFlash checkbox unchecked
4. **Qwen 3.5 DSpark support PRs are unmerged** (#47377, #47390)
5. **PR #40898 (hybrid DFlash) is stale** — needs MRV2 rework

### Recommended Path Forward

1. **Stay on vLLM 0.23.0** for production DFlash on Qwen 3.5 122B. The Entrpi recipe is the only working solution.

2. **For fp8 KV cache, use bf16 KV with the dense profile** (INT4+FP8 hybrid checkpoint). The fp8 KV cache dream is blocked by the non-causal attention incompatibility, which is a vLLM architectural issue, not a patch we can fix locally.

3. **If 0.25.x is required**, the path would be:
   - Wait for #48477 (CUBLAS crash on 122B) to be fixed
   - Wait for #47377/#47390 (Qwen3.5 DSpark support) to merge
   - Wait for FlashInfer non-causal fp8 dequant on SM121 (from #41559 discussion)
   - Port patch_unify2.py and patch_prefix_align.py to 0.25.x's kv_cache_utils.py (the API may have changed — verify the exact code structure)
   - The `unify_kv_cache_spec_page_size` function likely still exists but may have different surrounding code

4. **For maximizing KV pool on 0.23.0:**
   - Use `dflash` profile (not `dense`) — gives 457K vs 426K tokens
   - Use `serving-mode` to go headless (~10-15 GiB back to KV)
   - Set `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0`
   - Set `gpu-mem 0.82` (validated; 0.88+ swaps)
   - The 1.13M token run was a bug, not a feature — those drafts were garbage

### The `unify_kv_cache_spec_page_size` Assert — Correct Handling

Based on patch_unify2.py's documented analysis:

```python
# WRONG (old patch — gives large pool but garbage drafts):
new_spec = replace(layer_spec, page_size_padded=max_page_size)
# Drops block_size scaling → mis-strided drafter → accept len ~1.47

# CORRECT (patch_unify2.py — gives correct drafts, smaller pool):
new_spec = replace(layer_spec, block_size=new_block_size)
if new_spec.page_size_bytes != max_page_size:
    new_spec = replace(
        layer_spec,
        block_size=new_block_size,
        page_size_padded=max_page_size,
    )
assert new_spec.page_size_bytes == max_page_size
# Keeps scaled block_size → correct strides → accept len ~7.7
```

The assert should NOT be removed. It should be satisfied by padding the remainder while keeping the scaled block_size. The `page_size_padded` field exists exactly for this purpose (it's already used for `HiddenStateCacheSpec` and `MambaSpec`).

### `get_uniform_page_size`: `pop()` vs `max()`

- **`max()`** is correct — it picks the largest page size, which is the padded hybrid target page. All other groups get scaled UP to match, which is the intended behavior.
- **`pop()`** would pick an arbitrary group's page size, which could be the smaller drafter page. This would avoid scaling (larger pool) but corrupt the drafter's KV layout.
- The 1.13M run likely used `pop()` or the unscaled approach, giving a large pool with broken drafts.

---

## Summary Table

| Question | Answer |
|---|---|
| Anyone running Qwen 3.5 122B + DFlash on vLLM 0.25.x? | **No.** The Entrpi repo uses 0.23.0. No one has publicly ported to 0.25.x. |
| Anyone solved the KV cache page size unification with DFlash? | **Only on 0.23.0** via patch_unify2.py + patch_prefix_align.py (Entrpi repo) |
| Any recipes for Qwen 122B on DGX Spark with fp8 KV? | **No.** fp8 KV is fundamentally incompatible with DFlash's non-causal attention (#41559). albond uses TurboQuant but with MTP-2, not DFlash, on vLLM 0.19.1. |
| Does vLLM 0.25.1 have native DFlash for Qwen 3.5? | **No.** Qwen 3.5 DSpark PRs (#47377, #47390) are open/unmerged. Hybrid DFlash is unchecked in the tracker (#46105). |
| Why 1.13M tokens once but not reproducible? | The first run dropped block_size scaling (mis-strided drafts, accept ~1.47). Corrected patch keeps scaling (correct drafts, accept ~7.7) but smaller pool. |
| Correct way to handle the assert? | Keep scaled `block_size` + pad remainder via `page_size_padded=max_page_size` — do NOT remove the assert or drop the scaling. |
| `pop()` vs `max()` for `get_uniform_page_size`? | Use `max()`. `pop()` gives larger pool but corrupts drafter KV. |

---

## Sources

- [Entrpi/qwen3.5-122B-A10B-on-spark](https://github.com/Entrpi/qwen3.5-122B-A10B-on-spark) — README, runtime patches, commits
- [albond/DGX_Spark_Qwen3.5-122B-A10B-AR-INT4](https://github.com/albond/DGX_Spark_Qwen3.5-122B-A10B-AR-INT4) — README
- [vLLM #40898](https://github.com/vllm-project/vllm/pull/40898) — DFlash SWA support (stale, needs MRV2 rework)
- [vLLM #41559](https://github.com/vllm-project/vllm/issues/41559) — DFlash incompatible with all KV cache quantization (OPEN)
- [vLLM #41657](https://github.com/vllm-project/vllm/pull/41657) — Fix KV cache tensor sharing across block sizes (merged)
- [vLLM #46105](https://github.com/vllm-project/vllm/issues/46105) — DFlash Bring-Up Tracker (OPEN)
- [vLLM #48477](https://github.com/vllm-project/vllm/issues/48477) — Qwen3.5-122B-A10B-FP8 crashes on 0.25.0 (OPEN)
- [vLLM #49701](https://github.com/vllm-project/vllm/pull/49701) — page_size_padded fallback for non-CUDA backends (OPEN)
- [vLLM #47377](https://github.com/vllm-project/vllm/pull/47377) — Add DSpark support for Qwen3.5 (OPEN)
- [vLLM #47390](https://github.com/vllm-project/vllm/pull/47390) — Fix Qwen3.5 family for DSpark (OPEN)
- [vLLM #49293](https://github.com/vllm-project/vllm/pull/49293) — FlashInfer non-causal model-dtype query fix (OPEN)
- [vLLM #48381](https://github.com/vllm-project/vllm/pull/48381) — DSpark draft KV in model dtype for MLA layouts (OPEN)
- [Spark Arena Leaderboard](https://spark-arena.com/leaderboard) — client-rendered, could not extract data via curl
- [NVIDIA DGX Spark Forum](https://forums.developer.nvidia.com/c/accelerated-computing/dgx-spark-gb10/dgx-spark-gb10/721) — no relevant topics found