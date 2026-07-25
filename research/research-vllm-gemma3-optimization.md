# vLLM Optimization Recipes: Gemma 3 / MedGemma 27B on DGX Spark / GB10

Research findings from vLLM GitHub issues, PRs, and documentation (vLLM 0.24.0).

---

## 1. V0 vs V1 Engine Performance for Gemma 3

| Aspect | Finding |
|--------|---------|
| V1 status | `VLLM_USE_V1=1` works in v0.24.0 but is not formally recognized yet (skill confirms). V1 is the newer architecture with redesigned scheduler + worker. |
| Gemma 3 stability | Text-only Gemma 3 (`gemma3_text`) is significantly more stable on vLLM than multimodal Gemma 3 (`gemma3`) or Gemma 4. MedGemma 27B text-it is the safest variant. |
| Known V1 deadlock | Issue #37729: V1 engine core deadlocks under concurrent load with **fp8 + prefix caching + Qwen3.5**. The pattern (FP8 weights + prefix caching + high concurrency) matches the user's config. Deadlock risk increases with batch size. |
| Recommendation | **Try both.** Start with V1 (`VLLM_USE_V1=1`). If deadlocks or stalls occur, fall back to V0 (unset). For v0.24.0, V0 is the safer default for 27B dense models at high concurrency on GB10. |

---

## 2. JIT Compilation Issues & Warmup Strategies

| Aspect | Finding |
|--------|---------|
| Root cause | Triton kernels JIT-compile on first invocation with the exact tensor shapes/strides encountered. vLLM 0.24.0 has **partial** warmup infrastructure (issues #47451, #42815, #45601, #45245) but does NOT pre-compile all attention or sampler variants for every possible batch shape. |
| GB10-specific hang | Issue #37431: Mamba-2 / Triton kernels crash with illegal instruction on SM121 without `CUDA_LAUNCH_BLOCKING=1`. While this is Mamba-specific, the pattern shows SM121 Triton compilation can desync CUDA streams. |
| JIT during inference | Issue #43009: Confirmed bug — Triton kernel JIT compilation fires inside the served request window, causing 10-20ms stalls per compile. For 48 concurrent requests, compiles can serialize and spike ITL. |
| Warmup gaps | vLLM warms: KV block zeroing (#45245), top-k/top-p samplers (#45601), steering kernels (#42815), some attention paths (#42215). It does **not** warm all batch-size specializations for Triton attention or all decoder shapes. |
| Practical mitigation | 1. Send a synthetic warmup burst covering your expected batch sizes (1, 8, 16, 32, 48) before real load. 2. Set `CUDA_LAUNCH_BLOCKING=1` ONLY if hangs occur (costs ~5-10% throughput). 3. Consider `TRITON_CACHE_DIR` persistence to reuse compiled kernels across restarts. |

---

## 3. `--enforce-eager` vs CUDA Graphs Tradeoffs for Gemma 3

| Aspect | Finding |
|--------|---------|
| CUDA graph benefit | Eliminates CPU launch overhead for decode steps. On small-batch / single-GPU serving, decode throughput can improve **2-3×** with graphs vs eager. |
| Gemma 3 graph issues | - Issue #38834: Oversized piecewise CUDA graphs for Gemma3n cross-decoder (multimodal). **Not applicable to text-only MedGemma.**<br>- Issue #46253: Cross-node graph capture fails on GB10 due to host-staged NCCL (no GPUDirect). **Only applies to TP>1 / multi-node.**<br>- Issue #47275: Gemma4 MoE aborts during graph capture with FA4. **Not applicable to dense Gemma 3.** |
| Single-GPU GB10 (TP=1) | CUDA graphs are **safe and recommended** for dense Gemma 3 text-only on a single GB10. No NCCL, no MoE, no multimodal encoder = no known graph blockers. |
| `--enforce-eager` | Disables graphs. Useful only if you see `RuntimeError` during graph capture or if using multi-node TP. For the user's single-GPU setup, **do NOT use `--enforce-eager`** — it will significantly hurt decode throughput. |
| Breakable cudagraph | `VLLM_USE_BREAKABLE_CUDAGRAPH=1` is auto-enabled for some models. On single-node TP=1 dense models, this is usually harmless but can be disabled (`=0`) if it causes issues. |

---

## 4. FlashInfer vs Triton Attention Backend Performance

| Aspect | Finding |
|--------|---------|
| Gemma 3 text attention | MedGemma 27B text-it uses **uniform head_dim=256**, standard GQA, sliding_window=1024. It does **NOT** have the heterogeneous head dimensions (256/512) that plague Gemma 4 and force Triton fallback. |
| FlashInfer on SM121 | Issue #31740 (GB10 support): FlashInfer attention **works** on SM75-SM121. FlashInfer MLA does not (SM100 only). For dense MHA models like Gemma 3, FlashInfer is viable.<br>Issue #46329: NVFP4 KV cache on SM120/SM121 uses FlashInfer FA2 successfully. |
| Triton attention perf issue | Issue #48076: **Critical finding** — Triton attention drops long-context decode from 3D split-KV to 2D at batch ≥ ~12, **~doubling ITL**. This directly explains high inter-token latency at 48 concurrent requests. |
| Recommendation | **Force FlashInfer if possible.** Triton is the safe fallback on SM121, but its decode path degrades at batch > 12. For 48 concurrent requests, FlashInfer should maintain better ITL. Try:<br>`--attention-backend flashinfer` |
| Caveat | Issue #40677: Forcing FlashInfer on Gemma 4 with head_size=512 failed on SM120. Gemma 3 uses head_size=256, which FlashInfer supports universally. Should be safe. |

---

## 5. Spark-Specific Community Recipes & Flags

| Flag / Env Var | Status | Exact Command |
|----------------|--------|---------------|
| `VLLM_USE_DEEP_GEMM=0` | **MANDATORY** on SM120/SM121 | Prevents FP8 kernel crash from "Unknown recipe" assertion (issue #47130). Without this, FP8 checkpoints fail at startup warmup on Blackwell. |
| `--kv-cache-dtype fp8` | **MANDATORY** on unified memory | BF16 KV cache doubles memory use and starves the OS. FP8 KV is required for 27B + 8K context on 121GB unified. |
| `--gpu-memory-utilization 0.88` | **Recommended max** | Start at 0.85-0.88. 0.95 causes system stall; 0.90 still swaps. Verified by real crash (skill). |
| `TRITON_PTXAS_PATH` | Auto-configured on SM121 | vLLM auto-sets this to `/usr/local/cuda/bin/ptxas` (issue #31740). Only needed if Triton fails to find ptxas. |
| `CUDA_LAUNCH_BLOCKING=1` | Conditional | If Triton JIT hangs or illegal instruction during inference, set this. Costs throughput but prevents SM121 stream desync. |
| `--max-num-seqs 128` | Recommended | Default 256 is too high for 27B + 8K context on 121GB unified. 128 reduces KV pressure. |
| `--max-num-batched-tokens 4096` | Recommended | Prevents memory spikes during long-document chunked prefill. |
| `--enable-chunked-prefill` | Recommended | Essential for long-context stability on unified memory. |
| `--enable-prefix-caching` | Recommended | Reduces TTFT for repeated prompts. Note: combined with FP8 + V1, this triggered deadlock #37729 on Qwen3.5; monitor for similar issue on Gemma 3. |
| `--attention-backend flashinfer` | Try this | On SM121 with uniform head_dim, FlashInfer should outperform Triton at batch > 12. |
| `VLLM_USE_V1=1` | Optional trial | If stable, V1 can improve scheduler efficiency. If deadlocks occur, remove it. |
| `--enforce-eager` | **Avoid** on single-GPU | Only needed for multi-node TP or if graph capture crashes. |

### Exact Recommended Serve Command

```bash
VLLM_USE_DEEP_GEMM=0 \
vllm serve SaitBurak/medgemma-27b-text-it-FP8-dynamic \
  --gpu-memory-utilization 0.88 \
  --max-model-len 8192 \
  --max-num-seqs 128 \
  --max-num-batched-tokens 4096 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --kv-cache-dtype fp8 \
  --attention-backend flashinfer \
  --served-model-name medgemma \
  --port 8000
```

Optional variants to test:
- **Try V1:** prepend `VLLM_USE_V1=1`
- **If Triton hangs:** prepend `CUDA_LAUNCH_BLOCKING=1` (slower but stable)
- **If FlashInfer fails:** omit `--attention-backend flashinfer` (falls back to TRITON_ATTN)

---

## 6. Expected Throughput Benchmarks for Gemma 3 27B on Similar Hardware

### Direct Comparable Numbers from vLLM Issues

| Model | Hardware | Config | Perf | Source |
|-------|----------|--------|------|--------|
| Qwen3-Coder-30B-A3B-FP8 | GB10 (SM121) | v0.21.0, FP8 weights, FP8 KV | **44 tok/s** (single-request decode) | Issue #31740 |
| Qwen3-Next-80B-A3B-FP8 | GB10 (SM121) | v0.21.0, FP8 weights, FP8 KV | **45 tok/s** | Issue #31740 |
| Qwen3.6-27B-FP8 | GB10 (SM121) | v0.21.0, dense hybrid attention, 262K ctx | **19-35 tok/s gen** at run=4-5, prompt 31-222 tok/s | Issue #43885 (healthy logs) |
| Gemma 4 E4B (4.5B effective) | RTX 4090 (SM89) | v0.19.0, TRITON_ATTN fallback | **~9 tok/s** (abnormally slow due to heterogeneous attention) | Issue #38887 |
| Gemma 3 12B MM quantized | 96-core ARM | v0.24.0-ish, chunked prefill | **0.68 req/s, 324 tok/s total, 87 tok/s output** | Issue #43078 |

### Analysis for MedGemma 27B on DGX Spark

- **Architecture:** Gemma 3 27B is a **dense** model (NOT MoE), roughly comparable to Qwen3.6-27B in parameter count and memory footprint.
- **Hardware:** GB10/SM121 is a single-GPU Blackwell-class SoC with unified memory. It is slower than H100/B200 for raw compute but has massive memory bandwidth.
- **Current user performance:** 5 tok/s per request, 207ms ITL, 240 tok/s aggregate at 48 concurrent.

### Expected Targets (with optimizations applied)

| Metric | Current | Optimized Target | Notes |
|--------|---------|------------------|-------|
| Single-request decode | ~5 tok/s | **15-25 tok/s** | With FlashInfer + CUDA graphs + no JIT stalls |
| ITL @ batch=48 | 207 ms | **40-80 ms** | Triton→FlashInfer switch should prevent 2D fallback at batch>12 |
| Aggregate throughput @ 48 req | 240 tok/s | **400-600 tok/s** | Better batching efficiency + reduced JIT overhead |
| Prefill throughput | varies | **50-150 tok/s** depending on prompt length | Chunked prefill + prefix caching |

### Why Current Performance is Suboptimal

1. **Triton JIT compilation during inference** adds 10-20ms stalls per request window, serializing the batch and inflating ITL.
2. **Triton attention backend** likely dropped to 2D decode path at batch=48, doubling ITL (issue #48076).
3. **No FlashInfer** — the default backend selection on SM121 may have chosen Triton even though FlashInfer supports head_dim=256.
4. **V1 engine + prefix caching + FP8** combination has known deadlock/stability issues (issue #37729 pattern), potentially causing scheduler stalls.

### Benchmarking Verification

Use the built-in benchmark after applying optimizations:

```bash
vllm bench serve \
  --base-url http://localhost:8000 \
  --endpoint /v1/chat/completions \
  --backend openai-chat \
  --dataset-name random \
  --input-len 512 \
  --output-len 256 \
  --num-prompts 200 \
  --num-warmups 50 \
  --request-rate 48
```

Monitor with:
```bash
curl -s http://localhost:8000/metrics | grep -E \
  'time_to_first_token|inter_token_latency|num_requests_running|gpu_cache_usage'
```

---

## Summary of Actions

1. **Apply the exact serve command** above (FlashInfer + FP8 KV + `VLLM_USE_DEEP_GEMM=0`).
2. **Send a warmup burst** (1, 8, 16, 32, 48 concurrent dummy requests) to pre-compile Triton/FlashInfer kernels before real load.
3. **Do NOT use `--enforce-eager`** on single-GPU — keep CUDA graphs enabled.
4. **Try V1 engine** (`VLLM_USE_V1=1`) for one trial; if deadlocks occur, revert to V0.
5. **If Triton JIT warnings persist** or hangs occur, set `CUDA_LAUNCH_BLOCKING=1` as a safety net.
6. **Expected improvement:** ITL should drop from 207ms to ~40-80ms; aggregate throughput should rise from 240 tok/s to 400-600 tok/s at 48 concurrent requests.
