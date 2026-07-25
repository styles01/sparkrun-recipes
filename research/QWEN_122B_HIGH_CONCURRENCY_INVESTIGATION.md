# Qwen 3.5 122B-A10B High-Concurrency Testing on DGX Spark — Investigation Summary

**For:** James (jaita/jamey)  
**Date:** 2026-07-09  
**Question:** Has anyone tested Qwen 3.5 122B-A10B (or any 122B variant) at high concurrency (c=8, c=16, c=32+) on a single NVIDIA DGX Spark (GB10/SM121, 121 GB unified)?

---

## Bottom Line

**No published evidence exists.** Nobody has publicly tested Qwen 122B at c≥8 on a single DGX Spark. The highest documented concurrency is **c=3** (Entrpi DFlash repo). Everything beyond c=3 is uncharted territory.

---

## What Was Searched

1. GitHub issues in `vllm-project/vllm` for "122B" + concurrent/batch/concurrency
2. GitHub issues/discussions in `Entrpi/qwen3.5-122B-A10B-on-spark` and `albond/DGX_Spark_Qwen3.5-122B-A10B-AR-INT4`
3. Spark Arena (spark-arena.com) leaderboard for any 122B benchmarks beyond single-user
4. Reddit r/LocalLLaMA, r/NVIDIA, r/MachineLearning
5. NVIDIA Developer Forums for DGX Spark + Qwen 122B concurrency threads
6. DuckDuckGo / web search for blogs, Medium, YouTube
7. GitHub issues for stock vLLM 0.24 + 122B at high batch

---

## Concrete Findings

### 1. Entrpi DFlash Repo (qwen3.5-122B-A10B-on-spark)
- **Default `MAX_NUM_SEQS = 3`** — explicitly tuned for "single user with up to 3 concurrent decode streams"
- **Only benchmarks c=1, 2, 3**
- Decode-only throughput table (streaming, excludes prefill):

| Workload | 1 stream | 2 streams | 3 streams | Aggregate @ 3 |
|---|---|---|---|---|
| prose | 48.8 tok/s | 38.4 tok/s | 30.8 tok/s | 92.5 tok/s |
| code | 66.9 tok/s | 55.3 tok/s | 43.0 tok/s | 129 tok/s |
| agentic (real 6k tool-call ctx) | 121.8 tok/s | 98.9 tok/s | 66.7 tok/s | 200 tok/s |

- Script `conc_workloads.py` accepts `--levels 1,2,3` as default; no c>3 tested or documented
- README notes: "batch > 4, which is slower under speculative decode; both are fixed in vLLM 0.20+" — implies they *know* c>4 has issues but did not test/publish it

### 2. Albond MTP-2 Repo (DGX_Spark_Qwen3.5-122B-A10B-AR-INT4)
- **Only single-stream (c=1) benchmarks published**
- `bench_qwen35.sh` runs sequential (not concurrent) non-streaming requests
- Reported: **51.58 tok/s** cross-prompt average (e2e, includes prefill)
- README mentions TurboQuant variant supports **5 concurrent users @ 256K context**, but **no benchmark numbers** are given for c>1
- No issues or discussions about high concurrency in the repo (6 open issues, none about concurrency)

### 3. Spark Arena (spark-arena.com)
- Searched leaderboard for `Qwen3.5-122B-A10B` with concurrency filters 1, 2, 4, 5, 10
- **Only 8 total results** for this model, all at low concurrency:
  - `Qwen3.5-122B-A10B-int4-AutoRound` — 58.77 tok/s, **2 nodes**, c=2 (implied by leaderboard filter)
  - `Qwen3.5-122B-A10B-int4-fp8-hybrid` — 55.04 tok/s, **single node**, c=2
  - `Qwen3.5-122B-A10B-int4-AutoRound-EC` — 49.47 tok/s, single node, c=2
- **No c=8, c=16, or c=32 entries** exist on Spark Arena for any 122B variant
- The site’s concurrency dropdown only offers 1, 2, 4, 5, 10 — but no 122B results at c=4, 5, or 10 were visible

### 4. GitHub vLLM Issues (vllm-project/vllm)
- **93 issues** match "122B" + "concurrency/batch/max_num_seqs" — almost none are Spark-specific
- Relevant ones:
  - **#37602** — Qwen3.5-122B-A10B-FP8 EngineCore crash on **concurrent image requests** (100 reqs, 4×H200, TP=4). Text-only stable. **Not Spark.**
  - **#39985** — Qwen3.5-122B-A10B Engine **hangs at Prefill** under high concurrency (40 reqs) with multi-node PP=2 on dual 4090D nodes. **Not Spark.**
  - **#40381** — Buffer overflow allocating memory on Qwen3.5-122B-A10B-GPTQ-Int4 and NVFP4 on **GB10 DGX Spark**. This is a **startup/load issue**, not a concurrency benchmark.
  - **#35519** — Qwen3.5 NVFP4 models crash on ARM64 GB10 DGX Spark (CUDA illegal instruction). **Not concurrency-related.**
  - **#41725** — Recurring CUDA kernel hang on **2× DGX Spark** (GB10, sm_12.1) with **MiniMax-M2.7-NVFP4**, TP=2 across 2 nodes. **Not Qwen 122B.**
- **No issue** documents 122B tested at c≥8 on a single Spark

### 5. NVIDIA Developer Forums
- Thread: *"Fastest Qwen 3.5 122B Int4 recipe on DGX Spark tested and published on Spark-Arena"* (May 2026)
  - Community recipes using DFlash speculative decode, up to ~40 tok/s single-stream
  - One user (`btvd`) posted a recipe with `max-num-seqs: 1` and DFlash n=5
  - **No mention of c=8, c=16, or c=32 testing**
- Thread: *"Introducing PrismaQuant"* — mentions testing Qwen 3.6 35B for "multiple concurrent requests", but explicitly says "Going to test now the 122B versus Albons Hybrid" — i.e., **122B concurrency testing was planned, not done**

### 6. Reddit / Other Web Sources
- Reddit search returned no usable posts (bot blocking or genuinely empty)
- No Medium articles, blog posts, or YouTube videos found showing 122B concurrency scaling on DGX Spark

### 7. Stock vLLM 0.24 + 122B at High Batch
- **No evidence found.** No GitHub issues, no forum posts, no Spark Arena entries for stock vLLM 0.24 running Qwen 122B at c>3 on Spark

---

## What This Means for James

| Concurrency | Evidence | Source |
|---|---|---|
| **c=1** | Extensively benchmarked (~28–81 tok/s depending on stack) | Albond, Entrpi, Spark Arena |
| **c=2** | Benchmarked (aggregate ~92–129 tok/s) | Entrpi README table |
| **c=3** | Benchmarked (aggregate ~200 tok/s agentic) | Entrpi README table |
| **c=4** | Mentioned as "slower under speculative decode" but **no numbers** | Entrpi README |
| **c=5** | TurboQuant README claims "5 concurrent users @ 256K" but **no tok/s numbers** | Albond README |
| **c=8, 16, 32+** | **No published tests anywhere** | — |

### Suspected Bottlenecks (informed guess, not evidence)
- **KV cache size:** Entrpi’s default pool is ~426k–456k tokens. At 256K context × 3 streams = 768K needed; they already rely on chunked prefill and headroom. c=8 at long context would likely OOM or require aggressive KV compression (TurboQuant, FP8 KV, etc.)
- **Routed-expert bandwidth:** Entrpi notes per-stream throughput drops as batch grows because "more routed-expert traffic per step". At c=8+ this could become the dominant bottleneck on 273 GB/s LPDDR5x
- **Speculative decode degradation:** Entrpi explicitly says batch > 4 is slower under speculative decode (DFlash/MTP). This is fixed in vLLM 0.20+, but no one has published post-fix numbers for 122B on Spark

---

## Files Created

- `/tmp/entrpi_readme.md` — raw Entrpi README
- `/tmp/entrpi_conc_workloads.py` — concurrency sweep script (max default c=3)
- `/tmp/entrpi_bench_decode.py` — single-stream decode benchmark
- `/tmp/entrpi_bench_albond.py` — albond-method e2e benchmark
- `/tmp/albond_readme.md` — raw Albond README
- `/tmp/albond_bench.sh` — albond bench script (sequential, not concurrent)
- `/tmp/issue_*.json` — GitHub issue JSONs for referenced vLLM issues

---

## Recommendation

If James wants to know whether c=8/16/32 is viable, **he will likely need to test it himself**. No one has published the numbers. The safe starting hypothesis:

- **c=4–5** might work with TurboQuant KV compression or short contexts, but throughput per stream will drop significantly
- **c=8+** is uncharted; likely feasible only with very short contexts, aggressive quantization, or chunked prefill tuning — but no one has proven it yet

