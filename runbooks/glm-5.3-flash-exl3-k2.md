# Runbook: GLM-5.3-Flash EXL3 K2 — single DGX Spark

> **Status: staged candidate; not running.** This runbook prepares a safe, reversible evaluation of Vic Cruz's GLM-5.3 Flash EXL3 K2 stack. It is not an Arena recipe and it must not alter Loca/Hermes configuration without James explicitly choosing that change.

## What this is

- **Model pack:** [`vcruz305/GLM-5.3-Flash-EXL3-K2`](https://huggingface.co/vcruz305/GLM-5.3-Flash-EXL3-K2), pinned 91.017 GiB / 120 safetensor shards.
- **Upstream operational recipe:** [`vcruz305/GLM-5.3-Flash-EXL3-K2-DGX-Spark-recipe`](https://github.com/vcruz305/GLM-5.3-Flash-EXL3-K2-DGX-Spark-recipe) at `841d864694056db202e3c75f6759400af6672293`.
- **Runtime:** its custom CUDA-13 / Python-3.12 / aarch64 vLLM lineage plus EXL3 plugin and ExLlamaV3 extensions. **Never substitute stock vLLM**: it lacks both `Glm5Next` architecture support and `exl3` quantization.
- **Hardware:** exactly one DGX Spark GB10 (SM121), unified memory.
- **Quantization:** EXL3 2-bit MCG routed experts; attention, shared components, embeddings, lm-head, and vision remain BF16.
- **Serve API:** OpenAI-compatible vLLM at port `8000` when launched by our canonical script.

## Evidence boundary

Upstream reports:

- 8K, MTP k=2, one stream: **15.7–16.5 tok/s**.
- 64K, MTP k=2: **14.6–15.7 tok/s**.
- Verbatim retrieval through **163,479 prompt tokens**.
- A successful **258,048-token** cold prefill at roughly **604 prompt tok/s / 427.3 seconds** without speculation, using an exact 3 GiB KV pool and `max-num-seqs=1`.

The 258K result is a successful long prefill, **not** proof of 258K retrieval, captured-text decode quality, or multi-agent 258K serving. Treat this as a one-long-lane candidate until our own gates pass.

## Non-negotiable safeguards

1. The Spark runs exclusive lanes. A GLM launch stops H3/Qwen/DS4 before it starts.
2. Keep all model data only beneath `~/models/hf/`; this pack belongs at:
   ```text
   ~/models/hf/GLM-5.3-Flash-EXL3-K2
   ```
3. Do not modify Loca/Hermes config. The script only prepares/serves the backend.
4. Do not use `pip install vllm`.
5. Do not mix native MTP with DFlash.
6. Preserve torch compile caches during normal switching. Do not clear them for this new model.
7. The canonical launcher wraps vLLM in `systemd-run --user --scope --collect -p MemoryMax=110G -p MemorySwapMax=0`; if an experiment fails, stop the scope and verify memory is released before any other model comes up.
8. Do not run an Arena submission from this unpublished/custom runtime. A future Arena submission requires a validated registered public recipe and a clean `verify_arena_recipe.py` pass.

## Canonical script

The canonical Spark-side script is:

```text
scripts/switch-to-glm53-exl3-k2.sh
```

It is staged in `styles01/sparkrun-recipes`, and has no behavior without an explicit action.

### Preparation only — no workload change

Copy or pull the script/repo on the Spark, then run:

```bash
bash scripts/switch-to-glm53-exl3-k2.sh --check
bash scripts/switch-to-glm53-exl3-k2.sh --stage
bash scripts/switch-to-glm53-exl3-k2.sh --download
```

- `--check` validates architecture, CUDA, Python, custom vLLM capabilities, FlashInfer prerequisites, and pack completeness if already present.
- `--stage` pins the upstream source and installs its prebuilt patched wheels under `~/venvs/glm53-exl3-local`.
- `--download` resumes directly to `~/models/hf/GLM-5.3-Flash-EXL3-K2`; it never stages weights on the Mac.

Expected pack integrity:

```text
120 safetensor files
97,728,721,536 bytes
```

### First live lane: 64K / MTP k=2

After preparation and only when intentionally switching the exclusive Spark lane:

```bash
bash scripts/switch-to-glm53-exl3-k2.sh --start --lane 64k-mtp
```

This is our first production-shaped evaluation target:

```text
MAX_MODEL_LEN=65536
SPEC_METHOD=mtp
MTP_TOKENS=2
GPU_MEM_UTIL=0.91
max-num-seqs=1
fp8 KV
prefix caching OFF for clean evaluation
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

It waits for `/v1/models` on `127.0.0.1:8000`, allowing up to 15 minutes for the 91-GiB load plus JIT/graphs. The model load itself is expected to take about 11–12 minutes.

### Research-only 258K prefill probe

Do **not** call this a concurrent agent lane:

```bash
bash scripts/switch-to-glm53-exl3-k2.sh --start --lane 258k-probe
```

This uses the upstream evidence configuration:

```text
MAX_MODEL_LEN=262144
SPEC_METHOD=none
MAX_NUM_SEQS=1
MAX_NUM_BATCHED_TOKENS=2048
KV_CACHE_MEMORY_BYTES=3221225472  # exact 3 GiB
prefix caching OFF
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

The allocator setting is required. Upstream's no-model reproducer found the GB10 sparse-indexer allocator pattern reaching 32.26 GiB reserved under defaults versus 1.49 GiB with expandable segments. The upstream 258K outcome also used a venv-side workspace-sizing patch that is **not shipped in the repository**, so do not claim a reproduction based on launching alone.

## Smoke and evaluation order

1. **Runtime identity**
   ```bash
   curl -sf http://127.0.0.1:8000/v1/models
   curl -sf http://127.0.0.1:8000/health
   ```
   Confirm the served ID is `GLM-5.3-Flash-EXL3` and logs show `fused_moe=exl3_moe`.

2. **Minimal generation / parser smoke**
   Run one exact `pong` completion, thinking enabled for normal agent work, and tool-call parsing with the upstream `glm47` parser.

3. **Correctness before throughput**
   - 8K and 64K context needle ladder;
   - real agent tool-call loop battery;
   - repeated 64K prefill/decode cycles;
   - process restart and allocator/headroom audit.

4. **Metrics to record**
   - TTFT and prefill tok/s;
   - output tok/s based on completion-token accounting, not chunk rate;
   - p50/p95 inter-token latency;
   - MTP mean/per-position acceptance;
   - tool-call parse/correctness rate;
   - `MemAvailable` over every long prefill and after repeated cycles;
   - exact revisions, command line, server log, and raw JSONL.

5. **Only then test 258K**
   First run a real-text needle/captured-text decode workload—not merely a summarize request. Keep full completions. Require stable memory, exact needle answer, and usable decode rate before promoting the ceiling claim.

## Promotion gates

GLM becomes a selectable Spark lane only after all of the following are independently reproduced:

- Stable repeated 64K agent/tool loops with thinking **on**.
- No K-pool tail crash, silent looping, malformed tool calls, or health-without-progress behavior.
- Stable long-prefill memory with at least 10 GiB headroom; no allocator drift across cycles.
- Measured real-text 64K decode, TTFT, and p95 ITL meeting a worthwhile quality/speed tradeoff against the existing lanes.
- 256K: exact retrieval plus captured-text decode, rather than HTTP-200-only prefill.
- Clean stop/start leaves unified memory free and H3/Qwen/DS4 can come back from their canonical scripts.

## Known upstream risks

- Custom fork/plugin/wheel supply chain; artifacts must remain revision-pinned.
- Historical K-pool tail out-of-bounds bug. Require the post-2026-08-30 wheel lineage and run the provided eager bounds soak before trusting prolonged generation.
- `max-num-seqs > 1` with long speculative drafts is reported to garble output; first lane is deliberately one request.
- FlashInfer sparse MLA needs `nvcc` and venv `ninja` on `PATH`.
- A root-owned Triton cache can break a user-mode server; use a user-writable `TRITON_CACHE_DIR` rather than `sudo`ing the service.

## Sources

- [Upstream recipe](https://github.com/vcruz305/GLM-5.3-Flash-EXL3-K2-DGX-Spark-recipe)
- [Upstream measurements](https://github.com/vcruz305/GLM-5.3-Flash-EXL3-K2-DGX-Spark-recipe/blob/main/docs/MEASUREMENTS.md)
- [Upstream long-context evidence](https://github.com/vcruz305/GLM-5.3-Flash-EXL3-K2-DGX-Spark-recipe/blob/main/docs/IMPROVEMENTS_AND_EVIDENCE.md)
- [K2 model pack](https://huggingface.co/vcruz305/GLM-5.3-Flash-EXL3-K2)
- [Patched runtime wheels](https://huggingface.co/vcruz305/GLM-5.3-Flash-EXL3-K2-spark-vllm)
- [Source GLM-5.3 Flash](https://huggingface.co/zai-org/GLM-5.3-Flash)
