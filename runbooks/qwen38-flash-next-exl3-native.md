# Qwen3.8-Flash-Next EXL3 3.05bpw — Native ExLlamaV3 (P5, planned)

**Status: PLANNED, NOT YET EXECUTED.** No downloads, no installs. Everything below is
from the community recipes (verified by 4 independent parties) — our own numbers get
filled in during the Phase 1-3 gates.

**What this is:** the alternate single-stream latency lane for qwen3.8-flash-next.
~80 tok/s code decode vs our NVFP4 lane's ~37 tok/s — at B-grade (84.6) vs A-grade
(91.9) quality. Tradeoff is explicit and measured: **EXL3 84.6 @ 80 t/s vs NVFP4
91.9 @ 37 t/s**. Role split: latency-sensitive + deep-context work → EXL3;
quality-critical agentic + concurrent multi-stream → NVFP4 (stays the daily driver).

## Provenance (who verified what)

| Claim | Source |
|---|---|
| 79.95 tok/s code, native | ViC305 recipe README (2026-09-17) |
| 79.5 wall / 83.8 engine, 74% accept | Yume (@yume_arasaki) repro (2026-09-20) |
| 80.0 tok/s, TrueScore 84.6 (B), Pass@1 92.1 | Wesche (@WescheNex1q) SparkBench 76-scenario (2026-09-18) |
| Served: 56-59 t/s @ 1K-32K ctx, 2.9s median turn | Wesche — fastest turn of ANY model on his leaderboard |
| 84→90 tok/s (per-expert-K kernel, 5e0ba47) | vcruz305/exllamav3 fork, 2026-09-20 |
| +9% on GB10 (EXL3_MIXEDK_LEGACY=1, f6e42ec) | vcruz305/exllamav3 fork, 2026-09-21 |

## The pack

`turboderp/Qwen3.8-Flash-Next-exl3` @ revision **`3.05bpw_h5_ng5`**
(commit `69e33439…`, non-gated, ~2.5K downloads):

- 7 safetensors shards (7.61-7.91 GB each) = ~48.7 GB
- `ngram_embedding.safetensors` = 30.4 GB (the n-gram/PLE table)
- `mtp_hyper_connection_mixer_patch` (mixer weights)
- **Total: 79.29 GB across 25 files.** Resident ~78.6 GB, 30 s load.
- Destination per our models rule: `~/models/hf/turboderp/…` (NOT a new top-level dir)

## Two serving paths

### Path A — Native ExLlamaV3 (PRIMARY, the 80 t/s path)

1 stream only. This is the "80 t/s" everyone quotes. Served via TabbyAPI or the
recipe's OpenAI shim for API compatibility.

**Setup (one-time, ~15 min build):**
- Fork: `vcruz305/exllamav3` @ head `bf9e10f` (NOT upstream turboderp — GR_INT8
  default-on is worth +7-13%; per-expert-K kernel 84→90 t/s; MIXEDK_LEGACY +9%)
- `scripts/exl3_native/setup_exllamav3_150.sh` → exllamav3 1.5.0 venv @
  `~/venvs/exl3-150`, extension JIT-built for sm_121, needs nvcc CUDA 13.0 +
  the vllm-exl3 checkout's aarch64 patch tool
- Torch 2.13.0, TORCH_CUDA_ARCH_LIST=12.1

**Launcher env (from `run-qwen38-exl3.sh`, all measured-best):**
```bash
export TORCH_CUDA_ARCH_LIST=12.1
export EXL3_INT8_GEMV=0            # int8 GEMV SLOWER than fp16 on GB10
export EXL3_MOE_COOP_WIDE=1        # wide 128-col MoE coop tile
export EXL3_GR_INT8=1              # hyperconnection mixers int8 (default on since 523ecd3)
export EXL3_MTP_HEAD_N=65536       # draft chain argmaxes over 64K lm_head slice
export EXL3_NGRAM_STREAM=0
BIGCORES="5-9,15-19"               # 10 Cortex-X925; launch thread on little core costs ~2 t/s
```
**Launch:** `taskset -c $BIGCORES python examples/chat.py -m ~/models/Qwen3.8-Flash-Next-EXL3
-mode qwen35 -mtp -ndt 5 -dds -dc 0.6 -cq 8,8 -cs 262144 -tps`
- `-ndt 5 -dds -dc 0.6`: up to 5 drafts, stop early when running confidence < 0.6
  (prose collapses past draft position 1; code does not)
- `-cq 8,8`: 8-bit KV (+5% @ 4k, +11% @ 240k, acceptance unchanged)
- KV = 24 KB/token fp16 (12 of 48 layers full attention, 2 KV heads), 12 KB at 8-bit
- Page cache: run `drop-model-cache.sh` before load (GPU-allocatable memory on GB10)

**Expected (native, cold, greedy, 400 new tokens):**
code 79-90 (fork head), DevOps 62, prose 53, no-draft 33; at 240k ctx: 72 (8-bit KV);
prefill ~1,150 t/s flat to 480k. Draft acceptance: code 73%, prose 46%.

**Single stream ONLY.** Never set this as the daily driver — it cannot do concurrent
agentic work. This is the deep-work/latency lane.

### Path B — vLLM + vllm-exl3 plugin (SECONDARY, concurrency-capable)

Same pack, vLLM 0.29.0 (Qwen4ExpForConditionalGeneration in-tree, aarch64 wheel),
exllamav3 1.4.7 from source + aarch64 patch, vllm-exl3 @ `94c29ba`+ (unsharded
n-gram + disk-backed table mode, PRs #22-24; anything older than 6b26e5c has the
fat-expert prefill corruption bug + the 33-144-token prefill wedge).

Serve: `bash scripts/serve_one_spark_qwen.sh` (auto-detects MODEL_DIR, defaults
MTP k=3 + bf16 recurrent state + 262144 ctx).

**Concurrency (unique prefix, MAX_NUM_SEQS=8, temp 0, short prompts):**

| streams | MTP k=3 steady | no draft steady |
|---:|---:|---:|
| 1 | 54.8 | 29.0 |
| 2 | 89.4 | 49.1 |
| 4 | **155.6** | 87.2 |
| 8 | **157.6** | 153.4 |

Per-stream at 4: ~39 t/s. MTP k=3 plateaus at 8 streams (draft overhead).

**CRITICAL — the MTP acceptance cliff:** acceptance collapses at 163,840 prompt
tokens. Past the cliff, draft is a net LOSS of ~21% → for workloads routinely
exceeding 163,840 prompt tokens: `SPEC_CONFIG=none`. Depth policy: k=3 default;
k=2 or none for >120k prompts.

**Known vLLM-path limitation:** CUDA internal errors at 50-80k prompts reported
(Yume, on this pack); native engine survived an 80k Hermes tool loop clean.

## Quality expectations (from Wesche's 76-scenario bench)

- TrueScore **84.6 (B)** vs NVFP4's 91.9; Pass@1 92.1 (code 95.4 / structured 100 /
  instruction 95.1)
- **What 3.05bpw costs:** safety 72.5, long-context 26.7, agentic 88.1
  - long-context 26.7 CONTRADICTS Yume's 15/15 needle @ 240k — likely measures beyond
    retrieval. **Gate G3:** replicate on OUR evals before trusting.
- Prose per-class sensitivity: prose 52.3 t/s (draft acceptance collapses on prose)

## Execution plan (when Spark is up and James says go)

**Phase 0 — prerequisites (no lane impact):**
- [ ] Spark disk check: need ~85 GB free for pack + ~6 GB venv/build. NVFP4 pack
      (~135 GB) coexists fine — they are alternate lanes, never co-resident in memory.
- [ ] `git clone https://github.com/vcruz305/exllamav3 ~/work/exllamav3-fork`
      (checkout `bf9e10f`) + `git clone https://github.com/vcruz305/vllm-exl3 ~/work/vllm-exl3`
- [ ] Run `setup_exllamav3_150.sh` (~15 min JIT build, no lane impact)
- [ ] Download pack to `~/models/hf/turboderp/Qwen3.8-Flash-Next-exl3/`
      revision `3.05bpw_h5_ng5` (79.29 GB — background download, ~10-40 min on gigabit)
- [ ] Copy `run-qwen38-exl3.sh` + `drop-model-cache.sh` to `~/sparkrun-recipes/scripts/`
      (reconcile model dir to `~/models/hf/` per our models rule)

**Phase 1 — native smoke (lane swap, ~30 min):**
- [ ] STOP NVFP4 lane: `./stop.sh` (never pkill); verify memory free
- [ ] Run native launcher on a short prompt; verify load (30 s), gen works
- [ ] Measure: code prompt (expect ~80+ t/s w/ fork head), prose (expect ~52),
      no-draft (expect ~33-35). Compare vs ViC305/Yume/Wesche — ±3% = success.
- [ ] RESTORE NVFP4 lane (switch-to-qwen38-flash-next-vllm-nvfp4.sh)

**Phase 2 — quality gates (lane down or parallel on Mac):**
- [ ] Safety probe (Wesche's 72.5 — our own safety set)
- [ ] Long-context probe at 240k (needle + beyond-retrieval task to resolve the
      26.7-vs-15/15 contradiction)
- [ ] Agentic probe (tool-calling loop vs our Hermes pattern — Yume's 80k loop as bar)

**Phase 3 — decision gate:**
- Keep EXL3 as an alternate lane IF: Phase 1 numbers reproduce AND quality gates pass
  AND the per-stream latency gain matters for real workloads.
- Then wire into SparkDash + Loca provider as a second model entry
  (`qwen3.8-flash-next-exl3`), keep NVFP4 as default.
- If Phase 2 quality gates fail → document, park permanently.

## Rules inherited from our standards

- **Never run attached to foreground SSH** — detached + cron follow-up
- **Lane swap rule:** stop NVFP4 lane first, verify memory free, then load EXL3
- **Models live in `~/models/` only** — reconcile any recipe default dir to `~/models/hf/`
- **Update the canonical serve script** (`scripts/switch-to-qwen38-flash-next-exl3.sh`)
  to match whatever actually runs — never ad-hoc launch and let the script go stale
- **exact_tg + tg≥400** for all our benchmarks (P2 rule)
- Container rule: this runs BARE on the host (venv), not in a container — exllamav3
  JIT-builds for sm_121; keep it out of the docker lane's way (host networking free)

## Files & links

- Recipe repo: https://github.com/vcruz305/Qwen3.8-Flash-Next-EXL3-DGX-Spark-recipe
  (pinned README archived: /tmp/exl3_recipe_readme.md — copy to references/ on Phase 0)
- Fork: https://github.com/vcruz305/exllamav3 @ bf9e10f
- Plugin: https://github.com/vcruz305/vllm-exl3 @ 94c29ba+
- Pack: https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3 (rev 3.05bpw_h5_ng5)
- Turnkey (alternative install): https://github.com/jayleaton/qwen38-flash-next-exl3-spark
- Tuning logs (their measured data): scripts/exl3_native/tuning/logs/ in recipe repo

## Confidence

HIGH on existence/numbers (4-party triangulation: ViC305, Yume, Wesche, recipe README
internally consistent; ±1.5% agreement). MEDIUM on our ability to reproduce exactly
(fork moved past our pinned evaluation — bf9e10f is newer than anything benchmarked
in the README tables; treat our Phase 1 as the first measurement of that head).
Unverified: safety 72.5 / long-context 26.7 sub-scores (single-source, our Phase 2
relicates them before any adoption decision).


## SparkDash integration (verified in dashboard code, 2026-09-21)

SparkDash's LlmProbe natively supports backendType `exl3` — the detector is
data-presence-keyed, not name-keyed (memory rule honored):

- Detection: `/v1/models` owned_by containing `exl3`, OR `/health` returning
  `{ok:true, busy:<bool>}` or `{backend:"exl3"}` → `_healthLooksLikeExl3`
- Live metrics: `_applyExl3Health` reads cumulative `prompt_tokens_total` +
  `completion_tokens_total` diffs (idle→0), `busy` → slotsActive=1, plus
  `context_length`
- Probe port = `llmPorts[0]` from sparks.json = **8000** (single port, resolveLlmPort
  uses only the first entry). The EXL3 shim must therefore serve on **:8000** during
  the EXL3 session (vLLM lane stopped) — SparkDash then auto-detects with zero
  dashboard changes. Multiple ports are NOT watched simultaneously.

**Gap found + fixed (server-side first):** the recipe's `serve_openai.py` /health
returned `{status, engine}` which does NOT match the detector → would have been
misclassified as vLLM with no live tok/s. Patched shim (committed to our repo at
`scripts/exl3_serve_openai.py`) now emits the full contract:
`{status:"ok", engine:"exllamav3-native", backend:"exl3", busy:<bool>,
context_length:<int>, prompt_tokens_total:<int>, completion_tokens_total:<int>}`
+ busy clears in a finally block (no stuck-active if generation throws).

Upstream shim (unpatched): scripts/beta_testing/serve_openai.py in the recipe repo.

### VERIFIED WORKING (2026-09-21 22:15, live E2E)

- **Detection + live metrics confirmed working**: `/api/sparks/spark-001/metrics`
  returns `metrics.llm[0]: {available: true, backend: "exl3", modelId:
  "Qwen3.8-Flash-Next-EXL3", contextLength: 262144, slotsActive: 1 during gen,
  generationTps: 250-300 caught live mid-gen, idle→0}`. The probe classifies via
  `/health {backend:"exl3"}` and computes tok/s from cumulative-counter diffs.
- **Root cause of the original dead telemetry** (fixed): my first shim patch left
  the ORIGINAL `return {` in `run_generate()` — an early return BEFORE the
  counter-bump block, so `/health` counters never moved. Removed the dead block;
  counters now accumulate on the executed path (bump → return `_res`, `busy=False`
  in `finally`).
- **context_length now real**: `_ctx_len()` includes `max_position_embeddings`
  (exllamav3 1.5.0's field, resolved via `read_cfg(...text_config->...)`) →
  `/health` reports 262144.
- **Probe-poll timing note**: SparkDash polls ~2-5 s; generations shorter than ~3 s
  may complete between polls (card shows idle). Longer gens light the card solidly.
- Shim committed to `scripts/exl3_serve_openai.py` in styles01/sparkrun-recipes.
