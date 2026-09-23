# Qwen3.8-Flash-Next EXL3 Native — SparkDash Telemetry Audit (control-by-control)

> **This is the telemetry appendix of the [daily-driver runbook](qwen38-flash-next-exl3-daily-driver.md).**
> For launch/ops (launcher, env, restart rules) use that runbook; this document covers the
> SparkDash telemetry contract only. Note two post-audit upgrades: `perPositionAcceptance`
> is now a real per-position array (`mtp_accept_by_position`, via `record_draft_stats`),
> and prefill tok/s now comes from the engine-weighted live `prefill_rate` field.

**Date:** 2026-09-21 · **Scope:** every telemetry control the SparkDash UI renders for the EXL3
card, audited adversarially against the vLLM/DS4 parity standard ("make it work like vLLM" = full
parity, null only where the backend genuinely cannot provide the value).

**Data path:** exllamav3 engine → `~/sparkrun-recipes/scripts/exl3_serve_openai.py` (`/health` v3
contract) → LlmProbe `_probeOpenAICompatible` → `_applyExl3Health` → `/api/sparks/spark-001/metrics`
(`metrics.llm[0]`) → frontend panel dials + showcase.

## Shim /health contract (v3, verified live)

```
status, engine: exllamav3-native, backend: exl3, busy (refcounted inflight > 0),
context_length (262144), prompt_tokens_total (bumps at ENQUEUE),
completion_tokens_total (climbs DURING decode, per-job delta — no EOS spike),
requests_completed_total, requests_failed_total, requests_waiting (lock-queue aware),
mtp_accepted_tokens_total, mtp_drafted_tokens_total (accepted+rejected),
ttft_seconds_sum/_count (engine time_prefill = TTFT), e2e_seconds_sum/_count,
itl_seconds_sum/_count (time_generate / (new_tokens-1)), context_last,
kv_cache_usage ((used+cached)/max via GEN.get_cache_stats()),
prefix_cache_hit_rate (alloc_cached_pages/alloc_pages),
gpu_memory_utilization (torch.cuda.mem_get_info census)
```

## Audit table

| # | UI control | API field | exl3 status | fix applied | verified value (live) |
|---|------------|-----------|-------------|-------------|----------------------|
| 1 | Decode tok/s (hero dial) | `generationTps` | ✅ poll-diff of cumulative counter | shim: counter climbs per-token during decode (C1) | 49–55 tok/s live during decode; 0 idle; no EOS spike |
| 2 | Prefill tok/s | `prefillTps` | ✅ poll-diff, prompt bumped at ENQUEUE | (pre-existing, verified) | 1040 spike at prefill, 0 between |
| 3 | Peak Aggregate | `peakAggregateTps` | ✅ | C1 fix removed fake 225–450 spikes | 55.47 (true max decode rate) |
| 4 | Per-stream Hi/Lo/Avg | `perStreamHigh/Low/Avg` | ✅ single stream ⇒ == aggregate | (21fdd2c, verified) | Hi 55.47 / Lo 1 / Avg 40.91 |
| 5 | Rolling avg/slot | `rollingAvgTpsPerSlot` | ✅ was missing for exl3 | added (mirror of perStreamAvg) | 40.91 |
| 6 | Agg tok/s (sparkline alias) | `aggregateDecodeTps` | ✅ was null | added (= generationTps, vLLM parity) | tracks #1 |
| 7 | Slots Active | `slotsActive` | ✅ busy?1:0 | (verified) | 1 during gen, 0 idle |
| 8 | Slots Total | `slotsTotal` | ✅ 1 (single-stream engine) | documented, correct | 1 |
| 9 | Running | `requestsRunning` | ✅ | (verified) | 1/0 |
| 10 | Waiting | `requestsWaiting`/`waitingSlots` | ✅ was null | shim: inflight − engine_jobs (m3) | 1 with queued request; 0 idle |
| 11 | Total Tokens | `totalTokensDecoded ?? totalOutputTokens` | ✅ | `totalTokensDecoded` now set (was null) | 1065 == usage sum |
| 12 | Tokens Decoded/Output | `totalOutputTokens` | ✅ | (verified) | 1065, reconciles with per-req usage |
| 13 | MTP Rate | `mtpAcceptanceRate` | ✅ was null — shim had the data | shim cumulative counters + probe rate | 0.5185 (== usage draft_accept 0.541 on long gen) |
| 14 | MTP Accepted/Drafted | `mtpAcceptedTokens/DraftedTokens` | ✅ were null | added | 533 / 1028 |
| 15 | Pos 0/Pos 1 bars | `perPositionAcceptance` | ✅ (single-element; engine has no per-pos breakdown) | set from overall rate (DS4-style) | [0.5185] |
| 16 | TTFT (showcase/bench tile) | `ttftSeconds`, `ttft` | ✅ was null — engine measures it | shim ttft_seconds_sum/count (time_prefill) → mean per window | 0.918 s |
| 17 | TTFT p95 card | `ttftP95Seconds` | ✅ mean==observed on single-stream window | set = window mean | 0.918 s |
| 18 | E2E card | `e2eLatency`, `e2eP95Seconds` | ✅ was null | shim e2e sums (time_prefill+time_generate) | 14.213 s (2.3k-prompt/700-tok req) |
| 19 | ITL p95 card | `itlP95Seconds` | ✅ was null | shim itl sums → mean inter-token gap | 0.019 s (≈ 1/52 tok/s ✓) |
| 20 | KV Cache | `kvCacheUsage` | ✅ was null — get_cache_stats has real data | shim (used+cached)/max (M3) | 0.0127 during decode; 0 when fully released |
| 21 | Prefix / Cache Hit | `prefixCacheHitRate` | ✅ was null | shim hit_rate from cache stats | 0.3077 on repeated 2.3k prompt |
| 22 | GPU Mem / GMU | `gpuMemoryUtilization` | ✅ was null | shim torch.cuda.mem_get_info | 0.9839 |
| 23 | Active Context | `activeContext` | ✅ was null | shim context_last | 2280 (matches long prompt) |
| 24 | Requests Started/Done/Failed | `requestsStarted/Completed/Failed` | ✅ were null | shim counters; failed incremented on engine + handler errors (M5, M4) | 5 / 5 / 0 |
| 25 | Gen tokens/req | `genTokensPerReq` + `rollingAvgTokensPerReq` | ✅ was null | completion deltas per completed req | 81 (700-tok long req window) |
| 26 | Rolling E2E / TTFT | `rollingAvgE2e`, `rollingAvgTtft` | ✅ were null | mirror window means (single-stream) | 14.213 / 0.918 |
| 27 | Preempts | `preemptionsTotal` | ✅ N/A-honest: engine has no preemption concept | null-clear retained (m2: no vLLM leak) | null → card hidden |
| 28 | Prefix cache split (cached/uncached prefill) | `cachedPrefillTps`/`uncachedPrefillTps` | ✅ N/A-honest: single stream, prefill never overlaps decode | null-cleared | null |
| 29 | Busy/Running during queue | `busy`/`requestsRunning` | ✅ was bool-per-request (C2: idle during queued decode) | refcounted inflight | busy=True held through queue+decode |
| 30 | Decode dial during decode | `generationTps` | ✅ was 0-then-spike (C1) | per-token incremental counter | live 50–55 tok/s mid-decode |
| 31 | Fake spike after probe reset | `peakAggregateTps` | ✅ M1: reset diffed lifetime counters over ~2s | seed-only first sample after reset | no spike (sim-verified) |
| 32 | Post-failover exl3 state | (internal) | ✅ m1: perStream/lastExl3* leaked across failovers | reset in `_resetDetection` | clean |
| 33 | Backend-switch leaks | `preemptionsTotal` etc. | ✅ m2 | exl3 path null-clears preemptions | null |
| 34 | Client disconnect mid-stream | counters | ✅ M4: orphaned job, busy cleared early | on_chunk failure contained, job drained, stats folded as failed | completed=5 failed=0 clean runs |
| 35 | Showcase live tok/s / peak / TTFT | stream usage `decode_tok_s`, `draft_accept` | ✅ shim emits both in final SSE usage; frontend computes live/peak client-side | (verified semantics) | decode 52.65 tok/s on 700-tok gen |
| 36 | Ctx chip | `contextLength` | ✅ | (verified) | 262144 (262K) |
| 37 | Model chip | `modelId` | ✅ | (verified) | Qwen3.8-Flash-Next-EXL3, badge EXL3 |

## Known limitations (documented N/A-honest)

- **p95 cards show window means** for exl3 (TTFT p95 == mean TTFT): the engine reports no latency
  distribution; single-stream means the latest request *is* the latest window. Honest, labelled.
- **`perPositionAcceptance` is a single-element array** (same as DS4): MTP acceptance is not
  broken down by draft position by exllamav3.
- **`itlP95Seconds` = mean ITL** over the window, not a p95 quantile — no distribution exists.
- **kv_cache_usage reads 0.0 when fully idle** — the engine releases job pages on completion;
  cached-prefix pages keep it >0 when reusable prefixes exist. This is true occupancy semantics.
- **Flat sparklines between requests**: shim is single-request; between gens the dial correctly
  reads 0 (vLLM multi-lane shows overlapping lanes instead).
- **Frontend `isThinking`/reasoning drift, Pos-0-only bar**: frontend concerns, not fixable
  shim/probe-side without a dashboard frontend change; out of scope for this backend parity pass.

## Red-team findings → fixes (all verified)

| ID | Severity | Finding | Fix |
|----|----------|---------|-----|
| C1 | critical | completion counter bumped once at job end → dial 0 during decode + EOS spike polluting peak/perStream (225–450 fake) | per-token incremental bump from `job.new_tokens` delta inside iterate loop; final tail-fold after EOS |
| C2 | critical | `busy` single bool: queued request sets busy while blocked on GEN_LOCK, then busy=False during its decode | `_HL["inflight"]` refcount (inc on entry, dec in finally); busy = inflight > 0 |
| M1 | major | probe reset (`_noteFailure`→`_resetDetection`) zeroed lastTokenCounts but kept recent lastProbeTime → lifetime counter diff over ~2s = fake spike | seed-only first sample after reset (`_exl3Seeded`), also guards counter decreases |
| M3 | major | KV gauge = active-job residency only → ~0 under load | (used_tokens + cached_tokens) / max_tokens |
| M4 | major | client disconnect mid-stream orphaned job, escaped `_hl_job_done`, busy cleared while engine decoded | on_chunk write-failure contained → drain job, fold stats, count failed; SSE tail wrapped |
| M5 | major | `requests_failed_total` never incremented | incremented in `do_POST` exception handler (and M4 path) |
| m1 | minor | `_resetDetection` didn't clear exl3 probe state → leaks across failovers | all `lastExl3*`, `_exl3StreamSamples`, `_exl3Seeded` reset |
| m2 | minor | v2 dropped null-clears → vLLM values leaked into exl3 card | preemptionsTotal (and cached/uncached) null-cleared every exl3 poll |
| m3 | minor | requests_waiting missed requests queued on GEN_LOCK | waiting = inflight − (engine active+pending jobs) |

## Verification evidence (live runs, 2026-09-21)

- Concurrency test (A streaming 350-tok + B queued): busy=True held t+2s→t+12s across queue+decode;
  waiting=1 during queue; counter monotonic 2→74→176→287→363 (live ≈50 tok/s, no drop, no spike).
- Long high-temp gen (2280-prompt, 700 tok, temp 1.05): usage decode 52.65 tok/s,
  draft_accept 0.541; /health folded e2e 26.92s/5, ttft 6.53s/5, mtp 533/1028 (0.5185 rate ≈ usage).
- Short gen: ttft/e2e/itl counters bumped; completed=4→5.
- Dashboard API final: `metrics.llm[0]` = table column "verified value" above.
- Idle poll: gtps=0, prefill=0, slots 0/1, waiting=0 — no phantom values.
- Sim harness (node): reset-spike (M1), counter-reset (server restart), backend-switch leak (m2),
  kv-null handling — all pass; `_resetDetection` includes exl3 state.

## Ops notes

- Restart shim: `kill $(pgrep -f "exl3_serve[_]openai"); setsid nohup /tmp/exl3_serve.sh >
  /tmp/exl3_serveN.log 2>&1 < /dev/null &` (model reload ≈45 s). Beware `pkill -f` self-match: use
  the bracket pattern or `kill $(pgrep -f "exl3_serve[_]openai")`.
- Container probe copy: `docker cp ~/sparkDash/server/collectors/LlmProbe.js
  sparkDash:/app/server/collectors/LlmProbe.js && docker restart sparkDash` (~15 s).
- Commits: sparkrun-recipes (shim + this runbook), sparkDash fork (probe). Shim was previously
  UNTRACKED — now committed.