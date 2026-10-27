/**
 * VllmMetricsParser — parses vLLM's Prometheus /metrics endpoint and tracks
 * deltas between poll cycles to produce per-interval telemetry.
 *
 * Metrics captured:
 *  1. Per-slot gauges: running, waiting, KV-cache usage %
 *  2. Per-request histograms (count/sum deltas → avg for last interval):
 *     TTFT, inter-token latency, E2E latency, prompt tokens, generation tokens
 *  3. Speculative decoding counters: accepted/drafted deltas + acceptance rate
 *  4. Prefix caching counters: hits/queries deltas + hit rate
 *  5. Throughput counters: generation tok/s, prompt tok/s
 *  6. Moving averages over last 10 completed requests:
 *     rolling E2E latency, TTFT, tokens/request, tok/s per slot
 *
 * The parser is stateful — it keeps the previous poll's raw counter/histogram
 * values so it can compute deltas. Call .parse(metricsText, dtSec) each cycle.
 */
export class VllmMetricsParser {
  constructor() {
    // Previous raw counter/histogram values (for delta computation)
    this._prev = {
      // Histograms: { count, sum }
      ttft: null,
      itl: null,
      e2e: null,
      promptTokens: null,
      genTokens: null,
      // Counters
      specAccepted: null,
      specDrafted: null,
      prefixHits: null,
      prefixQueries: null,
      promptTokensTotal: null,
      genTokensTotal: null,
      // KV gauge + pool size (for the prefill-rate derivative)
      kvCacheUsage: null,
      kvPoolTokens: null,
      // Timestamp of previous poll
      time: 0,
    };

    // Rolling window of last 10 completed-request observations.
    // Each entry: { e2e, ttft, tokens, tpsPerSlot }
    this._rolling = [];
  }

  /**
   * Parse a Prometheus metrics body and return a structured telemetry object.
   *
   * @param {string} body - Raw Prometheus text from /metrics
   * @param {number} dtSec - Seconds since last poll (for rate calculations)
   * @returns {object|null} Parsed telemetry, or null if metrics unavailable
   */
  parse(body, dtSec) {
    if (!body || typeof body !== "string") return null;

    const now = Date.now();
    const dt = dtSec > 0 ? dtSec : 0;

    // ─── Gauges (instantaneous) ──────────────────────────
    const runningSlots = this._extractGauge(body, "num_requests_running");
    const waitingSlots = this._extractGauge(body, "num_requests_waiting");
    // vLLM uses kv_cache_usage_perc in newer versions; gpu_cache_usage_perc in older
    const kvCacheUsage = this._extractGauge(body, "kv_cache_usage_perc") ??
      this._extractGauge(body, "gpu_cache_usage_perc") ?? null;

    // ─── Histograms (count/sum deltas) ───────────────────
    const ttftHist = this._extractHistogram(body, "time_to_first_token_seconds");
    const itlHist = this._extractHistogram(body, "inter_token_latency_seconds");
    const e2eHist = this._extractHistogram(body, "e2e_request_latency_seconds");
    const promptTokHist = this._extractHistogram(body, "request_prompt_tokens");
    const genTokHist = this._extractHistogram(body, "request_generation_tokens");

    // ─── KV pool size (for the prefill-rate derivative) ────
    // vLLM exposes the pool size as a LABEL on cache_config_info
    // (kv_cache_size_tokens="627733"), not as a standalone gauge.
    const kvPoolTokensGauge = this._extractGauge(body, "gpu_kv_cache_size_tokens") ??
      this._extractGauge(body, "kv_cache_size_tokens") ??
      this._extractLabelValue(body, "cache_config_info", "kv_cache_size_tokens");

    // ─── Counters (deltas) ────────────────────────────────
    const specAccepted = this._extractCounter(body, "spec_decode_num_accepted_tokens_total");
    const specDrafted = this._extractCounter(body, "spec_decode_num_draft_tokens_total");
    const prefixHits = this._extractCounter(body, "prefix_cache_hits_total");
    const prefixQueries = this._extractCounter(body, "prefix_cache_queries_total");
    const promptTokensTotal = this._extractCounter(body, "prompt_tokens_total");
    const genTokensTotal = this._extractCounter(body, "generation_tokens_total");

    // ─── Per-position speculative acceptance (if available) ──
    const perPositionAcceptance = this._extractPerPositionAcceptance(body);

    // ─── Compute deltas ───────────────────────────────────
    const ttftDelta = this._histDelta(ttftHist, this._prev.ttft);
    const itlDelta = this._histDelta(itlHist, this._prev.itl);
    const e2eDelta = this._histDelta(e2eHist, this._prev.e2e);
    const promptTokDelta = this._histDelta(promptTokHist, this._prev.promptTokens);
    const genTokDelta = this._histDelta(genTokHist, this._prev.genTokens);

    const specAcceptedDelta = this._counterDelta(specAccepted, this._prev.specAccepted);
    const specDraftedDelta = this._counterDelta(specDrafted, this._prev.specDrafted);
    const prefixHitsDelta = this._counterDelta(prefixHits, this._prev.prefixHits);
    const prefixQueriesDelta = this._counterDelta(prefixQueries, this._prev.prefixQueries);
    const promptTokensDelta = this._counterDelta(promptTokensTotal, this._prev.promptTokensTotal);
    const genTokensDelta = this._counterDelta(genTokensTotal, this._prev.genTokensTotal);

    // ─── Compute derived metrics ──────────────────────────

    // Per-request averages for last interval (sum delta / count delta)
    const avgTtft = ttftDelta.count > 0 ? ttftDelta.sum / ttftDelta.count : null;
    const avgItl = itlDelta.count > 0 ? itlDelta.sum / itlDelta.count : null;
    const avgE2e = e2eDelta.count > 0 ? e2eDelta.sum / e2eDelta.count : null;
    const avgPromptTokens = promptTokDelta.count > 0 ? promptTokDelta.sum / promptTokDelta.count : null;
    const avgGenTokens = genTokDelta.count > 0 ? genTokDelta.sum / genTokDelta.count : null;

    // Speculative decoding
    const specAcceptanceRate = specDraftedDelta > 0
      ? specAcceptedDelta / specDraftedDelta
      : null;

    // Prefix caching
    const prefixHitRate = prefixQueriesDelta > 0
      ? prefixHitsDelta / prefixQueriesDelta
      : null;

    // Throughput (tok/s) from counter deltas
    const generationTpsFromCounters = dt > 0 && genTokensDelta > 0
      ? genTokensDelta / dt
      : 0;
    const promptTpsFromCounters = dt > 0 && promptTokensDelta > 0
      ? promptTokensDelta / dt
      : 0;

    // ─── Prefill-rate estimate from the KV gauge derivative ──
    // vLLM increments prompt_tokens_total only when a prompt FULLY
    // finishes prefilling, so counter deltas measure batch-completion
    // rate, not prefill rate: a 60K-token prompt shows ~0 for its
    // whole duration, then one 2-second spike of prompt_size/dt
    // (~30K tok/s observed). While prefill is actually running, KV
    // usage climbs every poll — d(kv_usage * pool_tokens)/dt is the
    // true instantaneous prefill rate. Gate on data availability:
    // (a) pool size known, (b) requests running, (c) counter flat
    // (i.e. no prompt completed this cycle). Falls back to the
    // counter delta otherwise (accurate for small prompts).
    let prefillRateEstimate = null;
    if (this._prev.kvPoolTokens == null && kvPoolTokensGauge != null) {
      this._prev.kvPoolTokens = kvPoolTokensGauge;
    }
    const poolTokens = kvPoolTokensGauge ?? this._prev.kvPoolTokens;
    if (
      poolTokens != null && poolTokens > 0 &&
      kvCacheUsage != null && this._prev.kvCacheUsage != null &&
      dt > 0 &&
      (runningSlots ?? 0) > 0 &&
      promptTokensDelta === 0
    ) {
      const kvTokensNow = kvCacheUsage * poolTokens;
      const kvTokensPrev = this._prev.kvCacheUsage * poolTokens;
      const kvTokenDelta = kvTokensNow - kvTokensPrev;
      if (kvTokenDelta > 0) {
        prefillRateEstimate = kvTokenDelta / dt;
      }
    }

    // ─── Rolling window (last 10 completed requests) ──────
    // Each histogram count delta tells us how many requests completed.
    // We record one observation per completed request batch using the average.
    const completedCount = e2eDelta.count;
    if (completedCount > 0 && avgE2e != null) {
      const tokensPerReq = (avgPromptTokens ?? 0) + (avgGenTokens ?? 0);
      const activeSlots = runningSlots ?? 1;
      const tpsPerSlot = activeSlots > 0
        ? (genTokensDelta / dt) / activeSlots
        : 0;

      // Push a representative observation for this batch
      this._rolling.push({
        e2e: avgE2e,
        ttft: avgTtft ?? 0,
        tokens: tokensPerReq,
        tpsPerSlot,
      });

      // Keep only last 10
      if (this._rolling.length > 10) {
        this._rolling = this._rolling.slice(-10);
      }
    }

    const rolling = this._computeRollingAverages();

    // ─── Save state for next cycle ────────────────────────
    this._prev = {
      ttft: ttftHist,
      itl: itlHist,
      e2e: e2eHist,
      promptTokens: promptTokHist,
      genTokens: genTokHist,
      specAccepted,
      specDrafted,
      prefixHits,
      prefixQueries,
      promptTokensTotal,
      genTokensTotal,
      kvCacheUsage,
      time: now,
    };

    return {
      // Per-slot gauges
      runningSlots: runningSlots != null ? Math.round(runningSlots) : 0,
      waitingSlots: waitingSlots != null ? Math.round(waitingSlots) : 0,
      kvCacheUsage: kvCacheUsage != null ? this._round(kvCacheUsage, 4) : 0,

      // Per-request interval averages (from histogram deltas)
      ttft: avgTtft != null ? this._round(avgTtft, 6) : null,
      interTokenLatency: avgItl != null ? this._round(avgItl, 6) : null,
      e2eRequestLatency: avgE2e != null ? this._round(avgE2e, 6) : null,
      promptTokensPerRequest: avgPromptTokens != null ? this._round(avgPromptTokens, 2) : null,
      generationTokensPerRequest: avgGenTokens != null ? this._round(avgGenTokens, 2) : null,

      // Interval counts
      completedRequests: completedCount,
      intervalTtftCount: ttftDelta.count,
      intervalE2eCount: e2eDelta.count,

      // Speculative decoding
      specAcceptedTokens: specAcceptedDelta,
      specDraftedTokens: specDraftedDelta,
      specAcceptanceRate: specAcceptanceRate != null ? this._round(specAcceptanceRate, 4) : null,
      specPerPositionAcceptance: perPositionAcceptance,

      // Prefix caching
      prefixCacheHits: prefixHitsDelta,
      prefixCacheQueries: prefixQueriesDelta,
      prefixCacheHitRate: prefixHitRate != null ? this._round(prefixHitRate, 4) : null,

      // Throughput (from counter deltas)
      generationTpsFromCounters: this._round(generationTpsFromCounters, 2),
      promptTpsFromCounters: this._round(promptTpsFromCounters, 2),

      // Prefill-rate estimate from the KV gauge derivative (see note above);
      // null when not derivable this cycle.
      prefillRateEstimate: prefillRateEstimate != null ? this._round(prefillRateEstimate, 2) : null,

      // Rolling averages (last 10 completed request batches)
      rolling: {
        avgE2eLatency: rolling.avgE2eLatency,
        avgTtft: rolling.avgTtft,
        avgTokensPerRequest: rolling.avgTokensPerRequest,
        avgTpsPerSlot: rolling.avgTpsPerSlot,
        windowSize: this._rolling.length,
      },
    };
  }

  /** Reset all state (e.g., when target changes). */
  reset() {
    this._prev = {
      ttft: null,
      itl: null,
      e2e: null,
      promptTokens: null,
      genTokens: null,
      specAccepted: null,
      specDrafted: null,
      prefixHits: null,
      prefixQueries: null,
      promptTokensTotal: null,
      genTokensTotal: null,
      time: 0,
    };
    this._rolling = [];
  }

  // ─── Prometheus parsing helpers ──────────────────────────

  /**
   * Extract a gauge value (sum across all label permutations).
   * @returns {number|null}
   */

  /**
   * Extract a label value from the first metric line whose name matches.
   * e.g. _extractLabelValue(body, "cache_config_info", "kv_cache_size_tokens")
   * returns 627733 for vllm:cache_config_info{...,kv_cache_size_tokens="627733"} 1.0
   */
  _extractLabelValue(body, metricName, labelName) {
    const re = new RegExp("^[^#\\n]*" + metricName + "\\{[^}]*" + labelName + '="([^"]+)"', "m");
    const m = body.match(re);
    if (!m) return null;
    const v = parseFloat(m[1]);
    return Number.isFinite(v) ? v : null;
  }

  _extractGauge(body, name) {
    return this._extractCounter(body, name);
  }

  /**
   * Extract a counter value (sum across all label permutations).
   * Matches `vllm:<name>{labels} <value>` or `vllm:<name> <value>`.
   * Excludes _bucket, _count, _sum, _created suffixed lines.
   * @returns {number|null}
   */
  _extractCounter(body, name) {
    const esc = name.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    // Match the exact metric name (not a suffix like _bucket/_count/_sum/_created)
    const re = new RegExp(
      `^vllm:${esc}(?:\\{[^}]*\\})?\\s+([\\d.eE+-]+)\\s*$`,
      "gm",
    );
    let sum = 0;
    let found = false;
    let m;
    while ((m = re.exec(body)) !== null) {
      const v = parseFloat(m[1]);
      if (Number.isFinite(v)) {
        sum += v;
        found = true;
      }
    }
    return found ? sum : null;
  }

  /**
   * Extract a histogram's count and sum.
   * @param {string} body
   * @param {string} name - base histogram name (e.g., "time_to_first_token_seconds")
   * @returns {{count: number, sum: number}|null}
   */
  _extractHistogram(body, name) {
    const count = this._extractCounter(body, `${name}_count`);
    const sum = this._extractCounter(body, `${name}_sum`);
    if (count == null && sum == null) return null;
    return {
      count: count ?? 0,
      sum: sum ?? 0,
    };
  }

  /**
   * Compute count and sum deltas between two histogram snapshots.
   * @returns {{count: number, sum: number}}
   */
  _histDelta(current, previous) {
    if (!current) return { count: 0, sum: 0 };
    if (!previous) {
      // First poll — report the full value as the delta (cumulative since server start)
      return { count: current.count, sum: current.sum };
    }
    return {
      count: Math.max(0, current.count - previous.count),
      sum: Math.max(0, current.sum - previous.sum),
    };
  }

  /**
   * Compute counter delta between two values.
   * @returns {number}
   */
  _counterDelta(current, previous) {
    if (current == null) return 0;
    if (previous == null) return current; // First poll
    return Math.max(0, current - previous);
  }

  /**
   * Extract per-position speculative decoding acceptance rates if available.
   * vLLM may expose these as `spec_decode_num_accepted_tokens_total{position="0"} ...`
   * or similar per-position counters.
   * @returns {number[]|null} Array of per-position acceptance rates, or null
   */
  _extractPerPositionAcceptance(body) {
    // Look for per-position accepted and drafted counters
    // Pattern: vllm:spec_decode_num_accepted_tokens_total{...position="N"} value
    const acceptedByPos = this._extractLabeledCounterByLabel(body, "spec_decode_num_accepted_tokens_total", "position");
    const draftedByPos = this._extractLabeledCounterByLabel(body, "spec_decode_num_draft_tokens_total", "position");

    if (!acceptedByPos && !draftedByPos) return null;

    const positions = new Set([
      ...Object.keys(acceptedByPos || {}),
      ...Object.keys(draftedByPos || {}),
    ]);
    if (positions.size === 0) return null;

    const result = [];
    for (const pos of [...positions].sort((a, b) => Number(a) - Number(b))) {
      const acc = acceptedByPos?.[pos] ?? 0;
      const draft = draftedByPos?.[pos] ?? 0;
      result.push({
        position: Number(pos),
        accepted: acc,
        drafted: draft,
        acceptanceRate: draft > 0 ? this._round(acc / draft, 4) : 0,
      });
    }
    return result;
  }

  /**
   * Extract a counter's values grouped by a specific label.
   * @returns {Record<string, number>|null}
   */
  _extractLabeledCounterByLabel(body, name, label) {
    const esc = name.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const re = new RegExp(
      `^vllm:${esc}\\{([^}]*)\\}\\s+([\\d.eE+-]+)\\s*$`,
      "gm",
    );
    const result = {};
    let found = false;
    let m;
    while ((m = re.exec(body)) !== null) {
      const labels = m[1];
      const value = parseFloat(m[2]);
      if (!Number.isFinite(value)) continue;

      // Extract the specific label value
      const labelMatch = new RegExp(`${label}="([^"]*)"`).exec(labels);
      if (labelMatch) {
        result[labelMatch[1]] = (result[labelMatch[1]] ?? 0) + value;
        found = true;
      }
    }
    return found ? result : null;
  }

  /** Compute averages over the rolling window. */
  _computeRollingAverages() {
    if (this._rolling.length === 0) {
      return {
        avgE2eLatency: null,
        avgTtft: null,
        avgTokensPerRequest: null,
        avgTpsPerSlot: null,
      };
    }

    const n = this._rolling.length;
    let sumE2e = 0, sumTtft = 0, sumTokens = 0, sumTps = 0;
    for (const r of this._rolling) {
      sumE2e += r.e2e;
      sumTtft += r.ttft;
      sumTokens += r.tokens;
      sumTps += r.tpsPerSlot;
    }

    return {
      avgE2eLatency: this._round(sumE2e / n, 6),
      avgTtft: this._round(sumTtft / n, 6),
      avgTokensPerRequest: this._round(sumTokens / n, 2),
      avgTpsPerSlot: this._round(sumTps / n, 2),
    };
  }

  /** Round to N decimal places. */
  _round(value, decimals) {
    const factor = Math.pow(10, decimals);
    return Math.round(value * factor) / factor;
  }
}