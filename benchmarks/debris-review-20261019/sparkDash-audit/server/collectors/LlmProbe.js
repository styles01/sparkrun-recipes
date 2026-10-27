/**
 * LlmProbe — probes an LLM server on port 8888, auto-detects backend,
 * computes live tokens/sec (generation + prefill).
 *
 * Ported from legacy `probeLlamaServerType` and `_getLlamaMetricsFor`.
 *
 * Supports vLLM, llama.cpp, sglang, and ds4 (DeepSeek-V4-Flash CUDA engine).
 * The ds4 backend is detected via /v1/models `owned_by: "ds4.c"` and exposes
 * its own ds4_* Prometheus metrics.
 */
import { LLM_PROBE_TIMEOUT_MS } from "../config.js";
import { execSync } from "node:child_process";
import { readFileSync, readdirSync, statSync, openSync, readSync, closeSync } from "node:fs";
import { classifyHostScope } from "../validate.js";
import { llmProbeHost } from "./llmHost.js";
import { VllmMetricsParser } from "./VllmMetricsParser.js";

const FAIL_RESET_THRESHOLD = 3;
const REDETECT_INTERVAL_MS = 60_000;
/** Current SGLang names first. Deprecated aliases still work but log a warning per hit. */
const SGLANG_SERVER_INFO_PATHS = ["/server_info", "/get_server_info"];
const SGLANG_MODEL_INFO_PATHS = ["/model_info", "/get_model_info"];
/**
 * SGLang's last_gen_throughput is a sticky gauge (holds last decode rate when
 * idle). Only treat it as live after we observe a change between polls, and
 * expire back to 0 if it stops changing.
 */
const SGLANG_STICKY_TPS_LIVE_MS = 6_000;

const HOST_ROOT = process.env.HOST_ROOT_PATH || "/host/root";
const HOST_PROC = process.env.HOST_PROC_PATH || "/host/proc";
const DS4_LOG_PATH = process.env.DS4_LOG_PATH || "/host/root/tmp/ds4-serve.log";
const KV_PAGE_SIZE_BYTES = 2048 * 1024; // 2048 KiB per page
// GB10 device total memory: 121 GiB (128 GB nominal, 121 GiB usable)
const DS4_DEVICE_MEMORY_BYTES = 121 * 1024 * 1024 * 1024;

function resolveLlamaLogPath() {
  const envPath = process.env.LLAMA_LOG_PATH;
  if (envPath) return envPath;
  try {
    const dir = "/host/root/tmp";
    const logs = readdirSync(dir)
      .filter((f) => /qwen.*\.log$/.test(f))
      .map((f) => ({ f, mtime: statSync(`${dir}/${f}`).mtimeMs }))
      .sort((a, b) => b.mtime - a.mtime);
    if (logs.length > 0) return `${dir}/${logs[0].f}`;
  } catch (e) {
    console.log("[llama-log] resolve error:", e && e.message);
  }
  return "/host/root/tmp/qwen4-q4-server.log";
}
const LLAMA_LOG_PATH = resolveLlamaLogPath();

/**
 * Prefer a short model id when the server returns a Hugging Face hub cache path.
 * e.g. /root/.cache/huggingface/hub/models--org--Name/snapshots/<hash>
 *   → org/Name
 * @param {unknown} id
 * @returns {string | null}
 */
export function normalizeModelId(id) {
  if (id == null) return null;
  const s = String(id).trim();
  if (!s) return null;

  const hub = s.match(/(?:^|\/)models--([^/]+?)(?:\/snapshots\/[^/]+)?\/?$/);
  if (hub) return hub[1].replace(/--/g, "/");

  const mid = s.match(/models--([^/]+)\/snapshots\//);
  if (mid) return mid[1].replace(/--/g, "/");

  return s;
}

/** True when `id` looks like a Hugging Face hub cache directory (models--org--name). */
export function isHfHubCachePath(id) {
  if (id == null) return false;
  return /(?:^|\/)models--[^/]+/.test(String(id));
}

/**
 * Set modelId (always normalized) and modelPath (omit HF hub cache paths —
 * they duplicate the short id and clutter the LLM panel).
 * @param {unknown} raw
 */
function applyModelRef(probe, raw) {
  if (raw == null || raw === "") return;
  const s = String(raw);
  probe.modelId = normalizeModelId(s);
  probe.modelPath = isHfHubCachePath(s) ? null : s;
}

export class LlmProbe {
  constructor(spark, port = 8888) {
    this.spark = spark;
    this.port = port;
    this.baseUrl = `http://${llmProbeHost(spark)}:${port}`;

    // State
    this.backendType = null; // 'vllm' | 'llama.cpp' | 'sglang' | 'ds4' | 'exl3' | 'q27' | 'tensorfold' | 'freetoken' | null
    this.serverIsOpenAI = null; // true = OpenAI-compatible
    /** Whether /v1/models (or /slots) answered without credentials. null = unknown. */
    this.authOpen = null;
    this.stepId = 0;
    this.modelId = null;
    this.modelPath = null;
    this.models = [];
    this.contextLength = null;
    this.gpuMemoryUtilization = null;
    this.slotsActive = 0;
    this.slotsTotal = 0;
    this.generationTps = 0;
    this.prefillTps = 0;
    /** Live cached-prefill tok/s when the backend splits kinds (ds4 / llama.cpp / sglang). null otherwise. */
    this.cachedPrefillTps = null;
    /** Live uncached/computed prefill tok/s when split is available. null otherwise. */
    this.uncachedPrefillTps = null;
    this.error = null;

    // Per-slot rate tracking (for llama.cpp native path)
    this.slotState = new Map();
    this.lastTokenCounts = { input: 0, output: 0 };
    /** TensorFold: first /health sample with counters seeds the baseline (lifetime total is not a rate). */
    this._tensorfoldSeeded = false;
    /** Previous TensorFold `prefill_seconds_total`. null until a sample includes it. */
    this._tensorfoldPrefillSeconds = null;
    this._tensorfoldLastPrefill = null;
    this.prefillActive = false;
    /** Previous prefill counters by kind; null until first labeled sample. */
    this.lastPrefillKinds = null;
    /** Previous vLLM TTFT histogram `_sum` (seconds). null until first sample. */
    this.lastTtftSum = null;
    /** Gauge families the last vLLM /metrics poll did not expose (panel UNAVAILABLE). */
    this._vllmSeriesUnavailable = null;
    /** Previous vLLM TTFT histogram `_count` (requests). null until first sample. */
    this.lastTtftCount = null;
    /** Previous `vllm:iteration_tokens_total_sum` (engine-step tokens). */
    this.lastIterSum = null;
    this.lastProbeTime = 0;

    // ── llama.cpp /slots expanded data surface ──
    this.promptTokens = null;          // n_prompt_tokens (total prompt tokens in current slot)
    this.promptTokensProcessed = null; // n_prompt_tokens_processed
    this.promptTokensCache = null;     // n_prompt_tokens_cache (tokens served from cache)
    this.cacheHitRatio = null;         // computed: cache / (cache + processed)
    this.nCtx = null;                  // n_ctx (context window size)
    this.isProcessing = false;         // any slot currently processing
    this.nRemain = null;               // next_token.n_remain (tokens remaining to generate)
    this.nDecoded = null;              // next_token.n_decoded (tokens decoded this request)
    this.samplingParams = null;        // params: temperature, top_k, top_p, min_p, etc.
    this.speculativeTypes = null;      // params["speculative.types"] e.g. "none,ngram-mod"
    this.reasoningFormat = null;        // params.reasoning_format
    this.chatFormat = null;            // params.chat_format
    this.samplers = null;              // params.samplers[]
    // ── llama.cpp speculative-decode (from server log) ──
    this.specAcceptanceRate = null;    // latest "draft acceptance = X"
    this.specAcceptedTokens = null;    // latest accepted count
    this.specGeneratedTokens = null;   // latest generated (drafted) count
    this.specMeanLen = null;           // latest mean draft length
    this._llamaLogSize = 0;            // last read position in llama log

    // Cumulative total output tokens (generation) as reported by the LLM server
    this.totalOutputTokens = 0;
    /** Cumulative total prompt (prefill) tokens as reported by the LLM server. */
    this.totalPromptTokens = null;
    /**
     * Cumulative cached (prefix-cache served) prompt tokens, when the backend
     * reports the split. Subset of totalPromptTokens (for ds4/q27 the raw
     * prefill counter is computed-only, so prompt = computed + cached).
     */
    this.totalCachedTokens = null;

    // vLLM inference metrics from /metrics (null when not vLLM / missing series)
    // Metric names follow stock vLLM Prometheus exposition (versions may differ).
    this.kvCacheUsage = null; // 0–1 fraction
    /** Allocated KV-cache pool, tokens (vLLM `cache_config_info`). */
    this.kvCacheTokens = null;
    /** Free tokens in that pool. null until both pool size and usage are known. */
    this.kvCacheTokensAvailable = null;
    /** Allocated KV-cache pool, bytes. */
    this.kvCacheMemoryBytes = null;
    /** Engine KV cache pool size in GB (SGLang only). null when not reported. */
    this.kvCacheGb = null;
    /** Engine model weights resident in GPU memory, GB (SGLang only). null when not reported. */
    this.weightsGb = null;
    this.requestsRunning = null;
    this.isPrefilling = null;
    this.requestsWaiting = null;
    this.engineUptimeSeconds = null;
    this.ttftP95Seconds = null;
    /** Live recent-window mean TTFT (seconds) from histogram sum/count deltas. null when unavailable. */
    this.ttftSeconds = null;
    this.preemptionsTotal = null; // cumulative counter
    /** Prefix cache hit rate 0–1 (hits/queries). */
    this.prefixCacheHitRate = null;
    this.e2eP95Seconds = null;
    this.itlP95Seconds = null;
    this.mtpAcceptanceRate = null;

    // ── llama.cpp /slots expanded data surface ──
    this.promptTokens = null;          // n_prompt_tokens (total prompt tokens in current slot)
    this.promptTokensProcessed = null; // n_prompt_tokens_processed
    this.promptTokensCache = null;     // n_prompt_tokens_cache (tokens served from cache)
    this.cacheHitRatio = null;         // computed: cache / (cache + processed)
    this.nCtx = null;                  // n_ctx (context window size)
    this.isProcessing = false;         // any slot currently processing
    this.nRemain = null;               // next_token.n_remain (tokens remaining to generate)
    this.nDecoded = null;              // next_token.n_decoded (tokens decoded this request)
    this.samplingParams = null;        // params: temperature, top_k, top_p, min_p, etc.
    this.speculativeTypes = null;      // params["speculative.types"] e.g. "none,ngram-mod"
    this.reasoningFormat = null;       // params.reasoning_format
    this.chatFormat = null;            // params.chat_format
    this.samplers = null;              // params.samplers[]
    // ── llama.cpp speculative-decode (from server log) ──
    this.specAcceptanceRate = null;    // latest "draft acceptance = X"
    this.specAcceptedTokens = null;    // latest accepted count
    this.specGeneratedTokens = null;   // latest generated (drafted) count
    this.specMeanLen = null;           // latest mean draft length
    this._llamaLogSize = 0;            // last read position in llama log

    // llama.cpp per-cycle deltas + rolling window (for latency/moving-avg derivation)
    this._llamaPrev = { decoded: 0, prompted: 0, time: 0 };
    this._llamaRolling = [];

    // Reasoning effort tracking (from ds4 request logs)
    this.reasoningEffort = null; // 'low' | 'medium' | 'high' | null
    this.reasoningEffortTs = null; // ms epoch when last set
    this.waitingSlots = null;
    this.ttft = null;
    this.e2eLatency = null;
    this.genTokensPerReq = null;
    this.mtpAcceptedTokens = null;
    this.mtpDraftedTokens = null;
    this.perPositionAcceptance = null;
    this.aggregateDecodeTps = null;
    this.rollingAvgE2e = null;
    this.rollingAvgTtft = null;
    this.rollingAvgTokensPerReq = null;
    this.rollingAvgTpsPerSlot = null;
    this.recipeInfo = null;
    this.recipeMetadata = null;
    this.peakAggregateTps = null;
    this.perStreamHigh = null;
    this.perStreamLow = null;
    this.perStreamAvg = null;
    this.totalTokensDecoded = null;
    this.dsparkAcceptRatio = null;
    this.banksLive = null;
    this.banksTotal = null;
    this.kvPagesResident = null;
    this.prefillCached = null;
    this.prefillComputed = null;
    this.specDrafts = null;
    this.specHits = null;
    this.warmRecords = null;
    this.specQuench = null;
    this.tokPerStep = null;
    this.decodeSteps = null;
    this.derivedArtifacts = null;
    this.derivedArtifactBytes = null;
    this.ds4Uptime = null;
    this.admitsCold = null;
    this.admitsWarm = null;
    this.admitsFork = null;
    this.admitsPartialFork = null;
    this.admitsPartialTruncate = null;
    this.requestsStarted = null;
    this.requestsCompleted = null;
    this.requestsFailed = null;
    this.requestsRefusedDeepSerial = null;
    this.requestsInflight = null;
    this.requestsSerial = null;
    this.contAdmitRejects = null;
    this.contBatchFailures = null;
    this.graphFitRefusals = null;
    this.contextUsedBytes = null;
    this._ds4LogSize = 0; // last read position in ds4 log file

    // Active context tracking (from ds4 request logs: ctx=0..N:N)
    this.activeContext = null; // total context tokens in the most recent request
    this.activeContextTs = null; // ms epoch when last seen
    this._ds4LogSizeCtx = 0; // separate read position for context tailing

    // DS4 per-cycle deltas + rolling window (for latency/moving-avg derivation)
    this._ds4Prev = {
      tokensDecoded: null,
      decodeSteps: null,
      requestsStarted: null,
      requestsCompleted: null,
      prefillComputed: null,
      prefillCached: null,
      time: 0,
    };
    this._ds4Rolling = [];
    this._consecutiveFailures = 0;
    this._lastDetectAt = 0;
    /** @type {{ value: number, liveUntil: number } | null} */
    this._sglangStickyTps = null;
    /** Latest gen_throughput from /v1/loads, if that payload carried one. */
    this._sglangLoadGenTps = null;
    /** Whether this poll's /server_info carried total_input/output_tokens. */
    this._sglangTotalsPolled = false;
    /**
     * Series that seeded `lastTokenCounts` — "server_info" or "prometheus".
     * They count the same work under different names, so a hand-off re-seeds
     * the baseline instead of differencing across the two.
     * @type {"server_info" | "prometheus" | null}
     */
    this._sglangTokenSource = null;
  }

  /**
   * Prefill is a short burst; decode then runs with Δprompt=0.
   * Keep the last real prefill rate while the engine is still in-flight, then 0.
   * @param {number} rate
   * @param {boolean} inflight
   */
  _setPrefillTps(rate, inflight) {
    const rounded = Math.max(0, Math.round(rate * 100) / 100);
    if (rounded > 0) this.prefillTps = rounded;
    else if (!inflight) this.prefillTps = 0;
  }

  /**
   * Live cached vs computed prefill tok/s from cumulative counters.
   * First sample seeds the baseline (0 tok/s). Missing either side clears the split.
   * @param {number|null|undefined} cachedCount
   * @param {number|null|undefined} computedCount
   * @param {number} dtSec
   */
  _setPrefillSplitRates(cachedCount, computedCount, dtSec) {
    if (cachedCount == null || computedCount == null || !Number.isFinite(cachedCount) || !Number.isFinite(computedCount)) {
      this.cachedPrefillTps = null;
      this.uncachedPrefillTps = null;
      this.lastPrefillKinds = null;
      return;
    }
    const total = cachedCount + computedCount;
    this.prefixCacheHitRate =
      total > 0 ? Math.round((cachedCount / total) * 10000) / 10000 : null;
    if (this.lastPrefillKinds == null) {
      this.lastPrefillKinds = { cached: cachedCount, computed: computedCount };
      this.cachedPrefillTps = 0;
      this.uncachedPrefillTps = 0;
      return;
    }
    if (dtSec > 0 && dtSec < 10) {
      const dCached = cachedCount - this.lastPrefillKinds.cached;
      const dComputed = computedCount - this.lastPrefillKinds.computed;
      this.cachedPrefillTps = Math.max(0, Math.round((dCached / dtSec) * 100) / 100);
      this.uncachedPrefillTps = Math.max(0, Math.round((dComputed / dtSec) * 100) / 100);
    }
    this.lastPrefillKinds = { cached: cachedCount, computed: computedCount };
  }

  /** Update probe port (and host from spark). Resets detection when the target changes. */
  setPort(port) {
    const next = Number(port);
    const prevUrl = this.baseUrl;
    if (Number.isInteger(next) && next >= 1 && next <= 65535) {
      this.port = next;
    }
    this.baseUrl = `http://${llmProbeHost(this.spark)}:${this.port}`;
    if (this.baseUrl !== prevUrl) {
      this._resetDetection();
      this._lastDetectAt = 0;
      this._consecutiveFailures = 0;
    }
  }

  /** Probe the LLM server and return a snapshot. */
  async probe() {
    try {
      const shouldDetect =
        this.serverIsOpenAI === null ||
        Date.now() - this._lastDetectAt > REDETECT_INTERVAL_MS;

      if (shouldDetect) {
        await this._detectServerType();
        this._lastDetectAt = Date.now();
      }

      if (this.serverIsOpenAI === false) {
        const snap = await this._probeLlamaCpp();
        this._noteSuccess();
        return snap;
      } else if (this.serverIsOpenAI === true) {
        if (this.backendType === "ds4") {
          const snap = await this._probeDs4();
          this._noteSuccess();
          return snap;
        }
        const snap = await this._probeOpenAICompatible();
        this._noteSuccess();
        return snap;
      } else {
        this._noteFailure("LLM server not reachable");
        return this._defaultLlm();
      }
    } catch (err) {
      this._noteFailure(err.message);
      return this._defaultLlm();
    }
  }

  _noteSuccess() {
    this._consecutiveFailures = 0;
    this.error = null;
  }

  _noteFailure(message) {
    this.models = [];
    this.error = message;
    this._consecutiveFailures += 1;
    if (this._consecutiveFailures >= FAIL_RESET_THRESHOLD) {
      this._resetDetection();
    }
  }

  _resetDetection() {
    this.serverIsOpenAI = null;
    this.backendType = null;
    this.authOpen = null;
    this.modelId = null;
    this.modelPath = null;
    this.models = [];
    this.generationTps = 0;
    this.prefillTps = 0;
    this.cachedPrefillTps = null;
    this.uncachedPrefillTps = null;
    this.contextLength = null;
    this.gpuMemoryUtilization = null;
    this.slotsActive = 0;
    this.slotsTotal = 0;
    this.totalOutputTokens = 0;
    this.totalPromptTokens = null;
    this.totalCachedTokens = null;
    this.kvCacheUsage = null;
    this._clearEngineMemory();
    this.requestsRunning = null;
    this.requestsWaiting = null;
    this.ttftP95Seconds = null;
    this.ttftSeconds = null;
    this.preemptionsTotal = null;
    this.prefixCacheHitRate = null;
    this.e2eP95Seconds = null;
    this.itlP95Seconds = null;
    this.mtpAcceptanceRate = null;
    // DS4
    this.ds4Uptime = null;
    this.peakAggregateTps = 0;
    this.perStreamHigh = null;
    this.perStreamLow = null;
    this.perStreamAvg = null;
    this.totalTokensDecoded = null;
    this.dsparkAcceptRatio = null;
    this.banksLive = null;
    this.banksTotal = null;
    this.kvPagesResident = null;
    this.prefillCached = null;
    this.prefillComputed = null;
    this.specDrafts = null;
    this.specHits = null;
    this.specQuench = null;
    this.warmRecords = null;
    this.derivedArtifacts = null;
    this.derivedArtifactBytes = null;
    this.requestsStarted = null;
    this.requestsCompleted = null;
    this.requestsFailed = null;
    this.requestsRefusedDeepSerial = null;
    this.requestsInflight = null;
    this.requestsSerial = null;
    this.contAdmitRejects = null;
    this.contBatchFailures = null;
    this.graphFitRefusals = null;
    this.admitsCold = null;
    this.admitsWarm = null;
    this.admitsFork = null;
    this.admitsPartialFork = null;
    this.admitsPartialTruncate = null;
    this.decodeSteps = null;
    this.tokPerStep = null;
    this.recipeMetadata = null;
    this.recipeInfo = null;
    this._ds4Prev = {
      tokensDecoded: null,
      decodeSteps: null,
      requestsStarted: null,
      requestsCompleted: null,
      prefillComputed: null,
      prefillCached: null,
      time: 0,
    };
    this._ds4Rolling = [];
    this.slotState.clear();
    this.lastTokenCounts = { input: 0, output: 0 };
    this._tensorfoldSeeded = false;
    this._tensorfoldPrefillSeconds = null;
    this._tensorfoldLastPrefill = null;
    this.prefillActive = false;
    this.lastPrefillKinds = null;
    this.lastTtftSum = null;
    this.lastTtftCount = null;
    this.lastIterSum = null;
    this._sglangStickyTps = null;
    // EXL3 (red-team m1): probe state must not leak across failovers
    this._exl3Seeded = false;
    this._exl3StreamSamples = 0;
    this.lastExl3TtftSum = null;
    this.lastExl3TtftCount = null;
    this.lastExl3E2eSum = null;
    this.lastExl3E2eCount = null;
    this.lastExl3ItlSum = null;
    this.lastExl3ItlCount = null;
    this.lastExl3Completed = null;
    this.lastExl3CompletedTokens = null;
    this.recipeInfo = null;
    this._vllmPrefixCaching = null;
    this.reasoningEffort = null; // 'low' | 'medium' | 'high' | null
    this.reasoningEffortTs = null; // ms epoch when last set
    this.waitingSlots = null;
    this.ttft = null;
    this.e2eLatency = null;
    this.genTokensPerReq = null;
    this.mtpAcceptedTokens = null;
    this.mtpDraftedTokens = null;
    this.perPositionAcceptance = null;
    this.aggregateDecodeTps = null;
    this.rollingAvgE2e = null;
    this.rollingAvgTtft = null;
    this.rollingAvgTokensPerReq = null;
    this.rollingAvgTpsPerSlot = null;
    this.recipeInfo = null;
    this.recipeMetadata = null;
    this.peakAggregateTps = null;
    this.perStreamHigh = null;
    this.perStreamLow = null;
    this.perStreamAvg = null;
    this.totalTokensDecoded = null;
    this.dsparkAcceptRatio = null;
    this.banksLive = null;
    this.banksTotal = null;
    this.kvPagesResident = null;
    this.prefillCached = null;
    this.prefillComputed = null;
    this.specDrafts = null;
    this.specHits = null;
    this.warmRecords = null;
    this.specQuench = null;
    this.tokPerStep = null;
    this.decodeSteps = null;
    this.derivedArtifacts = null;
    this.derivedArtifactBytes = null;
    this.ds4Uptime = null;
    this.admitsCold = null;
    this.admitsWarm = null;
    this.admitsFork = null;
    this.admitsPartialFork = null;
    this.admitsPartialTruncate = null;
    this.requestsStarted = null;
    this.requestsCompleted = null;
    this.requestsFailed = null;
    this.requestsRefusedDeepSerial = null;
    this.requestsInflight = null;
    this.requestsSerial = null;
    this.contAdmitRejects = null;
    this.contBatchFailures = null;
    this.graphFitRefusals = null;
    this.contextUsedBytes = null;
    this._ds4LogSize = 0; // last read position in ds4 log file
    this.activeContext = null; // total context tokens in the most recent request
    this.activeContextTs = null; // ms epoch when last seen
    this._ds4LogSizeCtx = 0; // separate read position for context tailing
    this._ds4Prev = {
      tokensDecoded: null,
      decodeSteps: null,
      requestsStarted: null,
      requestsCompleted: null,
      prefillComputed: null,
      prefillCached: null,
      time: 0,
    };
    this._ds4Rolling = [];
    this._consecutiveFailures = 0;
    this._lastDetectAt = 0;
    this._vllmMetricsParser = new VllmMetricsParser();
    this._sglangLoadGenTps = null;
  }

  /** Note auth from an HTTP status on an unauthenticated probe request. */
  _noteAuthStatus(status) {
    if (status >= 200 && status < 300) {
      this.authOpen = true;
      return "ok";
    }
    if (status === 401 || status === 403) {
      this.authOpen = false;
      return "auth";
    }
    return "other";
  }

  // ─── Server type detection ───────────────────────────────
  async _detectServerType() {
    // Skip the llama.cpp /slots probe once we've positively identified an
    // OpenAI-compatible backend. vLLM / sglang / ds4-server have no /slots,
    // so re-probing it on every re-detect cycle just spams 404s in the
    // backend's access log (#15). Still probe /slots on first contact, when
    // the type is unknown, or when the backend was previously llama.cpp.
    if (
      this.backendType !== "vllm" &&
      this.backendType !== "sglang" &&
      this.backendType !== "ds4" &&
      this.backendType !== "exl3" &&
      this.backendType !== "q27" &&
      this.backendType !== "tensorfold" &&
      this.backendType !== "freetoken"
    ) {
      const slotUrl = `${this.baseUrl}/slots`;
      try {
        const slotRes = await this._fetch(slotUrl);
        const auth = this._noteAuthStatus(slotRes.status);
        if (auth === "ok") {
          const slots = await slotRes.json();
          if (Array.isArray(slots)) {
            this.serverIsOpenAI = false;
            this.backendType = "llama.cpp";
            return;
          }
        } else if (auth === "auth") {
          // Authenticated llama.cpp — treat as protected OpenAI-style for posture
          this.serverIsOpenAI = false;
          this.backendType = "llama.cpp";
          return;
        }
      } catch {}
    }

    // Try OpenAI-compatible (vLLM, SGLang, or ds4-server)
    try {
      const modelRes = await this._fetch(`${this.baseUrl}/v1/models`);
      const auth = this._noteAuthStatus(modelRes.status);
      if (auth === "ok" || auth === "auth") {
        this.serverIsOpenAI = true;
        let owned = null;
        if (auth === "ok") {
          try {
            const modelsData = await modelRes.json();
            const records = Array.isArray(modelsData?.data) ? modelsData.data : [];
            owned = records.find((model) => typeof model?.id === "string" && model.id.trim().length > 0)?.owned_by;
          } catch {
            /* body optional for detection */
          }
        }
        this.backendType = await this._classifyOpenAIBackend(owned);
        return;
      }
    } catch {}

    this.serverIsOpenAI = null;
    this.backendType = null;
  }

  // ─── DS4 engine path ─────────────────────────────────────
  async _probeDs4() {
    const now = Date.now();
    const dtSec = (now - this.lastProbeTime) / 1000;
    this.lastProbeTime = now;

    // Model info from /v1/models
    let modelsOk = false;
    try {
      const modelsRes = await this._fetch(`${this.baseUrl}/v1/models`);
      if (modelsRes.ok) {
        modelsOk = true;
        const modelsData = await modelsRes.json();
        const model = modelsData?.data?.[0];
        this.modelId = model?.id || null;
        this.contextLength = model?.context_length || null;
        this.recipeMetadata = {
          name: model?.id || null,
          model: model?.name || null,
          contextLength: model?.context_length || null,
          ownedBy: model?.owned_by || null,
          supportedParameters: model?.supported_parameters || [],
        };
      }
    } catch {}

    if (!modelsOk) {
      throw new Error("ds4 /v1/models unreachable");
    }

    // Real reasoning effort from the engine itself (/v1/stats carries
    // server.reasoning_effort). Beats the log-tail guess.
    try {
      const statsRes = await this._fetch(`${this.baseUrl}/v1/stats`, {
        headers: { Accept: "application/json" },
      });
      if (statsRes.ok) {
        const stats = await statsRes.json();
        const effort = stats?.server?.reasoning_effort;
        if (effort) {
          this.reasoningEffort = effort;
          this.reasoningEffortTs = Date.now();
        }
      }
    } catch {}

    // Parse /metrics
    try {
      const metricsRes = await this._fetch(`${this.baseUrl}/metrics`);
      if (metricsRes.ok) {
        const txt = await metricsRes.text();

        // Gauges
        this.ds4Uptime = this._getDs4Metric(txt, "ds4_uptime_seconds");
        this.generationTps = this._getDs4Metric(txt, "ds4_decode_tok_s") ?? 0;
        this.prefillTps = this._getDs4Metric(txt, "ds4_prefill_tok_s") ?? 0;
        this.dsparkAcceptRatio = this._getDs4Metric(txt, "ds4_spec_accept_ratio");
        this.tokPerStep = this._getDs4Metric(txt, "ds4_tok_per_step");
        this.banksLive = this._getDs4Metric(txt, "ds4_banks_live");
        this.banksTotal = this._getDs4Metric(txt, "ds4_banks_total");
        this.kvPagesResident = this._getDs4Metric(txt, "ds4_kv_pages_resident");
        this.warmRecords = this._getDs4Metric(txt, "ds4_warm_records");
        this.derivedArtifacts = this._getDs4Metric(txt, "ds4_derived_artifacts");
        this.derivedArtifactBytes = this._getDs4Metric(txt, "ds4_derived_artifact_bytes");
        this.requestsInflight = this._getDs4Metric(txt, "ds4_requests_inflight");

        // Counters
        this.totalTokensDecoded = this._getDs4Metric(txt, "ds4_tokens_decoded_total");
        this.decodeSteps = this._getDs4Metric(txt, "ds4_decode_steps_total");
        this.specDrafts = this._getDs4Metric(txt, "ds4_spec_drafts_total");
        this.specHits = this._getDs4Metric(txt, "ds4_spec_hits_total");
        this.specQuench = this._getDs4Metric(txt, "ds4_spec_quench_total");
        this.requestsStarted = this._getDs4Metric(txt, "ds4_requests_started_total");
        this.requestsSerial = this._getDs4Metric(txt, "ds4_requests_serial_total");
        this.contAdmitRejects = this._getDs4Metric(txt, "ds4_cont_admit_rejects_total");
        this.contBatchFailures = this._getDs4Metric(txt, "ds4_cont_batch_failures_total");
        this.graphFitRefusals = this._getDs4Metric(txt, "ds4_graph_fit_refusals_total");

        // Labeled counters
        this.requestsCompleted = this._getDs4LabeledMetric(txt, "ds4_requests_total", "outcome", "completed");
        this.requestsFailed = this._getDs4LabeledMetric(txt, "ds4_requests_total", "outcome", "failed");
        this.requestsRefusedDeepSerial = this._getDs4LabeledMetric(txt, "ds4_requests_total", "outcome", "refused_deep_serial");
        this.prefillCached = this._getDs4LabeledMetric(txt, "ds4_tokens_prefilled_total", "kind", "cached");
        this.prefillComputed = this._getDs4LabeledMetric(txt, "ds4_tokens_prefilled_total", "kind", "computed");
        this.admitsCold = this._getDs4LabeledMetric(txt, "ds4_admits_total", "kind", "cold");
        this.admitsWarm = this._getDs4LabeledMetric(txt, "ds4_admits_total", "kind", "warm");
        this.admitsFork = this._getDs4LabeledMetric(txt, "ds4_admits_total", "kind", "fork");
        this.admitsPartialFork = this._getDs4LabeledMetric(txt, "ds4_admits_total", "kind", "partial_fork");
        this.admitsPartialTruncate = this._getDs4LabeledMetric(txt, "ds4_admits_total", "kind", "partial_truncate");

        // Slots = banks_live (active lanes), slotsTotal = banks_total
        this.slotsActive = this.banksLive != null ? Math.round(this.banksLive) : 0;
        this.slotsTotal = this.banksTotal != null ? Math.round(this.banksTotal) : 0;
        this.requestsRunning = this.requestsInflight;

        // Total output tokens from decoded counter
        if (this.totalTokensDecoded != null) {
          this.totalOutputTokens = Math.round(this.totalTokensDecoded);
        }

        // Track peak aggregate tok/s
        const currentAggregate = this.generationTps;
        if (currentAggregate > this.peakAggregateTps) {
          this.peakAggregateTps = currentAggregate;
        }

        // Per-stream tracking: use banks_live as the number of active streams
        // When inflight > 0, per-stream = decode_tok_s / inflight
        const inflight = this.requestsInflight != null ? this.requestsInflight : 0;
        if (inflight > 0 && currentAggregate > 0) {
          const perStream = currentAggregate / inflight;
          if (this.perStreamHigh == null || perStream > this.perStreamHigh) {
            this.perStreamHigh = Math.round(perStream * 100) / 100;
          }
          if (this.perStreamLow == null || perStream < this.perStreamLow) {
            this.perStreamLow = Math.round(perStream * 100) / 100;
          }
          this.perStreamAvg = Math.round(perStream * 100) / 100;
        }

        // MTP/spec acceptance — use ds4_spec_accept_ratio as the gauge
        this.mtpAcceptanceRate = this.dsparkAcceptRatio;
        this.mtpAcceptedTokens = this.specHits;
        this.mtpDraftedTokens = this.specDrafts;

        // ── Derive latency, genTokensPerReq, and rolling averages from
        //    ds4 counter deltas (ds4 has no latency histograms, so we
        //    approximate from throughput + completed-request counts). ──
        const nowMs = Date.now();
        const prev = this._ds4Prev;
        const dt = prev.time > 0 ? (nowMs - prev.time) / 1000 : 0;

        const deltaDecoded =
            this.totalTokensDecoded != null && prev.tokensDecoded != null
                ? Math.max(0, this.totalTokensDecoded - prev.tokensDecoded)
                : 0;
        const deltaSteps =
            this.decodeSteps != null && prev.decodeSteps != null
                ? Math.max(0, this.decodeSteps - prev.decodeSteps)
                : 0;
        const deltaStarted =
            this.requestsStarted != null && prev.requestsStarted != null
                ? Math.max(0, this.requestsStarted - prev.requestsStarted)
                : 0;
        const deltaCompleted =
            this.requestsCompleted != null && prev.requestsCompleted != null
                ? Math.max(0, this.requestsCompleted - prev.requestsCompleted)
                : 0;
        const deltaPrefillComputed =
            this.prefillComputed != null && prev.prefillComputed != null
                ? Math.max(0, this.prefillComputed - prev.prefillComputed)
                : 0;

        // Per-request average tokens (generation) — if requests completed
        // this cycle, avg tokens per request = deltaDecoded / deltaCompleted.
        // Fallback to cumulative if we have totals.
        if (deltaCompleted > 0) {
          this.genTokensPerReq =
              Math.round((deltaDecoded / deltaCompleted) * 100) / 100;
        } else if (this.requestsCompleted != null && this.requestsCompleted > 0) {
          this.genTokensPerReq =
              Math.round((this.totalTokensDecoded / this.requestsCompleted) * 100) / 100;
        }

        // Approximate TTFT: prefill time for the average request.
        // Use counter-based prefill rate (deltaPrefillComputed / dt) instead of
        // the instantaneous prefillTps gauge, which is near-zero between bursts.
        // When prefillTps is very low or zero (idle), fall back to the rolling
        // average prefill rate from the DS4 rolling window.
        const prefillRate = dt > 0 && deltaPrefillComputed > 0
            ? deltaPrefillComputed / dt
            : this.prefillTps > 0
                ? this.prefillTps
                : (this._ds4Rolling.length > 0
                    ? (() => {
                        // Estimate prefill rate from rolling window tokens and e2e
                        const last = this._ds4Rolling[this._ds4Rolling.length - 1];
                        return last && last.e2e > 0 ? last.tokens / last.e2e : 0;
                      })()
                    : 0);
        // Also compute average prompt tokens from cumulative counters as fallback
        // (used by the rolling-window prefill rate estimation above)
        if (deltaCompleted > 0 && deltaPrefillComputed > 0 && prefillRate > 0) {
          const avgPromptTokens = deltaPrefillComputed / deltaCompleted;
          this.ttft = Math.round((avgPromptTokens / prefillRate) * 1000) / 1000;
          this.ttftP95Seconds = this.ttft; // best estimate (no histogram)
        } else if (this.rollingAvgTtft != null && this.rollingAvgTtft > 0) {
          // Use rolling average TTFT from prior cycles — more reliable than
          // a cumulative estimate when prefill rate is very low or zero.
          this.ttft = this.rollingAvgTtft;
          this.ttftP95Seconds = this.rollingAvgTtft;
        }

        // Approximate E2E: TTFT + decode time for avg request.
        // decode time ≈ avgGenTokens / decodeRate, where decodeRate = generationTps / inflight.
        if (deltaCompleted > 0 && this.genTokensPerReq != null && this.genTokensPerReq > 0) {
          const inflight = this.requestsInflight != null ? Math.max(1, this.requestsInflight) : 1;
          const decodeRate = this.generationTps > 0 ? this.generationTps / inflight : 0;
          const ttftEst = this.ttft ?? 0;
          if (decodeRate > 0) {
            const decodeTime = this.genTokensPerReq / decodeRate;
            this.e2eLatency = Math.round((ttftEst + decodeTime) * 1000) / 1000;
            this.e2eP95Seconds = this.e2eLatency;
          } else {
            this.e2eLatency = Math.round(ttftEst * 1000) / 1000;
            this.e2eP95Seconds = this.e2eLatency;
          }
        }

        // ── Rolling window: last 10 completed-request batches ──
        if (deltaCompleted > 0 && this.e2eLatency != null) {
          const activeSlots = this.banksLive != null ? Math.max(1, this.banksLive) : 1;
          const tpsPerSlot = dt > 0 && this.generationTps > 0
              ? this.generationTps / activeSlots
              : 0;
          const tokensPerReq = (deltaPrefillComputed + deltaDecoded) / deltaCompleted;

          this._ds4Rolling.push({
            e2e: this.e2eLatency,
            ttft: this.ttft ?? 0,
            tokens: tokensPerReq,
            tpsPerSlot: Math.round(tpsPerSlot * 100) / 100,
          });
          if (this._ds4Rolling.length > 10) {
            this._ds4Rolling = this._ds4Rolling.slice(-10);
          }
        }

        // Compute rolling averages
        if (this._ds4Rolling.length > 0) {
          const n = this._ds4Rolling.length;
          let sumE2e = 0, sumTtft = 0, sumTokens = 0, sumTps = 0;
          for (const r of this._ds4Rolling) {
            sumE2e += r.e2e;
            sumTtft += r.ttft;
            sumTokens += r.tokens;
            sumTps += r.tpsPerSlot;
          }
          this.rollingAvgE2e = Math.round((sumE2e / n) * 1000) / 1000;
          this.rollingAvgTtft = Math.round((sumTtft / n) * 1000) / 1000;
          this.rollingAvgTokensPerReq = Math.round((sumTokens / n) * 100) / 100;
          this.rollingAvgTpsPerSlot = Math.round((sumTps / n) * 100) / 100;
        }

        // ── Derive prefixCacheHitRate from prefill counters ──
        // hit rate = prefillCached / (prefillCached + prefillComputed)
        if (this.prefillCached != null && this.prefillComputed != null) {
          const total = this.prefillCached + this.prefillComputed;
          this.prefixCacheHitRate =
            total > 0 ? Math.round((this.prefillCached / total) * 10000) / 10000 : null;
        }

        // ── Derive itlP95Seconds ≈ 1 / perStreamAvg (inter-token latency) ──
        if (this.perStreamAvg != null && this.perStreamAvg > 0) {
          this.itlP95Seconds = Math.round((1 / this.perStreamAvg) * 1000) / 1000;
        } else if (this.generationTps > 0) {
          // Fallback: use aggregate generation rate
          this.itlP95Seconds = Math.round((1 / this.generationTps) * 1000) / 1000;
        }

        // ── Derive kvCacheUsage and gpuMemoryUtilization from DS4 memory census ──
        // ds4_memory_bytes{domain="unified_device",class="kv_primary",state="allocated"}
        // = KV cache live bytes on device.
        // gpuMemoryUtilization = total unified_device allocated / 121GB
        const kvBytes = this._getDs4MultiLabeledMetric(txt,
            "ds4_memory_bytes",
            { domain: "unified_device", class: "kv_primary", state: "allocated" });
        if (kvBytes != null && kvBytes > 0) {
          this.kvCacheUsage = Math.round((kvBytes / DS4_DEVICE_MEMORY_BYTES) * 10000) / 10000;
        }
        // gpuMemoryUtilization: sum all unified_device allocated bytes / 121GB
        const totalDeviceAllocated = this._getDs4MemoryDomainTotal(txt, "unified_device", "allocated");
        if (totalDeviceAllocated != null && totalDeviceAllocated > 0) {
          this.gpuMemoryUtilization = Math.round((totalDeviceAllocated / DS4_DEVICE_MEMORY_BYTES) * 10000) / 10000;
        }

        // ── Derive perPositionAcceptance as single-element array from overall ratio ──
        // DS4 doesn't break down spec acceptance by position, so we provide a
        // single-element array so the spec decode graph always has data.
        if (this.dsparkAcceptRatio != null) {
          this.perPositionAcceptance = [Math.round(this.dsparkAcceptRatio * 10000) / 10000];
        } else if (this.specHits != null && this.specDrafts != null && this.specDrafts > 0) {
          this.perPositionAcceptance = [Math.round((this.specHits / this.specDrafts) * 10000) / 10000];
        }

        // Aggregate decode TPS alias
        this.aggregateDecodeTps = this.generationTps;

        // Save state for next cycle
        this._ds4Prev = {
          tokensDecoded: this.totalTokensDecoded,
          decodeSteps: this.decodeSteps,
          requestsStarted: this.requestsStarted,
          requestsCompleted: this.requestsCompleted,
          prefillComputed: this.prefillComputed,
          prefillCached: this.prefillCached,
          time: nowMs,
        };
      }
    } catch {}

    this.backendType = "ds4";
    this._collectRecipeInfo();
    return this._getSnapshot();
  }

  /**
   * Classify an OpenAI-compatible server: ds4, SGLang, EXL3, q27, TensorFold, FreeToken, or vLLM (default).
   * @param {unknown} ownedBy
   * @returns {Promise<"ds4" | "sglang" | "exl3" | "q27" | "tensorfold" | "freetoken" | "vllm">}
   */
  async _classifyOpenAIBackend(ownedBy) {
    if (typeof ownedBy === "string") {
      if (/ds4/i.test(ownedBy)) return "ds4";
      if (/sglang/i.test(ownedBy)) return "sglang";
      if (/exl3/i.test(ownedBy)) return "exl3";
      // q27's /v1/models reports owned_by: "q27" (signalnine/q27 engine).
      if (/q27/i.test(ownedBy)) return "q27";
      // TensorFold (ashhart/TensorFold) reports owned_by: "tensorfold" on both MLX and CUDA servers.
      if (/tensorfold/i.test(ownedBy)) return "tensorfold";
      // FreeToken reports owned_by: "FreeToken".
      if (/freetoken/i.test(ownedBy)) return "freetoken";
    }
    if (await this._probeIsDs4()) return "ds4";
    if (await this._probeIsSglang()) return "sglang";
    if (await this._probeIsExl3()) return "exl3";
    if (await this._probeIsQ27()) return "q27";
    if (await this._probeIsFreeToken()) return "freetoken";
    return "vllm";
  }

  /**
   * True when `/v1/stats` matches FreeToken's throughput + lifetime token contract.
   * A bare JSON object is not enough.
   */
  async _probeIsFreeToken() {
    try {
      const res = await this._fetch(`${this.baseUrl}/v1/stats`);
      if (!res.ok) return false;
      const data = await res.json().catch(() => null);
      return LlmProbe._statsLookLikeFreeToken(data);
    } catch {
      return false;
    }
  }

  /** @param {unknown} data */
  static _statsLookLikeFreeToken(data) {
    if (!data || typeof data !== "object" || Array.isArray(data)) return false;
    const throughput = data.throughput;
    const requests = data.requests;
    if (!throughput || typeof throughput !== "object" || Array.isArray(throughput)) return false;
    if (!requests || typeof requests !== "object" || Array.isArray(requests)) return false;
    return (
      Number.isFinite(Number(throughput.decode_tps)) &&
      Number.isFinite(Number(throughput.prefill_tps)) &&
      Number.isFinite(Number(requests.prompt_tokens_total)) &&
      Number.isFinite(Number(requests.completion_tokens_total))
    );
  }

  /**
   * True when EXL3 `tools/serve_openai.py` /health exposes backend + token totals.
   * Distinguishes from vLLM's /health (no `busy` + `completion_tokens_total`).
   */
  async _probeIsExl3() {
    try {
      const res = await this._fetch(`${this.baseUrl}/health`);
      if (!res.ok) return false;
      const data = await res.json().catch(() => null);
      return LlmProbe._healthLooksLikeExl3(data);
    } catch {
      return false;
    }
  }

  /** @param {unknown} data */
  static _healthLooksLikeExl3(data) {
    if (!data || typeof data !== "object" || Array.isArray(data)) return false;
    if (data.backend === "exl3") return true;
    // tools/serve_openai.py: { ok, busy } even before token counters existed.
    return data.ok === true && typeof data.busy === "boolean";
  }

  /**
   * First successful SGLang JSON from `paths` (current name, then deprecated alias).
   * @param {string[]} paths
   * @returns {Promise<object|null>}
   */
  async _fetchSglangJson(paths) {
    for (const path of paths) {
      try {
        const res = await this._fetch(`${this.baseUrl}${path}`);
        if (!res.ok) continue;
        const data = await res.json().catch(() => null);
        if (data && typeof data === "object" && !Array.isArray(data)) return data;
      } catch {
        /* try next */
      }
    }
    return null;
  }

  /** True when SGLang native server-info endpoints respond. */
  async _probeIsSglang() {
    return (await this._fetchSglangJson(SGLANG_SERVER_INFO_PATHS)) != null;
  }

  /** True when Prometheus /metrics exposes ds4-server series (ds4-on-spark). */
  async _probeIsDs4() {
    try {
      const res = await this._fetch(`${this.baseUrl}/metrics`);
      if (!res.ok) return false;
      const txt = await res.text();
      return LlmProbe._metricsLookLikeDs4(txt);
    } catch {
      return false;
    }
  }

  /** @param {string} body */
  static _metricsLookLikeDs4(body) {
    return /(?:^|\n)ds4_tokens_decoded_total(?:\{|\s)/m.test(String(body || ""));
  }

  /** True when Prometheus /metrics exposes q27-series (signalnine/q27 engine). */
  async _probeIsQ27() {
    try {
      const res = await this._fetch(`${this.baseUrl}/metrics`);
      if (!res.ok) return false;
      const txt = await res.text();
      return LlmProbe._metricsLookLikeQ27(txt);
    } catch {
      return false;
    }
  }

  /** @param {string} body */
  static _metricsLookLikeQ27(body) {
    return /(?:^|\n)q27_(?:decode_tokens_(?:processed_)?total|requests_total)(?:\{|\s)/m.test(
      String(body || "")
    );
  }

  // ─── OpenAI-compatible path (vLLM/sglang/ds4) ────────────
  async _probeOpenAICompatible() {
    const now = Date.now();
    const dtSec = (now - this.lastProbeTime) / 1000;
    this.lastProbeTime = now;

    // Model info from /v1/models — 401/403 means protected; other failure = down
    let modelsOk = false;
    let owned = null;
    let servedModelId = null;
    this.models = [];
    try {
      const modelsRes = await this._fetch(`${this.baseUrl}/v1/models`);
      const auth = this._noteAuthStatus(modelsRes.status);
      if (auth === "auth") {
        return this._getSnapshot();
      }
      if (auth === "ok") {
        modelsOk = true;
        const modelsData = await modelsRes.json();
        const records = Array.isArray(modelsData?.data) ? modelsData.data : [];
        const validModels = records.filter((model) =>
          typeof model?.id === "string" && model.id.trim().length > 0
        );
        this.models = [...new Set(validModels.map((model) => model.id))];
        const model = validModels[0];
        servedModelId = this.models.length > 1
          ? model.id
          : normalizeModelId(model?.id || null);
        this.modelId = servedModelId;
        // Only native enrichment owns modelPath, never the served ID list.
        this.modelPath = null;
        // ds4-server uses context_length; vLLM uses max_model_len
        this.contextLength =
          model?.max_model_len ?? model?.context_length ?? null;
        owned = model?.owned_by;
      }
    } catch {}

    if (!modelsOk) {
      throw new Error("OpenAI-compatible /v1/models unreachable");
    }

    // Self-heal backend from owned_by before branching (cheap, no extra HTTP)
    if (typeof owned === "string") {
      if (/ds4/i.test(owned)) this.backendType = "ds4";
      else if (/sglang/i.test(owned) && this.backendType !== "ds4") {
        this.backendType = "sglang";
      } else if (/exl3/i.test(owned) && this.backendType !== "ds4") {
        this.backendType = "exl3";
      } else if (/tensorfold/i.test(owned) && this.backendType !== "ds4") {
        this.backendType = "tensorfold";
      } else if (/freetoken/i.test(owned) && this.backendType !== "ds4") {
        this.backendType = "freetoken";
      }
    }

    if (this.backendType === "freetoken") {
      // FreeToken reports no engine memory split; never keep another engine's.
      this._clearEngineMemory();
      try {
        const statsRes = await this._fetch(`${this.baseUrl}/v1/stats`);
        // A protected /v1/stats is "needs a key", not "no data".
        this._noteAuthStatus(statsRes.status);
        if (!statsRes.ok) {
          this._clearFreeTokenRates();
        } else {
          const stats = await statsRes.json().catch(() => null);
          this._applyFreeTokenStats(stats);
        }
      } catch {
        this._clearFreeTokenRates();
      }
      return this._getSnapshot();
    }

    // Engine pool sizes: only SGLang and vLLM report them, and they re-read
    // them every poll below, so a size never outlives the payload it came from.
    this._clearEngineMemory();

    // TensorFold: no Prometheus. /health carries cumulative token totals when the
    // server publishes them; without them tok/s stays 0 rather than guessing.
    if (this.backendType === "tensorfold") {
      try {
        const healthRes = await this._fetch(`${this.baseUrl}/health`);
        if (healthRes.ok) {
          const health = await healthRes.json().catch(() => null);
          this._applyTensorFoldHealth(health, dtSec);
          this._applyTensorFoldLive(health);
          // Newer builds report live rates in /health but cumulative totals only on Prometheus /metrics.
          const totalRaw = health?.completion_tokens_total;
          if (totalRaw == null || !Number.isFinite(Number(totalRaw))) await this._applyTensorFoldMetrics();
        } else {
          this._applyTensorFoldHealth(null, dtSec);
        }
      } catch {
        this._applyTensorFoldHealth(null, dtSec);
      }
      return this._getSnapshot();
    }

    // EXL3 serve_openai.py: live tok/s from /health cumulative counters (no Prometheus).
    if (this.backendType === "exl3" || this.backendType == null) {
      try {
        const healthRes = await this._fetch(`${this.baseUrl}/health`);
        if (healthRes.ok) {
          const health = await healthRes.json().catch(() => null);
          if (LlmProbe._healthLooksLikeExl3(health)) {
            this.backendType = "exl3";
            this._applyExl3Health(health, dtSec);
            return this._getSnapshot();
          }
        }
      } catch {
        /* health optional unless already classified exl3 */
      }
      if (this.backendType === "exl3") return this._getSnapshot();
    }

    // SGLang: native info endpoints. Skip on known vLLM/ds4 to avoid 404 spam.
    // Prefer /server_info — /get_server_info is a deprecated alias that logs every poll.
    if (this.backendType === "sglang" || this.backendType == null) {
      this._sglangTotalsPolled = false;
      const sgData = await this._fetchSglangJson(SGLANG_SERVER_INFO_PATHS);
      if (sgData) {
        this.backendType = "sglang";
        // Load before last_gen_throughput so inflight can keep a steady rate live.
        await this._probeSglangLoad();
        this._applySglangServerInfo(sgData, dtSec);
        // Engine tile: Active vs Sleeping. SGLang has no live sleep gauge
        // (sleep_on_idle is a launch flag, not current state). A reachable
        // server with weights resident is Active / ready.
        if (this.gpuMemoryUtilization == null) this.gpuMemoryUtilization = 1;
      }
    }

    if (this.backendType === "sglang") {
      // KV fill comes from /metrics only; unknown when that poll misses it.
      this.kvCacheUsage = null;
      // Prometheus is optional (--enable-metrics). Do not mix those counters
      // into lastTokenCounts when this poll's server-info totals own them.
      // Gate on which series answered, never on the displayed rates: a live
      // rate keeps the split path selected forever, and the split path is not
      // allowed to clear prefillTps — the value then latches (#99).
      try {
        const metricsRes = await this._fetch(`${this.baseUrl}/metrics`);
        if (metricsRes.ok) {
          const txt = await metricsRes.text();
          this._applySglangKvMetrics(txt);
          if (this._sglangTotalsPolled) this._applySglangPrefillSplit(txt, dtSec);
          else this._applySglangMetrics(txt, dtSec);
        }
      } catch {
        /* metrics optional */
      }
      await this._enrichSglangModelInfo();
      await this._collectRecipeInfo();
      // Native SGLang endpoints expose the storage path (often just `/model`).
      // Keep that as modelPath, but show/use the client-facing served model ID.
      if (servedModelId) this.modelId = servedModelId;
      return this._getSnapshot();
    }

    // Single /metrics fetch: ds4-server or vLLM Prometheus exposition
    try {
      const metricsRes = await this._fetch(`${this.baseUrl}/metrics`);
      if (metricsRes.ok) {
        const txt = await metricsRes.text();
        if (
          this.backendType === "ds4" ||
          LlmProbe._metricsLookLikeDs4(txt)
        ) {
          this.backendType = "ds4";
          this._applyDs4Metrics(txt, dtSec);
        } else if (
          this.backendType === "q27" ||
          LlmProbe._metricsLookLikeQ27(txt)
        ) {
          this.backendType = "q27";
          this._applyQ27Metrics(txt, dtSec);
        } else {
          this.backendType = "vllm";
          this._applyVllmMetrics(txt, dtSec);
        }
      } else if (
        this.backendType !== "ds4" &&
        this.backendType !== "exl3" &&
        this.backendType !== "q27"
      ) {
        this.backendType = "vllm";
      }
    } catch {
      if (
        this.backendType !== "ds4" &&
        this.backendType !== "exl3" &&
        this.backendType !== "q27"
      ) {
        this.backendType = "vllm";
      }
    }
    if (this.backendType === "vllm" || this.backendType === "sglang") await this._collectRecipeInfo();

    return this._getSnapshot();
  }

  /**
   * Apply ds4-server Prometheus /metrics (Entrpi/ds4-on-spark).
   * Live tok/s from counter diffs (same as vLLM) so idle → 0. The engine's
   * `ds4_decode_tok_s` / `ds4_prefill_tok_s` gauges are ~60s windows and stay
   * non-zero long after requests finish — do not use them for the live panel.
   * @param {string} txt
   * @param {number} dtSec
   */
  _applyDs4Metrics(txt, dtSec) {
    const decoded = this._getPromMetric(txt, "ds4_tokens_decoded_total");
    const computedPrefill = this._getPromMetricLabeled(
      txt,
      "ds4_tokens_prefilled_total",
      "kind",
      "computed"
    );
    const prefilled =
      computedPrefill ?? this._getPromMetric(txt, "ds4_tokens_prefilled_total");
    const inflightHint = this._getPromMetric(txt, "ds4_requests_inflight");
    const inflight = inflightHint != null && inflightHint > 0;
    // Cumulative cached prefill for the token ledger (kind="cached" label).
    const cachedTotal = this._getPromMetricLabeled(
      txt,
      "ds4_tokens_prefilled_total",
      "kind",
      "cached"
    );

    if (decoded != null) {
      if (prefilled != null && dtSec > 0 && dtSec < 10) {
        const deltaIn = prefilled - this.lastTokenCounts.input;
        const deltaOut = decoded - this.lastTokenCounts.output;
        this.generationTps = Math.max(0, Math.round((deltaOut / dtSec) * 100) / 100);
        this.prefillTps = Math.max(0, Math.round((deltaIn / dtSec) * 100) / 100);
      } else if (dtSec > 0 && dtSec < 10) {
        const deltaOut = decoded - this.lastTokenCounts.output;
        this.generationTps = Math.max(0, Math.round((deltaOut / dtSec) * 100) / 100);
        if (!inflight && deltaOut <= 0) this.prefillTps = 0;
      }
      if (prefilled != null) this.lastTokenCounts.input = prefilled;
      this.lastTokenCounts.output = decoded;
      this.totalOutputTokens = decoded;
      // ds4's prefill counter is COMPUTED-only when the kind labels exist;
      // the token ledger wants the full prompt, so add the cached share.
      if (prefilled != null) {
        this.totalPromptTokens =
          cachedTotal != null && computedPrefill != null
            ? prefilled + cachedTotal
            : prefilled;
      }
      if (cachedTotal != null) this.totalCachedTokens = cachedTotal;
    } else {
      // No counters — fall back to window gauges only while something is in flight
      const gaugeGen = this._getPromMetric(txt, "ds4_decode_tok_s");
      const gaugePrefill = this._getPromMetric(txt, "ds4_prefill_tok_s");
      if (inflight) {
        if (gaugeGen != null) {
          this.generationTps = Math.max(0, Math.round(gaugeGen * 100) / 100);
        }
        if (gaugePrefill != null) {
          this.prefillTps = Math.max(0, Math.round(gaugePrefill * 100) / 100);
        }
      } else {
        this.generationTps = 0;
        this.prefillTps = 0;
      }
    }

    const inflightCount = this._getPromMetric(txt, "ds4_requests_inflight");
    this.requestsRunning = inflightCount;
    if (inflightCount != null) this.slotsActive = Math.round(inflightCount);

    const banksTotal = this._getPromMetric(txt, "ds4_banks_total");
    if (banksTotal != null) this.slotsTotal = Math.round(banksTotal);

    const specAccept = this._getPromMetric(txt, "ds4_spec_accept_ratio");
    this.mtpAcceptanceRate =
      specAccept != null ? Math.round(specAccept * 10000) / 10000 : null;

    // Prefix-cache hit rate + live tok/s from prefill kind labels
    const cached = this._getPromMetricLabeled(
      txt,
      "ds4_tokens_prefilled_total",
      "kind",
      "cached"
    );
    const computed = this._getPromMetricLabeled(
      txt,
      "ds4_tokens_prefilled_total",
      "kind",
      "computed"
    );
    this._setPrefillSplitRates(cached, computed, dtSec);

    // Clear tiles that are vLLM-histogram-specific (no ds4 equivalent yet)
    this.kvCacheUsage = null;
    this._clearKvPool();
    this.requestsWaiting = null;
    this.ttftP95Seconds = null;
    this.ttftSeconds = null;
    this.preemptionsTotal = null;
    this.e2eP95Seconds = null;
    this.itlP95Seconds = null;
  }

  /**
   * Apply q27 (signalnine/q27 engine) Prometheus /metrics.
   *
   * The engine exposes the [req]-universe telemetry that was previously only
   * in stderr logs, under q27_* series with an api= label (chat / completions /
   * messages / responses). The probe sums across label sets, exactly like the
   * ds4/vLLM paths: live tok/s from counter deltas so idle → 0.
   *
   * Semantics vs the vLLM path: prefill accounting is EXACT (per-request
   * token counts, not vLLM's estimates) and the prefix split (computed/cached)
   * doubles as the prefix-cache hit rate; the main prefill tile follows the
   * ds4 convention and counts COMPUTED tokens only. The waiting tile stays
   * null → the panel shows "—" (q27 FIFO-queues, no scheduler wait), while
   * preemptions are exposed as a constant-0 counter so Preempts reads 0.
   * @param {string} txt
   * @param {number} dtSec
   */
  _applyQ27Metrics(txt, dtSec) {
    // Live processed counters first (move during generation -> real-time
    // tok/s); fall back to the completion-based per-api totals for older
    // q27 binaries (step function: 0 during generation, jump at completion).
    const decoded =
      this._getPromMetric(txt, "q27_decode_tokens_processed_total") ??
      this._getPromMetric(txt, "q27_decode_tokens_total");
    // Exact prefill (not estimated like vLLM). Follow the ds4 convention:
    // the main prefill tile counts COMPUTED tokens only -- cache-served
    // tokens go to the cached/uncached split below, so a cache hit does not
    // inflate the "real work" rate.
    const computed =
      this._getPromMetric(txt, "q27_prefill_computed_tokens_processed_total") ??
      this._getPromMetric(txt, "q27_prefill_computed_tokens_total");
    if (decoded != null) {
      if (dtSec > 0 && dtSec < 10) {
        const deltaOut = decoded - this.lastTokenCounts.output;
        this.generationTps = Math.max(0, Math.round((deltaOut / dtSec) * 100) / 100);
        if (computed != null) {
          const deltaIn = computed - this.lastTokenCounts.input;
          this.prefillTps = Math.max(0, Math.round((deltaIn / dtSec) * 100) / 100);
        }
      }
      if (computed != null) this.lastTokenCounts.input = computed;
      this.lastTokenCounts.output = decoded;
      this.totalOutputTokens = decoded;
      // q27's prefill counter is computed-only; the token ledger wants the
      // full prompt (computed + cached).
      const cachedTok =
        this._getPromMetric(txt, "q27_prefill_cached_tokens_processed_total") ??
        this._getPromMetric(txt, "q27_prefill_cached_tokens_total");
      if (computed != null) {
        this.totalPromptTokens = cachedTok != null ? computed + cachedTok : computed;
      }
      if (cachedTok != null) this.totalCachedTokens = cachedTok;
    }

    const inflight = this._getPromMetric(txt, "q27_requests_inflight");
    this.requestsRunning = inflight;
    if (inflight != null) this.slotsActive = Math.round(inflight);

    const slotsTotal = this._getPromMetric(txt, "q27_slots_total");
    if (slotsTotal != null) this.slotsTotal = Math.round(slotsTotal);

    this.kvCacheUsage = this._getPromMetric(txt, "q27_kv_usage_perc");
    this._clearKvPool();
    this.requestsWaiting = null; // not exposed: q27 FIFO-queues, no wait gauge
    // q27 never preempts (FIFO admission) — the server exposes a constant-0
    // counter, so the Preempts tile reads 0 instead of "—".
    this.preemptionsTotal = this._getPromMetric(txt, "q27_preemptions_total");
    // Engine state: q27 keeps weights resident and is ready whenever the
    // server is up (no sleep state / memory release), so report Active like
    // the SGLang path does.
    if (this.gpuMemoryUtilization == null) this.gpuMemoryUtilization = 1;

    // Histograms (cumulative buckets, +Inf == _count by construction).
    const ttftHist = this._parseHistogram(
      txt,
      "q27_ttft_seconds",
      "q27_ttft_seconds_count"
    );
    const ttftP95 = this._histogramQuantile(ttftHist.buckets, ttftHist.total, 0.95);
    this.ttftP95Seconds = ttftP95 == null ? null : Math.round(ttftP95 * 1000) / 1000;

    const e2eHist = this._parseHistogram(
      txt,
      "q27_e2e_seconds",
      "q27_e2e_seconds_count"
    );
    const e2eP95 = this._histogramQuantile(e2eHist.buckets, e2eHist.total, 0.95);
    this.e2eP95Seconds = e2eP95 == null ? null : Math.round(e2eP95 * 1000) / 1000;

    const itlHist = this._parseHistogram(
      txt,
      "q27_itl_seconds",
      "q27_itl_seconds_count"
    );
    const itlP95 = this._histogramQuantile(itlHist.buckets, itlHist.total, 0.95);
    this.itlP95Seconds = itlP95 == null ? null : Math.round(itlP95 * 1000) / 1000;

    // Prefix-cache hit rate + live cached/uncached prefill split (live
    // processed counters, with completion-based fallback).
    const cachedSplit =
      this._getPromMetric(txt, "q27_prefill_cached_tokens_processed_total") ??
      this._getPromMetric(txt, "q27_prefill_cached_tokens_total");
    const computedSplit =
      this._getPromMetric(txt, "q27_prefill_computed_tokens_processed_total") ??
      this._getPromMetric(txt, "q27_prefill_computed_tokens_total");
    this._setPrefillSplitRates(cachedSplit, computedSplit, dtSec);

    const specAccept = this._getPromMetric(txt, "q27_spec_accept_ratio");
    this.mtpAcceptanceRate =
      specAccept != null ? Math.round(specAccept * 10000) / 10000 : null;
  }

  /**
   * Apply EXL3 serve_openai.py GET /health (v2 contract, audit 2026-09-21).
   * Live tok/s from cumulative counter diffs so idle → 0. Latency/acceptance
   * stats are engine-measured cumulative sums — the probe derives mean values
   * from sum/count deltas per poll window (same shape as vLLM histograms).
   * @param {Record<string, unknown>} data
   * @param {number} dtSec
   */
  _applyExl3Health(data, dtSec) {
    const prompt = Number(data?.prompt_tokens_total);
    const completion = Number(data?.completion_tokens_total);
    const busy = data?.busy === true;
    const ctx = Number(data?.context_length);
    if (Number.isFinite(ctx) && ctx > 0) this.contextLength = Math.round(ctx);

    this.requestsRunning = busy ? 1 : 0;
    this.slotsActive = busy ? 1 : 0;
    this.slotsTotal = 1;
    // Real prefill state from the engine (shim v5: set at enqueue, cleared at first
    // token + job end). Fallback: prefill rate visible this window.
    this.isPrefilling =
      data?.is_prefilling === true ||
      (data?.is_prefilling == null && Number(data?.prompt_tokens_total) !== Number(data?.completion_tokens_total) && this.prefillTps > 0);
    this.preemptionsTotal = null; // engine has no preemption concept (m2: clear, don't leak)
    // Single-stream engine: prefill and decode never overlap → no cached/uncached split.
    this.cachedPrefillTps = null;
    this.uncachedPrefillTps = null;

    // ── Live cache stats (GEN.get_cache_stats(), fail-open nulls) ──
    this.kvCacheUsage =
      data?.kv_cache_usage != null && Number.isFinite(Number(data.kv_cache_usage)) && Number(data.kv_cache_usage) >= 0
        ? Number(data.kv_cache_usage) : null;
    this.prefixCacheHitRate =
      data?.prefix_cache_hit_rate != null && Number.isFinite(Number(data.prefix_cache_hit_rate)) && Number(data.prefix_cache_hit_rate) >= 0
        ? Number(data.prefix_cache_hit_rate) : null;
    const waiting = Number(data?.requests_waiting);
    this.requestsWaiting = Number.isFinite(waiting) ? Math.round(waiting) : 0;

    // ── Request counters ──
    const completed = Number(data?.requests_completed_total);
    if (Number.isFinite(completed)) {
      this.requestsStarted = Math.round(completed + (Number(data?.requests_failed_total) || 0));
      this.requestsCompleted = Math.round(completed);
      this.requestsFailed = Math.round(Number(data?.requests_failed_total) || 0);
    }
    const ctxLast = Number(data?.context_last);
    if (Number.isFinite(ctxLast) && ctxLast > 0) {
      this.activeContext = Math.round(ctxLast);
      this.activeContextTs = Date.now();
    }

    // ── MTP / speculative acceptance (cumulative counters → rate + deltas) ──
    const mtpAcc = Number(data?.mtp_accepted_tokens_total);
    const mtpDrf = Number(data?.mtp_drafted_tokens_total);
    if (Number.isFinite(mtpAcc) && Number.isFinite(mtpDrf)) {
      this.mtpAcceptedTokens = Math.round(mtpAcc);
      this.mtpDraftedTokens = Math.round(mtpDrf);
      this.mtpAcceptanceRate =
        mtpDrf > 0 ? Math.round((mtpAcc / mtpDrf) * 10000) / 10000 : null;
      if (this.mtpAcceptanceRate != null) {
        // Real per-position bars from the engine when it reports them; else the
        // legacy single overall-ratio element (Pos 0 only).
        const byPos = Array.isArray(data?.mtp_accept_by_position) ? data.mtp_accept_by_position : null;
        if (byPos && byPos.length > 0) {
          this.perPositionAcceptance = byPos
            .slice()
            .sort((a, b) => (a?.position ?? 0) - (b?.position ?? 0))
            .map((e) => (e && e.rate != null ? e.rate : null));
        } else {
          this.perPositionAcceptance = [this.mtpAcceptanceRate];
        }
      }
    } else {
      this.mtpAcceptanceRate = null;
      this.mtpAcceptedTokens = null;
      this.mtpDraftedTokens = null;
      this.perPositionAcceptance = null;
    }

    // ── Latency stats: mean over the poll window from sum/count deltas ──
    // (vLLM parity: histogram sum/count → mean TTFT; p95s stay null because the
    // engine reports no latency distribution — honest absence.)
    const ttftSum = Number(data?.ttft_seconds_sum);
    const ttftCnt = Number(data?.ttft_seconds_count);
    if (
      Number.isFinite(ttftSum) && Number.isFinite(ttftCnt) &&
      this.lastExl3TtftSum != null && this.lastExl3TtftCount != null &&
      ttftCnt > this.lastExl3TtftCount && ttftSum >= this.lastExl3TtftSum
    ) {
      const dT = ttftSum - this.lastExl3TtftSum;
      const dC = ttftCnt - this.lastExl3TtftCount;
      if (dC > 0 && dT >= 0) {
        this.ttftSeconds = Math.round((dT / dC) * 1000) / 1000;
        this.ttftP95Seconds = this.ttftSeconds; // single-request window: mean == observed
      }
    }
    if (Number.isFinite(ttftSum) && Number.isFinite(ttftCnt)) {
      if (ttftSum < (this.lastExl3TtftSum ?? 0) || ttftCnt < (this.lastExl3TtftCount ?? 0)) {
        // server restart wiped counters: don't derive from a negative delta
        this.ttftSeconds = null;
        this.ttftP95Seconds = null;
      }
      this.lastExl3TtftSum = ttftSum;
      this.lastExl3TtftCount = ttftCnt;
      // vLLM parity: rollingAvgTtft mirrors the observed mean (shim is
      // single-stream — each window is one request, so window mean == rolling)
      if (this.ttftSeconds != null) this.rollingAvgTtft = this.ttftSeconds;
    } else {
      this.ttftSeconds = null;
      this.ttftP95Seconds = null;
    }
    const e2eSum = Number(data?.e2e_seconds_sum);
    const e2eCnt = Number(data?.e2e_seconds_count);
    if (
      Number.isFinite(e2eSum) && Number.isFinite(e2eCnt) &&
      this.lastExl3E2eSum != null && this.lastExl3E2eCount != null &&
      e2eCnt > this.lastExl3E2eCount && e2eSum >= this.lastExl3E2eSum
    ) {
      const dE = e2eSum - this.lastExl3E2eSum;
      const dC = e2eCnt - this.lastExl3E2eCount;
      if (dC > 0 && dE >= 0) {
        this.e2eLatency = Math.round((dE / dC) * 1000) / 1000;
        this.e2eP95Seconds = this.e2eLatency;
        this.rollingAvgE2e = this.e2eLatency;
      }
    }
    if (Number.isFinite(e2eSum) && Number.isFinite(e2eCnt)) {
      this.lastExl3E2eSum = e2eSum;
      this.lastExl3E2eCount = e2eCnt;
    } else {
      this.e2eLatency = null;
      this.e2eP95Seconds = null;
    }
    const itlSum = Number(data?.itl_seconds_sum);
    const itlCnt = Number(data?.itl_seconds_count);
    if (
      Number.isFinite(itlSum) && Number.isFinite(itlCnt) &&
      this.lastExl3ItlSum != null && this.lastExl3ItlCount != null &&
      itlCnt > this.lastExl3ItlCount && itlSum >= this.lastExl3ItlSum
    ) {
      const dI = itlSum - this.lastExl3ItlSum;
      const dC = itlCnt - this.lastExl3ItlCount;
      if (dC > 0 && dI >= 0) {
        this.itlP95Seconds = Math.round((dI / dC) * 1000) / 1000;
      }
    }
    if (Number.isFinite(itlSum) && Number.isFinite(itlCnt)) {
      this.lastExl3ItlSum = itlSum;
      this.lastExl3ItlCount = itlCnt;
    } else {
      this.itlP95Seconds = null;
    }

    // ── GPU memory utilization (vLLM/DS4 parity: fraction of device memory
    //    in use — shim reports torch.cuda.mem_get_info census). ──
    if (this.gpuMemoryUtilization == null) {
      const gmu = Number(data?.gpu_memory_utilization);
      if (Number.isFinite(gmu) && gmu >= 0 && gmu <= 1) {
        this.gpuMemoryUtilization = gmu;
      }
    }

    // ── genTokensPerReq from completed-request deltas ──
    if (Number.isFinite(completed) && Number.isFinite(completion)) {
      if (
        this.lastExl3Completed != null &&
        completed > this.lastExl3Completed &&
        completion >= (this.lastExl3CompletedTokens ?? 0)
      ) {
        const dReq = completed - this.lastExl3Completed;
        const dTok = completion - (this.lastExl3CompletedTokens ?? 0);
        if (dReq > 0 && dTok >= 0) {
          this.genTokensPerReq = Math.round((dTok / dReq) * 100) / 100;
          this.rollingAvgTokensPerReq = this.genTokensPerReq;
        }
      }
      this.lastExl3Completed = completed;
      this.lastExl3CompletedTokens = completion;
    }

    if (!Number.isFinite(completion)) {
      this.generationTps = busy ? this.generationTps : 0;
      if (!busy) this.prefillTps = 0;
      return;
    }

    if (dtSec > 0 && dtSec < 10) {
      const deltaOut = completion - this.lastTokenCounts.output;
      // Prefer the engine's DURING-prefill counter (climbs per prefill chunk) so
      // the timeline shows a rising line through prefill; prompt_tokens_total is
      // a single enqueue-time lump (one spike then flat) and stays the fallback.
      // Live prefill rate from the engine (tokens / elapsed during the active
      // prefill) — charts a real line during prefill. Counter diffs stay the
      // fallback for engines without it.
      const pfLive = Number(data?.prefill_rate);
      const pfProcessed = Number(data?.prefill_tokens_processed_total);
      const prefillBase = Number.isFinite(pfLive) && data?.prefill_rate != null
        ? pfProcessed
        : prompt;
      const deltaIn = Number.isFinite(this.lastTokenCounts.prefillIn ?? NaN)
        ? prefillBase - this.lastTokenCounts.prefillIn
        : NaN;
      // Red-team M1: after a probe-failure reset (or first contact), the
      // counters are lifetime values but the poll window is ~2s — seeding the
      // baseline only avoids a giant fake spike into peak/perStream.
      const seedOnly =
        this._exl3Seeded !== true ||
        deltaOut < 0 ||
        (Number.isFinite(deltaIn) && deltaIn < 0);
      if (seedOnly) {
        this._exl3Seeded = true;
      } else {
        this.generationTps = Math.max(0, Math.round((deltaOut / dtSec) * 100) / 100);
        if (Number.isFinite(pfLive) && data?.prefill_rate != null) {
          // Engine-reported live prefill rate — use as-is while prefilling.
          this.prefillTps = data.prefill_rate > 0 ? data.prefill_rate : (busy ? this.prefillTps : 0);
        } else if (Number.isFinite(deltaIn)) {
          this.prefillTps = Math.max(0, Math.round((deltaIn / dtSec) * 100) / 100);
        } else if (!busy && this.generationTps <= 0) {
          this.prefillTps = 0;
        }
      }
      if (Number.isFinite(deltaIn)) this.lastTokenCounts.prefillIn = prefillBase;
    } else if (!busy) {
      this.generationTps = 0;
      this.prefillTps = 0;
    }

    if (Number.isFinite(prompt)) this.lastTokenCounts.input = prompt;
    this.lastTokenCounts.output = completion;
    this.totalOutputTokens = completion;
    this.totalTokensDecoded = completion;

    // Aggregate decode TPS alias (vLLM parity)
    this.aggregateDecodeTps = this.generationTps;

    // Peak + per-stream tracking for the panel dials (parity with DS4/llama.cpp
    // paths). exl3 = single stream: per-stream == aggregate.
    const currentAggregate = this.generationTps;
    if (currentAggregate > (this.peakAggregateTps ?? 0)) {
      this.peakAggregateTps = currentAggregate;
    }
    if (currentAggregate > 0) {
      if (this.perStreamHigh == null || currentAggregate > this.perStreamHigh) {
        this.perStreamHigh = currentAggregate;
      }
      if (this.perStreamLow == null || currentAggregate < this.perStreamLow) {
        this.perStreamLow = currentAggregate;
      }
      const prevN = this._exl3StreamSamples ?? 0;
      this._exl3StreamSamples = prevN + 1;
      this.perStreamAvg = ((this.perStreamAvg ?? 0) * prevN + currentAggregate) / (prevN + 1);
      this.rollingAvgTpsPerSlot = Math.round(this.perStreamAvg * 100) / 100;
    }
    if (Number.isFinite(prompt)) this.totalPromptTokens = prompt;
  }

  /** Drop live FreeToken rates after a failed or empty `/v1/stats`. Totals stay. */
  _clearFreeTokenRates() {
    this.generationTps = 0;
    this.prefillTps = 0;
    this.kvCacheUsage = null;
    this.kvCacheTokens = null;
    this.kvCacheTokensAvailable = null;
    this.kvCacheMemoryBytes = null;
    this.ttftSeconds = null;
    this.ttftP95Seconds = null;
    this.e2eP95Seconds = null;
    this.requestsWaiting = null;
    this.preemptionsTotal = null;
  }

  /**
   * Apply FreeToken GET /v1/stats.
   * `throughput.*_tps` is already a 5s sliding window (0 when idle), so it is
   * not differenced. `requests.ttft_mean_ms` is a mean. `requests.p95_ms` is
   * end-to-end request latency, not TTFT p95. Queue length, slot capacity, and
   * preemptions are not in the payload and stay unset.
   * @param {unknown} data
   */
  _applyFreeTokenStats(data) {
    if (!LlmProbe._statsLookLikeFreeToken(data)) {
      this._clearFreeTokenRates();
      return;
    }
    const stats = /** @type {Record<string, any>} */ (data);
    // Rates come from FreeToken's window, not from differencing these totals,
    // so a restart (new instance_id, or counters that went backwards) cannot
    // spike tok/s. The new totals are adopted as-is.

    const decode = Number(stats.throughput.decode_tps);
    const prefill = Number(stats.throughput.prefill_tps);
    this.generationTps = Math.max(0, Math.round(decode * 100) / 100);
    this.prefillTps = Math.max(0, Math.round(prefill * 100) / 100);

    const requests = stats.requests;
    const active = Number(requests.active);
    this.requestsRunning = Number.isFinite(active) && active >= 0 ? Math.round(active) : null;
    // No concurrency capacity in /v1/stats. Do not invent a slot total from `active`.
    this.slotsActive = 0;
    this.slotsTotal = 0;
    this.requestsWaiting = null;
    this.preemptionsTotal = null;
    this.ttftP95Seconds = null;

    const prompt = Number(requests.prompt_tokens_total);
    const completion = Number(requests.completion_tokens_total);
    if (Number.isFinite(prompt) && prompt >= 0) this.totalPromptTokens = prompt;
    if (Number.isFinite(completion) && completion >= 0) this.totalOutputTokens = completion;

    const ttftMeanMs = Number(requests.ttft_mean_ms);
    this.ttftSeconds =
      Number.isFinite(ttftMeanMs) && ttftMeanMs >= 0
        ? Math.round((ttftMeanMs / 1000) * 1000) / 1000
        : null;
    const p95Ms = Number(requests.p95_ms);
    this.e2eP95Seconds =
      Number.isFinite(p95Ms) && p95Ms >= 0 ? Math.round((p95Ms / 1000) * 1000) / 1000 : null;

    const kv = stats.kv;
    if (
      kv &&
      typeof kv === "object" &&
      Number(kv.total_pages) > 0 &&
      Number.isFinite(Number(kv.used_pages))
    ) {
      const total = Number(kv.total_pages);
      const used = Math.max(0, Number(kv.used_pages));
      this.kvCacheUsage = Math.round((Math.min(used, total) / total) * 1000) / 1000;
      const page = Number(kv.page_size);
      if (Number.isFinite(page) && page > 0) {
        this.kvCacheTokens = Math.round(total * page);
        this.kvCacheTokensAvailable = Math.max(0, Math.round((total - used) * page));
      } else {
        this.kvCacheTokens = null;
        this.kvCacheTokensAvailable = null;
      }
    } else {
      this.kvCacheUsage = null;
      this.kvCacheTokens = null;
      this.kvCacheTokensAvailable = null;
    }

    const ctx = Number(stats.model?.ctx);
    if (Number.isFinite(ctx) && ctx > 0) this.contextLength = Math.round(ctx);
    if (stats.model?.id) applyModelRef(this, stats.model.id);
  }

  /**
   * Apply TensorFold GET /health.
   * Live decode tok/s is the completion-counter diff (idle → 0). The first sample
   * only seeds the baseline. Prefill tok/s uses `prefill_seconds_total` when the
   * server sends it (prompt tokens ÷ actual prefill time), otherwise the EXL3
   * poll-window diff. `{ok: true}` with no counters stays at 0.
   * CUDA 0.6.0 sizes the slot tile from `streams.max`; MLX uses `max_batch_size`.
   * @param {Record<string, unknown> | null} data
   * @param {number} dtSec
   */
  _applyTensorFoldHealth(data, dtSec) {
    const health = data && typeof data === "object" && !Array.isArray(data) ? data : {};
    // A prompt is being processed right now (no live rate exists for it, see below).
    this.prefillActive = Number(health.streams?.prefilling) > 0;
    // Number(null) is 0: a null counter means "absent", not "zero".
    const prompt = health.prompt_tokens_total == null ? NaN : Number(health.prompt_tokens_total);
    const completion = health.completion_tokens_total == null ? NaN : Number(health.completion_tokens_total);
    const prefillSec = Number(health.prefill_seconds_total);
    if (Number.isFinite(completion) && !this._tensorfoldSeeded) {
      if (Number.isFinite(prompt)) this.lastTokenCounts.input = prompt;
      this.lastTokenCounts.output = completion;
      if (Number.isFinite(prefillSec)) this._tensorfoldPrefillSeconds = prefillSec;
      this._tensorfoldSeeded = true;
      this.generationTps = 0;
      this.prefillTps = 0;
    }
    const prevIn = this.lastTokenCounts.input;
    const prevPrefillSec = this._tensorfoldPrefillSeconds;
    this._applyExl3Health(health, dtSec);
    // TensorFold 0.5.0 (cuda/health.py) publishes cumulative cached prompt tokens
    // alongside the EXL3-style counters; older builds omit it (stays null).
    const cached = Number(health.cached_tokens_total);
    if (Number.isFinite(cached) && cached >= 0) this.totalCachedTokens = cached;
    const batch = Number(health.max_batch_size);
    if (Number.isFinite(batch) && batch > 0) this.slotsTotal = Math.round(batch);
    // TensorFold 0.6.0 CUDA serves several streams at once: `streams.max` is the
    // slot count and `requests_running` the busy ones. The EXL3 path above only
    // knows a single `busy` flag, which would pin the tile at 1/1.
    const streams =
      health.streams && typeof health.streams === "object" && !Array.isArray(health.streams)
        ? health.streams
        : {};
    const max = Number(streams.max);
    if (Number.isFinite(max) && max > 0) this.slotsTotal = Math.round(max);
    // null must not read as 0 busy streams (Number(null) === 0).
    const count = (v) => (v == null ? NaN : Number(v));
    let running = count(health.requests_running);
    if (!Number.isFinite(running)) {
      const decoding = count(streams.decoding);
      const prefilling = count(streams.prefilling);
      if (Number.isFinite(decoding) || Number.isFinite(prefilling)) {
        running =
          (Number.isFinite(decoding) ? decoding : 0) +
          (Number.isFinite(prefilling) ? prefilling : 0);
      }
    }
    if (Number.isFinite(running) && running >= 0) {
      this.requestsRunning = Math.round(running);
      this.slotsActive = Math.round(running);
    }
    // 0.6.0 CUDA sizes its KV pool in tokens: `pool_tokens` in all,
    // `pool_free_tokens` not held by a request (kept prompts count as held).
    const pool = count(health.pool_tokens);
    const poolFree = count(health.pool_free_tokens);
    if (Number.isFinite(pool) && pool > 0 && Number.isFinite(poolFree) && poolFree >= 0) {
      const used = 1 - Math.min(poolFree, pool) / pool;
      this.kvCacheUsage = Math.round(used * 10000) / 10000;
    }
    if (!Number.isFinite(prefillSec) || !Number.isFinite(prompt)) return;
    if (prevPrefillSec == null) {
      this._tensorfoldPrefillSeconds = prefillSec;
      return;
    }
    if (!(dtSec > 0 && dtSec < 10)) return;
    const dSec = prefillSec - prevPrefillSec;
    const dIn = prompt - prevIn;
    if (dSec > 0 && dIn > 0) {
      // A request just finished: its average prefill rate. TensorFold publishes token and time
      // totals only for finished requests, so nothing is measurable while a long prefill runs;
      // keep the last finished rate on screen for a while instead of flashing to 0.
      this._tensorfoldLastPrefill = { tps: Math.max(0, Math.round((dIn / dSec) * 100) / 100), at: Date.now() };
      this.prefillTps = this._tensorfoldLastPrefill.tps;
    } else if (this._tensorfoldLastPrefill && Date.now() - this._tensorfoldLastPrefill.at < 20_000) {
      this.prefillTps = this._tensorfoldLastPrefill.tps;
    } else {
      this.prefillTps = 0;
    }
    this._tensorfoldPrefillSeconds = prefillSec;
  }

  /**
   * Newer TensorFold builds put the server's own live rates in /health:
   * `live: { decode_tokens_per_second, prefill_tokens_per_second, connections, waiting }`.
   * Those are used as-is (no counter diff needed). Older builds have no `live` block.
   * @param {Record<string, unknown> | null} data
   */
  _applyTensorFoldLive(data) {
    const live = data && typeof data === "object" ? data.live : null;
    if (!live || typeof live !== "object" || Array.isArray(live)) return;
    const rate = (v) => {
      const n = v == null ? NaN : Number(v);
      return Number.isFinite(n) && n >= 0 ? Math.round(n * 100) / 100 : null;
    };
    const decode = rate(live.decode_tokens_per_second);
    const prefill = rate(live.prefill_tokens_per_second);
    if (decode != null) this.generationTps = decode;
    if (prefill != null) this.prefillTps = prefill;
    const waiting = live.waiting == null ? NaN : Number(live.waiting);
    if (Number.isFinite(waiting) && waiting >= 0) this.requestsWaiting = Math.round(waiting);
    const conns = live.connections == null ? NaN : Number(live.connections);
    if (Number.isFinite(conns) && conns >= 0 && !(this.requestsRunning > 0)) {
      this.requestsRunning = Math.round(conns);
      this.slotsActive = Math.round(conns);
    }
  }

  /**
   * TensorFold Prometheus /metrics (`tensorfold:*`): cumulative token totals (finished requests),
   * running / waiting requests and the MTP draft acceptance rate. Best effort: a miss changes nothing.
   */
  async _applyTensorFoldMetrics() {
    let txt;
    try {
      const res = await this._fetch(`${this.baseUrl}/metrics`);
      if (!res.ok) return;
      txt = await res.text();
    } catch {
      return;
    }
    if (typeof txt !== "string" || !txt.includes("tensorfold:")) return;
    const m = (name) => this._getPromMetric(txt, `tensorfold:${name}`);
    const gen = m("generation_tokens_total");
    const prompt = m("prompt_tokens_total");
    if (Number.isFinite(gen) && gen >= 0) this.totalOutputTokens = gen;
    if (Number.isFinite(prompt) && prompt >= 0) this.totalPromptTokens = prompt;
    const running = m("requests_running");
    if (Number.isFinite(running) && running >= 0) {
      this.requestsRunning = Math.round(running);
      this.slotsActive = Math.round(running);
    }
    const waiting = m("requests_waiting");
    if (Number.isFinite(waiting) && waiting >= 0) this.requestsWaiting = Math.round(waiting);
    const drafted = m("mtp_drafted_total");
    const accepted = m("mtp_accepted_total");
    if (Number.isFinite(drafted) && drafted > 0 && Number.isFinite(accepted)) {
      this.mtpAcceptanceRate = Math.max(0, Math.min(1, accepted / drafted));
    }
  }

  /**
   * Apply stock vLLM Prometheus /metrics (tok/s + inference tiles).
   * @param {string} txt
   * @param {number} dtSec
   */
  _applyVllmMetrics(txt, dtSec) {
    // Model-proof series lookup: name sets drift across vLLM builds (0.31-dev
    // dsv41 vs 0.32 nightly expose different token-counter names, and future
    // builds may rename again). Resolve each counter by first match, with a
    // fuzzy suffix scan fallback over every counter series present. Nothing
    // is hardcoded to one build's vocabulary, and a fully absent series
    // yields null (→ honest UNAVAILABLE, never a dead 0).
    const promptTokens = this._getVllmSeries(txt, [
      "prompt_tokens_total",
      "prompt_tokens_processed_total",
      ["~", "prompt_tokens_total"],
    ]);
    const genTokens = this._getVllmSeries(txt, [
      "generation_tokens_total",
      "completion_tokens_total",
      ["~", "generation_tokens_total"],
      ["~", "completion_tokens_total"],
    ]);
    const running = this._getVllmSeries(txt, [
      "num_requests_running",
      "num_running_requests",
    ]);
    const iterSum = this._getVllmSeries(txt, [
      "iteration_tokens_total_sum",
      ["~", "iteration_tokens_total_sum"],
    ]);
    // Honest absence markers for the panel: which gauge families this engine
    // build does not expose at all. Never marks a present series unavailable.
    this._refreshVllmSeriesUnavailable(txt);
    if (promptTokens != null && genTokens != null) {
      // Engine restart/rebounce guard: vLLM counters are cumulative-since-process-start. When the
      // engine restarts (warmup/profile runs re-accumulate 60K+ tokens while the dashboard baseline
      // is 0 or stale), a naive diff shows bursts like 62K tok/s. Detect a counter regression
      // (genTokens went DOWN = fresh process) and re-seed the baseline without emitting a spike.
      const engineRestarted = genTokens < this.lastTokenCounts.output;
      if (engineRestarted) {
        this.lastTokenCounts.input = promptTokens;
        this.lastTokenCounts.output = genTokens;
        this.lastIterSum = iterSum != null ? iterSum : this.lastIterSum;
        this.prefillTps = 0;
        this.generationTps = 0;
        this.totalOutputTokens = genTokens;
        this.totalPromptTokens = promptTokens;
        return;
      }
      // Warmup guard: vLLM warmup/profile runs (e.g. 30x repeated max_num_batched_tokens prompts)
      // execute with num_requests_running = 0. Real prefill only happens with requests live.
      const requestsLive = (running != null && running > 0)
        || (this._vllmPrevRunning != null && this._vllmPrevRunning > 0);
      this._vllmPrevRunning = running;
      // Counter-reset guards, per counter: a regression (engine restart that
      // only touches this counter, or shard-level resets) is clamped so a
      // negative delta can never feed a rate (the old 62K "prefill" lie).
      const rawDeltaIn = promptTokens - this.lastTokenCounts.input;
      const deltaIn = rawDeltaIn >= 0 ? rawDeltaIn : 0;
      const deltaOut = genTokens - this.lastTokenCounts.output;
      this.lastTokenCounts.input = promptTokens;
      this.lastTokenCounts.output = genTokens;
      this.totalOutputTokens = genTokens;
      this.totalPromptTokens = promptTokens;
      // vLLM prefix-cache counters are token-granular (queries += prompt
      // tokens/request, hits += tokens served from cache), so cached_tokens
      // is directly the cumulative cached share of the prompt.
      const cachedTok = this._getVllmMetric(txt, "prompt_tokens_cached_total") ??
        this._getVllmMetric(txt, "prefix_cache_hits_total");
      if (cachedTok != null) this.totalCachedTokens = cachedTok;
      // The cached-share sibling is published in the SAME completion burst as
      // prompt_tokens_total (verified on dsv41 0.31-dev: lifetime deltas match
      // prefix_cache queries/hits exactly), so the COMPUTED share of a window
      // is prompt delta minus cached delta. Cached tokens are not prefill work.
      const deltaCached =
        cachedTok != null && this.lastTokenCounts.cached != null
          ? Math.max(0, cachedTok - this.lastTokenCounts.cached)
          : 0;
      if (cachedTok != null) this.lastTokenCounts.cached = cachedTok;
      // Honest prefill-work input for this window: prompt tokens that were
      // actually computed (cache-served tokens cost no prefill).
      const computedIn = Math.max(0, deltaIn - deltaCached);
      const ttftSum = this._getVllmMetric(txt, "time_to_first_token_seconds_sum");
      const deltaIter =
        iterSum != null && this.lastIterSum != null
          ? Math.max(0, iterSum - this.lastIterSum)
          : 0;
      if (dtSec > 0 && dtSec < 10) {
        // Engine-step tokens include prefill+decode, so across any window
        // decode can never exceed iteration tokens. When the iteration
        // histogram is present it is the cheaper, more trustworthy bound;
        // when it is missing (e.g. mid-boot windows where the counter registries
        // transiently re-register) fall back to a generous plausibility
        // ceiling. Either way an implausible window emits NO rate — the
        // counter-jump class of bug (the old 62K tok/s "prefill") must never
        // be rendered, not even once.
        const ITER_BOUND_MULTIPLIER = 1.1;
        const ITER_BOUND_SLACK = 4096;
        const PLAUSIBLE_TPS_CEILING = 50000;
        const iterBoundOk =
          iterSum == null ||
          (deltaOut <= deltaIter * ITER_BOUND_MULTIPLIER + ITER_BOUND_SLACK &&
            computedIn <= deltaIter * ITER_BOUND_MULTIPLIER + ITER_BOUND_SLACK);
        const ceilingOk =
          deltaOut / (dtSec || 1) <= PLAUSIBLE_TPS_CEILING &&
          computedIn / (dtSec || 1) <= PLAUSIBLE_TPS_CEILING;
        if (iterBoundOk && ceilingOk) {
        this.generationTps = Math.max(0, Math.round((deltaOut / dtSec) * 100) / 100);
        const deltaTtft =
          ttftSum != null && this.lastTtftSum != null
            ? Math.max(0, ttftSum - this.lastTtftSum)
            : 0;
        // Engine-step tokens include prefill+decode. Surplus over generation is
        // prefill, including the common case where a short/cached prefill lands
        // in the same poll as the first decode tokens.
        const prefillIter = Math.max(0, deltaIter - Math.max(0, deltaOut));
        const specNoise = deltaOut > 0 && prefillIter > 0 && prefillIter < deltaOut * 0.5;
        const livePrefill =
          requestsLive && prefillIter > 0 && !specNoise ? prefillIter / dtSec : 0;
        const finishedPrefill =
          requestsLive && computedIn > 0 && deltaTtft > 0
            ? computedIn / deltaTtft
            : requestsLive && computedIn > 0 && livePrefill <= 0
              ? computedIn / dtSec
              : 0;
        // A prefill burst lands in exactly one poll window (~1-2s wide); the
        // very next poll (decode-only) previously zeroed the gauge, making the
        // prefill reading effectively invisible at the panel's cadence. Hold
        // the last prefill reading ≤5s while the same request is still live,
        // then decay to 0 — never latches (idle → 0, and requestsLive is
        // required for the hold).
        const ttftCountNow = this._getVllmMetric(txt, "time_to_first_token_seconds_count");
        const deltaTtftCount =
          ttftCountNow != null && this.lastTtftCount != null
            ? Math.max(0, ttftCountNow - this.lastTtftCount)
            : 0;
        const prefillLive =
          finishedPrefill > 0 && deltaTtftCount > 0
            ? finishedPrefill
            : livePrefill > 0
              ? livePrefill
              : finishedPrefill;
        const prefillRecentMs = Date.now() - (this._lastPrefillAt ?? 0);
        if (prefillLive > 0) {
          this.prefillTps = Math.max(0, Math.round(prefillLive * 100) / 100);
          this._lastPrefillAt = Date.now();
        } else if (!(this.prefillTps > 0 && requestsLive && prefillRecentMs < 5000)) {
          this.prefillTps = 0;
        }
        } // end plausibility gate (implausible windows render no rate)
      }
      // SESSION/ROLLING AVERAGES (James 2026-10-16): instantaneous gauges idle-gate to 0
      // and read empty; averages over cumulative-counter deltas are immune to idle polls
      // (zero token delta), reseeds (max(0,·) drops the broken delta) and re-registration.
      // avgTps     = session-lifetime GENERATED tokens / observed poll seconds.
      // avgTps1h   = rolling 1h over the cumulative gen-token deque (min 30s window).
      // prefillAvg = session-lifetime prefill (iteration surplus over decode) tokens /
      //              observed poll seconds, when the iteration counter is available.
      if (dtSec > 0 && dtSec < 60) {
        const dOut = Math.max(0, deltaOut);
        this._avgTokAccum = (this._avgTokAccum || 0) + dOut;
        this._avgTimeAccum = (this._avgTimeAccum || 0) + dtSec;
        this._avgCum = (this._avgCum || 0) + dOut;
        this._avgPts = this._avgPts || [];
        this._avgPts.push([Date.now(), this._avgCum]);
        while (this._avgPts.length > 2 && Date.now() - this._avgPts[0][0] > 3600000) this._avgPts.shift();
        this.avgTps = this._avgTimeAccum > 0
          ? Math.round((this._avgTokAccum / this._avgTimeAccum) * 100) / 100 : 0;
        if (this._avgPts.length > 1) {
          const a = this._avgPts[0], b = this._avgPts[this._avgPts.length - 1];
          const wdt = (b[0] - a[0]) / 1000;
          this.avgTps1h = wdt >= 30 ? Math.round(((b[1] - a[1]) / wdt) * 100) / 100 : null;
        } else this.avgTps1h = null;
        if (iterSum != null) {
          const prefillTok = Math.max(0, deltaIter - dOut);
          this._prefillTokAccum = (this._prefillTokAccum || 0) + prefillTok;
          this.prefillAvg = this._avgTimeAccum > 0
            ? Math.round((this._prefillTokAccum / this._avgTimeAccum) * 100) / 100 : 0;
        }
        this.avgTpsBuildVersion = "avgtps-2026-10-16-1";
      } else { this._avgPts = this._avgPts || []; }
      // Live mean TTFT over the last poll window from histogram sum/count deltas.
      // Computed BEFORE lastTtftSum is advanced so the delta is real, not 0.
      const ttftCount = this._getVllmMetric(txt, "time_to_first_token_seconds_count");
      if (ttftCount != null) {
        const deltaSum =
          ttftSum != null && this.lastTtftSum != null
            ? Math.max(0, ttftSum - this.lastTtftSum)
            : null;
        const deltaCount =
          this.lastTtftCount != null ? ttftCount - this.lastTtftCount : null;
        this.ttftSeconds =
          deltaSum != null && deltaCount != null && deltaCount > 0 && deltaSum >= 0
            ? Math.round((deltaSum / deltaCount) * 1000) / 1000
            : null;
        this.lastTtftCount = ttftCount;
      } else {
        this.ttftSeconds = null;
      }
      if (ttftSum != null) this.lastTtftSum = ttftSum;
    }
    if (iterSum != null) this.lastIterSum = iterSum;

    this.requestsRunning = running;
    if (running != null) this.slotsActive = Math.round(running);

    if (this.gpuMemoryUtilization == null) {
      const sleepState = this._getVllmMetric(txt, "engine_sleep_state");
      if (sleepState != null) this.gpuMemoryUtilization = sleepState;
    }

    this.requestsWaiting = this._getVllmMetric(txt, "num_requests_waiting");

    // Scheduler slot capacity: some builds export it; otherwise leave 0
    // (honest absence) rather than showing pool size as slots.
    const maxSeqs =
      this._getVllmMetric(txt, "max_num_seqs") ??
      this._getVllmMetric(txt, "scheduler_max_num_seqs");
    if (maxSeqs != null && maxSeqs > 0) this.slotsTotal = Math.round(maxSeqs);
    // Older vLLM (V0 engine) names the same gauge gpu_cache_usage_perc.
    this.kvCacheUsage =
      this._getVllmMetric(txt, "kv_cache_usage_perc") ??
      this._getVllmMetric(txt, "gpu_cache_usage_perc");
    // styles01 fills for tiles the FreeToken /v1/stats path owns but vLLM
    // exposes by other means: n_ctx from /v1/models max_model_len, engine
    // uptime from the host PID table (lane runs on the host network ns),
    // sampler defaults from the served engine generation_config.json.
    if (this.contextLength != null && this.nCtx == null) {
      this.nCtx = Math.round(this.contextLength);
    }
    try {
      const procDirs = readdirSync("/host/proc").filter((p2) => /^[0-9]+$/.test(p2));
      for (const pid of procDirs) {
        try {
          const cl = readFileSync(`/host/proc/${pid}/cmdline`, "utf8").replace(/\u0000/g, " ");
          if (cl.includes("/usr/local/bin/vllm serve") && cl.includes("--port 8000")) {
            const st = readFileSync(`/host/proc/${pid}/stat`, "utf8");
            const f = st.substring(st.lastIndexOf(")") + 2).split(" ");
            const up = Number(readFileSync("/host/proc/uptime", "utf8").split(" ")[0]);
            const hz = 100;
            const age = up - Number(f[21]) / hz;
            if (Number.isFinite(age) && age > 0) {
              this.engineUptimeSeconds = Math.round(age);
            }
            break;
          }
        } catch {}
      }
    } catch (e) { console.error("[fills] uptime err:", e && e.message); }
    if (this.temperature == null || this.topP == null) {
      try {
          const hub = "/host/root/home/jaita/MODELS/hf/hub";
        const mdirs = readdirSync(hub).filter((d) => d.startsWith("models--"));
        const hubModel = mdirs.find((d) => this.modelId && this.modelId.length > 0 && d.toLowerCase().includes(this.modelId.toLowerCase().replace(/[^a-z0-9]/g, "").slice(0, 12)))
          ?? mdirs.find((d) => d.includes("GLM-5.3-Flash"));
        for (const d of hubModel ? [hubModel] : mdirs) {
          const snaps = `${hub}/${d}/snapshots`;
          const snap = readdirSync(snaps)[0];
          let raw;
          try {
            raw = readFileSync(`${snaps}/${snap}/generation_config.json`, "utf8");
          } catch {
            continue;
          }
          const gen = JSON.parse(raw);
          const t = Number(gen.temperature);
          const tp = Number(gen.top_p);
          const k = Number(gen.top_k);
          if (Number.isFinite(t)) this.temperature = t;
          if (Number.isFinite(tp)) this.topP = tp;
          if (Number.isFinite(k) && k > 0) this.topK = k;
          this.samplingParams = {
            temperature: Number.isFinite(t) ? t : null,
            top_p: Number.isFinite(tp) ? tp : null,
            top_k: Number.isFinite(k) && k > 0 ? k : null,
          };
          this.samplers = [
            Number.isFinite(t) ? `temp ${t}` : null,
            Number.isFinite(tp) ? `top_p ${tp}` : null,
            Number.isFinite(k) && k > 0 ? `top_k ${k}` : null,
          ].filter(Boolean);
          break;
        }
      } catch (e2) { console.error("[fills] gen err:", e2 && e2.message); }
    }
    if (this.totalCachedTokens != null) this.promptCachedTokens = Math.round(this.totalCachedTokens);

    this._applyVllmKvPool(txt);
    this._applyVllmSpec(txt);
    if (this.kvUsedTokens != null && this.kvPoolTokens != null) {
      this.kvCacheTokensAvailable = Math.max(0, this.kvPoolTokens - this.kvUsedTokens);
      this.remainingCtx = Math.max(0, this.kvPoolTokens - this.kvUsedTokens);
      this.nRemain = this.remainingCtx;
      this.kvUsedTokens = Math.max(this.kvUsedTokens, 0);
    }

    // -- Real context usage: KV-cache pool occupancy in tokens --
    // usage gauge x pool capacity = tokens resident in the KV cache right
    // now. prompt/generation totals are LIFETIME counters (1.5M+ since
    // engine start > whole pool) and must never feed a context card.
    const kvPoolUsed = this._getPromMetric(txt, "vllm:kv_cache_used_perc");
    const realUsage = this.kvCacheUsage != null && Number.isFinite(this.kvCacheUsage) && this.kvCacheUsage > 0
      ? this.kvCacheUsage
      : kvPoolUsed != null
        ? kvPoolUsed
        : this._getVllmMetric(txt, "cpu_cache_usage_perc");
    if (realUsage != null) {
      const used = realUsage * (this.kvCacheTokens ?? 0);
      this.kvUsedTokens = used > 0 ? Math.round(used) : 0;
      if (this.kvCacheTokens != null) this.kvPoolTokens = Math.round(this.kvCacheTokens);
    } else {
      this.kvUsedTokens = 0;
      if (this.kvCacheTokens != null) this.kvPoolTokens = Math.round(this.kvCacheTokens);
    }
    // Engine uptime: vLLM exports process_start_time_seconds, but on recent
    // builds it is a plain python_exporter gauge (no `vllm:` prefix) — try
    // both spellings before falling back to /proc.
    const procStart =
      this._getVllmMetric(txt, "process_start_time_seconds") ??
      this._getPromMetric(txt, "process_start_time_seconds");
    if (procStart != null && procStart > 0) {
      const up = Math.max(0, Math.round(Date.now() / 1000 - procStart));
      // Engine restart → the counter jumps backwards; adopt the new (smaller) value.
      if (this.engineUptimeSeconds == null || up < this.engineUptimeSeconds) {
        this.engineUptimeSeconds = up;
      }
    }
    this.preemptionsTotal = this._getVllmMetric(txt, "num_preemptions_total");

    const ttftHist = this._parseVllmHistogram(txt, "vllm:time_to_first_token_seconds");
    const ttftP95 = this._histogramQuantile(ttftHist.buckets, ttftHist.total, 0.95);
    this.ttftP95Seconds = ttftP95 == null ? null : Math.round(ttftP95 * 1000) / 1000;

    const e2eHist = this._parseVllmHistogram(txt, "vllm:e2e_request_latency_seconds");
    const e2eP95 = this._histogramQuantile(e2eHist.buckets, e2eHist.total, 0.95);
    this.e2eP95Seconds = e2eP95 == null ? null : Math.round(e2eP95 * 1000) / 1000;

    const itlHist = this._parseVllmHistogram(txt, "vllm:inter_token_latency_seconds");
    const itlP95 = this._histogramQuantile(itlHist.buckets, itlHist.total, 0.95);
    this.itlP95Seconds = itlP95 == null ? null : Math.round(itlP95 * 1000) / 1000;

    const prefixHits = this._getVllmMetric(txt, "prefix_cache_hits_total");
    const prefixQueries = this._getVllmMetric(txt, "prefix_cache_queries_total");
    this.prefixCacheHitRate =
      prefixHits != null && prefixQueries != null && prefixQueries > 0
        ? Math.round((prefixHits / prefixQueries) * 10000) / 10000
        : null;

    const mtpAccepted = this._getVllmMetric(txt, "spec_decode_num_accepted_tokens_total");
    const mtpDrafted = this._getVllmMetric(txt, "spec_decode_num_draft_tokens_total");
    if (mtpAccepted != null) this.specAcceptedTokens = mtpAccepted;
    if (mtpDrafted != null) this.specGeneratedTokens = mtpDrafted;
    this.mtpAcceptanceRate =
      mtpAccepted != null && mtpDrafted != null && mtpDrafted > 0
        ? Math.round((mtpAccepted / mtpDrafted) * 10000) / 10000
        : null;

    // Verify-step ledger (DFlash2/EXL3): drafts_total counts verify calls.
    const numDrafts = this._getVllmMetric(txt, "spec_decode_num_drafts_total");
    if (numDrafts != null) this.specDrafts = Math.round(numDrafts);
    if (mtpAccepted != null) this.specHits = Math.round(mtpAccepted);
    if (numDrafts != null && this.lastSpecDrafts != null && this.lastSpecOut != null && genTokens != null && numDrafts > this.lastSpecDrafts && genTokens > this.lastSpecOut) {
      const perStep = (genTokens - this.lastSpecOut) / (numDrafts - this.lastSpecDrafts);
      if (Number.isFinite(perStep) && perStep > 0) {
        this.tokPerStep = Math.round(perStep * 100) / 100;
      }
    }
    if (numDrafts != null) this.lastSpecDrafts = Math.round(numDrafts);
    if (genTokens != null) this.lastSpecOut = genTokens;
    // Peak + per-stream tracking for the vLLM flow (was only wired in the
    // FreeToken /v1/stats path, so peakAggregateTps stayed null forever).
    if (this.generationTps > (this.peakAggregateTps ?? 0)) {
      this.peakAggregateTps = this.generationTps;
    }
    if (this.generationTps > 0) {
      if (this.perStreamHigh == null || this.generationTps > this.perStreamHigh) {
        this.perStreamHigh = this.generationTps;
      }
      if (this.perStreamLow == null || this.generationTps < this.perStreamLow) {
        this.perStreamLow = this.generationTps;
      }
    }
    // Decoded-tokens lifetime alias so the panel's totalTokensDecoded tile lives.
    if (genTokens != null) this.totalTokensDecoded = Math.round(genTokens);

    // Reasoning effort — vLLM has no per-request effort gauge; read the served
    // model's chat-template default reasoning_effort (e.g. "medium" for Qwen3.8).
    // Fall back to null (hide the card) rather than guessing from the
    // --reasoning-parser flag, which does NOT equal reasoning effort.
    if (this.reasoningEffort == null) {
      this.reasoningEffort = this._vllmChatTemplateReasoningEffort();
      if (this.reasoningEffort != null) this.reasoningEffortTs = Date.now();
    }
  }

  /**
   * Apply SGLang /server_info (or deprecated /get_server_info).
   * Older builds expose total_input_tokens / total_output_tokens.
   * Current builds (metrics often off) expose sticky last_gen_throughput under
   * internal_states[i]. Only treat it as live after the value changes between
   * polls, then expire to 0 when it stops moving (idle leftover).
   * @param {Record<string, unknown>} sgData
   * @param {number} dtSec
   */
  _applySglangServerInfo(sgData, dtSec) {
    this._clearKvPool();
    // Where the engine's memory went: model weights and the KV pool it
    // pre-allocated at start-up (GB, from the first scheduler that reports).
    const mem = LlmProbe._sglangMemoryUsage(sgData);
    this.weightsGb = LlmProbe._positiveNumber(mem?.weight);
    this.kvCacheGb = LlmProbe._positiveNumber(mem?.kvcache);

    // Prefer true max context (context_length / max_total_tokens). Do NOT use
    // max_total_num_tokens — that is the KV-cache pool budget across concurrent
    // sequences and is often ~2× the configured context (showed 2.1M for a 1M run).
    const explicitCtx =
      LlmProbe._positiveNumber(sgData.context_length) ??
      LlmProbe._positiveNumber(sgData.max_total_tokens);
    if (explicitCtx != null) {
      this.contextLength = explicitCtx;
    } else if (this.contextLength == null) {
      this.contextLength =
        LlmProbe._positiveNumber(sgData.max_req_input_len) ??
        LlmProbe._positiveNumber(sgData.max_total_num_tokens) ??
        null;
    }

    if (sgData.model_path) {
      applyModelRef(this, sgData.model_path);
    }

    const maxRunning = Number(sgData.max_running_requests);
    if (Number.isFinite(maxRunning) && maxRunning > 0) {
      this.slotsTotal = Math.round(maxRunning);
    }

    const inTok = sgData.total_input_tokens;
    const outTok = sgData.total_output_tokens;
    // Newer SGLang builds expose total_cached_tokens (radix/prefix cache
    // served) next to the totals; older ones fall back to Prometheus.
    const cachedNum = Number(sgData.total_cached_tokens);
    const sgCached = Number.isFinite(cachedNum) && cachedNum >= 0 ? cachedNum : null;
    if (inTok != null && outTok != null) {
      const input = Number(inTok);
      const output = Number(outTok);
      if (Number.isFinite(input) && Number.isFinite(output)) {
        const deltaIn = input - this.lastTokenCounts.input;
        const deltaOut = output - this.lastTokenCounts.output;
        this.lastTokenCounts.input = input;
        this.lastTokenCounts.output = output;
        this.totalOutputTokens = output;
        this.totalPromptTokens = input;
        if (sgCached != null) this.totalCachedTokens = sgCached;
        this._sglangTotalsPolled = true;
        this._sglangTokenSource = "server_info";
        if (dtSec > 0 && dtSec < 10) {
          this.generationTps = Math.max(0, Math.round((deltaOut / dtSec) * 100) / 100);
          this._setPrefillTps(deltaIn / dtSec, deltaOut > 0);
        }
        return;
      }
    }

    // No cumulative counters — sticky last_gen_throughput only while it moves,
    // unless /v1/loads (or /get_load) says requests are in flight.
    const lastGen = LlmProbe._sglangLastGenThroughput(sgData);
    this.generationTps = this._sglangStickyThroughput(lastGen, this._sglangInflight());
  }

  /**
   * `internal_states[i].memory_usage` = `{ weight, kvcache, graph, … }` in GB.
   * @param {Record<string, unknown>} sgData
   * @returns {Record<string, unknown> | null}
   */
  static _sglangMemoryUsage(sgData) {
    const states = sgData?.internal_states;
    if (!Array.isArray(states)) return null;
    for (const st of states) {
      const mem = st && typeof st === "object" ? st.memory_usage : null;
      if (mem && typeof mem === "object" && !Array.isArray(mem)) return mem;
    }
    return null;
  }

  /**
   * KV pool fill (and its size, when /server_info did not give one) from
   * SGLang /metrics. `token_usage` is the share of the pool held by running
   * requests; builds without it get kv_used_tokens / max_total_num_tokens.
   * Max, not sum: tensor-parallel ranks each report the same pool.
   * @param {string} txt
   */
  _applySglangKvMetrics(txt) {
    const series = (name) =>
      this._getPromMetricMax(txt, `sglang:${name}`) ??
      this._getPromMetricMax(txt, `sglang_${name}`);
    let usage = series("token_usage");
    if (usage == null) {
      const used = series("kv_used_tokens");
      const total = series("max_total_num_tokens");
      if (used != null && total != null && total > 0) usage = used / total;
    }
    this.kvCacheUsage =
      usage != null && usage >= 0 ? Math.round(Math.min(1, usage) * 10000) / 10000 : null;
    if (this.kvCacheGb == null) {
      this.kvCacheGb = LlmProbe._positiveNumber(series("kv_cache_memory_usage_gb"));
    }
  }

  /** True when SGLang load probe reported running or waiting requests. */
  _sglangInflight() {
    return (
      this.slotsActive > 0 ||
      (this.requestsRunning != null && this.requestsRunning > 0) ||
      (this.requestsWaiting != null && this.requestsWaiting > 0)
    );
  }

  /**
   * Prefer /v1/loads (num_running_reqs). Fall back to /get_load, where
   * num_reqs is running + waiting.
   */
  async _probeSglangLoad() {
    this._sglangLoadGenTps = null;
    for (const path of ["/v1/loads", "/get_load"]) {
      try {
        const res = await this._fetch(`${this.baseUrl}${path}`);
        if (!res.ok) continue;
        const data = await res.json().catch(() => null);
        if (this._applySglangLoad(data)) return;
      } catch {
        /* try next */
      }
    }
  }

  /**
   * Apply SGLang /v1/loads or /get_load. Returns true when a row was applied.
   * @param {unknown} payload
   * @returns {boolean}
   */
  _applySglangLoad(payload) {
    const rows = LlmProbe._sglangLoadRows(payload);
    if (!rows.length) return false;
    let running = 0;
    let waiting = 0;
    let saw = false;
    let loadGen = null;
    for (const row of rows) {
      const gen = Number(row.gen_throughput);
      if (Number.isFinite(gen) && gen >= 0) {
        loadGen = loadGen == null ? gen : Math.max(loadGen, gen);
      }
      const wait = Number(row.num_waiting_reqs);
      const runDirect = Number(row.num_running_reqs);
      const total = Number(row.num_reqs);
      if (Number.isFinite(wait) && wait >= 0) {
        waiting += wait;
        saw = true;
      }
      if (Number.isFinite(runDirect) && runDirect >= 0) {
        running += runDirect;
        saw = true;
      } else if (Number.isFinite(total) && total >= 0) {
        const waitPart = Number.isFinite(wait) && wait >= 0 ? wait : 0;
        running += Math.max(0, total - waitPart);
        saw = true;
      }
    }
    if (!saw) return false;
    this._sglangLoadGenTps = loadGen;
    this.requestsRunning = running;
    this.requestsWaiting = waiting;
    this.slotsActive = Math.round(running);
    return true;
  }

  /**
   * @param {unknown} payload
   * @returns {Array<Record<string, unknown>>}
   */
  static _sglangLoadRows(payload) {
    if (payload == null) return [];
    if (Array.isArray(payload)) {
      return payload.filter((row) => row && typeof row === "object");
    }
    if (typeof payload !== "object") return [];
    if (Array.isArray(payload.loads)) {
      return payload.loads.filter((row) => row && typeof row === "object");
    }
    if (payload.num_reqs != null || payload.num_running_reqs != null) {
      return [payload];
    }
    return [];
  }

  /**
   * Map SGLang's sticky last_gen_throughput gauge to a live panel rate.
   * Returns 0 until the value changes between polls (avoids showing a stale
   * leftover after idle); stays live for a short window after each change.
   * When `inflight` is true (independent load signal), keep a positive rate
   * even if the gauge is not moving — a busy decode can report a constant value.
   * @param {number | null} raw
   * @param {boolean} [inflight]
   * @returns {number}
   */
  _sglangStickyThroughput(raw, inflight = false) {
    if (raw == null || !Number.isFinite(raw) || raw < 0) {
      this._sglangStickyTps = null;
      return 0;
    }
    const rounded = Math.round(raw * 100) / 100;
    const now = Date.now();
    const prev = this._sglangStickyTps;

    if (!prev) {
      // First sample after reset/start — seed only unless load says we are busy
      this._sglangStickyTps = {
        value: rounded,
        liveUntil: inflight && rounded > 0 ? now + SGLANG_STICKY_TPS_LIVE_MS : 0,
      };
      return inflight && rounded > 0 ? rounded : 0;
    }

    if (rounded !== prev.value) {
      this._sglangStickyTps = {
        value: rounded,
        liveUntil: now + SGLANG_STICKY_TPS_LIVE_MS,
      };
      return rounded;
    }

    if (prev.liveUntil > now) {
      return rounded;
    }
    if (inflight && rounded > 0) {
      return rounded;
    }
    return 0;
  }

  /**
   * @param {unknown} v
   * @returns {number | null}
   */
  static _positiveNumber(v) {
    const n = Number(v);
    return Number.isFinite(n) && n > 0 ? n : null;
  }

  /**
   * Max last_gen_throughput across internal_states (or top-level).
   * @param {Record<string, unknown>} sgData
   * @returns {number | null}
   */
  static _sglangLastGenThroughput(sgData) {
    if (!sgData || typeof sgData !== "object") return null;
    const top = Number(sgData.last_gen_throughput);
    if (Number.isFinite(top) && top >= 0) return top;

    const states = sgData.internal_states;
    if (!Array.isArray(states) || !states.length) return null;
    let best = null;
    for (const st of states) {
      if (!st || typeof st !== "object") continue;
      const v = Number(st.last_gen_throughput);
      if (!Number.isFinite(v) || v < 0) continue;
      if (best == null || v > best) best = v;
    }
    return best;
  }

  /**
   * Live decode tok/s. Current SGLang builds publish gen_throughput while a
   * request runs, and only then add the whole completion to
   * generation_tokens_total. Differencing that counter stays at 0 for the
   * whole decode and spikes once when the request ends.
   * Max, not sum: tensor-parallel ranks repeat the same gauge.
   * @param {string} txt
   * @returns {number | null}
   */
  _sglangGenGauge(txt) {
    const prom =
      this._getPromMetricMax(txt, "sglang:gen_throughput") ??
      this._getPromMetricMax(txt, "sglang_gen_throughput");
    if (prom != null) return prom;
    return this._sglangLoadGenTps;
  }

  /**
   * Cumulative prefill tokens that actually did compute (cache misses).
   * Prefers the per-mode counters; falls back to prompt − cached when only the
   * plain prompt counter exists. Kept separate from prompt_tokens_total, which
   * also counts cache-served tokens (the bulk of a warm session).
   * @param {string} txt
   * @returns {number | null}
   */
  _sglangPrefillComputeTokens(txt) {
    const direct =
      this._getPromMetricLabeled(txt, "sglang:realtime_tokens_total", "mode", "prefill_compute") ??
      this._getPromMetricLabeled(txt, "sglang_realtime_tokens_total", "mode", "prefill_compute") ??
      this._getPromMetricLabeled(txt, "sglang:prefill_effective_tokens_total", "mode", "input") ??
      this._getPromMetricLabeled(txt, "sglang_prefill_effective_tokens_total", "mode", "input");
    if (direct != null) return direct;
    const prompt =
      this._getPromMetric(txt, "sglang:prompt_tokens_total") ??
      this._getPromMetric(txt, "sglang_prompt_tokens_total");
    const cached = this._sglangCachedTokens(txt);
    if (prompt != null && cached != null) return Math.max(0, prompt - cached);
    return null;
  }

  /**
   * Apply SGLang Prometheus /metrics (--enable-metrics).
   * Supports both `sglang:` and `sglang_` prefixes.
   * @param {string} txt
   * @param {number} dtSec
   */
  _applySglangMetrics(txt, dtSec) {
    this._clearKvPool();
    const gen =
      this._getPromMetric(txt, "sglang:generation_tokens_total") ??
      this._getPromMetric(txt, "sglang_generation_tokens_total");
    const prompt =
      this._getPromMetric(txt, "sglang:prompt_tokens_total") ??
      this._getPromMetric(txt, "sglang_prompt_tokens_total");
    const running =
      this._getPromMetric(txt, "sglang:num_running_reqs") ??
      this._getPromMetric(txt, "sglang_num_running_reqs");
    if (running != null) {
      this.requestsRunning = running;
      this.slotsActive = Math.round(running);
    }
    if (gen == null) {
      const gauge = this._sglangGenGauge(txt);
      if (gauge != null) {
        this.generationTps = Math.max(0, Math.round(gauge * 100) / 100);
      }
      return;
    }

    // Prefill accounting: the tile counts every prompt token taken in this
    // window — cache-served + computed — and the split rows below break it
    // down. The per-mode counters are exact; prompt_tokens_total alone cannot
    // tell the two apart (it also folds in host/storage hits), so it is the
    // fallback for builds without the realtime series.
    const prefillCompute = this._sglangPrefillComputeTokens(txt);
    const cached = this._sglangCachedTokens(txt);

    // Difference only against our own baseline; after a server-info hand-off
    // the first sample seeds instead of reporting the gap between two series.
    const ownsBaseline = this._sglangTokenSource !== "server_info";
    const canDiff = ownsBaseline && dtSec > 0 && dtSec < 10;
    if (canDiff) {
      const deltaOut = gen - this.lastTokenCounts.output;
      const gauge = this._sglangGenGauge(txt);
      const busy = (running != null && running > 0) || this._sglangInflight();
      if (gauge != null && busy) {
        this.generationTps = Math.max(0, Math.round(gauge * 100) / 100);
      } else if (gauge != null) {
        // Idle, or the completion just landed in the counter. The gauge already
        // carried the live rate; the counter jump is not another tok/s sample.
        this.generationTps = 0;
      } else {
        this.generationTps = Math.max(0, Math.round((deltaOut / dtSec) * 100) / 100);
      }
      if (prefillCompute == null && prompt != null) {
        // No per-mode counters: the prompt counter is the only total we have.
        const deltaIn = prompt - this.lastTokenCounts.input;
        this.prefillTps = Math.max(0, Math.round((deltaIn / dtSec) * 100) / 100);
      } else if (prefillCompute == null && deltaOut <= 0) {
        this.prefillTps = 0;
      }
    }
    // Seed outside the rate window too: on the first poll after a restart the
    // window is unusable, and a stale 0 baseline would turn the engine's
    // lifetime prompt counter into a rate on the next poll (#99).
    this.lastTokenCounts.output = gen;
    if (prompt != null) this.lastTokenCounts.input = prompt;
    this._sglangTokenSource = "prometheus";
    this.totalOutputTokens = gen;
    this.totalPromptTokens = prompt ?? null;
    // Cached share from the same exposition (device layer preferred).
    const cachedNow = this._sglangCachedTokens(txt);
    if (cachedNow != null) this.totalCachedTokens = cachedNow;

    // The tile reads cache-served + computed; the split rows below show the
    // two parts. Poll-window rates, not latches: a window that prefilled
    // nothing reads 0 even while unrelated requests are still decoding
    // (holding the last value here kept a stale number on screen for hours).
    if (prefillCompute != null && cached != null) {
      this._setPrefillSplitRates(cached, prefillCompute, dtSec);
      if (
        canDiff &&
        this.uncachedPrefillTps != null &&
        this.cachedPrefillTps != null
      ) {
        const total = this.uncachedPrefillTps + this.cachedPrefillTps;
        this.prefillTps = Math.max(0, Math.round(total * 100) / 100);
      }
    }
  }

  /**
   * Cache split only — does not touch generation/prefill lastTokenCounts.
   * Prefers cache_source="device" so HiCache L1/L2/L3 labels are not summed.
   */
  _applySglangPrefillSplit(txt, dtSec) {
    const compute = this._sglangPrefillComputeTokens(txt);
    const cached = this._sglangCachedTokens(txt);
    if (cached != null) this.totalCachedTokens = cached;
    if (compute != null && cached != null) {
      this._setPrefillSplitRates(cached, compute, dtSec);
    }
  }

  _sglangCachedTokens(txt) {
    return (
      this._getPromMetricLabeled(txt, "sglang:realtime_tokens_total", "mode", "prefill_cache") ??
      this._getPromMetricLabeled(txt, "sglang_realtime_tokens_total", "mode", "prefill_cache") ??
      this._getPromMetricLabeled(txt, "sglang:cached_tokens_total", "cache_source", "device") ??
      this._getPromMetricLabeled(txt, "sglang_cached_tokens_total", "cache_source", "device") ??
      this._getPromMetricMax(txt, "sglang:cached_tokens_total") ??
      this._getPromMetricMax(txt, "sglang_cached_tokens_total")
    );
  }

  /** Prefer SGLang /model_info (or deprecated /get_model_info) over raw HF cache paths. */
  async _enrichSglangModelInfo() {
    for (const path of SGLANG_MODEL_INFO_PATHS) {
      try {
        const res = await this._fetch(`${this.baseUrl}${path}`);
        if (!res.ok) continue;
        const data = await res.json();
        const raw = data?.model_path || data?.tokenizer_path;
        if (!raw) continue;
        applyModelRef(this, raw);
        return;
      } catch {
        /* try next */
      }
    }
  }

  // ─── llama.cpp native path ────────────────────────────────
  async _probeDs4() {
    const now = Date.now();
    const dtSec = (now - this.lastProbeTime) / 1000;
    this.lastProbeTime = now;

    // Model info from /v1/models
    let modelsOk = false;
    try {
      const modelsRes = await this._fetch(`${this.baseUrl}/v1/models`);
      if (modelsRes.ok) {
        modelsOk = true;
        const modelsData = await modelsRes.json();
        const model = modelsData?.data?.[0];
        this.modelId = model?.id || null;
        this.contextLength = model?.context_length || null;
        this.recipeMetadata = {
          name: model?.id || null,
          model: model?.name || null,
          contextLength: model?.context_length || null,
          ownedBy: model?.owned_by || null,
          supportedParameters: model?.supported_parameters || [],
        };
      }
    } catch {}

    if (!modelsOk) {
      throw new Error("ds4 /v1/models unreachable");
    }

    // Real reasoning effort from the engine itself (/v1/stats carries
    // server.reasoning_effort). Beats the log-tail guess.
    try {
      const statsRes = await this._fetch(`${this.baseUrl}/v1/stats`, {
        headers: { Accept: "application/json" },
      });
      if (statsRes.ok) {
        const stats = await statsRes.json();
        const effort = stats?.server?.reasoning_effort;
        if (effort) {
          this.reasoningEffort = effort;
          this.reasoningEffortTs = Date.now();
        }
      }
    } catch {}

    // Parse /metrics
    try {
      const metricsRes = await this._fetch(`${this.baseUrl}/metrics`);
      if (metricsRes.ok) {
        const txt = await metricsRes.text();

        // Gauges
        this.ds4Uptime = this._getDs4Metric(txt, "ds4_uptime_seconds");
        this.generationTps = this._getDs4Metric(txt, "ds4_decode_tok_s") ?? 0;
        this.prefillTps = this._getDs4Metric(txt, "ds4_prefill_tok_s") ?? 0;
        this.dsparkAcceptRatio = this._getDs4Metric(txt, "ds4_spec_accept_ratio");
        this.tokPerStep = this._getDs4Metric(txt, "ds4_tok_per_step");
        this.banksLive = this._getDs4Metric(txt, "ds4_banks_live");
        this.banksTotal = this._getDs4Metric(txt, "ds4_banks_total");
        this.kvPagesResident = this._getDs4Metric(txt, "ds4_kv_pages_resident");
        this.warmRecords = this._getDs4Metric(txt, "ds4_warm_records");
        this.derivedArtifacts = this._getDs4Metric(txt, "ds4_derived_artifacts");
        this.derivedArtifactBytes = this._getDs4Metric(txt, "ds4_derived_artifact_bytes");
        this.requestsInflight = this._getDs4Metric(txt, "ds4_requests_inflight");

        // Counters
        this.totalTokensDecoded = this._getDs4Metric(txt, "ds4_tokens_decoded_total");
        this.decodeSteps = this._getDs4Metric(txt, "ds4_decode_steps_total");
        this.specDrafts = this._getDs4Metric(txt, "ds4_spec_drafts_total");
        this.specHits = this._getDs4Metric(txt, "ds4_spec_hits_total");
        this.specQuench = this._getDs4Metric(txt, "ds4_spec_quench_total");
        this.requestsStarted = this._getDs4Metric(txt, "ds4_requests_started_total");
        this.requestsSerial = this._getDs4Metric(txt, "ds4_requests_serial_total");
        this.contAdmitRejects = this._getDs4Metric(txt, "ds4_cont_admit_rejects_total");
        this.contBatchFailures = this._getDs4Metric(txt, "ds4_cont_batch_failures_total");
        this.graphFitRefusals = this._getDs4Metric(txt, "ds4_graph_fit_refusals_total");

        // Labeled counters
        this.requestsCompleted = this._getDs4LabeledMetric(txt, "ds4_requests_total", "outcome", "completed");
        this.requestsFailed = this._getDs4LabeledMetric(txt, "ds4_requests_total", "outcome", "failed");
        this.requestsRefusedDeepSerial = this._getDs4LabeledMetric(txt, "ds4_requests_total", "outcome", "refused_deep_serial");
        this.prefillCached = this._getDs4LabeledMetric(txt, "ds4_tokens_prefilled_total", "kind", "cached");
        this.prefillComputed = this._getDs4LabeledMetric(txt, "ds4_tokens_prefilled_total", "kind", "computed");
        this.admitsCold = this._getDs4LabeledMetric(txt, "ds4_admits_total", "kind", "cold");
        this.admitsWarm = this._getDs4LabeledMetric(txt, "ds4_admits_total", "kind", "warm");
        this.admitsFork = this._getDs4LabeledMetric(txt, "ds4_admits_total", "kind", "fork");
        this.admitsPartialFork = this._getDs4LabeledMetric(txt, "ds4_admits_total", "kind", "partial_fork");
        this.admitsPartialTruncate = this._getDs4LabeledMetric(txt, "ds4_admits_total", "kind", "partial_truncate");

        // Slots = banks_live (active lanes), slotsTotal = banks_total
        this.slotsActive = this.banksLive != null ? Math.round(this.banksLive) : 0;
        this.slotsTotal = this.banksTotal != null ? Math.round(this.banksTotal) : 0;
        this.requestsRunning = this.requestsInflight;

        // Total output tokens from decoded counter
        if (this.totalTokensDecoded != null) {
          this.totalOutputTokens = Math.round(this.totalTokensDecoded);
        }

        // Track peak aggregate tok/s
        const currentAggregate = this.generationTps;
        if (currentAggregate > this.peakAggregateTps) {
          this.peakAggregateTps = currentAggregate;
        }

        // Per-stream tracking: use banks_live as the number of active streams
        // When inflight > 0, per-stream = decode_tok_s / inflight
        const inflight = this.requestsInflight != null ? this.requestsInflight : 0;
        if (inflight > 0 && currentAggregate > 0) {
          const perStream = currentAggregate / inflight;
          if (this.perStreamHigh == null || perStream > this.perStreamHigh) {
            this.perStreamHigh = Math.round(perStream * 100) / 100;
          }
          if (this.perStreamLow == null || perStream < this.perStreamLow) {
            this.perStreamLow = Math.round(perStream * 100) / 100;
          }
          this.perStreamAvg = Math.round(perStream * 100) / 100;
        }

        // MTP/spec acceptance — use ds4_spec_accept_ratio as the gauge
        this.mtpAcceptanceRate = this.dsparkAcceptRatio;
        this.mtpAcceptedTokens = this.specHits;
        this.mtpDraftedTokens = this.specDrafts;

        // ── Derive latency, genTokensPerReq, and rolling averages from
        //    ds4 counter deltas (ds4 has no latency histograms, so we
        //    approximate from throughput + completed-request counts). ──
        const nowMs = Date.now();
        const prev = this._ds4Prev;
        const dt = prev.time > 0 ? (nowMs - prev.time) / 1000 : 0;

        const deltaDecoded =
            this.totalTokensDecoded != null && prev.tokensDecoded != null
                ? Math.max(0, this.totalTokensDecoded - prev.tokensDecoded)
                : 0;
        const deltaSteps =
            this.decodeSteps != null && prev.decodeSteps != null
                ? Math.max(0, this.decodeSteps - prev.decodeSteps)
                : 0;
        const deltaStarted =
            this.requestsStarted != null && prev.requestsStarted != null
                ? Math.max(0, this.requestsStarted - prev.requestsStarted)
                : 0;
        const deltaCompleted =
            this.requestsCompleted != null && prev.requestsCompleted != null
                ? Math.max(0, this.requestsCompleted - prev.requestsCompleted)
                : 0;
        const deltaPrefillComputed =
            this.prefillComputed != null && prev.prefillComputed != null
                ? Math.max(0, this.prefillComputed - prev.prefillComputed)
                : 0;

        // Per-request average tokens (generation) — if requests completed
        // this cycle, avg tokens per request = deltaDecoded / deltaCompleted.
        // Fallback to cumulative if we have totals.
        if (deltaCompleted > 0) {
          this.genTokensPerReq =
              Math.round((deltaDecoded / deltaCompleted) * 100) / 100;
        } else if (this.requestsCompleted != null && this.requestsCompleted > 0) {
          this.genTokensPerReq =
              Math.round((this.totalTokensDecoded / this.requestsCompleted) * 100) / 100;
        }

        // Approximate TTFT: prefill time for the average request.
        // Use counter-based prefill rate (deltaPrefillComputed / dt) instead of
        // the instantaneous prefillTps gauge, which is near-zero between bursts.
        // When prefillTps is very low or zero (idle), fall back to the rolling
        // average prefill rate from the DS4 rolling window.
        const prefillRate = dt > 0 && deltaPrefillComputed > 0
            ? deltaPrefillComputed / dt
            : this.prefillTps > 0
                ? this.prefillTps
                : (this._ds4Rolling.length > 0
                    ? (() => {
                        // Estimate prefill rate from rolling window tokens and e2e
                        const last = this._ds4Rolling[this._ds4Rolling.length - 1];
                        return last && last.e2e > 0 ? last.tokens / last.e2e : 0;
                      })()
                    : 0);
        // Also compute average prompt tokens from cumulative counters as fallback
        // (used by the rolling-window prefill rate estimation above)
        if (deltaCompleted > 0 && deltaPrefillComputed > 0 && prefillRate > 0) {
          const avgPromptTokens = deltaPrefillComputed / deltaCompleted;
          this.ttft = Math.round((avgPromptTokens / prefillRate) * 1000) / 1000;
          this.ttftP95Seconds = this.ttft; // best estimate (no histogram)
        } else if (this.rollingAvgTtft != null && this.rollingAvgTtft > 0) {
          // Use rolling average TTFT from prior cycles — more reliable than
          // a cumulative estimate when prefill rate is very low or zero.
          this.ttft = this.rollingAvgTtft;
          this.ttftP95Seconds = this.rollingAvgTtft;
        }

        // Approximate E2E: TTFT + decode time for avg request.
        // decode time ≈ avgGenTokens / decodeRate, where decodeRate = generationTps / inflight.
        if (deltaCompleted > 0 && this.genTokensPerReq != null && this.genTokensPerReq > 0) {
          const inflight = this.requestsInflight != null ? Math.max(1, this.requestsInflight) : 1;
          const decodeRate = this.generationTps > 0 ? this.generationTps / inflight : 0;
          const ttftEst = this.ttft ?? 0;
          if (decodeRate > 0) {
            const decodeTime = this.genTokensPerReq / decodeRate;
            this.e2eLatency = Math.round((ttftEst + decodeTime) * 1000) / 1000;
            this.e2eP95Seconds = this.e2eLatency;
          } else {
            this.e2eLatency = Math.round(ttftEst * 1000) / 1000;
            this.e2eP95Seconds = this.e2eLatency;
          }
        }

        // ── Rolling window: last 10 completed-request batches ──
        if (deltaCompleted > 0 && this.e2eLatency != null) {
          const activeSlots = this.banksLive != null ? Math.max(1, this.banksLive) : 1;
          const tpsPerSlot = dt > 0 && this.generationTps > 0
              ? this.generationTps / activeSlots
              : 0;
          const tokensPerReq = (deltaPrefillComputed + deltaDecoded) / deltaCompleted;

          this._ds4Rolling.push({
            e2e: this.e2eLatency,
            ttft: this.ttft ?? 0,
            tokens: tokensPerReq,
            tpsPerSlot: Math.round(tpsPerSlot * 100) / 100,
          });
          if (this._ds4Rolling.length > 10) {
            this._ds4Rolling = this._ds4Rolling.slice(-10);
          }
        }

        // Compute rolling averages
        if (this._ds4Rolling.length > 0) {
          const n = this._ds4Rolling.length;
          let sumE2e = 0, sumTtft = 0, sumTokens = 0, sumTps = 0;
          for (const r of this._ds4Rolling) {
            sumE2e += r.e2e;
            sumTtft += r.ttft;
            sumTokens += r.tokens;
            sumTps += r.tpsPerSlot;
          }
          this.rollingAvgE2e = Math.round((sumE2e / n) * 1000) / 1000;
          this.rollingAvgTtft = Math.round((sumTtft / n) * 1000) / 1000;
          this.rollingAvgTokensPerReq = Math.round((sumTokens / n) * 100) / 100;
          this.rollingAvgTpsPerSlot = Math.round((sumTps / n) * 100) / 100;
        }

        // ── Derive prefixCacheHitRate from prefill counters ──
        // hit rate = prefillCached / (prefillCached + prefillComputed)
        if (this.prefillCached != null && this.prefillComputed != null) {
          const total = this.prefillCached + this.prefillComputed;
          this.prefixCacheHitRate =
            total > 0 ? Math.round((this.prefillCached / total) * 10000) / 10000 : null;
        }

        // ── Derive itlP95Seconds ≈ 1 / perStreamAvg (inter-token latency) ──
        if (this.perStreamAvg != null && this.perStreamAvg > 0) {
          this.itlP95Seconds = Math.round((1 / this.perStreamAvg) * 1000) / 1000;
        } else if (this.generationTps > 0) {
          // Fallback: use aggregate generation rate
          this.itlP95Seconds = Math.round((1 / this.generationTps) * 1000) / 1000;
        }

        // ── Derive kvCacheUsage and gpuMemoryUtilization from DS4 memory census ──
        // ds4_memory_bytes{domain="unified_device",class="kv_primary",state="allocated"}
        // = KV cache live bytes on device.
        // gpuMemoryUtilization = total unified_device allocated / 121GB
        const kvBytes = this._getDs4MultiLabeledMetric(txt,
            "ds4_memory_bytes",
            { domain: "unified_device", class: "kv_primary", state: "allocated" });
        if (kvBytes != null && kvBytes > 0) {
          this.kvCacheUsage = Math.round((kvBytes / DS4_DEVICE_MEMORY_BYTES) * 10000) / 10000;
        }
        // gpuMemoryUtilization: sum all unified_device allocated bytes / 121GB
        const totalDeviceAllocated = this._getDs4MemoryDomainTotal(txt, "unified_device", "allocated");
        if (totalDeviceAllocated != null && totalDeviceAllocated > 0) {
          this.gpuMemoryUtilization = Math.round((totalDeviceAllocated / DS4_DEVICE_MEMORY_BYTES) * 10000) / 10000;
        }

        // ── Derive perPositionAcceptance as single-element array from overall ratio ──
        // DS4 doesn't break down spec acceptance by position, so we provide a
        // single-element array so the spec decode graph always has data.
        if (this.dsparkAcceptRatio != null) {
          this.perPositionAcceptance = [Math.round(this.dsparkAcceptRatio * 10000) / 10000];
        } else if (this.specHits != null && this.specDrafts != null && this.specDrafts > 0) {
          this.perPositionAcceptance = [Math.round((this.specHits / this.specDrafts) * 10000) / 10000];
        }

        // Aggregate decode TPS alias
        this.aggregateDecodeTps = this.generationTps;

        // Save state for next cycle
        this._ds4Prev = {
          tokensDecoded: this.totalTokensDecoded,
          decodeSteps: this.decodeSteps,
          requestsStarted: this.requestsStarted,
          requestsCompleted: this.requestsCompleted,
          prefillComputed: this.prefillComputed,
          prefillCached: this.prefillCached,
          time: nowMs,
        };
      }
    } catch {}

    this.backendType = "ds4";
    this._tailDs4LogForReasoningEffort();
    this._tailDs4LogForActiveContext();
    await this._collectRecipeInfo();
    return this._getSnapshot();
  }

  // ─── DS4 reasoning effort log tailing ──────────────────
  /**
   * Tail the ds4 log file for reasoning_effort entries.
   * The ds4 engine may log "reasoning_effort" or "effort=low|medium|high"
   * in request lines. We scan new bytes since last read.
   */
  _tailDs4LogForReasoningEffort() {
    try {
      const stat = statSync(DS4_LOG_PATH);
      if (!stat.isFile()) return;
      const currentSize = stat.size;
      // File was truncated or rotated — reset
      if (currentSize < this._ds4LogSize) {
        this._ds4LogSize = 0;
      }
      // No new bytes
      if (currentSize === this._ds4LogSize) return;

      const fd = openSync(DS4_LOG_PATH, "r");
      try {
        const buf = Buffer.alloc(Math.min(currentSize - this._ds4LogSize, 512 * 1024));
        const bytesRead = readSync(fd, buf, 0, buf.length, this._ds4LogSize);
        this._ds4LogSize = currentSize;
        if (bytesRead <= 0) return;

        const text = buf.subarray(0, bytesRead).toString("utf8");

        // Match patterns like:
        //   reasoning_effort=low
        //   reasoning_effort: medium
        //   "reasoning_effort":"high"
        //   effort=low
        const re = /reasoning_effort["'\s:=]+(\w+)|effort[=\s]+(low|medium|high)/gi;
        let m;
        let lastEffort = null;
        while ((m = re.exec(text)) !== null) {
          const val = (m[1] || m[2] || "").toLowerCase();
          if (val === "low" || val === "medium" || val === "high") {
            lastEffort = val;
          }
        }

        // DS4 engine doesn't log the explicit reasoning_effort value, but it
        // does log "thinking not closed, ignoring DSML in reasoning" when
        // reasoning/thinking mode is active. If we see that pattern and have
        // no explicit effort, default to "high" (DeepSeek V4 thinking mode).
        if (lastEffort == null && this.reasoningEffort == null &&
            /thinking.*reasoning|reasoning.*thinking/i.test(text)) {
          lastEffort = "high";
        }
        if (lastEffort) {
          this.reasoningEffort = lastEffort;
          this.reasoningEffortTs = Date.now();
        }
      } finally {
        closeSync(fd);
      }
    } catch {}
  }

  // ─── DS4 active context log tailing ─────────────────────
  /**
   * Tail the ds4 log file for active context size entries.
   * ds4 logs lines like:
   *   chat ctx=0..54515:54515 TOOLS prompt start
   *   chat ctx=0..69306:69306 TOOLS prompt start
   * The number after the last colon is the total context tokens in that request.
   * We scan new bytes since last read (shared with reasoning-effort tailing).
   */
  _tailDs4LogForActiveContext() {
    try {
      const stat = statSync(DS4_LOG_PATH);
      if (!stat.isFile()) return;
      const currentSize = stat.size;
      // File was truncated or rotated — reset
      if (currentSize < this._ds4LogSizeCtx) {
        this._ds4LogSizeCtx = 0;
      }
      // No new bytes
      if (currentSize === this._ds4LogSizeCtx) return;

      const fd = openSync(DS4_LOG_PATH, "r");
      try {
        const buf = Buffer.alloc(Math.min(currentSize - this._ds4LogSizeCtx, 512 * 1024));
        const bytesRead = readSync(fd, buf, 0, buf.length, this._ds4LogSizeCtx);
        this._ds4LogSizeCtx = currentSize;
        if (bytesRead <= 0) return;

        const text = buf.subarray(0, bytesRead).toString("utf8");

        // Match patterns like:
        //   chat ctx=0..54515:54515 TOOLS prompt start
        //   chat ctx=0..69306:69306 TOOLS prompt start
        // Capture the number after the last colon.
        const re = /ctx=0\.\.(\d+):(\d+)/g;
        let m;
        let lastCtx = null;
        while ((m = re.exec(text)) !== null) {
          const val = parseInt(m[2], 10);
          if (Number.isFinite(val) && val > 0) {
            lastCtx = val;
          }
        }

        // Fallback: parse "warm admit bank=N cached=C suffix=S" lines where
        // the total context tokens = cached + suffix. This covers the common
        // case where the ctx=0..N:N pattern hasn't been emitted (non-tools
        // requests). We take the most recent warm-admit as the active context.
        if (lastCtx == null) {
          const reWarm = /warm admit bank=\d+ cached=(\d+) suffix=(\d+)/g;
          let mw;
          while ((mw = reWarm.exec(text)) !== null) {
            const cached = parseInt(mw[1], 10);
            const suffix = parseInt(mw[2], 10);
            const total = cached + suffix;
            if (Number.isFinite(total) && total > 0) {
              lastCtx = total;
            }
          }
        }

        if (lastCtx != null) {
          this.activeContext = lastCtx;
          this.activeContextTs = Date.now();
        }
      } finally {
        closeSync(fd);
      }
    } catch {}
  }


  // ─── llama.cpp expanded /slots surface (re-applied from fork) ───────
  /** Capture expanded per-slot fields + derive llama.cpp latency/rolling metrics. */
  _applyLlamaExpandedSlots(slots, totalDecoded, totalPrompted, dtSec) {
    let promptTokens = null;
    let promptTokensProcessed = null;
    let promptTokensCache = null;
    let nCtx = null;
    let isProcessing = false;
    let nRemain = null;
    let nDecoded = null;
    let samplingParams = null;
    let speculativeTypes = null;
    let reasoningFormat = null;
    let chatFormat = null;
    let samplers = null;

    for (const slot of slots) {
      if (promptTokens == null && slot.n_prompt_tokens != null) promptTokens = slot.n_prompt_tokens;
      if (promptTokensProcessed == null && slot.n_prompt_tokens_processed != null) promptTokensProcessed = slot.n_prompt_tokens_processed;
      if (promptTokensCache == null && slot.n_prompt_tokens_cache != null) promptTokensCache = slot.n_prompt_tokens_cache;
      if (nCtx == null && slot.n_ctx != null) nCtx = slot.n_ctx;
      if (slot.is_processing) isProcessing = true;
      const nt = Array.isArray(slot.next_token) ? slot.next_token[0] : slot.next_token;
      if (nt) {
        if (nRemain == null && nt.n_remain != null) nRemain = nt.n_remain;
        if (nDecoded == null && nt.n_decoded != null) nDecoded = nt.n_decoded;
      }
      const p = slot.params;
      if (p) {
        if (samplingParams == null) {
          samplingParams = {
            temperature: p.temperature ?? null,
            top_k: p.top_k ?? null,
            top_p: p.top_p ?? null,
            min_p: p.min_p ?? null,
            max_tokens: p.max_tokens ?? p.n_predict ?? null,
            n_predict: p.n_predict ?? null,
            n_keep: p.n_keep ?? null,
            n_discard: p.n_discard ?? null,
            stream: p.stream ?? null,
            repeat_penalty: p.repeat_penalty ?? null,
            presence_penalty: p.presence_penalty ?? null,
            frequency_penalty: p.frequency_penalty ?? null,
          };
        }
        if (speculativeTypes == null && p["speculative.types"] != null) speculativeTypes = p["speculative.types"];
        if (reasoningFormat == null && p.reasoning_format != null) reasoningFormat = p.reasoning_format;
        if (chatFormat == null && p.chat_format != null) chatFormat = p.chat_format;
        if (samplers == null && Array.isArray(p.samplers)) samplers = p.samplers;
      }
    }

    this.promptTokens = promptTokens;
    this.promptTokensProcessed = promptTokensProcessed;
    this.promptTokensCache = promptTokensCache;
    this.nCtx = nCtx;
    this.isProcessing = isProcessing;
    this.nRemain = nRemain;
    this.nDecoded = nDecoded;
    this.samplingParams = samplingParams;
    this.speculativeTypes = speculativeTypes;
    this.reasoningFormat = reasoningFormat;
    this.chatFormat = chatFormat;
    this.samplers = samplers;
    // Cache hit ratio = cache tokens / (cache + processed)
    if (promptTokensCache != null && promptTokensProcessed != null) {
      const denom = promptTokensCache + promptTokensProcessed;
      this.cacheHitRatio = denom > 0 ? Math.round((promptTokensCache / denom) * 10000) / 10000 : null;
    } else {
      this.cacheHitRatio = null;
    }

    // Rolling/latency derivation from counter deltas across probe cycles.
    const nowMs = Date.now();
    const prev = this._llamaPrev;
    const dt = prev.time > 0 ? (nowMs - prev.time) / 1000 : 0;
    const dDecoded = Math.max(0, totalDecoded - prev.decoded);
    const dPrompted = Math.max(0, totalPrompted - prev.prompted);
    this._llamaPrev = { decoded: totalDecoded, prompted: totalPrompted, time: nowMs };

    if (dtSec > 0 && this.generationTps > 0) {
      const activeSlots = this.slotsActive > 0 ? this.slotsActive : 1;
      const perStream = this.generationTps / activeSlots;
      if (this.perStreamHigh == null || perStream > this.perStreamHigh) {
        this.perStreamHigh = Math.round(perStream * 100) / 100;
      }
      if (this.perStreamLow == null || perStream < this.perStreamLow) {
        this.perStreamLow = Math.round(perStream * 100) / 100;
      }
      this.perStreamAvg = Math.round(perStream * 100) / 100;
     this.itlP95Seconds = Math.round((1 / perStream) * 1000) / 1000;
    }

    if (dDecoded > 0 && this.e2eLatency != null) {
      const activeSlots = this.slotsActive > 0 ? this.slotsActive : 1;
      const tpsPerSlot = dt > 0 && this.generationTps > 0 ? this.generationTps / activeSlots : 0;
      this._llamaRolling.push({
        e2e: this.e2eLatency,
        ttft: this.ttft ?? 0,
        tpsPerSlot: Math.round(tpsPerSlot * 100) / 100,
      });
      if (this._llamaRolling.length > 10) {
        this._llamaRolling = this._llamaRolling.slice(-10);
      }
      if (this._llamaRolling.length > 0) {
        const n = this._llamaRolling.length;
        let sumE2e = 0, sumTtft = 0, sumTps = 0;
        for (const r of this._llamaRolling) {
          sumE2e += r.e2e;
          sumTtft += r.ttft;
          sumTps += r.tpsPerSlot;
        }
        this.rollingAvgE2e = Math.round((sumE2e / n) * 1000) / 1000;
        this.rollingAvgTtft = Math.round((sumTtft / n) * 1000) / 1000;
        this.rollingAvgTpsPerSlot = Math.round((sumTps / n) * 100) / 100;
      }
    }
  }

  // ─── llama.cpp speculative-decode log tailing (re-applied from fork) ──
  /**
   * Tail the llama.cpp server log for "draft acceptance = X" lines, which the
   * /slots API does NOT expose.
   */
  _tailLlamaLogForSpecDecode() {
    try {
      const stat = statSync(LLAMA_LOG_PATH);
      if (!stat.isFile()) return;
      const currentSize = stat.size;
      if (currentSize < this._llamaLogSize) {
        this._llamaLogSize = 0;
      }
      if (currentSize === this._llamaLogSize) return;

      const fd = openSync(LLAMA_LOG_PATH, "r");
      try {
        const buf = Buffer.alloc(Math.min(currentSize - this._llamaLogSize, 512 * 1024));
        const bytesRead = readSync(fd, buf, 0, buf.length, this._llamaLogSize);
        this._llamaLogSize = currentSize;
        if (bytesRead <= 0) return;

        const text = buf.subarray(0, bytesRead).toString("utf8");
        const re = /draft acceptance\s*=\s*([\d.]+)\s*\(\s*(\d+)\s+accepted\s*\/\s*(\d+)\s+generated\)[^,]*,\s*mean len\s*=\s*([\d.]+)/g;
        let m;
        let last = null;
        while ((m = re.exec(text)) !== null) {
          const rate = parseFloat(m[1]);
          const accepted = parseInt(m[2], 10);
          const generated = parseInt(m[3], 10);
          const meanLen = parseFloat(m[4]);
          if (Number.isFinite(rate)) {
            last = { rate, accepted, generated, meanLen };
          }
        }
        if (last) {
          this.specAcceptanceRate = last.rate;
          this.specAcceptedTokens = last.accepted;
          this.specGeneratedTokens = last.generated;
          this.specMeanLen = last.meanLen;
        }
      } finally {
        closeSync(fd);
      }
    } catch {}
  }

  async _probeLlamaCpp() {
    this._clearEngineMemory();
    const now = Date.now();
    const dtSec = (now - this.lastProbeTime) / 1000;
    this.lastProbeTime = now;

    let slotsOk = false;
    try {
      const slotsRes = await this._fetch(`${this.baseUrl}/slots`);
      const auth = this._noteAuthStatus(slotsRes.status);
      if (auth === "auth") {
        return this._getSnapshot();
      }
      if (auth === "ok") {
        const slots = await slotsRes.json();
        if (Array.isArray(slots)) {
          slotsOk = true;
          this.slotsTotal = slots.length;
          this.slotsActive = slots.filter((s) => s.is_processing || (s.state && s.state !== "idle")).length;

          let totalGen = 0;
          let totalPrefill = 0;
          let totalDecoded = 0;
          let promptedSum = 0;
          let cachedSum = 0;
          let sawCache = false;

          for (const slot of slots) {
            const slotId = slot.id ?? "default";
            const decoded = this._getSlotDecoded(slot);
            const prompted = this._getSlotPrefilled(slot);
            const cached = this._getSlotCached(slot);
            totalDecoded += decoded;
            promptedSum += prompted;
            if (cached != null) {
              sawCache = true;
              cachedSum += cached;
            }
            const lastState = this.slotState.get(slotId) || { decoded: 0, prompted: 0 };
            const dDecoded = decoded - lastState.decoded;
            const dPrompted = prompted - lastState.prompted;
            this.slotState.set(slotId, { decoded, prompted });
            if (dtSec > 0 && dtSec < 10) {
              totalGen += dDecoded / dtSec;
              totalPrefill += dPrompted / dtSec;
            }

            // Capture expanded fields from the first slot that has them.
            if (promptTokens == null && slot.n_prompt_tokens != null) promptTokens = slot.n_prompt_tokens;
            if (promptTokensProcessed == null && slot.n_prompt_tokens_processed != null) promptTokensProcessed = slot.n_prompt_tokens_processed;
            if (promptTokensCache == null && slot.n_prompt_tokens_cache != null) promptTokensCache = slot.n_prompt_tokens_cache;
            if (nCtx == null && slot.n_ctx != null) nCtx = slot.n_ctx;
            if (slot.is_processing) isProcessing = true;
            const nt = Array.isArray(slot.next_token) ? slot.next_token[0] : slot.next_token;
            if (nt) {
              if (nRemain == null && nt.n_remain != null) nRemain = nt.n_remain;
              if (nDecoded == null && nt.n_decoded != null) nDecoded = nt.n_decoded;
            }
            const p = slot.params;
            if (p) {
              if (samplingParams == null) {
                samplingParams = {
                  temperature: p.temperature ?? null,
                  top_k: p.top_k ?? null,
                  top_p: p.top_p ?? null,
                  min_p: p.min_p ?? null,
                  max_tokens: p.max_tokens ?? p.n_predict ?? null,
                  n_predict: p.n_predict ?? null,
                  n_keep: p.n_keep ?? null,
                  n_discard: p.n_discard ?? null,
                  stream: p.stream ?? null,
                  repeat_penalty: p.repeat_penalty ?? null,
                  presence_penalty: p.presence_penalty ?? null,
                  frequency_penalty: p.frequency_penalty ?? null,
                };
              }
              if (speculativeTypes == null && p["speculative.types"] != null) speculativeTypes = p["speculative.types"];
              if (reasoningFormat == null && p.reasoning_format != null) reasoningFormat = p.reasoning_format;
              if (chatFormat == null && p.chat_format != null) chatFormat = p.chat_format;
              if (samplers == null && Array.isArray(p.samplers)) samplers = p.samplers;
            }
          }

          this.totalOutputTokens = totalDecoded;
          this.totalPromptTokens = promptedSum;
          if (sawCache) this.totalCachedTokens = cachedSum;
          this.generationTps = Math.max(0, Math.round(totalGen * 100) / 100);
          this._setPrefillTps(totalPrefill, totalGen > 0);
          if (sawCache) this._setPrefillSplitRates(cachedSum, promptedSum, dtSec);
          this._applyLlamaExpandedSlots(slots, totalDecoded, promptedSum, dtSec);
        }
      }
    } catch {}

    if (!slotsOk) {
      throw new Error("llama.cpp /slots unreachable");
    }

    try {
      const propsRes = await this._fetch(`${this.baseUrl}/props`);
      if (propsRes.ok) {
        const props = await propsRes.json();
        const raw = props.model_alias || props.model_path || this.modelId;
        if (props.model_path && !isHfHubCachePath(props.model_path)) {
          this.modelPath = props.model_path;
        } else if (isHfHubCachePath(props.model_path) || isHfHubCachePath(props.model_alias)) {
          this.modelPath = null;
        }
        if (raw) this.modelId = normalizeModelId(raw);
        this.contextLength = props.total_context_length || props.context_length || this.contextLength;
      }
    } catch {}

    this.backendType = "llama.cpp";
    this._tailLlamaLogForSpecDecode();
    return this._getSnapshot();
  }

  // ─── Metrics helpers ─────────────────────────────────────
  /**
   * Sum all Prometheus series matching `name` (optional labels).
   * @param {string} body
   * @param {string} name Full metric name, e.g. "ds4_decode_tok_s" or "vllm:prompt_tokens_total"
   * @returns {number | null}
   */
  _getPromMetric(body, name) {
    const esc = name.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const re = new RegExp(`^${esc}(?:\\{[^}]*\\})?\\s+([\\d.eE+-]+)\\s*$`, "gm");
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
   * Max of Prometheus series matching `name` (avoids summing HiCache layers).
   * @param {string} body
   * @param {string} name
   * @returns {number | null}
   */
  _getPromMetricMax(body, name) {
    const esc = name.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const re = new RegExp(`^${esc}(?:\\{[^}]*\\})?\\s+([\\d.eE+-]+)\\s*$`, "gm");
    let best = null;
    let m;
    while ((m = re.exec(body)) !== null) {
      const v = parseFloat(m[1]);
      if (Number.isFinite(v)) best = best == null ? v : Math.max(best, v);
    }
    return best;
  }

  /**
   * Sum series of `name` whose label `labelKey` equals `labelValue`.
   * @param {string} body
   * @param {string} name
   * @param {string} labelKey
   * @param {string} labelValue
   * @returns {number | null}
   */
  _getPromMetricLabeled(body, name, labelKey, labelValue) {
    const escName = name.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const escKey = labelKey.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const escVal = labelValue.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const re = new RegExp(
      `^${escName}\\{[^}]*\\b${escKey}="${escVal}"[^}]*\\}\\s+([\\d.eE+-]+)\\s*$`,
      "gm"
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

  _getVllmMetric(body, name) {
    return this._getPromMetric(body, `vllm:${name}`);
  }

  /**
   * Model-proof vLLM series lookup. `names` is an ordered candidate list;
   * each entry is either a plain suffix ( tried as `vllm:<name>` for older
   * builds that export an unprefixed form) or a `["~", fragment]` fuzzy
   * match: the FIRST counter/gauge series whose name contains the fragment.
   * Returns the summed value of the first matching set, or null when the
   * engine exposes nothing for this family (honest absence — never 0).
   * @param {string} body
   * @param {(string | string[])[]} candidates
   * @returns {number | null}
   */
  _getVllmSeries(body, candidates) {
    for (const cand of candidates) {
      if (typeof cand === "object" && cand !== null) {
        const fragment = String(cand[1] ?? cand[0] ?? "");
        if (!fragment) continue;
        const esc = fragment.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
        const re = new RegExp(`^vllm:[a-z0-9_]*${esc}(?:\\{[^}]*\\})?\\s+([\\d.eE+-]+)\\s*$`, "gm");
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
        if (found) return sum;
      } else {
        let v = this._getPromMetric(body, `vllm:${cand}`);
        if (v == null) v = this._getPromMetric(body, cand);
        if (v != null) return v;
      }
    }
    return null;
  }

  /**
   * Record which vLLM metric families this engine build does NOT expose, for
   * honest UNAVAILABLE markers in the panel (vs a dead 0). Called each poll
   * on the current /metrics body, so it tracks the live engine, not a guess.
   * @param {string} txt
   * @returns {string[]} snapshot key names whose source series is absent
   */
  _refreshVllmSeriesUnavailable(txt) {
    const absent = [];
    if (!~txt.indexOf("generation_tokens_total") && !~txt.indexOf("completion_tokens_total")) absent.push("generationTps");
    if (!~txt.indexOf("prompt_tokens_total")) absent.push("prefillTps");
    if (!~txt.indexOf("spec_decode_num_accepted_tokens_total") && !~txt.indexOf("spec_decode_num_draft_tokens_total")) absent.push("specDecode");
    if (!~txt.indexOf("time_to_first_token_seconds")) absent.push("ttft");
    if (!~txt.indexOf("inter_token_latency_seconds")) absent.push("itlP95");
    if (!~txt.indexOf("prefix_cache_hits_total")) absent.push("prefixCache");
    if (!~txt.indexOf("kv_cache_usage_perc") && !~txt.indexOf("gpu_cache_usage_perc")) absent.push("kvCache");
    this._vllmSeriesUnavailable = absent;
    return absent;
  }

  /**
   * Sum a numeric Prometheus label across every series of `name`.
   * vLLM publishes the KV pool on the info series `vllm:cache_config_info`
   * (value is always 1; `kv_cache_size_tokens` / `kv_cache_memory_bytes` are labels).
   * @param {string} body
   * @param {string} name
   * @param {string} label
   * @returns {number | null}
   */
  _sumPromLabel(body, name, label) {
    const escName = name.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const escLabel = label.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const re = new RegExp(`^${escName}\\{([^}]*)\\}\\s+[\\d.eE+-]+\\s*$`, "gm");
    const labelRe = new RegExp(`(?:^|,)\\s*${escLabel}="([^"]*)"`);
    let sum = 0;
    let found = false;
    let m;
    while ((m = re.exec(body)) !== null) {
      const lm = labelRe.exec(m[1]);
      if (!lm) continue;
      const v = Number(lm[1]);
      if (Number.isFinite(v) && v > 0) {
        sum += v;
        found = true;
      }
    }
    return found ? sum : null;
  }

  /** Drop every engine memory figure: the KV pool and the weights/KV GB split. */
  _clearEngineMemory() {
    this._clearKvPool();
    this.kvCacheGb = null;
    this.weightsGb = null;
  }

  /** Drop a previously observed KV pool (other backends do not expose one). */
  _clearKvPool() {
    this.kvCacheTokens = null;
    this.kvCacheTokensAvailable = null;
    this.kvCacheMemoryBytes = null;
  }

  /**
   * vLLM KV-cache pool from `cache_config_info` plus the usage gauge.
   * Free tokens = pool × (1 − usage). Usage is clamped to 0–1 so a
   * multi-engine sum cannot drive the free count negative.
   * @param {string} txt
   */
  /**
   * Spec-decode + request counters from Prometheus /metrics (lifetime).
   * Our vLLM build exposes nothing in /server_info speculative; everything
   * lives in prom counters, readable even at idle.
   */
  _applyVllmSpec(txt) {
    const drafts = this._getPromMetric(txt, "vllm:spec_decode_num_drafts_total");
    const draftTokens = this._getPromMetric(txt, "vllm:spec_decode_num_draft_tokens_total");
    const accepted = this._getPromMetric(txt, "vllm:spec_decode_num_accepted_tokens_total");
    if (drafts != null) this.specDrafts = Math.round(drafts);
    if (accepted != null) this.specHits = Math.round(accepted);
    if (draftTokens != null) this.specGeneratedTokens = Math.round(draftTokens);
    if (accepted != null) this.mtpAcceptedTokens = Math.round(accepted);
    if (draftTokens != null) this.mtpDraftedTokens = Math.round(draftTokens);
    if (draftTokens > 0 && accepted != null) {
      this.mtpAcceptanceRate = Math.round((accepted / draftTokens) * 1000) / 1000;
      this.specAcceptanceRate = this.mtpAcceptanceRate;
    }
    if (drafts > 0 && draftTokens > 0) {
      this.specMeanLen = Math.round((draftTokens / drafts) * 100) / 100;
    }
    const perPos = [];
    for (let p = 0; p < 8; p++) {
      const v = this._getPromMetricLabeled(txt, "vllm:spec_decode_num_accepted_tokens_per_pos_total", "position", String(p));
      if (v != null) perPos.push(v);
      else break;
    }
    // The per-pos series is a CUMULATIVE accepted-token counter per draft
    // position, not a rate. The panel renders each slot as a 0..100% bar, so
    // normalize by the lifetime draft-step counter (every verify step drafts
    // every position ≤ k, so count_k / drafts_total is the per-slot
    // accept ratio and stays ≤ 1). Without drafts the raw counts would be a
    // lie on the panel (e.g. "140400%"), so emit null and let the panel fall
    // back to the overall acceptance bar — honest absence beats a wrong bar.
    if (perPos.length && drafts != null && drafts > 0) {
      this.perPositionAcceptance = perPos.map((c) =>
        Math.max(0, Math.min(1, Math.round((c / drafts) * 1000) / 1000))
      );
    } else if (perPos.length) {
      this.perPositionAcceptance = null;
    }
    this.speculativeTypes = ["mtp"];
    // vLLM has no slots API; activeContext = tokens resident in KV
    if (this.kvUsedTokens != null) this.activeContext = this.kvUsedTokens;
  }
  _applyVllmKvPool(txt) {
    const tokens = this._sumPromLabel(txt, "vllm:cache_config_info", "kv_cache_size_tokens");
    const bytes = this._sumPromLabel(txt, "vllm:cache_config_info", "kv_cache_memory_bytes");
    this.kvCacheTokens = tokens;
    this.kvCacheMemoryBytes = bytes;
    if (tokens != null && this.kvCacheUsage != null && Number.isFinite(this.kvCacheUsage)) {
      const used = Math.min(1, Math.max(0, this.kvCacheUsage));
      this.kvCacheTokensAvailable = Math.max(0, Math.round(tokens * (1 - used)));
      this.kvUsedTokens = Math.round(tokens * used);
      this.kvPoolTokens = Math.round(tokens);
      this.remainingCtx = Math.max(0, Math.round(tokens * (1 - used)));
      this.nRemain = this.remainingCtx;
    } else {
      this.kvCacheTokensAvailable = null;
    }
  }

  /**
   * Parse a Prometheus histogram from /metrics text.
   * Returns { buckets: [{upper, count}], total } with cumulative counts per
   * `le`, summed across label sets. `total` is the summed `_count` series
   * (or null). `countMetricName` is the full metric name (with prefix).
   */
  _parseHistogram(body, metricPrefix, countMetricName) {
    const esc = metricPrefix.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const bucketRe = new RegExp(
      `^${esc}_bucket\\{[^}]*\\ble="([^"]+)"[^}]*\\}\\s+([\\d.eE+-]+)\\s*$`,
      "gm"
    );
    const byUpper = new Map();
    let infCount = 0;
    let m;
    while ((m = bucketRe.exec(body)) !== null) {
      const le = m[1];
      const count = parseFloat(m[2]);
      if (!Number.isFinite(count)) continue;
      const upper = le === "+Inf" ? Infinity : parseFloat(le);
      if (upper !== Infinity && !Number.isFinite(upper)) continue;
      if (upper === Infinity) infCount += count;
      byUpper.set(upper, (byUpper.get(upper) || 0) + count);
    }
    const total = this._getPromMetric(body, countMetricName);
    // Prometheus invariant: +Inf bucket count == _count. Mismatch → refuse quantile.
    if (total != null && infCount > 0 && Math.abs(infCount - total) > 1e-6) {
      return { buckets: [], total: null };
    }
    const buckets = Array.from(byUpper, ([upper, count]) => ({ upper, count }));
    buckets.sort((a, b) => a.upper - b.upper);
    // Sum of the histogram (for deriving averages / rolling means).
    const sum = this._getVllmMetric(body, `${metricPrefix.replace(/^vllm:/, "")}_sum`);
    return { buckets, total, sum };
  }

  /**
   * Parse a vLLM Prometheus histogram from /metrics text (vllm: prefix).
   */
  _parseVllmHistogram(body, metricPrefix) {
    const name = metricPrefix.replace(/^vllm:/, "");
    return this._parseHistogram(body, metricPrefix, `vllm:${name}_count`);
  }

  /**
   * Prometheus-style linear interpolation for a histogram quantile.
   * Returns null when empty / invalid or target is in the +Inf tail.
   */
  _histogramQuantile(buckets, total, quantile) {
    if (!buckets || !buckets.length || total == null || total <= 0) return null;
    const target = total * quantile;
    let prevUpper = 0.0;
    let prevCount = 0.0;
    for (const { upper, count } of buckets) {
      if (count >= target) {
        if (!Number.isFinite(upper)) return null;
        if (count === prevCount) return upper;
        return prevUpper + (upper - prevUpper) * ((target - prevCount) / (count - prevCount));
      }
      prevUpper = upper;
      prevCount = count;
    }
    return null;
  }

  _getSlotDecoded(slot) {
    if (slot.n_decoded != null) {
      if (Array.isArray(slot.n_decoded)) return slot.n_decoded[0] || 0;
      return slot.n_decoded || 0;
    }
    if (Array.isArray(slot.next_token) && slot.next_token[0]?.n_decoded != null) {
      return slot.next_token[0].n_decoded;
    }
    return 0;
  }

  _getSlotPrefilled(slot) {
    if (slot?.n_prompt_tokens_processed != null) {
      const n = Number(slot.n_prompt_tokens_processed);
      if (Number.isFinite(n)) return n;
    }
    return slot?.n_prompt_tokens || 0;
  }

  /** Cached prompt tokens on a llama.cpp slot, or null when the field is absent. */
  _getSlotCached(slot) {
    if (slot == null || slot.n_prompt_tokens_cache == null) return null;
    const n = Number(slot.n_prompt_tokens_cache);
    return Number.isFinite(n) ? n : null;
  }

  /**
   * Observational exposure hint from probe target + unauthenticated reachability.
   * Does not claim process bind address (0.0.0.0 vs interface).
   */
  _buildPosture() {
    if (this.authOpen == null) return null;

    const host = llmProbeHost(this.spark);
    const scope = classifyHostScope(host);
    const keyed = Boolean(this._apiKey());
    /** @type {"open" | "protected" | "keyed"} */
    let auth;
    if (keyed) {
      // Key configured: success → keyed; 401/403 → protected (rejected)
      auth = this.authOpen === false ? "protected" : "keyed";
    } else {
      auth = this.authOpen ? "open" : "protected";
    }

    let level = "ok";
    if (auth === "open") {
      if (scope === "public") level = "danger";
      else if (scope === "local") level = "ok";
      else level = "warn"; // lan or unknown hostname
    } else if (keyed && auth === "protected") {
      level = "danger";
    }

    const scopeWords = {
      local: "loopback",
      lan: "LAN",
      public: "public",
      unknown: "unknown-host",
    };
    const shortScope = {
      local: "Local",
      lan: "LAN",
      public: "Public",
      unknown: "Host",
    };
    const label =
      auth === "protected"
        ? keyed
          ? "Bad API key"
          : "Auth required"
        : auth === "keyed"
          ? `API key · ${shortScope[scope]}`
          : `Open · ${shortScope[scope]}`;
    const detail =
      auth === "protected"
        ? keyed
          ? `Configured API key was rejected (401/403) · ${scopeWords[scope]} target (${host || "—"}).`
          : `API key required · ${scopeWords[scope]} target (${host || "—"}). Based on the configured probe host, not the process bind address.`
        : auth === "keyed"
          ? `Using configured API key · ${scopeWords[scope]} target (${host || "—"}). Based on the configured probe host, not the process bind address.`
          : `Unauthenticated · ${scopeWords[scope]} target (${host || "—"}). Based on the configured probe host, not the process bind address.`;

    return { level, auth, scope, label, detail };
  }

  async _collectRecipeInfo() {
    try {
      if (this.backendType === "ds4") {
        this.recipeInfo = this._collectDs4RecipeInfo();
      } else if (this.backendType === "sglang") {
        this.recipeInfo = await this._collectSglangRecipeInfo();
      } else if (this.backendType === "vllm") {
        this.recipeInfo = this._collectVllmRecipeInfo();
      } else {
        this.recipeInfo = null;
      }
    } catch(e) {
      console.error("[ds4-recipe] ERROR:", e.message, e.stack?.substring(0, 200));
      this.recipeInfo = null;
    }
  }

  /** Find the PID of the process listening on this.port by scanning host /proc. */
  _findHostPid() {
    try {
      const procDir = HOST_PROC;
      const entries = readdirSync(procDir);
      for (const pid of entries) {
        if (!/^\d+$/.test(pid)) continue;
        const cmdlinePath = `${procDir}/${pid}/cmdline`;
        try {
          const cmdline = readFileSync(cmdlinePath, "utf8");
          const parts = cmdline.split("\0").filter(Boolean);
          if (parts.length === 0) continue;
          // ds4-server or vllm or python processes
          const exe = parts[0].toLowerCase();
          if (exe.includes("ds4-server") || exe.includes("ds4")) {
            // Check if this process has --port matching our port
            const portArg = parts.find((p, i) => parts[i - 1] === "--port" && /^\d+$/.test(p));
            if (portArg && parseInt(portArg) === this.port) return parseInt(pid);
            // Also check for --host 0.0.0.0 --port <port> pattern
            const allArgs = parts.join(" ");
            if (allArgs.includes(`--port ${this.port}`) || allArgs.includes(`port=${this.port}`)) return parseInt(pid);
          }
          if (exe.includes("vllm") || exe.includes("python")) {
            const allArgs = parts.join(" ");
            if (allArgs.includes(`--port ${this.port}`) || allArgs.includes(`port=${this.port}`)) return parseInt(pid);
          }
        } catch {}
      }
    } catch {}
    return null;
  }


  /**
   * Read the served model's chat-template default reasoning_effort for vLLM.
   * Resolves the container model path to the host filesystem via the vLLM
   * process mountinfo, then parses the chat_template.jinja default. Returns
   * null when it cannot be read (caller hides the card rather than guessing).
   */
  _vllmChatTemplateReasoningEffort() {
    try {
      const pid = this._findHostPid();
      if (!pid) return null;
      const cl = readFileSync(`${HOST_PROC}/${pid}/cmdline`, "utf8").replace(/\0/g, " ");
      let modelPath = null;
      const modelMatch = cl.match(/--model\s+(\S+)/);
      if (modelMatch) modelPath = modelMatch[1];
      else {
        const serveIdx = cl.indexOf("serve");
        if (serveIdx >= 0) {
          const rest = cl.slice(serveIdx + 5).trim();
          const firstTok = rest.split(/\s+/)[0];
          if (firstTok && !firstTok.startsWith("--")) modelPath = firstTok;
        }
      }
      if (!modelPath) return null;
      // Resolve container path -> host path via the vLLM process mountinfo.
      let hostPath = modelPath;
      try {
        const mi = readFileSync(`${HOST_PROC}/${pid}/mountinfo`, "utf8");
        // Pick the mount whose mountpoint is the LONGEST prefix of modelPath
        // (e.g. /models over /) so the container path resolves to the host path.
        let bestLen = -1;
        for (const line of mi.split("\n")) {
          const m = line.match(/^\d+ \d+ \d+:\d+ (\S+) (\S+)/);
          if (m && modelPath.startsWith(m[2]) && m[2].length > bestLen) {
            bestLen = m[2].length;
            hostPath = m[1] + modelPath.slice(m[2].length);
          }
        }
      } catch {}
      const template = readFileSync(`${HOST_ROOT}${hostPath}/chat_template.jinja`, "utf8");
      const m = template.match(/reasoning_effort\s*\|\s*default\s*\(\s*['"]([^'"]+)['"]\s*\)/);
      return m ? m[1] : null;
    } catch {
      return null;
    }
  }

  /** Collect recipe info for the ds4 CUDA engine backend. */
  _collectDs4RecipeInfo() {
    const pid = this._findHostPid();
    if (!pid) return null;

    let environ = {};
    let cmdline = "";
    try {
      const envRaw = readFileSync(`${HOST_PROC}/${pid}/environ`, "utf8");
      for (const pair of envRaw.split("\0")) {
        if (!pair) continue;
        const eq = pair.indexOf("=");
        if (eq > 0) environ[pair.slice(0, eq)] = pair.slice(eq + 1);
      }
    } catch {}
    try {
      cmdline = readFileSync(`${HOST_PROC}/${pid}/cmdline`, "utf8").replace(/\0/g, " ").trim();
    } catch {}

    // Parse model file from cmdline: -m <path>
    const modelMatch = cmdline.match(/-m\s+(\S+)/);
    const modelPath = modelMatch ? modelMatch[1] : null;
    const modelFile = modelPath ? modelPath.split("/").pop() : null;

    // Detect quantization from model filename
    let quantization = null;
    if (modelFile) {
      if (/IQ2XXS/i.test(modelFile)) quantization = "IQ2XXS";
      else if (/IQ3/i.test(modelFile)) quantization = "IQ3";
      else if (/IQ4/i.test(modelFile)) quantization = "IQ4";
      else if (/Q2K/i.test(modelFile)) quantization = "Q2_K";
      else if (/Q4_K/i.test(modelFile)) quantization = "Q4_K";
      else if (/Q8_0/i.test(modelFile)) quantization = "Q8_0";
      else if (/FP8/i.test(modelFile)) quantization = "FP8";
      else if (/NVFP4/i.test(modelFile)) quantization = "NVFP4";
    }

    // Context length from cmdline: -c <num> (take last occurrence)
    let contextLength = this.contextLength;
    const ctxMatches = [...cmdline.matchAll(/-c\s+(\d+)/g)];
    if (ctxMatches.length > 0) {
      contextLength = parseInt(ctxMatches[ctxMatches.length - 1][1]);
    }

    // Max lanes from DS4_BATCH_FIT_HEADROOM_MB (maps to banks_total)
    const maxLanes = this.banksTotal ?? null;

    // DSpark config
    const dsparkEnabled = environ.DS4_CONT_DSPARK === "1" || environ.DS4_CONT_DSPARK === "true";
    const mtpMode = environ.DS4_CONT_MTP_MODE || null;
    const dsparkModel = environ.DS4_DSPARK_MODEL || null;

    let specDecodeMethod = null;
    if (dsparkEnabled) {
      const drafterFile = dsparkModel ? dsparkModel.split("/").pop() : null;
      // k value: MTP mode 2 = k=4 for DSpark typically
      const k = mtpMode ? `k=${mtpMode}` : "k=4";
      specDecodeMethod = `DSpark ${k}`;
    } else if (mtpMode) {
      specDecodeMethod = `MTP k=${mtpMode}`;
    }

    // KV cache dtype: ds4 uses native CUDA cache, no env var for dtype
    const kvCacheDtype = "native";

    // Prefix caching: ds4 always has warm/prefix cache (warmRecords)
    const prefixCaching = this.warmRecords != null ? this.warmRecords > 0 : null;

    // Author attribution for ds4
    const author = "@bleysg";
    const authorName = "Bleys Goodson";

    // Engine type
    const engineType = "DS4 CUDA Engine";

    // Container: native build
    const containerImage = "Native build (Entrpi/ds4 fork)";

    // Model display name from recipeMetadata
    const modelName = this.recipeMetadata?.model || this.modelId || modelFile || null;

    // Accept ratio
    const acceptRatio = this.dsparkAcceptRatio ?? null;

    // Uptime
    const uptime = this.ds4Uptime ?? null;

    return {
      engineType,
      modelName,
      containerImage,
      author,
      authorName,
      contextLength,
      maxLanes,
      specDecodeMethod,
      quantization,
      gmu: null, // ds4 doesn't expose GMU directly
      kvCacheDtype,
      prefixCaching,
      acceptRatio,
      uptime,
    };
  }

  /** Collect recipe info for a vLLM container backend. */
/** Collect recipe info for a vLLM container backend. */
  _collectVllmRecipeInfo() {
    // ── Source 1: docker (only reachable if the docker CLI is installed in this container) ──
    let containerImage = null;
    let containerName = null;
    let cmdline = "";
    let environ = {};
    try {
      // List containers, find one whose command line serves --port {this.port}.
      // Host-networked containers (--network host) expose NO port mapping in
      // {{.Ports}}, so match on the container's command line as well.
      const containersRaw = execSync(
        "docker ps --format '{{.Names}}\t{{.Image}}\t{{.Ports}}\t{{.Command}}'",
        { timeout: 5000, encoding: "utf8" }
      );
      for (const line of containersRaw.trim().split("\n")) {
        if (!line) continue;
        const [name, image, ports, command] = line.split("\t");
        const cmd = command || "";
        const portMatch =
          (ports && ports.includes(`${this.port}->`)) ||
          cmd.includes(`--port ${this.port}`) ||
          cmd.includes(`port=${this.port}`);
        if (portMatch) {
          containerImage = image;
          containerName = name;
          break;
        }
      }
    } catch {}

    // Pull env + cmdline from docker inspect when a container was found.
    if (containerName) {
      try {
        const inspectRaw = execSync(
          `docker inspect --format '{{range .Config.Env}}{{println .}}{{end}}' ${containerName}`,
          { timeout: 5000, encoding: "utf8" }
        );
        for (const line of inspectRaw.trim().split("\n")) {
          if (!line) continue;
          const eq = line.indexOf("=");
          if (eq > 0) environ[line.slice(0, eq)] = line.slice(eq + 1);
        }
      } catch {}
      try {
        cmdline = execSync(
          `docker inspect --format '{{range .Args}}{{.}} {{end}}' ${containerName}`,
          { timeout: 5000, encoding: "utf8" }
        ).trim();
      } catch {}
    }

    // ── Source 2: host /proc (works even without docker access) ──
    // The sparkDash container runs with pid:host and /proc mounted at HOST_PROC,
    // so we can read the vLLM process cmdline directly. This is the primary path
    // when the docker CLI is not installed inside the container.
    if (!cmdline) {
      const pid = this._findHostPid();
      if (pid) {
        try {
          cmdline = readFileSync(`${HOST_PROC}/${pid}/cmdline`, "utf8").replace(/\0/g, " ").trim();
        } catch {}
        try {
          const envRaw = readFileSync(`${HOST_PROC}/${pid}/environ`, "utf8");
          for (const pair of envRaw.split("\0")) {
            if (!pair) continue;
            const eq = pair.indexOf("=");
            if (eq > 0) environ[pair.slice(0, eq)] = pair.slice(eq + 1);
          }
        } catch {}
      }
    }

    if (!cmdline) return null;

    // Parse model from cmdline: --model <path>, or the positional arg after "serve"
    let modelPath = null;
    const modelMatch = cmdline.match(/--model\s+(\S+)/);
    if (modelMatch) modelPath = modelMatch[1];
    else {
      const serveIdx = cmdline.indexOf("serve");
      if (serveIdx >= 0) {
        const rest = cmdline.slice(serveIdx + 5).trim();
        const firstTok = rest.split(/\s+/)[0];
        if (firstTok && !firstTok.startsWith("--")) modelPath = firstTok;
      }
    }
    const modelFile = modelPath ? modelPath.split("/").pop() : null;
    // Normalize HF cache snapshots into their actual repository identity so the
    // dashboard shows the checkpoint/quant rather than only the served alias.
    const canonicalModel = normalizeModelId(modelPath);

    // Served model name (alias exposed via /v1/models)
    const servedNameMatch = cmdline.match(/--served-model-name\s+(\S+)/);
    const servedModelName = servedNameMatch ? servedNameMatch[1] : null;

    // Detect quantization
    let quantization = null;
    const quantArg = cmdline.match(/--quantization\s+(\S+)/);
    if (quantArg) {
      quantization = quantArg[1].toUpperCase();
    } else {
      // A snapshot basename is only a hash; the canonical HF repo carries the quant.
      const quantSource = canonicalModel || modelFile || "";
      if (/NVFP4/i.test(quantSource)) quantization = "NVFP4";
      else if (/FP8/i.test(quantSource)) quantization = "FP8";
      else if (/AWQ/i.test(quantSource)) quantization = "AWQ";
      else if (/GPTQ/i.test(quantSource)) quantization = "GPTQ";
    }

    // Context length from cmdline: --max-model-len <num>
    let contextLength = this.contextLength;
    const ctxMatch = cmdline.match(/--max-model-len\s+(\d+)/);
    if (ctxMatch) contextLength = parseInt(ctxMatch[1]);

    // Max lanes from cmdline: --tensor-parallel-size or -tp
    let maxLanes = null;
    const tpMatch = cmdline.match(/--tensor-parallel-size\s+(\d+)/);
    if (tpMatch) maxLanes = parseInt(tpMatch[1]);
    else {
      const tpShort = cmdline.match(/(?:^|\s)-tp\s+(\d+)/);
      if (tpShort) maxLanes = parseInt(tpShort[1]);
    }

    // Speculative decode method
    let specDecodeMethod = null;
    if (/--speculative-model/.test(cmdline) || /--speculative-config/.test(cmdline)) {
      const numSpecMatch = cmdline.match(/--num-speculative-tokens\s+(\d+)/);
      let k = numSpecMatch ? numSpecMatch[1] : null;
      if (!k) {
        // k may live inside the --speculative-config JSON, e.g. {"method":"mtp","num_speculative_tokens":3}
        const scMatch = cmdline.match(/--speculative-config\s+(\S+)/);
        if (scMatch) {
          const scTok = scMatch[1].replace(/^["']|["']$/g, "");
          try {
            const sc = JSON.parse(scTok);
            if (sc && sc.num_speculative_tokens != null) k = String(sc.num_speculative_tokens);
          } catch {}
        }
      }
      specDecodeMethod = `MTP k=${k || "?"}`;
    }

    // KV cache dtype
    let kvCacheDtype = null;
    const kvMatch = cmdline.match(/--kv-cache-dtype\s+(\S+)/);
    if (kvMatch) kvCacheDtype = kvMatch[1];
    else kvCacheDtype = "auto";

    // Prefix caching: from cache_config_info enable_prefix_caching label, else
    // fall back to the --enable-prefix-caching / --no-prefix-caching flags.
    let prefixCaching = this._vllmPrefixCaching;
    if (prefixCaching == null) {
      if (/--enable-prefix-caching/.test(cmdline)) prefixCaching = true;
      else if (/--no-prefix-caching/.test(cmdline)) prefixCaching = false;
    }
    try {
      const pid = this._findHostPid();
      if (pid) {
        const cl = readFileSync(`${HOST_PROC}/${pid}/cmdline`, "utf8").replace(/\0/g, " ");
        if (/--enable-prefix-caching/.test(cl)) prefixCaching = true;
        else if (/--no-prefix-caching/.test(cl)) prefixCaching = false;
      }
    } catch {}

    // GMU
    let gmu = null;
    const gmuMatch = cmdline.match(/--gpu-memory-utilization\s+([\d.]+)/);
    if (gmuMatch) gmu = parseFloat(gmuMatch[1]);

    // Reasoning / tool-call parsers + speculative config
    const reasoningParserMatch = cmdline.match(/--reasoning-parser\s+(\S+)/);
    const reasoningParser = reasoningParserMatch ? reasoningParserMatch[1] : null;
    const toolCallParserMatch = cmdline.match(/--tool-call-parser\s+(\S+)/);
    const toolCallParser = toolCallParserMatch ? toolCallParserMatch[1] : null;
    const speculativeConfigMatch = cmdline.match(/--speculative-config\s+(\S+)/);
    const speculativeConfig = speculativeConfigMatch ? speculativeConfigMatch[1] : null;

    // Author attribution for vLLM recipes
    const author = "@styles01";
    const authorName = "styles01";

    // Engine type
    const engineType = "vLLM";

    // Prefer the canonical checkpoint identity to its user-facing API alias.
    const modelName = canonicalModel || this.modelId || servedModelName || modelFile || null;

    // Accept ratio
    const acceptRatio = this.mtpAcceptanceRate ?? null;

    // Populate recipeMetadata so the provenance section has structured fields
    // (ownedBy, model, contextLength, parsers, etc.) even when docker is absent.
    this.recipeMetadata = {
      name: servedModelName || modelFile || this.modelId || null,
      model: servedModelName || modelFile || this.modelId || null,
      contextLength,
      ownedBy: "vllm",
      supportedParameters: [],
      quantization,
      gmu,
      maxModelLen: contextLength,
      servedModelName,
      reasoningParser,
      toolCallParser,
      speculativeConfig,
      modelPath,
    };

    return {
      engineType,
      modelName,
      containerImage,
      author,
      authorName,
      contextLength,
      maxLanes,
      specDecodeMethod,
      quantization,
      gmu,
      kvCacheDtype,
      prefixCaching,
      acceptRatio,
      uptime: this.ds4Uptime ?? null,
      servedModelName,
      reasoningParser,
      toolCallParser,
      speculativeConfig,
      modelPath,
    };
  }

  /**
   * Observational exposure hint from probe target + unauthenticated reachability.
   * Does not claim process bind address (0.0.0.0 vs interface).
   */
  _buildPosture() {
    if (this.authOpen == null) return null;

    const host = llmProbeHost(this.spark);
    const scope = classifyHostScope(host);
    const keyed = Boolean(this._apiKey());
    /** @type {"open" | "protected" | "keyed"} */
    let auth;
    if (keyed) {
      // Key configured: success → keyed; 401/403 → protected (rejected)
      auth = this.authOpen === false ? "protected" : "keyed";
    } else {
      auth = this.authOpen ? "open" : "protected";
    }

    let level = "ok";
    if (auth === "open") {
      if (scope === "public") level = "danger";
      else if (scope === "local") level = "ok";
      else level = "warn"; // lan or unknown hostname
    } else if (keyed && auth === "protected") {
      level = "danger";
    }

    const scopeWords = {
      local: "loopback",
      lan: "LAN",
      public: "public",
      unknown: "unknown-host",
    };
    const shortScope = {
      local: "Local",
      lan: "LAN",
      public: "Public",
      unknown: "Host",
    };
    const label =
      auth === "protected"
        ? keyed
          ? "Bad API key"
          : "Auth required"
        : auth === "keyed"
          ? `API key · ${shortScope[scope]}`
          : `Open · ${shortScope[scope]}`;
    const detail =
      auth === "protected"
        ? keyed
          ? `Configured API key was rejected (401/403) · ${scopeWords[scope]} target (${host || "—"}).`
          : `API key required · ${scopeWords[scope]} target (${host || "—"}). Based on the configured probe host, not the process bind address.`
        : auth === "keyed"
          ? `Using configured API key · ${scopeWords[scope]} target (${host || "—"}). Based on the configured probe host, not the process bind address.`
          : `Unauthenticated · ${scopeWords[scope]} target (${host || "—"}). Based on the configured probe host, not the process bind address.`;

    return { level, auth, scope, label, detail };
  }

  // ─── Recipe info / attribution collection ────────────────
  /**
   * Collect rich recipe info (engine type, model, container, author, config badges)
   * by inspecting the host process environment and command line.
   * Called once per probe cycle; cheap because it caches and short-circuits.
   */
async _collectSglangRecipeInfo() {
    try {
      const info = await this._getSglangModelInfo();
      console.log("[sglang-recipe] info=", JSON.stringify(info));
      if (!info) return null;
      const ctx = info.context_length || this.contextLength || null;
      const arch = Array.isArray(info.architectures) ? info.architectures[0] : null;
      const quant = info.quantization || (info.model_path && /nvfp4|fp4|int4|fp8/i.test(info.model_path) ? (info.model_path.match(/(nvfp4|fp4|int4|fp8)/i) || [])[1] : null) || null;
      return {
        engineType: "SGLang",
        modelName: info.model_path ? info.model_path.split("/").pop() : this.modelId,
        containerImage: "lmsysorg/sglang:dev-muse-glimmer",
        author: null,
        authorName: null,
        contextLength: ctx,
        maxLanes: null,
        specDecodeMethod: "DFlash",
        quantization: quant,
        gmu: info.mem_fraction_static != null ? info.mem_fraction_static : null,
        kvCacheDtype: info.kv_cache_dtype || null,
        prefixCaching: true,
        acceptRatio: this.mtpAcceptanceRate,
        ownedBy: info.model_type || arch || null,
      };
    } catch (e) {
      console.error("[sglang-recipe] ERROR:", e.message);
      return null;
    }
  }

  async _getSglangModelInfo() {
    try {
      const res = await this._fetch(`${this.baseUrl}/get_model_info`);
      if (!res.ok) return null;
      return await res.json();
    } catch {
      return null;
    }
  }

  async _collectRecipeInfo() {
    try {
      if (this.backendType === "ds4") {
        this.recipeInfo = this._collectDs4RecipeInfo();
      } else if (this.backendType === "sglang") {
        this.recipeInfo = await this._collectSglangRecipeInfo();
      } else if (this.backendType === "vllm") {
        this.recipeInfo = this._collectVllmRecipeInfo();
      } else {
        this.recipeInfo = null;
      }
    } catch(e) {
      console.error("[ds4-recipe] ERROR:", e.message, e.stack?.substring(0, 200));
      this.recipeInfo = null;
    }
  }

  /** Find the PID of the process listening on this.port by scanning host /proc. */
  _findHostPid() {
    try {
      const procDir = HOST_PROC;
      const entries = readdirSync(procDir);
      for (const pid of entries) {
        if (!/^\d+$/.test(pid)) continue;
        const cmdlinePath = `${procDir}/${pid}/cmdline`;
        try {
          const cmdline = readFileSync(cmdlinePath, "utf8");
          const parts = cmdline.split("\0").filter(Boolean);
          if (parts.length === 0) continue;
          // ds4-server or vllm or python processes
          const exe = parts[0].toLowerCase();
          if (exe.includes("ds4-server") || exe.includes("ds4")) {
            // Check if this process has --port matching our port
            const portArg = parts.find((p, i) => parts[i - 1] === "--port" && /^\d+$/.test(p));
            if (portArg && parseInt(portArg) === this.port) return parseInt(pid);
            // Also check for --host 0.0.0.0 --port <port> pattern
            const allArgs = parts.join(" ");
            if (allArgs.includes(`--port ${this.port}`) || allArgs.includes(`port=${this.port}`)) return parseInt(pid);
          }
          if (exe.includes("vllm") || exe.includes("python")) {
            const allArgs = parts.join(" ");
            if (allArgs.includes(`--port ${this.port}`) || allArgs.includes(`port=${this.port}`)) return parseInt(pid);
          }
        } catch {}
      }
    } catch {}
    return null;
  }


  /**
   * Read the served model's chat-template default reasoning_effort for vLLM.
   * Resolves the container model path to the host filesystem via the vLLM
   * process mountinfo, then parses the chat_template.jinja default. Returns
   * null when it cannot be read (caller hides the card rather than guessing).
   */
  _vllmChatTemplateReasoningEffort() {
    try {
      const pid = this._findHostPid();
      if (!pid) return null;
      const cl = readFileSync(`${HOST_PROC}/${pid}/cmdline`, "utf8").replace(/\0/g, " ");
      let modelPath = null;
      const modelMatch = cl.match(/--model\s+(\S+)/);
      if (modelMatch) modelPath = modelMatch[1];
      else {
        const serveIdx = cl.indexOf("serve");
        if (serveIdx >= 0) {
          const rest = cl.slice(serveIdx + 5).trim();
          const firstTok = rest.split(/\s+/)[0];
          if (firstTok && !firstTok.startsWith("--")) modelPath = firstTok;
        }
      }
      if (!modelPath) return null;
      // Resolve container path -> host path via the vLLM process mountinfo.
      let hostPath = modelPath;
      try {
        const mi = readFileSync(`${HOST_PROC}/${pid}/mountinfo`, "utf8");
        // Pick the mount whose mountpoint is the LONGEST prefix of modelPath
        // (e.g. /models over /) so the container path resolves to the host path.
        let bestLen = -1;
        for (const line of mi.split("\n")) {
          const m = line.match(/^\d+ \d+ \d+:\d+ (\S+) (\S+)/);
          if (m && modelPath.startsWith(m[2]) && m[2].length > bestLen) {
            bestLen = m[2].length;
            hostPath = m[1] + modelPath.slice(m[2].length);
          }
        }
      } catch {}
      const template = readFileSync(`${HOST_ROOT}${hostPath}/chat_template.jinja`, "utf8");
      const m = template.match(/reasoning_effort\s*\|\s*default\s*\(\s*['"]([^'"]+)['"]\s*\)/);
      return m ? m[1] : null;
    } catch {
      return null;
    }
  }

  /** Collect recipe info for the ds4 CUDA engine backend. */
  _collectDs4RecipeInfo() {
    const pid = this._findHostPid();
    if (!pid) return null;

    let environ = {};
    let cmdline = "";
    try {
      const envRaw = readFileSync(`${HOST_PROC}/${pid}/environ`, "utf8");
      for (const pair of envRaw.split("\0")) {
        if (!pair) continue;
        const eq = pair.indexOf("=");
        if (eq > 0) environ[pair.slice(0, eq)] = pair.slice(eq + 1);
      }
    } catch {}
    try {
      cmdline = readFileSync(`${HOST_PROC}/${pid}/cmdline`, "utf8").replace(/\0/g, " ").trim();
    } catch {}

    // Parse model file from cmdline: -m <path>
    const modelMatch = cmdline.match(/-m\s+(\S+)/);
    const modelPath = modelMatch ? modelMatch[1] : null;
    const modelFile = modelPath ? modelPath.split("/").pop() : null;

    // Detect quantization from model filename
    let quantization = null;
    if (modelFile) {
      if (/IQ2XXS/i.test(modelFile)) quantization = "IQ2XXS";
      else if (/IQ3/i.test(modelFile)) quantization = "IQ3";
      else if (/IQ4/i.test(modelFile)) quantization = "IQ4";
      else if (/Q2K/i.test(modelFile)) quantization = "Q2_K";
      else if (/Q4_K/i.test(modelFile)) quantization = "Q4_K";
      else if (/Q8_0/i.test(modelFile)) quantization = "Q8_0";
      else if (/FP8/i.test(modelFile)) quantization = "FP8";
      else if (/NVFP4/i.test(modelFile)) quantization = "NVFP4";
    }

    // Context length from cmdline: -c <num> (take last occurrence)
    let contextLength = this.contextLength;
    const ctxMatches = [...cmdline.matchAll(/-c\s+(\d+)/g)];
    if (ctxMatches.length > 0) {
      contextLength = parseInt(ctxMatches[ctxMatches.length - 1][1]);
    }

    // Max lanes from DS4_BATCH_FIT_HEADROOM_MB (maps to banks_total)
    const maxLanes = this.banksTotal ?? null;

    // DSpark config
    const dsparkEnabled = environ.DS4_CONT_DSPARK === "1" || environ.DS4_CONT_DSPARK === "true";
    const mtpMode = environ.DS4_CONT_MTP_MODE || null;
    const dsparkModel = environ.DS4_DSPARK_MODEL || null;

    let specDecodeMethod = null;
    if (dsparkEnabled) {
      const drafterFile = dsparkModel ? dsparkModel.split("/").pop() : null;
      // k value: MTP mode 2 = k=4 for DSpark typically
      const k = mtpMode ? `k=${mtpMode}` : "k=4";
      specDecodeMethod = `DSpark ${k}`;
    } else if (mtpMode) {
      specDecodeMethod = `MTP k=${mtpMode}`;
    }

    // KV cache dtype: ds4 uses native CUDA cache, no env var for dtype
    const kvCacheDtype = "native";

    // Prefix caching: ds4 always has warm/prefix cache (warmRecords)
    const prefixCaching = this.warmRecords != null ? this.warmRecords > 0 : null;

    // Author attribution for ds4
    const author = "@bleysg";
    const authorName = "Bleys Goodson";

    // Engine type
    const engineType = "DS4 CUDA Engine";

    // Container: native build
    const containerImage = "Native build (Entrpi/ds4 fork)";

    // Model display name from recipeMetadata
    const modelName = this.recipeMetadata?.model || this.modelId || modelFile || null;

    // Accept ratio
    const acceptRatio = this.dsparkAcceptRatio ?? null;

    // Uptime
    const uptime = this.ds4Uptime ?? null;

    return {
      engineType,
      modelName,
      containerImage,
      author,
      authorName,
      contextLength,
      maxLanes,
      specDecodeMethod,
      quantization,
      gmu: null, // ds4 doesn't expose GMU directly
      kvCacheDtype,
      prefixCaching,
      acceptRatio,
      uptime,
    };
  }

  /** Collect recipe info for a vLLM container backend. */
/** Collect recipe info for a vLLM container backend. */
  _collectVllmRecipeInfo() {
    // ── Source 1: docker (only reachable if the docker CLI is installed in this container) ──
    let containerImage = null;
    let containerName = null;
    let cmdline = "";
    let environ = {};
    try {
      // List containers, find one whose command line serves --port {this.port}.
      // Host-networked containers (--network host) expose NO port mapping in
      // {{.Ports}}, so match on the container's command line as well.
      const containersRaw = execSync(
        "docker ps --format '{{.Names}}\t{{.Image}}\t{{.Ports}}\t{{.Command}}'",
        { timeout: 5000, encoding: "utf8" }
      );
      for (const line of containersRaw.trim().split("\n")) {
        if (!line) continue;
        const [name, image, ports, command] = line.split("\t");
        const cmd = command || "";
        const portMatch =
          (ports && ports.includes(`${this.port}->`)) ||
          cmd.includes(`--port ${this.port}`) ||
          cmd.includes(`port=${this.port}`);
        if (portMatch) {
          containerImage = image;
          containerName = name;
          break;
        }
      }
    } catch {}

    // Pull env + cmdline from docker inspect when a container was found.
    if (containerName) {
      try {
        const inspectRaw = execSync(
          `docker inspect --format '{{range .Config.Env}}{{println .}}{{end}}' ${containerName}`,
          { timeout: 5000, encoding: "utf8" }
        );
        for (const line of inspectRaw.trim().split("\n")) {
          if (!line) continue;
          const eq = line.indexOf("=");
          if (eq > 0) environ[line.slice(0, eq)] = line.slice(eq + 1);
        }
      } catch {}
      try {
        cmdline = execSync(
          `docker inspect --format '{{range .Args}}{{.}} {{end}}' ${containerName}`,
          { timeout: 5000, encoding: "utf8" }
        ).trim();
      } catch {}
    }

    // ── Source 2: host /proc (works even without docker access) ──
    // The sparkDash container runs with pid:host and /proc mounted at HOST_PROC,
    // so we can read the vLLM process cmdline directly. This is the primary path
    // when the docker CLI is not installed inside the container.
    if (!cmdline) {
      const pid = this._findHostPid();
      if (pid) {
        try {
          cmdline = readFileSync(`${HOST_PROC}/${pid}/cmdline`, "utf8").replace(/\0/g, " ").trim();
        } catch {}
        try {
          const envRaw = readFileSync(`${HOST_PROC}/${pid}/environ`, "utf8");
          for (const pair of envRaw.split("\0")) {
            if (!pair) continue;
            const eq = pair.indexOf("=");
            if (eq > 0) environ[pair.slice(0, eq)] = pair.slice(eq + 1);
          }
        } catch {}
      }
    }

    if (!cmdline) return null;

    // Parse model from cmdline: --model <path>, or the positional arg after "serve"
    let modelPath = null;
    const modelMatch = cmdline.match(/--model\s+(\S+)/);
    if (modelMatch) modelPath = modelMatch[1];
    else {
      const serveIdx = cmdline.indexOf("serve");
      if (serveIdx >= 0) {
        const rest = cmdline.slice(serveIdx + 5).trim();
        const firstTok = rest.split(/\s+/)[0];
        if (firstTok && !firstTok.startsWith("--")) modelPath = firstTok;
      }
    }
    const modelFile = modelPath ? modelPath.split("/").pop() : null;
    // Normalize HF cache snapshots into their actual repository identity so the
    // dashboard shows the checkpoint/quant rather than only the served alias.
    const canonicalModel = normalizeModelId(modelPath);

    // Served model name (alias exposed via /v1/models)
    const servedNameMatch = cmdline.match(/--served-model-name\s+(\S+)/);
    const servedModelName = servedNameMatch ? servedNameMatch[1] : null;

    // Detect quantization
    let quantization = null;
    const quantArg = cmdline.match(/--quantization\s+(\S+)/);
    if (quantArg) {
      quantization = quantArg[1].toUpperCase();
    } else {
      // A snapshot basename is only a hash; the canonical HF repo carries the quant.
      const quantSource = canonicalModel || modelFile || "";
      if (/NVFP4/i.test(quantSource)) quantization = "NVFP4";
      else if (/FP8/i.test(quantSource)) quantization = "FP8";
      else if (/AWQ/i.test(quantSource)) quantization = "AWQ";
      else if (/GPTQ/i.test(quantSource)) quantization = "GPTQ";
    }

    // Context length from cmdline: --max-model-len <num>
    let contextLength = this.contextLength;
    const ctxMatch = cmdline.match(/--max-model-len\s+(\d+)/);
    if (ctxMatch) contextLength = parseInt(ctxMatch[1]);

    // Max lanes from cmdline: --tensor-parallel-size or -tp
    let maxLanes = null;
    const tpMatch = cmdline.match(/--tensor-parallel-size\s+(\d+)/);
    if (tpMatch) maxLanes = parseInt(tpMatch[1]);
    else {
      const tpShort = cmdline.match(/(?:^|\s)-tp\s+(\d+)/);
      if (tpShort) maxLanes = parseInt(tpShort[1]);
    }

    // Speculative decode method
    let specDecodeMethod = null;
    if (/--speculative-model/.test(cmdline) || /--speculative-config/.test(cmdline)) {
      const numSpecMatch = cmdline.match(/--num-speculative-tokens\s+(\d+)/);
      let k = numSpecMatch ? numSpecMatch[1] : null;
      if (!k) {
        // k may live inside the --speculative-config JSON, e.g. {"method":"mtp","num_speculative_tokens":3}
        const scMatch = cmdline.match(/--speculative-config\s+(\S+)/);
        if (scMatch) {
          const scTok = scMatch[1].replace(/^["']|["']$/g, "");
          try {
            const sc = JSON.parse(scTok);
            if (sc && sc.num_speculative_tokens != null) k = String(sc.num_speculative_tokens);
          } catch {}
        }
      }
      specDecodeMethod = `MTP k=${k || "?"}`;
    }

    // KV cache dtype
    let kvCacheDtype = null;
    const kvMatch = cmdline.match(/--kv-cache-dtype\s+(\S+)/);
    if (kvMatch) kvCacheDtype = kvMatch[1];
    else kvCacheDtype = "auto";

    // Prefix caching: from cache_config_info enable_prefix_caching label, else
    // fall back to the --enable-prefix-caching / --no-prefix-caching flags.
    let prefixCaching = this._vllmPrefixCaching;
    if (prefixCaching == null) {
      if (/--enable-prefix-caching/.test(cmdline)) prefixCaching = true;
      else if (/--no-prefix-caching/.test(cmdline)) prefixCaching = false;
    }
    try {
      const pid = this._findHostPid();
      if (pid) {
        const cl = readFileSync(`${HOST_PROC}/${pid}/cmdline`, "utf8").replace(/\0/g, " ");
        if (/--enable-prefix-caching/.test(cl)) prefixCaching = true;
        else if (/--no-prefix-caching/.test(cl)) prefixCaching = false;
      }
    } catch {}

    // GMU
    let gmu = null;
    const gmuMatch = cmdline.match(/--gpu-memory-utilization\s+([\d.]+)/);
    if (gmuMatch) gmu = parseFloat(gmuMatch[1]);

    // Reasoning / tool-call parsers + speculative config
    const reasoningParserMatch = cmdline.match(/--reasoning-parser\s+(\S+)/);
    const reasoningParser = reasoningParserMatch ? reasoningParserMatch[1] : null;
    const toolCallParserMatch = cmdline.match(/--tool-call-parser\s+(\S+)/);
    const toolCallParser = toolCallParserMatch ? toolCallParserMatch[1] : null;
    const speculativeConfigMatch = cmdline.match(/--speculative-config\s+(\S+)/);
    const speculativeConfig = speculativeConfigMatch ? speculativeConfigMatch[1] : null;

    // Author attribution for vLLM recipes
    const author = "@styles01";
    const authorName = "styles01";

    // Engine type
    const engineType = "vLLM";

    // Prefer the canonical checkpoint identity to its user-facing API alias.
    const modelName = canonicalModel || this.modelId || servedModelName || modelFile || null;

    // Accept ratio
    const acceptRatio = this.mtpAcceptanceRate ?? null;

    // Populate recipeMetadata so the provenance section has structured fields
    // (ownedBy, model, contextLength, parsers, etc.) even when docker is absent.
    this.recipeMetadata = {
      name: servedModelName || modelFile || this.modelId || null,
      model: servedModelName || modelFile || this.modelId || null,
      contextLength,
      ownedBy: "vllm",
      supportedParameters: [],
      quantization,
      gmu,
      maxModelLen: contextLength,
      servedModelName,
      reasoningParser,
      toolCallParser,
      speculativeConfig,
      modelPath,
    };

    return {
      engineType,
      modelName,
      containerImage,
      author,
      authorName,
      contextLength,
      maxLanes,
      specDecodeMethod,
      quantization,
      gmu,
      kvCacheDtype,
      prefixCaching,
      acceptRatio,
      uptime: this.ds4Uptime ?? null,
      servedModelName,
      reasoningParser,
      toolCallParser,
      speculativeConfig,
      modelPath,
    };
  }

  _getSnapshot() {
    const metricsLive = this.serverIsOpenAI !== null && this.authOpen !== false;
    return {
      available: metricsLive,
      backend: this.backendType,
      modelId: this.models.length > 1 ? this.models[0] : this.modelId || null,
      modelPath: this.modelPath || null,
      models: metricsLive ? [...this.models] : [],
      // Native enrichment may set a context limit; routers have no shared limit.
      contextLength: this.models.length > 1 ? null : this.contextLength,
      gpuMemoryUtilization: this.gpuMemoryUtilization,
      slotsActive: this.slotsActive,
      slotsTotal: this.slotsTotal,
      waitingSlots: this.requestsWaiting ?? this.waitingSlots ?? 0,
      generationTps: this.generationTps,
      prefillTps: this.prefillTps,
      isPrefilling: this.isPrefilling ?? false,
      prefillActive: this.prefillActive === true,
      cachedPrefillTps: this.cachedPrefillTps,
      uncachedPrefillTps: this.uncachedPrefillTps,
      avgTps: this.avgTps ?? null,
      avgTps1h: this.avgTps1h ?? null,
      prefillAvg: this.prefillAvg ?? null,
      avgTpsBuildVersion: this.avgTpsBuildVersion ?? null,
      totalOutputTokens: this.totalOutputTokens,
      totalPromptTokens: this.totalPromptTokens ?? null,
      totalCachedTokens: this.totalCachedTokens ?? null,
      kvCacheUsage: this.kvCacheUsage,
      kvCacheTokens: this.kvCacheTokens,
      kvCacheTokensAvailable: this.kvCacheTokensAvailable,
      kvCacheMemoryBytes: this.kvCacheMemoryBytes,
      kvCacheGb: this.kvCacheGb,
      weightsGb: this.weightsGb,
      kvUsedTokens: this.kvUsedTokens,
      kvPoolTokens: this.kvPoolTokens,
      remainingCtx: this.remainingCtx,
      requestsRunning: this.requestsRunning,
      requestsWaiting: this.requestsWaiting,
      ttftP95Seconds: this.ttftP95Seconds,
      ttftSeconds: this.ttftSeconds,
      preemptionsTotal: this.preemptionsTotal,
      prefixCacheHitRate: this.prefixCacheHitRate,
      e2eP95Seconds: this.e2eP95Seconds,
      itlP95Seconds: this.itlP95Seconds,
      mtpAcceptanceRate: this.mtpAcceptanceRate,
      ttft: this.ttft ?? this.ttftP95Seconds,
      e2eLatency: this.e2eLatency ?? this.e2eP95Seconds,
      genTokensPerReq: this.genTokensPerReq,
      mtpAcceptedTokens: this.mtpAcceptedTokens,
      mtpDraftedTokens: this.mtpDraftedTokens,
      perPositionAcceptance: this.perPositionAcceptance,
      aggregateDecodeTps: this.aggregateDecodeTps,
      rollingAvgE2e: this.rollingAvgE2e,
      rollingAvgTtft: this.rollingAvgTtft,
      rollingAvgTokensPerReq: this.rollingAvgTokensPerReq,
      rollingAvgTpsPerSlot: this.rollingAvgTpsPerSlot,
      posture: this._buildPosture(),
      recipeInfo: this.recipeInfo,
      recipeMetadata: this.recipeMetadata,
      // Gauge families the live engine build does not expose (panel shows
      // UNAVAILABLE for these instead of a dead 0).
      fieldsUnavailable: Array.isArray(this._vllmSeriesUnavailable)
        ? this._vllmSeriesUnavailable
        : [],
      peakAggregateTps: this.peakAggregateTps,
      perStreamHigh: this.perStreamHigh,
      perStreamLow: this.perStreamLow,
      perStreamAvg: this.perStreamAvg,
      totalTokensDecoded: this.totalTokensDecoded,
      dsparkAcceptRatio: this.dsparkAcceptRatio,
      banksLive: this.banksLive,
      banksTotal: this.banksTotal,
      kvPagesResident: this.kvPagesResident,
      prefillCached: this.prefillCached,
      prefillComputed: this.prefillComputed,
      specDrafts: this.specDrafts,
      specHits: this.specHits,
      warmRecords: this.warmRecords,
      specQuench: this.specQuench,
      tokPerStep: this.tokPerStep,
      decodeSteps: this.decodeSteps,
      derivedArtifacts: this.derivedArtifacts,
      derivedArtifactBytes: this.derivedArtifactBytes,
      ds4Uptime: this.ds4Uptime,
      engineUptimeSeconds: this.engineUptimeSeconds,
      admitsCold: this.admitsCold,
      admitsWarm: this.admitsWarm,
      admitsFork: this.admitsFork,
      admitsPartialFork: this.admitsPartialFork,
      admitsPartialTruncate: this.admitsPartialTruncate,
      requestsStarted: this.requestsStarted,
      requestsCompleted: this.requestsCompleted,
      requestsFailed: this.requestsFailed,
      requestsInflight: this.requestsInflight,
      activeContext: this.activeContext,
      activeContextTs: this.activeContextTs,
      contextUsedBytes: this.contextUsedBytes,
      reasoningEffort: this.reasoningEffort,
      reasoningEffortTs: this.reasoningEffortTs,
      // llama.cpp expanded surface
      promptTokens: this.promptTokens ?? this.totalPromptTokens,
      promptTokensProcessed: this.promptTokensProcessed ?? (
        this.totalPromptTokens != null && this.totalCachedTokens != null
          ? Math.max(0, Math.round(this.totalPromptTokens - this.totalCachedTokens))
          : null
      ),
      promptTokensCache: this.promptTokensCache ?? this.totalCachedTokens,
      cacheHitRatio: this.cacheHitRatio ?? this.prefixCacheHitRate,
      nCtx: this.nCtx,
      isProcessing: this.isProcessing,
      nRemain: this.nRemain,
      nDecoded: this.nDecoded ?? this.totalOutputTokens,
      samplingParams: this.samplingParams,
      speculativeTypes: this.speculativeTypes,
      reasoningFormat: this.reasoningFormat,
      chatFormat: this.chatFormat,
      samplers: this.samplers,
      specAcceptanceRate: this.specAcceptanceRate,
      specAcceptedTokens: this.specAcceptedTokens,
      specGeneratedTokens: this.specGeneratedTokens,
      specMeanLen: this.specMeanLen,
      error: this.error,
    };

    // DS4 fields (always include — null for non-ds4 backends)
    snap.ds4Uptime = this.ds4Uptime;
    snap.peakAggregateTps = this.peakAggregateTps;
    snap.perStreamHigh = this.perStreamHigh;
    snap.perStreamLow = this.perStreamLow;
    snap.perStreamAvg = this.perStreamAvg;
    snap.totalTokensDecoded = this.totalTokensDecoded;
    snap.dsparkAcceptRatio = this.dsparkAcceptRatio;
    snap.banksLive = this.banksLive;
    snap.banksTotal = this.banksTotal;
    snap.kvPagesResident = this.kvPagesResident;
    snap.prefillCached = this.prefillCached;
    snap.prefillComputed = this.prefillComputed;
    snap.specDrafts = this.specDrafts;
    snap.specHits = this.specHits;
    snap.specQuench = this.specQuench;
    snap.warmRecords = this.warmRecords;
    snap.derivedArtifacts = this.derivedArtifacts;
    snap.derivedArtifactBytes = this.derivedArtifactBytes;
    snap.requestsStarted = this.requestsStarted;
    snap.requestsCompleted = this.requestsCompleted;
    snap.requestsFailed = this.requestsFailed;
    snap.requestsRefusedDeepSerial = this.requestsRefusedDeepSerial;
    snap.requestsInflight = this.requestsInflight;
    snap.requestsSerial = this.requestsSerial;
    snap.contAdmitRejects = this.contAdmitRejects;
    snap.contBatchFailures = this.contBatchFailures;
    snap.graphFitRefusals = this.graphFitRefusals;
    snap.admitsCold = this.admitsCold;
    snap.admitsWarm = this.admitsWarm;
    snap.admitsFork = this.admitsFork;
    snap.admitsPartialFork = this.admitsPartialFork;
    snap.admitsPartialTruncate = this.admitsPartialTruncate;
    snap.decodeSteps = this.decodeSteps;
    snap.tokPerStep = this.tokPerStep;
    snap.engineUptimeSeconds = this.engineUptimeSeconds;
    snap.recipeMetadata = this.recipeMetadata;
    snap.recipeInfo = this.recipeInfo;

    return snap;
  }

  _defaultLlm() {
    const snap = {
      available: false,
      backend: this.backendType,
      modelId: null,
      modelPath: null,
      models: [],
      contextLength: null,
      gpuMemoryUtilization: null,
      slotsActive: 0,
      slotsTotal: 0,
      generationTps: 0,
      prefillTps: 0,
      prefillActive: false,
      cachedPrefillTps: null,
      uncachedPrefillTps: null,
      ttftSeconds: null,
      totalOutputTokens: 0,
      totalCachedTokens: null,
      kvCacheUsage: null,
      kvCacheTokens: null,
      kvCacheTokensAvailable: null,
      kvCacheMemoryBytes: null,
      kvCacheGb: null,
      weightsGb: null,
      requestsRunning: null,
      requestsWaiting: null,
      ttftP95Seconds: null,
      preemptionsTotal: null,
      prefixCacheHitRate: null,
      e2eP95Seconds: null,
      itlP95Seconds: null,
      mtpAcceptanceRate: null,
      ttft: null,
      e2eLatency: null,
      genTokensPerReq: null,
      mtpAcceptedTokens: null,
      mtpDraftedTokens: null,
      perPositionAcceptance: null,
      aggregateDecodeTps: null,
      rollingAvgE2e: null,
      rollingAvgTtft: null,
      rollingAvgTokensPerReq: null,
      rollingAvgTpsPerSlot: null,
      posture: this._buildPosture(),
      error: this.error,
      ds4Uptime: null,
      peakAggregateTps: 0,
      perStreamHigh: null,
      perStreamLow: null,
      perStreamAvg: null,
      totalTokensDecoded: null,
      dsparkAcceptRatio: null,
      banksLive: null,
      banksTotal: null,
      kvPagesResident: null,
      prefillCached: null,
      prefillComputed: null,
      specDrafts: null,
      specHits: null,
      specQuench: null,
      warmRecords: null,
      derivedArtifacts: null,
      derivedArtifactBytes: null,
      requestsStarted: null,
      requestsCompleted: null,
      requestsFailed: null,
      requestsRefusedDeepSerial: null,
      requestsInflight: null,
      requestsSerial: null,
      contAdmitRejects: null,
      contBatchFailures: null,
      graphFitRefusals: null,
      admitsCold: null,
      admitsWarm: null,
      admitsFork: null,
      admitsPartialFork: null,
      admitsPartialTruncate: null,
      decodeSteps: null,
      tokPerStep: null,
      recipeMetadata: null,
      recipeInfo: null,
      reasoningEffort: null,
      reasoningEffortTs: null,
      activeContext: null,
      activeContextTs: null,
      contextUsedBytes: null,
      waitingSlots: 0,
      // llama.cpp expanded surface
      promptTokens: null,
      promptTokensProcessed: null,
      promptTokensCache: null,
      cacheHitRatio: null,
      nCtx: null,
      isProcessing: false,
      nRemain: null,
      nDecoded: null,
      samplingParams: null,
      speculativeTypes: null,
      reasoningFormat: null,
      chatFormat: null,
      samplers: null,
      specAcceptanceRate: null,
      specAcceptedTokens: null,
      specGeneratedTokens: null,
      specMeanLen: null,
    };
    return snap;
  }

  // ─── HTTP helpers ────────────────────────────────────────
  _apiKey() {
    const keys = this.spark?.llmApiKeys;
    if (!keys || typeof keys !== "object") return null;
    const raw = keys[String(this.port)] ?? keys[this.port];
    const key = raw != null ? String(raw).trim() : "";
    return key || null;
  }

  async _fetch(url, init) {
    const headers = {};
    const apiKey = this._apiKey();
    if (apiKey) headers.Authorization = `Bearer ${apiKey}`;
    return fetch(url, { signal: AbortSignal.timeout(LLM_PROBE_TIMEOUT_MS), headers, ...(init || {}) });
  }
}