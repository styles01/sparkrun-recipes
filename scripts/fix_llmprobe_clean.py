with open("server/collectors/LlmProbe.js") as f:
    content = f.read()

# 1. Fix baseUrl to use localhost for local sparks (2 places)
content = content.replace(
    "this.baseUrl = `http://${spark.lanIp}:${port}`;",
    "this.baseUrl = `http://${spark.isLocal ? (process.env.LLM_HOST || \"localhost\") : spark.lanIp}:${port}`;"
)
content = content.replace(
    "this.baseUrl = `http://${this.spark.lanIp}:${this.port}`;",
    "this.baseUrl = `http://${this.spark.isLocal ? (process.env.LLM_HOST || \"localhost\") : this.spark.lanIp}:${this.port}`;"
)

# 2. Add VllmMetricsParser import + init
if "VllmMetricsParser" not in content:
    content = content.replace(
        "class LlmProbe {",
        "const VllmMetricsParser = require(\"./VllmMetricsParser\");\n\nclass LlmProbe {"
    )
    content = content.replace(
        "this.lastTokenCounts",
        "this.vllmParser = new VllmMetricsParser();\n    this.vllmTelemetry = null;\n    this.lastTokenCounts"
    )
    content = content.replace(
        "this.slotState.clear();",
        "this.slotState.clear();\n    this.vllmTelemetry = null;"
    )

# 3. Add expanded telemetry fields to snapshot
old_snap = "      error: this.error,"
new_snap = """      error: this.error,
      // Expanded telemetry from VllmMetricsParser
      runningSlots: this.vllmTelemetry?.runningSlots ?? this.slotsActive,
      waitingSlots: this.vllmTelemetry?.waitingSlots ?? null,
      kvCacheUsage: this.vllmTelemetry?.kvCacheUsage ?? this.kvCacheUsage,
      ttft: this.vllmTelemetry?.ttft ?? null,
      e2eLatency: this.vllmTelemetry?.e2eRequestLatency ?? null,
      interTokenLatency: this.vllmTelemetry?.interTokenLatency ?? null,
      mtpAcceptanceRate: this.vllmTelemetry?.specAcceptanceRate ?? this.mtpAcceptanceRate,
      mtpAcceptedTokens: this.vllmTelemetry?.specAcceptedTokens ?? null,
      mtpDraftedTokens: this.vllmTelemetry?.specDraftedTokens ?? null,
      prefixCacheHitRate: this.vllmTelemetry?.prefixCacheHitRate ?? this.prefixCacheHitRate,
      perPositionAcceptance: this.vllmTelemetry?.specPerPositionAcceptance ?? [],
      rollingAvgE2e: this.vllmTelemetry?.rolling?.avgE2eLatency ?? null,
      rollingAvgTtft: this.vllmTelemetry?.rolling?.avgTtft ?? null,
      rollingAvgTokensPerReq: this.vllmTelemetry?.rolling?.avgTokensPerRequest ?? null,
      rollingAvgTpsPerSlot: this.vllmTelemetry?.rolling?.avgTpsPerSlot ?? null,
      slots: this.vllmTelemetry?.slots ?? [],
      vllmTelemetry: this.vllmTelemetry,"""
content = content.replace(old_snap, new_snap, 1)

with open("server/collectors/LlmProbe.js", "w") as f:
    f.write(content)

print("LlmProbe.js fixed - upstream base + our 3 changes applied cleanly")