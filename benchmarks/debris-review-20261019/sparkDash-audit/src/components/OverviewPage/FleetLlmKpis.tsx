import { useMemo } from "react";
import type { LlmMetrics, SparkSnapshot } from "../../api/types";
import { formatTokensCompact } from "../../shared/tokenFormat";

/** Compact number with unit for token magnitudes (e.g. 19.4M). */
function tok(v: number | null | undefined): string {
  if (v == null || !Number.isFinite(v)) return "—";
  return formatTokensCompact(v);
}

function pct(v: number | null | undefined): string {
  if (v == null || !Number.isFinite(v)) return "—";
  return `${Math.round(v * 100)}%`;
}

function secs(v: number | null | undefined): string {
  if (v == null || !Number.isFinite(v)) return "—";
  if (v >= 100) return `${v.toFixed(0)}s`;
  if (v >= 10) return `${v.toFixed(1)}s`;
  return `${v.toFixed(2)}s`;
}

function uptime(v: number | null | undefined): string {
  if (v == null || !Number.isFinite(v) || v <= 0) return "—";
  const h = Math.floor(v / 3600);
  const m = Math.floor((v % 3600) / 60);
  if (h >= 48) return `${(h / 24).toFixed(1)}d`;
  if (h > 0) return `${h}h ${m}m`;
  return `${m}m`;
}

function Tile({
  label,
  value,
  sub,
}: {
  label: string;
  value: string;
  sub?: string;
}) {
  return (
    <div className="panel ov-kpi ov-llmkpi">
      <div className="ov-kpi__row">
        <span className="eyebrow">{label}</span>
      </div>
      <div className="big-num">
        {value}
        {sub ? <small>{sub}</small> : null}
      </div>
      <div className="ov-kpi__foot" />
    </div>
  );
}

export interface LlmFleetRow {
  sparkId: string;
  sparkName: string;
  llm: LlmMetrics;
}

/** Collect every LLM endpoint row from the fleet snapshot (pure). */
export function collectLlmRows(sparks: readonly SparkSnapshot[]): LlmFleetRow[] {
  const rows: LlmFleetRow[] = [];
  for (const s of sparks) {
    if (!s.online || !Array.isArray(s.metrics?.llm)) continue;
    for (const llm of s.metrics.llm) {
      rows.push({
        sparkId: s.id,
        sparkName: s.name,
        llm,
      });
    }
  }
  return rows;
}

/** The LLM stats strip for the Overview (homepage / workspace overview).
 *  Mirrors the spark page's live LLM numbers: decode + peak, TTFT/E2E,
 *  context and KV occupancy, spec decode and sampler cache stats. */
export function FleetLlmKpis({ sparks }: { sparks: SparkSnapshot[] }) {
  const rows = useMemo(() => collectLlmRows(sparks), [sparks]);
  const primary = rows.find((r) => r.llm.available) ?? rows[0] ?? null;
  const decode = rows.reduce((n, r) => n + (r.llm.available ? r.llm.generationTps || 0 : 0), 0);
  const peak = rows.reduce((n, r) => Math.max(n, r.llm.peakAggregateTps || 0), 0);
  const slots = rows.reduce((n, r) => n + (r.llm.available ? r.llm.slotsActive || 0 : 0), 0);
  if (!primary) return null;
  const llm = primary.llm;
  const ctxUsed = llm.kvUsedTokens;
  const ctxPool = llm.kvPoolTokens ?? llm.kvCacheTokens;
  const modelShort = (llm.modelId || "").replace(/-EXL3.*$/, "").slice(0, 26) || "LLM";
  return (
    <div className="panel ov-llmstats">
      <div className="ov-llmstats__head">
        <span className="eyebrow">LLM stats</span>
        <span className="ov-llmstats__model mono">
          {modelShort}
          {llm.contextLength ? ` · ctx ${formatTokensCompact(llm.contextLength)}` : ""}
          {rows.length > 1 ? ` · ${primary.sparkName}` : ""}
        </span>
      </div>
      <div className="ov-kpis ov-kpis--llm">
        <Tile label="Decode" value={decode.toFixed(1)} sub="tok/s" />
        <Tile label="Peak" value={peak ? peak.toFixed(1) : "0"} sub="tok/s" />
        <Tile label="TTFT" value={secs(llm.ttft ?? llm.ttftSeconds)} />
        <Tile label="E2E" value={secs(llm.e2eLatency)} />
        <Tile
          label="Ctx used"
          value={tok(ctxUsed ?? 0)}
          sub={ctxPool ? ` / ${tok(ctxPool)}` : undefined}
        />
        <Tile label="KV cache" value={pct(llm.kvCacheUsage ?? 0)} />
        <Tile label="Spec accept" value={pct(llm.specAcceptanceRate ?? llm.mtpAcceptanceRate)} />
        <Tile label="Cache hit" value={pct(llm.cacheHitRatio)} />
        <Tile label="Slots" value={String(slots)} />
        <Tile label="Uptime" value={uptime(llm.engineUptimeSeconds)} />
        <Tile label="Prompt cached" value={tok(llm.promptTokensCache ?? llm.totalCachedTokens)} />
        <Tile label="Prefill tok/s" value={(llm.prefillTps || 0).toFixed(1)} sub="tok/s" />
      </div>
    </div>
  );
}
