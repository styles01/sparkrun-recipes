import { benchId } from "../../constants";
import { BenchIcon } from "../bench/BenchIcon";
import { useState, useEffect, useRef, useCallback, useMemo } from "react";
import type { LlmMetrics, LlmBenchTarget } from "../../api/types";
import { setLlmApiKey, updateLlmPort, updateLlmPorts } from "../../api/client";
import { TrendLine } from "../ui/TrendLine";
import { Panel } from "../ui/Panel";
import { Tag, type TagTone } from "../ui/Tag";
import { BotIcon, ExpandIcon, FlaskIcon, GearIcon, InfoIcon, ServerIcon } from "../ui/icons";
import {
  useMetricsHistory,
  useMetricsHistoryTail,
  avgPositive,
} from "../../hooks/metricsStore";
import { BenchmarkDialog } from "./BenchmarkDialog";
import { PrefillBenchDialog } from "./PrefillBenchDialog";
import { QualityBenchDialog } from "./QualityBenchDialog";
import { LlmDailyChart } from "./LlmDailyChart";
import { LlmTokenTotals } from "./LlmTokenTotals";
import { ENGINE_GENERATED_LABEL, ENGINE_GENERATED_TITLE } from "./tokenTotalsCopy";
import { parseLlmTargetInput } from "../../shared/llmTarget.js";
import { backendLabel } from "../../shared/llmBackends.js";
import { TelemetryChart, type ChartSeries } from "../ui/TelemetryChart";
import { LlmTrendChart } from "./LlmTrendChart";
import type { BenchKind } from "./BenchSwitcher";
import { engineStateLabel } from "./llmEngineState";
import { idleLabel, isLlmIdle } from "../../shared/llmIdle";

interface LlmPanelProps {
  llm: LlmMetrics | null;
  sparkId: string;
  /** Unit display name — lands on the benchmark share card. */
  sparkName?: string;
  llmPort: number;
  llmPorts?: number[];
  hasApiKey?: boolean;
  /** Show "Copy image" in the benchmark dialogs (Settings, off by default). */
  shareImage?: boolean;
  onRemovePort?: (port: number) => void;
  className?: string;
}

const VLLM_METRIC_INFO = {
  kvCache:
    "Free tokens left in the engine’s KV-cache pool, over the allocated pool (tokens and memory). The pool is shared across concurrent requests and can be larger than one request’s max context. High usage (≥80% full) means little room for new or long contexts and often leads to queuing or preemptions.",
  requests:
    "Run = requests actively generating on the GPU. Wait = accepted but not yet scheduled (capacity or constraints). Growing wait with high KV cache usually means the server is overloaded.",
  ttftMean:
    "Mean time-to-first-token reported by FreeToken (a mean, not a percentile): how long requests wait until the first output token.",
  ttftP95:
    "95th percentile time-to-first-token from the engine’s request history: how long “slow” requests wait until the first output token. Spikes mean queueing, long prefills, or cold paths—not average decode speed.",
  preempts:
    "Cumulative times the engine paused a running request to free KV cache for others. Rising under load signals memory pressure; zero is normal when the server is comfortable.",
  prefixCache:
    "Lifetime fraction of prefix-cache lookups that hit (hits ÷ queries). Higher means more prompt reuse and less prefill work; — when the series is missing or unused.",
  e2eP95:
    "95th percentile end-to-end request latency from the engine’s request history: arrival until the request finishes. Includes queue wait, prefill, and decode—not just token generation speed.",
  itlP95:
    "95th percentile inter-token latency (time between successive output tokens) from the engine’s request history. Spikes mean decode stalls or contention; lower is smoother streaming.",
  mtpAccept:
    "Lifetime speculative / MTP acceptance rate (accepted draft tokens ÷ drafted tokens). Higher means speculative decoding is paying off; — when speculation is off or unused.",
} as const;

/** Compact token counts for the KV pool tile (883552 → 884k). */
function formatKvTokens(n: number): string {
  const abs = Math.abs(n);
  if (abs >= 1_000_000) {
    const v = Math.round((n / 1_000_000) * 10) / 10;
    return `${Number.isInteger(v) ? v.toFixed(0) : v.toFixed(1)}M`;
  }
  if (abs >= 1000) {
    const v = abs >= 10_000 ? Math.round(n / 1000) : Math.round((n / 1000) * 10) / 10;
    return `${Number.isInteger(v) ? v.toFixed(0) : v.toFixed(1)}k`;
  }
  return Math.round(n).toLocaleString();
}

function formatKvBytes(n: number): string {
  const gib = n / 1024 ** 3;
  if (gib >= 10) return `${Math.round(gib)} GB`;
  if (gib >= 1) {
    const v = Math.round(gib * 10) / 10;
    return `${Number.isInteger(v) ? v.toFixed(0) : v.toFixed(1)} GB`;
  }
  return `${Math.round(n / 1024 ** 2)} MB`;
}

const REMOTE_STORAGE_KEY = "sparkdash.remote-bench-target";

function readStoredRemote(): { host: string; port: string; tls: boolean } {
  try {
    const raw = localStorage.getItem(REMOTE_STORAGE_KEY);
    if (!raw) return { host: "", port: "443", tls: true };
    const v = JSON.parse(raw) as { host?: string; port?: number; tls?: boolean };
    return {
      host: typeof v.host === "string" ? v.host : "",
      port: v.port != null ? String(v.port) : "443",
      tls: v.tls !== false,
    };
  } catch {
    return { host: "", port: "443", tls: true };
  }
}

/** Decode / Prefill / Quality / Showcase launchers — shown even when the live probe is empty
 *  (remote loopback-bound servers can still be benched via SSH tunnel). */
function LlmLaunchers({
  sparkId,
  llmPort,
  modelId,
  onLaunch,
  onRemoteLaunch,
}: {
  sparkId: string;
  llmPort: number;
  modelId?: string | null;
  onLaunch: (kind: BenchKind) => void;
  onRemoteLaunch: (kind: BenchKind, target: LlmBenchTarget) => void;
}) {
  const [remoteOpen, setRemoteOpen] = useState(false);
  const [hostDraft, setHostDraft] = useState(() => readStoredRemote().host);
  const [portDraft, setPortDraft] = useState(() => readStoredRemote().port);
  const [tls, setTls] = useState(() => readStoredRemote().tls);
  const [remoteError, setRemoteError] = useState<string | null>(null);

  const persist = (t: LlmBenchTarget) => {
    try {
      localStorage.setItem(REMOTE_STORAGE_KEY, JSON.stringify(t));
    } catch {
      /* ignore */
    }
  };

  const applyHostBlur = () => {
    if (!hostDraft.trim()) return;
    try {
      const p = parseLlmTargetInput(hostDraft, portDraft, tls);
      setHostDraft(p.host);
      setPortDraft(String(p.port));
      setTls(p.tls);
      setRemoteError(null);
    } catch {
      /* leave as typed until Run */
    }
  };

  const launchRemote = (kind: BenchKind) => {
    try {
      const p = parseLlmTargetInput(hostDraft, portDraft, tls);
      persist(p);
      setHostDraft(p.host);
      setPortDraft(String(p.port));
      setTls(p.tls);
      setRemoteError(null);
      onRemoteLaunch(kind, p);
    } catch (err: unknown) {
      setRemoteError(err instanceof Error ? err.message : String(err));
    }
  };

  const localHint =
    "Runs against this Spark’s LLM. Remote units use LAN HTTP, or an SSH tunnel to loopback if the server only listens on 127.0.0.1.";

  return (
    <div className="sp-launchers">
      <div className="sp-btns">
        {(
          [
            ["decode", "Decode", `Decode benchmark: generation speed at rising concurrency, on this Spark. ${localHint}`],
            ["prefill", "Prefill", `Prefill benchmark: prompt processing speed and time to first token, on this Spark. ${localHint}`],
            ["quality", "Quality", "Quality suite (QA, reasoning, arithmetic, state tracking, GSM8K, MMLU) against this port's model. Compare runs across models, quantizations and KV-cache formats."],
            ["tool-eval", "Tool Eval Bench", "Benchmark this model's tool calling."],
          ] as const
        ).map(([type, label, title]) => (
          <button
            key={type}
            type="button"
            // Opens the benchmark's own page, already set to this Spark.
            onClick={() => window.dispatchEvent(new CustomEvent("sparkdash:navigate", { detail: { id: benchId(type), spark: sparkId } }))}
            className="btn btn--sm"
            title={title}
          >
            <BenchIcon id={type} className="h-3.5 w-3.5" />
            {label}
          </button>
        ))}
        <button
          type="button"
          onClick={() => {
            const params = new URLSearchParams();
            if (llmPort) params.set("port", String(llmPort));
            if (modelId) params.set("model", modelId);
            const q = params.toString() ? `?${params.toString()}` : "";
            window.open(`/showcase/${encodeURIComponent(sparkId)}${q}`, "_blank", "noopener,noreferrer");
          }}
          className="btn btn--sm"
        >
          <ExpandIcon className="h-3.5 w-3.5" />
          Showcase
        </button>
        <button
          type="button"
          onClick={() => {
            setRemoteOpen((v) => !v);
            setRemoteError(null);
          }}
          className={`btn btn--sm btn--ghost sp-btns__end ${remoteOpen ? "is-on" : ""}`}
          aria-expanded={remoteOpen}
          title="On-demand bench against a typed host (HTTPS Tailscale, LAN IP, …). Not probed until you run."
        >
          <ServerIcon className="h-3.5 w-3.5" />
          Remote
        </button>
      </div>
      {remoteOpen && (
        <div className="sp-remote">
          <p className="sp-hint">
            On-demand endpoint. Paste a URL or type host + port — nothing is probed until you run.
          </p>
          <label className="sp-field">
            <span className="eyebrow">Host</span>
            <input
              type="text"
              value={hostDraft}
              onChange={(e) => setHostDraft(e.target.value)}
              onBlur={applyHostBlur}
              placeholder="https://name.tailxxxxx.ts.net/v1/models"
              spellCheck={false}
              autoCapitalize="off"
              autoCorrect="off"
              className="sp-input"
            />
          </label>
          <div className="sp-field-row">
            <label className="sp-field sp-field--grow">
              <span className="eyebrow">Port</span>
              <input
                type="number"
                min={1}
                max={65535}
                inputMode="numeric"
                value={portDraft}
                onChange={(e) => setPortDraft(e.target.value)}
                className="sp-input"
              />
            </label>
            <label className="sp-check">
              <input
                type="checkbox"
                checked={tls}
                onChange={(e) => {
                  const next = e.target.checked;
                  setTls(next);
                  if (next && portDraft === "8888") setPortDraft("443");
                  if (!next && portDraft === "443") setPortDraft("8888");
                }}
              />
              HTTPS
            </label>
          </div>
          {remoteError && <p className="sp-error">{remoteError}</p>}
          <div className="sp-btns">
            <button type="button" onClick={() => launchRemote("decode")} className="btn btn--sm">
              Decode
            </button>
            <button type="button" onClick={() => launchRemote("prefill")} className="btn btn--sm">
              Prefill
            </button>
            <button type="button" onClick={() => launchRemote("quality")} className="btn btn--sm">
              Quality
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

const POSTURE_TONE: Record<NonNullable<LlmMetrics["posture"]>["level"], TagTone> = {
  ok: "good",
  warn: "warn",
  danger: "bad",
};

/** Exposure / auth posture from the unauthenticated probe (issue #17). */
function PostureBadge({ posture }: { posture: NonNullable<LlmMetrics["posture"]> }) {
  return (
    <Tag tone={POSTURE_TONE[posture.level]} title={posture.detail}>
      {posture.label}
    </Tag>
  );
}

/** Small (i) next to a metric label; one open tooltip at a time. */
function MetricInfoTip({
  id,
  label,
  text,
  openId,
  setOpenId,
  /** Anchor tooltip to the right so edge columns don’t clip off-screen */
  align = "left",
}: {
  id: string;
  label: string;
  text: string;
  openId: string | null;
  setOpenId: (id: string | null) => void;
  align?: "left" | "right";
}) {
  const open = openId === id;
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const clearTimer = useCallback(() => {
    if (timer.current != null) {
      clearTimeout(timer.current);
      timer.current = null;
    }
  }, []);

  const scheduleClose = useCallback(() => {
    clearTimer();
    timer.current = setTimeout(() => setOpenId(null), 2000);
  }, [clearTimer, setOpenId]);

  useEffect(() => () => clearTimer(), [clearTimer]);

  return (
    <div className="sp-tile__label">
      <span>{label}</span>
      <button
        type="button"
        onClick={() => {
          if (open) {
            clearTimer();
            setOpenId(null);
          } else {
            setOpenId(id);
            scheduleClose();
          }
        }}
        onMouseEnter={() => {
          clearTimer();
          setOpenId(id);
        }}
        onMouseLeave={scheduleClose}
        className="relative cursor-pointer opacity-60 hover:opacity-100"
        aria-label={`${label} info`}
      >
        <InfoIcon className="h-2.5 w-2.5" />
        {open && (
          <div
            onMouseEnter={clearTimer}
            onMouseLeave={scheduleClose}
            className={`absolute top-full z-20 mt-1 w-52 max-w-[min(13rem,calc(100vw-1.5rem))] rounded-xl border border-border-strong bg-surface-elevated px-3 py-2 text-left text-[11px] font-normal normal-case leading-snug text-text shadow-lg ${
              align === "right" ? "right-0 left-auto" : "left-0 right-auto"
            }`}
          >
            {text}
          </div>
        )}
      </button>
    </div>
  );
}

function arrMean(arr: number[]): number | null {
  const valid = arr.filter((v) => Number.isFinite(v));
  if (valid.length === 0) return null;
  return valid.reduce((a, b) => a + b, 0) / valid.length;
}

function fmtNum(v: number | undefined | null, digits = 1, suffix = ""): string {
  if (v == null || !Number.isFinite(v)) return "\u2014";
  return `${v.toFixed(digits)}${suffix}`;
}

function fmtInt(v: number | undefined | null): string {
  if (v == null || !Number.isFinite(v)) return "\u2014";
  return Math.round(v).toLocaleString();
}

function pct(v: number | undefined | null, digits = 0): string {
  if (v == null || !Number.isFinite(v)) return "\u2014";
  return `${(v * 100).toFixed(digits)}%`;
}

/* ── SYNTHWAVE COLOR SPECTRUM ── */

function tpsColor(tps: number): string {
  if (!Number.isFinite(tps) || tps <= 0) return "var(--color-muted)";
  if (tps >= 40) return "var(--color-success)";
  if (tps >= 15) return "var(--color-accent)";
  return "var(--color-warning)";
}

function mtpColor(rate: number | undefined | null): string {
  if (rate == null) return "var(--color-muted)";
  const p = rate * 100;
  if (p > 70) return "var(--color-success)";
  if (p >= 50) return "var(--color-accent)";
  return "var(--color-warning)";
}

function latencyColor(seconds: number): string {
  if (!Number.isFinite(seconds)) return "var(--color-muted)";
  if (seconds <= 0.5) return "var(--color-success)";
  if (seconds >= 3.0) return "var(--color-danger)";
  return "var(--color-accent)";
}

function fmtUptime(seconds: number | null | undefined): string {
  if (seconds == null || !Number.isFinite(seconds)) return "\u2014";
  const s = Math.floor(seconds);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  if (h > 0) return `${h}h ${m}m`;
  if (m > 0) return `${m}m ${sec}s`;
  return `${sec}s`;
}

function pushSample(arr: number[], v: number, max = HISTORY): number[] {
  const next = arr.length >= max ? arr.slice(arr.length - max + 1) : arr.slice();
  next.push(Number.isFinite(v) ? v : 0);
  return next;
}

/* ── SPARKLINE ── */

function Sparkline({ data, color, height = 24 }: { data: number[]; color: string; height?: number }) {
  const valid = data.filter((v) => Number.isFinite(v));
  if (valid.length < 2) return <div style={{ height }} />;
  const min = Math.min(...valid);
  const max = Math.max(...valid);
  const range = max - min || 1;
  const w = 100;
  const h = height;
  const n = valid.length;
  const points = valid.map((v, i) => {
    const x = (i / (n - 1)) * w;
    const y = h - ((v - min) / range) * h;
    return `${x},${y}`;
  }).join(" ");
  return (
    <svg viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" className="llm-sparkline" style={{ width: "100%", height }}>
      <polygon points={`0,${h} ${points} ${w},${h}`} fill={`${color}20`} />
      <polyline points={points} fill="none" stroke={color} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" vectorEffect="non-scaling-stroke" />
    </svg>
  );
}

/* ── ARC GAUGE ── */

function ArcGauge({
  value,
  max,
  label,
  displayValue,
  sub,
  color,
  size = 120,
}: {
  value: number;
  max: number;
  label: string;
  displayValue: string;
  sub?: string;
  color: string;
  size?: number;
}) {
  const r = size * 0.42;
  const strokeWidth = size * 0.075;
  const cx = size / 2;
  const cy = size / 2 + size * 0.04;
  const pctVal = max > 0 ? Math.max(0, Math.min(1, value / max)) : 0;

  const polarToCartesian = (angleDeg: number) => {
    const rad = (angleDeg * Math.PI) / 180;
    return { x: cx + r * Math.cos(rad), y: cy + r * Math.sin(rad) };
  };

  const describeArc = (angleStart: number, angleEnd: number) => {
    const start = polarToCartesian(angleStart);
    const end = polarToCartesian(angleEnd);
    const largeArc = angleEnd - angleStart > 180 ? 1 : 0;
    return `M ${start.x} ${start.y} A ${r} ${r} 0 ${largeArc} 1 ${end.x} ${end.y}`;
  };

  const startAngle = 135;
  const sweep = 270;
  const trackArc = describeArc(startAngle, startAngle + sweep);
  const currentAngle = startAngle + sweep * pctVal;
  const valueArc = describeArc(startAngle, currentAngle);

  return (
    <div className="llm-arc-gauge" style={{ width: size, height: size }}>
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        <path d={trackArc} fill="none" stroke="var(--color-border)" strokeWidth={strokeWidth} strokeLinecap="round" />
        <path d={valueArc} fill="none" stroke={color} strokeWidth={strokeWidth} strokeLinecap="round" style={{ transition: "d 0.4s ease, stroke 0.4s ease" }} />
        <text x={cx} y={cy - size * 0.02} textAnchor="middle" fill="var(--color-text-strong)" fontSize={Math.max(18, size * 0.22)} fontWeight={800} className="font-tabular">
          {displayValue}
        </text>
        {sub && (
          <text x={cx} y={cy + size * 0.14} textAnchor="middle" fill="var(--color-muted)" fontSize={Math.max(9, size * 0.085)} className="font-tabular">
            {sub}
          </text>
        )}
      </svg>
      <div className="llm-arc-gauge-label">{label}</div>
    </div>
  );
}

/* ── FULL CIRCLE GAUGE ── */

function FullCircleGauge({
  value,
  max,
  displayValue,
  sub,
  color,
  size = 120,
}: {
  value: number;
  max: number;
  displayValue: string;
  sub?: string;
  color: string;
  size?: number;
}) {
  const r = size * 0.42;
  const strokeWidth = size * 0.075;
  const cx = size / 2;
  const cy = size / 2;
  const pctVal = max > 0 ? Math.max(0, Math.min(1, value / max)) : 0;
  const circumference = 2 * Math.PI * r;
  const dashOffset = circumference * (1 - pctVal);

  return (
    <div className="llm-arc-gauge" style={{ width: size, height: size }}>
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        <circle cx={cx} cy={cy} r={r} fill="none" stroke="var(--color-border)" strokeWidth={strokeWidth} />
        <circle cx={cx} cy={cy} r={r} fill="none" stroke={color} strokeWidth={strokeWidth}
          strokeLinecap="round"
          strokeDasharray={`${circumference} ${circumference}`}
          strokeDashoffset={dashOffset}
          style={{ transition: "stroke-dashoffset 0.4s ease, stroke 0.4s ease", transform: "rotate(-90deg)", transformOrigin: `${cx}px ${cy}px` }}
        />
        <text x={cx} y={cy - size * 0.02} textAnchor="middle" fill="var(--color-text-strong)" fontSize={Math.max(18, size * 0.22)} fontWeight={800} className="font-tabular">
          {displayValue}
        </text>
        {sub && (
          <text x={cx} y={cy + size * 0.14} textAnchor="middle" fill="var(--color-muted)" fontSize={Math.max(9, size * 0.085)} className="font-tabular">
            {sub}
          </text>
        )}
      </svg>
    </div>
  );
}

/* ── GAUGE CARD ── */

function GaugeCard({
  label,
  children,
  sparkData,
  sparkColor,
}: {
  label: string;
  children: React.ReactNode;
  sparkData?: number[];
  sparkColor?: string;
}) {
  return (
    <div className="llm-gauge-card">
      <div className="llm-gauge-card-label">{label}</div>
      <div className="llm-gauge-card-body">{children}</div>
      {sparkData && sparkColor && <Sparkline data={sparkData} color={sparkColor} height={22} />}
    </div>
  );
}

/* ── STAT SPARK CARD ── */

function StatSparkCard({
  label,
  value,
  sub,
  color,
  sparkData,
}: {
  label: string;
  value: string;
  sub?: string;
  color?: string;
  sparkData: number[];
}) {
  return (
    <div className="llm-stat-spark-card">
      <div className="llm-stat-spark-label">{label}</div>
      <div className="llm-stat-spark-value font-tabular" style={color ? { color } : undefined}>{value}</div>
      {sub && <div className="llm-stat-spark-sub font-tabular">{sub}</div>}
      <div className="llm-stat-sparkline-wrap"><Sparkline data={sparkData} color={color || "var(--color-success)"} height={24} /></div>
    </div>
  );
}

/* ── HORIZONTAL BAR ── */

function HorizontalBar({
  label,
  value,
  pct: pctVal,
  color,
}: {
  label: string;
  value: string;
  pct: number;
  color: string;
}) {
  return (
    <div className="llm-hbar">
      <div className="llm-hbar-label font-tabular">{label}</div>
      <div className="llm-hbar-track">
        <div className="llm-hbar-fill" style={{ width: `${Math.max(0, Math.min(100, pctVal))}%`, background: color }} />
      </div>
      <div className="llm-hbar-val font-tabular">{value}</div>
    </div>
  );
}

/* ── COMPACT STAT CARD (for advanced grid) ── */

function CompactCard({
  label,
  value,
  color,
}: {
  label: string;
  value: string;
  color?: string;
}) {
  return (
    <div className="llm-adv-card">
      <div className="llm-adv-label">{label}</div>
      <div className="llm-adv-val font-tabular" style={color ? { color } : undefined}>{value}</div>
    </div>
  );
}

/* ── ODOMETER CARD (radial gauge for tok/s) ── */

function OdometerCard({
  label,
  value,
  max,
  displayValue,
  sub,
  color,
  sparkData,
}: {
  label: string;
  value: number;
  max: number;
  displayValue: string;
  sub?: string;
  color: string;
  sparkData: number[];
}) {
  return (
    <div className="llm-odometer-card">
      <div className="llm-odometer-label">{label}</div>
      <div className="llm-odometer-body">
        <ArcGauge value={value} max={max} label="" displayValue={displayValue} sub={sub} color={color} size={100} />
      </div>
      <div className="llm-odometer-spark"><Sparkline data={sparkData} color={color} height={18} /></div>
    </div>
  );
}

function RecipeBadge({ label, value, accent }: { label: string; value: string; accent?: string }) {
  if (!value || value === "\u2014") return null;
  return (
    <span className={`recipe-badge${accent ? ` recipe-badge--${accent}` : ""}`}>
      <span className="recipe-badge-label">{label}</span>
      <span className="recipe-badge-value">{value}</span>
    </span>
  );
}

/* ── CONFIG / PROVENANCE SECTION (top of panel) ── */

function ConfigProvenanceSection({
  llm,
}: {
  llm: LlmMetrics | null;
}) {
  const info = llm?.recipeInfo;
  const metadata = llm?.recipeMetadata;
  const hasInfo = info != null;
  const hasMeta = metadata != null && (metadata.model != null || metadata.supportedParameters.length > 0);
  if (!hasInfo && !hasMeta) return null;

  const params = metadata?.supportedParameters ?? [];
  const specDecodeMethod = info?.specDecodeMethod ?? null;
  const specDecodeLabel = specDecodeMethod
    ? specDecodeMethod.replace(/\s*k=\d+$/, "").trim()
    : llm?.speculativeTypes && llm.speculativeTypes !== "none"
      ? llm.speculativeTypes
      : "MTP";

  const author = info?.author ?? null;
  const authorName = info?.authorName ?? null;
  const containerImage = info?.containerImage ?? null;
  const engineType = info?.engineType ?? null;
  const modelName = info?.modelName ?? metadata?.model ?? null;
  const acceptPct = info?.acceptRatio != null ? `${Math.round(info.acceptRatio * 100)}%` : "—";
  const acceptAccent = info?.acceptRatio != null ? (info.acceptRatio > 0.7 ? "success" : info.acceptRatio >= 0.5 ? "warning" : "danger") : undefined;
  const [paramsExpanded, setParamsExpanded] = useState(false);

  const paramStats: Record<string, { label: string; value: string; accent?: string } | null> = {};
  if (llm) {
    for (const p of params) {
      // tools / tool_choice → tool call parser (show "parser ready" type info)
      if (p === "tools" || p === "tool_choice") {
        // No numeric stat, but we can indicate it is parsed
        paramStats[p] = null;
      }
      // stream → streaming status
      if (p === "stream") {
        paramStats[p] = { label: "streaming", value: "live", accent: "success" };
      }
      // reasoning_effort → show the reasoning effort level if available
      if (p === "reasoning_effort") {
        // DS4 supports reasoning — show spec decode method as proxy
        paramStats[p] = info?.specDecodeMethod
          ? { label: "reasoning", value: info.specDecodeMethod, accent: "accent" }
          : null;
      }
    }
  }

  // Stats that map to recipe config fields with live metric counterparts
  const liveStats: { label: string; value: string; sub?: string; accent?: string }[] = [];
  if (llm) {
    // Spec decode acceptance → relates to specDecodeMethod
    if (llm.dsparkAcceptRatio != null) {
      liveStats.push({ label: `${specDecodeLabel} accept`, value: pct(llm.dsparkAcceptRatio, 1), accent: llm.dsparkAcceptRatio > 0.7 ? "success" : llm.dsparkAcceptRatio >= 0.5 ? "warning" : "danger" });
    } else if (info?.acceptRatio != null) {
      liveStats.push({ label: "Accept ratio", value: acceptPct, accent: acceptAccent });
    }
    // Banks live/total → relates to maxLanes
    if (llm.banksLive != null) {
      liveStats.push({ label: "Banks", value: `${fmtInt(llm.banksLive)}/${fmtInt(llm.banksTotal)}`, accent: llm.banksLive > 0 ? "success" : undefined });
    }
    // Prefill cached/computed → relates to prefixCaching
    if (llm.prefillCached != null || llm.prefillComputed != null) {
      const cached = llm.prefillCached ?? 0;
      const computed = llm.prefillComputed ?? 0;
      const total = cached + computed;
      const ratio = total > 0 ? cached / total : 0;
      liveStats.push({ label: "Prefill cache", value: pct(ratio, 0), sub: `${fmtInt(cached)} cached`, accent: ratio > 0.5 ? "success" : "warning" });
    }
    // Warm records → relates to prefixCaching
    if (llm.warmRecords != null) {
      liveStats.push({ label: "Warm records", value: fmtInt(llm.warmRecords), accent: "accent" });
    }
    // Spec drafts/hits → relates to specDecodeMethod
    if (llm.specDrafts != null && llm.specHits != null) {
      liveStats.push({ label: "Spec drafts", value: fmtInt(llm.specDrafts), sub: `${fmtInt(llm.specHits)} hits`, accent: "accent" });
    }
    // Total tokens decoded
    if (llm.totalTokensDecoded != null) {
      liveStats.push({ label: "Tokens decoded", value: fmtInt(llm.totalTokensDecoded), accent: "accent" });
    }
    // Uptime
    if (llm.ds4Uptime != null) {
      liveStats.push({ label: "Uptime", value: fmtUptime(llm.ds4Uptime) });
    }
  }

  return (
    <div className="llm-config-section">
      {/* Provenance / Accreditation */}
      {(author || authorName || engineType || containerImage) && (
        <div className="llm-provenance-row">
          {engineType && <span className="recipe-info-engine-type">{engineType}</span>}
          {modelName && <span className="llm-config-model">{modelName}</span>}
          {(author || authorName) && (
            <span className="llm-provenance-author">
              <span className="recipe-info-author-label">Built by</span>{" "}
              <span className="recipe-info-author">{author || "\u2014"}</span>
              {authorName && <span className="recipe-info-author-name"> ({authorName})</span>}
            </span>
          )}
          {containerImage && (
            <span className="llm-provenance-container">
              <span className="recipe-info-sep">·</span>
              <span className="recipe-info-container" title={containerImage}>{containerImage}</span>
            </span>
          )}
        </div>
      )}

      {/* Config badges */}
      <div className="recipe-info-badges">
        {engineType && <RecipeBadge label="Engine" value={engineType} />}
        {modelName && <RecipeBadge label="Model" value={modelName} accent="accent" />}
        {(info?.contextLength ?? metadata?.contextLength) != null && (
          <RecipeBadge label="Ctx" value={(info?.contextLength ?? metadata?.contextLength)! >= 1024 ? `${Math.round((info?.contextLength ?? metadata?.contextLength)! / 1024)}K` : String((info?.contextLength ?? metadata?.contextLength)!)} />
        )}
        {info?.maxLanes != null && <RecipeBadge label="Lanes" value={String(info.maxLanes)} accent="accent" />}
        {specDecodeMethod && <RecipeBadge label="Spec" value={specDecodeMethod} accent="success" />}
        {info?.quantization && <RecipeBadge label="Quant" value={info.quantization} accent="warning" />}
        {info?.prefixCaching != null && <RecipeBadge label="Prefix" value={info.prefixCaching ? "ON" : "OFF"} accent={info.prefixCaching ? "success" : "danger"} />}
        {info?.kvCacheDtype && <RecipeBadge label="KV dtype" value={info.kvCacheDtype} />}
        {info?.gmu != null && <RecipeBadge label="GMU" value={pct(info.gmu, 0)} />}
      </div>

      {/* Params (collapsible) */}
      {params.length > 0 && (
        <div className="llm-config-params-wrap">
          <button
            type="button"
            className="llm-config-params-toggle"
            onClick={() => setParamsExpanded(!paramsExpanded)}
          >
            <span className="llm-config-params-toggle-icon">{paramsExpanded ? "\u25BC" : "\u25B8"}</span>
            Params ({params.length})
          </button>
          {paramsExpanded && (
            <div className="recipe-merged-params" style={{ marginTop: "6px" }}>
              {params.map((p) => (
                <span key={p} className="recipe-param-badge">
                  <span className="recipe-param-badge-name">{p}</span>
                  {paramStats[p] && (
                    <span className={`recipe-param-badge-stat${paramStats[p]!.accent ? ` recipe-param-badge-stat--${paramStats[p]!.accent}` : ""}`}>
                      {paramStats[p]!.value}
                    </span>
                  )}
                </span>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

const HISTORY = 60;
const DECODE_DIAL_MAX = 120;
const PREFILL_DIAL_MAX = 3000;

interface LlmChartHistory {
  genTps: number[];
  prefillTps: number[];
  ttft: number[];
  e2e: number[];
  specAccept: number[];
}

export function LlmPanel({
  llm,
  sparkId,
  sparkName,
  llmPort,
  llmPorts,
  hasApiKey = false,
  shareImage = false,
  onRemovePort,
  className,
}: LlmPanelProps) {
  // Tail keyed by port so multi-port LLM sparklines stay distinct (8b).
  const genHistory = useMetricsHistoryTail(sparkId, `llm:${llmPort}.tps`);
  const prefillHistory = useMetricsHistoryTail(sparkId, `llm:${llmPort}.prefill`);
  const cachedPrefillHistory = useMetricsHistoryTail(sparkId, `llm:${llmPort}.prefillCached`);
  const uncachedPrefillHistory = useMetricsHistoryTail(sparkId, `llm:${llmPort}.prefillUncached`);

  // Full series (~1 h) for running averages over busy (>0) samples only.
  const genFull = useMetricsHistory(sparkId, `llm:${llmPort}.tps`);
  const prefillFull = useMetricsHistory(sparkId, `llm:${llmPort}.prefill`);
  const cachedFull = useMetricsHistory(sparkId, `llm:${llmPort}.prefillCached`);
  const uncachedFull = useMetricsHistory(sparkId, `llm:${llmPort}.prefillUncached`);
  const genAvg = useMemo(() => avgPositive(genFull), [genFull]);
  const prefillAvg = useMemo(() => avgPositive(prefillFull), [prefillFull]);
  const cachedPrefillAvg = useMemo(() => avgPositive(cachedFull), [cachedFull]);
  const uncachedPrefillAvg = useMemo(() => avgPositive(uncachedFull), [uncachedFull]);
  const [showSettings, setShowSettings] = useState(false);
  const [portDraft, setPortDraft] = useState(String(llmPort));
  const [apiKeyDraft, setApiKeyDraft] = useState("");
  const [clearApiKey, setClearApiKey] = useState(false);
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [engineInfoOpen, setEngineInfoOpen] = useState(false);
  const [openBench, setOpenBench] = useState<BenchKind | null>(null);
  const benchOpen = openBench === "decode";
  const prefillBenchOpen = openBench === "prefill";
  const setBenchOpen = (v: boolean) => setOpenBench(v ? "decode" : null);
  const [remoteTarget, setRemoteTarget] = useState<LlmBenchTarget | null>(null);
  const launchLocal = useCallback((kind: BenchKind) => {
    setRemoteTarget(null);
    setOpenBench(kind);
  }, []);
  const launchRemote = useCallback((kind: BenchKind, target: LlmBenchTarget) => {
    setRemoteTarget(target);
    setOpenBench(kind);
  }, []);
  /** Which vLLM metric info tip is open (kvCache | requests | ttftP95 | preempts). */
  const [metricInfoId, setMetricInfoId] = useState<string | null>(null);
  const engineInfoTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const clearEngineInfoTimer = useCallback(() => {
    if (engineInfoTimer.current != null) {
      clearTimeout(engineInfoTimer.current);
      engineInfoTimer.current = null;
    }
  }, []);

  const startEngineInfoTimer = useCallback(() => {
    clearEngineInfoTimer();
    engineInfoTimer.current = setTimeout(() => setEngineInfoOpen(false), 2000);
  }, [clearEngineInfoTimer]);

  const generationTps = llm?.generationTps ?? 0;
  const prefillTps = llm?.prefillTps ?? 0;
  const showPrefillSplit = llm?.cachedPrefillTps != null || llm?.uncachedPrefillTps != null;
  const cachedPrefillTps = llm?.cachedPrefillTps ?? 0;
  const uncachedPrefillTps = llm?.uncachedPrefillTps ?? 0;
  const available = llm?.available ?? false;
  // While nothing is flowing, say when the endpoint last served.
  const idleNote = available && isLlmIdle({ generationTps, prefillTps })
    ? idleLabel(llm?.lastActiveAt)
    : null;
  // styles01 badge states
  const isThinking = available &&
    ((llm?.requestsRunning ?? llm?.slotsActive ?? 0) > 0) &&
    (llm?.isPrefilling !== true) &&
    (llm?.generationTps ?? 0) < 1;
  const isPrefilling = available &&
    ((llm?.requestsRunning ?? llm?.slotsActive ?? 0) > 0) &&
    (llm?.isPrefilling === true ||
      (llm?.isPrefilling == null && (llm?.generationTps ?? 0) < 1 && (llm?.prefillTps ?? 0) > 0));

  useEffect(() => {
    if (!llm || !available) return;
    const gen = llm.generationTps ?? 0;
    const rawPre = llm.prefillTps ?? 0;
    const ttft = llm.ttft ?? NaN;
    const e2e = llm.e2eLatency ?? NaN;
    const specRate = llm.specAcceptanceRate ?? llm.dsparkAcceptRatio ?? llm.mtpAcceptanceRate ?? NaN;
    setHistory((prev) => ({
      genTps: pushSample(prev.genTps, gen),
      prefillTps: pushSample(prev.prefillTps, rawPre),
      ttft: pushSample(prev.ttft, ttft),
      e2e: pushSample(prev.e2e, e2e),
      specAccept: pushSample(prev.specAccept, specRate),
    }));
  }, [llm, available]);

  // Keep draft in sync when server pushes a different port (other tab / reload)
  useEffect(() => {
    if (!showSettings) {
      setPortDraft(String(llmPort));
      setApiKeyDraft("");
      setClearApiKey(false);
    }
  }, [llmPort, showSettings]);

  const parsedPort = (() => {
    const n = parseInt(portDraft, 10);
    if (!Number.isInteger(n) || n < 1 || n > 65535) return null;
    return n;
  })();

  const portDirty = parsedPort !== null && parsedPort !== llmPort;
  const portInvalid = portDraft.trim() !== "" && parsedPort === null;
  const apiKeyDirty = apiKeyDraft.trim() !== "" || clearApiKey;
  const settingsDirty = portDirty || apiKeyDirty;

  const handleSaveSettings = async () => {
    if (parsedPort === null) {
      setSaveError("Port must be an integer 1–65535");
      return;
    }
    if (!settingsDirty) {
      setShowSettings(false);
      return;
    }
    setSaving(true);
    setSaveError(null);
    try {
      if (portDirty) {
        const currentPorts =
          Array.isArray(llmPorts) && llmPorts.length > 0 ? llmPorts : [llmPort];
        if (currentPorts.includes(parsedPort) && parsedPort !== llmPort) {
          setSaveError(`Port ${parsedPort} is already configured`);
          setSaving(false);
          return;
        }
        // Rename this panel's port in-place so sibling ports (and their keys) survive
        if (currentPorts.length > 1) {
          const next = currentPorts.map((p) => (p === llmPort ? parsedPort : p));
          await updateLlmPorts(sparkId, next);
        } else {
          await updateLlmPort(sparkId, parsedPort);
        }
      }
      const keyPort = parsedPort;
      if (clearApiKey) {
        await setLlmApiKey(sparkId, keyPort, "");
      } else if (apiKeyDraft.trim() !== "") {
        await setLlmApiKey(sparkId, keyPort, apiKeyDraft.trim());
      }
      setApiKeyDraft("");
      setClearApiKey(false);
      setShowSettings(false);
    } catch (err) {
      setSaveError(err instanceof Error ? err.message : "Failed to save LLM settings");
    } finally {
      setSaving(false);
    }
  };


  // styles01 chart history
  const [history, setHistory] = useState<LlmChartHistory>({
    genTps: [], prefillTps: [], ttft: [], e2e: [], specAccept: [],
  });
  // styles01 display values (sticky last-positive for idle gaps)
  const stickyGenTps = useRef<number>(0);
  const stickyPrefillTps = useRef<number>(0);
  const stickyPeak = useRef<number>(0);
  useEffect(() => {
    if (!llm || !available) return;
    if ((llm.generationTps ?? 0) > 0) stickyGenTps.current = llm.generationTps;
    if ((llm.prefillTps ?? 0) > 0) stickyPrefillTps.current = llm.prefillTps;
    else if ((llm.requestsRunning ?? llm.slotsActive ?? 0) === 0) stickyPrefillTps.current = 0;
    if ((llm.peakAggregateTps ?? 0) > 0) stickyPeak.current = llm.peakAggregateTps as number;
  }, [llm, available]);
  // Dead probe: drop all sticky display values immediately — a dead poll must never
  // keep a stale (possibly poisoned) spike on screen.
  useEffect(() => {
    if (!available) {
      stickyGenTps.current = 0;
      stickyPrefillTps.current = 0;
      stickyPeak.current = 0;
    }
  }, [available]);
  const displayGenTps = (llm?.generationTps ?? 0) > 0 ? (llm?.generationTps ?? 0) : stickyGenTps.current;
  const displayPrefillTps = (llm?.prefillTps ?? 0) > 0 ? (llm?.prefillTps ?? 0) : stickyPrefillTps.current;
  const displayAggTps = displayGenTps + displayPrefillTps;
  const displaySingleTps = llm?.perStreamAvg ?? llm?.rollingAvgTpsPerSlot ?? displayGenTps;
  const displayPeak = (llm?.peakAggregateTps ?? 0) > 0 ? (llm?.peakAggregateTps ?? 0) : stickyPeak.current;
  const prefillPeak = prefillFull.reduce((m, p) => (Number(p) > m ? Number(p) : m), 0);

  /* -- styles01 dashboard derivations (ported; data-availability gated) -- */
  const currentSpecAccept = llm?.specAcceptanceRate ?? llm?.dsparkAcceptRatio ?? llm?.mtpAcceptanceRate ?? null;
  const avgSpecAccept = arrMean(history.specAccept);
  const specAccepted = llm?.specAcceptedTokens ?? llm?.mtpAcceptedTokens ?? null;
  const specGenerated = llm?.specGeneratedTokens ?? llm?.mtpDraftedTokens ?? null;
  const mtpAccepted = specAccepted;
  const mtpDrafted = specGenerated;
  const specHits = llm?.specHits ?? specAccepted;
  const specDrafts = llm?.specDrafts ?? specGenerated;
  const perPos: number[] = llm?.perPositionAcceptance ?? [];
  const banksLive = llm?.banksLive ?? null;
  const banksTotal = llm?.banksTotal ?? null;
  const prefixHit = llm?.prefixCacheHitRate ?? llm?.cacheHitRatio ?? null;
  const runningSlots = llm?.runningSlots ?? llm?.slotsActive ?? 0;
  const waitingSlots = llm?.waitingSlots ?? 0;
  const promptTokens = llm?.promptTokens ?? null;
  const promptTokensProcessed = llm?.promptTokensProcessed ?? null;
  const promptTokensCache = llm?.promptTokensCache ?? null;
  const cacheHitRatio = llm?.cacheHitRatio ?? null;
  const nCtx = llm?.nCtx ?? null;
  const isProcessing = llm?.isProcessing ?? false;
  const nRemain = llm?.nRemain ?? null;
  const nDecoded = llm?.nDecoded ?? null;
  const samplingParams = llm?.samplingParams ?? null;
  const speculativeTypes = llm?.speculativeTypes ?? null;
  const reasoningFormat = llm?.reasoningFormat ?? null;
  const chatFormat = llm?.chatFormat ?? null;
  const samplers = llm?.samplers ?? null;
  const specAcceptanceRate = llm?.specAcceptanceRate ?? null;
  const specAcceptedTokens = llm?.specAcceptedTokens ?? null;
  const specGeneratedTokens = llm?.specGeneratedTokens ?? null;
  const specMeanLen = llm?.specMeanLen ?? null;
  const tokPerStepVal = llm?.tokPerStep ?? null;
  // REAL context usage (vLLM): kvUsedTokens = kv usage gauge x KV pool size
  // = tokens actually resident in the KV cache right now, plus a mid-flight
  // decode allowance (avg tok/step x active lanes). The prompt+decode sum is
  // the llama.cpp /slots convention only (per-slot state) — vLLM's
  // prompt/generation counters are LIFETIME totals and must not be summed.
  const llmKvUsed = llm?.kvUsedTokens ?? null;
  const decodeAllowance = Math.round((tokPerStepVal ?? 0) * Math.max(llm?.banksLive ?? 0, llm?.slotsActive ?? 0, 0));
  const contextUsedTokens = llmKvUsed != null
    ? llmKvUsed + decodeAllowance
    : (promptTokens != null ? promptTokens : 0) + (nDecoded != null ? nDecoded : 0);
  const contextUsedPct = nCtx != null && nCtx > 0 ? contextUsedTokens / nCtx : null;
  const kvUsage = llm?.kvCacheUsage ?? null;
  const specDecodeMethod = llm?.recipeInfo?.specDecodeMethod ?? null;
  const specDecodeLabel = specDecodeMethod
    ? specDecodeMethod.replace(/\s*k=\d+$/, "").trim()
    : speculativeTypes && speculativeTypes !== "none"
      ? speculativeTypes
      : "MTP";
  const kvColor = kvUsage != null && kvUsage >= 0.8 ? "var(--color-danger)" : kvUsage != null && kvUsage >= 0.5 ? "var(--color-accent)" : "var(--color-success)";
  const lanesColor = (banksLive ?? 0) > 0 ? "var(--color-success)" : "var(--color-muted)";
  const sparkGen = genHistory;
  const sparkPrefill = prefillHistory;
  const sparkAvg = genHistory;
  const sparkSingle = genHistory;
  const sparkPeak = genHistory;
  const reasoningEffort = llm?.reasoningEffort ?? null;
  const effortIndex = reasoningEffort === "high" ? 3 : reasoningEffort === "medium" ? 2 : reasoningEffort === "low" ? 1 : 0;
  const specDecodeBars: { label: string; value: number }[] = perPos.length > 0
    ? perPos.map((p, i) => ({ label: `Pos ${i}`, value: p }))
    : [{ label: "Pos 0", value: Number(currentSpecAccept ?? 0) }];
  const genSeries: ChartSeries = { label: "decode", color: "var(--color-success)", data: history.genTps, area: true, yAxis: "left" };
  const preSeries: ChartSeries = { label: "prefill", color: "var(--color-accent)", data: history.prefillTps, area: false, yAxis: "right", logScale: true };
  const ttftSeries: ChartSeries = { label: "TTFT", color: "var(--color-danger)", data: history.ttft, area: false };
  const e2eSeries: ChartSeries = { label: "E2E", color: "var(--color-info)", data: history.e2e, area: false };

  const fmtTps = (n: number) => (n >= 1000 ? Math.round(n).toLocaleString() : n.toFixed(1));
  const fmtAvg = (n: number) => (n >= 100 ? n.toFixed(0) : n.toFixed(1));
  const isVllm = llm != null && (llm.backend === "vllm" || llm.backend === "q27");
  // FreeToken reports KV use, active requests and a TTFT mean in /v1/stats; the tiles below are null-gated.
  const isFreeToken = llm?.backend === "freetoken";
  const showEngineTiles = isVllm || isFreeToken;
  const kvTone =
    llm?.kvCacheUsage == null
      ? ""
      : llm.kvCacheUsage >= 0.8
        ? "text-danger"
        : llm.kvCacheUsage >= 0.5
          ? "text-warning"
          : "text-success";
  const contextLabel = llm?.contextLength
    ? llm.contextLength >= 1000
      ? `${Math.round(llm.contextLength / 1024)}k ctx`
      : `${llm.contextLength} ctx`
    : null;
  const backend = backendLabel(llm?.backend ?? null);

  const tile = (id: string, label: string, value: string, opts?: { tone?: string; sub?: string; align?: "left" | "right" }) => (
    <div className="sp-tile" key={id}>
      <b className={opts?.tone ?? ""}>{value}</b>
      <MetricInfoTip
        id={id}
        label={label}
        text={VLLM_METRIC_INFO[id as keyof typeof VLLM_METRIC_INFO]}
        openId={metricInfoId}
        setOpenId={setMetricInfoId}
        align={opts?.align}
      />
      {opts?.sub && <small>{opts.sub}</small>}
    </div>
  );

  const kvValue =
    llm?.kvCacheUsage != null
      ? `${(llm.kvCacheUsage * 100).toFixed(0)}%`
      : llm?.kvCacheTokensAvailable != null
        ? `${formatKvTokens(llm.kvCacheTokensAvailable)} free`
        : llm?.kvCacheTokens != null
          ? formatKvTokens(llm.kvCacheTokens)
          : null;
  const kvSub = [
    llm?.kvCacheUsage != null && llm?.kvCacheTokensAvailable != null
      ? `${formatKvTokens(llm.kvCacheTokensAvailable)} free`
      : null,
    llm?.kvCacheTokens != null ? `${formatKvTokens(llm.kvCacheTokens)} pool` : null,
    llm?.kvCacheMemoryBytes != null ? formatKvBytes(llm.kvCacheMemoryBytes) : null,
  ]
    .filter(Boolean)
    .join(" · ");

  return (
    <Panel
      title="LLM server"
      accent={available}
      icon={<BotIcon />}
      className={`panel-llm ${className ?? ""}`}
      bodyClassName="sp-stack"
      actions={
        <div className="sp-actions">
          {available && backend && (
            <Tag tone="info" title={`Backend: ${backend}`}>
              {backend} · :{llmPort}
            </Tag>
          )}
          {onRemovePort && (
            <button
              type="button"
              title={`Remove port ${llmPort}`}
              onClick={() => onRemovePort(llmPort)}
              className="btn btn--sm btn--danger"
            >
              <span aria-hidden>×</span>
              <span>Remove</span>
            </button>
          )}
          <button
            type="button"
            title={showSettings ? "Done" : "LLM settings"}
            onClick={() => {
              if (showSettings) {
                setPortDraft(String(llmPort));
                setApiKeyDraft("");
                setClearApiKey(false);
                setSaveError(null);
              }
              setShowSettings(!showSettings);
            }}
            disabled={saving}
            className={`btn btn--sm btn--ghost ${showSettings ? "is-on" : ""}`}
          >
            <GearIcon />
            <span>{showSettings ? "Done" : "Settings"}</span>
          </button>
        </div>
      }
    >
      {showSettings ? (
        <div className="sp-stack">
          <p className="sp-hint">
            HTTP port of the LLM server on this Spark (vLLM / llama.cpp / sglang / ds4 / EXL3 / TensorFold / OpenAI-compatible gateway).
          </p>
          <label className="sp-field">
            <span className="eyebrow">Port</span>
            <input
              type="number"
              min={1}
              max={65535}
              inputMode="numeric"
              value={portDraft}
              onChange={(e) => {
                setPortDraft(e.target.value);
                setSaveError(null);
              }}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  e.preventDefault();
                  void handleSaveSettings();
                }
              }}
              className="sp-input"
            />
          </label>
          <label className="sp-field">
            <span className="eyebrow">API key (optional)</span>
            <input
              type="password"
              autoComplete="new-password"
              spellCheck={false}
              value={apiKeyDraft}
              disabled={clearApiKey}
              placeholder={hasApiKey && !clearApiKey ? "•••••••• (saved — leave blank to keep)" : "Bearer token if required"}
              onChange={(e) => {
                setApiKeyDraft(e.target.value);
                setClearApiKey(false);
                setSaveError(null);
              }}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  e.preventDefault();
                  void handleSaveSettings();
                }
              }}
              className="sp-input"
            />
          </label>
          {hasApiKey && (
            <label className="sp-check">
              <input
                type="checkbox"
                checked={clearApiKey}
                onChange={(e) => {
                  setClearApiKey(e.target.checked);
                  if (e.target.checked) setApiKeyDraft("");
                  setSaveError(null);
                }}
              />
              Clear saved API key
            </label>
          )}
          {portInvalid && <p className="sp-error">Enter an integer between 1 and 65535</p>}
          {saveError && <p className="sp-error">{saveError}</p>}
          <div className="sp-btns sp-btns--end">
            <button
              type="button"
              onClick={() => {
                setPortDraft(String(llmPort));
                setApiKeyDraft("");
                setClearApiKey(false);
                setSaveError(null);
                setShowSettings(false);
              }}
              disabled={saving}
              className="btn btn--sm"
            >
              Cancel
            </button>
            <button
              type="button"
              onClick={() => void handleSaveSettings()}
              disabled={saving || portInvalid || !settingsDirty}
              className="btn btn--sm btn--primary"
            >
              {saving ? "Saving…" : "Save"}
            </button>
          </div>
        </div>
      ) : !available ? (
        <div className="sp-stack">
          <div className="sp-chips">
            {llm?.posture ? <PostureBadge posture={llm.posture} /> : <span className="sdot sdot--off" />}
            <p className="sp-muted">
              {llm?.posture?.auth === "protected"
                ? `${llm.posture.label} on :${llmPort}`
                : `No model loaded on :${llmPort}`}
            </p>
          </div>
          <LlmLaunchers
            sparkId={sparkId}
            llmPort={llmPort}
            modelId={llm?.modelId}
            onLaunch={launchLocal}
            onRemoteLaunch={launchRemote}
          />
          <LlmDailyChart sparkId={sparkId} llmPort={llmPort} />
          <LlmTokenTotals sparkId={sparkId} llmPort={llmPort} />
        </div>
      ) : (
        <div className="sp-stack">
          <div className="sp-model">
            <div className="eyebrow">
              <span>Model</span>
              {contextLabel && <span>{contextLabel}</span>}
            </div>
            <code title={llm?.modelId ?? undefined}>{llm?.modelId ?? "—"}</code>
            {llm?.modelPath && llm.modelPath !== llm.modelId && !llm.modelPath.includes("models--") && (
              <small title={llm.modelPath}>{llm.modelPath}</small>
            )}
          </div>

        <div className="llm-dashboard">
          {/* ═══ ROW 1 — HEADER BAR ═══ */}
          <div className="llm-header-bar">
            <span className="llm-badge"><span className="h-1.5 w-1.5 rounded-full bg-accent" />{llm?.backend ?? "\u2014"}</span>
            {llm?.posture && <span className={`llm-posture llm-posture--${llm.posture.level}`}><span className="llm-posture__dot" />{llm.posture.label}</span>}
            {llm?.modelId && <span className="min-w-0 flex-1 truncate text-[11px] text-text" title={llm.modelId}>{llm.modelId}</span>}
            <span className="shrink-0 font-tabular text-[10px] text-muted">:{llmPort}</span>
            {(() => {
              const active = isThinking || isPrefilling;
              const label = isPrefilling ? "PREFILLING" : "ACTIVELY THINKING";
              return (
                <span className={`llm-thinking-badge ${active ? "llm-thinking-badge--active" : "llm-thinking-badge--idle"} ${isPrefilling ? "llm-thinking-badge--prefill" : ""}`}>
                  <span className="llm-thinking-pulse" />
                  <span className="llm-thinking-label">{label}</span>
                </span>
              );
            })()}
          </div>

          {/* ═══ ROW 2 — CONFIG / PROVENANCE SECTION (moved to top) ═══ */}
          <ConfigProvenanceSection llm={llm ?? null} />

          {/* ═══ ROW 3 — HERO STATS (4 cards: 2 spark + 2 odometer) ═══ */}
          <div className="llm-hero-grid">
            <OdometerCard label="Decode tok/s" value={displayGenTps} max={DECODE_DIAL_MAX} displayValue={fmtNum(displayGenTps, 1)} sub={`prefill ${fmtNum(displayPrefillTps, 1)}`} color={tpsColor(displayGenTps)} sparkData={sparkGen as number[]} />
            <OdometerCard label="Prefill tok/s" value={displayPrefillTps} max={PREFILL_DIAL_MAX} displayValue={fmtNum(displayPrefillTps, 1)} sub={`decode ${fmtNum(displayGenTps, 1)}`} color={tpsColor(displayPrefillTps)} sparkData={sparkPrefill as number[]} />
            <OdometerCard label="Avg tok/s" value={genAvg ?? 0} max={DECODE_DIAL_MAX} displayValue={fmtNum(genAvg ?? 0, 1)} sub={`prefill avg ${fmtNum(prefillAvg ?? 0, 1)}`} color={tpsColor(genAvg ?? 0)} sparkData={sparkSingle as number[]} />
            <OdometerCard label="Peak tok/s" value={displayPeak} max={DECODE_DIAL_MAX} displayValue={fmtNum(displayPeak, 1)} sub={`prefill peak ${fmtNum(prefillPeak ?? 0, 1)}`} color={tpsColor(displayPeak)} sparkData={sparkPeak as number[]} />
          </div>

          {/* ═══ ROW 4 — GAUGE ARCS ═══ */}
          <div className="llm-gauge-row">
            {mtpAccepted != null && mtpDrafted != null && mtpDrafted > 0 && (
              <GaugeCard label="Spec Accept">
                <ArcGauge value={mtpAccepted} max={mtpDrafted} label="" displayValue={fmtInt(mtpAccepted)} sub={`of ${fmtInt(mtpDrafted)}`} color={mtpColor(mtpAccepted / mtpDrafted)} size={90} />
              </GaugeCard>
            )}
            {currentSpecAccept != null && (
              <GaugeCard label="Spec Accept %" sparkData={history.specAccept} sparkColor={mtpColor(currentSpecAccept)}>
                <FullCircleGauge value={currentSpecAccept} max={1} displayValue={pct(currentSpecAccept, 0)} color={mtpColor(currentSpecAccept)} size={120} />
              </GaugeCard>
            )}
            {banksTotal != null && (
              <GaugeCard label="Active Lanes">
                <ArcGauge value={banksLive ?? 0} max={banksTotal} label="" displayValue={`${fmtInt(banksLive)}/${fmtInt(banksTotal)}`} color={lanesColor} size={100} />
              </GaugeCard>
            )}
            {kvUsage != null && (
              <GaugeCard label="KV Cache">
                <ArcGauge value={kvUsage} max={1} label="" displayValue={pct(kvUsage, 1)} sub={llm?.contextUsedBytes != null ? `${(llm.contextUsedBytes / 1e9).toFixed(1)}GB` : undefined} color={kvColor} size={100} />
              </GaugeCard>
            )}
            {reasoningEffort != null && effortIndex > 0 && (
              <GaugeCard label="Reasoning">
                <ArcGauge value={effortIndex} max={4} label="" displayValue={reasoningEffort} color="var(--color-accent)" size={90} />
              </GaugeCard>
            )}
            {nCtx != null && (
              <GaugeCard label="Context Used">
                <ArcGauge value={contextUsedTokens} max={llm?.kvPoolTokens ?? nCtx} label="" displayValue={fmtInt(contextUsedTokens)} sub={contextUsedPct != null ? pct(contextUsedPct, 1) : undefined} color={contextUsedPct != null && contextUsedPct >= 0.8 ? "var(--color-danger)" : contextUsedPct != null && contextUsedPct >= 0.5 ? "var(--color-accent)" : "var(--color-success)"} size={100} />
              </GaugeCard>
            )}
            {cacheHitRatio != null && (
              <GaugeCard label="Cache Hit">
                <FullCircleGauge value={cacheHitRatio} max={1} displayValue={pct(cacheHitRatio, 0)} color={cacheHitRatio >= 0.7 ? "var(--color-success)" : cacheHitRatio >= 0.4 ? "var(--color-accent)" : "var(--color-warning)"} size={100} />
              </GaugeCard>
            )}
          </div>

          {/* ═══ ROW 5 — CHART ROW: Throughput + Latency + Spec Decode (ALL THREE ALWAYS VISIBLE) ═══ */}
          <div className="llm-chart-row">
            <div className="llm-chart-block">
              <div className="llm-chart-title">Throughput <span className="llm-chart-sub">decode + prefill · last 60</span></div>
              <TelemetryChart series={[genSeries, preSeries]} maxPoints={HISTORY} height={110} yUnit="" yUnitRight="" yMin={0} yMax={100} yMaxRight={3000} />
            </div>
            <div className="llm-chart-block">
              <div className="llm-chart-title">Latency <span className="llm-chart-sub">TTFT + E2E · seconds</span></div>
              <TelemetryChart series={[ttftSeries, e2eSeries]} maxPoints={HISTORY} height={110} yUnit="s" yMin={0} />
            </div>
            <div className="llm-chart-block llm-specdecode-block">
              <div className="llm-chart-title">Speculative Decode <span className="llm-chart-sub">{specDecodeLabel} · acceptance</span></div>
              <div className={specDecodeBars.length > 6 ? "llm-hbar-grid llm-hbar-grid-multi" : "llm-hbar-grid"}>
                {specDecodeBars.map((bar, i) => (
                  <HorizontalBar key={i} label={bar.label} value={pct(bar.value, 0)} pct={bar.value * 100} color={mtpColor(bar.value)} />
                ))}
              </div>
              {specHits != null && specDrafts != null && (
                <div className="llm-specdecode-counters font-tabular">
                  <span className="text-muted text-[10px]">{fmtInt(specHits)} hits / {fmtInt(specDrafts)} drafts</span>
                </div>
              )}
              {specAcceptedTokens != null && specGeneratedTokens != null && (
                <div className="llm-specdecode-counters font-tabular">
                  <span className="text-muted text-[10px]">{fmtInt(specAcceptedTokens)} accepted / {fmtInt(specGeneratedTokens)} generated{specMeanLen != null ? ` · mean len ${fmtNum(specMeanLen, 1)}` : ""}</span>
                </div>
              )}
              {perPos.length === 0 && (
                <div className="llm-specdecode-note">
                  <span className="text-muted text-[9px]">⚠️ {llm?.backend === "ds4" ? "DS4 reports aggregate acceptance only — per-position bars are estimated from overall ratio (geometric decay)" : "No per-position breakdown available — bars estimated from overall ratio"}</span>
                </div>
              )}
            </div>
          </div>

          {/* ═══ ROW 6 — ADVANCED STATS (30+ compact cards) ═══ */}
          <div className="llm-advanced-grid">
            {/* Latency metrics */}
            <CompactCard label="TTFT" value={fmtNum(llm?.ttft, 3, "s")} color={latencyColor(llm?.ttft ?? NaN)} />
            <CompactCard label="E2E" value={fmtNum(llm?.e2eLatency, 2, "s")} color={latencyColor(llm?.e2eLatency ?? NaN)} />
            <CompactCard label="ITL p95" value={fmtNum(llm?.itlP95Seconds, 3, "s")} />
            <CompactCard label="TTFT p95" value={fmtNum(llm?.ttftP95Seconds, 3, "s")} />
            <CompactCard label="E2E p95" value={fmtNum(llm?.e2eP95Seconds, 2, "s")} />
            {/* Slot metrics */}
            <CompactCard label="Running" value={fmtInt(runningSlots)} color={runningSlots > 0 ? "var(--color-success)" : "var(--color-muted)"} />
            <CompactCard label="Waiting" value={fmtInt(waitingSlots)} color={waitingSlots > 0 ? "var(--color-accent)" : "var(--color-muted)"} />
            <CompactCard label="Preempts" value={fmtInt(llm?.preemptionsTotal)} />
            <CompactCard label="Slots Active" value={fmtInt(llm?.slotsActive)} />
            <CompactCard label="Slots Total" value={fmtInt(llm?.slotsTotal)} />
            {/* Token metrics */}
            <CompactCard label="Total Tokens" value={fmtInt(llm?.totalTokensDecoded ?? llm?.totalOutputTokens)} />
            <CompactCard label="Tokens Decoded" value={fmtInt(llm?.totalTokensDecoded ?? llm?.totalOutputTokens)} />
            <CompactCard label="Tokens Output" value={fmtInt(llm?.totalOutputTokens)} />
            <CompactCard label="Prefill Cached" value={fmtInt(llm?.prefillCached)} />
            <CompactCard label="Prefill Computed" value={fmtInt(llm?.prefillComputed)} />
            {/* Spec decode metrics */}
            <CompactCard label="Spec Accept" value={pct(avgSpecAccept, 0)} color={mtpColor(avgSpecAccept)} />
            <CompactCard label="Spec Hits" value={fmtInt(specHits)} />
            <CompactCard label="Spec Drafts" value={fmtInt(specDrafts)} />
            <CompactCard label="MTP Accepted" value={fmtInt(mtpAccepted)} />
            <CompactCard label="MTP Drafted" value={fmtInt(mtpDrafted)} />
            <CompactCard label="DSpark Ratio" value={pct(llm?.dsparkAcceptRatio, 1)} color={mtpColor(llm?.dsparkAcceptRatio)} />
            <CompactCard label="MTP Rate" value={pct(llm?.mtpAcceptanceRate, 1)} color={mtpColor(llm?.mtpAcceptanceRate)} />
            {/* Throughput metrics */}
            <CompactCard label="Decode tok/s" value={fmtNum(displayGenTps, 1)} color={tpsColor(displayGenTps)} />
            <CompactCard label="Prefill tok/s" value={fmtNum(displayPrefillTps, 1)} color={tpsColor(displayPrefillTps)} />
            <CompactCard label="Agg tok/s" value={fmtNum(displayAggTps, 1)} color={tpsColor(displayAggTps)} />
            <CompactCard label="Avg tok/s" value={fmtNum(displaySingleTps, 1)} color={tpsColor(displaySingleTps)} />
            <CompactCard label="Peak tok/s" value={fmtNum(displayPeak, 1)} color={tpsColor(displayPeak)} />
            <CompactCard label="Per-stream Hi" value={fmtNum(llm?.perStreamHigh, 1)} />
            <CompactCard label="Per-stream Lo" value={fmtNum(llm?.perStreamLow, 1)} />
            {/* Cache / memory metrics */}
            <CompactCard label="Prefix Hit" value={pct(prefixHit, 0)} color={prefixHit != null ? "var(--color-success)" : "var(--color-muted)"} />
            <CompactCard label="KV Cache" value={pct(kvUsage, 1)} color={kvColor} />
            <CompactCard label="Banks Live" value={fmtInt(banksLive)} color={(banksLive ?? 0) > 0 ? "var(--color-success)" : "var(--color-muted)"} />
            <CompactCard label="Banks Total" value={fmtInt(banksTotal)} />
            <CompactCard label="KV Pages" value={fmtInt(llm?.kvPagesResident)} />
            <CompactCard label="Warm Records" value={fmtInt(llm?.warmRecords)} />
            <CompactCard label="GPU Mem" value={pct(llm?.gpuMemoryUtilization, 1)} color={llm?.gpuMemoryUtilization != null && llm.gpuMemoryUtilization >= 0.9 ? "var(--color-danger)" : llm?.gpuMemoryUtilization != null && llm.gpuMemoryUtilization >= 0.7 ? "var(--color-accent)" : "var(--color-success)"} />
            {/* Context metrics */}
            <CompactCard label="Active Context" value={fmtInt(llm?.activeContext)} />
            <CompactCard label="Context Used" value={llm?.contextUsedBytes != null ? `${(llm.contextUsedBytes / 1e9).toFixed(1)}GB` : nCtx != null ? `${fmtInt(contextUsedTokens)} tok` : "\u2014"} />
            {/* llama.cpp /slots metrics (data-availability gated) */}
            <CompactCard label="Prompt Tokens" value={fmtInt(promptTokens)} />
            <CompactCard label="Prompt Processed" value={fmtInt(promptTokensProcessed)} />
            <CompactCard label="Prompt Cached" value={fmtInt(promptTokensCache)} />
            <CompactCard label="Cache Hit" value={pct(cacheHitRatio, 1)} color={cacheHitRatio != null ? (cacheHitRatio >= 0.7 ? "var(--color-success)" : cacheHitRatio >= 0.4 ? "var(--color-accent)" : "var(--color-warning)") : "var(--color-muted)"} />
            <CompactCard label="n_ctx" value={fmtInt(nCtx)} />
            <CompactCard label="Remaining" value={fmtInt(nRemain)} />
            <CompactCard label="Decoded" value={fmtInt(nDecoded)} />
            <CompactCard label="Processing" value={isProcessing ? "yes" : "no"} color={isProcessing ? "var(--color-success)" : "var(--color-muted)"} />
            <CompactCard label="Spec Types" value={speculativeTypes ?? "\u2014"} />
            <CompactCard label="Reasoning Fmt" value={reasoningFormat ?? "\u2014"} />
            <CompactCard label="Chat Fmt" value={chatFormat ?? "\u2014"} />
            <CompactCard label="Samplers" value={samplers && samplers.length > 0 ? samplers.join(",") : "\u2014"} />
            {/* llama.cpp spec-decode (from server log) */}
            <CompactCard label="Spec Accept Rate" value={pct(specAcceptanceRate, 1)} color={mtpColor(specAcceptanceRate)} />
            <CompactCard label="Spec Accepted" value={fmtInt(specAcceptedTokens)} />
            <CompactCard label="Spec Generated" value={fmtInt(specGeneratedTokens)} />
            <CompactCard label="Spec Mean Len" value={fmtNum(specMeanLen, 1)} />
            {/* Sampling params */}
            <CompactCard label="Temp" value={samplingParams?.temperature != null ? fmtNum(Number(samplingParams.temperature), 2) : "\u2014"} />
            <CompactCard label="Top-K" value={samplingParams?.top_k != null ? fmtInt(Number(samplingParams.top_k)) : "\u2014"} />
            <CompactCard label="Top-P" value={samplingParams?.top_p != null ? fmtNum(Number(samplingParams.top_p), 2) : "\u2014"} />
            <CompactCard label="Min-P" value={samplingParams?.min_p != null ? fmtNum(Number(samplingParams.min_p), 2) : "\u2014"} />
            <CompactCard label="Max Tokens" value={samplingParams?.max_tokens != null ? fmtInt(Number(samplingParams.max_tokens)) : "\u2014"} />
            {/* Engine metrics */}
            <CompactCard label="Uptime" value={fmtUptime(llm?.engineUptimeSeconds ?? llm?.ds4Uptime ?? llm?.recipeInfo?.uptime)} />
            <CompactCard label="Reasoning" value={reasoningEffort ?? "\u2014"} color={effortIndex > 0 ? "var(--color-accent)" : "var(--color-muted)"} />
            <CompactCard label="Requests Started" value={fmtInt(llm?.requestsStarted)} />
            <CompactCard label="Requests Done" value={fmtInt(llm?.requestsCompleted)} />
            <CompactCard label="Requests Failed" value={fmtInt(llm?.requestsFailed)} />
            <CompactCard label="Inflight" value={fmtInt(llm?.requestsInflight)} />
            <CompactCard label="Decode Steps" value={fmtInt(llm?.decodeSteps)} />
            <CompactCard label="Tok/Step" value={fmtNum(llm?.tokPerStep, 2)} />
          </div>

          {/* ═══ ROW 7 — ACTIONS ═══ */}
          <div className="llm-actions">
            <button type="button" onClick={() => setBenchOpen(true)} className="llm-action-btn">Run decode benchmark</button>
            <button type="button" onClick={() => { const params = new URLSearchParams(); if (llmPort) params.set("port", String(llmPort)); if (llm?.modelId) params.set("model", llm.modelId); const q = params.toString() ? `?${params.toString()}` : ""; window.open(`/showcase/${encodeURIComponent(sparkId)}${q}`, "_blank", "noopener,noreferrer"); }} className="llm-action-btn">Showcase</button>
          </div>
        </div>

        {/* ═══ UPSTREAM 2.0 MONITORING (engine tiles + charts) — at top of dashboard after row cards ═══ */}

          <div className="sp-decode">
            <div className="sp-metric">
              <span className="eyebrow">Decode</span>
              <div className="big-num sp-big-lg">
                {fmtTps(generationTps)}
                <small>tok/s</small>
              </div>
              <TrendLine data={genHistory} height={44} color="var(--color-accent)" />
              {genAvg != null && <span className="sp-avg mono">avg {fmtAvg(genAvg)}</span>}
              {idleNote && (
                <span className="sp-avg" data-llm-idle>
                  {idleNote}
                </span>
              )}
            </div>
            <div
              className="sp-metric"
              title="Prompt tokens/sec taken in during the last poll window — cache-served + computed. Opening a saved chat in the UI does not hit the GPU; send (or regenerate) so the history is sent as the prompt. Cached prefill does little GPU work; uncached prefill is what builds KV cache."
            >
              <span className="eyebrow">Prefill</span>
              <div className="big-num sp-big-lg">
                {llm?.prefillActive && prefillTps <= 0 ? (
                  <>
                    <span className="ov-tps__dots" aria-hidden />
                    <small title="A prompt is being processed. The engine reports its speed only when the request finishes.">prefilling</small>
                  </>
                ) : (
                  <>
                    {fmtTps(prefillTps)}
                    <small>tok/s</small>
                  </>
                )}
              </div>
              <TrendLine data={prefillHistory} height={44} color="var(--color-info)" />
              {prefillAvg != null && <span className="sp-avg mono">avg {fmtAvg(prefillAvg)}</span>}
            </div>
          </div>
          {showPrefillSplit && (
            <div className="sp-decode sp-decode--split">
              <div
                className="sp-metric"
                title="Prefill tokens served from prefix cache (little GPU work). High values mean prompt reuse, not a faster cold prefill."
              >
                <span className="eyebrow">Cached prefill</span>
                <div className="big-num sp-big-md">
                  {fmtTps(cachedPrefillTps)}
                  <small>tok/s</small>
                </div>
                <TrendLine data={cachedPrefillHistory} height={30} color="var(--color-muted)" />
                {cachedPrefillAvg != null && <span className="sp-avg mono">avg {fmtAvg(cachedPrefillAvg)}</span>}
              </div>
              <div
                className="sp-metric"
                title="Uncached (computed) prefill — tokens that actually build KV cache on the GPU."
              >
                <span className="eyebrow">Uncached prefill</span>
                <div className="big-num sp-big-md">
                  {fmtTps(uncachedPrefillTps)}
                  <small>tok/s</small>
                </div>
                <TrendLine data={uncachedPrefillHistory} height={30} color="var(--color-violet)" />
                {uncachedPrefillAvg != null && <span className="sp-avg mono">avg {fmtAvg(uncachedPrefillAvg)}</span>}
              </div>
            </div>
          )}

          <div className="sp-tiles">
            {showEngineTiles && kvValue != null && tile("kvCache", "KV cache", kvValue, { tone: kvTone, sub: kvSub || undefined })}
            {showEngineTiles &&
              llm?.requestsRunning != null &&
              tile(
                "requests",
                "Running / waiting",
                `${Math.round(llm.requestsRunning)} / ${llm.requestsWaiting != null ? Math.round(llm.requestsWaiting) : "—"}`,
                { align: "right" }
              )}
            {isFreeToken && llm?.ttftSeconds != null && tile("ttftMean", "TTFT mean", `${Math.round(llm.ttftSeconds * 1000)} ms`)}
            {showEngineTiles && llm?.ttftP95Seconds != null && tile("ttftP95", "TTFT p95", `${Math.round(llm.ttftP95Seconds * 1000)} ms`)}
            {showEngineTiles &&
              llm?.preemptionsTotal != null &&
              tile("preempts", "Preempts", Math.round(llm.preemptionsTotal).toLocaleString(), { align: "right" })}
            {showEngineTiles &&
              llm?.prefixCacheHitRate != null &&
              tile("prefixCache", "Prefix hit", `${(llm.prefixCacheHitRate * 100).toFixed(0)}%`)}
            {showEngineTiles && llm?.e2eP95Seconds != null && tile("e2eP95", "E2E p95", `${llm.e2eP95Seconds.toFixed(2)}s`, { align: "right" })}
            {showEngineTiles && llm?.itlP95Seconds != null && tile("itlP95", "Inter-token p95", `${Math.round(llm.itlP95Seconds * 1000)} ms`)}
            {showEngineTiles &&
              llm?.mtpAcceptanceRate != null &&
              tile("mtpAccept", "MTP accept", llm.mtpAcceptanceRate.toFixed(2), { align: "right" })}
            <div className="sp-tile">
              <b>
                {(llm?.slotsTotal ?? 0) > 0
                  ? `${llm?.slotsActive ?? 0} / ${llm?.slotsTotal ?? 0}`
                  : (llm?.slotsActive ?? 0) > 0
                    ? `${llm?.slotsActive} running`
                    : "—"}
              </b>
              <span>Slots</span>
            </div>
            <div className="sp-tile">
              <b>{llm?.contextLength ? llm.contextLength.toLocaleString() : "—"}</b>
              <span>Context</span>
            </div>
            <div className="sp-tile">
              {(() => {
                const engine = engineStateLabel(llm);
                return (
                  <b className={engine.muted ? "text-muted" : undefined} title={engine.title}>
                    {engine.text}
                  </b>
                );
              })()}
              <div className="sp-tile__label">
                <span>Engine</span>
                <button
                  type="button"
                  onClick={() => {
                    setEngineInfoOpen((v) => {
                      if (!v) startEngineInfoTimer();
                      return !v;
                    });
                  }}
                  onMouseEnter={clearEngineInfoTimer}
                  onMouseLeave={startEngineInfoTimer}
                  className="relative cursor-pointer opacity-60 hover:opacity-100"
                  aria-label="Engine state info"
                >
                  <InfoIcon className="h-2.5 w-2.5" />
                  {engineInfoOpen && (
                    <div
                      onMouseEnter={clearEngineInfoTimer}
                      onMouseLeave={startEngineInfoTimer}
                      className="absolute left-0 top-full z-10 mt-1 w-56 rounded-xl border border-border-strong bg-surface-elevated px-3 py-2 text-left text-[11px] font-normal normal-case text-text shadow-lg"
                    >
                      Active = processing or ready for requests. Sleeping = idle, GPU memory freed until next request.
                    </div>
                  )}
                </button>
              </div>
            </div>
            <div className="sp-tile" title={ENGINE_GENERATED_TITLE}>
              <b>{llm && llm.totalOutputTokens > 0 ? llm.totalOutputTokens.toLocaleString() : "—"}</b>
              <span>{ENGINE_GENERATED_LABEL}</span>
            </div>
          </div>

          <LlmLaunchers
            sparkId={sparkId}
            llmPort={llmPort}
            modelId={llm?.modelId}
            onLaunch={launchLocal}
            onRemoteLaunch={launchRemote}
          />
          {llm?.posture && (
            <div className="sp-chips">
              <PostureBadge posture={llm.posture} />
            </div>
          )}
          <LlmDailyChart sparkId={sparkId} llmPort={llmPort} />
          <LlmTrendChart sparkId={sparkId} llmPort={llmPort} />
          <LlmTokenTotals sparkId={sparkId} llmPort={llmPort} />
        </div>
      )}

      <BenchmarkDialog
        open={openBench === "decode"}
        onClose={() => setOpenBench(null)}
        onSwitchBench={setOpenBench}
        sparkId={sparkId}
        llmPort={llmPort}
        modelId={remoteTarget ? null : llm?.modelId ?? null}
        models={remoteTarget || (available && llm?.backend === "llama.cpp") ? undefined : llm?.models}
        remoteTarget={remoteTarget}
        shareImage={shareImage}
        sparkName={sparkName ?? null}
        engine={remoteTarget ? null : llm?.backend ?? null}
        posture={remoteTarget ? null : llm?.posture ?? null}
        liveTps={remoteTarget ? null : llm?.generationTps ?? null}
      />
      <PrefillBenchDialog
        open={openBench === "prefill"}
        onClose={() => setOpenBench(null)}
        onSwitchBench={setOpenBench}
        sparkId={sparkId}
        llmPort={llmPort}
        modelId={remoteTarget ? null : llm?.modelId ?? null}
        contextLength={remoteTarget ? null : llm?.contextLength ?? null}
        remoteTarget={remoteTarget}
        shareImage={shareImage}
        sparkName={sparkName ?? null}
        engine={remoteTarget ? null : llm?.backend ?? null}
        posture={remoteTarget ? null : llm?.posture ?? null}
      />
      <QualityBenchDialog
        open={openBench === "quality"}
        onClose={() => setOpenBench(null)}
        onSwitchBench={setOpenBench}
        sparkId={sparkId}
        llmPort={llmPort}
        modelId={remoteTarget ? null : llm?.modelId ?? null}
        contextLength={remoteTarget ? null : llm?.contextLength ?? null}
        remoteTarget={remoteTarget}
        sparkName={sparkName ?? null}
        engine={remoteTarget ? null : llm?.backend ?? null}
        posture={remoteTarget ? null : llm?.posture ?? null}
      />
    </Panel>
  );
}
