import { useEffect, useRef } from "react";

/**
 * TelemetryChart — zero-dependency canvas scrolling line chart.
 *
 * Renders one or more named series over a shared time axis (last N samples).
 * Designed for the SparkDash LLM telemetry dashboard: dark-theme aware via
 * CSS custom properties, high-DPI crisp, smooth requestAnimationFrame redraw.
 *
 * Each series: { label, color (CSS var or hex), data: number[] }
 * All series share the same x positions; missing trailing values are skipped.
 *
 * Supports dual y-axis: series with yAxis: 'right' are scaled to a secondary
 * axis drawn on the right side. Left-axis series use the primary (left) axis.
 */

export interface ChartSeries {
  label: string;
  color: string;
  data: number[];
  /** Optional: draw a translucent area-fill under the line. */
  area?: boolean;
  /** Which y-axis to scale against. Default 'left'. */
  yAxis?: "left" | "right";
}

export interface TelemetryChartProps {
  series: ChartSeries[];
  /** Max number of samples shown (the x-axis window). */
  maxPoints?: number;
  /** Fixed y-axis maximum (left axis); auto-scales from data when omitted. */
  yMax?: number;
  /** Fixed y-axis maximum for the right axis; auto-scales from data when omitted. */
  yMaxRight?: number;
  /** Fixed y-axis minimum (default 0, applies to both axes). */
  yMin?: number;
  /** Height in CSS pixels (canvas is sized to container width). */
  height?: number;
  /** Label for the y-axis values, e.g. "tok/s" or "s". */
  yUnit?: string;
  /** Label for the right y-axis values (shown in right ticks). */
  yUnitRight?: string;
  /** Show a faint legend top-left. */
  legend?: boolean;
  /** ClassName passthrough. */
  className?: string;
}

const PAD = { left: 44, right: 12, top: 14, bottom: 22 };
const PAD_DUAL = { left: 44, right: 44, top: 14, bottom: 22 };

function readCssVar(name: string, fallback: string): string {
  if (typeof window === "undefined") return fallback;
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || fallback;
}

interface ResolvedColors {
  text: string;
  muted: string;
  grid: string;
  border: string;
  surface: string;
}

function resolveColors(): ResolvedColors {
  return {
    text: readCssVar("--color-text", "#e4e4e4"),
    muted: readCssVar("--color-muted", "#8a8a8a"),
    grid: readCssVar("--color-grid", "rgba(255,255,255,0.04)"),
    border: readCssVar("--color-border", "#353535"),
    surface: readCssVar("--color-surface-elevated", "#262626"),
  };
}

function niceMax(max: number): number {
  if (max <= 0) return 1;
  const pow = Math.pow(10, Math.floor(Math.log10(max)));
  const n = max / pow;
  let step: number;
  if (n <= 1) step = 1;
  else if (n <= 2) step = 2;
  else if (n <= 5) step = 5;
  else step = 10;
  return step * pow;
}

export function TelemetryChart({
  series,
  maxPoints = 60,
  yMax,
  yMaxRight,
  yMin = 0,
  height = 160,
  yUnit = "",
  yUnitRight = "",
  legend = true,
  className = "",
}: TelemetryChartProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const rafRef = useRef<number | null>(null);
  // Keep latest props in a ref so the rAF loop reads fresh data without restarting.
  const propsRef = useRef({ series, maxPoints, yMax, yMaxRight, yMin, yUnit, yUnitRight, legend });
  propsRef.current = { series, maxPoints, yMax, yMaxRight, yMin, yUnit, yUnitRight, legend };

  useEffect(() => {
    const canvas = canvasRef.current;
    const wrap = wrapRef.current;
    if (!canvas || !wrap) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const draw = () => {
      const dpr = Math.max(1, window.devicePixelRatio || 1);
      const cssW = wrap.clientWidth || 320;
      const cssH = height;
      const pxW = Math.round(cssW * dpr);
      const pxH = Math.round(cssH * dpr);
      if (canvas.width !== pxW || canvas.height !== pxH) {
        canvas.width = pxW;
        canvas.height = pxH;
        canvas.style.width = `${cssW}px`;
        canvas.style.height = `${cssH}px`;
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, cssW, cssH);

      const colors = resolveColors();
      const {
        series: ss,
        maxPoints: mp,
        yMax: fixedMax,
        yMaxRight: fixedMaxRight,
        yMin: lo,
        yUnit: unit,
        yUnitRight: unitRight,
        legend: showLegend,
      } = propsRef.current;

      // Check if any series uses the right axis
      const hasRightAxis = ss.some((s) => s.yAxis === "right");
      const pad = hasRightAxis ? PAD_DUAL : PAD;

      // Determine y range for LEFT axis (series without yAxis='right')
      let leftDataMax = -Infinity;
      let leftDataMin = Infinity;
      for (const s of ss) {
        if (s.yAxis === "right") continue;
        for (const v of s.data) {
          if (Number.isFinite(v)) {
            if (v > leftDataMax) leftDataMax = v;
            if (v < leftDataMin) leftDataMin = v;
          }
        }
      }
      if (!Number.isFinite(leftDataMax)) leftDataMax = 1;
      if (!Number.isFinite(leftDataMin)) leftDataMin = 0;
      const yTopLeft = fixedMax != null ? fixedMax : niceMax(Math.max(leftDataMax, 1));
      const yBotLeft = lo;
      const ySpanLeft = yTopLeft - yBotLeft || 1;

      // Determine y range for RIGHT axis (series with yAxis='right')
      let rightDataMax = -Infinity;
      let rightDataMin = Infinity;
      for (const s of ss) {
        if (s.yAxis !== "right") continue;
        for (const v of s.data) {
          if (Number.isFinite(v)) {
            if (v > rightDataMax) rightDataMax = v;
            if (v < rightDataMin) rightDataMin = v;
          }
        }
      }
      if (!Number.isFinite(rightDataMax)) rightDataMax = 1;
      if (!Number.isFinite(rightDataMin)) rightDataMin = 0;
      const yTopRight = fixedMaxRight != null ? fixedMaxRight : niceMax(Math.max(rightDataMax, 1));
      const yBotRight = lo;
      const ySpanRight = yTopRight - yBotRight || 1;

      const plotX0 = pad.left;
      const plotX1 = cssW - pad.right;
      const plotY0 = pad.top;
      const plotY1 = cssH - pad.bottom;
      const plotW = plotX1 - plotX0;
      const plotH = plotY1 - plotY0;

      // Grid + LEFT y-axis ticks
      ctx.font = "10px ui-monospace, 'JetBrains Mono', monospace";
      ctx.textBaseline = "middle";
      ctx.textAlign = "right";
      const ticks = 4;
      for (let i = 0; i <= ticks; i++) {
        const frac = i / ticks;
        const y = plotY1 - frac * plotH;
        const val = yBotLeft + frac * ySpanLeft;
        ctx.strokeStyle = colors.grid;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(plotX0, y);
        ctx.lineTo(plotX1, y);
        ctx.stroke();
        ctx.fillStyle = colors.muted;
        ctx.fillText(formatTick(val, unit), plotX0 - 6, y);
      }

      // RIGHT y-axis ticks (only when dual-axis)
      if (hasRightAxis) {
        ctx.textAlign = "left";
        for (let i = 0; i <= ticks; i++) {
          const frac = i / ticks;
          const y = plotY1 - frac * plotH;
          const val = yBotRight + frac * ySpanRight;
          // No grid lines for right axis (avoid clutter), just labels
          ctx.fillStyle = colors.muted;
          ctx.fillText(formatTick(val, unitRight || unit), plotX1 + 6, y);
        }
        ctx.textAlign = "right"; // reset
      }

      // Plot frame
      ctx.strokeStyle = colors.border;
      ctx.lineWidth = 1;
      ctx.strokeRect(plotX0 + 0.5, plotY0 + 0.5, plotW, plotH);

      // X-axis time labels (oldest / mid / newest)
      ctx.fillStyle = colors.muted;
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      const now = Date.now();
      const xLabels = [0, 0.5, 1];
      for (const frac of xLabels) {
        const x = plotX0 + frac * plotW;
        const t = new Date(now - (1 - frac) * (mp - 1) * 2_000);
        const label = `${t.getHours().toString().padStart(2, "0")}:${t.getMinutes().toString().padStart(2, "0")}:${t.getSeconds().toString().padStart(2, "0")}`;
        ctx.fillText(label, x, plotY1 + 6);
      }

      // Draw each series
      for (const s of ss) {
        const data = s.data;
        if (data.length < 1) continue;
        const n = data.length;
        const isRight = s.yAxis === "right";
        const yTop = isRight ? yTopRight : yTopLeft;
        const yBot = isRight ? yBotRight : yBotLeft;
        const ySpan = yTop - yBot || 1;

        // Map data[i] -> x position within the window. Latest sample at right edge.
        const xFor = (i: number) => plotX0 + (mp <= 1 ? plotW : (i / (mp - 1)) * plotW);
        const yFor = (v: number) => plotY1 - ((Math.min(yTop, Math.max(yBot, v)) - yBot) / ySpan) * plotH;

        // Area fill
        if (s.area) {
          ctx.beginPath();
          let started = false;
          for (let i = 0; i < n; i++) {
            const v = data[i];
            if (!Number.isFinite(v)) continue;
            const x = xFor(i);
            const y = yFor(v);
            if (!started) {
              ctx.moveTo(x, plotY1);
              ctx.lineTo(x, y);
              started = true;
            } else {
              ctx.lineTo(x, y);
            }
          }
          if (started) {
            const lastX = xFor(n - 1);
            ctx.lineTo(lastX, plotY1);
            ctx.closePath();
            ctx.fillStyle = withAlpha(s.color, 0.16);
            ctx.fill();
          }
        }

        // Line
        ctx.beginPath();
        let started = false;
        for (let i = 0; i < n; i++) {
          const v = data[i];
          if (!Number.isFinite(v)) continue;
          const x = xFor(i);
          const y = yFor(v);
          if (!started) {
            ctx.moveTo(x, y);
            started = true;
          } else {
            ctx.lineTo(x, y);
          }
        }
        ctx.strokeStyle = resolveColor(s.color);
        ctx.lineWidth = 1.6;
        ctx.lineJoin = "round";
        ctx.lineCap = "round";
        ctx.stroke();

        // Last-point dot
        if (n > 0) {
          const lastV = data[n - 1];
          if (Number.isFinite(lastV)) {
            const x = xFor(n - 1);
            const y = yFor(lastV);
            ctx.beginPath();
            ctx.arc(x, y, 2.5, 0, Math.PI * 2);
            ctx.fillStyle = resolveColor(s.color);
            ctx.fill();
          }
        }
      }

      // Legend
      if (showLegend && ss.length > 0) {
        ctx.font = "10px ui-monospace, 'JetBrains Mono', monospace";
        ctx.textBaseline = "middle";
        ctx.textAlign = "left";
        let lx = plotX0 + 6;
        const ly = plotY0 + 8;
        for (const s of ss) {
          ctx.fillStyle = resolveColor(s.color);
          ctx.fillRect(lx, ly - 4, 8, 2);
          ctx.fillStyle = colors.text;
          const label = s.label;
          ctx.fillText(label, lx + 12, ly);
          lx += 14 + ctx.measureText(label).width + 14;
        }
      }
    };

    const loop = () => {
      draw();
      rafRef.current = requestAnimationFrame(loop);
    };
    rafRef.current = requestAnimationFrame(loop);

    const onResize = () => {
      // Canvas size handled in draw(); just trigger a redraw next frame.
    };
    window.addEventListener("resize", onResize);

    return () => {
      if (rafRef.current != null) cancelAnimationFrame(rafRef.current);
      window.removeEventListener("resize", onResize);
    };
  }, [height]);

  return (
    <div ref={wrapRef} className={`telemetry-chart-wrap ${className}`} style={{ height }}>
      <canvas ref={canvasRef} />
    </div>
  );
}

function formatTick(val: number, unit: string): string {
  let s: string;
  if (val >= 1000) s = (val / 1000).toFixed(1) + "k";
  else if (val >= 100) s = val.toFixed(0);
  else if (val >= 10) s = val.toFixed(0);
  else if (val >= 1) s = val.toFixed(1);
  else s = val.toFixed(2);
  return unit ? `${s}${unit}` : s;
}

function resolveColor(c: string): string {
  if (c.startsWith("var(")) {
    const m = c.match(/var\(\s*--([\w-]+)\s*\)/);
    if (m) return readCssVar(`--${m[1]}`, "#58a6ff");
    return "#58a6ff";
  }
  return c;
}

function withAlpha(c: string, alpha: number): string {
  const resolved = resolveColor(c);
  if (resolved.startsWith("#")) {
    const hex = resolved.slice(1);
    if (hex.length === 6) {
      const r = parseInt(hex.slice(0, 2), 16);
      const g = parseInt(hex.slice(2, 4), 16);
      const b = parseInt(hex.slice(4, 6), 16);
      return `rgba(${r},${g},${b},${alpha})`;
    }
    if (hex.length === 3) {
      const r = parseInt(hex[0] + hex[0], 16);
      const g = parseInt(hex[1] + hex[1], 16);
      const b = parseInt(hex[2] + hex[2], 16);
      return `rgba(${r},${g},${b},${alpha})`;
    }
  }
  if (resolved.startsWith("rgb")) {
    return resolved.replace(/rgba?\(([^)]+)\)/, (_m, inner) => {
      const parts = inner.split(",").map((p: string) => p.trim());
      return `rgba(${parts[0]},${parts[1]},${parts[2]},${alpha})`;
    });
  }
  return resolved;
}