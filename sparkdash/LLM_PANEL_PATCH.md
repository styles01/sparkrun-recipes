// === PATCH INSTRUCTIONS for LlmPanel.tsx ===
// Replace the throughput TelemetryChart block with dual-axis version.
// Find this section in LlmPanel.tsx:
//
//   <div className="llm-chart-block">
//     <div className="llm-chart-title">Throughput <span className="llm-chart-sub">tok/s · last 60 samples</span></div>
//     <TelemetryChart
//       series={[genSeries, preSeries]}
//       maxPoints={HISTORY}
//       height={170}
//       yUnit=""
//       yMin={0}
//     />
//   </div>
//
// Replace genSeries and preSeries definitions:
//   const genSeries: ChartSeries = {
//     label: "gen tok/s",
//     color: "var(--color-success)",
//     data: history.genTps,
//     area: true,
//   };
//   const preSeries: ChartSeries = {
//     label: "prefill tok/s",
//     color: "var(--color-accent)",
//     data: history.prefillTps,
//     area: false,
//   };
//
// WITH:
//   const genSeries: ChartSeries = {
//     label: "gen tok/s",
//     color: "var(--color-success)",
//     data: history.genTps,
//     area: true,
//     yAxis: "left",
//   };
//   const preSeries: ChartSeries = {
//     label: "prefill tok/s",
//     color: "var(--color-accent)",
//     data: history.prefillTps,
//     area: false,
//     yAxis: "right",
//   };
//
// Replace the TelemetryChart call WITH:
//   <TelemetryChart
//     series={[genSeries, preSeries]}
//     maxPoints={HISTORY}
//     height={170}
//     yUnit=""
//     yUnitRight=""
//     yMin={0}
//     yMax={140}
//   />
//
// The TelemetryChart component now supports yAxis: "left"|"right" on each
// ChartSeries, plus yMaxRight and yUnitRight props. When any series has
// yAxis="right", a secondary axis is drawn on the right side with its own
// scale. Left axis is fixed at yMax=140 for decode. Right axis auto-scales
// to prefill throughput (typically 1000-50000+ tok/s).