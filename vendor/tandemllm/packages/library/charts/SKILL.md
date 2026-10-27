---
name: charts
description: A chart from data in the prompt (CSV, JSON or numbers in the text) as one SVG block in the house style - dark readout, cobalt series, signal-orange highlight, tabular numbers, units on the axes; line, bar, scatter, histogram, heatmap, sparkline, stat tile.
---
# Charts

Steps:
1. Copy the data first: every row as a table in your plan, in the prompt's order. A value you did not copy will be missing from the chart.
2. Pick the form: a trend, line; categories, bars (horizontal when labels are long); a distribution, histogram; two measures per item, scatter; a value per cell, heatmap; one number, stat tile.
3. Draw it by hand in SVG, `viewBox="0 0 800 450"`, no library. The answer is one `<svg>` block.
4. Every data value is on the chart as text at its point or bar end.

Style: background `#161d25`, grid `#ffffff` at 6 %, axis text `#93a1b3` 12 px, title `#eef2f7` 16 px bold; series cobalt `#3987e5`, then orange `#d95926`; the one value the prompt asks about is orange `#f7a24f`. Monospace numbers with tabular digits, thousands with a thin space. The unit once, on the axis. 4 to 6 round ticks, bars from 0. No 3D, gradients, or legend for one series.

Done when:
- [ ] Every value of the data appears as text in the SVG (count checked).
- [ ] The highlighted value is orange and named in the title.
- [ ] Text 12 px or larger, nothing overlaps at 800x450.

## Reference

Scales and frame (margins left 70, right 24, top 48, bottom 44):
```js
const X0 = 70, X1 = 776, Y0 = 406, Y1 = 48;                      // plot area in the 800x450 viewBox
const sx = v => X0 + (v - xmin) / (xmax - xmin) * (X1 - X0);
const sy = v => Y0 - (v - ymin) / (ymax - ymin) * (Y0 - Y1);
```
Frame, ticks, title (write the SVG by hand; this is the shape):
```svg
<svg viewBox="0 0 800 450" xmlns="http://www.w3.org/2000/svg" font-family="ui-monospace, Menlo, monospace" style="font-variant-numeric: tabular-nums">
  <rect width="800" height="450" fill="#161d25"/>
  <text x="70" y="30" fill="#eef2f7" font-size="16" font-weight="700">Decode speed by context length (tok/s)</text>
  <g stroke="#ffffff" stroke-opacity="0.06"><line x1="70" x2="776" y1="406" y2="406"/><line x1="70" x2="776" y1="316.5" y2="316.5"/></g>
  <g fill="#93a1b3" font-size="12"><text x="60" y="410" text-anchor="end">0</text><text x="60" y="320" text-anchor="end">25</text></g>
  <text x="423" y="440" fill="#93a1b3" font-size="12" text-anchor="middle">context (k tokens)</text>
</svg>
```
Line: `<path d="M x1 y1 L x2 y2 ..." fill="none" stroke="#3987e5" stroke-width="2.5" stroke-linejoin="round"/>` plus a `<circle r="3.5">` and a `<text>` with the value at each point (value above the point, `text-anchor="middle"`, `dy="-8"`).
Bars: width `(X1 - X0) / n * 0.7`, `<rect y="sy(v)" height="Y0 - sy(v)" rx="2" fill="#3987e5"/>`, the value as text 6 px above the bar; the highlighted bar `fill="#f7a24f"`. Horizontal bars when a label is over 12 characters.
Histogram: bins of equal width, bars touching (gap 1 px), the count above each bar, the bin edges as ticks.
Scatter: `<circle r="4" fill="#3987e5" fill-opacity="0.85">`, the label of each point at `dx="7"`; one highlighted point in orange with its name.
Heatmap: cells `<rect width="w" height="h" rx="2">` with a five-step cobalt ramp by quantiles `#1b2d44 #1c5cab #2a78d6 #3987e5 #6aa9f0`, zero as `#1c242d`; row and column labels 12 px; the value inside a cell when the cell is 28 px or wider; today or the asked cell outlined `stroke="#f7a24f" stroke-width="2"`.
Sparkline: `viewBox="0 0 160 40"`, one path, no axes, the last point orange with its value at the right.
Stat tile: a `<rect>` 180x96 `fill="#1c242d" rx="8"`, the label 12 px `#93a1b3` at the top, the number 32 px bold `#eef2f7` with its unit 14 px beside it, a delta line below in orange when positive is asked.
Two axes: the second series in orange with its own ticks on the right (`text-anchor="start"` at x 786) and its unit in the title: "Latency (ms, left) and throughput (req/s, right)".
Reference on request: `references/chart-grammar.md` (forms, when each fits, label placement rules, contrast).
