# Chart grammar (house style)

Read with `package_read("references/chart-grammar.md")`. The dashboard draws its charts by hand
in SVG with these rules; a chart from this package must look like one of them.

## Forms and when each fits

| data shape | form | notes |
|-|-|-|
| one measure over ordered steps (time, context length, batch size) | line | points at every step, the value above each point when 12 or fewer points, else at the first, last, min and max |
| one measure per category (configs, models, cities) | bars | vertical for 8 or fewer short labels; horizontal when labels are long or there are more than 8 |
| two or more measures per category | grouped bars | at most 3 series; the group gap twice the bar gap |
| parts of a whole over steps | stacked bars | the total above each stack; no pie |
| a distribution of many values | histogram | 8 to 12 equal bins, counts above bars, edges as ticks |
| two measures per item | scatter | every item labelled when 15 or fewer; else the highlighted ones |
| a value per cell of a grid (day of year, hour x weekday) | heatmap | five-step ramp by quantiles of the non-zero cells |
| one number with context | stat tile | the number large, the label small, a delta or a sparkline beside it |
| a short trend beside a number | sparkline | 160x40, no axes, the last value orange |

## Colour

| role | hex | use |
|-|-|-|
| background | `#161d25` | the whole viewBox |
| surface | `#1c242d` | tiles, zero cells |
| grid | `#ffffff` at 6 % | horizontal rules only, no vertical grid for lines |
| text | `#eef2f7` | title, the big number |
| text 2 | `#93a1b3` | axis text, labels, units |
| cobalt | `#3987e5` | series 1, every bar that is not the highlight |
| cobalt ramp | `#1b2d44 #1c5cab #2a78d6 #3987e5 #6aa9f0` | heatmap, low to high |
| orange | `#d95926` | series 2 |
| orange highlight | `#f7a24f` | the one value the prompt asks about, today, the last point of a sparkline |
| aqua | `#2bb7a9` | series 3 |
| violet | `#8b6fe8` | series 4 |
| red | `#e25c5c` | an error or a failed state only |

Contrast: every text colour above reads at 4.5:1 or better on `#161d25`. Never put `#93a1b3`
text on a cobalt bar; put the value above the bar on the background instead.

## Numbers

- Monospace, tabular digits. `1 234` (thin space) not `1,234`; `12.5 %` with a space before the unit.
- The unit once, in the axis title or the chart title; never after every tick.
- One decimal unless the data has more; integers stay integers.
- A tick is a round number: 0, 25, 50, 75, 100; 0, 2k, 4k; never 0, 37, 74.
- Bars and histograms start at zero. A line axis may start above zero when the title says so ("y from 40").

## Labels

- The title says what, which unit, and which value is highlighted: "Throughput by config (req/s), config B highlighted".
- A point label sits above the point (`dy="-8"`); when two labels would overlap, move the second below (`dy="16"`).
- A bar label sits 6 px above the bar end (vertical) or 6 px right of it (horizontal), on the background.
- A long category label is wrapped at 14 characters or the chart turns horizontal.
- No legend for one series. For two or more, a legend row under the title: a 10x10 square and the name, 12 px.

## Layout

- `viewBox="0 0 800 450"`, `width="100%"`, so the chart scales to a phone.
- Plot area x 70 to 776, y 48 to 406. A horizontal bar chart moves the left edge to 180.
- Text 12 px or larger everywhere; the big number of a stat tile 32 px.
- No 3D, no shadows, no gradients on data marks, no rounded bars above `rx="2"`.

## Reading back

The automatic check `prompt_terms` reads every number of the prompt's data from the SVG's text
nodes. A number drawn as a tick alone is fine only when the tick equals the value exactly. A
number that is not in the text fails the chart, however it looks.
