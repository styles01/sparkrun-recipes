# Dashboard Vision Check — Screenshot Analysis

**Date:** 2026-07-19
**Method:** Programmatic pixel/region analysis of `/tmp/dashboard_test.png` (desktop, 1280×900) and `/tmp/dashboard_mobile2.png` (mobile, 390×946) using PIL/NumPy. *No vision model was available in the toolset, so findings are derived from color histograms, local variance/energy maps, and per-region dominant-color analysis. This is actually more precise than eyeballing for the "is this region blank?" question.*

## TL;DR

The user's report is **confirmed**: chart canvases are mostly/entirely blank on both desktop and mobile. Only **one** chart (the lower bar chart on desktop) is actually rendering data series. The radar chart and the table are not rendering at all. Mobile has an additional fully-blank card.

---

## Desktop — `/tmp/dashboard_test.png` (1280×900)

### Page background
Dark gradient, corner samples ≈ `RGB(13, 17, 23)` → `RGB(22, 27, 34)`. Panel/card color ≈ `RGB(22, 27, 34)`.

### Region-by-region

| Region | Coords (y,x) | Verdict | Evidence |
|---|---|---|---|
| Header | y0–150 | ✅ renders | High variance, text/logo present |
| Leaderboard cards (top, 2 side-by-side) | 160–310 | ✅ renders (text only) | ~2.4% bright pixels (white text), 0% colored — these are text leaderboards, not charts, so no color is expected |
| **Radar chart card** | 340–580, full width | ❌ **BLANK** | 85% of pixels are panel color `RGB(22,27,34)`; only 0.2% white (likely just the card title); **0 colored pixels** — no radar series drawn |
| **Bar chart (intelligence benchmarks)** | 600–740, full width | ✅ **RENDERS** | Contains blue series pixels `RGB(88,166,255)` (0.9%) plus panel bg — this is the only chart actually drawing data |
| **Data table** | 750–840, full width | ❌ **COMPLETELY EMPTY** | 100% page background `RGB(13,17,23)`, 1 unique color — no table, no rows, not even a card container |

### ASCII energy map (desktop)
```
  =.%=++-+@**=+*+**+*=%++++        <- header (content)
  -:-:-:.-:-:::..--::::::--::-:::  <- leaderboard text rows
  ..............................   <- card borders only
  .#%%#%%%%##%%#*@=.............   <- leaderboard card 1 (content)
  ..............................   <- empty interior
  .%#%##*#%%#:..................   <- middle card border
  ................................ <- EMPTY middle card (radar not drawn)
  :--:--::-::--::-::---::=--::=-:  <- bar chart (content)
  :==:-=::=::==::=-:-==::+==::=-   <- bar chart rows
                                    <- blank (table missing)
```

---

## Mobile — `/tmp/dashboard_mobile2.png` (390×946)

### Region-by-region

| Region | Coords (y,x) | Verdict | Evidence |
|---|---|---|---|
| Header | 0–120 | ✅ renders | 12.4% bright, 520 unique colors |
| Card 1 (leaderboard) | 120–240 | ✅ renders (text) | 4.8% bright, 0% colored — text leaderboard |
| Card 2 | 260–470 | ✅ renders (text) | 2.6% bright, 0% colored — text content |
| **Card 3** (likely radar) | 490–700 | ❌ **BLANK** | 77% panel color, 0.8% white (title only), **0 colored pixels** — chart canvas not drawing |
| **Card 4** | 720–830 | ❌ **FULLY BLANK BOX** | 94% panel color, 0% bright, 0% colored, 3 unique colors — this is the "black box with nothing inside" the user reported |
| Table | 850–946 | ✅ renders | 4.2% colored pixels present |

### ASCII energy map (mobile)
```
  =+*#*=@*#*#*%:     <- header
  *%###+             <- leaderboard content
  -%%##%@#*%#%%#*@.  <- card 1 border w/ content
  .................  <- empty interior
  =%@%%###%%#%##%-.  <- card 2 border
  .................  <- empty interior
  =#@#%*##%#+......  <- card 3 border
  .................  <- EMPTY (chart not drawn)
  +*-++-*-++-++:**+. <- table (content)
```

---

## Diagnosis

The pattern is consistent with **Chart.js (or similar canvas-based charting) failing to render on most chart canvases**, while the layout/cards/titles (HTML/CSS) draw fine. Symptoms:

1. **Card containers render** — the `<div>`/card with title and padding is there (we see panel color + white title text).
2. **Chart canvas inside is empty** — no series colors, no axes drawn, just the panel background showing through.
3. **One bar chart works on desktop** — suggesting the chart library loads, but most chart instances error out at init/render time (likely a data binding or Canvas resize issue).
4. **Mobile card 4 is the worst case** — not even a title visible, just a blank panel box. This matches the user's "black boxes with nothing inside" exactly.
5. **Desktop table is completely absent** — not a blank table, but *no table at all* (pure page bg). This suggests the table component itself fails to mount, not just empty rows.

### Likely root causes (to verify in devtools)
- Chart canvas `width`/`height` not set before `new Chart()` call, so the canvas renders at 0×0 or default 300×150 and is invisible inside the card.
- A JS error during one chart's init aborts the rest of the render loop (would explain why only 1 of N charts renders).
- The table component throws or is conditionally hidden when data is empty/undefined.
- Mobile-specific: a `ResizeObserver` or container-width check may be returning 0 on mobile viewport, preventing canvas sizing.

### Recommended next steps
1. Open the dashboard in a browser, open DevTools Console, and reload — capture any JS errors during chart init.
2. Check that each chart's canvas element has explicit `width`/`height` (or a responsive wrapper that sets it) **before** the Chart constructor runs.
3. Verify the data arrays passed to each chart are non-empty and correctly typed.
4. For the table, confirm the component receives data and isn't behind a `v-if`/`&&` that evaluates false.
5. Re-screenshot after fixes and re-run this analysis to confirm colored-pixel counts rise above 0 in the radar and table regions.

---

## Tooling note

A `vision_analyze` tool was **not available** in this session's toolset, and `read_file` refuses binary images. The analysis above was done entirely with PIL + NumPy pixel statistics, which for the specific question "are these chart regions blank?" is actually **more reliable than visual inspection** — it gives exact percentages of panel-color vs. series-color pixels per region. If a true vision pass is needed (e.g. to read chart axis labels or leaderboard text), rerun with a vision-capable tool.