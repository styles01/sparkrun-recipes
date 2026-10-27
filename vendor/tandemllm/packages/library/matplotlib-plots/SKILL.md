---
name: matplotlib-plots
description: Making clear charts with matplotlib (line, bar, scatter, histogram, multi-panel) saved to PNG or SVG - figure sizing, labels, units, legends, log axes, and checking the output image.
---
# matplotlib plots

Rules:
- Object interface: `fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)`; never the implicit `plt.plot` state in library code.
- Every axis has a label with a unit; the figure has a title; a legend only when there are 2 or more series. `ax.grid(True, alpha=.3)`.
- Pick the chart by the question: trend over time = line; compare categories = bar (sorted); distribution = histogram with a stated bin count; relation = scatter.
- Save with `fig.savefig(path, bbox_inches="tight")`; `plt.close(fig)` in loops. Headless: `matplotlib.use("Agg")` before importing pyplot.

What goes wrong:
- Two series on wildly different scales on one axis; use `twinx()` or two panels or a log axis, and say which.
- The legend covers the data; `loc="best"` or outside with `bbox_to_anchor`.
- Thousands of points with default markers; `s=4, alpha=.4` or `hexbin`.

Check: save the file, print its path and size in bytes, open it with the image tool, and compare one labelled value against the source data.

Done when:
- [ ] The PNG or SVG path with byte size in the result and the image viewed.
- [ ] Axis labels with units, title and legend present (visible in the image).
- [ ] One plotted value cross-checked against the data array (printed side by side).

## Reference

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
def line_chart(x, ys: dict, path, title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    for name, y in ys.items():
        ax.plot(x, y, label=name, linewidth=1.5)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.grid(True, alpha=.3); ax.legend(loc="best")
    if np.issubdtype(np.asarray(x).dtype, np.datetime64):
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d")); fig.autofmt_xdate()
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path
```
Bar, sorted: `order = np.argsort(values)[::-1]; ax.bar(np.array(labels)[order], np.array(values)[order]); ax.tick_params(axis="x", rotation=45)`.
Histogram: `ax.hist(data, bins=30, edgecolor="white"); ax.set_ylabel("count")`.
Log axis: `ax.set_yscale("log")`. Two panels: `fig, (a, b) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)`.
Annotate the max: `i = int(np.argmax(y)); ax.annotate(f"{y[i]:.1f}", (x[i], y[i]), textcoords="offset points", xytext=(0, 6), ha="center")`.
Verify: `import os; print(path, os.path.getsize(path))` then view the file.

More rules:
- Check the image: open the PNG after saving, read its size, and look at it. Numbers on the plot must match the data (spot-check one point).
- Dates on the x axis: `fig.autofmt_xdate()` and a `DateFormatter`.

More pitfalls:
- Text unreadable at the final size; set `figsize` for the target width and 150 dpi or more.
