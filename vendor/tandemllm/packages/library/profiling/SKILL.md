---
name: profiling
description: Finding why a program is slow before changing it - timing the real case, Python cProfile and py-spy, Node --cpu-prof, browser performance traces, picking the one hot spot, and proving the gain with numbers.
---
# Performance profiling

Rules:
- Measure the real case first: the real input size, the real command, three runs, the median. Write the number down. No optimization without it.
- Profile, then read the top 10 by cumulative and by self time. Fix the single largest item. Re-measure. Repeat. Guesses about "probably the loop" are not evidence.
- Separate the kinds: CPU (profile), I/O wait (timers around calls, async), memory (peak RSS, allocations), startup (imports).
- Change one thing per measurement; keep the output identical (compare a checksum).
- Stop when the target is met or the top item is under 10 % of the total; say which.

What goes wrong:
- Timing a 10-element input and extrapolating; the real cost is in a cache miss or a quadratic step that only shows at scale.
- Optimizing the profiler's top self-time function when the caller calls it a million times; fix the caller.
- A faster version with different output; check equality first.

Check: the before and after timing on the same command and input, the profiler's top lines before the change, and an output checksum equal on both.

Done when:
- [ ] Before and after medians of 3 runs with the exact command, in the result.
- [ ] The profiler's top 10 lines (before) attached and the fixed item named in them.
- [ ] Output checksum or test suite identical after the change.

## Reference

Python:
```bash
python -X importtime -c "import mytool" 2>&1 | sort -t'|' -k2 -n | tail -5      # startup
python -m cProfile -o /tmp/prof.out -m mytool.cli big.csv
python -c "import pstats; pstats.Stats('/tmp/prof.out').sort_stats('cumulative').print_stats(12)"
py-spy top -- python -m mytool.cli big.csv        # live sampling, no code change
py-spy record -o /tmp/flame.svg -- python -m mytool.cli big.csv
/usr/bin/time -l python -m mytool.cli big.csv 2>&1 | grep -E 'real|maximum resident'    # macOS; Linux: -v
```
In code: `t0 = time.perf_counter(); ...; log.info("ms=%.1f", (time.perf_counter()-t0)*1000)`.
Node:
```bash
node --cpu-prof --cpu-prof-dir=/tmp/prof src/server.js     # open the .cpuprofile in DevTools
node -e "console.time('x'); require('./lib').run(); console.timeEnd('x')"
```
Browser: `performance.mark('a'); ...; performance.measure('step', 'a'); performance.getEntriesByType('measure')`; or the DevTools performance trace tool (start, interact, stop, read the insight). Frame time: count frames over 3 s.
Median of 3: `for i in 1 2 3; do /usr/bin/time -p cmd 2>&1 | grep real; done`.
Checksum: `cmd | sha256sum` before and after.
SQL: `EXPLAIN (ANALYZE)` (see sql-queries-indexes).

More rules:
- Report before, after and the ratio, with the command used.

More pitfalls:
- Measuring with debug logging on or with the dev server's hot reload.
