# ADR-004: Memory & Boot Logic Strategy

**Date:** July 11, 2026
**Status:** Accepted
**Author:** Oracle

## Context

**This is the critical success factor for the entire project.**

Oracle's context resets between sessions. Sessions are lost. Memory is injected but limited to 2,200 chars. If Oracle boots cold and can't reconstruct what's happening on the Spark, the project fails. James has experienced this across three agent profiles (Lara, Mais, Oracle) each having partial knowledge — the left hand doesn't know what the right hand did.

## Decision

### Three-Layer Memory Architecture

```
Layer 1: Hermes Memory (auto-injected, 2,200 char limit)
  → Project anchor + critical constraints only
  → Points to INDEX.md

Layer 2: AGENTS.md (auto-loaded from cwd)
  → Boot sequence: what to read, in what order
  → Points to workspace and INDEX.md

Layer 3: Project Files (on disk, no size limit)
  → INDEX.md      — file index, current state summary
  → STATE.md      — what's running NOW, last-known-good per flavor
  → recipes/      — locked configs with every parameter
  → adr/          — WHY decisions were made
  → benchmarks/   — WHAT we measured
  → sources/      — WHERE recipes came from
```

### Boot Sequence (On Every New Oracle Session)

1. **Memory auto-injects** — Oracle sees the project anchor and knows to read INDEX.md
2. **AGENTS.md loads** — confirms workspace location and boot steps
3. **Read `INDEX.md`** — file index, current state summary, quick boot sequence
4. **Read `STATE.md`** — what's running on the Spark RIGHT NOW, last-known-good per flavor
5. **Check Spark health** (if relevant to the task) — `curl http://larryspark.local:8000/v1/models`
6. **Proceed with task** — using recipes/scripts as needed

### STATE.md Discipline

STATE.md is the **single source of truth** for what's on the Spark. It must be updated:

- After every model switch (what's running now)
- After every successful test (last-known-good update)
- After every crash or failure (what went wrong)
- After every recipe lock (status change)

The format is always:
```
## What's Running NOW
[model, served name, port, context, concurrency, tool calling status, started, PID]

## Last-Known-Good Per Flavor
[per flavor: status, verified by, last worked, config summary, benchmark]
```

### What Goes in Hermes Memory (Layer 1)

Only the essential anchor — enough to find the project and know the constraints:
- Project location
- Oracle's role (chief LLM-optimizer)
- Loca is test pilot
- Three flavors exist
- Never wire Oracle/global to Spark
- STATE.md is the source of truth

### What Goes in AGENTS.md (Layer 2)

The boot sequence — what to read and in what order. Updated when the project structure changes.

### What Goes in Project Files (Layer 3)

Everything else. Full recipes, full benchmarks, full ADRs, full sources. No size limit.

## Consequences

- A fresh Oracle session can boot cold and know the full state in <30 seconds
- STATE.md is the most important file — it must never be stale
- Memory stays compact (project anchor only, not recipe details)
- Cross-profile knowledge (Mais's workspace, Lara's memory) is referenced, not duplicated
- Every session that touches the Spark must end with a STATE.md update