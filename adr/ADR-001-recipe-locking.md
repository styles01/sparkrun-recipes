# ADR-001: Recipe Locking Methodology

**Date:** July 11, 2026
**Status:** Accepted
**Author:** Oracle

## Context

James finds new recipes from Twitter, GitHub, and Spark Arena regularly. We need a consistent process for taking a community recipe, testing it, and either locking it or rejecting it. Without a formal process, we end up with half-tested configs, no memory of what worked, and repeated work across sessions.

## Decision

### Recipe Lifecycle

```
DISCOVERED → DOCUMENTED → TESTED → BENCHMARKED → LOCKED → SUPERSEDED
```

1. **Discovered** — James brings a recipe from Twitter/GitHub/Spark Arena. Oracle creates a draft in `recipes/` with source URL and original config.
2. **Documented** — Oracle writes the full recipe with: exact Docker command or venv command, env vars, model path, tool parser, reasoning parser, GPU mem util, context, max seqs. Source provenance recorded in `sources/SOURCES.md`.
3. **Tested** — Recipe is loaded on the Spark. Loca runs an agent loop (tool call, file read, terminal command, multi-turn conversation). If Loca completes the loop, it passes. If she 400s on tool calls, crashes, or can't complete — it fails.
4. **Benchmarked** — Standardized benchmark run using `benchmarks/BENCHMARK-TEMPLATE.md`. Results saved to `benchmarks/`.
5. **Locked** — Recipe is marked ✅ in STATE.md as last-known-good. Switch script is finalized. Recipe file gets a "LOCKED" header with date and verifier.
6. **Superseded** — When a better recipe is found, the old one is archived (not deleted) and marked SUPERSEDED with a pointer to the replacement.

### What Gets Locked

For each flavor, the locked recipe must specify:
- Exact Docker image (or venv path for DS4)
- Exact vLLM flags (every one)
- Exact env vars
- Model path/repo
- Served-model-name
- Tool parser + reasoning parser
- GPU memory utilization
- Max model len
- Max num seqs
- Max num batched tokens
- Speculative config (if any)
- Start command
- Stop command
- Expected startup time
- Expected tok/s range

### What Doesn't Get Locked

- Experimental variations (documented in recipe file under "Alternatives" but not in the switch script)
- Untested community recipes (stay in `sources/` until tested)

## Consequences

- Every flavor has exactly one locked recipe at any time
- Switch scripts always reference the locked recipe
- STATE.md always reflects what's locked and what's running
- New recipes can be tested without breaking the locked config
- Archive, don't delete — old recipes remain accessible