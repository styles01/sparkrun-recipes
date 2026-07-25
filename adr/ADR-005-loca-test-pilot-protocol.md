# ADR-005: Loca as Test Pilot Protocol

**Date:** July 11, 2026
**Status:** Accepted
**Author:** Oracle

## Context

Loca is the only Hermes profile wired to the Spark. Her name is a pun — "crazy local model babe" — and her purpose is to be the local model specialist. She's the natural test pilot for every Spark flavor because she's already connected and configured for it.

Testing a model's raw tok/s is not enough. We need to know that an actual Hermes agent loop works end-to-end: tool calls, file reads, terminal commands, multi-turn reasoning, context retention. Only Loca can verify this.

## Decision

### Loca Tests Every Flavor Before It's Locked

No recipe is marked as last-known-good until Loca has successfully completed a full agent loop test.

### Test Protocol

For each flavor, Loca must complete:

1. **Tool call test** — Loca receives a request that requires a tool call (e.g., "read this file" → uses `read_file` tool). If the vLLM server returns 400 on tool calls, the test fails immediately.
2. **File operation test** — Loca reads a file and reports its contents.
3. **Terminal test** — Loca runs a shell command and reports output.
4. **Multi-turn test** — Loca maintains context across 3+ turns without losing track.
5. **Smoke test** — A simple "hello, what model are you?" to verify served-model-name and basic generation.

### Pass/Fail Criteria

| Test | Pass | Fail |
|---|---|---|
| Tool call | Tool executes, result returned | HTTP 400, empty tool_calls, or crash |
| File op | File contents returned correctly | Error, wrong file, or no response |
| Terminal | Command output returned | Error, timeout, or no response |
| Multi-turn | Context preserved across turns | Loca forgets previous turn |
| Smoke | Correct model name, coherent response | Wrong model, gibberish, or no response |

### After Test

- **All pass** → Update STATE.md (last-known-good), save benchmark, recipe is LOCKED
- **Any fail** → Document failure in `benchmarks/`, do NOT update last-known-good, debug and retry
- **Crash** → Document crash in STATE.md (what went wrong), kill processes, stabilize Spark

### Loca Config Requirements

Loca's `dflash-spark` provider must list all available flavors as models, so she can be pointed at whichever is running:

```yaml
dflash-spark:
  name: DFlash Spark
  api: http://larryspark.local:8000/v1
  default_model: <matches whatever is currently running>
  max_output_tokens: 32768
  models:
    - deepseek-v4-flash
    - qwen
    - qwen35b
```

When Oracle switches the Spark to a new flavor, TWO config fields must be updated in Loca's config.yaml:
1. `model.default` (top-level) — this is what Hermes actually sends as the model name
2. `dflash-spark.default_model` (provider-level) — this is what the provider offers
3. `fallback_providers[].model` — must also match

**Only updating the provider-level is NOT enough.** The top-level `model.default` wins and causes 404s if it doesn't match the served model name.

## Consequences

- A recipe with great tok/s but broken tool calls will be rejected
- Loca is the gatekeeper — nothing goes to production without her passing
- Loca's config must be kept in sync with what's running on the Spark
- Oracle runs the tests by messaging Loca (or James does directly)
- Benchmark results include both raw tok/s AND Loca agent loop pass/fail