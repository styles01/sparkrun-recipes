# ADR-003: Switch Script Architecture

**Date:** July 11, 2026
**Status:** Accepted
**Author:** Oracle

## Context

Switching models on the Spark is a multi-step process: stop the current model, clear caches, launch the new one, wait for health check, verify served-model-name. Doing this ad-hoc leads to mistakes — leftover processes, port conflicts, wrong served names. We need deterministic scripts that Oracle (or James) can run with one command.

## Decision

### One Script Per Flavor

Each flavor has a dedicated switch script in `scripts/`:

| Script | Flavor | Target Location |
|---|---|---|
| `switch-to-35b.sh` | Qwen 35B NVFP4 | `~/switch-to-35b.sh` on Spark |
| `switch-to-122b.sh` | Qwen 122B DFlash | `~/switch-to-122b.sh` on Spark |
| `switch-to-ds4.sh` | DeepSeek-V4-Flash | `~/switch-to-ds4.sh` on Spark |

### Script Structure (All Three Follow This Pattern)

1. **Stop everything** — kill Docker containers (`docker rm -f`) AND vLLM processes (`kill -9`)
2. **Clear caches** — FlashInfer cache, torch compile cache (if spec signature changed)
3. **Launch** — Docker run or venv vllm serve, with ALL locked flags
4. **Health check** — poll `http://127.0.0.1:8000/health` with timeout
5. **Verify** — `curl /v1/models` to confirm served-model-name
6. **Report** — print status (✅ or ❌)

### Invocation

From Mac (Oracle's default):
```bash
ssh jaita@larryspark.local 'bash ~/switch-to-35b.sh'
```

From Spark directly:
```bash
bash ~/switch-to-35b.sh
```

### Design Principles

- **Idempotent** — running a switch script when the same model is already running should stop and restart cleanly
- **Self-contained** — each script has all flags inline, no external env files needed
- **Verbose** — prints every step so logs can be debugged
- **Fail-fast** — if health check fails, print the log tail and exit non-zero
- **No orphaned processes** — stop step kills ALL vLLM processes and ALL known container names

## Consequences

- Switching is a single command, not a multi-step manual process
- Scripts must be updated when recipes are locked or superseded
- Scripts are deployed to Spark `~/` (not just in Oracle's workspace)
- Oracle's workspace has the source of truth; Spark has the deployed copy
- STATE.md must be updated after every switch (manual, not automated — see ADR-004)