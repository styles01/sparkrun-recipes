# Model Switch Playbook — Loca Config Update

**CRITICAL:** When switching models on the Spark, you MUST update THREE fields in `~/.hermes/profiles/loca/config.yaml`. Missing any one causes 404 errors, retries, and 5-10s latency per request.

## The Three Fields

### 1. `model.default` (top-level — THIS IS WHAT HERMES ACTUALLY SENDS)

```yaml
model:
  default: qwen35b    # ← MUST match the served-model-name on the Spark
  provider: dflash-spark
```

This is the model name Hermes sends in the API request body as `"model": "qwen35b"`. If this doesn't match what the vLLM server is serving (`--served-model-name`), you get HTTP 404.

### 2. `dflash-spark.default_model` (provider-level)

```yaml
providers:
  dflash-spark:
    default_model: qwen35b    # ← MUST match model.default
```

### 3. `fallback_providers[].model`

```yaml
fallback_providers:
  - provider: dflash-spark
    model: qwen35b    # ← MUST match
```

## Served Model Names Per Flavor

| Flavor | `--served-model-name` on Spark | Loca `model.default` |
|---|---|---|
| 35b | `qwen35b` | `qwen35b` |
| 122b | `qwen` | `qwen` |
| ds4 | `deepseek-v4-flash` | `deepseek-v4-flash` |

## The Bug We Hit (July 11, 2026)

Oracle updated `dflash-spark.default_model` to `qwen35b` but left `model.default` as `qwen` (from the previous 122B session). Hermes sent `"model": "qwen"` to a server serving `qwen35b` → 404 → retry → 404 → eventually fell through but added 5-10s latency per request. Loca felt slow on every flavor because of this.

**Root cause:** `model.default` (top-level) overrides `dflash-spark.default_model` (provider-level). Only updating the provider is not enough.

## Switch Procedure (Do Every Time)

1. Switch model on Spark (switch-to-X.sh)
2. Verify served name: `curl http://larryspark.local:8000/v1/models`
3. Update `model.default` in `~/.hermes/profiles/loca/config.yaml`
4. Update `dflash-spark.default_model` in same file
5. Update `fallback_providers[0].model` in same file
6. Restart Loca gateway: `kill <PID>; hermes --profile loca gateway run --replace`
7. Verify: message Loca, check for 404s in `docker logs` or vLLM logs