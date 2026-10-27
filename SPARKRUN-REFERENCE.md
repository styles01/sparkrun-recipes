# SparkRun — All-in-One Recipe Management System

**Status:** ✅ Primary workflow for all model switching and benchmarking  
**Date:** July 26, 2026

---

## Why SparkRun

SparkRun is now our **single source of truth** for:
- ✅ Recipe management (find, validate, list, search)
- ✅ Model switching (`sparkrun run <recipe>`)
- ✅ Benchmarking and leaderboard submission (`sparkrun arena benchmark`)
- ✅ Cluster/Host management
- ✅ Container lifecycle (launch, stop, logs)
- ✅ Configuration and setup

**No more manual Docker commands or switch scripts** — everything goes through SparkRun.

---

## Core Commands

### Recipe Management

```bash
# List all available recipes
sparkrun recipe list

# Search recipes by name, model, or description
sparkrun recipe search <query>

# Show detailed recipe information
sparkrun recipe show <recipe-name>

# Validate a recipe file
sparkrun recipe validate <recipe.yaml>
```

### Running Models

```bash
# Run a recipe on a specific host
sparkrun run <recipe-name> --hosts <host>

# Run with custom overrides
sparkrun run <recipe-name> --hosts <host> \
  --gpu-memory-utilization 0.85 \
  --max-model-len 262144 \
  -o kv_cache_dtype=fp8 \
  -o dflash_n=7
```

### Benchmarking & Arena

```bash
# Authenticate with Spark Arena
sparkrun arena login

# Check authentication status
sparkrun arena status

# Run benchmark and submit to leaderboard
sparkrun arena benchmark <recipe-name> --hosts <host>

# View benchmark results
sparkrun arena benchmark <recipe-name> --dry-run
```

### Container/Workload Management

```bash
# List running workloads
sparkrun status

# View logs
sparkrun logs <workload-id>

# Stop a running workload
sparkrun stop <workload-id>
```

### Cluster & Setup

```bash
# List saved cluster definitions
sparkrun cluster list

# Setup and configuration
sparkrun setup <command>

# Clear GPU cache
sparkrun setup clear-cache --save-sudo
```

---

## Recipe Format

All recipes must follow this structure:

```yaml
recipe_version: "2"
name: <kebab-case-name>
description: <description>

model:
  id: <huggingface-model-id>
  runtime: <vllm|sglang|llama-cpp>
  container: <docker-image>

defaults:
  gpu_memory_utilization: 0.85
  max_model_len: 262144
  kv_cache_dtype: fp8
  # ... other runtime parameters

command: |
  # Full serve command (optional, overrides defaults)

metadata:
  author: <username>
  created: "YYYY-MM-DD"
  tags:
    - <tag1>
    - <tag2>
```

**Naming Convention:** `kebab-case` with hyphens
- ✅ `qwen-122b-v26-fp8-kv-dflash-int8`
- ❌ `qwen_122b_v26_fp8_kv_dflash_int8`
- ❌ `Qwen122Bv26`

---

## Our Registered Recipes

| Recipe | Model | Runtime | Status |
|---|---|---|---|
| `@styles01/recipe` | bleysg/Qwen3.5-122B-A10B-int4-fp8-hybrid | vllm-distributed | ✅ Active |
| `@eugr/qwen3.5-122b-fp8` | Qwen/Qwen3.5-122B-A10B-FP8 | vllm-ray | Available |
| `@sparkrun-transitional/qwen3.5-122b-a10b-fp8-sglang` | Qwen/Qwen3.5-122B-A10B-FP8 | sglang | Available |

---

## Workflow

### 1. Create/Update Recipe

```bash
# Create recipe file in workspace
recipes/qwen-122b-v26-fp8-kv-dflash-int8.yaml

# Validate it
sparkrun recipe validate recipes/qwen-122b-v26-fp8-kv-dflash-int8.yaml
```

### 2. Register with SparkRun (if needed)

```bash
# Push recipe to registry (if you have permissions)
sparkrun registry push <recipe.yaml>

# Or use local recipe directly
sparkrun run ./recipes/qwen-122b-v26-fp8-kv-dflash-int8.yaml --hosts <spark-host>
```

### 3. Run Model

```bash
sparkrun run @styles01/recipe --hosts <spark-host>
```

### 4. Benchmark & Submit

```bash
sparkrun arena benchmark @styles01/recipe --hosts <spark-host>
```

### 5. Monitor & Manage

```bash
# Check status
sparkrun status

# View logs
sparkrun logs <workload-id>

# Stop when done
sparkrun stop <workload-id>
```

---

## Key Benefits

1. **Single Command** — No more manual Docker commands
2. **Automatic Management** — Container lifecycle handled by sparkrun
3. **Benchmark Integration** — Direct submission to Spark Arena leaderboard
4. **Recipe Versioning** — Track changes and versions
5. **Validation** — Catch errors before running
6. **Consistency** — Same workflow for all models

---

## Migration from Old Scripts

| Old Method | New SparkRun Method |
|---|---|
| `bash switch-to-122b.sh` | `sparkrun run @styles01/recipe --hosts <spark-host>` |
| Manual Docker run | `sparkrun run <recipe> --hosts <host>` |
| Manual benchmark | `sparkrun arena benchmark <recipe> --hosts <host>` |
| Manual container stop | `sparkrun stop <workload-id>` |
| Manual log check | `sparkrun logs <workload-id>` |

---

## Notes

- All recipes should be in `recipes/` directory (YAML contracts)
- All runbooks should be in `runbooks/` directory (detailed markdown guides)
- Runbooks link to recipes and vice versa
- Use `@registry/name` format for registered recipes
- Use `./path/to/recipe.yaml` for local recipes
- Always authenticate first: `sparkrun arena login`
- Use `--dry-run` to test before executing
- Check `sparkrun <command> --help` for all options

---

**Last Updated:** July 26, 2026  
**Status:** ✅ Active — Use SparkRun for all model operations going forward
