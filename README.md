# SparkRun Recipes

**Status:** ✅ Production Ready  
**Last Updated:** July 31, 2026  
**Hardware:** NVIDIA DGX Spark (GB10, 121GB unified memory, SM121, aarch64)

---

## Repository Structure

```
sparkrun-recipes/
├── runbooks/          # Detailed markdown docs — commands, tradeoffs, context, troubleshooting
├── recipes/           # YAML arena recipe contracts — structured format for sparkrun/arena consumption
├── benchmarks/        # Benchmark results and database
├── docker/            # Dockerfiles and patches
├── scripts/           # One-command switch scripts
└── SPARKRUN-REFERENCE.md  # SparkRun CLI reference
```

### Runbooks vs Recipes

- **Runbooks** (`runbooks/*.md`) — Human-readable deployment guides with full context: why each flag was chosen, performance results, tradeoffs, troubleshooting, OOM fallbacks
- **Recipes** (`recipes/*.yaml`) — Machine-readable arena recipe contracts in YAML format, consumable by `sparkrun`, spark-arena.com, or similar orchestration tools

Each runbook links to its corresponding recipe contract and vice versa.

---

## Available Flavors

| Flavor | Runbook | Recipe | Status |
|---|---|---|---|
| **DeepSeek-V4-Flash (0731)** | [runbook](runbooks/deepseek-v4-flash.md) | [recipe](recipes/deepseek-v4-flash-0731.yaml) | ✅ Production |
| **Qwen 3.5 122B DFlash** | [runbook](runbooks/qwen-122b.md) | [recipe](recipes/qwen-122b.yaml) | ✅ Production |
| **Qwen 122B v26 fp8 KV** | [runbook](runbooks/qwen-122b-v26-fp8-kv-dflash-int8.md) | [recipe](recipes/qwen-122b-v26-fp8-kv-dflash-int8.yaml) | ✅ Breakthrough |
| **Qwen 3.6 35B NVFP4** | [runbook](runbooks/qwen-35b.md) | [recipe](recipes/qwen-35b.yaml) | ✅ Production |
| **Qwen 3.6 27B FP8** | [runbook](runbooks/qwen-27b.md) | [recipe](recipes/qwen-27b.yaml) | ✅ Production |
| **Laguna S 2.1 NVFP4** | [runbook](runbooks/laguna-s-2.1.md) | [recipe](recipes/laguna-s-2.1.yaml) | ✅ Production |
| **MedGemma 27B FP8** | [runbook](runbooks/medgemma-27b.md) | [recipe](recipes/medgemma-27b.yaml) | ✅ Production |
| **MedGemma 27B Medicomp** | [runbook](runbooks/medgemma-27b-medicomp.md) | [recipe](recipes/medgemma-27b-medicomp.yaml) | ✅ Production |
| **Nemotron Super 120B** | [runbook](runbooks/nemotron-super-120b-nvfp4.md) | [recipe](recipes/nemotron-super-120b-nvfp4.yaml) | 🧪 Community recipe |
| **Puzzle 75B NVFP4** | [runbook](runbooks/puzzle-75b.md) | [recipe](recipes/puzzle-75b.yaml) | 🧪 Community-validated |

---

## Quick Start

### Using SparkRun

```bash
# List all available recipes
sparkrun recipe list

# Run a recipe on a specific host
sparkrun run deepseek-v4-flash-0731 --hosts larryspark.local

# Benchmark and submit to Spark Arena
sparkrun arena benchmark qwen-122b --hosts larryspark.local
```

### Using Switch Scripts (legacy)

```bash
# DS4 Flash (0731 weights)
ssh jaita@larryspark.local 'bash ~/switch-to-ds4.sh'

# Qwen 122B
ssh jaita@larryspark.local 'bash ~/switch-to-122b.sh'

# Qwen 35B
ssh jaita@larryspark.local 'bash ~/switch-to-35b.sh'

# Qwen 27B
ssh jaita@larryspark.local 'bash ~/switch-to-qwen27b.sh'
```

See [SPARKRUN-REFERENCE.md](SPARKRUN-REFERENCE.md) for the full SparkRun CLI guide.

---

## Recipe YAML Format

Recipes follow the spark-arena.com API contract format:

```yaml
recipe_version: "2"
name: <kebab-case-name>
description: <short description>
model: <huggingface-model-id>
runtime: vllm
container: <docker-image>

metadata:
  author: <username>
  tags: [...]
  runbook: runbooks/<name>.md    # Link back to detailed runbook

defaults:
  port: 8000
  host: 0.0.0.0
  gpu_memory_utilization: 0.85
  max_model_len: 262144
  kv_cache_dtype: fp8
  # ... other runtime parameters

env:
  KEY: "value"

command: |
  vllm serve {model} ...
```

See [SPARKRUN-REFERENCE.md](SPARKRUN-REFERENCE.md) for full format specification.

---

## Benchmark Highlights

| Model | Decode tok/s | Context | Lanes | KV Tokens | Notes |
|---|---|---|---|---|---|
| DS4-Flash 0731 | ~21 | 256K | 1 | — | Smartest, most constrained |
| Qwen 122B (aeon) | 50.2 | 256K | 3 | 549K | Stable across workloads |
| Qwen 122B (v26 fp8) | 45.98 | 256K | 3 | 1.37M | 2.6× KV, int8 lm-head |
| Qwen 35B | 109.3 | 256K | 4 | — | Fastest, best concurrency |
| Qwen 27B | TBD | 256K | 5 | — | MTP k=7, 48.7% accept |
| Laguna S 2.1 | TBD | 300K | 3 | 918K | Terminal-Bench 70.2% |
| MedGemma 27B | 336 agg | 8K | 75 | 618K | Corpus formatting |
| Puzzle 75B | 35.9 | 256K | 4 | 2.0M | MTP k=3, 74.7% accept |

See `benchmarks/` for detailed benchmark reports.