# SparkRun Recipes

**Status:** ✅ Production Ready  
**Last Updated:** August 4, 2026  
**Hardware:** NVIDIA DGX Spark (GB10, 121GB unified memory, SM121, aarch64)

---

## ⭐ Featured: DeepSeek-V4-Flash 0731 on DS4 CUDA Engine

The smartest model available, now serving on a single DGX Spark via [Bleysg's ds4 CUDA engine](https://github.com/Entrpi/ds4-on-spark) — a ground-up C/CUDA inference engine with DSpark lossless speculative decoding.

| Metric | Value |
|---|---|
| **Model** | DeepSeek-V4-Flash 0731 (284B params, 12B active MoE) |
| **Engine** | ds4 CUDA (Entrpi/ds4 fork v0.5.4) |
| **Quant** | IQ2XXS 2-bit with imatrix (~87 GB GGUF) |
| **Spec Decode** | DSpark k=2, ~60-75% acceptance |
| **Context** | 131K per lane, 14 banks (auto-scaled) |
| **Decode** | ~20 tok/s single-stream, ~25 tok/s aggregate |
| **Prefill** | ~1000 tok/s |
| **TTFT** | ~200ms |
| **Author** | [@bleysg](https://x.com/bleysg) (Bleys Goodson) / [@antirez](https://x.com/antirez) (Salvatore Sanfilippo) |

### Build & Upgrade

```bash
# Build (on Spark): make cuda-spark
# Upgrade: cd ~/code/ds4 && git fetch --all --tags && git checkout v0.5.4 && make cuda-spark
# v0.5.4 fixes: trim-on-evict, serial graph rightsize, mem-floor-gb admission gate, auto context compression
```

```bash
# Register our recipes
sparkrun registry add https://github.com/styles01/sparkrun-recipes.git

# Launch DS4-Flash on a single Spark
sparkrun run @styles01/deepseek-v4-flash-0731-ds4 --hosts <spark-ip> --trust

# Benchmark it
sparkrun benchmark performance @styles01/deepseek-v4-flash-0731-ds4 --hosts <spark-ip> --skip-run --no-stop -b model=deepseek-v4-flash
```

> **Requires:** ds4 CUDA engine installed (`curl -sSL https://raw.githubusercontent.com/Entrpi/ds4-on-spark/main/install.sh | bash`) and the `ds4-cuda` runtime plugin. See the [runbook](runbooks/deepseek-v4-flash-ds4.md) for setup details.

---

## Repository Structure

```
sparkrun-recipes/
├── runbooks/          # Detailed markdown docs — commands, tradeoffs, context, troubleshooting
├── recipes/           # YAML arena recipe contracts — structured format for sparkrun/arena consumption
├── runtime/           # Custom runtime plugins (ds4-cuda)
├── benchmarks/        # Benchmark results and database
├── docker/            # Dockerfiles and patches
├── scripts/           # One-command switch scripts
├── .sparkrun/         # Registry manifest for sparkrun
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
| ⭐ **DS4-Flash 0731 (ds4 CUDA v0.5.4)** | [runbook](runbooks/deepseek-v4-flash-ds4.md) | [recipe](recipes/deepseek-v4-flash-0731-ds4.yaml) | ✅ Production |
| **DS4-Flash 0731 (vLLM-Moet)** | [runbook](runbooks/deepseek-v4-flash.md) | [recipe](recipes/deepseek-v4-flash-0731.yaml) | ⚠️ OOM risk (167GB) |
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

| Model | Decode tok/s | Prefill tok/s | Context | Lanes | Engine | Notes |
|---|---|---|---|---|---|---|
| ⭐ **DS4-Flash 0731** | 20 | 1000 | 131K | 2 | ds4 CUDA | DSpark k=2, 75% accept, IQ2XXS |
| Qwen 122B (aeon) | 50.2 | — | 256K | 3 | vLLM | Stable across workloads |
| Qwen 122B (v26 fp8) | 45.98 | — | 256K | 3 | vLLM v26 | 2.6× KV, int8 lm-head |
| Qwen 35B | 109.3 | — | 256K | 4 | vLLM | Fastest, best concurrency |
| Qwen 27B | TBD | — | 256K | 5 | vLLM v26 | MTP k=7, 48.7% accept |
| Laguna S 2.1 | 30 | — | 250K | 2 | vLLM v26 | DFlash k=7, 33% accept |
| MedGemma 27B | 336 agg | — | 8K | 75 | vLLM | Corpus formatting |
| Puzzle 75B | 35.9 | — | 256K | 4 | vLLM | MTP k=3, 74.7% accept |

See `benchmarks/` for detailed benchmark reports.