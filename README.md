# SparkRun Recipes

**Status:** ✅ Production Ready  
**Last Updated:** August 5, 2026  
**Hardware:** NVIDIA DGX Spark (GB10, 121GB unified memory, SM121, aarch64)

---

## ⭐ Featured: DeepSeek-V4-Flash 0731 on DS4 CUDA Engine

The smartest model available, now serving on a single DGX Spark via [Bleysg's ds4 CUDA engine](https://github.com/Entrpi/ds4-on-spark) — a ground-up C/CUDA inference engine with DSpark lossless speculative decoding.

| Metric | Value |
|---|---|
| **Model** | DeepSeek-V4-Flash 0731 (284B params, 12B active MoE) |
| **Engine** | ds4 CUDA (Entrpi/ds4 fork v0.5.5) |
| **Quant** | IQ2XXS 2-bit with imatrix (~87 GB GGUF) |
| **Spec Decode** | DSpark k=2, ~60-75% acceptance |
| **Context** | 196K per lane, 20 banks (auto-scaled) |
| **Decode** | ~20 tok/s single-stream, ~65 tok/s peak (8 concurrent) |
| **Prefill** | ~1,127 tok/s |
| **TTFT** | ~200ms |
| **Author** | [@bleysg](https://x.com/bleysg) (Bleys Goodson) / [@antirez](https://x.com/antirez) (Salvatore Sanfilippo) |

### Three Flavor Configs

| Flavor | Context | Banks | Peak tok/s | Use Case |
|---|---|---|---|---|
| **Balanced ⭐ (default)** | 196K | 20 | 65 | Best balance, forum-proven sweet spot |
| **Speed** | 131K | 14 | ~40 | Max concurrency |
| **Deep** | 1M | 8 | ~25 | Deep context with disk KV overflow |

Switch flavors by changing env vars and `-c` value — no rebuild needed.

### Build & Upgrade

```bash
# Build (on Spark): make cuda-spark
# Upgrade: cd ~/code/ds4 && git fetch --all --tags && git checkout v0.5.5 && make cuda-spark
# v0.5.5 fixes: stream512 race (Xid 13 root cause), admission governance, budget double-booking
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
| ⭐ **DS4-Flash 0731 (ds4 CUDA v0.5.5)** | [runbook](runbooks/deepseek-v4-flash-ds4.md) | [recipe](recipes/deepseek-v4-flash-0731-ds4.yaml) | ✅ Production |
| **DS4-Flash 0731 (vLLM-Moet)** | [runbook](runbooks/deepseek-v4-flash.md) | [recipe](recipes/deepseek-v4-flash-0731.yaml) | ⚠️ OOM risk (167GB) |
| **Qwen 3.5 122B DFlash** | [runbook](runbooks/qwen-122b.md) | [recipe](recipes/qwen-122b.yaml) | ✅ Production |
| **Qwen 122B v26 fp8 KV** | [runbook](runbooks/qwen-122b-v26-fp8-kv-dflash-int8.md) | [recipe](recipes/qwen-122b-v26-fp8-kv-dflash-int8.yaml) | ✅ Breakthrough |
| **Qwen 3.6 35B NVFP4** | [runbook](runbooks/qwen-35b.md) | [recipe](recipes/qwen-35b.yaml) | ✅ Production |
| **Qwen 3.8 27B NVFP4** | [runbook](runbooks/qwen-38-27b.md) | [recipe](recipes/qwen-38-27b.yaml) | ✅ Production |
| **Laguna S 2.1 NVFP4** | [runbook](runbooks/laguna-s-2.1.md) | [recipe](recipes/laguna-s-2.1.yaml) | ✅ Production |
| **Muse-Glimmer 30B** | [runbook](runbooks/muse-glimmer-30b.md) | [recipe](recipes/muse-glimmer-30b.yaml) | ✅ Production (SGLang) |
| **MedGemma 27B FP8** | [runbook](runbooks/medgemma-27b.md) | [recipe](recipes/medgemma-27b.yaml) | ✅ Production |
| **MedGemma 27B Medicomp** | [runbook](runbooks/medgemma-27b-medicomp.md) | [recipe](recipes/medgemma-27b-medicomp.yaml) | ✅ Production |
| **Nemotron Super 120B** | [runbook](runbooks/nemotron-super-120b-nvfp4.md) | [recipe](recipes/nemotron-super-120b-nvfp4.yaml) | 🧪 Community recipe |
| **Nemotron 3.5 Lightning 30B-A3B** | [runbook](runbooks/nemotron-3.5-lightning-30b-a3b-nvfp4.md) | [recipe](recipes/nemotron-3.5-lightning-30b-a3b-nvfp4.yaml) | ✅ Production (arena-validated) |
| **Puzzle 75B NVFP4** | [runbook](runbooks/puzzle-75b.md) | [recipe](recipes/puzzle-75b.yaml) | 🧪 Community-validated |

---

## Quick Start

### Using SparkRun

```bash
# List all available recipes
sparkrun recipe list

# Run a recipe on a specific host
sparkrun run deepseek-v4-flash-0731 --hosts <spark-host>

# Benchmark and submit to Spark Arena
sparkrun arena benchmark qwen-122b --hosts <spark-host>
```

### Using Switch Scripts (legacy)

```bash
# DS4 Flash (0731 weights)
ssh user@<spark-host> 'bash ~/switch-to-ds4.sh'

# Qwen 122B
ssh user@<spark-host> 'bash ~/switch-to-122b.sh'

# Qwen 35B
ssh user@<spark-host> 'bash ~/switch-to-35b.sh'

# Qwen 3.8 27B
ssh user@<spark-host> 'bash ~/switch-to-qwen27b.sh'
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

| Model | Decode tok/s | Prefill tok/s | Context | Banks | Engine | Notes |
|---|---|---|---|---|---|---|
| ⭐ **DS4-Flash 0731 (balanced)** | 20 single / 65 peak | 1127 | 196K | 20 | ds4 CUDA v0.5.5 | DSpark k=2, 75% accept, IQ2XXS |
| **DS4-Flash 0731 (speed)** | 20 single / 40 peak | 1000 | 131K | 14 | ds4 CUDA v0.5.5 | Max concurrency |
| **DS4-Flash 0731 (deep)** | 20 single / 25 peak | 1000 | 1M | 8 | ds4 CUDA v0.5.5 | Disk KV overflow |
| Qwen 122B (aeon) | 50.2 | — | 256K | 3 | vLLM | Stable across workloads |
| Qwen 122B (v26 fp8) | 45.98 | — | 256K | 3 | vLLM v26 | 2.6× KV, int8 lm-head |
| Qwen 35B | 109.3 | — | 256K | 4 | vLLM | Fastest, best concurrency |
| Qwen 3.8 27B | TBD | — | 256K | 4 | vLLM nightly | triton_attn, MTP k=2, fp8 KV (flashinfer crashes on GDN+MTP) |
| Laguna S 2.1 | 30 | — | 250K | 2 | vLLM v26 | DFlash k=7, 33% accept |
| **Muse-Glimmer 30B** | 40.5 single | — | 131K | 4 | SGLang | NVFP4 + DFlash (prefill spikes 1.7–8.5M tok/s) |
| MedGemma 27B | 336 agg | — | 8K | 75 | vLLM | Corpus formatting |
| **Nemotron 3.5 Lightning 30B-A3B** | 108.1 single / 224 agg | 7665 | 100K | 10 | vLLM 0.27.1 | DSpark n=4, marlin, fp8 KV — arena sub `sub1786523649341` |
| **Nemotron Super 120B** | 30+ | — | 128K | 10 | vLLM v0.20 | MTP k=1, marlin, fp8 KV (community recipe) |
| **MedGemma 27B Medicomp** | 336 agg | — | 3.8K | 75 | vLLM | FlashInfer, fp8 KV, VPN host |
| Puzzle 75B | 35.9 | — | 256K | 4 | vLLM | MTP k=3, 74.7% accept |

See `benchmarks/` for detailed benchmark reports.