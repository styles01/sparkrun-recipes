# SparkRun Recipes

**Status:** ✅ Production Ready  
**Last Updated:** August 17, 2026  
**Hardware:** NVIDIA DGX Spark (GB10, 121GB unified memory, SM121, aarch64)

---

## ⭐ Featured: Qwen 3.8 27B NVFP4 on Drowzeys GB10 vLLM

The best daily-driver model for agentic workloads on a single DGX Spark. Qwen 3.8 is a hybrid-architecture model (48 Gated DeltaNet + 16 attention layers) with native Multi-Token Prediction (MTP) — its in-checkpoint draft head aligns natively, delivering stable speculative decode acceptance across all workload types without an external drafter.

| Metric | Value |
|---|---|
| **Model** | unsloth/Qwen3.8-27B-NVFP4 (22 GB NVFP4, NOT the 54GB FP8) |
| **Engine** | vLLM v0.27.1 (drowzeys GB10 build — `ghcr.io/drowzeys/keys-vllm-027-gb10-qwen38:mtp3-20260813`) |
| **Quant** | NVFP4 (compressed-tensors), fp8 KV cache |
| **Spec Decode** | MTP n=3 (native in-checkpoint head, no external drafter) |
| **Context** | 262,144 tokens (256K) |
| **Concurrency** | 4 seqs, 6.45× KV cache concurrency |
| **GMU** | 0.55 (safe ceiling on GB10 — 0.72+ hard-freezes the Spark) |
| **Decode** | 31.7 tok/s single-stream, **96 tok/s aggregate** (c=4) |
| **Prefix Cache** | `--enable-prefix-caching` (essential — cuts TTFT from 47s to single-digit on repeated context) |
| **Parser** | `--tool-call-parser qwen3_coder --enable-auto-tool-choice` |
| **Served name** | `qwen38-27b` (matches Loca's provider config — do NOT change) |

### What speed to expect

Speed is **content-dependent** on this hardware. The GB10's 225 GB/s memory bandwidth is the physics ceiling — speculative decode raises the effective rate by predicting multiple tokens per forward pass, but acceptance varies by workload:

| Workload | Single-stream tok/s | Spec accept | Notes |
|---|---|---|---|
| **Code / diffs / tool calls** | 28-32 | ~65% | MTP predicts structured output well |
| **Reasoning (thinking ON)** | 25-30 | ~50% | Reasoning tokens are semi-predictable |
| **Technical explanations** | 22-27 | ~45% | Mixed content |
| **Free-form prose** | 18-22 | ~35% | Least predictable — MTP still stable (no collapse) |
| **Aggregate (c=4 concurrent)** | ~96 total | — | ~24/stream under concurrent load |

> **Thinking stays ON.** Disabling thinking raises acceptance and tok/s significantly, but defeats the purpose of a native reasoning model. We do not dumb down the model for benchmarks.

### Why MTP (not DSpark / EAGLE)

We tested all three speculative decode methods on this model over a full day:

| Method | Framework | Single-stream | Tool calling | Acceptance | Verdict |
|---|---|---|---|---|---|
| **MTP n=3** | vLLM (drowzeys) | **31.7** | N/A | ~65% | ✅ **Winner — stable across all workloads** |
| EAGLE 3/1/4 | SGLang | 27 | 29.4 | 2.42-3.73 | Correct alignment, can't beat MTP aggregate |
| DSpark k=7 | vLLM | 13-17 | N/A | 0.28 (chance) | ❌ External drafter can't predict reasoning tokens |
| DSpark k=7 | SGLang | 27.8 | 24.0 | 2.02-4.08 | ❌ Collapses on tool calls (benchmark warned) |
| DSpark k=5 | SGLang | ~22 | ~22 | 0.22 | ❌ Worse — higher k = more wasted drafts |

**DSpark's external drafter (1.36B block-drafter) cannot predict reasoning tokens.** Acceptance collapses to chance-level (~1/k) on thinking-on prose. The author's 75 tok/s claim was measured with **thinking OFF** — with thinking ON, ~28% acceptance is the correct expected ceiling, not a config bug.

MTP uses the model's own in-checkpoint draft head — it aligns natively and doesn't collapse on any workload type. For Loca's tool-calling agent traffic, MTP's 96 aggregate tok/s beats every alternative we tested.

### ⚠️ GB10 Platform Traps

- **GMU > 0.55 can hard-freeze the Spark.** SGLang/vLLM don't account for 25-40 GB of transient GB10 allocations (flashinfer fp8 autotuner + CUDA graph capture). GMU 0.72 froze our Spark — required a physical power cycle. **Max safe GMU on drowzeys vLLM = 0.55.**
- **Stock vLLM has no GB10 NVFP4 kernels.** The drowzeys build ships the `FlashInferCutlassNvFp4LinearKernel` — stock `vllm/vllm-openai` cannot run NVFP4 on GB10. Verify the kernel loads: `docker logs qwen38 2>&1 | grep FlashInferCutlassNvFp4`.
- **`--kv-cache-dtype bf16` is INVALID.** The valid value is `bfloat16` (or `auto`). DSpark forces FLASH_ATTN which rejects fp8 KV, so use `bfloat16` with DSpark — but MTP on drowzeys uses `fp8` fine.
- **vLLM prefix caching was OFF by default.** Without `--enable-prefix-caching`, every API call re-prefills the entire conversation (47s TTFT on 50K context). With it, prefix cache hits 31%+ on first calls and 90%+ on repeated context.
- **Loca expects port 8000.** Always serve on port 8000, bound to `0.0.0.0` (`-p 8000:8000`, NOT `127.0.0.1:8000`). Loca's `larryspark` provider is `http://larryspark.local:8000/v1` — do NOT change it.

### Launch

```bash
# One-command switch (on the Spark)
ssh jaita@192.168.2.185 'GMU=0.55 bash ~/switch-to-qwen27b.sh'

# Or via sparkrun
sparkrun run @styles01/qwen-38-27b --hosts <spark-ip> --trust
```

> **Recipe:** [`recipes/qwen-38-27b.yaml`](recipes/qwen-38-27b.yaml) · **Runbook:** [`runbooks/qwen-38-27b.md`](runbooks/qwen-38-27b.md) · **Switch script:** [`scripts/switch-to-qwen27b.sh`](scripts/switch-to-qwen27b.sh)

### Alternative Recipes (tested, not daily drivers)

| Recipe | Method | Framework | Best for | Status |
|---|---|---|---|---|
| [`qwen-38-27b-nvfp4-eagle-sglang`](recipes/qwen-38-27b-nvfp4-eagle-sglang.yaml) | EAGLE 3/1/4 | SGLang | Correct spec alignment, SGLang-native path | 🧪 Reference |
| [`qwen-38-27b-fp8-dspark`](recipes/qwen-38-27b-fp8-dspark.yaml) | DSpark k=7 | vLLM | Code-only with thinking off | 🧪 Reference |
| [`qwen-38-27b-fp8-eugr-leader`](recipes/qwen-38-27b-fp8-eugr-leader.yaml) | MTP n=3 + instanttensor | vLLM (eugr) | Arena benchmark | 🧪 Reference |

### Benchmarking Traps (all hit for real)

1. **Cold-start JIT** — first request after boot includes CUDA graph compilation. Never measure it. Warm up with 2-3 throwaway requests first.
2. **Prefix cache artifacts** — repeated identical prompts show artificially high tok/s because prefill is fully cached. Always vary the prompt or measure the second request, not the first.
3. **Thinking-on vs thinking-off** — disabling thinking raises tok/s by 2-3× but defeats the model's purpose. Always verify whether a benchmark claim was measured with thinking on or off.
4. **Synthetic vs real tokens** — arena `tg128` benchmarks use synthetic tokens with high acceptance. Real agent traffic (reasoning, tool calls, prose) has lower acceptance. Compare shape across depth, not absolute values.
5. **Single-stream vs aggregate** — 31.7 single-stream is not 96 aggregate. Know which metric matters for your workload. For Loca (concurrent agent requests), aggregate is king.
6. **Hardware transfer** — 75 tok/s on RTX 5090 (sm_120) does not mean 75 on GB10 (sm_121). Acceptance transfers; tok/s doesn't — bandwidth differs.

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
| ⭐ **Qwen 3.8 27B NVFP4 (drowzeys MTP n=3)** | [runbook](runbooks/qwen-38-27b.md) | [recipe](recipes/qwen-38-27b.yaml) | ✅ Production (daily driver) |
| ⭐ **DS4-Flash 0731 (ds4 CUDA v0.5.5)** | [runbook](runbooks/deepseek-v4-flash-ds4.md) | [recipe](recipes/deepseek-v4-flash-0731-ds4.yaml) | ✅ Production |
| **DS4-Flash 0731 (vLLM-Moet)** | [runbook](runbooks/deepseek-v4-flash.md) | [recipe](recipes/deepseek-v4-flash-0731.yaml) | ⚠️ OOM risk (167GB) |
| **Qwen 3.5 122B DFlash** | [runbook](runbooks/qwen-122b.md) | [recipe](recipes/qwen-122b.yaml) | ✅ Production |
| **Qwen 122B v26 fp8 KV** | [runbook](runbooks/qwen-122b-v26-fp8-kv-dflash-int8.md) | [recipe](recipes/qwen-122b-v26-fp8-kv-dflash-int8.yaml) | ✅ Breakthrough |
| **Qwen 3.6 35B NVFP4** | [runbook](runbooks/qwen-35b.md) | [recipe](recipes/qwen-35b.yaml) | ✅ Production |
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
# Qwen 3.8 27B (daily driver)
ssh user@<spark-host> 'GMU=0.55 bash ~/switch-to-qwen27b.sh'

# DS4 Flash (0731 weights)
ssh user@<spark-host> 'bash ~/switch-to-ds4.sh'

# Qwen 122B
ssh user@<spark-host> 'bash ~/switch-to-122b.sh'

# Qwen 35B
ssh user@<spark-host> 'bash ~/switch-to-35b.sh'
```

See [SPARKRUN-REFERENCE.md](SPARKRUN-REFERENCE.md) for the full SparkRun CLI guide.

---

## Benchmark Highlights

| Model | Decode tok/s | Prefill tok/s | Context | Banks | Engine | Notes |
|---|---|---|---|---|---|---|
| ⭐ **Qwen 3.8 27B (drowzeys MTP)** | 31.7 single / **96 aggregate** | — | 256K | 4 | vLLM 0.27 (drowzeys) | MTP n=3, fp8 KV, prefix caching, GMU 0.55 |
| ⭐ **DS4-Flash 0731 (balanced)** | 20 single / 65 peak | 1127 | 196K | 20 | ds4 CUDA v0.5.5 | DSpark k=2, 75% accept, IQ2XXS |
| **DS4-Flash 0731 (speed)** | 20 single / 40 peak | 1000 | 131K | 14 | ds4 CUDA v0.5.5 | Max concurrency |
| **DS4-Flash 0731 (deep)** | 20 single / 25 peak | 1000 | 1M | 8 | ds4 CUDA v0.5.5 | Disk KV overflow |
| Qwen 122B (aeon) | 50.2 | — | 256K | 3 | vLLM | Stable across workloads |
| Qwen 122B (v26 fp8) | 45.98 | — | 256K | 3 | vLLM v26 | 2.6× KV, int8 lm-head |
| Qwen 35B | 109.3 | — | 256K | 4 | vLLM | Fastest, best concurrency |
| Laguna S 2.1 | 30 | — | 250K | 2 | vLLM v26 | DFlash k=7, 33% accept |
| **Muse-Glimmer 30B** | 40.5 single | — | 131K | 4 | SGLang | NVFP4 + DFlash (prefill spikes 1.7–8.5M tok/s) |
| MedGemma 27B | 336 agg | — | 8K | 75 | vLLM | Corpus formatting |
| **Nemotron 3.5 Lightning 30B-A3B** | 108.1 single / 224 agg | 7665 | 100K | 10 | vLLM 0.27.1 | DSpark n=4, marlin, fp8 KV — arena sub `sub1786523649341` |
| **Nemotron Super 120B** | 30+ | — | 128K | 10 | vLLM v0.20 | MTP k=1, marlin, fp8 KV (community recipe) |
| **MedGemma 27B Medicomp** | 336 agg | — | 3.8K | 75 | vLLM | FlashInfer, fp8 KV, VPN host |
| Puzzle 75B | 35.9 | — | 256K | 4 | vLLM | MTP k=3, 74.7% accept |

See `benchmarks/` for detailed benchmark reports.

---

## Recipe YAML Format

Recipes follow the spark-arena.com API contract format:

```yaml
recipe_version: "4"
name: <kebab-case-name>
description: <short description>
model: <huggingface-model-id>
runtime: vllm  # or sglang, ds4-cuda
container: <docker-image>

metadata:
  author: <username>
  tags: [...]
  runbook: runbooks/<name>.md

defaults:
  port: 8000
  host: 0.0.0.0
  gpu_memory_utilization: 0.55
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

## Credits

- **Drowzeys** — GB10 NVFP4 vLLM build ([`ghcr.io/drowzeys/keys-vllm-027-gb10-qwen38`](https://github.com/drowzeys))
- **@0xBakeer** — DSpark 75 tok/s recipe (thinking-off benchmark, edit-heavy workload)
- **@calneymgp** — MTP EAGLE 3/1/4 recipe on SGLang ([repo](https://huggingface.co/calneymgp/Qwen3.8-27B-NVFP4-lmhead4-recipe))
- **@hasso5703** — DSpark-block SGLang config and README structure ([repo](https://github.com/hasso5703/dgx-spark-qwen38))
- **@huchkw** — DGX Spark power/thermal tweak (`nvidia-smi -lgc 0,2200`)
- **@bleysg / @antirez** — DS4 CUDA engine and DSpark lossless speculative decoding

---

## License

MIT. Measurements are point-in-time on specific hardware. Your acceptance rates will vary by workload, context depth, and model version.