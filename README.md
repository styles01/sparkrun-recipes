<div align="center">

# SparkRun Recipes

**Source-pinned inference recipes for the NVIDIA DGX Spark.**

Production lanes, native Spark Arena recipes, benchmark evidence, and runbooks for getting serious long-context models onto one GB10 without pretending a social-media tok/s screenshot is a deployment.

[![DGX Spark](https://img.shields.io/badge/hardware-DGX%20Spark%20GB10-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://www.nvidia.com/en-us/products/workstations/dgx-spark/)
[![Spark Arena](https://img.shields.io/badge/Spark%20Arena-native%20recipes-181717?style=for-the-badge&logo=github&logoColor=white)](https://spark-arena.com)
[![Hugging Face](https://img.shields.io/badge/models-Hugging%20Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co)
[![Buy Me a Coffee](https://img.shields.io/badge/Buy%20me%20a%20coffee-support-yellow?style=for-the-badge&logo=buy-me-a-coffee&logoColor=white)](https://buymeacoffee.com/aitamedia)
[![Follow on X](https://img.shields.io/badge/Follow%20%40jaita%20on%20X-000000?style=for-the-badge&logo=x&logoColor=white)](https://x.com/jaita)

[Flavors](#the-flavors) · [Start Here](#start-here) · [Arena](#native-spark-arena-benchmarking) · [Recipe Catalog](#recipe-catalog) · [Benchmarks](#benchmark-interpretation) · [Contributing](#contributing)

</div>

> [!IMPORTANT]
> A DGX Spark has one unified 121 GB memory pool. These recipes are **exclusive lanes**: run one serious model or video workload at a time. Every production claim here belongs to a specific model revision, runtime, quantization, context, and workload. Read the linked runbook before switching anything.

---

## The flavors

This is the decision surface—not a wall of YAML. Pick the lane that matches the work, then follow its runbook exactly.

| Flavor | Best for | Shape | Evidence boundary | Start here |
|---|---|---|---|---|
| **Ling 3.0 Flash INT4** | Balanced long-context agent work | 124B MoE / 5.1B active · 256K · 2 lanes | Native Arena route; source-pinned Ling fork, FP8 KV, MTP k=1 | [Runbook](runbooks/ling-3.0-flash-int4.md) · [Arena recipe](recipes/ling-3.0-flash-vllm.yaml) |
| **Qwen 3.8 Flash-Next Q3** | Deep concurrent coding agents + vision | 3 × 220K lanes · ~57 aggregate tok/s observed | Requires the documented qwen4exp build and regression gates | [Runbook](runbooks/qwen38-flash-next-q3-3lane.md) · [Recipe](recipes/qwen3.8-flash-next-q3-3lane.yaml) |
| **DeepSeek-V4-Flash 0731 / DS4** | Hardest reasoning and very deep context | 20 tok/s single stream · 196K–1M configurations | Production DS4 CUDA route; DSpark k=2 | [Runbook](runbooks/deepseek-v4-flash-ds4.md) · [Recipe](recipes/deepseek-v4-flash-0731-ds4.yaml) |
| **Qwen 3.8 27B NVFP4** | Fast general agent traffic | 256K · native MTP · multi-request focus | GB10-specific vLLM kernel path; validate tool behavior on your workload | [Runbook](runbooks/qwen-38-27b.md) · [Recipe](recipes/qwen-38-27b.yaml) |
| **Qwen 122B DFlash** | Structured code and tool work | 3 lanes · 262K class | DFlash is workload-dependent; do not extrapolate code speed to prose | [Runbook](runbooks/qwen-122b.md) · [Recipe](recipes/qwen-122b.yaml) |
| **Qwen 35B NVFP4** | High-concurrency, lower-stakes work | 4+ lanes · 262K class | Fast, but do not substitute it for the deeper-reasoning lanes blindly | [Runbook](runbooks/qwen-35b.md) · [Recipe](recipes/qwen-35b.yaml) |

### How to choose

```text
Need the most dependable deep reasoning?             → DS4 Flash 0731
Need the best all-around agentic balance?             → Ling 3.0 Flash INT4
Need concurrent coding agents plus native vision?     → Qwen Flash-Next Q3
Need the fastest lightweight general lane?            → Qwen 3.8 27B NVFP4
Need structured code/tool throughput?                 → Qwen 122B DFlash
Need cheap high-concurrency background work?          → Qwen 35B NVFP4
```

> [!TIP]
> “Faster” only matters when the model still clears the task’s intelligence threshold. The benchmark section records throughput, not a universal quality ranking.

---

## Start here

### 1. Install SparkRun and register this catalog

```bash
python3 -m pip install --upgrade sparkrun
python3 -m sparkrun registry add https://github.com/styles01/sparkrun-recipes.git
python3 -m sparkrun recipe list
```

### 2. Read the runbook before touching the Spark

A recipe is a machine-readable contract. A runbook explains the context budget, known failure modes, tool parsers, exact artifact provenance, and rollback path.

```text
recipes/   → structured runtime and model contract
runbooks/  → why the contract exists; launch, validation, recovery
scripts/   → canonical switching helpers where a lane has one
benchmarks/→ dated observations, not evergreen marketing numbers
```

### 3. Start only one lane

Before a switch, stop the current inference or video workload, stop lingering user scopes, and verify that unified memory is actually free. Never assume a killed process released the scope that owns it.

```bash
# Inspect first. Do not use this as a blind switch command.
ssh jaita@larryspark.local 'free -h; systemctl --user --no-pager list-units --type=service --state=running'
```

### 4. Validate before an Arena submission

```bash
ssh jaita@larryspark.local \
  'python3 ~/verify_arena_recipe.py @styles01/<recipe-name>'
```

Only a passing gate may proceed to a native Arena benchmark.

---

## Native Spark Arena benchmarking

A `llama-benchy` run is not a Spark Arena submission. A native `sparkrun arena benchmark run` is the path that launches, measures, and uploads the result.

```bash
# REQUIRED: validate first. It fails closed on malformed submission metadata.
ssh jaita@larryspark.local \
  'python3 ~/verify_arena_recipe.py @styles01/<recipe-name>'

# REQUIRED: launch detached on the Spark. Do not attach a multi-hour benchmark to SSH.
ssh jaita@larryspark.local \
  'setsid nohup python3 -m sparkrun arena benchmark run @styles01/<recipe-name> --cluster spark \
    > /tmp/<recipe-name>-arena.log 2>&1 < /dev/null &'
```

### Submission invariants

- `model:` must be the public Hugging Face model ID—not a local path.
- `cluster_config.resolved_model_path` is not allowed in an Arena submission recipe.
- The runtime must be Arena-recognized.
- Any referenced container must be public and anonymously pullable.
- The model must fit the **entire** GB10 unified-memory budget: weights, KV cache, compile workspace, OS, and active services.
- Benchmark profiles queue concurrency; do **not** inflate server slots merely to match client concurrency. Preserve per-slot context for the deepest task.

See [SPARKRUN-REFERENCE.md](SPARKRUN-REFERENCE.md) for the CLI reference and [benchmarks/](benchmarks/) for dated evidence.

---

## Recipe catalog

### Production and production-shaped lanes

| Model / lane | Runtime | Context posture | Runbook | Recipe |
|---|---|---|---|---|
| Ling 3.0 Flash INT4 | Ling vLLM fork | 256K / 2 lanes | [Open](runbooks/ling-3.0-flash-int4.md) | [Open](recipes/ling-3.0-flash-vllm.yaml) |
| Qwen 3.8 Flash-Next Q3 3-lane | llama.cpp qwen4exp | 3 × 220K | [Open](runbooks/qwen38-flash-next-q3-3lane.md) | [Open](recipes/qwen3.8-flash-next-q3-3lane.yaml) |
| Qwen 3.8 Flash-Next Q4 | llama.cpp qwen4exp | 262K / one lane | [Open](runbooks/qwen38-flash-next-image.md) | [Open](recipes/qwen3.8-flash-next-image.yaml) |
| DeepSeek-V4-Flash 0731 | DS4 CUDA | 196K–1M profiles | [Open](runbooks/deepseek-v4-flash-ds4.md) | [Open](recipes/deepseek-v4-flash-0731-ds4.yaml) |
| Qwen 3.8 27B NVFP4 | GB10 vLLM | 256K / native MTP | [Open](runbooks/qwen-38-27b.md) | [Open](recipes/qwen-38-27b.yaml) |
| Qwen 122B DFlash | custom vLLM | 262K class | [Open](runbooks/qwen-122b.md) | [Open](recipes/qwen-122b.yaml) |
| Qwen 35B NVFP4 | vLLM | 262K class | [Open](runbooks/qwen-35b.md) | [Open](recipes/qwen-35b.yaml) |
| Laguna S 2.1 | vLLM | long-context code | [Open](runbooks/laguna-s-2.1.md) | [Open](recipes/laguna-s-2.1.yaml) |
| Muse-Glimmer 30B | SGLang | 131K class | [Open](runbooks/muse-glimmer-30b.md) | [Open](recipes/muse-glimmer-30b.yaml) |
| Nemotron 3.5 Lightning 30B | vLLM | 100K / multi-agent | [Open](runbooks/nemotron-3.5-lightning-30b-a3b-nvfp4.md) | [Open](recipes/nemotron-3.5-lightning-30b-a3b-nvfp4.yaml) |

### Staged candidates

Candidates are intentionally separate from healthy production lanes. A recipe existing here does **not** authorize a switch.

| Candidate | Why it is interesting | Current boundary | Artifacts |
|---|---|---|---|
| Qwen 3.8 27B EXL3 3.5bpw | compact text-only EXL3 target with native-MTP potential | 131K / batch 1 baseline; MTP depth must be independently verified | [Runbook](runbooks/qwen3.8-27b-exl3-native-mtp-experimental.md) · [Recipe](recipes/qwen3.8-27b-exl3-native-mtp-experimental.yaml) |
| GLM-5.3 Flash EXL3 K2 | credible one-Spark 258K prefill evidence | one 64K MTP lane first; 258K is research-only | [Runbook](runbooks/glm-5.3-flash-exl3-k2.md) · [Recipe](recipes/glm-5.3-flash-exl3-k2.yaml) |
| GLM-5.3 Flash Q2 MTP text | lower-headroom-risk local llama.cpp variant | one 32K q8-KV text lane first; 128K/256K are gated single-request probes | [Runbook](runbooks/glm-5.3-flash-q2-mtp-text-experimental.md) · [Recipe](recipes/glm-5.3-flash-q2-mtp-text-experimental.yaml) |
| DeepSeek V4 Flash Vision-Exp Q2 | separate experimental image route | 4K / one session; do not mutate production DS4 | [Runbook](runbooks/deepseek-v4-flash-vision-exp-ds4-experimental.md) · [Recipe](recipes/deepseek-v4-flash-vision-exp-ds4-experimental.yaml) |

---

## Benchmark interpretation

A benchmark row without its setup is a story, not evidence. Each published number should state:

- model revision and quantization;
- runtime, container/image digest, and critical patches;
- context depth and output length;
- requested concurrency versus actual server scheduling;
- sampling and thinking mode;
- TTFT, decode rate, tail latency, acceptance, and peak unified-memory use;
- failures, restarts, and recovery behavior.

### Common traps on GB10

| Trap | What to do instead |
|---|---|
| Treating `gpu_memory_utilization` as physical free memory | Budget the full 121 GB unified pool, including OS, compiler workspace, and KV. |
| Copying a 5090 or H200 throughput claim | Reproduce on GB10; bandwidth and runtime behavior do not transfer. |
| Disabling thinking to make a score look good | Keep thinking on for agentic quality tests unless the benchmark explicitly studies that switch. |
| Calling a 262K HTTP response “long-context correctness” | Run retrieval/needle tests and retain the actual completion. |
| Calling queued batch-1 requests “concurrency” | Record queue delay and fairness; do not advertise independent lanes. |
| Clearing compile caches during ordinary model switches | Preserve compiled assets unless the same model’s speculative configuration or runtime has changed. |

---

## Repository map

```text
sparkrun-recipes/
├── recipes/       Machine-readable SparkRun and Arena contracts
├── runbooks/      Operations notes, validation gates, recovery procedures
├── scripts/       Canonical switch/stage helpers
├── benchmarks/    Dated measurements and raw benchmark notes
├── docker/        Dockerfiles, launchers, and source patches
├── runtime/       Custom SparkRun runtime plugins
└── .sparkrun/     Registry manifest
```

---

## Contributing

Contributions are welcome when they make a lane more reproducible, safer, or better measured.

1. Open an issue with the model, quantization, runtime, GB10 hardware details, and primary source links.
2. Do not submit a benchmark screenshot alone. Include the exact command, model revision, image digest, context, concurrency, sampling, and failure state.
3. New recipes need a paired runbook and an explicit safe staging path.
4. New containers must be public before Arena use.
5. Never smuggle a local model path or a `resolved_model_path` into an Arena submission.

### Support the project

If this repository saved you a few failed boots, you can support the work at [Buy Me a Coffee](https://buymeacoffee.com/aitamedia), or follow [@jaita on X](https://x.com/jaita) for new recipes and benchmark notes.

> GitHub Sponsors is not currently active for `styles01`; the repository’s native GitHub **Sponsor** button is configured to route to the active Buy Me a Coffee page until a GitHub Sponsors profile is enabled.

---

## Credits

This repository builds on a broad set of primary sources and practitioner work. Individual runbooks cite the exact model cards, commits, papers, issues, and upstream recipes they rely on.

Special thanks to Drowzeys, 0xBakeer, Mia AI Lab, Daniel Han, Bleysg, antirez, the Qwen, DeepSeek, NVIDIA, SGLang, vLLM, and llama.cpp communities.
