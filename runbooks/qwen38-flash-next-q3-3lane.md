# Runbook: Qwen3.8-Flash-Next Q3_K_XL — 3 LANES @ 220K (current deployed)

> **Source and regression gate.** Before any multi-lane or native-vision claim,
> run `scripts/verify-qwen38-flash-next-build.sh --source-dir <llama.cpp-checkout>`
> for the source used to build the server. It requires PR #27941 merge
> `36b10154383b60eb15baac2c7a40d2a5f784faa7` or an auditable descendant. Then
> complete three simultaneous, request-distinct completions with correct outputs
> and no assert/error/restart; for vision, send an image through `--mmproj` and
> check its expected semantic result with no cross-request contamination. Retain
> command output and server logs. Existing throughput and vision results below are
> historical, not revalidated by this runbook change.

> **The 3-lane config.** Daniel Han's slot fix (`8b3ed0a40`, "keep the indexer
> cache in step across server slots") lifts the `--parallel 1` constraint on the
> qwen4exp fork. We validated **3 concurrent requests with NO indexer crash**
> (`qwen4exp.cpp:284`). Result: **3 lanes @ 220K context, ~57 tok/s aggregate**
> on a single DGX Spark — beats both our old 2-lane config (43-50 t/s) and the
> vLLM NVFP4 recipe (45.2 t/s, 10x75K = shallower per-lane).

## Model
- **HF repo:** `unsloth/Qwen3.8-Flash-Next-GGUF`
- **Quant:** `UD-Q3_K_XL` — 3 shards, 90 GB total, **whole model in memory** (no NVMe pin)
- **Arch:** Qwen4 (`Qwen4ExpForConditionalGeneration`), 125B MoE (6B active) + 51B n-gram + 4B MTP ≈ 180B
- **Vision:** `mmproj-F16.gguf` (904 MB, from the same HF repo) — native multimodal

## Runtime / build
- **llama.cpp qwen4exp fork:** post-PR-27941 source verified by the gate above; older named commits are historical and do not establish post-fix provenance.
- Build for SM `121a` with CUDA 13.0 (GB10). `LD_LIBRARY_PATH` must include `build/bin`.

## Launch (whole model in memory, 3 lanes @ 220K, with vision)
```bash
cd ~/code/llama.cpp-qwen4exp/build/bin && setsid nohup env LD_LIBRARY_PATH=$PWD ./llama-server \
  -m ~/gguf/qwen3.8-flash-next/Q3_K_XL/UD-Q3_K_XL/Qwen3.8-Flash-Next-UD-Q3_K_XL-00001-of-00003.gguf \
  --alias qwen3.8-flash-next --n-gpu-layers 999 --ctx-size 660000 --parallel 3 \
  --spec-type ngram-mod --temp 1.0 --top-p 0.95 --top-k 20 \
  --host 0.0.0.0 --port 8000 \
  --mmproj /home/jaita/gguf/qwen3.8-flash-next/mmproj-F16.gguf \
  < /dev/null > /tmp/qwen4-q3-p3.log 2>&1 &
```
- **`--ctx-size 660000 --parallel 3`** = 220K per slot (220160 each)
- **Whole model in memory** — NO `-ot per_layer_token_embd=CPU` / `-lm mmap` NVMe pin.
- **`--mmproj`** = native vision. Server reports `capabilities: ["completion","multimodal"]`.

## IMPORTANT: how lanes 2 & 3 engage
`--parallel 3` = 3 slots for **simultaneous** requests. Lanes 2 and 3 are **idle unless 2+ (or 3+) requests hit concurrently**:
- Sequential requests → 1 lane → ~22 tok/s
- 2 concurrent → both lanes → ~38-40 tok/s
- **3 concurrent → all three → ~57 tok/s aggregate** (validated: 19+19+19)

This is NOT model parallelism (splitting one request across lanes) — it's concurrent request capacity.

## Perf measurements (GB10, Q3_K_XL, 3 lanes @ 220K, current deploy)
| Workload | tok/s |
|---|---|
| Single lane (sequential) | ~22 |
| 2 lanes concurrent | ~38-40 aggregate |
| **3 lanes concurrent (validated)** | **~57 aggregate (19+19+19)** |
| Vision (multimodal) | verified working |

## Memory
- Q3_K_XL 90 GB + 15.8 GB KV (3×220K × 24KB/token f16) + ~8 GB buffers = **~114 GB used, ~5 GB headroom** on 119 GB usable. Tight but fits (no quantized KV needed).
- This is tighter than the 2-lane config (~15 GB headroom) — monitor `free` if you push beyond 220K/lane.

## Context advantage
3×220K = **660K aggregate context**, deeper per-lane than the vLLM NVFP4 recipe's 10×75K. And ~57 t/s beats its 45.2. Best of both: more context pool AND faster aggregate.

## Known issues / constraints
- **Lanes 2 & 3 need concurrent load** — idle under sequential requests (see above).
- **KV must stay f16** — quantized KV (`-ctk/-ctv`) aborts (`qwen4exp.cpp:544`).
- **No MTP on this GGUF** — the Flash-Next GGUF ships no `nextn` tensors. (MTP tested = too heavy vs ngram-mod for concurrent aggregate.)
- **Do NOT set `--spec-ngram-mod-n-min/max` high** — default n_max=3 is correct; drafting 48-64 tokens burns compute (drops to ~14-18 tok/s, measured).
- **Served on `0.0.0.0`** so Loca on the Mac reaches it.
- **3 lanes validated**; pushing to 4 lanes is unproven (indexer crash risk above 3).

## Source references
- 0xBeker: https://github.com/0xBeker/qwen38-flash-next-spark
- Daniel Han slot fix: commit `8b3ed0a40` in the qwen4exp branch
- Model: https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF
- Supersedes: `recipes/qwen3.8-flash-next-q3-2lane.yaml` (2 lanes @ 200K, ~43-50 t/s)
