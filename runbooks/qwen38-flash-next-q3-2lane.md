# Runbook: Qwen3.8-Flash-Next Q3_K_XL — 2 LANES @ 200K (alternative recipe)

> **Source and regression gate.** Before any two-lane or native-vision claim,
> run `scripts/verify-qwen38-flash-next-build.sh --source-dir <llama.cpp-checkout>`
> for the source used to build the server. It requires PR #27941 merge
> `36b10154383b60eb15baac2c7a40d2a5f784faa7` or an auditable descendant. Then
> complete two simultaneous, request-distinct completions with correct outputs and
> no assert/error/restart; for vision, send an image through `--mmproj` and check
> its expected semantic result with no cross-request contamination. Retain logs.
> Existing throughput and vision results below are historical, not revalidated by
> this runbook change.

> **The 2-lane unlock.** This is the first llama.cpp build to lift the `--parallel 1`
> constraint on Qwen3.8-Flash-Next. Daniel Han's slot fix (`8b3ed0a40`) in the qwen4exp
> fork fixed the `qwen4exp.cpp:284` GGML_ASSERT that crashed on a 2nd concurrent request.
> Result: **2 lanes @ 200K context, peak ~50 tok/s aggregate** on a single DGX Spark.

## Model
- **HF repo:** `unsloth/Qwen3.8-Flash-Next-GGUF`
- **Quant:** `UD-Q3_K_XL` — 3 shards, 90 GB total, **whole model in memory** (no NVMe pin)
- **Arch:** Qwen4 (`Qwen4ExpForConditionalGeneration`), 125B MoE (6B active) + 51B n-gram + 4B MTP ≈ 180B

## Why this works (the 2-lane fix)
The stock qwen4exp fork (commit `035e227`) crashes on a 2nd concurrent request:
`GGML_ASSERT` at `qwen4exp.cpp:284` (the QSA indexer cache desyncs from the attention
cache on slot switch). **Daniel Han's commit `8b3ed0a40`** ("qwen4exp: keep the indexer
cache in step across server slots") fixes the restore path so the indexer stays in step
across slots. Pull the latest fork (`ef6876693` + the slot fix) and rebuild.

## Runtime / build
- **llama.cpp qwen4exp fork:** PR [ggml-org/llama.cpp#27742](https://github.com/ggml-org/llama.cpp/pull/27742), latest commit `ef6876693` + Daniel Han's slot fix `8b3ed0a40`
- **Patches:** `canreuse-qwen4exp.patch` (+2.8% decode) — re-applied after the pull (conflicts with the slot fix; resolve by keeping both)
- Build for SM `121a` with CUDA 13.0 (GB10). `LD_LIBRARY_PATH` must include `build/bin`.

## Launch (whole model in memory, 2 lanes @ 200K)
```bash
./llama-server -m <UD-Q3_K_XL 00001-of-00003.gguf> \
  --alias qwen3.8-flash-next \
  --n-gpu-layers 999 --ctx-size 400000 --parallel 2 \
  --spec-type ngram-mod --temp 1.0 --top-p 0.95 --top-k 20 \
  --host 0.0.0.0 --port 8000 \
  --mmproj /home/jaita/gguf/qwen3.8-flash-next/mmproj-F16.gguf
```
- **`--ctx-size 400000 --parallel 2`** = 200K per slot (200192 each)
- **`--mmproj`** = native vision (see [Vision](#vision) below)
- **Whole model in memory** — NO `-ot per_layer_token_embd=CPU` / `-lm mmap` NVMe pin. At 2 lanes the n-gram table gets hammered, so keeping it resident is better.
- Detached (survives SSH):
  ```bash
  setsid nohup bash scripts/serve-qwen38-flash-next-q3-2lane.sh > /tmp/qwen4-q3-p2.log 2>&1 < /dev/null &
  ```

## Vision (native multimodal)
Native vision is enabled on this config by the `--mmproj` flag:
- **mmproj file:** `/home/jaita/gguf/qwen3.8-flash-next/mmproj-F16.gguf` (904 MB, SHA-256 `1f7b7f0b984cf065c604360c29c8098362ed61b290db0ff12c6f360bb1a8a980`)
- **Source:** `unsloth/Qwen3.8-Flash-Next-GGUF`
- **With it loaded**, the server reports capabilities `["completion","multimodal"]`.
- **Verified** on the live Q3 server — accurate image description.
- The only launch change vs. the text-only config is the added `--mmproj` flag; all other flags (2 lanes, 200K ctx, ngram-mod spec) are unchanged.

## IMPORTANT: how the 2nd lane engages
`--parallel 2` = 2 slots available for **simultaneous** requests. The 2nd lane is
**idle unless 2+ requests hit concurrently**:
- **Sequential requests** (one at a time) → 1 lane → ~22 tok/s
- **Concurrent requests** (2+ simultaneous, e.g. a subagent fan-out) → both lanes → **~43-50 tok/s aggregate**

This is NOT model parallelism (splitting one request across lanes) — it's concurrent
request capacity. If your workload is strictly sequential, you'll only see 1 lane.

## Perf measurements (GB10, Q3_K_XL, 2 lanes @ 200K)
| Workload | tok/s |
|---|---|
| Single lane (sequential) | ~22 |
| 2 lanes concurrent | ~43-50 aggregate (peak 50) |
| Per-lane under concurrent load | ~12-22 each |

## Memory
- Q3_K_XL 90 GB whole-in-memory + 9.6 GB KV (2×200K × 24KB/token) + ~8 GB buffers = **~106 GB used, ~15 GB headroom** on 119 GB usable. Comfortable.
- **IQ4_XS (93.7 GB)** also fits but tighter (~8 GB headroom) — not used, held for max-quality option.

## Known issues / constraints
- **2nd lane needs concurrent load** — idle under sequential requests (see above).
- **KV must stay f16** — quantized KV (`-ctk/-ctv`) aborts (`qwen4exp.cpp:544`).
- **No MTP on this GGUF** — the Flash-Next GGUF ships no `nextn` tensors. MTP wouldn't help this MoE anyway.
- **Do NOT set `--spec-ngram-mod-n-min/max` high** — default n_max=3 is correct; drafting 48-64 tokens burns compute on rejected drafts (drops to ~14-18 tok/s, measured).
- **Served on `0.0.0.0`** (NOT 127.0.0.1) so Loca on the Mac can reach it.

## Source references
- 0xBeker: https://github.com/0xBeker/qwen38-flash-next-spark
- llama.cpp PR #27742: https://github.com/ggml-org/llama.cpp/pull/27742
- Daniel Han slot fix: commit `8b3ed0a40` in the qwen4exp branch
- Model: https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF
