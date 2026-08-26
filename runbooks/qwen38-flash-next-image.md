# Runbook: Qwen3.8-Flash-Next (Q4_K_XL) on a single DGX Spark — first-post llama.cpp

> **The headline result:** a 180B-parameter model runs on ONE 128GB DGX Spark (GB10),
> at Q4 quality, full 262K context, ~17-22 tok/s decode (up to 45 tok/s on copy-heavy
> work). The trick: the 51B n-gram (PLE) table is pinned to CPU and served from NVMe,
> never GPU-resident.

## Model
- **HF repo:** `unsloth/Qwen3.8-Flash-Next-GGUF`
- **Quant:** `UD-Q4_K_XL` — 4 shards, 103.7 GiB total, only **~77 GiB resident**
- **Arch:** Qwen4 (`Qwen4ExpForConditionalGeneration`), 125B MoE (6B active) + 51B n-gram + 4B MTP ≈ 180B

## Why this fits 119 GB (the 0xBakeer trick)
The 51B n-gram/PLE table is one tensor `per_layer_token_embd.weight` (shape `[160, 320001536]`).
It is a **pure lookup** (gathers ~16 rows/token via 3-gram hash) — never a matmul — so it can
live on disk, not RAM:
```
-ot per_layer_token_embd=CPU   # pin the 51B tensor to the CPU backend, never GPU
-lm mmap                        # serve that table from NVMe via the OS page cache
```
Steady state ~95 GiB used / ~26 GiB page cache / RSS ~1.4 GiB. Fits 121 GiB with room for
KV (~6 GiB at 262K ctx).

## Runtime / build
- **llama.cpp qwen4exp fork:** PR [ggml-org/llama.cpp#27742](https://github.com/ggml-org/llama.cpp/pull/27742), commit `035e227`
- **Patches:** `canreuse-qwen4exp.patch` (+2.8% decode, from [0xBakeer/qwen38-flash-next-spark](https://github.com/0xBakeer/qwen38-flash-next-spark))
- Build for SM `121a` with CUDA 13.0 (GB10). `LD_LIBRARY_PATH` must include `build/bin` (sibling .so).

## Our canonical launch (serve-qwen38-flash-next.sh)
```bash
./llama-server -m <UD-Q4_K_XL 00001-of-00004.gguf> \
  --alias qwen3.8-flash-next \
  -lm mmap -ot per_layer_token_embd=CPU \
  --n-gpu-layers 999 --ctx-size 262144 --parallel 1 \
  --spec-type ngram-mod \
  --flash-attn on -b 2048 -ub 2048 \
  --spec-ngram-mod-n-min 48 --spec-ngram-mod-n-max 64 \
  --host 0.0.0.0 --port 8000
```
Detached (survives SSH):
```bash
setsid nohup bash scripts/serve-qwen38-flash-next.sh > /tmp/qwen4-q4-server.log 2>&1 < /dev/null &
```

## How to check logs
```bash
ssh jaita@192.168.2.185 'tail -f /tmp/qwen4-q4-server.log'   # live throughput
ssh jaita@192.168.2.185 'tail -50 /tmp/qwen4-q4-server.log'
```
Key lines:
- `listening on http://0.0.0.0:8000` → ready
- `slot print_timing ... eval time = X ms / Y tokens (… Z tokens per second)` → decode speed
- `prompt processing ... tok/s` → prefill speed

## Perf measurements (GB10, Q4_K_XL, tweaks on)
| Workload | tok/s |
|---|---|
| Free-form decode | 17-18 |
| Prefill @16K ctx | ~405 |
| Copy-heavy (ngram-mod spec) | 22-45 |

## Known issues / constraints (verified at commit 035e227)
- **`--parallel 1` required** — a 2nd in-flight request aborts (GGML_ASSERT `qwen4exp.cpp:284`). No fix yet.
- **Quantized KV (`-ctk/-ctv`) aborts** (`qwen4exp.cpp:544`). Keep KV f16. (KV is only ~24 B/token, so 262K ≈ 6 GiB — quantizing saves little.)
- **No MTP on this GGUF** — the Flash-Next GGUF ships no `nextn` tensors (`no_mtp=True` in the converter). MTP wouldn't help this MoE anyway.
- **`--fit` crashes** (`ggml.c:1804` — `LLM_ARCH_QWEN4EXP` not in `graph_max_nodes`).
- **Served on `0.0.0.0`** (NOT 127.0.0.1) so Loca on the Mac can reach it.

## Source references
- 0xBeker: https://github.com/0xBeker/qwen38-flash-next-spark
- llama.cpp PR #27742: https://github.com/ggml-org/llama.cpp/pull/27742
- Model: https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF
