# Runbook: Qwen3.8-Flash-Next (Q4_K_XL) on a single DGX Spark — our own llama.cpp container

> **The headline result:** a ~180B-parameter model runs on ONE 128GB DGX Spark (GB10),
> at Q4 quality, full 262K context, **~19-22 tok/s decode** (up to ~45 tok/s on
> copy-heavy n-gram-speculative work). The trick: the 51B n-gram (PLE) table is pinned
> to CPU and served from NVMe, never GPU-resident.

> **This runbook covers OUR OWN public container** `ghcr.io/styles01/qwen38-flash-next:q4`
> (the "container" recipe: `recipes/qwen3.8-flash-next-image.yaml`). A separate
> Q2-native recipe (`recipes/qwen3.8-flash-next-llamacpp.yaml`) is now **superseded**.

## What you get

- **Model:** `unsloth/Qwen3.8-Flash-Next-GGUF` quant `UD-Q4_K_XL` — 4 shards, ~104 GiB total, only ~77 GiB resident
- **Arch:** Qwen4 (`Qwen4ExpForConditionalGeneration`), 125B MoE (6B active) + 51B n-gram + 4B MTP ≈ 180B
- **Runtime:** llama.cpp `qwen4exp` fork (PR [ggml-org/llama.cpp#27742](https://github.com/ggml-org/llama.cpp/pull/27742), commit `035e227`) + `canreuse-qwen4exp.patch` (+2.8% decode)
- **Container:** **`ghcr.io/styles01/qwen38-flash-next:q4`** (public, ours)

> Reference `recipes/qwen3.8-flash-next-image.yaml` for the recipe contract.

## Why this fits in 119 GB (the 0xBakeer trick)

The 51B n-gram/PLE table is one tensor `per_layer_token_embd.weight` (shape `[160, 320001536]`).
It is a **pure lookup** (gathers ~16 rows/token via a 3-gram hash) — never a matmul — so it can
live on disk, not RAM:

```
-ot per_layer_token_embd=CPU   # pin the 51B tensor to the CPU backend, never GPU
-lm mmap                        # serve that table from NVMe via the OS page cache
```

Steady state ~95 GiB used / ~26 GiB page cache / RSS ~1.4 GiB. Fits 121 GiB with room for
KV (~6 GiB at 262K ctx).

## Exact current Q4 launch (the container entrypoint + our perf tweaks)

```
./llama-server -m <UD-Q4_K_XL 00001-of-00004.gguf> \
  --alias qwen3.8-flash-next \
  -lm mmap -ot per_layer_token_embd=CPU \
  --n-gpu-layers 999 --ctx-size 262144 --parallel 1 \
  --spec-type ngram-mod \
  --temp 1.0 --top-p 0.95 --top-k 20 \
  --host 0.0.0.0 --port 8000
```

The container entrypoint (`docker/qwen38-flash-next/launch.sh`) runs exactly this, with
`MODEL_DIR` (default `/models`) auto-downloaded from HF if not bind-mounted.

## Build & publish the container (on the Mac)

```bash
# 1. Pull the llama.cpp qwen4exp build dir from the Spark (build/bin/*):
#    scp -r jaita@192.168.2.185:/home/jaita/code/llama.cpp-qwen4exp/build/bin ./build
# 2. Write docker/qwen38-flash-next/{Dockerfile,launch.sh}
# 3. Build (aarch64/CUDA 13 base) + push on the Spark (the Mac's Docker Desktop
#    often hangs on the privileged socket; the Spark's docker works):
#    ssh jaita@192.168.2.185 'cd /tmp/qwen4-docker && docker build -t ghcr.io/styles01/qwen38-flash-next:q4 .'
#    docker login ghcr.io -u styles01 --password-stdin
#    docker push ghcr.io/styles01/qwen38-flash-next:q4
# 4. The Dockerfile's org.opencontainers.image.source label links the package to the
#    PUBLIC styles01/sparkrun-recipes repo, which makes the GHCR package inherit
#    public visibility. (No REST mutation endpoint exists — repo-linking via this
#    label is the supported mechanism.)
```

## Serve it

Run the launch script exactly as above (it is what the container entrypoint does).
To keep the server alive across an SSH disconnect:

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

## Internal benchmark

```bash
# On the Spark:
cd /home/jaita/code/llama.cpp-qwen4exp/build/bin
# simple decode timing via the OpenAI-compatible endpoint on 0.0.0.0:8000/v1:
curl http://0.0.0.0:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-flash-next","prompt":"Write a haiku about the moon.","max_tokens":200,"stream":false}'
# ...and read the token timings from the server log (slot print_timing line).
```
For spark-arena: benchmark with `--skip-run` against the live server at `0.0.0.0:8000/v1`.

## Perf tweaks we found

- **`-ot per_layer_token_embd=CPU` + `-lm mmap`** — the 0xBake trick (see above). Makes the whole 180B fit.
- **`canreuse-qwen4exp.patch`** (+2.8% decode) from `0xBakeer/qwen38-flash-next-spark`.
- **Keep the stock ngram-mod config** — do NOT set `--spec-ngram-mod-n-min/max` high. The default `n_max=3` is correct; drafting 48-64 tokens with ~17-39% acceptance burns compute on rejected drafts and drops decode to ~14-18 tok/s (measured). The proven baseline is just `--spec-type ngram-mod --temp 1.0 --top-p 0.95 --top-k 20`.

## Perf measurements (GB10, Q4_K_XL, clean baseline)

| Workload | tok/s |
|---|---|
| Free-form decode | ~22 |
| Prefill @16K ctx | ~405 |
| Copy-heavy (ngram-mod spec) | 22-45 |

## Known issues / constraints (verified at commit `035e227`)

- **`--parallel 1` REQUIRED** — a 2nd in-flight request aborts (`GGML_ASSERT qwen4exp.cpp:284`). No fix yet.
- **Quantized KV (`-ctk/-ctv`) aborts** (`qwen4exp.cpp:544`). Keep KV f16. (KV is only ~24 B/token, so 262K ≈ 6 GiB — quantizing saves little.)
- **No MTP on this GGUF** — the Flash-Next GGUF ships no `nextn` tensors (`no_mtp=True` in the converter). MTP wouldn't help this MoE anyway.
- **`--fit` crashes** (`ggml.c:1804` — `LLM_ARCH_QWEN4EXP` not in `graph_max_nodes`).
- **Served on `0.0.0.0`** (NOT 127.0.0.1) so other machines (e.g. Loca on the Mac) can reach it.
- **Build only for SM `121a` (GB10) w/ CUDA 13.0**; `LD_LIBRARY_PATH` must include `build/bin` (sibling `.so`).

## Source references

- [0xBakeer/qwen38-flash-next-spark](https://github.com/0xBakeer/qwen38-flash-next-spark)
- llama.cpp PR [#27742](https://github.com/ggml-org/llama.cpp/pull/27742)
- Model: [unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF)
- Our image: `ghcr.io/styles01/qwen38-flash-next:q4`
- Our Docker build: `docker/qwen38-flash-next/`
