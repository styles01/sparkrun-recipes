# Recipe: DeepSeek-V4-Flash (0731) — ds4 CUDA Engine

**Status:** ✅ Production — DSpark k=2, native execution
**Served name:** `deepseek-v4-flash`
**Engine:** ds4 CUDA (Entrpi/ds4 fork v0.5.0) — native C/CUDA binary by @bleysg/@antirez
**Quant:** IQ2XXS 2-bit with imatrix (~87 GB GGUF)
**Updated:** August 2, 2026

> **Recipe contract:** [`recipes/deepseek-v4-flash-0731-ds4.yaml`](../recipes/deepseek-v4-flash-0731-ds4.yaml)

## Overview

DeepSeek-V4-Flash 0731 (284B params, 12B active MoE) served on a single DGX Spark via the ds4 CUDA engine — a ground-up C/CUDA inference engine with DSpark lossless speculative decoding. No Docker, no Python, no vLLM.

**Why ds4 CUDA (not vLLM):**
- The 0731 weights are 167 GB in FP8 safetensors — too large for the Spark's 121 GB unified memory via vLLM (OOM during loading)
- The ds4 engine uses IQ2XXS GGUF quantization (87 GB) which fits comfortably
- Native C/CUDA with direct GPU access — no framework overhead
- DSpark speculative decoding built in — 75% acceptance at k=2
- 1000 tok/s prefill, 20 tok/s decode, 200ms TTFT

## Model Files on Spark

```
~/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf  (~87 GB, main model)
~/gguf/DSpark-drafter-Q2K-Q8-0731.gguf                                             (~7 GB, DSpark drafter)
```

## Installation

```bash
# One command — builds ds4, downloads GGUFs, runs smoke test
curl -sSL https://raw.githubusercontent.com/Entrpi/ds4-on-spark/main/install.sh | bash -s -- --start
```

This installs:
- `~/code/ds4/ds4-server` — native C/CUDA binary
- `~/.local/bin/ds4-serve` — bash launcher wrapper
- GGUF model files in `~/gguf/`

## Configuration

### CLI Flags

| Flag | Value | Description |
|------|-------|-------------|
| `-m` | `~/gguf/DeepSeek-V4-Flash-...0731.gguf` | Model path |
| `-c` | `131072` | Context window (131K) |
| `--port` | `8000` | HTTP API port |
| `--host` | `0.0.0.0` | Bind address |

### Environment Variables (DS4_*)

| Env var | Value | Description |
|---------|-------|-------------|
| `DS4_BATCH_FIT_HEADROOM_MB` | `8192` | GPU memory headroom for batch planner (controls max lanes) |
| `DS4_SERVER_SERIAL_MAX_TOKENS` | `131072` | Max tokens per response (match context) |
| `DS4_SERVER_COALESCE_MAX` | `2` | Max concurrent lanes (coalesce limit) |
| `DS4_CONT_DSPARK` | `1` | Enable DSpark drafter |
| `DS4_CONT_MTP_MODE` | `2` | MTP speculative decode mode |
| `DS4_DSPARK_MODEL` | `~/gguf/DSpark-drafter-Q2K-Q8-0731.gguf` | DSpark drafter model path |

### Memory Budget

| Component | Size |
|---|---|
| Model weights (IQ2XXS) | ~87 GB on disk |
| Resident after load | ~102 GB GPU |
| Free RAM | ~16 GB |
| Context buffers | ~3.1 GB (131K × 2 lanes) |
| Comp-cache budget | ~12 GB |
| Max lanes | 2 (auto-scaled from headroom) |

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /v1/models` | List served models |
| `POST /v1/chat/completions` | OpenAI-compatible chat API (supports tools, reasoning_effort, stream) |
| `GET /metrics` | Prometheus-format metrics (`ds4_*` prefix) |

## Running with sparkrun

```bash
# Add the registry (one-time)
sparkrun registry add https://github.com/styles01/sparkrun-recipes.git

# Enable local executor (one-time, for native binaries)
sparkrun setup features enable executor.local

# Run the ds4 recipe
sparkrun run @styles01/deepseek-v4-flash-0731-ds4 --hosts <spark-ip> --trust

# Benchmark (against running instance)
sparkrun benchmark performance @styles01/deepseek-v4-flash-0731-ds4 --hosts <spark-ip> --skip-run --no-stop -b model=deepseek-v4-flash
```

## Running Directly (without sparkrun)

```bash
# Start
DS4_BATCH_FIT_HEADROOM_MB=8192 \
DS4_SERVER_SERIAL_MAX_TOKENS=131072 \
DS4_SERVER_COALESCE_MAX=2 \
ds4-serve -c 131072 --host 0.0.0.0 --port 8000

# Stop
pkill -f ds4-server
```

## Performance (August 2, 2026)

| Metric | Value |
|---|---|
| Decode (single stream) | 20 tok/s |
| Decode (2 streams aggregate) | 25 tok/s |
| Prefill | 1000 tok/s |
| TTFT | 200ms |
| DSpark acceptance | 75% (k=2) |
| Context | 131K per lane |
| Lanes | 2 |
| Memory used | 105 GB |
| Free RAM | 16 GB |

## DSpark Notes

- DSpark only accelerates at 1 concurrent request (`DS4_DSPARK_MAX_NLIVE=1` default)
- At 2+ concurrent, falls back to plain batched decode
- Quench controller auto-disables speculation when acceptance can't pay verify cost
- Acceptance: ~75% average, per-position varies

## Runtime Plugin

This recipe uses the `ds4-cuda` custom runtime plugin at [`runtime/sparkrun-ds4/`](../runtime/sparkrun-ds4/).

For sparkrun v0.3.1+, enable external plugins in `~/.config/sparkrun/config.yaml`:
```yaml
features:
  core.external_plugins: true
  executor.local: true
plugins:
  paths:
    - ~/sparkrun-recipes/runtime/sparkrun-ds4/src
```

A PR to add `ds4-cuda` as a built-in runtime in sparkrun is open at:
https://github.com/spark-arena/sparkrun/pull/241

## Credits

- **@bleysg** (Bleys Goodson) — Entrpi/ds4 fork, ds4-on-spark installer, DSpark speculative decode
- **@antirez** (Salvatore Sanfilippo) — Original ds4 engine, GGUF quantization
- **@styles01** — Recipe, runtime plugin, sparkrun integration