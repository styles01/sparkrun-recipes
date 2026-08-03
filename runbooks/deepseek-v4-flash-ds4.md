# DeepSeek-V4-Flash 0731 on DS4 CUDA Engine

**Status:** ✅ Production — DSpark k=2, native execution  
**Served name:** `deepseek-v4-flash`  
**Engine:** ds4 CUDA (Entrpi/ds4 fork v0.5.0) — native C/CUDA binary  
**Author:** [@bleysg](https://x.com/bleysg) (Bleys Goodson) / [@antirez](https://x.com/antirez) (Salvatore Sanfilippo)  
**Updated:** August 2, 2026

> **Recipe contract:** [`recipes/deepseek-v4-flash-0731-ds4.yaml`](../recipes/deepseek-v4-flash-0731-ds4.yaml)

---

## ⭐ Featured Recipe

The smartest model available, now serving on a single DGX Spark via [Bleysg's ds4 CUDA engine](https://github.com/Entrpi/ds4-on-spark) — a ground-up C/CUDA inference engine with DSpark lossless speculative decoding.

| Metric | Value |
|---|---|
| **Model** | DeepSeek-V4-Flash 0731 (284B params, 12B active MoE) |
| **Engine** | ds4 CUDA (Entrpi/ds4 fork v0.5.0) |
| **Quant** | IQ2XXS 2-bit with imatrix (~87 GB GGUF) |
| **Spec Decode** | DSpark k=2, ~75% acceptance |
| **Context** | 131K per lane, 2 lanes |
| **Decode** | ~20 tok/s single-stream, ~25 tok/s aggregate |
| **Prefill** | ~1000 tok/s |
| **TTFT** | ~200ms |
| **Memory** | 105 GB used, 16 GB free |

---

## Why ds4 CUDA (not vLLM)

The 0731 weights are 167 GB in FP8 safetensors — too large for the Spark's 121 GB unified memory via vLLM (OOM during loading). The ds4 engine solves this with:

- **IQ2XXS GGUF quantization** — 87 GB on disk, fits comfortably
- **Native C/CUDA** — no Docker, no Python, no framework overhead
- **DSpark speculative decoding** — lossless, built in, 75% acceptance at k=2
- **Direct GPU access** — 1000 tok/s prefill, 200ms TTFT

---

## Quick Start

### 1. Install the ds4 engine

```bash
# One command — builds ds4, downloads GGUFs, runs smoke test
curl -sSL https://raw.githubusercontent.com/Entrpi/ds4-on-spark/main/install.sh | bash -s -- --start
```

This installs:
- `~/code/ds4/ds4-server` — native C/CUDA binary
- `~/.local/bin/ds4-serve` — bash launcher wrapper
- GGUF model files in `~/gguf/`

### 2. Run with sparkrun

```bash
# Add our registry (one-time)
sparkrun registry add https://github.com/styles01/sparkrun-recipes.git

# Enable local executor for native binaries (one-time)
sparkrun setup features enable executor.local

# Launch DS4-Flash on a single Spark
sparkrun run @styles01/deepseek-v4-flash-0731-ds4 --hosts <spark-ip> --trust

# Benchmark it
sparkrun benchmark performance @styles01/deepseek-v4-flash-0731-ds4 --hosts <spark-ip> --skip-run --no-stop -b model=deepseek-v4-flash
```

### 3. Or run directly (without sparkrun)

```bash
DS4_BATCH_FIT_HEADROOM_MB=8192 \
DS4_SERVER_SERIAL_MAX_TOKENS=131072 \
DS4_SERVER_COALESCE_MAX=2 \
ds4-serve -c 131072 --host 0.0.0.0 --port 8000
```

---

## Configuration

### CLI Flags

| Flag | Value | Description |
|------|-------|-------------|
| `-m` | `~/gguf/DeepSeek-V4-Flash-...0731.gguf` | Model path (auto-detected by ds4-serve) |
| `-c` | `131072` | Context window (131K) |
| `--port` | `8000` | HTTP API port |
| `--host` | `0.0.0.0` | Bind address |

### Environment Variables (DS4_*)

| Env var | Value | Description |
|---------|-------|-------------|
| `DS4_BATCH_FIT_HEADROOM_MB` | `8192` | GPU memory headroom (controls max lanes) |
| `DS4_SERVER_SERIAL_MAX_TOKENS` | `131072` | Max tokens per response (match context) |
| `DS4_SERVER_COALESCE_MAX` | `2` | Max concurrent lanes |
| `DS4_CONT_DSPARK` | `1` | Enable DSpark drafter |
| `DS4_CONT_MTP_MODE` | `2` | MTP speculative decode mode |
| `DS4_DSPARK_MODEL` | `~/gguf/DSpark-drafter-Q2K-Q8-0731.gguf` | DSpark drafter model path |

### Memory Budget

| Component | Size |
|---|---|
| Model weights (IQ2XXS) | ~87 GB on disk |
| Resident after load | ~102 GB GPU |
| Context buffers | ~3.1 GB (131K × 2 lanes) |
| Comp-cache budget | ~12 GB |
| Free RAM | ~16 GB |
| Max lanes | 2 (auto-scaled from headroom) |

---

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /v1/models` | List served models |
| `POST /v1/chat/completions` | OpenAI-compatible chat API (supports tools, reasoning_effort, stream) |
| `GET /metrics` | Prometheus-format metrics (`ds4_*` prefix) |

---

## DSpark Speculative Decoding

- DSpark only accelerates at 1 concurrent request (`DS4_DSPARK_MAX_NLIVE=1` default)
- At 2+ concurrent, falls back to plain batched decode
- Quench controller auto-disables speculation when acceptance can't pay verify cost
- Acceptance: ~75% average at k=2

---

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

---

## Credits

- **@bleysg** (Bleys Goodson) — Entrpi/ds4 fork, ds4-on-spark installer, DSpark speculative decode
- **@antirez** (Salvatore Sanfilippo) — Original ds4 engine, GGUF quantization
- **@styles01** — Recipe, runtime plugin, sparkrun integration