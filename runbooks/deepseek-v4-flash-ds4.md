# Recipe: DeepSeek-V4-Flash (0731) — ds4 CUDA Engine

**Status:** ✅ DSpark MTP k=2, native execution (no Docker)
**Served name:** `deepseek-v4-flash`
**Engine:** ds4 CUDA (Bleysg's fork of antirez/ds4) — native C/CUDA binary
**Quant:** IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix
**Updated:** August 2, 2026

> **Recipe contract:** [`recipes/deepseek-v4-flash-0731-ds4.yaml`](../recipes/deepseek-v4-flash-0731-ds4.yaml)

## Why ds4 (not vLLM)

The existing [vLLM recipe](deepseek-v4-flash.md) uses a Moet venv on top of vLLM for FP8 MTP. The ds4 engine provides an alternative native path:

- **Native C/CUDA** — no Docker container, no Python venv overhead
- **GGUF quantization** — IQ2XXS (87 GB) vs FP8 safetensors (~167 GB)
- **DSpark drafter** — speculative decoding via a separate 7 GB drafter GGUF
- **Lower latency** — direct CUDA, no framework layers

## Model Files on Spark

```
~/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf  (~87 GB, main model)
~/gguf/DSpark-drafter-Q2K-Q8-0731.gguf                                             (~7 GB, MTP drafter)
```

## ds4 Binary

```
~/code/ds4/ds4-server         # native C/CUDA binary
~/.local/bin/ds4-serve        # bash launcher wrapper
```

Install: `curl -sSL https://raw.githubusercontent.com/Entrpi/ds4-on-spark/main/install.sh | bash -s -- --start`

## Configuration

### CLI Flags

| Flag | Value | Description |
|------|-------|-------------|
| `-m` | `~/gguf/DeepSeek-V4-Flash-...0731.gguf` | Model path |
| `-c` | `131072` | Context window |
| `--port` | `8000` | HTTP API port |
| `--host` | `0.0.0.0` | Bind address |
| `-ngl` | `-1` | All layers on GPU |
| `-t` | `8` | CPU threads |

### Environment Variables (DS4_*)

| Env var | Value | Description |
|---------|-------|-------------|
| `DS4_BATCH_FIT_HEADROOM_MB` | `8192` | GPU memory headroom for batch planner |
| `DS4_SERVER_SERIAL_MAX_TOKENS` | `131072` | Max tokens per response |
| `DS4_SERVER_COALESCE_MAX` | `2` | Decode coalescing steps |
| `DS4_CONT_DSPARK` | `1` | Enable DSpark drafter |
| `DS4_CONT_MTP_MODE` | `2` | MTP speculative decode mode |
| `DS4_DSPARK_MODEL` | `~/gguf/DSpark-drafter-Q2K-Q8-0731.gguf` | Drafter model path |

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /v1/models` | List served models |
| `POST /v1/chat/completions` | OpenAI-compatible chat API |
| `GET /metrics` | Prometheus-format metrics (`ds4_*` prefix) |

## Running with sparkrun

```bash
# Add the registry (one-time)
sparkrun registry add https://github.com/styles01/sparkrun-recipes.git

# Run the ds4 recipe
sparkrun run @styles01/deepseek-v4-flash-0731-ds4 --hosts <spark-host>

# Check status
sparkrun status

# View logs
sparkrun logs <workload-id>

# Stop
sparkrun stop <workload-id>
```

## Runtime Plugin

This recipe uses the `ds4-cuda` custom runtime plugin, located at
[`runtime/sparkrun-ds4/`](../runtime/sparkrun-ds4/). Install it with:

```bash
pip install ./runtime/sparkrun-ds4
```

The plugin registers via entry points:
```toml
[project.entry-points."sparkrun.runtimes"]
ds4-cuda = "sparkrun_ds4:Ds4CudaRuntime"
```