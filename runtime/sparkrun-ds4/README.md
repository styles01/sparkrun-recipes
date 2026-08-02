# sparkrun-ds4

A **sparkrun** custom runtime plugin for the [ds4](https://github.com/Bleysg/ds4) CUDA engine — Bleysg's fork of antirez/ds4.

ds4 is a native C/CUDA inference server that serves GGUF models with an OpenAI-compatible HTTP API. It runs as a **host process** (not a Docker container), so this runtime integrates with sparkrun's `LocalExecutor` (no-container) path.

## Features

- **`Ds4CudaRuntime`** plugin class registered as `ds4-cuda`
- Translates recipe `defaults` → ds4 CLI flags (`-m`, `-c`, `--port`, `--host`)
- Translates recipe `env` → `DS4_*` environment variables
- Supports DSpark speculative-decoding drafter configuration
- `default_executor()` returns `"local"` so recipes don't need to set it explicitly
- OpenAI-compatible API at `/v1/chat/completions`, `/v1/models`
- Prometheus metrics at `/metrics` (ds4-native)

## Installation

```bash
pip install sparkrun-ds4
```

Or in development mode:

```bash
cd runtime/sparkrun-ds4
pip install -e .
```

The entry point is auto-discovered by sparkrun's SAF plugin system:

```toml
[project.entry-points."sparkrun.runtimes"]
ds4-cuda = "sparkrun_ds4:Ds4CudaRuntime"
```

## Usage

Add the recipe registry and run:

```bash
sparkrun registry add https://github.com/styles01/sparkrun-recipes.git
sparkrun run @ds4-cuda/deepseek-v4-flash-0731-ds4
```

## Recipe Example

```yaml
recipe_version: "2"
model: ~/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf
runtime: ds4-cuda
executor: local

defaults:
  context: 131072
  port: 8000
  host: 0.0.0.0

env:
  DS4_BATCH_FIT_HEADROOM_MB: "8192"
  DS4_SERVER_SERIAL_MAX_TOKENS: "131072"
  DS4_SERVER_COALESCE_MAX: "2"
  DS4_CONT_DSPARK: "1"
  DS4_CONT_MTP_MODE: "2"
  DS4_DSPARK_MODEL: "~/gguf/DSpark-drafter-Q2K-Q8-0731.gguf"
```

## ds4 CLI Flag Mapping

| Recipe key | ds4 flag |
|---|---|
| `model` | `-m <path>` |
| `context` | `-c <size>` |
| `port` | `--port <port>` |
| `host` | `--host <addr>` |
| `threads` | `-t <n>` |
| `n_gpu_layers` | `-ngl <n>` |

## DS4_* Environment Variables

All ds4 tuning knobs are environment variables (not CLI flags). Set them in the recipe `env:` block:

| Env var | Description |
|---|---|
| `DS4_BATCH_FIT_HEADROOM_MB` | GPU memory headroom for batch fit planner |
| `DS4_SERVER_SERIAL_MAX_TOKENS` | Max tokens per response |
| `DS4_SERVER_COALESCE_MAX` | Decode steps to coalesce |
| `DS4_CONT_DSPARK` | Enable DSpark drafter (1/0) |
| `DS4_CONT_MTP_MODE` | MTP mode (2 = speculative decode) |
| `DS4_DSPARK_MODEL` | Path to DSpark drafter GGUF |

## License

MIT