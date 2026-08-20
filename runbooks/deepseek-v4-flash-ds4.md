# DeepSeek-V4-Flash 0731 on DS4 CUDA Engine

**Status:** ✅ Production — DSpark k=2, native execution  
**Served name:** `deepseek-v4-flash`  
**Engine:** ds4 CUDA (Entrpi/ds4 fork **v0.6.2**) — native C/CUDA binary  
**Author:** [@bleysg](https://x.com/bleysg) (Bleys Goodson) / [@antirez](https://x.com/antirez) (Salvatore Sanfilippo)  
**Updated:** August 20, 2026  

> **Recipe contract:** [`recipes/deepseek-v4-flash-0731-ds4.yaml`](../recipes/deepseek-v4-flash-0731-ds4.yaml)
> **Switch script:** [`scripts/switch-to-ds4.sh`](../scripts/switch-to-ds4.sh)

---

## ⭐ Featured Recipe

The smartest model available, now serving on a single DGX Spark via [Bleysg's ds4 CUDA engine](https://github.com/Entrpi/ds4-on-spark) — a ground-up C/CUDA inference engine with DSpark lossless speculative decoding.

| Metric | Value |
|---|---|
| **Model** | DeepSeek-V4-Flash 0731 (284B params, 12B active MoE) |
| **Engine** | ds4 CUDA (Entrpi/ds4 fork **v0.6.2**) |
| **Quant** | IQ2XXS 2-bit with imatrix (~81 GB GGUF) |
| **Spec Decode** | DSpark k=2, ~60-75% acceptance |
| **Context** | Up to 1M per bank (975K tested, needle-in-haystack verified) |
| **Decode** | ~20 tok/s single-stream, ~65 tok/s peak (8 concurrent) |
| **Prefill** | ~1,127 tok/s |
| **TTFT** | ~200ms (2s warm on 975K context) |
| **Memory** | Precise accounting — demand-mapped context, idle reclaim, graceful refusal (no OOM) |
| **Stress Test** | 3M active tokens, 24h continuous load, dozens of parallel agents |

---

## What's new in v0.6.2 (upgraded August 19, 2026)

- **3M active tokens across many parallel agents** on one Spark — runs smoothly under stress
- **Precise memory accounting** — measures actual usage per request, demand-mapped context (nearly free until filled), reclaims idle state, refuses gracefully instead of OOM
- **Memory floor model** — set a memory floor; the engine manages the rest (run STT or other apps alongside without OOM risk)
- **Full 1M context in one bank** — 975K-token conversation ingested in ~25 min at 633 tok/s; needle found; next turn answers in 2s with everything warm
- **Stress-tested** — dozens of small convos, huge ones, deep ingestions mid-flight, 24h continuous load. Held perfectly.
- **Docs overhauled** — READMEs explain memory model + knobs; capacity claims include setup/measurements

### Previous versions

- **v0.5.5** (August 1): stream512 race fix (Xid 13 root cause), admission governance, budget double-booking
- **v0.5.4** (August 1): `--mem-floor-gb` admission gate, serial graph rightsize, trim-on-evict
- **v0.5.0** (July 29): initial CUDA engine release

---

## Why ds4 CUDA (not vLLM)

The 0731 weights are 167 GB in FP8 safetensors — too large for the Spark's 121 GB unified memory via vLLM (OOM during loading). The ds4 engine solves this with:

- **IQ2XXS GGUF quantization** — 81 GB on disk, fits comfortably
- **Native C/CUDA** — no Docker, no Python, no framework overhead
- **DSpark speculative decoding** — lossless, built in, 75% acceptance at k=2
- **Direct GPU access** — 1127 tok/s prefill, 200ms TTFT
- **Precise memory accounting (v0.6.2)** — demand-mapped context, idle reclaim, graceful refusal instead of OOM

---

## Three Flavor Configs

| Flavor | Context | Banks | Peak tok/s | Use Case |
|---|---|---|---|---|
| **Balanced ⭐ (default)** | 196K | 20 | 65 | Best balance, forum-proven sweet spot |
| **Speed** | 131K | 14 | ~40 | Max concurrency |
| **Deep** | 1M | 8 | ~25 | Deep context with disk KV overflow |

Switch flavors by changing `-c` value and env vars — no rebuild needed.

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

### 2. Upgrade an existing install

```bash
# From any version → v0.6.2
curl -sSL https://raw.githubusercontent.com/entrpi/ds4-on-spark/main/install.sh | bash -s -- --start

# Or build from source (on Spark)
cd ~/code/ds4 && git fetch --all --tags && git checkout v0.6.2 && make cuda-spark
```

### 3. Run with sparkrun

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

### 4. Or run directly (without sparkrun)

```bash
# Via switch script (recommended — includes pre-flight checklist)
ssh jaita@<spark-host> 'bash ~/switch-to-ds4.sh'

# Or via ds4-serve directly
DS4_BATCH_FIT_HEADROOM_MB=8192 \
DS4_SERVER_COALESCE_MAX=2 \
DS4_CONT_DSPARK=1 \
DS4_CONT_MTP_MODE=2 \
DS4_DSPARK_MODEL=~/gguf/DSpark-drafter-Q2K-Q8-0731.gguf \
ds4-serve -c 196608 --host 0.0.0.0 --port 8000
```

---

## Configuration

### CLI Flags

| Flag | Value | Description |
|------|-------|-------------|
| `-m` | `~/gguf/DeepSeek-V4-Flash-...0731.gguf` | Model path (auto-detected by ds4-serve) |
| `-c` | `196608` | Context window (196K balanced, 131K speed, 1048576 deep) |
| `--port` | `8000` | HTTP API port |
| `--host` | `0.0.0.0` | Bind address |
| `--no-dspark` | (flag) | Disable DSpark drafter, use MTP-2 only |
| `--no-spec` | (flag) | Disable all speculation, plain decode |

### Environment Variables (DS4_*)

| Env var | Default | Description |
|---------|---------|-------------|
| `DS4_BATCH_FIT_HEADROOM_MB` | 6144 (ctx≤16K) / 8192 (ctx>16K) | GPU memory headroom for batch planner |
| `DS4_SERVER_COALESCE_MAX` | 32 | Max concurrent lanes (we set 2) |
| `DS4_CONT_DSPARK` | 1 | Enable DSpark drafter |
| `DS4_CONT_MTP_MODE` | 2 | MTP speculative decode mode |
| `DS4_DSPARK_MODEL` | (path) | DSpark drafter GGUF path |
| `DS4_DSPARK_MAX_NLIVE` | 1 | Max concurrent DSpark streams (**DO NOT set to 2** — causes CUDA fragmentation crashes) |
| `DS4_SESSION_GRAPH_FIT` | 1 | Fit gate for session graph alloc |
| `DS4_SESSION_LAZY_GRAPH` | 1 (ON) | Lazy graph alloc (defer to first request) |
| `DS4_SERVER_SERIAL_RIGHTSIZE` | 1 (ON) | Rightsize serial graph to request |
| `DS4_BATCH_VMM_BUDGET_MB` | (auto) | Cap comp-cache pool |
| `--mem-floor-gb` | 4 | Admission gate against live free memory |
| `FORK_PARTIAL` | 0 | Prevents context corruption from client disconnects |

### Memory Budget (v0.6.2 — precise accounting)

| Component | Size |
|---|---|
| Model weights (IQ2XXS) | ~81 GB on disk |
| Resident after load | ~99-105 GB GPU (demand-mapped, reclaims idle state) |
| Context buffers | ~3.1 GB (131K × 2 lanes) |
| Comp-cache budget | ~12 GB (auto-capped, trim-on-evict) |
| Free RAM | ~11-22 GB (varies by active context) |
| Max lanes | Auto-scaled from headroom (20 at 196K) |

---

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /v1/models` | List served models |
| `POST /v1/chat/completions` | OpenAI-compatible chat (supports tools, reasoning_effort, stream) |
| `GET /metrics` | Prometheus-format metrics (`ds4_*` prefix) |

**reasoning_effort** supports: `max`, `medium`, `low` (controls thinking depth)

---

## Known Issues & Pitfalls

### Pitfall 1: DS4_DSPARK_MAX_NLIVE=2 Causes CUDA Crashes
Setting NLIVE=2 (DSpark on both lanes) causes CUDA memory fragmentation that accumulates and crashes with "illegal memory access" after ~1-2 hours. **Keep NLIVE=1 (default).** DSpark only accelerates at 1 concurrent request — at 2+ concurrent it falls back to plain batched decode.

### Pitfall 2: Build with `make cuda-spark` not `make CUDA_ARCH=sm_121`
The correct build target for DGX Spark is `make cuda-spark` which enables the Spark HBM weight cache and uses `sm_121a` architecture. `make CUDA_ARCH=sm_121` does NOT enable the HBM cache.

### Pitfall 3: Never Restart DS4 Without James's Permission
**CRITICAL RULE:** Never kill, restart, or modify the Spark inference server without James's EXPRESS PERMISSION. Propose first. Act never. Even if the server is crashing. Report the error, suggest the fix, WAIT for James to say yes.

### Pitfall 4: ComfyUI Auto-Restart Fights DS4 for GPU
ComfyUI had a systemd service (`comfyui.service`) with `Restart=always`. **Stopped and disabled** (Aug 3 2026). To run ComfyUI manually: `cd ~/ComfyUI && ~/ComfyUI/venv/bin/python3 main.py --listen 0.0.0.0 --port 8188 &`. Stop DS4 first to free GPU memory.

### Pitfall 5: SSH OOM Protection
Kernel OOM killer could kill sshd child processes (oom_score_adj=0). Primary defense is the engine's precise memory accounting (v0.6.2: graceful refusal instead of OOM). Belt-and-suspenders: `for p in $(pidof sshd); do echo -1000 > /proc/$p/oom_score_adj; done`

---

## DSpark Speculative Decoding

- DSpark drafter: `DSpark-drafter-Q2K-Q8-0731.gguf` (~6.5 GB)
- Acceptance: ~60-75% average at k=2
- Quench controller auto-disables speculation when acceptance can't pay verify cost
- 0731-generation bases have no MTP head — the DSpark drafter replaces it entirely

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

## File Locations (on Spark)

| Path | Contents |
|---|---|
| `~/code/ds4/ds4-server` | Native C/CUDA binary (v0.6.2) |
| `~/.local/bin/ds4-serve` | Bash launcher wrapper |
| `~/gguf/DeepSeek-V4-Flash-IQ2XXS-...0731.gguf` | Model weights (~81 GB) |
| `~/gguf/DSpark-drafter-Q2K-Q8-0731.gguf` | DSpark drafter (~6.5 GB) |
| `/tmp/ds4-262144.log` | Server log |

---

## Credits

- **@bleysg** (Bleys Goodson) — Entrpi/ds4 fork, ds4-on-spark installer, DSpark speculative decode
- **@antirez** (Salvatore Sanfilippo) — Original ds4 engine, GGUF quantization
- **@styles01** — Recipe, runtime plugin, sparkrun integration