# Runbook: DwarfStar (ds4) Qwen3.8-Flash-Next on DGX Spark

Companion recipe: `recipes/qwen38-flash-next-ds4.yaml` (this repo).
Full engineering notes: engine review + plan docs on the controller (oracle profile workspace).

## Quick switch (kill vLLM first, verify GPU free)

```bash
# 0. Stop whatever holds the GPU (lane trade discipline):
pkill -f ds4-server; pkill -f "vllm serve"; sleep 2
python3 -c "import ctypes; print('cuInit:', ctypes.CDLL('libcuda.so').cuInit(0))"  # must print 0

# 1. Build (first time only, ~5 min):
cd ~/src/cudafast-qwen38-125b-a6b-engine
export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH"
./setup.sh                    # builds engine + fetches/verifies ~112 GiB weights

# 2. Serve (SparkDash-visible, Loca-compatible):
DS4_SERVER_MODEL_ID=qwen3.8-flash-next \
DS4_MTP_DRAFT_VOCAB=~/src/arena-submission/draft_vocab_output_q8_0_47149.bin \
nohup setsid .build/ds4/src/ds4-server \
  -m reference_weights/Qwen3.8-Flash-Next-GGUF/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  --mtp-model reference_weights/Qwen3.8-Flash-Next-GGUF/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf \
  --mtp-draft 2 --host 0.0.0.0 --port 8000 \
  > /tmp/ds4-server.log 2>&1 < /dev/null &

# 3. Verify:
curl -sS http://127.0.0.1:8000/v1/models   # id=qwen3.8-flash-next, owned_by=ds4.c
curl -sS http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"qwen3.8-flash-next","messages":[{"role":"user","content":"ping"}],"max_tokens":10,"temperature":1}'
# SparkDash poll log should show: backend=ds4 model=qwen3.8-flash-next avail=true
```

## Boot sanity markers (in /tmp/ds4-server.log)

- `qwen4exp MTP head loaded: ... (block 48, 32 tensors, 2.58 GiB, draft=N)` — N = --mtp-draft value
- `draft vocabulary loaded: ... (47149 of 248320 ids ...)` — reduced vocab active
- `draft vocabulary resident: 122.48 MiB on device`
- `listening on http://0.0.0.0:8000`
- Memory plan line: `93.46 GiB required` — ACCEPTS, boots in ~90s

## Known behavior

| Thing | Behavior |
|---|---|
| Model id on /v1/models | `qwen3.8-flash-next` (via DS4_SERVER_MODEL_ID env; no server flag) |
| Empty `model` in requests | Serves the loaded engine (vLLM parity; Loca sends `model:''`) |
| Sampling (temp>0) | Depth-1: greedy fallback, warn-once (patched). Depth 2/3: sampled commits |
| Depth 2/3 + draft vocab | dv wins −0.38..−0.79 ms/tok vs stock; overall slower than depth-1 |
| Correctness | Exact: rejection sampling vs FULL target vocab; gates all pass |
| Escape hatch | `DS4_NO_DRAFT_VOCAB=1` → stock behavior |

## Gotchas (each cost real time once)

1. **`nohup setsid`** — plain `nohup ... &` in an ssh session dies when the ssh drops.
2. **Bind 0.0.0.0** — 127.0.0.1 is loopback-only; Loca/Hermes come over the LAN.
3. **No `--served-model-name` flag** — model id is hardcoded in ds4_server.c functions;
   the DS4_SERVER_MODEL_ID override is the supported knob (patched).
4. **Cargo PATH** — build needs `PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH`.
5. **`--mtp-draft N` semantics** — N = depth+1 (draft tokens); depth = N−1; max depth 3.
6. **Sequential only** — never concurrent with the vLLM lane (GB10 unified memory).
7. **Trivial-prompt degradation** — Q4 GGUF answers "I cannot comply" style on tiny
   prompts; template artifact, watch on real workloads.

## Measured (this box, 2026-09-12)

| config | decode tok/s | mean accept | correctness |
|---|---|---|---|
| stock depth-1 | 33.82 | 1.88 | PASS |
| dv depth-1 | 33.68 | 1.83 | PASS |
| stock depth-2 | 27.76 | 2.21 | PASS |
| dv depth-2 | 28.38 | 2.21 | PASS |
| stock depth-3 | 28.69 | 2.29 | PASS |
| dv depth-3 | 28.99 | 2.29 | PASS |

Prefill ~1,190 tok/s on Q4_K_XL (single 8K prompt).

## References

- Engine repo: github.com/Layr-Labs/cudafast-qwen38-125b-a6b-engine (MIT)
- Upstream: github.com/antirez/ds4 (no Qwen support)
- Rust fork: github.com/Baekpica/ds4-dfm-rs; GGUF: HF Baekpica/Qwen3.8-Flash-Next-Mixed-Quant-SSD-PLE-GGUF
- Forum: NVIDIA dev forums "Qwen3.8-Flash-Next reaches 1,023.9 tok/s prefill on a single DGX Spark / GB10"
- Community: dwarfstar.sh, yukon.org/mlxfast
- Full write-ups: workspace/ds4-qwen38-runbook.md, workspace/layr-engine-review.md, workspace/ds4-draft-vocab-plan.md (controller)