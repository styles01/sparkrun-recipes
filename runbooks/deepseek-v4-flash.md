# Recipe: DeepSeek-V4-Flash (0731)

**Status:** ✅ Tool calling enabled, cgroup contained, MTP k=2
**Served name:** `deepseek-v4-flash`
**Stack:** vLLM-Moet venv (NOT Docker) — `~/venvs/vllm-moet/`
**Tool parser:** `deepseek_v4`
**Reasoning parser:** `deepseek_v4`
**Updated:** July 31, 2026 — new weight update `DeepSeek-V4-Flash-0731`

> **Recipe contract:** [`recipes/deepseek-v4-flash-0731.yaml`](../recipes/deepseek-v4-flash-0731.yaml)

## Model Location on Spark

```
~/models/hf/DeepSeek-V4-Flash-0731/                 (~167GB, 48 safetensors, FP8)
~/models/hf/DeepSeek-V4-Flash-0731/moe_w2_planes/   (~75GB, prepacked planes, built on first load)
```

Old `DeepSeek-V4-Flash` (June 22 weights) deleted. Replaced by 0731 update.

## Current Serve Script

`~/vllm-moet-spark/spark/serve-ds4-flash-1node.sh`

### ARGS

```bash
ARGS=(
  --served-model-name deepseek-v4-flash --trust-remote-code
  --enable-auto-tool-choice --tool-call-parser deepseek_v4
  --reasoning-parser deepseek_v4
  --kv-cache-dtype fp8 --block-size 256 --max-model-len 8192
  --gpu-memory-utilization 0.78
  --max-num-batched-tokens 1024 --max-num-seqs 4
  --tokenizer-mode deepseek_v4 --no-scheduler-reserve-full-isl
  --speculative-config '{"method": "deepseek_mtp", "num_speculative_tokens": 2}'
  --port 8000
)
```

## Two Variants

### Variant A: Max Context (default)
```bash
ssh user@<spark-host> 'bash ~/switch-to-ds4.sh'
```
- Context: 256K | Concurrency: 1 | GMU: 0.78

### Variant B: 2 Concurrent (enables subagents)
```bash
ssh user@<spark-host> 'bash ~/switch-to-ds4.sh --max-model-len 131072 --max-num-seqs 2'
```
- Context: 128K per stream | Concurrency: 2 | GMU: 0.78
- Both streams get 128K context — within the 100K-256K success criteria
- Enables Hermes delegate_task (parent + child both hit the Spark)
- KV cache is NOT the constraint (MLA = 656 bytes/token, 2×128K = only 172MB)
- Constraint is activation memory for 2 concurrent forward passes (~2GB free RAM)

## Start Command

```bash
ssh user@<spark-host> '/usr/bin/bash ~/vllm-moet-spark/spark/serve-ds4-flash-1node.sh \
  --max-model-len 262144 --max-num-seqs 1 \
  > /tmp/ds4-256k.log 2>&1 &'
```

## Stop Command

```bash
ssh user@<spark-host> 'ps aux | grep "vllm serve" | grep -v grep | awk "{print \$2}" | xargs kill -9'
```

## Performance (July 11, 2026 benchmarks — pre-0731 weights)

| Parameter | Value |
|---|---|
| Context | 262,144 (256K) |
| Max concurrent | 1 |
| Speed (warm) | ~21 tok/s |
| Speed (200K prompt) | ~10K tok/s prefill |
| MTP speculative | k=2, ~60% acceptance |
| Model memory | ~98GB resident |
| Free RAM | ~2GB (tight) |
| Startup | ~5 min (JIT compilation on first boot) |

**0731 weights:** Benchmarks pending. Will update after first run.

## Trade-Off

DS4 is the smartest model but the most constrained:
- 1 concurrent request only (at 256K context)
- ~2GB free RAM — absolutely single-user
- 21 tok/s is 5x slower than Qwen 122B and 5x slower than Qwen 35B
- If you need 2 concurrent: drop to 128K context (`--max-model-len 131072 --max-num-seqs 2`)

## MTP k=2 — Why Not Higher?

**Decision:** k=2 is optimal. Documented July 11, 2026 after research.

- DS4 has `num_nextn_predict_layers=1` — the MTP head was trained to predict exactly 1 token ahead
- k=2 = one native prediction + one extrapolation. k=3+ compounds error on a head not trained for it
- vLLM explicitly warns: "num_speculative_tokens > 1 will run multiple times of forward on same MTP layer, which may result in lower acceptance rate"
- Sapid-Labs RUNBOOK: ~21 tok/s is "≈ the 273 GB/s bandwidth ceiling" — hardware is the bottleneck, not k
- Each speculative position reads ~6 routed experts from unified memory (bandwidth-bound at 273 GB/s)
- Acceptance drops geometrically with k; bandwidth cost scales linearly
- At 256K/1 concurrent: only 2GB free RAM — higher k risks OOM
- Sapid-Labs ships k=2 in all serve scripts, no higher k tested in the wild
- k=3 at 128K context is the only safe experiment, but expected gain is ≤5% or net slowdown

**Do not increase k without benchmarking first.** If experimenting:
1. Clear torch compile cache (ADR-006: changing spec config = stale cache = crash)
2. Reduce to 128K context for memory headroom
3. Benchmark with bench_decode.py before and after
4. Run mtp_correctness_battery.py for quality verification

## vLLM-Moet Specifics

- Uses venv `~/venvs/vllm-moet/` (vLLM 0.24.0 + 3 Spark patches)
- NOT Docker — runs directly via systemd-run user scope
- Memory cgroup: `MemoryMax=110G -p MemorySwapMax=0`
- FlashInfer 0.6.14 (must be exact version)
- DeepGEMM built from source
- PATH must include `/usr/local/cuda/bin` and venv bin (patched into script)