# Recipe: Ling-3.0-flash INT4

**Status:** ✅ Production — serving locally + native Spark Arena submission `sub1787530216046`
**Served name:** `inclusionAI/Ling-3.0-flash-int4`
**Stack:** Docker — `ghcr.io/styles01/ling-3.0-flash-int4:latest` (vLLM `ling_3_0` fork on base `vllm/vllm-openai:v0.25.1`)
**Model:** `inclusionAI/Ling-3.0-flash-int4` (72GB, compressed-tensors W4A16 / Marlin)
**Tool parser:** `ling3` (custom XML `<tool_call>`)
**Reasoning parser:** `ling3`
**Updated:** August 24, 2026

> **Recipe contract:** [`recipes/ling-3.0-flash-int4.yaml`](../recipes/ling-3.0-flash-int4.yaml)

## Model Specs

| Spec | Value |
|---|---|
| Total params | 124B |
| Active per token | 5.1B |
| Architecture | hybrid KDA + MLA (BailingMoeV3) |
| MTP | native, 1 layer (`num_nextn_predict_layers=1`), in-checkpoint |
| Context | 262,144 (256K native) |
| Quant | INT4 (compressed-tensors W4A16, Marlin) — NOT fp8 (doesn't fit single GB10) |

## Key Decisions

### MTP k=1 (NOT higher)
- `num_nextn_predict_layers=1` — the draft head was trained to predict exactly 1 token ahead
- Acceptance: 94.3% at pos 0, 57% at pos 1, 40% at pos 2 — k=1 is the sweet spot
- **k=3 CRASHES GB10:** Triton `OutOfResources: out of resource: shared memory, Required: 102400, Hardware limit: 101376` on `sm_121`. Never set `num_speculative_tokens > 1`.
- Repo §14: `num_speculative_tokens: 2` is explicitly UNTESTED upstream.

### GMU 0.80 — Do NOT exceed
- 0.80 is the safe ceiling on 128GB unified (77GB weights + KV stays under)
- **0.85 pushed total to 112GB — machine-freeze risk on unified memory. Hard no.**

### max-num-seqs 2 — NOT 4
- The `ling_3_0` fast recipe uses 4, but **4-lane config crashes at arena c=10** (KDA attention kernel needs 133KB+ shared memory > GB10's 101,376 at high batch)
- 2 lanes survives the full 28-task arena sweep including c=10

## Post-run optimizations (applied Aug 24)
- **`--load-format fastsafetensors`** — 11× faster cold start (removes the fragile 5-min window that caused repeated native-run failures)
- **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** — fragment-free KV allocation, headroom for deeper concurrency

## Model Location on Spark

```
~/models/hf/Ling-3.0-flash-int4/   (~72GB, 24 safetensors shards, flat dir)
```

## Start Command

```bash
ssh jaita@larryspark.local 'bash ~/switch-to-ling-container.sh'
```

This serves `inclusionAI/Ling-3.0-flash-int4` on port 8000 with k=1, 256K, 2 lanes, fp8 KV, fastsafetensors, mamba align, ling3 parsers.

## Stop Command

```bash
ssh jaita@larryspark.local 'docker rm -f ling-flash'
```

## Arena Benchmark (native, submitted Aug 24)

Full 28-task sweep (7 depths × 4 concurrency × 3 runs) via `sparkrun arena benchmark run` — a TRUE native launch (sparkrun launched the container itself, no `--skip-run`).

**Submission:** `sub_2026-08-24 ling` (auto-posted)

| depth | c=1 | c=2 | c=5 | c=10 |
|---|---|---|---|---|
| 0 | 21.5 | 44.5 | 33.4 | 38.1 |
| 4096 | 21.4 | 32.1 | 25.5 | 25.5 |
| 8192 | 21.5 | 29.3 | 20.5 | 20.5 |
| 16384 | 21.2 | 21.2 | 14.5 | 13.7 |
| 32768 | 20.9 | 11.4 | 8.9 | 8.3 |
| 65535 | 20.3 | 6.9 | 4.3 | 4.1 |
| 100000 | 19.6 | 3.7 | 2.5 | 2.3 |

(gg t/s)  — single-stream is remarkably flat (20-21 t/s at every depth), but deep-context + concurrency collapses (2.3 t/s at 100K/c=10) due to the documented Ling CUDA-graph fallback at long KV.

## Native Launch Learnings (CRITICAL — applies to ALL custom containers)

Getting `sparkrun arena benchmark run` to launch a custom container natively (no `--skip-run`) requires:

1. **Image ENTRYPOINT = `[]` (empty)** — so sparkrun's entrypoint-probe reports "absent" and passes. Any consuming/wrapping entrypoint (`vllm serve`, or `vllm-nonroot-entrypoint.sh` which prepends `vllm serve`) makes the probe return `verdict=fail` and aborts at `[3/6]` silently.
2. **Image CMD = `["sleep","infinity"]`** — container starts instantly, passes sparkrun's liveness check BEFORE the 72GB model loads (~5min), then sparkrun `docker exec`s the serve command in.
3. **Model in HF-cache format OR `cluster_config.resolved_model_path`** — sparkrun's distribution step looks for `<cache_dir>/hub/models--<org>--<name>/`. For a flat local dir, use `resolved_model_path` to identity-mount and skip the 72GB download.

**Troubleshooting:** sparkrun swallows launch errors (just `exit 1`). Enable DEBUG logging to a file to see the real error:
```python
import logging
logging.getLogger("sparkrun").setLevel(logging.DEBUG)
fh = logging.FileHandler("/tmp/sparkrun-debug.log"); fh.setLevel(logging.DEBUG)
logging.getLogger("sparkrun").addHandler(fh)
```

## Dockerfile (custom image)

The custom image overlays the `ling_3_0` fork onto the base. See [`docker/Dockerfile.ling-flash`](../docker/Dockerfile.ling-flash).

```dockerfile
FROM vllm/vllm-openai:v0.25.1
# overlay ling_3_0 fork python files
ENTRYPOINT []    # native sparkrun launch compat
CMD ["sleep", "infinity"]
```

## Known Issues

- **Deep-context decode degrades** — 7.9 t/s @45K, 4.6 t/s @90K. CUDA graph capture only covers small shapes.
- **4-lane crashes at c=10** — KDA smem limit. Use 2 lanes.
- **Shard loading can freeze ~50% of the time** around shard 7-8/24. Retry or watchdog.
- **Stock vLLM silently produces garbage** — must use `ling_3_0` fork.

## Community References
- https://github.com/sojufx/Ling-3.0-Flash-DGX-Spark-Recipe
- https://github.com/sudoingX/dgx-spark-ling
