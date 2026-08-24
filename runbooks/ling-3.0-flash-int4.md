# Recipe: Ling-3.0-flash INT4

**Status:** ✅ Production — serving locally + native Spark Arena submission `sub1787588904618`
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

## ⛔ CRITICAL — Do NOT add these "optimizations" (they break Ling on GB10)

- **`--load-format fastsafetensors`** — triggers the Ling cold-load race on GB10 aarch64 unified memory. Every wedge we hit showed `Loading fastsafetensors checkpoint shards:` frozen at 33-38%. The qwen-122b lesson (11× faster) does NOT transfer to the Ling fork. **Use plain safetensors (default).**
- **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** — unproven on Ling, adds risk. **Leave it out.**
- **`--max-num-batched-tokens` override** — unproven. Leave defaulted.

The PROVEN config (loaded clean 3×, completed the full native arena run) is plain safetensors, no extra env.

## ⛔ CRITICAL — KDA kernel num_stages=2 patch (fixes whole-box hang)

Ling's KDA attention kernels autotune `num_stages in [2,3,4]`. On GB10, stages 3/4 need **133-139KB** shared memory > GB10's **101KB**, so autotune intermittently fails and the engine wedges at generation time (**D-state, GPU 0%, WHOLE box hangs including SSH**, ~90s+ recovery). Intermittent — that's why it "runs fine" then hangs.

**Fix:** cap `num_stages` to `[2]` in `kda.py` (3 config loops). Same class as `patch_fla_shmem.py` for lightning_attn.

```bash
# inside the container:
sed -i 's/for num_stages in \[2, 3, 4\]/for num_stages in [2]/g' \
  /usr/local/lib/python3.12/dist-packages/vllm/third_party/flash_linear_attention/ops/kda.py
```

Baked into `docker/Dockerfile.ling-flash`. **Must be re-applied after any container rebuild** (it's in the writable layer, not the base image, until the image is rebuilt).

## Cold-JIT first-request penalty (NOT a hang)

The **first** request after a restart takes ~30s because the engine JIT-compiles the MTP/mamba kernels on the fly (`postprocess_mamba_fused_kernel`, `eagle_prepare_*`, `_causal_conv1d_update_kernel`, `fused_recurrent_gated_delta_rule_fwd_kernel`). The log warns "consider extending warmup to cover this shape/config." **Subsequent requests are fast (~270ms).** Not a hang — just cold-JIT.

## Model Location on Spark

```
~/models/hf/Ling-3.0-flash-int4/   (~72GB, 24 safetensors shards, flat dir)
```

## Start Command

```bash
ssh jaita@larryspark.local 'bash ~/switch-to-ling-container.sh'
```

This serves `inclusionAI/Ling-3.0-flash-int4` on port 8000 with k=1, 256K, 2 lanes, fp8 KV, mamba align, ling3 parsers. **Plain safetensors (proven).**

## Stop Command

```bash
ssh jaita@larryspark.local 'docker rm -f ling-flash'
```

## Arena Benchmark (native, submitted Aug 24)

Full 28-task sweep (7 depths × 4 concurrency × 3 runs) via `sparkrun arena benchmark run` — a TRUE native launch (sparkrun launched the container itself, no `--skip-run`).

**Submission:** `sub1787588904618` (re-uploaded with correct HF model id — see HARD RULE below)

| depth | c=1 | c=2 | c=5 | c=10 |
|---|---|---|---|---|
| 0 | 21.5 | 44.5 | 33.4 | 38.1 |
| 4096 | 21.4 | 32.1 | 25.5 | 25.5 |
| 8192 | 21.5 | 29.3 | 20.5 | 20.5 |
| 16384 | 21.2 | 21.2 | 14.5 | 13.7 |
| 32768 | 20.9 | 11.4 | 8.9 | 8.3 |
| 65535 | 20.3 | 6.9 | 4.3 | 4.1 |
| 100000 | 19.6 | 3.7 | 2.5 | 2.3 |

(tg t/s) — single-stream is remarkably flat (20-21 t/s at every depth), but deep-context + concurrency collapses (2.3 t/s at 100K/c=10) due to the documented Ling CUDA-graph fallback at long KV.

## ⛔ HARD RULE — Submission `model:` field MUST be the HF org id

**A Spark Arena submission is only valid if the recipe's `model:` field is the REAL HuggingFace org id** (e.g. `inclusionAI/Ling-3.0-flash-int4`). The arena validates the model against a real HF repo.

**NEVER submit with a local path** (e.g. `model: /home/jaita/models/hf/...`). A local-path model silently fails to post.

**This was violated TWICE** (Ling submissions `sub1787471399298` and `sub1787530216046`) — both wasted full runs. **CHECK the `model:` field in the recipe BEFORE every upload.** The `cluster_config.resolved_model_path` (for identity-mounting pre-placed weights) is SEPARATE from `model:` — keep `model:` as the HF id and put the local path only in `resolved_model_path`.

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
# + KDA num_stages=2 patch (see above)
ENTRYPOINT []    # native sparkrun launch compat
CMD ["sleep", "infinity"]
```

## Known Issues

- **Deep-context decode degrades** — 7.9 t/s @45K, 4.6 t/s @90K. CUDA graph capture only covers small shapes.
- **4-lane crashes at c=10** — KDA smem limit. Use 2 lanes.
- **Shard loading can freeze ~50% of the time** around shard 7-8/24. Retry or watchdog.
- **Stock vLLM silently produces garbage** — must use `ling_3_0` fork.
- **First request after restart is slow (~30s)** — cold-JIT of MTP/mamba kernels. Not a hang.
- **Container has no restart policy** — dies on reboot. Set `--restart always` or relaunch after reboot.

## Community References
- https://github.com/sojufx/Ling-3.0-Flash-DGX-Spark-Recipe
- https://github.com/sudoingX/dgx-spark-ling
