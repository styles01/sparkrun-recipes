# Qwen3.8-27B SGLang DFlash2 — experimental DGX Spark candidate

**Status:** qualification-only. This is not a production recipe, not a Loca/Hermes switch, and it must not replace an existing inference lane.

- **Recipe:** [`recipes/qwen3.8-27b-sglang-dflash2-experimental.yaml`](../recipes/qwen3.8-27b-sglang-dflash2-experimental.yaml)
- **Canonical script:** [`scripts/switch-to-qwen38-sglang-dflash2-experimental.sh`](../scripts/switch-to-qwen38-sglang-dflash2-experimental.sh)
- **Pinned upstream source:** [`pangoleen/qwen3.8-27b-dgx-spark-dflash2`](https://github.com/pangoleen/qwen3.8-27b-dgx-spark-dflash2/tree/bac473f81a9eff0d3c1ba7dbc56a0bfc311f36ce) at `bac473f81a9eff0d3c1ba7dbc56a0bfc311f36ce`.

## Evidence boundary

The source reports promising DFlash2 results, but its DGX Spark evidence is narrow for this candidate: single-stream outcomes are not a concurrency qualification; raw concurrency outputs were not adopted here; tool and vision behavior are not demonstrated by a boot check; and tactic-cache results depend on the exact image, silicon, shapes, cache key, and boot. Treat all performance claims as hypotheses to remeasure.

## Verified pins

These values are copied from the source tree at the pinned commit, rather than inferred from mutable tags.

| Component | Verified pin |
|---|---|
| Upstream recipe | `https://github.com/pangoleen/qwen3.8-27b-dgx-spark-dflash2.git@bac473f81a9eff0d3c1ba7dbc56a0bfc311f36ce` |
| Target | `RadixArk/Qwen3.8-27B-NVFP4@554ebba9b5f1b79dc11246341960360e6ef05ef4` |
| DFlash2 draft | `maurienne-ai/Qwen3.8-27B-DFlash2-NVFP4-RTNcal@bd7a934213c47a9e7ef69eef36bb3325f47fd1f1` |
| Base image | `lmsysorg/sglang:qwen38-27b@sha256:3c0abdf41ef22de9d7a859dc16ed71eae69452e36c91f071a25e60c85a6d1fc6` |
| DFlash2 SGLang source | `c14312a66420b75ca9a11bf1817c4db1fa26b097` (the source Dockerfile also carries the quantized-target selector PR overlay) |

No other dependency version is asserted as pinned. In particular, do not substitute a current `latest` image, branch-head model snapshot, or an unverified DFlash2 fork.

## Isolation contract

- All source checkout and checkpoint artifacts live under `~/models/hf` (override only with a child of that path).
- The container mounts the two selected snapshots read-only and sets `HF_HUB_OFFLINE=1`.
- `--start` refuses a container with the candidate name or a listener on its port. It never kills or restarts another workload.
- The launch runs in a dedicated user-systemd unit with `Delegate=yes`, Docker `--cgroup-parent`, a `flock` lane lock, and `--restart=no`.
- The baseline is loopback-only port `8003`, `BF16` KV, context `65,536`, and `--max-running-requests 1`. Do not widen it before passing the gates below.
- No command edits Loca, Hermes, user service units, model service configuration, or an existing recipe/script.

## Explicit actions

Run only one action at a time. There is intentionally no default action.

```bash
bash scripts/switch-to-qwen38-sglang-dflash2-experimental.sh --check
bash scripts/switch-to-qwen38-sglang-dflash2-experimental.sh --stage
bash scripts/switch-to-qwen38-sglang-dflash2-experimental.sh --download
bash scripts/switch-to-qwen38-sglang-dflash2-experimental.sh --start
```

`--stage` clones and detached-checks out the source commit under `~/models/hf/sources/`, verifies `HEAD`, then builds the source's pinned-image overlay locally. `--download` uses `hf download --revision` into deterministic snapshot directories under `~/models/hf/checkpoints/`. `--start` does not download, build, remove another container, or silently change a setting.

## Required qualification record

Record command line, source/image/model pins, host/driver details, boot ID, prompt/output token counts, timing method, and raw per-request rows. Mark a row failed rather than omitting it. Do not quote tokens/s from SSE event counts; use server-reported completion tokens and wall-clock timestamps.

### 1. Format and boot gates

1. Run `--check`, `--stage`, and `--download` successfully.
2. Retain the script's structured `config.json` NVFP4 gate and inspect the target checkpoint metadata/configuration. A repository/name match alone is **not** true NVFP4 validation. Record the quantization fields and at least one real safetensors shard; reject a BF16/FP8 substitution.
3. Start the 65K/one-stream/BF16-KV baseline. Record image ID, Docker inspect `RestartPolicy` and `RestartCount`, systemd cgroup/unit, startup logs, `/v1/models`, `/tokenize`, and `/metrics` availability.
4. Confirm the DFlash2 selector uses the intended quantized-target path in logs. A healthy HTTP endpoint is not proof of speculative execution.

### 2. Functional gates

All tests must use the launched served name `qwen3.8-27b-dflash2-experimental`.

- **Thinking on / real trace:** send `chat_template_kwargs: {"enable_thinking": true}` with a multi-step prompt. Preserve and archive the returned reasoning trace/`reasoning_content` and final answer; do not accept a synthetic or thinking-off trace.
- **Tool schema and loop:** submit a nontrivial OpenAI tool schema, validate the parsed `tool_calls` shape and arguments, execute a local deterministic tool, append the `tool` result, and require a coherent final answer. Repeat for at least two tool turns.
- **Image and video:** issue one image request and one video request using supported Qwen-VL content blocks. Archive request shape, media provenance/hash, HTTP response, parsed content, and failure logs. Text-only success does not clear either gate.

### 3. Context, memory, and performance gates

Use the same tokenization method, prompt fixture, max output, sampling, and machine state for cold/warm pairs. Run a new container/cleared relevant cache for each cold measurement; identify exactly what was retained for each warm measurement.

| Gate | Required baseline |
|---|---|
| Context ladder | Cold **and** warm measurements at 32K, 65K, and 130K prompt contexts; one stream first. Include success/failure, TTFT, decode rate, completion tokens, and acceptance metrics. |
| Memory high-water | At every context rung, record host unified-memory and GPU-process/container high-water during prefill and decode, plus free-memory floor. Abort on OOM, swap/thrash, watchdog/reset, or unexplained memory growth. |
| Concurrency | After single-stream clearance, test 1, 4, and 8 concurrent requests at a context that fits. Report per-request completion, TTFT, decode rate, p50/p95, aggregate rate, Jain fairness index, slowest/fastest ratio, errors, and achieved concurrency. Do not claim capacity from a single aggregate number. |
| Restart behavior | Intentionally stop the candidate, verify it remains stopped (`RestartPolicy=no`, restart count unchanged), then perform one explicit manual start. Also document behavior after an intentional process failure without enabling automatic restart. |

For Jain fairness, calculate \( (\sum x_i)^2 / (n \sum x_i^2) \) over per-stream decode rates and retain the raw `x_i` values. Report p95 from the raw request rows, not a dashboard rounded average.

## FlashInfer tactic-cache experiment

Do **not** copy the upstream tactic cache into this candidate and do not make its reuse a startup prerequisite. It is a separate A/B experiment only after the baseline gates pass:

1. Measure repeated cold boots with no imported tactic cache.
2. Mount an empty, candidate-specific cache below `~/models/hf` and measure warm/replayed boots under the exact same image, driver, model revisions, context, concurrency, and request shapes.
3. Record cache path, key/version, ownership, creation time, and hashes; compare distributions across multiple boots.
4. Reject reuse when the key changes, variance is unexplained, functional output regresses, or the warm improvement does not repeat.

## Promotion rule

Promotion requires every gate above to pass with raw evidence and an explicit reviewer decision. Failure, missing raw rows, an unproven vision/tool path, a cache-only win, or a restart anomaly leaves this recipe experimental. The conservative rollback is explicit: stop and remove only `qwen38-dflash2-experimental`; do not alter any other service.
