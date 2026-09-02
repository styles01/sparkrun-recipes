# Runbook: DeepSeek-V4-Flash-Vision-Exp Q2 — experimental DS4 lane

> **Experimental only.** This is a separate upstream `antirez/ds4` Vision-Exp lane; it must not replace the production 0731 DS4 CUDA route. It is not wired into Loca/Hermes and has no SLA.

## Why a separate lane

The production text service is the Entrpi/Bleysg DS4 CUDA `v0.6.2` route with its own `ds4-serve` wrapper, 0731 language GGUF, 0731-specific DSpark drafter, and `DS4_CONT_*`/`DS4_BATCH_*` memory-management contract. Upstream Antirez Vision-Exp has a different model family, support-GGUF layout, server behavior, and recent untagged implementation.

Do not carry production wrapper flags into this process. Do not use the 0731 drafter with Vision-Exp.

## Pinned upstream

```text
https://github.com/antirez/ds4
110afdd8886586f18fc9b28bc5533152dd10e728
```

- Build: `make cuda-spark` (GB10 / SM121a CUDA target)
- Primary model: Vision-Exp Q2, about 81 GiB
- Sidecar: `DeepSeek-V4-Flash-Vision-Encoder.gguf`, about 0.9 GiB
- Optional speculative sidecar: `DeepSeek-V4-Flash-Vision-Exp-DSpark-support.gguf`, about 5.6 GiB
- Supported media: **inline PNG/JPEG image inputs only**. Video is unsupported.

## Canonical staged launcher

```text
scripts/switch-to-ds4-vision-exp-experimental.sh
```

It performs nothing without an explicit action.

### Prepare only; no server switch

```bash
bash scripts/switch-to-ds4-vision-exp-experimental.sh --stage
bash scripts/switch-to-ds4-vision-exp-experimental.sh --download
bash scripts/switch-to-ds4-vision-exp-experimental.sh --check
```

The script pins/builds code under:

```text
~/src/antirez-ds4-vision-exp
```

and downloads all weights only under:

```text
~/models/hf/DeepSeek-V4-Flash-Vision-Exp-Q2
```

It does not use `~/gguf/` or write model artifacts to the Mac.

### First live test only

When intentionally switching the exclusive Spark lane:

```bash
bash scripts/switch-to-ds4-vision-exp-experimental.sh --start
```

The initial lane is deliberately constrained:

```text
port: 8101
context: 4096
resident sessions: 1
DSpark: OFF
Hermes/Loca: untouched
MemoryMax: 110G, MemorySwapMax: 0
```

It stops exclusive inference/video workloads only during `--start`, requires at least 100 GiB `MemAvailable` after that switch, and serves via a cgrouped user scope.

## Do not promote based on a successful boot

A running `/v1/models` endpoint only proves the process loaded. It does not prove image semantics, long-context behavior, or agent safety.

### Required gates, in order

1. **Build/runtime gate**
   - pinned SHA check;
   - `make cuda-spark` clean build;
   - upstream CUDA regression and vision-preprocessing test;
   - record driver, CUDA/NVCC, command line, compiler output, model SHA-256s.

2. **Memory/failure containment gate**
   - measure resident memory at boot and after first image;
   - measure `MemAvailable`, cgroup events, and actual inference health after failed/cancelled inputs;
   - real-health probe must make a minimal completion, never just call `/v1/models`.

3. **Image semantics gate**
   Test fixed public examples for multi-object recognition, OCR/screenshot, chart/diagram, spatial relation, unrelated-image negative control, same-image follow-up, and multiple-image turns. Compare against a trustworthy reference, not merely plausible prose.

4. **API compatibility gate**
   Test exact OpenAI Chat `image_url`, OpenAI Responses `input_image`, Anthropic base64 image, text tools, image-bearing tool results, SSE cancellation, reasoning-stream fields, and screenshots from actual clients. Current upstream parsing is strict: only inline PNG/JPEG data URIs or Anthropic base64 forms should be assumed.

5. **Context/agent gate**
   Test text-only then image-conditioned tools/followups at 4K, 32K, 65K, 100K, and 196K only after the preceding tier is stable. Include compaction/rebuild/restart; upstream has an open report of image-session compaction failing near 85K.

6. **Concurrency gate**
   `--batched-session` is capacity/fairness behavior on single CUDA GPU, not demonstrated grouped-decode scaling. Sweep 1, 2, and 4 concurrent mixed image/text workloads; record per-request correctness, TTFT, p50/p95 ITL, queue delay, memory peak, cancellation recovery, and aggregate decode.

7. **DSpark gate — optional, last**
   Only use the matching Vision-Exp support GGUF. First compare ordinary target decoding, strict support, and `--dspark` one at a time. Promote only on measurable end-to-end benefit with no quality/tool regressions. Never use the 0731 DSpark sidecar.

## Known blockers

- No upstream release/tag; vision commits are recent and must be pinned.
- No published one-GB10 end-to-end image-quality benchmark, long-context test, or concurrency sweep.
- Vision-Exp is a separate checkpoint, not an `--vision` add-on for text Flash 0731.
- Image sessions deliberately do not use generic persistent disk-KV because text cache keys lack image fingerprints. Do not promise restart-resume multimodal cache behavior.
- Do not begin with Q2/Q4 hybrid or MXFP4. The former reduces headroom; MXFP4 does not resident-fit a single GB10, and SSD-streaming vision behavior is unproven.

## Sources

- [Antirez DS4 README](https://github.com/antirez/ds4/blob/main/README.md)
- [Vision implementation commit](https://github.com/antirez/ds4/commit/fc8bf3c39c51892c41f439a4ce17b90643dd1984)
- [DeepSeek V4 Flash Vision Exp](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp)
- [Antirez DeepSeek GGUF artifacts](https://huggingface.co/antirez/deepseek-v4-gguf)
- [Open image/function-role API issue](https://github.com/antirez/ds4/issues/933)
- [Open image-session compaction issue](https://github.com/antirez/ds4/issues/945)
- [Vision DSpark sidecar mismatch issue](https://github.com/antirez/ds4/issues/949)
