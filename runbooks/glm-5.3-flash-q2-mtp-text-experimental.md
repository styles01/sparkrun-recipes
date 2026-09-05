# Runbook: GLM-5.3-Flash UD-Q2_K_XL + MTP — one-Spark text research variant

> **Status: staged candidate; not running.** This is a deliberately conservative local experiment, not a production lane, not an Arena recipe, and not a claim that GLM vision or multi-lane serving works on one Spark. It never changes Hermes/Loca configuration.

## Why this variant exists

An X post reported a single Spark running **GLM-5.3-Flash** with a 120 GB dynamic IQ3 GGUF and llama.cpp MTP. The author reports 20.8 tok/s and a higher private benchmark score than their own two-Spark EXL3 deployment. Those are author claims, not reproduced evidence.

The valuable signal is that GLM-5.3-Flash is a 320B-total / 18B-active MoE whose low-bit GGUF path may fit one GB10. The post's 120 GB IQ3 pack does **not** leave an operational safety margin. This variation uses the smaller Unsloth `UD-Q2_K_XL` pack instead.

## Exact pinned identities

- **Model:** [`unsloth/GLM-5.3-Flash-GGUF`](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF), revision `2975ab414d30340466d8c51533c6e91f0cca64c1`.
- **Quant:** `UD-Q2_K_XL`, reported 109 GB / 101.51 GiB of stored weights.
- **Runtime fork:** [`eauchs/llama.cpp`](https://github.com/eauchs/llama.cpp), commit `1d0c76f3c6d030fdfc269aa27db6334ea2834cec` (GLM5Next MTP path).
- **Evidence repository:** [`Weschera/glm53-flash-one-spark`](https://github.com/Weschera/glm53-flash-one-spark).
- **Official source model:** [`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash), MIT.

This is **GLM-5.3-Flash**, not the separate 744B-total GLM-5.3 flagship. The flagship has no credible one-GB10 route.

## Memory posture: carved-out context, not a headline window

A DGX Spark has roughly **121.7 GiB** of unified CPU/GPU memory. The Q2 pack leaves approximately **20.2 GiB** before runtime workspace, KV, OS, and page/cache pressure. We deliberately reserve that margin.

| Lane | Context | Requests | KV | Purpose |
|---|---:|---:|---|---|
| `32k-baseline` | 32,768 | 1 | q8 | First permitted local smoke, correctness, MTP acceptance, recovery |
| `128k-eval` | 131,072 | 1 | q8 | Gated retrieval and real-text decode evaluation |
| `256k-probe` | 262,144 | 1 | q8 | Gated single-request ceiling probe only |

**No concurrent lane is declared.** A second request, large batch, or 1M-context experiment is excluded until we have measured unified-memory low water, recovery, and tail stability ourselves. The one-Spark evidence warns that 1M q8 can OOM/wedge the host.

The post's 120 GB IQ3 option leaves only about 9.9 GiB before runtime allocations, so it is expressly excluded from the initial unattended route.

## Multimodality boundary

This is a **text-only** MTP recipe. The selected MTP fork calls itself text-only. A different open llama.cpp PR is required for GLM vision and documents its own correctness constraints (`NVIDIA_TF32_OVERRIDE=0`, `-fa off`). Therefore this recipe:

- has no `mmproj`;
- makes no image/video claim;
- keeps `-fa off` and TF32 override disabled pending deterministic correctness comparisons;
- cannot be represented as a substitute for Qwen Flash’s verified multimodal lanes.

## Canonical launcher

```text
scripts/switch-to-glm53-flash-q2-mtp-text-experimental.sh
```

The launcher is source-pinned and inert without an explicit action. It keeps model files only under:

```text
~/models/hf/GLM-5.3-Flash-GGUF-UD-Q2_K_XL
```

It builds the exact fork under `~/src/llama-glm53-flash-mtp`, launches only one request slot, and contains the server in a user systemd scope with `MemoryMax=118G` (decimal, approximately 109.9 GiB) and swap disabled. This preserves roughly 11.8 GiB of the 121.7-GiB pool outside the scope for the OS and SSH. A switch stops existing exclusive model/video workloads, so stage/download must be completed before a maintenance-window start.

### Prepare only — no workload replacement

```bash
bash scripts/switch-to-glm53-flash-q2-mtp-text-experimental.sh --check
bash scripts/switch-to-glm53-flash-q2-mtp-text-experimental.sh --stage
bash scripts/switch-to-glm53-flash-q2-mtp-text-experimental.sh --download
```

The download uses Python `huggingface_hub` rather than assuming `hf` is installed. It pins the model revision, requires all four Q2 GGUF shards beneath `UD-Q2_K_XL/`, and never writes model data on the Mac or outside `~/models/hf`.

### First permitted start: 32K baseline

```bash
bash scripts/switch-to-glm53-flash-q2-mtp-text-experimental.sh \
  --start --lane 32k-baseline
```

Fixed first-lane posture:

```text
context            32,768
parallel            1
KV                  q8_0
MTP                 draft-mtp
FlashAttention      off
NVIDIA_TF32_OVERRIDE=0
```

It exposes an unauthenticated OpenAI-compatible API service on trusted LAN port `8000` with model alias `GLM-5.3-Flash-Q2-MTP`. This name is intentionally variant-specific: this unvalidated Q2 text experiment must not masquerade as a production GLM route. Do not expose this endpoint outside the trusted network without an authentication proxy.

## Required gates

### Gate 1 — 32K correctness and recovery

Before increasing context:

1. Confirm `/health` and `/v1/models` return and record the exact launch command.
2. Compare deterministic non-speculative and MTP outputs on fixed prompts; record output agreement and MTP acceptance.
3. Run a 32K early/middle/late passkey set, a real tool-like structured-output task, and repeated prefill/decode cycles.
4. Record `MemAvailable`, cgroup/scope peak, TTFT, completion tok/s, p95 inter-token latency, and raw outputs.
5. Stop the scope, verify unified memory releases, then restart a known healthy lane from its canonical launcher.

### Gate 2 — 128K evaluation

Only after Gate 1 passes: test early/middle/late retrieval plus captured-text decode at 128K. A completed HTTP request is not a correctness result.

### Gate 3 — 256K single-request probe

Only after Gate 2 passes: test 256K q8 retrieval/decode under active memory telemetry. No parallel or second-request claim is allowed from this run.

## Known risks

- Both GLM MTP and vision llama.cpp implementations are open PR/fork paths; revisions must remain pinned.
- The author claims MTP gains of 1.4–1.9×, but acceptance is workload-dependent and not reproduced here.
- The 109 GB Q2 pack creates safety headroom at the cost of quality. It must be compared against GLM EXL3 and existing Spark lanes—not assumed to be good enough because it fits.
- `-fa on` was used in the social recipe, but a related vision PR says correct output requires `-fa off`; retain the conservative setting until deterministic evidence says otherwise.
- Do not test 1M context on an unattended one-Spark system.

## Sources

- [X claim](https://x.com/WescheNex1q/status/2095977892987707727)
- [One-Spark evidence repository](https://github.com/Weschera/glm53-flash-one-spark)
- [Unsloth GLM-5.3 Flash GGUF](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF)
- [Official GLM-5.3 Flash model card](https://huggingface.co/zai-org/GLM-5.3-Flash)
- [MTP llama.cpp PR #27752](https://github.com/ggml-org/llama.cpp/pull/27752)
- [Vision llama.cpp PR #27754](https://github.com/ggml-org/llama.cpp/pull/27754)
