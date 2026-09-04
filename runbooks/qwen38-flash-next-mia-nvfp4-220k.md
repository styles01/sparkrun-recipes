# Runbook: Qwen3.8-Flash-Next Mia NVFP4 — native 220K evaluation variation

> **Experimental and inert by default.** This is an independently reviewed variation of MiaAI-Lab’s one-DGX-Spark recipe—not a production replacement, not an Arena recipe, and not running. It never changes Hermes/Loca configuration. Its launcher refuses co-residence; it will not kill a healthy workload to make room.

## Why this variation exists

Mia’s recipe demonstrates a potentially useful vLLM route for the multimodal Qwen3.8-Flash-Next model on one GB10: a smaller third-party NVFP4 checkpoint, a memory-mapped/offloaded PLE n-gram table, native MTP, and a guarded container launch.

The key takeaway is **not** “enable 1M context.” The pinned upstream code uses native **262,144** tokens by default, caps YaRN at 512K, and explicitly says 1M does not fit on a single Spark. We therefore evaluate the native window first.

Our current Qwen Flash Next GGUF service is already configured for **262,144 tokens**, not approximately 200K. This is an engine/quantization variation at the same native context length.

## Pinned identities

- **Upstream launcher source:** [`MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark`](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark), commit `554f295f0ac744cff8a5ffd4dd3bcc96aa82ab7f`, AGPL-3.0-or-later.
- **Model:** [`Mia-AiLab/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/Mia-AiLab/Qwen3.8-Flash-Next-NVFP4), revision `925d7be6c14c6c9442ef83e8f05b5a3c39304f69`.
- **Runtime image (arm64):** `vllm/vllm-openai@sha256:3b0e188ffceb3d07e09c3cb5215433a0020eacf02d7f882ed3a8bfd15454477e`.
- **Our wrapper:** `scripts/switch-to-qwen38-flash-next-mia-nvfp4-262k.sh`.
- **Upstream source staging:** `~/src/qwen38-flash-next-mia-262k`.
- **Model cache:** `~/models/hf` only. No model directory may be created outside `~/models`.

The wrapper does not copy or relicense the upstream AGPL launch implementation. It clones the exact commit and writes a local `.env` with our constrained baseline.

## Baseline configuration

| Setting | Baseline |
|---|---|
| Context | **220,000** native; YaRN off |
| Lanes | **2** |
| KV | `auto` / BF16—not FP8 |
| Native MTP | Off |
| Modalities | **Vision and video on** — model config is multimodal; no `language_model_only`, `--limit-mm-per-prompt`, or vision-disable flag is set |
| PLE | upstream memory-mapped CPU/offload path required |
| Port | 8888, avoiding the current service port |
| Memory safety | upstream cgroup plus watchdog; 5 GiB host slack |
| Docker GPU | `--privileged --gpus all`, after an actual CUDA allocation preflight |

This deliberately does **not** reproduce the social post’s fastest configuration. MTP and FP8 KV are separate variables; turning them both on before checking basic model quality and memory behavior makes a failure impossible to interpret.

## Commands

All commands are inert until their named action:

```bash
bash scripts/switch-to-qwen38-flash-next-mia-nvfp4-262k.sh --check
bash scripts/switch-to-qwen38-flash-next-mia-nvfp4-262k.sh --stage
bash scripts/switch-to-qwen38-flash-next-mia-nvfp4-262k.sh --download
bash scripts/switch-to-qwen38-flash-next-mia-nvfp4-262k.sh --start
```

- `--check`: verifies aarch64, GB10 identity, and any staged source/model identities. It does not alter state.
- `--stage`: clones/detaches the source commit and generates the constrained `.env`; it does not launch or download weights.
- `--download`: fetches only the pinned model revision into `~/models/hf` and places a revision marker in that exact snapshot.
- `--start`: checks for existing vLLM/llama/DS4/Ling/Qwen processes and containers, **refuses if any exist**, runs the real in-image CUDA allocation preflight, then delegates to the pinned upstream launch script.

## Required evidence gates

Do not promote this lane because it returns HTTP 200, or because an X post gives a throughput number.

1. **Identity gate** — source commit, image digest, model snapshot revision, and wrapper identity marker all match exactly.
2. **Real-chat gate** — `/health`, `/v1/models`, multi-turn reasoning chat, tool call, strict JSON, and reasoning/content separation work.
3. **Native-context retrieval** — independent 100K, 200K, and 220K passkey tests with early/middle/late placements; preserve raw prompts, replies, timings, and server logs.
4. **Memory and recovery** — track `MemAvailable`, cgroup peak/current, container logs, and GPU telemetry during long prefill; stop cleanly and prove memory unload/restart works.
5. **Multimodal** — image and short video tests with known answers. Do not infer long-video or concurrent multimodal capacity from a single demo.
6. **Quality comparison** — compare exact same prompts with the existing Q4 llama.cpp lane. Include prose, tool use, coding, long-document retrieval, and multimodal tasks.

## Later experiments—not defaults

### Native MTP=3

Mia reports benefits from MTP=3. Enable it only after the baseline passes every gate, repeating the retrieval, quality, speed, and memory tests. MTP acceptance is task-dependent and can be lower on multimodal requests.

### FP8 KV

FP8 KV is a capacity trade-off. It must be evaluated independently from MTP. The upstream repository reports FP8 results, but it also contains caution that quantized sparse-attention keys can change attention-block selection. Treat its quality equivalence as unproven until our long-context suite passes.

### YaRN 512K

This is a distinct long-context experiment. It is not required to test the native 220K lane. A 1M setting remains excluded: the pinned upstream launch code says it does not fit and its README says it was never run on that host.

## Source evidence and caveats

- [Mia upstream README](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark/blob/main/README.md)
- [Pinned upstream `start.sh`](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark/blob/554f295f0ac744cff8a5ffd4dd3bcc96aa82ab7f/start.sh)
- [Official Qwen Flash-Next model](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)

Reported single-stream decode (~37 tok/s) and four-stream aggregate (~86 tok/s) are author measurements, not independently reproduced here. The upstream repository does not ship the benchmark harness/traces supporting those numbers.
