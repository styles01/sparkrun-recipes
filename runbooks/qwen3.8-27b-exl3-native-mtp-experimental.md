# Runbook: Qwen3.8-27B EXL3 3.5bpw native-MTP — experimental

> **Status: inert candidate; not running.** This is a single, exclusive, text-only evaluation lane. It changes neither Spark services nor Loca/Hermes configuration. It is not an Arena submission.

## Pinned artifacts and boundaries

- **Model:** [`Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw`](https://huggingface.co/Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw) at revision `19441ac874c4018295da848e250f23511361cda4` only.
- **Runtime:** [`MiaAI-Lab/exllamav3`](https://github.com/MiaAI-Lab/exllamav3) at `63b32f001d7b2cfed3b3e3aaf25f534ba53cc7ed` only. The pinned README identifies aarch64 GB10 SM120/SM121 support, EXL3 serving, and native MTP capability.
- **Local model directory:**
  ```text
  ~/models/hf/Mia-AiLab--Qwen3.8-27B-EXL3-3.5bpw/19441ac874c4018295da848e250f23511361cda4
  ```
- **Modality:** text only. This artifact has no vision tower in this recipe. Do not test, advertise, or infer image/vision support.
- **Capacity boundary:** the fork server serializes requests; the candidate is `batch=1` / one request. Queue tests measure fairness, not concurrent serving.

## Safety invariants

1. The script is inert without an explicit action. `--check`, `--stage`, and `--download` never start or stop a workload.
2. `--start` never kills anything. It prints every discovered H3, vLLM, DS4, or llama process/container/user-unit and refuses co-residence.
3. A permitted baseline is contained in a transient `systemd-run --user --collect` service with `MemoryMax=110G`, `MemorySwapMax=0`, and explicit `Restart=no`.
4. Initial live configuration is `131072` context, one request, batch 1, and `--draft_model none`: native MTP is off.
5. The pinned server exposes `-mtp` and `-ndt/--num_draft_tokens` internally, but its exposed `tools/serve_openai.py` CLI does **not** pass a requested `-ndt` through. Therefore this candidate refuses MTP n=2/3/4 rather than inventing a flag or patching the pinned runtime. It writes a launcher template that requires an explicit independently verified command.

## Canonical script

```bash
bash scripts/switch-to-qwen38-exl3-native-mtp-experimental.sh --check
bash scripts/switch-to-qwen38-exl3-native-mtp-experimental.sh --stage
bash scripts/switch-to-qwen38-exl3-native-mtp-experimental.sh --download
bash scripts/switch-to-qwen38-exl3-native-mtp-experimental.sh --start
```

`--stage` first checks `aarch64`, NVIDIA compute capability `12.1` (SM121), `nvcc`, Python, and `systemd-run --user`; only then it clones/detaches the exact Mia revision and builds it with that checkout's `requirements.txt` and `pip install .`. `--download` uses `huggingface_hub.snapshot_download` with the exact model revision and refuses a pre-existing destination without the matching revision marker. No weight is downloaded anywhere else.

`--start --mtp-n 2` is deliberately blocked. After all baseline gates, it creates `~/state/qwen38-exl3-native-mtp/verified-mtp-launch-template.sh` with the cgroup and exclusivity envelope but no invented server command. A reviewer must fill in a command verified against the pinned source and retain verification evidence before that template can be used.

## Required evidence gates

Record commands, source/model revisions, raw responses, server logs, timings, and memory samples for every gate. A successful HTTP status alone is not a pass.

### 1. Pinned source and model validation

- `git -C ~/src/exllamav3-mia-qwen38 rev-parse HEAD` equals `63b32f001d7b2cfed3b3e3aaf25f534ba53cc7ed`.
- Model destination has `.sparkrun-model-revision` exactly equal to `19441ac874c4018295da848e250f23511361cda4` and contains `config.json` plus the expected EXL3 files.
- Capture `huggingface_hub.model_info(..., revision=...)` identity before trusting the snapshot.

### 2. Clean SM121 build

- On a clean venv and clean `TORCH_EXTENSIONS_DIR`, record `uname -m`, `nvidia-smi --query-gpu=compute_cap`, `nvcc --version`, torch CUDA version, and compiler output.
- Require `aarch64` and SM121 (`12.1`) before build; preserve full build log.
- Import `exllamav3` and confirm the extension/JIT build completes without fallback errors.

### 3. Baseline real-chat health

Start only with native MTP off, then require `/health`, `/v1/models`, and a non-trivial multi-turn real chat to complete. Confirm response content, token accounting, and a clean server log—not just socket readiness.

### 4. Long prefill and retrieval

Run separate 100K, 200K, and 262K prompt tests. At each depth:

- Measure prefill duration/throughput, TTFT, decode rate, and peak/high-water memory.
- Use unique passkeys placed at early, middle, and late prompt positions and require exact retrieval.
- Preserve the generated text and raw request/response. A summary or HTTP 200 does not establish retrieval.

The initial 131072 launch does not itself authorize a 200K/262K claim. If higher-depth evaluation needs a changed cache size, record and independently verify that exact pinned-fork command first; do not modify this baseline launcher.

### 5. Native MTP ladder

- **Off:** establish repeated baseline quality, latency, retrieval, and memory results first.
- **n=2:** only after every prior baseline gate passes and an explicit MTP-n=2 command has been verified against the exact source. Fill and review the fail-closed template; then repeat health, 100K retrieval, tools, JSON, thinking, and unload checks.
- **n=3 and n=4:** separate experiments after n=2. Compare acceptance, correctness, output speed, p95 ITL, and memory against off/n=2. No setting is promoted merely for higher tok/s.

### 6. Behavioral protocol

At baseline and for each admitted MTP setting:

- **Tools:** run OpenAI function calls with typed integer, boolean, array, and object arguments; validate tool-call format and post-tool continuation.
- **Strict JSON:** use constrained/strict JSON requests and parse outputs with a real JSON parser. Record malformed-rate and refusal behavior.
- **Thinking:** verify reasoning/content separation for normal chat and that tool calls do not corrupt the thinking or final fields.
- **Output budget:** run reasoning-only prompts with a hard completion-token budget. Verify the server honors it and does not report reasoning as successful final content when no answer is requested.

### 7. Memory and unload

- Sample `memory.current`, `memory.peak`/`MemoryPeak` where available, `MemAvailable`, and GPU/unified-memory telemetry throughout every long run.
- Require conservative headroom, no continuing high-water growth across repetitions, and no OOM/cgroup event.
- Stop the launched scope, verify it is gone, and demonstrate clean unload before another candidate is considered. Do not kill unrelated workloads to obtain this evidence.

### 8. Queue fairness, not concurrency

Submit N=1, N=2, and N=4 independent requests at once. The expected behavior is serialization (`batch=1`), not concurrent execution. Record arrival/start/completion order, per-request wait, starvation, errors, and response integrity. Require bounded, explainable queue behavior; do not call this multi-lane capacity.

## Promotion threshold

This remains an experimental one-lane candidate until all gates above are independently repeated. Promotion requires exact pinned identities, clean SM121 build, real chat and retrieval at each tested depth, behavioral correctness, controlled memory/unload, fair serialized queues, and a separately verified MTP command for every nonzero n. Vision remains explicitly unvalidated and unsupported.
