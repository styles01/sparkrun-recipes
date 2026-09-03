# Runbook: Qwen3.8-27B llama.cpp native-MTP experimental candidate

> **Experimental, isolated lane.** This is not a production replacement and does not repoint, edit, or otherwise change Loca or Hermes. The only available external evidence is a 5090 native-MTP report with `parallel=1`; it is not GB10 proof.

## Source and artifact pins

- Model: [`unsloth/Qwen3.8-27B-GGUF`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) at `4ca720788d1e01f1bff70c033e0d0028fd02e502`.
- llama.cpp: no baked-in revision. PR [#28206](https://github.com/ggml-org/llama.cpp/pull/28206), the Qwen3.8 DFlash/MTP OOB-token fix, is closed but was not verified as merged into `master`; therefore the launcher fails closed until the operator supplies an audited 40-hex `LLAMA_CPP_REVISION` that contains the required fix.
- Candidate evidence: [`sudoingX/qwen38-mtp`](https://github.com/sudoingX/qwen38-mtp). Treat its 5090 result as an isolated lead only.

## Canonical inert launcher

```bash
LLAMA_CPP_REVISION=<verified-40-hex-sha> bash scripts/switch-to-qwen38-llamacpp-mtp-experimental.sh --check
LLAMA_CPP_REVISION=<verified-40-hex-sha> bash scripts/switch-to-qwen38-llamacpp-mtp-experimental.sh --stage
bash scripts/switch-to-qwen38-llamacpp-mtp-experimental.sh --download
```

No argument performs no workload or service action. Source is staged at `~/src/llama.cpp-qwen38-mtp-experimental`; model artifacts are downloaded directly and only under `~/models/hf/unsloth-Qwen3.8-27B-GGUF`.

`--start` is deliberately consequential: it stops known exclusive inference containers/processes, prints `MemAvailable` before and after that release, refuses to start below 100 GiB available memory, and starts only a `systemd --user` cgroup scope with `MemoryMax=110G` and `MemorySwapMax=0`. It does not touch Loca/Hermes configuration.

```bash
# MTP-off, one stream, 131K: canonical first live lane.
bash scripts/switch-to-qwen38-llamacpp-mtp-experimental.sh --start

# Only after baseline gates: native MTP n=2, still one stream and 131K.
MTP_DRAFT_N=2 bash scripts/switch-to-qwen38-llamacpp-mtp-experimental.sh --start
```

Do not begin with n=6/7. `MTP_DRAFT_N` accepts only `0`, `2`, `3`, or `4`; `0` is the default. The model and MTP GGUF file names are explicit defaults and can be overridden only with paths that remain inside `~/models/hf`.

## Required gates — all required before promotion

1. **Build/source gate.** Record the audited llama.cpp SHA, verify checkout equals it, retain the build log, and record CUDA/driver/GPU inventory. Do not substitute a branch name, tag name, or unverified short SHA.
2. **Real inference-health gate.** For each tested cell, require a real `/v1/chat/completions` completion with expected content after startup. `/v1/models` alone is never health evidence. Test restart and failed-request recovery.
3. **MTP/context matrix.** Run MTP off, then `n=2`, `n=3`, and `n=4` at **131K and 262K**. Do not advance to a higher depth or context when the preceding cell has errors, output corruption, unacceptable memory, or worse end-to-end behavior. Record TTFT, decode tok/s, p50/p95 ITL, completion correctness, and native MTP acceptance/log evidence when exposed.
4. **Tools and strict JSON.** Exercise OpenAI tools with valid/invalid arguments, multiple sequential tool turns, escaped strings/unicode, streamed and non-streamed responses, and strict JSON-schema outputs. Check actual parser output, not just text resembling JSON.
5. **Image/document gate.** Establish whether the selected llama.cpp/model artifact actually supports image/document inputs. If supported, test OCR, chart/document QA, multiple inputs, negative controls, and an image/document-followed tool call. If unsupported, record that limitation and do not claim multimodal support.
6. **Needle retrieval gate.** At both 131K and 262K, plant multiple exact needles at early/middle/late positions in real document-shaped text; require exact retrieval plus source/position discrimination. Include a no-needle negative control and a post-retrieval generation check.
7. **Memory high-water gate.** Record cgroup `memory.current`, `memory.peak` (when available), `memory.events`, process RSS, `MemAvailable`, and GPU memory at boot, after long prefill, during decode, after cancellation, and after each MTP variant. An HTTP 200 while memory silently spills or makes no decode progress fails this gate.
8. **Clean unload gate.** Stop the cgroup scope/process, verify no `llama-server` remains, then record recovered `MemAvailable`, GPU memory, and cgroup events. Do not run the next variant until release is measured.
9. **Concurrency gate.** Sweep **N=1, N=2, and N=4** at only already-correct settings. Concurrency is not assumed to improve throughput. Record per-request correctness, queue delay, TTFT, p95 ITL, aggregate decode, memory high-water, cancellation recovery, and fairness. A single-stream result does not establish N=2/4 capacity.

## Stop conditions

Stop the cell and preserve its logs on OOM/cgroup `oom_kill`, GPU reset/error, stalled real completion, corrupted tool/JSON output, failed retrieval, unacceptable high-water, or incomplete unload. The relevant historical risk is especially high for Qwen3.8 native MTP and long context; no GB10 conclusion follows from the 5090 report.

## Sources

- [llama.cpp PR #28206](https://github.com/ggml-org/llama.cpp/pull/28206)
- [llama.cpp source](https://github.com/ggml-org/llama.cpp)
- [unsloth Qwen3.8-27B GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF)
- [sudoingX Qwen3.8 MTP candidate evidence](https://github.com/sudoingX/qwen38-mtp)
