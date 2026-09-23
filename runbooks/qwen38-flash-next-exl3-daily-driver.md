# Qwen3.8-Flash-Next EXL3 Native — Daily Driver Runbook

**Status: CURRENT PRODUCTION DAILY DRIVER** (agent traffic on this box) · **Date:** 2026-09-23
· Supersedes the vLLM MTP3 + Draft Vocab 47K lane as the default lane. That lane remains
available as an alternate ([runbook](qwen38-flash-next-mtp3-draftvocab47k.md)).

**One-paragraph pitch:** the same Qwen3.8-Flash-Next (125B-A6B) served by the
[vcruz305 exllamav3 fork](https://github.com/vcruz305/exllamav3) with native MTP
speculative decoding, fronted by a single-file OpenAI shim with a vLLM-parity
telemetry contract. One process, no container, ~50 s load, 262K context, 54+ tok/s
single-stream decode. Every adaptation we made to get here is documented below so
you don't rediscover it the hard way.

---

## Why EXL3 became the daily driver

| Dimension | EXL3 native (this lane) | vLLM MTP3 NVFP4 (previous) |
|---|---|---|
| Load time | ~50 s (native engine, no container) | minutes (container + vLLM init) |
| Decode, single stream | **54.3 tok/s** (measured 2026-09-23) | ~21–55 tok/s (config-dependent) |
| Draft acceptance | **0.70 cumulative** (0.57–0.78 live window) | 0.73–0.85 |
| Context | 262K, full | 262K |
| Tool calling | ✅ canonical Qwen XML, replay verified | ✅ |
| Telemetry | full vLLM-parity `/health` contract (below) | native vLLM metrics |
| Footprint | one Python process in a venv, CPU-pinned | docker container, rootful |

Trade-offs, stated plainly: single-stream (one request at a time — Hermes/Loca agent
traffic is sequential, so this is fine); p95 latency cards read as window means (no
distribution exists engine-side); health counters reset on every restart.

## The pieces (exact provenance)

- **Engine:** `vcruz305/exllamav3` fork @ `bf9e10f`, venv `~/venvs/exl3-150`
  (exllamav3 1.5.0, torch 2.13.0, CUDA 13.0, `TORCH_CUDA_ARCH_LIST=12.1`).
- **Model:** `~/models/hf/turboderp/Qwen3.8-Flash-Next-exl3` (EXL3 quant with MTP head).
- **Shim:** `scripts/exl3_serve_openai.py` in this repo — hand-rolled OpenAI-compatible
  server, ~50 s reload, v3 `/health` contract with per-position MTP telemetry.
- **Launcher:** `scripts/switch-to-qwen38-exl3.sh` (canonical; `--check/--start/--status/--stop`).

## Launch (what the launcher does)

```bash
bash scripts/switch-to-qwen38-exl3.sh --check   # preflight
bash scripts/switch-to-qwen38-exl3.sh --start   # exclusive switch + serve
```

Live launch shape (what `--start` reproduces):

```bash
export EXL3_INT8_GEMV=0 EXL3_MOE_COOP_WIDE=1 EXL3_GR_INT8=1
export EXL3_MTP_HEAD_N=65536 EXL3_NGRAM_STREAM=0
export TORCH_CUDA_ARCH_LIST=12.1
cd ~/work/exllamav3-fork/examples
taskset -c 5-9,15-19 ~/venvs/exl3-150/bin/python \
  ~/sparkrun-recipes/scripts/exl3_serve_openai.py \
  -m ~/models/hf/turboderp/Qwen3.8-Flash-Next-exl3 \
  -mtp -ndt 5 -dds -dc 0.6 -cq 8,8 -cs 262144 \
  --host 0.0.0.0 --port 8000
```

Flag meanings that matter:

- `-mtp -ndt 5` — MTP draft head drafting up to 5 tokens deep.
- `-dds -dc 0.6` — dynamic draft scaling with confidence threshold 0.6 (shortens the
  draft chain when acceptance drops; this is why the per-position bars taper).
- `-cq 8,8` — quantized draft-head compute path.
- `-cs 262144` — 262K context ceiling.
- `taskset 5-9,15-19` — 10 cores for the engine; the rest stays responsive for
  SSH, SparkDash and the host.
- `--host 0.0.0.0` — REQUIRED. `127.0.0.1` breaks non-local clients (Loca on the Mac).

Safety net: the launcher runs the engine inside a `systemd-run --user --scope` with
`MemoryMax=110G` so a misbehaving engine OOM-kills **itself, never SSH**. Launches
refuse below 100 GiB MemAvailable. Health-check wait: up to 10 min, typically ~50 s.

## Adaptations that made this work (do not skip any)

1. **The fork, not upstream exllamav3.** Upstream exllamav3 does not carry the GB10
   (SM121) fixes and MTP head support this lane needs. Pin
   `vcruz305/exllamav3@bf9e10f`. The venv is named `exl3-150` (exllamav3 1.5.0).
2. **The OpenAI shim is load-bearing, not a detail.** It renders the model's own
   canonical chat template — tool calls replay as `<tool_call>/<function=…>` XML with
   empty think blocks, tool results wrap in `<tool_response>`. Earlier naive history
   rendering made the model "forget" its own tool calls and re-issue identical calls
   forever. If you swap the shim, replay a multi-turn tool loop before trusting it.
3. **`record_draft_stats=True`** is enabled on the `Generator` so per-verification-round
   draft stats exist; the shim folds them into `mtp_accept_by_position` (real
   per-position acceptance bars in SparkDash: pos0 ≈ 0.70–0.78, decaying by depth).
4. **Env tuning is load-bearing:** `EXL3_MOE_COOP_WIDE=1` + `EXL3_GR_INT8=1` +
   `EXL3_MTP_HEAD_N=65536` + `EXL3_NGRAM_STREAM=0` + `EXL3_INT8_GEMV=0`. Do not
   "clean these up" — they were measured, not guessed.
5. **Single-stream contract.** One job at a time (`busy` = refcounted inflight).
   Concurrency beyond 1 queues, it does not batch. This is the accepted posture for
   agent traffic; don't "fix" it into multi-lane without re-validating memory.
6. **`--parallel`-style multi-request is NOT used here** (that constraint is a
   llama.cpp/qwen4exp quirk, not this lane — but the shim is single-job by design).

## Telemetry (SparkDash)

The shim exposes a vLLM-parity `/health` (v3 contract): decode tok/s via per-token
counter, live `prefill_rate` during prefill (elapsed-weighted, converging to true
rate), per-position MTP acceptance, `is_prefilling` badge state, TTFT/E2E/ITL,
KV + prefix-cache occupancy. Full control-by-control audit (37 controls) lives in
[the telemetry runbook](qwen38-flash-next-exl3-native.md).

Operational notes:

- **Counters reset on restart.** Dashboard averages/acceptance bars rebuild from live
  traffic; a fresh restart shows zeros until the first requests land. Expected.
- **Health fields that exist:** `is_prefilling`, `prefill_rate`, `prefill_tps_live`,
  `mtp_accept_by_position[{position, tested, accepted, rate}]`, `kv_cache_usage`,
  `prefix_cache_hit_rate`, `gpu_memory_utilization`, `requests_waiting`, full
  counter set. See the shim source for the authoritative list.
- `prompt_tokens_total` bumps at enqueue; `completion_tokens_total` climbs per-token
  during decode (no EOS spike); `prefill_tokens_processed_total` climbs during prefill.

## Ops rules (hard-won)

- **Restart = ~50 s outage + counter reset.** Announce it to James BEFORE restarting
  the shim; never deploy shim changes silently. In-flight requests at restart time
  are dropped.
- **Never run two big models at once on GB10** — concurrent GPU memory allocation is
  what OOMs the box, not disk I/O. Wait for a lane to be fully up or fully down.
- **Deploy = edit the shim in the repo, scp to Spark, `pkill -f
  'exl3_serve_openai\.py'`, relaunch via the launcher.** The repo copy is the source
  of truth; never run ad-hoc launch commands and let the script drift.
- **`/tmp/exl3_serve.sh` on the Spark is ephemeral.** The canonical launcher in this
  repo (`scripts/switch-to-qwen38-exl3.sh`) replaces it.
- **Tool-calling regression gate:** after any shim change, run a two-turn tool loop
  (call → execute → result → final answer) at `temperature=0` and confirm a final
  answer, not a re-call.

## Verified numbers (2026-09-23, live)

| Metric | Value |
|---|---|
| Decode tok/s (single stream, prose) | 54.3 |
| Draft acceptance (single gen) | 0.573 · cumulative window 0.70 |
| Per-position acceptance | p0 0.78 · p1 0.65 · p2 0.64 · p3 0.68 · p4 0.70 |
| Context | 262,144 |
| Load → healthy | ~50 s |
| Requests served / failed (cumulative) | 53 / 0 |
| GPU mem util (idle-ish) | ~0.98 |