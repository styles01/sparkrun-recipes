# Runbook: Qwen3.8 Flash-Next — TensorFold single-Spark recipe (MiaAI Lab) — adoptable ALTERNATIVE

> **ALTERNATIVE — NOT DEPLOYED. Documentation only.** Nothing on this box was
> changed to produce this file. The running daily driver is and remains the
> **vLLM NVFP4 + MTP k=3** lane — container `vllm-fn-tp1`
> ([runbook](qwen38-flash-next-mtp3-draftvocab47k.md)), port 8000.
>
> **Why not switched yet:**
> 1. TF wants the machine to itself at first load (**~115 GiB free**) — so vLLM
>    **and** the decision-model sidecar (~4.3 GiB) must both be down for every
>    first start and every full restart; sidecar coexistence at steady state is
>    untested.
> 2. The TF numbers below are author-benchmarked; we have not reproduced them
>    (no third-party re-run exists that we know of).
>
> **Status:** documented, validated on paper against the repo, awaiting a
> green-light + measured coexistence test. Nothing is cloned or running.
> Sources: post https://x.com/MiaAI_lab/status/2104835240157945891 (MiaAI Lab,
> 2026-09-29 07:26 UTC) and the live repo (see *Source references*).

## Model
- **Checkpoint (served):** [`Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP`](https://huggingface.co/Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP) — **MLX 4-bit, group size 32, with the MTP draft head**. ~106 GiB of files (~114 GB by the scripts' disk check; README counts a download footprint of ~125 GB including temp).
- **⚠ This is a DIFFERENT quant than our daily driver.** Ours is `Mia-AiLab/Qwen3.8-Flash-Next-NVFP4` (hardware-native NVFP4 on Blackwell, @ `925d7be6`). TF serves a Mac-ecosystem MLX 4-bit (g32) checkpoint through MLX-format tensors on the CUDA backend — **no quality comparison between the two quantizations exists anywhere we have looked**. Outputs will differ from today's lane (different quant + different engine). Flag before any production use: A/B an eval set (retrieval + code + prose quality) across the two quantizations, not just speed.

## Engine / container
- **What TensorFold is:** [TensorFold](https://github.com/ashhart/TensorFold) (Ash Hart, MIT) — an OpenAI-compatible serving runtime for MLX **and** CUDA (Apple Silicon + NVIDIA GPUs) in which each model family supplies its own kernels and draft verification. Flash-Next is a supported family on both backends; upstream's model table also already lists **GLM-5.3-Flash** and **DeepSeek-V4-Flash** rows — the "GLM 5.3 Flash + DeepSeek v4.1 Flash TF recipes coming soon" tease from the announcement post has real engine headroom behind it.
- **Recipe repo (this runbook documents):** https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold — MIT. Weights are under the Qwen Community License 1.0. The prebuilt image is NVIDIA's PyTorch container redistributed as a value-added runtime; **pulling it implies accepting NVIDIA's license terms** (the container prints them at start).
- **Pin:** TensorFold **v0.3.6.3 exactly** (the patches are made for this release; do not bump `TF_VERSION` without re-verifying byte-identity), inside `nvcr.io/nvidia/pytorch:26.07-py3`, image `tensorfold-qwen38:v0.3.6.3` (prebuilt ~11 GB from `ghcr.io/miaai-lab/qwen3.8-flash-next-single-dgx-spark-tensorfold:v0.3.6.3-<patches-hash>`, or built locally).
- **The 8 speed patches** (baked into the image by `scripts/prepare.sh`, applied with `patch -p0`; rebuild image only when patches change):

| Patch | Change | Claimed effect |
|---|---|---|
| `0001-cuda-live-token-counters` | `/health` reports live token totals | monitoring (upstream #79) |
| `0002-flash-next-ssd-read-ahead` | next prompt-chunk's n-gram rows read from SSD while the GPU processes the previous chunk | multi-chunk prefill +50% |
| `0003-flash-next-ssd-native-reader` | those SSD reads run on a C++ thread pool outside the Python GIL | short-prompt TTFT −35%; 3k–12k prefill +10–40% on top; decode +4% |
| `0004-flash-next-qsa-tiled-select` | sparse-attention block selection no longer spills registers past 128k tokens | 149k prompts 25% faster; decode at 149k context +19% |
| `0005-cuda-stream-draft-stats` | drafted/accepted counts in concurrent-request stats | observability |
| `0006-flash-next-prefill-rows` | configurable prompt-chunk size (port of upstream #40) | +2–5% at 4,096 rows |
| `0007-flash-next-copy-drafts` | drafts copied from earlier text when the reply repeats the prompt | +6% on quoting/editing replies |
| `0008-flash-next-vision` | image + video input for Flash Next on CUDA — Qwen3.5 vision tower (27 layers, 0.84 GiB, from the same checkpoint), interleaved 3-D rotary positions, video frames in timestamped blocks | `--vision`; previously only TF's dense-27B had it |

- **Byte-identical claim (author's own):** every speed patch changes speed only — drafts are verified against the model's own keyed samples, and the prefill changes read the same bytes and select the same attention blocks. Author checked this by comparing reply hashes (sampled **and** greedy, prompts up to 149k tokens) against unpatched TensorFold, with vision on and off, plus a ~195k needle-in-a-haystack test. **Scope note:** that is byte-identical *to unpatched TensorFold*, NOT byte-identical *to our vLLM NVFP4 lane* — no such claim or comparison exists.
- **`"draft": false` serial escape hatch:** any request can be sent with `"draft": false` to get TensorFold's serial, one-token-at-a-time reference path — the engine's own correctness reference is always one cheap request away (use it to spot-check serving vs drafting drift).

## Served config (defaults, all in `scripts/config.sh`; override via env, e.g. `PARALLEL=4 ./start.sh restart`, or as extra `tensorfold serve` flags — flags after the defaults win; the last value of a flag counts)

| Setting | Default | Notes |
|---|---|---|
| `PARALLEL` | `5` | concurrent streams decoded together |
| `CONTEXT` | `262144` | full native window per stream |
| `KV_DTYPE` | `int8` | fp16 scale per 32 values; `bf16` / `int4` also fit other stream counts (int4/bf16 change outputs slightly) |
| KV pool | **1,310,720 tok** | 5 × 262,144; ~23.4 GiB (4,799 MiB/stream: KV + sparse-attn index + stream buffers); 25% more than 4 streams |
| `MTP_DRAFTS` / `MTP_CONFIDENCE` | `6` / `0.60` | ≤6 drafts a round; a chain stops before a draft under 60%. Author-swept 2026-09-29, output identical in every arm; 6/0.60 beat the stock 6/0.30 (~3% prose, ~4% code); 4/0.50, 3/0.30, 7/0.75 matched prose but not code |
| `PLE_ON_SSD` | `1` | the ~29.8 GiB n-gram/PLE tables read from SSD instead of RAM (that RAM is free for KV; patches 0002/0003 pipeline the reads) |
| `VISION` | `1` | image + video input (`image_url` / `video_url` parts; ~4 images/10 MB each and ~2 videos/64 MB each / 96 MB total, ≤16,384 video tokens/request). Vision scratch ~0.8 GiB is taken only while an image/video encodes. `VISION=0` = text-only with 4,096-row chunks (2–5% faster prefill from 3k tokens) |
| `TENSORFOLD_MTP_COPY` | `1` | prompt-lookup drafts for replies repeating the prompt (patch 0007; needs `PARALLEL ≥ 2`) |
| `PORT` / `HOST` | `8888` / `0.0.0.0` | OpenAI-compatible API |
| `SERVED_NAME` | `Qwen3.8-Flash-Next` | the model id clients see — **capitalized; differs from our lowercase `qwen3.8-flash-next`**. On adoption either set `SERVED_NAME=qwen3.8-flash-next ./start.sh restart` or update clients |
| Thinking / sampling | on; temp 1.0, top_p 0.95, top_k 20 | Qwen recommended thinking-mode values; per-request `temperature/top_p/top_k/seed` win; `chat_template_kwargs: {"enable_thinking": false}` or `reasoning_effort` per request. Note: no min_p / presence / repetition penalty support (served as 0.0 defaults) |

Streaming, typed tool-call parameters (arrays come back as JSON arrays), `reasoning_content` vs `content` reasoning blocks, and the vision parts are all supported through the API.

## Performance — THEIR claim vs OUR numbers (provenance per row)

**Their numbers** (author-benchmarked, through the OpenAI API, int8 KV, PLE on SSD, MTP drafting; the 5-stream row is the shipped default; other rows measured with 4 streams; the prefill table used 4,096-row chunks = `VISION=0` semantics, i.e. VISION=1 costs them 4–5% prefill at 12k–150k): aggregate decode @1/2/4/5 streams = **62.4 / 90.5 / 106.7 / 119.3 tok/s**, TTFT 152–528 ms; prefill @8k/16k/32k/64k/128k = **2,503 / 2,520 / 2,499 / 2,414 / 2,200 tok/s** (TTFT 3.29–59.60 s).

**Our numbers for the current vLLM NVFP4 lane**, by source (three different provenances — do not mix them up):

| Metric | TF (their claim) | vLLM NVFP4 (our value) | Provenance of the OUR value |
|---|---:|---:|---|
| Decode, prose, 1 stream | 62.4 tok/s | **48.7 tok/s** | Mia's TF-vs-vLLM comparison card for our lane — NOT reproduced by us |
| Decode, prose, 1 stream (live idle-ish) | — | **~27 tok/s** | our own live observation on this Spark, uncontrolled |
| Decode aggregate, 8 streams | — (they publish @5: 119.3) | **163 tok/s** | Mia's comparison card — not reproduced by us |
| Decode aggregate, conc 1–10 bench v27 | — | **33–57 tok/s** | our own measured sparkrun arena run (`@styles01/spark-arena-v2-p400`, exact_tg) |
| Prefill @ 32k | 2,499 tok/s | **1,769 tok/s** | Mia's comparison card — not reproduced by us |
| Prefill short-ctx / 4k–8k | — | **~1.9k / 500–700 tok/s** | our own measured (bench v27) |
| TTFT | 152 ms (1 stream, short) | ~not on the card | — |

**Reading this honestly:** the 62.4-vs-48.7 and 2,499-vs-1,769 deltas are *between different measurement setups on different days* — promising, not proven. The only same-day cross-engine comparison is Mia's own card (author-benchmarked both sides). No third-party TF number exists yet; the independent Jason McNab repro (11/12 cells ±3.5%) that validated the *vLLM NVFP4* lane has no TF counterpart. The A/B protocol below is therefore the gate for adoption, not the tables above.

## Resource math — why this is not deployed

**First load (every first start and full restart):** the default config budgets the machine's free memory as `MemAvailable − 12.2 GiB host reserve ≈ 103.26 GiB` and allocates a ~102.5 GiB startup estimate (weights 75.2 GiB, vision tower 0.84, stream caches 22.5, fixed buffers 4.0 — with PLE tables left on SSD). On an empty Spark that means **~115 GiB free when the server starts** (README requirement). Today, with vLLM holding ~83–85 GiB, `MemAvailable` is 7.9 GiB — so:

- `docker stop vllm-fn-tp1` alone leaves (roughly) 7.9 + 85 ≈ ~90–93 GiB → **not enough**; vLLM must be stopped **and** the sidecar pair (laya decider on :8712 + helper ≈ 4.3 GiB reserved) must also be down → projected ~115–117 GiB, borderline vs the ~115 check. If `free -g` shows `MemAvailable < ~115 GiB` at adopt time, clear page cache pressure first (`echo 1 > /proc/sys/vm/drop_caches` — safe, no file deletion) or wait for buffers to settle; **do not keep the sidecar hot for the first load**.
- **Co-residency with vLLM is arithmetically impossible**: TF ~102.5 GiB + vLLM cgroup-capped 111 GiB ≈ 213.5 GiB needed vs ~124.6 GiB total. No shared-memory trick changes this — it is strictly sequential switching, never parallel running.

**Steady state (after load, with sidecars coexisting — ESTIMATED, measure before trusting):**

| Component | GiB |
|---|---:|
| Weights + context caches + buffers (TF's startup estimate, vision on) | ~102.5 |
| Host overhead not in TF's estimate (their own worst-observed peak: free-memory floor **≥8.3 GiB** left during a 195k vision prompt + 5 concurrent long requests, i.e. ~112–114 GiB actually held) | ~9–12 |
| Decision-sidecar pair (laya decider + helper) | ~4.3 |
| SparkDash + sparkmon + dockerd + OS (observed small, ~1–2) | ~1.5 |
| **Projected peak total** | **~117–120** vs 124.6 |

**The reserve left for the host itself with the sidecar hot and a 195k-prompt peak load in flight: ~4–7 GiB. Tight.** Their ≥8.3 GiB floor was measured on an EMPTY box, without our sidecar. **Recommendation: before any switch, measure the true steady state** (`free -g` / `/proc/meminfo MemAvailable` sampled over a 195k-prompt run with TF live + sidecar hot). If the floor drops below ~2 GiB at peak, this is a no, or a `VISION=0`/`TENSORFOLD_VISION_WORKSPACE_MIB`/`PARALLEL=4` fallback config is needed (PARALLEL=4 estimate: 97.7 GiB, buys ~5 GiB).

**Disk:** recipe wants ~160 GB fresh (~114–125 GB checkpoint in `~/.cache/huggingface` + ~24–35 GB image under Docker root; `MIN_FREE_GB=125` / `IMAGE_FREE_GB=35` hard checks). We have **~781 GB free** (root nvme0n1p2: 2,868,903 of 3,845,093 used, 79%) — fine, even stacked with future recipes.

**Switch cost:** `stop.sh` frees the machine in seconds (kills + `docker rm`s the TF container by design; checkpoint, image, kernel cache persist). Loading weights once set up is **~2.5 min** warm. Going TF→vLLM or vLLM→TF means re-reading ~106 GB from NVMe — raw disk speed ~2,500 MB/s but lazy/cold page behavior means real-world **minutes**; count on a short outage, never serve through a switch.

## Install (one-time, when we green-light it) — exact commands on the Spark

```bash
ssh jaita@192.168.2.185

# 0) GATE — confirm the adoption decision first; everything below stops serving lanes.
# 1) stop the vLLM daily driver (frees ~83-85 GiB; container + config stay intact, no rebuild needed):
docker stop vllm-fn-tp1
# 2) stop the decision-model sidecar — it runs as a plain process (NOT a docker container),
#    check the actual pid/process first:
ps -ef | grep -E 'server_decider|laya'   # laya decider runs from /home/jaita/venvs/laya, port 8712
pkill -TERM -f 'deciserv/server_decider.py'   # ⚠ confirm how it is auto-restarted before using kill; restoration path below
# 3) hard memory gate, do not proceed on a fail:
free -g   # need MemAvailable ~>= 113-115 GiB; if short: echo 1 | sudo tee /proc/sys/vm/drop_caches, then re-check
# 4) clone (repo verified reachable; ~1-2 MB of shell + patch files, trivial disk):
git clone https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold.git \
  ~/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold
cd ~/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold
# 5) one command. First run: pulls (or builds) the image (tensorfold-qwen38:v0.3.6.3 + 8 patches),
#    downloads the ~106 GiB checkpoint into ~/.cache/huggingface (resumable), verifies with tensorfold info,
#    compiles GB10 CUDA kernels (first time only, minutes, cached in ~/.cache/tensorfold-qwen38),
#    starts the server on 8888, walks the log, runs its own smoke test:
./start.sh          # warm start ~2.5 min weight load; the FIRST ever start is the full-prep long tail (image ~11 GB pull + ~114 GB ckpt + compile)
# 6) smoke test (the API model id here is CAPITALIZED — mind client configs):
curl -s http://localhost:8888/v1/models
curl -s http://localhost:8888/v1/chat/completions -H 'Content-Type: application/json' -d \
  '{"model":"Qwen3.8-Flash-Next","messages":[{"role":"user","content":"Reply with exactly: ok"}],"max_tokens":50}'
# 7) repo's own checks (needles and vision included):
tools/bench.py [label]   # prefill ~0.85k/3.2k/12.6k/50k + short decode check
tools/needle.py          # ~195k-token NIAH passphrase check
tools/toolcheck.py       # tool call with an array param
tools/visioncheck.py     # drawn image -> names a red circle and a blue square
# 8) restart the sidecar — its restart path is NOT scripted today; restore whatever
#    launched it before (check ~/deci-serv for its normal start procedure), then measure coexistence.
```

Useful at any time: `./stop.sh` (stops + removes the TF container; frees the machine), `./start.sh restart` (only after setup and arg checks pass — a typo leaves the running server alone), `docker logs -f qwen38-flash-next-tf`, `curl http://localhost:8888/health` (busy flag + live token totals), `FOREGROUND=1 ./start.sh` for systemd use. `PARALLEL`, `CONTEXT`, `KV_DTYPE`, `VISION`, `SERVED_NAME`, `PORT` all re-configure without touching files.

## A/B benchmark protocol (adopt-gate — mirror of our lane's protocol)

Use the same discipline as the vLLM lane's bench runs (sparkrun arena, `@styles01/spark-arena-v2-p400`, exact_tg counter-snapshot pattern, 28-cell depth×conc grid as in bench v27 — see the daily-driver runbook's *Measurement-method A/B* section):

1. **Idle frame**: confirm the box is genuinely idle before each engine pass (`free -g`, no other lane running, no subagents decoding) and record MemAvailable + disk before/after every bench session, both engines, to catch leaks.
2. **Prose decode**: same prompt mix, same `tg=400` + exact_tg convention, same conc ladder (1/2/4/5/8), both engines, same day. For TF, `tools/bench.py` gives the prefill ladder; the *decode* rows must come from the identical harness as vLLM's, never from different scripts.
3. **32k prefill**: a fresh ~32,806-token random prompt through each engine's own `/v1/chat/completions`, TTFT + tok/s measured the same way; run TF also with `VISION=0` + `TENSORFOLD_PREFILL_ROWS=4096` once, since that is exactly the config the author's prefill table used (our default `VISION=1` costs 4–5% there).
4. Order-reversed second pass if time allows (quant/engine warm-up asymmetries).
5. **Coexistence measurement** (the actual adoption gate, see Risks): MemAvailable floor sampled continuously while TF serves a 195k-prompt peak load with the sidecar live.
6. Keep raw data dated under `~/sparkrun-recipes/benchmarks/` like prior benches.

**A/B rule from this runbook's purpose: the decision to switch is OURS on OUR data** — Mia's card is a strong prior, not evidence.

## Rollback (the safety contract — memorize this one)

```bash
# On the Spark — go back to the daily driver at any moment:
~/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold/stop.sh     # frees the whole TF footprint (it only touches ITS OWN container; checkpoint + image stay on disk)
free -g                                                      # verify MemAvailable ≥ ~111 GiB (vLLM's cgroup cap) — same check discipline as the daily-driver runbook; nvidia-smi alone is NOT enough
docker start vllm-fn-tp1                                     # the canonical container, untouched by anything TF did — config frozen inside
until curl -sf http://localhost:8000/v1/models >/dev/null; do sleep 10; done   # ~4-7 min shard load (NVMe, lazy strategy)
curl -s http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d \
  '{"model":"qwen3.8-flash-next","messages":[{"role":"user","content":"Reply with exactly: ok"}],"max_tokens":16}'
# then restore the sidecar (see Install step 8)
```

Rollback reach: the TF side only ever touches — the `qwen38-flash-next-tf` container (created/destroyed per serve cycle), the cloned repo dir, the docker image `tensorfold-qwen38:v0.3.6.3`, the checkpoint in `~/.cache/huggingface`, and `~/.cache/tensorfold-qwen38`. **The vLLM container/image/recipe/artifacts are untouched by design; rollback is `docker start`, nothing else.**

## Risks & gotchas
- **UMA freeze-not-OOM (the top risk):** running out on a Spark's unified memory generally *freezes the whole machine* instead of failing an allocation cleanly (README states this; TF mitigates with a 12.2 GiB host reserve and a pre-start free-memory check that refuses to launch). Our vLLM lane additionally runs under a **111 GiB cgroup cap** so an OOM kills the container, not the box — **the TF lane as shipped has NO cgroup cap**; consider adding an equivalent cap experimentally (~112–115 GiB, must not starve the 102.5 GiB estimate) or keep the sidecar cold during peak TF loads and watch peaks from the 32k/128k/195k prefill ladder before trusting a full 5-stream depth.
- **Author-benchmark bias:** every published TF number is the recipe author's own table; no third-party re-run exists yet (unlike the NVFP4/vLLM lane, which survived an independent 11/12-cell ±3.5% repro). Their 62.4 / 119.3 / 2,499 headline numbers are claims until our A/B says otherwise.
- **Sidecar coexistence untested (the blocker):** ~4.3 GiB sidecar reserved on top of TF's ~102.5–114 GiB steady footprint leaves only single-digit GiB at peak. Measure the floor before switching (see Resource math / protocol step 5); a `VISION=0` or `PARALLEL=4` fallback (~97.7 GiB) exists if needed.
- **Multilingual MTP draft vocab NOT in TF:** the recipe serves the checkpoint's own MTP head; Mia's same-day follow-ups note multilingual MTP draft vocabs (@jvr0x) to be **adopted upstream later** — track that; it may change decode numbers and (by their own method note) outputs once landed. Separately, our vLLM lane's 47,149-token code-oriented draft vocab is a different speculative mechanism entirely — not comparable row-for-row.
- **Quant-quality delta unknown (flagged):** MLX 4-bit g32 vs our NVFP4 — no cross-quant eval we know of; validate retrieval + code + prose quality before trusting the switch for production traffic.
- **Byte-identity is NOT cross-lane:** TF's byte-identical guarantee is vs unpatched TensorFold only. Anything requiring bit-exact continuity with current outputs must re-baseline through TF (or stay on vLLM).
- **`draft:false` does not replace benchmarking:** the serial escape hatch validates the served vs drafting path cheaply; it is not a substitute for the A/B protocol.
- **Disk growth:** each new recipe family costs 100+ GB on disk; teased upcoming TF recipes (GLM 5.3 Flash, DeepSeek v4.1 Flash) will want similar space — fine today (~781 GB free), plan for it.
- **Pins:** patches are made for **TensorFold v0.3.6.3 exactly**; do not bump `TF_VERSION` without re-verifying the byte-identical claim and the patch applicability (`TENSORFOLD_NO_UPDATE_CHECK=1` is default — no version-check call at start). After changing `TF_VERSION`/`TF_REPO`/`BASE_IMAGE`/patches: `scripts/prepare.sh --rebuild`.
- **Port/model-id mismatch:** TF = 8888 + `Qwen3.8-Flash-Next` (caps); current lane = 8000 + `qwen3.8-flash-next`. `SERVED_NAME` / `PORT` handle both if clients must match — check before flipping production traffic.

## sparkDash & sparkmon
- **sparkDash** (this box's container, up for the last ~2 days): TF support **landed upstream** (per Mia's same-day follow-up posts) — **verify the version running on this box** actually includes it before relying on TF telemetry in the dashboard.
- **sparkmon telemetry parity for TF: NOT DONE.** `~/sparkrun-recipes/scripts/sparkmon.py` (running today) parses the vLLM lane's Prometheus `/metrics`. **TF's `/metrics` shape is unknown to us** — TF's nearest equivalent is `GET /health` with live token/`requests_running` counters (patch 0001, upstream #79), which is not obviously a Prometheus body. Check TF's actual `/metrics` shape on the live container before building any monitoring/alerting on it. If we switch, sparkmon going quiet is *expected*, not a fault — do not kill it for that; it is idle-harmless.

## Verifying it works (TF lane)
```bash
curl -s http://localhost:8888/v1/models   # → Qwen3.8-Flash-Next (caps)
curl -s http://localhost:8888/health      # → requests_running + live token counters (patch 0001)
tools/needle.py                           # 195k-token retrieval passes
tools/toolcheck.py && tools/visioncheck.py
```

## Source references
- **Announcement post:** https://x.com/MiaAI_lab/status/2104835240157945891 (MiaAI Lab, 2026-09-29 07:26 UTC) — TF recipe for Qwen3.8 Flash-Next on one Spark; benchmark tables; TF-vs-vLLM comparison; vision add-on; multilingual MTP draft vocab work (@jvr0x) to be adopted; sparkDash TF support; **teased upcoming:** GLM 5.3 Flash + DeepSeek v4.1 Flash TF recipes.
- **Recipe repo:** https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold (MIT; checked `refs/heads/main` live: HEAD `856bb6be4b58ce6a6727e6d071fb1c52f3f80e6e`; README/config/prepare/stop/start + `patches/` 0001–0008 + `tools/` reviewed from raw.githubusercontent on 2026-09-29 — **verified cloneable without cloning**; this runbook kept self-contained by embedding the install commands above instead of keeping a local clone).
- **Upstream engine:** https://github.com/ashhart/TensorFold (MIT; MLX + CUDA backends; `--vision` docs; GLM-5.3-Flash and DeepSeek-V4-Flash listed in its model table).
- **Served checkpoint:** https://huggingface.co/Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP (MLX 4-bit group-32 + MTP head).
- **Base image:** `nvcr.io/nvidia/pytorch:26.07-py3` (NVIDIA license terms apply on pull).
- **Rollback target + bench-protocol provenance:** [qwen38-flash-next-mtp3-draftvocab47k.md](qwen38-flash-next-mtp3-draftvocab47k.md) (current daily driver; bench v27 grid and exact_tg convention).
- **House-style template:** [qwen38-flash-next-vllm-nvfp4.md](qwen38-flash-next-vllm-nvfp4.md) (alternative-recipe banner).