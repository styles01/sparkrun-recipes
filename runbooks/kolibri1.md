# Runbook: Aleph Alpha Kolibri-1 on DGX Spark (GB10)

> **Status: DRAFT-DO-NOT-DEPLOY.** Written 2026-10-04 from the Aleph Alpha
> launch post. Nothing on this box was changed, downloaded, or started to
> produce this file. The running daily driver is and remains the **EXL3 native
> MTP lane** ([runbook](qwen38-flash-next-exl3-daily-driver.md)), port 8000.
>
> **Why still draft:** (1) weights ~78.9 GB — NOT local, download blocked by
> the <20GB auto-download rule; (2) plugin image not yet built or smoke-tested
> on this box; (3) zero GB10/community runs of this exact stack exist yet;
> (4) the daily driver is 3 days old and passes the eval gate — no reason to
> swap until Kolibri wins the arena.
>
> House style: this doc documents, it does not deploy. Sources verified
> 2026-10-04 against HF API endpoints, GitHub, PyPI and Docker Hub manifests.

## Recipe

**Recipe:** [`recipes/kolibri1-vllm-plugin.yaml`](../recipes/kolibri1-vllm-plugin.yaml)
**Launcher (draft):** [`scripts/switch-to-kolibri1.sh`](../scripts/switch-to-kolibri1.sh)

## Model facts (verified 2026-10-04)

- **Model:** [Aleph-Alpha/Kolibri-1](https://huggingface.co/Aleph-Alpha/Kolibri-1) @ `e52eb4627d11516b0c01de49210ab5a4e4061444` (lastModified 2026-10-03T08:43Z)
- **Uploader:** Aleph Alpha (org). Announced: [x.com post](https://x.com/Aleph__Alpha/status/2106306843052052616) 2026-10-03 08:54 UTC, linking the HF repo. Launch blog: [kolibri-has-landed](https://aleph-alpha.com/en/blog/kolibri-has-landed-a-sovereign-open-weight-model/). Tech report: [PDF](https://aleph-alpha.com/downloads/tech-report.pdf).
- **Params:** 78.1B total (`78,103,074,560`), **3.46B active/token** (`3,457,573,120`)
- **MoE:** 384 experts/layer, **6 routed + 1 shared** per token, `norm_topk_prob: false`, moe_intermediate 512
- **Attention:** 50 layers, hidden 2560, 48 Q heads, **4 KV heads** (GQA), head_dim 128, **4:1 SWA:GQA hybrid** — `sliding_window: 513`, every 5th layer full attention (config `layer_types`), rope_theta 10000
- **Precision:** FP8 (float8_e4m3fn) weights in 128×128 blocks, **dynamic** activation quant; bf16 for embeddings, LM head, norms, MoE router gates (50 `mlp.gate` modules excluded); `head_dtype: float32` (nonstandard, plugin handles it)
- **Context:** native 262,144 (long-context phase trained at 262k); validated to 1,048,576 (vendor recommends ≤262k for serving/complex tasks). Beyond-262k serving needs `--max-model-len 1048576 --hf-overrides '{"max_position_embeddings": 1048576}'`
- **Languages:** German + English; German-morphology-optimized tokenizer; vocab 128,000
- **Reasoning mode:** yes, `reasoning_effort` none/low/medium/high via chat template kwargs. **Tool calling:** Hermes-style, `--tool-call-parser kolibri1`
- **License:** Apache-2.0 (weights, plugin, code)
- **Knowledge cutoff:** 2026-06-18 (EN+DE). **Release:** 2026-10-03
- **Size on disk:** safetensors total 78,103,074,560 params = **72.74 GiB readable weights**; repo `usedStorage` 78,852,606,661 B = **78.9 GB** (includes card assets); FP8 repo is 32 shards
- **BF16 checkpoint:** [Kolibri-1-BF16](https://huggingface.co/Aleph-Alpha/Kolibri-1-BF16) — 78.1B BF16 = 145.5 GiB readable, usedStorage 156.2 GB. **Does not fit** even solo (0.92·121 ≈ 111 < 145.5+KV). Not a candidate.
- **Third-party quants:** [audreyt/Kolibri-1-NVFP4-W4A16](https://huggingface.co/audreyt/Kolibri-1-NVFP4-W4A16) 47.4 GB storage / **~37.5 GiB readable — over the <20GB rule**; GGUF (Eliasfpv28 Q3_K_S, Hob-forge, Prompt48) and MLX (Apple-only) conversions exist — all above 20 GB. **No EXL3 quant exists** (turboderp has none; HF search 2026-10-04).

### LANE POLICY (James, 2026-10-04): Kolibri NEVER cohabits with the media stack.

Media stack = Music 3 (sgl-omni :8010) + ComfyUI image lane (:8189) + anything
serving them. Kolibri-1 at ~78 GiB replaces the ENTIRE box for the test
window; the ~78-20=~40 GiB cohabitation idea is dead. Teardown (switch
script) covers ALL of it: docker lanes, EXL3/ds4/llama-server natives,
sgl-omni/minimax_music3, decider family, ComfyUI ports. If an LLM is ever
needed alongside the media stack it will be a tiny one - not Kolibri.

## Claimed benchmarks (vendor post-training eval table, MoE peers, temp 0.6/top_p 0.6)

| Eval | Kolibri-1 | Same-table peer note |
|---|---:|---|
| Overall (EN) | **75.5** | best of 3B-active MoE group; Qwen3.8 27B dense claims 80.2 |
| Overall (DE) | **70.8** | GPT-OSS 120B claims 70.2 |
| GPQA Diamond (EN) | **84.3** | Qwen3.5 35B-A3B claims 83.8 |
| GSM8K (EN) | **89.8** | — |
| MATH Minerva (EN) | **80.1** | — |
| HumanEval (EN) | 86.7 | Qwen3.6 35B-A3B claims 96.3 |
| MBPP (EN) | 67.3 | — |
| ARC (EN) | 93.4 | — |
| MMLU-Pro (EN) | 61.1 | Qwen3-Next 80B-A3B-T claims 68.9 |

Also in the vendor table (exact values not quoted here — paging the full 178k HTML): AIME 2025/2026 (EN+DE), LiveCodeBench v6, SWE-Bench Verified, BFCL v3/v4 agentic splits, IFEval-style splits, DE twins of every eval. **Honesty note:** every Kolibri number is Aleph Alpha's own, from their eval-framework with identical prompts/settings for all models (good methodology, but self-reported — treat as claims until our arena says otherwise). AIME/LCB/SWE exact rows: read the HF card directly before quoting.

## Engine choice (Phase 2 inventory)

| Engine | Verdict | Reason |
|---|---|---|
| **vLLM 0.29 + `aleph-alpha-inference` plugin** | ✅ **CHOSEN** | The only supported stack. `Kolibri1ForCausalLM`/`kolibri1` is NOT in stock vLLM (grepped registry at tag v0.29.0 — zero hits). Plugin verified: PyPI 1.0.0, 13.4 KB pure-Python wheel, Apache-2.0, registers via `vllm.general_plugins`, subclasses **stock Qwen3Moe classes** (FusedMoE, FP8 block quant, SWA attention) — no custom CUDA, no arch-specific builds, arch-agnostic |
| EXL3 | ❌ | No EXL3 pack of Kolibri exists (checked turboderp org + HF search 2026-10-04); `kolibri1` unverified in exllamav3. **If turboderp lands `Kolibri-1-exl3` later, re-evaluate** — it would be ~40-45 GiB at 4.0bpw and change the whole math |
| TensorFold | ❌ | `kolibri1` not in TF's model table (Flash-Next/GLM-5.3/DS4 rows only; verified from TF lane runbook); 0.5.0 GPU-idle hang #122 still needs keepalive; no kolibri patches exist anywhere |
| llama.cpp | ❌ | Custom `kolibri1` model type; GGUF conversions are days old and experimental (`experimental` tag on the Q3_K_S); aarch64 build + SWA-513 MoE path on GB10 unverified; would also violate the 20GB rule (35+ GiB). Re-check if a maintained GGUF + verified arm64 path appears |
| sglang | ❌ | Arch registration is per-engine like vLLM's; no kolibri1 support exists there either |

**GB10/sm_121 verdict reasoning:** the plugin reuses vLLM 0.29's stock kernels (FP8 block GEMM, FusedMoE 384-expert, GQA + SWA attention) — all arch-agnostic CUDA already shipped in vLLM's official **arm64** image (`vllm/vllm-openai:v0.29.0` has linux/arm64, 9.6 GB compressed, per Docker Hub manifest API 2026-10-04). No JIT toolchain risk like exllamav3's sm_121 builds. Remaining risk: untested-but-plausible (no GB10 run of kolibri1 exists anywhere yet).

**Container choice:** the vendor's own image `ghcr.io/aleph-alpha/aleph-alpha-inference:latest` is **amd64-only** (OCI manifest = single amd64 entry, verified 2026-10-04) — **not usable on this aarch64 box**. Use official `vllm/vllm-openai:v0.29.0` (arm64) + `pip install --no-deps aleph-alpha-inference==1.0.0`. Flag `Dockerfile` in the vendor repo exists for an arm build later — not our problem today.

## GB10 VRAM math (DGX Spark: 121 GiB unified; ~119 GB in nvidia-smi)

Constants: weights readable **72.74 GiB**; CUDA-graph + activation transients ~2-5 GiB (budget **5**); KV cache **fp8**. Per-token KV bytes = full-attn layers × 2(K/V) × 4 kv-heads × 128 dim × 1 B = 102,400 B = **10 KiB/tok** → **2.50 GiB per 262,144-tok stream**. 40 SWA layers only need their 513-token windows (~0.02 GiB/stream) **if** the engine bounds SWA KV to the window; if it allocates SWA layers full-depth it is 50× the rate → **12.5 GiB/262k-stream**. Vendor's own hardware note (FP8 KV, 128k ctx on B200 180 GB: "78+10+free") is consistent with either reading — we must measure the actual pool at first boot.

| Config | Bounded-SWA | Unbounded-SWA | Verdict |
|---|---:|---:|---|
| 1 × 262k | 80.2 | 90.2 | ✓ safe either way (smoke) |
| 2 × 262k | 82.7 | 102.7 | ✓ tight worst-case |
| 4 × 262k | 87.7 | 127.7 | ✗ worst-case NO — measure first |
| 8 × 262k | 97.7 | 177.7 | ✗ worst-case NO |
| 4 × 128k | 82.7 | 102.7 | ✓ fits in both readings |
| 8 × 131k | 87.7 | 127.7 | ✗ worst-case NO |
| 8 × 64k | 82.7 | 102.7 | ✓ fits in both readings |

Reading: weights 72.74 + graphs 5.0 + (KV per config). All rows ≤ ~103 GiB leave ≥ ~8-10 GiB host margin (house floor under the freeze-not-OOM rule).

**Ladder plan:** smoke at **1 × 262k** → first bench at **4 × 128k** (safe under both SWA readings) → measure the real KV pool at 262k (`nvidia-smi` steady state vs 128k delta) → extend to **4 × 262k** only if the delta proves window-bounding. Cap the lane with the house systemd `MemoryMax=110G` so a pool miscalc kills the engine, not the box.

**Co-residency:**
- **Music 3 (~59 GB, ComfyUI):** 78 + 59 + KV ≈ **137+ GiB → never.** Music 3 down for the whole Kolibri lane (swap, don't stack).
- **Decision-sidecar pair (laya decider :8712 + helper, ~4.3 GiB):** fits next to the solo smoke (94.5 worst) but NOT next to 4 × 128k under unbounded reading (107 > 105) — decide per config after the pool measurement.
- Daily-driver swap frees the full ~121 GiB budget — that is the intended way to run Kolibri.

## Deploy steps (gated — read before ANY action)

**Gate:** do none of this until the swap is explicitly approved (EXL3 lane down for the duration; Music 3 down too). Weights download (~78.9 GB) is a commit-to-lane action; disk is fine (~648 GB free as of 2026-10-04, need 85).

One-time:

1. **NVMe coalescing reapply** if box was rebooted (pitfall below), then verify `nvme get-feature /dev/nvme0 -f 8 -H → TIME=0, THR=1`.
2. Build the arm64 plugin image (~9.6 GB pull from Docker Hub + ~30 s pip):
   ```bash
   ssh jaita@192.168.2.185
   d=/home/jaita/kolibri1-image; mkdir -p $d
   printf 'FROM vllm/vllm-openai:v0.29.0\nRUN pip install --no-deps aleph-alpha-inference==1.0.0\n' > $d/Dockerfile
   docker buildx build --platform linux/arm64 --load -t kolibri1:v0.29.0-p1 $d
   docker run --rm kolibri1:v0.29.0-p1 python -c "import aleph_alpha_inference, vllm; print('plugin', aleph_alpha_inference.__version__, 'vllm', vllm.__version__)"
   ```
   (`FROM arm64` natively — no emulation; official arm64 digest verified 2026-10-04.)
3. **Download weights** (the gate-approved ~78.9 GB pull; house bind-mount means HF cache lands in `/home/jaita/models/hf` directly):
   ```bash
   docker run --rm --platform linux/arm64 --ipc=host -v /home/jaita/models/hf:/root/.cache/huggingface \
     -e HF_HUB_OFFLINE=0 kolibri1:v0.29.0-p1 \
     huggingface-cli download Aleph-Alpha/Kolibri-1 \
     --revision e52eb4627d11516b0c01de49210ab5a4e4061444
   df -h /home/jaita/models   # ~79 GB gone after
   ```
4. Smoke: `~/sparkrun-recipes/scripts/switch-to-kolibri1.sh --start` (stops nothing besides EXL3/Music/Comfy per house teardown; memory preflight 100 GiB; waits on `/health`) then
   ```bash
   curl -s http://localhost:8000/v1/models | python3 -m json.tool | head -20
   curl -s http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
     -d '{"model":"Aleph-Alpha/Kolibri-1","messages":[{"role":"user","content":"Sag genau: ok"}],"max_tokens":16,"chat_template_kwargs":{"reasoning_effort":"low"}}'
   curl -s http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
     -d '{"model":"Aleph-Alpha/Kolibri-1","messages":[{"role":"user","content":"Call get_time for Berlin"}],"tools":[{"type":"function","function":{"name":"get_time","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}],"tool_choice":"auto","max_tokens":64}'   # tool-call path check
   ```
5. Then Phase B/C per the ladder, then the bench plan.

## Benchmark plan (the adopt gate — house standard)

- **Prompts:** code + chat mixes (house sets, same prompts as prior lane benches).
- **Streams:** 1 / 2 / 4 (only extend to 8 at reduced context after the pool measurement).
- **Decode:** 256 tok, **temp 0 + temp 1.0** arms both.
- **exact_tg true + tg 400** (P2 rule — never tg 128 without exact_tg); 28-cell depth×conc grid in the style of bench v27 when adopting.
- Memory floor: sample `MemAvailable` throughout; floor must stay > ~8 GiB; UMA freeze-not-OOM rule.
- Keep raw data dated under `~/sparkrun-recipes/benchmarks/` like prior benches.
- **Adopt rule:** decision is OURS on OUR data. Vendor tables are priors, not evidence.

## Rollback plan

Lane footprint (what this lane ever touches): the `kolibri1:v0.29.0-p1` image, the checkpoint under `/home/jaita/models/hf/hub/models--Aleph-Alpha--Kolibri-1`, the `kolibri1-service` systemd scope, `/tmp/kolibri1_serve.log`. The EXL3 daily driver, the vllm-fn-tp1 container, all recipes and sidecars are untouched by design. Rollback:

```bash
~/sparkrun-recipes/scripts/switch-to-kolibri1.sh --stop
free -g    # verify MemAvailable ≥ ~100 GiB for the EXL3 daily driver (house rule)
~/sparkrun-recipes/scripts/switch-to-qwen38-exl3.sh --start   # daily driver back on :8000
```

## Known pitfalls (pre-populated)

- **ninja PATH for JIT/sgl-omni-style builds:** if any on-the-fly compile stalls on "ninja is required", the CUDA toolchain bin dir is missing from PATH — `/usr/local/cuda-13.0/bin` must precede the venv bin (house pattern in `scripts/switch-to-qwen38-exl3.sh:14`). For the containerized vLLM lane this should be moot (kernels ship prebuilt), but keep the export ready for any plugin-side extension.
- **HF_HUB_OFFLINE flip is a deploy-time toggle:** first pull needs `HF_HUB_OFFLINE=0` (or unset); the recipe ships `HF_HUB_OFFLINE=1` for steady-state serving (weights local, no surprise re-fetches). The TF lane by contrast serves with `HF_HUB_OFFLINE=0` permanently — different engine, different needs; do not copy the TF env block here.
- **NVMe interrupt coalescing reapply after EVERY reboot** ([runbook](nvme-interrupt-coalescing.md)): feature 0x08 is NOT saveable on the Samsung drive — re-apply the privileged docker one-liner (`nvme set-feature /dev/nvme0 -f 8 --value=0` in `--privileged -v /dev:/dev`) and verify TIME=0/THR=1, or eat a 4.2× random-read latency penalty on uncached PLE/mmapped-KV lookups and weight loads.
- **torch cu130 + sm_121:** GB10 = sm_121 → host-venv torch must be cu130 wheels; in-container torch from the official image is already correct (NVIDIA_REQUIRE_CUDA cuda>=13.0 in the house container). The plugin is installed `--no-deps` **on purpose** — its deps (`vllm>=0.29.0,<0.30.0, torch>=2.9, transformers>=5.5.3`) come from the image; a bare `pip install aleph-alpha-inference` could replace the image's vLLM — never do that.
- **HF cache hardlink trick for local packs:** when the pack IS local, the house pattern is a straight bind-mount (`/home/jaita/models/hf:/root/.cache/huggingface`) so the container reads the cache in place — no `cp -al` needed. The hardlink trick is only for environments that cannot bind-mount (Docker Desktop/macOS).
- **UMA freeze-not-OOM:** running out of unified memory on GB10 freezes the whole box instead of raising OOM — the lane must run under `MemoryMax=110G` + `MemorySwapMax=0` (house systemd-scope pattern) so an over-allocation kills the engine, not SSH.
- **Plugin/vLLM minor pin is load-bearing:** plugin 1.0.0 supports exactly vLLM 0.29.x. Do not float the image to vllm-openai:0.30+ without re-verifying the plugin (each plugin release supports one vLLM minor — vendor README).
- **`head_dtype: float32` + `norm_topk_prob: false`** are nonstandard config fields — the plugin's `Kolibri1Config` handles both. Do NOT attempt to serve via stock `Qwen3MoeForCausalLM` with hand-edited config.
- **Vendor image is amd64-only** (`ghcr.io/aleph-alpha/aleph-alpha-inference`) — a well-meaning `docker pull` of the vendor image gives a manifest error on aarch64; use the vllm-openai arm64 build path above.
- **Served model name is the full HF id** (`Aleph-Alpha/Kolibri-1`) — clients configured for lowercase short names must be updated or the name aliased in the serve command.

## Source references

- Announcement post: https://x.com/Aleph__Alpha/status/2106306843052052616 (2026-10-03 08:54 UTC, fetched via fxtwitter API)
- HF repo: https://huggingface.co/Aleph-Alpha/Kolibri-1 (API verified 2026-10-04: sha e52eb462, 32 shards, usedStorage 78,852,606,661 B, library_name vllm; config.json: 384/6+1 experts, SWA 513, 262144 mpe)
- HF base: https://huggingface.co/Aleph-Alpha/Kolibri-1-BF16 (156.2 GB storage)
- Plugin: https://github.com/Aleph-Alpha/aleph-alpha-inference (Apache-2.0) · PyPI 1.0.0 (13.4 KB wheel, deps vllm<0.30,>=0.29.0 / torch>=2.9.0 / transformers>=5.5.3) · vendored image ghcr.io/aleph-alpha/aleph-alpha-inference (amd64-only, verified)
- Launch blog: https://aleph-alpha.com/en/blog/kolibri-has-landed-a-sovereign-open-weight-model/ · Tech report: https://aleph-alpha.com/downloads/tech-report.pdf
- Engine image: vllm/vllm-openai:v0.29.0 — arm64 manifest confirmed via Docker Hub API (8.7 GB amd64 / 9.6 GB arm64)
- House rollback target: [qwen38-flash-next-exl3-daily-driver.md](qwen38-flash-next-exl3-daily-driver.md) · NVMe: [nvme-interrupt-coalescing.md](nvme-interrupt-coalescing.md) · TF context: [qwen38-flash-next-tensorfold.md](qwen38-flash-next-tensorfold.md)
