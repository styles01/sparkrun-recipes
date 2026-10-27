# ESTATE SIGNALS LOG — scout reports worth keeping (Oracle)

## 2026-10-17 · ComfyUI-Ref2VA-VSA (parked Spark A/B)
- Signal: video sparse attention (VSA) for MiniMax H3 Ref2VA character video —
  claimed 9× speedup (72s vs ~650s for 5.16s/124-frame/1344×768 on RTX 4090),
  peak VRAM 13.5 GB (from ~22.5), 4-step Euler, 75% video sparsity,
  REFERENCE TOKENS LOCKED OUT OF PRUNING (face-drift protection).
- Source: X promo by @forhanvv (low authority, engagement-farming format, NSFW
  roundup in same post) → artifact legit: github.com/Kablex/ComfyUI-Ref2VA-VSA
  (560★, pushed same day, community PRs, Apache-style measured docs).
- Why it matters: our estate runs MiniMax H3 Ref2VA via ComfyUI on the Sparks;
  VSA claims fix exactly our pains (ref clips ~3× render time, memory-guard
  kills, int8 Turbo fast-motion regression). Orthogonal to adapter/quant work —
  compounds if it ports to sm_121.
- Confidence: HIGH (tweet verbatim + repo real), MEDIUM (9×/13.5GB — repo
  reported, 4090 only), LOW-MEDIUM (ports to our int8/TF stack without motion
  fidelity loss — unknown).
- PARKED ACTION (behind GPU windows): head-to-head Spark A/B — Ref2VA-VSA node
  vs our MiniMaxH3ReferenceToVideo path, same reference clip; check sm_121
  compat + fast-motion fidelity vs the known Turbo regression. No recipe merge
  before by-eye comparison.
- Trigger to re-check: repo maturity, third-party sm_121 repack, or a Spark
  owner publishing numbers.
## 2026-10-17 — Mia-AiLab: Qwen3.8-Flash-Next FP4-QAD on a SINGLE DGX Spark (TensorFold)
- **Signal**: Mia posts "Rejoice single DGX Spark owners!" — Qwen3.8-Flash-Next served on ONE spark via TensorFold with Nvidia FP4 QAD quant: FP8 KV 1.1M pool, 256K ctx, 5 concurrent, 46-55 tok/s single-stream prose, 132 tok/s x5. Link: mia-ai.net/models/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold (X status 2111540267840901174, 10/17 19:30Z, fxtwitter-verified).
- **Why it matters**: estate has measured FP4-on-GB10 at 26-33 tok/s (Marlin-emulated) and parked FP4 because TF 0.6.0 cannot read csf scales (repack needed). Mia's numbers are ~1.5-2x that and on TensorFold natively — implies the repack/native-csf path already exists publicly. If real: qwen-fn could run on spark2 alone (frees spark1 entirely for dsv41 TP2, no lane-sharing) or give 5-concurrency 256K on one node.
- **Timeline**: probe next media/lane window: fetch their model page + check whether it is a csf→TF repack of nvidia/Qwen3.8-Flash-Next-FP4-Local (which we already mirror-hold upstream). Parked A/B per James 10/16 'canonical retained' still stands — this is the published-throughput re-poll trigger the registry named.
- **Stakeholders**: qwen-fn lane, media-mode sovereignty plan, spark2 capacity.
- **Confidence**: MEDIUM-HIGH that the artifact exists (Mia has shipped real TF recipes before); throughput UNVERIFIED on our hardware.
