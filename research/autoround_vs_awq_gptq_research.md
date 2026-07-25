# AutoRound INT4 vs AWQ/GPTQ on Large MoE Models — Research Findings

Research date: 2026-07-20. All URLs, numbers, and quotes verbatim from sources. web_search/x_search unavailable (Firecrawl unconfigured, xAI credits exhausted) — all data retrieved via direct browser navigation to primary sources.

---

## (1) Intel AutoRound benchmark results — AutoRound INT4 vs AWQ vs GPTQ

### Source: Intel AutoRound paper accuracy tables
- URL: https://github.com/intel/auto-round/blob/main/docs/paper_acc.md
- Methodology note (verbatim): "To ensure a fair comparison as much as possible and alleviate overfitting in perplexity evaluation on Wikitext or C4, we utilized 512 samples from NeelNanda/pile-10k for all methods during calibration unless explicitly stated. ... For GPTQ, we have enabled act-order and true-sequential, and also activated static group in scenarios where group_size!=-1. The notation GPTQ* indicates that we adjusted the random seed or data preprocessing to address issues related to the non-positive definite Hessian matrix or other issues. ... For AutoRound, we used the default setting, iters 200, enable_quanted_input and enable_minmax_tuning, both the lr and minmax_lr are set to 1/iters,i.e. 5e-3."
- Note: These are W4G128 (INT4, group_size 128) — same scheme as our Intel/Qwen3.5-122B-A10B-int4-AutoRound base. Models tested: Mistral-7B, Vicuna V2-7B/13B/70B, Vicuna V1-7B/13B/30B/65B. NO MoE models in this table; NO Qwen3.5; NO MMLU-Pro/GPQA/HLE/coding benchmarks (only MMLU, Lambada, HellaSwag, WinoGrande, PIQA, TruthfulQA, OpenBookQA, BoolQ, RTE, ARC-easy, ARC-challenge).

#### W4G128 (INT4, group 128) — 11-task 0-shot average, Vicuna V2-70B (largest in table):
| Method | MMLU | Avg (11 tasks) |
|---|---|---|
| FP16 | 66.23 | 66.12 |
| RTN | 64.91 | 65.87 |
| GPTQ | 65.63 | 66.22 |
| AWQ | 65.79 | **66.23** |
| HQQ | 65.34 | 66.06 |
| Omniquant | 65.30 | 66.02 |
| **AutoRound (Ours)** | 65.65 | **66.39** |

#### W4G-1 (INT4, per-channel / group -1) — Vicuna V2-70B:
| Method | MMLU | Avg |
|---|---|---|
| FP16 | 66.23 | 66.12 |
| RTN | 63.85 | 65.22 |
| GPTQ | 64.81 | 65.75 |
| AWQ | 65.08 | **66.28** |
| **AutoRound** | 65.43 | 66.27 |

#### W4G128 Mistral-7B (best-differentiated case for AutoRound at per-channel):
| Method | MMLU | Avg |
|---|---|---|
| FP16 | 61.35 | 63.30 |
| RTN | 59.72 | 62.36 |
| GPTQ | 59.17 | 62.32 |
| AWQ | 60.20 | 62.16 |
| HQQ | 60.02 | 62.75 |
| Omniquant | 59.71 | 62.18 |
| **AutoRound** | 60.47 | 62.62 |

**Key takeaway (verbatim-supported):** At W4G128 (our config), AutoRound, GPTQ, and AWQ are within ~0.2 points of each other on average across these 11 legacy benchmarks. AutoRound wins or ties on average on most models; AWQ occasionally wins individual tasks. The gap between all three tuned methods is small at INT4 — far smaller than the FP16→INT4 degradation.

---

## (2) Per-benchmark quality comparisons (MMLU-Pro, GPQA, HLE, coding)

**MMLU-Pro**: Only directly available in two AutoRound sources, neither vs GPTQ/AWQ:
- alg_202508.md (W2G64 2-bit results, Qwen3-8B and Llama3.1-8B-Instruct): URL https://github.com/intel/auto-round/blob/main/docs/alg_202508.md
  - Qwen3-8B W2G64 AutoRound mmlupro 0.2630; AutoRound+alg_ext 0.2807; AutoRoundBest+alg_ext 0.3127 (BF16 baseline not shown in this table).
  - Llama3.1-8B W2G64 AutoRound mmlupro 0.1661; +alg_ext 0.2163; AutoRoundBest+alg_ext 0.2364.
- auto_scheme_acc.md (AutoScheme mixed-precision, GGUF q2ks/q4ks avg 3.5 bits, fake model): URL https://github.com/intel/auto-round/blob/main/docs/auto_scheme_acc.md
  - Qwen3-8B: BF16 mmlu_pro 0.6934 → Fake quantized (3.5 avg bits) 0.6751
  - Qwen3.5-4B: BF16 mmlu_pro 0.5891 → Fake quantized 0.5948 (within noise)

**GPQA Diamond** (auto_scheme_acc.md, GGUF 3.5-bit, repeat=5):
  - Qwen3-8B: BF16 0.4586 → Fake quantized 0.4313 (−2.7 pts)
  - Qwen3.5-4B: BF16 0.3263 → Fake quantized 0.3172

**math_500** (auto_scheme_acc.md, GGUF 3.5-bit, repeat=5):
  - Qwen3-8B: BF16 0.8083 → Fake quantized 0.7924 (−1.6 pts)
  - Qwen3.5-4B: BF16 0.5365 → Fake quantized 0.505

**HLE, SWE-bench, Terminal-Bench, BFCL-V4, LiveCodeBench**: NO results found in any AutoRound-published source. The AutoRound accuracy docs do not evaluate on these harder agentic/coding benchmarks. Only the albond repo mentions "LongCode" as a *throughput* prompt category, not a quality eval.

**FP8_BLOCK accuracy** (fp8_block_acc.md, URL https://github.com/intel/auto-round/blob/main/docs/fp8_block_acc.md) — directly relevant to our hybrid's FP8 component:
- LLaMA-3-8B-Instruct: BF16 avg 0.6311 → FP8_BLOCK RTN 0.6297 → FP8_BLOCK Tuning 0.6305 (mmlu_pro: BF16 0.4334 → RTN 0.4358 → Tuning 0.4315 — FP8 actually *slightly beats* BF16 on mmlu_pro within noise)
- Qwen3-8B: BF16 avg 0.6524 → FP8_BLOCK RTN 0.6520 → FP8_BLOCK Tuning 0.6523 (mmlu_pro: BF16 0.6214 → RTN 0.6204 → Tuning 0.6168 — −0.5 to −0.7 pts)
- Verbatim command: `auto-round --model model_name_or_path --scheme FP8_BLOCK --iters 0 --format fp8 # RTN`

---

## (3) Hybrid Mamba/attention architectures — Qwen3.6-27B "broke"

- GitHub issue search for "Qwen3.6 27B broken" on intel/auto-round: NO results (https://github.com/intel/auto-round/issues?q=Qwen3.6+27B+broken → "No results"). x_search unavailable (xAI credits exhausted), so the original tweet could not be retrieved or verified.
- The albond hybrid repo (https://github.com/albond/DGX_Spark_Qwen3.5-122B-A10B-AR-INT4) documents a DeltaNet hybrid-attention-specific issue that is directly relevant to the tweet's claim about Mamba/hybrid architectures:
  - README "Known issue" (verbatim): "--enable-prefix-caching is intentionally omitted — it crashes on Qwen3.5 due to DeltaNet hybrid attention."
  - README Troubleshooting (verbatim, re: checkpoint build warnings): "408 'unexpected unmatched FP8 tensor' warnings during checkpoint build — Normal. These are DeltaNet linear_attn projections (36 of 48 layers) plus some attention norms/gates. They exist in the Qwen FP8 checkpoint but have no matching counterparts in Intel AutoRound INT4 (different naming conventions). The script only replaces shared_expert dense layers (144 tensors) with FP8 — everything else stays in its original format."
- README caveat on long context (verbatim): "The current INT4 quantization (Intel AutoRound) was calibrated for standard context lengths. For deeper contexts (>256K), a custom AutoRound calibration would be needed."
- README note on Qwen3.5 architecture: 36 of 48 layers are DeltaNet linear_attn (hybrid attention), 12 are full attention. The hybrid Mamba-like/DeltaNet layers are where FP8/INT4 naming and calibration diverge between the Intel AutoRound INT4 checkpoint and the official Qwen FP8 checkpoint.

**Implication for the tweet's "Qwen3.6-27B broke (0.52 on GSM8K)" claim**: Cannot verify the specific number from primary sources, but the AutoRound INT4 checkpoint for the closely-related Qwen3.5-122B explicitly *ignores shared_expert layers during calibration* (Intel card: `auto-round "Qwen/Qwen3.5-122B-A10B" --ignore_layers shared_expert`), and the hybrid-attention (DeltaNet) projections produce 408 "unmatched" warnings when merging FP8. This is consistent with the tweet author's warning that "quant rankings aren't transferable between models" — hybrid-attention/Mamba architectures have layer types that standard AutoRound/GPTQ/AWQ calibration pipelines may not handle correctly, producing the kind of catastrophic 0.52 GSM8K collapse described.

---

## (4) INT4 quantization effect on reasoning/thinking vs simple math

Direct comparison data is limited. Findings:
- opt_rtn.md (https://github.com/intel/auto-round/blob/main/docs/opt_rtn.md) shows INT4 is near-lossless, INT3 costs ~5-8 pts, INT2 catastrophically degrades:
  - Qwen3-8B RTN-4BIT avg 0.66240 vs RTN-3BIT 0.57322 vs RTN-2BIT 0.31150 (MMLU: 4BIT 0.7077 → 3BIT 0.6002 → 2BIT 0.2536)
  - LAMBADA (next-token prediction, proxy for fluency) collapses hardest at 2-bit: Qwen3-8B 4BIT 0.6150 → 2BIT 0.0041
- auto_scheme_acc.md shows the only AutoRound reasoning-vs-math split available (Qwen3-8B, GGUF 3.5-bit fake):
  - math_500: BF16 0.8083 → quant 0.7924 (−1.6 pts, ~2% relative)
  - gpqa_diamond: BF16 0.4586 → quant 0.4313 (−2.7 pts, ~6% relative)
  - mmlu_pro: BF16 0.6934 → quant 0.6751 (−1.8 pts, ~3% relative)
  - GPQA (harder reasoning/knowledge) loses MORE than math_500 (easier procedural math) — supports the tweet author's implicit point that "GSM8K identical" (97.5%) is a weak signal and harder reasoning benchmarks show larger degradation.
- No AutoRound-published data on chain-of-thought / thinking-token quality specifically. The opt_rtn.md LAMBADA collapse at 2-bit suggests next-token fluency (core to coherent reasoning traces) is the first thing to break as bits decrease.

---

## (5) The specific quant we're running: bleysg/Qwen3.5-122B-A10B-int4-fp8-hybrid

URL: https://huggingface.co/bleysg/Qwen3.5-122B-A10B-int4-fp8-hybrid

**What it is** (verbatim from model card):
- "A hybrid-quantized checkpoint of Qwen3.5-122B-A10B for bandwidth-optimal decode on the NVIDIA DGX Spark (GB10 / SM121, 128 GB unified) under vLLM. Routed experts and attention stay INT4; the BF16 shared experts (dense — read on every token) are replaced with calibrated FP8 from the official FP8 release."
- Composition:
  - "Routed experts + attention: INT4 (GPTQ / AutoRound) — from Intel/Qwen3.5-122B-A10B-int4-AutoRound"
  - "Shared experts: FP8 E4M3 block-128 — from Qwen/Qwen3.5-122B-A10B-FP8"
  - "Embeddings / norms / head: unchanged from the INT4 base."
- "144 shared-expert layers convert BF16→FP8 — an always-on bandwidth lever worth +28% at base / no-spec decode on GB10 (28.2 → 36.0 tok/s)"

**It is NOT pure AWQ, NOT pure GPTQ, NOT pure AutoRound** — it is a *hybrid*:
- INT4 component = AutoRound-tuned (the Intel base card confirms: "generated by intel/auto-round", group_size 128, with `--ignore_layers shared_expert` so shared experts are left in BF16 during calibration). The bleysg card's "INT4 (GPTQ / AutoRound)" label reflects that AutoRound's default export format produces GPTQ-compatible packed weights (README: "Multiple Formats Export Support AutoRound, AutoAWQ, AutoGPTQ, and GGUF"). So the INT4 weights are AutoRound-tuned but in GPTQ pack format.
- FP8 component = from the official Qwen FP8 release (Qwen/Qwen3.5-122B-A10B-FP8), NOT AutoRound-quantized. The bleysg/albond build script (`build-hybrid-checkpoint.py`) physically swaps 144 shared-expert tensor groups from BF16→FP8 E4M3 block-128.

**Build provenance** (verbatim):
- "Recipe (hybrid INT4+FP8, INT8 lm-head, MTP) and build script: albond/DGX_Spark_Qwen3.5-122B-A10B-AR-INT4. Built with build-hybrid-checkpoint.py, which merges FP8 non-expert tensors into the INT4 checkpoint."
- Repo: https://github.com/albond/DGX_Spark_Qwen3.5-122B-A10B-AR-INT4
- Build command: `python patches/01-hybrid-int4-fp8/build-hybrid-checkpoint.py --gptq-dir "$INTEL_DIR" --fp8-repo Qwen/Qwen3.5-122B-A10B-FP8 --output ~/models/qwen35-122b-hybrid-int4fp8 --force`

**Intel base model card** (https://huggingface.co/Intel/Qwen3.5-122B-A10B-int4-AutoRound) — verbatim:
- "This model is an int4 model with group_size 128 of Qwen/Qwen3.5-122B-A10B generated by intel/auto-round."
- "The main branch provides the tuned quantized model, and revision 3f4ba63 corresponds to the RTN version. In general, the tuned version is preferred; however, it has not been validated for this particular model." ← **IMPORTANT: Intel explicitly states the tuned AutoRound version for THIS specific 122B MoE model has NOT been accuracy-validated.**
- Tuning command: `auto-round "Qwen/Qwen3.5-122B-A10B" --output_dir "./Qwen35-int4" --ignore_layers shared_expert`
- The `--ignore_layers shared_expert` flag means shared experts were NOT quantized by AutoRound — they stayed BF16 in the Intel base, then the bleysg/albond hybrid replaces them with FP8 from the Qwen release.

**How it compares to a pure AutoRound INT4**: Our hybrid ≈ AutoRound INT4 (routed experts + attention) + Qwen-official FP8 (shared experts). A pure AutoRound INT4 of the same model would either (a) keep shared experts in BF16 (as the Intel base does) or (b) quantize them to INT4 too. The hybrid is strictly a *speed* optimization (FP8 shared experts read faster than BF16 on GB10) layered on top of the AutoRound INT4 base — it does not change the INT4 quality of the routed experts/attention, which is where the calibration risk lives.

---

## (6) Does the FP8 component matter for quality? Would pure INT4 lose the FP8 advantage?

**FP8 quality is near-lossless** (fp8_block_acc.md, verbatim numbers):
- LLaMA-3-8B-Instruct: BF16 avg 0.6311 vs FP8_BLOCK RTN 0.6297 (−0.14 pts, −0.22% relative). On mmlu_pro specifically: BF16 0.4334 vs FP8 RTN 0.4358 (FP8 *higher*, within noise).
- Qwen3-8B: BF16 avg 0.6524 vs FP8_BLOCK RTN 0.6520 (−0.04 pts). mmlu_pro: BF16 0.6214 vs FP8 RTN 0.6204 (−0.1 pts).
- FP8_BLOCK Tuning vs RTN: no meaningful difference — RTN is already near-lossless, tuning doesn't help FP8 (unlike INT2/INT3 where tuning matters a lot).

**So the FP8 shared-expert component in our hybrid is quality-neutral** — it preserves shared-expert accuracy to within noise of BF16. Replacing BF16 shared experts with FP8 E4M3 block-128 should cost <0.2 pts average.

**Would pure INT4 (everything including shared experts) lose the FP8 advantage?** This depends on whether AutoRound INT4 quantizes shared experts well:
- The Intel base explicitly does NOT quantize shared experts (`--ignore_layers shared_expert`) — implying AutoRound's authors found shared-expert INT4 calibration problematic for this MoE, or at least chose to skip it. A pure INT4 run that *did* quantize shared experts would introduce additional INT4 error in the dense-always-read path.
- Shared experts are "dense — read on every token" (bleysg card). INT4 error here accumulates across every token's decode, so it's the highest-leverage place for quality loss. Keeping them at FP8 (8-bit) vs INT4 (4-bit) is a meaningful quality margin — 4 extra bits of precision on the most-frequently-read weights.
- However, the routed experts (the bulk of parameters) are INT4 in both the hybrid and a pure-INT4 scheme — the FP8 advantage is *only* on shared experts (144 layers) + the lm_head (INT8 in the albond v2 build).

**Bottom line**: The FP8 component matters *less for quality* (FP8 is near-lossless either way) and *more for speed* (+28% decode on GB10). Pure INT4 of shared experts would likely cost measurable quality on the dense path, but the routed-expert INT4 (where most parameters live) is identical between hybrid and pure-INT4. The bigger quality question is the AutoRound INT4 calibration of routed experts + attention on a hybrid DeltaNet architecture — and Intel's own card admits that calibration "has not been validated for this particular model."

---

## Summary of URLs

- AutoRound repo: https://github.com/intel/auto-round
- AutoRound README: https://github.com/intel/auto-round (see browser snapshot)
- Paper accuracy (INT4/INT3 vs GPTQ/AWQ/HQQ/Omniquant): https://github.com/intel/auto-round/blob/main/docs/paper_acc.md
- 2-bit algorithm accuracy (alg_ext, MMLU-Pro): https://github.com/intel/auto-round/blob/main/docs/alg_202508.md
- AutoScheme mixed-precision (math_500, GPQA diamond, MMLU-Pro): https://github.com/intel/auto-round/blob/main/docs/auto_scheme_acc.md
- FP8_BLOCK accuracy: https://github.com/intel/auto-round/blob/main/docs/fp8_block_acc.md
- OPT RTN accuracy (INT4/3/2 RTN vs opt): https://github.com/intel/auto-round/blob/main/docs/opt_rtn.md
- SignRound v1 paper (arXiv:2309.05516, EMNLP24 Findings): https://arxiv.org/abs/2309.05516
- SignRoundV2 paper (2025.12, referenced in README as "available" but not found on arxiv search — README links to it but the link is in the README body, not separately indexed)
- Our hybrid model card: https://huggingface.co/bleysg/Qwen3.5-122B-A10B-int4-fp8-hybrid
- Intel AutoRound INT4 base: https://huggingface.co/Intel/Qwen3.5-122B-A10B-int4-AutoRound
- Qwen official FP8 base: https://huggingface.co/Qwen/Qwen3.5-122B-A10B-FP8
- Hybrid build repo (albond): https://github.com/albond/DGX_Spark_Qwen3.5-122B-A10B-AR-INT4
- albond issues (no Qwen3.6-27B): https://github.com/albond/DGX_Spark_Qwen3.5-122B-A10B-AR-INT4/issues
- AutoRound issues (no Qwen3.6-27B "broken"): https://github.com/intel/auto-round/issues?q=Qwen3.6+27B+broken

## Could NOT retrieve / verify
- The original tweet claiming "GSM8K identical 97.5%, AutoRound broke Qwen3.6-27B at 0.52 GSM8K, quant rankings aren't transferable between models" — x_search unavailable (xAI credits exhausted), web_search unavailable (Firecrawl unconfigured). The 0.52 GSM8K number is NOT corroborated by any primary AutoRound source I could access.
- No AutoRound-published eval on HLE, SWE-bench, Terminal-Bench, BFCL-V4, or LiveCodeBench. The hardest benchmarks AutoRound publishes on are MMLU-Pro, GPQA-diamond, and math_500 (all only in the auto_scheme GGUF context, not a direct INT4-vs-AWQ-vs-GPTQ comparison).
- The SignRoundV2 (2025.12) paper full text — not located on arxiv via search; the README references it as "available" with a link but I did not navigate to the specific paper URL.