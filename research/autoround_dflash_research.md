# AutoRound INT4 × DFlash Compatibility Research — Qwen3.5-122B-A10B

## Summary Verdict
**AutoRound INT4 IS compatible with DFlash. You are ALREADY using AutoRound.**
The bleysg hybrid repo is built directly on Intel's official AutoRound INT4 checkpoint.
The DFlash drafter is quantization-agnostic — it reads target hidden states, not quantized weights.

---

## (1) Intel AutoRound — Qwen3.5 MoE Support: ✅ YES
- Repo: https://github.com/intel/auto-round (1.5k stars, 1290 commits, active — last commit 13h ago)
- Qwen3.5 MoE support PR (MERGED): https://github.com/intel/auto-round/pull/1476
  - Title: "support qwen3_5 moe" by wenhuach21, merged Mar 1, 2026
  - 21 commits, 269 additions / 134 deletions
- README explicitly lists Qwen3 examples (`Qwen/Qwen3-0.6B`, `Qwen/Qwen3-8B`)
- Architecture `qwen3_5_moe` / `Qwen3_5MoeForConditionalGeneration` confirmed in Intel's checkpoint config
- MTP layer quantization supported (2026/03 PR, per README changelog)

## (2) AutoRound + Speculative Decoding / MTP / DFlash
- **No AutoRound issues mention DFlash** (searched: `DFlash OR speculative OR MTP` → 0 DFlash hits)
- MTP quantization is supported but had Qwen3.5-specific bugs:
  - "ignore mtp.fc for qwen3_5 due to vllm failure" — MTP.fc SKIPPED for Qwen3.5 (vLLM incompat)
  - "Revert ignore mtp.fc for qwen3_5 due to vllm failure and fix"
  - "[Bug]: Qwen3.5-9B with quantized lm_head throws exceptions in vLLM serving" (#1650, fixed in 0.13.0)
  - "[Bug]: Qwen3.5 series model fail infer for gguf format" (#1513, fixed 0.12.0)
- DFlash support in vLLM is NOT merged (PRs closed, not merged):
  - https://github.com/vllm-project/vllm/pull/40898 — DFlash + SWA drafter (Closed, not merged)
  - https://github.com/vllm-project/vllm/pull/39995 — DFlash FlashInfer + FP8 KV cache (Closed, not merged)
- DFlash support in SGLang IS available (`--speculative-algorithm DFLASH`)

## (3) HuggingFace AutoRound-quantized Qwen3.5-122B-A10B repos (5 found)
Search: https://huggingface.co/models?search=autoround+qwen+122b
1. **Intel/Qwen3.5-122B-A10B-int4-AutoRound** — OFFICIAL, 180k dl/mo, 76.8GB, 14 shards
   - https://huggingface.co/Intel/Qwen3.5-122B-A10B-int4-AutoRound
2. **happypatrick/Qwen3.5-122B-A10B-heretic-int4-AutoRound** — 5k dl, 19B, Image-Text-to-Text
   - https://huggingface.co/happypatrick/Qwen3.5-122B-A10B-heretic-int4-AutoRound
3. **Intel/Qwen3.5-122B-A10B-gguf-q2ks-mixed-AutoRound** — GGUF mixed-bits, 154 dl
   - https://huggingface.co/Intel/Qwen3.5-122B-A10B-gguf-q2ks-mixed-AutoRound
4. **shieldstar/Qwen3.5-122B-A10B-int4-AutoRound-EC** — 6.8k dl, 21B
   - https://huggingface.co/shieldstar/Qwen3.5-122B-A10B-int4-AutoRound-EC
5. **simonepstein/Qwen3.5-122B-A10B-int4-AutoRound-OpenCode** — 454 dl, 18B
   - https://huggingface.co/simonepstein/Qwen3.5-122B-A10B-int4-AutoRound-OpenCode

## (4) bleysg Hybrid Quant — IT IS AutoRound, NOT AWQ
- Repo: https://huggingface.co/bleysg/Qwen3.5-122B-A10B-int4-fp8-hybrid
- **Tags include `auto-round`** (explicitly)
- Model card: "A hybrid-quantized checkpoint... Routed experts and attention stay INT4; the BF16 shared_expert (dense — read on every token) are replaced with FP8 from the official FP8 release."
- Composition (from model card):
  - INT4 base: `Intel/Qwen3.5-122B-A10B-int4-AutoRound`
  - Shared experts: `Qwen/Qwen3.5-122B-A10B-FP8` (FP8)
  - Recipe/build script: `albond/DGX_Spark_Qwen3.5-122B-A10B-AR-INT4` (404 on HF — likely GitHub repo)
- bleysg config.json confirms: `quant_method: auto-round`, `packing_format: auto_round:auto_gptq`, `bits: 4`, `group_size: 128`, `autoround_version: 0.12.0`
- Intel base config.json confirms identical quant config + shared experts kept at 16-bit float in `extra_config`
- **Conclusion: bleysg IS AutoRound INT4 (routed experts) + FP8 (shared experts). NOT AWQ.**

## (5) vLLM Issues — AutoRound INT4 + Speculative Decoding
- No vLLM issues found specifically about AutoRound + speculative decoding
- DFlash vLLM PRs are CLOSED/unmerged (#40898, #39995) — DFlash in vLLM is pre-merge
- Qwen3.5 lm_head quantization had vLLM serving bugs (#1650, fixed in 0.13.0)
- AutoRound README: "Please note that support for the MoE models and visual language models is currently limited" (SGLang note)

## (6) DFlash Drafter — Quantization-Agnostic: ✅ YES
- Repo: https://huggingface.co/z-lab/Qwen3.5-122B-A10B-DFlash (also mirrored: modal-labs/Qwen3.5-122B-A10B-DFlash)
- config.json:
  - `architectures: ["DFlashDraftModel"]`
  - `quantization_config: NONE` — drafter is BF16, unquantized
  - `dflash_config: {"block_size": 16, "mask_token_id": 248077, "target_layer_ids": [1, 7, 14, 20, 26, 32, 39, 45]}`
- The drafter reads **hidden states** from target model layers [1,7,14,20,26,32,39,45] — NOT quantized weights
- Model card: "This repository contains a DFlash draft model for Qwen/Qwen3.5-122B-A10B. It is not a standalone language model."
- Example SGLang launch uses unquantized `Qwen/Qwen3.5-122B-A10B` as target — but the mechanism (hidden-state transfer) is quant-format agnostic
- The drafter was trained against the BF16 target; quantization of target affects hidden-state distribution, which *could* marginally affect acceptance length, but does NOT break compatibility

## Key URLs
- https://github.com/intel/auto-round
- https://github.com/intel/auto-round/pull/1476 (Qwen3.5 MoE support, merged)
- https://github.com/intel/auto-round/issues?q=DFlash+OR+speculative+OR+MTP (no DFlash issues)
- https://huggingface.co/Intel/Qwen3.5-122B-A10B-int4-AutoRound (official AutoRound INT4)
- https://huggingface.co/bleysg/Qwen3.5-122B-A10B-int4-fp8-hybrid (your current model — IS AutoRound)
- https://huggingface.co/z-lab/Qwen3.5-122B-A10B-DFlash (drafter, BF16, no quant)
- https://huggingface.co/models?search=autoround+qwen+122b (5 AutoRound Qwen 122B variants)
- https://github.com/vllm-project/vllm/pull/40898 (DFlash vLLM PR, closed/unmerged)
- https://github.com/vllm-project/vllm/pull/39995 (DFlash FlashInfer+FP8 KV, closed/unmerged)

## Critical Implications for Your Setup
1. You are NOT on AWQ — bleysg is AutoRound INT4+FP8 hybrid. The @superalesha tweet comparing "AutoRound vs AWQ" is comparing your current quant family vs a different one.
2. The drafter is quantization-agnostic by architecture (reads hidden states, `quantization_config: NONE`).
3. Switching to a pure AutoRound INT4 (without FP8 shared experts) WOULD change hidden-state distributions vs the hybrid, potentially affecting DFlash acceptance rate — but this is an accuracy/throughput tradeoff, not a hard incompatibility.
4. DFlash in vLLM is NOT merged — if you're running DFlash, you're likely on SGLang or a custom vLLM fork.
5. The 110 tok/s / 347K KV claim from AutoRound INT4 vs your 82.8 tok/s / 504K KV may come from dropping the FP8 shared-expert path (pure INT4 = smaller footprint, more KV headroom but different speed/quality tradeoff).