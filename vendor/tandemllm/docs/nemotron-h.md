# NemotronH (Nemotron 3.5 Lightning, Bolt, Super VL)

NVIDIA's NemotronH hybrid is the fourth model family TandemLLM runs: Mamba2, NoPE attention and mixture-of-experts blocks, with an MTP head on some checkpoints. The engine serves Nemotron 3.5 Bolt 74B-A9B from NVIDIA's own 24 GB GGUF (IQ2_XXS experts, NVFP4 Mamba2 projections, Q8_0 head and attention), read as stored, and from EXL3 packs converted from NVIDIA's BF16 checkpoint (2.0 to 6.0 bpw; the 2.0 pack has 4-bit dense projections), all with NVIDIA's LiLiCorr drafter; Nemotron 3.5 Super VL 120B-A12B from NVIDIA's ModelOpt checkpoint (NVFP4 experts, FP8 Mamba2 and attention projections and head) and from EXL3 packs of its BF16 checkpoint, with its own LiLiCorr drafter; and Nemotron 3.5 Lightning 30B-A3B in BF16, the test model it was built on. Super VL reads images through its checkpoint's RADIO tower ([Super VL: images](#super-vl-images)); Bolt and Lightning are served text only.

Code: `engine/models/nemotron_h/` (registry name `nemotron_h`; the image path `vision.py`, selected by `NemotronHForCausalLM`, `NemotronHPuzzleForCausalLM` and the Omni architectures `NemotronH_*Omni_Reasoning_V3`). Ladder tool: `tools/nh_check.py`; quality against BF16 logits: `tools/nh_quality.py`. GGUF weights (IQ2_XXS, NVFP4, Q8_0): `tools/iq2_linear.py`. EXL3 packs: `tools/fn_kernels.py` (`dense_rows`) and `tools/exl3_relu2.py`. ModelOpt NVFP4 / FP8 checkpoints: `tools/nvfp4_relu2.py`. Draft head: `engine/drafters/lilicorr.py`.

## The model

| part | Lightning 30B-A3B | Bolt 74B-A9B | Puzzle 75B-A9B (Bolt's trunk design) | Super VL 120B-A12B |
|-|-|-|-|-|
| layers | 52: 23 Mamba2, 23 MoE, 6 attention | 88: 40 Mamba2, 40 MoE, 8 attention | the same as Bolt | the same as Bolt (`layers_block_type`) |
| hidden | 2,688 | 4,096 | 4,096 | 4,096 |
| Mamba2 | 64 heads of 64, state 128, 8 groups, conv 4 taps with bias | 128 heads of 64, state 128, 8 groups | 128 heads of 64, state 96, 8 groups | 128 heads of 64, state 128, 8 groups |
| attention | 32 query heads, 2 KV heads of 128, no positional encoding | the same | the same | the same |
| MoE | 128 experts, top 6, relu^2 experts 1,856 wide, shared expert 3,712 | 240, 304, 368, 432 or 496 experts a layer, top 9, width 1,536..2,560, latent 1,024, shared 5,376 | 512 experts, top 4..18 and width 1,280..2,688 a layer, latent 1,024, shared 5,376 | 512 experts in every layer, top 22, width 2,688, latent 1,024, shared 5,376 |
| router | sigmoid, correction bias for the choice, top k / (sum + 1e-20) x 2.5 | the same, x 5.0 | the same, x 5.0 | the same, x 5.0 |
| MTP | one block pair, attention + MoE | none | the same as Lightning, top 22, width 2,688 | one block pair, attention + MoE (`mtp_layers_block_type`), top 22, width 2,688 |
| vocabulary, context | 131,072, 262,144 | the same | the same | 131,072, 1,048,576 |

Bolt's checkpoint is an Omni wrapper (`NemotronH_Omni_Reasoning_V3`) with a RADIO image tower; its GGUF holds the text trunk only (the tower is the separate `mmproj` file), and TandemLLM serves text.

The config reader (`config.py`) builds one layer table from `hybrid_override_pattern`, `layers_block_type` or `block_configs` (per-block `num_experts_per_tok`, `moe_intermediate_size` and `n_routed_experts`), and from an Omni checkpoint's `llm_config`. An unknown field stops the load with its name.

A decode token reads 5.16 GB on Bolt: the NVFP4 Mamba2 projections 2.47 GB, the Q8_0 head 0.57, shared experts 0.45 and routed experts 0.44 (IQ2_XXS, 9 a layer), latent projections 0.36 and attention 0.30 (Q8_0), the SSM states 0.34 (read and written), the F32 routers 0.23 (`GGUFIndex.decode_bytes`); 52.9 tok/s at 273 GB/s is the bytes ceiling. A decode token reads 6.55 GB of Lightning's BF16 weights: experts 2.75 GB, Mamba2 projections 1.78, shared experts 0.92, head 0.70, attention 0.28 (`PackIndex.decode_bytes`); 41.7 tok/s at 273 GB/s is the bytes ceiling.

## Profiles

### Bolt: nemotron-bolt-iq2

`nemotron-bolt-iq2` (`ops/profiles/nemotron-bolt-iq2.env`): NVIDIA's GGUF of revision 724e3fee of nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF (22.0 GB, the registry entry's asset, fetched from NVIDIA's repository and never re-hosted; `QSE_NH_GGUF` names it), with the config, tokenizer and chat template of the BF16 repository (revision 153d4551) in the profile's weights directory; 22.19 GiB at load; 262,144 tokens; 3 slots; the lookup cut by StairCut (Bolt has no MTP head). With NVIDIA's LiLiCorr drafter beside the lookup (`QSE_NH_HEAD=lilicorr`) StairCut kept plain steps on this build, every ask at 1.0 token a block; cause not isolated. Lookup alone, 2026-10-16, one run: prose / code / chat 27.6 / 31.7 / 27.7 tok/s, edit / copy 97.9 / 131.2 (10.43 / 14.19 tokens a block), plain 27.4-27.5, every text the plain run's; price file re-measured on this build. Setup Guide entry: [models/nemotron-3.5-bolt-74b-a9b/model.yaml](../models/nemotron-3.5-bolt-74b-a9b/model.yaml).

DGX Spark, 2026-10-15, one fresh server a run, the GPU to itself, max_len 32,768, conductor raw, suffix store off.

| measurement | plain | lookup |
|-|-|-|
| decode step from its graph (`nh_check.py bench`, 256 tokens) | 27.94 tok/s (35.79 ms) | |
| prose / chat / code, 512 greedy tokens, thinking off (`fn_lookup_ab.py`) | 27.47 / 27.45 / 27.44 | 27.53 / 27.31 / 29.34 |
| edit / copy asks | 27.39 / 27.39 | 57.70 / 58.90 (4.52 / 9.46 tokens a block) |
| 1 / 2 / 3 requests at once, 256 tokens each, total (`lfm_parallel.py`) | | 26.94 / 32.57 / 38.30 tok/s, 2 and 3 slots decoding together (254 joint rounds), each text its request's alone |
| verify chain of 1 / 2 / 4 / 8 / 16 rows at 1,024 tokens (`nh_check.py staircase`, the profile's price file) | | 34.8 / 41.5 / 47.2 / 73.2 / 114.3 ms |

- llama.cpp on the same GGUF (master 620d2225c with #30520, same board): `llama-bench -ngl 99 -p 0 -n 128` 35.37 tok/s; llama-server greedy on the 8 prompts below 38.6-39.4 tok/s. The engine's 27.94 is 1.27x below llama-bench.
- Every lookup answer is the plain run's text. Through `fn_server_check.py`, 19 of 20 answers are identical; the thinking-on one is identical up to the reasoning budget's forced close, which lands one token later at the end of the verified block (docs/exactness.md, output changes).
- A verify row costs 2.7 to 14.6 ms more than the one before it (about 4 ms, a step at 5 and 9 rows): the IQ2_XXS shared expert decodes its weights once for up to 16 rows (5,376 x 4,096: 0.10 ms for one row, 0.36 for 8, 0.47 for 16) and the routed experts run one (row, expert) pair a program, so the lookup pays off on edit and copy asks and not on prose.
- A CUDA GEMV for the IQ2_XXS shared experts took the decode step to 32.7 tok/s and 8 / 16-row chains to 57.9 / 93.2 ms; it is not in this tree, so this profile decodes at the rates above, and the LiLiCorr rows of the IQ2 table under [LiLiCorr](#lilicorr), measured with it, are higher than this tree gives.
- Prefill (random tokens, one engine): 246 tok/s at 1,024 tokens, 799 at 8,192, 752 at 32,768, 698 at 131,072, 680 at 262,000 (385 s; 28.9 GiB reserved at the peak, the box never under 82 GiB available). Decode at 32,768 and 131,072 tokens of context: 26.06 and 23.27 tok/s.

### Bolt: nemotron-bolt-exl3-2.0

`nemotron-bolt-exl3-2.0` (`ops/profiles/nemotron-bolt-exl3-2.0.env`): an EXL3 pack converted from NVIDIA's BF16 checkpoint (nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16, revision bca0a8cb) by exllamav3 1.6.2 with a patch for the Puzzle/Bolt layer table: routed experts at 2 bits, the Mamba2, attention, latent and shared-expert projections at 4, head 6 (23.48 GB, the RADIO tower in the pack in BF16 and not loaded); 20.22 GiB at load; 262,144 tokens; 3 slots; NVIDIA's LiLiCorr drafter and the lookup cut by StairCut (prices `ops/prices/nemotron-3.5-bolt-74b-a9b/nemotron-bolt-exl3-2.0.dgx-spark.json`, measured on the rows path). The pack is in a public Hugging Face repository (0xBakeer/TandemLLM-Nemotron-3.5-Bolt-74B-A9B-EXL3-2.0, pinned by commit).

Quality against BF16 (`tools/nh_quality.py`, the BF16 checkpoint's logits on 8 wikitext-2 rows of 1,024 tokens; BF16 perplexity 5.97):

| weights | KL | top-1 | perplexity |
|-|-|-|-|
| `nemotron-bolt-exl3-2.0` | 0.170 | 84.1 % | 6.81 |
| `nemotron-bolt-iq2` (NVIDIA's GGUF) | 0.259 | 79.6 % | 7.48 |

exllamav3's `model_diff` on 100 wikitext-2 rows of 2,048 tokens: the 2.0 pack KL 0.088, top-1 87.6 %. A pack with the dense projections at 2 bits as well measured KL 0.379, top-1 73.9 % in this engine and is not published as a profile.

DGX Spark, 2026-10-15: decode step 44.2 tok/s from its graph on the rows path (41.0 with a CTA a row); with LiLiCorr and the lookup, prose / code / chat / edit / copy 65.8 / 99.0 / 69.3 / 152.8 / 256.0 tok/s, every text the plain run's (table and conditions under [LiLiCorr](#lilicorr)).

### Bolt: nemotron-bolt-exl3-2.5 to 6.0

`nemotron-bolt-exl3-<bpw>` (`ops/profiles/nemotron-bolt-exl3-<bpw>.env`): EXL3 packs of the same BF16 checkpoint: below 4.0 the routed experts at the level and every other projection at 4 bits, from 4.0 up every projection at the level; head 6 (8 at 6.0); each in its public repository 0xBakeer/TandemLLM-Nemotron-3.5-Bolt-74B-A9B-EXL3-<bpw>, pinned by commit; 262,144 tokens; 3 slots; LiLiCorr and the lookup cut by StairCut on the pack's price file (`ops/prices/nemotron-3.5-bolt-74b-a9b/`, SPD-127).

Quality as above (`tools/nh_quality.py`, 8 wikitext-2 rows of 1,024 tokens against the BF16 logits); decode step from its graph with one CTA a row (`dense`, measured before the rows path; the 2.0 pack: 41.3 there, 44.2 on the rows path); DGX Spark, 2026-10-15:

| profile | download | at load | KL | top-1 | perplexity | decode step | plain, server |
|-|-|-|-|-|-|-|-|
| `nemotron-bolt-exl3-2.0` | 23.48 GB | 20.22 GiB | 0.170 | 84.1 % | 6.81 | 41.3 tok/s | 43.3 tok/s |
| `nemotron-bolt-exl3-2.5` | 27.61 GB | 24.07 GiB | 0.094 | 87.3 % | 6.34 | 39.9 tok/s | 42.1 tok/s |
| `nemotron-bolt-exl3-3.0` | 31.75 GB | 27.92 GiB | 0.061 | 89.7 % | 6.21 | 39.6 tok/s | 41.7 tok/s |
| `nemotron-bolt-exl3-3.5` | 35.88 GB | 32.01 GiB | 0.044 | 91.1 % | 6.12 | 38.7 tok/s | 40.6 tok/s |
| `nemotron-bolt-exl3-4.0` | 40.01 GB | 36.18 GiB | 0.034 | 92.7 % | 6.10 | 38.4 tok/s | 40.5 tok/s |
| `nemotron-bolt-exl3-5.0` | 49.13 GB | 44.42 GiB | 0.016 | 94.8 % | 6.00 | 32.0 tok/s | 34.1 tok/s |
| `nemotron-bolt-exl3-6.0` | 58.37 GB | 53.00 GiB | 0.0087 | 96.2 % | 5.98 | 27.8 tok/s | 29.2 tok/s |
| `nemotron-bolt-iq2` (NVIDIA's GGUF) | 22.04 GB | 22.19 GiB | 0.259 | 79.6 % | 7.48 | 27.9 tok/s | 27.8 tok/s |

Each pack passed the load-time self-check and a chat on the Spark. The plain column is 0.10.0's (ENG-345, below); the drafted rows of every pack follow.

Plain, server (ENG-345, 0.10.0, 2026-10-16): one fresh server a run on the Spark with the GPU to itself, `tools/fn_lookup_ab.py` (prose, chat, code, edit and copy asks, 512 greedy tokens, thinking off, max_len 32,768, conductor raw, the suffix store off), `--drafter none`, one slot; the mean over the five asks, then the mean of 3 runs (spread 0.1 to 0.5 tok/s).

Drafted against plain (SPD-127, 0.11.1, 2026-10-16): the same asks, one fresh server a run, 3 slots, the session and prefix caches off; plain `--drafter none`, drafted the profile's own drafting (EXL3: LiLiCorr and the lookup, `--draft-head lilicorr --depth 3`; IQ2: the lookup alone) with the release's price files; 3 runs of each, alternating; means in tok/s:

| profile | plain | prose | chat | code | edit | copy | tokens a block, prose / chat / code |
|-|-|-|-|-|-|-|-|
| `nemotron-bolt-exl3-2.0` | 43.2 | 65.6 | 71.1 | 102.1 | 163.5 | 247.2 | 3.0 / 3.09 / 4.88 |
| `nemotron-bolt-exl3-2.5` | 41.9 | 65.8 | 68.9 | 114.7 | 161.9 | 208.7 | 3.23 / 3.37 / 5.68 |
| `nemotron-bolt-exl3-3.0` | 41.6 | 58.1 | 69.3 | 100.8 | 149.6 | 203.7 | 2.83 / 3.27 / 5.06 |
| `nemotron-bolt-exl3-3.5` | 40.6 | 59.5 | 65.9 | 101.0 | 129.7 | 217.4 | 2.97 / 3.25 / 5.4 |
| `nemotron-bolt-exl3-4.0` | 40.4 | 52.0 | 62.0 | 97.9 | 137.4 | 210.2 | 2.49 / 3.01 / 5.43 |
| `nemotron-bolt-exl3-5.0` | 33.9 | 51.7 | 55.5 | 81.7 | 121.3 | 184.4 | 2.98 / 3.14 / 4.91 |
| `nemotron-bolt-exl3-6.0` | 29.1 | 40.1 | 50.1 | 76.3 | 110.8 | 157.3 | 2.57 / 3.05 / 5.32 |
| `nemotron-bolt-iq2` | 27.7 | 27.7 | 27.8 | 31.9 | 97.8 | 131.6 | 1.02 / 1.03 / 1.22 |

Every drafted text is the plain run's. Each EXL3 price file comes from one load on the Spark: `nh_check.py bench` (256 tokens), `staircase` (1,024 and 8,192 tokens, 20 replays a count) and `lilicorr` (8 prompts of 256 tokens, thinking on), then `nh_check.py prices --lilicorr`: the chain of 1 to 16 rows and the plain step per context class, and the `spec` prior, (8-row chain + draft) / (0.75 x accepted length), 13.9 ms a token on 2.5 to 20.2 on 6.0.

Without a price file (0.11.0, 2.5 to 6.0) the one-row block kept a price of 1 ms: a round the cut cuts to the anchor alone prices `spec`, never the chain. Once the 2- to 8-row blocks were priced at 33 to 44 ms, the cut took the anchor alone wherever the draft was unsure, after the draft (about 10 ms) had run. On 4.0, 247 of 277 prose rounds were such rounds, 35.6 ms a token against the 24.6 ms plain step, and 3.0 to 4.0 wrote prose at 31 to 33 tok/s and chat at 39 to 40. A price file without the `spec` prior (0.10.0, 4.0) kept every ask on plain steps, at 40.3 to 40.5 tok/s. The IQ2 profile drafts with the lookup alone: prose and chat run at its plain rate (27.7 and 27.8 tok/s against 27.7), code at 31.9; LiLiCorr beside it with a `spec` prior is not measured yet. llama.cpp (620d2225c, `--spec-type draft-dflash`, 7 draft tokens, one slot) decodes the GGUF plain at 38.2 tok/s and with LiLiCorr at 49.3 / 62.0 / 92.2 / 92.5 / 132.0 (0.10.0's windows); its LiLiCorr texts differ from its plain texts on prose and chat.

### Super VL: nemotron-super-vl-exl3-4.0, nemotron-super-vl-exl3-2.0, nemotron-super-vl-nvfp4

Nemotron 3.5 Super VL 120B-A12B with its RADIO image tower (`VISION=on`, [Super VL: images](#super-vl-images)), each profile with NVIDIA's LiLiCorr drafter for Super VL (nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4-LiLiCorr: five taps, layers 1, 22, 43, 64 and 85, a window of 8,192, its own NVFP4 head) and the lookup, cut by StairCut (price files `ops/prices/nemotron-3.5-super-vl-120b-a12b/`, each with a `spec` prior: the 8-row chain plus the draft over 0.75 of LiLiCorr's offline accepted length).

- `nemotron-super-vl-exl3-4.0`: an EXL3 pack of the BF16 checkpoint (nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16, revision 97e00975) by exllamav3 1.6.2 with the Puzzle/Bolt and Super VL patches: every projection at 4 bits, head 6, the MTP head at 4 (65.0 GB); 58.35 GiB at load (59.75 with the MTP head). Public repository 0xBakeer/TandemLLM-Nemotron-3.5-Super-VL-120B-A12B-EXL3-4.0, pinned by commit.
- `nemotron-super-vl-exl3-2.0`: the same converter at 2 bits with `-hq`: routed experts at 2, every other projection at 4, head 6, the MTP head at 4 to 6 (36.84 GB); 31.24 GiB at load. Public repository 0xBakeer/TandemLLM-Nemotron-3.5-Super-VL-120B-A12B-EXL3-2.0, pinned by commit.
- `nemotron-super-vl-nvfp4`: NVIDIA's ModelOpt checkpoint (revision 8a115b9d, 75.1 GB) read as stored ([Weights](#weights)); 66.63 GiB at load without the MTP head; 131,072 tokens.

Quality against BF16 (`tools/nh_quality.py`, the BF16 checkpoint's logits on 8 wikitext-2 rows of 1,024 tokens from an HF job; BF16 perplexity 3.840) and speed, DGX Spark, 2026-10-16, one fresh server a run, the GPU to itself, conductor raw, suffix store off, 512 greedy tokens, thinking off (client-side streaming rate, mean of 3 alternating runs):

| profile | at load | KL | top-1 | perplexity | plain | LiLiCorr + lookup: prose / chat / code / edit / copy |
|-|-|-|-|-|-|-|
| `nemotron-super-vl-exl3-4.0` | 58.35 GiB | 0.049 | 92.9 % | 3.94 | 31.1 | 33.1 / 43.9 / 53.6 / 101.0 / 108.3 |
| `nemotron-super-vl-exl3-2.0` | 31.24 GiB | 0.325 | 82.1 % | 4.93 | 36.5 | 35.6 / 46.2 / 46.3 / 125.1 / 139.2 |
| `nemotron-super-vl-nvfp4` | 66.63 GiB | 0.041 | 93.4 % | 3.83 | 19.6 | 23.9 / 28.7 / 39.9 / 73.5 / 81.7 |
| NVIDIA's NVFP4 in vLLM (nightly-3f1bcfd8, the card's Spark flags) | | | | | 19.4 | 29.9 / 34.8 / 52.7 / 72.4 / 72.7 (LiLiCorr, 7 tokens) |
| the same, the checkpoint's MTP head (5 tokens) | | | | | | 29.1 / 32.3 / 43.2 / 52.9 / 53.1 |

- exllamav3's `model_diff` on 100 wikitext-2 rows of 2,048 tokens, the 4.0 pack: KL 0.0224, top-1 94.3 %; on the 8 rows above KL 0.049, top-1 93.0 %, perplexity 3.948 (the engine 0.0486, 92.9 %, 3.941).
- A decode token reads 10.11 GB of the NVFP4 checkpoint: the FP8 Mamba2 projections 4.38 GB, routed experts 2.73 (22 a layer), shared experts 0.99, the BF16 latent pair 0.67, the FP8 head 0.54, the SSM states 0.34; 19.6 tok/s is 198 GB/s.
- Verify chains of 1 / 2 / 4 / 8 / 16 rows at 1k context: EXL3 4.0 31.3 / 39.8 / 53.3 / 74.5 / 105.8 ms, NVFP4 51.7 / 60.0 / 78.4 / 109.0 / 156.7 ms. A row reads 22 experts in each of the 40 MoE layers (880); 8 rows read about 4,170, so a verify block costs more a row than on Bolt (9 experts a row).
- LiLiCorr offline (`nh_check.py lilicorr`, 4 prompts x 192 tokens, thinking on): accepted length 4.82 on NVFP4 (2.86 .. 6.16), 4.65 on EXL3 4.0; the draft takes 21.3 ms (its NVFP4 head over 131,072 tokens and the 8,192-key window).
- The checkpoint's MTP head (`nh_check.py mtp`, depth 3, the 4.0 pack's 4-bit head): the draft equals the trunk's greedy token at depth 1, 2 and 3 in 90.6, 82.2 and 69.4 % of positions, mean accepted run 2.38 of 3. Not served: it runs eagerly and LiLiCorr drafts 7 tokens at a higher accepted length.
- Every LiLiCorr text of the five asks is the plain run's. `fn_server_check.py`, 20 requests: 16 the same on NVFP4 and on EXL3 4.0, 17 on EXL3 2.0; the others (thinking on with a low cap, JSON with a reasoning budget) the same up to the reasoning budget's forced close, which lands at the end of the verified block (docs/exactness.md, output changes).
- 1 / 2 / 3 requests at once with LiLiCorr + lookup (256 tokens, the server's `parallel.running` sampled every 50 ms: 1, 2 and 3): EXL3 4.0 20.8 / 18.5 / 19.6 tok/s in total, EXL3 2.0 22.5 / 19.4 / 20.7, NVFP4 16.7 / 15.9 / 17.6. Speculating slots take turns; with 22 experts a row the turns gain nothing over one slot. vLLM batches the same three requests plain: 18.9 / 31.1 / 35.9.
- Prefill (one cold prompt): EXL3 4.0 564 / 577 / 572 tok/s at 7k / 30k / 115k tokens, NVFP4 476 / 490 / 502; vLLM on the NVFP4 checkpoint 2,318 / 2,521 / 2,158 (its grouped MoE GEMMs; the engine runs one GEMM pair an expert, 512 a layer).
- The 2.0 pack: KL 0.325 is twice the loss of Bolt's 2.0 with the same recipe (0.170); decode 36.8 tok/s.
- Window (ENG-363): up to 1,048,576 tokens, the checkpoint's own (`llm_config.max_position_embeddings`, the tokenizer's `model_max_length`), for slot 0 and the side slots; the default stays 262,144. Attention has no positional encoding, and nothing in the engine is sized below that: K and V are [2, window, 128] a layer (8 GiB at 1,048,576), attention runs 4 to 2,048 programs of 512 keys, prompts run in 4,096-row chunks. Checked on EXL3 4.0 with slot 0 at 1,048,576, three side slots at 262,144, LiLiCorr and the lookup, the caches and conductor as served, the response cache off, a request raw (2026-10-16): five facts at 5 / 25 / 50 / 75 / 95 % of synthetic filler, greedy, thinking off, all five answered at 131,054 prompt tokens (first token after 228 s, 574 tok/s of prefill) and at 249,983 (475 s, 526 tok/s). A 1,000,000-token prompt was not measured.
- The prefill's workspace does not grow with the prompt: 3.34 GiB above the engine at 131,054 tokens, 3.18 at 249,983 (the memory policy's `observed`). The adapter says so (`cost()`: a floor of 3.5 GiB, capped at 4.0); before, the 27B's line (30 kB a token) refused a 1,000,000-token prefill as 29 GiB of workspace. A slot 0 window above 262,144 adds its KV above it to the memory policy's ceiling and peak (+6 GiB at 1,048,576), as the side slots' bytes are added, so the caches are not refused for it during a long prefill.
- Each side slot at 262,144 takes 2.82 GiB allocated and 3.10 GiB of MemAvailable at load (its KV 2.00 GiB), built in 9.2 s; measured with 3 and with 4 slots. With slot 0 at 1,048,576 and 4 slots the residency planner refused Laya (its 3 GiB claim under the 15 GiB floor against the engine's peak); with 3 slots it fits.

### Super VL: community EXL3 packs

ENG-377. A community pack on Super VL ([community-weights.md](community-weights.md#bases)) is served as `community/<id>`: the nearest own EXL3 profile (`nemotron-super-vl-exl3-4.0` for 3.5 bpw) with the pack in place of its weights, so the same limits, sampler, LiLiCorr drafter and lookup; price file `ops/prices/nemotron-3.5-super-vl-120b-a12b/community/<id>.dgx-spark.json`.

The first one, aitamedia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-EXL3-3.5bpw at 358667cf (57.8 GB), differs from our packs in layout:

| | community 3.5 pack | our 4.0 pack |
|-|-|-|
| config | flat `NemotronHForCausalLM` (remapped from the Omni wrapper, which is kept as `config.json.orig-omni`) | the Omni wrapper, the trunk in `llm_config` |
| keys | `backbone.*`, `lm_head.*`, `mtp.*` | `language_model.` before each |
| tower | none (text only) | RADIO tower and projector in BF16 |
| head | fp16 `lm_head.weight`, with `tie_word_embeddings: true` in the config | EXL3 at 6 bits, untied |
| bits | routed experts 3.5, every other projection 5; MTP experts 4, MTP projections 6, `eh_proj` 5 | 4 everywhere, MTP 4 |
| converter | exllamav3 1.6.0, mul1 codebook, `out_scales` auto | exllamav3 1.6.2, mul1 codebook |

The reader takes it as stored: the prefixes map to the same canonical names, every trellis rate is one the kernels decode (3.5 / 4 / 5 / 6), the channel factors are `suh` / `svh`. The original Super VL head is untied, and the flat config's tie flag is the remap's: a stored head is read and the tie dropped (`PackIndex.stored_head`, an audit note); without a stored head the embedding stays the head. Anything else the layer table does not expect stops the load by name.

Checked on the DGX Spark, 2026-10-20, on the 0.12.1 code before its version bump (`measurements/0.12.1-summary.json` has the runs):

- Against the converter's own forward (exllamav3 1.6.0 on the same pack, `flash_attn_nc`, teacher-forced on the engine's greedy tokens; `nh_check.py engine`, 8 prompts x 64 tokens, thinking on): top-1 510 of 512 (99.6 %); both differences are near ties (the reference's margin 0.22 and 0.00 logits); mean |diff| 0.23 logits (0.0047 of the row's largest), mean |diff| over the reference's top 10 0.27, KL 0.0019 nats; max |diff| 6.1 on a token of rank 21,810. Every position with a margin of 0.25 logits or more agrees (494 of 494).
- The MTP head (`nh_check.py mtp`, depth 3, 4 prompts x 128 tokens): equal to the trunk's greedy token at depth 1 / 2 / 3 in 92.6 / 84.4 / 73.6 % of positions, mean accepted run 2.45 of 3.
- Greedy answers in English and German, code, a two-turn chat and one tool call (`get_weather` with its arguments) through `/v1/chat/completions` are correct.
- One fresh server a run, the GPU to itself, conductor raw, suffix store off, `--max-len 32768`, 512 greedy tokens, thinking off, one request at a time (n = 1): plain (`--drafter none`, 1 slot) 25.9 tok/s; LiLiCorr + lookup (`--drafter mtp --draft-head lilicorr --parallel 3`) prose 33.2, code 53.9, chat 37.4, edit 99.0, copy 112.3 tok/s (2.97 / 5.38 / 3.41 / 15.5 / 16.0 tokens a block), every text the plain run's. Prefill of one cold 7,507-token prompt 529 tok/s.
- Load: weights 53.58 GiB allocated (55.3 GiB of MemAvailable), 47 to 58 s; the served profile (`ops/use-profile.sh community/nemotron-super-vl-exl3-3.5bpw`: 262,144 tokens, 3 slots, LiLiCorr) up and smoked in 110 s, the process 61.7 GiB on the GPU, MemAvailable 118 -> 52 GiB.
- Plain decode 25.9 tok/s against the 4.0 pack's 31.1. What a decode token reads, from the headers: 7.95 GB here, 6.74 GB for the 4.0 pack (here the 5-bit dense projections 4.23 GB, the fp16 head 1.07 GB, the 3.5-bit experts 2.13 GB). The decode step's profile here: 5-bit dense rows 20.7 ms, the head and the routers 5.8 ms, the experts 10.2 ms, about 205 GB/s each; the 4.0 pack's profile was not measured in this run. Verify chains of 1 / 2 / 4 / 8 / 16 rows at 1k context: 38.2 / 45.6 / 58.7 / 77.1 / 106.2 ms (4.0: 31.3 / 39.8 / 53.3 / 74.5 / 105.8). LiLiCorr offline (`nh_check.py lilicorr`, 8 prompts x 256 tokens, thinking on): accepted length 4.79, the draft 21.9 ms.

### Super VL: images

ENG-365. Every Super VL checkpoint carries the Omni model's image tower in BF16: the EXL3 packs copy it unquantised (shard 9 of the 4.0 pack, shard 5 of the 2.0), NVIDIA's ModelOpt checkpoint keeps it BF16 (shard 1; its 392 tensors are byte for byte the 4.0 pack's). `engine/models/nemotron_h/vision.py` loads it beside the trunk (`ctx.vision`, the server's image path of the Qwen towers) and runs the checkpoint's own arithmetic: transformers' `RadioModel` and `NemotronH_Omni_Reasoning_V3VisionProjector`, bf16, the reference's ops in its order.

| part | Super VL |
|-|-|
| tower | C-RADIO ViT-H/16: 32 blocks, hidden 1,280, 16 heads of 80, MLP 5,120 (exact gelu), qkv with bias, no layer scale; 10 prefix tokens (3 class, 7 registers) dropped after the last block |
| positions | a learned 128 x 128 table (2,048-pixel images), bilinear in fp32 to the grid's longer side squared, then cropped (RADIO's cropped position embedding); past a 247-patch side the same without the square, to fp32 rounding |
| projector | `vision_final_layernorm` (the trunk has an MTP head), the v2 pixel shuffle of 2 x 2 patches into one 5,120-wide row, `mlp1`: RMSNorm (fp32, eps 1e-5), 5,120 -> 20,480, relu^2, 20,480 -> 4,096 |
| processor | one tile an image: the 16-pixel grid of `image_processing_nemotron_h_omni.py` (`_compute_target_patches`: rounded up, scaled into 1,024..13,312 patches, each side even), torch's antialiased bicubic in fp32, CLIP mean and std; no remote code |
| prompt | the template writes `<image>` (18) for an image part, in a user or a tool message; the engine writes `<img>` (19), one `<image>` a row, `</img>` (20): 256 to 3,328 rows, 1,000 for 1,280 x 800 |
| rows | written over the placeholders' embeddings in the prompt chunk that holds them; no positional encoding, so no position moves and every text row is computed as without the image |
| memory | 1.57 GB of tensors, 1.60 GiB allocated beside the EXL3 4.0 trunk; the image cache 512 MB (`--image-cache-mb`); the admission bound for the largest image 1.6 GiB of activations |

The caches key an image's rows on its content (`key_ids` from the digest of the patches), as on the Qwen models: the same image resumes, another one under the same placeholder does not. A request with images runs on slot 0: the tower builds its own context, which the side slots do not read yet (the 27B's images take a side slot since SRV-629). Video, the sound tower and RADIO's summary output are not served. Bolt's EXL3 packs hold the same kind of tower and stay text only (not checked).

Checked against the checkpoint's own processor and modeling code (`tools/nh_vision_check.py`, DGX Spark, 2026-10-17, three synthetic pictures: shapes 640 x 480, a scene 1,024 x 768, a dark app screen with text 1,280 x 800):

| picture | rows | pixels | ours vs reference bf16 | ours vs reference fp32: cosine (lowest row), relative | reference bf16 vs its fp32 |
|-|-|-|-|-|-|
| shapes | 300 | identical | max abs 0 | 0.99990 (0.99668), 0.0138 | 0.0138 |
| scene | 768 | identical | max abs 0 | 0.99973 (0.98957), 0.0234 | 0.0234 |
| app | 1,000 | identical | max abs 0 | 0.99985 (0.99628), 0.0171 | 0.0171 |

The prompt ids equal the checkpoint processor's `input_ids` for each. On the engine (EXL3 4.0 trunk, "Describe this image.", thinking off, 96 greedy tokens), the answers with our rows and with the reference's rows are identical; they name the three shapes with their colours and places, the sky, sun, grass and red-roofed house, and the app's "Garden settings" header and orange START button. The trunk's own distance to BF16 is the pack's (KL 0.049 above); a BF16 reference of the whole model does not fit the board. `tests/test_nh_vision.py`: the processor's grids, the position table, the shuffle, the splice across prompt chunks, and the checkpoint's modeling code built small.

### Lightning: nemotron-lightning-bf16

`nemotron-lightning-bf16` (`ops/profiles/nemotron-lightning-bf16.env`): the BF16 checkpoint at revision a9904d24, 65.8 GB, 59.04 GiB at load without the MTP head; 262,144 tokens; 3 slots; the lookup cut by StairCut. The checkpoint's MTP head joins the lookup with `QSE_NH_HEAD=mtp` (off in the profile, see [Speculation](#speculation)). Setup Guide entry: [models/nemotron-3.5-lightning-30b-a3b/model.yaml](../models/nemotron-3.5-lightning-30b-a3b/model.yaml), a test model (order 90, not recommended). Raw data: `models/nemotron-3.5-lightning-30b-a3b/measurements/0.9.1-summary.json`.

DGX Spark, 2026-10-15, one fresh server a run, the GPU to itself, max_len 32,768, conductor raw, suffix store off.

| measurement | plain | lookup | MTP head (depth 3) + lookup |
|-|-|-|-|
| decode step from its graph (`nh_check.py bench`, 256 tokens) | 33.63 tok/s (29.73 ms) | | |
| prose / chat / code, 512 greedy tokens, thinking off (`fn_lookup_ab.py`) | 34.14 / 33.77 / 33.97 | 33.08 / 34.13 / 36.21 | 34.17 / 34.16 / 34.15 (no drafted round: the cut chose the plain step every turn) |
| edit / copy asks | 33.85 / 33.74 | 105.77 / 140.74 (9.83 / 14.19 tokens a block) | 34.01 / 34.06 |
| `fn_server_check.py` set (thinking on, tools, JSON), per request | 32.3-33.9 tok/s | 1.0-1.28 tokens a block | 34.2-53.3 tok/s, 1.0-4.3 tokens a block |
| 1 / 2 / 3 requests at once, 256 tokens each, total (`lfm_parallel.py`, lookup drafter) | | 33.73 / 44.73 / 50.65 tok/s, 2 and 3 slots decoding together (254 joint rounds), each text its request's alone | |

- Same weights in llama.cpp (5ecbe1ac1, `llama-bench -ngl 99 -p 0 -n 256`, the BF16 GGUF of ggml-org at 8a08a1c8, same board): 33.31 tok/s against the engine's 33.63.
- Every answer of the lookup and the MTP runs is the plain run's text. Through `fn_server_check.py`, 15 of 20 MTP answers equal the plain ones; the other 5 differ only from the reasoning budget's close on, which lands at the end of the verified block (2 to 3 tokens later than the plain step's), one of the output changes docs/exactness.md lists.
- The MTP head runs eagerly (no graph) and reads the 0.70 GB head once a depth; with thinking off StairCut priced its rounds above the plain step and took plain steps. Not the profile's default until its rounds are captured and priced.
- Prefill (`nh_check.py staircase`, one expert GEMM pair an expert): 545 tok/s at 1,024 tokens, 2,537 at 8,192. Two batched GEMMs padded to the busiest expert measured 359 and 1,248 and were dropped.

## Arithmetic

The reference forward (`ref.py`) follows transformers' NemotronH and the CUDA path the checkpoints were run with (mamba_ssm's `selective_state_update`, `causal_conv1d_update`, `rmsnorm_fn`):

- RMS norm in fp32 times the plain weight; the gated norm `norm_before_gate=False`: y silu(z) in fp32, RMS over groups of 512 (inner / 8), times the weight, one rounding.
- dt = softplus(dt + dt_bias), no clamp (`time_step_limit` [0, inf]; transformers' torch path clamps at `time_step_min`, its CUDA path does not).
- The SSM state is fp32 (`mamba_ssm_cache_dtype`), y = S C + D x rounded to bf16.
- Router: fp32 logits, sigmoid, the bias for the choice only, ties to the lower id.
- Experts: bf16(relu(bf16(up x))^2), down; the weighted sum in fp32 in ascending expert id, one rounding.

`tests/test_nemotron_h_ref.py` holds it to transformers in fp32 within 2e-5 on three tiny models (Lightning-like, latent, Puzzle per-layer). transformers' multi-chunk torch SSD disagrees with its own one-chunk result by 3e-2 on these weights; the references run one chunk a sequence.

## How the engine runs it

- Row path: every projection on 16-row tiles, the Mamba2 conv over the row's window, the SSM update walked from the committed state along the row's path in fp32, the gated norm, attention over fixed key chunks, the router, relu^2 experts and the ordered combine: a row of a verify block is the decode step's row bit for bit. A verify block walks the SSM once a head (`_ssm_walk`): a row whose parent is the row before it goes on from that row's state, a branch walks its path from the committed state again, so the 4 MB state of a Bolt layer is read once a block, not once a row.
- Verify writes nothing but its KV rows. A commit walks each SSM state along the kept path with the decode step's own kernel (the same products in the same order) and takes the conv state from the path's last three inputs: the committed state is the decode steps' state bit for bit.
- Prefill: 4,096-token chunks, the chunked SSD form in fp32 matmuls (chunk 128), SDPA, one GEMM pair an expert. The next token agrees with the row path; the bits do not.
- CUDA graphs: the decode step per attention class, verify blocks, commits, joint decode steps of the side slots.
- Taps (`set_taps`, a block drafter's context): the residual stream entering the tapped layers (llama.cpp's `t_layer_inp`) of every row, copied inside the graphs; a committed row's taps go to the state's ring by position (window + 64 rows): a decode step's in the step, a verify block's kept path's in the commit, a prompt chunk's last rows in the chunk. A restore or a restart moves `tap_gen`, so a drafter drops what it read of the rows below the length.
- EXL3 projections (a pack's Mamba2, attention, latent, shared-expert and head blocks): `tools/fn_kernels.dense_rows` for the decode step and verify blocks up to 32 rows. Each row's input is rotated once (`dense_rot_kernel`, the arithmetic `dense` fuses), then one CTA a 128-column block, group of 16 rows and cluster rank decodes the trellis tiles once for the group: rows 0..7 in the m16n8k16 tile's a0/a2 lane groups, rows 8..15 in a1/a3. A row's K slices, warp and rank order, output rotation and svh are the one-row `dense` call's, so every output byte is its byte; the ranks' partials are added by every warp of the cluster, a row a warp. `QSE_NH_DENSE_ROWS=off` runs `dense`, a CTA a row.
- EXL3 experts (`tools/exl3_relu2.Relu2Bank`): a verify block's (row, slot) pairs in ascending expert id, cut into groups of up to 8 pairs of one expert; a group's CTA decodes the expert's tiles once (row r in lane group r) with each pair's lone-call arithmetic. `QSE_NH_BANK_GROUP=off`: a CTA a pair.
- Load-time self-check: a 24-token decode replayed as one verify block, every row and the committed states compared bit for bit (with taps, each row's taps and the committed ring too); a difference turns speculation off.
- Caches: sections `mamba.conv.<L>` (bf16 [3, conv_dim]) and `mamba.S.<L>` (fp32 [heads, 64, state]), views K and V. Lightning: 46.8 MiB of sections a sequence, 6,144 bytes of KV a token.

## Weights

| source | reader | notes |
|-|-|-|
| BF16 safetensors | `pack.py`, `load.py` | `backbone.` and an Omni wrapper's `language_model.` stripped, tower tensors skipped; experts stacked; page cache dropped while loading |
| ModelOpt NVFP4 / FP8 (Super VL's NVFP4 release, `hf_quant_config.json` MIXED_PRECISION) | `pack.py`, `load.py`, `tools/nvfp4_relu2.py` | NVFP4 (`weight` uint8 [N, K / 2], e4m3 `weight_scale` [N, K / 16], fp32 `weight_scale_2`): the routed experts an `NVFP4Bank` (one program a (row, expert) pair and 32 outputs, fp32 products, the tensor scale on the sum, the BF16 experts' roundings; row-invariant), the shared experts on `tools/nvfp4_skinny` (a down projection's input scaled by a power of two into fp16's range first, `NVFP4Wide`); FP8 (`weight` e4m3 [N, K], fp32 `weight_scale`): `FP8Linear`, the codes widened to bf16 in a 16-row tensor-core product, the scale on the fp32 sum (fp32 logits for the head). `input_scale`, `k_scale` and `v_scale` are read past: activations and the KV cache stay bf16 |
| GGUF (`nemotron_h_moe`) | `gguf.py`, `gguf_load.py` | config from the metadata (per-layer expert counts from the router's rows), checked against the checkpoint's config.json when there is one; IQ2_XXS, NVFP4 and Q8_0 matrices and the expert banks kept packed (the embedding dequantised to bf16 for its row gather); other types refused by name |

The GGUF types Bolt holds, each with a twin equal to gguf-py's dequant bit for bit (`tests/test_iq2_kernels.py`; NVFP4 needs gguf-py from llama.cpp after GGML_TYPE_NVFP4):

| type | where in Bolt | format | kernel (rows 1..32 row-invariant) | prompt chunk |
|-|-|-|-|-|
| IQ2_XXS (16) | routed and shared experts | 256 weights in 66 bytes, w = d (0.5 + s) / 4 x grid x sign | dense: 8 outputs and 4 groups a step, decoded once for up to 16 rows of a program (`IQ2Linear`); bank: one (row, expert) pair a program (`IQ2Bank`) | decoded to bf16 on `_iq2_dequant`, cuBLAS |
| NVFP4 (40) | every Mamba2 in_proj and out_proj | 64 weights in 36 bytes: 4 UE4M3 scales (one a 16) and E2M1 nibbles, w = kvalue x ue4m3; a per-tensor fp32 scale (`blk.N.ssm_in.scale`, `ssm_out.scale`) multiplies the matmul's output, as llama.cpp's build_lora_mm does | repacked at load into the checkpoints' NVFP4 layout without changing a value (kvalues = 2 E2M1, ue4m3 = E4M3 / 2) and run on `tools/nvfp4_skinny.py`, the tensor scale on the fp32 sum (tiles in `ops/skinny-tiles.json`) | `nvfp4_matmul` |
| Q8_0 (8) | head, attention, latent projections, embedding | 32 int8 and an fp16 scale | 16-row tiles, one bf16 tensor-core product a block of 32 (int8 is exact in bf16) times its scale: the same time for 1..32 rows (`Q8Linear`) | the twin in bf16, cuBLAS |

IQ2_XXS needs K a multiple of 256: Lightning's 2,688 and 1,856 are not, so its community IQ2 GGUFs hold IQ4_NL / Q5_0 experts. Bolt's widths 1,536..2,560, latent 1,024 and shared 5,376 are multiples.

## Speculation

- The lookup (the request's text, the corpus, the suffix store) cut by StairCut through the shared tree drafter, from the staircase's prices (`ops/prices/nemotron-3.5-bolt-74b-a9b/`, `ops/prices/nemotron-3.5-lightning-30b-a3b/`).
- Bolt ships no MTP head (its config lists no MTP blocks and the BF16 index no `mtp.*` tensors). NVIDIA's drafter for it is LiLiCorr, served beside the lookup ([LiLiCorr](#lilicorr)).
- The checkpoint's MTP head (`mtp.py`, `--draft-head mtp`): one native depth, deeper drafts iterate the head on its own final normed row and the drafted token; its attention KV is a cache view. On Lightning BF16 (`nh_check.py mtp`, 8 chat prompts, 1,000 positions, depth 3): the head's draft equals the trunk's greedy token at depth 1, 2 and 3 in 91.9, 80.6 and 69.5 % of positions; mean accepted run 2.38 of 3.
- Verify cost (chain, `nh_check.py staircase`, 1k context): 1, 2, 4, 8 and 16 rows take 29.5, 37.0, 51.6, 74.8 and 103.4 ms.

### LiLiCorr

NVIDIA's drafter for Bolt (nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr, revision 4c6a5613, 0.96 GB, ModelOpt NVFP4 W4A16, arXiv 2608.20530), `engine/drafters/lilicorr.py`, ported from vLLM's `LiLiCorrForCausalLM` and llama.cpp's `src/models/dflash.cpp`:

- context: for every committed token Bolt's residual rows entering layers 1, 18, 35, 51, 68 and 85 (the engine's taps), concatenated, `fc` (NVFP4) and `hidden_norm` = g; each of the five layers' K (k_norm, RoPE) and V of g at the token's position, kept in a window ring of 1,024 positions; computed for the rows a draft has not read yet, from the engine's tap ring
- block: [anchor, mask 990 x 7] at positions n .. n + 7 on Bolt's embedding; per layer RMSNorm, the grouped dynamic convolution's input side (kernel 2, group 16), q/k/v with q_norm and k_norm, RoPE (theta 10,000), attention of each block row over the context keys within its window (query - key < 1,024) and the whole block with a sink logit a head in the softmax, o_proj, the convolution's output side; the same around the SiLU MLP (12,288, NVFP4); the final norm; Bolt's head on rows 1 .. 7
- lattice: the top 8 candidates a slot with their features (log p, p, log p - log p_top, rank / 7, rank 0) through a LayerNorm and an MLP, token, pass, slot and rank states, two transformer layers over the 56 nodes with a learned bias by slot distance, then factor vectors: score(slot 1) = 8 cos(anchor, in), score(slot s | slot s - 1) = 8 cos(out, in)
- walk: greedy, left to right, on the GPU: 7 tokens, each with its row's softmax as its confidence (the decile calibration of StairCut: AUC 0.89 against acceptance, 0.86 for the drafter's own token probability)
- the draft is one CUDA graph (11.2 ms on Bolt IQ2: Bolt's Q8_0 head 4.3, the NVFP4 MLPs 2.6, the bf16 attention and convolution projections 2.3); the context's rows run eagerly
- the NVFP4 kernels multiply in fp16, and the draft's MLP inputs pass 65,504 (the convolution reaches +-800 before the gate and up products): a row past 2^15 is scaled by a power of two around the product (`_lin_wide`, exact)
- a draft head of the shared tree drafter beside the lookup, cut by StairCut; the lookup beside it matches 6 tokens at least (`LOOK_MIN_ORDER_HEAD`; at 3 its short matches cost chat 7 % and code 4 %); `QSE_NH_LOOKUP=off` drafts with the head alone
- load-time gate: drafts against the trunk's greedy decode of the self-check's prompt; a mean run under 1.0, missing files or a load error leave the lookup drafting alone, with the reason in the log and in `/health` (`weights.lilicorr`)

Offline, a draft at every position of the trunk's greedy text (`nh_check.py lilicorr`, 8 chat prompts x 256 tokens, thinking on): accepted length 4.55 (3.43 .. 6.49), against llama.cpp master 620d2225c with the LiLiCorr GGUF on the same prompts 4.57 (3.45 .. 6.38) and NVIDIA's 4.68 (greedy, SPEED-Bench); the draft agrees with the trunk at depth 1 .. 7 in 89.3, 80.0, 70.3, 60.4, 53.0, 45.5, 39.5 % of positions.

DGX Spark, 2026-10-15, `nemotron-bolt-iq2` on a build with the CUDA GEMV for the IQ2_XXS shared experts (not in this tree, see [the IQ2 profile](#bolt-nemotron-bolt-iq2)), one fresh server a run, the GPU to itself, max_len 32,768, conductor raw, suffix store off, 512 greedy tokens, thinking off (`fn_lookup_ab.py`; tok/s, tokens a block):

| class | plain | lookup | LiLiCorr | LiLiCorr + lookup | llama.cpp plain | llama.cpp + LiLiCorr |
|-|-|-|-|-|-|-|
| prose | 32.11 | 32.45 | 40.99 (2.53) | 40.47 (2.51) | 38.49 | 47.93 |
| code | 32.17 | 36.14 | 73.99 (4.96) | 73.51 (4.96) | 38.31 | 89.00 |
| chat | 32.24 | 32.45 | 49.67 (3.01) | 48.04 (3.01) | 38.61 | 60.28 |
| edit | 32.09 | 103.36 (6.47) | 79.00 (5.32) | 101.77 (13.10) | 38.34 | 94.48 |
| copy | 32.10 | 121.63 (13.45) | 106.57 (7.86) | 147.94 (15.97) | 38.11 | 135.11 |

- Every text of the four engine columns is the plain run's; llama.cpp's LiLiCorr texts differ from its plain ones on prose and chat. llama.cpp's accepted lengths on these asks: 2.93, 5.31, 3.52, 5.44, 7.86 (it verifies every drafted token; StairCut cuts a block where the next row costs more than it is worth).
- The shared bench of the engine survey (`bench_decode.py`, a 256-token story prompt, 256 tokens, greedy, thinking off, 3 runs): plain 32.2, LiLiCorr 40.5-47.9, LiLiCorr + lookup 38.7-47.6; llama.cpp plain 38.1, llama.cpp + LiLiCorr 47.2 (median).
- 1 / 2 / 3 requests at once with LiLiCorr + lookup (`lfm_parallel.py`, 256 tokens): 41.6 / 50.0 / 56.0 tok/s in total, every text its request's alone.
- `fn_server_check.py`, 20 requests: 17 the same as plain; the other 3 (thinking on, a low thinking cap, JSON with 150 tokens) the same up to the reasoning budget's close, which lands at the end of the verified block (4 to 6 tokens later, docs/exactness.md, output changes).
- A verify block's cost is the IQ2_XXS kernels (CUDA events, 1k context, 1 / 8 / 16 rows): the expert bank 3.9 / 22.1 / 42.7 ms, the shared experts' GEMV 3.4 / 11.3 / 22.4; the NVFP4 projections (11.0 .. 11.6), the Q8_0 head, attention and latents (9.5 .. 9.9) and the SSM walk (0.8 .. 2.2) barely move. Chains of 1 / 2 / 4 / 8 / 16 rows: 29.4 / 34.4 / 41.5 / 57.6 / 90.2 ms.

On the EXL3 2.0 pack (`nemotron-bolt-exl3-2.0`: experts at 2 bits, the dense projections at 4, head 6), with the projections on `dense_rows` and the bank grouped ([How the engine runs it](#how-the-engine-runs-it)); DGX Spark, 2026-10-15, one fresh server a run, the GPU to itself, max_len 32,768, conductor raw, suffix store off, 512 greedy tokens, thinking off (`fn_lookup_ab.py`; tok/s, tokens a block):

| class | EXL3 plain | EXL3 LiLiCorr + lookup | IQ2 LiLiCorr + lookup (above) | llama.cpp + LiLiCorr (IQ2 GGUF) |
|-|-|-|-|-|
| prose | 43.04 | 65.81 (3.01) | 42.59 | 47.93 |
| code | 43.01 | 98.98 (4.87) | 73.83 | 89.00 |
| chat | 43.16 | 69.27 (3.10) | 47.99 | 60.28 |
| edit | 43.10 | 152.75 (12.46) | 114.16 | 94.48 |
| copy | 43.16 | 256.00 (15.97) | 147.15 | 135.11 |

- Every LiLiCorr text is the plain run's. `fn_server_check.py`, 20 requests: 16 the same; thinking on, a low thinking cap and JSON at 150 and 140 tokens the same up to the reasoning budget's close (docs/exactness.md, output changes). 1 / 2 / 3 requests at once (3 slots, 256 tokens): 58.8 / 66.4 / 72.1 tok/s in total, every text its request's alone.
- Chains of 1 / 2 / 4 / 8 / 16 rows at 1k context: 22.1 / 24.7 / 28.7 / 33.8 / 44.9 ms (a CTA a row: 23.8 / 28.0 / 41.7 / 74.7 / 142.1); the decode step 23.1 ms (44.2 tok/s from its graph in a later window; `dense`: 41.0). At 8 rows: `dense_rows` 17.3 ms (the decode step's `dense` 18.0), the bank 10.6, the head 1.6, the SSM walk 1.3.
- Rows of a real 8 / 16-row block route to 57 / 46 % distinct experts; the grouped bank reads each once.

## Checks

| check | result |
|-|-|
| config: the five published configs | `tests/test_nemotron_h_config.py` |
| reference vs transformers fp32 | within 2e-5, three tiny models |
| kernels: chain rows = decode steps, commit = decode state | bit for bit, GB10 and the CPU interpreter |
| engine: verify = decode, tree, moved-path commit, rollback, joint passes, graphs | 8 tests, GB10 |
| GGUF: tiny IQ2_XXS / Q8_0 / NVFP4 (with a tensor scale) file through the engine | self-check bit for bit on the packed kernels |
| Bolt's GGUF against llama.cpp on the same file, teacher-forced on llama.cpp's greedy tokens (`nh_check.py llamacpp`, `engine --force`, `llamacmp`; 8 chat prompts x 128 tokens) | top-1 97.75 % of 1,024 positions; all 23 differences near ties by llama.cpp's own margin, none other; mean absolute log-prob difference over llama.cpp's top 10: 0.17 |
| Bolt load-time self-check | PASS, 24 rows and the committed states |
| Bolt server: plain vs lookup, 20 requests | 19 the same; the thinking-on one the same up to the reasoning budget's close (one token later) |
| Lightning BF16 against transformers BF16, teacher-forced (`nh_check.py hf --shared`, 8 chat prompts x 128 tokens) | top-1 98.63 % of 1,024 positions; all 14 differences near ties (margin under 8 bf16 steps), none other; mean relative logit difference 0.0075 |
| Lightning BF16 load-time self-check | PASS, 24 rows and the committed states |
| server: plain vs lookup, 20 requests (greedy, streaming, tools, JSON, penalties, seeds, stop) | every answer the same |
| caches (`fn_cache_check.py`): two sessions parked and restored, a 10k document twice, against cold answers | every answer the same token for token; logprobs within 0.07 where the end anchor restored (the tail's prefill chunks differ from the cold prompt's) |
| taps (`tests/test_lilicorr.py`): verify rows' taps and the committed ring against the decode steps', graphs, joint steps, prompt chunks | bit for bit; chunk rows near the row path |
| SSM walk of a verify block (`tests/test_nemotron_h_kernels.py`): a chain and a branching tree | every row the decode steps along its path, bit for bit |
| LiLiCorr against an fp32 reference written from vLLM's and llama.cpp's code (tiny weights) | block within 2e-5, lattice scores within 1e-4, the same walk |
| LiLiCorr on Bolt against llama.cpp + LiLiCorr, 8 prompts x 256 | accepted length 4.55 against 4.57 |
| Bolt server: plain vs LiLiCorr + lookup, 20 requests and 5 asks | 22 the same, 3 the same up to the reasoning budget's close |
| EXL3 rows path (`tools/fn_dense_rows_check.py`): `dense_rows` against the one-row `dense` at Bolt's, the 27B's and Flash-Next's shapes, M 1..32, clusters, 8 warps, groups, the fp32 head, every rate; all 281 projections of the 2.0 pack; `dense` against 0.9.x's | every byte the same |
| EXL3 bank grouped (`tests/gpu/test_nemotron_h_exl3_gpu.py`, `fn_dense_rows_check.py bank`): M 2..32, every row on the same 9 experts up to few shared | every pair its lone call's bytes |
| Bolt EXL3 2.0 engine on the rows path (`fn_dense_rows_check.py engine`, 3 prompts): self-check at 24 and 8 rows, a 16-row tree against each branch's sequential decode, graph replay, 32 decode steps with `QSE_NH_DENSE_ROWS` on and off | bit for bit |
| Bolt EXL3 2.0 server: plain vs LiLiCorr + lookup, 20 requests and 5 asks | 21 the same, 4 the same up to the reasoning budget's close |
