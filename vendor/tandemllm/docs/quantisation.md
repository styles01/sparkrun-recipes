# Quantisation and profiles

Four-bit weights take a decode step from 26.93 GB to 15.01 GB, and on this board bytes are time. This page covers the formats the engine reads, the quantiser that writes the 4-bit copies, the gate that decides whether they ship, and the three profiles built from them.

The served weights are the published set, [`0xBakeer/TandemLLM-Qwen3.8-27B-NVFP4`](https://huggingface.co/0xBakeer/TandemLLM-Qwen3.8-27B-NVFP4): every projection in NVFP4, quantised once from Qwen's BF16 release with GPTQ, plus an e4m3 output head ([From the BF16 release](quantisation.md#from-the-bf16-release)). Against the BF16 model its held-out loss rises by +0.0221 nats on prose and +0.0426 on code, inside the gate's bar of +0.05. The sections below start with the first weight sets, quantised from the FP8 release with the clip search alone, because the published set uses the same format and the same search.

The second model, Qwen3.8-Flash-Next, is not quantised here at all. It runs from turboderp's published EXL3 packs, read as they are; [the last section](quantisation.md#flash-next-on-turboderps-exl3-packs) covers them.

## What the checkpoint stores

Qwen ships the model as block-scaled FP8. Every large projection is a pair of tensors: `X.weight` holds e4m3 codes of shape `[N, K]`, and `X.weight_scale_inv` holds one BF16 scale per 128 by 128 block. A weight is `code * scale[n // 128, k // 128]`.

`engine/loader.py` keeps that pair as an `FP8Block` and never expands it in memory, because a decode step pays for exactly these bytes. `tools/fp8_linear.py` multiplies by it directly. Its tile is 128 by 128 on purpose: every step along K covers one scale block, so the scale is one number applied to the fp32 sum.

Some tensors stay BF16 in the checkpoint: the output head, the embedding, every norm, the convolution weights and the two small projections `in_proj_a` and `in_proj_b`.

## NVFP4

NVFP4 stores a weight in three tensors, held by `tools/nvfp4_linear.py::NVFP4Block`:

| tensor | type | shape | meaning |
|-|-|-|-|
| `weight` | uint8 | `[N, K/2]` | two e2m1 codes a byte, low nibble first |
| `weight_scale` | e4m3 | `[N, K/16]` | one scale per 16 weights of a row |
| `weight_scale_2` | fp32 | one number | the scale of the scales |

A weight is `e2m1(code) * weight_scale[n, k // 16] * weight_scale_2`. The e2m1 grid has eight sizes (0, 0.5, 1, 1.5, 2, 3, 4 and 6) and a sign. A weight costs 0.5625 bytes against 1.0 in FP8, and one layer's MLP drops from 267.4 MB to 150.4 MB.

A group of 16 is what makes 4 bits usable. An FP8 block shares one scale across 16,384 weights, while an NVFP4 group shares it across 16, so the grid follows a row's local range.

Our kernel is W4A16: it unpacks the 4-bit codes to 16-bit numbers in registers and multiplies on the ordinary tensor cores. On this board NVFP4 is a storage format and not a compute format. It saves bytes, and the arithmetic stays the same.

## The clip search

Rounding each group to nearest fails the quality gate on code by 0.007 nats. The same weights with a clip search pass it with 0.014 to spare.

`tools/quant_nvfp4.py` tries five scale multipliers per group (1.00, 0.95, 0.90, 0.85, 0.80). A smaller multiplier clips the group's largest weight and gives the rest a finer grid. It keeps the multiplier with the smallest error as the model sees it:

    sum over k of  act_ms[k] * (w[k] - dequant(quant(w[k])))^2

`act_ms[k]` is the mean square of input channel `k` over a calibration text. A channel the model never uses can take a coarse grid at no cost, and a channel it leans on cannot. The statistics come from running `bench/calib.txt` (63 KB, two thirds Python and one third package documentation, 8,192 tokens) through the engine and recording the input of every projection. Statistics for all 64 layers take 41 s, and quantising takes 120 s.

## Which tensors, and what they cost

`tools/quant_nvfp4.py` has three targets, one output file each:

| target | projections | count | bytes saved a step |
|-|-|-|-|
| `mlp` | `gate_proj`, `up_proj`, `down_proj` | 192 | 7.49 GB |
| `gdn` | `in_proj_qkv`, `in_proj_z`, `out_proj` | 144 | 2.43 GB |
| `attn` | `q_proj`, `k_proj`, `v_proj`, `o_proj` | 64 | 0.74 GB |

These first sets were quantised from the FP8 release, and the gate scored each against that FP8 checkpoint on held-out text, as a change in loss per token:

| weights | prose | code | code argmax, confident positions |
|-|-|-|-|
| FP8 as shipped | 1.5366 nats | 0.7900 nats | |
| NVFP4 MLPs | +0.0131 | +0.0356 | 0.9754 |
| NVFP4 MLPs and GDN | +0.0186 | +0.0436 | 0.9694 |
| NVFP4 MLPs and attention | +0.0185 | +0.0429 | 0.9748 |
| NVFP4 everywhere | +0.0211 | +0.0493 | 0.9688 |

Its bar is +0.05 nats on the worse of the two texts, and all four pass. All projections at NVFP4 pass by 0.0007 nats on a code file of 2,047 tokens, which is a thin margin. Re-run the gate before quantising anything more.

Code costs about three times what prose costs in every row. One published NVFP4 export of this model, read through the same loader (its MLPs only), scored +0.0576 on code and fails the same bar.

## From the BF16 release

The set this section builds is the one the engine serves, published as [`0xBakeer/TandemLLM-Qwen3.8-27B-NVFP4`](https://huggingface.co/0xBakeer/TandemLLM-Qwen3.8-27B-NVFP4) with its calibration manifest and quality reports.

The weight set above came from Qwen's FP8 release, so every weight in it was rounded twice: to e4m3 on a 128 by 128 grid by Qwen, then to NVFP4 by us. `tools/quant_nvfp4.py` also reads the BF16 release `Qwen/Qwen3.8-27B`, whose projections are plain BF16 matrices in 18 shards. `Source` finds each projection through the checkpoint's index in either layout and skips the vision tower and the MTP layer.

`quant_nvfp4.py build` loads the BF16 model once (about 50 GB) and does everything in that process:

1. It runs a calibration text through the engine. `tools/calib_corpus.py` writes it: `bench/calib.txt` plus Python standard-library modules, English Wikipedia and German Wikipedia, 149,682 tokens in the build we shipped. Any document that shares a run of 12 words with a held-out file is dropped.
2. It records each projection's mean square input, which the clip search weights by, and the full second moment of each input, `H = X^T X`. At fp32 the second moments are 101 GB for 64 layers, so the build takes them in passes of about 13 layers (`--h-budget-gb 22`).
3. It quantises every projection twice: with the clip search, and with GPTQ on the NVFP4 grid. GPTQ rounds one input column at a time and pushes each column's error onto the columns not yet rounded, weighted by the inverse of `H`. At the first column of each group of 16 the clip search picks the group's scale from the weights as they stand. The format and the kernel do not change; only the stored codes do.

On the calibration inputs, GPTQ leaves 0.0024 relative output error against 0.0044 for the clip search. On held-out text it is also the better set. Against the BF16 model, on the same tokens:

| weights | prose | code (8,002 tokens) | more code | more prose | GB a step |
|-|-|-|-|-|-|
| FP8 release, BF16 head | +0.0079 | +0.0032 | +0.0050 | +0.0028 | 26.93 |
| NVFP4 everywhere, from FP8, clip | +0.0304 | +0.0220 | +0.0530 | +0.0286 | 15.01 |
| NVFP4 everywhere, from BF16, clip | +0.0267 | +0.0198 | +0.0373 | +0.0233 | 15.01 |
| NVFP4 everywhere, from BF16, GPTQ | +0.0234 | +0.0184 | +0.0272 | +0.0174 | 15.01 |
| NVFP4 MLPs, from FP8, clip | +0.0239 | +0.0144 | +0.0372 | +0.0197 | 18.17 |
| NVFP4 MLPs, from BF16, GPTQ | +0.0136 | +0.0144 | +0.0245 | +0.0159 | 18.17 |

For the published set (GPTQ, from BF16, with the e4m3 head), the paper reports, against the BF16 model: +0.0221 nats on prose and +0.0426 on code on the 2,048-token gate files, within the +0.05 bar, and +0.0174 on 12k tokens of prose, +0.0184 on 8k tokens of code and +0.0272 on 10.5k tokens of code.

"More code" and "more prose" are 10,500 and 11,999 tokens that neither the calibration nor anything else reads (`calib_corpus.py --eval-chars`). `quality_gate.py --plan` scores weight sets over one checkpoint against a reference loaded from another, which is how these rows compare against BF16 while the NVFP4 files load over the FP8 release as they do in serving.

## Mixing NVFP4 and FP8

`tools/quant_sensitivity.py measure` swaps one piece of one layer to NVFP4 at a time and measures how far the next-token distribution moves (KL over the reference's top 256 tokens, on 8,192 held-out tokens). A piece is a whole fusion group: the MLP's gate and up projections, its down projection, the GDN input projections, the GDN output projection, the attention's input projections (`q_proj`, `k_proj`, `v_proj`), the attention output. Per byte kept, the GDN and attention output projections cost the most, and layers 35 to 41 are the most sensitive. `quant_sensitivity.py mix --keep-gb N` keeps the pieces with the highest cost per byte in FP8 until N GB a step are spent and writes the rest into one file.

| weights | prose | code | more code | more prose | GB a step |
|-|-|-|-|-|-|
| mix, 0.99 GB kept in FP8 | +0.0144 | +0.0145 | +0.0211 | +0.0174 | 16.00 |
| mix, 3.16 GB kept in FP8 | +0.0090 | +0.0084 | +0.0185 | +0.0132 | 18.17 |

The second mix reads the same bytes as NVFP4 MLPs alone and beats it on all four texts. The per-piece costs do not add: the 256 pieces sum to 3.4 times the cost of all of them at once. A mix is a ranking, and the gate decides what it costs.

## The gate

`tools/quality_gate.py` reports three things, and a weight set ships only when all three hold.

1. Held-out loss, teacher-forced, as a change against a reference on the same tokens in the same process. By default the reference is the FP8 engine (`--baseline fp8`); with `--plan` it is a model loaded from another checkpoint, which is how the published set is scored against BF16. The held-out texts are `bench/heldout_code.txt`, a Python file the calibration text does not contain, and `bench/heldout_prose.txt`, English prose on unrelated topics.
2. Argmax agreement with the reference, overall and at the positions where the reference is confident (top-1 ahead of top-2 by at least 1.0 in the logits). Overall agreement for the full set from FP8 is 0.9101. On a flat distribution over 248,320 tokens the top choice is decided by rounding, so the confident number (0.9688) is the one to watch.
3. A free generation of 900 greedy tokens on five prompts the calibration never saw, with repetition statistics. On the full set from FP8 the share of unique 4-grams was 0.939 to 1.000, and every generation stopped at its own end.

Test 3 matters most. Teacher-forced loss never lets an error grow, so it cannot see a model that drifts off its own path. On an earlier engine on this board, loss rated a broken configuration better than a healthy one.

## The FP8 head

As shipped, the output head is 248,320 by 5,120 in BF16, 2.54 GB, and every verify reads all of it. `tools/quant_head.py build` writes it as e4m3 with one fp32 scale per vocabulary row (1.27 GB), searching the multipliers 1.0, 0.95 and 0.90 of each row's largest weight. One scale per row, because rows are tokens and their sizes differ more than a block scale shared by 128 neighbouring ids can follow.

This head decides what the engine writes, so it took the full gate: prose +0.0015 nats, code -0.0009, and agreement 1.0000 on all 2,618 confident positions. Its kernel returns fp32 logits at every row count. `--fp8-head build` builds the same head at load, without the file.

## What stays at full precision

The 48 recurrent states stay fp32. They are 151 MB, read and written once a step, about 1.5 ms of traffic. A recurrent state is a sum over the whole sequence carried across every step, so its rounding error grows with every step, and a weight's does not. The convolution weights, `A_log`, `dt_bias`, every norm and the drafters' own layers also keep their shipped precision.

## Profiles

A profile is a `serve.env` file. The three that ship differ only in their weights. Drafters, StairCut, caches and the exactness gate are the same in all of them. StairCut's verify prices (`ops/stair-tables-nvfp4.json`) were measured on the full-NVFP4 profile. The router corrects their level online, with its per-class block factor (`engine/lenrouter.py`), but the staircase's shape for the balanced and FP8 profiles is not measured, and neither is their speed on this release.

| profile | file | MLP | GDN and attention | head | bytes a step | single request |
|-|-|-|-|-|-|-|
| NVFP4 (served) | `ops/serve.env` | NVFP4 | NVFP4 | e4m3 | 15.0 GB | 49.89 tok/s with StairCut (paper) |
| balanced | `ops/serve-balanced.env` | NVFP4 (the published set's) | checkpoint FP8 | e4m3 (the published set's) | about 18.2 GB | not measured on this release |
| FP8 | `ops/serve-fp8.env` | checkpoint FP8 | checkpoint FP8 | checkpoint BF16 | 26.9 GB | not measured on this release |

An FP8 profile reads nothing but the vendor's files, so its quality is the checkpoint's. Balanced keeps the recurrent and attention layers on the vendor's weights and quantises only the MLPs, which carry 70 % of a verify's projection bytes.

Profiles and bytes interact with speculation. At one token a step, the full NVFP4 set is worth 14 % over NVFP4 MLPs alone. On a 16-node tree it is worth 0.2 %, because a round that commits 13 tokens has already spread its weight read over 13 tokens. Bytes and accepted tokens substitute for each other.

The 27B also has weight profiles in EXL3 (`ops/profiles/exl3-<bpw>.env`, switched with `ops/use-profile.sh`, [operations.md](operations.md)). The six sizes are public on Hugging Face, `0xBakeer/TandemLLM-Qwen3.8-27B-EXL3-{2.5,3.0,3.5,4.0,4.5,6.0}`, with the K3 drafters and the K4 draft head in `0xBakeer/TandemLLM-Qwen3.8-27B-EXL3-drafters`; the collection is [TandemLLM Qwen3.8-27B EXL3](https://huggingface.co/collections/0xBakeer/tandemllm-qwen38-27b-exl3-6acdd787008264fa4f35eab4). We quantised them with exllamav3 from Qwen's BF16 release. Each card lists every size's KL against BF16 on wikitext-2, from 0.1112 at 2.5 bits to 0.0018 at 6.0, and its speed on one DGX Spark. Running them needs nothing beyond those repositories, the BF16 drafters (`...-DFlash2-b8` and `-b16`) and the base checkpoint; `ops/use-profile.sh exl3-<bpw>` fetches all of it. This page does not cover how the EXL3 files were made.

### Standalone 27B EXL3 checkpoints

A complete BF16-origin, `mul1` EXL3 checkpoint of the supported 27B family can be passed directly as `--model`. 0xWhiteMage wrote this path in pull request #1 (0.8.2), and the first community pack runs on it ([community-weights.md](community-weights.md)).
The loader recognises `quantization_config.quant_method: exl3` in its config; a separate BF16/FP8
base download is not needed when the pack includes the embedding, norms and GDN auxiliary tensors.
For example, for a text-only run without speculation:

```bash
python server/app.py --model /path/to/exl3-pack --drafter none --vision off
```

The projected weights and output head retain their original EXL3 codes, channel scales and mixed
bit widths, including indexed packs whose projection components span multiple shards. When a standalone
pack declares a BF16 source through `dtype` or `torch_dtype`, the loader re-encodes its finite FP16
plain tensors as BF16 only when every stored value survives an exact round trip. Comparisons use
bounded chunks, including for a plain output head. Missing or non-BF16 source metadata, non-finite
values and lossy conversions fail with a named error rather than silently changing precision.
This preserves the pack's stored numerical values, not necessarily its pre-conversion BF16 source:
precision already lost while creating the pack cannot be recovered. Source-equivalence needs the
original checkpoint, and the original BF16 base plus an EXL3 overlay remains an alternative.
The existing separate-base overlay precision path is unchanged, and explicit
EXL3 paths take precedence over detection (an empty `exl3` setting disables detection).

This does not add another codebook, model architecture or MTP format. The existing `mul1` reader
and its validation still apply. A stock DFlash2 checkpoint can be supplied with `--drafter dflash2
--dflash2-ckpt /path/to/drafter` for one sequence; speculative parallel serving still requires the
engine's `lenrouter` configuration and a suitable block-16 drafter. Vision and quantised MTP
loading are not qualified by this text-only path.

## Knobs

| knob | what |
|-|-|
| `--nvfp4 a,b,c` or `QWEN38_NVFP4` | the NVFP4 files to load, one per target; empty means the checkpoint's FP8 |
| `--fp8-head FILE`, `--fp8-head build` or `QWEN38_FP8_HEAD` | the e4m3 head; empty means the checkpoint's BF16 head |
| `tools/quant_nvfp4.py quant --mode rtn\|clip --targets --stats` | the quantiser |
| `tools/quant_head.py build --ratios` | the per-row scale search for the head |
| `tools/quant_nvfp4.py build --model <BF16> --corpus --methods clip,gptq --h-budget-gb` | the statistics and both quantisers (clip and GPTQ) from one load |
| `tools/calib_corpus.py --out DIR --eval-chars` | the calibration, sensitivity and wide-gate texts |
| `tools/quant_sensitivity.py measure` and `mix --keep-gb` | the per-piece costs and the FP8/NVFP4 mix file |
| `tools/quality_gate.py --plan plan.json --corpus name=path` | weight sets against a reference checkpoint, on extra texts |

## Limits

The gate files are one held-out file per domain, 2,048 tokens each; the wider texts add 8k to 12k tokens per domain. No task benchmark ran against the quantised engine, so this page makes no claim about task accuracy. The clip search assumes that input channels are uncorrelated, which is a first-order model.

## Flash-Next on turboderp's EXL3 packs

Qwen3.8-Flash-Next comes in five sizes on Hugging Face, `turboderp/Qwen3.8-Flash-Next-exl3`, one branch each. The engine reads those packs as they are. It runs no quantiser of its own on them and adds no format. A pack is lossy against Qwen's BF16 release by design; speculation is still lossless against the pack that is served ([exactness.md](exactness.md)).

### The five packs

The branch name says what is inside: `3.05bpw_h5_ng5` is 3.05 bits a weight on average, the output head at 5 bits, and the n-gram table at 5 bits a value. The figures below are the registry's (`models/qwen3.8-flash-next/model.yaml`). The weights are the GiB the engine holds on the device, the draft layer and the int8 mixers included: measured for 3.05 at load, and estimated from the file sizes for the other four.

| profile | branch | disk, with the table | weights on the device | n-gram table | largest window on one DGX Spark |
|-|-|-|-|-|-|
| `flashnext-2.05` | `2.05bpw_h4_ng4` | 61.5 GB | 32.85 GiB (estimate) | 24.44 GiB, warm | 262,144 |
| `flashnext-3.05` (recommended) | `3.05bpw_h5_ng5` | 85.1 GB | 47.87 GiB (measured) | 30.4 GiB, cold | 262,144 |
| `flashnext-4.05` | `4.05bpw_h6_ng6` | 106.1 GB | 62.47 GiB (estimate) | 36.36 GiB, cold | 208,896 |
| `flashnext-5.05` | `5.05bpw_h6_ng6` | 121.9 GB | 77.16 GiB (estimate) | 36.36 GiB, cold | does not load yet |
| `flashnext-6.05` | `6.05bpw_h6_ng6` | 137.7 GB | 91.87 GiB (estimate) | 36.36 GiB, cold | none: it does not fit |

The windows come from the Setup Guide's estimate ([setup.md](setup.md)), calibrated on one 260,800-token request each on 2.05, 3.05 and 4.05 (2026-10-12). It predicts the lowest MemAvailable of those three runs within 0.7 GiB. 2.05 and 3.05 hold the model's full 262,144 tokens. 4.05 ran a 260,800-token request with MemAvailable never under 17.8 GiB on a box whose other programs took 4.6 GiB; under the Setup Guide's frame of 10 GiB for the rest of the system, its window comes out at 208,896 with the vision tower on (0.31 GiB, up to 0.81 with the 512 MB image cache; VIS-97) and 212,992 without it.

5.05 does not load on this engine yet. Its routed experts use 5 and 7 bits a weight, and the MoE kernels take 2 to 6 and 8 (`tools/fn_kernels.py` `MoEBank` refuses the 7). By the estimate its window would be 53,248 with the vision tower and 65,536 without. 6.05 needs about 92 GiB for its weights, which leaves too little under the 15 GiB floor at any window.

The 4.05, 5.05 and 6.05 branches carry the same n-gram table file (the same sha256), so `ops/fetch_weights.py` hard-links it between packs instead of downloading 39 GB again ([operations.md](operations.md#one-model-at-a-time-the-27b-and-flash-next)).

### What a pack holds, and how the engine reads it

`engine/models/flashnext/pack.py` (`PackIndex`) reads every shard header and puts each tensor in one kind:

| kind | what | how the engine holds it |
|-|-|-|
| `expert` | the routed experts' gate, up and down projections, 98 % of the bytes | EXL3, one image a layer, through the MoE kernels |
| `exl3` | every other coded projection: GDN, attention, the sparse-attention indexer, the shared expert, the MTP layer's projections, the head | EXL3 |
| `mixer` | the hyper-connection mixers' projections | fp16 in the pack; on the device int8 with one fp32 scale for each group of 128 weights |
| `plain` | norms, the router, `in_proj_a` and `in_proj_b`, the n-gram projections, conv weights, the embedding | as stored |
| `table` | the n-gram table, its own file | never on the device; read a row at a time |
| `vision` | the vision tower | not loaded |

An EXL3 projection is four tensors. `X.trellis` (int16) holds the codes at K bits a weight, and K is read from each tensor's own shape, so a pack that mixes K inside a layer is described as it is. `X.suh` and `X.svh` (fp16) scale and sign the input and output sides, which are rotated, and `X.mul1` is the codebook's multiplier. On the 3.05 pack the header audit counted 303,121 tensors: every routed expert at K 3, the dense projections and the head at K 5, except the sparse-attention indexer's projection at K 3 and the MTP layer's projections at K 3 to 5. By those headers a plain decode token reads about 4.23 GB in all at 1k of context, with the mixers as int8 (`PackIndex.decode_bytes`).

`tools/fn_exl3_xcheck.py` holds our reading against exllamav3's own on the 3.05 pack. The trellis decode is equal bit for bit on 16 tensors (experts in layers 0, 11 and 47, GDN, attention, the indexer, a shared expert, a slice of the head). The dequantised weight agrees within 0.36 % of the largest weight, which is our BF16 rounding of the fp32 rotations. 4,096 n-gram rows are equal bit for bit.

### The mixers as int8

The pack stores the mixers in fp16, and every token reads them. The engine keeps them as int8 with one fp32 scale for each group of 128 weights of a row (gcd(128, K): 128 for the down and inject weights, 64 for the up weights). A weight is then off by at most its group's largest weight over 254, plus fp32 rounding. The kernels sum each 16-weight chunk's products and add them times the chunk's group scale, and the prefill dequantises with the same scales. `QWEN38_FN_MIX=int8rows` keeps one scale a row, the form before ENG-221, for an A/B; `fp16` keeps the pack's values.

`tools/fn_mix_quality.py` measures what that costs against the pack's fp16 mixers, on the same kernels. On 3.05 with 8,192 tokens, the groups gave a KL of 0.0031 on code and 0.0188 on prose. The arithmetic noise between the reference forward and the kernels is 0.0038 and 0.0255 on the same prompts, so both are inside it. One scale a row gave 0.267 on code and 0.047 on prose: a row whose largest weight is 10 to 420 times its median left its small weights coarse. Int4 gave 0.38 on prose (0.35 in groups of 128) and changed the continuation at its first token, so the mixers stay int8. The groups cost no decode speed (one run each at 1k, rows against groups: prose 41.91 and 41.76, code 42.64 and 43.24, chat 42.24 and 42.21 tok/s) and 0.02 GiB of device memory. Speculation stays lossless with them, and the load-time self-check passes.

### The n-gram table

For every position the model hashes the token and the one or two before it into 16 row indices (8 hash heads for the bigram, 8 for the trigram), reads those 16 rows of 160 values from one table of about 320 million rows, and joins them into a 2,560-wide feature. The table is coded too: each row is a ring of trellis codes at K bits a value (`ng4` to `ng6`), with one fp16 scale a row and one bias vector a hash head (`engine/models/flashnext/ple.py`).

The table is 32.6 GB in the 3.05 pack, and it never sits in device memory. The engine reads the packed rows a step needs and decodes them on the device. Where the rows come from is the profile's table policy (`QWEN38_FN_TABLE`):

- warm: the file mapped and kept in the page cache. MemAvailable counts that page cache as free, so every floor check takes the table's resident bytes off first ([caches.md](caches.md#memory)).
- cold: the rows read with O_DIRECT through a set-associative row cache of `QWEN38_FN_TABLE_CACHE_GB` (2) in host memory; nothing stays in the page cache.

On 3.05, cold wrote the same greedy text as warm at 1k and 8k and decoded 40.9 tok/s against 42.0 (one run each). 2.05 keeps its table warm. 3.05 reads it cold: with the whole table resident, MemAvailable less the table fell to 8.4 GiB during a 146,400-token request at a 147,456-token window, under the 15 GiB floor. 4.05 and up read it cold because their weights and a warm table do not fit together.

### Exact weights where the prefill needs them

The prompt runs through other kernels than a decode step ([kernels.md](kernels.md)). There the mixers, the router and the small projections take exact fp32 copies of their weights (`QWEN38_FN_PREFILL_MM=fp32`). With fp16 copies the prefill failed its agreement gate on a 3k-token code prompt (95.6 % against the 96.2 % the gate asked), so fp32 is the default.

### What is not measured

Nobody has measured a pack's quality against the BF16 model yet. The BF16 release is 360 GB and does not fit the box, and the reference run that would give KL a pack is not done, so the registry's quality fields are empty. What is measured is our arithmetic against the reference on the same pack: all 48 layers equal the official `transformers` decoder layer on the served weights within BF16 noise (relative error 0.19 to 1.49 %). Teacher-forced, the decode path picks the reference's top token at 64 of 64 positions at 3k of context and 63 of 64 at 8k, where the one miss is a near tie (a margin of 0.0095). No task benchmark has run on any pack.

### Knobs

| knob | what |
|-|-|
| `ops/use-profile.sh flashnext-<bpw>` | switch to a pack; fetches the branch when it is missing |
| `QWEN38_FN_MIX` | `int8` (the default): int8 with an fp32 scale a group of 128 weights; `int8rows`: one scale a row; `fp16`: the pack's values |
| `QWEN38_FN_TABLE`, `QWEN38_FN_TABLE_CACHE_GB` | the n-gram table's policy (`warm` or `cold`) and the cold row cache in GiB |
| `QWEN38_FN_PREFILL_MM` | the prefill's weights for the mixers, router and small projections: `fp32` (the default) |
| `tools/fn_exl3_xcheck.py` | our reading of a pack against exllamav3's |
| `tools/fn_mix_quality.py` | a mixer weight form against the pack's fp16 mixers |
