#!/usr/bin/env bash
# PREP: the calibration and evaluation data every EXL3 target shares. One h200 job, ~1 h.
#   1. self-sampled calibration trace from the BF16 model (sc_trace.py, 250 x 2048 tokens)
#   2. a disjoint held-out trace, new prompts over the same distribution (eval/qbench_prompts.py)
#   3. the engine's own held-out texts packed into 2048-token rows (pack_rows.py), checked disjoint
#   4. BF16 reference perplexities on both held-out sets (eval/ppl.py), so every quant's loss
#      delta is against the same numbers
# Every step is skipped when its output is already in the bucket, so a rerun finishes the rest.
# Outputs: bucket exl3/cal/ and the private repo <REPO_PREFIX>-calibration.
source "$(dirname "$0")/common.sh"
log_to $W/logs/prep.log
el "PREP start"
gpu_banner
setup_env
fetch_bf16
mkdir -p $CAL

# ---- 1. calibration trace from the unquantized model (exllamav3 runs BF16 checkpoints directly)
if bhas cal/cal_trace.safetensors; then
  el "trace present in the bucket"; bget cal/cal_trace.safetensors $CAL/cal_trace.safetensors; bget cal/cal_trace.json $CAL/cal_trace.json
else
  el "sc_trace: 250 x 2048 from BF16, 32 recurrent slots"
  python3 $EXL3_SRC/sc_trace.py -m $BF16 -o $CAL/cal_trace.json -co $CAL/cal_trace.safetensors \
      -cr 250 -cc 2048 -ambs 32 --seed 0 2>&1 | tee $CAL/sc_trace.txt
  bput cal/cal_trace.safetensors $CAL/cal_trace.safetensors
  bput cal/cal_trace.json $CAL/cal_trace.json
  bput cal/sc_trace.txt $CAL/sc_trace.txt
fi
el "trace ready"

# ---- 2. held-out in-domain trace: a different prompt set, 30 % tool-calling conversations
if bhas cal/qbench_prompts_gen.json; then
  bget cal/qbench_prompts_gen.json $CAL/qbench_prompts_gen.json
else
  el "qbench_prompts: >= 45,000 sampled response tokens"
  python3 $EXL3_SRC/eval/qbench_prompts.py -m $BF16 -o $CAL/qbench_prompts_gen.json \
      --min_tokens 45000 --max_new_tokens 2560 -ambs 32 2>&1 | tee $CAL/qbench_prompts.txt
  bput cal/qbench_prompts_gen.json $CAL/qbench_prompts_gen.json
  bput cal/qbench_prompts.txt $CAL/qbench_prompts.txt
fi
el "held-out trace ready"

# ---- 3. the engine's held-out texts (bench/heldout_*.txt of TandemLLM, pushed by jobs.sh, and
#         the NVFP4 build's sensitivity split, held out from that build's calibration) as rows
if bhas cal/heldout_rows.safetensors; then
  bget cal/heldout_rows.safetensors $CAL/heldout_rows.safetensors
  bget cal/heldout_rows.safetensors.json $CAL/heldout_rows.safetensors.json
else
  mkdir -p /tmp/texts
  cp $W/texts/*.txt /tmp/texts/
  hf buckets cp hf://buckets/$BUCKET/nvfp4-bf16/calib/sens-code.txt /tmp/texts/sens-code.txt >/dev/null
  hf buckets cp hf://buckets/$BUCKET/nvfp4-bf16/calib/sens-prose.txt /tmp/texts/sens-prose.txt >/dev/null
  ls -la /tmp/texts
  python3 $W/code/pack_rows.py --tokenizer $BF16 --length 2048 --out $CAL/heldout_rows.safetensors \
      --disjoint-from $CAL/cal_trace.safetensors \
      /tmp/texts/heldout_prose.txt /tmp/texts/heldout_code.txt /tmp/texts/sens-prose.txt /tmp/texts/sens-code.txt
  bput cal/heldout_rows.safetensors $CAL/heldout_rows.safetensors
  bput cal/heldout_rows.safetensors.json $CAL/heldout_rows.safetensors.json
fi
el "held-out rows ready"

# ---- 4. BF16 reference numbers on both held-out sets
if ! bhas cal/bf16_ref.json; then
  python3 $EXL3_SRC/eval/ppl.py -m $BF16 -t $CAL/heldout_rows.safetensors 2>&1 | tee $CAL/ppl_bf16_rows.txt
  python3 $EXL3_SRC/eval/ppl.py -m $BF16 -t $CAL/qbench_prompts_gen.json 2>&1 | tee $CAL/ppl_bf16_trace.txt
  python3 $W/code/summarize.py --ppl-rows $CAL/ppl_bf16_rows.txt --ppl-trace $CAL/ppl_bf16_trace.txt \
      --label bf16 --out $CAL/bf16_ref.json
  bput cal/bf16_ref.json $CAL/bf16_ref.json
  bup $CAL cal
fi
el "BF16 reference ready"

# ---- upload the calibration set to its private repo
mkdir -p /tmp/up/calibration
cp $CAL/cal_trace.json $CAL/cal_trace.safetensors $CAL/qbench_prompts_gen.json \
   $CAL/heldout_rows.safetensors $CAL/heldout_rows.safetensors.json $CAL/bf16_ref.json $CAL/versions.json /tmp/up/calibration/
cp $CAL/*.txt /tmp/up/calibration/ 2>/dev/null || true
(cd /tmp/up && sha_dir .) | tee /tmp/up/calibration/sha256.txt
upload_private "$REPO_PREFIX-calibration" /tmp/up "EXL3 calibration set: self-sampled trace, held-out trace, held-out rows, BF16 reference (exllamav3 $EXL3_VERSION)"
el "PREP done"
