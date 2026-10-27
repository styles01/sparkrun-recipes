#!/bin/bash
#, the parallelism gate, 2026-09-23. Runs inside ops/hold.sh (the board alone, :8000 stopped).
#
#   1. the cold curve of every NVFP4 projection shape, at the row counts N sequences x 17 rows
#      produce (N = 1, 2, 4, 8, 16) and at the plan's 48 / 64 / 128 / 272, on the engine's shipped
#      dispatch and on v2 with the tiles a batch would need, each with the row-independence check
#      against one sequence's 17 rows
#   2. the e4m3 head at the same rows
#   3. the block cycle, phase by phase, at fixed width 16 and 8, for the block model
set -u
cd "$(dirname "$0")/.."
set -a; . ops/serve.env; set +a
export PYTHONPATH=$PWD:$HOME/pylibs
OUT=results/vis8-0923; mkdir -p $OUT
ROWS=1,17,34,48,64,68,128,136,272
TILES="m64:n64:k1:w8:s3,m64:n64:k1:w4:s3,m32:n64:k1:w4:s3,m64:n128:k1:w8:s3,m128:n64:k1:w8:s2,m128:n64:k1:w8:s3"
date -Is
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
for S in "17408 5120 3.0" "5120 17408 3.0" "10240 5120 3.0" "6144 5120 3.0" "5120 6144 3.0" \
         "12288 5120 3.0" "1024 5120 2.0"; do
  set -- $S
  $PY -u tools/cold_bw.py --impl shipped --N $1 --K $2 --gb $3 --rows $ROWS --reps 4 \
      --rowcheck 17 --json $OUT/shipped.jsonl
  $PY -u tools/cold_bw.py --impl v2 --N $1 --K $2 --gb $3 --rows $ROWS --reps 4 \
      --rowcheck 17 --tiles "$TILES" --json $OUT/v2.jsonl
done
$PY -u tools/cold_bw.py --head --K 5120 --gb 2.6 --rows $ROWS --reps 4 --rowcheck 17 \
    --json $OUT/head.jsonl
date -Is
for W in 16 8; do
  $PY -u tools/profile_cycle.py --nvfp4 "$NV" --fp8-head "$HEAD" --ckpt8 "$CKPT8" \
      --ckpt16 "$CKPT16" --corpus "$CORPUS" --fixed $W --workloads prose,chat,quote \
      --json $OUT/cycle-fixed$W.json
done
date -Is
