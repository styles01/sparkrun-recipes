#!/bin/bash
# tool-eval-bench rows against one server, one run at a time, with what each row ran under:
# /health before and after (the version, the pack, the slots, the response cache's hits), the
# bench's flags and version, and the server's own knobs when a command reads them. A row whose
# answers came from the response cache is marked; `tools/teb_summary.py OUT` prints the table.
#
#   bash tools/teb_batch.sh [options] [standard | pack | ROW...]
#
#   --base-url URL     the server's API (default http://127.0.0.1:8000/v1)
#   --out DIR          where the rows go (default results/teb-<time>)
#   --health CMD       prints the server's /health (default: curl on --base-url's /health);
#                      e.g. "ssh box curl -s http://127.0.0.1:8000/health"
#   --restart CMD      runs once before the first row: a restart empties the response cache, so
#                      no row replays an earlier batch's answers
#   --knobs CMD        its output is saved as the server's knobs (the effort default, the slots,
#                      the cache flags), e.g. "ssh box grep -E '^(REASONING_EFFORT|PARALLEL)=' serve.env"
#   --bench CMD        how to start the bench (default: tool-eval-bench)
#   --pack DIR         the scenario pack of the `pack` rows
#   --dry-run          print the rows and their commands, run nothing
#
# A ROW is "label:flags", the flags as the bench takes them, for example
#   "raw-high:--header X-Tandem=raw --backend-kwargs '{\"reasoning_effort\":\"high\"}'"
# `standard` is the full suite on both paths and hard mode on both; `pack` adds the variant seed 7
# and a held-out pack on both paths. The conductor rows send no profile header: the server resolves
# the bench by what each request offers (`tools` with tools, the default profile without), as a
# client would meet it. A row of your own can pin one with --header X-Tandem-Profile=<name>.
#
# Every row: --seed 42 --temperature 0 --timeout 600 --no-live --parallel 1. The batch refuses to
# start when the server's slots (`parallel.max_running`) are fewer than --parallel.
set -u
BASE="http://127.0.0.1:8000/v1"; OUT=""; HEALTH=""; RESTART=""; KNOBS=""; BENCH="tool-eval-bench"
PACK=""; DRY=0; ROWS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --base-url) BASE="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --health) HEALTH="$2"; shift 2 ;;
    --restart) RESTART="$2"; shift 2 ;;
    --knobs) KNOBS="$2"; shift 2 ;;
    --bench) BENCH="$2"; shift 2 ;;
    --pack) PACK="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *) ROWS+=("$1"); shift ;;
  esac
done
RAW="--header X-Tandem=raw --backend-kwargs '{\"reasoning_effort\":\"xhigh\"}'"
COND=''
expand() {
  case "$1" in
    standard) printf '%s\n' "raw-xhigh:$RAW" "cond:$COND" \
                "hard-raw:$RAW --hardmode-only" "hard-cond:$COND --hardmode-only" ;;
    pack) printf '%s\n' "raw-v7:$RAW --variant-seed 7" "cond-v7:$COND --variant-seed 7"
          [ -n "$PACK" ] && printf '%s\n' \
            "pack-raw:$RAW --pack-only --scenario-pack $PACK" \
            "pack-cond:$COND --pack-only --scenario-pack $PACK" ;;
    *) printf '%s\n' "$1" ;;
  esac
}
[ ${#ROWS[@]} -gt 0 ] || ROWS=(standard)
LIST=()
for r in "${ROWS[@]}"; do
  while IFS= read -r line; do [ -n "$line" ] && LIST+=("$line"); done < <(expand "$r")
done
COMMON=(--base-url "$BASE" --model TandemLLM --backend tandemllm --seed 42 --temperature 0
        --timeout 600 --no-live --parallel 1)
if [ $DRY = 1 ]; then
  for row in "${LIST[@]}"; do echo "${row%%:*}: $BENCH run ${COMMON[*]} --label ${row%%:*} ${row#*:}"; done
  exit 0
fi
OUT=${OUT:-results/teb-$(date +%Y%m%d-%H%M)}
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
HEALTH=${HEALTH:-"curl -s -m 10 ${BASE%/v1}/health"}


health() { bash -c "$HEALTH" > "$1" 2>/dev/null; }
field() {   # field FILE PY-EXPR: one value of a saved /health
  python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print($2)" "$1" 2>/dev/null
}

if [ -n "$RESTART" ]; then
  echo "[teb] restart: $RESTART"
  bash -c "$RESTART" || { echo "[teb] the restart failed"; exit 1; }
fi
health "$OUT/batch.health.json" || { echo "[teb] no /health from: $HEALTH"; exit 1; }
SLOTS=$(field "$OUT/batch.health.json" "(d.get('parallel') or {}).get('max_running') or 1")
[ "${SLOTS:-0}" -ge 1 ] || { echo "[teb] the server reports no slot"; exit 1; }
{
  echo "date $(date -u +%FT%TZ)"
  echo "base_url $BASE"
  echo "bench $($BENCH --version 2>/dev/null | head -1)"
  echo "common ${COMMON[*]}"
  echo "server_version $(field "$OUT/batch.health.json" "d.get('version')")"
  echo "weights $(field "$OUT/batch.health.json" "(d.get('weights') or {}).get('name')")"
  echo "slots $SLOTS (bench --parallel 1)"
  echo "max_len $(field "$OUT/batch.health.json" "d.get('max_len')")"
  echo "default_max_tokens $(field "$OUT/batch.health.json" "d.get('default_max_tokens')")"
  echo "sampling $(field "$OUT/batch.health.json" "json.dumps(d.get('sampling'), sort_keys=True)")"
  echo "prefix_cache $(field "$OUT/batch.health.json" "(d.get('cache') or {}).get('prefix_cache')")"
  # no brace with a comma in these expressions: bash 3.2 expands it inside "$(...)"
  echo "response_cache $(field "$OUT/batch.health.json" "json.dumps(dict((k, v) for k, v in ((d.get('cache') or dict()).get('response_cache') or dict()).items() if k in ('budget', 'ttl_s')))")"
  if [ -n "$KNOBS" ]; then echo "knobs:"; bash -c "$KNOBS" 2>&1 | sed 's/^/  /'; fi
} > "$OUT/batch.knobs"
cat "$OUT/batch.knobs"

for row in "${LIST[@]}"; do
  label=${row%%:*}; flags=${row#*:}
  # the flags as the shell would split them, quotes kept (a JSON value is one word)
  eval "args=($flags)"
  # bash 3.2 (macOS) under set -u calls an empty array unbound: a row with no flags (`cond`) needs the guards
  echo "[teb] $label: ${args[*]-}"
  health "$OUT/$label.health.before.json"
  date -u +%FT%TZ > "$OUT/$label.start"
  printf '%s\n' "$flags" > "$OUT/$label.flags"
  # from the results folder: the bench keeps its history (data/benchmarks.sqlite) where it runs
  ( cd "$OUT" && $BENCH run "${COMMON[@]}" --label "$label" --json-file "$OUT/$label.json" \
    --output-dir "$OUT/reports" ${args[@]+"${args[@]}"} ) > "$OUT/$label.log" 2>&1
  echo "exit $?" >> "$OUT/$label.log"
  date -u +%FT%TZ > "$OUT/$label.end"
  health "$OUT/$label.health.after.json"
  hits=$(python3 - "$OUT/$label.health.before.json" "$OUT/$label.health.after.json" <<'PY'
import json, sys
try:
    a, b = (json.load(open(p))["cache"]["response_cache"]["hits"] for p in sys.argv[1:3])
    print(b - a)
except Exception:
    print("unknown")
PY
)
  echo "response_cache_hits $hits" >> "$OUT/$label.log"
  [ "$hits" = "0" ] || echo "[teb] $label: $hits answers came from the response cache; its timings are not the engine's"
done
echo "[teb] done: python3 tools/teb_summary.py $OUT"
