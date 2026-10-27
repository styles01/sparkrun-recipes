#!/bin/bash
# The Setup Guide's release gate (OPS-177): a fresh scratch install runs the whole Setup Guide flow on
# the box before every release, beside the live engine and never touching it.
#
#   bash tools/release_gate_setup.sh [--keep] [--no-browser] [--py PYTHON] [--load-s N]
#
# Runs, each on a scratch install of this checkout (install.sh --setup into a HOME of its own under
# ~/gate-setup-<utc>, spare ports, QSE_CHATD=off, the CPU fake engine instead of a model, a few KB of
# stand-in weights instead of a download):
#
#   A  install --setup, setup mode answers; the Setup Guide's apply through its API, followed by
#      WebKit at 390 px in the Playwright container: the first try stops at the preflight (this script
#      breaks run/ops/stop.sh and puts it back once the job failed), the failed screen says so in
#      plain words and Copy diagnostics works; Retry runs again without a download; the restart card
#      shows while the fake engine loads, the finished screen with Start chatting within 10 s of the
#      engine answering; the weights step's Remove button removes a second profile's weights; then
#      the job is done, /health names the profile, a chat answers, the re-download line is written,
#      and removing the running profile is refused (screenshots of each screen)
#   B  the same install with the execute bit stripped from src/ops/start.sh, run/ops/start.sh,
#      run/ops/stop.sh and run/ops/use-profile.sh first: the apply must still end done
#
# The live box is not touched: the scratch HOME has its own box lock, conductor, ledger, chatd data
# and secrets; its engines run from a copy of the interpreter named tandem-gate-py, which the live
# box's ops/engines.sh (python* executables running server/app.py) does not count, while the scratch's
# own scripts see them through ENGINE_PROC, a process table this script keeps. Every process it
# started is stopped at the end; --keep leaves the scratch folders for a look. The report, with every
# command, its output and the screenshots, goes to results/gate-setup/<utc>/report.md (not under
# version control). Exit 0 only when every run passes. docs/operations.md: mandatory before a release.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_TREE="$(cd "$HERE/.." && pwd)"
KEEP=0; BROWSER=1; LOAD_S=20; BASEPY=""
while [ $# -gt 0 ]; do
    case "$1" in
        --keep) KEEP=1; shift ;;
        --no-browser) BROWSER=0; shift ;;
        --py) BASEPY="$2"; shift 2 ;;
        --load-s) LOAD_S="$2"; shift 2 ;;
        -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
        *) echo "[gate] unknown argument $1"; exit 2 ;;
    esac
done
[ "$(uname -s)" = Linux ] || { echo "[gate] Linux only (the box)"; exit 2; }
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
GATE="$HOME/gate-setup-$STAMP"
OUT="$SRC_TREE/results/gate-setup/$STAMP"
mkdir -p "$GATE" "$OUT"
REPORT="$OUT/report.md"
PIDS_TO_STOP=()
FAILS=0

say() { echo "[gate $(date +%H:%M:%S)] $*" | tee -a "$REPORT"; }
section() { printf '\n## %s\n\n' "$*" >> "$REPORT"; echo "[gate] == $*"; }
# run a command, its output into the report (and its last lines on the screen); returns its status
cmd() {
    printf '\n```\n$ %s\n' "$*" >> "$REPORT"
    "$@" > "$GATE/last.out" 2>&1
    local rc=$?
    cat "$GATE/last.out" >> "$REPORT"
    printf '(exit %s)\n```\n' "$rc" >> "$REPORT"
    tail -3 "$GATE/last.out" | sed 's/^/    /'
    return "$rc"
}
check() {   # check NAME CONDITION-COMMAND...
    local name="$1"; shift
    if "$@" >> "$REPORT" 2>&1; then say "PASS  $name"; else say "FAIL  $name"; FAILS=$((FAILS + 1)); fi
}
free_port() {
    local p
    for p in $(shuf -i 18100-18900 -n 200); do
        case "$p" in 8000|8009|8016) continue ;; esac
        ss -ltn 2>/dev/null | awk '{print $4}' | grep -qE "[:.]($p|$((p + 9)))\$" && continue
        echo "$p"; return 0
    done
    return 1
}

# ---------------------------------------------------------------- the interpreter
# a copy of a Python 3.11 that has torch (the fake engine runs on its CPU), as a venv of its own
# named tandem-gate-py: a scratch engine never reads as one of the live box's engines
if [ -z "$BASEPY" ]; then
    for c in "$HOME/TandemLLM/venv/bin/python" python3.11; do
        if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys, torch, yaml; sys.exit(sys.version_info[:2] != (3, 11))' 2>/dev/null; then
            BASEPY="$c"; break
        fi
    done
fi
[ -n "$BASEPY" ] || { echo "[gate] no Python 3.11 with torch and yaml found: pass --py"; exit 2; }
REALPY="$(readlink -f "$(command -v "$BASEPY")")"
SITE="$("$BASEPY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
"$REALPY" -m venv --without-pip --copies "$GATE/pyenv" || { echo "[gate] venv failed"; exit 2; }
cp "$GATE/pyenv/bin/$(basename "$REALPY")" "$GATE/pyenv/bin/tandem-gate-py"
echo "$SITE" > "$("$GATE/pyenv/bin/tandem-gate-py" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')/gate-base.pth"
GATEPY="$GATE/pyenv/bin/tandem-gate-py"
"$GATEPY" -c 'import torch, yaml' || { echo "[gate] $GATEPY does not import torch"; exit 2; }

# ---------------------------------------------------------------- the release and the stand-ins
section "Release"
REL="$GATE/release"
mkdir -p "$REL"
( cd "$SRC_TREE" && { git ls-files 2>/dev/null || find . -type f -not -path './.git/*' | sed 's|^\./||'; } \
    | grep -vE '^(run|logs|train|results|dashboard/node_modules)/' | tar -cf - -T - ) | tar -xf - -C "$REL"
git -C "$REL" init -q && git -C "$REL" add -A && \
    git -C "$REL" -c user.name=gate -c user.email=gate@example.invalid commit -qm "release under test" || exit 2
REF="$(git -C "$REL" rev-parse HEAD)"
say "release $(cat "$REL/VERSION") from $SRC_TREE ($(git -C "$SRC_TREE" rev-parse --short HEAD 2>/dev/null || echo 'no git')), commit $REF"
STUBS="$GATE/stubs"
mkdir -p "$STUBS/bin" "$STUBS/cuda/bin" "$GATE/fakeexe"
cat > "$STUBS/bin/nvidia-smi" <<'EOF'
#!/bin/bash
case "$*" in
  -L) echo "GPU 0: NVIDIA GB10 (UUID: GPU-00000000-0000-0000-0000-000000000000)" ;;
  *query-gpu=name,driver_version,compute_cap,memory.total*) echo "NVIDIA GB10, 580.95.05, 12.1, [N/A]" ;;
  *compute_cap*) echo "12.1" ;;
  *) exit 0 ;;
esac
EOF
printf '#!/bin/bash\necho "Cuda compilation tools, release 13.0, V13.0.88"\n' > "$STUBS/cuda/bin/nvcc"
printf '#!/bin/bash\n[ "$1" = --version ] && echo "uv 0.12.12"\nexit 0\n' > "$STUBS/bin/uv"
printf '#!/bin/bash\nexit 0\n' > "$STUBS/bin/g++"
# huggingface.co's model API answers 200 (no network for the checks); everything else is the real curl
printf '#!/bin/bash\ncase "$*" in *huggingface.co/api/models/*) printf 200; exit 0 ;; esac\nexec %s "$@"\n' "$(command -v curl)" > "$STUBS/bin/curl"
printf '#!/bin/sh\nexit 0\n' > "$GATE/fakeexe/python3"
chmod 755 "$STUBS/bin/"* "$STUBS/cuda/bin/nvcc" "$GATE/fakeexe/python3"

# ---------------------------------------------------------------- the process table
# ENGINE_PROC/<pid> for each scratch process of tandem-gate-py: cmdline is the process's own (gone
# with it), exe a name ops/engines.sh takes for Python
PROC="$GATE/proc"
mkdir -p "$PROC"
(
    exec 0</dev/null >/dev/null 2>&1
    while [ -d "$PROC" ]; do
        for pid in $(pgrep -f -- "$GATEPY"); do
            [ -e "$PROC/$pid/cmdline" ] && continue
            mkdir -p "$PROC/$pid" && ln -sfn "/proc/$pid/cmdline" "$PROC/$pid/cmdline" \
                && ln -sfn "$GATE/fakeexe/python3" "$PROC/$pid/exe"
        done
        sleep 0.3
    done
) &
KEEPER=$!

cleanup() {
    section "Cleanup"
    for d in "$GATE"/run-*/home/TandemLLM; do
        [ -d "$d" ] || continue
        env -i HOME="$(dirname "$d")" PATH="$STUBS/bin:/usr/local/bin:/usr/bin:/bin" ENGINE_PROC="$PROC" \
            bash "$d/bin/tandem" stop 20 >> "$REPORT" 2>&1
        SP="$(sed -n 's/^QSE_STATUS_PORT=//p' "$d/run/ops/serve.env" | tail -1 | tr -dc 0-9)"
        [ -n "$SP" ] && QSE_STATUS_TREE="$d/src" python3 "$d/run/ops/engine_status.py" stop --port "$SP" >> "$REPORT" 2>&1
    done
    for p in "${PIDS_TO_STOP[@]}"; do kill "$p" 2>/dev/null; done
    # anything of the interpreter copy still alive: the scratch's alone
    for pid in $(pgrep -f -- "$GATEPY"); do kill "$pid" 2>/dev/null; done
    sleep 1
    for pid in $(pgrep -f -- "$GATEPY"); do kill -9 "$pid" 2>/dev/null; done
    rm -rf "$PROC"
    kill "$KEEPER" 2>/dev/null
    if [ "$KEEP" = 1 ]; then say "kept $GATE"; else rm -rf "$GATE"; fi
}
trap cleanup EXIT

# ---------------------------------------------------------------- one scratch install
# install NAME PORT -> sets DIR, H, ENVS (the environment the box's own commands run in)
install_box() {
    local name="$1" port="$2"
    H="$GATE/run-$name/home"; DIR="$H/TandemLLM"
    mkdir -p "$DIR/venv/bin" "$H/.qwen38-spark-engine/models/qwen3.5-2b"
    cat > "$DIR/venv/bin/python" <<EOF
#!/bin/bash
if [ "\$1" = -c ] && [[ "\$2" == *version_info* ]]; then exit 0; fi
if [ "\$1" = - ]; then
    s="\$(cat)"
    case "\$s" in *"import torch"*) echo "    ok torch (the gate's CPU interpreter)"; exit 0 ;; esac
    shift; exec $GATEPY -c "\$s" "\$@"
fi
exec $GATEPY "\$@"
EOF
    chmod 755 "$DIR/venv/bin/python"
    echo '{}' > "$H/.qwen38-spark-engine/models/qwen3.5-2b/config.json"
    echo '{}' > "$H/.qwen38-spark-engine/models/qwen3.5-2b/tokenizer.json"
    ENVS=(HOME="$H" PATH="$STUBS/bin:/usr/local/bin:/usr/bin:/bin" CUDA_HOME="$STUBS/cuda" LANG=C.UTF-8
          TANDEM_SKIP_MEMCHECK=1)
    cmd env -i "${ENVS[@]}" bash "$REL/install.sh" --setup --no-start --yes --no-corpus --dir "$DIR" \
        --repo "$REL" --ref "$REF" --port "$port" --host 127.0.0.1 || return 1
    # the scratch's own choices, read last by serve.env: no chatd (never a second one beside the live
    # one, SRV-524), the fake engine with a load of LOAD_S seconds, its process table, the memory the
    # switch waits for (the live engine holds most of the board)
    cat >> "$DIR/run/local.env" <<EOF
QSE_CHATD=off
QSE_ENGINE_FLAGS="--fake-engine --fake-load-s $LOAD_S"
ENGINE_PROC=$PROC
MEM_NEED_GB=1
EOF
    # the weights of two profiles and every file they read beside them: a few KB of stand-ins
    "$GATEPY" - "$DIR/src" "$H" <<'EOF'
import os, sys
src, home = sys.argv[1], sys.argv[2]
sys.path.insert(0, src)
from server.setup import registry
m = registry.load_all(os.path.join(src, "models"))[0]["qwen3.8-27b"]
for pid in ("exl3-2.5", "exl3-3.0"):
    for aid in registry.profile(m, pid)["assets_required"]:
        a = registry.asset(m, aid)
        for p in [a["path"], *a.get("also", [])]:
            p = p.replace("${HOME}", home).replace("${REPO}", src)
            if p.endswith((".json", ".safetensors")):
                os.makedirs(os.path.dirname(p), exist_ok=True)
                open(p, "w").write("{}")
            else:
                os.makedirs(p, exist_ok=True)
    d = os.path.join(home, "exl3-quants", pid.split("-")[1])
    os.makedirs(d, exist_ok=True)
    for f in ("config.json", "model.safetensors"):
        open(os.path.join(d, f), "w").write("{}" if f.endswith(".json") else "w" * 4096)
print("stand-in weights for exl3-2.5 and exl3-3.0 and their assets")
EOF
}

api() {   # api METHOD PATH [JSON]: the answer on stdout, the status in $CODE
    local out
    out="$(curl -s -m 30 -o "$GATE/api.json" -w '%{http_code}' -X "$1" -H 'Content-Type: application/json' \
        ${3:+-d "$3"} "http://127.0.0.1:$PORT$2")"
    CODE="$out"; cat "$GATE/api.json"
}
job_state() {   # the job's state, through the engine or, while it restarts, the status responder
    local j
    j="$(curl -s -m 5 "http://127.0.0.1:$PORT/v1/setup/jobs/$1?format=json" 2>/dev/null)"
    printf '%s' "$j" | "$GATEPY" -c 'import json,sys; print(json.load(sys.stdin)["state"])' 2>/dev/null && return
    j="$(curl -s -m 5 "http://127.0.0.1:$((PORT + 9))/v1/setup/jobs/$1?format=json" 2>/dev/null)"
    printf '%s' "$j" | "$GATEPY" -c 'import json,sys; print(json.load(sys.stdin)["state"])' 2>/dev/null || echo unknown
}
wait_job() {   # wait_job ID -> the final state on stdout
    local s t0; t0=$(date +%s)
    while [ $(( $(date +%s) - t0 )) -lt 600 ]; do
        s="$(job_state "$1")"
        case "$s" in done|failed|rolled_back) echo "$s"; return 0 ;; esac
        sleep 2
    done
    echo "timeout"
}
# the fixture's selection: the scratch's own port and address (the plan refuses 8000, held by the live
# engine), no tool with a model of its own, no conductor (a fake engine may not use one under HOME)
selection() {
    printf '{"model": "qwen3.8-27b", "profile": "exl3-2.5", "components": {"conductor": false, "narrator": false, "laya": false, "shadow": false, "proxy": false}, "advanced": {"port": %s, "host": "127.0.0.1"}}' "$PORT"
}

start_setup_mode() {
    cmd env -i "${ENVS[@]}" bash "$DIR/bin/tandem" start || return 1
    check "setup mode answers /health with mode setup" bash -c \
        "curl -s -m 5 http://127.0.0.1:$PORT/health | grep -q '\"mode\": \"setup\"'"
    check "setup mode's /health names the status responder's port $((PORT + 9))" bash -c \
        "curl -s -m 5 http://127.0.0.1:$PORT/health | grep -q '\"status_port\": $((PORT + 9))'"
}
verify_live() {
    check "the job ended done" test "$1" = done
    check "/health names exl3-2.5" bash -c "curl -s -m 5 http://127.0.0.1:$PORT/health | grep -q '\"name\": \"exl3-2.5\"'"
    local model
    model="$(curl -s -m 5 "http://127.0.0.1:$PORT/v1/models" | "$GATEPY" -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null)"
    check "a chat answers ($model)" bash -c "curl -s -m 60 http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' \
        -d '{\"model\": \"$model\", \"max_tokens\": 16, \"messages\": [{\"role\": \"user\", \"content\": \"Say ready\"}]}' \
        | grep -q '\"content\"'"
}

# ---------------------------------------------------------------- run A
section "Run A: install, the Setup Guide apply in the browser at 390 px, verify, remove"
PORT="$(free_port)" || { say "no free port"; exit 1; }
say "ports: engine $PORT, status responder $((PORT + 9)) (never 8000, 8009, 8016)"
if install_box a "$PORT" && start_setup_mode; then
    JOB=""
    if [ "$BROWSER" = 1 ]; then
        PW="$HOME/.cache/tandem-gate-pw"
        mkdir -p "$PW" "$OUT/shots"
        cp "$SRC_TREE/dashboard/e2e/setup-gate.spec.ts" "$SRC_TREE/dashboard/playwright.gate.config.ts" "$PW/"
        # as this user, so every file it writes is this user's (the report, the cache), HOME a scratch
        DU=(--user "$(id -u):$(id -g)" -e HOME=/tmp)
        [ -d "$PW/node_modules/@playwright/test" ] || cmd docker run --rm "${DU[@]}" --network host -v "$PW:/work" -w /work \
            mcr.microsoft.com/playwright:v1.63.0-noble npm install --no-save --no-audit --no-fund @playwright/test@1.63.0
        # the first try must fail at the preflight: a stop.sh that does not parse (VIS-213); put back
        # once the job says failed, which the spec waits for (shots/fixed.txt) before its Retry
        cp -p "$DIR/run/ops/stop.sh" "$GATE/stop.sh.good"
        printf '\nif then fi (\n' >> "$DIR/run/ops/stop.sh"
        (
            exec 0</dev/null >/dev/null 2>&1
            for _ in $(seq 1 600); do
                J="$(cat "$OUT/shots/first-job-id.txt" 2>/dev/null)"
                if [ -n "$J" ] && [ "$(job_state "$J")" = failed ]; then
                    cp -p "$GATE/stop.sh.good" "$DIR/run/ops/stop.sh" && echo fixed > "$OUT/shots/fixed.txt"
                    exit 0
                fi
                sleep 1
            done
        ) &
        PIDS_TO_STOP+=("$!")
        cmd docker run --rm "${DU[@]}" --network host --ipc=host -v "$PW:/work" -v "$OUT/shots:/shots" -w /work \
            -e GATE_URL="http://127.0.0.1:$PORT" -e GATE_SELECTION="$(selection)" -e GATE_SHOTS=/shots \
            mcr.microsoft.com/playwright:v1.63.0-noble npx playwright test -c playwright.gate.config.ts
        PWRC=$?
        check "the browser at 390 px: a failed try in plain words, Copy diagnostics, Retry without a download, the restart card, the finished screen within 10 s, Remove with its confirmation" test "$PWRC" = 0
        JOB="$(cat "$OUT/shots/job-id.txt" 2>/dev/null)"
        check "the first try failed at the preflight and nothing stopped" bash -c \
            "curl -s -m 5 'http://127.0.0.1:$PORT/v1/setup/jobs/$(cat "$OUT/shots/first-job-id.txt" 2>/dev/null)?format=json' | grep -q '\"failed_step\": \"preflight\"'"
    else
        api POST /v1/setup/apply "{\"selection\": $(selection)}" > "$GATE/apply.json"
        JOB="$("$GATEPY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["job"]["id"])' "$GATE/apply.json" 2>/dev/null)"
        [ -n "$JOB" ] || say "the apply answered: $(head -c 600 "$GATE/apply.json")"
    fi
    if [ -n "$JOB" ]; then
        STATE="$(wait_job "$JOB")"
        say "job $JOB: $STATE"
        verify_live "$STATE"
        api GET "/v1/setup/jobs/$JOB?format=json" >> "$REPORT"
        # the other profile's weights: removed by the browser's Remove button, else through the API
        if [ "$BROWSER" != 1 ]; then
            R="$(api POST /v1/setup/remove '{"profiles": ["exl3-3.0"], "confirm": true}')"
            echo "$R" >> "$REPORT"
        fi
        check "exl3-3.0 removed: its folder is gone" test ! -e "$H/exl3-quants/3.0"
        check "the re-download line is written" grep -q "exl3-3.0.*bash ops/use-profile.sh exl3-3.0 --fetch-only" "$DIR/src/run/weights-removed.txt"
        check "the running profile is refused" bash -c "curl -s -o /dev/null -w '%{http_code}' -X POST -H 'Content-Type: application/json' \
            -d '{\"profiles\": [\"exl3-2.5\"], \"confirm\": true}' http://127.0.0.1:$PORT/v1/setup/remove | grep -q 409"
    else
        say "FAIL  no job started"; FAILS=$((FAILS + 1))
    fi
else
    say "FAIL  run A did not install or start setup mode"; FAILS=$((FAILS + 1))
fi
env -i "${ENVS[@]}" ENGINE_PROC="$PROC" bash "$DIR/bin/tandem" stop 20 >> "$REPORT" 2>&1

# ---------------------------------------------------------------- run B
section "Run B: the same with the execute bit stripped from the scripts"
PORT="$(free_port)" || { say "no free port"; exit 1; }
say "ports: engine $PORT, status responder $((PORT + 9))"
if install_box b "$PORT"; then
    chmod -x "$DIR/src/ops/start.sh" "$DIR/run/ops/start.sh" "$DIR/run/ops/stop.sh" "$DIR/run/ops/use-profile.sh"
    say "stripped: $(stat -c '%a %n' "$DIR/src/ops/start.sh" "$DIR/run/ops/start.sh" "$DIR/run/ops/stop.sh" "$DIR/run/ops/use-profile.sh" | tr '\n' ' ')"
    if start_setup_mode; then
        api POST /v1/setup/apply "{\"selection\": $(selection)}" > "$GATE/apply.json"
        JOB="$("$GATEPY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["job"]["id"])' "$GATE/apply.json" 2>/dev/null)"
        if [ -n "$JOB" ]; then
            STATE="$(wait_job "$JOB")"
            say "job $JOB: $STATE"
            verify_live "$STATE"
            api GET "/v1/setup/jobs/$JOB?format=json" >> "$REPORT"
        else
            say "FAIL  no job started: $(head -c 600 "$GATE/apply.json")"; FAILS=$((FAILS + 1))
        fi
    else
        say "FAIL  setup mode did not start"; FAILS=$((FAILS + 1))
    fi
else
    say "FAIL  run B did not install"; FAILS=$((FAILS + 1))
fi

section "Result"
if [ "$FAILS" = 0 ]; then say "PASSED: every run"; else say "FAILED: $FAILS check(s)"; fi
echo "[gate] report: $REPORT"
[ "$FAILS" = 0 ]
