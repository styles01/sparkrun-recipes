#!/bin/bash
# The proxy's combined end-to-end pass, for a quiet hour: every piece of P1 together.
#
#   bash tools/proxy_e2e.sh [--engine-url http://<engine host>:8000/v1] [--out DIR] [--box]
#                           [--engines]
#
#   1 unit       tests/test_proxy_{config,adapters,relay,harness}.py (the fake backends, CPU,
#                seconds; harness: P4's switches, notes and replay, E1, the checks, the guards)
#   2 bench      tools/proxy_bench.py: the proxy's own cost (added TTFT, proxy_ms) and 64 streams
#                at once to a fake backend (the backend must see all 64 at once)
#   3 engine     tools/proxy_gate.py against a live TandemLLM engine (20 small raw requests;
#                skipped without --engine-url): tokens equal, decode within 2 %, proxy_ms < 2 ms;
#                tools/proxy_harness_check.py, the harness (P4) in front of the same engine:
#                the class question through it, the note and its replay, a title capped
#                tools/loose_ends_live.py: a streamed call cut by the token limit reaches the
#                client once (no block text as content); a streamed turn behind a running one
#                gets `: waiting` comments and its answer, no 429
#   4 dashboard  vitest, the size budget, and the Playwright suite on the mock (backends, shell,
#                usage, dev, performance, system) on chromium desktop and webkit at 390 px,
#                one worker
#   5 standalone `python -m proxy` from an env file alone, two fake backends: /v1/models, a
#                stream, a request with a fallback, the dashboard's endpoints against the contract
#   6 box        (--box, run on the engine's host) tests/test_proxy_engine.py: the engine's own
#                server with --backends and the fake engine; tools/hostcollect_check.py: the
#                host collector against /proc/meminfo and nvidia-smi, the process behind :8000,
#                its own CPU cost
#   P1.5 + P5    tests/test_hostcollect.py and tests/test_package.py with the unit steps, and
#                tools/package_check.sh: the wheel built and installed into a clean venv without
#                torch, `tandem-dash check` and `serve` against a fake backend (needs pip's
#                network); the docker image builds in CI only (ops/Dockerfile.dash)
#   P6           tests/test_proxy_plugins.py (the vLLM and SGLang plugins against stand-ins shaped
#                as v0.30.0 and v0.5.21 define their stats and hooks) and tests/test_proxy_collect.py
#                (the collect route, the join, the launcher) with the unit steps; with --box
#                --engines the real engines run with the plugins inside (below)
#   P2 + P3      tests/test_proxy_engines.py with the unit steps (the vLLM and SGLang name tables
#                on pages shaped as v0.30.0 and v0.5.21 send them, the poller, the per-request
#                counts) and engines.spec.ts with the dashboard's e2e; with --box --engines,
#                tools/proxy_engines_check.sh: a small model in vLLM, then in SGLang, beside the
#                live engine (only when MemAvailable allows, one at a time under the box lock),
#                the adapters checked against each real server, the containers removed after
#
# Every step writes its log to DIR (default results/proxy-e2e-<time>) and one line to
# DIR/summary.txt; the exit code is the number of failed steps. Light on purpose: one worker, no
# GPU, no paid call.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-python3}
ENGINE=""; OUT=""; BOX=0; ENGINES=0
while [ $# -gt 0 ]; do
  case "$1" in
    --engine-url) ENGINE="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --box) BOX=1; shift ;;
    --engines) ENGINES=1; shift ;;
    *) echo "unknown argument $1"; exit 2 ;;
  esac
done
OUT=${OUT:-results/proxy-e2e-$(date +%Y%m%d-%H%M)}
mkdir -p "$OUT"
: > "$OUT/summary.txt"
FAILS=0
step() {                                   # step <name> <command...>
  local name="$1"; shift
  local t0=$(date +%s)
  if "$@" > "$OUT/$name.log" 2>&1; then r=PASS; else r=FAIL; FAILS=$((FAILS+1)); fi
  echo "$r $name $(( $(date +%s) - t0 ))s" | tee -a "$OUT/summary.txt"
}

step unit-config   $PY tests/test_proxy_config.py
step unit-adapters $PY tests/test_proxy_adapters.py
step unit-relay    $PY tests/test_proxy_relay.py
step unit-hostcollect $PY tests/test_hostcollect.py
step unit-package  $PY tests/test_package.py
step unit-engines  $PY tests/test_proxy_engines.py
step unit-harness  $PY tests/test_proxy_harness.py
step unit-plugins  $PY tests/test_proxy_plugins.py
step unit-collect  $PY tests/test_proxy_collect.py
step package       bash tools/package_check.sh --extras host --out "$OUT/package"
step bench         $PY tools/proxy_bench.py --n 200 --streams 64 --out "$OUT/bench.json"
if [ -n "$ENGINE" ]; then
  step engine-gate $PY tools/proxy_gate.py --url "$ENGINE" --n 20 --out "$OUT/gate.json"
  step engine-harness $PY tools/proxy_harness_check.py --url "$ENGINE" --out "$OUT/harness.json"
  # 2026-10-05: a live call cut by the token limit is sent once; a turn behind a turn waits
  # with `: waiting` comments and is served, never a 429
  step engine-loose-ends $PY tools/loose_ends_live.py --base "${ENGINE%/v1}" --out "$OUT/loose.json"
else
  echo "SKIP engine-gate (no --engine-url)" | tee -a "$OUT/summary.txt"
fi
if [ -d dashboard/node_modules ]; then
  step dash-vitest bash -c "cd dashboard && npx vitest run"
  step dash-build  bash -c "cd dashboard && npm run build && npm run test:size"
  step dash-e2e    bash -c "cd dashboard && npx playwright test e2e/backends.spec.ts e2e/shell.spec.ts e2e/usage.spec.ts e2e/dev.spec.ts e2e/performance.spec.ts e2e/system.spec.ts e2e/engines.spec.ts --project=chromium-desktop --project=webkit-phone --workers=1"
else
  echo "SKIP dashboard (no dashboard/node_modules: npm ci in dashboard/)" | tee -a "$OUT/summary.txt"
fi
step standalone $PY - "$OUT" <<'EOF'
import json, os, subprocess, sys, tempfile, time, urllib.request
sys.path[:0] = [".", "tests"]
import fake_backend
from tools import contract_check
out = sys.argv[1]
a, pa = fake_backend.start(models=["fake-a"], refuse=503)
b, pb = fake_backend.start(models=["fake-a", "fake-b"], name="b")
d = tempfile.mkdtemp(prefix="px-e2e-")
env_file = os.path.join(d, "proxy.env")
with open(env_file, "w") as f:
    f.write(f"TANDEM_BACKENDS=a,b\nTANDEM_BACKEND_A_KIND=generic\nTANDEM_BACKEND_A_URL=http://127.0.0.1:{pa}\n"
            f"TANDEM_BACKEND_A_FALLBACK=b\nTANDEM_BACKEND_B_KIND=generic\nTANDEM_BACKEND_B_URL=http://127.0.0.1:{pb}\n"
            "TANDEM_ROUTE_1=fake-a=a\nTANDEM_PRICE_1=b/fake-a=1,0.1,2\nTANDEM_SECRETS_KEY=a2tra2tra2tra2tra2tra2tra2tra2tra2tra2tra2s=\n")
port = 18400 + os.getpid() % 500
env = dict(os.environ, QSE_TEST="1")
p = subprocess.Popen([sys.executable, "-m", "proxy", "--env-file", env_file, "--port", str(port),
                      "--config-db", os.path.join(d, "config.sqlite3"),
                      "--usage-ledger", os.path.join(d, "ledger.sqlite3"), "--no-keychain"],
                     env=env, stdout=open(os.path.join(out, "standalone-server.log"), "w"),
                     stderr=subprocess.STDOUT)
base = f"http://127.0.0.1:{port}"
def get(path):
    return json.loads(urllib.request.urlopen(base + path, timeout=10).read())
def post(path, body):
    r = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=30) as resp:
        return resp.status, dict(resp.headers), resp.read()
try:
    for _ in range(50):
        try:
            get("/health"); break
        except OSError:
            time.sleep(0.2)
    time.sleep(1.5)                                    # the first health poll lists the models
    ids = sorted(m["id"] for m in get("/v1/models")["data"])
    assert ids == ["a/fake-a", "b/fake-a", "fake-b"], ids
    st, h, body = post("/v1/chat/completions", {"model": "fake-a", "stream": True,
                                                "messages": [{"role": "user", "content": "hi"}]})
    assert st == 200 and h.get("X-Tandem-Backend") == "b" and b"[DONE]" in body, (st, h)
    time.sleep(1.5)
    for name, path in (("summary", "/v1/dashboard/summary"), ("requests", "/v1/dashboard/requests"),
                       ("usage", "/v1/dashboard/usage?bucket=day"), ("system", "/v1/dashboard/system")):
        errs = contract_check.validate(get(path), contract_check.load(name))
        assert not errs, (name, errs[:3])
    row = get("/v1/dashboard/requests")["requests"][0]
    assert row["backend"]["id"] == "b" and row["fallback_from"] == "a" and row["cost_micros"], row
    print("standalone: PASS", ids, row["proxy_ms"])
finally:
    p.terminate(); p.wait(10); a.shutdown(); b.shutdown()
EOF
if [ $BOX = 1 ]; then
  step box-engine $PY tests/test_proxy_engine.py
  # P1.5: the collector against /proc/meminfo and nvidia-smi on this machine, the engine's
  # process behind :8000 with its GPU memory, and the collector's own CPU cost
  step box-hostcollect $PY tools/hostcollect_check.py --port 8000 --samples 60 \
    --out "$OUT/hostcollect.json"
  if [ $ENGINES = 1 ]; then
    # P2, P3: real vLLM and SGLang servers with a small model, one at a time, only when memory
    # allows (the script skips an engine rather than squeeze it in)
    # P6: with the plugins inside both (tools/proxy_engines_check.sh --plugins): the batches,
    # the joined rows, the plugin's sums against each engine's /metrics (1 %), and on vLLM the
    # plugin's cost, decode tok/s with and without it (off, on, off; within 1 %)
    step box-engines bash tools/proxy_engines_check.sh --out "$OUT/engines" --plugins
  fi
fi
echo "failed steps: $FAILS (logs in $OUT)" | tee -a "$OUT/summary.txt"
exit $FAILS
