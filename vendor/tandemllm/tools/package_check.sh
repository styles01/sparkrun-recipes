#!/bin/bash
# The installable package, checked as a stranger would use it: build the wheel, install it into a
# clean venv with no torch, run `tandem-dash check`, start `tandem-dash serve` from an env file
# with a fake backend, and read /v1/models, the page and the System body.
#
#   bash tools/package_check.sh [--extras host] [--out DIR]
#
# Needs Python 3.11 or newer as $PY (default python3) and the network for pip (cryptography,
# setuptools; psutil and nvidia-ml-py with --extras host). Writes nothing outside a temp dir and
# DIR (default results/package-check-<time>). Exit 0 when every check passed.
set -u
cd "$(dirname "$0")/.."
REPO=$PWD
PY=${PY:-python3}
EXTRAS=""; OUT=""
while [ $# -gt 0 ]; do
  case "$1" in
    --extras) EXTRAS="[$2]"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    *) echo "unknown argument $1"; exit 2 ;;
  esac
done
OUT=${OUT:-results/package-check-$(date +%Y%m%d-%H%M)}
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
T=$(mktemp -d "${TMPDIR:-/tmp}/tandem-pkg-XXXXXX")
trap 'kill $SRV $FAKE 2>/dev/null; rm -rf "$T"' EXIT
SRV=""; FAKE=""
fail() { echo "FAIL $*" | tee -a "$OUT/summary.txt"; exit 1; }

# the source as git would ship it (no node_modules, no results), so the wheel sees what CI sees
mkdir -p "$T/src"
if git -C "$REPO" rev-parse >/dev/null 2>&1; then
  git -C "$REPO" ls-files -z --cached --others --exclude-standard \
    | (cd "$REPO" && xargs -0 tar -cf - 2>/dev/null) | tar -xf - -C "$T/src"
else
  cp -R "$REPO/." "$T/src/"
fi
$PY -m venv "$T/build" && "$T/build/bin/pip" -q install build >"$OUT/build.log" 2>&1 \
  || fail "the build venv"
"$T/build/bin/python" -m build --wheel "$T/src" -o "$T/wheel" >>"$OUT/build.log" 2>&1 \
  || fail "the wheel build (build.log)"
WHL=$(ls "$T"/wheel/*.whl)
echo "wheel $(basename "$WHL") $(du -k "$WHL" | cut -f1) KB" | tee "$OUT/summary.txt"
unzip -l "$WHL" > "$OUT/wheel-files.txt"
if grep -Eq "tests/|node_modules|\.safetensors|\.pt$|/src/.*\.ts$" "$OUT/wheel-files.txt"; then
  fail "the wheel holds tests, node_modules, weights or dashboard sources"
fi
grep -q "proxy/dashboard/index.html" "$OUT/wheel-files.txt" || fail "no built page in the wheel"

$PY -m venv "$T/venv" || fail "the clean venv"
"$T/venv/bin/pip" -q install "$WHL$EXTRAS" >"$OUT/install.log" 2>&1 || fail "pip install (install.log)"
"$T/venv/bin/pip" list --format=freeze > "$OUT/pip-freeze.txt"
if grep -Eiq "^(torch|numpy|transformers)==" "$OUT/pip-freeze.txt"; then
  fail "the install pulled torch, numpy or transformers"
fi
echo "PASS install without torch ($(wc -l < "$OUT/pip-freeze.txt") packages)" | tee -a "$OUT/summary.txt"

# a fake backend from the checkout's tests (not part of the wheel)
FPORT=$((19100 + $$ % 400)); PORT=$((19600 + $$ % 300))
$PY "$REPO/tests/fake_backend.py" --port $FPORT --models fake-a >"$OUT/fake.log" 2>&1 &
FAKE=$!
cat > "$T/dash.env" <<EOF
TANDEM_BACKENDS=fake
TANDEM_BACKEND_FAKE_KIND=generic
TANDEM_BACKEND_FAKE_URL=http://127.0.0.1:$FPORT/v1
TANDEM_SECRETS_KEY=a2tra2tra2tra2tra2tra2tra2tra2tra2tra2tra2s=
EOF
cd "$T"                                     # nowhere near the checkout: the wheel alone
DASH="$T/venv/bin/tandem-dash"
"$DASH" version | tee -a "$OUT/summary.txt"
"$DASH" check --env-file "$T/dash.env" --config-db "$T/data/config.sqlite3" >"$OUT/check.log" 2>&1 \
  || fail "tandem-dash check (check.log)"
grep -q "^page .*proxy/dashboard$" "$OUT/check.log" || fail "check did not find the packaged page"
[ -e "$T/data/config.sqlite3" ] && fail "check wrote the config database"
echo "PASS check" | tee -a "$OUT/summary.txt"
QSE_TEST=1 "$DASH" serve --env-file "$T/dash.env" --config-db "$T/data/config.sqlite3" \
  --usage-ledger "$T/data/ledger.sqlite3" --port $PORT --no-keychain >"$OUT/serve.log" 2>&1 &
SRV=$!
"$T/venv/bin/python" - "$PORT" "$OUT" <<'EOF' || fail "serve (serve.log, $OUT/serve-check.log)"
import json, sys, time, urllib.request
port, out = sys.argv[1], sys.argv[2]
base = f"http://127.0.0.1:{port}"
log = open(f"{out}/serve-check.log", "w")
def get(path, raw=False):
    with urllib.request.urlopen(base + path, timeout=10) as r:
        body = r.read()
        return (r.status, body) if raw else json.loads(body)
for _ in range(100):
    try:
        get("/health"); break
    except OSError:
        time.sleep(0.2)
time.sleep(1.5)
ids = [m["id"] for m in get("/v1/models")["data"]]
print("models", ids, file=log)
assert ids == ["fake-a"], ids
st, page = get("/dashboard/", raw=True)
assert st == 200 and b"<script" in page and b"not built" not in page, page[:200]
sysd = get("/v1/dashboard/system")
print("version", sysd["engine"]["version"], "hosts", [h["source"] for h in sysd.get("hosts", [])],
      file=log)
assert sysd["engine"]["version"], "the version from the package metadata"
assert sysd.get("hosts"), "the host collector answered"
r = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(
    {"model": "fake-a", "messages": [{"role": "user", "content": "hi"}]}).encode(),
    headers={"Content-Type": "application/json"})
with urllib.request.urlopen(r, timeout=30) as resp:
    assert resp.status == 200
print("ok", file=log)
EOF
echo "PASS serve: /v1/models, /dashboard/, system with hosts, one request" | tee -a "$OUT/summary.txt"
"$T/venv/bin/pip" list --format=freeze | grep -Eiq "^torch==" && fail "torch appeared"
echo "PASS all ($OUT)" | tee -a "$OUT/summary.txt"
