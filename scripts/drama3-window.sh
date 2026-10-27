#!/usr/bin/env bash
# DRAMA-3 WINDOW — spark2 (plotarmordev/drama-tensorfold engine, James 2026-10-20)
# Build + serve + verify. Runs ONLY while spark2's dsv41 GPU worker is down
# (tonight's Ornith window). Gates are hard; on any refuse -> log + exit.
set -uo pipefail
REPO=/home/jaita/drama-tensorfold
BUILD=$REPO/build/tensorfold
ZIG=/home/jaita/opt/zig-0.17.0/zig
CKPT=/home/jaita/MODELS/MEDIA/Drama3
LOCK=/home/jaita/drama3/gpu.lock
RUNTIME=/home/jaita/drama3/runtime
LOGSTAMP=$(date +%Y%m%d-%H%M)
EV=/home/jaita/drama3/evidence-$LOGSTAMP
OUT=/tmp/drama3-window.out
SUDO_PW=${DRAMA3_SUDO_PW:-}
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT"; }
say "=== DRAMA3 WINDOW START $LOGSTAMP ==="

# ---- GATE 1: spark2 dsv41 worker must be down; wait up to 40 min for MemAvailable >= 88GiB
for i in $(seq 1 80); do
  mem=$(grep MemAvailable /proc/meminfo | awk '{print int($2/1048576)}')
  serving=$(docker ps -q --filter name=dsv41-serve 2>/dev/null | wc -l)
  if [ "$mem" -ge 88 ] && [ "$serving" -eq 0 ]; then break; fi
  if [ $i -eq 80 ]; then say "REFUSE: MemAvailable ${mem}GiB / dsv41-serve containers $serving after 40min wait"; exit 2; fi
  sleep 30
done
say "GATE1 PASS: MemAvailable ${mem}GiB, dsv41-serve absent"

# ---- Linger: user systemd transient unit must survive the operator's ssh logout
loginctl enable-linger jaita 2>/dev/null && say "linger: enabled (self)" \
  || { echo "$SUDO_PW" | sudo -S -p "" loginctl enable-linger jaita 2>/dev/null && say "linger: enabled (sudo)" \
    || say "WARN: linger NOT enabled - pilot unit will die when the last jaita session closes"; }
loginctl show-user jaita 2>/dev/null | grep -i linger | tee -a "$OUT"

# ---- BUILD (CPU-only, nvcc inside nvcr 26.07 container; host has no nvcc)
cd "$REPO"
if [ ! -x "$BUILD/zig-out/native/bin/tensorfold-native" ]; then
  say "build: container build start (nice, ~10-30 min)"
  docker run --rm --name drama3-build \
    -v "$REPO":/work -v /home/jaita/opt:/opt/zigtool:ro \
    -e HOME=/tmp/zighome -e JOBS=4 \
    nvcr.io/nvidia/pytorch:26.07-py3 \
    bash -lc 'nice -n 10 bash -c "cd /work && ZIG=/opt/zigtool/zig-0.17.0/zig NVCC=/usr/local/cuda/bin/nvcc scripts/build-engine.sh" ' \
    >> /tmp/drama3-build.log 2>&1 && say "BUILD-OK" || { say "BUILD-FAILED (see /tmp/drama3-build.log tail below)"; tail -30 /tmp/drama3-build.log >> "$OUT"; exit 3; }
  chown -R jaita:jaita "$REPO/build" 2>/dev/null
else
  say "build: cached engine present, skipping"
fi
[ -x "$BUILD/zig-out/native/bin/tensorfold-native" ] || { say "REFUSE: native binary missing after build"; exit 3; }
say "NATIVE-BIN: $(sha256sum "$BUILD/zig-out/native/bin/tensorfold-native" | cut -c1-16)"

# ---- CHECKPOINT checksums were verified pre-window (gate done by operator); re-assert marker file
[ -f /tmp/drama-ck-out.txt ] && grep -q DRAMA3-CHECKPOINT-VERIFIED /tmp/drama-ck-out.txt \
  && say "checkpoint: pre-verified (marker present)" \
  || say "WARN: no checkpoint marker found - operator must confirm /tmp/drama-ck-out.txt"

# ---- SERVER via pilot.py (user systemd unit, :18793 loopback)
mkdir -p "$RUNTIME" "$EV"
say "server: pilot.py up"
cd "$REPO"
nohup python3 recipe/pilot.py \
  --native-bin "$BUILD/zig-out/native/bin/tensorfold-native" \
  --runtime-dir "$RUNTIME" \
  --checkpoint "$CKPT" \
  --gpu-lock "$LOCK" \
  --evidence-dir "$EV" \
  --port 18793 > /tmp/drama3-pilot.log 2>&1 &
echo $! > /tmp/drama3-pilot.pid
say "server: pid $(cat /tmp/drama3-pilot.pid) (pilot manages its own systemd unit + MemAvailable kill-guard)"

# ---- VERIFY (L8): health, whole WAV, stream, finish-reason
for i in $(seq 1 60); do
  curl -s -m 2 http://127.0.0.1:18793/health > /tmp/drama3-health.json 2>/dev/null && \
    grep -q '"ready"' /tmp/drama3-health.json 2>/dev/null && break
  sleep 5
  [ $i -eq 60 ] && { say "SERVER-NOT-READY in 5min"; tail -30 /tmp/drama3-pilot.log >> "$OUT"; exit 4; }
done
say "health: $(cat /tmp/drama3-health.json)"

T=/home/jaita/drama3/takes-$LOGSTAMP; mkdir -p "$T"
curl -sS -D /tmp/drama3-hdr.txt http://127.0.0.1:18793/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model": "fishaudio/drama-3", "input": "The train leaves at seven fifteen."}' -o "$T/hello.wav" \
  && say "whole: $(stat -c%s "$T/hello.wav") bytes, finish: $(grep -i x-finish-reason /tmp/drama3-hdr.txt | tr -d \\r)"
python3 - "$T/hello.wav" <<'EOF' >> "$OUT" 2>&1 || true
import wave, sys
w = wave.open(sys.argv[1]); print("wav: %ds frames %dHz dur %.2fs" % (w.getnframes(), w.getframerate(), w.getnframes()/w.getframerate()))
EOF
curl -sSN -D /tmp/drama3-hdr2.txt http://127.0.0.1:18793/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model": "fishaudio/drama-3", "input": "Spark lane live. All engines nominal.", "response_format": "pcm", "stream": true}' \
  -o "$T/hello-stream.pcm" && say "stream: $(stat -c%s "$T/hello-stream.pcm") bytes, trailer: $(grep -i x-finish-reason /tmp/drama3-hdr2.txt | tr -d \\r)"

say "=== DRAMA3-WINDOW-DONE (takes in $T) ==="