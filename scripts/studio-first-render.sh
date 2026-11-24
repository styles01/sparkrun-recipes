#!/usr/bin/env bash
# Boot the Studio web app and prove it renders one image.
set -uo pipefail
cd "$HOME/imagination-studio" || exit 1
export PATH="$HOME/.local/bin:$PATH"

echo "=== $(date -u) studio first render ==="

# 1. app up
nohup setsid bash -c 'cd "$HOME/imagination-studio" && bash scripts/app.sh' > "$HOME/studio-app.log" 2>&1 < /dev/null &
for i in $(seq 1 40); do
  if curl -s -m4 http://127.0.0.1:7870/health 2>/dev/null | grep -qi "ok\|healthy" \
     || curl -s -m4 -o /dev/null -w "%{http_code}" http://127.0.0.1:7870/ 2>/dev/null | grep -q "200"; then
    echo "APP UP at $(date -u)"; break
  fi
  sleep 5
done
echo "app http: $(curl -s -m5 -o /dev/null -w '%{http_code}' http://127.0.0.1:7870/ 2>/dev/null)"
echo "comfy http: $(curl -s -m5 -o /dev/null -w '%{http_code}' http://127.0.0.1:8190/ 2>/dev/null)"

# 2. one image render (the acceptance bar)
echo "--- image render ---"
timeout 900 bash scripts/image.sh "a lighthouse on a rocky coast at dusk, storm clouds, cinematic" "$HOME/studio-test-render.png" 2>&1 | tail -15

echo "--- result ---"
ls -la "$HOME/studio-test-render.png" 2>/dev/null && file "$HOME/studio-test-render.png" 2>/dev/null
echo "=== done $(date -u) ==="
