#!/usr/bin/env bash
# Install + start the attribution gateway on spark1 as a user systemd unit (survives logout, restarts on failure).
set -euo pipefail
DEST=/home/jaita/sparkgateway
mkdir -p "$DEST"
install -m 0755 "$(dirname "$0")/sparkdash-llm-gateway.py" "$DEST/sparkdash-llm-gateway.py"

cat > "$DEST/env" <<'EOF'
UPSTREAM=http://127.0.0.1:8000
LISTEN=0.0.0.0
PORT=8001
LOG_PATH=/home/jaita/sparkgateway/requests.jsonl
UPSTREAM_API_KEY=
EOF

mkdir -p ~/.config/systemd/user
cat > ~/.config/systemd/user/sparkdash-llm-gateway.service <<'EOF'
[Unit]
Description=sparkdash LLM attribution gateway (:8001 -> :8000)
After=network-online.target

[Service]
EnvironmentFile=/home/jaita/sparkgateway/env
ExecStart=/usr/bin/python3 /home/jaita/sparkgateway/sparkdash-llm-gateway.py
Restart=always
RestartSec=3

[Install]
WantedBy=default.target
EOF

systemctl --user daemon-reload
systemctl --user enable --now sparkdash-llm-gateway.service
sleep 2
systemctl --user is-active sparkdash-llm-gateway.service
ss -ltn | grep -E ":8001" || echo "NOT LISTENING"
