#!/bin/bash
# Make the access tokens for the :8000 service, once:
#
#   bash ops/make-secrets.sh                    creates secrets.env (mode 600) with the metrics token
#                                               and the chat's forward key
#   bash ops/make-secrets.sh --admin            the same, plus the admin password
#   bash ops/make-secrets.sh --rotate [--admin] replaces the file (every dashboard session ends);
#                                               an admin password already there stays one
#
# QSE_CHAT_FORWARD_KEY (OPS-103) is the key the engine server's /v1/chat-app forward sends to chatd
# (server/chat_forward.py); both read it from here. A file made before it gets the line added.
#
# QSE_METRICS_TOKEN is for Prometheus (the k8s Secret `qse-metrics-token`). QSE_ADMIN_TOKEN is the
# admin password: with it set, every admin route (the conductor's writes, the memory, the cache
# routes) needs it, and the dashboard, which has no sign-in screen yet, can no longer edit. Without
# it every admin route is open, so the server belongs on a trusted network (docs/server.md "The
# admin rule"). 32 random bytes each, hex. The values are never printed: read them with `cat` on the
# box when a client needs one. ops/start.sh sources the file.
#
# The file is the one the engine reads (OPS-153): QSE_SECRETS, else serve.env's QSE_SECRETS beside
# this script (an install's DIR/state/secrets.env; the setup guide's apply job runs this script
# without the engine's environment), else $QSE_STATE_DIR/secrets.env, else
# ~/.qwen38-spark-engine/secrets.env.
set -eu
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -z "${QSE_SECRETS:-}" ] && [ -f "$HERE/serve.env" ]; then
    QSE_SECRETS="$(set +eu; . "$HERE/serve.env" >/dev/null 2>&1; printf '%s' "${QSE_SECRETS:-}")"
fi
FILE="${QSE_SECRETS:-${QSE_STATE_DIR:+$QSE_STATE_DIR/secrets.env}}"
FILE="${FILE:-$HOME/.qwen38-spark-engine/secrets.env}"
DIR="$(dirname "$FILE")"
ROTATE=0; ADMIN=0
for a in "$@"; do
    case "$a" in
        --rotate) ROTATE=1 ;;
        --admin) ADMIN=1 ;;
        *) echo "[secrets] unknown argument $a (--rotate, --admin)"; exit 2 ;;
    esac
done
tok() { head -c 32 /dev/urandom | od -An -tx1 | tr -d ' \n'; }
if [ -f "$FILE" ] && [ $ROTATE = 0 ]; then
    if ! grep -q "^QSE_CHAT_FORWARD_KEY=." "$FILE"; then
        echo "QSE_CHAT_FORWARD_KEY=$(tok)" >> "$FILE"
        echo "[secrets] added QSE_CHAT_FORWARD_KEY to $FILE; restart the engine and chatd to use it"
        exit 0
    fi
    echo "[secrets] $FILE exists; --rotate replaces it"; exit 0
fi
[ -f "$FILE" ] && grep -q "^QSE_ADMIN_TOKEN=." "$FILE" && ADMIN=1
umask 077
mkdir -p "$DIR"
TMP="$FILE.tmp.$$"
{
    echo "# access tokens for the qwen38-spark-engine :8000 service; made $(date -Iseconds)"
    [ $ADMIN = 1 ] && echo "QSE_ADMIN_TOKEN=$(tok)"
    echo "QSE_METRICS_TOKEN=$(tok)"
    echo "QSE_CHAT_FORWARD_KEY=$(tok)"
} > "$TMP"
chmod 600 "$TMP"
mv "$TMP" "$FILE"
echo "[secrets] wrote $FILE (mode 600, admin password $([ $ADMIN = 1 ] && echo set || echo 'not set: every admin route is open')); restart the service to use it"
