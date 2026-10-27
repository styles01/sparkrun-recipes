#!/bin/bash
# Removes what ops/sandbox/install.sh and chatd's sandboxes made. Containers and the image go with
# no question; the chats' files (workspaces) and the tandem-sbx user only after typing "delete".
#   sudo bash ops/sandbox/uninstall.sh          (docker-only setups: without sudo, step 1 alone)
set -u
ENGINE="${ENGINE:-docker}"
echo "1. containers and the image"
$ENGINE ps -aq --filter label=tandem.chat | xargs -r $ENGINE rm -f
$ENGINE rmi tandem-sandbox:1 2>/dev/null || true
[ "$(id -u)" = 0 ] || { echo "the rest needs sudo"; exit 0; }
nft delete table inet tandem_sbx 2>/dev/null; rm -f /etc/nftables.d/tandem-sbx.nft
read -r -p "Type delete to remove the user tandem-sbx and the chats' workspaces: " ok
if [ "$ok" = delete ]; then
    loginctl disable-linger tandem-sbx 2>/dev/null
    userdel -r tandem-sbx 2>/dev/null
    rm -rf "${QSE_CHAT_DATA:-/home/${SUDO_USER}/.tandemllm/chat}"/{workspaces,homes,snapshots,egress}
fi
echo "done"
