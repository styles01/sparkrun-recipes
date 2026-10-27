#!/bin/bash
# The plan's stronger sandbox runtime (OPS-103, Chat 6): rootless Podman under its own user
# `tandem-sbx`, so an escape from a container lands in a user that owns nothing. Needs root once:
#
#   sudo bash ops/sandbox/install.sh            then restart chatd with the line it prints
#
# What it does (each step is undone by ops/sandbox/uninstall.sh):
#   1. apt install podman crun passt uidmap (Ubuntu 24.04 ships AppArmor profiles for podman and
#      crun in /etc/apparmor.d, which allow their user namespaces under
#      kernel.apparmor_restrict_unprivileged_userns=1)
#   2. the user tandem-sbx (no login shell, no sudo), subuid/subgid ranges, linger, its
#      podman.socket user unit, and an ACL so only chatd's user ($SUDO_USER) may open the socket
#   3. an nftables table that drops everything tandem-sbx sends to private, loopback, link-local
#      and CGNAT ranges (defence in depth: chatd runs every sandbox with --network none anyway)
#   4. builds tandem-sandbox:1 as tandem-sbx
# Until this runs, chatd uses the rootful Docker socket with the same locked-down containers.
set -eu
[ "$(id -u)" = 0 ] || { echo "run it with sudo"; exit 1; }
OWNER="${SUDO_USER:?run it with sudo from the user chatd runs as}"
HERE="$(cd "$(dirname "$0")" && pwd)"
apt-get update && apt-get install -y podman crun passt uidmap acl
if ! id tandem-sbx >/dev/null 2>&1; then
    useradd --system --create-home --shell /usr/sbin/nologin tandem-sbx
fi
grep -q '^tandem-sbx:' /etc/subuid || usermod --add-subuids 200000-265535 tandem-sbx
grep -q '^tandem-sbx:' /etc/subgid || usermod --add-subgids 200000-265535 tandem-sbx
loginctl enable-linger tandem-sbx
UID_SBX="$(id -u tandem-sbx)"
sudo -u tandem-sbx XDG_RUNTIME_DIR="/run/user/$UID_SBX" systemctl --user enable --now podman.socket
setfacl -m "u:$OWNER:x" "/run/user/$UID_SBX" "/run/user/$UID_SBX/podman"
setfacl -m "u:$OWNER:rw" "/run/user/$UID_SBX/podman/podman.sock"
sudo -u tandem-sbx podman unshare true
install -m 644 "$HERE/tandem-sbx.nft" /etc/nftables.d/tandem-sbx.nft 2>/dev/null || {
    mkdir -p /etc/nftables.d && install -m 644 "$HERE/tandem-sbx.nft" /etc/nftables.d/; }
nft -f /etc/nftables.d/tandem-sbx.nft
grep -q 'nftables.d' /etc/nftables.conf || echo 'include "/etc/nftables.d/*.nft"' >> /etc/nftables.conf
cp -r "$HERE" "/home/tandem-sbx/sandbox-build" && chown -R tandem-sbx "/home/tandem-sbx/sandbox-build"
sudo -u tandem-sbx bash -c "cd /home/tandem-sbx/sandbox-build && podman build -t tandem-sandbox:1 -f Containerfile ."
echo
echo "done. Start chatd with: QSE_SANDBOX_SOCKET=/run/user/$UID_SBX/podman/podman.sock"
