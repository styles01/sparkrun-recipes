#!/bin/bash
# Build tandem-sandbox:1 on this box with the container engine chatd uses (docker, or podman with
# the same arguments). The build needs the internet (Debian packages, uv); the containers it makes
# do not get it except through chatd's filtering proxy (chatd/egress.py).
#   bash ops/sandbox/build.sh [docker|podman]
set -eu
ENGINE="${1:-docker}"
cd "$(dirname "$0")"
"$ENGINE" build -t tandem-sandbox:1 -f Containerfile .
"$ENGINE" image inspect tandem-sandbox:1 --format '{{.Id}} {{.Size}}'
