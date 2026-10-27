#!/bin/bash

# Exit immediately if any command fails
set -e

echo "Stopping and removing containers..."
docker compose down

echo "Building images..."
if ! docker compose build; then
  echo ""
  echo "Build failed. If you saw auth.docker.io / 'network is unreachable' on an IPv6 address,"
  echo "Docker Hub is being reached over broken IPv6. This repo pulls Node from"
  echo "public.ecr.aws (IPv4-first). Pull latest deploy.sh + Dockerfiles and retry."
  echo "Host workaround: add this line to /etc/gai.conf then retry:"
  echo "  precedence ::ffff:0:0/96  100"
  exit 1
fi

echo "Starting services in detached mode..."
docker compose up -d

echo "✅ All done! Services are now running."
