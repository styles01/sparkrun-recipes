#!/bin/bash
# Fix Docker apt mirrors for ARM64 Ubuntu

# Replace sources.list with Google Cloud mirror
cat > /etc/apt/sources.list << 'EOF'
deb http://mirror.gcp.ports.ubuntu.com/ubuntu-ports/ jammy main restricted
deb http://mirror.gcp.ports.ubuntu.com/ubuntu-ports/ jammy-updates main restricted
deb http://mirror.gcp.ports.ubuntu.com/ubuntu-ports/ jammy universe
deb http://mirror.gcp.ports.ubuntu.com/ubuntu-ports/ jammy-updates universe
deb http://mirror.gcp.ports.ubuntu.com/ubuntu-ports/ jammy multiverse
deb http://mirror.gcp.ports.ubuntu.com/ubuntu-ports/ jammy-updates multiverse
deb http://mirror.gcp.ports.ubuntu.com/ubuntu-ports/ jammy-backports main restricted universe multiverse
deb http://security.ubuntu.com/ubuntu jammy-security main restricted
deb http://security.ubuntu.com/ubuntu jammy-security universe
deb http://security.ubuntu.com/ubuntu jammy-security multiverse
EOF

apt-get update
apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    build-essential \
    libsndfile1 \
    libsndfile1-dev \
    ffmpeg \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    python3-pip \
    gnupg2 \
    ca-certificates

rm -rf /var/lib/apt/lists/*
