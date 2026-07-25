# Docker-Based Deployment of MOSS-TTS and Higgs TTS 3 on Linux ARM64

## Executive Summary

Based on research across GitHub, technical forums, and official documentation, here's what we found about deploying MOSS-TTS Realtime and Higgs TTS 3 on Linux ARM64 (NVIDIA GB10/DGX Spark):

### Key Findings

1. **No Official ARM64 Docker Images Exist** - Both projects currently only provide x86_64 Docker images
2. **PyTorch Base Images Lack ARM64 Support** - The pytorch/pytorch CUDA images (used as base for both projects) don't support ARM64 architecture
3. **Build-from-Source Required** - You'll need to build custom Docker images for ARM64
4. **MOSS-TTS has Better ARM64 Prospects** - MOSS-TTS-Nano runs on macOS ARM via MLX, suggesting ARM64 compatibility at the code level

---

## 1. MOSS-TTS Realtime Docker Deployment on ARM64

### Current Status
- **No official Docker images** for ARM64
- **No Docker Compose setups** documented for Linux ARM64
- **MLX-Audio support** exists for Apple Silicon (MOSS-TTS-Nano specifically)

### Repository Links
- Main repo: https://github.com/OpenMOSS/MOSS-TTS
- Nano variant: https://github.com/OpenMOSS/MOSS-TTS-Nano
- MLX-Audio integration: https://github.com/Blaizzy/mlx-audio

### Recommended Approach: Build Custom ARM64 Image

```dockerfile
# Dockerfile for MOSS-TTS on Linux ARM64
FROM ubuntu:22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_VERSION=12.8

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    libsndfile1 \
    ffmpeg \
    python3.11 \
    python3.11-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install CUDA toolkit for ARM64 (SBSA - Scalable Binary Architecture)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gnupg2 \
    ca-certificates \
    && wget -qO - https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/sbsa/3bf863cc.pub | apt-key add - \
    && echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/sbsa/ /" > /etc/apt/sources.list.d/cuda.list \
    && apt-get update \
    && apt-get install -y cuda-toolkit-12-8 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python3.11 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

# Install PyTorch with CUDA support (ARM64-specific)
# Note: Use nightly or build from source as stable releases may not have ARM64 CUDA wheels
RUN pip install --no-cache-dir \
    torch \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu128

# Install MOSS-TTS dependencies
RUN pip install --no-cache-dir \
    transformers \
    accelerate \
    soundfile \
    librosa \
    numpy \
    gradio

# Clone and install MOSS-TTS
RUN git clone https://github.com/OpenMOSS/MOSS-TTS.git /app/moss-tts
WORKDIR /app/moss-tts
RUN pip install -e .

# Set working directory
WORKDIR /app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
  CMD curl -sf http://localhost:8000/health || exit 1

# Default command
CMD ["python", "-m", "moss_tts.server", "--host", "0.0.0.0", "--port", "8000"]
```

### Build Commands for ARM64

```bash
# Build on ARM64 host (native build)
docker build -t moss-tts-arm64:latest .

# Or build multi-arch using buildx (requires QEMU)
docker buildx build \
  --platform linux/arm64 \
  -t moss-tts-arm64:latest \
  --push \
  .
```

### Docker Compose for ARM64

```yaml
# docker-compose.moss-tts.arm64.yml
version: '3.8'

services:
  moss-tts:
    build:
      context: ./moss-tts
      dockerfile: Dockerfile.arm64
    image: moss-tts-arm64:latest
    container_name: moss-tts
    ports:
      - "8000:8000"
    volumes:
      - moss-tts-models:/app/models
      - moss-tts-cache:/root/.cache
    environment:
      - MOSS_TTS_MODEL=OpenMOSS-Team/MOSS-TTS-Realtime
      - HF_HOME=/app/cache
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-sf", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 300s

volumes:
  moss-tts-models:
  moss-tts-cache:
```

---

## 2. Higgs TTS 3 Docker Deployment on ARM64

### Current Status
- **Docker repository exists**: https://github.com/fan92rus/higgs-tts-ui-docker
- **Official Docker image**: `ghcr.io/fan92rus/higgs-tts:latest` (x86_64 only)
- **No ARM64 variant** available
- **Docker Compose files provided** but only for x86_64

### Repository Links
- Docker repo: https://github.com/fan92rus/higgs-tts-ui-docker
- Dockerfile: https://github.com/fan92rus/higgs-tts-ui-docker/blob/main/Dockerfile

### Existing Dockerfile (x86_64 only)

```dockerfile
# Current Dockerfile from fan92rus/higgs-tts-ui-docker
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime AS build

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir bitsandbytes

FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime AS runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /opt/conda /opt/conda
ENV PATH=/opt/conda/bin:$PATH

COPY src/ src/
COPY voices/ voices/
COPY requirements.txt .

RUN mkdir -p models outputs .cache/huggingface .cache/torch && \
    chmod 777 models outputs .cache

ENV HIGGS_HOST=0.0.0.0
ENV HIGGS_PORT=7861
ENV HIGGS_MODE=auto
ENV HF_HOME=/app/.cache/huggingface
ENV TORCH_HOME=/app/.cache/torch

EXPOSE 7861

HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
  CMD curl -sf http://localhost:7861/api/state || exit 1

CMD ["uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "7861"]
```

### Modified Dockerfile for ARM64

```dockerfile
# Dockerfile.arm64 for Higgs TTS 3
FROM ubuntu:22.04

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    curl \
    python3.11 \
    python3.11-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install CUDA toolkit for ARM64 (SBSA architecture)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gnupg2 \
    ca-certificates \
    && wget -qO - https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/sbsa/3bf863cc.pub | apt-key add - \
    && echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/sbsa/ /" > /etc/apt/sources.list.d/cuda.list \
    && apt-get update \
    && apt-get install -y cuda-toolkit-12-1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python3.11 -m venv /opt/venv
ENV PATH=/opt/conda/bin:/opt/venv/bin:$PATH

# Install PyTorch for ARM64 with CUDA 12.1
# Note: May need to build from source or use nightly builds
RUN pip install --no-cache-dir \
    torch==2.5.1 \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu121

# Install Higgs TTS dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir bitsandbytes uvicorn

# Copy application source
COPY src/ src/
COPY voices/ voices/

# Create persistent directories
RUN mkdir -p models outputs .cache/huggingface .cache/torch && \
    chmod 777 models outputs .cache

# Environment variables
ENV HIGGS_HOST=0.0.0.0
ENV HIGGS_PORT=7861
ENV HIGGS_MODE=auto
ENV HF_HOME=/app/.cache/huggingface
ENV TORCH_HOME=/app/.cache/torch
ENV PYTHONPATH=/app/src

EXPOSE 7861

HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
  CMD curl -sf http://localhost:7861/api/state || exit 1

CMD ["uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "7861"]
```

### ARM64 Docker Compose

```yaml
# docker-compose.higgs-tts.arm64.yml
version: '3.8'

services:
  higgs-tts:
    build:
      context: .
      dockerfile: Dockerfile.arm64
    image: higgs-tts-arm64:latest
    container_name: higgs-tts
    ports:
      - "7861:7861"
    volumes:
      - higgs-models:/app/models
      - higgs-outputs:/app/outputs
      - higgs-cache:/app/.cache
      - ./voices:/app/voices:ro
    environment:
      - HIGGS_MODE=auto
      - HIGGS_IDLE_UNLOAD_SEC=1800
      - HIGGS_HOST=0.0.0.0
      - HIGGS_PORT=7861
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-sf", "http://localhost:7861/api/state"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 300s

volumes:
  higgs-models:
  higgs-outputs:
  higgs-cache:
```

---

## 3. Known ARM64 Compatibility Issues & Workarounds

### Issue 1: PyTorch Docker Images Only Support x86_64
**GitHub Issue**: https://github.com/pytorch/pytorch/issues/168168

**Problem**: 
```
WARNING: The requested image's platform (linux/amd64) does not match 
the detected host platform (linux/arm64/v8) and no specific platform was requested
```

**Workarounds**:
1. **Build PyTorch from source** for ARM64
2. **Use nightly builds** with ARM64 support:
   ```bash
   pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
   ```
3. **Use conda** instead of Docker for ARM64:
   ```bash
   conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
   ```

### Issue 2: CUDA Toolkit for ARM64 (SBSA)
**Problem**: NVIDIA CUDA on ARM64 uses SBSA (Scalable Binary Architecture), not standard ARM64

**Solution**:
```bash
# Install CUDA for ARM64/SBSA
wget -qO - https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/sbsa/3bf863cc.pub | apt-key add -
echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/sbsa/ /" > /etc/apt/sources.list.d/cuda.list
apt-get update
apt-get install cuda-toolkit-12-8
```

### Issue 3: bitsandbytes Compatibility
**Problem**: bitsandbytes may not have ARM64 Linux CUDA builds

**Workarounds**:
1. **Skip bitsandbytes** if not critical:
   ```dockerfile
   # Remove bitsandbytes from requirements
   RUN pip install --no-cache-dir -r requirements.txt
   # Don't install bitsandbytes
   ```
2. **Build from source**:
   ```bash
   pip install bitsandbytes --no-binary=:all:
   ```

---

## 4. Alternative Containerization Approaches

### Option 1: Podman with ARM64 Support
```bash
# Install Podman on ARM64
sudo apt-get install podman

# Build with Podman
podman build -t moss-tts-arm64:latest .

# Run with GPU support
podman run --rm -it \
  --device /dev/nvidia0 \
  --device /dev/nvidia-uvm \
  --device /dev/nvidia-mux \
  -p 8000:8000 \
  moss-tts-arm64:latest
```

### Option 2: Conda-Based Deployment (No Docker)
```bash
# Create conda environment
conda create -n moss-tts-arm64 python=3.11 -y
conda activate moss-tts-arm64

# Install PyTorch with CUDA for ARM64
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Install MOSS-TTS
git clone https://github.com/OpenMOSS/MOSS-TTS.git
cd MOSS-TTS
pip install -e .

# Run
python -m moss_tts.server --host 0.0.0.0 --port 8000
```

### Option 3: Buildx Multi-Arch Build
```bash
# Enable buildx
docker buildx create --use --name arm64-builder

# Build multi-arch image
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t moss-tts:multiarch \
  --push \
  .
```

**Note**: This requires PyTorch to have ARM64 wheels available.

---

## 5. Production Deployment Recommendations

### For NVIDIA GB10/DGX Spark (ARM64)

#### Step 1: Verify CUDA Support
```bash
# Check CUDA version and ARM64 support
nvidia-smi
nvcc --version

# Verify PyTorch can see GPU
python3 -c "import torch; print(torch.cuda.is_available())"
```

#### Step 2: Use Conda for Base Environment
```bash
# Install Miniconda for ARM64
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh
bash Miniconda3-latest-Linux-aarch64.sh

# Create environment
conda create -n tts-arm64 python=3.11 -y
conda activate tts-arm64

# Install PyTorch with CUDA
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

#### Step 3: Install TTS Models
```bash
# MOSS-TTS
git clone https://github.com/OpenMOSS/MOSS-TTS.git
cd MOSS-TTS
pip install -e .

# Higgs TTS
git clone https://github.com/fan92rus/higgs-tts-ui-docker.git
cd higgs-tts-ui-docker
pip install -r requirements.txt
```

#### Step 4: Run with Systemd (Production)
```bash
# Create systemd service
sudo nano /etc/systemd/system/moss-tts.service
```

```ini
[Unit]
Description=MOSS-TTS Server
After=network.target

[Service]
Type=simple
User=youruser
WorkingDirectory=/opt/moss-tts
Environment="PATH=/home/youruser/miniconda3/envs/tts-arm64/bin:$PATH"
Environment="CUDA_VISIBLE_DEVICES=0"
ExecStart=/home/youruser/miniconda3/envs/tts-arm64/bin/python -m moss_tts.server --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable moss-tts
sudo systemctl start moss-tts
```

---

## 6. Key Resources & Links

### GitHub Repositories
- **MOSS-TTS**: https://github.com/OpenMOSS/MOSS-TTS
- **MOSS-TTS-Nano**: https://github.com/OpenMOSS/MOSS-TTS-Nano
- **Higgs TTS Docker**: https://github.com/fan92rus/higgs-tts-ui-docker
- **MLX-Audio** (ARM reference): https://github.com/Blaizzy/mlx-audio
- **vLLM-Omni** (MOSS-TTS support): https://github.com/vllm-project/vllm-omni
- **SGLang-Omni** (MOSS-TTS support): https://github.com/sgl-project/sglang-omni

### Documentation
- **MOSS-TTS vLLM Recipe**: https://github.com/vllm-project/vllm-omni/blob/main/recipes/OpenMOSS/MOSS-TTS.md
- **MOSS-TTS Local Cookbook**: https://github.com/sgl-project/sglang-omni/blob/main/docs/cookbook/moss_tts_local.md
- **MOSS-TTS Cookbook**: https://github.com/sgl-project/sglang-omni/blob/main/docs/cookbook/moss_tts.md

### Issues & Discussions
- **PyTorch ARM64 Docker Issue**: https://github.com/pytorch/pytorch/issues/168168
- **MOSS-TTS Deployment Issues**: https://github.com/OpenMOSS/MOSS-TTS-Nano/issues/79

---

## 7. Summary Table

| Component | ARM64 Status | Recommendation |
|-----------|--------------|----------------|
| MOSS-TTS Official Docker | ❌ Not available | Build from source |
| Higgs TTS Official Docker | ❌ x86_64 only | Build custom ARM64 image |
| PyTorch Base Image | ❌ No ARM64 CUDA | Use nightly builds or conda |
| MOSS-TTS via MLX | ✅ macOS ARM only | Not applicable to Linux ARM64 |
| CUDA on ARM64 | ✅ SBSA support | Use CUDA toolkit for SBSA |
| bitsandbytes | ⚠️ Limited support | May need to build from source |
| Production Deployment | ⚠️ Manual setup required | Use conda + systemd |

---

## 8. Next Steps for GB10/DGX Spark Deployment

1. **Test PyTorch ARM64 compatibility** on your GB10 system
2. **Build custom Docker images** using the provided Dockerfiles
3. **Consider conda-based deployment** if Docker proves problematic
4. **Monitor PyTorch** for official ARM64 CUDA Docker support
5. **Contribute ARM64 builds** to the community if successful

---

*Research conducted: July 2026*
*Last updated: July 15, 2026*
