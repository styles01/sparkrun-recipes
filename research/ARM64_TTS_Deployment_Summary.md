# ARM64 TTS Deployment Summary

## Research Findings

### What Was Researched
- Docker-based deployment of MOSS-TTS Realtime on Linux ARM64
- Docker-based deployment of Higgs TTS 3 on Linux ARM64
- ARM64 compatibility issues and workarounds
- Alternative containerization approaches
- Production deployment strategies for NVIDIA GB10/DGX Spark

### Key Findings

#### 1. No Official ARM64 Docker Images Exist
- **MOSS-TTS**: No official Docker images for any architecture
- **Higgs TTS**: Official image `ghcr.io/fan92rus/higgs-tts:latest` is x86_64 only
- **PyTorch Base**: Official PyTorch CUDA images don't support ARM64 (Issue: https://github.com/pytorch/pytorch/issues/168168)

#### 2. Build-from-Source Required
Both projects require building custom Docker images for ARM64:
- PyTorch must be built from source or use nightly builds
- CUDA toolkit for ARM64 uses SBSA (Scalable Binary Architecture)
- bitsandbytes may not be available for ARM64 Linux

#### 3. MOSS-TTS Has Better ARM64 Prospects
- MOSS-TTS-Nano supports MLX on macOS ARM
- Code is portable to ARM64 Linux
- vLLM-Omni and SGLang-Omni both support MOSS-TTS

#### 4. Higgs TTS Docker Repository Exists
- Repository: https://github.com/fan92rus/higgs-tts-ui-docker
- Docker Compose files provided but only for x86_64
- Can be adapted for ARM64 with modified Dockerfile

---

## Files Created

### 1. Research Document
**Path**: `/Users/clawdio/moss-tts-higgs-arm64-deployment-research.md`

Comprehensive research document covering:
- Current status of ARM64 support
- Known issues and workarounds
- Alternative deployment approaches
- Production deployment recommendations
- Key resources and links

### 2. MOSS-TTS ARM64 Deployment Package
**Directory**: `/Users/clawdio/moss-tts-arm64/`

Files:
- `Dockerfile` - ARM64 Docker build for MOSS-TTS
- `docker-compose.yml` - Docker Compose configuration
- `README.md` - Deployment guide and API examples

### 3. Higgs TTS ARM64 Deployment Package
**Directory**: `/Users/clawdio/higgs-tts-arm64/`

Files:
- `Dockerfile` - ARM64 Docker build for Higgs TTS 3
- `docker-compose.yml` - Docker Compose configuration
- `README.md` - Deployment guide and API examples

---

## Deployment Commands

### MOSS-TTS on ARM64

```bash
# Build and run
cd /Users/clawdio/moss-tts-arm64
docker compose build
docker compose up -d

# Test
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "OpenMOSS-Team/MOSS-TTS-Realtime",
    "input": "Hello!",
    "voice": "default",
    "response_format": "wav"
  }' --output test.wav
```

### Higgs TTS on ARM64

```bash
# Build and run
cd /Users/clawdio/higgs-tts-arm64
docker compose build
docker compose up -d

# Test
curl -X POST http://localhost:7861/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello!",
    "voice": "default"
  }' --output test.wav
```

---

## Known Issues & Workarounds

### Issue 1: PyTorch ARM64 Docker Support
**Problem**: Official PyTorch CUDA images are x86_64 only

**Workarounds**:
1. Use nightly builds: `pip install --index-url https://download.pytorch.org/whl/nightly/cu128`
2. Build PyTorch from source for ARM64
3. Use conda instead of Docker

### Issue 2: CUDA on ARM64 (SBSA)
**Problem**: NVIDIA CUDA on ARM64 uses SBSA architecture

**Solution**: Install CUDA toolkit for SBSA:
```bash
wget -qO - https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/sbsa/3bf863cc.pub | apt-key add -
echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/sbsa/ /" > /etc/apt/sources.list.d/cuda.list
apt-get update
apt-get install cuda-toolkit-12-8
```

### Issue 3: bitsandbytes Compatibility
**Problem**: bitsandbytes may not have ARM64 Linux CUDA builds

**Workarounds**:
1. Skip bitsandbytes if not critical
2. Build from source (may fail)
3. Use alternative quantization methods

---

## Alternative Approaches

### 1. Conda-Based Deployment
Recommended for ARM64 if Docker proves problematic:

```bash
conda create -n tts-arm64 python=3.11 -y
conda activate tts-arm64
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install moss-tts  # or higgs-tts
```

### 2. Podman with ARM64 Support
```bash
podman build -t tts-arm64 .
podman run --rm -it --device /dev/nvidia0 -p 8000:8000 tts-arm64
```

### 3. Buildx Multi-Arch Build
```bash
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t tts:multiarch \
  --push \
  .
```

---

## Production Recommendations

### For NVIDIA GB10/DGX Spark

1. **Verify CUDA Support First**
   ```bash
   nvidia-smi
   python3 -c "import torch; print(torch.cuda.is_available())"
   ```

2. **Start with Conda** (easier than Docker for ARM64)
   ```bash
   # Install Miniconda for ARM64
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh
   bash Miniconda3-latest-Linux-aarch64.sh -b
   ```

3. **Use Systemd for Production**
   Create service file at `/etc/systemd/system/tts.service`

4. **Monitor Resource Usage**
   ```bash
   nvidia-smi dmon  # Monitor GPU
   htop             # Monitor CPU/RAM
   ```

---

## Key Resources

### Repositories
- **MOSS-TTS**: https://github.com/OpenMOSS/MOSS-TTS
- **MOSS-TTS-Nano**: https://github.com/OpenMOSS/MOSS-TTS-Nano
- **Higgs TTS Docker**: https://github.com/fan92rus/higgs-tts-ui-docker
- **MLX-Audio**: https://github.com/Blaizzy/mlx-audio
- **vLLM-Omni**: https://github.com/vllm-project/vllm-omni
- **SGLang-Omni**: https://github.com/sgl-project/sglang-omni

### Documentation
- **MOSS-TTS vLLM Recipe**: https://github.com/vllm-project/vllm-omni/blob/main/recipes/OpenMOSS/MOSS-TTS.md
- **MOSS-TTS Local Cookbook**: https://github.com/sgl-project/sglang-omni/blob/main/docs/cookbook/moss_tts_local.md
- **MOSS-TTS Cookbook**: https://github.com/sgl-project/sglang-omni/blob/main/docs/cookbook/moss_tts.md

### Issues
- **PyTorch ARM64 Docker**: https://github.com/pytorch/pytorch/issues/168168
- **MOSS-TTS Deployment**: https://github.com/OpenMOSS/MOSS-TTS-Nano/issues/79

---

## Next Steps

1. **Test on GB10/DGX Spark**
   - Verify CUDA support
   - Try conda-based deployment first
   - Build Docker images if conda works

2. **Monitor for Updates**
   - Watch PyTorch for official ARM64 Docker support
   - Check MOSS-TTS and Higgs TTS repos for ARM64 builds

3. **Contribute to Community**
   - If successful, share ARM64 build instructions
   - Consider submitting PRs with ARM64 Dockerfiles

---

*Research completed: July 15, 2026*
*Files created: 7 (1 research doc, 2 deployment packages with 3 files each)*
