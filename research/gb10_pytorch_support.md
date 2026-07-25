# NVIDIA GB10 / DGX Spark (sm_121) PyTorch Support Research

## Summary

**NVIDIA GB10 (DGX Spark)** has compute capability **sm_121** (CUDA compute capability 12.1). This architecture is supported starting with **CUDA Toolkit 13.0** and requires specific PyTorch builds.

## Key Findings

### 1. Hardware Information
- **GPU**: NVIDIA GB10 (DGX Spark)
- **Compute Capability**: sm_121 (12.1)
- **Architecture**: Blackwell variant for SBSA/Arm platforms
- **Related devices**: Jetson Thor (sm_110), GB300/B300 (sm_10.3), GB200/B200 (sm_10.0)

### 2. CUDA Support Timeline

**CUDA 12.8** (First Blackwell support):
- Added initial support for sm_10x (GB200/B200) and sm_12x architectures
- **Does NOT include full sm_121 support** for GB10/DGX Spark

**CUDA 13.0** (August 2025):
- **First version with full sm_121 support** for GB10/DGX Spark
- Unifies Arm platform support (critical for DGX Spark)
- Includes proper PTXAS support for sm_121 architecture
- Supports all Blackwell variants including Jetson Thor

### 3. PyTorch Version Requirements

**PyTorch 2.9+ with CUDA 13.0** is required for sm_121 support:

- **PyTorch 2.5.1** (your current version): Does NOT support sm_121
  - Built with CUDA 12.x which lacks full sm_121 support
  - Results in "sm_121 is not compatible" errors

- **PyTorch 2.9+ with cu130**: **SUPPORTED**
  - First PyTorch version with CUDA 13.0 binaries
  - Includes sm_121 architecture support
  - Available via: `pip install torch --index-url https://download.pytorch.org/whl/cu130`

- **PyTorch 2.8+ with cu128**: Partial support
  - May work for basic operations but has limitations
  - Triton issues with sm_121 (missing PTXAS support)

### 4. Known Issues & Workarounds

**Issue**: Triton wheel missing CUDA 13.0 PTXAS
- **Error**: `ptxas fatal: Value 'sm_121a' is not defined for option 'gpu-name'`
- **Workaround 1**: Set explicit PTXAS path:
  ```bash
  export TRITON_PTXAS_PATH=/usr/local/cuda-13.0/bin/ptxas
  ```
- **Workaround 2**: Install CUDA 13.0 toolkit system-wide

**Issue**: cuDNN attention backend limitations
- Some cuDNN features may have head_dim restrictions on sm_121
- Use FLASH_ATTENTION backend as fallback for large head_dim models

### 5. Recommended Installation

```bash
# Install PyTorch with CUDA 13.0 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

# Or for latest nightly with full sm_121 support
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu130

# Set Triton PTXAS path if needed
export TRITON_PTXAS_PATH=/usr/local/cuda-13.0/bin/ptxas
```

### 6. Verification

After installation, verify sm_121 support:

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU name: {torch.cuda.get_device_name(0)}")
print(f"Compute capability: {torch.cuda.get_device_capability(0)}")
# Should show: (12, 1) for sm_121
```

## Available PyTorch CUDA Builds

| CUDA Version | PyTorch Version | sm_121 Support | Notes |
|--------------|-----------------|----------------|-------|
| cu118 | 2.0-2.4 | ❌ No | Too old |
| cu121 | 2.1-2.5 | ❌ No | No sm_121 support |
| cu124 | 2.3-2.6 | ❌ No | No sm_121 support |
| cu126 | 2.6-2.7 | ⚠️ Partial | Initial Blackwell, limited sm_121 |
| cu128 | 2.7-2.8 | ⚠️ Partial | Better Blackwell support, Triton issues |
| **cu130** | **2.9+** | **✅ Full** | **Recommended for GB10** |

## Sister Devices (Dell/Asus)

The user mentioned sister devices from Dell and Asus. These would be:
- **Dell**: PowerEdge servers with GB10/Blackwell GPUs
- **Asus**: Workstation/server offerings with Blackwell architecture

All these devices with sm_121 compute capability require the same CUDA 13.0+ and PyTorch 2.9+ solution.

## References

- [GitHub Issue #159779](https://github.com/pytorch/pytorch/issues/159779) - Enable CUDA 13.0 binaries
- [GitHub Issue #163801](https://github.com/pytorch/pytorch/issues/163801) - Triton PTXAS issues with sm_121
- [GitHub Issue #181379](https://github.com/pytorch/pytorch/issues/181379) - sm_120/sm_121 cuDNN support
- [NVIDIA CUDA 13.0 Blog](https://developer.nvidia.com/blog/whats-new-and-important-in-cuda-toolkit-13-0/)
- [NVIDIA Compute Capability Table](https://developer.nvidia.com/cuda-gpus)

## Action Items

1. **Upgrade to PyTorch 2.9+ with CUDA 13.0** (cu130)
2. **Install CUDA 13.0 Toolkit** system-wide for full Triton support
3. **Set TRITON_PTXAS_PATH** environment variable if needed
4. **Test with simple CUDA operations** before running TTS models
5. **Consider FLASH_ATTENTION backend** as fallback for cuDNN issues
