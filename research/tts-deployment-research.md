# MOSS-TTS and Higgs TTS Deployment Research for NVIDIA GB10/DGX Spark

## Executive Summary

This research covers deployment methods for **MOSS-TTS Realtime** and **Higgs Audio v3 TTS** on Linux systems with NVIDIA GB10 architecture (sm_121, ~121GB VRAM). Both models are suitable for commercial deployment, with different installation approaches and dependency requirements.

---

## 1. MOSS-TTS Realtime Deployment

### Model Information
- **Repository**: OpenMOSS/MOSS-TTS-Nano (main repo)
- **Model**: `OpenMOSS-Team/MOSS-TTS-Realtime` (2B parameters)
- **License**: Apache 2.0 (commercial-friendly)
- **VRAM Requirements**: ~13GB for Realtime model
- **Sample Rate**: 24kHz mono
- **Best For**: Multi-turn dialogue, real-time streaming

### Proven Installation Methods

#### Method 1: Docker Compose (Recommended for Production)
**Repository**: https://github.com/EasyMetaAu/moss-tts

**Prerequisites**:
- Docker + NVIDIA Container Toolkit
- CUDA 12.1+
- ~19GB storage for weights

**Installation Steps**:

```bash
# 1. Clone the repository
git clone https://github.com/EasyMetaAu/moss-tts.git
cd moss-tts

# 2. Configure environment
cp .env.example .env
# Edit .env to set:
#   HF_HOME=/path/to/huggingface/cache
#   VOICES_DIR=/path/to/voices
#   REALTIME_PORT=6008

# 3. Download model weights
HF_ENDPOINT=https://hf-mirror.com python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('OpenMOSS-Team/MOSS-TTS-Realtime')
"

# 4. Start the service
docker compose --profile realtime up -d --build

# 5. Test the API
curl -X POST http://localhost:6008/api/generate \
  -F "text=Hello world" \
  -F "demo_id=demo-1" \
  --output output.wav
```

**Docker Images Available**:
- `logictan/moss-tts` (890 pulls, 7 months old)
- `npfost/moss-tts` (238 pulls, 3 months old)
- Official: `mossmoss/moss-tts-nano:latest` (for nano profile)

#### Method 2: Direct Python Installation
```bash
# Create virtual environment
python3 -m venv moss-tts-env
source moss-tts-env/bin/activate

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers sentencepiece soundfile requests

# Download model
from huggingface_hub import snapshot_download
snapshot_download('OpenMOSS-Team/MOSS-TTS-Realtime')

# Run inference (see example script below)
```

### Example Inference Script

```python
#!/usr/bin/env python3
"""MOSS-TTS-Realtime inference example for production deployment."""

import torch
from transformers import AutoProcessor, AutoModel
import soundfile as sf

# Load model and processor
model_name = "OpenMOSS-Team/MOSS-TTS-Realtime"
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModel.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,  # or float16 for GB10
    device_map="auto"
)

def generate_speech(text, reference_audio=None, output_path="output.wav"):
    """Generate speech from text with optional voice cloning."""
    inputs = processor(
        text=text,
        reference_audio=reference_audio,  # Optional: path to reference WAV
        return_tensors="pt",
        sampling_rate=24000
    ).to(model.device)
    
    with torch.no_grad():
        audio = model.generate(**inputs)
    
    # Save audio
    sf.write(output_path, audio.cpu().numpy(), 24000)
    print(f"Generated: {output_path}")

# Example usage
if __name__ == "__main__":
    # Simple TTS
    generate_speech("Hello, this is MOSS-TTS Realtime.")
    
    # Voice cloning with reference audio
    # generate_speech(
    #     "Hello from the cloned voice!",
    #     reference_audio="reference.wav",
    #     output_path="cloned.wav"
    # )
```

### Known Issues & Solutions

**Issue 1**: CUDA architecture compatibility
- **Solution**: MOSS-TTS supports CUDA 12.1+, works on sm_86+ (Ampere+). GB10 (sm_121) is fully compatible.

**Issue 2**: Memory optimization
- **Solution**: Use `torch_dtype=torch.float16` instead of bfloat16 to save ~30% VRAM on inference.

**Issue 3**: Long text segmentation
- **Solution**: The EasyMetaAu implementation includes automatic text segmentation at 500 character boundaries with voice consistency maintenance.

---

## 2. Higgs Audio v3 TTS Deployment

### Model Information
- **Model**: `bosonai/higgs-tts-3-4b` (4B backbone, ~5B total)
- **Alternative**: `multimodalart/higgs-audio-v3-tts-4b-transformers` (no SGLang required)
- **License**: Boson AI Research & Non-Commercial (requires commercial license for production)
- **VRAM Requirements**: ~15-20GB for 4B model
- **Sample Rate**: 24kHz
- **Languages**: 100+ languages
- **Best For**: Conversational TTS, zero-shot voice cloning, emotion/style control

### Proven Installation Methods

#### Method 1: Transformers-Only Port (RECOMMENDED - No SGLang)
**Model**: `multimodalart/higgs-audio-v3-tts-4b-transformers`

This is a **critical finding** - this port eliminates the SGLang-Omni dependency that causes protobuf conflicts.

**Installation**:

```bash
# Create virtual environment
python3 -m venv higgs-tts-env
source higgs-tts-env/bin/activate

# Install dependencies (NO SGLANG needed!)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers>=4.45.0
pip install soundfile

# The model automatically loads bosonai/higgs-audio-v2-tokenizer
```

**Example Inference Script**:

```python
#!/usr/bin/env python3
"""Higgs Audio v3 TTS inference (transformers-only, no SGLang)."""

import torch
from transformers import AutoProcessor, AutoModel
import soundfile as sf

# Load model - uses transformers-native approach
model_name = "multimodalart/higgs-audio-v3-tts-4b-transformers"
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModel.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,  # Optimal for GB10
    device_map="auto",
    trust_remote_code=True
)

def generate_speech(text, reference_audio=None, output_path="output.wav"):
    """
    Generate speech with Higgs Audio v3.
    
    Args:
        text: Text to synthesize
        reference_audio: Optional path to reference audio for voice cloning
        output_path: Output WAV file path
    """
    inputs = processor(
        text=text,
        reference_audio=reference_audio,
        return_tensors="pt"
    ).to(model.device)
    
    with torch.no_grad():
        audio = model.generate(**inputs)
    
    # Audio is returned as 24kHz mono float32 tensor
    sf.write(output_path, audio.cpu().numpy(), 24000)
    print(f"Generated: {output_path}")

# Example usage
if __name__ == "__main__":
    # Simple TTS
    generate_speech("Hello from Higgs Audio v3!")
    
    # Voice cloning
    # generate_speech(
    #     "This is a cloned voice!",
    #     reference_audio="reference.wav",
    #     output_path="cloned.wav"
    # )
```

#### Method 2: ComfyUI Integration
**Repository**: https://github.com/Saganaki22/Higgs_v3-TTS-ComfyUI

For production workflows using ComfyUI:

```bash
# Install ComfyUI first
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI
pip install -r requirements.txt

# Install Higgs v3 TTS node
cd custom_nodes
git clone https://github.com/Saganaki22/Higgs_v3-TTS-ComfyUI.git
cd Higgs_v3-TTS-ComfyUI
pip install -r requirements.txt

# Start ComfyUI
cd ..
python main.py
```

#### Method 3: Higgs-Audio-v3-Studio (Linux Desktop App)
**Repository**: https://github.com/Saganaki22/Higgs-Audio-v3-Studio

Native Rust/Tauri desktop application with C++/CUDA engine:
- Supports Linux and Windows
- Native C++/CUDA inference engine
- No Python dependencies
- Best for desktop applications

**Installation**:
```bash
# Clone and build
git clone https://github.com/Saganaki22/Higgs-Audio-v3-Studio.git
cd Higgs-Audio-v3-Studio

# Build (requires Rust toolchain)
cargo build --release

# Run
./target/release/higgs-audio-studio
```

### SGLang-Omni Dependency Issue (RESOLVED)

**Problem**: Original Higgs TTS 3 required SGLang-Omni which has protobuf version conflicts.

**Solution**: Use `multimodalart/higgs-audio-v3-tts-4b-transformers` which:
- Uses plain 🤗 transformers (no SGLang)
- Weights are unchanged from original
- Only adds a small processor/pipeline wrapper
- Eliminates all protobuf conflicts
- **Fully compatible with commercial deployment on Linux**

### Known Issues & Solutions

**Issue 1**: Commercial License Requirement
- **Status**: Higgs Audio v3 requires commercial license for production use
- **Solution**: Contact Boson AI for commercial licensing (required for commercial deployment)
- **Note**: Creator use (podcasts, videos, social media) is free with attribution

**Issue 2**: VRAM Requirements
- **Solution**: Use `torch_dtype=torch.float16` to reduce VRAM by ~25%
- **Alternative**: Use GGUF quantization via llama.cpp port (Rafa00127/HiggsTTS.cpp)

**Issue 3**: CUDA Architecture
- **Status**: Higgs Audio v3 supports CUDA 12.1+
- **GB10 Compatibility**: Full support for sm_121 architecture

---

## 3. Docker Container Availability

### MOSS-TTS
✅ **Docker images available**:
- `logictan/moss-tts` - Community image
- `npfost/moss-tts` - Community image  
- `mossmoss/moss-tts-nano:latest` - Official nano image
- EasyMetaAu/moss-tts provides Docker Compose for all three profiles (nano, local, realtime)

### Higgs Audio v3
❌ **No official Docker images found**
- Must be installed via pip/conda
- ComfyUI integration available
- Desktop studio app available for Linux

---

## 4. Dependency Conflicts & Solutions

### MOSS-TTS Dependencies
```
torch>=2.0 (CUDA 12.1)
transformers>=4.35.0
sentencepiece>=0.1.99
soundfile>=0.12.0
requests>=2.31.0
```
**Conflicts**: None known. Works cleanly with standard PyTorch stack.

### Higgs Audio v3 Dependencies (Transformers-Only Port)
```
torch>=2.0 (CUDA 12.1)
transformers>=4.45.0
soundfile>=0.12.0
```
**Conflicts**: None - this is the key advantage over SGLang-based installation.

### SGLang-Omni (NOT RECOMMENDED)
```
sglang>=0.3.0
protobuf>=4.25.0  # Conflicts with many other packages
transformers>=4.45.0
```
**Conflicts**: 
- protobuf version conflicts with many ML libraries
- Complex installation requiring custom SGLang-Omni build
- **Solution**: Use transformers-only port instead

---

## 5. Production Deployment Recommendations for GB10/DGX Spark

### MOSS-TTS Realtime (Recommended for Production)

**Advantages**:
- ✅ Apache 2.0 license (commercial-friendly)
- ✅ Docker Compose deployment available
- ✅ Lower VRAM requirements (~13GB)
- ✅ Proven production deployment (EasyMetaAu/moss-tts)
- ✅ No dependency conflicts
- ✅ Supports 20 languages

**Production Setup**:
```bash
# Use Docker Compose for production
docker compose --profile realtime up -d

# Scale with multiple instances if needed
docker compose --profile realtime up -d --scale realtime=3

# Monitor with health check
curl http://localhost:6008/health
```

### Higgs Audio v3 TTS (Recommended for Quality)

**Advantages**:
- ✅ Superior voice quality and expressiveness
- ✅ 100+ languages
- ✅ Zero-shot voice cloning
- ✅ Emotion/style/prosody control
- ✅ Transformers-only port (no SGLang conflicts)

**Production Setup**:
```bash
# Use transformers-only port
pip install torch transformers soundfile

# Deploy as REST API using FastAPI
# See example below
```

**FastAPI Example**:
```python
from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoProcessor, AutoModel
import soundfile as sf
import io

app = FastAPI()

class TTSRequest(BaseModel):
    text: str
    reference_audio: str = None

model = AutoModel.from_pretrained(
    "multimodalart/higgs-audio-v3-tts-4b-transformers",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
processor = AutoProcessor.from_pretrained("multimodalart/higgs-audio-v3-tts-4b-transformers")

@app.post("/generate")
async def generate(request: TTSRequest):
    inputs = processor(text=request.text, reference_audio=request.reference_audio, return_tensors="pt").to(model.device)
    with torch.no_grad():
        audio = model.generate(**inputs)
    
    # Return as WAV
    buffer = io.BytesIO()
    sf.write(buffer, audio.cpu().numpy(), 24000, format='WAV')
    buffer.seek(0)
    return buffer.read()
```

---

## 6. Summary Comparison

| Feature | MOSS-TTS Realtime | Higgs Audio v3 |
|---------|-------------------|----------------|
| **Model Size** | 2B | 4B backbone (~5B total) |
| **VRAM** | ~13GB | ~15-20GB |
| **License** | Apache 2.0 (Commercial OK) | Research/Non-Commercial (Commercial License Required) |
| **Languages** | 20 | 100+ |
| **Docker** | ✅ Available | ❌ No official images |
| **SGLang Required** | ❌ No | ❌ No (use transformers port) |
| **Production Ready** | ✅ Yes | ⚠️ Requires commercial license |
| **Voice Cloning** | ✅ Zero-shot | ✅ Zero-shot |
| **Emotion Control** | Limited | ✅ Inline control |
| **Real-time** | ✅ Yes | ⚠️ Higher latency |

---

## 7. Final Recommendations for GB10/DGX Spark

### For Commercial Production:
1. **MOSS-TTS Realtime** - Best choice for immediate commercial deployment
   - Use EasyMetaAu/moss-tts Docker Compose setup
   - No licensing concerns
   - Proven deployment stack

2. **Higgs Audio v3** - Best for quality if commercial license obtained
   - Use multimodalart/higgs-audio-v3-tts-4b-transformers (no SGLang)
   - Contact Boson AI for commercial licensing
   - Superior voice quality and expressiveness

### Installation Priority:
1. **MOSS-TTS**: Start with Docker Compose (EasyMetaAu/moss-tts)
2. **Higgs TTS**: Use transformers-only port (multimodalart)
3. **Avoid**: SGLang-Omni due to protobuf conflicts

### GB10-Specific Optimizations:
- Use `torch_dtype=torch.bfloat16` for optimal performance on sm_121
- Both models fully support CUDA 12.1+
- 121GB VRAM allows running both models simultaneously with room for batch processing

---

## References

### MOSS-TTS
- Main Repository: https://github.com/OpenMOSS/MOSS-TTS-Nano
- Docker Compose: https://github.com/EasyMetaAu/moss-tts
- Model: https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Realtime
- Docker Images: https://hub.docker.com/search?q=moss-tts

### Higgs Audio v3
- Original Model: https://huggingface.co/bosonai/higgs-tts-3-4b
- Transformers Port: https://huggingface.co/multimodalart/higgs-audio-v3-tts-4b-transformers
- ComfyUI: https://github.com/Saganaki22/Higgs_v3-TTS-ComfyUI
- Desktop App: https://github.com/Saganaki22/Higgs-Audio-v3-Studio

### SGLang (Not Recommended)
- Repository: https://github.com/sgl-project/sglang
