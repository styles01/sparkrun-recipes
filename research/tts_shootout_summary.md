# TTS Shootout Status - DGX Spark (GB10)

## Hardware
- **GPU:** NVIDIA GB10 (sm_121)
- **CUDA:** 13.0
- **PyTorch:** 2.11.0+cu128 (✅ Supports sm_121)
- **VRAM:** 121GB total, ~25GB free (Qwen 122B running)

## Model Test Results

| Model | Status | Notes |
|-------|--------|-------|
| **Chatterbox Turbo** | ✅ **WORKS** | Generated audio successfully on GPU. ~16 min generation time. |
| **Qwen3-TTS 1.7B Base** | ✅ **WORKS** | Generated audio successfully using `qwen-tts` library. Voice cloning supported. |
| **MOSS-TTS Realtime** | ⚠️ **BLOCKED** | Requires custom transformers integration. Import error with current setup. |
| **Higgs TTS 3** | ⚠️ **API ONLY** | Local inference requires SGLang-Omni. API available (need API key). |

## Dependencies Installed
- `torch==2.11.0+cu128`
- `torchaudio==2.11.0+cu128`
- `transformers==4.57.3`
- `chatterbox-tts==0.1.7`
- `qwen-tts==0.1.1` (from GitHub)
- `sgl-kernel==0.3.21`
- `soundfile`, `librosa`, `onnxruntime`

## Generated Audio Files
- `~/tts/results/chatterbox.wav` - Chatterbox test
- `~/tts/results/qwen3.wav` - Qwen3-TTS test

## Next Steps

### Option 1: Complete MOSS-TTS Setup
- Requires fixing transformers import issues
- May need to install specific transformers version or patch the code
- Could take significant time to debug

### Option 2: Use Higgs via Boson API
- No local inference needed
- Requires API key from Boson AI
- Fast setup, but depends on external service

### Option 3: Install SGLang-Omni
- Enables local inference for both MOSS-TTS and Higgs TTS
- Complex setup (Docker + GPU resources)
- Full control over all models

### Option 4: Proceed with Working Models
- Chatterbox and Qwen3-TTS are fully functional
- Can generate comparative samples now
- Defer MOSS and Higgs to later

## Recommendation
**Proceed with Option 4** - Generate comparative samples with Chatterbox and Qwen3-TTS first. These two models are working and can provide immediate results. MOSS and Higgs can be added later when time permits for deeper integration work.

---
*Generated: 2026-07-15*
