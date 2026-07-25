# TTS Shootout Status - DGX Spark (NVIDIA GB10)

## Executive Summary
- **Working Models:** Chatterbox Turbo, Qwen3-TTS 1.7B
- **Blocked Models:** MOSS-TTS (requires implementation), Higgs TTS (unavailable)

## Model Status Table

| Model | Status | Inference | Blockers |
|-------|--------|-----------|----------|
| **Chatterbox Turbo** | ✅ **Working** | CPU | None - fully functional |
| **Qwen3-TTS 1.7B** | ✅ **Working** | GPU (bf16) | None - fully functional |
| **MOSS-TTS** | ⏳ **Research** | GPU | Requires custom streaming API (`MossTTSRealtimeStreamingSession`). No simple `generate()` method. Subagent researching implementation. |
| **Higgs TTS 3** | ❌ **Unavailable** | GPU | 1. Custom model type (`HiggsMultimodalQwen3ForConditionalGeneration`) not in transformers<br>2. No official Python package<br>3. BosonAI/Higgs-TTS repo inaccessible<br>4. Requires source code for inference implementation |

## Technical Details

### Chatterbox Turbo
- **Location:** `~/tts/chatterbox/`
- **Test:** `test_chatterbox.py` - ✅ Passed
- **Audio:** Goldilocks story generated successfully

### Qwen3-TTS 1.7B
- **Location:** `~/tts/qwen3-base/`
- **Test:** `test_qwen3_v4.py` - ✅ Passed
- **Audio:** Goldilocks story generated successfully
- **Hardware:** NVIDIA GB10, bf16 mode

### MOSS-TTS
- **Location:** `~/tts/moss-realtime/`
- **Issue:** Model loads but no inference API found
- **Classes found:**
  - `MossTTSRealtimeStreamingSession` (in `streaming_mossttsrealtime.py`)
  - `MossTTSRealtimeInference` (wrapper class)
- **Missing:** Documentation on how to use these classes for text-to-audio
- **Status:** Awaiting subagent research results

### Higgs TTS 3
- **Location:** `~/tts/higgs-tts3/` (model weights only)
- **Issue:** Model architecture not supported
- **Error:** `KeyError: 'higgs_multimodal_qwen3'`
- **Root Cause:** 
  - Custom model type requires implementation in `transformers` library
  - No official package on PyPI
  - Source code repository (BosonAI/Higgs-TTS) inaccessible
- **Attempted Solutions:**
  - Docker container with NGC PyTorch - ❌ torchaudio incompatibility
  - Venv with system PyTorch 2.13 - ❌ model class not found
- **Status:** Blocked until source code is available

## Recommendations

1. **Proceed with Chatterbox + Qwen3** for the shootout
2. **Document MOSS/Higgs** as "requires additional development"
3. **MOSS:** Wait for subagent research; may need to implement streaming inference
4. **Higgs:** Monitor BosonAI repo for accessibility; may need to wait for official release

## Next Steps

- [ ] Complete MOSS research (subagent `deleg_48d19393`)
- [ ] Generate shootout results for Chatterbox + Qwen3
- [ ] Create final comparison report
- [ ] Optional: Attempt MOSS streaming implementation if research provides guidance
