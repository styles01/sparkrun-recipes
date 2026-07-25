# Research Report: CoCoEmo & EmoShift — Activation Steering for TTS

## 1. CoCoEmo Paper & Code

### Paper
- **Title**: "CoCoEmo: Composable and Controllable Human-Like Emotional TTS via Activation Steering"
- **Venue**: ICML 2026 (NOT ICASSP)
- **arXiv**: https://arxiv.org/abs/2602.03420 (v1: Feb 3, 2026; v2: Jun 16, 2026)
- **Authors**: Siyi Wang, Shihong Tan, Siyi Liu, Hong Jia, Gongping Huang, James Bailey, Ting Dang
- **Key finding**: Emotional prosody is primarily synthesized by the TTS **language module (SLM)**, NOT the flow-matching module. Steering the SLM is sufficient.

### GitHub Repository
- **URL**: https://github.com/wsssy/CoCoEmo
- **Demo**: https://wsssy.github.io/cocoemo_demo/
- **License**: MIT
- **Stars**: 10, 7 commits, created ~June 2026
- **Backbones supported**: CosyVoice2 (Qwen2-based, 24 layers, hidden 896) and IndexTTS2 (GPT2-based, 24 layers, hidden 1024)

### Code Structure
```
cocoemo/
├── backbones/
│   ├── __init__.py        # Backbone adapter interface
│   ├── base.py            # BackboneSpec dataclass
│   ├── cosyvoice2.py      # CosyVoice2 adapter
│   └── indextts2.py       # IndexTTS2 adapter (GPT2-style, most relevant to us)
├── steering/
│   ├── __init__.py        # Public API re-exports
│   ├── _core_cosyvoice.py # 1047 lines - CosyVoice2 steering implementation
│   └── _core_indextts.py  # IndexTTS2 steering implementation (GPT2-style)
├── discriminability/
│   └── _probe.py          # Linear separability probing for layer/operator selection
├── evaluation/
├── data/
└── utils.py
scripts/
├── discriminability.py    # Stage 1: Find best layers
├── extract.py            # Stage 2: Extract steering vectors
├── synthesize.py         # Stage 3: Steered synthesis
├── evaluate.py           # Stage 4: Evaluation
└── run_pipeline.sh       # Full pipeline
steering_vectors/
├── cosyvoice2/           # Precomputed: angry, happy, sad, surprise (vs neutral)
└── indextts2/             # Precomputed: angry, happy, sad, surprise (vs neutral)
```

### Precomputed Steering Vectors
The repo ships ready-to-use `.pt` files for both backbones:
- `steering_vectors/indextts2/angry_neutral_attn_output.pt`
- `steering_vectors/indextts2/happy_neutral_attn_output.pt`
- `steering_vectors/indextts2/sad_neutral_attn_output.pt`
- `steering_vectors/indextts2/surprise_neutral_attn_output.pt`
- Same 4 emotions for `steering_vectors/cosyvoice2/`

**Key**: These are mean-difference vectors: `mean(emotional_activations) - mean(neutral_activations)` at specific layers.

---

## 2. How Activation Steering Works for TTS (Technical Details)

### Core Mechanism (from CoCoEmo paper, Eq. 6 & 8)

**Step 1: Steering Vector Construction (Eq. 6)**
```
v_e^(l,o) = (1/|D^(e)|) * Σ h_i^(l,o)  -  (1/|D_0^(e)|) * Σ h_j^(l,o)
```
Where:
- `D^(e)` = samples with emotion e (e.g., "angry")
- `D_0^(e)` = paired neutral samples (same speaker, same transcript)
- `h_i^(l,o)` = last-token activation at layer `l`, operator `o`
- The vector is the **mean difference** between emotional and neutral representations

**Step 2: Mixed-Emotion Composition (Eq. 7)**
```
v_mix^(l,o) = Σ_e p_e * v_e^(l,o)
```
Where `p_e` are emotion proportions summing to 1.0. This enables composable mixed emotions.

**Step 3: Inference-Time Injection (Eq. 8)**
```
h̃_i^(l,o) = h_i^(l,o) + α · v^(l,o)
```
Where `α` controls steering intensity. Then norm-preserving renormalization:
```
h̃_i^(l,o) ← (||h_i^(l,o)|| / ||h̃_i^(l,o)||) · h̃_i^(l,o)
```

### Two Steering Operators (in code)

1. **Translation** (`translation_op_`): Simple additive
   ```python
   x.add_(alpha * t)  # x += alpha * steering_vector
   ```

2. **Norm-preserving** (`norm_preserve_steer_op_`): Add then renormalize
   ```python
   h_prime = x + alpha * t
   h_tilde = h_prime * (||x|| / ||h_prime||)
   x.copy_(h_tilde)
   ```

### Hook Mechanism
- **Forward hooks** (`register_forward_hook`): Intercept module OUTPUT
- **Pre-forward hooks** (`register_forward_pre_hook`): Intercept module INPUT
- For `attn_output` operator: uses `forward_pre` hook on `c_proj` (GPT2) or `o_proj` (Qwen2)
  - This captures the attention output BEFORE the output projection is applied
- Hooks are registered, normal inference runs, hooks modify activations in-place, hooks are removed

### Layer & Operator Selection (Discriminability Analysis)
CoCoEmo uses a **discriminability-driven approach** to find where emotions are most linearly separable:
- **CosyVoice2**: Best layers 17, 14 (mid-to-late layers 10-17), operator `attn_output`
- **IndexTTS2**: Best layers 6, 8, 1, operator `attn_output`
- Mid-to-late layers and attention outputs consistently show highest emotion separability
- The `discriminability.py` script probes all layers/operators and ranks them

### Extraction Process
1. Feed emotional audio through the TTS model (single forward pass, NOT generation)
2. Hook captures last-token activations at each layer/operator
3. Repeat for neutral audio (same speaker, same text)
4. Compute mean difference → steering vector
5. Save as `.pt` file

### Injection Process
1. Load precomputed steering vector
2. Register forward hooks on target layers
3. Run normal TTS inference (hooks intercept and modify activations)
4. Remove hooks after generation

---

## 3. EmoShift Paper (ICASSP 2026)

### Paper Details
- **Title**: "EmoShift: Lightweight Activation Steering for Enhanced Emotion-Aware Speech Synthesis"
- **Venue**: ICASSP 2026 (accepted)
- **arXiv**: https://arxiv.org/abs/2601.22873 (Jan 30, 2026)
- **Authors**: Li Zhou, Hao Jiang, Junjie Li, Tianrui Wang, Haizhou Li
- **Backbone**: CosyVoice-300M-Instruct

### Key Difference from CoCoEmo
EmoShift **learns** steering vectors via training (10M trainable params), while CoCoEmo **extracts** them from activations (training-free).

### EmoShift Technical Details
- Introduces an **EmoSteer layer** — a learnable per-emotion projection matrix `W_e ∈ R^(d×d)`
- Steering vector: `v_e = h · W_e` (projection of hidden state through learned matrix)
- Training modification: `h' = h + ε · v_e` where `ε = 0.001` (training coefficient)
- Inference modification: `h' = h + α · ε · v_e` where `α ≥ 1` (intensity control)
- Training: 5 epochs, lr=1e-4, 300 training utterances
- 5 emotions: neutral, angry, happy, sad, surprise
- Results: 75.94% overall emotion recall (vs 69.68% baseline, 69.74% full fine-tune)
- EmoShift's relationship to EmoSteer-TTS: "EmoSteer-TTS derives emotion-specific activation offsets from few-shot neutral and target activations, whereas EmoShift learns them as trainable steering parameters"

### No GitHub repo found for EmoShift.

---

## 4. Related: EmoSteer-TTS (the original training-free TTS steering paper)

- **arXiv**: https://arxiv.org/abs/2508.03543 (Aug 5, 2025; v3 Oct 25, 2025)
- **Title**: "EmoSteer-TTS: Fine-Grained and Training-Free Emotion-Controllable Text-to-Speech via Activation Steering"
- **Authors**: Tianxin Xie, Shan Yang, Chenxing Li, Dong Yu, Li Liu
- **Key**: Training-free, steers the **flow-matching module** (not the SLM)
- CoCoEmo paper notes: CoCoEmo achieves "comparable and stronger mixed-emotion control while better preserving speaker similarity" vs EmoSteer-TTS

---

## 5. Implementation for GPT2-Medium (Chatterbox T3)

### Relevance of IndexTTS2 Code Path
The IndexTTS2 backbone in CoCoEmo is **directly relevant** to Chatterbox T3 because:
- IndexTTS2 uses a **GPT2-style transformer** for speech token generation
- Chatterbox T3 is also a **GPT2-medium** transformer
- Both use GPT2 Conv1D layers (`c_attn`, `c_proj`, `c_fc`)

### IndexTTS2 Operation Dictionary (GPT2-style, from code)
```python
INDEXTTS2_OP_DICTS = {
    'attn_output': {
        'module': 'gpt.h.{layer}.attn.c_proj',  # Attention output projection
        'hook type': 'forward_pre'              # Hook BEFORE projection
    },
    'qkv_proj': {
        'module': 'gpt.h.{layer}.attn.c_attn',  # Combined Q,K,V projection
        'hook type': 'forward'
    },
    'emb_pre_attn_post_ln': {
        'module': 'gpt.h.{layer}.ln_1',         # Pre-attention layer norm
        'hook type': 'forward'
    },
    'emb_post_attn_pre_ln': {
        'module': 'gpt.h.{layer}.ln_2',         # Post-attention layer norm
        'hook type': 'forward_pre'
    },
    'emb_post_mlp_residual': {
        'module': 'gpt.h.{layer}.mlp',          # MLP output
        'hook type': 'forward'
    },
    'layer_output': {
        'module': 'gpt.h.{layer}',              # Full layer output
        'hook type': 'forward'
    },
}
```

### IndexTTS2 Steering Config (from paper Table 4)
- **Layers**: 6, 8, 1 (top-3 by discriminability)
- **Operator**: `attn_output` (pre-hook on `c_proj`)
- **Hidden dim**: 1024
- **Alpha range**: 0.0 to 6.0 (stable, intelligibility-preserving)
- **Default alpha**: 3.0

### Adapting to Chatterbox T3 (GPT2-medium, 24 layers, hidden 1024)
The T3 model structure would be: `t3.gpt.h.{layer}.attn.c_proj` (matching GPT2's `.h` layer indexing).

Hook path adaptation:
```python
# For Chatterbox T3 (GPT2-medium):
CHATTERBOX_OP_DICT = {
    'attn_output': {
        'module': 'gpt.h.{layer}.attn.c_proj',  # or t3.gpt.h.{layer}...
        'hook type': 'forward_pre'
    },
    # ... other operators
}
```

The implementation pattern is:
1. Register `forward_pre_hook` on `c_proj` modules at selected layers
2. Hook intercepts input to `c_proj` (the attention output before projection)
3. Apply `x += alpha * steering_vector` (translation) or norm-preserving variant
4. Run normal T3 generation
5. Remove hooks

---

## 6. Zero-Shot Steering Feasibility

### Can we steer WITHOUT any training data?

**Short answer: No, not truly zero-shot. You need a small calibration set.**

### What CoCoEmo requires:
- **Training data for extraction**: Paired emotional/neutral audio (same speaker, same text)
  - Datasets used: ESD + RAVDESS + CREMA-D (20,691 utterances total)
  - But the precomputed vectors are shipped — you can use those directly if your model matches
- **However**: The precomputed vectors are for CosyVoice2 and IndexTTS2, NOT for Chatterbox T3
- **For Chatterbox T3**: You need to extract your own steering vectors

### Minimum data needed:
- **~10-30 paired audio samples** per emotion (emotional + neutral, same speaker, same text)
- This is a "few-shot" calibration, not full training
- ESD (Emotional Speech Dataset) provides exactly this: same text spoken in different emotions by same speaker
- The extraction is a single forward pass per sample — very fast

### What EmoSteer-TTS requires:
- "Few-shot neutral and target activations" — same concept, small calibration set
- Also training-free, just needs reference audio

### What EmoShift requires:
- 300 training utterances + 5 epochs of training
- NOT training-free — requires learning the W_e matrices
- Less relevant for our use case

### Practical approach for Chatterbox T3:
1. **Option A (recommended)**: Use ESD dataset (free, ~350 utterances per emotion, same speaker/text pairs)
   - Run T3 forward pass on ~20-30 emotional audio samples → extract activations
   - Run T3 forward pass on matched neutral samples → extract activations
   - Compute mean-difference → steering vectors
   - This takes minutes, not hours

2. **Option B (minimal)**: Record ~10 paired samples yourself (same sentence in angry/happy/sad/neutral)
   - Even 10 samples per emotion should give reasonable steering vectors

3. **Option C (truly zero-shot, risky)**: Try using IndexTTS2's precomputed vectors directly
   - Both are GPT2-based, hidden dim 1024
   - But different models have different internal representations
   - Likely won't work well — vectors are model-specific

### Key insight from CoCoEmo paper:
The precomputed vectors are shipped specifically because they were extracted from the specific backbone model. The README says: "Precomputed steering vectors are shipped in `steering_vectors/`, so you can synthesize steered speech immediately, no dataset or training needed" — but this only works because the vectors match the backbone model. For a different model (Chatterbox T3), you must extract new vectors.

---

## 7. Feasibility Assessment for Chatterbox T3

### Favorable factors:
- ✅ T3 is GPT2-medium (same architecture family as IndexTTS2's GPT)
- ✅ Hidden dim 1024 (same as IndexTTS2)
- ✅ CoCoEmo already has GPT2-style hook code (`_core_indextts.py`)
- ✅ The `emotion_adv` slot in T3Cond suggests the model was designed with emotion conditioning in mind
- ✅ Activation steering is architecture-agnostic — just needs forward hooks
- ✅ No fine-tuning required — just hook registration at inference time

### Challenges:
- ⚠️ Need to adapt hook paths from `tts.gpt.gpt.h.{layer}` to T3's actual module path
- ⚠️ Need to run discriminability analysis to find best layers for T3 (may differ from IndexTTS2's 6,8,1)
- ⚠️ Need a small calibration dataset (ESD or similar) to extract steering vectors
- ⚠️ T3 was trained with emotion_adv disabled — the emotion representations may be weaker/absent
- ⚠️ Alpha range needs empirical tuning per model

### Implementation plan:
1. Map T3's GPT2 module structure (find `c_proj` in attention layers)
2. Create adapted op_dict for T3's path (e.g., `t3.gpt.h.{layer}.attn.c_proj`)
3. Get ESD dataset (or record ~10-20 paired samples)
4. Run extraction: forward pass emotional + neutral audio through T3, hook activations
5. Compute mean-difference steering vectors
6. Run discriminability analysis to find best layers
7. Register injection hooks during T3 inference with alpha tuning
8. Evaluate emotion transfer quality

### Code adaptation:
The CoCoEmo `_core_indextts.py` code can be directly adapted. Key changes:
- Module path: `gpt.h.{layer}.attn.c_proj` → T3's equivalent path
- Model access: `tts.gpt` → T3 model handle
- Audio processing: Use Chatterbox's tokenizer/codec instead of IndexTTS2's
- The steering operators (`translation_op_`, `norm_preserve_steer_op_`) and hook classes (`PreHookInject`, `ForwardHookInject`) are model-agnostic and need no changes