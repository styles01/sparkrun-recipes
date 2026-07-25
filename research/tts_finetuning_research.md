# TTS Datasets & Resources for Fine-Tuning Expressive TTS Models

## Research Summary for Fine-Tuning Chatterbox Turbo with Emotion Control

---

## 1. Academic Datasets for Emotional/Expressive Speech

### ESD (Emotional Speech Dataset)
- **HF**: `sonchuate/ESD-dataset` (3.5k rows, audio modality, soundfolder format)
  - URL: https://huggingface.co/datasets/sonchuate/ESD-dataset
- **HF**: `oxschaos/esd_dataset` (1.75k rows, audio+text, parquet, Apache-2.0)
  - URL: https://huggingface.co/datasets/oxschaos/esd_dataset
  - Has columns: id, style (5 emotion classes), transcription, transcription_normalised, audio
  - **MOST RELEVANT** - has text + audio + emotion labels (5 styles)
- **Original**: Published by Kun Zhou et al. (Interspeech 2021)
  - 10 native English + 10 native Chinese speakers
  - 5 emotions: Neutral, Happy, Sad, Angry, Surprise
  - ~350 utterances per speaker per emotion = ~29k total utterances
  - Access: Free academic download from https://github.com/KunZhou9642/ESD
- **Relevance**: ★★★★★ - Ideal for our use case. Has audio + text transcript + emotion labels. 5 emotions is a manageable set for conditioning.

### IEMOCAP (Interactive Emotional Dyadic Motion Capture)
- **HF**: `AbstractTTS/IEMOCAP` (10k rows, audio+text, parquet, 1.18k downloads)
  - URL: https://huggingface.co/datasets/AbstractTTS/IEMOCAP
  - Columns: file, audio, emotion scores (frustrated, angry, happy, sad, excited, fearful, surprised, neutral, etc.) as float32 values 0.01-0.95
  - **This is dimension-level emotion annotations, not just categorical!**
- **HF**: `Ar4ikov/iemocap_audio_text` (10k rows, 385 downloads)
  - URL: https://huggingface.co/datasets/Ar4ikov/iemocap_audio_text
- **HF**: `tarasabkar/IEMOCAP_Speech` (5.53k rows)
  - URL: https://huggingface.co/datasets/tarasabkar/IEMOCAP_Speech
- **Original**: 10 actors (5 male, 5 female), ~12 hours of improvised dyadic interactions
  - 9 emotion categories: anger, happiness, sadness, fear, excitement, frustration, surprise, neutral, disgust
  - Access: Academic license from https://sail.usc.edu/download/iemocap/ (requires registration)
- **Relevance**: ★★★★★ - Rich emotional labels including dimensional (valence/arousal) annotations. The float32 emotion scores in the HF version could directly map to Chatterbox's `emotion_adv` scalar.

### RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)
- **HF**: `confit/ravdess-parquet` (14.4k rows, audio+text, parquet, 444 downloads)
  - URL: https://huggingface.co/datasets/confit/ravdess-parquet
  - Columns: file, audio, emotion (8 classes), 5 folds with train/test splits
  - 8 emotions: neutral, calm, happy, sad, angry, fearful, disgust, surprised
  - Audio duration: 2.94-5.27 seconds
- **HF**: `mteb/RavdessZeroshot` (1.45k rows)
- **HF**: `Gray1y/RAVDESS_preprocessed_npy` (54 downloads)
- **Original**: 24 actors (12 male, 12 female), 7356 utterances
  - Access: Free download from https://zenodo.org/record/1188976
- **Relevance**: ★★★★☆ - Clean, well-labeled, compact. Good for initial experiments. Short utterances (3-5s) match TTS training format well.

### CREMA-D (Crowd-Sourced Emotional Multimodal Actors Dataset)
- **HF**: `razahtet/crema-d-audio` (7.44k rows, 1.13k downloads)
  - URL: https://huggingface.co/datasets/razahtet/crema-d-audio
- **HF**: `MahiA/CREMA-D` (7.44k rows, 665 downloads)
  - URL: https://huggingface.co/datasets/MahiA/CREMA-D
- **HF**: `confit/cremad-parquet` (7.44k rows, 511 downloads)
  - URL: https://huggingface.co/datasets/confit/cremad-parquet
- **HF**: `AbstractTTS/CREMA-D` (7.44k rows, 273 downloads)
  - URL: https://huggingface.co/datasets/AbstractTTS/CREMA-D
- **Original**: 91 actors, 7442 utterances, 6 emotions
  - Access: Free from https://github.com/CREMA-D/CREMA-D
- **Relevance**: ★★★★☆ - Large speaker diversity (91 actors) good for generalization

### TESS (Toronto Emotional Speech Set)
- Available via the `arshadjamal6002/audio-emotion-dataset-generator` pipeline (see below)
- 2 speakers (young female + older female), 2800 utterances, 7 emotions
- Access: Free from https://tspace.library.utoronto.ca/handle/1807/24487
- **Relevance**: ★★★☆☆ - Limited speakers but clean labels

### SAVEE (Surrey Audio-Visual Expressed Emotion)
- Available via the `arshadjamal6002/audio-emotion-dataset-generator` pipeline
- 4 male actors, 480 utterances, 7 emotions
- Access: Free from http://kahlan.eps.surrey.ac.uk/datasets/SAVEE/
- **Relevance**: ★★☆☆☆ - Small dataset, male-only

### Multi-Dataset Aggregation Tool
- **GitHub**: `arshadjamal6002/audio-emotion-dataset-generator` (0 stars)
  - URL: https://github.com/arshadjamal6002/audio-emotion-dataset-generator
  - Downloads RAVDESS, TESS, CREMA-D, SAVEE, ESD automatically
  - Resamples to 16kHz mono WAV, standardizes labels
  - ~25,000 English emotional speech audio files total
- **Relevance**: ★★★★☆ - Great starting point for building a combined training set

---

## 2. How People Fine-Tune Chatterbox or Similar Autoregressive TTS Models

### Chatterbox Architecture (from source code analysis)
- **Repo**: https://github.com/resemble-ai/chatterbox (25.5k stars, MIT license)
- **Turbo model**: 350M params, English-only, uses S3 speech tokens + GPT2 text tokenizer
- **Key file**: `src/chatterbox/tts_turbo.py` (296 lines)
- **T3 model config** (`src/chatterbox/models/t3/modules/t3_config.py`):
  - `text_tokens_dict_size=704` (English), `speech_tokens_dict_size=8194`
  - `max_text_tokens=2048`, `max_speech_tokens=4096`
  - `llama_config_name="Llama_520M"` (but Turbo is 350M)
  - `emotion_adv=True` ← **KEY: emotion conditioning is already built in!**
  - `speaker_embed_size=256`, `use_perceiver_resampler=True`
- **T3Cond** (`src/chatterbox/models/t3/modules/cond_enc.py`):
  - `speaker_emb`: Tensor (256-dim)
  - `clap_emb`: Optional (not implemented yet, commented as TODO)
  - `cond_prompt_speech_tokens`: Optional
  - `cond_prompt_speech_emb`: Optional
  - `emotion_adv`: Optional[Tensor] = 0.5 ← **single scalar, projected via Linear(1, n_channels)**
- **T3CondEnc.forward()**: Concatenates speaker_emb + clap_emb + cond_prompt_speech_emb + emotion_adv into cond_embeds
- **Critical insight**: The `emotion_adv` is a SINGLE SCALAR (default 0.5) projected through a single linear layer. This is very similar to the "exaggeration" parameter in the base Chatterbox model. Fine-tuning would involve:
  1. Training with different `emotion_adv` values paired with corresponding emotional audio
  2. The model already has the infrastructure - just needs training data with emotion labels

### Chatterbox Fine-Tuning Issues
- 44 GitHub issues mention fine-tuning/training (39 open, 5 closed)
  - URL: https://github.com/resemble-ai/chatterbox/issues?q=fine-tune+OR+finetune+OR+fine+tune+OR+training
- **No official fine-tuning script provided** by Resemble AI
- Community forks:
  - `SamirWagle/chatterbox-nepali` - Nepali fine-tuned Chatterbox with training tools
    - URL: https://github.com/SamirWagle/chatterbox-nepali
  - `randombk/chatterbox-vllm` (379 stars) - vLLM port, could enable faster training
    - URL: https://github.com/randombk/chatterbox-vllm
  - `petermg/Chatterbox-TTS-Extended` (572 stars) - Extended for audiobooks
    - URL: https://github.com/petermg/Chatterbox-TTS-Extended

### CosyVoice Fine-Tuning (Similar Architecture)
- **Repo**: https://github.com/FunAudioLLM/CosyVoice (acknowledged by Chatterbox)
- `Jatshi/cosyvoice-emotion-tts` - Fine-tuned CosyVoice2 on ESD dataset with emotion alignment loss
  - URL: https://github.com/Jatshi/cosyvoice-emotion-tts (repo currently empty)
- `thesimanta-saha/cosyvoice3-bengali-emotion-tts` - Fine-tuning CosyVoice3 with emotion instruction control
  - URL: https://github.com/thesimanta-saha/cosyvoice3-bengali-emotion-tts
  - Full pipeline: data collection, Gemini enrichment, ASR verification, quality filtering
- **Relevance**: CosyVoice uses a similar autoregressive architecture (text→speech tokens) and is a proven path for emotion fine-tuning

---

## 3. Papers & Blog Posts on Emotion Control for Fast TTS Models

### EmoShift (ICASSP 2026) - MOST RELEVANT
- **arXiv**: https://arxiv.org/abs/2601.22873
- **Title**: "EmoShift: Lightweight Activation Steering for Enhanced Emotion-Aware Speech Synthesis"
- **Key insight**: Adds an "EmoSteer" layer that learns a steering vector for each target emotion. Only 10M trainable parameters (<1/30 of full fine-tuning). Outperforms both zero-shot and fully fine-tuned baselines.
- **Relevance**: ★★★★★ - Directly applicable to Chatterbox Turbo. The activation steering approach could work with the existing `emotion_adv` infrastructure without full fine-tuning.

### CoCoEmo (ICML 2026) - Training-Free Approach
- **arXiv**: Search "CoCoEmo: Composable and Controllable Human-Like Emotional TTS via Activation Steering"
- **GitHub**: https://github.com/wsssy/CoCoEmo (10 stars, MIT license)
- **Key insight**: Training-free framework for mixed-emotion speech synthesis via activation steering. Has `steering_vectors/` directory.
- **Relevance**: ★★★★★ - Could provide pre-computed steering vectors that work without any training data. Mixed-emotion capability is interesting for toddler-friendly TTS (e.g., "happy + calm").

### Task-Vector Arithmetic for Emotional Expressivity Control (June 2026)
- **arXiv**: https://arxiv.org/abs/2606.05367
- **Title**: "Task-Vector Arithmetic for Emotional Expressivity Control in Language-Model-Based Text-to-Speech"
- **Relevance**: ★★★★☆ - Uses task-vector arithmetic on LM-based TTS, directly relevant to Chatterbox's GPT2-based architecture.

### TED-TTS (ACL 2026)
- **GitHub**: https://github.com/Simon-leong/TED-TTS (10 stars)
- **Title**: "TED-TTS: Training-Free Intra-Utterance Emotion and Duration Control for Text-to-Speech Synthesis"
- **Relevance**: ★★★★☆ - Training-free, intra-utterance emotion control (varying emotion within a single utterance) - useful for storytelling.

### Controllable TTS Survey (EMNLP 2025)
- **arXiv**: https://arxiv.org/abs/2412.06602
- **Title**: "Towards Controllable Speech Synthesis in the Era of Large Language Models: A Systematic Survey"
- **Relevance**: ★★★★☆ - Comprehensive survey of controllable TTS methods, datasets, and evaluation. Good starting reference.

### UltraVoice (Oct 2025)
- **arXiv**: https://arxiv.org/abs/2510.22588
- **Title**: "UltraVoice: Scaling Fine-Grained Style-Controlled Speech Conversations for Spoken Dialogue Models"
- **Relevance**: ★★★☆☆ - Fine-grained style control, relevant for conversational TTS.

### Word-Level Emotional Expression Control (Sept 2025)
- **arXiv**: https://arxiv.org/abs/2509.24629
- **Relevance**: ★★★☆☆ - Word-level emotion control in zero-shot TTS.

### IndexTTS 2.5 (Jan 2026)
- **arXiv**: https://arxiv.org/abs/2601.03888
- **Title**: "IndexTTS 2.5 Technical Report"
- **Relevance**: ★★★☆☆ - Latest TTS foundation model with emotion control capabilities.

### TTS-1 Technical Report (July 2025)
- **arXiv**: https://arxiv.org/abs/2507.21138
- **Code**: https://github.com/inworld-ai/tts
- **Relevance**: ★★★☆☆ - Another autoregressive TTS with emotion control.

---

## 4. Synthetic Data Generation Approaches

### dscripka/synthetic_speech_dataset_generation (30 stars)
- **URL**: https://github.com/dscripka/synthetic_speech_dataset_generation
- **License**: Apache-2.0
- **Description**: TTS models and utilities to produce synthetic training datasets for other speech models
- Uses VITS model, supports GPU, slerp interpolation
- **Relevance**: ★★★☆☆ - Template for synthetic data generation pipeline, though older (3 years)

### SeifEldenOsama/TTS-synthetic-data-generation-using-gemeni-emotion-controll (2 stars)
- **URL**: https://github.com/SeifEldenOsama/TTS-synthetic-data-generation-using-gemeni-emotion-controll
- **Description**: Generates synthetic TTS datasets using Google Gemini TTS with multiple voices, styles, emotions, and automatic API key rotation
- **Relevance**: ★★★★☆ - Directly relevant! Uses a cloud TTS (Gemini) to generate emotion-labeled training data for fine-tuning another TTS model.

### 0xrushi/dailytalk-contiguous-generator (1 star)
- **URL**: https://github.com/0xrushi/dailytalk-contiguous-generator
- **Description**: Pipeline for generating emotion-tagged conversational audio datasets. Processes stereo audio, transcribes, tags emotions, reconstructs with TTS.
- **Relevance**: ★★★☆☆ - Conversational emotion tagging pipeline.

### Synthetic Data Strategy for Our Use Case
The most viable approach for toddler-friendly TTS:
1. **Use Chatterbox Base** (with exaggeration/cfg_weight params) to generate training data with varying emotion levels
2. **Label each generated sample** with the exaggeration value used → becomes the `emotion_adv` target for Turbo fine-tuning
3. **Generate diverse text** from children's stories/books with emotional context
4. **Use the Base model's emotion dial** as ground truth labels for Turbo training
5. This creates a self-distillation pipeline where Base teaches Turbo about emotion

---

## 5. Hugging Face Datasets Tagged for TTS/Speech Synthesis with Emotion

### TTS-AGI/balanced-emotion-dataset-majestrino-withtemporal-detailed-captions
- **URL**: https://huggingface.co/datasets/TTS-AGI/balanced-emotion-dataset-majestrino-withtemporal-detailed-captions
- **Size**: 482,594 samples, 40 emotion categories, 12,997 samples per category
- **Format**: WebDataset (tar files with FLAC audio + JSON metadata)
- **License**: CC-BY-4.0
- **Language**: English
- **Tags**: audio, emotion, caption, speech, balanced, webdataset
- **40 emotion categories** including: Amusement, Elation, Pleasure/Ecstasy, Contentment, Thankfulness, Affection, Infatuation, Hope/Optimism, Triumph, Pride, Interest, Awe, Astonishment/Surprise, Concentration, and more (including negative emotions)
- **Relevance**: ★★★★★ - Largest and most diverse emotion-labeled speech dataset on HF. 40 categories provide fine-grained emotion control. Balanced across categories.

### TTS-AGI/Emotion-Voice-Attribute-Reference-Snippets-DACVAE-Wave
- **URL**: https://huggingface.co/datasets/TTS-AGI/Emotion-Voice-Attribute-Reference-Snippets-DACVAE-Wave
- **Size**: 691k rows
- **Relevance**: ★★★★☆ - Very large, voice attribute reference snippets with emotion

### TTS-AGI/Emotion-Voice-Attribute-Reference-Snippets-DACVAE
- **URL**: https://huggingface.co/datasets/TTS-AGI/Emotion-Voice-Attribute-Reference-Snippets-DACVAE
- **Size**: 512k rows
- **Relevance**: ★★★★☆ - Companion dataset

### Hemg/Emotion-audio-Dataset
- **URL**: https://huggingface.co/datasets/Hemg/Emotion-audio-Dataset
- **Size**: 12.8k rows, 7 emotion classes, parquet format
- **Labels**: Angry, Disgust, Fear, Happy, Neutral, Pleasant Surprise, Sad
- **Audio duration**: 1.25-7.14 seconds
- **Relevance**: ★★★☆☆ - Clean, simple emotion classification dataset. No text transcripts though.

### Darknsu/emotion_audio_dataset_preprocess_claude
- **URL**: https://huggingface.co/datasets/Darknsu/emotion_audio_dataset_preprocess_claude
- **Size**: 17.2k rows
- **Relevance**: ★★★☆☆ - Preprocessed with Claude (likely AI-labeled emotions)

### UniDataPro/speech-emotion-recognition
- **URL**: https://huggingface.co/datasets/UniDataPro/speech-emotion-recognition
- **Size**: 69 rows
- **Relevance**: ★☆☆☆☆ - Too small for training

---

## Summary: Recommended Strategy for Fine-Tuning Chatterbox Turbo

### Best Datasets (Ranked by Relevance)
1. **ESD** (`oxschaos/esd_dataset`) - text + audio + 5 emotion styles, Apache-2.0
2. **IEMOCAP** (`AbstractTTS/IEMOCAP`) - text + audio + dimensional emotion scores
3. **TTS-AGI/balanced-emotion-dataset** - 482k samples, 40 emotion categories, CC-BY-4.0
4. **RAVDESS** (`confit/ravdess-parquet`) - text + audio + 8 emotions, clean format
5. **CREMA-D** (`confit/cremad-parquet`) - 7.4k samples, 6 emotions, diverse speakers

### Best Approaches (Ranked by Feasibility)
1. **Activation Steering** (EmoShift/CoCoEmo approach) - No training data needed, add steering vectors to existing model
2. **Self-Distillation** - Use Chatterbox Base's exaggeration/cfg_weight to generate labeled training data for Turbo
3. **Fine-tune emotion_adv layer** - The infrastructure exists (Linear(1, n_channels)), just needs paired (audio, text, emotion_value) training data
4. **Full fine-tuning with ESD/IEMOCAP** - More compute-intensive but most thorough

### Key Architecture Insight
Chatterbox Turbo already has `emotion_adv` infrastructure:
- `T3Cond.emotion_adv`: scalar (default 0.5)
- `T3CondEnc.emotion_adv_fc`: `nn.Linear(1, hp.n_channels, bias=False)`
- This means the model CAN accept emotion conditioning - it just wasn't trained with varied values
- Fine-tuning could be as simple as: encode audio → get S3 tokens, pair with text + emotion_adv value, train the T3 model to predict speech tokens given varied emotion_adv inputs