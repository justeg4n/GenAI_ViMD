# VietSpeech AI — Architecture Guide

## Overview

VietSpeech AI is a production-grade speech AI platform supporting all 63 Vietnamese provincial dialects, ethnic languages (Ede, Bahnar), ASR, TTS, voice cloning, and Vietnamese↔Ede/Bahnar machine translation.

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLIENT LAYER                             │
│  Flutter Mobile App (iOS/Android)  ←→  Web Browser (future)    │
└───────────────────────────┬─────────────────────────────────────┘
                            │ HTTPS / WebSocket
┌───────────────────────────▼─────────────────────────────────────┐
│                      NGINX REVERSE PROXY                        │
│           TLS termination · Rate limiting · Static files        │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│                   FASTAPI BACKEND (backend/)                    │
│                                                                 │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────┐ ┌────────┐  │
│  │  /transcr│ │ /detect- │ │/generate │ │/transl │ │ /voice │  │
│  │   ibe    │ │  accent  │ │ -speech  │ │  ate   │ │  clone │  │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └───┬────┘ └───┬────┘  │
│       │             │             │           │          │       │
│  ┌────▼─────────────▼─────────────▼───────────▼──────────▼────┐ │
│  │              SERVICE LAYER (backend/services/)              │ │
│  │  ASRService  DialectService  TTSService  TranslationService │ │
│  └────────────────────────┬────────────────────────────────────┘ │
│                           │                                     │
│  ┌────────────────────────▼────────────────────────────────────┐ │
│  │              MODEL REGISTRY (lazy loading)                  │ │
│  │     WavLM · Whisper · PhoWhisper · XTTS-v2 · mBART-50      │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                 │
│  ┌─────────────────┐  ┌──────────────────────────────────────┐  │
│  │  JWT Middleware  │  │  Rate Limit (Redis) · Audit Logging  │  │
│  └─────────────────┘  └──────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
         │ Celery tasks            │ DB writes
┌────────▼────────────┐  ┌────────▼──────────┐
│   CELERY WORKERS    │  │    POSTGRESQL     │
│  (GPU inference)    │  │  (user data, logs)│
│  Queue: Redis       │  └───────────────────┘
└─────────────────────┘
         │
┌────────▼────────────────────────────────────┐
│              MODEL FILES (storage/)         │
│  weights/asr/    weights/dialect/           │
│  weights/xtts-v2/  weights/translation/     │
│  (ONNX | TorchScript | HF weights)          │
└─────────────────────────────────────────────┘
```

## Component Descriptions

### Dataset Pipeline (`dataset/`)

| Module | Purpose |
|--------|---------|
| `ingestion/vimd_loader.py` | Load ViMD parquet shards (102h, 63 provinces) |
| `ingestion/ethnic_corpus_loader.py` | Load Ede/Bahnar text corpora |
| `validation/audio_validator.py` | Format/LUFS/duration validation |
| `validation/label_validator.py` | Label deduplication, silence removal |
| `feature/extractor.py` | Mel/MFCC/F0/speaker embeddings |
| `augmentation/audio_augment.py` | SpecAugment, noise, reverb, speed |
| `splits/splitter.py` | Stratified 80/10/10 split |
| `export/exporter.py` | HuggingFace Dataset format export |

### Model Inference (`models/`)

| Module | Model | Purpose |
|--------|-------|---------|
| `asr/whisper_inference.py` | Whisper-large-v3 + LoRA | Primary ASR |
| `asr/vad.py` | Silero VAD | Voice activity detection |
| `asr/denoiser.py` | DeepFilterNet | Audio denoising |
| `dialect/dialect_classifier.py` | WavLM-Large | 63-province + regional + ethnic classification |
| `tts/xtts_engine.py` | XTTS-v2 | Dialect-conditioned TTS synthesis |
| `voice_clone/cloner.py` | XTTS-v2 | Zero-shot voice cloning from 5-30s reference |
| `translation/mbart_translator.py` | mBART-50 | Vi↔Ede, Vi↔Bahnar translation |
| `prosody/gst_model.py` | GST | Global Style Token prosody conditioning |
| `export/onnx_exporter.py` | — | ONNX/TorchScript export for local inference |

### Training Notebooks (`training/notebooks/`)

| Notebook | Purpose |
|----------|---------|
| `00_setup_colab.ipynb` | Environment setup, Drive mount |
| `01_dataset_prep.ipynb` | Full data pipeline run |
| `02_dialect_identification.ipynb` | WavLM-Large fine-tuning |
| `03_asr_whisper.ipynb` | Whisper-large-v3 + QLoRA fine-tuning |
| `04_asr_phowhisper.ipynb` | PhoWhisper fine-tuning (fallback) |
| `05_translation_mbart.ipynb` | mBART-50 Vi↔Ede/Bahnar fine-tuning |
| `06_tts_xtts.ipynb` | XTTS-v2 dialect TTS fine-tuning |
| `07_voice_clone.ipynb` | Voice cloning evaluation |
| `08_export_onnx.ipynb` | ONNX/TorchScript export for local use |

### Backend API (`backend/`)

**Routes** (`backend/api/routes/`):

| Route | Method | Description |
|-------|--------|-------------|
| `/api/v1/upload-audio` | POST | Upload audio file, get `audio_id` |
| `/api/v1/transcribe` | POST | ASR from `audio_id` |
| `/api/v1/transcribe/file` | POST | Upload + transcribe in one step |
| `/api/v1/detect-accent` | POST | Dialect identification |
| `/api/v1/generate-speech` | POST | TTS synthesis |
| `/api/v1/voice-clone` | POST | Voice cloning |
| `/api/v1/translate` | POST | Machine translation |
| `/health` | GET | Health check |
| `/metrics` | GET | Prometheus metrics |

### Security Architecture

```
Request → [Rate Limit] → [JWT Auth] → [Audit Log] → [Route Handler]
                                                          │
                                                    [Input Validation (Pydantic)]
                                                          │
                                                    [Service Layer]
                                                          │
                                                    [Model (GPU)]
```

- **Auth**: JWT RS256 + refresh tokens (auth adapter preserves existing auth routes)
- **RBAC**: User/Admin roles; admin-only model management endpoints
- **Rate Limiting**: Redis sliding window, 60 req/min for transcription
- **Input Validation**: Pydantic models, audio format/size validation
- **Secrets**: Environment variables / AWS Secrets Manager in production
- **Audit Log**: All API calls logged with user ID, timestamp, duration

## Data Flow: ASR Pipeline

```
Audio File
    │
    ▼
[Upload: POST /upload-audio]
    │ Returns: audio_id (UUID)
    ▼
[Transcribe: POST /transcribe]
    │
    ├─[VAD: Silero]──────→ speech_timestamps
    │
    ├─[Denoise: DeepFilterNet]──→ clean_audio
    │
    ├─[Whisper-large-v3]──→ {text, segments, confidence}
    │
    └─[Fallback: PhoWhisper / Wav2Vec2]──→ text
    │
    ▼
Response: {text, language, segments, confidence, latency_ms, model_used}
```

## Model Loading Strategy

Models are **lazy-loaded** on first request, then cached in memory:

```python
# backend/services/model_registry.py
registry = ModelRegistry()
# Models loaded on first use, shared across requests
asr = registry.get_asr_service()
```

**Priority**: ONNX > TorchScript > HF weights (falls back down the chain)

## KPI Targets

| Metric | Target |
|--------|--------|
| Province Accuracy | ≥ 70% |
| Regional Accuracy | ≥ 95% |
| ASR WER | ≤ 8% |
| CER | ≤ 4% |
| TTS MOS | ≥ 4.3 |
| Speaker Similarity | ≥ 0.90 |
| Streaming Latency | ≤ 300ms |
