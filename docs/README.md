# 🇻🇳 DISA Travel AI Platform

**Production-grade Vietnamese Multi-Dialect Speech AI**

> ASR · Dialect Identification · TTS · Voice Cloning · Vi↔Ede/Bahnar Translation

---

## Overview

VietSpeech AI is a comprehensive speech intelligence platform covering:

- **63-province dialect detection** across all Vietnamese provinces
- **ASR** using Whisper Large-v3 + LoRA fine-tuning on ViMD (102.56h, 18,949 utterances)
- **TTS** using XTTS-v2 with dialect/prosody preservation
- **Voice Cloning** from 5–30 second reference audio
- **Translation** between Vietnamese ↔ Ede and Vietnamese ↔ Bahnar
- **Ethnic language detection** (Viet / Ede / Bahnar code-switching)

## KPI Targets

| Metric | Target | Model |
|--------|--------|-------|
| Province Accuracy | ≥ 70% | WavLM-Large |
| Regional Accuracy | ≥ 95% | WavLM-Large |
| ASR WER | ≤ 8% | Whisper Large-v3 |
| CER | ≤ 4% | Whisper Large-v3 |
| MOS | ≥ 4.3 | XTTS-v2 |
| Speaker Similarity | ≥ 0.90 | XTTS-v2 |
| Streaming Latency | ≤ 300ms | Pipeline |

## Quick Start

```bash
# 1. Setup environment
make setup

# 2. Start backend
make backend

# 3. Run tests
make test

# 4. Start full stack
make docker-up
```

## Project Structure

```
vietspeech-ai/
├── dataset/            # Data pipeline (ingestion → validation → features → export)
├── models/             # ML model inference wrappers
│   ├── asr/            # Whisper + VAD + denoiser
│   ├── dialect/        # WavLM-Large 63-province classifier
│   ├── tts/            # XTTS-v2 synthesis engine
│   ├── voice_clone/    # Speaker embedding + voice cloning
│   ├── translation/    # mBART-50 Vi↔Ede/Bahnar
│   └── prosody/        # Global Style Token prosody model
├── training/
│   ├── notebooks/      # Google Colab training notebooks
│   └── configs/        # Training YAML configurations
├── backend/            # FastAPI backend
│   ├── api/            # Route handlers
│   ├── services/       # Business logic + model registry
│   ├── workers/        # Celery async tasks
│   ├── db/             # SQLAlchemy models + sessions
│   └── middleware/     # Auth, rate limiting, audit logging
├── mobile/             # Flutter mobile app
├── deployment/         # Docker + Compose configs
├── monitoring/         # Prometheus + Grafana configs
├── tests/              # Unit, integration, load, security tests
└── docs/               # Documentation
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/metrics` | Prometheus metrics |
| POST | `/api/v1/upload-audio` | Upload audio file |
| POST | `/api/v1/transcribe` | ASR transcription |
| POST | `/api/v1/detect-accent` | Dialect identification |
| POST | `/api/v1/generate-speech` | TTS synthesis |
| POST | `/api/v1/voice-clone` | Voice cloning |
| POST | `/api/v1/translate` | Vi↔Ede/Bahnar translation |

*Preserved existing auth routes: /login /logout /register /profile*

## Training on Google Colab

1. Open `training/notebooks/00_setup_colab.ipynb` in Colab
2. Set `HF_TOKEN` in Colab Secrets
3. Run all cells — auto-mounts Google Drive and installs deps
4. Proceed to dialect training: `02_dialect_identification.ipynb`
5. ASR training: `03_asr_whisper.ipynb`
6. TTS fine-tuning: `06_tts_xtts.ipynb`

## Dataset

**ViMD** (Vietnamese Multi-Dialect Dataset):
- Source: `ViMD/data/` — parquet shards
- 102.56 hours, 18,949 utterances, 63 provinces
- Train: 81.43h / Valid: 10.26h / Test: 10.87h
- Attributes: filename, text, length, province, region, speakerID, gender

Reference: [EMNLP 2024 — Dinh et al.](https://aclanthology.org/2024.emnlp-main.426)

## Ethnic Language Resources

- **Ede**: IJERT bilingual corpus (Vietnamese-Ede Machine Translation)
- **Bahnar**: Language resource paper + seed vocabulary

## Model Stack

| Task | Model | Fine-tuning |
|------|-------|-------------|
| ASR Primary | Whisper Large-v3 | QLoRA |
| ASR Secondary | PhoWhisper-large | LoRA |
| ASR Fallback | Wav2Vec2 | None |
| Dialect ID | WavLM-Large | Full + Adapter |
| Translation | mBART-50 | Adapter Layers |
| TTS | XTTS-v2 | LoRA |
| Voice Clone | XTTS-v2 | Speaker encoder |
| VAD | Silero VAD | None |

## Security

- JWT + refresh tokens
- RBAC role-based access
- Redis-backed rate limiting (60 req/min default)
- AES encryption for sensitive data
- Audit logging of all requests
- CSRF + CORS protection
- Non-root Docker containers

## Observability

- **Prometheus** — metrics scraping (`/metrics`)
- **Grafana** — dashboards (`localhost:3000`)
- **MLflow** — experiment tracking (`localhost:5000`)
- **OpenTelemetry** — distributed tracing
- **Sentry** — error monitoring

## License

Research use. See ViMD dataset license (CC-BY-NC-ND-4.0).
