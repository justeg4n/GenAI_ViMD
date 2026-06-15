# VietSpeech AI — API Reference Guide

Base URL: `http://localhost:8000` (development) | `https://your-domain.com` (production)

---

## Authentication

All AI routes support optional JWT authentication. Auth token is passed as a Bearer token:

```
Authorization: Bearer <access_token>
```

### POST /login

```json
// Request
{ "username": "user@example.com", "password": "secret" }

// Response 200
{
  "access_token": "eyJ...",
  "refresh_token": "eyJ...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

### POST /register

```json
{ "username": "user@example.com", "password": "secret", "full_name": "Nguyen Van A" }
```

---

## Audio Upload

### POST /api/v1/upload-audio

Upload an audio file and receive an `audio_id` for use in downstream API calls.

**Content-Type**: `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | File | ✓ | Audio file (WAV/MP3/FLAC/OGG) |
| `language` | string | — | Language hint (`vi`, `ede`, `bah`) |

**Response 200**:
```json
{
  "audio_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "duration_s": 4.82,
  "sample_rate": 16000,
  "channels": 1,
  "file_size_bytes": 154112,
  "format": "wav"
}
```

**Errors**: `413` file too large (>100MB) | `415` unsupported format | `422` corrupt audio

---

## ASR (Speech Recognition)

### POST /api/v1/transcribe

Transcribe audio referenced by `audio_id`.

```json
// Request
{
  "audio_id": "3fa85f64-...",
  "language": "vi",
  "task": "transcribe",
  "use_vad": true,
  "use_denoise": true,
  "province_hint": "HAN"
}
```

| Field | Values | Default | Description |
|-------|--------|---------|-------------|
| `audio_id` | UUID | — | From `/upload-audio` |
| `language` | `vi` `ede` `bah` | `vi` | Target language |
| `task` | `transcribe` `translate` | `transcribe` | `translate` outputs English |
| `use_vad` | bool | `true` | Apply Voice Activity Detection |
| `use_denoise` | bool | `true` | Apply DeepFilterNet denoising |
| `province_hint` | province code | `null` | Dialect-aware Whisper prompt |

**Response 200**:
```json
{
  "audio_id": "3fa85f64-...",
  "text": "Xin chào, hôm nay thời tiết đẹp lắm.",
  "language": "vi",
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 2.5,
      "text": "Xin chào, hôm nay",
      "avg_logprob": -0.12
    }
  ],
  "confidence": 0.89,
  "latency_ms": 245.3,
  "model_used": "whisper-large-v3"
}
```

### POST /api/v1/transcribe/file

Upload + transcribe in one step.

**Content-Type**: `multipart/form-data`

| Field | Type | Description |
|-------|------|-------------|
| `file` | File | Audio file |
| `language` | string | Language code |
| `use_vad` | bool | Apply VAD |
| `use_denoise` | bool | Apply denoising |

---

## Dialect Identification

### POST /api/v1/detect-accent

Identify Vietnamese provincial dialect, region, and ethnic language.

**Content-Type**: `multipart/form-data`

| Field | Type | Description |
|-------|------|-------------|
| `file` | File | Audio file |
| `audio_id` | string | Alternative to file upload |

**Response 200**:
```json
{
  "province": "HUE",
  "province_name": "Thừa Thiên Huế",
  "province_confidence": 0.82,
  "region": "Central",
  "region_confidence": 0.97,
  "ethnicity": "viet",
  "ethnicity_confidence": 0.99,
  "top_provinces": [
    {"province": "HUE", "confidence": 0.82},
    {"province": "DAN", "confidence": 0.11},
    {"province": "QNM", "confidence": 0.07}
  ],
  "latency_ms": 178.4,
  "model_used": "wavlm-large"
}
```

---

## Text-to-Speech

### POST /api/v1/generate-speech

Synthesize speech from text.

```json
// Request
{
  "text": "Xin chào thế giới!",
  "language": "vi",
  "speaker_id": null,
  "province_code": "HAN",
  "speed": 1.0,
  "pitch_shift": 0.0
}
```

| Field | Values | Default | Description |
|-------|--------|---------|-------------|
| `text` | string | — | Input text (max 500 chars) |
| `language` | `vi` `ede` `bah` `en` | `vi` | Target language |
| `province_code` | province code | `null` | Dialect conditioning |
| `speed` | 0.5–2.0 | `1.0` | Speech rate |
| `pitch_shift` | -12 to 12 | `0.0` | Pitch in semitones |

**Response 200**: `audio/wav` binary stream

**Response Headers**:
```
Content-Type: audio/wav
X-Duration-S: 2.3
X-Model-Used: xtts-v2
X-Latency-Ms: 1240
```

---

## Voice Cloning

### POST /api/v1/voice-clone

Clone a voice from 5-30s reference audio and synthesize new speech.

**Content-Type**: `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `reference_audio` | File | ✓ | Reference WAV (5-30 seconds) |
| `text` | string | ✓ | Text to synthesize |
| `language` | string | — | Language code |
| `speed` | float | — | Speech rate |

**Response 200**: `audio/wav` binary stream

---

## Machine Translation

### POST /api/v1/translate

Translate text between supported language pairs.

```json
// Request
{
  "text": "Xin chào, bạn có khỏe không?",
  "source_lang": "vi",
  "target_lang": "ede",
  "max_length": 512,
  "num_beams": 5
}
```

**Supported language pairs**:
- `vi` ↔ `ede` (Vietnamese ↔ Ede)
- `vi` ↔ `bah` (Vietnamese ↔ Bahnar)
- `vi` ↔ `en` (Vietnamese ↔ English)

**Response 200**:
```json
{
  "source_text": "Xin chào, bạn có khỏe không?",
  "translated_text": "Khoa êa, ih dưk jăk mơ̆ng?",
  "source_lang": "vi",
  "target_lang": "ede",
  "confidence": 0.73,
  "latency_ms": 412.1,
  "model_used": "mbart-50-finetuned",
  "tokens_used": 12
}
```

---

## Health & Metrics

### GET /health

```json
{
  "status": "healthy",
  "version": "2.0.0",
  "models": {
    "asr": "loaded",
    "dialect": "loaded",
    "tts": "not_loaded",
    "translation": "loaded"
  },
  "uptime_s": 3600,
  "gpu_available": true
}
```

### GET /metrics

Returns Prometheus metrics in text format.

---

## Error Responses

All errors follow this format:

```json
{
  "detail": "Audio ID '3fa8...' not found",
  "request_id": "b5c0a216-e4d4-4bc6-9c38-41861fa7d9bb"
}
```

| Status | Meaning |
|--------|---------|
| `400` | Bad request — missing/invalid parameters |
| `401` | Unauthorized — invalid or missing JWT |
| `403` | Forbidden — insufficient permissions |
| `404` | Not found — audio_id doesn't exist |
| `413` | File too large |
| `415` | Unsupported media type |
| `422` | Validation error |
| `429` | Rate limit exceeded (60 req/min) |
| `500` | Internal server error |
| `503` | Service unavailable — model not loaded |

---

## Rate Limits

| Endpoint | Limit |
|----------|-------|
| `/api/v1/transcribe*` | 60 req/min |
| `/api/v1/generate-speech` | 30 req/min |
| `/api/v1/voice-clone` | 10 req/min |
| `/api/v1/translate` | 120 req/min |
| All others | 300 req/min |

Rate limit headers returned on every response:
```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1718323200
```
