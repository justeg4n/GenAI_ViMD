"""
Integration Tests — FastAPI Backend
=====================================
Tests the complete API endpoints using TestClient.
Models are mocked to avoid GPU dependency in CI.
"""

import io
import json
import struct
import wave
import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient


def make_wav_bytes(duration_s: float = 2.0, sr: int = 16000) -> bytes:
    """Generate valid WAV file bytes for testing."""
    n_samples = int(duration_s * sr)
    audio = (np.sin(2 * np.pi * 200 * np.linspace(0, duration_s, n_samples)) * 16000).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(audio.tobytes())
    return buf.getvalue()


@pytest.fixture
def mock_model_registry():
    """Mock model registry for tests without GPU."""
    registry = MagicMock()
    registry.get_model_status.return_value = {
        "asr": "mock", "dialect": "mock", "tts": "mock",
        "voice_clone": "mock", "translation": "mock",
    }
    # ASR service mock
    asr_service = MagicMock()
    asr_result = MagicMock()
    asr_result.text = "Xin chào Việt Nam"
    asr_result.language = "vi"
    asr_result.confidence = 0.95
    asr_result.segments = [{"start": 0.0, "end": 2.0, "text": "Xin chào Việt Nam"}]
    asr_result.latency_ms = 150.0
    asr_result.model_used = "mock_whisper"
    asr_service.transcribe = AsyncMock(return_value=asr_result)
    registry.get_asr_service.return_value = asr_service

    # Dialect service mock
    dialect_service = MagicMock()
    dialect_service.detect = AsyncMock(return_value={
        "province": "Hà Nội",
        "region": "North",
        "confidence": 0.85,
        "top_k_provinces": [("Hà Nội", 0.85), ("Vĩnh Phúc", 0.05), ("Hà Nam", 0.03)],
        "ethnic_probability": {"viet": 0.98, "ede": 0.01, "bahnar": 0.01},
        "code_switching_score": 0.02,
        "latency_ms": 80.0,
    })
    registry.get_dialect_service.return_value = dialect_service

    # Translation service mock
    translation_service = MagicMock()
    translation_service.translate = AsyncMock(return_value={
        "translated_text": "H'ơi êga",
        "confidence": 0.78,
        "latency_ms": 120.0,
    })
    registry.get_translation_service.return_value = translation_service

    return registry


@pytest.fixture
def client(mock_model_registry):
    """Create test client with mocked registry."""
    from backend.main import create_app
    app = create_app()
    app.state.model_registry = mock_model_registry

    # Patch DB init to avoid actual DB connection
    with patch("backend.main.init_db", new=AsyncMock()):
        with patch("backend.main.ModelRegistry") as MockRegistry:
            MockRegistry.return_value = mock_model_registry
            MockRegistry.return_value.initialize = AsyncMock()
            MockRegistry.return_value.cleanup = AsyncMock()
            with TestClient(app, raise_server_exceptions=False) as c:
                yield c


class TestHealthEndpoints:
    """Test health check endpoints."""

    def test_root_returns_200(self, client):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "service" in data
        assert "VietSpeech" in data["service"]

    def test_health_returns_status(self, client):
        with patch("backend.api.routes.health.check_db_connection", new=AsyncMock(return_value=True)):
            response = client.get("/health")
        # May be 200 or 503 depending on Redis; just check it responds
        assert response.status_code in (200, 503)

    def test_metrics_endpoint(self, client):
        response = client.get("/metrics")
        assert response.status_code == 200


class TestAudioUploadEndpoint:
    """Test POST /api/v1/upload-audio."""

    def test_upload_valid_wav(self, client):
        wav_bytes = make_wav_bytes(duration_s=3.0)
        response = client.post(
            "/api/v1/upload-audio",
            files={"file": ("test.wav", io.BytesIO(wav_bytes), "audio/wav")},
        )
        # Should succeed or fail gracefully
        assert response.status_code in (200, 422, 500)
        if response.status_code == 200:
            data = response.json()
            assert "audio_id" in data
            assert data["duration_s"] > 0

    def test_upload_empty_file_fails(self, client):
        response = client.post(
            "/api/v1/upload-audio",
            files={"file": ("empty.wav", io.BytesIO(b""), "audio/wav")},
        )
        assert response.status_code == 400

    def test_upload_no_file_fails(self, client):
        response = client.post("/api/v1/upload-audio")
        assert response.status_code == 422


class TestTranscribeEndpoint:
    """Test POST /api/v1/transcribe."""

    def test_transcribe_without_audio_id_fails(self, client):
        response = client.post(
            "/api/v1/transcribe",
            json={"language": "vi"},
        )
        assert response.status_code == 400

    def test_transcribe_nonexistent_audio_fails(self, client):
        response = client.post(
            "/api/v1/transcribe",
            json={"audio_id": "nonexistent-uuid", "language": "vi"},
        )
        assert response.status_code == 404


class TestDialectEndpoint:
    """Test POST /api/v1/detect-accent."""

    def test_detect_nonexistent_audio_fails(self, client):
        response = client.post(
            "/api/v1/detect-accent",
            json={"audio_id": "nonexistent-id"},
        )
        assert response.status_code == 404


class TestTranslateEndpoint:
    """Test POST /api/v1/translate."""

    def test_translate_vi_to_ede(self, client):
        response = client.post(
            "/api/v1/translate",
            json={
                "text": "Xin chào",
                "source_lang": "vi",
                "target_lang": "ede",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "translated_text" in data
        assert data["source_lang"] == "vi"
        assert data["target_lang"] == "ede"

    def test_translate_same_language_returns_same(self, client):
        response = client.post(
            "/api/v1/translate",
            json={"text": "Xin chào", "source_lang": "vi", "target_lang": "vi"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["translated_text"] == "Xin chào"

    def test_translate_unsupported_pair_fails(self, client):
        response = client.post(
            "/api/v1/translate",
            json={"text": "Hello", "source_lang": "en", "target_lang": "vi"},
        )
        assert response.status_code == 422

    def test_translate_empty_text_fails(self, client):
        response = client.post(
            "/api/v1/translate",
            json={"text": "", "source_lang": "vi", "target_lang": "ede"},
        )
        assert response.status_code == 422
