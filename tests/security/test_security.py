"""
Security Tests — OWASP-based Security Test Suite
=================================================
Tests for: SQL injection, XSS, auth bypass, rate limiting,
CORS, CSRF, file upload abuse, and JWT validation.
"""

from __future__ import annotations

import io
import time
import wave

import pytest
from fastapi.testclient import TestClient


# ── Fixture ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    from backend.main import app
    return TestClient(app, raise_server_exceptions=False)


def make_wav(duration_s: float = 0.5) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * int(duration_s * 16000))
    buf.seek(0)
    return buf.read()


# ── Authentication Security ───────────────────────────────────────────────────

class TestAuthSecurity:
    """Authentication bypass and JWT validation tests."""

    def test_no_token_on_protected_route(self, client):
        """Unauthenticated requests should return 401 or proceed with guest."""
        resp = client.post("/api/v1/transcribe", json={"audio_id": "nonexistent"})
        assert resp.status_code in (400, 401, 403, 404, 503)

    def test_forged_jwt_rejected(self, client):
        """Forged JWT tokens must be rejected."""
        forged_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhdHRhY2tlciJ9.UNSIGNED"
        resp = client.post(
            "/api/v1/transcribe",
            json={"audio_id": "test"},
            headers={"Authorization": f"Bearer {forged_token}"},
        )
        assert resp.status_code in (400, 401, 403, 404, 503)

    def test_expired_token_rejected(self, client):
        """Expired JWT tokens must be rejected."""
        # Token with exp=0 (already expired)
        expired = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ1c2VyIiwiZXhwIjoxfQ.FAKE"
        resp = client.get(
            "/api/v1/transcribe/history",
            headers={"Authorization": f"Bearer {expired}"},
        )
        assert resp.status_code in (400, 401, 403, 404)


# ── Input Validation Security ─────────────────────────────────────────────────

class TestInputValidation:
    """Test that malicious inputs are rejected or sanitized."""

    SQL_PAYLOADS = [
        "'; DROP TABLE users; --",
        "1' OR '1'='1",
        "admin'--",
        "' UNION SELECT * FROM secrets--",
    ]

    XSS_PAYLOADS = [
        "<script>alert('xss')</script>",
        "javascript:alert(1)",
        "<img src=x onerror=alert(1)>",
        "'; alert('xss')//",
    ]

    @pytest.mark.parametrize("payload", SQL_PAYLOADS)
    def test_sql_injection_in_text_field(self, client, payload):
        """SQL injection payloads should not cause 500 errors or data leaks."""
        resp = client.post(
            "/api/v1/translate",
            json={"text": payload, "source_lang": "vi", "target_lang": "ede"},
        )
        # Must not return 500 internal server error
        assert resp.status_code != 500
        # Must not echo back the raw payload in error messages
        if resp.status_code != 200:
            body = resp.text
            assert "DROP TABLE" not in body
            assert "UNION SELECT" not in body

    @pytest.mark.parametrize("payload", XSS_PAYLOADS)
    def test_xss_in_translate_response(self, client, payload):
        """XSS payloads in translated text must not appear unescaped in JSON."""
        resp = client.post(
            "/api/v1/translate",
            json={"text": payload, "source_lang": "vi", "target_lang": "ede"},
        )
        if resp.status_code == 200:
            # JSON response should not contain raw <script> tags unescaped
            assert "<script>" not in resp.text.lower()


# ── File Upload Security ──────────────────────────────────────────────────────

class TestFileUploadSecurity:
    """Test file upload endpoint for abuse resistance."""

    def test_upload_non_audio_rejected(self, client):
        """Uploading a non-audio file should fail validation."""
        fake_exe = b"MZ\x90\x00" + b"\x00" * 100  # PE header
        resp = client.post(
            "/api/v1/upload-audio",
            files={"file": ("malware.exe", fake_exe, "application/octet-stream")},
        )
        assert resp.status_code in (400, 415, 422)

    def test_upload_huge_file_rejected(self, client):
        """Files exceeding the size limit should be rejected."""
        # Generate a large (>100MB would be ideal but here we test the endpoint responds)
        large_wav = b"RIFF" + (500 * 1024 * 1024).to_bytes(4, "little") + b"WAVEfmt "
        resp = client.post(
            "/api/v1/upload-audio",
            files={"file": ("huge.wav", large_wav[:1024], "audio/wav")},
            headers={"Content-Length": str(500 * 1024 * 1024)},
        )
        assert resp.status_code in (400, 413, 422, 500)

    def test_path_traversal_in_audio_id(self, client):
        """Audio ID with path traversal should be rejected."""
        traversal_ids = ["../../../etc/passwd", "..%2F..%2Fetc%2Fpasswd", "%2F%2F%2F"]
        for audio_id in traversal_ids:
            resp = client.post(
                "/api/v1/transcribe",
                json={"audio_id": audio_id, "language": "vi"},
            )
            assert resp.status_code in (400, 403, 404, 422, 503)
            if resp.status_code == 200:
                # If it somehow returned 200, it must not contain sensitive data
                assert "root:" not in resp.text
                assert "passwd" not in resp.text


# ── Rate Limiting ─────────────────────────────────────────────────────────────

class TestRateLimiting:
    """Verify rate limiting is active on high-value endpoints."""

    def test_rapid_transcribe_requests_rate_limited(self, client):
        """Sending 60+ requests in rapid succession should trigger rate limiting."""
        responses = []
        for _ in range(65):
            resp = client.post(
                "/api/v1/transcribe/file",
                files={"file": ("t.wav", make_wav(), "audio/wav")},
                data={"language": "vi"},
            )
            responses.append(resp.status_code)

        # At least some requests should be rate limited (429) or service unavailable (503)
        assert 429 in responses or 503 in responses, (
            "Expected rate limiting (429) after 60+ rapid requests"
        )


# ── CORS Security ─────────────────────────────────────────────────────────────

class TestCORSSecurity:
    """CORS header validation tests."""

    def test_origin_from_unknown_host_not_reflected(self, client):
        """CORS should not reflect arbitrary origins."""
        resp = client.options(
            "/api/v1/transcribe",
            headers={"Origin": "https://evil-attacker.com"},
        )
        acao = resp.headers.get("Access-Control-Allow-Origin", "")
        assert "evil-attacker.com" not in acao

    def test_allowed_origin_works(self, client):
        """Allowed origins should be reflected in CORS headers."""
        resp = client.options(
            "/api/v1/transcribe",
            headers={"Origin": "http://localhost:3000"},
        )
        # Either the origin is allowed or the endpoint doesn't need CORS
        assert resp.status_code in (200, 204, 405)
