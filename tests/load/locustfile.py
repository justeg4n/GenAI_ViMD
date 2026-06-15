"""
Load Tests — Locust API Load Testing
=====================================
Run with:
    locust -f tests/load/locustfile.py --host=http://localhost:8000 \
           --users=50 --spawn-rate=5 --run-time=60s --headless

Targets:
  - /api/v1/transcribe/file   → P95 ≤ 300ms (streaming latency)
  - /api/v1/detect-accent     → P95 ≤ 200ms
  - /health                   → P99 ≤ 50ms
"""

from __future__ import annotations

import io
import os
import struct
import wave
from typing import Optional

from locust import HttpUser, between, events, task  # type: ignore


# ── Helpers ───────────────────────────────────────────────────────────────────

def generate_wav_bytes(duration_s: float = 1.0, sample_rate: int = 16000) -> bytes:
    """Generate a minimal silent WAV file as synthetic test audio."""
    n_samples = int(duration_s * sample_rate)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * n_samples)
    buf.seek(0)
    return buf.read()


TEST_AUDIO_1S = generate_wav_bytes(1.0)
TEST_AUDIO_5S = generate_wav_bytes(5.0)


# ── Load Test User ────────────────────────────────────────────────────────────

class VietSpeechUser(HttpUser):
    """Simulates a typical API user hitting various endpoints."""

    wait_time = between(0.5, 2.0)
    _token: Optional[str] = None

    def on_start(self) -> None:
        """Authenticate before running tasks."""
        resp = self.client.post(
            "/login",
            json={"username": "testuser", "password": "testpassword"},
        )
        if resp.status_code == 200:
            data = resp.json()
            self._token = data.get("access_token")
        else:
            self._token = None

    def _headers(self) -> dict:
        headers = {"Accept": "application/json"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        return headers

    @task(3)
    def health_check(self) -> None:
        """GET /health — should always be fast."""
        with self.client.get(
            "/health",
            catch_response=True,
            name="/health",
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"Health check failed: {resp.status_code}")

    @task(5)
    def transcribe_1s(self) -> None:
        """POST /api/v1/transcribe/file — 1s audio."""
        with self.client.post(
            "/api/v1/transcribe/file",
            files={"file": ("test_1s.wav", TEST_AUDIO_1S, "audio/wav")},
            data={"language": "vi", "use_vad": "true", "use_denoise": "false"},
            headers=self._headers(),
            catch_response=True,
            name="/api/v1/transcribe/file (1s)",
        ) as resp:
            if resp.status_code == 200:
                latency = resp.elapsed.total_seconds() * 1000
                if latency > 300:
                    resp.failure(f"P95 target missed: {latency:.0f}ms > 300ms")
            elif resp.status_code in (503, 429):
                resp.failure(f"Service unavailable: {resp.status_code}")

    @task(3)
    def detect_dialect(self) -> None:
        """POST /api/v1/detect-accent — 1s audio."""
        with self.client.post(
            "/api/v1/detect-accent",
            files={"file": ("test_1s.wav", TEST_AUDIO_1S, "audio/wav")},
            headers=self._headers(),
            catch_response=True,
            name="/api/v1/detect-accent",
        ) as resp:
            if resp.status_code not in (200, 422, 503):
                resp.failure(f"Unexpected status: {resp.status_code}")

    @task(2)
    def translate(self) -> None:
        """POST /api/v1/translate."""
        with self.client.post(
            "/api/v1/translate",
            json={
                "text": "Xin chào thế giới",
                "source_lang": "vi",
                "target_lang": "en",
            },
            headers=self._headers(),
            catch_response=True,
            name="/api/v1/translate",
        ) as resp:
            if resp.status_code not in (200, 422, 503):
                resp.failure(f"Unexpected status: {resp.status_code}")

    @task(1)
    def tts_generate(self) -> None:
        """POST /api/v1/generate-speech."""
        with self.client.post(
            "/api/v1/generate-speech",
            json={
                "text": "Xin chào",
                "language": "vi",
                "speed": 1.0,
            },
            headers=self._headers(),
            catch_response=True,
            name="/api/v1/generate-speech",
        ) as resp:
            if resp.status_code not in (200, 422, 503):
                resp.failure(f"Unexpected status: {resp.status_code}")


# ── Event Listeners ───────────────────────────────────────────────────────────

@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Print P95/P99 latency summary on stop."""
    stats = environment.stats
    for name, stat in stats.entries.items():
        if stat.num_requests > 0:
            print(
                f"\n{name}: "
                f"reqs={stat.num_requests}, "
                f"fails={stat.num_failures}, "
                f"P50={stat.get_response_time_percentile(0.50):.0f}ms, "
                f"P95={stat.get_response_time_percentile(0.95):.0f}ms, "
                f"P99={stat.get_response_time_percentile(0.99):.0f}ms"
            )
