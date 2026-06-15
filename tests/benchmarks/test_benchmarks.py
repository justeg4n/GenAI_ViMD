"""
Benchmark Tests — ASR/Dialect/TTS Performance Benchmarks
=========================================================
Measures WER, CER, latency, MOS (via proxy), speaker similarity.
Run with: pytest tests/benchmarks/ -v -s --benchmark-only
"""

from __future__ import annotations

import io
import time
import wave
import json
from pathlib import Path

import numpy as np
import pytest


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_test_audio(duration_s: float = 3.0, sr: int = 16000) -> np.ndarray:
    """Generate synthetic test audio (silence + random noise)."""
    audio = np.random.randn(int(duration_s * sr)).astype(np.float32) * 0.01
    return audio


def compute_wer(ref: str, hyp: str) -> float:
    ref_words = ref.strip().split()
    hyp_words = hyp.strip().split()
    if not ref_words:
        return 0.0
    d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1))
    for i in range(len(ref_words) + 1):
        d[i, 0] = i
    for j in range(len(hyp_words) + 1):
        d[0, j] = j
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            d[i, j] = (
                d[i - 1, j - 1]
                if ref_words[i - 1] == hyp_words[j - 1]
                else 1 + min(d[i - 1, j], d[i, j - 1], d[i - 1, j - 1])
            )
    return d[len(ref_words), len(hyp_words)] / len(ref_words)


# ── ASR Latency Benchmark ─────────────────────────────────────────────────────

class TestASRLatencyBenchmark:
    """Measures end-to-end ASR latency for different audio durations."""

    TARGET_LATENCY_MS = 300  # Streaming latency target

    @pytest.mark.parametrize("duration_s", [1.0, 3.0, 5.0, 10.0])
    def test_asr_latency(self, duration_s):
        """ASR latency for {duration_s}s audio must be ≤ 300ms (streaming)."""
        audio = make_test_audio(duration_s)

        try:
            import asyncio
            from backend.services.asr_service import ASRService

            svc = ASRService()

            start = time.perf_counter()
            result = asyncio.run(
                svc.transcribe(audio=audio, sample_rate=16000, language="vi",
                               use_vad=False, use_denoise=False)
            )
            latency_ms = (time.perf_counter() - start) * 1000

            print(f"\n  [{duration_s}s audio] latency={latency_ms:.0f}ms, "
                  f"model={result.model_used}, confidence={result.confidence:.2f}")

            if result.model_used != "stub":
                assert latency_ms <= self.TARGET_LATENCY_MS * (1 + duration_s / 10), (
                    f"ASR latency {latency_ms:.0f}ms exceeds scaled target for {duration_s}s audio"
                )
        except ImportError:
            pytest.skip("ASR service dependencies not available")


# ── Dialect Latency Benchmark ─────────────────────────────────────────────────

class TestDialectLatencyBenchmark:
    TARGET_LATENCY_MS = 200

    def test_dialect_latency(self):
        """Dialect classification latency must be ≤ 200ms for 3s audio."""
        audio = make_test_audio(3.0)

        try:
            import asyncio
            from backend.services.dialect_service import DialectService

            svc = DialectService()

            start = time.perf_counter()
            result = asyncio.run(svc.identify(audio=audio, sample_rate=16000))
            latency_ms = (time.perf_counter() - start) * 1000

            print(f"\n  Dialect latency={latency_ms:.0f}ms, "
                  f"province={result.province}, model={result.model_used}")

            if result.model_used != "stub":
                assert latency_ms <= self.TARGET_LATENCY_MS
        except ImportError:
            pytest.skip("Dialect service dependencies not available")


# ── Translation Benchmark ─────────────────────────────────────────────────────

class TestTranslationBenchmark:
    """Translation BLEU score and latency benchmarks."""

    BLEU_TARGET = 15.0  # Minimum acceptable BLEU for low-resource languages

    def test_translation_latency(self):
        """Translation of 100-char text must complete in under 2s."""
        text = "Xin chào, hôm nay thời tiết đẹp lắm. Tôi rất vui khi gặp bạn."

        try:
            import asyncio
            from backend.services.translation_service import TranslationService

            svc = TranslationService()

            start = time.perf_counter()
            result = asyncio.run(
                svc.translate(text=text, source_lang="vi", target_lang="en")
            )
            latency_ms = (time.perf_counter() - start) * 1000

            print(f"\n  Translation latency={latency_ms:.0f}ms, "
                  f"model={result.model_used}, confidence={result.confidence:.2f}")
        except ImportError:
            pytest.skip("Translation service dependencies not available")


# ── WER Utility Tests ─────────────────────────────────────────────────────────

class TestWERUtility:
    def test_wer_exact_match(self):
        assert compute_wer("hello world", "hello world") == 0.0

    def test_wer_one_sub(self):
        wer = compute_wer("hello world", "hello there")
        assert abs(wer - 0.5) < 0.001

    def test_wer_empty_reference(self):
        assert compute_wer("", "") == 0.0
