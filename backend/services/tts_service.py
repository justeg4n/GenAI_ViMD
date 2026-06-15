"""
TTS Service — Text-to-Speech Orchestration
==========================================
Synthesizes speech using XTTS-v2 with dialect/prosody controls.
Supports Vietnamese dialects, Ede, and Bahnar output.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class TTSResult:
    audio_bytes: bytes          # WAV bytes
    sample_rate: int = 22050
    duration_s: float = 0.0
    latency_ms: float = 0.0
    model_used: str = "xtts-v2"
    language: str = "vi"


class TTSService:
    """
    Production TTS service backed by XTTS-v2.
    Supports dialect-conditioned synthesis via speaker embeddings.
    """

    def __init__(self, model_registry: Any = None):
        self.model_registry = model_registry
        self._engine = None
        self._ready = False

    async def initialize(self) -> None:
        """Lazy-load the TTS engine."""
        if self._ready:
            return
        try:
            from models.tts.xtts_engine import XTTSEngine
            self._engine = XTTSEngine()
            await asyncio.get_event_loop().run_in_executor(None, self._engine.load)
            logger.info("✓ TTS XTTS-v2 engine loaded")
        except Exception as e:
            logger.warning(f"TTS engine load failed, using stub: {e}")
            self._engine = None
        self._ready = True

    async def synthesize(
        self,
        text: str,
        language: str = "vi",
        speaker_id: Optional[str] = None,
        province_code: Optional[str] = None,
        speed: float = 1.0,
        pitch_shift: float = 0.0,
        speaker_wav: Optional[bytes] = None,
    ) -> TTSResult:
        """
        Synthesize speech from text.

        Args:
            text: Input text to synthesize
            language: Language code ('vi', 'ede', 'bah')
            speaker_id: Speaker ID for voice selection
            province_code: Province code for dialect conditioning
            speed: Speech rate multiplier (0.5–2.0)
            pitch_shift: Pitch shift in semitones (-12 to 12)
            speaker_wav: Optional reference audio bytes for voice cloning

        Returns:
            TTSResult containing WAV bytes and metadata
        """
        await self.initialize()
        t_start = time.perf_counter()

        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        # Truncate to max 500 chars per request for safety
        text = text[:500].strip()

        if self._engine is not None:
            try:
                audio_array, sr = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self._engine.synthesize(
                        text=text,
                        language=language,
                        speaker_id=speaker_id,
                        province_code=province_code,
                        speed=speed,
                        pitch_shift=pitch_shift,
                        speaker_wav=speaker_wav,
                    ),
                )
                wav_bytes = self._array_to_wav(audio_array, sr)
                duration_s = len(audio_array) / sr
                model_used = "xtts-v2"
            except Exception as e:
                logger.error(f"TTS synthesis failed: {e}", exc_info=True)
                wav_bytes = self._generate_silence_wav(duration_s=1.0)
                duration_s = 1.0
                model_used = "fallback-silence"
        else:
            # Stub: return 1-second silence
            wav_bytes = self._generate_silence_wav(duration_s=1.0)
            duration_s = 1.0
            model_used = "stub"

        latency_ms = (time.perf_counter() - t_start) * 1000
        logger.info(
            f"TTS synthesized: {len(text)} chars → {duration_s:.1f}s audio, "
            f"latency={latency_ms:.0f}ms, model={model_used}"
        )

        return TTSResult(
            audio_bytes=wav_bytes,
            sample_rate=22050,
            duration_s=duration_s,
            latency_ms=latency_ms,
            model_used=model_used,
            language=language,
        )

    @staticmethod
    def _array_to_wav(audio_array: "np.ndarray", sample_rate: int) -> bytes:
        """Convert numpy float32 array to WAV bytes."""
        import numpy as np
        import soundfile as sf

        buf = BytesIO()
        # Normalize to [-1, 1]
        peak = np.abs(audio_array).max()
        if peak > 0:
            audio_array = audio_array / peak * 0.95
        sf.write(buf, audio_array.astype(np.float32), sample_rate, format="WAV", subtype="PCM_16")
        buf.seek(0)
        return buf.read()

    @staticmethod
    def _generate_silence_wav(duration_s: float = 1.0, sample_rate: int = 22050) -> bytes:
        """Generate silent WAV as a fallback."""
        import numpy as np
        import soundfile as sf

        silence = np.zeros(int(duration_s * sample_rate), dtype=np.float32)
        buf = BytesIO()
        sf.write(buf, silence, sample_rate, format="WAV", subtype="PCM_16")
        buf.seek(0)
        return buf.read()

    async def cleanup(self) -> None:
        """Release model resources."""
        self._engine = None
        self._ready = False
        logger.info("TTS service cleaned up")
