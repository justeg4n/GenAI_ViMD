"""
Model Registry
==============
Lazy-loading singleton registry for all AI models.
Ensures models are loaded once and shared across requests.
Uses async initialization to avoid blocking startup.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Centralized model registry with lazy initialization.
    Models are loaded on first access to minimize startup time.
    """

    def __init__(self):
        self._asr_service = None
        self._dialect_service = None
        self._tts_service = None
        self._clone_service = None
        self._translation_service = None
        self._initialized = False
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Perform initial setup (does not load heavy models)."""
        self._initialized = True
        logger.info("ModelRegistry initialized (lazy loading enabled)")

    async def cleanup(self) -> None:
        """Release model resources on shutdown."""
        logger.info("ModelRegistry cleaning up...")
        # Allow GC to handle model cleanup

    def get_model_status(self) -> dict:
        """Return current model load status."""
        return {
            "asr": "loaded" if self._asr_service else "not_loaded",
            "dialect": "loaded" if self._dialect_service else "not_loaded",
            "tts": "loaded" if self._tts_service else "not_loaded",
            "voice_clone": "loaded" if self._clone_service else "not_loaded",
            "translation": "loaded" if self._translation_service else "not_loaded",
        }

    # ── ASR ───────────────────────────────────────────────────────────────────

    def get_asr_service(self):
        if self._asr_service is None:
            self._asr_service = _LazyASRService()
        return self._asr_service

    # ── Dialect ───────────────────────────────────────────────────────────────

    def get_dialect_service(self):
        if self._dialect_service is None:
            self._dialect_service = _LazyDialectService()
        return self._dialect_service

    # ── TTS ───────────────────────────────────────────────────────────────────

    def get_tts_service(self):
        if self._tts_service is None:
            self._tts_service = _LazyTTSService()
        return self._tts_service

    # ── Clone ─────────────────────────────────────────────────────────────────

    def get_clone_service(self):
        if self._clone_service is None:
            self._clone_service = _LazyCloneService()
        return self._clone_service

    # ── Translation ───────────────────────────────────────────────────────────

    def get_translation_service(self):
        if self._translation_service is None:
            self._translation_service = _LazyTranslationService()
        return self._translation_service


# ── Lazy Service Wrappers ─────────────────────────────────────────────────────

class _LazyASRService:
    """Lazy ASR service — imports and loads model on first use."""
    _engine = None

    async def transcribe(self, audio, sample_rate=16000, language="vi",
                         task="transcribe", use_vad=True, use_denoise=True):
        if self._engine is None:
            from models.asr.whisper_inference import WhisperASREngine
            from models.asr.vad import VAD
            self._engine = WhisperASREngine(device="auto")
            self._vad = VAD() if use_vad else None

        import asyncio
        loop = asyncio.get_event_loop()

        def _sync_transcribe():
            audio_to_process = audio
            if use_vad and self._vad:
                segments = self._vad.detect(audio, sample_rate)
                if segments:
                    import numpy as np
                    audio_to_process = np.concatenate([s.audio for s in segments])
            return self._engine.transcribe(audio_to_process, sample_rate, language=language, task=task)

        return await loop.run_in_executor(None, _sync_transcribe)


class _LazyDialectService:
    _classifier = None

    async def detect(self, audio, sample_rate=16000, top_k=3):
        if self._classifier is None:
            from models.dialect.dialect_classifier import DialectClassifierInference
            weights_path = Path(os.getenv("MODEL_CACHE_DIR", "storage/weights")) / "dialect"
            self._classifier = DialectClassifierInference.from_pretrained(
                str(weights_path), device="auto"
            )

        import asyncio
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self._classifier.predict(audio, sample_rate, top_k=top_k)
        )
        return result


class _LazyTTSService:
    _engine = None

    async def synthesize(self, text, language="vi", province=None, region=None,
                         speed=1.0, emotion="neutral", speaker_id=None):
        if self._engine is None:
            from models.tts.xtts_engine import XTTSEngine
            self._engine = XTTSEngine(device="auto")

        import asyncio
        loop = asyncio.get_event_loop()
        audio_bytes = await loop.run_in_executor(
            None,
            lambda: self._engine.synthesize(
                text=text, language=language, province=province,
                region=region, speed=speed, emotion=emotion,
                speaker_id=speaker_id,
            )
        )
        return audio_bytes


class _LazyCloneService:
    _cloner = None

    async def clone_and_synthesize(self, reference_audio_path, text,
                                    language="vi", preserve_dialect=True,
                                    preserve_prosody=True, speed=1.0):
        if self._cloner is None:
            from models.voice_clone.cloner import VoiceCloner
            self._cloner = VoiceCloner(device="auto")

        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._cloner.clone_and_synthesize(
                reference_path=reference_audio_path,
                text=text, language=language,
                preserve_dialect=preserve_dialect,
                preserve_prosody=preserve_prosody,
                speed=speed,
            )
        )


class _LazyTranslationService:
    _translator = None

    async def translate(self, text, source_lang, target_lang, use_backtranslation=False):
        if self._translator is None:
            from models.translation.mbart_translator import MBartTranslator
            self._translator = MBartTranslator(device="auto")

        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._translator.translate(
                text=text,
                source_lang=source_lang,
                target_lang=target_lang,
                use_backtranslation=use_backtranslation,
            )
        )
