"""
Celery Tasks — Async Background Task Definitions
================================================
Defines Celery tasks for:
  - Batch audio transcription
  - Async TTS generation
  - Voice cloning
  - Translation jobs
  - Model warming
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from backend.workers.celery_app import celery_app

logger = logging.getLogger(__name__)

STORAGE_BASE = Path(os.getenv("STORAGE_BASE", "storage"))
STORAGE_AUDIO = STORAGE_BASE / "audio"
STORAGE_EXPORTS = STORAGE_BASE / "exports"


# ── ASR Tasks ─────────────────────────────────────────────────────────────────

@celery_app.task(
    name="tasks.transcribe_audio",
    bind=True,
    max_retries=3,
    default_retry_delay=5,
    acks_late=True,
    reject_on_worker_lost=True,
)
def transcribe_audio(
    self,
    audio_id: str,
    language: str = "vi",
    use_vad: bool = True,
    use_denoise: bool = True,
    province_hint: str | None = None,
) -> dict:
    """
    Async task: transcribe a stored audio file.

    Args:
        audio_id: UUID referencing the file in storage/audio/
        language: Target transcription language
        use_vad: Apply Voice Activity Detection
        use_denoise: Apply audio denoising
        province_hint: Optional province code for dialect-aware prompting

    Returns:
        dict with keys: text, segments, confidence, latency_ms, model_used
    """
    import asyncio
    import soundfile as sf

    audio_path = STORAGE_AUDIO / f"{audio_id}.wav"
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_id}")

    try:
        audio, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
    except Exception as exc:
        raise self.retry(exc=exc)

    try:
        from backend.services.asr_service import ASRService
        svc = ASRService()
        result = asyncio.run(
            svc.transcribe(
                audio=audio,
                sample_rate=sr,
                language=language,
                use_vad=use_vad,
                use_denoise=use_denoise,
                province_hint=province_hint,
            )
        )
        return {
            "audio_id": audio_id,
            "text": result.text,
            "segments": result.segments,
            "confidence": result.confidence,
            "latency_ms": result.latency_ms,
            "model_used": result.model_used,
            "language": result.language,
        }
    except Exception as exc:
        logger.error(f"Transcription task failed for {audio_id}: {exc}", exc_info=True)
        raise self.retry(exc=exc)


# ── TTS Tasks ─────────────────────────────────────────────────────────────────

@celery_app.task(
    name="tasks.generate_speech",
    bind=True,
    max_retries=2,
    acks_late=True,
)
def generate_speech(
    self,
    text: str,
    language: str = "vi",
    speaker_id: str | None = None,
    province_code: str | None = None,
    speed: float = 1.0,
    output_filename: str | None = None,
) -> dict:
    """
    Async task: synthesize speech and save to storage/exports/.

    Returns:
        dict with keys: output_path, duration_s, latency_ms, model_used
    """
    import asyncio

    try:
        from backend.services.tts_service import TTSService
        svc = TTSService()
        result = asyncio.run(
            svc.synthesize(
                text=text,
                language=language,
                speaker_id=speaker_id,
                province_code=province_code,
                speed=speed,
            )
        )

        STORAGE_EXPORTS.mkdir(parents=True, exist_ok=True)
        fname = output_filename or f"tts_{id(result)}.wav"
        out_path = STORAGE_EXPORTS / fname
        out_path.write_bytes(result.audio_bytes)

        return {
            "output_path": str(out_path),
            "duration_s": result.duration_s,
            "latency_ms": result.latency_ms,
            "model_used": result.model_used,
        }
    except Exception as exc:
        logger.error(f"TTS task failed: {exc}", exc_info=True)
        raise self.retry(exc=exc)


# ── Translation Tasks ──────────────────────────────────────────────────────────

@celery_app.task(
    name="tasks.translate_text",
    bind=True,
    max_retries=3,
    acks_late=True,
)
def translate_text(
    self,
    text: str,
    source_lang: str,
    target_lang: str,
) -> dict:
    """
    Async task: translate text between supported language pairs.

    Returns:
        dict with keys: translated_text, confidence, latency_ms
    """
    import asyncio

    try:
        from backend.services.translation_service import TranslationService
        svc = TranslationService()
        result = asyncio.run(
            svc.translate(
                text=text,
                source_lang=source_lang,
                target_lang=target_lang,
            )
        )
        return {
            "source_text": result.source_text,
            "translated_text": result.translated_text,
            "source_lang": result.source_lang,
            "target_lang": result.target_lang,
            "confidence": result.confidence,
            "latency_ms": result.latency_ms,
        }
    except Exception as exc:
        logger.error(f"Translation task failed: {exc}", exc_info=True)
        raise self.retry(exc=exc)


# ── Model Warmup Task ──────────────────────────────────────────────────────────

@celery_app.task(name="tasks.warm_models", ignore_result=True)
def warm_models() -> None:
    """Pre-load all AI models to reduce first-request latency."""
    import asyncio

    logger.info("Warming up AI models...")
    try:
        from backend.services.asr_service import ASRService
        from backend.services.dialect_service import DialectService
        from backend.services.tts_service import TTSService
        from backend.services.translation_service import TranslationService

        async def _warm():
            await ASRService().initialize()
            await DialectService().initialize()
            await TTSService().initialize()
            await TranslationService().initialize()

        asyncio.run(_warm())
        logger.info("✓ All models warmed up")
    except Exception as e:
        logger.error(f"Model warmup failed: {e}", exc_info=True)
