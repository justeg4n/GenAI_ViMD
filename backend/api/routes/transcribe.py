"""
Transcription Route — POST /api/v1/transcribe
==============================================
ASR pipeline: VAD → Denoising → Whisper transcription
Supports both audio_id (from /upload-audio) and raw file upload.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from pydantic import BaseModel, Field

from backend.api.auth_adapter import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter()
STORAGE_AUDIO = Path(os.getenv("STORAGE_BASE", "storage")) / "audio"


class TranscribeRequest(BaseModel):
    audio_id: Optional[str] = Field(None, description="Audio ID from /upload-audio")
    language: str = Field("vi", description="Language code: vi, ede, bah")
    task: str = Field("transcribe", description="transcribe or translate")
    use_vad: bool = Field(True, description="Apply Voice Activity Detection")
    use_denoise: bool = Field(True, description="Apply audio denoising")
    province_hint: Optional[str] = Field(None, description="Province hint for dialect-aware ASR")


class TranscribeResponse(BaseModel):
    audio_id: Optional[str]
    text: str
    language: str
    segments: list[dict]
    confidence: float
    latency_ms: float
    model_used: str
    wer: Optional[float] = None
    cer: Optional[float] = None


@router.post(
    "/transcribe",
    response_model=TranscribeResponse,
    summary="Transcribe audio to text",
    description="Run ASR pipeline (VAD → Denoise → Whisper) on uploaded audio.",
)
async def transcribe(
    request: Request,
    body: TranscribeRequest,
    user: Optional[dict] = Depends(get_current_user),
):
    """Transcribe audio referenced by audio_id."""
    if not body.audio_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="audio_id is required. Upload audio first via POST /api/v1/upload-audio",
        )

    audio_path = STORAGE_AUDIO / f"{body.audio_id}.wav"
    if not audio_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Audio ID '{body.audio_id}' not found",
        )

    # Load audio
    try:
        import soundfile as sf
        audio, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load audio: {e}")

    # Get model registry from app state
    registry = getattr(request.app.state, "model_registry", None)
    if registry is None:
        raise HTTPException(status_code=503, detail="Model registry not available")

    # Run ASR pipeline
    try:
        asr_service = registry.get_asr_service()
        result = await asr_service.transcribe(
            audio=audio,
            sample_rate=sr,
            language=body.language,
            task=body.task,
            use_vad=body.use_vad,
            use_denoise=body.use_denoise,
        )
    except Exception as e:
        logger.error(f"Transcription error for {body.audio_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")

    logger.info(
        f"Transcribed audio_id={body.audio_id} "
        f"language={body.language} "
        f"latency={result.latency_ms:.0f}ms"
    )

    return TranscribeResponse(
        audio_id=body.audio_id,
        text=result.text,
        language=result.language,
        segments=result.segments,
        confidence=result.confidence,
        latency_ms=result.latency_ms,
        model_used=result.model_used,
    )


@router.post(
    "/transcribe/file",
    response_model=TranscribeResponse,
    summary="Transcribe audio file directly",
    description="Upload and transcribe in one step.",
)
async def transcribe_file(
    request: Request,
    file: Annotated[UploadFile, File(description="Audio file")],
    language: str = Form("vi"),
    use_vad: bool = Form(True),
    use_denoise: bool = Form(True),
    user: Optional[dict] = Depends(get_current_user),
):
    """Upload and transcribe audio in a single request."""
    import io
    import numpy as np
    import soundfile as sf

    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        audio, sr = sf.read(io.BytesIO(contents), dtype="float32", always_2d=False)
    except Exception:
        try:
            import librosa
            audio, sr = librosa.load(io.BytesIO(contents), sr=16000, mono=True)
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Cannot decode audio: {e}")

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    registry = getattr(request.app.state, "model_registry", None)
    if registry is None:
        raise HTTPException(status_code=503, detail="Model registry not available")

    try:
        asr_service = registry.get_asr_service()
        result = await asr_service.transcribe(audio=audio, sample_rate=sr, language=language,
                                              use_vad=use_vad, use_denoise=use_denoise)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")

    return TranscribeResponse(
        audio_id=None,
        text=result.text,
        language=language,
        segments=result.segments,
        confidence=result.confidence,
        latency_ms=result.latency_ms,
        model_used=result.model_used,
    )
