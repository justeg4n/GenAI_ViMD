"""
Voice Clone Route — POST /api/v1/voice-clone
=============================================
Clone a speaker's voice from a 5–30 second reference audio.
Preserves dialect, prosody, and regional intonation.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import Response
from pydantic import BaseModel, Field

from backend.api.auth_adapter import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter()
STORAGE_AUDIO = Path(os.getenv("STORAGE_BASE", "storage")) / "audio"


class VoiceCloneRequest(BaseModel):
    reference_audio_id: str = Field(..., description="Audio ID of the reference speaker (5–30s)")
    text: str = Field(..., min_length=1, max_length=5000, description="Text to synthesize")
    language: str = Field("vi", description="Target language")
    preserve_dialect: bool = Field(True, description="Preserve regional dialect characteristics")
    preserve_prosody: bool = Field(True, description="Preserve prosody patterns")
    speed: float = Field(1.0, ge=0.5, le=2.0)


class VoiceCloneResponse(BaseModel):
    speaker_embedding_id: str
    voice_similarity_score: float
    dialect_preserved: bool
    latency_ms: float


@router.post(
    "/voice-clone",
    summary="Clone voice from reference audio",
    description="Extract speaker embedding from 5–30s reference, synthesize target text. Returns WAV audio.",
    responses={200: {"content": {"audio/wav": {}}, "description": "Cloned voice WAV audio"}},
)
async def voice_clone(
    request: Request,
    body: VoiceCloneRequest,
    user: Optional[dict] = Depends(get_current_user),
):
    ref_path = STORAGE_AUDIO / f"{body.reference_audio_id}.wav"
    if not ref_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Reference audio ID '{body.reference_audio_id}' not found",
        )

    # Validate reference duration
    try:
        import soundfile as sf
        info = sf.info(str(ref_path))
        duration_s = info.duration
        if duration_s < 5.0:
            raise HTTPException(
                status_code=422,
                detail=f"Reference audio too short ({duration_s:.1f}s). Minimum 5 seconds required.",
            )
        if duration_s > 30.0:
            logger.warning(f"Reference audio {body.reference_audio_id} is {duration_s:.1f}s (max 30s recommended)")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cannot read reference audio: {e}")

    registry = getattr(request.app.state, "model_registry", None)
    if registry is None:
        raise HTTPException(status_code=503, detail="Model registry not available")

    try:
        clone_service = registry.get_clone_service()
        audio_bytes, metadata = await clone_service.clone_and_synthesize(
            reference_audio_path=str(ref_path),
            text=body.text,
            language=body.language,
            preserve_dialect=body.preserve_dialect,
            preserve_prosody=body.preserve_prosody,
            speed=body.speed,
        )
    except Exception as e:
        logger.error(f"Voice cloning error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Voice cloning failed: {e}")

    return Response(
        content=audio_bytes,
        media_type="audio/wav",
        headers={
            "Content-Disposition": 'attachment; filename="cloned_voice.wav"',
            "X-Speaker-Similarity": str(round(metadata.get("similarity", 0.0), 4)),
            "X-Latency-Ms": str(round(metadata.get("latency_ms", 0.0), 2)),
        },
    )
