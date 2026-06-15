"""
TTS Route — POST /api/v1/generate-speech
=========================================
Text-to-Speech synthesis using XTTS-v2.
Supports dialect/province conditioning, speed, emotion, intonation controls.
Returns 24kHz WAV audio.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import Response
from pydantic import BaseModel, Field, field_validator

from backend.api.auth_adapter import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter()

VALID_EMOTIONS = ["neutral", "happy", "sad", "angry", "surprised", "fearful"]
VALID_LANGUAGES = ["vi", "ede", "bah"]


class GenerateSpeechRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="Text to synthesize")
    language: str = Field("vi", description="Language: vi, ede, bah")
    province: Optional[str] = Field(None, description="Province for dialect conditioning")
    region: Optional[str] = Field(None, description="Region: North, Central, South")
    speed: float = Field(1.0, ge=0.5, le=2.0, description="Speech speed multiplier")
    emotion: str = Field("neutral", description="Emotion: neutral, happy, sad, angry, surprised, fearful")
    speaker_id: Optional[str] = Field(None, description="Speaker ID for voice cloning reference")
    output_format: str = Field("wav", description="Output format: wav")
    sample_rate: int = Field(24000, description="Output sample rate")

    @field_validator("emotion")
    @classmethod
    def validate_emotion(cls, v):
        if v not in VALID_EMOTIONS:
            raise ValueError(f"emotion must be one of: {VALID_EMOTIONS}")
        return v

    @field_validator("language")
    @classmethod
    def validate_language(cls, v):
        if v not in VALID_LANGUAGES:
            raise ValueError(f"language must be one of: {VALID_LANGUAGES}")
        return v


@router.post(
    "/generate-speech",
    summary="Generate speech from text",
    description="TTS synthesis with dialect, speed, emotion, and intonation controls. Returns 24kHz WAV audio.",
    responses={
        200: {
            "content": {"audio/wav": {}},
            "description": "WAV audio file",
        }
    },
)
async def generate_speech(
    request: Request,
    body: GenerateSpeechRequest,
    user: Optional[dict] = Depends(get_current_user),
):
    registry = getattr(request.app.state, "model_registry", None)
    if registry is None:
        raise HTTPException(status_code=503, detail="Model registry not available")

    try:
        tts_service = registry.get_tts_service()
        audio_bytes = await tts_service.synthesize(
            text=body.text,
            language=body.language,
            province=body.province,
            region=body.region,
            speed=body.speed,
            emotion=body.emotion,
            speaker_id=body.speaker_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"TTS synthesis error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Speech synthesis failed: {e}")

    return Response(
        content=audio_bytes,
        media_type="audio/wav",
        headers={
            "Content-Disposition": 'attachment; filename="speech.wav"',
            "X-Sample-Rate": str(body.sample_rate),
            "X-Language": body.language,
        },
    )
