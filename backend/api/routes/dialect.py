"""
Dialect Detection Route — POST /api/v1/detect-accent
=====================================================
Identifies provincial dialect, region, and ethnic language probabilities
from uploaded audio using WavLM-Large classifier.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from backend.api.auth_adapter import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter()
STORAGE_AUDIO = Path(os.getenv("STORAGE_BASE", "storage")) / "audio"


class DetectAccentRequest(BaseModel):
    audio_id: str = Field(..., description="Audio ID from /upload-audio")
    top_k: int = Field(3, ge=1, le=10, description="Number of top province predictions")


class DetectAccentResponse(BaseModel):
    audio_id: str
    province: str
    region: str
    confidence: float
    top_k_provinces: list[dict]
    ethnic_probability: dict[str, float]
    code_switching_score: float
    latency_ms: float


@router.post(
    "/detect-accent",
    response_model=DetectAccentResponse,
    summary="Detect dialect and accent",
    description="Identify province-level dialect, regional accent, and ethnic language influences.",
)
async def detect_accent(
    request: Request,
    body: DetectAccentRequest,
    user: Optional[dict] = Depends(get_current_user),
):
    audio_path = STORAGE_AUDIO / f"{body.audio_id}.wav"
    if not audio_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Audio ID '{body.audio_id}' not found",
        )

    try:
        import soundfile as sf
        audio, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load audio: {e}")

    registry = getattr(request.app.state, "model_registry", None)
    if registry is None:
        raise HTTPException(status_code=503, detail="Model registry not available")

    try:
        dialect_service = registry.get_dialect_service()
        result = await dialect_service.detect(audio=audio, sample_rate=sr, top_k=body.top_k)
    except Exception as e:
        logger.error(f"Dialect detection error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Dialect detection failed: {e}")

    return DetectAccentResponse(
        audio_id=body.audio_id,
        province=result["province"],
        region=result["region"],
        confidence=result["confidence"],
        top_k_provinces=[
            {"province": p, "probability": prob}
            for p, prob in result["top_k_provinces"]
        ],
        ethnic_probability=result["ethnic_probability"],
        code_switching_score=result["code_switching_score"],
        latency_ms=result.get("latency_ms", 0.0),
    )
