"""
Translation Route — POST /api/v1/translate
==========================================
Vietnamese ↔ Ede and Vietnamese ↔ Bahnar translation using mBART.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from backend.api.auth_adapter import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter()

VALID_LANG_PAIRS = {
    ("vi", "ede"), ("ede", "vi"),
    ("vi", "bah"), ("bah", "vi"),
}


class TranslateRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)
    source_lang: str = Field(..., description="Source language: vi, ede, bah")
    target_lang: str = Field(..., description="Target language: vi, ede, bah")
    use_backtranslation: bool = Field(False, description="Use backtranslation for confidence scoring")

    @field_validator("source_lang", "target_lang")
    @classmethod
    def validate_lang(cls, v):
        if v not in {"vi", "ede", "bah"}:
            raise ValueError("Language must be vi, ede, or bah")
        return v


class TranslateResponse(BaseModel):
    source_text: str
    translated_text: str
    source_lang: str
    target_lang: str
    confidence: float
    backtranslated_text: Optional[str] = None
    latency_ms: float


@router.post(
    "/translate",
    response_model=TranslateResponse,
    summary="Translate text",
    description="Translate between Vietnamese and ethnic languages (Ede, Bahnar) using mBART.",
)
async def translate(
    request: Request,
    body: TranslateRequest,
    user: Optional[dict] = Depends(get_current_user),
):
    lang_pair = (body.source_lang, body.target_lang)
    if lang_pair not in VALID_LANG_PAIRS:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported language pair: {body.source_lang}→{body.target_lang}. "
                   f"Supported: vi↔ede, vi↔bah",
        )

    if body.source_lang == body.target_lang:
        return TranslateResponse(
            source_text=body.text,
            translated_text=body.text,
            source_lang=body.source_lang,
            target_lang=body.target_lang,
            confidence=1.0,
            latency_ms=0.0,
        )

    registry = getattr(request.app.state, "model_registry", None)
    if registry is None:
        raise HTTPException(status_code=503, detail="Model registry not available")

    try:
        translation_service = registry.get_translation_service()
        result = await translation_service.translate(
            text=body.text,
            source_lang=body.source_lang,
            target_lang=body.target_lang,
            use_backtranslation=body.use_backtranslation,
        )
    except Exception as e:
        logger.error(f"Translation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Translation failed: {e}")

    return TranslateResponse(
        source_text=body.text,
        translated_text=result["translated_text"],
        source_lang=body.source_lang,
        target_lang=body.target_lang,
        confidence=result.get("confidence", 0.9),
        backtranslated_text=result.get("backtranslated_text"),
        latency_ms=result.get("latency_ms", 0.0),
    )
