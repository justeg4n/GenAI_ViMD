"""
Translation Service — Vietnamese ↔ Ede/Bahnar Machine Translation
=================================================================
Uses mBART-50 fine-tuned on Ede/Bahnar bilingual corpora.
Supports: vi→ede, ede→vi, vi→bah, bah→vi
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

SUPPORTED_DIRECTIONS = {
    ("vi", "ede"), ("ede", "vi"),
    ("vi", "bah"), ("bah", "vi"),
    ("vi", "en"), ("en", "vi"),   # via Whisper/mBART
}


@dataclass
class TranslationResult:
    source_text: str
    translated_text: str
    source_lang: str
    target_lang: str
    confidence: float = 0.0
    latency_ms: float = 0.0
    model_used: str = "mbart-50"
    tokens_used: int = 0


class TranslationService:
    """
    Machine translation service for Vietnamese ↔ Ede/Bahnar.
    Backed by a fine-tuned mBART-50 model.
    """

    LANG_TO_MBART: dict[str, str] = {
        "vi": "vi_VN",
        "ede": "vi_VN",   # Ede uses Vietnamese script, treated as vi domain
        "bah": "vi_VN",   # Bahnar similarly
        "en": "en_XX",
    }

    def __init__(self, model_registry: Any = None):
        self.model_registry = model_registry
        self._translator = None
        self._ready = False

    async def initialize(self) -> None:
        """Lazy-load the mBART translator."""
        if self._ready:
            return
        try:
            from models.translation.mbart_translator import MBartTranslator
            self._translator = MBartTranslator()
            await asyncio.get_event_loop().run_in_executor(None, self._translator.load)
            logger.info("✓ mBART translator loaded")
        except Exception as e:
            logger.warning(f"mBART translator load failed, using stub: {e}")
            self._translator = None
        self._ready = True

    async def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        max_length: int = 512,
        num_beams: int = 5,
    ) -> TranslationResult:
        """
        Translate text between supported language pairs.

        Args:
            text: Source text to translate
            source_lang: Source language code ('vi', 'ede', 'bah', 'en')
            target_lang: Target language code ('vi', 'ede', 'bah', 'en')
            max_length: Maximum output length in tokens
            num_beams: Beam search width

        Returns:
            TranslationResult with translated text and metadata

        Raises:
            ValueError: If language pair is not supported
        """
        await self.initialize()
        t_start = time.perf_counter()

        if (source_lang, target_lang) not in SUPPORTED_DIRECTIONS:
            raise ValueError(
                f"Unsupported language pair: {source_lang}→{target_lang}. "
                f"Supported: {', '.join(f'{s}→{t}' for s, t in SUPPORTED_DIRECTIONS)}"
            )

        if not text or not text.strip():
            raise ValueError("Source text cannot be empty")

        # Cap input length
        text = text[:2000].strip()

        if self._translator is not None:
            try:
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self._translator.translate(
                        text=text,
                        source_lang=source_lang,
                        target_lang=target_lang,
                        max_length=max_length,
                        num_beams=num_beams,
                    ),
                )
                translated = result.get("text", "")
                confidence = result.get("confidence", 0.5)
                tokens_used = result.get("tokens_used", 0)
                model_used = "mbart-50-finetuned"
            except Exception as e:
                logger.error(f"Translation failed: {e}", exc_info=True)
                translated = f"[Translation error: {e}]"
                confidence = 0.0
                tokens_used = 0
                model_used = "fallback"
        else:
            translated = f"[STUB] Translation of: {text[:50]}..."
            confidence = 0.0
            tokens_used = 0
            model_used = "stub"

        latency_ms = (time.perf_counter() - t_start) * 1000
        logger.info(
            f"Translation {source_lang}→{target_lang}: {len(text)} chars → "
            f"{len(translated)} chars, latency={latency_ms:.0f}ms"
        )

        return TranslationResult(
            source_text=text,
            translated_text=translated,
            source_lang=source_lang,
            target_lang=target_lang,
            confidence=confidence,
            latency_ms=latency_ms,
            model_used=model_used,
            tokens_used=tokens_used,
        )

    async def cleanup(self) -> None:
        """Release model resources."""
        self._translator = None
        self._ready = False
        logger.info("Translation service cleaned up")
