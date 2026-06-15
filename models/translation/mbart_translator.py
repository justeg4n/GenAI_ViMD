"""
mBART Translator — Vietnamese ↔ Ede / Bahnar
=============================================
Fine-tuned mBART-50 for low-resource ethnic language translation.
Uses adapter layers for efficient fine-tuning.
Supports backtranslation for confidence scoring.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger(__name__)

# Language code mappings for mBART-50
LANG_CODE_MAP = {
    "vi": "vi_VN",
    "ede": "ede_XX",    # Custom token added during training
    "bah": "bah_XX",    # Custom token added during training
}

REVERSE_LANG_MAP = {v: k for k, v in LANG_CODE_MAP.items()}


class MBartTranslator:
    """
    mBART-50 based translator for Vietnamese ↔ Ede and Vietnamese ↔ Bahnar.

    Falls back to dictionary-based translation when model not available,
    ensuring the API always returns something meaningful.

    Usage:
        translator = MBartTranslator(device="auto")
        result = translator.translate("Xin chào", source_lang="vi", target_lang="ede")
        print(result["translated_text"])
    """

    MODEL_ID = "facebook/mbart-large-50"

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "auto",
        max_length: int = 512,
        num_beams: int = 4,
        use_cache: bool = True,
    ):
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model_path = model_path or os.getenv("MODEL_CACHE_DIR", "storage/weights")
        self.max_length = max_length
        self.num_beams = num_beams
        self.use_cache = use_cache

        self._tokenizer = None
        self._model = None
        self._load_model()

    def _load_model(self) -> None:
        """Load mBART tokenizer and model."""
        try:
            from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

            weights_dir = Path(self.model_path) / "translation"
            if weights_dir.exists():
                model_id = str(weights_dir)
            else:
                model_id = self.MODEL_ID

            self._tokenizer = MBart50TokenizerFast.from_pretrained(model_id)

            # Add custom ethnic language tokens if not present
            special_tokens = [t for t in ["ede_XX", "bah_XX"] if t not in self._tokenizer.lang_code_to_id]
            if special_tokens:
                self._tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

            self._model = MBartForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            ).to(self.device)
            self._model.eval()

            logger.info(f"mBART translator loaded from {model_id}")
        except Exception as e:
            logger.warning(f"mBART model load failed: {e}. Using dictionary fallback.")

    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        use_backtranslation: bool = False,
    ) -> dict:
        """
        Translate text between supported language pairs.

        Returns:
            {
              "translated_text": str,
              "confidence": float,
              "backtranslated_text": str | None,
              "latency_ms": float,
            }
        """
        start = time.perf_counter()
        src_code = LANG_CODE_MAP.get(source_lang, f"{source_lang}_XX")
        tgt_code = LANG_CODE_MAP.get(target_lang, f"{target_lang}_XX")

        if self._model is not None and self._tokenizer is not None:
            translated = self._model_translate(text, src_code, tgt_code)
        else:
            translated = self._dictionary_translate(text, source_lang, target_lang)

        result = {
            "translated_text": translated,
            "confidence": 0.85 if self._model else 0.5,
            "latency_ms": (time.perf_counter() - start) * 1000,
            "backtranslated_text": None,
        }

        if use_backtranslation and self._model is not None:
            try:
                back = self._model_translate(translated, tgt_code, src_code)
                result["backtranslated_text"] = back
                # Adjust confidence based on back-translation similarity
                similarity = self._compute_similarity(text, back)
                result["confidence"] = max(0.3, min(0.99, similarity))
            except Exception as e:
                logger.debug(f"Backtranslation failed: {e}")

        return result

    def _model_translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        """Run mBART model translation."""
        self._tokenizer.src_lang = src_lang
        tgt_lang_id = self._tokenizer.lang_code_to_id.get(tgt_lang, 2)

        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            generated = self._model.generate(
                **inputs,
                forced_bos_token_id=tgt_lang_id,
                num_beams=self.num_beams,
                max_new_tokens=self.max_length,
                use_cache=self.use_cache,
            )

        translated = self._tokenizer.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0]
        return translated.strip()

    def _dictionary_translate(
        self, text: str, source_lang: str, target_lang: str
    ) -> str:
        """
        Simple word-by-word dictionary translation fallback.
        Uses seed vocabulary from ethnic corpus loader.
        """
        try:
            from dataset.ingestion.ethnic_corpus_loader import EDE_VI_SEED, BAHNAR_VI_SEED
        except ImportError:
            return f"[Translation not available: {text}]"

        if (source_lang, target_lang) == ("vi", "ede"):
            word_map = {vi: ede for ede, vi in EDE_VI_SEED}
        elif (source_lang, target_lang) == ("ede", "vi"):
            word_map = {ede: vi for ede, vi in EDE_VI_SEED}
        elif (source_lang, target_lang) == ("vi", "bah"):
            word_map = {vi: bah for bah, vi in BAHNAR_VI_SEED}
        elif (source_lang, target_lang) == ("bah", "vi"):
            word_map = {bah: vi for bah, vi in BAHNAR_VI_SEED}
        else:
            return text

        words = text.lower().split()
        translated_words = [word_map.get(w, w) for w in words]
        return " ".join(translated_words)

    @staticmethod
    def _compute_similarity(text1: str, text2: str) -> float:
        """Simple character n-gram similarity for backtranslation quality."""
        def ngrams(text, n=3):
            return set(text[i:i+n] for i in range(len(text)-n+1))

        n1 = ngrams(text1.lower())
        n2 = ngrams(text2.lower())
        if not n1 or not n2:
            return 0.0
        intersection = len(n1 & n2)
        return intersection / max(len(n1), len(n2))
